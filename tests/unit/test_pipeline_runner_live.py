"""실행기 실시간 소스 경로 시험: 결과 공개 전 입력 반영(지점 A·B·C), 세대 전이, 실시간 정책, 정리.

실제 `LiveLinkSource`에 가짜 수신 연결을 붙이고 검출 worker 안에서 `handle_*`로 입력 사건을 만든다.
ROS·YOLO는 쓰지 않고, 실행기는 데몬 스레드에서 유한 join 한다.
"""
import io
import threading
import time
import unittest
from unittest import mock

from eyetoarm.common.types import InputStatus
from eyetoarm.inputs.ros_image import LiveLinkSource, LiveSourceConfig
from eyetoarm.inputs.live_slot import LiveSlotConfig
from eyetoarm.vision import pipeline_runner as pr
from eyetoarm.vision.pipeline_records import JsonlRecorder, PipelineLog
from eyetoarm.vision.pipeline_runner import RunnerConfig, run
from test_pipeline import FakeClock, observation, settings
from test_ros_image_adapter import FakeExecutor, FakeTransport, SpinTransport, image_msg, info_msg, view_convert
from test_run_pipeline import lines

S = 1_000_000_000
WATCH = RunnerConfig(close_attempts=1, close_retry_interval_s=0.0)          # 감시 켬, 기본 10/1 그대로
SYNC = RunnerConfig(close_attempts=1, first_detect_timeout_s=None, detect_timeout_s=None)


class SourceClock:
    """소스(수신·제한시간)용 PC 단조 시계. 시험·검출 hook이 직접 움직인다."""

    def __init__(self):
        """100 s에서 시작."""
        self.now = 100 * S

    def __call__(self):
        """현재 값."""
        return self.now


class Live:
    """실시간 실행 한 번의 준비·실행·결과 확인 도구."""

    def __init__(self, *, feed=None, hooks=None, cfg=WATCH, source_cfg=None, transport=None, should_stop=None,
                 b_consumer=None):
        """feed(src, clock): 시작 직후 넣을 메시지. hooks[job_seq](live): 검출 중 동작."""
        self.clock = SourceClock()
        self.feed, self.hooks = feed, hooks or {}
        self.cfg, self.should_stop, self.b_consumer = cfg, should_stop, b_consumer
        self.transport = transport or FakeTransport()
        self.source_cfg = source_cfg or LiveSourceConfig(frame_wait_s=0.005)
        self.src = None
        self.pipeline = None
        self.detect_calls = []

    def factory(self, issuer, generation_provider):
        """실행기가 부르는 소스 팩토리."""
        self.src = LiveLinkSource(source_id="gz_link", issuer=issuer, generation_provider=generation_provider,
                                  config=self.source_cfg, transport_factory=lambda s: self.transport,
                                  convert=view_convert, clock_ns=self.clock)

        if self.feed is not None:
            self.feed(self.src, self.clock)

        return self.src

    def detector(self, job, token):
        """검출 hook을 부르고 무효 관측을 돌려준다."""
        self.detect_calls.append(job.job_seq)
        hook = self.hooks.get(job.job_seq)

        if hook is not None:
            hook(self)

        return observation(job), None

    def run(self, limit_s=10.0):
        """데몬 스레드에서 실행기를 돌리고 유한 join 한다. (결과, 기록 목록, stderr)."""
        stream, err, box = io.StringIO(), io.StringIO(), {}
        real = pr.VisionPipeline

        def capture(*a, **kw):
            """코어 인스턴스를 잡아 둔다(관측 상태 확인용)."""
            self.pipeline = real(*a, **kw)
            return self.pipeline

        def target():
            """실행기 한 번."""
            with mock.patch.object(pr, "VisionPipeline", side_effect=capture):
                box["res"] = run(self.factory, settings(source_id="gz_link", profile=None),
                                 PipelineLog(JsonlRecorder(stream)), run_id="run-live", detector=self.detector,
                                 detector_takes_token=True, config=self.cfg, stderr=err, clock_ns=FakeClock(),
                                 should_stop=self.should_stop, b_consumer=self.b_consumer)

        t = threading.Thread(target=target, daemon=True)
        t.start()
        t.join(limit_s)

        if t.is_alive():
            raise AssertionError(f"runner did not finish within {limit_s} s; stderr={err.getvalue()!r}")

        return box["res"], lines(stream), err.getvalue()


def stop_after_jobs(live, n):
    """작업 n개가 terminal로 끝나면 정지를 요청하는 should_stop."""
    def check():
        """코어 집계로 판단."""
        p = live.pipeline
        c = p.counters if p is not None else None
        done = 0 if c is None else c.completed + c.failed + c.aborted + c.generation_discarded
        return done >= n

    return check


def processes(records):
    """처리 기록 목록."""
    return [r for r in records if r["kind"] == "process"]


def index_of(records, pred):
    """조건을 만족하는 첫 기록 위치(없으면 -1)."""
    return next((i for i, r in enumerate(records) if pred(r)), -1)


class PublishTest(unittest.TestCase):
    """정상 공개와 지점 C."""

    def test_normal_frame_published(self):
        """새 영상 → 작업 완료(세대 0), 종료 0, source_stats 기록, 수신 중단 없음."""
        live = Live(feed=lambda s, c: s.handle_image(image_msg(1)))
        live.should_stop = stop_after_jobs(live, 1)
        res, recs, err = live.run()

        self.assertEqual(res.exit_code, 0, err)
        self.assertEqual([(p["status"], p["generation"], p["timestamp_base"]) for p in processes(recs)],
                         [("COMPLETED", 0, "SIM_TIME")])
        self.assertEqual(res.input_boundary_aborts, 0)
        self.assertEqual(recs[-1]["kind"], "source_stats")
        self.assertEqual(live.transport.closed, [live.source_cfg.close_limit_s])

    def test_c1_frame_timeout_during_detection_aborts_publication(self):
        """검출 중 수신 중단 → 공개 전에 FRAME_TIMEOUT을 반영하고 작업은 input_boundary 중단, 종료 0."""
        def hook(live):
            """수신이 2 s 끊긴 것으로 시계를 민다."""
            live.clock.now += 2 * S

        live = Live(feed=lambda s, c: s.handle_image(image_msg(1)), hooks={0: hook})
        live.should_stop = stop_after_jobs(live, 1)
        res, recs, err = live.run()

        self.assertEqual(res.exit_code, 0, err)
        self.assertEqual((res.input_boundary_aborts, res.input_errors, res.live_inputs),
                         (1, 0, {"frame_timeout": 1}))
        self.assertEqual([p["status"] for p in processes(recs)], ["ABORTED"])
        self.assertIsNone(live.pipeline.current_observation)
        live_at = index_of(recs, lambda r: r["kind"] == "live_input" and r["input_kind"] == "frame_timeout")
        proc_at = index_of(recs, lambda r: r["kind"] == "process")
        self.assertTrue(0 <= live_at < proc_at, recs)
        self.assertTrue(any(r["kind"] == "event" and r["cause"].startswith("input_boundary") for r in recs))

    def test_point_b_applies_during_long_detection(self):
        """지점 B: 검출이 길게 걸리는 동안에도 대기 회차마다 입력을 반영한다."""
        seen = {}

        def hook(live):
            """수신 중단을 만들고, 검출이 끝나기 전에 반영되는지 최대 1 s 지켜본다."""
            live.clock.now += 2 * S
            deadline = time.monotonic() + 1.0

            while time.monotonic() < deadline and live.src.slot.stats()["applied_epoch"] < 1:
                time.sleep(0.005)

            seen["applied_during_detection"] = live.src.slot.stats()["applied_epoch"]

        live = Live(feed=lambda s, c: s.handle_image(image_msg(1)), hooks={0: hook})
        live.should_stop = stop_after_jobs(live, 1)
        res, recs, err = live.run()

        self.assertEqual(seen["applied_during_detection"], 1)
        self.assertEqual(res.input_boundary_aborts, 1, err)

    def test_c1_sync_path_same_rule(self):
        """감시 끔 동기 경로도 같은 규칙으로 input_boundary 중단."""
        live = Live(feed=lambda s, c: s.handle_image(image_msg(1)), cfg=SYNC,
                    hooks={0: lambda lv: setattr(lv.clock, "now", lv.clock.now + 2 * S)})
        live.should_stop = stop_after_jobs(live, 1)
        res, recs, err = live.run()

        self.assertEqual((res.exit_code, res.input_boundary_aborts), (0, 1), err)

    def test_c2_rewind_during_detection_discards_by_generation(self):
        """검출 중 되감기 → GENERATION_CHANGED가 처리 기록보다 먼저, 결과는 세대 폐기."""
        def feed(src, clock):
            """clock 10 s, 영상 9.9 s."""
            src.handle_clock(SimpleClock(10))
            src.handle_image(image_msg(9, 900_000_000))

        def hook(live):
            """되감기를 넣고 대기 회차가 반영할 시간을 준다."""
            live.src.handle_clock(SimpleClock(1))
            time.sleep(0.1)

        live = Live(feed=feed, hooks={0: hook})
        live.should_stop = stop_after_jobs(live, 1)
        res, recs, err = live.run()

        self.assertEqual(res.exit_code, 0, err)
        self.assertEqual([p["status"] for p in processes(recs)], ["DISCARDED_GENERATION"])
        ev_at = index_of(recs, lambda r: r["kind"] == "event" and r["cause"] == "sim_time_rewind")
        self.assertTrue(0 <= ev_at < index_of(recs, lambda r: r["kind"] == "process"), recs)
        self.assertEqual(res.input_boundary_aborts, 0)

    def test_c3_reconnect_during_detection(self):
        """검출 중 발행자 GID 집합 교체({A}→{B}) → cause reconnect, 결과는 세대 폐기."""
        def feed(src, clock):
            """발행자 {A}, 영상 1개."""
            src.handle_publishers([b"A"], ["reliable"])
            src.handle_image(image_msg(1))

        def hook(live):
            """끊김 없이 발행자 집합이 {B}로 바뀐다."""
            live.src.handle_publishers([b"B"], ["reliable"])

        live = Live(feed=feed, hooks={0: hook})
        live.should_stop = stop_after_jobs(live, 1)
        res, recs, err = live.run()

        self.assertEqual([p["status"] for p in processes(recs)], ["DISCARDED_GENERATION"])
        self.assertTrue(any(r["kind"] == "event" and r["cause"] == "reconnect" for r in recs), recs)
        self.assertEqual(res.live_inputs, {"reconnect": 1})

    def test_c4_boundary_and_result_arrive_together(self):
        """결과 반환 직전 경계가 같은 회차에 함께 있으면 결과 처리 전에 경계를 반영한다."""
        def feed(src, clock):
            """clock 10 s, 영상."""
            src.handle_clock(SimpleClock(10))
            src.handle_image(image_msg(9))

        live = Live(feed=feed, hooks={0: lambda lv: lv.src.handle_clock(SimpleClock(1))})
        live.should_stop = stop_after_jobs(live, 1)
        res, recs, err = live.run()

        live_at = index_of(recs, lambda r: r["kind"] == "live_input")
        self.assertTrue(0 <= live_at < index_of(recs, lambda r: r["kind"] == "process"), recs)
        self.assertEqual([p["status"] for p in processes(recs)], ["DISCARDED_GENERATION"])


class ResumePolicyRunnerTest(unittest.TestCase):
    """중단 뒤 첫 영상은 새 세대에서만 처리한다(발행자 조회 순서·일시 정지 재개)."""

    def test_123_order_no_old_generation_completed(self):
        """수신 중단 → 새 영상 → 뒤늦은 발행자 조회: 이전 세대 COMPLETED 0, 재접속 1회."""
        def hook(live):
            """검출 중 끊김(2 s) → 새 영상 → 뒤늦은 그래프 {B}."""
            live.clock.now += 2 * S
            live.src.handle_image(image_msg(5))
            live.src.handle_publishers([b"B"], ["reliable"])

        def feed(src, clock):
            """발행자 {A}, 영상 1개."""
            src.handle_publishers([b"A"], ["reliable"])
            src.handle_image(image_msg(1))

        live = Live(feed=feed, hooks={0: hook})
        live.should_stop = stop_after_jobs(live, 2)
        res, recs, err = live.run()

        self.assertEqual(res.exit_code, 0, err)
        self.assertEqual([(p["job_seq"], p["status"], p["generation"], p["timestamp_ms"]) for p in processes(recs)],
                         [(0, "DISCARDED_GENERATION", 0, 1000), (1, "COMPLETED", 1, 5000)])
        self.assertEqual(res.live_inputs, {"frame_timeout": 1, "reconnect": 1})
        self.assertEqual(sum(1 for r in recs if r["kind"] == "event" and r["cause"] == "reconnect"), 1)

    def test_pause_resume_new_generation(self):
        """일시 정지 재개(발행자 동일)도 새 세대에서 처리. 중단 관측은 이전 관측을 비운다."""
        live = Live(feed=lambda s, c: s.handle_image(image_msg(1)))
        seen = []

        def should_stop():
            """작업 0 완료 뒤 끊김을 만들고, 반영 뒤 새 영상을 넣는다. 작업 1이 끝나면 멈춘다."""
            p = live.pipeline
            done = 0 if p is None else p.counters.completed

            if done == 1 and not seen:
                live.clock.now += 2 * S
                seen.append("gap")
            elif seen == ["gap"] and p.current_observation is None and live.src.slot.stats()["state"] == "timed_out":
                seen.append("cleared")
                live.src.handle_image(image_msg(2))

            return done >= 2

        live.should_stop = should_stop
        res, recs, err = live.run()

        self.assertEqual(res.exit_code, 0, err)
        self.assertEqual(seen, ["gap", "cleared"])
        self.assertEqual([(p["status"], p["generation"]) for p in processes(recs)], [("COMPLETED", 0), ("COMPLETED", 1)])
        self.assertEqual(res.live_inputs, {"frame_timeout": 1, "reconnect": 1})


class SimpleClock:
    """가짜 /clock 메시지(초 단위)."""

    def __init__(self, sec):
        """clock.sec·nanosec."""
        self.clock = type("T", (), {"sec": sec, "nanosec": 0})()


class LateAndAfterTest(unittest.TestCase):
    """늦은 결과·판정 뒤 경계·이전 관측 무효."""

    def test_c5_timeout_late_result_and_boundary(self):
        """시간 초과 뒤 유예 안 늦은 결과 + 되감기: 추가 terminal 없음, 다음 작업은 새 세대, 종료 1."""
        cfg = RunnerConfig(close_attempts=1, first_detect_timeout_s=0.05, detect_timeout_s=0.05, cancel_grace_s=1.0,
                           stop_check_interval_s=0.01)

        def feed(src, clock):
            """clock 10 s, 영상."""
            src.handle_clock(SimpleClock(10))
            src.handle_image(image_msg(9))

        def hook(live):
            """기한을 넘긴 뒤 되감기와 새 영상을 넣고 반환한다."""
            time.sleep(0.15)
            live.src.handle_clock(SimpleClock(1))
            live.src.handle_image(image_msg(1))

        live = Live(feed=feed, hooks={0: hook}, cfg=cfg)
        live.should_stop = stop_after_jobs(live, 2)
        res, recs, err = live.run()

        self.assertEqual((res.exit_code, res.timeouts, res.late_results), (1, 1, 1), err)
        procs = processes(recs)
        self.assertEqual([(p["job_seq"], p["status"], p["generation"]) for p in procs],
                         [(0, "ABORTED", 0), (1, "COMPLETED", 1)])

    def test_c6_boundary_after_decision_invalidates_next(self):
        """공개 판정 직후 경계는 그 결과를 되돌리지 않고, 다음 반영 지점에서 현재 관측을 비운다."""
        live = Live(feed=lambda s, c: s.handle_image(image_msg(1)))
        seen = []

        def should_stop():
            """공개 뒤 관측을 보고 시계를 밀어 수신 중단을 만든다. 관측이 비면 멈춘다."""
            obs = live.pipeline.current_observation if live.pipeline is not None else None
            seen.append(obs is not None)

            if obs is not None:
                live.clock.now += 2 * S

            return len(seen) > 2 and obs is None

        live.should_stop = should_stop
        res, recs, err = live.run()

        self.assertEqual(res.exit_code, 0, err)
        self.assertEqual([p["status"] for p in processes(recs)], ["COMPLETED"])
        self.assertIn(True, seen)
        self.assertFalse(seen[-1])
        self.assertLess(index_of(recs, lambda r: r["kind"] == "process"),
                        index_of(recs, lambda r: r["kind"] == "live_input"))

    def test_rewind_before_read_new_frame_new_generation(self):
        """경계 직후 첫 영상은 STALE_CAPTURE로 버려지지 않고 새 세대로 처리된다."""
        def feed(src, clock):
            """clock 되감기 뒤 새 영상."""
            src.handle_clock(SimpleClock(10))
            src.handle_clock(SimpleClock(1))
            src.handle_image(image_msg(1))

        live = Live(feed=feed)
        live.should_stop = stop_after_jobs(live, 1)
        res, recs, err = live.run()

        self.assertEqual([(p["status"], p["capture_generation"], p["generation"]) for p in processes(recs)],
                         [("COMPLETED", 1, 1)])
        self.assertFalse([r for r in recs if r["kind"] == "queue_drop"])


class PolicyTest(unittest.TestCase):
    """실시간 정책·실패·정리."""

    def test_c7_stop_while_waiting_inputs(self):
        """영상 없이 수신 중단이 생겨도 정지 요청으로 0. FRAME_TIMEOUT은 입력 오류가 아니다."""
        calls = []

        def feed(src, clock):
            """영상 하나만 넣는다(작업 0 뒤 끊김)."""
            src.handle_image(image_msg(1))

        live = Live(feed=feed)

        def should_stop():
            """매번 시계를 밀고 몇 번 뒤 멈춘다."""
            calls.append(1)
            live.clock.now += S // 2
            return len(calls) > 8

        live.should_stop = should_stop
        res, recs, err = live.run()

        self.assertEqual((res.exit_code, res.stop_reason, res.input_errors), (0, "stop_requested", 0), err)
        self.assertEqual(res.live_inputs.get("frame_timeout"), 1)

    def test_transition_storm_stops_run(self):
        """세대를 올린 경계가 1 s 안에 10개를 넘으면 input_transition_storm으로 종료 1."""
        def feed(src, clock):
            """수신 중 발행자 집합 교체 12번."""
            src.handle_publishers([b"A"], ["reliable"])
            src.handle_image(image_msg(1))

            for k in range(12):
                src.handle_publishers([b"B" if k % 2 == 0 else b"A"], ["reliable"])

        res, recs, err = Live(feed=feed).run()

        self.assertEqual((res.exit_code, res.stop_reason), (1, "input_transition_storm"), err)

    def test_input_overflow_stops_run(self):
        """입력 대기열 상한을 넘으면 input_overflow로 종료 1."""
        def feed(src, clock):
            """프로필 변경 3번(상한 2)."""
            for fx in (100.0, 101.0, 102.0):
                src.handle_camera_info(info_msg(fx=fx))

        cfg = LiveSourceConfig(frame_wait_s=0.005, slot=LiveSlotConfig(max_pending_inputs=2))
        res, recs, err = Live(feed=feed, source_cfg=cfg).run()

        self.assertEqual((res.exit_code, res.stop_reason), (1, "input_overflow"), err)

    def test_b_consumer_not_allowed(self):
        """실시간 표시 단계는 B 소비 지점을 허용하지 않는다. 검출 없이 종료 1."""
        live = Live(feed=lambda s, c: s.handle_image(image_msg(1)), b_consumer=lambda f, o: None)
        res, recs, err = live.run()

        self.assertEqual((res.exit_code, res.stop_reason, live.detect_calls), (1, "b_consumer_not_allowed", []))

    def test_start_failure_ends_with_no_input(self):
        """연결 시작 실패 → SOURCE_UNAVAILABLE 반영(입력 오류 1) → read closed → no_input 종료 1."""
        live = Live(transport=FakeTransport(start_error=RuntimeError("rclpy init failed")))
        res, recs, err = live.run()

        self.assertEqual((res.exit_code, res.stop_reason, res.input_errors), (1, "no_input", 1), err)
        self.assertTrue(any(r["kind"] == "live_input" and r["input_kind"] == "start_failed" and "start failed" in r["detail"] for r in recs))

    def test_receiver_failure_before_first_frame(self):
        """첫 영상 전 수신기 예외 → receiver_failed 종료 1, 입력 오류 1, 원인 기록, 소스 정리 1회."""
        transport = SpinTransport(FakeExecutor("raise"))
        live = Live(transport=transport)
        res, recs, err = live.run()

        self.assertEqual((res.exit_code, res.stop_reason, res.input_errors), (1, "receiver_failed", 1), err)
        self.assertTrue(any(r["kind"] == "live_input" and r["input_kind"] == "receiver_failed"
                            and "receiver crashed" in r["detail"] for r in recs))
        self.assertIn("receiver crashed", res.source_stats["receiver_error"])
        self.assertIn("shutdown", transport._done)

    def test_receiver_failure_while_streaming(self):
        """검출 중 수신기 예외 → 대기 회차가 반영해 receiver_failed 종료 1, 활성 작업은 정리로 끝남."""
        ex = FakeExecutor("raise_on")
        live = Live(feed=lambda s, c: s.handle_image(image_msg(1)), transport=SpinTransport(ex),
                    hooks={0: lambda lv: (ex.release.set(), time.sleep(0.2))})
        res, recs, err = live.run()

        self.assertEqual((res.exit_code, res.stop_reason), (1, "receiver_failed"), err)
        self.assertEqual(len(processes(recs)), 1)
        self.assertTrue(live.pipeline.closed)

    def test_source_close_failure_exit_1(self):
        """수신 스레드 정리 실패(close RuntimeError)는 보고하고 종료 1."""
        live = Live(feed=lambda s, c: s.handle_image(image_msg(1)),
                    transport=FakeTransport(close_error=RuntimeError("ros receive thread did not stop")))
        live.should_stop = stop_after_jobs(live, 1)
        res, recs, err = live.run()

        self.assertEqual(res.exit_code, 1)
        self.assertIn("source close failed", err)

    def test_log_failure_during_live_input(self):
        """live_input 기록 실패는 log_failure로 종료 1이고 정리는 끝까지 한다(소스 닫힘)."""
        live = Live(feed=lambda s, c: s.handle_publishers([b"A"], ["best_effort"]))

        with mock.patch.object(PipelineLog, "write_live_input", side_effect=OSError("disk full")):
            res, recs, err = live.run()

        self.assertEqual((res.exit_code, res.stop_reason), (1, "log_failure"), err)
        self.assertEqual(len(live.transport.closed), 1)

    def test_default_watchdog_values_kept(self):
        """감시 기본값 10 s/1 s와 실시간 임시값 10개/1 s."""
        cfg = RunnerConfig()

        self.assertEqual((cfg.first_detect_timeout_s, cfg.detect_timeout_s), (10.0, 1.0))
        self.assertEqual((cfg.live_transition_limit, cfg.live_transition_window_s), (10, 1.0))

    def test_live_config_validation(self):
        """실시간 상한 값이 잘못되면 ValueError."""
        for over in ({"live_transition_limit": 0}, {"live_transition_window_s": 0}):
            with self.subTest(over=over), self.assertRaises(ValueError):
                RunnerConfig(**over)


if __name__ == "__main__":
    unittest.main()
