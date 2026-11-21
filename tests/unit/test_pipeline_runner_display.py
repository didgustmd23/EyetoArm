"""실행기 표시 전달 지점(`display_sink`) 시험: 호출 위치·순서, 현재 결과만 표시, 경계 전달, B 미사용.

`test_pipeline_runner_live.Live`(가짜 수신 연결을 붙인 실제 `LiveLinkSource`)를 쓴다. ROS·YOLO·창 없음.
"""
import io
import threading
import time
import unittest
from unittest import mock

from eyetoarm.vision import pipeline_runner as pr
from eyetoarm.vision.live_view import DisplayBridge, ViewState
from eyetoarm.vision.pipeline_records import JsonlRecorder, PipelineLog
from eyetoarm.vision.pipeline_runner import RunnerConfig, display_gate, run
from eyetoarm.vision.pipeline_types import JobStatus
from test_ai_config import ai_config
from test_ai_pipeline import ai_display, ai_observation
from test_pipeline import FakeClock, settings
from test_pipeline_runner_live import S, Live, SimpleClock, stop_after_jobs
from test_ros_image_adapter import image_msg
from test_run_pipeline import lines


class RecordingSink:
    """호출과 그 순간까지 쓰인 기록 줄 수를 순서대로 남기고 실제 브리지에도 넘기는 전달 지점."""

    def __init__(self, stream):
        """기록 스트림(호출 시점 확인용)."""
        self.stream, self.calls, self.bridge = stream, [], DisplayBridge()

    def on_outcome(self, outcome, current):
        """terminal 응답."""
        self.calls.append(("outcome", outcome.context.job_seq, outcome.status.value, current, outcome.duplicate,
                           self.stream.getvalue()))
        self.bridge.on_outcome(outcome, current)

    def on_input(self, kind, detail, generation, changed):
        """입력 항목."""
        self.calls.append(("input", kind, generation, changed, self.stream.getvalue()))
        self.bridge.on_input(kind, detail, generation, changed)


class DisplayLive(Live):
    """`Live`에 AI 설정·표시 결과·전달 지점을 더한 실행 도구."""

    def __init__(self, *, with_sink=True, sink_cls=None, **kw):
        """with_sink: display_sink를 넘긴다. sink_cls: 전달 지점 클래스(기본 RecordingSink)."""
        super().__init__(**kw)
        self.with_sink, self.sink_cls = with_sink, sink_cls or RecordingSink
        self.sink = None

    def detector(self, job, token):
        """hook 뒤 AI 표시 관측(valid=False)과 표시 결과."""
        self.detect_calls.append(job.job_seq)
        hook = self.hooks.get(job.job_seq)

        if hook is not None:
            hook(self)

        return ai_observation(job), ai_display(job)

    def run(self, limit_s=10.0):
        """데몬 스레드 실행·유한 join. (결과, 기록, stderr)."""
        stream, err, box = io.StringIO(), io.StringIO(), {}
        self.sink = self.sink_cls(stream) if self.with_sink else None
        real = pr.VisionPipeline

        def capture(*a, **kw):
            """코어를 잡고 응답을 엿본다."""
            self.pipeline = real(*a, **kw)
            return self.pipeline

        def target():
            """실행기 한 번."""
            cfg = settings(source_id="gz_link", profile=None, detector_config=ai_config())
            kw = {} if self.sink is None else {"display_sink": self.sink}

            with mock.patch.object(pr, "VisionPipeline", side_effect=capture):
                box["res"] = run(self.factory, cfg, PipelineLog(JsonlRecorder(stream)), run_id="run-live",
                                 detector=self.detector, detector_takes_token=True, config=self.cfg, stderr=err,
                                 clock_ns=FakeClock(), should_stop=self.should_stop, **kw)

        t = threading.Thread(target=target, daemon=True)
        t.start()
        t.join(limit_s)

        if t.is_alive():
            raise AssertionError(f"runner did not finish within {limit_s} s; stderr={err.getvalue()!r}")

        return box["res"], lines(stream), err.getvalue()


class DisplaySinkTest(unittest.TestCase):
    """전달 지점 호출 조건·순서."""

    def test_default_off_same_records(self):
        """display_sink 없이(기본) 같은 입력을 돌리면 기존과 같이 끝나고 표시 호출이 없다."""
        live = DisplayLive(with_sink=False, feed=lambda s, c: s.handle_image(image_msg(1)))
        live.should_stop = stop_after_jobs(live, 1)
        res, recs, err = live.run()

        self.assertEqual(res.exit_code, 0, err)
        self.assertEqual([r["status"] for r in recs if r["kind"] == "process"], ["COMPLETED"])

    def test_current_result_shown_after_log_valid_false_no_b(self):
        """정상 프레임: 처리 기록 뒤 on_outcome(current=True) 1회, 관측 valid=False, B 호출 0."""
        live = DisplayLive(feed=lambda s, c: s.handle_image(image_msg(1)))
        live.should_stop = stop_after_jobs(live, 1)
        res, recs, err = live.run()

        self.assertEqual((res.exit_code, res.b_calls), (0, 0), err)
        outcome_calls = [c for c in live.sink.calls if c[0] == "outcome"]
        self.assertEqual([c[1:5] for c in outcome_calls], [(0, "COMPLETED", True, False)])
        self.assertIn('"kind":"process"', outcome_calls[0][5].replace(" ", ""))
        item = live.sink.bridge.snapshot().item
        self.assertEqual(live.sink.bridge.shown, 1)
        self.assertEqual((item.frame.image_seq, item.job_seq, len(item.result.instances)), (0, 0, 1))
        self.assertFalse(any(r.get("valid") for r in recs if r["kind"] == "process"))

    def test_frame_timeout_during_detection(self):
        """검출 중 수신 중단: on_input(frame_timeout)이 먼저, 작업은 ABORTED로 current=False, STALE."""
        live = DisplayLive(feed=lambda s, c: s.handle_image(image_msg(1)),
                           hooks={0: lambda lv: setattr(lv.clock, "now", lv.clock.now + 2 * S)})
        live.should_stop = stop_after_jobs(live, 1)
        res, recs, err = live.run()

        self.assertEqual(res.exit_code, 0, err)
        kinds = [(c[0], c[1]) if c[0] == "input" else (c[0], c[2], c[3]) for c in live.sink.calls]
        self.assertEqual(kinds, [("input", "frame_timeout"), ("outcome", "ABORTED", False)])
        snap = live.sink.bridge.snapshot()
        self.assertIsNone(snap.item)
        self.assertEqual(snap.state, ViewState.STALE)
        self.assertEqual(live.sink.bridge.shown, 0)

    def test_rewind_during_detection_discarded(self):
        """검출 중 되감기: on_input(sim_time_rewind)이 먼저, 결과는 DISCARDED_GENERATION으로 current=False."""
        def feed(src, clock):
            """clock 10 s, 영상."""
            src.handle_clock(SimpleClock(10))
            src.handle_image(image_msg(9, 900_000_000))

        def hook(live):
            """되감기 후 대기 회차가 반영할 시간을 준다."""
            live.src.handle_clock(SimpleClock(1))
            time.sleep(0.1)

        live = DisplayLive(feed=feed, hooks={0: hook})
        live.should_stop = stop_after_jobs(live, 1)
        res, recs, err = live.run()

        self.assertEqual(res.exit_code, 0, err)
        seq = [(c[0], c[1], c[3]) if c[0] == "input" else (c[0], c[2], c[3]) for c in live.sink.calls]
        self.assertEqual(seq, [("input", "sim_time_rewind", True), ("outcome", "DISCARDED_GENERATION", False)])
        self.assertIsNone(live.sink.bridge.snapshot().item)

    def test_reconnect_then_new_generation_shown(self):
        """수신 중단 → 재접속: 이전 세대 결과는 표시되지 않고 새 세대 결과만 current=True로 표시된다."""
        def hook(live):
            """검출 중 끊김(2 s) 뒤 새 영상."""
            live.clock.now += 2 * S
            live.src.handle_image(image_msg(5))

        live = DisplayLive(feed=lambda s, c: s.handle_image(image_msg(1)), hooks={0: hook})
        live.should_stop = stop_after_jobs(live, 2)
        res, recs, err = live.run()

        self.assertEqual(res.exit_code, 0, err)
        outcomes = [(c[1], c[2], c[3]) for c in live.sink.calls if c[0] == "outcome"]
        self.assertEqual(outcomes[-1][1:], ("COMPLETED", True))
        self.assertTrue(all(not cur for _, status, cur in outcomes[:-1]), outcomes)
        inputs = [c[1] for c in live.sink.calls if c[0] == "input"]
        self.assertEqual(inputs, ["frame_timeout", "reconnect"])
        item = live.sink.bridge.snapshot().item
        self.assertEqual((item.generation, item.frame.timestamp_ms), (1, 5000))

    def test_timeout_late_result_not_shown(self):
        """검출 시간 초과: ABORTED detect_timeout만 전달하고 늦은 결과는 전달하지 않는다. 종료 1."""
        cfg = RunnerConfig(close_attempts=1, first_detect_timeout_s=0.05, detect_timeout_s=0.05, cancel_grace_s=1.0,
                           stop_check_interval_s=0.01)
        live = DisplayLive(cfg=cfg, feed=lambda s, c: s.handle_image(image_msg(1)),
                           hooks={0: lambda lv: time.sleep(0.15)})
        live.should_stop = stop_after_jobs(live, 1)
        res, recs, err = live.run()

        self.assertEqual((res.exit_code, res.timeouts, res.late_results), (1, 1, 1), err)
        self.assertEqual([(c[2], c[3], c[4]) for c in live.sink.calls if c[0] == "outcome"],
                         [("ABORTED", False, False)])
        self.assertEqual(live.sink.bridge.snapshot().state, ViewState.ERROR)

    def test_detector_error_clears(self):
        """검출 예외: INFERENCE_ERROR FAILED가 current=False로 전달되고 칸은 비며 ERROR."""
        def boom(live):
            """검출 실패."""
            raise RuntimeError("boom")

        live = DisplayLive(feed=lambda s, c: s.handle_image(image_msg(1)), hooks={0: boom})
        live.should_stop = stop_after_jobs(live, 1)
        res, recs, err = live.run()

        self.assertEqual(res.exit_code, 1, err)
        self.assertEqual([(c[2], c[3]) for c in live.sink.calls if c[0] == "outcome"], [("FAILED", False)])
        snap = live.sink.bridge.snapshot()
        self.assertEqual((snap.state, snap.item), (ViewState.ERROR, None))

    def test_sink_failure_is_run_failure(self):
        """전달 지점 예외는 실행 실패(1)로 정리된다(소스 close까지 진행)."""
        class Broken(RecordingSink):
            """on_outcome에서 실패."""

            def on_outcome(self, outcome, current):
                """실패."""
                raise RuntimeError("display sink broke")

        live = DisplayLive(feed=lambda s, c: s.handle_image(image_msg(1)), sink_cls=Broken)
        live.should_stop = stop_after_jobs(live, 1)
        res, recs, err = live.run()

        self.assertEqual(res.exit_code, 1)
        self.assertIn("display sink broke", err)
        self.assertEqual(live.transport.closed, [live.source_cfg.close_limit_s])


class DisplayGateTest(unittest.TestCase):
    """`display_gate`: valid를 보지 않고 최초 현재 적용 결과만 통과."""

    def test_gate_matches_b_gate_except_validity(self):
        """표시용 invalid 관측은 통과, duplicate·세대 변경 뒤는 불통과. b_gate는 invalid라 불통과."""
        from eyetoarm.vision.pipeline import GenerationCause, VisionPipeline
        from test_pipeline import RUN_ID, frame

        s = settings(detector_config=ai_config())
        p = VisionPipeline(s, run_id=RUN_ID, clock_ns=FakeClock())
        p.offer_frame(frame(0))
        job = p.start_next()
        out = p.complete(job, ai_observation(job), display_result=ai_display(job))

        self.assertEqual(out.status, JobStatus.COMPLETED)
        self.assertTrue(display_gate(p, out))
        self.assertFalse(pr.b_gate(p, out))
        self.assertFalse(display_gate(p, p.complete(job, ai_observation(job), display_result=ai_display(job))))
        p.change_settings(p.settings, cause=GenerationCause.RECONNECT)
        self.assertFalse(display_gate(p, out))


if __name__ == "__main__":
    unittest.main()
