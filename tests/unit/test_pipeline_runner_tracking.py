"""실행기 추적 선택 연결(#151 S2 `tracker`) 시험 — 갱신 지점·경계 reset·실패 격리·기본 None 회귀·CLI opt-in.

근거: reviewer 구현 명세 #151 §1(D6·D10·D11)·§3, 계획 `_workspace/150_builder2_part_track_id_plan.md` r2 §4.1·§4.2. 가짜 저장 소스·가짜 실시간
소스·실제 LiveLinkSource(가짜 수신)만 쓴다(ROS·YOLO·창 없음). 추적 성능·신원 정확도는 확인하지 않는다.
"""
import io
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

from eyetoarm.common.types import InputStatus
from eyetoarm.vision import pipeline_runner as pr
from eyetoarm.vision.handoff import AToCHandoff
from eyetoarm.vision.pipeline_records import JsonlRecorder, PipelineLog
from eyetoarm.vision.pipeline_runner import RunnerConfig, run
from eyetoarm.vision.tracking import PartTracker, PartTrackIdIssuer, TrackMode
from test_ai_pipeline import ai_display, ai_observation
from test_pipeline import RUN_ID, FakeClock, frame
from test_pipeline_runner_display import DisplayLive
from test_pipeline_runner_handoff import FakeLiveSource, ai_settings, stored_factory
from test_pipeline_runner_live import S, SimpleClock, stop_after_jobs
from test_pipeline_watchdog import watch
from test_ros_image_adapter import image_msg, info_msg
from test_run_pipeline import FakeSource, event, lines

OFF = RunnerConfig(close_attempts=1, close_retry_interval_s=0.0, first_detect_timeout_s=None, detect_timeout_s=None)


class RecTracker(PartTracker):
    """호출을 기록하는 추적기(시험 관측용). fail_at: 그 번째 update에서 예외."""

    def __init__(self, *a, fail_at=None, fail_reset=None, fail_disable=False, fail_stats=False, **kw):
        """기록 목록 준비. fail_reset: 그 사유의 reset에서 예외, fail_disable·fail_stats: 그 호출에서 예외."""
        super().__init__(*a, **kw)
        self.frames, self.resets, self.fail_at, self.calls = [], [], fail_at, 0
        self.fail_reset, self.fail_disable, self.fail_stats = fail_reset, fail_disable, fail_stats

    def update(self, key, timestamp_ms, inputs, **kw):
        """update 기록(실패 주입)."""
        self.calls += 1

        if self.fail_at is not None and self.calls == self.fail_at:
            raise RuntimeError("tracker boom")

        f = super().update(key, timestamp_ms, inputs, **kw)
        self.frames.append(f)
        return f

    def reset(self, reason):
        """reset 기록(실패 주입)."""
        self.resets.append(reason)

        if reason == getattr(self, "fail_reset", None):
            raise RuntimeError(f"reset failure: {reason}")

        super().reset(reason)

    def disable(self, reason):
        """disable(실패 주입)."""
        if getattr(self, "fail_disable", False):
            raise RuntimeError("disable failure")

        super().disable(reason)

    def stats(self):
        """stats(실패 주입)."""
        if getattr(self, "fail_stats", False):
            raise RuntimeError("stats failure")

        return super().stats()


def tracker(**kw):
    """새 발급기·M2 기록 추적기."""
    return RecTracker(PartTrackIdIssuer(namespace="ns-run"), mode=TrackMode.HEURISTIC, **kw)


def detector(hooks=None):
    """AI 표시 관측(valid=False)+표시 결과. hooks[job_seq]()로 실패·지연 주입."""
    def det(job):
        """검출 대체."""
        hook = (hooks or {}).get(job.job_seq)

        if hook is not None:
            hook()

        return ai_observation(job), ai_display(job)

    return det


def run_stored(factory, *, trk=None, config=OFF, log=None, stream=None, **kw):
    """저장 소스 실행. (결과, 기록, stderr)."""
    stream = stream if stream is not None else io.StringIO()
    err = io.StringIO()
    log = log or PipelineLog(JsonlRecorder(stream))
    extra = {} if trk is None else {"part_tracker": trk}
    res = run(factory, ai_settings(), log, run_id=RUN_ID, detector=kw.pop("det", detector()), config=config,
              stderr=err, clock_ns=FakeClock(), **extra, **kw)

    return res, lines(stream), err.getvalue()


class DefaultAndNormalTest(unittest.TestCase):
    """기본 None 회귀와 정상 갱신."""

    def test_default_off_same_records(self):
        """tracker를 넘겨도(정상 실행) `ai_tracks` 줄(S3, 갱신마다 1줄)을 뺀 기록·계수·종료 코드가 넘기지 않은 실행과 같고,
        넘기지 않으면 ai_tracks 줄이 없고 result.tracking은 None."""
        res0, recs0, _ = run_stored(stored_factory(4, {}))
        res1, recs1, err1 = run_stored(stored_factory(4, {}), trk=tracker())

        self.assertEqual((res0.exit_code, res1.exit_code), (0, 0), err1)
        self.assertEqual(recs0, [r for r in recs1 if r["kind"] != "ai_tracks"])
        self.assertEqual(sum(r["kind"] == "ai_tracks" for r in recs1), 4)
        self.assertFalse(any(r["kind"] == "ai_tracks" for r in recs0))
        self.assertEqual(res0.counters, res1.counters)
        self.assertIsNone(res0.tracking)
        self.assertEqual(res1.tracking["updates"], 4)

    def test_update_once_per_published_result_with_job_keys(self):
        """공개된 결과마다 1회 갱신, FrameKey가 작업·프레임 식별자와 같고 번호는 [None, None, 1, 1]."""
        trk = tracker()
        res, _, err = run_stored(stored_factory(4, {}), trk=trk)

        self.assertEqual(res.exit_code, 0, err)
        self.assertEqual([(f.image_seq, f.job_seq, f.generation) for f in trk.frames],
                         [(0, 0, 0), (1, 1, 0), (2, 2, 0), (3, 3, 0)])
        self.assertEqual([f.assignments[0].part_track_id for f in trk.frames], [None, None, 1, 1])
        self.assertEqual([f.timestamp_ms for f in trk.frames], [10000, 10001, 10002, 10003])
        self.assertTrue(all(f.updated for f in trk.frames))

    def test_run_start_and_end_reset(self):
        """실행 시작·끝에서 reset(run_start·run_end). 같은 추적기를 다음 실행에 써도 이전 트랙이 이어지지 않는다."""
        trk = tracker()
        run_stored(stored_factory(3, {}), trk=trk)
        self.assertEqual((trk.resets[0], trk.resets[-1]), ("run_start", "run_end"))
        self.assertEqual(trk.stats()["tracks"], 0)


class NotUpdatedTest(unittest.TestCase):
    """실패·중단·늦은·보류 결과는 갱신하지 않는다(정보 없음)."""

    def test_detector_failure_not_tracked(self):
        """job1 검출 예외: 그 작업은 갱신 없음(3장 중 2회)."""
        def boom():
            """검출 실패."""
            raise RuntimeError("boom")

        trk = tracker()
        res, _, _ = run_stored(stored_factory(3, {}), trk=trk, det=detector({1: boom}))

        self.assertEqual(res.exit_code, 1)
        self.assertEqual([f.job_seq for f in trk.frames], [0, 2])

    def test_timeout_and_late_result_not_tracked(self):
        """시간 초과 작업과 유예 안 늦은 결과(duplicate)는 갱신하지 않는다."""
        trk = tracker()
        res, _, err = run_stored(stored_factory(1, {}), trk=trk, det=detector({0: lambda: time.sleep(0.15)}),
                                 config=watch(0.05, 0.05, grace=1.0, interval=0.01))

        self.assertEqual((res.exit_code, res.timeouts, res.late_results), (1, 1, 1), err)
        self.assertEqual(trk.frames, [])

    def test_withheld_not_tracked(self):
        """공개 직전 수신 epoch가 현재가 아니면(withheld) 갱신하지 않는다."""
        trk = tracker()
        box = {}

        def factory(issuer, provider):
            """job0 공개, job1 withheld."""
            box["src"] = FakeLiveSource([frame(issuer.next_seq(), provider()) for _ in range(2)],
                                        current_answers=[True, True, True, False])
            return box["src"]

        p = {}
        real = pr.VisionPipeline

        def capture(*a, **kw):
            """코어 잡기."""
            p["c"] = real(*a, **kw)
            return p["c"]

        def stop():
            """작업 2개 끝나면 정지."""
            c = p["c"].counters
            return c.completed + c.failed + c.aborted >= 2

        with mock.patch.object(pr, "VisionPipeline", side_effect=capture):
            res, _, err = run_stored(factory, trk=trk, should_stop=stop)

        self.assertEqual(res.exit_code, 0, err)
        self.assertEqual([f.job_seq for f in trk.frames], [0])


class BoundaryTest(unittest.TestCase):
    """입력 경계 reset(§4.2): 되감기·프로필·재접속 종료, FRAME_TIMEOUT 보류, 입력 오류 종료."""

    def run_live(self, live, trk):
        """DisplayLive에 tracker를 더해 실행."""
        import test_pipeline_runner_display as tdisp

        def patched(*a, **kw):
            """tracker 추가."""
            kw["part_tracker"] = trk
            return run(*a, **kw)

        with mock.patch.object(tdisp, "run", side_effect=patched):
            return live.run()

    def test_rewind_profile_reset(self):
        """검출 중 되감기·프로필 변경 → 그 경계 이름으로 reset."""
        for name, boundary in (("sim_time_rewind", lambda lv: (lv.src.handle_clock(SimpleClock(1)), time.sleep(0.1))),
                               ("profile_changed", lambda lv: (lv.src.handle_camera_info(info_msg(fx=101.0)), time.sleep(0.1)))):
            with self.subTest(name=name):
                def feed(src, clock):
                    """시계 10 s·영상."""
                    src.handle_clock(SimpleClock(10))
                    src.handle_image(image_msg(9, 900_000_000))

                trk = tracker()
                live = DisplayLive(with_sink=False, feed=feed, hooks={0: boundary})
                live.should_stop = stop_after_jobs(live, 1)
                res, _, err = self.run_live(live, trk)

                self.assertEqual(res.exit_code, 0, err)
                self.assertIn(name, trk.resets)

    def test_frame_timeout_hold_then_reconnect_reset(self):
        """수신 중단은 reset하지 않고(보류), 이어진 재접속은 reconnect reset."""
        def hook(lv):
            """끊김 뒤 새 영상(재접속)."""
            lv.clock.now += 2 * S
            time.sleep(0.05)
            lv.src.handle_image(image_msg(5))

        trk = tracker()
        live = DisplayLive(with_sink=False, feed=lambda s, c: s.handle_image(image_msg(1)), hooks={0: hook})
        live.should_stop = stop_after_jobs(live, 2)
        res, _, err = self.run_live(live, trk)

        self.assertEqual(res.exit_code, 0, err)
        middle = [r for r in trk.resets if r not in ("run_start", "run_end")]
        self.assertIn("reconnect", middle)
        self.assertNotIn("input_status:FRAME_TIMEOUT", middle)

    def test_stored_input_error_resets(self):
        """저장 입력의 디코딩 실패 사건 → input_status 사유 reset."""
        def factory(issuer, provider):
            """프레임 → 디코딩 실패 → 프레임."""
            return FakeSource([frame(issuer.next_seq(), provider()), event(InputStatus.DECODE_FAILED),
                               frame(issuer.next_seq(), provider())])

        trk = tracker()
        run_stored(factory, trk=trk)

        self.assertIn("input_status:DECODE_FAILED", trk.resets)


class FailureIsolationTest(unittest.TestCase):
    """D10: 추적 예외 격리 vs 기록·연결부 실패 전파."""

    def test_tracking_exception_isolated(self):
        """2번째 갱신에서 추적 예외: tracking_failed 기록 1줄·stderr, 추적 끔(이후 update·reset 호출 없음, S2 r1 latch), 종료 코드·코어 계수는 추적 없는 실행과 같음."""
        res0, _, _ = run_stored(stored_factory(4, {}))
        trk = tracker(fail_at=2)
        res, recs, err = run_stored(stored_factory(4, {}), trk=trk)

        self.assertEqual(res.exit_code, 0, err)
        self.assertEqual(res.counters, res0.counters)
        failed = [r for r in recs if r["kind"] == "tracking_failed"]
        self.assertEqual(len(failed), 1)
        self.assertEqual((failed[0]["job_seq"], failed[0]["image_seq"]), (1, 1))
        self.assertIn("tracker boom", failed[0]["error"])
        self.assertIn("tracking_failed", err)
        self.assertTrue(trk.disabled)
        self.assertEqual((trk.calls, len(trk.frames)), (2, 1))
        self.assertNotIn("run_end", trk.resets)
        self.assertEqual(res.tracking["failed"], "RuntimeError: tracker boom")

    def test_recorder_failure_still_fail_stop(self):
        """tracking_failed 기록이 실패하면 추적 예외로 삼키지 않고 기존 log_failure(종료 1)."""
        class BrokenLog(PipelineLog):
            """tracking_failed 쓰기 실패."""

            def write_tracking_failed(self, record):
                """실패."""
                raise OSError("disk full")

        res, _, err = run_stored(stored_factory(3, {}), trk=tracker(fail_at=1),
                                 log=BrokenLog(JsonlRecorder(io.StringIO())))

        self.assertEqual((res.exit_code, res.stop_reason), (1, "log_failure"), err)

    def test_handoff_failure_still_handoff_failed(self):
        """추적을 켜도 연결부 넘침은 기존대로 handoff_failed(종료 1)."""
        res, _, err = run_stored(stored_factory(4, {}), trk=tracker(), handoff_sink=AToCHandoff(capacity=2))

        self.assertEqual((res.exit_code, res.stop_reason), (1, "handoff_failed"), err)

    def test_tracker_contract(self):
        """tracker가 PartTracker가 아니면 소스를 만들기 전에 실행 실패(tracker_contract)."""
        made = []
        res, _, err = run_stored(lambda i, p: made.append(1), trk="not a tracker")

        self.assertEqual((res.exit_code, res.stop_reason, made), (1, "tracker_contract", []), err)


class FailureLatchTest(unittest.TestCase):
    """S2 r1: 실행기 추적 실패 latch(F2)와 run_end reset 실패 진단(F1)."""

    def test_disable_failure_still_stops_tracking(self):
        """update 예외 뒤 disable도 실패해도 그 실행의 추적 갱신은 멈춘다(update 2회뿐), 진단 1줄·계수 불변."""
        res0, _, _ = run_stored(stored_factory(4, {}))
        trk = tracker(fail_at=2, fail_disable=True)
        res, recs, err = run_stored(stored_factory(4, {}), trk=trk)

        self.assertEqual(res.exit_code, 0, err)
        self.assertEqual(trk.calls, 2)
        self.assertEqual(len(trk.frames), 1)
        self.assertEqual(res.counters, res0.counters)
        failed = [r for r in recs if r["kind"] == "tracking_failed"]
        self.assertEqual(len(failed), 1)
        self.assertIn("disable failure", failed[0]["disable_error"])
        self.assertEqual(sum("tracking_failed" in e for e in res.errors), 1)
        self.assertNotIn("run_end", trk.resets)        # 실패 뒤 reset도 다시 시도하지 않음

    def test_run_end_reset_failure_diagnosed_once(self):
        """run_end reset 실패: 추적 실패 규칙대로 errors·stderr·JSONL tracking_failed 1줄(작업 ids 없음), 종료 코드 0 유지."""
        trk = tracker(fail_reset="run_end")
        res, recs, err = run_stored(stored_factory(2, {}), trk=trk)

        self.assertEqual(res.exit_code, 0, err)
        failed = [r for r in recs if r["kind"] == "tracking_failed"]
        self.assertEqual(len(failed), 1)
        self.assertIn("reset failure: run_end", failed[0]["error"])
        self.assertNotIn("job_seq", failed[0])
        self.assertIn("tracking_failed", err)
        self.assertEqual(res.tracking["failed"], "RuntimeError: reset failure: run_end")

    def test_run_end_reset_failure_with_log_failure_keeps_cleanup(self):
        """run_end reset 실패의 기록도 실패하면 기존 기록 실패 규칙(종료 1, 먼저 정해진 정지 사유 유지, 오류 진단)이고 코어 close·소스 close·연결부 정리는 그대로."""
        class BrokenLog(PipelineLog):
            """tracking_failed 쓰기 실패."""

            def write_tracking_failed(self, record):
                """실패."""
                raise OSError("disk full")

        holder = {}
        h = AToCHandoff()
        res, _, err = run_stored(stored_factory(2, holder), trk=tracker(fail_reset="run_end"),
                                 log=BrokenLog(JsonlRecorder(io.StringIO())), handoff_sink=h)

        self.assertEqual((res.exit_code, res.stop_reason), (1, "end_of_stream"), err)
        self.assertTrue(any(e.startswith("log write failed") for e in res.errors), res.errors)
        self.assertTrue(res.closed)
        self.assertEqual(holder["source"].closes, 1)
        self.assertIsNotNone(res.handoff)

    def test_stats_failure_does_not_break_run_return(self):
        """종료 보고용 stats 예외도 run 반환을 막지 않는다(tracking에 stats_error)."""
        res, _, err = run_stored(stored_factory(2, {}), trk=tracker(fail_stats=True))

        self.assertEqual(res.exit_code, 0, err)
        self.assertIn("stats failure", res.tracking["stats_error"])


class CliTest(unittest.TestCase):
    """opt-in CLI: 진입점이 발급기 1개를 만들어 주입, 기본은 추적 없음."""

    def setUp(self):
        """scripts를 import 경로에 둔다."""
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        sys.path.insert(0, os.path.join(root, "scripts"))
        self.addCleanup(sys.path.remove, os.path.join(root, "scripts"))

    def test_view_live_detection_tracking_flag(self):
        """--tracking이면 runner에 M2 PartTracker(주입된 발급기)를 part_tracker로 넘기고, 없으면 넘기지 않는다."""
        from test_view_live_detection import FakeRunner, FakeWindow, ViewerMainTest

        base = ViewerMainTest("test_runner_wired_like_cli")
        base.setUp()
        self.addCleanup(base.doCleanups)

        for flag, expect in ((("--tracking",), True), ((), False)):
            with self.subTest(flag=flag):
                runner = FakeRunner()
                code, err = base.main(FakeWindow(keys={3: 27}), runner, extra=flag)
                self.assertEqual(code, 0, err)
                trk = runner.kw.get("part_tracker")
                self.assertEqual(isinstance(trk, PartTracker), expect)

                if expect:
                    self.assertIs(trk.mode, TrackMode.HEURISTIC)

    def test_run_pipeline_tracking_requires_ai(self):
        """run_pipeline --tracking: OpenCV 검출이면 argparse 오류(종료 2, 기존 인자 조합 규약), AI면 run에 part_tracker를 넘긴다."""
        import run_pipeline

        with tempfile.TemporaryDirectory() as tmp:
            img = os.path.join(tmp, "a.png")
            open(img, "wb").close()
            log = os.path.join(tmp, "out.jsonl")
            with mock.patch("sys.stderr", new_callable=io.StringIO) as perr:
                with self.assertRaises(SystemExit) as cm:
                    run_pipeline.main(["--images", img, "--log", log, "--detect-config", img, "--tracking"])

            self.assertEqual(cm.exception.code, 2)
            self.assertIn("--tracking is for --detector ai only", perr.getvalue())

            seen = {}

            def fake_run(*a, **kw):
                """run 대체."""
                seen.update(kw)
                return pr.RunResult(exit_code=0, stop_reason="end_of_stream", run_id="r")

            cfg_path = os.path.join(tmp, "ai.json")
            write_ai_config(cfg_path, tmp)

            with mock.patch.object(run_pipeline, "run", side_effect=fake_run):
                code = run_pipeline.main(["--images", img, "--log", log, "--detector", "ai", "--ai-config", cfg_path,
                                          "--tracking"], stderr=io.StringIO(), model_factory=lambda *a: None)

            self.assertEqual(code, 0)
            self.assertIsInstance(seen.get("part_tracker"), PartTracker)


def write_ai_config(path, tmp):
    """가짜 가중치를 가리키는 AI 설정 JSON."""
    import hashlib
    import json

    from test_ai_config import AI_FIELDS

    weights = os.path.join(tmp, "best.pt")

    with open(weights, "wb") as f:
        f.write(b"fake")

    fields = dict(AI_FIELDS)
    fields.update(weights_path=weights, weights_sha256=hashlib.sha256(b"fake").hexdigest())

    with open(path, "w", encoding="utf-8") as f:
        json.dump(fields, f)



if __name__ == "__main__":
    unittest.main()
