"""작업 6a 저장 입력 실행기 단위·결합시험 — 실제 저장 이미지·짧은 영상, terminal 통지, B 소비 조건, 정리·종료 코드.

근거: 계획 r4 작업 6(P1~P3 보완), A 상세설계 §5.4·§9, 작업 4 순환 계약(통지 없으면 활성 유지), 작업 5 fail-stop 기록.
유효 관측과 B 소비 지점 시험은 **시험용 detector**를 쓴다. 현재 OpenCV 표시 어댑터는 유효 관측을 내지 않는다.
실제 C·ROS·YOLO·B 계산을 쓰지 않으며, 이 시험은 C 영향 검증이 아니다.
"""
import dataclasses
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest

import cv2

from eyetoarm.common.types import InputStatus, InputStatusEvent, InvalidationKind, TimestampBase
from eyetoarm.inputs.base import FrameIdIssuer
from eyetoarm.inputs.stored_images import StoredImageSource
from eyetoarm.inputs.video import VideoSource
from eyetoarm.vision.pipeline import GenerationCause, PipelineContractError, VisionPipeline
from eyetoarm.vision.pipeline_records import JsonlRecorder, PipelineLog
from eyetoarm.vision.pipeline_runner import JobCancelled, RunnerConfig, apply_seek, b_gate, run
from eyetoarm.vision.pipeline_types import QueueDropReason
from test_pipeline import RUN_ID, FakeClock, config, frame, observation, profile, settings
from test_pipeline_opencv import CENTER, EDGE, scene
from test_pipeline_terminal import frameless_event, valid_observation
from test_video_input import write_video

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT = os.path.join(ROOT, "scripts", "run_pipeline.py")
# 6a 동기 경로 시험: 6b 기본값(감시 켬 10/1)과 달리 감시를 명시적으로 끈다(동기 호출·스레드 1개 전제 유지)
FAST = RunnerConfig(close_attempts=3, close_retry_interval_s=0.0, max_consecutive_input_errors=3,
                    first_detect_timeout_s=None, detect_timeout_s=None)


def lines(stream):
    """JSONL 문자열 스트림을 dict 목록으로 읽는다."""
    return [json.loads(t) for t in stream.getvalue().splitlines()]


def kinds(stream):
    """기록 종류 목록."""
    return [r["kind"] for r in lines(stream)]


class FakeSource:
    """정해 둔 항목(FramePacket·InputStatusEvent·예외)을 차례로 돌려주는 가짜 소스. close 호출 수를 센다."""

    def __init__(self, items, eos=True):
        """돌려줄 항목 목록을 받는다. 다 쓰면 END_OF_STREAM(eos=False면 마지막 항목 반복)."""
        self.items, self.eos, self.closes, self.last = list(items), eos, 0, None

    def read(self):
        """다음 항목."""
        if self.items:
            item = self.items.pop(0)
            self.last = item
        elif self.eos:
            item = InputStatusEvent(run_id=RUN_ID, source_id="src0", status=InputStatus.END_OF_STREAM,
                                    event_monotonic_ns=1, detail="end of list")
        else:
            item = self.last

        if isinstance(item, BaseException):
            raise item

        return item

    def close(self):
        """닫기 수를 센다."""
        self.closes += 1


def frames_factory(n, holder=None, **frame_over):
    """n개 프레임(현재 세대·run_id 반영)을 주는 가짜 소스 팩토리. holder에 소스를 넣는다."""
    def factory(issuer, generation_provider):
        """issuer·세대 공급자를 받아 FakeSource를 만든다."""
        items = [dataclasses.replace(frame(issuer.next_seq(), generation_provider(), **frame_over))
                 for _ in range(n)]
        src = FakeSource(items)
        if holder is not None:
            holder["source"], holder["issuer"], holder["provider"] = src, issuer, generation_provider
        return src

    return factory


def event(status, detail="x"):
    """입력 사건."""
    return InputStatusEvent(run_id=RUN_ID, source_id="src0", status=status, event_monotonic_ns=5, detail=detail)


class ScriptedDetector:
    """작업 순번별로 정한 동작을 하는 시험용 detector. 동시 호출 수를 센다."""

    def __init__(self, plan):
        """plan: job_seq → 'valid' | 'invalid' | 예외 객체. 없으면 'invalid'."""
        self.plan, self.calls, self.active, self.max_active = plan, [], 0, 0

    def __call__(self, job):
        """정해 둔 결과를 돌려주거나 예외를 낸다."""
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.calls.append(job.job_seq)

        try:
            what = self.plan.get(job.job_seq, "invalid")

            if isinstance(what, BaseException):
                raise what

            return (valid_observation(job) if what == "valid" else observation(job)), None
        finally:
            self.active -= 1


class Base(unittest.TestCase):
    """기록 스트림·기본 실행 도우미."""

    def run_with(self, factory, detector=None, sett=None, stream=None, **kw):
        """주어진 소스 팩토리로 실행하고 (결과, 스트림, stderr)를 돌려준다."""
        stream = stream if stream is not None else io.StringIO()
        err = io.StringIO()
        log = PipelineLog(JsonlRecorder(stream))
        kw.setdefault("config", FAST)
        kw.setdefault("clock_ns", FakeClock())
        res = run(factory, sett or settings(), log, run_id=RUN_ID, detector=detector or ScriptedDetector({}),
                  stderr=err, **kw)

        return res, stream, err

    def assert_balance(self, res):
        """종료 후 보존식(A §9)."""
        c = res.counters
        self.assertEqual(c.received, c.first_started + sum(c.queue_dropped.values()))
        self.assertEqual(c.first_started + c.reprocess_started,
                         c.completed + c.generation_discarded + c.failed + c.aborted)


class RealStoredInputTest(Base):
    """실제 저장 이미지·짧은 영상으로 끝까지 실행(OpenCV 표시 어댑터, 계획 작업 6 항목 7)."""

    def test_real_stored_images_run_to_end(self):
        """실제 PNG 3장과 깨진 파일 1개를 끝까지 처리: EOF까지 계속, 처리 기록 3, 입력 오류 1(입력 상태 기록), 원본 시각 보존, 보존식. 입력 오류가 있으므로 종료 1."""
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i, img in enumerate((scene(CENTER), scene(EDGE), scene())):
                paths.append(os.path.join(d, f"{i}.png"))
                cv2.imwrite(paths[-1], img)
            bad = os.path.join(d, "bad.png")
            with open(bad, "wb") as f:
                f.write(b"not an image")
            paths.insert(1, bad)

            def factory(issuer, provider):
                """실제 StoredImageSource."""
                return StoredImageSource(paths, source_id="src0", issuer=issuer, generation_provider=provider)

            res, stream, err = self.run_with(factory, detector=__import__(
                "eyetoarm.vision.pipeline_opencv", fromlist=["process_opencv"]).process_opencv,
                sett=settings(profile=None), clock_ns=__import__("time").monotonic_ns)

        self.assertEqual((res.exit_code, res.stop_reason), (1, "end_of_stream"), err.getvalue())
        self.assertEqual((res.frames, res.input_errors, res.closed), (3, 1, True))
        self.assertIn("completed with 0 failed job(s) and 1 input error(s)", err.getvalue())
        recs = lines(stream)
        procs = [r for r in recs if r["kind"] == "process"]
        self.assertEqual([p["reason"] for p in procs], ["CALIBRATION_MISMATCH"] * 3)
        self.assertTrue(all(p["timestamp_base"] == TimestampBase.PC_RECEIVE.value for p in procs))
        self.assertEqual([r["status"] for r in recs if r["kind"] == "input_state"],
                         ["DECODE_FAILED", "END_OF_STREAM"])
        self.assertEqual(recs[0]["kind"], "run_meta")
        self.assert_balance(res)

    def test_short_video_runs_to_end(self):
        """짧은 실제 영상(MJPG 5프레임)을 끝까지 처리: 종료 0, 처리 기록 5, VIDEO_POS 시각이 프레임 번호÷FPS."""
        with tempfile.TemporaryDirectory() as d:
            path = write_video(os.path.join(d, "clip.avi"), 5, fps=10.0)

            def factory(issuer, provider):
                """실제 VideoSource."""
                return VideoSource(path, source_id="src0", issuer=issuer, generation_provider=provider)

            res, stream, err = self.run_with(factory, detector=__import__(
                "eyetoarm.vision.pipeline_opencv", fromlist=["process_opencv"]).process_opencv,
                sett=settings(profile=None))

        self.assertEqual((res.exit_code, res.stop_reason, res.frames), (0, "end_of_stream", 5), err.getvalue())
        procs = [r for r in lines(stream) if r["kind"] == "process"]
        self.assertEqual([p["timestamp_ms"] for p in procs], [0, 100, 200, 300, 400])
        self.assertTrue(all(p["timestamp_base"] == "VIDEO_POS" for p in procs))
        self.assert_balance(res)

    def test_source_receives_one_shared_issuer_and_generation_provider(self):
        """소스 팩토리는 실행기의 단일 issuer(같은 run_id)와 파이프라인 현재 세대 공급자를 받는다."""
        holder = {}
        res, _, _ = self.run_with(frames_factory(2, holder))

        self.assertIsInstance(holder["issuer"], FrameIdIssuer)
        self.assertEqual(holder["issuer"].run_id, res.run_id)
        self.assertEqual(holder["provider"](), 0)
        self.assertEqual(holder["source"].closes, 1)

    def test_runner_has_no_ros_qt_c_or_yolo_imports(self):
        """실행기·CLI를 불러도 ROS·Qt·ultralytics·C 서버·GUI·TCP 모듈이 로드되지 않는다."""
        code = ("import sys, importlib.util\n"
                "import eyetoarm.vision.pipeline_runner\n"
                f"spec = importlib.util.spec_from_file_location('rp', {SCRIPT!r})\n"
                "m = importlib.util.module_from_spec(spec)\n"
                "spec.loader.exec_module(m)\n"
                "roots = ('rclpy', 'PySide6', 'ultralytics', 'torch', 'robot_server_package')\n"
                "bad = [k for k in sys.modules if k.split('.')[0] in roots\n"
                "       or k.startswith(('eyetoarm.tcp', 'eyetoarm.gui'))]\n"
                "print(','.join(sorted(bad)) or 'none')")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                             env={"PYTHONPATH": os.path.join(ROOT, "src"), "PATH": ""}, cwd=ROOT)
        self.assertEqual(out.stdout.strip(), "none")


class BGateTest(Base):
    """B 소비 조건(계획 r4 작업 6 「소비 조건」)."""

    def test_b_callback_called_once_for_current_valid_only(self):
        """유효→무효 순서에서 B 지점은 유효 완료 응답 1회만 받고, 두 번째 무효 결과는 이전 유효 결과를 재사용하지 않는다."""
        calls = []
        sett = settings()
        res, stream, _ = self.run_with(frames_factory(2), detector=ScriptedDetector({0: "valid", 1: "invalid"}),
                                       sett=sett, b_consumer=lambda f, o: calls.append((f.image_seq, o.valid)))

        self.assertEqual(calls, [(0, True)])
        self.assertEqual(res.b_calls, 1)
        procs = [r for r in lines(stream) if r["kind"] == "process"]
        self.assertEqual([(p["valid"], p["reason"]) for p in procs], [(True, None), (False, "NO_OBJECT")])

    def test_b_gate_rejects_stale_replayed_or_unapplied_outcome(self):
        """duplicate·차단선 이하(applied=False)·무효·다음 결과 뒤 재전달·세대/설정/프로파일 전이 뒤 재전달은 모두 거부된다."""
        p = VisionPipeline(settings(), run_id=RUN_ID, clock_ns=FakeClock())

        def go(seq, valid=True):
            """작업 하나를 시작해 완료한다."""
            p.offer_frame(frame(seq, p.current_generation()))
            j = p.start_next()
            return j, p.complete(j, valid_observation(j) if valid else observation(j))

        j0, o0 = go(0)
        self.assertTrue(b_gate(p, o0))
        self.assertFalse(b_gate(p, p.complete(j0, None)))         # duplicate

        _, o1 = go(1, valid=False)
        self.assertFalse(b_gate(p, o1))                            # 무효
        self.assertFalse(b_gate(p, o0))                            # 다음 결과 뒤 재전달

        p.offer_frame(frame(2))
        j2 = p.start_next()
        p.apply_invalidation(frameless_event(p))
        o2 = p.complete(j2, valid_observation(j2))
        self.assertFalse(o2.applied)
        self.assertFalse(b_gate(p, o2))                            # 차단선 이하 늦은 완료

        _, o3 = go(3)
        self.assertTrue(b_gate(p, o3))
        p.change_settings(dataclasses.replace(p.settings, detector_config=config(fixed_threshold=90)),
                          cause=GenerationCause.DETECTOR_CONFIG_CHANGED)
        self.assertFalse(b_gate(p, o3))                            # 세대·설정 전이 뒤

        q = VisionPipeline(settings(profile=None), run_id=RUN_ID, clock_ns=FakeClock())
        q.offer_frame(frame(0))
        jq = q.start_next()
        self.assertFalse(b_gate(q, q.complete(jq, valid_observation(jq))))   # 프로파일 없음

        q2 = VisionPipeline(settings(profile=profile(calibration_id="cal-9")), run_id=RUN_ID, clock_ns=FakeClock())
        q2.offer_frame(frame(0))
        jq2 = q2.start_next()
        self.assertFalse(b_gate(q2, q2.complete(jq2, valid_observation(jq2))))  # 프로파일 불일치


class TerminalNoticeTest(Base):
    """terminal 통지 보장(반환하는 호출·전달된 중단) — 계획 r4 작업 6 「terminal 통지 보장」."""

    def test_worker_exception_becomes_fail_and_next_job_runs(self):
        """detector 예외는 그 작업만 INFERENCE_ERROR 실패이고 다음 작업이 이어지며, 끝까지 처리한 뒤 실패 누적으로 종료 1이다."""
        det = ScriptedDetector({1: RuntimeError("boom")})
        res, stream, err = self.run_with(frames_factory(3), detector=det)

        self.assertEqual((res.exit_code, res.stop_reason, det.calls), (1, "end_of_stream", [0, 1, 2]))
        self.assertIn("completed with 1 failed job(s)", err.getvalue())
        c = res.counters
        self.assertEqual((c.completed, c.failed), (2, 1))
        ev = [r for r in lines(stream) if r["kind"] == "event"]
        self.assertEqual([(e["invalidation_kind"], e["job_seq"]) for e in ev], [("INFERENCE_ERROR", 1)])
        self.assertIn("RuntimeError: boom", ev[0]["cause"])
        self.assert_balance(res)

    def test_all_jobs_failing_still_processes_everything_and_exits_1(self):
        """모든 작업이 실패해도 입력 끝까지 처리하고(실패마다 기록) 종료 1이며 보존식이 성립한다."""
        det = ScriptedDetector({i: RuntimeError("down") for i in range(4)})
        res, stream, err = self.run_with(frames_factory(4), detector=det)

        self.assertEqual((res.exit_code, res.stop_reason, res.counters.failed), (1, "end_of_stream", 4))
        self.assertEqual(det.calls, [0, 1, 2, 3])
        self.assertEqual(kinds(stream).count("process"), 4)
        self.assertIn("completed with 4 failed job(s)", err.getvalue())
        self.assert_balance(res)

    def test_clean_run_exits_0(self):
        """실패·입력 오류가 없으면 종료 0이고 오류 보고가 없다."""
        res, _, err = self.run_with(frames_factory(3))

        self.assertEqual((res.exit_code, res.stop_reason), (0, "end_of_stream"))
        self.assertEqual(err.getvalue(), "")

    def test_cancel_aborts_and_late_result_is_duplicate(self):
        """detector가 JobCancelled를 올리면 JOB_ABORTED로 정리되고 실행은 계속되며 의도한 취소라 종료 0이다. 중단 뒤 늦은 완료는 duplicate라 소비·기록되지 않는다."""
        res, stream, _ = self.run_with(frames_factory(2), detector=ScriptedDetector({0: JobCancelled("user")}))

        self.assertEqual(res.exit_code, 0)
        self.assertEqual((res.counters.aborted, res.counters.completed), (1, 1))
        self.assertEqual([r["status"] for r in lines(stream) if r["kind"] == "process"], ["ABORTED", "COMPLETED"])

        # 늦은 결과(코어 직접): 중단 뒤 complete는 duplicate, b_gate 거부
        p = VisionPipeline(settings(), run_id=RUN_ID, clock_ns=FakeClock())
        p.offer_frame(frame(0))
        j = p.start_next()
        p.fail(j, InvalidationKind.JOB_ABORTED, "cancelled")
        late = p.complete(j, valid_observation(j))
        self.assertTrue(late.duplicate)
        self.assertFalse(b_gate(p, late))

    def test_keyboard_interrupt_aborts_job_and_closes_once(self):
        """detector 안의 KeyboardInterrupt는 그 작업을 JOB_ABORTED로 정리한 뒤 실행을 끝낸다: 종료 1, close·소스 close 각 1회, 사건 1줄."""
        holder = {}
        res, stream, err = self.run_with(frames_factory(3, holder), detector=ScriptedDetector({1: KeyboardInterrupt()}))

        self.assertEqual((res.exit_code, res.stop_reason, res.closed), (1, "interrupted", True))
        self.assertEqual(holder["source"].closes, 1)
        self.assertEqual(res.counters.aborted, 1)
        self.assertEqual([(r["invalidation_kind"], r["job_seq"], r["cause"]) for r in lines(stream)
                          if r["kind"] == "event"], [("JOB_ABORTED", 1, "interrupted")])
        self.assertIn("interrupted", err.getvalue())
        self.assert_balance(res)

    def test_unhandled_base_exception_leaves_active_job_for_close(self):
        """detector가 잡지 않는 BaseException(SystemExit)을 내면 활성 작업은 close가 JOB_ABORTED로 한 번 정리하고 그 결과·사건을 1회 기록한다."""
        res, stream, _ = self.run_with(frames_factory(2), detector=ScriptedDetector({0: SystemExit(3)}))

        self.assertEqual((res.exit_code, res.closed), (1, True))
        recs = lines(stream)
        self.assertEqual([r["status"] for r in recs if r["kind"] == "process"], ["ABORTED"])
        self.assertEqual([(r["invalidation_kind"], r["cause"]) for r in recs if r["kind"] == "event"],
                         [("JOB_ABORTED", "closed")])
        self.assert_balance(res)

    def test_no_overlapping_workers(self):
        """detector는 동시에 두 번 불리지 않고 작업 수만큼 순서대로 불린다(동기 호출, 이전 호출이 반환한 뒤 다음 시작)."""
        det = ScriptedDetector({})
        res, _, _ = self.run_with(frames_factory(5), detector=det)

        self.assertEqual((det.max_active, det.calls), (1, [0, 1, 2, 3, 4]))
        self.assertEqual(threading.active_count(), 1)
        self.assertEqual(res.counters.first_started, 5)


class CloseRetryTest(Base):
    """close 유한 재시도·최종 실패 보고(계획 r4 P2)."""

    class ArmedClock:
        """무장 뒤 지정 횟수만큼 실패하는 시계(None이면 계속 실패)."""

        def __init__(self):
            """정상 시계로 시작한다."""
            self.base, self.fail_left, self.fail_calls = FakeClock(), 0, 0

        def __call__(self):
            """무장 상태면 실패한다."""
            if self.fail_left is None or self.fail_left > 0:
                self.fail_calls += 1
                if self.fail_left is not None:
                    self.fail_left -= 1
                raise RuntimeError("clock unavailable")
            return self.base()

    def stop_and_arm(self, clock, fails):
        """한 번 처리한 뒤 정지 요청과 함께 시계를 무장하는 should_stop."""
        state = {"n": 0}

        def should_stop():
            """두 번째 확인에서 무장·정지."""
            state["n"] += 1
            if state["n"] == 2:
                clock.fail_left = fails
                return True
            return False

        return should_stop

    def test_close_retry_succeeds_after_transient_clock_error(self):
        """close 첫 시도만 시계 오류이면 재시도로 닫히고 종료 0이며, 일시 오류는 stderr에 남는다."""
        clock = self.ArmedClock()
        holder = {}
        res, _, err = self.run_with(frames_factory(3, holder), clock_ns=clock,
                                    should_stop=self.stop_and_arm(clock, 1))

        self.assertEqual((res.exit_code, res.stop_reason, res.closed), (0, "stop_requested", True))
        self.assertIn("close attempt 1/3 failed", err.getvalue())
        self.assertEqual(holder["source"].closes, 1)

    def test_close_retry_hands_over_abort_once_with_active_job(self):
        """활성 작업이 남은 채(SystemExit) close 첫 시계 조회만 실패하면 재시도가 JOB_ABORTED로 한 번 정리하고 결과·사건 1회, 원래 실패 코드 1."""
        clock = self.ArmedClock()
        holder = {}
        stream = io.StringIO()
        log = PipelineLog(JsonlRecorder(stream))
        err = io.StringIO()

        def detector(job):
            """시계를 1회 실패로 무장하고 잡지 않는 BaseException을 낸다(활성 작업 남김)."""
            clock.fail_left = 1
            raise SystemExit(3)

        res = run(frames_factory(2, holder), settings(), log, run_id=RUN_ID, detector=detector, config=FAST,
                  stderr=err, clock_ns=clock)

        self.assertEqual((res.exit_code, res.closed), (1, True))
        self.assertEqual(holder["source"].closes, 1)
        self.assertIn("close attempt 1/3 failed", err.getvalue())
        self.assertIn("SystemExit", err.getvalue())
        c = res.counters
        self.assertEqual((c.aborted, c.failed, c.completed), (1, 0, 0))
        recs = lines(stream)
        self.assertEqual([r["status"] for r in recs if r["kind"] == "process"], ["ABORTED"])
        self.assertEqual([(r["invalidation_kind"], r["job_seq"]) for r in recs if r["kind"] == "event"],
                         [("JOB_ABORTED", 0)])
        self.assertEqual(res.missing, {"process": 0, "event": 0, "queue_drop": 0})
        self.assert_balance(res)

    def test_close_gives_up_after_max_attempts(self):
        """시계가 계속 실패하면 정해진 시도 수(3) 안에 끝나고 종료 1, stderr에 원인·남은 상태가 남으며 소스 close는 실행된다."""
        clock = self.ArmedClock()
        holder = {}
        res, _, err = self.run_with(frames_factory(3, holder), clock_ns=clock,
                                    should_stop=self.stop_and_arm(clock, None))

        self.assertEqual((res.exit_code, res.closed), (1, False))
        text = err.getvalue()
        self.assertIn("close failed after 3 attempts", text)
        self.assertIn("remaining state: closed=False", text)
        self.assertEqual(holder["source"].closes, 1)
        self.assertGreaterEqual(clock.fail_calls, 3)

    def test_close_failure_reported_even_if_recorder_broken(self):
        """기록기 고장과 close 실패가 겹쳐도 stderr에 기록 오류·close 오류·미확인 기록 수가 모두 남는다."""

        class BreakOnSecondWrite(io.StringIO):
            """두 번째 write부터 실패."""

            n = 0

            def write(self, t):
                """두 번째부터 OSError."""
                self.n += 1
                if self.n >= 2:
                    raise OSError("disk full")
                return super().write(t)

        clock = self.ArmedClock()
        state = {"n": 0}

        def detector(job):
            """처리 중 시계를 무장(close가 계속 실패하도록)."""
            clock.fail_left = None
            return observation(job), None

        res, _, err = self.run_with(frames_factory(2), detector=detector, clock_ns=clock,
                                    stream=BreakOnSecondWrite())
        text = err.getvalue()

        self.assertEqual(res.exit_code, 1)
        self.assertIn("log write failed", text)
        self.assertIn("close failed after 3 attempts", text)
        self.assertIn("unrecorded", text)
        self.assertIsNotNone(res.missing)
        del state


class LogAndConsumerFailureTest(Base):
    """기록·소비자 실패의 정리와 종료 코드."""

    def test_log_failure_stops_recording_and_reports_missing(self):
        """세 번째 기록에서 쓰기가 실패하면 처리를 멈추고 더 쓰지 않으며, 코어·소스 정리를 마친 뒤 종료 1과 미확인 기록 수를 보고한다."""

        class FailThird(io.StringIO):
            """세 번째 write만 실패."""

            n = 0

            def write(self, t):
                """세 번째에서 OSError."""
                self.n += 1
                if self.n == 3:
                    raise OSError("disk full")
                return super().write(t)

        holder = {}
        stream = FailThird()
        res, _, err = self.run_with(frames_factory(5, holder), stream=stream)

        self.assertEqual((res.exit_code, res.stop_reason, res.closed), (1, "log_failure", True))
        self.assertEqual(holder["source"].closes, 1)
        self.assertEqual(stream.getvalue().count("\n"), 2)        # 실패 뒤 추가 기록 없음
        self.assertIn("unrecorded", err.getvalue())
        self.assertGreaterEqual(res.missing["process"], 1)
        self.assertLess(res.counters.first_started, 5)            # 처리를 멈춤
        self.assert_balance(res)

    def test_consumer_failure_closes_source(self):
        """B 지점이 예외를 내면 실행을 멈추고 코어·소스를 정리한 뒤 종료 1이다."""
        holder = {}

        def boom(f, o):
            """소비자 실패."""
            raise ValueError("consumer down")

        res, _, err = self.run_with(frames_factory(3, holder), detector=ScriptedDetector({0: "valid"}), b_consumer=boom)

        self.assertEqual((res.exit_code, res.stop_reason, res.closed), (1, "error", True))
        self.assertEqual(holder["source"].closes, 1)
        self.assertIn("consumer down", err.getvalue())
        self.assert_balance(res)


class InputPathTest(Base):
    """입력 사건·EOF·입력 없음·연속 입력 오류·탐색 실패."""

    def test_input_status_clears_current_without_allocating_ids(self):
        """입력 사건은 현재 관측·선택을 비우고 대기를 input_status로 폐기하며 세대·image_seq·job_seq를 만들지 않는다."""
        p = VisionPipeline(settings(), run_id=RUN_ID, clock_ns=FakeClock())
        p.offer_frame(frame(0))
        j = p.start_next()
        p.complete(j, valid_observation(j))
        p.offer_frame(frame(1))
        gen, next_seq = p.current_generation(), p._next_job_seq

        p.note_input_status(event(InputStatus.DECODE_FAILED))

        self.assertIsNone(p.current_observation)
        self.assertIsNone(p.selected_frame)
        self.assertIsNone(p.pending_frame)
        self.assertEqual(p.counters.queue_dropped, {QueueDropReason.INPUT_STATUS.value: 1})
        self.assertEqual((p.current_generation(), p._next_job_seq, p.invalidation_events), (gen, next_seq, ()))
        self.assertIsNone(p.start_reprocess())

        for bad in (dataclasses.replace(event(InputStatus.FRAME_TIMEOUT), source_id="other"), "not an event"):
            with self.assertRaises(PipelineContractError):
                p.note_input_status(bad)

    def test_decode_error_is_recorded_and_run_continues(self):
        """중간 입력 오류는 입력 상태로 기록하고 다음 프레임을 계속 처리하며, 입력 오류 누적으로 종료 1이다."""
        def factory(issuer, provider):
            """프레임·오류·프레임."""
            return FakeSource([frame(issuer.next_seq(), 0), event(InputStatus.DECODE_FAILED, "bad"),
                               frame(issuer.next_seq(), 0)])

        res, stream, _ = self.run_with(factory)

        self.assertEqual((res.exit_code, res.stop_reason, res.frames, res.input_errors), (1, "end_of_stream", 2, 1))
        self.assertEqual([r["status"] for r in lines(stream) if r["kind"] == "input_state"],
                         ["DECODE_FAILED", "END_OF_STREAM"])
        self.assert_balance(res)

    def test_input_errors_between_frames_do_not_hit_limit(self):
        """프레임이 사이사이 오면 연속 오류 수가 0으로 돌아가 상한(3)보다 많은 오류가 있어도 끝까지 처리한다(누적 오류로 종료 1)."""
        def factory(issuer, provider):
            """오류·프레임을 네 번 번갈아."""
            items = []
            for _ in range(4):
                items += [event(InputStatus.DECODE_FAILED, "bad"), frame(issuer.next_seq(), 0)]
            return FakeSource(items)

        res, _, _ = self.run_with(factory)

        self.assertEqual((res.exit_code, res.stop_reason, res.input_errors, res.frames), (1, "end_of_stream", 4, 4))

    def test_repeating_input_error_stops_at_limit(self):
        """진행 없이 같은 입력 오류가 반복되면(영상 열기 실패 등) 연속 상한(3)에서 멈추고 종료 1이다."""
        res, _, err = self.run_with(lambda i, g: FakeSource([event(InputStatus.DECODE_FAILED, "cannot open")],
                                                            eos=False))

        self.assertEqual((res.exit_code, res.stop_reason, res.input_errors), (1, "input_error_limit", 3))
        self.assertIn("consecutive input errors", err.getvalue())

    def test_empty_input_is_failure(self):
        """빈 목록(입력 없음)은 종료 1이다."""
        res, _, _ = self.run_with(lambda i, g: FakeSource([event(InputStatus.SOURCE_UNAVAILABLE, "empty list")]))

        self.assertEqual((res.exit_code, res.stop_reason, res.frames), (1, "no_input", 0))

    def test_max_frames_stops_normally(self):
        """max_frames에 닿으면 정상 종료 0이다."""
        res, _, _ = self.run_with(frames_factory(5), config=dataclasses.replace(FAST, max_frames=2))

        self.assertEqual((res.exit_code, res.stop_reason, res.frames), (0, "max_frames", 2))

    def test_seek_failure_preserves_current_selection(self):
        """실제 VideoSource에서 실패한 탐색은 위치·세대·선택을 보존하고, 성공한 탐색은 세대를 올리고 다음 프레임이 새 capture_generation을 받는다."""
        with tempfile.TemporaryDirectory() as d:
            path = write_video(os.path.join(d, "clip.avi"), 4, fps=10.0)
            p = VisionPipeline(settings(source_id="src0", profile=None), run_id=RUN_ID, clock_ns=FakeClock())
            src = VideoSource(path, source_id="src0", issuer=FrameIdIssuer(RUN_ID),
                              generation_provider=p.current_generation)
            f0 = src.read()
            p.offer_frame(f0)
            j = p.start_next()
            p.complete(j, observation(j))
            selected, index = p.selected_frame, src.next_frame_index

            self.assertFalse(apply_seek(p, src, 999))
            self.assertEqual((p.current_generation(), src.next_frame_index), (0, index))
            self.assertIs(p.selected_frame, selected)

            self.assertTrue(apply_seek(p, src, 0))
            self.assertEqual(p.current_generation(), 1)
            self.assertIsNone(p.selected_frame)
            f1 = src.read()
            self.assertEqual((f1.capture_generation, f1.source_index), (1, 0))
            src.close()


class ConfigAndCliTest(unittest.TestCase):
    """실행기 설정 검증과 CLI."""

    def test_runner_config_validation(self):
        """시도 수·연속 오류 상한은 1 이상 정수, 간격은 유한한 0 이상, max_frames는 None 또는 1 이상이다."""
        for bad in ({"close_attempts": 0}, {"close_attempts": True}, {"close_retry_interval_s": -1},
                    {"close_retry_interval_s": float("inf")}, {"max_consecutive_input_errors": 0},
                    {"max_frames": 0}):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    RunnerConfig(**bad)

    def load_cli(self):
        """scripts/run_pipeline.py를 모듈로 불러온다."""
        spec = importlib.util.spec_from_file_location("run_pipeline_under_test", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_cli_runs_images_and_reports_errors(self):
        """CLI는 실제 이미지로 JSONL을 쓰고 종료 0이며, 설정 오류는 종료 2, 인자 오류는 argparse 종료 2다."""
        mod = self.load_cli()
        with tempfile.TemporaryDirectory() as d:
            img = os.path.join(d, "a.png")
            cv2.imwrite(img, scene(CENTER))
            cfg = os.path.join(d, "cfg.json")
            with open(cfg, "w") as f:
                json.dump({k: getattr(config(), k) for k in (
                    "format_version", "status", "roi", "resize_wh", "resize_interpolation", "gaussian_blur_ksize",
                    "polarity", "threshold_method", "fixed_threshold", "morph_open_ksize", "contour_retrieval",
                    "contour_approximation", "min_area_source_px2", "max_area_source_px2",
                    "border_margin_processed_px")}, f)
            out = os.path.join(d, "run.jsonl")
            err = io.StringIO()

            self.assertEqual(mod.main(["--images", img, "--detect-config", cfg, "--log", out], stderr=err), 0,
                             err.getvalue())
            with open(out) as f:
                recs = [json.loads(t) for t in f]
            self.assertEqual([r["kind"] for r in recs][:2], ["run_meta", "process"])
            self.assertIn("opencv", recs[0]["versions"])

            self.assertEqual(mod.main(["--images", img, "--detect-config", os.path.join(d, "none.json"),
                                       "--log", out], stderr=io.StringIO()), 2)

            with self.assertRaises(SystemExit) as ctx:
                mod.main(["--images", img, "--video", img, "--detect-config", cfg, "--log", out])
            self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
