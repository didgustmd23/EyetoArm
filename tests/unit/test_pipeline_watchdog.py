"""검출 감시 시험: 검출 전용 worker 스레드 1개 + 제어 흐름 감시, 첫 검출/일반 상한, 멈춤 처리.

멈춤이 없는 시험은 데몬 스레드에서 돌리고 유한 join 뒤 생존을 단언한다. 10 s·1 s 기본값은
주입 시계(`watch_clock`)로 확인한다. 격리 하네스 `run_isolated`는 시험 전용이다.
"""
import gc
import importlib.util
import io
import json
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import weakref
from unittest import mock

from eyetoarm.vision.pipeline import VisionPipeline
from eyetoarm.vision.pipeline_records import JsonlRecorder, PipelineLog
from eyetoarm.vision import pipeline_runner as pr
from eyetoarm.vision.pipeline_runner import CancelToken, JobCancelled, RunnerConfig, run
from test_pipeline import RUN_ID, FakeClock, frame, observation, settings
from test_pipeline_terminal import valid_observation
from test_run_pipeline import SCRIPT, FakeSource, frames_factory, kinds, lines

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UNIT = os.path.dirname(os.path.abspath(__file__))
WORKER_NAME = "eyetoarm-detector"
OFF = RunnerConfig(first_detect_timeout_s=None, detect_timeout_s=None)


def watch(first, normal, grace=0.2, cleanup=0.3, interval=0.02, **over):
    """감시 켠 시험 설정(close 재시도 간격 0, 연속 입력 오류 3)."""
    return RunnerConfig(close_attempts=3, close_retry_interval_s=0.0, max_consecutive_input_errors=3,
                        first_detect_timeout_s=first, detect_timeout_s=normal, cancel_grace_s=grace,
                        worker_cleanup_limit_s=cleanup, stop_check_interval_s=interval, **over)


def detector_threads():
    """살아 있는 검출 worker 스레드 목록."""
    return [t for t in threading.enumerate() if t.name == WORKER_NAME]


class WatchClock:
    """시험이 직접 움직이는 감시용 시계(초). worker 스레드의 detector가 앞으로 돌린다."""

    def __init__(self):
        """0초에서 시작한다."""
        self.now = 0.0

    def __call__(self):
        """현재 값."""
        return self.now

    def advance(self, seconds):
        """시계를 앞으로 돌린다."""
        self.now += seconds


class PlanDetector:
    """작업 순번별 동작을 정한 시험용 detector. 호출 시각·스레드·동시 실행 수를 적는다.

    plan[job_seq]: advance → sleep → wait_token → raise → result 순. 없으면 즉시 무효 관측.
    """

    def __init__(self, plan=None, clock=None):
        """동작 표와 감시 시계를 받는다."""
        self.plan, self.clock = plan or {}, clock
        self.calls, self.threads, self.names, self.spans = [], set(), set(), {}
        self.active, self.max_active, self.threads_seen = 0, 0, 0
        self.lock = threading.Lock()

    def __call__(self, job, token):
        """정한 동작을 한다."""
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        start = time.monotonic()
        self.calls.append(job.job_seq)
        self.threads.add(threading.get_ident())
        self.names.add(threading.current_thread().name)
        self.threads_seen = max(self.threads_seen, threading.active_count())
        step = self.plan.get(job.job_seq, {})

        try:
            if "advance" in step:
                self.clock.advance(step["advance"])
            if "sleep" in step:
                time.sleep(step["sleep"])
            if "wait_token" in step and token.wait(step["wait_token"]):
                raise JobCancelled("token")
            if "raise" in step:
                raise step["raise"]
            return (valid_observation(job) if step.get("result") == "valid" else observation(job)), None
        finally:
            self.spans[job.job_seq] = (start, time.monotonic())
            with self.lock:
                self.active -= 1


def run_bounded(factory, detector, cfg, *, limit_s=5.0, stream=None, **kw):
    """실행기를 데몬 스레드에서 돌리고 유한 join 한다. (결과, 스트림, stderr, 제어 흐름 스레드 id)."""
    stream = stream if stream is not None else io.StringIO()
    err = io.StringIO()
    box = {}

    def target():
        """실행기 한 번."""
        box["ident"] = threading.get_ident()
        box["res"] = run(factory, settings(), PipelineLog(JsonlRecorder(stream)), run_id=RUN_ID, detector=detector,
                         detector_takes_token=True, config=cfg, stderr=err, clock_ns=FakeClock(), **kw)

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(limit_s)

    if t.is_alive():
        raise AssertionError(f"runner did not finish within {limit_s} s; stderr={err.getvalue()!r}")

    return box["res"], stream, err, box["ident"]


def _text(value):
    """TimeoutExpired 부분 출력(POSIX에서는 text=True여도 bytes)을 str로 바꾼다."""
    if value is None:
        return ""

    return value.decode("utf-8", "replace") if isinstance(value, bytes) else value


HANG_CODE = "import sys, time; print('started', flush=True); time.sleep(60)"
# 별도 세션 손자가 stdout·stderr 파이프를 물려받아 쥔다(자식 그룹 kill 뒤에도 파이프가 닫히지 않음)
ORPHAN_CODE = ("import subprocess, sys, time\n"
               "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)\n"
               "open(sys.argv[1], 'w').write(str(c.pid))\n"
               "print('started', flush=True)\n"
               "time.sleep(60)\n")


def run_isolated(argv, deadline_s=10.0, reap_s=2.0, script=None):
    """자식 프로세스 `python [script] argv...`를 실행한다. 부모가 Popen 핸들·기한을 소유한다.

    기한 초과면 그룹 SIGKILL → 상한 있는 communicate 한 번 → 그래도 넘기면 poll·파이프 닫기.
    """
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([os.path.join(ROOT, "src"), UNIT])}
    t0 = time.monotonic()
    p = subprocess.Popen([sys.executable, *([script] if script else []), *argv], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, start_new_session=True, env=env, cwd=ROOT)
    info = {"pid": p.pid, "timed_out": False, "kill": None, "reaped": None, "second_timeout": False}

    try:
        out, err = p.communicate(timeout=deadline_s)
        info["reaped"] = True
    except subprocess.TimeoutExpired as exc:
        info["timed_out"] = True
        out, err = _text(exc.stdout), _text(exc.stderr)

        try:
            os.killpg(p.pid, signal.SIGKILL)
            info["kill"] = "killpg"
        except OSError as kill_error:
            try:
                p.kill()
                info["kill"] = f"kill ({type(kill_error).__name__})"
            except OSError as second:
                info["kill"] = f"failed ({type(second).__name__})"

        try:
            out, err = p.communicate(timeout=reap_s)
            info["reaped"] = p.returncode is not None
        except subprocess.TimeoutExpired as exc2:
            info["second_timeout"] = True
            out, err = _text(exc2.stdout) or out, _text(exc2.stderr) or err
            info["reaped"] = p.poll() is not None

            for pipe in (p.stdout, p.stderr):
                try:
                    pipe.close()
                except OSError:
                    pass

    info.update(returncode=p.returncode, stdout=out or "", stderr=err or "", elapsed_s=time.monotonic() - t0)

    return info


class IsolationHarnessTest(unittest.TestCase):
    """시험 격리 하네스 자체: 기한 초과·회수 실패 분기에서도 부모가 무한 대기하지 않는다."""

    def test_harness_kills_hung_child_within_bound(self):
        """끝나지 않는 자식은 기한(1 s) 뒤 그룹 kill·회수되고 부분 출력이 한 번만 남는다."""
        info = run_isolated(["-c", HANG_CODE], deadline_s=1.0, reap_s=2.0)

        self.assertTrue(info["timed_out"])
        self.assertEqual((info["kill"], info["reaped"], info["second_timeout"]), ("killpg", True, False))
        self.assertEqual(info["returncode"], -signal.SIGKILL)
        self.assertEqual(info["stdout"].count("started"), 1)
        self.assertLess(info["elapsed_s"], 5.0)

    def test_harness_bounded_when_reap_times_out(self):
        """손자가 파이프를 쥐어 회수가 다시 기한을 넘기면 poll·파이프 닫기 후 끝난다."""
        with tempfile.TemporaryDirectory() as d:
            pid_path = os.path.join(d, "orphan.pid")
            info = run_isolated(["-c", ORPHAN_CODE, pid_path], deadline_s=2.0, reap_s=0.5)

            if os.path.exists(pid_path):
                with open(pid_path) as f:
                    orphan = int(f.read())
                try:
                    os.kill(orphan, signal.SIGKILL)
                except ProcessLookupError:
                    pass

        self.assertTrue(info["timed_out"])
        self.assertTrue(info["second_timeout"])
        self.assertEqual(info["kill"], "killpg")
        self.assertIn("started", info["stdout"])
        self.assertLess(info["elapsed_s"], 5.0)


class WatchdogRunTest(unittest.TestCase):
    """감시 켠 실행 경로(같은 프로세스, 유한 join)."""

    def assert_balance(self, res):
        """종료 후 보존식."""
        c = res.counters
        self.assertEqual(c.received, c.first_started + sum(c.queue_dropped.values()))
        self.assertEqual(c.first_started + c.reprocess_started,
                         c.completed + c.generation_discarded + c.failed + c.aborted)

    def test_default_first_10s_then_1s_with_injected_clock(self):
        """첫 작업은 5 s에도 완료, 둘째는 5 s에 일반 상한으로 중단, 셋째는 정상 완료된다."""
        clock = WatchClock()
        det = PlanDetector({0: {"advance": 5.0}, 1: {"advance": 5.0, "wait_token": 2.0}}, clock)
        res, stream, err, _ = run_bounded(frames_factory(3), det, RunnerConfig(), watch_clock=clock)

        self.assertEqual((RunnerConfig().first_detect_timeout_s, RunnerConfig().detect_timeout_s), (10.0, 1.0))
        self.assertEqual([r["status"] for r in lines(stream) if r["kind"] == "process"],
                         ["COMPLETED", "ABORTED", "COMPLETED"])
        self.assertEqual([(r["invalidation_kind"], r["job_seq"], r["cause"]) for r in lines(stream)
                          if r["kind"] == "event"], [("JOB_ABORTED", 1, "detect_timeout")])
        self.assertEqual((res.timeouts, res.late_results, res.exit_code, det.calls), (1, 1, 1, [0, 1, 2]))
        self.assertIn("detect timeouts: 1", err.getvalue())
        self.assertEqual(detector_threads(), [])
        self.assertFalse(res.stalled)
        self.assert_balance(res)

    def test_first_job_timeout_then_normal_limit_still_applies(self):
        """첫 작업이 첫 상한(10 s)을 넘겨 중단돼도 다음 작업은 다시 10 s가 아니라 일반 상한(1 s)을 받는다."""
        clock = WatchClock()
        det = PlanDetector({0: {"advance": 11.0, "wait_token": 2.0}, 1: {"advance": 5.0, "wait_token": 2.0}}, clock)
        res, stream, _, _ = run_bounded(frames_factory(2), det, RunnerConfig(), watch_clock=clock)

        self.assertEqual([(r["job_seq"], r["cause"]) for r in lines(stream) if r["kind"] == "event"],
                         [(0, "detect_timeout"), (1, "detect_timeout")])
        self.assertEqual((res.timeouts, res.exit_code), (2, 1))

    def test_normal_and_exception_record_once(self):
        """감시 켠 경로에서 정상 2건·예외 1건은 completed 2·failed 1로 한 번씩 남고 종료 1이다."""
        det = PlanDetector({1: {"raise": RuntimeError("boom")}})
        res, stream, err, _ = run_bounded(frames_factory(3), det, watch(0.5, 0.5))

        c = res.counters
        self.assertEqual((c.completed, c.failed, res.timeouts, res.exit_code), (2, 1, 0, 1))
        self.assertEqual(kinds(stream).count("process"), 3)
        self.assertEqual([(r["invalidation_kind"], r["job_seq"]) for r in lines(stream) if r["kind"] == "event"],
                         [("INFERENCE_ERROR", 1)])
        self.assertIn("completed with 1 failed job(s)", err.getvalue())
        self.assert_balance(res)

    def test_watchdog_runs_detector_in_one_worker_without_overlap(self):
        """감시 켠 경로에서 detector는 제어 흐름이 아닌 worker 스레드 하나에서만 불리고 동시에 두 번 불리지 않는다."""
        det = PlanDetector()
        res, _, _, control = run_bounded(frames_factory(5), det, watch(0.5, 0.5))

        self.assertEqual((det.max_active, det.calls, res.exit_code), (1, [0, 1, 2, 3, 4], 0))
        self.assertEqual(len(det.threads), 1)
        self.assertNotIn(control, det.threads)
        self.assertEqual(detector_threads(), [])

    def test_watchdog_disabled_keeps_synchronous_path(self):
        """감시 끔(둘 다 None)이면 6a처럼 제어 흐름에서 동기 호출하고 worker 스레드를 만들지 않는다."""
        det = PlanDetector()
        res, _, _, control = run_bounded(frames_factory(3), det, OFF)

        self.assertEqual(det.threads, {control})
        self.assertEqual((res.exit_code, res.timeouts, res.worker), (0, 0, None))
        self.assertNotIn(WORKER_NAME, det.names)

    def test_timeout_cooperative_detector_stops_then_next_job(self):
        """협조적 detector가 시간 초과 뒤 취소 표지를 보고 끝나면, 다음 작업은 그 반환 뒤에만 시작하고 정상 완료된다."""
        det = PlanDetector({0: {"wait_token": 2.0}})
        res, stream, _, _ = run_bounded(frames_factory(2), det, watch(0.05, 0.05, grace=1.0))

        self.assertEqual([r["status"] for r in lines(stream) if r["kind"] == "process"], ["ABORTED", "COMPLETED"])
        self.assertGreaterEqual(det.spans[1][0], det.spans[0][1])
        self.assertEqual((res.timeouts, res.late_results, det.max_active, res.exit_code), (1, 1, 1, 1))
        self.assert_balance(res)

    def test_late_completion_after_abort_is_duplicate(self):
        """시간 초과 뒤 유예 안에 돌아온 유효 결과는 duplicate로 버려져 B 지점 0회, 처리·사건 기록이 늘지 않는다."""
        calls = []
        det = PlanDetector({0: {"sleep": 0.3, "result": "valid"}})
        res, stream, _, _ = run_bounded(frames_factory(1), det, watch(0.05, 0.05, grace=1.0),
                                        b_consumer=lambda f, o: calls.append(f.image_seq))

        self.assertEqual((calls, res.b_calls, res.late_results), ([], 0, 1))
        self.assertEqual(kinds(stream).count("process"), 1)
        self.assertEqual(kinds(stream).count("event"), 1)
        self.assertEqual(res.counters.completed, 0)

    def test_repeated_timeouts_do_not_accumulate(self):
        """협조적 시간 초과 20회가 이어져도 스레드는 늘지 않고(실행기+worker), 끝난 뒤 모든 프레임 영상이 회수되며 5 s 안에 끝난다."""
        refs = []

        def factory(issuer, provider):
            """프레임 20개(영상 약한 참조 보관)."""
            items = []
            for _ in range(20):
                f = frame(issuer.next_seq(), provider())
                refs.append(weakref.ref(f.image))
                items.append(f)
            return FakeSource(items)

        det = PlanDetector({i: {"wait_token": 2.0} for i in range(20)})
        before = threading.active_count()
        res, _, _, _ = run_bounded(factory, det, watch(0.01, 0.01, grace=1.0, interval=0.005))
        threads_seen = det.threads_seen
        del det
        gc.collect()

        self.assertEqual((res.timeouts, res.late_results), (20, 20))
        self.assertEqual([r for r in refs if r() is not None], [])
        self.assertLessEqual(threads_seen, before + 2)          # 실행 중: 시험 실행기 스레드 + worker 1개
        self.assertEqual(threading.active_count(), before)

    def test_idle_worker_stops_cleanly_and_is_not_stall(self):
        """정상 2건 뒤 close에서 IDLE worker는 종료 표지로 끝나고(join 성공) STALLED가 아니며 종료 0이다."""
        res, _, err, _ = run_bounded(frames_factory(2), PlanDetector(), watch(0.5, 0.5))

        self.assertEqual((res.exit_code, res.stalled), (0, False), err.getvalue())
        self.assertEqual(res.worker, {"stalled": False, "stalled_job_seq": None, "thread_stopped": True,
                                      "discarded": 0})
        self.assertEqual(detector_threads(), [])

    def test_late_handoff_after_close_is_discarded(self):
        """close가 작업을 정리한 뒤 돌아온 결과는 worker 정리 단계에서 버려진다(버린 수 1)."""
        state = {"n": 0}
        calls = []

        def should_stop():
            """두 번째 확인(검출 대기 중)에서 오류."""
            state["n"] += 1
            if state["n"] == 2:
                raise RuntimeError("stop check failed")
            return False

        det = PlanDetector({0: {"wait_token": 2.0, "result": "valid"}})
        res, stream, err, _ = run_bounded(frames_factory(2), det, watch(5.0, 5.0), should_stop=should_stop,
                                          b_consumer=lambda f, o: calls.append(1))

        self.assertEqual((res.exit_code, res.stop_reason), (1, "error"))
        self.assertIn("stop check failed", err.getvalue())
        self.assertEqual([(r["invalidation_kind"], r["cause"]) for r in lines(stream) if r["kind"] == "event"],
                         [("JOB_ABORTED", "closed")])
        self.assertEqual((res.worker["discarded"], res.worker["stalled"], calls), (1, False, []))
        self.assertEqual(detector_threads(), [])
        self.assert_balance(res)


class WaitLoopOrderTest(unittest.TestCase):
    """단일 판정 순서(결과 → 정지 → 기한)를 대기 함수로 결정적으로 확인."""

    def test_result_already_handed_off_wins_over_stop_and_deadline(self):
        """기한이 지났고 정지 요청도 참이어도, 그 회차에 결과가 칸에 있으면 결과를 그대로 돌려준다."""
        item = ("job", ("obs", None), None)
        what = pr._wait_for_detection(lambda timeout: item, lambda: True, lambda: 100.0, 1.0, 0.02)

        self.assertEqual(what, ("result", item))

    def test_empty_then_stop_before_deadline_and_timeout(self):
        """칸이 비면 정지 요청이 기한보다 먼저 판정되고, 정지가 없으면 기한 초과로 판정된다. 기한 전에는 계속 기다린다."""
        def empty(timeout):
            """항상 비어 있음."""
            raise queue.Empty

        self.assertEqual(pr._wait_for_detection(empty, lambda: True, lambda: 100.0, 1.0, 0.02), ("stop", None))
        self.assertEqual(pr._wait_for_detection(empty, None, lambda: 100.0, 1.0, 0.02), ("timeout", None))

        ticks = iter([0.0, 0.0, 0.5, 0.5, 2.0, 2.0])
        timeouts = []

        def empty_record(timeout):
            """요청된 대기 시간을 적고 비어 있음."""
            timeouts.append(timeout)
            raise queue.Empty

        self.assertEqual(pr._wait_for_detection(empty_record, lambda: False, lambda: next(ticks), 1.0, 0.02),
                         ("timeout", None))
        self.assertEqual(timeouts, [0.02, 0.02, 0.0])

    def test_cancel_token_is_read_only_for_detector(self):
        """취소 표지는 detector 쪽에서 읽기(cancelled·wait)만 공개하고, 실행기가 세우면 참이 된다."""
        token = CancelToken()

        self.assertFalse(token.cancelled)
        self.assertFalse(token.wait(0))
        self.assertFalse(hasattr(token, "set"))
        token._cancel()
        self.assertTrue(token.cancelled and token.wait(0))


class GatedQueue(queue.Queue):
    """get이 문(Event)이 열릴 때까지 막히는 큐. worker가 수신 칸에서 작업을 꺼내기 전 상태를 결정적으로 만든다."""

    def __init__(self, gate, maxsize=0):
        """문을 받는다."""
        super().__init__(maxsize)
        self.gate = gate

    def get(self, block=True, timeout=None):
        """문이 열린 뒤에만 꺼낸다(최대 5 s 대기, 시험 안전 상한)."""
        self.gate.wait(5.0)
        return super().get(block, timeout)


class WorkerShutdownTest(unittest.TestCase):
    """worker 정리 단계: 종료 의도가 수신 칸 Full과 무관하게 보존되는가."""

    def test_stop_intent_survives_full_inbox_before_worker_takes_job(self):
        """수신 칸의 작업이 정리 한도 뒤 남아도 worker는 detector를 부르지 않고 끝난다."""
        gate = threading.Event()
        made = []
        real_queue = queue.Queue

        def make_queue(maxsize=0):
            """첫 큐(수신 칸)만 문 달린 큐."""
            q = GatedQueue(gate, maxsize) if not made else real_queue(maxsize)
            made.append(q)
            return q

        calls = []

        def detector(job):
            """불리면 안 된다(호출 기록)."""
            calls.append(job.job_seq)
            return observation(job), None

        with mock.patch.object(pr.queue, "Queue", make_queue):
            worker = pr._DetectorWorker(detector, False)

        def release():
            """시험 실패 때도 worker를 남기지 않게 문을 열고 종료 표지를 넣는다."""
            gate.set()
            try:
                made[0].put_nowait(pr._STOP)
            except queue.Full:
                pass
            worker._thread.join(2.0)

        self.addCleanup(release)
        p = VisionPipeline(settings(), run_id=RUN_ID, clock_ns=FakeClock())
        p.offer_frame(frame(0))
        job = p.start_next()
        image_ref = weakref.ref(job.frame.image)
        worker.submit(job, CancelToken())

        report = worker.shutdown(0.05, 0.01)

        self.assertEqual((report["stalled"], report["stalled_job_seq"], report["thread_stopped"]), (True, 0, False))
        gate.set()
        worker._thread.join(2.0)
        self.assertFalse(worker._thread.is_alive(), "worker kept waiting after its stop intent was dropped")
        self.assertEqual(calls, [])
        self.assertTrue(worker._outbox.empty())

        p.close()
        del job, p
        worker.job = worker.token = None
        gc.collect()
        self.assertIsNone(image_ref())

    def test_stop_intent_ends_worker_after_stalled_detector_returns(self):
        """정리 한도 안에 반환하지 않던 detector가 나중에 반환하면 worker는 다음 작업을 기다리지 않고 끝난다(결과는 소비되지 않음)."""
        release_event = threading.Event()

        def detector(job):
            """해제될 때까지 반환하지 않는다(최대 5 s)."""
            release_event.wait(5.0)
            return observation(job), None

        worker = pr._DetectorWorker(detector, False)
        self.addCleanup(lambda: (release_event.set(), worker._thread.join(2.0)))
        p = VisionPipeline(settings(), run_id=RUN_ID, clock_ns=FakeClock())
        p.offer_frame(frame(0))
        worker.submit(p.start_next(), CancelToken())

        report = worker.shutdown(0.05, 0.01)

        self.assertTrue(report["stalled"])
        release_event.set()
        worker._thread.join(2.0)
        self.assertFalse(worker._thread.is_alive())
        self.assertTrue(worker._outbox.empty())


class WorkerRaisedBaseExceptionTest(unittest.TestCase):
    """worker 경로에서 detector가 올린 KeyboardInterrupt·SystemExit의 분류가 6a 동기 경로와 같은가."""

    def test_detector_keyboard_interrupt_in_worker(self):
        """detector의 KeyboardInterrupt는 JOB_ABORTED(interrupted) 1회, 실행 종료 1, worker도 끝."""
        holder = {}
        det = PlanDetector({1: {"raise": KeyboardInterrupt()}})
        res, stream, err, _ = run_bounded(frames_factory(3, holder), det, watch(0.5, 0.5))

        self.assertEqual((res.exit_code, res.stop_reason, res.closed, det.calls), (1, "interrupted", True, [0, 1]))
        self.assertEqual([(r["invalidation_kind"], r["job_seq"], r["cause"]) for r in lines(stream)
                          if r["kind"] == "event"], [("JOB_ABORTED", 1, "interrupted")])
        self.assertEqual(holder["source"].closes, 1)
        self.assertIn("interrupted", err.getvalue())
        self.assertEqual(detector_threads(), [])

    def test_detector_system_exit_in_worker(self):
        """detector의 SystemExit은 close가 JOB_ABORTED(closed)로 정리하고 종료 1, worker도 끝."""
        det = PlanDetector({0: {"raise": SystemExit(3)}})
        res, stream, _, _ = run_bounded(frames_factory(2), det, watch(0.5, 0.5))

        recs = lines(stream)
        self.assertEqual((res.exit_code, res.closed, det.calls), (1, True, [0]))
        self.assertEqual([r["status"] for r in recs if r["kind"] == "process"], ["ABORTED"])
        self.assertEqual([(r["invalidation_kind"], r["cause"]) for r in recs if r["kind"] == "event"],
                         [("JOB_ABORTED", "closed")])
        self.assertEqual(detector_threads(), [])


class WatchdogConfigTest(unittest.TestCase):
    """감시 설정 검증과 CLI 인자."""

    def test_defaults_are_development_values(self):
        """기본값: 첫 검출 10 s·이후 1 s, 유예 0.2 s·정리 한도 0.3 s·확인 간격 0.02 s."""
        c = RunnerConfig()

        self.assertEqual((c.first_detect_timeout_s, c.detect_timeout_s), (10.0, 1.0))
        self.assertEqual((c.cancel_grace_s, c.worker_cleanup_limit_s, c.stop_check_interval_s), (0.2, 0.3, 0.02))
        self.assertTrue(c.watchdog)
        self.assertFalse(OFF.watchdog)

    def test_watchdog_config_validation(self):
        """상한은 둘 다 None이거나 둘 다 유한 양수이고, 그 밖의 값은 ValueError다."""
        bads = []
        for name in ("first_detect_timeout_s", "detect_timeout_s", "cancel_grace_s", "worker_cleanup_limit_s",
                     "stop_check_interval_s"):
            for value in (0, 0.0, -1.0, float("inf"), float("nan"), True, "1"):
                bads.append({name: value})
        for name in ("cancel_grace_s", "worker_cleanup_limit_s", "stop_check_interval_s"):
            bads.append({name: None})
        bads += [{"first_detect_timeout_s": None}, {"detect_timeout_s": None}]

        for bad in bads:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    RunnerConfig(**bad)

        self.assertFalse(RunnerConfig(first_detect_timeout_s=None, detect_timeout_s=None).watchdog)
        self.assertEqual(RunnerConfig(first_detect_timeout_s=3, detect_timeout_s=0.5).first_detect_timeout_s, 3)

    def load_cli(self):
        """scripts/run_pipeline.py를 모듈로 불러온다."""
        spec = importlib.util.spec_from_file_location("run_pipeline_watchdog_test", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_cli_detect_timeout_arguments(self):
        """CLI 기본 10/1, 상한 옵션은 그대로 반영, --no-detect-timeout은 감시 끔, 잘못된 값은 종료 2."""
        mod = self.load_cli()
        base = ["--images", "a.png", "--detect-config", "c.json", "--log", "o.jsonl"]

        def cfg(*extra):
            """인자를 해석해 실행 설정을 만든다."""
            return mod.build_runner_config(mod.build_parser().parse_args(base + list(extra)))

        self.assertEqual((cfg().first_detect_timeout_s, cfg().detect_timeout_s), (10.0, 1.0))
        c = cfg("--first-detect-timeout", "20", "--detect-timeout", "0.5", "--max-frames", "4")
        self.assertEqual((c.first_detect_timeout_s, c.detect_timeout_s, c.max_frames), (20.0, 0.5, 4))
        self.assertEqual((cfg("--detect-timeout", "2").first_detect_timeout_s, cfg("--detect-timeout", "2").detect_timeout_s),
                         (10.0, 2.0))
        self.assertFalse(cfg("--no-detect-timeout").watchdog)

        for bad in (["--detect-timeout", "0"], ["--detect-timeout", "-1"], ["--first-detect-timeout", "inf"],
                    ["--detect-timeout", "x"], ["--no-detect-timeout", "--detect-timeout", "1"]):
            with self.subTest(bad=bad):
                with self.assertRaises(SystemExit) as ctx:
                    cfg(*bad)
                self.assertEqual(ctx.exception.code, 2)

    def test_cli_applies_watchdog_in_real_run(self):
        """CLI 실제 실행이 감시 설정(기본 10/1)을 실행기에 넘겨 worker 경로로 처리하고 그 값을 stderr 요약에 남긴다."""
        import cv2
        from test_pipeline import config
        from test_pipeline_opencv import CENTER, scene

        mod = self.load_cli()
        with tempfile.TemporaryDirectory() as d:
            img = os.path.join(d, "a.png")
            cv2.imwrite(img, scene(CENTER))
            cfg_path = os.path.join(d, "cfg.json")
            with open(cfg_path, "w") as f:
                json.dump({k: getattr(config(), k) for k in (
                    "format_version", "status", "roi", "resize_wh", "resize_interpolation", "gaussian_blur_ksize",
                    "polarity", "threshold_method", "fixed_threshold", "morph_open_ksize", "contour_retrieval",
                    "contour_approximation", "min_area_source_px2", "max_area_source_px2",
                    "border_margin_processed_px")}, f)
            out = os.path.join(d, "run.jsonl")
            err = io.StringIO()
            code = mod.main(["--images", img, "--detect-config", cfg_path, "--log", out], stderr=err)
            err2 = io.StringIO()
            # 기본 경로도 기존 --log를 덮어쓰지 않으므로 두 번째 실행은 새 기록 파일을 쓴다
            out2 = os.path.join(d, "run_no_watch.jsonl")
            code2 = mod.main(["--images", img, "--detect-config", cfg_path, "--log", out2, "--no-detect-timeout"],
                             stderr=err2)

        self.assertEqual(code, 0, err.getvalue())
        self.assertIn("detect_timeout=10.0/1.0", err.getvalue())
        self.assertIn("timeouts=0", err.getvalue())
        self.assertEqual(code2, 0, err2.getvalue())
        self.assertIn("detect_timeout=off", err2.getvalue())
        self.assertEqual(detector_threads(), [])


if __name__ == "__main__":
    unittest.main()
