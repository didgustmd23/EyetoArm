"""감독 스크립트(`scripts/run_pipeline_supervisor.py`) 시험: 종료 3 재시작 횟수 한도, 정지 신호 전달·대기·단계적 종료,
정지 뒤 재시작 안 함(D-11·D-9). 실제 감독 스크립트를 자식 프로세스로 띄우고 가짜 파이프라인을 감독하게 한다.
"""
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUPERVISOR = ROOT / "scripts" / "run_pipeline_supervisor.py"

# 가짜 파이프라인: 시작할 때 실행 기록을 한 줄 남기고, 받은 신호를 기록한다.
# 인자: 기록 파일, 시작 때 종료 코드(없으면 계속 돎), 신호 받은 뒤 정리 시간, 신호 뒤 종료 코드, 무시할 신호 목록
FAKE = r"""
import json, os, signal, sys, time
log, code, cleanup_s, after_code, ignore = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
def note(**kw):
    with open(log, "a") as fh:
        fh.write(json.dumps({"pid": os.getpid(), **kw}) + "\n")
got = []
def on(sig, frame):
    got.append(sig)
    note(signal=signal.Signals(sig).name)
for name in ("SIGINT", "SIGTERM", "SIGHUP"):
    sig = getattr(signal, name)
    signal.signal(sig, signal.SIG_IGN if name in ignore.split(",") else on)
note(start=True)
if code != "run":
    sys.exit(int(code))
while not got:
    time.sleep(0.01)
time.sleep(cleanup_s)
note(cleaned=True)
sys.exit(after_code)
"""


def load_supervisor():
    """감독 스크립트를 모듈로 읽는다(함수 단위 시험용)."""
    spec = importlib.util.spec_from_file_location("run_pipeline_supervisor_under_test", SUPERVISOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


class Base(unittest.TestCase):
    """임시 폴더·가짜 파이프라인·감독 프로세스 띄우기와 기록 읽기."""

    def setUp(self):
        """임시 폴더와 가짜 파이프라인 파일."""
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.fake = Path(self.dir.name) / "fake_pipeline.py"
        self.fake.write_text(FAKE, encoding="utf-8")
        self.log = Path(self.dir.name) / "runs.jsonl"

    def start(self, *sup_args, code="run", cleanup_s=0.2, after_code=0, ignore=""):
        """감독 스크립트를 새 세션에서 띄운다(SIGINT 기본 처리). 끝날 때 남은 것은 그룹째 죽인다."""
        cmd = [sys.executable, str(SUPERVISOR), *sup_args, "--", sys.executable, str(self.fake), str(self.log), code,
               str(cleanup_s), str(after_code), ignore]
        proc = subprocess.Popen(cmd, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
        self.addCleanup(self.kill_group, proc)

        return proc

    def kill_group(self, proc):
        """시험이 띄운 그룹 정리(이미 끝났으면 무변)."""
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

        proc.wait(10)

    def records(self):
        """가짜 파이프라인 기록."""
        if not self.log.exists():
            return []

        return [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]

    def wait_started(self, n=1, limit_s=10.0):
        """가짜 파이프라인 n번째 시작까지 기다린다. 그 pid."""
        deadline = time.monotonic() + limit_s

        while time.monotonic() < deadline:
            starts = [r for r in self.records() if r.get("start")]

            if len(starts) >= n:
                return starts[n - 1]["pid"]

            time.sleep(0.02)

        raise AssertionError(f"fake pipeline start {n} not seen: {self.records()}")


class RestartLimitTest(Base):
    """종료 3 재시작은 창 안에서 횟수 한도까지만 한다."""

    def test_exit3_restarts_up_to_limit_then_stops(self):
        """늘 종료 3인 파이프라인: 한도 2면 처음 1번 + 재시작 2번 = 3번 돌고 감독은 3으로 끝난다."""
        proc = self.start("--restart-delay", "0.01", "--max-restarts", "2", "--restart-window", "600", code="3")
        out, _ = proc.communicate(timeout=20)

        self.assertEqual(proc.returncode, 3, out)
        self.assertEqual(len([r for r in self.records() if r.get("start")]), 3)
        self.assertIn("restart limit reached (2 within 600 s)", out)

    def test_other_exit_codes_are_not_restarted(self):
        """종료 1(예: worker_stalled)은 지금처럼 재시작하지 않는다(D11-d 「가」 유지)."""
        proc = self.start("--restart-delay", "0.01", code="1")
        out, _ = proc.communicate(timeout=20)

        self.assertEqual(proc.returncode, 1, out)
        self.assertEqual(len([r for r in self.records() if r.get("start")]), 1)

    def test_sliding_window_forgets_old_restarts(self):
        """창 밖으로 나간 재시작은 세지 않는다."""
        mod = load_supervisor()
        history = deque([0.0, 10.0, 20.0])

        self.assertFalse(mod._restart_allowed(history, 30.0, max_restarts=3, window=600.0))
        self.assertTrue(mod._restart_allowed(history, 615.0, max_restarts=3, window=600.0))
        self.assertEqual(list(history), [20.0])


class StopSignalTest(Base):
    """정지 신호는 파이프라인에 한 번 넘기고, 끝날 때까지 기다리고, 그 뒤 재시작하지 않는다."""

    def test_sigint_to_supervisor_is_forwarded_once_and_waited(self):
        """감독에만 SIGINT: 파이프라인이 SIGINT를 한 번 받고 정리(1 s)를 끝내며, 종료 3이어도 재시작하지 않는다."""
        proc = self.start("--restart-delay", "0.01", cleanup_s=1.0, after_code=3)
        pid = self.wait_started()
        proc.send_signal(signal.SIGINT)
        out, _ = proc.communicate(timeout=20)
        mine = [r for r in self.records() if r["pid"] == pid]

        self.assertEqual(proc.returncode, 3, out)
        self.assertEqual([r.get("signal") for r in mine if "signal" in r], ["SIGINT"])
        self.assertTrue(any(r.get("cleaned") for r in mine), out)
        self.assertEqual(len([r for r in self.records() if r.get("start")]), 1)
        self.assertIn("stopped by SIGINT", out)

    def test_group_sigint_is_not_doubled(self):
        """그룹 전체 SIGINT(터미널 Ctrl+C): 파이프라인은 직접 받은 것 + 감독 전달로 두 번까지 받지만 정리는 끝난다."""
        proc = self.start(cleanup_s=1.0)
        pid = self.wait_started()
        os.killpg(proc.pid, signal.SIGINT)
        out, _ = proc.communicate(timeout=20)
        mine = [r for r in self.records() if r["pid"] == pid]

        self.assertEqual(proc.returncode, 0, out)
        self.assertTrue(any(r.get("cleaned") for r in mine), out)
        self.assertLessEqual(len([r for r in mine if "signal" in r]), 2)

    def test_sigterm_and_sighup_are_forwarded(self):
        """감독에 SIGTERM·SIGHUP: 같은 신호가 파이프라인에 넘어가고 감독이 먼저 죽지 않는다(고아 없음)."""
        for sig in (signal.SIGTERM, signal.SIGHUP):
            with self.subTest(sig=sig.name):
                self.log.unlink(missing_ok=True)
                proc = self.start(cleanup_s=0.5)
                pid = self.wait_started()
                proc.send_signal(sig)
                out, _ = proc.communicate(timeout=20)
                mine = [r for r in self.records() if r["pid"] == pid]

                self.assertEqual(proc.returncode, 0, out)
                self.assertEqual([r.get("signal") for r in mine if "signal" in r], [sig.name])
                self.assertTrue(any(r.get("cleaned") for r in mine))

    def test_escalates_to_sigterm_then_sigkill(self):
        """정지를 무시하는 파이프라인: stop-wait 뒤 SIGTERM, 그것도 무시하면 term-wait 뒤 SIGKILL."""
        proc = self.start("--stop-wait", "0.3", "--term-wait", "0.3", ignore="SIGINT,SIGTERM")
        pid = self.wait_started()
        t0 = time.monotonic()
        proc.send_signal(signal.SIGINT)
        out, _ = proc.communicate(timeout=20)

        self.assertEqual(proc.returncode, 128 + signal.SIGKILL, out)   # 셸과 같은 표기
        self.assertIn("sending SIGTERM", out)
        self.assertIn("sending SIGKILL", out)
        self.assertLess(time.monotonic() - t0, 10.0)
        self.assertFalse(os.path.exists(f"/proc/{pid}") and "Z" not in Path(f"/proc/{pid}/stat").read_text().split()[2])

    def test_stop_during_restart_delay_does_not_restart(self):
        """재시작 대기 중 정지 신호: 다음 세대를 띄우지 않고 바로 끝난다."""
        proc = self.start("--restart-delay", "5", code="3")
        self.wait_started()
        time.sleep(0.3)
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=10)

        self.assertEqual(proc.returncode, 3, out)
        self.assertEqual(len([r for r in self.records() if r.get("start")]), 1)
        self.assertIn("before restart", out)


if __name__ == "__main__":
    unittest.main()
