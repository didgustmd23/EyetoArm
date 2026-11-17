"""Optional Day7 check: Windows StatusLink <-> unchanged C TcpServer in an ALREADY RUNNING WSL distro.

GET_STATUS only. The WSL peer reports robot_ready=False, so the GUI must show a fresh STATUS and keep MOVE
locked; no MOVE is ever sent across WSL, no robot/ROS/Gazebo is used and no fixed port (47090) is opened.

Skipped unless explicitly enabled (it starts a process in WSL):
    set EYETOARM_RUN_WSL_TESTS=1
    set EYETOARM_WSL_DISTRO=Ubuntu-24.04        (optional; this is the default)
    set PYTHONPATH=dashboard-client\\src
    python -B -m unittest tests.integration.test_d_day7_wsl
The peer stops on STOP/stdin EOF and by itself after 30 s; this test never kills or terminates it and never
runs `wsl --shutdown`. The repository path is translated by the documented /mnt/<drive>/ default mount rule.
The distro must already be running (`wsl.exe -l --running -q`, read-only); a stopped distro is skipped, never
booted. Windows reaches the peer's 127.0.0.1 port through WSL2 localhost forwarding (the NAT default
`localhostForwarding=true`; mirrored networking also works). If that is disabled the test fails to connect.
"""
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock
from pathlib import Path, PureWindowsPath

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dashboard-client" / "src"))   # canonical D GUI
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication
from eyetoarm.gui.client import StatusLink
from eyetoarm.tcp.protocol import MoveTarget

ENABLED = os.environ.get("EYETOARM_RUN_WSL_TESTS") == "1"
DISTRO = os.environ.get("EYETOARM_WSL_DISTRO", "Ubuntu-24.04")
PEER = ROOT / "tests" / "support" / "d_day7_wsl_peer.py"
FAST = dict(readiness_poll_ms=100, readiness_timeout_ms=1000, status_watchdog_ms=1500, reconnect_ms=200,
            connect_timeout_ms=2000)


def linux_path(path: Path) -> str:
    """C:\\a\\b -> /mnt/c/a/b (WSL default automount). Other path forms are refused, not guessed."""
    windows = PureWindowsPath(path)
    drive = windows.drive
    if len(drive) != 2 or drive[1] != ":" or not drive[0].isalpha():
        raise ValueError(f"not a drive-letter path: {path}")
    return "/mnt/" + drive[0].lower() + "/" + "/".join(windows.parts[1:])


def distro_running(distro, run=subprocess.run):
    """Read-only: is `distro` in `wsl.exe -l --running -q`? (UTF-16LE output). Never starts a distro."""
    try:
        result = run(["wsl.exe", "-l", "--running", "-q"], capture_output=True, timeout=15,
                     stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return False
    if result.returncode != 0:
        return False
    names = result.stdout.decode("utf-16-le", errors="replace").replace("\ufeff", "").replace("\x00", "")
    return distro.casefold() in {name.strip().casefold() for name in names.splitlines() if name.strip()}


def start_peer(distro, popen=subprocess.Popen, running=distro_running):
    """Start the WSL peer only when the distro is already running; otherwise return (None, args)."""
    args = ["wsl.exe", "-d", distro, "--", "python3", "-B", "-u", linux_path(PEER), "--repo", linux_path(ROOT),
            "--deadline", "30"]
    if not running(distro):
        return None, args
    return popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                 text=True, encoding="utf-8", errors="replace"), args


def until(predicate, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


@unittest.skipUnless(os.name == "nt" and ENABLED, "opt-in: EYETOARM_RUN_WSL_TESTS=1 with a running WSL distro")
class WslMockC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_get_status_only_not_ready_peer_locks_execute(self):
        lines: "queue.Queue[str]" = queue.Queue(maxsize=200)
        errors: list[str] = []
        temp = tempfile.TemporaryDirectory(prefix="eyetoarm-d7-wsl-win-")
        self.addCleanup(temp.cleanup)
        launched = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        process, args = start_peer(DISTRO)
        if process is None:
            self.skipTest(f"WSL distro {DISTRO!r} is not running; it is not started by this test")
        print(f"\n[WSL launch] windows_pid={process.pid} at={launched} args={args}", file=sys.stderr)

        def note(text):   # bounded diagnostics from both streams
            if len(errors) < 200:
                errors.append(text)

        def pump(stream, sink):
            for line in stream:
                try:
                    sink(line.rstrip("\n"))
                except queue.Full:   # bounded: drop rather than block the child
                    pass

        out_reader = threading.Thread(target=pump, args=(process.stdout, lines.put_nowait), name="d7-wsl-stdout")
        err_reader = threading.Thread(target=pump, args=(process.stderr, note), name="d7-wsl-stderr")
        out_reader.start()
        err_reader.start()

        def next_event(timeout):
            end = time.monotonic() + timeout
            while time.monotonic() < end:
                try:
                    line = lines.get(timeout=0.1)
                except queue.Empty:
                    if process.poll() is not None and lines.empty():
                        return None
                    continue
                try:
                    return json.loads(line)
                except ValueError:
                    note("stdout: " + line)
            return None

        link = None
        stopped = None
        try:
            ready = next_event(30)
            self.assertIsNotNone(ready, f"peer not ready: {errors[-10:]}")
            self.assertEqual((ready["event"], ready["host"]), ("ready", "127.0.0.1"))
            self.assertNotEqual(ready["port"], 47090)
            print(f"[WSL peer] windows_pid={process.pid} linux_pid={ready['pid']} ppid={ready['ppid']} "
                  f"created_ms={ready['created_ms']} exe={ready['executable']} "
                  f"port={ready['port']} temp={ready['temp']}", file=sys.stderr)

            log_path = Path(temp.name) / "d-audit.jsonl"
            link = StatusLink("127.0.0.1", ready["port"], log_path=log_path, config=FAST, stable_count=2)
            statuses = []
            link.status_received.connect(statuses.append)
            link.start()
            self.assertTrue(until(lambda: statuses and not link.is_stale(), 15), "no fresh STATUS from WSL C")
            self.assertFalse(statuses[-1]["robot_ready"])
            self.assertIn("SIM_PAUSED", statuses[-1]["not_ready_reasons"])
            self.assertFalse(link.can_execute())
            now = time.monotonic_ns() // 1_000_000
            target = MoveTarget((100, 200), 30, "manual_test", 1, now, "PC_RECEIVE", "manual_test",
                                created_monotonic_ms=now)
            self.assertEqual(link.execute(target), "")   # locked locally: nothing is sent
            sent = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            self.assertFalse(any(row["dir"] == "TX" and row["kind"] == "MOVE" for row in sent))
        finally:
            if link is not None:   # close the Qt link first, then ask the peer to stop
                link.stop()
                link.deleteLater()
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                self.app.processEvents()
            try:
                process.stdin.write("STOP\n")
                process.stdin.flush()
            except OSError:
                pass
            try:
                process.stdin.close()
            except OSError:
                pass
            try:
                code = process.wait(timeout=40)   # graceful only; the peer also has its own 30 s deadline
            except subprocess.TimeoutExpired:
                code = None
            if code is not None:
                while stopped is None:
                    event = next_event(5)
                    if event is None or event.get("event") == "stopped":
                        stopped = event
                        break
            out_reader.join(timeout=10)
            err_reader.join(timeout=10)
            if not out_reader.is_alive() and not err_reader.is_alive():   # close our pipe ends only when unused
                process.stdout.close()
                process.stderr.close()
        self.assertEqual(code, 0, f"WSL peer did not exit gracefully (left running, not killed): {errors[-10:]}")
        self.assertFalse(out_reader.is_alive())
        self.assertFalse(err_reader.is_alive())
        self.assertIsNotNone(stopped, errors[-10:])
        self.assertIn(stopped["reason"], ("stdin STOP", "stdin EOF"))
        self.assertTrue(stopped["temp_removed"])
        self.assertGreaterEqual(stopped["requests"].get("GET_STATUS", 0), 1)   # the peer really saw our queries
        self.assertEqual(set(stopped["requests"]), {"GET_STATUS"})              # and nothing else (no MOVE)
        print(f"[WSL peer] stopped reason={stopped['reason']} requests={stopped['requests']} "
              f"temp_removed={stopped['temp_removed']}", file=sys.stderr)


class LinuxPathRule(unittest.TestCase):
    def test_stopped_distro_is_skipped_without_spawning_the_peer(self):
        popen = unittest.mock.Mock()
        process, args = start_peer("Ubuntu-24.04", popen=popen, running=lambda _distro: False)
        self.assertIsNone(process)
        popen.assert_not_called()
        self.assertEqual(args[:4], ["wsl.exe", "-d", "Ubuntu-24.04", "--"])

    def test_running_list_is_utf16_and_exact(self):
        def fake(output, code=0):
            return lambda *_a, **_k: subprocess.CompletedProcess([], code, ("\ufeff" + output).encode("utf-16-le"), b"")
        self.assertTrue(distro_running("Ubuntu-24.04", run=fake("docker-desktop\r\nUbuntu-24.04\r\n")))
        self.assertFalse(distro_running("Ubuntu-24.04", run=fake("Ubuntu\r\n")))
        self.assertFalse(distro_running("Ubuntu-24.04", run=fake("Ubuntu-24.04\r\n", code=1)))
        self.assertFalse(distro_running("Ubuntu-24.04", run=fake("")))

    def test_drive_paths_map_to_default_mount_and_others_are_refused(self):
        self.assertEqual(linux_path(Path(r"C:\Users\user\WSL\EyetoArm")), "/mnt/c/Users/user/WSL/EyetoArm")
        self.assertEqual(linux_path(Path(r"D:\x y\z.py")), "/mnt/d/x y/z.py")
        with self.assertRaises(ValueError):
            linux_path(PureWindowsPath(r"\\server\share\x"))


if __name__ == "__main__":
    unittest.main()
