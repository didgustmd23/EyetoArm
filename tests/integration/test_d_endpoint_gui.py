"""8일차: 서버 주소 입력 → 「적용·재접속」. 명령 TCP와 C 3채널 중계 주소를 앱 재시작 없이 바꾼다.

모든 서버는 이 프로세스 안의 루프백 QTcpServer 대역이다. 실제 C 서버·중계·WSL·Gazebo·로봇은 쓰지 않는다.
명령 대역은 GET_STATUS 에만 미준비 STATUS 로 답하고, 받은 요청 종류를 기록한다(MOVE 0건 확인).
저장소는 임시 INI QSettings 이며 사용자 레지스트리를 쓰지 않는다. 모든 서버·소켓·창은 tearDown 에서 닫는다.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
GUI_SRC = ROOT / "dashboard-client" / "src"
sys.path.insert(0, str(GUI_SRC))
sys.path.insert(0, str(ROOT / "robot_ws" / "src" / "robot_server_package"))
from PySide6.QtCore import QCoreApplication, QEvent, QSettings
from PySide6.QtNetwork import QHostAddress, QTcpServer
from PySide6.QtWidgets import QApplication
from robot_server_package.display_relay import pack_packet
from eyetoarm.gui import motion, theme
from eyetoarm.gui.client import CommandRecord, StatusLink, endpoint_error
from eyetoarm.gui.demo import DemoLink, DemoView
from eyetoarm.gui.relay_view import RelayViewReceiver
from eyetoarm.gui.window import MainWindow, saved_endpoints

sys.path.insert(0, str(ROOT / "tests" / "integration"))
from test_d_relay_gui import jpeg, until  # noqa: E402 - same JPEG/wait helpers as the relay suite


def not_ready(marker):
    return (json.dumps({"version": "1.1", "type": "STATUS", "robot_ready": False,
                        "not_ready_reasons": ["ROS_DISCONNECTED", "SCENE_NOT_READY"],
                        "active_command_id": None, "server_time_ms": marker}) + "\n").encode()


def frame(channel, seq, run):
    meta = {"channel": channel, "variant": "raw", "source_id": f"gz_{channel}", "image_seq": seq,
            "generation": 0, "calibration_id": None, "run_id": run, "capture_generation": 0,
            "detector_config_id": "raw-camera-v1", "job_seq": seq, "stamp_sec": 1, "stamp_nanosec": 2}
    return pack_packet(meta, jpeg())


class FakeCommandServer:
    """GET_STATUS 한 줄마다 미준비 STATUS(server_time_ms=marker)를 돌려준다. 받은 type 을 모두 기록한다."""

    def __init__(self, marker):
        self.marker, self.kinds, self.clients = marker, [], []
        self.server = QTcpServer()
        assert self.server.listen(QHostAddress("127.0.0.1"), 0)
        self.port = self.server.serverPort()
        self.server.newConnection.connect(self._accept)

    def _accept(self):
        sock = self.server.nextPendingConnection()
        self.clients.append(sock)
        buffer = bytearray()

        def read():
            buffer.extend(bytes(sock.readAll()))
            while b"\n" in buffer:
                line, _, rest = bytes(buffer).partition(b"\n")
                buffer[:] = rest
                kind = json.loads(line).get("type")
                self.kinds.append(kind)
                if kind == "GET_STATUS":
                    sock.write(not_ready(self.marker))
        sock.readyRead.connect(read)

    def close(self):
        if getattr(self, "closed", False):
            return
        self.closed = True
        for sock in self.clients:
            sock.abort()
            sock.deleteLater()
        self.server.close()
        self.server.deleteLater()


def cleanup(app, *items):
    for item in items:
        item.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


class EndpointRules(unittest.TestCase):
    def test_only_connectable_numeric_ips_and_valid_ports(self):
        for host, port in (("127.0.0.1", 47090), ("::1", 1), ("10.0.0.5", 65535)):
            self.assertEqual(endpoint_error(host, port), "", (host, port))
        for host, port in (("localhost", 47090), ("c-pc", 47090), ("", 47090), ("0.0.0.0", 47090), ("::", 47090),
                           ("224.0.0.1", 47090), ("ff02::1", 47090), ("255.255.255.255", 47090),
                           ("127.0.0.1", 0), ("127.0.0.1", 70000), ("127.0.0.1", True), ("127.0.0.1", "47090")):
            self.assertNotEqual(endpoint_error(host, port), "", (host, port))

    def test_saved_values_are_checked_and_an_empty_video_ip_follows_the_command_ip(self):
        with tempfile.TemporaryDirectory() as folder:
            store = QSettings(str(Path(folder) / "s.ini"), QSettings.Format.IniFormat)
            self.assertEqual(saved_endpoints(store), {})
            for key, value in {"host": "10.0.0.7", "port": 47100, "view_host": "", "view_port": 47200}.items():
                store.setValue(f"endpoint/{key}", value)
            self.assertEqual(saved_endpoints(store), {"host": "10.0.0.7", "port": 47100, "view_host": "", "view_port": 47200})
            store.setValue("endpoint/host", "0.0.0.0")      # damaged/hand-edited values fall back to client.json
            store.setValue("endpoint/view_port", "abc")
            self.assertEqual(saved_endpoints(store), {})
            store.setValue("endpoint/host", "10.0.0.7")
            for bad in (True, 47100.5, "47100.0", -1):        # bool/float/negative are not ports
                store.setValue("endpoint/port", bad)
                self.assertNotIn("host", saved_endpoints(store), bad)
            del store

    def test_qline_edit_is_styled_in_both_themes(self):
        for name in ("dark", "light"):
            theme.apply(name)
            self.assertIn("QLineEdit {", theme.QSS)
            self.assertIn(f"background: {theme.SURFACE_2}; color: {theme.TEXT}", theme.QSS)
        theme.apply("dark")


class CommandSwitch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.a, self.b = FakeCommandServer(111), FakeCommandServer(222)
        self.link = StatusLink("127.0.0.1", self.a.port, log_path=None)
        self.link.start()
        self.assertTrue(until(lambda: (self.link.last_status or {}).get("server_time_ms") == 111))

    def tearDown(self):
        self.link.stop()
        self.assertFalse(any(t.isActive() for t in (self.link._poll, self.link._retry, self.link._connect_deadline)))
        self.a.close()
        self.b.close()
        cleanup(self.app, self.link)

    def test_switch_invalidates_at_once_ignores_the_old_socket_and_sends_only_get_status(self):
        old_socket = self.link._socket
        self.assertEqual(self.link.set_endpoint("127.0.0.1", self.b.port), "")
        self.assertIsNone(self.link.last_status)                  # old readiness gone before the new reply
        self.assertNotEqual(self.link.state, "connected")
        self.assertIsNot(self.link._socket, old_socket)
        self.assertEqual((self.link.host, self.link.port), ("127.0.0.1", self.b.port))
        for sock in self.a.clients:                               # a late reply from the old server
            sock.write(not_ready(111))
            sock.flush()
        self.assertTrue(until(lambda: (self.link.last_status or {}).get("server_time_ms") == 222))
        until(lambda: False, timeout=0.2)
        self.assertEqual(self.link.last_status["server_time_ms"], 222)
        self.assertFalse(self.link.last_status["robot_ready"])
        self.assertFalse(self.link.can_execute())
        self.assertEqual({*self.a.kinds, *self.b.kinds}, {"GET_STATUS"})   # MOVE/CANCEL 0

    def test_bad_input_or_a_command_in_flight_keeps_the_current_connection(self):
        socket = self.link._socket
        for host, port in (("localhost", self.b.port), ("0.0.0.0", self.b.port), ("127.0.0.1", 0)):
            self.assertNotEqual(self.link.set_endpoint(host, port), "")
        self.link._pending_move = True
        self.assertIn("명령", self.link.set_endpoint("127.0.0.1", self.b.port))
        self.link._pending_move = False
        self.assertIs(self.link._socket, socket)
        self.assertEqual((self.link.port, self.link.state), (self.a.port, "connected"))
        self.assertEqual(self.b.clients, [])

    def test_unconfirmed_result_or_server_active_command_blocks_only_moving_to_another_server(self):
        socket = self.link._socket
        self.link.current_record = CommandRecord("c0ffee00-0000-4000-8000-000000000000", None, b"", 1,
                                                 acceptance="unconfirmed", closed_unconfirmed=True)
        self.assertIn("미확인", self.link.set_endpoint("127.0.0.1", self.b.port))
        self.assertIs(self.link._socket, socket)
        self.assertEqual(self.link.endpoint_busy("127.0.0.1", self.a.port), "")   # same server: reconnect allowed
        self.link.current_record = None
        self.link.last_status = {**self.link.last_status, "active_command_id": "abcdef12-0000-4000-8000-000000000000"}
        self.assertIn("abcdef12", self.link.set_endpoint("127.0.0.1", self.b.port))
        self.assertEqual((self.link.port, self.b.clients), (self.a.port, []))

    def test_keep_unconfirmed_switch_only_when_the_old_server_is_gone_and_the_record_stays(self):
        record = CommandRecord("c0ffee00-0000-4000-8000-000000000000", None, b"MOVE-BYTES", 1, attempts=1,
                               acceptance="unconfirmed", result="연결 끊김 · 실행 결과 미확인")
        self.link.current_record = record
        # the old server still answers: explicit keep is refused, check the result there instead
        self.assertIn("응답 중", self.link.set_endpoint("127.0.0.1", self.b.port, keep_unconfirmed=True))
        self.a.close()                                                       # the old server goes away
        self.assertTrue(until(lambda: self.link.state != "connected"))
        self.assertIn("미확인", self.link.set_endpoint("127.0.0.1", self.b.port))   # not without the explicit choice
        self.assertEqual(self.link.set_endpoint("127.0.0.1", self.b.port, keep_unconfirmed=True), "")
        self.assertIs(self.link.current_record, record)                      # kept, not deleted or resent
        self.assertTrue(record.closed_unconfirmed)
        self.assertEqual((record.raw, record.attempts), (b"MOVE-BYTES", 1))
        self.assertIn("미확인 보존", record.result)
        self.assertTrue(until(lambda: (self.link.last_status or {}).get("server_time_ms") == 222))
        self.assertEqual(set(self.b.kinds), {"GET_STATUS"})                  # no MOVE to the new server

    def test_same_address_reconnects_on_the_same_socket(self):
        socket = self.link._socket
        self.assertEqual(self.link.set_endpoint("127.0.0.1", self.a.port), "")
        self.assertIs(self.link._socket, socket)
        self.assertTrue(until(lambda: len(self.a.clients) == 2 and self.link.last_status is not None))


class WindowSwitch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        motion.set_reduced(True)

    @classmethod
    def tearDownClass(cls):
        motion.set_reduced(False)

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.store = QSettings(str(Path(self.folder.name) / "s.ini"), QSettings.Format.IniFormat)
        self.cmd = FakeCommandServer(333)
        self.relays, self.clients = [], {0: [], 1: []}
        for index in (0, 1):
            server = QTcpServer()
            self.assertTrue(server.listen(QHostAddress("127.0.0.1"), 0))
            server.newConnection.connect(lambda s=server, i=index: self.clients[i].append(s.nextPendingConnection()))
            self.relays.append(server)
        self.link = StatusLink("127.0.0.1", self.cmd.port, log_path=None)
        self.view = RelayViewReceiver("127.0.0.1", self.relays[0].serverPort())
        self.window = MainWindow(self.link, self.view, store=self.store)
        self.window.show()
        self.assertTrue(until(lambda: self.view.state == "connected" and self.clients[0] and self.link.last_status))

    def tearDown(self):
        self.window.close()
        self.assertEqual((self.view.state, self.link.state), ("disconnected", "disconnected"))
        self.assertFalse(self.view._retry.isActive())
        for sock in (*self.clients[0], *self.clients[1]):
            sock.abort()
        for server in self.relays:
            server.close()
        self.cmd.close()
        cleanup(self.app, self.window, self.view, self.link, *self.relays, *self.clients[0], *self.clients[1])
        del self.store
        self.folder.cleanup()

    def send(self, index, run, seq):
        sock = self.clients[index][-1]
        sock.write(b"".join(frame(ch, seq, run) for ch in ("fixed", "gripper", "link")))
        sock.flush()

    def fresh_channels(self):
        display = self.window.display
        return [ch for ch in ("fixed", "gripper", "link") if display.fresh(display.frame(ch))]

    def test_apply_moves_video_and_command_without_restart_and_old_frames_never_return(self):
        self.send(0, "run-A", 1)
        self.assertTrue(until(lambda: len(self.fresh_channels()) == 3))
        edits = self.window.endpoint_edits
        self.assertEqual(edits["view_host"].text(), "")             # inherits the command IP
        edits["view_port"].setText(str(self.relays[1].serverPort()))
        self.window.endpoint_button.click()
        self.assertEqual(self.fresh_channels(), [])                 # invalidated at once, before any new frame
        self.assertIsNone(self.window._status)
        self.assertIn("연결 요청함", self.window.endpoint_note.text())   # a request, not a claim of reception
        self.assertIn("저장함", self.window.endpoint_note.text())
        self.assertTrue(until(lambda: self.view.state == "connected" and self.clients[1]))
        count = self.view.frame_count
        self.send(0, "run-A", 2)                                    # late frames on the old relay connection
        until(lambda: False, timeout=0.2)
        self.assertEqual(self.view.frame_count, count)
        self.send(1, "run-B", 1)
        self.assertTrue(until(lambda: len(self.fresh_channels()) == 3))
        self.assertIn(str(self.relays[1].serverPort()), self.view.title)
        self.assertTrue(until(lambda: self.link.state == "connected" and self.link.last_status is not None))
        self.assertEqual(set(self.cmd.kinds), {"GET_STATUS"})
        self.assertEqual(saved_endpoints(self.store), {"host": "127.0.0.1", "port": self.cmd.port,
                                                       "view_host": "", "view_port": self.relays[1].serverPort()})

    def test_a_new_server_may_reuse_the_run_id_and_low_numbers_but_the_same_address_keeps_its_guard(self):
        self.send(0, "run-A", 5)
        self.assertTrue(until(lambda: self.view.run_id == "run-A"))   # seen first, else newest-only merge skips it
        self.send(0, "run-B", 1)                                    # run-A is now retired on this server
        self.assertTrue(until(lambda: self.view.run_id == "run-B" and len(self.fresh_channels()) == 3))
        self.window.endpoint_button.click()                         # same address: reconnect, keep the guard
        self.assertTrue(until(lambda: self.view.state == "connected" and len(self.clients[0]) == 2))
        drops = self.view.stale_session_drops
        self.send(0, "run-A", 9)
        self.assertTrue(until(lambda: self.view.stale_session_drops > drops))
        self.assertEqual(self.window.display.run_id, "run-B")
        self.window.endpoint_edits["view_port"].setText(str(self.relays[1].serverPort()))
        self.window.endpoint_button.click()                         # another server: its history starts empty
        self.assertIsNone(self.window.display.run_id)
        self.assertTrue(until(lambda: self.view.state == "connected" and self.clients[1]))
        self.send(1, "run-A", 1)                                    # same run id and a lower number than before
        self.assertTrue(until(lambda: len(self.fresh_channels()) == 3))
        self.assertEqual((self.view.run_id, self.window.display.run_id), ("run-A", "run-A"))

    def test_a_failed_save_is_reported_not_claimed(self):
        store = Mock()
        store.status.return_value = QSettings.Status.AccessError
        self.window.store = store
        self.window.endpoint_button.click()
        self.assertIn("저장 실패", self.window.endpoint_note.text())
        self.assertNotIn("저장함", self.window.endpoint_note.text())

    def test_invalid_or_busy_apply_changes_nothing(self):
        edits, view_socket, link_socket = self.window.endpoint_edits, self.view._socket, self.link._socket
        self.assertEqual((edits["port"].maxLength(), edits["view_port"].maxLength()), (5, 5))
        for key, value in (("host", "c-server"), ("port", "0"), ("port", "65536"), ("port", "9" * 40), ("port", "-1"),
                           ("port", "４７０９０"), ("view_host", "0.0.0.0"), ("view_port", "x"), ("view_port", "")):
            original = edits[key].text()
            edits[key].setText(value)
            self.window.endpoint_button.click()
            self.assertIn("적용 안 함", self.window.endpoint_note.text())
            edits[key].setText(original)
        self.link._pending_move = True
        self.window.endpoint_button.click()
        self.assertIn("명령", self.window.endpoint_note.text())
        self.link._pending_move = False
        self.assertIs(self.view._socket, view_socket)
        self.assertIs(self.link._socket, link_socket)
        self.assertEqual(self.view.state, "connected")
        self.assertEqual(saved_endpoints(self.store), {})


class DemoAndLegacy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_demo_has_no_live_address_controls_and_stores_nothing(self):
        link, view = DemoLink(), DemoView()
        store = Mock()
        window = MainWindow(link, view, demo=True, store=store)
        self.assertFalse(self.window_enabled(window))
        self.assertIn("모의", window.endpoint_note.text())
        window._apply_endpoint()
        store.setValue.assert_not_called()
        window.close()
        cleanup(self.app, window, link, view)

    @staticmethod
    def window_enabled(window):
        return window.endpoint_button.isEnabled() or any(e.isEnabled() for e in window.endpoint_edits.values())

    def run_main(self, saved, args):
        probe = ("import json, sys; import eyetoarm.gui.window as w\n"
                 "from PySide6.QtCore import QSettings\n"
                 "w.settings = lambda: QSettings(sys.argv[1], QSettings.Format.IniFormat)\n"
                 "def fake(link, view, demo=False, store=None):\n"
                 "    print(json.dumps([type(link).__name__, link.host, link.port, type(view).__name__,\n"
                 "                      getattr(view, 'host', None), getattr(view, 'port', None)])); raise SystemExit(0)\n"
                 "w.MainWindow = fake; w.main(json.loads(sys.argv[2]))\n")
        with tempfile.TemporaryDirectory() as folder:
            ini = str(Path(folder) / "s.ini")
            store = QSettings(ini, QSettings.Format.IniFormat)
            for key, value in saved.items():
                store.setValue(f"endpoint/{key}", value)
            store.sync()
            del store
            env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(GUI_SRC), "PYTHONIOENCODING": "utf-8"}
            result = subprocess.run([sys.executable, "-B", "-c", probe, ini, json.dumps(args)], env=env, cwd=str(ROOT),
                                    capture_output=True, text=True, encoding="utf-8", timeout=60, stdin=subprocess.DEVNULL)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout.strip().splitlines()[-1])

    def test_explicit_cli_beats_saved_which_beats_client_json(self):
        saved = {"host": "10.0.0.7", "port": 47100, "view_host": "", "view_port": 47200}
        self.assertEqual(self.run_main(saved, []), ["StatusLink", "10.0.0.7", 47100, "RelayViewReceiver", "10.0.0.7", 47200])
        self.assertEqual(self.run_main(saved, ["--host", "10.1.1.1"]),
                         ["StatusLink", "10.1.1.1", 47100, "RelayViewReceiver", "10.1.1.1", 47200])
        self.assertEqual(self.run_main(saved, ["--view-host", "10.9.9.9", "--view-port", "47111"]),
                         ["StatusLink", "10.0.0.7", 47100, "RelayViewReceiver", "10.9.9.9", 47111])
        self.assertEqual(self.run_main({}, []), ["StatusLink", "127.0.0.1", 47090, "RelayViewReceiver", "127.0.0.1", 47092])
        self.assertEqual(self.run_main({**saved, "host": "localhost"}, [])[1:3], ["127.0.0.1", 47090])
        self.assertEqual(self.run_main(saved, ["--view", "camera"])[3:], ["GzViewReceiver", "127.0.0.1", 47091])
        self.assertEqual(self.run_main(saved, ["--demo"])[0], "DemoLink")


if __name__ == "__main__":
    unittest.main()
