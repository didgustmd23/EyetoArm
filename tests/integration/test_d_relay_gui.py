"""--view relay: MainWindow + RelayViewReceiver against an in-process loopback QTcpServer fixture.

Packets are made by the UNCHANGED C pack_packet. The fixture's link detect/center packets are test data only;
the current C relay sends raw for all three channels. No real C relay, WSL, Gazebo or robot is used; the
command link never opens a socket. Every server/socket/window is closed in tearDown.
"""
import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
GUI_SRC = ROOT / "dashboard-client" / "src"
sys.path.insert(0, str(GUI_SRC))   # canonical D GUI
sys.path.insert(0, str(ROOT / "robot_ws" / "src" / "robot_server_package"))
from PySide6.QtCore import QBuffer, QCoreApplication, QEvent, QIODevice
from PySide6.QtGui import QColor, QImage
from PySide6.QtNetwork import QHostAddress, QTcpServer
from PySide6.QtWidgets import QApplication
from robot_server_package.display_relay import pack_packet
from eyetoarm.gui import motion, relay_view
from eyetoarm.gui.client import StatusLink
from eyetoarm.gui.relay_view import RelayViewReceiver
from eyetoarm.gui.window import MainWindow, NO_VALUE

READY = {"version": "1.1", "type": "STATUS", "robot_ready": True, "not_ready_reasons": [],
         "active_command_id": None, "server_time_ms": 1}


def until(predicate, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def jpeg(color="#3080c0"):
    image = QImage(64, 36, QImage.Format.Format_RGB32)
    image.fill(QColor(color))
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "JPG")
    return bytes(buffer.data())


def packet(channel, seq, run="run-A", variant="raw", data=None):
    meta = {"channel": channel, "variant": variant, "source_id": f"gz_{channel}", "image_seq": seq,
            "generation": 0, "calibration_id": None, "run_id": run, "capture_generation": 0,
            "detector_config_id": "raw-camera-v1", "job_seq": seq, "stamp_sec": 1, "stamp_nanosec": 2}
    return pack_packet(meta, data or jpeg())


class RelayWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        motion.set_reduced(True)

    @classmethod
    def tearDownClass(cls):
        motion.set_reduced(False)

    def setUp(self):
        self.server = QTcpServer()
        self.assertTrue(self.server.listen(QHostAddress("127.0.0.1"), 0))
        self.clients = []
        self.server.newConnection.connect(lambda: self.clients.append(self.server.nextPendingConnection()))
        link = StatusLink(log_path=None)
        link.start = Mock()                       # command TCP stays closed in this test
        link._send = Mock(return_value=True)
        link.state, link._conn_id = "connected", 1
        link._handle_line(json.dumps(READY).encode() + b"\n")
        self.link = link
        self.view = RelayViewReceiver("127.0.0.1", self.server.serverPort())
        self.window = MainWindow(link, self.view)
        self.window.show()
        self.now = 1000.0
        self.window.display.clock = lambda: self.now
        self.assertTrue(until(lambda: self.view.state == "connected" and self.clients))

    def tearDown(self):
        self.window.close()
        self.assertFalse(self.view._retry.isActive())
        self.assertFalse(self.view._flush_timer.isActive())
        self.assertEqual(self.view.state, "disconnected")
        for client in self.clients:
            client.abort()
        self.server.close()
        for item in (self.window, self.view, self.link, self.server, *self.clients):
            item.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()

    def send(self, *packets, client=-1):
        socket = self.clients[client]
        socket.write(b"".join(packets))
        socket.flush()

    def settle(self, predicate):
        self.assertTrue(until(predicate))
        self.window._update_status()

    def views(self):
        stage = self.window.stage
        return {"fixed": stage.main, "gripper": stage.gripper, "link": stage.link}

    def mode_of(self, channel):
        return self.views()[channel]._mode

    def test_three_raw_channels_fill_their_own_views_and_status_needs_every_channel(self):
        self.send(packet("fixed", 1))   # first packet is not link: the session (and link source) still starts
        self.settle(lambda: self.view.frame_count == 1)
        self.assertEqual((self.view.run_id, self.window.display.link_source_id), ("run-A", "gz_link"))
        self.assertEqual(self.mode_of("fixed"), "live")
        self.assertNotEqual(self.mode_of("gripper"), "live")
        self.assertEqual(self.window.chip_view._level, "warn")       # TCP connected alone is not "good video"
        self.assertIn("그리퍼", self.window.chip_view._text)
        self.send(packet("gripper", 1), packet("link", 1))
        self.settle(lambda: self.view.frame_count == 3)
        self.assertEqual([self.mode_of(c) for c in ("fixed", "gripper", "link")], ["live"] * 3)
        self.assertEqual(self.window.chip_view._level, "ok")
        self.assertTrue(all(value.text() == NO_VALUE for value in self.window.measure_values))   # no results
        self.assertIn("측정 결과 없음", self.window.stage.popup.values["detail"].toolTip())
        self.assertEqual(self.clients[0].bytesAvailable(), 0)      # D never sends anything to the relay

    def test_link_selection_is_display_only_synced_and_never_falls_back_to_raw(self):
        self.send(packet("fixed", 1), packet("gripper", 1), packet("link", 1))
        self.settle(lambda: self.view.frame_count == 3)
        buttons = self.window.mode_buttons
        self.assertEqual(len(buttons), 6)   # setup page + operate page
        detect = [b for b in buttons if b._mode == "detect"]
        detect[-1].click()                  # operate-page selector
        self.window._update_status()
        self.assertEqual({b._mode for b in buttons if b.isChecked()}, {"detect"})
        self.assertEqual(sum(b.isChecked() for b in buttons), 2)        # both pages show the same selection
        self.assertEqual(self.mode_of("link"), "waiting")
        self.assertIn("수신 대기", self.views()["link"]._reason)
        self.assertIn("raw 로 대신 보이지 않음", self.views()["link"]._reason)
        self.assertEqual((self.mode_of("fixed"), self.mode_of("gripper")), ("live", "live"))   # unaffected
        self.assertEqual(self.window.chip_view._level, "warn")
        self.send(packet("link", 1, variant="detect", data=jpeg("#c03030")))   # fixture-only detect packet
        self.settle(lambda: self.view.frame_count == 4)
        self.assertEqual(self.mode_of("link"), "live")
        self.assertIn("detect", self.views()["link"]._footer)
        [b for b in buttons if b._mode == "center"][0].click()          # setup-page selector
        self.window._update_status()
        self.assertEqual(self.mode_of("link"), "waiting")
        [b for b in buttons if b._mode == "raw"][0].click()
        self.window._update_status()
        self.assertEqual(self.mode_of("link"), "live")
        self.assertIn("raw", self.views()["link"]._footer)
        self.assertEqual(self.clients[0].bytesAvailable(), 0)           # selection never requested anything from C

    def test_newest_per_turn_bad_jpeg_and_old_session_cannot_come_back(self):
        self.send(*(packet("link", seq) for seq in range(1, 6)))       # coalesced in one write
        self.settle(lambda: self.view.frame_count >= 1 and not self.view._flush_timer.isActive())
        self.assertEqual(self.view.frame_count, 1)
        self.assertEqual(self.window.display.frame("link")["image_seq"], 5)
        self.send(packet("gripper", 1, data=b"\xff\xd8broken\xff\xd9"))
        self.settle(lambda: self.view.decode_errors == 1)
        self.assertEqual(self.view.state, "connected")
        self.send(packet("fixed", 1, run="run-B"))                       # relay restarted: new session
        self.settle(lambda: self.view.run_id == "run-B")
        self.assertIsNone(self.window.display.frame("link"))             # old run frames are gone
        self.send(packet("link", 9, run="run-A"))                        # an old-run packet must not flip back
        self.settle(lambda: self.view.stale_session_drops == 1)
        self.assertEqual((self.view.run_id, self.window.display.run_id), ("run-B", "run-B"))
        self.assertIsNone(self.window.display.frame("link"))

    def test_corrupt_stream_is_visible_not_just_missing_video(self):
        self.window._update_status()
        self.assertNotIn("수신 오류", self.window.chip_view.toolTip())
        bad_meta = b"{not json"
        body = len(bad_meta).to_bytes(4, "big") + bad_meta + b"\xff\xd8x"
        self.send(b"ETV1" + len(body).to_bytes(4, "big") + body,
                  packet("gripper", 1, data=b"\xff\xd8broken\xff\xd9"))
        self.settle(lambda: self.view.packet_errors == 1 and self.view.decode_errors == 1)
        self.assertIn("수신 오류 있음", self.window.chip_view._text)
        tip = self.window.chip_view.toolTip()
        self.assertIn("메타/패킷 1", tip)
        self.assertIn("JPEG 1", tip)
        self.assertIn("gripper JPEG", tip)                     # the latest error is named, bounded to one line
        self.assertEqual(self.view.state, "connected")

    def test_per_channel_staleness(self):
        self.send(packet("fixed", 1), packet("gripper", 1), packet("link", 1))
        self.settle(lambda: self.view.frame_count == 3)
        self.now = 1002.5
        self.send(packet("gripper", 2))
        self.settle(lambda: self.view.frame_count == 4)
        self.now = 1004.0                                                 # fixed/link 4 s old, gripper 1.5 s
        self.window._update_status()
        self.assertEqual((self.mode_of("fixed"), self.mode_of("gripper"), self.mode_of("link")),
                         ("stale", "live", "stale"))
        self.assertEqual(self.window.chip_view._level, "warn")
        self.assertIn("탑뷰", self.window.chip_view._text)
        self.assertIn("링크", self.window.chip_view._text)
        self.assertNotIn("그리퍼", self.window.chip_view._text)

    def test_disconnect_and_same_run_cached_replay_are_not_revived(self):
        frames = [packet("fixed", 3), packet("gripper", 3), packet("link", 3)]
        self.send(*frames)
        self.settle(lambda: self.view.frame_count == 3)
        self.clients[0].disconnectFromHost()
        self.settle(lambda: self.view.state == "disconnected")
        self.assertTrue(all(self.mode_of(c) == "empty" for c in ("fixed", "gripper", "link")))
        self.assertIn("연결 안 됨", self.views()["fixed"]._reason)
        self.assertNotEqual(self.window.chip_view._level, "ok")
        self.assertTrue(until(lambda: self.view.state == "connected" and len(self.clients) == 2, 6))
        self.send(*frames)                    # C replays its cached newest packets to a new client
        self.settle(lambda: self.view.frame_count == 6)
        self.assertTrue(all(self.window.display.frame(c) is None for c in ("fixed", "gripper", "link")))
        self.assertNotEqual(self.window.chip_view._level, "ok")
        self.send(packet("fixed", 4))
        self.settle(lambda: self.view.frame_count == 7)
        self.assertEqual(self.mode_of("fixed"), "live")

    def test_broken_framing_drops_the_connection_and_retries_with_the_error_kept_visible(self):
        self.send(b"ETV1" + (relay_view.MAX_PACKET + 1).to_bytes(4, "big"))
        self.settle(lambda: self.view.state == "disconnected")
        self.assertTrue(self.view._retry.isActive())
        self.assertEqual(self.view.framing_errors, 1)
        tip = self.window.chip_view.toolTip()
        self.assertIn("경계 1", tip)                                     # visible while disconnected
        self.assertIn(f"127.0.0.1:{self.server.serverPort()}", tip)
        self.assertIn("수신 오류 있음", self.window.chip_view._text)
        self.assertTrue(until(lambda: self.view.state == "connected" and len(self.clients) == 2, 6))
        self.send(b"EAV1" + (10).to_bytes(4, "big"), client=1)          # wrong magic on the new connection
        self.settle(lambda: self.view.framing_errors == 2)
        self.window._update_status()
        self.assertIn("경계 2", self.window.chip_view.toolTip())        # counts persist across reconnects
        self.assertIn("ETV1 아님", self.view.last_error)


CHILD = r"""
import json, sys
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QAbstractButton, QApplication, QLabel, QScrollArea
from PySide6.QtCore import QPoint, Qt
from eyetoarm.gui import motion, theme
from eyetoarm.gui.demo import DemoLink
from eyetoarm.gui.relay_view import RelayViewReceiver
from eyetoarm.gui.window import MainWindow
app = QApplication([])
for font in ("C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/segoeui.ttf"):
    QFontDatabase.addApplicationFont(font)
theme.apply("dark")                    # the real app style: QSS padding/fonts set the true minimum widths
app.setStyleSheet(theme.QSS)
motion.set_reduced(True)
link = DemoLink()
view = RelayViewReceiver("203.0.113.255", 65535)   # long title; never started, so no socket is opened
view.start = lambda: None
window = MainWindow(link, view, demo=False)
window.show()
clipped, checked, outside = [], 0, []
for size in json.loads(sys.argv[1]):
    window.resize(*size)
    for page in range(3):
        window._show_page(page, animate=False)
        for _ in range(5):
            app.processEvents()
        window._update_status()   # the 250 ms refresh that elides long one-line labels to their current width
        app.processEvents()
        for child in window.pages.widget(page).findChildren(QLabel) + window.pages.widget(page).findChildren(QAbstractButton):
            if not child.isVisible():
                continue
            checked += 1
            text = child.text()
            need_h = child.heightForWidth(child.width()) if child.hasHeightForWidth() else child.minimumSizeHint().height()
            if isinstance(child, QLabel) and not child.wordWrap() and text and child.textFormat() != Qt.TextFormat.RichText:
                m = child.contentsMargins()
                need_w = child.fontMetrics().horizontalAdvance(text) + m.left() + m.right() + 2 * child.margin()
            else:
                need_w = child.minimumSizeHint().width() if isinstance(child, QAbstractButton) else 0
            if child.height() < need_h or child.width() < need_w:
                clipped.append([size, page, type(child).__name__, text[:40], [child.width(), child.height()], [need_w, need_h]])
        # Horizontal geometry guard: the page never scrolls sideways, so content, the video stage and the
        # link selector must end inside the scroll viewport (vertical scrolling is expected and not checked).
        scroll = window.pages.widget(page)
        if isinstance(scroll, QScrollArea):
            viewport = scroll.viewport()
            if scroll.widget().width() > viewport.width():
                outside.append([size, page, "page content", scroll.widget().width(), viewport.width()])
            if page == window.pages.currentIndex():
                for widget in [window.stage, *window.mode_buttons]:
                    if not widget.isVisible() or not viewport.isAncestorOf(widget):
                        continue
                    right = widget.mapTo(viewport, QPoint(widget.width(), 0)).x()
                    if right > viewport.width():
                        outside.append([size, page, type(widget).__name__, right, viewport.width()])
modes = sorted({b._mode for b in window.mode_buttons if b.isVisible() or True})
print(json.dumps({"dpr": window.devicePixelRatio(), "checked": checked, "clipped": clipped, "outside": outside,
                  "size": window.size().toTuple(), "buttons": len(window.mode_buttons), "modes": modes}))
window.close()
for item in (window, link, view):
    item.deleteLater()
app.processEvents()
"""


class RelayCli(unittest.TestCase):
    def run_window(self, *args):
        env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(GUI_SRC), "PYTHONIOENCODING": "utf-8"}
        return subprocess.run([sys.executable, "-B", "-m", "eyetoarm.gui.window", *args], env=env, cwd=str(ROOT),
                              capture_output=True, text=True, encoding="utf-8", timeout=60, stdin=subprocess.DEVNULL)

    def test_relay_is_the_default_and_host_is_validated_before_any_window_or_socket(self):
        result = self.run_window("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("relay(기본)", result.stdout)
        self.assertIn("47092", result.stdout)
        self.assertIn("47091", result.stdout)                    # legacy camera port unchanged
        for args in (("--view-host", "localhost"), ("--view", "camera", "--view-host", "c-pc"),
                     ("--view-port", "70000")):
            with self.subTest(args=args):
                result = self.run_window(*args)                  # parser error: exit 2, nothing started
                self.assertEqual(result.returncode, 2)
                self.assertIn("--view-", result.stderr)

    def test_relay_host_follows_the_command_host_unless_given(self):
        # Separate process; MainWindow is replaced so no window is shown and neither socket is ever started.
        probe = ("import json, sys, tempfile; import eyetoarm.gui.window as w\n"
                 "from PySide6.QtCore import QSettings\n"
                 "tmp = tempfile.TemporaryDirectory(); ini = tmp.name + '/s.ini'\n"   # never the user's registry (saved addresses would change the answer)
                 "w.settings = lambda: QSettings(ini, QSettings.Format.IniFormat)\n"
                 "def fake(link, view, demo=False, store=None):\n"
                 "    print(json.dumps([type(view).__name__, view.host, view.port, link.host])); raise SystemExit(0)\n"
                 "w.MainWindow = fake; w.main(json.loads(sys.argv[1]))\n")
        for args, expected in ((["--host", "10.1.2.3"], ["RelayViewReceiver", "10.1.2.3", 47092, "10.1.2.3"]),
                               (["--host", "10.1.2.3", "--view-host", "10.9.9.9", "--view-port", "47111"],
                                ["RelayViewReceiver", "10.9.9.9", 47111, "10.1.2.3"]),
                               (["--host", "10.1.2.3", "--view", "camera"],
                                ["GzViewReceiver", "127.0.0.1", 47091, "10.1.2.3"])):
            with self.subTest(args=args):
                result = self.run_window_code(probe, json.dumps(args))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout.strip().splitlines()[-1]), expected)

    def run_window_code(self, code, *args):
        env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(GUI_SRC), "PYTHONIOENCODING": "utf-8"}
        return subprocess.run([sys.executable, "-B", "-c", code, *args], env=env, cwd=str(ROOT),
                              capture_output=True, text=True, encoding="utf-8", timeout=60, stdin=subprocess.DEVNULL)


class RelayScale(unittest.TestCase):
    def test_relay_controls_fit_at_100_150_and_200_percent_with_the_real_theme(self):
        for scale in (1.0, 1.5, 2.0):
            with self.subTest(scale=scale):
                env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "QT_SCALE_FACTOR": str(scale),
                       "PYTHONPATH": str(GUI_SRC), "PYTHONIOENCODING": "utf-8"}
                result = subprocess.run([sys.executable, "-B", "-c", CHILD, json.dumps([[1180, 760], [1280, 800]])],
                                        env=env, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
                                        timeout=120, stdin=subprocess.DEVNULL)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout.strip().splitlines()[-1])
                self.assertEqual(report["dpr"], scale)
                self.assertEqual((report["buttons"], report["modes"]), (6, ["center", "detect", "raw"]))
                self.assertGreater(report["checked"], 100)
                self.assertEqual(report["size"], [1280, 800])        # target size kept, not silently widened
                self.assertEqual(report["clipped"], [])
                self.assertEqual(report["outside"], [])


if __name__ == "__main__":
    unittest.main()
