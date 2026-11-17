"""Day7 GUI checks: execute->ACK metric binding, mock responsiveness, DPI 150/200 %, keyboard and non-color status.

Offscreen only. No socket is opened (StatusLink.start is replaced), no C server, robot or camera is used.
The heartbeat numbers are a local MOCK sample, not a production SLA or the A 100-run measurement.
"""
import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
GUI_SRC = ROOT / "dashboard-client" / "src"
sys.path.insert(0, str(GUI_SRC))
from PySide6.QtCore import QCoreApplication, QElapsedTimer, QEvent, QEventLoop, QObject, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDoubleSpinBox
from eyetoarm.gui import motion
from eyetoarm.gui.client import ACK_METRIC_BOUNDARY, StatusLink
from eyetoarm.gui.demo import DemoLink, DemoView
from eyetoarm.gui.window import MainWindow
from eyetoarm.tcp import protocol

READY = {"version": "1.1", "type": "STATUS", "robot_ready": True, "not_ready_reasons": [],
         "active_command_id": None, "server_time_ms": 1}


def wire(message):
    return protocol.encode(message)


def target():
    now = time.monotonic_ns() // 1_000_000
    return protocol.MoveTarget((100.0, 200.0), 30.0, "manual", 1, 1, "PC_RECEIVE", "cal", created_monotonic_ms=now)


class Day7Window(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        motion.set_reduced(True)   # deterministic show/hide; animation timing is not under test here

    @classmethod
    def tearDownClass(cls):
        motion.set_reduced(False)

    def make(self, link, view, demo=False):
        window = MainWindow(link, view, demo=demo)
        self.addCleanup(self.close, window, link, view)
        window.show()
        return window

    def close(self, window, link, view):
        window.close()
        self.assertFalse(window._refresh.isActive())
        for item in (window, link, view):
            item.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()

    def socketless_link(self):
        link = StatusLink(log_path=None, stable_count=3, stable_ms=5000)
        link.start = Mock()                       # MainWindow calls start(); never connect anywhere
        link._send = Mock(return_value=True)
        link.state, link._conn_id = "connected", 1
        link._handle_line(wire(READY))
        return link

    def command_status(self, command_id, state):
        return wire({**READY, "robot_ready": False, "not_ready_reasons": ["COMMAND_ACTIVE"],
                     "active_command_id": command_id, "command_id": command_id, "state": state})

    def test_ack_metric_row_waits_then_shows_first_ack_and_ui_stays_responsive(self):
        clock = [5_000]
        with patch("eyetoarm.gui.client._now_ms", lambda: clock[0]):
            link = self.socketless_link()
            window = self.make(link, DemoView())
            value, note = window.ack_metric_value, window.ack_metric_note
            self.assertEqual((value.text(), note.text()), ("—", "명령 없음"))
            self.assertEqual(value.toolTip(), ACK_METRIC_BOUNDARY)
            command_id = link.execute(target())
            link._handle_line(wire({**READY, "server_time_ms": 2}))   # fresh readiness -> MOVE sent
            self.assertEqual(value.text(), "—")
            self.assertIn("ACK 대기", note.text())

            # ACK is delayed: the GUI event loop (incl. the 250 ms status refresh) must keep running.
            gaps, elapsed = [], QElapsedTimer()
            heartbeat = QTimer(window, interval=20)
            elapsed.start()

            def beat():
                gaps.append(elapsed.restart())
            heartbeat.timeout.connect(beat)
            heartbeat.start()
            loop = QEventLoop()
            QTimer.singleShot(600, loop.quit)
            loop.exec()
            heartbeat.stop()
            self.assertGreaterEqual(len(gaps), 10)
            p95 = sorted(gaps)[int(len(gaps) * 0.95) - 1]
            print(f"\n[MOCK-only heartbeat] n={len(gaps)} p95={p95} ms max={max(gaps)} ms (20 ms timer, ACK delayed)",
                  file=sys.stderr)
            self.assertLess(max(gaps), 1000)   # generous freeze guard, not a target

            clock[0] = 5_180
            link._handle_line(self.command_status(command_id, "ACK"))
            self.assertEqual(value.text(), "180 ms")
            clock[0] = 9_999
            link._handle_line(self.command_status(command_id, "ACK"))   # duplicate ACK
            self.assertEqual(value.text(), "180 ms")

    def test_resend_shows_dash_with_reason_and_demo_never_fakes_a_value(self):
        link = self.socketless_link()
        window = self.make(link, DemoView())
        command_id = link.execute(target())
        link._handle_line(wire({**READY, "server_time_ms": 2}))
        link._ack_timeout()                                          # one resend
        link._handle_line(self.command_status(command_id, "ACK"))
        self.assertEqual(window.ack_metric_value.text(), "—")
        self.assertIn("재전송 뒤 수락", window.ack_metric_note.text())

        demo_link, demo_view = DemoLink(), DemoView()
        demo = self.make(demo_link, demo_view, demo=True)
        demo.start_button.click()
        self.app.processEvents()
        self.assertEqual(demo.ack_metric_value.text(), "—")
        self.assertIn("모의", demo.ack_metric_note.text())

    def test_keyboard_reaches_aux_view_swaps_and_escape_closes_popup(self):
        window = self.make(DemoLink(), DemoView())
        window.resize(1440, 960)
        window._show_page(1, animate=False)
        window.activateWindow()
        self.assertTrue(QTest.qWaitForWindowActive(window, 2000))
        stage = window.stage
        window.detail_button.setFocus(Qt.FocusReason.OtherFocusReason)
        for _ in range(80):   # Tab order must reach the auxiliary view without a mouse
            if QApplication.focusWidget() is stage.gripper:
                break
            QTest.keyClick(QApplication.focusWidget() or window, Qt.Key.Key_Tab)
        self.assertIs(QApplication.focusWidget(), stage.gripper)
        QTest.keyClick(stage.gripper, Qt.Key.Key_Return)
        self.assertIs(stage.slots[0], stage.gripper)
        self.assertIs(QApplication.focusWidget(), stage.main)          # focus stays on the same aux position

        window.detail_button.setFocus(Qt.FocusReason.OtherFocusReason)
        QTest.keyClick(window.detail_button, Qt.Key.Key_Space)
        self.assertTrue(stage.popup.isVisible())
        QTest.keyClick(QApplication.focusWidget() or window, Qt.Key.Key_Escape)
        self.assertFalse(window.detail_button.isChecked())
        self.assertFalse(stage.popup.isVisible())
        self.assertIs(QApplication.focusWidget(), window.detail_button)

        # Popup closed: the window shortcut is off, so Escape still reaches an input field (not swallowed).
        window._show_page(0, animate=False)
        spin = window.findChild(QDoubleSpinBox)
        spin.setFocus(Qt.FocusReason.OtherFocusReason)
        received = []

        class Spy(QObject):
            def eventFilter(self, _obj, event):
                if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
                    received.append(True)
                return False
        spy = Spy(window)
        spin.installEventFilter(spy)
        self.assertFalse(window.close_popup_shortcut.isEnabled())
        QTest.keyClick(spin, Qt.Key.Key_Escape)
        self.assertEqual(received, [True])
        spin.removeEventFilter(spy)


CHILD = r"""
import json, sys
from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QAbstractButton, QAbstractSpinBox, QApplication, QLabel, QWidget
from eyetoarm.gui import motion
from eyetoarm.gui.demo import DemoLink, DemoView
from eyetoarm.gui.widgets import StatusChip
from eyetoarm.gui.window import MainWindow
app = QApplication([])
QFontDatabase.addApplicationFont("C:/Windows/Fonts/malgun.ttf")
motion.set_reduced(True)
link, view = DemoLink(), DemoView()
window = MainWindow(link, view, demo=False)
window.show()
clipped, checked = [], 0
for size in json.loads(sys.argv[1]):
    window.resize(*size)
    for page in range(3):
        window._show_page(page, animate=False)
        for _ in range(5):
            app.processEvents()
        for card in window._page_cards[page]:
            for child in card.findChildren(QWidget):
                if not child.isVisible() or not isinstance(child, (QLabel, QAbstractButton, QAbstractSpinBox)):
                    continue
                checked += 1
                text = getattr(child, "text", lambda: "")()
                need_h = child.heightForWidth(child.width()) if child.hasHeightForWidth() else child.minimumSizeHint().height()
                if isinstance(child, QLabel) and not child.wordWrap() and text and not child.textFormat() == Qt.TextFormat.RichText:
                    # Single-line label: the current text (already elided by the window where it does that on purpose,
                    # full text in its tooltip) must fit the label's own content box.
                    margins = child.contentsMargins()
                    need_w = child.fontMetrics().horizontalAdvance(text) + margins.left() + margins.right() + 2 * child.margin()
                    have_w = child.width()
                elif isinstance(child, QAbstractButton):
                    need_w, have_w = child.minimumSizeHint().width(), child.width()
                else:   # wrapped labels: width is handled by wrapping, height checked below; spin boxes: Qt minimum
                    need_w, have_w = child.minimumSizeHint().width(), child.width()
                if child.height() < need_h or have_w < need_w:
                    clipped.append([size, page, type(child).__name__, text[:30], [child.width(), child.height()],
                                    [need_w, need_h]])
link.state = "disconnected"          # force problem states so the non-color check is not vacuous
window._update_status()
chips = [[chip._caption, chip._level, chip._text, chip.accessibleDescription()] for chip in window.findChildren(StatusChip)]
print(json.dumps({"dpr": window.devicePixelRatio(), "size": window.size().toTuple(), "clipped": clipped,
                  "checked": checked, "chips": chips}))
window.close()
for item in (window, link, view):
    item.deleteLater()
app.processEvents()
"""


SIZES = [[1180, 760], [1280, 800]]   # minimum window and the planned 1280x800 check, in logical pixels


class Day7Scale(unittest.TestCase):
    def run_child(self, scale):
        env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "QT_SCALE_FACTOR": str(scale),
               "PYTHONPATH": str(GUI_SRC), "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run([sys.executable, "-B", "-c", CHILD, json.dumps(SIZES)], env=env, cwd=str(ROOT), capture_output=True,
                                text=True, encoding="utf-8", timeout=120, stdin=subprocess.DEVNULL)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout.strip().splitlines()[-1])

    def test_150_and_200_percent_keep_text_unclipped_and_status_not_color_only(self):
        for scale in (1.5, 2.0):
            with self.subTest(scale=scale):
                report = self.run_child(scale)
                self.assertEqual(report["dpr"], scale)
                self.assertEqual(report["size"], SIZES[-1])
                self.assertGreater(report["checked"], 100)
                self.assertEqual(report["clipped"], [])
                self.assertGreaterEqual(len(report["chips"]), 5)
                self.assertTrue(any(level == "fail" for _caption, level, _text, _d in report["chips"]))
                for caption, level, text, description in report["chips"]:
                    self.assertTrue(caption)
                    if level in ("fail", "warn"):   # problems must be readable without the dot color
                        self.assertNotIn(text, ("", "—"))
                        self.assertEqual(description, text)


if __name__ == "__main__":
    unittest.main()
