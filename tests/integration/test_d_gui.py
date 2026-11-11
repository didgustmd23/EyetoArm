"""대시보드·펜던트 창(PySide6)을 화면 없이 띄워 조작 → 신호 작성 → 기록 → 표시를 확인한다.

C 서버 없이 도는 시험이다. 실제 C 서버·로봇 검증이 아니다.
"""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtCore import QCoreApplication, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from eyetoarm.gui.bridge import Bridge  # noqa: E402
from eyetoarm.gui.dashboard import Dashboard  # noqa: E402
from eyetoarm.gui.pendant_window import PendantWindow  # noqa: E402
from eyetoarm.tcp.client import ClientConfig  # noqa: E402

APP = QApplication.instance() or QApplication([])


def until(predicate, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class GuiCase(unittest.TestCase):
    def setUp(self):
        self.bridge = Bridge(ClientConfig(port=1, connect_timeout_ms=200, reconnect_ms=100), None)
        self.bridge.start()
        self.window = Dashboard(self.bridge)
        self.window.show()
        self.pendant = PendantWindow(self.bridge)
        self.pendant.show()
        self.log = self.bridge.session.log

    def tearDown(self):
        self.pendant.close()
        self.window.close()  # 통신 스레드도 함께 멈춘다
        self.assertFalse(self.bridge._thread.is_alive())
        # Qt 객체를 QApplication 보다 먼저 지운다(종료 때 정리 순서 문제로 죽지 않게)
        for thing in (self.pendant, self.window, self.bridge):
            thing.deleteLater()
        self.pendant = self.window = self.bridge = self.log = None
        QCoreApplication.sendPostedEvents(None, 0)
        QCoreApplication.processEvents()

    def kinds(self):
        return [row["kind"] for row in list(self.log.rows)]

    def press(self, key, modifier=Qt.KeyboardModifier.NoModifier):
        QTest.keyClick(self.pendant, key, modifier)


class DashboardTests(GuiCase):
    def test_six_pages_and_initial_state(self):
        window = self.window
        self.assertEqual(window.stack.count(), 6)
        self.assertEqual([window.nav.button(i).text() for i in range(6)],
                         ["조작·설정", "운영 현황", "실시간 셀", "분류", "사람 제어", "평가·기록"])
        self.assertEqual(window.side.values["tcp"].text(), "연결 안 됨")
        self.assertEqual(window.side.values["ready"].text(), "미확인")  # 연결 전에는 준비로 보지 않는다
        self.assertIn("기록만 하고 보내지 않음", window.exec_note.text())

    def test_execute_without_server_composes_logs_and_shows_unsent(self):
        self.window.execute_button.click()
        self.assertTrue(until(lambda: "MOVE" in self.kinds()))
        row = next(r for r in self.log.rows if r["kind"] == "MOVE")
        self.assertEqual((row["send_state"], json.loads(row["raw"])["source_id"]), ("unsent", "pendant_manual"))
        self.assertTrue(until(lambda: self.window.log_ops.rowCount() >= 1))
        self.assertIn("미전송", self.window.log_ops.item(0, 4).text())
        self.assertTrue(until(lambda: self.window.stats and self.window.kpi.values["unsent"].text() == "1"))
        self.assertTrue(until(lambda: self.window.alarms.rowCount() >= 1))
        self.assertIn("MOVE 미전송", self.window.exec_note.text())

    def test_manual_target_fields_drive_the_signal(self):
        for axis, text in (("X", "500"), ("Y", "120.5"), ("RZ", "45")):
            self.window.target_edits[axis].setText(text)
        self.window.on_apply_target()
        self.assertTrue(until(lambda: self.bridge.session.pendant.cart["X"] == 500.0))
        self.window.execute_button.click()
        self.assertTrue(until(lambda: "MOVE" in self.kinds()))
        move = json.loads(next(r for r in self.log.rows if r["kind"] == "MOVE")["raw"])
        self.assertEqual((move["position_mm"], move["theta_deg"]), ([500.0, 120.5], 45.0))
        self.assertTrue(until(lambda: self.window.twin_ops.goal is not None and self.window.twin_ops.goal["X"] == 500.0))

    def test_row_click_opens_detail_sheet_with_raw_signal(self):
        self.window.execute_button.click()
        self.assertTrue(until(lambda: self.window.log_all.rowCount() >= 1))
        self.window.go(5)
        index = next(i for i in range(self.window.log_all.rowCount()) if self.window.log_all.item(i, 2).text() == "MOVE")
        self.window.show_detail(self.window.log_all.row_data(index))
        self.assertTrue(self.window.overlays.modal_open)
        QTest.keyClick(self.window, Qt.Key.Key_Escape)
        self.assertFalse(self.window.overlays.modal_open)

    def test_export_and_clear_keep_the_record(self):
        self.window.execute_button.click()
        self.assertTrue(until(lambda: "MOVE" in self.kinds()))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "out.csv"
            self.window.export_to(str(path))
            text = path.read_text(encoding="utf-8-sig")
        self.assertIn("pendant_manual", text)
        until(lambda: self.window.log_all.rowCount() >= 1)
        self.window.on_clear()
        self.assertEqual(self.window.log_all.rowCount(), 0)
        self.assertTrue(self.log.rows)  # 화면만 비우고 기록은 그대로

    def test_connect_attempt_to_missing_server_is_logged_not_ready(self):
        self.window.port_edit.setText("1")
        self.window.on_connect()
        self.assertTrue(until(lambda: "연결 실패" in self.kinds(), 4))
        self.assertTrue(until(lambda: self.window.side.values["tcp"].text() in ("재접속 대기", "접속 중")))
        self.assertEqual(self.window.side.values["ready"].text(), "미확인")
        self.window.on_connect()  # 연결 해제
        self.assertTrue(until(lambda: self.window.side.values["tcp"].text() == "연결 안 됨"))


class PendantTests(GuiCase):
    def test_keys_do_nothing_until_enabled(self):
        self.assertIn("먼저 인에이블을 켜세요", self.pendant.message.text())
        self.press(Qt.Key.Key_W)
        self.assertTrue(until(lambda: "인에이블 꺼짐" in self.pendant.message.text()))
        self.assertIsNotNone(self.pendant.overlays._toast)  # 왜 안 움직이는지 바로 안내한다
        self.pendant.jog_buttons[0][1].click()  # 버튼도 눌리지만 신호는 만들지 않는다
        self.pendant.start.click()
        time.sleep(0.05)
        self.assertTrue(until(lambda: "START 불가" in self.pendant.message.text()))
        self.assertNotIn("PENDANT JOG", self.kinds())
        self.assertNotIn("MOVE", self.kinds())
        self.assertEqual(self.pendant.enable.text().splitlines()[0], "인에이블 켜기")

    def test_buttons_jog_after_enable_click(self):
        self.pendant.enable.click()
        self.assertTrue(until(lambda: self.pendant.state["pendant"]["enabled"]))
        self.assertEqual(self.pendant.enable.text().splitlines()[0], "인에이블 끄기")
        self.pendant.jog_buttons[0][1].click()   # X +
        self.pendant.jog_buttons[1][0].click()   # Y −
        self.pendant.grip_close.click()
        cart = self.bridge.session.pendant.cart
        self.assertTrue(until(lambda: (cart["X"], cart["Y"], self.bridge.session.pendant.grip_mm) == (475.0, 155.0, 80.0)))
        self.assertTrue(until(lambda: self.pendant.axis_values[0].text() == "475.0 mm"))

    def test_wasd_arrows_jog_and_show_composed_signal(self):
        self.press(Qt.Key.Key_E)
        self.assertTrue(until(lambda: self.pendant.state["pendant"]["enabled"]))
        for key in (Qt.Key.Key_W, Qt.Key.Key_A, Qt.Key.Key_Up, Qt.Key.Key_Q):
            self.press(key)
        cart = self.bridge.session.pendant.cart
        self.assertTrue(until(lambda: (cart["X"], cart["Y"], cart["Z"], cart["RZ"]) == (475.0, 165.0, 220.0, 175.0)))
        self.assertTrue(until(lambda: self.pendant.axis_values[0].text() == "475.0 mm"))
        self.assertTrue(until(lambda: "PENDANT JOG" in self.pendant.side["signal"].text()))
        self.assertIn('"contract":"pendant-draft/0.1"', self.pendant.raw.text().replace("​", ""))
        self.assertIn("미전송", self.pendant.side["signal"].text())  # 전송 경로 미합의: 작성·기록만
        self.assertTrue(until(lambda: self.window.twin_ops.goal["X"] == 475.0))  # 대시보드 그림도 같은 목표를 따른다

    def test_numpad_gripper_speed_and_frame_keys(self):
        self.press(Qt.Key.Key_E)
        until(lambda: self.pendant.state["pendant"]["enabled"])
        self.press(Qt.Key.Key_8, Qt.KeyboardModifier.KeypadModifier)
        self.assertTrue(until(lambda: self.bridge.session.pendant.grip_mm == 80.0))
        self.press(Qt.Key.Key_8)  # 윗줄 숫자는 그리퍼에 쓰지 않는다
        self.press(Qt.Key.Key_Plus)
        self.assertTrue(until(lambda: self.bridge.session.pendant.speed_ratio == 0.5))
        self.assertEqual(self.bridge.session.pendant.grip_mm, 80.0)
        self.press(Qt.Key.Key_Tab)
        self.assertTrue(until(lambda: self.pendant.state["pendant"]["frame"] == "joint"))
        self.press(Qt.Key.Key_D)
        self.press(Qt.Key.Key_W)
        self.assertTrue(until(lambda: self.bridge.session.pendant.joint["J2"] == -89.0))
        self.assertTrue(self.pendant.jog_buttons[5][0].isVisible())  # 관절 좌표계에서는 J6 까지 6줄

    def test_start_sends_tcp_move_attempt_and_stop_requests_cancel(self):
        self.press(Qt.Key.Key_E)
        until(lambda: self.pendant.state["pendant"]["enabled"])
        self.press(Qt.Key.Key_Return)
        self.assertTrue(until(lambda: "MOVE" in self.kinds()))
        row = next(r for r in self.log.rows if r["kind"] == "MOVE")
        self.assertEqual((row["channel"], row["send_state"]), ("tcp1.1", "unsent"))
        self.press(Qt.Key.Key_Space)
        self.assertTrue(until(lambda: "PENDANT STOP" in self.kinds()))

    def test_estop_locks_and_help_popup_toggles(self):
        self.press(Qt.Key.Key_E)
        until(lambda: self.pendant.state["pendant"]["enabled"])
        self.pendant.estop.click()
        self.assertTrue(until(lambda: self.pendant.state["pendant"]["estop"]))
        self.press(Qt.Key.Key_E)
        self.press(Qt.Key.Key_W)
        time.sleep(0.05)
        self.assertTrue(until(lambda: not self.pendant.state["pendant"]["enabled"]))
        self.assertEqual(self.bridge.session.pendant.cart["X"], 470.0)
        self.press(Qt.Key.Key_H)
        self.assertTrue(self.pendant.overlays.modal_open)
        self.press(Qt.Key.Key_H)
        self.assertFalse(self.pendant.overlays.modal_open)


if __name__ == "__main__":
    unittest.main()
