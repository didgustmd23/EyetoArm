"""D 분류 버튼 누름·응답 대기·결과 효과(FeedbackPill). 루프백 대역 상대만 쓴다(실제 C·로봇·Gazebo 없음).

효과는 화면 장식이다. 송신 여부·C 상태 판단은 test_d_client_v12 가 본다. 여기서는 효과 상태만 본다.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_d_client_v12 import V12Case, run_status, snap, until  # noqa: E402

from PySide6.QtGui import QColor  # noqa: E402

from eyetoarm.gui import motion, theme  # noqa: E402


class ButtonFeedback(V12Case):
    def setUp(self):
        super().setUp()
        from eyetoarm.gui.demo import DemoView
        from eyetoarm.gui.window import MainWindow
        self.window = MainWindow(self.link, DemoView())
        self.assertTrue(until(lambda: "START 송신" in self.window.sort_detail.text()))

    def tearDown(self):
        motion.set_reduced(False)
        self.window.close()
        self.window.deleteLater()
        super().tearDown()

    def test_unsent_click_shows_grey_press_only(self):
        button = self.window.stop_button   # IDLE allowed=["START"]: STOP 미전송
        button.click()
        self.assertGreater(button._wash, 0.0)
        self.assertEqual(button._wash_color, QColor(theme.IDLE))
        self.assertFalse(button.is_waiting())
        self.assertTrue(until(lambda: button._wash == 0.0))   # 한 번 옅어지고 흔적이 남지 않는다

    def test_sent_click_waits_then_shows_c_result_once(self):
        w, button = self.window, self.window.start_button
        replies = []
        self.peer.on_request = lambda r: replies.append(r) or []   # 응답을 붙잡아 대기 상태를 본다
        button.click()
        self.assertEqual(button._wash_color, QColor(theme.ACCENT))
        self.assertTrue(until(lambda: replies))
        self.assertTrue(button.is_waiting())
        request_id = replies[0]["request_id"]
        self.peer.push(run_status(request_id, "ACCEPTED", "RUN_STARTED", snap("RUNNING", 13, ("STOP",))))
        self.assertTrue(until(lambda: not button.is_waiting()))
        self.assertEqual(button._wash_color, QColor(theme.OK))
        self.assertEqual(button._ring_color, QColor(theme.OK))      # 결과 색 테두리
        self.assertIn(request_id, w._fx_seen)
        self.assertTrue(until(lambda: button._ring == 0.0 and button._trace < 0.0))   # 테두리 선도 흔적 없이 끝난다
        self.assertTrue(until(lambda: button._wash == 0.0))
        w._update_live_sort()   # 같은 결과로 다시 그려도 효과를 반복하지 않는다
        self.assertEqual(button._wash, 0.0)

    def test_rejected_reply_shows_fail_colour(self):
        button = self.window.start_button
        self.peer.on_request = lambda r: [run_status(r["request_id"], "REJECTED", "BUSY", snap())]
        button.click()
        self.assertTrue(until(lambda: button._wash_color == QColor(theme.FAIL)))
        self.assertFalse(button.is_waiting())

    def test_press_shrinks_and_traces_border(self):
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
        button = self.window.stop_button
        QTest.mousePress(button, Qt.MouseButton.LeftButton, pos=QPoint(10, 10))
        self.assertTrue(until(lambda: button._press == 1.0))         # 눌린 동안 작게
        QTest.mouseRelease(button, Qt.MouseButton.LeftButton, pos=QPoint(10, 10))
        self.assertGreaterEqual(button._trace, 0.0)                    # 굵은 선이 테두리를 돈다
        self.assertEqual(button._trace_color, QColor(theme.IDLE))     # 미전송 = 회색 선
        self.assertTrue(until(lambda: button._press == 0.0 and button._trace < 0.0))
        button.grab()                                                  # 효과 그리기 경로가 오류 없이 돈다

    def test_reduced_motion_leaves_no_animation(self):
        motion.set_reduced(True)
        button = self.window.start_button
        self.peer.on_request = lambda r: []
        button.click()
        self.assertEqual(button._wash, 0.0)                  # 퍼짐 없이 최종 상태
        self.assertLess(button._trace, 0.0)                   # 테두리 선도 돌지 않는다
        self.assertTrue(until(lambda: button.is_waiting()))
        self.assertFalse(button._spin_timer.isActive())     # 테두리만 고정, 돌지 않는다


class V12SetupTheme(ButtonFeedback.__mro__[1]):
    """TCP 1.2 모드 조작·설정 탭: 비활성 수동 이동 입력칸도 테마 색을 따른다. 번짐 지도는 화면 픽셀 크기다."""

    def setUp(self):
        super().setUp()
        from eyetoarm.gui.demo import DemoView
        from eyetoarm.gui.window import MainWindow
        self.window = MainWindow(self.link, DemoView())
        self.window.resize(1200, 800)
        self.window.show()

    def tearDown(self):
        self.window.set_theme("light", save=False)
        self.window.close()
        self.window.deleteLater()
        super().tearDown()

    def test_disabled_spin_boxes_follow_dark_theme(self):
        w = self.window
        w._show_page(0, animate=False)
        motion.set_reduced(True)
        w.set_theme("dark", save=False)
        motion.set_reduced(False)
        spin = w.target_inputs[0]
        for enabled in (False, True):   # 비활성·활성 모두 흰 기본 위젯이 아니다
            spin.setEnabled(enabled)
            image = spin.grab().toImage()
            self.assertLess(image.pixelColor(image.width() // 3, image.height() // 2).lightness(), 80, enabled)

    def test_ink_mask_matches_snapshot_pixels(self):
        w = self.window
        w.set_theme_fx("ink", save=False)
        w.set_theme("dark", save=False)
        wipe = w.theme_wipe
        self.assertEqual(wipe._ink_field.size(), wipe._pixmap.size())   # 저해상도 지도를 늘려 자르지 않는다
        wipe.stop()


if __name__ == "__main__":
    unittest.main()
