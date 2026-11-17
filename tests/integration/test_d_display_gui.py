import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QAbstractButton, QApplication

from eyetoarm.gui.demo import DemoLink, DemoView
from eyetoarm.gui.window import MainWindow


class DisplayGuiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/malgun.ttf")

    def test_generation_transition_clears_visible_measurement(self):
        link, view = DemoLink(), DemoView()
        window = MainWindow(link, view, demo=True)
        try:
            view._emit_frame()
            self.assertIn("12.0", window.measure_values[1].text())
            self.assertTrue(window.display.advance_generation("mock-link", run_id="mock-run",
                            generation=2, order=1, detector_config_id="mock-config"))
            self.assertEqual(window.measure_values[1].text(), "—")
            self.assertEqual(window.stage.link._mode, "empty")
        finally:
            window.close()
            window.deleteLater()
            link.deleteLater()
            view.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def test_demo_channels_selection_disconnect_and_cleanup(self):
        link, view = DemoLink(), DemoView()
        window = MainWindow(link, view, demo=True)
        try:
            view._emit_frame()
            self.assertEqual(window.stage.main._mode, "live")
            self.assertEqual(window.stage.gripper._mode, "live")
            self.assertEqual(window.stage.link._mode, "live")
            self.assertIn("12.0", window.stage.popup.values["mm"].text())
            self.assertEqual(window.measure_values[1].text(), window.stage.popup.values["mm"].text())
            self.assertIn("mock-link", window.input_values[0].text())
            next(b for b in window.findChildren(QAbstractButton) if b.text() == "center 중심점").click()
            self.assertEqual(window.display.mode, "center")
            window.display.frame("gripper")["received"] -= 10
            window._update_display()
            self.assertEqual(window.stage.gripper._mode, "empty")
            self.assertIn("12.0", window.measure_values[1].text())
            self.assertFalse(window.move_button.isEnabled())
            window.display.invalidate("mock-link", run_id="mock-run", generation=1, order=10, reason="모의 무효", detector_config_id="mock-config")
            self.assertEqual(window.stage.popup.values["mm"].text(), "—")
            self.assertEqual(window.measure_values[1].text(), "—")
            self.assertEqual(window.stage.popup.values["detail"].text(), "모의 무효")
            self.assertEqual(window.stage.link._mode, "empty")
            self.assertIsNotNone(window.display.frames.get(("link", "raw")))
            self.assertEqual(window.stage.main._mode, "live")
            self.assertEqual(window.stage.gripper._mode, "empty")  # 위에서 노후 처리한 상태 유지
            next(b for b in window.findChildren(QAbstractButton) if b.text() == "raw 원본").click()
            self.assertEqual(window.stage.link._mode, "live")
            window.display.set_connected(False)
            self.assertEqual(window.stage.link._mode, "empty")
        finally:
            window.close()
        self.assertFalse(view._timer.isActive())
        self.assertFalse(link._poll.isActive())
        self.assertFalse(window._refresh.isActive())
        window.deleteLater()
        link.deleteLater()
        view.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def test_minimum_window_popup_and_views_are_reachable(self):
        link, view = DemoLink(), DemoView()
        window = MainWindow(link, view, demo=True)
        try:
            window.resize(1180, 760)
            window.show()
            view._emit_frame()
            window.detail_button.click()
            self.app.processEvents()
            stage = window.stage
            self.assertTrue(stage.rect().contains(stage.popup.geometry()))
            self.assertTrue(stage.rect().contains(stage.gripper.geometry()))
            self.assertTrue(stage.rect().contains(stage.link.geometry()))
            viewport = window.pages.widget(1).viewport()
            for item in (stage.popup, stage.gripper, stage.link):
                bounds = item.rect().translated(item.mapTo(viewport, item.rect().topLeft()))
                self.assertTrue(viewport.rect().contains(bounds), (item.objectName(), bounds, viewport.rect()))
            for value in stage.popup.values.values():
                self.assertLessEqual(value.fontMetrics().horizontalAdvance(value.text()), value.width())
            window._show_page(0, animate=False)
            self.app.processEvents()
            setup = window.pages.widget(0)
            self.assertLessEqual(setup.widget().width(), setup.viewport().width())
            for value in window.input_values:
                self.assertLessEqual(value.fontMetrics().horizontalAdvance(value.text()), value.width())
            reason = "모의 무효 · " + "아주 긴 무효 사유 문장 " * 12
            window.display.invalidate("mock-link", run_id="mock-run", generation=1, order=10_000,
                                      reason=reason, detector_config_id="mock-config")
            for page in (0, 1):
                window._show_page(page, animate=False)
                self.app.processEvents()
                window._update_display()
                self.app.processEvents()
            for value in (stage.popup.values["detail"], window.input_values[2]):
                self.assertEqual(value.toolTip(), reason)
                self.assertNotEqual(value.text(), reason)
                self.assertLessEqual(value.fontMetrics().horizontalAdvance(value.text()), value.width())
        finally:
            window.close()
            window.deleteLater()
            link.deleteLater()
            view.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


    def test_stage_swaps_aux_with_main_by_click_and_keyboard(self):
        from PySide6.QtCore import QPoint, QPropertyAnimation, Qt
        from PySide6.QtTest import QTest
        from eyetoarm.gui import motion, theme

        reduced = motion.is_reduced()
        motion.set_reduced(True)   # 움직임 없이 바로 최종 배치를 확인한다
        link, view = DemoLink(), DemoView()
        window = MainWindow(link, view, demo=True)
        try:
            window.resize(1180, 760)
            window.show()
            window._show_page(1, animate=False)
            view._emit_frame()
            window._update_display()
            self.app.processEvents()
            stage = window.stage
            feeds = (stage.main, stage.gripper, stage.link)

            def check(expected):
                self.assertEqual(stage.slots, expected)
                big, aux = stage.slots[0], stage.slots[1:]
                self.assertEqual(big.geometry(), stage.rect())
                self.assertFalse(big._clickable)
                self.assertFalse(big._compact)
                for item in aux:
                    self.assertTrue(item._clickable)
                    self.assertTrue(item._compact)
                    self.assertTrue(stage.rect().contains(item.geometry()))
                    self.assertLess(item.width(), stage.width() // 2)
                self.assertFalse(aux[0].geometry().intersects(aux[1].geometry()))
                order = stage.children()   # 뒤일수록 위. 큰 칸 < 보조 칸 < 팝업
                self.assertLess(order.index(big), min(order.index(a) for a in aux))
                self.assertGreater(order.index(stage.popup), max(order.index(a) for a in aux))
                self.assertEqual(sorted(map(id, stage.slots)), sorted(map(id, feeds)))

            check([stage.main, stage.gripper, stage.link])
            QTest.mouseClick(stage.gripper, Qt.MouseButton.LeftButton, pos=QPoint(10, 10))
            check([stage.gripper, stage.main, stage.link])
            QTest.mouseClick(stage.gripper, Qt.MouseButton.LeftButton, pos=QPoint(10, 10))   # 큰 칸 누름은 무시
            check([stage.gripper, stage.main, stage.link])
            QTest.mouseClick(stage.link, Qt.MouseButton.LeftButton, pos=QPoint(10, 10))
            check([stage.link, stage.main, stage.gripper])

            stage.main.setFocus()
            QTest.keyClick(stage.main, Qt.Key.Key_Return)
            check([stage.main, stage.link, stage.gripper])
            self.assertTrue(stage.link.hasFocus())   # 같은 보조 칸 자리에 초점이 남는다
            QTest.keyClick(stage.link, Qt.Key.Key_Space)
            check([stage.link, stage.main, stage.gripper])

            before = stage.size()
            window.resize(1500, 900)
            self.app.processEvents()
            self.assertNotEqual(before, stage.size())
            check([stage.link, stage.main, stage.gripper])

            motion.set_reduced(False)   # 실제 움직임: 빠른 연속 교체 + 움직임 도중 크기 변경
            QTest.mouseClick(stage.gripper, Qt.MouseButton.LeftButton, pos=QPoint(10, 10))
            self.assertIsNotNone(stage.link.findChild(QPropertyAnimation, "geometry_motion"))
            QTest.mouseClick(stage.main, Qt.MouseButton.LeftButton, pos=QPoint(10, 10))
            before = stage.size()
            window.resize(1180, 760)   # 움직임 도중 크기 변경: 옛 목표로 가던 움직임은 멈추고 지운다
            self.app.processEvents()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            self.assertNotEqual(before, stage.size())
            for item in feeds:
                self.assertEqual(item.findChildren(QPropertyAnimation), [])
            check([stage.main, stage.gripper, stage.link])
            QTest.mouseClick(stage.link, Qt.MouseButton.LeftButton, pos=QPoint(10, 10))
            QTest.mouseClick(stage.gripper, Qt.MouseButton.LeftButton, pos=QPoint(10, 10))
            QTest.qWait(theme.MS_POPUP + 200)   # 움직임이 끝난 뒤 최종 배치
            check([stage.gripper, stage.link, stage.main])
            motion.set_reduced(True)

            view._emit_frame()   # 갱신은 자리와 무관하게 출처별 위젯으로 간다
            window._update_display()
            self.app.processEvents()
            for item, channel in ((stage.main, "fixed"), (stage.gripper, "gripper"), (stage.link, "link")):
                self.assertEqual(item._mode, "live")
                self.assertEqual(item._display_item["source_id"], f"mock-{channel}")
                self.assertEqual(item._display_item["image_seq"], view.frame_count)
                self.assertIs(item._image, item._display_item["image"])
            self.assertEqual(stage.main.accessibleName(), "탑뷰")
            self.assertEqual(stage.gripper.accessibleName(), "그리퍼")
            self.assertEqual(stage.link.accessibleName(), "링크")
        finally:
            motion.set_reduced(reduced)
            window.close()
            window.deleteLater()
            link.deleteLater()
            view.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


if __name__ == "__main__":
    unittest.main()
