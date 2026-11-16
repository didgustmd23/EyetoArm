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


if __name__ == "__main__":
    unittest.main()
