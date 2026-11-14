"""Current three-page GUI regression; offscreen, demo data, no live C or robot."""
import csv
import io
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QAbstractButton, QAbstractSpinBox, QApplication, QLabel, QWidget
from eyetoarm.gui.demo import DemoLink, DemoView
from eyetoarm.gui.client import TrafficEntry
from eyetoarm.gui.window import MainWindow

def until(predicate, timeout=3):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()

class GuiCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        # 프로세스 안에서만 등록한다(설치·설정 저장 없음). 한글 글리프 높이를 실제 화면과 맞춘다
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/malgun.ttf")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="eyetoarm-day3-gui-")
        self.link, self.view = DemoLink(), DemoView()
        self.window = MainWindow(self.link, self.view, demo=True)
        self.window.show()

    def tearDown(self):
        self.window.close()
        self.assertFalse(self.window._refresh.isActive())
        self.assertFalse(self.window._export_timer.isActive())
        self.assertFalse(self.link._poll.isActive())
        self.window.deleteLater()
        self.link.deleteLater()
        self.view.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        folder = self.temp.name
        self.temp.cleanup()
        self.assertFalse(Path(folder).exists())

    def test_pages_share_link_and_demo_keeps_real_move_disabled(self):
        for index in range(3):
            self.window._show_page(index, animate=False)
            self.assertEqual(self.window.pages.currentIndex(), index)
            self.assertIs(self.window.link, self.link)
        self.window.target_confirm.setChecked(True)
        self.window._update_command_controls()
        self.assertFalse(self.window.move_button.isEnabled())
        self.assertTrue(self.window.start_button.isEnabled())
        self.assertFalse(self.window.stop_button.isEnabled())

    def test_demo_stop_acceptance_precedes_stopped_and_logs_are_labeled(self):
        self.window.start_button.click()
        self.assertEqual(self.window.sort_demo.snapshot["state"], "STARTING")
        self.assertFalse(self.window.start_button.isEnabled())
        self.assertTrue(until(lambda: self.window.sort_demo.snapshot["state"] == "RUNNING"))
        self.window.stop_button.click()
        self.assertEqual(self.window.sort_demo.snapshot["state"], "STOPPING")
        self.assertIn("STOPPING", self.window.chip_sort.toolTip())
        self.assertTrue(until(lambda: self.window.sort_demo.snapshot["state"] == "STOPPED"))
        self.assertTrue(self.window.start_button.isEnabled())
        table = self.window.traffic_full
        summaries = [table.item(row, 3).text() for row in range(table.rowCount()) if table.item(row, 2).text() == "모의 분류"]
        self.assertTrue(summaries)
        self.assertTrue(all("실제 전송 없음" in text for text in summaries))

    def test_demo_recovery_requires_confirmation_and_does_not_restart(self):
        self.window.sort_fault_button.click()
        self.assertFalse(self.window.start_button.isEnabled())
        self.assertFalse(self.window.recover_button.isEnabled())
        self.window.sort_confirm.setChecked(True)
        self.assertTrue(self.window.recover_button.isEnabled())
        self.window.recover_button.click()
        self.assertEqual(self.window.sort_demo.snapshot["state"], "RECOVERING")
        self.assertFalse(self.window.sort_confirm.isChecked())
        self.assertTrue(until(lambda: self.window.sort_demo.snapshot["state"] == "STOPPED"))
        self.assertTrue(self.window.start_button.isEnabled())
        self.assertFalse(self.window.stop_button.isEnabled())

    def test_demo_disconnect_cancels_transition_and_resync_locks_unknown_job(self):
        self.window.start_button.click()
        self.window.sort_connection_button.click()
        self.assertFalse(self.window.sort_demo._timer.isActive())
        self.assertFalse(self.window.start_button.isEnabled())
        self.assertFalse(self.window.stop_button.isEnabled())
        self.window.sort_connection_button.click()
        self.assertTrue(self.window.sort_demo.synchronized)
        self.assertEqual(self.window.sort_demo.snapshot["state"], "RECOVERY_REQUIRED")
        self.assertEqual(self.window.sort_demo.snapshot["holding"], "HOLD_UNKNOWN")
        self.assertFalse(self.window.start_button.isEnabled())

    def test_demo_identical_readiness_keeps_fresh_without_new_record_rows(self):
        table = self.window.traffic_full
        self.link.set_connected(True)
        received = []
        self.link.status_received.connect(received.append)
        rows = table.rowCount()
        for _ in range(5):   # 1초 준비 조회 5회와 같다
            self.link._last_at = None
            self.link._emit_status()
        self.assertEqual(len(received), 5)
        self.assertIsNotNone(self.link.status_age_ms())
        self.assertFalse(self.link.is_stale())
        self.assertEqual(table.rowCount(), rows)
        self.link.set_connected(False)
        self.link.set_connected(True)
        kinds = [table.item(row, 2).text() for row in range(table.rowCount() - rows)]   # 새 행이 위에 쌓인다
        self.assertCountEqual(kinds, ["모의 끊김", "모의 접속", "준비 조회", "STATUS", "모의 분류"])

    def test_real_mode_keeps_unapproved_sorting_disabled(self):
        link, view = DemoLink(), DemoView()
        window = MainWindow(link, view, demo=False)
        try:
            self.assertIsNone(window.sort_demo)
            self.assertFalse(window.start_button.isEnabled())
            self.assertFalse(window.stop_button.isEnabled())
            self.assertFalse(window.recover_button.isEnabled())
            window._request_sort("START")
            self.assertIsNone(window.sort_demo)
        finally:
            window.close()
            window.deleteLater()
            link.deleteLater()
            view.deleteLater()

    def assert_setup_page_unclipped(self, window, size):
        window.resize(*size)
        window._show_page(0, animate=False)
        self.assertTrue(until(lambda: window.size().toTuple() == size))
        self.app.processEvents()
        scroll = window.pages.widget(0)
        self.assertLessEqual(scroll.widget().width(), scroll.viewport().width())
        for card in window._page_cards[0]:
            for child in card.findChildren(QWidget):
                if child.parentWidget() is not card or not child.isVisible() or not isinstance(child, (QLabel, QAbstractButton, QAbstractSpinBox)):
                    continue
                need = child.heightForWidth(child.width()) if child.hasHeightForWidth() else child.minimumSizeHint().height()
                self.assertGreaterEqual(child.height(), need, f"{size} {type(child).__name__} {getattr(child, 'text', lambda: '')()[:20]!r}")

    def test_setup_page_controls_and_demo_detail_fit_minimum_and_default_window(self):
        self.window.sort_fault_button.click()   # 세 줄짜리 상세 문구가 가장 긴 상태
        self.assertEqual(self.window.sort_detail.text().count("\n"), 2)
        for size in ((1180, 760), (1440, 960)):
            self.assert_setup_page_unclipped(self.window, size)
        link, view = DemoLink(), DemoView()
        window = MainWindow(link, view, demo=False)
        try:
            window.show()
            for size in ((1180, 760), (1440, 960)):
                self.assert_setup_page_unclipped(window, size)
            self.assertFalse(window.move_button.isEnabled())
        finally:
            window.close()
            window.deleteLater()
            link.deleteLater()
            view.deleteLater()

    def test_filter_csv_matches_visible_rows_and_guards_formula(self):
        table = self.window.traffic_full
        table.setRowCount(0)
        table.add_entry(TrafficEntry("12:00", "TX", "MOVE", "=SUM(1,2)", "raw,quoted"))
        table.add_entry(TrafficEntry("12:01", "RX", "STATUS", "ok", "reply"))
        self.window.record_filters[1].setChecked(True)
        path = Path(self.temp.name) / "selected.csv"
        self.assertTrue(self.window._start_export(str(path)))
        self.assertFalse(self.window.export_button.isEnabled())
        self.assertTrue(until(lambda: self.window._csv_file is None))
        raw = path.read_bytes()
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
        self.assertIn(b"\r\n", raw)
        rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1][1], "TX")
        self.assertEqual(rows[1][3], "'=SUM(1,2)")
        self.assertEqual(rows[1][4], "raw,quoted")
        self.assertEqual(self.window.export_progress.value, 100)
        self.assertTrue(self.window.export_button.isEnabled())

    def test_empty_export_failure_recovery_and_close_cancel(self):
        self.window.traffic_full.setRowCount(0)
        self.assertFalse(self.window._start_export(str(Path(self.temp.name) / "missing" / "no.csv")))
        path = Path(self.temp.name) / "empty.csv"
        self.assertTrue(self.window._start_export(str(path)))
        self.assertTrue(until(lambda: self.window._csv_file is None))
        self.assertEqual(len(list(csv.reader(io.StringIO(path.read_text(encoding="utf-8-sig"))))), 1)
        canceled = Path(self.temp.name) / "cancel.csv"
        self.assertTrue(self.window._start_export(str(canceled)))
        self.window.close()
        self.assertFalse(canceled.exists())
        self.assertFalse(self.window._export_timer.isActive())

if __name__ == "__main__":
    unittest.main()
