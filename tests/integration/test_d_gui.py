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
from PySide6.QtWidgets import QApplication
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

    def test_pages_share_link_and_demo_never_enables_move_or_sorting(self):
        for index in range(3):
            self.window._show_page(index, animate=False)
            self.assertEqual(self.window.pages.currentIndex(), index)
            self.assertIs(self.window.link, self.link)
        self.window.target_confirm.setChecked(True)
        self.window._update_command_controls()
        self.assertFalse(self.window.move_button.isEnabled())
        self.assertFalse(self.window.start_button.isEnabled())
        self.assertFalse(self.window.stop_button.isEnabled())

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
