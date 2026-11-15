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
        self.assertCountEqual(kinds, ["모의 끊김", "모의 접속", "준비 조회", "STATUS", "모의 분류", "모의 집계"])

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

    # ── 5일차 모의 공정·집계 ──────────────────────────────────────────────

    def rows(self, kind="모의 공정"):
        table = self.window.traffic_full
        return [table.item(row, 3).text() for row in range(table.rowCount()) if table.item(row, 2).text() == kind]

    def counts_text(self):
        return {key: value.text() for key, value in self.window.count_values.items()}

    def run_to_running(self):
        w = self.window
        self.assertFalse(w.sort_cycle_button.isEnabled())   # IDLE에서는 시나리오 불가
        w.start_button.click()
        self.assertTrue(until(lambda: w.sort_demo.snapshot["state"] == "RUNNING"))
        self.assertTrue(w.sort_cycle_button.isEnabled())
        w._cycle_timer.setInterval(5)

    def test_demo_cycle_counts_three_bins_with_stage_ids_and_labeled_records(self):
        w = self.window
        self.run_to_running()
        self.assertEqual(w.count_tag.text(), "모의 · 동기화됨")
        w.sort_cycle_button.click()
        self.assertFalse(w.sort_cycle_button.isEnabled())
        self.assertTrue(until(lambda: not w.supplier.active and not w._cycle_timer.isActive()))
        self.assertEqual(self.counts_text(), {"BOLT_A": "1", "BOLT_B": "1", "NUT": "1", "QUARANTINE": "0", "EXCEPTION": "0 / 0"})
        self.assertIn("P-0002 · BOLT_B", w.bolt_caption.text())
        self.assertIn("P-0003 · NUT", w.nut_caption.text())
        self.assertEqual((w.bolt_stages._current, w.nut_stages._current), (3, 1))
        self.assertIn("인식 NUT 인식", w.process_detail.text())
        self.assertIn("보유 DETACHED", w.process_detail.text())
        records = self.rows()
        self.assertEqual(sum(" · 반영 · " in text for text in records), 12)   # 볼트 2×5 + 너트 2
        self.assertTrue(all(text.startswith("모의 · 실제 아님") for text in records))
        self.assertTrue(w.sort_cycle_button.isEnabled())
        # 중복 재전달·실패는 처분 행으로 남고 수량은 그대로
        duplicate = dict(kind="ARRIVED", scope=w.supplier.scope, epoch=1, seq=4, part="P-0001", spec="BOLT_A", bin="BOLT_A", verdict="OK")
        self.assertEqual(w.feed_demo_event(duplicate).code, "중복")
        self.assertEqual(w.feed_demo_event(w.supplier.make("FAILED", "P-0009", 1, reason="집기 실패")).code, "반영")
        self.assertEqual(w.feed_demo_event({"kind": "ARRIVED"}).code, "형식오류")
        self.assertIn("실패: 최근 P-0009 · 집기 실패", w.process_detail.text())
        self.assertIn("집기 실패", w.stage_note.toolTip())
        self.assertIn("중복", self.rows()[2])
        self.assertIn("형식오류", self.rows()[0])
        self.assertEqual(self.counts_text()["BOLT_A"], "1")
        # 표 지우기·CSV는 집계를 바꾸지 않는다. CSV에는 모의 표기·처분이 남는다
        path = Path(self.temp.name) / "demo.csv"
        self.assertTrue(w._start_export(str(path)))
        self.assertTrue(until(lambda: w._csv_file is None))
        exported = path.read_text(encoding="utf-8-sig")
        self.assertIn("모의 · 실제 아님", exported)
        self.assertIn("중복", exported)
        w.clear_records_button.click()
        self.assertEqual(w.traffic_full.rowCount(), 0)
        self.assertEqual(self.counts_text()["BOLT_A"], "1")
        self.assertEqual(w.tally.counts()["NUT"], 1)
        for _ in range(501):
            w.feed_demo_event(duplicate)
        self.assertEqual(w.traffic_full.rowCount(), 500)
        self.assertEqual(w.tally.counts()["NUT"], 1)
        self.assertEqual(w.tally.counts()["BOLT_A"], 1)

    def test_demo_cycle_stops_on_stop_disconnect_and_recovery_keeping_permissions(self):
        w = self.window
        self.run_to_running()
        w.sort_cycle_button.click()
        self.assertTrue(until(lambda: w.tally.counts()["BOLT_A"] == 1))
        w.stop_button.click()                                  # STOP 접수: 시나리오만 멈추고 조작 흐름 유지
        self.assertFalse(w._cycle_timer.isActive())
        self.assertFalse(w.supplier.active)
        self.assertIn("단계 미확인", w.bolt_caption.text())
        self.assertIsNone(w.bolt_stages._current)
        self.assertTrue(any("1주기 중단 · 상태 STOPPING" in text for text in self.rows()))
        counted = w.tally.counts()
        self.assertTrue(until(lambda: w.sort_demo.snapshot["state"] == "STOPPED"))
        self.assertTrue(w.start_button.isEnabled())
        self.assertFalse(w.sort_cycle_button.isEnabled())
        self.assertEqual(w.tally.counts(), counted)
        # 단절: 마지막 값+미확인, 단절 중 사건은 미반영, 재접속 후 동기화
        w.start_button.click()
        self.assertTrue(until(lambda: w.sort_demo.snapshot["state"] == "RUNNING"))
        w.sort_cycle_button.click()
        w.sort_connection_button.click()
        self.assertFalse(w._cycle_timer.isActive())
        self.assertEqual(w.count_tag.text(), "모의 · 마지막 값 · 미확인")
        self.assertEqual(w.feed_demo_event(w.supplier.make("ARRIVED", "P-0100", 1, spec="NUT", bin="NUT", verdict="OK")).code, "미동기화")
        before = w.tally.counts()
        self.assertFalse(w.sort_cycle_button.isEnabled())
        w.sort_connection_button.click()
        self.assertEqual(w.count_tag.text(), "모의 · 동기화됨")
        self.assertEqual(w.tally.counts()["NUT"], before["NUT"] + 1)   # 공급자 스냅샷에 포함된 단절 중 도착
        self.assertEqual(w.tally.counts(), w.supplier.snapshot(1)["counts"])
        self.assertEqual(w.sort_demo.snapshot["state"], "RECOVERY_REQUIRED")
        self.assertIn("단계 미확인", w.nut_caption.text())
        self.assertFalse(w.sort_cycle_button.isEnabled())
        self.assertFalse(w.start_button.isEnabled())

    def test_demo_new_epoch_and_close_stop_cycle(self):
        w = self.window
        self.run_to_running()
        w.sort_cycle_button.click()
        snapshot = w.sort_demo.snapshot
        self.assertFalse(w.sort_demo.apply_snapshot(dict(snapshot, epoch=2, seq=0)))   # 새 세대 → 동기화 상실
        self.assertFalse(w._cycle_timer.isActive())
        self.assertFalse(w.tally.synchronized)
        self.assertTrue(any("동기화 상실·세대 전환" in text for text in self.rows()))
        self.assertTrue(w.sort_demo.apply_snapshot(dict(snapshot, epoch=2, seq=0), resync=True))
        self.assertTrue(w.tally.synchronized)
        self.assertEqual(w.tally.epoch, 2)
        w.sort_cycle_button.click()
        self.assertTrue(w._cycle_timer.isActive())
        w.close()
        self.assertFalse(w._cycle_timer.isActive())
        self.assertFalse(w.supplier.active)

    def test_direct_resync_to_new_epoch_stops_existing_cycle(self):
        w = self.window
        self.run_to_running()
        w.sort_cycle_button.click()
        snapshot = w.sort_demo.snapshot
        self.assertTrue(w.sort_demo.apply_snapshot(dict(snapshot, epoch=2, seq=0), resync=True))
        self.assertFalse(w._cycle_timer.isActive())
        self.assertFalse(w.supplier.active)
        self.assertEqual(w.tally.epoch, 2)
        self.assertTrue(w.tally.synchronized)

    def assert_operate_cards_unclipped(self, window, size):
        window.resize(*size)
        window._show_page(1, animate=False)
        self.assertTrue(until(lambda: window.size().toTuple() == size))
        self.app.processEvents()
        for card in window._page_cards[1][1:]:
            inner = card.contentsRect()
            for child in card.findChildren(QLabel):
                if not child.isVisible():
                    continue
                need = child.heightForWidth(child.width()) if child.hasHeightForWidth() else child.minimumSizeHint().height()
                self.assertGreaterEqual(child.height(), need, f"{size} {child.text()[:20]!r}")
                self.assertLessEqual(child.geometry().bottom(), inner.bottom(), f"{size} {child.text()[:20]!r}")
                if child.text() == "오분류 / 미판정":
                    self.assertLessEqual(child.fontMetrics().horizontalAdvance(child.text()), child.width())

    def test_operate_cards_fit_minimum_and_default_window_and_real_mode_stays_unknown(self):
        w = self.window
        w.feed_demo_event(w.supplier.make("FAILED", "P-0001-긴-부품-식별자", 1, reason="집기 실패 · 그리퍼 접촉 미확인"))
        w.feed_demo_event(w.supplier.make("FAILED", "P-0002", 1, reason="판정 불가"))
        for size in ((1180, 760), (1440, 960)):
            self.assert_operate_cards_unclipped(w, size)
            self.assert_setup_page_unclipped(w, size)
        link, view = DemoLink(), DemoView()
        window = MainWindow(link, view, demo=False)
        try:
            window.show()
            self.assertIsNone(window.tally)
            self.assertFalse(hasattr(window, "sort_cycle_button"))
            self.assertEqual({value.text() for value in window.count_values.values()}, {"—"})
            self.assertIn("단계 미확인", window.bolt_caption.text())
            self.assertIsNone(window.bolt_stages._current)
            self.assertIsNone(window.feed_demo_event({"kind": "ARRIVED"}))
            for size in ((1180, 760), (1440, 960)):
                self.assert_operate_cards_unclipped(window, size)
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
