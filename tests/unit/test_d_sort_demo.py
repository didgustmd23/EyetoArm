"""Internal demo only: no socket or hardware assertions."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication
from eyetoarm.gui.sort_demo import DemoProcessSupplier, DemoSortController, SortTally


def snapshot(**changes):
    value = dict(epoch=1, seq=1, state="IDLE", holding="DETACHED", job_id=None, detail="")
    value.update(changes)
    return value


class SortDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.controller = DemoSortController()
        self.addCleanup(self.controller.close)
        self.controller.set_connected(True)
        self.assertTrue(self.controller.apply_snapshot(snapshot(), resync=True))

    def finish(self):
        self.controller._timer.stop()
        self.controller._finish()

    def test_start_stop_and_duplicate(self):
        c = self.controller
        self.assertTrue(c.request("START"))
        self.assertEqual(c.snapshot["state"], "STARTING")
        self.assertTrue(c.snapshot["job_id"])
        self.assertFalse(c.request("START"))
        self.finish()
        self.assertEqual(c.snapshot["state"], "RUNNING")
        self.assertTrue(c.request("STOP"))
        self.assertEqual(c.snapshot["state"], "STOPPING")
        self.assertFalse(c.request("STOP"))
        self.finish()
        self.assertEqual(c.snapshot["state"], "STOPPED")

    def test_unknown_holding_requires_confirmed_recovery(self):
        c = self.controller
        self.assertTrue(c.apply_snapshot(snapshot(seq=2, state="RUNNING", holding="HOLD_UNKNOWN")))
        self.assertTrue(c.request("STOP"))
        self.finish()
        self.assertEqual(c.snapshot["state"], "RECOVERY_REQUIRED")
        self.assertFalse(c.request("START"))
        self.assertFalse(c.request("RECOVER"))
        self.assertTrue(c.request("RECOVER", confirmed=True))
        self.assertFalse(c.request("START"))
        self.finish()
        self.assertEqual((c.snapshot["state"], c.snapshot["holding"]), ("STOPPED", "DETACHED"))

    def test_snapshot_order_validation_and_copy(self):
        c = self.controller
        self.assertFalse(c.apply_snapshot(snapshot(seq=-1)))
        self.assertFalse(c.apply_snapshot(snapshot(state="RUNNING"), resync=True))
        self.assertFalse(c.apply_snapshot(snapshot(epoch=2)))
        self.assertFalse(c.synchronized)
        self.assertFalse(c.allowed("START")[0])
        self.assertFalse(c.apply_snapshot(snapshot(seq=20), resync=True))
        self.assertTrue(c.apply_snapshot(snapshot(epoch=2), resync=True))
        self.assertFalse(c.apply_snapshot(snapshot(epoch=1, seq=999), resync=True))
        self.assertFalse(c.apply_snapshot(snapshot(epoch=2, seq=0), resync=True))
        self.assertFalse(c.apply_snapshot(snapshot(epoch=2, seq=2, state=[])))
        external = snapshot(epoch=2, seq=2)
        self.assertTrue(c.apply_snapshot(external))
        external["state"] = "RUNNING"
        c.snapshot["state"] = "RUNNING"
        self.assertEqual(c.snapshot["state"], "IDLE")

    def test_disconnect_cancels_and_requires_resync(self):
        c = self.controller
        self.assertTrue(c.request("START"))
        c.set_connected(False)
        self.assertFalse(c._timer.isActive())
        c.set_connected(True)
        self.assertFalse(c.request("STOP"))
        self.assertFalse(c.apply_snapshot(snapshot(seq=3)))
        self.assertTrue(c.apply_snapshot(snapshot(seq=3, state="STOPPED"), resync=True))
        self.assertEqual(c.snapshot["state"], "STOPPED")

    def test_external_snapshot_cancels_pending_transition(self):
        c = self.controller
        self.assertTrue(c.request("START"))
        self.assertTrue(c.apply_snapshot(snapshot(seq=3, state="RECOVERY_REQUIRED")))
        self.assertFalse(c._timer.isActive())
        self.assertEqual(c.snapshot["state"], "RECOVERY_REQUIRED")

    def test_new_generation_locks_pending_action_until_snapshot(self):
        c = self.controller
        self.assertTrue(c.request("START"))
        self.assertFalse(c.apply_snapshot(snapshot(epoch=2)))
        self.assertFalse(c._timer.isActive())
        self.assertFalse(c.allowed("STOP")[0])
        c._finish()
        self.assertEqual(c.snapshot["state"], "STARTING")
        self.assertTrue(c.apply_snapshot(snapshot(epoch=2), resync=True))
        self.assertTrue(c.allowed("START")[0])

    def test_identical_resync_does_not_cancel_pending_transition(self):
        c = self.controller
        self.assertTrue(c.request("START"))
        self.assertTrue(c.apply_snapshot(c.snapshot, resync=True))
        self.assertTrue(c._timer.isActive())
        self.finish()
        self.assertEqual(c.snapshot["state"], "RUNNING")

    def test_injected_pending_snapshot_waits_for_completion_snapshot(self):
        c = self.controller
        for state in ("STOPPING", "RECOVERING"):
            seq = c.snapshot["seq"] + 1
            self.assertTrue(c.apply_snapshot(snapshot(seq=seq, state=state)))
            self.assertFalse(c._timer.isActive())
            self.assertFalse(c.allowed("START")[0])
            self.assertTrue(c.apply_snapshot(snapshot(seq=seq + 1, state="STOPPED")))
            self.assertTrue(c.allowed("START")[0])

    def test_same_snapshot_after_disconnect_needs_later_snapshot(self):
        c = self.controller
        self.assertTrue(c.request("START"))
        pending = c.snapshot
        c.set_connected(False)
        c.set_connected(True)
        self.assertTrue(c.apply_snapshot(pending, resync=True))
        self.assertFalse(c._timer.isActive())
        self.assertEqual(c.snapshot["state"], "STARTING")
        pending.update(seq=pending["seq"] + 1, state="RUNNING")
        self.assertTrue(c.apply_snapshot(pending))
        self.assertTrue(c.allowed("STOP")[0])

    def test_held_part_blocks_new_start_and_close_revokes_actions(self):
        c = self.controller
        self.assertTrue(c.apply_snapshot(snapshot(seq=2, state="RUNNING", holding="HELD")))
        self.assertTrue(c.request("STOP"))
        self.finish()
        self.assertEqual(c.snapshot["holding"], "HELD")
        self.assertFalse(c.allowed("START")[0])
        self.assertTrue(c.apply_snapshot(snapshot(seq=c.snapshot["seq"] + 1, state="STOPPED")))
        self.assertTrue(c.allowed("START")[0])
        c.close()
        self.assertFalse(c.request("START"))
        self.assertFalse(c._timer.isActive())

    def test_timer_completes_in_event_loop(self):
        c = self.controller
        self.assertTrue(c.request("START"))
        loop = QEventLoop()
        QTimer.singleShot(650, loop.quit)
        loop.exec()
        self.assertEqual(c.snapshot["state"], "RUNNING")


def arrived(supplier, part, spec, epoch=1, bin_=None, verdict="OK"):
    return supplier.make("ARRIVED", part, epoch, spec=spec, bin=bin_ or spec, verdict=verdict)


class SortTallyTests(unittest.TestCase):
    """5일차 모의 집계. 공급자는 소비자 집합을 읽지 않는 독립 객체다."""

    def setUp(self):
        self.supplier = DemoProcessSupplier("S1")
        self.tally = SortTally("S1", epoch=1)
        self.assertEqual(self.tally.apply_snapshot(self.supplier.snapshot(1)).code, "반영")

    def feed(self, event):
        return self.tally.process(event)

    def counts(self):
        return self.tally.counts()

    def test_normal_cycle_counts_each_bin_once_and_shows_stage(self):
        s = self.supplier
        self.assertTrue(s.start_cycle())
        self.assertFalse(s.start_cycle())
        outcomes = []
        while (event := s.next_event(1)) is not None:
            outcomes.append(self.feed(event))
        self.assertFalse(s.active)
        counts = self.counts()
        self.assertEqual((counts["BOLT_A"], counts["BOLT_B"], counts["NUT"], counts["QUARANTINE"]), (1, 1, 1, 0))
        self.assertEqual(sum(o.counted for o in outcomes), 3)
        self.assertTrue(all(o.code == "반영" for o in outcomes))
        self.assertTrue(any("해제 요청" in o.note and not o.counted for o in outcomes))
        self.assertEqual(self.tally.display["bolt"], {"part": "P-0002", "spec": "BOLT_B", "stage": 3})
        self.assertEqual(self.tally.display["nut"], {"part": "P-0003", "spec": "NUT", "stage": 1})
        self.assertEqual(self.tally.fields["recognition"], "NUT 인식")
        self.assertEqual(self.supplier.snapshot(1)["counts"], counts)

    def test_redelivery_release_and_unconfirmed_arrival_add_nothing(self):
        s = self.supplier
        first = arrived(s, "P1", "BOLT_A")
        self.assertEqual(self.feed(first).code, "반영")
        self.assertEqual(self.feed(first).code, "중복")                      # 같은 순번 재전달
        self.assertEqual(self.feed(arrived(s, "P1", "BOLT_A")).code, "중복")  # 다른 알림으로 재전달
        for note in ("해제 요청", "운반 완료", "MOVE REACHED", "도착 미확인"):
            outcome = self.feed(s.make("PROGRESS", "P2", 1, spec="BOLT_B", stage=1, note=note))
            self.assertFalse(outcome.counted)
        self.assertEqual(self.counts()["BOLT_A"], 1)
        self.assertEqual(self.counts()["BOLT_B"], 0)

    def test_wrong_bin_unresolved_quarantine_and_failure_keys(self):
        s = self.supplier
        self.assertEqual(self.feed(arrived(s, "W1", "BOLT_A", bin_="NUT", verdict="WRONG_BIN")).code, "반영")
        self.assertEqual(self.feed(arrived(s, "U1", "UNRESOLVED", bin_="BOLT_B", verdict="UNRESOLVED")).code, "반영")
        self.assertEqual(self.feed(s.make("FAILED", "Q1", 1, reason="집기 실패")).code, "반영")
        self.assertEqual(self.feed(s.make("FAILED", "Q1", 1, reason="집기 실패")).code, "중복")
        self.assertEqual(self.feed(s.make("FAILED", "Q1", 1, reason="판정 불가")).code, "반영")
        self.assertEqual(self.counts()["QUARANTINE"], 0)                    # 실패만으로 격리 수 증가 없음
        self.assertEqual(self.feed(s.make("QUARANTINED", "Q1", 1, reason="집기 실패")).code, "반영")
        counts = self.counts()
        self.assertEqual((counts["BOLT_A"], counts["BOLT_B"], counts["NUT"]), (0, 0, 0))
        self.assertEqual((counts["QUARANTINE"], counts["WRONG_BIN"], counts["UNRESOLVED"]), (1, 1, 1))
        self.assertEqual(self.tally.failure_counts(), {"집기 실패": 1, "판정 불가": 1})
        self.assertEqual(self.tally.last_failure, "Q1 · 판정 불가")

    def test_conflicting_final_results_are_not_summed(self):
        s = self.supplier
        self.feed(arrived(s, "P1", "BOLT_A"))
        self.feed(arrived(s, "W1", "NUT", bin_="BOLT_A", verdict="WRONG_BIN"))
        before = self.counts()
        self.assertEqual(self.feed(s.make("QUARANTINED", "P1", 1)).code, "불일치")
        self.assertEqual(self.feed(arrived(s, "W1", "NUT")).code, "불일치")
        self.assertEqual(self.feed(s.make("QUARANTINED", "W1", 1)).code, "불일치")
        self.assertEqual(self.counts(), before)
        self.assertTrue(self.tally.conflict)
        normal = sum(self.counts()[key] for key in ("BOLT_A", "BOLT_B", "NUT", "QUARANTINE"))
        self.assertEqual(normal, sum(1 for c in self.tally.finals.values() if c not in ("WRONG_BIN", "UNRESOLVED")))
        self.assertEqual(self.tally.apply_snapshot(s.snapshot(1)).code, "반영")   # 더 최신 권위 스냅샷만 신뢰 재설정
        self.assertFalse(self.tally.conflict)

    def test_late_unique_arrival_counts_but_keeps_latest_stage(self):
        s = self.supplier
        late_arrival = arrived(s, "P1", "BOLT_A")
        late_progress = s.make("PROGRESS", "P1", 1, spec="BOLT_A", stage=1)
        newer = s.make("PROGRESS", "P2", 1, spec="BOLT_B", stage=0, robot="이동 중 집기")
        self.assertEqual(self.feed(newer).code, "반영")
        self.assertEqual(self.feed(late_progress).code, "지연")
        outcome = self.feed(late_arrival)
        self.assertEqual((outcome.code, outcome.counted), ("지연 반영", True))
        self.assertEqual(self.counts()["BOLT_A"], 1)
        self.assertEqual(self.tally.display["bolt"], {"part": "P2", "spec": "BOLT_B", "stage": 0})
        self.assertEqual(self.tally.fields["robot"], "이동 중 집기")

    def test_old_epoch_other_scope_unsynchronized_and_malformed_are_record_only(self):
        s = self.supplier
        good = arrived(s, "P1", "BOLT_A")
        self.assertEqual(self.feed(dict(good, epoch=0)).code, "옛 세대")
        self.assertEqual(self.feed(dict(good, scope="S0")).code, "다른 범위")
        self.assertEqual(self.feed(dict(good, epoch=2)).code, "미동기화")
        for bad in (None, [], dict(good, seq=0), dict(good, seq=True), dict(good, part=" "), dict(good, spec="BOLT_C"),
                    dict(good, verdict="OK", bin="NUT"), dict(good, extra=1), dict(good, kind="MOVE"),
                    {k: v for k, v in good.items() if k != "bin"}, dict(s.make("PROGRESS", "P1", 1, spec="NUT", stage=0), stage=2),
                    dict(kind="FAILED", scope="S1", epoch=1, seq=99, part="P9", reason=""),
                    dict(kind="FAILED", scope="S1", epoch=1, seq=99, part="P9", reason=3)):
            self.assertEqual(self.feed(bad).code, "형식오류", bad)
        self.tally.desync()
        self.assertEqual(self.feed(good).code, "미동기화")
        self.assertEqual(self.counts(), SortTally("S1").counts())

    def test_snapshot_barrier_covers_disconnect_arrivals_and_rejects_redelivery(self):
        s, t = self.supplier, self.tally
        delivered = [arrived(s, "P1", "BOLT_A"), s.make("FAILED", "P2", 1, reason="집기 실패")]
        for event in delivered:
            self.assertEqual(self.feed(event).code, "반영")
        t.desync()
        lost = arrived(s, "P3", "NUT")                       # 단절 중 도착: 소비자에게 전달되지 않음
        unsynced = s.make("QUARANTINED", "P2", 1)
        self.assertEqual(self.feed(unsynced).code, "미동기화")
        self.assertEqual(self.counts()["NUT"], 0)
        barrier = s.snapshot(1)                              # 장벽: 이전 사건 전부 포함, 이후 사건은 채택 뒤 전달
        self.assertEqual(t.apply_snapshot(barrier).code, "반영")
        barrier["finals"]["X"] = "NUT"                       # 공급자 원본과 소비자 집합은 공유하지 않는다
        self.assertNotIn("X", t.finals)
        self.assertEqual(t.counts(), s.snapshot(1)["counts"])
        for event in (*delivered, lost, unsynced):
            self.assertEqual(self.feed(event).code, "중복")
        self.assertEqual(self.feed(arrived(s, "P4", "BOLT_B")).code, "반영")
        self.assertEqual(t.counts(), s.snapshot(1)["counts"])
        ghost = dict(kind="ARRIVED", scope="S1", epoch=1, seq=2, part="GHOST", spec="NUT", bin="NUT", verdict="OK")
        self.assertEqual(self.feed(ghost).code, "불일치")    # 경계 이하인데 집합에 없음
        self.assertTrue(t.conflict)
        self.assertEqual(t.counts(), s.snapshot(1)["counts"])

    def test_snapshot_order_scope_and_explicit_new_session(self):
        s, t = self.supplier, self.tally
        stale = s.snapshot(1)
        self.feed(arrived(s, "P1", "BOLT_A"))
        self.feed(arrived(s, "P2", "BOLT_A"))
        self.assertEqual(t.apply_snapshot(stale).code, "낮은 경계")
        self.assertEqual(t.apply_snapshot(dict(s.snapshot(1), epoch=0)).code, "옛 세대")
        bad = s.snapshot(1)
        bad["counts"] = dict(bad["counts"], BOLT_A=5)
        self.assertEqual(t.apply_snapshot(bad).code, "형식오류")
        self.assertEqual(t.apply_snapshot(dict(s.snapshot(1), finals={"P1": "BOLT_C"})).code, "형식오류")
        self.assertEqual(t.counts()["BOLT_A"], 2)
        # 더 최신 권위 스냅샷의 수량이 더 작으면 대체하고 감소를 기록한다(임의 합산 없음)
        smaller = dict(s.snapshot(2), boundary=s.seq + 1, finals={"P1": "BOLT_A"})
        smaller["counts"] = dict(smaller["counts"], BOLT_A=1)
        outcome = t.apply_snapshot(smaller)
        self.assertEqual((outcome.code, t.counts()["BOLT_A"], t.epoch), ("반영", 1, 2))
        self.assertIn("BOLT_A 2→1", outcome.note)
        other = DemoProcessSupplier("S2")
        self.assertEqual(t.apply_snapshot(other.snapshot(2)).code, "다른 범위")
        reused = arrived(other, "P1", "BOLT_A", epoch=2)      # 새 범위의 순번 재시작·같은 부품 ID 재사용
        self.assertEqual(self.feed(reused).code, "다른 범위")
        t.reset_session("S2", 2)
        self.assertEqual(t.counts()["BOLT_A"], 0)
        self.assertEqual(self.feed(reused).code, "반영")
        self.assertEqual(self.feed(arrived(s, "P9", "NUT", epoch=2)).code, "다른 범위")

    def test_bad_kind_and_boolean_snapshot_counts_are_rejected_without_exception(self):
        good = arrived(self.supplier, "P1", "BOLT_A")
        for kind in ([], {}, None):
            self.assertEqual(self.feed(dict(good, kind=kind)).code, "형식오류")
        bad = self.supplier.snapshot(1)  # 실제 수량 1과 True의 동등 비교로 통과하면 안 된다
        bad["counts"]["BOLT_A"] = True
        self.assertEqual(self.tally.apply_snapshot(bad).code, "형식오류")
        self.assertEqual(self.counts()["BOLT_A"], 0)

    def test_same_boundary_cannot_change_authority_or_clear_conflict(self):
        s, t = self.supplier, self.tally
        event = arrived(s, "P1", "BOLT_A")
        self.feed(event)
        authority = s.snapshot(1)
        self.assertEqual(t.apply_snapshot(authority).code, "반영")
        changed = dict(authority, finals={"P1": "NUT"})
        changed["counts"] = dict(authority["counts"], BOLT_A=0, NUT=1)
        self.assertEqual(t.apply_snapshot(changed).code, "불일치")
        self.assertEqual(t.counts()["BOLT_A"], 1)
        ghost = dict(event, part="GHOST")
        self.assertEqual(self.feed(ghost).code, "불일치")
        self.assertTrue(t.conflict)
        self.assertEqual(t.apply_snapshot(authority).code, "불일치")
        self.assertTrue(t.conflict)

    def test_reused_event_sequence_with_different_content_is_not_a_duplicate(self):
        event = arrived(self.supplier, "P1", "BOLT_A")
        self.feed(event)
        self.assertEqual(self.feed(dict(event, part="P2")).code, "불일치")
        self.assertEqual(self.counts()["BOLT_A"], 1)
        self.assertTrue(self.tally.conflict)

    def test_same_authority_reconnect_keeps_conflict_without_retry_loop(self):
        event = arrived(self.supplier, "P1", "BOLT_A")
        self.feed(event)
        authority = self.supplier.snapshot(1)
        self.tally.apply_snapshot(authority)
        self.feed(dict(event, part="GHOST"))
        self.tally.desync()
        self.assertEqual(self.tally.apply_snapshot(authority).code, "불일치")
        self.assertTrue(self.tally.synchronized)
        self.assertTrue(self.tally.conflict)

    def test_delayed_failure_does_not_replace_recent_failure(self):
        older = self.supplier.make("FAILED", "OLD", 1, reason="옛 실패")
        newer = self.supplier.make("FAILED", "NEW", 1, reason="새 실패")
        self.feed(newer)
        self.feed(older)
        self.assertEqual(self.tally.last_failure, "NEW · 새 실패")
        self.assertEqual(sum(self.tally.failure_counts().values()), 2)

    def test_supplier_does_not_restore_old_epoch_arrival_through_snapshot(self):
        stale = arrived(self.supplier, "OLD", "BOLT_A", epoch=0)
        self.assertEqual(self.feed(stale).code, "옛 세대")
        self.tally.desync()
        self.tally.apply_snapshot(self.supplier.snapshot(1))
        self.assertEqual(self.counts()["BOLT_A"], 0)

    def test_authority_replacement_does_not_leave_an_obsolete_recent_failure(self):
        self.feed(self.supplier.make("FAILED", "P1", 1, reason="과거 실패"))
        replacement = dict(self.supplier.snapshot(1), boundary=2, failures=[])
        self.assertEqual(self.tally.apply_snapshot(replacement).code, "반영")
        self.assertEqual(self.tally.failure_counts(), {})
        self.assertIsNone(self.tally.last_failure)


if __name__ == "__main__":
    unittest.main()
