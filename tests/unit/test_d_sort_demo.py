"""Internal demo only: no socket or hardware assertions."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication
from eyetoarm.gui.sort_demo import DemoSortController


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


if __name__ == "__main__":
    unittest.main()
