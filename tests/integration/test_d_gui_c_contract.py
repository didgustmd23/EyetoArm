"""Loopback contract with unchanged C package and a fake executor, never a robot."""
import asyncio
import base64
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "dashboard-client" / "src"), str(ROOT / "robot_ws/src/robot_server_package")]
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication
from eyetoarm.gui.client import StatusLink
from eyetoarm.tcp import protocol
from eyetoarm.tcp.protocol import MoveTarget
from robot_server_package.tcp_server import TcpServer
from robot_server_package.command_log import CommandLog

FAST = dict(readiness_poll_ms=30, readiness_timeout_ms=200, ack_timeout_ms=100,
            status_watchdog_ms=250, reconnect_ms=40, connect_timeout_ms=200)

def until(predicate, timeout=3):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()

class ActualC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="eyetoarm-day3-c-contract-")
        self.c_path = Path(self.temp.name) / "c-command.jsonl"
        # Logical initialized-empty log fixture; POSIX directory fsync bootstrap is not tested on Windows.
        self.c_path.touch(exist_ok=False)
        self.ready = True
        self.hold = False
        self.hold_s = 2   # fake executor waits this long for CANCEL before giving up
        self.executions = 0
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="day3-owned-C-loop")
        self.thread.start()
        asyncio.run_coroutine_threadsafe(self._start(), self.loop).result(timeout=3)
        self.d_path = Path(self.temp.name) / "d-audit.jsonl"
        self.link = StatusLink("127.0.0.1", self.port, log_path=self.d_path, config=FAST, stable_count=2)
        self.emissions = []
        self.link.status_received.connect(self.emissions.append)
        self.link.start()
        self.assertTrue(until(self.link.can_execute))

    async def _start(self):
        def readiness():
            return {"robot_ready": self.ready, "not_ready_reasons": [] if self.ready else ["SIM_PAUSED"],
                    "active_command_id": None}
        async def execute(request, report, cancel):
            self.executions += 1
            async def dispatch():
                return None
            await report("MOVING", goal_id=str(uuid.uuid4()), dispatch=dispatch)
            if self.hold:
                await asyncio.wait_for(cancel.wait(), self.hold_s)
                await report("FAILED", stage="EXECUTION", code="CANCELLED")
            else:
                await asyncio.sleep(0.03)
                await report("REACHED", reach_error={"position_mm": 0.1, "rotation_deg": 0.2})
        self.server = TcpServer(readiness_provider=readiness, command_log_path=self.c_path,
                                workspace_mm={"x": (-1000, 1000), "y": (-1000, 1000)}, move_executor=execute)
        self.listener = await asyncio.start_server(self.server._handle_client, "127.0.0.1", 0)
        self.port = self.listener.sockets[0].getsockname()[1]

    async def _close(self):
        self.listener.close()
        await self.listener.wait_closed()
        assert not self.listener.is_serving()
        for session in list(self.server._sessions.values()):
            session.writer.close()
        await self.server.close()
        await asyncio.sleep(0)
        tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def tearDown(self):
        self.link.stop()
        self.link.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        try:
            asyncio.run_coroutine_threadsafe(self._close(), self.loop).result(timeout=3)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=3)
            self.assertFalse(self.thread.is_alive())
            self.loop.close()
            self.assertTrue(self.loop.is_closed())
            folder = self.temp.name
            self.temp.cleanup()
            self.assertFalse(Path(folder).exists())

    def target(self):
        now = time.monotonic_ns() // 1_000_000
        return MoveTarget((100, 200), 30, "manual_test", 1, now, "PC_RECEIVE",
                          "manual_test", created_monotonic_ms=now)

    def test_move_updates_c_durable_log_and_d_exact_audit(self):
        command_id = self.link.execute(self.target())
        self.assertTrue(command_id)
        self.assertTrue(until(lambda: self.link.current_record.state == "REACHED"))
        self.assertEqual(self.executions, 1)
        events = CommandLog(self.c_path).load()
        self.assertTrue(any(row["event"] == "REGISTER" and row["command_id"] == command_id for row in events))
        self.assertTrue(any(row.get("status", {}).get("state") == "REACHED" for row in events))
        audit = [json.loads(line) for line in self.d_path.read_text(encoding="utf-8").splitlines()]
        sent = [row for row in audit if row["dir"] == "TX" and row.get("command_id") == command_id]
        self.assertTrue(any(base64.b64decode(row["raw_b64"]) == self.link.current_record.raw for row in sent))
        self.assertTrue(until(self.link.can_execute))
        # Day7: the real C ACK on the original connection yields one local metric row; REACHED does not change it
        latency, reason = self.link.current_record.ack_metric()
        self.assertEqual((self.link.current_record.acceptance, reason), ("first_ack", ""))
        metric = [row for row in audit if row["kind"] == "실행→ACK 지연"]
        self.assertEqual([row["ack_latency_ms"] for row in metric], [latency])
        self.assertTrue(0 <= latency < 3000)

    def test_cancel_updates_c_log_without_fake_stop_success(self):
        self.hold = True
        self.link.execute(self.target())
        self.assertTrue(until(lambda: self.link.current_record.state == "MOVING"))
        self.assertTrue(self.link.cancel())
        self.assertFalse(self.link.current_record.finished)
        self.assertTrue(until(lambda: self.link.current_record.state == "FAILED"))
        self.assertEqual(self.link.current_record.code, "CANCELLED")
        events = CommandLog(self.c_path).load()
        self.assertTrue(any(row.get("status", {}).get("code") == "CANCELLED" for row in events))

    def test_unchanged_status_suppresses_ui_but_keeps_liveness_and_change(self):
        self.assertTrue(until(lambda: self.link.suppressed_count >= 3))
        self.assertEqual(len(self.emissions), 1)
        self.assertFalse(self.link.is_stale())
        self.ready = False
        self.assertTrue(until(lambda: len(self.emissions) == 2))
        self.assertFalse(self.link.can_execute())
        self.assertEqual(self.emissions[-1]["not_ready_reasons"], ["SIM_PAUSED"])
        # Day7 readiness lock: a not-ready C refuses the request locally; nothing reaches C
        self.assertEqual(self.link.execute(self.target()), "")
        self.assertFalse(any(row["dir"] == "TX" and row["kind"] == "MOVE" for row in self.link.log.rows()))
        self.assertEqual(self.executions, 0)

    def test_server_side_disconnect_reconnects_query_only_without_state_regression(self):
        self.hold, self.hold_s = True, 10
        states = []
        self.link.command_changed.connect(lambda record: states.append(record.state))
        command_id = self.link.execute(self.target())
        self.assertTrue(until(lambda: self.link.current_record.state == "MOVING"))
        def replies():   # exact JSONL audit, including repeated statuses hidden from the table
            audit = [json.loads(line) for line in self.d_path.read_text(encoding="utf-8").splitlines()]
            return sum(row["dir"] == "RX" and row.get("command_id") == command_id for row in audit)
        before = replies()

        async def drop_clients():   # C keeps running and keeps the command; only the D connection is cut
            for session in list(self.server._sessions.values()):
                session.writer.close()
        asyncio.run_coroutine_threadsafe(drop_clients(), self.loop).result(timeout=3)
        self.assertTrue(until(lambda: self.link.state != "connected"))
        self.assertTrue(until(lambda: self.link.state == "connected"))
        self.assertTrue(until(lambda: replies() > before))      # the new connection asked C about the same ID
        record = self.link.current_record
        self.assertEqual(record.state, "MOVING")
        self.assertFalse(record.finished)                       # no invented REACHED/FAILED after the gap
        moves = [row for row in self.link.log.rows() if row["dir"] == "TX" and row["kind"] == "MOVE"]
        self.assertEqual(len(moves), 1)                          # reconnect queried, never re-sent MOVE
        self.assertEqual(self.executions, 1)
        self.assertTrue(self.link.cancel())                      # the same C command is still controllable
        self.assertTrue(until(lambda: record.state == "FAILED"))
        ranks = [protocol.STATE_RANK[state] for state in states if state]
        self.assertEqual(ranks, sorted(ranks))                   # duplicate/old statuses never moved it back

if __name__ == "__main__":
    unittest.main()
