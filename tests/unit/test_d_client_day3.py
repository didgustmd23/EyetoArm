"""Socket-free Qt command and audit contract tests; no robot or server is used."""

import base64
import csv
import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from PySide6.QtWidgets import QApplication  # noqa: E402
from eyetoarm.gui.client import CommandRecord, StatusLink  # noqa: E402
from eyetoarm.gui.records import SessionLog  # noqa: E402
from eyetoarm.tcp import protocol  # noqa: E402


READY = {"version": "1.1", "type": "STATUS", "robot_ready": True, "not_ready_reasons": [],
         "active_command_id": None, "server_time_ms": 1}


def wire(message):
    return protocol.encode(message)


def target():
    return protocol.MoveTarget((100.0, 200.0), 30.0, "manual", 1, 1, "PC_RECEIVE", "cal",
                               created_monotonic_ms=time.monotonic_ns() // 1_000_000)


class LinkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.link = StatusLink(log_path=None, stable_count=3, stable_ms=5000)
        self.link.state = "connected"
        self.link._conn_id = 1
        self.link._send = Mock(return_value=True)
        self.link._handle_line(wire(READY))

    def tearDown(self):
        self.link.stop()
        self.link.deleteLater()
        self.app.processEvents()

    def start_move(self):
        command_id = self.link.execute(target())
        self.assertTrue(command_id)
        self.link._handle_line(wire({**READY, "server_time_ms": 2}))
        self.assertEqual(self.link.current_record.attempts, 1)
        return command_id

    def test_identical_status_emits_once_but_refreshes_age(self):
        seen = []
        self.link.status_received.connect(seen.append)
        self.link._last_status_at = time.monotonic() - 1
        self.link._handle_line(wire({**READY, "server_time_ms": 99}))
        self.assertEqual(seen, [])
        self.assertLess(self.link.status_age_ms(), 500)
        self.assertEqual(self.link.last_status["server_time_ms"], 99)

    def test_reason_order_does_not_trigger_status_change(self):
        seen = []
        self.link.status_received.connect(seen.append)
        busy = {**READY, "robot_ready": False, "not_ready_reasons": ["ROS_DISCONNECTED", "SIM_PAUSED"]}
        self.link._handle_line(wire(busy))
        self.link._handle_line(wire({**busy, "not_ready_reasons": list(reversed(busy["not_ready_reasons"]))}))
        self.assertEqual(len(seen), 1)

    def test_count_threshold_suppresses_duplicate_traffic(self):
        events = []
        self.link.traffic.connect(events.append)
        for n in range(2, 7):
            self.link._handle_line(wire({**READY, "server_time_ms": n}))
        self.assertGreater(self.link.suppressed_count, 0)
        self.assertLess(sum(event.kind == "STATUS" for event in events), 5)

    def test_time_threshold_suppresses_and_change_resumes(self):
        self.link._stable_since = time.monotonic() - 6
        self.link._handle_line(wire({**READY, "server_time_ms": 2}))
        self.assertTrue(self.link._suppressing)
        seen = []
        self.link.status_received.connect(seen.append)
        self.link._handle_line(wire({**READY, "robot_ready": False, "not_ready_reasons": ["SIM_PAUSED"]}))
        self.assertFalse(self.link._suppressing)
        self.assertEqual(len(seen), 1)

    def test_disconnect_invalidates_readiness_and_reconnect_emits_again(self):
        seen = []
        self.link.status_received.connect(seen.append)
        self.link._drop("test")
        self.assertTrue(self.link.is_stale())
        self.assertIsNone(self.link.last_status)
        self.link.state = "connected"
        self.link._handle_line(wire(READY))
        self.assertEqual(len(seen), 1)

    def test_readiness_has_only_one_outstanding_query(self):
        self.link._send_get_status()
        self.link._send_get_status()
        self.assertEqual(self.link._send.call_count, 1)
        self.assertTrue(self.link._ready_pending)

    def test_readiness_timeout_drops_connection_and_invalidates_status(self):
        self.link._send_get_status()
        self.link._ready_timeout()
        self.assertEqual(self.link.state, "disconnected")
        self.assertTrue(self.link.is_stale())
        self.assertFalse(self.link._ready_pending)

    def test_execute_requires_new_readiness_response(self):
        command_id = self.link.execute(target())
        self.assertTrue(command_id)
        self.assertEqual(self.link.current_record.attempts, 0)
        self.assertEqual(json.loads(self.link._send.call_args_list[0].args[0])["type"], "GET_STATUS")
        self.link._handle_line(wire({**READY, "server_time_ms": 2}))
        self.assertEqual(json.loads(self.link._send.call_args_list[-1].args[0])["type"], "MOVE")

    def test_preexisting_query_cannot_authorize_move(self):
        self.link._send_get_status()
        self.assertTrue(self.link.execute(target()))
        self.link._handle_line(wire(READY))
        self.assertEqual(self.link.current_record.attempts, 0)
        self.assertEqual(self.link._send.call_count, 2)
        self.link._handle_line(wire(READY))
        self.assertEqual(self.link.current_record.attempts, 1)

    def test_double_execute_is_blocked(self):
        self.assertTrue(self.link.execute(target()))
        self.assertEqual(self.link.execute(target()), "")

    def test_not_ready_fresh_response_blocks_move(self):
        self.assertTrue(self.link.execute(target()))
        self.link._handle_line(wire({**READY, "robot_ready": False, "not_ready_reasons": ["SIM_PAUSED"]}))
        self.assertEqual(self.link.current_record.attempts, 0)
        self.assertTrue(self.link.current_record.finished)

    def test_two_retries_reuse_exact_id_and_bytes(self):
        command_id = self.start_move()
        self.link._ack_timeout()
        self.link._ack_timeout()
        self.link._ack_timeout()
        moves = [call.args[0] for call in self.link._send.call_args_list if json.loads(call.args[0])["type"] == "MOVE"]
        self.assertEqual(len(moves), 3)
        self.assertTrue(all(raw == moves[0] for raw in moves))
        self.assertEqual(json.loads(moves[0])["command_id"], command_id)
        self.assertEqual(self.link.current_record.acceptance, "unconfirmed")

    def test_invalidation_prevents_retry(self):
        self.start_move()
        self.link.invalidate_target("source changed")
        self.link._ack_timeout()
        moves = [call for call in self.link._send.call_args_list if json.loads(call.args[0])["type"] == "MOVE"]
        self.assertEqual(len(moves), 1)
        self.assertEqual(self.link.current_record.acceptance, "unconfirmed")

    def test_reconnect_only_queries_existing_command(self):
        command_id = self.start_move()
        self.link._drop("test")
        self.link._send.reset_mock()
        self.link._on_connected()
        requests = [json.loads(call.args[0]) for call in self.link._send.call_args_list]
        self.assertTrue(requests)
        self.assertTrue(all(request["type"] == "GET_STATUS" for request in requests))
        self.assertTrue(any(request.get("command_id") == command_id for request in requests))

    def test_rejection_is_not_failed(self):
        command_id = self.start_move()
        rejection = {**READY, "robot_ready": False, "not_ready_reasons": ["SIM_PAUSED"],
                     "command_id": command_id, "code": "NOT_READY"}
        self.link._handle_line(wire(rejection))
        self.assertTrue(self.link.current_record.rejected)
        self.assertIsNone(self.link.current_record.state)

    def test_unknown_command_is_unconfirmed_and_blocks_new_move(self):
        command_id = self.start_move()
        self.link._handle_line(wire({**READY, "command_id": command_id, "code": "UNKNOWN_COMMAND"}))
        self.assertEqual(self.link.current_record.acceptance, "unconfirmed")
        self.assertFalse(self.link.current_record.finished)
        self.assertFalse(self.link.can_execute())

    def test_rank_never_moves_backwards_and_final_is_immutable(self):
        command_id = self.start_move()
        record = self.link.current_record
        self.link._apply_command({"command_id": command_id, "state": "MOVING"}, "command")
        self.link._apply_command({"command_id": command_id, "state": "ACK"}, "command")
        self.assertEqual(record.state, "MOVING")
        self.link._apply_command({"command_id": command_id, "state": "FAILED", "code": "TEST"}, "command")
        self.link._apply_command({"command_id": command_id, "state": "REACHED"}, "command")
        self.assertEqual((record.state, record.code), ("FAILED", "TEST"))

    def test_cancel_is_request_not_terminal_completion(self):
        command_id = self.start_move()
        self.assertTrue(self.link.cancel())
        self.assertEqual(json.loads(self.link._send.call_args.args[0])["command_id"], command_id)
        self.assertFalse(self.link.current_record.finished)
        self.assertEqual(self.link.current_record.cancel, "requested")

    def test_invalid_bytes_are_preserved_in_log(self):
        raw = b"\xff\xfe\r\n"
        self.link._handle_line(raw)
        self.assertEqual(base64.b64decode(self.link.log.rows()[-1]["raw_b64"]), raw)

    def test_stop_cancels_timers_and_queued_send(self):
        self.assertTrue(self.link.execute(target()))
        self.link.stop()
        self.assertFalse(self.link._ready_deadline.isActive())
        self.assertFalse(self.link._ack_deadline.isActive())
        self.assertFalse(self.link._pending_move)


    def test_unconfirmed_can_be_closed_only_by_explicit_user_confirmation(self):
        command_id = self.start_move()
        self.link._handle_line(wire({**READY, "command_id": command_id, "code": "UNKNOWN_COMMAND"}))
        self.assertFalse(self.link.can_execute())
        self.assertTrue(self.link.confirm_new_command())
        self.assertTrue(self.link.current_record.closed_unconfirmed)
        self.assertEqual(self.link.current_record.acceptance, "unconfirmed")
        self.assertIsNone(self.link.current_record.state)
        self.assertTrue(self.link.can_execute())

    def test_retry_reply_planning_is_accepted_after_resend(self):
        command_id = self.start_move()
        self.link._ack_timeout()
        self.link._handle_line(wire({**READY, "command_id": command_id, "state": "PLANNING",
                                    "robot_ready": False, "not_ready_reasons": ["COMMAND_ACTIVE"],
                                    "active_command_id": command_id}))
        self.assertEqual(self.link.current_record.acceptance, "accepted_after_resend")

    def test_stuck_unacknowledged_command_is_queried_without_resend(self):
        command_id = self.start_move()
        self.link.current_record.acceptance = "unconfirmed"
        self.link._last_command_at = 0
        self.link._ready_pending = False
        self.link._send.reset_mock()
        self.link._send_get_status()
        requests = [json.loads(call.args[0]) for call in self.link._send.call_args_list]
        self.assertTrue(any(request.get("command_id") == command_id for request in requests))
        self.assertFalse(any(request["type"] == "MOVE" for request in requests))


class SessionLogTests(unittest.TestCase):
    def test_invalid_config_shapes_and_intervals_fall_back(self):
        from eyetoarm.gui.client import load_tcp_config, DEFAULTS
        with tempfile.TemporaryDirectory(prefix="eyetoarm-config-test-") as folder:
            path = Path(folder) / "config.json"
            for document in ([], {"tcp": []}, {"tcp": {"readiness_poll_ms": -1, "port": 70000}}):
                path.write_text(json.dumps(document), encoding="utf-8")
                self.assertEqual(load_tcp_config(path), DEFAULTS)

    def test_invalid_explicit_port_is_not_silently_replaced(self):
        app = QApplication.instance() or QApplication([])
        with self.assertRaises(ValueError):
            StatusLink(port=0, log_path=None)
        app.processEvents()

    def test_bounded_rows_and_filter(self):
        log = SessionLog(None)
        for index in range(510):
            log.add("TX" if index % 2 else "RX", "test")
        self.assertEqual(len(log.rows()), 500)
        self.assertEqual(len(log.rows("TX")), 250)
        self.assertEqual(log.rows()[0]["seq"], 11)

    def test_csv_bom_crlf_empty_filter_and_formula_prefixes(self):
        log = SessionLog(None)
        dangerous = ("=1+1", "+1", "-1", "@x", "\t=x", "\r=x", "  =1")
        for value in dangerous:
            log.add("RX", "test", note=value)
        data = log.to_csv("RX")
        self.assertTrue(data.startswith("\ufeff"))
        self.assertIn("\r\n", data)
        rows = list(csv.DictReader(io.StringIO(data.lstrip("\ufeff"), newline="")))
        self.assertTrue(all(row["note"].startswith("'") for row in rows))
        self.assertEqual(len(list(csv.DictReader(io.StringIO(log.to_csv("TX").lstrip("\ufeff"))))), 0)

    def test_jsonl_preserves_invalid_bytes_and_write_error_recovers(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "audit.jsonl"
            log = SessionLog(path)
            raw = b"\xff\r\n"
            log.add("RX", "bad", raw=raw)
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(base64.b64decode(stored["raw_b64"]), raw)
            log.path = Path(folder)
            log.add("RX", "failure", raw=raw)
            self.assertTrue(log.write_error)
            self.assertEqual(log.rows()[-1]["kind"], "failure")
            log.path = path
            log.add("RX", "recovery")
            self.assertFalse(log.write_error)
            saved = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["kind"] for row in saved], ["bad", "failure", "recovery"])


if __name__ == "__main__":
    unittest.main()
