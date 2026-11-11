"""D 클라이언트의 신호 작성·수신 검사·펜던트 규칙 시험. 합성 값이며 실제 C 서버 검증이 아니다."""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eyetoarm.pendant.model import CONTRACT, GRIP_MAX_MM, PendantModel  # noqa: E402
from eyetoarm.tcp.event_log import EventLog  # noqa: E402
from eyetoarm.tcp.protocol import (  # noqa: E402
    MoveTarget, ProtocolError, build_cancel, build_get_status, build_move, new_command_id, parse_status, status_kind,
)

try:
    import jsonschema
    C2S = jsonschema.Draft202012Validator(json.loads((ROOT / "docs/protocol/c2s.schema.json").read_text(encoding="utf-8")))
except Exception:  # noqa: BLE001
    C2S = None

READY = {"version": "1.1", "type": "STATUS", "robot_ready": True, "not_ready_reasons": [], "active_command_id": None,
         "server_time_ms": 1}
TARGET = MoveTarget((470.2, 160.4), 35.2, "pendant_manual", 7, 1234, "PC_RECEIVE", "manual_test")


def line(message):
    return json.dumps(message, separators=(",", ":")).encode()


class Build(unittest.TestCase):
    def test_messages_are_one_json_line_with_lf_and_pass_the_repository_schema(self):
        command_id = new_command_id()
        for raw in (build_get_status(), build_get_status(command_id), build_cancel(command_id), build_move(command_id, TARGET)):
            self.assertTrue(raw.endswith(b"\n"))
            self.assertEqual(raw.count(b"\n"), 1)
            self.assertNotIn(b" ", raw.replace(b"pendant_manual", b""))  # 들여쓰기·공백 없음
            if C2S is not None:
                self.assertEqual(list(C2S.iter_errors(json.loads(raw))), [], raw)

    def test_move_fields(self):
        command_id = new_command_id()
        message = json.loads(build_move(command_id, TARGET))
        self.assertEqual(message["position_mm"], [470.2, 160.4])
        self.assertEqual((message["theta_deg"], message["frame"], message["valid"]), (35.2, "work", True))
        self.assertIsInstance(message["image_seq"], int)
        self.assertNotIn("orientation_xyzw", message)
        self.assertRegex(command_id, r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")

    def test_planar_theta_range_is_enforced_before_sending(self):
        for theta in (180.0, -1.0, None):
            with self.assertRaises(ValueError):
                build_move(new_command_id(), MoveTarget((1.0, 2.0), theta, "s", 1, 1, "PC_RECEIVE", "c"))


class Parse(unittest.TestCase):
    def test_three_kinds(self):
        command_id = new_command_id()
        self.assertEqual(status_kind(parse_status(line(READY))), "ready")
        busy = {**READY, "robot_ready": False, "not_ready_reasons": ["COMMAND_ACTIVE"], "active_command_id": command_id}
        self.assertEqual(status_kind(parse_status(line({**busy, "command_id": command_id, "state": "ACK"}))), "command")
        reject = {**READY, "robot_ready": False, "not_ready_reasons": ["RECOVERING"], "command_id": command_id, "code": "NOT_READY"}
        self.assertEqual(status_kind(parse_status(line(reject))), "reject")

    def test_c_test_server_reply_is_accepted(self):
        reply = b'{"version":"1.1","type":"STATUS","robot_ready":false,"not_ready_reasons":["RECOVERING"],"active_command_id":null,"server_time_ms":1790000000000}'
        self.assertFalse(parse_status(reply)["robot_ready"])

    def test_rule_violations_are_rejected_and_never_read_as_ready(self):
        bad = (
            b'{"version":"1.1","type":"STATUS","robot_ready":true,"robot_ready":false,"not_ready_reasons":[],"active_command_id":null,"server_time_ms":1}',
            line({**READY, "extra": 1}),                      # 정의되지 않은 필드
            line(READY).replace(b'"server_time_ms":1', b'"server_time_ms":NaN'),
            line(READY).replace(b'"server_time_ms":1', b'"server_time_ms":1.0'),
            line(READY).replace(b'"server_time_ms":1', b'"server_time_ms":1e400'),
            line({**READY, "not_ready_reasons": ["RECOVERING"]}),  # ready 인데 사유 있음
            b"[1,2]", b"not json", b'{"error":"BAD_FORMAT"}',
            line({**READY, "state": "REACHED", "command_id": new_command_id()}),  # reach_error 없음
        )
        for raw in bad:
            with self.assertRaises(ProtocolError, msg=raw):
                parse_status(raw)

    def test_trailing_cr_is_allowed(self):
        self.assertTrue(parse_status(line(READY) + b"\r")["robot_ready"])


class Log(unittest.TestCase):
    def test_rows_keep_raw_bytes_and_send_state_and_export_matches(self):
        log = EventLog(None)
        seen = []
        log.listeners.append(seen.append)
        raw = build_get_status()
        row = log.add("TX", "GET_STATUS", raw=raw, send_state="unsent", note="연결 없음")
        self.assertEqual(row["raw"], '{"version":"1.1","type":"GET_STATUS"}')
        self.assertEqual((row["bytes"], row["send_state"], seen), (len(raw), "unsent", [row]))
        csv = log.to_csv()
        self.assertIn('"{""version"":""1.1"",""type"":""GET_STATUS""}"', csv)
        self.assertEqual(len(csv.strip().splitlines()), 2)


class Pendant(unittest.TestCase):
    def test_no_jog_signal_without_enable(self):
        pendant = PendantModel()
        self.assertIsNone(pendant.act("jog", axis="X", sign=1))
        self.assertEqual(pendant.cart["X"], 470.0)
        self.assertIn("인에이블 꺼짐", pendant.message)

    def test_jog_composes_draft_signal_and_moves_target(self):
        pendant = PendantModel()
        pendant.act("enable", on=True)
        signal = pendant.act("jog", axis="X", sign=1)
        self.assertEqual((signal["contract"], signal["type"], signal["axis"], signal["delta"], signal["unit"]),
                         (CONTRACT, "JOG", "X", 5.0, "mm"))
        self.assertEqual(signal["target"]["X"], 475.0)
        self.assertEqual(pendant.act("jog", axis="X", sign=-1)["seq"], signal["seq"] + 1)
        self.assertEqual(pendant.cart["X"], 470.0)

    def test_theta_wraps_and_z_floor(self):
        pendant = PendantModel()
        pendant.act("enable", on=True)
        pendant.act("jog", axis="RZ", sign=-1)
        self.assertEqual(pendant.cart["RZ"], 175.0)
        pendant.cart["Z"] = 2.0
        self.assertIsNone(pendant.act("jog", axis="Z", sign=-1))
        self.assertEqual(pendant.cart["Z"], 2.0)

    def test_frame_step_speed_and_gripper(self):
        pendant = PendantModel()
        pendant.act("enable", on=True)
        pendant.act("frame")
        self.assertIsNone(pendant.act("jog", axis="X", sign=1))  # 관절 좌표계에서 X 는 축이 아님
        self.assertEqual(pendant.act("jog", axis="J1", sign=1)["unit"], "deg")
        self.assertEqual(pendant.act("speed", sign=1)["speed_ratio"], 0.5)
        for _ in range(40):
            last = pendant.act("grip", open=False)
        self.assertEqual((last["opening_mm"], pendant.grip_mm), (0.0, 0.0))
        self.assertEqual(pendant.act("grip", open=True)["opening_mm"], 5.0)
        self.assertLessEqual(pendant.grip_mm, GRIP_MAX_MM)

    def test_estop_locks_until_reset(self):
        pendant = PendantModel()
        pendant.act("enable", on=True)
        self.assertEqual(pendant.act("estop")["type"], "STOP")
        self.assertFalse(pendant.enabled)
        pendant.act("enable", on=True)
        self.assertFalse(pendant.enabled)  # 잠금 중에는 켜지지 않는다
        self.assertIsNone(pendant.act("jog", axis="X", sign=1))
        pendant.act("reset")
        pendant.act("enable", on=True)
        self.assertIsNotNone(pendant.act("jog", axis="X", sign=1))


if __name__ == "__main__":
    unittest.main()
