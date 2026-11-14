"""C의 실제 파서/직렬화와 D 호환성. ROS·실서버·로봇 실행 시험은 아니다."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "robot_ws/src/robot_server_package"))

from eyetoarm.tcp import protocol as d_protocol
from robot_server_package import protocol as c_protocol
from robot_server_package.readiness import get_readiness_fields


class Day1ContractTest(unittest.TestCase):
    def test_d_query_is_accepted_by_c(self):
        wire = d_protocol.build_get_status()
        self.assertTrue(wire.endswith(b"\n"))
        request = c_protocol.parse_request(wire[:-1])
        self.assertIsNone(c_protocol.validate_get_status_request(request))
        self.assertEqual(request, {"version": "1.1", "type": "GET_STATUS"})

    def test_c_readiness_response_is_accepted_by_d(self):
        request = c_protocol.parse_request(d_protocol.build_get_status()[:-1])
        status = c_protocol.build_get_status_response(request, get_readiness_fields())
        received = d_protocol.parse_status(c_protocol.encode_response(status)[:-1])
        self.assertEqual(d_protocol.status_kind(received), "ready")
        self.assertFalse(received["robot_ready"])
        self.assertEqual(received["not_ready_reasons"], ["ROS_DISCONNECTED"])
        self.assertIsNone(received["active_command_id"])


if __name__ == "__main__":
    unittest.main()
