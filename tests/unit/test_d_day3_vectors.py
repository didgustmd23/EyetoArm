"""Repository vectors against the immutable C parser and D STATUS parser."""
import json
import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "robot_ws/src/robot_server_package")]
from eyetoarm.tcp import protocol as d
from robot_server_package import protocol as c
VECTORS = json.loads((ROOT / "docs/protocol/test_vectors.json").read_text(encoding="utf-8"))

class Vectors(unittest.TestCase):
    def test_97_wire_vectors(self):
        self.assertEqual(len(VECTORS["vectors"]), 97)
        for row in VECTORS["vectors"]:
            with self.subTest(id=row["id"], layer=row["layer"]):
                def check():
                    if row["direction"] == "s2c":
                        return d.parse_status(row["raw"].encode("utf-8"))
                    request = c.parse_request(row["raw"].encode("utf-8"))
                    return c.validate_request(request)
                if row["expect"] == "accept":
                    check()
                else:
                    with self.assertRaises((c.ProtocolError, d.ProtocolError, ValueError)):
                        check()

    def test_14_c_dedupe_vectors(self):
        self.assertEqual(len(VECTORS["dedupe"]), 14)
        for row in VECTORS["dedupe"]:
            with self.subTest(id=row["id"]):
                a, b = c.parse_request(row["a"].encode()), c.parse_request(row["b"].encode())
                self.assertEqual(c.same_move_content(a, b), row["expect"] == "same")

if __name__ == "__main__":
    unittest.main()
