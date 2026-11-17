"""ETV1 parser/metadata/decoder for the C display_relay prototype; packets made by the UNCHANGED C pack_packet."""
import json
import os
import struct
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dashboard-client" / "src"))   # canonical D GUI
sys.path.insert(0, str(ROOT / "robot_ws" / "src" / "robot_server_package"))
from PySide6.QtCore import QBuffer, QIODevice
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication
from robot_server_package.display_relay import MAX_PACKET as C_MAX_PACKET, pack_packet   # stdlib-only at import
from eyetoarm.gui import relay_view
from eyetoarm.gui.relay_view import EtvParser, decode_jpeg, parse_metadata


def jpeg(width=64, height=36, color="#3080c0"):
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor(color))
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "JPG")
    return bytes(buffer.data())


def meta(channel="link", seq=1, run="run-A", variant="raw", **override):
    """Exactly the C display_relay metadata shape (including the stamp keys D must not forward)."""
    value = {"channel": channel, "variant": variant, "source_id": f"gz_{channel}", "image_seq": seq,
             "generation": 0, "calibration_id": None, "run_id": run, "capture_generation": 0,
             "detector_config_id": "raw-camera-v1", "job_seq": seq, "stamp_sec": 12, "stamp_nanosec": 345}
    value.update(override)
    return value


class ParserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.jpeg = jpeg()

    def test_c_constants_match(self):
        self.assertEqual(relay_view.MAX_PACKET, C_MAX_PACKET)
        self.assertEqual(relay_view.DEFAULT_PORT, 47092)

    def test_split_and_coalesced_packets_keep_only_ingest_keys(self):
        stream = b"".join(pack_packet(meta(ch, 1), self.jpeg) for ch in ("fixed", "gripper", "link"))
        parser = EtvParser()
        got = []
        for index in range(len(stream)):   # one byte at a time
            got += parser.feed(stream[index:index + 1])
        self.assertEqual([channel for channel, _e, _j in got], ["fixed", "gripper", "link"])
        channel, envelope, data = got[0]
        self.assertEqual(data, self.jpeg)
        self.assertEqual(set(envelope), {"source_id", "image_seq", "generation", "capture_generation", "job_seq",
                                         "run_id", "detector_config_id", "calibration_id", "variant"})
        self.assertNotIn("stamp_sec", envelope)
        coalesced = EtvParser().feed(stream + stream)
        self.assertEqual(len(coalesced), 6)

    def test_framing_errors_raise_before_buffering_a_body(self):
        with self.assertRaises(ValueError):
            EtvParser().feed(b"EAV1" + struct.pack(">I", 10))
        with self.assertRaises(ValueError):   # oversize declared length is refused from the 8-byte header alone
            EtvParser().feed(b"ETV1" + struct.pack(">I", C_MAX_PACKET + 1))
        with self.assertRaises(ValueError):
            EtvParser().feed(b"ETV1" + struct.pack(">I", 4))

    def test_bad_packets_are_skipped_and_the_stream_continues(self):
        good = pack_packet(meta("fixed", 2), self.jpeg)

        def raw_packet(meta_bytes, payload=b"\xff\xd8x"):
            body = struct.pack(">I", len(meta_bytes)) + meta_bytes + payload
            return b"ETV1" + struct.pack(">I", len(body)) + body
        bad = [
            raw_packet(b"{not json"),
            raw_packet(json.dumps(meta(image_seq=True)).encode()),                 # bool is not an int
            raw_packet(json.dumps(meta(image_seq=-1)).encode()),
            raw_packet(json.dumps(meta(source_id="gz_fixed")).encode()),            # wrong source for channel
            raw_packet(json.dumps(meta("fixed", variant="detect")).encode()),       # only link has variants
            raw_packet(json.dumps(meta(channel="top")).encode()),
            raw_packet(b'{"channel":"link","image_seq":NaN}'),
            raw_packet(json.dumps(meta(run="")).encode()),
            raw_packet(b"[]"),
            raw_packet(json.dumps(meta()).encode(), payload=b""),                    # no JPEG
            raw_packet(b"x" * (relay_view.MAX_META + 1)),
            raw_packet(json.dumps(meta(channel=[])).encode()),                      # unhashable channel
            raw_packet(json.dumps(meta(channel={})).encode()),
            raw_packet(json.dumps(meta(variant=["raw"])).encode()),
            raw_packet(b"[" * 8000 + b"]" * 8000),                                  # deep nesting under 16 KiB
            raw_packet(b'{"a":' * 3000 + b"1" + b"}" * 3000),
        ]
        parser = EtvParser()
        got = parser.feed(b"".join(bad) + good)
        self.assertEqual([(channel, envelope["image_seq"]) for channel, envelope, _j in got], [("fixed", 2)])
        self.assertEqual(parser.packet_errors, len(bad))
        missing = dict(meta())
        del missing["calibration_id"]
        with self.assertRaises(relay_view.PacketError):
            parse_metadata(json.dumps(missing).encode())

    def test_link_variants_are_accepted_as_metadata(self):
        for variant in ("raw", "detect", "center"):
            channel, envelope = parse_metadata(json.dumps(meta(variant=variant)).encode())
            self.assertEqual((channel, envelope["variant"]), ("link", variant))

    def test_decode_checks_size_before_allocation(self):
        self.assertIsNotNone(decode_jpeg(self.jpeg))
        self.assertIsNone(decode_jpeg(b"\xff\xd8 not a jpeg \xff\xd9"))
        original = relay_view.MAX_PIXELS
        relay_view.MAX_PIXELS = 64 * 36 - 1
        try:
            self.assertIsNone(decode_jpeg(self.jpeg))   # declared size over the cap: not decoded
        finally:
            relay_view.MAX_PIXELS = original


if __name__ == "__main__":
    unittest.main()
