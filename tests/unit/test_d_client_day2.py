"""D 2일차 골격의 수신 분해 로직 시험. 소켓·창을 열지 않는다.

  PYTHONPATH=src python -m unittest tests.unit.test_d_client_day2
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eyetoarm.gui.client import load_tcp_config, split_lines  # noqa: E402
from eyetoarm.gui.gz_view_relay import HEADER, MAGIC, MAX_FRAME_BYTES, FrameParser, pack_frame  # noqa: E402


class SplitLinesTest(unittest.TestCase):
    def test_split_and_merged_lines(self) -> None:
        buffer = bytearray(b'{"a":1}\n{"b"')
        self.assertEqual(split_lines(buffer, 4096), [b'{"a":1}'])
        self.assertEqual(buffer, bytearray(b'{"b"'))          # 미완성 꼬리는 남는다
        buffer += b':2}\r\n{"c":3}\n'
        self.assertEqual(split_lines(buffer, 4096), [b'{"b":2}', b'{"c":3}'])   # CR 은 뗀다
        self.assertEqual(buffer, bytearray())

    def test_overflow(self) -> None:
        with self.assertRaises(OverflowError):
            split_lines(bytearray(b"x" * 20), 16)             # LF 없이 한도 초과
        with self.assertRaises(OverflowError):
            split_lines(bytearray(b"x" * 17 + b"\n"), 16)     # 한 줄이 한도 초과


class FrameParserTest(unittest.TestCase):
    def test_roundtrip_across_chunks(self) -> None:
        first, second = b"\xff\xd8first", b"\xff\xd8second-frame"
        stream = pack_frame(first) + pack_frame(second)
        parser = FrameParser()
        frames = []
        for index in range(0, len(stream), 5):                # 5바이트씩 쪼개서 넣는다
            frames += parser.feed(stream[index:index + 5])
        self.assertEqual(frames, [first, second])

    def test_rejects_bad_header(self) -> None:
        with self.assertRaises(ValueError):
            FrameParser().feed(b"NOPE" + b"\x00\x00\x00\x01x")
        with self.assertRaises(ValueError):
            FrameParser().feed(HEADER.pack(MAGIC, MAX_FRAME_BYTES + 1))
        with self.assertRaises(ValueError):
            pack_frame(b"")


class ConfigTest(unittest.TestCase):
    def test_reads_shared_config_and_falls_back(self) -> None:
        config = load_tcp_config()
        self.assertEqual((config["host"], config["port"]), ("127.0.0.1", 47090))
        self.assertEqual(load_tcp_config(ROOT / "없는_파일.json")["port"], 47090)


class ThemeTest(unittest.TestCase):
    def test_palettes_have_same_keys_and_apply_switches(self) -> None:
        from eyetoarm.gui import theme
        self.assertEqual(set(theme.PALETTES["dark"]), set(theme.PALETTES["light"]))   # 한쪽에만 있는 색이 없어야 한다
        try:
            theme.apply("light")
            self.assertEqual((theme.name, theme.BG), ("light", theme.PALETTES["light"]["BG"]))
            self.assertIn(theme.PALETTES["light"]["BG"], theme.QSS)
            self.assertEqual(theme.LEVEL_COLOR["ok"], theme.PALETTES["light"]["OK"])
            theme.apply("없는 테마")                                                    # 모르는 이름은 dark 로
            self.assertEqual(theme.name, "dark")
        finally:
            theme.apply("dark")


if __name__ == "__main__":
    unittest.main()
