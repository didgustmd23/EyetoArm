"""inputs/stored_images.py 저장 이미지·명시 순서 목록 입력 단위시험.

근거: A 상세설계 §5.1(image_seq 수명, 시각 관계), §5.4(입력 상태, 재처리),
공통 자료형 관리 초안 §3.2·§4 불변조건 8, A 상세설계 §6(저장 이미지 행).
합성 PNG·깨진 파일은 임시 폴더에 만들고, 시계는 가짜 단조 시계를 주입한다.
"""
import os
import tempfile
import types
import unittest
from unittest import mock

import cv2
import numpy as np

from eyetoarm.inputs import stored_images
from eyetoarm.inputs.base import (
    FrameIdIssuer,
    FramePacket,
    InputStatus,
    InputStatusEvent,
    TimestampBase,
)
from eyetoarm.inputs.stored_images import StoredImageSource

RUN_ID = "run-test"

# 공개 데이터(data/raw) train 표본. 디코딩되는 것을 확인한 파일이다
PUBLIC_SAMPLES = (
    "data/raw/train/z4773355985498_0b2ac5577b3fbe55daacf39124e8c881_jpg.rf.000fb5ac0b21b6bb802ec8b64ef699e9.jpg",
    "data/raw/train/M6_frame_50_jpg.rf.0ea94e92b4974ee895686f3b383ecfab.jpg",
    "data/raw/train/062_png_jpg.rf.005e97902406f75fcd43317fbe7f4b51.jpg",
)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeClock:
    """호출마다 1_000_000_123 ns씩 증가하는 가짜 단조 시계. 호출 수를 센다."""

    def __init__(self, start_ns=5_000_000_789, step_ns=1_000_000_123):
        self.calls = 0
        self._now = start_ns
        self._step = step_ns

    def __call__(self):
        self.calls += 1
        value = self._now
        self._now += self._step
        return value


class StoredInputTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        # 크기와 BGR 값이 서로 다른 합성 PNG 3장
        self.img_a = self._write_png("a.png", (4, 6), (10, 20, 30))
        self.img_b = self._write_png("b.png", (5, 3), (200, 100, 50))
        self.img_c = self._write_png("c.png", (2, 7), (0, 255, 128))
        self.missing = os.path.join(self.dir, "missing.png")
        self.broken = os.path.join(self.dir, "broken.jpg")
        with open(self.broken, "wb") as f:
            f.write(b"\x00\x01not an image\xff\xfe" * 8)
        self.clock = FakeClock()
        self.issuer = FrameIdIssuer(RUN_ID)

    def tearDown(self):
        self._tmp.cleanup()

    def _write_png(self, name, hw, bgr):
        h, w = hw
        image = np.empty((h, w, 3), dtype=np.uint8)
        image[:] = bgr
        path = os.path.join(self.dir, name)
        self.assertTrue(cv2.imwrite(path, image))
        return path

    def _source(self, paths, **kw):
        kw.setdefault("source_id", "stored0")
        kw.setdefault("issuer", self.issuer)
        kw.setdefault("clock_ns", self.clock)
        return StoredImageSource(paths, **kw)


class OrderAndSequenceTest(StoredInputTestBase):
    def test_reads_in_explicit_order(self):
        paths = [self.img_c, self.missing, self.img_a, self.img_b]  # 정렬되지 않은 명시 순서
        src = self._source(paths)
        results = [src.read() for _ in range(4)]
        frames = [r for r in results if isinstance(r, FramePacket)]
        self.assertEqual([f.source_ref for f in frames], [self.img_c, self.img_a, self.img_b])
        self.assertEqual([f.source_index for f in frames], [0, 2, 3])  # 실패 뒤에도 목록 위치
        self.assertEqual([f.image_seq for f in frames], [0, 1, 2])
        self.assertIsInstance(results[1], InputStatusEvent)
        self.assertEqual(results[1].source_ref, self.missing)

    def test_paths_snapshot(self):
        paths = [self.img_a, self.img_b]
        src = self._source(paths)
        paths.reverse()
        paths.append(self.img_c)
        first, second, third = src.read(), src.read(), src.read()
        self.assertEqual([first.source_ref, second.source_ref], [self.img_a, self.img_b])
        self.assertIsInstance(third, InputStatusEvent)
        self.assertIs(third.status, InputStatus.END_OF_STREAM)

    def test_pathlike_is_stringified(self):
        import pathlib
        src = self._source([pathlib.Path(self.img_a)])
        frame = src.read()
        self.assertEqual(frame.source_ref, self.img_a)
        self.assertIsInstance(frame.source_ref, str)

    def test_seq_monotonic_across_sources(self):
        s1 = self._source([self.img_a, self.img_b], source_id="s1")
        s2 = self._source([self.img_c, self.img_a], source_id="s2")
        got = [s1.read(), s2.read(), s1.read(), s2.read()]
        self.assertEqual([g.image_seq for g in got], [0, 1, 2, 3])
        self.assertEqual([g.source_id for g in got], ["s1", "s2", "s1", "s2"])

    def test_seq_not_consumed_on_failure(self):
        src = self._source([self.img_a, self.missing, self.broken, self.img_b])
        results = [src.read() for _ in range(4)]
        self.assertEqual(results[0].image_seq, 0)
        self.assertEqual(results[3].image_seq, 1)
        self.assertEqual(self.issuer.next_seq(), 2)


class StatusEventTest(StoredInputTestBase):
    def test_status_events_have_no_image_seq(self):
        src = self._source([self.missing, self.broken])
        events = [src.read(), src.read(), src.read()]
        for ev in events:
            self.assertIsInstance(ev, InputStatusEvent)
            self.assertNotIsInstance(ev, FramePacket)
            self.assertFalse(hasattr(ev, "image_seq"))
            self.assertFalse(hasattr(ev, "image"))
            self.assertEqual(ev.run_id, RUN_ID)
        self.assertEqual(self.issuer.next_seq(), 0)

    def test_missing_vs_decode_failed_vs_end(self):
        src = self._source([self.missing, self.dir, self.broken])
        missing, directory, broken, end = (src.read() for _ in range(4))
        self.assertIs(missing.status, InputStatus.SOURCE_UNAVAILABLE)
        self.assertEqual(missing.source_ref, self.missing)
        self.assertIs(directory.status, InputStatus.SOURCE_UNAVAILABLE)  # 일반 파일이 아님
        self.assertEqual(directory.source_ref, self.dir)
        self.assertIs(broken.status, InputStatus.DECODE_FAILED)
        self.assertEqual(broken.source_ref, self.broken)
        self.assertIs(end.status, InputStatus.END_OF_STREAM)
        self.assertIsNone(end.source_ref)
        details = {missing.detail, broken.detail, end.detail}
        self.assertEqual(len(details), 3)  # 원인이 구분되는 설명

    def test_expected_decode_exception_becomes_event(self):
        def raising_imread(path, flags):
            raise cv2.error("synthetic decode error")

        fake_cv2 = types.SimpleNamespace(
            imread=raising_imread, error=cv2.error, IMREAD_COLOR=cv2.IMREAD_COLOR)
        with mock.patch.object(stored_images, "cv2", fake_cv2):
            src = self._source([self.img_a, self.img_b])
            ev = src.read()
        self.assertIsInstance(ev, InputStatusEvent)
        self.assertIs(ev.status, InputStatus.DECODE_FAILED)
        self.assertIn("synthetic decode error", ev.detail)
        self.assertEqual(self.issuer.next_seq(), 0)

    def test_unexpected_exception_is_not_swallowed(self):
        def broken_imread(path, flags):
            raise RuntimeError("programming error")

        fake_cv2 = types.SimpleNamespace(
            imread=broken_imread, error=cv2.error, IMREAD_COLOR=cv2.IMREAD_COLOR)
        with mock.patch.object(stored_images, "cv2", fake_cv2):
            src = self._source([self.img_a])
            with self.assertRaises(RuntimeError):
                src.read()

    def test_no_stale_frame_after_end(self):
        src = self._source([self.img_a])
        frame = src.read()
        self.assertIs(src.current(), frame)
        for _ in range(3):
            ev = src.read()
            self.assertIsInstance(ev, InputStatusEvent)
            self.assertIs(ev.status, InputStatus.END_OF_STREAM)
            self.assertIsNone(src.current())

    def test_no_stale_frame_after_failure(self):
        src = self._source([self.img_a, self.broken, self.missing])
        src.read()
        self.assertIsNotNone(src.current())
        self.assertIsInstance(src.read(), InputStatusEvent)
        self.assertIsNone(src.current())
        self.assertIsInstance(src.read(), InputStatusEvent)
        self.assertIsNone(src.current())

    def test_empty_list_repeated_read(self):
        src = self._source([])
        for _ in range(3):
            ev = src.read()
            self.assertIs(ev.status, InputStatus.SOURCE_UNAVAILABLE)
            self.assertEqual(ev.detail, "empty list")
            self.assertIsNone(ev.source_ref)
        self.assertIsNone(src.current())
        self.assertEqual(self.issuer.next_seq(), 0)

    def test_close_then_read_and_current(self):
        src = self._source([self.img_a, self.img_b])
        frame = src.read()
        before = frame.image.copy()
        src.close()
        src.close()  # 반복 호출 안전
        self.assertIsNone(src.current())
        with mock.patch.object(stored_images, "cv2") as spy:
            for _ in range(2):
                ev = src.read()
                self.assertIs(ev.status, InputStatus.SOURCE_UNAVAILABLE)
                self.assertEqual(ev.detail, "closed")
                self.assertIsNone(ev.source_ref)
            spy.imread.assert_not_called()
        self.assertIsNone(src.current())
        self.assertEqual(self.issuer.next_seq(), 1)
        np.testing.assert_array_equal(frame.image, before)


class TimeAndBufferTest(StoredInputTestBase):
    def test_reprocess_keeps_ids_and_times(self):
        src = self._source([self.img_a], capture_generation=3)
        frame = src.read()
        calls = self.clock.calls
        again = [src.current() for _ in range(3)]
        for item in again:
            self.assertIs(item, frame)
            self.assertEqual(
                (item.image_seq, item.timestamp_ms, item.received_monotonic_ns,
                 item.capture_generation),
                (frame.image_seq, frame.timestamp_ms, frame.received_monotonic_ns, 3),
            )
        self.assertEqual(self.clock.calls, calls)
        self.assertEqual(self.issuer.next_seq(), 1)

    def test_pc_receive_single_clock_read(self):
        src = self._source([self.img_a, self.missing])
        frame = src.read()
        self.assertEqual(self.clock.calls, 1)
        self.assertEqual(frame.received_monotonic_ns, 5_000_000_789)
        self.assertEqual(frame.timestamp_ms, 5_000)  # 5_000_000_789 // 1_000_000
        self.assertIs(frame.timestamp_base, TimestampBase.PC_RECEIVE)
        ev = src.read()
        self.assertEqual(self.clock.calls, 2)
        self.assertEqual(ev.event_monotonic_ns, 6_000_000_912)

    def test_image_buffer_independent(self):
        src = self._source([self.img_a, self.img_a])
        f1, f2 = src.read(), src.read()
        self.assertFalse(np.shares_memory(f1.image, f2.image))
        self.assertFalse(f1.image.flags.writeable)
        self.assertFalse(f2.image.flags.writeable)
        with self.assertRaises(ValueError):
            f1.image[0, 0, 0] = 1
        mine = f1.image.copy()  # 소비자는 복사해서 수정
        mine[:] = 0
        self.assertEqual(int(f1.image[0, 0, 0]), 10)
        self.assertEqual(int(f2.image[0, 0, 0]), 10)

    def test_bgr_uint8_shape(self):
        src = self._source([self.img_a, self.img_b], source_id="stored9", capture_generation=2)
        for expected_hw, bgr in (((4, 6), (10, 20, 30)), ((5, 3), (200, 100, 50))):
            frame = src.read()
            self.assertEqual(frame.image.dtype, np.uint8)
            self.assertEqual(frame.image.shape, (*expected_hw, 3))
            self.assertEqual((frame.height, frame.width), expected_hw)
            np.testing.assert_array_equal(frame.image[0, 0], np.array(bgr, dtype=np.uint8))
            self.assertEqual(frame.run_id, RUN_ID)
            self.assertEqual(frame.source_id, "stored9")
            self.assertEqual(frame.capture_generation, 2)
            self.assertIsNone(frame.crop)
            self.assertIsNone(frame.calibration_id)
            self.assertIsNone(frame.optical_frame)
            self.assertIsNone(frame.ros_stamp_ns)


class PublicSampleTest(unittest.TestCase):
    def test_public_samples_decode(self):
        paths = [os.path.join(REPO_ROOT, p) for p in PUBLIC_SAMPLES]
        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            self.skipTest(f"UNVERIFIED: public samples missing: {missing}")
        src = StoredImageSource(paths, source_id="public0", issuer=FrameIdIssuer(RUN_ID))
        for i, p in enumerate(paths):
            frame = src.read()
            self.assertIsInstance(frame, FramePacket, p)
            self.assertEqual(frame.image.shape, (640, 640, 3))
            self.assertEqual(frame.image.dtype, np.uint8)
            self.assertEqual(frame.source_index, i)
        self.assertIs(src.read().status, InputStatus.END_OF_STREAM)


if __name__ == "__main__":
    unittest.main()
