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
        """시작 시각과 호출마다 늘릴 간격(ns)을 받는다. 1 ms 단위로 나누어떨어지지 않는 값을 기본으로 쓴다."""
        self.calls = 0
        self._now = start_ns
        self._step = step_ns

    def __call__(self):
        """현재 값을 돌려주고 간격만큼 늘린 뒤 호출 수를 1 올린다."""
        self.calls += 1
        value = self._now
        self._now += self._step
        return value


class StoredInputTestBase(unittest.TestCase):
    """합성 PNG·깨진 파일·가짜 시계·발급기를 준비하는 저장 이미지 시험 공통 기반."""

    def setUp(self):
        """임시 폴더에 크기·색이 다른 PNG 3장과 깨진 파일을 만들고, 없는 경로·가짜 시계·발급기를 준비한다."""
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
        """임시 폴더를 지운다."""
        self._tmp.cleanup()

    def _write_png(self, name, hw, bgr):
        """한 색으로 채운 (h, w) BGR PNG를 임시 폴더에 저장하고 경로를 돌려준다."""
        h, w = hw
        image = np.empty((h, w, 3), dtype=np.uint8)
        image[:] = bgr
        path = os.path.join(self.dir, name)
        self.assertTrue(cv2.imwrite(path, image))

        return path

    def _source(self, paths, **kw):
        """기본 소스 ID·공용 발급기·가짜 시계로 `StoredImageSource`를 만든다. 키워드로 바꿀 수 있다."""
        kw.setdefault("source_id", "stored0")
        kw.setdefault("issuer", self.issuer)
        kw.setdefault("clock_ns", self.clock)
        return StoredImageSource(paths, **kw)


class OrderAndSequenceTest(StoredInputTestBase):
    """명시 순서대로 읽기와 `image_seq` 발급 규칙을 확인한다(A 상세설계 §5.1)."""

    def test_reads_in_explicit_order(self):
        """정렬되지 않은 목록도 준 순서대로 읽고, `source_index`는 실패 항목을 포함한 목록 위치, `image_seq`는 성공한 것만 0부터 센다."""
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
        """생성 뒤 호출자가 목록을 뒤집거나 늘려도 소스는 생성 때 목록 순서대로 읽고 끝난다."""
        paths = [self.img_a, self.img_b]
        src = self._source(paths)
        paths.reverse()
        paths.append(self.img_c)

        first, second, third = src.read(), src.read(), src.read()

        self.assertEqual([first.source_ref, second.source_ref], [self.img_a, self.img_b])
        self.assertIsInstance(third, InputStatusEvent)
        self.assertIs(third.status, InputStatus.END_OF_STREAM)

    def test_pathlike_is_stringified(self):
        """`pathlib.Path`로 준 경로는 `source_ref`에 str로 남는다."""
        import pathlib
        src = self._source([pathlib.Path(self.img_a)])

        frame = src.read()

        self.assertEqual(frame.source_ref, self.img_a)
        self.assertIsInstance(frame.source_ref, str)

    def test_seq_monotonic_across_sources(self):
        """두 소스가 발급기 하나를 함께 쓰면 번갈아 읽어도 `image_seq`가 0~3으로 이어진다."""
        s1 = self._source([self.img_a, self.img_b], source_id="s1")
        s2 = self._source([self.img_c, self.img_a], source_id="s2")

        got = [s1.read(), s2.read(), s1.read(), s2.read()]

        self.assertEqual([g.image_seq for g in got], [0, 1, 2, 3])
        self.assertEqual([g.source_id for g in got], ["s1", "s2", "s1", "s2"])

    def test_seq_not_consumed_on_failure(self):
        """없는 파일·깨진 파일은 `image_seq`를 소비하지 않아 다음 성공 프레임이 1을 받는다."""
        src = self._source([self.img_a, self.missing, self.broken, self.img_b])
        results = [src.read() for _ in range(4)]

        self.assertEqual(results[0].image_seq, 0)
        self.assertEqual(results[3].image_seq, 1)
        self.assertEqual(self.issuer.next_seq(), 2)


class StatusEventTest(StoredInputTestBase):
    """실패·끝·닫힘을 입력 사건으로 알리고 이전 프레임을 다시 내지 않는지 확인한다(A 상세설계 §5.4)."""

    def test_status_events_have_no_image_seq(self):
        """실패·끝 사건에는 `image_seq`·영상이 없고, 발급기 순번도 쓰지 않는다."""
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
        """없는 파일·폴더는 SOURCE_UNAVAILABLE, 깨진 파일은 DECODE_FAILED, 목록 끝은 END_OF_STREAM이고 설명이 서로 다르다."""
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
        """`cv2.imread`가 `cv2.error`를 내면 예외 메시지를 담은 DECODE_FAILED 사건이 되고 순번을 쓰지 않는다."""
        def raising_imread(path, flags):
            """항상 `cv2.error`를 낸다(디코딩 오류 주입)."""
            raise cv2.error("synthetic decode error")

        # imread만 바꾼 가짜 cv2. 소스의 except가 같은 예외 형을 잡도록 error는 실제 cv2.error를 넣는다.
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
        """`cv2.error`가 아닌 예외(프로그램 오류)는 입력 사건으로 바꾸지 않고 그대로 올라온다."""
        def broken_imread(path, flags):
            """항상 RuntimeError를 낸다(프로그램 오류 주입)."""
            raise RuntimeError("programming error")

        # cv2.error가 아닌 예외를 내는 가짜 cv2. 이 예외는 사건으로 바뀌면 안 된다.
        fake_cv2 = types.SimpleNamespace(
            imread=broken_imread, error=cv2.error, IMREAD_COLOR=cv2.IMREAD_COLOR)

        with mock.patch.object(stored_images, "cv2", fake_cv2):
            src = self._source([self.img_a])

            with self.assertRaises(RuntimeError):
                src.read()

    def test_no_stale_frame_after_end(self):
        """목록 끝 뒤에는 `current()`가 None이고, 다시 읽어도 매번 END_OF_STREAM이다."""
        src = self._source([self.img_a])
        frame = src.read()

        self.assertIs(src.current(), frame)

        for _ in range(3):
            ev = src.read()

            self.assertIsInstance(ev, InputStatusEvent)
            self.assertIs(ev.status, InputStatus.END_OF_STREAM)
            self.assertIsNone(src.current())

    def test_no_stale_frame_after_failure(self):
        """실패 사건 뒤에는 `current()`가 None이 되어 이전 성공 프레임을 다시 내지 않는다."""
        src = self._source([self.img_a, self.broken, self.missing])
        src.read()

        self.assertIsNotNone(src.current())
        self.assertIsInstance(src.read(), InputStatusEvent)
        self.assertIsNone(src.current())
        self.assertIsInstance(src.read(), InputStatusEvent)
        self.assertIsNone(src.current())

    def test_empty_list_repeated_read(self):
        """빈 목록은 매번 SOURCE_UNAVAILABLE("empty list")이고 순번을 쓰지 않는다."""
        src = self._source([])

        for _ in range(3):
            ev = src.read()

            self.assertIs(ev.status, InputStatus.SOURCE_UNAVAILABLE)
            self.assertEqual(ev.detail, "empty list")
            self.assertIsNone(ev.source_ref)

        self.assertIsNone(src.current())
        self.assertEqual(self.issuer.next_seq(), 0)

    def test_close_then_read_and_current(self):
        """`close()`는 두 번 불러도 되고, 닫은 뒤 `read()`는 파일을 열지 않고 "closed" 사건을 낸다. 이미 내보낸 프레임은 바뀌지 않는다."""
        src = self._source([self.img_a, self.img_b])
        frame = src.read()
        before = frame.image.copy()

        src.close()
        src.close()  # 반복 호출 안전

        self.assertIsNone(src.current())

        # 닫은 뒤 파일을 다시 열지 않는지 imread 호출 여부로 확인한다.
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
    """수신 시각·재처리·영상 버퍼 규칙을 확인한다(A 상세설계 §5.1, 공통 초안 §4 불변조건 8)."""

    def test_reprocess_keeps_ids_and_times(self):
        """`current()`로 다시 처리하면 같은 프레임 객체를 돌려주고, 시계·발급기를 부르지 않아 순번·시각·세대가 그대로다."""
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
        """성공 프레임과 실패 사건은 시계를 한 번씩 읽고, `timestamp_ms`는 수신 ns를 1 ms로 내림한 PC_RECEIVE 시각이다."""
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
        """같은 파일을 두 번 읽어도 버퍼가 따로이고 읽기 전용이다. 소비자가 복사본을 고쳐도 원래 프레임은 그대로다."""
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
        """프레임은 파일 크기 그대로의 BGR uint8 영상이고, 실행·소스·세대 값이 들어가며 보정 관련 필드는 None이다."""
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
    """실제 공개 데이터 표본으로 읽기를 확인한다."""

    def test_public_samples_decode(self):
        """공개 데이터 train 표본 3장이 640×640 BGR uint8로 디코딩되고, 다 읽으면 END_OF_STREAM이다(파일이 없으면 UNVERIFIED로 건너뜀)."""
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
