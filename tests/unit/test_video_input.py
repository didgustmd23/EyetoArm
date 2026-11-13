"""inputs/video.py 저장 영상 입력 단위시험.

근거: A 상세설계 §5.1·§5.4·§6(저장 영상 행), 계획서 §2.2(`VIDEO_POS`), `inputs/video.py` docstring의
「사용자 확정」 규칙(2026-11-13). 시험 이름의 `r<번호>`는 검토 보고서의 요구 번호와 맞추기 위한 것이다.
시험 영상은 임시 폴더에 `cv2.VideoWriter`로 만든다. 프레임 i의 위쪽 띠에 i를 6비트 흑백
블록(8×8 px)으로 새겨 읽은 프레임의 번호를 내용으로 역산한다(손실 코덱에도 안정).
코덱 쓰기에 실패하면 건너뛰지 않고 실패한다.
"""
import ast
import os
import stat
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from eyetoarm.inputs import video
from eyetoarm.inputs.base import (
    FrameIdIssuer,
    FramePacket,
    InputStatus,
    InputStatusEvent,
    TimestampBase,
)
from eyetoarm.inputs.stored_images import StoredImageSource
from eyetoarm.inputs.video import VideoSource, video_pos_ms

RUN_ID = "run-video-test"
W, H = 64, 48
CODECS = {"MJPG": ".avi", "mp4v": ".mp4", "FFV1": ".avi"}

# 동영상 컨테이너로 해석되지 않는 고정 바이트. 무작위 바이트는 실행마다 내용이 달라
# OpenCV 4.6(FFmpeg)이 드물게 열기에서 멈췄다(30회 중 1회 재현).
_GARBAGE = bytes(range(256)) * 16


class FakeClock:
    """호출마다 증가하는 가짜 단조 시계. 호출 수를 센다."""

    def __init__(self, start_ns=7_000_000_001, step_ns=1_000_000_007):
        """시작 시각과 호출마다 늘릴 간격(ns)을 받는다. 1 ms로 나누어떨어지지 않는 값을 기본으로 쓴다."""
        self.calls = 0
        self._now = start_ns
        self._step = step_ns

    def __call__(self):
        """현재 값을 돌려주고 간격만큼 늘린 뒤 호출 수를 1 올린다."""
        self.calls += 1
        v = self._now
        self._now += self._step
        return v


class Gen:
    """파이프라인 세대 공급자 흉내. 호출 수를 센다."""

    def __init__(self, value=0):
        """돌려줄 세대 값을 받는다. 시험이 `value`를 바꿔 세대 증가를 흉내 낸다."""
        self.value = value
        self.calls = 0

    def __call__(self):
        """현재 세대 값을 돌려주고 호출 수를 1 올린다."""
        self.calls += 1
        return self.value


def write_video(path, n, fps=10.0, fourcc="MJPG", size=(W, H), color=None):
    """프레임 i의 위쪽 띠에 i를 6비트 블록으로 새긴 n프레임 영상을 쓰고 경로를 돌려준다. 쓰기 실패·빈 파일이면 AssertionError."""
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, size)

    if not w.isOpened():
        raise AssertionError(f"VideoWriter를 열지 못함: {fourcc} {path}")

    for i in range(n):
        f = np.full((size[1], size[0], 3), 128, np.uint8) if color is None else np.full(
            (size[1], size[0], 3), color, np.uint8)
        f[:16, :48] = 0

        for b in range(6):
            if (i >> b) & 1:
                f[4:12, 8 * b:8 * b + 8] = 255

        w.write(f)

    w.release()

    if not os.path.getsize(path):
        raise AssertionError(f"빈 영상: {path}")

    return path


def frame_no(frame):
    """`write_video`가 새긴 6비트 블록을 읽어 프레임 번호를 역산한다(블록 가운데 평균 > 127이면 1)."""
    g = frame.image[5:11, :48].mean(axis=(0, 2))
    return sum(1 << b for b in range(6) if g[8 * b + 2:8 * b + 6].mean() > 127)


class VideoTestBase(unittest.TestCase):
    """20프레임·10 fps 시험 영상과 가짜 시계·세대 공급자(값 3)·발급기를 준비하는 저장 영상 시험 기반."""

    def setUp(self):
        """임시 폴더에 20프레임 MJPG 영상을 만들고 가짜 시계·발급기·세대 공급자를 준비한다."""
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.vid = write_video(os.path.join(self.dir, "v.avi"), 20, fps=10.0)
        self.clock = FakeClock()
        self.issuer = FrameIdIssuer(RUN_ID)
        self.gen = Gen(3)
        self.addCleanup(self._tmp.cleanup)  # 정리 함수는 역순으로 돌므로 다른 정리(권한 복구 등)가 먼저 돈다

    def src(self, path=None, **kw):
        """기본 소스 ID·공용 발급기·세대 공급자·가짜 시계로 `VideoSource`를 만들고, 시험이 끝나면 닫히게 등록한다."""
        kw.setdefault("source_id", "video_test")
        kw.setdefault("issuer", self.issuer)
        kw.setdefault("generation_provider", self.gen)
        kw.setdefault("clock_ns", self.clock)
        s = VideoSource(path or self.vid, **kw)
        self.addCleanup(s.close)
        return s

    def read_all(self, s):
        """프레임이 아닌 결과가 나올 때까지 읽어 (프레임 목록, 마지막 사건)을 돌려준다."""
        out = []

        while True:
            r = s.read()

            if not isinstance(r, FramePacket):
                return out, r

            out.append(r)


# ---------------------------------------------------------------- A. 인터페이스·입력 상태
class InterfaceAndStatusTest(VideoTestBase):
    """`read()` 반환 형식과 입력 상태 구분, 자원 해제를 확인한다(A 상세설계 §5.4)."""

    def test_r1_read_returns_only_frame_or_event_and_close(self):
        """`read()`는 영상 끝을 지나도 `FramePacket`이나 `InputStatusEvent`만 돌려주고, `close()` 뒤에는 사건을 돌려준다."""
        s = self.src()
        kinds = {type(s.read()) for _ in range(25)}

        self.assertTrue(kinds <= {FramePacket, InputStatusEvent})

        s.close()

        self.assertIsInstance(s.read(), InputStatusEvent)

    def test_r2_events_have_no_frame_id_and_one_clock_read(self):
        """입력 사건에는 프레임 번호·영상·시각이 없고, 시계를 한 번만 읽으며 `image_seq`를 쓰지 않는다."""
        s = self.src(os.path.join(self.dir, "missing.avi"))
        before = self.clock.calls

        e = s.read()

        self.assertIsInstance(e, InputStatusEvent)

        for name in ("image_seq", "image", "timestamp_ms"):
            self.assertFalse(hasattr(e, name), name)

        self.assertEqual(self.clock.calls, before + 1)
        self.assertEqual(e.run_id, RUN_ID)
        self.assertEqual(e.source_id, "video_test")
        self.assertEqual(self.issuer.next_seq(), 0)  # 사건은 순번을 쓰지 않음

    def test_r3_missing_directory_unreadable_are_source_unavailable(self):
        """없는 경로·폴더·읽기 권한 없는 파일은 모두 SOURCE_UNAVAILABLE이고 `source_ref`에 그 경로가 남는다."""
        missing = os.path.join(self.dir, "missing.avi")
        cases = [missing, self.dir]
        unreadable = os.path.join(self.dir, "locked.avi")
        write_video(unreadable, 3)
        os.chmod(unreadable, 0)
        self.addCleanup(os.chmod, unreadable, stat.S_IRUSR | stat.S_IWUSR)

        if not os.access(unreadable, os.R_OK):  # root가 아니면 읽기 불가
            cases.append(unreadable)

        for path in cases:
            e = self.src(path).read()

            self.assertIs(e.status, InputStatus.SOURCE_UNAVAILABLE, path)
            self.assertEqual(e.source_ref, path)

        self.assertGreaterEqual(len(cases), 3)

    def test_r4_garbage_file_is_decode_failed(self):
        """영상이 아닌 고정 바이트 파일·빈 파일은 DECODE_FAILED이고, 다시 읽어도 같은 결과이며 `current()`는 None이다."""
        for name in ("junk.avi", "junk.mp4", "empty.avi"):
            p = os.path.join(self.dir, name)

            with open(p, "wb") as f:
                f.write(b"" if name.startswith("empty") else _GARBAGE)

            s = self.src(p)
            e = s.read()

            self.assertIs(e.status, InputStatus.DECODE_FAILED, name)
            self.assertIs(s.read().status, InputStatus.DECODE_FAILED)  # 열기 결과 고정
            self.assertIsNone(s.current())

    def test_r5_end_of_stream_after_all_frames(self):
        """20프레임을 순서대로 모두 읽은 뒤에는 실패가 아닌 END_OF_STREAM이다."""
        s = self.src()
        frames, end = self.read_all(s)

        self.assertEqual(len(frames), 20)
        self.assertIs(end.status, InputStatus.END_OF_STREAM)
        self.assertNotIn(end.status, (InputStatus.SOURCE_UNAVAILABLE, InputStatus.DECODE_FAILED))
        self.assertEqual([frame_no(f) for f in frames], list(range(20)))

    def test_r6_after_end_current_cleared_repeated_end_no_seq(self):
        """영상 끝 뒤에는 `current()`가 None이고, 다시 읽으면 매번 시계만 한 번 읽는 END_OF_STREAM이며 순번을 쓰지 않는다."""
        s = self.src()
        frames, _ = self.read_all(s)

        self.assertIsNone(s.current())

        calls = self.clock.calls

        for _ in range(3):
            self.assertIs(s.read().status, InputStatus.END_OF_STREAM)

        self.assertEqual(self.clock.calls, calls + 3)
        self.assertEqual(self.issuer.next_seq(), frames[-1].image_seq + 1)

    def test_r7_close_idempotent_releases_and_keeps_frames(self):
        """`close()`는 두 번 불러도 되고 캡처를 놓는다. 이후 `read()`는 "closed" 사건이며 이미 내보낸 프레임은 바뀌지 않는다."""
        s = self.src()
        f = s.read()
        before = f.image.copy()

        s.close()
        s.close()

        self.assertIsNone(s._cap)

        e = s.read()

        self.assertIs(e.status, InputStatus.SOURCE_UNAVAILABLE)
        self.assertEqual(e.detail, "closed")
        self.assertIsNone(e.source_ref)
        self.assertTrue(np.array_equal(f.image, before))
        self.assertEqual(self.issuer.next_seq(), 1)

    def test_r7_release_on_open_failure_and_end(self):
        """열기에 실패하면 캡처를 즉시 해제하고, 정상 영상은 끝까지 읽고 `close()`할 때 한 번 해제한다."""
        released = []
        real = cv2.VideoCapture

        class Spy:
            """실제 `cv2.VideoCapture`를 감싸 `release()` 호출을 기록하는 대리 객체."""

            def __init__(self, path):
                """실제 `VideoCapture`로 경로를 연다."""
                self._c = real(path)

            def __getattr__(self, n):
                """`release` 밖의 속성은 실제 캡처에 넘긴다."""
                return getattr(self._c, n)

            def release(self):
                """해제 호출을 기록하고 실제 캡처를 해제한다."""
                released.append(True)
                self._c.release()

        p = os.path.join(self.dir, "junk.avi")

        with open(p, "wb") as fh:
            fh.write(_GARBAGE)

        with mock.patch.object(video.cv2, "VideoCapture", Spy):
            self.src(p).read()

            self.assertEqual(len(released), 1)  # 열기 실패 시 즉시 해제

            s = self.src()
            self.read_all(s)

            s.close()

            self.assertEqual(len(released), 2)


# ---------------------------------------------------------------- B. 프레임 필드·시각
class FrameFieldTest(VideoTestBase):
    """프레임 필드와 `VIDEO_POS` 시각·시계·순번·버퍼 규칙을 확인한다(A 상세설계 §5.1)."""

    def test_r8_r9_fields(self):
        """파란 영상 첫 프레임은 BGR uint8 (H, W, 3)이고, 실행·소스·경로·번호 0·세대 3이 들어가며 보정 관련 필드는 None이다."""
        blue = write_video(os.path.join(self.dir, "blue.avi"), 3, color=(255, 0, 0))
        f = self.src(blue).read()

        self.assertEqual(f.image.shape, (H, W, 3))
        self.assertEqual((f.width, f.height), (W, H))
        self.assertEqual(f.image.dtype, np.uint8)

        b, g, r = f.image[16:, :].reshape(-1, 3).mean(axis=0)

        self.assertGreater(b, 200)
        self.assertLess(max(g, r), 60)  # BGR 순서
        self.assertEqual(f.run_id, RUN_ID)
        self.assertEqual(f.source_id, "video_test")
        self.assertEqual(f.source_ref, blue)
        self.assertEqual(f.source_index, 0)
        self.assertEqual(f.capture_generation, 3)

        for name in ("crop", "calibration_id", "optical_frame", "ros_stamp_ns"):
            self.assertIsNone(getattr(f, name), name)

    def test_r10_video_pos_timestamps(self):
        """시각 기준은 VIDEO_POS이고 `timestamp_ms`는 정수 프레임 번호÷fps(0, 100, …)이며 PC 수신 시각과 다르다."""
        frames, _ = self.read_all(self.src())

        self.assertTrue(all(f.timestamp_base is TimestampBase.VIDEO_POS for f in frames))
        self.assertTrue(all(type(f.timestamp_ms) is int for f in frames))
        self.assertEqual(frames[0].timestamp_ms, 0)
        self.assertEqual(frames[1].timestamp_ms, 100)
        self.assertNotEqual(frames[5].timestamp_ms, frames[5].received_monotonic_ns // 1_000_000)

    def test_r10_rounding(self):
        """`video_pos_ms`는 ms를 반올림한다(30 fps에서 33·67, 0.5 ms는 1로 올림)."""
        self.assertEqual(video_pos_ms(1, 30.0), 33)
        self.assertEqual(video_pos_ms(2, 30.0), 67)
        self.assertEqual(video_pos_ms(1, 2000.0), 1)  # 0.5는 올림

    def test_r11_clock_read_once_per_frame_and_event(self):
        """프레임 하나를 읽으면 시계와 세대 공급자를 한 번씩만 부르고, 수신 시각은 그 시계 값이다."""
        s = self.src()
        c0 = self.clock.calls

        f = s.read()

        self.assertEqual(self.clock.calls, c0 + 1)
        self.assertEqual(f.received_monotonic_ns, 7_000_000_001 + c0 * 1_000_000_007)
        self.assertEqual(self.gen.calls, 1)  # 세대도 한 번

    def test_r12_shared_issuer_and_seq_after_failure(self):
        """두 영상 소스가 발급기 하나를 함께 쓰면 순번이 이어지고, 다른 소스의 실패는 순번을 쓰지 않는다."""
        a = self.src()
        b = self.src(write_video(os.path.join(self.dir, "b.avi"), 5))

        seqs = [a.read().image_seq, b.read().image_seq, a.read().image_seq, b.read().image_seq]

        self.assertEqual(seqs, [0, 1, 2, 3])

        bad = self.src(os.path.join(self.dir, "missing.avi"))
        bad.read()

        self.assertEqual(a.read().image_seq, 4)

    def test_r13_independent_read_only_buffers(self):
        """프레임마다 버퍼가 따로이고 읽기 전용이며, 다음 프레임을 읽어도 이전 프레임 내용이 바뀌지 않는다."""
        s = self.src()
        f1, f2 = s.read(), s.read()

        self.assertFalse(np.shares_memory(f1.image, f2.image))
        self.assertFalse(f1.image.flags.writeable)

        with self.assertRaises(ValueError):
            f1.image[0, 0, 0] = 1

        keep = f1.image.copy()
        s.read()

        self.assertTrue(np.array_equal(f1.image, keep))


# ---------------------------------------------------------------- C. 탐색·세대·재처리
class SeekGenerationTest(VideoTestBase):
    """탐색·세대·재처리 규칙을 확인한다(A 상세설계 §5.1·§5.4)."""

    def test_r14_seek_reads_target_and_position(self):
        """`seek(12)` 뒤 다음 읽기는 내용·`source_index`가 12인 프레임이고 재생 위치가 13으로 간다."""
        s = self.src()
        s.read()

        s.seek(12)

        self.assertEqual(s.next_frame_index, 12)

        f = s.read()

        self.assertEqual(frame_no(f), 12)
        self.assertEqual(f.source_index, 12)
        self.assertEqual(s.next_frame_index, 13)

    def test_r15_generation_from_provider(self):
        """프레임 세대는 공급자 값을 따른다. 소스는 `seek()`만으로 세대 공급자를 부르지 않는다."""
        s = self.src()
        before = s.read()
        self.gen.value = 4  # 파이프라인이 탐색에 맞춰 세대를 올림
        s.seek(5)

        after = s.read()

        self.assertEqual(before.capture_generation, 3)
        self.assertEqual(after.capture_generation, 4)

        calls = self.gen.calls
        s.seek(6)  # 소스는 탐색만으로 세대를 읽거나 바꾸지 않음

        self.assertEqual(self.gen.calls, calls)

    def test_r16_seq_not_reset_by_seek(self):
        """앞으로 되돌아가는 탐색에도 `image_seq`는 계속 늘고, 시각은 프레임 번호로 다시 계산된다(1번 → 100 ms)."""
        s = self.src()
        frames = [s.read() for _ in range(4)]
        s.seek(1)

        f = s.read()

        self.assertEqual(f.image_seq, frames[-1].image_seq + 1)
        self.assertEqual(f.timestamp_ms, 100)

        s.seek(1)
        g = s.read()

        self.assertEqual(g.image_seq, f.image_seq + 1)
        self.assertEqual(g.timestamp_ms, 100)

    def test_r17_seek_clears_current(self):
        """`seek()` 뒤 `current()`는 None이고, 다음 읽은 프레임이 새 현재 프레임이 된다."""
        s = self.src()
        s.read()

        s.seek(3)

        self.assertIsNone(s.current())

        f = s.read()

        self.assertIs(s.current(), f)

    def test_r18_current_reuse_reads_nothing(self):
        """`current()`를 반복해 불러도 같은 객체이고 시계·세대 공급자·발급기를 부르지 않는다."""
        s = self.src()
        f = s.read()
        c, g, n = self.clock.calls, self.gen.calls, self.issuer._next

        for _ in range(3):
            self.assertIs(s.current(), f)

        self.assertEqual((self.clock.calls, self.gen.calls, self.issuer._next), (c, g, n))

    def test_r19_new_source_continues_seq(self):
        """같은 파일을 새 소스로 다시 열거나 다른 종류 소스를 섞어도 공용 발급기 순번이 이어진다."""
        a = self.src()
        last = [a.read() for _ in range(3)][-1]
        a.close()

        b = self.src()  # 같은 파일을 새 객체로 다시 엶

        self.assertEqual(b.read().image_seq, last.image_seq + 1)

        img = StoredImageSource([], source_id="x", issuer=self.issuer)  # 다른 소스와도 공유
        img.read()

        self.assertEqual(b.read().image_seq, last.image_seq + 2)

    def test_seek_after_end_resumes(self):
        """영상 끝 뒤에도 `seek()`하면 그 위치부터 다시 읽고 끝에서 다시 END_OF_STREAM이다."""
        s = self.src()
        self.read_all(s)

        s.seek(18)

        self.assertEqual([frame_no(s.read()), frame_no(s.read())], [18, 19])
        self.assertIs(s.read().status, InputStatus.END_OF_STREAM)

    def test_seek_errors_leave_state_unchanged(self):
        """정수가 아니거나 범위 밖 탐색은 TypeError·ValueError이고 상태가 그대로다. 닫힌 소스·열기 실패 소스의 탐색은 RuntimeError이며 시계·순번을 쓰지 않는다."""
        s = self.src()
        f = s.read()

        for bad, exc in ((1.5, TypeError), (True, TypeError), ("3", TypeError), (-1, ValueError), (20, ValueError)):
            with self.assertRaises(exc):
                s.seek(bad)

            self.assertIs(s.current(), f)
            self.assertEqual(s.next_frame_index, 1)

        s.seek(np.int64(2))

        self.assertEqual(s.next_frame_index, 2)

        s.close()

        with self.assertRaises(RuntimeError):
            s.seek(0)

        bad_src = self.src(os.path.join(self.dir, "missing.avi"))
        calls, seq = self.clock.calls, self.issuer._next

        with self.assertRaises(RuntimeError):
            bad_src.seek(0)

        self.assertEqual((self.clock.calls, self.issuer._next), (calls, seq))


# ---------------------------------------------------------------- 끝·중간 실패 판정과 fps 경계(가짜 VideoCapture)
class FakeCap:
    """`cv2.VideoCapture` 대역. 전체 프레임 수·fps·실패 위치를 정해 끝과 중간 실패 판정을 시험한다."""

    def __init__(self, frames, frame_count, fps=10.0, fail_at=None, advance_on_fail=False):
        """읽을 수 있는 프레임 수, 보고할 전체 프레임 수, fps, 실패 위치, 실패 때 위치 진전 여부를 받는다."""
        self.frames, self.frame_count, self.fps, self.fail_at = frames, frame_count, fps, fail_at
        self.advance_on_fail = advance_on_fail  # 실패한 프레임을 건너뛰고 위치를 진전시키는 디코더 흉내
        self.pos = 0
        self.released = False

    def isOpened(self):
        """항상 열린 것으로 보고한다."""
        return True

    def get(self, prop):
        """fps·전체 프레임 수·현재 위치만 돌려준다(그 밖의 속성은 KeyError)."""
        return {cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_COUNT: float(self.frame_count),
                cv2.CAP_PROP_POS_FRAMES: float(self.pos)}[prop]

    def set(self, prop, value):
        """위치를 바꾼다(탐색)."""
        self.pos = int(value)
        return True

    def read(self):
        """읽을 프레임이 남았으면 4×4 검은 프레임을, 끝이거나 실패 위치면 (False, None)을 돌려준다."""
        if self.pos >= self.frames:
            return False, None

        if self.pos == self.fail_at:
            if self.advance_on_fail:
                self.pos += 1

            return False, None

        self.pos += 1
        return True, np.zeros((4, 4, 3), np.uint8)

    def release(self):
        """해제됐음을 기록한다."""
        self.released = True


class BoundaryTest(VideoTestBase):
    """끝·중간 실패 판정과 fps 경계를 가짜 캡처로 확인한다(`inputs/video.py` 「사용자 확정」 규칙)."""

    def run_fake(self, cap):
        """`cv2.VideoCapture`를 주어진 가짜 캡처로 바꿔 소스를 만들고 6번 읽은 결과를 돌려준다."""
        with mock.patch.object(video.cv2, "VideoCapture", lambda p: cap):
            s = self.src()
            return s, [s.read() for _ in range(6)]

    def status(self, r):
        """읽기 결과를 "frame" 또는 상태 값 문자열로 줄인다."""
        return "frame" if isinstance(r, FramePacket) else r.status.value

    def test_m4_failure_before_frame_count_is_decode_failed(self):
        """전체 프레임 수 5에 닿기 전(2번)에서 읽기에 실패하면 끝이 아닌 DECODE_FAILED다."""
        _, rs = self.run_fake(FakeCap(frames=5, frame_count=5, fail_at=2))
        self.assertEqual([self.status(r) for r in rs[:3]], ["frame", "frame", "DECODE_FAILED"])

    def test_d1_index_follows_decoder_position_after_mid_failure(self):
        """디코더가 실패한 프레임을 건너뛰면 다음 프레임의 번호·시각이 실제 위치(3, 300 ms)를 따른다."""
        # 디코더가 실패한 프레임 2를 건너뛰면 다음 성공 프레임은 인덱스 3·시각 300 ms
        s, rs = self.run_fake(FakeCap(frames=5, frame_count=5, fail_at=2, advance_on_fail=True))

        self.assertEqual([self.status(r) for r in rs[:6]],
                         ["frame", "frame", "DECODE_FAILED", "frame", "frame", "END_OF_STREAM"])
        self.assertEqual([(r.source_index, r.timestamp_ms) for r in (rs[3], rs[4])], [(3, 300), (4, 400)])

    def test_d1_index_kept_when_decoder_does_not_advance(self):
        """실패 뒤 디코더 위치가 그대로면 다음 읽기는 같은 번호를 다시 시도해 또 DECODE_FAILED다."""
        s, rs = self.run_fake(FakeCap(frames=5, frame_count=5, fail_at=2, advance_on_fail=False))

        self.assertEqual(self.status(rs[2]), "DECODE_FAILED")
        self.assertEqual(s.next_frame_index, 2)  # 위치가 그대로면 같은 인덱스를 다시 시도
        self.assertEqual(self.status(rs[3]), "DECODE_FAILED")

    def test_m4_failure_at_frame_count_is_end(self):
        """전체 프레임 수만큼 읽은 뒤의 실패는 END_OF_STREAM이고 반복해도 같다."""
        _, rs = self.run_fake(FakeCap(frames=2, frame_count=2))
        self.assertEqual([self.status(r) for r in rs[:4]], ["frame", "frame", "END_OF_STREAM", "END_OF_STREAM"])

    def test_m4_unknown_frame_count(self):
        """전체 프레임 수를 모르면 한 장도 못 읽은 첫 실패는 DECODE_FAILED, 프레임을 읽은 뒤 실패는 END_OF_STREAM이다."""
        _, rs = self.run_fake(FakeCap(frames=0, frame_count=0))

        self.assertEqual(self.status(rs[0]), "DECODE_FAILED")

        _, rs = self.run_fake(FakeCap(frames=2, frame_count=-1))

        self.assertEqual([self.status(r) for r in rs[:3]], ["frame", "frame", "END_OF_STREAM"])

    def test_invalid_fps_is_decode_failed_and_released(self):
        """fps가 0·음수·NaN·Inf면 열기에서 "invalid fps" DECODE_FAILED이고 캡처를 해제한다."""
        for fps in (0.0, -5.0, float("nan"), float("inf")):
            cap = FakeCap(frames=3, frame_count=3, fps=fps)
            _, rs = self.run_fake(cap)

            self.assertEqual(self.status(rs[0]), "DECODE_FAILED", fps)
            self.assertIn("invalid fps", rs[0].detail)
            self.assertTrue(cap.released)


# ---------------------------------------------------------------- D. 코덱별 시각
class CodecTimestampTest(VideoTestBase):
    """실제 코덱으로 만든 영상에서 시각이 프레임 번호를 따르는지 확인한다."""

    def test_r20_timestamps_follow_frame_index_for_all_codecs(self):
        """MJPG·mp4v·FFV1, 10·30 fps 모두 40프레임 시각이 번호÷fps와 1 ms 안에서 같고, 끝쪽 탐색(35번)에서도 0으로 돌아가지 않는다."""
        for fourcc, ext in CODECS.items():
            for fps in (10.0, 30.0):
                with self.subTest(codec=fourcc, fps=fps):
                    p = write_video(os.path.join(self.dir, f"r20_{fourcc}_{int(fps)}{ext}"), 40, fps, fourcc)
                    s = self.src(p)

                    frames, end = self.read_all(s)

                    self.assertEqual(len(frames), 40)
                    self.assertIs(end.status, InputStatus.END_OF_STREAM)

                    for i, f in enumerate(frames):
                        self.assertLessEqual(abs(f.timestamp_ms - round(i * 1000 / fps)), 1, (i, f.timestamp_ms))
                        self.assertEqual(frame_no(f), i)

                    ts = [f.timestamp_ms for f in frames]

                    self.assertEqual(ts, sorted(ts))
                    self.assertGreater(ts[-1], 0)

                    s.seek(35)  # 끝쪽 탐색에서도 0으로 돌아가지 않음
                    g = s.read()

                    self.assertEqual(frame_no(g), 35)
                    self.assertLessEqual(abs(g.timestamp_ms - round(35 * 1000 / fps)), 1)


# ---------------------------------------------------------------- E. 경계
class BoundaryImportTest(unittest.TestCase):
    """`inputs/video.py`의 의존성 경계를 확인한다."""

    def test_r21_r23_no_forbidden_imports(self):
        """`inputs/video.py`는 비전·보정·변환·TCP·GUI·ROS·Qt·평가 모듈을 import하지 않고 정답(`ground_truth`)을 언급하지 않는다(코드 규약 §5·§8)."""
        tree = ast.parse(open(video.__file__, encoding="utf-8").read())
        names = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add(node.module or "")

        forbidden = ("eyetoarm.vision", "eyetoarm.calibration", "eyetoarm.transforms", "eyetoarm.tcp",
                     "eyetoarm.gui", "rclpy", "PySide6", "cv_bridge", "evaluation")

        self.assertFalse([n for n in names if n.startswith(forbidden)], names)
        self.assertNotIn("ground_truth", open(video.__file__, encoding="utf-8").read())


if __name__ == "__main__":
    unittest.main()
