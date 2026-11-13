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


class FakeClock:
    """호출마다 증가하는 가짜 단조 시계. 호출 수를 센다."""

    def __init__(self, start_ns=7_000_000_001, step_ns=1_000_000_007):
        self.calls = 0
        self._now = start_ns
        self._step = step_ns

    def __call__(self):
        self.calls += 1
        v = self._now
        self._now += self._step
        return v


class Gen:
    """파이프라인 세대 공급자 흉내. 호출 수를 센다."""

    def __init__(self, value=0):
        self.value = value
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.value


def write_video(path, n, fps=10.0, fourcc="MJPG", size=(W, H), color=None):
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
    g = frame.image[5:11, :48].mean(axis=(0, 2))
    return sum(1 << b for b in range(6) if g[8 * b + 2:8 * b + 6].mean() > 127)


class VideoTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.vid = write_video(os.path.join(self.dir, "v.avi"), 20, fps=10.0)
        self.clock = FakeClock()
        self.issuer = FrameIdIssuer(RUN_ID)
        self.gen = Gen(3)
        self.addCleanup(self._tmp.cleanup)  # 정리 함수는 역순으로 돌므로 다른 정리(권한 복구 등)가 먼저 돈다

    def src(self, path=None, **kw):
        kw.setdefault("source_id", "video_test")
        kw.setdefault("issuer", self.issuer)
        kw.setdefault("generation_provider", self.gen)
        kw.setdefault("clock_ns", self.clock)
        s = VideoSource(path or self.vid, **kw)
        self.addCleanup(s.close)
        return s

    def read_all(self, s):
        out = []

        while True:
            r = s.read()

            if not isinstance(r, FramePacket):
                return out, r

            out.append(r)


# ---------------------------------------------------------------- A. 인터페이스·입력 상태
class InterfaceAndStatusTest(VideoTestBase):
    def test_r1_read_returns_only_frame_or_event_and_close(self):
        s = self.src()
        kinds = {type(s.read()) for _ in range(25)}

        self.assertTrue(kinds <= {FramePacket, InputStatusEvent})

        s.close()

        self.assertIsInstance(s.read(), InputStatusEvent)

    def test_r2_events_have_no_frame_id_and_one_clock_read(self):
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
        for name in ("junk.avi", "junk.mp4", "empty.avi"):
            p = os.path.join(self.dir, name)

            with open(p, "wb") as f:
                f.write(b"" if name.startswith("empty") else os.urandom(4096))

            s = self.src(p)
            e = s.read()

            self.assertIs(e.status, InputStatus.DECODE_FAILED, name)
            self.assertIs(s.read().status, InputStatus.DECODE_FAILED)  # 열기 결과 고정
            self.assertIsNone(s.current())

    def test_r5_end_of_stream_after_all_frames(self):
        s = self.src()
        frames, end = self.read_all(s)

        self.assertEqual(len(frames), 20)
        self.assertIs(end.status, InputStatus.END_OF_STREAM)
        self.assertNotIn(end.status, (InputStatus.SOURCE_UNAVAILABLE, InputStatus.DECODE_FAILED))
        self.assertEqual([frame_no(f) for f in frames], list(range(20)))

    def test_r6_after_end_current_cleared_repeated_end_no_seq(self):
        s = self.src()
        frames, _ = self.read_all(s)

        self.assertIsNone(s.current())

        calls = self.clock.calls

        for _ in range(3):
            self.assertIs(s.read().status, InputStatus.END_OF_STREAM)

        self.assertEqual(self.clock.calls, calls + 3)
        self.assertEqual(self.issuer.next_seq(), frames[-1].image_seq + 1)

    def test_r7_close_idempotent_releases_and_keeps_frames(self):
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
        released = []
        real = cv2.VideoCapture

        class Spy:
            def __init__(self, path):
                self._c = real(path)

            def __getattr__(self, n):
                return getattr(self._c, n)

            def release(self):
                released.append(True)
                self._c.release()

        p = os.path.join(self.dir, "junk.avi")

        with open(p, "wb") as fh:
            fh.write(os.urandom(2048))

        with mock.patch.object(video.cv2, "VideoCapture", Spy):
            self.src(p).read()

            self.assertEqual(len(released), 1)  # 열기 실패 시 즉시 해제

            s = self.src()
            self.read_all(s)

            s.close()

            self.assertEqual(len(released), 2)


# ---------------------------------------------------------------- B. 프레임 필드·시각
class FrameFieldTest(VideoTestBase):
    def test_r8_r9_fields(self):
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
        frames, _ = self.read_all(self.src())

        self.assertTrue(all(f.timestamp_base is TimestampBase.VIDEO_POS for f in frames))
        self.assertTrue(all(type(f.timestamp_ms) is int for f in frames))
        self.assertEqual(frames[0].timestamp_ms, 0)
        self.assertEqual(frames[1].timestamp_ms, 100)
        self.assertNotEqual(frames[5].timestamp_ms, frames[5].received_monotonic_ns // 1_000_000)

    def test_r10_rounding(self):
        self.assertEqual(video_pos_ms(1, 30.0), 33)
        self.assertEqual(video_pos_ms(2, 30.0), 67)
        self.assertEqual(video_pos_ms(1, 2000.0), 1)  # 0.5는 올림

    def test_r11_clock_read_once_per_frame_and_event(self):
        s = self.src()
        c0 = self.clock.calls

        f = s.read()

        self.assertEqual(self.clock.calls, c0 + 1)
        self.assertEqual(f.received_monotonic_ns, 7_000_000_001 + c0 * 1_000_000_007)
        self.assertEqual(self.gen.calls, 1)  # 세대도 한 번

    def test_r12_shared_issuer_and_seq_after_failure(self):
        a = self.src()
        b = self.src(write_video(os.path.join(self.dir, "b.avi"), 5))

        seqs = [a.read().image_seq, b.read().image_seq, a.read().image_seq, b.read().image_seq]

        self.assertEqual(seqs, [0, 1, 2, 3])

        bad = self.src(os.path.join(self.dir, "missing.avi"))
        bad.read()

        self.assertEqual(a.read().image_seq, 4)

    def test_r13_independent_read_only_buffers(self):
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
    def test_r14_seek_reads_target_and_position(self):
        s = self.src()
        s.read()

        s.seek(12)

        self.assertEqual(s.next_frame_index, 12)

        f = s.read()

        self.assertEqual(frame_no(f), 12)
        self.assertEqual(f.source_index, 12)
        self.assertEqual(s.next_frame_index, 13)

    def test_r15_generation_from_provider(self):
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
        s = self.src()
        s.read()

        s.seek(3)

        self.assertIsNone(s.current())

        f = s.read()

        self.assertIs(s.current(), f)

    def test_r18_current_reuse_reads_nothing(self):
        s = self.src()
        f = s.read()
        c, g, n = self.clock.calls, self.gen.calls, self.issuer._next

        for _ in range(3):
            self.assertIs(s.current(), f)

        self.assertEqual((self.clock.calls, self.gen.calls, self.issuer._next), (c, g, n))

    def test_r19_new_source_continues_seq(self):
        a = self.src()
        last = [a.read() for _ in range(3)][-1]
        a.close()

        b = self.src()  # 같은 파일을 새 객체로 다시 엶

        self.assertEqual(b.read().image_seq, last.image_seq + 1)

        img = StoredImageSource([], source_id="x", issuer=self.issuer)  # 다른 소스와도 공유
        img.read()

        self.assertEqual(b.read().image_seq, last.image_seq + 2)

    def test_seek_after_end_resumes(self):
        s = self.src()
        self.read_all(s)

        s.seek(18)

        self.assertEqual([frame_no(s.read()), frame_no(s.read())], [18, 19])
        self.assertIs(s.read().status, InputStatus.END_OF_STREAM)

    def test_seek_errors_leave_state_unchanged(self):
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
    def __init__(self, frames, frame_count, fps=10.0, fail_at=None, advance_on_fail=False):
        self.frames, self.frame_count, self.fps, self.fail_at = frames, frame_count, fps, fail_at
        self.advance_on_fail = advance_on_fail  # 실패한 프레임을 건너뛰고 위치를 진전시키는 디코더 흉내
        self.pos = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_COUNT: float(self.frame_count),
                cv2.CAP_PROP_POS_FRAMES: float(self.pos)}[prop]

    def set(self, prop, value):
        self.pos = int(value)
        return True

    def read(self):
        if self.pos >= self.frames:
            return False, None

        if self.pos == self.fail_at:
            if self.advance_on_fail:
                self.pos += 1

            return False, None

        self.pos += 1
        return True, np.zeros((4, 4, 3), np.uint8)

    def release(self):
        self.released = True


class BoundaryTest(VideoTestBase):
    def run_fake(self, cap):
        with mock.patch.object(video.cv2, "VideoCapture", lambda p: cap):
            s = self.src()
            return s, [s.read() for _ in range(6)]

    def status(self, r):
        return "frame" if isinstance(r, FramePacket) else r.status.value

    def test_m4_failure_before_frame_count_is_decode_failed(self):
        _, rs = self.run_fake(FakeCap(frames=5, frame_count=5, fail_at=2))
        self.assertEqual([self.status(r) for r in rs[:3]], ["frame", "frame", "DECODE_FAILED"])

    def test_d1_index_follows_decoder_position_after_mid_failure(self):
        # 디코더가 실패한 프레임 2를 건너뛰면 다음 성공 프레임은 인덱스 3·시각 300 ms
        s, rs = self.run_fake(FakeCap(frames=5, frame_count=5, fail_at=2, advance_on_fail=True))

        self.assertEqual([self.status(r) for r in rs[:6]],
                         ["frame", "frame", "DECODE_FAILED", "frame", "frame", "END_OF_STREAM"])
        self.assertEqual([(r.source_index, r.timestamp_ms) for r in (rs[3], rs[4])], [(3, 300), (4, 400)])

    def test_d1_index_kept_when_decoder_does_not_advance(self):
        s, rs = self.run_fake(FakeCap(frames=5, frame_count=5, fail_at=2, advance_on_fail=False))

        self.assertEqual(self.status(rs[2]), "DECODE_FAILED")
        self.assertEqual(s.next_frame_index, 2)  # 위치가 그대로면 같은 인덱스를 다시 시도
        self.assertEqual(self.status(rs[3]), "DECODE_FAILED")

    def test_m4_failure_at_frame_count_is_end(self):
        _, rs = self.run_fake(FakeCap(frames=2, frame_count=2))
        self.assertEqual([self.status(r) for r in rs[:4]], ["frame", "frame", "END_OF_STREAM", "END_OF_STREAM"])

    def test_m4_unknown_frame_count(self):
        _, rs = self.run_fake(FakeCap(frames=0, frame_count=0))

        self.assertEqual(self.status(rs[0]), "DECODE_FAILED")

        _, rs = self.run_fake(FakeCap(frames=2, frame_count=-1))

        self.assertEqual([self.status(r) for r in rs[:3]], ["frame", "frame", "END_OF_STREAM"])

    def test_invalid_fps_is_decode_failed_and_released(self):
        for fps in (0.0, -5.0, float("nan"), float("inf")):
            cap = FakeCap(frames=3, frame_count=3, fps=fps)
            _, rs = self.run_fake(cap)

            self.assertEqual(self.status(rs[0]), "DECODE_FAILED", fps)
            self.assertIn("invalid fps", rs[0].detail)
            self.assertTrue(cap.released)


# ---------------------------------------------------------------- D. 코덱별 시각
class CodecTimestampTest(VideoTestBase):
    def test_r20_timestamps_follow_frame_index_for_all_codecs(self):
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
    def test_r21_r23_no_forbidden_imports(self):
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
