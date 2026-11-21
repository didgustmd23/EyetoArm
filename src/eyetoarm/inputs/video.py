"""저장 영상 입력 소스(`VideoSource`).

`timestamp_ms`는 프레임 번호 × 1000 / fps(`VIDEO_POS`)다. OpenCV 4.6 `CAP_PROP_POS_MSEC`는 끝쪽에서
0이 나와 쓰지 않는다. 벨트 시각과 대응하지 않으므로 벨트 연동 분류 입력으로 쓰지 않는다.
"""
from __future__ import annotations

import math
import os
import time
from typing import Callable

import cv2
import numpy as np

from eyetoarm.inputs.base import (
    FrameIdIssuer,
    FramePacket,
    InputStatus,
    InputStatusEvent,
    TimestampBase,
)


def video_pos_ms(frame_index: int, fps: float) -> int:
    """프레임 인덱스의 재생 위치(ms, 정수). 0.5를 올리는 반올림."""
    return int(math.floor(frame_index * 1000.0 / fps + 0.5))


class VideoSource:
    """저장 영상 파일을 프레임 단위로 읽는 입력 소스(한 스레드에서 차례로 호출).

    `read()`는 `FramePacket` 또는 `InputStatusEvent`(열기·해석 실패 `DECODE_FAILED`, 끝 `END_OF_STREAM`).
    프레임 배열은 매번 복사한 읽기 전용 버퍼이고, fps에 맞춰 기다리지 않는다.
    """

    def __init__(self, path: str | os.PathLike, *, source_id: str, issuer: FrameIdIssuer,
                 generation_provider: Callable[[], int] = lambda: 0,
                 clock_ns: Callable[[], int] = time.monotonic_ns) -> None:
        """경로와 발급기·세대 공급자·시계를 받아 둔다. 파일은 처음 `read()`·`seek()` 때 연다."""
        self._path = os.fspath(path)
        self._source_id = source_id
        self._issuer = issuer
        self._generation_provider = generation_provider
        self._clock_ns = clock_ns
        self._cap = None
        self._opened = False
        self._open_failure: tuple[InputStatus, str] | None = None
        self._fps = 0.0
        self._frame_count = 0
        self._next_index = 0
        self._any_read = False
        self._at_end = False
        self._current: FramePacket | None = None
        self._closed = False

    # ------------------------------------------------------------ 조회
    @property
    def path(self) -> str:
        """생성할 때 받은 영상 경로(`source_ref`와 같은 값)."""
        return self._path

    @property
    def fps(self) -> float:
        """열린 영상의 fps. 열기 전·실패 시 0.0."""
        return self._fps

    @property
    def frame_count(self) -> int:
        """`CAP_PROP_FRAME_COUNT`(정수). 모르면 0 이하."""
        return self._frame_count

    @property
    def next_frame_index(self) -> int:
        """다음 `read()`가 읽을 프레임 번호(현재 재생 위치)."""
        return self._next_index

    def current(self) -> FramePacket | None:
        """마지막으로 읽은 프레임(같은 객체). 정지 화면을 다시 처리할 때 쓴다.

        시계·발급기·세대 공급자를 부르지 않는다. 생성 직후, 실패·끝·`seek()`·`close()` 뒤에는 None이다.
        """
        return self._current

    # ------------------------------------------------------------ 열기
    def _ensure_open(self) -> tuple[InputStatus, str] | None:
        """처음 한 번만 파일을 연다. 실패했으면 (상태, 설명), 성공했으면 None. 결과는 고정된다."""
        if not self._opened:
            self._opened = True
            self._open_failure = self._open()

        return self._open_failure

    def _open(self) -> tuple[InputStatus, str] | None:
        """파일을 열고 fps·전체 프레임 수를 읽는다. 실패하면 (상태, 설명), 성공하면 None."""
        # 파일에 접근할 수 없는 경우와, 파일은 있지만 영상으로 못 여는 경우를 나눈다.
        # OpenCV는 둘 다 isOpened()가 False라 구분하지 못하므로 경로를 먼저 검사한다.
        if not os.path.isfile(self._path):
            return InputStatus.SOURCE_UNAVAILABLE, "not a regular file or missing"

        if not os.access(self._path, os.R_OK):
            return InputStatus.SOURCE_UNAVAILABLE, "file not readable"

        try:
            cap = cv2.VideoCapture(self._path)
        except cv2.error as exc:
            return InputStatus.DECODE_FAILED, f"opencv open error: {exc}"

        if not cap.isOpened():
            cap.release()
            return InputStatus.DECODE_FAILED, "cannot open as video"

        fps = float(cap.get(cv2.CAP_PROP_FPS))

        if not math.isfinite(fps) or fps <= 0.0:
            cap.release()
            return InputStatus.DECODE_FAILED, f"invalid fps {fps!r}"

        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self._cap = cap
        self._fps = fps
        self._frame_count = int(frame_count) if math.isfinite(frame_count) else 0
        return None

    # ------------------------------------------------------------ 읽기
    def read(self) -> FramePacket | InputStatusEvent:
        """다음 프레임을 읽어 `FramePacket`을, 읽지 못하면 `InputStatusEvent`를 돌려준다."""
        if self._closed:
            return self._event(InputStatus.SOURCE_UNAVAILABLE, "closed", None)

        failure = self._ensure_open()

        if failure is not None:
            return self._event(failure[0], failure[1], self._path)

        if self._at_end:
            return self._event(InputStatus.END_OF_STREAM, "end of video", self._path)

        index = self._next_index

        try:
            ok, image = self._cap.read()
        except cv2.error as exc:
            ok, image = False, None
            read_error = f"opencv read error: {exc}"
        else:
            read_error = "read returned no frame"

        if not ok or image is None:
            return self._read_failed(index, read_error)

        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            # BGR 3채널 uint8이 아니면 프레임으로 내보내지 않는다(FramePacket 계약).
            self._sync_index(index + 1)
            return self._event(InputStatus.DECODE_FAILED,
                               f"unexpected array dtype={image.dtype} shape={image.shape} at frame {index}",
                               self._path)

        # 디코더가 다음 읽기에서 버퍼를 다시 쓸 수 있으므로 복사해 넘긴다.
        image = np.array(image, dtype=np.uint8, copy=True, order="C")
        image.setflags(write=False)
        self._next_index = index + 1
        self._any_read = True

        seq = self._issuer.next_seq()
        received_ns = self._clock_ns()
        generation = self._generation_provider()
        frame = FramePacket(
            image=image,
            run_id=self._issuer.run_id,
            source_id=self._source_id,
            image_seq=seq,
            timestamp_ms=video_pos_ms(index, self._fps),
            timestamp_base=TimestampBase.VIDEO_POS,
            received_monotonic_ns=received_ns,
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            crop=None,
            calibration_id=None,
            capture_generation=generation,
            optical_frame=None,
            ros_stamp_ns=None,
            source_ref=self._path,
            source_index=index,
        )
        self._current = frame
        return frame

    def _read_failed(self, index: int, detail: str) -> InputStatusEvent:
        """읽기 실패를 끝(`END_OF_STREAM`)과 중간 실패(`DECODE_FAILED`)로 나눈다. 기준은 클래스 docstring."""
        if self._frame_count > 0:
            pos = self._cap.get(cv2.CAP_PROP_POS_FRAMES)

            if math.isfinite(pos) and pos >= self._frame_count:
                self._at_end = True
                return self._event(InputStatus.END_OF_STREAM, "end of video", self._path)

            self._sync_index(index)
            return self._event(InputStatus.DECODE_FAILED,
                               f"{detail} at frame {index} of {self._frame_count}", self._path)

        if not self._any_read:
            return self._event(InputStatus.DECODE_FAILED,
                               f"{detail} on first read (frame count unknown)", self._path)

        self._at_end = True
        return self._event(InputStatus.END_OF_STREAM, "end of video (frame count unknown)", self._path)

    def _sync_index(self, fallback: int) -> None:
        """읽기 실패 뒤 다음 프레임 번호를 디코더 위치(`CAP_PROP_POS_FRAMES`)에 맞춘다.

        디코더가 실패한 프레임을 건너뛰었다면 다음 프레임의 번호와 시각도 그만큼 뒤여야 한다.
        디코더 위치가 유한하지 않거나 음수면 `fallback`을 쓴다.
        """
        pos = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
        self._next_index = int(pos) if math.isfinite(pos) and pos >= 0 else fallback

    # ------------------------------------------------------------ 탐색
    def seek(self, frame_index: int) -> None:
        """다음 `read()`가 `frame_index`번째 프레임(0부터)을 읽게 한다. 세대·`image_seq`는 건드리지 않는다.

        정수가 아니면 TypeError, 범위 밖이면 ValueError, 닫혔거나 탐색이 거부되면 RuntimeError(상태 무변).
        """
        if isinstance(frame_index, bool) or not isinstance(frame_index, (int, np.integer)):
            raise TypeError(f"frame_index must be int, got {type(frame_index).__name__}")

        frame_index = int(frame_index)

        if self._closed:
            raise RuntimeError("source is closed")

        failure = self._ensure_open()

        if failure is not None:
            raise RuntimeError(f"source not open: {failure[0].value} ({failure[1]})")

        if frame_index < 0 or (self._frame_count > 0 and frame_index >= self._frame_count):
            raise ValueError(f"frame_index {frame_index} out of range (frame count {self._frame_count})")

        if not self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
            raise RuntimeError(f"seek to frame {frame_index} failed")

        self._next_index = frame_index
        self._at_end = False
        self._current = None

    # ------------------------------------------------------------ 닫기
    def close(self) -> None:
        """디코더를 해제한다. 여러 번 불러도 된다. 이미 내보낸 프레임은 그대로 쓸 수 있다."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        self._closed = True
        self._current = None

    def _event(self, status: InputStatus, detail: str, source_ref: str | None) -> InputStatusEvent:
        """입력 사건을 만든다. 사건이 나면 `current()`를 비워 이전 프레임을 다시 내지 않는다."""
        self._current = None

        return InputStatusEvent(
            run_id=self._issuer.run_id,
            source_id=self._source_id,
            status=status,
            event_monotonic_ns=self._clock_ns(),
            detail=detail,
            source_ref=source_ref,
        )
