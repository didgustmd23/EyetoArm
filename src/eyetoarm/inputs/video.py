"""저장 영상 입력.

근거: A 상세설계 §5.1(공통 프레임)·§5.4(입력 상태·세대)·§6(입력 소스별 동작의 「저장 영상」 행),
계획서 §2.2(`VIDEO_POS`). 아래 「사용자 확정」 항목은 2026-11-13에 정한 구현 규칙이다.

사용법:
    source = VideoSource("clip.avi", source_id="video", issuer=issuer,
                         generation_provider=pipeline.current_generation)
    item = source.read()   # FramePacket 또는 InputStatusEvent
    source.seek(120)       # 다음 read()가 120번째 프레임을 읽음
    source.close()

설계 문서와 다른 점:
- 상세 §6은 `open(config)`·`read()`·`close()`를 「제안」한다. 이 모듈은 `StoredImageSource`와 같이
  생성자로 경로를 받고 `open()`을 따로 두지 않는다(사용자 확정). 설정 파일 형식은 아직 없다.
- 열기 실패는 예외가 아니라 첫 `read()`가 돌려주는 입력 사건이다.

시각(사용자 확정):
- `timestamp_base`는 `VIDEO_POS`, `timestamp_ms`는 프레임 번호 × 1000 / fps를 반올림한 정수 ms다.
  원점은 파일 시작이다.
- OpenCV의 `CAP_PROP_POS_MSEC`는 쓰지 않는다. OpenCV 4.6.0에서 mp4v·FFV1 영상은 끝쪽 13~14프레임에서
  이 값이 0으로 나온다.
- fps가 0 이하이거나 유한하지 않으면 열기 실패(`DECODE_FAILED`)로 본다. 가변 fps 영상은 다루지 않는다.

세대(사용자 확정):
- 세대를 올리는 것은 파이프라인이다(상세 §5.4). 이 소스는 프레임을 받을 때 `generation_provider()`를
  한 번 읽어 `capture_generation`에 싣기만 한다. `seek()`도 세대를 바꾸지 않으므로, 탐색을 일으킨 쪽이
  파이프라인 세대를 올려야 한다.

하지 않는 일: 검출·보정·좌표 변환·TCP·GUI, 정답 자료 읽기. `VIDEO_POS`는 C의 벨트 시각과 대응하지
않으므로 벨트 연동 분류의 입력으로 쓰지 않는다(계획 §2.2 제안). Qt·ROS는 import하지 않는다.
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
    """저장 영상 파일을 프레임 단위로 읽는 입력 소스.

    `read()`가 돌려주는 것
    - 프레임을 읽으면 `FramePacket`, 읽지 못하면 `InputStatusEvent`(프레임 번호를 만들지 않음).
    - 상태 코드:
      - `SOURCE_UNAVAILABLE`: 경로가 없음, 일반 파일이 아님, 읽기 권한 없음, 이미 `close()`함.
      - `DECODE_FAILED`: 파일은 있지만 영상으로 열지 못함, fps가 잘못됨, 중간 프레임 해석 실패.
      - `END_OF_STREAM`: 영상 끝. 장애가 아니다. `seek()`하기 전까지 매번 같은 사건을 돌려준다.
    - 파일은 처음 `read()`나 `seek()`를 부를 때 연다. 열기에 실패하면 그 결과가 고정돼 이후에도
      같은 사건을 돌려준다.

    끝과 중간 실패를 가르는 기준(사용자 확정)
    - 영상의 전체 프레임 수(`CAP_PROP_FRAME_COUNT`)를 알면, 읽기에 실패했을 때 디코더 위치가
      전체 프레임 수에 닿았으면 끝, 그 전이면 중간 실패다.
    - 전체 프레임 수를 모르면(0 이하), 한 장도 못 읽은 첫 실패는 중간 실패, 그 뒤 실패는 끝으로 본다.
    - 중간 실패 뒤에는 다음 프레임 번호를 디코더 위치에 맞춘다. 디코더가 실패한 프레임을 건너뛰었다면
      다음 프레임의 `source_index`·`timestamp_ms`도 실제 위치를 따른다.

    프레임 필드
    - `image_seq`는 주입받은 발급기로 성공한 프레임마다 하나씩 받는다. 탐색·소스 전환에도
      초기화하지 않는다.
    - `received_monotonic_ns`는 프레임을 받은 PC 시각이다. 수신 감시용이며 `timestamp_ms`와 기준이
      다르므로 서로 빼지 않는다.
    - `source_index`는 영상 안의 프레임 번호(0부터), `source_ref`는 생성할 때 받은 경로다.
    - 프레임 배열은 매번 복사한 독립 버퍼이고 읽기 전용으로 표시한다. 다른 별칭으로 고치는 것까지
      막지는 못하므로, 소비자가 고치려면 복사해서 쓴다.
    - `crop`·`calibration_id`·`optical_frame`·`ros_stamp_ns`는 None이다. 보정 연결은 아직 없다.

    그 밖의 계약
    - 성공한 프레임 하나에 시계·세대 공급자·발급기를 각각 한 번만 부른다. 입력 사건은 시계만 한 번 읽는다.
    - `read()`는 부르는 즉시 다음 프레임을 디코딩한다. 영상 fps에 맞춰 기다리지 않는다.
    - 한 스레드에서 차례로 부른다고 가정한다. 여러 스레드에서 동시에 부르는 것은 지원하지 않는다.
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
        """다음 `read()`가 `frame_index`번째 프레임(0부터)을 읽게 한다.

        성공하면 `current()`를 비우고 영상 끝 상태를 푼다. 세대·`image_seq`·시계는 건드리지 않는다.
        예외가 나면 위치·`current()`·끝 상태는 그대로다.

        Raises:
            TypeError: 정수가 아님(bool 포함).
            ValueError: 음수이거나, 전체 프레임 수를 알 때 그 수 이상.
            RuntimeError: 이미 닫혔거나, 파일을 열지 못한 소스이거나, 디코더가 탐색을 거부함.
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
