"""저장 이미지·명시 순서 목록 입력 — A 내부 잠정안.

공동 API 확정이 아니다. 자료형은 `inputs/base.py`(공통 자료형의 A 확장)를 쓰고,
항목 단위 실패 정책(실패한 항목을 소비하고 다음 항목으로 진행)은 A 제안이다.
근거: A 상세설계 §5.1(공통 프레임)·§5.4(입력 상태)·§6(입력 소스별 동작의 「저장 이미지」 행).
"""
from __future__ import annotations

import os
import time
from typing import Callable, Sequence

import cv2

from eyetoarm.inputs.base import (
    FrameIdIssuer,
    FramePacket,
    InputStatus,
    InputStatusEvent,
    TimestampBase,
)


class StoredImageSource:
    """호출자가 정한 순서의 이미지 파일 목록을 하나씩 읽는 입력 소스.

    읽기 계약
    - `paths`는 생성 때 문자열 목록으로 복사한다(`os.fspath`). 이후 호출자가 목록을 바꿔도
      순서가 변하지 않는다. 정렬·폴더 탐색·목록 파일 해석은 하지 않는다. 상대 경로는
      프로세스 작업 디렉터리 기준이며 `source_ref`에는 전달된 경로를 그대로 남긴다.
    - `read()`는 목록 항목 하나를 소비한다. 실패한 항목도 소비하고 다음 호출은 다음
      항목으로 간다. 실패·끝에서는 현재 프레임을 비워 이전 성공 프레임을 다시 내지 않는다.
    - 일반 파일이 아니거나 읽을 수 없는 경로는 `SOURCE_UNAVAILABLE`, 존재하는 파일에서
      `cv2.imread(path, cv2.IMREAD_COLOR)`가 None이거나 `cv2.error`를 내면 `DECODE_FAILED`.
      그 밖의 예외는 입력 사건으로 바꾸지 않고 그대로 올린다.
    - 성공일 때만 `issuer.next_seq()`를 한 번 부르고 `clock_ns()`를 한 번 읽는다.
      `timestamp_base = PC_RECEIVE`, `timestamp_ms = received_monotonic_ns // 1_000_000`.
      이 시각은 파일 디코딩을 마친 PC 시각이며 촬영 시각이 아니다. 촬영 지연·실제 관측
      시간·로봇 목표 유효성의 근거로 쓰지 않는다.
    - `source_index`는 원래 목록에서의 0부터 시작하는 위치다(성공 순번이 아님).
    - 디코딩한 배열의 소유권은 프레임에 넘긴다. 소스는 그 배열을 다시 쓰지 않고 읽기 전용으로
      표시한다. 읽기 전용 표시는 다른 별칭을 통한 변경까지 막지 못한다. 소비자가 고치려면 복사한다.
    - `current()`는 마지막 성공 프레임과 같은 객체다(정지 화면 재처리용). 시계·순번을 읽지 않는다.
      생성 직후와 실패·끝·close 뒤에는 None.
    - 입력 사건은 `clock_ns()`를 한 번 읽어 `event_monotonic_ns`에 기록한다. 파일 관련 실패의
      `source_ref`는 그 경로, 목록 끝·빈 목록·닫힘은 None.
    - 빈 목록은 매번 `SOURCE_UNAVAILABLE`("empty list"), 다 읽은 목록은 매번 `END_OF_STREAM`.
      `close()`는 여러 번 불러도 되며, 이후 `read()`는 파일을 열거나 순번을 쓰지 않고 매번
      `SOURCE_UNAVAILABLE`("closed"). 이미 내보낸 프레임은 건드리지 않는다.
    - 단일 흐름에서 호출한다는 잠정 계약이다. 다중 스레드 동시 호출은 지원하지 않는다.
    """

    def __init__(self, paths: Sequence[str | os.PathLike], *, source_id: str,
                 issuer: FrameIdIssuer, capture_generation: int = 0,
                 clock_ns: Callable[[], int] = time.monotonic_ns) -> None:
        self._paths = [os.fspath(p) for p in paths]
        self._source_id = source_id
        self._issuer = issuer
        self._capture_generation = capture_generation
        self._clock_ns = clock_ns
        self._next_index = 0
        self._current: FramePacket | None = None
        self._closed = False

    def read(self) -> FramePacket | InputStatusEvent:
        """목록의 다음 항목을 읽어 `FramePacket`을, 읽지 못하면 `InputStatusEvent`를 돌려준다."""
        if self._closed:
            return self._event(InputStatus.SOURCE_UNAVAILABLE, "closed", None)
        if not self._paths:
            return self._event(InputStatus.SOURCE_UNAVAILABLE, "empty list", None)
        if self._next_index >= len(self._paths):
            return self._event(InputStatus.END_OF_STREAM, "end of list", None)

        index = self._next_index
        path = self._paths[index]
        self._next_index += 1

        if not os.path.isfile(path):
            return self._event(InputStatus.SOURCE_UNAVAILABLE,
                               "not a regular file or missing", path)
        if not os.access(path, os.R_OK):
            return self._event(InputStatus.SOURCE_UNAVAILABLE, "file not readable", path)

        try:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
        except cv2.error as exc:
            return self._event(InputStatus.DECODE_FAILED, f"opencv decode error: {exc}", path)
        if image is None:
            return self._event(InputStatus.DECODE_FAILED, "imread returned None", path)
        if image.dtype != "uint8" or image.ndim != 3 or image.shape[2] != 3:
            return self._event(InputStatus.DECODE_FAILED,
                               f"unexpected array dtype={image.dtype} shape={image.shape}", path)

        # 다른 버퍼를 빌려 쓰는 배열이면 복사해 프레임이 버퍼를 단독으로 갖게 한다.
        if not image.flags.owndata:
            image = image.copy()
        image.setflags(write=False)

        seq = self._issuer.next_seq()
        received_ns = self._clock_ns()
        frame = FramePacket(
            image=image,
            run_id=self._issuer.run_id,
            source_id=self._source_id,
            image_seq=seq,
            timestamp_ms=received_ns // 1_000_000,
            timestamp_base=TimestampBase.PC_RECEIVE,
            received_monotonic_ns=received_ns,
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            crop=None,
            calibration_id=None,
            capture_generation=self._capture_generation,
            optical_frame=None,
            ros_stamp_ns=None,
            source_ref=path,
            source_index=index,
        )
        self._current = frame
        return frame

    def current(self) -> FramePacket | None:
        """마지막 성공 프레임(같은 객체). 없으면 None."""
        return self._current

    def close(self) -> None:
        """소스를 닫는다. 이후 `read()`는 `SOURCE_UNAVAILABLE`("closed")."""
        self._closed = True
        self._current = None

    def _event(self, status: InputStatus, detail: str,
               source_ref: str | None) -> InputStatusEvent:
        self._current = None
        return InputStatusEvent(
            run_id=self._issuer.run_id,
            source_id=self._source_id,
            status=status,
            event_monotonic_ns=self._clock_ns(),
            detail=detail,
            source_ref=source_ref,
        )
