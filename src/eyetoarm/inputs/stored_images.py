"""저장 이미지·명시 순서 목록 입력 소스(A 내부 잠정안, 공동 API 아님).

자료형은 `inputs/base.py`(공통 자료형의 A 확장)를 쓰고, 실패한 항목도 소비하고 다음 항목으로 진행한다.
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
    """호출자가 정한 순서의 이미지 파일 목록을 하나씩 읽는 입력 소스(단일 흐름 전용).

    `read()`는 실패도 항목 하나를 소비한다. 접근 불가는 `SOURCE_UNAVAILABLE`, 디코딩 실패는 `DECODE_FAILED`,
    끝은 매번 `END_OF_STREAM`. `timestamp_ms`는 PC 수신 시각이라 촬영 시각으로 쓰지 않는다.
    """

    def __init__(self, paths: Sequence[str | os.PathLike], *, source_id: str,
                 issuer: FrameIdIssuer, capture_generation: int = 0,
                 clock_ns: Callable[[], int] = time.monotonic_ns,
                 generation_provider: Callable[[], int] | None = None) -> None:
        """경로 목록을 복사해 두고 발급기·세대(고정 값 또는 공급자)·시계를 주입받는다. 파일은 `read()` 때 연다."""
        self._paths = [os.fspath(p) for p in paths]
        self._source_id = source_id
        self._issuer = issuer
        self._capture_generation = capture_generation
        self._generation_provider = generation_provider
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

        # 공급자가 있으면 새 프레임마다 한 번 읽고, 없으면 고정 세대를 쓴다
        if self._generation_provider is None:
            generation = self._capture_generation
        else:
            generation = self._generation_provider()

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
            capture_generation=generation,
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
        """현재 프레임을 비우고, 시계를 한 번 읽어 입력 사건을 만든다."""
        self._current = None
        return InputStatusEvent(
            run_id=self._issuer.run_id,
            source_id=self._source_id,
            status=status,
            event_monotonic_ns=self._clock_ns(),
            detail=detail,
            source_ref=source_ref,
        )
