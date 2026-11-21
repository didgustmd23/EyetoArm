"""A 입력 자료형(공통 자료형 확장)과 프레임 식별자 발급기.

공통 enum은 그대로 다시 내보내고, `FramePacket`·`InputStatusEvent`는 공통 클래스를 상속해 A 내부
필드(`source_ref`·`source_index`)를 덧붙인다. 표준 라이브러리와 NumPy만 쓴다.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from eyetoarm.common.types import FramePacket as CommonFramePacket
from eyetoarm.common.types import InputStatus, TimestampBase
from eyetoarm.common.types import InputStatusEvent as CommonInputStatusEvent

__all__ = [
    "TimestampBase",
    "InputStatus",
    "FramePacket",
    "InputStatusEvent",
    "FrameIdIssuer",
    "new_run_id",
]


@dataclass(frozen=True)
class FramePacket(CommonFramePacket):
    """A 확장 프레임: 공통 `FramePacket` 필드 + `source_ref`(원본 위치)·`source_index`(소스 안 순번).

    `frozen`은 필드 재할당만 막는다. 정지 화면 재처리는 같은 객체를 다시 쓰고 새 `image_seq`를 만들지 않는다.
    """

    source_ref: str
    source_index: int


@dataclass(frozen=True)
class InputStatusEvent(CommonInputStatusEvent):
    """A 확장 입력 사건: 공통 `InputStatusEvent` 필드 + `source_ref`. 시각은 PC 단조 시각이다."""

    source_ref: str | None


class FrameIdIssuer:
    """한 실행(`run_id`)의 `image_seq` 발급기. 0부터 1씩 늘고 되돌리지 않는다.

    실행 하나에 하나를 두고 모든 소스가 함께 쓴다. 성공한 수신에서만 `next_seq()`를 부른다.
    단일 흐름에서 직렬 호출하는 계약이며 다중 스레드 원자성은 보장하지 않는다.
    """

    def __init__(self, run_id: str) -> None:
        """`run_id` 실행의 발급기를 만든다. 첫 `next_seq()`는 0이다."""
        self._run_id = run_id
        self._next = 0

    @property
    def run_id(self) -> str:
        """생성할 때 받은 실행 ID."""
        return self._run_id

    def next_seq(self) -> int:
        """다음 `image_seq`를 돌려주고 1 늘린다(첫 호출은 0)."""
        seq = self._next
        self._next += 1
        return seq


def new_run_id() -> str:
    """새 실행 ID(`uuid.uuid4().hex`). 호출할 때만 발급한다."""
    return uuid.uuid4().hex
