"""A 입력 자료형(공통 자료형 확장)과 프레임 식별자 발급기.

- 공통 자료형은 `eyetoarm.common.types`(공통 자료형 관리 초안 기준, 사용자 확정 2026-11-11)에 있다.
  이 모듈은 같은 enum 클래스를 그대로 다시 내보내고, `FramePacket`·`InputStatusEvent`는 공통
  클래스를 상속해 A 내부 재현용 필드만 덧붙인다. 기존 import 경로·생성 인자·동작은 같다.
- 이름 구분: 이 모듈의 `FramePacket`은 A 확장(`common.types.FramePacket` + `source_ref`·
  `source_index`), `InputStatusEvent`는 A 확장(공통 사건 + `source_ref`)이다. 코드 안에서는
  공통 클래스를 `CommonFramePacket`·`CommonInputStatusEvent`라는 별칭으로 부른다.
- `source_ref`·`source_index`는 공통 자료형 초안에 없는 A 내부 필드다.

표준 라이브러리와 NumPy만 쓴다(공통 모듈 포함). Qt·ROS·OpenCV를 import하지 않는다.
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
    """A 확장 프레임: 공통 `FramePacket` 14개 필드(A 상세설계 §5.1) + A 내부 재현용 2개 필드.

    - 공통 필드의 계약(BGR `uint8` (H, W, 3), 기본값 없음, 시계·순번을 읽지 않음, `frozen`은
      필드 재할당만 막고 배열 내용 변경은 막지 않음)은 `common.types.FramePacket`과 같다.
      버퍼 소유권 이전·독립 복사·쓰기 제한은 프레임을 만드는 저장 소스가 책임진다.
    - 정지 화면 재처리는 같은 FramePacket 객체를 그대로 다시 쓴다. 새 `image_seq`나
      새 시각을 만들지 않으며, 처리 세대·`job_seq`로 재처리 결과를 구분한다(§5.4).
    - `source_ref`(원본 위치)·`source_index`(소스 안 순번)는 A 내부 확장 필드다.
    """

    source_ref: str
    source_index: int


@dataclass(frozen=True)
class InputStatusEvent(CommonInputStatusEvent):
    """A 확장 입력 사건: 공통 `InputStatusEvent` 5개 필드(A 상세설계 §5.4) + `source_ref`.

    프레임이 없으므로 `image_seq`·`image`·`timestamp_ms`를 두지 않는다. 시각은 PC 단조
    시각(`event_monotonic_ns`)이다. `source_ref`는 A 내부 확장 필드다.
    """

    source_ref: str | None


class FrameIdIssuer:
    """한 실행(`run_id`)의 `image_seq` 발급기(A 상세설계 §5.1 「image_seq 수명」).

    - 첫 호출은 0을 돌려주고 이후 1씩 늘어난다. 되돌리거나 초기화하지 않는다.
    - 실행 하나에 인스턴스 하나를 두고 모든 소스가 함께 쓴다. 소스 변경·실패·재처리 때
      새 발급기를 만들지 않는다.
    - 성공한 수신에서만 `next_seq()`를 부른다. 실패 사건과 현재 프레임 재사용은 부르지 않는다.
    - 입력 담당의 단일 흐름에서 직렬로 호출한다는 잠정 계약이다. 다중 스레드 원자성은 보장하지 않는다.
    """

    def __init__(self, run_id: str) -> None:
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
