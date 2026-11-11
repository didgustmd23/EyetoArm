"""공통 입력 자료형 — 공통 자료형 관리 초안 기준.

근거: 공통 자료형 관리 초안 §3.1(`TimestampBase`), §3.2(`FramePacket`), §3.6
(`InputStatusEvent`, 가칭), 사용자 확정(2026-11-11: 공통 자료형은 초안 기준으로 진행).

- 현재 쓰는 최소 자료형만 둔다. `PlanarObservation`·`WorkPlanarResult`·무효화 사건 등
  초안의 나머지 형식은 실제로 쓰는 시점에 추가한다.
- 담당별 확장(예: A 입력의 `source_ref`·`source_index`)은 각 모듈에서 이 클래스를 상속해 둔다.
- 표준 라이브러리와 NumPy만 쓴다. Qt·ROS·OpenCV를 import하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class TimestampBase(str, Enum):
    """`timestamp_ms`의 시간 기준(초안 §3.1, TCP 명세서 §4.3). 기준이 다른 시각끼리 빼지 않는다."""

    PC_RECEIVE = "PC_RECEIVE"
    PHONE_CAPTURE = "PHONE_CAPTURE"
    SIM_TIME = "SIM_TIME"
    VIDEO_POS = "VIDEO_POS"


class InputStatus(str, Enum):
    """프레임이 없는 입력 사건의 상태(초안 §3.6). 검출 판정 사유와 섞지 않는다."""

    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    FRAME_TIMEOUT = "FRAME_TIMEOUT"
    DECODE_FAILED = "DECODE_FAILED"
    END_OF_STREAM = "END_OF_STREAM"


@dataclass(frozen=True)
class FramePacket:
    """성공적으로 받은 프레임 하나(초안 §3.2, 14개 필드).

    - `image`는 BGR `uint8`, shape (H, W, 3) 입력 계약을 따른다. 이 자료형은 값을 검사하지 않는다.
    - 모든 필드에 기본값이 없다. 호출자가 값이나 None을 명시한다.
    - 이 자료형은 시계를 읽거나 순번을 발급하지 않는다. 그것은 입력 소스의 일이다.
    - `frozen=True`는 필드 재할당만 막는다. `image` 배열의 내용이나 다른 별칭을 통한 변경은
      막지 않는다. 버퍼 소유권 이전·독립 복사·쓰기 제한은 이 생성자가 하지 않으며, 프레임을
      만드는 입력 소스가 책임진다(초안 §4 불변조건 8).
    """

    image: np.ndarray
    run_id: str
    source_id: str
    image_seq: int
    timestamp_ms: int
    timestamp_base: TimestampBase
    received_monotonic_ns: int
    width: int
    height: int
    crop: tuple[int, int, int, int] | None
    calibration_id: str | None
    capture_generation: int
    optical_frame: str | None
    ros_stamp_ns: int | None


@dataclass(frozen=True)
class InputStatusEvent:
    """프레임을 받지 못했거나 받기를 끝낸 사건(초안 §3.6, 가칭).

    프레임이 없으므로 `image_seq`·`image`·`timestamp_ms`를 두지 않는다. 시각은 PC 단조
    시각(`event_monotonic_ns`)이다. 모든 필드에 기본값이 없다.
    """

    run_id: str
    source_id: str
    status: InputStatus
    event_monotonic_ns: int
    detail: str | None
