"""공통 입력 자료형 — 공통 자료형 관리 초안 기준.

근거: 공통 자료형 관리 초안 §3.1(`TimestampBase`), §3.2(`FramePacket`), §3.6
(`InputStatusEvent`, 가칭), 사용자 확정(2026-11-11: 공통 자료형은 초안 기준으로 진행).

- 현재 쓰는 최소 자료형만 둔다. A 파이프라인 작업 1(계획 2026-11-15 r3)에서 초안 §3.1의
  `PixelBasis`·`Contour`, §3.3 `PlanarObservation`, §3.5 `CandidateInvalidationEvent`를 추가했다
  (계획 제안 단계, 영향 담당 확인 전). `WorkPlanarResult`(B 소유)·`PixelToSourceTransform`
  (전달 방식 미결)은 아직 두지 않는다.
- 담당별 확장(예: A 입력의 `source_ref`·`source_index`)은 각 모듈에서 이 클래스를 상속해 둔다.
- 표준 라이브러리와 NumPy만 쓴다. Qt·ROS·OpenCV를 import하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

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


class PixelBasis(str, Enum):
    """윤곽 점 좌표의 기준(초안 §3.1, B §6.1 `pixel_basis`). 왜곡 보정 전(`raw`)·후(`undistorted`)."""

    RAW = "raw"
    UNDISTORTED = "undistorted"


# 윤곽 점 배열(초안 §3.1): numpy shape (N, 2), float32. ROI·표시 축소를 되돌린 원본 해상도·크롭
# 기준 픽셀 좌표이며, 왜곡 보정 여부는 함께 담는 `PixelBasis`로 구분한다.
Contour: TypeAlias = np.ndarray

# `PlanarObservation.method` 허용 값(초안 §3.3)
_OBSERVATION_METHODS = ("opencv", "ai")

# AI 관측에서만 값을 갖는 조건부 필드(초안 §3.3 「조건부(AI)」)
_AI_ONLY_FIELDS = ("class_name", "confidence", "model_id")


def _is_int(value) -> bool:
    """bool을 뺀 정수인지 확인한다."""
    return isinstance(value, int) and not isinstance(value, bool)


def _owned_contour(points, name: str) -> Contour | None:
    """윤곽을 소유 복사본(float32 (N, 2), 읽기 전용)으로 만든다. None은 그대로, 형태가 다르면 ValueError."""
    if points is None:
        return None

    # 외부 배열과 메모리를 나누지 않도록 항상 새로 복사한다(초안 §4 불변조건 8)
    owned = np.array(points, dtype=np.float32, copy=True, order="C")

    if owned.ndim != 2 or owned.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2): {owned.shape}")

    owned.setflags(write=False)

    return owned


@dataclass(frozen=True)
class PlanarObservation:
    """A 평면 관측 하나(초안 §3.3, A 상세설계 §5.2). 모든 필드에 기본값이 없다.

    - 유효 조건: `valid=True`이면 `observed_face`·`calc_contour`·`selected_index`가 모두 None이
      아니어야 한다. 하나라도 없으면 ValueError. 관측면이 미확정이면 `valid=False`·
      `calc_contour=None`으로 둔다. `observed_face` 기본값을 `top_face`로 두지 않는다.
    - 관측면이 None이면 `calc_contour`도 None이어야 한다(무효 관측 포함, 초안 §3.3 유효 조건·§7 예시 1-b).
    - 조건부 필드는 그 조건 밖에서 None이다: 유효 관측의 `reason`은 None, `method="opencv"`이면
      `class_name`·`confidence`·`model_id`는 None. 어기면 ValueError.
    - 무효 관측은 값이 없는 필드를 None으로 둔다. 단 `reason`은 무효 시 필수이며 비어 있지 않은
      문자열이어야 한다(초안 §3.3 reason 행 「조건부(무효 시)」). 사유 이름 목록은 미결이므로
      값은 검사하지 않는다. 사유를 정하지 못한 생산자는 관측 생성을 보류한다.
    - `calc_contour`·`display_mask_contour`는 생성 시 소유 복사본(float32 (N, 2), 읽기 전용)으로
      저장한다. 외부 배열을 나중에 고쳐도 관측은 변하지 않는다. 전체 실루엣(`display_mask_contour`)을
      계산용 윤곽으로 승격하지 않는다.
    - 생성자는 클래스·보정·시간 값을 새로 만들어 넣지 않는다. 처리 예외·중단·세대 변경 폐기에는
      관측을 만들지 않는다(초안 §3.3 끝).
    - `selected_index`는 A 관측 식별 필드이며 B의 최종 후보 선택을 대신하지 않는다.
    """

    run_id: str
    source_id: str
    image_seq: int
    capture_generation: int
    generation: int
    job_seq: int
    detector_config_id: str
    pixel_basis: PixelBasis
    calibration_id: str | None
    observed_face: str | None
    calc_contour: Contour | None
    display_mask_contour: Contour | None
    candidate_count: int
    selected_index: int | None
    valid: bool
    reason: str | None
    process_ms: float
    method: str
    class_name: str | None
    confidence: float | None
    model_id: str | None

    def __post_init__(self) -> None:
        """열거 값 정규화·유효 조건 검사·윤곽 소유 복사를 한다. 어긋나면 ValueError."""
        put = lambda k, v: object.__setattr__(self, k, v)  # noqa: E731

        # 열거·플래그 값 검사
        put("pixel_basis", PixelBasis(self.pixel_basis))

        if self.method not in _OBSERVATION_METHODS:
            raise ValueError(f"method must be 'opencv' or 'ai': {self.method!r}")

        if not isinstance(self.valid, bool):
            raise ValueError(f"valid must be bool: {self.valid!r}")

        # 윤곽 소유 복사(형태 검사 포함)
        put("calc_contour", _owned_contour(self.calc_contour, "calc_contour"))
        put("display_mask_contour", _owned_contour(self.display_mask_contour, "display_mask_contour"))

        # 관측면이 없으면 계산용 윤곽도 없다(전체 실루엣 승격 방지)
        if self.observed_face is None and self.calc_contour is not None:
            raise ValueError("calc_contour requires observed_face")

        # 유효 조건(초안 §3.3)
        if self.valid:
            missing = [k for k in ("observed_face", "calc_contour", "selected_index")
                       if getattr(self, k) is None]

            if missing:
                raise ValueError(f"valid observation requires {missing}")

            if self.reason is not None:
                raise ValueError("valid observation must not carry reason")

        # 무효 관측의 사유는 필수(이름 목록은 미결이라 형식만 검사)
        elif not isinstance(self.reason, str) or not self.reason:
            raise ValueError(f"invalid observation requires non-empty str reason: {self.reason!r}")

        # AI 전용 필드는 opencv 관측에서 None
        if self.method == "opencv":
            present = [k for k in _AI_ONLY_FIELDS if getattr(self, k) is not None]

            if present:
                raise ValueError(f"opencv observation must not carry {present}")


class InvalidationKind(str, Enum):
    """후보 무효화 사건의 종류(초안 §3.5, A 상세설계 §5.4). 이 5개 외의 값을 임의로 추가하지 않는다."""

    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    MODEL_CONFIG_INVALID = "MODEL_CONFIG_INVALID"
    INFERENCE_ERROR = "INFERENCE_ERROR"
    JOB_ABORTED = "JOB_ABORTED"
    GENERATION_CHANGED = "GENERATION_CHANGED"


# 프레임 없는 사건: job_seq·image_seq가 반드시 None(초안 §3.5·§4-6, A 상세설계 §5.4 표)
_FRAMELESS_KINDS = (InvalidationKind.GENERATION_CHANGED, InvalidationKind.MODEL_LOAD_FAILED)

# 작업에서 생긴 사건: job_seq·image_seq가 반드시 값(A 상세설계 §5.4 표 「해당 작업의 값」)
_JOB_KINDS = (InvalidationKind.INFERENCE_ERROR, InvalidationKind.JOB_ABORTED)


@dataclass(frozen=True)
class CandidateInvalidationEvent:
    """이동 후보 무효화 사건(초안 §3.5, 가칭). 모든 필드에 기본값이 없다.

    - `kind`는 `InvalidationKind` 5개 값만 허용한다(문자열이면 열거로 정규화, 그 밖은 ValueError).
    - `GENERATION_CHANGED`·`MODEL_LOAD_FAILED`는 프레임 없는 사건이라 `job_seq`·`image_seq`가 None이다.
      가상 값을 넣으면 ValueError. `GENERATION_CHANGED`의 `generation`은 새 현재 세대다.
    - `INFERENCE_ERROR`·`JOB_ABORTED`는 해당 작업의 `job_seq`·`image_seq`를 함께 가진다.
    - `MODEL_CONFIG_INVALID`는 설정 적용 시(둘 다 None) 또는 작업 중(둘 다 값) 중 하나다.
    - `event_monotonic_ns`는 PC 단조 시각이다. 이 생성자는 시계를 읽지 않는다.
    - 필수 문자열(`run_id`·`source_id`·`cause`·`detector_config_id`)은 비어 있지 않아야 하고,
      `event_monotonic_ns`·`generation`·값이 있는 `job_seq`·`image_seq`는 bool 아닌 정수여야 한다.
    """

    run_id: str
    source_id: str
    kind: InvalidationKind
    cause: str
    event_monotonic_ns: int
    generation: int
    detector_config_id: str
    job_seq: int | None
    image_seq: int | None

    def __post_init__(self) -> None:
        """필드 형식을 검사하고 kind를 정규화한 뒤 kind별 프레임 식별자 규칙을 검사한다. 어긋나면 ValueError."""
        # 필수 필드 형식
        for name in ("run_id", "source_id", "cause", "detector_config_id"):
            value = getattr(self, name)

            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string: {value!r}")

        for name in ("event_monotonic_ns", "generation"):
            if not _is_int(getattr(self, name)):
                raise ValueError(f"{name} must be an integer: {getattr(self, name)!r}")

        for name in ("job_seq", "image_seq"):
            value = getattr(self, name)

            if value is not None and not _is_int(value):
                raise ValueError(f"{name} must be an integer or None: {value!r}")

        kind = InvalidationKind(self.kind)
        object.__setattr__(self, "kind", kind)

        has_job = self.job_seq is not None
        has_frame = self.image_seq is not None

        if kind in _FRAMELESS_KINDS and (has_job or has_frame):
            raise ValueError(f"{kind.value} must not carry job_seq/image_seq")

        if kind in _JOB_KINDS and not (has_job and has_frame):
            raise ValueError(f"{kind.value} requires job_seq and image_seq")

        if has_job != has_frame:
            raise ValueError("job_seq and image_seq must be both set or both None")
