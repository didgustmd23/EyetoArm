"""A 비전 파이프라인 내부 타입: 설정 스냅샷·작업·완료 상태·처리 기록·계수.

A 내부 타입이며 D API·B 계약·공통형이 아니다. 명령 ID·가짜 식별자를 만들지 않는다.
Qt·ROS·TCP를 import하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from eyetoarm.common.types import (
    CandidateInvalidationEvent,
    FramePacket,
    InputStatus,
    PlanarObservation,
    TimestampBase,
)
from eyetoarm.vision.ai_config import AiSegConfig
from eyetoarm.vision.ai_result import AiSegResult
from eyetoarm.vision.planar import PlanarConfig, PlanarResult

# 검출 설정·표시 결과로 허용하는 형
DetectorConfig = PlanarConfig | AiSegConfig
DisplayResult = PlanarResult | AiSegResult


def _require_text(value, name: str) -> None:
    """비어 있지 않은 문자열인지 확인한다. 아니면 ValueError."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string: {value!r}")


@dataclass(frozen=True)
class SourceProfile:
    """소스·보정 프로파일의 적용 조건. B 보정 수치는 담지 않는다."""

    source_id: str
    width: int
    height: int
    crop: tuple[int, int, int, int] | None
    calibration_id: str | None

    def applies_to(self, frame: FramePacket) -> bool:
        """이 프로파일이 그 프레임에 유효한지(source_id·해상도·크롭·calibration_id 일치, 보정 있음)."""
        # 보정 누락은 B 유효 입력으로 허용하지 않는다
        if self.calibration_id is None:
            return False

        return (frame.source_id == self.source_id
                and frame.width == self.width
                and frame.height == self.height
                and frame.crop == self.crop
                and frame.calibration_id == self.calibration_id)


@dataclass(frozen=True)
class PipelineSettings:
    """처리 설정 스냅샷(frozen). `mode`는 A 내부 처리 모드(TCP 명령 모드 아님).

    `profile`이 None이면 표시용 처리만 가능하다(B 유효 입력·재처리 불가).
    """

    source_id: str
    mode: str
    detector_config: DetectorConfig
    profile: SourceProfile | None

    def __post_init__(self) -> None:
        """필드 형식을 검사한다. 어긋나면 ValueError."""
        _require_text(self.source_id, "source_id")
        _require_text(self.mode, "mode")

        if not isinstance(self.detector_config, (PlanarConfig, AiSegConfig)):
            raise ValueError(f"detector_config must be PlanarConfig or AiSegConfig: "
                             f"{type(self.detector_config).__name__}")

        if self.profile is not None and not isinstance(self.profile, SourceProfile):
            raise ValueError(f"profile must be SourceProfile or None: {type(self.profile).__name__}")

    @property
    def detector_config_id(self) -> str:
        """검출 설정 식별자(`PlanarConfig`·`AiSegConfig`의 `config_sha256`)."""
        return self.detector_config.config_sha256


@dataclass(frozen=True)
class JobContext:
    """시작한 처리 작업 하나의 스냅샷(frozen). 설정이 바뀌어도 시작 시점 `settings`로 계산·기록한다.

    `frame`은 파이프라인 소유 복사본(재처리는 같은 객체 재사용), `generation`은 시작 시점 세대다.
    """

    frame: FramePacket
    generation: int
    job_seq: int
    settings: PipelineSettings
    started_monotonic_ns: int
    reprocess: bool

    @property
    def detector_config_id(self) -> str:
        """시작 설정의 검출 설정 식별자(`settings.detector_config.config_sha256`)."""
        return self.settings.detector_config.config_sha256


class JobStatus(str, Enum):
    """시작한 작업의 종료 구분. 대기열 폐기는 작업을 시작하지 않았으므로 이 상태로 표현하지 않는다."""

    COMPLETED = "COMPLETED"
    DISCARDED_GENERATION = "DISCARDED_GENERATION"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


class QueueDropReason(str, Enum):
    """대기 프레임 폐기 사유(사유별 집계 키)."""

    REPLACED_BY_NEWER = "replaced_by_newer"
    STALE_CAPTURE = "stale_capture_generation"
    GENERATION_CHANGED = "generation_changed"
    INPUT_STATUS = "input_status"
    CLOSED = "closed"


@dataclass(frozen=True)
class JobOutcome:
    """작업 하나의 terminal 응답(frozen). 관측·표시 결과는 최초 완료 응답에서만 담는다.

    duplicate 응답은 최초 상태·사건을 참조할 뿐이라 소비자가 다시 발행·집계하지 않는다.
    `applied`는 완료 관측을 현재 슬롯에 반영했는지이며 B 유효 입력·로봇 실행 가능을 뜻하지 않는다.
    """

    context: JobContext
    status: JobStatus
    observation: PlanarObservation | None
    display_result: DisplayResult | None
    error: str | None
    invalidation: CandidateInvalidationEvent | None
    applied: bool = False
    duplicate: bool = False


@dataclass(frozen=True)
class InvalidationRecord:
    """코어가 기록한 무효화 사건 하나와 그 평가 결과(frozen). 사건마다 정확히 한 번 만든다.

    `applied`는 기록 당시 현재 세대 차단에 반영했는지이며 `JobOutcome.applied`와 다르다.
    """

    event: CandidateInvalidationEvent
    applied: bool


@dataclass(frozen=True)
class ProcessRecord:
    """처리 작업 기록 한 줄. 원본 시각과 시작 설정 ID를 그대로 남긴다.

    `command_id`는 A가 만들지 않아 항상 None, `eval_run_id`는 평가 도구가 발급한 값 또는 None이다.
    `axis_valid_count`는 B로 간 수가 아니라 픽셀 후보 수다. omit_if_none 필드는 None이면 키를 쓰지 않는다.
    """

    run_id: str
    source_id: str
    image_seq: int
    capture_generation: int
    generation: int
    job_seq: int
    mode: str
    timestamp_ms: int
    timestamp_base: TimestampBase
    calibration_id: str | None
    detector_config_id: str
    reprocess: bool
    status: JobStatus
    valid: bool | None
    reason: str | None
    reason_owner: str | None
    candidate_count: int | None
    process_ms: float | None
    command_id: None
    eval_run_id: str | None = None
    method: str | None = None
    model_id: str | None = None
    infer_ms: float | None = None
    ai_instances: tuple | None = None
    axis_valid_count: int | None = field(default=None, metadata={"omit_if_none": True})
    list_candidate_count: int | None = field(default=None, metadata={"omit_if_none": True})
    list_indices: tuple[int, ...] | None = field(default=None, metadata={"omit_if_none": True})
    axis_duplicates_merged: int | None = field(default=None, metadata={"omit_if_none": True})   # #292 합친 중복 수(>0일 때만)

    def __post_init__(self) -> None:
        """명령 ID를 만들지 않는다는 계약과 eval_run_id 형식(None 또는 비어 있지 않은 문자열)을 검사한다."""
        if self.command_id is not None:
            raise ValueError("command_id is not generated by A and must be None")

        if self.eval_run_id is not None:
            _require_text(self.eval_run_id, "eval_run_id")


@dataclass(frozen=True)
class QueueDropRecord:
    """작업을 시작하지 않고 대기 중 폐기한 프레임의 기록. `dropped_monotonic_ns`는 PC 단조 시각."""

    run_id: str
    source_id: str
    image_seq: int
    capture_generation: int
    timestamp_ms: int
    timestamp_base: TimestampBase
    received_monotonic_ns: int
    ros_stamp_ns: int | None
    reason: QueueDropReason
    dropped_monotonic_ns: int


@dataclass(frozen=True)
class InputStateRecord:
    """입력 상태 기록: `InputStatusEvent` 5개 필드 + 상태 지속 시간(`duration_ns`, PC 단조, 미확정 None)."""

    run_id: str
    source_id: str
    status: InputStatus
    event_monotonic_ns: int
    detail: str | None
    duration_ns: int | None


@dataclass
class PipelineCounters:
    """처리 건수(파이프라인 안에서만 바꾼다). 사유·종류별 계수 키는 열거의 `.value` 문자열이다.

    무효 완료는 `failed`로 세지 않는다. 시작한 작업은 네 종료 계수 중 정확히 하나로 끝난다.
    """

    received: int = 0
    first_started: int = 0
    reprocess_started: int = 0
    completed: int = 0
    valid_completed: int = 0
    invalid_completed_by_reason: dict[str, int] = field(default_factory=dict)
    queue_dropped: dict[str, int] = field(default_factory=dict)
    generation_discarded: int = 0
    failed: int = 0
    aborted: int = 0
    invalidation_applied_by_kind: dict[str, int] = field(default_factory=dict)
    invalidation_ignored_by_kind: dict[str, int] = field(default_factory=dict)
