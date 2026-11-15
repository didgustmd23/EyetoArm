"""A 비전 파이프라인 내부 타입 — 설정 스냅샷·작업·완료 상태·처리 기록·계수.

근거: 계획 `docs/superpowers/plans/2026-11-15-a-vision-pipeline.md` r3 「인터페이스 제안」·작업 1,
A 상세설계 §5.1(식별자)·§5.4(세대·재처리 조건)·§9(처리 기록·처리 건수).

- 이름·필드 묶음은 계획의 **제안**이며 A 내부 타입이다. D API·B 계약·공통형이 아니다.
- 검출 설정은 기존 `PlanarConfig` 또는 AI 분할 설정 `AiSegConfig`(둘 다 frozen, `config_sha256`)다(YOLO 연결
  계획 r2 G1). 같은 의미의 설정·해시를 새로 만들지 않고, AI 설정을 외부 클로저에 숨겨 같은 설정 ID로 여러 모델을
  돌리지 않는다. 표시 결과도 `PlanarResult` 또는 `AiSegResult`다(G3·G4).
- `JobOutcome`·`PipelineCounters`는 작업 4에서 채운다. `ProcessRecord`·`InputStateRecord`는 필드
  정의만 두며 채우는 동작은 작업 5에서 붙인다.
- 명령 ID·가짜 식별자를 만들지 않는다. Qt·ROS·TCP를 import하지 않는다.
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

# 검출 설정·표시 결과로 허용하는 형(YOLO 연결 계획 r2 G1–G4)
DetectorConfig = PlanarConfig | AiSegConfig
DisplayResult = PlanarResult | AiSegResult


def _require_text(value, name: str) -> None:
    """비어 있지 않은 문자열인지 확인한다. 아니면 ValueError."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string: {value!r}")


@dataclass(frozen=True)
class SourceProfile:
    """소스·보정 프로파일의 적용 조건(계획 「인터페이스 제안」, A 상세설계 §5.4 재처리 조건 3).

    - 적용 조건만 보관한다. B 보정 수치를 계산하거나 담지 않는다.
    - `crop`·`calibration_id`는 `FramePacket`의 같은 이름 필드와 같은 형이다.
      B 프로파일 필드 이름은 초안 §8에서 미결이다.
    """

    source_id: str
    width: int
    height: int
    crop: tuple[int, int, int, int] | None
    calibration_id: str | None

    def applies_to(self, frame: FramePacket) -> bool:
        """이 프로파일이 그 프레임에 유효한지(source_id·해상도·크롭·calibration_id 일치, 보정 있음)."""
        # 보정 누락은 B 유효 입력으로 허용하지 않는다(초안 §4-7)
        if self.calibration_id is None:
            return False

        return (frame.source_id == self.source_id
                and frame.width == self.width
                and frame.height == self.height
                and frame.crop == self.crop
                and frame.calibration_id == self.calibration_id)


@dataclass(frozen=True)
class PipelineSettings:
    """처리 설정 스냅샷(계획 「인터페이스 제안」). frozen.

    - `mode`는 A 내부 처리 모드다. TCP 명령 모드가 아니며 허용 값은 문서 미정이라
      비어 있지 않은 문자열이면 받는다.
    - `detector_config`는 `PlanarConfig`(OpenCV) 또는 `AiSegConfig`(AI). 생성자가 이미 값을 검증했다.
    - `profile`이 None이면 표시용 처리만 가능하다(B 유효 입력·재처리 불가).
    - 오버레이 색 등 표시 전용 설정은 처리 설정이 아니므로 두지 않는다(A 상세설계 §5.4).
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
    """시작한 처리 작업 하나의 스냅샷(계획 「인터페이스 제안」). frozen.

    - `frame`은 파이프라인이 소유한 복사 프레임이다. 재처리는 같은 객체를 다시 쓴다.
    - `generation`은 시작 시점의 현재 세대, `job_seq`는 실행 내 시작 순번이다.
    - `settings`는 시작 시점 설정이며 이후 설정이 바뀌어도 이 작업은 이 값으로 계산·기록한다.
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
    """대기 프레임 폐기 사유(계획 작업 3·5의 사유별 집계). 이름은 A 내부 임시 결정이다."""

    REPLACED_BY_NEWER = "replaced_by_newer"
    STALE_CAPTURE = "stale_capture_generation"
    GENERATION_CHANGED = "generation_changed"
    INPUT_STATUS = "input_status"
    CLOSED = "closed"


@dataclass(frozen=True)
class JobOutcome:
    """작업 하나의 terminal 응답(계획 「인터페이스 제안」, 작업 4 보완 103 §4). frozen.

    - observation은 최초 완료 응답에서만 제공할 수 있다. 실패·폐기·중단 및 duplicate 응답은 None이다.
      `display_result`도 같다.
    - duplicate 응답은 최초 status·error·사건을 유지하되 데이터 재전달·상태 변경을 하지 않는다.
      `invalidation`을 담아도 새로 발행한 사건이 아니라 최초 사건의 참조 정보다. 소비자는 duplicate
      응답을 관측·사건·처리 기록으로 다시 발행하거나 집계하지 않는다.
    - applied는 해당 호출이 완료 관측을 현재 슬롯에 반영했는지이며, 중복 응답은 False다.
      무효 관측으로 교체해도 True일 수 있으므로 B 유효 입력·로봇 실행 가능을 뜻하지 않는다.
      오류 차단선 이전에 시작한 늦은 완료는 관측이 유효해도 False다. 무효화 사건의 applied 계수와는
      다른 개념이다.
    - `error`는 실패·중단 원인 문자열(`fail()`의 cause, 계약 위반은 `adapter_contract_violation:...`).
      완료·세대 폐기는 None.
    - `invalidation`은 해당 실패·중단의 사건 또는 None이며 다시 만들어 중복 발행하지 않는다.
    - `applied`·`duplicate`는 기본값이 있는 끝 필드라 기존 6인자 생성자를 그대로 쓸 수 있다.
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
class ProcessRecord:
    """처리 작업 기록 한 줄(A 상세설계 §9 「처리 기록」). 작업 5의 `pipeline_records.process_record`가 채운다.

    - 원본 시각(`timestamp_ms`·`timestamp_base`)과 시작 설정 ID를 그대로 남긴다.
    - `command_id`는 A가 만들지 않는다. 항상 None이며 다른 값은 ValueError.
    - `eval_run_id`는 평가 실행 중일 때만 평가 도구가 발급한 값이고 아니면 None이다(A §9 「처리 작업마다
      `eval_run_id`(평가 중일 때)」). A가 만들지 않는다. 기본값이 있는 끝 필드라 기존 생성자와 호환된다.
    - AI 기록(YOLO 연결 계획 r2 §4.6, A §9): `method`("opencv"·"ai", 시작 설정 기준), `model_id`(AI만), `infer_ms`
      (AI 완료의 predict 호출 ms), `ai_instances`(AI 완료의 인스턴스별 클래스·점수·거부·잘림·면적 dict 튜플, 다각형 없음).
      모두 기본값 None인 끝 필드다. 실패·중단·세대 폐기에는 인스턴스가 없다.
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

    def __post_init__(self) -> None:
        """명령 ID를 만들지 않는다는 계약과 eval_run_id 형식(None 또는 비어 있지 않은 문자열)을 검사한다."""
        if self.command_id is not None:
            raise ValueError("command_id is not generated by A and must be None")

        if self.eval_run_id is not None:
            _require_text(self.eval_run_id, "eval_run_id")


@dataclass(frozen=True)
class QueueDropRecord:
    """작업을 시작하지 않고 대기 중 폐기한 프레임의 기록(계획 「인터페이스 제안」).

    작업이 없으므로 `job_seq`·처리 `generation`을 두지 않는다. `dropped_monotonic_ns`는 PC 단조 시각.
    """

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
    """입력 상태 기록(A 상세설계 §9 「입력 상태 로그」). 필드 정의만 두며 채우는 동작은 작업 5.

    공통 `InputStatusEvent` 5개 필드 + 상태 지속 시간(`duration_ns`, PC 단조 시계, 미확정이면 None).
    프레임 ID 필드를 두지 않는다.
    """

    run_id: str
    source_id: str
    status: InputStatus
    event_monotonic_ns: int
    detail: str | None
    duration_ns: int | None


@dataclass
class PipelineCounters:
    """처리 건수(A 상세설계 §9 「처리 건수」, 계획의 12개 계수). 파이프라인 안에서만 바꾼다.

    - 사유·종류별 계수는 열거의 `.value` 문자열을 키로 쓴다.
    - 작업 1~4가 모든 필드를 채운다. 무효 완료(`invalid_completed_by_reason`)를 처리 실패(`failed`)로
      세지 않는다. 시작한 작업은 completed·generation_discarded·failed·aborted 중 정확히 하나로 끝난다.
    - `invalidation_*_by_kind`에는 처리 오류 4종과 `GENERATION_CHANGED`가 kind 키로 나뉘어 들어간다.
      보고할 때 오류 4종 합계와 세대 전이 수를 따로 낸다.
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
