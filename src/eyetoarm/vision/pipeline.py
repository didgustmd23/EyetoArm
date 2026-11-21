"""A 비전 파이프라인 상태 코어: 세대·설정 전이·최신 대기 프레임·작업 시작·재처리.

단일 제어 흐름에서 호출하며 스레드를 만들지 않고, 주입한 `clock_ns`만 읽는다. C/D로 보내지 않는다.
사용: pipeline.offer_frame(frame); job = pipeline.start_next(); pipeline.complete(job, observation)
"""
from __future__ import annotations

import dataclasses
import weakref
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

from eyetoarm.common.types import (
    CandidateInvalidationEvent,
    FramePacket,
    InputStatusEvent,
    InvalidationKind,
    PixelBasis,
    PlanarObservation,
    check_axis_path,
)
from eyetoarm.vision.pipeline_types import (
    InvalidationRecord,
    JobContext,
    JobOutcome,
    JobStatus,
    PipelineCounters,
    PipelineSettings,
    QueueDropReason,
    QueueDropRecord,
)
from eyetoarm.vision.ai_config import AiSegConfig
from eyetoarm.vision.ai_result import AiSegResult
from eyetoarm.vision.planar import PlanarConfig, PlanarResult


class GenerationCause(str, Enum):
    """세대 증가 원인. 앞의 네 값은 설정 차이로 확인하고, 나머지는 설정이 같아도 세대를 올린다."""

    SOURCE_CHANGED = "source_changed"
    PROFILE_CHANGED = "profile_changed"
    MODE_CHANGED = "mode_changed"
    DETECTOR_CONFIG_CHANGED = "detector_config_changed"
    SEEK = "seek"
    RECONNECT = "reconnect"
    SIM_TIME_REWIND = "sim_time_rewind"


# 설정 객체의 차이로 확인하는 원인. 차이가 없으면 변경으로 세지 않는다
_SETTINGS_CAUSES = (
    GenerationCause.SOURCE_CHANGED,
    GenerationCause.PROFILE_CHANGED,
    GenerationCause.MODE_CHANGED,
    GenerationCause.DETECTOR_CONFIG_CHANGED,
)


class PipelineContractError(ValueError):
    """파이프라인 호출 계약 위반(다른 run/source 프레임, 앞선 수신 세대, 잘못된 설정·원인 조합 등)."""


# fail()이 받는 작업 오류 종류. 프레임 없는 로딩 실패·세대 전이는 작업 실패가 아니다
_FAIL_KINDS = (InvalidationKind.INFERENCE_ERROR, InvalidationKind.JOB_ABORTED,
               InvalidationKind.MODEL_CONFIG_INVALID)

# 관측 식별자 계약 위반의 cause 접두어. 새 kind를 만들지 않고 INFERENCE_ERROR로 정리한다
ADAPTER_CONTRACT_VIOLATION = "adapter_contract_violation"


def _same_id(value, expected) -> bool:
    """식별자가 기대값과 같은 타입·같은 값인지. 타입을 먼저 봐 배열 비교 예외·`False == 0` 일치를 막는다."""
    return type(value) is type(expected) and value == expected


def _endpoints_ok(observation, width, height) -> bool:
    """A-8 끝점 경로가 공통형 조합 규칙을 지키고, `raw` 끝점이면 영상 범위 안인지 본다.

    `undistorted`는 범위를 보지 않는다. 생성 뒤 변조된 값도 예외 없이 False로 돌려준다.
    """
    try:
        points = check_axis_path(
            observed_face=observation.observed_face, calc_contour=observation.calc_contour,
            axis_endpoints_px=observation.axis_endpoints_px, class_name=observation.class_name,
            valid=observation.valid)
    except (TypeError, ValueError):
        return False

    if points is None:
        return True

    basis = observation.pixel_basis

    if not isinstance(basis, str):
        return False

    if basis == PixelBasis.UNDISTORTED:
        return True

    if basis != PixelBasis.RAW:
        return False

    u, v = points[:, 0], points[:, 1]

    return bool((u >= 0).all() and (u <= width - 1).all() and (v >= 0).all() and (v <= height - 1).all())


@dataclass(frozen=True)
class _JobMeta:
    """시작한 작업의 스칼라 식별자. 영상·설정 객체를 붙잡지 않으려고 따로 둔다."""

    generation: int
    image_seq: int
    source_id: str
    detector_config_id: str


@dataclass(frozen=True)
class _TerminalMeta:
    """종료한 작업의 최초 terminal 의미(영상·윤곽 없음). `first_applied`는 duplicate에 복제하지 않는다."""

    status: JobStatus
    error: str | None
    invalidation: CandidateInvalidationEvent | None
    first_applied: bool


def _revalidated_config(config):
    """검출 설정이 자기 형의 생성자 검증을 다시 통과하는지 확인한다. 생성 뒤 변조면 계약 오류."""
    if not isinstance(config, (PlanarConfig, AiSegConfig)):
        raise PipelineContractError(f"detector_config must be PlanarConfig or AiSegConfig: {type(config).__name__}")

    init_fields = {f.name: getattr(config, f.name) for f in dataclasses.fields(config) if f.init}

    try:
        rebuilt = type(config)(**init_fields)
    except ValueError as exc:
        raise PipelineContractError(f"detector_config was modified after validation: {exc}") from exc

    if rebuilt.config_sha256 != config.config_sha256:
        raise PipelineContractError("detector_config was modified after validation")

    return config


def _setting_diffs(old: PipelineSettings, new: PipelineSettings) -> set[GenerationCause]:
    """두 설정의 차이를 원인 집합으로 돌려준다. 검출 설정은 `config_sha256`으로 비교한다."""
    diffs: set[GenerationCause] = set()

    if old.source_id != new.source_id:
        diffs.add(GenerationCause.SOURCE_CHANGED)

    if old.profile != new.profile:
        diffs.add(GenerationCause.PROFILE_CHANGED)

    if old.mode != new.mode:
        diffs.add(GenerationCause.MODE_CHANGED)

    if old.detector_config.config_sha256 != new.detector_config.config_sha256:
        diffs.add(GenerationCause.DETECTOR_CONFIG_CHANGED)

    return diffs


def _validated_settings(settings) -> PipelineSettings:
    """설정 형식과 검출 설정을 다시 검증한다. 상태를 바꾸기 전에 부른다."""
    if not isinstance(settings, PipelineSettings):
        raise PipelineContractError(f"settings must be PipelineSettings: {type(settings).__name__}")

    # 생성자 검증(형식)과 검출 설정(PlanarConfig·AiSegConfig) 검증을 다시 통과해야 한다
    PipelineSettings(settings.source_id, settings.mode, settings.detector_config, settings.profile)
    _revalidated_config(settings.detector_config)

    return settings


class VisionPipeline:
    """A 처리 상태 코어: 현재 세대·현재 관측·최신 대기 프레임 1개·활성 작업 1개·선택 프레임.

    수신 프레임은 소유 복사(읽기 전용)한다. 계약 오류는 수신 수·대기·기록에 영향을 주지 않는다.
    시작한 작업은 정확히 한 번 끝나고 재통지는 duplicate 응답이다. 작업 판별은 등록 JobContext와의 `is` 비교다.
    등록표는 JobContext를 약한 참조로만 두며, 종료 후 API는 계약 오류·None·duplicate·빈 tuple을 돌려준다.
    """

    def __init__(self, settings: PipelineSettings, *, run_id: str,
                 clock_ns: Callable[[], int]) -> None:
        """처음 설정·실행 ID·PC 단조 시계를 받는다. 세대·job_seq는 0에서 시작한다."""
        if not isinstance(run_id, str) or not run_id:
            raise PipelineContractError(f"run_id must be a non-empty string: {run_id!r}")

        if not callable(clock_ns):
            raise PipelineContractError("clock_ns must be callable")

        self._settings = _validated_settings(settings)
        self._run_id = run_id
        self._clock_ns = clock_ns

        # 세대·작업 순번
        self._generation = 0
        self._next_job_seq = 0

        # 마지막 비검출 전이 직후의 세대. 이보다 오래 전에 받은 프레임은 재처리하지 않는다
        self._non_detector_barrier = 0

        # 프레임 슬롯: 최신 대기 1개, 활성 작업 1개, 선택 프레임
        self._pending: FramePacket | None = None
        self._active: JobContext | None = None
        self._selected: FramePacket | None = None

        # 현재 관측(complete가 채운다)
        self._current_observation: PlanarObservation | None = None

        # 계수·기록 목록
        self._counters = PipelineCounters()
        self._queue_drops: list[QueueDropRecord] = []
        self._invalidation_events: list[CandidateInvalidationEvent] = []
        self._invalidation_records: list[InvalidationRecord] = []
        self._event_keys: set[CandidateInvalidationEvent] = set()

        # 세대별 오류 차단선(이 job_seq 이하의 결과는 현재 관측이 될 수 없다)
        self._error_watermarks: dict[int, int] = {}

        # 세대별 출처(source_id, detector_config_id). 한 세대 안에서 설정은 고정이다
        self._generation_history: dict[int, tuple[str, str]] = {
            0: (self._settings.source_id, self._settings.detector_config_id)}

        # 작업 등록표(약한 참조)·작업 스칼라 요약·terminal 요약
        self._job_refs: weakref.WeakValueDictionary[int, JobContext] = weakref.WeakValueDictionary()
        self._job_meta: dict[int, _JobMeta] = {}
        self._terminal: dict[int, _TerminalMeta] = {}

        self._closed = False

    # ------------------------------------------------------------------ 조회

    @property
    def run_id(self) -> str:
        """이 파이프라인의 실행 ID."""
        return self._run_id

    @property
    def settings(self) -> PipelineSettings:
        """현재 처리 설정 스냅샷."""
        return self._settings

    def current_generation(self) -> int:
        """현재 세대. 입력 소스의 `generation_provider`로 주입한다."""
        return self._generation

    @property
    def current_observation(self) -> PlanarObservation | None:
        """현재 관측. 세대 전이 직후에는 None."""
        return self._current_observation

    @property
    def selected_frame(self) -> FramePacket | None:
        """선택 프레임(마지막으로 `start_next()`가 시작한 소유 복사 프레임). 없으면 None."""
        return self._selected

    @property
    def pending_frame(self) -> FramePacket | None:
        """최신 대기 프레임(소유 복사). 없으면 None."""
        return self._pending

    @property
    def counters(self) -> PipelineCounters:
        """처리 건수의 복사본. 고쳐도 파이프라인 계수는 바뀌지 않는다."""
        c = self._counters

        return dataclasses.replace(
            c,
            invalid_completed_by_reason=dict(c.invalid_completed_by_reason),
            queue_dropped=dict(c.queue_dropped),
            invalidation_applied_by_kind=dict(c.invalidation_applied_by_kind),
            invalidation_ignored_by_kind=dict(c.invalidation_ignored_by_kind),
        )

    @property
    def queue_drop_records(self) -> tuple[QueueDropRecord, ...]:
        """대기 중 폐기한 프레임 기록(발생 순)."""
        return tuple(self._queue_drops)

    @property
    def invalidation_events(self) -> tuple[CandidateInvalidationEvent, ...]:
        """기록한 무효화 사건(적용·기록만 모두, 발생 순). 메모리 내 조회이며 외부 발행이 아니다."""
        return tuple(self._invalidation_events)

    @property
    def invalidation_records(self) -> tuple[InvalidationRecord, ...]:
        """기록한 사건과 applied 판정 쌍(발생 순, `invalidation_events`와 1:1)."""
        return tuple(self._invalidation_records)

    @property
    def closed(self) -> bool:
        """`close()`를 불렀는지."""
        return self._closed

    # ------------------------------------------------------------------ 수신·시작

    def offer_frame(self, frame: FramePacket) -> bool:
        """수신 계약을 검사하고 소유 복사해 대기에 둔다. 대기에 두면 True, 오래된 캡처로 폐기하면 False."""
        # 수신 계약(위반은 예외, 상태 무변)
        if self._closed:
            raise PipelineContractError("offer_frame after close")

        if not isinstance(frame, FramePacket):
            raise PipelineContractError(f"offer_frame needs FramePacket: {type(frame).__name__}")

        if frame.run_id != self._run_id:
            raise PipelineContractError(f"run_id mismatch: {frame.run_id!r} != {self._run_id!r}")

        if frame.source_id != self._settings.source_id:
            raise PipelineContractError(
                f"source_id mismatch: {frame.source_id!r} != {self._settings.source_id!r}")

        if frame.capture_generation > self._generation:
            raise PipelineContractError(
                f"capture_generation {frame.capture_generation} is ahead of {self._generation}")

        if not isinstance(frame.image, np.ndarray):
            raise PipelineContractError("frame.image must be a numpy array")

        self._counters.received += 1

        # 오래된 캡처는 새 수신으로 시작하지 않는다
        if frame.capture_generation < self._generation:
            self._record_drop(frame, QueueDropReason.STALE_CAPTURE, self._clock_ns())
            return False

        owned = self._owned_copy(frame)

        # 최신 대기 1개: 이전 대기만 교체·폐기한다
        if self._pending is not None:
            self._drop_pending(QueueDropReason.REPLACED_BY_NEWER, self._clock_ns())

        self._pending = owned

        return True

    def start_next(self) -> JobContext | None:
        """활성 작업이 없고 대기 프레임이 있으면 작업을 시작해 돌려준다. 아니면 None(상태 무변)."""
        if self._closed or self._active is not None or self._pending is None:
            return None

        frame = self._pending
        job = JobContext(frame=frame, generation=self._generation, job_seq=self._next_job_seq,
                         settings=self._settings, started_monotonic_ns=self._clock_ns(),
                         reprocess=False)

        self._pending = None
        self._register(job)
        self._selected = frame
        self._counters.first_started += 1

        return job

    def start_reprocess(self) -> JobContext | None:
        """선택 프레임 재처리를 시작한다. 재처리 조건을 만족하지 않거나 활성 작업이 있으면 None."""
        frame = self._selected

        if self._closed or self._active is not None or frame is None or not self._reprocess_allowed(frame):
            return None

        # 같은 프레임 객체 재사용: image_seq·capture_generation·원본/수신 시각 그대로
        job = JobContext(frame=frame, generation=self._generation, job_seq=self._next_job_seq,
                         settings=self._settings, started_monotonic_ns=self._clock_ns(),
                         reprocess=True)

        self._register(job)
        self._counters.reprocess_started += 1

        return job

    # ------------------------------------------------------------------ 세대 전이

    def change_settings(self, settings: PipelineSettings, *,
                        cause: GenerationCause | str) -> CandidateInvalidationEvent | None:
        """설정 변경·전이를 적용한다. 세대를 1 올리면 `GENERATION_CHANGED` 사건, 변경이 아니면 None.

        설정 차이 원인에서 차이가 없으면 None, 선언한 원인이 안 바뀌었으면 계약 오류다.
        검출 설정만 바뀐 경우에만 선택 프레임을 유지한다. 검증은 상태를 바꾸기 전에 끝낸다.
        """
        # 검증(실패 시 상태 무변)
        if self._closed:
            raise PipelineContractError("change_settings after close")

        cause = GenerationCause(cause)
        settings = _validated_settings(settings)
        diffs = _setting_diffs(self._settings, settings)

        if cause in _SETTINGS_CAUSES:
            if not diffs:
                return None

            if cause not in diffs:
                raise PipelineContractError(
                    f"cause {cause.value} does not match changed items {sorted(d.value for d in diffs)}")

        detector_only = cause is GenerationCause.DETECTOR_CONFIG_CHANGED and diffs == {cause}

        # 원인 기록: 선언한 원인 + 함께 바뀐 항목
        extra = sorted(d.value for d in diffs - {cause})
        cause_text = "+".join([cause.value] + extra)

        now = self._clock_ns()
        new_generation = self._generation + 1
        event = CandidateInvalidationEvent(
            run_id=self._run_id, source_id=settings.source_id,
            kind=InvalidationKind.GENERATION_CHANGED, cause=cause_text, event_monotonic_ns=now,
            generation=new_generation, detector_config_id=settings.detector_config.config_sha256,
            job_seq=None, image_seq=None)

        # 적용: 세대 증가, 현재 관측 즉시 무효, 남은 대기 폐기
        self._generation = new_generation
        self._settings = settings
        self._current_observation = None
        self._drop_pending(QueueDropReason.GENERATION_CHANGED, now)

        if not detector_only:
            self._clear_selection()
            self._non_detector_barrier = new_generation

        self._generation_history[new_generation] = (settings.source_id, settings.detector_config_id)
        self._record_event(event, applied=True)

        return event

    # ------------------------------------------------------------------ terminal 처리·무효화

    def complete(self, job: JobContext, observation: PlanarObservation, *,
                 display_result: PlanarResult | AiSegResult | None = None) -> JobOutcome:
        """작업 완료를 적용한다. 무효 관측도 정상 완료다.

        식별자 불일치는 INFERENCE_ERROR 실패, 세대가 바뀌었으면 폐기, 차단선 뒤 작업만 현재 관측이 된다.
        """
        self._require_registered(job)

        if job.job_seq in self._terminal:
            return self._duplicate(job)

        mismatches = self._observation_mismatches(job, observation, display_result)

        if mismatches:
            cause = f"{ADAPTER_CONTRACT_VIOLATION}:{','.join(mismatches)}"
            return self._fail_job(job, InvalidationKind.INFERENCE_ERROR, cause)

        # 처리 중 세대가 바뀌었으면 결과를 버린다. 기록은 시작 당시 context 그대로다
        if job.generation != self._generation:
            self._counters.generation_discarded += 1
            outcome = JobOutcome(context=job, status=JobStatus.DISCARDED_GENERATION, observation=None,
                                 display_result=None, error=None, invalidation=None)
            return self._finish(job, outcome)

        # 정상 완료(유효·무효 모두). 무효 관측은 실패로 세지 않는다
        c = self._counters
        c.completed += 1

        if observation.valid:
            c.valid_completed += 1
        else:
            reasons = c.invalid_completed_by_reason
            reasons[observation.reason] = reasons.get(observation.reason, 0) + 1

        # 차단선 이하(오류 전 시작) 작업은 현재 관측이 될 수 없다
        applied = job.job_seq > self._error_watermarks.get(self._generation, -1)

        if applied:
            self._current_observation = observation

        outcome = JobOutcome(context=job, status=JobStatus.COMPLETED, observation=observation,
                             display_result=display_result, error=None, invalidation=None,
                             applied=applied)

        return self._finish(job, outcome)

    def fail(self, job: JobContext, error_kind: InvalidationKind, cause: str) -> JobOutcome:
        """작업 실패·중단을 적용한다. 관측을 만들지 않고 해당 작업의 사건을 한 번 평가·기록한다.

        `error_kind`는 INFERENCE_ERROR·MODEL_CONFIG_INVALID·JOB_ABORTED만 받는다. 끝난 작업이면 duplicate.
        """
        self._require_registered(job)

        if job.job_seq in self._terminal:
            return self._duplicate(job)

        try:
            kind = InvalidationKind(error_kind)
        except ValueError as exc:
            raise PipelineContractError(f"unknown error_kind: {error_kind!r}") from exc

        if kind not in _FAIL_KINDS:
            raise PipelineContractError(f"fail() does not accept {kind.value}")

        if not isinstance(cause, str) or not cause:
            raise PipelineContractError(f"cause must be a non-empty string: {cause!r}")

        return self._fail_job(job, kind, cause)

    def apply_invalidation(self, event: CandidateInvalidationEvent) -> bool:
        """외부(어댑터·로더)에서 온 무효화 사건을 적용한다. 현재 세대 차단에 반영하면 True.

        출처가 맞지 않으면 계약 오류, 같은 사건이면 False다. 활성 작업을 끝내지 않는다.
        """
        if self._closed:
            raise PipelineContractError("apply_invalidation after close")

        if not isinstance(event, CandidateInvalidationEvent):
            raise PipelineContractError(
                f"apply_invalidation needs CandidateInvalidationEvent: {type(event).__name__}")

        self._validate_event_origin(event)

        if event in self._event_keys:
            return False

        if event.kind is InvalidationKind.GENERATION_CHANGED:
            raise PipelineContractError("GENERATION_CHANGED is issued only by change_settings")

        return self._evaluate_event(event)

    def note_input_status(self, event: InputStatusEvent) -> None:
        """입력 사건(입력 오류·EOF 등)을 반영한다: 현재 관측·선택을 비우고 대기를 폐기한다.

        세대를 올리지 않고 무효화 사건을 만들지 않는다. 활성 작업은 건드리지 않는다.
        """
        if self._closed:
            raise PipelineContractError("note_input_status after close")

        if not isinstance(event, InputStatusEvent):
            raise PipelineContractError(f"note_input_status needs InputStatusEvent: {type(event).__name__}")

        if event.run_id != self._run_id or event.source_id != self._settings.source_id:
            raise PipelineContractError("input status event from another run/source")

        self._current_observation = None
        self._clear_selection()
        self._drop_pending(QueueDropReason.INPUT_STATUS, self._clock_ns())

    def close(self) -> tuple[JobOutcome, ...]:
        """종료 처리. 활성 작업을 ABORTED로 한 번 정리하고 관측·선택·대기를 비운다. 재호출은 빈 tuple.

        시계 예외·비정수는 상태 변경 전에 나므로 다시 부를 수 있다.
        """
        if self._closed:
            return ()

        # 시계를 미리 한 번만 읽어 정리 도중 시계 예외로 일부만 정리되는 일을 막는다
        now = self._clock_ns()

        if type(now) is not int:
            raise PipelineContractError(f"clock_ns must return int: {now!r}")

        outcomes: list[JobOutcome] = []

        if self._active is not None:
            outcomes.append(self._fail_job(self._active, InvalidationKind.JOB_ABORTED, "closed", now_ns=now))

        self._current_observation = None
        self._clear_selection()
        self._drop_pending(QueueDropReason.CLOSED, now)
        self._closed = True

        return tuple(outcomes)

    # ------------------------------------------------------------------ 내부 헬퍼

    def _release_active(self, job: JobContext) -> None:
        """그 작업이 활성 작업이면 활성 슬롯만 비운다."""
        if self._active is job:
            self._active = None

    def _register(self, job: JobContext) -> None:
        """시작한 작업을 활성 슬롯·약한 참조 등록표·스칼라 요약에 올리고 job_seq를 1 올린다."""
        self._next_job_seq += 1
        self._active = job
        self._job_refs[job.job_seq] = job
        self._job_meta[job.job_seq] = _JobMeta(
            generation=job.generation, image_seq=job.frame.image_seq,
            source_id=job.frame.source_id, detector_config_id=job.detector_config_id)

    def _require_registered(self, job) -> None:
        """이 파이프라인이 발급한 바로 그 JobContext인지 `is`로 확인한다. 아니면 계약 오류."""
        if not isinstance(job, JobContext):
            raise PipelineContractError(f"job must be JobContext: {type(job).__name__}")

        seq = job.job_seq

        if not isinstance(seq, int) or isinstance(seq, bool) or self._job_refs.get(seq) is not job:
            raise PipelineContractError(f"unknown or copied job: job_seq={seq!r}")

    def _duplicate(self, job: JobContext) -> JobOutcome:
        """이미 끝난 작업의 재통지 응답. 최초 status·error·사건만 담고 아무것도 바꾸지 않는다."""
        meta = self._terminal[job.job_seq]

        return JobOutcome(context=job, status=meta.status, observation=None, display_result=None,
                          error=meta.error, invalidation=meta.invalidation, applied=False, duplicate=True)

    def _finish(self, job: JobContext, outcome: JobOutcome) -> JobOutcome:
        """terminal 응답을 스칼라 요약으로 남기고 활성 슬롯을 비운다."""
        self._terminal[job.job_seq] = _TerminalMeta(
            status=outcome.status, error=outcome.error, invalidation=outcome.invalidation,
            first_applied=outcome.applied)
        self._release_active(job)

        return outcome

    def _fail_job(self, job: JobContext, kind: InvalidationKind, cause: str, *,
                  now_ns: int | None = None) -> JobOutcome:
        """작업 실패·중단: 실제 작업 식별자로 사건을 만들어 평가·기록하고 FAILED/ABORTED로 끝낸다.

        `now_ns`를 주면 시계를 다시 읽지 않는다. 사건 생성이 먼저라 예외가 나면 상태는 그대로다.
        """
        event = CandidateInvalidationEvent(
            run_id=self._run_id, source_id=job.frame.source_id, kind=kind, cause=cause,
            event_monotonic_ns=self._clock_ns() if now_ns is None else now_ns, generation=job.generation,
            detector_config_id=job.detector_config_id, job_seq=job.job_seq, image_seq=job.frame.image_seq)

        self._evaluate_event(event)

        if kind is InvalidationKind.JOB_ABORTED:
            status = JobStatus.ABORTED
            self._counters.aborted += 1
        else:
            status = JobStatus.FAILED
            self._counters.failed += 1

        outcome = JobOutcome(context=job, status=status, observation=None, display_result=None,
                             error=cause, invalidation=event)

        return self._finish(job, outcome)

    def _evaluate_event(self, event: CandidateInvalidationEvent) -> bool:
        """사건을 평가·기록한다. 현재 세대에 반영하면 True(applied 계수).

        이전 세대는 ignored, 작업 사건은 그 job_seq까지, 프레임 없는 사건은 시작한 모든 작업까지 차단한다.
        같은 9필드 사건이 이미 있으면 다시 반영하지 않고 False다.
        """
        if event in self._event_keys:
            return False

        if event.generation < self._generation:
            self._record_event(event, applied=False)
            return False

        if event.job_seq is None:
            limit = self._next_job_seq - 1
        else:
            limit = event.job_seq

        marks = self._error_watermarks
        marks[self._generation] = max(marks.get(self._generation, -1), limit)

        current = self._current_observation

        if current is not None and current.job_seq <= limit:
            self._current_observation = None

        self._record_event(event, applied=True)

        return True

    def _record_event(self, event: CandidateInvalidationEvent, *, applied: bool) -> None:
        """사건을 한 번 기록하고 kind별 applied/ignored 계수를 1 올린다."""
        self._invalidation_events.append(event)
        self._invalidation_records.append(InvalidationRecord(event, applied))
        self._event_keys.add(event)

        counts = (self._counters.invalidation_applied_by_kind if applied
                  else self._counters.invalidation_ignored_by_kind)
        counts[event.kind.value] = counts.get(event.kind.value, 0) + 1

    def _validate_event_origin(self, event: CandidateInvalidationEvent) -> None:
        """사건이 이 실행에서 실제로 있었던 세대·출처·작업에서 왔는지 확인한다. 아니면 계약 오류."""
        int_or_none = (int, type(None))
        typed = all((
            type(event.run_id) is str, type(event.source_id) is str,
            type(event.detector_config_id) is str, type(event.cause) is str,
            type(event.generation) is int, type(event.event_monotonic_ns) is int,
            type(event.job_seq) in int_or_none, type(event.image_seq) in int_or_none,
            isinstance(event.kind, InvalidationKind),
        ))

        if not typed:
            raise PipelineContractError("event fields have unexpected types")

        if event.run_id != self._run_id:
            raise PipelineContractError(f"event run_id mismatch: {event.run_id!r}")

        origin = self._generation_history.get(event.generation)

        if origin is None:
            raise PipelineContractError(f"event generation {event.generation} was never issued")

        if (event.source_id, event.detector_config_id) != origin:
            raise PipelineContractError(
                f"event source/config does not match generation {event.generation}")

        if event.job_seq is None:
            return

        meta = self._job_meta.get(event.job_seq)

        if meta is None:
            raise PipelineContractError(f"event job_seq {event.job_seq} was never issued")

        if (meta.generation, meta.image_seq, meta.source_id, meta.detector_config_id) != (
                event.generation, event.image_seq, event.source_id, event.detector_config_id):
            raise PipelineContractError(f"event does not match job {event.job_seq}")

    def _observation_mismatches(self, job: JobContext, observation, display_result) -> list[str]:
        """관측 타입·식별자가 작업과 어긋난 필드 이름 목록. 비어 있으면 계약을 지킨 관측이다."""
        if not isinstance(observation, PlanarObservation):
            return ["observation_type"]

        f = job.frame
        expected = {
            "run_id": f.run_id, "source_id": f.source_id, "image_seq": f.image_seq,
            "capture_generation": f.capture_generation, "generation": job.generation,
            "job_seq": job.job_seq, "detector_config_id": job.detector_config_id,
            "calibration_id": f.calibration_id,
        }
        bad = [name for name, value in expected.items() if not _same_id(getattr(observation, name), value)]

        # 생성 뒤 변조된 판정 필드가 계수·현재 관측 판단에서 예외로 새지 않게 여기서 거른다
        valid, reason = observation.valid, observation.reason

        if type(valid) is not bool:
            bad.append("valid")
        elif valid and reason is not None:
            bad.append("reason")
        elif not valid and (type(reason) is not str or not reason):
            bad.append("reason")

        # 공통형은 프레임 크기를 모르므로 raw 끝점의 영상 범위는 여기서 본다
        if not _endpoints_ok(observation, f.width, f.height):
            bad.append("axis_endpoints_px")

        # 방식 일치: 시작 설정의 검출 방식과 관측·표시 결과 형이 같아야 한다
        is_ai = isinstance(job.settings.detector_config, AiSegConfig)

        # 생성 뒤 변조된 배열·비문자열이 진릿값 예외로 새지 않도록 타입까지 같은지 본다
        if not _same_id(observation.method, "ai" if is_ai else "opencv"):
            bad.append("method")

        if display_result is not None:
            if not isinstance(display_result, AiSegResult if is_ai else PlanarResult):
                bad.append("display_result_type")
            elif is_ai and not _same_id(display_result.config_sha256, job.detector_config_id):
                bad.append("display_result_config")

        return bad

    def _clear_selection(self) -> None:
        """선택 프레임을 비운다."""
        self._selected = None

    def _reprocess_allowed(self, frame: FramePacket) -> bool:
        """선택 프레임 재처리 조건: 수신 뒤 비검출 전이 없음, 현재 소스·프로파일이 그 프레임에 유효."""
        if frame.capture_generation < self._non_detector_barrier:
            return False

        if frame.run_id != self._run_id or frame.source_id != self._settings.source_id:
            return False

        profile = self._settings.profile

        return profile is not None and profile.applies_to(frame)

    def _owned_copy(self, frame: FramePacket) -> FramePacket:
        """영상을 독립 복사·읽기 전용으로 바꾼 같은 타입의 프레임. 다른 필드는 그대로 옮긴다."""
        image = np.array(frame.image, copy=True, order="C")
        image.setflags(write=False)

        return dataclasses.replace(frame, image=image)

    def _drop_pending(self, reason: QueueDropReason, now_ns: int) -> None:
        """대기 프레임이 있으면 사유와 함께 한 번 폐기 기록하고 비운다."""
        if self._pending is None:
            return

        frame = self._pending
        self._pending = None
        self._record_drop(frame, reason, now_ns)

    def _record_drop(self, frame: FramePacket, reason: QueueDropReason, now_ns: int) -> None:
        """시작하지 않은 프레임의 폐기 기록을 남기고 사유별 계수를 1 올린다."""
        self._queue_drops.append(QueueDropRecord(
            run_id=frame.run_id, source_id=frame.source_id, image_seq=frame.image_seq,
            capture_generation=frame.capture_generation, timestamp_ms=frame.timestamp_ms,
            timestamp_base=frame.timestamp_base, received_monotonic_ns=frame.received_monotonic_ns,
            ros_stamp_ns=frame.ros_stamp_ns, reason=reason, dropped_monotonic_ns=now_ns))

        counts = self._counters.queue_dropped
        counts[reason.value] = counts.get(reason.value, 0) + 1
