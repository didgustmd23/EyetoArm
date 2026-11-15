"""A 비전 파이프라인 상태 코어 — 세대·설정 전이·최신 대기 프레임·작업 시작·재처리.

근거: 계획 `docs/superpowers/plans/2026-11-15-a-vision-pipeline.md` r3 「인터페이스 제안」·작업 2·3,
A 상세설계 §3(최신 프레임 1개·처리 중 버퍼 비덮어쓰기)·§5.1(image_seq 수명)·§5.4(세대 증가 조건·
설정 스냅샷과 폐기·동일 프레임 재처리 조건)·§9(처리 건수), 공통 자료형 관리 초안 §3.5·§4.

사용법(작업 1~3 범위):
    pipeline = VisionPipeline(settings, run_id=issuer.run_id, clock_ns=time.monotonic_ns)
    source = StoredImageSource(paths, source_id=settings.source_id, issuer=issuer,
                               generation_provider=pipeline.current_generation)
    pipeline.offer_frame(source.read())      # 소유 복사 후 최신 대기 1개
    job = pipeline.start_next()              # JobContext 또는 None
    event = pipeline.change_settings(new_settings, cause=GenerationCause.DETECTOR_CONFIG_CHANGED)

- 이름·API는 계획의 **제안**이다. 상태 코어는 단일 제어 흐름에서 호출하며 내부에서 스레드를 만들지 않는다.
  시계는 주입한 `clock_ns`만 읽는다.
- `complete`·`fail`·`apply_invalidation`·`close`는 작업 4 범위라 아직 NotImplementedError다. 활성 작업·
  오류 이력 자리·기록 목록은 작업 4·5가 붙일 수 있게 분리해 두었다.
- 입력 사건을 코어에 알리는 메서드는 계획에 없어 만들지 않았다(M2). 선택·대기를 비우는 내부 헬퍼만 둔다.
- TCP·ROS·Qt·B/C/D 모듈을 import하지 않는다. A가 최종 후보·좌표·규격을 정하지 않는다.
"""
from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Callable

import numpy as np

from eyetoarm.common.types import (
    CandidateInvalidationEvent,
    FramePacket,
    InvalidationKind,
    PlanarObservation,
)
from eyetoarm.vision.pipeline_types import (
    JobContext,
    JobOutcome,
    PipelineCounters,
    PipelineSettings,
    QueueDropReason,
    QueueDropRecord,
)
from eyetoarm.vision.planar import PlanarConfig, PlanarResult


class GenerationCause(str, Enum):
    """세대 증가 원인(A 상세설계 §5.4 「세대 증가 조건」). 문자열 값은 A 내부 임시 결정이다.

    앞의 네 값은 설정 객체의 차이로 확인하는 변경이고, 탐색·재접속·시각 되감기는 설정이 같아도
    세대를 올리는 전이다. 재처리를 허용하는 것은 검출 설정 변경뿐이다.
    """

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


def _revalidated_config(config: PlanarConfig) -> PlanarConfig:
    """`PlanarConfig` 생성자 검증을 다시 통과하는지 확인한다. 생성 뒤 값이 바뀌었으면 계약 오류."""
    init_fields = {f.name: getattr(config, f.name) for f in dataclasses.fields(config) if f.init}
    rebuilt = PlanarConfig(**init_fields)

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

    # 생성자 검증(형식)과 PlanarConfig 검증을 다시 통과해야 한다
    PipelineSettings(settings.source_id, settings.mode, settings.detector_config, settings.profile)
    _revalidated_config(settings.detector_config)

    return settings


class VisionPipeline:
    """A 처리 상태 코어: 현재 세대·현재 관측·최신 대기 프레임 1개·활성 작업 1개·선택 프레임.

    수신 계약(`offer_frame`)
    - `run_id`가 파이프라인과 다르거나 `source_id`가 현재 설정과 다르면 `PipelineContractError`.
      `capture_generation`이 현재 세대보다 크면(공급자가 앞섬) 역시 계약 오류다(M4 임시 결정).
      계약 오류는 수신 수·대기·기록에 영향을 주지 않는다.
    - 계약을 통과하면 `received` +1. 현재 세대보다 오래된 캡처는 시작하지 않고 폐기 기록(False).
    - 통과한 프레임은 이 자리에서 영상을 독립 복사하고 읽기 전용으로 표시해 소유한다. 복사 프레임은
      원본과 같은 타입(공통형·A 확장형)이며 모든 식별자·시각·A 확장 필드를 그대로 갖는다.
      복사 시간을 따로 재거나 `process_ms`에 합치지 않는다.
    - 대기 프레임이 있으면 교체하고 이전 대기 프레임을 폐기 기록한다. 처리 중 작업은 건드리지 않는다.

    작업 시작
    - `start_next()`는 활성 작업이 없고 대기 프레임이 있을 때만 시작한다. 시작한 프레임이 선택 프레임이 된다.
    - `start_reprocess()`는 선택 프레임을 같은 객체로 다시 쓴다. 입력 소스·발급기를 부르지 않는다.
    - `generation`·`job_seq`는 0에서 시작한다. `job_seq`는 세대 변경으로 초기화하지 않는다.

    세대 전이(`change_settings`)
    - 전이마다 세대를 1 올리고 현재 관측을 즉시 None으로 두며 남은 대기 프레임을 폐기 기록한다.
    - 검출 설정만 바뀐 전이는 선택 프레임을 유지하고, 그 밖의 전이는 선택을 비운다.
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

        # 마지막 비검출 전이 직후의 세대. 이보다 오래 전에 받은 프레임은 재처리하지 않는다(§5.4 조건 2)
        self._non_detector_barrier = 0

        # 프레임 슬롯: 최신 대기 1개, 활성 작업 1개, 선택 프레임
        self._pending: FramePacket | None = None
        self._active: JobContext | None = None
        self._selected: FramePacket | None = None

        # 현재 관측(작업 4의 complete가 채운다)
        self._current_observation: PlanarObservation | None = None

        # 계수·기록 목록
        self._counters = PipelineCounters()
        self._queue_drops: list[QueueDropRecord] = []
        self._invalidation_events: list[CandidateInvalidationEvent] = []

        # 작업 4 자리: 세대별 오류 차단선(job_seq)과 종료한 작업의 terminal 결과
        self._error_watermarks: dict[int, int] = {}
        self._terminal_outcomes: dict[int, JobOutcome] = {}

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
        """최신 대기 프레임(소유 복사). 없으면 None. 보존식 확인용 조회(구현 추가)."""
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

    # ------------------------------------------------------------------ 수신·시작

    def offer_frame(self, frame: FramePacket) -> bool:
        """수신 계약을 검사하고 소유 복사해 대기에 둔다. 대기에 두면 True, 오래된 캡처로 폐기하면 False."""
        # 수신 계약(위반은 예외, 상태 무변)
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
        if self._active is not None or self._pending is None:
            return None

        frame = self._pending
        job = JobContext(frame=frame, generation=self._generation, job_seq=self._next_job_seq,
                         settings=self._settings, started_monotonic_ns=self._clock_ns(),
                         reprocess=False)

        self._pending = None
        self._next_job_seq += 1
        self._active = job
        self._selected = frame
        self._counters.first_started += 1

        return job

    def start_reprocess(self) -> JobContext | None:
        """선택 프레임 재처리를 시작한다. 조건(§5.4 1~3)을 만족하지 않거나 활성 작업이 있으면 None."""
        frame = self._selected

        if self._active is not None or frame is None or not self._reprocess_allowed(frame):
            return None

        # 같은 프레임 객체 재사용: image_seq·capture_generation·원본/수신 시각 그대로
        job = JobContext(frame=frame, generation=self._generation, job_seq=self._next_job_seq,
                         settings=self._settings, started_monotonic_ns=self._clock_ns(),
                         reprocess=True)

        self._next_job_seq += 1
        self._active = job
        self._counters.reprocess_started += 1

        return job

    # ------------------------------------------------------------------ 세대 전이

    def change_settings(self, settings: PipelineSettings, *,
                        cause: GenerationCause | str) -> CandidateInvalidationEvent | None:
        """설정 변경·전이를 적용한다. 세대를 1 올리면 `GENERATION_CHANGED` 사건, 변경이 아니면 None.

        - 설정 차이로 확인하는 원인(소스·프로파일·모드·검출 설정)에서 설정이 같으면(검출 설정은 같은
          `config_sha256`) 변경으로 세지 않고 None. 차이가 있는데 선언한 원인의 항목이 안 바뀌었으면
          계약 오류.
        - 탐색·재접속·시각 되감기는 설정이 같아도 세대를 올린다.
        - 검출 설정만 바뀐 경우에만 선택 프레임을 유지한다. 다른 항목이 함께 바뀌면 비검출 전이로 본다.
        - 검증은 상태를 바꾸기 전에 끝낸다. 실패하면 세대·현재 결과·선택·대기·기록이 그대로다.
        """
        # 검증(실패 시 상태 무변)
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

        self._invalidation_events.append(event)

        return event

    # ------------------------------------------------------------------ 작업 4 자리

    def complete(self, job: JobContext, observation: PlanarObservation, *,
                 display_result: PlanarResult | None = None) -> JobOutcome:
        """작업 완료 적용(작업 4에서 구현)."""
        raise NotImplementedError("complete() is planned for task 4")

    def fail(self, job: JobContext, error_kind: InvalidationKind, cause: str) -> JobOutcome:
        """작업 실패 적용(작업 4에서 구현)."""
        raise NotImplementedError("fail() is planned for task 4")

    def apply_invalidation(self, event: CandidateInvalidationEvent) -> bool:
        """무효화 사건 적용(작업 4에서 구현)."""
        raise NotImplementedError("apply_invalidation() is planned for task 4")

    def close(self) -> tuple[JobOutcome, ...]:
        """종료 처리(작업 4에서 구현)."""
        raise NotImplementedError("close() is planned for task 4")

    # ------------------------------------------------------------------ 내부 헬퍼

    def _release_active(self, job: JobContext) -> None:
        """그 작업이 활성 작업이면 활성 슬롯만 비운다. 결과·계수·기록은 바꾸지 않는다.

        작업 4의 terminal 처리(complete·fail·close)가 쓸 골격이다. 작업 4 전에는 시험에서만 쓴다.
        """
        if self._active is job:
            self._active = None

    def _clear_selection(self) -> None:
        """선택 프레임을 비운다. 입력 사건·close(작업 4·6)에서 재사용할 골격(M2)."""
        self._selected = None

    def _reprocess_allowed(self, frame: FramePacket) -> bool:
        """선택 프레임 재처리 조건(§5.4 2·3): 수신 뒤 비검출 전이 없음, 현재 소스·프로파일이 그 프레임에 유효."""
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
