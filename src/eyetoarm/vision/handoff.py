"""A→C 같은 프로세스 내부 연결부: 순서 있는 제어 대기열·최신 1칸 영상·현재 결과 거울·fail-stop.

생산자 메서드는 첫 생산자 호출 스레드에 고정되고, 소비자(수신 측 1개)는 꺼내기·현재성 확인만 한다. 잠금 안에서는
참조 대입·계수만 하며, 넘침·notify 예외·수신 측 실패는 `FAILED(reason)`로 latch되고 자동 복구는 없다.
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

from eyetoarm.common.types import (
    CandidateInvalidationEvent,
    InputStatus,
    InputStatusEvent,
    InvalidationKind,
    PlanarObservation,
    TimestampBase,
)
from eyetoarm.vision.ai_config import AiSegConfig
from eyetoarm.vision.ai_result import AiSegResult
from eyetoarm.vision.live_view import draw_detections, draw_track_unavailable
from eyetoarm.vision.pipeline import VisionPipeline
from eyetoarm.vision.pipeline_types import DisplayResult, JobOutcome, JobStatus
from eyetoarm.vision.tracking import TrackFrame, TrackMode, matches_job
from eyetoarm.vision.world_tracking_display import PrevDisplay

# 제어 대기열 용량(개발 기본값, 확정값 아님)
DEFAULT_CAPACITY = 64


class HandoffError(RuntimeError):
    """연결부 오류의 공통 부모."""


class HandoffContractError(HandoffError):
    """호출 계약 위반(잘못된 인자·스레드·순서). 상태를 바꾸지 않는다."""


class HandoffFailed(HandoffError):
    """연결부가 FAILED 상태다(넘침·notify 예외·수신 측 실패). 데이터는 넣지 않았다."""


class HandoffClosed(HandoffError):
    """연결부가 닫혔다. 데이터는 넣지 않았다."""


class HandoffState(str, Enum):
    """연결부 상태. OPEN에서 FAILED 또는 CLOSED로만 간다(latch)."""

    OPEN = "open"
    FAILED = "failed"
    CLOSED = "closed"


class MsgKind(str, Enum):
    """제어 메시지 종류."""

    RUN_START = "RUN_START"          # 전체 해제
    INVALIDATION = "INVALIDATION"    # 범위 무효화, applied=False는 상태 무변
    INPUT_STATUS = "INPUT_STATUS"    # 전체 해제(세대 무변)
    RESULT = "RESULT"                # 현재 결과 교체
    CLEAR = "CLEAR"                  # 전체 해제(결과 보류 등)
    RUN_END = "RUN_END"              # 전체 해제


@dataclass(frozen=True)
class HandoffStatus:
    """상태와 사유. 사유는 FAILED가 된 첫 원인이며 close 뒤에도 유지된다."""

    state: HandoffState
    reason: str | None


@dataclass(frozen=True)
class HandoffMessage:
    """제어 메시지 하나. 해당 없는 필드는 None(가짜 0 없음).

    `msg_seq`는 1부터 단조이며 실행이 바뀌어도 초기화하지 않는다. SIM_TIME 촬영 시각과 PC 단조 시각을 서로 빼지 않는다.
    """

    msg_seq: int
    msg_kind: MsgKind
    run_id: str
    source_id: str
    core_generation: int
    published_monotonic_ns: int
    event: CandidateInvalidationEvent | None = None
    applied: bool | None = None
    status: InputStatus | None = None
    detail: str | None = None
    event_monotonic_ns: int | None = None
    image_seq: int | None = None
    capture_generation: int | None = None
    timestamp_ms: int | None = None
    timestamp_base: TimestampBase | None = None
    ros_stamp_ns: int | None = None
    received_monotonic_ns: int | None = None
    calibration_id: str | None = None
    detector_config_id: str | None = None
    method: str | None = None
    model_id: str | None = None
    generation: int | None = None
    job_seq: int | None = None
    observation: PlanarObservation | None = None
    display: DisplayResult | None = None
    display_only: bool | None = None
    reason: str | None = None
    exit_code: int | None = None
    stop_reason: str | None = None
    tracks: TrackFrame | None = None   # RESULT만: 같은 작업의 추적 결과(추적 켬일 때), 영상 참조 없음


@dataclass(frozen=True)
class HandoffView:
    """영상 칸 항목: 같은 응답의 프레임(읽기 전용 복사)과 표시 결과 쌍. 원본 배열에 쓰지 않는다."""

    msg_seq: int
    frame: object
    display: DisplayResult | None
    generation: int
    job_seq: int
    tracks: TrackFrame | None = None   # 같은 RESULT의 TrackFrame
    # 실행기 tracks가 없을 때 그 작업의 W 결과를 그릴 때 읽는 함수(기다리지 않음).
    # 그 작업이 아닌 TrackFrame은 `job_mismatch`로 바꿔 돌려준다
    track_lookup: Callable[[], object] | None = None


@dataclass(frozen=True)
class HandoffStats:
    """계수 스냅샷. 연결부 수명 전체 계수와 현재 실행(`events_*`) 계수를 섞지 않는다.

    보존식: published = taken + pending, events_enqueued + events_unsent = `records_seen`.
    """

    capacity: int
    published: int
    taken: int
    pending: int
    rejected: int
    views_put: int
    views_taken: int
    views_replaced: int
    views_cleared: int
    view_pending: int
    run_id: str | None
    records_seen: int
    events_enqueued: int
    events_unsent: int
    events_applied: int
    events_ignored: int
    events_enqueued_total: int
    results_offered: int
    results_enqueued: int
    results_rejected: int
    notify_pending: bool
    notify_calls: int


class _ExternalClearRequest:
    """A CLEAR requested by a control thread and applied by the producer thread."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        self.done = threading.Event()
        self.cancelled = threading.Event()
        self.result: int | None = None
        self.error: BaseException | None = None


def _method_of(pipeline: VisionPipeline) -> tuple[str, str | None]:
    """현재 설정의 검출 방식과 model_id(OpenCV면 None)."""
    config = pipeline.settings.detector_config

    if isinstance(config, AiSegConfig):
        return "ai", config.model_id

    return "opencv", None


def annotate(view: HandoffView) -> np.ndarray:
    """영상 칸 항목의 주석 영상. 수신 측이 꺼낸 뒤 잠금 밖에서 부르며 원본은 바꾸지 않는다.

    AI 표시 결과가 아니면 오버레이 없는 복사본을 돌려준다. 공개 직전 `is_current` 재확인은 호출자 몫이다.
    """
    if not isinstance(view, HandoffView):
        raise HandoffContractError(f"annotate needs HandoffView: {type(view).__name__}")

    image = view.frame.image

    if not isinstance(view.display, AiSegResult):
        return image.copy()

    if view.tracks is not None or view.track_lookup is None:
        return draw_detections(image, view.display, view.tracks)[0]

    # 그릴 때 같은 작업의 W 결과를 한 번 읽는다(기다리지 않음). 없거나 사유면 그 상태만 표기한다
    value = view.track_lookup()

    # 같은 작업 결과가 없어 직전 W 결과를 위치로 짝지은 표시 전용 번호(배너 `(world, prev)`로 구분)
    if isinstance(value, PrevDisplay):
        return draw_detections(image, view.display, value.tracks, mode_text=f"{TrackMode.WORLD.value}, prev")[0]

    if isinstance(value, TrackFrame):
        if {a.instance_index for a in value.assignments} == {i.index for i in view.display.instances}:
            return draw_detections(image, view.display, value)[0]

        value = "instances_mismatch"

    canvas = draw_detections(image, view.display)[0]
    draw_track_unavailable(canvas, TrackMode.WORLD.value, "pending" if value is None else str(value))

    return canvas


def _bound_track_lookup(source: Callable[[object, object], object], context, display) -> Callable[[], object]:
    """W 조회 함수를 그 작업 문맥·AI 결과에 묶는다. 그 작업이 아닌 TrackFrame은 쓰지 않는다."""

    def lookup() -> object:
        value = source(context, display)
        frame_tracks = value.tracks if isinstance(value, PrevDisplay) else value

        if isinstance(frame_tracks, TrackFrame) and not matches_job(frame_tracks, context):
            return "job_mismatch"

        return value

    return lookup


class AToCHandoff:
    """A→C 내부 연결부. 제어 FIFO·영상 1칸·현재 결과 거울·상태를 한 잠금으로 바꾼다."""

    def __init__(self, *, capacity: int = DEFAULT_CAPACITY, notify: Callable[[], None] | None = None,
                 clock_ns: Callable[[], int] = time.monotonic_ns) -> None:
        """용량(양의 정수, bool 제외)·선택 알림·PC 단조 시계를 받는다. 어긋나면 HandoffContractError."""
        if type(capacity) is not int or capacity <= 0:
            raise HandoffContractError(f"capacity must be a positive int: {capacity!r}")

        if notify is not None and not callable(notify):
            raise HandoffContractError("notify must be callable or None")

        if not callable(clock_ns):
            raise HandoffContractError("clock_ns must be callable")

        self._capacity = capacity
        self._notify = notify
        # W 표시 조회(start_run 전에만 설정)
        self._track_source: Callable[[object, object], object] | None = None
        self._clock_ns = clock_ns
        self._lock = threading.Lock()

        # 상태 latch
        self._state = HandoffState.OPEN
        self._reason: str | None = None

        # 제어 대기열·영상 칸·현재 결과 거울 (msg_seq, generation, job_seq)
        self._queue: deque[HandoffMessage] = deque()
        self._view: HandoffView | None = None
        self._current: tuple[int, int, int] | None = None
        self._next_seq = 1

        # 생산자 스레드·실행 결합(코어·실행 ID). 사건 커서는 현재 실행의 _events_enqueued다
        self._producer: int | None = None
        self._pipeline: VisionPipeline | None = None
        self._run_id: str | None = None
        self._records_seen = 0
        self._external_clear_requests: queue.Queue[_ExternalClearRequest] = queue.Queue()

        # 알림
        self._notify_pending = False
        self._notify_calls = 0

        # 계수
        self._published = self._taken = self._rejected = 0
        self._views_put = self._views_taken = self._views_replaced = self._views_cleared = 0
        self._events_enqueued = self._events_applied = self._events_ignored = 0   # 현재 실행
        self._events_enqueued_total = 0                                          # 수명 전체
        self._results_offered = self._results_enqueued = self._results_rejected = 0

    # ------------------------------------------------------------------ 생산자(A 제어 흐름)

    def set_track_source(self, source: Callable[[object, object], object] | None) -> None:
        """W 추적 결과 조회 함수(예: `LiveWorldTracking.peek`)를 둔다. 첫 `start_run` 전에만 허용한다.

        `source(JobContext, AI 표시 결과)`는 기다리지 않아야 하며, 실행기 `tracks`가 있는 응답에는 쓰지 않는다.
        """
        if source is not None and not callable(source):
            raise HandoffContractError("track source must be callable or None")

        with self._lock:
            if self._pipeline is not None:
                raise HandoffContractError("track source must be set before start_run")

            self._track_source = source

    def start_run(self, pipeline: VisionPipeline) -> int:
        """코어를 결합하고 현재 실행 사건 계수를 0으로 둔 뒤 RUN_START(전체 해제)를 넣는다. msg_seq를 돌려준다.

        이전 실행 사건은 다시 보내지 않으며, 대기열에 남은 이전 실행 메시지는 순서대로 남는다.
        """
        self._enter_producer()

        if not isinstance(pipeline, VisionPipeline):
            raise HandoffContractError(f"start_run needs VisionPipeline: {type(pipeline).__name__}")

        method, model_id = _method_of(pipeline)
        seen = len(pipeline.invalidation_records)

        with self._lock:
            self._raise_if_not_open()
            self._pipeline = pipeline
            self._run_id = pipeline.run_id
            self._records_seen = seen
            self._events_enqueued = self._events_applied = self._events_ignored = 0
            msg = self._message(MsgKind.RUN_START, detector_config_id=pipeline.settings.detector_config_id,
                                method=method, model_id=model_id)
            notify = self._enqueue(msg, clear=True)

        self._call_notify(notify)

        return msg.msg_seq

    def sync_invalidations(self, pipeline: VisionPipeline) -> int:
        """코어 `invalidation_records`의 새 항목을 순서대로 INVALIDATION으로 넣는다. 넣은 수를 돌려준다.

        각 기록은 한 번만 넘긴다. 넣지 못하면 커서가 그 항목에 멈추고 나머지가 `events_unsent`가 된다.
        """
        self._enter_producer()
        self._require_bound(pipeline)
        records = pipeline.invalidation_records
        sent = 0

        with self._lock:
            self._records_seen = len(records)
            start = self._events_enqueued

        for record in records[start:]:
            with self._lock:
                self._raise_if_not_open()
                msg = self._message(MsgKind.INVALIDATION, event=record.event, applied=record.applied)
                notify = self._enqueue(msg, clear=False)
                self._events_enqueued += 1
                self._events_enqueued_total += 1

                if record.applied:
                    self._events_applied += 1
                    self._apply_scope(record.event)
                else:
                    self._events_ignored += 1

            self._call_notify(notify)
            sent += 1

        return sent

    def note_input_status(self, event: InputStatusEvent) -> int:
        """실행기가 코어 `note_input_status`를 부른 직후 INPUT_STATUS(전체 해제)를 넣는다."""
        self._enter_producer()
        pipeline = self._require_bound(None)

        if not isinstance(event, InputStatusEvent):
            raise HandoffContractError(f"note_input_status needs InputStatusEvent: {type(event).__name__}")

        if event.run_id != pipeline.run_id:
            raise HandoffContractError("input status event from another run")

        with self._lock:
            self._raise_if_not_open()
            msg = self._message(MsgKind.INPUT_STATUS, source_id=event.source_id, status=event.status,
                                detail=event.detail, event_monotonic_ns=event.event_monotonic_ns)
            notify = self._enqueue(msg, clear=True)

        self._call_notify(notify)

        return msg.msg_seq

    def offer_result(self, outcome: JobOutcome, *, tracks: TrackFrame | None = None) -> int:
        """현재 반영을 마친 응답을 RESULT로 넣고 영상 칸을 같은 응답의 프레임으로 교체한다. msg_seq를 돌려준다.

        결합한 코어의 현재 관측이 아니면 HandoffContractError. FAILED·CLOSED면 넣지 않고 `results_rejected`.
        """
        self._enter_producer()
        pipeline = self._require_bound(None)

        if not isinstance(outcome, JobOutcome):
            raise HandoffContractError(f"offer_result needs JobOutcome: {type(outcome).__name__}")

        if (outcome.status is not JobStatus.COMPLETED or not outcome.applied or outcome.duplicate
                or outcome.observation is None):
            raise HandoffContractError("offer_result needs the first applied COMPLETED outcome")

        if pipeline.current_observation is not outcome.observation:
            raise HandoffContractError("outcome is not the current core observation")

        if tracks is not None and not matches_job(tracks, outcome.context):
            raise HandoffContractError("tracks do not belong to this outcome's job")

        ctx = outcome.context
        f = ctx.frame
        obs = outcome.observation

        with self._lock:
            self._results_offered += 1

            try:
                self._raise_if_not_open()
                msg = self._message(
                    MsgKind.RESULT, source_id=f.source_id, run_id=f.run_id, image_seq=f.image_seq,
                    capture_generation=f.capture_generation, timestamp_ms=f.timestamp_ms,
                    timestamp_base=f.timestamp_base, ros_stamp_ns=f.ros_stamp_ns,
                    received_monotonic_ns=f.received_monotonic_ns, calibration_id=f.calibration_id,
                    detector_config_id=ctx.detector_config_id, generation=ctx.generation, job_seq=ctx.job_seq,
                    observation=obs, display=outcome.display_result, display_only=not obs.valid, tracks=tracks)
                notify = self._enqueue(msg, clear=False)
            except HandoffError:
                self._results_rejected += 1
                raise

            self._results_enqueued += 1
            self._current = (msg.msg_seq, ctx.generation, ctx.job_seq)
            lookup = (None if tracks is not None or self._track_source is None
                      else _bound_track_lookup(self._track_source, ctx, outcome.display_result))
            self._put_view(HandoffView(msg.msg_seq, f, outcome.display_result, ctx.generation, ctx.job_seq, tracks,
                                       track_lookup=lookup))

        self._call_notify(notify)

        return msg.msg_seq

    def clear(self, reason: str) -> int:
        """CLEAR(전체 해제)를 넣는다. 예: 코어가 현재 관측을 바꿨지만 결과를 공개하지 않은 경우."""
        self._enter_producer()
        self._require_bound(None)
        return self._clear_on_producer(reason)

    def clear_from_control_thread(self, reason: str, timeout_s: float = 7.0) -> int:
        """Request CLEAR from a control thread; the producer applies it at its next check()."""
        if not isinstance(reason, str) or not reason:
            raise HandoffContractError(f"reason must be a non-empty string: {reason!r}")
        if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
                or not 0 < timeout_s < float("inf")):
            raise HandoffContractError(f"timeout_s must be positive: {timeout_s!r}")
        if self._producer == threading.get_ident():
            return self.clear(reason)

        request = _ExternalClearRequest(reason)
        self._external_clear_requests.put(request)
        if not request.done.wait(float(timeout_s)):
            request.cancelled.set()
            raise TimeoutError("producer thread did not apply requested CLEAR in time")
        if request.error is not None:
            raise request.error
        assert request.result is not None
        return request.result

    def _clear_on_producer(self, reason: str) -> int:
        self._require_bound(None)
        if not isinstance(reason, str) or not reason:
            raise HandoffContractError(f"reason must be a non-empty string: {reason!r}")
        with self._lock:
            self._raise_if_not_open()
            msg = self._message(MsgKind.CLEAR, reason=reason)
            notify = self._enqueue(msg, clear=True)

        self._call_notify(notify)

        return msg.msg_seq

    def end_run(self, exit_code: int, stop_reason: str) -> int:
        """RUN_END(전체 해제)를 넣는다. 연결부는 닫지 않는다(수명은 통합 쪽)."""
        self._enter_producer()
        self._require_bound(None)

        if type(exit_code) is not int:
            raise HandoffContractError(f"exit_code must be int: {exit_code!r}")

        with self._lock:
            self._raise_if_not_open()
            msg = self._message(MsgKind.RUN_END, exit_code=exit_code, stop_reason=stop_reason)
            notify = self._enqueue(msg, clear=True)

        self._call_notify(notify)

        return msg.msg_seq

    def check(self) -> None:
        """대기 지점의 실패 확인. OPEN이면 무동작, FAILED면 HandoffFailed, CLOSED면 HandoffClosed."""
        self._enter_producer()

        while True:
            try:
                request = self._external_clear_requests.get_nowait()
            except queue.Empty:
                break
            if request.cancelled.is_set():
                request.done.set()
                continue
            try:
                request.result = self._clear_on_producer(request.reason)
            except BaseException as exc:
                request.error = exc
            finally:
                request.done.set()

        with self._lock:
            self._raise_if_not_open(count=False)

    # ------------------------------------------------------------------ 소비자(수신 측)

    def take_control(self, max_n: int) -> tuple[tuple[HandoffMessage, ...], bool]:
        """순서대로 최대 max_n개를 꺼낸다(대기 없음). (메시지들, 꺼낸 뒤 대기열·영상 칸이 모두 비었는지)."""
        if type(max_n) is not int or max_n <= 0:
            raise HandoffContractError(f"max_n must be a positive int: {max_n!r}")

        with self._lock:
            out = tuple(self._queue.popleft() for _ in range(min(max_n, len(self._queue))))
            self._taken += len(out)

            return out, self._after_take()

    def take_view(self) -> tuple[HandoffView | None, bool]:
        """영상 칸을 꺼내고 비운다. (항목 또는 None, 꺼낸 뒤 모두 비었는지)."""
        with self._lock:
            item, self._view = self._view, None

            if item is not None:
                self._views_taken += 1

            return item, self._after_take()

    def is_current(self, msg_seq: int) -> bool:
        """그 RESULT가 아직 현재인지: 상태 OPEN이고 현재 결과 거울의 msg_seq와 같을 때만 True."""
        with self._lock:
            return (self._state is HandoffState.OPEN and self._current is not None
                    and self._current[0] == msg_seq)

    @property
    def current_result(self) -> tuple[int, int, int] | None:
        """현재 결과 거울 (msg_seq, generation, job_seq) 또는 None."""
        with self._lock:
            return self._current

    def state(self) -> HandoffStatus:
        """상태와 첫 실패 사유."""
        with self._lock:
            return HandoffStatus(self._state, self._reason)

    def report_consumer_failure(self, reason: str) -> None:
        """수신 측 실패를 latch한다. 생산자는 다음 생산자 호출이나 `check()`에서 알게 된다."""
        with self._lock:
            self._fail(f"consumer: {reason}")

    def stats(self) -> HandoffStats:
        """계수 스냅샷. 생산자 경계에서 갱신한 수치만 읽고 코어를 읽지 않는다."""
        with self._lock:
            return HandoffStats(
                capacity=self._capacity, published=self._published, taken=self._taken, pending=len(self._queue),
                rejected=self._rejected, views_put=self._views_put, views_taken=self._views_taken,
                views_replaced=self._views_replaced, views_cleared=self._views_cleared,
                view_pending=0 if self._view is None else 1, run_id=self._run_id, records_seen=self._records_seen,
                events_enqueued=self._events_enqueued, events_unsent=self._records_seen - self._events_enqueued,
                events_applied=self._events_applied, events_ignored=self._events_ignored,
                events_enqueued_total=self._events_enqueued_total,
                results_offered=self._results_offered, results_enqueued=self._results_enqueued,
                results_rejected=self._results_rejected, notify_pending=self._notify_pending,
                notify_calls=self._notify_calls)

    def close(self) -> None:
        """CLOSED로 latch한다(FAILED 사유 유지). 재호출 멱등이며 남은 메시지는 꺼낼 수 있다."""
        with self._lock:
            if self._state is HandoffState.CLOSED:
                return

            self._state = HandoffState.CLOSED
            self._drop_current()

    # ------------------------------------------------------------------ 내부(잠금 안에서 부름: 이름 끝 설명 참조)

    def _enter_producer(self) -> None:
        """생산자 스레드 고정: 첫 호출 스레드를 기억하고 다른 스레드면 계약 오류."""
        ident = threading.get_ident()

        with self._lock:
            if self._producer is None:
                self._producer = ident
            elif self._producer != ident:
                raise HandoffContractError("producer methods must be called from one thread")

    def _require_bound(self, pipeline: VisionPipeline | None) -> VisionPipeline:
        """start_run으로 결합한 코어를 돌려준다. 결합 전이거나(인자를 주면) 다른 코어면 계약 오류."""
        with self._lock:
            bound = self._pipeline

        if bound is None:
            raise HandoffContractError("start_run has not been called")

        if pipeline is not None and pipeline is not bound:
            raise HandoffContractError("pipeline is not the one bound by start_run")

        return bound

    def _raise_if_not_open(self, *, count: bool = True) -> None:
        """(잠금 안) OPEN이 아니면 거부 계수를 올리고 해당 예외."""
        if self._state is HandoffState.OPEN:
            return

        if count:
            self._rejected += 1

        if self._state is HandoffState.FAILED:
            raise HandoffFailed(self._reason)

        raise HandoffClosed(self._reason or "closed")

    def _message(self, kind: MsgKind, **values) -> HandoffMessage:
        """(잠금 안) 바깥 공통 필드를 채운 메시지. msg_seq는 넣을 때 확정한다(여기서는 다음 번호)."""
        p = self._pipeline
        base = {"msg_seq": self._next_seq, "msg_kind": kind, "run_id": p.run_id, "source_id": p.settings.source_id,
                "core_generation": p.current_generation(), "published_monotonic_ns": self._clock_ns()}
        base.update(values)

        return HandoffMessage(**base)

    def _enqueue(self, msg: HandoffMessage, *, clear: bool) -> bool:
        """(잠금 안) 넣는다. 가득 차면 FAILED(overflow) latch 뒤 HandoffFailed. clear면 현재 결과·영상 칸 해제.
        알림이 필요하면 True(호출은 잠금 밖)."""
        if len(self._queue) >= self._capacity:
            self._rejected += 1
            self._fail("overflow")
            raise HandoffFailed("overflow")

        self._queue.append(msg)
        self._next_seq += 1
        self._published += 1

        if clear:
            self._drop_current()

        if self._notify is None or self._notify_pending:
            return False

        self._notify_pending = True

        return True

    def _call_notify(self, needed: bool) -> None:
        """(잠금 밖) 알림을 부른다. 예외면 FAILED("notify: …") latch 뒤 HandoffFailed."""
        if not needed:
            return

        try:
            self._notify()
        except Exception as exc:  # noqa: BLE001 — 수신 측 알림 실패를 상태로 남긴다
            with self._lock:
                self._fail(f"notify: {type(exc).__name__}: {exc}")

            raise HandoffFailed(f"notify: {exc}") from exc

        with self._lock:
            self._notify_calls += 1

    def _apply_scope(self, event: CandidateInvalidationEvent) -> None:
        """(잠금 안) applied 사건의 무효 범위를 현재 결과 거울에 적용한다."""
        if self._current is None:
            return

        _, gen, job = self._current

        if event.kind is InvalidationKind.GENERATION_CHANGED:
            hit = gen < event.generation
        elif event.job_seq is None:
            hit = gen == event.generation
        else:
            hit = gen == event.generation and job <= event.job_seq

        if hit:
            self._drop_current()

    def _put_view(self, item: HandoffView) -> None:
        """(잠금 안) 영상 칸을 교체한다."""
        if self._view is not None:
            self._views_replaced += 1

        self._view = item
        self._views_put += 1

    def _drop_current(self) -> None:
        """(잠금 안) 현재 결과 거울과 영상 칸을 비운다."""
        self._current = None

        if self._view is not None:
            self._view = None
            self._views_cleared += 1

    def _fail(self, reason: str) -> None:
        """(잠금 안) FAILED latch(첫 사유 유지, CLOSED면 상태 유지). 현재 결과·영상 칸을 비운다."""
        if self._state is HandoffState.OPEN:
            self._state = HandoffState.FAILED
            self._reason = reason

        self._drop_current()

    def _after_take(self) -> bool:
        """(잠금 안) 모두 비었으면 알림 대기를 내리고 True."""
        drained = not self._queue and self._view is None

        if drained:
            self._notify_pending = False

        return drained
