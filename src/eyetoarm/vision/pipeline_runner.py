"""저장 이미지·영상·실시간 입력 실행기: 입력 → 작업 시작 → 검출 → terminal → 기록 → 정리.

감시를 켜면 detector를 worker 스레드 1개에서 돌리고, 코어·기록·B 지점 호출은 제어 흐름만 한다.
종료 코드: 0 정상, 1 실행 실패 또는 처리 실패·입력 오류·시간 초과, 2 인자·설정 오류(CLI).
"""
from __future__ import annotations

import collections
import dataclasses
import math
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable

from eyetoarm.common.types import (CandidateInvalidationEvent, FramePacket, InputStatus, InputStatusEvent,
                                   InvalidationKind, PlanarObservation)
from eyetoarm.inputs.base import FrameIdIssuer, new_run_id
from eyetoarm.inputs.live_slot import BoundaryKind, LiveInputOverflow, NoticeKind
from eyetoarm.vision.ai_convert import list_candidate_indices
from eyetoarm.vision.ai_result import AiJobConfigError, AiSegResult, ModelPrepareError
from eyetoarm.vision.handoff import HandoffClosed, HandoffError, HandoffFailed, HandoffState
from eyetoarm.vision.pipeline import GenerationCause, VisionPipeline
from eyetoarm.vision.pipeline_opencv import process_opencv
from eyetoarm.vision.pipeline_records import InputStateTracker, PipelineLog
from eyetoarm.vision.tracking import FrameKey, PartTracker, inputs_from_ai
from eyetoarm.vision.pipeline_types import (JobContext, JobOutcome, JobStatus, PipelineCounters, PipelineSettings,
                                            SourceProfile)

# 저장 입력 소스가 「더 읽을 것이 없음」으로 돌려주는 사건 설명
_NO_INPUT_DETAILS = ("empty list", "closed")


class JobCancelled(Exception):
    """detector가 실행기에 취소를 알릴 때 올린다. 그 작업은 `fail(JOB_ABORTED)`로 정리되고 실행은 계속된다."""


class CancelToken:
    """협조 취소 표지. 실행기가 세우고 detector는 읽기만 한다(확인해야만 효과가 있음)."""

    __slots__ = ("_event",)

    def __init__(self) -> None:
        """세워지지 않은 표지."""
        self._event = threading.Event()

    @property
    def cancelled(self) -> bool:
        """취소 요청이 있었는가."""
        return self._event.is_set()

    def wait(self, timeout: float | None) -> bool:
        """취소될 때까지 최대 timeout 초 기다린다. 취소됐으면 True."""
        return self._event.wait(timeout)

    def _cancel(self) -> None:
        """실행기 전용: 취소를 요청한다."""
        self._event.set()


def _positive_finite(value) -> bool:
    """bool이 아닌 유한 양수(int·float)인가."""
    return (not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
            and value > 0)


@dataclass(frozen=True)
class RunnerConfig:
    """실행기 설정(A 내부). 시간 값은 개발 초기값이며 운영 최종값이 아니다.

    검출 상한 둘 다 None이면 감시 끔(동기 호출)이고 한쪽만 None은 오류다. 첫 상한은 run의 첫 작업에만 쓴다.
    """

    close_attempts: int = 3
    close_retry_interval_s: float = 0.05
    max_consecutive_input_errors: int = 8
    max_frames: int | None = None
    first_detect_timeout_s: float | None = 10.0
    detect_timeout_s: float | None = 1.0
    model_load_timeout_s: float = 10.0
    cancel_grace_s: float = 0.2
    worker_cleanup_limit_s: float = 0.3
    stop_check_interval_s: float = 0.02
    live_transition_limit: int = 10
    live_transition_window_s: float = 1.0
    stall_dump_path: str | None = None   # 멈춤 때 스레드 호출 위치를 덧붙일 파일(없으면 stderr만, D-11)

    @property
    def watchdog(self) -> bool:
        """검출 감시를 켰는가(상한이 설정됨)."""
        return self.detect_timeout_s is not None

    def __post_init__(self) -> None:
        """값 범위를 검사한다. 어긋나면 ValueError."""
        first, normal = self.first_detect_timeout_s, self.detect_timeout_s

        if (first is None) != (normal is None):
            raise ValueError("first_detect_timeout_s and detect_timeout_s must both be None (watchdog off) "
                             f"or both be set: {first!r}, {normal!r}")

        for name in ("first_detect_timeout_s", "detect_timeout_s"):
            value = getattr(self, name)

            if value is not None and not _positive_finite(value):
                raise ValueError(f"{name} must be None or a finite number > 0: {value!r}")

        for name in ("model_load_timeout_s", "cancel_grace_s", "worker_cleanup_limit_s", "stop_check_interval_s",
                     "live_transition_window_s"):
            if not _positive_finite(getattr(self, name)):
                raise ValueError(f"{name} must be a finite number > 0: {getattr(self, name)!r}")

        for name in ("close_attempts", "max_consecutive_input_errors", "live_transition_limit"):
            value = getattr(self, name)

            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be an int >= 1: {value!r}")

        interval = self.close_retry_interval_s

        if isinstance(interval, bool) or not isinstance(interval, (int, float)) or not math.isfinite(interval) \
                or interval < 0:
            raise ValueError(f"close_retry_interval_s must be a finite number >= 0: {interval!r}")

        if self.max_frames is not None and (type(self.max_frames) is not int or self.max_frames < 1):
            raise ValueError(f"max_frames must be None or an int >= 1: {self.max_frames!r}")


@dataclass
class RunResult:
    """실행 결과 요약(시험·CLI 보고용)."""

    exit_code: int
    stop_reason: str
    run_id: str
    frames: int = 0
    input_errors: int = 0
    b_calls: int = 0
    b_list_calls: int = 0        # 단계 L: 목록 소비자 호출 수(b_list_consumer=None이면 0)
    counters: PipelineCounters | None = None
    closed: bool = False
    missing: dict | None = None  # 기록이 확인되지 않은 처리·사건·폐기 수(정상 기록이면 모두 0)
    errors: list[str] = field(default_factory=list)
    timeouts: int = 0            # 검출 시간 초과 수(실패로 셈, 코어 aborted에도 들어감)
    stalled: bool = False        # 유예·정리 한도 안에 검출 호출이 반환하지 않음
    late_results: int = 0        # 시간 초과·정지 뒤 유예 안에 돌아와 duplicate로 버린 결과 수
    worker: dict | None = None   # worker 정리 보고(감시 끔이면 None)
    live_inputs: dict = field(default_factory=dict)  # 실시간 소스: 반영한 경계·알림 종류별 수
    input_boundary_aborts: int = 0  # 실시간 소스: 공개 직전 epoch가 현재가 아니라 JOB_ABORTED로 끝낸 작업 수(실패 아님)
    source_stats: dict | None = None  # 실시간 소스 통계(종료 때)
    handoff: dict | None = None  # A→C 연결부 종료 보고(handoff_sink=None이면 None)
    tracking: dict | None = None  # 인스턴스 추적 종료 보고(part_tracker=None이면 None)


def b_gate(pipeline: VisionPipeline, outcome: JobOutcome) -> bool:
    """B 소비 조건. 응답을 받은 그 자리에서만 부른다.

    유효 관측이 현재 관측 그 자체이고 세대·설정·소스·프로파일이 맞아야 한다. applied만으로는 B 전달 허가가 아니다.
    """
    if not _b_gate_common(pipeline, outcome) or outcome.observation.valid is not True:
        return False

    profile = pipeline.settings.profile

    return profile is not None and profile.applies_to(outcome.context.frame)


def _b_gate_common(pipeline: VisionPipeline, outcome: JobOutcome) -> bool:
    """`b_gate`·`b_list_gate` 공통 조건(유효 여부·프로필 제외): 현재 관측 그 자체이고 식별자가 작업과 같음."""
    obs = outcome.observation

    if outcome.duplicate or outcome.status is not JobStatus.COMPLETED or not outcome.applied:
        return False

    if not isinstance(obs, PlanarObservation) or pipeline.closed:
        return False

    ctx = outcome.context
    settings = pipeline.settings

    return (pipeline.current_observation is obs
            and ctx.generation == pipeline.current_generation()
            and ctx.detector_config_id == settings.detector_config_id
            and ctx.frame.source_id == settings.source_id
            and (obs.job_seq, obs.image_seq, obs.generation) == (ctx.job_seq, ctx.frame.image_seq, ctx.generation))


def b_list_gate(pipeline: VisionPipeline, outcome: JobOutcome) -> bool:
    """단계 L 목록 소비 조건. 공통 조건에 더해 AI 결과·프로필 적용·목록 후보 있음을 본다."""
    if not _b_gate_common(pipeline, outcome):
        return False

    display = outcome.display_result

    if not isinstance(display, AiSegResult):
        return False

    return bool(list_candidate_indices(display, outcome.context.frame, pipeline.settings.profile))


def b_admit(pipeline: VisionPipeline, outcome: JobOutcome, live_source=None) -> bool:
    """B 지점 호출 조건: `b_gate` 그리고 실시간 소스면 그 프레임 수신 epoch가 현재여야 한다."""
    if not b_gate(pipeline, outcome):
        return False

    return live_source is None or bool(live_source.frame_current(outcome.context.frame))


def b_skip_reason(pipeline: VisionPipeline, outcome: JobOutcome, live_source=None) -> str | None:
    """`b_admit`가 거부한 이유(통과면 None). 판정은 `b_admit`과 같다."""
    if b_admit(pipeline, outcome, live_source):
        return None

    obs = outcome.observation
    ctx = outcome.context

    if outcome.duplicate:
        return "duplicate"

    if outcome.status is JobStatus.DISCARDED_GENERATION:
        return "generation_discarded"

    if outcome.status is not JobStatus.COMPLETED or not outcome.applied:
        return f"job_{outcome.status.value.lower()}"

    if not isinstance(obs, PlanarObservation) or obs.valid is not True:
        return "invalid_observation"

    if pipeline.closed:
        return "pipeline_closed"

    if ctx.generation != pipeline.current_generation():
        return "generation_discarded"

    profile = pipeline.settings.profile

    if profile is None or not profile.applies_to(ctx.frame):
        return "calibration_mismatch"

    if pipeline.current_observation is not obs:
        return "not_current_observation"

    if live_source is not None and not live_source.frame_current(ctx.frame):
        return "rx_epoch_stale"

    return "b_gate_rejected"


def b_list_skip_reason(pipeline: VisionPipeline, outcome: JobOutcome, live_source=None) -> str | None:
    """단계 L 목록 경로의 건너뜀 사유(통과면 None). 단계 T 유효 여부는 보지 않는다."""
    if b_list_gate(pipeline, outcome) and (live_source is None or bool(live_source.frame_current(outcome.context.frame))):
        return None

    obs = outcome.observation
    ctx = outcome.context

    if outcome.duplicate:
        return "duplicate"

    if outcome.status is JobStatus.DISCARDED_GENERATION:
        return "generation_discarded"

    if outcome.status is not JobStatus.COMPLETED or not outcome.applied:
        return f"job_{outcome.status.value.lower()}"

    if not isinstance(obs, PlanarObservation):
        return "invalid_observation"

    if pipeline.closed:
        return "pipeline_closed"

    if ctx.generation != pipeline.current_generation():
        return "generation_discarded"

    if pipeline.current_observation is not obs:
        return "not_current_observation"

    if not isinstance(outcome.display_result, AiSegResult):
        return "not_ai_result"

    profile = pipeline.settings.profile

    if profile is None or not profile.applies_to(ctx.frame):
        return "calibration_mismatch"

    if not list_candidate_indices(outcome.display_result, ctx.frame, profile):
        return "no_list_candidates"

    if live_source is not None and not live_source.frame_current(ctx.frame):
        return "rx_epoch_stale"

    return "b_list_gate_rejected"


def display_gate(pipeline: VisionPipeline, outcome: JobOutcome) -> bool:
    """표시 조건. 최초 현재 적용 결과만 그리며 유효 여부·프로필은 보지 않는다(B 전달 허가 아님)."""
    obs = outcome.observation

    if outcome.duplicate or outcome.status is not JobStatus.COMPLETED or not outcome.applied:
        return False

    if not isinstance(obs, PlanarObservation) or pipeline.closed:
        return False

    ctx = outcome.context
    settings = pipeline.settings

    return (pipeline.current_observation is obs
            and ctx.generation == pipeline.current_generation()
            and ctx.detector_config_id == settings.detector_config_id
            and ctx.frame.source_id == settings.source_id
            and (obs.job_seq, obs.image_seq, obs.generation) == (ctx.job_seq, ctx.frame.image_seq, ctx.generation))


def apply_seek(pipeline: VisionPipeline, source, frame_index: int) -> bool:
    """탐색을 시도한다. 성공하면 세대를 올리고(SEEK) True, 실패하면 상태를 그대로 두고 False."""
    try:
        source.seek(frame_index)
    except (ValueError, RuntimeError, TypeError):
        return False

    pipeline.change_settings(pipeline.settings, cause=GenerationCause.SEEK)

    return True


class _StopRun(Exception):
    """실행 수준 실패를 이미 기록하고 루프를 끝낼 때 올린다(실시간 경계 반복 등)."""


class _LogFailure(Exception):
    """기록 단계에서 난 실패(원래 예외를 감쌈). 실행기는 기록을 멈추고 정리를 계속한다."""

    def __init__(self, cause: BaseException) -> None:
        """원래 예외를 받는다."""
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.cause = cause


def _logged(fn, *args, **kwargs):
    """기록 호출을 감싸 실패를 `_LogFailure`로 바꾼다."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        raise _LogFailure(exc) from exc


_STOP = object()       # worker 수신 칸 종료 표지


class _Call:
    """worker 수신 칸에 넣는 작업 아닌 호출(모델 준비)."""

    __slots__ = ("fn",)

    def __init__(self, fn) -> None:
        """호출할 함수."""
        self.fn = fn

_NO_RESULT = object()  # 유예 안에 반환을 확인하지 못함


class _DetectorWorker:
    """검출 전용 상주 worker 스레드 1개. 수신 칸·핸드오프 칸은 각 용량 1.

    `busy`는 제어 흐름 소유이며, 결과를 `take`로 받아야 풀려 작업이 겹치지 않는다.
    받은 `JobContext`를 복제 없이 돌려준다. 데몬 스레드지만 멈춘 스레드의 자원 회수는 보장하지 않는다.
    """

    def __init__(self, detector, takes_token: bool) -> None:
        """스레드를 만들고 시작한다."""
        self._detector, self._takes_token = detector, takes_token
        self._inbox: queue.Queue = queue.Queue(maxsize=1)
        self._outbox: queue.Queue = queue.Queue(maxsize=1)
        self._stopping = threading.Event()  # 종료 의도(수신 칸 Full과 무관하게 보존)
        self.job: JobContext | None = None
        self.call_name: str | None = None   # 작업이 아닌 호출(모델 준비) 이름. JobContext를 위조하지 않는다
        self.token: CancelToken | None = None
        self.submitted_monotonic: float | None = None
        self._thread = threading.Thread(target=self._loop, name="eyetoarm-detector", daemon=True)
        self._thread.start()

    @property
    def busy(self) -> bool:
        """넣은 작업·호출의 반환을 아직 받지 못했는가."""
        return self.job is not None or self.call_name is not None

    def submit(self, job: JobContext, token: CancelToken) -> None:
        """작업 하나를 넘긴다. 이전 호출이 반환하지 않았으면 계약 오류(겹침 금지)."""
        if self.busy:
            raise RuntimeError(f"detector worker busy with {self.describe()}")

        self.job, self.token, self.submitted_monotonic = job, token, time.monotonic()
        self._inbox.put_nowait((job, token))

    def submit_call(self, name: str, fn: Callable[[], object], token: CancelToken) -> None:
        """작업이 아닌 호출(모델 준비) 하나를 같은 칸으로 넘긴다. 결과는 `(None, 값, 예외)`로 핸드오프된다."""
        if self.busy:
            raise RuntimeError(f"detector worker busy with {self.describe()}")

        self.call_name, self.token, self.submitted_monotonic = name, token, time.monotonic()
        self._inbox.put_nowait(_Call(fn))

    def describe(self) -> str:
        """지금 붙잡힌 일(보고용): "job_seq=N" 또는 호출 이름."""
        return f"job_seq={self.job.job_seq}" if self.job is not None else str(self.call_name)

    def take(self, timeout: float):
        """핸드오프 칸에서 결과 하나를 최대 timeout 초 기다려 꺼낸다. 없으면 queue.Empty. 꺼내면 IDLE."""
        item = self._outbox.get(timeout=max(0.0, timeout))
        self.job = self.call_name = self.token = self.submitted_monotonic = None

        return item

    def shutdown(self, limit_s: float, interval_s: float) -> dict:
        """종료 절차: BUSY면 취소 후 한도 안 반환 대기 → 종료 표지 → 유한 join → 남은 핸드오프 버림.

        반환 대기 뒤에 종료 상태를 세운다(먼저 세우면 한도 안 정상 반환을 버림). 무한 join하지 않는다.
        """
        end = time.monotonic() + limit_s
        report = {"stalled": False, "stalled_job_seq": None, "thread_stopped": False, "discarded": 0}

        if self.busy:
            self.token._cancel()

            while self.busy and end - time.monotonic() > 0:
                try:
                    self.take(min(end - time.monotonic(), interval_s))
                    report["discarded"] += 1
                except queue.Empty:
                    pass

        if self.busy:
            report["stalled"] = True
            report["stalled_job_seq"] = None if self.job is None else self.job.job_seq
            report["stalled_call"] = self.call_name
            report["stalled_what"] = self.describe()
            report["busy_s"] = round(time.monotonic() - self.submitted_monotonic, 3)

        # 수신 칸이 차 표지를 못 넣어도 worker가 이 상태를 보고 끝나도록 먼저 세운다
        self._stopping.set()

        try:
            self._inbox.put_nowait(_STOP)  # IDLE worker를 깨운다
        except queue.Full:
            pass

        if not report["stalled"]:
            self._thread.join(max(end - time.monotonic(), interval_s))

        report["thread_stopped"] = not self._thread.is_alive()

        while True:
            try:
                self._outbox.get_nowait()
                report["discarded"] += 1
            except queue.Empty:
                break

        return report

    def _loop(self) -> None:
        """수신 칸에서 작업을 받아 detector를 부르고 결과를 핸드오프한다. 종료 표지를 받으면 끝난다."""
        while True:
            item = self._inbox.get()

            if item is _STOP or self._stopping.is_set():
                return  # 종료 뒤 꺼낸 작업은 detector를 부르지 않는다

            if isinstance(item, _Call):
                job, token, call = None, None, item.fn
            else:
                (job, token), call = item, None

            value = exc = None

            try:
                if call is not None:
                    value = call()
                else:
                    value = self._detector(job, token) if self._takes_token else self._detector(job)
            except BaseException as caught:  # 예외 분류는 제어 흐름이 한다
                exc = caught

            if self._stopping.is_set():
                return  # 정리가 끝난 뒤 반환한 결과는 소비자가 없으므로 남기지 않는다(영상 참조를 놓음)

            self._outbox.put_nowait((job, value, exc))

            # 다음 작업까지 IDLE로 기다리는 동안 끝난 작업의 영상을 붙잡지 않는다
            item = job = token = call = value = exc = None


def _wait_for_detection(take, should_stop, clock, deadline: float, interval_s: float, on_idle=None):
    """검출 대기 루프. 매 회차 결과 → (입력 반영) → 정지 요청 → 기한 순서로 하나만 판정한다.

    반환: ("result", 값) | ("stop", None) | ("timeout", None).
    """
    while True:
        remaining = deadline - clock()

        try:
            return "result", take(max(0.0, min(remaining, interval_s)))
        except queue.Empty:
            pass

        if on_idle is not None:
            on_idle()

        if should_stop is not None and should_stop():
            return "stop", None

        if deadline - clock() <= 0:
            return "timeout", None


def _await_return(worker: _DetectorWorker, limit_s: float, interval_s: float):
    """유예 단계: 실제 단조 시계로 최대 limit_s 동안 worker 반환을 기다린다. 못 받으면 `_NO_RESULT`."""
    end = time.monotonic() + limit_s

    while True:
        try:
            return worker.take(min(end - time.monotonic(), interval_s))
        except queue.Empty:
            if end - time.monotonic() <= 0:
                return _NO_RESULT


def thread_dump(where: str) -> str:
    """지금 모든 스레드의 이름·번호와 호출 위치(가장 안쪽이 마지막). 멈춤 원인을 찾는 진단용(D-11)."""
    frames = sys._current_frames()
    lines = [f"=== thread dump ({where}) monotonic={time.monotonic():.3f} threads={len(frames)} ==="]
    names = {t.ident: t for t in threading.enumerate()}

    for ident, frame in sorted(frames.items()):
        t = names.get(ident)
        label = "unknown" if t is None else f"{t.name}{' daemon' if t.daemon else ''}"
        lines.append(f"--- thread {label} ident={ident} native_id={getattr(t, 'native_id', None)}")
        lines.extend(line.rstrip("\n") for line in traceback.format_stack(frame))

    lines.append("=== end thread dump ===")

    return "\n".join(lines) + "\n"


def _write_thread_dump(err, path: str | None, where: str) -> None:
    """스레드 호출 위치를 stderr(와 설정한 파일)에 남긴다. 쓰기 실패는 실행 결과를 바꾸지 않는다."""
    try:
        text = thread_dump(where)
    except Exception as exc:   # 진단 실패는 무시하고 사유만 남긴다
        text = f"=== thread dump ({where}) failed: {type(exc).__name__}: {exc} ===\n"

    try:
        err.write(text)
        err.flush()
    except Exception:   # stderr가 닫혔어도 실행 정리를 계속한다
        pass

    if path is not None:
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(text)
        except OSError as exc:
            try:
                err.write(f"run_pipeline: cannot write thread dump to {path}: {exc}\n")
            except Exception:   # 위와 같음
                pass


def run(source_factory: Callable[[FrameIdIssuer, Callable[[], int]], object], settings: PipelineSettings,
        log: PipelineLog, *, run_id: str | None = None,
        detector: Callable[[JobContext], tuple[PlanarObservation, object]] = process_opencv,
        b_consumer: Callable[[FramePacket, PlanarObservation], None] | None = None,
        config: RunnerConfig = RunnerConfig(), stderr=None, clock_ns: Callable[[], int] = time.monotonic_ns,
        should_stop: Callable[[], bool] | None = None, versions: dict | None = None,
        detector_takes_token: bool = False, watch_clock: Callable[[], float] = time.monotonic,
        prepare: Callable[[], dict] | None = None, display_sink=None, handoff_sink=None,
        part_tracker: PartTracker | None = None, live_observation_sink=None,
        b_list_consumer: Callable[[JobContext, AiSegResult], None] | None = None,
        live_observation_b_input: str = "single") -> RunResult:
    """입력 한 번을 끝까지 처리한다. 예외를 밖으로 내지 않고 `RunResult`(종료 코드 포함)를 돌려준다.

    정리는 코어 close → worker 정리 → 남은 기록 → 소스 close 순서로 한 번씩 하며 앞 단계 실패가 뒤를 막지 않는다.
    선택 연결(`display_sink`·`handoff_sink`·`part_tracker`·`live_observation_sink`·`b_list_consumer`)은
    None이면 꺼지고, 관측·valid·기록 내용을 바꾸지 않는다. 시작 계약 위반은 프레임을 읽기 전에 종료 1로 끝낸다.
    """
    if live_observation_b_input not in ("single", "list"):
        raise ValueError(f"live_observation_b_input must be 'single' or 'list': {live_observation_b_input!r}")

    err = stderr if stderr is not None else sys.stderr
    issuer = FrameIdIssuer(run_id or new_run_id())

    # 추적기 계약: 진입점이 발급기를 주입해 만든 PartTracker만 받는다. 소스·코어를 만들기 전에 끝낸다
    if part_tracker is not None and not isinstance(part_tracker, PartTracker):
        message = f"part_tracker must be PartTracker (issuer injected by the entry point): {type(part_tracker).__name__}"
        err.write(f"run_pipeline: {message}\n")
        return RunResult(exit_code=1, stop_reason="tracker_contract", run_id=issuer.run_id, errors=[message])

    pipeline = VisionPipeline(settings, run_id=issuer.run_id, clock_ns=clock_ns)
    result = RunResult(exit_code=0, stop_reason="", run_id=issuer.run_id)
    tracker = InputStateTracker(issuer.run_id, settings.source_id)
    source = None
    log_ok = True
    worker: _DetectorWorker | None = None
    first_pending = True   # run 한 번에서 아직 첫 검출을 넘기지 않음(정지로 시작하지 않은 작업은 소모하지 않음)
    live = False           # 실시간 소스(poll_input 있음)인가. 소스를 만든 뒤 정한다
    transitions: collections.deque[int] = collections.deque()  # 실시간: 세대를 올린 경계의 PC 단조 시각(반복 감시)
    handoff_started = False  # start_run을 시도했는가(연결부 결합 여부)
    tracking_failed: list[str] = []  # 추적 실패 latch: 비어 있지 않으면 이번 실행의 추적 호출을 멈춘다

    def fail_run(reason: str, message: str) -> None:
        """실행 수준 실패를 기록한다(종료 1)."""
        result.exit_code = 1
        result.stop_reason = result.stop_reason or reason
        result.errors.append(message)

    def sync_records() -> None:
        """코어를 바꾼 뒤 남은 사건·폐기를 기록하고, 연결부가 있으면 새 사건을 넘긴다."""
        _logged(log.sync, pipeline)

        if handoff_sink is not None:
            handoff_sink.sync_invalidations(pipeline)

    def check_handoff() -> None:
        """대기 회차의 연결부 실패 확인. 실패면 HandoffFailed·HandoffClosed가 올라간다."""
        if handoff_sink is not None:
            handoff_sink.check()

    def note_status(event: InputStatusEvent) -> None:
        """입력 사건을 코어에 반영하고 연결부에 INPUT_STATUS를 넣는다."""
        pipeline.note_input_status(event)

        if handoff_sink is not None:
            handoff_sink.note_input_status(event)

    def tracking_failure(exc: BaseException, outcome: JobOutcome | None) -> None:
        """추적 예외 격리: latch를 먼저 세워 추적을 멈추고, disable 시도·stderr·`tracking_failed` 1줄을 남긴다."""
        if tracking_failed:
            return

        message = f"{type(exc).__name__}: {exc}"
        tracking_failed.append(message)
        disable_error = None

        try:
            part_tracker.disable(message)
        except Exception as dexc:   # 끄기 자체가 실패해도 latch로 추적은 이미 멈춤
            disable_error = f"{type(dexc).__name__}: {dexc}"

        ids = {} if outcome is None else {"image_seq": outcome.context.frame.image_seq,
                                           "job_seq": outcome.context.job_seq, "generation": outcome.context.generation}
        extra = {} if disable_error is None else {"disable_error": disable_error}
        result.errors.append(f"tracking_failed (tracking stopped for this run, run continues): {message}"
                             + ("" if disable_error is None else f"; disable also failed: {disable_error}"))
        _logged(log.write_tracking_failed, {"run_id": issuer.run_id, "source_id": pipeline.settings.source_id,
                                            **ids, "error": message, **extra})

    def reset_tracker(reason: str) -> None:
        """경계에서 전 트랙을 종료한다. 예외는 추적 실패로 격리한다. latch 뒤에는 부르지 않는다."""
        if part_tracker is None or tracking_failed:
            return

        try:
            part_tracker.reset(reason)
        except Exception as exc:
            tracking_failure(exc, None)

    def track(outcome: JobOutcome):
        """공개 허용된 AI 결과 하나로 추적을 1회 갱신해 TrackFrame을 돌려준다(OpenCV 결과·추적 끔·예외는 None)."""
        display = outcome.display_result

        if part_tracker is None or tracking_failed or not isinstance(display, AiSegResult):
            return None

        ctx = outcome.context
        f = ctx.frame

        try:
            key = FrameKey(run_id=f.run_id, source_id=f.source_id, image_seq=f.image_seq,
                           capture_generation=f.capture_generation, generation=ctx.generation, job_seq=ctx.job_seq,
                           detector_config_id=ctx.detector_config_id)
            return part_tracker.update(key, f.timestamp_ms, inputs_from_ai(display))
        except Exception as exc:
            tracking_failure(exc, outcome)
            return None

    def handle(outcome: JobOutcome) -> None:
        """terminal 응답을 기록·표시·전달하고, 소비 조건을 통과하면 그 자리에서 B 지점을 한 번 부른다."""
        _logged(log.log_outcome, outcome)
        sync_records()
        track_frame = None

        if display_sink is not None or handoff_sink is not None or part_tracker is not None:
            current = display_gate(pipeline, outcome) and (not live or source.frame_current(outcome.context.frame))
            track_frame = track(outcome) if current else None

            # 추적을 켠 실행만 같은 작업 TrackFrame을 기록·표시·연결부에 싣는다
            if track_frame is not None:
                _logged(log.write_ai_tracks, track_frame)

            track_kw = {} if part_tracker is None else {"tracks": track_frame}

            if display_sink is not None:
                display_sink.on_outcome(outcome, current, **track_kw)

            if handoff_sink is not None:
                if current:
                    handoff_sink.offer_result(outcome, **track_kw)
                elif outcome.status is JobStatus.COMPLETED and outcome.applied and not outcome.duplicate:
                    # 수신 epoch 경합으로 공개하지 않은 결과라 연결부 현재 결과를 비운다
                    handoff_sink.clear("result_withheld")

        if b_consumer is not None and b_admit(pipeline, outcome, source if live else None):
            result.b_calls += 1
            b_consumer(outcome.context.frame, outcome.observation)

        # 단계 L은 저장 입력에서만 허용된다(실시간은 시작 계약에서 거부)
        if b_list_consumer is not None and not live and b_list_gate(pipeline, outcome):
            result.b_list_calls += 1
            b_list_consumer(outcome.context, outcome.display_result)

        # 실시간 관측 연결에는 판정 사유와 수신 epoch만 넘기고 B는 전용 디스패처가 부른다
        if live_observation_sink is not None and live:
            if live_observation_b_input == "list":
                reason = b_list_skip_reason(pipeline, outcome, source)
            else:
                reason = b_skip_reason(pipeline, outcome, source)

            # 추적을 켠 실행만 같은 작업 TrackFrame(없으면 None)을 넘긴다
            sink_kw = {} if part_tracker is None else {"tracks": track_frame}
            live_observation_sink.offer(outcome, reason, source.frame_epoch(outcome.context.frame), **sink_kw)

    def finish_detection(job: JobContext, value, exc: BaseException | None) -> None:
        """detector 반환·예외를 terminal로 정확히 한 번 보낸다(동기·worker 경로 공통).

        KeyboardInterrupt는 JOB_ABORTED 뒤 다시 올리고, Exception이 아닌 BaseException은 그대로 올린다.
        """
        if exc is None:
            try:
                obs, display = value
            except (TypeError, ValueError) as bad:
                exc = bad
            else:
                handle(pipeline.complete(job, obs, display_result=display))
                return

        if isinstance(exc, JobCancelled):
            handle(pipeline.fail(job, InvalidationKind.JOB_ABORTED, f"cancelled: {exc}" if str(exc) else "cancelled"))
        elif isinstance(exc, AiJobConfigError):
            # 작업 중 설정 불일치(ROI·시작 설정·class_map 밖 클래스)는 MODEL_CONFIG_INVALID
            handle(pipeline.fail(job, InvalidationKind.MODEL_CONFIG_INVALID, exc.cause))
        elif isinstance(exc, KeyboardInterrupt):
            handle(pipeline.fail(job, InvalidationKind.JOB_ABORTED, "interrupted"))
            raise exc
        elif isinstance(exc, Exception):
            handle(pipeline.fail(job, InvalidationKind.INFERENCE_ERROR, f"{type(exc).__name__}: {exc}"))
        else:
            raise exc

    def note_transition(now_ns: int) -> None:
        """세대를 올린 경계를 세고, 창 안에서 상한을 넘으면 실행을 끝낸다."""
        window_ns = int(config.live_transition_window_s * 1e9)
        transitions.append(now_ns)

        while transitions and now_ns - transitions[0] > window_ns:
            transitions.popleft()

        if len(transitions) > config.live_transition_limit:
            fail_run("input_transition_storm", f"{len(transitions)} generation-changing input boundaries within "
                                               f"{config.live_transition_window_s} s (limit {config.live_transition_limit})")
            raise _StopRun()

    def apply_live_inputs() -> None:
        """실시간 소스의 경계·알림을 순서대로 코어에 반영한다(제어 흐름 전용).

        반영마다 `live_input`을 기록하고 소스에 반영 세대를 알린다. 수신 중단은 입력 오류로 세지 않는다.
        """
        for item in source.poll_input():
            kind = item.kind
            event = None

            if kind in (BoundaryKind.RECONNECT, BoundaryKind.SIM_TIME_REWIND):
                event = pipeline.change_settings(pipeline.settings, cause=GenerationCause(kind.value))
            elif kind is BoundaryKind.PROFILE_CHANGED:
                p = item.profile   # None = 해석할 수 없는 CameraInfo → 보정 없음 설정으로 전이
                profile = None if p is None else SourceProfile(source_id=pipeline.settings.source_id, width=p.width,
                                                               height=p.height, crop=None,
                                                               calibration_id=p.calibration_id)
                event = pipeline.change_settings(dataclasses.replace(pipeline.settings, profile=profile),
                                                 cause=GenerationCause.PROFILE_CHANGED)

            if event is not None:
                reset_tracker(kind.value)

            status_event = source.status_event(item)

            if status_event is not None:
                note_status(status_event)

                if InputStatus(status_event.status) is not InputStatus.FRAME_TIMEOUT:
                    reset_tracker(f"input_status:{InputStatus(status_event.status).value}")

                _logged(log.log_input_states, tracker.on_status(status_event))

                if InputStatus(status_event.status) is not InputStatus.FRAME_TIMEOUT:
                    result.input_errors += 1

            result.live_inputs[kind.value] = result.live_inputs.get(kind.value, 0) + 1
            _logged(log.write_live_input, {"input_kind": kind.value, "epoch": item.epoch, "detail": item.detail,
                                           "event_monotonic_ns": item.event_monotonic_ns,
                                           "generation": pipeline.current_generation(),
                                           "generation_changed": event is not None})
            sync_records()
            source.input_applied(item, pipeline.current_generation())

            if live_observation_sink is not None:
                live_observation_sink.on_live_input(item, pipeline.current_generation())

            if display_sink is not None:
                display_sink.on_input(kind.value, item.detail, pipeline.current_generation(), event is not None)

            # 수신기 비정상 종료는 자동 회복되지 않으므로 실행 실패로 끝낸다
            if kind is NoticeKind.RECEIVER_FAILED:
                fail_run("receiver_failed", f"live receiver failed: {item.detail}")
                raise _StopRun()

            if event is not None:
                note_transition(item.event_monotonic_ns)

    def finish_published(job: JobContext, value, exc: BaseException | None) -> None:
        """결과 공개 직전 처리. 실시간이면 입력을 먼저 반영하고, 수신 epoch가 지난 프레임은 관측 없이 끝낸다."""
        if live:
            apply_live_inputs()

            if exc is None and job.generation == pipeline.current_generation() and not source.frame_current(job.frame):
                result.input_boundary_aborts += 1
                handle(pipeline.fail(job, InvalidationKind.JOB_ABORTED,
                                     "input_boundary: frame receive epoch is no longer current"))
                return

        finish_detection(job, value, exc)

    def run_job(job: JobContext) -> None:
        """감시 끔: detector를 제어 흐름에서 동기 호출한다."""
        token = CancelToken()

        try:
            value = detector(job, token) if detector_takes_token else detector(job)
        except BaseException as exc:
            finish_published(job, None, exc)
            return

        finish_published(job, value, None)

    def discard_late(item) -> None:
        """terminal 뒤 유예 안에 돌아온 결과를 코어에 넘겨 duplicate 응답만 받고 버린다(기록·B 없음)."""
        late_job, value, exc = item

        try:
            obs, display = value
        except (TypeError, ValueError):
            exc = exc or ValueError("late result is not (observation, display)")

        if exc is None:
            outcome = pipeline.complete(late_job, obs, display_result=display)
        else:
            outcome = pipeline.fail(late_job, InvalidationKind.INFERENCE_ERROR, "late result after terminal")

        if not outcome.duplicate:
            raise RuntimeError(f"late result for job_seq={late_job.job_seq} was not a duplicate")

        result.late_results += 1

    def run_job_watched(job: JobContext) -> str | None:
        """감시 켬: worker에 넘기고 기한·정지 요청을 감시한다. 반환: None(계속) | "stop" | "stalled"."""
        nonlocal worker, first_pending

        limit = config.first_detect_timeout_s if first_pending else config.detect_timeout_s
        first_pending = False
        token = CancelToken()

        if worker is None:
            worker = _DetectorWorker(detector, detector_takes_token)

        started = watch_clock()
        worker.submit(job, token)

        def on_idle() -> None:
            """검출 대기 회차: 실시간 입력 반영, 연결부 실패 확인."""
            if live:
                apply_live_inputs()

            check_handoff()

        try:
            what, item = _wait_for_detection(worker.take, should_stop, watch_clock, started + limit,
                                             config.stop_check_interval_s,
                                             on_idle=on_idle if live or handoff_sink is not None else None)
        except KeyboardInterrupt:
            token._cancel()
            handle(pipeline.fail(job, InvalidationKind.JOB_ABORTED, "interrupted"))
            raise

        if what == "result":
            finish_published(*item)
            return None

        # 정지·기한 초과: 취소 → (실시간 입력 반영) → terminal 1회 → 유예 동안 실제 반환 대기
        token._cancel()

        if live:
            apply_live_inputs()

        if what == "stop":
            handle(pipeline.fail(job, InvalidationKind.JOB_ABORTED, "stop_requested"))
        else:
            result.timeouts += 1
            handle(pipeline.fail(job, InvalidationKind.JOB_ABORTED, "detect_timeout"))

        late = _await_return(worker, config.cancel_grace_s, config.stop_check_interval_s)

        if late is _NO_RESULT:
            fail_run("worker_stalled", f"detector worker stalled: job_seq={job.job_seq} did not return within "
                                       f"cancel grace {config.cancel_grace_s} s after {what}; no new detection")
            _write_thread_dump(err, config.stall_dump_path, f"worker_stalled job_seq={job.job_seq} after {what}")
            result.stalled = True
            return "stalled"

        discard_late(late)

        return "stop" if what == "stop" else None

    def prepare_failed(kind: InvalidationKind, cause: str, reason: str | None = None) -> None:
        """준비 실패: 원래 원인을 먼저 실행 실패로 남긴 뒤 프레임 없는 사건 1건을 적용·기록한다.

        사건 기록이 실패해도 원래 준비 실패 원인이 남는다. 프레임은 읽지 않는다.
        """
        reason = reason or ("model_load_failed" if kind is InvalidationKind.MODEL_LOAD_FAILED
                            else "model_config_invalid")
        fail_run(reason, f"model prepare failed: {kind.value}: {cause}")
        event = CandidateInvalidationEvent(
            run_id=issuer.run_id, source_id=pipeline.settings.source_id, kind=kind, cause=cause,
            event_monotonic_ns=clock_ns(), generation=pipeline.current_generation(),
            detector_config_id=pipeline.settings.detector_config_id, job_seq=None, image_seq=None)
        pipeline.apply_invalidation(event)
        sync_records()

    def prepare_outcome(value, exc) -> bool:
        """준비 반환·예외를 처리한다. 성공이면 model_ready를 쓰고 True."""
        if exc is None:
            _logged(log.write_model_ready, value if isinstance(value, dict) else {"info": repr(value)})
            return True

        if isinstance(exc, ModelPrepareError):
            prepare_failed(exc.kind, exc.cause)
        elif isinstance(exc, Exception):
            prepare_failed(InvalidationKind.MODEL_LOAD_FAILED, f"{type(exc).__name__}: {exc}")
        else:
            raise exc   # KeyboardInterrupt·SystemExit 등은 실행 수준으로 올린다

        return False

    def run_prepare() -> bool:
        """모델 준비 단계. 프레임 처리를 계속해도 되면 True."""
        nonlocal worker

        if not config.watchdog:
            try:
                value = prepare()
            except BaseException as exc:
                return prepare_outcome(None, exc)

            return prepare_outcome(value, None)

        if worker is None:
            worker = _DetectorWorker(detector, detector_takes_token)

        token = CancelToken()
        started = watch_clock()
        worker.submit_call("prepare", prepare, token)

        try:
            what, item = _wait_for_detection(worker.take, should_stop, watch_clock,
                                             started + config.model_load_timeout_s, config.stop_check_interval_s,
                                             on_idle=check_handoff if handoff_sink is not None else None)
        except KeyboardInterrupt:
            token._cancel()
            raise

        if what == "result":
            _, value, exc = item
            return prepare_outcome(value, exc)

        token._cancel()

        if what == "timeout":
            prepare_failed(InvalidationKind.MODEL_LOAD_FAILED,
                           f"prepare_timeout: model prepare did not finish within {config.model_load_timeout_s} s",
                           reason="model_load_timeout")

        late = _await_return(worker, config.cancel_grace_s, config.stop_check_interval_s)

        if late is _NO_RESULT:
            fail_run("worker_stalled", f"detector worker stalled: prepare did not return within cancel grace "
                                       f"{config.cancel_grace_s} s after {what}; no detection")
            _write_thread_dump(err, config.stall_dump_path, f"worker_stalled prepare after {what}")
            result.stop_reason = "worker_stalled"   # 시간 초과 사건을 남겼어도 멈춤이 최종 정지 사유
            result.stalled = True
            return False

        result.late_results += 1   # 늦은 준비 결과는 버린다(실행은 이미 실패·정지로 끝남)

        if what == "stop":
            result.stop_reason = "stop_requested"

        return False

    try:
        source = source_factory(issuer, pipeline.current_generation)
        live = callable(getattr(source, "poll_input", None))
        _logged(log.write_run_meta, pipeline, started_monotonic_ns=clock_ns(), versions=versions)

        if handoff_sink is not None:
            handoff_started = True   # 실패해도 결합 여부는 정리에서 연결부 상태로 판단한다
            handoff_sink.start_run(pipeline)

        reset_tracker("run_start")

        consecutive_errors = 0

        # 실시간 소스에서는 B 계산 소비를 허용하지 않는다
        if live and b_consumer is not None:
            fail_run("b_consumer_not_allowed", "live source does not allow b_consumer in the display stage")
            proceed = False
        elif live_observation_sink is not None and (not live or b_consumer is not None):
            fail_run("live_observation_contract", "live_observation_sink requires a live source and no b_consumer")
            proceed = False
        elif live_observation_b_input == "list" and (
                live_observation_sink is None
                or getattr(settings.detector_config, "axis_extractor", None) is None):
            fail_run("live_observation_contract",
                     "live_observation_b_input='list' requires live_observation_sink and an AI config with axis_extractor")
            proceed = False
        elif b_list_consumer is not None and b_consumer is not None:
            fail_run("b_input_contract", "b_consumer (stage T) and b_list_consumer (stage L) must not be given together")
            proceed = False
        elif b_list_consumer is not None and (live or live_observation_sink is not None):
            fail_run("b_list_consumer_not_allowed", "live source does not allow b_list_consumer (D13)")
            proceed = False
        elif b_list_consumer is not None and getattr(settings.detector_config, "axis_extractor", None) is None:
            fail_run("b_list_requires_axis", "b_list_consumer requires an AI config with axis_extractor")
            proceed = False
        else:
            proceed = prepare is None or run_prepare()

        while proceed:
            if should_stop is not None and should_stop():
                result.stop_reason = "stop_requested"
                break

            # 입력 대기 회차(read()가 None인 회차 포함): 추가 입력 없이도 수신 측 실패를 알아챈다
            check_handoff()

            # 읽기 전에 입력 경계·알림을 반영한다
            if live:
                apply_live_inputs()

            item = source.read()

            if item is None and live:
                continue

            if isinstance(item, InputStatusEvent):
                note_status(item)

                if InputStatus(item.status) is not InputStatus.FRAME_TIMEOUT:
                    reset_tracker(f"input_status:{InputStatus(item.status).value}")

                _logged(log.log_input_states, tracker.on_status(item))
                sync_records()
                status = InputStatus(item.status)

                if status is InputStatus.END_OF_STREAM:
                    result.stop_reason = "end_of_stream"
                    break

                if status is InputStatus.SOURCE_UNAVAILABLE and item.detail in _NO_INPUT_DETAILS:
                    fail_run("no_input", f"input unavailable: {item.detail}")
                    break

                result.input_errors += 1
                consecutive_errors += 1

                if consecutive_errors >= config.max_consecutive_input_errors:
                    fail_run("input_error_limit",
                             f"{consecutive_errors} consecutive input errors (last: {status.value} {item.detail})")
                    break

                continue

            if not isinstance(item, FramePacket):
                raise TypeError(f"source.read() returned {type(item).__name__}")

            consecutive_errors = 0
            _logged(log.log_input_states, tracker.on_frame(clock_ns()))
            result.frames += 1
            pipeline.offer_frame(item)
            job = pipeline.start_next()

            if job is not None and config.watchdog:
                verdict = run_job_watched(job)

                if verdict == "stop":
                    result.stop_reason = "stop_requested"
                    break

                if verdict == "stalled":
                    break
            elif job is not None:
                run_job(job)
            else:
                sync_records()

            if config.max_frames is not None and result.frames >= config.max_frames:
                result.stop_reason = "max_frames"
                break
    except _LogFailure as exc:
        log_ok = False
        fail_run("log_failure", f"log write failed: {exc}")
    except _StopRun:
        pass   # 사유는 이미 fail_run으로 남겼다
    except LiveInputOverflow as exc:
        fail_run("input_overflow", f"live input queue overflow: {exc}")
    except (HandoffFailed, HandoffClosed) as exc:
        # 연결부 실패는 원래 실행 오류와 구분된 사유로 남긴다(앞선 정지 사유가 있으면 그대로 둠)
        fail_run("handoff_failed", f"handoff failed: {type(exc).__name__}: {exc}")
    except KeyboardInterrupt:
        fail_run("interrupted", "interrupted (KeyboardInterrupt)")
    except BaseException as exc:  # 소비자 실패·예상하지 못한 예외·SystemExit도 정리를 거쳐 1로 끝낸다
        fail_run("error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
    finally:
        # 정리 전에 run_end reset을 같은 실패 규칙으로 하고, 기록 실패가 나도 정리는 계속한다
        if part_tracker is not None:
            try:
                reset_tracker("run_end")
            except _LogFailure as exc:
                log_ok = False
                fail_run("log_failure", f"log write failed: {exc}")
            except BaseException as exc:   # 정리를 막지 않는다
                fail_run("error", f"tracking cleanup: {type(exc).__name__}: {exc}")

        _finish(pipeline, log, tracker, source, config, clock_ns, result, log_ok, err, worker, live=live,
                handoff=handoff_sink, handoff_started=handoff_started)

        # 추적 보고(외부 호출이라 예외가 run 반환을 막지 않게)
        if part_tracker is not None:
            report = {"failed": tracking_failed[0] if tracking_failed else None}

            try:
                report = {**part_tracker.stats(), **report}
            except Exception as exc:
                report["stats_error"] = f"{type(exc).__name__}: {exc}"

            result.tracking = report

    return result


def _finish(pipeline, log, tracker, source, config, clock_ns, result, log_ok, err, worker=None, *,
            live: bool = False, handoff=None, handoff_started: bool = False) -> None:
    """정리: 코어 close(유한 재시도) → worker 정리 → 남은 기록 → 소스 close → 연결부 마지막 sync·RUN_END.

    각 단계 실패는 보고하고 다음 단계를 계속한다.
    """
    outcomes: tuple = ()
    close_error = None

    for attempt in range(1, config.close_attempts + 1):
        try:
            outcomes = pipeline.close()
            close_error = None
            break
        except Exception as exc:
            close_error = exc
            result.errors.append(f"close attempt {attempt}/{config.close_attempts} failed: "
                                 f"{type(exc).__name__}: {exc}")

            if attempt < config.close_attempts:
                time.sleep(config.close_retry_interval_s)

    if close_error is not None:
        result.exit_code = 1
        result.stop_reason = result.stop_reason or "close_failed"
        active = pipeline._active  # 남은 상태 보고용(읽기만)
        result.errors.append(
            f"close failed after {config.close_attempts} attempts: {type(close_error).__name__}: {close_error}; "
            f"remaining state: closed={pipeline.closed} active_job_seq={None if active is None else active.job_seq} "
            f"pending={pipeline.pending_frame is not None}")

    # 코어가 닫힌 뒤라 남은 worker 결과는 terminal·기록·B 없이 버리고 수만 보고한다
    if worker is not None:
        try:
            report = worker.shutdown(config.worker_cleanup_limit_s, config.stop_check_interval_s)
        except Exception as exc:
            report = {"stalled": worker.busy, "stalled_job_seq": None, "thread_stopped": False, "discarded": 0,
                      "error": f"{type(exc).__name__}: {exc}"}
            result.errors.append(f"worker shutdown failed: {report['error']}")

        result.worker = report

        if report["stalled"]:
            if not result.stalled:   # 위 두 곳에서 이미 남겼으면 다시 남기지 않는다
                _write_thread_dump(err, config.stall_dump_path, "worker still running at cleanup")

            result.stalled = True
            result.exit_code = 1
            result.stop_reason = result.stop_reason or "worker_stalled"
            what = report.get("stalled_what") or f"job_seq={report['stalled_job_seq']}"
            result.errors.append(f"worker still running: {what}, busy "
                                 f"{report.get('busy_s')} s, did not return within cleanup limit "
                                 f"{config.worker_cleanup_limit_s} s (thread cannot be force-stopped)")
        elif not report["thread_stopped"]:
            result.exit_code = 1
            result.errors.append("worker thread did not stop (idle) within cleanup limit")

    # 기록이 살아 있으면 close의 중단 결과(그 호출에서만 옴)·입력 구간·남은 사건/폐기를 한 번 쓴다
    if log_ok:
        try:
            for outcome in outcomes:
                log.log_outcome(outcome)

            log.log_input_states(tracker.finish(clock_ns()))
            log.sync(pipeline)
        except Exception as exc:
            log_ok = False
            result.exit_code = 1
            result.stop_reason = result.stop_reason or "log_failure"
            result.errors.append(f"log write failed during cleanup: {type(exc).__name__}: {exc}")

    # 실시간 소스 통계(수신·덮어쓰기·epoch·콜백 시간 등)를 한 번 기록한다
    if live and source is not None:
        try:
            result.source_stats = source.stats()

            if log_ok:
                log.write_source_stats(result.source_stats)
        except Exception as exc:
            result.exit_code = 1
            result.stop_reason = result.stop_reason or "log_failure"
            result.errors.append(f"source stats failed: {type(exc).__name__}: {exc}")

    result.missing = log.missing(pipeline)

    if not log_ok or result.stalled:
        result.errors.append(f"unrecorded (not confirmed in log): {result.missing}")

    if source is not None:
        try:
            source.close()
        except Exception as exc:
            result.exit_code = 1
            result.errors.append(f"source close failed: {type(exc).__name__}: {exc}")

    result.closed = pipeline.closed
    result.counters = pipeline.counters

    # 끝까지 처리했어도 작업 실패·입력 오류·검출 시간 초과가 있으면 1(누적 보고)
    if result.exit_code == 0 and (result.counters.failed or result.input_errors or result.timeouts):
        result.exit_code = 1
        result.errors.append(f"completed with {result.counters.failed} failed job(s) and "
                             f"{result.input_errors} input error(s); detect timeouts: {result.timeouts}")

    if handoff is not None:
        _finish_handoff(handoff, handoff_started, pipeline, result)

    # 최종 실패 통로는 stderr다(기록기가 깨졌어도 원래 오류가 남는다)
    for message in result.errors:
        try:
            err.write(f"run_pipeline: {message}\n")
        except Exception:
            pass


def _finish_handoff(handoff, started: bool, pipeline, result) -> None:
    """연결부 정리: 마지막 sync로 close 사건을 넘기고 미전달 수를 확정한 뒤, OPEN이면 RUN_END를 넣는다.

    결합된 연결부가 끝에 FAILED·CLOSED면 종료 1로 보고하되 앞선 정지 사유는 덮지 않는다. 연결부를 닫지 않는다.
    """
    report = {"run_end_sent": False, "events_unsent": 0}
    bound = False

    if started:
        try:
            handoff.sync_invalidations(pipeline)
            bound = True
        except (HandoffFailed, HandoffClosed) as exc:
            bound = True
            if not any(e.startswith("handoff failed") for e in result.errors):
                result.exit_code = 1
                result.stop_reason = result.stop_reason or "handoff_failed"
                result.errors.append(f"handoff failed: {type(exc).__name__}: {exc}")
        except HandoffError:
            bound = False   # 이 코어와 결합되지 않음(start_run 거부)

    if bound and handoff.state().state is HandoffState.OPEN:
        try:
            handoff.end_run(result.exit_code, result.stop_reason)
            report["run_end_sent"] = True
        except (HandoffFailed, HandoffClosed) as exc:
            result.exit_code = 1
            result.stop_reason = result.stop_reason or "handoff_failed"
            result.errors.append(f"handoff failed: {type(exc).__name__}: {exc}")

    stats = handoff.stats()
    status = handoff.state()

    # 이 상태 읽기가 실패 관측 경계다. 결합된 연결부가 OPEN이 아니면 실패로 보고한다
    if bound and status.state is not HandoffState.OPEN:
        message = f"handoff failed: state={status.state.value} reason={status.reason}"
        result.exit_code = 1
        result.stop_reason = result.stop_reason or "handoff_failed"

        if message not in result.errors:
            result.errors.append(message)

    report.update(state=status.state.value, reason=status.reason, bound=bound,
                  events_unsent=stats.events_unsent if bound else 0, events_enqueued=stats.events_enqueued if bound else 0,
                  records_seen=stats.records_seen if bound else 0, results_offered=stats.results_offered,
                  results_enqueued=stats.results_enqueued, published=stats.published, pending=stats.pending)

    if bound and (stats.events_unsent or not report["run_end_sent"]):
        result.errors.append(f"handoff_unsent: events={stats.events_unsent} run_end_sent={report['run_end_sent']}")

    result.handoff = report


def stored_source_factory(kind: str, paths: list[str], source_id: str):
    """CLI용 소스 팩토리. kind는 'images' 또는 'video'. 실행기의 단일 issuer·세대 공급자를 주입한다."""
    from eyetoarm.inputs.stored_images import StoredImageSource
    from eyetoarm.inputs.video import VideoSource

    def factory(issuer: FrameIdIssuer, generation_provider: Callable[[], int]):
        """issuer·generation_provider를 받아 소스를 만든다."""
        if kind == "images":
            return StoredImageSource(paths, source_id=source_id, issuer=issuer,
                                     generation_provider=generation_provider)

        return VideoSource(paths[0], source_id=source_id, issuer=issuer, generation_provider=generation_provider)

    return factory


__all__ = ["CancelToken", "JobCancelled", "RunnerConfig", "RunResult", "apply_seek", "b_admit", "b_gate", "b_list_gate",
           "b_list_skip_reason", "b_skip_reason", "display_gate", "run", "stored_source_factory", "thread_dump"]
