"""저장 이미지·영상 입력 실행기(계획 r4 작업 6a). 입력 → 작업 시작 → 검출 → terminal → 기록 → 정리.

근거: 계획 `docs/superpowers/plans/2026-11-15-a-vision-pipeline.md` r4 작업 6(P1~P3 보완 포함), A 상세설계 §5.4·§9,
작업 4 순환 시험(terminal 통지가 없으면 코어는 활성 작업을 풀지 않음), 작업 5 fail-stop 기록 정책.

범위·한계(6a·6b)
- C 서버·ROS·YOLO를 연결하지 않는다. 저장 입력 재현 경로다. A→C 같은 프로세스 통합은 별도 단계다.
- **감시 켬(6b, 기본):** detector는 같은 프로세스의 **검출 전용 worker 스레드 1개**에서 돌고, 제어 흐름이 시간 상한으로
  감시한다(설계 113 r3 A1). 코어·기록·B 지점 호출은 제어 흐름만 한다. 상한은 사용자 확정 **개발 초기값** 첫 검출
  10 s·이후 1 s이며 운영 최종값·최악 실행 시간 보장이 아니다. 시간 초과는 `JOB_ABORTED`(원인 `detect_timeout`)이고
  worker가 실제로 반환할 때까지 다음 작업을 시작하지 않는다. 유예 안에 반환하지 않으면 STALLED로 새 검출을 멈추고
  종료 1. 멈춘 Python 스레드는 강제로 끝낼 수 없으며(데몬이어도 자원 회수 보장 없음), GIL을 놓지 않는 네이티브 호출·
  제어 흐름 자체의 블로킹(입력·B 지점·기록 I/O)에서는 감시가 돌지 못한다(미검증).
- **감시 끔(6a 동기 경로):** 상한 둘 다 None이면 detector를 **제어 흐름 안에서 동기 호출**한다. terminal 보장은 호출이
  반환(정상·예외)하거나 실행기에 전달된 중단(`KeyboardInterrupt`, `JobCancelled`, `should_stop`)까지다.
- 실제 B 계산은 없다. `b_consumer`는 소비 조건(`b_gate`)을 통과한 응답을 받은 그 자리에서 한 번 부르는 지점이다.

종료 코드(A 내부 임시 결정)
- 0: 입력 끝(EOF)·`max_frames` 도달·`should_stop` 요청으로 끝났고 **처리 실패·입력 오류가 하나도 없음**.
- 1: 실행 수준 실패(입력 없음(빈 목록·닫힘), 연속 입력 오류 상한 초과, 기록 실패, 소비자 실패, `KeyboardInterrupt` 등
  중단, 예상하지 못한 예외, close 최종 실패), **또는** 끝까지 처리했지만 작업 처리 실패(`failed`: INFERENCE_ERROR·
  MODEL_CONFIG_INVALID)·입력 오류(EOF 아닌 입력 사건)·검출 시간 초과가 1건 이상 있음, 또는 검출 worker 멈춤(STALLED). 작업 실패·입력 오류는 그 자리에서 실행을 멈추지
  않고 기록한 뒤 다음 입력을 계속 처리하며, 마지막에 누적 수를 stderr에 적고 1로 끝낸다(계획 r4 「1=처리·전달·기록 실행
  오류」, 「정상 EOF는 디코딩 오류와 구분」에 맞춤, r1 Q1). `JobCancelled`로 정리한 작업(aborted)은 의도한 취소라 실패로
  세지 않는다. 세대 폐기도 실패가 아니다.
- 2: 인자·설정 오류(CLI에서 argparse·설정 읽기).
"""
from __future__ import annotations

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
from eyetoarm.vision.ai_result import AiJobConfigError, ModelPrepareError
from eyetoarm.vision.pipeline import GenerationCause, VisionPipeline
from eyetoarm.vision.pipeline_opencv import process_opencv
from eyetoarm.vision.pipeline_records import InputStateTracker, PipelineLog
from eyetoarm.vision.pipeline_types import JobContext, JobOutcome, JobStatus, PipelineCounters, PipelineSettings

# 저장 입력 소스가 「더 읽을 것이 없음」으로 돌려주는 사건 설명(inputs/stored_images.py)
_NO_INPUT_DETAILS = ("empty list", "closed")


class JobCancelled(Exception):
    """detector가 실행기에 취소를 알릴 때 올린다. 그 작업은 `fail(JOB_ABORTED)`로 정리되고 실행은 계속된다."""


class CancelToken:
    """협조 취소 표지(6b). 실행기가 시간 초과·정지 요청·정리 때 세우고, detector는 읽기만 한다.

    `run(..., detector_takes_token=True)`일 때만 `detector(job, token)`으로 넘긴다. detector가 확인해야만 효과가 있다
    (OpenCV·YOLO 단일 네이티브 호출 안에서는 확인할 수 없음). 확인했으면 `JobCancelled`를 올려 끝내면 된다.
    """

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
    """실행기 설정(A 내부). 시험은 값을 명시해 넣는다.

    - `close_attempts`(기본 3)·`close_retry_interval_s`(기본 0.05 s): 코어 close 재시도. 무한 재시도하지 않는다.
    - `max_consecutive_input_errors`(기본 8): 프레임 없이 연속된 입력 오류 상한. 영상 열기 실패처럼 진행 없이
      같은 실패가 반복될 때 실행이 끝나도록 한다. 저장 이미지의 파일별 실패는 프레임이 오면 0으로 되돌린다.
    - `max_frames`: 받은 프레임 수 상한(None이면 끝까지).
    - 검출 감시(6b). `first_detect_timeout_s`(기본 10.0)·`detect_timeout_s`(기본 1.0)는 **사용자 확정 개발 초기값**
      (2026-11-15 「일단 개발값으로 10, 1로 넣고 운용 테스트하면서 조절」)이며 운영 최종값이 아니다. 첫 값은 run 한 번에서
      실제 detector에 넘긴 최초 작업 하나에만, 둘째 값은 그 뒤 모든 작업에 쓴다. **둘 다 None이면 감시 끔**(6a 동기
      경로), 한쪽만 None은 오류다.
    - `model_load_timeout_s`(10.0): AI 모델 준비(`prepare`: 파일 확인·해시·import·생성·검사) 상한. **A 임시 개발 기본값**
      (YOLO 연결 구현 지시 Q1)이며 사용자 확정 첫 검출 10 s·이후 1 s와 다르다. 준비는 첫 검출 기회를 소모하지 않는다.
      감시를 끄면 쓰지 않는다(준비를 동기로 부름).
    - `cancel_grace_s`(0.2)·`worker_cleanup_limit_s`(0.3)·`stop_check_interval_s`(0.02): 시간 초과·정지 뒤 worker 반환
      유예, close 때 worker 정리 한도, 검출 대기 중 정지 요청·기한 확인 간격. 문서 근거 없는 **A 임시 개발 기본안**
      (설계 113 r3 시험 값)이며 사용자 확정 10/1과 구분한다. 유예·정리 한도는 실제 단조 시계로 잰다.
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

        for name in ("model_load_timeout_s", "cancel_grace_s", "worker_cleanup_limit_s", "stop_check_interval_s"):
            if not _positive_finite(getattr(self, name)):
                raise ValueError(f"{name} must be a finite number > 0: {getattr(self, name)!r}")

        for name in ("close_attempts", "max_consecutive_input_errors"):
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
    counters: PipelineCounters | None = None
    closed: bool = False
    missing: dict | None = None  # 기록이 확인되지 않은 처리·사건·폐기 수(정상 기록이면 모두 0)
    errors: list[str] = field(default_factory=list)
    timeouts: int = 0            # 검출 시간 초과 수(실패로 셈, 코어 aborted에도 들어감)
    stalled: bool = False        # 유예·정리 한도 안에 검출 호출이 반환하지 않음
    late_results: int = 0        # 시간 초과·정지 뒤 유예 안에 돌아와 duplicate로 버린 결과 수
    worker: dict | None = None   # worker 정리 보고(감시 끔이면 None)


def b_gate(pipeline: VisionPipeline, outcome: JobOutcome) -> bool:
    """B 소비 조건(계획 r4 작업 6). 응답을 받은 그 자리에서만 부른다.

    duplicate 아님, COMPLETED·applied, 유효 관측이고, 소비 시점에 현재 관측이 바로 그 관측이며 세대·검출 설정·
    소스가 같고 현재 프로파일이 그 프레임에 적용되며 관측 식별자가 작업과 같아야 한다. applied만으로는 B 전달
    허가가 아니다. 보관해 둔 과거 응답은 현재 관측이 바뀌면 통과하지 못한다.
    """
    obs = outcome.observation

    if outcome.duplicate or outcome.status is not JobStatus.COMPLETED or not outcome.applied:
        return False

    if not isinstance(obs, PlanarObservation) or obs.valid is not True or pipeline.closed:
        return False

    ctx = outcome.context
    settings = pipeline.settings
    profile = settings.profile

    return (pipeline.current_observation is obs
            and ctx.generation == pipeline.current_generation()
            and ctx.detector_config_id == settings.detector_config_id
            and ctx.frame.source_id == settings.source_id
            and profile is not None and profile.applies_to(ctx.frame)
            and (obs.job_seq, obs.image_seq, obs.generation) == (ctx.job_seq, ctx.frame.image_seq, ctx.generation))


def apply_seek(pipeline: VisionPipeline, source, frame_index: int) -> bool:
    """탐색을 시도한다. 성공하면 세대를 올리고(SEEK) True, 실패하면 위치·세대·선택을 그대로 두고 False.

    순서: 소스 seek 성공 → 세대 증가·대기 폐기·선택 비움 → 다음 read가 새 capture_generation을 받는다.
    첫 CLI에는 탐색 옵션이 없다. 이 함수는 코어와 VideoSource의 결합 시험용이다.
    """
    try:
        source.seek(frame_index)
    except (ValueError, RuntimeError, TypeError):
        return False

    pipeline.change_settings(pipeline.settings, cause=GenerationCause.SEEK)

    return True


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
    """검출 전용 상주 worker 스레드 1개(설계 113 r3 §6.1). 수신 칸·핸드오프 칸은 각 용량 1.

    - IDLE: 수신 칸 대기(살아 있어도 멈춤이 아님). BUSY: detector 호출 중.
    - `busy`는 제어 흐름이 소유한다. `submit`에서 참이 되고, 그 작업의 결과를 `take`로 **실제로 받아야** 거짓이 된다.
      그래서 코어가 작업을 ABORTED로 끝냈어도 호출이 반환하기 전에는 다음 작업을 넣을 수 없다(겹침 금지).
    - worker는 받은 바로 그 `JobContext`를 detector에 넘기고 같은 객체를 돌려준다(복제 없음, 코어 `is` 판별 유지).
      코어·기록·B 지점은 만지지 않는다. 반환·예외(BaseException 포함)를 `(job, 값, 예외)`로 핸드오프한다.
    - 데몬 스레드다. 인터프리터 종료를 막지 않을 뿐 멈춘 스레드의 자원 회수는 보장하지 않는다.
    """

    def __init__(self, detector, takes_token: bool) -> None:
        """스레드를 만들고 시작한다."""
        self._detector, self._takes_token = detector, takes_token
        self._inbox: queue.Queue = queue.Queue(maxsize=1)
        self._outbox: queue.Queue = queue.Queue(maxsize=1)
        self._stopping = threading.Event()  # 종료 의도(수신 칸 Full과 무관하게 보존, r1 F1)
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
        """종료 절차(§6.1 ①~⑤): BUSY면 취소 후 한도 안 반환 대기 → 종료 상태·표지 → 유한 join → 남은 핸드오프 버림.

        반환 대기가 끝난 뒤에야 종료 상태를 세운다(그 전에 세우면 한도 안의 정상 반환을 결과 없이 끝내 버림).
        실행 중인 detector를 강제로 멈추지 않으며 무한 join하지 않는다.
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

        # 종료 의도는 먼저 상태로 남긴다. 수신 칸이 차 있어(작업을 아직 꺼내지 않음) 표지를 못 넣어도, worker는 작업을
        # 꺼낸 직후·호출 반환 직후 이 상태를 보고 detector를 부르지 않거나 결과를 버리고 끝난다
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
            except BaseException as caught:  # 예외 분류는 제어 흐름이 한다(6a 동기 경로와 같은 규칙)
                exc = caught

            if self._stopping.is_set():
                return  # 정리가 끝난 뒤 반환한 결과는 소비자가 없으므로 남기지 않는다(영상 참조를 놓음)

            self._outbox.put_nowait((job, value, exc))

            # 다음 작업까지 IDLE로 기다리는 동안 끝난 작업의 영상을 붙잡지 않는다
            item = job = token = call = value = exc = None


def _wait_for_detection(take, should_stop, clock, deadline: float, interval_s: float):
    """검출 대기 루프(설계 113 r3 §5). 매 회차: 결과 → 정지 요청 → 기한 순서로 하나만 판정한다.

    `take(timeout)`이 돌려준 값은 그 자리에서 ("result", 값)으로 보존해 돌려준다. 비었을 때(queue.Empty)만 정지 요청,
    그다음 기한을 확인한다. 반환: ("result", 값) | ("stop", None) | ("timeout", None).
    """
    while True:
        remaining = deadline - clock()

        try:
            return "result", take(max(0.0, min(remaining, interval_s)))
        except queue.Empty:
            pass

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


def run(source_factory: Callable[[FrameIdIssuer, Callable[[], int]], object], settings: PipelineSettings,
        log: PipelineLog, *, run_id: str | None = None,
        detector: Callable[[JobContext], tuple[PlanarObservation, object]] = process_opencv,
        b_consumer: Callable[[FramePacket, PlanarObservation], None] | None = None,
        config: RunnerConfig = RunnerConfig(), stderr=None, clock_ns: Callable[[], int] = time.monotonic_ns,
        should_stop: Callable[[], bool] | None = None, versions: dict | None = None,
        detector_takes_token: bool = False, watch_clock: Callable[[], float] = time.monotonic,
        prepare: Callable[[], dict] | None = None) -> RunResult:
    """저장 입력 한 번을 끝까지 처리한다. 예외를 밖으로 내지 않고 `RunResult`(종료 코드 포함)를 돌려준다.

    `source_factory(issuer, generation_provider)`는 실행기가 만든 단일 발급기와 `pipeline.current_generation`을
    받아 소스(`read()`·`close()`)를 만든다. 정리(finally)는 코어 close(유한 재시도) → worker 정리(감시 켬) → 남은 기록
    → 소스 close 순서로 각각 한 번씩 하며, 앞 단계 실패가 뒤 단계를 막지 않는다.

    - `detector_takes_token`: True면 `detector(job, CancelToken)`, False(기본)면 기존 `detector(job)`. 호출 규약은
      이 명시 인자로만 정한다(시그니처 추측·TypeError 재시도 없음).
    - `watch_clock`: 검출 기한을 재는 단조 시계(초). 시험이 결정적으로 바꿔 넣는 주입 지점이다. 유예·정리 한도는
      실제 단조 시계로 잰다(주입 시계가 멈춰도 정리는 유한하게 끝남).
    - 검출 기한 경계: 제어 흐름이 worker에 작업을 넘기기 직전 `watch_clock()`부터, 핸드오프 칸에서 결과를 꺼내거나
      칸이 빈 채 기한을 판정할 때까지. 실측(#114)의 `process_opencv`·`predict` 호출 경계에 스레드 전달·깨우기 지연이
      더해진다. 파일 열기·디코딩·B 지점·기록은 포함하지 않는다.
    - `prepare`(YOLO 연결 R3): AI 모델 준비 함수. run_meta 뒤 **프레임을 읽기 전에** 한 번 부른다. 감시를 켜면 같은 worker에
      작업 아닌 호출로 넘기고 `model_load_timeout_s`로 감시한다(첫 검출 기회를 쓰지 않음). 성공하면 반환 dict를
      `model_ready`로 기록한다. 실패는 프레임 없는 사건(`ModelPrepareError`의 kind, 그 밖은 MODEL_LOAD_FAILED, 시간 초과는
      MODEL_LOAD_FAILED `prepare_timeout`)으로 적용·기록하고 프레임 없이 종료 1. 정지 요청·KeyboardInterrupt는 사건 없이
      정리한다(계획 r2 §4.2 표). 가짜 image_seq·job_seq를 만들지 않는다.
    """
    err = stderr if stderr is not None else sys.stderr
    issuer = FrameIdIssuer(run_id or new_run_id())
    pipeline = VisionPipeline(settings, run_id=issuer.run_id, clock_ns=clock_ns)
    result = RunResult(exit_code=0, stop_reason="", run_id=issuer.run_id)
    tracker = InputStateTracker(issuer.run_id, settings.source_id)
    source = None
    log_ok = True
    worker: _DetectorWorker | None = None
    first_pending = True   # run 한 번에서 아직 첫 검출을 넘기지 않음(정지로 시작하지 않은 작업은 소모하지 않음)

    def fail_run(reason: str, message: str) -> None:
        """실행 수준 실패를 기록한다(종료 1)."""
        result.exit_code = 1
        result.stop_reason = result.stop_reason or reason
        result.errors.append(message)

    def handle(outcome: JobOutcome) -> None:
        """terminal 응답을 기록하고, 소비 조건을 통과하면 그 자리에서 B 지점을 한 번 부른다."""
        _logged(log.log_outcome, outcome)
        _logged(log.sync, pipeline)

        if b_consumer is not None and b_gate(pipeline, outcome):
            result.b_calls += 1
            b_consumer(outcome.context.frame, outcome.observation)

    def finish_detection(job: JobContext, value, exc: BaseException | None) -> None:
        """detector 반환·예외를 terminal로 정확히 한 번 보낸다(동기·worker 경로 공통 분류).

        JobCancelled → JOB_ABORTED(cancelled), KeyboardInterrupt → JOB_ABORTED(interrupted) 후 다시 올림,
        Exception → INFERENCE_ERROR, 그 밖의 BaseException → 다시 올림(활성 작업은 close가 정리).
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
            # 작업 중 설정 불일치(ROI·시작 설정·class_map 밖 클래스)는 MODEL_CONFIG_INVALID 작업 값(A §5.4)
            handle(pipeline.fail(job, InvalidationKind.MODEL_CONFIG_INVALID, exc.cause))
        elif isinstance(exc, KeyboardInterrupt):
            handle(pipeline.fail(job, InvalidationKind.JOB_ABORTED, "interrupted"))
            raise exc
        elif isinstance(exc, Exception):
            handle(pipeline.fail(job, InvalidationKind.INFERENCE_ERROR, f"{type(exc).__name__}: {exc}"))
        else:
            raise exc

    def run_job(job: JobContext) -> None:
        """감시 끔(6a): detector를 제어 흐름에서 동기 호출한다."""
        token = CancelToken()

        try:
            value = detector(job, token) if detector_takes_token else detector(job)
        except BaseException as exc:
            finish_detection(job, None, exc)
            return

        finish_detection(job, value, None)

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
        """감시 켬(6b): worker에 넘기고 기한·정지 요청을 감시한다. 반환: None(계속) | "stop" | "stalled"."""
        nonlocal worker, first_pending

        limit = config.first_detect_timeout_s if first_pending else config.detect_timeout_s
        first_pending = False
        token = CancelToken()

        if worker is None:
            worker = _DetectorWorker(detector, detector_takes_token)

        started = watch_clock()
        worker.submit(job, token)

        try:
            what, item = _wait_for_detection(worker.take, should_stop, watch_clock, started + limit,
                                             config.stop_check_interval_s)
        except KeyboardInterrupt:
            token._cancel()
            handle(pipeline.fail(job, InvalidationKind.JOB_ABORTED, "interrupted"))
            raise

        if what == "result":
            finish_detection(*item)
            return None

        # 정지 요청 또는 기한 초과: 취소 표지 → 즉시 terminal 1회 → 유예 동안 실제 반환 대기(멱등, 추가 terminal 없음)
        token._cancel()

        if what == "stop":
            handle(pipeline.fail(job, InvalidationKind.JOB_ABORTED, "stop_requested"))
        else:
            result.timeouts += 1
            handle(pipeline.fail(job, InvalidationKind.JOB_ABORTED, "detect_timeout"))

        late = _await_return(worker, config.cancel_grace_s, config.stop_check_interval_s)

        if late is _NO_RESULT:
            fail_run("worker_stalled", f"detector worker stalled: job_seq={job.job_seq} did not return within "
                                       f"cancel grace {config.cancel_grace_s} s after {what}; no new detection")
            result.stalled = True
            return "stalled"

        discard_late(late)

        return "stop" if what == "stop" else None

    def prepare_failed(kind: InvalidationKind, cause: str, reason: str | None = None) -> None:
        """준비 실패: 실행 실패(원래 원인)를 **먼저** 보존한 뒤 프레임 없는 사건 1건을 코어에 적용·기록한다(r1 F1).

        사건 기록이 실패해도(_LogFailure) 원래 준비 실패 원인이 RunResult.errors·stderr에 남는다. 대표 정지 사유는
        원래 준비 실패(먼저 정해짐)이고, 기록 실패는 바깥 처리에서 errors에 추가된다. 프레임은 읽지 않는다.
        """
        reason = reason or ("model_load_failed" if kind is InvalidationKind.MODEL_LOAD_FAILED
                            else "model_config_invalid")
        fail_run(reason, f"model prepare failed: {kind.value}: {cause}")
        event = CandidateInvalidationEvent(
            run_id=issuer.run_id, source_id=pipeline.settings.source_id, kind=kind, cause=cause,
            event_monotonic_ns=clock_ns(), generation=pipeline.current_generation(),
            detector_config_id=pipeline.settings.detector_config_id, job_seq=None, image_seq=None)
        pipeline.apply_invalidation(event)
        _logged(log.sync, pipeline)

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
            raise exc   # KeyboardInterrupt·SystemExit 등은 6a 분류대로 실행 수준으로 올린다

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
                                             started + config.model_load_timeout_s, config.stop_check_interval_s)
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
            result.stop_reason = "worker_stalled"   # 시간 초과 사건을 남겼어도 멈춤이 최종 정지 사유(계획 §4.2 표)
            result.stalled = True
            return False

        result.late_results += 1   # 늦은 준비 결과는 버린다(실행은 이미 실패·정지로 끝남)

        if what == "stop":
            result.stop_reason = "stop_requested"

        return False

    try:
        source = source_factory(issuer, pipeline.current_generation)
        _logged(log.write_run_meta, pipeline, started_monotonic_ns=clock_ns(), versions=versions)
        consecutive_errors = 0
        proceed = prepare is None or run_prepare()

        while proceed:
            if should_stop is not None and should_stop():
                result.stop_reason = "stop_requested"
                break

            item = source.read()

            if isinstance(item, InputStatusEvent):
                pipeline.note_input_status(item)
                _logged(log.log_input_states, tracker.on_status(item))
                _logged(log.sync, pipeline)
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
                _logged(log.sync, pipeline)

            if config.max_frames is not None and result.frames >= config.max_frames:
                result.stop_reason = "max_frames"
                break
    except _LogFailure as exc:
        log_ok = False
        fail_run("log_failure", f"log write failed: {exc}")
    except KeyboardInterrupt:
        fail_run("interrupted", "interrupted (KeyboardInterrupt)")
    except BaseException as exc:  # 소비자 실패·예상하지 못한 예외·SystemExit도 정리를 거쳐 1로 끝낸다
        fail_run("error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
    finally:
        _finish(pipeline, log, tracker, source, config, clock_ns, result, log_ok, err, worker)

    return result


def _finish(pipeline, log, tracker, source, config, clock_ns, result, log_ok, err, worker=None) -> None:
    """정리: 코어 close(유한 재시도) → worker 정리(감시 켬) → 중단 결과·남은 기록 → 소스 close.

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

    # worker 정리(§6.1): 코어가 닫힌 뒤라 남은 결과는 terminal·기록·B 없이 버리고 수만 보고한다
    if worker is not None:
        try:
            report = worker.shutdown(config.worker_cleanup_limit_s, config.stop_check_interval_s)
        except Exception as exc:
            report = {"stalled": worker.busy, "stalled_job_seq": None, "thread_stopped": False, "discarded": 0,
                      "error": f"{type(exc).__name__}: {exc}"}
            result.errors.append(f"worker shutdown failed: {report['error']}")

        result.worker = report

        if report["stalled"]:
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

    # 최종 실패 통로는 stderr다(기록기가 깨졌어도 원래 오류가 남는다)
    for message in result.errors:
        try:
            err.write(f"run_pipeline: {message}\n")
        except Exception:
            pass


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


__all__ = ["CancelToken", "JobCancelled", "RunnerConfig", "RunResult", "apply_seek", "b_gate", "run", "stored_source_factory"]
