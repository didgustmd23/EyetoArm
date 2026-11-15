"""저장 이미지·영상 입력 실행기(계획 r4 작업 6a). 입력 → 작업 시작 → 검출 → terminal → 기록 → 정리.

근거: 계획 `docs/superpowers/plans/2026-11-15-a-vision-pipeline.md` r4 작업 6(P1~P3 보완 포함), A 상세설계 §5.4·§9,
작업 4 순환 시험(terminal 통지가 없으면 코어는 활성 작업을 풀지 않음), 작업 5 fail-stop 기록 정책.

범위·한계(6a)
- C 서버·ROS·YOLO를 연결하지 않는다. 저장 입력 재현 경로다. A→C 같은 프로세스 통합은 별도 단계다.
- detector는 **제어 흐름 안에서 동기 호출**한다. 그래서 worker가 겹치지 않고, 다음 작업은 이전 호출이 실제로
  반환한 뒤에만 시작한다. terminal 보장은 호출이 반환(정상·예외)하거나 실행기에 실제로 전달된 중단
  (`KeyboardInterrupt`, detector가 올린 `JobCancelled`, `should_stop`)까지다. **반환하지 않는 호출은 보장하지 못한다**
  (감시·격리는 작업 6b, 미검증).
- 실제 B 계산은 없다. `b_consumer`는 소비 조건(`b_gate`)을 통과한 응답을 받은 그 자리에서 한 번 부르는 지점이다.

종료 코드(A 내부 임시 결정)
- 0: 입력 끝(EOF)·`max_frames` 도달·`should_stop` 요청으로 끝났고 **처리 실패·입력 오류가 하나도 없음**.
- 1: 실행 수준 실패(입력 없음(빈 목록·닫힘), 연속 입력 오류 상한 초과, 기록 실패, 소비자 실패, `KeyboardInterrupt` 등
  중단, 예상하지 못한 예외, close 최종 실패), **또는** 끝까지 처리했지만 작업 처리 실패(`failed`: INFERENCE_ERROR·
  MODEL_CONFIG_INVALID)나 입력 오류(EOF 아닌 입력 사건)가 1건 이상 있음. 작업 실패·입력 오류는 그 자리에서 실행을 멈추지
  않고 기록한 뒤 다음 입력을 계속 처리하며, 마지막에 누적 수를 stderr에 적고 1로 끝낸다(계획 r4 「1=처리·전달·기록 실행
  오류」, 「정상 EOF는 디코딩 오류와 구분」에 맞춤, r1 Q1). `JobCancelled`로 정리한 작업(aborted)은 의도한 취소라 실패로
  세지 않는다. 세대 폐기도 실패가 아니다.
- 2: 인자·설정 오류(CLI에서 argparse·설정 읽기).
"""
from __future__ import annotations

import math
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable

from eyetoarm.common.types import FramePacket, InputStatus, InputStatusEvent, InvalidationKind, PlanarObservation
from eyetoarm.inputs.base import FrameIdIssuer, new_run_id
from eyetoarm.vision.pipeline import GenerationCause, VisionPipeline
from eyetoarm.vision.pipeline_opencv import process_opencv
from eyetoarm.vision.pipeline_records import InputStateTracker, PipelineLog
from eyetoarm.vision.pipeline_types import JobContext, JobOutcome, JobStatus, PipelineCounters, PipelineSettings

# 저장 입력 소스가 「더 읽을 것이 없음」으로 돌려주는 사건 설명(inputs/stored_images.py)
_NO_INPUT_DETAILS = ("empty list", "closed")


class JobCancelled(Exception):
    """detector가 실행기에 취소를 알릴 때 올린다. 그 작업은 `fail(JOB_ABORTED)`로 정리되고 실행은 계속된다."""


@dataclass(frozen=True)
class RunnerConfig:
    """실행기 설정(A 내부). 기본값은 문서 근거가 없는 **임시 기본안**이며 시험은 값을 명시해 넣는다.

    - `close_attempts`(기본 3)·`close_retry_interval_s`(기본 0.05 s): 코어 close 재시도. 무한 재시도하지 않는다.
    - `max_consecutive_input_errors`(기본 8): 프레임 없이 연속된 입력 오류 상한. 영상 열기 실패처럼 진행 없이
      같은 실패가 반복될 때 실행이 끝나도록 한다. 저장 이미지의 파일별 실패는 프레임이 오면 0으로 되돌린다.
    - `max_frames`: 받은 프레임 수 상한(None이면 끝까지).
    """

    close_attempts: int = 3
    close_retry_interval_s: float = 0.05
    max_consecutive_input_errors: int = 8
    max_frames: int | None = None

    def __post_init__(self) -> None:
        """값 범위를 검사한다. 어긋나면 ValueError."""
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


def run(source_factory: Callable[[FrameIdIssuer, Callable[[], int]], object], settings: PipelineSettings,
        log: PipelineLog, *, run_id: str | None = None,
        detector: Callable[[JobContext], tuple[PlanarObservation, object]] = process_opencv,
        b_consumer: Callable[[FramePacket, PlanarObservation], None] | None = None,
        config: RunnerConfig = RunnerConfig(), stderr=None, clock_ns: Callable[[], int] = time.monotonic_ns,
        should_stop: Callable[[], bool] | None = None, versions: dict | None = None) -> RunResult:
    """저장 입력 한 번을 끝까지 처리한다. 예외를 밖으로 내지 않고 `RunResult`(종료 코드 포함)를 돌려준다.

    `source_factory(issuer, generation_provider)`는 실행기가 만든 단일 발급기와 `pipeline.current_generation`을
    받아 소스(`read()`·`close()`)를 만든다. 정리(finally)는 코어 close(유한 재시도) → 남은 기록 → 소스 close
    순서로 각각 한 번씩 하며, 앞 단계 실패가 뒤 단계를 막지 않는다.
    """
    err = stderr if stderr is not None else sys.stderr
    issuer = FrameIdIssuer(run_id or new_run_id())
    pipeline = VisionPipeline(settings, run_id=issuer.run_id, clock_ns=clock_ns)
    result = RunResult(exit_code=0, stop_reason="", run_id=issuer.run_id)
    tracker = InputStateTracker(issuer.run_id, settings.source_id)
    source = None
    log_ok = True

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

    def run_job(job: JobContext) -> None:
        """detector를 동기 호출하고 결과를 terminal로 정확히 한 번 보낸다."""
        try:
            obs, display = detector(job)
        except JobCancelled as exc:
            handle(pipeline.fail(job, InvalidationKind.JOB_ABORTED, f"cancelled: {exc}" if str(exc) else "cancelled"))
            return
        except KeyboardInterrupt:
            handle(pipeline.fail(job, InvalidationKind.JOB_ABORTED, "interrupted"))
            raise
        except Exception as exc:
            handle(pipeline.fail(job, InvalidationKind.INFERENCE_ERROR, f"{type(exc).__name__}: {exc}"))
            return

        handle(pipeline.complete(job, obs, display_result=display))

    try:
        source = source_factory(issuer, pipeline.current_generation)
        _logged(log.write_run_meta, pipeline, started_monotonic_ns=clock_ns(), versions=versions)
        consecutive_errors = 0

        while True:
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

            if job is not None:
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
        _finish(pipeline, log, tracker, source, config, clock_ns, result, log_ok, err)

    return result


def _finish(pipeline, log, tracker, source, config, clock_ns, result, log_ok, err) -> None:
    """정리: 코어 close(유한 재시도) → 중단 결과·남은 기록 → 소스 close. 각 단계 실패는 보고하고 다음 단계를 계속한다."""
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

    if not log_ok:
        result.errors.append(f"unrecorded (not confirmed in log): {result.missing}")

    if source is not None:
        try:
            source.close()
        except Exception as exc:
            result.exit_code = 1
            result.errors.append(f"source close failed: {type(exc).__name__}: {exc}")

    result.closed = pipeline.closed
    result.counters = pipeline.counters

    # 끝까지 처리했어도 작업 실패·입력 오류가 있으면 1(누적 보고)
    if result.exit_code == 0 and (result.counters.failed or result.input_errors):
        result.exit_code = 1
        result.errors.append(f"completed with {result.counters.failed} failed job(s) and "
                             f"{result.input_errors} input error(s)")

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


__all__ = ["JobCancelled", "RunnerConfig", "RunResult", "apply_seek", "b_gate", "run", "stored_source_factory"]
