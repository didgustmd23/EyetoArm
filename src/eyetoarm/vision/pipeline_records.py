"""파이프라인 처리 기록·사건·대기 폐기·입력 상태의 JSONL 기록(계획 r4 작업 5).

근거: 계획 `docs/superpowers/plans/2026-11-15-a-vision-pipeline.md` r4 작업 5, A 상세설계 §9(처리 기록·입력 상태
로그·처리 건수·평가 실행 식별), 작업 4 합의(`_workspace/103_builder_task4_q13_policy.md` §4: duplicate 응답은
새로 발행·집계하지 않음).

기록 형식(A 내부 제안, `format_version` 1): JSON 한 줄 = 기록 하나. 모든 줄에 `format_version`·`kind`가 있다.
- `run_meta`: 실행 하나의 메타데이터(run_id·eval_run_id·소스·모드·검출 설정 ID·보정 ID·라이브러리 버전·시작 시각)
- `process`: 작업 하나의 최초 terminal 결과(`ProcessRecord`)
- `event`: 무효화 사건 하나(`CandidateInvalidationEvent`). 사건의 `kind` 필드는 기록 종류 `kind`와 겹치므로
  `invalidation_kind`로 이름을 바꿔 쓴다
- `queue_drop`: 시작하지 않고 폐기한 프레임 하나(`QueueDropRecord`)
- `input_state`: 입력 상태 구간 하나(`InputStateRecord`)

1회 기록 규칙
- 작업마다 최초 terminal(`duplicate=False`) 응답의 처리 기록 1줄. duplicate 응답은 쓰지 않는다.
- 사건은 같은 9필드 사건을 1줄만 쓴다(응답의 `invalidation`과 코어 사건 목록이 같은 사건이어도 1줄).
- 대기 폐기 기록은 코어 기록 목록 순서대로 1줄씩.
- 정확히 1회는 **기록 실패가 없는 실행**에서만 보장한다. 쓰기·flush 실패나 짧은 쓰기는 그 줄이 스트림에 들어갔는지
  알 수 없으므로(일부·전부 기록 가능), 실패 뒤 다시 쓰면 중복이나 잘린 줄이 생길 수 있다.

실패 정책(fail-stop, r1 U1·U2): 기록 여부가 불확실한 실패가 한 번이라도 나면 `JsonlRecorder`는 **실패 상태**가
되고 이후 모든 쓰기를 `RecorderFailedError`로 거부한다(재시도하지 않음). `PipelineLog`도 같은 실패를 올린다.
호출자(작업 6 실행기)는 이 오류를 실행 종료 오류로 다룬다. 코어 close 등 자원 정리는 계속하되 기록은 더 하지 않고,
`PipelineLog.missing(pipeline)`이 돌려주는 「기록되지 않은 처리·사건·폐기 수」와 원래 오류를 stderr로 보고하며
종료 코드 1로 끝낸다. 코어 계수는 다시 세지 않는다. 실패 직전 줄은 스트림에 일부·전부 남았을 수 있으므로
결과 파일은 손상 가능성이 있는 것으로 본다.

`eval_run_id` 위치(A 내부 결정): A §9가 처리 작업마다 남기라고 하므로 `ProcessRecord.eval_run_id`에 둔다.
같은 값을 `run_meta`에도 둔다. 값은 평가 도구가 발급해 넘기며 A가 만들지 않는다. 평가가 아니면 None이다.

로그 오류: 스트림 쓰기 실패는 그대로 올린다(기록 오류). 코어 계수·terminal 상태는 건드리지 않으며,
이미 끝난 작업을 실패로 다시 집계하지 않는다. 기록기의 성공·실패 줄 수는 코어 계수와 따로 센다.
영상·윤곽 배열은 기록하지 않는다.
"""
from __future__ import annotations

import dataclasses
import json
import platform
from enum import Enum
from typing import Any

from eyetoarm.common.types import CandidateInvalidationEvent, InputStatus, InputStatusEvent
from eyetoarm.vision.pipeline_types import (
    InputStateRecord,
    JobOutcome,
    JobStatus,
    ProcessRecord,
)

FORMAT_VERSION = 1

# A가 판정하는 사유(A 상세설계 §5.4 「판정 주체」 A). 그 밖의 값은 주체를 적지 않는다
_A_REASONS = frozenset({
    "NO_OBJECT", "MULTIPLE_OBJECTS", "CONTOUR_CLIPPED", "TOP_FACE_UNRESOLVED", "IMAGE_ASPECT_LOW",
    "CALIBRATION_MISMATCH", "MARKER_NOT_FOUND", "PNP_FAILED",
})


def _check_eval_run_id(eval_run_id) -> str | None:
    """None 또는 비어 있지 않은 문자열인지 확인한다. 아니면 ValueError(가짜 ID를 만들지 않음)."""
    if eval_run_id is not None and (not isinstance(eval_run_id, str) or not eval_run_id):
        raise ValueError(f"eval_run_id must be None or a non-empty string: {eval_run_id!r}")

    return eval_run_id


def process_record(outcome: JobOutcome, *, eval_run_id: str | None = None) -> ProcessRecord:
    """최초 terminal 응답을 처리 기록으로 바꾼다. 시작 당시 context(원본 시각·시작 설정·세대)를 그대로 쓴다.

    duplicate 응답은 ValueError(재기록 금지). 관측이 있는 완료만 valid·reason·후보 수·처리 시간을 채운다.
    """
    if outcome.duplicate:
        raise ValueError("duplicate outcome must not be recorded again")

    job = outcome.context
    f = job.frame
    obs = outcome.observation if outcome.status is JobStatus.COMPLETED else None
    reason = obs.reason if obs is not None else None

    return ProcessRecord(
        run_id=f.run_id, source_id=f.source_id, image_seq=f.image_seq,
        capture_generation=f.capture_generation, generation=job.generation, job_seq=job.job_seq,
        mode=job.settings.mode, timestamp_ms=f.timestamp_ms, timestamp_base=f.timestamp_base,
        calibration_id=f.calibration_id, detector_config_id=job.detector_config_id,
        reprocess=job.reprocess, status=outcome.status,
        valid=obs.valid if obs is not None else None, reason=reason,
        reason_owner="A" if reason in _A_REASONS else None,
        candidate_count=obs.candidate_count if obs is not None else None,
        process_ms=obs.process_ms if obs is not None else None,
        command_id=None, eval_run_id=_check_eval_run_id(eval_run_id))


def _plain(value: Any) -> Any:
    """기록용 값으로 바꾼다: 열거는 값, 튜플은 목록, dataclass는 dict. 배열은 허용하지 않는다."""
    if isinstance(value, Enum):
        return value.value

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}

    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]

    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}

    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    raise TypeError(f"unsupported value in record: {type(value).__name__}")


class RecorderFailedError(RuntimeError):
    """앞선 쓰기 실패로 기록기가 실패 상태라 더 쓰지 않는다. `cause`에 처음 실패한 예외가 있다."""

    def __init__(self, cause: BaseException) -> None:
        """처음 실패한 예외를 받는다."""
        super().__init__(f"recorder is in failed state after: {type(cause).__name__}: {cause}")
        self.cause = cause


class JsonlRecorder:
    """기록 하나를 JSON 한 줄로 스트림에 쓴다. 쓰기 성공·실패 줄 수를 센다(코어 계수와 별개).

    쓰기·flush 예외나 짧은 쓰기(write가 돌려준 길이 < 줄 길이)가 나면 실패 상태가 되고, 이후 쓰기는
    `RecorderFailedError`다. write가 None을 돌려주는 스트림은 길이를 확인할 수 없어 성공으로 본다.
    직렬화 오류(스트림에 쓰기 전)는 실패 상태로 만들지 않는다.
    """

    def __init__(self, stream) -> None:
        """쓰기 가능한 텍스트 스트림(`write`·`flush`)을 받는다."""
        self._stream = stream
        self.written = 0
        self.failed = 0
        self.error: BaseException | None = None

    @property
    def broken(self) -> bool:
        """실패 상태인지."""
        return self.error is not None

    def write(self, kind: str, record) -> None:
        """기록을 직렬화해 쓴다. 직렬화 오류는 ValueError·TypeError, 쓰기 실패는 원래 예외(이후 RecorderFailedError)."""
        if self.error is not None:
            raise RecorderFailedError(self.error)

        payload = _plain(record)
        clash = {"format_version", "kind"} & payload.keys()

        if clash:
            raise ValueError(f"record field names clash with envelope keys: {sorted(clash)}")

        body = {"format_version": FORMAT_VERSION, "kind": kind}
        body.update(payload)
        text = json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"

        try:
            n = self._stream.write(text)

            if n is not None and n != len(text):
                raise OSError(f"short write: {n} of {len(text)} characters")

            self._stream.flush()
        except Exception as exc:
            self.failed += 1
            self.error = exc
            raise

        self.written += 1


class PipelineLog:
    """파이프라인 기록을 1회씩만 쓰는 층. 무엇을 이미 썼는지(작업·사건·폐기)를 기억한다."""

    def __init__(self, recorder: JsonlRecorder, *, eval_run_id: str | None = None) -> None:
        """기록기와 평가 실행 ID(평가가 아니면 None)를 받는다."""
        self._recorder = recorder
        self._eval_run_id = _check_eval_run_id(eval_run_id)
        self._jobs_logged: set[int] = set()
        self._events_logged: set[CandidateInvalidationEvent] = set()
        self._drops_logged = 0

    @property
    def eval_run_id(self) -> str | None:
        """이 실행의 평가 실행 ID."""
        return self._eval_run_id

    def _check(self) -> None:
        """기록기가 실패 상태면 쓸 내용이 없어도 RecorderFailedError(fail-stop을 호출자에게 알림)."""
        if self._recorder.error is not None:
            raise RecorderFailedError(self._recorder.error)

    def write_run_meta(self, pipeline, *, started_monotonic_ns: int, versions: dict | None = None) -> None:
        """실행 메타데이터 1줄. 라이브러리 버전은 넘겨받은 값과 Python 버전만 쓴다."""
        self._check()
        s = pipeline.settings
        meta = {
            "run_id": pipeline.run_id, "eval_run_id": self._eval_run_id, "source_id": s.source_id,
            "mode": s.mode, "detector_config_id": s.detector_config_id,
            "calibration_id": s.profile.calibration_id if s.profile is not None else None,
            "started_monotonic_ns": started_monotonic_ns,
            "versions": {"python": platform.python_version(), **(versions or {})},
        }
        self._recorder.write("run_meta", meta)

    def log_outcome(self, outcome: JobOutcome) -> bool:
        """최초 terminal 응답의 처리 기록과 그 사건을 1회 쓴다. duplicate·이미 쓴 작업이면 False(무기록)."""
        self._check()
        if outcome.duplicate or outcome.context.job_seq in self._jobs_logged:
            return False

        self._recorder.write("process", process_record(outcome, eval_run_id=self._eval_run_id))
        self._jobs_logged.add(outcome.context.job_seq)

        if outcome.invalidation is not None:
            self._write_event(outcome.invalidation)

        return True

    def sync(self, pipeline) -> int:
        """코어의 사건 목록·대기 폐기 기록 중 아직 쓰지 않은 것을 순서대로 쓴다. 쓴 줄 수를 돌려준다."""
        self._check()
        count = 0

        for event in pipeline.invalidation_events:
            count += self._write_event(event)

        drops = pipeline.queue_drop_records

        while self._drops_logged < len(drops):
            self._recorder.write("queue_drop", drops[self._drops_logged])
            self._drops_logged += 1
            count += 1

        return count

    def missing(self, pipeline) -> dict[str, int]:
        """기록이 확인되지 않은 것의 수(실패 보고용): 끝난 작업의 처리 기록, 사건, 대기 폐기.

        실패 직전 줄은 스트림에 실제로 남았을 수도 있으므로 「확인되지 않음」으로 센다(누락 또는 손상 가능).

        끝난 작업 수는 코어 계수(completed+discarded+failed+aborted)에서 센다. 계수를 바꾸지 않는다.
        """
        c = pipeline.counters
        finished = c.completed + c.generation_discarded + c.failed + c.aborted

        return {
            "process": finished - len(self._jobs_logged),
            "event": sum(1 for e in set(pipeline.invalidation_events) if e not in self._events_logged),
            "queue_drop": len(pipeline.queue_drop_records) - self._drops_logged,
        }

    def log_input_states(self, records) -> None:
        """입력 상태 구간 기록을 쓴다(프레임 ID 없음)."""
        self._check()
        for rec in records:
            self._recorder.write("input_state", rec)

    def _write_event(self, event: CandidateInvalidationEvent) -> int:
        """같은 9필드 사건은 1줄만 쓴다. 썼으면 1, 이미 썼으면 0."""
        if event in self._events_logged:
            return 0

        payload = _plain(event)
        payload["invalidation_kind"] = payload.pop("kind")
        self._recorder.write("event", payload)
        self._events_logged.add(event)

        return 1


class InputStateTracker:
    """입력 상태 구간의 지속 시간을 PC 단조 시계로 확정한다(A §9 「입력 상태 로그」).

    같은 상태가 이어지면 한 구간이다. 다른 상태로 바뀌거나, 정상 프레임이 다시 오거나, 실행이 끝날 때
    직전 구간의 지속 시간을 확정해 `InputStateRecord`로 돌려준다. VIDEO_POS·SIM_TIME 같은 영상 시각을
    빼서 지속 시간으로 쓰지 않는다. 프레임 ID를 만들지 않는다.
    """

    def __init__(self, run_id: str, source_id: str) -> None:
        """실행·소스 ID를 받는다."""
        self._run_id = run_id
        self._source_id = source_id
        self._open: InputStatusEvent | None = None

    def on_status(self, event: InputStatusEvent) -> list[InputStateRecord]:
        """입력 사건을 받는다. 상태가 바뀌면 직전 구간 기록을 돌려준다. 다른 run/source는 ValueError."""
        if event.run_id != self._run_id or event.source_id != self._source_id:
            raise ValueError("input status event from another run/source")

        status = InputStatus(event.status)

        if self._open is not None and InputStatus(self._open.status) is status:
            return []

        closed = self._close(event.event_monotonic_ns)
        self._open = event

        return closed

    def on_frame(self, now_ns: int) -> list[InputStateRecord]:
        """정상 프레임 복귀(PC 단조 시각). 열린 구간이 있으면 확정해 돌려준다."""
        return self._close(now_ns)

    def finish(self, now_ns: int) -> list[InputStateRecord]:
        """실행 종료(PC 단조 시각). 열린 구간이 있으면 확정해 돌려준다."""
        return self._close(now_ns)

    def _close(self, now_ns: int) -> list[InputStateRecord]:
        """열린 구간을 now_ns에서 닫는다."""
        if self._open is None:
            return []

        e = self._open
        self._open = None

        return [InputStateRecord(run_id=e.run_id, source_id=e.source_id, status=InputStatus(e.status),
                                 event_monotonic_ns=e.event_monotonic_ns, detail=e.detail,
                                 duration_ns=now_ns - e.event_monotonic_ns)]
