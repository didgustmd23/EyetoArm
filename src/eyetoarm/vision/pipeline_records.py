"""파이프라인 처리 기록·사건·대기 폐기·입력 상태를 JSONL(`format_version` 1)로 쓴다.

한 줄 = 기록 하나(`kind`로 구분). 작업마다 최초 terminal 응답 1줄, 같은 사건 1줄만 쓰고 duplicate는 쓰지 않는다.
쓰기 실패가 한 번이라도 나면 기록기는 실패 상태가 되어 이후 쓰기를 거부한다(fail-stop, 재시도 없음).
"""
from __future__ import annotations

import dataclasses
import json
import platform
from enum import Enum
from typing import Any

from eyetoarm.common.types import CandidateInvalidationEvent, InputStatus, InputStatusEvent
from eyetoarm.vision.ai_config import AiSegConfig
from eyetoarm.vision.ai_convert import AXIS_DUP_EP_PX, AXIS_DUP_IOU, list_candidate_indices
from eyetoarm.vision.ai_result import DUPLICATE_MASK, AiSegResult, axis_candidate_reason
from eyetoarm.vision.pipeline_types import (
    InputStateRecord,
    JobOutcome,
    JobStatus,
    ProcessRecord,
)
from eyetoarm.vision.tracking import track_record

FORMAT_VERSION = 1

# A가 판정하는 사유. 그 밖의 값은 주체를 적지 않는다
_A_REASONS = frozenset({
    "NO_OBJECT", "MULTIPLE_OBJECTS", "CONTOUR_CLIPPED", "TOP_FACE_UNRESOLVED", "IMAGE_ASPECT_LOW",
    "CALIBRATION_MISMATCH", "MARKER_NOT_FOUND", "PNP_FAILED",
})


def _check_eval_run_id(eval_run_id) -> str | None:
    """None 또는 비어 있지 않은 문자열인지 확인한다. 아니면 ValueError(가짜 ID를 만들지 않음)."""
    if eval_run_id is not None and (not isinstance(eval_run_id, str) or not eval_run_id):
        raise ValueError(f"eval_run_id must be None or a non-empty string: {eval_run_id!r}")

    return eval_run_id


B_INPUT_SINGLE = "single"   # 단계 T(기본): 관측 하나(`eyetoarm.b_input/1`)
B_INPUT_LIST = "list"       # 단계 L: 같은 프레임 후보 목록(`eyetoarm.b_input/2`)
B_INPUT_MODES = (B_INPUT_SINGLE, B_INPUT_LIST)


def process_record(outcome: JobOutcome, *, eval_run_id: str | None = None,
                   b_input: str = B_INPUT_SINGLE) -> ProcessRecord:
    """최초 terminal 응답을 처리 기록으로 바꾼다. 시작 당시 context를 그대로 쓴다.

    duplicate 응답은 ValueError(재기록 금지).
    """
    if b_input not in B_INPUT_MODES:
        raise ValueError(f"b_input must be one of {B_INPUT_MODES}: {b_input!r}")

    if outcome.duplicate:
        raise ValueError("duplicate outcome must not be recorded again")

    job = outcome.context
    f = job.frame
    obs = outcome.observation if outcome.status is JobStatus.COMPLETED else None
    reason = obs.reason if obs is not None else None
    cfg = job.settings.detector_config
    is_ai = isinstance(cfg, AiSegConfig)
    display = outcome.display_result if outcome.status is JobStatus.COMPLETED else None
    ai = display if is_ai and isinstance(display, AiSegResult) else None
    axis_on = ai is not None and bool(getattr(cfg, "axis_extractor", None))

    # 실행기와 같은 함수로 목록을 구성한다(프레임 차단이면 빈 목록)
    listed = (list_candidate_indices(ai, f, job.settings.profile)
              if axis_on and b_input == B_INPUT_LIST else None)

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
        command_id=None, eval_run_id=_check_eval_run_id(eval_run_id),
        method="ai" if is_ai else "opencv", model_id=cfg.model_id if is_ai else None,
        infer_ms=ai.infer_ms if ai is not None else None,
        ai_instances=None if ai is None else tuple(_instance_summary(i) for i in ai.instances),
        axis_valid_count=ai.axis_valid_count if axis_on else None,
        list_candidate_count=None if listed is None else len(listed), list_indices=listed,
        axis_duplicates_merged=_duplicates_merged(ai) if axis_on else None)


def _duplicates_merged(ai) -> int | None:
    """같은 볼트 중복으로 합쳐 빠진 인스턴스 수(0이면 None: 키를 쓰지 않아 기존 기록과 바이트 동일)."""
    n = sum(1 for i in ai.instances if i.rejected == DUPLICATE_MASK)

    return n or None


def _instance_summary(inst) -> dict:
    """필수 AI 기록의 인스턴스 요약(다각형·박스 없음). 축 경로가 다룬 인스턴스만 `axis_*` 키를 더한다."""
    out = {"index": inst.index, "class_index": inst.class_index, "class_name": inst.class_name,
           "confidence": inst.confidence, "rejected": inst.rejected, "clipped": inst.clipped,
           "area_px": inst.area_px}

    if getattr(inst, "axis_evaluated", False):
        out.update(axis_endpoints_px=None if inst.axis_endpoints_px is None else inst.axis_endpoints_px.tolist(),
                   axis_reason=inst.axis_reason, axis_max_iou=inst.axis_max_iou, axis_diag=inst.axis_diag,
                   axis_overlap_with=None if inst.axis_overlap_with is None else list(inst.axis_overlap_with),
                   axis_candidate_reason=axis_candidate_reason(inst))   # None = 유효 후보

    return out


def _instance_geometry(inst) -> dict:
    """선택 geometry 기록의 인스턴스 기하(원본 픽셀)."""
    return {"index": inst.index, "box_xyxy": list(inst.box_xyxy),
            "polygon": None if inst.polygon is None else inst.polygon.astype(float).tolist(),
            "component_count": inst.component_count, "hole_count": inst.hole_count}


def _detector_block(cfg) -> dict:
    """run_meta의 검출 설정 묶음."""
    if not isinstance(cfg, AiSegConfig):
        return {"method": "opencv", "config_id": cfg.config_sha256}

    block = {"method": "ai", "config_id": cfg.config_sha256, "model_id": cfg.model_id,
             "model_family": cfg.model_family, "weights_path": cfg.weights_path,
             "weights_sha256": cfg.weights_sha256, "class_map": {str(i): n for i, n in cfg.class_map},
             "target_classes": list(cfg.target_classes), "conf_threshold": cfg.conf_threshold,
             "iou_threshold": cfg.iou_threshold, "imgsz": cfg.imgsz, "max_det": cfg.max_det, "device": cfg.device,
             "half": cfg.half, "retina_masks": cfg.retina_masks, "roi": None if cfg.roi is None else list(cfg.roi),
             "min_mask_area_px": cfg.min_mask_area_px, "border_margin_px": cfg.border_margin_px}

    # 같은 볼트 중복 합치기 문턱(#292, ai_convert 코드 상수라 설정 지문에 들지 않음). 축 경로 설정만 남긴다
    if getattr(cfg, "axis_extractor", None):
        block["axis_dup"] = {"iou": AXIS_DUP_IOU, "ep_px": AXIS_DUP_EP_PX}

    # 시선 기반 끝점 보정은 켠 설정만 남긴다
    if getattr(cfg, "axis_view_bias", None) is not None:
        block["axis_view_bias"] = cfg.axis_view_bias.to_json()

    return block


def _plain(value: Any) -> Any:
    """기록용 값으로 바꾼다: 열거는 값, 튜플은 목록, dataclass는 dict. 배열은 허용하지 않는다."""
    if isinstance(value, Enum):
        return value.value

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        # omit_if_none 필드는 None이면 키를 쓰지 않는다(기록 바이트 호환)
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)
                if not (f.metadata.get("omit_if_none") and getattr(value, f.name) is None)}

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
    """기록 하나를 JSON 한 줄로 스트림에 쓰고 성공·실패 줄 수를 센다(코어 계수와 별개).

    쓰기·flush 예외나 짧은 쓰기 뒤에는 실패 상태가 되어 이후 쓰기는 `RecorderFailedError`다.
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
        """기록을 직렬화해 쓴다. 직렬화 오류는 ValueError·TypeError, 쓰기 실패는 원래 예외."""
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

    def __init__(self, recorder: JsonlRecorder, *, eval_run_id: str | None = None, log_geometry: bool = False,
                 b_input: str = B_INPUT_SINGLE, infer_process: bool = False,
                 infer_process_source: str | None = None, arm_region_filter: dict | None = None,
                 belt_region_filter: dict | None = None) -> None:
        """기록기, 평가 실행 ID(평가가 아니면 None), AI geometry 기록 여부, B 입력 단계, 추론 프로세스 사용 여부·출처를 받는다.

        출처는 None·"argument"·"default". 켬이거나 인자로 끈 경우에만 run_meta에 표시한다(기본 끔이면 바이트 동일).
        `arm_region_filter`(팔 영역 제외)·`belt_region_filter`(벨트 밖 제외) 표시 사전은 주어졌을 때만 run_meta에 넣는다.
        """
        if not isinstance(log_geometry, bool):
            raise ValueError(f"log_geometry must be a bool: {log_geometry!r}")

        if not isinstance(infer_process, bool):
            raise ValueError(f"infer_process must be a bool: {infer_process!r}")

        if infer_process_source not in (None, "argument", "default"):
            raise ValueError(f"infer_process_source must be None, 'argument' or 'default': {infer_process_source!r}")

        self._infer_process, self._infer_source = infer_process, infer_process_source

        if arm_region_filter is not None and not isinstance(arm_region_filter, dict):
            raise ValueError(f"arm_region_filter must be a dict or None: {arm_region_filter!r}")

        self._arm_region = arm_region_filter

        if belt_region_filter is not None and not isinstance(belt_region_filter, dict):
            raise ValueError(f"belt_region_filter must be a dict or None: {belt_region_filter!r}")

        self._belt_region = belt_region_filter

        if b_input not in B_INPUT_MODES:
            raise ValueError(f"b_input must be one of {B_INPUT_MODES}: {b_input!r}")

        self._b_input = b_input

        self._log_geometry = log_geometry
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
            "detector": _detector_block(s.detector_config),
        }

        # 단계 L만 표시한다. 단계 T run_meta는 키를 더하지 않는다(바이트 동일)
        if self._b_input == B_INPUT_LIST:
            meta["b_input"] = B_INPUT_LIST

        # 추론 프로세스는 켬이거나 인자로 껐을 때만 표시한다(기본 끔이면 바이트 동일). 설정 지문과는 무관하다
        if (self._infer_process or self._infer_source == "argument") and isinstance(meta["detector"], dict):
            mark = {"infer_process": self._infer_process}

            if self._infer_source is not None:
                mark["infer_process_source"] = self._infer_source

            meta["detector"] = {**meta["detector"], **mark}

        if self._arm_region is not None:
            meta["arm_region_filter"] = dict(self._arm_region)

        if self._belt_region is not None:
            meta["belt_region_filter"] = dict(self._belt_region)

        self._recorder.write("run_meta", meta)

    def write_model_ready(self, info: dict) -> None:
        """AI 모델 준비 성공 1줄(`AiSegDetector.prepare` 반환 정보 그대로). 실패 상태면 RecorderFailedError."""
        self._check()

        if not isinstance(info, dict):
            raise TypeError(f"model_ready info must be a dict: {type(info).__name__}")

        self._recorder.write("model_ready", info)

    def write_infer_process(self, summary: dict) -> None:
        """추론 프로세스 요약 1줄(`infer_process`, 실행 끝). 실패 상태면 RecorderFailedError."""
        self._check()

        if not isinstance(summary, dict):
            raise TypeError(f"infer_process summary must be a dict: {type(summary).__name__}")

        self._recorder.write("infer_process", summary)

    def write_live_input(self, record: dict) -> None:
        """실시간 소스의 경계·알림 반영 1줄(`live_input`). 반영 순서가 기록 순서다."""
        self._check()

        if not isinstance(record, dict):
            raise TypeError(f"live_input record must be a dict: {type(record).__name__}")

        self._recorder.write("live_input", record)

    def write_ai_tracks(self, tracks) -> None:
        """추적 갱신 1줄(`ai_tracks`)을 process 줄 뒤에 쓴다. 번호는 표시 실험 값(신원 보장 없음)."""
        self._recorder.write("ai_tracks", track_record(tracks))

    def write_tracking_failed(self, record: dict) -> None:
        """추적 예외 1줄(`tracking_failed`): 추적을 끈 사실과 사유·작업 식별자. 계수와 무관."""
        self._recorder.write("tracking_failed", record)

    def write_source_stats(self, stats: dict) -> None:
        """실시간 소스 통계 1줄(`source_stats`, 종료 때)."""
        self._check()

        if not isinstance(stats, dict):
            raise TypeError(f"source_stats must be a dict: {type(stats).__name__}")

        self._recorder.write("source_stats", stats)

    def log_outcome(self, outcome: JobOutcome) -> bool:
        """최초 terminal 응답의 처리 기록과 그 사건을 1회 쓴다. duplicate·이미 쓴 작업이면 False(무기록)."""
        self._check()
        if outcome.duplicate or outcome.context.job_seq in self._jobs_logged:
            return False

        self._recorder.write("process", process_record(outcome, eval_run_id=self._eval_run_id, b_input=self._b_input))
        self._jobs_logged.add(outcome.context.job_seq)
        display = outcome.display_result

        # 선택 geometry: AI 완료 작업만, process 바로 다음 줄(같은 1회 규칙)
        if self._log_geometry and outcome.status is JobStatus.COMPLETED and isinstance(display, AiSegResult):
            job = outcome.context
            self._recorder.write("ai_geometry", {
                "run_id": job.frame.run_id, "source_id": job.frame.source_id, "image_seq": job.frame.image_seq,
                "generation": job.generation, "job_seq": job.job_seq, "detector_config_id": job.detector_config_id,
                "instances": [_instance_geometry(i) for i in display.instances]})

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
        """기록이 확인되지 않은 처리·사건·대기 폐기 수(실패 보고용). 코어 계수는 바꾸지 않는다."""
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
    """입력 상태 구간의 지속 시간을 PC 단조 시계로 확정한다(영상 시각은 쓰지 않음).

    상태가 바뀌거나 정상 프레임이 오거나 실행이 끝날 때 직전 구간을 `InputStateRecord`로 돌려준다.
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
