"""저장 입력 실제 A→B 경로(#168)의 B 결과·주석 영상 기록기.

근거: `_workspace/168_a_to_b_real_path_implementation_spec.md` §3, `transforms/b_stage.py`(`BStage.consume`·`render_center`).

- `CaptureObservationRecorder.wrap(consume)`은 실행기 `b_consumer` 자리 함수다. `b_admit`(= `b_gate`)를 통과한 관측마다 실행기가 부른다.
  같은 `FramePacket`(관측을 만든 그 프레임)으로 `consume`을 부르고, **같은 `BOutput`**으로 `render_center(frame.image, [out])`를
  불러 원본 크기 주석 프레임을 만든다. 입력 배열은 바꾸지 않는다(호출 전후 지문 비교, 다르면 예외 → 실행 실패).
- 주석 영상 정책: B 결과에 `T_W_C`가 있을 때만 PNG를 쓴다. `T_W_C`가 없으면(TF 없음·시간 초과 → B SKIP) 그럴듯한 위치를
  그리지 않으며 `annotation: null`과 사유를 남긴다. B로 보내지 않은 관측(무효·`b_gate` 거부)은 `frames.jsonl`에 사유만 남고
  주석 영상이 없다(`annotation: "not_sent_to_b"`).
- `valid=True` 관측은 추출기 형식 조건 통과 뜻이다. 기록에 `valid_meaning`으로 명시한다(정확도·운영 승인 아님, #167 K0).
- 출력 폴더는 새로 만든다(이미 있으면 FileExistsError). 파일은 덮어쓰지 않는다(`open(..., "x")`).
- C·D로 보내지 않는다. `stage.on_output`을 쓰지 않는다.
- A→B 경계는 입력 봉투(`vision/observation_envelope.ObservationEnvelope`, 버전 명시)다. 촬영 정보·TF는 녹화 메타 레코드에서 만들고 프레임·관측과
  대조한다. 봉투 오류는 B를 부르지 않고 `b_input_rejected`로 기록한다(실행 끝 판정에서 종료 1). 봉투 전체는 B 입구 어댑터
  (`transforms/observation_envelope_adapter.ObservationEnvelopeAdapter`)로 간다. 어댑터는 계산에 frame·관측만 쓰고 확장은 받아서 돌려준다. TF는 봉투 것
  (`EnvelopeTfLookup`)만 쓴다. 지금 A는 확장 필드를 만들지 않는다. 받은 확장·미지원·계산 사용 여부는 `b_adapter`로 기록한다(기록 키는 기존 계약 문자열 유지).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from eyetoarm.vision.observation_envelope import ObservationEnvelopeError, CaptureInfo, CaptureTf, make_observation_envelope

VALID_MEANING = "extractor-format (not accuracy/operational approval)"


def _plain(value):
    """JSON으로 쓸 수 있는 값으로."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]

    if isinstance(value, np.ndarray):
        return _plain(value.tolist())

    if isinstance(value, np.generic):
        return _plain(value.item())

    if isinstance(value, float) and not math.isfinite(value):
        return None

    if isinstance(value, (str, int, float, bool, type(None))):
        return value

    return getattr(value, "value", str(value))


def _image_sha(image: np.ndarray) -> str:
    """영상 배열 지문(모양·형식·바이트)."""
    arr = np.ascontiguousarray(image)
    h = hashlib.sha256(f"{arr.shape}|{arr.dtype}".encode())
    h.update(arr.tobytes())

    return h.hexdigest()


def _frame_keys(frame, obs) -> dict:
    """프레임·관측 상관 키."""
    return {"run_id": obs.run_id, "source_id": obs.source_id, "image_seq": obs.image_seq,
            "source_index": getattr(frame, "source_index", None), "capture_generation": obs.capture_generation,
            "generation": obs.generation, "job_seq": obs.job_seq, "detector_config_id": obs.detector_config_id,
            "calibration_id": obs.calibration_id, "timestamp_ms": frame.timestamp_ms,
            "capture_stamp_ns": frame.ros_stamp_ns, "optical_frame": frame.optical_frame}


class _Jsonl:
    """새 JSONL 파일(덮어쓰지 않음)."""

    def __init__(self, path: Path) -> None:
        """파일을 새로 만든다."""
        self._stream = open(path, "x", encoding="utf-8")

    def write(self, record: dict) -> None:
        """한 줄."""
        self._stream.write(json.dumps(_plain(record), ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        self._stream.flush()

    def close(self) -> None:
        """닫는다."""
        self._stream.close()


class CaptureObservationRecorder:
    """B 결과(`b_outputs.jsonl`)·주석 영상(`annotated/*.png`)·terminal 응답(`frames.jsonl`) 기록기."""

    def __init__(self, out_dir: str, stage, meta, adapter, extension_source=None) -> None:
        """출력 폴더를 새로 만든다(있으면 FileExistsError).

        `meta`는 CaptureMeta, `adapter`는 그 stage의 ObservationEnvelopeAdapter. `extension_source(frame, obs)`는 봉투에 실을 확장 사전을 주는
        주입 지점이다(기본 None = 확장 없음; A는 아직 확장을 생산하지 않는다).
        """
        self._meta = meta
        self._adapter = adapter
        self._extension_source = extension_source
        self.rejected = 0
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=False, exist_ok=False)
        (self.dir / "annotated").mkdir()
        self._stage = stage
        self._b = _Jsonl(self.dir / "b_outputs.jsonl")
        self._frames = _Jsonl(self.dir / "frames.jsonl")
        self.calls = 0
        self.annotated = 0
        self.kinds: dict[str, int] = {}
        self.reasons: dict[str, int] = {}

    # ---- display_sink 규약(실행기 handle()이 terminal 응답마다 기록 뒤·B 지점 앞에 부름) ----
    def on_outcome(self, outcome, current, **_kw) -> None:
        """terminal 응답 하나를 frames.jsonl에 남긴다(관측 유효·사유·끝점). B 호출 여부는 b_outputs.jsonl과 job_seq로 맞춘다."""
        ctx, obs = outcome.context, outcome.observation
        f = ctx.frame
        record = {"kind": "outcome", "image_seq": f.image_seq, "source_index": getattr(f, "source_index", None),
                  "job_seq": ctx.job_seq, "generation": ctx.generation, "status": getattr(outcome.status, "value", None),
                  "applied": outcome.applied, "duplicate": outcome.duplicate, "display_current": bool(current),
                  "capture_stamp_ns": f.ros_stamp_ns, "calibration_id": f.calibration_id, "error": outcome.error}

        if obs is not None:
            record.update(valid=obs.valid, reason=obs.reason, observed_face=obs.observed_face,
                          selected_index=obs.selected_index,
                          axis_endpoints_px=None if obs.axis_endpoints_px is None else obs.axis_endpoints_px.tolist())
            key = "VALID" if obs.valid else obs.reason
            self.reasons[key] = self.reasons.get(key, 0) + 1

        self._frames.write(record)

    def on_input(self, *_a, **_kw) -> None:
        """실시간 입력 알림(저장 입력에서는 오지 않음)."""

    # ---- b_consumer ----
    def envelope(self, frame, obs):
        """녹화 메타 레코드로 촬영 정보·TF를 만들어 봉투를 검사한다. 오류는 ObservationEnvelopeError."""
        record = self._meta.frames.get(getattr(frame, "source_index", -1))

        if record is None:
            raise ObservationEnvelopeError(f"no capture meta for source_index {getattr(frame, 'source_index', None)}")

        cam = self._meta.camera
        capture = CaptureInfo(stamp_ns=record.stamp_ns, optical_frame=record.frame_id, calibration_id=cam.calibration_id,
                              width=cam.width, height=cam.height)
        tf = None if record.T_W_C is None else CaptureTf(self._meta.world_frame, record.tf_child, record.tf_stamp_ns,
                                                         record.T_W_C)

        extensions = None if self._extension_source is None else self._extension_source(frame, obs)

        return make_observation_envelope(frame, obs, capture=capture, camera=cam, tf=tf, world_frame=self._meta.world_frame,
                            extensions=extensions)

    def wrap(self, consume: Callable) -> Callable:
        """`b_consumer` 자리 함수. 봉투 오류는 기록만 하고 B를 부르지 않는다. B 예외·입력 배열 변경은 그대로 올린다."""
        def consumer(frame, obs):
            """봉투 검사 → B 계산 → 같은 BOutput으로 주석 영상 → 기록."""
            before = _image_sha(frame.image)

            try:
                env = self.envelope(frame, obs)
            except ObservationEnvelopeError as exc:
                self.rejected += 1
                self._b.write({"kind": "b_input_rejected", "input": _frame_keys(frame, obs), "error": str(exc)})
                return None

            adapted = self._adapter.consume(env)
            out = adapted.output
            drawable = out.T_W_C is not None
            annotation, png_sha = None, None

            if drawable:
                annotated = self._stage.render_center(frame.image, [out])

                if annotated.shape != frame.image.shape:
                    raise RuntimeError(f"annotated shape {annotated.shape} != frame {frame.image.shape}")

                ok, buf = cv2.imencode(".png", annotated)

                if not ok:
                    raise RuntimeError("png encode failed")

                name = f"annotated/{obs.image_seq:06d}_job{obs.job_seq:06d}.png"

                with open(self.dir / name, "xb") as fh:
                    fh.write(buf.tobytes())

                annotation, png_sha = name, hashlib.sha256(buf.tobytes()).hexdigest()
                self.annotated += 1

            after = _image_sha(frame.image)

            if after != before:
                raise RuntimeError("input frame image changed during B/annotation")

            self.calls += 1
            kind = out.goal.get("kind")
            self.kinds[kind] = self.kinds.get(kind, 0) + 1
            self._b.write({"kind": "b_output", "b_call": self.calls, "valid_meaning": VALID_MEANING,
                           "b_input": env.to_record(),
                           "b_adapter": {"envelope_version": adapted.envelope_version,
                                         "extensions_received": sorted(adapted.extensions),
                                         "unsupported_received": sorted(adapted.unsupported),
                                         "used_in_compute": list(adapted.used_in_compute)},
                           "input": {**_frame_keys(frame, obs), "observed_face": obs.observed_face,
                                     "selected_index": obs.selected_index,
                                     "axis_endpoints_px": None if obs.axis_endpoints_px is None
                                     else obs.axis_endpoints_px.tolist(),
                                     "frame_image_sha256": before},
                           "result_dict": out.result_dict, "goal": out.goal, "b_valid": out.result.valid,
                           "b_reason": out.result.reason, "b_reason_owner": out.result.reason_owner,
                           "b_compute_ms": out.result.compute_ms,
                           "annotation": annotation, "annotation_png_sha256": png_sha,
                           "annotation_note": None if drawable else "no T_W_C (B SKIP); not drawn",
                           "frame_image_unchanged": True})

            return out

        return consumer

    def summary(self, record: dict) -> None:
        """끝 요약을 summary.json으로(새 파일)."""
        doc = {"b_outputs": self.calls, "b_input_rejected": self.rejected, "annotated_frames": self.annotated, "goal_kinds": self.kinds,
               "observation_reasons": self.reasons, "valid_meaning": VALID_MEANING, **record}

        with open(self.dir / "summary.json", "x", encoding="utf-8") as fh:
            json.dump(_plain(doc), fh, ensure_ascii=False, indent=1, allow_nan=False)

    def close(self) -> None:
        """파일을 닫는다."""
        self._b.close()
        self._frames.close()


__all__ = ["CaptureObservationRecorder", "VALID_MEANING"]
