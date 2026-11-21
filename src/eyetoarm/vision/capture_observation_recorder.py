"""저장 입력 A→B 경로의 B 결과·주석 영상 기록기(`b_consumer` 자리 함수).

관측을 봉투로 검사해 B에 넘기고, 같은 `BOutput`으로 주석 영상을 그린다. `T_W_C`가 없으면 주석을 그리지 않고,
봉투 오류는 B를 부르지 않고 `b_input_rejected`로 남긴다. 출력 폴더는 새로 만들고 파일을 덮어쓰지 않는다.
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
from eyetoarm.vision.view_bias import apply_view_bias, summarize

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


def capture_from_meta(meta, frame, error=ValueError):
    """녹화 메타 레코드로 그 프레임의 촬영 정보·촬영 시각 TF를 만든다(메타가 없으면 `error`)."""
    record = meta.frames.get(getattr(frame, "source_index", -1))

    if record is None:
        raise error(f"no capture meta for source_index {getattr(frame, 'source_index', None)}")

    cam = meta.camera
    capture = CaptureInfo(stamp_ns=record.stamp_ns, optical_frame=record.frame_id, calibration_id=cam.calibration_id,
                          width=cam.width, height=cam.height)
    tf = None if record.T_W_C is None else CaptureTf(meta.world_frame, record.tf_child, record.tf_stamp_ns, record.T_W_C)

    return capture, tf


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

    def __init__(self, out_dir: str, stage, meta, adapter, extension_source=None, view_bias=None) -> None:
        """출력 폴더를 새로 만든다(있으면 FileExistsError).

        `extension_source(frame, obs)`는 봉투에 실을 확장 사전을 주는 주입 지점(None = 확장 없음).
        """
        self._meta = meta
        self._view_bias = view_bias
        self._view_bias_records: list = []
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

    # ---- display_sink 규약(terminal 응답마다 기록 뒤·B 지점 앞) ----
    def on_outcome(self, outcome, current, **_kw) -> None:
        """terminal 응답 하나를 frames.jsonl에 남긴다. B 호출 여부는 b_outputs.jsonl과 job_seq로 맞춘다."""
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
        capture, tf = capture_from_meta(self._meta, frame, ObservationEnvelopeError)
        cam = self._meta.camera

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

            # 공통 보정 지점(끔이면 같은 봉투)
            env, bias = apply_view_bias(env, self._meta.camera, self._view_bias)

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
                                     "frame_image_sha256": before, **self._bias_input(env, bias)},
                           "result_dict": out.result_dict, "goal": out.goal, "b_valid": out.result.valid,
                           "b_reason": out.result.reason, "b_reason_owner": out.result.reason_owner,
                           "b_compute_ms": out.result.compute_ms,
                           "annotation": annotation, "annotation_png_sha256": png_sha,
                           "annotation_note": None if drawable else "no T_W_C (B SKIP); not drawn",
                           "frame_image_unchanged": True})

            return out

        return consumer

    def _bias_input(self, env, bias: list) -> dict:
        """`input`에 더할 보정 키(켬일 때만). 끝점을 B에 넘긴 값으로 덮고 원 끝점·보정 결과를 더한다."""
        if self._view_bias is None:
            return {}

        self._view_bias_records.append(bias)
        vb = bias[0]
        obs = env.observation

        return {"axis_endpoints_px": None if obs.axis_endpoints_px is None else obs.axis_endpoints_px.tolist(),
                "axis_endpoints_px_raw": vb["axis_endpoints_px_raw"],
                "view_bias": {k: vb[k] for k in ("applied", "s", "delta_mm", "reason")}}

    def summary(self, record: dict) -> None:
        """끝 요약을 summary.json으로(새 파일)."""
        doc = {"b_outputs": self.calls, "b_input_rejected": self.rejected, "annotated_frames": self.annotated, "goal_kinds": self.kinds,
               "observation_reasons": self.reasons, "valid_meaning": VALID_MEANING, **record}

        if self._view_bias is not None:
            doc["view_bias"] = {"enabled": True, **self._view_bias.to_json(), **summarize(self._view_bias_records)}

        with open(self.dir / "summary.json", "x", encoding="utf-8") as fh:
            json.dump(_plain(doc), fh, ensure_ascii=False, indent=1, allow_nan=False)

    def close(self) -> None:
        """파일을 닫는다."""
        self._b.close()
        self._frames.close()


__all__ = ["CaptureObservationRecorder", "VALID_MEANING", "capture_from_meta"]
