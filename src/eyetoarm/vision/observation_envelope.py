"""A→B 입력 봉투(`ObservationEnvelope`): 프레임·관측·촬영 메타·촬영 시각 TF·상관 키를 버전을 명시해 한 묶음으로 넘긴다.

생성 때 서로 묶인 값을 모두 대조하고 어긋나면 `ObservationEnvelopeError`(조용히 고치지 않음). 모르는 확장은 JSON 정규
문자열로 보존만 하고 해석하지 않는다. pickle·ROS 직렬화를 쓰지 않고 ROS·Qt를 import하지 않는다.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

from eyetoarm.calibration.profile import CameraProfile
from eyetoarm.common import types as _common
from eyetoarm.common.types import CaptureInfo, CaptureTf, FramePacket, PixelBasis, PlanarObservation
from eyetoarm.transforms.b_stage import correlation_keys

OBSERVATION_ENVELOPE_VERSION = "eyetoarm.b_input/1"

# 확장 필드 정의(이름 → 설명). A가 생산하지 않고 B가 소비하지 않는 후보다
SUPPORTED_EXTENSIONS = MappingProxyType({
    "ai_instances": "AI 인스턴스 목록(index·class_name·confidence·rejected·polygon_px RAW): 다중 검출·polygon 후보",
    "tracking": "추적 정보(part_track_id 정수 또는 None): A-6 실험 표시 필드, 중복 집기 방지 보증 아님",
})


class ObservationEnvelopeError(ValueError):
    """봉투 필수 필드·버전·결합·확장 형식 오류. B를 부르지 않고 기록한다."""


# 공통형 `CaptureInfo`·`CaptureTf`를 이 모듈에서도 그대로 내보낸다


@dataclass(frozen=True)
class ObservationEnvelope:
    """검증을 마친 A→B 입력 봉투. `make_observation_envelope`으로만 만든다."""

    version: str
    frame: FramePacket
    observation: PlanarObservation
    capture: CaptureInfo
    tf: CaptureTf | None
    correlation: Mapping
    extensions: Mapping = field(default_factory=dict)   # 지원 확장(이름 → 정규화 값)
    unsupported: Mapping = field(default_factory=dict)  # 모르는 확장(이름 → JSON 정규 문자열, 해석·소비 안 함)

    def to_record(self) -> dict:
        """JSON 안전한 요약(영상 바이트·확장 내용 없음)."""
        img = np.ascontiguousarray(self.frame.image)
        h = hashlib.sha256(f"{img.shape}|{img.dtype}".encode())
        h.update(img.tobytes())

        return {"version": self.version, "image_shape": list(img.shape), "image_sha256": h.hexdigest(),
                "capture": {"stamp_ns": self.capture.stamp_ns, "optical_frame": self.capture.optical_frame,
                            "calibration_id": self.capture.calibration_id,
                            "size": [self.capture.width, self.capture.height]},
                "tf": None if self.tf is None else {"parent": self.tf.parent, "child": self.tf.child,
                                                    "stamp_ns": self.tf.stamp_ns},
                "correlation": dict(self.correlation),
                "extensions": {name: _jsonable(value) for name, value in sorted(self.extensions.items())},
                "unsupported": {name: json.loads(text) for name, text in sorted(self.unsupported.items())}}


def _jsonable(value):
    """확장 정규 값 → JSON 안전 값(배열은 목록, 튜플은 목록)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


def _wrapped(check, *args):
    """공통형 검사 함수의 `ValueError`를 같은 메시지의 `ObservationEnvelopeError`로 바꿔 부른다."""
    try:
        return check(*args)
    except ObservationEnvelopeError:
        raise
    except ValueError as exc:
        raise ObservationEnvelopeError(str(exc)) from exc


def _int(value, where: str) -> int:
    """bool이 아닌 0 이상 정수."""
    return _wrapped(_common.nonneg_int, value, where)


def _real(value, where: str) -> float:
    """유한 실수 하나. bool·문자열·복소수·None·넘침·비유한은 ObservationEnvelopeError."""
    return _wrapped(_common.real_number, value, where)


def _real_array(value, where: str) -> np.ndarray:
    """실수 배열(float64 새 복사본). 조용한 형 변환 없이 dtype·원소를 검사한다."""
    return _wrapped(_common.real_array, value, where)


def _text(value, where: str) -> str:
    """비어 있지 않은 문자열."""
    return _wrapped(_common.nonempty_text, value, where)


def _rigid(T, where: str) -> np.ndarray:
    """4×4 유한 강체변환의 읽기 전용 복사본(원소 형 엄격 검사)."""
    return _wrapped(_common.rigid_transform, T, where)


def _check_core(frame, obs, capture: CaptureInfo, camera: CameraProfile) -> None:
    """프레임·관측·촬영 정보·카메라 프로파일 결합을 대조한다."""
    if not isinstance(frame, FramePacket):
        raise ObservationEnvelopeError(f"frame must be FramePacket: {type(frame).__name__}")

    if not isinstance(obs, PlanarObservation):
        raise ObservationEnvelopeError(f"observation must be PlanarObservation: {type(obs).__name__}")

    _wrapped(_common.check_frame_image, frame)

    for name in ("run_id", "source_id", "image_seq", "capture_generation", "calibration_id"):
        if getattr(obs, name) != getattr(frame, name):
            raise ObservationEnvelopeError(f"observation.{name} {getattr(obs, name)!r} != frame.{name} {getattr(frame, name)!r}")

    if obs.pixel_basis not in (PixelBasis.RAW, PixelBasis.UNDISTORTED):
        raise ObservationEnvelopeError(f"observation.pixel_basis unsupported: {obs.pixel_basis!r}")

    _wrapped(_common.check_frame_capture, frame, capture)

    if not isinstance(camera, CameraProfile):
        raise ObservationEnvelopeError("camera profile (calibration_id/frame_id/size) does not match capture info")

    _wrapped(_common.check_camera_capture, camera, capture)


def _check_tf(tf, capture: CaptureInfo, world_frame: str) -> CaptureTf | None:
    """촬영 시각 TF 결합 대조와 정규화."""
    return _wrapped(_common.check_capture_tf, tf, capture, world_frame)


def _check_binding(value: dict, obs: PlanarObservation, capture: CaptureInfo, where: str) -> None:
    """확장의 프레임·작업 결합(image_seq·job_seq·generation·capture_stamp_ns)이 봉투와 같은지."""
    binding = value.get("binding")

    if not isinstance(binding, dict) or set(binding) != {"image_seq", "job_seq", "generation", "capture_stamp_ns"}:
        raise ObservationEnvelopeError(f"{where}.binding must have image_seq/job_seq/generation/capture_stamp_ns")

    expected = {"image_seq": obs.image_seq, "job_seq": obs.job_seq, "generation": obs.generation,
                "capture_stamp_ns": capture.stamp_ns}

    # 형까지 같아야 한다(True == 1 같은 느슨한 일치 거부)
    for key, val in binding.items():
        _int(val, f"{where}.binding.{key}")

    if binding != expected:
        raise ObservationEnvelopeError(f"{where}.binding {binding} != envelope {expected}")


def _ext_ai_instances(value, obs, capture, frame) -> dict:
    """`ai_instances` 확장 검사(binding·pixel_basis·items 목록, 항목 polygon_px는 None 허용)."""
    if not isinstance(value, dict) or set(value) != {"binding", "pixel_basis", "items"}:
        raise ObservationEnvelopeError("ai_instances must have binding/pixel_basis/items")

    _check_binding(value, obs, capture, "ai_instances")

    if value["pixel_basis"] != PixelBasis(obs.pixel_basis).value:
        raise ObservationEnvelopeError(f"ai_instances.pixel_basis {value['pixel_basis']!r} != observation pixel_basis")

    if not isinstance(value["items"], list):
        raise ObservationEnvelopeError("ai_instances.items must be a list")

    items = []

    for k, item in enumerate(value["items"]):
        where = f"ai_instances.items[{k}]"

        if not isinstance(item, dict) or set(item) != {"index", "class_name", "confidence", "rejected", "polygon_px"}:
            raise ObservationEnvelopeError(f"{where} keys must be index/class_name/confidence/rejected/polygon_px")

        conf = _real(item["confidence"], f"{where}.confidence")

        if not 0.0 <= conf <= 1.0:
            raise ObservationEnvelopeError(f"{where}.confidence must be in [0, 1]")

        poly = item["polygon_px"]

        if poly is not None:
            arr = _real_array(poly, f"{where}.polygon_px")

            if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3 or not np.isfinite(arr).all():
                raise ObservationEnvelopeError(f"{where}.polygon_px must be finite (N>=3, 2)")

            if value["pixel_basis"] == PixelBasis.RAW.value and (
                    (arr[:, 0] < 0).any() or (arr[:, 0] > frame.width - 1).any()
                    or (arr[:, 1] < 0).any() or (arr[:, 1] > frame.height - 1).any()):
                raise ObservationEnvelopeError(f"{where}.polygon_px out of frame")

            arr.setflags(write=False)
            poly = arr

        if item["rejected"] is not None:
            _text(item["rejected"], f"{where}.rejected")

        items.append({"index": _int(item["index"], f"{where}.index"),
                      "class_name": _text(item["class_name"], f"{where}.class_name"),
                      "confidence": conf, "rejected": item["rejected"], "polygon_px": poly})

    return {"binding": dict(value["binding"]), "pixel_basis": value["pixel_basis"], "items": tuple(items)}


def _ext_tracking(value, obs, capture, frame) -> dict:
    """`tracking`: {binding, part_track_id: int|None}."""
    if not isinstance(value, dict) or set(value) != {"binding", "part_track_id"}:
        raise ObservationEnvelopeError("tracking must have binding/part_track_id")

    _check_binding(value, obs, capture, "tracking")
    track = value["part_track_id"]

    if track is not None:
        _int(track, "tracking.part_track_id")

    return {"binding": dict(value["binding"]), "part_track_id": track}


_EXTENSION_CHECKS = {"ai_instances": _ext_ai_instances, "tracking": _ext_tracking}


def make_observation_envelope(frame, observation, *, capture: CaptureInfo, camera: CameraProfile, tf: CaptureTf | None,
                 world_frame: str = "world", extensions: Mapping | None = None,
                 version: str = OBSERVATION_ENVELOPE_VERSION) -> ObservationEnvelope:
    """봉투를 만들고 검사한다. 버전·필수 필드·결합·확장 형식 오류는 ObservationEnvelopeError."""
    if version != OBSERVATION_ENVELOPE_VERSION:
        raise ObservationEnvelopeError(f"unsupported a_to_b_envelope version {version!r} (expected {OBSERVATION_ENVELOPE_VERSION!r})")

    if not isinstance(capture, CaptureInfo):
        raise ObservationEnvelopeError(f"capture must be CaptureInfo: {type(capture).__name__}")

    _check_core(frame, observation, capture, camera)
    tf_checked = _check_tf(tf, capture, world_frame)
    accepted, unsupported = {}, {}

    if extensions is not None:
        if not isinstance(extensions, Mapping):
            raise ObservationEnvelopeError("extensions must be a mapping of name -> value")

        for name, value in extensions.items():
            if not isinstance(name, str) or not name:
                raise ObservationEnvelopeError(f"extension name must be a non-empty string: {name!r}")

            check = _EXTENSION_CHECKS.get(name)

            if check is None:
                try:
                    unsupported[name] = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                                                   allow_nan=False)
                except (TypeError, ValueError) as exc:
                    raise ObservationEnvelopeError(f"unsupported extension {name!r} is not JSON-representable: {exc}") from exc

                continue

            accepted[name] = check(value, observation, capture, frame)

    return ObservationEnvelope(version=version, frame=frame, observation=observation, capture=capture, tf=tf_checked,
                          correlation=MappingProxyType(correlation_keys(observation, frame)),
                          extensions=MappingProxyType(accepted), unsupported=MappingProxyType(unsupported))


__all__ = ["OBSERVATION_ENVELOPE_VERSION", "ObservationEnvelope", "ObservationEnvelopeError", "CaptureInfo", "CaptureTf", "SUPPORTED_EXTENSIONS",
           "make_observation_envelope"]
