"""A→B 입력 봉투(`ObservationEnvelope`, #168): 프레임·관측·촬영 메타·촬영 시각 TF·상관 키를 한 묶음으로, 버전을 명시해 넘긴다.

근거: `_workspace/168_a_to_b_real_path_implementation_spec.md` 「전달 자료의 확장성」, A-8 끝점 계약(`PlanarObservation.axis_endpoints_px`),
`transforms/b_stage.py`(`BStage.consume(frame, obs)`, `correlation_keys`, TF 50 ms).

규칙
- 봉투 형식 이름은 `OBSERVATION_ENVELOPE_VERSION`이다. 다른 버전은 받지 않는다(`ObservationEnvelopeError`).
- 필수 필드: 원본 `FramePacket`(영상 포함), `PlanarObservation`, `CaptureInfo`(촬영 시각·광학 프레임·보정 번호·CameraInfo 해상도),
  `CaptureTf` 또는 None(그 촬영 시각 TF가 없으면 None, B는 TF_UNAVAILABLE), 상관 키(`correlation_keys`로 계산해 고정).
  끝점은 관측 안에 그대로 둔다(봉투가 복사·변형하지 않음).
- 생성 때 서로 묶인 값을 모두 대조한다: 관측 ↔ 프레임 식별자·보정 번호, 프레임 ↔ 촬영 정보(시각·광학 프레임·크기·보정 번호),
  카메라 프로파일 ↔ 촬영 정보, TF ↔ 촬영 시각·광학 프레임, 4×4 강체. 어긋나면 `ObservationEnvelopeError`(조용히 고치지 않음).
- 확장 필드(`extensions`): 이름 있는 선택 항목이다.
  - `SUPPORTED_EXTENSIONS`의 이름은 형식·좌표 기준·프레임/작업 결합을 검사해 `extensions`에 정규화해 담는다. 잘못된 형식은 `ObservationEnvelopeError`.
  - 모르는 이름은 **내용을 JSON 정규 문자열로 그대로 보존**해 `unsupported`에 담는다(JSON으로 표현할 수 없으면 `ObservationEnvelopeError`).
    의미를 해석하지 않으며 핵심 입력으로 승격하지 않는다. 볼트 끝점 경로는 그대로 평가된다.
  - 봉투 전체(확장·미지원 포함)는 B 입구 어댑터(`transforms/observation_envelope_adapter.ObservationEnvelopeAdapter`)까지 간다. 어댑터는 확장을 받아
    결과에 함께 돌려주지만 **B 계산(`BStage.consume`)에는 frame·관측만 쓴다**(의미·정확도·소유권 합의 뒤 소비).
  - 지금 A는 확장 필드를 만들지 않는다(생산은 담당·계약이 정해진 항목부터, 명세).
- 직렬화: `to_record()`는 JSON 안전한 기록(영상은 지문·모양만, 확장 값 포함)이다. pickle·ROS 메시지 직렬화를 쓰지 않는다.
- ROS·Qt를 import하지 않는다.
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

# 확장 필드 정의(이름 → 설명). 계약 전 후보이며 지금 A가 생산하지 않는다. B는 소비하지 않는다
SUPPORTED_EXTENSIONS = MappingProxyType({
    "ai_instances": "AI 인스턴스 목록(index·class_name·confidence·rejected·polygon_px RAW): 다중 검출·polygon 후보",
    "tracking": "추적 정보(part_track_id 정수 또는 None): A-6 실험 표시 필드, 중복 집기 방지 보증 아님",
})


class ObservationEnvelopeError(ValueError):
    """봉투 필수 필드·버전·결합·확장 형식 오류. B를 부르지 않고 기록한다."""


# `CaptureInfo`·`CaptureTf`는 #189에서 공통형(`common/types`)으로 옮겼다. 같은 클래스를 이 모듈에서도 그대로 내보낸다


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
    """공통형 검사 함수(`ValueError`)를 같은 메시지의 `ObservationEnvelopeError`로 바꿔 부른다(#189: 검사 규칙은 한 곳)."""
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
    """유한 실수 하나(#168 F3). bool·문자열·복소수·None·그 밖의 형은 거부, float 변환 넘침(거대 정수)·비유한도 ObservationEnvelopeError."""
    return _wrapped(_common.real_number, value, where)


def _real_array(value, where: str) -> np.ndarray:
    """실수 배열(float64 새 복사본). ndarray는 정수·실수 dtype만, 목록·튜플은 원소마다 검사한다(조용한 형 변환 없음)."""
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
    """`ai_instances`: {binding, pixel_basis, items:[{index, class_name, confidence, rejected, polygon_px|None}]}."""
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
