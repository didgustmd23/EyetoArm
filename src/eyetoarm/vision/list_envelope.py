"""A→B 목록 봉투 `eyetoarm.b_input/2`: 같은 프레임의 유효 물체 후보 목록(`FrameObservations`)과 상관 키.

후보 대조는 `FrameObservations` 생성자가 하고, 여기서는 버전·형과 TF 부모 = world 프레임을 본다(어긋나면
`ListEnvelopeError`). A는 후보를 고르거나 자르지 않는다. ROS·Qt를 import하지 않는다.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Mapping

import numpy as np

from eyetoarm.common.types import FrameObservations
from eyetoarm.transforms.b_stage import correlation_keys

LIST_ENVELOPE_VERSION = "eyetoarm.b_input/2"


class ListEnvelopeError(ValueError):
    """목록 봉투 버전·형 오류. B를 부르지 않고 기록한다."""


@dataclass(frozen=True)
class ListEnvelope:
    """검증을 마친 A→B 목록 봉투. `make_list_envelope`으로만 만든다."""

    version: str
    observations: FrameObservations
    correlation: Mapping

    def to_record(self) -> dict:
        """JSON 안전한 요약(영상 바이트·TF 행렬 없음). 후보는 원 인덱스·클래스·신뢰도·관측면·좌표."""
        obs = self.observations
        img = np.ascontiguousarray(obs.frame.image)
        h = hashlib.sha256(f"{img.shape}|{img.dtype}".encode())
        h.update(img.tobytes())

        return {"version": self.version, "observations_version": obs.version,
                "image_shape": list(img.shape), "image_sha256": h.hexdigest(),
                "capture": {"stamp_ns": obs.capture.stamp_ns, "optical_frame": obs.capture.optical_frame,
                            "calibration_id": obs.capture.calibration_id,
                            "size": [obs.capture.width, obs.capture.height]},
                "tf": None if obs.tf is None else {"parent": obs.tf.parent, "child": obs.tf.child,
                                                   "stamp_ns": obs.tf.stamp_ns},
                "correlation": dict(self.correlation),
                "source_id": obs.frame.source_id, "image_seq": obs.frame.image_seq,
                "candidate_count": len(obs.candidates),
                "candidates": [{"instance_index": c.instance_index, "class_name": c.class_name,
                                "confidence": c.confidence, "pixel_basis": c.pixel_basis.value,
                                "observed_face": c.observed_face,
                                "axis_endpoints_px": None if c.axis_endpoints_px is None else c.axis_endpoints_px.tolist(),
                                "calc_contour": None if c.calc_contour is None else c.calc_contour.astype(float).tolist(),
                                **_track_keys(c)}
                               for c in obs.candidates]}


def _track_keys(c) -> dict:
    """추적 번호를 실은 후보일 때만 번호·상태 키. 그 밖은 빈 사전(추적 끔 기록 형식 그대로)."""
    if not hasattr(c, "part_track_id"):
        return {}

    return {"part_track_id": c.part_track_id, "track_state": getattr(c, "track_state", None)}


def _correlation(obs: FrameObservations) -> dict:
    """B `correlation_keys`와 같은 상관 키(관측 자리에 프레임 공통 값 보기 객체)."""
    view = SimpleNamespace(run_id=obs.frame.run_id, capture_generation=obs.frame.capture_generation,
                           generation=obs.generation, job_seq=obs.job_seq, detector_config_id=obs.detector_config_id)

    return correlation_keys(view, obs.frame)


def make_list_envelope(observations, *, world_frame: str = "world",
                       version: str = LIST_ENVELOPE_VERSION) -> ListEnvelope:
    """목록 봉투를 만든다. 버전·형이 다르거나 TF 부모가 `world_frame`이 아니면 ListEnvelopeError.

    TF가 None이면 부모 대조를 하지 않는다.
    """
    if version != LIST_ENVELOPE_VERSION:
        raise ListEnvelopeError(f"unsupported list envelope version {version!r} (expected {LIST_ENVELOPE_VERSION!r})")

    if not isinstance(observations, FrameObservations):
        raise ListEnvelopeError(f"observations must be FrameObservations: {type(observations).__name__}")

    if observations.tf is not None and observations.tf.parent != world_frame:
        raise ListEnvelopeError(f"tf.parent {observations.tf.parent!r} != world frame {world_frame!r}")

    return ListEnvelope(version=version, observations=observations,
                        correlation=MappingProxyType(_correlation(observations)))


__all__ = ["LIST_ENVELOPE_VERSION", "ListEnvelope", "ListEnvelopeError", "make_list_envelope"]
