"""A→B 목록 봉투 `eyetoarm.b_input/2`(#189 단계 L): 같은 프레임의 유효 물체 후보 목록(`FrameObservations`)과 상관 키.

근거: A 상세설계 §5.2 「후보 목록의 필드와 타입 (B-09 협의안)」, B 협업 요청 §8.2 「B-09 B 회신 (2026-11-19)」(형식 그대로 수락,
프레임 공통 TF 하나, B가 `instance_index` 순 앞 8개만 계산), 계획 `_workspace/189_builder2_stage_L_plan.md` r0 §4.

규칙
- `b_input/1`(`vision/observation_envelope`, 관측 하나)과 별개 자료형이다. 1은 바꾸지 않는다.
- 프레임·촬영 정보·카메라·TF·후보 대조는 `FrameObservations` 생성자가 이미 했다(공통 검사 함수, 봉투 1과 같은 규칙).
  `make_list_envelope`는 버전·형만 보고 상관 키를 고정한다. 어긋나면 `ListEnvelopeError`(B를 부르지 않고 기록).
- 상관 키는 B `correlation_keys`를 그대로 불러 같은 키 8개를 만든다. 후보별 대응 열쇠는 여기에 `instance_index`를 더한다.
- 확장(`ai_instances`·`tracking`)은 두지 않는다(A가 만들지 않고 B가 쓰지 않음).
- A는 후보를 고르거나 자르지 않는다. 선택·C 송신은 B 소관이다.
- `to_record()`는 JSON 안전한 요약(영상은 지문·모양만, TF 행렬 없음 — 봉투 1과 같음)이다.
- ROS·Qt를 import하지 않는다. B 입구(`BStage.process_frame`)는 B가 만든다(이 모듈은 B 파일을 바꾸지 않음).
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
                                "calc_contour": None if c.calc_contour is None else c.calc_contour.astype(float).tolist()}
                               for c in obs.candidates]}


def _correlation(obs: FrameObservations) -> dict:
    """B `correlation_keys`와 같은 상관 키(관측 자리에 프레임 공통 값 보기 객체)."""
    view = SimpleNamespace(run_id=obs.frame.run_id, capture_generation=obs.frame.capture_generation,
                           generation=obs.generation, job_seq=obs.job_seq, detector_config_id=obs.detector_config_id)

    return correlation_keys(view, obs.frame)


def make_list_envelope(observations, *, version: str = LIST_ENVELOPE_VERSION) -> ListEnvelope:
    """목록 봉투를 만든다. 버전이 다르거나 `FrameObservations`가 아니면 ListEnvelopeError."""
    if version != LIST_ENVELOPE_VERSION:
        raise ListEnvelopeError(f"unsupported list envelope version {version!r} (expected {LIST_ENVELOPE_VERSION!r})")

    if not isinstance(observations, FrameObservations):
        raise ListEnvelopeError(f"observations must be FrameObservations: {type(observations).__name__}")

    return ListEnvelope(version=version, observations=observations,
                        correlation=MappingProxyType(_correlation(observations)))


__all__ = ["LIST_ENVELOPE_VERSION", "ListEnvelope", "ListEnvelopeError", "make_list_envelope"]
