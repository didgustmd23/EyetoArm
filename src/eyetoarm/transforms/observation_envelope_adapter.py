"""B 입구 어댑터(#168 F1): A→B 입력 봉투(`vision/observation_envelope.ObservationEnvelope`) 전체를 받아 B 계산과 확장 자료 수령을 나눈다.

근거: `_workspace/168_a_to_b_real_path_implementation_spec.md` 「전달 자료의 확장성」, reviewer #168 검토 F1 수정 명세.
**B 담당 검토 대상 제안**이다(B 파일 `b_stage.py`는 바꾸지 않고 새 파일로 둠).

- `ObservationEnvelopeAdapter.consume(envelope)`
  1. 버전·타입을 확인한다(봉투가 아니거나 다른 버전이면 `ObservationEnvelopeError`).
  2. 봉투 TF를 `EnvelopeTfLookup`에 그 소비 동안만 건다(같은 프레임 객체·같은 stamp·같은 광학 프레임일 때만, 50 ms 초과는 None).
  3. B 계산은 지금 계약대로 `stage.consume(frame, observation)`만 부른다.
  4. 지원 확장(`envelope.extensions`)과 미지원 확장(`envelope.unsupported`, JSON 문자열)을 **그대로** 결과(`ObservationAdapterResult`)에 담아
     돌려준다. `used_in_compute`는 항상 빈 튜플이다. 의미·정확도·소유권이 합의된 확장만 이후에 계산 입력으로 바꾼다.
- 새 자료가 생겨도 이 어댑터가 봉투 전체를 받으므로 B 공개 API(`BStage.consume`)를 바꾸지 않고 확장 수령을 늘릴 수 있다.
- ROS를 import하지 않는다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from eyetoarm.transforms.b_stage import BOutput
from eyetoarm.vision.observation_envelope import OBSERVATION_ENVELOPE_VERSION, ObservationEnvelope, ObservationEnvelopeError

TF_WAIT_LIMIT_S = 0.05   # B 시간 상한: TF 대기 50 ms (B 협업 요청 §8.2 A-7)


class EnvelopeTfLookup:
    """B `tf_lookup` 자리: 지금 소비 중인 봉투의 촬영 시각 TF만 돌려준다(다른 프레임·최신 TF 없음, 50 ms 초과는 None)."""

    def __init__(self, *, timeout_s: float = TF_WAIT_LIMIT_S, clock: Callable[[], float] = time.monotonic) -> None:
        """상한과 시계."""
        self.timeout_s = timeout_s
        self._clock = clock
        self._current: ObservationEnvelope | None = None
        self.calls = 0
        self.unavailable = 0
        self.timed_out = 0
        self.wait_ms: list[float] = []

        def from_envelope(frame):
            """봉투 TF: 같은 프레임 객체·같은 촬영 시각·같은 광학 프레임일 때만."""
            env = self._current

            if env is None or env.tf is None or frame is not env.frame or frame.ros_stamp_ns != env.tf.stamp_ns \
                    or frame.optical_frame != env.tf.child:
                return None

            return env.tf.T_W_C

        self.inner = from_envelope

    def __call__(self, frame) -> np.ndarray | None:
        """조회. 상한 초과·없음은 None."""
        self.calls += 1
        t0 = self._clock()
        T = self.inner(frame)
        waited = self._clock() - t0
        self.wait_ms.append(waited * 1000.0)

        if waited > self.timeout_s:
            self.timed_out += 1
            self.unavailable += 1
            return None

        if T is None:
            self.unavailable += 1

        return T


@dataclass(frozen=True)
class ObservationAdapterResult:
    """봉투 하나의 B 처리 결과: B 출력 + 받은 확장(계산 미사용)."""

    output: BOutput
    envelope_version: str
    extensions: Mapping            # 지원 확장(정규화 값 그대로)
    unsupported: Mapping           # 미지원 확장(JSON 문자열 그대로)
    used_in_compute: tuple = ()    # B 계산에 쓴 확장 이름(지금은 항상 비어 있음)


class ObservationEnvelopeAdapter:
    """B 입구: 봉투 전체를 받는다. 계산은 frame·관측만, 확장은 수령·보존만."""

    def __init__(self, stage, tf_lookup: EnvelopeTfLookup) -> None:
        """B 단계와 그 단계의 tf_lookup(같은 EnvelopeTfLookup 객체)."""
        if getattr(stage, "tf_lookup", None) is not tf_lookup:
            raise ValueError("stage.tf_lookup must be the adapter's EnvelopeTfLookup (single TF source)")

        self.stage = stage
        self.tf_lookup = tf_lookup
        self.received_extensions: dict[str, int] = {}
        self.received_unsupported: dict[str, int] = {}

    def consume(self, envelope: ObservationEnvelope) -> ObservationAdapterResult:
        """봉투 하나를 처리한다. 봉투가 아니거나 버전이 다르면 ObservationEnvelopeError(B를 부르지 않음)."""
        if not isinstance(envelope, ObservationEnvelope) or envelope.version != OBSERVATION_ENVELOPE_VERSION:
            raise ObservationEnvelopeError(f"B adapter requires {OBSERVATION_ENVELOPE_VERSION} envelope")

        self.tf_lookup._current = envelope

        try:
            out = self.stage.consume(envelope.frame, envelope.observation)
        finally:
            self.tf_lookup._current = None

        for name in envelope.extensions:
            self.received_extensions[name] = self.received_extensions.get(name, 0) + 1

        for name in envelope.unsupported:
            self.received_unsupported[name] = self.received_unsupported.get(name, 0) + 1

        return ObservationAdapterResult(output=out, envelope_version=envelope.version, extensions=envelope.extensions,
                              unsupported=envelope.unsupported, used_in_compute=())


__all__ = ["ObservationAdapterResult", "ObservationEnvelopeAdapter", "EnvelopeTfLookup", "TF_WAIT_LIMIT_S"]
