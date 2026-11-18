"""실시간 A→B 봉투 공급자(#171 R3): 실시간 프레임 + 관측 + CameraProfile + 정확 촬영 stamp TF → 기존 `ObservationEnvelope`.

근거: `_workspace/171_builder2_live_a_to_b_plan.md` r3 §1.2·§1.3·§3.5, `vision/observation_envelope.py`(#168 봉투·검사 그대로 재사용).
**준비 부품이며 실시간 실행 경로에 연결하지 않는다**(D13 유지).

- `LiveCaptureTfLookup(lookup, world_frame=, wait_limit_s=)`: 프레임의 `ros_stamp_ns`·`optical_frame`으로 `lookup(parent, child, stamp_ns,
  timeout_s)`를 **한 번** 부른다(최신값 `Time(0)` 대체 없음). 시간은 호출 직전 → 반환 직후를 단조 시계로 잰다(대기열 대기 미포함).
  반환이 `wait_limit_s`를 넘으면 성공해도 `late`(tf=None)다. tf2는 벽시계 20 ms 간격으로 확인해 상한을 넘겨 반환할 수 있기 때문이다
  (계획 §3.5). 반환 stamp가 요청 stamp와 다르면 `stamp_mismatch`(tf=None). 예외는 `unavailable`.
  `wait_limit_s`에는 기본값이 없다(B A-7 50 ms, 판정 방식은 결정 L-3 전 시험 후보).
- `make_live_envelope(frame, obs, camera=, tf_result=)`: `CaptureInfo`를 프레임 값으로 만들고 기존 `make_observation_envelope`로 검사한다.
  실시간에서는 프레임↔촬영 정보 비교가 항상 같고, 실제 판별은 관측↔프레임과 카메라 프로파일↔프레임에서 일어난다(계획 §1.2).
- B1(#173): `ros_stamp_ns`가 양의 정수가 아니면(0·음수·bool·float 등) 조회 함수를 부르지 않고 `invalid_stamp`다. tf2 시각 0은 최신 조회라
  호출 자체가 최신 대체 금지 위반이기 때문이다. None은 기존대로 `no_stamp`.
- `camera_info_pairing(frame, info_stamps_ns)`: CameraInfo 동시각 짝 **진단**(판정에 쓰지 않음, 결정 L-7). 제품 소스는 짝짓지 않으므로
  stamp 목록이 없으면 "unknown"이다.
- ROS·Qt를 import하지 않는다(`lookup`은 주입).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from eyetoarm.vision.observation_envelope import ObservationEnvelope, CaptureInfo, CaptureTf, make_observation_envelope

TfLookupFn = Callable[[str, str, int, float], tuple]   # (parent, child, stamp_ns, timeout_s) → (T 4×4, 반환 stamp_ns)


@dataclass(frozen=True)
class TfLookupResult:
    """TF 조회 결과. `status`: ok | unavailable | late | stamp_mismatch | no_stamp."""

    tf: CaptureTf | None
    status: str
    wait_ms: float | None
    detail: str | None = None


class LiveCaptureTfLookup:
    """정확 촬영 stamp TF 조회(반환 뒤 상한 판정)."""

    def __init__(self, lookup: TfLookupFn, *, world_frame: str, wait_limit_s: float,
                 clock: Callable[[], float] = time.monotonic) -> None:
        """주입 조회 함수·world 프레임·상한(초)·단조 시계."""
        if not (isinstance(wait_limit_s, (int, float)) and wait_limit_s > 0):
            raise ValueError(f"wait_limit_s must be positive: {wait_limit_s!r}")

        self._lookup = lookup
        self.world_frame = world_frame
        self.wait_limit_s = float(wait_limit_s)
        self._clock = clock

    def __call__(self, frame) -> TfLookupResult:
        """프레임 촬영 시각의 world → optical TF."""
        stamp, child = getattr(frame, "ros_stamp_ns", None), getattr(frame, "optical_frame", None)

        if stamp is None or not child:
            return TfLookupResult(None, "no_stamp", None, "frame has no ros_stamp_ns/optical_frame")

        # B1(#173): tf2에서 시각 0은 "최신" 조회다. 양의 정수가 아닌 stamp는 조회 함수를 부르지 않는다(최신 대체 금지)
        if type(stamp) is not int or stamp <= 0:
            return TfLookupResult(None, "invalid_stamp", None, f"ros_stamp_ns must be a positive int: {stamp!r}")

        t0 = self._clock()

        try:
            T, got = self._lookup(self.world_frame, child, stamp, self.wait_limit_s)
        except Exception as exc:  # 없음·시각 밖·기타: 대체하지 않고 None
            return TfLookupResult(None, "unavailable", (self._clock() - t0) * 1000.0, f"{type(exc).__name__}: {exc}")

        waited = self._clock() - t0
        wait_ms = waited * 1000.0

        if waited > self.wait_limit_s:
            return TfLookupResult(None, "late", wait_ms, f"returned after {wait_ms:.1f} ms > {self.wait_limit_s * 1000:.1f} ms")

        if got != stamp:
            return TfLookupResult(None, "stamp_mismatch", wait_ms, f"tf stamp {got} != frame stamp {stamp}")

        return TfLookupResult(CaptureTf(self.world_frame, child, stamp, np.asarray(T, dtype=np.float64)), "ok", wait_ms)


def make_live_envelope(frame, observation, *, camera, tf_result: TfLookupResult, world_frame: str) -> ObservationEnvelope:
    """실시간 프레임으로 촬영 정보를 만들고 기존 봉투 검사를 거친다. 오류는 `ObservationEnvelopeError`."""
    capture = CaptureInfo(stamp_ns=frame.ros_stamp_ns, optical_frame=frame.optical_frame,
                          calibration_id=frame.calibration_id, width=frame.width, height=frame.height)

    return make_observation_envelope(frame, observation, capture=capture, camera=camera, tf=tf_result.tf,
                                world_frame=world_frame)


def camera_info_pairing(frame, info_stamps_ns) -> str:
    """CameraInfo 동시각 짝 진단: "true" | "false" | "unknown"(stamp 목록 없음). 판정에 쓰지 않는다."""
    if info_stamps_ns is None or getattr(frame, "ros_stamp_ns", None) is None:
        return "unknown"

    return "true" if frame.ros_stamp_ns in info_stamps_ns else "false"


__all__ = ["LiveCaptureTfLookup", "TfLookupFn", "TfLookupResult", "camera_info_pairing", "make_live_envelope"]
