"""실시간 프레임·관측·CameraProfile·촬영 시각 TF로 `ObservationEnvelope`를 만든다.

TF는 촬영 stamp로 한 번만 조회하고(최신값 대체 없음) 반환 뒤 상한을 넘으면 `late`로 버린다.
ROS·Qt를 import하지 않는다(`lookup`은 주입).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from eyetoarm.vision.observation_envelope import ObservationEnvelope, CaptureInfo, CaptureTf, make_observation_envelope

TfLookupFn = Callable[[str, str, int, float], tuple]   # (부모, 자식, stamp_ns, 상한 s) → (T, stamp)


@dataclass(frozen=True)
class TfLookupResult:
    """TF 조회 결과. `status`: ok | unavailable | late | stamp_mismatch | no_stamp | invalid_stamp."""

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

        # tf2에서 시각 0은 최신 조회라, 양의 정수가 아닌 stamp는 조회하지 않는다(최신 대체 금지)
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


def make_live_envelope(frame, observation, *, camera, tf_result: TfLookupResult, world_frame: str,
                       extensions=None) -> ObservationEnvelope:
    """실시간 프레임으로 촬영 정보를 만들어 봉투 검사를 거친다. 오류는 `ObservationEnvelopeError`.

    `extensions`: 선택 봉투 확장(예 `tracking`), None이면 확장 없음.
    """
    capture = CaptureInfo(stamp_ns=frame.ros_stamp_ns, optical_frame=frame.optical_frame,
                          calibration_id=frame.calibration_id, width=frame.width, height=frame.height)

    return make_observation_envelope(frame, observation, capture=capture, camera=camera, tf=tf_result.tf,
                                world_frame=world_frame, extensions=extensions)


def camera_info_pairing(frame, info_stamps_ns) -> str:
    """CameraInfo 동시각 짝 진단: "true" | "false" | "unknown"(stamp 목록 없음). 판정에 쓰지 않는다."""
    if info_stamps_ns is None or getattr(frame, "ros_stamp_ns", None) is None:
        return "unknown"

    return "true" if frame.ros_stamp_ns in info_stamps_ns else "false"


__all__ = ["LiveCaptureTfLookup", "TfLookupFn", "TfLookupResult", "camera_info_pairing", "make_live_envelope"]
