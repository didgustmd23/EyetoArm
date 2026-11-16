"""B 카메라 보정 프로파일 — 렌즈 값(K, dist)과 보정 식별자.

근거: B 상세설계 §6(렌즈 보정)·§6.2(원본 픽셀, B 1회 보정), A 협업 요청 B-01·B-02.

- Gazebo 카메라는 렌즈 왜곡이 없으므로 `camera_info`(또는 화각)로 K를 만들고 dist는 0이다.
- 실제 카메라는 체스판 보정 결과(K, dist)를 `from_matrix`로 넣는다.
- `calibration_id`는 해상도·K·dist에서 만든 짧은 해시다. A 관측의 `calibration_id`와
  다르면 B는 결과를 무효(`CALIBRATION_MISMATCH`)로 둔다.
- ROS를 import하지 않는다. ROS `CameraInfo` 메시지는 `from_camera_info`에 그대로 넘기면 된다.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CameraProfile:
    """카메라 하나의 렌즈 값.

    - `K`: 3×3 내부 행렬 (fx, fy, cx, cy, 픽셀).
    - `dist`: 왜곡 계수 (k1, k2, p1, p2, k3). Gazebo는 0.
    - `frame_id`: 광학 프레임 TF 이름 (Z 앞·X 오른쪽·Y 아래).
    """

    width: int
    height: int
    K: np.ndarray
    dist: np.ndarray
    frame_id: str
    calibration_id: str = field(default="")

    def __post_init__(self) -> None:
        K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        dist = np.asarray(self.dist, dtype=np.float64).reshape(-1)

        if dist.size < 5:
            dist = np.concatenate([dist, np.zeros(5 - dist.size)])

        if self.width <= 0 or self.height <= 0:
            raise ValueError("width/height must be positive")

        if K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("fx/fy must be positive")

        object.__setattr__(self, "K", K)
        object.__setattr__(self, "dist", dist)

        if not self.calibration_id:
            object.__setattr__(self, "calibration_id", make_calibration_id(self.width, self.height, K, dist))


def make_calibration_id(width: int, height: int, K: np.ndarray, dist: np.ndarray) -> str:
    """해상도·K·dist를 소수 6자리로 고정해 해시한 식별자(앞 12자리)."""
    vals = [width, height, *np.round(K.reshape(-1), 6), *np.round(dist.reshape(-1), 6)]
    text = ",".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in map(_py, vals))
    return "cal-" + hashlib.sha256(text.encode()).hexdigest()[:12]


def _py(v):
    """NumPy 스칼라를 파이썬 값으로."""
    return v.item() if hasattr(v, "item") else v


def from_fov(width: int, height: int, horizontal_fov: float, frame_id: str) -> CameraProfile:
    """Gazebo 센서 설정(해상도·수평 화각)으로 K를 만든다. 정사각 픽셀, 중심은 영상 가운데."""
    fx = (width / 2.0) / math.tan(horizontal_fov / 2.0)
    K = np.array([[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]])

    return CameraProfile(width, height, K, np.zeros(5), frame_id)


def from_camera_info(msg) -> CameraProfile:
    """ROS `sensor_msgs/CameraInfo`(또는 같은 이름의 속성을 가진 객체)에서 만든다."""
    K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(msg.d, dtype=np.float64) if len(msg.d) else np.zeros(5)

    return CameraProfile(int(msg.width), int(msg.height), K, dist, str(msg.header.frame_id))


def from_matrix(width: int, height: int, K, dist, frame_id: str) -> CameraProfile:
    """체스판 보정 결과(K, dist)로 만든다."""
    return CameraProfile(width, height, np.asarray(K), np.asarray(dist), frame_id)
