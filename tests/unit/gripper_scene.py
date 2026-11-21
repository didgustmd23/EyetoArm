"""그리퍼 깊이 시험용 합성 장면: 평면 + 누운 볼트(원기둥 두 개)를 광선 교차로 그린다.

정답(끝점·윗면)은 그린 기하 값이며 관측기로 역산하지 않는다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from eyetoarm.transforms.planar_transform import look_at
from eyetoarm.vision.gripper_depth import GripperDepthFrame

W, H = 640, 480
F = W / 2 / math.tan(math.radians(30))          # 수평 화각 60°(그리퍼 센서와 같음)
K = np.array([[F, 0.0, W / 2], [0.0, F, H / 2], [0.0, 0.0, 1.0]])
SUPPORT_Z_M = 0.100                              # 볼트가 놓인 물리 벨트 윗면
RENDER_BELT_Z_M = 0.096                          # 그려지는 벨트(렌더 결함 재현: 4 mm 낮음)
ROI_XY_M = (0.45, 0.65, -0.63, 0.63)             # frames.yaml 벨트 사각형


@dataclass(frozen=True)
class Bolt:
    """누운 볼트: 머리·축 경계 xy(m), 머리 → 축 끝 방향 각도(deg), 치수(m)."""

    x: float
    y: float
    yaw_deg: float
    head_r: float
    head_len: float
    shank_r: float
    shank_len: float

    @property
    def axis(self) -> np.ndarray:
        """머리 → 축 끝 단위 벡터(수평)."""
        a = math.radians(self.yaw_deg)

        return np.array([math.cos(a), math.sin(a), 0.0])

    @property
    def axis_z(self) -> float:
        """축 중심 높이(머리가 받침면에 닿음, 축은 수평으로 둔다)."""
        return SUPPORT_Z_M + self.head_r

    def head_end_mm(self) -> np.ndarray:
        """머리 끝(축선) xy mm."""
        return (np.array([self.x, self.y]) - self.axis[:2] * self.head_len) * 1000.0

    def tip_mm(self) -> np.ndarray:
        """축 끝 xy mm."""
        return (np.array([self.x, self.y]) + self.axis[:2] * self.shank_len) * 1000.0

    def top_z_mm(self) -> float:
        """가장 높은 점(머리 윗면) mm."""
        return (self.axis_z + self.head_r) * 1000.0


def m3(x=0.55, y=0.0, yaw_deg=0.0) -> Bolt:
    """M3x20 모양."""
    return Bolt(x, y, yaw_deg, 0.0025, 0.003, 0.0015, 0.020)


def m6(x=0.55, y=0.0, yaw_deg=0.0) -> Bolt:
    """M6x40 모양."""
    return Bolt(x, y, yaw_deg, 0.005, 0.006, 0.003, 0.040)


def camera_above(x=0.55, y=0.0, height_m=0.12, tilt_m=0.0) -> np.ndarray:
    """(x, y) 위 `height_m`에서 아래를 보는 `T_W_C`. `tilt_m`만큼 바라볼 점을 x로 옮겨 기울인다."""
    return look_at((x, y, SUPPORT_Z_M + height_m), (x + tilt_m, y, SUPPORT_Z_M), up=(1.0, 0.0, 0.0))


def _cylinder_t(o, d, p0, a, r, s0, s1):
    """광선(o, d) × 원기둥 옆면(축 p0 + s a, s∈[s0, s1]) 최소 양의 t. 없으면 inf."""
    w = o - p0
    dp = d - (d @ a)[..., None] * a
    wp = w - (w @ a)[..., None] * a
    A = (dp * dp).sum(-1)
    B = 2 * (dp * wp).sum(-1)
    C = (wp * wp).sum(-1) - r * r
    disc = B * B - 4 * A * C
    out = np.full(A.shape, np.inf)
    ok = disc >= 0
    sq = np.sqrt(np.where(ok, disc, 0.0))

    for t in ((-B - sq) / (2 * A), (-B + sq) / (2 * A)):
        s = ((o + t[..., None] * d - p0) @ a)
        hit = ok & (t > 0) & (s >= s0) & (s <= s1) & (t < out)
        out = np.where(hit, t, out)

    return out


def render(T_W_C, bolts=(), *, stamp_ns=1_000_000_000, belt_z_m=RENDER_BELT_Z_M, near_m=None) -> GripperDepthFrame:
    """깊이(광축 z, m) 프레임. `near_m`이면 그보다 가까운 화소를 0으로(근접 무효 재현)."""
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    dc = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1)
    R, o = T_W_C[:3, :3], T_W_C[:3, 3]
    d = dc @ R.T
    t = np.where(d[..., 2] < 0, (belt_z_m - o[2]) / np.where(d[..., 2] < 0, d[..., 2], -1.0), np.inf)

    for b in bolts:
        p0 = np.array([b.x, b.y, b.axis_z])
        a = b.axis
        t = np.minimum(t, _cylinder_t(o, d, p0, a, b.head_r, -b.head_len, 0.0))
        t = np.minimum(t, _cylinder_t(o, d, p0, a, b.shank_r, 0.0, b.shank_len))

    depth = t.astype(np.float32)   # dc의 z 성분이 1이라 t = 광축 z

    if near_m is not None:
        depth[depth < near_m] = 0.0

    depth.setflags(write=False)

    return GripperDepthFrame(ros_stamp_ns=stamp_ns, timestamp_ms=stamp_ns // 1_000_000, optical_frame="gripper_cam_optical_frame",
                             depth_m=depth, K=K.copy(), width=W, height=H)
