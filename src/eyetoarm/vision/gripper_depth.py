"""그리퍼 깊이 관측기: 깊이(광축 z, m)·K·촬영 시각 `T_W_C`로 벨트 위 볼트 성분의 W 끝점·윗면을 잰다.

클래스·B 규격·잡는 점은 정하지 않는다(추적 보조 입력). 받침면 높이는 분할 문턱에만 쓴다. ROS·Qt를 import하지 않는다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

import cv2
import numpy as np


class GripperConfigError(ValueError):
    """그리퍼 넘기기 설정 오류."""


@dataclass(frozen=True)
class GripperConfig:
    """그리퍼 입력·관측기·짝짓기 시험값(이름에 단위). 모두 필수."""

    optical_frame: str
    depth_topic: str
    camera_info_topic: str
    max_rate_hz: float
    reorder_wait_ms: float
    late_limit_ms: int          # 마지막 링크보다 이만큼 넘게 이른 그리퍼 관측은 버린다
    gate_x_mm: float
    gate_y_mm: float
    near_m: float
    invalid_ratio_max: float
    belt_margin_mm: float
    h_min_mm: float
    h_max_mm: float
    min_px: int
    border_px: int

    def to_json(self) -> dict:
        """요약용 사전."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


_TEXT_KEYS = ("optical_frame", "depth_topic", "camera_info_topic")
_INT_KEYS = ("min_px", "border_px", "late_limit_ms")


def parse_gripper(doc) -> GripperConfig:
    """`gripper` 블록 → 설정. 모르는 키·누락·형·범위 오류는 `GripperConfigError`."""
    names = {f.name for f in fields(GripperConfig)}

    if not isinstance(doc, dict):
        raise GripperConfigError("gripper block must be an object")

    unknown, missing = set(doc) - names, names - set(doc)

    if unknown or missing:
        raise GripperConfigError(f"gripper: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")

    values = {}

    for name in names:
        v = doc[name]

        if name in _TEXT_KEYS:
            if not isinstance(v, str) or not v:
                raise GripperConfigError(f"{name} must be a non-empty string: {v!r}")
        elif name in _INT_KEYS:
            if type(v) is not int or v < 0 or (name == "min_px" and v < 1):
                raise GripperConfigError(f"{name} must be an int (min_px >= 1, border_px >= 0): {v!r}")
        elif isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise GripperConfigError(f"{name} must be a positive number: {v!r}")
        else:
            v = float(v)

        values[name] = v

    if values["invalid_ratio_max"] > 1.0:
        raise GripperConfigError("invalid_ratio_max must be <= 1")

    if values["h_min_mm"] >= values["h_max_mm"]:
        raise GripperConfigError("h_min_mm must be < h_max_mm")

    return GripperConfig(**values)


@dataclass(frozen=True)
class GripperDepthFrame:
    """그리퍼 깊이 프레임 하나(소유 사본). `depth_m`은 읽기 전용 float32 (H, W), `K`는 3×3."""

    ros_stamp_ns: int
    timestamp_ms: int
    optical_frame: str
    depth_m: np.ndarray
    K: np.ndarray
    width: int
    height: int


@dataclass(frozen=True)
class GripperComponent:
    """벨트 위 성분 하나(W, mm). 끝점 행 0 = 머리 쪽(더 높은 끝), 행 1 = 축 끝."""

    index: int
    center_mm: tuple[float, float]
    endpoints_mm: tuple[tuple[float, float], tuple[float, float]]
    length_mm: float
    angle_deg: float
    top_z_mm: float
    n_px: int
    clipped: bool


@dataclass(frozen=True)
class GripperObservation:
    """관측 결과. 무효면 `reason`이 있고 성분은 비어 있다."""

    valid: bool
    reason: str | None
    components: tuple[GripperComponent, ...]
    diag: dict


_BIN_MM = 0.5
_EDGE_FRAC = 0.15
_CENTER_FRAC = 0.5


def _invalid(reason: str, diag: dict) -> GripperObservation:
    """무효 결과."""
    return GripperObservation(False, reason, (), diag)


def observe(frame: GripperDepthFrame, T_W_C: np.ndarray, config: GripperConfig,
            roi_xy_m: tuple[float, float, float, float]) -> GripperObservation:
    """깊이 프레임 → 벨트 위 성분 목록. `roi_xy_m` = 벨트 사각형(x_min, x_max, y_min, y_max, W m, 설정값)."""
    depth = np.asarray(frame.depth_m, dtype=np.float64)
    h, w = depth.shape
    valid = np.isfinite(depth) & (depth >= config.near_m)

    # 가운데 영역의 무효 비율(near 미만·nan·inf·0)이 크면 근접·가림으로 본다
    ch0, ch1 = int(h * (1 - _CENTER_FRAC) / 2), int(h * (1 + _CENTER_FRAC) / 2)
    cw0, cw1 = int(w * (1 - _CENTER_FRAC) / 2), int(w * (1 + _CENTER_FRAC) / 2)
    invalid_ratio = float(1.0 - valid[ch0:ch1, cw0:cw1].mean())
    diag = {"invalid_ratio": round(invalid_ratio, 4)}

    if invalid_ratio > config.invalid_ratio_max:
        return _invalid("depth_invalid", diag)

    # 광축 z 역투영 → W(mm)
    K = frame.K
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    z = np.where(valid, depth, np.nan)
    pc = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], axis=-1)
    pw = (pc @ np.asarray(T_W_C)[:3, :3].T + np.asarray(T_W_C)[:3, 3]) * 1000.0

    # 벨트 사각형(± 여유) 안의 점만
    m = config.belt_margin_mm
    x0, x1, y0, y1 = (r * 1000.0 for r in roi_xy_m)
    with np.errstate(invalid="ignore"):
        roi = valid & (pw[..., 0] >= x0 - m) & (pw[..., 0] <= x1 + m) & (pw[..., 1] >= y0 - m) & (pw[..., 1] <= y1 + m)

    if roi.sum() < config.min_px:
        return _invalid("no_belt_plane", diag)

    # 받침면 = 관심 영역 z 최빈값. 분할 문턱에만 쓴다(렌더 벨트는 실제보다 낮을 수 있음)
    zs = pw[..., 2][roi]
    bins = np.arange(zs.min(), zs.max() + 2 * _BIN_MM, _BIN_MM)
    hist, edges = np.histogram(zs, bins=bins)
    base_mm = float(edges[int(np.argmax(hist))] + _BIN_MM / 2)
    diag["base_z_mm"] = round(base_mm, 2)

    with np.errstate(invalid="ignore"):
        height = pw[..., 2] - base_mm
        mask = (roi & (height > config.h_min_mm) & (height < config.h_max_mm)).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = []

    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < config.min_px:
            continue

        comps.append(_component(len(comps), labels == i, stats[i], pw, w, h, config.border_px))

    diag["n_components"] = len(comps)

    if not comps:
        return _invalid("no_component", diag)

    return GripperObservation(True, None, tuple(comps), diag)


def _component(index: int, sel: np.ndarray, stat, pw: np.ndarray, w: int, h: int, border_px: int) -> GripperComponent:
    """성분 하나: xy 주축·양 끝(더 높은 끝을 머리로)·윗면."""
    pts = pw[sel]
    xy = pts[:, :2]
    c = xy.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov((xy - c).T))
    axis = evecs[:, int(np.argmax(evals))]
    s = (xy - c) @ axis
    s_min, s_max = float(s.min()), float(s.max())
    span = s_max - s_min

    # 양 끝 띠의 최고 높이가 큰 쪽이 머리
    lo, hi = s <= s_min + _EDGE_FRAC * span, s >= s_max - _EDGE_FRAC * span
    head_s, tip_s = (s_max, s_min) if pts[hi, 2].max() >= pts[lo, 2].max() else (s_min, s_max)
    head, tip = c + axis * head_s, c + axis * tip_s
    d = tip - head

    x, y, bw, bh = (int(v) for v in stat[:4])
    clipped = x <= border_px or y <= border_px or x + bw >= w - border_px or y + bh >= h - border_px

    return GripperComponent(index=index, center_mm=(float((head[0] + tip[0]) / 2), float((head[1] + tip[1]) / 2)),
                            endpoints_mm=((float(head[0]), float(head[1])), (float(tip[0]), float(tip[1]))),
                            length_mm=float(math.hypot(d[0], d[1])), angle_deg=float(math.degrees(math.atan2(d[1], d[0])) % 180.0),
                            top_z_mm=float(np.percentile(pts[:, 2], 99)), n_px=int(sel.sum()), clipped=bool(clipped))


def project_w(points_mm, T_W_C: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """W 점(mm, N×3) → (픽셀 N×2, 광축 깊이 m N). 왜곡 없음(그리퍼 camera_info 왜곡 0)."""
    p = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3) / 1000.0
    T = np.asarray(T_W_C, dtype=np.float64)
    pc = (p - T[:3, 3]) @ T[:3, :3]
    zc = pc[:, 2]

    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([K[0, 0] * pc[:, 0] / zc + K[0, 2], K[1, 1] * pc[:, 1] / zc + K[1, 2]], axis=1)

    return uv, zc


__all__ = ["GripperComponent", "GripperConfig", "GripperConfigError", "GripperDepthFrame", "GripperObservation",
           "observe", "parse_gripper", "project_w"]
