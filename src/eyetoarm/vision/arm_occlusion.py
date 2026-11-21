"""팔·그리퍼 가림 판정: 링크별 캡슐(설정값)을 촬영 시각 TF로 W에 놓고, 카메라 → 트랙 점 선분이 캡슐을 지나면 가림으로 본다.

치수는 설정 파일 값이며 코드 상수가 없다. ROS·Qt를 import하지 않는다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class OccluderConfigError(ValueError):
    """가림 도형 설정 오류."""


@dataclass(frozen=True)
class Occluder:
    """링크 좌표의 캡슐: 선분 p0–p1(m)과 반지름(m)."""

    link: str
    p0_m: tuple[float, float, float]
    p1_m: tuple[float, float, float]
    radius_m: float

    def to_json(self) -> dict:
        """요약용 사전."""
        return {"link": self.link, "p0_m": list(self.p0_m), "p1_m": list(self.p1_m), "radius_m": self.radius_m}


def _vec3(v, name: str) -> tuple[float, float, float]:
    if not isinstance(v, list) or len(v) != 3 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                                    or not math.isfinite(x) for x in v):
        raise OccluderConfigError(f"{name} must be a list of 3 finite numbers: {v!r}")

    return float(v[0]), float(v[1]), float(v[2])


def parse_occluders(doc) -> tuple[Occluder, ...]:
    """`occluders` 목록 → 캡슐들. 각 항목 키는 link·p0_m·p1_m·radius_m(모두 필수)."""
    if not isinstance(doc, list) or not doc:
        raise OccluderConfigError("occluders must be a non-empty list")

    out = []

    for i, item in enumerate(doc):
        if not isinstance(item, dict) or set(item) != {"link", "p0_m", "p1_m", "radius_m"}:
            raise OccluderConfigError(f"occluders[{i}] must have exactly link, p0_m, p1_m, radius_m")

        r = item["radius_m"]

        if not isinstance(item["link"], str) or not item["link"]:
            raise OccluderConfigError(f"occluders[{i}].link must be a non-empty string")

        if isinstance(r, bool) or not isinstance(r, (int, float)) or not math.isfinite(r) or r <= 0:
            raise OccluderConfigError(f"occluders[{i}].radius_m must be a positive number: {r!r}")

        out.append(Occluder(item["link"], _vec3(item["p0_m"], f"occluders[{i}].p0_m"), _vec3(item["p1_m"], f"occluders[{i}].p1_m"),
                            float(r)))

    return tuple(out)


def world_capsules(occluders, T_W_L_by_link: dict) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """링크별 `T_W_L`(4×4)로 캡슐을 W에 놓는다 → [(A, B, r)]."""
    out = []

    for o in occluders:
        T = np.asarray(T_W_L_by_link[o.link], dtype=np.float64)
        a = T[:3, :3] @ np.asarray(o.p0_m) + T[:3, 3]
        b = T[:3, :3] @ np.asarray(o.p1_m) + T[:3, 3]
        out.append((a, b, o.radius_m))

    return out


def segment_distance(p0, p1, q0, q1) -> float:
    """두 선분 p0–p1, q0–q1 사이 최소 거리."""
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)

    if a <= 1e-15 and e <= 1e-15:
        return float(np.linalg.norm(r))

    if a <= 1e-15:
        s, t = 0.0, min(max(f / e, 0.0), 1.0)
    else:
        c = float(d1 @ r)

        if e <= 1e-15:
            t, s = 0.0, min(max(-c / a, 0.0), 1.0)
        else:
            b = float(d1 @ d2)
            denom = a * e - b * b
            s = min(max((b * f - c * e) / denom, 0.0), 1.0) if denom > 1e-15 else 0.0
            t = (b * s + f) / e

            if t < 0.0:
                t, s = 0.0, min(max(-c / a, 0.0), 1.0)
            elif t > 1.0:
                t, s = 1.0, min(max((b - c) / a, 0.0), 1.0)

    return float(np.linalg.norm((p0 + d1 * s) - (q0 + d2 * t)))


def occluded_fn(camera_w: np.ndarray, capsules, plane_z_m: float):
    """가림 판정 함수: (x_mm, y_mm) 점(볼트 평면)과 카메라 사이 선분이 어느 캡슐 반지름 안을 지나면 True."""
    c = np.asarray(camera_w, dtype=np.float64)

    def occluded(x_mm: float, y_mm: float, margin_mm: float = 0.0) -> bool:
        """그 점이 팔·그리퍼에 가려졌나(`margin_mm`만큼 캡슐을 부풀려 판정할 수 있다)."""
        p = np.array([x_mm / 1000.0, y_mm / 1000.0, plane_z_m])
        m = margin_mm / 1000.0

        return any(segment_distance(c, p, a, b) < r + m for a, b, r in capsules)

    return occluded


__all__ = ["Occluder", "OccluderConfigError", "occluded_fn", "parse_occluders", "segment_distance", "world_capsules"]
