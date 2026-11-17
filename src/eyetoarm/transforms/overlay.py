"""B 중심점 주석 영상 — 링크 카메라 사진 위에 B 결과와 작업 좌표 안내선을 그린다.

근거: A 상세설계·시스템 계획 영상 표시 경로(사용자 확정 2026-11-12): B는 링크 카메라의
'중심점 주석 영상'을 발행하고 C가 D로 전달한다.

- 그리는 위치는 모두 W 좌표를 `w_to_image`로 사진에 다시 투영한 것이다.
  사진에서 본 각도를 그대로 쓰지 않으므로 원근이 있어도 화살표가 실제 방향과 맞는다.
- 색: BOLT_A 파랑, BOLT_B 초록, NUT 주황, UNRESOLVED 빨강 (시뮬레이션 상자 색과 같음).
- ROS를 import하지 않는다.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from eyetoarm.calibration.profile import CameraProfile
from eyetoarm.transforms.planar_transform import w_to_image
from eyetoarm.transforms.work_result import WorkPlanarResult

COLORS = {  # BGR
    "BOLT_A": (235, 140, 40),
    "BOLT_B": (60, 200, 60),
    "NUT": (40, 150, 255),
    "UNRESOLVED": (60, 60, 235),
}
GUIDE = (200, 200, 200)


def _pt(p) -> tuple[int, int] | None:
    if p is None or not np.all(np.isfinite(p)):
        return None

    return int(round(p[0])), int(round(p[1]))


def _poly(img, pts_w, cam, T_W_C, color, thickness=1, closed=True):
    px = w_to_image(pts_w, cam.K, cam.dist, T_W_C)
    ok = np.all(np.isfinite(px), axis=1)

    if ok.sum() >= 2:
        cv2.polylines(img, [px[ok].round().astype(np.int32).reshape(-1, 1, 2)], closed, color, thickness, cv2.LINE_AA)


def _rect_w(center, size, yaw, z):
    cx, cy = center
    hx, hy = size[0] / 2.0, size[1] / 2.0
    c, s = math.cos(yaw), math.sin(yaw)
    corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]

    return np.array([[cx + c * x - s * y, cy + s * x + c * y, z] for x, y in corners])


def draw_guides(img: np.ndarray, cam: CameraProfile, T_W_C: np.ndarray, frames: dict) -> np.ndarray:
    """벨트 테두리·10 cm 격자·상자 3개·W 원점 축을 그린다(작업 좌표가 맞는지 눈으로 확인)."""
    b = frames["belt"]
    z = b["top_z"]
    cx, cy = b["center"]
    hx, hy = b["width"] / 2.0, b["length"] / 2.0

    _poly(img, _rect_w((cx, cy), (b["width"], b["length"]), 0.0, z), cam, T_W_C, GUIDE, 2)

    for y in np.arange(math.ceil((cy - hy) * 10) / 10, cy + hy + 1e-9, 0.1):
        _poly(img, np.array([[cx - hx, y, z], [cx + hx, y, z]]), cam, T_W_C, GUIDE, 1, closed=False)
        p = _pt(w_to_image([[cx + hx + 0.01, y, z]], cam.K, cam.dist, T_W_C)[0])

        if p:
            cv2.putText(img, f"y={y + 0.0:+.1f}".replace("-0.0", "+0.0"), p, cv2.FONT_HERSHEY_SIMPLEX, 0.4, GUIDE, 1, cv2.LINE_AA)

    for name, bin_ in frames["bins"].items():
        sx, sy, sz = bin_["size"]
        top = _rect_w(bin_["center"], (sx, sy), bin_["yaw"], sz)
        _poly(img, top, cam, T_W_C, COLORS.get(name, GUIDE), 2)
        p = _pt(w_to_image([[*bin_["center"], sz]], cam.K, cam.dist, T_W_C)[0])

        if p:
            cv2.putText(img, name, p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.get(name, GUIDE), 1, cv2.LINE_AA)

    o = w_to_image([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], cam.K, cam.dist, T_W_C)
    p0 = _pt(o[0])

    for i, (label, col) in enumerate((("X", (60, 60, 235)), ("Y", (60, 200, 60)), ("Z", (235, 140, 40))), 1):
        p1 = _pt(o[i])

        if p0 and p1:
            cv2.arrowedLine(img, p0, p1, col, 2, cv2.LINE_AA, tipLength=0.2)
            cv2.putText(img, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    return img


def draw_results(img: np.ndarray, cam: CameraProfile, T_W_C: np.ndarray,
                 results: list[WorkPlanarResult]) -> np.ndarray:
    """B 결과마다 W 윤곽·중심점·장축 화살표·글자(규격, 길이 mm, 각도)를 그린다."""
    for r in results:
        col = COLORS.get(r.spec, COLORS["UNRESOLVED"])
        z = r.plane_z_m
        cw = np.column_stack([r.contour_w_m, np.full(len(r.contour_w_m), z)])
        _poly(img, cw, cam, T_W_C, col, 1)

        cx, cy = r.center_w_m
        c = _pt(w_to_image([[cx, cy, z]], cam.K, cam.dist, T_W_C)[0])

        if c is None:
            continue

        cv2.circle(img, c, 5, col, -1, cv2.LINE_AA)
        cv2.circle(img, c, 5, (255, 255, 255), 1, cv2.LINE_AA)

        if r.theta_valid and r.theta_deg is not None:
            th = math.radians(r.theta_deg)
            d = np.array([math.cos(th), math.sin(th)]) * max(r.measured_length_m * 0.7, 0.012)
            ends = w_to_image([[cx - d[0], cy - d[1], z], [cx + d[0], cy + d[1], z]], cam.K, cam.dist, T_W_C)
            a, b = _pt(ends[0]), _pt(ends[1])

            if a and b:
                tip = 12.0 / max(math.hypot(b[0] - a[0], b[1] - a[1]), 1.0)
                cv2.arrowedLine(img, a, b, col, 2, cv2.LINE_AA, tipLength=min(tip, 0.4))

        text = r.spec if r.valid else f"{r.spec} ({r.reason})"
        info = f"{r.measured_length_m * 1000:.1f}mm"

        if r.theta_valid and r.theta_deg is not None:
            info += f"  {r.theta_deg:.0f}deg"

        info += f"  ({cx * 1000:.0f}, {cy * 1000:.0f})mm"

        for k, line in enumerate((text, info)):
            org = (c[0] + 18, c[1] - 22 + 16 * k)
            cv2.putText(img, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    return img


def simple_part_contours(img: np.ndarray, cam: CameraProfile, T_W_C: np.ndarray, frames: dict,
                         min_area_px: float = 30.0) -> list[np.ndarray]:
    """확인용 임시 검출: 벨트 영역 안의 밝은 물체 윤곽(원본 픽셀).

    운영 검출은 A 담당이다. 이 함수는 A 결과 없이 B 계산을 눈으로 확인할 때만 쓴다.
    """
    b = frames["belt"]
    belt = w_to_image(_rect_w(b["center"], (b["width"], b["length"]), 0.0, b["top_z"]), cam.K, cam.dist, T_W_C)
    mask = np.zeros(img.shape[:2], np.uint8)

    if np.all(np.isfinite(belt)):
        cv2.fillPoly(mask, [belt.round().astype(np.int32)], 255)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vals = gray[mask > 0]

    if vals.size == 0:
        return []

    thr = max(float(np.median(vals)) + 40.0, 80.0)
    binary = ((gray > thr) & (mask > 0)).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))   # 볼트 머리·몸통이 갈라지지 않게 (실습 5단계)
    found = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = found[0] if len(found) == 2 else found[1]

    return [c.reshape(-1, 2).astype(np.float64) for c in contours if cv2.contourArea(c) >= min_area_px]
