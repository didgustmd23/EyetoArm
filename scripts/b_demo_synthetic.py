"""B 확인용 합성 데모 — ROS·Gazebo 없이 B 계산 결과를 그림으로 본다.

하는 일
1. 링크 카메라와 같은 렌즈(1280×720, 수평 화각 1.0 rad)로 벨트·상자·부품 3개를 그린다.
   부품은 정답 위치·각도를 알고 있다.
2. 그림에서 밝은 물체 윤곽을 찾고(확인용 임시 검출), B 계산으로 중심·각도·규격을 구한다.
3. 중심점 주석 영상과 정답 비교표를 출력하고 PNG로 저장한다.

실행 (저장소 맨 위 폴더에서):
    PYTHONPATH=src python3 scripts/b_demo_synthetic.py            # → b_demo_overlay.png
    PYTHONPATH=src python3 scripts/b_demo_synthetic.py --show     # 창으로도 보기
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import yaml

from eyetoarm.calibration.profile import from_fov
from eyetoarm.transforms.overlay import draw_guides, draw_results, simple_part_contours
from eyetoarm.transforms.planar_transform import look_at, w_to_image
from eyetoarm.transforms.work_result import compute_work_result, load_specs

ROOT = Path(__file__).resolve().parents[1]

# 정답: (이름, 중심 x, y [m], 각도 deg)
TRUTH = [("BOLT_A", 0.56, -0.22, 30.0), ("BOLT_B", 0.53, -0.06, 110.0), ("NUT", 0.58, 0.09, 15.0)]


def bolt_outline(total_len: float, head_d=0.010, head_h=0.006, shaft_d=0.006) -> np.ndarray:
    """누운 볼트를 위에서 본 윤곽(부품 좌표, 축 = +X, 중심 = 길이 가운데)."""
    x0 = -total_len / 2.0
    xs = x0 + head_h
    xe = x0 + total_len

    return np.array([[x0, -head_d / 2], [xs, -head_d / 2], [xs, -shaft_d / 2], [xe, -shaft_d / 2],
                     [xe, shaft_d / 2], [xs, shaft_d / 2], [xs, head_d / 2], [x0, head_d / 2]])


def nut_outline(across_flats=0.010) -> np.ndarray:
    r = across_flats / math.sqrt(3.0)

    return np.array([[r * math.cos(math.radians(30 + 60 * k)), r * math.sin(math.radians(30 + 60 * k))] for k in range(6)])


def place(local: np.ndarray, cx: float, cy: float, deg: float, z: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    xy = local @ np.array([[c, s], [-s, c]]) + [cx, cy]

    return np.column_stack([xy, np.full(len(xy), z)])


def fill(img, pts_w, cam, T_W_C, color):
    px = w_to_image(pts_w, cam.K, cam.dist, T_W_C)

    if np.all(np.isfinite(px)):
        cv2.fillPoly(img, [px.round().astype(np.int32)], color, cv2.LINE_AA)


def render(cam, T_W_C, frames, specs) -> np.ndarray:
    img = np.full((cam.height, cam.width, 3), (90, 95, 100), np.uint8)
    b = frames["belt"]
    cx, cy = b["center"]
    hx, hy = b["width"] / 2, b["length"] / 2
    fill(img, np.array([[cx - hx, cy - hy, b["top_z"]], [cx + hx, cy - hy, b["top_z"]],
                        [cx + hx, cy + hy, b["top_z"]], [cx - hx, cy + hy, b["top_z"]]]), cam, T_W_C, (25, 25, 25))
    z = specs["plane_z"]

    for name, x, y, deg in TRUTH:
        local = nut_outline() if name == "NUT" else bolt_outline(specs["specs"][name]["total_length"])
        fill(img, place(local, x, y, deg, z), cam, T_W_C, (215, 215, 220))

    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="b_demo_overlay.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    frames = yaml.safe_load(open(ROOT / "config/robot/frames.yaml", encoding="utf-8"))
    specs = load_specs(ROOT / "config/calibration/part_specs.json")
    c = frames["cameras"]["link"]
    cam = from_fov(c["width"], c["height"], c["horizontal_fov"], c["optical_frame"])
    T_W_C = look_at((0.30, -0.08, 0.40), (0.56, -0.07, 0.10))  # 데모용 카메라 자세

    img = render(cam, T_W_C, frames, specs)
    contours = simple_part_contours(img, cam, T_W_C, frames)
    results = [compute_work_result(cnt, cam, T_W_C, specs) for cnt in contours]

    out = img.copy()
    draw_guides(out, cam, T_W_C, frames)
    draw_results(out, cam, T_W_C, results)
    cv2.imwrite(args.out, out)

    print(f"카메라 보정 식별자: {cam.calibration_id}")
    print(f"{'정답':8} {'B 판정':10} {'위치 오차(mm)':>14} {'각도 오차(deg)':>15} {'길이(mm)':>9}")

    for name, x, y, deg in TRUTH:
        r = min(results, key=lambda r: math.hypot(r.center_w_m[0] - x, r.center_w_m[1] - y))
        err = math.hypot(r.center_w_m[0] - x, r.center_w_m[1] - y) * 1000
        dth = "-" if not r.theta_valid else f"{abs((r.theta_deg - deg + 90) % 180 - 90):.2f}"
        print(f"{name:8} {r.spec:10} {err:14.2f} {dth:>15} {r.measured_length_m * 1000:9.1f}")

    print(f"저장: {Path(args.out).resolve()}")

    if args.show:
        cv2.imshow("B demo", out)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
