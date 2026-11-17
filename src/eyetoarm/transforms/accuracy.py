"""B 정확도 확인 — B 계산 결과를 시뮬레이션의 실제 부품 위치와 비교한다.

시뮬레이션 월드(`sort_cell.sdf`)에 적힌 부품 자리를 "정답"으로 두고, B가 사진에서 계산한
중심·각도·규격과 비교해 오차 표를 만든다. 벨트가 아직 움직이지 않으므로 부품은 처음 자리에 그대로 있다.

정답 중심은 위에서 본 모양의 가운데(면적 중심)다. 볼트 모델의 원점은 머리 아래 면이라
몸통 쪽으로 옮긴 자리가 정답이다 (`BOLT_CENTROID_OFFSET_M`).
"""
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# 모델 이름 → 규격
MODEL_SPEC = {"bolt_m6x12": "BOLT_A", "bolt_m6x40": "BOLT_B", "nut_m6": "NUT"}

# 볼트를 눕혔을 때(피치 90°) 위에서 본 모양의 면적 중심이 모델 원점(머리 아래 면)에서 축을 따라 떨어진 거리.
# 머리 Ø10 × 6 mm(원점 +0~+6 mm), 몸통 Ø6 × L(원점 0~−L) → (60·3 − 6L·L/2) / (60 + 6L)
BOLT_CENTROID_OFFSET_M = {
    "BOLT_A": (60 * 3 - 6 * 12 * 6) / (60 + 6 * 12) / 1000.0,    # ≈ −1.9 mm
    "BOLT_B": (60 * 3 - 6 * 40 * 20) / (60 + 6 * 40) / 1000.0,   # ≈ −15.4 mm
}


@dataclass(frozen=True)
class TruthPart:
    name: str
    spec: str
    xy_m: tuple[float, float]        # 위에서 본 모양의 가운데 (m)
    theta_deg: float | None          # 볼트 긴 축 방향 [0, 180), 너트는 None


def load_truth(sdf_path: str | Path) -> list[TruthPart]:
    """월드 파일에서 `part_*` 부품의 정답 자리를 읽는다."""
    text = Path(sdf_path).read_text(encoding="utf-8")
    parts = []

    for uri, name, pose in re.findall(
            r"<include>\s*<uri>model://([\w]+)</uri>\s*<name>(part_\w+)</name>\s*<pose>([^<]+)</pose>", text):
        spec = MODEL_SPEC.get(uri)

        if spec is None:
            continue

        x, y, _z, _roll, _pitch, yaw = (float(v) for v in pose.split())

        if spec == "NUT":
            parts.append(TruthPart(name, spec, (x, y), None))
            continue

        d = BOLT_CENTROID_OFFSET_M[spec]           # 눕힌 볼트: 모델 z축 → 월드 (cos yaw, sin yaw)
        parts.append(TruthPart(name, spec, (x + d * math.cos(yaw), y + d * math.sin(yaw)),
                               math.degrees(yaw) % 180.0))

    return parts


def angle_error_deg(a: float, b: float) -> float:
    """[0, 180) 방향끼리의 차이 (0~90°)."""
    d = abs(a - b) % 180.0

    return min(d, 180.0 - d)


def match(results, truth: list[TruthPart], max_dist_m: float = 0.05) -> list[dict]:
    """B 결과마다 가장 가까운 정답을 붙여 한 줄씩 만든다. 정답이 없으면 `truth=None`."""
    rows = []

    for r in results:
        x, y = r.center_w_m

        if not (np.isfinite(x) and np.isfinite(y)):
            continue

        best = min(truth, key=lambda t: math.hypot(x - t.xy_m[0], y - t.xy_m[1]), default=None)
        dist = math.hypot(x - best.xy_m[0], y - best.xy_m[1]) if best else math.inf

        if best is None or dist > max_dist_m:
            rows.append({"truth": None, "b_spec": r.spec, "x_mm": x * 1000, "y_mm": y * 1000})
            continue

        th_err = None

        if best.theta_deg is not None and r.theta_valid and r.theta_deg is not None:
            th_err = angle_error_deg(r.theta_deg, best.theta_deg)

        rows.append({
            "truth": best.name, "true_spec": best.spec, "b_spec": r.spec, "spec_ok": r.spec == best.spec,
            "x_mm": x * 1000, "y_mm": y * 1000,
            "dx_mm": (x - best.xy_m[0]) * 1000, "dy_mm": (y - best.xy_m[1]) * 1000, "dist_mm": dist * 1000,
            "theta_deg": r.theta_deg, "theta_err_deg": th_err,
            "length_mm": r.measured_length_m * 1000, "width_mm": r.measured_width_m * 1000,
        })

    return rows


def summarize(rows: list[dict], truth: list[TruthPart]) -> list[dict]:
    """부품별 요약: 잡힌 횟수, 규격 맞은 비율, 위치 오차 평균·최대, 각도 오차 평균·최대, 길이 평균."""
    out = []

    for t in truth:
        mine = [r for r in rows if r.get("truth") == t.name]
        dist = [r["dist_mm"] for r in mine]
        th = [r["theta_err_deg"] for r in mine if r.get("theta_err_deg") is not None]
        out.append({
            "part": t.name, "spec": t.spec, "seen": len(mine),
            "spec_ok_pct": 100.0 * sum(r["spec_ok"] for r in mine) / len(mine) if mine else None,
            "pos_mean_mm": float(np.mean(dist)) if dist else None,
            "pos_max_mm": float(np.max(dist)) if dist else None,
            "dx_mean_mm": float(np.mean([r["dx_mm"] for r in mine])) if mine else None,
            "dy_mean_mm": float(np.mean([r["dy_mm"] for r in mine])) if mine else None,
            "theta_mean_deg": float(np.mean(th)) if th else None,
            "theta_max_deg": float(np.max(th)) if th else None,
            "length_mean_mm": float(np.mean([r["length_mm"] for r in mine])) if mine else None,
        })

    return out


def format_table(summary: list[dict], extra: int = 0) -> str:
    """사람이 읽는 오차 표 (터미널·발표용)."""
    f = lambda v, n=1: "—" if v is None else f"{v:.{n}f}"
    lines = ["| 부품 | 규격 | 잡힌 수 | 규격 맞음 % | 위치 오차 평균/최대 mm | 평균 dx/dy mm | 각도 오차 평균/최대 ° | 잰 길이 mm |",
             "|---|---|---|---|---|---|---|---|"]

    for s in summary:
        lines.append(f"| {s['part']} | {s['spec']} | {s['seen']} | {f(s['spec_ok_pct'], 0)} | "
                     f"{f(s['pos_mean_mm'])} / {f(s['pos_max_mm'])} | {f(s['dx_mean_mm'])} / {f(s['dy_mean_mm'])} | "
                     f"{f(s['theta_mean_deg'])} / {f(s['theta_max_deg'])} | {f(s['length_mean_mm'])} |")

    if extra:
        lines.append(f"\n정답과 5 cm 넘게 떨어진 검출(벨트 위 다른 물체·그림자 등): {extra}번")

    return "\n".join(lines)


def write_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["frame", "stamp_ns", "truth", "true_spec", "b_spec", "spec_ok", "x_mm", "y_mm", "dx_mm", "dy_mm",
            "dist_mm", "theta_deg", "theta_err_deg", "length_mm", "width_mm"]

    with open(path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
