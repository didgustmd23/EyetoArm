#!/usr/bin/env python3
"""Gazebo 학습·판정용 장면 계획과 학습 조각 라벨 계산(#278, Gazebo·ROS 실행 없음).

평가 도구(`gz_eval_scene.py`)의 슬롯·난수 순서는 바꾸지 않는다(기존 평가 세트 재현성 유지).
이 모듈은 그와 별도로 다음을 정한다.

- 시드 구간: 학습은 1,000,000번대, 평가는 `gz_eval_scene`의 9,000,000번대. 용도와 시드가 맞지 않으면 거부한다.
- 부품 표: M3×20(B-12 BOLT_A 후보)·M6×12·M6×40·M6 너트. 종류별 뽑기 비중을 둔다.
- 자세: yaw는 균등이 기본이고, 영상에서 축이 가로(< 22.5°)로 보이는 구간은 `horiz_weight`배, 완전 가로(< 7°)는
  `narrow_weight`배로 더 자주 뽑는다(HOME 기준 가로 약 56 %, 그중 완전 가로 약 0.59 — reviewer #278 요청 「가로 구간의 절반 이상」).
- 투영 중심: 볼트는 원점(머리·축 경계)이 아니라 머리 끝~축 끝의 중점을 투영해 관문 거리에 쓴다.
- 학습 조각: 1280×720 영상을 원래 화소 크기 그대로 640×640 조각으로 자르고 다각형을 조각에 맞춰 자른다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import gz_eval_scene as scene

TRAIN_SEED_BASE = 1_000_000
TRAIN_SEED_SPAN = 1_000_000
PARK = scene.PARK


@dataclass(frozen=True)
class PartSpec:
    """부품 종류 하나. 길이는 m, 각도는 rad."""

    model: str          # simulation/models 아래 모델 이름
    yolo_cls: int       # 0 bolt, 1 nut
    roll_pitch: tuple   # 놓는 자세 rpy 앞 두 값
    z_m: float          # 놓는 원점 높이(m)
    mid_offset_m: float  # 원점에서 모델 z축 방향 중점까지 거리(볼트: (머리 높이 − 축 길이)/2, 너트 0)
    weight: float       # 종류 뽑기 비중(합이 1일 필요는 없음)


# 모델 SDF 근거: simulation/models/*/model.sdf(관성 pose z = 중점), 머리 +z·축 −z, 놓는 높이 = 벨트 0.100 + 머리 반지름.
PARTS = {
    "BOLT_M3X20": PartSpec("bolt_m3x20", 0, (0.0, math.pi / 2), 0.1025, -0.0085, 0.40),
    "BOLT_M6X40": PartSpec("bolt_m6x40", 0, (0.0, math.pi / 2), 0.105, -0.0170, 0.25),
    "NUT_M6": PartSpec("nut_m6", 1, (0.0, 0.0), 0.1005, 0.0, 0.25),
    "BOLT_M6X12": PartSpec("bolt_m6x12", 0, (0.0, math.pi / 2), 0.105, -0.0030, 0.10),
}
SLOTS_PER_KIND = 6
GATE_PX = {"BOLT_M3X20": 25.0, "BOLT_M6X40": 40.0, "NUT_M6": 8.0, "BOLT_M6X12": 25.0}


# ---------------------------------------------------------------- 시드
def train_seed(batch: int) -> int:
    """학습 묶음 번호의 시드. 번호가 0 미만이거나 구간 폭 이상이면 ValueError."""
    if not 0 <= batch < TRAIN_SEED_SPAN:
        raise ValueError("batch out of range")
    return TRAIN_SEED_BASE + batch


def is_train_seed(seed: int) -> bool:
    """학습 시드 구간이면 True."""
    return TRAIN_SEED_BASE <= seed < TRAIN_SEED_BASE + TRAIN_SEED_SPAN


def check_seed(seed: int, purpose: str) -> None:
    """용도(train/eval)와 시드 구간이 맞지 않으면 ValueError. 학습 자료가 평가 장면과 섞이지 않게 막는다."""
    if purpose == "train":
        ok = is_train_seed(seed)
    elif purpose == "eval":
        ok = scene.is_eval_seed(seed)
    else:
        raise ValueError(f"unknown purpose {purpose!r}")

    if not ok:
        raise ValueError(f"seed {seed} is not in the {purpose} range")


# ---------------------------------------------------------------- 슬롯·월드
def slot_table() -> dict:
    """슬롯 이름 → (종류, 라벨). 라벨은 1부터, `PARTS` 순서·종류별 `SLOTS_PER_KIND`개."""
    out, label = {}, 1

    for kind in PARTS:
        for k in range(SLOTS_PER_KIND):
            out[f"{kind}_{k}"] = (kind, label)
            label += 1

    return out


def slot_include_xml(slot: str, kind: str, label: int) -> str:
    """슬롯 부품을 대기 위치(PARK)에 정적으로 두고 분할 라벨을 붙이는 `<include>` 조각."""
    pose = " ".join(f"{v:.4f}" for v in (*PARK, 0, 0, 0))
    return (f'\n    <include><uri>model://{PARTS[kind].model}</uri><name>{slot}</name><static>true</static>'
            f'<pose>{pose}</pose>'
            f'<plugin filename="gz-sim-label-system" name="gz::sim::systems::Label"><label>{label}</label></plugin>'
            f'</include>')


def build_world(src: Path) -> str:
    """원본 월드에서 part_00~09 include를 빼고 라벨 슬롯을 더한 사본 문자열(원본은 수정하지 않음)."""
    lines = [ln for ln in Path(src).read_text(encoding="utf-8").splitlines()
             if not (ln.strip().startswith("<include>") and any(f"<name>part_0{k}</name>" in ln for k in range(10)))]
    text = "\n".join(lines) + "\n"

    if text.count("  </world>") != 1:
        raise ValueError("</world>를 하나로 찾지 못함")

    add = "".join(slot_include_xml(s, kind, label) for s, (kind, label) in slot_table().items())
    return text.replace("  </world>", add + "\n  </world>")


def pose_vector_request(parts: list) -> str:
    """모든 슬롯의 pose(protobuf 텍스트). 표본에 없는 슬롯은 PARK 줄에 0.1 m 간격으로 둔다."""
    placed = {p["slot"]: p for p in parts}
    items = []

    for i, slot in enumerate(slot_table()):
        if slot in placed:
            x, y, z = placed[slot]["xyz"]
            r, pch, yw = placed[slot]["rpy"]
        else:
            x, y, z = PARK[0], PARK[1] + 0.1 * i, PARK[2]
            r = pch = yw = 0.0

        qx, qy, qz, qw = scene.rpy_to_quat(r, pch, yw)
        items.append(f'pose {{ name: "{slot}" position {{ x: {x:.6f} y: {y:.6f} z: {z:.6f} }} '
                     f'orientation {{ x: {qx:.8f} y: {qy:.8f} z: {qz:.8f} w: {qw:.8f} }} }}')

    return " ".join(items)


# ---------------------------------------------------------------- 기하
def project_optical(t_W_C, R_W_C, K, p_W):
    """world 점(m) → (u, v, depth). 광학 프레임(z 앞, x 오른쪽, y 아래), 왜곡 없음. 카메라 뒤면 None."""
    c = np.asarray(R_W_C).T @ (np.asarray(p_W, float) - np.asarray(t_W_C, float))

    if c[2] <= 1e-9:
        return None

    return (K[0] * c[0] / c[2] + K[2], K[4] * c[1] / c[2] + K[5], float(c[2]))


def axis_dir_world(rpy) -> np.ndarray:
    """모델 z축(볼트 머리 방향)의 world 방향. SDF 고정축 rpy."""
    return scene.rpy_to_matrix(*rpy) @ np.array([0.0, 0.0, 1.0])


def part_mid_world(kind: str, xyz, rpy) -> np.ndarray:
    """부품 중점(m). 볼트는 원점 + mid_offset × 모델 z축, 너트는 원점."""
    return np.asarray(xyz, float) + PARTS[kind].mid_offset_m * axis_dir_world(rpy)


def image_axis_angle_deg(t_W_C, R_W_C, K, p_W, axis_W, half_len_m=0.01):
    """world 점 p에서 axis 방향 선분의 영상 각도(0 = 가로, 90 = 세로, 0~90). 투영이 안 되면 None."""
    a = project_optical(t_W_C, R_W_C, K, np.asarray(p_W) - half_len_m * np.asarray(axis_W))
    b = project_optical(t_W_C, R_W_C, K, np.asarray(p_W) + half_len_m * np.asarray(axis_W))

    if a is None or b is None:
        return None

    du, dv = b[0] - a[0], b[1] - a[1]

    if math.hypot(du, dv) < 1e-9:
        return 90.0  # 축이 시선과 나란함: 가로가 아니다

    return math.degrees(math.atan2(abs(dv), abs(du)))


def yaw_weight(ang_deg, horiz_weight=1.3, band_deg=22.5, narrow_weight=4.0, narrow_deg=7.0) -> float:
    """영상 각도별 상대 가중. 완전 가로(< narrow_deg) narrow_weight, 가로(< band_deg) horiz_weight, 그 밖 1.

    narrow 구간은 #278 사전 진단에서 확대해도 검출되지 않은 자세(HOME에서 yaw 90° ± 10°가 약 6.5° 이내)다.
    """
    if ang_deg is None:
        return 1.0
    if ang_deg < narrow_deg:
        return max(narrow_weight, horiz_weight, 1.0)
    if ang_deg < band_deg:
        return max(horiz_weight, 1.0)
    return 1.0


def sample_yaw(rng, t_W_C, R_W_C, K, kind, xyz, horiz_weight=1.3, band_deg=22.5, narrow_weight=4.0, narrow_deg=7.0,
               max_tries=100):
    """yaw(rad). 균등 제안을 yaw_weight / 최대 가중 확률로 받아들여 가로·완전 가로 비중을 높인다.

    너트(축 대칭이 아닌 판형)는 균등 그대로다. max_tries 안에 못 받으면 마지막 제안을 쓴다.
    """
    yaw = float(rng.uniform(-math.pi, math.pi))
    w_max = max(narrow_weight, horiz_weight, 1.0)

    if PARTS[kind].yolo_cls != 0 or w_max <= 1.0:
        return yaw

    rp = PARTS[kind].roll_pitch

    for _ in range(max_tries):
        rpy = (*rp, yaw)
        ang = image_axis_angle_deg(t_W_C, R_W_C, K, part_mid_world(kind, xyz, rpy), axis_dir_world(rpy))
        accept = yaw_weight(ang, horiz_weight, band_deg, narrow_weight, narrow_deg) / w_max

        if rng.uniform() < accept:
            return yaw

        yaw = float(rng.uniform(-math.pi, math.pi))

    return yaw


def plan_samples(n_samples, t_W_C, R_W_C, K, width, height, seed, purpose, max_parts=6, min_sep_m=0.035,
                 margin_px=24, horiz_weight=1.3, narrow_weight=4.0, belt_x=scene.BELT_X, belt_y=scene.BELT_Y):
    """고정 카메라에서 보이는 벨트 위 무작위 배치 목록. 같은 입력이면 같은 계획이다.

    부품 수는 1~max_parts, 종류는 `PARTS` 비중으로 뽑고, 중점 투영이 화면 가장자리에서 margin_px 안쪽,
    서로 min_sep_m 이상 떨어지게 놓는다. 2000번 시도해도 못 채우면 그 표본의 부품 수가 적을 수 있다.
    """
    check_seed(seed, purpose)
    rng = np.random.RandomState(seed)
    kinds = list(PARTS)
    w = np.array([PARTS[k].weight for k in kinds], float)
    w = w / w.sum()
    slots = slot_table()
    samples = []

    for si in range(n_samples):
        n = int(rng.randint(1, max_parts + 1))
        used = {k: 0 for k in kinds}
        parts, pos, tries = [], [], 0

        while len(parts) < n and tries < 2000:
            tries += 1
            kind = kinds[int(rng.choice(len(kinds), p=w))]

            if used[kind] >= SLOTS_PER_KIND:
                continue

            spec = PARTS[kind]
            x = float(rng.uniform(belt_x[0] + 0.02, belt_x[1] - 0.02))
            y = float(rng.uniform(belt_y[0] + 0.02, belt_y[1] - 0.02))
            xyz = (x, y, spec.z_m)
            pr0 = project_optical(t_W_C, R_W_C, K, xyz)

            if pr0 is None or not (margin_px <= pr0[0] < width - margin_px and margin_px <= pr0[1] < height - margin_px):
                continue

            if any(math.hypot(x - a, y - b) < min_sep_m for a, b in pos):
                continue

            yaw = sample_yaw(rng, t_W_C, R_W_C, K, kind, xyz, horiz_weight, narrow_weight=narrow_weight)
            rpy = (*spec.roll_pitch, yaw)
            pr = project_optical(t_W_C, R_W_C, K, part_mid_world(kind, xyz, rpy))

            if pr is None or not (margin_px <= pr[0] < width - margin_px and margin_px <= pr[1] < height - margin_px):
                continue

            ang = image_axis_angle_deg(t_W_C, R_W_C, K, part_mid_world(kind, xyz, rpy), axis_dir_world(rpy))
            slot = f"{kind}_{used[kind]}"
            used[kind] += 1
            pos.append((x, y))
            parts.append({"slot": slot, "label": slots[slot][1], "cls": kind, "yolo_cls": spec.yolo_cls,
                          "xyz": [round(x, 4), round(y, 4), spec.z_m], "rpy": [rpy[0], rpy[1], round(yaw, 4)],
                          "cam_depth_m": round(pr[2], 4), "expected_px": [round(pr[0], 1), round(pr[1], 1)],
                          "image_angle_deg": None if ang is None else round(ang, 1)})

        samples.append({"index": si, "tries": tries, "parts": parts})

    return samples


def yaw_distribution(samples) -> dict:
    """계획 볼트의 영상 각도·yaw(mod 180) 분포 요약(manifest 기록용). 각도 구간 [0,7)·[7,22.5)·[22.5,90]."""
    ang = [p["image_angle_deg"] for s in samples for p in s["parts"] if p["yolo_cls"] == 0 and p["image_angle_deg"] is not None]
    yaw = [math.degrees(p["rpy"][2]) % 180.0 for s in samples for p in s["parts"] if p["yolo_cls"] == 0]
    n = len(ang)
    bins = {"narrow_lt7": sum(a < 7.0 for a in ang), "band_7_22.5": sum(7.0 <= a < 22.5 for a in ang),
            "other_ge22.5": sum(a >= 22.5 for a in ang)}
    hist = [sum(1 for y in yaw if k * 22.5 <= y < (k + 1) * 22.5) for k in range(8)]
    horiz = bins["narrow_lt7"] + bins["band_7_22.5"]
    return {"bolts": n, "image_angle_bins": bins, "narrow_share_of_horizontal": round(bins["narrow_lt7"] / horiz, 3) if horiz else None,
            "yaw_mod180_hist_22.5deg": hist}


def any_part_in_view(t_W_C, R_W_C, K, width, height, parts_world, margin_px=40):
    """부품 중 하나라도 (여백 margin_px만큼 넓힌) 화면 안에 투영되면 True. 음성 프레임 고르기용.

    parts_world: [(종류, xyz, 쿼터니언 xyzw)]. 볼트는 머리 끝·중점·축 끝, 너트는 중심을 검사한다.
    카메라 뒤 점은 화면 밖으로 본다.
    """
    for kind, xyz, q in parts_world:
        spec = PARTS[kind]
        pts = [np.asarray(xyz, float)]

        if spec.yolo_cls == 0:
            x, y, z, w = q
            R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            ax = R @ np.array([0.0, 0.0, 1.0])
            half = abs(spec.mid_offset_m)
            mid = np.asarray(xyz, float) + spec.mid_offset_m * ax
            pts = [mid, mid + (half + 0.006) * ax, mid - (half + 0.006) * ax]

        for p in pts:
            pr = project_optical(t_W_C, R_W_C, K, p)

            if pr is not None and -margin_px <= pr[0] < width + margin_px and -margin_px <= pr[1] < height + margin_px:
                return True

    return False


# ---------------------------------------------------------------- 학습 조각
def polygon_area(poly) -> float:
    """다각형 넓이(px², 신발끈 공식)."""
    p = np.asarray(poly, float)

    if len(p) < 3:
        return 0.0

    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def clip_polygon_rect(poly, x0, y0, x1, y1) -> np.ndarray:
    """다각형을 축 정렬 사각형 [x0, x1]×[y0, y1]로 자른다(Sutherland–Hodgman). 남는 것이 없으면 빈 배열."""
    out = [tuple(v) for v in np.asarray(poly, float)]

    for axis, bound, keep_ge in ((0, x0, True), (0, x1, False), (1, y0, True), (1, y1, False)):
        if not out:
            break

        inp, out = out, []

        def inside(v):
            return v[axis] >= bound if keep_ge else v[axis] <= bound

        for i, cur in enumerate(inp):
            prev = inp[i - 1]

            if inside(cur):
                if not inside(prev):
                    out.append(_intersect(prev, cur, axis, bound))
                out.append(cur)
            elif inside(prev):
                out.append(_intersect(prev, cur, axis, bound))

    return np.array(out, float).reshape(-1, 2)


def _intersect(a, b, axis, bound):
    """선분 ab와 axis=bound 직선의 교점."""
    t = (bound - a[axis]) / (b[axis] - a[axis])
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


def tile_origins(width, height, tile=640, xs=(0, 320, 640), y0=None):
    """조각 왼쪽 위 좌표 목록. y0가 None이면 세로 가운데. 조각이 영상 밖으로 나가면 ValueError."""
    if y0 is None:
        y0 = (height - tile) // 2

    out = [(int(x), int(y0)) for x in xs]

    for x, y in out:
        if x < 0 or y < 0 or x + tile > width or y + tile > height:
            raise ValueError(f"tile ({x}, {y}) outside {width}x{height}")

    return out


def tile_labels(polys, x0, y0, tile=640, keep_frac=0.15, min_area_px=8.0):
    """한 조각의 라벨. polys: [(yolo_cls, 화소 다각형)].

    조각 안 넓이가 원래의 keep_frac 이상이고 min_area_px 이상이면 잘린 다각형을 라벨로 둔다.
    조각에 걸쳤지만 그보다 작게 남는 부품이 있으면 None(애매한 조각: 라벨 없는 부품 조각이 배경으로 학습되는 것을 막는다).
    반환: [(yolo_cls, 조각 기준 화소 다각형)] 또는 None.
    """
    out = []

    for cls, poly in polys:
        full = polygon_area(poly)
        cut = clip_polygon_rect(poly, x0, y0, x0 + tile, y0 + tile)
        area = polygon_area(cut)

        if area <= 0.0:
            continue

        if full > 0 and area >= keep_frac * full and area >= min_area_px:
            out.append((cls, cut - [x0, y0]))
        else:
            return None

    return out
