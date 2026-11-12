#!/usr/bin/env python3
"""Gazebo 평가 세트 장면 생성(월드 사본·표본 배치) + 공용 함수.

A 상세설계 §7.1 「2단계 전이 학습 방향」의 Gazebo 평가 세트용(A 제안). 이 스크립트는 **Gazebo를 실행하지 않는다**.
월드 사본 SDF와 표본 계획(JSON)만 만든다. 실행·수집은 `scripts/gz_eval_capture.py`.

방식(A 제안, 정적 검사로 확인):
- RGB 카메라와 같은 링크·같은 pose에 `segmentation` 카메라(`<segmentation_type>semantic</segmentation_type>`)를 둔다.
- 부품 슬롯마다 **고유 라벨**(1..N)을 `gz-sim-label-system`으로 붙인다. 라벨 값이 곧 부품 슬롯이므로 semantic
  모드로도 인스턴스를 구분한다(panoptic 모드여도 라벨 채널만 읽으면 같다). 배경·벨트는 라벨 0.
- 한 번의 `gz sim` 실행(=묶음)에서 카메라·조명은 고정하고, 부품 pose만 `set_pose_vector` 서비스로 바꿔 표본을 찍는다.
  묶음마다 카메라 pose·조명을 바꾼 월드 사본을 새로 만든다.
- 클래스 슬롯: 클래스마다 최대 6개(BOLT_A M6×12, BOLT_B M6×40, NUT M6) = 18 슬롯. 표본에 쓰지 않는 슬롯은
  카메라 뒤 바닥(화면 밖)으로 옮긴다. 부품은 `<static>true</static>`로 넣어 물리로 움직이지 않게 한다.

시드 규칙: 평가 시드는 `EVAL_SEED_BASE`(= 9_000_000) + 묶음 번호. 학습용 시드는 이 구간을 쓰지 않는다
(`is_eval_seed`). 같은 시드 → 같은 계획.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- 상수(월드 배치는 simulation/worlds/sort_cell.sdf 임시값)
WORLD_SRC = Path(__file__).resolve().parents[1] / "simulation" / "worlds" / "sort_cell.sdf"
WORLD_NAME = "sort_cell"
EVAL_SEED_BASE = 9_000_000
EVAL_SEED_SPAN = 1_000_000
BELT_X = (0.45, 0.65)          # 벨트 폭 방향(m)
BELT_Y = (-0.60, 0.60)         # 벨트 길이 방향(m)
BELT_TOP_Z = 0.100
IMG_W, IMG_H, HFOV = 1280, 720, 1.0
CLASSES = {  # 세부 클래스 → (모델, YOLO 2클래스 id, 놓는 자세 rpy 앞 두 값, z)
    "BOLT_A": ("bolt_m6x12", 0, (0.0, math.pi / 2), 0.105),
    "BOLT_B": ("bolt_m6x40", 0, (0.0, math.pi / 2), 0.105),
    "NUT": ("nut_m6", 1, (0.0, 0.0), 0.1005),
}
SLOTS_PER_CLASS = 6
CAM_BASE = {"xyz": (0.20, -0.15, 0.525), "aim": (0.55, -0.25, BELT_TOP_Z)}  # §6.1 경사 예시(사본 확인값)
PARK = (-2.0, 0.0, 0.05)        # 쓰지 않는 슬롯(카메라 뒤 바닥)


def is_eval_seed(seed: int) -> bool:
    return EVAL_SEED_BASE <= seed < EVAL_SEED_BASE + EVAL_SEED_SPAN


def eval_seed(batch: int) -> int:
    if not 0 <= batch < EVAL_SEED_SPAN:
        raise ValueError("batch out of range")
    return EVAL_SEED_BASE + batch


# ---------------------------------------------------------------- 기하
def rpy_to_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def rpy_to_quat(r, p, y):
    """(x, y, z, w), SDF/URDF 고정축 rpy."""
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


def aim_rpy(cam_xyz, target_xyz):
    """카메라 링크 +X가 target을 보도록 하는 (0, pitch, yaw). Gazebo 카메라는 링크 +X를 본다."""
    d = np.subtract(target_xyz, cam_xyz)
    yaw = math.atan2(d[1], d[0])
    pitch = math.atan2(-d[2], math.hypot(d[0], d[1]))
    return (0.0, pitch, yaw)


def fx_from_hfov(width=IMG_W, hfov=HFOV):
    return (width / 2) / math.tan(hfov / 2)


def project(cam_xyz, cam_rpy, point, width=IMG_W, height=IMG_H, hfov=HFOV):
    """월드 점 → (u, v, depth). 링크 좌표(x 앞, y 왼쪽, z 위) 기준 핀홀, 왜곡 없음."""
    fx = fx_from_hfov(width, hfov)
    v = rpy_to_matrix(*cam_rpy).T @ (np.asarray(point, float) - np.asarray(cam_xyz, float))
    z = v[0]
    if z <= 1e-9:
        return None
    return (width / 2 - fx * v[1] / z, height / 2 - fx * v[2] / z, z)


# ---------------------------------------------------------------- 표본 계획
@dataclass
class PartPlacement:
    slot: str            # 모델 이름(월드 안), 예: NUT_2
    label: int           # 분할 라벨(슬롯 고유)
    cls: str             # 세부 클래스
    yolo_cls: int
    xyz: tuple
    rpy: tuple
    cam_depth_m: float = 0.0
    expected_px: tuple = ()  # 투영 중심(u, v)


@dataclass
class Sample:
    index: int
    parts: list = field(default_factory=list)


@dataclass
class Batch:
    batch: int
    seed: int
    cam_xyz: tuple
    cam_rpy: tuple
    cam_aim: tuple
    light_diffuse: float
    samples: list = field(default_factory=list)


def slot_table():
    """슬롯 이름 → (세부 클래스, 라벨). 라벨은 1부터, 클래스 순서 고정."""
    out = {}
    label = 1
    for cls in CLASSES:
        for k in range(SLOTS_PER_CLASS):
            out[f"{cls}_{k}"] = (cls, label)
            label += 1
    return out


def _sample_camera(rng):
    bx, by, bz = CAM_BASE["xyz"]
    cam = (bx + rng.uniform(-0.05, 0.05), by + rng.uniform(-0.05, 0.05), rng.uniform(0.40, 0.65))
    ax, ay, az = CAM_BASE["aim"]
    aim = (ax + rng.uniform(-0.03, 0.03), ay + rng.uniform(-0.10, 0.10), az)
    return cam, aim


def plan_batch(batch: int, samples_per_batch: int, max_parts=6, min_sep_m=0.035, margin_px=24) -> Batch:
    seed = eval_seed(batch)
    rng = np.random.RandomState(seed)
    cam, aim = _sample_camera(rng)
    cam_rpy = aim_rpy(cam, aim)
    light = float(rng.uniform(0.75, 1.0))
    b = Batch(batch, seed, tuple(round(v, 4) for v in cam), tuple(round(v, 4) for v in cam_rpy),
              tuple(round(v, 4) for v in aim), round(light, 3))
    slots = slot_table()
    for si in range(samples_per_batch):
        n = int(rng.randint(1, max_parts + 1))
        used = {c: 0 for c in CLASSES}
        parts, positions = [], []
        tries = 0
        while len(parts) < n and tries < 500:
            tries += 1
            cls = list(CLASSES)[int(rng.randint(len(CLASSES)))]
            if used[cls] >= SLOTS_PER_CLASS:
                continue
            model, ycls, (r, p), z = CLASSES[cls]
            x = float(rng.uniform(BELT_X[0] + 0.02, BELT_X[1] - 0.02))
            y = float(rng.uniform(BELT_Y[0] + 0.02, BELT_Y[1] - 0.02))
            proj = project(b.cam_xyz, b.cam_rpy, (x, y, z))
            if proj is None or not (margin_px <= proj[0] < IMG_W - margin_px and margin_px <= proj[1] < IMG_H - margin_px):
                continue
            if any(math.hypot(x - px, y - py) < min_sep_m for px, py in positions):
                continue
            yaw = float(rng.uniform(-math.pi, math.pi))
            slot = f"{cls}_{used[cls]}"
            used[cls] += 1
            positions.append((x, y))
            parts.append(PartPlacement(slot, slots[slot][1], cls, ycls, (round(x, 4), round(y, 4), z),
                                       (r, p, round(yaw, 4)), round(float(proj[2]), 4),
                                       (round(float(proj[0]), 1), round(float(proj[1]), 1))))
        b.samples.append(Sample(si, parts))
    return b


# ---------------------------------------------------------------- 월드 사본
def _camera_model_xml(cam_xyz, cam_rpy, width=IMG_W, height=IMG_H, hfov=HFOV):
    pose = " ".join(f"{v:.6f}" for v in (*cam_xyz, *cam_rpy))
    cam = (f"<camera><horizontal_fov>{hfov}</horizontal_fov>"
           f"<image><width>{width}</width><height>{height}</height><format>R8G8B8</format></image>"
           f"<clip><near>0.05</near><far>5.0</far></clip>{{extra}}</camera>")
    return f"""
    <!-- 평가 세트 카메라(같은 링크·같은 pose의 RGB + 분할 카메라). §6.1 경사 예시 기준 변동 -->
    <model name="eval_cam">
      <static>true</static>
      <pose>{pose}</pose>
      <link name="link_cam_link">
        <sensor name="rgb" type="camera">
          <always_on>1</always_on><update_rate>5</update_rate>
          <topic>eval_cam/image</topic>
          <gz_frame_id>link_cam_optical_frame</gz_frame_id>
          {cam.format(extra="")}
        </sensor>
        <sensor name="seg" type="segmentation">
          <always_on>1</always_on><update_rate>5</update_rate>
          <topic>eval_cam/seg</topic>
          <gz_frame_id>link_cam_optical_frame</gz_frame_id>
          {cam.format(extra="<segmentation_type>semantic</segmentation_type>")}
        </sensor>
      </link>
    </model>"""


def _slot_include_xml(slot, cls, label):
    model = CLASSES[cls][0]
    pose = " ".join(f"{v:.4f}" for v in (*PARK[:2], PARK[2], 0, 0, 0))
    return (f'\n    <include><uri>model://{model}</uri><name>{slot}</name><static>true</static>'
            f'<pose>{pose}</pose>'
            f'<plugin filename="gz-sim-label-system" name="gz::sim::systems::Label"><label>{label}</label></plugin>'
            f'</include>')


def build_world(batch: Batch, world_src: Path = WORLD_SRC) -> str:
    """원본 월드를 읽어 사본 문자열을 만든다(원본은 수정하지 않음).

    - 원본의 외형 확인용 부품 3개(part_00~02)와 `appearance_check_cam`을 뺀다.
    - 해의 diffuse를 묶음 조명값으로 바꾼다.
    - 평가 카메라와 18개 부품 슬롯(라벨 1..18, 초기 위치는 화면 밖)을 넣는다.
    """
    src = Path(world_src).read_text(encoding="utf-8")
    out_lines = []
    skip_model = False
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("<include>") and any(f"<name>part_0{k}</name>" in s for k in range(10)):
            continue
        if s.startswith('<model name="appearance_check_cam">'):
            skip_model = True
            continue
        if skip_model:
            if s.startswith("</model>"):
                skip_model = False
            continue
        if s.startswith("<diffuse>0.9 0.9 0.9 1</diffuse>"):
            d = batch.light_diffuse
            line = line.replace("0.9 0.9 0.9 1", f"{d:.3f} {d:.3f} {d:.3f} 1")
        out_lines.append(line)
    text = "\n".join(out_lines) + "\n"
    add = _camera_model_xml(batch.cam_xyz, batch.cam_rpy)
    for slot, (cls, label) in slot_table().items():
        add += _slot_include_xml(slot, cls, label)
    if text.count("  </world>") != 1:
        raise ValueError("원본 월드에서 </world>를 하나로 찾지 못함")
    return text.replace("  </world>", add + "\n  </world>")


def batch_to_json(batch: Batch) -> dict:
    return asdict(batch)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Gazebo 평가 세트 장면 계획·월드 사본 생성(실행 없음)")
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--samples-per-batch", type=int, default=10)
    p.add_argument("--first-batch", type=int, default=0)
    p.add_argument("--out", required=True, help="월드 사본·계획을 쓸 폴더(프로젝트 밖 또는 Git 제외 위치)")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plans = []
    for b in range(a.first_batch, a.first_batch + a.batches):
        batch = plan_batch(b, a.samples_per_batch)
        (out / f"world_b{b:04d}.sdf").write_text(build_world(batch), encoding="utf-8")
        plans.append(batch_to_json(batch))
    (out / "plan.json").write_text(json.dumps(plans, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "batches": len(plans),
                      "samples": sum(len(x["samples"]) for x in plans)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
