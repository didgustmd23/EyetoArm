#!/usr/bin/env python3
"""육각 머리 볼트(ISO 4017) 시뮬레이션 모델 생성기: `bolt_m3x20`·`bolt_m6x40`의 model.sdf·model.config·meshes/*.stl.

    python3 scripts/generate_hex_bolt_models.py [OUT_MODELS_DIR]    (기본 simulation/models)

치수 출처: ISO 4017 표준 치수(s·k), 호칭값. M3 s 5.5 k 2.0 d 3 l 20 / M6 s 10 k 4.0 d 6 l 40 (mm).
맞모서리 e는 날카로운 육각 e = 2s/√3(모따기·모서리 둥글림 생략; ISO e_min M3 6.01, M6 11.05).
나사산은 표현하지 않는다(매끈한 원기둥).
원점·축: 지금 모델과 같다(원점 = 머리·축 경계, +Z = 머리 쪽). 육각은 맞변이 모델 x축에 수직(볼트를 pitch 90°로 눕히면
맞변 하나가 아래를 향한다).
충돌: 머리 = 상자 3개(크기 s × a × k, a = s/√3, 0·60·120° 회전)의 합집합 = 정육각 기둥(DART 기본 도형만), 축 = 원기둥.
질량·관성: 강철 7850 kg/m³, 생성한 닫힌 메시에서 적분(발산 정리).
"""
from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

import numpy as np

RHO = 7850.0
SEG = 48                                   # 축 원기둥 둘레 분할
SPECS = {  # 이름: (s, k, d, l) mm
    "bolt_m3x20": (5.5, 2.0, 3.0, 20.0),
    "bolt_m6x40": (10.0, 4.0, 6.0, 40.0),
}


def hex_prism(s, z0, z1):
    """맞변 s, z0~z1 육각 기둥 삼각형 목록(mm). 꼭짓점 각 30+60i° (맞변이 x축에 수직)."""
    a = s / math.sqrt(3.0)
    ring = [(a * math.cos(math.radians(30 + 60 * i)), a * math.sin(math.radians(30 + 60 * i))) for i in range(6)]
    tris = []

    for i in range(6):
        (x0, y0), (x1, y1) = ring[i], ring[(i + 1) % 6]
        tris += [((x0, y0, z0), (x1, y1, z0), (x1, y1, z1)), ((x0, y0, z0), (x1, y1, z1), (x0, y0, z1))]   # 옆면(바깥)
        tris += [((0, 0, z1), (x0, y0, z1), (x1, y1, z1))]                                                  # 윗면(+z)
        tris += [((0, 0, z0), (x1, y1, z0), (x0, y0, z0))]                                                  # 아랫면(−z)

    return tris


def cylinder(r, z0, z1, seg=SEG):
    """반지름 r, z0~z1 원기둥 삼각형 목록(mm)."""
    ring = [(r * math.cos(2 * math.pi * i / seg), r * math.sin(2 * math.pi * i / seg)) for i in range(seg)]
    tris = []

    for i in range(seg):
        (x0, y0), (x1, y1) = ring[i], ring[(i + 1) % seg]
        tris += [((x0, y0, z0), (x1, y1, z0), (x1, y1, z1)), ((x0, y0, z0), (x1, y1, z1), (x0, y0, z1))]
        tris += [((0, 0, z1), (x0, y0, z1), (x1, y1, z1))]
        tris += [((0, 0, z0), (x1, y1, z0), (x0, y0, z0))]

    return tris


def mass_props(tris_m):
    """닫힌 삼각형 메시(m) → (부피, 질량중심, 질량중심 기준 관성 텐서) (균일 밀도 RHO)."""
    vol = 0.0
    c = np.zeros(3)
    second = np.zeros((3, 3))   # ∫ x x^T dV

    for a, b, d in tris_m:
        a, b, d = map(np.asarray, (a, b, d))
        v = np.dot(a, np.cross(b, d)) / 6.0
        vol += v
        c += v * (a + b + d) / 4.0
        # 사면체(원점, a, b, d)의 2차 모멘트
        pts = (a, b, d)
        s = sum(np.outer(p, p) for p in pts) + np.outer(sum(pts), sum(pts))
        second += v * s / 20.0

    c /= vol
    m = RHO * vol
    cov = RHO * second - m * np.outer(c, c)
    inertia = np.trace(cov) * np.eye(3) - cov

    return vol, m, c, inertia


def write_stl(path: Path, tris_mm):
    """이진 STL(mm 단위, SDF에서 0.001 배율)."""
    with open(path, "wb") as f:
        f.write(b"EyetoArm ISO 4017 hex bolt (generated)".ljust(80, b" "))
        f.write(struct.pack("<I", len(tris_mm)))

        for a, b, d in tris_mm:
            n = np.cross(np.subtract(b, a), np.subtract(d, a))
            n = n / (np.linalg.norm(n) or 1.0)
            f.write(struct.pack("<12fH", *n, *a, *b, *d, 0))


def model(out: Path, name: str):
    """모델 하나(`out/name/`)를 쓰고 치수·질량 요약을 돌려준다."""
    s, k, d, l = SPECS[name]
    a = s / math.sqrt(3.0)
    tris = hex_prism(s, 0.0, k) + cylinder(d / 2.0, -l, 0.0)
    head_m = [tuple(tuple(v / 1000.0 for v in p) for p in t) for t in hex_prism(s, 0.0, k)]
    shank_m = [tuple(tuple(v / 1000.0 for v in p) for p in t) for t in cylinder(d / 2.0, -l, 0.0)]
    vol_h, m_h, c_h, i_h = mass_props(head_m)
    vol_s, m_s, c_s, i_s = mass_props(shank_m)
    m = m_h + m_s
    c = (m_h * c_h + m_s * c_s) / m
    c[:2] = 0.0   # 축 대칭: 적분 반올림 잔차를 지운다
    inertia = (i_h + m_h * (np.dot(c_h - c, c_h - c) * np.eye(3) - np.outer(c_h - c, c_h - c))
               + i_s + m_s * (np.dot(c_s - c, c_s - c) * np.eye(3) - np.outer(c_s - c, c_s - c)))

    d_dir = out / name
    (d_dir / "meshes").mkdir(parents=True, exist_ok=True)
    write_stl(d_dir / "meshes" / f"{name}.stl", tris)

    boxes = "".join(
        f'\n      <collision name="head{i}"><pose>0 0 {k / 2000.0:.6g} 0 0 {math.radians(60 * i):.6f}</pose>'
        f'\n        <geometry><box><size>{s / 1000.0:.6g} {a / 1000.0:.6g} {k / 1000.0:.6g}</size></box></geometry></collision>'
        for i in range(3))
    sdf = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <!-- ISO 4017 육각 머리 볼트 {name[5:].upper().replace('X', '×')}(scripts/generate_hex_bolt_models.py 생성, 매끈한 축·모따기 생략). 원점 = 머리·축 경계, +Z 머리 쪽.
       맞변 s {s} mm, 머리 높이 k {k} mm, 맞모서리 e {2 * a:.2f} mm, 축 d {d} mm × l {l} mm, 전체 {l + k:g} mm.
       충돌: 머리 = 상자 3개(0·60·120°) 합집합 = 정육각 기둥, 축 = 원기둥. 질량·관성: 강철 7850 kg/m³ 메시 적분 -->
  <model name="{name}">
    <link name="link">
      <inertial>
        <pose>{c[0]:.6g} {c[1]:.6g} {c[2]:.6g} 0 0 0</pose>
        <mass>{m:.6g}</mass>
        <inertia><ixx>{inertia[0, 0]:.4e}</ixx><iyy>{inertia[1, 1]:.4e}</iyy><izz>{inertia[2, 2]:.4e}</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>{boxes}
      <collision name="shank"><pose>0 0 {-l / 2000.0:.6g} 0 0 0</pose>
        <geometry><cylinder><radius>{d / 2000.0:.6g}</radius><length>{l / 1000.0:.6g}</length></cylinder></geometry></collision>
      <visual name="visual">
        <geometry><mesh><uri>model://{name}/meshes/{name}.stl</uri><scale>0.001 0.001 0.001</scale></mesh></geometry>
        <material>
          <ambient>0.468 0.480 0.504 1</ambient>
          <diffuse>0.78 0.8 0.84 1</diffuse>
          <specular>0.3 0.3 0.3 1</specular>
          <pbr><metal><metalness>0.35</metalness><roughness>0.35</roughness></metal></pbr>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""
    (d_dir / "model.sdf").write_text(sdf, encoding="utf-8")
    (d_dir / "model.config").write_text(f"""<?xml version="1.0"?>
<model>
  <name>{name}</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <author><name>EyetoArm</name></author>
  <description>ISO 4017 hex head bolt M{d:g}x{l:g}, nominal dimensions (s {s} k {k} d {d} l {l} mm, total {l + k:g} mm). Generated by scripts/generate_hex_bolt_models.py; no third-party mesh.</description>
</model>
""", encoding="utf-8")

    return {"name": name, "s_mm": s, "k_mm": k, "e_mm": round(2 * a, 3), "d_mm": d, "l_mm": l, "total_mm": l + k,
            "mass_kg": round(m, 7), "com_z_mm": round(c[2] * 1000, 3), "ixx": float(inertia[0, 0]), "izz": float(inertia[2, 2]),
            "triangles": len(tris)}


def main() -> int:
    """모델 두 개를 만들고 치수·질량 요약을 한 줄씩 출력한다."""
    import json

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "simulation" / "models"

    for n in SPECS:
        print(json.dumps(model(out, n), ensure_ascii=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
