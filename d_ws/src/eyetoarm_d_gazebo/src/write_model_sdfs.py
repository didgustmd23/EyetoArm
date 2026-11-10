#!/usr/bin/env python3
"""models/*/model.config, model.sdf 생성(색상·재질·충돌·관성). 실행: python3 src/write_model_sdfs.py"""
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "models"

MAT = {  # (diffuse rgb, metalness, roughness)
    "zinc_bolt": ((0.78, 0.80, 0.84), 0.35, 0.35),
    "zinc_nut": ((0.86, 0.82, 0.66), 0.35, 0.35),
    "alu_frame": ((0.74, 0.76, 0.79), 0.30, 0.50),
    "steel_roller": ((0.60, 0.61, 0.63), 0.40, 0.40),
    "rubber_belt": ((0.12, 0.13, 0.14), 0.0, 0.90),
    "bin_gray": ((0.60, 0.60, 0.60), 0.0, 0.60),
    "bin_bolt_a": ((0.18, 0.44, 0.84), 0.0, 0.55),
    "bin_bolt_b": ((0.21, 0.63, 0.38), 0.0, 0.55),
    "bin_nut": ((0.91, 0.54, 0.17), 0.0, 0.55),
}


def material(key):
    (r, g, b), metal, rough = MAT[key]
    return f"""<material>
          <ambient>{r*.6:.3f} {g*.6:.3f} {b*.6:.3f} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
          <specular>0.3 0.3 0.3 1</specular>
          <pbr><metal><metalness>{metal}</metalness><roughness>{rough}</roughness></metal></pbr>
        </material>"""


def config(name, desc, author="EyetoArm D", extra=""):
    return f"""<?xml version="1.0"?>
<model>
  <name>{name}</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <author><name>{author}</name></author>
  <description>{desc}{extra}</description>
</model>
"""


def inertial(mass, ixx, iyy, izz, z=0.0):
    return f"""<inertial>
        <pose>0 0 {z:.4f} 0 0 0</pose>
        <mass>{mass:.6f}</mass>
        <inertia><ixx>{ixx:.3e}</ixx><iyy>{iyy:.3e}</iyy><izz>{izz:.3e}</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>"""


def cyl_inertia(m, r, h):
    return m * (3 * r * r + h * h) / 12, m * (3 * r * r + h * h) / 12, m * r * r / 2


def write(name, cfg, sdf):
    d = ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.config").write_text(cfg, encoding="utf-8")
    (d / "model.sdf").write_text(sdf, encoding="utf-8")


ATTR = (" Mesh: FreeCAD-library ({file}), CC-BY 3.0, author {author}, "
        "https://github.com/FreeCAD/FreeCAD-library commit f5ca01cdcaa2f69b9295a6382719c2d90628780c. "
        "Unit mm, scaled 0.001.")
RHO = 7850.0
for name, length, file, author in (
        ("bolt_m6x12", 0.012, "ISO4762_Hex_Socket_Head_Cap_Screw_M6x12.stl", "Normand C"),
        ("bolt_m6x40", 0.040, "ISO4762_Hex_Socket_Head_Cap_Screw_M6x40.stl", "obijuan")):
    m = RHO * (math.pi * 0.005**2 * 0.006 + math.pi * 0.003**2 * length)
    ixx, iyy, izz = cyl_inertia(m, 0.004, length + 0.006)
    sdf = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <!-- ISO 4762 M6 hex socket head cap screw. 원점=머리 윗면 아래 6 mm(머리·축 경계), +Z가 머리 쪽 -->
  <model name="{name}">
    <link name="link">
      {inertial(m, ixx, iyy, izz, z=0.003 - length / 2)}
      <collision name="head"><pose>0 0 0.003 0 0 0</pose>
        <geometry><cylinder><radius>0.005</radius><length>0.006</length></cylinder></geometry></collision>
      <collision name="shank"><pose>0 0 {-length/2:.4f} 0 0 0</pose>
        <geometry><cylinder><radius>0.003</radius><length>{length}</length></cylinder></geometry></collision>
      <visual name="visual">
        <geometry><mesh><uri>model://{name}/meshes/{name}.stl</uri><scale>0.001 0.001 0.001</scale></mesh></geometry>
        {material("zinc_bolt")}
      </visual>
    </link>
  </model>
</sdf>
"""
    write(name, config(name, f"ISO 4762 M6x{int(length*1000)} socket head cap screw, zinc finish.", author, ATTR.format(file=file, author=author)), sdf)

m_nut = RHO * ((math.sqrt(3) / 2) * 0.010**2 - math.pi * 0.003**2) * 0.0052
ixx, iyy, izz = cyl_inertia(m_nut, 0.0055, 0.0052)
write("nut_m6", config("nut_m6", "ISO 4032 M6 hex nut, zinc finish.", "Normand C",
                       ATTR.format(file="ISO4032_Hex_Nut_M6.stl", author="Normand C")), f"""<?xml version="1.0"?>
<sdf version="1.9">
  <!-- ISO 4032 M6 hex nut. 원점=아랫면 중심, 두께 5.2 mm, 맞변 10 mm -->
  <model name="nut_m6">
    <link name="link">
      {inertial(m_nut, ixx, iyy, izz, z=0.0026)}
      <collision name="collision"><pose>0 0 0.0026 0 0 0</pose>
        <geometry><cylinder><radius>0.0053</radius><length>0.0052</length></cylinder></geometry></collision>
      <visual name="visual">
        <geometry><mesh><uri>model://nut_m6/meshes/nut_m6.stl</uri><scale>0.001 0.001 0.001</scale></mesh></geometry>
        {material("zinc_nut")}
      </visual>
    </link>
  </model>
</sdf>
""")

# 컨베이어: 외형 전용 정적 모델. 벨트 구동·정지 제어기는 C 소유(계획서 §3)
write("conveyor_belt", config("conveyor_belt", "Belt conveyor appearance, placeholder size 1.2 x 0.2 m, belt top 0.10 m (B 확정 전 임시값)."), f"""<?xml version="1.0"?>
<sdf version="1.9">
  <!-- 원점=바닥 중심, 진행 방향 +X, 벨트 윗면 Z 0.100 m(임시값). 구동 없음(외형 전용) -->
  <model name="conveyor_belt">
    <static>true</static>
    <link name="link">
      <collision name="belt_top"><pose>0 0 0.090 0 0 0</pose>
        <geometry><box><size>1.26 0.20 0.020</size></box></geometry></collision>
      <visual name="frame">
        <geometry><mesh><uri>model://conveyor_belt/meshes/frame.stl</uri></mesh></geometry>
        {material("alu_frame")}
      </visual>
      <visual name="rollers">
        <geometry><mesh><uri>model://conveyor_belt/meshes/rollers.stl</uri></mesh></geometry>
        {material("steel_roller")}
      </visual>
      <visual name="belt">
        <geometry><mesh><uri>model://conveyor_belt/meshes/belt.stl</uri></mesh></geometry>
        {material("rubber_belt")}
      </visual>
    </link>
  </model>
</sdf>
""")


def bin_sdf(name, mat):
    w, x, y, z = 0.004, 0.160, 0.120, 0.080
    walls = [("floor", 0, 0, w / 2, x, y, w), ("wall_py", 0, (y - w) / 2, z / 2, x, w, z), ("wall_ny", 0, -(y - w) / 2, z / 2, x, w, z),
             ("wall_px", (x - w) / 2, 0, z / 2, w, y, z), ("wall_nx", -(x - w) / 2, 0, z / 2, w, y, z)]
    cols = "\n".join(f"""      <collision name="{n}"><pose>{cx} {cy} {cz} 0 0 0</pose>
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry></collision>""" for n, cx, cy, cz, sx, sy, sz in walls)
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <!-- 윗면이 열린 상자 160 x 120 x 80 mm, 벽 4 mm(임시값). 원점=바닥 중심 -->
  <model name="{name}">
    <static>true</static>
    <link name="link">
{cols}
      <visual name="visual">
        <geometry><mesh><uri>model://sort_bin/meshes/bin.stl</uri></mesh></geometry>
        {material(mat)}
      </visual>
    </link>
  </model>
</sdf>
"""


for name, mat, desc in (("sort_bin", "bin_gray", "Open sort bin (neutral)."),
                        ("sort_bin_bolt_a", "bin_bolt_a", "Sort bin for BOLT_A (M6x12), blue."),
                        ("sort_bin_bolt_b", "bin_bolt_b", "Sort bin for BOLT_B (M6x40), green."),
                        ("sort_bin_nut", "bin_nut", "Sort bin for NUT (M6), orange.")):
    write(name, config(name, desc + " Placeholder size (B 확정 전)."), bin_sdf(name, mat))
print("models written")
