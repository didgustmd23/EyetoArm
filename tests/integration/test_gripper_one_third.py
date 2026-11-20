"""시뮬레이션 Robotiq 2F-85 를 모든 길이 1/3 로 고정 축소한 모델의 형상 회귀 시험.

설치된 robotiq_description 원본 매크로를 같은 장착 위치로 펼친 기준 URDF와 우리 모델을 비교한다.
xacro 와 ROS(robotiq_description·ur_description)가 있는 환경(WSL)에서만 돈다. 그 밖에서는 건너뛴다.
실제 Gazebo·로봇은 띄우지 않는다.
"""

import os
import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = [ROOT / "simulation/models/ur5e_2f85", ROOT / "robot_ws/src/ur5e_moveit_config/config"]
S = 1.0 / 3.0
GRIPPER_LINKS = ("base", "left_knuckle", "right_knuckle", "left_finger", "right_finger", "left_inner_knuckle",
                 "right_inner_knuckle", "left_finger_tip", "right_finger_tip")

REFERENCE = """<?xml version="1.0"?>
<robot xmlns:xacro="http://wiki.ros.org/xacro" name="reference">
  <xacro:include filename="$(find robotiq_description)/urdf/robotiq_2f_85_macro.urdf.xacro"/>
  <link name="gripper_mount_link"/>
  <xacro:robotiq_gripper name="robotiq_gripper" prefix="" parent="gripper_mount_link" include_ros2_control="false">
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </xacro:robotiq_gripper>
</robot>
"""


def xacro(path: Path) -> ET.Element:
    return ET.fromstring(subprocess.run(["xacro", str(path)], check=True, capture_output=True, text=True).stdout)


def floats(text: str | None) -> list[float]:
    return [float(v) for v in (text or "0 0 0").split()]


def by_name(root: ET.Element, tag: str) -> dict[str, ET.Element]:
    return {e.get("name"): e for e in root.findall(tag)}


@unittest.skipUnless(shutil.which("xacro") and os.environ.get("AMENT_PREFIX_PATH"), "xacro/ROS 환경 없음")
class GripperOneThird(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            reference = Path(tmp) / "reference.urdf.xacro"
            reference.write_text(REFERENCE, encoding="utf-8")
            cls.ref = xacro(reference)
        cls.models = [xacro(folder / "ur5e_2f85.urdf.xacro") for folder in MODELS]

    def test_both_descriptions_expand_identically(self):
        self.assertEqual(*(ET.tostring(model) for model in self.models))

    def test_gripper_joints_scaled_and_kinematics_rules_kept(self):
        ref_joints = by_name(self.ref, "joint")
        for model in self.models:
            joints = by_name(model, "joint")
            for name, ref in ref_joints.items():
                joint = joints[name]
                self.assertEqual(joint.get("type"), ref.get("type"), name)
                self.assertEqual(joint.find("parent").get("link"), ref.find("parent").get("link"), name)
                self.assertEqual(joint.find("child").get("link"), ref.find("child").get("link"), name)
                for child in ("axis", "limit", "mimic"):
                    a, b = joint.find(child), ref.find(child)
                    self.assertEqual(None if a is None else a.attrib, None if b is None else b.attrib, f"{name} {child}")
                expected = [S * v for v in floats(ref.find("origin").get("xyz"))]
                for got, want in zip(floats(joint.find("origin").get("xyz")), expected):
                    self.assertAlmostEqual(got, want, places=9, msg=name)
                self.assertEqual(floats(joint.find("origin").get("rpy")), floats(ref.find("origin").get("rpy")), name)

    def test_meshes_inertials_scaled_mass_kept(self):
        ref_links = by_name(self.ref, "link")
        for model in self.models:
            links = by_name(model, "link")
            for short in GRIPPER_LINKS:
                name = f"robotiq_85_{short}_link"
                link, ref = links[name], ref_links[name]
                for kind in ("visual", "collision"):
                    mesh, ref_mesh = link.find(f"{kind}/geometry/mesh"), ref.find(f"{kind}/geometry/mesh")
                    self.assertEqual(mesh.get("filename"), ref_mesh.get("filename"), name)
                    self.assertEqual([round(v, 12) for v in floats(mesh.get("scale"))], [round(S, 12)] * 3, name)
                inertial, ref_inertial = link.find("inertial"), ref.find("inertial")
                self.assertEqual(float(inertial.find("mass").get("value")), float(ref_inertial.find("mass").get("value")))
                for got, want in zip(floats(inertial.find("origin").get("xyz")), floats(ref_inertial.find("origin").get("xyz"))):
                    self.assertAlmostEqual(got, S * want, places=12, msg=name)
                for key, value in ref_inertial.find("inertia").attrib.items():
                    self.assertAlmostEqual(float(inertial.find("inertia").get(key)), S * S * float(value), places=15, msg=name)

    def test_arm_adapter_mount_and_control_unchanged(self):
        for model in self.models:
            joints = by_name(model, "joint")
            self.assertEqual(joints["ur_to_robotiq_joint"].find("parent").get("link"), "tool0")
            self.assertEqual(floats(joints["gripper_side_joint"].find("origin").get("xyz")), [0.0, 0.0, 0.011])
            self.assertEqual(floats(joints["robotiq_85_base_joint"].find("origin").get("xyz")), [0.0, 0.0, 0.0])
            adapter = by_name(model, "link")["ur_to_robotiq_link"].find("visual/geometry/mesh")
            self.assertIsNone(adapter.get("scale"))
            self.assertEqual(floats(joints["gripper_cam_joint"].find("origin").get("xyz")), [0.0, 0.0, 0.008])
            control = [j.get("name") for j in model.findall("ros2_control/joint")]
            self.assertIn("robotiq_85_left_knuckle_joint", control)
            self.assertIn("wrist_3_joint", control)
            clips = {s.get("name"): s.find("camera/clip") for s in model.findall("gazebo/sensor")}
            self.assertEqual((clips["gripper_rgbd"].findtext("near"), clips["gripper_rgbd"].findtext("far")), ("0.01", "5.0"))
            self.assertEqual(clips["link_cam"].findtext("near"), "0.05")


if __name__ == "__main__":
    unittest.main()
