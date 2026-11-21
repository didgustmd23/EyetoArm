"""그리퍼 깊이 관측기 시험: 합성 깊이에서 끝점·윗면·무효 사유."""
from __future__ import annotations

import json
import math
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

import gripper_scene as gs
from eyetoarm.vision.gripper_depth import GripperConfigError, observe, parse_gripper, project_w

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "vision" / "world_tracker_gripper_test.json"
TOL_MM = 0.5


def gcfg(**over):
    """시험 설정(예시 파일 값 + 덮어쓰기)."""
    doc = json.loads(CONFIG.read_text(encoding="utf-8"))["gripper"]
    doc.update(over)

    return parse_gripper(doc)


def obs(T, bolts, **kw):
    """합성 프레임 하나를 관측한다."""
    frame = gs.render(T, bolts, **kw)

    return observe(frame, T, gcfg(), gs.ROI_XY_M)


class ObserveTest(unittest.TestCase):
    """정상 경로."""

    def check(self, bolt, T):
        """한 볼트의 끝점·길이·윗면이 그린 기하와 같은지 본다."""
        o = obs(T, [bolt])
        self.assertTrue(o.valid, o.reason)
        self.assertEqual(len(o.components), 1)
        c = o.components[0]
        self.assertFalse(c.clipped)
        self.assertLess(np.linalg.norm(np.array(c.endpoints_mm[0]) - bolt.head_end_mm()), TOL_MM)
        self.assertLess(np.linalg.norm(np.array(c.endpoints_mm[1]) - bolt.tip_mm()), TOL_MM)
        self.assertLess(abs(c.length_mm - (bolt.head_len + bolt.shank_len) * 1000), 2 * TOL_MM)
        self.assertLess(abs(c.top_z_mm - bolt.top_z_mm()), TOL_MM)
        expect = (bolt.yaw_deg % 180.0)
        diff = abs(c.angle_deg - expect) % 180.0
        self.assertLess(min(diff, 180 - diff), 1.0)

    def test_m3_straight_down(self):
        """M3x20을 바로 위에서 재면 끝점·윗면이 0.5 mm 안이다."""
        self.check(gs.m3(yaw_deg=30.0), gs.camera_above(height_m=0.10))

    def test_m6_tilted_camera(self):
        """M6x40을 기운 카메라로 재도 끝점·윗면이 0.5 mm 안이다."""
        self.check(gs.m6(yaw_deg=-60.0), gs.camera_above(height_m=0.12, tilt_m=0.03))

    def test_top_ignores_low_render_belt(self):
        """그려진 벨트가 물리 벨트보다 4 mm 낮아도 윗면 z는 실제 값이다."""
        b = gs.m3()
        T = gs.camera_above(height_m=0.10)
        low = observe(gs.render(T, [b], belt_z_m=0.096), T, gcfg(), gs.ROI_XY_M)
        level = observe(gs.render(T, [b], belt_z_m=0.100), T, gcfg(), gs.ROI_XY_M)
        self.assertAlmostEqual(low.components[0].top_z_mm, level.components[0].top_z_mm, delta=0.05)
        self.assertLess(abs(low.diag["base_z_mm"] - 96.0), 0.5)

    def test_two_bolts_two_components(self):
        """떨어진 두 볼트는 성분 두 개다."""
        o = obs(gs.camera_above(height_m=0.15), [gs.m3(y=-0.03), gs.m6(y=0.03)])
        self.assertEqual(len(o.components), 2)

    def test_project_w_inverse(self):
        """W 점을 투영한 픽셀에서 깊이로 되돌리면 같은 점이다."""
        T = gs.camera_above(height_m=0.12, tilt_m=0.02)
        uv, zc = project_w([[550.0, 10.0, 100.0]], T, gs.K)
        self.assertTrue(0 <= uv[0, 0] < gs.W and 0 <= uv[0, 1] < gs.H and zc[0] > 0)


class InvalidTest(unittest.TestCase):
    """무효 사유."""

    def test_near_invalid(self):
        """가운데가 near 미만(0)이면 depth_invalid."""
        T = gs.camera_above(height_m=0.12)
        frame = gs.render(T, [gs.m3()])
        d = np.array(frame.depth_m)
        d[100:380, 150:490] = 0.0
        o = observe(replace(frame, depth_m=d), T, gcfg(), gs.ROI_XY_M)
        self.assertEqual(o.reason, "depth_invalid")

    def test_nan_inf_invalid(self):
        """nan·inf도 무효 화소로 센다."""
        T = gs.camera_above(height_m=0.12)
        frame = gs.render(T, [])
        d = np.array(frame.depth_m)
        d[120:360, 160:320] = np.nan
        d[120:360, 320:480] = np.inf
        self.assertEqual(observe(replace(frame, depth_m=d), T, gcfg(), gs.ROI_XY_M).reason, "depth_invalid")

    def test_empty_belt(self):
        """빈 벨트는 no_component."""
        self.assertEqual(obs(gs.camera_above(), []).reason, "no_component")

    def test_off_belt(self):
        """벨트 사각형 밖만 보이면 no_belt_plane."""
        T = gs.camera_above(x=1.0, height_m=0.10)
        self.assertEqual(obs(T, []).reason, "no_belt_plane")

    def test_clipped(self):
        """화면 가장자리에 걸친 성분은 clipped."""
        T = gs.camera_above(height_m=0.04)
        o = obs(T, [gs.m6(x=0.55 - 0.01)])
        self.assertTrue(o.valid)
        self.assertTrue(any(c.clipped for c in o.components))

    def test_tall_object_excluded(self):
        """받침면 위 h_max 넘는 물체(들어 올린 볼트·손가락)는 성분이 아니다."""
        T = gs.camera_above(height_m=0.12)
        frame = gs.render(T, [], belt_z_m=0.096)
        d = np.array(frame.depth_m)
        d[230:250, 300:340] = 0.03   # 렌즈 앞 3 cm 물체
        o = observe(replace(frame, depth_m=d), T, gcfg(), gs.ROI_XY_M)
        self.assertEqual(o.reason, "no_component")


class ConfigTest(unittest.TestCase):
    """설정 검사."""

    def test_missing_and_unknown(self):
        """키 누락·모르는 키는 거부."""
        doc = json.loads(CONFIG.read_text(encoding="utf-8"))["gripper"]

        with self.assertRaises(GripperConfigError):
            parse_gripper({k: v for k, v in doc.items() if k != "near_m"})

        with self.assertRaises(GripperConfigError):
            parse_gripper({**doc, "extra": 1})

    def test_bad_values(self):
        """음수·비유한·bool·빈 문자열·범위 오류는 거부."""
        for key, bad in (("near_m", -0.01), ("gate_x_mm", math.inf), ("min_px", True), ("optical_frame", ""),
                         ("invalid_ratio_max", 1.5), ("h_min_mm", 50.0)):
            with self.subTest(key=key), self.assertRaises(GripperConfigError):
                gcfg(**{key: bad})


if __name__ == "__main__":
    unittest.main()
