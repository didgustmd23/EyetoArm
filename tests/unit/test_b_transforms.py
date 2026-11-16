"""B 보정·위치 변환·판정·주석 영상 단위시험.

근거: B 상세설계 §6(렌즈 보정)·§6.2(원본 픽셀 → W, 1회 보정)·§7(높이별 Homography·중심·각도)·§15(규격).

- 기대값은 정답 자세·치수에서 손으로 정한 값이다. 윤곽은 정답 W 점을 카메라로 투영해 만든다
  (래스터화 없음). 그래서 B 계산은 수치 오차 수준(1e-6 m)으로 정답과 같아야 한다.
- 설정 파일은 저장소의 config/robot/frames.yaml, config/calibration/part_specs.json을 쓴다.
"""
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from eyetoarm.calibration.profile import CameraProfile, from_camera_info, from_fov
from eyetoarm.transforms.overlay import draw_guides, draw_results
from eyetoarm.transforms.planar_transform import (
    homography_w_to_image, inv_T, look_at, make_T, pixels_to_w, quat_to_rot, rot_rpy,
    tf_matrix_from_echo, w_to_image,
)
from eyetoarm.transforms.work_result import compute_work_result, load_specs

ROOT = Path(__file__).resolve().parents[2]
FRAMES = yaml.safe_load(open(ROOT / "config/robot/frames.yaml", encoding="utf-8"))
SPECS = load_specs(ROOT / "config/calibration/part_specs.json")
CAM = from_fov(1280, 720, 1.0, "link_cam_optical_frame")
T_W_C = look_at((0.30, -0.08, 0.40), (0.56, -0.07, 0.10))
Z = SPECS["plane_z"]


def bolt(total_len, cx, cy, deg):
    """누운 볼트 윤곽(W, m). 머리 지름 10 mm·높이 6 mm, 축 지름 6 mm."""
    x0, xs, xe = -total_len / 2, -total_len / 2 + 0.006, total_len / 2
    local = np.array([[x0, -0.005], [xs, -0.005], [xs, -0.003], [xe, -0.003],
                      [xe, 0.003], [xs, 0.003], [xs, 0.005], [x0, 0.005]])
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return local @ np.array([[c, s], [-s, c]]) + [cx, cy]


def to_px(xy, cam=CAM, T=T_W_C, z=Z):
    return w_to_image(np.column_stack([xy, np.full(len(xy), z)]), cam.K, cam.dist, T)


def area_centroid(xy):
    x, y = xy[:, 0], xy[:, 1]
    cr = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = cr.sum() / 2
    return np.array([((x + np.roll(x, -1)) * cr).sum(), ((y + np.roll(y, -1)) * cr).sum()]) / (6 * a)


class TestProfile(unittest.TestCase):
    def test_from_fov(self):
        self.assertAlmostEqual(CAM.K[0, 0], 640 / math.tan(0.5), places=9)
        self.assertEqual((CAM.K[0, 2], CAM.K[1, 2]), (640.0, 360.0))
        self.assertFalse(np.any(CAM.dist))

    def test_calibration_id_stable_and_sensitive(self):
        again = from_fov(1280, 720, 1.0, "x")
        other = from_fov(1280, 720, 1.1, "x")
        self.assertEqual(CAM.calibration_id, again.calibration_id)
        self.assertNotEqual(CAM.calibration_id, other.calibration_id)

    def test_from_camera_info(self):
        msg = SimpleNamespace(width=640, height=480, k=[500, 0, 320, 0, 500, 240, 0, 0, 1], d=[],
                              header=SimpleNamespace(frame_id="gripper_cam_optical_frame"))
        p = from_camera_info(msg)
        self.assertEqual(p.frame_id, "gripper_cam_optical_frame")
        self.assertEqual(p.dist.shape, (5,))

    def test_rejects_bad_K(self):
        with self.assertRaises(ValueError):
            CameraProfile(10, 10, np.zeros((3, 3)), np.zeros(5), "x")


class TestPlanarTransform(unittest.TestCase):
    def test_inv_and_rotations(self):
        T = make_T(rot_rpy(0.3, -0.2, 1.1), [0.1, 0.2, 0.3])
        np.testing.assert_allclose(inv_T(T) @ T, np.eye(4), atol=1e-12)
        R = rot_rpy(0, 0, math.pi / 2)
        np.testing.assert_allclose(quat_to_rot(0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)), R, atol=1e-12)

    def test_look_at_optical_axes(self):
        T = look_at((0, 0, 1), (1, 0, 0))
        z = T[:3, 2]
        np.testing.assert_allclose(z, np.array([1, 0, -1]) / math.sqrt(2), atol=1e-12)
        self.assertLess(T[2, 1], 0)  # 광학 Y(아래)는 월드 아래쪽 성분을 가진다

    def test_roundtrip_pixel_w(self):
        pts = np.array([[0.50, -0.10], [0.60, 0.05], [0.56, -0.07]])
        px = to_px(pts)
        np.testing.assert_allclose(pixels_to_w(px, CAM.K, CAM.dist, T_W_C, Z), pts, atol=1e-9)

    def test_roundtrip_with_distortion(self):
        cam = CameraProfile(1280, 720, CAM.K, [-0.25, 0.08, 0.0005, -0.0004, 0.0], "x")
        pts = np.array([[0.50, -0.10], [0.62, -0.02]])
        px = to_px(pts, cam=cam)
        np.testing.assert_allclose(pixels_to_w(px, cam.K, cam.dist, T_W_C, Z), pts, atol=2e-5)

    def test_wrong_height_shifts(self):
        px = to_px(np.array([[0.56, -0.07]]))
        at_belt = pixels_to_w(px, CAM.K, CAM.dist, T_W_C, FRAMES["belt"]["top_z"])
        self.assertGreater(np.linalg.norm(at_belt - [0.56, -0.07]), 0.002)

    def test_homography_matches_projection(self):
        H = homography_w_to_image(CAM.K, T_W_C, Z)
        p = H @ [0.55, 0.0, 1.0]
        np.testing.assert_allclose(p[:2] / p[2], to_px(np.array([[0.55, 0.0]]))[0], atol=1e-6)

    def test_behind_camera_is_nan(self):
        behind = T_W_C[:3, 3] - T_W_C[:3, 2]
        self.assertTrue(np.all(np.isnan(w_to_image([behind], CAM.K, CAM.dist, T_W_C))))

    def test_tf_echo_parse(self):
        text = """- Matrix:
 -0.000 -1.000 -0.001  0.492
 -1.000  0.000  0.001  0.133
 -0.001  0.001 -1.000  0.488
  0.000  0.000  0.000  1.000"""
        T = tf_matrix_from_echo(text)
        self.assertEqual(T.shape, (4, 4))
        self.assertAlmostEqual(T[0, 3], 0.492)


class TestWorkResult(unittest.TestCase):
    def check_bolt(self, name, deg):
        L = SPECS["specs"][name]["total_length"]
        xy = bolt(L, 0.55, -0.05, deg)
        r = compute_work_result(to_px(xy), CAM, T_W_C, SPECS, class_name="bolt")
        self.assertTrue(r.valid, r.reason)
        self.assertEqual(r.spec, name)
        np.testing.assert_allclose(r.center_w_m, area_centroid(xy), atol=1e-6)
        self.assertAlmostEqual(r.measured_length_m, L, delta=1e-6)
        self.assertAlmostEqual(abs((r.theta_deg - deg + 90) % 180 - 90), 0.0, delta=1e-4)

    def test_bolts_all_angles(self):
        for name in ("BOLT_A", "BOLT_B"):
            for deg in (0, 30, 89, 135, 179):
                with self.subTest(name=name, deg=deg):
                    self.check_bolt(name, deg)

    def test_theta_folded(self):
        r = compute_work_result(to_px(bolt(0.046, 0.55, 0, 200)), CAM, T_W_C, SPECS, class_name="bolt")
        self.assertTrue(0 <= r.theta_deg < 180)
        self.assertAlmostEqual(r.theta_deg, 20.0, delta=1e-4)

    def test_nut(self):
        r_hex = 0.010 / math.sqrt(3)
        xy = np.array([[0.58 + r_hex * math.cos(math.radians(30 + 60 * k)),
                        0.09 + r_hex * math.sin(math.radians(30 + 60 * k))] for k in range(6)])
        r = compute_work_result(to_px(xy), CAM, T_W_C, SPECS)
        self.assertEqual(r.spec, "NUT")
        self.assertFalse(r.theta_valid)
        self.assertIsNone(r.theta_deg)
        np.testing.assert_allclose(r.center_w_m, (0.58, 0.09), atol=1e-6)

    def test_length_out_of_range(self):
        r = compute_work_result(to_px(bolt(0.030, 0.55, 0, 0)), CAM, T_W_C, SPECS, class_name="bolt")
        self.assertFalse(r.valid)
        self.assertEqual((r.spec, r.reason, r.reason_owner), ("UNRESOLVED", "LENGTH_OUT_OF_RANGE", "B"))

    def test_calibration_mismatch(self):
        r = compute_work_result(to_px(bolt(0.018, 0.55, 0, 0)), CAM, T_W_C, SPECS,
                                class_name="bolt", observed_calibration_id="cal-other")
        self.assertFalse(r.valid)
        self.assertEqual((r.reason, r.reason_owner), ("CALIBRATION_MISMATCH", "A"))

    def test_too_small(self):
        r = compute_work_result([[10, 10]], CAM, T_W_C, SPECS)
        self.assertFalse(r.valid)
        self.assertEqual(r.reason, "CONTOUR_TOO_SMALL")


class TestOverlay(unittest.TestCase):
    def test_draws_something(self):
        img = np.zeros((720, 1280, 3), np.uint8)
        r = compute_work_result(to_px(bolt(0.046, 0.55, -0.05, 40)), CAM, T_W_C, SPECS, class_name="bolt")
        draw_guides(img, CAM, T_W_C, FRAMES)
        before = int(img.sum())
        draw_results(img, CAM, T_W_C, [r])
        self.assertGreater(int(img.sum()), before)


if __name__ == "__main__":
    unittest.main()


# ---------- A 관측 → B 결과 → C 목표 (DB-01·DB-02) ----------
from eyetoarm.common.types import FramePacket, PlanarObservation, TimestampBase  # noqa: E402
from eyetoarm.transforms.grasp import build_sort_goal, down_quat_xyzw  # noqa: E402
from eyetoarm.transforms.work_result import from_observation, to_output_dict  # noqa: E402


def make_frame(seq=7):
    return FramePacket(image=np.zeros((720, 1280, 3), np.uint8), run_id="r1", source_id="link_cam",
                       image_seq=seq, timestamp_ms=94264, timestamp_base=TimestampBase.SIM_TIME,
                       received_monotonic_ns=1, width=1280, height=720, crop=None,
                       calibration_id=CAM.calibration_id, capture_generation=0,
                       optical_frame="link_cam_optical_frame", ros_stamp_ns=94_264_000_000)


def make_obs(contour, cls="bolt", valid=True, seq=7, cal=None):
    return PlanarObservation(run_id="r1", source_id="link_cam", image_seq=seq, capture_generation=0,
                             generation=0, job_seq=1, detector_config_id="det", pixel_basis="raw",
                             calibration_id=cal or CAM.calibration_id,
                             observed_face="silhouette" if valid else None,
                             calc_contour=contour if valid else None, display_mask_contour=None,
                             candidate_count=1, selected_index=0 if valid else None, valid=valid,
                             reason=None if valid else "NO_DETECTION", process_ms=1.0, method="ai",
                             class_name=cls, confidence=0.9, model_id="yolo")


class TestObservationToGoal(unittest.TestCase):
    def test_bolt_chain(self):
        xy = bolt(0.046, 0.55, -0.05, 30)
        r = from_observation(make_obs(to_px(xy).astype(np.float32)), make_frame(), CAM, T_W_C, SPECS)
        self.assertTrue(r.valid, r.reason)
        self.assertEqual(r.spec, "BOLT_B")
        out = to_output_dict(r, "link_cam")
        self.assertAlmostEqual(out["length_mm"], 46.0, delta=0.2)
        self.assertEqual(out["image_seq"], 7)

        goal = build_sort_goal(r, source_id="link_cam", stamp_ns=94_264_000_000, frames=FRAMES, specs=SPECS)
        self.assertEqual(goal["kind"], "PICK_PLACE")
        self.assertEqual(goal["frame_id"], "base_link")
        self.assertEqual(len(goal["grasp_candidates"]), 2)
        yaws = sorted(c["closing_yaw_deg"] for c in goal["grasp_candidates"])
        self.assertAlmostEqual(yaws[1] - yaws[0], 180.0, delta=0.2)
        self.assertAlmostEqual(goal["grasp_candidates"][0]["closing_yaw_deg"] % 180, 120.0, delta=0.2)  # 볼트 30° + 90°
        self.assertEqual(goal["place"]["bin"], "BOLT_B")
        self.assertEqual(goal["place"]["position_mm"], [320.0, 450.0, 180.0])
        self.assertEqual(goal["gripper_opening_mm"], 20.0)
        self.assertEqual(goal["capture_stamp"], {"base": "SIM_TIME", "ns": 94_264_000_000})

    def test_nut_pass_through(self):
        r_hex = 0.010 / math.sqrt(3)
        xy = np.array([[0.58 + r_hex * math.cos(math.radians(30 + 60 * k)),
                        0.0 + r_hex * math.sin(math.radians(30 + 60 * k))] for k in range(6)])
        r = from_observation(make_obs(to_px(xy).astype(np.float32), cls="nut"), make_frame(), CAM, T_W_C, SPECS)
        goal = build_sort_goal(r, source_id="link_cam", stamp_ns=1, frames=FRAMES, specs=SPECS)
        self.assertEqual((r.spec, goal["kind"]), ("NUT", "PASS_THROUGH"))
        self.assertNotIn("grasp_candidates", goal)
        self.assertIsNone(to_output_dict(r, "link_cam")["theta_deg"])

    def test_invalid_observation_skips(self):
        r = from_observation(make_obs(None, valid=False), make_frame(), CAM, T_W_C, SPECS)
        self.assertEqual((r.valid, r.reason, r.reason_owner), (False, "NO_DETECTION", "A"))
        goal = build_sort_goal(r, source_id="link_cam", stamp_ns=1, frames=FRAMES, specs=SPECS)
        self.assertEqual((goal["kind"], goal["reason"]), ("SKIP", "NO_DETECTION"))

    def test_calibration_mismatch_skips(self):
        xy = bolt(0.018, 0.55, 0, 0)
        r = from_observation(make_obs(to_px(xy).astype(np.float32), cal="cal-old"), make_frame(), CAM, T_W_C, SPECS)
        self.assertEqual((r.reason, r.reason_owner), ("CALIBRATION_MISMATCH", "A"))
        self.assertEqual(build_sort_goal(r, source_id="link_cam", stamp_ns=1, frames=FRAMES, specs=SPECS)["kind"], "SKIP")

    def test_frame_mismatch_raises(self):
        xy = bolt(0.018, 0.55, 0, 0)
        with self.assertRaises(ValueError):
            from_observation(make_obs(to_px(xy).astype(np.float32), seq=8), make_frame(seq=7), CAM, T_W_C, SPECS)

    def test_down_quaternion(self):
        R = quat_to_rot(*down_quat_xyzw(math.radians(40)))
        np.testing.assert_allclose(R[:, 2], [0, 0, -1], atol=1e-6)
        np.testing.assert_allclose(R[:, 0], [math.cos(math.radians(40)), math.sin(math.radians(40)), 0], atol=1e-6)


# ---------- B 입구 (A 파이프라인이 부르는 함수) ----------
from types import SimpleNamespace as NS  # noqa: E402

from eyetoarm.transforms.b_stage import BStage, StaticTfLookup, goal_to_json, transform_to_matrix  # noqa: E402


class TestBStage(unittest.TestCase):
    def setUp(self):
        self.stage = BStage.from_config(ROOT, CAM, StaticTfLookup(T_W_C))

    def test_process_bolt(self):
        obs = make_obs(to_px(bolt(0.018, 0.55, -0.05, 60)).astype(np.float32))
        out = self.stage.process(obs, make_frame())
        self.assertEqual((out.result.spec, out.goal["kind"]), ("BOLT_A", "PICK_PLACE"))
        self.assertEqual(out.goal["place"]["bin"], "BOLT_A")
        self.assertEqual(out.result_dict["spec"], "BOLT_A")
        self.assertIn('"kind":"PICK_PLACE"', goal_to_json(out.goal))

    def test_tf_missing_skips(self):
        stage = BStage.from_config(ROOT, CAM, lambda frame: None)
        out = stage.process(make_obs(to_px(bolt(0.018, 0.55, 0, 0)).astype(np.float32)), make_frame())
        self.assertEqual((out.result.reason, out.result.reason_owner, out.goal["kind"]), ("TF_UNAVAILABLE", "B", "SKIP"))

    def test_render_center_keeps_original(self):
        frame = make_frame()
        out = self.stage.process(make_obs(to_px(bolt(0.046, 0.55, -0.05, 20)).astype(np.float32)), frame)
        img = self.stage.render_center(frame.image, [out])
        self.assertEqual(int(frame.image.sum()), 0)        # 원본은 그대로
        self.assertGreater(int(img.sum()), 0)              # 복사본에 그려짐

    def test_transform_to_matrix(self):
        tr = NS(translation=NS(x=0.06, y=0.14, z=0.505), rotation=NS(x=0.0, y=0.0, z=0.0, w=1.0))
        np.testing.assert_allclose(transform_to_matrix(tr)[:3, 3], [0.06, 0.14, 0.505])


class TestBStageKeys(unittest.TestCase):
    def test_correlation_keys(self):
        stage = BStage.from_config(ROOT, CAM, StaticTfLookup(T_W_C))
        stage.sort_job_id = "job-1"
        out = stage.process(make_obs(to_px(bolt(0.046, 0.55, 0, 0)).astype(np.float32)), make_frame())
        g = out.goal
        self.assertEqual((g["sort_job_id"], g["run_id"], g["job_seq"], g["timestamp_base"]), ("job-1", "r1", 1, "SIM_TIME"))
        self.assertEqual(len(g["goal_id"]), 36)
        self.assertNotEqual(g["goal_id"], stage.process(make_obs(to_px(bolt(0.046, 0.55, 0, 0)).astype(np.float32)), make_frame()).goal["goal_id"])
