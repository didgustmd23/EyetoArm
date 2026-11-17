"""B 보정·위치 변환·판정·주석 영상 단위시험.

근거: B 상세설계 §6(렌즈 보정)·§6.2(원본 픽셀 → W, 1회 보정)·§7(높이별 Homography·중심·각도)·§15(규격).

- 기대값은 정답 자세·치수에서 손으로 정한 값이다. 윤곽은 정답 W 점을 카메라로 투영해 만든다
  (래스터화 없음). 그래서 B 계산은 수치 오차 수준(1e-6 m)으로 정답과 같아야 한다.
- 설정 파일은 저장소의 config/robot/frames.yaml, config/calibration/part_specs.json을 쓴다.
"""
import dataclasses
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
        self.assertEqual(goal["place"]["position_mm"], [720.0, 170.0, 180.0])
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


class TestAConnection(unittest.TestCase):
    def test_consume_matches_runner_order(self):
        stage = BStage.from_config(ROOT, CAM, StaticTfLookup(T_W_C))
        got = []
        stage.on_output = got.append
        frame = make_frame()
        out = stage.consume(frame, make_obs(to_px(bolt(0.046, 0.55, 0, 30)).astype(np.float32)))
        self.assertIs(stage.last, out)
        self.assertEqual([o.goal["kind"] for o in got], ["PICK_PLACE"])

    def test_runner_signature(self):
        import inspect
        from eyetoarm.vision.pipeline_runner import run
        self.assertIn("b_consumer", inspect.signature(run).parameters)

    def test_yolo_class_names(self):
        from eyetoarm.transforms.work_result import normalize_class
        for name, want in [("bolt", "bolt"), ("Bolt", "bolt"), ("BOLT_A", "bolt"), ("BOLT_B", "bolt"),
                           ("nut", "nut"), ("NUT", "nut"), (None, None), ("washer", None)]:
            self.assertEqual(normalize_class(name), want, name)
        r = compute_work_result(to_px(bolt(0.046, 0.55, 0, 10)), CAM, T_W_C, SPECS, class_name="BOLT_A")
        self.assertEqual(r.spec, "BOLT_B")    # A 이름이 BOLT_A여도 길이로 다시 판정


# ---------- D 번호표·픽셀 중심, C 대기열 넘기기 ----------
class TestHandoffAndKeys(unittest.TestCase):
    def setUp(self):
        self.stage = BStage.from_config(ROOT, CAM, StaticTfLookup(T_W_C))

    def test_result_dict_keys_and_center_px(self):
        px = to_px(bolt(0.046, 0.55, -0.05, 30)).astype(np.float32)
        out = self.stage.process(make_obs(px), make_frame())
        d = out.result_dict
        for k in ("run_id", "capture_generation", "generation", "job_seq", "detector_config_id",
                  "timestamp_ms", "timestamp_base", "capture_stamp_ns"):
            self.assertEqual(d[k], out.goal[k], k)
        self.assertEqual((d["run_id"], d["detector_config_id"], d["capture_stamp_ns"]), ("r1", "det", 94_264_000_000))
        want = to_px(np.array([out.result.center_w_m]))[0]     # B 중심을 다시 사진에 찍은 자리
        np.testing.assert_allclose(d["center_px"], want, atol=0.06)
        self.assertLess(abs(d["center_px"][0] - 578.4) + abs(d["center_px"][1] - 371.2), 15)   # 대략 그 볼트 위

    def test_center_px_none_without_tf(self):
        stage = BStage.from_config(ROOT, CAM, lambda f: None)
        out = stage.process(make_obs(to_px(bolt(0.046, 0.55, 0, 0)).astype(np.float32)), make_frame())
        self.assertIsNone(out.result_dict["center_px"])

    def test_make_c_handoff_threadsafe_and_drop_oldest(self):
        import asyncio
        import threading
        from eyetoarm.transforms.b_stage import make_c_handoff

        loop = asyncio.new_event_loop()
        queue = loop.run_until_complete(self._make_queue(2))
        handoff = make_c_handoff(loop, queue)
        self.stage.on_output = handoff

        bolt_px = to_px(bolt(0.046, 0.55, 0, 0)).astype(np.float32)

        def worker():   # A 검출 스레드 흉내
            for seq in range(1, 4):
                self.stage.consume(make_frame(seq), make_obs(bolt_px, seq=seq))
            self.stage.consume(make_frame(9), make_obs(None, valid=False, seq=9))   # SKIP → 안 넣음

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        loop.run_until_complete(asyncio.sleep(0.01))
        got = [queue.get_nowait()["image_seq"] for _ in range(queue.qsize())]
        loop.close()
        self.assertEqual(got, [2, 3])                       # 크기 2 → 가장 오래된 1번을 버림
        self.assertEqual(handoff.stats, {"sent": 3, "skipped": 1, "dropped_old": 1})

    @staticmethod
    async def _make_queue(n):
        import asyncio
        return asyncio.Queue(maxsize=n)


# ---------- B 결정 2026-11-17: 집는 구간·높이·같은 부품 한 번만 ----------
class TestBDecisions(unittest.TestCase):
    def setUp(self):
        self.stage = BStage.from_config(ROOT, CAM, StaticTfLookup(T_W_C))

    def test_heights_in_goal(self):
        out = self.stage.process(make_obs(to_px(bolt(0.046, 0.55, -0.05, 20)).astype(np.float32)), make_frame())
        self.assertEqual((out.goal["kind"], out.goal["approach_height_mm"], out.goal["lift_height_mm"]),
                         ("PICK_PLACE", 205.0, 250.0))

    def test_out_of_reach_skips(self):
        from eyetoarm.transforms.grasp import build_sort_goal
        r = compute_work_result(to_px(bolt(0.046, 0.55, -0.05, 20)), CAM, T_W_C, SPECS, class_name="bolt")
        r = dataclasses.replace(r, center_w_m=(0.55, 0.50))          # 집는 구간(+0.45) 밖
        g = build_sort_goal(r, source_id="link_cam", stamp_ns=1, frames=self.stage.frames, specs=SPECS)
        self.assertEqual((g["kind"], g["reason"]), ("SKIP", "OUT_OF_REACH"))

    def test_same_track_sent_once_until_release(self):
        g1 = {"kind": "PICK_PLACE", "part_track_id": 7, "goal_id": "a"}
        g2 = {"kind": "PICK_PLACE", "part_track_id": 7, "goal_id": "b"}
        self.stage._dedupe(g1)
        self.stage._dedupe(g2)
        self.assertEqual((g1["kind"], g2["kind"], g2["reason"]), ("PICK_PLACE", "SKIP", "ALREADY_SENT"))
        self.stage.release(7)
        g3 = {"kind": "PICK_PLACE", "part_track_id": 7, "goal_id": "c"}
        self.stage._dedupe(g3)
        self.assertEqual(g3["kind"], "PICK_PLACE")

    def test_new_epoch_clears_memory(self):
        self.stage._dedupe({"kind": "PICK_PLACE", "part_track_id": 9, "goal_id": "a"})
        self.stage.run_epoch = 2
        g = {"kind": "PICK_PLACE", "part_track_id": 9, "goal_id": "b"}
        self.stage._dedupe(g)
        self.assertEqual(g["kind"], "PICK_PLACE")

    def test_no_track_id_not_filtered(self):
        for gid in ("a", "b"):
            g = {"kind": "PICK_PLACE", "part_track_id": None, "goal_id": gid}
            self.stage._dedupe(g)
            self.assertEqual(g["kind"], "PICK_PLACE")


# ---------- B 정확도 확인 (시뮬레이션 정답과 비교) ----------
class TestAccuracy(unittest.TestCase):
    def test_load_truth_from_world(self):
        from eyetoarm.transforms.accuracy import load_truth
        t = {p.name: p for p in load_truth(ROOT / "simulation/worlds/sort_cell.sdf")}
        self.assertEqual({k: v.spec for k, v in t.items()}, {"part_00": "BOLT_A", "part_01": "BOLT_B", "part_02": "NUT"})
        np.testing.assert_allclose(t["part_01"].xy_m, (0.55 - 0.0154, -0.15), atol=1e-4)   # 머리 아래 원점 → 면적 중심
        self.assertEqual(t["part_01"].theta_deg, 0.0)
        self.assertIsNone(t["part_02"].theta_deg)

    def test_match_and_summary(self):
        from eyetoarm.transforms.accuracy import TruthPart, angle_error_deg, format_table, match, summarize
        truth = [TruthPart("p0", "BOLT_B", (0.55, -0.15), 0.0)]
        r = compute_work_result(to_px(bolt(0.046, 0.55, -0.15, 2.0)), CAM, T_W_C, SPECS)
        far = dataclasses.replace(r, center_w_m=(0.10, 0.40))
        rows = match([r, far], truth)
        self.assertEqual([x["truth"] for x in rows], ["p0", None])
        self.assertLess(rows[0]["dist_mm"], 25.0)
        s = summarize(rows, truth)[0]
        self.assertEqual((s["seen"], s["spec_ok_pct"]), (1, 100.0))
        self.assertIn("| p0 | BOLT_B | 1 | 100 |", format_table([s]))
        self.assertAlmostEqual(angle_error_deg(179.0, 1.0), 2.0)


# ---------- B 선택 2026-11-17 (A-2): 볼트 두 끝점, 종류별 높이, 집는 점 Pb ----------
class TestEndpointInput(unittest.TestCase):
    def _endpoints(self, length, cx, cy, deg, z=0.105):
        """머리 끝·축 끝 world → 픽셀 (Pa = (cx, cy))."""
        u = np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])
        head = np.array([cx, cy]) - u * length / 2
        tip = np.array([cx, cy]) + u * length / 2
        px = w_to_image(np.array([[*head, z], [*tip, z]]), CAM.K, CAM.dist, T_W_C)
        return px[0], px[1], u

    def test_bolt_b_from_endpoints(self):
        from eyetoarm.transforms.work_result import compute_bolt_from_endpoints
        h, t, u = self._endpoints(0.046, 0.55, -0.10, 30.0)
        r = compute_bolt_from_endpoints(h, t, CAM, T_W_C, SPECS)
        self.assertEqual((r.spec, r.method, r.plane_z_m), ("BOLT_B", "axis_endpoints", 0.105))
        self.assertAlmostEqual(r.measured_length_m, 0.046, places=4)
        self.assertAlmostEqual(r.theta_deg, 30.0, places=2)
        np.testing.assert_allclose(r.center_w_m, (0.55, -0.10), atol=1e-4)                 # Pa
        np.testing.assert_allclose(r.grasp_w_m, np.array([0.55, -0.10]) + u * 0.003, atol=1e-4)   # Pb = 축 끝 쪽 3 mm

    def test_bolt_a_from_endpoints_and_goal_uses_pb(self):
        from eyetoarm.transforms.grasp import build_sort_goal
        from eyetoarm.transforms.work_result import compute_bolt_from_endpoints
        h, t, u = self._endpoints(0.018, 0.55, 0.05, 120.0)
        r = compute_bolt_from_endpoints(h, t, CAM, T_W_C, SPECS)
        self.assertEqual(r.spec, "BOLT_A")
        frames = BStage.from_config(ROOT, CAM, StaticTfLookup(T_W_C)).frames
        g = build_sort_goal(r, source_id="link_cam", stamp_ns=1, frames=frames, specs=SPECS)
        pb = (np.array([0.55, 0.05]) + u * 0.003) * 1000
        np.testing.assert_allclose(g["grasp_candidates"][0]["position_mm"][:2], pb, atol=0.15)
        np.testing.assert_allclose(g["observed_position_mm"][:2], [550.0, 50.0], atol=0.15)   # 위치 보고는 Pa

    def test_nut_uses_top_face_height(self):
        from eyetoarm.transforms.work_result import class_plane_z
        self.assertEqual((class_plane_z(SPECS, "nut"), class_plane_z(SPECS, "bolt"), class_plane_z(SPECS, None)),
                         (0.1057, 0.105, 0.105))
        a = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        hexagon = np.column_stack([0.55 + 0.0058 * np.cos(a), 0.0058 * np.sin(a)])
        r = compute_work_result(to_px(hexagon, z=0.1057), CAM, T_W_C, SPECS, class_name="nut")
        self.assertEqual((r.plane_z_m, r.spec), (0.1057, "NUT"))

    def test_from_observation_endpoints(self):
        h, t, _ = self._endpoints(0.046, 0.55, -0.05, 0.0)
        base = make_obs(None, valid=False)        # 공통형은 아직 끝점 칸이 없어 시험용 대역으로 만든다
        obs = NS(source_id=base.source_id, image_seq=base.image_seq, valid=True, class_name="bolt",
                 calc_contour=None, calibration_id=None, pixel_basis=base.pixel_basis, axis_endpoints_px=(h, t))
        r = from_observation(obs, make_frame(), CAM, T_W_C, SPECS)
        self.assertEqual((r.spec, r.method), ("BOLT_B", "axis_endpoints"))

    def test_compute_budget_skips(self):
        stage = BStage.from_config(ROOT, CAM, StaticTfLookup(T_W_C))
        stage.compute_budget_ms = -1.0                       # 항상 넘게 해서 확인
        out = stage.process(make_obs(to_px(bolt(0.046, 0.55, -0.05, 20)).astype(np.float32)), make_frame())
        self.assertEqual((out.goal["kind"], out.goal["reason"]), ("SKIP", "B_TIMEOUT"))
