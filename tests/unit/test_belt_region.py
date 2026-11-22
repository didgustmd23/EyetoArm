"""벨트 밖 검출 제외 시험: 광선–윗면 교점 판정, 영상 다각형 판정과의 동치, 경계, 거부 표시, 감싸개, 설정."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from eyetoarm.transforms.planar_transform import look_at, undistort_pixels, w_to_image
from eyetoarm.vision.ai_result import ARM_REGION, OUTSIDE_BELT, AiInstance
from eyetoarm.vision.arm_region import ArmRegionFilter
from eyetoarm.vision.arm_region import filter_result as arm_filter_result
from eyetoarm.vision.arm_region import parse_arm_region
from eyetoarm.vision.belt_region import (BeltRegionConfigError, BeltRegionFilter, belt_from_frames, filter_result,
                                         inside_fraction, load_belt_region_config, parse_belt_region, rays_to_plane)
from eyetoarm.vision.live_observation_envelope import TfLookupResult
from eyetoarm.vision.observation_envelope import CaptureTf
from eyetoarm.vision.world_tracking import world_inputs
from test_ai_result import seg_result

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "vision" / "belt_region_filter.json"
FRAMES = ROOT / "config" / "robot" / "frames.yaml"
ARM_CONFIG = ROOT / "config" / "vision" / "arm_region_filter.json"
K = np.array([[1171.5, 0.0, 640.0], [0.0, 1171.5, 360.0], [0.0, 0.0, 1.0]])   # #287 링크 카메라 camera_info와 비슷한 값
DIST = np.zeros(5)
# #287 gt/tf.jsonl 링크 카메라 위치(약 (0.06, 0.14, 0.50))에서 벨트 쪽을 내려다보는 자세
CAM_T = look_at((0.06, 0.14, 0.50), (0.55, 0.25, 0.10))
TOP_Z = 0.100   # frames.yaml belt.top_z
RECT = (0.45, 0.65, -0.63, 0.63)   # frames.yaml belt: center (0.55, 0)·width 0.20(X)·length 1.26(Y)


def frames_doc():
    """저장소 frames.yaml(읽기만)."""
    return yaml.safe_load(FRAMES.read_text(encoding="utf-8"))


def cfg(**over):
    """저장소 설정 파일 값 + 덮어쓰기."""
    doc = json.loads(CONFIG.read_text(encoding="utf-8"))
    doc.update(over)

    return parse_belt_region(doc, frames_doc())


def square_at_w(x, y, z, half_m=0.004, cam=CAM_T):
    """W 점 주변 수평 정사각형을 링크 영상에 투영한 다각형(픽셀)."""
    pts = np.array([[x - half_m, y - half_m, z], [x + half_m, y - half_m, z], [x + half_m, y + half_m, z],
                    [x - half_m, y + half_m, z]])

    return w_to_image(pts, K, DIST, cam).astype(np.float32)


def bolt(index, polygon, overlap=None, rejected=None):
    """축 경로를 거친 볼트 인스턴스."""
    x0, y0 = polygon.min(axis=0)
    x1, y1 = polygon.max(axis=0)
    ep = None if rejected else np.array([[x0 + 1.0, y0 + 1.0], [x1 - 1.0, y1 - 1.0]])

    return AiInstance(index=index, class_index=0, class_name="bolt", confidence=0.9,
                      box_xyxy=(float(x0), float(y0), float(x1), float(y1)), polygon=polygon,
                      area_px=float((x1 - x0) * (y1 - y0)), component_count=1, hole_count=0, clipped=False,
                      rejected=rejected, axis_endpoints_px=ep, axis_reason=None, axis_overlap_with=overlap)


def ray_hits_top_inside(cam, p):
    """카메라 → W 점 p 선분이 벨트 윗면(사각형, z = TOP_Z)을 지나는지(지나면 p는 벨트에 가려 보이지 않는다)."""
    c = cam[:3, 3]
    s = (c[2] - TOP_Z) / (c[2] - p[2])

    if not 0.0 < s < 1.0:
        return False

    x, y = c[:2] + s * (np.asarray(p[:2]) - c[:2])

    return RECT[0] <= x <= RECT[1] and RECT[2] <= y <= RECT[3]


class ConfigTest(unittest.TestCase):
    """설정 읽기·검사."""

    def test_repo_config_and_frames(self):
        """저장소 설정: 사각형·top_z는 frames.yaml 값, 여유 10 mm·비율 0.5, sha256 두 개."""
        c, sha, frames_sha = load_belt_region_config(str(CONFIG), str(FRAMES))
        np.testing.assert_allclose(c.rect_xy_m, RECT, atol=1e-12)
        self.assertEqual((c.top_z_m, c.margin_mm, c.min_inside_fraction, c.world_frame), (TOP_Z, 10.0, 0.5, "world"))
        self.assertEqual((len(sha), len(frames_sha)), (64, 64))

    def test_bad_values_rejected(self):
        """키 누락·추가, status, 범위 밖·bool·비유한 값, 표본 수, world_frame은 오류."""
        base = json.loads(CONFIG.read_text(encoding="utf-8"))
        bad = [{k: v for k, v in base.items() if k != "margin_mm"}, {**base, "extra": 1},
               {**base, "status": "production"}, {**base, "margin_mm": -1.0}, {**base, "margin_mm": True},
               {**base, "margin_mm": float("nan")}, {**base, "min_inside_fraction": 0.0},
               {**base, "min_inside_fraction": 1.5}, {**base, "max_samples": 2}, {**base, "max_samples": 8.0},
               {**base, "world_frame": ""}, []]

        for doc in bad:
            with self.subTest(doc=doc), self.assertRaises(BeltRegionConfigError):
                parse_belt_region(doc, frames_doc())

    def test_bad_frames_rejected(self):
        """frames belt 블록이 없거나 top_z가 수가 아니면 오류."""
        good = json.loads(CONFIG.read_text(encoding="utf-8"))
        f = frames_doc()

        for frames in ({}, {**f, "belt": {**f["belt"], "top_z": "x"}}, {**f, "belt": {**f["belt"], "top_z": True}},
                       {**f, "belt": {**f["belt"], "direction": [0.0, 0.0]}}):
            with self.subTest(frames=frames.get("belt")), self.assertRaises(BeltRegionConfigError):
                parse_belt_region(good, frames)

        self.assertEqual(belt_from_frames(f)[1], TOP_Z)


class RayTest(unittest.TestCase):
    """광선–평면 교점."""

    def test_round_trip_and_behind(self):
        """W 점을 투영한 픽셀의 교점은 원래 점, 카메라 뒤에서만 만나는 광선(평면보다 아래 카메라)은 NaN."""
        pts = np.array([[0.5, 0.1, TOP_Z], [0.62, -0.3, TOP_Z], [0.3, 0.4, TOP_Z]])
        px = w_to_image(pts, K, DIST, CAM_T)
        np.testing.assert_allclose(rays_to_plane(px, K, DIST, CAM_T, TOP_Z), pts[:, :2], atol=1e-9)

        low = look_at((0.0, 0.0, 0.05), (1.0, 0.0, 0.05))   # 수평으로 보는 카메라, 평면 z = 0.1 위쪽 광선만 만난다
        out = rays_to_plane(np.array([[640.0, 600.0], [640.0, 100.0]]), K, DIST, low, TOP_Z)
        self.assertTrue(np.isnan(out[0]).all())   # 아래로 향한 광선
        self.assertTrue(np.isfinite(out[1]).all())

    def test_distortion_undone_once(self):
        """왜곡이 있으면 왜곡 픽셀에서도 같은 교점(OpenCV 투영 → 교점이 원래 점)."""
        dist = np.array([-0.12, 0.03, 0.001, -0.001, 0.0])
        pts = np.array([[0.5, 0.1, TOP_Z], [0.62, -0.3, TOP_Z]])
        px = w_to_image(pts, K, dist, CAM_T)
        np.testing.assert_allclose(rays_to_plane(px, K, dist, CAM_T, TOP_Z), pts[:, :2], atol=1e-5)   # 반복 보정 오차 10 µm 안


class SameAsImagePolygonTest(unittest.TestCase):
    """W(top_z) 사각형 판정 = 영상에 투영한 벨트 윗면 다각형 판정(대조용, 계획 §1)."""

    def test_random_pixels_agree(self):
        """임의 픽셀 2000개(왜곡 0·있음): 「교점이 사각형 안」과 「(왜곡을 뺀) 픽셀이 영상 다각형 안」이 같다."""
        import cv2

        rng = np.random.default_rng(288)
        corners = np.array([[RECT[0], RECT[2], TOP_Z], [RECT[1], RECT[2], TOP_Z], [RECT[1], RECT[3], TOP_Z],
                            [RECT[0], RECT[3], TOP_Z]])

        for dist in (DIST, np.array([-0.12, 0.03, 0.001, -0.001, 0.0])):
            # 왜곡을 뺀 픽셀 공간에서는 평면 사각형이 직선 사각형이다(꼭짓점은 왜곡 없이 투영)
            poly = w_to_image(corners, K, np.zeros(5), CAM_T).astype(np.float32)
            px = np.column_stack([rng.uniform(0, 1280, 2000), rng.uniform(0, 720, 2000)])
            w = rays_to_plane(px, K, dist, CAM_T, TOP_Z)
            by_w = (w[:, 0] >= RECT[0]) & (w[:, 0] <= RECT[1]) & (w[:, 1] >= RECT[2]) & (w[:, 1] <= RECT[3])
            und = undistort_pixels(px, K, dist)
            by_img = np.array([cv2.pointPolygonTest(poly, (float(u), float(v)), True) for u, v in und])
            clear = np.abs(by_img) > 1e-3   # 경계선 위(부동소수 오차) 표본은 빼고 비교

            with self.subTest(distorted=bool(np.any(dist))):
                np.testing.assert_array_equal(by_w[clear], by_img[clear] > 0)
                self.assertGreater(int(by_w.sum()), 100)      # 안·밖이 모두 충분히 섞였다
                self.assertGreater(int((~by_w).sum()), 100)


class FractionTest(unittest.TestCase):
    """판정: 벨트 위 유지, 바닥 제외, 가장자리 걸침, 높이 밀림."""

    def test_on_belt_kept_floor_excluded(self):
        """벨트 가운데 볼트(z 0.105)는 비율 1, 로봇 쪽·벨트 끝 너머 바닥(z 0.0025)은 0."""
        c = cfg()
        on, n, center = inside_fraction(square_at_w(0.55, 0.20, 0.105), K, DIST, CAM_T, c)
        self.assertEqual(on, 1.0)
        self.assertGreaterEqual(n, 3)
        # 볼트 평면(0.105)의 점을 top_z(0.100)에서 자르면 카메라에서 먼 쪽으로 0.005·(0.49/0.40) ≈ 6 mm 밀린다
        self.assertAlmostEqual(center[0], 556.1, delta=1.0)

        for x, y in ((0.40, 0.38), (0.35, 0.34), (0.42, 0.40), (0.55, 0.85)):   # #287 바닥 볼트 정답 위치 근처·벨트 끝 너머
            with self.subTest(x=x, y=y):
                self.assertFalse(ray_hits_top_inside(CAM_T, (x, y, 0.0025)))   # 실제로 보이는 바닥 점
                frac, _, _ = inside_fraction(square_at_w(x, y, 0.0025), K, DIST, CAM_T, c)
                self.assertEqual(frac, 0.0)

    def test_visible_floor_never_inside_without_margin(self):
        """보이는(벨트 윗면에 가려지지 않는) 바닥 점은 여유 0이면 언제나 밖이다 — 가림 근거(계획 §1)."""
        rng = np.random.default_rng(1)
        c = cfg(margin_mm=0.0)
        seen = 0

        for _ in range(400):
            p = (rng.uniform(0.2, 1.0), rng.uniform(-0.8, 0.8), 0.0025)

            if ray_hits_top_inside(CAM_T, p):
                continue

            px = w_to_image(np.array([p]), K, DIST, CAM_T)

            if not (0 <= px[0, 0] < 1280 and 0 <= px[0, 1] < 720):
                continue

            seen += 1
            w = rays_to_plane(px, K, DIST, CAM_T, c.top_z_m)[0]
            inside = RECT[0] <= w[0] <= RECT[1] and RECT[2] <= w[1] <= RECT[3]
            self.assertFalse(inside, p)

        self.assertGreater(seen, 50)

    def test_edge_straddle_threshold(self):
        """가장자리 걸침: 안쪽 비율이 문턱 위면 유지, 아래면 제외(여유 0으로 경계를 정확히 둔다)."""
        c = cfg(margin_mm=0.0)
        mostly_in = square_at_w(0.452, 0.0, TOP_Z, half_m=0.008)   # x 0.444~0.460: 약 62% 안
        mostly_out = square_at_w(0.448, 0.0, TOP_Z, half_m=0.008)  # x 0.440~0.456: 약 38% 안
        f_in, _, _ = inside_fraction(mostly_in, K, DIST, CAM_T, c)
        f_out, _, _ = inside_fraction(mostly_out, K, DIST, CAM_T, c)
        self.assertGreater(f_in, 0.5)
        self.assertLess(f_out, 0.5)
        self.assertGreater(f_out, 0.0)

        res = seg_result([bolt(0, mostly_in), bolt(1, mostly_out)])
        out, decisions = filter_result(res, K=K, dist=DIST, T_W_C=CAM_T, config=c)
        self.assertEqual([i.rejected for i in out.instances], [None, OUTSIDE_BELT])
        self.assertEqual([d["excluded"] for d in decisions], [False, True])

    def test_far_edge_raised_top_kept_by_margin(self):
        """먼 쪽 가장자리(x 0.645) 볼트 윗면(z 0.106)은 top_z 투영이 밖으로 밀리지만 여유 10 mm로 남는다."""
        poly = square_at_w(0.645, 0.25, TOP_Z + 0.006, half_m=0.003)
        f0, _, _ = inside_fraction(poly, K, DIST, CAM_T, cfg(margin_mm=0.0))
        f10, _, _ = inside_fraction(poly, K, DIST, CAM_T, cfg())
        self.assertLess(f0, 0.5)
        self.assertEqual(f10, 1.0)

    def test_lifted_above_belt_depends_on_geometry(self):
        """벨트 위 공중에 들린 볼트는 이 필터가 가르지 않는다(팔 영역 필터 몫, 계획 보완 a).

        top_z 교점이 카메라에서 먼 쪽으로 밀려, 벨트 안이면 통과(0.47, z 0.15), 먼 쪽 가장자리를 넘으면 제외(0.55, z 0.20).
        """
        near, _, _ = inside_fraction(square_at_w(0.47, 0.2, 0.15), K, DIST, CAM_T, cfg())
        far, _, _ = inside_fraction(square_at_w(0.55, 0.2, 0.20), K, DIST, CAM_T, cfg())
        self.assertEqual((near, far), (1.0, 0.0))

    def test_no_plane_hit_counts_outside(self):
        """평면과 만나지 않는 표본은 밖으로 센다(중심 None)."""
        up = look_at((0.0, 0.0, 0.05), (1.0, 0.0, 0.05))
        poly = np.array([[600, 600], [680, 600], [680, 680], [600, 680]], dtype=np.float32)
        frac, n, center = inside_fraction(poly, K, DIST, up, cfg())
        self.assertEqual((frac, center), (0.0, None))
        self.assertGreaterEqual(n, 3)


class FilterTest(unittest.TestCase):
    """거부 표시와 소비자 쪽 제외."""

    def test_reject_marks_out_of_scope_and_clears_overlap(self):
        """바닥 인스턴스는 OUTSIDE_BELT·범위 밖·축 필드 없음, 겹침 짝이 풀려 벨트 위 볼트만 축 유효."""
        floor = bolt(1, square_at_w(0.40, 0.38, 0.0025), overlap=(0,))
        real = bolt(0, square_at_w(0.55, 0.20, 0.105), overlap=(1,))
        res = seg_result([real, floor])
        self.assertEqual(res.axis_valid_indices, ())
        out, decisions = filter_result(res, K=K, dist=DIST, T_W_C=CAM_T, config=cfg())
        f = out.instances[1]
        self.assertEqual((f.rejected, f.in_scope, f.axis_evaluated, f.axis_endpoints_px), (OUTSIDE_BELT, False, False,
                                                                                            None))
        self.assertIsNone(out.instances[0].axis_overlap_with)
        self.assertEqual((out.axis_valid_indices, out.candidate_count), ((0,), 1))
        self.assertEqual([d["excluded"] for d in decisions], [False, True])
        self.assertIsNotNone(decisions[1]["center_w_top_mm"])

    def test_world_inputs_skip_outside_belt(self):
        """W 추적 입력에서 OUTSIDE_BELT는 not_in_scope."""
        out, _ = filter_result(seg_result([bolt(0, square_at_w(0.40, 0.38, 0.0025))]), K=K, dist=DIST, T_W_C=CAM_T,
                               config=cfg())
        w = world_inputs(out, K=K, dist=DIST, T_W_C=CAM_T, plane_z_m=0.105)
        self.assertEqual((w[0].candidate, w[0].untracked_reason), (False, "not_in_scope"))

    def test_no_change_returns_same_object_and_skips_out_of_scope(self):
        """제외가 없으면 같은 결과 객체. 이미 ARM_REGION인 인스턴스는 판정하지 않는다."""
        res = seg_result([bolt(0, square_at_w(0.55, 0.2, 0.105)),
                          bolt(1, square_at_w(0.40, 0.38, 0.0025), rejected=ARM_REGION)])
        out, decisions = filter_result(res, K=K, dist=DIST, T_W_C=CAM_T, config=cfg())
        self.assertIs(out, res)
        self.assertEqual([d["index"] for d in decisions], [0])

    def test_arm_filter_unchanged_by_shared_helper(self):
        """팔 필터 거부 결과는 공통 함수 분리 뒤에도 ARM_REGION·축 비움 그대로(겹침 정리 포함)."""
        doc = json.loads(ARM_CONFIG.read_text(encoding="utf-8"))
        doc["occluders"] = [{"link": "grip", "p0_m": [0, 0, 0], "p1_m": [0, 0, 0.001], "radius_m": 0.05}]
        mid = (CAM_T[:3, 3] + np.array([0.55, 0.2, 0.105])) / 2.0
        T = np.eye(4)
        T[:3, 3] = mid
        res = seg_result([bolt(0, square_at_w(0.55, 0.2, 0.105), overlap=(1,)),
                          bolt(1, square_at_w(0.55, 0.45, 0.105), overlap=(0,))])
        out, _ = arm_filter_result(res, K=K, dist=DIST, T_W_C=CAM_T, T_W_L_by_link={"grip": T},
                                   config=parse_arm_region(doc))
        self.assertEqual([i.rejected for i in out.instances], [ARM_REGION, None])
        self.assertIsNone(out.instances[1].axis_overlap_with)


class FakeTf:
    """카메라 TF 대역. `fail`이면 조회 실패, 조회 횟수를 센다."""

    def __init__(self, fail=False):
        """실패 여부."""
        self.fail = fail
        self.calls = 0

    def __call__(self, frame):
        """촬영 stamp 조회 대역."""
        self.calls += 1

        if self.fail:
            return TfLookupResult(None, "unavailable", 0.1, "fake")

        return TfLookupResult(CaptureTf("world", frame.optical_frame, frame.ros_stamp_ns, CAM_T), "ok", 0.1)


def job():
    """작업 대역."""
    return SimpleNamespace(frame=SimpleNamespace(image_seq=7, ros_stamp_ns=1_000_000,
                                                 optical_frame="link_cam_optical_frame"))


class WrapperTest(unittest.TestCase):
    """감싸개: 관측 재선택·토큰 인자·TF·카메라 없음·오류·기록."""

    def make(self, tf=None, camera=True, inner_result=None):
        """감싼 검출기(벨트 위 볼트 0 + 바닥 볼트 1)."""
        res = inner_result or seg_result([bolt(0, square_at_w(0.55, 0.2, 0.105), overlap=(1,)),
                                          bolt(1, square_at_w(0.40, 0.38, 0.0025), overlap=(0,))])
        self.calls, self.records, self.selected = [], [], []

        def inner(j, *args):
            """감쌀 검출기 대역."""
            self.calls.append(args)
            return "obs-before", res

        def select(j, r):
            """관측 재선택 대역."""
            self.selected.append(r.axis_valid_indices)
            return "obs-after"

        cam = SimpleNamespace(K=K, dist=DIST) if camera else None
        self.tf = tf or FakeTf()

        return BeltRegionFilter(inner, tf_lookup=self.tf, camera_for=lambda frame: cam, config=cfg(), select=select,
                                on_decision=self.records.append)

    def test_reselects_and_passes_token(self):
        """바닥 제외 뒤 관측을 다시 고르고(벨트 위만 유효), 토큰 인자를 넘기며, 카메라 TF는 한 번만 조회한다."""
        f = self.make()
        obs, out = f(job(), "token")
        self.assertEqual((obs, self.selected, self.calls, self.tf.calls), ("obs-after", [(0,)], [("token",)], 1))
        self.assertEqual(out.instances[1].rejected, OUTSIDE_BELT)
        self.assertEqual(self.records[0]["status"], "ok")
        s = f.summary()
        self.assertEqual((s["counts"]["excluded"], s["counts"]["reselected"], s["counts"]["judged"]), (1, 1, 1))
        self.assertEqual(s["config"]["margin_mm"], 10.0)
        self.assertIsNotNone(s["tf_ms_median"])

    def test_tf_failure_keeps_result(self):
        """카메라 TF가 없으면 제외하지 않고 사유를 남긴다."""
        f = self.make(tf=FakeTf(fail=True))
        obs, out = f(job())
        self.assertEqual(obs, "obs-before")
        self.assertIsNone(out.instances[1].rejected)
        self.assertEqual(self.records[0]["status"], "tf_unavailable:camera")
        self.assertEqual(f.summary()["counts"]["tf_unavailable"], 1)

    def test_no_camera_keeps_result(self):
        """카메라 보정이 없으면 제외하지 않는다."""
        f = self.make(camera=False)
        obs, _ = f(job())
        self.assertEqual((obs, self.records[0]["status"]), ("obs-before", "no_camera"))

    def test_error_keeps_result(self):
        """판정 중 예외는 제외 없이 사유만 남기고 오류 수를 센다."""
        f = self.make()
        f._camera_for = lambda frame: SimpleNamespace(K="bad", dist=DIST)
        obs, out = f(job())
        self.assertEqual(obs, "obs-before")
        self.assertTrue(self.records[0]["status"].startswith("error:"))
        self.assertEqual(f.summary()["counts"]["errors"], 1)

    def test_no_candidates_no_record(self):
        """범위 안 후보가 없으면 TF를 조회하지 않고 기록도 없다."""
        f = self.make(inner_result=seg_result([bolt(0, square_at_w(0.4, 0.38, 0.0025), rejected=ARM_REGION)]))
        f(job())
        self.assertEqual((self.tf.calls, self.records), (0, []))

    def test_stacked_after_arm_filter(self):
        """팔 필터 바깥에 쌓으면 팔로 제외된 인스턴스는 벨트 판정에서 빠진다(판정 근거에 없음)."""
        doc = json.loads(ARM_CONFIG.read_text(encoding="utf-8"))
        doc["occluders"] = [{"link": "grip", "p0_m": [0, 0, 0], "p1_m": [0, 0, 0.001], "radius_m": 0.05}]
        mid = (CAM_T[:3, 3] + np.array([0.55, 0.2, 0.105])) / 2.0
        T = np.eye(4)
        T[:3, 3] = mid
        res = seg_result([bolt(0, square_at_w(0.55, 0.2, 0.105)), bolt(1, square_at_w(0.40, 0.38, 0.0025)),
                          bolt(2, square_at_w(0.55, 0.45, 0.105))])

        def tf(frame):
            """카메라·grip 링크."""
            pose = T if frame.optical_frame == "grip" else CAM_T
            return TfLookupResult(CaptureTf("world", frame.optical_frame, frame.ros_stamp_ns, pose), "ok", 0.1)

        cam = SimpleNamespace(K=K, dist=DIST)
        records = []
        arm = ArmRegionFilter(lambda j: ("obs", res), tf_lookup=tf, camera_for=lambda f: cam,
                              config=parse_arm_region(doc), select=lambda j, r: "arm-obs")
        belt = BeltRegionFilter(arm, tf_lookup=tf, camera_for=lambda f: cam, config=cfg(),
                                select=lambda j, r: r.axis_valid_indices, on_decision=records.append)
        obs, out = belt(job())
        self.assertEqual([i.rejected for i in out.instances], [ARM_REGION, OUTSIDE_BELT, None])
        self.assertEqual(obs, (2,))
        self.assertEqual([d["index"] for d in records[0]["decisions"]], [1, 2])


if __name__ == "__main__":
    unittest.main()
