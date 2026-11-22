"""팔·그리퍼 투영 영역 검출 제외 시험: 광선 판정, 거부 표시·관측 재선택, TF·카메라 없음, 추적·목록 제외, 설정."""
from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from eyetoarm.transforms.planar_transform import look_at, pixels_to_w, w_to_image
from eyetoarm.vision.ai_result import ARM_REGION, AiInstance
from eyetoarm.vision.arm_occlusion import Occluder, occluded_fn, segment_distance, world_capsules
from eyetoarm.vision.arm_region import (ArmRegionConfigError, ArmRegionFilter, arm_fraction, filter_result,
                                        load_arm_region_config, parse_arm_region, sample_points, segment_distances)
from eyetoarm.vision.live_observation_envelope import TfLookupResult
from eyetoarm.vision.observation_envelope import CaptureTf
from eyetoarm.vision.world_tracking import world_inputs
from test_ai_result import seg_result

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "vision" / "arm_region_filter.json"
K = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]])
DIST = np.zeros(5)
CAM_T = look_at((0.0, 0.0, 0.5), (0.5, 0.0, 0.105))
GRIP = Occluder("grip", (0.0, 0.0, 0.0), (0.0, 0.0, 0.03), 0.03)   # 카메라–(0.5, 0) 사이 (0.25, 0, 0.29)에 둠


def T_at(xyz):
    """회전 없는 4×4."""
    T = np.eye(4)
    T[:3, 3] = xyz
    return T


POSES = {"grip": T_at([0.25, 0.0, 0.29])}


def cfg(**over):
    """예시 파일 값 + 캡슐 GRIP 하나 + 덮어쓰기."""
    doc = json.loads(CONFIG.read_text(encoding="utf-8"))
    doc["occluders"] = [{"link": "grip", "p0_m": [0, 0, 0], "p1_m": [0, 0, 0.03], "radius_m": 0.03}]
    doc.update(over)

    return parse_arm_region(doc)


def square_at_w(x, y, half_m=0.004, z=0.105):
    """W 점 주변 정사각형을 링크 영상에 투영한 다각형(픽셀)."""
    pts = np.array([[x - half_m, y - half_m, z], [x + half_m, y - half_m, z], [x + half_m, y + half_m, z],
                    [x - half_m, y + half_m, z]])

    return w_to_image(pts, K, DIST, CAM_T).astype(np.float32)


def square_px(cx, cy, half=8.0):
    """픽셀 중심 정사각형."""
    return np.array([[cx - half, cy - half], [cx + half, cy - half], [cx + half, cy + half], [cx - half, cy + half]],
                    dtype=np.float32)


def bolt(index, polygon, endpoints=True, overlap=None, conf=0.9):
    """축 경로를 거친 볼트 인스턴스."""
    x0, y0 = polygon.min(axis=0)
    x1, y1 = polygon.max(axis=0)
    ep = np.array([[x0 + 1.0, y0 + 1.0], [x1 - 1.0, y1 - 1.0]]) if endpoints else None

    return AiInstance(index=index, class_index=0, class_name="bolt", confidence=conf,
                      box_xyxy=(float(x0), float(y0), float(x1), float(y1)), polygon=polygon,
                      area_px=float((x1 - x0) * (y1 - y0)), component_count=1, hole_count=0, clipped=False,
                      rejected=None, axis_endpoints_px=ep, axis_reason=None if endpoints else "AXIS_EXTRACTION_ERROR",
                      axis_overlap_with=overlap)


def grip_px():
    """그리퍼 캡슐 중심의 링크 영상 픽셀."""
    return w_to_image(np.array([[0.25, 0.0, 0.305]]), K, DIST, CAM_T)[0]


class FractionTest(unittest.TestCase):
    """광선 판정."""

    def test_arm_silhouette_excluded_bolt_kept(self):
        """그리퍼 실루엣 위 다각형은 가림 비율 1, 벨트 위 다른 곳 볼트는 0."""
        caps = world_capsules((GRIP,), POSES)
        c = cfg()
        on_arm, _, _ = arm_fraction(square_px(*grip_px()), K, DIST, CAM_T, caps, c)
        away, n, center = arm_fraction(square_at_w(0.5, 0.10), K, DIST, CAM_T, caps, c)
        self.assertEqual((on_arm, away), (1.0, 0.0))
        self.assertGreaterEqual(n, 3)
        self.assertAlmostEqual(center[0], 500.0, delta=2.0)   # 표본 평면 위 중심 W [mm]
        self.assertAlmostEqual(center[1], 100.0, delta=2.0)

    def test_bolt_under_gripper_partially_hidden_kept(self):
        """그리퍼 실루엣 가장자리에 반쯤 걸친 볼트(보이는 부분만 마스크)는 비율 < 0.6이라 남는다."""
        caps = world_capsules((GRIP,), POSES)
        u, v = grip_px()
        r_px = 0.03 / np.linalg.norm(np.array([0.25, 0.0, 0.305]) - CAM_T[:3, 3]) * 1000.0
        poly = square_px(u + r_px, v, half=r_px * 0.8)   # 절반 남짓 실루엣 밖
        frac, _, _ = arm_fraction(poly, K, DIST, CAM_T, caps, cfg())
        self.assertLess(frac, 0.6)
        self.assertGreater(frac, 0.0)

    def test_threshold_boundary(self):
        """같은 비율이라도 문턱이 그 아래면 제외, 위면 유지."""
        caps = world_capsules((GRIP,), POSES)
        u, v = grip_px()
        r_px = 0.03 / np.linalg.norm(np.array([0.25, 0.0, 0.305]) - CAM_T[:3, 3]) * 1000.0
        poly = square_px(u + r_px, v, half=r_px * 0.8)
        frac, _, _ = arm_fraction(poly, K, DIST, CAM_T, caps, cfg())
        res = seg_result([bolt(0, poly)])
        out_lo, _ = filter_result(res, K=K, dist=DIST, T_W_C=CAM_T, T_W_L_by_link=POSES,
                                  config=cfg(min_fraction=frac - 0.01))
        out_hi, _ = filter_result(res, K=K, dist=DIST, T_W_C=CAM_T, T_W_L_by_link=POSES,
                                  config=cfg(min_fraction=frac + 0.01))
        self.assertEqual((out_lo.instances[0].rejected, out_hi.instances[0].rejected), (ARM_REGION, None))


def scalar_fraction(polygon, K, dist, T_W_C, capsules, config):
    """비교 기준: 표본마다 `occluded_fn`(스칼라 선분 거리)으로 센 가림 비율·표본 수·중심."""
    pts = sample_points(polygon, config.max_samples)
    w = pixels_to_w(pts, K, dist, T_W_C, config.plane_z_m) * 1000.0
    occ = occluded_fn(np.asarray(T_W_C)[:3, 3], capsules, config.plane_z_m)
    finite = [(float(x), float(y)) for x, y in w if math.isfinite(x) and math.isfinite(y)]
    hits = sum(1 for x, y in finite if occ(x, y, config.margin_mm))
    center = None if not finite else [round(float(np.mean([p[0] for p in finite])), 1),
                                      round(float(np.mean([p[1] for p in finite])), 1)]

    return hits / len(pts), len(pts), center


def random_pose(rng):
    """카메라와 볼트 평면 사이 임의 위치·회전의 4×4."""
    q = rng.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = [rng.uniform(0.0, 0.6), rng.uniform(-0.2, 0.2), rng.uniform(0.12, 0.45)]

    return T


class VectorizedSameAsScalarTest(unittest.TestCase):
    """배열 계산이 기존 스칼라 구현과 같은 판정을 낸다."""

    def test_segment_distances_match_scalar(self):
        """임의·퇴화(길이 0 캡슐, 카메라와 같은 점, 평행 선분) 선분 쌍의 거리가 `segment_distance`와 같다."""
        rng = np.random.default_rng(7)
        c = np.array([0.0, 0.0, 0.5])
        P = np.vstack([rng.uniform(-0.5, 0.8, size=(40, 3)), c, c + [0.1, 0.0, 0.0]])
        A = np.vstack([rng.uniform(-0.3, 0.6, size=(6, 3)), [0.2, 0.1, 0.3], c + [0.0, 0.05, 0.0]])
        B = np.vstack([rng.uniform(-0.3, 0.6, size=(6, 3)), [0.2, 0.1, 0.3], c + [0.2, 0.05, 0.0]])
        got = segment_distances(c, P, A, B)
        want = np.array([[segment_distance(c, p, a, b) for a, b in zip(A, B)] for p in P])
        np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-12)

    def test_fraction_matches_scalar_on_random_scenes(self):
        """실제 설정 캡슐 7개·임의 링크 자세·임의 다각형·여백 300경우에서 비율·표본 수·중심이 모두 같다."""
        rng = np.random.default_rng(262)
        base = load_arm_region_config(str(CONFIG))[0]
        links = sorted({o.link for o in base.occluders})
        fractions = set()

        for k in range(300):
            config = parse_arm_region({**json.loads(CONFIG.read_text(encoding="utf-8")),
                                       "margin_mm": float(rng.choice([0.0, 5.0, 20.0]))})
            caps = world_capsules(config.occluders, {name: random_pose(rng) for name in links})
            poly = square_px(rng.uniform(40, 1240), rng.uniform(0 if k % 10 == 0 else 200, 700),
                             half=rng.uniform(2, 60))

            with self.subTest(case=k):
                got = arm_fraction(poly, K, DIST, CAM_T, caps, config)
                self.assertEqual(got, scalar_fraction(poly, K, DIST, CAM_T, caps, config))

            fractions.add(got[0])

        self.assertTrue(any(0.0 < f < 1.0 for f in fractions))   # 부분 가림 경우가 실제로 섞였다
        self.assertIn(0.0, fractions)
        self.assertIn(1.0, fractions)

    def test_no_capsules_and_non_finite_samples(self):
        """캡슐이 없으면 0, 평면 교점이 비유한인 표본은 세지 않고 모두 비유한이면 중심 None — 스칼라 구현과 같다."""
        c = cfg()
        caps = world_capsules((GRIP,), POSES)
        poly = square_px(*grip_px())
        real = pixels_to_w   # 바꿔 끼우기 전 원래 함수
        self.assertEqual(arm_fraction(poly, K, DIST, CAM_T, [], c), scalar_fraction(poly, K, DIST, CAM_T, [], c))

        for bad in (np.nan, np.inf):
            def fake(pts, *a, bad=bad, **kw):
                """앞 절반 표본만 비유한 교점."""
                w = real(pts, *a, **kw)
                w[: len(w) // 2] = bad
                return w

            def none(pts, *a, bad=bad, **kw):
                """모든 표본이 비유한 교점."""
                return np.full((len(pts), 2), bad)

            for fn in (fake, none):
                with self.subTest(bad=bad, fn=fn.__name__), \
                        mock.patch("eyetoarm.vision.arm_region.pixels_to_w", fn), \
                        mock.patch(f"{__name__}.pixels_to_w", fn):
                    got = arm_fraction(poly, K, DIST, CAM_T, caps, c)
                    self.assertEqual(got, scalar_fraction(poly, K, DIST, CAM_T, caps, c))
                    self.assertEqual(got[2] is None, fn is none)


class FilterTest(unittest.TestCase):
    """거부 표시와 소비자 쪽 제외."""

    def test_reject_marks_out_of_scope_and_clears_overlap(self):
        """그리퍼 인스턴스는 ARM_REGION·범위 밖·축 필드 없음, 짝 겹침이 풀려 실제 볼트만 축 유효로 남는다."""
        arm = bolt(1, square_px(*grip_px()), overlap=(0,))
        real = bolt(0, square_at_w(0.5, 0.10), overlap=(1,))
        res = seg_result([real, arm])
        self.assertEqual(res.axis_valid_indices, ())
        out, decisions = filter_result(res, K=K, dist=DIST, T_W_C=CAM_T, T_W_L_by_link=POSES, config=cfg())
        a = out.instances[1]
        self.assertEqual((a.rejected, a.in_scope, a.axis_evaluated), (ARM_REGION, False, False))
        self.assertIsNone(out.instances[0].axis_overlap_with)
        self.assertEqual(out.axis_valid_indices, (0,))
        self.assertEqual(out.candidate_count, 1)
        self.assertEqual([d["excluded"] for d in decisions], [False, True])

    def test_world_inputs_skip_arm_region(self):
        """W 추적 입력에서 ARM_REGION은 not_in_scope."""
        out, _ = filter_result(seg_result([bolt(0, square_px(*grip_px()))]), K=K, dist=DIST, T_W_C=CAM_T,
                               T_W_L_by_link=POSES, config=cfg())
        w = world_inputs(out, K=K, dist=DIST, T_W_C=CAM_T, plane_z_m=0.105)
        self.assertEqual((w[0].candidate, w[0].untracked_reason), (False, "not_in_scope"))

    def test_no_change_returns_same_object(self):
        """제외가 없으면 같은 결과 객체(바이트 동일 경로)."""
        res = seg_result([bolt(0, square_at_w(0.5, 0.10))])
        out, _ = filter_result(res, K=K, dist=DIST, T_W_C=CAM_T, T_W_L_by_link=POSES, config=cfg())
        self.assertIs(out, res)


class FakeTf:
    """카메라·팔 링크 TF 대역. `fail`이면 그 이름 조회 실패."""

    def __init__(self, fail=None):
        """실패시킬 링크 이름(없으면 None)."""
        self.fail = fail

    def __call__(self, frame):
        """촬영 stamp 조회 대역."""
        child = getattr(frame, "optical_frame", None)

        if child == self.fail:
            return TfLookupResult(None, "unavailable", 0.1, "fake")

        T = POSES[child] if child in POSES else CAM_T

        return TfLookupResult(CaptureTf("world", child or "cam", frame.ros_stamp_ns, T), "ok", 0.1)


def job():
    """작업 대역."""
    return SimpleNamespace(frame=SimpleNamespace(image_seq=7, ros_stamp_ns=1_000_000,
                                                 optical_frame="link_cam_optical_frame"))


class WrapperTest(unittest.TestCase):
    """감싸개: 관측 재선택·토큰 인자·TF·카메라 없음·기록."""

    def make(self, tf=None, camera=True, takes_token=False):
        """감싼 검출기."""
        res = seg_result([bolt(0, square_at_w(0.5, 0.10), overlap=(1,)),
                          bolt(1, square_px(*grip_px()), overlap=(0,))])
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
        f = ArmRegionFilter(inner, tf_lookup=tf or FakeTf(), camera_for=lambda frame: cam, config=cfg(),
                            select=select, on_decision=self.records.append)
        f.prepare = lambda: "prepared"
        return f

    def test_reselects_and_passes_token(self):
        """그리퍼 제외 뒤 관측을 다시 고르고(실제 볼트만 유효), 토큰 인자를 그대로 넘긴다."""
        f = self.make()
        obs, out = f(job(), "token")
        self.assertEqual((obs, self.selected, self.calls), ("obs-after", [(0,)], [("token",)]))
        self.assertEqual(out.instances[1].rejected, ARM_REGION)
        self.assertEqual(self.records[0]["status"], "ok")
        self.assertEqual(f.summary()["counts"]["excluded"], 1)

    def test_summary_splits_tf_time(self):
        """요약은 전체 처리 ms와 그중 TF 조회 ms를 따로 낸다(TF 실패 프레임은 TF ms에 넣지 않는다)."""
        ticks = iter(range(0, 1000, 5))
        f = self.make()
        f._clock = lambda: next(ticks) / 1000.0   # 부를 때마다 5 ms
        f(job())
        s = f.summary()
        self.assertEqual((s["ms_median"], s["tf_ms_median"], s["tf_ms_p95"]), (15.0, 5.0, 5.0))

        g = self.make(tf=FakeTf(fail="grip"))
        g(job())
        self.assertEqual((g.summary()["tf_ms_median"], g.summary()["counts"]["tf_unavailable"]), (None, 1))

    def test_tf_failure_keeps_result(self):
        """팔 링크 TF가 없으면 제외하지 않고 사유를 남긴다."""
        f = self.make(tf=FakeTf(fail="grip"))
        obs, out = f(job())
        self.assertEqual(obs, "obs-before")
        self.assertIsNone(out.instances[1].rejected)
        self.assertEqual(self.records[0]["status"], "tf_unavailable:grip")
        self.assertEqual(f.summary()["counts"]["tf_unavailable"], 1)

    def test_no_camera_keeps_result(self):
        """카메라 보정이 없으면 제외하지 않는다."""
        f = self.make(camera=False)
        obs, _ = f(job())
        self.assertEqual((obs, self.records[0]["status"]), ("obs-before", "no_camera"))

    def test_filter_error_keeps_result_and_counts(self):
        """판정 중 오류(잘못된 카메라 K)는 제외 없이 결과를 그대로 두고 사유·오류 수를 남긴다."""
        res = seg_result([bolt(0, square_px(*grip_px()))])
        f = ArmRegionFilter(lambda j: ("obs-before", res), tf_lookup=FakeTf(),
                            camera_for=lambda fr: SimpleNamespace(K="bad", dist=DIST), config=cfg(),
                            select=lambda j, r: "obs-after", on_decision=(recs := []).append)
        obs, out = f(job())

        self.assertEqual((obs, out), ("obs-before", res))
        self.assertTrue(recs[0]["status"].startswith("error:"))
        self.assertEqual(f.summary()["counts"]["errors"], 1)

    def test_non_ai_value_passthrough(self):
        """AI 결과가 아닌 값은 그대로 통과, 다른 속성도 넘긴다."""
        f = ArmRegionFilter(lambda j: "planar", tf_lookup=FakeTf(), camera_for=lambda fr: None, config=cfg(),
                            select=lambda j, r: None)
        self.assertEqual(f(job()), "planar")


class ReselectTest(unittest.TestCase):
    """실제 관측 선택(`observation_from_ai`)으로 다시 고르기와 기록·표시."""

    def setUp(self):
        """끝점 추출 켠 AI 설정·보정 프로파일이 있는 1280×720 작업."""
        from eyetoarm.vision.pipeline import VisionPipeline
        from test_ai_config import ai_config
        from test_pipeline import RUN_ID, FakeClock, frame, profile, settings

        self.cfg = ai_config(axis_extractor="m2")
        fr = frame(0, image=np.zeros((720, 1280, 3), np.uint8), width=1280, height=720, calibration_id="cal-arm",
                   optical_frame="link_cam_optical_frame", ros_stamp_ns=1_000_000)
        prof = profile(width=1280, height=720, calibration_id="cal-arm")
        pipe = VisionPipeline(settings(detector_config=self.cfg, profile=prof), run_id=RUN_ID, clock_ns=FakeClock())
        pipe.offer_frame(fr)
        self.job = pipe.start_next()

    def run_filter(self, instances):
        """실제 선택 함수로 감싸 (관측 전, 관측 후, 결과 후)."""
        from eyetoarm.vision.ai_convert import observation_from_ai

        res = seg_result(instances, config_sha256=self.cfg.config_sha256, crop_wh=(1280, 720))
        before = observation_from_ai(self.job, res)
        f = ArmRegionFilter(lambda j: (before, res), tf_lookup=FakeTf(),
                            camera_for=lambda fr: SimpleNamespace(K=K, dist=DIST), config=cfg(),
                            select=observation_from_ai)
        after, out = f(self.job)

        return before, after, out

    def test_only_gripper_valid_becomes_invalid(self):
        """진짜 볼트는 끝점 없음(무효)·그리퍼만 유효 → 그리퍼를 고르던 관측이 무효로 바뀐다."""
        before, after, _ = self.run_filter([bolt(0, square_at_w(0.5, 0.10), endpoints=False),
                                            bolt(1, square_px(*grip_px()))])

        self.assertEqual((before.valid, before.selected_index), (True, 1))
        self.assertFalse(after.valid)
        self.assertIsNone(after.selected_index)

    def test_multiple_becomes_real_bolt(self):
        """진짜·그리퍼 둘 다 유효(MULTIPLE_OBJECTS) → 그리퍼 제외 뒤 진짜 볼트를 고른다."""
        before, after, _ = self.run_filter([bolt(0, square_at_w(0.5, 0.10)), bolt(1, square_px(*grip_px()))])

        self.assertEqual((before.valid, before.reason), (False, "MULTIPLE_OBJECTS"))
        self.assertEqual((after.valid, after.selected_index), (True, 0))

    def test_process_record_and_detect_drawing(self):
        """처리 기록의 인스턴스 요약에 rejected=ARM_REGION이 남고, detect 그림에는 그 인스턴스를 그리지 않는다."""
        import io
        from eyetoarm.vision.live_view import draw_detections
        from eyetoarm.vision.pipeline_records import _instance_summary

        _, _, out = self.run_filter([bolt(0, square_at_w(0.5, 0.10)), bolt(1, square_px(*grip_px()))])
        summary = [_instance_summary(i) for i in out.instances]
        self.assertEqual([s["rejected"] for s in summary], [None, ARM_REGION])
        json.dump(summary, io.StringIO())   # JSON으로 직렬화된다

        _, drawn = draw_detections(np.zeros((720, 1280, 3), np.uint8), out)
        self.assertEqual(drawn, 1)


class ConfigTest(unittest.TestCase):
    """설정 검사."""

    def test_example_file(self):
        """예시 파일: 캡슐 7개, 여유 0, 문턱 0.6, 표본 64, sha256."""
        c, sha = load_arm_region_config(str(CONFIG))
        self.assertEqual((len(c.occluders), c.margin_mm, c.min_fraction, c.max_samples, len(sha)),
                         (7, 0.0, 0.6, 64, 64))

    def test_bad_values(self):
        """키 누락·모르는 키·status·범위 오류는 거부."""
        doc = json.loads(CONFIG.read_text(encoding="utf-8"))

        for bad in ({k: v for k, v in doc.items() if k != "margin_mm"}, {**doc, "x": 1}, {**doc, "status": "prod"},
                    {**doc, "min_fraction": 0}, {**doc, "min_fraction": 1.5}, {**doc, "max_samples": 2},
                    {**doc, "plane_z_m": 0}, {**doc, "occluders": []}):
            with self.subTest(bad=list(bad)[-1]), self.assertRaises(ArmRegionConfigError):
                parse_arm_region(bad)


if __name__ == "__main__":
    unittest.main()
