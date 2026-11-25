"""팔·벨트 판정을 사진마다 한 번에 계산(#434) 시험: 부품별 계산과 판정·기록 값이 같음(무작위·경계), 계산 횟수,
팔→벨트 카메라 위치·표본 다시 쓰기와 그 조건, 스레드 전환 간격 옵션."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import yaml

from eyetoarm.transforms.planar_transform import look_at, w_to_image
from eyetoarm.vision import arm_region, belt_region
from eyetoarm.vision.ai_result import ARM_REGION, OUTSIDE_BELT, AiInstance, reject_instances
from eyetoarm.vision.arm_occlusion import world_capsules
from eyetoarm.vision.arm_region import ArmRegionFilter, arm_fraction, parse_arm_region
from eyetoarm.vision.belt_region import BeltRegionFilter, inside_fraction, parse_belt_region
from eyetoarm.vision.live_observation_envelope import TfLookupResult
from eyetoarm.vision.observation_envelope import CaptureTf
from test_ai_result import seg_result

ROOT = Path(__file__).resolve().parents[2]
ARM_CONFIG = ROOT / "config" / "vision" / "arm_region_filter.json"
BELT_CONFIG = ROOT / "config" / "vision" / "belt_region_filter.json"
FRAMES = ROOT / "config" / "robot" / "frames.yaml"
SCRIPT = ROOT / "scripts" / "run_pipeline.py"
K = np.array([[1171.5, 0.0, 640.0], [0.0, 1171.5, 360.0], [0.0, 0.0, 1.0]])   # 링크 카메라와 비슷한 값(test_belt_region)
DIST = np.array([0.01, -0.02, 0.001, 0.0005, 0.0])   # 왜곡 보정 경로도 지나게 조금 둠
CAM_T = look_at((0.06, 0.14, 0.50), (0.55, 0.25, 0.10))


def arm_cfg(**over):
    """저장소 팔 설정(7개 캡슐) + 덮어쓰기."""
    doc = json.loads(ARM_CONFIG.read_text(encoding="utf-8"))
    doc.update(over)

    return parse_arm_region(doc)


def belt_cfg(**over):
    """저장소 벨트 설정 + frames.yaml + 덮어쓰기."""
    doc = json.loads(BELT_CONFIG.read_text(encoding="utf-8"))
    doc.update(over)

    return parse_belt_region(doc, yaml.safe_load(FRAMES.read_text(encoding="utf-8")))


def T_at(xyz):
    """회전 없는 4×4."""
    T = np.eye(4)
    T[:3, 3] = xyz

    return T


def link_poses(rng=None, near=(0.40, 0.10, 0.20)):
    """설정의 팔 링크마다 위치. rng가 있으면 near 둘레 무작위, 없으면 모두 near."""
    links = dict.fromkeys(o.link for o in arm_cfg().occluders)

    if rng is None:
        return {name: T_at(near) for name in links}

    return {name: T_at(np.asarray(near) + rng.uniform(-0.15, 0.15, size=3)) for name in links}


def square_at_w(x, y, z, half_m=0.004, cam=CAM_T):
    """W 점 주변 수평 정사각형을 영상에 투영한 다각형(픽셀)."""
    pts = np.array([[x - half_m, y - half_m, z], [x + half_m, y - half_m, z], [x + half_m, y + half_m, z],
                    [x - half_m, y + half_m, z]])

    return w_to_image(pts, K, DIST, cam).astype(np.float32)


def quad_px(rng, cx, cy):
    """픽셀 중심 둘레 임의 볼록 사각형(크기 3~40 px)."""
    r = rng.uniform(3.0, 40.0, size=4)
    a = np.sort(rng.uniform(0.0, 2 * np.pi, size=4))

    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)]).astype(np.float32)


def inst(index, polygon, cls="bolt"):
    """범위 안 인스턴스."""
    x0, y0 = polygon.min(axis=0)
    x1, y1 = polygon.max(axis=0)

    return AiInstance(index=index, class_index=0 if cls == "bolt" else 1, class_name=cls, confidence=0.9,
                      box_xyxy=(float(x0), float(y0), float(x1), float(y1)), polygon=polygon,
                      area_px=float(max(1.0, (x1 - x0) * (y1 - y0))), component_count=1, hole_count=0, clipped=False,
                      rejected=None, axis_endpoints_px=None, axis_reason=None, axis_overlap_with=None)


def ref_arm(result, *, K, dist, T_W_C, T_W_L_by_link, config):
    """비교 기준: #434 전 팔 판정(부품마다 `arm_fraction`)."""
    capsules = world_capsules(config.occluders, T_W_L_by_link)
    decisions, removed = [], set()

    for i in result.instances:
        if not i.in_scope or i.polygon is None:
            continue

        frac, n, center = arm_fraction(i.polygon, K, dist, T_W_C, capsules, config)
        excluded = frac >= config.min_fraction
        decisions.append({"index": i.index, "class_name": i.class_name, "confidence": round(i.confidence, 4),
                          "samples": n, "arm_fraction": round(frac, 4), "excluded": excluded, "center_w_mm": center})

        if excluded:
            removed.add(i.index)

    return reject_instances(result, removed, ARM_REGION), decisions


def ref_belt(result, *, K, dist, T_W_C, config):
    """비교 기준: #434 전 벨트 판정(부품마다 `inside_fraction`)."""
    decisions, removed = [], set()

    for i in result.instances:
        if not i.in_scope or i.polygon is None:
            continue

        frac, n, center = inside_fraction(i.polygon, K, dist, T_W_C, config)
        excluded = frac < config.min_inside_fraction
        decisions.append({"index": i.index, "class_name": i.class_name, "confidence": round(i.confidence, 4),
                          "samples": n, "inside_fraction": round(frac, 4), "excluded": excluded,
                          "center_w_top_mm": center})

        if excluded:
            removed.add(i.index)

    return reject_instances(result, removed, OUTSIDE_BELT), decisions


def random_scene(rng, n):
    """무작위 사진 하나: 후보 n개(영상 안·밖·가장자리 섞음), 카메라는 벨트를 보거나 위를 봄(평면과 안 만나는 표본)."""
    if rng.uniform() < 0.2:
        cam = look_at((0.2, 0.0, 0.3), (0.5, 0.0, 0.9))   # 위를 봄: 교점 없는 표본(NaN)
    else:
        cam = look_at((rng.uniform(0.0, 0.3), rng.uniform(-0.2, 0.3), rng.uniform(0.35, 0.6)),
                      (rng.uniform(0.45, 0.65), rng.uniform(-0.3, 0.3), 0.10))

    polys = [quad_px(rng, rng.uniform(-50, 1330), rng.uniform(-50, 770)) for _ in range(n)]

    return seg_result([inst(i, p, "bolt" if i % 3 else "nut") for i, p in enumerate(polys)]), cam


class SameAsPerPartTest(unittest.TestCase):
    """사진마다 한 번에 계산한 판정이 부품별 계산과 같다(값 그대로, 비트까지)."""

    def test_random_scenes_arm_and_belt_same(self):
        """무작위 사진 300장(후보 0~25개, 캡슐 위치·카메라 무작위): 판정 근거·거부 결과가 부품별 계산과 같다."""
        rng = np.random.default_rng(434)
        a, b = arm_cfg(), belt_cfg()

        for _ in range(300):
            res, cam = random_scene(rng, int(rng.integers(0, 26)))
            poses = link_poses(rng)
            got_out, got = arm_region.filter_result(res, K=K, dist=DIST, T_W_C=cam, T_W_L_by_link=poses, config=a)
            want_out, want = ref_arm(res, K=K, dist=DIST, T_W_C=cam, T_W_L_by_link=poses, config=a)
            self.assertEqual(got, want)
            self.assertEqual([i.rejected for i in got_out.instances], [i.rejected for i in want_out.instances])

            got_b_out, got_b = belt_region.filter_result(got_out, K=K, dist=DIST, T_W_C=cam, config=b)
            want_b_out, want_b = ref_belt(want_out, K=K, dist=DIST, T_W_C=cam, config=b)
            self.assertEqual(got_b, want_b)
            self.assertEqual([i.rejected for i in got_b_out.instances], [i.rejected for i in want_b_out.instances])

    def test_fractions_bitwise_same(self):
        """부품별 비율·중심이 `arm_fraction`·`inside_fraction`의 반올림 전 값과 같다(부동소수 차이 없음)."""
        rng = np.random.default_rng(7)
        a, b = arm_cfg(), belt_cfg()

        for _ in range(100):
            res, cam = random_scene(rng, int(rng.integers(1, 30)))
            poses = link_poses(rng)
            caps = world_capsules(a.occluders, poses)
            samples = [arm_region.sample_points(i.polygon, a.max_samples) for i in res.instances]
            got = arm_region.arm_fractions(samples, K, DIST, cam, caps, a)
            want = [arm_fraction(i.polygon, K, DIST, cam, caps, a) for i in res.instances]
            self.assertEqual(got, want)
            got_b = belt_region.inside_fractions(samples, K, DIST, cam, b)
            want_b = [inside_fraction(i.polygon, K, DIST, cam, b) for i in res.instances]
            self.assertEqual(got_b, want_b)

    def test_boundary_cases_same(self):
        """경계: 문턱 근처 가림(그리퍼 캡슐 가장자리), 벨트 가장자리 걸침, 교점 없는 카메라, 캡슐 없음이 부품별 계산과 같다."""
        a, b = arm_cfg(), belt_cfg()
        polys = [square_at_w(x, 0.0, 0.105, half_m=h) for x in (0.443, 0.448, 0.452, 0.457, 0.55) for h in (0.002, 0.006)]
        polys += [square_at_w(0.55, y, 0.10, half_m=0.004) for y in (-0.632, 0.628, 0.631, 0.634)]
        res = seg_result([inst(i, p) for i, p in enumerate(polys)])

        # 캡슐을 카메라–볼트 사이 여러 높이에 두어 가린 비율이 0.6 문턱 근처를 지나게 한다
        for z in np.linspace(0.12, 0.45, 23):
            poses = link_poses(near=(0.30, 0.08, float(z)))

            for config in (a, arm_cfg(margin_mm=3.0)):
                got = arm_region.filter_result(res, K=K, dist=DIST, T_W_C=CAM_T, T_W_L_by_link=poses, config=config)
                want = ref_arm(res, K=K, dist=DIST, T_W_C=CAM_T, T_W_L_by_link=poses, config=config)
                self.assertEqual(got[1], want[1])

        # 캡슐 없음(설정은 빈 목록을 받지 않으므로 계산 함수에 직접): 가림 0, 중심은 그대로
        samples = [arm_region.sample_points(i.polygon, a.max_samples) for i in res.instances]
        self.assertEqual(arm_region.arm_fractions(samples, K, DIST, CAM_T, [], a),
                         [arm_fraction(i.polygon, K, DIST, CAM_T, [], a) for i in res.instances])

        up = look_at((0.2, 0.0, 0.3), (0.5, 0.0, 0.9))

        for cam in (CAM_T, up):
            got = belt_region.filter_result(res, K=K, dist=DIST, T_W_C=cam, config=b)
            want = ref_belt(res, K=K, dist=DIST, T_W_C=cam, config=b)
            self.assertEqual(got[1], want[1])

    def test_zero_and_one_candidate(self):
        """후보 0개는 판정 없이 그대로, 1개는 부품별 계산과 같다. 범위 밖·다각형 없음은 건너뛴다."""
        a, b = arm_cfg(), belt_cfg()
        empty = seg_result([])
        self.assertEqual(arm_region.filter_result(empty, K=K, dist=DIST, T_W_C=CAM_T, T_W_L_by_link=link_poses(),
                                                  config=a), (empty, []))
        self.assertEqual(belt_region.filter_result(empty, K=K, dist=DIST, T_W_C=CAM_T, config=b), (empty, []))
        self.assertEqual(arm_region.arm_fractions([], K, DIST, CAM_T, [], a), [])
        self.assertEqual(belt_region.inside_fractions([], K, DIST, CAM_T, b), [])

        one = seg_result([inst(0, square_at_w(0.55, 0.1, 0.105)),
                          AiInstance(index=1, class_index=0, class_name="bolt", confidence=0.4, box_xyxy=(0, 0, 4, 4),
                                     polygon=None, area_px=0.0, component_count=0, hole_count=0, clipped=False,
                                     rejected="EMPTY_MASK", axis_endpoints_px=None, axis_reason=None,
                                     axis_overlap_with=None)])
        got = arm_region.filter_result(one, K=K, dist=DIST, T_W_C=CAM_T, T_W_L_by_link=link_poses(), config=a)
        want = ref_arm(one, K=K, dist=DIST, T_W_C=CAM_T, T_W_L_by_link=link_poses(), config=a)
        self.assertEqual((got[1], len(got[1])), (want[1], 1))

    def test_tiny_polygon_uses_vertices(self):
        """내부 픽셀이 없는(아주 작은) 다각형은 꼭짓점을 표본으로 쓰는 것까지 부품별 계산과 같다."""
        tiny = np.array([[500.2, 300.2], [500.4, 300.2], [500.3, 300.4]], dtype=np.float32)
        res = seg_result([inst(0, tiny), inst(1, square_at_w(0.55, 0.0, 0.105))])
        a = arm_cfg()
        got = arm_region.filter_result(res, K=K, dist=DIST, T_W_C=CAM_T, T_W_L_by_link=link_poses(), config=a)
        want = ref_arm(res, K=K, dist=DIST, T_W_C=CAM_T, T_W_L_by_link=link_poses(), config=a)
        self.assertEqual(got[1], want[1])


class Counter:
    """함수 호출 수를 세는 감싸개."""

    def __init__(self, fn):
        """감쌀 함수."""
        self.fn, self.calls = fn, 0

    def __call__(self, *a, **k):
        """부르고 센다."""
        self.calls += 1

        return self.fn(*a, **k)


class OncePerFrameTest(unittest.TestCase):
    """투영·교점 계산은 사진 하나에 한 번이다(부품 수와 관계없음)."""

    def test_arm_projects_once(self):
        """후보 5개 사진에서 팔 투영(`pixels_to_w`)과 캡슐 거리는 각각 한 번만 부른다."""
        res = seg_result([inst(i, square_at_w(0.50 + 0.02 * i, 0.0, 0.105)) for i in range(5)])
        ptw, seg = Counter(arm_region.pixels_to_w), Counter(arm_region.segment_distances)

        with mock.patch.object(arm_region, "pixels_to_w", ptw), mock.patch.object(arm_region, "segment_distances", seg):
            arm_region.filter_result(res, K=K, dist=DIST, T_W_C=CAM_T, T_W_L_by_link=link_poses(), config=arm_cfg())

        self.assertEqual((ptw.calls, seg.calls), (1, 1))

    def test_belt_intersects_once(self):
        """후보 5개 사진에서 벨트 교점(`rays_to_plane`)은 한 번만 부른다."""
        res = seg_result([inst(i, square_at_w(0.50 + 0.02 * i, 0.0, 0.10)) for i in range(5)])
        rays = Counter(belt_region.rays_to_plane)

        with mock.patch.object(belt_region, "rays_to_plane", rays):
            belt_region.filter_result(res, K=K, dist=DIST, T_W_C=CAM_T, config=belt_cfg())

        self.assertEqual(rays.calls, 1)


class CountingTf:
    """촬영 stamp TF 대역: 조회마다 이름을 남긴다. `fail`이면 그 이름이 실패."""

    def __init__(self, fail=None):
        """실패시킬 이름(카메라는 None)."""
        self.fail, self.names = fail, []

    def __call__(self, frame):
        """조회 대역."""
        child = getattr(frame, "optical_frame", None)
        self.names.append(child)

        if child == self.fail:
            return TfLookupResult(None, "unavailable", 0.1, "fake")

        T = CAM_T if child == "link_cam_optical_frame" else T_at((0.40, 0.10, 0.20))

        return TfLookupResult(CaptureTf("world", child, frame.ros_stamp_ns, T), "ok", 0.1)


def make_job(seq=7):
    """작업 대역(사진마다 새 프레임 객체)."""
    return SimpleNamespace(frame=SimpleNamespace(image_seq=seq, ros_stamp_ns=1_000_000 + seq,
                                                 optical_frame="link_cam_optical_frame"))


def chain(res_for, *, arm_tf, belt_tf, belt_config=None, arm_records=None, belt_records=None):
    """운영 연결과 같은 순서: 검출기 → 팔 감싸개 → 벨트 감싸개(run_pipeline과 같음)."""
    cam = SimpleNamespace(K=K, dist=DIST)
    arm = ArmRegionFilter(lambda j, *a: ("obs", res_for(j)), tf_lookup=arm_tf, camera_for=lambda f: cam,
                          config=arm_cfg(), select=lambda j, r: "reselected",
                          on_decision=None if arm_records is None else arm_records.append)
    belt = BeltRegionFilter(arm, tf_lookup=belt_tf, camera_for=lambda f: cam, config=belt_config or belt_cfg(),
                            select=lambda j, r: "reselected",
                            on_decision=None if belt_records is None else belt_records.append)

    return arm, belt


class ShareTest(unittest.TestCase):
    """벨트 감싸개가 같은 사진의 팔 카메라 위치·표본을 다시 쓰는 조건과 그 결과."""

    def scene(self):
        """후보 4개: 벨트 안 둘, 벨트 밖 하나, 가장자리 하나."""
        return seg_result([inst(0, square_at_w(0.55, 0.0, 0.10)), inst(1, square_at_w(0.60, 0.2, 0.10)),
                           inst(2, square_at_w(0.30, 0.2, 0.0025)), inst(3, square_at_w(0.452, 0.1, 0.10))])

    def test_reuses_camera_tf_and_samples(self):
        """팔이 판정한 사진이면 벨트는 카메라 위치를 다시 조회하지 않고, 표본 뽑기도 후보마다 한 번뿐이다."""
        res = self.scene()
        arm_tf, belt_tf = CountingTf(), CountingTf()
        _, belt = chain(lambda j: res, arm_tf=arm_tf, belt_tf=belt_tf)
        sp = Counter(arm_region.sample_points)

        with mock.patch.object(arm_region, "sample_points", sp), mock.patch.object(belt_region, "sample_points", sp):
            belt(make_job())

        self.assertEqual(belt_tf.names, [])
        self.assertEqual(arm_tf.names.count("link_cam_optical_frame"), 1)
        self.assertEqual(sp.calls, 4)

    def test_records_same_as_per_part(self):
        """팔·벨트 기록(arm_region.jsonl·belt_region.jsonl 줄)이 #434 전 계산으로 만든 기록과 같다."""
        res = self.scene()
        arm_rec, belt_rec = [], []
        _, belt = chain(lambda j: res, arm_tf=CountingTf(), belt_tf=CountingTf(), arm_records=arm_rec,
                        belt_records=belt_rec)
        j = make_job()
        _, out = belt(j)

        poses = {name: T_at((0.40, 0.10, 0.20)) for name in dict.fromkeys(o.link for o in arm_cfg().occluders)}
        want_arm_out, want_arm = ref_arm(res, K=K, dist=DIST, T_W_C=CAM_T, T_W_L_by_link=poses, config=arm_cfg())
        want_belt_out, want_belt = ref_belt(want_arm_out, K=K, dist=DIST, T_W_C=CAM_T, config=belt_cfg())
        base = {"image_seq": 7, "ros_stamp_ns": 1_000_007, "status": "ok"}
        self.assertEqual(arm_rec, [{**base, "decisions": want_arm}])
        self.assertEqual(belt_rec, [{**base, "decisions": want_belt}])
        self.assertEqual([i.rejected for i in out.instances], [i.rejected for i in want_belt_out.instances])
        self.assertIn(OUTSIDE_BELT, [i.rejected for i in out.instances])

    def test_other_frame_not_reused(self):
        """팔이 판정한 사진과 다른 프레임 객체(같은 번호라도)에는 공유를 주지 않는다."""
        res = self.scene()
        arm, _ = chain(lambda j: res, arm_tf=CountingTf(), belt_tf=CountingTf())
        first = make_job()
        arm(first)
        self.assertIsNotNone(arm.region_share(first.frame))
        self.assertIsNone(arm.region_share(make_job().frame))

    def test_cleared_when_arm_does_not_judge(self):
        """다음 사진에서 팔이 판정하지 않으면(후보 없음·TF 실패) 앞 사진 것은 남지 않고 벨트가 직접 조회한다."""
        results = {7: self.scene(), 8: seg_result([]), 9: self.scene()}
        arm_tf, belt_tf = CountingTf(), CountingTf()
        arm, belt = chain(lambda j: results[j.frame.image_seq], arm_tf=arm_tf, belt_tf=belt_tf)
        belt(make_job(7))
        j8 = make_job(8)
        belt(j8)
        self.assertIsNone(arm.region_share(j8.frame))

        # 팔 링크 TF 실패: 팔은 판정하지 않고, 벨트는 자기 조회로 판정한다(지금과 같은 동작)
        arm_tf.fail = "wrist_1_link"
        belt_rec = []
        belt._on_decision = belt_rec.append
        j9 = make_job(9)
        belt(j9)
        self.assertIsNone(arm.region_share(j9.frame))
        self.assertEqual(belt_tf.names, ["link_cam_optical_frame"])
        self.assertEqual(belt_rec[0]["status"], "ok")

    def test_belt_uses_arm_camera_tf(self):
        """팔이 카메라 위치를 얻은 사진에서는 벨트도 같은 값을 쓴다(벨트 조회 실패로 기록이 갈리지 않음)."""
        res = self.scene()
        belt_tf = CountingTf(fail="link_cam_optical_frame")
        belt_rec = []
        _, belt = chain(lambda j: res, arm_tf=CountingTf(), belt_tf=belt_tf, belt_records=belt_rec)
        belt(make_job())
        self.assertEqual((belt_tf.names, belt_rec[0]["status"]), ([], "ok"))

    def test_different_max_samples_draws_own(self):
        """벨트 표본 상한이 팔과 다르면 표본은 다시 쓰지 않고 새로 뽑는다(판정은 그 상한의 부품별 계산과 같음)."""
        res = self.scene()
        b32 = belt_cfg(max_samples=32)
        belt_rec = []
        _, belt = chain(lambda j: res, arm_tf=CountingTf(), belt_tf=CountingTf(), belt_config=b32,
                        belt_records=belt_rec)
        sp = Counter(arm_region.sample_points)

        with mock.patch.object(arm_region, "sample_points", sp), mock.patch.object(belt_region, "sample_points", sp):
            belt(make_job())

        poses = {name: T_at((0.40, 0.10, 0.20)) for name in dict.fromkeys(o.link for o in arm_cfg().occluders)}
        arm_out, _ = ref_arm(res, K=K, dist=DIST, T_W_C=CAM_T, T_W_L_by_link=poses, config=arm_cfg())
        _, want = ref_belt(arm_out, K=K, dist=DIST, T_W_C=CAM_T, config=b32)
        self.assertEqual(belt_rec[0]["decisions"], want)
        self.assertEqual(sp.calls, 4 + len(want))
        self.assertTrue(all(d["samples"] <= 32 for d in want))

    def test_belt_without_arm_looks_up_itself(self):
        """팔 감싸개 없이 벨트만 쓰면(팔 필터 끔) 지금처럼 직접 조회·표본 추출한다."""
        res = self.scene()
        belt_tf = CountingTf()
        cam = SimpleNamespace(K=K, dist=DIST)
        rec = []
        belt = BeltRegionFilter(lambda j, *a: ("obs", res), tf_lookup=belt_tf, camera_for=lambda f: cam,
                                config=belt_cfg(), select=lambda j, r: "reselected", on_decision=rec.append)
        belt(make_job())
        _, want = ref_belt(res, K=K, dist=DIST, T_W_C=CAM_T, config=belt_cfg())
        self.assertEqual((belt_tf.names, rec[0]["decisions"]), (["link_cam_optical_frame"], want))


def load_run_pipeline():
    """scripts/run_pipeline.py를 모듈로(시험 전용 이름)."""
    spec = importlib.util.spec_from_file_location("run_pipeline_switch_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


class SwitchIntervalOptionTest(unittest.TestCase):
    """`--switch-interval-ms`: 해석·적용, 기본은 바꾸지 않음."""

    @classmethod
    def setUpClass(cls):
        """실행기 모듈."""
        cls.rp = load_run_pipeline()

    def test_parse_bounds(self):
        """0 초과 100 이하 유한 실수만 받는다."""
        import argparse

        self.assertEqual(self.rp._switch_interval_ms("0.5"), 0.5)
        self.assertEqual(self.rp._switch_interval_ms("100"), 100.0)

        for bad in ("0", "-1", "100.5", "nan", "inf", "abc", ""):
            with self.assertRaises(argparse.ArgumentTypeError, msg=bad):
                self.rp._switch_interval_ms(bad)

    def test_default_none_keeps_interval(self):
        """옵션이 없으면 기본값 None이고 전환 간격을 건드리지 않는다."""
        action = self.rp.build_parser()._option_string_actions["--switch-interval-ms"]
        self.assertIsNone(action.default)
        before = sys.getswitchinterval()
        err = io.StringIO()
        self.assertIsNone(self.rp._apply_switch_interval(SimpleNamespace(switch_interval_ms=None), err))
        self.assertEqual((sys.getswitchinterval(), err.getvalue()), (before, ""))

    def test_apply_sets_and_reports(self):
        """옵션이 있으면 이 프로세스의 간격을 바꾸고 이전 값을 돌려주며 한 줄 남긴다."""
        before = sys.getswitchinterval()
        err = io.StringIO()

        try:
            prev = self.rp._apply_switch_interval(SimpleNamespace(switch_interval_ms=0.5), err)
            self.assertEqual(prev, before)
            self.assertAlmostEqual(sys.getswitchinterval(), 0.0005, places=9)
            self.assertIn("switch_interval=0.5ms", err.getvalue())
        finally:
            sys.setswitchinterval(before)


if __name__ == "__main__":
    unittest.main()
