"""종류별 투영 높이·시야 보정(#270): 설정 형식, B 규격 종류 고르기, 시작 대조, 종류별 보정, W 추적 일관성."""
from __future__ import annotations

import copy
import json
import math
import random
import unittest
from pathlib import Path

import numpy as np

from eyetoarm.transforms.planar_transform import look_at, pixels_to_w, w_to_image
from eyetoarm.transforms.work_result import _classify, load_specs
from eyetoarm.vision.ai_config import AiSegConfig, ViewBiasConfig, ViewBiasSpecEntry, load_ai_config, parse_view_bias
from eyetoarm.vision.bolt_spec import BoltSpec, bolt_specs_from, choose_spec, length_at, select_spec
from eyetoarm.vision.view_bias import (FIT_PLANE_TOL_M, ViewBiasPlan, apply_view_bias, build_view_bias_plan, check_runtime,
                                       correct_bolt, correct_endpoints)
from eyetoarm.vision.world_tracking import BoltPlanes, WorldTrackerConfigError, world_inputs
from test_ai_config import AI_FIELDS
from test_ai_result import instance, seg_result

ROOT = Path(__file__).resolve().parents[2]
SPECS_PATH = ROOT / "config" / "calibration" / "part_specs.json"

# 합성 장면: 1280×720, 왜곡 0, 카메라 (0, 0, 0.5)에서 (0.5, 0, 0.105)를 비스듬히 내려다봄(test_view_bias와 같음)
KMAT = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]])
DIST0 = np.zeros(5)
T_OBLIQUE = look_at((0.0, 0.0, 0.5), (0.5, 0.0, 0.105))

K_A, K_B = 0.62, 1.39        # 시험용 k(mm). 값 자체는 보정 계산 검증에만 쓴다
Z_A, Z_B = 0.102, 0.105      # B-12 예상 BOLT_A(M3×20)·지금 BOLT_B(M6×40) 기준 높이


def b12_specs() -> dict:
    """저장소 B 규격 사본 + B-12 예상 BOLT_A(M3x20, 0.102 m, 23 ± 6 mm). B 파일은 바꾸지 않는다."""
    d = copy.deepcopy(load_specs(SPECS_PATH))
    d["specs"]["BOLT_A"].update(label="M3x20", plane_z=Z_A, total_length=0.023, length_tolerance=0.006)

    return d


def spec_cfg(**over) -> ViewBiasConfig:
    """종류별 형식 설정(B-12 예상 규격에 맞음). `over`로 종류 항목을 바꾼다."""
    by_spec = {"BOLT_A": {"label": "M3x20", "k_mm": K_A, "fit_plane_z_m": Z_A},
               "BOLT_B": {"label": "M6x40", "k_mm": K_B, "fit_plane_z_m": Z_B}}
    by_spec.update(over)

    return parse_view_bias({"model": "C1", "valid_for_model_id": "m@x", "by_spec": by_spec})


def bolt_at(center_xy, length_m, z_m, axis=(0.0, 1.0), T=T_OBLIQUE):
    """높이 z_m에 누운 볼트(중심·길이·축 방향) → (W 두 끝점 (2, 2) m, 픽셀 (2, 2))."""
    d = np.asarray(axis, float) / np.linalg.norm(axis)
    c = np.asarray(center_xy, float)
    pts = np.array([c - d * length_m / 2, c + d * length_m / 2])
    px = w_to_image(np.column_stack([pts, np.full(2, z_m)]), KMAT, DIST0, T)

    return pts, px


# ---------------------------------------------------------------------- 설정 형식

class SpecConfigTest(unittest.TestCase):
    """`axis_view_bias.by_spec` 형식 검사와 기존 형식 지문 보존."""

    def test_parses_and_normalizes(self):
        """종류별 형식을 받고 이름 순 튜플로 정규화하며, 기존 형식 키(k_mm·plane_z_m)는 None이다."""
        cfg = spec_cfg()
        self.assertFalse(cfg.legacy)
        self.assertEqual((cfg.k_mm, cfg.plane_z_m), (None, None))
        self.assertEqual([n for n, _ in cfg.by_spec], ["BOLT_A", "BOLT_B"])
        self.assertEqual(dict(cfg.by_spec)["BOLT_A"], ViewBiasSpecEntry("M3x20", K_A, Z_A))
        self.assertEqual(cfg.to_json()["by_spec"]["BOLT_B"], {"label": "M6x40", "k_mm": K_B, "fit_plane_z_m": Z_B})
        self.assertEqual(parse_view_bias(cfg.to_json()), cfg)

    def test_refuses_bad_entries(self):
        """k ≤ 0·비유한·bool·상한 초과, 평면 ≤ 0·비유한, 빈 라벨, 모르는·빠진 키, 빈 by_spec, 형식 섞기는 거부."""
        bad_entries = [{"label": "M3x20", "k_mm": 0.0, "fit_plane_z_m": Z_A}, {"label": "M3x20", "k_mm": -1.0, "fit_plane_z_m": Z_A},
                       {"label": "M3x20", "k_mm": float("nan"), "fit_plane_z_m": Z_A}, {"label": "M3x20", "k_mm": True, "fit_plane_z_m": Z_A},
                       {"label": "M3x20", "k_mm": 11.0, "fit_plane_z_m": Z_A}, {"label": "M3x20", "k_mm": K_A, "fit_plane_z_m": 0.0},
                       {"label": "M3x20", "k_mm": K_A, "fit_plane_z_m": float("inf")}, {"label": "", "k_mm": K_A, "fit_plane_z_m": Z_A},
                       {"label": "M3x20", "k_mm": K_A}, {"label": "M3x20", "k_mm": K_A, "fit_plane_z_m": Z_A, "extra": 1}, [1, 2]]

        for entry in bad_entries:
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                spec_cfg(BOLT_A=entry)

        with self.assertRaises(ValueError):
            parse_view_bias({"model": "C1", "valid_for_model_id": "m@x", "by_spec": {}})
        with self.assertRaises(ValueError):
            parse_view_bias({"model": "C1", "valid_for_model_id": "m@x", "by_spec": spec_cfg().to_json()["by_spec"], "k_mm": 1.0})
        with self.assertRaises(ValueError):
            ViewBiasConfig("C1", 1.0, 0.105, "m@x", by_spec=spec_cfg().by_spec)

    def test_ai_config_accepts_and_checks_model_id(self):
        """AI 설정에 넣으면 설정 ID가 기존 형식과 다르고, model_id 대조는 그대로 적용된다."""
        base = {**AI_FIELDS, "axis_extractor": "m2"}
        mid = AiSegConfig(**base).model_id
        doc = spec_cfg().to_json()
        on = AiSegConfig(**base, axis_view_bias={**doc, "valid_for_model_id": mid})
        legacy = AiSegConfig(**base, axis_view_bias={"model": "C1", "k_mm": K_B, "plane_z_m": Z_B, "valid_for_model_id": mid})

        self.assertNotEqual(on.config_sha256, legacy.config_sha256)
        self.assertEqual(AiSegConfig(**on.init_fields()).config_sha256, on.config_sha256)

        with self.assertRaisesRegex(ValueError, "valid_for_model_id"):
            AiSegConfig(**base, axis_view_bias=doc)

    def test_repo_legacy_config_id_unchanged(self):
        """운영 설정(기존 형식)의 설정 ID는 #270 전 값 그대로다(#259 기록 `6af8c944…`)."""
        cfg = load_ai_config(ROOT / "config/vision/ai_s1_08_axis_m2_viewbias.json")
        self.assertTrue(cfg.axis_view_bias.legacy)
        self.assertEqual(cfg.config_sha256, "6af8c9449eee3b297e35e6034906cf00b6704129f3a15a8b1b24c2ee505ecf3f")


# ---------------------------------------------------------------------- 종류 고르기

class SpecSelectTest(unittest.TestCase):
    """B 규격 종류 고르기: B `_classify`와 같은 비교, 길이는 종류별 높이로."""

    def test_bolt_specs_from_repo_specs(self):
        """저장소 B 규격에서 볼트 종류만 파일 순서로(너트 제외), 값은 B 파일 그대로."""
        specs = bolt_specs_from(load_specs(SPECS_PATH))
        self.assertEqual([s.name for s in specs], ["BOLT_A", "BOLT_B"])
        self.assertEqual((specs[1].label, specs[1].plane_z_m, specs[1].total_length_m, specs[1].length_tolerance_m),
                         ("M6x40", 0.105, 0.046, 0.006))

    def test_bolt_specs_from_refuses_incomplete(self):
        """볼트 종류가 없거나 총길이가 빠지면 ValueError."""
        d = b12_specs()
        no_bolt = {**d, "specs": {"NUT": d["specs"]["NUT"]}}
        with self.assertRaisesRegex(ValueError, "no bolt"):
            bolt_specs_from(no_bolt)

        d["specs"]["BOLT_A"].pop("total_length")
        with self.assertRaisesRegex(ValueError, "BOLT_A"):
            bolt_specs_from(d)

    def test_choose_boundary_and_ties(self):
        """허용 경계(차 = 허용)는 포함, 밖은 None + 가장 가까운 종류, 같은 차는 앞 종류. 2진 정확 값으로 시험."""
        specs = (BoltSpec("A", "a", 0.1, 0.03125, 0.0078125), BoltSpec("B", "b", 0.1, 0.0625, 0.0078125))
        self.assertEqual(choose_spec([0.0390625, 0.0390625], specs).name, "A")      # 차 = 허용
        out = choose_spec([0.04, 0.04], specs)
        self.assertEqual((out.name, out.nearest), (None, "A"))
        self.assertEqual(choose_spec([0.046875, 0.046875], specs).nearest, "A")    # 차가 같으면 앞 종류
        self.assertEqual(choose_spec([None, 0.0625], specs).name, "B")             # 투영 실패 종류는 건너뜀
        self.assertEqual(choose_spec([None, None], specs), type(out)(None, None, (("A", None), ("B", None))))

    def test_matches_b_classify_on_random_lengths(self):
        """같은 길이를 주면 B `_classify`와 고른 종류가 같다(B-12 예상·지금 규격, 1만 개)."""
        rng = random.Random(270)

        for d in (b12_specs(), load_specs(SPECS_PATH)):
            specs = bolt_specs_from(d)

            for _ in range(10000):
                L = rng.uniform(0.005, 0.065)
                b_spec, b_reason = _classify("bolt", L, 0.0, float("inf"), d)
                self.assertEqual(choose_spec([L] * len(specs), specs).name, None if b_reason else b_spec, L)

    def test_select_uses_each_spec_plane(self):
        """M3×20(23 mm, 0.102 m)·M6×40(46 mm, 0.105 m)는 자기 종류를 고르고, 길이는 그 종류 높이로 잰 값이다."""
        specs = bolt_specs_from(b12_specs())
        _, px_a = bolt_at((0.5, 0.0), 0.023, Z_A)
        _, px_b = bolt_at((0.45, 0.05), 0.046, Z_B, axis=(0.3, 1.0))

        ca, cb = select_spec(px_a, KMAT, DIST0, T_OBLIQUE, specs), select_spec(px_b, KMAT, DIST0, T_OBLIQUE, specs)
        self.assertEqual((ca.name, cb.name), ("BOLT_A", "BOLT_B"))
        self.assertAlmostEqual(dict(ca.lengths_m)["BOLT_A"], 0.023, places=9)
        self.assertAlmostEqual(dict(cb.lengths_m)["BOLT_B"], 0.046, places=9)
        self.assertLess(dict(ca.lengths_m)["BOLT_B"], 0.023)   # 더 높은 평면(카메라에 가까움)으로 올리면 길이가 줄어든다
        self.assertAlmostEqual(length_at(px_a, KMAT, DIST0, T_OBLIQUE, Z_A), 0.023, places=9)


# ---------------------------------------------------------------------- 시작 대조

class PlanBuildTest(unittest.TestCase):
    """`build_view_bias_plan`: 설정 ↔ B 규격 라벨·평면·종류 집합 대조."""

    def test_builds_from_matching_specs(self):
        """맞는 규격이면 종류별 높이·k를 B·설정에서 가져오고 기록에 종류별 값·지문을 남긴다."""
        plan = build_view_bias_plan(spec_cfg(), b12_specs(), part_specs_sha256="f" * 64)
        self.assertEqual((plan.plane_of("BOLT_A"), plan.k_of("BOLT_A"), plan.plane_of("BOLT_B"), plan.k_of("BOLT_B")),
                         (Z_A, K_A, Z_B, K_B))
        rec = plan.to_json()
        self.assertEqual(rec["part_specs_sha256"], "f" * 64)
        self.assertEqual(rec["specs"][0], {"name": "BOLT_A", "label": "M3x20", "plane_z_m": Z_A, "total_length_m": 0.023,
                                           "length_tolerance_m": 0.006, "k_mm": K_A, "fit_plane_z_m": Z_A})

    def test_refuses_current_repo_specs(self):
        """지금 저장소 규격(BOLT_A M6x12·0.105)에는 M3x20 설정이 붙지 않는다(라벨·평면 둘 다 사유)."""
        with self.assertRaisesRegex(ValueError, r"BOLT_A label 'M3x20' != B 'M6x12'.*fit_plane_z_m 102.0mm != B plane_z 105.0mm"):
            build_view_bias_plan(spec_cfg(), load_specs(SPECS_PATH))

    def test_fit_plane_tolerance_boundary(self):
        """k를 잰 평면 차 0.5 mm는 통과, 0.6 mm는 거부."""
        self.assertEqual(FIT_PLANE_TOL_M, 0.0005)
        ok = spec_cfg(BOLT_A={"label": "M3x20", "k_mm": K_A, "fit_plane_z_m": Z_A + 0.0005 - 1e-12})
        build_view_bias_plan(ok, b12_specs())
        bad = spec_cfg(BOLT_A={"label": "M3x20", "k_mm": K_A, "fit_plane_z_m": Z_A + 0.0006})
        with self.assertRaisesRegex(ValueError, "BOLT_A fit_plane_z_m"):
            build_view_bias_plan(bad, b12_specs())

    def test_refuses_spec_set_mismatch_and_legacy(self):
        """설정에 B 종류가 빠지거나 B에 없는 종류가 있으면 거부, 기존 형식 설정은 계획을 만들지 않는다."""
        only_b = parse_view_bias({"model": "C1", "valid_for_model_id": "m@x",
                                  "by_spec": {"BOLT_B": {"label": "M6x40", "k_mm": K_B, "fit_plane_z_m": Z_B}}})
        with self.assertRaisesRegex(ValueError, "by_spec"):
            build_view_bias_plan(only_b, b12_specs())

        extra = spec_cfg(BOLT_C={"label": "M8x50", "k_mm": 1.0, "fit_plane_z_m": 0.106})
        with self.assertRaisesRegex(ValueError, "by_spec"):
            build_view_bias_plan(extra, b12_specs())

        with self.assertRaises(ValueError):
            build_view_bias_plan(ViewBiasConfig("C1", K_B, Z_B, "m@x"), b12_specs())


# ---------------------------------------------------------------------- 종류별 보정

class CorrectBoltTest(unittest.TestCase):
    """`correct_bolt`: 종류를 고르고 그 종류 높이·k로 보정한다."""

    def setUp(self):
        """B-12 예상 규격 계획."""
        self.plan = build_view_bias_plan(spec_cfg(), b12_specs())

    def test_m3_corrected_at_its_plane_by_its_k(self):
        """M3×20: 0.102 평면에서 두 끝점이 카메라 쪽으로 k_A·|s| mm 옮겨지고, 기록에 종류·높이·k·길이가 남는다."""
        pts, px = bolt_at((0.5, 0.0), 0.023, Z_A)
        res = correct_bolt(px, KMAT, DIST0, T_OBLIQUE, self.plan, width=1280, height=720)

        self.assertTrue(res.applied)
        self.assertEqual((res.spec, res.plane_z_m, res.k_mm), ("BOLT_A", Z_A, K_A))
        self.assertAlmostEqual(abs(res.s), 1.0, places=9)
        moved = pixels_to_w(res.endpoints_px, KMAT, DIST0, T_OBLIQUE, Z_A)
        np.testing.assert_allclose(moved - pts, [[-K_A / 1000.0, 0.0]] * 2, atol=1e-9)   # 카메라는 −x 쪽
        self.assertEqual(set(res.to_record()), {"applied", "s", "delta_mm", "reason", "spec", "plane_z_m", "k_mm", "lengths_mm"})
        self.assertAlmostEqual(res.lengths_mm["BOLT_A"], 23.0, places=3)

    def test_m6_same_as_legacy_single_plane(self):
        """M6×40: 0.105 평면·k 1.39로 기존 형식(평면·k 하나) 결과와 끝점이 바이트 같다."""
        _, px = bolt_at((0.45, 0.05), 0.046, Z_B, axis=(0.3, 1.0))
        res = correct_bolt(px, KMAT, DIST0, T_OBLIQUE, self.plan, width=1280, height=720)
        old = correct_endpoints(px, KMAT, DIST0, T_OBLIQUE, plane_z_m=Z_B, k_mm=K_B, width=1280, height=720)

        self.assertEqual(res.spec, "BOLT_B")
        self.assertEqual(res.endpoints_px.tobytes(), old.endpoints_px.tobytes())
        self.assertEqual((res.applied, res.s, res.delta_mm, res.reason), (old.applied, old.s, old.delta_mm, old.reason))

    def test_out_of_range_length_is_not_corrected(self):
        """어느 길이 범위에도 들지 않으면(34 mm) 원 끝점·`no_spec`·종류별 길이 기록."""
        _, px = bolt_at((0.5, 0.0), 0.034, Z_B)
        res = correct_bolt(px, KMAT, DIST0, T_OBLIQUE, self.plan)

        self.assertEqual((res.applied, res.reason, res.spec), (False, "no_spec", None))
        np.testing.assert_array_equal(res.endpoints_px, px)
        self.assertEqual(set(res.lengths_mm), {"BOLT_A", "BOLT_B"})
        self.assertAlmostEqual(res.lengths_mm["BOLT_B"], 34.0, places=3)

    def test_no_tf_and_non_finite_before_selection(self):
        """TF 없음·비유한 끝점은 고르기 전에 그 사유(길이 기록은 빈 사전)."""
        _, px = bolt_at((0.5, 0.0), 0.023, Z_A)
        self.assertEqual(correct_bolt(px, KMAT, DIST0, None, self.plan).to_record()["reason"], "no_tf")
        bad = px.copy()
        bad[0, 0] = np.nan
        res = correct_bolt(bad, KMAT, DIST0, T_OBLIQUE, self.plan)
        self.assertEqual((res.reason, res.lengths_mm), ("non_finite", {}))

    def test_legacy_record_keys_unchanged(self):
        """기존 형식 결과의 기록 키는 #270 전 4개 그대로."""
        _, px = bolt_at((0.5, 0.0), 0.046, Z_B)
        res = correct_endpoints(px, KMAT, DIST0, T_OBLIQUE, plane_z_m=Z_B, k_mm=K_B)
        self.assertEqual(set(res.to_record()), {"applied", "s", "delta_mm", "reason"})

    def test_runtime_check(self):
        """보정 인자: None·기존 형식·계획은 통과, 종류별 형식 설정을 계획 없이 쓰면 ValueError, 다른 형은 TypeError."""
        check_runtime(None)
        check_runtime(ViewBiasConfig("C1", K_B, Z_B, "m@x"))
        check_runtime(self.plan)

        with self.assertRaisesRegex(ValueError, "build_view_bias_plan"):
            check_runtime(spec_cfg())
        with self.assertRaises(TypeError):
            check_runtime(object())


class ApplyPlanTest(unittest.TestCase):
    """봉투 경로에서 계획 사용."""

    def setUp(self):
        """320×240 합성 봉투 도구(test_observation_envelope)."""
        import test_observation_envelope as oe
        self.oe = oe

    def test_envelope_records_spec_or_no_spec(self):
        """봉투 1 + 계획: 기록에 종류별 키가 있고, 종류를 고르지 못하면 같은 봉투 객체와 `no_spec`."""
        env = self.oe.build()
        ep = env.observation.axis_endpoints_px
        L = length_at(ep, self.oe.CAM.K, self.oe.CAM.dist, env.tf.T_W_C, Z_B)
        d = b12_specs()
        # 이 봉투 볼트 길이를 BOLT_B 범위 한가운데로 두는 시험 규격(높이는 그대로)
        d["specs"]["BOLT_B"].update(total_length=L, length_tolerance=0.001)
        d["specs"]["BOLT_A"].update(total_length=L / 3, length_tolerance=0.001)
        plan = build_view_bias_plan(spec_cfg(), d)
        new, recs = apply_view_bias(env, self.oe.CAM, plan)

        self.assertEqual((recs[0]["spec"], recs[0]["plane_z_m"]), ("BOLT_B", Z_B))
        self.assertEqual(recs[0]["applied"], not np.array_equal(new.observation.axis_endpoints_px, ep))

        d["specs"]["BOLT_B"]["total_length"] = L * 2
        same, recs = apply_view_bias(env, self.oe.CAM, build_view_bias_plan(spec_cfg(), d))
        self.assertIs(same, env)
        self.assertEqual((recs[0]["applied"], recs[0]["reason"], recs[0]["spec"]), (False, "no_spec", None))

    def test_spec_config_without_plan_is_refused(self):
        """종류별 형식 설정을 그대로 넘기면 ValueError(B 규격 없이 조용히 한 평면으로 보정하지 않음)."""
        with self.assertRaisesRegex(ValueError, "build_view_bias_plan"):
            apply_view_bias(self.oe.build(), self.oe.CAM, spec_cfg())


# ---------------------------------------------------------------------- W 추적 일관성

class WorldConsistencyTest(unittest.TestCase):
    """W 추적이 시야 보정·B 입력과 같은 종류·높이를 쓴다."""

    def setUp(self):
        """B-12 예상 규격의 계획과 허용 포함 `BoltPlanes`."""
        d = b12_specs()
        self.plan = build_view_bias_plan(spec_cfg(), d)
        specs = bolt_specs_from(d)
        self.planes = BoltPlanes(default_z_m=Z_A, specs=tuple((s.name, s.plane_z_m, s.total_length_m) for s in specs),
                                 tolerances_m=tuple(s.length_tolerance_m for s in specs))

    def inputs(self, px, view_bias):
        """끝점 볼트 하나 → W 입력."""
        res = seg_result([instance(0, axis_endpoints_px=px)], crop_wh=(1280, 720))

        return world_inputs(res, K=KMAT, dist=DIST0, T_W_C=T_OBLIQUE, plane_z_m=self.planes, view_bias=view_bias,
                            width=1280, height=720)[0]

    def test_world_uses_view_bias_spec_and_plane(self):
        """M3×20: W 위치 = 보정 끝점을 0.102로 올린 값, 원 위치 = 0.102 정답, 종류 BOLT_A."""
        pts, px = bolt_at((0.5, 0.0), 0.023, Z_A)
        inp = self.inputs(px, self.plan)
        vb = correct_bolt(px, KMAT, DIST0, T_OBLIQUE, self.plan, width=1280, height=720)
        want = pixels_to_w(vb.endpoints_px, KMAT, DIST0, T_OBLIQUE, Z_A).mean(axis=0) * 1000.0

        self.assertEqual((inp.plane_spec, inp.plane_z_m), ("BOLT_A", Z_A))
        np.testing.assert_allclose(inp.center_mm, want, atol=1e-9)
        np.testing.assert_allclose(inp.center_raw_mm, pts.mean(axis=0) * 1000.0, atol=1e-6)
        self.assertAlmostEqual(math.dist(inp.center_mm, inp.center_raw_mm), K_A, places=6)

    def test_world_pick_matches_select_spec_without_bias(self):
        """보정 없이도 W 높이 고르기는 `select_spec`과 같은 종류(허용 안), 범위 밖은 가장 가까운 종류로 추적 유지."""
        for length, z, want in ((0.023, Z_A, "BOLT_A"), (0.046, Z_B, "BOLT_B"), (0.034, Z_B, None)):
            _, px = bolt_at((0.5, 0.0), length, z)
            inp = self.inputs(px, None)
            choice = select_spec(px, KMAT, DIST0, T_OBLIQUE, self.plan.specs)
            self.assertEqual(choice.name, want)
            self.assertEqual(inp.plane_spec, choice.name or choice.nearest)
            self.assertTrue(inp.candidate)

    def test_tolerances_must_match_specs(self):
        """허용 수가 종류 수와 다르거나 음수·bool이면 설정 오류."""
        rows = (("BOLT_A", Z_A, 0.023), ("BOLT_B", Z_B, 0.046))

        for tol in ((0.006,), (0.006, -0.001), (0.006, True)):
            with self.subTest(tol=tol), self.assertRaises(WorldTrackerConfigError):
                BoltPlanes(default_z_m=Z_A, specs=rows, tolerances_m=tol)


if __name__ == "__main__":
    unittest.main()
