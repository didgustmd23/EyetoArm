"""시선 기반 끝점 보정(C1) 시험: 순수 함수, 설정 검사, 공통 지점(실시간·저장 단계 T·L)의 B 입력·기록.

순수 함수는 numpy 합성 장면을 쓰고, 경로 시험은 가짜 모델·마스크·TF·목록 소비자를 쓴다(ROS·Gazebo 없음).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import unittest
from pathlib import Path

import numpy as np

from eyetoarm.calibration.profile import CameraProfile
from eyetoarm.transforms.planar_transform import look_at, pixels_to_w, w_to_image
from eyetoarm.transforms.work_result import compute_bolt_from_endpoints, load_specs
from eyetoarm.vision.ai_config import AiSegConfig, ViewBiasConfig, load_ai_config, parse_view_bias
from eyetoarm.vision.view_bias import apply_view_bias, correct_endpoints, summarize
from test_ai_config import AI_FIELDS
from test_live_stage_l import ListBase, three_bolts
from test_run_pipeline_capture_observation import Base as CaptureBase
from test_run_pipeline_capture_observation import bolt_mask
from test_run_pipeline_live_observation import Base as LiveBase
from test_run_pipeline_live_observation import Scenario
from test_stage_l_cli import FakeListConsumer as StoredListConsumer
from test_stage_l_cli import StageLBase

ROOT = Path(__file__).resolve().parents[2]
SPECS = ROOT / "config" / "calibration" / "part_specs.json"
PLANE_Z = 0.105
K_MM = 1.39

# 합성 장면: 1280×720, 왜곡 0, 카메라 (0, 0, 0.5)에서 (0.5, 0, 0.105)를 비스듬히 내려다봄
KMAT = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]])
DIST0 = np.zeros(5)
T_OBLIQUE = look_at((0.0, 0.0, 0.5), (0.5, 0.0, PLANE_Z))


def project(points_w, T=T_OBLIQUE):
    """작업면 점 (N, 2) → 원본 픽셀 (N, 2)."""
    P = np.column_stack([np.asarray(points_w, float), np.full(len(points_w), PLANE_Z)])

    return w_to_image(P, KMAT, DIST0, T)


def model_id_of(path) -> str:
    """가짜 가중치 파일의 model_id(설정 규칙 `family@sha256[:16]`)."""
    return f"{AI_FIELDS['model_family']}@{hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]}"


def write_bias_config(src_path, dst_path, **over) -> str:
    """기존 AI 설정 파일에 `axis_view_bias`를 더한 새 파일."""
    doc = json.loads(Path(src_path).read_text(encoding="utf-8"))
    doc["axis_view_bias"] = {"model": "C1", "k_mm": K_MM, "plane_z_m": PLANE_Z,
                             "valid_for_model_id": model_id_of(doc["weights_path"]), **over}
    Path(dst_path).write_text(json.dumps(doc), encoding="utf-8")

    return str(dst_path)


# ---------------------------------------------------------------------- 순수 함수

class CorrectEndpointsTest(unittest.TestCase):
    """C1 계산: 부호·크기·불변량·끔 사유."""

    def bolt(self, axis_dir, center=(0.5, 0.0), length=0.046):
        """작업면 볼트 두 끝점(머리, 축 끝)과 그 픽셀."""
        d = np.asarray(axis_dir, float) / np.linalg.norm(axis_dir)
        c = np.asarray(center, float)
        pts = np.array([c - d * length / 2, c + d * length / 2])

        return pts, project(pts)

    def correct(self, px, T=T_OBLIQUE, **kw):
        """기본 인자로 보정."""
        return correct_endpoints(px, KMAT, DIST0, T, plane_z_m=PLANE_Z, k_mm=K_MM, **kw)

    def test_moves_toward_camera_by_k_when_axis_perpendicular_to_view(self):
        """축이 시선에 수직(s = ±1)이면 두 끝점이 작업면에서 카메라 쪽으로 정확히 k mm 옮겨진다."""
        pts, px = self.bolt((0.0, 1.0))
        res = self.correct(px)
        self.assertTrue(res.applied)
        self.assertAlmostEqual(abs(res.s), 1.0, places=9)

        moved = pixels_to_w(res.endpoints_px, KMAT, DIST0, T_OBLIQUE, PLANE_Z)
        shift = moved - pts
        np.testing.assert_allclose(shift[:, 0], -K_MM / 1000.0, atol=1e-9)   # 카메라는 −x 쪽
        np.testing.assert_allclose(shift[:, 1], 0.0, atol=1e-9)

    def test_direction_does_not_depend_on_endpoint_order(self):
        """머리·축 끝 순서를 바꿔도(n 부호 반전) 이동은 같은 방향·같은 크기다."""
        pts, px = self.bolt((0.3, 1.0))
        a = self.correct(px)
        b = self.correct(px[::-1])
        np.testing.assert_allclose(a.endpoints_px, b.endpoints_px[::-1], atol=1e-9)
        self.assertAlmostEqual(a.s, -b.s, places=12)
        self.assertAlmostEqual(a.delta_mm, -b.delta_mm, places=12)

        moved = pixels_to_w(a.endpoints_px, KMAT, DIST0, T_OBLIQUE, PLANE_Z).mean(axis=0)
        cam = T_OBLIQUE[:2, 3]
        self.assertLess(np.linalg.norm(moved - cam), np.linalg.norm(pts.mean(axis=0) - cam))

    def test_axis_along_view_is_unchanged(self):
        """축이 시선과 나란(s = 0)하면 끝점이 바뀌지 않는다(왕복 수치 오차 이내)."""
        _, px = self.bolt((1.0, 0.0))
        res = self.correct(px)
        self.assertTrue(res.applied)
        self.assertAlmostEqual(res.s, 0.0, places=12)
        np.testing.assert_allclose(res.endpoints_px, px, atol=1e-6)

    def test_length_and_theta_unchanged_in_b_computation(self):
        """B `compute_bolt_from_endpoints`로 계산한 길이·θ가 보정 전후 같고, 중심만 k·|s| 옮겨진다."""
        cam = CameraProfile(width=1280, height=720, K=KMAT, dist=DIST0, frame_id="cam")
        specs = load_specs(SPECS)
        _, px = self.bolt((0.4, 1.0))
        res = self.correct(px)
        before = compute_bolt_from_endpoints(px[0], px[1], cam, T_OBLIQUE, specs)
        after = compute_bolt_from_endpoints(res.endpoints_px[0], res.endpoints_px[1], cam, T_OBLIQUE, specs)
        self.assertAlmostEqual(before.measured_length_m, after.measured_length_m, places=9)
        self.assertAlmostEqual(before.theta_deg, after.theta_deg, places=6)
        moved = math.dist(before.center_w_m, after.center_w_m) * 1000
        self.assertAlmostEqual(moved, K_MM * abs(res.s), places=6)

    def test_no_tf_keeps_raw_with_reason(self):
        """TF가 없으면 보정하지 않고 원 끝점·사유 `no_tf`."""
        _, px = self.bolt((0.0, 1.0))
        res = self.correct(px, T=None)
        self.assertFalse(res.applied)
        self.assertEqual(res.reason, "no_tf")
        np.testing.assert_array_equal(res.endpoints_px, px)

    def test_degenerate_axis_and_camera_above(self):
        """두 끝점이 같으면 `degenerate_axis`, 카메라가 바로 위면 `camera_above_point`(보정 없음)."""
        _, px = self.bolt((0.0, 1.0))
        res = self.correct(np.array([px[0], px[0]]))
        self.assertEqual((res.applied, res.reason), (False, "degenerate_axis"))

        T_down = look_at((0.5, 0.0, 0.5), (0.5, 0.0001, PLANE_Z))
        T_down[0, 3], T_down[1, 3] = 0.5, 0.0  # 카메라 수평 위치 = 볼트 중점
        pts = np.array([[0.5, -0.02], [0.5, 0.02]])
        res = correct_endpoints(project(pts, T_down), KMAT, DIST0, T_down, plane_z_m=PLANE_Z, k_mm=K_MM)
        self.assertEqual((res.applied, res.reason), (False, "camera_above_point"))

    def test_non_finite_and_projection_failure(self):
        """비유한 입력은 `non_finite`, 지평선 위 픽셀(평면과 만나지 않음)은 `projection_failed`."""
        _, px = self.bolt((0.0, 1.0))
        bad = px.copy()
        bad[0, 0] = np.nan
        self.assertEqual(self.correct(bad).reason, "non_finite")

        sky = np.array([[600.0, 5.0], [700.0, 5.0]])     # 위쪽 끝: 시선이 평면 위로 지나감
        T_level = look_at((0.0, 0.0, 0.2), (1.0, 0.0, 0.2))   # 수평 시선(지평선 = 영상 가운데)
        res = correct_endpoints(sky, KMAT, DIST0, T_level, plane_z_m=PLANE_Z, k_mm=K_MM)
        self.assertEqual((res.applied, res.reason), (False, "projection_failed"))

    def test_out_of_frame_result_is_not_applied(self):
        """보정 결과가 프레임 밖이면 `out_of_frame`(원 끝점 유지, 무효로 바꾸지 않음)."""
        _, px = self.bolt((0.0, 1.0))
        res = self.correct(px, width=1280, height=int(px[:, 1].max()) + 1)
        self.assertEqual((res.applied, res.reason), (False, "out_of_frame"))
        np.testing.assert_array_equal(res.endpoints_px, px)

    def test_input_array_not_modified(self):
        """입력 배열은 바뀌지 않는다."""
        _, px = self.bolt((0.0, 1.0))
        keep = px.copy()
        self.correct(px)
        np.testing.assert_array_equal(px, keep)

    def test_none_config_returns_same_envelope(self):
        """설정이 None이면 같은 객체와 빈 기록(기존 동작 그대로)."""
        sentinel = object()
        out, records = apply_view_bias(sentinel, None, None)
        self.assertIs(out, sentinel)
        self.assertEqual(records, [])

    def test_unknown_envelope_type_is_refused(self):
        """모르는 봉투 형은 TypeError(새 B 경로가 조용히 빠지지 않게)."""
        cfg = ViewBiasConfig("C1", K_MM, PLANE_Z, "m@x")
        with self.assertRaises(TypeError):
            apply_view_bias(object(), None, cfg)

    def test_summarize_counts(self):
        """요약: 적용 수와 사유별 건너뜀 수."""
        recs = [[{"applied": True, "reason": None}], [{"applied": False, "reason": "no_tf"}, {"applied": False, "reason": "no_tf"}]]
        self.assertEqual(summarize(recs), {"applied": 1, "skipped_by_reason": {"no_tf": 2}})


class EnvelopeImmutabilityTest(unittest.TestCase):
    """공통 지점: 원 봉투·관측은 그대로, 새 봉투는 끝점만 다르고 세대·식별자·상관 키·TF가 같다."""

    def setUp(self):
        """봉투 시험 도구(320×240 합성 카메라, 비스듬한 TF)."""
        import test_observation_envelope as oe
        self.oe = oe
        self.cfg = ViewBiasConfig("C1", K_MM, PLANE_Z, "m@x")

    def test_single_envelope_new_object_same_keys(self):
        """봉투 1: 새 봉투의 관측 끝점만 바뀌고 원 봉투·원 관측 끝점 배열은 그대로다."""
        env = self.oe.build()
        keep = env.observation.axis_endpoints_px.copy()
        new, recs = apply_view_bias(env, self.oe.CAM, self.cfg)

        self.assertIsNot(new, env)
        self.assertTrue(recs[0]["applied"])
        np.testing.assert_array_equal(env.observation.axis_endpoints_px, keep)
        self.assertEqual(recs[0]["axis_endpoints_px_raw"], keep.tolist())
        self.assertFalse(np.array_equal(new.observation.axis_endpoints_px, keep))

        for name in ("generation", "job_seq", "image_seq", "run_id", "capture_generation", "detector_config_id",
                     "valid", "reason", "selected_index"):
            self.assertEqual(getattr(new.observation, name), getattr(env.observation, name), name)

        self.assertEqual(dict(new.correlation), dict(env.correlation))
        self.assertIs(new.frame, env.frame)
        self.assertIs(new.tf, env.tf)

    def test_single_envelope_without_tf_is_same_object(self):
        """봉투 1 + TF 없음: 같은 봉투 객체를 돌려주고 사유 `no_tf`를 남긴다."""
        env = self.oe.build(tf=None)
        new, recs = apply_view_bias(env, self.oe.CAM, self.cfg)
        self.assertIs(new, env)
        self.assertEqual((recs[0]["applied"], recs[0]["reason"]), (False, "no_tf"))

    def test_list_envelope_new_object_same_keys(self):
        """봉투 2: 볼트 후보 끝점만 바뀌고 원 목록·후보와 끝점 없는 후보(너트)는 그대로다."""
        from eyetoarm.common.types import FRAME_OBSERVATIONS_VERSION, FrameObservations, ObjectObservation, PixelBasis
        from eyetoarm.vision.list_envelope import make_list_envelope

        oe = self.oe
        bolt = ObjectObservation(instance_index=0, class_name="bolt", confidence=0.9, pixel_basis=PixelBasis.RAW,
                                 observed_face="axis", axis_endpoints_px=((100.0, 120.0), (140.0, 120.0)), calc_contour=None)
        nut = ObjectObservation(instance_index=2, class_name="nut", confidence=0.8, pixel_basis=PixelBasis.RAW,
                                observed_face="top_face", axis_endpoints_px=None,
                                calc_contour=np.array([[10, 10], [20, 10], [20, 20], [10, 20]], np.int32))
        image = np.zeros((oe.H, oe.W, 3), np.uint8)
        image.setflags(write=False)
        fo = FrameObservations(version=FRAME_OBSERVATIONS_VERSION, frame=oe.make_frame(image=image), capture=oe.capture(), camera=oe.CAM,
                               tf=oe.tf(), generation=4, job_seq=7, detector_config_id="a" * 64, candidates=(bolt, nut))
        env = make_list_envelope(fo)
        keep = bolt.axis_endpoints_px.copy()
        new, recs = apply_view_bias(env, None, self.cfg)

        self.assertIsNot(new, env)
        self.assertEqual([r["instance_index"] for r in recs], [0])
        np.testing.assert_array_equal(env.observations.candidates[0].axis_endpoints_px, keep)
        self.assertFalse(np.array_equal(new.observations.candidates[0].axis_endpoints_px, keep))
        self.assertIs(new.observations.candidates[1], nut)
        self.assertEqual((new.observations.generation, new.observations.job_seq), (4, 7))
        self.assertEqual(dict(new.correlation), dict(env.correlation))
        self.assertEqual(new.version, env.version)
        self.assertEqual(new.to_record()["tf"], env.to_record()["tf"])
        np.testing.assert_array_equal(new.observations.tf.T_W_C, env.observations.tf.T_W_C)


# ---------------------------------------------------------------------- 설정

class ViewBiasConfigTest(unittest.TestCase):
    """AI 설정 `axis_view_bias` 검사."""

    def fields(self, **bias_over):
        """M1 켠 AI 필드 + 보정 키."""
        base = {**AI_FIELDS, "axis_extractor": "m1"}
        mid = AiSegConfig(**base).model_id
        base["axis_view_bias"] = {"model": "C1", "k_mm": K_MM, "plane_z_m": PLANE_Z, "valid_for_model_id": mid, **bias_over}

        return base

    def test_accepts_and_changes_config_id_only_when_present(self):
        """키가 있으면 받고 설정 ID가 달라지며, 키가 없으면 기존 설정 ID 그대로다."""
        off = AiSegConfig(**{**AI_FIELDS, "axis_extractor": "m1"})
        on = AiSegConfig(**self.fields())
        self.assertIsNone(off.axis_view_bias)
        self.assertIsInstance(on.axis_view_bias, ViewBiasConfig)
        self.assertNotEqual(off.config_sha256, on.config_sha256)
        self.assertNotIn("axis_view_bias", off.canonical_json())
        self.assertEqual(AiSegConfig(**on.init_fields()).config_sha256, on.config_sha256)

    def test_refuses_model_id_mismatch(self):
        """`valid_for_model_id`가 설정의 model_id와 다르면 거부."""
        with self.assertRaisesRegex(ValueError, "valid_for_model_id"):
            AiSegConfig(**self.fields(valid_for_model_id="yolo11n-seg@0000000000000000"))

    def test_refuses_without_m1(self):
        """축 끝점 추출기가 m1이 아니면 거부."""
        doc = self.fields()
        doc.pop("axis_extractor")
        with self.assertRaisesRegex(ValueError, "axis_extractor"):
            AiSegConfig(**doc)

    def test_refuses_bad_values(self):
        """모형 C1 아님, k ≤ 0·비유한·bool·상한 초과, 비유한 높이, 모르는 키·빠진 키는 거부."""
        for over in ({"model": "C2"}, {"k_mm": 0.0}, {"k_mm": -1.0}, {"k_mm": float("nan")}, {"k_mm": True},
                     {"k_mm": 11.0}, {"plane_z_m": float("inf")}, {"extra": 1}):
            with self.subTest(over=over), self.assertRaises(ValueError):
                AiSegConfig(**self.fields(**over))

        with self.assertRaises(ValueError):
            parse_view_bias({"model": "C1", "k_mm": 1.0, "plane_z_m": 0.1})
        with self.assertRaises(ValueError):
            parse_view_bias([1, 2])

    def test_repo_config_file_matches_s1_08(self):
        """`ai_s1_08_axis_viewbias.json`은 `ai_s1_08_axis.json` + 보정 키이며 S1-08 model_id에 묶인다."""
        base = load_ai_config(ROOT / "config/vision/ai_s1_08_axis.json")
        bias = load_ai_config(ROOT / "config/vision/ai_s1_08_axis_viewbias.json")
        strip = lambda cfg: {k: v for k, v in cfg.init_fields().items() if k != "axis_view_bias"}

        self.assertIsNone(base.axis_view_bias)
        self.assertEqual(strip(bias), strip(base))
        self.assertNotEqual(bias.config_sha256, base.config_sha256)
        self.assertEqual(bias.axis_view_bias.valid_for_model_id, base.model_id)
        self.assertEqual(bias.axis_view_bias.k_mm, K_MM)


# ---------------------------------------------------------------------- 저장 경로 T·L

class StoredStageTTest(CaptureBase):
    """저장 녹화 + TF 메타 단계 T: B가 보정 끝점을 받고 기록에 원값·보정값이 남는다."""

    def setUp(self):
        """B 입력 관측 감시."""
        super().setUp()
        self.b_obs = []
        import eyetoarm.transforms.b_stage as bs
        spy_prev = bs.BStage.consume

        def consume(stage, frame, obs):
            """B에 들어온 관측 보관."""
            self.b_obs.append(obs)
            return spy_prev(stage, frame, obs)

        from unittest import mock
        p = mock.patch.object(bs.BStage, "consume", consume)
        p.start()
        self.addCleanup(p.stop)

    def test_b_receives_corrected_endpoints(self):
        """켬: B 관측 끝점 = 기록의 보정 끝점 ≠ A 검출 기록(원) = `axis_endpoints_px_raw`, applied 1."""
        c = self.case()
        ai = write_bias_config(c.ai, Path(self.tmp.name) / "ai_vb.json")
        code, err = self.cli(c, c.write_meta(c.meta()), ai=ai)
        self.assertEqual(code, 0, err)

        rows = [r for r in c.b_outputs() if r["kind"] == "b_output"]
        self.assertEqual(len(rows), 1)
        inp = rows[0]["input"]
        raw = [f for f in c.frames() if f.get("axis_endpoints_px")][0]["axis_endpoints_px"]
        self.assertEqual(inp["axis_endpoints_px_raw"], raw)
        self.assertTrue(inp["view_bias"]["applied"])
        self.assertNotEqual(inp["axis_endpoints_px"], raw)
        np.testing.assert_array_equal(self.b_obs[-1].axis_endpoints_px, np.array(inp["axis_endpoints_px"]))
        self.assertEqual(c.summary()["view_bias"]["applied"], 1)

        # A 검출 기록(run.jsonl)은 원 끝점 그대로
        proc = [p for p in c.processes() if p.get("ai_instances")]
        self.assertEqual(proc[0]["ai_instances"][0]["axis_endpoints_px"], raw)

    def test_off_keeps_record_keys(self):
        """끔(키 없음): b_output.input에 보정 키가 없고 B 끝점 = 원 끝점, 요약에 view_bias 없음."""
        c = self.case()
        code, err = self.cli(c, c.write_meta(c.meta()))
        self.assertEqual(code, 0, err)
        inp = [r for r in c.b_outputs() if r["kind"] == "b_output"][0]["input"]
        self.assertNotIn("view_bias", inp)
        self.assertNotIn("axis_endpoints_px_raw", inp)
        self.assertNotIn("view_bias", c.summary())
        raw = [f for f in c.frames() if f.get("axis_endpoints_px")][0]["axis_endpoints_px"]
        self.assertEqual(inp["axis_endpoints_px"], raw)

    def test_missing_tf_is_recorded_not_silent(self):
        """켬 + 그 프레임 TF 없음(`tf: null`): B는 원 끝점을 받고 기록에 `no_tf` 사유가 남는다."""
        c = self.case()
        meta = c.meta()
        meta["frames"][0]["tf"] = None
        ai = write_bias_config(c.ai, Path(self.tmp.name) / "ai_vb.json")
        code, err = self.cli(c, c.write_meta(meta), ai=ai)
        self.assertIn(code, (0, 1), err)
        rows = [r for r in c.b_outputs() if r["kind"] == "b_output"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["input"]["view_bias"], {"applied": False, "s": None, "delta_mm": None, "reason": "no_tf"})
        self.assertEqual(c.summary()["view_bias"]["skipped_by_reason"], {"no_tf": 1})

    def test_plane_z_mismatch_refused_at_start(self):
        """켬 + 보정 높이가 B 볼트 높이와 5 mm 넘게 다르면 시작 때 종료 2(B 호출 0)."""
        c = self.case()
        ai = write_bias_config(c.ai, Path(self.tmp.name) / "ai_vb.json", plane_z_m=PLANE_Z + 0.0051)
        code, err = self.cli(c, c.write_meta(c.meta()), ai=ai)
        self.assertEqual(code, 2, err)
        self.assertIn("plane_z", err)
        self.assertIn("by more than 5 mm", err)
        self.assertEqual(self.b_obs, [])

    def test_plane_z_within_5mm_runs_and_prints_heights(self):
        """켬 + 보정 높이가 B 볼트 높이와 5 mm 안이면 시작하고, 시작 줄에 종류별 높이·차를 찍는다."""
        c = self.case()
        ai = write_bias_config(c.ai, Path(self.tmp.name) / "ai_vb.json", plane_z_m=PLANE_Z + 0.0007)
        code, err = self.cli(c, c.write_meta(c.meta()), ai=ai)
        self.assertIn(code, (0, 1), err)
        self.assertIn("run_pipeline: axis_view_bias plane_z=105.7mm; B bolt heights bolt=105.0mm(-0.7) "
                      "BOLT_A=105.0mm(-0.7) BOLT_B=105.0mm(-0.7) within ±5 mm", err)

    def test_model_id_mismatch_refused_at_start(self):
        """켬 + model_id 불일치는 설정 오류 종료 2."""
        c = self.case()
        ai = write_bias_config(c.ai, Path(self.tmp.name) / "ai_vb.json", valid_for_model_id="yolo11n-seg@ffffffffffffffff")
        code, err = self.cli(c, c.write_meta(c.meta()), ai=ai)
        self.assertEqual(code, 2, err)
        self.assertIn("valid_for_model_id", err)

    def test_plane_check_reads_given_specs(self):
        """시작 대조: B 볼트 높이가 보정 높이와 5 mm 안이면 높이를 돌려주고, 넘으면 ValueError."""
        good = ViewBiasConfig("C1", K_MM, PLANE_Z, "m@x")
        self.assertEqual(self.mod._check_view_bias_plane(good), {"bolt": PLANE_Z, "BOLT_A": PLANE_Z, "BOLT_B": PLANE_Z})
        specs = json.loads(SPECS.read_text(encoding="utf-8"))
        p = Path(self.tmp.name) / "specs.json"

        # BOLT_A 102 mm·BOLT_B 105 mm(차 3 mm) → 통과, 종류별 높이를 그대로 돌려준다
        split = copy.deepcopy(specs)
        split["specs"]["BOLT_A"]["plane_z"] = 0.102
        p.write_text(json.dumps(split), encoding="utf-8")
        self.assertEqual(self.mod._check_view_bias_plane(good, p), {"bolt": 0.102, "BOLT_A": 0.102, "BOLT_B": PLANE_Z})

        # 경계: 5 mm 바로 안(4.999 mm)은 통과, 5 mm를 넘으면(5.1 mm) 그 종류 이름과 함께 거부
        edge = copy.deepcopy(specs)
        edge["specs"]["BOLT_B"]["plane_z"] = PLANE_Z + 0.004999
        p.write_text(json.dumps(edge), encoding="utf-8")
        self.mod._check_view_bias_plane(good, p)
        bad = copy.deepcopy(specs)
        bad["specs"]["BOLT_B"]["plane_z"] = PLANE_Z + 0.0051
        p.write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "BOLT_B"):
            self.mod._check_view_bias_plane(good, p)


class StoredStageLTest(StageLBase):
    """저장 녹화 + TF 메타 단계 L: 목록 소비자가 보정 끝점을 받는다."""

    def test_list_consumer_receives_corrected_endpoints(self):
        """켬: 소비자 후보 끝점 = 원 끝점 + 보정, 기록 `view_bias`에 후보별 결과·원값, applied 2."""
        c = self.case(1)
        c.ai = write_bias_config(c.ai, Path(self.tmp.name) / "ai_vb.json")
        fake = StoredListConsumer()
        code, err = self.cli_list(c, c.write_meta(c.meta()), masks=lambda: [bolt_mask(y=60), bolt_mask(y=180)], consumer=fake)
        self.assertEqual(code, 0, err)

        env = fake.envelopes[0]
        line = [r for r in self.lines(c, "list_frames.jsonl") if r["kind"] == "b_list_input"][0]
        vb = {r["instance_index"]: r for r in line["view_bias"]}
        self.assertEqual(sorted(vb), [cnd.instance_index for cnd in env.observations.candidates])

        for cnd in env.observations.candidates:
            r = vb[cnd.instance_index]
            self.assertTrue(r["applied"])
            self.assertNotEqual(cnd.axis_endpoints_px.tolist(), r["axis_endpoints_px_raw"])

        recorded = {x["instance_index"]: x["axis_endpoints_px"] for x in line["b_input"]["candidates"]}
        self.assertEqual(recorded, {cnd.instance_index: cnd.axis_endpoints_px.tolist() for cnd in env.observations.candidates})
        self.assertEqual(json.loads((Path(c.out) / "summary.json").read_text())["view_bias"]["applied"], 2)

    def test_off_has_no_view_bias_key(self):
        """끔: `b_list_input`·요약에 보정 키가 없다."""
        c = self.case(1)
        fake = StoredListConsumer()
        code, err = self.cli_list(c, c.write_meta(c.meta()), masks=lambda: [bolt_mask(y=60), bolt_mask(y=180)], consumer=fake)
        self.assertEqual(code, 0, err)
        line = [r for r in self.lines(c, "list_frames.jsonl") if r["kind"] == "b_list_input"][0]
        self.assertNotIn("view_bias", line)
        self.assertNotIn("view_bias", json.loads((Path(c.out) / "summary.json").read_text()))


# ---------------------------------------------------------------------- 실시간 경로 T·L

class LiveStageTTest(LiveBase):
    """실시간 단계 T: 디스패처 공통 지점에서 보정한 끝점이 B로 간다."""

    def setUp(self):
        """B 입력 관측 감시(기반 감시 위에 한 겹)."""
        super().setUp()
        import eyetoarm.transforms.b_stage as bs
        from unittest import mock
        self.b_obs = []
        prev = bs.BStage.consume

        def consume(stage, frame, obs):
            """B에 들어온 관측 보관."""
            self.b_obs.append(obs)
            return prev(stage, frame, obs)

        p = mock.patch.object(bs.BStage, "consume", consume)
        p.start()
        self.addCleanup(p.stop)

    def test_live_t_corrected_and_recorded(self):
        """켬: accept 기록의 보정 끝점 = B 관측, `axis_endpoints_px_raw`·`view_bias` 있음, enabled."""
        self.case.ai = write_bias_config(self.case.ai, Path(self.tmp.name) / "ai_vb.json")
        code, err, out = self.run_cli(Scenario(3))
        self.assertEqual(code, 0, err)

        acc = [r for r in self.records(out) if r["kind"] == "accept"]
        self.assertTrue(acc)

        for r in acc:
            inp = r["input"]
            self.assertTrue(inp["view_bias"]["applied"], inp)
            self.assertNotEqual(inp["axis_endpoints_px"], inp["axis_endpoints_px_raw"])
            self.assertEqual(r["b_input"]["version"], "eyetoarm.b_input/1")

        b_eps = [o.axis_endpoints_px.tolist() for o in self.b_obs]
        self.assertTrue(all(r["input"]["axis_endpoints_px"] in b_eps for r in acc))
        s = self.summary(out)["view_bias"]
        self.assertEqual((s["enabled"], s["k_mm"]), (True, K_MM))
        self.assertGreaterEqual(s["applied"], len(acc))

    def test_live_t_off_keeps_input_keys(self):
        """끔: accept `input` 키가 기존과 같고 요약에 view_bias가 없다."""
        code, err, out = self.run_cli(Scenario(3))
        self.assertEqual(code, 0, err)
        acc = [r for r in self.records(out) if r["kind"] == "accept"]
        self.assertTrue(acc)

        for r in acc:
            self.assertEqual(sorted(r["input"]), sorted(["run_id", "image_seq", "capture_generation", "ros_stamp_ns",
                                                         "calibration_id", "job_seq", "generation", "axis_endpoints_px"]))

        self.assertNotIn("view_bias", self.summary(out))


class LiveStageLTest(ListBase):
    """실시간 단계 L: 같은 공통 지점에서 목록 후보 끝점이 보정된다."""

    def test_live_l_corrected_and_recorded(self):
        """켬: 가짜 목록 소비자 봉투의 후보 끝점이 accept 기록 `view_bias`의 원값과 다르고 B 입력 기록과 같다."""
        self.case.ai = write_bias_config(self.case.ai, Path(self.tmp.name) / "ai_vb.json")
        code, err, out = self.run_list(Scenario(3, masks={i: three_bolts for i in range(3)}))
        self.assertEqual(code, 0, err)
        self.assertTrue(self.consumer.envelopes)

        acc = self.accepts(out)
        self.assertTrue(acc)

        for r in acc:
            vb = {x["instance_index"]: x for x in r["view_bias"]}
            cands = {x["instance_index"]: x["axis_endpoints_px"] for x in r["b_input"]["candidates"]}
            self.assertEqual(sorted(vb), sorted(cands))

            for idx, eps in cands.items():
                self.assertTrue(vb[idx]["applied"])
                self.assertNotEqual(eps, vb[idx]["axis_endpoints_px_raw"])

        sent = [[c.axis_endpoints_px.tolist() for c in e.observations.candidates] for e in self.consumer.envelopes]
        rec = [[x["axis_endpoints_px"] for x in r["b_input"]["candidates"]] for r in acc]
        self.assertTrue(all(x in sent for x in rec))

    def test_live_l_off_has_no_view_bias(self):
        """끔: accept 기록 키 목록이 기존 단계 L 키와 같다."""
        code, err, out = self.run_list(Scenario(3, masks={i: three_bolts for i in range(3)}))
        self.assertEqual(code, 0, err)

        for r in self.accepts(out):
            self.assertNotIn("view_bias", r)


if __name__ == "__main__":
    unittest.main()
