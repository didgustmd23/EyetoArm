"""`vision.axis_endpoints.extract_m2`(#225 = #223 정책 P7)와 설정·convert 연결 단위시험.

근거: 지시 `_workspace/225_reviewer_instruction_axis_p7_impl.md` 확인 1, #223 보고 `_workspace/223_builder3_axis_failure_report.md` §3·§5.
- 실제 사례: `tests/unit/data/m2_p7_cases.json`(#219 제품 경로 다각형·M1 결과, 출처는 파일 안 `source`). 정답(G1 투영 끝점·깊이)은
  시험 판정에만 쓰고 추출기에 넣지 않는다. 오차 상한 4.5 mm는 #223 (다) 분류 기준(제품 M1 유효 최대 4.44 mm)이다.
- 합성 마스크는 numpy·OpenCV로 그린 도형이며 기대값(머리 쪽·사유)은 그린 도형에서 정한다. 같은 추출기로 역산하지 않는다.
"""
import json
import math
import unittest
from pathlib import Path

import cv2
import numpy as np

from eyetoarm.vision.ai_config import AXIS_EXTRACTORS, load_ai_config
from eyetoarm.vision.ai_convert import MULTIPLE_CANDIDATES, rasterize_polygon
from eyetoarm.vision.axis_endpoints import MODEL_DERIVED, extract_m1, extract_m2
from test_ai_axis_path import bolt_mask, run_job
from test_ai_config import ai_config

ROOT = Path(__file__).resolve().parents[2]
CASES = json.loads((Path(__file__).resolve().parent / "data" / "m2_p7_cases.json").read_text())
FX = CASES["fx_px"]
W, H = CASES["image_wh"]
MAX_ERR_MM = 4.5
M2_KEYS = ("vote_width_ratio", "head_vote", "order_gate", "centerline", "tilt_used_deg")


def case(name):
    """이름으로 실제 사례 하나."""
    return next(c for c in CASES["cases"] if c["name"] == name)


def case_mask(c):
    """사례 다각형 → 제품 래스터 규약의 원본 프레임 마스크."""
    return rasterize_polygon(np.asarray(c["polygon"], np.float64), np.zeros(2), (W, H))


def errors_mm(points, c):
    """(순서대로 최대 오차, 뒤집은 최대 오차) mm. 화소 오차 × 끝점 깊이 / fx(#223 run_lab.err와 같은 식)."""
    (h, t), (zh, zt) = np.asarray(c["gt_head_tip_uv"]), c["gt_depth_head_tip_m"]
    ordered = max(np.linalg.norm(points[0] - h) * zh, np.linalg.norm(points[1] - t) * zt) / FX * 1000
    swapped = max(np.linalg.norm(points[1] - h) * zh, np.linalg.norm(points[0] - t) * zt) / FX * 1000

    return ordered, swapped


def drawn_bolt(angle_deg, *, head_len=12, head_w=20, shaft_len=40, shaft_w=10, front_ramp=0, size=(240, 240)):
    """중심 (120, 120)에서 angle_deg 방향으로 머리 → 축 끝인 합성 볼트. (마스크, 머리 끝 점, 축 끝 점).

    front_ramp > 0이면 머리 앞면을 그 길이만큼 뾰족하게 깎는다(둥근 예측 머리 앞면 흉내, 폭 부족 면적이 커져 기울기 과대추정).
    """
    a = math.radians(angle_deg)
    u, n = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    total = head_len + shaft_len
    head_end = np.array([size[0] / 2, size[1] / 2]) - u * total / 2
    tip_end = head_end + u * total

    def p(s, t):
        """축 좌표 s, 수직 t → 화소."""
        return head_end + u * s + n * t

    hw, sw = head_w / 2, shaft_w / 2
    head = [p(0, -hw * (0.3 if front_ramp else 1)), p(front_ramp, -hw), p(head_len, -hw), p(head_len, hw), p(front_ramp, hw),
            p(0, hw * (0.3 if front_ramp else 1))]
    shaft = [p(head_len - 1, -sw), p(total, -sw), p(total, sw), p(head_len - 1, sw)]
    m = np.zeros((size[1], size[0]), np.uint8)

    for poly in (head, shaft):
        cv2.fillPoly(m, [np.rint(np.array(poly)).astype(np.int32)], 1)

    return m.astype(bool), head_end, tip_end


class RealCaseTest(unittest.TestCase):
    """#223 대표 사례: M1 실패 유형별로 M2가 순서 맞는 끝점을 낸다."""

    def assert_m2_close(self, c):
        """M2 유효, 머리·끝 뒤바뀜 없음, 순서대로 최대 오차 ≤ 4.5 mm."""
        est = extract_m2(case_mask(c))
        self.assertTrue(est.valid, est.reason)
        self.assertEqual(est.method, "M2")
        ordered, swapped = errors_mm(est.endpoints_px, c)
        self.assertLess(ordered, swapped)
        self.assertLessEqual(ordered, MAX_ERR_MM)

        return est

    def test_oblique_overestimate_capped(self):
        """M1 AXIS_VIEW_OBLIQUE(실제 α ≈ 39°를 90°로 과대추정) → M2는 기울기를 60°에서 잘라 통과한다."""
        c = case("oblique_overestimate")
        self.assertEqual((extract_m1(case_mask(c)).reason, c["m1_reason"]), ("AXIS_VIEW_OBLIQUE", "AXIS_VIEW_OBLIQUE"))
        self.assertLess(c["gt_alpha_deg"], MODEL_DERIVED.max_tilt_deg)
        est = self.assert_m2_close(c)

        self.assertTrue(est.diag["tilt_capped"])
        self.assertGreater(est.diag["tilt_deg"], MODEL_DERIVED.max_tilt_deg)
        self.assertAlmostEqual(est.diag["tilt_used_deg"], MODEL_DERIVED.max_tilt_deg, places=9)
        self.assertEqual((est.diag["centerline"], est.diag["order_gate"]), ("shaft", "none"))

    def test_short_bolt_a_without_step_uses_far_centerline(self):
        """짧은 BOLT_A(머리 계단 없음, M1 AXIS_ORDER_INCONSISTENT) → M2는 먼 쪽 절반 중심선으로 통과한다."""
        c = case("short_bolt_a_no_step")
        self.assertEqual((extract_m1(case_mask(c)).reason, c["gt_cls"]), ("AXIS_ORDER_INCONSISTENT", "BOLT_A"))
        est = self.assert_m2_close(c)

        self.assertEqual(est.diag["centerline"], "far_half")
        self.assertIn("shaft_ge_head", est.diag["order_gate"])

    def test_head_ambiguous_in_m1_resolved_by_vote(self):
        """M1 12% 띠 폭 비 < 1.2로 AXIS_HEAD_AMBIGUOUS인 볼트 → 25% 띠 폭과 질량이 같은 쪽을 가리켜 M2는 통과한다."""
        c = case("head_ambiguous_m1")
        self.assertEqual(extract_m1(case_mask(c)).reason, "AXIS_HEAD_AMBIGUOUS")
        est = self.assert_m2_close(c)

        self.assertEqual(est.diag["head_vote"], "agree")
        self.assertGreaterEqual(est.diag["vote_width_ratio"], MODEL_DERIVED.vote_min_ratio)

    def test_m1_valid_case_same_endpoints(self):
        """M1이 통과하는 사례에서 M2 끝점은 M1과 같다(투표·순서·기울기 처리가 개입하지 않음)."""
        c = case("m1_valid")
        mask = case_mask(c)
        m1, m2 = extract_m1(mask), self.assert_m2_close(c)

        self.assertTrue(m1.valid)
        np.testing.assert_array_equal(m2.endpoints_px, m1.endpoints_px)
        self.assertEqual((m2.diag["centerline"], m2.diag["tilt_capped"]), ("shaft", False))

    def test_nut_detections_rejected_by_vote(self):
        """볼트로 검출된 너트 두 건은 투표 관문(질량 반대·띠 폭 비 < 1.1)에서 AXIS_HEAD_AMBIGUOUS로 거부된다."""
        expected = {"nut_vote_disagree": "disagree", "nut_vote_ratio": "agree"}

        for name, vote in expected.items():
            with self.subTest(name=name):
                c = case(name)
                self.assertEqual(c["category"], "NUT")
                est = extract_m2(case_mask(c))
                self.assertEqual((est.valid, est.reason, est.endpoints_px), (False, "AXIS_HEAD_AMBIGUOUS", None))
                self.assertEqual(est.diag["head_vote"], vote)

                if vote == "agree":
                    self.assertLess(est.diag["vote_width_ratio"], MODEL_DERIVED.vote_min_ratio)

    def test_m1_unchanged_on_cases(self):
        """회귀: M1 사유·끝점이 #219 제품 기록과 같다(M2 추가가 M1을 바꾸지 않음)."""
        for c in CASES["cases"]:
            with self.subTest(name=c["name"]):
                est = extract_m1(case_mask(c))
                self.assertEqual(est.reason, c["m1_reason"])

                if c["m1_endpoints_px"] is None:
                    self.assertIsNone(est.endpoints_px)
                else:
                    np.testing.assert_allclose(est.endpoints_px, c["m1_endpoints_px"], rtol=0, atol=1e-9)


class SyntheticTest(unittest.TestCase):
    """합성 도형: 방향 전수·뒤바뀜·형상 관문·공통 거부 사유."""

    def test_no_swap_all_directions(self):
        """30° 간격 12방향 × (보통·짧은) 볼트에서 M2 머리 끝은 그린 머리 끝 쪽이다(뒤바뀜 0)."""
        for shape in (dict(), dict(head_len=8, shaft_len=16, head_w=16, shaft_w=10)):
            for angle in range(0, 360, 30):
                with self.subTest(shape=shape, angle=angle):
                    mask, head, tip = drawn_bolt(angle, **shape)
                    est = extract_m2(mask)
                    self.assertTrue(est.valid, est.reason)
                    e = est.endpoints_px
                    self.assertLess(np.linalg.norm(e[0] - head) + np.linalg.norm(e[1] - tip),
                                    np.linalg.norm(e[0] - tip) + np.linalg.norm(e[1] - head))

    def test_pointed_head_front_oblique_in_m1_capped_in_m2(self):
        """뾰족한 머리 앞면(폭 부족 면적 큼): M1은 AXIS_VIEW_OBLIQUE, M2는 sin 60°로 자른 보정량으로 통과한다."""
        mask, head, tip = drawn_bolt(20, front_ramp=8)
        m1, m2 = extract_m1(mask), extract_m2(mask)

        self.assertEqual(m1.reason, "AXIS_VIEW_OBLIQUE")
        self.assertTrue(m2.valid, m2.reason)
        self.assertTrue(m2.diag["tilt_capped"])
        k = m2.diag["scale_px_per_mm"]
        sin60 = math.sin(math.radians(60.0))
        self.assertAlmostEqual(m2.diag["protrusion_head_px"], 5.0 * sin60 * k, places=9)
        self.assertAlmostEqual(m2.diag["protrusion_tip_px"], 2.5 * sin60 * k, places=9)
        self.assertLess(np.linalg.norm(m2.endpoints_px[0] - head), np.linalg.norm(m2.endpoints_px[0] - tip))

    def test_headless_shapes_rejected(self):
        """머리 없는 막대·타원(배경 줄무늬·너트 옆모습 흉내)은 띠 폭 비 < 1.1로 AXIS_HEAD_AMBIGUOUS다."""
        bar = np.zeros((120, 200), bool)
        bar[50:62, 40:150] = True
        ellipse = np.zeros((120, 200), np.uint8)
        cv2.ellipse(ellipse, (100, 60), (40, 14), 15, 0, 360, 1, -1)

        for name, mask in (("bar", bar), ("ellipse", ellipse.astype(bool))):
            with self.subTest(name=name):
                est = extract_m2(mask)
                self.assertEqual((est.valid, est.reason), (False, "AXIS_HEAD_AMBIGUOUS"))
                self.assertLess(est.diag["vote_width_ratio"], MODEL_DERIVED.vote_min_ratio)

    def test_width_and_mass_disagree_rejected(self):
        """한쪽 끝에 얇고 넓은 가로대(폭은 넓지만 화소가 적음), 반대쪽 절반이 두꺼운 도형: 폭과 질량이 엇갈려 거부된다."""
        m = np.zeros((120, 220), bool)
        m[30:90, 40:43] = True       # 왼쪽 끝 가로대: 폭 60, 화소 적음
        m[54:66, 43:110] = True      # 왼쪽 반: 폭 12
        m[46:74, 110:170] = True     # 오른쪽 반: 폭 28, 화소 많음
        est = extract_m2(m)

        self.assertEqual((est.valid, est.reason), (False, "AXIS_HEAD_AMBIGUOUS"))
        self.assertEqual(est.diag["head_vote"], "disagree")
        self.assertGreater(est.diag["vote_mass_hi"], est.diag["vote_mass_lo"])

    def test_common_rejections_same_as_m1(self):
        """화소 부족·잘림·조각·구멍·퇴화 축은 M1과 같은 사유로 거부된다."""
        tiny = np.zeros((60, 60), bool)
        tiny[10:13, 10:15] = True
        clipped = bolt_mask(x0=0)
        frag = np.zeros((60, 120), bool)
        frag[20:30, 10:40] = frag[20:30, 60:100] = True
        hole = np.zeros((60, 120), bool)
        hole[10:40, 10:100] = True
        hole[20:30, 40:60] = False
        square = np.zeros((60, 60), bool)
        square[20:40, 20:40] = True
        expected = {"NO_OBJECT": tiny, "CONTOUR_CLIPPED": clipped, "MASK_FRAGMENTED": frag, "MASK_HOLE": hole,
                    "AXIS_DEGENERATE": square}

        for reason, mask in expected.items():
            with self.subTest(reason=reason):
                m1, m2 = extract_m1(mask), extract_m2(mask)
                self.assertEqual((m2.valid, m2.reason, m2.endpoints_px), (False, reason, None))
                self.assertEqual(m1.reason, reason)

    def test_bad_input_rejected(self):
        """2차원 bool 배열이 아닌 입력은 ValueError다(M1과 같은 입력 경계)."""
        for bad in (np.zeros((10, 10), np.uint8), np.zeros((4, 4, 2), bool), [[True]]):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(ValueError):
                    extract_m2(bad)

    def test_endpoints_read_only_float64(self):
        """유효 끝점은 float64 (2, 2) 읽기 전용이다(M1과 같은 출력 형식)."""
        est = extract_m2(drawn_bolt(45)[0])

        self.assertEqual((est.endpoints_px.dtype, est.endpoints_px.shape), (np.float64, (2, 2)))
        self.assertFalse(est.endpoints_px.flags.writeable)


class ConfigAndPathTest(unittest.TestCase):
    """설정 허용값·저장소 설정 파일·convert 연결."""

    def test_config_accepts_m2_rejects_unknown(self):
        """"m2"는 허용되고 설정 ID가 m1과 다르며, 목록에 없는 값은 ValueError다. 시야 보정은 추출기를 켠 설정에만 허용된다."""
        self.assertEqual(AXIS_EXTRACTORS, ("m1", "m2"))
        m2 = ai_config(axis_extractor="m2")
        self.assertIn('"axis_extractor":"m2"', m2.canonical_json())
        self.assertNotEqual(m2.config_sha256, ai_config(axis_extractor="m1").config_sha256)

        for bad in ("m3", "M2", "m0"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    ai_config(axis_extractor=bad)

        bias = {"model": "C1", "k_mm": 1.39, "plane_z_m": 0.105, "valid_for_model_id": m2.model_id}
        self.assertEqual(ai_config(axis_extractor="m2", axis_view_bias=bias).axis_view_bias.k_mm, 1.39)

        with self.assertRaises(ValueError):
            ai_config(axis_view_bias=bias)

    def test_repo_config_files(self):
        """기본 축 설정 해시는 #219 기록값 그대로이고, 새 시험 설정 두 개는 m2(시야 보정 끔·켬)다."""
        base = load_ai_config(ROOT / "config/vision/ai_s1_08_axis.json")
        self.assertEqual(base.config_sha256, "5a8e31a7f4527f866231df7f39e69c29c13dac2dc68a16eb87cb97158a516777")
        self.assertEqual(base.axis_extractor, "m1")
        m2 = load_ai_config(ROOT / "config/vision/ai_s1_08_axis_m2.json")
        m2_bias = load_ai_config(ROOT / "config/vision/ai_s1_08_axis_m2_viewbias.json")

        self.assertEqual((m2.axis_extractor, m2.axis_view_bias), ("m2", None))
        self.assertEqual((m2_bias.axis_extractor, m2_bias.axis_view_bias.k_mm), ("m2", 1.39))
        self.assertEqual(m2.init_fields() | {"axis_extractor": "m1"}, base.init_fields())

    def test_convert_uses_m2_and_records_vote(self):
        """m2 설정의 convert는 extract_m2 끝점을 쓰고 진단에 투표·중심선을 남긴다. m1 설정 진단에는 M2 키가 없다."""
        mask = bolt_mask()
        _, result, obs = run_job(ai_config(axis_extractor="m2"), [mask])
        inst = result.instances[0]

        self.assertTrue(obs.valid)
        np.testing.assert_array_equal(inst.axis_endpoints_px, extract_m2(mask).endpoints_px)
        self.assertEqual((inst.axis_diag["head_vote"], inst.axis_diag["centerline"]), ("agree", "shaft"))

        _, result_m1, _ = run_job(ai_config(axis_extractor="m1"), [mask])
        self.assertFalse(set(M2_KEYS) & set(result_m1.instances[0].axis_diag))

    def test_convert_m2_keeps_overlap_and_clip_rules(self):
        """m2 설정에서도 겹친 두 볼트는 MULTIPLE_CANDIDATES, 가장자리에 닿은 볼트는 CONTOUR_CLIPPED다(자격 규칙 무변)."""
        _, result, obs = run_job(ai_config(axis_extractor="m2"), [bolt_mask(), bolt_mask(x0=42)])
        self.assertEqual((obs.valid, obs.reason), (False, MULTIPLE_CANDIDATES))
        self.assertEqual(result.axis_valid_indices, ())

        _, result, obs = run_job(ai_config(axis_extractor="m2"), [bolt_mask(x0=0)])
        self.assertEqual((obs.valid, obs.reason, result.instances[0].axis_reason), (False, "CONTOUR_CLIPPED", "CONTOUR_CLIPPED"))


if __name__ == "__main__":
    unittest.main()
