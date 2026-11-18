"""AI 볼트 축 끝점 opt-in 경로(#168) 단위시험: 설정 키·convert 안 M1 호출·프레임 판정 사유·기록·B 호출 조건.

근거: `_workspace/168_a_to_b_real_path_implementation_spec.md` §1·§2, #163 명세 r1 §3~§5(사유 표 순서·마스크 수명),
#167 계획 r2 K0(opt-in의 valid는 추출기 형식 조건 통과 뜻).

- 마스크는 numpy로 그린 합성 볼트(머리 18×10 px + 축 10×34 px)이며 정답을 같은 추출기로 역산하지 않는다.
  기대 끝점은 그린 도형의 머리 끝·축 끝 화소 중심(축선 위)이다.
- 이 시험의 「유효」는 추출기 형식 조건 통과이고 정확도 판정이 아니다.
"""
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from eyetoarm.vision import ai_convert
from eyetoarm.vision.ai_config import AiSegConfig, load_ai_config
from eyetoarm.vision.ai_convert import (AXIS_EXTRACTION_ERROR, MULTIPLE_CANDIDATES, MULTIPLE_OBJECTS, convert,
                                        observation_from_ai)
from eyetoarm.vision.pipeline import VisionPipeline
from eyetoarm.vision.pipeline_records import _instance_summary
from eyetoarm.vision.pipeline_runner import b_admit
from test_ai_config import AI_FIELDS, ai_config
from test_ai_convert import raw
from test_pipeline import RUN_ID, FakeClock, frame, profile, settings

ROOT = Path(__file__).resolve().parents[2]
W, H = 200, 120
CAL = "cal-axis-test"


def bolt_mask(x0=40, y=60, head_len=10, head_w=18, shaft_len=34, shaft_w=10, flip=False):
    """가로 볼트 합성 마스크(crop = 프레임 크기). flip이면 머리가 오른쪽."""
    m = np.zeros((H, W), bool)
    m[y - head_w // 2:y - head_w // 2 + head_w, x0:x0 + head_len] = True
    m[y - shaft_w // 2:y - shaft_w // 2 + shaft_w, x0 + head_len:x0 + head_len + shaft_len] = True

    return m[:, ::-1].copy() if flip else m


def axis_cfg(**over):
    """끝점 추출을 켠 AI 설정."""
    return ai_config(axis_extractor="m1", **over)


def run_job(cfg, masks, cls=None, *, prof=True, roi_frame=None):
    """프레임 하나로 작업을 시작해 (job, result, observation)을 돌려준다."""
    fr = roi_frame or frame(0, image=np.zeros((H, W, 3), np.uint8), width=W, height=H, calibration_id=CAL)
    prof_obj = profile(width=W, height=H, calibration_id=CAL) if prof else None
    p = VisionPipeline(settings(detector_config=cfg, profile=prof_obj), run_id=RUN_ID, clock_ns=FakeClock())
    p.offer_frame(fr)
    job = p.start_next()
    crop_wh = (W, H) if cfg.roi is None else cfg.roi[2:]
    result = convert(raw(masks, crop_wh, cls=cls), job.frame, cfg)

    return job, result, observation_from_ai(job, result)


class AxisConfigTest(unittest.TestCase):
    """설정 키: 선택·기본 None·해시 보존·값 검사."""

    def test_optional_key_keeps_existing_hash(self):
        """axis_extractor가 없는(=None) 설정은 키를 생략한 설정과 해시가 같고, 저장소 AI 설정 파일의 해시도 변경 전 기록값과 같다."""
        self.assertEqual(ai_config().config_sha256, AiSegConfig(**AI_FIELDS).config_sha256)
        self.assertNotIn("axis_extractor", ai_config().canonical_json())
        # 변경 전 기록값: _workspace/168_evidence/ai_config_sha_before.txt
        self.assertEqual(load_ai_config(ROOT / "config/vision/ai_s1_01_live.json").config_sha256,
                         "e4f57ffa68a4159999087d0b5f5b725830017007850bba0c4f5b8b96eb7bf355")

    def test_enabled_key_changes_id_and_bad_value_rejected(self):
        """"m1"을 켜면 설정 ID가 달라지고, 다른 값은 ValueError다."""
        self.assertNotEqual(axis_cfg().config_sha256, ai_config().config_sha256)
        self.assertIn('"axis_extractor":"m1"', axis_cfg().canonical_json())

        for bad in ("m0", "", 1, True):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    ai_config(axis_extractor=bad)


class AxisOffTest(unittest.TestCase):
    """기본 꺼짐: 기존 표시용 무효 관측·기록 그대로."""

    def test_off_keeps_display_invalid_and_no_axis_fields(self):
        """axis_extractor None이면 볼트가 있어도 TOP_FACE_UNRESOLVED 무효, 인스턴스에 끝점 평가 없음, 기록에 axis 키 없음."""
        _, result, obs = run_job(ai_config(), [bolt_mask()])

        self.assertEqual((obs.valid, obs.reason, obs.observed_face, obs.axis_endpoints_px), (False, "TOP_FACE_UNRESOLVED",
                                                                                            None, None))
        self.assertFalse(result.instances[0].axis_evaluated)
        self.assertNotIn("axis_reason", _instance_summary(result.instances[0]))


class AxisSelectionTest(unittest.TestCase):
    """켬: #163 §4 표 순서의 첫 사유 또는 유일 볼트의 계약 관측."""

    def test_single_bolt_valid_contract_observation(self):
        """유일한 볼트의 추출 성공은 valid·axis·[머리 끝, 축 끝]·선택 인덱스·클래스 bolt 관측이고 끝점은 그린 도형 양 끝(축선 위)이다."""
        _, result, obs = run_job(axis_cfg(), [bolt_mask()])

        self.assertTrue(obs.valid)
        self.assertEqual((obs.observed_face, obs.calc_contour, obs.selected_index, obs.class_name, obs.reason),
                         ("axis", None, 0, "bolt", None))
        np.testing.assert_allclose(obs.axis_endpoints_px, [[40.0, 59.5], [83.0, 59.5]], atol=0.5)
        np.testing.assert_array_equal(obs.axis_endpoints_px, result.instances[0].axis_endpoints_px)

    def test_head_on_right_keeps_head_first(self):
        """머리가 오른쪽이어도 행 0이 머리 끝(오른쪽)이다(순서 보존)."""
        _, _, obs = run_job(axis_cfg(), [bolt_mask(flip=True)])

        self.assertTrue(obs.valid)
        self.assertGreater(obs.axis_endpoints_px[0, 0], obs.axis_endpoints_px[1, 0])

    def test_reasons_in_table_order(self):
        """프로파일 없음→CALIBRATION_MISMATCH, 볼트 없음(너트만)→TOP_FACE_UNRESOLVED, 무검출→NO_OBJECT, 겹침→MULTIPLE_CANDIDATES,
        2개→MULTIPLE_OBJECTS, 모호→AXIS_HEAD_AMBIGUOUS, 조각→MASK_FRAGMENTED. 모두 끝점 없는 무효다."""
        rect = np.zeros((H, W), bool)
        rect[50:60, 40:90] = True
        frag = bolt_mask()
        frag[10:14, 150:160] = True
        cases = {
            "CALIBRATION_MISMATCH": dict(masks=[bolt_mask()], prof=False),
            "TOP_FACE_UNRESOLVED": dict(masks=[bolt_mask()], cls=[1]),
            "NO_OBJECT": dict(masks=[]),
            MULTIPLE_CANDIDATES: dict(masks=[bolt_mask(), bolt_mask(x0=42)]),
            MULTIPLE_OBJECTS: dict(masks=[bolt_mask(y=30), bolt_mask(y=90)]),
            "AXIS_HEAD_AMBIGUOUS": dict(masks=[rect]),
            "MASK_FRAGMENTED": dict(masks=[frag]),
        }

        for reason, kw in cases.items():
            with self.subTest(reason=reason):
                _, _, obs = run_job(axis_cfg(), kw["masks"], cls=kw.get("cls"), prof=kw.get("prof", True))
                self.assertEqual((obs.valid, obs.reason, obs.axis_endpoints_px, obs.observed_face),
                                 (False, reason, None, None))

    def test_bolt_with_nut_still_valid(self):
        """볼트 1개 + 너트 1개는 볼트 경로로 유효(너트는 윗면 미구현이라 볼트 선택에 끼지 않음)."""
        nut = np.zeros((H, W), bool)
        nut[20:35, 150:165] = True
        _, _, obs = run_job(axis_cfg(), [bolt_mask(), nut], cls=[0, 1])

        self.assertTrue(obs.valid)
        self.assertEqual(obs.selected_index, 0)

    def test_clipped_bolt_is_contour_clipped(self):
        """프레임 가장자리에 닿은 볼트는 추출하지 않고 CONTOUR_CLIPPED 무효다."""
        _, result, obs = run_job(axis_cfg(), [bolt_mask(x0=0)])

        self.assertEqual((obs.valid, obs.reason), (False, "CONTOUR_CLIPPED"))
        self.assertEqual(result.instances[0].axis_reason, "CONTOUR_CLIPPED")

    def test_extractor_exception_is_invalid_not_raised(self):
        """추출기 내부 예외는 관측 예외로 새지 않고 AXIS_EXTRACTION_ERROR 무효다."""
        with mock.patch.object(ai_convert, "extract_m1", side_effect=RuntimeError("boom")):
            _, result, obs = run_job(axis_cfg(), [bolt_mask()])

        self.assertEqual((obs.valid, obs.reason), (False, AXIS_EXTRACTION_ERROR))
        self.assertIn("boom", result.instances[0].axis_diag["error"])

    def test_roi_offset_added(self):
        """ROI를 쓰면 끝점에 ROI 오프셋만 더해 원본 프레임 픽셀이 된다."""
        cfg = axis_cfg(roi=[20, 10, 160, 100])

        def crop_bolt():
            """crop(160×100) 좌표 볼트."""
            m = np.zeros((100, 160), bool)
            m[41:59, 30:40] = True
            m[45:55, 40:74] = True
            return m

        fr = frame(0, image=np.zeros((H, W, 3), np.uint8), width=W, height=H, calibration_id=CAL)
        p = VisionPipeline(settings(detector_config=cfg, profile=profile(width=W, height=H, calibration_id=CAL)),
                           run_id=RUN_ID, clock_ns=FakeClock())
        p.offer_frame(fr)
        job = p.start_next()
        result = convert(raw([crop_bolt()], (160, 100)), job.frame, cfg)
        obs = observation_from_ai(job, result)

        self.assertTrue(obs.valid, obs.reason)
        np.testing.assert_allclose(obs.axis_endpoints_px, [[50.0, 59.5], [93.0, 59.5]], atol=0.5)

    def test_core_accepts_valid_and_records_axis(self):
        """코어 complete가 유효 관측을 현재 관측으로 받고(계약 위반 아님), 인스턴스 기록에 axis 끝점·사유·IoU가 남는다."""
        cfg = axis_cfg()
        fr = frame(0, image=np.zeros((H, W, 3), np.uint8), width=W, height=H, calibration_id=CAL)
        p = VisionPipeline(settings(detector_config=cfg, profile=profile(width=W, height=H, calibration_id=CAL)),
                           run_id=RUN_ID, clock_ns=FakeClock())
        p.offer_frame(fr)
        job = p.start_next()
        result = convert(raw([bolt_mask()], (W, H)), job.frame, cfg)
        outcome = p.complete(job, observation_from_ai(job, result), display_result=result)

        self.assertEqual(outcome.status.value, "COMPLETED")
        self.assertTrue(outcome.applied)
        self.assertTrue(outcome.observation.valid)
        summary = _instance_summary(result.instances[0])
        self.assertEqual((summary["axis_reason"], summary["axis_max_iou"]), (None, 0.0))
        self.assertEqual(len(summary["axis_endpoints_px"]), 2)
        self.assertTrue(b_admit(p, outcome))


class PolygonRasterTest(unittest.TestCase):
    """M1 입력 = 표시 polygon 래스터(#168 §1, F2). 위상은 인스턴스 계수로."""

    def test_raster_reproduces_simple_masks(self):
        """조각·구멍 없는 마스크(볼트·기운 볼트·원·불규칙 덩어리)는 polygon 래스터가 원본 마스크와 화소 단위로 같다."""
        import cv2

        shapes = [bolt_mask(), bolt_mask(flip=True)]
        tilted = np.zeros((H, W), np.uint8)
        cv2.fillPoly(tilted, [np.array([[60, 40], [70, 34], [110, 90], [100, 96]], np.int32)], 1)
        disk = np.zeros((H, W), np.uint8)
        cv2.circle(disk, (100, 60), 17, 1, -1)
        rng = np.random.default_rng(168)
        blob = np.zeros((H, W), np.uint8)
        x, y = 80, 50

        for _ in range(12):   # 서로 겹치는 원을 경로를 따라 이어 한 덩어리로(반지름 ≥ 걸음)
            cv2.circle(blob, (x, y), int(rng.integers(5, 9)), 1, -1)
            x, y = x + int(rng.integers(1, 5)), y + int(rng.integers(-3, 4))

        shapes += [tilted.astype(bool), disk.astype(bool), blob.astype(bool)]

        for k, m in enumerate(shapes):
            with self.subTest(shape=k):
                _, result, _ = run_job(axis_cfg(), [m])
                inst = result.instances[0]
                self.assertEqual((inst.component_count, inst.hole_count), (1, 0))
                raster = ai_convert.rasterize_polygon(inst.polygon, np.zeros(2), (W, H))
                np.testing.assert_array_equal(raster, m)

    def test_topology_from_counts_matches_raw_extractor(self):
        """구멍 있는 볼트는 MASK_HOLE, 조각난 볼트는 MASK_FRAGMENTED이며 원본 마스크를 M1에 넣었을 때의 사유와 같다."""
        from eyetoarm.vision.axis_endpoints import extract_m1

        holed = bolt_mask()
        holed[58:60, 60:63] = False
        frag = bolt_mask()
        frag[10:14, 150:160] = True

        for m, reason in ((holed, "MASK_HOLE"), (frag, "MASK_FRAGMENTED")):
            with self.subTest(reason=reason):
                _, result, obs = run_job(axis_cfg(), [m])
                self.assertEqual((obs.valid, obs.reason, result.instances[0].axis_reason), (False, reason, reason))
                self.assertEqual(extract_m1(m).reason, reason)

    def test_non_integer_polygon_is_extraction_error(self):
        """정수 화소 중심이 아닌 polygon은 래스터 규약 위반이라 AXIS_EXTRACTION_ERROR 무효다(관측 예외로 새지 않음)."""
        with self.assertRaises(ValueError):
            ai_convert.rasterize_polygon(np.array([[1.5, 1.0], [5.0, 1.0], [5.0, 5.0]]), np.zeros(2), (10, 10))


class BAdmitTest(unittest.TestCase):
    """B 호출 조건: b_gate + 실시간이면 수신 epoch 현재."""

    def test_live_stale_frame_not_admitted(self):
        """b_gate를 통과한 결과도 실시간 소스의 frame_current가 거짓이면 B로 가지 않고, 참이면 간다(D13 해제 전 대비 조건)."""
        cfg = axis_cfg()
        fr = frame(0, image=np.zeros((H, W, 3), np.uint8), width=W, height=H, calibration_id=CAL)
        p = VisionPipeline(settings(detector_config=cfg, profile=profile(width=W, height=H, calibration_id=CAL)),
                           run_id=RUN_ID, clock_ns=FakeClock())
        p.offer_frame(fr)
        job = p.start_next()
        result = convert(raw([bolt_mask()], (W, H)), job.frame, cfg)
        outcome = p.complete(job, observation_from_ai(job, result), display_result=result)

        class Live:
            """frame_current 값을 정해 둔 실시간 소스 흉내."""

            def __init__(self, current):
                """현재 여부."""
                self.current = current

            def frame_current(self, f):
                """정해 둔 값."""
                return self.current

        self.assertFalse(b_admit(p, outcome, Live(False)))
        self.assertTrue(b_admit(p, outcome, Live(True)))

    def test_invalid_never_admitted(self):
        """무효 관측은 실시간 여부와 무관하게 B로 가지 않는다."""
        cfg = axis_cfg()
        fr = frame(0, image=np.zeros((H, W, 3), np.uint8), width=W, height=H, calibration_id=CAL)
        p = VisionPipeline(settings(detector_config=cfg, profile=profile(width=W, height=H, calibration_id=CAL)),
                           run_id=RUN_ID, clock_ns=FakeClock())
        p.offer_frame(fr)
        job = p.start_next()
        rect = np.zeros((H, W), bool)
        rect[50:60, 40:90] = True
        result = convert(raw([rect], (W, H)), job.frame, cfg)
        outcome = p.complete(job, observation_from_ai(job, result), display_result=result)

        self.assertFalse(outcome.observation.valid)
        self.assertFalse(b_admit(p, outcome))


if __name__ == "__main__":
    unittest.main()
