"""모델 원시 출력 → 표시 결과 → 표시용 무효 관측 변환(`ai_convert`) 시험(가짜 원시 배열).

좌표 기대값은 손으로 정한 상수(사각형 꼭짓점)다. 실제 모델·torch를 쓰지 않는다.
"""
import unittest

import numpy as np

from eyetoarm.vision.ai_convert import (convert, crop_frame, display_reason, observation_from_ai)
from eyetoarm.vision.ai_result import AiJobConfigError, RawSegOutput
from eyetoarm.vision.pipeline import VisionPipeline
from test_ai_config import ai_config
from test_pipeline import RUN_ID, FakeClock, frame, profile, settings

W, H = 200, 120   # 시험 프레임 크기(원본)


def big_frame(seq=0, w=W, h=H, fill=0):
    """원본 W×H 시험 프레임(BGR uint8)."""
    return frame(seq, image=np.full((h, w, 3), fill, np.uint8), width=w, height=h)


def rect_mask(crop_wh, x0, y0, x1, y1):
    """crop (h, w) 크기 bool 마스크에 [x0..x1]×[y0..y1](양 끝 포함) 사각형을 칠한다."""
    w, h = crop_wh
    m = np.zeros((h, w), bool)
    m[y0:y1 + 1, x0:x1 + 1] = True
    return m


def raw(masks, crop_wh, cls=None, conf=None, boxes=None):
    """가짜 원시 출력. boxes 기본값은 각 마스크의 경계 사각형(crop 좌표)."""
    k = len(masks)
    cls = [0] * k if cls is None else cls
    conf = [0.9] * k if conf is None else conf

    if boxes is None:
        boxes = []
        for m in masks:
            ys, xs = np.nonzero(m)
            boxes.append([xs.min(), ys.min(), xs.max(), ys.max()] if xs.size else [0, 0, 0, 0])

    w, h = crop_wh
    return RawSegOutput(boxes_xyxy=np.array(boxes, np.float32).reshape(k, 4), confidence=np.array(conf, np.float32),
                        class_index=np.array(cls, np.int64), masks=np.array(masks, bool).reshape(k, h, w),
                        crop_wh=crop_wh, infer_ms=4.0)


def corners(polygon):
    """다각형 점 집합(정수 튜플)."""
    return {(int(x), int(y)) for x, y in polygon}


def job_for(cfg, fr, prof=None):
    """AI 설정 파이프라인에서 프레임 하나로 작업을 시작한다."""
    p = VisionPipeline(settings(detector_config=cfg, profile=prof), run_id=RUN_ID, clock_ns=FakeClock())
    p.offer_frame(fr)
    return p.start_next()


class CropTest(unittest.TestCase):
    """입력 검사·ROI 자르기."""

    def test_crop_is_bgr_view_without_conversion(self):
        """ROI crop은 원본 BGR 값을 그대로 담고(색 변환 없음) 오프셋을 함께 돌려준다. ROI 없음이면 원본 전체."""
        img = np.zeros((H, W, 3), np.uint8)
        img[50:60, 100:110] = (255, 0, 0)   # B 채널만
        fr = frame(0, image=img, width=W, height=H)

        crop, offset = crop_frame(fr, ai_config(roi=[100, 50, 64, 48]))
        self.assertEqual((crop.shape, offset), ((48, 64, 3), (100, 50)))
        self.assertEqual(tuple(int(v) for v in crop[0, 0]), (255, 0, 0))

        full, off0 = crop_frame(fr, ai_config())
        self.assertEqual((full.shape, off0), ((H, W, 3), (0, 0)))

    def test_roi_outside_frame_is_job_config_error(self):
        """ROI가 프레임 밖이면 작업 설정 오류(AiJobConfigError → MODEL_CONFIG_INVALID 작업 값)다."""
        with self.assertRaises(AiJobConfigError):
            crop_frame(big_frame(), ai_config(roi=[150, 0, 64, 48]))

    def test_bad_frame_image_is_value_error(self):
        """BGR uint8 (H,W,3)가 아니거나 크기 메타데이터가 다르면 ValueError(INFERENCE_ERROR로 정리될 일반 오류)."""
        for img, w, h in ((np.zeros((H, W), np.uint8), W, H), (np.zeros((H, W, 3), np.float32), W, H),
                          (np.zeros((H, W, 3), np.uint8), W + 1, H)):
            with self.subTest(shape=img.shape, dtype=str(img.dtype), w=w):
                with self.assertRaises(ValueError):
                    crop_frame(frame(0, image=img, width=w, height=h), ai_config())


class ConvertGeometryTest(unittest.TestCase):
    """좌표 복원·거부·기하."""

    def test_roi_offset_only_no_letterbox_inverse(self):
        """crop 마스크 사각형 x10..19·y5..14 → ROI (100,50) 오프셋만 더한 원본 꼭짓점·박스."""
        cw = (64, 48)
        r = convert(raw([rect_mask(cw, 10, 5, 19, 14)], cw), big_frame(), ai_config(roi=[100, 50, 64, 48]))
        inst = r.instances[0]

        self.assertTrue({(110, 55), (119, 55), (119, 64), (110, 64)} <= corners(inst.polygon))
        self.assertTrue(all(110 <= x <= 119 and 55 <= y <= 64 for x, y in corners(inst.polygon)))
        self.assertEqual(inst.box_xyxy, (110.0, 55.0, 119.0, 64.0))
        self.assertEqual((inst.area_px, inst.component_count, inst.hole_count), (100.0, 1, 0))
        self.assertEqual((r.roi, r.crop_wh, inst.rejected, inst.clipped), ((100, 50, 64, 48), (64, 48), None, False))

    def test_odd_and_non_square_roi(self):
        """홀수·비정사각 ROI (3,7,641,361): crop x100..119·y200..229 → 원본 x103..122·y207..236."""
        cw = (641, 361)
        fr = big_frame(w=700, h=400)
        r = convert(raw([rect_mask(cw, 100, 200, 119, 229)], cw), fr, ai_config(roi=[3, 7, 641, 361]))

        self.assertTrue({(103, 207), (122, 207), (122, 236), (103, 236)} <= corners(r.instances[0].polygon))

    def test_no_roi_full_frame_coordinates(self):
        """ROI 없음이면 crop = 원본이고 좌표는 그대로다."""
        cw = (W, H)
        r = convert(raw([rect_mask(cw, 40, 30, 59, 49)], cw), big_frame(), ai_config())

        self.assertTrue({(40, 30), (59, 30), (59, 49), (40, 49)} <= corners(r.instances[0].polygon))
        self.assertIsNone(r.roi)

    def test_multiple_instances_index_order(self):
        """인스턴스는 모델 출력 순서대로 index 0..K-1이며 클래스 이름은 class_map으로 붙는다."""
        cw = (W, H)
        r = convert(raw([rect_mask(cw, 10, 10, 29, 29), rect_mask(cw, 100, 50, 129, 79)], cw, cls=[1, 0],
                        conf=[0.8, 0.6]), big_frame(), ai_config())

        self.assertEqual([(i.index, i.class_name, round(i.confidence, 2)) for i in r.instances],
                         [(0, "nut", 0.8), (1, "bolt", 0.6)])
        self.assertEqual((r.candidate_count, r.eligible_count), (2, 2))

    def test_rejection_order(self):
        """비대상이면서 작은 마스크는 먼저 걸린 NON_TARGET_CLASS이고 기하는 그대로 남는다. 대상·작은 마스크는 SMALL_MASK."""
        cw = (W, H)
        cfg = ai_config(target_classes=["bolt"], min_mask_area_px=50)
        r = convert(raw([rect_mask(cw, 10, 10, 13, 13), rect_mask(cw, 50, 50, 53, 53)], cw, cls=[1, 0]),
                    big_frame(), cfg)

        self.assertEqual([i.rejected for i in r.instances], ["NON_TARGET_CLASS", "SMALL_MASK"])
        self.assertTrue(all(i.polygon is not None for i in r.instances))

    def test_low_confidence_is_rejected_defensively(self):
        """conf_threshold 미만 점수(라이브러리가 걸렀어야 하지만)는 LOW_CONFIDENCE로 남긴다."""
        cw = (W, H)
        r = convert(raw([rect_mask(cw, 10, 10, 29, 29)], cw, conf=[0.1]), big_frame(), ai_config())

        self.assertEqual((r.instances[0].rejected, r.candidate_count), ("LOW_CONFIDENCE", 0))

    def test_non_target_or_low_conf_with_empty_or_degenerate_mask(self):
        """비대상+1화소·점수 미달+빈 마스크·대상+1화소 → 거부 3종, polygon None, reason NO_OBJECT."""
        cw = (W, H)
        one = rect_mask(cw, 20, 20, 20, 20)
        empty = np.zeros((H, W), bool)
        cfg = ai_config(target_classes=["bolt"])
        fr = big_frame()
        r = convert(raw([one, empty, one], cw, cls=[1, 0, 0], conf=[0.9, 0.1, 0.9],
                        boxes=[[20, 20, 20, 20]] * 3), fr, cfg)

        self.assertEqual([i.rejected for i in r.instances], ["NON_TARGET_CLASS", "LOW_CONFIDENCE", "EMPTY_MASK"])
        self.assertEqual([i.polygon for i in r.instances], [None, None, None])
        self.assertEqual([i.area_px for i in r.instances], [1.0, 0.0, 1.0])
        self.assertEqual((len(r.instances), r.candidate_count, r.eligible_count), (3, 1, 0))
        self.assertEqual(display_reason(r, fr, profile(width=W, height=H)), "NO_OBJECT")

    def test_line_mask_has_no_geometry(self):
        """1화소 폭 직선 마스크는 면적 0 윤곽이라 기하 없음 → 대상이면 EMPTY_MASK, polygon None."""
        cw = (W, H)
        r = convert(raw([rect_mask(cw, 30, 40, 60, 40)], cw), big_frame(), ai_config())

        self.assertEqual((r.instances[0].rejected, r.instances[0].polygon, r.instances[0].area_px),
                         ("EMPTY_MASK", None, 31.0))

    def test_mixed_non_target_low_conf_small_clipped(self):
        """비대상·점수 미달·작고 잘린 대상 → candidate 1, reason NO_OBJECT(CONTOUR_CLIPPED 아님)."""
        cw = (W, H)
        cfg = ai_config(target_classes=["bolt"], min_mask_area_px=50)
        fr = big_frame()
        r = convert(raw([rect_mask(cw, 50, 50, 69, 69), rect_mask(cw, 100, 50, 119, 69), rect_mask(cw, 0, 0, 3, 3)],
                        cw, cls=[1, 0, 0], conf=[0.9, 0.1, 0.9]), fr, cfg)

        self.assertEqual([i.rejected for i in r.instances], ["NON_TARGET_CLASS", "LOW_CONFIDENCE", "SMALL_MASK"])
        self.assertTrue(r.instances[2].clipped)
        self.assertEqual((r.candidate_count, r.eligible_count), (1, 0))
        self.assertEqual(display_reason(r, fr, profile(width=W, height=H)), "NO_OBJECT")

    def test_clipped_eligible_size_gives_contour_clipped(self):
        """면적 통과 대상이 경계 2 px 안이면 clipped·CONTOUR_CLIPPED, 3 px 떨어지면 잘림 아님."""
        cw = (64, 48)
        fr = big_frame()
        cfg = ai_config(roi=[100, 50, 64, 48], border_margin_px=2)
        clipped = convert(raw([rect_mask(cw, 2, 10, 21, 29)], cw), fr, cfg)
        inside = convert(raw([rect_mask(cw, 3, 10, 22, 29)], cw), fr, cfg)
        right = convert(raw([rect_mask(cw, 40, 10, 61, 29)], cw), fr, cfg)

        self.assertEqual([r.instances[0].clipped for r in (clipped, inside, right)], [True, False, True])
        self.assertEqual(display_reason(clipped, fr, profile(width=W, height=H)), "CONTOUR_CLIPPED")
        self.assertEqual(display_reason(inside, fr, profile(width=W, height=H)), "TOP_FACE_UNRESOLVED")

    def test_hole_and_disjoint_counted_largest_outer_used(self):
        """구멍 난 사각형은 component 1·hole 1, 떨어진 두 사각형은 component 2·polygon은 큰 쪽."""
        cw = (W, H)
        ring = rect_mask(cw, 20, 20, 59, 59)
        ring[30:50, 30:50] = False
        two = rect_mask(cw, 100, 20, 139, 59) | rect_mask(cw, 160, 20, 169, 29)
        r = convert(raw([ring, two], cw), big_frame(), ai_config())

        self.assertEqual([(i.component_count, i.hole_count) for i in r.instances], [(1, 1), (2, 0)])
        self.assertEqual(r.instances[0].area_px, 40 * 40 - 20 * 20)
        self.assertTrue({(100, 20), (139, 20), (139, 59), (100, 59)} <= corners(r.instances[1].polygon))
        self.assertTrue(all(x <= 139 for x, _ in corners(r.instances[1].polygon)))

    def test_unknown_class_is_job_config_error(self):
        """class_map에 없는 클래스 번호는 작업 설정 오류(AiJobConfigError)다."""
        cw = (W, H)

        with self.assertRaises(AiJobConfigError):
            convert(raw([rect_mask(cw, 10, 10, 29, 29)], cw, cls=[7]), big_frame(), ai_config())

    def test_crop_size_mismatch_is_value_error(self):
        """원시 출력의 crop 크기가 실제 crop과 다르면(retina 계약 위반·이중 변환 위험) ValueError."""
        cw = (64, 48)

        with self.assertRaises(ValueError):
            convert(raw([rect_mask(cw, 10, 10, 20, 20)], cw), big_frame(), ai_config())

    def test_result_owns_polygons_not_masks(self):
        """결과는 마스크 배열·원시 출력을 들지 않고, polygon은 원시 마스크와 메모리를 나누지 않는 읽기 전용 복사본이다."""
        cw = (W, H)
        rw = raw([rect_mask(cw, 10, 10, 29, 29)], cw)
        r = convert(rw, big_frame(), ai_config())
        poly = r.instances[0].polygon

        self.assertFalse(np.shares_memory(poly, rw.masks))
        self.assertFalse(poly.flags.writeable)
        self.assertFalse(any(isinstance(getattr(r, f), np.ndarray) for f in r.__dataclass_fields__))

    def test_timings_and_identity(self):
        """infer_ms는 원시 출력 값, post_ms·process_ms는 0 이상, model_id·config_sha256은 설정 값이다."""
        cw = (W, H)
        cfg = ai_config()
        r = convert(raw([], cw), big_frame(), cfg)

        self.assertEqual((r.infer_ms, r.model_id, r.config_sha256, r.instances), (4.0, cfg.model_id,
                                                                                    cfg.config_sha256, ()))
        self.assertGreaterEqual(r.post_ms, 0.0)
        self.assertGreaterEqual(r.process_ms, r.infer_ms)


class ReasonAndObservationTest(unittest.TestCase):
    """reason 우선순위와 표시용 무효 관측."""

    def result_with(self, masks, fr, cfg=None, **kw):
        """원본 전체 crop 변환 결과."""
        return convert(raw(masks, (fr.width, fr.height), **kw), fr, cfg or ai_config())

    def test_reason_priority(self):
        """CALIBRATION_MISMATCH > TOP_FACE_UNRESOLVED(복수 대상도) > CONTOUR_CLIPPED > NO_OBJECT."""
        fr = big_frame()
        cw = (W, H)
        good = profile(width=W, height=H)
        one = self.result_with([rect_mask(cw, 50, 50, 69, 69)], fr)
        two = self.result_with([rect_mask(cw, 50, 50, 69, 69), rect_mask(cw, 100, 50, 119, 69)], fr)
        clipped = self.result_with([rect_mask(cw, 0, 50, 19, 69)], fr)
        none = self.result_with([], fr)

        self.assertEqual(display_reason(one, fr, None), "CALIBRATION_MISMATCH")
        self.assertEqual(display_reason(one, fr, profile()), "CALIBRATION_MISMATCH")   # 6×4 불일치
        self.assertEqual(display_reason(one, fr, good), "TOP_FACE_UNRESOLVED")
        self.assertEqual(display_reason(two, fr, good), "TOP_FACE_UNRESOLVED")
        self.assertEqual(display_reason(clipped, fr, good), "CONTOUR_CLIPPED")
        self.assertEqual(display_reason(none, fr, good), "NO_OBJECT")

    def test_observation_never_valid_and_ai_fields(self):
        """관측은 valid=False·method ai·model_id 값, 계산 필드는 None, candidate_count = in_scope 수."""
        cfg = ai_config()
        fr = big_frame()
        job = job_for(cfg, fr, profile(width=W, height=H))
        r = self.result_with([rect_mask((W, H), 50, 50, 69, 69), rect_mask((W, H), 100, 50, 119, 69)], fr, cfg,
                             cls=[0, 1])

        obs = observation_from_ai(job, r)

        self.assertEqual((obs.valid, obs.reason, obs.method, obs.model_id, obs.candidate_count),
                         (False, "TOP_FACE_UNRESOLVED", "ai", cfg.model_id, 2))
        self.assertEqual((obs.observed_face, obs.calc_contour, obs.display_mask_contour, obs.selected_index,
                          obs.class_name, obs.confidence), (None,) * 6)
        self.assertEqual((obs.job_seq, obs.generation, obs.detector_config_id, obs.process_ms),
                         (job.job_seq, job.generation, job.detector_config_id, r.process_ms))

    def test_config_id_mismatch_with_job_snapshot_raises(self):
        """표시 결과의 설정 해시가 작업 시작 설정과 다르면 관측을 만들지 않고 AiJobConfigError다."""
        fr = big_frame()
        job = job_for(ai_config(), fr)
        r = self.result_with([], fr, ai_config(conf_threshold=0.5))

        with self.assertRaises(AiJobConfigError):
            observation_from_ai(job, r)


if __name__ == "__main__":
    unittest.main()
