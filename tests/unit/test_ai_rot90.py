"""90° 돌린 추론(#425·#429) 시험: 설정 키, 되돌리기, R5 합치기, AiSegDetector 연결, 기록, 별도 프로세스.

실제 ultralytics·torch·가중치 없이 돈다. 돌린 영상의 좌표 약속은 `ai_rot90` 모듈 docstring을 따른다.
"""
import json
import os
import unittest

import cv2
import numpy as np

from eyetoarm.vision.ai_config import Rotate90Config, load_ai_config
from eyetoarm.vision.ai_convert import convert, observation_from_ai
from eyetoarm.vision.ai_result import EMPTY_MASK, SOURCE_ROT90, AiInstance, AiSegResult, RawSegOutput
from eyetoarm.vision.ai_rot90 import merge_r5, rotate_back, rotate_crop
from eyetoarm.vision.pipeline import VisionPipeline
from eyetoarm.vision.pipeline_records import _detector_block, _instance_summary, _plain, process_record
from eyetoarm.vision.pipeline_runner import CancelToken
from test_ai_config import ai_config
from test_ai_convert import raw, rect_mask
from test_ai_detector import FakeModel, WeightsMixin, result_for
from test_ai_process import Base as ProcessBase, image
from test_pipeline import RUN_ID, FakeClock, frame, profile, settings

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OPS = os.path.join(ROOT, "config", "vision", "ai_s1_08_axis_m2_viewbias.json")
OPS_ROT = os.path.join(ROOT, "config", "vision", "ai_s1_08_axis_m2_viewbias_rot90.json")
OPS_ROT_N2 = os.path.join(ROOT, "config", "vision", "ai_s1_08_axis_m2_viewbias_rot90_n2.json")
OPS_ID = "6af8c9449eee3b297e35e6034906cf00b6704129f3a15a8b1b24c2ee505ecf3f"   # #259 기록·test_view_bias_spec와 같은 값
ROT = {"direction": "cw", "every_n": 1, "nms_iou": 0.5, "nut_veto_iou": 0.1}
NAMES = {0: "bolt", 1: "nut"}
CW, CH = 200, 120   # 시험 crop (폭, 높이)


def rot_rect(x0, y0, x1, y1, h=CH):
    """원래 crop의 사각형(양 끝 포함)을 시계 방향 90° 돌린 영상의 사각형으로(양 끝 포함)."""
    return h - 1 - y1, x0, h - 1 - y0, x1


def out(masks, cls=None, conf=None, wh=(CW, CH), boxes=None, infer_ms=0.0):
    """시험용 원시 출력(상자 기본값 = 마스크 경계, 끝 포함 좌표)."""
    r = raw(masks, wh, cls=cls, conf=conf, boxes=boxes)

    return RawSegOutput(boxes_xyxy=r.boxes_xyxy, confidence=r.confidence, class_index=r.class_index, masks=r.masks,
                        crop_wh=wh, infer_ms=infer_ms)


def m(x0, y0, x1, y1, wh=(CW, CH)):
    """사각형 마스크(양 끝 포함)."""
    return rect_mask(wh, x0, y0, x1, y1)


# ---------------------------------------------------------------------- 설정


class ConfigTest(unittest.TestCase):
    """선택 키 `rotate90_pass`의 검증·정규형·설정 ID."""

    def test_absent_key_keeps_canonical_json_and_id(self):
        """키가 없으면 정규 JSON에 키가 없고 운영 설정 ID는 지금 값 그대로다."""
        self.assertNotIn("rotate90_pass", ai_config().canonical_json())
        self.assertIsNone(ai_config().rotate90_pass)
        self.assertEqual(ai_config(rotate90_pass=None).config_sha256, ai_config().config_sha256)
        self.assertEqual(load_ai_config(OPS).config_sha256, OPS_ID)

    def test_key_changes_id_and_every_n_matters(self):
        """키가 있으면 ID가 바뀌고, every_n·IoU 값이 달라도 바뀐다. dict와 Rotate90Config는 같은 ID다."""
        base = ai_config().config_sha256
        on = ai_config(rotate90_pass=ROT)

        self.assertNotEqual(on.config_sha256, base)
        self.assertIsInstance(on.rotate90_pass, Rotate90Config)
        self.assertIn('"rotate90_pass":{"direction":"cw","every_n":1,"nms_iou":0.5,"nut_veto_iou":0.1}',
                      on.canonical_json())
        self.assertEqual(ai_config(rotate90_pass=Rotate90Config(**ROT)).config_sha256, on.config_sha256)

        for over in ({"every_n": 2}, {"nms_iou": 0.6}, {"nut_veto_iou": 0.2}):
            with self.subTest(over=over):
                self.assertNotEqual(ai_config(rotate90_pass={**ROT, **over}).config_sha256, on.config_sha256)

    def test_invalid_values_rejected(self):
        """방향 ccw·모르는 방향, every_n 0·소수·bool, IoU 0·1 초과·bool, 키 빠짐·모르는 키, 객체 아님은 ValueError."""
        bad = [{**ROT, "direction": "ccw"}, {**ROT, "direction": "up"}, {**ROT, "every_n": 0},
               {**ROT, "every_n": 1.5}, {**ROT, "every_n": True}, {**ROT, "nms_iou": 0.0}, {**ROT, "nms_iou": 1.01},
               {**ROT, "nut_veto_iou": True}, {**ROT, "nut_veto_iou": float("nan")},
               {k: v for k, v in ROT.items() if k != "every_n"}, {**ROT, "extra": 1}, "cw", [ROT]]

        for v in bad:
            with self.subTest(v=v), self.assertRaises(ValueError):
                ai_config(rotate90_pass=v)

    def test_boundary_values_accepted(self):
        """IoU 1.0·큰 every_n은 받고, 정수 IoU는 float로 정규화한다."""
        c = ai_config(rotate90_pass={**ROT, "every_n": 7, "nms_iou": 1, "nut_veto_iou": 1.0}).rotate90_pass

        self.assertEqual((c.every_n, c.nms_iou, type(c.nms_iou)), (7, 1.0, float))

    def test_allowed_without_axis_extractor(self):
        """축 추출기가 꺼진 설정에서도 켤 수 있다(A 결정 #429)."""
        self.assertIsNotNone(ai_config(axis_extractor=None, rotate90_pass=ROT).rotate90_pass)

    def test_operational_rot90_files_differ_only_by_key(self):
        """새 설정 파일은 운영 설정과 rotate90_pass 키만 다르고, ID가 서로 다르다."""
        with open(OPS, encoding="utf-8") as f:
            ops = json.load(f)

        ids = {OPS_ID}

        for path, every_n in ((OPS_ROT, 1), (OPS_ROT_N2, 2)):
            with self.subTest(path=path):
                with open(path, encoding="utf-8") as f:
                    doc = json.load(f)

                self.assertEqual(doc.pop("rotate90_pass"), {**ROT, "every_n": every_n})
                self.assertEqual(doc, ops)
                ids.add(load_ai_config(path).config_sha256)

        self.assertEqual(len(ids), 3)


# ---------------------------------------------------------------------- 돌리기·되돌리기


class RotateBackTest(unittest.TestCase):
    """돌린 crop 좌표 → 원래 crop 좌표."""

    def test_rotate_crop_is_clockwise(self):
        """rotate_crop은 시계 방향: 원래 (x, y)가 돌린 영상 (열 H-1-y, 행 x)에 간다. ccw는 거부한다."""
        crop = np.zeros((CH, CW, 3), np.uint8)
        crop[10, 30] = (1, 2, 3)
        rc = rotate_crop(crop)

        self.assertEqual(rc.shape, (CW, CH, 3))
        self.assertEqual(tuple(rc[30, CH - 1 - 10]), (1, 2, 3))

        with self.assertRaises(ValueError):
            rotate_crop(crop, "ccw")

    def test_masks_and_boxes_round_trip(self):
        """원래 마스크·상자를 cv2로 돌린 출력을 되돌리면 원래 마스크·상자와 같다(사각형 둘, 비정사각 crop)."""
        rects = [(10, 5, 59, 14), (150, 40, 160, 110)]
        masks = [m(*r) for r in rects]
        rot_masks = [cv2.rotate(x.astype(np.uint8), cv2.ROTATE_90_CLOCKWISE).astype(bool) for x in masks]
        # 화소 경계 상자: 원래 (x0, y0, x1+1, y1+1) ↔ 돌린 (H-1-y1, x0, H-y0, x1+1)
        rot_boxes = [[CH - 1 - y1, x0, CH - y0, x1 + 1] for x0, y0, x1, y1 in rects]
        r = rotate_back(out(rot_masks, cls=[0, 1], conf=[0.8, 0.6], wh=(CH, CW), boxes=rot_boxes, infer_ms=3.0),
                        (CW, CH))

        self.assertEqual(r.crop_wh, (CW, CH))
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(r.masks, masks)))
        self.assertTrue(r.masks.flags["C_CONTIGUOUS"])
        np.testing.assert_allclose(r.boxes_xyxy, [[x0, y0, x1 + 1, y1 + 1] for x0, y0, x1, y1 in rects])
        self.assertEqual((r.class_index.tolist(), r.confidence.tolist(), r.infer_ms), ([0, 1], [0.800000011920929,
                                                                                               0.6000000238418579], 3.0))

    def test_empty_and_size_mismatch(self):
        """K=0은 빈 출력, 돌린 출력의 crop 크기가 (H, W)가 아니면 ValueError."""
        r = rotate_back(out([], wh=(CH, CW)), (CW, CH))

        self.assertEqual((r.count, r.masks.shape), (0, (0, CH, CW)))

        with self.assertRaises(ValueError):
            rotate_back(out([], wh=(CW, CH)), (CW, CH))


# ---------------------------------------------------------------------- 합치기 R5


class MergeR5Test(unittest.TestCase):
    """R5: 너트 우선, 같은 종류 IoU ≥ nms_iou 신뢰도 높은 쪽, 그대로 → 돌린 쪽 순서."""

    def merge(self, plain, rot, nms=0.5, veto=0.1):
        """기본 문턱으로 합친다."""
        return merge_r5(plain, rot, NAMES, nms_iou=nms, nut_veto_iou=veto)

    def test_same_class_overlap_keeps_higher_confidence(self):
        """같은 볼트를 양쪽이 내면 신뢰도 높은 쪽 하나만 남고, 돌린 쪽이면 출처가 rot90이다."""
        bolt = m(10, 10, 60, 20)

        for pc, rc, want_src in ((0.9, 0.7, None), (0.6, 0.8, SOURCE_ROT90)):
            with self.subTest(pc=pc, rc=rc):
                merged, src = self.merge(out([bolt], conf=[pc]), out([bolt], conf=[rc]))

                self.assertEqual((merged.count, src), (1, (want_src,)))
                self.assertAlmostEqual(float(merged.confidence[0]), max(pc, rc), places=6)

    def test_nms_threshold_inclusive(self):
        """IoU가 정확히 nms_iou면 합치고, 조금 아래면 둘 다 남는다."""
        a = m(0, 0, 9, 9)          # 100화소
        b_half = m(0, 0, 9, 9)     # 아래에서 IoU 0.5가 되도록 b를 만든다
        b_half[:, :] = False
        b_half[0:10, 5:15] = True  # 교집합 50, 합집합 150 → 1/3
        exact = m(0, 0, 9, 4)      # 교집합 50, 합집합 100 → 0.5

        merged, _ = self.merge(out([a], conf=[0.9]), out([exact], conf=[0.8]))
        self.assertEqual(merged.count, 1)

        merged, _ = self.merge(out([a], conf=[0.9]), out([b_half], conf=[0.8]))
        self.assertEqual(merged.count, 2)

    def test_different_classes_not_merged(self):
        """볼트와 너트가 겹쳐도(너트 우선 대상이 아닌 방향) 종류가 다르면 합치지 않는다."""
        x = m(10, 10, 30, 30)
        merged, src = self.merge(out([x], cls=[0], conf=[0.9]), out([x], cls=[1], conf=[0.8]))

        self.assertEqual((merged.count, src, merged.class_index.tolist()), (2, (None, SOURCE_ROT90), [0, 1]))

    def test_rotated_bolt_over_plain_nut_dropped(self):
        """돌린 쪽 볼트가 그대로 쪽 너트와 IoU ≥ 0.1이면 버리고, 0.1 미만이면 남긴다. 돌린 쪽 너트는 거부하지 않는다."""
        nut = m(10, 10, 29, 29)                      # 400화소
        over = m(10, 10, 29, 13)                     # 교집합 80/400 = 0.2
        small = m(26, 10, 45, 11)                    # 교집합 8, 합집합 432 → 0.0185
        plain = out([nut], cls=[1], conf=[0.9])

        merged, src = self.merge(plain, out([over], cls=[0], conf=[0.95]))
        self.assertEqual((merged.count, src), (1, (None,)))

        merged, src = self.merge(plain, out([small], cls=[0], conf=[0.95]))
        self.assertEqual((merged.count, src), (2, (None, SOURCE_ROT90)))

        merged, src = self.merge(plain, out([m(60, 60, 70, 70)], cls=[1], conf=[0.5]))
        self.assertEqual((merged.count, src), (2, (None, SOURCE_ROT90)))

    def test_plain_plain_overlap_also_merged(self):
        """잰 R5 그대로: 그대로 쪽끼리 같은 종류 IoU ≥ 0.5여도 신뢰도 높은 쪽만 남는다(A 결정 #429)."""
        bolt = m(10, 10, 60, 20)
        merged, src = self.merge(out([bolt, bolt], conf=[0.4, 0.7]), out([]))

        self.assertEqual((merged.count, src), (1, (None,)))
        self.assertAlmostEqual(float(merged.confidence[0]), 0.7, places=6)

    def test_order_and_infer_ms(self):
        """남은 것은 그대로 쪽 원래 순서 → 돌린 쪽 원래 순서이고, infer_ms는 두 추론의 합이다."""
        plain = out([m(0, 0, 5, 5), m(100, 0, 105, 5)], conf=[0.3, 0.9], infer_ms=10.0)
        rot = out([m(50, 50, 55, 55), m(150, 50, 155, 55)], conf=[0.95, 0.2], infer_ms=12.5)
        merged, src = self.merge(plain, rot)

        np.testing.assert_allclose(merged.confidence, [0.3, 0.9, 0.95, 0.2], rtol=1e-6)
        self.assertEqual(src, (None, None, SOURCE_ROT90, SOURCE_ROT90))
        self.assertEqual(merged.infer_ms, 22.5)

    def test_disjoint_boxes_skip_mask_compare(self):
        """상자가 떨어져 있으면 마스크가 상자 밖으로 겹쳐도 같은 물체로 보지 않는다(상자 안 마스크 계약)."""
        a = m(10, 10, 60, 20)
        merged, _ = self.merge(out([a], boxes=[[10, 10, 60, 20]]), out([a], boxes=[[100, 100, 120, 110]]))

        self.assertEqual(merged.count, 2)

    def test_empty_sides(self):
        """양쪽이나 한쪽이 비어도 된다."""
        merged, src = self.merge(out([]), out([]))
        self.assertEqual((merged.count, merged.masks.shape, src), (0, (0, CH, CW), ()))

        merged, src = self.merge(out([]), out([m(1, 1, 5, 5)]))
        self.assertEqual(src, (SOURCE_ROT90,))

    def test_crop_mismatch_rejected(self):
        """두 출력의 crop 크기가 다르면 ValueError."""
        with self.assertRaises(ValueError):
            self.merge(out([]), out([], wh=(CH, CW)))


# ---------------------------------------------------------------------- AiSegDetector 연결


def rot_aware(plain_rects, rot_rects, plain_cls=None, rot_cls=None, plain_conf=None, rot_conf=None):
    """crop 모양으로 그대로·돌린 호출을 구분하는 가짜 결과 생성기(돌린 쪽 사각형은 원래 좌표로 준다)."""
    def make(source):
        """가로가 긴 crop이면 그대로, 세로가 길면 돌린 호출."""
        h, w = source.shape[:2]

        if w >= h:
            return result_for(source, plain_rects, cls=plain_cls, conf=plain_conf)

        return result_for(source, [rot_rect(*r, h=w) for r in rot_rects], cls=rot_cls, conf=rot_conf)

    return make


class DetectorTest(WeightsMixin, unittest.TestCase):
    """AiSegDetector.infer: 꺼짐이면 지금 그대로, 켜짐이면 돌린 추론을 더해 합친다."""

    def ready(self, make, cfg_over=None):
        """준비를 마친 detector와 작업(프레임 200×120, 보정 없음)."""
        from test_ai_convert import big_frame, job_for

        det, cfg = self.make(model=FakeModel(make=make), cfg_over=cfg_over)
        det.prepare()
        fr = big_frame()
        img = np.arange(CH * CW * 3, dtype=np.uint32).reshape(CH, CW, 3).astype(np.uint8)
        fr = type(fr)(**{**{f: getattr(fr, f) for f in fr.__dataclass_fields__}, "image": img})

        return det, cfg, job_for(cfg, fr, None)

    def test_off_predicts_once_without_new_fields(self):
        """꺼짐: predict 1회, 결과의 rot_infer_ms·source는 None(지금과 같음)."""
        det, _, job = self.ready(rot_aware([(10, 10, 60, 20)], [(100, 50, 110, 100)]))
        _, res = det(job)

        self.assertEqual(len(self.model.calls), 1)
        self.assertIsNone(res.rot_infer_ms)
        self.assertEqual([i.source for i in res.instances], [None])

    def test_on_predicts_rotated_crop_and_merges(self):
        """켜짐: 두 번째 predict는 시계 방향으로 돌린 crop을 같은 인자로 받고, 돌린 쪽 새 볼트가 원래 좌표로 더해진다."""
        det, _, job = self.ready(rot_aware([(10, 10, 60, 20)], [(10, 10, 60, 20), (100, 50, 110, 100)],
                                           plain_conf=[0.9], rot_conf=[0.5, 0.8]),
                                 cfg_over={"rotate90_pass": ROT})
        _, res = det(job)

        (c0, _, k0, _), (c1, _, k1, _) = self.model.calls
        self.assertTrue(np.array_equal(c1, cv2.rotate(c0, cv2.ROTATE_90_CLOCKWISE)))
        self.assertEqual(k0, k1)

        self.assertEqual([i.source for i in res.instances], [None, SOURCE_ROT90])
        # 가짜 결과의 상자는 「끝 포함 화소」 좌표라 경계 좌표 변환에서 y가 1 밀린다(실제 모델 상자는 경계 좌표)
        self.assertEqual(res.instances[1].box_xyxy, (100.0, 51.0, 110.0, 101.0))
        pts = {(int(x), int(y)) for x, y in res.instances[1].polygon}
        self.assertTrue({(100, 50), (110, 100)} <= pts)
        self.assertIsNotNone(res.rot_infer_ms)
        self.assertGreaterEqual(res.infer_ms, res.rot_infer_ms)

    def test_every_n_alternates(self):
        """every_n 2: 첫 호출·셋째 호출만 돌린 추론을 하고, 건너뛴 호출은 rot_infer_ms가 None이다."""
        det, _, job = self.ready(rot_aware([], [(100, 50, 110, 100)]),
                                 cfg_over={"rotate90_pass": {**ROT, "every_n": 2}})
        counts, rot_ms = [], []

        for _ in range(3):
            before = len(self.model.calls)
            _, res = det(job)
            counts.append(len(self.model.calls) - before)
            rot_ms.append(res.rot_infer_ms is not None)

        self.assertEqual((counts, rot_ms), ([2, 1, 2], [True, False, True]))

    def test_roi_crop_rotation(self):
        """ROI가 있으면 ROI crop을 돌리고, 되돌린 인스턴스에 ROI 오프셋이 더해진다."""
        det, _, job = self.ready(rot_aware([], [(5, 5, 30, 10)]),
                                 cfg_over={"rotate90_pass": ROT, "roi": [100, 50, 64, 48]})
        _, res = det(job)

        self.assertEqual(self.model.calls[1][0].shape, (64, 48, 3))
        self.assertEqual(res.instances[0].box_xyxy, (105.0, 56.0, 130.0, 61.0))   # 끝 포함 상자 → y 1 밀림(위 시험 참고)
        pts = {(int(x), int(y)) for x, y in res.instances[0].polygon}
        self.assertTrue({(105, 55), (130, 60)} <= pts)


# ---------------------------------------------------------------------- 기록


def run_job(cfg, display):
    """보정 있는 파이프라인에서 작업 하나를 표시 결과로 끝낸 결과(JobOutcome)."""
    from test_ai_axis_path import CAL

    fr = frame(0, image=np.zeros((CH, CW, 3), np.uint8), width=CW, height=CH, calibration_id=CAL)
    p = VisionPipeline(settings(detector_config=cfg, profile=profile(width=CW, height=CH, calibration_id=CAL)),
                       run_id=RUN_ID, clock_ns=FakeClock())
    p.offer_frame(fr)
    job = p.start_next()
    res = display(cfg, job)

    return p.complete(job, observation_from_ai(job, res), display_result=res)


# 기존 기록의 키(#429 전): 꺼짐이면 정확히 이것만 있어야 한다
_PROCESS_KEYS = {"run_id", "source_id", "image_seq", "capture_generation", "generation", "job_seq", "mode",
                 "timestamp_ms", "timestamp_base", "calibration_id", "detector_config_id", "reprocess", "status",
                 "valid", "reason", "reason_owner", "candidate_count", "process_ms", "command_id", "eval_run_id",
                 "method", "model_id", "infer_ms", "ai_instances", "axis_valid_count"}
_SUMMARY_KEYS = {"index", "class_index", "class_name", "confidence", "rejected", "clipped", "area_px",
                 "axis_endpoints_px", "axis_reason", "axis_max_iou", "axis_diag", "axis_overlap_with",
                 "axis_candidate_reason"}


class RecordTest(unittest.TestCase):
    """process 기록·run_meta: 꺼짐이면 키가 지금과 같고, 켜짐이면 출처·돌린 추론 시간이 더해진다."""

    def test_off_record_keys_unchanged(self):
        """꺼짐: process 기록·인스턴스 요약·run_meta 검출 묶음에 새 키가 없다."""
        cfg = ai_config(axis_extractor="m2", min_mask_area_px=0)
        outcome = run_job(cfg, lambda c, job: convert(out([m(20, 50, 80, 60)]), job.frame, c))
        rec = _plain(process_record(outcome))

        self.assertEqual(set(rec), _PROCESS_KEYS)
        self.assertEqual(set(rec["ai_instances"][0]), _SUMMARY_KEYS)
        self.assertNotIn("rotate90_pass", _detector_block(cfg))

    def test_on_record_has_source_and_rot_ms(self):
        """켜짐: 돌린 쪽 인스턴스 요약에만 source, process 기록에 rot_infer_ms, run_meta에 rotate90_pass."""
        cfg = ai_config(axis_extractor="m2", min_mask_area_px=0, rotate90_pass=ROT)
        outcome = run_job(cfg, lambda c, job: convert(out([m(20, 50, 80, 60), m(120, 20, 130, 90)]), job.frame, c,
                                                      sources=(None, SOURCE_ROT90), rot_infer_ms=4.5))
        rec = _plain(process_record(outcome))

        self.assertEqual(rec["rot_infer_ms"], 4.5)
        self.assertNotIn("source", rec["ai_instances"][0])
        self.assertEqual(rec["ai_instances"][1]["source"], SOURCE_ROT90)
        self.assertEqual(_detector_block(cfg)["rotate90_pass"], ROT)

    def test_result_field_validation(self):
        """AiInstance.source는 None·rot90만, AiSegResult.rot_infer_ms는 None·유한 ≥ 0만, convert의 sources 길이는 K."""
        inst = AiInstance(index=0, class_index=0, class_name="bolt", confidence=0.5, box_xyxy=(0, 0, 1, 1),
                          polygon=None, area_px=0.0, component_count=0, hole_count=0, clipped=False, rejected=EMPTY_MASK)

        with self.assertRaises(ValueError):
            AiInstance(**{**inst.__dict__, "source": "rot180"})

        base = dict(instances=(inst,), model_id="m", config_sha256="0" * 64, roi=None, crop_wh=(CW, CH), infer_ms=1.0,
                    post_ms=1.0, process_ms=2.0)

        for bad in (-1.0, float("nan"), True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                AiSegResult(**base, rot_infer_ms=bad)

        cfg = ai_config()
        outcome_frame = frame(0, image=np.zeros((CH, CW, 3), np.uint8), width=CW, height=CH)

        with self.assertRaises(ValueError):
            convert(out([m(1, 1, 5, 5)]), outcome_frame, cfg, sources=(None, None))


# ---------------------------------------------------------------------- 별도 프로세스


class ProcessTest(ProcessBase):
    """추론 별도 프로세스: 돌린 추론 설정이 자식에서 같은 결과를 낸다."""

    def test_rot90_same_result_as_in_process(self):
        """켜짐 설정으로 자식 프로세스 결과(출처 포함)가 같은 프로세스의 AiSegDetector와 같다."""
        import ai_process_fake as fake

        path = self.write_cfg(name="rot.json", weights_path=self.cfg.weights_path,
                              weights_sha256=self.cfg.weights_sha256, rotate90_pass=ROT)
        cfg = load_ai_config(path)
        local = fake.make(cfg)
        local.prepare()
        px = self.proxy(cfg=cfg, cfg_path=path)
        px.prepare()
        _, job = self.job(image(pos=0), cfg=cfg)

        _, res_l = local(job)
        _, res_p = px(job, CancelToken())

        self.assertEqual([(i.class_name, i.source, i.box_xyxy) for i in res_p.instances],
                         [(i.class_name, i.source, i.box_xyxy) for i in res_l.instances])
        self.assertIn(SOURCE_ROT90, [i.source for i in res_p.instances])
        self.assertIsNotNone(res_p.rot_infer_ms)


if __name__ == "__main__":
    unittest.main()
