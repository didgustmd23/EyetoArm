"""같은 볼트 2중 검출 합치기(#292) 시험: 거의 같은 마스크는 conf 최고 하나만 남기고, 길이 차·끝점 없음·서로 다른 볼트는 그대로 둔다.

마스크는 합성 볼트(가로·세로)다. 「같은 볼트」는 IoU ≥ 0.9·둘 다 끝점·끝점 차 ≤ 3 px(묶음의 모든 쌍)일 때뿐이다.
"""
import unittest

import numpy as np

from eyetoarm.vision.ai_convert import (AXIS_DUP_EP_PX, AXIS_DUP_IOU, MULTIPLE_CANDIDATES, MULTIPLE_OBJECTS,
                                        axis_selection, convert, duplicate_groups, endpoint_gap_px,
                                        list_candidate_indices, observation_from_ai)
from eyetoarm.vision.ai_result import DUPLICATE_MASK, axis_candidate_reason
from eyetoarm.vision.pipeline import VisionPipeline
from eyetoarm.vision.pipeline_records import _detector_block, _instance_summary, process_record
from eyetoarm.vision.world_tracking import world_inputs
from test_ai_axis_candidates import a8_like_small_frame
from test_ai_axis_path import CAL, H, W, axis_cfg, bolt_mask
from test_ai_config import ai_config
from test_ai_convert import raw
from test_ai_result import instance
from test_b_transforms import CAM, T_W_C
from test_pipeline import RUN_ID, FakeClock, profile, settings


def vbolt(x=100, y0=20, head_len=10, head_w=18, shaft_len=34, shaft_w=10):
    """세로 볼트 합성 마스크(머리가 위)."""
    m = np.zeros((H, W), bool)
    m[y0:y0 + head_len, x - head_w // 2:x - head_w // 2 + head_w] = True
    m[y0 + head_len:y0 + head_len + shaft_len, x - shaft_w // 2:x - shaft_w // 2 + shaft_w] = True

    return m


def run(masks, conf=None):
    """작업 하나를 끝까지: (pipeline, result, outcome)."""
    cfg = axis_cfg()
    p = VisionPipeline(settings(detector_config=cfg, profile=profile(width=W, height=H, calibration_id=CAL)),
                       run_id=RUN_ID, clock_ns=FakeClock())
    p.offer_frame(a8_like_small_frame())
    job = p.start_next()
    result = convert(raw(masks, (W, H), conf=conf), job.frame, cfg)

    return job, result, p.complete(job, observation_from_ai(job, result), display_result=result)


def mask_iou(a, b):
    """두 마스크 IoU(시험 전제 확인용)."""
    return np.logical_and(a, b).sum() / np.logical_or(a, b).sum()


class ConvertTest(unittest.TestCase):
    """convert 끝까지: 대표 선택·거부 표시·관측·목록."""

    def test_identical_masks_keep_highest_conf(self):
        """같은 마스크 둘(conf 0.76/0.38, #290 D1_c 모양): 높은 쪽만 유효, 낮은 쪽은 DUPLICATE_MASK·범위 밖·축 비움."""
        job, result, outcome = run([bolt_mask(y=30), bolt_mask(y=30)], conf=[0.76, 0.38])
        keep, drop = result.instances

        self.assertEqual(result.axis_valid_indices, (0,))
        self.assertIsNone(axis_candidate_reason(keep))
        self.assertIsNone(keep.axis_overlap_with)
        self.assertEqual(keep.axis_diag["duplicates"], [{"index": 1, "iou": 1.0, "ep_px": 0.0}])
        self.assertEqual((drop.rejected, drop.in_scope, drop.axis_evaluated, drop.axis_overlap_with),
                         (DUPLICATE_MASK, False, False, None))
        self.assertEqual(drop.axis_diag, {"duplicate_of": 0, "iou": 1.0, "ep_px": 0.0})
        self.assertIsNotNone(drop.polygon)   # 기하는 기록·표시용으로 남는다

        obs = outcome.observation
        self.assertEqual((obs.valid, obs.reason, obs.selected_index), (True, None, 0))

    def test_reversed_conf_keeps_other(self):
        """conf가 뒤 index가 높으면 그쪽을 남긴다."""
        _, result, _ = run([bolt_mask(y=30), bolt_mask(y=30)], conf=[0.38, 0.76])
        self.assertEqual(result.axis_valid_indices, (1,))
        self.assertEqual(result.instances[0].rejected, DUPLICATE_MASK)

    def test_one_px_shift_merged_two_px_not(self):
        """1 px 어긋난 쌍(IoU 0.93)은 합치고, 2 px(IoU 0.87, 기존 R3 장면)는 지금처럼 둘 다 MULTIPLE_CANDIDATES."""
        a, b1, b2 = bolt_mask(y=30), bolt_mask(x0=41, y=30), bolt_mask(x0=42, y=30)
        self.assertGreater(mask_iou(a, b1), AXIS_DUP_IOU)
        self.assertLess(mask_iou(a, b2), AXIS_DUP_IOU)

        _, merged, _ = run([a, b1], conf=[0.5, 0.6])
        _, kept, _ = run([a, b2], conf=[0.5, 0.6])
        self.assertEqual(merged.axis_valid_indices, (1,))
        self.assertEqual([axis_candidate_reason(i) for i in kept.instances], [MULTIPLE_CANDIDATES] * 2)

    def test_longer_duplicate_stays_invalid(self):
        """한쪽이 6 px 긴 같은 자리 볼트(길이 모호, #274 D13 모양)는 합치지 않고 둘 다 무효."""
        a, b = bolt_mask(y=30), bolt_mask(y=30, shaft_len=40)
        self.assertGreaterEqual(mask_iou(a, b), 0.5)

        _, result, outcome = run([a, b], conf=[0.9, 0.3])
        self.assertEqual([axis_candidate_reason(i) for i in result.instances], [MULTIPLE_CANDIDATES] * 2)
        self.assertEqual(outcome.observation.reason, MULTIPLE_CANDIDATES)

    def test_pair_plus_independent_gives_two_candidates(self):
        """중복 쌍 + 떨어진 볼트: 유효 2(단계 T는 MULTIPLE_OBJECTS, 단계 L 목록은 둘 다)."""
        job, result, outcome = run([bolt_mask(y=30), bolt_mask(y=30), bolt_mask(y=90)], conf=[0.8, 0.3, 0.9])

        self.assertEqual(result.axis_valid_indices, (0, 2))
        self.assertEqual(outcome.observation.reason, MULTIPLE_OBJECTS)
        self.assertEqual(list_candidate_indices(result, job.frame, job.settings.profile), (0, 2))

    def test_three_identical_keep_one(self):
        """같은 마스크 셋: conf 최고 하나만 남고 나머지 둘이 DUPLICATE_MASK."""
        _, result, _ = run([bolt_mask(y=30)] * 3, conf=[0.4, 0.9, 0.6])
        self.assertEqual(result.axis_valid_indices, (1,))
        self.assertEqual([i.rejected for i in result.instances], [DUPLICATE_MASK, None, DUPLICATE_MASK])
        self.assertEqual([d["index"] for d in result.instances[1].axis_diag["duplicates"]], [0, 2])

    def test_clipped_duplicate_stays_invalid(self):
        """잘린 짝(끝점 없음)은 합치지 않는다: 잘림은 CONTOUR_CLIPPED, 짝은 MULTIPLE_CANDIDATES(안 D는 후속)."""
        _, result, _ = run([bolt_mask(x0=0, y=30), bolt_mask(x0=0, y=30)], conf=[0.9, 0.3])
        self.assertEqual([axis_candidate_reason(i) for i in result.instances], ["CONTOUR_CLIPPED"] * 2)
        self.assertTrue(all(i.rejected is None for i in result.instances))


class DistinctBoltsTest(unittest.TestCase):
    """서로 다른 볼트(맞닿음·평행·끝점 공유·한 마스크가 둘을 덮음)는 합치지 않는다. 실제 장면 기록은 없다(합성만)."""

    def test_touching_parallel_bolts_both_valid(self):
        """평행하게 맞닿은 두 볼트(축 사이 14 px, 머리 경계 접촉): 겹침도 합치기도 없이 둘 다 유효."""
        a, b = bolt_mask(y=30, head_w=14, shaft_w=6), bolt_mask(y=44, head_w=14, shaft_w=6)
        self.assertEqual(np.logical_and(a, b).sum(), 0)
        self.assertTrue(np.logical_and(np.roll(a, 1, axis=0), b).any())   # 한 줄 밀면 닿는다(맞닿음)
        _, result, _ = run([a, b])
        self.assertEqual(result.axis_valid_indices, (0, 1))
        self.assertTrue(all(i.rejected is None for i in result.instances))

    def test_mask_covering_two_bolts_not_chosen(self):
        """두 볼트를 덮는 큰 마스크(conf 높음) + 그중 한 볼트 마스크: IoU < 0.9라 합치지 않고 둘 다 무효."""
        one, other = bolt_mask(y=30, head_w=14, shaft_w=6), bolt_mask(y=44, head_w=14, shaft_w=6)
        cover = one | other
        cover[36:38, 40:50] = True   # 두 머리 사이를 이어 한 덩어리 마스크로
        self.assertGreaterEqual(mask_iou(one, cover), 0.5)
        self.assertLess(mask_iou(one, cover), AXIS_DUP_IOU)

        _, result, _ = run([cover, one], conf=[0.95, 0.5])
        self.assertTrue(all(i.rejected != DUPLICATE_MASK for i in result.instances))
        self.assertEqual(result.axis_valid_indices, ())

    def test_shared_endpoint_crossing_bolts(self):
        """끝점(머리) 자리를 나누는 가로·세로 볼트: 마스크가 일부만 겹쳐 합치지 않는다."""
        h, v = bolt_mask(x0=60, y=30), vbolt(x=65, y0=21)
        self.assertLess(mask_iou(h, v), AXIS_DUP_IOU)

        _, result, _ = run([h, v], conf=[0.9, 0.8])
        self.assertTrue(all(i.rejected != DUPLICATE_MASK for i in result.instances))


def bolt_inst(index, endpoints, overlap, conf=0.9, clipped=False, reason=None):
    """축 경로를 거친 인스턴스(끝점 또는 사유)."""
    kw = {"axis_endpoints_px": None if reason else np.array(endpoints, float), "axis_reason": reason}

    if clipped:
        kw.update(clipped=True, axis_endpoints_px=None, axis_reason="CONTOUR_CLIPPED")

    return instance(index, confidence=conf, axis_overlap_with=tuple(overlap), **kw)


EP = [[10.0, 10.0], [40.0, 10.0]]


class GroupRuleTest(unittest.TestCase):
    """`duplicate_groups` 문턱·묶음 규칙."""

    def pair(self, iou, shift_px, conf=(0.9, 0.5)):
        """두 인스턴스 + 쌍 IoU."""
        insts = [bolt_inst(0, EP, (1,), conf[0]),
                 bolt_inst(1, [[10.0 + shift_px, 10.0], [40.0 + shift_px, 10.0]], (0,), conf[1])]
        return duplicate_groups(insts, {(0, 1): iou})

    def test_thresholds_inclusive(self):
        """IoU 0.9·끝점 차 3.0 px는 합치고(경계 포함), 0.8999·3.01 px는 합치지 않는다."""
        self.assertEqual(self.pair(AXIS_DUP_IOU, AXIS_DUP_EP_PX), [(0, [{"index": 1, "iou": 0.9, "ep_px": 3.0}])])
        self.assertEqual(self.pair(AXIS_DUP_IOU - 1e-4, 0.0), [])
        self.assertEqual(self.pair(0.99, AXIS_DUP_EP_PX + 0.01), [])

    def test_tie_keeps_smaller_index(self):
        """conf가 같으면 작은 index."""
        self.assertEqual(self.pair(0.95, 0.0, conf=(0.5, 0.5))[0][0], 0)

    def test_chain_requires_all_pairs(self):
        """사슬 묶음(0–1, 1–2 겹침): 0–2 쌍도 조건을 만족해야 합친다. 하나라도 어긋나면 묶음 전체 그대로."""
        insts = [bolt_inst(0, EP, (1,), 0.5), bolt_inst(1, EP, (0, 2), 0.9), bolt_inst(2, EP, (1,), 0.7)]
        self.assertEqual(duplicate_groups(insts, {(0, 1): 0.95, (1, 2): 0.95}), [])   # 0–2 IoU 없음(0)
        out = duplicate_groups(insts, {(0, 1): 0.95, (1, 2): 0.95, (0, 2): 0.92})
        self.assertEqual([(k, [d["index"] for d in ds]) for k, ds in out], [(1, [0, 2])])

    def test_member_without_endpoints_blocks(self):
        """묶음에 끝점 없는(추출기 사유·잘림) 인스턴스가 있으면 합치지 않는다."""
        for bad in (bolt_inst(1, EP, (0,), reason="AXIS_HEAD_AMBIGUOUS"), bolt_inst(1, EP, (0,), clipped=True)):
            with self.subTest(reason=bad.axis_reason):
                self.assertEqual(duplicate_groups([bolt_inst(0, EP, (1,)), bad], {(0, 1): 1.0}), [])

    def test_endpoint_gap_order_free(self):
        """끝점 차는 순서 무관(머리·끝 바뀐 쌍도 같은 값)."""
        self.assertEqual(endpoint_gap_px(EP, EP[::-1]), 0.0)
        self.assertAlmostEqual(endpoint_gap_px(EP, [[11.0, 10.0], [42.0, 10.0]]), 2.0)


class RecordTest(unittest.TestCase):
    """처리 기록·run_meta·W 추적 입력."""

    def test_process_record_counts_merged(self):
        """합친 수는 process 기록 `axis_duplicates_merged`(>0일 때만, 없으면 키 없음), 요약은 DUPLICATE_MASK·근거."""
        _, _, merged = run([bolt_mask(y=30), bolt_mask(y=30)], conf=[0.76, 0.38])
        _, _, plain = run([bolt_mask(y=30)])
        rec, rec_plain = process_record(merged), process_record(plain)

        self.assertEqual(rec.axis_duplicates_merged, 1)
        self.assertIsNone(rec_plain.axis_duplicates_merged)
        summary = rec.ai_instances[1]
        self.assertEqual((summary["rejected"], "axis_endpoints_px" in summary), (DUPLICATE_MASK, False))
        self.assertEqual(_instance_summary(merged.display_result.instances[0])["axis_candidate_reason"], None)

    def test_run_meta_marks_thresholds_axis_only(self):
        """run_meta 검출 묶음에 문턱 값(축 경로 설정만). 설정 지문(config_id)은 같다."""
        block = _detector_block(axis_cfg())
        self.assertEqual(block["axis_dup"], {"iou": 0.9, "ep_px": 3.0})
        self.assertNotIn("axis_dup", _detector_block(ai_config()))
        self.assertEqual(block["config_id"], axis_cfg().config_sha256)

    def test_world_inputs_skip_duplicate(self):
        """W 추적 입력에서 DUPLICATE_MASK는 not_in_scope(대표만 후보)."""
        _, result, _ = run([bolt_mask(y=30), bolt_mask(y=30)], conf=[0.76, 0.38])
        w = world_inputs(result, K=CAM.K, dist=CAM.dist, T_W_C=T_W_C, plane_z_m=0.105)
        self.assertEqual([(x.candidate, x.untracked_reason) for x in w][1], (False, "not_in_scope"))

    def test_axis_selection_matches(self):
        """axis_selection도 대표 하나를 고른다(같은 자격 함수)."""
        job, result, _ = run([bolt_mask(y=30), bolt_mask(y=30)], conf=[0.3, 0.8])
        chosen, reason = axis_selection(result, job.frame, job.settings.profile)
        self.assertEqual((chosen.index, reason), (1, None))


if __name__ == "__main__":
    unittest.main()
