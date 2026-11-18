"""#174 전환 단계 T: 볼트 축 경로의 물체별 후보 자격(R-1~R-15)과 저장·실시간 시험 경로의 B 제출/실제 호출 구분.

근거: `_workspace/174_builder_a_multi_object_candidate_plan.md` r2 §5(단계 T) · r1 §2·§3·§5 표, reviewer #174 구현 착수 명세.

- 마스크는 numpy로 그린 합성 볼트(`test_ai_axis_path.bolt_mask`)·직사각형이며 정답을 같은 추출기로 역산하지 않는다.
- 이 시험의 「유효」는 추출기 형식 조건 통과이고 정확도 판정이 아니다. B 결과의 값(규격·길이)은 보지 않고 호출 수만 본다.
- 실제 ROS·Gazebo·모델을 쓰지 않는다. 저장 경로는 실제 `BStage`(고정 TF `StaticTfLookup`), 실시간 경로는 디스패처 하네스(가짜 TF·B).
"""
import itertools
import json
import unittest

import numpy as np

from eyetoarm.transforms.b_stage import BStage, StaticTfLookup
from eyetoarm.vision.ai_convert import (MULTIPLE_CANDIDATES, MULTIPLE_OBJECTS, axis_selection, convert,
                                        observation_from_ai, overlaps)
from eyetoarm.vision.ai_result import AiSegResult, axis_candidate_reason
from eyetoarm.vision.pipeline import VisionPipeline
from eyetoarm.vision.pipeline_records import _instance_summary, _plain, process_record
from eyetoarm.vision.pipeline_runner import b_admit
from test_a8_axis_to_b import A8Base, a8_frame, a8_settings
from test_ai_axis_path import CAL, H, W, axis_cfg, bolt_mask, run_job
from test_ai_config import ai_config
from test_ai_convert import raw
from test_ai_result import instance
from test_b_transforms import CAM, ROOT, T_W_C
from test_pipeline import RUN_ID, FakeClock, profile, settings


def rect(r0, r1, c0, c1, shape=(H, W)):
    """직사각형 마스크(머리·축 구분이 없어 M1이 AXIS_HEAD_AMBIGUOUS를 낸다)."""
    m = np.zeros(shape, bool)
    m[r0:r1, c0:c1] = True

    return m


def frag_bolt():
    """조각난 볼트(MASK_FRAGMENTED)."""
    m = bolt_mask(y=60)
    m[10:14, 150:160] = True

    return m


# 시나리오: (마스크, 클래스, 기대 유효 후보 마스크 번호 목록). 번호는 마스크 목록 순서(순열 시험에서 물체 식별에 씀)
SCENARIOS = {
    "R1_normal_plus_invalid": ([bolt_mask(y=30), rect(85, 95, 40, 90)], None, [0]),
    "R2_clipped_plus_normal": ([bolt_mask(x0=0, y=30), bolt_mask(y=90)], None, [1]),
    "R3_pair_plus_independent": ([bolt_mask(y=30), bolt_mask(x0=42, y=30), bolt_mask(y=90)], None, [2]),
    "R4_two_valid": ([bolt_mask(y=30), bolt_mask(y=90)], None, [0, 1]),
    "R5_zero_valid": ([frag_bolt(), rect(95, 105, 40, 90)], None, []),
    "R13_clipped_overlap_plus_independent": ([bolt_mask(x0=0, y=30), bolt_mask(x0=4, y=30), bolt_mask(y=90)], None, [2]),
}


def outcome_of(masks, cls=None, *, cfg=None, prof=True):
    """작업 하나를 코어에 완료시켜 (pipeline, result, outcome)."""
    cfg = cfg or axis_cfg()
    fr = a8_like_small_frame()
    prof_obj = profile(width=W, height=H, calibration_id=CAL) if prof else None
    p = VisionPipeline(settings(detector_config=cfg, profile=prof_obj), run_id=RUN_ID, clock_ns=FakeClock())
    p.offer_frame(fr)
    job = p.start_next()
    result = convert(raw(masks, (W, H), cls=cls), job.frame, cfg)

    return p, result, p.complete(job, observation_from_ai(job, result), display_result=result)


def a8_like_small_frame():
    """200×120 프레임(보정 번호 CAL)."""
    from test_pipeline import frame

    return frame(0, image=np.zeros((H, W, 3), np.uint8), width=W, height=H, calibration_id=CAL)


class CandidateScenarioTest(unittest.TestCase):
    """R-1~R-5·R-13: 물체별 제외와 프레임 결과."""

    def check(self, name):
        """시나리오 하나: 유효 후보 index·관측·선택을 확인하고 (result, obs)를 돌려준다."""
        masks, cls, expect = SCENARIOS[name]
        _, result, obs = run_job(axis_cfg(), masks, cls=cls)

        self.assertEqual(result.axis_valid_indices, tuple(expect), name)
        self.assertEqual(result.axis_valid_count, len(expect))

        if len(expect) == 1:
            self.assertEqual((obs.valid, obs.reason, obs.selected_index, obs.observed_face),
                             (True, None, expect[0], "axis"))
            np.testing.assert_array_equal(obs.axis_endpoints_px, result.instances[expect[0]].axis_endpoints_px)
        else:
            self.assertEqual((obs.valid, obs.selected_index, obs.axis_endpoints_px), (False, None, None))

        self.assertEqual(obs.candidate_count, result.candidate_count)   # 의미 유지(in_scope 수)

        return result, obs

    def test_r1_normal_plus_invalid(self):
        """정상 1 + 머리 모호 1: 정상 볼트로 기존 단일 관측, 끝점은 단독 실행과 같고 무효 쪽은 자기 사유로만 빠진다."""
        result, obs = self.check("R1_normal_plus_invalid")
        _, _, alone = run_job(axis_cfg(), [bolt_mask(y=30)])

        np.testing.assert_array_equal(obs.axis_endpoints_px, alone.axis_endpoints_px)
        self.assertEqual(axis_candidate_reason(result.instances[1]), "AXIS_HEAD_AMBIGUOUS")

    def test_r2_clipped_plus_normal(self):
        """잘림 1 + 정상 1: 잘린 볼트만 CONTOUR_CLIPPED로 빠지고 정상 볼트가 관측된다(예전에는 프레임 전체 무효)."""
        result, _ = self.check("R2_clipped_plus_normal")

        self.assertEqual(axis_candidate_reason(result.instances[0]), "CONTOUR_CLIPPED")
        self.assertIsNone(result.instances[0].axis_overlap_with)

    def test_r3_pair_plus_independent(self):
        """겹친 쌍 + 독립 정상 1: 쌍 양쪽은 MULTIPLE_CANDIDATES(상대 index 기록), 독립 볼트가 관측된다."""
        result, _ = self.check("R3_pair_plus_independent")
        a, b, c = result.instances

        self.assertEqual((a.axis_overlap_with, b.axis_overlap_with, c.axis_overlap_with), ((1,), (0,), None))
        self.assertEqual([axis_candidate_reason(i) for i in (a, b)], [MULTIPLE_CANDIDATES] * 2)
        self.assertIsNotNone(a.axis_endpoints_px)    # 끝점은 진단으로 남지만 관측에 쓰이지 않는다

    def test_r4_two_valid(self):
        """유효 2: MULTIPLE_OBJECTS 무효, 임의 선택 없음, 두 후보 끝점은 기록에 남는다."""
        result, obs = self.check("R4_two_valid")

        self.assertEqual(obs.reason, MULTIPLE_OBJECTS)
        self.assertTrue(all(_instance_summary(i)["axis_endpoints_px"] is not None for i in result.instances))

    def test_r5_zero_valid(self):
        """유효 0(서로 다른 M1 실패 2): 무효, 대표 사유는 가장 작은 index의 추출기 사유, 후보별 사유 2개."""
        result, obs = self.check("R5_zero_valid")

        self.assertEqual([axis_candidate_reason(i) for i in result.instances], ["MASK_FRAGMENTED", "AXIS_HEAD_AMBIGUOUS"])
        self.assertEqual(obs.reason, "MASK_FRAGMENTED")

    def test_r13_clipped_overlap_plus_independent(self):
        """잘린 A와 겹친 B는 보류(A: CONTOUR_CLIPPED, B: MULTIPLE_CANDIDATES), 떨어진 C는 독립 후보로 남는다."""
        result, _ = self.check("R13_clipped_overlap_plus_independent")
        a, b, c = result.instances

        self.assertEqual((a.clipped, b.clipped), (True, False))
        self.assertEqual((axis_candidate_reason(a), axis_candidate_reason(b), axis_candidate_reason(c)),
                         ("CONTOUR_CLIPPED", MULTIPLE_CANDIDATES, None))
        self.assertEqual((a.axis_overlap_with, b.axis_overlap_with), ((1,), (0,)))

    def test_zero_valid_representative_order(self):
        """유효 0의 대표 사유는 CONTOUR_CLIPPED > MULTIPLE_CANDIDATES > 추출기 사유 순(기존 어휘 재사용, Q1 (가))."""
        cases = [([bolt_mask(x0=0, y=30), rect(85, 95, 40, 90)], "CONTOUR_CLIPPED"),
                 ([rect(85, 95, 40, 90), bolt_mask(y=30), bolt_mask(x0=42, y=30)], MULTIPLE_CANDIDATES),
                 ([bolt_mask(), bolt_mask(x0=42)], MULTIPLE_CANDIDATES)]

        for masks, reason in cases:
            with self.subTest(reason=reason):
                _, result, obs = run_job(axis_cfg(), masks)
                self.assertEqual((obs.valid, obs.reason, result.axis_valid_count), (False, reason, 0))


class IouBoundaryTest(unittest.TestCase):
    """R-14: 겹침 판정 IoU ≥ 0.5(경계 포함)."""

    def test_predicate(self):
        """술어 단위: 0.5는 겹침, 바로 아래 값은 아님."""
        self.assertTrue(overlaps(0.5))
        self.assertFalse(overlaps(float(np.nextafter(0.5, 0.0))))

    def test_exact_half_raster(self):
        """래스터 IoU가 정확히 0.5(200/400)인 쌍은 양쪽 MULTIPLE_CANDIDATES, 0.463(190/410) 쌍은 겹침 아님."""
        a = rect(55, 65, 40, 70)

        for dx, overlap in ((10, True), (11, False)):
            with self.subTest(dx=dx):
                _, result, obs = run_job(axis_cfg(), [a, rect(55, 65, 40 + dx, 70 + dx)])
                i0, i1 = result.instances
                iou = 200 / 400 if overlap else 190 / 410

                self.assertEqual(i0.axis_max_iou, iou)
                self.assertEqual(i0.axis_overlap_with is not None, overlap)
                expect = MULTIPLE_CANDIDATES if overlap else "AXIS_HEAD_AMBIGUOUS"
                self.assertEqual([axis_candidate_reason(i) for i in (i0, i1)], [expect, expect])
                self.assertEqual((obs.valid, obs.reason), (False, expect))


class PermutationTest(unittest.TestCase):
    """R-6: 인스턴스 순서를 바꿔도 유효 수·valid·B 허용·선택 물체가 같다(대표 사유는 진단용이라 달라도 됨)."""

    def test_all_permutations(self):
        """모든 시나리오의 모든 순열에서 결과가 순서와 무관하다."""
        for name, (masks, cls, expect) in SCENARIOS.items():
            reasons_seen = set()

            for order in itertools.permutations(range(len(masks))):
                with self.subTest(name=name, order=order):
                    p, result, out = outcome_of([masks[k] for k in order])
                    obs = out.observation
                    chosen = sorted(order[i] for i in result.axis_valid_indices)    # 원래 마스크 번호로

                    self.assertEqual(chosen, expect)
                    self.assertEqual(obs.valid, len(expect) == 1)
                    self.assertEqual(b_admit(p, out), len(expect) == 1)

                    if len(expect) == 1:
                        self.assertEqual(order[obs.selected_index], expect[0])

                    reasons_seen.add(obs.reason)
                    self.assertEqual(sorted(str(axis_candidate_reason(i)) for i in result.instances),
                                     sorted(str(axis_candidate_reason(i)) for i in run_job(axis_cfg(), masks)[1].instances))

            if name == "R5_zero_valid":       # 대표 사유는 순서에 따라 달라진다(진단용)
                self.assertEqual(reasons_seen, {"MASK_FRAGMENTED", "AXIS_HEAD_AMBIGUOUS"})


class DeliveryBlockTest(unittest.TestCase):
    """R-7: 픽셀 후보 수 ≠ 전달 가능 수(보정 불일치·stale)."""

    def test_calibration_mismatch_pixel_one_deliver_zero(self):
        """보정 불일치: 픽셀 후보 1이어도 관측은 CALIBRATION_MISMATCH 무효이고 B로 가지 않는다."""
        p, result, out = outcome_of(SCENARIOS["R1_normal_plus_invalid"][0], prof=False)

        self.assertEqual(result.axis_valid_count, 1)
        self.assertEqual((out.observation.valid, out.observation.reason), (False, "CALIBRATION_MISMATCH"))
        self.assertFalse(b_admit(p, out))

    def test_stale_live_frame_pixel_one_deliver_zero(self):
        """다중 물체 중 유효 1개 관측도 실시간 소스가 현재 아님(stale epoch)이면 B 허용 거짓, 현재면 참."""
        p, result, out = outcome_of(SCENARIOS["R3_pair_plus_independent"][0])

        class Live:
            """frame_current 값을 정해 둔 실시간 소스 흉내."""

            def __init__(self, current):
                """현재 여부."""
                self.current = current

            def frame_current(self, f):
                """정해 둔 값."""
                return self.current

        self.assertEqual(result.axis_valid_count, 1)
        self.assertFalse(b_admit(p, out, Live(False)))
        self.assertTrue(b_admit(p, out, Live(True)))

    def test_generation_change_discards(self):
        """작업 중 세대가 바뀌면 유효 1개 관측도 현재 관측으로 반영되지 않아 B로 가지 않는다."""
        from eyetoarm.vision.pipeline import GenerationCause

        cfg = axis_cfg()
        p = VisionPipeline(settings(detector_config=cfg, profile=profile(width=W, height=H, calibration_id=CAL)),
                           run_id=RUN_ID, clock_ns=FakeClock())
        p.offer_frame(a8_like_small_frame())
        job = p.start_next()
        result = convert(raw(SCENARIOS["R1_normal_plus_invalid"][0], (W, H)), job.frame, cfg)
        p.change_settings(settings(detector_config=cfg, profile=profile(width=W, height=H, calibration_id="cal-other")),
                          cause=GenerationCause.PROFILE_CHANGED)
        out = p.complete(job, observation_from_ai(job, result), display_result=result)

        self.assertFalse(b_admit(p, out))


class RecordTest(unittest.TestCase):
    """R-11·R-15: 기록 키 호환과 단일 자격 함수 일치."""

    def test_axis_off_records_unchanged_keys(self):
        """축 경로를 끈 설정은 관측이 표시용 무효이고, 처리 기록에 axis_valid_count·axis_* 키가 없다."""
        legacy = {"index", "class_index", "class_name", "confidence", "rejected", "clipped", "area_px"}

        for name, (masks, cls, _) in SCENARIOS.items():
            with self.subTest(name=name):
                p, result, out = outcome_of(masks, cfg=ai_config())
                rec = _plain(process_record(out))

                self.assertFalse(out.observation.valid)
                self.assertNotIn("axis_valid_count", rec)
                self.assertTrue(all(set(i) == legacy for i in rec["ai_instances"]))

    def test_axis_on_record_keys_and_single_function(self):
        """축 경로를 켠 기록: axis_valid_count = 픽셀 후보 수, 인스턴스 axis_candidate_reason이 집계·선택과 일치(R-15)."""
        for name, (masks, cls, expect) in SCENARIOS.items():
            with self.subTest(name=name):
                p, result, out = outcome_of(masks)
                rec = json.loads(json.dumps(_plain(process_record(out))))
                valid_in_rec = [i["index"] for i in rec["ai_instances"]
                                if "axis_candidate_reason" in i and i["axis_candidate_reason"] is None]

                self.assertEqual(rec["axis_valid_count"], len(expect))
                self.assertEqual(tuple(valid_in_rec), result.axis_valid_indices)
                self.assertEqual(rec["candidate_count"], result.candidate_count)
                chosen, _ = axis_selection(result, out.context.frame, out.context.settings.profile)
                self.assertEqual(None if chosen is None else (chosen.index,),
                                 result.axis_valid_indices if len(expect) == 1 else None)


class TypeCheckTest(unittest.TestCase):
    """R-12: 새 필드 검사."""

    def test_overlap_field_validation(self):
        """빈 목록·음수·중복·역순·자기 자신·축 경로 밖 인스턴스의 axis_overlap_with는 ValueError."""
        import dataclasses

        _, result, _ = run_job(axis_cfg(), [bolt_mask()])
        inst = result.instances[0]

        for bad in ((), (-1,), (1, 1), (2, 1), (0,), [True]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                dataclasses.replace(inst, axis_overlap_with=bad)

        with self.assertRaises(ValueError):
            instance(axis_overlap_with=(1,))

    def test_result_requires_symmetric_overlap(self):
        """한쪽만 상대를 가리키거나 없는 index를 가리키면 AiSegResult가 ValueError."""
        import dataclasses

        _, result, _ = run_job(axis_cfg(), [bolt_mask(y=30), bolt_mask(y=90)])
        a, b = result.instances

        with self.assertRaises(ValueError):
            dataclasses.replace(result, instances=(dataclasses.replace(a, axis_overlap_with=(1,)), b))

        with self.assertRaises(ValueError):
            dataclasses.replace(result, instances=(dataclasses.replace(a, axis_overlap_with=(5,)), b))

    def test_reason_function_rejects_non_axis_instance(self):
        """축 경로가 다루지 않은 인스턴스(너트·거부)에 자격 함수를 부르면 ValueError(조용히 유효로 세지 않음)."""
        with self.assertRaises(ValueError):
            axis_candidate_reason(instance())


class Replay173Test(unittest.TestCase):
    """R-9: #173 실기록 유형(볼트 2개 모두 끝점 실패 + 너트 1) — 새 정책에서도 유효 0, B 0."""

    def test_two_failed_bolts_and_nut(self):
        """candidate_count 3 ≠ axis_valid_count 0, 무효, B 허용 거짓, 후보별 사유 2개(대표 사유는 진단용)."""
        nut = rect(20, 35, 170, 185)
        p, result, out = outcome_of([frag_bolt(), nut, rect(95, 105, 40, 90)], cls=[0, 1, 0])

        self.assertEqual((result.candidate_count, result.axis_valid_count), (3, 0))
        self.assertFalse(out.observation.valid)
        self.assertFalse(b_admit(p, out))
        self.assertEqual([axis_candidate_reason(i) for i in result.instances if i.axis_evaluated],
                         ["MASK_FRAGMENTED", "AXIS_HEAD_AMBIGUOUS"])


# ---------------------------------------------------------------- 저장 입력 경로: b_gate 제출 수 vs 실제 BStage 호출 수
BW, BH = CAM.width, CAM.height


def big_bolt(x0, y, shape=(BH, BW), **kw):
    """큰 프레임(기본 1280×720)의 합성 볼트(형상은 bolt_mask와 같음)."""
    m = np.zeros(shape, bool)
    small = bolt_mask(x0=0, y=60, **kw)
    ys, xs = np.nonzero(small)
    m[ys - 60 + y, xs + x0] = True

    return m


def big_rect(y, x0, shape=(BH, BW)):
    """큰 프레임(기본 1280×720)의 머리 모호 직사각형."""
    m = np.zeros(shape, bool)
    m[y - 5:y + 5, x0:x0 + 50] = True

    return m


STORED_PLAN = {   # job_seq → (마스크, 기대 유효 수)
    0: ([big_bolt(600, 300), big_rect(500, 300)], 1),                          # 정상+무효 → B 1
    1: ([big_bolt(600, 300), big_bolt(600, 500)], 2),                          # 유효 2 → B 0
    2: ([big_rect(300, 600), big_rect(500, 600)], 0),                          # 유효 0 → B 0
    3: ([big_bolt(600, 300), big_bolt(602, 300), big_bolt(600, 500)], 1),     # 겹친 쌍+독립 → B 1
}


class ConvertDetector:
    """작업 순번별 마스크로 실제 convert·observation_from_ai를 부르는 저장 경로 검출기 대역."""

    def __init__(self, cfg):
        """설정."""
        self.cfg = cfg
        self.valid_counts = {}

    def __call__(self, job):
        """(관측, 표시 결과)."""
        masks, _ = STORED_PLAN[job.job_seq]
        result = convert(raw(masks, (BW, BH)), job.frame, self.cfg)
        self.valid_counts[job.job_seq] = result.axis_valid_count

        return observation_from_ai(job, result), result


class StoredPathBStageTest(A8Base):
    """R-8 저장: 실제 convert → 코어 → b_gate(제출) → 실제 BStage(고정 TF) 호출."""

    def test_submit_and_bstage_calls(self):
        """유효 1개 프레임만 제출·BStage 호출(2건), 유효 0·2 프레임은 0. 처리 기록의 axis_valid_count는 픽셀 후보 수."""
        cfg = ai_config(axis_extractor="m1")
        det = ConvertDetector(cfg)
        stage = BStage.from_config(ROOT, CAM, StaticTfLookup(T_W_C))
        outputs = []
        stage.on_output = outputs.append
        consumed = []

        def consume(frame, obs):
            """BStage에 넘기기 직전 기록(제출 수)."""
            consumed.append((frame.image_seq, obs.selected_index))
            return stage.consume(frame, obs)

        from test_a8_axis_to_b import frames

        res, stream, err = self.run_with(frames(4), detector=det, sett=a8_settings(detector_config=cfg), b_consumer=consume)

        self.assertEqual(res.exit_code, 0, err.getvalue())
        self.assertEqual(det.valid_counts, {k: v[1] for k, v in STORED_PLAN.items()})
        self.assertEqual(res.b_calls, 2)                      # A → B 제출(b_gate 통과)
        self.assertEqual([s for s, _ in consumed], [0, 3])
        self.assertEqual([i for _, i in consumed], [0, 2])    # 원래 인스턴스 index
        self.assertEqual(len(outputs), 2)                     # 실제 BStage 출력(호출) 수
        procs = self.processes(stream)
        self.assertEqual([r["axis_valid_count"] for r in procs], [1, 2, 0, 1])
        self.assertEqual([r["valid"] for r in procs], [True, False, False, True])
        self.assertEqual([r["reason"] for r in procs], [None, MULTIPLE_OBJECTS, "AXIS_HEAD_AMBIGUOUS", None])


# ---------------------------------------------------------------- 실시간 시험 경로: A 제출 vs TF 통과 vs B 호출
class LivePathCountsTest(unittest.TestCase):
    """R-8 실시간: 디스패처 하네스(가짜 TF·B, 실행기·D13 무관)에서 세 수를 따로 센다."""

    def test_submit_tf_gate_and_b_calls(self):
        """프레임 4개(유효 1·2·0·1) 중 A 제출 2, TF 성공 1(다른 하나는 TF 없음), B 호출 2(TF 없는 것은 SKIP 계산)."""
        from test_live_observation_harness import Harness, make_frame
        from test_observation_envelope import CAM as LCAM

        shape = (LCAM.height, LCAM.width)
        plan = {0: ([big_bolt(100, 60, shape=shape), big_rect(150, 100, shape=shape)], 1),
                1: ([big_bolt(100, 60, shape=shape), big_bolt(100, 150, shape=shape)], 2),
                2: ([big_rect(60, 100, shape=shape), big_rect(150, 100, shape=shape)], 0),
                3: ([big_bolt(100, 60, shape=shape), big_bolt(102, 60, shape=shape), big_bolt(100, 150, shape=shape)], 1)}

        class TfFailHarness(Harness):
            """image_seq 3의 TF 조회를 실패시킨다."""

            def fake_lookup(self, parent, child, stamp_ns, timeout_s):
                """seq 3의 stamp면 예외(unavailable)."""
                if stamp_ns == make_frame(3).ros_stamp_ns:
                    raise LookupError("no transform at capture stamp")
                return super().fake_lookup(parent, child, stamp_ns, timeout_s)

        h = TfFailHarness(writer=False)
        h.disp.start()
        cfg = ai_config(axis_extractor="m1")
        p = VisionPipeline(settings(detector_config=cfg, profile=profile(width=LCAM.width, height=LCAM.height,
                                                                          calibration_id=LCAM.calibration_id)),
                           run_id=RUN_ID, clock_ns=FakeClock())
        done = ("accepted", "not_accepted", "stale_before_b", "b_input_rejected", "stale_after_tf")
        submitted = 0

        for seq, (masks, n_valid) in plan.items():
            f = make_frame(seq)
            p.offer_frame(f)
            job = p.start_next()
            result = convert(raw(masks, (LCAM.width, LCAM.height)), job.frame, cfg)
            out = p.complete(job, observation_from_ai(job, result), display_result=result)

            self.assertEqual(result.axis_valid_count, n_valid)

            if b_admit(p, out):        # A 제출 조건(실행기 링크는 b_skip_reason None일 때만 제출)
                t = h.state.token
                self.assertTrue(h.disp.submit(f, out.observation, rx_epoch=t.rx_epoch, generation=t.generation,
                                              calibration_id=f.calibration_id))
                submitted += 1
                self.assertTrue(h.wait_counts(lambda c, n=submitted: sum(c.get(k, 0) for k in done) >= n))

        h.finish()
        calls = h.stages[0].calls

        self.assertEqual(submitted, 2)                                         # A 제출
        self.assertEqual(sum(1 for _, T in calls if T is not None), 1)        # TF 게이트 통과
        self.assertEqual([s for s, _ in calls], [0, 3])                        # 실제 B 호출(TF 없는 3은 SKIP 계산)


if __name__ == "__main__":
    unittest.main()
