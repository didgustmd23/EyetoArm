"""#189 단계 L 생성·실행기·기록 시험: 원 인덱스·자격 함수 공유·프레임 차단·0/1/복수/혼합·시작 거부·단계 T 기록 바이트 동일.

근거: 계획 `_workspace/189_builder2_stage_L_plan.md` r0 §3·§5·§6·§7·§8(L-8, L-10~L-19, L-21~L-23),
reviewer #189 계획 검토(b_list_gate = b_gate 공통 조건 + 프로필 적용 + 후보 비어 있지 않음, 보정 불일치 빈 목록, 단계 T 바이트 비교).
"""
from __future__ import annotations

import functools
import io
import itertools
import json
import unittest
from unittest import mock

import numpy as np

from eyetoarm.common.types import CaptureInfo, CaptureTf
from eyetoarm.vision.ai_convert import convert, frame_observations_from_ai, list_candidate_indices, observation_from_ai
from eyetoarm.vision.ai_result import AiJobConfigError
from eyetoarm.vision.pipeline import GenerationCause, VisionPipeline
from eyetoarm.vision.pipeline_records import JsonlRecorder, PipelineLog, _plain, process_record
from eyetoarm.vision.pipeline_runner import b_gate, b_list_gate, run
from test_a8_axis_to_b import a8_settings, frames
from test_ai_axis_candidates import BH, BW, SCENARIOS, STORED_PLAN, a8_like_small_frame, outcome_of
from test_ai_axis_path import CAL, H, W, axis_cfg, bolt_mask
from test_ai_config import ai_config
from test_ai_convert import raw
from test_b_transforms import CAM, T_W_C
from test_pipeline import RUN_ID, FakeClock, profile, settings
from test_run_pipeline import FAST

LIST_KEYS = ("list_candidate_count", "list_indices")


def capture_for(frame, camera=None):
    """프레임 값과 같은 촬영 정보(카메라 프로파일 대역 포함)."""
    cap = CaptureInfo(stamp_ns=frame.ros_stamp_ns, optical_frame=frame.optical_frame,
                      calibration_id=frame.calibration_id, width=frame.width, height=frame.height)
    cam = camera or type("Cam", (), dict(calibration_id=frame.calibration_id, frame_id=frame.optical_frame,
                                         width=frame.width, height=frame.height))()

    return cap, cam


def build(out, **kw):
    """완료 응답 하나에서 목록을 만든다(TF 없음 기본)."""
    ctx = out.context
    cap, cam = capture_for(ctx.frame)

    return frame_observations_from_ai(ctx, out.display_result, capture=cap, camera=cam, tf=kw.pop("tf", None), **kw)


class GenerationTest(unittest.TestCase):
    """L-10·L-12·L-13·L-14~L-18: 원 인덱스, 자격 함수 공유, 프레임 차단, 0/1/복수/혼합."""

    def test_list_matches_stage_t_eligibility(self):
        """모든 시나리오에서 목록 인덱스 = axis_valid_indices(단계 T 자격 함수), 후보 0개면 목록 None."""
        for name, (masks, cls, expect) in SCENARIOS.items():
            with self.subTest(name=name):
                p, result, out = outcome_of(masks)
                fo = build(out)
                idx = list_candidate_indices(result, out.context.frame, out.context.settings.profile)

                self.assertEqual(idx, result.axis_valid_indices)
                self.assertEqual(idx, tuple(expect))
                self.assertEqual(None if fo is None else fo.instance_indices, tuple(expect) or None)
                self.assertEqual(b_list_gate(p, out), bool(expect))

    def test_mixed_failures_keep_normal_two(self):
        """혼합(잘림 1 + 겹친 쌍 + 정상 2 + 머리 모호 1): 목록 = 정상 2개의 원 인덱스, 나머지 4개는 기록 사유만."""
        masks = [bolt_mask(x0=0, y=20), bolt_mask(y=50), bolt_mask(x0=42, y=50), bolt_mask(x0=110, y=20),
                 bolt_mask(x0=110, y=90), np.pad(np.ones((10, 50), bool), ((95, H - 105), (20, W - 70)))]
        p, result, out = outcome_of(masks)
        fo = build(out)
        rec = _plain(process_record(out, b_input="list"))

        self.assertEqual(fo.instance_indices, (3, 4))
        self.assertEqual((rec["list_candidate_count"], rec["list_indices"]), (2, [3, 4]))
        self.assertEqual(sum(1 for i in rec["ai_instances"] if i.get("axis_candidate_reason")), 4)
        self.assertEqual((out.observation.valid, out.observation.reason), (False, "MULTIPLE_OBJECTS"))   # 단계 T 관측 무변
        self.assertFalse(b_gate(p, out))
        self.assertTrue(b_list_gate(p, out))

    def test_original_index_kept_with_nut_and_rejects_first(self):
        """너트·머리 모호 인스턴스가 앞에 있어도 목록 인덱스는 원 순번이고 너트는 목록에 없다(분류 대상은 볼트만)."""
        nut = np.zeros((H, W), bool)
        nut[10:22, 10:22] = True
        masks = [nut, np.pad(np.ones((10, 50), bool), ((95, H - 105), (20, W - 70))), bolt_mask(x0=110, y=40)]
        p, result, out = outcome_of(masks, cls=[1, 0, 0])
        fo = build(out)

        self.assertEqual([i.class_name for i in result.instances], ["nut", "bolt", "bolt"])
        self.assertEqual(fo.instance_indices, (2,))
        self.assertTrue(all(c.class_name == "bolt" and c.observed_face == "axis" for c in fo.candidates))

    def test_calibration_mismatch_builds_empty_list(self):
        """보정 불일치(프로필 없음): 픽셀 후보가 있어도 목록 인덱스는 비고 목록·관문 모두 없다. 기록은 0·[]."""
        p, result, out = outcome_of(SCENARIOS["R4_two_valid"][0], prof=False)
        rec = _plain(process_record(out, b_input="list"))

        self.assertEqual(result.axis_valid_count, 2)
        self.assertEqual(list_candidate_indices(result, out.context.frame, None), ())
        self.assertIsNone(build(out))
        self.assertFalse(b_list_gate(p, out))
        self.assertEqual((rec["axis_valid_count"], rec["list_candidate_count"], rec["list_indices"]), (2, 0, []))

    def test_single_bolt_endpoints_byte_identical_to_stage_t(self):
        """유효 1개: 목록 끝점은 단계 T 관측 끝점과 바이트 동일하고, 원 인스턴스 배열과 메모리를 나누지 않는다."""
        p, result, out = outcome_of(SCENARIOS["R1_normal_plus_invalid"][0])
        fo = build(out)
        cand = fo.candidates[0]

        self.assertEqual(cand.instance_index, out.observation.selected_index)
        self.assertEqual(cand.axis_endpoints_px.tobytes(), out.observation.axis_endpoints_px.tobytes())
        self.assertFalse(np.shares_memory(cand.axis_endpoints_px, result.instances[0].axis_endpoints_px))
        self.assertEqual((cand.class_name, cand.confidence), (result.instances[0].class_name, result.instances[0].confidence))

    def test_frame_fields_and_common_tf(self):
        """L-8: 프레임 공통 값은 작업에서, TF는 목록에 하나뿐이며 후보는 TF를 따로 갖지 않는다."""
        p, result, out = outcome_of(SCENARIOS["R4_two_valid"][0])
        f = out.context.frame
        t = CaptureTf("world", f.optical_frame, f.ros_stamp_ns, T_W_C)
        fo = build(out, tf=t)

        self.assertIs(fo.frame, f)
        self.assertEqual((fo.generation, fo.job_seq, fo.detector_config_id),
                         (out.context.generation, out.context.job_seq, out.context.detector_config_id))
        self.assertEqual(len(fo.candidates), 2)
        self.assertTrue(all(not hasattr(c, "tf") for c in fo.candidates))
        np.testing.assert_array_equal(fo.tf.T_W_C, T_W_C)

    def test_config_and_world_frame_mismatch(self):
        """결과 설정 해시가 작업과 다르면 AiJobConfigError, TF 부모가 world가 아니면 ValueError."""
        p, result, out = outcome_of(SCENARIOS["R4_two_valid"][0])
        f = out.context.frame
        other = convert(raw(SCENARIOS["R4_two_valid"][0], (W, H)), f, axis_cfg(conf_threshold=0.3))
        cap, cam = capture_for(f)

        with self.assertRaises(AiJobConfigError):
            frame_observations_from_ai(out.context, other, capture=cap, camera=cam, tf=None)

        with self.assertRaises(ValueError):
            build(out, tf=CaptureTf("map", f.optical_frame, f.ros_stamp_ns, T_W_C))


class PermutationTest(unittest.TestCase):
    """L-11: 인스턴스 순열을 바꿔도 같은 물체 집합이 실리고 물체–끝점 대응이 같다."""

    def test_all_permutations(self):
        """모든 시나리오·순열에서 목록 물체(원 마스크 번호) 집합과 물체별 끝점이 원래 순서와 같다."""
        for name, (masks, cls, expect) in SCENARIOS.items():
            base = build(outcome_of(masks)[2])
            base_pts = {} if base is None else {c.instance_index: c.axis_endpoints_px.tobytes() for c in base.candidates}

            for order in itertools.permutations(range(len(masks))):
                with self.subTest(name=name, order=order):
                    fo = build(outcome_of([masks[k] for k in order])[2])
                    got = {} if fo is None else {order[c.instance_index]: c.axis_endpoints_px.tobytes()
                                                 for c in fo.candidates}

                    self.assertEqual(sorted(got), expect)
                    self.assertEqual(got, base_pts)

                    if fo is not None:
                        self.assertEqual(list(fo.instance_indices), sorted(fo.instance_indices))


class GateTest(unittest.TestCase):
    """L-19: 세대 변경(stale)·현재 관측 아님은 목록 관문도 닫힌다."""

    def test_generation_change_blocks_list(self):
        """작업 중 세대가 바뀌면 유효 후보 2개여도 목록 관문을 지나지 못한다."""
        cfg = axis_cfg()
        p = VisionPipeline(settings(detector_config=cfg, profile=profile(width=W, height=H, calibration_id=CAL)),
                           run_id=RUN_ID, clock_ns=FakeClock())
        p.offer_frame(a8_like_small_frame())
        job = p.start_next()
        result = convert(raw(SCENARIOS["R4_two_valid"][0], (W, H)), job.frame, cfg)
        before = p.current_generation()
        # 프로필·설정·소스는 그대로 두고 처리 모드만 바꿔 세대만 올린다(관문이 세대 조건으로 닫히는지 본다)
        p.change_settings(settings(mode="planar_other", detector_config=cfg,
                                   profile=profile(width=W, height=H, calibration_id=CAL)),
                          cause=GenerationCause.MODE_CHANGED)
        out = p.complete(job, observation_from_ai(job, result), display_result=result)

        self.assertGreater(p.current_generation(), before)
        self.assertFalse(b_list_gate(p, out))

    def test_not_current_observation_blocks_list(self):
        """다음 작업이 현재 관측을 바꾼 뒤 이전 응답은 목록 관문을 지나지 못한다."""
        cfg = axis_cfg()
        p = VisionPipeline(settings(detector_config=cfg, profile=profile(width=W, height=H, calibration_id=CAL)),
                           run_id=RUN_ID, clock_ns=FakeClock())
        outs = []

        for _ in range(2):
            p.offer_frame(a8_like_small_frame())
            job = p.start_next()
            result = convert(raw(SCENARIOS["R4_two_valid"][0], (W, H)), job.frame, cfg)
            outs.append(p.complete(job, observation_from_ai(job, result), display_result=result))

        self.assertFalse(b_list_gate(p, outs[0]))
        self.assertTrue(b_list_gate(p, outs[1]))

    def test_opencv_result_blocks_list(self):
        """AI 결과가 아닌 응답(표시 결과 없음)은 목록 관문을 지나지 못한다."""
        p, result, out = outcome_of(SCENARIOS["R4_two_valid"][0])
        no_display = type(out)(**{**out.__dict__, "display_result": None})

        self.assertFalse(b_list_gate(p, no_display))


# ---------------------------------------------------------------- 실행기: 저장 입력 단계 L
_CACHE: dict = {}


class CachedDetector:
    """STORED_PLAN 마스크의 convert 결과를 작업 순번별로 한 번만 만들어 재사용한다(실행 사이 기록 바이트 비교용)."""

    def __init__(self, cfg):
        """설정."""
        self.cfg = cfg

    def __call__(self, job):
        """(단계 T 관측, 표시 결과)."""
        key = (self.cfg.config_sha256, job.job_seq)

        if key not in _CACHE:
            _CACHE[key] = convert(raw(STORED_PLAN[job.job_seq][0], (BW, BH)), job.frame, self.cfg)

        return observation_from_ai(job, _CACHE[key]), _CACHE[key]


def run_stage(*, b_input="single", b_list_consumer=None, b_consumer=None, cfg=None, log_b_input=None):
    """프레임 4개(유효 1·2·0·1)를 실행한다. (결과, 기록 문자열, stderr)."""
    cfg = cfg or ai_config(axis_extractor="m1")
    stream, err = io.StringIO(), io.StringIO()
    kw = {} if log_b_input is None else {"b_input": log_b_input}
    log = PipelineLog(JsonlRecorder(stream), **kw)
    res = run(frames(4), a8_settings(detector_config=cfg), log, run_id=RUN_ID, detector=CachedDetector(cfg),
              stderr=err, config=FAST, clock_ns=FakeClock(), b_consumer=b_consumer, b_list_consumer=b_list_consumer)

    return res, stream.getvalue(), err.getvalue()


class FakeListConsumer:
    """시험 전용 가짜 B 목록 소비자: 받은 (context, 표시 결과)로 목록을 만들어 보관한다."""

    def __init__(self):
        """받은 목록."""
        self.lists = []

    def __call__(self, ctx, display):
        """목록 하나를 만들어 보관한다(촬영 정보는 프레임 값, TF는 고정)."""
        f = ctx.frame
        cap, _ = capture_for(f, CAM)
        fo = frame_observations_from_ai(ctx, display, capture=cap, camera=CAM,
                                        tf=CaptureTf("world", f.optical_frame, f.ros_stamp_ns, T_W_C))
        self.lists.append(fo)


class RunnerStageLTest(unittest.TestCase):
    """L-14~L-17 실행기: 목록 소비자 호출 수·길이·원 인덱스, 단일 B 지점은 부르지 않음."""

    def test_list_calls_and_records(self):
        """유효 1·2·0·1 → 목록 호출 3(길이 1·2·1, 원 인덱스 [0]·[0,1]·[2]), b_calls 0, 기록 list 키, 단계 T 관측 무변."""
        fake = FakeListConsumer()
        res, text, err = run_stage(b_list_consumer=fake, log_b_input="list")
        procs = [json.loads(t) for t in text.splitlines() if json.loads(t)["kind"] == "process"]
        meta = json.loads(text.splitlines()[0])

        self.assertEqual(res.exit_code, 0, err)
        self.assertEqual((res.b_list_calls, res.b_calls), (3, 0))
        self.assertEqual([fo.job_seq for fo in fake.lists], [0, 1, 3])
        self.assertEqual([fo.instance_indices for fo in fake.lists], [(0,), (0, 1), (2,)])
        self.assertTrue(all(isinstance(fo.tf, CaptureTf) and fo.tf.stamp_ns == fo.capture.stamp_ns for fo in fake.lists))
        self.assertEqual([r["list_candidate_count"] for r in procs], [1, 2, 0, 1])
        self.assertEqual([r["list_indices"] for r in procs], [[0], [0, 1], [], [2]])
        self.assertEqual([r["axis_valid_count"] for r in procs], [1, 2, 0, 1])
        self.assertEqual([r["valid"] for r in procs], [True, False, False, True])
        self.assertEqual(meta["b_input"], "list")

    def test_list_endpoints_follow_instances(self):
        """목록 끝점은 같은 작업 표시 결과 인스턴스의 끝점과 같다(복사본)."""
        fake = FakeListConsumer()
        run_stage(b_list_consumer=fake, log_b_input="list")

        for fo in fake.lists:
            res = _CACHE[(fo.detector_config_id, fo.job_seq)]

            for c in fo.candidates:
                self.assertEqual(c.axis_endpoints_px.tobytes(), res.instances[c.instance_index].axis_endpoints_px.tobytes())

    def test_two_valid_stage_t_sends_nothing(self):
        """대조: 같은 입력의 단계 T는 유효 2 프레임을 보내지 않는다(b_calls 2 = 유효 1 프레임만)."""
        sent = []
        res, _, err = run_stage(b_consumer=lambda f, o: sent.append(o.selected_index))

        self.assertEqual((res.exit_code, res.b_calls, res.b_list_calls), (0, 2, 0), err)
        self.assertEqual(sent, [0, 2])


class StartRefusalTest(unittest.TestCase):
    """L-23 외: 시작 계약 거부(프레임을 읽기 전, 단계 T로 바꾸지 않음)."""

    def test_both_consumers_refused(self):
        """단계 T·L 소비자를 함께 주면 b_input_contract로 종료 1이고 어느 소비자도 불리지 않는다."""
        calls = []
        res, _, err = run_stage(b_consumer=lambda f, o: calls.append("T"),
                                b_list_consumer=lambda c, d: calls.append("L"))

        self.assertEqual((res.exit_code, res.stop_reason, res.frames, calls), (1, "b_input_contract", 0, []), err)

    def test_axis_extractor_required(self):
        """축 경로를 끈 설정의 단계 L은 b_list_requires_axis로 종료 1."""
        calls = []
        res, _, err = run_stage(b_list_consumer=lambda c, d: calls.append(1), cfg=ai_config())

        self.assertEqual((res.exit_code, res.stop_reason, res.frames, calls), (1, "b_list_requires_axis", 0, []), err)

    def test_live_source_refused(self):
        """D13: 실시간 소스의 단계 L 소비자는 b_list_consumer_not_allowed로 검출 없이 종료 1."""
        import test_pipeline_runner_live as live_mod

        calls = []
        patched = functools.partial(live_mod.run, b_list_consumer=lambda c, d: calls.append(1))

        with mock.patch.object(live_mod, "run", patched):
            live = live_mod.Live(feed=lambda s, c: s.handle_image(live_mod.image_msg(1)))
            res, recs, err = live.run()

        self.assertEqual((res.exit_code, res.stop_reason, live.detect_calls, calls),
                         (1, "b_list_consumer_not_allowed", [], []), err)


class ListRecorderRejectTest(unittest.TestCase):
    """목록 기록기: 메타 없음·대조 실패는 소비자를 부르지 않고 b_input_rejected로 남긴다."""

    def test_missing_meta_and_mismatch_rejected(self):
        """그 프레임 메타가 없거나 카메라가 어긋나면 rejected가 늘고 소비자 호출 0, list_frames.jsonl에 사유가 남는다."""
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        from eyetoarm.vision.capture_list_recorder import CaptureListRecorder

        p, result, out = outcome_of(SCENARIOS["R4_two_valid"][0])
        f = out.context.frame
        called = []
        row = SimpleNamespace(stamp_ns=f.ros_stamp_ns, frame_id=f.optical_frame, T_W_C=None, tf_child=None,
                              tf_stamp_ns=None)
        cam_bad = SimpleNamespace(calibration_id=f.calibration_id, frame_id="other", width=f.width, height=f.height)
        metas = [SimpleNamespace(frames={}, camera=None, world_frame="world"),
                 SimpleNamespace(frames={getattr(f, "source_index", -1): row}, camera=cam_bad, world_frame="world")]

        for k, meta in enumerate(metas):
            with self.subTest(case=k), tempfile.TemporaryDirectory() as d:
                rec = CaptureListRecorder(str(Path(d) / "out"), meta, consumer_kind="test")
                rec.wrap(called.append)(out.context, result)
                rec.close()
                lines = [json.loads(t) for t in (Path(d) / "out" / "list_frames.jsonl").read_text().splitlines()]

                self.assertEqual((rec.rejected, rec.calls, called), (1, 0, []))
                self.assertEqual(lines[0]["kind"], "b_input_rejected")

    def test_existing_output_refused(self):
        """출력 폴더가 이미 있으면 FileExistsError(덮어쓰지 않음), 빈 소비자 종류는 ValueError."""
        import tempfile

        from eyetoarm.vision.capture_list_recorder import CaptureListRecorder

        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):
                CaptureListRecorder(d, None, consumer_kind="test")

            with self.assertRaises(ValueError):
                CaptureListRecorder(d + "/new", None, consumer_kind="")


class StageTBytesTest(unittest.TestCase):
    """L-21·L-22: 단계 T 기본 실행의 기록은 키·바이트가 그대로다(새 비교 시험)."""

    def test_default_and_explicit_single_identical(self):
        """PipelineLog 기본값과 b_input="single"의 기록은 바이트 동일하고 list 키·b_input 키가 없다."""
        sent = []
        consume = lambda f, o: sent.append(1)   # noqa: E731
        _, a, _ = run_stage(b_consumer=consume)
        _, b, _ = run_stage(b_consumer=consume, log_b_input="single")

        self.assertEqual(a, b)

        for key in (*LIST_KEYS, '"b_input"'):
            self.assertNotIn(key, a)

    def test_stage_l_minus_new_keys_equals_stage_t(self):
        """같은 입력의 단계 L 기록에서 새 키(list_*·run_meta b_input)만 빼면 단계 T(소비자 없음) 기록과 같다."""
        _, t_text, _ = run_stage()
        _, l_text, _ = run_stage(b_list_consumer=FakeListConsumer(), log_b_input="list")
        stripped = []

        for line in l_text.splitlines():
            rec = json.loads(line)

            for key in (*LIST_KEYS, "b_input"):
                rec.pop(key, None)

            stripped.append(rec)

        self.assertEqual(stripped, [json.loads(t) for t in t_text.splitlines()])

    def test_process_record_rejects_unknown_mode(self):
        """b_input 값은 single·list만(기록 층·PipelineLog 모두)."""
        p, result, out = outcome_of(SCENARIOS["R1_normal_plus_invalid"][0])

        with self.assertRaises(ValueError):
            process_record(out, b_input="multi")

        with self.assertRaises(ValueError):
            PipelineLog(JsonlRecorder(io.StringIO()), b_input="L")


if __name__ == "__main__":
    unittest.main()
