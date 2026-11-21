"""A-8 볼트 두 끝점 관측 → A 코어·`b_gate` → 실제 B(`BStage`) 종단 시험.
끝점·너트 테두리는 알려진 기하를 고정 카메라 자세로 투영한 시험용 관측이다.
B는 실제 `BStage.consume`을 부르고 ROS·Qt·GPU·모델은 쓰지 않는다.
"""
import dataclasses
import math
import unittest

import numpy as np

from eyetoarm.calibration.profile import from_fov
from eyetoarm.common.types import InvalidationKind, PixelBasis, PlanarObservation
from eyetoarm.transforms.b_stage import BStage, StaticTfLookup
from eyetoarm.transforms.planar_transform import w_to_image
from eyetoarm.transforms.work_result import class_plane_z, load_specs
from eyetoarm.vision.pipeline import ADAPTER_CONTRACT_VIOLATION, GenerationCause, VisionPipeline
from eyetoarm.vision.pipeline_runner import b_gate
from eyetoarm.vision.ai_convert import observation_from_ai
from eyetoarm.vision.handoff import AToCHandoff, MsgKind
from eyetoarm.vision.pipeline import GenerationCause as Cause
from eyetoarm.vision.pipeline_opencv import process_opencv
from eyetoarm.vision.pipeline_types import JobStatus
from test_ai_config import ai_config
from test_ai_result import seg_result
from test_b_transforms import CAM, ROOT, T_W_C
from test_pipeline import RUN_ID, FakeClock, frame, profile, settings
from test_run_pipeline import Base, FakeSource

W, H = CAM.width, CAM.height
# 종류별 기준 높이는 B 규격 파일에서 읽어 합성 투영과 기대값이 함께 따라가게 한다
_SPECS = load_specs(ROOT / "config/calibration/part_specs.json")
BOLT_Z, NUT_Z = class_plane_z(_SPECS, "bolt"), class_plane_z(_SPECS, "nut")
# 볼트 총길이도 B 규격에서 읽어 규격이 바뀌어도 단언이 따라가게 한다
LEN_A, LEN_B = float(_SPECS["specs"]["BOLT_A"]["total_length"]), float(_SPECS["specs"]["BOLT_B"]["total_length"])
# 집는 점 Pb = Pa에서 축 끝 쪽으로 머리 높이/2(B는 모든 볼트에 BOLT_A `head_height`를 씀)
GRASP_OFFSET_M = float(_SPECS["specs"]["BOLT_A"].get("head_height", 0.006)) / 2.0


def _project(points_w, z):
    """작업 좌표(m) 점들을 기준 높이 z에서 원본 픽셀로 투영한다."""
    pts = np.asarray(points_w, dtype=np.float64)

    return w_to_image(np.column_stack([pts, np.full(len(pts), z)]), CAM.K, CAM.dist, T_W_C)


def bolt_endpoints(length, cx, cy, deg):
    """총길이 `length` 볼트의 머리 끝·축 끝 픽셀과 축 단위벡터(Pa = (cx, cy))."""
    u = np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])
    head = np.array([cx, cy]) - u * length / 2
    tip = np.array([cx, cy]) + u * length / 2

    return _project([head, tip], BOLT_Z), u


def nut_top_rim(cx, cy, r=0.005, n=24):
    """너트 윗면 바깥 테두리(r 5 mm 원)의 원본 픽셀 윤곽."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)

    return _project(np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)]), NUT_Z).astype(np.float32)


def a8_settings(**over):
    """AI 설정(모델 불러오지 않음)·1280×720·B 카메라와 같은 보정 번호의 프로파일."""
    fields = {"detector_config": ai_config(),
              "profile": profile(width=W, height=H, calibration_id=CAM.calibration_id)}
    fields.update(over)

    return settings(**fields)


def a8_frame(seq, gen):
    """1280×720 저장 입력 프레임(보정 번호 = B 카메라)."""
    return frame(seq, gen, image=np.zeros((H, W, 3), np.uint8), width=W, height=H,
                 calibration_id=CAM.calibration_id, optical_frame=CAM.frame_id)


def frames(n):
    """n개 프레임을 주는 가짜 저장 소스 팩토리."""
    def factory(issuer, generation_provider):
        """실행기의 발급기·세대로 프레임을 만든다."""
        return FakeSource([a8_frame(issuer.next_seq(), generation_provider()) for _ in range(n)])

    return factory


def axis_observation(job, endpoints, **over):
    """작업 식별자를 그대로 옮긴 시험용 볼트 축 유효 관측(GT 투영 끝점)."""
    f = job.frame
    fields = dict(
        run_id=f.run_id, source_id=f.source_id, image_seq=f.image_seq, capture_generation=f.capture_generation,
        generation=job.generation, job_seq=job.job_seq, detector_config_id=job.detector_config_id,
        pixel_basis=PixelBasis.RAW, calibration_id=f.calibration_id, observed_face="axis", calc_contour=None,
        display_mask_contour=None, candidate_count=1, selected_index=0, valid=True, reason=None, process_ms=0.1,
        method="ai", class_name="bolt", confidence=0.9, model_id=getattr(job.settings.detector_config, "model_id", None),
        axis_endpoints_px=endpoints)
    fields.update(over)

    return PlanarObservation(**fields)


def nut_observation(job, contour):
    """시험용 너트 윗면 유효 관측(top_face + 윗면 바깥 테두리)."""
    return dataclasses.replace(axis_observation(job, None, observed_face="top_face", calc_contour=contour,
                                                class_name="nut", axis_endpoints_px=None))


class Planned:
    """작업 순번 → 관측 생성 함수. 정하지 않은 순번은 B로 가지 않는 무효 관측."""

    def __init__(self, plan):
        """plan: job_seq → callable(job) → PlanarObservation."""
        self.plan = plan

    def __call__(self, job):
        """정해 둔 관측을 돌려준다(표시 결과 없음)."""
        make = self.plan.get(job.job_seq)

        if make is None:
            return axis_observation(job, None, valid=False, reason="NO_OBJECT", observed_face=None,
                                    selected_index=None, candidate_count=0, class_name=None, confidence=None), None

        return make(job), None


class A8Base(Base):
    """실제 BStage를 b_consumer로 단 저장 입력 실행 도우미."""

    def run_b(self, n, plan, *, camera=CAM, tf_lookup=None, sett=None, exit_code=0):
        """n프레임을 실행해 (결과, 기록 스트림, B 출력 목록)을 돌려준다(실패 작업이 있으면 종료 코드 1)."""
        stage = BStage.from_config(ROOT, camera, tf_lookup or StaticTfLookup(T_W_C))
        outputs = []
        stage.on_output = outputs.append      # 기록용. 액션·C 명령을 보내지 않는다
        res, stream, err = self.run_with(frames(n), detector=Planned(plan), sett=sett or a8_settings(),
                                         b_consumer=stage.consume)

        self.assertEqual(res.exit_code, exit_code, err.getvalue())

        return res, stream, outputs

    def processes(self, stream):
        """처리 기록 목록."""
        import json

        return [json.loads(t) for t in stream.getvalue().splitlines() if json.loads(t)["kind"] == "process"]


class AxisToBStageTest(A8Base):
    """정상 경로: 시험용 볼트 두 점·너트 윗면 → A 코어 → `b_gate` → 실제 BStage."""

    def test_bolt_a_and_b_reach_bstage_with_pa_pb_and_keys(self):
        """B 규격 총길이 볼트 두 점 관측이 B에 도달해 종류·길이·Pa·Pb·PICK_PLACE·연결 키가 맞다."""
        (p46, u46), (p18, u18) = bolt_endpoints(LEN_B, 0.55, -0.05, 30.0), bolt_endpoints(LEN_A, 0.55, 0.05, 120.0)
        seen = {}

        def make(points, key):
            """관측을 만들고 나중 비교를 위해 보관한다."""
            def build(job):
                """작업 식별자로 시험용 축 관측을 만든다."""
                seen[key] = axis_observation(job, points)
                return seen[key]
            return build

        res, _, outs = self.run_b(2, {0: make(p46, "b"), 1: make(p18, "a")})

        self.assertEqual(res.b_calls, 2)
        self.assertEqual(len(outs), 2)

        for out, key, spec, length, (cx, cy), u in ((outs[0], "b", "BOLT_B", LEN_B, (0.55, -0.05), u46),
                                                     (outs[1], "a", "BOLT_A", LEN_A, (0.55, 0.05), u18)):
            with self.subTest(spec=spec):
                r, obs = out.result, seen[key]
                self.assertTrue(r.valid, r.reason)
                self.assertEqual((r.spec, r.method, r.plane_z_m), (spec, "axis_endpoints", BOLT_Z))
                self.assertAlmostEqual(r.measured_length_m, length, places=4)
                # Pa는 보고 위치, Pb는 Pa에서 축 끝 쪽으로 머리 높이/2
                np.testing.assert_allclose(r.center_w_m, (cx, cy), atol=1e-4)
                np.testing.assert_allclose(r.grasp_w_m, np.array([cx, cy]) + u * GRASP_OFFSET_M, atol=1e-4)
                self.assertEqual(out.goal["kind"], "PICK_PLACE")

                # A 관측 → B → C·D 연결 키는 관측·프레임 식별자 그대로다
                keys = {k: out.goal[k] for k in ("run_id", "capture_generation", "generation", "job_seq",
                                                 "detector_config_id", "timestamp_ms", "capture_stamp_ns")}
                self.assertEqual(keys, {"run_id": obs.run_id, "capture_generation": obs.capture_generation,
                                        "generation": obs.generation, "job_seq": obs.job_seq,
                                        "detector_config_id": obs.detector_config_id,
                                        "timestamp_ms": 10_000 + obs.image_seq,
                                        "capture_stamp_ns": 77_000_000_000 + obs.image_seq})

    def test_nut_top_face_uses_contour_path(self):
        """너트 윗면 관측은 윤곽 경로로 B에 도달해 NUT·105.7 mm 기준 높이로 계산된다."""
        rim = nut_top_rim(0.58, 0.0)
        res, _, outs = self.run_b(1, {0: lambda job: nut_observation(job, rim)})

        self.assertEqual(res.b_calls, 1)
        r = outs[0].result
        self.assertTrue(r.valid, r.reason)
        self.assertEqual((r.spec, r.plane_z_m), ("NUT", NUT_Z))
        self.assertNotEqual(r.method, "axis_endpoints")
        np.testing.assert_allclose(r.center_w_m, (0.58, 0.0), atol=2e-4)


class AxisBlockedBeforeBTest(A8Base):
    """A 코어·게이트에서 막혀 B가 한 번도 불리지 않는 경우."""

    def test_profile_none_blocks_at_gate(self):
        """프로파일이 없으면 축 관측이 유효여도 `b_gate`가 막아 B 0회다."""
        points, _ = bolt_endpoints(LEN_B, 0.55, -0.05, 30.0)
        res, stream, outs = self.run_b(1, {0: lambda job: axis_observation(job, points)},
                                       sett=a8_settings(profile=None))

        self.assertEqual((res.b_calls, len(outs)), (0, 0))
        self.assertEqual([p["valid"] for p in self.processes(stream)], [True])

    def test_identifier_mismatch_is_contract_violation(self):
        """관측 식별자가 작업 프레임과 다르면 계약 위반(INFERENCE_ERROR)이며 B 0회다."""
        points, _ = bolt_endpoints(LEN_B, 0.55, -0.05, 30.0)

        for field, value in (("image_seq", 99), ("calibration_id", "other-cal")):
            with self.subTest(field=field):
                res, stream, outs = self.run_b(1, {0: lambda job: axis_observation(job, points, **{field: value})},
                                               exit_code=1)
                self.assertEqual((res.b_calls, len(outs)), (0, 0))
                self.assertEqual([p["status"] for p in self.processes(stream)], ["FAILED"])

    def test_raw_endpoint_range_boundaries(self):
        """raw 끝점은 영상 범위 안이면 B에 도달하고, 범위 밖이면 계약 위반으로 B 0회다."""
        inside = {"corners": [[0.0, 0.0], [W - 1.0, H - 1.0]], "edges": [[0.0, H - 1.0], [W - 1.0, 0.0]]}
        outside = {"neg_u": [[-0.5, 10.0], [100.0, 100.0]], "neg_v": [[10.0, -1e-6], [100.0, 100.0]],
                   "u_eq_w": [[float(W), 10.0], [100.0, 100.0]], "v_over": [[10.0, 10.0], [100.0, H - 1 + 1e-3]]}

        for label, points in inside.items():
            with self.subTest(case=label):
                res, _, outs = self.run_b(1, {0: lambda job: axis_observation(job, points)})
                self.assertEqual((res.b_calls, len(outs)), (1, 1))   # B 판정(길이 등)은 B 몫

        for label, points in outside.items():
            with self.subTest(case=label):
                res, stream, outs = self.run_b(1, {0: lambda job: axis_observation(job, points)}, exit_code=1)
                self.assertEqual((res.b_calls, len(outs)), (0, 0))
                self.assertEqual([p["status"] for p in self.processes(stream)], ["FAILED"])

    def test_undistorted_range_is_not_checked(self):
        """undistorted 끝점은 형식·유한성만 보고 범위는 코어가 막지 않는다."""
        points = [[-5.0, 10.0], [W + 5.0, 100.0]]
        res, _, outs = self.run_b(1, {0: lambda job: axis_observation(job, points, pixel_basis=PixelBasis.UNDISTORTED)})

        self.assertEqual(res.b_calls, 1)

    def test_tampered_endpoints_fail_without_exception(self):
        """생성 뒤 변조된 끝점은 예외 없이 계약 위반 INFERENCE_ERROR·`b_gate` 거짓으로 남는다."""
        points, _ = bolt_endpoints(LEN_B, 0.55, -0.05, 30.0)
        p = VisionPipeline(a8_settings(), run_id=RUN_ID, clock_ns=FakeClock())
        tampered = {
            "nan": np.array([[np.nan, 1.0], [2.0, 3.0]]),
            "shape": np.zeros((3, 2)),
            "text": "abc",
            "text_list": [["1", "2"], ["3", "4"]],
            "bool_list": [[True, False], [False, True]],
            "bool_array": np.array([[True, False], [False, True]]),
            "object_numeric": np.array([[10, 20], [30, 40]], dtype=object),
            "same_point": np.array([[10.0, 10.0], [10.0, 10.0]]),
            "none_on_valid_axis": None,
            "basis": "keep",
            "on_top_face": "top_face",
            "class_nut": ("class_name", "nut"),
            "class_washer": ("class_name", "washer"),
            "class_not_str": ("class_name", 5),
            "contour_with_endpoints": ("calc_contour", np.array([[1, 1], [2, 1], [2, 2]], dtype=np.float32)),
        }

        for seq, (label, value) in enumerate(tampered.items()):
            with self.subTest(case=label):
                p.offer_frame(a8_frame(seq, p.current_generation()))
                job = p.start_next()
                obs = axis_observation(job, points)

                if label == "basis":
                    object.__setattr__(obs, "pixel_basis", "rectified")
                elif label == "on_top_face":
                    # 윤곽 경로 관측에 끝점만 주입(공통형이면 거부되는 조합)
                    obs = nut_observation(job, nut_top_rim(0.58, 0.0))
                    object.__setattr__(obs, "axis_endpoints_px", np.asarray(points, dtype=np.float64))
                elif isinstance(value, tuple):
                    # 끝점은 그대로 두고 다른 한 필드만 공통형이 금지한 조합으로 변조
                    object.__setattr__(obs, value[0], value[1])
                else:
                    object.__setattr__(obs, "axis_endpoints_px", value)

                outcome = p.complete(job, obs)
                self.assertIs(outcome.status, JobStatus.FAILED)
                self.assertIs(outcome.invalidation.kind, InvalidationKind.INFERENCE_ERROR)
                self.assertEqual(outcome.invalidation.cause, f"{ADAPTER_CONTRACT_VIOLATION}:axis_endpoints_px")
                self.assertFalse(b_gate(p, outcome))

    def test_opencv_axis_with_class_none_is_forwarded(self):
        """OpenCV 설정의 축 관측(class None)은 코어 재검사를 통과해 `b_gate`가 참이다."""
        points, _ = bolt_endpoints(LEN_B, 0.55, -0.05, 30.0)
        p = VisionPipeline(a8_settings(detector_config=settings().detector_config), run_id=RUN_ID, clock_ns=FakeClock())
        p.offer_frame(a8_frame(0, p.current_generation()))
        job = p.start_next()
        obs = axis_observation(job, points, method="opencv", class_name=None, confidence=None, model_id=None)

        outcome = p.complete(job, obs)
        self.assertIs(outcome.status, JobStatus.COMPLETED)
        self.assertTrue(b_gate(p, outcome))

    def test_invalid_axis_without_endpoints_still_completes(self):
        """무효 axis 관측은 끝점 None이어도 COMPLETED이며 B로 가지 않는다."""
        p = VisionPipeline(a8_settings(), run_id=RUN_ID, clock_ns=FakeClock())
        p.offer_frame(a8_frame(0, p.current_generation()))
        job = p.start_next()
        obs = axis_observation(job, None, valid=False, reason="ENDPOINT_CLIPPED")

        outcome = p.complete(job, obs)
        self.assertIs(outcome.status, JobStatus.COMPLETED)
        self.assertFalse(b_gate(p, outcome))

    def test_duplicate_and_stale_generation_not_forwarded(self):
        """축 관측도 duplicate·다음 결과 뒤·세대 변경 뒤 재전달은 `b_gate`가 막는다."""
        points, _ = bolt_endpoints(LEN_B, 0.55, -0.05, 30.0)
        p = VisionPipeline(a8_settings(), run_id=RUN_ID, clock_ns=FakeClock())

        def go(seq):
            """작업 하나를 시작해 축 관측으로 완료한다."""
            p.offer_frame(a8_frame(seq, p.current_generation()))
            job = p.start_next()
            return job, p.complete(job, axis_observation(job, points))

        j0, o0 = go(0)
        self.assertTrue(b_gate(p, o0))
        self.assertFalse(b_gate(p, p.complete(j0, axis_observation(j0, points))))   # duplicate

        _, o1 = go(1)
        self.assertTrue(b_gate(p, o1))
        self.assertFalse(b_gate(p, o0))                                             # 다음 결과 뒤 재전달

        p.change_settings(a8_settings(detector_config=ai_config(conf_threshold=0.5)), cause=GenerationCause.DETECTOR_CONFIG_CHANGED)
        self.assertFalse(b_gate(p, o1))                                             # 세대 변경 뒤 재전달


class AxisReachesBThenSkipsTest(A8Base):
    """A는 통과했지만 B가 받은 뒤 SKIP하는 경우(B 1회, 결과 무효)."""

    def test_b_camera_calibration_mismatch_skips_in_b(self):
        """B 카메라 보정만 다르면 B에 도달한 뒤 CALIBRATION_MISMATCH로 SKIP한다."""
        other = from_fov(W, H, 1.05, CAM.frame_id)
        self.assertNotEqual(other.calibration_id, CAM.calibration_id)
        points, _ = bolt_endpoints(LEN_B, 0.55, -0.05, 30.0)
        res, _, outs = self.run_b(1, {0: lambda job: axis_observation(job, points)}, camera=other)

        self.assertEqual(res.b_calls, 1)
        self.assertEqual((outs[0].result.valid, outs[0].result.reason), (False, "CALIBRATION_MISMATCH"))
        self.assertEqual(outs[0].goal["kind"], "SKIP")

    def test_missing_tf_skips_in_b(self):
        """촬영 시각 TF가 없으면 B에 도달한 뒤 TF_UNAVAILABLE로 SKIP한다(최신 TF로 대신하지 않음)."""
        points, _ = bolt_endpoints(LEN_B, 0.55, -0.05, 30.0)
        res, _, outs = self.run_b(1, {0: lambda job: axis_observation(job, points)}, tf_lookup=lambda frame: None)

        self.assertEqual(res.b_calls, 1)
        self.assertEqual((outs[0].result.reason, outs[0].goal["kind"]), ("TF_UNAVAILABLE", "SKIP"))


class AxisPreservationTest(A8Base):
    """내부 전달·재처리·기록·기본 생산자에서 끝점 보존·회귀."""

    def test_handoff_result_keeps_same_endpoints_readonly(self):
        """내부 전달 RESULT는 같은 관측 객체이며 끝점·순서·읽기 전용·식별자가 보존된다."""
        points, _ = bolt_endpoints(LEN_B, 0.55, -0.05, 30.0)
        p = VisionPipeline(a8_settings(), run_id=RUN_ID, clock_ns=FakeClock())
        h = AToCHandoff(clock_ns=FakeClock())
        h.start_run(p)
        p.offer_frame(a8_frame(0, p.current_generation()))
        job = p.start_next()
        out = p.complete(job, axis_observation(job, points))
        seq = h.offer_result(out)

        msgs, _ = h.take_control(10)
        result = [m for m in msgs if m.msg_kind is MsgKind.RESULT]
        self.assertEqual(len(result), 1)
        msg = result[0]
        obs = msg.observation

        self.assertEqual(msg.msg_seq, seq)
        self.assertIs(obs, out.observation)
        self.assertFalse(msg.display_only)
        np.testing.assert_array_equal(obs.axis_endpoints_px, np.asarray(points, dtype=np.float64))
        self.assertLess(np.linalg.norm(obs.axis_endpoints_px[0] - points[0]), 1e-9)  # 행 0 = 머리 끝
        self.assertFalse(obs.axis_endpoints_px.flags.writeable)
        self.assertEqual((obs.run_id, obs.image_seq, obs.generation, obs.job_seq),
                         (job.frame.run_id, job.frame.image_seq, job.generation, job.job_seq))
        self.assertEqual((msg.image_seq, msg.generation, msg.job_seq),
                         (job.frame.image_seq, job.generation, job.job_seq))

        with self.assertRaises(ValueError):
            obs.axis_endpoints_px[0, 0] = 0.0

    def test_reprocess_uses_new_detector_output_only(self):
        """검출 설정 변경 뒤 재처리 관측은 새 detector 출력이며 이전 끝점이 남지 않는다."""
        old_points, _ = bolt_endpoints(LEN_B, 0.55, -0.05, 30.0)
        new_points, _ = bolt_endpoints(LEN_A, 0.55, 0.05, 120.0)
        p = VisionPipeline(a8_settings(), run_id=RUN_ID, clock_ns=FakeClock())
        p.offer_frame(a8_frame(0, p.current_generation()))
        j0 = p.start_next()
        o0 = p.complete(j0, axis_observation(j0, old_points))
        self.assertTrue(b_gate(p, o0))

        p.change_settings(a8_settings(detector_config=ai_config(conf_threshold=0.5)), cause=Cause.DETECTOR_CONFIG_CHANGED)
        j1 = p.start_reprocess()
        self.assertTrue(j1.reprocess)
        self.assertEqual(j1.frame.image_seq, j0.frame.image_seq)
        o1 = p.complete(j1, axis_observation(j1, new_points))

        np.testing.assert_array_equal(p.current_observation.axis_endpoints_px, np.asarray(new_points, dtype=np.float64))
        self.assertTrue(b_gate(p, o1))
        self.assertFalse(b_gate(p, o0))

        # 무검출 재처리면 끝점 없는 무효 관측이 현재가 된다(이전 끝점 재사용 없음)
        p.change_settings(a8_settings(detector_config=ai_config(conf_threshold=0.6)), cause=Cause.DETECTOR_CONFIG_CHANGED)
        j2 = p.start_reprocess()
        o2 = p.complete(j2, Planned({})(j2)[0])

        self.assertIsNone(p.current_observation.axis_endpoints_px)
        self.assertFalse(b_gate(p, o2))
        self.assertFalse(b_gate(p, o1))

    def test_jsonl_process_keys_unchanged(self):
        """JSONL 처리 기록의 키 집합은 축 관측이어도 기존과 같고 끝점 필드를 쓰지 않는다."""
        points, _ = bolt_endpoints(LEN_B, 0.55, -0.05, 30.0)
        _, axis_stream, _ = self.run_b(1, {0: lambda job: axis_observation(job, points)})
        _, base_stream, _ = self.run_b(1, {})

        axis_rec, base_rec = self.processes(axis_stream)[0], self.processes(base_stream)[0]
        self.assertEqual(set(axis_rec), set(base_rec))
        self.assertNotIn("axis_endpoints_px", axis_rec)
        self.assertNotIn("calc_contour", axis_rec)
        self.assertEqual((axis_rec["valid"], base_rec["valid"]), (True, False))

    def test_default_producers_still_emit_invalid_without_endpoints(self):
        """실제 OpenCV·AI 생산자는 여전히 끝점 없는 무효 표시 관측을 낸다."""
        p = VisionPipeline(a8_settings(detector_config=settings().detector_config), run_id=RUN_ID, clock_ns=FakeClock())
        p.offer_frame(a8_frame(0, p.current_generation()))
        obs_cv, _ = process_opencv(p.start_next())

        q = VisionPipeline(a8_settings(), run_id=RUN_ID, clock_ns=FakeClock())
        q.offer_frame(a8_frame(0, q.current_generation()))
        job = q.start_next()
        obs_ai = observation_from_ai(job, seg_result((), config_sha256=job.detector_config_id,
                                                     model_id=job.settings.detector_config.model_id, crop_wh=(W, H)))

        for obs in (obs_cv, obs_ai):
            with self.subTest(method=obs.method):
                self.assertFalse(obs.valid)
                self.assertIsNone(obs.observed_face)
                self.assertIsNone(obs.calc_contour)
                self.assertIsNone(obs.axis_endpoints_px)


if __name__ == "__main__":
    unittest.main()
