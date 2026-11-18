"""`run_pipeline.py` 저장 입력 시험 B 연결(#166, synthetic/test-only) 시험.

근거: `_workspace/166_a_b_pipeline_connection_spec.md` §6 필수 시험 1~9, B 협업 요청 §8.2 A-7·A-8, A 상세설계 D13.

- 모든 시험은 CLI `main()`을 부른다(실제 저장 이미지 → 실행기 `run()` → 코어 → `b_gate` → `BStage.consume`). B를 직접 부르지 않는다.
- fixture 끝점·너트 윗면 테두리는 알려진 기하(총길이 46 mm 볼트, r 5 mm 너트 윗면)를 고정 카메라 자세로 투영한 **합성 값**이다.
  실제 RGB 검출·끝점 추출·mm 정확도의 증거가 아니다. ROS·Qt·GPU·모델을 쓰지 않는다.
- `b_gate` 경유 증거: 실행기 모듈의 `b_gate`를 감싸 호출·통과 수를 센다. C 송신 증거: `make_c_handoff`·`SortPartClient`를 감시한다.
"""
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from eyetoarm.calibration.profile import from_fov
from eyetoarm.transforms import b_stage as b_stage_mod
from eyetoarm.transforms import sort_part_client
from eyetoarm.transforms.b_stage import StaticTfLookup
from eyetoarm.transforms.planar_transform import look_at, w_to_image
from eyetoarm.vision import pipeline_runner
from eyetoarm.vision.ai_config import load_ai_config
from eyetoarm.vision.synthetic_a_to_b_fixture import FIXTURE_FORMAT, TEST_MODE_LABEL
from eyetoarm.vision.planar import load_config

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_pipeline.py"
OPENCV_CFG = ROOT / "config" / "vision" / "opencv_planar.json"
AI_CFG = ROOT / "config" / "vision" / "ai_s1_01_live.json"

# 시험 카메라·자세: test_b_transforms와 같은 값(1280×720, 수평 화각 1 rad, 고정 자세)
CAM = from_fov(1280, 720, 1.0, "link_cam_optical_frame")
T_W_C = look_at((0.30, -0.08, 0.40), (0.56, -0.07, 0.10))
BOLT_Z, NUT_Z = 0.105, 0.1057   # B 선택 2026-11-17: 볼트 축 105.0 mm, 너트 윗면 105.7 mm
RUN_ID = "b166-test-run"
OPENCV_ID = load_config(OPENCV_CFG).config_sha256
AI_ID = load_ai_config(AI_CFG).config_sha256


def project(points_w, z):
    """작업 좌표(m) 점들을 높이 z에서 원본 픽셀로 투영한다."""
    pts = np.asarray(points_w, dtype=np.float64)

    return w_to_image(np.column_stack([pts, np.full(len(pts), z)]), CAM.K, CAM.dist, T_W_C)


def bolt_endpoints(length=0.046, cx=0.55, cy=-0.05, deg=30.0):
    """총길이 `length` 볼트의 {head, tip} 픽셀(Pa = (cx, cy)) — 합성 투영 값."""
    u = np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])
    head, tip = project([np.array([cx, cy]) - u * length / 2, np.array([cx, cy]) + u * length / 2], BOLT_Z)

    return {"head": head.tolist(), "tip": tip.tolist()}


def nut_rim(cx=0.58, cy=0.0, r=0.005, n=24):
    """너트 윗면 바깥 테두리(r 5 mm 원) 픽셀 — 합성 투영 값."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)

    return project(np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)]), NUT_Z).tolist()


def sha(path):
    """파일 sha256."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bolt_obs(seq, *, config_id=OPENCV_ID, **over):
    """seq 번째 프레임의 유효 볼트 축 관측 레코드(식별자는 실제 저장 입력 실행 값: 세대 0, job_seq = image_seq)."""
    obs = {"run_id": RUN_ID, "source_id": "stored0", "image_seq": seq, "capture_generation": 0, "generation": 0,
           "job_seq": seq, "detector_config_id": config_id, "calibration_id": CAM.calibration_id,
           "pixel_basis": "raw", "class_name": None, "selected_index": 0, "candidate_count": 1, "valid": True,
           "reason": None, "observed_face": "axis", "axis_endpoints_px": bolt_endpoints(), "calc_contour": None}
    obs.update(over)

    return obs


def nut_obs(seq, **over):
    """seq 번째 프레임의 유효 너트 윗면 관측 레코드(AI 설정 — 클래스 이름 nut을 실어야 하므로)."""
    fields = {"class_name": "nut", "observed_face": "top_face", "axis_endpoints_px": None, "calc_contour": nut_rim()}
    fields.update(over)

    return bolt_obs(seq, config_id=AI_ID, **fields)


def invalid_obs(seq):
    """seq 번째 프레임의 무효 관측 레코드(B로 가면 안 됨)."""
    return bolt_obs(seq, valid=False, reason="NO_OBJECT", observed_face=None, axis_endpoints_px=None,
                    selected_index=None, candidate_count=0)


def record(seq, obs, *, stamp=None, T=None):
    """프레임 키(source_index = image_seq = seq)에 귀속된 레코드."""
    return {"source_index": seq, "image_seq": seq, "ros_stamp_ns": stamp, "T_W_C": T, "observation": obs}


class Case:
    """임시 폴더 하나의 입력 이미지·fixture·출력 경로."""

    def __init__(self, d, n=1):
        """n장(1280×720 검은 PNG, 장마다 한 픽셀씩 다르게)을 쓴다."""
        self.d = Path(d)
        self.images = []

        for i in range(n):
            img = np.zeros((CAM.height, CAM.width, 3), np.uint8)
            img[0, i] = (i + 1, 0, 0)
            path = self.d / f"f{i}.png"
            cv2.imwrite(str(path), img)
            self.images.append(str(path))

        self.log = str(self.d / "run.jsonl")
        self.out = str(self.d / "b.jsonl")

    def fixture(self, records, *, motion="fixed", tf=None, **over):
        """fixture 사전(고정 카메라 정적 TF 기본)."""
        doc = {"fixture_format": FIXTURE_FORMAT, "fixture_id": "b166-unit", "test_only": True, "run_id": RUN_ID,
               "inputs_sha256": [sha(p) for p in self.images],
               "source": {"source_id": "stored0", "width": CAM.width, "height": CAM.height, "crop": None,
                          "calibration_id": CAM.calibration_id, "pixel_basis": "raw"},
               "camera": {"width": CAM.width, "height": CAM.height, "K": CAM.K.tolist(), "dist": CAM.dist.tolist(),
                          "frame_id": CAM.frame_id},
               "camera_motion": motion, "tf": tf or {"mode": "static", "T_W_C": T_W_C.tolist()},
               "records": records}
        doc.update(over)

        return doc

    def write(self, doc, name="fixture.json"):
        """fixture를 파일로 쓰고 경로를 돌려준다."""
        path = self.d / name
        path.write_text(json.dumps(doc), encoding="utf-8")

        return str(path)

    def outputs(self):
        """B 시험 출력 줄 (출력 목록, 요약)."""
        if not os.path.exists(self.out):
            return [], None

        recs = [json.loads(t) for t in Path(self.out).read_text(encoding="utf-8").splitlines()]

        return [r for r in recs if r["kind"] == "b_test_output"], next((r for r in recs if r["kind"] == "b_test_summary"),
                                                                        None)

    def processes(self):
        """처리 기록."""
        return [json.loads(t) for t in Path(self.log).read_text(encoding="utf-8").splitlines()
                if json.loads(t)["kind"] == "process"]


def load_cli():
    """scripts/run_pipeline.py를 모듈로 불러온다."""
    spec = importlib.util.spec_from_file_location("run_pipeline_b166_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


class SyntheticAToBBase(unittest.TestCase):
    """CLI 실행 도우미와 b_gate·B·C 감시."""

    def setUp(self):
        """임시 폴더·CLI·감시 장치."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.mod = load_cli()
        self.gate_calls, self.gate_passed, self.consume_calls, self.c_calls = [], [], [], []
        real_gate, real_consume = pipeline_runner.b_gate, b_stage_mod.BStage.consume

        def gate(pipeline, outcome):
            """실행기 b_gate 감시: 원래 판정을 그대로 돌려준다."""
            ok = real_gate(pipeline, outcome)
            self.gate_calls.append(outcome.context.frame.image_seq)

            if ok:
                self.gate_passed.append(outcome.context.frame.image_seq)

            return ok

        def consume(stage, frame, obs):
            """BStage.consume 감시: 받은 프레임·관측 키를 남기고 원래 함수를 부른다."""
            self.consume_calls.append((frame.image_seq, obs.image_seq, obs.job_seq))
            return real_consume(stage, frame, obs)

        def c_spy(*a, **k):
            """C 송신 경로가 불리면 기록하고 실패시킨다."""
            self.c_calls.append(a)
            raise AssertionError("C/robot output must not be used in synthetic-a-to-b")

        for p in (mock.patch.object(pipeline_runner, "b_gate", gate),
                  mock.patch.object(b_stage_mod.BStage, "consume", consume),
                  mock.patch.object(b_stage_mod, "make_c_handoff", c_spy),
                  mock.patch.object(sort_part_client.SortPartClient, "send", c_spy),
                  mock.patch.object(sort_part_client.SortPartClient, "__init__", c_spy)):
            p.start()
            self.addCleanup(p.stop)

    def case(self, n=1):
        """새 Case."""
        return Case(self.tmp.name, n)

    def cli(self, c, fixture_path=None, *, detector="opencv", extra=(), hook=None, images=None, out=True):
        """CLI를 실행해 (종료 코드, stderr)를 돌려준다."""
        argv = ["--images", *(images or c.images)]
        argv += ["--detect-config", str(OPENCV_CFG)] if detector == "opencv" else ["--detector", "ai", "--ai-config",
                                                                                     str(AI_CFG)]
        argv += ["--log", c.log, "--no-detect-timeout", *extra]

        if fixture_path is not None:
            argv += ["--synthetic-a-to-b-fixture", fixture_path]

        if out and fixture_path is not None:
            argv += ["--synthetic-a-to-b-out", c.out]

        err = io.StringIO()

        def no_model(*a, **k):
            """시험 B 모드는 모델을 만들지 않는다."""
            raise AssertionError("model must not be loaded in synthetic-a-to-b mode")

        code = self.mod.main(argv, stderr=err, model_factory=no_model, b_stage_hook=hook)

        return code, err.getvalue()

    def assert_no_b(self, c):
        """B 호출 0, B 출력 줄 0, C 호출 0."""
        self.assertEqual(self.consume_calls, [])
        self.assertEqual(c.outputs()[0], [])
        self.assertEqual(self.c_calls, [])


class DefaultRegressionTest(SyntheticAToBBase):
    """§6-1 기본 회귀: opt-in 없으면 BStage를 만들지 않고 B 호출 0, 표시용 무효 관측 그대로."""

    def test_default_run_builds_no_bstage_and_keeps_invalid_observations(self):
        """opt-in 없는 저장 입력 실행은 종료 0, BStage.from_config 0회, b_gate 통과 0, 관측은 CALIBRATION_MISMATCH 무효다."""
        c = self.case(2)

        with mock.patch.object(b_stage_mod.BStage, "from_config", side_effect=AssertionError("no BStage")) as fc:
            code, err = self.cli(c)

        self.assertEqual(code, 0, err)
        self.assertEqual(fc.call_count, 0)
        self.assertEqual(self.gate_passed, [])
        self.assert_no_b(c)
        self.assertNotIn(TEST_MODE_LABEL, err)
        self.assertEqual([(p["valid"], p["reason"]) for p in c.processes()], [(False, "CALIBRATION_MISMATCH")] * 2)
        self.assertFalse(os.path.exists(c.out))

    def test_synthetic_out_alone_is_argument_error(self):
        """`--synthetic-a-to-b-out`만 주거나 `--synthetic-a-to-b-fixture`만 주면 argparse 오류(2)다."""
        c = self.case()

        for extra in (["--synthetic-a-to-b-out", c.out], ["--synthetic-a-to-b-fixture", c.write(c.fixture([record(0, bolt_obs(0))]))]):
            with self.subTest(extra=extra[0]):
                with self.assertRaises(SystemExit) as ctx:
                    self.cli(c, extra=extra)
                self.assertEqual(ctx.exception.code, 2)


class BoltSuccessTest(SyntheticAToBBase):
    """§6-2 시험 성공 경로: 합성 볼트 끝점 하나 → run() → b_gate → B 정확히 1회."""

    def test_bolt_reaches_b_once_through_gate_with_same_keys(self):
        """유효 볼트 레코드 하나가 b_gate를 통과해 B가 정확히 한 번 불리고, 출력 상관 키가 입력과 같으며 결과는 BOLT_B·46 mm다."""
        c = self.case()
        code, err = self.cli(c, c.write(c.fixture([record(0, bolt_obs(0))])))

        self.assertEqual(code, 0, err)
        self.assertEqual(self.gate_calls, [0])
        self.assertEqual(self.gate_passed, [0])
        self.assertEqual(self.consume_calls, [(0, 0, 0)])
        outs, summary = c.outputs()
        self.assertEqual(len(outs), 1)
        o = outs[0]
        self.assertEqual(o["mode"], TEST_MODE_LABEL)

        # 상관 키: 입력 관측·프레임 = B 표시 사전 = C 목표 사전(값을 새로 만들지 않음)
        for key in ("run_id", "capture_generation", "generation", "job_seq", "detector_config_id", "timestamp_ms",
                    "capture_stamp_ns"):
            with self.subTest(key=key):
                self.assertEqual(o["goal"][key], o["input"][key])
                self.assertEqual(o["result_dict"][key], o["input"][key])

        self.assertEqual((o["input"]["run_id"], o["input"]["image_seq"], o["input"]["detector_config_id"]),
                         (RUN_ID, 0, OPENCV_ID))
        self.assertEqual(o["result_dict"]["image_seq"], 0)
        self.assertEqual(o["result_dict"]["source_id"], "stored0")
        self.assertEqual((o["result_dict"]["spec"], o["result_dict"]["valid"]), ("BOLT_B", True))
        self.assertAlmostEqual(o["result_dict"]["length_mm"], 46.0, delta=0.2)
        self.assertEqual(o["goal"]["kind"], "PICK_PLACE")
        self.assertEqual((summary["runner_b_calls"], summary["expected_b_calls"], summary["exit_code"]), (1, 1, 0))
        self.assertIn(f"A->B SYNTHETIC FIXTURE MODE {TEST_MODE_LABEL}", err)
        self.assertEqual(self.c_calls, [])


class NutPathTest(SyntheticAToBBase):
    """§6-3 너트 경로: top_face 윗면 바깥 테두리 fixture가 윤곽 경로로 B에 도달(볼트와 별도)."""

    def test_nut_top_face_reaches_b_contour_path(self):
        """AI 설정(모델 불러오지 않음)의 너트 top_face 레코드는 b_gate를 지나 B NUT·유효로 계산된다."""
        c = self.case()
        code, err = self.cli(c, c.write(c.fixture([record(0, nut_obs(0))])), detector="ai")

        self.assertEqual(code, 0, err)
        self.assertEqual(self.gate_passed, [0])
        outs, _ = c.outputs()
        self.assertEqual(len(outs), 1)
        self.assertEqual(outs[0]["input"]["observed_face"], "top_face")
        self.assertEqual((outs[0]["result_dict"]["spec"], outs[0]["result_dict"]["valid"]), ("NUT", True))
        self.assertEqual(outs[0]["input"]["detector_config_id"], AI_ID)

    def test_nut_contour_with_other_face_rejected(self):
        """관측면이 top_face가 아닌 윤곽(전체 실루엣 대용 등)은 fixture 오류(2)이고 B 0이다."""
        c = self.case()
        code, err = self.cli(c, c.write(c.fixture([record(0, nut_obs(0, observed_face="silhouette"))])),
                             detector="ai")

        self.assertEqual(code, 2, err)
        self.assert_no_b(c)


class InvalidAndMismatchTest(SyntheticAToBBase):
    """§6-4 무효·불일치: B 호출 0, 오류·거부 이유 기록, b_gate를 우회하지 않음."""

    def run_case(self, records=None, *, n=1, fixture_over=None, extra=(), detector="opencv", mutate=None):
        """fixture를 만들어 실행하고 (코드, stderr, Case)를 돌려준다."""
        c = self.case(n)
        doc = c.fixture(records if records is not None else [record(0, bolt_obs(0))], **(fixture_over or {}))

        if mutate is not None:
            mutate(doc)

        code, err = self.cli(c, c.write(doc), extra=extra, detector=detector)

        return code, err, c

    def test_valid_false_record_not_sent(self):
        """valid=False 레코드는 b_gate를 통과하지 못해 B 0이고 처리 기록에 사유 NO_OBJECT가 남는다(기대 B 0이라 종료 0)."""
        code, err, c = self.run_case([record(0, invalid_obs(0))])

        self.assertEqual(code, 0, err)
        self.assertEqual(self.gate_calls, [0])
        self.assertEqual(self.gate_passed, [])
        self.assert_no_b(c)
        self.assertEqual([(p["valid"], p["reason"]) for p in c.processes()], [(False, "NO_OBJECT")])

    def test_runtime_mismatches_fail_without_b(self):
        """관측 식별자(run_id·source_id·detector_config_id·calibration_id)·오래된 세대·작업 순번 불일치는 코어가 거부해 B 0, 종료 1이다."""
        cases = {"run_id": {"run_id": "other-run"}, "source_id": {"source_id": "other"},
                 "detector_config_id": {"detector_config_id": "0" * 64},
                 "calibration_id": {"calibration_id": "cal-000000000000"},
                 "stale_generation": {"generation": 1}, "stale_capture_generation": {"capture_generation": 1},
                 "job_seq": {"job_seq": 5}, "image_seq": {"image_seq": 3}}

        for name, over in cases.items():
            with self.subTest(case=name):
                self.consume_calls.clear()
                self.gate_passed.clear()
                code, err, c = self.run_case([record(0, bolt_obs(0, **over))])
                self.assertEqual(code, 1, err)
                self.assert_no_b(c)
                self.assertEqual(self.gate_passed, [])
                self.assertIn("b_calls 0 != expected 1", err)
                _, summary = c.outputs()
                self.assertEqual(summary["problems"], ["b_calls 0 != expected 1 (valid fixture records)"])
                os.remove(c.log)
                os.remove(c.out)

    def test_profile_and_source_mismatch(self):
        """크기·crop이 실제 프레임과 다르면 b_gate가 거부(종료 1), 보정 번호·소스 ID 불일치는 시작 전 fixture 오류(2)다. 모두 B 0."""
        def size(doc):
            """fixture 크기를 실제 프레임(1280×720)과 다르게(카메라도 같이 바꿔 보정 번호를 맞춤)."""
            cam = from_fov(1280, 721, 1.0, CAM.frame_id)
            doc["source"].update(height=721, calibration_id=cam.calibration_id)
            doc["camera"].update(height=721, K=cam.K.tolist())
            doc["records"][0]["observation"]["calibration_id"] = cam.calibration_id

        def crop(doc):
            """fixture crop을 실제 프레임(crop 없음)과 다르게."""
            doc["source"]["crop"] = [0, 0, 1280, 720]

        def calib(doc):
            """source 보정 번호를 카메라와 다르게."""
            doc["source"]["calibration_id"] = "cal-000000000000"

        for name, mutate, extra, want in (("size", size, (), 1), ("crop", crop, (), 1), ("calibration", calib, (), 2),
                                          ("source_id", None, ("--source-id", "cam9"), 2)):
            with self.subTest(case=name):
                self.consume_calls.clear()
                code, err, c = self.run_case(mutate=mutate, extra=extra)
                self.assertEqual(code, want, err)
                self.assert_no_b(c)

                for p in (c.log, c.out):
                    if os.path.exists(p):
                        os.remove(p)

    def test_missing_and_duplicate_records(self):
        """프레임 2장에 레코드 1개(누락)는 INFERENCE_ERROR로 종료 1, 같은 프레임 키 중복은 fixture 오류(2)다. 누락 프레임은 B 0."""
        code, err, c = self.run_case([record(0, bolt_obs(0))], n=2)
        self.assertEqual(code, 1, err)
        self.assertEqual([cc[0] for cc in self.consume_calls], [0])   # 레코드가 있는 프레임만
        procs = c.processes()
        self.assertEqual(procs[1]["status"], "FAILED")
        os.remove(c.log)
        os.remove(c.out)

        self.consume_calls.clear()
        code, err, c = self.run_case([record(0, bolt_obs(0)), record(0, bolt_obs(0))])
        self.assertEqual(code, 2, err)
        self.assertIn("duplicate frame key", err)
        self.assert_no_b(c)

    def test_bad_endpoints_rejected(self):
        """프레임 밖 끝점·순서를 알 수 없는 2×2 목록(역순 위험)은 fixture 오류(2), 머리=끝 같은 점은 관측 생성 실패(1). 모두 B 0."""
        ep = bolt_endpoints()

        for name, endpoints, want in (("out_of_frame", {"head": [1280.0, 10.0], "tip": ep["tip"]}, 2),
                                      ("negative", {"head": [-0.5, 10.0], "tip": ep["tip"]}, 2),
                                      ("unordered_list", [ep["tip"], ep["head"]], 2),
                                      ("same_point", {"head": ep["head"], "tip": ep["head"]}, 1)):
            with self.subTest(case=name):
                self.consume_calls.clear()
                code, err, c = self.run_case([record(0, bolt_obs(0, axis_endpoints_px=endpoints))])
                self.assertEqual(code, want, err)
                self.assert_no_b(c)

                for p in (c.log, c.out):
                    if os.path.exists(p):
                        os.remove(p)

    def test_input_fingerprint_and_order(self):
        """입력 순서를 바꾸거나 다른 파일이면(지문 불일치) 시작 전 fixture 오류(2)이고 B 0이다."""
        c = self.case(2)
        path = c.write(c.fixture([record(0, bolt_obs(0)), record(1, bolt_obs(1))]))
        code, err = self.cli(c, path, images=list(reversed(c.images)))

        self.assertEqual(code, 2, err)
        self.assertIn("inputs_sha256", err)
        self.assert_no_b(c)
        self.assertFalse(os.path.exists(c.log))


class TfPathTest(SyntheticAToBBase):
    """§6-5 TF 경로: 고정 자세만 StaticTfLookup, 촬영 시각 TF 정확 일치, 없음·다른 시각·50 ms 초과는 B SKIP."""

    def test_fixed_camera_uses_static_lookup(self):
        """고정 카메라 fixture는 StaticTfLookup으로 조회하고 B 결과가 유효하다."""
        c = self.case()
        seen = {}
        code, err = self.cli(c, c.write(c.fixture([record(0, bolt_obs(0))])),
                             hook=lambda stage: seen.update(inner=stage.tf_lookup.inner))

        self.assertEqual(code, 0, err)
        self.assertIsInstance(seen["inner"], StaticTfLookup)
        self.assertTrue(c.outputs()[0][0]["result_dict"]["valid"])

    def test_moving_camera_static_tf_rejected(self):
        """움직이는 카메라에 정적 TF 하나를 쓰는 fixture는 오류(2)이고 B 0이다."""
        c = self.case()
        code, err = self.cli(c, c.write(c.fixture([record(0, bolt_obs(0))], motion="moving")))

        self.assertEqual(code, 2, err)
        self.assert_no_b(c)

    def test_capture_time_tf_exact_missing_and_other_time(self):
        """촬영 시각 TF: 정확한 시각은 유효, 그 시각 TF가 없으면 다른 시각(앞 프레임) TF로 대체하지 않고 TF_UNAVAILABLE SKIP이다."""
        c = self.case(2)
        doc = c.fixture([record(0, bolt_obs(0), stamp=77_000_000_000, T=T_W_C.tolist()),
                         record(1, bolt_obs(1), stamp=77_066_000_000, T=None)],
                        motion="moving", tf={"mode": "per_frame"})
        code, err = self.cli(c, c.write(doc))

        self.assertEqual(code, 0, err)
        outs, summary = c.outputs()
        self.assertEqual(len(outs), 2)
        self.assertEqual(outs[0]["input"]["capture_stamp_ns"], 77_000_000_000)
        self.assertTrue(outs[0]["result_dict"]["valid"])
        self.assertEqual(outs[0]["goal"]["kind"], "PICK_PLACE")
        self.assertEqual(outs[1]["input"]["capture_stamp_ns"], 77_066_000_000)
        self.assertEqual((outs[1]["b_reason"], outs[1]["b_reason_owner"]), ("TF_UNAVAILABLE", "B"))
        self.assertEqual(outs[1]["goal"]["kind"], "SKIP")
        self.assertEqual(summary["tf"]["unavailable"], 1)

    def test_tf_wait_over_50ms_is_skip(self):
        """TF 조회가 50 ms를 넘기면 None으로 보고 B는 TF_UNAVAILABLE SKIP이다(성공 목표 없음)."""
        c = self.case()

        def slow(stage):
            """원래 조회 앞에 60 ms 지연을 넣는다."""
            inner = stage.tf_lookup.inner

            def delayed(frame):
                """지연 뒤 원래 값."""
                time.sleep(0.06)
                return inner(frame)

            stage.tf_lookup.inner = delayed

        code, err = self.cli(c, c.write(c.fixture([record(0, bolt_obs(0))])), hook=slow)

        self.assertEqual(code, 0, err)
        outs, summary = c.outputs()
        self.assertEqual((outs[0]["b_reason"], outs[0]["goal"]["kind"]), ("TF_UNAVAILABLE", "SKIP"))
        self.assertEqual(summary["tf"]["timed_out"], 1)

    def test_per_frame_needs_capture_stamp(self):
        """per_frame TF에서 촬영 시각이 없는 레코드는 fixture 오류(2)다."""
        c = self.case()
        doc = c.fixture([record(0, bolt_obs(0), stamp=None, T=T_W_C.tolist())], motion="moving",
                        tf={"mode": "per_frame"})
        code, err = self.cli(c, c.write(doc))

        self.assertEqual(code, 2, err)
        self.assert_no_b(c)


class ComputeBudgetTest(SyntheticAToBBase):
    """§6-6 B 계산 제한: 정상 20 ms 이내, 초과는 B_TIMEOUT SKIP."""

    def test_normal_compute_within_budget(self):
        """정상 계산의 B compute_ms는 20 ms 이하이고 목표는 PICK_PLACE다."""
        c = self.case()
        seen = {}
        code, err = self.cli(c, c.write(c.fixture([record(0, bolt_obs(0))])), hook=lambda s: seen.update(stage=s))

        self.assertEqual(code, 0, err)
        self.assertEqual(seen["stage"].compute_budget_ms, 20.0)
        self.assertLessEqual(seen["stage"].last.result.compute_ms, 20.0)
        self.assertEqual(c.outputs()[0][0]["goal"]["kind"], "PICK_PLACE")

    def test_over_budget_is_b_timeout_skip(self):
        """계산 상한을 넘긴 것으로 만들면(상한 −1 ms) 목표는 SKIP/B_TIMEOUT이고 성공 목표로 나가지 않는다."""
        c = self.case()

        def tighten(stage):
            """상한을 음수로 둬 어떤 계산도 초과가 되게 한다."""
            stage.compute_budget_ms = -1.0

        code, err = self.cli(c, c.write(c.fixture([record(0, bolt_obs(0))])), hook=tighten)

        self.assertEqual(code, 0, err)
        goal = c.outputs()[0][0]["goal"]
        self.assertEqual((goal["kind"], goal["reason"]), ("SKIP", "B_TIMEOUT"))


class LiveForbiddenTest(SyntheticAToBBase):
    """§6-7 실시간 금지: --ros-link와 시험 B 옵션 조합은 argparse 오류(2), B 0."""

    def test_ros_link_with_synthetic_fixture_is_argument_error(self):
        """`--ros-link --synthetic-a-to-b-fixture ... --synthetic-a-to-b-out ...`은 rclpy 확인 전에도 종료 2이고 BStage를 만들지 않는다."""
        c = self.case()
        path = c.write(c.fixture([record(0, bolt_obs(0))]))

        with mock.patch.object(b_stage_mod.BStage, "from_config", side_effect=AssertionError("no BStage")):
            with self.assertRaises(SystemExit) as ctx:
                self.mod.main(["--ros-link", "--detector", "ai", "--ai-config", str(AI_CFG), "--log", c.log,
                               "--synthetic-a-to-b-fixture", path, "--synthetic-a-to-b-out", c.out], stderr=io.StringIO(),
                              ros_available=lambda: True,
                              live_source_factory=lambda *a: (_ for _ in ()).throw(AssertionError("no live")))

        self.assertEqual(ctx.exception.code, 2)
        self.assert_no_b(c)

    def test_tracking_with_synthetic_fixture_is_argument_error(self):
        """`--tracking`과 시험 B 연결은 함께 쓸 수 없다(종료 2)."""
        c = self.case()

        with self.assertRaises(SystemExit) as ctx:
            self.cli(c, c.write(c.fixture([record(0, nut_obs(0))])), detector="ai", extra=("--tracking",))

        self.assertEqual(ctx.exception.code, 2)


class NoCOutputTest(SyntheticAToBBase):
    """§6-8 C 미연결: on_output None, C/ROS action 송신 0, rclpy를 불러오지 않음."""

    def test_on_output_none_and_no_c_or_ros(self):
        """시험 B 실행에서 stage.on_output은 None이고 C 송신 감시 호출 0, 이 실행이 rclpy를 새로 import하지 않는다."""
        c = self.case()
        seen = {}
        had_rclpy = "rclpy" in sys.modules
        code, err = self.cli(c, c.write(c.fixture([record(0, bolt_obs(0))])), hook=lambda s: seen.update(stage=s))

        self.assertEqual(code, 0, err)
        self.assertIsNone(seen["stage"].on_output)
        self.assertEqual(self.c_calls, [])
        self.assertEqual("rclpy" in sys.modules, had_rclpy)
        self.assertIsNone(c.outputs()[1]["on_output"])

    def test_on_output_set_is_refused(self):
        """누군가 on_output을 연결하면 실행 전에 종료 1이고 B 0이다."""
        c = self.case()
        code, err = self.cli(c, c.write(c.fixture([record(0, bolt_obs(0))])),
                             hook=lambda s: setattr(s, "on_output", lambda out: None))

        self.assertEqual(code, 1, err)
        self.assert_no_b(c)


class ContinuityTest(SyntheticAToBBase):
    """§6-9 연속성·무결성: 반복 실행 결과 고정, 출력 덮어쓰기 거부, 시험 표시."""

    STABLE = ("run_id", "capture_generation", "generation", "job_seq", "detector_config_id", "kind", "source_id")

    def test_repeat_runs_same_and_outputs_not_overwritten(self):
        """같은 fixture·입력 2회는 프레임 순서·B 판정·지문이 같고, 같은 출력 경로 재실행은 2로 거부되며 기존 파일은 그대로다."""
        c = self.case(2)
        path = c.write(c.fixture([record(0, bolt_obs(0)), record(1, invalid_obs(1))]))
        runs = []

        for i in range(2):
            c.log, c.out = str(c.d / f"run{i}.jsonl"), str(c.d / f"b{i}.jsonl")
            code, err = self.cli(c, path)
            self.assertEqual(code, 0, err)
            outs, summary = c.outputs()
            runs.append(([(o["input"]["image_seq"], o["input"]["source_index"], o["result_dict"]["spec"],
                           {k: o["goal"].get(k) for k in self.STABLE}) for o in outs],
                         summary["inputs_sha256"], summary["fixture_records_used"]))
            self.assertTrue(all(o["mode"] == TEST_MODE_LABEL and o["fixture_sha256"] == sha(path) for o in outs))

        self.assertEqual(runs[0], runs[1])
        self.assertEqual(runs[0][2], [0, 1])

        # 출력 덮어쓰기 거부: 기록·B 출력 각각
        before = Path(c.out).read_bytes()
        code, err = self.cli(c, path)
        self.assertEqual(code, 2)
        self.assertIn("refusing to overwrite", err)
        self.assertEqual(Path(c.out).read_bytes(), before)

        c.out = str(c.d / "b_new.jsonl")
        code, err = self.cli(c, path)
        self.assertEqual(code, 2)
        self.assertFalse(os.path.exists(c.out))

    def test_fixture_format_guards(self):
        """test_only가 true가 아니거나 알 수 없는 키·NaN이 있으면 fixture 오류(2)다."""
        c = self.case()
        base = c.fixture([record(0, bolt_obs(0))])

        for name, doc in (("test_only", {**base, "test_only": False}), ("unknown", {**base, "extra": 1})):
            with self.subTest(case=name):
                code, err = self.cli(c, c.write(doc))
                self.assertEqual(code, 2, err)

        path = c.d / "nan.json"
        path.write_text(json.dumps(base).replace('"candidate_count": 1', '"candidate_count": NaN'), encoding="utf-8")
        code, err = self.cli(c, str(path))
        self.assertEqual(code, 2, err)
        self.assert_no_b(c)


if __name__ == "__main__":
    unittest.main()
