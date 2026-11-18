"""`run_pipeline.py` 저장 입력 실제 A→B 경로(#168, `--a-to-b-capture-meta`/`--a-to-b-out`) CLI 시험.

근거: `_workspace/168_a_to_b_real_path_implementation_spec.md` §2·§3·「검증 및 완료 기준」.

- 모두 CLI `main()`으로 실행한다: 저장 이미지 → `CaptureMetaSource` → 실행기 → `AiSegDetector`(가짜 모델, 실제 convert·M1) →
  코어 → `b_admit`(`b_gate`) → `BStage.consume` + `render_center`. 관측은 주입하지 않는다(#166 fixture와 다름).
- 가짜 모델 마스크는 numpy로 그린 합성 볼트이고, 메타(K·TF)는 시험이 정한 값이다. 결과는 배선 확인이며 정확도가 아니다.
- C 송신 감시: `make_c_handoff`·`SortPartClient` 호출 0.
"""
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from eyetoarm.calibration.profile import from_camera_info
from eyetoarm.transforms import b_stage as b_stage_mod
from eyetoarm.transforms import sort_part_client
from eyetoarm.transforms.planar_transform import look_at
from eyetoarm.vision import pipeline_runner
from eyetoarm.vision.capture_meta import CAPTURE_META_FORMAT
from test_ai_config import AI_FIELDS
from test_ai_detector import FakeBoxes, FakeMasks, FakeModel, FakeResult

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_pipeline.py"
W, H = 320, 240
K = [300.0, 0.0, 160.0, 0.0, 300.0, 120.0, 0.0, 0.0, 1.0]
FRAME_ID = "link_cam_optical_frame"
T_W_C = look_at((0.45, 0.0, 0.45), (0.5, 0.0, 0.10))   # 시험용 고정 자세(위에서 내려다봄)
STAMP0 = 4_291_000_000


def quat_xyzw(R):
    """회전행렬 → 단위 쿼터니언 [x, y, z, w](w ≥ 0)."""
    from scipy.spatial.transform import Rotation

    q = Rotation.from_matrix(R).as_quat()   # scipy는 xyzw 순서

    return (q if q[3] >= 0 else -q).tolist()


def bolt_mask(x0=120, y=120, head_len=10, head_w=18, shaft_len=34, shaft_w=10):
    """프레임 크기 합성 볼트 마스크(머리 왼쪽)."""
    m = np.zeros((H, W), np.float32)
    m[y - head_w // 2:y - head_w // 2 + head_w, x0:x0 + head_len] = 1
    m[y - shaft_w // 2:y - shaft_w // 2 + shaft_w, x0 + head_len:x0 + head_len + shaft_len] = 1

    return m


def rect_mask():
    """머리·축 구분이 없는 사각형(추출기가 모호로 거름)."""
    m = np.zeros((H, W), np.float32)
    m[115:125, 120:170] = 1

    return m


def model_with(masks_fn):
    """crop(=프레임)마다 masks_fn()의 마스크 목록을 내는 가짜 YOLO."""
    def make(source):
        """가짜 결과 하나."""
        masks = masks_fn()
        k = len(masks)
        boxes = []

        for m in masks:
            ys, xs = np.nonzero(m)
            boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])

        return [FakeResult(FakeBoxes(np.array(boxes, np.float32).reshape(k, 4), np.full(k, 0.9, np.float32),
                                     np.zeros(k, np.float32)),
                           FakeMasks(np.array(masks, np.float32).reshape(k, H, W)) if k else None, (H, W))]

    return FakeModel(make=make)


def sha(path):
    """파일 sha256."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Case:
    """임시 폴더: 이미지·가중치·AI 설정·메타·출력."""

    def __init__(self, d, n=1):
        """n장 이미지(장마다 한 픽셀 다르게)와 가짜 가중치·AI 설정."""
        self.d = Path(d)
        self.images = []

        for i in range(n):
            img = np.full((H, W, 3), 40, np.uint8)
            img[0, i] = (i + 1, 0, 0)
            p = self.d / f"f{i}.png"
            cv2.imwrite(str(p), img)
            self.images.append(str(p))

        weights = self.d / "best.pt"
        weights.write_bytes(b"fake weights for a-to-b capture tests")
        self.ai = self.write_ai(weights, axis="m1")
        self.ai_off = self.write_ai(weights, axis=None, name="ai_off.json")
        self.log = str(self.d / "run.jsonl")
        self.out = str(self.d / "bout")

    def write_ai(self, weights, axis, name="ai.json"):
        """AI 설정 JSON(cpu, 가짜 가중치)."""
        doc = {**AI_FIELDS, "weights_path": str(weights), "weights_sha256": sha(weights), "device": "cpu"}

        if axis is not None:
            doc["axis_extractor"] = axis

        path = self.d / name
        path.write_text(json.dumps(doc), encoding="utf-8")

        return str(path)

    def meta(self, *, frames=None, **over):
        """녹화 메타 사전(프레임마다 촬영 시각·같은 시각 TF)."""
        q = quat_xyzw(T_W_C[:3, :3])
        rows = frames if frames is not None else [
            {"source_index": i, "stamp_ns": STAMP0 + i * 66_000_000, "frame_id": FRAME_ID, "width": W, "height": H,
             "tf": {"parent": "world", "child": FRAME_ID, "stamp_ns": STAMP0 + i * 66_000_000,
                    "translation_m": T_W_C[:3, 3].tolist(), "rotation_xyzw": q}}
            for i in range(len(self.images))]
        doc = {"capture_meta_format": CAPTURE_META_FORMAT, "capture_id": "b168-unit", "source_id": "stored0",
               "world_frame": "world", "inputs_sha256": [sha(p) for p in self.images],
               "camera_info": {"width": W, "height": H, "k": K, "d": [0.0] * 5, "distortion_model": "plumb_bob",
                               "frame_id": FRAME_ID},
               "frames": rows, "provenance": {"note": "unit test synthetic meta"}}
        doc.update(over)

        return doc

    def write_meta(self, doc, name="meta.json"):
        """메타 파일."""
        p = self.d / name
        p.write_text(json.dumps(doc), encoding="utf-8")

        return str(p)

    def b_outputs(self):
        """b_outputs.jsonl 줄."""
        p = Path(self.out) / "b_outputs.jsonl"

        return [] if not p.exists() else [json.loads(t) for t in p.read_text(encoding="utf-8").splitlines()]

    def frames(self):
        """frames.jsonl 줄."""
        p = Path(self.out) / "frames.jsonl"

        return [] if not p.exists() else [json.loads(t) for t in p.read_text(encoding="utf-8").splitlines()]

    def summary(self):
        """summary.json."""
        return json.loads((Path(self.out) / "summary.json").read_text(encoding="utf-8"))

    def processes(self):
        """처리 기록."""
        return [json.loads(t) for t in Path(self.log).read_text(encoding="utf-8").splitlines()
                if json.loads(t)["kind"] == "process"]


class Base(unittest.TestCase):
    """CLI 도우미와 B·C 감시."""

    def setUp(self):
        """임시 폴더·CLI·감시."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        spec = importlib.util.spec_from_file_location("run_pipeline_b168_under_test", SCRIPT)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self.consume_calls, self.c_calls = [], []
        real_consume = b_stage_mod.BStage.consume

        def consume(stage, frame, obs):
            """BStage.consume 감시."""
            self.consume_calls.append((frame.image_seq, obs.image_seq, obs.job_seq))
            return real_consume(stage, frame, obs)

        def c_spy(*a, **k):
            """C 송신 경로가 불리면 실패."""
            self.c_calls.append(a)
            raise AssertionError("no C output in #168")

        for p in (mock.patch.object(b_stage_mod.BStage, "consume", consume),
                  mock.patch.object(b_stage_mod, "make_c_handoff", c_spy),
                  mock.patch.object(sort_part_client.SortPartClient, "send", c_spy),
                  mock.patch.object(sort_part_client.SortPartClient, "__init__", c_spy)):
            p.start()
            self.addCleanup(p.stop)

    def case(self, n=1):
        """새 Case."""
        return Case(self.tmp.name, n)

    def cli(self, c, meta_path=None, *, masks=lambda: [bolt_mask()], ai=None, extra=(), hook=None, images=None,
            out=True):
        """CLI 실행 → (종료 코드, stderr)."""
        argv = ["--images", *(images or c.images), "--detector", "ai", "--ai-config", ai or c.ai, "--log", c.log,
                "--no-detect-timeout", *extra]

        if meta_path is not None:
            argv += ["--a-to-b-capture-meta", meta_path]

        if out and meta_path is not None:
            argv += ["--a-to-b-out", c.out]

        err = io.StringIO()
        model = model_with(masks)
        code = self.mod.main(argv, stderr=err, model_factory=lambda path: model, b_stage_hook=hook)

        return code, err.getvalue()


class RealPathSuccessTest(Base):
    """정상 볼트: 실제 AI 경로 관측 → b_gate → B 1회 → 같은 BOutput 주석 영상."""

    def test_bolt_reaches_b_with_annotation(self):
        """유일한 합성 볼트는 추출기 유효 관측으로 B에 1회 도달하고, 원본 크기 주석 PNG·지문·상관 키·입력 배열 불변이 기록된다."""
        c = self.case()
        code, err = self.cli(c, c.write_meta(c.meta()))

        self.assertEqual(code, 0, err)
        self.assertEqual(self.consume_calls, [(0, 0, 0)])
        outs = c.b_outputs()
        self.assertEqual(len(outs), 1)
        o = outs[0]
        self.assertEqual(o["valid_meaning"], "extractor-format (not accuracy/operational approval)")
        self.assertEqual((o["input"]["observed_face"], o["input"]["selected_index"]), ("axis", 0))
        self.assertEqual(o["input"]["capture_stamp_ns"], STAMP0)
        self.assertTrue(o["frame_image_unchanged"])

        # 입력 봉투(버전 명시): 촬영 정보·TF 결합과 상관 키가 기록되고 확장 필드는 없다
        env = o["b_input"]
        self.assertEqual(env["version"], "eyetoarm.b_input/1")
        self.assertEqual((env["capture"]["stamp_ns"], env["tf"]["stamp_ns"], env["tf"]["child"]), (STAMP0, STAMP0, FRAME_ID))
        self.assertEqual(env["correlation"]["job_seq"], o["input"]["job_seq"])
        self.assertEqual((env["extensions"], env["unsupported"]), ({}, {}))
        self.assertEqual(o["b_adapter"], {"envelope_version": "eyetoarm.b_input/1", "extensions_received": [],
                                          "unsupported_received": [], "used_in_compute": []})

        # 상관 키: 입력 = B 표시 사전 = C 목표 사전
        for key in ("run_id", "capture_generation", "generation", "job_seq", "detector_config_id", "timestamp_ms"):
            self.assertEqual(o["goal"][key], o["input"][key])
            self.assertEqual(o["result_dict"][key], o["input"][key])

        self.assertEqual(o["goal"]["capture_stamp_ns"], STAMP0)
        self.assertEqual(o["result_dict"]["calibration_id"], o["input"]["calibration_id"])

        # 주석 영상: 원본 크기, 지문 일치
        png = Path(c.out) / o["annotation"]
        self.assertEqual(sha(png), o["annotation_png_sha256"])
        self.assertEqual(cv2.imread(str(png)).shape, (H, W, 3))

        s = c.summary()
        self.assertEqual((s["runner_b_calls"], s["b_outputs"], s["annotated_frames"], s["exit_code"]), (1, 1, 1, 0))
        self.assertEqual(s["observation_reasons"], {"VALID": 1})
        self.assertIsNone(s["on_output"])
        self.assertEqual(self.c_calls, [])
        self.assertIn("A->B REAL PATH opt-in", err)

        # 처리 기록에 추출기 끝점이 남는다
        inst = c.processes()[0]["ai_instances"][0]
        self.assertEqual(len(inst["axis_endpoints_px"]), 2)

    def test_meta_calibration_matches_live_path(self):
        """프레임에 붙는 보정 번호는 실시간과 같은 from_camera_info 계산값이다."""
        c = self.case()
        code, err = self.cli(c, c.write_meta(c.meta()))
        self.assertEqual(code, 0, err)

        from types import SimpleNamespace
        cam = from_camera_info(SimpleNamespace(k=K, d=[0.0] * 5, width=W, height=H,
                                               header=SimpleNamespace(frame_id=FRAME_ID)))
        self.assertEqual(c.b_outputs()[0]["input"]["calibration_id"], cam.calibration_id)


class DefaultAndArgsTest(Base):
    """기본 꺼짐·인자 조합."""

    def test_default_run_has_no_b(self):
        """opt-in 없이 axis 설정으로 돌려도 B를 만들지 않고 관측은 CALIBRATION_MISMATCH 무효(프로파일 없음)다."""
        c = self.case()

        with mock.patch.object(b_stage_mod.BStage, "from_config", side_effect=AssertionError("no BStage")) as fc:
            code, err = self.cli(c)

        self.assertEqual(code, 0, err)
        self.assertEqual(fc.call_count, 0)
        self.assertEqual(self.consume_calls, [])
        self.assertEqual([(p["valid"], p["reason"]) for p in c.processes()], [(False, "CALIBRATION_MISMATCH")])
        self.assertFalse(os.path.exists(c.out))

    def test_ai_config_without_axis_is_config_error(self):
        """--a-to-b-capture-meta에 axis_extractor 없는 AI 설정이면 설정 오류 2이고 B 0이다."""
        c = self.case()
        code, err = self.cli(c, c.write_meta(c.meta()), ai=c.ai_off)

        self.assertEqual(code, 2, err)
        self.assertIn("axis_extractor", err)
        self.assertEqual(self.consume_calls, [])

    def test_argument_conflicts(self):
        """--a-to-b-out 단독, --ros-link·--tracking·--b-test-* 결합, OpenCV 검출은 argparse 오류(2)다."""
        c = self.case()
        meta = c.write_meta(c.meta())
        cases = {
            "out_only": ["--images", c.images[0], "--detector", "ai", "--ai-config", c.ai, "--log", c.log,
                         "--a-to-b-out", c.out],
            "tracking": ["--images", c.images[0], "--detector", "ai", "--ai-config", c.ai, "--log", c.log,
                         "--a-to-b-capture-meta", meta, "--a-to-b-out", c.out, "--tracking"],
            "synthetic": ["--images", c.images[0], "--detector", "ai", "--ai-config", c.ai, "--log", c.log,
                       "--a-to-b-capture-meta", meta, "--a-to-b-out", c.out, "--synthetic-a-to-b-fixture", meta,
                       "--synthetic-a-to-b-out", c.out + "x"],
            "opencv": ["--images", c.images[0], "--detect-config", str(ROOT / "config/vision/opencv_planar.json"),
                       "--log", c.log, "--a-to-b-capture-meta", meta, "--a-to-b-out", c.out],
            "ros_link": ["--ros-link", "--detector", "ai", "--ai-config", c.ai, "--log", c.log,
                         "--a-to-b-capture-meta", meta, "--a-to-b-out", c.out],
        }

        for name, argv in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(SystemExit) as ctx:
                    self.mod.main(argv, stderr=io.StringIO(), ros_available=lambda: True,
                                  live_source_factory=lambda *a: (_ for _ in ()).throw(AssertionError("no live")))
                self.assertEqual(ctx.exception.code, 2)

        self.assertEqual(self.consume_calls, [])


class InvalidObservationTest(Base):
    """추출 무효·복수: B 0, 사유 기록, 주석 영상 없음."""

    def test_ambiguous_and_multiple_not_sent(self):
        """사각형(머리 모호)·볼트 2개는 B 0이고 frames.jsonl에 AXIS_HEAD_AMBIGUOUS·MULTIPLE_OBJECTS가 남으며 주석 영상이 없다."""
        for reason, masks in (("AXIS_HEAD_AMBIGUOUS", lambda: [rect_mask()]),
                              ("MULTIPLE_OBJECTS", lambda: [bolt_mask(y=60), bolt_mask(y=180)])):
            with self.subTest(reason=reason):
                c = Case(tempfile.mkdtemp(dir=self.tmp.name))
                self.consume_calls.clear()
                code, err = self.cli(c, c.write_meta(c.meta()), masks=masks)
                self.assertEqual(code, 0, err)
                self.assertEqual(self.consume_calls, [])
                self.assertEqual([(f["valid"], f["reason"]) for f in c.frames()], [(False, reason)])
                self.assertEqual(c.summary()["annotated_frames"], 0)
                self.assertEqual(list((Path(c.out) / "annotated").iterdir()), [])


class MetaMismatchTest(Base):
    """메타 불일치: 프레임에 촬영 메타를 붙이지 않아 B 0, 사유 기록."""

    def run_rows(self, mutate):
        """기본 메타의 프레임 행을 바꿔 실행."""
        c = Case(tempfile.mkdtemp(dir=self.tmp.name))
        doc = c.meta()
        mutate(doc["frames"][0])
        self.consume_calls.clear()
        code, err = self.cli(c, c.write_meta(doc))

        return code, err, c

    def test_per_frame_mismatches(self):
        """frame id·해상도·TF 시각·TF 자식 프레임이 어긋나면 B 0, 관측은 CALIBRATION_MISMATCH, summary에 사유가 남는다."""
        cases = {"META_FRAME_ID_MISMATCH": lambda r: r.update(frame_id="other_frame"),
                 "META_SIZE_MISMATCH": lambda r: r.update(width=W + 1),
                 "META_TF_STAMP_MISMATCH": lambda r: r["tf"].update(stamp_ns=r["stamp_ns"] + 1)}

        for reason, mutate in cases.items():
            with self.subTest(reason=reason):
                code, err, c = self.run_rows(mutate)
                self.assertEqual(code, 0, err)
                self.assertEqual(self.consume_calls, [])
                self.assertEqual([m["reason"] for m in c.summary()["meta_mismatches"]], [reason])
                self.assertEqual([(f["valid"], f["reason"]) for f in c.frames()], [(False, "CALIBRATION_MISMATCH")])

        code, err, c = self.run_rows(lambda r: r["tf"].update(child="other_frame"))
        self.assertEqual(self.consume_calls, [])
        self.assertEqual(c.summary()["meta_mismatches"][0]["reason"], "META_TF_STAMP_MISMATCH")

    def test_missing_frame_meta(self):
        """2장 중 1장만 메타가 있으면 메타 없는 프레임은 B 0(META_MISSING), 있는 프레임만 B 1회다."""
        c = self.case(2)
        doc = c.meta()
        doc["frames"] = doc["frames"][:1]
        code, err = self.cli(c, c.write_meta(doc))

        self.assertEqual(code, 0, err)
        self.assertEqual([cc[0] for cc in self.consume_calls], [0])
        self.assertEqual(c.summary()["meta_mismatches"], [{"image_seq": 1, "source_index": 1, "reason": "META_MISSING"}])

    def test_tf_missing_and_timeout_are_b_skip_without_drawing(self):
        """그 시각 TF가 녹화되지 않았거나(tf null) 조회가 50 ms를 넘으면 B는 TF_UNAVAILABLE SKIP이고 주석 영상을 그리지 않는다."""
        c = Case(tempfile.mkdtemp(dir=self.tmp.name))
        doc = c.meta()
        doc["frames"][0]["tf"] = None
        code, err = self.cli(c, c.write_meta(doc))

        self.assertEqual(code, 0, err)
        o = c.b_outputs()[0]
        self.assertEqual((o["b_reason"], o["goal"]["kind"], o["annotation"]), ("TF_UNAVAILABLE", "SKIP", None))
        self.assertEqual(c.summary()["annotated_frames"], 0)

        # 다른 프레임(앞 프레임) TF가 있어도 대체하지 않는다
        c = Case(tempfile.mkdtemp(dir=self.tmp.name), n=2)
        doc = c.meta()
        doc["frames"][1]["tf"] = None
        code, err = self.cli(c, c.write_meta(doc))

        self.assertEqual(code, 0, err)
        outs = c.b_outputs()
        self.assertEqual([o["input"]["image_seq"] for o in outs], [0, 1])
        self.assertIsNotNone(outs[0]["annotation"])
        self.assertEqual((outs[1]["b_reason"], outs[1]["annotation"]), ("TF_UNAVAILABLE", None))

        c = Case(tempfile.mkdtemp(dir=self.tmp.name))

        def slow(stage):
            """조회 앞에 60 ms 지연."""
            inner = stage.tf_lookup.inner

            def delayed(frame):
                """지연 뒤 원래 값."""
                time.sleep(0.06)
                return inner(frame)

            stage.tf_lookup.inner = delayed

        code, err = self.cli(c, c.write_meta(c.meta()), hook=slow)
        self.assertEqual(code, 0, err)
        o = c.b_outputs()[0]
        self.assertEqual((o["b_reason"], o["annotation"]), ("TF_UNAVAILABLE", None))
        self.assertEqual(c.summary()["tf"]["timed_out"], 1)

    def test_meta_file_errors(self):
        """소스 ID·입력 지문·쿼터니언·중복 프레임·지원하지 않는 왜곡 모델 오류는 실행 전 2이고 B 0이다."""
        def bad_q(doc):
            """단위가 아닌 쿼터니언."""
            doc["frames"][0]["tf"]["rotation_xyzw"] = [0.0, 0.0, 0.0, 2.0]

        def dup(doc):
            """같은 source_index 두 번."""
            doc["frames"].append(dict(doc["frames"][0]))

        def model(doc):
            """왜곡이 있는 비 plumb_bob."""
            doc["camera_info"].update(distortion_model="equidistant", d=[0.1, 0.0, 0.0, 0.0, 0.0])

        def inputs(doc):
            """입력 지문 다름."""
            doc["inputs_sha256"] = ["0" * 64]

        for name, mutate, extra in (("source_id", None, ("--source-id", "cam9")), ("inputs", inputs, ()),
                                    ("quaternion", bad_q, ()), ("duplicate", dup, ()), ("distortion", model, ())):
            with self.subTest(case=name):
                c = Case(tempfile.mkdtemp(dir=self.tmp.name))
                doc = c.meta()

                if mutate is not None:
                    mutate(doc)

                code, err = self.cli(c, c.write_meta(doc), extra=extra)
                self.assertEqual(code, 2, err)
                self.assertEqual(self.consume_calls, [])
                self.assertFalse(os.path.exists(c.log))


class EnvelopeRejectionTest(Base):
    """입력 봉투 거부: B를 부르지 않고 기록, 실행은 1로 끝난다."""

    def test_rejected_envelope_not_sent(self):
        """봉투 검사가 실패하면(시험 주입) BStage.consume 0회, b_input_rejected 1줄, 주석 영상 0, summary 문제 기록, 종료 1이다."""
        from eyetoarm.vision import a_to_b_capture_recorder
        from eyetoarm.vision.a_to_b_envelope import AToBEnvelopeError

        c = self.case()

        with mock.patch.object(a_to_b_capture_recorder, "make_a_to_b_envelope", side_effect=AToBEnvelopeError("injected binding error")):
            code, err = self.cli(c, c.write_meta(c.meta()))

        self.assertEqual(code, 1, err)
        self.assertEqual(self.consume_calls, [])
        lines = c.b_outputs()
        self.assertEqual([(l["kind"], l["error"]) for l in lines], [("b_input_rejected", "injected binding error")])
        s = c.summary()
        self.assertEqual((s["b_input_rejected"], s["b_outputs"], s["annotated_frames"]), (1, 0, 0))
        self.assertIn("1 a_to_b_envelope envelope(s) rejected (B not called)", s["problems"])


class ExtensionDeliveryTest(Base):
    """확장 자료가 봉투로 B 입구 어댑터까지 가고, B 계산은 바뀌지 않는다(F1)."""

    def test_extensions_reach_adapter_without_changing_b(self):
        """지원 확장(tracking)과 미지원 확장(depth_hint)을 실으면 어댑터가 받아 기록하고(계산 미사용), B 결과·목표 종류는 확장 없는 실행과 같다."""
        def plain_run():
            """확장 없는 기준 실행."""
            c = Case(tempfile.mkdtemp(dir=self.tmp.name))
            code, err = self.cli(c, c.write_meta(c.meta()))
            self.assertEqual(code, 0, err)
            return c.b_outputs()[0]

        def source(frame, obs):
            """봉투 확장(시험 주입): 결합 키는 그 프레임·작업."""
            binding = {"image_seq": obs.image_seq, "job_seq": obs.job_seq, "generation": obs.generation,
                       "capture_stamp_ns": frame.ros_stamp_ns}
            return {"tracking": {"binding": binding, "part_track_id": 11}, "depth_hint": {"mm": [1, 2]}}

        base = plain_run()
        c = Case(tempfile.mkdtemp(dir=self.tmp.name))
        err = io.StringIO()
        model = model_with(lambda: [bolt_mask()])
        argv = ["--images", *c.images, "--detector", "ai", "--ai-config", c.ai, "--log", c.log, "--no-detect-timeout",
                "--a-to-b-capture-meta", c.write_meta(c.meta()), "--a-to-b-out", c.out]
        code = self.mod.main(argv, stderr=err, model_factory=lambda p: model, a_to_b_extension_source=source)

        self.assertEqual(code, 0, err.getvalue())
        o = c.b_outputs()[0]
        self.assertEqual(o["b_adapter"]["extensions_received"], ["tracking"])
        self.assertEqual(o["b_adapter"]["unsupported_received"], ["depth_hint"])
        self.assertEqual(o["b_adapter"]["used_in_compute"], [])
        self.assertEqual(o["b_input"]["extensions"]["tracking"]["part_track_id"], 11)
        self.assertEqual(o["b_input"]["unsupported"], {"depth_hint": {"mm": [1, 2]}})
        self.assertEqual((o["result_dict"]["spec"], o["result_dict"]["center_w_mm"], o["goal"]["kind"]),
                         (base["result_dict"]["spec"], base["result_dict"]["center_w_mm"], base["goal"]["kind"]))
        self.assertEqual(c.summary()["b_adapter"]["extensions_received"], {"tracking": 1})

    def test_bad_extension_rejects_envelope(self):
        """결합 키가 다른 지원 확장은 봉투 거부(B 0, 종료 1)로 끝난다."""
        c = self.case()
        err = io.StringIO()
        model = model_with(lambda: [bolt_mask()])
        argv = ["--images", *c.images, "--detector", "ai", "--ai-config", c.ai, "--log", c.log, "--no-detect-timeout",
                "--a-to-b-capture-meta", c.write_meta(c.meta()), "--a-to-b-out", c.out]
        bad = {"tracking": {"binding": {"image_seq": 0, "job_seq": 9, "generation": 0, "capture_stamp_ns": STAMP0},
                            "part_track_id": 1}}
        code = self.mod.main(argv, stderr=err, model_factory=lambda p: model, a_to_b_extension_source=lambda f, o: bad)

        self.assertEqual(code, 1, err.getvalue())
        self.assertEqual(self.consume_calls, [])
        self.assertEqual(c.b_outputs()[0]["kind"], "b_input_rejected")


class BadExtensionRunContinuesTest(Base):
    """잘못된 확장 수치(F3)는 실행을 멈추지 않고 그 프레임만 b_input_rejected로 기록된다."""

    def test_overflow_confidence_rejected_and_next_frame_processed(self):
        """1장째 확장 confidence=10**400은 OverflowError로 새지 않고 b_input_rejected, 2장째 정상 프레임은 B로 간다. 종료 1(거부 있음)."""
        c = self.case(2)
        err = io.StringIO()
        model = model_with(lambda: [bolt_mask()])

        def source(frame, obs):
            """1장째만 잘못된 확장."""
            if obs.image_seq != 0:
                return None

            b = {"image_seq": obs.image_seq, "job_seq": obs.job_seq, "generation": obs.generation,
                 "capture_stamp_ns": frame.ros_stamp_ns}
            return {"ai_instances": {"binding": b, "pixel_basis": "raw",
                                     "items": [{"index": 0, "class_name": "bolt", "confidence": 10 ** 400,
                                                "rejected": None, "polygon_px": None}]}}

        argv = ["--images", *c.images, "--detector", "ai", "--ai-config", c.ai, "--log", c.log, "--no-detect-timeout",
                "--a-to-b-capture-meta", c.write_meta(c.meta()), "--a-to-b-out", c.out]
        code = self.mod.main(argv, stderr=err, model_factory=lambda p: model, a_to_b_extension_source=source)

        self.assertEqual(code, 1, err.getvalue())
        lines = c.b_outputs()
        self.assertEqual([l["kind"] for l in lines], ["b_input_rejected", "b_output"])
        self.assertIn("confidence", lines[0]["error"])
        self.assertEqual([cc[0] for cc in self.consume_calls], [1])
        s = c.summary()
        self.assertEqual((s["frames"], s["b_input_rejected"], s["b_outputs"], s["runner_exit_code"]), (2, 1, 1, 0))


class OutputSafetyTest(Base):
    """덮어쓰기 거부·입력 불변·C 미연결."""

    def test_existing_outputs_refused(self):
        """출력 폴더나 기록 파일이 이미 있으면 2이고 기존 내용은 그대로다."""
        c = self.case()
        meta = c.write_meta(c.meta())
        code, err = self.cli(c, meta)
        self.assertEqual(code, 0, err)
        before = (Path(c.out) / "b_outputs.jsonl").read_bytes()

        code, err = self.cli(c, meta)
        self.assertEqual(code, 2)
        self.assertIn("refusing to overwrite", err)
        self.assertEqual((Path(c.out) / "b_outputs.jsonl").read_bytes(), before)

        c.out = str(c.d / "bout2")
        code, err = self.cli(c, meta)   # 기록 파일(run.jsonl)이 이미 있음
        self.assertEqual(code, 2)
        self.assertFalse(os.path.exists(c.out))

    def test_on_output_must_stay_none(self):
        """누군가 on_output을 연결하면 실행 전 1이고 B·C 0이다."""
        c = self.case()
        code, err = self.cli(c, c.write_meta(c.meta()), hook=lambda s: setattr(s, "on_output", lambda out: None))

        self.assertEqual(code, 1, err)
        self.assertEqual(self.consume_calls, [])
        self.assertEqual(self.c_calls, [])

    def test_render_does_not_modify_frame(self):
        """주석 생성 중 입력 배열이 바뀌면(시험 주입) 실행 실패 1로 끝난다(입력 불변 검사가 실제로 동작)."""
        c = self.case()

        def corrupt(stage):
            """render_center가 입력 배열을 고치는 것처럼 바꾼다."""
            real = stage.render_center

            def bad(image, outputs, guides=True):
                """입력을 고친 뒤 원래 결과."""
                image.setflags(write=True)
                image[0, 0, 0] ^= 1
                return real(image, outputs, guides)

            stage.render_center = bad

        code, err = self.cli(c, c.write_meta(c.meta()), hook=corrupt)
        self.assertEqual(code, 1, err)


if __name__ == "__main__":
    unittest.main()
