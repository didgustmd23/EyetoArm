"""#189 단계 L CLI 시험: `run_pipeline.py --b-input list`(저장 입력 `--capture-meta` 경로 전용).

근거: 계획 `_workspace/189_builder2_stage_L_plan.md` r0 §5·§6·§7(reviewer 승인 hop 3).

- 모두 CLI `main()`으로 실행한다: 저장 이미지 → `CaptureMetaSource` → 실행기 → `AiSegDetector`(가짜 모델, 실제 convert·M1) →
  코어 → `b_list_gate` → `CaptureListRecorder` → 목록 봉투 v2 → 시험 전용 가짜 목록 소비자(`list_consumer_factory` 주입).
- 주입이 없고 B 목록 입구(`process_frame`)가 없으면 단계 L은 시작 때 거부되어야 한다(단계 T로 떨어지지 않음). 실제 B에는
  `process_frame`이 생겼으므로(d4fab37) 거부 시험은 그 입구가 없는 가짜 B 단계형을 끼워 확인한다.
- 단계 T 단일 소비(`BStage.consume`)와 C 송신은 불리지 않아야 한다(기존 시험 기반 클래스의 감시 그대로).
"""
import io
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from eyetoarm.transforms import b_stage as b_stage_mod

from test_run_pipeline_capture_observation import FRAME_ID, STAMP0, Base, Case, bolt_mask, model_with, rect_mask


class FakeListConsumer:
    """시험 전용 가짜 B 목록 소비자: 받은 v2 봉투를 보관하고 후보 수를 돌려준다."""

    def __init__(self):
        """받은 봉투."""
        self.envelopes = []
        self.stages = []

    def factory(self, stage):
        """`list_consumer_factory(stage)`: 받은 BStage를 보관하고 소비 함수를 돌려준다."""
        self.stages.append(stage)

        return self.consume

    def consume(self, envelope):
        """봉투 하나를 받는다(계산 없음)."""
        self.envelopes.append(envelope)

        return {"fake": True, "candidates": len(envelope.observations.candidates)}


def plan_masks(*plans):
    """호출마다 차례로 다른 마스크 목록을 내는 함수(프레임 순서)."""
    it = itertools.cycle(plans)

    return lambda: next(it)()


class StageLBase(Base):
    """단계 L CLI 도우미."""

    def cli_list(self, c, meta_path, *, masks, consumer=None, extra=()):
        """`--b-input list`로 실행 → (종료 코드, stderr)."""
        argv = ["--images", *c.images, "--detector", "ai", "--ai-config", c.ai, "--log", c.log,
                "--no-detect-timeout", "--capture-meta", meta_path, "--observation-out", c.out, "--b-input", "list",
                *extra]
        err = io.StringIO()
        model = model_with(masks)
        code = self.mod.main(argv, stderr=err, model_factory=lambda path: model,
                             list_consumer_factory=None if consumer is None else consumer.factory)

        return code, err.getvalue()

    def lines(self, c, name):
        """출력 폴더 JSONL 줄."""
        p = Path(c.out) / name

        return [] if not p.exists() else [json.loads(t) for t in p.read_text(encoding="utf-8").splitlines()]


class StageLPathTest(StageLBase):
    """저장 입력 끝까지: 프레임 3장(볼트 1·볼트 2·머리 모호) → 목록 2회(길이 1·2)."""

    def test_fake_consumer_receives_lists(self):
        """가짜 소비자는 v2 봉투 2개(원 인덱스 [0]·[0,1], 촬영 시각 TF)를 받고 BStage.consume·C는 불리지 않으며 주석 영상이 없다."""
        c = self.case(3)
        fake = FakeListConsumer()
        masks = plan_masks(lambda: [bolt_mask()], lambda: [bolt_mask(y=60), bolt_mask(y=180)], lambda: [rect_mask()])
        code, err = self.cli_list(c, c.write_meta(c.meta()), masks=masks, consumer=fake)

        self.assertEqual(code, 0, err)
        self.assertEqual(self.consume_calls, [])
        self.assertEqual(self.c_calls, [])
        self.assertEqual([e.version for e in fake.envelopes], ["eyetoarm.b_input/2"] * 2)
        self.assertEqual([e.observations.instance_indices for e in fake.envelopes], [(0,), (0, 1)])
        self.assertEqual([e.observations.frame.image_seq for e in fake.envelopes], [0, 1])
        self.assertTrue(all(e.observations.tf.stamp_ns == e.observations.capture.stamp_ns for e in fake.envelopes))
        self.assertEqual(fake.envelopes[0].observations.capture.stamp_ns, STAMP0)
        self.assertEqual(fake.envelopes[0].observations.tf.child, FRAME_ID)
        self.assertIsNone(fake.stages[0].on_output)

        lists = self.lines(c, "list_frames.jsonl")
        self.assertEqual([r["kind"] for r in lists], ["b_list_input"] * 2)
        self.assertEqual([r["b_input"]["candidate_count"] for r in lists], [1, 2])
        self.assertEqual([r["consumer_result"] for r in lists],
                         [{"fake": True, "candidates": 1}, {"fake": True, "candidates": 2}])
        self.assertTrue(all(r["consumer_kind"] == "injected_test_only" and r["frame_image_unchanged"] for r in lists))

        frames = self.lines(c, "frames.jsonl")
        self.assertEqual([(f["stage_t_valid"], f["stage_t_reason"]) for f in frames],
                         [(True, None), (False, "MULTIPLE_OBJECTS"), (False, "AXIS_HEAD_AMBIGUOUS")])

        s = c.summary()
        self.assertEqual((s["exit_code"], s["runner_b_list_calls"], s["runner_b_calls"], s["b_list_calls_recorded"],
                          s["candidates_sent"], s["b_input_rejected"]), (0, 2, 0, 2, 3, 0))
        self.assertEqual(s["b_input_version"], "eyetoarm.b_input/2")
        self.assertFalse((Path(c.out) / "annotated").exists())

        procs = c.processes()
        self.assertEqual([(p["list_candidate_count"], p["list_indices"]) for p in procs], [(1, [0]), (2, [0, 1]), (0, [])])
        self.assertEqual([p["axis_valid_count"] for p in procs], [1, 2, 0])
        self.assertIn("A->B STAGE L opt-in", err)

    def test_missing_tf_meta_still_sends_list_without_tf(self):
        """그 촬영 시각 TF가 메타에 없으면 tf=None 목록을 보낸다(B가 프레임 전체 TF_UNAVAILABLE로 처리)."""
        c = self.case(1)
        doc = c.meta()
        doc["frames"][0]["tf"] = None   # 녹화 메타 형식: 그 시각 TF 없음은 null
        fake = FakeListConsumer()
        code, err = self.cli_list(c, c.write_meta(doc), masks=lambda: [bolt_mask(y=60), bolt_mask(y=180)], consumer=fake)

        self.assertEqual(code, 0, err)
        self.assertEqual(len(fake.envelopes), 1)
        self.assertIsNone(fake.envelopes[0].observations.tf)
        self.assertIsNone(self.lines(c, "list_frames.jsonl")[0]["b_input"]["tf"])


class StageLRefusalTest(StageLBase):
    """시작 거부: B 목록 소비자 없음·금지 조합."""

    def test_no_process_frame_refused_before_files(self):
        """주입 없고 B 단계에 목록 입구가 없으면 종료 1, 기록·출력 파일을 만들지 않고 단계 T 소비도 하지 않는다(가짜 B 단계형)."""
        self.assertTrue(callable(getattr(b_stage_mod.BStage, "process_frame", None)))   # 실제 B에는 있음

        class NoListBStage(b_stage_mod.BStage):
            """목록 입구(`process_frame`)가 없는 가짜 B 단계형(단계 T `consume` 등은 실제 그대로)."""

            process_frame = None

        c = self.case(1)

        with mock.patch.object(b_stage_mod, "BStage", NoListBStage):
            code, err = self.cli_list(c, c.write_meta(c.meta()), masks=lambda: [bolt_mask()])

        self.assertEqual(code, 1, err)
        self.assertIn("list consumer unavailable", err)
        self.assertIn("not falling back to stage T", err)
        self.assertFalse(Path(c.log).exists())
        self.assertFalse(Path(c.out).exists())
        self.assertEqual(self.consume_calls, [])

    def test_argument_conflicts(self):
        """--b-input list는 --capture-meta 없이·합성 fixture와 쓸 수 없고 값은 single·list만(인자 오류 2)."""
        c = self.case(1)
        base = ["--images", *c.images, "--detector", "ai", "--ai-config", c.ai, "--log", c.log]

        for argv in (base + ["--b-input", "list"], base + ["--b-input", "multi"],
                     base + ["--b-input", "list", "--synthetic-observation-fixture", "x.json",
                             "--synthetic-observation-out", "y.jsonl"]):
            with self.subTest(argv=argv[-4:]):
                with self.assertRaises(SystemExit) as cm:
                    self.mod.main(argv, stderr=io.StringIO())

                self.assertEqual(cm.exception.code, 2)

    def test_axis_off_config_is_config_error(self):
        """축 경로를 끈 AI 설정의 단계 L은 기존 --capture-meta 규칙대로 설정 오류 2."""
        c = self.case(1)
        argv = ["--images", *c.images, "--detector", "ai", "--ai-config", c.ai_off, "--log", c.log,
                "--capture-meta", c.write_meta(c.meta()), "--observation-out", c.out, "--b-input", "list"]

        self.assertEqual(self.mod.main(argv, stderr=io.StringIO(), list_consumer_factory=FakeListConsumer().factory), 2)


class StageTDefaultTest(StageLBase):
    """단계 T 기본: 옵션 생략과 --b-input single은 같은 경로·같은 기록 키."""

    def test_single_equals_default(self):
        """두 실행 모두 BStage.consume 1회, b_input/1 봉투, 기록에 list 키·b_input 키가 없다(값은 run_id·시간 외 같음)."""
        texts = []

        for extra in ((), ("--b-input", "single")):
            c = Case(tempfile.mkdtemp(dir=self.tmp.name))
            self.consume_calls.clear()
            code, err = self.cli(c, c.write_meta(c.meta()), extra=extra)

            self.assertEqual(code, 0, err)
            self.assertEqual(len(self.consume_calls), 1)
            self.assertEqual(c.b_outputs()[0]["b_input"]["version"], "eyetoarm.b_input/1")
            text = Path(c.log).read_text(encoding="utf-8")
            self.assertNotIn("list_candidate_count", text)
            self.assertNotIn('"b_input"', text)
            texts.append([sorted(json.loads(t)) for t in text.splitlines()])

        self.assertEqual(texts[0], texts[1])


if __name__ == "__main__":
    unittest.main()
