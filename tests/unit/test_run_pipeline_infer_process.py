"""`run_pipeline.py --infer-process` 연결 시험. 자식은 `ai_process_fake` 공장(가짜 모델을 쓰는 실제 AiSegDetector)을 쓴다.

확인: 인자 조합, 저장 입력·capture·실시간 경로의 대리 객체 연결과 정리, 실패 종료 코드, 꺼짐이면 기존과 같은 기록.
"""
import importlib.util
import io
import json
import os
import unittest
from pathlib import Path

import cv2
import numpy as np

import ai_process_fake as fake
from handoff_isolation import isolate_cli
from eyetoarm.vision.ai_process import SHM_DIR, SHM_PREFIX
from test_ai_process import canon
from test_live_sort_part import CliBase
from test_run_pipeline_capture_observation import Base as CaptureBase
from test_run_pipeline_live_observation import FakeTf, Scenario

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPT = ROOT / "scripts" / "run_pipeline.py"
TIMING_KEYS = {"infer_ms", "post_ms", "process_ms", "timestamp_ms"}   # 저장 입력 시각은 PC 수신 시각


def child_env(**extra):
    """자식이 src·시험 폴더를 import하도록 한 환경(자동 설치 방지 변수는 대리 객체가 넣음)."""
    env = {k: v for k, v in os.environ.items() if k not in ("YOLO_AUTOINSTALL", "YOLO_OFFLINE")}
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(HERE), env.get("PYTHONPATH", "")])
    env.update(extra)

    return env


def options(**kw):
    """`infer_process_options` 시험값: 가짜 공장·환경·남은 슬롯 정리 끔(실제 /dev/shm을 건드리지 않음)."""
    return {"factory": "ai_process_fake:make", "env": child_env(), "sweep": no_sweep, **kw}


def no_sweep():
    """남은 슬롯 정리 대역: 아무것도 지우지 않는다."""
    return {"removed": [], "kept": {}}


def shm_entries():
    """이 프로세스(부모 pid 이름)의 /dev/shm 슬롯. 동시에 도는 다른 pytest의 슬롯은 세지 않는다."""
    own = f"{SHM_PREFIX}{os.getpid()}_"
    return sorted(n for n in os.listdir(SHM_DIR) if n.startswith(own))


def child_processes():
    """이 프로세스가 띄운(`--parent-pid` 자기 pid) 살아 있는 자식 진입점 수. 다른 pytest의 자식은 세지 않는다."""
    count = 0
    own = str(os.getpid()).encode()

    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue

        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except OSError:
            continue

        mine = b"--parent-pid" in cmd and cmd.index(b"--parent-pid") + 1 < len(cmd) and \
            cmd[cmd.index(b"--parent-pid") + 1] == own

        if b"eyetoarm.vision.ai_process_child" in cmd and mine and state != "Z":
            count += 1

    return count


RUN_KEYS = {"run_id", "goal_id", "sort_job_id", "timestamp_ms", "valid_until_ns"}   # 실행마다 다른 값


def drop_run_keys(value):
    """실행마다 달라지는 키(실행 ID·그로부터 만든 번호·PC 시각)를 재귀로 뺀다."""
    if isinstance(value, dict):
        return {k: drop_run_keys(v) for k, v in value.items() if k not in RUN_KEYS}

    if isinstance(value, list):
        return [drop_run_keys(v) for v in value]

    return value


def records(path):
    """JSONL 기록."""
    return [json.loads(t) for t in Path(path).read_text(encoding="utf-8").splitlines()]


def comparable(recs):
    """처리·AI 기록의 비교 형태(시간 필드·실행 ID 제외)."""
    out = []

    for r in recs:
        if r["kind"] in ("process", "ai_geometry"):
            out.append({k: canon(v) for k, v in r.items() if k not in TIMING_KEYS | {"run_id"}})

    return out


class LeakCheck(unittest.TestCase):
    """시험마다 자식·슬롯 잔여 0 확인."""

    def setUp(self):
        """시작 상태."""
        super().setUp()
        self.shm_before = shm_entries()
        self.children_before = child_processes()
        self.addCleanup(self.check_leaks)

    def check_leaks(self):
        """자식·슬롯이 늘지 않았다."""
        self.assertEqual(shm_entries(), self.shm_before)
        self.assertEqual(child_processes(), self.children_before)


class StoredCase(LeakCheck):
    """저장 이미지·가짜 가중치·AI 설정 임시 폴더."""

    def setUp(self):
        """CLI 모듈·임시 폴더."""
        super().setUp()
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        spec = importlib.util.spec_from_file_location("run_pipeline_infer_under_test", SCRIPT)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        isolate_cli(self.mod)   # 운영 포트로 detect 영상을 보내지 않는다
        weights = self.d / "best.pt"
        weights.write_bytes(b"fake weights for infer-process cli tests")
        import hashlib
        doc = {**json.loads(json.dumps(__import__("test_ai_config").AI_FIELDS)), "weights_path": str(weights),
               "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(), "device": "cpu",
               "min_mask_area_px": 0, "axis_extractor": "m2"}
        self.ai = self.d / "ai.json"
        self.ai.write_text(json.dumps(doc), encoding="utf-8")

    def images(self, codes):
        """코드(B)·위치(G)를 왼쪽 위 픽셀에 넣은 200×120 PNG들."""
        paths = []

        for i, code in enumerate(codes):
            img = np.full((120, 200, 3), 30, np.uint8)
            img[0, 0] = (code, 7 * i, 0)
            p = self.d / f"img{i}.png"
            cv2.imwrite(str(p), img)
            paths.append(str(p))

        return paths

    def cli(self, argv, **kw):
        """main 실행 → (종료 코드, stderr)."""
        err = io.StringIO()

        try:
            code = self.mod.main(argv, stderr=err, **kw)
        except SystemExit as exc:
            code = exc.code

        return code, err.getvalue()

    def stored(self, images, log, *flags, **kw):
        """저장 입력 AI 실행."""
        return self.cli(["--images", *images, "--detector", "ai", "--ai-config", str(self.ai), "--log", str(log),
                         *flags], **kw)


class ArgsTest(StoredCase):
    """인자 조합 오류는 종료 2."""

    def test_flag_combinations(self):
        """--infer-process는 AI 전용, --infer-torch-threads는 --infer-process 필요, synthetic fixture와 결합 불가."""
        imgs = self.images([0])
        cases = [
            (["--images", *imgs, "--detect-config", "x.json", "--log", str(self.d / "a.jsonl"), "--infer-process"],
             "--infer-process/--no-infer-process are for --detector ai only"),
            (["--images", *imgs, "--detector", "ai", "--ai-config", str(self.ai), "--log", str(self.d / "b.jsonl"),
              "--infer-torch-threads", "2"], "--infer-torch-threads requires the inference process"),
            (["--images", *imgs, "--detector", "ai", "--ai-config", str(self.ai), "--log", str(self.d / "c.jsonl"),
              "--infer-process", "--synthetic-observation-fixture", "f.json", "--synthetic-observation-out",
              str(self.d / "o")], "--synthetic-observation-fixture"),
            (["--images", *imgs, "--detector", "ai", "--ai-config", str(self.ai), "--log", str(self.d / "d.jsonl"),
              "--infer-process", "--no-infer-process"], "mutually exclusive"),
        ]

        for argv, msg in cases:
            with self.subTest(msg=msg):
                import contextlib
                buf = io.StringIO()

                with contextlib.redirect_stderr(buf):
                    code, _ = self.cli(argv)

                self.assertEqual(code, 2)
                self.assertIn(msg, buf.getvalue())


class StoredPathTest(StoredCase):
    """저장 입력 경로: 켬·끔 결과 동일, 기록 키, 실패 종료 코드."""

    def test_on_matches_off_and_records_infer_process(self):
        """켬은 끔과 같은 처리·AI 기록(시간 제외)을 내고, run_meta·model_ready·infer_process 기록이 더해진다."""
        imgs = self.images([0, 0, 0])
        off_log, on_log = self.d / "off.jsonl", self.d / "on.jsonl"
        code_off, err_off = self.stored(imgs, off_log, "--log-instance-geometry",
                                        model_factory=fake.FakeModel)
        code_on, err_on = self.stored(imgs, on_log, "--log-instance-geometry", "--infer-process",
                                      "--infer-torch-threads", "2", infer_process_options=options())

        self.assertEqual((code_off, code_on), (0, 0), err_off + err_on)
        off, on = records(off_log), records(on_log)
        self.assertEqual(comparable(on), comparable(off))
        self.assertEqual(len([r for r in on if r["kind"] == "process"]), 3)

        meta_off = next(r for r in off if r["kind"] == "run_meta")
        meta_on = next(r for r in on if r["kind"] == "run_meta")
        self.assertNotIn("infer_process", meta_off["detector"])
        self.assertEqual((meta_on["detector"]["infer_process"], meta_on["detector"]["infer_process_source"]),
                         (True, "argument"))
        marks = {"infer_process", "infer_process_source"}
        self.assertEqual({k: v for k, v in meta_on["detector"].items() if k not in marks}, meta_off["detector"])
        self.assertEqual(meta_on["detector_config_id"], meta_off["detector_config_id"])

        ready = next(r for r in on if r["kind"] == "model_ready")
        self.assertEqual(ready["infer_process"]["torch_threads_requested"], 2)
        self.assertFalse(any(r["kind"] == "infer_process" for r in off))
        summary = next(r for r in on if r["kind"] == "infer_process")
        self.assertEqual((summary["enabled"], summary["source"], summary["results"], summary["failed"]),
                         (True, "argument", 3, None))
        self.assertEqual(summary["exit"], {"returncode": 0, "forced": False})
        self.assertEqual(summary["times"]["roundtrip_ms"]["n"], 3)
        self.assertIn("infer_process=on failed=None", err_on)
        self.assertIn("run_pipeline: infer_process=on source=argument\n", err_on)
        self.assertNotIn("infer_process=", err_off)

    def test_child_death_exits_1_with_reason(self):
        """자식이 죽으면 종료 1, stderr·infer_process 기록에 사유가 남고 그 뒤 영상은 처리하지 않는다."""
        imgs = self.images([0, fake.DIE, 0, 0])
        log = self.d / "die.jsonl"
        code, err = self.stored(imgs, log, "--infer-process", infer_process_options=options())

        self.assertEqual(code, 1, err)
        self.assertIn("run_pipeline: infer process failed: infer_process_died", err)
        recs = records(log)
        summary = next(r for r in recs if r["kind"] == "infer_process")
        self.assertIn("infer_process_died", summary["failed"])
        self.assertLessEqual(len([r for r in recs if r["kind"] == "process"]), 3)
        self.assertTrue(any(r.get("invalidation_kind") == "INFERENCE_ERROR" for r in recs if r["kind"] == "event"))

    def test_prepare_failure_exits_1(self):
        """자식 준비 실패는 기존 준비 실패와 같이 MODEL_CONFIG_INVALID 사건 + 종료 1."""
        log = self.d / "prep.jsonl"
        code, err = self.stored(self.images([0]), log, "--infer-process",
                                infer_process_options=options(factory="ai_process_fake:make_bad_prepare"))

        self.assertEqual(code, 1, err)
        recs = records(log)
        self.assertTrue(any(r["kind"] == "event" and r["invalidation_kind"] == "MODEL_CONFIG_INVALID" for r in recs))
        self.assertIn("fake model names mismatch", next(r for r in recs if r["kind"] == "infer_process")["failed"])


class CapturePathTest(LeakCheck, CaptureBase):
    """저장 입력 실제 A→B 경로(--capture-meta): 켬·끔 B 결과 동일, 요약 키."""

    def run_capture(self, c, *, infer):
        """capture 경로 실행 → (종료 코드, stderr)."""
        argv = ["--images", *c.images, "--detector", "ai", "--ai-config", c.ai, "--log", c.log, "--no-detect-timeout",
                "--a-to-b-capture-meta", c.write_meta(c.meta()), "--a-to-b-out", c.out]
        err = io.StringIO()
        kw = {"infer_process_options": options()} if infer else {"model_factory": fake.FakeModel}
        code = self.mod.main(argv + (["--infer-process"] if infer else []), stderr=err, **kw)

        return code, err.getvalue()

    def fresh_case(self, sub):
        """정상 동작 코드(0)로 덮어쓴 이미지 2장 Case(하위 폴더)."""
        from test_run_pipeline_capture_observation import Case
        d = Path(self.tmp.name) / sub
        d.mkdir()
        c = Case(d, 2)

        for i, p in enumerate(c.images):
            img = cv2.imread(p)
            img[0, 0] = (0, 11 * i, 0)
            cv2.imwrite(p, img)

        return c

    def test_capture_on_matches_off(self):
        """켬·끔이 같은 B 출력 입력·목표를 내고, 켬 요약에 infer_process가 있다."""
        off, on = self.fresh_case("off"), self.fresh_case("on")
        code_off, err_off = self.run_capture(off, infer=False)
        code_on, err_on = self.run_capture(on, infer=True)

        self.assertEqual((code_off, code_on), (0, 0), err_off + err_on)
        strip = lambda o: drop_run_keys(canon({k: o[k] for k in ("input", "goal")}))  # noqa: E731
        self.assertEqual([strip(o) for o in on.b_outputs()], [strip(o) for o in off.b_outputs()])
        self.assertNotIn("infer_process", off.summary())
        self.assertEqual(on.summary()["infer_process"]["results"], 2)


class DefaultTest(StoredCase):
    """기본값: 실시간 AI만 켬, 저장 입력은 끔, 끄기 인자, 시험 주입이 있으면 끔."""

    def resolved(self, argv, **inject):
        """인자 해석 뒤 (적용, 출처). 실행은 하지 않는다."""
        args = self.mod.build_parser().parse_args(argv)
        self.mod._check_detector_args(args)
        args._model_factory, args._import_module = inject.get("model_factory"), inject.get("import_module")
        self.mod._resolve_infer_process(args)

        return args.infer_process, args.infer_process_source

    def test_resolution_table(self):
        """--ros-link AI는 기본 켬, 저장 입력 AI는 기본 끔, 인자는 그대로, 주입이 있으면 기본 끔·명시 켬은 오류."""
        live = ["--ros-link", "--detector", "ai", "--ai-config", str(self.ai), "--log", "x.jsonl"]
        stored = ["--images", "a.png", "--detector", "ai", "--ai-config", str(self.ai), "--log", "x.jsonl"]
        opencv = ["--ros-link", "--detect-config", "c.json", "--log", "x.jsonl"]

        self.assertEqual(self.resolved(live), (True, "default"))
        self.assertEqual(self.resolved(live + ["--no-infer-process"]), (False, "argument"))
        self.assertEqual(self.resolved(live + ["--infer-process"]), (True, "argument"))
        self.assertEqual(self.resolved(stored), (False, "default"))
        self.assertEqual(self.resolved(stored + ["--infer-process"]), (True, "argument"))
        self.assertEqual(self.resolved(opencv), (False, "default"))
        self.assertEqual(self.resolved(live, model_factory=fake.FakeModel), (False, "default"))
        self.assertEqual(self.resolved(live + ["--infer-torch-threads", "4"]), (True, "default"))

        import contextlib
        for argv, inject in ((live + ["--infer-process"], {"model_factory": fake.FakeModel}),
                             (stored + ["--infer-torch-threads", "4"], {})):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as ctx:
                self.resolved(argv, **inject)

            self.assertEqual(ctx.exception.code, 2)

    def test_stored_off_by_argument_is_recorded(self):
        """저장 입력에서 --no-infer-process는 끔 그대로 동작하고 run_meta·기록·시작 줄에 인자로 껐다고 남는다."""
        log = self.d / "noarg.jsonl"
        code, err = self.stored(self.images([0]), log, "--no-infer-process", model_factory=fake.FakeModel)

        self.assertEqual(code, 0, err)
        recs = records(log)
        meta = next(r for r in recs if r["kind"] == "run_meta")
        self.assertEqual((meta["detector"]["infer_process"], meta["detector"]["infer_process_source"]),
                         (False, "argument"))
        rec = next(r for r in recs if r["kind"] == "infer_process")
        self.assertEqual((rec["enabled"], rec["source"]), (False, "argument"))
        self.assertIn("run_pipeline: infer_process=off source=argument\n", err)

    def test_stored_default_off_adds_nothing(self):
        """저장 입력 기본(끔)은 run_meta·기록·stderr에 추론 프로세스 표시를 더하지 않는다(기존과 같음)."""
        log = self.d / "plain.jsonl"
        code, err = self.stored(self.images([0]), log, model_factory=fake.FakeModel)

        self.assertEqual(code, 0, err)
        recs = records(log)
        self.assertNotIn("infer_process", next(r for r in recs if r["kind"] == "run_meta")["detector"])
        self.assertFalse(any(r["kind"] == "infer_process" for r in recs))
        self.assertNotIn("infer_process", err)


class LivePathTest(LeakCheck, CliBase):
    """실시간 기록 연결 경로: 1장 실행에서 대리 객체 연결·요약·정리."""

    def test_live_one_frame_with_infer_process(self):
        """--ros-link --live-observation-out(추론 프로세스 인자 없음) 1장: 기본 켬으로 자식이 뜨고 종료 0, 표시는 source=default."""
        out = self.out()
        log = str(Path(self.tmp.name) / "live.jsonl")
        sc = Scenario(1)
        sc.link = lambda: self.controller.current
        made = []
        from test_run_pipeline_live_observation import CONFIG
        argv = ["--ros-link", "--detector", "ai", "--ai-config", self.case.ai, "--log", log,
                "--live-observation-out", out, "--live-observation-config", str(CONFIG), "--max-frames", "1"]
        err = io.StringIO()
        code = self.mod.main(argv, stderr=err, ros_available=lambda: True, live_source_factory=sc.factory_maker(made),
                             live_tf_factory=lambda c: FakeTf(), live_run_controller=self.controller,
                             infer_process_options=options())

        self.assertEqual(code, 0, err.getvalue())
        meta = next(r for r in records(log) if r["kind"] == "run_meta")
        self.assertEqual((meta["detector"]["infer_process"], meta["detector"]["infer_process_source"]),
                         (True, "default"))
        self.assertIn("run_pipeline: infer_process=on source=default (default for --ros-link --detector ai;",
                      err.getvalue())
        s = self.summary(out)
        self.assertEqual((s["infer_process"]["enabled"], s["infer_process"]["source"], s["infer_process"]["results"],
                          s["infer_process"]["failed"]), (True, "default", 1, None))
        self.assertEqual(s["infer_process"]["exit"], {"returncode": 0, "forced": False})


if __name__ == "__main__":
    unittest.main()
