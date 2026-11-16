"""학습기 I2 — CLI(`scripts/train_candidates.py`) 시험. 실제 학습·torch 없이 하위 명령·자식 argv·무import를 확인한다.

근거: 학습기 설계 r2 §1(진입점·하위 명령, 부모는 torch·ultralytics import 없음), 구현 지시(check/status/--help는 GPU·설치·가중치
자동 다운로드를 강제하지 않음), §3(prepare-weights는 명시할 때만 내려받기).
"""
import importlib.util
import io
import json
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stdout

from test_training_runner import FAMILIES, Fixture, _write

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT = os.path.join(ROOT, "scripts", "train_candidates.py")


def load_cli():
    """CLI 모듈."""
    spec = importlib.util.spec_from_file_location("train_candidates_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_configs(fx):
    """임시 저장소에 공통·후보 설정 파일을 쓰고 CLI 기본 인자를 돌려준다."""
    common, cands = fx.config_docs()
    os.makedirs(os.path.join(fx.root, "config", "training"))
    _write(os.path.join(fx.root, "config", "training", "c.json"), json.dumps(common))
    _write(os.path.join(fx.root, "config", "training", "k.json"), json.dumps(cands))
    return ["--repo-root", fx.root, "--common", "config/training/c.json", "--candidates", "config/training/k.json"]


class CliTest(Fixture):
    """하위 명령."""

    def setUp(self):
        """임시 저장소에 설정 파일을 쓴다."""
        super().setUp()
        self.base = write_configs(self)

    def call(self, *argv):
        """CLI main을 부르고 (종료 코드, stdout JSON)."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = load_cli().main(self.base + list(argv))
        return rc, json.loads(buf.getvalue())

    def test_help_check_status_import_no_torch(self):
        """--help·check·status는 torch·ultralytics를 import하지 않고, check는 가중치·fingerprint 정상이면 0이다."""
        code = ("import sys, io, contextlib, importlib.util\n"
                f"spec = importlib.util.spec_from_file_location('tc', {SCRIPT!r})\n"
                "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
                "buf = io.StringIO()\n"
                "with contextlib.redirect_stdout(buf):\n"
                f"    rc1 = m.main({self.base + ['check']!r})\n"
                f"    rc2 = m.main({self.base + ['status', 's1_x']!r})\n"
                "bad = [k for k in sys.modules if k.split('.')[0] in ('torch', 'ultralytics')]\n"
                "print(rc1, rc2, ','.join(bad) or 'none')")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                             env={"PYTHONPATH": os.path.join(ROOT, "src"), "PATH": ""}, cwd=ROOT)
        self.assertEqual(out.stdout.strip().splitlines()[-1], "0 0 none", out.stderr[-1500:])

        h = subprocess.run([sys.executable, SCRIPT, "--help"], capture_output=True, text=True, timeout=30)
        self.assertEqual(h.returncode, 0)
        self.assertIn("prepare-weights", h.stdout)

    def test_check_reports_weight_and_data_problems(self):
        """가중치 불일치는 check 종료 1, 데이터가 바뀌면 종료 2(fingerprint 차이 보고)."""
        _write(os.path.join(self.root, "outputs", "weights", "yolo26n-seg.pt"), b"x", "wb")
        rc, out = self.call("check")
        self.assertEqual((rc, out["weights"]["S1-03"]), (1, "mismatch"))

        _write(os.path.join(self.root, "data", "ds", "labels", "test", "a.txt"), "1 0.5 0.5 0.6 0.6 0.7 0.5\n")
        rc, out = self.call("check")
        self.assertEqual(rc, 2)
        self.assertTrue(out["fingerprint_diffs"])

    def test_prepare_weights_without_allow_does_not_download(self):
        """없는 가중치는 --allow-download 없이 종료 1·오류(내려받기 시도 없음), 있는 가중치는 existing."""
        os.remove(os.path.join(self.root, "outputs", "weights", "yolo11n-seg.pt"))
        rc, out = self.call("prepare-weights")

        self.assertEqual(rc, 1)
        self.assertIn("allow_download", out["S1-02"]["error"])
        self.assertEqual((out["S1-01"]["method"], out["S1-03"]["method"]), ("existing", "existing"))
        self.assertFalse(os.path.exists(os.path.join(self.root, "outputs", "weights", "yolo11n-seg.pt.part")))

    def test_child_argv_and_recover_cli(self):
        """학습 자식 argv는 같은 스크립트의 _child-train과 실험 폴더(재개 시 --resume-from)를 담고, 빈 run_set의 recover는 lock 해제 보고."""
        cli = load_cli()
        args = cli.build_parser().parse_args(self.base + ["train", "s1_x"])
        cand = self.config().candidates[0]

        a = cli.child_argv(args, cand, "/tmp/exp", None)
        b = cli.child_argv(args, cand, "/tmp/exp", "/tmp/exp/weights/last.pt")
        self.assertEqual(a[1:2] + a[a.index("_child-train"):], [SCRIPT, "_child-train", "--exp-id", "S1-01",
                                                                "--exp-dir", "/tmp/exp"])
        self.assertEqual(b[-2:], ["--resume-from", "/tmp/exp/weights/last.pt"])

        rc, rep = self.call("recover", "s1_empty")
        self.assertEqual((rc, rep["lock_released"]), (0, True))
        self.assertEqual(len(FAMILIES), 3)


class CliErrorTest(Fixture):
    """설정·사전 입력 오류에서 traceback 대신 JSON 오류와 종료 2(F2)."""

    def setUp(self):
        """임시 저장소(설정 파일은 일부러 공통 설정을 빼고 후보만)."""
        super().setUp()
        _, cands = self.config_docs()
        os.makedirs(os.path.join(self.root, "config", "training"))
        _write(os.path.join(self.root, "config", "training", "k.json"), json.dumps(cands))

    def test_bad_config_paths_exit_2_json(self):
        """없는 공통 설정으로 train·resume·status·recover·prepare-weights·check는 모두 종료 2와 error JSON."""
        for argv in (["train", "s1_x"], ["resume", "s1_x", "S1-01"], ["status", "s1_x"], ["recover", "s1_x"],
                     ["prepare-weights"], ["check"]):
            with self.subTest(argv=argv):
                buf = io.StringIO()
                bad = ["--repo-root", self.root, "--common", "config/training/missing.json", "--candidates",
                       "config/training/k.json"]
                with redirect_stdout(buf):
                    rc = load_cli().main(bad + argv)
                self.assertEqual(rc, 2)
                self.assertIn("error", json.loads(buf.getvalue()))


class CliEvalTest(Fixture):
    """I3 하위 명령(eval·register-legacy)과 평가 자식 argv(CliTest를 상속하지 않아 기존 시험이 중복 실행되지 않음)."""

    def setUp(self):
        """임시 저장소에 설정 파일을 쓴다."""
        super().setUp()
        self.base = write_configs(self)

    def test_eval_commands_and_child_argv(self):
        """--help에 eval·register-legacy, 시간 자식은 크기 하나(--imgsz), 정확도 자식은 --val-dir(시도 폴더 아래)를 받는다."""
        h = subprocess.run([sys.executable, SCRIPT, "--help"], capture_output=True, text=True, timeout=30)
        self.assertIn("register-legacy", h.stdout)
        self.assertIn("eval", h.stdout)

        cli = load_cli()
        args = cli.build_parser().parse_args(self.base + ["eval", "s1_x"])
        t = cli.eval_child_argv(args, "timing", "/w/best.pt", 1280, "/a/timing1280.json", "/a")
        a = cli.eval_child_argv(args, "accuracy", "/w/best.pt", None, "/a/accuracy.json", "/a")
        self.assertEqual(t[t.index("_child-eval-timing"):], ["_child-eval-timing", "--weights", "/w/best.pt",
                                                             "--imgsz", "1280", "--out", "/a/timing1280.json"])
        self.assertEqual(a[a.index("_child-eval-accuracy"):], ["_child-eval-accuracy", "--weights", "/w/best.pt",
                                                               "--out", "/a/accuracy.json", "--val-dir", "/a/val"])
        self.assertIn("--eval-config", t)

    def test_eval_errors_exit_2_without_torch(self):
        """없는 평가 설정·없는 run_set은 종료 2와 JSON 오류이며, 부모는 torch·ultralytics를 import하지 않는다."""
        code = ("import sys, io, contextlib, importlib.util, json\n"
                f"spec = importlib.util.spec_from_file_location('tc', {SCRIPT!r})\n"
                "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
                "outs = []\n"
                f"for argv in ({self.base + ['--eval-config', 'config/training/missing.json', 'eval', 's1_x']!r},"
                f" {self.base + ['--eval-config', 'config/training/k.json', 'eval', 's1_x']!r}):\n"
                "    buf = io.StringIO()\n"
                "    with contextlib.redirect_stdout(buf):\n"
                "        rc = m.main(argv)\n"
                "    outs.append(rc)\n"
                "bad = [k for k in sys.modules if k.split('.')[0] in ('torch', 'ultralytics')]\n"
                "print(outs, ','.join(bad) or 'none')")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                           env={"PYTHONPATH": os.path.join(ROOT, "src"), "PATH": ""}, cwd=ROOT)
        self.assertEqual(r.stdout.strip().splitlines()[-1], "[2, 2] none", r.stderr[-1500:])


if __name__ == "__main__":
    unittest.main()


class SingleModeCliTest(Fixture):
    """selection_mode single 설정의 check(#139)."""

    def test_check_single_mode_ok(self):
        """single 후보(S1-01 하나) 설정으로 check는 종료 0이고 weights에 S1-01만 나온다."""
        common, _ = self.config_docs()
        os.makedirs(os.path.join(self.root, "config", "training"))
        _write(os.path.join(self.root, "config", "training", "c.json"), json.dumps(common))
        _write(os.path.join(self.root, "config", "training", "k.json"),
               json.dumps({"format_version": 1, "selection_mode": "single", "candidates": [self.cands[0]]}))
        base = ["--repo-root", self.root, "--common", "config/training/c.json", "--candidates", "config/training/k.json"]

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = load_cli().main(base + ["check"])
        out = json.loads(buf.getvalue())

        self.assertEqual(rc, 0, out)
        self.assertEqual(out["weights"], {"S1-01": "ok"})
        self.assertEqual(out["fingerprint_diffs"], [])


class FakeYOLO:
    """가짜 ultralytics.YOLO: train 인자를 기록만 한다(GPU·모델 로드 없음)."""

    calls = []

    def __init__(self, path):
        """생성 경로를 기록한다."""
        self.path = path

    def add_callback(self, name, fn):
        """콜백 등록(무시)."""

    def train(self, **kw):
        """train 인자 기록."""
        FakeYOLO.calls.append(kw)


class HsvVChildTest(Fixture):
    """학습 자식이 hsv_v를 model.train에 넘기는지(#141), 가짜 torch·ultralytics로 확인한다."""

    def run_child(self, hsv):
        """가짜 모듈로 _child-train을 부르고 train 인자를 돌려준다."""
        import types
        from unittest import mock
        common, _ = self.config_docs()
        if hsv is not None:
            common["train"] = {**common["train"], "hsv_v": hsv}
        os.makedirs(os.path.join(self.root, "config", "training"), exist_ok=True)
        _write(os.path.join(self.root, "config", "training", "c.json"), json.dumps(common))
        _write(os.path.join(self.root, "config", "training", "k.json"),
               json.dumps({"format_version": 1, "selection_mode": "single", "candidates": [self.cands[0]]}))
        fake_ul = types.ModuleType("ultralytics")
        fake_ul.YOLO, fake_ul.__version__ = FakeYOLO, "fake"
        FakeYOLO.calls = []
        exp = os.path.join(self.root, "outputs", "training", "x", "S1-01_yolov8n-seg")
        os.makedirs(exp, exist_ok=True)
        argv = ["--repo-root", self.root, "--common", "config/training/c.json", "--candidates",
                "config/training/k.json", "_child-train", "--exp-id", "S1-01", "--exp-dir", exp]
        with mock.patch.dict(sys.modules, {"torch": types.ModuleType("torch"), "ultralytics": fake_ul}):
            self.assertEqual(load_cli().main(argv), 0)
        return FakeYOLO.calls[0]

    def test_hsv_v_passed_only_when_configured(self):
        """hsv_v 0.6 설정이면 model.train(hsv_v=0.6, 나머지 공통 값 그대로), 설정에 없으면 hsv_v 인자 없음."""
        with_hsv = self.run_child(0.6)
        without = self.run_child(None)

        self.assertEqual(with_hsv["hsv_v"], 0.6)
        self.assertNotIn("hsv_v", without)
        self.assertEqual({k: v for k, v in with_hsv.items() if k != "hsv_v"}, without)

    def test_inspect_reports_hsv_v(self):
        """_child-inspect는 체크포인트 train_args의 hsv_v를 메타에 넣는다(재개·완료 대조용)."""
        import types
        from unittest import mock
        fake_torch = types.ModuleType("torch")
        fake_torch.load = lambda *a, **k: {"epoch": 3, "optimizer": object(), "train_args": {"hsv_v": 0.6, "imgsz": 640}}
        buf = io.StringIO()
        with mock.patch.dict(sys.modules, {"torch": fake_torch}), redirect_stdout(buf):
            load_cli().main(["_child-inspect", "x.pt"])

        self.assertEqual(json.loads(buf.getvalue())["train_args"]["hsv_v"], 0.6)


class HsvVBadConfigCliTest(Fixture):
    """잘못된 hsv_v 설정의 CLI check(#141 r1 F1): traceback 대신 오류 JSON과 종료 2."""

    def test_huge_int_hsv_v_exit_2_json(self):
        """hsv_v가 ±10**400(정상 JSON 정수)이면 check는 종료 2와 ConfigError 오류 JSON을 낸다."""
        for v in (10 ** 400, -(10 ** 400)):
            common, cands = self.config_docs()
            common["train"] = {**common["train"], "hsv_v": v}
            os.makedirs(os.path.join(self.root, "config", "training"), exist_ok=True)
            _write(os.path.join(self.root, "config", "training", "c.json"), json.dumps(common))
            _write(os.path.join(self.root, "config", "training", "k.json"), json.dumps(cands))
            base = ["--repo-root", self.root, "--common", "config/training/c.json",
                    "--candidates", "config/training/k.json"]
            buf = io.StringIO()
            with self.subTest(sign=v > 0), redirect_stdout(buf):
                rc = load_cli().main(base + ["check"])
            out = json.loads(buf.getvalue())
            self.assertEqual(rc, 2)
            self.assertFalse(out["ok"])
            self.assertIn("ConfigError", out["error"])
