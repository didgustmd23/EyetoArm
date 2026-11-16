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
