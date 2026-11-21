"""기존 S1 결과 등록(`register_legacy`) 시험. 임시 저장소에 가짜 legacy 산출물을 만든다(실제 outputs 미사용).

legacy는 applied·progress를 만들지 않고 산출물을 바꾸지 않으며, 설정이 다른 결과를 같은 비교 조건으로 두지 않는다.
"""
import hashlib
import json
import os
import unittest

from eyetoarm.training.eval_runner import register_legacy
from eyetoarm.training.layout import read_lock
from eyetoarm.training.legacy import parse_log, parse_summary
from test_training_runner import E, FAMILIES, Fixture, _read_json, _write

SRC = "outputs/s1_candidates"
ANSI_LOG = ("\x1b[34m\x1b[1moptimizer:\x1b[0m AdamW(lr=0.001667, momentum=0.9) with parameter groups 1\n"
            "{e} epochs completed in 0.173 hours.\n")


def all_files(root):
    """root 아래 파일 → sha256."""
    out = {}
    for d, _, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            with open(p, "rb") as fh:
                out[os.path.relpath(p, root)] = hashlib.sha256(fh.read()).hexdigest()
    return out


class LegacyFixture(Fixture):
    """가짜 legacy 산출물(run_all 형식)."""

    def make_legacy(self, *, epochs=E, args_over=None, summary_rc=0, missing=(), stop_reason="none",
                    stripped=True):
        """후보 3개의 legacy 폴더·로그·요약·실행기 파일을 만든다."""
        src = os.path.join(self.root, SRC)
        os.makedirs(src)
        _write(os.path.join(src, "train_candidate.py"), "# legacy runner\n")
        _write(os.path.join(src, "run_all.sh"), "#!/bin/bash\n")
        lines = []
        self.stripped = stripped
        for exp_id, fam in FAMILIES:
            name = f"{exp_id}_{fam}"
            if exp_id in missing:
                continue
            exp = os.path.join(src, name)
            os.makedirs(os.path.join(exp, "weights"))
            args = {"model": os.path.join(self.root, "outputs", "weights", f"{fam}.pt"),
                    "data": os.path.join(self.root, "data", "ds", "dataset.yaml"), "epochs": E, "batch": 16,
                    "imgsz": 640, "cache": False, "device": "0", "workers": 2, "seed": 0, "deterministic": True,
                    "plots": False, "project": src, "name": name, "optimizer": "auto"}
            args.update(args_over or {})
            _write(os.path.join(exp, "args.yaml"), "".join(f"{k}: {json.dumps(v)}\n" for k, v in args.items()))
            _write(os.path.join(exp, "results.csv"), "epoch,loss\n" + "".join(f"{e},0.1\n" for e in range(1, epochs + 1)))
            _write(os.path.join(exp, "run_info.json"), json.dumps({"wall_s": 12.5, "torch": "t", "ultralytics": "u"}))
            for n in ("best.pt", "last.pt"):
                _write(os.path.join(exp, "weights", n), f"{name} {n}".encode(), "wb")
            _write(os.path.join(src, f"{name}.log"), ANSI_LOG.format(e=epochs))
            _write(os.path.join(src, f"{name}.watch.txt"), "start\n")
            lines.append(f"{name} started pid=1 pgid=1 log=x 2026")
            rc = summary_rc if exp_id == "S1-02" else 0
            lines.append(f"{name} end rc={rc} stop_reason={stop_reason} remaining_in_pgid=0 2026")
        _write(os.path.join(src, "run_all_summary.txt"), "\n".join(lines) + "\n")
        return src

    def legacy_inspect(self, path):
        """legacy 체크포인트 메타(train_args.data는 절대 경로, strip 여부 선택)."""
        if not os.path.exists(path):
            return {"exists": False}
        with open(path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        args = {"data": os.path.join(self.root, "data", "ds", "dataset.yaml"), "imgsz": 640, "batch": 16, "epochs": E,
                "seed": 0, "deterministic": True, "workers": 2, "device": "0"}
        return {"exists": True, "sha256": sha, "loadable": True, "epoch": -1 if self.stripped else E - 1,
                "optimizer_none": self.stripped, "train_args": args}

    def register(self, run_set="s1_legacy", cfg=None):
        """register_legacy 호출."""
        return register_legacy(cfg or self.config(), repo_root=self.root, source_dir=SRC, run_set=run_set,
                               inspect_fn=self.legacy_inspect, argv=["register-legacy", run_set])

    def reg(self, res, exp_id, fam):
        """등록 기록."""
        return _read_json(os.path.join(res.run_set_dir, f"{exp_id}_{fam}", "registration.json"))


class RegisterLegacyTest(LegacyFixture):
    """등록."""

    def test_completed_registration_keeps_sources_and_fabricates_nothing(self):
        """정상: 세 후보 completed·comparable, 한계 4개, 원본 sha256 무변, applied·progress·job_state 없음,
        lock 해제·SHA256SUMS, rc 0."""
        src = self.make_legacy()
        before = all_files(src)
        res = self.register()

        self.assertEqual(res.rc, 0, res.errors)
        self.assertEqual(all_files(src), before)
        for exp_id, fam in FAMILIES:
            r = self.reg(res, exp_id, fam)
            self.assertEqual((r["status"], r["comparable"], r["runner"]), ("completed", True, "legacy_untracked"), r)
            self.assertEqual(r["evidence"]["optimizer_from_log"], {"optimizer": "AdamW", "lr": 0.001667,
                                                                   "source": "log text"})
            self.assertEqual(len(r["limitations"]), 4)
            self.assertEqual(r["applied_json"], "absent (not created)")
            self.assertEqual(r["files"]["best"]["sha256"], before[f"{exp_id}_{fam}/weights/best.pt"])
        made = set(all_files(res.run_set_dir)) | set(all_files(src))
        self.assertFalse([p for p in made if os.path.basename(p) in ("applied.json", "progress.json",
                                                                       "job_state.json")])
        rs = _read_json(os.path.join(res.run_set_dir, "run_set.json"))
        self.assertTrue(rs["sources_unchanged"])
        self.assertIsNone(read_lock(res.run_set_dir))
        self.assertTrue(os.path.exists(os.path.join(res.run_set_dir, "SHA256SUMS")))

    def test_config_difference_is_not_comparable(self):
        """args.yaml batch 8(공통 16)·다른 초기 가중치 파일은 등록하되 comparable false와 차이 기록(같은 조건으로 표시하지 않음)."""
        other = os.path.join(self.root, "outputs", "weights", "other.pt")
        _write(other, b"other", "wb")
        self.make_legacy(args_over={"batch": 8, "model": other})
        res = self.register()

        r = self.reg(res, "S1-01", "yolov8n-seg")
        self.assertEqual((r["status"], r["comparable"]), ("completed", False))
        self.assertTrue(any("args.batch 8" in d for d in r["config_diffs"]), r["config_diffs"])
        self.assertFalse(r["initial_weights"]["matches_official"])

    def test_incomplete_failed_missing_registered_as_is(self):
        """에포크 부족·S1-02 rc 1·공백 stop_reason → incomplete, S1-03 폴더 없음 → missing. rc 1."""
        self.make_legacy(epochs=E - 1, summary_rc=1, missing=("S1-03",), stop_reason="mem_avail_below_2GiB avail=1")
        res = self.register()

        self.assertEqual(res.rc, 1)
        r1, r2 = self.reg(res, "S1-01", "yolov8n-seg"), self.reg(res, "S1-02", "yolo11n-seg")
        self.assertEqual(r1["status"], "incomplete")
        self.assertTrue(any("epochs" in x for x in r1["reasons"]))
        self.assertEqual((r2["evidence"]["summary_rc"], r2["evidence"]["summary_stop_reason"]),
                         (1, "mem_avail_below_2GiB avail=1"))
        self.assertEqual(self.reg(res, "S1-03", "yolo26n-seg")["status"], "missing")

    def test_unstripped_checkpoint_is_incomplete(self):
        """strip되지 않은 best·last(epoch E-1, optimizer 있음)는 완료본이 아니므로 incomplete."""
        self.make_legacy()
        self.stripped = False
        res = self.register()

        r = self.reg(res, "S1-01", "yolov8n-seg")
        self.assertEqual(r["status"], "incomplete")
        self.assertTrue(any("stripped" in x for x in r["reasons"]))

    def test_second_registration_does_not_overwrite(self):
        """같은 run_set에 다시 등록하면 already_registered이고 registration.json 바이트는 그대로, rc 1."""
        self.make_legacy()
        res = self.register()
        path = os.path.join(res.run_set_dir, "S1-01_yolov8n-seg", "registration.json")
        with open(path, "rb") as f:
            first = f.read()

        res2 = self.register()
        self.assertEqual(res2.rc, 1)
        self.assertEqual({c["status"] for c in res2.candidates}, {"already_registered"})
        with open(path, "rb") as f:
            self.assertEqual(f.read(), first)

    def test_smoke_prefix_rejected_for_comparison(self):
        """비교(kind comparison) 설정에 smoke_ run_set 이름은 시작 거부(rc 2), 아무 폴더도 만들지 않음."""
        self.make_legacy()
        res = self.register(run_set="smoke_legacy")
        self.assertEqual(res.rc, 2)
        self.assertFalse(os.path.exists(res.run_set_dir))


class ParseTest(unittest.TestCase):
    """로그·요약 해석."""

    def test_parse_log_and_summary(self):
        """ANSI 색 코드가 있는 optimizer 줄·마지막 완료 문구·마지막 종료 줄을 읽고, 없으면 None."""
        log = parse_log(ANSI_LOG.format(e=50) + ANSI_LOG.format(e=3))
        self.assertEqual((log["log_completed_epochs"], log["optimizer_from_log"]["optimizer"]), (3, "AdamW"))
        self.assertEqual(parse_log("nothing")["optimizer_from_log"], None)
        s = parse_summary("A end rc=0 stop_reason=none remaining_in_pgid=0 t\nA end rc=2 stop_reason=x y "
                          "remaining_in_pgid=1 t\n", "A")
        self.assertEqual((s["summary_rc"], s["summary_remaining"], s["summary_end_lines"]), (2, 1, 2))
        self.assertIsNone(parse_summary("", "A")["summary_rc"])


if __name__ == "__main__":
    unittest.main()
