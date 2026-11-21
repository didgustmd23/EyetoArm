"""학습기 평가 실행(`run_eval`) 시험: 가짜 시간·정확도 자식과 임시 저장소로 순서·기록·실패·소유권을 본다."""
import json
import os
import sys
import unittest

from eyetoarm.training.evaluation import SUMMARY_COLUMNS, eval_config_from_dict
from eyetoarm.training.eval_runner import run_eval
from eyetoarm.training.layout import read_lock
from eyetoarm.training.runner import StopFlag
from test_training_legacy import LegacyFixture
from test_training_runner import FAMILIES, SLEEP, _lines, _read_json, _write

SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
PREFIX = f"import sys; sys.path.insert(0, {SRC_PATH!r})\n"   # 자식 cwd는 임시 저장소라 상대 PYTHONPATH가 통하지 않음

TIMING_CHILD = PREFIX + r"""
import json, os, sys, time
from eyetoarm.training.evaluation import run_timing
from eyetoarm.training.layout import write_json_atomic
out, imgsz, w, m, mode = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
if mode == "fail":
    sys.exit(3)
if mode == "slow":
    time.sleep(0.4)
if mode == "partial":
    open(out, "w").write('{"imgsz": %d, "phase_order": ' % imgsz)
    sys.exit(0)
hist, box = [], {}
class M:
    def __init__(self):
        hist.append("construct"); self.calls = 0
        if mode == "warm":
            self.predict("init")
    def predict(self, img):
        self.calls += 1; hist.append("predict")
def prep():
    box["m"] = M(); return box["m"]
r = run_timing(prepare=prep, images=[1, 2, 3], predict=lambda mm, i: mm.predict(i), sync=lambda: None,
               peak_reset=lambda: None, peak_read=lambda: 100, clock_ns=time.perf_counter_ns, warmup_n=w,
               measure_n=m, calls_seen=lambda: box["m"].calls)
r.update(imgsz=imgsz, pid=os.getpid(), history=hist)
write_json_atomic(out, r)
"""

ACC_CHILD = PREFIX + r"""
import json, os, sys
from eyetoarm.training.evaluation import run_accuracy
from eyetoarm.training.layout import write_json_atomic
out, plan, mode = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]
def entry(split, imgsz, per_gt):
    return {"split": split, "images": len(per_gt), "imgsz": imgsz, "conf": 0.25, "iou": 0.5, "scale": 1.0,
            "edges_px": [10, 12, 15, 25], "by_size": {"bolt": {"<10": {"n": 1, "tp": 1, "recall": 1.0}},
                                                      "nut": {"<10": {"n": 1, "tp": 0, "recall": 0.0}}},
            "total": {"bolt": {"n": 2, "tp": 2}, "nut": {"n": 1, "tp": 0}},
            "predictions": {"bolt": {"n": 3, "tp": 2, "fp": 1, "precision": 0.66},
                            "nut": {"n": 0, "tp": 0, "fp": 0, "precision": None}}, "per_gt": per_gt}
GZ = [{"image": "g0.png", "matched": [True, False]}, {"image": "g1.png", "matched": [True]}]
def ev(model, data_dir, split, imgsz):
    return entry(split, imgsz, GZ if "gz" in data_dir else [{"image": "a.png", "matched": [True]}])
def val(model, data_dir, split, imgsz, key):
    return {"map50_box": 0.5, "map50_mask": 0.4}
r = run_accuracy(model="m", plan=[tuple(p) for p in plan], evaluate_fn=ev, val_fn=val)
if mode == "nan":
    r["gz640"]["map50_box"] = float("nan")
if mode == "badpred":
    r["gz640"]["predictions"] = {}
if mode == "badsize":
    r["test640"]["by_size"] = {"bolt": {"<10": {"n": "x"}}}
write_json_atomic(out, r)
"""


class EvalFixture(LegacyFixture):
    """학습 완료 run_set + Gazebo 가짜 manifest + 평가 설정."""

    def setUp(self):
        """평가용 가짜 데이터."""
        super().setUp()
        gz = os.path.join(self.root, "data", "gz")
        os.makedirs(gz)
        rows = [{"sample_id": "g0", "parts": [{"cls": "BOLT_A", "labeled": True}, {"cls": "NUT", "labeled": True}]},
                {"sample_id": "g1", "parts": [{"cls": "BOLT_B", "labeled": True}]}]
        _write(os.path.join(gz, "manifest.jsonl"), "".join(json.dumps(r) + "\n" for r in rows))
        for kind in ("images", "labels"):
            os.makedirs(os.path.join(gz, kind, "test"))
        for stem, n in (("g0", 2), ("g1", 1)):
            _write(os.path.join(gz, "images", "test", f"{stem}.png"), f"img {stem}".encode(), "wb")
            _write(os.path.join(gz, "labels", "test", f"{stem}.txt"), "0 0.1 0.1 0.2 0.2 0.3 0.1\n" * n)
        _write(os.path.join(gz, "dataset.yaml"), "test: images/test\nnames:\n  0: bolt\n  1: nut\n")
        self.ecfg = eval_config_from_dict({
            "format_version": 1, "conf": 0.25, "iou": 0.5, "device": "0", "edges_px": [10, 12, 15, 25],
            "public": {"data_dir": "data/ds", "split": "test", "imgsz": [640]},
            "gazebo": {"data_dir": "data/gz", "split": "test", "imgsz": [640, 1280],
                       "manifest": "data/gz/manifest.jsonl"},
            "timing": {"imgsz": [640, 1280], "warmup_n": 2, "measure_n": 3}, "val": {"batch": 8, "workers": 0}})
        self.jobs = []

    def eval_factory(self, modes=None):
        """가짜 평가 자식 argv. modes[(exp 폴더 이름 접두, 작업)] = 'ok'|'fail'|'partial'|'warm'|'sleep'|'nan'."""
        modes = modes or {}

        def make(kind, weights, imgsz, out_path, attempt_dir):
            """argv."""
            exp = os.path.basename(os.path.dirname(os.path.dirname(attempt_dir)))
            job = f"timing{imgsz}" if kind == "timing" else "accuracy"
            mode = next((v for (e, j), v in modes.items() if exp.startswith(e) and j == job), "ok")
            self.jobs.append((exp[:5], job, weights))
            if mode == "sleep":
                return [sys.executable, "-c", SLEEP]
            if kind == "timing":
                return [sys.executable, "-c", TIMING_CHILD, out_path, str(imgsz), "2", "3", mode]
            return [sys.executable, "-c", ACC_CHILD, out_path, json.dumps(self.ecfg.accuracy_plan()), mode]
        return make

    def evaluate(self, run_set="s1_t1", modes=None, stop=None, cfg=None, **kw):
        """run_eval 호출(부하 관측은 가짜 명령)."""
        res = run_eval(cfg or self.config(), self.ecfg, repo_root=self.root, run_set=run_set, argv=["eval", run_set],
                       child_factory=self.eval_factory(modes), stop=stop or StopFlag(install=False), poll_s=0.05,
                       gpu_apps_cmd=[sys.executable, "-c", "print('123, 456')"], other_counter=lambda: 0, **kw)
        self.tokens = getattr(self, "tokens", []) + [
            (_read_json(os.path.join(res.run_set_dir, d, "job_state.json")) or {}).get("job_token")
            for d in os.listdir(res.run_set_dir) if os.path.exists(os.path.join(res.run_set_dir, d, "job_state.json"))]
        return res

    def trained(self):
        """가짜 학습으로 완료 run_set을 만든다."""
        res = self.run_batch({})
        self.assertEqual(res.rc, 0, res.errors)
        return res


class EvalSuccessTest(EvalFixture):
    """정상 평가."""

    def test_order_records_summary(self):
        """후보마다 timing640 → timing1280 → accuracy 자식, 평가 기록·요약 3행·이전 판 보존, rc 0."""
        tr = self.trained()
        res = self.evaluate()

        self.assertEqual(res.rc, 0, res.errors)
        self.assertEqual([(e, j) for e, j, _ in self.jobs],
                         [(e, j) for e, _ in FAMILIES for j in ("timing640", "timing1280", "accuracy")])
        for exp_id, fam in FAMILIES:
            d = self.exp(tr, exp_id, fam)
            rec = _read_json(os.path.join(d, "eval", "eval.json"))
            per = _read_json(os.path.join(d, "eval", "eval_per_gt.json"))
            self.assertEqual(rec["accuracy"]["gz1280"]["fine"]["BOLT_A"], {"n": 1, "tp": 1, "recall": 1.0})
            self.assertEqual(rec["accuracy"]["gz640"]["fine"]["NUT"]["tp"], 0)
            self.assertEqual(len(per["per_gt"]["gz640"]), 2)
            self.assertEqual(rec["weights"]["sha256"], _read_json(os.path.join(d, "run_manifest.json"))["best_sha256"])
            self.assertEqual([x["job"] for x in rec["loads"]], ["timing640", "timing1280", "accuracy"])
            self.assertEqual(rec["loads"][0]["start"]["gpu_apps"], [["123", "456"]])
            pids = {_read_json(os.path.join(d, "eval", "attempt_1", f"timing{s}.json"))["pid"] for s in (640, 1280)}
            self.assertEqual(len(pids), 2)                                       # 크기마다 새 자식
            hist = _read_json(os.path.join(d, "eval", "attempt_1", "timing640.json"))["history"]
            self.assertEqual(hist[:2], ["construct", "predict"])  # 생성 뒤 첫 predict가 cold
            js = _read_json(os.path.join(d, "job_state.json"))
            self.assertEqual((js["state"], js["job"]), ("exited_cleaned", "eval:accuracy"))
        summ = _read_json(os.path.join(tr.run_set_dir, "summary.json"))["rows"]
        self.assertEqual([r["번호"] for r in summ], ["S1-01", "S1-02", "S1-03"])
        self.assertEqual(tuple(summ[0]), SUMMARY_COLUMNS)
        self.assertTrue(any(n.startswith("SHA256SUMS.") for n in os.listdir(tr.run_set_dir)))
        self.assertIsNone(read_lock(tr.run_set_dir))

    def test_no_overwrite_on_second_eval(self):
        """두 번째 eval은 already_evaluated로 자식 0개, eval.json 바이트 무변."""
        tr = self.trained()
        self.evaluate()
        path = os.path.join(self.exp(tr, "S1-01", "yolov8n-seg"), "eval", "eval.json")
        with open(path, "rb") as f:
            first = f.read()
        self.jobs.clear()

        res = self.evaluate()
        self.assertEqual(([c["status"] for c in res.candidates], self.jobs), (["already_evaluated"] * 3, []))
        with open(path, "rb") as f:
            self.assertEqual(f.read(), first)


class EvalFailureTest(EvalFixture):
    """무효·부분 기록, 실패, 중단, 소유권, 정리 실패."""

    def test_invalid_records_fail_without_eval_json(self):
        """시간 기록 일부·준비 중 predict·mAP NaN → 각각 failed, eval.json 없음, 요약 0행, rc 1."""
        tr = self.trained()
        res = self.evaluate(modes={("S1-01", "timing640"): "partial", ("S1-02", "timing1280"): "warm",
                                   ("S1-03", "accuracy"): "nan"})

        self.assertEqual(res.rc, 1)
        self.assertEqual([c["status"] for c in res.candidates], ["failed"] * 3)
        for exp_id, fam in FAMILIES:
            self.assertFalse(os.path.exists(os.path.join(self.exp(tr, exp_id, fam), "eval", "eval.json")))
        st = _read_json(os.path.join(self.exp(tr, "S1-02", "yolo11n-seg"), "eval", "attempt_1", "status.json"))
        self.assertTrue(any("predict called before cold call" in p for p in st["problems"]), st)
        self.assertEqual(_read_json(os.path.join(tr.run_set_dir, "summary.json"))["rows"], [])
        # 실패 뒤 그 후보의 다음 자식 없음
        self.assertEqual([j for e, j, _ in self.jobs if e == "S1-01"], ["timing640"])

    def test_child_error_and_deadline(self):
        """자식 rc 3 → failed, 기한 초과(잠자는 자식) → 정리 확인·failed, 다음 후보는 평가된다."""
        tr = self.trained()
        res = self.evaluate(modes={("S1-01", "timing640"): "fail", ("S1-02", "accuracy"): "sleep"},
                            cfg=self.config(eval_deadline_s=0.5))

        self.assertEqual([c["status"] for c in res.candidates], ["failed", "failed", "evaluated"])
        att = _lines(os.path.join(self.exp(tr, "S1-02", "yolo11n-seg"), "eval", "attempts.jsonl"))
        end = [a for a in att if a["event"] == "job_end" and a["job"] == "accuracy"][0]
        self.assertEqual((end["stop_reason"], end["cleanup_confirmed"]), ("deadline", True))
        self.assertIsNone(read_lock(tr.run_set_dir))

    def test_stop_flag_stops_and_rest_not_started(self):
        """정지 플래그(잠자는 S1-01 정확도 자식 중 설정) → stopped, 나머지 not_started, lock 해제, rc 1."""
        tr = self.trained()
        stop = StopFlag(install=False)
        orig = self.eval_factory({("S1-01", "accuracy"): "sleep"})

        def factory(kind, *a):
            """정확도 자식을 띄울 때 정지 요청."""
            if kind == "accuracy":
                stop.set()
            return orig(kind, *a)

        res = run_eval(self.config(), self.ecfg, repo_root=self.root, run_set="s1_t1", argv=["eval"],
                       child_factory=factory, stop=stop, poll_s=0.05, gpu_apps_cmd=[sys.executable, "-c", "pass"],
                       other_counter=lambda: 0)
        self.assertEqual((res.rc, res.stop_reason), (1, "parent_signal"))
        self.assertEqual([c["status"] for c in res.candidates], ["stopped", "not_started", "not_started"])
        self.assertIsNone(read_lock(tr.run_set_dir))

    def test_cleanup_unconfirmed_aborts_and_keeps_lock(self):
        """신호 실패로 정리 미확인 → 묶음 중단·lock 유지·핸들 보존, 이후 후보 자식 0."""
        tr = self.trained()

        def no_kill(pid, sig):
            """신호 실패."""
            raise PermissionError("not permitted")

        res = self.evaluate(modes={("S1-01", "timing640"): "sleep"}, cfg=self.config(eval_deadline_s=0.3),
                            kill_fn=no_kill, killpg_fn=no_kill)
        for p in res.unreaped:
            self.addCleanup(lambda p=p: (p.kill(), p.wait(5)))

        self.assertEqual(res.rc, 1)
        self.assertEqual(len(res.unreaped), 1)
        self.assertEqual(read_lock(tr.run_set_dir)["state"], "cleanup_unconfirmed")
        self.assertEqual([e for e, _, _ in self.jobs], ["S1-01"])

    def test_weights_mismatch_and_not_completed_and_blocked(self):
        """best.pt가 바뀌면 failed_not_started, 학습 실패 후보는 skipped, 실행 중 job_state면 rc 2."""
        tr = self.trained()
        _write(os.path.join(self.exp(tr, "S1-01", "yolov8n-seg"), "weights", "best.pt"), b"tampered", "wb")
        _write(os.path.join(self.exp(tr, "S1-02", "yolo11n-seg"), "status.json"), json.dumps({"status": "failed"}))
        res = self.evaluate()

        self.assertEqual([c["status"] for c in res.candidates][:2], ["failed_not_started", "skipped: not_completed (failed)"])
        self.assertEqual({e for e, _, _ in self.jobs}, {"S1-03"})

        js = os.path.join(self.exp(tr, "S1-03", "yolo26n-seg"), "job_state.json")
        st = _read_json(js)
        st["state"] = "running"
        _write(js, json.dumps(st))
        self.jobs.clear()
        res = self.evaluate()
        self.assertEqual((res.rc, self.jobs), (2, []))

    def test_fine_class_mismatch_fails_record(self):
        """manifest의 부품 수가 라벨 줄과 다르면 기록 단계 EvalError → failed, eval.json 없음."""
        tr = self.trained()
        _write(os.path.join(self.root, "data", "gz", "manifest.jsonl"),
               json.dumps({"sample_id": "g0", "parts": [{"cls": "BOLT_A", "labeled": True}]}) + "\n"
               + json.dumps({"sample_id": "g1", "parts": [{"cls": "BOLT_B", "labeled": True}]}) + "\n")
        res = self.evaluate()

        self.assertEqual([c["status"] for c in res.candidates], ["failed"] * 3)
        self.assertFalse(os.path.exists(os.path.join(self.exp(tr, "S1-01", "yolov8n-seg"), "eval", "eval.json")))


class EvalLegacyTest(EvalFixture):
    """legacy 등록 run_set 평가."""

    def test_legacy_registration_is_evaluated_from_source_weights(self):
        """등록된 completed 후보는 원본 best.pt 경로로 평가되고, incomplete 후보는 skipped, 원본 파일은 무변."""
        from test_training_legacy import all_files
        src = self.make_legacy(summary_rc=1)
        reg = self.register()
        before = all_files(src)
        res = self.evaluate(run_set="s1_legacy")

        self.assertEqual([c["status"] for c in res.candidates],
                         ["evaluated", "skipped: not_completed (incomplete)", "evaluated"])
        self.assertTrue(all(w.startswith(src) for _, _, w in self.jobs), self.jobs)
        rec = _read_json(os.path.join(reg.run_set_dir, "S1-01_yolov8n-seg", "eval", "eval.json"))
        self.assertIn("legacy_untracked", rec["train"]["notes"])
        self.assertEqual(all_files(src), before)


class EvalR1Test(EvalFixture):
    """오류 경로 정리, 기록 실패의 완료 게시 금지, 실제 평가 입력 fingerprint 경계."""

    def patched(self, module, name, fn):
        """모듈 속성을 시험 동안 바꾼다."""
        orig = getattr(module, name)
        setattr(module, name, fn)
        self.addCleanup(setattr, module, name, orig)
        return orig

    def test_f1_invalid_existing_eval_json_no_children_no_lock(self):
        """기존 eval.json이 잘렸거나 필드가 없으면 invalid_existing_eval(자식 0), lock 해제·rc 1."""
        tr = self.trained()
        bad = {"S1-01_yolov8n-seg": "{", "S1-02_yolo11n-seg": json.dumps({"accuracy": {}})}
        for exp, text in bad.items():
            os.makedirs(os.path.join(tr.run_set_dir, exp, "eval"))
            _write(os.path.join(tr.run_set_dir, exp, "eval", "eval.json"), text)
        res = self.evaluate()

        self.assertEqual(res.rc, 1)
        self.assertEqual([c["status"] for c in res.candidates], ["invalid_existing_eval"] * 2 + ["evaluated"])
        self.assertEqual({e for e, _, _ in self.jobs}, {"S1-03"})
        for exp, text in bad.items():
            with open(os.path.join(tr.run_set_dir, exp, "eval", "eval.json")) as f:
                self.assertEqual(f.read(), text)
        self.assertIsNone(read_lock(tr.run_set_dir))
        self.assertEqual([r["번호"] for r in _read_json(os.path.join(tr.run_set_dir, "summary.json"))["rows"]],
                         ["S1-03"])

    def test_f2_modified_published_eval_is_not_accepted(self):
        """게시된 eval.json 값을 바꾸면 해시가 맞지 않아 다음 호출에서 invalid_existing_eval이다."""
        tr = self.trained()
        self.evaluate()
        path = os.path.join(self.exp(tr, "S1-01", "yolov8n-seg"), "eval", "eval.json")
        rec = _read_json(path)
        rec["accuracy"]["gz640"]["map50_box"] = 0.99
        _write(path, json.dumps(rec))
        self.jobs.clear()
        res = self.evaluate()

        self.assertEqual([c["status"] for c in res.candidates],
                         ["invalid_existing_eval", "already_evaluated", "already_evaluated"])
        self.assertEqual((res.rc, self.jobs), (1, []))
        self.assertEqual([r["번호"] for r in _read_json(os.path.join(tr.run_set_dir, "summary.json"))["rows"]],
                         ["S1-02", "S1-03"])

    def test_f1_malformed_accuracy_fields_rejected(self):
        """정확도 기록의 predictions가 빈 객체이거나 by_size 값이 정수가 아니면 failed, eval.json 없음."""
        tr = self.trained()
        res = self.evaluate(modes={("S1-01", "accuracy"): "badpred", ("S1-02", "accuracy"): "badsize"})

        self.assertEqual([c["status"] for c in res.candidates], ["failed", "failed", "evaluated"])
        for exp_id, fam in FAMILIES[:2]:
            self.assertFalse(os.path.exists(os.path.join(self.exp(tr, exp_id, fam), "eval", "eval.json")))

    def test_f1_summary_failure_keeps_error_and_releases_lock(self):
        """요약 생성 예외는 밖으로 새지 않고 원래 오류를 남기며 lock 해제, rc 1, 평가 결과는 그대로다."""
        from eyetoarm.training import eval_runner as er
        tr = self.trained()

        def boom(rows):
            """요약 실패."""
            raise KeyError("summary field")

        self.patched(er, "summary_markdown", boom)
        res = self.evaluate()
        self.assertEqual(res.rc, 1)
        self.assertTrue(any("summary field" in e for e in res.errors), res.errors)
        self.assertIsNone(read_lock(tr.run_set_dir))
        self.assertEqual([c["status"] for c in res.candidates], ["evaluated"] * 3)

    def test_f2_watch_write_failure_is_not_success(self):
        """감시 기록 쓰기 실패 → 자식이 정상이어도 failed, eval.json 없음, 요약 제외, rc 1."""
        from eyetoarm.training import runner as rn
        tr = self.trained()
        real = rn.append_jsonl

        def failing(path, obj):
            """평가 감시 기록만 실패."""
            if path.endswith("watch.jsonl") and os.sep + "eval" + os.sep in path:
                raise OSError("watch disk full")
            return real(path, obj)

        self.patched(rn, "append_jsonl", failing)
        res = self.evaluate(modes={("S1-01", "timing640"): "slow"})

        self.assertEqual(res.rc, 1)
        self.assertEqual(res.candidates[0]["status"], "failed")
        self.assertFalse(os.path.exists(os.path.join(self.exp(tr, "S1-01", "yolov8n-seg"), "eval", "eval.json")))
        st = _read_json(os.path.join(self.exp(tr, "S1-01", "yolov8n-seg"), "eval", "attempt_1", "status.json"))
        self.assertTrue(any("watch disk full" in p for p in st["problems"]), st)

    def test_f2_status_or_attempts_write_failure_not_published(self):
        """결과 뒤 status.json·attempts end 쓰기가 실패하면 게시되지 않고 다음은 already_evaluated가 아니다."""
        from eyetoarm.training import eval_runner as er
        for target in ("status", "attempts_end"):
            with self.subTest(target=target):
                tr = self.run_batch({}, run_set=f"s1_{target.replace('_', '')}")
                real_w, real_a = er.write_json_atomic, er.append_jsonl

                def w(path, obj, real_w=real_w, target=target):
                    """attempt status.json 실패."""
                    if target == "status" and path.endswith("status.json") and "attempt_" in path:
                        raise OSError("status write failed")
                    return real_w(path, obj)

                def a(path, obj, real_a=real_a, target=target):
                    """attempts end 실패."""
                    if target == "attempts_end" and obj.get("event") == "end" and path.endswith("attempts.jsonl") \
                            and os.sep + "eval" + os.sep in path:
                        raise OSError("attempts write failed")
                    return real_a(path, obj)

                er.write_json_atomic, er.append_jsonl = w, a
                try:
                    res = self.evaluate(run_set=tr.run_set_dir.rsplit(os.sep, 1)[1])
                finally:
                    er.write_json_atomic, er.append_jsonl = real_w, real_a

                self.assertEqual(res.rc, 1)
                self.assertEqual([c["status"] for c in res.candidates], ["failed"] * 3)
                d = self.exp(tr, "S1-01", "yolov8n-seg")
                self.assertFalse(os.path.exists(os.path.join(d, "eval", "eval.json")))
                self.assertEqual(_read_json(os.path.join(tr.run_set_dir, "summary.json"))["rows"], [])
                res2 = self.evaluate(run_set=tr.run_set_dir.rsplit(os.sep, 1)[1])
                self.assertEqual([c["status"] for c in res2.candidates], ["evaluated"] * 3)

    def test_f3_eval_inputs_fingerprint_recorded_and_tracks_content(self):
        """eval.json에 평가 입력의 개수·내용 해시가 남고, 라벨·manifest 내용이 바뀌면 fingerprint가 달라진다."""
        a = self.run_batch({}, run_set="s1_fa")
        self.evaluate(run_set="s1_fa")
        fa = _read_json(os.path.join(self.exp(a, "S1-01", "yolov8n-seg"), "eval", "eval.json"))
        fa = fa["conditions"]["data_fingerprint"]["eval_inputs"]
        self.assertEqual((fa["gz640"]["images"]["count"], fa["gz640"]["labels"]["count"]), (2, 2))
        self.assertEqual(len(fa["gz640"]["manifest"]["sha256"]), 64)
        self.assertEqual(len(fa["test640"]["dataset_yaml"]["sha256"]), 64)

        _write(os.path.join(self.root, "data", "gz", "labels", "test", "g1.txt"), "0 0.2 0.2 0.3 0.3 0.4 0.2\n")
        b = self.run_batch({}, run_set="s1_fb")
        self.evaluate(run_set="s1_fb")
        fb = _read_json(os.path.join(self.exp(b, "S1-01", "yolov8n-seg"), "eval", "eval.json"))
        fb = fb["conditions"]["data_fingerprint"]["eval_inputs"]
        self.assertNotEqual(fa["gz640"]["labels"]["sha256"], fb["gz640"]["labels"]["sha256"])
        self.assertEqual(fa["gz640"]["images"]["sha256"], fb["gz640"]["images"]["sha256"])

    def test_f3_inputs_changed_during_eval_fails(self):
        """평가 도중 Gazebo manifest 내용이 바뀌면 그 후보는 failed(입력 변경), eval.json 없음."""
        tr = self.trained()
        orig = self.eval_factory()

        def factory(kind, *a):
            """S1-01 정확도 자식 시작 때 manifest를 바꾼다."""
            argv = orig(kind, *a)
            if kind == "accuracy" and self.jobs[-1][0] == "S1-01":
                with open(os.path.join(self.root, "data", "gz", "manifest.jsonl"), "a") as f:
                    f.write("\n")
            return argv

        res = run_eval(self.config(), self.ecfg, repo_root=self.root, run_set="s1_t1", argv=["eval"],
                       child_factory=factory, stop=StopFlag(install=False), poll_s=0.05,
                       gpu_apps_cmd=[sys.executable, "-c", "pass"], other_counter=lambda: 0)
        self.assertEqual(res.candidates[0]["status"], "failed")
        st = _read_json(os.path.join(self.exp(tr, "S1-01", "yolov8n-seg"), "eval", "attempt_1", "status.json"))
        self.assertTrue(any("inputs changed" in p for p in st["problems"]), st)
        self.assertFalse(os.path.exists(os.path.join(self.exp(tr, "S1-01", "yolov8n-seg"), "eval", "eval.json")))


class LegacyR1Test(EvalFixture):
    """등록 묶음이 최종 검증을 마치지 못한 후보는 평가가 소비하지 않는다."""

    def test_f4_sources_changed_during_registration_not_consumed(self):
        """등록 검증 중 원본이 바뀌면 rc 1, completed 게시 없음, 이어지는 eval은 자식 0."""
        src = self.make_legacy()
        real = self.legacy_inspect

        def touching(path):
            """S1-02 best.pt 검사 때 원본 로그를 바꾼다."""
            if "S1-02" in path and path.endswith("best.pt"):
                with open(os.path.join(src, "S1-02_yolo11n-seg.log"), "a") as f:
                    f.write("x\n")
            return real(path)

        self.legacy_inspect = touching
        reg = self.register()
        self.assertEqual(reg.rc, 1)
        for exp_id, fam in FAMILIES:
            self.assertNotEqual(self.reg(reg, exp_id, fam)["status"], "completed")
        self.assertFalse(os.path.exists(os.path.join(reg.run_set_dir, "REGISTERED.json")))
        res = self.evaluate(run_set="s1_legacy")
        self.assertEqual(self.jobs, [])
        self.assertTrue(all(c["status"].startswith("skipped") for c in res.candidates), res.candidates)

    def test_f4_late_write_failure_not_consumed(self):
        """등록 후반 SHA256SUMS 쓰기 실패 → rc 1·오류 보고, 이어지는 eval 자식 0, 변조 기록도 안 쓴다."""
        from eyetoarm.training import eval_runner as er
        self.make_legacy()
        orig = er.write_sha256sums

        def boom(*a, **k):
            """쓰기 실패."""
            raise OSError("sums failed")

        er.write_sha256sums = boom
        try:
            reg = self.register()
        finally:
            er.write_sha256sums = orig
        self.assertEqual(reg.rc, 1)
        self.assertTrue(any("sums failed" in e for e in reg.errors))
        self.evaluate(run_set="s1_legacy")
        self.assertEqual(self.jobs, [])

    def test_f4_tampered_registration_not_consumed(self):
        """정상 등록 뒤 registration.json을 바꾸면 게시 표지·해시가 달라 eval이 소비하지 않는다."""
        self.make_legacy(summary_rc=1)
        reg = self.register()
        path = os.path.join(reg.run_set_dir, "S1-02_yolo11n-seg", "registration.json")
        r = _read_json(path)
        r["status"] = "completed"
        _write(path, json.dumps(r))
        res = self.evaluate(run_set="s1_legacy")

        self.assertEqual([e for e, _, _ in self.jobs if e == "S1-02"], [])
        self.assertTrue(res.candidates[1]["status"].startswith("skipped"), res.candidates)
        self.assertEqual(res.candidates[0]["status"], "evaluated")


if __name__ == "__main__":
    unittest.main()
