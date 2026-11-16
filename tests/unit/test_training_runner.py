"""학습기 I2 — 순차 실행·격리 자식·감시·정리·재개·복구(`eyetoarm.training.runner`) 시험. 실제 모델·GPU 없이 가짜 자식을 쓴다.

근거: 학습기 설계 r2 §4(job_state·lock·재개)·§5(자식 실행·감시·그룹 정리·다음 후보 조건·부모 중단·기록 실패·종료 코드),
검토 r2 확인점 1(Popen 전 job_token·starting 기록), I1 재검토 r2의 I2 지시 1~5.
가짜 자식은 `python -c`로 results.csv·weights·progress·args.yaml을 쓰거나, 잠자거나, TERM을 무시하거나, 별도 세션 손자를 만든다.
체크포인트 메타(torch.load)는 `inspect_fn`을 주입한다. 모든 시험은 유한 시간에 끝나며 남은 프로세스를 정리한다.
"""
import hashlib
import json
import os
import signal
import socket
import sys
import tempfile
import time
import unittest

from eyetoarm.training.config import training_config_from_dicts
from eyetoarm.training.fingerprint import compute_fingerprint
from eyetoarm.training.identity import ProcFS, job_min_start, own_processes
from eyetoarm.training.layout import read_lock
from eyetoarm.training.runner import StopFlag, recover, run_train, status, terminate_group

E = 2
FAMILIES = (("S1-01", "yolov8n-seg"), ("S1-02", "yolo11n-seg"), ("S1-03", "yolo26n-seg"))


def _write(path, data, mode="w"):
    """파일을 쓰고 닫는다(시험 도우미)."""
    with open(path, mode) as f:
        f.write(data)


def _read_json(path):
    """JSON 파일."""
    with open(path) as f:
        return json.load(f)


def _lines(path):
    """JSONL 줄 목록."""
    with open(path) as f:
        return [json.loads(t) for t in f if t.strip()]


# 가짜 자식: 성공(epoch 1..E, strip 형식은 inspect_fn이 판단), 환경 토큰 기록
SUCCESS = r"""
import json, os, sys, time
exp = sys.argv[1]; E = int(sys.argv[2]); sleep_s = float(sys.argv[3])
time.sleep(sleep_s)
os.makedirs(os.path.join(exp, "weights"), exist_ok=True)
open(os.path.join(exp, "results.csv"), "w").write("epoch,loss\n" + "".join(f"{e},0.1\n" for e in range(1, E + 1)))
for n in ("best.pt", "last.pt"):
    open(os.path.join(exp, "weights", n), "wb").write(n.encode())
open(os.path.join(exp, "args.yaml"), "w").write("epochs: %d\n" % E)
json.dump({"optimizer": "AdamW", "lr": 0.0016}, open(os.path.join(exp, "applied.json"), "w"))
json.dump({"final_epoch": E, "finished": True}, open(os.path.join(exp, "progress.json"), "w"))
open(os.path.join(exp, "child_env_token"), "w").write(os.environ.get("EYETOARM_TRAIN_JOB", ""))
open(os.path.join(exp, "child_times"), "w").write(f"{time.monotonic()}")
"""
SLEEP = "import time; time.sleep(60)"
IGNORE_TERM = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
# 별도 세션 손자(토큰 상속)를 만들고 잔다: 그룹 kill만으로는 손자가 남는 경우
GRANDCHILD = (r"import subprocess, sys, time, os; "
              r"subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True); "
              r"time.sleep(60)")


class Fixture(unittest.TestCase):
    """임시 저장소(데이터·가중치)와 설정."""

    def setUp(self):
        """임시 데이터·가중치·설정을 만든다."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.kill_leftovers)
        self.root = self.tmp.name
        ds = os.path.join(self.root, "data", "ds")
        for split in ("train", "val", "test"):
            for kind in ("images", "labels"):
                os.makedirs(os.path.join(ds, kind, split))
            _write(os.path.join(ds, "images", split, "a.png"), b"img", "wb")
            _write(os.path.join(ds, "labels", split, "a.txt"), "0 0.1 0.1 0.2 0.2 0.3 0.1\n")
        _write(os.path.join(ds, "dataset.yaml"), "train: images/train\nval: images/val\ntest: images/test\n"
                                                 "names:\n  0: bolt\n  1: nut\n")
        os.makedirs(os.path.join(self.root, "data", "splits"))
        _write(os.path.join(self.root, "data", "splits", "train.txt"), "a\n")
        self.lists = ["data/splits/train.txt"]
        os.makedirs(os.path.join(self.root, "outputs", "weights"))
        self.cands = []
        for exp_id, fam in FAMILIES:
            content = f"weights {fam}".encode()
            _write(os.path.join(self.root, "outputs", "weights", f"{fam}.pt"), content, "wb")
            self.cands.append({"exp_id": exp_id, "family": fam, "weights_file": f"{fam}.pt",
                               "source": {"repo": "ultralytics/assets", "tag": "v8.4.0",
                                          "url": f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{fam}.pt"},
                               "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)})
        self.spawned = []

    def kill_leftovers(self):
        """시험이 만든 토큰 프로세스를 정리한다(실패 시험 안전망)."""
        proc = ProcFS()
        for token in getattr(self, "tokens", []):
            for pid in own_processes(token, proc, os.getuid())[0]:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def config(self, **watch_over):
        """시험 설정(E=2, 짧은 감시 간격·기한)."""
        return training_config_from_dicts(*self.config_docs(**watch_over))

    def config_docs(self, **watch_over):
        """시험 설정 문서(공통, 후보)."""
        fp = compute_fingerprint(self.root, "data/ds/dataset.yaml", self.lists)
        common = {"format_version": 1, "kind": "comparison",
                  "train": {"imgsz": 640, "epochs": E, "seed": 0, "deterministic": True, "batch": 16, "device": "0",
                            "workers": 2, "cache": False, "plots": False},
                  "data": {"data_yaml": "data/ds/dataset.yaml", "split_lists": self.lists,
                           "class_names": ["bolt", "nut"], "expected_fingerprint": fp},
                  "watch": {"watch_interval_s": 0.1, "mem_available_stop_kib": 1, "kill_grace_s": 0.5,
                            "train_deadline_s": 20, "eval_deadline_s": 20, "reap_timeout_s": 2,
                            "monitor_cmd_timeout_s": 0.3, **watch_over},
                  "weights_dir": "outputs/weights", "outputs_dir": "outputs/training",
                  "required_env": {"YOLO_AUTOINSTALL": "false", "YOLO_OFFLINE": "1"}}
        return common, {"format_version": 1, "candidates": self.cands}

    def factory(self, plan):
        """후보별 가짜 자식 argv. plan[exp_id] = 'ok'|'sleep'|'ignore'|'grand'|('ok', sleep_s)."""
        def make(cand, exp_dir, resume_from=None):
            """argv 목록."""
            kind = plan.get(cand.exp_id, "ok")
            sleep_s = 0.0
            if isinstance(kind, tuple):
                kind, sleep_s = kind
            self.spawned.append((cand.exp_id, resume_from))
            code = {"sleep": SLEEP, "ignore": IGNORE_TERM, "grand": GRANDCHILD}.get(kind)
            if code:
                return [sys.executable, "-c", code]
            return [sys.executable, "-c", SUCCESS, exp_dir, str(E), str(sleep_s)]
        return make

    @staticmethod
    def inspect(path):
        """가짜 체크포인트 메타(strip 완료 형식)."""
        if not os.path.exists(path):
            return {"exists": False}
        with open(path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        exp = os.path.dirname(os.path.dirname(path))
        args = {"data": "data/ds/dataset.yaml", "imgsz": 640, "batch": 16, "epochs": E, "seed": 0,
                "deterministic": True, "workers": 2, "device": "0", "project": os.path.dirname(exp),
                "name": os.path.basename(exp)}
        return {"exists": True, "sha256": sha, "loadable": True, "epoch": -1, "optimizer_none": True,
                "train_args": args}

    def run_batch(self, plan=None, stop=None, run_set="s1_t1", cfg=None, **kw):
        """run_train 호출."""
        res = run_train(cfg or self.config(), repo_root=self.root, run_set=run_set, argv=["train", run_set],
                        child_factory=self.factory(plan or {}), inspect_fn=self.inspect,
                        stop=stop or StopFlag(install=False), poll_s=0.05, versions={"python": "x"},
                        git={"commit": "t", "dirty": False}, **kw)
        self.tokens = getattr(self, "tokens", []) + [
            _read_json(os.path.join(res.run_set_dir, d, "job_state.json")).get("job_token")
            for d in os.listdir(res.run_set_dir) if os.path.exists(os.path.join(res.run_set_dir, d, "job_state.json"))]
        return res

    def exp(self, res, exp_id, fam):
        """실험 폴더."""
        return os.path.join(res.run_set_dir, f"{exp_id}_{fam}")


class SuccessPathTest(Fixture):
    """정상 순차 실행."""

    def test_all_three_complete_sequentially(self):
        """세 후보 모두 completed, 순차(시작 시각 겹침 없음), job_state exited_cleaned, attempts 시작·끝, lock 해제, run_set 요약, SHA256SUMS, rc 0."""
        res = self.run_batch({"S1-01": ("ok", 0.2), "S1-02": ("ok", 0.2), "S1-03": ("ok", 0.2)})

        self.assertEqual(res.rc, 0, res.errors)
        rs = _read_json(os.path.join(res.run_set_dir, "run_set.json"))
        self.assertEqual([c["status"] for c in rs["candidates"]], ["completed"] * 3)
        times = []
        for exp_id, fam in FAMILIES:
            d = self.exp(res, exp_id, fam)
            self.assertEqual(_read_json(os.path.join(d, "job_state.json"))["state"], "exited_cleaned")
            self.assertEqual(_read_json(os.path.join(d, "status.json"))["status"], "completed")
            att = _lines(os.path.join(d, "attempts.jsonl"))
            self.assertEqual([a["event"] for a in att], ["start", "end"])
            self.assertEqual(len(att[1]["end_last_sha256"]), 64)
            m = _read_json(os.path.join(d, "run_manifest.json"))
            self.assertEqual((m["status"], m["rc"], m["cleanup"]["confirmed"]), ("completed", 0, True))
            self.assertTrue(os.path.exists(os.path.join(d, "applied_args.yaml")))
            with open(os.path.join(d, "child_times")) as f:
                times.append(float(f.read()))
        self.assertEqual(times, sorted(times))
        self.assertIsNone(read_lock(res.run_set_dir))
        self.assertTrue(os.path.exists(os.path.join(res.run_set_dir, "SHA256SUMS")))

    def test_job_state_written_before_spawn_and_token_passed(self):
        """자식 argv를 만들 때(Popen 전) job_state는 이미 starting·job_token을 담고, 자식은 같은 토큰을 환경에서 받는다."""
        seen = []
        base = self.factory({})

        def factory(cand, exp_dir, resume_from=None):
            """Popen 직전 job_state를 확인한다."""
            seen.append(_read_json(os.path.join(exp_dir, "job_state.json")))
            return base(cand, exp_dir, resume_from)

        res = run_train(self.config(), repo_root=self.root, run_set="s1_t2", argv=["train"], child_factory=factory,
                        inspect_fn=self.inspect, stop=StopFlag(install=False), poll_s=0.05, versions={}, git={})

        self.assertEqual([s["state"] for s in seen], ["starting"] * 3)
        for s, (exp_id, fam) in zip(seen, FAMILIES):
            with open(os.path.join(res.run_set_dir, f"{exp_id}_{fam}", "child_env_token")) as f:
                self.assertEqual(f.read(), s["job_token"])
            self.assertEqual(len(s["job_token"]), 32)

    def test_existing_run_set_experiment_not_overwritten(self):
        """같은 run_set을 다시 train하면 실험 폴더가 이미 있어 그 후보는 시작하지 않는다(덮어쓰기 없음)."""
        self.run_batch(run_set="s1_t3")
        res = self.run_batch(run_set="s1_t3")

        self.assertNotEqual(res.rc, 0)
        self.assertTrue(all(c["status"] == "failed_not_started" for c in res.candidates))


class StopAndCleanupTest(Fixture):
    """정지·정리·중단."""

    def test_parent_stop_cleans_child_and_grandchild_and_stops_batch(self):
        """첫 후보 실행 중 정지 플래그(SIGTERM 모사) → 자식과 별도 세션 손자(토큰 상속)까지 정리 확인, 나머지 not_started, lock 해제, rc 1, 사유 parent_signal."""
        stop = StopFlag(install=False)
        t = __import__("threading").Timer(0.6, stop.set)
        t.start()
        res = self.run_batch({"S1-01": "grand"}, stop=stop)
        t.cancel()

        self.assertEqual(res.rc, 1)
        self.assertEqual([c["status"] for c in res.candidates], ["stopped", "not_started", "not_started"])
        js = _read_json(os.path.join(self.exp(res, "S1-01", "yolov8n-seg"), "job_state.json"))
        self.assertEqual(js["state"], "exited_cleaned")
        self.assertEqual(own_processes(js["job_token"], ProcFS(), os.getuid(), job_min_start(js)), ([], []))
        self.assertEqual(_read_json(os.path.join(res.run_set_dir, "run_set.json"))["stop_reason"], "parent_signal")
        self.assertIsNone(read_lock(res.run_set_dir))
        self.assertEqual([s[0] for s in self.spawned], ["S1-01"])

    def test_term_ignored_child_is_killed(self):
        """TERM을 무시하는 자식은 유예 뒤 KILL로 정리되고(정리 확인), 그 후보는 deadline 실패로 기록된 뒤 다음 후보가 계속된다."""
        res = self.run_batch({"S1-01": "ignore"}, cfg=self.config(train_deadline_s=0.5))

        self.assertEqual([c["status"] for c in res.candidates][:1], ["failed"])
        m = _read_json(os.path.join(self.exp(res, "S1-01", "yolov8n-seg"), "run_manifest.json"))
        self.assertEqual((m["stop_reason"], m["cleanup"]["confirmed"], m["cleanup"]["killed"]),
                         ("deadline", True, True))
        self.assertEqual([c["status"] for c in res.candidates][1:], ["completed", "completed"])
        self.assertEqual(res.rc, 1)

    def test_memory_threshold_stops_candidate(self):
        """MemAvailable이 기준보다 작으면(가짜 측정) 그 후보만 mem 사유로 정리·실패하고 다음 후보는 계속된다."""
        calls = {"n": 0}

        def mem():
            """첫 측정(첫 후보 감시 첫 회차)만 부족."""
            calls["n"] += 1
            return 1 if calls["n"] < 2 else 10 ** 9

        res = self.run_batch({"S1-01": "sleep"}, cfg=self.config(mem_available_stop_kib=100), mem_reader=mem)
        m = _read_json(os.path.join(self.exp(res, "S1-01", "yolov8n-seg"), "run_manifest.json"))

        self.assertEqual((m["stop_reason"], m["cleanup"]["confirmed"]), ("mem_available", True))
        self.assertEqual([c["status"] for c in res.candidates], ["failed", "completed", "completed"])

    def test_cleanup_unconfirmed_aborts_keeps_lock_and_blocks_next(self):
        """신호를 보내지 못해(권한 실패 모사) 정리가 확인되지 않으면 묶음 중단·job_state cleanup_unconfirmed·lock 유지·rc 1, 다음 train은 lock을 지워도 거부(rc 2)."""
        def no_kill(pid, sig):
            """신호 실패."""
            raise PermissionError("not permitted")

        res = self.run_batch({"S1-01": "sleep"}, cfg=self.config(train_deadline_s=0.3), kill_fn=no_kill, killpg_fn=no_kill)
        d = self.exp(res, "S1-01", "yolov8n-seg")
        for p in res.unreaped:                               # 시험 정리: 남긴 자식을 직접 회수
            self.addCleanup(lambda p=p: (p.kill(), p.wait(5)))
        self.assertEqual(len(res.unreaped), 1)

        self.assertEqual(res.rc, 1)
        self.assertEqual(_read_json(os.path.join(d, "job_state.json"))["state"], "cleanup_unconfirmed")
        self.assertEqual([c["status"] for c in res.candidates][1:], ["not_started", "not_started"])
        self.assertIsNotNone(read_lock(res.run_set_dir))
        self.assertEqual(_read_json(os.path.join(res.run_set_dir, "run_set.json"))["aborted"], "cleanup_unconfirmed")

        os.remove(os.path.join(res.run_set_dir, "OWNER.lock"))
        again = self.run_batch(run_set="s1_t1")
        self.assertEqual(again.rc, 2)
        self.assertTrue(any("cleanup_unconfirmed" in e for e in again.errors))

    def test_record_failure_preserves_original_error(self):
        """실행 중 필수 기록(status) 쓰기 실패 → 자식 정리·묶음 중단·rc 1, 원래 기록 오류와 정리 결과가 errors에 모두 남는다."""
        from eyetoarm.training import runner as runner_mod
        real = runner_mod.write_json_atomic

        def failing(path, obj):
            """첫 후보 running 상태 기록에서 실패."""
            if path.endswith("job_state.json") and obj.get("state") == "running":
                raise OSError("disk full")
            return real(path, obj)

        runner_mod.write_json_atomic = failing
        try:
            res = self.run_batch({"S1-01": "sleep"})
        finally:
            runner_mod.write_json_atomic = real

        text = "\n".join(res.errors)
        self.assertEqual(res.rc, 1)
        self.assertIn("disk full", text)
        self.assertIn("cleanup", text)
        self.assertEqual([c["status"] for c in res.candidates][1:], ["not_started", "not_started"])

    def test_slow_monitor_command_does_not_stall_deadline(self):
        """GPU 감시 명령이 느려도(가짜 nvidia-smi 5 s, 제한 0.3 s) 표본은 null·오류로 남고 기한(1 s) 판정은 제한시간 이상 늦지 않는다."""
        t0 = time.monotonic()
        res = self.run_batch({"S1-01": "sleep"}, cfg=self.config(train_deadline_s=1.0),
                       gpu_cmd=[sys.executable, "-c", "import time; time.sleep(5)"])
        elapsed_first = _read_json(os.path.join(self.exp(res, "S1-01", "yolov8n-seg"), "run_manifest.json"))["wall_s"]
        samples = _lines(os.path.join(self.exp(res, "S1-01", "yolov8n-seg"), "watch.jsonl"))

        self.assertLess(elapsed_first, 1.0 + 0.3 + 1.5)
        self.assertTrue(samples and all(s["gpu"] is None and "timeout" in s["gpu_error"] for s in samples))
        self.assertLess(time.monotonic() - t0, 30)


class GuardsTest(Fixture):
    """시작 전 검사."""

    def test_weights_mismatch_skips_only_that_candidate(self):
        """가중치가 설정 해시와 다르면 그 후보는 시작하지 않고(failed_not_started) 나머지는 학습한다."""
        _write(os.path.join(self.root, "outputs", "weights", "yolo11n-seg.pt"), b"tampered", "wb")
        res = self.run_batch()

        self.assertEqual([c["status"] for c in res.candidates], ["completed", "failed_not_started", "completed"])
        self.assertEqual([s[0] for s in self.spawned], ["S1-01", "S1-03"])

    def test_fingerprint_mismatch_refuses_everything(self):
        """데이터 fingerprint가 다르면 아무것도 시작하지 않고 rc 2."""
        cfg = self.config()
        _write(os.path.join(self.root, "data", "ds", "labels", "val", "a.txt"), "1 0.1 0.1 0.2 0.2 0.3 0.1\n")
        res = self.run_batch(cfg=cfg)

        self.assertEqual((res.rc, self.spawned), (2, []))

    def test_concurrent_owner_lock_refused(self):
        """다른 부모가 lock을 쥐고 있으면 rc 2이고 아무것도 쓰거나 띄우지 않는다."""
        d = os.path.join(self.root, "outputs", "training", "s1_t1")
        os.makedirs(d)
        _write(os.path.join(d, "OWNER.lock"), json.dumps({"pid": os.getpid(), "start_ticks": 1}))
        res = self.run_batch()

        self.assertEqual((res.rc, self.spawned), (2, []))
        self.assertEqual(sorted(os.listdir(d)), ["OWNER.lock"])


class ResumeRecoverTest(Fixture):
    """재개·상태·복구."""

    def test_resume_uses_experiment_last_pt(self):
        """중단된 실험(미완료 last.pt)을 resume하면 그 실험의 weights/last.pt 경로로 자식을 만들고 attempts에 재개를 덧붙인다."""
        res = self.run_batch({"S1-01": "sleep"}, cfg=self.config(train_deadline_s=0.3))
        d = self.exp(res, "S1-01", "yolov8n-seg")
        os.makedirs(os.path.join(d, "weights"), exist_ok=True)
        _write(os.path.join(d, "weights", "last.pt"), b"mid", "wb")
        last_sha = hashlib.sha256(b"mid").hexdigest()
        # 직전 시도 끝 기록의 사슬 sha를 실제 파일에 맞춘다(중단 시 부모가 남긴 값과 같음을 모사)
        att = _lines(os.path.join(d, "attempts.jsonl"))
        self.assertEqual(att[-1]["event"], "end")

        def mid_inspect(path):
            """미완료 체크포인트(epoch 0, optimizer 있음) 다음엔 완료본."""
            meta = self.inspect(path)
            if path.endswith("last.pt") and meta.get("sha256") == last_sha:
                meta.update(epoch=0, optimizer_none=False)
            return meta

        from eyetoarm.training.runner import run_resume
        with open(os.path.join(d, "attempts.jsonl"), "a") as f:
            f.write(json.dumps({"event": "note", "end_last_sha256": last_sha}) + "\n")
        r2 = run_resume(self.config(), repo_root=self.root, run_set="s1_t1", exp_id="S1-01", argv=["resume"],
                        child_factory=self.factory({}), inspect_fn=mid_inspect, stop=StopFlag(install=False),
                        poll_s=0.05, versions={}, git={})

        self.assertEqual(r2.rc, 0, r2.errors)
        self.assertEqual(self.spawned[-1], ("S1-01", os.path.join(d, "weights", "last.pt")))
        self.assertEqual([a["event"] for a in _lines(os.path.join(d, "attempts.jsonl"))][-2:], ["start", "end"])

    def test_status_and_recover_kill_own(self):
        """부모 사망 + 자식 생존을 모사(lock·job_state running·토큰 프로세스): status는 stale·자기 작업을 보고하고, recover --kill-own은 토큰 프로세스만 정리한 뒤 lock을 푼다."""
        d = os.path.join(self.root, "outputs", "training", "s1_t9")
        exp = os.path.join(d, "S1-01_yolov8n-seg")
        os.makedirs(exp)
        token = "f" * 32
        self.tokens = [token]
        child = __import__("subprocess").Popen([sys.executable, "-c", SLEEP], start_new_session=True,
                                               env={**os.environ, "EYETOARM_TRAIN_JOB": token})
        other = __import__("subprocess").Popen([sys.executable, "-c", SLEEP], start_new_session=True)
        self.addCleanup(lambda: (other.kill(), other.wait(5)))
        proc = ProcFS()
        _write(os.path.join(d, "OWNER.lock"), json.dumps({"pid": 999999, "start_ticks": 1, "host": os.uname().nodename,
                                                          "boot_id": proc.boot_id()}))
        self.addCleanup(lambda: (child.kill(), child.wait(5)))
        _write(os.path.join(exp, "job_state.json"), json.dumps({"state": "running", "job_token": token,
                                                                 "pid": child.pid,
                                                                 "start_ticks": proc.start_ticks(child.pid),
                                                                 "host": os.uname().nodename,
                                                                 "boot_id": proc.boot_id()}))   # 실제 running 기록 형태
        time.sleep(0.2)

        st = status(d)
        self.assertTrue(st["lock_stale_candidate"])
        self.assertEqual(st["own_processes"]["S1-01_yolov8n-seg"], [child.pid])

        rec = recover(d, kill_own=True, grace_s=0.5)
        child.wait(5)
        self.assertTrue(rec["lock_released"], rec)
        self.assertIsNone(other.poll())                     # 토큰 없는 다른 프로세스는 살아 있음
        self.assertIsNone(read_lock(d))
        self.assertEqual(_read_json(os.path.join(exp, "job_state.json"))["state"], "exited_cleaned")


class I2FixTest(Fixture):
    """I2 r0 검토 F1~F4 경계."""

    def recover_setup(self, lock_over=None, js_over=None):
        """토큰 자식 하나와 lock·job_state(running)를 만든다."""
        import subprocess
        d = os.path.join(self.root, "outputs", "training", "s1_rc")
        exp = os.path.join(d, "S1-01_yolov8n-seg")
        os.makedirs(exp)
        token = "e" * 32
        self.tokens = [token]
        child = subprocess.Popen([sys.executable, "-c", SLEEP], start_new_session=True,
                                 env={**os.environ, "EYETOARM_TRAIN_JOB": token})
        self.addCleanup(lambda: (child.kill(), child.wait(5)))
        proc = ProcFS()
        lock = {"pid": 999999, "start_ticks": 1, "host": os.uname().nodename, "boot_id": proc.boot_id()}
        lock.update(lock_over or {})
        js = {"state": "running", "job_token": token, "pid": child.pid, "start_ticks": proc.start_ticks(child.pid),
              "host": os.uname().nodename, "boot_id": proc.boot_id()}
        js.update(js_over or {})
        _write(os.path.join(d, "OWNER.lock"), json.dumps(lock))
        _write(os.path.join(exp, "job_state.json"), json.dumps(js))
        time.sleep(0.2)
        return d, exp, child

    def test_recover_refuses_before_any_signal(self):
        """F1: 살아 있는 lock 소유자·다른 호스트·다른 부팅·job_state의 다른 호스트에서는 신호 0, lock·job_state 무변, 자식 생존."""
        proc = ProcFS()
        me = {"pid": os.getpid(), "start_ticks": proc.start_ticks(os.getpid())}
        for name, lock_over, js_over in (("live owner", me, None), ("other host", {"host": "elsewhere"}, None),
                                         ("other boot", {"boot_id": "x"}, None),
                                         ("job other host", None, {"host": "elsewhere"})):
            with self.subTest(case=name):
                d, exp, child = self.recover_setup(lock_over, js_over)
                with open(os.path.join(d, "OWNER.lock")) as f:
                    lock_before = f.read()
                with open(os.path.join(exp, "job_state.json")) as f:
                    js_before = f.read()

                rec = recover(d, kill_own=True, grace_s=0.3)

                self.assertEqual((rec["signals"], rec["lock_released"]), ([], False), rec)
                self.assertIsNone(child.poll())
                with open(os.path.join(d, "OWNER.lock")) as f:
                    self.assertEqual(f.read(), lock_before)
                with open(os.path.join(exp, "job_state.json")) as f:
                    self.assertEqual(f.read(), js_before)
                child.kill()
                child.wait(5)
                __import__("shutil").rmtree(d)

    def test_recover_refuses_on_unknown_without_signal(self):
        """F1: 작업 뒤 시작한 같은 uid 판별 불가 프로세스가 있으면(가짜 /proc) 신호 0·무변으로 거부한다."""
        from test_training_identity import MY_UID, FakeProc
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        fake = FakeProc(tmp.name)
        fake.add(500, start=900, env={"EYETOARM_TRAIN_JOB": "t" * 32})
        fake.add(501, start=950, env={}, readable=False)
        self.addCleanup(os.chmod, os.path.join(tmp.name, "501", "environ"), 0o600)
        d = os.path.join(self.root, "outputs", "training", "s1_rf")
        os.makedirs(os.path.join(d, "E"))
        boot = ProcFS(tmp.name).boot_id()
        _write(os.path.join(d, "OWNER.lock"), json.dumps({"pid": 77, "start_ticks": 1, "host": os.uname().nodename,
                                                          "boot_id": boot}))
        _write(os.path.join(d, "E", "job_state.json"), json.dumps({"state": "running", "job_token": "t" * 32,
                                                                   "start_ticks": 900, "host": os.uname().nodename,
                                                                   "boot_id": boot}))
        sent = []

        rec = recover(d, kill_own=True, grace_s=0.1, proc=ProcFS(tmp.name), uid=MY_UID,
                      kill_fn=lambda pid, sig: sent.append((pid, sig)))

        self.assertEqual((sent, rec["lock_released"]), ([], False))
        self.assertEqual(len(rec["reasons"]), 1, rec["reasons"])            # 신원은 유효, 판별 불가만 사유
        self.assertIn("unreadable environ", rec["reasons"][0])

    def test_preflight_data_error_releases_own_lock_rc2(self):
        """F2: 데이터 파일이 사라지면(fingerprint 계산 오류) 자식 0·원래 오류 보고·자기 lock 해제·rc 2."""
        cfg = self.config()
        os.remove(os.path.join(self.root, "data", "splits", "train.txt"))
        res = self.run_batch(cfg=cfg)

        self.assertEqual((res.rc, self.spawned), (2, []))
        self.assertTrue(any("train.txt" in e for e in res.errors), res.errors)
        self.assertIsNone(read_lock(res.run_set_dir))

    def test_unconfirmed_plus_state_write_failure_keeps_lock(self):
        """F3: 정리 미확인 뒤 job_state 쓰기까지 실패해도 lock 유지·핸들 보존·다음 후보 금지·원래 오류와 정리 결과 보고."""
        from eyetoarm.training import runner as runner_mod
        real = runner_mod.write_json_atomic

        def failing(path, obj):
            """정리 미확인 상태 기록에서 실패."""
            if path.endswith("job_state.json") and obj.get("state") == "cleanup_unconfirmed":
                raise OSError("disk full at cleanup")
            return real(path, obj)

        def no_kill(pid, sig):
            """신호 실패."""
            raise PermissionError("not permitted")

        runner_mod.write_json_atomic = failing
        try:
            res = self.run_batch({"S1-01": "sleep"}, cfg=self.config(train_deadline_s=0.3), kill_fn=no_kill,
                                 killpg_fn=no_kill)
        finally:
            runner_mod.write_json_atomic = real
        for p in res.unreaped:
            self.addCleanup(lambda p=p: (p.kill(), p.wait(5)))

        text = "\n".join(res.errors)
        self.assertEqual(res.rc, 1)
        self.assertEqual(len(res.unreaped), 1)
        self.assertIsNotNone(read_lock(res.run_set_dir))
        self.assertIn("disk full at cleanup", text)
        self.assertIn("confirmed=False", text)
        self.assertEqual([c["status"] for c in res.candidates][1:], ["not_started", "not_started"])

    def test_cleanup_function_raising_is_unconfirmed(self):
        """F3: 정리 함수가 예외를 올리면(killpg가 RuntimeError) 확인된 것으로 보지 않는다: lock 유지·핸들 보존·rc 1·오류 보고."""
        def boom(pid, sig):
            """예상하지 못한 예외."""
            raise RuntimeError("signal subsystem broken")

        res = self.run_batch({"S1-01": "sleep"}, cfg=self.config(train_deadline_s=0.3), killpg_fn=boom)
        for p in res.unreaped:
            self.addCleanup(lambda p=p: (p.kill(), p.wait(5)))

        self.assertEqual(res.rc, 1)
        self.assertIsNotNone(read_lock(res.run_set_dir))
        self.assertEqual(len(res.unreaped), 1)
        self.assertIn("signal subsystem broken", "\n".join(res.errors))

    def test_watch_samples_rss_other_sessions_and_max_summary(self):
        """F4: 표본에 그룹 RSS 합(공유 중복 가능 표시)·다른 세션 수가 있고, manifest에 표본 기준 최대 RSS·최소 MemAvailable·최대 GPU가 있다. 다른 세션 수는 중단 판정에 쓰지 않는다."""
        res = self.run_batch({"S1-01": "sleep"}, cfg=self.config(train_deadline_s=0.8),
                             other_counter=lambda: 2, gpu_cmd=[sys.executable, "-c", "print('100, 7')"])
        d = self.exp(res, "S1-01", "yolov8n-seg")
        samples = _lines(os.path.join(d, "watch.jsonl"))
        m = _read_json(os.path.join(d, "run_manifest.json"))

        self.assertTrue(samples)
        self.assertTrue(all(s["group_rss_kib"] > 0 and s["other_sessions"] == 2 for s in samples))
        self.assertEqual(m["stop_reason"], "deadline")
        self.assertEqual(m["watch_summary"]["max_group_rss_kib"], max(s["group_rss_kib"] for s in samples))
        self.assertEqual(m["watch_summary"]["max_gpu_mem_mib"], 100)
        self.assertIn("sampled", m["watch_summary"]["note"])
        self.assertIn("shared", m["watch_summary"]["note"])


class RecoverIdentityTest(unittest.TestCase):
    """I2 r1 F1-R: lock 없음 + parent에만 신원이 있는 기록(starting·비상 정리)의 호스트·부팅 확인."""

    def setUp(self):
        """가짜 /proc(같은 uid 토큰 프로세스 선택)과 lock 없는 run_set."""
        from test_training_identity import BOOT, MY_UID, FakeProc
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fake = FakeProc(os.path.join(self.tmp.name, "proc"))
        self.proc = ProcFS(os.path.join(self.tmp.name, "proc"))
        self.uid, self.boot, self.host = MY_UID, BOOT, socket.gethostname()
        self.d = os.path.join(self.tmp.name, "run")
        os.makedirs(os.path.join(self.d, "E"))
        self.token = "p" * 32

    def write_state(self, parent, state="starting", **top):
        """job_state·attempts를 쓰고 그 바이트를 돌려준다."""
        js = {"state": state, "job_token": self.token, "parent": parent, "attempt": 1, **top}
        _write(os.path.join(self.d, "E", "job_state.json"), json.dumps(js))
        _write(os.path.join(self.d, "E", "attempts.jsonl"), json.dumps({"event": "start"}) + "\n")
        return self.snapshot()

    def snapshot(self):
        """job_state·attempts 바이트."""
        out = []
        for n in ("job_state.json", "attempts.jsonl"):
            with open(os.path.join(self.d, "E", n), "rb") as f:
                out.append(f.read())
        return out

    def recover(self):
        """주입 kill_fn으로 복구하고 (보고, 보낸 신호)."""
        sent = []
        rep = recover(self.d, kill_own=True, grace_s=0.05, proc=self.proc, uid=self.uid,
                      kill_fn=lambda pid, sig: sent.append((pid, sig)))
        return rep, sent

    def test_parent_only_foreign_or_missing_identity_refused(self):
        """parent의 다른 host·다른 boot·신원 누락·top-level과 모순 → 토큰 프로세스 유무와 무관하게 신호 0·파일 무변·lock 미해제."""
        dead = {"pid": 4242, "start_ticks": 1}
        cases = {"other host": ({**dead, "host": "elsewhere", "boot_id": self.boot}, {}),
                 "other boot": ({**dead, "host": self.host, "boot_id": "boot-old"}, {}),
                 "missing": (dict(dead), {}),
                 "contradiction": ({**dead, "host": self.host, "boot_id": self.boot}, {"host": "elsewhere"})}
        for with_proc in (False, True):
            if with_proc:
                self.fake.add(600, start=50, env={"EYETOARM_TRAIN_JOB": self.token})
            for name, (parent, top) in cases.items():
                with self.subTest(case=name, token_process=with_proc):
                    before = self.write_state(parent, **top)
                    rep, sent = self.recover()

                    self.assertEqual((sent, rep["signals"], rep["lock_released"]), ([], [], False), rep)
                    self.assertEqual(self.snapshot(), before)
                    self.assertTrue(rep["reasons"])
                    if name == "contradiction":
                        self.assertTrue(any("contradicts" in r for r in rep["reasons"]), rep["reasons"])

    def test_parent_only_same_host_dead_parent_recovers(self):
        """같은 host·boot·죽은 parent의 starting과 비상 정리(cleanup_unconfirmed) 기록은 프로세스가 없으면 복구된다."""
        for state in ("starting", "cleanup_unconfirmed"):
            with self.subTest(state=state):
                self.write_state({"pid": 4242, "start_ticks": 1, "host": self.host, "boot_id": self.boot}, state=state)
                rep, sent = self.recover()

                self.assertEqual((sent, rep["lock_released"]), ([], True), rep)
                self.assertEqual(_read_json(os.path.join(self.d, "E", "job_state.json"))["state"], "exited_cleaned")


class SettleDecisionTest(unittest.TestCase):
    """U1: 결정 지점(terminate 최종 확인·recover 사전 검사·status)이 상한 있는 재조사를 쓴다(가짜 /proc)."""

    def setUp(self):
        """끝난 실제 자식(Popen 핸들)과 lock 없는 run_set."""
        import subprocess
        from test_training_identity import BOOT, MY_UID, ScriptProc, _bad, _ok
        self.SP, self.bad, self.ok, self.uid, self.boot = ScriptProc, _bad, _ok, MY_UID, BOOT
        self.p = subprocess.Popen([sys.executable, "-c", "pass"])
        self.p.wait(10)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = os.path.join(self.tmp.name, "run")
        os.makedirs(os.path.join(self.d, "E"))
        _write(os.path.join(self.d, "E", "job_state.json"), json.dumps(
            {"state": "running", "job_token": "q" * 32, "start_ticks": 50, "host": socket.gethostname(),
             "boot_id": BOOT}))

    def term(self, rounds, reap_s=1.0):
        """terminate_group(신호는 기록만)."""
        return terminate_group(self.p, "q" * 32, grace_s=0.1, reap_s=reap_s, proc=self.SP(rounds), uid=self.uid,
                               killpg_fn=lambda *a: None, kill_fn=lambda *a: None, min_start_ticks=50)

    def test_terminate_final_confirmation(self):
        """일시 unknown → confirmed, 지속 → 미확인(unknown 보고), unknown→토큰 판독 → remaining·미확인, 잔여 기한 없음 → 재조사 없이 미확인."""
        self.assertTrue(self.term([{7: self.bad()}, {}])["confirmed"])
        r = self.term([{7: self.bad()}] * 5)
        self.assertEqual((r["confirmed"], r["unknown"]), (False, [7]))
        r = self.term([{7: self.bad()}, {7: self.ok({"EYETOARM_TRAIN_JOB": "q" * 32})}])
        self.assertEqual((r["confirmed"], r["remaining"]), (False, [7]))
        r = self.term([{7: self.bad()}, {}], reap_s=0.0)
        self.assertEqual((r["confirmed"], r["unknown"]), (False, [7]))

    def test_recover_and_status_use_settled_scan(self):
        """recover 사전 검사: 일시 unknown이면 복구, 지속이면 신호 0·거부. status도 같은 재조사 결과를 보고한다."""
        sent = []
        rep = recover(self.d, kill_own=True, grace_s=0.05, proc=self.SP([{7: self.bad()}] * 9), uid=self.uid,
                      kill_fn=lambda pid, sig: sent.append(pid))
        self.assertEqual((rep["lock_released"], sent), (False, []))
        self.assertEqual(status(self.d, proc=self.SP([{7: self.bad()}] * 9), uid=self.uid)["unknown"]["E"], [7])
        self.assertEqual(status(self.d, proc=self.SP([{7: self.bad()}, {}]), uid=self.uid)["unknown"]["E"], [])

        rep = recover(self.d, kill_own=True, grace_s=0.05, proc=self.SP([{7: self.bad()}, {}]), uid=self.uid,
                      kill_fn=lambda pid, sig: sent.append(pid))
        self.assertTrue(rep["lock_released"], rep)


if __name__ == "__main__":
    unittest.main()


class SingleModeRunTest(Fixture):
    """selection_mode single(#139): 가짜 자식으로 선택한 후보 하나만 실행된다."""

    def test_single_mode_runs_only_selected_candidate(self):
        """single 설정(S1-01 하나)으로 run_train하면 자식은 S1-01만 뜨고, run_set 후보 1개 completed, 다른 계열 폴더는 없다, rc 0."""
        common, _ = self.config_docs()
        cfg = training_config_from_dicts(common, {"format_version": 1, "selection_mode": "single",
                                                  "candidates": [self.cands[0]]})

        res = self.run_batch({"S1-01": ("ok", 0.1)}, cfg=cfg)

        self.assertEqual(res.rc, 0, res.errors)
        self.assertEqual(self.spawned, [("S1-01", None)])
        rs = _read_json(os.path.join(res.run_set_dir, "run_set.json"))
        self.assertEqual([(c["exp_id"], c["status"]) for c in rs["candidates"]], [("S1-01", "completed")])
        self.assertEqual(sorted(d for d in os.listdir(res.run_set_dir) if d.startswith("S1-")),
                         ["S1-01_yolov8n-seg"])


class HsvVRunnerTest(Fixture):
    """hsv_v(#141): 실행기의 기대 공통 값·기록에 연결된다."""

    def test_expected_common_includes_hsv_v_only_when_configured(self):
        """_expected_common은 설정에 hsv_v가 있을 때만 넣는다."""
        from eyetoarm.training.runner import _expected_common
        common, cands = self.config_docs()

        self.assertNotIn("hsv_v", _expected_common(training_config_from_dicts(common, cands)))
        common["train"] = {**common["train"], "hsv_v": 0.6}
        self.assertEqual(_expected_common(training_config_from_dicts(common, cands))["hsv_v"], 0.6)

    def test_single_hsv_v_run_records_train_args(self):
        """single + hsv_v 0.6 가짜 실행: S1-01만 뜨고 run_manifest의 train 기록에 hsv_v 0.6이 남는다."""
        common, _ = self.config_docs()
        common["train"] = {**common["train"], "hsv_v": 0.6}
        cfg = training_config_from_dicts(common, {"format_version": 1, "selection_mode": "single",
                                                  "candidates": [self.cands[0]]})

        res = self.run_batch({"S1-01": ("ok", 0.1)}, cfg=cfg)

        self.assertEqual(self.spawned, [("S1-01", None)])
        m = _read_json(os.path.join(self.exp(res, "S1-01", "yolov8n-seg"), "run_manifest.json"))
        self.assertEqual(m["train"]["hsv_v"], 0.6)
