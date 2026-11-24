"""학습 산출물 배치·원자 기록·소유권 lock·완료 판정·재개 판정(`eyetoarm.training.layout`) 시험.

legacy 실행에 없는 콜백 기록은 꾸미지 않는다.
"""
import json
import multiprocessing
import os
import tempfile
import unittest

from eyetoarm.training.layout import (LayoutError, acquire_lock, append_jsonl, check_completion, check_resume,
                                      make_exp_dir, parse_epochs, read_lock, release_lock, validate_run_set_name,
                                      write_json_atomic)


def _write(path, data, mode="w"):
    """파일을 쓰고 닫는다(시험 도우미)."""
    with open(path, mode) as f:
        f.write(data)


def _read(path, mode="r"):
    """파일을 읽고 닫는다(시험 도우미)."""
    with open(path, mode) as f:
        return f.read()


E = 50
COMMON = {"data": "data/derived/bolt_nut_v2/dataset.yaml", "imgsz": 640, "batch": 16, "epochs": 50, "seed": 0,
          "deterministic": True, "workers": 2, "device": "0"}


def csv(epochs):
    """results.csv 본문(epoch 열 + 다른 열)."""
    return "epoch,time,loss\n" + "".join(f"{e},{e * 1.5},0.1\n" for e in epochs)


def stripped(**over):
    """정상 완료(strip) 체크포인트 메타."""
    meta = {"exists": True, "sha256": "a" * 64, "loadable": True, "epoch": -1, "optimizer_none": True,
            "train_args": {**COMMON, "project": "outputs/training/s1_r1", "name": "S1-01_yolov8n-seg",
                           "model": "outputs/weights/yolov8n-seg.pt"}}
    meta.update(over)
    return meta


def evidence(**over):
    """새 실행기 완료 증거(모두 충족)."""
    ev = {"rc": 0, "cleanup_confirmed": True, "epochs": list(range(1, E + 1)), "target_epochs": E,
          "best": stripped(), "last": stripped(sha256="b" * 64), "common": COMMON,
          "progress": {"final_epoch": E, "finished": True}, "records_ok": True, "legacy": None}
    ev.update(over)
    return ev


def take_lock(path, q):
    """다른 프로세스에서 lock을 시도한다(경쟁 시험용)."""
    try:
        acquire_lock(path, {"pid": os.getpid()})
        q.put("ok")
    except LayoutError:
        q.put("busy")


class NamesAndFilesTest(unittest.TestCase):
    """이름·폴더·원자 기록."""

    def setUp(self):
        """임시 폴더."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = self.tmp.name

    def test_run_set_names(self):
        """비교 run_set은 smoke_ 접두사 금지, smoke 설정은 smoke_ 접두사 필수, 경로 문자·빈 이름 거부."""
        validate_run_set_name("s1_r1", "comparison")
        validate_run_set_name("smoke_v8", "smoke")

        for name, kind in (("smoke_x", "comparison"), ("s1", "smoke"), ("../x", "comparison"), ("", "comparison"),
                           ("a/b", "comparison"), ("S1 R1", "comparison")):
            with self.subTest(name=name, kind=kind):
                with self.assertRaises(LayoutError):
                    validate_run_set_name(name, kind)

    def test_exp_dir_never_reused(self):
        """실험 폴더는 원자적으로 새로 만들고 이미 있으면 LayoutError(덮어쓰기 없음, 기존 파일 보존)."""
        p = make_exp_dir(self.d, "S1-01", "yolov8n-seg")
        _write(os.path.join(p, "keep.txt"), "x", "w")

        with self.assertRaises(LayoutError):
            make_exp_dir(self.d, "S1-01", "yolov8n-seg")

        self.assertEqual(_read(os.path.join(p, "keep.txt")), "x")
        self.assertEqual(os.path.basename(p), "S1-01_yolov8n-seg")

    def test_atomic_json_and_jsonl(self):
        """write_json_atomic은 임시 파일 없이 교체되고, append_jsonl은 줄 추가만 한다. 쓰기 불가 경로는 OSError."""
        p = os.path.join(self.d, "m.json")
        write_json_atomic(p, {"a": 1})
        write_json_atomic(p, {"a": 2})
        append_jsonl(os.path.join(self.d, "a.jsonl"), {"n": 1})
        append_jsonl(os.path.join(self.d, "a.jsonl"), {"n": 2})

        self.assertEqual(json.loads(_read(p)), {"a": 2})
        self.assertEqual([json.loads(t)["n"] for t in _read(os.path.join(self.d, "a.jsonl")).splitlines()], [1, 2])
        self.assertEqual(sorted(os.listdir(self.d)), ["a.jsonl", "m.json"])

        with self.assertRaises(OSError):
            write_json_atomic(os.path.join(self.d, "missing_dir", "x.json"), {})

        with self.assertRaises(ValueError):
            write_json_atomic(p, {"nan": float("nan")})
        self.assertEqual(json.loads(_read(p)), {"a": 2})


class LockTest(unittest.TestCase):
    """OWNER.lock 소유권."""

    def setUp(self):
        """임시 run_set 폴더."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = self.tmp.name

    def test_exclusive_and_release_by_owner_only(self):
        """lock은 하나만 잡히고, 해제는 내용의 pid·시작 시각이 같은 소유자만 할 수 있다."""
        acquire_lock(self.d, {"pid": 123, "start_ticks": 9})

        with self.assertRaises(LayoutError):
            acquire_lock(self.d, {"pid": 456, "start_ticks": 1})

        self.assertFalse(release_lock(self.d, pid=456, start_ticks=1))
        self.assertEqual(read_lock(self.d)["pid"], 123)
        self.assertTrue(release_lock(self.d, pid=123, start_ticks=9))
        self.assertIsNone(read_lock(self.d))

    def test_concurrent_processes_one_wins(self):
        """두 프로세스가 동시에 잡으려 하면 정확히 하나만 성공한다."""
        q = multiprocessing.get_context("spawn").Queue()
        ps = [multiprocessing.get_context("spawn").Process(target=take_lock, args=(self.d, q)) for _ in range(2)]
        for p in ps:
            p.start()
        for p in ps:
            p.join(10)

        self.assertEqual(sorted(q.get(timeout=5) for _ in ps), ["busy", "ok"])


class CompletionTest(unittest.TestCase):
    """완료 판정(조건 1~5)."""

    def test_new_run_completed_with_strip_format(self):
        """strip 완료본·epoch 1..E·progress·rc 0·정리 확인·기록 성공이 모두 있으면 completed."""
        ok, reasons = check_completion(evidence())

        self.assertEqual((ok, reasons), (True, []))

    def test_smoke_one_epoch(self):
        """E=1 smoke 완료본도 같은 규칙으로 completed."""
        one = {**COMMON, "epochs": 1}
        ok, _ = check_completion(evidence(epochs=[1], target_epochs=1, common=one,
                                          best=stripped(train_args={**one, "project": "p", "name": "n"}),
                                          last=stripped(train_args={**one, "project": "p", "name": "n"}),
                                          progress={"final_epoch": 1, "finished": True}))
        self.assertTrue(ok)

    def test_incomplete_cases(self):
        """완료 조건 하나라도 어긋나면(rc·정리·epoch·last·best·설정·progress·기록) incomplete."""
        cases = {
            "rc": {"rc": 1},
            "cleanup": {"cleanup_confirmed": False},
            "missing epoch": {"epochs": list(range(1, E))},
            "dup epoch": {"epochs": [1, 1] + list(range(2, E + 1))},
            "order": {"epochs": [2, 1] + list(range(3, E + 1))},
            "unstripped last": {"last": stripped(epoch=37, optimizer_none=False)},
            "no best": {"best": stripped(exists=False)},
            "not loadable": {"last": stripped(loadable=False)},
            "epochs arg": {"best": stripped(train_args={**COMMON, "epochs": 10, "project": "p", "name": "n"})},
            "common key": {"last": stripped(train_args={**COMMON, "imgsz": 1280, "project": "p", "name": "n"})},
            "no progress": {"progress": None},
            "progress unfinished": {"progress": {"final_epoch": 49, "finished": False}},
            "records": {"records_ok": False},
        }

        for name, over in cases.items():
            with self.subTest(case=name):
                ok, reasons = check_completion(evidence(**over))
                self.assertFalse(ok)
                self.assertTrue(reasons)

    def test_legacy_uses_substitute_evidence_without_progress(self):
        """legacy: progress 없이 로그 완료 문구·rc 0·잔존 0이면 completed, 대체 근거가 빠지면 incomplete."""
        legacy = {"log_completed_epochs": E, "summary_rc": 0, "summary_remaining": 0}
        ok, reasons = check_completion(evidence(progress=None, legacy=legacy, records_ok=True))
        self.assertEqual((ok, reasons), (True, []))

        for bad in ({"log_completed_epochs": None}, {"log_completed_epochs": 49}, {"summary_rc": 1},
                    {"summary_remaining": 2}):
            with self.subTest(bad=bad):
                ok, _ = check_completion(evidence(progress=None, legacy={**legacy, **bad}))
                self.assertFalse(ok)

    def test_parse_epochs(self):
        """results.csv의 epoch 열(공백 포함 헤더 허용)을 정수 목록으로 읽고, 열이 없거나 정수가 아니면 LayoutError."""
        self.assertEqual(parse_epochs(csv([1, 2, 3])), [1, 2, 3])
        self.assertEqual(parse_epochs("  epoch, x\n 1, 2\n 2, 3\n"), [1, 2])

        for bad in ("time,x\n1,2\n", "epoch\n1.5\n", ""):
            with self.subTest(bad=bad):
                with self.assertRaises(LayoutError):
                    parse_epochs(bad)


class ResumeTest(unittest.TestCase):
    """재개 판정(체크포인트 신원 (a)~(d))."""

    def args(self, **over):
        """정상 재개 입력."""
        mid = {"exists": True, "sha256": "c" * 64, "loadable": True, "epoch": 20, "optimizer_none": False,
               "train_args": {**COMMON, "project": "outputs/training/s1_r1", "name": "S1-01_yolov8n-seg",
                              "model": "outputs/weights/yolov8n-seg.pt"}}
        a = {"last": mid, "target_epochs": E, "common": COMMON,
             "exp_project": "outputs/training/s1_r1", "exp_name": "S1-01_yolov8n-seg",
             "attempts": [{"attempt": 1, "end_last_sha256": "c" * 64}],
             "manifest": {"exp_id": "S1-01", "family": "yolov8n-seg", "initial_weights": {"sha256": "d" * 64}},
             "candidate": {"exp_id": "S1-01", "family": "yolov8n-seg", "sha256": "d" * 64},
             "job_state": {"state": "exited_cleaned"}}
        a.update(over)
        return a

    def test_resume_allowed(self):
        """미완료(epoch 20, optimizer 있음) 그 실험 last.pt, 사슬 sha 일치, 정리 확인 → 허용(이유 없음)."""
        self.assertEqual(check_resume(**self.args()), [])

    def test_second_resume_model_path_changed(self):
        """두 번째 재개(train_args.model이 last.pt 경로로 바뀜)도 신원 (a)~(d)가 같으면 허용."""
        last = self.args()["last"]
        last = {**last, "train_args": {**last["train_args"], "model": "outputs/training/s1_r1/S1-01_yolov8n-seg/weights/last.pt"}}
        a = self.args(last=last, attempts=[{"attempt": 1, "end_last_sha256": "x" * 64},
                                           {"attempt": 2, "end_last_sha256": "c" * 64}])
        self.assertEqual(check_resume(**a), [])

    def test_resume_rejected(self):
        """strip 완료본·끝난 epoch·다른 위치·설정 다름·sha 불일치·다른 후보·실행 중 → 재개 거부."""
        mid = self.args()["last"]
        cases = {
            "stripped": {"last": {**mid, "epoch": -1, "optimizer_none": True}},
            "finished": {"last": {**mid, "epoch": 49}},
            "other exp": {"last": {**mid, "train_args": {**mid["train_args"], "name": "S1-02_yolo11n-seg"}}},
            "common": {"last": {**mid, "train_args": {**mid["train_args"], "batch": 8}}},
            "chain": {"attempts": [{"attempt": 1, "end_last_sha256": "e" * 64}]},
            "candidate": {"candidate": {"exp_id": "S1-01", "family": "yolov8n-seg", "sha256": "f" * 64}},
            "running": {"job_state": {"state": "running"}},
            "unconfirmed": {"job_state": {"state": "cleanup_unconfirmed"}},
            "not loadable": {"last": {**mid, "loadable": False}},
        }

        for name, over in cases.items():
            with self.subTest(case=name):
                self.assertTrue(check_resume(**self.args(**over)))


if __name__ == "__main__":
    unittest.main()


class HsvVIdentityTest(unittest.TestCase):
    """선택 공통 키 hsv_v의 완료·재개 대조: 설정에 있을 때만 체크포인트 train_args와 비교한다."""

    def resume_args(self, ckpt_hsv, common):
        """재개 입력(체크포인트 hsv_v·기대 공통 값 지정)."""
        args = {**COMMON, "project": "p", "name": "n", "model": "m"}
        if ckpt_hsv is not None:
            args["hsv_v"] = ckpt_hsv
        mid = {"exists": True, "sha256": "c" * 64, "loadable": True, "epoch": 20, "optimizer_none": False,
               "train_args": args}
        return {"last": mid, "target_epochs": E, "common": common, "exp_project": "p", "exp_name": "n",
                "attempts": [{"end_last_sha256": "c" * 64}],
                "manifest": {"exp_id": "S1-08", "family": "yolov8n-seg", "initial_weights": {"sha256": "d" * 64}},
                "candidate": {"exp_id": "S1-08", "family": "yolov8n-seg", "sha256": "d" * 64},
                "job_state": {"state": "exited_cleaned"}}

    def test_resume_hsv_v_mismatch_rejected(self):
        """설정 hsv_v 0.6인데 체크포인트가 0.4이거나 hsv_v가 없으면 재개 거부, 0.6이면 허용."""
        common = {**COMMON, "hsv_v": 0.6}

        self.assertEqual(check_resume(**self.resume_args(0.6, common)), [])
        for ck in (0.4, None):
            with self.subTest(ckpt=ck):
                reasons = check_resume(**self.resume_args(ck, common))
                self.assertTrue(any("hsv_v" in r for r in reasons), reasons)

    def test_resume_without_configured_hsv_v_unchanged(self):
        """설정에 hsv_v가 없으면 체크포인트 hsv_v(기본 0.4)는 비교하지 않는다(기존 동작)."""
        self.assertEqual(check_resume(**self.resume_args(0.4, COMMON)), [])

    def test_completion_hsv_v_checked_only_when_configured(self):
        """완료 판정: 설정 hsv_v 0.6이면 best·last train_args의 hsv_v가 0.6이어야 하고, 설정에 없으면 비교하지 않는다."""
        common = {**COMMON, "hsv_v": 0.6}
        good = {**stripped()["train_args"], "hsv_v": 0.6}
        bad = {**stripped()["train_args"], "hsv_v": 0.4}

        ok, _ = check_completion(evidence(common=common, best=stripped(train_args=good),
                                          last=stripped(sha256="b" * 64, train_args=good)))
        self.assertTrue(ok)
        ok, reasons = check_completion(evidence(common=common, best=stripped(train_args=bad),
                                                last=stripped(sha256="b" * 64, train_args=good)))
        self.assertFalse(ok)
        self.assertTrue(any("hsv_v" in r for r in reasons), reasons)
        ok, _ = check_completion(evidence(best=stripped(train_args=bad), last=stripped(sha256="b" * 64, train_args=bad)))
        self.assertTrue(ok)


class DegreesScaleIdentityTest(unittest.TestCase):
    """선택 공통 키 degrees·scale도 설정에 있을 때만 체크포인트 train_args와 대조한다(hsv_v와 같은 규칙)."""

    def resume_with(self, ckpt_over, common):
        """체크포인트 train_args에 ckpt_over를 넣은 재개 입력(hsv_v 시험의 입력 구성을 그대로 씀)."""
        r = HsvVIdentityTest.resume_args(self, None, common)
        r["last"]["train_args"] = {**r["last"]["train_args"], **ckpt_over}
        return r

    def test_resume_degrees_mismatch_rejected(self):
        """설정 degrees 180인데 체크포인트가 0(ultralytics 기본)이면 재개 거부, 180이면 허용."""
        common = {**COMMON, "degrees": 180.0}

        self.assertEqual(check_resume(**self.resume_with({"degrees": 180.0}, common)), [])
        reasons = check_resume(**self.resume_with({"degrees": 0.0}, common))
        self.assertTrue(any("degrees" in r for r in reasons), reasons)

    def test_resume_scale_list_compared(self):
        """설정 scale [0.1, 1.5]이면 체크포인트 scale 0.5는 거부, 같은 목록은 허용."""
        common = {**COMMON, "scale": [0.1, 1.5]}

        self.assertEqual(check_resume(**self.resume_with({"scale": [0.1, 1.5]}, common)), [])
        reasons = check_resume(**self.resume_with({"scale": 0.5}, common))
        self.assertTrue(any("scale" in r for r in reasons), reasons)

    def test_unconfigured_degrees_not_compared(self):
        """설정에 degrees·scale이 없으면 체크포인트 값(기본 0·0.5)은 비교하지 않는다(기존 동작)."""
        self.assertEqual(check_resume(**self.resume_with({"degrees": 0.0, "scale": 0.5}, COMMON)), [])
