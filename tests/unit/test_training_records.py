"""학습기 I1 — 시작 전 manifest·SHA256SUMS(`eyetoarm.training.records`) 시험.

근거: 학습기 설계 r2 §4(시작 전 `run_manifest` status starting: 해석된 공통 설정·후보·초기 가중치·fingerprint 결과·명령줄·
git·버전), §6(SHA256SUMS는 자기 자신·OWNER.lock 제외, 갱신 때 이전 판 보존·원자 교체).
"""
import hashlib
import json
import os
import tempfile
import unittest

from eyetoarm.training.config import load_training_config
from eyetoarm.training.records import start_manifest, write_sha256sums


def _write(path, data, mode="w"):
    """파일을 쓰고 닫는다(시험 도우미)."""
    with open(path, mode) as f:
        f.write(data)


def _read(path, mode="r"):
    """파일을 읽고 닫는다(시험 도우미)."""
    with open(path, mode) as f:
        return f.read()


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class RecordsTest(unittest.TestCase):
    """manifest·해시 목록."""

    def setUp(self):
        """임시 폴더."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = self.tmp.name

    def test_start_manifest_contents(self):
        """시작 전 manifest는 status starting, config_id, 공통 학습 값, 후보·공식 초기 가중치 sha256, fingerprint 결과, 명령줄, git, 버전을 담고 JSON으로 직렬화된다."""
        cfg = load_training_config(os.path.join(ROOT, "config/training/s1_common.json"),
                                   os.path.join(ROOT, "config/training/s1_candidates.json"))
        cand = cfg.candidates[0]
        m = start_manifest(cfg, cand, run_set="s1_r1", argv=["train", "s1_r1"],
                           fingerprint_diffs=[], git={"commit": "c5fbc05", "dirty": True},
                           versions={"python": "3.12"})

        self.assertEqual((m["status"], m["config_id"], m["exp_id"], m["family"], m["run_set"]),
                         ("starting", cfg.config_id, "S1-01", "yolov8n-seg", "s1_r1"))
        self.assertEqual(m["initial_weights"], {"file": "yolov8n-seg.pt", "sha256": cand.sha256,
                                                "source": cand.source})
        self.assertEqual(m["train"]["epochs"], 50)
        self.assertEqual((m["fingerprint_ok"], m["git"]["dirty"], m["argv"]), (True, True, ["train", "s1_r1"]))
        json.dumps(m, allow_nan=False)

    def test_sha256sums_excludes_and_preserves(self):
        """SHA256SUMS는 하위 파일 전부(자기 자신·OWNER.lock·이전 판 제외)의 sha256이고, 다시 쓰면 이전 판을 SHA256SUMS.<태그>로 보존한다."""
        os.makedirs(os.path.join(self.d, "e"))
        _write(os.path.join(self.d, "e", "a.txt"), b"a", "wb")
        _write(os.path.join(self.d, "OWNER.lock"), "{}", "w")
        write_sha256sums(self.d, tag="t1")
        first = _read(os.path.join(self.d, "SHA256SUMS"))

        self.assertEqual(first, f"{hashlib.sha256(b'a').hexdigest()}  e/a.txt\n")

        _write(os.path.join(self.d, "e", "b.txt"), b"b", "wb")
        write_sha256sums(self.d, tag="t2")

        self.assertEqual(_read(os.path.join(self.d, "SHA256SUMS.t2")), first)
        self.assertIn("e/b.txt", _read(os.path.join(self.d, "SHA256SUMS")))
        self.assertNotIn("SHA256SUMS", _read(os.path.join(self.d, "SHA256SUMS")).replace("  e/", ""))

        with self.assertRaises(FileExistsError):
            write_sha256sums(self.d, tag="t2")


if __name__ == "__main__":
    unittest.main()
