"""학습 데이터 fingerprint(`eyetoarm.training.fingerprint`) 시험. 임시 데이터로 계산·비교한다.

기대값과 다르면 학습을 거부하고 원본은 바꾸지 않는다.
"""
import os
import tempfile
import unittest

from eyetoarm.training.fingerprint import compare_fingerprint, compute_fingerprint


def _write(path, data, mode="w"):
    """파일을 쓰고 닫는다(시험 도우미)."""
    with open(path, mode) as f:
        f.write(data)


def _read(path, mode="r"):
    """파일을 읽고 닫는다(시험 도우미)."""
    with open(path, mode) as f:
        return f.read()


YAML = "train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: bolt\n  1: nut\n"


class FingerprintTest(unittest.TestCase):
    """계산과 비교."""

    def setUp(self):
        """임시 저장소 구조(data/ds·data/splits)를 만든다."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        ds = os.path.join(self.root, "data", "ds")
        os.makedirs(os.path.join(self.root, "data", "splits"))
        os.makedirs(ds)
        _write(os.path.join(ds, "dataset.yaml"), YAML, "w")
        for split, n in (("train", 3), ("val", 1), ("test", 1)):
            for kind in ("images", "labels"):
                os.makedirs(os.path.join(ds, kind, split))
            for i in range(n):
                _write(os.path.join(ds, "images", split, f"{split}{i}.png"), bytes([i]) * 8, "wb")
                _write(os.path.join(ds, "labels", split, f"{split}{i}.txt"), "0 0.1 0.1 0.2 0.2 0.3 0.1\n1 0.5 0.5 0.6 0.6 0.7 0.5\n", "w")
        for name in ("train.txt", "valid.txt", "test.txt"):
            _write(os.path.join(self.root, "data", "splits", name), "a\nb\n", "w")
        self.lists = ["data/splits/train.txt", "data/splits/valid.txt", "data/splits/test.txt"]

    def fp(self):
        """현재 fingerprint."""
        return compute_fingerprint(self.root, "data/ds/dataset.yaml", self.lists)

    def test_compute_counts_and_hashes(self):
        """분할별 영상·라벨 수, stem 일치, 클래스 인스턴스, 목록 줄 수·sha256이 계산된다."""
        f = self.fp()

        self.assertEqual((f["splits"]["train"]["images"], f["splits"]["train"]["labels"],
                          f["splits"]["train"]["stems_match"], f["splits"]["train"]["instances"]),
                         (3, 3, True, {"bolt": 3, "nut": 3}))
        self.assertEqual(f["lists"]["data/splits/train.txt"]["lines"], 2)
        self.assertEqual(len(f["lists"]["data/ds/dataset.yaml"]["sha256"]), 64)
        self.assertEqual(compare_fingerprint(f, self.fp()), [])

    def test_changes_detected(self):
        """영상 바이트·라벨 내용·목록 내용 변경, 파일 추가, stem 불일치는 비교에서 차이로 나온다."""
        ds = None
        cases = [
            lambda: _write(os.path.join(ds, "images", "val", "val0.png"), b"changed", "wb"),
            lambda: _write(os.path.join(ds, "labels", "test", "test0.txt"), "0 0.1 0.1 0.2 0.2 0.3 0.3\n", "a"),
            lambda: _write(os.path.join(self.root, "data", "splits", "valid.txt"), "c\n", "a"),
            lambda: os.rename(os.path.join(ds, "labels", "train", "train2.txt"),
                              os.path.join(ds, "labels", "train", "zzz.txt")),
        ]

        for i, change in enumerate(cases):
            with self.subTest(i=i):
                self.setUp()
                base = self.fp()
                ds = os.path.join(self.root, "data", "ds")
                change()
                self.assertNotEqual(compare_fingerprint(base, self.fp()), [])

    def test_does_not_modify_data(self):
        """계산은 파일을 바꾸지 않는다(수정 시각·내용 무변)."""
        p = os.path.join(self.root, "data", "ds", "labels", "val", "val0.txt")
        before = (os.stat(p).st_mtime_ns, _read(p))
        self.fp()

        self.assertEqual((os.stat(p).st_mtime_ns, _read(p)), before)


if __name__ == "__main__":
    unittest.main()
