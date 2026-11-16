"""데이터 fingerprint(학습기 설계 r2 §3). 파일을 읽기만 하고 바꾸지 않는다.

- 목록 파일(분할 목록·manifest)과 dataset.yaml: sha256·줄 수.
- 분할(dataset.yaml의 train·val·test 폴더)마다: 영상·라벨 수, 이름(확장자 뺀 stem) 일치, 클래스 번호별 인스턴스 수,
  (이름 + 파일 sha256)을 정렬 순서로 누적한 영상·라벨 해시. 심볼릭 링크는 가리키는 내용을 읽는다.
- 클래스 이름은 dataset.yaml `names`의 순서를 따른다(현재 bolt·nut).
"""
from __future__ import annotations

import hashlib
import os

_CHUNK = 1 << 20


def _sha256(path: str) -> str:
    """파일 sha256."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_yaml_min(path: str) -> tuple[dict, dict]:
    """dataset.yaml의 분할 경로와 names를 읽는다(PyYAML 없이, 이 저장소 형식 한정)."""
    splits, names, in_names = {}, {}, False

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    for raw in lines:
        line = raw.split("#", 1)[0].rstrip()
        if not line:
            continue
        if line.startswith("names:"):
            in_names = True
            continue
        if in_names and line.startswith(" "):
            k, v = line.strip().split(":", 1)
            names[int(k)] = v.strip()
            continue
        in_names = False
        k, v = line.split(":", 1)
        if k.strip() in ("train", "val", "test"):
            splits[k.strip()] = v.strip()

    return splits, names


def compute_fingerprint(repo_root: str, data_yaml: str, split_lists: list[str]) -> dict:
    """fingerprint dict({"lists": …, "splits": …})를 계산한다."""
    yaml_path = os.path.join(repo_root, data_yaml)
    base = os.path.dirname(yaml_path)
    splits, names = _read_yaml_min(yaml_path)
    out = {"lists": {}, "splits": {}}

    for rel in list(split_lists) + [data_yaml]:
        p = os.path.join(repo_root, rel)
        with open(p, "rb") as f:
            n_lines = sum(1 for _ in f)
        out["lists"][rel] = {"sha256": _sha256(p), "lines": n_lines}

    for split, img_rel in splits.items():
        img_dir = os.path.join(base, img_rel)
        lab_dir = os.path.join(base, img_rel.replace("images", "labels", 1))
        imgs, labs = sorted(os.listdir(img_dir)), sorted(os.listdir(lab_dir))
        hi, hl = hashlib.sha256(), hashlib.sha256()
        inst = {n: 0 for n in names.values()}

        for n in imgs:
            hi.update(n.encode())
            hi.update(_sha256(os.path.join(img_dir, n)).encode())

        for n in labs:
            with open(os.path.join(lab_dir, n), "rb") as f:
                data = f.read()
            hl.update(n.encode())
            hl.update(data)
            for line in data.decode("utf-8").splitlines():
                if line.strip():
                    cls = names.get(int(line.split()[0]), "unknown")
                    inst[cls] = inst.get(cls, 0) + 1

        out["splits"][split] = {
            "images": len(imgs), "labels": len(labs),
            "stems_match": [os.path.splitext(a)[0] for a in imgs] == [os.path.splitext(b)[0] for b in labs],
            "images_sha256": hi.hexdigest(), "labels_sha256": hl.hexdigest(), "instances": inst}

    return out


def compare_fingerprint(expected: dict, actual: dict) -> list[str]:
    """두 fingerprint의 차이를 「경로: 기대 != 실제」 목록으로 돌려준다(같으면 빈 목록)."""
    diffs = []

    def walk(a, b, path):
        """재귀 비교."""
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                walk(a.get(k), b.get(k), f"{path}/{k}")
        elif a != b:
            diffs.append(f"{path}: {a!r} != {b!r}")

    walk(expected, actual, "")

    return diffs


__all__ = ["compare_fingerprint", "compute_fingerprint"]
