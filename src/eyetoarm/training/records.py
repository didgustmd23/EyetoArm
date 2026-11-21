"""학습 시작 전 manifest(`start_manifest`)와 폴더 해시 목록(`write_sha256sums`).

SHA256SUMS를 다시 쓸 때 기존 판은 `SHA256SUMS.<tag>`로 보존하고(있으면 FileExistsError) 새 판은 원자 교체한다.
legacy 등록의 게시 표지 `REGISTERED.json`은 SHA256SUMS 뒤에 쓰므로 등록 직후 목록에는 표지가 없다.
"""
from __future__ import annotations

import hashlib
import os
import time

from eyetoarm.training.config import Candidate, TrainingConfig, thaw

_CHUNK = 1 << 20


def start_manifest(cfg: TrainingConfig, cand: Candidate, *, run_set: str, argv: list, fingerprint_diffs: list,
                   git: dict, versions: dict) -> dict:
    """시작 전 manifest dict."""
    return {"status": "starting", "run_set": run_set, "config_id": cfg.config_id, "kind": cfg.common.kind,
            "exp_id": cand.exp_id, "family": cand.family, "train": cfg.train_args(),
            "data": {k: thaw(v) for k, v in cfg.common.data.items() if k != "expected_fingerprint"},
            "initial_weights": {"file": cand.weights_file, "sha256": cand.sha256, "source": thaw(cand.source)},
            "fingerprint_ok": not fingerprint_diffs, "fingerprint_diffs": list(fingerprint_diffs),
            "argv": list(argv), "git": dict(git), "versions": dict(versions),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def _sha256(path: str) -> str:
    """파일 sha256."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _excluded(rel: str) -> bool:
    """SHA256SUMS에서 뺄 파일."""
    name = os.path.basename(rel)
    return (rel == "SHA256SUMS" or rel.startswith("SHA256SUMS.") or name == "OWNER.lock" or ".tmp." in name
            or name.endswith(".tmp"))


def write_sha256sums(root: str, *, tag: str) -> str:
    """root 아래 파일 목록 해시를 SHA256SUMS로 쓴다. 기존 판은 SHA256SUMS.<tag>로 보존."""
    path = os.path.join(root, "SHA256SUMS")
    lines = []

    for dirpath, _, files in os.walk(root):
        for f in files:
            rel = os.path.relpath(os.path.join(dirpath, f), root)
            if not _excluded(rel):
                lines.append(f"{_sha256(os.path.join(root, rel))}  {rel}")

    if os.path.exists(path):
        backup = f"{path}.{tag}"
        if os.path.exists(backup):
            raise FileExistsError(backup)
        os.link(path, backup)

    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("".join(line + "\n" for line in sorted(lines, key=lambda s: s.split("  ", 1)[1])))
    os.replace(tmp, path)

    return path


__all__ = ["start_manifest", "write_sha256sums"]
