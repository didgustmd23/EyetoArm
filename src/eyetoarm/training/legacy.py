"""기존 S1 결과를 검증해 등록 기록을 만든다(`register-legacy`). torch·ultralytics를 import하지 않는다.

기존 산출물은 읽기만 하고, 없는 콜백 기록을 만들어 넣지 않는다. 공통 설정과 다른 인자·fingerprint
불일치는 `comparable: false`로 표시하고, 불완전·실패 후보도 그 상태 그대로 등록한다.
"""
from __future__ import annotations

import hashlib
import os
import re

import yaml

from eyetoarm.training.config import TRAIN_KEYS, TrainingConfig, thaw
from eyetoarm.training.fingerprint import compare_fingerprint, compute_fingerprint
from eyetoarm.training.layout import COMMON_KEYS, LayoutError, check_completion, parse_epochs

RUNNER_FILES = ("train_candidate.py", "run_all.sh")
SUMMARY_FILE = "run_all_summary.txt"
_END_RE = re.compile(r"^(?P<name>\S+) end rc=(?P<rc>-?\d+) stop_reason=(?P<reason>.*?) remaining_in_pgid=(?P<left>\d+)")
_DONE_RE = re.compile(r"(\d+) epochs completed in")
_OPT_RE = re.compile(r"optimizer:\S*\s+(?P<opt>[A-Za-z]+)\(lr=(?P<lr>[0-9.eE+-]+)")
LIMITATIONS = (
    "runner legacy_untracked: trained by the untracked script outputs/s1_candidates/train_candidate.py via run_all.sh "
    "(hashes recorded); the tracked runner did not run this training",
    "applied.json and progress.json are absent because the legacy runner had no callbacks; they were not created",
    "optimizer and learning rate are parsed from the training log text, not from a callback record",
    "initial weights hash is that of the current file under outputs/weights (download records in "
    "outputs/s1_candidates/downloads/)",
)


def sha256_file(path: str) -> str | None:
    """파일 sha256(없으면 None)."""
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_hashes(paths: list) -> dict:
    """파일·폴더 목록 → {경로: sha256}(폴더는 아래 파일 전체, 없으면 None)."""
    out = {}
    for p in paths:
        if os.path.isdir(p):
            for dirpath, _, files in os.walk(p):
                for f in sorted(files):
                    q = os.path.join(dirpath, f)
                    out[q] = sha256_file(q)
        else:
            out[p] = sha256_file(p)
    return out


def _rel(repo_root: str, path) -> str | None:
    """저장소 안 절대 경로 → 저장소 상대 경로(밖이면 그대로)."""
    if not isinstance(path, str):
        return path
    if os.path.isabs(path):
        r = os.path.relpath(path, repo_root)
        return path if r.startswith("..") else r
    return path


def parse_summary(text: str, name: str) -> dict:
    """run_all_summary.txt에서 그 실험의 종료 줄(마지막)을 읽는다."""
    ends = [m.groupdict() for m in (_END_RE.match(line) for line in text.splitlines()) if m and m["name"] == name]
    if not ends:
        return {"summary_rc": None, "summary_remaining": None, "summary_stop_reason": None, "summary_end_lines": 0}
    e = ends[-1]
    return {"summary_rc": int(e["rc"]), "summary_remaining": int(e["left"]), "summary_stop_reason": e["reason"],
            "summary_end_lines": len(ends)}


def parse_log(text: str) -> dict:
    """학습 로그: 완료 문구의 에포크 수(마지막), optimizer·lr(로그 문자열)."""
    done = _DONE_RE.findall(text)
    opt = _OPT_RE.search(re.sub(r"\x1b\[[0-9;]*m", "", text))
    return {"log_completed_epochs": int(done[-1]) if done else None,
            "optimizer_from_log": ({"optimizer": opt["opt"], "lr": float(opt["lr"]), "source": "log text"}
                                   if opt else None)}


def _norm_meta(repo_root: str, meta: dict | None) -> dict | None:
    """체크포인트 메타의 train_args.data를 저장소 상대 경로로 맞춘다(절대/상대 표기 차이 제거)."""
    if not meta:
        return meta
    m = dict(meta)
    ta = dict(m.get("train_args") or {})
    if "data" in ta:
        ta["data"] = _rel(repo_root, ta["data"])
    m["train_args"] = ta
    return m


def verify_legacy(*, repo_root: str, cfg: TrainingConfig, source_dir: str, cand, inspect_fn) -> dict:
    """한 후보의 기존 산출물을 검증해 등록 기록(dict)을 만든다. 파일을 쓰지 않는다."""
    name = f"{cand.exp_id}_{cand.family}"
    src = os.path.join(repo_root, source_dir)
    exp = os.path.join(src, name)
    paths = {"exp_dir": exp, "log": os.path.join(src, f"{name}.log"), "watch": os.path.join(src, f"{name}.watch.txt"),
             "summary": os.path.join(src, SUMMARY_FILE),
             **{f"runner:{f}": os.path.join(src, f) for f in RUNNER_FILES}}
    rec = {"exp_id": cand.exp_id, "family": cand.family, "runner": "legacy_untracked", "source_dir": source_dir,
           "name": name, "limitations": list(LIMITATIONS), "applied_json": "absent (not created)",
           "progress_json": "absent (not created)"}
    rec["runner_sha256"] = {f: sha256_file(paths[f"runner:{f}"]) for f in RUNNER_FILES}

    if not os.path.isdir(exp):
        rec.update(status="missing", reasons=[f"source experiment directory not found: {source_dir}/{name}"],
                   comparable=False)
        return rec

    # 원본 근거
    files = {"args": os.path.join(exp, "args.yaml"), "results": os.path.join(exp, "results.csv"),
             "run_info": os.path.join(exp, "run_info.json"), "best": os.path.join(exp, "weights", "best.pt"),
             "last": os.path.join(exp, "weights", "last.pt"), "log": paths["log"], "watch": paths["watch"],
             "summary": paths["summary"]}
    rec["files"] = {k: {"path": _rel(repo_root, v), "sha256": sha256_file(v)} for k, v in files.items()}
    reasons_extra, diffs = [], []

    try:
        with open(files["args"], encoding="utf-8") as f:
            args = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as exc:
        args = {}
        reasons_extra.append(f"args.yaml unreadable: {type(exc).__name__}: {exc}")
    train = thaw(cfg.common.train)
    for k in TRAIN_KEYS:
        if k in args and str(args[k]) != str(train[k]):
            diffs.append(f"args.{k} {args[k]!r} != common {train[k]!r}")
        elif k not in args:
            diffs.append(f"args.{k} missing")
    if _rel(repo_root, args.get("data")) != cfg.common.data["data_yaml"]:
        diffs.append(f"args.data {args.get('data')!r} != {cfg.common.data['data_yaml']!r}")

    model = args.get("model")
    init_path = os.path.join(repo_root, _rel(repo_root, model)) if isinstance(model, str) else None
    init_sha = sha256_file(init_path) if init_path else None
    rec["initial_weights"] = {"args_model": model, "file": os.path.basename(model) if isinstance(model, str) else None,
                              "current_sha256": init_sha, "expected_sha256": cand.sha256,
                              "matches_official": init_sha == cand.sha256
                              and os.path.basename(model or "") == cand.weights_file}
    if not rec["initial_weights"]["matches_official"]:
        diffs.append("initial weights do not match the configured official file/digest")

    fp = cfg.common.data["expected_fingerprint"]
    try:
        actual = compute_fingerprint(repo_root, cfg.common.data["data_yaml"], list(cfg.common.data["split_lists"]))
        fp_diffs = [] if fp is None else compare_fingerprint(thaw(fp), actual)
    except Exception as exc:
        fp_diffs = [f"fingerprint failed: {type(exc).__name__}: {exc}"]
    rec["fingerprint_diffs"] = fp_diffs
    if fp_diffs:
        diffs.append("data fingerprint differs from expected")

    try:
        with open(files["results"], encoding="utf-8") as f:
            epochs = parse_epochs(f.read())
    except (OSError, LayoutError) as exc:
        epochs = []
        reasons_extra.append(f"results.csv unreadable: {exc}")
    try:
        with open(files["log"], encoding="utf-8", errors="replace") as f:
            log = parse_log(f.read())
    except OSError:
        log = {"log_completed_epochs": None, "optimizer_from_log": None}
    try:
        with open(files["summary"], encoding="utf-8") as f:
            summ = parse_summary(f.read(), name)
    except OSError:
        summ = parse_summary("", name)

    best = _norm_meta(repo_root, inspect_fn(files["best"]))
    last = _norm_meta(repo_root, inspect_fn(files["last"]))
    common = {"data": cfg.common.data["data_yaml"], **{k: thaw(cfg.common.train)[k] for k in COMMON_KEYS
                                                        if k != "data"}}
    records_ok = all(os.path.isfile(files[k]) for k in ("args", "results", "log", "summary"))
    ok, reasons = check_completion({
        "rc": summ["summary_rc"], "cleanup_confirmed": summ["summary_remaining"] == 0, "epochs": epochs,
        "target_epochs": train["epochs"], "best": best, "last": last, "common": common, "progress": None,
        "records_ok": records_ok, "legacy": {**log, **summ}})

    rec.update(evidence={**log, **summ, "epochs_rows": len(epochs),
                         "best": {k: best.get(k) for k in ("exists", "loadable", "epoch", "optimizer_none", "sha256")}
                         if best else None,
                         "last": {k: last.get(k) for k in ("exists", "loadable", "epoch", "optimizer_none", "sha256")}
                         if last else None},
               config_diffs=diffs, comparable=ok and not diffs,
               status="completed" if ok and not reasons_extra else "incomplete",
               reasons=reasons + reasons_extra)
    return rec


__all__ = ["LIMITATIONS", "RUNNER_FILES", "parse_log", "parse_summary", "sha256_file", "tree_hashes",
           "verify_legacy"]
