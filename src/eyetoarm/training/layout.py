"""산출물 배치·원자 기록·소유권 lock·완료/재개 판정(학습기 설계 r2 §4). 순수 로직이며 프로세스를 띄우지 않는다.

- run_set 이름: 소문자·숫자·`_.-`, 64자 이하. smoke 설정은 `smoke_` 접두사 필수, 비교 설정은 금지.
- 실험 폴더 `<run_set>/<exp_id>_<family>`는 `os.mkdir`로 원자 생성(있으면 거부, 덮어쓰기 없음).
- `OWNER.lock`은 `O_CREAT|O_EXCL`로 원자 생성. 해제는 내용의 pid·시작 시각이 같은 소유자만.
- 기록: JSON은 임시 파일 → fsync → `os.replace`(원자 교체), 이력은 JSONL 줄 추가만.
- 완료 판정은 strip된 정상 완료본(epoch −1·optimizer None)을 인정하고, 최종 epoch 근거는 results.csv 1..E + progress(새 실행)
  또는 대체 근거(legacy: 로그 완료 문구·run_all rc·잔존)로 따로 본다. legacy에 없는 콜백 기록을 꾸며 넣지 않는다.
- 체크포인트 메타(epoch·optimizer 유무·train_args·load 가능)는 torch를 쓰는 자식 프로세스가 읽어 넘긴다.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re

LOCK_NAME = "OWNER.lock"
_RUN_SET_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")
# 완료·재개에서 같아야 하는 공통 학습 키(train_args 기준)
COMMON_KEYS = ("data", "imgsz", "batch", "epochs", "seed", "deterministic", "workers", "device")


class LayoutError(RuntimeError):
    """배치·소유권 오류."""


def validate_run_set_name(name: str, kind: str) -> str:
    """run_set 이름 규칙(설정 kind와 smoke_ 접두사 일치)."""
    if not isinstance(name, str) or not _RUN_SET_RE.fullmatch(name):
        raise LayoutError(f"invalid run_set name: {name!r}")
    if (kind == "smoke") != name.startswith("smoke_"):
        raise LayoutError(f"run_set {name!r} must {'start' if kind == 'smoke' else 'not start'} with 'smoke_' "
                          f"for kind {kind!r}")
    return name


def make_exp_dir(run_set_dir: str, exp_id: str, family: str) -> str:
    """실험 폴더를 원자적으로 새로 만든다. 있으면 LayoutError."""
    os.makedirs(run_set_dir, exist_ok=True)
    path = os.path.join(run_set_dir, f"{exp_id}_{family}")
    try:
        os.mkdir(path)
    except FileExistsError as exc:
        raise LayoutError(f"experiment directory already exists (no overwrite): {path}") from exc
    return path


def write_json_atomic(path: str, obj) -> None:
    """JSON을 임시 파일에 쓰고 fsync 뒤 원자 교체한다. 직렬화 실패(ValueError)면 기존 파일을 건드리지 않는다."""
    text = json.dumps(obj, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def append_jsonl(path: str, obj) -> None:
    """JSONL 한 줄 추가(이전 줄은 바꾸지 않음)."""
    line = json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def acquire_lock(run_set_dir: str, info: dict) -> str:
    """OWNER.lock을 원자적으로 만든다. 이미 있으면 LayoutError(아무것도 쓰지 않음)."""
    os.makedirs(run_set_dir, exist_ok=True)
    path = os.path.join(run_set_dir, LOCK_NAME)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise LayoutError(f"run_set is owned by another process: {path}") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, allow_nan=False)
    return path


def read_lock(run_set_dir: str) -> dict | None:
    """lock 내용(없으면 None)."""
    path = os.path.join(run_set_dir, LOCK_NAME)
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def release_lock(run_set_dir: str, *, pid: int, start_ticks: int | None) -> bool:
    """내용의 pid·시작 시각이 같을 때만 lock을 지운다. 지웠으면 True."""
    lock = read_lock(run_set_dir)
    if lock is None or lock.get("pid") != pid or lock.get("start_ticks") != start_ticks:
        return False
    os.remove(os.path.join(run_set_dir, LOCK_NAME))
    return True


def parse_epochs(text: str) -> list[int]:
    """results.csv의 epoch 열을 정수 목록으로(헤더 공백 허용). 열이 없거나 정수가 아니면 LayoutError."""
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise LayoutError("results.csv is empty")
    header = [h.strip() for h in rows[0]]
    if "epoch" not in header:
        raise LayoutError("results.csv has no epoch column")
    i = header.index("epoch")
    out = []
    for r in rows[1:]:
        if not r:
            continue
        v = r[i].strip()
        if not re.fullmatch(r"[0-9]+", v):
            raise LayoutError(f"non-integer epoch value: {v!r}")
        out.append(int(v))
    return out


def _ckpt_reasons(name: str, meta: dict | None, target_epochs: int, common: dict) -> list[str]:
    """완료본 가중치(strip 형식) 검사."""
    if not meta or not meta.get("exists"):
        return [f"{name}: missing"]
    r = []
    if not meta.get("loadable"):
        r.append(f"{name}: not loadable")
    if not meta.get("sha256"):
        r.append(f"{name}: sha256 not recorded")
    if not (meta.get("epoch") == -1 and meta.get("optimizer_none") is True):
        r.append(f"{name}: not a finished (stripped) checkpoint: epoch={meta.get('epoch')} "
                 f"optimizer_none={meta.get('optimizer_none')}")
    args = meta.get("train_args") or {}
    if args.get("epochs") != target_epochs:
        r.append(f"{name}: train_args.epochs {args.get('epochs')} != {target_epochs}")
    for k in COMMON_KEYS:
        if args.get(k) != common.get(k):
            r.append(f"{name}: train_args.{k} {args.get(k)!r} != {common.get(k)!r}")
    return r


def check_completion(ev: dict) -> tuple[bool, list[str]]:
    """완료 판정(설계 r2 §4 조건 1~5). (완료 여부, 미충족 이유 목록)."""
    E = ev["target_epochs"]
    r = []

    if ev.get("rc") != 0:
        r.append(f"exit code {ev.get('rc')!r}")
    if ev.get("cleanup_confirmed") is not True:
        r.append("process group cleanup not confirmed")
    if ev.get("epochs") != list(range(1, E + 1)):
        r.append(f"results.csv epochs are not exactly 1..{E}")
    r += _ckpt_reasons("best", ev.get("best"), E, ev["common"])
    r += _ckpt_reasons("last", ev.get("last"), E, ev["common"])

    legacy = ev.get("legacy")
    if legacy is None:
        p = ev.get("progress")
        if not p or p.get("finished") is not True or p.get("final_epoch") != E:
            r.append(f"progress.json missing or not finished at epoch {E}")
    else:
        # legacy: 콜백 기록이 원래 없다. 대체 근거만 본다(만들어 넣지 않음)
        if legacy.get("log_completed_epochs") != E:
            r.append(f"legacy log completion line epochs {legacy.get('log_completed_epochs')!r} != {E}")
        if legacy.get("summary_rc") != 0:
            r.append(f"legacy run_all rc {legacy.get('summary_rc')!r}")
        if legacy.get("summary_remaining") != 0:
            r.append(f"legacy run_all remaining_in_pgid {legacy.get('summary_remaining')!r}")

    if ev.get("records_ok") is not True:
        r.append("required records not all written")

    return (not r), r


def check_resume(*, last: dict, target_epochs: int, common: dict, exp_project: str, exp_name: str,
                 attempts: list, manifest: dict, candidate: dict, job_state: dict) -> list[str]:
    """재개 허용 판정(설계 r2 §4 (a)~(d)). 빈 목록이면 허용."""
    r = []

    if job_state.get("state") in ("running", "starting", "cleanup_unconfirmed"):
        r.append(f"job state {job_state.get('state')} blocks resume")
    if not last or not last.get("exists") or not last.get("loadable"):
        r.append("last.pt missing or not loadable")
        return r
    epoch = last.get("epoch")
    if epoch is None or epoch < 0 or last.get("optimizer_none"):
        r.append("last.pt is a finished (stripped) checkpoint; nothing to resume")
    elif epoch + 1 >= target_epochs:
        r.append(f"checkpoint epoch {epoch} is already at the target ({target_epochs})")

    args = last.get("train_args") or {}
    if (args.get("project"), args.get("name")) != (exp_project, exp_name):          # (a) 실험 신원
        r.append(f"checkpoint belongs to {args.get('project')}/{args.get('name')}")
    for k in COMMON_KEYS:                                                             # (b) 공통 설정
        if args.get(k) != common.get(k):
            r.append(f"checkpoint train_args.{k} {args.get(k)!r} != {common.get(k)!r}")
    chain = attempts[-1].get("end_last_sha256") if attempts else None                 # (c) 체크포인트 사슬
    if chain != last.get("sha256"):
        r.append("last.pt sha256 does not match the last attempt record")
    init = (manifest.get("initial_weights") or {}).get("sha256")                      # (d) 후보
    if (manifest.get("exp_id"), manifest.get("family"), init) != (
            candidate.get("exp_id"), candidate.get("family"), candidate.get("sha256")):
        r.append("manifest candidate/initial weights differ from the configured candidate")

    return r


__all__ = ["COMMON_KEYS", "LOCK_NAME", "LayoutError", "acquire_lock", "append_jsonl", "check_completion",
           "check_resume", "make_exp_dir", "parse_epochs", "read_lock", "release_lock", "validate_run_set_name",
           "write_json_atomic"]
