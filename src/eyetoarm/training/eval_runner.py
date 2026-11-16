"""학습기 I3 — 평가 실행(`eval`)과 기존 결과 등록(`register-legacy`). torch·ultralytics를 import하지 않는다.

**평가(`run_eval`, 설계 r2 §6):**
- 대상: run_set의 완료 후보(새 실행 `status.json` completed + `run_manifest.best_sha256`, 또는 legacy `registration.json`
  status completed)의 `best.pt`. 시작 전에 best.pt sha256이 기록과 같은지 확인한다(다르면 그 후보 평가 시작 안 함).
- 후보마다 자식 3개를 **순서대로**: 시간(크기별 1개, 설정 순서) → 정확도(모델 하나로 공개 test·Gazebo 640·1280).
  크기마다 새 자식이므로 각 크기의 첫 호출은 그 프로세스·그 모델의 첫 predict다. 정확도는 시간 측정이 끝난 뒤에 돈다.
- 자식 수명·감시·정리·소유권은 학습과 같은 `_Batch.child_run`(Popen 전 토큰, eval_deadline_s, 유한 정리, 미확인이면 중단·lock 유지).
- 각 자식 시작·끝에 GPU 사용 프로세스(nvidia-smi, 제한시간)·다른 세션 gz sim/ax_ws 수·MemAvailable을 기록한다(종료시키지 않음).
- 자식 기록이 없거나 일부만 쓰였거나 검사(`validate_*`)에 실패하면 그 후보 평가는 failed이고 `eval.json`을 쓰지 않는다(요약 제외).
  성공하면 `eval/eval_per_gt.json` → `eval/eval.json` 순으로 원자 기록한다. 기존 `eval.json`이 있으면 다시 평가하지 않는다(덮어쓰기 없음).
- test·Gazebo 정답은 평가 기록에만 쓰인다. 학습·조기 종료·신뢰도·후보 설정을 바꾸지 않는다.

**등록(`register_legacy`, 설계 r2 §8·확인점 3):** `legacy.verify_legacy`로 검증한 기록만 새 run_set에 쓴다. 원본은 읽기만 하고 등록 전후
sha256을 비교한다. 콜백 기록(applied·progress)을 만들지 않는다. 쓰는 순서는 등록 기록 → run_set.json → SHA256SUMS → `REGISTERED.json`
(게시 표지, 마지막)이다. 따라서 등록 직후 SHA256SUMS는 표지를 포함하지 않는다(의도된 시점 예외, I3 r1 L1). 표지 자체는 등록 기록 해시로
결합되고, 이후 eval 종료 때 갱신되는 SHA256SUMS에 포함된다.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import uuid

from eyetoarm.training.config import TrainingConfig
from eyetoarm.training.evaluation import (EvalConfig, EvalError, accepted_eval, build_eval_record, canonical_sha,
                                          eval_inputs_fingerprint, load_gz_parts, summary_markdown, summary_row,
                                          validate_accuracy, validate_timing)
from eyetoarm.training.identity import ProcFS, blocking_states
from eyetoarm.training.layout import (LayoutError, acquire_lock, append_jsonl, make_exp_dir, release_lock,
                                      validate_run_set_name, write_json_atomic)
from eyetoarm.training.legacy import RUNNER_FILES, SUMMARY_FILE, sha256_file, tree_hashes, verify_legacy
from eyetoarm.training.records import write_sha256sums
from eyetoarm.training.runner import (RunResult, StopFlag, _Batch, _host_identity, _job_states, _read_json,
                                      count_other_sessions, read_mem_available_kib)

REGISTERED = "REGISTERED.json"   # legacy 등록 묶음 게시 표지(I3 r1 F4)
GPU_APPS_CMD = ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]


def observe_load(cmd: list, timeout_s: float, proc: ProcFS, other_counter=None) -> dict:
    """동시 부하 관측(기록만): GPU 사용 프로세스·다른 세션 수·MemAvailable. 명령 실패·초과는 오류 문자열."""
    out = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "gpu_apps": None, "gpu_error": None}
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        if r.returncode == 0:
            out["gpu_apps"] = [[x.strip() for x in line.split(",")] for line in r.stdout.splitlines() if line.strip()]
        else:
            out["gpu_error"] = f"rc {r.returncode}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        out["gpu_error"] = f"{type(exc).__name__}: {exc}"
    try:
        out["other_sessions"] = (other_counter or (lambda: count_other_sessions(proc)))()
    except Exception as exc:
        out["other_sessions"] = None
        out["other_sessions_error"] = f"{type(exc).__name__}: {exc}"
    try:
        out["mem_available_kib"] = read_mem_available_kib()
    except (OSError, ValueError):
        out["mem_available_kib"] = None
    return out


def _target(repo_root: str, exp_dir: str) -> dict:
    """평가 대상 판정: 가중치 경로·기대 sha256·학습 정보(요약용). 완료가 아니면 status만."""
    reg = _read_json(os.path.join(exp_dir, "registration.json"))
    if reg is not None:
        # 게시 계약(F4): 같은 묶음의 표지가 있고, 원본 무변이 확인됐고, 이 등록 기록의 해시가 표지와 같을 때만 소비한다
        mark = _read_json(os.path.join(os.path.dirname(exp_dir), REGISTERED)) or {}
        if (mark.get("batch_id") != reg.get("batch_id") or mark.get("sources_unchanged") is not True
                or (mark.get("registrations") or {}).get(os.path.basename(exp_dir)) != canonical_sha(reg)):
            return {"status": "not_completed (registration not finalized or modified)"}
        if reg.get("status") != "completed":
            return {"status": f"not_completed ({reg.get('status')})"}
        best = reg["files"]["best"]
        info = _read_json(os.path.join(repo_root, reg["source_dir"], reg["name"], "run_info.json")) or {}
        return {"status": "completed", "weights": os.path.join(repo_root, best["path"]), "sha256": best["sha256"],
                "train": {"family": reg["family"], "weights_source": "official v8.4.0 (legacy args.model)",
                          "weights_sha256": reg["initial_weights"]["current_sha256"],
                          "train_args_text": "공통 설정" + ("" if reg.get("comparable") else
                                                       f" (차이: {'; '.join(reg.get('config_diffs') or [])})"),
                          "data_text": "resplit_v2 " + ("fingerprint 일치" if not reg.get("fingerprint_diffs")
                                                        else "fingerprint 불일치"),
                          "versions_text": f"torch {info.get('torch')}, ultralytics {info.get('ultralytics')}",
                          "train_wall_text": f"{info.get('wall_s')} s" if info.get("wall_s") else "-",
                          "notes": "학습: 임시 실행기(legacy_untracked, 출처·해시 기록)"
                                   + ("" if reg.get("comparable") else ", 비교 조건 차이 있음")}}

    st = _read_json(os.path.join(exp_dir, "status.json")) or {}
    if st.get("status") != "completed":
        return {"status": f"not_completed ({st.get('status')})"}
    man = _read_json(os.path.join(exp_dir, "run_manifest.json")) or {}
    applied = _read_json(os.path.join(exp_dir, "applied.json")) or {}
    iw = man.get("initial_weights") or {}
    return {"status": "completed", "weights": os.path.join(exp_dir, "weights", "best.pt"),
            "sha256": man.get("best_sha256"),
            "train": {"family": man.get("family"),
                      "weights_source": (iw.get("source") or {}).get("url", "-"), "weights_sha256": iw.get("sha256"),
                      "train_args_text": json.dumps(man.get("train_args"), ensure_ascii=False, sort_keys=True),
                      "data_text": "fingerprint 일치" if man.get("fingerprint_ok") else "fingerprint 불일치",
                      "versions_text": f"torch {applied.get('torch')}, ultralytics {applied.get('ultralytics')}",
                      "train_wall_text": f"{man.get('wall_s')} s", "notes": "학습: 추적 실행기"}}


class _EvalBatch(_Batch):
    """eval 묶음(lock·차단 상태·fingerprint 검사와 자식 수명은 _Batch 그대로)."""

    summary_name = "eval_set.json"

    def __init__(self, cfg, eval_cfg: EvalConfig, *, repo_root, run_set, argv, child_factory, stop, poll_s,
                 gpu_apps_cmd, mem_reader, gpu_cmd, proc, uid, killpg_fn, kill_fn, clock, other_counter):
        """주입 의존성."""
        super().__init__(cfg, repo_root, run_set, argv, child_factory, None, stop, poll_s, {}, {}, mem_reader,
                         gpu_cmd, proc, uid, killpg_fn, kill_fn, clock, other_counter)
        self.eval_cfg = eval_cfg
        self.gpu_apps_cmd = gpu_apps_cmd or GPU_APPS_CMD
        self.other_counter_raw = other_counter

    def load(self) -> dict:
        """부하 관측 1회."""
        return observe_load(self.gpu_apps_cmd, self.watch["monitor_cmd_timeout_s"], self.proc, self.other_counter_raw)

    def evaluate_candidate(self, cand, exp_dir: str, target: dict) -> str:
        """한 후보: 시간(크기별 자식) → 정확도(자식 1) → 기록 검사 → eval.json. 상태 문자열."""
        eval_dir = os.path.join(exp_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True)
        n = 1 + sum(1 for x in os.listdir(eval_dir) if x.startswith("attempt_"))
        attempt_dir = os.path.join(eval_dir, f"attempt_{n}")
        os.mkdir(attempt_dir)   # 원자적 새 시도 폴더(있으면 예외)
        log = os.path.join(eval_dir, "attempts.jsonl")
        append_jsonl(log, {"event": "start", "attempt": n, "weights": target["weights"], "sha256": target["sha256"],
                           "previous_job_state": _read_json(os.path.join(exp_dir, "job_state.json")),
                           "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})

        tm = self.eval_cfg.timing
        jobs = [(f"timing{s}", "timing", s) for s in tm["imgsz"]] + [("accuracy", "accuracy", None)]
        outputs, loads, problems, status = {}, [], [], "failed"
        try:
            inputs_start = eval_inputs_fingerprint(self.repo_root, self.eval_cfg)   # 실제 평가 입력(F3)
        except (OSError, ValueError) as exc:
            inputs_start = None
            problems.append(f"evaluation inputs unreadable: {type(exc).__name__}: {exc}")

        for job, kind, imgsz in ([] if problems else jobs):
            out_path = os.path.join(attempt_dir, f"{job}.json")
            load = {"job": job, "start": self.load()}
            run = self.child_run(
                exp_dir=exp_dir, label=f"{cand.exp_id}:{job}",
                argv_fn=lambda kind=kind, imgsz=imgsz, out_path=out_path: self.child_factory(
                    kind, target["weights"], imgsz, out_path, attempt_dir),
                log_path=os.path.join(attempt_dir, f"{job}.log"), deadline_s=self.watch["eval_deadline_s"],
                attempt=n, js_extra={"job": f"eval:{job}"}, watch_path=os.path.join(attempt_dir, "watch.jsonl"))
            load["end"] = self.load()
            loads.append(load)
            append_jsonl(log, {"event": "job_end", "attempt": n, "job": job, "rc": run["rc"],
                               "stop_reason": run["stop_reason"], "cleanup_confirmed": run["cleanup"]["confirmed"],
                               "watch_write_error": run["watch_fail"], "wall_s": run["wall"], "pid": run["pid"]})
            if run["stop_reason"] == "parent_signal":
                status = "stopped"
                problems.append(f"{job}: parent_signal")
                break
            if not run["cleanup"]["confirmed"]:
                problems.append(f"{job}: cleanup not confirmed")
                break
            if run["rc"] != 0 or run["stop_reason"]:
                problems.append(f"{job}: rc {run['rc']} stop_reason {run['stop_reason']}")
                break
            if run["watch_fail"] is not None:   # 필수 기록(감시 표본) 실패는 성공이 아니다(F2)
                problems.append(f"{job}: watch record write failed: {run['watch_fail']}")
                break
            try:
                with open(out_path, encoding="utf-8") as f:
                    outputs[job] = json.load(f)
            except (OSError, ValueError) as exc:
                problems.append(f"{job}: record unreadable: {type(exc).__name__}: {exc}")
                break
            bad = (validate_timing(outputs[job], imgsz=imgsz, warmup_n=tm["warmup_n"], measure_n=tm["measure_n"])
                   if kind == "timing" else validate_accuracy(outputs[job], self.eval_cfg.accuracy_plan()))
            if bad:
                problems += [f"{job}: {b}" for b in bad]
                break

        if not problems:
            try:
                if eval_inputs_fingerprint(self.repo_root, self.eval_cfg) != inputs_start:
                    raise EvalError("evaluation inputs changed during evaluation")
                parts = load_gz_parts(os.path.join(self.repo_root, self.eval_cfg.gazebo["manifest"]))
                record, per_gt = build_eval_record(
                    exp_id=cand.exp_id, weights={"path": target["weights"], "sha256": target["sha256"]},
                    eval_cfg=self.eval_cfg,
                    fingerprint={"training_data": {"ok": not self.fp_diffs, "diffs": self.fp_diffs},
                                 "eval_inputs": inputs_start},
                    timing={s: outputs[f"timing{s}"] for s in tm["imgsz"]}, accuracy=outputs["accuracy"],
                    plan=self.eval_cfg.accuracy_plan(), parts=parts, loads=loads)
                record.update(train=target["train"], attempt=n)
                summary_row(exp_id=cand.exp_id, train=record["train"], record=record)   # 요약에 쓸 수 있는지 미리 확인
                sha = canonical_sha(record)
                # 게시 계약: per_gt → attempt status(evaluated, eval_sha256) → attempts end → eval.json(마지막).
                # 앞의 필수 기록이 실패하면 eval.json을 쓰지 않는다. eval.json은 status의 해시와 맞을 때만 완료다(accepted_eval).
                write_json_atomic(os.path.join(eval_dir, "eval_per_gt.json"), per_gt)
                write_json_atomic(os.path.join(attempt_dir, "status.json"),
                                  {"status": "evaluated", "eval_sha256": sha, "problems": [], "loads": loads})
                append_jsonl(log, {"event": "end", "attempt": n, "status": "evaluated", "eval_sha256": sha,
                                   "problems": []})
                write_json_atomic(os.path.join(eval_dir, "eval.json"), record)
                return "evaluated"
            except (EvalError, OSError, ValueError, KeyError, TypeError) as exc:
                problems.append(f"record: {type(exc).__name__}: {exc}")

        # 실패 기록(최선 노력): 실패 증거를 남기되 그 쓰기 실패가 원래 문제를 덮지 않게 한다
        for write in (lambda: write_json_atomic(os.path.join(attempt_dir, "status.json"),
                                                {"status": status, "problems": problems, "loads": loads}),
                      lambda: append_jsonl(log, {"event": "end", "attempt": n, "status": status,
                                                 "problems": problems})):
            try:
                write()
            except Exception as exc:
                problems.append(f"failure record write failed: {type(exc).__name__}: {exc}")
        self.result.errors.append(f"{cand.exp_id} eval {status}: {problems[:5]}")
        return status


def run_eval(cfg: TrainingConfig, eval_cfg: EvalConfig, *, repo_root: str, run_set: str, argv: list, child_factory,
             stop: StopFlag, poll_s: float = 0.5, gpu_apps_cmd=None, mem_reader=None, gpu_cmd=None,
             proc: ProcFS | None = None, uid: int | None = None, killpg_fn=os.killpg, kill_fn=os.kill,
             clock=time.monotonic, other_counter=None) -> RunResult:
    """run_set의 완료 후보를 순서대로 평가한다. rc 0 전부 평가, 1 일부 실패·중단, 2 시작 거부."""
    batch = _EvalBatch(cfg, eval_cfg, repo_root=repo_root, run_set=run_set, argv=argv, child_factory=child_factory,
                       stop=stop, poll_s=poll_s, gpu_apps_cmd=gpu_apps_cmd, mem_reader=mem_reader, gpu_cmd=gpu_cmd,
                       proc=proc, uid=uid, killpg_fn=killpg_fn, kill_fn=kill_fn, clock=clock,
                       other_counter=other_counter)
    if not os.path.isdir(batch.run_set_dir):
        batch.result.rc = 2
        batch.result.errors.append(f"run_set not found: {batch.run_set_dir}")
        return batch.result
    if not batch.preflight():
        return batch.result

    entries, aborted = batch.result.candidates, None
    for cand in cfg.candidates:   # 중단돼도 모든 후보가 결과에 남도록 먼저 not_started로 넣는다
        entries.append({"exp_id": cand.exp_id, "family": cand.family, "status": "not_started"})
    try:
        for cand, entry in zip(cfg.candidates, entries):
            exp_dir = os.path.join(batch.run_set_dir, f"{cand.exp_id}_{cand.family}")
            if batch.stop.is_set():
                aborted = batch.result.stop_reason = "parent_signal"
                break
            if not os.path.isdir(exp_dir):
                entry["status"] = "absent"
                continue
            if os.path.exists(os.path.join(exp_dir, "eval", "eval.json")):
                rec, why = accepted_eval(os.path.join(exp_dir, "eval"))
                if rec is not None:
                    entry["status"] = "already_evaluated"   # 덮어쓰지 않는다
                else:   # 게시 계약에 맞지 않는 기존 기록: 덮어쓰지도 다시 평가하지도 않고 실패로 보고(F1)
                    entry.update(status="invalid_existing_eval", reason=why)
                    batch.result.errors.append(f"{cand.exp_id} existing eval.json invalid: {why}")
                continue
            target = _target(repo_root, exp_dir)
            if target["status"] != "completed":
                entry["status"] = f"skipped: {target['status']}"
                continue
            actual = sha256_file(target["weights"])
            if not target.get("sha256") or actual != target["sha256"]:
                entry.update(status="failed_not_started",
                             reason=f"best.pt sha256 {actual} != recorded {target.get('sha256')}")
                continue
            entry["status"] = batch.evaluate_candidate(cand, exp_dir, target)
            if entry["status"] == "stopped":
                aborted = batch.result.stop_reason = "parent_signal"
                break
            if not batch.all_confirmed:
                aborted = "cleanup_unconfirmed"
                break
    except BaseException as exc:   # 기록 실패·예상하지 못한 예외·KeyboardInterrupt
        batch.emergency_cleanup(exc)
        aborted = "cleanup_unconfirmed" if not batch.all_confirmed else "error"

    # 요약: 게시 계약을 만족한 eval.json만, 설정 후보 순서대로. 실패는 원래 오류로 남기고 lock 정리로 넘어간다(F1)
    try:
        rows = []
        for cand in cfg.candidates:
            eval_dir = os.path.join(batch.run_set_dir, f"{cand.exp_id}_{cand.family}", "eval")
            if not os.path.exists(os.path.join(eval_dir, "eval.json")):
                continue
            rec, why = accepted_eval(eval_dir)
            if rec is not None:
                rows.append(summary_row(exp_id=cand.exp_id, train=rec.get("train") or {}, record=rec))
        write_json_atomic(os.path.join(batch.run_set_dir, "summary.json"), {"rows": rows})
        tmp = os.path.join(batch.run_set_dir, f"summary.md.tmp.{os.getpid()}")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(summary_markdown(rows))
        os.replace(tmp, os.path.join(batch.run_set_dir, "summary.md"))
    except Exception as exc:
        batch.result.errors.append(f"summary failed: {type(exc).__name__}: {exc}")
        batch.result.rc = max(batch.result.rc, 1)

    batch.finish(aborted)
    ok = {"evaluated", "already_evaluated"}
    batch.result.rc = max(batch.result.rc, 0 if aborted is None and all(
        e["status"] in ok or e["status"] == "absent" or e["status"].startswith("skipped") for e in entries)
        and not any(e["status"] in ("failed", "failed_not_started", "invalid_existing_eval") for e in entries) else 1)
    return batch.result


def register_legacy(cfg: TrainingConfig, *, repo_root: str, source_dir: str, run_set: str, inspect_fn, argv: list,
                    proc: ProcFS | None = None) -> RunResult:
    """기존 S1 결과를 검증해 새 run_set에 등록한다(원본 무변·콜백 기록 미생성). rc 0 전부 completed·무변, 1 일부, 2 거부."""
    proc = proc or ProcFS()
    run_set_dir = os.path.join(repo_root, cfg.common.outputs_dir, run_set)
    result = RunResult(rc=0, run_set_dir=run_set_dir)

    try:
        validate_run_set_name(run_set, cfg.common.kind)
    except LayoutError as exc:
        result.rc = 2
        result.errors.append(str(exc))
        return result
    blocked = blocking_states(_job_states(run_set_dir))
    if blocked:
        result.rc = 2
        result.errors.append(f"blocked by unfinished job states: {blocked}")
        return result
    identity = _host_identity(proc)
    try:
        acquire_lock(run_set_dir, {**identity, "argv": list(argv), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    except LayoutError as exc:
        result.rc = 2
        result.errors.append(str(exc))
        return result

    src = os.path.join(repo_root, source_dir)
    watched = [os.path.join(src, f) for f in RUNNER_FILES + (SUMMARY_FILE,)]
    for c in cfg.candidates:
        name = f"{c.exp_id}_{c.family}"
        watched += [os.path.join(src, name), os.path.join(src, f"{name}.log"), os.path.join(src, f"{name}.watch.txt")]

    unchanged = False
    try:
        # 이미 등록 기록이 있으면 아무것도 쓰지 않는다(덮어쓰기·묶음 섞임 방지)
        existing = [c for c in cfg.candidates if os.path.exists(os.path.join(run_set_dir, f"{c.exp_id}_{c.family}"))]
        if existing or os.path.exists(os.path.join(run_set_dir, REGISTERED)):
            for cand in cfg.candidates:
                result.candidates.append({"exp_id": cand.exp_id, "family": cand.family,
                                          "status": "already_registered" if cand in existing else
                                          "not_registered (run_set already has registrations)"})
            result.errors.append("run_set already has registrations; nothing written")
        else:
            # 1) 모두 검증(쓰기 없음) → 2) 원본 전후 해시 비교 → 3) 등록 기록 → 4) SHA256SUMS → 5) 게시 표지(마지막)
            batch_id = uuid.uuid4().hex
            before = tree_hashes(watched)
            recs = []
            for cand in cfg.candidates:
                rec = verify_legacy(repo_root=repo_root, cfg=cfg, source_dir=source_dir, cand=cand,
                                    inspect_fn=inspect_fn)
                rec.update(config_id=cfg.config_id, batch_id=batch_id,
                           registered_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
                recs.append((cand, rec))
            after = tree_hashes(watched)
            unchanged = before == after
            changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
            if not unchanged:
                result.errors.append(f"legacy sources changed during registration: "
                                     f"{[os.path.relpath(k, repo_root) for k in changed[:5]]}")

            digests = {}
            for cand, rec in recs:
                if not unchanged:   # 검증 근거가 흔들렸으므로 어떤 후보도 completed로 쓰지 않는다
                    rec.update(status="unverified", comparable=False,
                               reasons=rec["reasons"] + ["sources changed during registration"])
                exp_dir = make_exp_dir(run_set_dir, cand.exp_id, cand.family)
                write_json_atomic(os.path.join(exp_dir, "registration.json"), rec)
                digests[f"{cand.exp_id}_{cand.family}"] = canonical_sha(rec)
                result.candidates.append({"exp_id": cand.exp_id, "family": cand.family, "status": rec["status"],
                                          "comparable": rec["comparable"], "reasons": rec["reasons"][:10]})

            write_json_atomic(os.path.join(run_set_dir, "run_set.json"), {
                "run_set": run_set, "kind": "legacy_registration", "batch_id": batch_id, "config_id": cfg.config_id,
                "source_dir": source_dir, "argv": list(argv), "candidates": result.candidates,
                "sources_unchanged": unchanged,
                "changed_sources": [os.path.relpath(k, repo_root) for k in changed],
                "source_sha256": {os.path.relpath(k, repo_root): v for k, v in sorted(before.items())},
                "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
            write_sha256sums(run_set_dir, tag=time.strftime("%Y%m%dT%H%M%S"))
            if unchanged:
                # 게시 표지: 이 묶음의 최종 검증이 끝났다는 마지막 기록. eval은 이것과 등록 기록 해시가 맞을 때만 소비한다
                write_json_atomic(os.path.join(run_set_dir, REGISTERED),
                                  {"batch_id": batch_id, "sources_unchanged": True, "registrations": digests,
                                   "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    except Exception as exc:   # 기록 실패 등: 원래 오류를 남기고 rc 1(쓴 등록 기록은 덮어쓰지 않고 표지 없이 남는다)
        result.errors.append(f"{type(exc).__name__}: {exc}")
        unchanged = False
    finally:
        release_lock(run_set_dir, pid=identity["pid"], start_ticks=identity["start_ticks"])

    done = bool(result.candidates) and all(e.get("status") == "completed" for e in result.candidates)
    result.rc = 0 if done and unchanged and not result.errors else 1
    return result


__all__ = ["GPU_APPS_CMD", "observe_load", "register_legacy", "run_eval"]
