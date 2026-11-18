"""후보 순차 학습 실행기 부모 로직(학습기 설계 r2 §4·§5, 검토 r2 확인점 1, I2 지시 1~4). torch·ultralytics를 import하지 않는다.

- 부모 하나가 후보를 **하나씩** 실행한다. 자식은 `start_new_session=True`로 띄워 pgid = 자식 pid를 얻는다.
- **Popen 전에** `job_state.json`에 `job_token`·`starting`·부모 신원을 쓰고(확인점 1), 띄운 뒤 pid·pgid·시작 시각을 갱신한다.
  자식 환경에는 `EYETOARM_TRAIN_JOB=<token>`과 자동 설치 방지 변수를 넣는다(손자도 상속).
- 감시: `poll_s`마다 정지 플래그·기한·MemAvailable을 보고, `watch_interval_s`마다 표본(GPU는 제한시간 있는 외부 명령)을 쓴다.
  중단 판정은 `/proc/meminfo`와 부모 단조 시계만 쓴다.
- 정리(`terminate_group`): 그룹 TERM → 유예 → 그룹 KILL → leader 회수 → **토큰으로 자기 작업 잔존 확인**(별도 세션 손자 포함)
  → 남으면 pid별 KILL(신호 직전 토큰 재확인) → 다시 확인. 남은 것·판별 불가가 없을 때만 정리 확인이다.
- 다음 후보 조건: 이전 후보의 정리가 확인됐을 때만. 미확인이면 묶음 중단·lock 유지(`cleanup_unconfirmed`).
- 부모 중단(정지 플래그)·예외·기록 실패: 실행 중 자식을 정리 → 다음 후보 시작 금지 → 원래 오류와 정리 결과를 함께 기록
  → 정리가 확인됐을 때만 lock 해제. 종료 코드: 모두 완료 0, 실패·중단·정리 미확인 1, 설정·가중치 묶음·fingerprint·lock·차단 상태 2.
- 기록은 한 번만 시도하고 실패하면 다시 시도하지 않는다(필수 기록 실패 → 묶음 중단).
"""
from __future__ import annotations

import hashlib
import os
import shutil
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field

from eyetoarm.training.config import TrainingConfig, thaw
from eyetoarm.training.fingerprint import compare_fingerprint, compute_fingerprint
from eyetoarm.training.identity import (TOKEN_ENV, ProcFS, _scan, blocking_states, job_min_start, new_job_token,
                                        own_processes, parent_alive, settled_processes, stale_lock_decision)
from eyetoarm.training.layout import (OPTIONAL_COMMON_KEYS, LayoutError, acquire_lock, append_jsonl,
                                      check_completion, check_resume, make_exp_dir, parse_epochs, read_lock,
                                      release_lock, validate_run_set_name, write_json_atomic)
from eyetoarm.training.records import start_manifest, write_sha256sums
from eyetoarm.training.weights import check_weights

GPU_CMD = ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"]


class StopFlag:
    """정지 플래그. `install=True`면 SIGINT·SIGTERM 처리기가 플래그만 세운다(감시 루프가 다음 회차에 확인)."""

    def __init__(self, install: bool = True) -> None:
        """처리기를 설치할지."""
        self._set = False
        self.signal_name: str | None = None
        self._old = {}
        if install:
            for sig in (signal.SIGINT, signal.SIGTERM):
                self._old[sig] = signal.signal(sig, self._handler)

    def _handler(self, signum, frame) -> None:
        """신호 처리기(플래그만)."""
        self._set = True
        self.signal_name = signal.Signals(signum).name

    def set(self) -> None:
        """정지를 요청한다."""
        self._set = True

    def is_set(self) -> bool:
        """정지 요청이 있었는가."""
        return self._set

    def restore(self) -> None:
        """처리기를 원래대로 돌린다."""
        for sig, old in self._old.items():
            signal.signal(sig, old)
        self._old = {}


@dataclass
class RunResult:
    """실행 결과."""

    rc: int
    run_set_dir: str
    candidates: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    stop_reason: str | None = None
    unreaped: list = field(default_factory=list)   # 정리 미확인으로 남긴 자식 Popen 핸들(호출자·복구가 사용)


def read_mem_available_kib(proc_root: str = "/proc") -> int:
    """MemAvailable(kB)."""
    with open(os.path.join(proc_root, "meminfo"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    raise ValueError("MemAvailable not found")


def gpu_sample(cmd: list, timeout_s: float) -> tuple[dict | None, str | None]:
    """GPU 표본(제한시간 있는 외부 명령). 실패·시간 초과면 (None, 사유)."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        return None, f"timeout after {timeout_s} s"
    except OSError as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if out.returncode != 0:
        return None, f"rc {out.returncode}"
    try:
        used, util = (int(v.strip()) for v in out.stdout.strip().splitlines()[0].split(","))
        return {"mem_used_mib": used, "util_pct": util}, None
    except (ValueError, IndexError):
        return None, f"unparsable output {out.stdout[:80]!r}"


def group_rss_kib(token: str, proc: ProcFS, uid: int, floor: int | None, *, failed: list | None = None) -> int:
    """토큰 자기 작업 프로세스 RSS 합(kB). 공유 메모리(데이터로더 fork 등)를 중복 계산할 수 있는 참고값이다.

    읽기에 실패한 pid(조사와 읽기 사이에 끝난 프로세스 등)는 합에서 빠진다. 이 합은 중단 판정에 쓰지 않는 참고값이라 실패를
    전파하지 않지만, 과소 집계가 드러나도록 `failed` 목록이 주어지면 그 pid를 넣는다(#180 #11).
    """
    total = 0
    for pid in own_processes(token, proc, uid, floor)[0]:
        try:
            total += proc.rss_kib(pid)
        except (OSError, ValueError, IndexError):
            if failed is not None:
                failed.append(pid)
    return total


def count_other_sessions(proc: ProcFS) -> int:
    """다른 세션의 `gz sim`·`ax_ws` 프로세스 수(기록만, 중단 판정·신호에 쓰지 않음)."""
    n = 0
    for pid in proc.pids():
        try:
            cmd = proc.cmdline(pid)
        except (OSError, ValueError):
            continue
        if ("gz sim" in cmd or "ax_ws" in cmd) and "count_other_sessions" not in cmd:
            n += 1
    return n


def decide(mem_kib: int | None, elapsed_s: float, watch, deadline_s: float | None = None) -> str | None:
    """중단 판정(순수): MemAvailable 기준 미만 → mem_available, 경과 > 기한(기본 학습 기한, 평가는 eval_deadline_s) → deadline."""
    if mem_kib is not None and mem_kib < watch["mem_available_stop_kib"]:
        return "mem_available"
    if elapsed_s > (watch["train_deadline_s"] if deadline_s is None else deadline_s):
        return "deadline"
    return None


def _pid_has_token(pid: int, token: str, proc: ProcFS, uid: int) -> bool:
    """신호 직전 신원 재확인: 그 pid가 지금도 자기 uid·같은 토큰인가."""
    try:
        return proc.uid(pid) == uid and proc.environ(pid).get(TOKEN_ENV) == token
    except (OSError, ValueError, IndexError):
        return False


def terminate_group(p, token: str, *, grace_s: float, reap_s: float, proc: ProcFS, uid: int, poll_s: float = 0.05,
                    killpg_fn=os.killpg, kill_fn=os.kill, min_start_ticks: int | None = None) -> dict:
    """자식 그룹과 토큰 잔존 프로세스를 유한 정리한다. 결과 dict(confirmed 포함). 예외를 내지 않는다."""
    info = {"leader_reaped": False, "terminated": False, "killed": False, "remaining": [], "unknown": [],
            "confirmed": False, "errors": []}

    def signal_group(sig):
        """그룹 신호(실패는 기록)."""
        try:
            killpg_fn(p.pid, sig)
            return True
        except ProcessLookupError:
            return False
        except OSError as exc:
            info["errors"].append(f"killpg {signal.Signals(sig).name}: {type(exc).__name__}: {exc}")
            return False

    if p.poll() is None:
        info["terminated"] = signal_group(signal.SIGTERM)
        try:
            p.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            info["killed"] = signal_group(signal.SIGKILL) or info["killed"]
            try:
                p.wait(timeout=reap_s)
            except subprocess.TimeoutExpired:
                pass
    else:
        signal_group(signal.SIGKILL)   # leader가 끝났어도 그룹에 남은 worker 정리(없으면 ProcessLookupError)

    info["leader_reaped"] = p.poll() is not None

    end = time.monotonic() + reap_s
    own, unknown = _scan(token, proc, uid, min_start_ticks)

    while own and time.monotonic() < end:
        for pid in own:
            if _pid_has_token(pid, token, proc, uid):
                try:
                    kill_fn(pid, signal.SIGKILL)
                    info["killed"] = True
                except ProcessLookupError:
                    pass
                except OSError as exc:
                    info["errors"].append(f"kill {pid}: {type(exc).__name__}: {exc}")
        time.sleep(poll_s)
        own, unknown = _scan(token, proc, uid, min_start_ticks)

    # 최종 확인: 판별 불가가 남았으면 정리 기한(end) 안에서 상한 있는 재조사(U1 P2). 재조사 중 토큰이 판독되면 remaining
    if unknown:
        own, unknown, info["settle"] = settled_processes(token, proc, uid, min_start_ticks, deadline=end,
                                                         initial=(own, unknown))
    else:
        unknown = []
    info["remaining"], info["unknown"] = own, sorted(unknown)
    info["confirmed"] = info["leader_reaped"] and not own and not unknown

    return info


def _expected_common(cfg: TrainingConfig) -> dict:
    """train_args와 비교할 공통 값(ultralytics에 넘기는 형태 그대로)."""
    t = cfg.train_args()
    out = {"data": cfg.common.data["data_yaml"], "imgsz": t["imgsz"], "batch": t["batch"], "epochs": t["epochs"],
           "seed": t["seed"], "deterministic": t["deterministic"], "workers": t["workers"], "device": t["device"]}
    for k in OPTIONAL_COMMON_KEYS:   # 선택 키(hsv_v)는 설정에 있을 때만 대조한다
        if k in t:
            out[k] = t[k]
    return out


def _host_identity(proc: ProcFS) -> dict:
    """부모 신원(lock·job_state용)."""
    return {"pid": os.getpid(), "start_ticks": proc.start_ticks(os.getpid()), "host": socket.gethostname(),
            "boot_id": proc.boot_id()}


def _job_states(run_set_dir: str) -> dict:
    """실험별 job_state(읽기 실패는 unknown 상태로)."""
    out = {}
    if not os.path.isdir(run_set_dir):
        return out
    for name in sorted(os.listdir(run_set_dir)):
        path = os.path.join(run_set_dir, name, "job_state.json")
        if os.path.exists(path):
            try:
                import json
                with open(path, encoding="utf-8") as f:
                    out[name] = json.load(f)
            except (OSError, ValueError) as exc:
                out[name] = {"state": "cleanup_unconfirmed", "read_error": str(exc)}
    return out


def _read_json(path: str):
    """JSON(없으면 None)."""
    import json
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: str) -> list:
    """JSONL(없으면 빈 목록)."""
    import json
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(t) for t in f if t.strip()]


class _Batch:
    """한 번의 train/resume 실행 상태(부모 소유)."""

    def __init__(self, cfg, repo_root, run_set, argv, child_factory, inspect_fn, stop, poll_s, versions, git,
                 mem_reader, gpu_cmd, proc, uid, killpg_fn, kill_fn, clock, other_counter=None):
        """주입 의존성을 받는다."""
        self.cfg, self.repo_root, self.run_set, self.argv = cfg, repo_root, run_set, list(argv)
        self.child_factory, self.inspect_fn, self.stop, self.poll_s = child_factory, inspect_fn, stop, poll_s
        self.versions, self.git = versions, git
        self.mem_reader = mem_reader or read_mem_available_kib
        self.gpu_cmd = gpu_cmd or GPU_CMD
        self.proc, self.uid = proc or ProcFS(), os.getuid() if uid is None else uid
        self.killpg_fn, self.kill_fn, self.clock = killpg_fn, kill_fn, clock
        self.other_counter = other_counter or (lambda: count_other_sessions(self.proc))
        self.watch = cfg.common.watch
        self.run_set_dir = os.path.join(repo_root, cfg.common.outputs_dir, run_set)
        self.result = RunResult(rc=0, run_set_dir=self.run_set_dir)
        self.lock_ok = False
        self.all_confirmed = True
        self.current = None   # (Popen, token, exp_dir) 실행 중 자식

    # ------------------------------------------------------------------ 시작 전 검사·lock
    def preflight(self) -> bool:
        """이름·차단 상태·lock·fingerprint. 실패하면 rc 2."""
        try:
            validate_run_set_name(self.run_set, self.cfg.common.kind)
        except LayoutError as exc:
            return self.refuse(str(exc))

        blocked = blocking_states(_job_states(self.run_set_dir))
        if blocked:
            return self.refuse(f"blocked by unfinished job states: {blocked}")

        try:
            self.identity = _host_identity(self.proc)
            acquire_lock(self.run_set_dir, {**self.identity, "argv": self.argv,
                                            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
            self.lock_ok = True
        except LayoutError as exc:
            return self.refuse(str(exc), touched=False)

        # lock을 잡은 뒤의 검사 실패·예외는 자기 lock만 풀고 rc 2(F2)
        try:
            if not self.cfg.verify():
                return self.refuse("config snapshot does not match config_id")

            fp = self.cfg.common.data["expected_fingerprint"]
            actual = compute_fingerprint(self.repo_root, self.cfg.common.data["data_yaml"],
                                         list(self.cfg.common.data["split_lists"]))
            self.fp_diffs = [] if fp is None else compare_fingerprint(thaw(fp), actual)
        except Exception as exc:
            return self.refuse(f"preflight failed: {type(exc).__name__}: {exc}")

        if self.fp_diffs:
            return self.refuse(f"data fingerprint mismatch: {self.fp_diffs[:5]}")

        return True

    def refuse(self, message: str, touched: bool = True) -> bool:
        """시작 전 거부(rc 2)."""
        self.result.rc = 2
        self.result.errors.append(message)
        if self.lock_ok and touched:
            release_lock(self.run_set_dir, pid=self.identity["pid"], start_ticks=self.identity["start_ticks"])
            self.lock_ok = False
        return False

    # ------------------------------------------------------------------ 후보 하나
    def child_run(self, *, exp_dir: str, label: str, argv_fn, log_path: str, deadline_s: float, attempt: int,
                  js_extra: dict, before_spawn=None, on_running=None, watch_path: str | None = None) -> dict:
        """자식 하나의 공통 수명(학습·평가 공용): Popen 전 starting·토큰 기록 → 실행 → 감시 → 유한 정리 → 최종 job_state.

        정리 결과·미회수 핸들은 외부 기록보다 먼저 메모리에 반영한다(F3). 돌려주는 dict: rc·stop_reason·cleanup·wall·summary·
        watch_fail·token·pid·start_ticks.
        """
        token = new_job_token()
        js_path = os.path.join(exp_dir, "job_state.json")

        # 확인점 1: Popen 전에 token·starting을 남긴다
        write_json_atomic(js_path, {"state": "starting", "job_token": token, "parent": self.identity,
                                    "attempt": attempt, **js_extra})
        if before_spawn is not None:
            before_spawn()

        argv = argv_fn()
        env = {**os.environ, **thaw(self.cfg.common.required_env), TOKEN_ENV: token}
        log = open(log_path, "ab")
        t0 = self.clock()

        try:
            p = subprocess.Popen(argv, cwd=self.repo_root, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        finally:
            log.close()

        try:
            start_ticks = self.proc.start_ticks(p.pid)
        except (OSError, ValueError, IndexError):
            start_ticks = None
        self.current = (p, token, exp_dir, start_ticks)
        write_json_atomic(js_path, {"state": "running", "job_token": token, "parent": self.identity,
                                    "attempt": attempt, "pid": p.pid, "pgid": p.pid, "start_ticks": start_ticks,
                                    "boot_id": self.identity["boot_id"], "host": self.identity["host"], **js_extra})
        if on_running is not None:
            on_running(p)

        floor = start_ticks if start_ticks is not None else self.identity["start_ticks"]
        stop_reason, summary, watch_fail = self.watch_child(
            p, token, t0, floor, watch_path or os.path.join(exp_dir, "watch.jsonl"), deadline_s)

        cleanup = self.safe_cleanup(p, token, floor)
        # 정리 결과·미회수 핸들을 외부 기록보다 먼저 메모리에 보존한다(F3)
        if not cleanup["confirmed"]:
            self.all_confirmed = False
            if p.poll() is None and p not in self.result.unreaped:
                self.result.unreaped.append(p)
            self.result.errors.append(f"{label} cleanup: confirmed=False remaining={cleanup['remaining']} "
                                      f"unknown={cleanup['unknown']} errors={cleanup['errors']}")
        self.current = None
        wall = round(self.clock() - t0, 3)
        rc = p.returncode

        state = "exited_cleaned" if cleanup["confirmed"] else "cleanup_unconfirmed"
        write_json_atomic(js_path, {"state": state, "job_token": token, "parent": self.identity, "attempt": attempt,
                                    "pid": p.pid, "pgid": p.pid, "start_ticks": start_ticks, "rc": rc,
                                    "cleanup": cleanup, "boot_id": self.identity["boot_id"],
                                    "host": self.identity["host"], **js_extra})
        return {"rc": rc, "stop_reason": stop_reason, "cleanup": cleanup, "wall": wall, "summary": summary,
                "watch_fail": watch_fail, "token": token, "pid": p.pid, "start_ticks": start_ticks}

    def watch_child(self, p, token: str, t0: float, floor, watch_path: str, deadline_s: float):
        """감시 루프: 정지 플래그·메모리·기한 판정, watch_interval_s마다 표본. (stop_reason, 요약, 표본 쓰기 실패)."""
        stop_reason, last_sample, watch_fail = None, -1e9, None
        summary = {"samples": 0, "max_group_rss_kib": None, "min_mem_available_kib": None, "max_gpu_mem_mib": None,
                   "note": "sampled every watch_interval_s (not an absolute peak); group RSS sums processes and may "
                           "double-count shared memory"}

        while p.poll() is None:
            if self.stop.is_set():
                stop_reason = "parent_signal"
                break
            elapsed = self.clock() - t0
            try:
                mem = self.mem_reader()
            except (OSError, ValueError):
                mem = None
            if elapsed - last_sample >= self.watch["watch_interval_s"]:
                last_sample = elapsed
                gpu, gpu_err = gpu_sample(self.gpu_cmd, self.watch["monitor_cmd_timeout_s"])
                rss_failed: list = []
                rss = group_rss_kib(token, self.proc, self.uid, floor, failed=rss_failed)
                summary["rss_read_failures"] = summary.get("rss_read_failures", 0) + len(rss_failed)
                try:
                    others = self.other_counter()
                except Exception:
                    others = None
                summary["samples"] += 1
                summary["max_group_rss_kib"] = max(summary["max_group_rss_kib"] or 0, rss)
                if mem is not None:
                    summary["min_mem_available_kib"] = min(summary["min_mem_available_kib"] or mem, mem)
                if gpu is not None:
                    summary["max_gpu_mem_mib"] = max(summary["max_gpu_mem_mib"] or 0, gpu["mem_used_mib"])
                if watch_fail is None:
                    try:
                        append_jsonl(watch_path, {"elapsed_s": round(elapsed, 2), "mem_available_kib": mem,
                                                  "group_rss_kib": rss, "rss_read_failures": len(rss_failed),
                                                  "other_sessions": others,
                                                  "gpu": gpu, "gpu_error": gpu_err})
                    except (OSError, ValueError) as exc:
                        watch_fail = f"{type(exc).__name__}: {exc}"   # 첫 실패만 남기고 표본 생략
            stop_reason = decide(mem, self.clock() - t0, self.watch, deadline_s)
            if stop_reason:
                break
            time.sleep(self.poll_s)

        return stop_reason, summary, watch_fail

    def run_candidate(self, cand, exp_dir: str, resume_from: str | None = None, prior_attempts: int = 0) -> str:
        """학습 자식 하나를 실행·감시·정리·판정한다. 후보 상태 문자열을 돌려준다(예외는 호출자가 처리)."""
        manifest_path = os.path.join(exp_dir, "run_manifest.json")
        attempt_no = prior_attempts + 1
        manifest = {}

        def before_spawn():
            """시작 manifest·시도 기록(starting 기록 뒤, Popen 전)."""
            if resume_from is None:
                manifest.update(start_manifest(self.cfg, cand, run_set=self.run_set, argv=self.argv,
                                               fingerprint_diffs=self.fp_diffs, git=self.git, versions=self.versions))
            else:
                manifest.update(_read_json(manifest_path) or {})
                manifest["status"] = "starting"
                manifest.setdefault("resume_inputs", []).append({"path": resume_from, "attempt": attempt_no})
            write_json_atomic(manifest_path, manifest)
            append_jsonl(os.path.join(exp_dir, "attempts.jsonl"),
                         {"event": "start", "attempt": attempt_no, "resume_from": resume_from,
                          "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})

        def on_running(p):
            """manifest running."""
            manifest.update(status="running", pid=p.pid, pgid=p.pid)
            write_json_atomic(manifest_path, manifest)

        run = self.child_run(exp_dir=exp_dir, label=cand.exp_id,
                             argv_fn=lambda: self.child_factory(cand, exp_dir, resume_from),
                             log_path=os.path.join(exp_dir, "train.log"), deadline_s=self.watch["train_deadline_s"],
                             attempt=attempt_no, js_extra={}, before_spawn=before_spawn, on_running=on_running)
        rc, stop_reason, cleanup, wall = run["rc"], run["stop_reason"], run["cleanup"], run["wall"]
        summary, watch_fail = run["summary"], run["watch_fail"]

        # 적용 인자 보존(ultralytics args.yaml)
        args_src = os.path.join(exp_dir, "args.yaml")
        if os.path.exists(args_src):
            shutil.copyfile(args_src, os.path.join(exp_dir, "applied_args.yaml"))

        best = self.inspect_fn(os.path.join(exp_dir, "weights", "best.pt"))
        last = self.inspect_fn(os.path.join(exp_dir, "weights", "last.pt"))
        try:
            with open(os.path.join(exp_dir, "results.csv"), encoding="utf-8") as f:
                epochs = parse_epochs(f.read())
        except (OSError, LayoutError):
            epochs = []
        records_ok = all(os.path.exists(os.path.join(exp_dir, n)) for n in ("applied_args.yaml", "applied.json",
                                                                              "progress.json"))
        ok, reasons = check_completion({
            "rc": rc, "cleanup_confirmed": cleanup["confirmed"], "epochs": epochs,
            "target_epochs": self.cfg.train_args()["epochs"], "best": best, "last": last,
            "common": _expected_common(self.cfg), "progress": _read_json(os.path.join(exp_dir, "progress.json")),
            "records_ok": records_ok and watch_fail is None, "legacy": None})

        if stop_reason == "parent_signal":
            status = "stopped"
        elif stop_reason is None and ok:
            status = "completed"
        else:
            status = "failed"

        manifest.update(status=status, rc=rc, stop_reason=stop_reason, cleanup=cleanup, wall_s=wall,
                        watch_summary=summary, best_sha256=best.get("sha256") if best else None,
                        completion_reasons=reasons, watch_write_error=watch_fail,
                        ended_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        write_json_atomic(manifest_path, manifest)
        write_json_atomic(os.path.join(exp_dir, "status.json"), {"status": status, "reasons": reasons,
                                                                 "stop_reason": stop_reason})
        append_jsonl(os.path.join(exp_dir, "attempts.jsonl"),
                     {"event": "end", "attempt": attempt_no, "rc": rc, "status": status, "stop_reason": stop_reason,
                      "cleanup_confirmed": cleanup["confirmed"], "wall_s": wall,
                      "end_last_sha256": last.get("sha256") if last else None})

        return status

    # ------------------------------------------------------------------ 예외 정리
    def safe_cleanup(self, p, token: str, floor) -> dict:
        """terminate_group을 부르되, 그 자체가 예외를 올리면 미확인으로 본다(확인된 것으로 처리하지 않음, F3)."""
        try:
            return terminate_group(p, token, grace_s=self.watch["kill_grace_s"], reap_s=self.watch["reap_timeout_s"],
                                   proc=self.proc, uid=self.uid, killpg_fn=self.killpg_fn, kill_fn=self.kill_fn,
                                   min_start_ticks=floor)
        except Exception as exc:
            return {"leader_reaped": p.poll() is not None, "terminated": False, "killed": False, "remaining": [],
                    "unknown": [], "confirmed": False, "errors": [f"terminate_group raised {type(exc).__name__}: {exc}"]}

    def emergency_cleanup(self, exc: BaseException) -> None:
        """실행 중 자식을 정리하고 원래 오류와 정리 결과를 함께 남긴다."""
        self.result.errors.append(f"{type(exc).__name__}: {exc}")
        if self.current is None:
            return
        p, token, exp_dir, start_ticks = self.current
        cleanup = self.safe_cleanup(p, token, start_ticks if start_ticks is not None else self.identity["start_ticks"])
        self.current = None
        self.result.errors.append(f"cleanup after error: confirmed={cleanup['confirmed']} "
                                  f"remaining={cleanup['remaining']} unknown={cleanup['unknown']} "
                                  f"errors={cleanup['errors']}")
        if not cleanup["confirmed"]:
            self.all_confirmed = False
            if p.poll() is None and p not in self.result.unreaped:
                self.result.unreaped.append(p)
        try:
            write_json_atomic(os.path.join(exp_dir, "job_state.json"),
                              {"state": "exited_cleaned" if cleanup["confirmed"] else "cleanup_unconfirmed",
                               "job_token": token, "parent": self.identity, "pid": p.pid, "cleanup": cleanup,
                               "error": f"{type(exc).__name__}: {exc}"})
        except Exception as exc2:   # 기록 실패도 원래 오류와 함께 보존
            self.result.errors.append(f"job_state write after error failed: {type(exc2).__name__}: {exc2}")
            self.all_confirmed = False

    summary_name = "run_set.json"   # 묶음 요약 파일(평가는 eval_set.json)

    def finish(self, aborted: str | None) -> None:
        """run_set 요약·SHA256SUMS·lock 처리."""
        summary = {"run_set": self.run_set, "config_id": self.cfg.config_id, "kind": self.cfg.common.kind,
                   "argv": self.argv, "git": self.git, "versions": self.versions,
                   "candidates": self.result.candidates, "errors": self.result.errors, "aborted": aborted,
                   "stop_reason": self.result.stop_reason, "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        try:
            write_json_atomic(os.path.join(self.run_set_dir, self.summary_name), summary)
        except Exception as exc:
            self.result.errors.append(f"run_set write failed: {type(exc).__name__}: {exc}")
            self.result.rc = max(self.result.rc, 1)

        if self.all_confirmed:
            try:
                write_sha256sums(self.run_set_dir, tag=time.strftime("%Y%m%dT%H%M%S"))
            except Exception as exc:
                self.result.errors.append(f"SHA256SUMS failed: {type(exc).__name__}: {exc}")
                self.result.rc = max(self.result.rc, 1)
            if self.lock_ok:
                release_lock(self.run_set_dir, pid=self.identity["pid"], start_ticks=self.identity["start_ticks"])
        else:
            # 정리 미확인: lock을 지우지 않고 상태를 덧쓴다
            lock = read_lock(self.run_set_dir) or {}
            lock.update(state="cleanup_unconfirmed")
            try:
                write_json_atomic(os.path.join(self.run_set_dir, "OWNER.lock"), lock)
            except Exception as exc:
                self.result.errors.append(f"lock update failed: {type(exc).__name__}: {exc}")


def _run(batch: _Batch, plan) -> RunResult:
    """plan: [(cand, exp_dir 또는 None(새로 만듦), resume_from, prior_attempts)] 순서대로 실행."""
    if not batch.preflight():
        return batch.result

    aborted = None
    weights_dir = os.path.join(batch.repo_root, batch.cfg.common.weights_dir)
    entries = batch.result.candidates
    for cand, *_ in plan:
        entries.append({"exp_id": cand.exp_id, "family": cand.family, "status": "not_started"})

    try:
        for i, (cand, exp_dir, resume_from, prior) in enumerate(plan):
            if batch.stop.is_set():
                batch.result.stop_reason = "parent_signal"
                aborted = "parent_signal"
                break
            if resume_from is None:
                chk = check_weights(cand, weights_dir)
                if chk.status != "ok":
                    entries[i].update(status="failed_not_started", reason=f"weights {chk.status}")
                    continue
                try:
                    exp_dir = make_exp_dir(batch.run_set_dir, cand.exp_id, cand.family)
                except LayoutError as exc:
                    entries[i].update(status="failed_not_started", reason=str(exc))
                    continue
            status = batch.run_candidate(cand, exp_dir, resume_from, prior)
            entries[i]["status"] = status
            if status == "stopped":
                batch.result.stop_reason = "parent_signal"
                aborted = "parent_signal"
                break
            if not batch.all_confirmed:
                aborted = "cleanup_unconfirmed"
                break
    except BaseException as exc:   # 기록 실패·예상하지 못한 예외·KeyboardInterrupt
        batch.emergency_cleanup(exc)
        aborted = "cleanup_unconfirmed" if not batch.all_confirmed else "error"

    if aborted == "error" or not batch.all_confirmed:
        aborted = "cleanup_unconfirmed" if not batch.all_confirmed else "error"

    batch.finish(aborted)
    done = [e["status"] for e in entries]
    batch.result.rc = max(batch.result.rc, 0 if (aborted is None and all(s == "completed" for s in done)) else 1)

    return batch.result


def run_train(cfg: TrainingConfig, *, repo_root: str, run_set: str, argv: list, child_factory, inspect_fn,
              stop: StopFlag, poll_s: float = 0.5, versions: dict | None = None, git: dict | None = None,
              mem_reader=None, gpu_cmd=None, proc: ProcFS | None = None, uid: int | None = None,
              killpg_fn=os.killpg, kill_fn=os.kill, clock=time.monotonic, other_counter=None) -> RunResult:
    """설정의 후보를 순서대로 새로 학습한다."""
    batch = _Batch(cfg, repo_root, run_set, argv, child_factory, inspect_fn, stop, poll_s, versions or {}, git or {},
                   mem_reader, gpu_cmd, proc, uid, killpg_fn, kill_fn, clock, other_counter)
    return _run(batch, [(c, None, None, 0) for c in cfg.candidates])


def _resume_sha_refusal(path: str, expected: str | None) -> str | None:
    """재개 체크포인트를 열기 전 sha256 대조. 문제가 있으면 사유, 없으면 None(파일 읽기 실패도 사유)."""
    if not expected:
        return "no recorded end_last_sha256 for last.pt in attempts.jsonl"
    try:
        with open(path, "rb") as f:
            got = hashlib.sha256(f.read()).hexdigest()
    except OSError as exc:
        return f"cannot read last.pt: {type(exc).__name__}: {exc}"
    if got != expected:
        return f"last.pt sha256 {got[:12]} != last attempt record {expected[:12]}"
    return None


def run_resume(cfg: TrainingConfig, *, repo_root: str, run_set: str, exp_id: str, argv: list, child_factory,
               inspect_fn, stop: StopFlag, poll_s: float = 0.5, versions: dict | None = None,
               git: dict | None = None, mem_reader=None, gpu_cmd=None, proc: ProcFS | None = None,
               uid: int | None = None, killpg_fn=os.killpg, kill_fn=os.kill, clock=time.monotonic,
               other_counter=None) -> RunResult:
    """한 실험을 그 실험의 weights/last.pt에서 이어 학습한다(설계 r2 §4 재개 (a)~(d))."""
    batch = _Batch(cfg, repo_root, run_set, argv, child_factory, inspect_fn, stop, poll_s, versions or {}, git or {},
                   mem_reader, gpu_cmd, proc, uid, killpg_fn, kill_fn, clock, other_counter)
    cand = next((c for c in cfg.candidates if c.exp_id == exp_id), None)
    if cand is None:
        batch.result.rc = 2
        batch.result.errors.append(f"unknown exp_id {exp_id}")
        return batch.result

    exp_dir = os.path.join(batch.run_set_dir, f"{cand.exp_id}_{cand.family}")
    last_path = os.path.join(exp_dir, "weights", "last.pt")
    attempts = _read_jsonl(os.path.join(exp_dir, "attempts.jsonl"))

    # pickle 기반 체크포인트는 기록된 기대 sha와 대조한 뒤에만 연다(#180 #8, 규약 §10 출처 검증). inspect_fn이 torch.load를 한다
    refused = _resume_sha_refusal(last_path, attempts[-1].get("end_last_sha256") if attempts else None)
    if refused:
        batch.result.rc = 2
        batch.result.errors.append(f"resume refused before loading: {refused}")
        return batch.result

    reasons = check_resume(last=inspect_fn(last_path), target_epochs=cfg.train_args()["epochs"],
                           common=_expected_common(cfg), exp_project=batch.run_set_dir,
                           exp_name=os.path.basename(exp_dir), attempts=attempts,
                           manifest=_read_json(os.path.join(exp_dir, "run_manifest.json")) or {},
                           candidate={"exp_id": cand.exp_id, "family": cand.family, "sha256": cand.sha256},
                           job_state=_read_json(os.path.join(exp_dir, "job_state.json")) or {})
    if reasons:
        batch.result.rc = 2
        batch.result.errors.append(f"resume refused: {reasons}")
        return batch.result

    prior = sum(1 for a in attempts if a.get("event") == "start")
    return _run(batch, [(cand, exp_dir, last_path, prior)])


def status(run_set_dir: str, *, proc: ProcFS | None = None, uid: int | None = None) -> dict:
    """run_set 상태: lock, 부모 생존·stale 후보 여부, 실험별 job_state·자기 작업·에포크 수."""
    proc = proc or ProcFS()
    uid = os.getuid() if uid is None else uid
    lock = read_lock(run_set_dir)
    states = _job_states(run_set_dir)
    out = {"lock": lock, "lock_stale_candidate": bool(lock) and not parent_alive(lock, proc)
           and lock.get("host") == socket.gethostname() and lock.get("boot_id") == proc.boot_id(),
           "job_states": {k: v.get("state") for k, v in states.items()}, "own_processes": {}, "unknown": {},
           "epochs": {}}
    for name, st in states.items():
        if st.get("job_token"):
            own, unknown, _ = settled_processes(st["job_token"], proc, uid, job_min_start(st))
            out["own_processes"][name], out["unknown"][name] = own, unknown
        try:
            with open(os.path.join(run_set_dir, name, "results.csv"), encoding="utf-8") as f:
                out["epochs"][name] = len(parse_epochs(f.read()))
        except (OSError, LayoutError):
            out["epochs"][name] = 0
    return out


def _job_identity_problems(js: dict, host: str, boot: str) -> list[str]:
    """job_state의 유효 호스트·부팅 신원 문제(F1-R). top-level과 parent 양쪽에서 읽어, 서로 모순되거나 어느 쪽에도 없거나
    현재 호스트·부팅과 다르면 문제다. 현재 값으로 빈 신원을 채우지 않는다.
    """
    parent = js.get("parent") if isinstance(js.get("parent"), dict) else {}
    out = []
    for key, current in (("host", host), ("boot_id", boot)):
        seen = {v for v in (js.get(key), parent.get(key)) if v is not None}
        if not seen:
            out.append(f"job {key} unknown (identity unverifiable)")
        elif len(seen) > 1:
            out.append(f"job {key} contradicts between top-level and parent {sorted(seen)}")
        elif any(v != current for v in seen):
            out.append(f"job {key} differs from current (other host/boot)")
    return out


def _recover_preconditions(lock: dict | None, states: dict, proc: ProcFS, host: str, uid: int) -> list[str]:
    """신호 전 복구 적격성(F1). 같은 호스트·부팅, lock 소유자 사망, 실험 신원(토큰·호스트·부팅) 확인, 판별 불가 0.
    남은 자기 작업 프로세스는 거부 사유가 아니다(정리 대상). 사유가 하나라도 있으면 신호·파일 변경을 하지 않는다.
    """
    r = []
    boot = proc.boot_id()

    if lock is not None:
        if lock.get("host") != host:
            r.append(f"lock host {lock.get('host')!r} != {host!r}")
        if lock.get("boot_id") != boot:
            r.append("lock boot_id differs (other boot)")
        if parent_alive(lock, proc):
            r.append(f"lock owner pid {lock.get('pid')} is alive")

    for name, state in blocking_states(states):
        js = states[name]
        if not js.get("job_token"):
            r.append(f"{name} ({state}) has no job_token; identity unknown")
            continue
        r.extend(f"{name} {m}" for m in _job_identity_problems(js, host, boot))
        if js.get("parent") and parent_alive(js["parent"], proc):
            r.append(f"{name} parent pid {js['parent'].get('pid')} is alive")
        unknown = settled_processes(js["job_token"], proc, uid, job_min_start(js))[1]
        if unknown:
            r.append(f"{name} ({state}) has same-uid processes with unreadable environ (unknown) {unknown}")

    return r


def recover(run_set_dir: str, *, kill_own: bool = False, grace_s: float = 5.0, proc: ProcFS | None = None,
            uid: int | None = None, kill_fn=os.kill) -> dict:
    """명시적 복구(설계 r2 §4).

    1) 신호 전 적격성(`_recover_preconditions`): 다른 호스트·다른 부팅·살아 있는 소유자·판별 불가면 신호 0·파일 무변으로 거부.
    2) kill_own이면 토큰으로 확인한 자기 작업에만 TERM→KILL(신호 직전 pid별 재확인).
    3) 파일 갱신·lock 제거 직전에 lock을 다시 읽어 처음과 같은지 확인하고 stale 조건 전체를 재검사한다.
    """
    proc = proc or ProcFS()
    uid = os.getuid() if uid is None else uid
    host = socket.gethostname()
    states = _job_states(run_set_dir)
    lock = read_lock(run_set_dir)
    report = {"signals": [], "errors": [], "lock_released": False, "reasons": []}

    pre = _recover_preconditions(lock, states, proc, host, uid)
    if pre:
        report["reasons"] = pre
        return report

    for name, state in blocking_states(states):
        token = states[name].get("job_token")
        if not kill_own:
            continue
        floor = job_min_start(states[name])
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for pid in own_processes(token, proc, uid, floor)[0]:
                if _pid_has_token(pid, token, proc, uid):
                    try:
                        kill_fn(pid, sig)
                        report["signals"].append((name, pid, signal.Signals(sig).name))
                    except ProcessLookupError:
                        pass
                    except OSError as exc:
                        report["errors"].append(f"{pid}: {type(exc).__name__}: {exc}")
            end = time.monotonic() + grace_s
            while own_processes(token, proc, uid, floor)[0] and time.monotonic() < end:
                time.sleep(0.05)

    # 파일 변경 전 소유권 재확인: lock이 바뀌었으면(다른 실행이 잡음) 손대지 않는다
    if read_lock(run_set_dir) != lock or _job_states(run_set_dir) != states:
        report["reasons"] = ["lock or job_state changed during recover; not modified"]
        return report
    ok, reasons = stale_lock_decision(lock or {"pid": -1, "host": host, "boot_id": proc.boot_id()},
                                      states, proc, host=host, uid=uid)
    report["reasons"] = reasons
    if not ok:
        return report

    for name, state in blocking_states(states):
        path = os.path.join(run_set_dir, name, "job_state.json")
        st = dict(states[name])
        prev = st.get("state")
        st.update(state="exited_cleaned", recovered=True, recovered_from=prev,
                  recovered_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        write_json_atomic(path, st)
        append_jsonl(os.path.join(run_set_dir, name, "attempts.jsonl"),
                     {"event": "recovered", "previous_state": prev, "previous_lock": lock})
    if lock is not None:
        os.remove(os.path.join(run_set_dir, "OWNER.lock"))
    report["lock_released"] = True

    return report


__all__ = ["RunResult", "StopFlag", "decide", "gpu_sample", "read_mem_available_kib", "recover", "run_resume",
           "run_train", "status", "terminate_group"]
