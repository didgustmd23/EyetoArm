"""작업 신원·자기 작업 판별·새 작업 허용·stale lock 해제 판정(학습기 설계 r2 §4, 검토 r2 확인점 1·2). 신호는 보내지 않는다.

- 자식을 띄울 때마다 `job_token`(uuid4 hex)을 환경 변수 `EYETOARM_TRAIN_JOB`로 준다. 손자(데이터로더 worker)도 물려받는다.
- `own_processes`: **자기 uid 프로세스만** 조사한다. 자기 자식은 같은 uid로 만들어지므로, 다른 uid의 프로세스는(environ을 읽지
  못해도) 자기 작업일 수 없어 제외한다. 자기 uid인데 environ을 읽지 못하면 「판별 불가」로 따로 센다(복구를 거부할 근거).
  pid·pgid 숫자만으로 판단하지 않는다(PID 재사용 방지). 조사 중 사라진 프로세스는 건너뛴다.
- 새 작업은 lock 유무와 무관하게 `running`·`starting`·`cleanup_unconfirmed` 실험이 있으면 막힌다.
- stale lock 해제는 같은 호스트·부팅 ID, lock 부모 사망(pid 없음 또는 시작 시각 불일치), 막힌 실험의 자기 작업 0·판별 불가 0일 때만.
"""
from __future__ import annotations

import os
import time
import uuid

TOKEN_ENV = "EYETOARM_TRAIN_JOB"
BLOCKING = ("running", "starting", "cleanup_unconfirmed")


def new_job_token() -> str:
    """새 작업 토큰."""
    return uuid.uuid4().hex


class ProcFS:
    """`/proc` 읽기(루트 주입 가능, 시험은 가짜 트리)."""

    def __init__(self, root: str = "/proc") -> None:
        """루트 경로."""
        self.root = root

    def _read(self, pid: int, name: str) -> bytes:
        """`/proc/<pid>/<name>` 바이트(파일을 닫는다)."""
        with open(os.path.join(self.root, str(pid), name), "rb") as f:
            return f.read()

    def pids(self) -> list[int]:
        """숫자 이름의 항목."""
        return sorted(int(n) for n in os.listdir(self.root) if n.isdigit())

    def _stat_fields(self, pid: int) -> list[str]:
        """stat의 `)` 뒤 필드(3번째 필드부터)."""
        text = self._read(pid, "stat").decode(errors="replace")
        return text[text.rindex(")") + 2:].split()

    def start_ticks(self, pid: int) -> int:
        """시작 시각(부팅 뒤 clock ticks, stat 22번째 필드)."""
        return int(self._stat_fields(pid)[19])

    def state(self, pid: int) -> str:
        """프로세스 상태 문자(stat 3번째 필드, Z = 좀비)."""
        return self._stat_fields(pid)[0]

    def rss_kib(self, pid: int) -> int:
        """상주 메모리(status의 VmRSS, kB). 없으면 0(커널 스레드·좀비)."""
        for line in self._read(pid, "status").decode(errors="replace").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
        return 0

    def cmdline(self, pid: int) -> str:
        """명령줄(NUL을 공백으로)."""
        return self._read(pid, "cmdline").replace(b"\0", b" ").decode(errors="replace").strip()

    def pgid(self, pid: int) -> int:
        """프로세스 그룹(stat 5번째 필드)."""
        return int(self._stat_fields(pid)[2])

    def uid(self, pid: int) -> int:
        """실제 uid(status의 Uid 첫 값)."""
        for line in self._read(pid, "status").decode(errors="replace").splitlines():
            if line.startswith("Uid:"):
                return int(line.split()[1])
        raise ValueError(f"no Uid line for {pid}")

    def environ(self, pid: int) -> dict:
        """환경 변수(읽기 권한이 없으면 PermissionError)."""
        data = self._read(pid, "environ")
        out = {}
        for item in data.split(b"\0"):
            if b"=" in item:
                k, v = item.split(b"=", 1)
                out[k.decode(errors="replace")] = v.decode(errors="replace")
        return out

    def boot_id(self) -> str:
        """부팅 ID."""
        with open(os.path.join(self.root, "sys", "kernel", "random", "boot_id"), encoding="utf-8") as f:
            return f.read().strip()


def own_processes(token: str, proc: ProcFS, uid: int,
                  min_start_ticks: int | None = None) -> tuple[list[int], list[int]]:
    """(토큰을 가진 자기 uid 프로세스, 자기 uid인데 environ을 읽지 못한 프로세스).

    `min_start_ticks`(작업 자식의 시작 시각, 모르면 부모 시작 시각)보다 **먼저 시작한** 프로세스는 그 작업의 자손일 수 없으므로
    조사하지 않는다. 같은 uid라도 environ을 읽을 수 없는 오래된 프로세스(예: `systemd --user`, `sd-pam`, 실측 확인)가 판별 불가로
    남아 정상 정리·복구를 영구히 막는 것을 피하는 근거다. 작업 시작 뒤에 생긴 읽기 불가 프로세스는 그대로 판별 불가다.
    """
    own, unknown = _scan(token, proc, uid, min_start_ticks)
    return own, sorted(unknown)


def _vanished_or_zombie(pid: int, proc: ProcFS) -> bool:
    """읽기 실패 직후 그 pid가 사라졌거나 Z·X인가(P1: 실제로 확인된 해소만 참)."""
    try:
        return proc.state(pid) in ("Z", "X")
    except (FileNotFoundError, ProcessLookupError):
        return True
    except (ValueError, IndexError, OSError):
        return False


def _scan(token: str, proc: ProcFS, uid: int, min_start_ticks: int | None) -> tuple[list[int], dict]:
    """/proc 1회 조사. (토큰 자기 프로세스, {판별 불가 pid: 시작 시각 또는 None})."""
    own, unknown = [], {}

    for pid in proc.pids():
        start = None
        try:
            if proc.uid(pid) != uid:
                continue   # 다른 uid: 자기 자식일 수 없음(조사 범위 밖)
            start = proc.start_ticks(pid)
            if min_start_ticks is not None and start < min_start_ticks:
                continue   # 작업보다 먼저 시작: 그 작업의 자손일 수 없음
            if proc.state(pid) in ("Z", "X"):
                continue   # 좀비·종료: 실행 중이 아니며 자원을 쥐지 않음(부모 회수 대기)
            env = proc.environ(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue       # 조사 중 사라짐
        except (ValueError, IndexError, OSError):
            # 읽기 거부·파싱 실패: 직후 사라짐·Z가 확인되면 종료 전이(U1 P1), 아니면 판별 불가(fail-closed)
            if not _vanished_or_zombie(pid, proc):
                unknown[pid] = start
            continue
        if env.get(TOKEN_ENV) == token:
            own.append(pid)

    return own, unknown


def _resolve(pid: int, start, token: str, proc: ProcFS, uid: int) -> str | None:
    """이전 회차의 판별 불가 (pid, 시작 시각)을 지금 명시적으로 확인한다. 해소 사유 또는 None(여전히 미확인).

    gone(사라짐)·reused(같은 pid가 다른 시작 시각 = 원래 프로세스는 끝남)·zombie·other_uid·own(토큰 판독)·not_own(정상 판독).
    """
    try:
        if start is not None and proc.start_ticks(pid) != start:
            return "reused"
        if proc.uid(pid) != uid:
            return "other_uid"
        if proc.state(pid) in ("Z", "X"):
            return "zombie"
        if start is None:
            return None
        return "own" if proc.environ(pid).get(TOKEN_ENV) == token else "not_own"
    except (FileNotFoundError, ProcessLookupError):
        return "gone"
    except (ValueError, IndexError, OSError):
        return None


def settled_processes(token: str, proc: ProcFS, uid: int, min_start_ticks: int | None = None, *,
                      retries: int = 3, gap_s: float = 0.1, sleep=time.sleep, clock=time.monotonic,
                      deadline: float | None = None, initial: tuple | None = None) -> tuple[list[int], list[int], dict]:
    """상한 있는 재조사(U1 P2). (own, unknown, info{scans, slept_s, resolved}).

    - 첫 조사에 판별 불가가 없으면 그대로 끝(추가 비용 0).
    - 판별 불가가 있으면 `gap_s` 쉬고 전체를 다시 조사한다. 최대 `retries`회(임시값 3회·0.1 s, 대기 합 ≤ 0.2 s)이고,
      `deadline`(상위 정리 기한, clock 기준)을 넘길 대기는 하지 않는다.
    - **최종 판단은 마지막 조사 기준이다.** 마지막 조사의 unknown·own은 그대로 남긴다(교집합 금지: 새로 나타난 unknown도 거부 사유).
    - 이전 회차에서만 보인 판별 불가 (pid, 시작 시각)은 `_resolve`로 명시적으로 확인될 때만 해소한다. 결과에서 빠졌다는
      사실만으로는 해소하지 않는다. 토큰으로 판독되면 own에 넣는다.
    - `initial`(호출자가 방금 한 `_scan` 결과)이 있으면 그것을 첫 조사로 쓴다(조사를 버리지 않는다).
    """
    own, unknown = initial if initial is not None else _scan(token, proc, uid, min_start_ticks)
    own, unknown = list(own), dict(unknown)
    seen = set(unknown.items())
    scans, slept = 1, 0.0

    while unknown and scans < retries:
        if deadline is not None and clock() + gap_s > deadline:
            break   # 상위 기한 안에 대기할 여유가 없으면 지금 결과로 판단(미확인 유지)
        sleep(gap_s)
        slept += gap_s
        own, unknown = _scan(token, proc, uid, min_start_ticks)
        seen |= set(unknown.items())
        scans += 1

    final_unknown, final_own, resolved = set(unknown), set(own), {}
    for pid, start in seen - set(unknown.items()):
        if pid in final_unknown or pid in final_own:
            continue   # 같은 pid(다른 시작 시각 포함)는 마지막 조사 판정이 우선
        why = _resolve(pid, start, token, proc, uid)
        if why is None:
            final_unknown.add(pid)
        else:
            resolved[f"{pid}@{start}"] = why
            if why == "own":
                final_own.add(pid)

    return sorted(final_own), sorted(final_unknown), {"scans": scans, "slept_s": round(slept, 3),
                                                      "resolved": resolved}


def job_min_start(job_state: dict) -> int | None:
    """job_state에서 조사 하한 시작 시각: 자식 start_ticks, 없으면 부모 start_ticks, 둘 다 없으면 None(전체 조사)."""
    if isinstance(job_state.get("start_ticks"), int):
        return job_state["start_ticks"]
    parent = job_state.get("parent") or {}
    return parent.get("start_ticks") if isinstance(parent.get("start_ticks"), int) else None


def blocking_states(job_states: dict) -> list[tuple[str, str]]:
    """새 작업을 막는 (실험, 상태) 목록."""
    return [(k, v.get("state")) for k, v in sorted(job_states.items()) if v.get("state") in BLOCKING]


def parent_alive(lock: dict, proc: ProcFS) -> bool:
    """lock 부모가 살아 있는가(pid 있고 시작 시각 같음)."""
    try:
        return proc.start_ticks(int(lock["pid"])) == lock.get("start_ticks")
    except (FileNotFoundError, ProcessLookupError, ValueError, KeyError):
        return False


def stale_lock_decision(lock: dict, job_states: dict, proc: ProcFS, *, host: str, uid: int) -> tuple[bool, list[str]]:
    """stale lock 해제 허용 여부와 거부 이유."""
    r = []

    if lock.get("host") != host:
        r.append(f"lock host {lock.get('host')!r} != {host!r}")
    if lock.get("boot_id") != proc.boot_id():
        r.append("lock boot_id differs (other boot)")
    if parent_alive(lock, proc):
        r.append(f"lock owner pid {lock.get('pid')} is alive")

    for exp, state in blocking_states(job_states):
        token = job_states[exp].get("job_token")
        if not token:
            r.append(f"{exp} ({state}) has no job_token; identity unknown")
            continue
        own, unknown, _ = settled_processes(token, proc, uid, job_min_start(job_states[exp]))
        if own:
            r.append(f"{exp} ({state}) still has own processes {own}")
        if unknown:
            r.append(f"{exp} ({state}) has same-uid processes with unreadable environ (unknown) {unknown}")

    return (not r), r


__all__ = ["BLOCKING", "ProcFS", "TOKEN_ENV", "blocking_states", "job_min_start", "new_job_token", "own_processes", "parent_alive", "settled_processes",
           "stale_lock_decision"]
