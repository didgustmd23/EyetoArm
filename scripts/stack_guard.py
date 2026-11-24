"""C 실행 묶음(c_stack.launch.py)의 남은 Gazebo 검사와 종료 정리.

두 가지를 한다.

1. 시작 전 검사(find_conflicts): 같은 ROS_DOMAIN_ID(없으면 0)와 같은 GZ_PARTITION(없음과 빈 값은 같게 봄)을 가진
   gz sim 서버나 이 묶음이 띄우는 노드가 이미 있으면 충돌로 본다. 판정은 /proc/<pid>/cmdline·environ만 읽는다.
   environ을 못 읽은 프로세스는 건너뛰고 따로 돌려준다.
2. 종료 정리(watch): launch가 실행마다 만든 표식 환경 변수(MARKER_ENV=<값>)를 자식 프로세스가 물려받는다.
   감시 프로세스는 따로 된 세션에서 launch 프로세스가 끝나기를 기다렸다가, 같은 표식을 가진 프로세스만
   SIGINT → SIGTERM → SIGKILL 순서로 정리한다. 표식이 다른 프로세스(다른 실행·다른 사람의 시험)는 건드리지 않는다.
   ros2 launch는 SIGTERM을 받으면 자식을 정리하지 않고 끝나고, SIGHUP은 처리하지 않는다. 이 두 경우에 남는
   Gazebo·노드를 감시 프로세스가 치운다.

    python3 scripts/stack_guard.py check                       # 지금 환경 기준 충돌 목록(종료 코드 3 = 충돌)
    python3 scripts/stack_guard.py watch --launch-pid P --marker M --log FILE

표준 라이브러리만 쓴다(ros2 launch의 시스템 Python에서 불림).
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

MARKER_ENV = "EYETOARM_STACK_RUN"

# 이 묶음이 띄우는 실행 파일 이름(cmdline 앞쪽 인자의 basename으로 맞춘다)
STACK_EXECUTABLES = frozenset({
    "parameter_bridge", "robot_state_publisher", "controller_recovery", "spawner", "create",
    "move_group", "tcp_server", "display_relay", "sort_goal_receiver",
    "run_pipeline.py", "run_pipeline_supervisor.py",
})
_SH_GZ = re.compile(r"^ruby\s+\S*/gz\s+sim(\s|$)")   # ros_gz_sim launch의 /bin/sh -c "ruby …/gz sim …" 감싸개
_HEAD_ARGS = 3          # 해석기(python3·ruby 등) 뒤의 스크립트까지 보려고 앞 3개 인자를 본다


@dataclass(frozen=True)
class ProcInfo:
    """충돌 후보 프로세스 하나."""

    pid: int
    kind: str
    cmdline: str
    started: dt.datetime | None


@dataclass
class ScanResult:
    """시작 전 검사 결과."""

    conflicts: list[ProcInfo] = field(default_factory=list)
    unreadable: list[tuple[int, str]] = field(default_factory=list)
    others: list[tuple[ProcInfo, str, str]] = field(default_factory=list)   # (프로세스, 도메인, 파티션) — 건드리지 않음


def norm_domain(value: str | None) -> str:
    """ROS_DOMAIN_ID 정규화: 없거나 빈 값은 "0"."""
    v = (value or "").strip()
    return v if v else "0"


def norm_partition(value: str | None) -> str:
    """GZ_PARTITION 정규화: 없음과 빈 값을 같게 본다."""
    return (value or "").strip()


def read_cmdline(proc_root: Path, pid: int) -> list[str]:
    """/proc/<pid>/cmdline 인자 목록(NUL 구분)."""
    raw = (proc_root / str(pid) / "cmdline").read_bytes()
    return [a.decode("utf-8", "replace") for a in raw.split(b"\0") if a]


def read_environ(proc_root: Path, pid: int) -> dict[str, str]:
    """/proc/<pid>/environ 환경 변수(NUL 구분). 읽기 실패는 OSError로 올린다."""
    raw = (proc_root / str(pid) / "environ").read_bytes()
    env = {}

    for item in raw.split(b"\0"):
        if b"=" in item:
            k, v = item.split(b"=", 1)
            env[k.decode("utf-8", "replace")] = v.decode("utf-8", "replace")

    return env


def _is_gz_sim(args: list[str]) -> bool:
    """실행 파일 자리에 gz sim이 있는지(명령 문자열 속 글자만 있는 셸·검색 명령은 아님)."""
    if not args:
        return False
    head = args[0].split()                         # ruby가 $0을 "gz sim -s …" 한 문자열로 바꾼 경우
    if len(head) >= 2 and os.path.basename(head[0]) == "gz" and head[1] == "sim":
        return True
    if os.path.basename(args[0]) == "gz" and args[1:2] == ["sim"]:
        return True
    if os.path.basename(args[0]) == "ruby" and len(args) >= 3 and os.path.basename(args[1]) == "gz" and args[2] == "sim":
        return True
    return (os.path.basename(args[0]) in ("sh", "bash", "dash") and args[1:2] == ["-c"] and len(args) >= 3
            and bool(_SH_GZ.match(args[2])))


def stack_kind(args: list[str]) -> str | None:
    """cmdline이 gz sim 서버나 묶음 노드면 종류 이름, 아니면 None."""
    if _is_gz_sim(args):
        return "gz sim"

    for a in args[:_HEAD_ARGS]:
        # ruby가 $0을 바꾼 경우처럼 한 인자에 공백이 들어 있으면 첫 단어만 본다
        name = os.path.basename(a.split(" ", 1)[0])
        if name in STACK_EXECUTABLES:
            return name

    return None


def ppid_of(proc_root: Path, pid: int) -> int | None:
    """부모 pid(/proc/<pid>/stat 4번째 필드). 읽기 실패면 None."""
    try:
        stat = (proc_root / str(pid) / "stat").read_text()
    except OSError:
        return None
    return int(stat.rsplit(")", 1)[1].split()[1])


def ancestors(proc_root: Path, pid: int) -> set[int]:
    """pid와 그 조상 pid 모음(1까지)."""
    out = set()

    while pid and pid not in out:
        out.add(pid)
        pid = ppid_of(proc_root, pid) or 0

    return out


def start_time(proc_root: Path, pid: int) -> dt.datetime | None:
    """프로세스 시작 시각(지역 시각). 읽기 실패면 None."""
    try:
        ticks = int((proc_root / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19])
        btime = next(int(line.split()[1]) for line in (proc_root / "stat").read_text().splitlines()
                     if line.startswith("btime "))
    except (OSError, StopIteration, ValueError, IndexError):
        return None
    return dt.datetime.fromtimestamp(btime + ticks / os.sysconf("SC_CLK_TCK"))


def iter_pids(proc_root: Path):
    """숫자 이름 디렉터리 = pid."""
    for p in proc_root.iterdir():
        if p.name.isdigit():
            yield int(p.name)


def find_conflicts(domain: str | None, partition: str | None, *, proc_root: Path = Path("/proc"),
                   exclude: set[int] | None = None) -> ScanResult:
    """같은 도메인·파티션의 gz sim 서버·묶음 노드를 찾는다(exclude의 pid는 뺀다)."""
    want = (norm_domain(domain), norm_partition(partition))
    exclude = exclude or set()
    res = ScanResult()

    for pid in sorted(iter_pids(proc_root)):
        if pid in exclude:
            continue
        try:
            args = read_cmdline(proc_root, pid)
        except OSError:
            continue                                  # 이미 끝난 프로세스
        kind = stack_kind(args)
        if kind is None:
            continue

        try:
            env = read_environ(proc_root, pid)
        except OSError as e:
            res.unreadable.append((pid, f"{kind}: {type(e).__name__}"))
            continue

        info = ProcInfo(pid, kind, " ".join(args)[:160], start_time(proc_root, pid))
        have = (norm_domain(env.get("ROS_DOMAIN_ID")), norm_partition(env.get("GZ_PARTITION")))
        if have == want:
            res.conflicts.append(info)
        else:
            res.others.append((info, *have))

    return res


def format_conflicts(res: ScanResult, domain: str | None, partition: str | None) -> str:
    """사람이 읽을 거부 메시지(PID·시작 시각·정리 명령)."""
    d, p = norm_domain(domain), norm_partition(partition) or "(없음)"
    lines = [f"같은 ROS_DOMAIN_ID={d}, GZ_PARTITION={p}에 이미 실행 중인 Gazebo·노드가 있어 시작하지 않습니다:"]

    for c in res.conflicts:
        when = c.started.strftime("%Y-%m-%d %H:%M:%S") if c.started else "알 수 없음"
        lines.append(f"  pid {c.pid}  시작 {when}  [{c.kind}]  {c.cmdline}")

    pids = " ".join(str(c.pid) for c in res.conflicts)
    lines += [
        "남은 것이 자기 실행이면 정리한 뒤 다시 시작하십시오(다른 사람의 실행이면 도메인이나 파티션을 바꾸십시오):",
        f"  kill -INT {pids}; sleep 10; kill -TERM {pids} 2>/dev/null; sleep 5; kill -KILL {pids} 2>/dev/null",
        "  (Gazebo는 SIGTERM으로도 안 끝나는 때가 있어 마지막 kill -KILL까지 필요할 수 있습니다)",
        "c_stack.launch.py는 run_mode:=single(기본)이면 이것을 정리한 뒤 시작하고, run_mode:=multi면 경고만 하고 시작합니다.",
    ]
    if res.unreadable:
        lines.append("읽지 못해 건너뛴 프로세스: " + ", ".join(f"{pid}({why})" for pid, why in res.unreadable))

    return "\n".join(lines)


# ---------------------------------------------------------------- 종료 정리

def marked_pids(marker: str, *, proc_root: Path = Path("/proc"), exclude: set[int] | None = None) -> list[int]:
    """표식 환경 변수 값이 marker인 프로세스 pid(읽기 실패는 건너뜀)."""
    exclude = exclude or set()
    out = []

    for pid in sorted(iter_pids(proc_root)):
        if pid in exclude:
            continue
        try:
            env = read_environ(proc_root, pid)
        except OSError:
            continue
        if env.get(MARKER_ENV) == marker:
            out.append(pid)

    return out


def cleanup(find, send, sleep, *, steps=((signal.SIGINT, 10.0), (signal.SIGTERM, 5.0), (signal.SIGKILL, 2.0)),
            poll_s: float = 0.25, log=lambda msg: None) -> list[int]:
    """find()가 돌려주는 pid가 없어질 때까지 신호를 단계별로 보낸다. 끝까지 남은 pid를 돌려준다.

    find: () -> list[int], send: (pid, sig) -> None(ProcessLookupError는 무시), sleep: (s) -> None.
    """
    for sig, wait_s in steps:
        pids = find()
        if not pids:
            return []
        log(f"send {signal.Signals(sig).name} to {pids}")

        for pid in pids:
            try:
                send(pid, sig)
            except ProcessLookupError:
                pass
            except PermissionError as e:
                log(f"cannot signal {pid}: {e}")

        waited = 0.0
        while waited < wait_s and find():
            sleep(poll_s)
            waited += poll_s

    left = find()
    if left:
        log(f"still alive after SIGKILL: {left}")
    return left


def start_ticks(proc_root: Path, pid: int) -> int | None:
    """프로세스 시작 틱(/proc/<pid>/stat 22번째 필드, 부팅 뒤 경과). 읽기 실패면 None.

    pid 재사용 구별에는 이 값을 쓴다. 시각(start_time)은 /proc/stat btime에 기대는데, WSL은 시계 보정 때
    btime이 바뀌어 같은 프로세스의 시작 시각이 달라 보일 수 있다(#357 시험에서 살아 있는 launch를 끝난 것으로 오판).
    """
    try:
        return int((proc_root / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def process_alive(pid: int, ticks: int | None, *, proc_root: Path = Path("/proc")) -> bool:
    """pid가 살아 있고 시작 틱이 같다(pid 재사용 구별). 좀비는 끝난 것으로 본다."""
    try:
        stat = (proc_root / str(pid) / "stat").read_text()
    except OSError:
        return False
    fields = stat.rsplit(")", 1)[1].split()
    if fields[0] in ("Z", "X"):
        return False
    return ticks is None or int(fields[19]) == ticks


# ---------------------------------------------------------------- 기본 포트 주인

DEFAULT_PORTS = (47090, 47092, 47093)            # tcp_server, display_relay(영상·제어)


def listening_inodes(proc_root: Path, ports) -> dict[int, int]:
    """LISTEN 상태 TCP 소켓 inode → 포트(/proc/net/tcp·tcp6, ports에 든 것만)."""
    want, out = set(ports), {}

    for name in ("tcp", "tcp6"):
        try:
            lines = (proc_root / "net" / name).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            f = line.split()
            if len(f) < 10 or f[3] != "0A":
                continue
            port = int(f[1].rsplit(":", 1)[1], 16)
            if port in want:
                out[int(f[9])] = port

    return out


def port_owners(ports=DEFAULT_PORTS, *, proc_root: Path = Path("/proc")) -> list[tuple[int, int | None, str]]:
    """ports를 LISTEN 중인 (포트, pid 또는 None(주인 못 찾음), 명령) 목록."""
    inodes = listening_inodes(proc_root, ports)
    found = {}

    for pid in iter_pids(proc_root):
        if len(found) == len(inodes):
            break
        try:
            fds = list((proc_root / str(pid) / "fd").iterdir())
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if target.startswith("socket:[") and int(target[8:-1]) in inodes:
                found.setdefault(int(target[8:-1]), pid)

    out = []
    for inode, port in sorted(inodes.items(), key=lambda kv: kv[1]):
        pid = found.get(inode)
        cmd = ""
        if pid is not None:
            try:
                cmd = " ".join(read_cmdline(proc_root, pid))[:160]
            except OSError:
                pass
        out.append((port, pid, cmd))

    return out


# ---------------------------------------------------------------- 시작 전 처리(run_mode)

RUN_MODES = ("single", "multi")


def describe_others(res: ScanResult) -> str | None:
    """다른 도메인·파티션 묶음 프로세스를 건드리지 않는다는 한 줄(없으면 None)."""
    if not res.others:
        return None
    items = ", ".join(f"{p.pid}[{p.kind}] 도메인 {d}·파티션 {pt or '(없음)'}" for p, d, pt in res.others)
    return f"다른 도메인·파티션의 Gazebo·노드 {len(res.others)}개는 다른 실행이라 건드리지 않습니다: {items}"


def prestart(mode: str, domain: str | None, partition: str | None, *, proc_root: Path = Path("/proc"),
             exclude: set[int] | None = None, send=os.kill, sleep=time.sleep, ports=DEFAULT_PORTS,
             steps=((signal.SIGINT, 10.0), (signal.SIGTERM, 5.0), (signal.SIGKILL, 2.0)),
             log=lambda msg: None) -> tuple[bool, list[str]]:
    """run_mode에 따른 시작 전 처리. (시작해도 되는지, 화면·기록에 남길 줄들).

    single: 같은 도메인·파티션의 gz sim·묶음 노드를 SIGINT → SIGTERM → SIGKILL로 정리한 뒤, 기본 포트를
            다른 프로세스가 잡고 있으면(죽이지 않고) 거부한다. 끝까지 안 죽은 프로세스가 있어도 거부한다.
    multi:  중복과 포트 점유를 경고만 하고 시작한다.
    """
    lines = []
    if mode not in RUN_MODES:
        msg = f"run_mode는 {'·'.join(RUN_MODES)} 중 하나여야 합니다(받은 값 {mode!r})."
        log(msg)
        return False, [msg]

    def say(msg):
        """화면용 줄 모음과 기록에 함께 남긴다."""
        lines.append(msg)
        log(msg)

    res = find_conflicts(domain, partition, proc_root=proc_root, exclude=exclude)
    other = describe_others(res)
    if other:
        say(other)
    if res.unreadable:
        say("읽지 못해 건너뛴 프로세스: " + ", ".join(f"{pid}({why})" for pid, why in res.unreadable))

    d, p = norm_domain(domain), norm_partition(partition) or "(없음)"
    if res.conflicts:
        listing = [f"  pid {c.pid} [{c.kind}] {c.cmdline}" for c in res.conflicts]
        if mode == "multi":
            say(f"경고(run_mode:=multi): 같은 도메인 {d}·파티션 {p}에 이미 Gazebo·노드 {len(res.conflicts)}개가 있지만 "
                "정리하지 않고 시작합니다. 같은 토픽·시계가 섞일 수 있습니다.")
            for line in listing:
                say(line)
        else:
            say(f"run_mode:=single: 같은 도메인 {d}·파티션 {p}의 남은 Gazebo·노드 {len(res.conflicts)}개를 정리합니다 "
                "(SIGINT → 10초 → SIGTERM → 5초 → SIGKILL).")
            for line in listing:
                say(line)
            targets = [(c.pid, start_ticks(proc_root, c.pid)) for c in res.conflicts]
            find = lambda: [pid for pid, t in targets if process_alive(pid, t, proc_root=proc_root)]   # noqa: E731
            left = cleanup(find, send, sleep, steps=steps, log=say)
            if left:
                say(f"정리 실패: SIGKILL 뒤에도 남은 pid {left} — 시작하지 않습니다.")
                return False, lines
            say("정리 완료: 남은 것 없음.")

    held = port_owners(ports, proc_root=proc_root)
    if held:
        desc = [f"  포트 {port}: pid {pid if pid is not None else '알 수 없음'} {cmd}" for port, pid, cmd in held]
        if mode == "multi":
            say("경고(run_mode:=multi): 기본 포트를 이미 다른 프로세스가 쓰고 있습니다(이 실행의 서버는 열리지 않을 수 있음):")
            for line in desc:
                say(line)
        else:
            say("기본 포트를 다른 프로세스가 쓰고 있어 시작하지 않습니다(다른 도메인 시험일 수 있어 죽이지 않음):")
            for line in desc:
                say(line)
            return False, lines

    return True, lines


def watch(launch_pid: int, marker: str, log_path: str | None, *, poll_s: float = 0.5) -> int:
    """launch 프로세스가 끝나면 같은 표식 프로세스를 정리한다(따로 된 세션에서 실행)."""
    me = os.getpid()
    logf = open(log_path, "a", encoding="utf-8") if log_path else None

    def log(msg):
        """시각을 붙여 기록한다."""
        if logf:
            logf.write(f"[{dt.datetime.now().isoformat(timespec='milliseconds')}] {msg}\n")
            logf.flush()

    for s in (signal.SIGINT, signal.SIGHUP):
        signal.signal(s, signal.SIG_IGN)               # 터미널 Ctrl+C·창 닫기로 감시가 먼저 죽지 않게(SIGTERM으로는 끌 수 있음)
    ticks = start_ticks(Path("/proc"), launch_pid)
    log(f"watch launch pid {launch_pid} (start ticks {ticks}, started {start_time(Path('/proc'), launch_pid)}) marker {marker}")

    while process_alive(launch_pid, ticks):
        time.sleep(poll_s)

    log("launch ended")
    find = lambda: marked_pids(marker, exclude={me})   # noqa: E731
    left = cleanup(find, os.kill, time.sleep, log=log)
    log(f"done, left {left}")
    return 0 if not left else 1


def main(argv=None) -> int:
    """명령 진입점."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    w = sub.add_parser("watch")
    w.add_argument("--launch-pid", type=int, required=True)
    w.add_argument("--marker", required=True)
    w.add_argument("--log")
    args = ap.parse_args(argv)

    if args.cmd == "check":
        domain, partition = os.environ.get("ROS_DOMAIN_ID"), os.environ.get("GZ_PARTITION")
        res = find_conflicts(domain, partition, exclude=ancestors(Path("/proc"), os.getpid()))
        if res.conflicts:
            print(format_conflicts(res, domain, partition))
            return 3
        print("충돌 없음" + (f" (읽지 못한 프로세스 {len(res.unreadable)}개)" if res.unreadable else ""))
        return 0

    return watch(args.launch_pid, args.marker, args.log)


if __name__ == "__main__":
    sys.exit(main())
