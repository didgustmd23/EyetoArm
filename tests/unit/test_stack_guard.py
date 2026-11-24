"""scripts/stack_guard.py 단위시험: 가짜 /proc 디렉터리로 남은 Gazebo 판정과 종료 정리 순서를 확인한다."""
import importlib.util
import shutil
import signal
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("stack_guard_under_test", _ROOT / "scripts" / "stack_guard.py")
sg = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = sg                   # dataclass가 모듈을 찾을 수 있게
_SPEC.loader.exec_module(sg)

GZ = ["gz sim -s -r --physics-engine gz-physics-dartsim-plugin /w/sort_cell.sdf"]
SH_GZ = ["/bin/sh", "-c", "ruby /opt/ros/jazzy/opt/gz_tools_vendor/bin/gz sim -s -r /w/sort_cell.sdf --force-version 8"]
BRIDGE = ["/opt/ros/jazzy/lib/ros_gz_bridge/parameter_bridge", "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"]
TCP = ["/usr/bin/python3", "/x/install/robot_server_package/lib/robot_server_package/tcp_server"]


def make_proc(tmp_path, procs, btime=1_700_000_000, listen=None):
    """가짜 /proc: procs = {pid: dict(cmd=[...], env={...} 또는 None(읽기 실패), ppid=, ticks=, sockets=[inode])},
    listen = {inode: port}(LISTEN 소켓)."""
    root = tmp_path / "proc"
    root.mkdir()
    (root / "stat").write_text(f"cpu  1 2 3\nbtime {btime}\n")

    for pid, p in procs.items():
        d = root / str(pid)
        d.mkdir()
        (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in p["cmd"]) + b"\0")
        fields = ["S", str(p.get("ppid", 1))] + ["0"] * 17 + [str(p.get("ticks", 100))]
        (d / "stat").write_text(f"{pid} (x y) " + " ".join(fields) + "\n")

        if p.get("sockets"):
            (d / "fd").mkdir()
            for i, inode in enumerate(p["sockets"]):
                (d / "fd" / str(i + 3)).symlink_to(f"socket:[{inode}]")

        if p.get("env") is None:
            (d / "environ").mkdir()                   # 읽으면 IsADirectoryError(OSError) = 읽기 실패
        else:
            (d / "environ").write_bytes(b"\0".join(f"{k}={v}".encode() for k, v in p["env"].items()) + b"\0")

    net = root / "net"
    net.mkdir()
    rows = ["  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode"]
    for i, (inode, port) in enumerate((listen or {}).items()):
        rows.append(f"   {i}: 00000000:{port:04X} 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 {inode} 1")
    rows.append(f"   9: 00000000:{47090:04X} 0100007F:1234 01 00000000:00000000 00:00000000 00000000  1000        0 999 1")
    (net / "tcp").write_text("\n".join(rows) + "\n")
    return root


def test_same_domain_and_partition_is_conflict(tmp_path):
    """같은 도메인·파티션의 gz sim(직접·sh 감싸개)과 묶음 노드는 모두 충돌이다."""
    env = {"ROS_DOMAIN_ID": "57", "GZ_PARTITION": "p1"}
    root = make_proc(tmp_path, {10: {"cmd": GZ, "env": env}, 11: {"cmd": SH_GZ, "env": env},
                                12: {"cmd": BRIDGE, "env": env}, 13: {"cmd": TCP, "env": env}})

    res = sg.find_conflicts("57", "p1", proc_root=root)

    assert [(c.pid, c.kind) for c in res.conflicts] == [(10, "gz sim"), (11, "gz sim"), (12, "parameter_bridge"),
                                                         (13, "tcp_server")]
    assert res.unreadable == []


def test_other_domain_or_partition_is_not_conflict(tmp_path):
    """도메인이나 파티션 중 하나라도 다르면 충돌이 아니다."""
    root = make_proc(tmp_path, {20: {"cmd": GZ, "env": {"ROS_DOMAIN_ID": "58", "GZ_PARTITION": "p1"}},
                                21: {"cmd": GZ, "env": {"ROS_DOMAIN_ID": "57", "GZ_PARTITION": "p2"}},
                                22: {"cmd": GZ, "env": {"ROS_DOMAIN_ID": "57"}}})

    res = sg.find_conflicts("57", "p1", proc_root=root)

    assert res.conflicts == []


def test_unset_domain_is_zero_and_unset_partition_is_empty(tmp_path):
    """ROS_DOMAIN_ID 없음 = "0", GZ_PARTITION 없음 = 빈 값으로 본다."""
    root = make_proc(tmp_path, {30: {"cmd": GZ, "env": {"ROS_DOMAIN_ID": "0", "GZ_PARTITION": ""}},
                                31: {"cmd": GZ, "env": {}}})

    res = sg.find_conflicts(None, None, proc_root=root)

    assert [c.pid for c in res.conflicts] == [30, 31]


def test_unreadable_environ_is_skipped_and_recorded(tmp_path):
    """environ을 못 읽은 묶음 프로세스는 충돌로 세지 않고 따로 기록한다."""
    root = make_proc(tmp_path, {40: {"cmd": GZ, "env": None}, 41: {"cmd": ["bash"], "env": None}})

    res = sg.find_conflicts(None, None, proc_root=root)

    assert res.conflicts == []
    assert [pid for pid, _ in res.unreadable] == [40]          # 묶음과 무관한 bash는 기록도 안 함


def test_unrelated_and_text_only_processes_are_ignored(tmp_path):
    """명령 문자열에 "gz sim" 글자만 있는 셸·검색 명령, 관계없는 프로세스는 판정에서 뺀다."""
    env = {}
    root = make_proc(tmp_path, {50: {"cmd": ["bash", "-c", "pgrep -af 'gz sim'"], "env": env},
                                51: {"cmd": ["grep", "gz sim", "log.txt"], "env": env},
                                52: {"cmd": ["/usr/bin/python3", "train.py"], "env": env}})

    res = sg.find_conflicts(None, None, proc_root=root)

    assert res.conflicts == []


def test_excluded_pids_are_skipped(tmp_path):
    """exclude에 든 pid(자기·조상)는 보지 않는다."""
    root = make_proc(tmp_path, {60: {"cmd": GZ, "env": {}}, 61: {"cmd": BRIDGE, "env": {}}})

    res = sg.find_conflicts(None, None, proc_root=root, exclude={60})

    assert [c.pid for c in res.conflicts] == [61]


def test_ancestors_walks_parent_chain(tmp_path):
    """ancestors는 자기와 부모들을 1까지 모은다."""
    root = make_proc(tmp_path, {1: {"cmd": ["init"], "env": {}, "ppid": 0},
                                70: {"cmd": ["bash"], "env": {}, "ppid": 1},
                                71: {"cmd": ["ros2"], "env": {}, "ppid": 70}})

    assert sg.ancestors(root, 71) == {71, 70, 1}


def test_start_time_and_message(tmp_path):
    """시작 시각은 btime + 틱/CLK_TCK이고, 메시지에 pid·시각·정리 명령·우회 인자가 들어간다."""
    root = make_proc(tmp_path, {80: {"cmd": GZ, "env": {"ROS_DOMAIN_ID": "5"}, "ticks": 0}}, btime=1_700_000_000)

    res = sg.find_conflicts("5", None, proc_root=root)
    msg = sg.format_conflicts(res, "5", None)

    assert res.conflicts[0].started.timestamp() == pytest.approx(1_700_000_000)
    assert "pid 80" in msg and "run_mode:=multi" in msg
    assert msg.index("kill -INT 80") < msg.index("kill -TERM 80") < msg.index("kill -KILL 80")   # 단계적 정리 안내
    assert "ROS_DOMAIN_ID=5" in msg and "GZ_PARTITION=(없음)" in msg


def test_marked_pids_selects_only_same_marker(tmp_path):
    """표식 값이 같은 프로세스만 고르고, 다른 표식·표식 없음·읽기 실패·exclude는 뺀다."""
    root = make_proc(tmp_path, {90: {"cmd": GZ, "env": {sg.MARKER_ENV: "m1"}},
                                91: {"cmd": BRIDGE, "env": {sg.MARKER_ENV: "m2"}},
                                92: {"cmd": TCP, "env": {}},
                                93: {"cmd": TCP, "env": None},
                                94: {"cmd": ["python3", "stack_guard.py"], "env": {sg.MARKER_ENV: "m1"}}})

    assert sg.marked_pids("m1", proc_root=root, exclude={94}) == [90]


def test_cleanup_escalates_until_gone():
    """SIGINT에 안 끝나는 프로세스만 SIGTERM·SIGKILL로 넘어가고, 다 끝나면 빈 목록을 돌려준다."""
    alive = {1, 2}
    sent = []

    def send(pid, sig):
        """SIGINT엔 1만, SIGKILL엔 2도 끝난다고 흉내 낸다."""
        sent.append((pid, sig))
        if sig == signal.SIGINT and pid == 1 or sig == signal.SIGKILL:
            alive.discard(pid)

    left = sg.cleanup(lambda: sorted(alive), send, lambda s: None,
                      steps=((signal.SIGINT, 1.0), (signal.SIGTERM, 1.0), (signal.SIGKILL, 1.0)))

    assert left == []
    assert sent == [(1, signal.SIGINT), (2, signal.SIGINT), (2, signal.SIGTERM), (2, signal.SIGKILL)]


def test_cleanup_ignores_already_gone_and_reports_survivors():
    """이미 끝난 pid(ProcessLookupError)는 무시하고, 끝까지 남은 pid를 돌려준다."""
    def send(pid, sig):
        """pid 3은 이미 없고, pid 4는 어떤 신호에도 안 끝난다."""
        if pid == 3:
            raise ProcessLookupError

    left = sg.cleanup(lambda: [3, 4], send, lambda s: None, steps=((signal.SIGINT, 0.5),))

    assert left == [3, 4]


def test_cleanup_with_nothing_sends_nothing():
    """정리할 프로세스가 없으면 신호를 하나도 보내지 않는다."""
    sent = []

    left = sg.cleanup(lambda: [], lambda pid, sig: sent.append(pid), lambda s: None)

    assert left == [] and sent == []


def test_process_alive_uses_start_ticks_not_boot_time(tmp_path):
    """살아 있음 판정은 시작 틱으로 한다: btime(시계 보정)이 바뀌어도 살아 있고, 틱이 다르면(pid 재사용) 끝난 것이다."""
    root = make_proc(tmp_path, {100: {"cmd": ["ros2", "launch"], "env": {}, "ticks": 555}}, btime=1_700_000_000)
    ticks = sg.start_ticks(root, 100)

    (root / "stat").write_text("cpu  1 2 3\nbtime 1700000002\n")      # WSL 시계 보정으로 btime이 바뀐 상황

    assert ticks == 555
    assert sg.process_alive(100, ticks, proc_root=root)
    assert not sg.process_alive(100, 556, proc_root=root)
    assert not sg.process_alive(101, ticks, proc_root=root)


def test_process_alive_treats_zombie_as_ended(tmp_path):
    """좀비(Z) 상태의 launch는 끝난 것으로 본다."""
    root = make_proc(tmp_path, {110: {"cmd": ["ros2", "launch"], "env": {}, "ticks": 7}})
    stat = (root / "110" / "stat").read_text().replace(") S ", ") Z ")
    (root / "110" / "stat").write_text(stat)

    assert not sg.process_alive(110, 7, proc_root=root)


def killer(root, survive_until=None):
    """가짜 kill: 받은 신호가 그 pid의 버티는 단계 이상이면 /proc 디렉터리를 지운다(기본은 SIGINT에 끝남)."""
    survive_until = survive_until or {}
    order = {signal.SIGINT: 0, signal.SIGTERM: 1, signal.SIGKILL: 2}
    sent = []

    def send(pid, sig):
        """신호를 기록하고, 끝날 신호면 프로세스를 없앤다."""
        sent.append((pid, sig))
        if order[sig] >= order.get(survive_until.get(pid), 0) + (1 if pid in survive_until else 0):
            shutil.rmtree(root / str(pid), ignore_errors=True)

    return send, sent


FAST = ((signal.SIGINT, 0.5), (signal.SIGTERM, 0.5), (signal.SIGKILL, 0.5))


def test_port_owners_maps_listen_socket_to_pid(tmp_path):
    """LISTEN 소켓 inode를 fd에서 찾아 포트 주인 pid를 돌려주고, 주인을 못 찾으면 None이다. LISTEN이 아닌 줄은 뺀다."""
    root = make_proc(tmp_path, {200: {"cmd": ["python3", "srv.py"], "env": {}, "sockets": [5001]}},
                     listen={5001: 47090, 5002: 47093, 5003: 8080})

    owners = sg.port_owners(proc_root=root)

    assert owners == [(47090, 200, "python3 srv.py"), (47093, None, "")]


def test_single_cleans_same_domain_including_term_resistant(tmp_path):
    """single: 같은 도메인·파티션의 gz sim과 SIGTERM에도 버티는 노드를 SIGKILL까지 정리하고, 다른 도메인은 건드리지 않고 시작한다."""
    env = {"ROS_DOMAIN_ID": "57", "GZ_PARTITION": "p"}
    root = make_proc(tmp_path, {300: {"cmd": GZ, "env": env}, 301: {"cmd": BRIDGE, "env": env},
                                302: {"cmd": GZ, "env": {"ROS_DOMAIN_ID": "58", "GZ_PARTITION": "p"}}})
    send, sent = killer(root, survive_until={301: signal.SIGTERM})

    ok, lines = sg.prestart("single", "57", "p", proc_root=root, send=send, sleep=lambda s: None, steps=FAST)

    assert ok
    assert [(pid, sig) for pid, sig in sent] == [(300, signal.SIGINT), (301, signal.SIGINT), (301, signal.SIGTERM),
                                                 (301, signal.SIGKILL)]
    assert (root / "302").exists()
    assert any("건드리지 않습니다" in line and "302" in line for line in lines)
    assert lines[-1] == "정리 완료: 남은 것 없음."


def test_single_refuses_when_port_held_by_other(tmp_path):
    """single: 기본 포트를 다른 프로세스가 쓰면 그 프로세스를 죽이지 않고 거부한다."""
    root = make_proc(tmp_path, {400: {"cmd": ["python3", "other_server.py"], "env": {"ROS_DOMAIN_ID": "101"},
                                      "sockets": [7001]}}, listen={7001: 47092})
    send, sent = killer(root)

    ok, lines = sg.prestart("single", "57", "p", proc_root=root, send=send, sleep=lambda s: None, steps=FAST)

    assert not ok and sent == []
    assert any("포트 47092: pid 400" in line for line in lines)


def test_single_refuses_when_cleanup_fails(tmp_path):
    """single: SIGKILL 뒤에도 남는 프로세스가 있으면 시작하지 않는다."""
    root = make_proc(tmp_path, {500: {"cmd": GZ, "env": {}}})

    ok, lines = sg.prestart("single", None, None, proc_root=root, send=lambda pid, sig: None,
                            sleep=lambda s: None, steps=FAST)

    assert not ok
    assert "정리 실패" in lines[-1]


def test_multi_only_warns(tmp_path):
    """multi: 같은 도메인 중복과 포트 점유가 있어도 신호를 보내지 않고 경고만 한 뒤 시작한다."""
    root = make_proc(tmp_path, {600: {"cmd": GZ, "env": {}, "sockets": [8001]}}, listen={8001: 47090})
    send, sent = killer(root)

    ok, lines = sg.prestart("multi", None, None, proc_root=root, send=send, sleep=lambda s: None, steps=FAST)

    assert ok and sent == []
    assert any(line.startswith("경고(run_mode:=multi): 같은 도메인") for line in lines)
    assert any(line.startswith("경고(run_mode:=multi): 기본 포트") for line in lines)


def test_invalid_run_mode_is_refused(tmp_path):
    """run_mode 값이 single·multi가 아니면 시작하지 않는다."""
    root = make_proc(tmp_path, {})

    ok, lines = sg.prestart("both", None, None, proc_root=root)

    assert not ok and "run_mode" in lines[0]


def test_single_without_duplicates_starts_quietly(tmp_path):
    """single: 중복·포트 점유가 없으면 신호 없이 바로 시작한다(기존 동작과 같음)."""
    root = make_proc(tmp_path, {700: {"cmd": ["bash"], "env": {}}})
    send, sent = killer(root)

    ok, lines = sg.prestart("single", None, None, proc_root=root, send=send, sleep=lambda s: None, steps=FAST)

    assert ok and sent == [] and lines == []
