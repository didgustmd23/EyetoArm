"""대시보드 확인용 한 번에 실행: Gazebo + C TCP 서버 + C 영상 중계기 (+ Windows D 대시보드, + A 실시간 YOLO).

가장 쉬운 실행 (어느 폴더에서나):
    ~/EyetoArm/scripts/dashboard_up.sh                 # ROS source 까지 해 줌
    ~/EyetoArm/scripts/dashboard_up.sh yolo:=true      # A YOLO 검출 영상(detect)도 같이 (.venv-a 필요)

직접 실행:
    source /opt/ros/jazzy/setup.bash && source ~/EyetoArm/robot_ws/install/setup.bash
    ros2 launch ~/EyetoArm/scripts/dashboard_stack.launch.py [world:=...] [dashboard:=false] [yolo:=true]

순서는 docs/design/서버-대시보드 실행.md 와 같다: TCP 서버 → Gazebo → (지연) 중계기 → 대시보드 → YOLO.
Ctrl+C 한 번으로 WSL 쪽은 모두 꺼진다.

남은 Gazebo 처리(c_stack.launch.py와 같음, scripts/stack_guard.py):
    run_mode:=single(기본)  같은 ROS_DOMAIN_ID·GZ_PARTITION의 남은 Gazebo·노드를 SIGINT → 10초 → SIGTERM → 5초 → SIGKILL로
                           정리한 뒤 시작. 기본 포트(47090·47092·47093)를 다른 프로세스가 쓰면 죽이지 않고 시작하지 않음
    run_mode:=multi         경고만 하고 시작
    끝날 때(Ctrl+C·kill·창 닫기)는 이 실행이 띄운 WSL 프로세스만 정리한다. Windows 대시보드 창은 정리하지 못한다.

Windows 대시보드 폴더(dashboard-client)는 자동으로 찾는다:
  1) 환경 변수 EYETOARM_DASHBOARD_DIR (예: 'C:\\...\\dashboard-client') 가 있으면 그것
  2) 지난번 찾은 폴더(~/.cache/eyetoarm/dashboard_dir) 가 아직 쓸 수 있으면 그것
  3) 없으면 /mnt/<드라이브> 를 5단계 깊이까지 찾음 (.venv/Scripts/Activate.ps1 + src/eyetoarm/gui/window.py 가 둘 다 있는 폴더,
     여럿이면 window.py 가 가장 최근인 것). 다시 찾게 하려면: rm ~/.cache/eyetoarm/dashboard_dir
"""
import importlib.util
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, LogInfo, OpaqueFunction,
                            SetEnvironmentVariable, TimerAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

REPO = Path(__file__).resolve().parent.parent
SIM_LAUNCH = REPO / "scripts" / "sim_sort_cell.launch.py"
PS = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
DASH_RUN = ("; .\\.venv\\Scripts\\Activate.ps1; $env:PYTHONPATH=(Resolve-Path .\\src).Path; "
            "python -m eyetoarm.gui.window --host 127.0.0.1 --port 47090 "
            "--view relay --view-host 127.0.0.1 --view-port 47092")
CACHE = Path.home() / ".cache" / "eyetoarm" / "dashboard_dir"
SKIP = {"Windows", "Program Files", "Program Files (x86)", "ProgramData", "AppData", "$Recycle.Bin",
        "System Volume Information", "node_modules", ".git", ".venv", "anaconda3",
        "Application Data", "Local Settings", "My Documents"}
YOLO_PY = REPO / ".venv-a" / "bin" / "python"
YOLO_CONFIG = "config/vision/ai_s1_08_axis.json"
GUARD_SCRIPT = REPO / "scripts" / "stack_guard.py"
_spec = importlib.util.spec_from_file_location("eyetoarm_stack_guard", GUARD_SCRIPT)
stack_guard = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = stack_guard          # dataclass가 모듈을 찾을 수 있게
_spec.loader.exec_module(stack_guard)


def _guard_and_watch(context, marker, stamp):
    """시작 전 처리(run_mode)와 종료 정리 감시 프로세스 실행(c_stack.launch.py `_guard_and_watch`와 같은 동작).

    run_mode:=single(기본): 같은 ROS_DOMAIN_ID·GZ_PARTITION의 남은 gz sim·묶음 노드를 SIGINT → SIGTERM → SIGKILL로
    정리한 뒤 시작한다. 기본 포트(47090·47092·47093)를 다른 프로세스가 잡고 있거나 정리가 끝까지 안 되면
    예외로 launch를 멈춘다(뒤의 Gazebo·노드는 하나도 띄우지 않음). 다른 도메인·파티션은 건드리지 않는다.
    run_mode:=multi: 중복·포트 점유를 경고만 하고 시작한다.
    감시 프로세스는 따로 된 세션에서 이 launch가 끝나기를 기다렸다가 같은 표식의 프로세스만 정리한다
    (SIGTERM·SIGHUP으로 launch만 끝나 Gazebo·노드가 남는 경우 대비). sim_sort_cell.launch.py에는 넣지 않는다
    (이 런처가 포함하므로 검사가 겹친다).
    """
    mode = LaunchConfiguration("run_mode").perform(context).strip().lower()
    domain, partition = os.environ.get("ROS_DOMAIN_ID"), os.environ.get("GZ_PARTITION")
    log = REPO / "logs" / f"dashboard_stack_guard_{stamp}.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    with open(log, "a", encoding="utf-8") as f:
        def record(msg):
            """화면(표준 오류)과 기록 파일에 함께 남긴다."""
            line = f"[{datetime.now().isoformat(timespec='milliseconds')}] prestart: {msg}"
            print(f"[dashboard_stack guard] {msg}", file=sys.stderr, flush=True)
            f.write(line + "\n")
            f.flush()

        record(f"run_mode={mode} ROS_DOMAIN_ID={stack_guard.norm_domain(domain)} "
               f"GZ_PARTITION={stack_guard.norm_partition(partition) or '(없음)'}")
        ok, _ = stack_guard.prestart(mode, domain, partition,
                                     exclude=stack_guard.ancestors(Path("/proc"), os.getpid()), log=record)
        if not ok:
            record("시작하지 않음")
            raise RuntimeError(f"dashboard_stack 시작 전 확인 실패(run_mode={mode}) — 위 메시지와 {log} 참고")

    subprocess.Popen([sys.executable, str(GUARD_SCRIPT), "watch", "--launch-pid", str(os.getpid()),
                      "--marker", marker, "--log", str(log)],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True, close_fds=True)
    return [LogInfo(msg=f"종료 정리 감시 시작(표식 {marker}, 기록 {log})")]


def _usable(d: Path) -> bool:
    """실행 가능한 D 폴더: 가상환경 + 대시보드 코드가 둘 다 있어야 함."""
    return (d / ".venv/Scripts/Activate.ps1").is_file() and (d / "src/eyetoarm/gui/window.py").is_file()


def _search(max_depth: int = 5) -> list:
    found = []
    for drive in sorted(Path("/mnt").glob("?")):                      # /mnt/c, /mnt/d ...
        for root, dirs, _ in os.walk(drive):
            depth = len(Path(root).relative_to(drive).parts)
            dirs[:] = [] if depth >= max_depth else [x for x in dirs if x not in SKIP]
            if "dashboard-client" in dirs and _usable(Path(root) / "dashboard-client"):
                found.append(Path(root) / "dashboard-client")
    return sorted(found, key=lambda d: (d / "src/eyetoarm/gui/window.py").stat().st_mtime, reverse=True)


def _win(p: Path) -> str:
    return subprocess.run(["wslpath", "-w", str(p)], capture_output=True, text=True, check=True).stdout.strip()


def find_dashboard_dir():
    """(Windows 주소 또는 None, 안내 문장)."""
    env = os.environ.get("EYETOARM_DASHBOARD_DIR")
    if env:
        return env, f"EYETOARM_DASHBOARD_DIR 사용: {env}"
    try:
        cached = Path(CACHE.read_text(encoding="utf-8").strip())
        if _usable(cached):
            return _win(cached), f"지난번 찾은 폴더 사용: {_win(cached)}"
    except OSError:
        pass
    found = _search()
    if not found:
        return None, ("dashboard-client(.venv 포함)를 못 찾음 → 대시보드 생략. "
                      "export EYETOARM_DASHBOARD_DIR='C:\\...\\dashboard-client' 로 지정하세요")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(str(found[0]), encoding="utf-8")
    others = "".join(f" | 다른 후보: {_win(d)}" for d in found[1:])
    return _win(found[0]), f"자동으로 찾음: {_win(found[0])}{others}"


def _later(context, *_):
    """중계기 뒤에 켤 것들: Windows 대시보드(자동 폴더 찾기) + 선택 YOLO."""
    acts = []
    if LaunchConfiguration("dashboard").perform(context) == "true":
        d, msg = find_dashboard_dir()
        acts.append(LogInfo(msg=f"[대시보드] {msg}"))
        if d:
            acts.append(ExecuteProcess(cmd=[PS, "-NoExit", "-Command", f"cd '{d}'{DASH_RUN}"], output="screen"))

    if LaunchConfiguration("yolo").perform(context) == "true":
        if not YOLO_PY.is_file():
            acts.append(LogInfo(msg=f"[YOLO] {YOLO_PY} 없음 → 생략 (python3 -m venv --system-site-packages .venv-a 후 "
                                    "pip install -r requirements.txt)"))
        else:
            log = REPO / "outputs" / "live" / f"detect_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
            log.parent.mkdir(parents=True, exist_ok=True)
            path = os.pathsep.join(p for p in (str(REPO / "src"), os.environ.get("PYTHONPATH", "")) if p)
            acts.append(LogInfo(msg=f"[YOLO] 기록: {log}"))
            acts.append(ExecuteProcess(
                cmd=[str(YOLO_PY), "scripts/run_pipeline.py", "--ros-link", "--detector", "ai",
                     "--ai-config", LaunchConfiguration("yolo_config").perform(context), "--log", str(log)],
                cwd=str(REPO), output="screen",
                additional_env={"PYTHONPATH": path, "YOLO_AUTOINSTALL": "false", "YOLO_OFFLINE": "1"}))
    return acts


def generate_launch_description():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    marker = uuid.uuid4().hex

    return LaunchDescription([
        DeclareLaunchArgument("world", default_value=str(REPO / "simulation" / "worlds" / "sort_cell.sdf")),
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("dashboard", default_value="true", description="Windows 대시보드도 띄울지"),
        DeclareLaunchArgument("yolo", default_value="false", description="A 실시간 YOLO 검출(detect 영상)도 켤지"),
        DeclareLaunchArgument("yolo_config", default_value=YOLO_CONFIG, description="A AI 설정 JSON"),
        DeclareLaunchArgument("relay_delay_s", default_value="8.0", description="Gazebo 카메라가 뜰 때까지 기다림"),
        DeclareLaunchArgument("run_mode", default_value="single",
                              description="single(기본): 같은 ROS_DOMAIN_ID·GZ_PARTITION의 남은 Gazebo·노드를 정리 후 시작, "
                                          "기본 포트를 남이 쓰면 거부 / multi: 경고만 하고 시작"),

        # 0) 이 실행이 띄우는 모든 프로세스에 표식을 물려주고, 시작 전 검사·종료 정리 감시를 가장 먼저 한다.
        SetEnvironmentVariable(stack_guard.MARKER_ENV, marker),
        OpaqueFunction(function=lambda context: _guard_and_watch(context, marker, stamp)),

        # 1) C TCP 서버 (저장소 루트에서 실행 → config/gui/client.json 사용)
        # Node 대신 ros2 run: Node는 '--ros-args'를 붙이는데 tcp_server·display_relay의 argparse가 이를 거부함
        ExecuteProcess(cmd=["ros2", "run", "robot_server_package", "tcp_server"], cwd=str(REPO), output="screen"),

        # 2) Gazebo (DDS 대용량 영상 설정은 sim_sort_cell 이 모든 노드에 넣어 줌)
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(SIM_LAUNCH)),
                                 launch_arguments={"world": LaunchConfiguration("world"),
                                                   "gui": LaunchConfiguration("gui")}.items()),

        # 3) 영상 중계기 → 4) Windows 대시보드 · 5) YOLO: 카메라 토픽이 생긴 뒤에 켠다
        TimerAction(period=LaunchConfiguration("relay_delay_s"), actions=[
            ExecuteProcess(cmd=["ros2", "run", "robot_server_package", "display_relay"], cwd=str(REPO),
                           output="screen"),
            OpaqueFunction(function=_later),
        ]),
    ])
