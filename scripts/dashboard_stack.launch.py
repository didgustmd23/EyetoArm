"""대시보드 확인용 한 번에 실행: Gazebo + C TCP 서버 + C 영상 중계기 (+ Windows D 대시보드, + A 실시간 YOLO).

가장 쉬운 실행 (어느 폴더에서나):
    ~/EyetoArm/scripts/dashboard_up.sh                 # ROS source 까지 해 줌
    ~/EyetoArm/scripts/dashboard_up.sh yolo:=true      # A YOLO 검출 영상(detect)도 같이 (.venv-a 필요)

직접 실행:
    source /opt/ros/jazzy/setup.bash && source ~/EyetoArm/robot_ws/install/setup.bash
    ros2 launch ~/EyetoArm/scripts/dashboard_stack.launch.py [world:=...] [dashboard:=false] [yolo:=true]

순서는 docs/design/서버-대시보드 실행.md 와 같다: TCP 서버 → Gazebo → (지연) 중계기 → 대시보드 → YOLO.
Ctrl+C 한 번으로 WSL 쪽은 모두 꺼진다.

Windows 대시보드 폴더(dashboard-client)는 자동으로 찾는다:
  1) 환경 변수 EYETOARM_DASHBOARD_DIR (예: 'C:\\...\\dashboard-client') 가 있으면 그것
  2) 지난번 찾은 폴더(~/.cache/eyetoarm/dashboard_dir) 가 아직 쓸 수 있으면 그것
  3) 없으면 /mnt/<드라이브> 를 5단계 깊이까지 찾음 (.venv/Scripts/Activate.ps1 + src/eyetoarm/gui/window.py 가 둘 다 있는 폴더,
     여럿이면 window.py 가 가장 최근인 것). 다시 찾게 하려면: rm ~/.cache/eyetoarm/dashboard_dir
"""
import os
import subprocess
import time
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, LogInfo, OpaqueFunction,
                            TimerAction)
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
    return LaunchDescription([
        DeclareLaunchArgument("world", default_value=str(REPO / "simulation" / "worlds" / "sort_cell.sdf")),
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("dashboard", default_value="true", description="Windows 대시보드도 띄울지"),
        DeclareLaunchArgument("yolo", default_value="false", description="A 실시간 YOLO 검출(detect 영상)도 켤지"),
        DeclareLaunchArgument("yolo_config", default_value=YOLO_CONFIG, description="A AI 설정 JSON"),
        DeclareLaunchArgument("relay_delay_s", default_value="8.0", description="Gazebo 카메라가 뜰 때까지 기다림"),

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
