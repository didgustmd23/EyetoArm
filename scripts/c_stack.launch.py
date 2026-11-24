"""C stack launch: Gazebo, TCP server, video relay, and optional live A/B vision.

B is integrated into run_pipeline.py rather than started as a separate process.
start_a:=true start_b:=true runs A input, B list processing, and CenterOverlay publishing.
Set start_b:=false to run the live A path without the B consumer.
"""

from datetime import datetime
import importlib.util
import os
import subprocess
import sys
import uuid
from pathlib import Path

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    ExecuteProcess,
    GroupAction,
    LogInfo,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

ROOT = Path(__file__).resolve().parent.parent
SIM_LAUNCH = ROOT / "scripts" / "sim_sort_cell.launch.py"
# M2 live A input feeds the B list consumer; B publishes center overlays from this same process.
AI_CONFIG = ROOT / "config" / "vision" / "ai_s1_08_axis_m2_viewbias.json"
LIVE_CONFIG = ROOT / "config" / "vision" / "live_observation_test.json"
GUARD_SCRIPT = ROOT / "scripts" / "stack_guard.py"
_spec = importlib.util.spec_from_file_location("eyetoarm_stack_guard", GUARD_SCRIPT)
stack_guard = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = stack_guard          # dataclass가 모듈을 찾을 수 있게
_spec.loader.exec_module(stack_guard)


def _guard_and_watch(context, marker, stamp):
    """시작 전 처리(run_mode)와 종료 정리 감시 프로세스 실행.

    run_mode:=single(기본): 같은 ROS_DOMAIN_ID·GZ_PARTITION의 남은 gz sim·묶음 노드를 SIGINT → SIGTERM → SIGKILL로
    정리한 뒤 시작한다. 기본 포트(47090·47092·47093)를 다른 프로세스가 잡고 있거나 정리가 끝까지 안 되면
    예외로 launch를 멈춘다(뒤의 Gazebo·노드는 하나도 띄우지 않음). 다른 도메인·파티션은 건드리지 않는다.
    run_mode:=multi: 중복·포트 점유를 경고만 하고 시작한다.
    감시 프로세스는 따로 된 세션에서 이 launch가 끝나기를 기다렸다가 같은 표식의 프로세스만 정리한다
    (SIGTERM·SIGHUP으로 launch만 끝나 Gazebo·노드가 남는 경우 대비, scripts/stack_guard.py).
    """
    mode = LaunchConfiguration("run_mode").perform(context).strip().lower()
    domain, partition = os.environ.get("ROS_DOMAIN_ID"), os.environ.get("GZ_PARTITION")
    log = ROOT / "logs" / f"c_stack_guard_{stamp}.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    with open(log, "a", encoding="utf-8") as f:
        def record(msg):
            """화면(표준 오류)과 기록 파일에 함께 남긴다."""
            line = f"[{datetime.now().isoformat(timespec='milliseconds')}] prestart: {msg}"
            print(f"[c_stack guard] {msg}", file=sys.stderr, flush=True)
            f.write(line + "\n")
            f.flush()

        record(f"run_mode={mode} ROS_DOMAIN_ID={stack_guard.norm_domain(domain)} "
               f"GZ_PARTITION={stack_guard.norm_partition(partition) or '(없음)'}")
        ok, _ = stack_guard.prestart(mode, domain, partition,
                                     exclude=stack_guard.ancestors(Path("/proc"), os.getpid()), log=record)
        if not ok:
            record("시작하지 않음")
            raise RuntimeError(f"c_stack 시작 전 확인 실패(run_mode={mode}) — 위 메시지와 {log} 참고")

    subprocess.Popen([sys.executable, str(GUARD_SCRIPT), "watch", "--launch-pid", str(os.getpid()),
                      "--marker", marker, "--log", str(log)],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True, close_fds=True)
    return [LogInfo(msg=f"종료 정리 감시 시작(표식 {marker}, 기록 {log})")]


def _live_pipeline_actions(context, root, ai_config, live_config, ai_log, stamp):
    start_a = LaunchConfiguration("start_a").perform(context).lower() == "true"
    start_b = LaunchConfiguration("start_b").perform(context).lower() == "true"
    sorting = LaunchConfiguration('auto_sort').perform(context).lower() == 'true'
    if sorting and not (start_a and start_b):
        raise ValueError('auto_sort requires start_a and start_b')
    if sorting:
        live_config = root / 'config/vision/live_observation_sort_part_managed_test.json'
    if not start_a:
        if start_b:
            return [LogInfo(msg="start_b requires start_a:=true; B consumes live A candidates.")]
        return []

    python = LaunchConfiguration("a_python").perform(context)
    cmd = [
        python, str(root / "scripts" / "run_pipeline_supervisor.py"), "--",
        python, str(root / "scripts" / "run_pipeline.py"),
        "--ros-link", "--detector", "ai",
        "--ai-config", str(ai_config),
        "--live-observation-out", str(root / "logs" / f"a_live_obs_{stamp}"),
        "--live-observation-config", str(live_config),
    ]
    if start_b:
        cmd.extend(["--b-input", "list", "--b-center-overlay"])
    if sorting:
        cmd.extend(["--live-sort-part", "--tracking", "--tracking-space", "world",
                    "--world-tracker-config", str(root / "config/vision/world_tracker_gripper_test.json"),
                    "--track-stream", str(root / "config/vision/track_stream_test.json"), "--gripper-handoff"])
    cmd.extend(["--log", str(ai_log)])
    return [ExecuteProcess(cmd=cmd, cwd=str(root), output="screen")]


def generate_launch_description():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    marker = uuid.uuid4().hex
    ai_log = ROOT / "logs" / f"a_live_display_{stamp}.jsonl"
    gui = LaunchConfiguration("gui")
    start_moveit = LaunchConfiguration("start_moveit")
    moveit_gui = LaunchConfiguration("moveit_gui")
    move_group_launch = PathJoinSubstitution([
        FindPackageShare("ur5e_moveit_config"), "launch", "move_group.launch.py",
    ])
    moveit_rviz_launch = PathJoinSubstitution([
        FindPackageShare("ur5e_moveit_config"), "launch", "moveit_rviz.launch.py",
    ])
    c_executables = Path(get_package_prefix("robot_server_package")) / "lib" / "robot_server_package"
    python_site = Path("lib") / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    ros_python = Path(get_package_prefix("rclpy")) / python_site
    interface_python = Path(get_package_prefix("eyetoarm_interfaces")) / python_site

    return LaunchDescription([
        DeclareLaunchArgument("auto_sort", default_value="true"),
        DeclareLaunchArgument("gui", default_value="false",
                              description="Gazebo GUI 실행 여부 (false면 headless)"),
        DeclareLaunchArgument("start_a", default_value="true",
                              description="A의 AI 검출·C 영상 handoff 실행 여부"),
        DeclareLaunchArgument("start_b", default_value="true",
                              description="Enable B list processing and center publisher inside the A pipeline (requires start_a=true)"),
        DeclareLaunchArgument("start_moveit", default_value="true",
                              description="MoveGroup 실행 여부"),
        DeclareLaunchArgument("moveit_gui", default_value="false",
                              description="MoveIt RViz GUI 실행 여부 (start_moveit:=true일 때만 적용)"),
        DeclareLaunchArgument("a_python", default_value=str(ROOT / ".venv" / "bin" / "python"),
                              description="A 파이프라인에 사용할 Python 실행 파일"),
        DeclareLaunchArgument("run_mode", default_value="single",
                              description="single(기본): 같은 ROS_DOMAIN_ID·GZ_PARTITION의 남은 Gazebo·노드를 정리 후 시작, "
                                          "기본 포트를 남이 쓰면 거부 / multi: 경고만 하고 시작"),

        # 이 실행이 띄우는 모든 프로세스에 표식을 물려주고, 시작 전 검사·종료 정리 감시를 가장 먼저 한다.
        SetEnvironmentVariable(stack_guard.MARKER_ENV, marker),
        OpaqueFunction(function=lambda context: _guard_and_watch(context, marker, stamp)),

        # A 파이프라인도 ROS 이미지 토픽을 읽으므로 시뮬레이션과 같은 DDS 설정을 사용한다.
        SetEnvironmentVariable("ROS_AUTOMATIC_DISCOVERY_RANGE", "SUBNET"),
        SetEnvironmentVariable("FASTRTPS_DEFAULT_PROFILES_FILE",
                              str(ROOT / "config" / "dds" / "fastdds_large_image.xml")),
        AppendEnvironmentVariable("PYTHONPATH", str(ros_python)),
        AppendEnvironmentVariable("PYTHONPATH", str(interface_python)),
        AppendEnvironmentVariable("PYTHONPATH", str(ROOT / "src")),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(SIM_LAUNCH)),
            launch_arguments={"gui": gui}.items(),
        ),

        GroupAction(actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(move_group_launch),
            ),
            GroupAction(actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(moveit_rviz_launch),
                ),
            ], condition=IfCondition(moveit_gui)),
        ], condition=IfCondition(start_moveit)),

        # 이 두 entry point는 rclpy Node가 아니라 argparse 기반 프로세스이므로
        # launch_ros Node가 붙이는 --ros-args를 전달하지 않는다.
        ExecuteProcess(cmd=[str(c_executables / "sort_goal_receiver"),
                            "--gazebo-project-root", str(ROOT), "--moving-belt",
                            "--pose-source", "track", "--feed-parts", "--run-state",
                            str(ROOT / "logs" / "sort_run.sqlite3")],
                       condition=IfCondition(LaunchConfiguration("auto_sort")), output="screen"),
        ExecuteProcess(cmd=[str(c_executables / "tcp_server")], output="screen"),
        ExecuteProcess(cmd=[str(c_executables / "display_relay")], output="screen"),
        OpaqueFunction(function=lambda context: _live_pipeline_actions(
            context, ROOT, AI_CONFIG, LIVE_CONFIG, ai_log, stamp
        )),
    ])
