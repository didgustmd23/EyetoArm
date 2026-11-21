"""C stack launch: Gazebo, TCP server, video relay, and optional live A/B vision.

B is integrated into run_pipeline.py rather than started as a separate process.
start_a:=true start_b:=true runs A input, B list processing, and CenterOverlay publishing.
Set start_b:=false to run the live A path without the B consumer.
"""

from datetime import datetime
import sys
from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
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
from launch.substitutions import LaunchConfiguration

ROOT = Path(__file__).resolve().parent.parent
SIM_LAUNCH = ROOT / "scripts" / "sim_sort_cell.launch.py"
# M2 live A input feeds the B list consumer; B publishes center overlays from this same process.
AI_CONFIG = ROOT / "config" / "vision" / "ai_s1_08_axis_m2_viewbias.json"
LIVE_CONFIG = ROOT / "config" / "vision" / "live_observation_test.json"


def _live_pipeline_actions(context, root, ai_config, live_config, ai_log, stamp):
    start_a = LaunchConfiguration("start_a").perform(context).lower() == "true"
    start_b = LaunchConfiguration("start_b").perform(context).lower() == "true"
    if not start_a:
        if start_b:
            return [LogInfo(msg="start_b requires start_a:=true; B consumes live A candidates.")]
        return []

    python = LaunchConfiguration("a_python").perform(context)
    cmd = [
        python, str(root / "scripts" / "run_pipeline.py"),
        "--ros-link", "--detector", "ai",
        "--ai-config", str(ai_config),
        "--live-observation-out", str(root / "logs" / f"a_live_obs_{stamp}"),
        "--live-observation-config", str(live_config),
    ]
    if start_b:
        cmd.extend(["--b-input", "list", "--b-center-overlay"])
    cmd.extend(["--log", str(ai_log)])
    return [ExecuteProcess(cmd=cmd, cwd=str(root), output="screen")]


def generate_launch_description():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ai_log = ROOT / "logs" / f"a_live_display_{stamp}.jsonl"
    gui = LaunchConfiguration("gui")
    start_moveit = LaunchConfiguration("start_moveit")
    moveit_share = Path(get_package_share_directory("ur5e_moveit_config"))
    c_executables = Path(get_package_prefix("robot_server_package")) / "lib" / "robot_server_package"
    python_site = Path("lib") / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    ros_python = Path(get_package_prefix("rclpy")) / python_site
    interface_python = Path(get_package_prefix("eyetoarm_interfaces")) / python_site

    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="false",
                              description="Gazebo GUI 실행 여부 (false면 headless)"),
        DeclareLaunchArgument("start_a", default_value="true",
                              description="A의 AI 검출·C 영상 handoff 실행 여부"),
        DeclareLaunchArgument("start_b", default_value="true",
                              description="Enable B list processing and center publisher inside the A pipeline (requires start_a=true)"),
        DeclareLaunchArgument("start_moveit", default_value="false",
                              description="MoveGroup과 MoveIt RViz 실행 여부"),
        DeclareLaunchArgument("a_python", default_value=str(ROOT / ".venv" / "bin" / "python"),
                              description="A 파이프라인에 사용할 Python 실행 파일"),

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
                PythonLaunchDescriptionSource(str(moveit_share / "launch" / "move_group.launch.py")),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(moveit_share / "launch" / "moveit_rviz.launch.py")),
            ),
        ], condition=IfCondition(start_moveit)),

        # 이 두 entry point는 rclpy Node가 아니라 argparse 기반 프로세스이므로
        # launch_ros Node가 붙이는 --ros-args를 전달하지 않는다.
        ExecuteProcess(cmd=[str(c_executables / "tcp_server")], output="screen"),
        ExecuteProcess(cmd=[str(c_executables / "display_relay")], output="screen"),
        OpaqueFunction(function=lambda context: _live_pipeline_actions(
            context, ROOT, AI_CONFIG, LIVE_CONFIG, ai_log, stamp
        )),
    ])
