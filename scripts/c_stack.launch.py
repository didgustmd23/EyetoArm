"""C 실행 묶음: Gazebo, TCP 서버, 영상 중계, 선택형 A AI 영상 handoff.

실행 (저장소 루트에서 ROS/워크스페이스 환경을 source한 뒤):
    ros2 launch scripts/c_stack.launch.py

B center publisher는 미구현이므로 여기서 실행하지 않는다. 구현되면 아래 TODO 위치에
B의 Node 액션 한 개를 추가하면 된다. display_relay는 center 선택 때 토픽을 구독한다.
"""
from datetime import datetime
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
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

ROOT = Path(__file__).resolve().parent.parent
SIM_LAUNCH = ROOT / "scripts" / "sim_sort_cell.launch.py"
AI_CONFIG = ROOT / "config" / "vision" / "ai_s1_08_live.json"


def generate_launch_description():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ai_log = ROOT / "logs" / f"a_live_display_{stamp}.jsonl"
    gui = LaunchConfiguration("gui")
    start_a = LaunchConfiguration("start_a")
    start_moveit = LaunchConfiguration("start_moveit")
    a_python = LaunchConfiguration("a_python")
    moveit_share = Path(get_package_share_directory("ur5e_moveit_config"))
    c_executables = Path(get_package_prefix("robot_server_package")) / "lib" / "robot_server_package"

    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="false",
                              description="Gazebo GUI 실행 여부 (false면 headless)"),
        DeclareLaunchArgument("start_a", default_value="true",
                              description="A의 AI 검출·C 영상 handoff 실행 여부"),
        DeclareLaunchArgument("start_moveit", default_value="false",
                              description="MoveGroup과 MoveIt RViz 실행 여부"),
        DeclareLaunchArgument("a_python", default_value=str(ROOT / ".venv" / "bin" / "python"),
                              description="A 파이프라인에 사용할 Python 실행 파일"),

        # A 파이프라인도 ROS 이미지 토픽을 읽으므로 시뮬레이션과 같은 DDS 설정을 사용한다.
        SetEnvironmentVariable("ROS_AUTOMATIC_DISCOVERY_RANGE", "SUBNET"),
        SetEnvironmentVariable("FASTRTPS_DEFAULT_PROFILES_FILE",
                              str(ROOT / "config" / "dds" / "fastdds_large_image.xml")),
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

        # A의 ROS 링크 AI 파이프라인 안에 C handoff probe/47093 송신기가 포함되어 있다.
        ExecuteProcess(
            cmd=[a_python, str(ROOT / "scripts" / "run_pipeline.py"),
                 "--ros-link", "--detector", "ai",
                 "--ai-config", str(AI_CONFIG),
                 "--log", str(ai_log)],
            cwd=str(ROOT),
            output="screen",
            condition=IfCondition(start_a),
        ),

        # B center 영상 발행기가 준비되면 여기 추가:
        # Node(package="<B package>", executable="<B center publisher>", output="screen"),
    ])