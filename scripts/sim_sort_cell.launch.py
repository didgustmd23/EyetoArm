"""분류 셀 월드 + UR5e·2F-85 생성 + 제어기 기동 (D 외형 확인·데모, C 실행 구성의 대역).

    source /opt/ros/jazzy/setup.bash
    ros2 launch scripts/sim_sort_cell.launch.py            # GUI
    ros2 launch scripts/sim_sort_cell.launch.py gui:=false # 헤드리스(저사양)
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, IfElseSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

SIM = Path(__file__).resolve().parent.parent / "simulation"
ROBOT = SIM / "models" / "ur5e_2f85"


def generate_launch_description():
    world = LaunchConfiguration("world")
    gui = LaunchConfiguration("gui")

    description = Command([
        FindExecutable(name="xacro"), " ", str(ROBOT / "ur5e_2f85.urdf.xacro"),
        " simulation_controllers:=", str(ROBOT / "controllers.yaml"),
    ])
    robot_description = {"robot_description": ParameterValue(description, value_type=str)}

    # package://robotiq_description/... 메시를 Gazebo가 찾도록 share의 상위 폴더를 더한다
    resource_paths = {os.path.dirname(get_package_share_directory(p)) for p in ("robotiq_description", "ur_description")}

    def spawner(name):
        return Node(package="controller_manager", executable="spawner",
                    arguments=[name, "--controller-manager", "/controller_manager"], output="screen")

    return LaunchDescription([
        DeclareLaunchArgument("world", default_value=str(SIM / "worlds" / "sort_cell.sdf")),
        DeclareLaunchArgument("gui", default_value="true"),
        AppendEnvironmentVariable("GZ_SIM_RESOURCE_PATH", str(SIM / "models")),
        *[AppendEnvironmentVariable("GZ_SIM_RESOURCE_PATH", p) for p in sorted(resource_paths)],
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")),
            launch_arguments={"gz_args": IfElseSubstitution(gui, if_value=["-r ", world], else_value=["-s -r ", world])}.items(),
        ),
        Node(package="robot_state_publisher", executable="robot_state_publisher", output="screen",
             parameters=[{"use_sim_time": True}, robot_description]),
        Node(package="ros_gz_sim", executable="create", output="screen",
             arguments=["-topic", "robot_description", "-name", "ur5e_2f85"]),
        Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
             arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"]),
        spawner("joint_state_broadcaster"),
        spawner("joint_trajectory_controller"),
        spawner("gripper_controller"),
    ])
