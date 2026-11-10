"""로봇 없이 분류 셀 외형(벨트·상자·볼트·너트)만 띄운다.

    source /opt/ros/jazzy/setup.bash
    ros2 launch scripts/sim_scene.launch.py
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import IfElseSubstitution, LaunchConfiguration

SIM = Path(__file__).resolve().parent.parent / "simulation"


def generate_launch_description():
    world = LaunchConfiguration("world")
    gui = LaunchConfiguration("gui")
    return LaunchDescription([
        DeclareLaunchArgument("world", default_value=str(SIM / "worlds" / "sort_cell.sdf")),
        DeclareLaunchArgument("gui", default_value="true"),
        AppendEnvironmentVariable("GZ_SIM_RESOURCE_PATH", str(SIM / "models")),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")),
            launch_arguments={"gz_args": IfElseSubstitution(gui, if_value=["-r ", world], else_value=["-s -r ", world])}.items(),
        ),
    ])
