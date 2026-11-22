"""분류 셀 월드 + UR5e·2F-85 생성 + 제어기 기동 (D 외형 확인·데모, C 실행 구성의 대역).

    source /opt/ros/jazzy/setup.bash
    ros2 launch scripts/sim_sort_cell.launch.py            # GUI
    ros2 launch scripts/sim_sort_cell.launch.py gui:=false # 헤드리스(저사양)

DDS 설정: 이 launch는 모든 노드에 대용량 영상용 Fast DDS 프로필(config/dds/fastdds_large_image.xml)과
ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET을 준다(기본 SHM 512 KiB에서 2.76 MB 영상의 bridge 송신이 수 분 뒤 느려지는 문제, A #187·#191).
이 launch 밖에서 띄우는 영상 구독 프로세스(A run_pipeline, B·D 노드 등)도 같은 두 변수를 준다:
    export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
    export FASTRTPS_DEFAULT_PROFILES_FILE=<저장소>/config/dds/fastdds_large_image.xml
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, IfElseSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

SIM = Path(__file__).resolve().parent.parent / "simulation"
DDS_PROFILE = Path(__file__).resolve().parent.parent / "config" / "dds" / "fastdds_large_image.xml"
ROBOT = SIM / "models" / "ur5e_2f85"


def generate_launch_description():
    world = LaunchConfiguration("world")
    gui = LaunchConfiguration("gui")
    physics_engine = LaunchConfiguration("physics_engine")

    description = Command([
        FindExecutable(name="xacro"), " ", str(ROBOT / "ur5e_2f85.urdf.xacro"),
        " simulation_controllers:=", str(ROBOT / "controllers.yaml"),
        ' link_camera_xyz:="', LaunchConfiguration("link_camera_xyz"), '"',
        ' link_camera_rpy:="', LaunchConfiguration("link_camera_rpy"), '"',
        ' gripper_camera_xyz:="', LaunchConfiguration("gripper_camera_xyz"), '"',
        ' gripper_camera_rpy:="', LaunchConfiguration("gripper_camera_rpy"), '"',
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
        DeclareLaunchArgument("physics_engine", default_value="gz-physics-dartsim-plugin",
                              description="Gazebo physics engine"),
        DeclareLaunchArgument("link_camera_xyz", default_value="-0.3420864 -0.0601521 0.1400000",
                              description="upper_arm_link 기준 매립 시험 카메라 위치(m), 흰색 몸체 위쪽 옆면"),
        DeclareLaunchArgument("link_camera_rpy", default_value="1.5707963 0 -0.815927",
                              description="upper_arm_link 기준 자세(rad), 초기 대기 자세에서 정면 아래 약 43.3도"),
        DeclareLaunchArgument("gripper_camera_xyz", default_value="0 0 0.008",
                              description="robotiq_85_base_link 기준 앞면 중앙 매립 RGB-D 카메라 위치(m). 1/3 그리퍼 앞면 z 0.030 m 기준(xacro 기본값과 같음)"),
        DeclareLaunchArgument("gripper_camera_rpy", default_value="0 -1.57079632679 0",
                              description="그리퍼 기준 자세(rad), 기본값은 손가락 전방(+Z)을 봄"),
        # 대용량 영상용 DDS 설정(모듈 설명 참고). LOCALHOST에서는 rmw가 전송 설정을 덮어쓰므로 SUBNET과 함께 준다
        SetEnvironmentVariable("ROS_AUTOMATIC_DISCOVERY_RANGE", "SUBNET"),
        SetEnvironmentVariable("FASTRTPS_DEFAULT_PROFILES_FILE", str(DDS_PROFILE)),
        AppendEnvironmentVariable("GZ_SIM_RESOURCE_PATH", str(SIM / "models")),
        *[AppendEnvironmentVariable("GZ_SIM_RESOURCE_PATH", p) for p in sorted(resource_paths)],
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")),
            launch_arguments={"gz_args": IfElseSubstitution(gui, if_value=["-r --physics-engine ", physics_engine, " ", world], else_value=["-s -r --physics-engine ", physics_engine, " ", world])}.items(),
        ),
        # publish_frequency: /tf 발행 상한(기본 20 Hz). 카메라(sim 40 Hz)보다 낮으면 촬영 시각 TF를 다음 발행까지 기다려
        # 50 ms TF 대기 한도를 넘는 프레임이 생긴다. 50 Hz에서 tf_late 0, TF 대기 즉시 100%(A 측정 #197·#198)
        Node(package="robot_state_publisher", executable="robot_state_publisher", output="screen",
             parameters=[{"use_sim_time": True, "publish_frequency": 50.0}, robot_description]),
        Node(package="ros_gz_sim", executable="create", output="screen",
             arguments=["-topic", "robot_description", "-name", "ur5e_2f85"]),
        Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
             arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
                        "/link_cam/image@sensor_msgs/msg/Image[gz.msgs.Image",
                        "/link_cam/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
                        "/gripper_cam/image@sensor_msgs/msg/Image[gz.msgs.Image",
                        "/gripper_cam/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
                        "/gripper_cam/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
                        "/overview_cam/image@sensor_msgs/msg/Image[gz.msgs.Image"]),
        spawner("joint_state_broadcaster"),
        spawner("joint_trajectory_controller"),
        spawner("gripper_controller"),
        # Gazebo world reset recreates gz_ros2_control's controller_manager;
        # this monitor respawns controllers when the services return.
        Node(package="robot_server_package", executable="controller_recovery", output="screen"),
    ])
