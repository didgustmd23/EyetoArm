from setuptools import setup

package_name = 'robot_server_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='EyetoArm C 담당 (TODO)',
    maintainer_email='todo@example.com',
    description='EyetoArm C 담당 ROS 2 TCP 서버 및 로봇 실행 패키지',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tcp_server = robot_server_package.tcp_server:main',
            'display_relay = robot_server_package.display_relay:main',
            'planning_scene_manager = robot_server_package.planning_scene_manager:main',
            'pose_plan_test = robot_server_package.pose_plan_test:main',
            'conveyor_controller = robot_server_package.conveyor_controller:main',
        ],
    },
)
