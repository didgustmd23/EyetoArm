#!/usr/bin/env python3
"""Gazebo UR5e 관절 수동 조작(키보드).

    cd ~/EyetoArm && source /opt/ros/jazzy/setup.bash
    python3 scripts/manual_joint_teleop.py            # 기본 5도씩
    python3 scripts/manual_joint_teleop.py --step 2   # 2도씩

키를 누를 때마다 해당 관절 목표를 한 스텝 옮겨 joint_trajectory_controller에 보낸다.
ROS_DOMAIN_ID는 Gazebo를 실행한 터미널과 같아야 한다.
"""
import argparse
import math
import os
import select
import sys
import termios
import time
import tty

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

JOINTS = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]
# 관절별 (+키, -키)
KEYS = [('q', 'a'), ('w', 's'), ('e', 'd'), ('r', 'f'), ('t', 'g'), ('y', 'h')]
LIMIT = 2 * math.pi  # UR5e 관절 한계(rad)
STEPS = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0]  # 도

HELP = """\
관절 수동 조작            +  -
  1 베이스(shoulder_pan)  q  a
  2 어깨(shoulder_lift)   w  s
  3 팔꿈치(elbow)         e  d
  4 손목1(wrist_1)        r  f
  5 손목2(wrist_2)        t  g
  6 손목3(wrist_3)        y  h
  [ / ]  스텝 줄이기/늘리기    0  시작 자세로 복귀    x 또는 Ctrl+C  종료
"""


def main():
    parser = argparse.ArgumentParser(description='Gazebo UR5e 관절 키보드 수동 조작')
    parser.add_argument('--step', type=float, default=5.0, help='키 한 번당 이동 각도(도)')
    parser.add_argument('--duration', type=float, default=0.5, help='한 스텝 이동 시간(초)')
    args = parser.parse_args()
    if args.step <= 0 or args.duration <= 0:
        parser.error('--step, --duration은 0보다 커야 합니다')
    if not sys.stdin.isatty():
        raise SystemExit('키보드 입력이 필요합니다. 터미널에서 파일로 직접 실행하세요.')

    rclpy.init()
    node = rclpy.create_node('gazebo_manual_joint_teleop')
    state = {}
    node.create_subscription(
        JointState, '/joint_states',
        lambda msg: state.update(zip(msg.name, msg.position)),
        qos_profile_sensor_data,
    )
    pub = node.create_publisher(
        JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)

    deadline = time.monotonic() + 10
    while not all(j in state for j in JOINTS) or pub.get_subscription_count() == 0:
        if time.monotonic() > deadline:
            node.destroy_node()
            rclpy.shutdown()
            raise SystemExit('관절 상태 또는 제어기 연결 없음 (Gazebo 실행·ROS_DOMAIN_ID 확인)')
        rclpy.spin_once(node, timeout_sec=0.1)

    # 목표는 현재 자세에서 시작해 누적한다(이동 중 연타해도 스텝이 누락되지 않음)
    target = [state[j] for j in JOINTS]
    start = list(target)
    step_deg = args.step
    seconds = int(args.duration)
    nanoseconds = int((args.duration - seconds) * 1e9)

    def send():
        msg = JointTrajectory()
        msg.joint_names = JOINTS
        point = JointTrajectoryPoint()
        point.positions = list(target)
        point.time_from_start.sec = seconds
        point.time_from_start.nanosec = nanoseconds
        msg.points = [point]
        pub.publish(msg)

    def show(note=''):
        degrees = ' '.join(f'{math.degrees(v):7.1f}' for v in target)
        sys.stdout.write(f'\r\033[K목표(도): {degrees} | 스텝 {step_deg:g}도 {note}')
        sys.stdout.flush()

    print(HELP)
    show()
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            if not select.select([sys.stdin], [], [], 0.05)[0]:
                continue
            # 버퍼링 없이 한 글자씩 읽어 연타·키 반복 입력이 밀리지 않게 한다
            key = os.read(fd, 1).decode(errors='ignore').lower()
            if key in ('x', '\x03'):
                break
            if key == '[':
                step_deg = max([s for s in STEPS if s < step_deg], default=step_deg)
                show()
            elif key == ']':
                step_deg = min([s for s in STEPS if s > step_deg], default=step_deg)
                show()
            elif key == '0':
                target[:] = start
                send()
                show('(시작 자세)')
            else:
                for index, (plus, minus) in enumerate(KEYS):
                    if key not in (plus, minus):
                        continue
                    delta = math.radians(step_deg) * (1 if key == plus else -1)
                    wanted = target[index] + delta
                    if abs(wanted) > LIMIT:
                        show('(관절 한계)')
                    else:
                        target[index] = wanted
                        send()
                        show()
                    break
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        print('\n종료')
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
