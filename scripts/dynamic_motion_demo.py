#!/usr/bin/env python3
"""Gazebo UR5e 역동 동작 시연(약 2분): 좌우 → 상하 → 관절별 → 전 관절 복합.

    cd ~/EyetoArm && source /opt/ros/jazzy/setup.bash
    python3 scripts/dynamic_motion_demo.py
    python3 scripts/dynamic_motion_demo.py --scale 0.7   # 진폭을 70%로

현재 자세를 기준으로 움직이고 끝나면 그 자세로 돌아온다. 궤적 전체를 한 번에
joint_trajectory_controller에 보내며, 충돌 검사는 하지 않는다(시뮬레이션 전용).
ROS_DOMAIN_ID는 Gazebo를 실행한 터미널과 같아야 한다.
"""
import argparse
import math

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.msg import JointTrajectoryControllerState
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

JOINTS = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
          'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
DT = 0.05          # 궤적 점 간격(초)
RAMP = 1.0         # 구간 시작·끝에서 진폭을 부드럽게 올리고 내리는 시간(초)
SOLO = [0.7, 0.4, 0.6, 0.9, 0.9, 1.2]          # 관절별 구간 진폭(rad)
MIX = [(0.8, 7.0), (0.3, 5.0), (0.45, 4.2), (0.7, 3.0), (0.7, 3.5), (1.0, 2.5)]  # 복합 구간 (진폭 rad, 주기 초)
# (이름, 길이 초)
PHASES = [('좌우', 20.0), ('상하', 20.0), ('관절별', 36.0), ('복합', 42.0)]
TOTAL = sum(d for _, d in PHASES)


def envelope(t, duration):
    """구간 양 끝 RAMP 초 동안 0↔1로 매끄럽게 변하는 값."""
    x = min(t, duration - t, RAMP) / RAMP
    return x * x * (3 - 2 * x)


def offsets(t):
    """시각 t(초)에서 기준 자세에 더할 관절 여섯 개의 변위(rad)."""
    q = [0.0] * 6
    for name, duration in PHASES:
        if t >= duration:
            t -= duration
            continue
        env = envelope(t, duration)
        if name == '좌우':        # 베이스를 크게 좌우로 4회
            q[0] = 0.9 * math.sin(2 * math.pi * t / 5.0)
        elif name == '상하':      # 어깨·팔꿈치를 반대로 써서 손끝을 위아래로 4회
            s = math.sin(2 * math.pi * t / 5.0)
            q[1], q[2], q[3] = 0.45 * s, -0.6 * s, 0.15 * s
        elif name == '관절별':    # 관절 하나씩 6초 동안 2회 흔들기
            index = min(int(t // 6.0), 5)
            local = t - 6.0 * index
            env = envelope(local, 6.0)
            q[index] = SOLO[index] * math.sin(2 * math.pi * local / 3.0)
        else:                     # 여섯 관절을 서로 다른 주기로 동시에
            q = [amp * math.sin(2 * math.pi * t / period) for amp, period in MIX]
        return [env * value for value in q]
    return q


def main():
    parser = argparse.ArgumentParser(description='UR5e 역동 동작 시연(약 2분)')
    parser.add_argument('--scale', type=float, default=1.0, help='진폭 배율 (0~1.2, 기본 1.0)')
    args = parser.parse_args()
    if not 0 < args.scale <= 1.2:
        parser.error('--scale must be in (0, 1.2]')

    rclpy.init()
    node = rclpy.create_node('dynamic_motion_demo')
    state, stamp, worst = {}, [0.0], [0.0] * 6

    def on_joints(message):
        state.update(zip(message.name, message.position))
        stamp[0] = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9

    def on_controller(message):
        for i, name in enumerate(message.joint_names):
            if name in JOINTS and i < len(message.error.positions):
                k = JOINTS.index(name)
                worst[k] = max(worst[k], abs(message.error.positions[i]))

    node.create_subscription(JointState, '/joint_states', on_joints, 10)
    node.create_subscription(JointTrajectoryControllerState, '/joint_trajectory_controller/controller_state', on_controller, 10)
    publisher = node.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
    while not all(joint in state for joint in JOINTS) or publisher.get_subscription_count() == 0:
        rclpy.spin_once(node, timeout_sec=0.2)
    home = [state[joint] for joint in JOINTS]

    def pose(t):
        return [h + args.scale * o for h, o in zip(home, offsets(t))]

    message = JointTrajectory(joint_names=JOINTS)
    steps = int(round(TOTAL / DT))
    for k in range(1, steps + 1):
        t = k * DT
        before, after = pose(t - DT), pose(min(t + DT, TOTAL))
        velocity = [(a - b) / (2 * DT) for a, b in zip(after, before)] if k < steps else [0.0] * 6
        message.points.append(JointTrajectoryPoint(
            positions=pose(t) if k < steps else home, velocities=velocity,
            time_from_start=Duration(sec=int(t), nanosec=int(round((t % 1) * 1e9)) % 1_000_000_000)))
    began = stamp[0]
    publisher.publish(message)
    print(f'시작: 총 {TOTAL:.0f}초 (시뮬레이션 시간), 진폭 배율 {args.scale:g}', flush=True)

    edges, elapsed = [], 0.0
    for name, duration in PHASES:
        edges.append((elapsed, name)); elapsed += duration
    try:
        while stamp[0] - began < TOTAL + 1.0:
            rclpy.spin_once(node, timeout_sec=0.1)
            while edges and stamp[0] - began >= edges[0][0]:
                print(f'[{edges[0][0]:5.0f}초] {edges.pop(0)[1]} 구간', flush=True)
    except KeyboardInterrupt:      # 중단하면 현재 위치에서 멈춘다
        stop = JointTrajectory(joint_names=JOINTS)
        stop.points = [JointTrajectoryPoint(positions=[state[j] for j in JOINTS], time_from_start=Duration(nanosec=300_000_000))]
        publisher.publish(stop)
        print('중단: 현재 위치에서 정지', flush=True)
    final = max(abs(state[j] - h) for j, h in zip(JOINTS, home))
    print('관절별 최대 추종 오차(rad): ' + ', '.join(f'{j.replace("_joint", "")}={e:.3f}' for j, e in zip(JOINTS, worst)), flush=True)
    print(f'종료 시 기준 자세와의 최대 차이: {final:.4f} rad', flush=True)
    node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
