#!/usr/bin/env python3
"""Gazebo UR5e·2F-85로 벨트 위 볼트를 집어 올렸다가 제자리에 내려놓는 시연.

    cd ~/EyetoArm && source /opt/ros/jazzy/setup.bash
    python3 scripts/pick_bolt_demo.py                  # 기본 대상: part_01 (M6x40 볼트)
    python3 scripts/pick_bolt_demo.py --part part_00   # M6x12 볼트
    python3 scripts/pick_bolt_demo.py --lift 0.20      # 20 cm 들어 올리기

볼트의 실제 위치·방향을 Gazebo에서 읽어 축 가운데를 위에서 잡는다. 역기구학은 이 파일 안의
수치 해법을 쓰며 MoveIt·충돌 검사는 쓰지 않는다(시뮬레이션 시연 전용, 운영 경로 아님).
ROS_DOMAIN_ID·GZ_PARTITION은 Gazebo를 실행한 터미널과 같아야 한다.
"""
import argparse
import math
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory, ParallelGripperCommand
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ROBOT = Path(__file__).resolve().parent.parent / 'simulation' / 'models' / 'ur5e_2f85'
JOINTS = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
          'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
GRIPPER_JOINT = 'robotiq_85_left_knuckle_joint'
TOOL_LINK = 'robotiq_85_base_link'
GRIP_OPEN, GRIP_CLOSED = 0.0, 0.8     # 구동 관절(rad)
FINGER_REACH = 0.163                  # 그리퍼 밑면에서 닫힌 손끝 끝까지(m). 열렸을 때는 0.149
TIP_CLEARANCE = 0.0015                # 닫힌 손끝과 받침면 사이 여유(m)
SHANK_RADIUS = 0.003                  # M6 축 반지름(m)
SHANK_LENGTH = {'part_00': 0.012, 'part_01': 0.040}
APPROACH = 0.10                       # 잡기 전·놓은 뒤 위쪽 대기 높이(m)
RELEASE_HEIGHT = 0.0                  # 놓을 때 잡은 높이 기준 추가 높이(m). 띄워서 놓으면 볼트가 굴러 벨트 밖으로 나간다
RELEASE_STEP = 0.02                   # 놓을 때 그리퍼를 이만큼씩 천천히 연다(rad). 한 번에 열면 볼트가 튕겨 구른다


# ── 기구학 ────────────────────────────────────────────────────────────────
def rpy_matrix(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def transform(rotation, xyz):
    t = np.eye(4); t[:3, :3] = rotation; t[:3, 3] = xyz
    return t


def load_chain():
    """base_link → 그리퍼 밑면까지의 (고정 변환, 관절 번호 또는 None) 목록을 URDF에서 만든다."""
    urdf = subprocess.run(['xacro', str(ROBOT / 'ur5e_2f85.urdf.xacro')], check=True, capture_output=True, text=True).stdout
    joints = {j.find('child').get('link'): j for j in ET.fromstring(urdf).findall('joint')}
    chain, link = [], TOOL_LINK
    while link != 'base_link':
        joint = joints[link]
        origin = joint.find('origin')
        xyz = [float(v) for v in origin.get('xyz', '0 0 0').split()] if origin is not None else [0, 0, 0]
        rpy = [float(v) for v in origin.get('rpy', '0 0 0').split()] if origin is not None else [0, 0, 0]
        index = JOINTS.index(joint.get('name')) if joint.get('name') in JOINTS else None
        if index is None and joint.get('type') != 'fixed':
            raise RuntimeError(f'unexpected moving joint {joint.get("name")}')
        chain.append((transform(rpy_matrix(*rpy), xyz), index))
        link = joint.find('parent').get('link')
    return chain[::-1]


def forward(chain, q):
    t = np.eye(4)
    for fixed, index in chain:
        t = t @ fixed
        if index is not None:      # UR 관절 축은 모두 자기 좌표계의 +Z
            c, s = math.cos(q[index]), math.sin(q[index])
            t = t @ transform(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]), [0, 0, 0])
    return t


def pose_error(current, target):
    rotation = target[:3, :3] @ current[:3, :3].T
    angle = math.acos(max(-1.0, min(1.0, (np.trace(rotation) - 1) / 2)))
    axis = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]])
    norm = np.linalg.norm(axis)
    spin = axis / norm * angle if norm > 1e-9 else np.zeros(3)
    return np.concatenate([target[:3, 3] - current[:3, 3], spin])


def solve_ik(chain, target, seed):
    """감쇠 최소제곱 수치 역기구학. 위치 0.1 mm·자세 0.001 rad 안에 들면 해로 본다."""
    q = np.array(seed, dtype=float)
    for _ in range(300):
        error = pose_error(forward(chain, q), target)
        if np.linalg.norm(error[:3]) < 1e-4 and np.linalg.norm(error[3:]) < 1e-3:
            return q
        jacobian = np.zeros((6, 6))
        for i in range(6):
            dq = q.copy(); dq[i] += 1e-6
            jacobian[:, i] = pose_error(forward(chain, q), forward(chain, dq)) / 1e-6
        step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-4 * np.eye(6), error)
        q += np.clip(step, -0.2, 0.2)
    raise RuntimeError('역기구학 해를 찾지 못했다(작업 범위 밖일 수 있음)')


# ── Gazebo 물체 자세 ─────────────────────────────────────────────────────
class PartTracker:
    def __init__(self, world, part):
        from gz.msgs10.pose_v_pb2 import Pose_V
        from gz.transport13 import Node
        self._part, self._lock, self._pose, self._node = part, threading.Lock(), None, Node()
        topic = f'/world/{world}/pose/info'
        if not self._node.subscribe(Pose_V, topic, self._on_poses):
            raise RuntimeError(f'cannot subscribe to {topic}')

    def _on_poses(self, message):
        for pose in message.pose:
            if pose.name == self._part:
                p, o = pose.position, pose.orientation
                with self._lock: self._pose = (np.array([p.x, p.y, p.z]), (o.w, o.x, o.y, o.z))

    def pose(self, timeout=5.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self._lock:
                if self._pose is not None: return self._pose
            time.sleep(0.05)
        raise RuntimeError(f'Gazebo에서 {self._part} 자세를 받지 못했다(GZ_PARTITION 확인)')


def quaternion_matrix(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# ── 실행 ──────────────────────────────────────────────────────────────────
class Arm:
    def __init__(self, node):
        self.node, self.state = node, {}
        node.create_subscription(JointState, '/joint_states', lambda m: self.state.update(zip(m.name, m.position)), 10)
        self.trajectory = ActionClient(node, FollowJointTrajectory, '/joint_trajectory_controller/follow_joint_trajectory')
        self.gripper = ActionClient(node, ParallelGripperCommand, '/gripper_controller/gripper_cmd')
        for client in (self.trajectory, self.gripper):
            if not client.wait_for_server(timeout_sec=10.0): raise RuntimeError('제어기 액션 서버가 없다(Gazebo 실행 확인)')
        while not all(j in self.state for j in JOINTS + [GRIPPER_JOINT]): rclpy.spin_once(node, timeout_sec=0.2)

    def joints(self): return np.array([self.state[j] for j in JOINTS])

    def _call(self, client, goal, limit):
        future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=10.0)
        handle = future.result()
        if handle is None or not handle.accepted: raise RuntimeError('제어기가 목표를 받지 않았다')
        result = handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result, timeout_sec=limit)
        if not result.done(): raise RuntimeError('동작이 제한 시간 안에 끝나지 않았다')
        return result.result().result

    def follow(self, waypoints, seconds):
        """관절 목표들을 같은 간격으로 지나간다. 끝난 뒤 실제 관절이 자리 잡을 때까지 기다린다."""
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory(joint_names=JOINTS)
        for k, q in enumerate(waypoints, 1):
            t = seconds * k / len(waypoints)
            goal.trajectory.points.append(JointTrajectoryPoint(
                positions=[float(v) for v in q], time_from_start=Duration(sec=int(t), nanosec=int((t % 1) * 1e9))))
        self._call(self.trajectory, goal, seconds * 4 + 20)
        end = time.monotonic() + 8.0
        while time.monotonic() < end and np.max(np.abs(self.joints() - waypoints[-1])) > 0.003:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        return float(np.max(np.abs(self.joints() - waypoints[-1])))

    def grip(self, position, settle=0.5):
        goal = ParallelGripperCommand.Goal()
        goal.command = JointState(name=[GRIPPER_JOINT], position=[float(position)])
        result = self._call(self.gripper, goal, 20.0)
        self.wait(settle)
        return result

    def release(self):
        """손가락을 조금씩 벌려 물체를 건드리지 않고 놓은 뒤 완전히 연다."""
        position = self.state[GRIPPER_JOINT]
        while position > 0.5:
            position -= RELEASE_STEP
            self.grip(position, settle=0.15)
        self.grip(GRIP_OPEN)

    def wait(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end: rclpy.spin_once(self.node, timeout_sec=0.05)


def main():
    parser = argparse.ArgumentParser(description='볼트를 집어 올렸다가 내려놓는 Gazebo 시연')
    parser.add_argument('--part', default='part_01', choices=sorted(SHANK_LENGTH), help='대상 볼트의 Gazebo 모델 이름')
    parser.add_argument('--world', default='sort_cell')
    parser.add_argument('--lift', type=float, default=0.15, help='들어 올리는 높이(m)')
    parser.add_argument('--hold', type=float, default=3.0, help='들어 올린 채 머무는 시간(초)')
    args = parser.parse_args()
    if not 0.02 <= args.lift <= 0.4: parser.error('--lift must be between 0.02 and 0.4')

    chain = load_chain()
    tracker = PartTracker(args.world, args.part)
    rclpy.init(); node = rclpy.create_node('pick_bolt_demo'); arm = Arm(node)
    home = arm.joints()

    position, quaternion = tracker.pose()
    axis = quaternion_matrix(*quaternion) @ np.array([0.0, 0.0, -1.0])     # 머리 → 축 끝 방향
    flat = np.array([axis[0], axis[1], 0.0])
    if abs(axis[2]) > 0.5 or np.linalg.norm(flat) < 1e-6:
        raise RuntimeError('볼트가 누워 있지 않아 이 시연의 잡기 방식으로는 잡을 수 없다')
    flat /= np.linalg.norm(flat)
    center = position + axis * SHANK_LENGTH[args.part] / 2                 # 축 가운데
    support = center[2] - SHANK_RADIUS                                     # 축 아래 받침면 높이
    grasp_z = support + TIP_CLEARANCE + FINGER_REACH
    print(f'{args.part}: 위치 ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}) m, '
          f'잡는 점 ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) m', flush=True)

    def tool_pose(z, closing):
        down = np.array([0.0, 0.0, -1.0])                                  # 손가락 전방(+Z)이 아래를 봄
        return transform(np.column_stack([closing, np.cross(down, closing), down]), [center[0], center[1], z])

    # 손가락이 닫히는 방향(그리퍼 X)은 볼트 축과 수직. 두 방향 중 지금 자세에서 덜 도는 쪽을 쓴다.
    options = []
    for sign in (1.0, -1.0):
        closing = sign * np.array([-flat[1], flat[0], 0.0])
        try:
            q = solve_ik(chain, tool_pose(grasp_z + APPROACH, closing), home)
            options.append((float(np.max(np.abs(q - home))), closing, q))
        except RuntimeError:
            pass
    if not options: raise RuntimeError('볼트 위쪽 자세의 역기구학 해가 없다')
    _, closing, above = min(options, key=lambda o: o[0])

    def line(start_q, z_from, z_to, steps=30):
        path, q = [], start_q
        for k in range(1, steps + 1):
            q = solve_ik(chain, tool_pose(z_from + (z_to - z_from) * k / steps, closing), q)
            path.append(q)
        return path

    down = line(above, grasp_z + APPROACH, grasp_z)
    up = line(down[-1], grasp_z, grasp_z + args.lift)
    back = line(up[-1], grasp_z + args.lift, grasp_z + RELEASE_HEIGHT)
    retreat = line(back[-1], grasp_z + RELEASE_HEIGHT, grasp_z + APPROACH)

    def bolt_z(): return float(tracker.pose()[0][2])
    start_z = bolt_z()
    try:
        print('1/8 그리퍼 열기', flush=True); arm.grip(GRIP_OPEN)
        print('2/8 볼트 위로 이동', flush=True); arm.follow([above], 4.0)
        print('3/8 내려가기', flush=True); arm.follow(down, 3.0)
        print('4/8 잡기', flush=True); result = arm.grip(GRIP_CLOSED)
        knuckle = arm.state[GRIPPER_JOINT]
        print(f'    그리퍼 관절 {knuckle:.3f} rad (완전히 닫히면 {GRIP_CLOSED}), stalled={result.stalled}', flush=True)
        print(f'5/8 {args.lift * 100:.0f} cm 들어 올리기', flush=True); arm.follow(up, 3.0); arm.wait(args.hold)
        lifted = bolt_z() - start_z
        print(f'    볼트 높이 변화 {lifted * 1000:+.1f} mm → {"들어 올림 성공" if lifted > args.lift * 0.8 else "들어 올리지 못함"}', flush=True)
        print('6/8 내리기', flush=True); arm.follow(back, 3.0)
        print(f'    놓기 직전 볼트 높이 {bolt_z() * 1000:.1f} mm (처음 {start_z * 1000:.1f} mm)', flush=True)
        print('7/8 놓기', flush=True); arm.release(); arm.wait(1.0)
        print(f'    놓은 뒤 볼트 높이 {bolt_z() * 1000:.1f} mm', flush=True)
        print('8/8 위로 빠진 뒤 처음 자세로', flush=True); arm.follow(retreat, 2.5); error = arm.follow([home], 4.0)
        arm.wait(1.0)
        final, _ = tracker.pose()
        moved = np.linalg.norm(final[:2] - position[:2])
        print(f'끝: 볼트 최종 위치 ({final[0]:.3f}, {final[1]:.3f}, {final[2]:.3f}) m, 처음 자리에서 {moved * 1000:.1f} mm, '
              f'팔 복귀 오차 {error:.4f} rad', flush=True)
        return 0 if lifted > args.lift * 0.8 else 1
    except KeyboardInterrupt:
        print('중단: 팔은 현재 궤적 지점에서 멈춘다', flush=True); return 130
    finally:
        node.destroy_node(); rclpy.try_shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
