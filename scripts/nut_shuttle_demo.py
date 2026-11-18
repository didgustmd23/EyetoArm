#!/usr/bin/env python3
"""Gazebo UR5e·2F-85로 너트를 벨트 ↔ 초록색 상자 사이에서 반복해 옮기는 시연.

    cd ~/EyetoArm && source /opt/ros/jazzy/setup.bash
    python3 scripts/nut_shuttle_demo.py               # 20회
    python3 scripts/nut_shuttle_demo.py --cycles 3

한 회: 벨트의 너트 집기 → 초록색 상자(bin_bolt_b) 안에 놓기 → 처음 자세 복귀
      → 상자 안 너트 집기 → 벨트의 처음 자리에 놓기 → 처음 자세 복귀.
너트·상자의 실제 위치는 Gazebo에서 읽는다. 기구학·실행 코드는 pick_bolt_demo.py와 같고
MoveIt·충돌 검사는 쓰지 않는다(시뮬레이션 시연 전용, 운영 경로 아님).
"""
import argparse
import time

import numpy as np
import rclpy

from pick_bolt_demo import (FINGER_REACH, GRIP_CLOSED, GRIPPER_JOINT, Arm, PartTracker, load_chain,
                            quaternion_matrix, solve_ik, transform)

BIN_FLOOR = 0.004            # 상자 바닥 윗면 높이(상자 원점 기준, m)
BIN_HALF = (0.076, 0.056)    # 상자 안쪽 반폭: 긴 변, 짧은 변(m)
PRE_OPEN = 0.5               # 집기 전·놓은 뒤 그리퍼 벌림(rad). 다 벌리면 상자 벽에 닿는다(손끝 바깥 폭 약 15 cm)
TIP_CLEARANCE = 0.001        # 닫힌 손끝과 받침면 사이 여유(m)
TRAVEL_TIP = 0.14            # 옮길 때 손끝 높이(m). 상자 벽 윗면은 0.08
RELEASE_STEP = 0.03          # 놓을 때 그리퍼를 이만큼씩 연다(rad)


def main():
    parser = argparse.ArgumentParser(description='너트를 벨트와 초록색 상자 사이에서 반복해 옮기는 Gazebo 시연')
    parser.add_argument('--cycles', type=int, default=20)
    parser.add_argument('--part', default='part_02', help='너트의 Gazebo 모델 이름')
    parser.add_argument('--bin', default='bin_bolt_b', help='상자의 Gazebo 모델 이름(기본: 초록색)')
    parser.add_argument('--world', default='sort_cell')
    args = parser.parse_args()
    if args.cycles < 1: parser.error('--cycles must be at least 1')

    chain = load_chain()
    nut, box = PartTracker(args.world, args.part), PartTracker(args.world, args.bin)
    rclpy.init(); node = rclpy.create_node('nut_shuttle_demo'); arm = Arm(node)
    home = arm.joints()
    travel_z = TRAVEL_TIP + FINGER_REACH

    belt_spot, _ = nut.pose()                                   # 너트 원점 = 아랫면 중심
    box_position, box_quaternion = box.pose()
    box_rotation = quaternion_matrix(*box_quaternion)
    long_side = box_rotation @ np.array([1.0, 0.0, 0.0])        # 상자의 긴 변 방향. 손가락은 이 방향으로 벌린다
    box_spot = box_position + np.array([0.0, 0.0, BIN_FLOOR])
    print(f'너트 {args.part}: ({belt_spot[0]:.3f}, {belt_spot[1]:.3f}, {belt_spot[2]:.3f}) m, '
          f'상자 {args.bin} 바닥 중심: ({box_spot[0]:.3f}, {box_spot[1]:.3f}, {box_spot[2]:.3f}) m', flush=True)

    down = np.array([0.0, 0.0, -1.0])
    options = []
    for sign in (1.0, -1.0):
        closing = sign * np.array([long_side[0], long_side[1], 0.0]) / np.hypot(long_side[0], long_side[1])
        rotation = np.column_stack([closing, np.cross(down, closing), down])
        try:
            q = solve_ik(chain, transform(rotation, [belt_spot[0], belt_spot[1], travel_z]), home)
            options.append((float(np.max(np.abs(q - home))), rotation, q))
        except RuntimeError:
            pass
    if not options: raise RuntimeError('너트 위쪽 자세의 역기구학 해가 없다')
    _, rotation, seed = min(options, key=lambda o: o[0])

    def ik(xyz, near): return solve_ik(chain, transform(rotation, xyz), near)

    def line(start_q, start, end, steps):
        path, q = [], start_q
        for k in range(1, steps + 1):
            q = ik(start + (end - start) * k / steps, q); path.append(q)
        return path

    def tool_xyz(spot, z): return np.array([spot[0], spot[1], z])

    def nut_flat():
        position, quaternion = nut.pose()
        if abs((quaternion_matrix(*quaternion) @ np.array([0.0, 0.0, 1.0]))[2]) < 0.95:
            raise RuntimeError('너트가 눕지 않고 서 있어 집을 수 없다')
        return position

    def pick(label):
        """지금 너트가 놓인 곳에서 집어 옮김 높이까지 올린다. 끝 관절값과 그리퍼 기준 너트 어긋남을 돌려준다."""
        spot = nut_flat()
        grasp_z = spot[2] + TIP_CLEARANCE + FINGER_REACH
        above = ik(tool_xyz(spot, travel_z), seed)
        arm.grip(PRE_OPEN, settle=0.2)
        arm.follow([above], 2.5)
        descent = line(above, tool_xyz(spot, travel_z), tool_xyz(spot, grasp_z), 20)
        arm.follow(descent, 1.2 + 6 * (travel_z - grasp_z))
        arm.grip(GRIP_CLOSED, settle=0.3)
        arm.follow(line(descent[-1], tool_xyz(spot, grasp_z), tool_xyz(spot, travel_z), 20), 1.2 + 6 * (travel_z - grasp_z))
        held = nut.pose()[0]
        if held[2] - spot[2] < 0.6 * (travel_z - grasp_z):
            raise RuntimeError(f'{label}: 너트를 들어 올리지 못했다 (높이 변화 {(held[2] - spot[2]) * 1000:+.1f} mm)')
        slip = held - np.array([spot[0], spot[1], spot[2] + travel_z - grasp_z])   # 집는 동안 손끝에서 밀린 양
        return above, spot, slip

    def place(above_from, from_spot, target, slip):
        """옮김 높이에서 목표 위로 가서 너트 아랫면이 받침면에 닿을 만큼 내린 뒤 조금씩 벌려 놓는다."""
        goal = target - np.array([slip[0], slip[1], 0.0])
        release_z = target[2] + TIP_CLEARANCE + FINGER_REACH - min(slip[2], 0.0) + 0.0005
        over = line(above_from, tool_xyz(from_spot, travel_z), tool_xyz(goal, travel_z), 25)
        arm.follow(over, 2.2)
        descent = line(over[-1], tool_xyz(goal, travel_z), tool_xyz(goal, release_z), 20)
        arm.follow(descent, 1.2 + 6 * (travel_z - release_z))
        position = arm.state[GRIPPER_JOINT]
        while position > PRE_OPEN + 1e-6:
            position = max(PRE_OPEN, position - RELEASE_STEP)
            arm.grip(position, settle=0.1)
        arm.follow(line(descent[-1], tool_xyz(goal, release_z), tool_xyz(goal, travel_z), 20), 1.2 + 6 * (travel_z - release_z))
        arm.follow([home], 2.5)
        arm.wait(0.3)
        return nut.pose()[0]

    began, done = time.monotonic(), 0
    try:
        for cycle in range(1, args.cycles + 1):
            above, spot, slip = pick(f'{cycle}회 벨트')
            landed = place(above, spot, box_spot, slip)
            local = box_rotation.T @ (landed - box_position)
            if abs(local[0]) > BIN_HALF[0] or abs(local[1]) > BIN_HALF[1] or abs(landed[2] - box_spot[2]) > 0.003:
                raise RuntimeError(f'{cycle}회: 너트가 상자 안에 놓이지 않았다 ({landed[0]:.3f}, {landed[1]:.3f}, {landed[2]:.3f})')
            box_miss = np.linalg.norm(landed[:2] - box_spot[:2])
            above, spot, slip = pick(f'{cycle}회 상자')
            landed = place(above, spot, belt_spot, slip)
            belt_miss = np.linalg.norm(landed[:2] - belt_spot[:2])
            if abs(landed[2] - belt_spot[2]) > 0.003:
                raise RuntimeError(f'{cycle}회: 너트가 벨트 위에 놓이지 않았다 ({landed[0]:.3f}, {landed[1]:.3f}, {landed[2]:.3f})')
            done = cycle
            print(f'{cycle}/{args.cycles}회 완료: 상자 중심에서 {box_miss * 1000:.1f} mm, 벨트 처음 자리에서 {belt_miss * 1000:.1f} mm '
                  f'(누적 {time.monotonic() - began:.0f}초)', flush=True)
        print(f'끝: {done}회 모두 성공', flush=True)
        return 0
    except RuntimeError as error:
        print(f'중단: {error} — {done}회까지 성공', flush=True); return 1
    except KeyboardInterrupt:
        print(f'사용자 중단 — {done}회까지 성공', flush=True); return 130
    finally:
        node.destroy_node(); rclpy.try_shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
