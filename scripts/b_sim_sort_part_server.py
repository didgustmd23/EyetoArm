#!/usr/bin/env python3
"""B 연결 시험용 Gazebo `/sort_part` 서버 — B 주문을 받아 **시뮬레이션 로봇을 실제로 움직인다** (C 정식 서버 아님).

가짜 서버(`b_fake_sort_part_server.py`)는 받기만 하는데, 이 서버는 B가 보낸 값 그대로 집고 상자에 넣는다.
- 잡는 위치·각도: `grasp_pose_candidates` (B 집는 점 Pb, 닫는 방향 2개 중 팔이 덜 도는 쪽)
- 높이: `approach_height_mm`·`lift_height_mm`, 놓을 곳: `place_position_mm`
- 역기구학·관절 궤적·그리퍼는 `scripts/pick_bolt_demo.py`(C 시연)의 것을 그대로 빌려 쓴다 (MoveIt·충돌 검사 없음)
- 너트(`PASS_THROUGH`)는 잡지 않고 바로 성공. 일하는 중 새 주문은 거절(B는 `rejected`로 세고 다시 보냄)

주의
- 벨트가 움직이면 B 위치는 "찍은 순간" 위치라 빗나간다(움직임 예측은 C 몫). 시험은 벨트를 멈추고 한다.
- 물리 엔진: 11/21부터 기본값인 Bullet Featherstone에서는 그리퍼가 볼트를 물고 떨려 닫기가 끝나지 않는다(시간 초과).
  DART(`physics_engine:=gz-physics-dartsim-plugin`)에서는 같은 파일로 집기·상자 넣기 성공 (B 시험 2026-11-22, A L-N1).
- C가 정식 서버를 띄우면 이 서버는 끈다 (같은 이름 `/sort_part` 서버가 둘이면 안 됨).

실행 (WSL, 저장소 맨 위, 시뮬을 켠 뒤):
    source /opt/ros/jazzy/setup.bash && source robot_ws/install/setup.bash
    PYTHONPATH=src:$PYTHONPATH python3 scripts/b_sim_sort_part_server.py
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import threading
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SHANK_RADIUS = {"BOLT_A": 0.0015, "BOLT_B": 0.003}   # shaft radius (m): M3x20, M6x40 (B-12)
TIP_CLEARANCE = 0.0015        # pick_bolt_demo 값
MISS_RATIO = 0.97             # 그리퍼가 거의 끝까지 닫히면 못 잡은 것으로 봄
FINGER_REACH = 0.163 / 3      # 그리퍼 밑면 → 닫힌 손끝(m). 1/3 2F-85. 데모 상수와 분리(A L-N1): 데모는 M3 전용 값으로 바뀜


# ---------- 계산 (ROS 없이 시험 가능) ----------
def shank_radius(part_class: str) -> float:
    return SHANK_RADIUS.get(part_class, 0.003)


def tool_z(axis_z_m: float, finger_reach_m: float, radius: float = 0.003) -> float:
    """B 높이(축 중심 z) → 그리퍼 밑면 z. pick_bolt_demo와 같은 식: 받침면 + 여유 + 손끝 길이."""
    return axis_z_m - radius + TIP_CLEARANCE + finger_reach_m


def closing_vector(closing_yaw_deg: float) -> np.ndarray:
    """B 닫는 각도(deg, world) → 그리퍼 X(손가락이 닫히는 방향) 단위 벡터."""
    a = math.radians(closing_yaw_deg)
    return np.array([math.cos(a), math.sin(a), 0.0])


def tool_pose(xy, z: float, closing: np.ndarray) -> np.ndarray:
    """그리퍼 밑면 4×4 자세. 손가락 전방(+Z)이 아래, X = 닫는 방향 (pick_bolt_demo와 같은 모양)."""
    down = np.array([0.0, 0.0, -1.0])
    t = np.eye(4)
    t[:3, :3] = np.column_stack([closing, np.cross(down, closing), down])
    t[:3, 3] = [xy[0], xy[1], z]
    return t


def plan_targets(goal, finger_reach_m: float) -> list[dict]:
    """주문 → 후보별 목표(위치 m·닫는 방향·각 높이). 후보 순서 그대로."""
    out, r = [], shank_radius(goal.part_class)
    for c in goal.grasp_pose_candidates:
        x, y, z = (v / 1000.0 for v in c.position_mm)
        out.append({
            "xy": (x, y), "closing": closing_vector(c.closing_yaw_deg),
            "grasp": tool_z(z, finger_reach_m, r),
            "approach": tool_z(goal.approach_height_mm / 1000.0, finger_reach_m, r),
            "lift": tool_z(goal.lift_height_mm / 1000.0, finger_reach_m, r),
        })
    return out


def choose_candidate(targets, solve, home):
    """두 후보 중 접근 자세 역기구학이 되고 관절이 덜 도는 쪽. (목표, 접근 관절) 또는 None."""
    best = None
    for t in targets:
        try:
            q = solve(tool_pose(t["xy"], t["approach"], t["closing"]), home)
        except RuntimeError:
            continue
        cost = float(np.max(np.abs(np.asarray(q) - np.asarray(home))))
        if best is None or cost < best[0]:
            best = (cost, t, q)
    return None if best is None else (best[1], best[2])


# ---------- ROS 서버 ----------
def _load_demo():
    spec = importlib.util.spec_from_file_location("pick_bolt_demo", REPO / "scripts" / "pick_bolt_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class SimSortServer:
    def __init__(self, args):
        import rclpy
        from rclpy.action import ActionServer, CancelResponse, GoalResponse
        from rclpy.callback_groups import ReentrantCallbackGroup

        from eyetoarm_interfaces.action import SortPart

        self.rclpy, self.SortPart, self.args = rclpy, SortPart, args
        self.demo = _load_demo()
        self.chain = self.demo.load_chain()
        self.reach = args.finger_reach if args.finger_reach is not None else FINGER_REACH
        self.arm_node = rclpy.create_node("b_sim_sort_arm")       # 팔 전용 노드 (실행기 밖에서 직접 spin)
        self.arm = self.demo.Arm(self.arm_node)
        self.home = self.arm.joints()
        self.busy = threading.Lock()
        self.node = rclpy.create_node("b_sim_sort_part_server")
        self.server = ActionServer(
            self.node, SortPart, "/sort_part", self.execute, callback_group=ReentrantCallbackGroup(),
            goal_callback=lambda g: GoalResponse.REJECT if self.busy.locked() else GoalResponse.ACCEPT,
            cancel_callback=lambda h: CancelResponse.REJECT)
        self.log = self.node.get_logger()
        self.log.info(f"B 시험용 Gazebo /sort_part 서버 대기 중 (로봇을 실제로 움직임, Ctrl+C 종료) "
                      f"finger_reach={self.reach:.4f} m. 물리 엔진은 DART 권장")

    # 직선 위·아래 이동 (pick_bolt_demo.line과 같은 방식)
    def _line(self, q0, xy, z0, z1, closing, steps=25):
        path, q = [], q0
        for k in range(1, steps + 1):
            q = self.demo.solve_ik(self.chain, tool_pose(xy, z0 + (z1 - z0) * k / steps, closing), q)
            path.append(q)
        return path

    def execute(self, handle):
        g = handle.request
        res = self.SortPart.Result()
        stage = "START"                                            # 실패 때 어느 단계였는지 기록(A L-N1)

        def fb(s):
            handle.publish_feedback(self.SortPart.Feedback(stage=s))

        def step(name):
            nonlocal stage
            stage = name

        if not self.busy.acquire(blocking=False):
            handle.abort()
            res.status, res.reason = "FAILED", "BUSY"
            return res
        try:
            self.log.info(f"[주문] {g.part_class} {g.goal_kind} id={g.goal_id[:8]} 후보 {len(g.grasp_pose_candidates)}개")
            for c in g.grasp_pose_candidates:
                self.log.info(f"  grasp_mm=({c.position_mm[0]:.1f}, {c.position_mm[1]:.1f}, {c.position_mm[2]:.1f}) yaw={c.closing_yaw_deg:.1f} reach={self.reach:.4f}")
            fb("ACCEPTED")
            if g.goal_kind == "PASS_THROUGH":                     # 너트: 잡지 않음
                handle.succeed()
                res.status = "SUCCEEDED"
                return res

            fb("PLANNING")
            picked = choose_candidate(plan_targets(g, self.reach), lambda T, s: self.demo.solve_ik(self.chain, T, s),
                                      self.home)
            if picked is None:
                handle.abort()
                res.status, res.reason = "FAILED", "IK_NO_SOLUTION"
                return res
            t, above = picked
            down = self._line(above, t["xy"], t["approach"], t["grasp"], t["closing"])
            up = self._line(down[-1], t["xy"], t["grasp"], t["lift"], t["closing"])

            px, py, pz = (v / 1000.0 for v in g.place_position_mm)
            r = shank_radius(g.part_class)
            place_z = tool_z(pz, self.reach, r) + r   # place height
            over_bin = self.demo.solve_ik(self.chain, tool_pose((px, py), t["lift"], t["closing"]), up[-1])
            to_bin = self._line(over_bin, (px, py), t["lift"], place_z, t["closing"])

            fb("EXECUTING")
            step("OPEN")
            self.arm.grip(self.demo.GRIP_OPEN)
            step("MOVE_ABOVE")
            self.arm.follow([above], self.args.move_s)
            step("DESCEND")
            self.arm.follow(down, self.args.move_s * 0.75)
            step("CLOSE")
            self.arm.grip(self.demo.GRIP_CLOSED)
            knuckle = self.arm.state[self.demo.GRIPPER_JOINT]
            if knuckle > self.demo.GRIP_CLOSED * MISS_RATIO:          # 끝까지 닫힘 = 손가락 사이에 아무것도 없음
                self.arm.grip(self.demo.GRIP_OPEN)
                self.arm.follow([above, self.home], self.args.move_s)
                handle.abort()
                res.status, res.reason = "FAILED", "MISSED"
                self.log.info(f"[결과] 못 잡음 (그리퍼 {knuckle:.3f} rad)")
                return res

            step("LIFT")
            self.arm.follow(up, self.args.move_s * 0.75)
            fb("PLACING")
            step("MOVE_BIN")
            self.arm.follow([over_bin], self.args.move_s * 1.5)
            self.arm.follow(to_bin, self.args.move_s * 0.75)
            step("RELEASE")
            self.arm.release()
            step("RETURN")
            self.arm.follow([over_bin, self.home], self.args.move_s * 1.5)
            res.arrived_stamp_ns = self.node.get_clock().now().nanoseconds
            handle.succeed()
            res.status = "SUCCEEDED"
            self.log.info(f"[결과] {g.part_class} → {g.place_bin} 상자 넣기 완료")
            return res
        except Exception as exc:                                   # 역기구학 실패·제어기 오류
            self.log.error(f"[결과] 실패 (단계 {stage}): {type(exc).__name__}: {exc}")
            try:
                self.arm.follow([self.home], self.args.move_s)
            except Exception:
                pass
            handle.abort()
            res.status, res.reason = "FAILED", f"EXEC_ERROR:{stage}:{type(exc).__name__}"
            return res
        finally:
            self.busy.release()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="B 시험용 Gazebo /sort_part 서버 (로봇 실제 동작)")
    p.add_argument("--finger-reach", type=float, default=None, help="손끝 길이(m). 기본 0.0543 (데모 상수와 분리)")
    p.add_argument("--move-s", type=float, default=3.0, help="구간별 이동 시간(초)")
    args = p.parse_args(argv)

    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    rclpy.init()
    srv = SimSortServer(args)
    ex = MultiThreadedExecutor(num_threads=2)
    ex.add_node(srv.node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        ex.shutdown()
        srv.node.destroy_node()
        srv.arm_node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())