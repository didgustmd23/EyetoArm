"""B 시험용 가짜 C 서버 — `/sort_part`를 받아 진행·결과만 돌려준다 (로봇 안 움직임).

C 액션 서버가 나오기 전 B 클라이언트(`transforms/sort_part_client.py`)를 ROS에서 확인하는 용도다.
C 서버가 나오면 쓰지 않는다.

    source /opt/ros/jazzy/setup.bash && source robot_ws/install/setup.bash
    python3 scripts/b_fake_sort_part_server.py [--step-s 0.5] [--fail]
"""
import argparse
import time

import rclpy
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor

from eyetoarm_interfaces.action import SortPart

STAGES = ("ACCEPTED", "PLANNING", "EXECUTING", "PLACING")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--step-s", type=float, default=0.5, help="단계 사이 시간(s)")
    ap.add_argument("--fail", action="store_true", help="결과를 FAILED로")
    a = ap.parse_args()

    rclpy.init()
    node = rclpy.create_node("b_fake_sort_part_server")

    def execute(handle):
        g = handle.request
        p = g.grasp_pose_candidates[0].position_mm if g.grasp_pose_candidates else g.observed_position_mm
        node.get_logger().info(f"받음 {g.goal_kind} {g.part_class} 집는점={[round(v, 1) for v in p]} "
                               f"상자={g.place_bin or '-'} id={g.goal_id[:8]}")
        for s in STAGES:
            handle.publish_feedback(SortPart.Feedback(stage=s))
            time.sleep(a.step_s)
        if a.fail:
            handle.abort()
            return SortPart.Result(status="FAILED", reason="FAKE_FAIL")
        handle.succeed()
        return SortPart.Result(status="SUCCEEDED", reason="")

    ActionServer(node, SortPart, "/sort_part", execute)
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    node.get_logger().info("가짜 /sort_part 서버 대기 중 (Ctrl+C 종료)")
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
