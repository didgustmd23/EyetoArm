"""C-side receiver for B's /sort_part action goals.

This first integration step validates the ROS action identity, logs the received
payload, and returns an explicit non-success result until robot execution is
connected. It never reports a simulated receipt as a completed sort.
"""
from __future__ import annotations

import argparse
import uuid

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from eyetoarm_interfaces.action import SortPart


ACTION_NAME = "/sort_part"


def _goal_summary(goal) -> str:
    candidates = [
        {
            "position_mm": list(candidate.position_mm),
            "orientation_xyzw": list(candidate.orientation_xyzw),
            "closing_yaw_deg": candidate.closing_yaw_deg,
        }
        for candidate in goal.grasp_pose_candidates
    ]
    return (
        f"goal_id={goal.goal_id} kind={goal.goal_kind} sort_job_id={goal.sort_job_id} "
        f"run_epoch={goal.run_epoch} part={goal.part_class} track={goal.part_track_id} "
        f"source_id={goal.source_id} image_seq={goal.image_seq} capture_generation={goal.capture_generation} "
        f"generation={goal.generation} job_seq={goal.job_seq} capture_stamp_ns={goal.capture_stamp_ns} "
        f"timestamp_base={goal.timestamp_base} frame_id={goal.frame_id} "
        f"observed_position_mm={list(goal.observed_position_mm)} candidates={candidates} "
        f"opening_width_mm={goal.opening_width_mm} approach_height_mm={goal.approach_height_mm} "
        f"lift_height_mm={goal.lift_height_mm} place_bin={goal.place_bin} "
        f"place_position_mm={list(goal.place_position_mm)} calibration_id={goal.calibration_id} "
        f"valid_until_ns={goal.valid_until_ns}"
    )


class SortGoalReceiver(Node):
    """Receive and inspect B goals without dispatching robot motion."""

    def __init__(self, action_name: str = ACTION_NAME) -> None:
        super().__init__("c_sort_goal_receiver")
        self._server = ActionServer(
            self,
            SortPart,
            action_name,
            execute_callback=self._execute,
            goal_callback=self._goal_request,
            cancel_callback=self._cancel_request,
        )
        self.get_logger().info(
            f"B→C SortPart 수신 대기 {action_name}; 실행기는 연결되지 않아 목표를 수행하지 않습니다"
        )

    def _goal_request(self, request) -> GoalResponse:
        if request.goal_kind not in ("PICK_PLACE", "PASS_THROUGH"):
            self.get_logger().warning(f"goal 거부: unsupported goal_kind={request.goal_kind!r}")
            return GoalResponse.REJECT
        if not request.goal_id:
            self.get_logger().warning("goal 거부: goal_id가 비어 있습니다")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_request(self, _goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle):
        request = goal_handle.request
        try:
            ros_goal_id = str(uuid.UUID(bytes=bytes(goal_handle.goal_id.uuid)))
        except (ValueError, AttributeError):
            ros_goal_id = "<invalid>"
        self.get_logger().info(f"B_GOAL_RECEIVED ros_goal_id={ros_goal_id} {_goal_summary(request)}")

        feedback = SortPart.Feedback()
        feedback.stage = "ACCEPTED"
        goal_handle.publish_feedback(feedback)
        result = SortPart.Result()
        result.arrived_stamp_ns = 0

        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result.status = "CANCELED"
            result.reason = "CANCELED_BEFORE_EXECUTION"
            return result

        if request.goal_id != ros_goal_id:
            goal_handle.abort()
            result.status = "FAILED"
            result.reason = "GOAL_ID_MISMATCH"
            self.get_logger().error(
                f"B goal_id와 ROS goal UUID가 다릅니다: payload={request.goal_id!r} ros={ros_goal_id}"
            )
            return result

        goal_handle.abort()
        result.status = "FAILED"
        result.reason = "EXECUTION_NOT_CONFIGURED"
        self.get_logger().warning(
            f"B_GOAL_RECEIVED_ONLY goal_id={request.goal_id}: execution backend is not connected"
        )
        return result


def main(args=None) -> int:
    parser = argparse.ArgumentParser(description="Receive B SortPart goals without executing robot motion")
    parser.add_argument("--action", default=ACTION_NAME)
    parsed, ros_args = parser.parse_known_args(args)
    rclpy.init(args=ros_args)
    node = SortGoalReceiver(parsed.action)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
