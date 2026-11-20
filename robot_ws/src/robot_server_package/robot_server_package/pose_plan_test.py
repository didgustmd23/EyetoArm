"""Plan-only MoveIt test for a B grasp-centre pose.

This utility converts a grasp-centre pose expressed in millimetres to a tool0
pose, then submits a MoveGroup goal with ``plan_only=True``. It never executes
the returned trajectory. The tool0-to-G translation must be supplied from the
currently selected gripper configuration; the measured 0.4 rad midpoint offset
is only a provisional value, not a calibrated contact-point transform.
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Sequence

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
)
from moveit_msgs.srv import GetPositionIK
from rclpy.action import ActionClient
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive


def _normalize_quaternion_xyzw(values: Sequence[float]) -> tuple[float, float, float, float]:
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("orientation must contain four finite xyzw values")
    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1e-12:
        raise ValueError("orientation quaternion cannot be zero")
    x, y, z, w = (value / norm for value in values)
    # q and -q represent the same rotation; keep the representation deterministic.
    if w < 0:
        x, y, z, w = -x, -y, -z, -w
    return x, y, z, w


def _rotate_vector_xyzw(
    quaternion: tuple[float, float, float, float], vector: tuple[float, float, float]
) -> tuple[float, float, float]:
    x, y, z, w = quaternion
    vx, vy, vz = vector
    # Equivalent to q * [v, 0] * conjugate(q), without constructing matrices.
    uv = (y * vz - z * vy, z * vx - x * vz, x * vy - y * vx)
    uuv = (y * uv[2] - z * uv[1], z * uv[0] - x * uv[2], x * uv[1] - y * uv[0])
    return tuple(vector[i] + 2.0 * (w * uv[i] + uuv[i]) for i in range(3))


def grasp_to_tool0_pose(
    position_mm: Sequence[float],
    orientation_xyzw: Sequence[float],
    tool0_to_g_xyz_m: Sequence[float],
) -> Pose:
    """Convert ``T_base_G`` to ``T_base_tool0`` using ``T_tool0_G``.

    ``G`` and ``tool0`` are assumed to have aligned axes. The origin offset is
    expressed in tool0-local metres. Position input is converted mm -> m once.
    """
    if len(position_mm) != 3 or not all(math.isfinite(value) for value in position_mm):
        raise ValueError("grasp position must contain three finite millimetre values")
    if len(tool0_to_g_xyz_m) != 3 or not all(math.isfinite(value) for value in tool0_to_g_xyz_m):
        raise ValueError("tool0-to-G offset must contain three finite metre values")

    q = _normalize_quaternion_xyzw(orientation_xyzw)
    grasp_position_m = tuple(value / 1000.0 for value in position_mm)
    offset_in_base = _rotate_vector_xyzw(q, tuple(tool0_to_g_xyz_m))

    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (
        grasp_position_m[i] - offset_in_base[i] for i in range(3)
    )
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
    return pose


def _constraints(
    *,
    frame: str,
    link: str,
    pose: Pose,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
) -> Constraints:
    if position_tolerance_m <= 0 or orientation_tolerance_rad <= 0:
        raise ValueError("planning tolerances must be positive")

    region = BoundingVolume()
    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = [2.0 * position_tolerance_m] * 3
    region.primitives.append(box)
    region.primitive_poses.append(Pose())
    region.primitive_poses[0].position = pose.position
    region.primitive_poses[0].orientation.w = 1.0

    position = PositionConstraint()
    position.header.frame_id = frame
    position.link_name = link
    position.constraint_region = region
    position.weight = 1.0

    orientation = OrientationConstraint()
    orientation.header.frame_id = frame
    orientation.link_name = link
    orientation.orientation = pose.orientation
    orientation.absolute_x_axis_tolerance = orientation_tolerance_rad
    orientation.absolute_y_axis_tolerance = orientation_tolerance_rad
    orientation.absolute_z_axis_tolerance = orientation_tolerance_rad
    orientation.weight = 1.0

    result = Constraints()
    result.position_constraints.append(position)
    result.orientation_constraints.append(orientation)
    return result


class PosePlanTest(Node):
    def __init__(self, args: argparse.Namespace, pose: Pose) -> None:
        super().__init__("pose_plan_test")
        self._args = args
        self._pose = pose
        self._client = ActionClient(self, MoveGroup, args.action)
        self._ik_client = self.create_client(GetPositionIK, args.ik_service)

    def _check_ik(self, *, avoid_collisions: bool) -> bool | None:
        if not self._ik_client.wait_for_service(timeout_sec=self._args.server_timeout_sec):
            self.get_logger().warning(f"IK service {self._args.ik_service} unavailable; skipping IK precheck")
            return None

        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name = self._args.group
        ik.ik_link_name = self._args.link
        ik.pose_stamped.header.frame_id = self._args.frame
        ik.pose_stamped.header.stamp = self.get_clock().now().to_msg()
        ik.pose_stamped.pose = self._pose
        ik.robot_state.is_diff = True
        ik.avoid_collisions = avoid_collisions
        ik.timeout.sec = 1

        future = self._ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self._args.result_timeout_sec)
        if not future.done() or future.result() is None:
            self.get_logger().warning("Timed out waiting for IK precheck")
            return None

        response = future.result()
        code = response.error_code.val
        label = "with collision checking" if avoid_collisions else "without collision checking"
        if code == MoveItErrorCodes.SUCCESS:
            state = response.solution.joint_state
            pairs = ", ".join(
                f"{name}={position:.3f}"
                for name, position in zip(state.name, state.position)
            )
            self.get_logger().info(f"IK {label}: SUCCESS; solution [{pairs}]")
            return True

        self.get_logger().warning(f"IK {label}: error_code={code}")
        return False

    def run_plan(self) -> int:
        self.get_logger().info(f"Waiting for MoveGroup action {self._args.action}")
        if not self._client.wait_for_server(timeout_sec=self._args.server_timeout_sec):
            self.get_logger().error("MoveGroup action server was not available")
            return 2

        ik_without_collision = self._check_ik(avoid_collisions=False)
        ik_with_collision = self._check_ik(avoid_collisions=True)
        if ik_without_collision is False:
            self.get_logger().warning(
                "No IK solution found even without collision checking; inspect target pose, frame, and tool0-to-G offset"
            )
        elif ik_without_collision is True and ik_with_collision is False:
            self.get_logger().warning(
                "An IK solution exists, but collision checking rejected the tested solution(s); inspect Planning Scene"
            )

        goal = MoveGroup.Goal()
        request = goal.request
        request.group_name = self._args.group
        # An empty, non-diff RobotState is interpreted as an explicit empty
        # joint state by MoveIt. Mark it diff so the planning scene's current
        # robot state is used as the start state.
        request.start_state.is_diff = True
        request.num_planning_attempts = self._args.attempts
        request.allowed_planning_time = self._args.planning_time_sec
        request.max_velocity_scaling_factor = self._args.velocity_scaling
        request.max_acceleration_scaling_factor = self._args.acceleration_scaling
        request.goal_constraints.append(
            _constraints(
                frame=self._args.frame,
                link=self._args.link,
                pose=self._pose,
                position_tolerance_m=self._args.position_tolerance_m,
                orientation_tolerance_rad=self._args.orientation_tolerance_rad,
            )
        )
        goal.planning_options.plan_only = True
        goal.planning_options.replan = False

        self.get_logger().info(
            "Planning only: "
            f"{self._args.link} in {self._args.frame}, "
            f"position=({self._pose.position.x:.4f}, {self._pose.position.y:.4f}, "
            f"{self._pose.position.z:.4f}) m, "
            f"orientation_xyzw=({self._pose.orientation.x:.4f}, "
            f"{self._pose.orientation.y:.4f}, {self._pose.orientation.z:.4f}, "
            f"{self._pose.orientation.w:.4f})"
        )
        send_future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=self._args.result_timeout_sec)
        if not send_future.done() or send_future.result() is None:
            self.get_logger().error("Timed out waiting for MoveGroup to accept the planning goal")
            return 3

        handle = send_future.result()
        if not handle.accepted:
            self.get_logger().error("MoveGroup rejected the planning goal")
            return 4

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=self._args.result_timeout_sec)
        if not result_future.done() or result_future.result() is None:
            self.get_logger().error("Timed out waiting for the planning result")
            return 5

        wrapped = result_future.result()
        result = wrapped.result
        code = result.error_code.val
        point_count = len(result.planned_trajectory.joint_trajectory.points)
        joint_names = result.planned_trajectory.joint_trajectory.joint_names
        self.get_logger().info(
            f"MoveIt error_code={code} ({'SUCCESS' if code == MoveItErrorCodes.SUCCESS else 'not successful'}), "
            f"planning_time={result.planning_time:.3f}s, "
            f"trajectory_points={point_count}, joints={joint_names}"
        )
        return 0 if code == MoveItErrorCodes.SUCCESS and point_count > 0 else 1


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a B grasp-centre pose to tool0 and request a MoveIt plan only."
    )
    parser.add_argument("--position-mm", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--orientation-xyzw", nargs=4, type=float, required=True, metavar=("X", "Y", "Z", "W")
    )
    parser.add_argument(
        "--tool0-to-g-xyz-m",
        nargs=3,
        type=float,
        required=True,
        metavar=("X", "Y", "Z"),
        help="provisional or calibrated tool0-local offset to grasp centre G, in metres",
    )
    parser.add_argument("--frame", default="base_link", help="frame of the incoming B grasp pose")
    parser.add_argument("--link", default="tool0", help="MoveIt link to constrain")
    parser.add_argument("--group", default="ur_manipulator")
    parser.add_argument("--action", default="/move_action")
    parser.add_argument("--ik-service", default="/compute_ik")
    parser.add_argument("--position-tolerance-m", type=float, default=0.005)
    parser.add_argument("--orientation-tolerance-rad", type=float, default=0.05)
    parser.add_argument("--planning-time-sec", type=float, default=5.0)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--velocity-scaling", type=float, default=0.1)
    parser.add_argument("--acceleration-scaling", type=float, default=0.1)
    parser.add_argument("--server-timeout-sec", type=float, default=10.0)
    parser.add_argument("--result-timeout-sec", type=float, default=30.0)
    return parser.parse_args(argv)


def main(args: Sequence[str] | None = None) -> None:
    options = _arguments(args)
    try:
        pose = grasp_to_tool0_pose(
            options.position_mm,
            options.orientation_xyzw,
            options.tool0_to_g_xyz_m,
        )
    except ValueError as error:
        print(f"Invalid pose: {error}", file=sys.stderr)
        raise SystemExit(2) from error

    # argparse already consumed the application arguments; do not pass them to
    # ROS argument parsing (e.g. --position-mm is not a ROS remapping option).
    rclpy.init(args=[])
    node = PosePlanTest(options, pose)
    try:
        exit_code = node.run_plan()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(exit_code)


__all__ = ["PosePlanTest", "grasp_to_tool0_pose", "main"]
