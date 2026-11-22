"""Kinematic conveyor-motion prototype for Gazebo.

This moves the three demo part models along world Y with Gazebo SetEntityPose.
It is useful for integration experiments, but it is not a physical belt model:
parts do not roll or respond to contact forces, and grasp/attachment handling is
not included. The controller is stopped by default; choose a trial speed when
launching it and start it explicitly through ``~/set_running``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import TwistStamped
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from std_srvs.srv import SetBool


@dataclass
class PartPose:
    name: str
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float


class ConveyorController(Node):
    """Move demo parts kinematically in the configured belt direction."""

    def __init__(self) -> None:
        super().__init__("conveyor_controller")
        self.declare_parameter("speed_mps", 0.0)
        self.declare_parameter("update_rate_hz", 20.0)
        self.declare_parameter("min_y_m", -0.58)
        self.declare_parameter("max_y_m", 0.58)

        self.speed_mps = float(self.get_parameter("speed_mps").value)
        self.update_rate_hz = float(self.get_parameter("update_rate_hz").value)
        self.min_y_m = float(self.get_parameter("min_y_m").value)
        self.max_y_m = float(self.get_parameter("max_y_m").value)
        if not math.isfinite(self.speed_mps):
            raise ValueError("speed_mps must be finite")
        if not math.isfinite(self.update_rate_hz) or self.update_rate_hz <= 0:
            raise ValueError("update_rate_hz must be greater than zero")
        if not (math.isfinite(self.min_y_m) and math.isfinite(self.max_y_m)
                and self.min_y_m < self.max_y_m):
            raise ValueError("min_y_m must be finite and less than max_y_m")

        # Poses match simulation/worlds/sort_cell.sdf. Quaternion is xyzw.
        self.parts = [
            # M3x20 part_00 settles at z=0.102 m in sort_cell.sdf; keeping
            # the controller pose aligned prevents each update from lifting it.
            PartPose("part_00", 0.55, -0.30, 0.102, 0.0, 0.70710678, 0.0, 0.70710678),
            PartPose("part_01", 0.55, -0.15, 0.105, 0.0, 0.70710678, 0.0, 0.70710678),
            PartPose("part_02", 0.55, 0.00, 0.1005, 0.0, 0.0, 0.0, 1.0),
        ]
        self._running = False
        self._batch_in_flight = False
        self._last_clock_ns: int | None = None
        self._last_state_stamp_ns: int | None = None
        state_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                               reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._state_pub = self.create_publisher(TwistStamped, "/conveyor/state", state_qos)
        self._pose_client = self.create_client(SetEntityPose, "/world/sort_cell/set_pose")
        self._run_service = self.create_service(SetBool, "~/set_running", self._set_running)
        self._timer = self.create_timer(1.0 / self.update_rate_hz, self._tick)
        self._state_watchdog = self.create_timer(0.1, self._check_state_clock, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self._publish_state()
        self.get_logger().info(
            "Kinematic conveyor ready (stopped). Set speed_mps explicitly, then call "
            "'/conveyor_controller/set_running' with data=true."
        )

    def _set_running(self, request: SetBool.Request, response: SetBool.Response):
        if request.data and self.speed_mps == 0.0:
            response.success = False
            response.message = "speed_mps is zero; restart node with a nonzero trial speed"
            return response
        if request.data and not self._pose_client.service_is_ready():
            response.success = False
            response.message = "Gazebo /world/sort_cell/set_pose service is not ready"
            return response
        self._running = request.data
        self._last_clock_ns = self.get_clock().now().nanoseconds if self._running else None
        response.success = True
        direction = "+Y" if self.speed_mps > 0 else "-Y"
        response.message = (
            f"kinematic movement started at {abs(self.speed_mps):.4f} m/s toward {direction}"
            if self._running else "kinematic movement stopped"
        )
        self._publish_state()
        self.get_logger().info(response.message)
        return response

    def _publish_state(self) -> None:
        """Publish signed belt Y velocity stamped with the active ROS clock."""
        now = self.get_clock().now()
        msg = TwistStamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "world"
        # Signed Y velocity encodes direction; zero means stopped.
        msg.twist.linear.y = self.speed_mps if self._running else 0.0
        self._state_pub.publish(msg)
        self._last_state_stamp_ns = now.nanoseconds

    def _check_state_clock(self) -> None:
        """Republish the retained state after a SIM_TIME rewind."""
        now_ns = self.get_clock().now().nanoseconds
        if self._last_state_stamp_ns is not None and now_ns < self._last_state_stamp_ns:
            self._last_clock_ns = now_ns if self._running else None
            self._publish_state()
            self.get_logger().info("Simulation time rewind detected; conveyor state republished")

    def _tick(self) -> None:
        if not self._running or self._batch_in_flight:
            return
        now_ns = self.get_clock().now().nanoseconds
        if self._last_clock_ns is None:
            self._last_clock_ns = now_ns
            return
        dt = max(0.0, (now_ns - self._last_clock_ns) / 1_000_000_000.0)
        self._last_clock_ns = now_ns
        if dt == 0.0:
            return
        if not self._pose_client.service_is_ready():
            self._running = False
            self._last_clock_ns = None
            self._publish_state()
            self.get_logger().error("Gazebo set_pose service disappeared; movement stopped")
            return

        next_y = [part.y + self.speed_mps * dt for part in self.parts]
        if any(y < self.min_y_m or y > self.max_y_m for y in next_y):
            self._running = False
            self._last_clock_ns = None
            self._publish_state()
            self.get_logger().info("A part reached the configured belt boundary; movement stopped")
            return

        self._batch_in_flight = True
        remaining = len(self.parts)
        failed: list[str] = []

        def done_callback(part: PartPose, y: float, future) -> None:
            nonlocal remaining
            try:
                result = future.result()
                if result is None or not result.success:
                    failed.append(part.name)
                else:
                    part.y = y
            except Exception as exc:  # surface transport errors and stop safely
                failed.append(f"{part.name} ({exc})")
            remaining -= 1
            if remaining == 0:
                self._batch_in_flight = False
                if failed:
                    self._running = False
                    self._last_clock_ns = None
                    self._publish_state()
                    self.get_logger().error(
                        "Gazebo rejected pose update for " + ", ".join(failed) + "; movement stopped"
                    )

        for part, y in zip(self.parts, next_y):
            request = SetEntityPose.Request()
            request.entity = Entity()
            request.entity.name = part.name
            request.entity.type = Entity.MODEL
            request.pose.position.x = part.x
            request.pose.position.y = y
            request.pose.position.z = part.z
            request.pose.orientation.x = part.qx
            request.pose.orientation.y = part.qy
            request.pose.orientation.z = part.qz
            request.pose.orientation.w = part.qw
            future = self._pose_client.call_async(request)
            future.add_done_callback(lambda completed, item=part, target_y=y: done_callback(item, target_y, completed))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ConveyorController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
