"""Re-spawn ros2_control controllers after a Gazebo world reset."""

from __future__ import annotations

import subprocess
import threading
import time

import rclpy
from controller_manager_msgs.srv import ListControllers
from rclpy.node import Node


CONTROLLERS = (
    "joint_state_broadcaster",
    "joint_trajectory_controller",
    "gripper_controller",
)


class ControllerRecovery(Node):
    """Keep the three simulation controllers active across Gazebo resets."""

    def __init__(self) -> None:
        super().__init__("controller_recovery")
        self._list = self.create_client(ListControllers, "/controller_manager/list_controllers")
        self._repairing: set[str] = set()
        self._lock = threading.Lock()
        # The launch file starts the normal spawners in parallel.  Give them
        # time to finish before the recovery monitor takes ownership.
        self._monitor_after = time.monotonic() + 15.0
        self.create_timer(2.0, self._check)
        self.get_logger().info("controller recovery monitor started")

    def _check(self) -> None:
        if time.monotonic() < self._monitor_after:
            return
        if not self._list.service_is_ready():
            return
        future = self._list.call_async(ListControllers.Request())
        future.add_done_callback(self._handle_list)

    def _handle_list(self, future) -> None:
        try:
            controllers = future.result().controller
        except Exception as exc:  # service can disappear while Gazebo resets
            self.get_logger().debug(f"controller list unavailable: {exc}")
            return
        states = {item.name: item.state for item in controllers}
        for name in CONTROLLERS:
            if states.get(name) == "active":
                continue
            with self._lock:
                if name in self._repairing:
                    continue
                self._repairing.add(name)
            self.get_logger().warning(
                f"controller {name} is {states.get(name, 'missing')}; respawning")
            threading.Thread(target=self._spawn, args=(name,), daemon=True).start()

    def _spawn(self, name: str) -> None:
        try:
            result = subprocess.run(
                ["ros2", "run", "controller_manager", "spawner", name,
                 "--controller-manager", "/controller_manager",
                 "--controller-manager-timeout", "30"],
                check=False, capture_output=True, text=True, timeout=40,
            )
            if result.returncode != 0:
                self.get_logger().warning(
                    f"controller {name} respawn failed ({result.returncode}): "
                    f"{result.stderr.strip()[-240:]}")
            else:
                self.get_logger().info(f"controller {name} active after reset")
        except Exception as exc:
            self.get_logger().warning(f"controller {name} respawn error: {exc}")
        finally:
            with self._lock:
                self._repairing.discard(name)


def main(args=None) -> int:
    rclpy.init(args=args)
    node = ControllerRecovery()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
