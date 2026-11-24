"""ROS clients for the managed sort-run controls exposed through TCP 1.2."""
from __future__ import annotations

from threading import Event, Thread


class RunControlClient:
    """Own a small ROS executor in a background thread and expose blocking calls."""

    def __init__(self, node_name: str = "tcp_run_control") -> None:
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from std_srvs.srv import Trigger
        from eyetoarm_interfaces.srv import GetSortRun

        self._rclpy = rclpy
        self._context = Context()
        rclpy.init(context=self._context)
        self._node = Node(node_name, context=self._context)
        self._executor = MultiThreadedExecutor(context=self._context, num_threads=2)
        self._executor.add_node(self._node)
        self._get_run = self._node.create_client(GetSortRun, "/c_sort_goal_receiver/get_run")
        self._services = {
            "START": self._node.create_client(Trigger, "/c_sort_goal_receiver/start_run"),
            "STOP": self._node.create_client(Trigger, "/c_sort_goal_receiver/stop_run"),
            "E_STOP": self._node.create_client(Trigger, "/c_sort_goal_receiver/e_stop"),
            "RECOVER": self._node.create_client(Trigger, "/c_sort_goal_receiver/recover"),
        }
        self._request_types = {"GET_RUN": GetSortRun, **{name: Trigger for name in self._services}}
        self._thread = Thread(target=self._executor.spin, name="tcp-run-control-ros", daemon=True)
        self._thread.start()

    def available(self, action: str) -> bool:
        client = self._get_run if action == "GET_RUN" else self._services.get(action)
        return bool(client and client.service_is_ready())

    def call(self, action: str, timeout_s: float = 8.0):
        client = self._get_run if action == "GET_RUN" else self._services.get(action)
        if client is None or not client.wait_for_service(timeout_sec=min(timeout_s, 1.0)):
            raise TimeoutError(f"{action} service unavailable")
        request = self._request_types[action].Request()
        future = client.call_async(request)
        done = Event()
        future.add_done_callback(lambda _future: done.set())
        if not done.wait(timeout_s):
            raise TimeoutError(f"{action} service timed out")
        result = future.result()
        if result is None:
            raise RuntimeError(f"{action} returned no response")
        return result

    def close(self) -> None:
        if self._executor is None:
            return
        self._executor.shutdown(timeout_sec=1.0)
        self._node.destroy_node()
        self._context.try_shutdown()
        self._thread.join(timeout=1.0)
        self._executor = None

