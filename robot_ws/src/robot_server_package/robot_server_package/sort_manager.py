"""Receive B's /sort_part goals with optional Gazebo execution."""
from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
from threading import Lock
import uuid

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_srvs.srv import Trigger

from eyetoarm_interfaces.action import SortPart
from eyetoarm_interfaces.srv import GetSortRun
from .gazebo_sort_executor import POSE_SOURCES, TRACK_TOPIC, GazeboSortExecutor, SortCanceled, SortFailure


ACTION_NAME = "/sort_part"


def next_epoch(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, isolation_level=None) as database:
        database.execute("BEGIN IMMEDIATE")
        database.execute("CREATE TABLE IF NOT EXISTS sort_epoch (id INTEGER PRIMARY KEY, value INTEGER)")
        database.execute("INSERT OR IGNORE INTO sort_epoch VALUES (1, 0)")
        database.execute("UPDATE sort_epoch SET value = value + 1 WHERE id = 1")
        epoch = database.execute("SELECT value FROM sort_epoch WHERE id = 1").fetchone()[0]
        database.commit()
    return epoch


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
    """Validate goals and dispatch them to the configured executor."""

    def __init__(self, action_name: str = ACTION_NAME, backend=None, epoch_file=None) -> None:
        super().__init__("c_sort_goal_receiver")
        self._backend = backend
        self._execution_lock = Lock()
        self._epoch_file = epoch_file
        self._epoch = next_epoch(epoch_file) if epoch_file is not None else 0
        self._job_id = ""
        self._seen = set()
        self._seen_lock = Lock()
        self._recovery_required = False
        self._server = ActionServer(
            self,
            SortPart,
            action_name,
            execute_callback=self._execute,
            goal_callback=self._goal_request,
            cancel_callback=self._cancel_request,
            callback_group=ReentrantCallbackGroup(),
            result_timeout=3600,
        )
        if epoch_file is not None:
            self.create_service(Trigger, "~/start_run", self._start_run)
            self.create_service(GetSortRun, "~/get_run", self._get_run)
        self.get_logger().info(
            f"B→C SortPart {action_name}; mode={'gazebo' if backend is not None else 'receive-only'}"
        )

    def _start_run(self, request, response):
        if not self._execution_lock.acquire(blocking=False):
            response.message = "BUSY"
            return response
        try:
            if self._recovery_required:
                response.message = "RECOVERY_REQUIRED"
                return response
            if self._job_id:
                self._epoch = next_epoch(self._epoch_file)
            self._job_id = str(uuid.uuid4())
            with self._seen_lock:
                self._seen.clear()
            response.success = True
            response.message = "RUN_STARTED"
            return response
        finally:
            self._execution_lock.release()

    def _get_run(self, request, response):
        response.sort_job_id = self._job_id
        response.run_epoch = self._epoch
        response.active = bool(response.sort_job_id) and not self._recovery_required
        return response

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

        if self._backend is None:
            result.status, result.reason = "FAILED", "EXECUTION_NOT_CONFIGURED"
        elif not self._execution_lock.acquire(blocking=False):
            result.status, result.reason = "FAILED", "BUSY"
        else:
            def publish(stage):
                goal_handle.publish_feedback(SortPart.Feedback(stage=stage))

            try:
                if self._recovery_required:
                    raise SortFailure("RECOVERY_REQUIRED")
                if self._epoch_file is not None:
                    if not self._job_id or request.sort_job_id != self._job_id or request.run_epoch != self._epoch:
                        raise SortFailure("SORT_RUN_MISMATCH")
                    if request.timestamp_base != "SIM_TIME" or not 0 < request.capture_stamp_ns < request.valid_until_ns:
                        raise SortFailure("INVALID_GOAL_TIME")
                with self._seen_lock:
                    if request.goal_id in self._seen:
                        raise SortFailure("DUPLICATE_GOAL")
                    self._seen.add(request.goal_id)
                result.arrived_stamp_ns = self._backend.execute(
                    request, publish, lambda: goal_handle.is_cancel_requested)
                result.status = "SUCCEEDED"
                goal_handle.succeed()
                self.get_logger().info(f"B_GOAL_ARRIVED goal_id={request.goal_id} stamp={result.arrived_stamp_ns}")
                return result
            except SortCanceled:
                result.status, result.reason = "CANCELED", "ROBOT_STOP_CONFIRMED"
                goal_handle.canceled()
                return result
            except SortFailure as exc:
                result.reason = str(exc)
                result.status = "EXPIRED" if result.reason == "GOAL_EXPIRED" else "FAILED"
                if result.reason == "STOP_UNCONFIRMED":
                    self._recovery_required = True
                    if goal_handle.is_cancel_requested:
                        goal_handle.abort()
                        return result
            except Exception as exc:
                self.get_logger().error(f"SortPart execution error: {type(exc).__name__}: {exc}")
                result.status, result.reason = "FAILED", "EXECUTION_ERROR"
            finally:
                self._execution_lock.release()
        goal_handle.abort()
        self.get_logger().warning(f"SortPart goal={request.goal_id} status={result.status} reason={result.reason}")
        return result


def main(args=None) -> int:
    parser = argparse.ArgumentParser(description="Receive B SortPart goals")
    parser.add_argument("--action", default=ACTION_NAME)
    parser.add_argument("--gazebo-project-root", type=Path,
                        help="Enable sort_cell simulation execution using this EyetoArm checkout")
    parser.add_argument("--run-state", type=Path,
                        help="Enable managed sort runs and persist the execution epoch in this SQLite file")
    parser.add_argument("--moving-belt", action="store_true",
                        help="Enable moving target prediction in managed runs")
    parser.add_argument("--pose-source", choices=POSE_SOURCES, default="sim",
                        help="Target position source: sim (Gazebo poses) or track (A track stream)")
    parser.add_argument("--track-topic", default=TRACK_TOPIC,
                        help="A TrackStateArray topic for --pose-source track")
    parser.add_argument("--sim-truth-log", type=Path,
                        help="With --pose-source track, append track minus simulation pose lines (evaluation only)")
    parsed, ros_args = parser.parse_known_args(args)
    if parsed.run_state and not parsed.gazebo_project_root:
        parser.error("--run-state requires --gazebo-project-root")
    if parsed.moving_belt and not parsed.run_state:
        parser.error("--moving-belt requires --run-state")
    if parsed.pose_source == "track" and not parsed.gazebo_project_root:
        parser.error("--pose-source track requires --gazebo-project-root")
    if parsed.sim_truth_log and parsed.pose_source != "track":
        parser.error("--sim-truth-log requires --pose-source track")
    rclpy.init(args=ros_args)
    arm_node = None
    node = None
    executor = MultiThreadedExecutor(num_threads=2)
    try:
        backend = None
        if parsed.gazebo_project_root is not None:
            arm_node = Node("c_sort_gazebo_arm", parameter_overrides=[
                Parameter("use_sim_time", value=True),
            ])
            backend = GazeboSortExecutor(arm_node, parsed.gazebo_project_root.resolve(),
                                         moving=parsed.moving_belt, pose_source=parsed.pose_source,
                                         track_topic=parsed.track_topic, sim_truth_log=parsed.sim_truth_log)
        node = SortGoalReceiver(parsed.action, backend, parsed.run_state)
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        if node is not None:
            node.destroy_node()
        if arm_node is not None:
            arm_node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
