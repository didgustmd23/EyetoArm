"""Receive B's /sort_part goals with optional Gazebo execution."""
from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
import signal
import sys
import time
from threading import Condition, Event, Lock, Thread
import uuid

import rclpy
from rclpy.signals import SignalHandlerOptions
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_srvs.srv import Trigger

from eyetoarm_interfaces.action import SortPart
from eyetoarm_interfaces.srv import GetSortRun
from .gazebo_sort_executor import POSE_SOURCES, TRACK_TOPIC, GazeboSortExecutor, SortCanceled, SortFailure
from .action_server_guard import ActionServerConflict, acquire_action_server_guard
from .sort_event_log import SortEventLog


ACTION_NAME = "/sort_part"
STOP_CONFIRM_TIMEOUT_S = 35.0
VISION_DISCARD_TIMEOUT_S = 8.0
SHUTDOWN_WAIT_S = 20.0       # #394: on SIGINT/SIGTERM, wait this long (wall) for the running goal to send its result
SHUTDOWN_FLUSH_S = 1.0       # then keep spinning so the client's get_result request is answered (#374 G-C1b)
START_CHECK_PERIOD_S = 2.0   # #394: retry the start home check while the controllers are not up yet


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


def finish_goal(goal_handle, state, logger):
    """#394: send the terminal state (succeed/canceled/abort); sending can fail once the ROS context is
    closing (RCLError "feedback publisher is invalid", C item 1). Reported instead of raised. True if sent."""
    try:
        getattr(goal_handle, state)()
        return True
    except Exception as exc:
        logger.error(f"SortPart result not sent state={state}: {type(exc).__name__}: {exc}")
        return False


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

    def __init__(self, action_name: str = ACTION_NAME, backend=None, epoch_file=None, feeder=None, event_log=None) -> None:
        super().__init__("c_sort_goal_receiver")
        self._feeder = feeder
        self._stop_run_requested = Event()
        self._backend = backend
        self._event_log = event_log
        self._execution_lock = Lock()
        self._control_lock = Lock()
        self._epoch_file = epoch_file
        self._managed = epoch_file is not None
        self._epoch = next_epoch(epoch_file) if epoch_file is not None else 0
        self._job_id = ""
        self._run_paused = True
        self._estop_latched = False
        self._has_started = False
        self._purge_required = False
        self._seen = set()
        self._seen_lock = Lock()
        self._recovery_required = False
        self._shutdown = Event()      # #394: set by the first SIGINT/SIGTERM
        self._active = 0
        self._active_cond = Condition()
        self._home_checked = False   # #394: set by the first finished home check (node start or first goal)
        self._vision_discard_client = (self.create_client(
            Trigger, "/eyetoarm_live_sort_part/discard_detections",
            callback_group=ReentrantCallbackGroup()) if epoch_file is not None else None)
        self._server_guard = acquire_action_server_guard(self, action_name)
        try:
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
        except Exception:
            self._server_guard.close()
            self.destroy_node()
            raise
        if epoch_file is not None:
            self.create_service(Trigger, "~/start_run", self._start_run)
            self.create_service(Trigger, "~/stop_run", self._stop_run)
            self.create_service(Trigger, "~/e_stop", self._e_stop_run)
            self.create_service(Trigger, "~/recover", self._recover_run)
            if feeder is not None:
                # A managed run always starts with the belt stopped. START is the
                # only operation that enables both movement and part generation.
                feeder.set_running(False)
                self.create_timer(0.25, self._feed)
            self.create_service(GetSortRun, "~/get_run", self._get_run)
        self._start_timer = (self.create_timer(START_CHECK_PERIOD_S, self._start_check)
                             if backend is not None else None)
        self.get_logger().info(
            f"B→C SortPart {action_name}; mode={'gazebo' if backend is not None else 'receive-only'}"
        )

    def _latch_if_holding(self, why):
        """#394 (A decision 2, separable): at home with a part still held, latch like an e-stop and pause,
        so the part is released only through C's ~/recover after a person checks (no automatic opening at
        the next goal). Returns True when it latched."""
        if not getattr(self._backend, "holding_object", False):
            return False
        self._estop_latched = True
        self._run_paused = True
        self.get_logger().warning(f"SortPart holding a part after {why}: E-STOP latched, use ~/recover")
        if self._event_log:
            self._event_log.record("holding_latched", reason=why)
        return True

    def _home_check(self, why):
        """#394 (C item 9②): after a (re)start the arm may be in a carry pose where the link camera cannot
        see the belt. If it is not at the fixed home, use C's move_home (gripper untouched). A failed move
        locks like C's stop_run home failure. Raises RuntimeError while the controllers are not up."""
        if self._backend.at_home():
            self.get_logger().info(f"SortPart home check ({why}) AT_HOME")
        else:
            try:
                self._backend.move_home()
            except SortFailure as exc:
                self._recovery_required = True
                self.get_logger().error(f"SortPart home check ({why}) HOME_FAILED: {exc}")
                self._home_checked = True
                return False
            self.get_logger().info(f"SortPart home check ({why}) MOVED_HOME")
        self._home_checked = True
        self._backend.refresh_hold_from_joint()
        self._latch_if_holding(f"home check ({why})")
        if self._start_timer is not None:
            self._start_timer.cancel()
        return True

    def _start_check(self):
        """Once after start (retried until the controllers answer); skipped while a goal or control call runs."""
        if self._home_checked:
            self._start_timer.cancel()
            return
        if not self._execution_lock.acquire(blocking=False):
            return
        try:
            self._home_check("node start")
        except SortFailure as exc:    # e.g. STOP_UNCONFIRMED from at_home's arm check: lock like a HOME failure
            self._recovery_required = True
            self._home_checked = True
            self._start_timer.cancel()
            self.get_logger().error(f"SortPart start home check failed: {exc}")
        except RuntimeError as exc:   # controllers not available yet: try again
            self.get_logger().info(f"SortPart start home check waiting: {exc}")
        finally:
            self._execution_lock.release()

    def destroy_node(self):
        try:
            return super().destroy_node()
        finally:
            guard = getattr(self, "_server_guard", None)
            if guard is not None:
                guard.close()
                self._server_guard = None

    def _feed(self):
        if self._job_id and not self._run_paused and not self._recovery_required:
            try:
                self._feeder.tick()
            except Exception as exc:
                self.get_logger().error(f"Part supply failed: {exc}")
                self._stop_run(None, Trigger.Response())

    def _stop_run(self, request, response):
        with self._control_lock:
            stopped, reason = self._pause_run()
            if not stopped:
                self._estop_latched = True
                response.success = False
                response.message = reason
                return response
            try:
                if self._backend is None:
                    raise SortFailure("EXECUTION_NOT_CONFIGURED")
                self._backend.move_home()
            except Exception as exc:
                self._recovery_required = True
                self._estop_latched = True
                response.success = False
                response.message = f"HOME_FAILED: {type(exc).__name__}: {exc}"
                return response
            if self._latch_if_holding("stop_run"):
                response.success = False
                response.message = "HOLDING_OBJECT_AT_HOME: E_STOP_LATCHED_USE_RECOVER"
                return response
            discarded, detail = self._discard_vision_detections("STOP")
            if not discarded:
                self._purge_required = True
                response.success = False
                response.message = f"A_B_DISCARD_FAILED: {detail}"
                return response
            self._job_id = ""
            self._estop_latched = False
            self._recovery_required = False
            self._purge_required = False
            self._run_paused = True
            response.success = True
            response.message = "STOPPED_HOME_DATA_DISCARDED"
            return response

    def _e_stop_run(self, _request, response):
        with self._control_lock:
            self._estop_latched = True
            stopped, reason = self._pause_run()
            response.success = stopped
            response.message = "E_STOPPED_IN_PLACE" if stopped else reason
            return response

    def _pause_run(self):
        """Stop part spawning/belt and wait for the current robot goal to stop in place."""
        self._stop_run_requested.set()
        self._run_paused = True
        if self._feeder is not None:
            try:
                self._feeder.set_running(False)
            except Exception as exc:
                self._recovery_required = True
                return False, f"BELT_STOP_FAILED: {type(exc).__name__}: {exc}"
        acquired = self._execution_lock.acquire(timeout=STOP_CONFIRM_TIMEOUT_S)
        if not acquired:
            self._recovery_required = True
            return False, "STOP_UNCONFIRMED"
        self._execution_lock.release()
        return True, "STOPPED_IN_PLACE"

    def _discard_vision_detections(self, reason):
        client = self._vision_discard_client
        if client is None or not client.wait_for_service(timeout_sec=2.0):
            return False, "A/B discard service unavailable"
        future = client.call_async(Trigger.Request())
        completed = Event()
        future.add_done_callback(lambda _future: completed.set())
        if not completed.wait(VISION_DISCARD_TIMEOUT_S):
            return False, "A/B discard timed out"
        try:
            reply = future.result()
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
        if reply is None or not reply.success:
            return False, "A/B discard refused" if reply is None else reply.message
        self.get_logger().info(f"A/B detections discarded reason={reason}: {reply.message}")
        return True, reply.message

    def _start_run(self, request, response):
        with self._control_lock:
            if not self._execution_lock.acquire(blocking=False):
                response.message = "BUSY"
                return response
            try:
                if self._recovery_required:
                    response.message = "RECOVERY_REQUIRED"
                    return response
                if self._purge_required:
                    response.message = "A_B_DISCARD_REQUIRED"
                    return response
                if self._job_id and not self._run_paused:
                    response.success = True
                    response.message = "ALREADY_RUNNING"
                    return response
                resuming = bool(self._job_id)
                if self._feeder is not None:
                    self._feeder.set_running(True)
                self._stop_run_requested.clear()
                if not self._job_id:
                    if self._has_started or self._seen:
                        self._epoch = next_epoch(self._epoch_file)
                    self._job_id = str(uuid.uuid4())
                    with self._seen_lock:
                        self._seen.clear()
                    self._has_started = True
                self._run_paused = False
                self._estop_latched = False
                response.success = True
                response.message = "RUN_RESUMED" if resuming else "RUN_STARTED"
                return response
            finally:
                self._execution_lock.release()

    def _recover_run(self, _request, response):
        """E-stop recovery: release a held object at the current pose, then home and purge A/B."""
        with self._control_lock:
            if not self._estop_latched:
                response.message = "E_STOP_REQUIRED"
                return response
            if self._backend is None:
                response.message = "EXECUTION_NOT_CONFIGURED"
                return response
            if not self._execution_lock.acquire(blocking=False):
                response.message = "BUSY"
                return response
            try:
                self._stop_run_requested.set()
                try:
                    self._backend.recover_home()
                except Exception as exc:
                    self._recovery_required = True
                    response.message = f"RECOVERY_FAILED: {type(exc).__name__}: {exc}"
                    return response
                discarded, detail = self._discard_vision_detections("RECOVER")
                if not discarded:
                    self._purge_required = True
                    response.message = f"A_B_DISCARD_FAILED: {detail}"
                    return response
                self._job_id = ""
                self._recovery_required = False
                self._run_paused = True
                self._estop_latched = False
                self._purge_required = False
                response.success = True
                response.message = "OBJECT_RELEASED_HOME_REACHED_DATA_DISCARDED"
                return response
            finally:
                self._execution_lock.release()

    def _get_run(self, request, response):
        response.sort_job_id = self._job_id
        response.run_epoch = self._epoch
        response.active = bool(response.sort_job_id) and not self._run_paused and not self._recovery_required
        response.purge_required = self._purge_required
        return response

    def _goal_request(self, request) -> GoalResponse:
        if self._shutdown.is_set():
            self.get_logger().warning("goal 거부: shutting down")
            return GoalResponse.REJECT
        if self._managed and (self._run_paused or self._recovery_required or not self._job_id):
            self.get_logger().warning("goal 거부: managed sort run is not active")
            return GoalResponse.REJECT
        if request.goal_kind not in ("PICK_PLACE", "PASS_THROUGH"):
            self.get_logger().warning(f"goal 거부: unsupported goal_kind={request.goal_kind!r}")
            return GoalResponse.REJECT
        if not request.goal_id:
            self.get_logger().warning("goal 거부: goal_id가 비어 있습니다")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_request(self, _goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def begin_shutdown(self, executor, wait_s=SHUTDOWN_WAIT_S, flush_s=SHUTDOWN_FLUSH_S):
        """#394 (C items 1·9①): first signal: block new goals, stop the belt, let the running goal stop
        in place (C's pause flag) and send its result, then stop the executor. No home move here: the
        restart home check does it. A goal still running after wait_s is only logged."""
        if self._shutdown.is_set():
            return
        self._shutdown.set()
        self._stop_run_requested.set()
        self._run_paused = True
        if self._feeder is not None:
            try:
                self._feeder.set_running(False)
            except Exception as exc:
                self.get_logger().error(f"Belt stop on shutdown failed: {exc}")
        self.get_logger().info("SortPart shutdown requested: finishing the running goal")

        def wait_then_stop():
            with self._active_cond:
                finished = self._active_cond.wait_for(lambda: self._active == 0, timeout=wait_s)
            if not finished:
                self.get_logger().error(f"SortPart shutdown: goal still running after {wait_s:g} s")
            time.sleep(flush_s)   # the result reaches the client only when its get_result is answered
            executor.shutdown()

        Thread(target=wait_then_stop, daemon=True).start()

    def _execute(self, goal_handle):
        """Count running goals for the shutdown wait (#394), then run the C goal callback."""
        with self._active_cond:
            self._active += 1
        try:
            return self._execute_goal(goal_handle)
        finally:
            with self._active_cond:
                self._active -= 1
                self._active_cond.notify_all()

    def _execute_goal(self, goal_handle):
        request = goal_handle.request
        try:
            ros_goal_id = str(uuid.UUID(bytes=bytes(goal_handle.goal_id.uuid)))
        except (ValueError, AttributeError):
            ros_goal_id = "<invalid>"
        self.get_logger().info(f"B_GOAL_RECEIVED ros_goal_id={ros_goal_id} {_goal_summary(request)}")
        if self._event_log:
            self._event_log.record("goal_received", goal_id=request.goal_id, ros_goal_id=ros_goal_id,
                                   part_class=request.part_class, goal_kind=request.goal_kind,
                                   track_id=request.part_track_id, frame_id=request.frame_id,
                                   timestamp_base=request.timestamp_base,
                                   observed_position_mm=list(request.observed_position_mm),
                                   grasp_candidates=[{"position_mm": list(c.position_mm),
                                                      "orientation_xyzw": list(c.orientation_xyzw),
                                                      "closing_yaw_deg": c.closing_yaw_deg}
                                                     for c in request.grasp_pose_candidates],
                                   opening_width_mm=request.opening_width_mm,
                                   approach_height_mm=request.approach_height_mm,
                                   lift_height_mm=request.lift_height_mm, place_bin=request.place_bin,
                                   place_position_mm=list(request.place_position_mm),
                                   calibration_id=request.calibration_id)

        def record_result(status, reason="", **fields):
            if self._event_log:
                self._event_log.record("goal_result", goal_id=request.goal_id, part_class=request.part_class,
                                       status=status, reason=reason, **fields)

        feedback = SortPart.Feedback()
        feedback.stage = "ACCEPTED"
        goal_handle.publish_feedback(feedback)
        result = SortPart.Result()
        result.arrived_stamp_ns = 0

        if goal_handle.is_cancel_requested:
            finish_goal(goal_handle, "canceled", self.get_logger())
            result.status = "CANCELED"
            result.reason = "CANCELED_BEFORE_EXECUTION"
            record_result(result.status, result.reason)
            return result

        if request.goal_id != ros_goal_id:
            finish_goal(goal_handle, "abort", self.get_logger())
            result.status = "FAILED"
            result.reason = "GOAL_ID_MISMATCH"
            self.get_logger().error(
                f"B goal_id와 ROS goal UUID가 다릅니다: payload={request.goal_id!r} ros={ros_goal_id}"
            )
            record_result(result.status, result.reason, ros_goal_id=ros_goal_id)
            return result

        if self._backend is None:
            result.status, result.reason = "FAILED", "EXECUTION_NOT_CONFIGURED"
        elif not self._execution_lock.acquire(blocking=False):
            result.status, result.reason = "FAILED", "BUSY"
        else:
            def publish(stage):
                goal_handle.publish_feedback(SortPart.Feedback(stage=stage))

            try:
                if self._managed and (self._run_paused or not self._job_id):
                    raise SortFailure("SORT_RUN_PAUSED")
                if self._recovery_required:
                    raise SortFailure("RECOVERY_REQUIRED")
                if not self._home_checked:
                    # A goal can arrive 0.1 s after start, before the start timer's check (#374 G-X3b).
                    try:
                        if not self._home_check("first goal"):
                            raise SortFailure("RECOVERY_REQUIRED")
                    except RuntimeError as exc:
                        if isinstance(exc, SortFailure):
                            raise
                        raise SortFailure("CONTROLLERS_UNAVAILABLE") from exc
                if self._managed:
                    if not self._job_id or request.sort_job_id != self._job_id or request.run_epoch != self._epoch:
                        raise SortFailure("SORT_RUN_MISMATCH")
                    if request.timestamp_base != "SIM_TIME" or not 0 < request.capture_stamp_ns < request.valid_until_ns:
                        raise SortFailure("INVALID_GOAL_TIME")
                with self._seen_lock:
                    if request.goal_id in self._seen:
                        raise SortFailure("DUPLICATE_GOAL")
                    self._seen.add(request.goal_id)
                result.arrived_stamp_ns = self._backend.execute(
                    request, publish, lambda: goal_handle.is_cancel_requested or self._stop_run_requested.is_set(),
                    # #394: only a cancel from A/B ends at home here; stop_run homes in C's _stop_run, e-stop stays
                    lambda: (goal_handle.is_cancel_requested and not self._stop_run_requested.is_set()
                             and not self._estop_latched))
                result.status = "SUCCEEDED"
                sent = finish_goal(goal_handle, "succeed", self.get_logger())
                record_result(result.status, arrived_stamp_ns=result.arrived_stamp_ns, result_sent=sent)
                self.get_logger().info(f"B_GOAL_ARRIVED goal_id={request.goal_id} stamp={result.arrived_stamp_ns}")
                return result
            except SortCanceled:
                result.status, result.reason = "CANCELED", "ROBOT_STOP_CONFIRMED"
                self._latch_if_holding("cancel")
                if self._shutdown.is_set() and not goal_handle.is_cancel_requested:
                    result.reason = "SHUTDOWN"
                sent = finish_goal(goal_handle, "canceled" if goal_handle.is_cancel_requested else "abort",
                                   self.get_logger())
                self.get_logger().warning(f"SortPart goal={request.goal_id} status={result.status} reason={result.reason}")
                record_result(result.status, result.reason, result_sent=sent)   # #394: this path had no record
                return result
            except SortFailure as exc:
                result.reason = str(exc)
                result.status = "EXPIRED" if result.reason == "GOAL_EXPIRED" else "FAILED"
                if result.reason == "STOP_UNCONFIRMED":
                    self._recovery_required = True
                    self._run_paused = True
                    self._stop_run_requested.set()
                    if self._feeder is not None:
                        try:
                            self._feeder.set_running(False)
                        except Exception as belt_exc:
                            self.get_logger().error(f"Belt stop after unconfirmed robot stop failed: {belt_exc}")
                    if goal_handle.is_cancel_requested:
                        sent = finish_goal(goal_handle, "abort", self.get_logger())
                        self.get_logger().warning(f"SortPart goal={request.goal_id} status={result.status} reason={result.reason}")
                        record_result(result.status, result.reason, result_sent=sent)   # #394: had no record
                        return result
            except Exception as exc:
                self.get_logger().error(f"SortPart execution error: {type(exc).__name__}: {exc}")
                result.status, result.reason = "FAILED", "EXECUTION_ERROR"
                record_result(result.status, result.reason, exception=f"{type(exc).__name__}: {exc}")
            finally:
                self._execution_lock.release()
        sent = finish_goal(goal_handle, "abort", self.get_logger())
        self.get_logger().warning(f"SortPart goal={request.goal_id} status={result.status} reason={result.reason}")
        record_result(result.status, result.reason, result_sent=sent)
        return result


def main(args=None) -> int:
    parser = argparse.ArgumentParser(description="Receive B SortPart goals")
    parser.add_argument("--action", default=ACTION_NAME)
    parser.add_argument("--gazebo-project-root", type=Path,
                        help="Enable sort_cell simulation execution using this EyetoArm checkout")
    parser.add_argument("--run-state", type=Path,
                        help="Enable managed sort runs and persist the execution epoch in this SQLite file")
    parser.add_argument("--feed-parts", action="store_true", help="Drive the Gazebo belt and supply parts during a managed run")
    parser.add_argument("--moving-belt", action="store_true",
                        help="Enable moving target prediction in managed runs")
    parser.add_argument("--pose-source", choices=POSE_SOURCES, default="sim",
                        help="Target position source: sim (Gazebo poses) or track (A track stream)")
    parser.add_argument("--track-topic", default=TRACK_TOPIC,
                        help="A TrackStateArray topic for --pose-source track")
    parser.add_argument("--sim-truth-log", type=Path,
                        help="With --pose-source track, append track minus simulation pose lines (evaluation only)")
    parser.add_argument("--m3-grasp-offset-mm", type=float, default=None,
                        help="BOLT_A (M3) grasp height offset in mm (default -2, executor constant; #380 comparison)")
    parser.add_argument("--execution-log", type=Path,
                        help="JSONL runtime log (default: <gazebo-project-root>/logs/sort_runs.jsonl)")
    parsed, ros_args = parser.parse_known_args(args)
    if parsed.feed_parts and (not parsed.moving_belt or parsed.pose_source != 'track'):
        parser.error('--feed-parts requires --moving-belt --pose-source track')
    if parsed.run_state and not parsed.gazebo_project_root:
        parser.error("--run-state requires --gazebo-project-root")
    if parsed.moving_belt and not parsed.run_state:
        parser.error("--moving-belt requires --run-state")
    if parsed.pose_source == "track" and not parsed.gazebo_project_root:
        parser.error("--pose-source track requires --gazebo-project-root")
    if parsed.sim_truth_log and parsed.pose_source != "track":
        parser.error("--sim-truth-log requires --pose-source track")
    # #394: own SIGINT/SIGTERM handling; rclpy's default handler closes the context first, so a running
    # goal could not send its result (C items 1·9①).
    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
    arm_node = None
    node = None
    executor = MultiThreadedExecutor(num_threads=2)
    try:
        backend = None
        log_path = parsed.execution_log or ((parsed.gazebo_project_root / "logs" / "sort_runs.jsonl")
                                            if parsed.gazebo_project_root else
                                            (Path.cwd() / "logs" / "sort_runs.jsonl"))
        event_log = SortEventLog(log_path)
        if parsed.gazebo_project_root is not None:
            arm_node = Node("c_sort_gazebo_arm", parameter_overrides=[
                Parameter("use_sim_time", value=True),
            ])
            backend = GazeboSortExecutor(arm_node, parsed.gazebo_project_root.resolve(),
                                         moving=parsed.moving_belt, pose_source=parsed.pose_source,
                                         track_topic=parsed.track_topic, sim_truth_log=parsed.sim_truth_log,
                                         grasp_height_offset_m=(None if parsed.m3_grasp_offset_mm is None else
                                                                {"BOLT_A": parsed.m3_grasp_offset_mm / 1000.}))
            backend.event_log = event_log
        feeder = None
        if parsed.feed_parts:
            from .gazebo_part_feeder import GazeboPartFeeder
            feeder = GazeboPartFeeder(parsed.gazebo_project_root)
        node = SortGoalReceiver(parsed.action, backend, parsed.run_state, feeder, event_log)
        node.get_logger().info(f"SortPart runtime log: {event_log.path}")
        executor.add_node(node)

        def on_signal(_signum, _frame):
            if node._shutdown.is_set():          # second signal: stop now
                Thread(target=executor.shutdown, daemon=True).start()
                return
            node.begin_shutdown(executor)

        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)
        executor.spin()
    except ActionServerConflict as exc:
        print(f"sort_goal_receiver: {exc}", file=sys.stderr)
        return 2
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None and node._feeder is not None:
            node._stop_run_requested.set()
            node._feeder.set_running(False)
        executor.shutdown()
        if node is not None:
            node.destroy_node()
        if arm_node is not None:
            arm_node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
