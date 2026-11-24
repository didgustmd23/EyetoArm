"""Execute B sort goals in the DART sort_cell simulation."""
from __future__ import annotations

import importlib.util
from collections import deque
from collections.abc import Callable
from pathlib import Path
import time

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from sensor_msgs.msg import JointState

from .contact_closer import CLOSE_LIMIT_NS, CLOSE_REAL_FACTOR, CLOSE_TARGET, ContactCloser
from .stop_detector import STOP_LIMIT_NS, STOP_REAL_FACTOR, StopDetector
from .track_stream_tracker import STILL_M, SimExecutionFeedback, SimTruthLog, TrackStreamTracker

PARTS = {"BOLT_A": ("part_00", 0.0015, (0.72, -0.05)),
         "BOLT_B": ("part_01", 0.003, (0.72, 0.17))}
# Grasp height offset per part (m), kept in the executor (#380, A decision 1); the same values as
# pick_bolt_demo.GRASP_HEIGHT_OFFSET (C demo), which is no longer read here. BOLT_A (M3) -2 mm stays the
# default until A decides on -1 mm after the Gazebo comparison.
GRASP_HEIGHT_OFFSET_M = {"BOLT_A": -.002, "BOLT_B": 0.}
DOWN = np.array([0., 0., -1.])
POSE_SOURCES = ("sim", "track")
NO_HOME_AFTER = ("STOP_UNCONFIRMED",)   # stop not confirmed: recovery first, no automatic motion
# Fixed home = the simulation model's initial_positions (simulation/models/ur5e_2f85/ur5e_2f85.urdf.xacro).
# It used to be the joints seen first, so a restart in a carry pose made that pose "home" (A #374, #394).
HOME_JOINTS = (0.0, -1.57, 1.57, -1.57, -1.57, 0.0)
GRIP_SHORT_RAD = .003                  # #394: closed short of the command by more than this = something is held
EMPTY_NEAR_M = .030                    # #380 (a): no simulation part this close to the fingertips = empty hand
EMPTY_CLOSED_RAD = .1                  # gripper joint above this counts as closed (not DETACHED)
OPEN_GRACE_MAX_NS = 2_500_000_000      # #380 (b): goal-start opening time added to the deadline, at most this
START_OPEN_SETTLE_S = .1               # A decision 4: wait after the goal-start opening (placing keeps 0.5 s)
BLOCKED_RISE_M = .001                  # #380 (A decision 2): rise this much and close once more after a belt block
TRACK_TOPIC = "/a/track_states"
PICK_X_M = (.46, .64)                  # pick region in base_link
PICK_ABS_Y_M = .53

# Static track pick: wait above the part until it settles (test values, logged at start).
STILL_WINDOW_S = 1.0                   # settled: measured track within STILL_M for this long
STILL_WAIT_MAX_S = 10.0                # SIM time; longer is PART_MOVED
STILL_WAIT_REAL_FACTOR = 3.            # real-time guard against a stopped SIM clock
REDESCEND_MAX = 2                      # rises after the part moved during the descent
FOLLOW_PERIOD_S = .2                   # wait between checks when the arm does not move
FOLLOW_DEADBAND_M = .002               # follow the track only beyond this xy change
FOLLOW_SPEED_MPS = .10                 # follow move duration = distance / speed
FOLLOW_MIN_S = .2
FOLLOW_TURN_S = 1.                     # minimum duration of a move that also turns the grasp
YAW_DEADBAND_RAD = np.radians(2.)      # re-aim the grasp only beyond this axis change


class SortFailure(RuntimeError):
    """A failure that can be returned to B as a reason code."""


class SortCanceled(RuntimeError):
    """Controller goals ended and subsequent joint feedback confirmed a stop."""


class SimPoses:
    """Latest simulation poses of all sort_cell parts (execution feedback and evaluation log only)."""

    def __init__(self, world="sort_cell"):
        from threading import Lock
        from gz.msgs10.pose_v_pb2 import Pose_V
        from gz.transport13 import Node
        self._lock, self._poses, self._node = Lock(), {}, Node()
        self._topic = f"/world/{world}/pose/info"
        if not self._node.subscribe(Pose_V, self._topic, self._on_poses):
            raise RuntimeError(f"cannot subscribe to {self._topic}")

    def _on_poses(self, message):
        poses = {}
        for pose in message.pose:
            if pose.name.startswith("part_"):
                p, q = pose.position, pose.orientation
                poses[pose.name] = (np.array([p.x, p.y, p.z]), (q.w, q.x, q.y, q.z))
        with self._lock:
            self._poses.update(poses)

    def __call__(self):
        with self._lock:
            return dict(self._poses)

    def close(self):
        self._node.unsubscribe(self._topic)


class GazeboSortExecutor:
    """Use B's static pick target and verify it against the simulation."""

    def __init__(self, node, project_root: Path, *, moving=False, pose_source="sim",
                 track_topic=TRACK_TOPIC, sim_truth_log=None, grasp_height_offset_m=None):
        if pose_source not in POSE_SOURCES:
            raise ValueError(f"pose_source must be one of {POSE_SOURCES}: {pose_source!r}")
        self.node = node
        path = project_root / "scripts" / "pick_bolt_demo.py"
        spec = importlib.util.spec_from_file_location("c_pick_bolt_kinematics", path)
        self.demo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.demo)
        self.chain = self.demo.load_chain()
        self.grasp_height_offset = dict(GRASP_HEIGHT_OFFSET_M)
        for part, value in (grasp_height_offset_m or {}).items():
            if part not in PARTS or not np.isfinite(value) or abs(value) > .005:
                raise ValueError(f"grasp height offset must be a part in {tuple(PARTS)} within 5 mm: {part}={value}")
            self.grasp_height_offset[part] = float(value)
        self.arm = None
        self.moving = moving
        self.cancel_requested = lambda: False
        self.home_after_cancel = lambda: False   # #394: set per goal; a cancel from A/B ends at home
        self._active_grip_target = None
        self._contact_seen = False             # #380: the last close stopped on contact (fingers on a part)
        self._grip_watch = None                # #380: (closer, first result future) watched while the arm moves
        self.grip_outcome = None               # #380: last close decision, for the event log
        self._goal_tracker = None              # #380 (a): sim-mode pose tracker of the running goal
        self.stop_detector = StopDetector(self.demo.JOINTS)   # arm joints only, SIM-time window (#394)
        self.active_grip = None
        self._active_grip_revision = 0
        self._grip_command_revision = 0
        self.holding_object = False
        self.home = np.array(HOME_JOINTS)
        self._home_logged = False
        self._arm_moved = False
        self._retreat_z = None
        self.event_log = None
        self.node.create_subscription(JointState, "/joint_states", self._observe_velocity, 10)
        self.pose_source = pose_source
        self.sim_poses = None
        self.truth_log = None
        self._track = None
        self._last_track_message = None
        if pose_source == "track":
            from eyetoarm_interfaces.msg import TrackStateArray
            self.node.create_subscription(TrackStateArray, track_topic, self._on_track_message, 10)
            self.sim_poses = SimPoses()
            if sim_truth_log is not None:
                self.truth_log = SimTruthLog(self.sim_poses, open(sim_truth_log, "a", encoding="utf-8"))
        self.node.get_logger().info(f"SortPart executor pose_source={pose_source} moving={moving} "
                                    f"grasp_height_offset_mm={ {k: v * 1000. for k, v in self.grasp_height_offset.items()} }")
        if pose_source == "track" and not moving:
            self.node.get_logger().info(
                f"SortPart static settle window_s={STILL_WINDOW_S} still_mm={STILL_M * 1000.:g} "
                f"wait_max_s={STILL_WAIT_MAX_S} redescend_max={REDESCEND_MAX} follow_period_s={FOLLOW_PERIOD_S} "
                f"follow_deadband_mm={FOLLOW_DEADBAND_M * 1000.:g} follow_speed_mps={FOLLOW_SPEED_MPS} "
                f"yaw_deadband_deg={np.degrees(YAW_DEADBAND_RAD):g}")

    def _on_track_message(self, message):
        self._last_track_message = message
        if self._track is not None:
            self._track.on_message(message)

    def _now_ns(self):
        return self.node.get_clock().now().nanoseconds

    def _record_truth(self, stage, goal, xyz):
        if self.truth_log is not None:
            self.truth_log.record(stage, goal.goal_id, goal.part_track_id, xyz, self._now_ns())

    def _event(self, goal, event, **fields):
        if self.event_log is not None:
            self.event_log.record(event, goal_id=goal.goal_id, part_class=goal.part_class,
                                  track_id=goal.part_track_id, pose_source=self.pose_source, **fields)

    def _observe_velocity(self, message):
        """Feed arm joint positions with their SIM stamp to the stop detector. The gripper joint is not
        used: while it holds a small bolt its velocity keeps jittering (A #354 D-13, #358)."""
        stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        self.stop_detector.add(stamp or self._now_ns(), dict(zip(message.name, message.position)))

    def _confirm_stopped(self):
        """Wait until the arm joints stay still for the detector window. The SIM-time limit is measured on
        the joint samples' own stamps, not the node clock (stale after an idle period, #394 H-ES); a wall
        guard covers a stopped SIM clock."""
        self.stop_detector.reset()
        wall_end = time.monotonic() + STOP_LIMIT_NS / 1e9 * STOP_REAL_FACTOR
        while not self.stop_detector.stopped() and time.monotonic() < wall_end:
            if self.stop_detector.span_ns() >= STOP_LIMIT_NS:
                break
            rclpy.spin_once(self.node, timeout_sec=.05)
        if not self.stop_detector.stopped():
            raise SortFailure("STOP_UNCONFIRMED")

    def _ensure_arm_home(self):
        if self.arm is None:
            self.arm = self.demo.Arm(self.node)
            self.arm._call = self._dispatch
        self._log_home()

    def _log_home(self):
        """Log the fixed home and the joints seen first, once."""
        if not self._home_logged:
            self._home_logged = True
            self.node.get_logger().info(f"SortPart HOME fixed joints={np.round(self.home, 4).tolist()} "
                                        f"current={np.round(self.arm.joints(), 4).tolist()}")

    def refresh_hold_from_joint(self):
        """After a restart nothing was commanded yet: treat a gripper closed beyond open + 0.05 rad as
        holding (the same fallback C uses when a grip goal did not succeed). Returns holding_object."""
        self._ensure_arm_home()
        joint = self.arm.state.get(self.demo.GRIPPER_JOINT, self.demo.GRIP_OPEN)
        if joint > self.demo.GRIP_OPEN + .05:
            self.holding_object = True
        return self.holding_object

    def at_home(self, tolerance=.01):
        """True when every arm joint is within `tolerance` rad of the fixed home (connects the arm first)."""
        self._ensure_arm_home()
        return bool(np.max(np.abs(self.arm.joints() - self.home)) <= tolerance)

    def move_home(self):
        """Move the arm to its saved initial pose without touching the gripper."""
        if self.active_grip is not None:
            raise SortFailure("GRIPPER_ACTION_ACTIVE")
        self._ensure_arm_home()
        self.cancel_requested = lambda: False
        self._confirm_stopped()
        self.node.get_logger().info("SortPart HOME start; gripper and conveyor unchanged")
        error = self.arm.follow([self.home], 3.)
        if error > .015:
            raise SortFailure(f"HOME_TRACKING_ERROR: {error:.6f} rad")
        self._confirm_stopped()
        self.node.get_logger().info("SortPart HOME done")

    def recover_home(self):
        """If grasp feedback says an object is held, release it here, then home the arm."""
        if self.active_grip is not None:
            raise SortFailure("GRIPPER_ACTION_ACTIVE")
        self._ensure_arm_home()
        self.cancel_requested = lambda: False
        self._confirm_stopped()
        if self.holding_object:
            self.node.get_logger().info("SortPart RECOVER releasing held object at current pose")
            current = self.arm.joints()
            pose = self.demo.forward(self.chain, current)
            x, y, z = pose[:3, 3]
            if np.linalg.norm(pose[:3, 2] - DOWN) > .02:
                raise SortFailure("RECOVERY_TOOL_NOT_VERTICAL")
            if .47 <= x <= .63 and abs(y) <= .60:
                surface = .100  # sort_cell conveyor top, with clearance from the edges
            elif any(abs(x - bx) <= .025 and abs(y - by) <= .045
                     for _, _, (bx, by) in PARTS.values()):
                surface = .004  # bottom of a bolt bin, well inside its walls
            else:
                raise SortFailure("RECOVERY_NO_KNOWN_SUPPORT")
            release_z = surface + self.demo.FINGER_REACH + self.demo.TIP_CLEARANCE
            if z < release_z - .002:
                raise SortFailure("RECOVERY_BELOW_RELEASE_HEIGHT")
            self._move_vertical(release_z, 2.)
            self._grip(self.demo.GRIP_OPEN)
            self._move_vertical(max(z, release_z + .09), 1.5)
        self.move_home()

    def _move_vertical(self, z, seconds):
        current = self.arm.joints()
        pose = self.demo.forward(self.chain, current)
        start = pose[:3, 3]
        if abs(z - start[2]) > .001:
            self._follow(self._line(start, np.r_[start[:2], z], pose[:3, :3], current), seconds)

    def _check_cancel(self):
        if self.cancel_requested():
            self._finish_grip()
            self._confirm_stopped()
            raise SortCanceled()

    def _finish_grip(self):
        self._grip_watch = None
        if self.active_grip is None:
            return
        handle, self.active_grip = self.active_grip, None
        result = handle.get_result_async()
        if not result.done():
            handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self.node, result, timeout_sec=5.)
        if not result.done():
            raise SortFailure("STOP_UNCONFIRMED")
        reply = result.result()
        if self._active_grip_revision == self._grip_command_revision:
            if getattr(reply, "status", None) == GoalStatus.STATUS_SUCCEEDED:
                self.holding_object = (bool(getattr(reply.result, "stalled", False))
                                       or self._closed_short(self._active_grip_target)
                                       or self._closing_contact(self._active_grip_target))
            else:
                self.holding_object = (self.arm is not None and
                                       self.arm.state.get(self.demo.GRIPPER_JOINT, self.demo.GRIP_OPEN)
                                       > self.demo.GRIP_OPEN + .05)

    def _grip(self, position, settle=0.5):
        """Command the gripper and retain its controller's stalled/holding indication."""
        self._grip_command_revision += 1
        try:
            result = self.arm.grip(position, settle=settle)
        except SortCanceled:
            joint = self.arm.state.get(self.demo.GRIPPER_JOINT)
            if joint is not None:
                if float(position) <= self.demo.GRIP_OPEN + 1e-6:
                    self.holding_object = self.holding_object and joint > self.demo.GRIP_OPEN + .015
                elif float(position) >= self.demo.GRIP_CLOSED - .01:
                    self.holding_object = joint > self.demo.GRIP_OPEN + .05 and self._closed_short(position)
            raise
        if float(position) <= self.demo.GRIP_OPEN + 1e-6:
            self.holding_object = False
            self._contact_seen = False
        else:
            self.holding_object = (bool(getattr(result, "stalled", False)) or self._closed_short(position)
                                   or self._closing_contact(position))
        return result

    def _closing_contact(self, target):
        """#380: a closing command after the fingers stopped on contact (ContactCloser "stall") holds a
        part even when the re-command is reached exactly (contact + 0.005 leaves no shortfall)."""
        return self._contact_seen and target is not None and float(target) > self.demo.GRIP_OPEN + 1e-6

    def _closed_short(self, target):
        """#394: the gripper stopped short of a closing command, so a part is between the fingers. The
        controller's `stalled` is not reported while a small bolt makes the joint velocity jitter (A #358)."""
        if target is None or self.arm is None or float(target) <= self.demo.GRIP_OPEN + 1e-6:
            return False
        joint = self.arm.state.get(self.demo.GRIPPER_JOINT)
        return joint is not None and float(target) - joint > GRIP_SHORT_RAD

    def _dispatch(self, client, goal, limit):
        self._check_cancel()
        send = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, send, timeout_sec=5.)
        if not send.done():
            raise SortFailure("STOP_UNCONFIRMED")
        handle = send.result()
        if not handle.accepted:
            raise SortFailure("CONTROLLER_REJECTED_GOAL")
        result = handle.get_result_async()
        end = time.monotonic() + limit
        while not result.done() and not self.cancel_requested() and time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=.05)
            if self._grip_watch is not None:
                self._watch_grip()
        if not result.done():
            handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self.node, result, timeout_sec=5.)
            if not result.done():
                raise SortFailure("STOP_UNCONFIRMED")
            self._confirm_stopped()
            if not self.cancel_requested():
                raise SortFailure("CONTROLLER_TIMEOUT")
        self._check_cancel()
        response = result.result()
        if response.status != GoalStatus.STATUS_SUCCEEDED and not getattr(response.result, "stalled", False):
            raise SortFailure("CONTROLLER_FAILED")
        return response.result

    def _rotation(self, xyzw):
        q = np.asarray(xyzw)
        if not np.all(np.isfinite(q)) or not .99 <= np.linalg.norm(q) <= 1.01:
            raise SortFailure("INVALID_ORIENTATION")
        x, y, z, w = q / np.linalg.norm(q)
        rotation = self.demo.quaternion_matrix(w, x, y, z)
        if np.linalg.norm(rotation[:, 2] - DOWN) > .02:
            raise SortFailure("UNSUPPORTED_GRIPPER_ORIENTATION")
        return rotation

    def _solve(self, xyz, rotation, seed):
        joints = self.demo.solve_ik(self.chain, self.demo.transform(rotation, xyz), seed)
        if np.max(np.abs(joints)) >= 6.2 or abs(joints[2]) >= 3.1:
            raise SortFailure("JOINT_LIMIT")
        return joints

    def _line(self, start, target, rotation, seed):
        path = []
        for t in np.linspace(.05, 1., 20):
            seed = self._solve(start + (target - start) * t, rotation, seed)
            path.append(seed)
        return path

    def _follow(self, path, seconds):
        self._arm_moved = True
        error = self.arm.follow(path, seconds)
        if error > .015:
            target = np.asarray(path[-1], dtype=float)
            actual = self.arm.joints()
            self.node.get_logger().error(
                f"ARM_TRACKING_ERROR max_error={error:.6f} rad "
                f"target={np.round(target, 4).tolist()} actual={np.round(actual, 4).tolist()}")
            raise SortFailure("ARM_TRACKING_ERROR")

    def _check_stationary(self, tracker, initial):
        if self.pose_source == "track":
            tracker.check_stationary(initial)
        elif np.linalg.norm(tracker.pose()[0] - initial) > .002:
            raise SortFailure("PART_MOVED")

    def execute(self, goal, feedback: Callable[[str], None], cancel_requested: Callable[[], bool],
                home_after_cancel: Callable[[], bool] = lambda: False,
                *, wait_for_resume: Callable[[], bool] = lambda: False):
        self.cancel_requested = cancel_requested
        self.home_after_cancel = home_after_cancel
        self._check_cancel()
        if goal.goal_kind != "PICK_PLACE" or goal.part_class not in PARTS:
            raise SortFailure("UNSUPPORTED_GOAL_KIND")
        if goal.place_bin != goal.part_class:
            raise SortFailure("INVALID_BIN")
        if goal.frame_id != "base_link" or goal.timestamp_base != "SIM_TIME":
            raise SortFailure("UNSUPPORTED_FRAME_OR_CLOCK")

        name, radius, bin_xy = PARTS[goal.part_class]
        observed = np.asarray(goal.observed_position_mm) / 1000.
        place = np.asarray(goal.place_position_mm) / 1000.
        heights = np.array([goal.approach_height_mm, goal.lift_height_mm]) / 1000.
        width = goal.opening_width_mm / 1000.
        if (not np.all(np.isfinite(np.r_[observed, place, heights, width]))
                or not 2 * radius < width <= .030
                or np.any(np.abs(place[:2] - bin_xy) > [.035, .055])
                or not .10 <= place[2] <= .24):
            raise SortFailure("INVALID_TARGET")
        place_rotation = self._rotation(goal.place_orientation_xyzw)

        demo = self.demo

        class PoseTracker(demo.PartTracker):
            def __init__(self, world, part):
                self.history = deque(maxlen=600)
                super().__init__(world, part)

            def _on_poses(self, message):
                stamp = message.header.stamp
                for pose in message.pose:
                    if pose.name == self._part:
                        p, q = pose.position, pose.orientation
                        with self._lock:
                            self._pose = (np.array([p.x, p.y, p.z]), (q.w, q.x, q.y, q.z),
                                          stamp.sec * 1_000_000_000 + stamp.nsec)
                            self.history.append(self._pose)
                        break

            def recent_motion(self, prefer_gripper=False):
                with self._lock:
                    sample = self._pose
                    old = next((pose for pose in reversed(self.history)
                                if pose[2] <= sample[2] - 250_000_000), None)
                if old is None or sample[2] - old[2] > 500_000_000:
                    raise SortFailure("SIM_POSE_STALE")
                return sample, (sample[0][:2] - old[0][:2]) / ((sample[2] - old[2]) / 1e9)

            def close(self):
                self._node.unsubscribe("/world/sort_cell/pose/info")

        if self.pose_source == "track":
            tracker = TrackStreamTracker(goal.part_track_id, goal.part_class, self._now_ns, failure=SortFailure)
            if self._last_track_message is not None:
                tracker.on_message(self._last_track_message)
            self._track = tracker
        else:
            tracker = PoseTracker("sort_cell", name)
        self._goal_tracker = tracker
        try:
            return self._pick_and_place(goal, feedback, tracker, observed,
                                        place, place_rotation, heights, radius, bin_xy, wait_for_resume)
        finally:
            self._goal_tracker = None
            self._track = None
            tracker.close()
            self._finish_grip()

    def _deadline(self, goal, grace_ns=0):
        """GOAL_EXPIRED when the SIM clock reached valid_until_ns (+ grace_ns, #380 (b))."""
        if goal.valid_until_ns:
            now = self.node.get_clock().now().nanoseconds
            if now <= 0:
                raise SortFailure("SIM_CLOCK_UNAVAILABLE")
            if goal.capture_stamp_ns > now + 50_000_000:
                raise SortFailure("CAPTURE_IN_FUTURE")
            if now >= goal.valid_until_ns + grace_ns:
                raise SortFailure("GOAL_EXPIRED")

    def _pick_and_place(self, goal, feedback, tracker, observed, place,
                        place_rotation, heights, radius, bin_xy, wait_for_resume=lambda: False):
        """Run one goal. After a failure once the arm has moved, return home (not after a cancel
        or an unconfirmed stop); the failure reason is raised unchanged."""
        self._arm_moved = False
        self._retreat_z = None
        checkpoint = {}
        while True:
            try:
                if self.holding_object and "part" not in checkpoint and "bind_part" in checkpoint:
                    checkpoint["part"] = checkpoint["bind_part"]()
                if "part" in checkpoint and (self.holding_object or checkpoint.get("release_started")):
                    return self._place_part(goal, feedback, checkpoint, place, place_rotation, bin_xy, resuming=True)
                if self._retreat_z is not None:
                    self._deadline(goal)
                    self._move_vertical(max(self._retreat_z, self.demo.forward(self.chain, self.arm.joints())[2, 3]), 1.5)
                checkpoint.clear()
                return self._pick_and_place_steps(goal, feedback, tracker, observed, place,
                                                  place_rotation, heights, radius, bin_xy, checkpoint)
            except SortFailure as exc:
                if self._arm_moved and str(exc) not in NO_HOME_AFTER:
                    self._return_home(goal, f"failure:{exc}", retreat=True)
                raise
            except SortCanceled:
                if wait_for_resume():
                    continue
                if self._arm_moved and self.home_after_cancel():
                    self._home_after_cancel(goal)
                raise

    def _home_after_cancel(self, goal):
        """#394 (A decision, C item 10): after a confirmed cancel from A/B, rise straight up if low and
        use C's move_home (gripper untouched), so the link camera sees the belt again. Run stop and
        e-stop are left to C (stop_run homes afterwards, e-stop stays in place). A failed move only logs."""
        log = self.node.get_logger()
        log.info(f"SortPart CANCEL_HOME start goal={goal.goal_id} holding={self.holding_object}")
        self.cancel_requested = lambda: False
        try:
            self._finish_grip()
            if self._retreat_z is not None:
                current = self.arm.joints()
                pose = self.demo.forward(self.chain, current)
                start = pose[:3, 3]
                if start[2] < self._retreat_z:
                    try:
                        self._follow(self._line(start, np.r_[start[:2], self._retreat_z], pose[:3, :3], current), 1.5)
                    except SortFailure:
                        raise
                    except RuntimeError:
                        log.warning(f"SortPart CANCEL_HOME no straight retreat path goal={goal.goal_id}")
            self.move_home()
        except RuntimeError as exc:
            log.warning(f"SortPart CANCEL_HOME failed goal={goal.goal_id}: {type(exc).__name__}: {exc}")
            return
        log.info(f"SortPart CANCEL_HOME done goal={goal.goal_id} holding={self.holding_object}")

    def _pick_and_place_steps(self, goal, feedback, tracker, observed, place,
                              place_rotation, heights, radius, bin_xy, checkpoint):
        feedback("PLANNING")
        if self.pose_source == "track":
            initial, initial_orientation, still, velocity, correction = self._plan_from_track(goal, tracker, observed)
        else:
            start = tracker.pose()
            self._spin_for(.5)
            initial, initial_orientation, stamp = tracker.pose()
            dt = (stamp - start[2]) / 1e9
            if dt <= 0:
                raise SortFailure("SIM_POSE_STALE")
            self._deadline(goal)
            velocity = (initial - start[0]) / dt
            age = (stamp - goal.capture_stamp_ns) / 1e9
            correction = np.r_[velocity[:2] * max(0., age), 0.] if self.moving else np.zeros(3)
            if np.linalg.norm(initial - observed - correction) > .025:
                raise SortFailure("PART_NOT_AT_OBSERVATION")
            still = initial
            if not self.moving:
                self._check_stationary(tracker, start[0])
        if self.arm is None:
            self.arm = self.demo.Arm(self.node)
            self.arm._call = self._dispatch
        self._log_home()
        current = self.arm.joints()
        offset = self.demo.FINGER_REACH + self.demo.TIP_CLEARANCE - radius
        self._retreat_z = heights[0] + offset
        candidates = []
        # Moving grasps apply the per-part correction incrementally in _moving_grasp.
        # Stationary grasps use the planned descent path, so apply it to that path's
        # pick height here. This also covers the stopped-belt track-settle path.
        grasp_height_offset = (0.0 if self.moving else
                               self.grasp_height_offset[goal.part_class])
        for candidate in goal.grasp_pose_candidates:
            pick = np.asarray(candidate.position_mm) / 1000.
            if (not np.all(np.isfinite(pick)) or np.linalg.norm(pick - observed) > .035
                    or not .097 <= pick[2] <= .115
                    or not np.all((heights >= pick[2] + .05) & (heights <= .35))):
                continue
            try:
                rotation = self._rotation(candidate.orientation_xyzw)
                pick_xy = pick[:2] + correction[:2]
                lead = velocity[:2] * 2. if self.moving else 0
                approach, descent, lift, carry, lowering, at_pick, raised = self._paths(
                    pick_xy, pick[2] + grasp_height_offset + offset, lead, rotation, heights, offset,
                    place, place_rotation, current)
            except RuntimeError:
                continue
            cost = np.max(np.abs(approach - current))
            candidates.append((cost, approach, descent, lift, carry, lowering, rotation, at_pick, raised))
        if not candidates:
            raise SortFailure("IK_NO_SOLUTION")
        _, approach, descent, lift, carry, lowering, rotation, at_pick, raised = min(
            candidates, key=lambda plan: plan[0])
        self._deadline(goal)
        self._event(goal, "pick_plan", observed_position_m=observed.tolist(),
                    part_position_m=initial.tolist(), correction_m=correction.tolist(),
                    candidate_count=len(goal.grasp_pose_candidates), valid_candidate_count=len(candidates),
                    tool_pick_position_m=at_pick.tolist(), finger_offset_m=offset,
                    target_heights_m=heights.tolist(), grasp_height_offset_m=grasp_height_offset,
                    gripper_opening_mm=goal.opening_width_mm)
        feedback("EXECUTING")
        # #380 (b): check the deadline before opening a gripper left closed; the opening time does not
        # count against the goal (at most OPEN_GRACE_MAX_NS), and the wait after it is 0.1 s.
        self._deadline(goal)
        if abs(self.arm.state[self.demo.GRIPPER_JOINT] - self.demo.GRIP_OPEN) > .015:
            opened_from = self._now_ns()
            self._grip(self.demo.GRIP_OPEN, settle=START_OPEN_SETTLE_S)
            spent = min(max(0, self._now_ns() - opened_from), OPEN_GRACE_MAX_NS)
            self._event(goal, "start_open", spent_ns=spent)
            self._deadline(goal, spent)
        self._follow([approach], 3.)
        if self.moving:
            result, base_z = self._moving_grasp(tracker, initial, initial_orientation, at_pick, raised,
                                                rotation, goal, offset, checkpoint)
            # The transport endpoints are derived from B's placement, with IK seeded
            # from the actual lift pose after the moving grasp.
            over_pose = self.demo.forward(self.chain, carry)
            carry = self._solve(over_pose[:3,3], place_rotation, self.arm.joints())
            release_pose = self.demo.forward(self.chain, lowering[-1])
            lowering = self._line(over_pose[:3,3], release_pose[:3,3], place_rotation, carry)
        elif self.pose_source == "track":
            plan = (still, initial_orientation)
            lift, carry, lowering = self._settle_and_descend(
                goal, tracker, plan, (approach, descent, lift, carry, lowering, rotation, at_pick),
                heights, offset, place, place_rotation)
            checkpoint["bind_part"] = lambda: self._result_source(tracker, initial, offset)[0]
            self._close_on_part(goal)
            result, base_z = self._result_source(tracker, initial, offset)
            checkpoint["part"] = result
            self._follow(lift, 2.)
        else:
            self._check_stationary(tracker, still)
            self._follow(descent, 2.)
            try:
                self._check_stationary(tracker, still)
            except SortFailure:
                self._follow(list(reversed(descent[:-1])) + [approach], 2.)
                raise
            checkpoint["bind_part"] = lambda: self._result_source(tracker, initial, offset)[0]
            self._close_on_part(goal)
            result, base_z = self._result_source(tracker, initial, offset)
            checkpoint["part"] = result
            self._follow(lift, 2.)
        lifted_pose = result.pose()[0]
        lift_delta_m = float(lifted_pose[2] - base_z)
        self._event(goal, "lift_check", before_position_m=[float(v) for v in initial],
                    after_position_m=lifted_pose.tolist(), lift_delta_m=lift_delta_m,
                    required_lift_m=.025)
        if lift_delta_m < .025:
            self._grip(self.demo.GRIP_OPEN)
            raise SortFailure("MISSED")
        self.holding_object = True             # #380: the part came up with the fingers (simulation feedback)
        self._contact_seen = True              # kept until the gripper opens (a later grip result cannot clear it)
        checkpoint["part"] = result
        return self._place_part(goal, feedback, checkpoint, place, place_rotation, bin_xy,
                                planned=(carry, lowering))

    def _place_part(self, goal, feedback, checkpoint, place, place_rotation, bin_xy,
                    *, planned=None, resuming=False):
        result = checkpoint["part"]
        feedback("PLACING")
        release_z = place[2] + self.demo.FINGER_REACH
        clearance = max(release_z, self._retreat_z or release_z)
        if resuming and "arrival" not in checkpoint and not checkpoint.get("release_started"):
            tool = self.demo.forward(self.chain, self.arm.joints())[:3, 3]
            if np.linalg.norm(result.pose()[0] - (tool - [0., 0., self.demo.FINGER_REACH])) > .025:
                raise SortFailure("PART_LOST_BEFORE_RESUME")
        if "arrival" not in checkpoint and (not checkpoint.get("release_started") or self.holding_object):
            if planned is None:
                tool_z = self.demo.forward(self.chain, self.arm.joints())[2, 3]
                clearance = max(clearance, tool_z)
                self._move_vertical(clearance, 1.5)
                above = np.r_[place[:2], clearance]
                carry = self._solve(above, place_rotation, self.arm.joints())
                lowering = self._line(above, np.r_[place[:2], release_z], place_rotation, carry)
            else:
                carry, lowering = planned
            self._follow([carry], 3.)
            self._follow(lowering, 1.5)
            held = result.pose()[0]
            if np.any(np.abs(held[:2] - bin_xy) >= [.05, .07]) or held[2] <= .08:
                raise SortFailure("PART_LOST_BEFORE_RELEASE")
            checkpoint["release_started"] = True
            self._grip(self.demo.GRIP_OPEN)
        if "arrival" not in checkpoint:
            arrival = self._wait_for_bin(result, bin_xy)
            checkpoint["arrival"] = arrival
            self._event(goal, "bin_arrival", arrived_stamp_ns=arrival, final_part_position_m=result.pose()[0].tolist())
        if planned is None:
            self._move_vertical(max(clearance, self.demo.forward(self.chain, self.arm.joints())[2, 3]), 1.5)
        else:
            self._follow([carry], 1.5)
        self._return_home(goal, "success")
        return checkpoint["arrival"]

    def _paths(self, pick_xy, z, lead, rotation, heights, offset, place, place_rotation, seed):
        """Joint paths for one grasp point: (approach, descent, lift, carry, lowering, at_pick, raised).
        Raises RuntimeError (SortFailure included) when IK has no solution."""
        above = np.r_[pick_xy + lead, heights[0] + offset]
        at_pick = np.r_[pick_xy, z]
        raised = np.r_[pick_xy, heights[1] + offset]
        release_z = place[2] + self.demo.FINGER_REACH
        over_bin = np.r_[place[:2], max(raised[2], release_z)]
        release = np.r_[place[:2], release_z]
        approach = self._solve(above, rotation, seed)
        descent = self._line(above, at_pick, rotation, approach)
        lift = self._line(at_pick, raised, rotation, descent[-1])
        carry = self._solve(over_bin, place_rotation, lift[-1])
        lowering = self._line(over_bin, release, place_rotation, carry)
        return approach, descent, lift, carry, lowering, at_pick, raised

    def _settle_and_descend(self, goal, tracker, plan, paths, heights, offset, place, place_rotation):
        """Static track pick from the approach height: wait until the track settles, following it at
        the approach height, then descend to the settled position. If the part moves during the
        descent, rise and wait again (at most REDESCEND_MAX times). Returns (lift, carry, lowering)
        with the fingers at the part.

        plan is (measured xyz at planning, track orientation at planning or None); paths are the
        planned (approach, descent, lift, carry, lowering, rotation, at_pick)."""
        log = self.node.get_logger()
        plan_position, plan_orientation = plan
        approach, descent, lift, carry, lowering, base_rotation, at_pick = paths
        pick_xy, z = at_pick[:2].copy(), at_pick[2]
        z_above = heights[0] + offset
        arm_xy, rotation = pick_xy.copy(), base_rotation    # where the arm waits now
        followed_m, redescents = 0., 0

        while True:
            log.info(f"SortPart STILL_WAIT start goal={goal.goal_id} redescents={redescents}")
            started_ns, started_real = self._now_ns(), time.monotonic()
            while True:
                self._check_cancel()
                still, spread, position, orientation, _ = tracker.stillness(int(STILL_WINDOW_S * 1e9))
                target_xy = pick_xy + (position[:2] - plan_position[:2])
                if not (PICK_X_M[0] <= target_xy[0] <= PICK_X_M[1] and abs(target_xy[1]) <= PICK_ABS_Y_M):
                    raise SortFailure("PART_LEFT_PICK_REGION")
                if still:
                    break

                waited_s = (self._now_ns() - started_ns) / 1e9
                if (waited_s > STILL_WAIT_MAX_S
                        or time.monotonic() - started_real > STILL_WAIT_MAX_S * STILL_WAIT_REAL_FACTOR):
                    log.warning(f"SortPart STILL_WAIT timeout goal={goal.goal_id} waited_s={waited_s:.2f} "
                                f"spread_mm={spread * 1000.:.1f} followed_mm={followed_m * 1000.:.1f} "
                                f"redescents={redescents}")
                    raise SortFailure("PART_MOVED")

                step = float(np.linalg.norm(target_xy - arm_xy))
                if step > FOLLOW_DEADBAND_M:
                    self._move_above(arm_xy, target_xy, z_above, rotation, rotation, step)
                    arm_xy, followed_m = target_xy, followed_m + step
                else:
                    self._spin_for(FOLLOW_PERIOD_S)

            # Settled: re-aim the grasp at the settled position and axis.
            new_rotation = self._turned(base_rotation, plan_orientation, orientation)
            # Keep the planned paths when the part is still where they aim and the arm waits there.
            planned = (np.linalg.norm(target_xy - pick_xy) <= FOLLOW_DEADBAND_M
                       and np.linalg.norm(arm_xy - pick_xy) <= FOLLOW_DEADBAND_M
                       and new_rotation is base_rotation and rotation is base_rotation)
            if not planned:
                step = float(np.linalg.norm(target_xy - arm_xy))
                self._move_above(arm_xy, target_xy, z_above, rotation, new_rotation, step)
                arm_xy, rotation, followed_m = target_xy, new_rotation, followed_m + step
                try:
                    approach, descent, lift, carry, lowering, _, _ = self._paths(
                        target_xy, z, 0, rotation, heights, offset, place, place_rotation, self.arm.joints())
                except SortFailure:
                    raise
                except RuntimeError:
                    raise SortFailure("IK_NO_SOLUTION")
                pick_xy, plan_position = target_xy, position
                base_rotation, plan_orientation = rotation, orientation
            log.info(f"SortPart STILL_WAIT done goal={goal.goal_id} "
                     f"waited_s={(self._now_ns() - started_ns) / 1e9:.2f} spread_mm={spread * 1000.:.1f} "
                     f"followed_mm={followed_m * 1000.:.1f} redescents={redescents} "
                     f"pick_mm={np.round(target_xy * 1000., 1).tolist()} replanned={not planned}")

            self._follow(descent, 2.)
            try:
                tracker.check_stationary(position)
            except SortFailure as exc:
                if str(exc) != "PART_MOVED":
                    raise
                self._follow(list(reversed(descent[:-1])) + [approach], 2.)
                redescents += 1
                log.info(f"SortPart REDESCEND goal={goal.goal_id} redescents={redescents} max={REDESCEND_MAX}")
                if redescents > REDESCEND_MAX:
                    raise
                continue
            return lift, carry, lowering

    def _move_above(self, start_xy, target_xy, z, rotation, target_rotation, distance):
        """Move the tool at height z from start_xy to target_xy; the last point takes target_rotation."""
        current = self.arm.joints()
        try:
            path = self._line(np.r_[start_xy, z], np.r_[target_xy, z], rotation, current)
            if target_rotation is not rotation:
                path.append(self._solve(np.r_[target_xy, z], target_rotation, path[-1]))
        except SortFailure:
            raise
        except RuntimeError:
            raise SortFailure("IK_NO_SOLUTION")
        seconds = max(FOLLOW_MIN_S, distance / FOLLOW_SPEED_MPS)
        self._follow(path, seconds if target_rotation is rotation else max(seconds, FOLLOW_TURN_S))

    def _turned(self, rotation, reference, orientation):
        """B's grasp rotation turned about the vertical by the track axis change from reference to
        orientation. The same object when either axis is unknown or the change is within the deadband."""
        if reference is None or orientation is None:
            return rotation
        a = self.demo.quaternion_matrix(*reference)[:2, 2]
        b = self.demo.quaternion_matrix(*orientation)[:2, 2]
        angle = float(np.arctan2(a[0] * b[1] - a[1] * b[0], np.dot(a, b)))
        if abs(angle) <= YAW_DEADBAND_RAD:
            return rotation
        c, s = np.cos(angle), np.sin(angle)
        closing = np.r_[np.array([[c, -s], [s, c]]) @ rotation[:2, 0], 0.]
        closing /= np.linalg.norm(closing)
        return np.column_stack([closing, np.cross(DOWN, closing), DOWN])

    def _return_home(self, goal, why, retreat=False):
        """Move back to the saved home joints. After a failure first rise to the approach
        height where possible. A failed return only logs a warning; the goal result is unchanged."""
        log = self.node.get_logger()
        log.info(f"SortPart HOME_RETURN start goal={goal.goal_id} after={why}")
        try:
            self._finish_grip()
            opening = retreat and self._empty_and_closed()
            if opening:      # #380 (a): open on the way home so the next goal starts open
                log.info(f"SortPart HOME_RETURN opening empty gripper goal={goal.goal_id}")
                self._start_grip(self.demo.GRIP_OPEN)
            if retreat and self._retreat_z is not None:
                current = self.arm.joints()
                pose = self.demo.forward(self.chain, current)
                start = pose[:3, 3]
                if start[2] < self._retreat_z:
                    try:
                        up = self._line(start, np.r_[start[:2], self._retreat_z], pose[:3, :3], current)
                    except RuntimeError:
                        up = None
                        log.warning(f"SortPart HOME_RETURN no straight retreat path goal={goal.goal_id}")
                    if up is not None:
                        self._follow(up, 1.5)
            self._follow([self.home], 3.)
            if opening:
                self._finish_grip()
        except SortCanceled:
            if why == "success":
                raise
            return
        except RuntimeError as exc:
            log.warning(f"SortPart HOME_RETURN failed goal={goal.goal_id} after={why}: {type(exc).__name__}: {exc}")
            return
        log.info(f"SortPart HOME_RETURN done goal={goal.goal_id}")

    def _part_positions(self):
        """Simulation part positions near which the fingers may hold something: all parts in track mode,
        the goal part in sim mode, None when unknown (#380 (a); simulation truth, no real-robot sensor)."""
        if self.sim_poses is not None:
            return [np.asarray(p) for p, _ in self.sim_poses().values()]
        if self._goal_tracker is not None:
            return [np.asarray(self._goal_tracker.pose()[0])]
        return None

    def _empty_and_closed(self):
        """#380 (a): the gripper is closed (not DETACHED) and no part is within EMPTY_NEAR_M of the
        fingertips. Unknown parts or a failed check keep it closed (a held part is never dropped)."""
        joint = self.arm.state.get(self.demo.GRIPPER_JOINT, self.demo.GRIP_OPEN)
        if joint <= EMPTY_CLOSED_RAD:
            return False
        try:
            parts = self._part_positions()
            if parts is None:
                return False
            tip = self.demo.forward(self.chain, self.arm.joints())[:3, 3] - [0., 0., self.demo.FINGER_REACH]
        except Exception as exc:   # noqa: BLE001 - a missing pose keeps the gripper as it is
            self.node.get_logger().warning(f"SortPart empty-hand check skipped: {type(exc).__name__}: {exc}")
            return False
        return all(np.linalg.norm(p - tip) > EMPTY_NEAR_M for p in parts)

    def _plan_from_track(self, goal, tracker, observed):
        """Track mode planning: position, motion and identity all come from the A track.

        The node is spun only while a goal runs, so its SIM clock and the cached track message are
        as old as the previous goal. Spin (at least once, at most 0.5 s) until a frame at or after
        this goal's capture arrives; without one the cached sample is TRACK_STALE."""
        started = time.monotonic()
        self._spin_until(lambda: tracker.failed or (tracker.last_frame_ns or 0) >= goal.capture_stamp_ns, .5)
        self.node.get_logger().info(
            f"SortPart track planning goal={goal.goal_id} now={self._now_ns()} capture={goal.capture_stamp_ns} "
            f"frame={tracker.last_frame_ns} waited_ms={(time.monotonic() - started) * 1000.:.0f}")
        initial, initial_orientation, stamp = tracker.pose(require_axis=self.moving)
        if self.moving and not tracker.handoff_on():
            raise SortFailure("TRACK_HANDOFF_OFF")
        self._deadline(goal)
        velocity = tracker.velocity()
        tracker.check_observation(observed, goal.capture_stamp_ns)
        age = (stamp - goal.capture_stamp_ns) / 1e9
        correction = np.r_[velocity[:2] * max(0., age), 0.] if self.moving else np.zeros(3)
        still = tracker.measured_position()
        if not self.moving:
            tracker.check_stationary(still)
        self._record_truth("planning", goal, initial)
        return initial, initial_orientation, still, velocity, correction

    def _result_source(self, tracker, initial, offset):
        """Lift, hold and arrival feedback after closing: the tracker itself in sim mode,
        the simulation part nearest the fingers in track mode."""
        if self.pose_source != "track":
            return tracker, initial[2]
        tool = self.demo.forward(self.chain, self.arm.joints())[:3, 3]
        result = SimExecutionFeedback(self.sim_poses, failure=SortFailure)
        bound = result.bind(tool - np.array([0., 0., offset]))
        self.node.get_logger().info(f"SortPart execution feedback part={result.part}")
        return result, bound[0][2]

    def _spin_until(self, done, seconds):
        stop = time.monotonic() + seconds
        rclpy.spin_once(self.node, timeout_sec=.05)   # always once: pending /clock and track messages
        while not done() and time.monotonic() < stop:
            rclpy.spin_once(self.node, timeout_sec=.05)

    def _moving_grasp(self, tracker, initial, initial_orientation, at_pick, raised, rotation, goal, offset, checkpoint):
        # B's candidate sets the point and direction. Pose increments (Gazebo in sim
        # mode, the A track in track mode) supply relative XY corrections while a goal
        # is already executing.
        part = goal.part_class
        track = self.pose_source == "track"
        self._grip(.62 if part == "BOLT_A" else .55, settle=.1)
        grasp_goal = None
        reference = self.demo.quaternion_matrix(*initial_orientation)[:2, 2]
        base_rotation = rotation
        for clearance in (.05, .035, .023, .014, .007, 0., 0., 0., 0., 0.):
            self._check_cancel()
            sample, velocity = tracker.recent_motion(prefer_gripper=track and not clearance)
            if track:
                self._record_truth(f"follow:{clearance:.3f}", goal, sample[0])
                if not clearance:
                    self.node.get_logger().info(f"SortPart grasp_source={tracker.grasp_source} goal={goal.goal_id}")
            age = self.node.get_clock().now().nanoseconds - sample[2]
            if not -250_000_000 <= age <= 250_000_000:
                raise SortFailure("SIM_POSE_STALE")
            current = self.arm.joints()
            start = self.demo.forward(self.chain, current)[:3,3]
            axis = self.demo.quaternion_matrix(*sample[1])[:2, 2]
            if np.linalg.norm(axis) < .7:
                raise SortFailure("PART_TILTED")
            angle = np.arctan2(reference[0] * axis[1] - reference[1] * axis[0], np.dot(reference, axis))
            c, s = np.cos(angle), np.sin(angle)
            yaw = np.array([[c, -s], [s, c]])
            target = at_pick.copy()
            target[:2] = sample[0][:2] + yaw @ (at_pick[:2] - initial[:2]) + np.clip(velocity, -.05, .05) * .36
            target[2] += clearance + self.grasp_height_offset[part]
            if not clearance:
                self._event(goal, "final_grasp_target", target_tool_position_m=target.tolist(),
                            clearance_m=clearance,
                            height_offset_m=self.grasp_height_offset[part],
                            part_position_m=sample[0].tolist(), part_velocity_mps=velocity.tolist())
            # Track z is the A projection plane, not the simulation part origin; only xy is checked there.
            if not (PICK_X_M[0] <= target[0] <= PICK_X_M[1] and abs(target[1]) <= PICK_ABS_Y_M
                    and (track or .097 <= sample[0][2] <= .12)):
                raise SortFailure("PART_LEFT_PICK_REGION")
            closing = yaw @ base_rotation[:2, 0]
            closing = np.r_[closing, 0.]
            if np.dot(closing, rotation[:, 0]) < 0.:
                closing = -closing
            rotation = np.column_stack([closing, np.cross(DOWN, closing), DOWN])
            self._follow(self._line(start, target, rotation, current), .32)
            if not clearance and grasp_goal is None:
                checkpoint["bind_part"] = lambda: self._result_source(tracker, initial, offset)[0]
                grasp_goal = self._start_watched_close(part)
        self._check_cancel()
        result, _ = self._result_source(tracker, initial, offset)
        checkpoint["part"] = result
        grasping_position = result.pose()[0]
        current = self.arm.joints()
        start = self.demo.forward(self.chain, current)[:3,3]
        probe = start + np.array([0., 0., .020])
        self._follow(self._line(start, probe, rotation, current), .55)
        moved = result.pose()[0] - grasping_position
        self._event(goal, "gripper_closed", joint_position_rad=float(self.arm.state[self.demo.GRIPPER_JOINT]),
                    **(self.grip_outcome or {"trigger": "pending"}))
        if moved[2] < .014 or np.linalg.norm(moved - [0., 0., .020]) > .008:
            self._finish_grip()
            self._grip(self.demo.GRIP_OPEN)
            raise SortFailure("MISSED")
        self.holding_object = True             # #380: the probe lift moved the part with the fingers
        self._contact_seen = True              # kept until the gripper opens (a later grip result cannot clear it)
        current = self.arm.joints()
        probe = self.demo.forward(self.chain, current)[:3,3]
        target = np.r_[probe[:2], raised[2]]
        self._follow(self._line(probe, target, rotation, current), 1.2)
        return result, initial[2] if not track else grasping_position[2]

    def _closer(self, part_class):
        return ContactCloser(PARTS[part_class][1] * 2000.)

    def _grip_sample(self, closer):
        joint = self.arm.state.get(self.demo.GRIPPER_JOINT)
        closer.add(self._now_ns(), joint)

    def _close_on_part(self, goal):
        """Stationary close (#380, A decision 2): close on contact; after a belt block open, rise
        BLOCKED_RISE_M and close once more; blocked again is GRIPPER_BLOCKED_LOW."""
        for attempt in (0, 1):
            outcome = self._close_once(goal.part_class)
            if outcome["trigger"] != "blocked_low":
                break
            self._event(goal, "gripper_blocked_low", attempt=attempt, **outcome)
            self.node.get_logger().warning(f"SortPart GRIPPER_BLOCKED_LOW goal={goal.goal_id} attempt={attempt} "
                                           f"joint={outcome['joint_rad']}")
            self._grip(self.demo.GRIP_OPEN, settle=.1)
            if attempt:
                raise SortFailure("GRIPPER_BLOCKED_LOW")
            self._rise(BLOCKED_RISE_M)
        self._event(goal, "gripper_closed", joint_position_rad=float(self.arm.state[self.demo.GRIPPER_JOINT]),
                    attempts=attempt + 1, **outcome)

    def _close_once(self, part_class):
        """Start toward CLOSE_TARGET and re-command on the ContactCloser decision. Returns the outcome
        record; GRIP_TIMEOUT after CLOSE_LIMIT_NS of samples (wall guard x CLOSE_REAL_FACTOR)."""
        closer = self._closer(part_class)
        self._contact_seen = False
        handle = self._start_grip(CLOSE_TARGET)
        first = handle.get_result_async() if handle is not None else None
        wall_end = time.monotonic() + CLOSE_LIMIT_NS / 1e9 * CLOSE_REAL_FACTOR
        while True:
            self._check_cancel()
            self._grip_sample(closer)
            action, value, trigger = closer.decide(first is not None and first.done())
            if action == "wait" and time.monotonic() >= wall_end:
                action, value, trigger = "timeout", closer.latest(), "timeout"
            if action != "wait":
                break
            rclpy.spin_once(self.node, timeout_sec=.01)
        outcome = {"trigger": trigger, "contact_rad": closer.contact, "joint_rad": value if action != "recommand"
                   else closer.latest(), "cap_rad": round(closer.cap, 4), "k_min_rad": round(closer.k_min, 4),
                   "ignored_stalls_rad": closer.ignored_stalls}
        if action == "recommand":
            self.active_grip = None                 # preempted by the re-command; its end is not a failure
            self._contact_seen = trigger == "stall"
            outcome["target_rad"] = round(value, 4)
            self._grip(value, settle=.1)
        elif action == "no_contact":
            self._finish_grip()
        elif action == "blocked":
            self.active_grip = None                 # the caller opens, which preempts it
        else:
            self._finish_grip()
            self._grip(self.demo.GRIP_OPEN, settle=.1)
            raise SortFailure("GRIP_TIMEOUT")
        self.grip_outcome = outcome
        return outcome

    def _start_watched_close(self, part_class):
        """Moving close (#380): start toward CLOSE_TARGET (replaces f18cd7d's fixed 0.75 for BOLT_A) and
        let _dispatch re-command on contact while the arm keeps moving."""
        closer = self._closer(part_class)
        self._contact_seen = False
        self.grip_outcome = None
        handle = self._start_grip(CLOSE_TARGET)
        self._grip_watch = (closer, handle.get_result_async() if handle is not None else None)
        return handle

    def _watch_grip(self):
        """One check of the moving close; a decision ends the watch. Re-command asynchronously (the arm
        is moving); a belt block or no contact is left to the lift probe (MISSED)."""
        if self._grip_watch is None:
            return
        closer, first = self._grip_watch
        self._grip_sample(closer)
        action, value, trigger = closer.decide(first is not None and first.done())
        if action == "wait":
            return
        self._grip_watch = None
        self.grip_outcome = {"trigger": trigger, "contact_rad": closer.contact, "cap_rad": round(closer.cap, 4),
                             "k_min_rad": round(closer.k_min, 4), "ignored_stalls_rad": closer.ignored_stalls,
                             "joint_rad": closer.latest()}
        if action == "recommand":
            self._contact_seen = trigger == "stall"
            self.grip_outcome["target_rad"] = round(value, 4)
            self._start_grip(value)

    def _rise(self, dz):
        """Straight up by dz from the current tool pose (same orientation)."""
        current = self.arm.joints()
        pose = self.demo.forward(self.chain, current)
        start = pose[:3, 3]
        try:
            path = self._line(start, start + [0., 0., dz], pose[:3, :3], current)
        except SortFailure:
            raise
        except RuntimeError:
            raise SortFailure("IK_NO_SOLUTION")
        self._follow(path, .3)

    def _start_grip(self, position=None):
        from control_msgs.action import ParallelGripperCommand
        goal = ParallelGripperCommand.Goal()
        goal.command = JointState(name=[self.demo.GRIPPER_JOINT], position=[self.demo.GRIP_CLOSED if position is None else position])
        future = self.arm.gripper.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=5.)
        if not future.done():
            raise SortFailure("STOP_UNCONFIRMED")
        handle = future.result()
        if not handle.accepted:
            raise SortFailure("CONTROLLER_REJECTED_GOAL")
        self._grip_command_revision += 1
        self._active_grip_revision = self._grip_command_revision
        self._active_grip_target = goal.command.position[0]
        self.active_grip = handle
        return handle

    def _spin_for(self, seconds):
        stop = time.monotonic() + seconds
        while time.monotonic() < stop:
            rclpy.spin_once(self.node, timeout_sec=.05)

    def _wait_for_bin(self, tracker, bin_xy):
        deadline = self.node.get_clock().now().nanoseconds + 3_000_000_000
        stable_since = None
        previous = None
        for _ in range(150):
            self.arm.wait(.1)
            self._check_cancel()
            p = tracker.pose()[0]
            now = self.node.get_clock().now().nanoseconds
            inside = np.all(np.abs(p[:2] - bin_xy) < [.05, .07]) and .002 < p[2] < .07
            if inside and previous is not None and np.linalg.norm(previous - p) < .001:
                stable_since = now if stable_since is None else stable_since
                if now - stable_since >= 250_000_000:
                    return now
            else:
                stable_since = None
            previous = p
            if now >= deadline:
                break
        raise SortFailure("BIN_ARRIVAL_UNCONFIRMED")
