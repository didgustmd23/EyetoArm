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

PARTS = {"BOLT_A": ("part_00", 0.0015, (0.72, -0.05)),
         "BOLT_B": ("part_01", 0.003, (0.72, 0.17))}
DOWN = np.array([0., 0., -1.])


class SortFailure(RuntimeError):
    """A failure that can be returned to B as a reason code."""


class SortCanceled(RuntimeError):
    """Controller goals ended and subsequent joint feedback confirmed a stop."""


class GazeboSortExecutor:
    """Use B's static pick target and verify it against the simulation."""

    def __init__(self, node, project_root: Path, *, moving=False):
        self.node = node
        path = project_root / "scripts" / "pick_bolt_demo.py"
        spec = importlib.util.spec_from_file_location("c_pick_bolt_kinematics", path)
        self.demo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.demo)
        self.chain = self.demo.load_chain()
        self.arm = None
        self.moving = moving
        self.cancel_requested = lambda: False
        self.stationary_samples = 0
        self.active_grip = None
        self.node.create_subscription(JointState, "/joint_states", self._observe_velocity, 10)

    def _observe_velocity(self, message):
        joints = dict(zip(message.name, message.velocity))
        names = [*self.demo.JOINTS, self.demo.GRIPPER_JOINT]
        if all(name in joints and abs(joints[name]) <= .005 for name in names):
            self.stationary_samples += 1
        else:
            self.stationary_samples = 0

    def _confirm_stopped(self):
        self.stationary_samples = 0
        stop = time.monotonic() + 5.
        while self.stationary_samples < 5 and time.monotonic() < stop:
            rclpy.spin_once(self.node, timeout_sec=.05)
        if self.stationary_samples < 5:
            raise SortFailure("STOP_UNCONFIRMED")

    def _check_cancel(self):
        if self.cancel_requested():
            self._finish_grip()
            self._confirm_stopped()
            raise SortCanceled()

    def _finish_grip(self):
        if self.active_grip is None:
            return
        handle, self.active_grip = self.active_grip, None
        result = handle.get_result_async()
        if not result.done():
            handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self.node, result, timeout_sec=5.)
        if not result.done():
            raise SortFailure("STOP_UNCONFIRMED")

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
        error = self.arm.follow(path, seconds)
        if error > .015:
            target = np.asarray(path[-1], dtype=float)
            actual = self.arm.joints()
            self.node.get_logger().error(
                f"ARM_TRACKING_ERROR max_error={error:.6f} rad "
                f"target={np.round(target, 4).tolist()} actual={np.round(actual, 4).tolist()}")
            raise SortFailure("ARM_TRACKING_ERROR")

    def _check_stationary(self, tracker, initial):
        if np.linalg.norm(tracker.pose()[0] - initial) > .002:
            raise SortFailure("PART_MOVED")

    def execute(self, goal, feedback: Callable[[str], None], cancel_requested: Callable[[], bool]):
        self.cancel_requested = cancel_requested
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

            def recent_motion(self):
                with self._lock:
                    sample = self._pose
                    old = next((pose for pose in reversed(self.history)
                                if pose[2] <= sample[2] - 250_000_000), None)
                if old is None or sample[2] - old[2] > 500_000_000:
                    raise SortFailure("SIM_POSE_STALE")
                return sample, (sample[0][:2] - old[0][:2]) / ((sample[2] - old[2]) / 1e9)

        tracker = PoseTracker("sort_cell", name)
        try:
            return self._pick_and_place(goal, feedback, tracker, observed,
                                        place, place_rotation, heights, radius, bin_xy)
        finally:
            tracker._node.unsubscribe("/world/sort_cell/pose/info")
            self._finish_grip()

    def _deadline(self, goal):
        if goal.valid_until_ns:
            now = self.node.get_clock().now().nanoseconds
            if now <= 0:
                raise SortFailure("SIM_CLOCK_UNAVAILABLE")
            if goal.capture_stamp_ns > now + 50_000_000:
                raise SortFailure("CAPTURE_IN_FUTURE")
            if now >= goal.valid_until_ns:
                raise SortFailure("GOAL_EXPIRED")

    def _pick_and_place(self, goal, feedback, tracker, observed, place,
                        place_rotation, heights, radius, bin_xy):
        feedback("PLANNING")
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
        if not self.moving:
            self._check_stationary(tracker, start[0])
        if self.arm is None:
            self.arm = self.demo.Arm(self.node)
            self.arm._call = self._dispatch
        current = self.arm.joints()
        offset = self.demo.FINGER_REACH + self.demo.TIP_CLEARANCE - radius
        candidates = []
        for candidate in goal.grasp_pose_candidates:
            pick = np.asarray(candidate.position_mm) / 1000.
            if (not np.all(np.isfinite(pick)) or np.linalg.norm(pick - observed) > .035
                    or not .097 <= pick[2] <= .115
                    or not np.all((heights >= pick[2] + .05) & (heights <= .35))):
                continue
            try:
                rotation = self._rotation(candidate.orientation_xyzw)
                z = pick[2] + offset
                pick_xy = pick[:2] + correction[:2]
                above = np.r_[pick_xy + (velocity[:2] * 2. if self.moving else 0), heights[0] + offset]
                at_pick = np.r_[pick_xy, z]
                raised = np.r_[pick_xy, heights[1] + offset]
                release_z = place[2] + self.demo.FINGER_REACH
                over_bin = np.r_[place[:2], max(raised[2], release_z)]
                release = np.r_[place[:2], release_z]
                approach = self._solve(above, rotation, current)
                descent = self._line(above, at_pick, rotation, approach)
                lift = self._line(at_pick, raised, rotation, descent[-1])
                carry = self._solve(over_bin, place_rotation, lift[-1])
                lowering = self._line(over_bin, release, place_rotation, carry)
            except RuntimeError:
                continue
            cost = np.max(np.abs(approach - current))
            candidates.append((cost, approach, descent, lift, carry, lowering, rotation, at_pick, raised))
        if not candidates:
            raise SortFailure("IK_NO_SOLUTION")
        _, approach, descent, lift, carry, lowering, rotation, at_pick, raised = min(
            candidates, key=lambda plan: plan[0])
        self._deadline(goal)
        feedback("EXECUTING")
        if abs(self.arm.state[self.demo.GRIPPER_JOINT] - self.demo.GRIP_OPEN) > .015:
            self.arm.grip(self.demo.GRIP_OPEN)
        self._deadline(goal)
        self._follow([approach], 3.)
        if self.moving:
            self._moving_grasp(tracker, initial, initial_orientation, at_pick, raised,
                               rotation, goal.part_class)
            # The transport endpoints are derived from B's placement, with IK seeded
            # from the actual lift pose after the moving grasp.
            over_pose = self.demo.forward(self.chain, carry)
            carry = self._solve(over_pose[:3,3], place_rotation, self.arm.joints())
            release_pose = self.demo.forward(self.chain, lowering[-1])
            lowering = self._line(over_pose[:3,3], release_pose[:3,3], place_rotation, carry)
        else:
            self._check_stationary(tracker, initial)
            self._follow(descent, 2.)
            try:
                self._check_stationary(tracker, initial)
            except SortFailure:
                self._follow(list(reversed(descent[:-1])) + [approach], 2.)
                raise
            self.arm.grip(self.demo.GRIP_CLOSED, settle=.1)
            self._follow(lift, 2.)
        if tracker.pose()[0][2] - initial[2] < .025:
            self.arm.grip(self.demo.GRIP_OPEN)
            raise SortFailure("MISSED")
        feedback("PLACING")
        self._follow([carry], 3.)
        self._follow(lowering, 1.5)
        held = tracker.pose()[0]
        if np.any(np.abs(held[:2] - bin_xy) >= [.05, .07]) or held[2] <= .08:
            raise SortFailure("PART_LOST_BEFORE_RELEASE")
        self.arm.grip(self.demo.GRIP_OPEN)
        arrival = self._wait_for_bin(tracker, bin_xy)
        self._follow([carry], 1.5)
        return arrival

    def _moving_grasp(self, tracker, initial, initial_orientation, at_pick, raised, rotation, part):
        # B's candidate sets the point and direction. Gazebo pose increments
        # supply relative XY corrections while a goal is already executing.
        self.arm.grip(.62 if part == "BOLT_A" else .55, settle=.1)
        grasp_goal = None
        reference = self.demo.quaternion_matrix(*initial_orientation)[:2, 2]
        base_rotation = rotation
        for clearance in (.05, .035, .023, .014, .007, 0., 0., 0., 0., 0.):
            self._check_cancel()
            sample, velocity = tracker.recent_motion()
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
            target[2] += clearance + self.demo.GRASP_HEIGHT_OFFSET[PARTS[part][0]]
            if not (.46 <= target[0] <= .64 and abs(target[1]) <= .53 and .097 <= sample[0][2] <= .12):
                raise SortFailure("PART_LEFT_PICK_REGION")
            closing = yaw @ base_rotation[:2, 0]
            closing = np.r_[closing, 0.]
            if np.dot(closing, rotation[:, 0]) < 0.:
                closing = -closing
            rotation = np.column_stack([closing, np.cross(DOWN, closing), DOWN])
            self._follow(self._line(start, target, rotation, current), .32)
            if not clearance and grasp_goal is None:
                grasp_goal = self._start_grip()
        self._check_cancel()
        grasping_position = tracker.pose()[0]
        current = self.arm.joints()
        start = self.demo.forward(self.chain, current)[:3,3]
        probe = start + np.array([0., 0., .020])
        self._follow(self._line(start, probe, rotation, current), .55)
        moved = tracker.pose()[0] - grasping_position
        if moved[2] < .014 or np.linalg.norm(moved - [0., 0., .020]) > .008:
            self._finish_grip()
            self.arm.grip(self.demo.GRIP_OPEN)
            raise SortFailure("MISSED")
        current = self.arm.joints()
        probe = self.demo.forward(self.chain, current)[:3,3]
        target = np.r_[probe[:2], raised[2]]
        self._follow(self._line(probe, target, rotation, current), 1.2)

    def _start_grip(self):
        from control_msgs.action import ParallelGripperCommand
        goal = ParallelGripperCommand.Goal()
        goal.command = JointState(name=[self.demo.GRIPPER_JOINT], position=[self.demo.GRIP_CLOSED])
        future = self.arm.gripper.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=5.)
        if not future.done():
            raise SortFailure("STOP_UNCONFIRMED")
        handle = future.result()
        if not handle.accepted:
            raise SortFailure("CONTROLLER_REJECTED_GOAL")
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
