"""Select one A track from the A track stream and predict it for the executor.

This module does not import rclpy or Gazebo. The executor feeds it
TrackStateArray messages (or any object with the same fields) and reads
positions through the same pose()/recent_motion() calls as the simulation
pose tracker. Simulation poses are never passed into this module.
"""
from __future__ import annotations

import math
from threading import Lock

import numpy as np

STALE_NS = 500_000_000          # capture age above this is TRACK_STALE
JUMP_M = .025                   # new position vs previous prediction above this is TRACK_JUMP
OBSERVATION_M = .025            # goal observation vs track at capture time
STILL_M = .003                  # stationary check threshold in track mode (measured position only)
GRIPPER_STATE = "GRIPPER"


class TrackFailure(RuntimeError):
    """A track stream failure with a reason code."""


def yaw_quaternion(yaw_rad: float) -> tuple:
    """Horizontal part pose whose z axis is the bolt axis, as (w, x, y, z)."""
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)

    # Columns: x = (-s, c, 0), y = up, z = axis. Right handed: x = y cross z.
    m = np.array([[-s, 0., c],
                  [c, 0., s],
                  [0., 1., 0.]])

    return _matrix_quaternion(m)


def _matrix_quaternion(m) -> tuple:
    trace = m[0, 0] + m[1, 1] + m[2, 2]

    if trace > 0:
        k = math.sqrt(trace + 1.) * 2
        q = (k / 4, (m[2, 1] - m[1, 2]) / k, (m[0, 2] - m[2, 0]) / k, (m[1, 0] - m[0, 1]) / k)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        k = math.sqrt(1. + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = ((m[2, 1] - m[1, 2]) / k, k / 4, (m[0, 1] + m[1, 0]) / k, (m[0, 2] + m[2, 0]) / k)
    elif m[1, 1] > m[2, 2]:
        k = math.sqrt(1. + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = ((m[0, 2] - m[2, 0]) / k, (m[0, 1] + m[1, 0]) / k, k / 4, (m[1, 2] + m[2, 1]) / k)
    else:
        k = math.sqrt(1. + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = ((m[1, 0] - m[0, 1]) / k, (m[0, 2] + m[2, 0]) / k, (m[1, 2] + m[2, 1]) / k, k / 4)

    return q if q[0] >= 0 else tuple(-v for v in q)


class TrackSample:
    """One received state of the selected track (SI units)."""

    def __init__(self, track, frame):
        self.position = np.asarray(track.position_mm, dtype=float) / 1000.
        velocity = np.asarray(track.velocity_mm_per_s, dtype=float) / 1000.
        self.velocity = velocity if track.velocity_source != "NONE" and np.all(np.isfinite(velocity)) else np.zeros(3)
        self.yaw = math.radians(float(track.axis_yaw_deg))
        self.state = str(track.state)
        self.camera = str(track.last_hit_camera)
        self.stamp_ns = int(frame.capture_stamp_ns)
        self.valid = bool(np.all(np.isfinite(self.position)))   # yaw may be NaN (A does not know the axis)

    @property
    def gripper(self) -> bool:
        """Updated by the gripper camera."""
        return self.state == GRIPPER_STATE or self.camera == "gripper"

    def predict(self, stamp_ns: int):
        """Position at stamp_ns from this sample's velocity."""
        return self.position + self.velocity * ((stamp_ns - self.stamp_ns) / 1e9)


class TrackStreamTracker:
    """Track one goal part by part_track_id. Same calls as the simulation pose tracker.

    Messages arrive on the executor node's subscription callback; reads come from
    the executing goal. Both go through one lock.
    """

    def __init__(self, part_track_id, part_class, clock_ns, *, failure=TrackFailure, stale_ns=STALE_NS):
        self.failure = failure
        if not part_track_id:
            raise failure("TRACK_ID_MISSING")
        self.part_track_id = str(part_track_id)
        self.part_class = str(part_class)
        self.clock_ns = clock_ns
        self.stale_ns = stale_ns
        self.still_threshold = STILL_M
        self._lock = Lock()
        self._key = None                # (run_id, tracker_epoch) of the first frame used
        self._sample = None
        self._gripper_sample = None
        self._handoff_on = None
        self._problem = None            # sticky failure reason
        self._reference_yaw = None
        self._known_yaw = None          # last finite axis yaw of this track
        self.grasp_source = None
        self.frames = 0
        self._last_frame_ns = None      # newest frame capture seen, with or without the selected id

    @property
    def last_frame_ns(self):
        """Capture time of the newest frame received (None before any)."""
        with self._lock:
            return self._last_frame_ns

    @property
    def failed(self) -> bool:
        """A sticky failure was recorded."""
        with self._lock:
            return self._problem is not None

    @property
    def has_sample(self) -> bool:
        """A sample of the selected track was received."""
        with self._lock:
            return self._sample is not None or self._problem is not None

    def on_message(self, message) -> None:
        """Take one TrackStateArray. Reset, run end, epoch change or a missing id ends the track."""
        kind = str(message.kind)
        with self._lock:
            if self._problem is not None:
                return

            if kind in ("reset", "marker"):
                if self._sample is not None or self._key is not None:
                    self._problem = "TRACK_LOST"
                return

            if kind != "frame":
                return

            stamp = int(message.capture_stamp_ns)
            if self._last_frame_ns is None or stamp > self._last_frame_ns:
                self._last_frame_ns = stamp

            key = (str(message.run_id), int(message.tracker_epoch))
            if self._key is None:
                self._key = key
            elif key != self._key:
                self._problem = "TRACK_LOST"
                return

            if self._sample is not None and int(message.capture_stamp_ns) <= self._sample.stamp_ns:
                return

            self.frames += 1
            self._handoff_on = bool(message.handoff_on)
            track = next((t for t in message.tracks if str(t.part_track_id) == self.part_track_id), None)

            if track is None:
                if self._sample is not None:
                    self._problem = "TRACK_LOST"
                return

            # An empty part_class means A does not know the kind; only a different known kind fails.
            if str(track.part_class) and str(track.part_class) != self.part_class:
                self._problem = "TRACK_CLASS_MISMATCH"
                return

            sample = TrackSample(track, message)
            if not sample.valid:
                self._problem = "TRACK_INVALID"
                return

            if self._sample is not None and np.linalg.norm(sample.position[:2] - self._sample.predict(sample.stamp_ns)[:2]) > JUMP_M:
                self._problem = "TRACK_JUMP"
                return

            self._sample = sample
            if math.isfinite(sample.yaw):
                self._known_yaw = sample.yaw
            if sample.gripper:
                self._gripper_sample = sample

    def _current(self, prefer_gripper=False):
        now = self.clock_ns()
        with self._lock:
            if self._problem is not None:
                raise self.failure(self._problem)
            sample = self._sample
            gripper = self._gripper_sample

        if sample is None:
            raise self.failure("TRACK_NOT_FOUND")

        if prefer_gripper:
            if gripper is not None and now - gripper.stamp_ns <= self.stale_ns:
                sample, self.grasp_source = gripper, "gripper"
            else:
                self.grasp_source = "link_pred"

        if not -50_000_000 <= now - sample.stamp_ns <= self.stale_ns:
            raise self.failure("TRACK_STALE")

        return sample, now

    def _orientation(self, yaw, require_axis):
        # An unknown (NaN) yaw falls back to the last known one; with none the axis is unknown.
        if not math.isfinite(yaw):
            with self._lock:
                yaw = self._known_yaw
            if yaw is None:
                if require_axis:
                    raise self.failure("TRACK_AXIS_UNKNOWN")
                return None
        # The axis yaw is folded to [-90, 90). Keep it within 90 degrees of the first one used.
        if self._reference_yaw is None:
            self._reference_yaw = yaw
        while yaw - self._reference_yaw > math.pi / 2:
            yaw -= math.pi
        while yaw - self._reference_yaw <= -math.pi / 2:
            yaw += math.pi
        return yaw_quaternion(yaw)

    def pose(self, require_axis=False):
        """(xyz m predicted to now, (w, x, y, z) or None if the axis is unknown, now ns)."""
        sample, now = self._current()
        return sample.predict(now), self._orientation(sample.yaw, require_axis), now

    def recent_motion(self, prefer_gripper=False):
        """((xyz m, (w, x, y, z), now ns), xy velocity m/s). Gripper samples first when asked; needs the axis."""
        sample, now = self._current(prefer_gripper)
        return (sample.predict(now), self._orientation(sample.yaw, True), now), sample.velocity[:2]

    def velocity(self):
        """Track velocity m/s used for prediction."""
        sample, _ = self._current()
        return sample.velocity

    def handoff_on(self):
        """handoff_on of the latest frame, None before any frame."""
        with self._lock:
            return self._handoff_on

    def check_observation(self, observed, capture_stamp_ns, failure_reason="PART_NOT_AT_OBSERVATION"):
        """Goal observation must be within 25 mm (xy) of this track predicted to the goal capture time."""
        sample, _ = self._current()
        if np.linalg.norm(sample.predict(capture_stamp_ns)[:2] - np.asarray(observed, dtype=float)[:2]) > OBSERVATION_M:
            raise self.failure(failure_reason)

    def measured_position(self):
        """Last measured position (not predicted)."""
        sample, _ = self._current()
        return sample.position

    def check_stationary(self, initial):
        """Measured track within the stationary threshold. The A belt speed is not used: it is
        noisy while the belt is stopped (several mm/s)."""
        sample, _ = self._current()
        if np.linalg.norm(sample.position - np.asarray(initial, dtype=float)) > self.still_threshold:
            raise self.failure("PART_MOVED")

    def close(self):
        """Nothing to release; the executor owns the subscription."""


class SimExecutionFeedback:
    """Execution result feedback from simulation poses (lift, hold, bin arrival).

    The part is the one nearest the closed fingers, bound once after closing.
    It is never used to choose or correct a pick position.
    """

    def __init__(self, poses, failure=TrackFailure, radius=.03):
        self._poses = poses          # callable -> {name: (xyz m, (w, x, y, z))}
        self.failure = failure
        self.radius = radius
        self.part = None

    def bind(self, finger_point):
        """Choose the simulation part nearest finger_point. None in range is MISSED."""
        finger_point = np.asarray(finger_point, dtype=float)
        best = None
        for name, (xyz, _) in self._poses().items():
            distance = np.linalg.norm(np.asarray(xyz) - finger_point)
            if distance <= self.radius and (best is None or distance < best[0]):
                best = (distance, name)
        if best is None:
            raise self.failure("MISSED")
        self.part = best[1]
        return self.pose()

    def pose(self):
        """(xyz m, (w, x, y, z)) of the bound part."""
        if self.part is None:
            raise self.failure("MISSED")
        poses = self._poses()
        if self.part not in poses:
            raise self.failure("FEEDBACK_LOST")
        return poses[self.part]


class SimTruthLog:
    """Write track-vs-simulation differences for evaluation only. Returns nothing."""

    def __init__(self, poses, stream):
        self._poses = poses
        self._stream = stream

    def record(self, stage, goal_id, part_track_id, track_xyz, stamp_ns) -> None:
        import json

        try:
            poses = self._poses()
            nearest = min(((np.linalg.norm(np.asarray(xyz)[:2] - np.asarray(track_xyz)[:2]), name)
                           for name, (xyz, _) in poses.items()), default=None)
            line = {"stage": stage, "goal_id": goal_id, "part_track_id": part_track_id, "stamp_ns": stamp_ns,
                    "track_mm": [round(float(v) * 1000., 2) for v in track_xyz],
                    "nearest_part": None if nearest is None else nearest[1],
                    "error_xy_mm": None if nearest is None else round(float(nearest[0]) * 1000., 2)}
            self._stream.write(json.dumps(line, separators=(",", ":")) + "\n")
            self._stream.flush()
        except (OSError, ValueError, TypeError):  # evaluation log only; never affects execution
            pass
