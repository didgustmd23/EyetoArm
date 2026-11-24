"""Arm stop detection from joint positions on the SIM clock (no ROS imports).

The executor used to require five /joint_states velocity samples at or below 0.005 rad/s
for the arm joints and the gripper joint. While the gripper holds a small bolt with its
close command active, the gripper joint velocity keeps jittering, so a stopped arm was
never confirmed (STOP_UNCONFIRMED, A #354 D-13). This detector looks only at the arm
joints and at how far they moved within a SIM-time window.
"""
from __future__ import annotations

from collections import deque

STOP_WINDOW_NS = 250_000_000      # SIM time the arm must stay within the tolerance
STOP_TOLERANCE_RAD = 1e-3         # largest change of any arm joint inside the window
STOP_LIMIT_NS = 5_000_000_000     # SIM time to wait for a confirmed stop
STOP_REAL_FACTOR = 3.             # wall-clock guard against a stopped SIM clock (limit x factor)


class StopDetector:
    """Keeps recent (SIM stamp, arm joint positions) samples and answers whether the arm is still."""

    def __init__(self, joints, window_ns=STOP_WINDOW_NS, tolerance_rad=STOP_TOLERANCE_RAD):
        """Watch the named arm joints (the gripper joint is deliberately not one of them)."""
        self.joints = tuple(joints)
        self.window_ns = int(window_ns)
        self.tolerance_rad = float(tolerance_rad)
        self._samples = deque()
        self._first_ns = None

    def reset(self):
        """Forget all samples (a new stop check starts from fresh feedback)."""
        self._samples.clear()
        self._first_ns = None

    def add(self, stamp_ns, positions):
        """Add one sample. `positions` maps joint name to position; samples missing an arm joint,
        with a non-increasing stamp or a non-finite value are ignored."""
        try:
            values = tuple(float(positions[name]) for name in self.joints)
        except (KeyError, TypeError, ValueError):
            return
        if any(v != v or v in (float("inf"), float("-inf")) for v in values):
            return
        stamp_ns = int(stamp_ns)
        if self._samples and stamp_ns <= self._samples[-1][0]:
            return
        self._samples.append((stamp_ns, values))
        if self._first_ns is None:
            self._first_ns = stamp_ns
        # Keep one sample older than the window so the window is fully covered.
        while len(self._samples) > 2 and self._samples[1][0] <= stamp_ns - self.window_ns:
            self._samples.popleft()

    def span_ns(self):
        """SIM time covered by the kept samples since the last reset (0 without two samples). Used as the
        stop-check clock: a node that was not spun for a while has a stale clock that jumps on the first
        spin and would end the check before any sample arrived (#394 H-ES)."""
        if len(self._samples) < 2:
            return 0
        return self._samples[-1][0] - self._first_ns

    def stopped(self):
        """True when the kept samples span the whole window and no arm joint moved more than the tolerance."""
        if len(self._samples) < 2:
            return False
        newest = self._samples[-1][0]
        if newest - self._samples[0][0] < self.window_ns:
            return False
        recent = [values for stamp, values in self._samples if stamp >= newest - self.window_ns]
        recent.insert(0, self._samples[0][1])
        for index in range(len(self.joints)):
            column = [values[index] for values in recent]
            if max(column) - min(column) > self.tolerance_rad:
                return False
        return True
