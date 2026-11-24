"""Gripper close-on-contact decision for small bolts (A #346·#351·#358, #380).

Closing straight to 0.8 kept squeezing a contacted M3 until it turned in the fingers. Instead the
gripper starts toward 0.8 and, on contact, is re-commanded to the contact value plus a little, capped
per bolt diameter. This module only decides; the executor sends the commands. No ROS imports.
"""
from __future__ import annotations

import math
from collections import deque

CLOSE_TARGET = 0.8                 # first command (fully closed); the controller preempts it later
DELTA_RAD = .005                   # closed beyond the contact value
STALL_WINDOW_NS = 100_000_000      # stall: joint within STALL_EPS_RAD for 0.1 s SIM (#351, #358 §5 2)
STALL_EPS_RAD = .002
MOVING_MIN_RAD = .05               # a stall at or below this is the open gripper, not a pause
BLOCKED_LOW_NS = 1_000_000_000     # a stall below k_min longer than this: fingers on the belt (#358 §5 3)
CLOSE_LIMIT_NS = 10_000_000_000    # whole close, SIM time (A decision 3)
CLOSE_REAL_FACTOR = 3.             # wall-clock guard against a stopped SIM clock


def cap_for(diameter_mm):
    """Highest re-command value for a bolt shaft diameter (#351 §3.1; checked for M3 and M6 only)."""
    return .849 - .031 * float(diameter_mm)


def k_min_for(diameter_mm):
    """Stalls below this are pauses while closing, not contact (#351 §3.1)."""
    return .807 - .0357 * float(diameter_mm)


class ContactCloser:
    """Feed (SIM stamp ns, gripper joint rad) samples, then ask decide(first_done).

    decide returns one of
      ("wait", None, None)                 keep closing
      ("recommand", target, trigger)       send target; trigger is "cap", "stall" or "goal_done"
      ("blocked", joint, "blocked_low")    stalled below k_min longer than BLOCKED_LOW_NS
      ("no_contact", joint, "goal_done")   the first command ended below k_min
      ("timeout", joint, "timeout")        CLOSE_LIMIT_NS of samples without a decision
    Only "stall" means the fingers stopped on something; "cap" and "goal_done" are also reached
    with an empty hand.
    """

    def __init__(self, diameter_mm):
        self.cap = cap_for(diameter_mm)
        self.k_min = k_min_for(diameter_mm)
        self.reset()

    def reset(self):
        """Forget all samples (a new close starts)."""
        self._samples = deque()
        self._first_ns = None
        self._low_since = None
        self.ignored_stalls = []
        self.contact = None

    def add(self, stamp_ns, joint):
        """Add one sample; non-finite values and non-increasing stamps are ignored."""
        if joint is None or not math.isfinite(float(joint)) or stamp_ns is None:
            return
        if self._samples and stamp_ns <= self._samples[-1][0]:
            return
        self._samples.append((int(stamp_ns), float(joint)))
        if self._first_ns is None:
            self._first_ns = int(stamp_ns)
        while len(self._samples) > 2 and self._samples[1][0] <= stamp_ns - STALL_WINDOW_NS:
            self._samples.popleft()
        # Low-stall timer, kept per sample so it does not depend on how often decide() is asked.
        joint = float(joint)
        if self._stalled() and MOVING_MIN_RAD < joint < self.k_min:
            if self._low_since is None:
                self._low_since = self._samples[0][0]
                self.ignored_stalls.append(round(joint, 4))
        else:
            self._low_since = None

    def latest(self):
        return self._samples[-1][1] if self._samples else None

    def span_ns(self):
        """SIM time covered since the first sample (the close limit runs on sample stamps, #394 H-ES)."""
        if not self._samples:
            return 0
        return self._samples[-1][0] - self._first_ns

    def _stalled(self):
        if len(self._samples) < 2 or self._samples[-1][0] - self._samples[0][0] < STALL_WINDOW_NS:
            return False
        values = [q for _, q in self._samples]
        return max(values) - min(values) <= STALL_EPS_RAD

    def _target(self, joint):
        return min(CLOSE_TARGET, joint + DELTA_RAD, self.cap)

    def decide(self, first_done=False):
        joint = self.latest()
        if joint is None:
            return "wait", None, None
        stamp = self._samples[-1][0]
        if joint >= self.cap:
            self.contact = joint
            return "recommand", self.cap, "cap"
        if self._stalled() and joint > MOVING_MIN_RAD:
            if joint >= self.k_min:
                self.contact = joint
                return "recommand", self._target(joint), "stall"
            if self._low_since is not None and stamp - self._low_since > BLOCKED_LOW_NS:
                return "blocked", joint, "blocked_low"
        if first_done:
            if joint >= self.k_min:
                self.contact = joint
                return "recommand", self._target(joint), "goal_done"
            return "no_contact", joint, "goal_done"
        if self.span_ns() >= CLOSE_LIMIT_NS:
            return "timeout", joint, "timeout"
        return "wait", None, None
