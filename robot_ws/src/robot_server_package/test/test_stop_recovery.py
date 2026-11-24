"""#394 (A takes over C items on top of C 0d3d69b): stop detection, fixed home, start home check.

Piece ①: arm stop detection from joint positions on the SIM clock (gripper not used) and the fixed
home. Piece ②: the receiver's start/first-goal home check that uses C's move_home. Executor tests
reuse the fake world of test_track_stream_tracker; receiver tests build a real SortGoalReceiver on a
private action name with a fake backend and call its callbacks directly (no Gazebo, no controllers).
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from robot_server_package.stop_detector import STOP_TOLERANCE_RAD, STOP_WINDOW_NS, StopDetector

JOINTS = ("a", "b")
MS = 1_000_000


# ---------------------------------------------------------------- piece ①: stop detector

def feed(detector, stamps_ms, values):
    """Add samples (SIM ms; one value for both arm joints or a callable of the index); the gripper jitters."""
    for i, t in enumerate(stamps_ms):
        v = values(i) if callable(values) else values
        detector.add(t * MS, {"a": v, "b": v, "gripper": (-1) ** i * .5})


def test_still_arm_is_stopped_even_with_gripper_jitter():
    """Arm joints within 1e-9 rad over the window: stopped, whatever the gripper joint does (#354 D-13)."""
    d = StopDetector(JOINTS)
    feed(d, range(0, 400, 10), lambda i: 1. + i * 1e-11)
    assert d.stopped()


def test_moving_arm_is_not_stopped():
    """An arm joint drifting beyond the tolerance inside the window is not stopped."""
    d = StopDetector(JOINTS)
    feed(d, range(0, 400, 10), lambda i: i * STOP_TOLERANCE_RAD)
    assert not d.stopped()


def test_window_must_be_covered():
    """Samples spanning less than the window are not enough; reset forgets them."""
    d = StopDetector(JOINTS)
    feed(d, range(0, STOP_WINDOW_NS // MS - 10, 10), 0.)
    assert not d.stopped()
    feed(d, range(STOP_WINDOW_NS // MS, STOP_WINDOW_NS // MS + 30, 10), 0.)
    assert d.stopped()
    d.reset()
    assert not d.stopped()


def test_bad_samples_ignored():
    """Missing arm joint, non-finite value or a non-increasing stamp does not count."""
    d = StopDetector(JOINTS)
    d.add(0, {"a": 0.})
    d.add(10 * MS, {"a": float("nan"), "b": 0.})
    d.add(20 * MS, {"a": 0., "b": 0.})
    d.add(20 * MS, {"a": 5., "b": 5.})
    assert len(d._samples) == 1


# ---------------------------------------------------------------- piece ①: fixed home

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("action_msgs.msg")
import test_track_stream_tracker as base  # noqa: E402  (fake world, arm and goals)

NEW = base.NEW


def test_home_is_fixed_and_matches_the_model():
    """HOME_JOINTS equals the simulation model's initial_positions; the executor starts with it (#394)."""
    xacro = (base.ROOT / "simulation" / "models" / "ur5e_2f85" / "ur5e_2f85.urdf.xacro").read_text()
    found = dict(re.findall(r"(\w+_joint)=(-?[\d.]+)",
                            re.search(r"initial_positions=\"\$\{dict\(([^)]*)\)", xacro).group(1)))
    names = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
    assert tuple(float(found[n]) for n in names) == NEW.HOME_JOINTS


def test_move_home_goes_to_fixed_home_not_start_pose(monkeypatch):
    """C's move_home moves to the fixed home even when the arm started elsewhere (C item 9②)."""
    ex, _ = base.make(NEW, monkeypatch=monkeypatch)
    start = ex.arm.q.copy()
    assert np.max(np.abs(start - ex.home)) > .01
    ex.move_home()
    assert np.allclose(ex.arm.q, NEW.HOME_JOINTS)


def test_at_home(monkeypatch):
    """at_home is false for the fake arm's start pose and true after C's move_home."""
    ex, _ = base.make(NEW, monkeypatch=monkeypatch)
    assert not ex.at_home()
    ex.move_home()
    assert ex.at_home()


# ---------------------------------------------------------------- piece ②: receiver start home check

import os  # noqa: E402
import uuid  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from robot_server_package import sort_manager as SM  # noqa: E402


class FakeBackend:
    """Backend stand-in for the receiver: scripted at_home/move_home, execute returns a stamp."""

    def __init__(self, at_home=True):
        self.home = at_home
        self.calls = []
        self.home_error = None
        self.holding_object = False

    def at_home(self):
        self.calls.append("at_home")
        if isinstance(self.home, Exception):
            raise self.home
        return self.home

    def move_home(self):
        self.calls.append("move_home")
        if self.home_error is not None:
            raise self.home_error
        self.home = True

    def execute(self, request, publish, cancel_requested, *args, **kwargs):
        self.calls.append("execute")
        return 5


class FakeHandle:
    """ServerGoalHandle stand-in recording the terminal state."""

    def __init__(self, request):
        self.goal_id = SimpleNamespace(uuid=list(uuid.UUID(request.goal_id).bytes))
        self.request, self.is_cancel_requested, self.state = request, False, None

    def publish_feedback(self, feedback):
        pass

    def succeed(self):
        self.state = "succeed"

    def canceled(self):
        self.state = "canceled"

    def abort(self):
        self.state = "abort"


@pytest.fixture
def ros():
    """rclpy context for the receiver tests (no signal handlers)."""
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    yield
    rclpy.try_shutdown()


def receiver(backend):
    """Unmanaged SortGoalReceiver (no run-state) on a private action name; start timer stopped."""
    node = SM.SortGoalReceiver(f"/sort_part_test_{os.getpid()}_{uuid.uuid4().hex[:8]}", backend)
    node._start_timer.cancel()
    return node


def request():
    """A valid unmanaged SortPart goal."""
    from eyetoarm_interfaces.action import SortPart
    goal = SortPart.Goal()
    goal.goal_id = str(uuid.uuid4())
    goal.goal_kind, goal.part_class, goal.place_bin = "PICK_PLACE", "BOLT_B", "BOLT_B"
    return goal


def test_start_check_moves_an_off_home_arm_once(ros):
    """Node start: an arm off home is moved with C's move_home; the check does not repeat."""
    backend = FakeBackend(at_home=False)
    node = receiver(backend)
    try:
        node._start_check()
        node._start_check()
        assert backend.calls == ["at_home", "move_home"] and node._home_checked
        assert node._execute(FakeHandle(request())).status == "SUCCEEDED"
        assert backend.calls == ["at_home", "move_home", "execute"]
    finally:
        node.destroy_node()


def test_start_check_waits_for_controllers_and_locks_on_failure(ros):
    """Controllers not up (RuntimeError): retried later; a failed home move locks (RECOVERY_REQUIRED)."""
    backend = FakeBackend()
    backend.home = RuntimeError("제어기 액션 서버가 없다")
    node = receiver(backend)
    try:
        node._start_check()
        assert not node._home_checked and not node._recovery_required
        backend.home = False
        backend.home_error = SM.SortFailure("STOP_UNCONFIRMED")
        node._start_check()
        assert node._home_checked and node._recovery_required
        assert node._execute(FakeHandle(request())).reason == "RECOVERY_REQUIRED"
    finally:
        node.destroy_node()


def test_goal_before_start_check_checks_home_first(ros):
    """A goal arriving before the timer's check runs the home check first; later goals do not repeat it."""
    backend = FakeBackend(at_home=False)
    node = receiver(backend)
    try:
        assert node._execute(FakeHandle(request())).status == "SUCCEEDED"
        assert node._execute(FakeHandle(request())).status == "SUCCEEDED"
        assert backend.calls == ["at_home", "move_home", "execute", "execute"]
    finally:
        node.destroy_node()
