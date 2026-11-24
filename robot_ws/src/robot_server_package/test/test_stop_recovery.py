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
        self.plan = None

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
        if self.plan is not None:
            return self.plan(cancel_requested, *args)
        return 5

    def refresh_hold_from_joint(self):
        self.calls.append("refresh_hold")
        return self.holding_object


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
        assert backend.calls == ["at_home", "move_home", "refresh_hold"] and node._home_checked
        assert node._execute(FakeHandle(request())).status == "SUCCEEDED"
        assert backend.calls == ["at_home", "move_home", "refresh_hold", "execute"]
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
        assert backend.calls == ["at_home", "move_home", "refresh_hold", "execute", "execute"]
    finally:
        node.destroy_node()


# ---------------------------------------------------------------- piece ③: cancel → home, holding, latch

def _follow_checking_cancel(ex):
    """Wrap _follow so a follow checks the cancel like the real controller dispatch does."""
    original = ex._follow

    def follow(path, seconds):
        original(path, seconds)
        ex._check_cancel()
    return follow


def test_cancel_from_ab_ends_at_home(monkeypatch):
    """A cancel from A/B (home_after_cancel true): stop confirmed, rise, C's move_home → fixed home (C item 10)."""
    ex, world = base.make(NEW, monkeypatch=monkeypatch)
    ex.cancel_requested = lambda: sum(1 for e in ex.arm.log if e[0] == "follow") >= 2
    ex.home_after_cancel = lambda: True
    monkeypatch.setattr(ex, "_follow", _follow_checking_cancel(ex))
    result, _ = base.run_pick(ex, world, base.goal(world))
    assert result[0] == "SortCanceled"
    assert np.allclose(ex.arm.q, NEW.HOME_JOINTS)
    assert any("CANCEL_HOME done" in str(line) for line in ex.node.logs)


def test_cancel_from_run_stop_or_estop_stays(monkeypatch):
    """A cancel caused by stop_run or e-stop (home_after_cancel false) stays in place: C handles it."""
    ex, world = base.make(NEW, monkeypatch=monkeypatch)
    ex.cancel_requested = lambda: sum(1 for e in ex.arm.log if e[0] == "follow") >= 2
    monkeypatch.setattr(ex, "_follow", _follow_checking_cancel(ex))
    result, _ = base.run_pick(ex, world, base.goal(world))
    assert result[0] == "SortCanceled"
    assert not np.allclose(ex.arm.q, NEW.HOME_JOINTS)
    assert not any("CANCEL_HOME" in str(line) for line in ex.node.logs)


def test_cancel_holding_goes_home_closed(monkeypatch):
    """Cancel after the grasp: the fingers stopped on the bolt (#380 contact re-command), holding_object
    true, home with the gripper closed (no opening); C's stalled flag is not needed for this."""
    ex, world = base.make(NEW, monkeypatch=monkeypatch)
    ex.cancel_requested = lambda: any(e[0] == "grip" and e[1] > ex.demo.GRIP_OPEN for e in ex.arm.log)
    ex.home_after_cancel = lambda: True
    monkeypatch.setattr(ex, "_follow", _follow_checking_cancel(ex))
    result, log = base.run_pick(ex, world, base.goal(world))
    assert result[0] == "SortCanceled" and ex.holding_object
    assert np.allclose(ex.arm.q, NEW.HOME_JOINTS)
    closed_at = max(i for i, e in enumerate(log) if e[0] == "grip" and e[1] > ex.demo.GRIP_OPEN)
    assert not any(e[0] == "grip" and e[1] == ex.demo.GRIP_OPEN for e in log[closed_at:])


def test_closed_short_judgement(monkeypatch):
    """Short of the command beyond 3 mrad = holding; a full close or an opening command = not holding."""
    ex, _ = base.make(NEW, monkeypatch=monkeypatch)
    ex.arm.state[ex.demo.GRIPPER_JOINT] = .745
    assert ex._closed_short(.8) and not ex._closed_short(.747) and not ex._closed_short(0.)
    ex.arm.state[ex.demo.GRIPPER_JOINT] = .799
    assert not ex._closed_short(.8)
    assert not ex._closed_short(None)


def managed_receiver(backend, tmp_path):
    """Managed SortGoalReceiver (run-state file) on a private action name; start timer stopped; run started."""
    from std_srvs.srv import Trigger
    node = SM.SortGoalReceiver(f"/sort_part_test_{os.getpid()}_{uuid.uuid4().hex[:8]}", backend,
                               tmp_path / "run.sqlite")
    node._start_timer.cancel()
    node._home_checked = True
    assert node._start_run(None, Trigger.Response()).message == "RUN_STARTED"
    return node


def managed_request(node):
    """A valid goal for the node's current managed run."""
    goal = request()
    goal.sort_job_id, goal.run_epoch = node._job_id, node._epoch
    goal.timestamp_base, goal.capture_stamp_ns, goal.valid_until_ns = "SIM_TIME", 1, 2
    return goal


def test_receiver_passes_home_after_cancel_only_for_ab_cancel(ros, tmp_path):
    """home_after_cancel is true for an A/B cancel, false once stop_run or e-stop is in effect."""
    backend = FakeBackend()
    node = managed_receiver(backend, tmp_path)
    try:
        handle = FakeHandle(managed_request(node))
        seen = {}

        def plan(cancel, home_after_cancel):
            handle.is_cancel_requested = True
            seen["ab"] = home_after_cancel()
            node._stop_run_requested.set()
            seen["stop"] = home_after_cancel()
            node._stop_run_requested.clear()
            node._estop_latched = True
            seen["estop"] = home_after_cancel()
            node._estop_latched = False
            raise SM.SortCanceled()
        backend.plan = plan
        out = node._execute(handle)
        assert (out.status, handle.state) == ("CANCELED", "canceled")
        assert seen == {"ab": True, "stop": False, "estop": False}
    finally:
        node.destroy_node()


def test_holding_after_cancel_latches_until_recover(ros, tmp_path):
    """Canceled with a part held (after the home move): e-stop latch and pause, so goals are rejected and
    C's ~/recover is the way out (A decision 2)."""
    backend = FakeBackend()
    node = managed_receiver(backend, tmp_path)
    try:
        handle = FakeHandle(managed_request(node))

        def plan(cancel, home_after_cancel):
            handle.is_cancel_requested = True
            backend.holding_object = True
            raise SM.SortCanceled()
        backend.plan = plan
        node._execute(handle)
        assert node._estop_latched and node._run_paused
        from rclpy.action import GoalResponse
        assert node._goal_request(managed_request(node)) == GoalResponse.REJECT
    finally:
        node.destroy_node()


def test_restart_holding_latches(ros):
    """Start home check with the gripper closed (part held after a restart): moved home, then latched."""
    backend = FakeBackend(at_home=False)
    backend.holding_object = True
    node = receiver(backend)
    try:
        node._start_check()
        assert backend.calls == ["at_home", "move_home", "refresh_hold"] and node._estop_latched
    finally:
        node.destroy_node()


def test_holding_after_stop_run_home_latches(ros, tmp_path):
    """C's stop_run homes with the gripper untouched; a held part then latches (no data discard yet:
    C's ~/recover does it) instead of being dropped by the next goal's opening."""
    from std_srvs.srv import Trigger
    backend = FakeBackend()
    backend.holding_object = True
    node = managed_receiver(backend, tmp_path)
    try:
        reply = node._stop_run(None, Trigger.Response())
        assert (reply.success, reply.message) == (False, "HOLDING_OBJECT_AT_HOME: E_STOP_LATCHED_USE_RECOVER")
        assert node._estop_latched and node._run_paused and backend.calls[-1] == "move_home"
    finally:
        node.destroy_node()


# ---------------------------------------------------------------- piece ④: shutdown order, records

import threading  # noqa: E402
import time  # noqa: E402


class Records:
    """SortEventLog stand-in."""

    def __init__(self):
        self.rows = []

    def record(self, event, **fields):
        self.rows.append({"event": event, **fields})


class FailingHandle(FakeHandle):
    """Goal handle whose terminal sends raise like a closed ROS context."""

    def _fail(self):
        raise RuntimeError("Failed get goal status array: feedback publisher is invalid")

    succeed = canceled = abort = _fail


def recording_receiver(backend, tmp_path):
    """Managed receiver with a recording event log."""
    from std_srvs.srv import Trigger
    events = Records()
    node = SM.SortGoalReceiver(f"/sort_part_test_{os.getpid()}_{uuid.uuid4().hex[:8]}", backend,
                               tmp_path / "run.sqlite", None, events)
    node._start_timer.cancel()
    node._home_checked = True
    node._start_run(None, Trigger.Response())
    return node, events


def results(events):
    """Recorded goal results (status, reason, result_sent)."""
    return [(r["status"], r["reason"], r.get("result_sent")) for r in events.rows if r["event"] == "goal_result"]


def test_shutdown_lets_the_goal_stop_and_send_then_stops(ros, tmp_path):
    """First signal: new goals rejected, the running goal sees C's pause flag, its result is recorded and
    sent before the executor stops (after the flush)."""
    backend = FakeBackend()
    node, events = recording_receiver(backend, tmp_path)
    try:
        handle = FakeHandle(managed_request(node))

        def plan(cancel, home_after_cancel):
            while not cancel():
                time.sleep(.01)
            assert not home_after_cancel()       # shutdown: stop in place, no home move
            raise SM.SortCanceled()
        backend.plan = plan
        worker = threading.Thread(target=node._execute, args=(handle,))
        worker.start()
        time.sleep(.1)
        stopped = threading.Event()
        node.begin_shutdown(SimpleNamespace(shutdown=stopped.set), wait_s=5., flush_s=.2)
        worker.join(5.)
        assert stopped.wait(5.)
        assert handle.state == "abort" and results(events)[-1] == ("CANCELED", "SHUTDOWN", True)
        from rclpy.action import GoalResponse
        assert node._goal_request(managed_request(node)) == GoalResponse.REJECT
    finally:
        node.destroy_node()


def test_send_failure_and_cancel_paths_are_recorded(ros, tmp_path):
    """A failing result send is recorded (result_sent false) instead of raising; the cancel path and the
    STOP_UNCONFIRMED-after-cancel path, which had no record, are recorded."""
    backend = FakeBackend()
    node, events = recording_receiver(backend, tmp_path)
    try:
        backend.plan = lambda cancel, home: 7
        assert node._execute(FailingHandle(managed_request(node))).status == "SUCCEEDED"
        handle = FakeHandle(managed_request(node))

        def canceled(cancel, home):
            handle.is_cancel_requested = True
            raise SM.SortCanceled()
        backend.plan = canceled
        node._execute(handle)
        handle = FakeHandle(managed_request(node))

        def unconfirmed(cancel, home):
            handle.is_cancel_requested = True
            raise SM.SortFailure("STOP_UNCONFIRMED")
        backend.plan = unconfirmed
        node._execute(handle)
        assert results(events) == [("SUCCEEDED", "", False), ("CANCELED", "ROBOT_STOP_CONFIRMED", True),
                                   ("FAILED", "STOP_UNCONFIRMED", True)]
    finally:
        node.destroy_node()


def test_stop_check_clock_is_the_samples_not_the_node(monkeypatch):
    """#394 H-ES: after an idle period the node clock is stale and jumps on the first spin; the stop check
    must not end on that jump. The limit runs on the joint samples' stamps, so a still arm is confirmed."""
    from types import SimpleNamespace as N
    ex, world = base.make(NEW, monkeypatch=monkeypatch)
    del ex._confirm_stopped                       # use the real method (make() stubs it)
    clock = {"ns": 1 * 1_000_000_000}
    ex._now_ns = lambda: clock["ns"]
    stamp = {"ns": 100 * 1_000_000_000}

    def spin_once(node, timeout_sec=0):
        clock["ns"] += 60 * 1_000_000_000        # the stale node clock jumps far ahead
        stamp["ns"] += 20_000_000                 # joint_states every 20 ms of SIM time
        ex._observe_velocity(N(header=N(stamp=N(sec=stamp["ns"] // 1_000_000_000,
                                                 nanosec=stamp["ns"] % 1_000_000_000)),
                               name=list(ex.demo.JOINTS), position=[0.] * len(ex.demo.JOINTS)))
    monkeypatch.setattr(NEW, "rclpy", N(spin_once=spin_once))
    ex._confirm_stopped()                         # does not raise STOP_UNCONFIRMED
