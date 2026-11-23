"""Track stream tracker and executor pose source tests (no Gazebo, no controllers).

Tracker tests use fake TrackStateArray objects and need no ROS. Executor tests need
rclpy and the robot model (xacro) and run the real kinematics against a fake arm,
fake clock and fake world; they are skipped without ROS.
"""
from __future__ import annotations

import importlib.util
import io
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[2]
sys.path.insert(0, str(PACKAGE))

from robot_server_package.track_stream_tracker import (  # noqa: E402
    SimExecutionFeedback, SimTruthLog, TrackFailure, TrackStreamTracker, yaw_quaternion)

SEC = 1_000_000_000


# ---------------------------------------------------------------- fake messages

def track(tid="7", cls="BOLT_B", pos=(550., 100., 100.), vel=(0., 0., 0.), yaw=0., state="VISIBLE",
          camera="link", source="TRACK"):
    """Fake TrackState (mm, deg)."""
    return SimpleNamespace(part_track_id=tid, part_class=cls, state=state, position_mm=list(pos),
                           velocity_mm_per_s=list(vel), velocity_source=source, axis_yaw_deg=yaw,
                           length_mm=40., last_hit_camera=camera, confirmed=True)


def frame(stamp_ns, tracks, *, kind="frame", run="run-A", epoch=1, belt=0., handoff=True):
    """Fake TrackStateArray."""
    return SimpleNamespace(kind=kind, run_id=run, tracker_epoch=epoch, capture_stamp_ns=stamp_ns,
                           belt_velocity_mm_per_s=belt, handoff_on=handoff, tracks=list(tracks))


class Clock:
    """Settable SIM clock (ns)."""

    def __init__(self, ns=10 * SEC):
        self.ns = ns

    def __call__(self):
        return self.ns


def tracker(clock, tid="7", cls="BOLT_B"):
    return TrackStreamTracker(tid, cls, clock)


def reason(call):
    """Failure reason of a call."""
    with pytest.raises(TrackFailure) as err:
        call()
    return str(err.value)


# ---------------------------------------------------------------- tracker

def test_selects_by_track_id_among_same_class():
    """Two BOLT_B tracks: the goal's part_track_id decides, not the class."""
    clock = Clock()
    t = tracker(clock, "8")
    t.on_message(frame(clock.ns, [track("7", pos=(500., 0., 100.)), track("8", pos=(600., 50., 100.))]))
    assert np.allclose(t.pose()[0], [.6, .05, .1])


@pytest.mark.parametrize("tid", ["", None])
def test_missing_track_id(tid):
    """Empty part_track_id is TRACK_ID_MISSING."""
    assert reason(lambda: TrackStreamTracker(tid, "BOLT_B", Clock())) == "TRACK_ID_MISSING"


def test_not_found_and_lost():
    """Never seen is TRACK_NOT_FOUND; seen then missing from a frame is TRACK_LOST."""
    clock = Clock()
    t = tracker(clock)
    t.on_message(frame(clock.ns, [track("9")]))
    assert reason(t.pose) == "TRACK_NOT_FOUND"
    t.on_message(frame(clock.ns + 1, [track("7")]))
    t.on_message(frame(clock.ns + 2, [track("9")]))
    assert reason(t.pose) == "TRACK_LOST"


def test_class_mismatch_and_unknown_class():
    """A known different class fails; an empty class (A unknown) is accepted."""
    clock = Clock()
    t = tracker(clock)
    t.on_message(frame(clock.ns, [track(cls="BOLT_A")]))
    assert reason(t.pose) == "TRACK_CLASS_MISMATCH"
    u = tracker(clock)
    u.on_message(frame(clock.ns, [track(cls="")]))
    assert np.allclose(u.pose()[0], [.55, .1, .1])


@pytest.mark.parametrize("message", [
    frame(0, [], kind="reset"),
    frame(0, [], kind="marker"),
    frame(10 * SEC + 1, [track()], epoch=2),
    frame(10 * SEC + 1, [track()], run="run-B"),
])
def test_reset_marker_epoch_run_change_lose_track(message):
    """reset, run end, tracker_epoch change or run change after use: TRACK_LOST."""
    clock = Clock()
    t = tracker(clock)
    t.on_message(frame(clock.ns, [track()]))
    t.on_message(message)
    assert reason(t.pose) == "TRACK_LOST"


def test_prediction_to_now_and_velocity_none():
    """Position is predicted to now from the track velocity; velocity_source NONE predicts nothing."""
    clock = Clock()
    t = tracker(clock)
    t.on_message(frame(clock.ns, [track(vel=(0., 20., 0.))]))
    clock.ns += SEC // 4
    xyz, _, now = t.pose()
    assert now == clock.ns and np.allclose(xyz, [.55, .105, .1])
    assert np.allclose(t.recent_motion()[1], [0., .02])
    u = tracker(clock)
    u.on_message(frame(clock.ns, [track(vel=(0., 20., 0.), source="NONE")]))
    clock.ns += SEC // 4
    assert np.allclose(u.velocity(), 0.) and np.allclose(u.pose()[0], [.55, .1, .1])


def test_stale():
    """Capture older than 500 ms is TRACK_STALE (500 ms itself is not)."""
    clock = Clock()
    t = tracker(clock)
    t.on_message(frame(clock.ns, [track()]))
    clock.ns += SEC // 2
    t.pose()
    clock.ns += 1
    assert reason(t.pose) == "TRACK_STALE"


def test_jump():
    """New position more than 25 mm from the previous prediction is TRACK_JUMP."""
    clock = Clock()
    t = tracker(clock)
    t.on_message(frame(clock.ns, [track(pos=(550., 100., 100.))]))
    t.on_message(frame(clock.ns + 1, [track(pos=(550., 124.9, 100.))]))
    t.pose()
    t.on_message(frame(clock.ns + 2, [track(pos=(550., 150.1, 100.))]))
    assert reason(t.pose) == "TRACK_JUMP"


def test_old_frame_ignored():
    """A frame not newer than the last one is ignored."""
    clock = Clock()
    t = tracker(clock)
    t.on_message(frame(clock.ns, [track(pos=(550., 100., 100.))]))
    t.on_message(frame(clock.ns - 1, [track(pos=(550., 110., 100.))]))
    assert np.allclose(t.pose()[0], [.55, .1, .1])


def test_invalid_values():
    """Non-finite position is TRACK_INVALID."""
    clock = Clock()
    t = tracker(clock)
    t.on_message(frame(clock.ns, [track(pos=(float("nan"), 0., 100.))]))
    assert reason(t.pose) == "TRACK_INVALID"


def test_unknown_axis():
    """NaN yaw (A does not know the axis): position still usable, axis from the last known yaw;
    with none, a static pose has no orientation and the moving calls fail TRACK_AXIS_UNKNOWN."""
    clock = Clock()
    t = tracker(clock)
    t.on_message(frame(clock.ns, [track(yaw=float("nan"))]))
    xyz, q, _ = t.pose()
    assert q is None and np.allclose(xyz, [.55, .1, .1])
    assert reason(lambda: t.pose(require_axis=True)) == "TRACK_AXIS_UNKNOWN"
    assert reason(t.recent_motion) == "TRACK_AXIS_UNKNOWN"
    t.on_message(frame(clock.ns + 1, [track(yaw=30.)]))
    t.on_message(frame(clock.ns + 2, [track(yaw=float("nan"))]))
    assert t.recent_motion()[0][1] == yaw_quaternion(math.radians(30.))


@pytest.mark.parametrize("dy, ok", [(24.9, True), (25.1, False)])
def test_observation_boundary(dy, ok):
    """Goal observation vs track predicted to the goal capture time: 25 mm limit."""
    clock = Clock()
    t = tracker(clock)
    t.on_message(frame(clock.ns, [track(pos=(550., 100., 100.), vel=(0., 20., 0.))]))
    capture = clock.ns - SEC   # 20 mm earlier along +y
    observed = [.55, .080 + dy / 1000., .1]
    if ok:
        t.check_observation(observed, capture)
    else:
        assert reason(lambda: t.check_observation(observed, capture)) == "PART_NOT_AT_OBSERVATION"


def test_quaternion_axis_and_fold():
    """Quaternion z axis is the bolt axis; a 180 degree flip is folded back to the first yaw."""
    def axis(q):
        w, x, y, z = q
        return np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])

    for deg in (-89., 0., 30., 89.):
        a = axis(yaw_quaternion(math.radians(deg)))
        assert np.allclose(a, [math.cos(math.radians(deg)), math.sin(math.radians(deg)), 0.], atol=1e-9)
    clock = Clock()
    t = tracker(clock)
    t.on_message(frame(clock.ns, [track(yaw=-89.)]))
    first = axis(t.pose()[1])
    t.on_message(frame(clock.ns + 1, [track(yaw=89.)]))
    second = axis(t.pose()[1])
    assert np.dot(first, second) > .99


def test_stationary_uses_measured_track_only():
    """Track mode stationary check: 3 mm on the measured position; A belt speed noise is ignored."""
    clock = Clock()
    t = tracker(clock)
    t.on_message(frame(clock.ns, [track(pos=(550., 100., 100.))]))
    still = t.measured_position()
    t.on_message(frame(clock.ns + 1, [track(pos=(550., 102.9, 100.))]))
    t.check_stationary(still)
    t.on_message(frame(clock.ns + 2, [track(pos=(550., 103.1, 100.))]))
    assert reason(lambda: t.check_stationary(still)) == "PART_MOVED"
    u = tracker(clock)
    u.on_message(frame(clock.ns, [track()], belt=7.9))   # p99 of A belt speed while stopped (#314)
    u.check_stationary(u.measured_position())


def test_prefer_gripper_sample():
    """At grasp height a fresh gripper sample is used before a newer link sample."""
    clock = Clock()
    t = tracker(clock)
    t.on_message(frame(clock.ns, [track(pos=(550., 100., 100.), state="GRIPPER", camera="gripper")]))
    t.on_message(frame(clock.ns + 1, [track(pos=(550., 101., 100.))]))
    sample, _ = t.recent_motion(prefer_gripper=True)
    assert (t.grasp_source, round(sample[0][1], 4)) == ("gripper", .1)
    clock.ns += SEC
    t.on_message(frame(clock.ns, [track(pos=(550., 101., 100.))]))
    t.recent_motion(prefer_gripper=True)
    assert t.grasp_source == "link_pred"


def test_handoff_flag():
    """handoff_on follows the latest frame."""
    clock = Clock()
    t = tracker(clock)
    assert t.handoff_on() is None
    t.on_message(frame(clock.ns, [track()], handoff=False))
    assert t.handoff_on() is False


def test_execution_feedback_nearest_part():
    """Feedback binds the simulation part nearest the fingers; none within 30 mm is MISSED."""
    poses = {"part_00": (np.array([.5, 0., .1]), (1., 0., 0., 0.)),
             "part_01": (np.array([.55, .1, .1]), (1., 0., 0., 0.))}
    f = SimExecutionFeedback(lambda: poses)
    assert f.bind([.551, .101, .1])[0][1] == .1 and f.part == "part_01"
    g = SimExecutionFeedback(lambda: poses)
    assert reason(lambda: g.bind([.9, .9, .1])) == "MISSED"


def test_truth_log_never_raises_or_returns():
    """The evaluation log returns nothing and swallows bad truth input."""
    out = io.StringIO()
    log = SimTruthLog(lambda: {"part_01": (np.array([.56, .1, .1]), (1., 0., 0., 0.))}, out)
    assert log.record("planning", "g", "7", np.array([.55, .1, .1]), 1) is None
    assert json.loads(out.getvalue())["error_xy_mm"] == 10.0
    bad = SimTruthLog(lambda: {"part_01": "garbage"}, out)
    assert bad.record("planning", "g", "7", np.array([.55, .1, .1]), 1) is None


def test_tracker_holds_no_simulation_reference():
    """The track tracker has no simulation pose source among its attributes."""
    t = tracker(Clock())
    assert not any(isinstance(v, (SimExecutionFeedback, SimTruthLog)) or callable(v) and v.__class__.__name__ == "SimPoses"
                   for v in vars(t).values())


# ---------------------------------------------------------------- executor (ROS)

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("action_msgs.msg")
try:
    subprocess.run(["xacro", "--help"], capture_output=True, check=False)
except FileNotFoundError:
    pytest.skip("xacro not available", allow_module_level=True)

_SEED = None


# Baseline executor before the pose source change (commit 5099e65), pinned as a test fixture.
BASELINE = Path(__file__).resolve().parent / "fixtures" / "gazebo_sort_executor_5099e65.py"
BASELINE_SHA256 = "7bf00cf602d95a5f5362b592aedf1f3915ee90b8e33613b3ad1042b9a008576f"


def load_executor_modules():
    """(current module, pinned baseline module). The baseline sha256 is checked so it cannot drift."""
    import hashlib

    import robot_server_package.gazebo_sort_executor as new

    digest = hashlib.sha256(BASELINE.read_bytes()).hexdigest()
    if digest != BASELINE_SHA256:
        raise RuntimeError(f"baseline executor fixture changed: sha256 {digest}")
    spec = importlib.util.spec_from_file_location("baseline_gazebo_sort_executor", BASELINE)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    return new, old


NEW, OLD = load_executor_modules()
AXIS_X = (math.cos(math.pi / 4), 0., math.sin(math.pi / 4), 0.)   # part z axis along +x


class World:
    """Fake sort cell: one BOLT_B on the belt, a fake arm, SIM clock and the A track stream."""

    def __init__(self, demo, chain, *, belt_y_mps=0., truth_shift=(0., 0.), track_error_m=(0., 0.)):
        self.demo, self.chain = demo, chain
        self.ns = 20 * SEC
        self.node_ns = self.ns          # executor node clock: moves only when the node is spun
        self.t0 = self.ns
        self.base = np.array([.55, .10, .10])
        self.belt = np.array([0., belt_y_mps, 0.])
        self.truth_shift = np.r_[truth_shift, 0.]
        self.track_error = np.r_[track_error_m, 0.]
        self.attached = None
        self.dropped = None
        self.executor = None
        self.no_attach = False          # closing grabs nothing (MISSED after the lift)
        self.offset = demo.FINGER_REACH + demo.TIP_CLEARANCE - .003

    def part(self):
        """True part position."""
        if self.dropped is not None:
            return self.dropped
        if self.attached is not None:
            return self.demo.forward(self.chain, self.arm.q)[:3, 3] - [0., 0., self.offset] + self.attached
        return self.base + self.belt * ((self.ns - self.t0) / 1e9)

    def sim_poses(self):
        return {"part_01": (self.part() + self.truth_shift, AXIS_X),
                "part_00": (np.array([.40, -.30, .10]), AXIS_X)}

    def idle(self, seconds):
        """Time passes without spinning the executor node: its clock and track messages fall behind."""
        self.ns += int(seconds * 1e9)

    def advance(self, seconds):
        self.ns += int(seconds * 1e9)
        self.node_ns = self.ns
        if self.executor is not None and getattr(self.executor, "pose_source", "sim") == "track":
            # The error starts after planning: a constant offset cancels in the relative moving correction
            p = self.part() + (self.track_error if self.ns - self.t0 > SEC else 0.)
            self.executor._on_track_message(frame(self.ns, [track(pos=tuple(p * 1000.), vel=tuple(self.belt * 1000.))],
                                                  belt=self.belt[1] * 1000.))

    def attach(self):
        if self.no_attach:
            return
        tool = self.demo.forward(self.chain, self.arm.q)[:3, 3] - [0., 0., self.offset]
        if np.linalg.norm(self.part()[:2] - tool[:2]) < .01:
            self.attached = self.part() - tool

    def release(self):
        if self.attached is not None:
            p = self.part()
            self.attached = None
            self.dropped = np.r_[p[:2], .03]


class FakeArm:
    def __init__(self, world, demo, q):
        self.world, self.demo, self.q = world, demo, np.asarray(q, dtype=float)
        self.state = {demo.GRIPPER_JOINT: demo.GRIP_OPEN}
        self.log = []
        self.fail = None                # fail(path, follow count) -> exception to raise, or None
        world.arm = self

    def joints(self):
        return self.q.copy()

    def follow(self, path, seconds):
        error = None if self.fail is None else self.fail(path, sum(1 for e in self.log if e[0] == "follow"))
        if error is not None:
            raise error
        self.q = np.asarray(path[-1], dtype=float)
        self.log.append(("follow", len(path), seconds, tuple(np.round(self.q, 5))))
        self.world.advance(seconds)
        return 0.

    def grip(self, position, settle=.5):
        self.log.append(("grip", position))
        self.state[self.demo.GRIPPER_JOINT] = position
        if position == self.demo.GRIP_CLOSED:
            self.world.attach()
        elif position == self.demo.GRIP_OPEN:
            self.world.release()
        self.wait(settle)

    def wait(self, seconds):
        self.world.advance(seconds)


class FakeNode:
    def __init__(self, world):
        self.world = world
        self.logs = []

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=getattr(self.world, "node_ns", self.world.ns)))

    def get_logger(self):
        return SimpleNamespace(info=self.logs.append, error=self.logs.append, warning=self.logs.append)

    def create_subscription(self, *args):
        return None


class SimTracker:
    """Simulation pose tracker stand-in (sim mode only)."""

    def __init__(self, world):
        self.world = world

    def pose(self):
        return self.world.part(), AXIS_X, self.world.ns

    def recent_motion(self, prefer_gripper=False):
        return self.pose(), self.world.belt[:2]

    def close(self):
        pass


def goal(world, **over):
    from types import SimpleNamespace as N
    down = (1., 0., 0., 0.)
    g = N(goal_id="g-1", goal_kind="PICK_PLACE", part_class="BOLT_B", place_bin="BOLT_B", frame_id="base_link",
          timestamp_base="SIM_TIME", part_track_id="7",
          observed_position_mm=list(world.part() * 1000.), place_position_mm=[720., 170., 150.],
          place_orientation_xyzw=down, approach_height_mm=200., lift_height_mm=250., opening_width_mm=20.,
          capture_stamp_ns=world.ns, valid_until_ns=world.ns + 600 * SEC,
          grasp_pose_candidates=[N(position_mm=list(world.part() * 1000.), orientation_xyzw=down)])
    vars(g).update(over)
    return g


def make(module, *, moving=False, track=False, world_kw=None, truth_out=None, monkeypatch=None):
    """Executor of `module` in a fake world. Track mode is set after construction (no msg import, no gz)."""
    global _SEED
    world_kw = world_kw or {}
    root = ROOT
    dummy = World.__new__(World)
    dummy.ns = 0
    node = FakeNode(dummy)
    ex = module.GazeboSortExecutor(node, root, moving=moving)
    world = World(ex.demo, ex.chain, **world_kw)
    node.world = world
    if _SEED is None:
        R = ex.demo.quaternion_matrix(0, 1, 0, 0)
        _SEED = ex.demo.solve_ik(ex.chain, ex.demo.transform(R, [.55, .1, .3]), [0, -1.57, 1.57, -1.57, -1.57, 0])
    ex.arm = FakeArm(world, ex.demo, _SEED)
    ex._start_grip = lambda: (ex.arm.log.append(("start_grip",)), world.attach())[0]
    ex._confirm_stopped = lambda: None
    world.executor = ex
    if track:
        ex.pose_source = "track"
        ex.sim_poses = world.sim_poses
        ex._track = None
        ex._last_track_message = None
        ex.truth_log = None if truth_out is None else module.SimTruthLog(world.sim_poses, truth_out)
    def spin_once(node, timeout_sec=0):
        # The executor waits on real time; keep SIM and real time in step (50 ms each).
        time.sleep(.05)
        world.advance(.05)

    fake_rclpy = SimpleNamespace(spin_once=spin_once)
    monkeypatch.setattr(module, "rclpy", fake_rclpy)
    return ex, world


def run_pick(ex, world, g):
    """Run one goal; (result or reason, arm log)."""
    radius = .003
    place = np.asarray(g.place_position_mm) / 1000.
    heights = np.array([g.approach_height_mm, g.lift_height_mm]) / 1000.
    try:
        if getattr(ex, "pose_source", "sim") == "track":
            world.advance(.05)
            out = ex.execute(g, lambda stage: None, lambda: False)
        else:
            out = ex._pick_and_place(g, lambda stage: None, SimTracker(world), np.asarray(g.observed_position_mm) / 1000.,
                                     place, ex._rotation(g.place_orientation_xyzw), heights, radius, (0.72, 0.17))
        return ("ok", out - world.t0), ex.arm.log
    except Exception as exc:  # the reason is the result under test
        return (type(exc).__name__, str(exc)), ex.arm.log


def home_entry(ex):
    """Arm log entry of the home return move."""
    return ("follow", 1, 3., tuple(np.round(ex.home, 5)))


def tool_z(ex, entry):
    """Tool height after a logged follow."""
    return ex.demo.forward(ex.chain, np.asarray(entry[3]))[2, 3]


@pytest.mark.parametrize("moving", [False, True])
def test_sim_mode_same_as_baseline(monkeypatch, moving):
    """sim mode: same result and arm commands as the pinned 5099e65 executor in the same fake world,
    except the home return added after placing (one final move to the saved home joints)."""
    world_kw = {"belt_y_mps": .01} if moving else {}
    results = []
    for module in (NEW, OLD):
        ex, world = make(module, moving=moving, world_kw=world_kw, monkeypatch=monkeypatch)
        results.append(run_pick(ex, world, goal(world)))
        if module is NEW:
            new_ex = ex
    (new_result, new_log), (old_result, old_log) = results
    assert new_result == old_result and new_result[0] == "ok"
    assert new_log[:-1] == old_log
    assert new_log[-1] == home_entry(new_ex)


@pytest.mark.parametrize("over", [{"observed_position_mm": [550., 140., 100.]}, {"grasp_pose_candidates": []}])
def test_sim_mode_failure_same_as_baseline(monkeypatch, over):
    """sim mode failures before any arm motion (observation 40 mm away, no usable candidate): same
    reason and arm commands as the pinned 5099e65 executor; no home return because the arm never moved."""
    results = []
    for module in (NEW, OLD):
        ex, world = make(module, monkeypatch=monkeypatch)
        results.append(run_pick(ex, world, goal(world, **over)))
    assert results[0] == results[1]


@pytest.mark.parametrize("moving", [False, True])
def test_track_mode_succeeds_and_truth_does_not_steer(monkeypatch, moving):
    """track mode picks from the A track. Shifting the simulation truth by 15 mm changes the
    truth log but not a single arm command: truth only feeds the result check and the log."""
    runs = []
    world_kw = {"belt_y_mps": .01} if moving else {}
    for shift in ((0., 0.), (.015, -.015)):
        out = io.StringIO()
        ex, world = make(NEW, moving=moving, track=True, truth_out=out,
                         world_kw={**world_kw, "truth_shift": shift}, monkeypatch=monkeypatch)
        runs.append((run_pick(ex, world, goal(world)), out.getvalue()))
    (first, log_a), (second, log_b) = runs
    assert first[0][0] == "ok", first
    assert first == second
    assert log_a != log_b and log_a


def test_track_mode_follows_track_not_truth(monkeypatch):
    """Moving pick (the track steers the arm; a static pick uses B's point and the track only checks it):
    a 2 mm A track change after planning moves the arm path, a 2 mm truth shift does not."""
    logs = {}
    for name, kw in (("base", {}), ("track_err", {"track_error_m": (0., .002)}), ("truth", {"truth_shift": (0., .002)})):
        ex, world = make(NEW, moving=True, track=True, world_kw={"belt_y_mps": .01, **kw}, monkeypatch=monkeypatch)
        logs[name] = run_pick(ex, world, goal(world))   # goal stays at the true position
    assert logs["truth"] == logs["base"]
    assert logs["track_err"][1] != logs["base"][1]


def test_track_mode_moving_requires_handoff(monkeypatch):
    """Moving pick in track mode needs handoff_on: TRACK_HANDOFF_OFF before any arm motion."""
    ex, world = make(NEW, moving=True, track=True, world_kw={"belt_y_mps": .01}, monkeypatch=monkeypatch)
    real = world.advance

    def advance(seconds):
        world.ns += int(seconds * 1e9)
        world.node_ns = world.ns
        p = world.part()
        ex._on_track_message(frame(world.ns, [track(pos=tuple(p * 1000.), vel=(0., 10., 0.))], belt=10., handoff=False))

    world.advance = advance
    result, log = run_pick(ex, world, goal(world))
    world.advance = real
    assert result == ("SortFailure", "TRACK_HANDOFF_OFF")
    assert not any(entry[0] == "follow" for entry in log)


def test_track_mode_reasons(monkeypatch):
    """Track mode: missing id, wrong id and observation mismatch reasons."""
    cases = [({"part_track_id": ""}, "TRACK_ID_MISSING"), ({"part_track_id": "99"}, "TRACK_NOT_FOUND"),
             ({"observed_position_mm": [550., 140., 100.]}, "PART_NOT_AT_OBSERVATION")]
    for over, expected in cases:
        ex, world = make(NEW, track=True, monkeypatch=monkeypatch)
        assert run_pick(ex, world, goal(world, **over))[0] == ("SortFailure", expected)


def test_track_mode_missed_when_no_part_near_fingers(monkeypatch):
    """Execution feedback: no simulation part within 30 mm of the closed fingers is MISSED."""
    ex, world = make(NEW, track=True, world_kw={"truth_shift": (.05, 0.)}, monkeypatch=monkeypatch)
    assert run_pick(ex, world, goal(world))[0] == ("SortFailure", "MISSED")


# ---------------------------------------------------------------- home return

@pytest.mark.parametrize("moving", [False, True])
def test_failure_after_motion_returns_home(monkeypatch, moving):
    """MISSED after the arm moved: same commands as the 5099e65 executor up to the failure, then home.
    Static MISSED happens at lift height (above the approach height, so no retreat); moving MISSED
    happens after the probe, so the arm first rises to the approach height."""
    world_kw = {"belt_y_mps": .01} if moving else {}
    results = []
    for module in (NEW, OLD):
        ex, world = make(module, moving=moving, world_kw=world_kw, monkeypatch=monkeypatch)
        world.no_attach = True
        results.append((ex, run_pick(ex, world, goal(world))))
    (new_ex, (new_result, new_log)), (_, (old_result, old_log)) = results
    assert new_result == old_result == ("SortFailure", "MISSED")
    assert new_log[:len(old_log)] == old_log
    extra = new_log[len(old_log):]
    assert extra[-1] == home_entry(new_ex)
    if moving:
        assert [e[:3] for e in extra] == [("follow", 20, 1.5), ("follow", 1, 3.)]
        assert tool_z(new_ex, extra[0]) == pytest.approx(new_ex._retreat_z, abs=1e-3)
    else:
        assert extra == [home_entry(new_ex)]


def test_track_failure_during_moving_returns_home(monkeypatch):
    """track mode: a reset in the middle of the moving grasp is TRACK_LOST, and the arm returns home."""
    ex, world = make(NEW, moving=True, track=True, world_kw={"belt_y_mps": .01}, monkeypatch=monkeypatch)

    def fail(path, count):
        if count == 3:
            ex._on_track_message(frame(0, [], kind="reset", epoch=2))
        return None

    ex.arm.fail = fail
    result, log = run_pick(ex, world, goal(world))
    assert result == ("SortFailure", "TRACK_LOST")
    assert log[-1] == home_entry(ex)
    assert any("HOME_RETURN done" in str(line) for line in ex.node.logs)


def test_cancel_and_unconfirmed_stop_do_not_return_home(monkeypatch):
    """A cancel (SortCanceled) and STOP_UNCONFIRMED keep the current stop rule: no automatic home move."""
    ex, world = make(NEW, moving=True, world_kw={"belt_y_mps": .01}, monkeypatch=monkeypatch)
    ex.cancel_requested = lambda: sum(1 for e in ex.arm.log if e[0] == "follow") >= 3
    result, log = run_pick(ex, world, goal(world))
    assert result[0] == "SortCanceled" and log[-1] != home_entry(ex)

    ex, world = make(NEW, moving=True, world_kw={"belt_y_mps": .01}, monkeypatch=monkeypatch)
    ex.arm.fail = lambda path, count: NEW.SortFailure("STOP_UNCONFIRMED") if count == 3 else None
    result, log = run_pick(ex, world, goal(world))
    assert result == ("SortFailure", "STOP_UNCONFIRMED") and log[-1] != home_entry(ex)
    assert not any("HOME_RETURN" in str(line) for line in ex.node.logs)


@pytest.mark.parametrize("missed", [False, True])
def test_home_return_failure_keeps_result(monkeypatch, missed):
    """A failing home move only logs a warning: success stays success, MISSED stays MISSED."""
    ex, world = make(NEW, monkeypatch=monkeypatch)
    world.no_attach = missed
    home = None

    def fail(path, count):
        return NEW.SortFailure("ARM_TRACKING_ERROR") if home is not None and np.allclose(path[-1], home) else None

    ex.arm.fail = fail
    home = np.asarray(ex.arm.q)
    result, _ = run_pick(ex, world, goal(world))
    assert result[0] == ("SortFailure" if missed else "ok")
    if missed:
        assert result[1] == "MISSED"
    assert any("HOME_RETURN failed" in str(line) for line in ex.node.logs)


# ---------------------------------------------------------------- node clock between goals

def test_track_clock_refreshed_before_deadline(monkeypatch):
    """The executor node is spun only while a goal runs. Between goals its SIM clock and the cached
    track message stop. A second goal for the same track, captured 5 s later, must refresh the clock
    (spin until a frame at or after its capture) instead of failing CAPTURE_IN_FUTURE on the stale clock."""
    ex, world = make(NEW, track=True, monkeypatch=monkeypatch)
    world.advance(.05)
    with pytest.raises(NEW.SortFailure) as err:   # first goal: fails at planning, leaves the cache with track 7
        ex.execute(goal(world, observed_position_mm=[550., 140., 100.]), lambda stage: None, lambda: False)
    assert str(err.value) == "PART_NOT_AT_OBSERVATION"
    world.idle(5.)
    assert world.ns - world.node_ns == 5 * SEC
    g = goal(world)
    result = ex.execute(g, lambda stage: None, lambda: False)
    assert result > g.capture_stamp_ns
    planning = [line for line in ex.node.logs if "SortPart track planning" in str(line)]
    assert planning and f"capture={g.capture_stamp_ns}" in str(planning[-1])


def test_track_no_new_frame_is_stale(monkeypatch):
    """If no frame at or after the goal capture arrives within 0.5 s, the stale cached sample is TRACK_STALE."""
    ex, world = make(NEW, track=True, monkeypatch=monkeypatch)
    world.advance(.05)
    world.idle(5.)
    g = goal(world)
    real = world.advance

    def advance(seconds):   # clock moves, but A publishes nothing
        world.ns += int(seconds * 1e9)
        world.node_ns = world.ns

    world.advance = advance
    with pytest.raises(NEW.SortFailure) as err:
        ex.execute(g, lambda stage: None, lambda: False)
    world.advance = real
    assert str(err.value) == "TRACK_STALE"


def test_frames_with_and_without_optional_fields():
    """Extra message fields (e.g. publish_stamp_ns) are ignored; the tracker reads only the fields it needs."""
    clock = Clock()
    t = tracker(clock)
    m = frame(clock.ns, [track()])
    m.publish_stamp_ns = clock.ns + 5
    m.some_future_field = "x"
    t.on_message(m)
    assert t.last_frame_ns == clock.ns and np.allclose(t.pose()[0], [.55, .1, .1])
