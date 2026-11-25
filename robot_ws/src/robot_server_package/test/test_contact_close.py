"""#380: gripper close on contact (ContactCloser and its use in the C executor)."""
import math

import numpy as np
import pytest

import test_track_stream_tracker as base
from robot_server_package import contact_closer as CC

NEW = base.NEW
MS = 1_000_000


def feed(closer, start_ms, values, step_ms=10):
    """Add values at step_ms SIM intervals; returns the stamp (ms) after the last one."""
    t = start_ms
    for q in values:
        closer.add(t * MS, q)
        t += step_ms
    return t


def test_limits_from_diameter():
    """cap/k_min from the #351 formulas: M3 0.756/0.70, M6 0.663/0.59."""
    assert CC.cap_for(3) == pytest.approx(.756) and CC.cap_for(6) == pytest.approx(.663)
    assert CC.k_min_for(3) == pytest.approx(.70, abs=1e-3) and CC.k_min_for(6) == pytest.approx(.593, abs=1e-3)


def test_m3_stall_recommands_contact_plus_delta():
    """M3 stops at 0.745 for 0.1 s: re-command 0.750 (stall = fingers on the bolt)."""
    c = CC.ContactCloser(3)
    feed(c, 0, np.linspace(.3, .745, 20))
    assert c.decide()[0] == "wait"
    feed(c, 200, [.745] * 11)
    assert c.decide() == ("recommand", pytest.approx(.750), "stall")
    assert c.contact == pytest.approx(.745)


def test_cap_reached_recommands_cap():
    """Closing through the cap without a stall (#351 (0.60, 0.05) 0°): re-command the cap."""
    c = CC.ContactCloser(3)
    feed(c, 0, [.70, .73, .757])
    assert c.decide() == ("recommand", pytest.approx(.756), "cap")


def test_stall_near_cap_is_capped():
    """A stall at 0.753 gives min(0.758, cap) = 0.756."""
    c = CC.ContactCloser(3)
    feed(c, 0, [.753] * 12)
    assert c.decide() == ("recommand", pytest.approx(.756), "stall")


def test_low_stall_short_is_ignored_long_is_blocked():
    """A stall below k_min is a pause for up to 1 s SIM, then the fingers are blocked (on the belt)."""
    c = CC.ContactCloser(3)
    t = feed(c, 0, [.634] * 60)                       # 0.6 s
    assert c.decide()[0] == "wait" and c.ignored_stalls == [.634]
    t = feed(c, t, [.634] * 45)                       # stalled for more than 1 s in all
    assert c.decide() == ("blocked", pytest.approx(.634), "blocked_low")


def test_low_stall_then_moving_resets_block_timer():
    """Closing resumes after a pause: the block timer starts again at the next pause."""
    c = CC.ContactCloser(3)
    t = feed(c, 0, [.634] * 80)
    t = feed(c, t, np.linspace(.64, .69, 10))
    assert c.decide()[0] == "wait"
    feed(c, t, [.69] * 50)
    assert c.decide()[0] == "wait"


def test_first_goal_done():
    """The first command ended: re-command from the joint if at or above k_min, else no contact."""
    c = CC.ContactCloser(6)
    feed(c, 0, [.60, .62, .64])
    assert c.decide(first_done=True) == ("recommand", pytest.approx(.645), "goal_done")
    c = CC.ContactCloser(6)
    feed(c, 0, [.40, .45, .50])
    assert c.decide(first_done=True) == ("no_contact", pytest.approx(.50), "goal_done")


def test_open_gripper_stall_is_not_a_pause():
    """A still open gripper (at most 0.05) is not counted as a low stall (no block)."""
    c = CC.ContactCloser(3)
    feed(c, 0, [0.] * 150)
    assert c.decide()[0] == "wait" and c.ignored_stalls == []


def test_timeout_runs_on_sample_stamps():
    """The 10 s limit is measured on the samples' SIM stamps, so it does not end early on a node clock
    jump (#394 H-ES) and ends after 10 s of samples."""
    c = CC.ContactCloser(3)
    t = feed(c, 0, np.linspace(.1, .4, 999), step_ms=10)   # still closing slowly
    assert c.decide()[0] == "wait"
    feed(c, 10_000, np.linspace(.40, .41, 3), step_ms=10)
    assert c.decide()[0] == "timeout"


def test_bad_samples_ignored():
    """Non-finite joints and non-increasing stamps are ignored."""
    c = CC.ContactCloser(3)
    c.add(10 * MS, .3)
    c.add(10 * MS, .9)
    c.add(5 * MS, .9)
    c.add(20 * MS, math.nan)
    c.add(30 * MS, None)
    assert c.latest() == .3


def test_stationary_close_recommands_on_contact(monkeypatch):
    """Stationary pick: start toward 0.8, the fingers stop on the bolt, re-command contact + 0.005;
    holding_object true without C's stalled flag."""
    ex, world = base.make(NEW, monkeypatch=monkeypatch)
    result, log = base.run_pick(ex, world, base.goal(world))
    assert result[0] == "ok"
    assert base.closes(log) == [("start_grip",), ("grip", pytest.approx(.650))]
    assert ex.grip_outcome["trigger"] == "stall" and ex.grip_outcome["target_rad"] == pytest.approx(.65)


def test_holding_from_contact_survives_exact_recommand(monkeypatch):
    """After a contact stall the re-command is reached exactly; holding_object is still true."""
    ex, world = base.make(NEW, monkeypatch=monkeypatch)
    ex._contact_seen = True
    ex.arm.state[ex.demo.GRIPPER_JOINT] = .65
    ex._grip(.65)
    assert ex.holding_object
    ex._grip(ex.demo.GRIP_OPEN)
    assert not ex.holding_object and not ex._contact_seen


@pytest.mark.parametrize("blocked_closes, expected", [(1, "ok"), (2, "GRIPPER_BLOCKED_LOW")])
def test_blocked_low_opens_rises_and_closes_once_more(monkeypatch, blocked_closes, expected):
    """Fingers stopped below k_min for over 1 s (on the belt): open, rise 1 mm, close once more
    (A decision 2). Blocked again: GRIPPER_BLOCKED_LOW, then home."""
    ex, world = base.make(NEW, monkeypatch=monkeypatch)
    starts = []
    original = ex._start_grip

    def start(position=None):
        starts.append(position)
        if len(starts) > blocked_closes:
            return original(position)
        ex.arm.log.append(("start_grip",))          # fingers stop on the belt, the bolt is not grabbed
        ex.arm.state[ex.demo.GRIPPER_JOINT] = .5
        return None
    ex._start_grip = start
    result, log = base.run_pick(ex, world, base.goal(world))
    rises = [e for e in log if e[0] == "follow" and e[2] == .3]
    assert len(rises) == 1
    if expected == "ok":
        assert result[0] == "ok"
        assert len(starts) == 2
    else:
        assert result == ("SortFailure", "GRIPPER_BLOCKED_LOW")
        assert len(starts) == 2 and log[-1] == base.home_entry(ex)
    opened_after_block = [i for i, e in enumerate(log) if e == ("grip", ex.demo.GRIP_OPEN)]
    assert opened_after_block and opened_after_block[0] < log.index(rises[0])


def test_grip_timeout_opens_and_fails(monkeypatch):
    """No decision within the close limit: GRIP_TIMEOUT after opening the gripper."""
    ex, world = base.make(NEW, monkeypatch=monkeypatch)
    monkeypatch.setattr(CC.ContactCloser, "decide", lambda self, first_done=False: ("timeout", .3, "timeout"))
    result, log = base.run_pick(ex, world, base.goal(world))
    assert result == ("SortFailure", "GRIP_TIMEOUT")
    first_start = log.index(("start_grip",))
    assert ("grip", ex.demo.GRIP_OPEN) in log[first_start:]


def test_moving_watch_recommands_without_waiting(monkeypatch):
    """Moving close: the watch run from the controller wait loop re-commands asynchronously on contact
    (start_grip with contact + 0.005), then stops watching."""
    ex, world = base.make(NEW, moving=True, monkeypatch=monkeypatch)
    sent = []
    ex._start_grip = lambda position=None: sent.append(position)
    ex._grip_watch = (ex._closer("BOLT_A"), None)
    ex.arm.state[ex.demo.GRIPPER_JOINT] = .745
    for _ in range(4):
        world.advance(.05)
        ex._watch_grip()
    assert sent == [pytest.approx(.750)]
    assert ex._grip_watch is None and ex._contact_seen
    assert ex.grip_outcome["trigger"] == "stall"


def test_moving_close_starts_toward_full_close(monkeypatch):
    """The moving path starts the close toward 0.8 for both bolts (f18cd7d's fixed 0.75 replaced)."""
    ex, world = base.make(NEW, moving=True, monkeypatch=monkeypatch)
    sent = []
    ex._start_grip = lambda position=None: sent.append(position)
    for part in ("BOLT_A", "BOLT_B"):
        ex._start_watched_close(part)
    assert sent == [CC.CLOSE_TARGET, CC.CLOSE_TARGET]
    assert ex._grip_watch is not None


def test_finish_grip_ends_the_watch(monkeypatch):
    """Finishing the grip (cancel, failure, end) stops the moving watch."""
    ex, _ = base.make(NEW, moving=True, monkeypatch=monkeypatch)
    ex._grip_watch = (ex._closer("BOLT_B"), None)
    ex._finish_grip()
    assert ex._grip_watch is None


# --- #380 (a)(b): goal expiry after a failed pick ---------------------------------------------------

def _closed_after_failure(monkeypatch, part_far=True, tracker=True):
    ex, world = base.make(NEW, monkeypatch=monkeypatch)
    ex.arm.state[ex.demo.GRIPPER_JOINT] = .62                   # pre-closed or half closed, as #378 found
    if part_far:
        world.base = np.array([.50, -.20, .10])
    else:
        tip = ex.demo.forward(ex.chain, ex.arm.q)[:3, 3] - [0., 0., ex.demo.FINGER_REACH]
        world.base = tip + [0., 0., .005]
    if tracker:
        ex._goal_tracker = base.SimTracker(world)
    return ex, world


def test_failure_home_opens_an_empty_closed_gripper(monkeypatch):
    """(a) A failure with the gripper closed and no part at the fingertips: open on the way home."""
    ex, world = _closed_after_failure(monkeypatch)
    ex._return_home(base.goal(world), "failure:TRACK_LOST", retreat=True)
    assert ("start_grip",) in ex.arm.log and ex.arm.state[ex.demo.GRIPPER_JOINT] == ex.demo.GRIP_OPEN
    assert ex.arm.log[-1] == base.home_entry(ex)
    assert any("opening empty gripper" in str(line) for line in ex.node.logs)


def test_failure_home_keeps_a_held_part(monkeypatch):
    """(a) A part at the fingertips: no automatic opening (#374/#394 rule kept)."""
    ex, world = _closed_after_failure(monkeypatch, part_far=False)
    ex._return_home(base.goal(world), "failure:ARM_TRACKING_ERROR", retreat=True)
    assert ("start_grip",) not in ex.arm.log and ex.arm.state[ex.demo.GRIPPER_JOINT] == .62


def test_failure_home_unknown_parts_keeps_closed(monkeypatch):
    """(a) No simulation part positions known: keep the gripper as it is."""
    ex, world = _closed_after_failure(monkeypatch, tracker=False)
    ex._return_home(base.goal(world), "failure:TRACK_LOST", retreat=True)
    assert ("start_grip",) not in ex.arm.log


def test_success_home_does_not_open(monkeypatch):
    """(a) Only failure returns open; the success return leaves the (already open) gripper alone."""
    ex, world = _closed_after_failure(monkeypatch)
    ex._return_home(base.goal(world), "success")
    assert ("start_grip",) not in ex.arm.log


def test_goal_start_opening_is_not_counted_against_the_deadline(monkeypatch):
    """(b) A gripper left closed: deadline check first, then open with a 0.1 s wait, then the deadline
    again with the opening time added (at most 2.5 s)."""
    ex, world = base.make(NEW, monkeypatch=monkeypatch)
    ex.arm.state[ex.demo.GRIPPER_JOINT] = .62
    calls = []
    deadline, grip = ex._deadline, ex._grip
    monkeypatch.setattr(ex, "_deadline", lambda g, grace_ns=0: (calls.append(("deadline", grace_ns)), deadline(g, grace_ns))[1])
    monkeypatch.setattr(ex, "_grip", lambda p, settle=.5: (calls.append(("grip", p, settle)), grip(p, settle))[1])
    result, _ = base.run_pick(ex, world, base.goal(world))
    assert result[0] == "ok"
    opened = calls.index(("grip", ex.demo.GRIP_OPEN, NEW.START_OPEN_SETTLE_S))
    assert calls[opened - 1] == ("deadline", 0)
    assert calls[opened + 1][0] == "deadline" and 0 < calls[opened + 1][1] <= NEW.OPEN_GRACE_MAX_NS


def test_deadline_grace(monkeypatch):
    """(b) The grace moves the expiry later by exactly that much."""
    ex, world = base.make(NEW, monkeypatch=monkeypatch)
    g = base.goal(world, valid_until_ns=world.node_ns - 100)
    with pytest.raises(NEW.SortFailure, match="GOAL_EXPIRED"):
        ex._deadline(g)
    ex._deadline(g, 200)
    with pytest.raises(NEW.SortFailure, match="GOAL_EXPIRED"):
        ex._deadline(g, 100)


# --- #380 A decision 1: grasp height offset kept in the executor ------------------------------------

def test_grasp_height_default():
    """Executor default offsets: M3 -1 mm (A decision 2026-11-25, was -2 mm), M6 0. The C demo keeps its own
    values (M3 -2 mm) and the executor does not read them; M6 still agrees with the demo."""
    import importlib.util
    assert NEW.GRASP_HEIGHT_OFFSET_M == {"BOLT_A": -.001, "BOLT_B": 0.}
    spec = importlib.util.spec_from_file_location("demo_offsets", base.ROOT / "scripts" / "pick_bolt_demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.GRASP_HEIGHT_OFFSET["part_01"] == NEW.GRASP_HEIGHT_OFFSET_M["BOLT_B"]


def test_grasp_height_override_moves_the_stationary_grasp(monkeypatch):
    """An offset of +1 mm raises the stationary grasp point (end of the descent) by 1 mm."""
    tips = []
    for value in (0., .001):
        ex, world = base.make(NEW, monkeypatch=monkeypatch)
        ex.grasp_height_offset["BOLT_B"] = value
        _, log = base.run_pick(ex, world, base.goal(world))
        descent = next(e for e in log if e[0] == "follow" and e[1] == 20 and e[2] == 2.)
        tips.append(base.tool_z(ex, descent))
    assert tips[1] - tips[0] == pytest.approx(.001, abs=1e-5)


def test_grasp_height_override_is_checked(monkeypatch):
    """Unknown parts and offsets beyond 5 mm are refused at start."""
    world = base.World.__new__(base.World)
    world.ns = 0
    node = base.FakeNode(world)
    ex = NEW.GazeboSortExecutor(node, base.ROOT, grasp_height_offset_m={"BOLT_A": -.001})
    assert ex.grasp_height_offset == {"BOLT_A": -.001, "BOLT_B": 0.}
    for bad in ({"BOLT_C": 0.}, {"BOLT_A": .006}, {"BOLT_A": float("nan")}):
        with pytest.raises(ValueError):
            NEW.GazeboSortExecutor(node, base.ROOT, grasp_height_offset_m=bad)
