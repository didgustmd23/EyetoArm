"""실행 세션 단위시험: 리셋 = 현재 실행 종료, 새 실행은 명시적으로 처음부터.

입력 칸·상태·디스패처는 실제 부품, TF·B는 대역이다. TF 병진 x에 출처 실행을 싣는다
(실행 A = 100대, 실행 B = 200대, 이전 실행 잔여 = −1000).
"""
import threading
import time
import types
import unittest

import numpy as np

from eyetoarm.inputs.live_slot import CameraProfileInfo, GenerationMirror, LiveFrameSlot, LiveSlotConfig
from eyetoarm.vision.live_observation_dispatch import LiveObservationDispatcher
from eyetoarm.vision.live_observation_envelope import LiveCaptureTfLookup
from eyetoarm.vision.live_observation_state import LiveObservationConfig, LiveObservationState
from eyetoarm.vision.live_run_session import LiveRunController, LiveRunSession, RunFrame

S = 1_000_000_000
WORLD, OPT = "world", "cam_optical"
PROFILE = CameraProfileInfo("cal-1", 640, 480, OPT)


def mat_x(x):
    """병진 x만 있는 4×4."""
    T = np.eye(4)
    T[0, 3] = x
    return T


class FakeTf:
    """실행마다 새로 만드는 TF 수신 묶음 대역(정확 stamp 표본만, 보간 없음)."""

    def __init__(self):
        """빈 버퍼."""
        self.samples = {}
        self.closed = False
        self.cleared = 0
        self.fail_close = 0      # 남은 닫기 실패 횟수(정리 실패 재현용)

    def put(self, stamp_ns, x):
        """표본 하나(같은 stamp는 덮어씀)."""
        self.samples[stamp_ns] = x

    def lookup(self, parent, child, stamp_ns, timeout_s):
        """정확 stamp 조회."""
        if stamp_ns not in self.samples:
            raise LookupError(f"no tf at {stamp_ns}")

        return mat_x(self.samples[stamp_ns]), stamp_ns

    def has_static(self, parent, child):
        """정적 확인(이 시험에서는 정적 쌍을 쓰지 않는다)."""
        return True

    def clear(self):
        """비우기(세션은 부르지 않아야 한다)."""
        self.cleared += 1
        self.samples.clear()

    def close(self, limit_s):
        """닫기. `fail_close`가 남아 있으면 예외(정리 실패)."""
        if self.fail_close > 0:
            self.fail_close -= 1
            raise RuntimeError("tf close failed")

        self.closed = True


class World:
    """세션 공장과 기록."""

    def __init__(self, hooks=None):
        """공장 호출 기록과 B 계산 기록."""
        self.tfs, self.slots, self.computed = [], [], []
        self.hooks = hooks or {}
        self.sessions = 0

    def make_session(self, run_id):
        """부품을 모두 새로 만드는 세션."""
        self.sessions += 1

        def make_tf():
            tf = FakeTf()
            self.tfs.append(tf)
            return tf

        def make_slot(hook):
            slot = LiveFrameSlot(LiveSlotConfig(frame_timeout_s=30.0), generation=GenerationMirror(1), boundary_hook=hook)
            self.slots.append(slot)
            return slot

        def make_dispatcher(state, tf):
            d = LiveObservationDispatcher(
                state, camera="cam", stage_factory=lambda c: "stage",
                tf_lookup=LiveCaptureTfLookup(tf.lookup, world_frame=WORLD, wait_limit_s=5.0),
                build_envelope=lambda fr, obs, cam, tf_result: tf_result,
                compute=lambda stage, env: (self.computed.append((run_id, env)) or env), hooks=self.hooks)
            d.start()
            return d

        def frame_of(record, generation):
            return types.SimpleNamespace(ros_stamp_ns=record.stamp_ns, optical_frame=record.frame_id,
                                         calibration_id=None if record.profile is None else record.profile.calibration_id,
                                         width=record.width, height=record.height,
                                         received_monotonic_ns=record.received_monotonic_ns)

        return LiveRunSession(run_id, generation=1,
                              state_config=LiveObservationConfig(write_capacity=16, reserved_capacity=64),
                              make_slot=make_slot, make_tf=make_tf, make_dispatcher=make_dispatcher, frame_of=frame_of,
                              world_frame=WORLD, static_pairs=(), wait_limit_s=5.0, future_tolerance_ns=S // 2)


def feed(session, stamp_ns, *, profile=True, clock=True):
    """CameraInfo(처음 한 번)·/clock·영상 → 입력 반영 → RunFrame."""
    slot = session.slot

    if profile:
        slot.on_camera_info(PROFILE)

    if clock:
        slot.on_clock(stamp_ns)

    slot.on_image(object(), stamp_ns=stamp_ns, frame_id=OPT, width=640, height=480)

    for item in slot.poll_inputs():
        slot.input_applied(item, 1)

    took = session.take(0.5)
    assert isinstance(took, RunFrame), took

    return took


def wait_for(pred, limit_s=2.0):
    """조건이 참이 될 때까지 짧게 기다린다."""
    end = time.monotonic() + limit_s

    while time.monotonic() < end:
        if pred():
            return True

        time.sleep(0.005)

    return pred()


def accepts(state):
    """수락 기록(꺼냄)."""
    return [e for e in state.take_events(1000) if e.kind == "accept"]


class OpenTest(unittest.TestCase):
    """새 실행은 보정 정보 반영과 촬영 시각 TF가 모두 준비돼야 열린다."""

    def setUp(self):
        """실행 A 하나."""
        self.world = World()
        self.ctl = LiveRunController(self.world.make_session, close_limit_s=2.0)
        self.run = self.ctl.start_run("A")

    def tearDown(self):
        """정리."""
        self.ctl.stop("test_end")

    def test_closed_until_calibration_and_tf(self):
        """보정 정보만으로는 닫힘(TF 없음), 그 영상 촬영 stamp TF가 새 버퍼에 오면 열린다. 버퍼를 비우지 않는다."""
        rf = feed(self.run, 1 * S)
        self.assertIsNone(self.run.state.admit(rx_epoch=rf.rx_epoch, generation=1, calibration_id="cal-1"))
        self.assertTrue(self.run.try_open(rf).startswith("dynamic_unavailable"))
        self.assertFalse(self.run.state.token.ready)

        self.world.tfs[-1].put(1 * S, 100.0)
        self.assertEqual(self.run.try_open(rf), "ready")
        self.assertTrue(self.run.state.token.ready)
        self.assertEqual(self.world.tfs[-1].cleared, 0)

    def test_no_calibration_not_open(self):
        """CameraInfo가 없어 보정 정보가 없는 영상으로는 열지 않는다."""
        rf = feed(self.run, 1 * S, profile=False)
        self.world.tfs[-1].put(1 * S, 100.0)
        self.assertEqual(self.run.try_open(rf), "no_calibration")
        self.assertFalse(self.run.state.token.ready)

    def test_tf_before_calibration_still_needs_both(self):
        """TF가 먼저 있어도 첫 열림 전 profile 경계는 TF 준비를 요구하고, 보정 반영과 함께 열린다."""
        self.world.tfs[-1].put(1 * S, 100.0)
        rf = feed(self.run, 1 * S)
        self.assertFalse(self.run.state.boundary.tf_ready)
        self.assertEqual(self.run.try_open(rf), "ready")

    def test_accept_after_open(self):
        """열린 뒤 그 실행 영상은 그 실행 버퍼의 TF로 계산·수락된다."""
        self.world.tfs[-1].put(1 * S, 100.0)
        rf = feed(self.run, 1 * S)
        self.run.try_open(rf)
        self.assertEqual(self.run.submit(rf, None), "ok")
        self.assertTrue(wait_for(lambda: self.run.dispatcher.counts.get("accepted") == 1))
        acc = accepts(self.run.state)
        self.assertEqual(len(acc), 1)
        self.assertEqual(acc[0].payload.envelope.tf.T_W_C[0, 3], 100.0)


class ResetTest(unittest.TestCase):
    """리셋·시간 역행: 현재 실행 종료, 현재성 해제, 자동 재시작 없음, 이전 결과가 새 실행에 섞이지 않음."""

    def open_run(self, ctl, world, run_id, stamp_ns, x):
        """실행을 시작해 연다."""
        run = ctl.start_run(run_id)
        world.tfs[-1].put(stamp_ns, x)
        rf = feed(run, stamp_ns)
        self.assertEqual(run.try_open(rf), "ready")
        return run, rf

    def test_clock_rewind_ends_run_and_releases(self):
        """/clock 역행: 계산 중 결과 수락 0, 이전 수락 철회, 이후 제출 거부, service 뒤 slot·TF·worker 닫힘, 새 실행 없음."""
        gate = {"armed": False}
        world = None

        def before_accept():
            if gate["armed"]:
                gate["armed"] = False
                ctl.current.slot.on_clock(1)                      # 계산 중에 리셋(시간 역행) 감지

        world = World(hooks={"before_accept": before_accept})
        ctl = LiveRunController(world.make_session, close_limit_s=2.0)
        run, rf = self.open_run(ctl, world, "A", 5 * S, 100.0)

        self.assertEqual(run.submit(rf, None), "ok")              # 첫 결과(현재)
        self.assertTrue(wait_for(lambda: run.dispatcher.counts.get("accepted") == 1))
        world.tfs[-1].put(6 * S, 101.0)
        rf2 = feed(run, 6 * S, profile=False)
        gate["armed"] = True
        self.assertEqual(run.submit(rf2, None), "ok")
        self.assertTrue(wait_for(lambda: run.dispatcher.counts.get("not_accepted") == 1))

        kinds = [e.kind for e in run.state.take_events(1000)]
        self.assertEqual(kinds.count("accept"), 1)
        self.assertIn("revoke", kinds)
        self.assertIn("run_end", kinds)
        self.assertTrue(run.reset_pending)
        self.assertEqual(run.submit(rf2, None), "ended")

        record = ctl.service()
        self.assertEqual((record.run_id, record.close_errors), ("A", ()))
        self.assertTrue(record.reason.startswith("reset:sim_time_rewind"))
        self.assertEqual(run.slot.on_image(object(), stamp_ns=7 * S, frame_id=OPT, width=640, height=480), "closed")
        self.assertTrue(world.tfs[-1].closed)
        self.assertFalse(run.dispatcher._thread.is_alive())
        self.assertEqual(world.sessions, 1)                       # 자동 재시작 없음
        self.assertIsNone(ctl.service())

    def test_image_stamp_regression_also_resets(self):
        """영상 stamp 역행도 리셋으로 보고 실행을 끝낸다."""
        world = World()
        ctl = LiveRunController(world.make_session, close_limit_s=2.0)
        run, _ = self.open_run(ctl, world, "A", 5 * S, 100.0)
        run.slot.on_image(object(), stamp_ns=1 * S, frame_id=OPT, width=640, height=480)
        self.assertTrue(run.reset_pending)
        self.assertTrue(run.state.closed)
        ctl.service()

    def test_non_rewind_boundary_keeps_run(self):
        """되감기가 아닌 경계(CameraInfo 변경)는 실행을 끝내지 않고 기존 규칙(반영 뒤 열림, TF 재준비 불필요)을 따른다."""
        world = World()
        ctl = LiveRunController(world.make_session, close_limit_s=2.0)
        run, _ = self.open_run(ctl, world, "A", 5 * S, 100.0)
        run.slot.on_camera_info(CameraProfileInfo("cal-2", 640, 480, OPT))
        self.assertFalse(run.reset_pending)
        world.tfs[-1].put(6 * S, 101.0)
        rf = feed(run, 6 * S, profile=False)
        self.assertEqual(run.try_open(rf), "ready")
        self.assertEqual(run.state.token.calibration_id, "cal-2")
        ctl.stop("test_end")

    def test_explicit_new_run_rules(self):
        """새 실행은 명시 호출로만: 현재 실행이 살아 있으면 거부, run_id 재사용 거부."""
        world = World()
        ctl = LiveRunController(world.make_session, close_limit_s=2.0)
        ctl.start_run("A")

        with self.assertRaises(ValueError):
            ctl.start_run("B")

        ctl.stop("manual")

        with self.assertRaises(ValueError):
            ctl.start_run("A")

        run_b = ctl.start_run("B")
        self.assertEqual((run_b.run_id, run_b.state.token.run_id, world.sessions), ("B", "B", 2))
        ctl.stop("test_end")

    def test_old_run_results_do_not_mix(self):
        """실행 A 리셋 뒤 실행 B: 부품이 모두 새것, A 영상 제출은 foreign_run, A의 늦은 계산은 수락 0, B 수락은 B 버퍼 값뿐."""
        release = threading.Event()
        entered = threading.Event()

        def in_compute():
            if not release.is_set():
                entered.set()
                release.wait(2.0)

        world = World(hooks={"in_compute": in_compute})
        ctl = LiveRunController(world.make_session, close_limit_s=2.0)
        run_a, rf_a = self.open_run(ctl, world, "A", 5 * S, 100.0)
        tf_a, slot_a = world.tfs[-1], world.slots[-1]

        self.assertEqual(run_a.submit(rf_a, None), "ok")          # A 계산이 걸린 채로 리셋
        self.assertTrue(entered.wait(2.0))
        slot_a.on_clock(1)
        release.set()
        self.assertTrue(wait_for(lambda: run_a.dispatcher.counts.get("not_accepted") == 1))
        ctl.service()

        run_b, rf_b = self.open_run(ctl, world, "B", 2 * S, 200.0)
        self.assertIsNot(world.tfs[-1], tf_a)
        self.assertIsNot(run_b.slot, slot_a)
        self.assertIsNot(run_b.state, run_a.state)
        self.assertIsNot(run_b.dispatcher, run_a.dispatcher)

        self.assertEqual(run_b.submit(rf_a, None), "foreign_run")  # 우연히 epoch·세대·보정이 같아도 거부
        self.assertEqual((rf_a.rx_epoch, rf_a.generation), (rf_b.rx_epoch, rf_b.generation))
        self.assertEqual(run_a.submit(rf_b, None), "ended")

        self.assertEqual(run_b.submit(rf_b, None), "ok")
        self.assertTrue(wait_for(lambda: run_b.dispatcher.counts.get("accepted") == 1))
        acc_b = accepts(run_b.state)
        self.assertEqual([e.payload.envelope.tf.T_W_C[0, 3] for e in acc_b], [200.0])
        self.assertEqual(accepts(run_a.state), [])
        self.assertEqual(run_b.state.token.run_id, "B")
        ctl.stop("test_end")

    def test_limit_late_old_publisher_tf_in_new_run(self):
        """남는 한계(보장 아님): 이전 실행 발행자의 늦은 /tf가 새 실행 버퍼에 같은 stamp로 들어오면 새 실행이 그 값을 쓴다."""
        world = World()
        ctl = LiveRunController(world.make_session, close_limit_s=2.0)
        run_b = ctl.start_run("B")
        world.tfs[-1].put(2 * S, 200.0)
        world.tfs[-1].put(2 * S, -1000.0)                         # 이전 실행 잔여가 나중에 도착(덮어씀)
        rf = feed(run_b, 2 * S)
        self.assertEqual(run_b.try_open(rf), "ready")
        run_b.submit(rf, None)
        self.assertTrue(wait_for(lambda: run_b.dispatcher.counts.get("accepted") == 1))
        self.assertEqual(accepts(run_b.state)[0].payload.envelope.tf.T_W_C[0, 3], -1000.0)
        ctl.stop("test_end")


class CleanupBarrierTest(unittest.TestCase):
    """정리 완료가 확인되기 전에는 다음 실행을 거부한다. 기존 계산 차단은 유지."""

    def open_run(self, ctl, world, run_id, stamp_ns, x):
        """실행을 시작해 연다."""
        run = ctl.start_run(run_id)
        world.tfs[-1].put(stamp_ns, x)
        rf = feed(run, stamp_ns)
        self.assertEqual(run.try_open(rf), "ready")
        return run, rf

    def check_new_run_b(self, ctl, world):
        """정리 확인 뒤 명시 새 실행 B가 열리고 B 값만 수락한다."""
        run_b, rf_b = self.open_run(ctl, world, "B", 2 * S, 200.0)
        self.assertEqual(run_b.submit(rf_b, None), "ok")
        self.assertTrue(wait_for(lambda: run_b.dispatcher.counts.get("accepted") == 1))
        self.assertEqual([e.payload.envelope.tf.T_W_C[0, 3] for e in accepts(run_b.state)], [200.0])
        ctl.stop("test_end")

    def test_worker_close_timeout_blocks_next_run(self):
        """장벽으로 worker가 상한 안에 안 멈춤 → 다음 실행 거부, 이전 계산 수락 0. 장벽 해제·재정리 뒤 B 성공."""
        release, entered = threading.Event(), threading.Event()

        def in_compute():
            if not release.is_set():
                entered.set()
                release.wait(5.0)

        world = World(hooks={"in_compute": in_compute})
        ctl = LiveRunController(world.make_session, close_limit_s=0.1)
        run_a, rf_a = self.open_run(ctl, world, "A", 5 * S, 100.0)
        self.assertEqual(run_a.submit(rf_a, None), "ok")
        self.assertTrue(entered.wait(2.0))

        run_a.slot.on_clock(1)                                   # 리셋
        record = ctl.service()
        self.assertTrue(any(e.startswith("dispatcher: did not stop") for e in record.close_errors))
        self.assertTrue(run_a.dispatcher._thread.is_alive())
        self.assertFalse(run_a.cleanup_confirmed)

        with self.assertRaises(ValueError):
            ctl.start_run("B")
        self.assertEqual(world.sessions, 1)

        release.set()
        self.assertTrue(wait_for(lambda: run_a.dispatcher.counts.get("not_accepted") == 1))
        self.assertEqual(accepts(run_a.state), [])                # 기존 계산 차단 유지

        record = ctl.retry_cleanup()
        self.assertEqual(record.close_errors, ())
        self.assertTrue(run_a.cleanup_confirmed)
        self.assertIsNone(ctl.retry_cleanup())
        self.check_new_run_b(ctl, world)

    def test_tf_close_failure_blocks_next_run(self):
        """TF 정리 실패 → 다음 실행 거부. 재정리 성공 뒤 B 성공. 성공한 단계(worker·slot)는 다시 하지 않는다."""
        world = World()
        ctl = LiveRunController(world.make_session, close_limit_s=1.0)
        run_a, _ = self.open_run(ctl, world, "A", 5 * S, 100.0)
        tf_a = world.tfs[-1]
        tf_a.fail_close = 1

        run_a.slot.on_clock(1)
        record = ctl.service()
        self.assertEqual(record.close_errors, ("tf: RuntimeError: tf close failed",))
        self.assertFalse(run_a.cleanup_confirmed)

        with self.assertRaises(ValueError):
            ctl.start_run("B")

        record = ctl.retry_cleanup()
        self.assertEqual((record.close_errors, tf_a.closed, run_a.cleanup_attempts), ((), True, 2))
        self.check_new_run_b(ctl, world)

    def test_persistent_close_false_keeps_refusing(self):
        """재현 결함 형태: dispatcher.close가 계속 False면 재정리해도 다음 실행을 계속 거부한다."""
        world = World()
        ctl = LiveRunController(world.make_session, close_limit_s=0.1)
        run_a, _ = self.open_run(ctl, world, "A", 5 * S, 100.0)
        real_close = run_a.dispatcher.close
        run_a.dispatcher.close = lambda limit_s: False

        run_a.slot.on_clock(1)
        ctl.service()
        ctl.retry_cleanup()

        with self.assertRaises(ValueError):
            ctl.start_run("B")
        self.assertEqual(world.sessions, 1)

        run_a.dispatcher.close = real_close
        self.assertEqual(ctl.retry_cleanup().close_errors, ())
        self.check_new_run_b(ctl, world)

    def test_manual_stop_with_failure_blocks_too(self):
        """명시 종료(stop)에서도 정리 실패면 다음 실행을 거부한다."""
        world = World()
        ctl = LiveRunController(world.make_session, close_limit_s=1.0)
        run_a, _ = self.open_run(ctl, world, "A", 5 * S, 100.0)
        world.tfs[-1].fail_close = 1
        ctl.stop("manual")

        with self.assertRaises(ValueError):
            ctl.start_run("B")

        ctl.retry_cleanup()
        self.check_new_run_b(ctl, world)


class StartClosedStateTest(unittest.TestCase):
    """`LiveObservationState(start_closed=True)`."""

    def make(self, start_closed):
        """상태 하나."""
        return LiveObservationState(run_id="r", generation=1, calibration_id=None, start_closed=start_closed,
                                    config=LiveObservationConfig(write_capacity=4, reserved_capacity=16))

    def test_default_is_open(self):
        """기본값은 기존처럼 열린 채 시작한다."""
        self.assertTrue(self.make(False).token.ready)

    def test_needs_applied_and_tf(self):
        """닫힌 채 시작, 반영만으로는 닫힘, TF 준비까지 끝나야 열림."""
        st = self.make(True)
        self.assertFalse(st.token.ready)
        st.on_applied(0, 1, "c")
        self.assertFalse(st.token.ready)
        st.on_tf_ready(0, st.begin_tf_attempt(0))
        self.assertTrue(st.token.ready)

    def test_boundary_before_first_open_needs_tf(self):
        """첫 열림 전 profile 경계도 TF 준비를 요구하고, 첫 열림 뒤의 같은 경계는 요구하지 않는다."""
        st = self.make(True)
        st.on_rx_boundary(1, "profile_changed")
        self.assertFalse(st.boundary.tf_ready)
        st.on_applied(1, 1, "c")
        st.on_tf_ready(1, st.begin_tf_attempt(1))
        self.assertTrue(st.token.ready)
        st.on_rx_boundary(2, "profile_changed")
        self.assertTrue(st.boundary.tf_ready)


if __name__ == "__main__":
    unittest.main()
