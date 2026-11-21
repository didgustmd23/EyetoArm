"""실시간 입력·기록 수정 시험: 리셋 경계, TF 수신 실패, 대기 기록 상한, 입력 오류 집계,
확인 창 valid 문구, 세션 생성 실패 정리."""
from __future__ import annotations

import tempfile
import threading
import time
import unittest
from types import SimpleNamespace

from eyetoarm.inputs.live_slot import BoundaryKind, GenerationMirror, LiveFrameSlot, LiveSlotConfig
from eyetoarm.transforms.live_tf_buffer import LiveTfRuntime
from eyetoarm.transforms.observation_envelope_adapter import EnvelopeTfLookup
from eyetoarm.vision.live_observation_link import LiveObservationLink, LiveObservationLinkConfig

GAP_NS = 2_000_000_000          # frame_timeout_s(기본 1 s)보다 긴 영상 중단
OLD_NS = 50_000_000_000         # 리셋 전 실행의 시각
NEW_NS = 100_000_000            # 리셋 뒤 새 실행의 시각


def make_slot(hook=None):
    """시계를 손으로 움직이는 slot과 (경계 목록, 시계)."""
    t = [0]
    kinds = []

    def record(epoch, kind):
        """경계 종류를 모으고 주어진 훅도 부른다."""
        kinds.append(kind.value)

        if hook is not None:
            hook(epoch, kind)

    slot = LiveFrameSlot(LiveSlotConfig(), generation=GenerationMirror(0), clock_ns=lambda: t[0], boundary_hook=record)

    return slot, kinds, t


def image(slot, stamp_ns):
    """영상 하나(4×4, frame_id f)."""
    return slot.on_image(object(), stamp_ns=stamp_ns, frame_id="f", width=4, height=4)


def reset_in_gap(slot, t, *, clock_first: bool):
    """이전 실행 → 영상 중단 중 Gazebo 리셋 → 재개(clock_first면 새 /clock이 먼저 온다)."""
    slot.on_clock(OLD_NS)
    image(slot, OLD_NS)
    t[0] += GAP_NS

    if clock_first:
        slot.on_clock(NEW_NS)
        image(slot, NEW_NS)
    else:
        image(slot, NEW_NS)
        slot.on_clock(NEW_NS)


LINK_CFG = LiveObservationLinkConfig(world_frame="world", tf_wait_limit_s=0.05, static_pairs=(), future_tolerance_ns=0,
                                     write_capacity=8, reserved_capacity=8, lag_warn_s=1.0, close_limit_s=1.0,
                                     skip_queue_capacity=8)


def fake_tf():
    """조회·정적 확인·닫기만 있는 TF 묶음 대역."""
    return SimpleNamespace(lookup=lambda p, c, s, t: (None, s), has_static=lambda p, c: True, close=lambda s: None)


def make_link(tf=None):
    """새 출력 폴더의 기록 연결(시작 전)."""
    d = tempfile.mkdtemp()

    return LiveObservationLink("run-196", d + "/o", config=LINK_CFG, tf=tf or fake_tf(),
                               stage_factory=lambda cam: (SimpleNamespace(on_output=None), None), render=None)


def open_link(link):
    """첫 열림: 프로필 경계 반영 + TF 준비."""
    st = link.state
    link.boundary_hook(1, BoundaryKind.PROFILE_CHANGED)
    st.on_applied(1, 0, "cal")
    st.on_tf_ready(1, st.begin_tf_attempt(1))


class ResetInGapSlotTest(unittest.TestCase):
    """slot: 중단 뒤 재개 영상의 stamp 되감기는 재접속이 아니라 리셋(SIM_TIME_REWIND)이다."""

    def test_image_first_makes_rewind_boundary(self):
        """재개 첫 영상이 /clock보다 먼저 와도 SIM_TIME_REWIND 경계가 생기고 되감기 필터가 켜진다."""
        slot, kinds, t = make_slot()
        reset_in_gap(slot, t, clock_first=False)

        self.assertIn("sim_time_rewind", kinds)
        self.assertEqual(kinds.count("sim_time_rewind"), 1)
        self.assertTrue(slot.stats()["rewind_filter_active"])
        self.assertEqual(slot.stats()["rewind_merged_into_resume"], 0)

    def test_both_orders_same_reset_decision(self):
        """두 도착 순서 모두 되감기 경계가 1번이고 늦게 온 /clock은 경계를 더 만들지 않는다."""
        for clock_first in (False, True):
            with self.subTest(clock_first=clock_first):
                slot, kinds, t = make_slot()
                reset_in_gap(slot, t, clock_first=clock_first)

                self.assertEqual(kinds.count("sim_time_rewind"), 1, kinds)
                self.assertEqual(set(kinds) - {"sim_time_rewind", "frame_timeout", "reconnect"}, set())

    def test_image_first_filter_drops_late_old_run_image(self):
        """영상 먼저 순서에서도 리셋 뒤 늦게 도착한 이전 실행 영상을 넘기지 않는다."""
        slot, kinds, t = make_slot()
        reset_in_gap(slot, t, clock_first=False)
        t[0] += 10_000_000

        self.assertEqual(image(slot, OLD_NS + 33_000_000), "stale_after_rewind")

    def test_plain_resume_without_rewind_unchanged(self):
        """stamp가 줄지 않은 재개는 FRAME_TIMEOUT → RECONNECT뿐이다."""
        slot, kinds, t = make_slot()
        slot.on_clock(OLD_NS)
        image(slot, OLD_NS)
        t[0] += GAP_NS
        image(slot, OLD_NS + 66_000_000)

        self.assertEqual(kinds, ["frame_timeout", "reconnect"])


class ResetInGapLinkTest(unittest.TestCase):
    """연결부: 두 순서 모두 실행이 끝나고 같은 run_id로 다시 열리지 않는다."""

    def test_link_ends_run_in_both_orders(self):
        """slot 경계 훅을 꽂으면 두 순서 모두 reset_pending·closed이고 토큰이 다시 열리지 않는다."""
        for clock_first in (False, True):
            with self.subTest(clock_first=clock_first):
                link = make_link()
                open_link(link)
                slot, kinds, t = make_slot(hook=lambda e, k, link=link: link.boundary_hook(e + 1, k))
                reset_in_gap(slot, t, clock_first=clock_first)

                # 실행기가 다른 경계를 반영해도 닫힌 실행은 다시 열리지 않는다
                link.state.on_applied(99, 5, "cal")

                self.assertTrue(link.reset_pending)
                self.assertTrue(link.state.closed)
                self.assertFalse(link.state.token.ready)
                self.assertIn("sim_time_rewind", link.reset_reason)


class _CrashExec:
    """spin이 곧바로 예외를 내는 executor 대역."""

    def __init__(self, context=None):
        """컨텍스트 무시."""

    def add_node(self, n):
        """노드 무시."""

    def spin(self):
        """수신 스레드 사망."""
        raise RuntimeError("executor crashed")

    def shutdown(self, timeout_sec=None):
        """종료(아무것도 안 함)."""


class _ReturnExec(_CrashExec):
    """spin이 예외 없이 곧바로 반환하는 executor 대역(예기치 않은 반환)."""

    def spin(self):
        """닫지 않았는데 반환."""
        return None


def fake_modules(executor):
    """LiveTfRuntime 시험 주입 모듈(ROS 없음)."""
    return {"rclpy": SimpleNamespace(create_node=lambda *a, **k: SimpleNamespace(destroy_node=lambda: None)),
            "Buffer": lambda **k: SimpleNamespace(clear=lambda: None), "TransformListener": lambda *a, **k: None,
            "SingleThreadedExecutor": executor, "Time": None, "Duration": lambda seconds: None,
            "Parameter": lambda *a, **k: None}


def wait_until(pred, limit_s=2.0):
    """조건이 참이 될 때까지 짧게 기다린다(시험 동기화)."""
    end = time.monotonic() + limit_s

    while time.monotonic() < end:
        if pred():
            return True

        threading.Event().wait(0.01)

    return pred()


class TfReceiverFailureTest(unittest.TestCase):
    """B TF 수신 스레드 사망은 콜백으로 알리고 연결 상태를 실패로 올리며 정리 확인은 거짓이다."""

    def test_runtime_reports_crash_and_unexpected_return(self):
        """예외·예기치 않은 반환 모두 실패 콜백이 사유와 함께 한 번 불린다."""
        for executor, text in ((_CrashExec, "executor crashed"), (_ReturnExec, "returned")):
            with self.subTest(executor=executor.__name__):
                got = []
                rt = LiveTfRuntime(None, modules=fake_modules(executor))
                rt.set_failure_callback(got.append)
                rt.start()

                self.assertTrue(wait_until(lambda: got))
                self.assertEqual(len(got), 1)
                self.assertIn(text, got[0])
                self.assertIn(text, rt.spin_error)
                rt.close(0.5)

    def test_close_does_not_report_failure(self):
        """정상 close로 spin이 끝나면 실패로 알리지 않는다."""
        stop = threading.Event()

        class BlockExec(_CrashExec):
            """shutdown이 올 때까지 막혀 있는 spin."""

            def spin(self):
                """정상 대기."""
                stop.wait(2.0)

            def shutdown(self, timeout_sec=None):
                """spin을 풀어 준다."""
                stop.set()

        got = []
        rt = LiveTfRuntime(None, modules=fake_modules(BlockExec))
        rt.set_failure_callback(got.append)
        rt.start()
        rt.close(1.0)

        self.assertEqual(got, [])
        self.assertIsNone(rt.spin_error)

    def test_exception_during_close_not_reported(self):
        """정리 중 executor 예외는 spin_error만 남고 실패 콜백은 불리지 않는다."""
        stop = threading.Event()

        class RaiseOnShutdownExec(_CrashExec):
            """shutdown이 오면 spin이 예외로 끝난다."""

            def spin(self):
                """정리 요청까지 대기한 뒤 예외."""
                stop.wait(2.0)
                raise RuntimeError("executor raised during shutdown")

            def shutdown(self, timeout_sec=None):
                """spin을 풀어 준다."""
                stop.set()

        got = []
        rt = LiveTfRuntime(None, modules=fake_modules(RaiseOnShutdownExec))
        rt.set_failure_callback(got.append)
        rt.start()
        rt.close(1.0)

        self.assertEqual(got, [])
        self.assertIn("executor raised during shutdown", rt.spin_error)

        late = []
        rt.set_failure_callback(late.append)   # 정리 중 예외는 실패가 아니므로 늦게 등록해도 불리지 않는다

        self.assertEqual(late, [])

    def test_link_fails_state_and_cleanup_not_confirmed(self):
        """연결부 TF가 죽으면 상태는 tf_receiver_failed, cleanup_confirmed 거짓, 정리 오류에 사유."""
        rt = LiveTfRuntime(None, modules=fake_modules(_CrashExec))
        link = make_link(tf=rt)
        rt.start()
        link.start()

        self.assertTrue(wait_until(lambda: link.state.failed is not None))
        self.assertEqual(link.state.failed, "tf_receiver_failed")

        record = link.end("test_end", limit_s=1.0)

        self.assertFalse(link.cleanup_confirmed)
        self.assertTrue(any("tf receiver failed" in e for e in record.close_errors), record.close_errors)
        self.assertIn("executor crashed", link.summary()["tf_receiver_error"])


class EnvelopeTfLookupBoundTest(unittest.TestCase):
    """EnvelopeTfLookup 대기 기록은 상한이 있고 요약 최대값은 전체 실행 기준이다."""

    def test_wait_history_bounded_and_max_kept(self):
        """호출 수가 상한을 넘어도 wait_ms 길이는 상한 이하이고 max_wait_ms는 유지된다."""
        ticks = iter([0.0, 0.040] + [1.0, 1.0] * 5000)
        lk = EnvelopeTfLookup(clock=lambda: next(ticks))

        for _ in range(5001):
            lk(SimpleNamespace(ros_stamp_ns=1, optical_frame="f"))

        self.assertLessEqual(len(lk.wait_ms), EnvelopeTfLookup.WAIT_HISTORY)
        self.assertAlmostEqual(lk.max_wait_ms, 40.0, places=6)
        self.assertEqual(lk.calls, 5001)


class InvalidInputCountTest(unittest.TestCase):
    """잘못된 영상 헤더·/clock을 통계로 센다."""

    def test_invalid_header_and_clock_counted(self):
        """헤더 해석 실패는 invalid_header, 잘못된 /clock은 invalid_clock으로 센다."""
        from eyetoarm.inputs.ros_image import LiveLinkSource, LiveSourceConfig

        src = LiveLinkSource(source_id="gz_link", issuer=SimpleNamespace(next_seq=lambda: 0),
                             generation_provider=lambda: 0, config=LiveSourceConfig(frame_wait_s=0.005),
                             transport_factory=lambda s: SimpleNamespace(), convert=lambda m: m,
                             clock_ns=time.monotonic_ns)

        self.assertEqual(src.handle_image(SimpleNamespace(header=None)), "invalid_header")
        src.handle_clock(SimpleNamespace(clock=None))
        stats = src.slot.stats()

        self.assertEqual((stats["invalid_header"], stats["invalid_clock"]), (1, 1))


class LiveViewValidTextTest(unittest.TestCase):
    """확인 창은 실제 관측 valid 값을 쓴다(로봇 목표 아님 문구 유지)."""

    def test_valid_observation_text(self):
        """확인 창 문구가 관측 valid 값을 따르고 로봇 목표가 아니라는 문구는 남는다."""
        from eyetoarm.vision import live_view as lv

        frame = SimpleNamespace(run_id="r", source_id="s", image_seq=0, timestamp_ms=0, timestamp_base="SIM_TIME")

        for valid in (True, False):
            with self.subTest(valid=valid):
                item = lv.ViewItem(frame=frame, result=None, job_seq=0, generation=0, conf_threshold=None,
                                   reason=None if valid else "NO_OBJECT", valid=valid)
                snap = lv.ViewSnapshot(version=1, state=lv.ViewState.LIVE, item=item, last_frame=None,
                                       message=lv.status_text(valid), events=())
                text = " ".join(lv._info_lines(snap))

                self.assertIn(f"valid={valid}", text)
                self.assertNotIn(f"valid={not valid}", text)
                self.assertIn("not a robot target", text)
                self.assertEqual(lv.status_text(valid), f"display-only detection (valid={valid}, not for robot)")


class SessionInitCleanupTest(unittest.TestCase):
    """LiveRunSession 생성 중 실패하면 이미 만든 slot도 닫는다."""

    def test_slot_closed_when_dispatcher_fails(self):
        """디스패처 생성이 실패하면 원래 예외가 오르고 slot·TF가 닫힌다."""
        from eyetoarm.vision.live_run_session import LiveRunSession

        closed = []
        slot = SimpleNamespace(close=lambda: closed.append("slot"))
        tf = SimpleNamespace(close=lambda s: closed.append("tf"), has_static=lambda p, c: True,
                             lookup=lambda *a: None)

        def bad_dispatcher(state, tf):
            """생성 실패."""
            raise RuntimeError("dispatcher boom")

        from eyetoarm.vision.live_observation_state import LiveObservationConfig

        with self.assertRaises(RuntimeError):
            LiveRunSession("run-196", generation=1, state_config=LiveObservationConfig(write_capacity=4, reserved_capacity=4),
                           make_slot=lambda hook: slot, make_tf=lambda: tf, make_dispatcher=bad_dispatcher,
                           frame_of=lambda r, g: r, world_frame="world", static_pairs=(), wait_limit_s=0.05,
                           future_tolerance_ns=0)

        self.assertEqual(sorted(closed), ["slot", "tf"])


if __name__ == "__main__":
    unittest.main()
