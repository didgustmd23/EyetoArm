"""추적 발행 시각을 영상 수신기의 `/clock` 값으로 읽는 방식(#419) 시험. rclpy·eyetoarm_interfaces는 가짜 모듈을 쓴다.

확인: 수신기의 마지막 시각(못 받음 0·되감기·슬롯이 비운 뒤), 쓰는 스레드와 읽는 스레드가 다를 때, 발행 시각 계수,
발행기의 시계 스레드 유무와 종료, 실행 진입점의 옵션 연결.
"""
from __future__ import annotations

import importlib.util
import sys
import threading
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

from eyetoarm.inputs.live_slot import BoundaryKind
from eyetoarm.inputs.ros_image import LiveLinkSource, LiveSourceConfig
from eyetoarm.vision.track_stream import RosTrackStreamSender, TrackStream
from test_ros_image_adapter import CountingIssuer, FakeTransport, image_msg, stamp, view_convert
from test_run_pipeline_live_observation import Scenario
from test_run_pipeline_track_stream import STREAM, FakeSender, StreamCli
from test_run_pipeline_gripper_handoff import GRIP
from test_track_stream import Sink, cfg, frame_msg

ROOT = Path(__file__).resolve().parents[2]
CLOCK_THREAD = "eyetoarm-track-stream-clock"


def clock_msg(ns):
    """가짜 `/clock` 메시지(ns → sec·nanosec)."""
    return SimpleNamespace(clock=stamp(ns // 1_000_000_000, ns % 1_000_000_000))


def make_source():
    """가짜 연결의 링크 소스."""
    return LiveLinkSource(source_id="gz_link", issuer=CountingIssuer(), generation_provider=lambda: 0,
                          config=LiveSourceConfig(frame_wait_s=0.01), transport_factory=lambda src: FakeTransport(),
                          convert=view_convert)


def clock_threads():
    """살아 있는 추적 발행 시계 스레드 수."""
    return sum(1 for t in threading.enumerate() if t.name == CLOCK_THREAD and t.is_alive())


class SourceClockTest(unittest.TestCase):
    """`LiveLinkSource.sim_now_ns`: 노드 SIM 시계와 같은 뜻의 마지막 `/clock` 값."""

    def setUp(self):
        self.src = make_source()
        self.addCleanup(self.src.close)

    def test_zero_before_clock(self):
        """`/clock`을 받기 전에는 0(노드 시계의 못 받음과 같음)."""
        self.assertEqual(self.src.sim_now_ns(), 0)

    def test_follows_last_clock_and_rewind(self):
        """받은 값을 그대로, 되감기면 같이 내려간다(되감기 경계는 슬롯이 그대로 만든다)."""
        self.src.handle_clock(clock_msg(5_000_000_000))
        self.assertEqual(self.src.sim_now_ns(), 5_000_000_000)
        self.src.handle_clock(clock_msg(5_000_000_001))
        self.assertEqual(self.src.sim_now_ns(), 5_000_000_001)
        self.src.handle_clock(clock_msg(1_000_000))
        self.assertEqual(self.src.sim_now_ns(), 1_000_000)
        self.assertIn(BoundaryKind.SIM_TIME_REWIND, [i.kind for i in self.src.poll_input()])

    def test_invalid_clock_keeps_value(self):
        """잘못된 `/clock`은 버리고(계수만) 값은 그대로."""
        self.src.handle_clock(clock_msg(7_000))
        self.src.handle_clock(SimpleNamespace(clock=stamp(-1, 0)))
        self.assertEqual(self.src.sim_now_ns(), 7_000)
        self.assertEqual(self.src.slot.stats()["invalid_clock"], 1)

    def test_kept_when_slot_clears_rewind_reference(self):
        """영상 시각 되감기(R6)로 슬롯이 되감기 기준을 비워도 지금 시각은 남는다."""
        self.src.handle_clock(clock_msg(3_000_000_000))
        self.src.handle_image(image_msg(3, 0))
        self.src.handle_image(image_msg(2, 0))   # 영상 시각 되감기: 슬롯의 마지막 clock을 비움
        self.assertIsNone(self.src.slot._last_clock_ns)
        self.assertEqual(self.src.sim_now_ns(), 3_000_000_000)

    def test_writer_and_reader_threads(self):
        """쓰는 스레드(수신)와 읽는 스레드(발행)가 다를 때: 읽은 값은 쓴 값 가운데 하나이고 줄지 않으며 끝값이 같다."""
        n = 20_000
        seen: list[int] = []
        done = threading.Event()

        def writer():
            """수신 스레드 흉내: 늘어나는 `/clock`."""
            for i in range(1, n + 1):
                self.src.handle_clock(clock_msg(i * 1_000))

            done.set()

        def reader():
            """발행 스레드 흉내: 끝날 때까지 읽는다."""
            while not done.is_set():
                seen.append(self.src.sim_now_ns())

            seen.append(self.src.sim_now_ns())

        threads = [threading.Thread(target=writer, name="rx"), threading.Thread(target=reader, name="pub")]

        for t in threads:
            t.start()

        for t in threads:
            t.join(30.0)

        self.assertFalse(any(t.is_alive() for t in threads))
        self.assertEqual(seen[-1], n * 1_000)
        self.assertTrue(all(v % 1_000 == 0 and 0 <= v <= n * 1_000 for v in seen))
        self.assertTrue(all(a <= b for a, b in zip(seen, seen[1:])))


class StreamCountTest(unittest.TestCase):
    """`TrackStream` 발행 시각 계수와 시계 방식 표지."""

    def run_stream(self, now_ns, msgs, **kw):
        """메시지를 넣고 닫은 뒤 (받은 것, 요약)."""
        sink = Sink()
        s = TrackStream(cfg(max_rate_hz=1000.0, queue_capacity=16), sink, now_ns, **kw)
        s.start()

        for m in msgs:
            s.offer(m)

        self.assertTrue(s.close(2.0))

        return sink.got, s.summary()

    def test_zero_stamp_counted_and_no_latency(self):
        """시각 0(못 받음)으로 나간 발행은 따로 세고 지연에 넣지 않는다(지금 동작 그대로)."""
        got, summ = self.run_stream(lambda: 0, [frame_msg(1_000_000_000)])
        self.assertEqual(got[0]["publish_stamp_ns"], 0)
        self.assertEqual(summ["counts"]["published_stamp_zero"], 1)
        self.assertEqual(summ["counts"]["published_before_capture"], 0)
        self.assertEqual(summ["capture_to_publish_sim_ms"]["n"], 0)

    def test_before_capture_counted(self):
        """프레임 발행 시각이 촬영보다 이르면 센다. 같거나 늦으면 세지 않는다. marker는 촬영 시각이 없어 세지 않는다."""
        from eyetoarm.vision.track_stream import marker_message

        ms = 1_000_000
        stamps = iter([10 * ms - 1, 20 * ms, 31 * ms, 40 * ms])
        msgs = [frame_msg(10 * ms, image_seq=1), frame_msg(20 * ms, image_seq=2), frame_msg(30 * ms, image_seq=3),
                marker_message(run_id="r", source_id="s", tracker_epoch=1, handoff_on=True)]
        _, summ = self.run_stream(lambda: next(stamps), msgs)
        self.assertEqual(summ["counts"]["published"], 4)
        self.assertEqual(summ["counts"]["published_before_capture"], 1)
        self.assertEqual(summ["counts"]["published_stamp_zero"], 0)

    def test_clock_source_label(self):
        """요약에 시계 방식 표지(기본 own)."""
        self.assertEqual(self.run_stream(lambda: 1, [])[1]["clock_source"], "own")
        self.assertEqual(self.run_stream(lambda: 1, [], clock_source="link")[1]["clock_source"], "link")


# ---------------------------------------------------------------------- 가짜 rclpy

class FakeNode:
    """가짜 노드: 받은 매개변수·발행기·SIM 시계."""

    def __init__(self, name, overrides):
        self.name, self.overrides = name, overrides
        self.published: list = []
        self.destroyed = False
        self.sim_ns = 42

    def create_publisher(self, msg_type, topic, qos):
        """발행기."""
        node = self
        return SimpleNamespace(publish=lambda m: node.published.append(m))

    def get_clock(self):
        """노드 시계."""
        return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=self.sim_ns))

    def destroy_node(self):
        """정리."""
        self.destroyed = True


class FakeExecutor:
    """가짜 executor: spin은 shutdown까지 막힌다."""

    def __init__(self, context=None):
        self._stop = threading.Event()
        self.nodes = []

    def add_node(self, node):
        """노드 추가."""
        self.nodes.append(node)

    def spin(self):
        """종료까지 대기."""
        self._stop.wait(10.0)

    def shutdown(self, timeout_sec=None):
        """종료."""
        self._stop.set()


def fake_ros(nodes):
    """`RosTrackStreamSender`가 import하는 모듈 묶음."""
    class Context:
        """가짜 문맥."""

        def __init__(self):
            self.alive = True

        def ok(self):
            """살아 있나."""
            return self.alive

    def init(context=None, signal_handler_options=None):
        """초기화."""
        context.alive = True

    def shutdown(context=None):
        """종료."""
        context.alive = False

    def create_node(name, context=None, parameter_overrides=()):
        """노드를 만들고 적는다."""
        nodes.append(FakeNode(name, list(parameter_overrides)))
        return nodes[-1]

    class Parameter:
        """가짜 매개변수."""

        class Type:
            """형식 이름."""

            BOOL = "bool"

        def __init__(self, name, type_=None, value=None):
            self.name, self.value = name, value

    enum = SimpleNamespace(RELIABLE=1, BEST_EFFORT=2, KEEP_LAST=1, VOLATILE=2)
    rclpy = ModuleType("rclpy")
    rclpy.init, rclpy.shutdown, rclpy.create_node = init, shutdown, create_node
    mods = {"rclpy": rclpy,
            "rclpy.context": SimpleNamespace(Context=Context),
            "rclpy.executors": SimpleNamespace(SingleThreadedExecutor=FakeExecutor),
            "rclpy.parameter": SimpleNamespace(Parameter=Parameter),
            "rclpy.qos": SimpleNamespace(DurabilityPolicy=enum, HistoryPolicy=enum, ReliabilityPolicy=enum,
                                         QoSProfile=lambda **kw: kw),
            "rclpy.signals": SimpleNamespace(SignalHandlerOptions=SimpleNamespace(NO=0)),
            "eyetoarm_interfaces": ModuleType("eyetoarm_interfaces"),
            "eyetoarm_interfaces.msg": SimpleNamespace(TrackState=SimpleNamespace, TrackStateArray=lambda: SimpleNamespace(tracks=[]))}

    return mods


class SenderTest(unittest.TestCase):
    """`RosTrackStreamSender`: own은 시계 스레드, link는 구독·스레드 없음. 닫은 뒤 남는 스레드 없음."""

    def make(self, **kw):
        """가짜 모듈로 발행기를 만든다."""
        nodes: list = []

        with mock.patch.dict(sys.modules, fake_ros(nodes)):
            sender = RosTrackStreamSender(cfg(), **kw)

        return sender, nodes[0]

    def test_own_has_clock_thread_and_stops(self):
        """기본(own): use_sim_time 노드 + 시계 스레드, 발행 시각은 노드 시계, 닫으면 스레드가 끝난다."""
        before = clock_threads()
        sender, node = self.make()
        self.assertEqual(sender.clock_source, "own")
        self.assertEqual([(p.name, p.value) for p in node.overrides], [("use_sim_time", True)])
        self.assertEqual(clock_threads(), before + 1)
        self.assertEqual(sender.now_ns(), 42)
        self.assertTrue(sender.close(2.0))
        self.assertEqual(clock_threads(), before)
        self.assertTrue(node.destroyed)

    def test_link_has_no_clock_subscription_or_thread(self):
        """link: use_sim_time 없음(= /clock 구독 없음), 시계 스레드 없음, 발행 시각은 주입 함수, 닫기는 True."""
        before = clock_threads()
        src = make_source()
        self.addCleanup(src.close)
        sender, node = self.make(now_ns=src.sim_now_ns)
        self.assertEqual(sender.clock_source, "link")
        self.assertEqual(node.overrides, [])
        self.assertEqual(clock_threads(), before)
        self.assertEqual(sender.now_ns(), 0)
        src.handle_clock(clock_msg(9_000_000_000))
        self.assertEqual(sender.now_ns(), 9_000_000_000)
        self.assertTrue(sender.close(2.0))
        self.assertTrue(node.destroyed)
        self.assertEqual(clock_threads(), before)


# ---------------------------------------------------------------------- 실행 진입점

def load_pipeline():
    """scripts/run_pipeline.py 모듈."""
    spec = importlib.util.spec_from_file_location("run_pipeline_419", ROOT / "scripts" / "run_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


class LinkSimClockTest(unittest.TestCase):
    """`_LinkSimClock`: 소스가 붙기 전 0, 팩토리를 감싸 붙이고, 다른 스레드에서 읽는다."""

    def test_before_and_after_attach(self):
        """붙기 전 0, 감싼 팩토리가 만든 소스의 값을 읽는다."""
        clock = load_pipeline()._LinkSimClock()
        self.assertEqual(clock.now_ns(), 0)
        made = []
        factory = clock.wrap(lambda issuer, gp: made.append(make_source()) or made[-1])
        src = factory(None, None)
        self.addCleanup(src.close)
        self.assertIs(src, made[0])
        src.handle_clock(clock_msg(123))
        self.assertEqual(clock.now_ns(), 123)

    def test_read_from_other_thread(self):
        """수신 스레드가 쓴 값을 다른 스레드에서 읽는다."""
        clock = load_pipeline()._LinkSimClock()
        src = make_source()
        self.addCleanup(src.close)
        clock.attach(src)
        writer = threading.Thread(target=lambda: src.handle_clock(clock_msg(77)), name="rx")
        writer.start()
        writer.join(5.0)
        got = []
        reader = threading.Thread(target=lambda: got.append(clock.now_ns()), name="pub")
        reader.start()
        reader.join(5.0)
        self.assertEqual(got, [77])


class CliClockTest(StreamCli):
    """`--track-stream-clock`: 기본 link(#430, 발행기에 수신기 시각 함수를 넘김), own이면 예전 호출."""

    def test_default_is_link(self):
        """기본(옵션 없음, #430): link — 발행기 팩토리에 시각 함수를 넘기고, 시작 줄 clock=link, 요약 clock_source=link."""
        got = []

        def factory(config, now_ns=None):
            """시각 함수가 오는지 적는다."""
            got.append(now_ns)
            return FakeSender(config)

        code, err, out = self.run_stream(Scenario(3), GRIP + STREAM, factory)
        self.assertEqual(code, 0, err)
        self.assertEqual(len(got), 1)
        self.assertTrue(callable(got[0]))
        self.assertIn("clock=link", err)
        self.assertEqual(self.summary(out)["world_tracking"]["track_stream"]["clock_source"], "link")

    def test_own_restores_old_call(self):
        """own(끄는 옵션): 발행기 팩토리를 예전처럼 설정 하나로 부르고, 시작 줄 clock=own, 요약 clock_source=own."""
        made = []
        code, err, out = self.run_stream(Scenario(3), GRIP + STREAM + ("--track-stream-clock", "own"),
                                         lambda c: made.append(FakeSender(c)) or made[-1])
        self.assertEqual(code, 0, err)
        self.assertEqual(len(made), 1)
        self.assertIn("clock=own", err)
        self.assertEqual(self.summary(out)["world_tracking"]["track_stream"]["clock_source"], "own")

    def test_default_without_track_stream_is_noop(self):
        """--track-stream 없이 옵션도 안 주면 기본값 때문에 거부되지 않고 발행기도 만들지 않는다."""
        code, err, out = self.run_stream(Scenario(3), GRIP, lambda c, now_ns=None: self.fail("no sender"))
        self.assertEqual(code, 0, err)
        self.assertNotIn("TRACK STREAM", err)
        self.assertNotIn("track_stream", self.summary(out)["world_tracking"])

    def test_own_without_track_stream_is_noop(self):
        """--track-stream 없이 own만 주면 예전처럼 무동작(거부는 link를 명시했을 때만)."""
        code, err, _ = self.run_stream(Scenario(3), GRIP + ("--track-stream-clock", "own"),
                                       lambda c, now_ns=None: self.fail("no sender"))
        self.assertEqual(code, 0, err)

    def test_link_uses_source_clock(self):
        """link: 발행 시각 = 링크 소스의 마지막 /clock(시나리오가 영상마다 넣음), 촬영보다 이른 발행·0 발행 없음."""
        made = []

        class LinkedSender(FakeSender):
            """주입된 시각 함수를 쓰는 대역."""

            def __init__(self, config, now_ns):
                super().__init__(config)
                self._now = now_ns

            def now_ns(self):
                """주입된 시각."""
                return self._now()

        def factory(config, now_ns=None):
            """link면 now_ns가 와야 한다."""
            self.assertIsNotNone(now_ns)
            made.append(LinkedSender(config, now_ns))
            return made[-1]

        code, err, out = self.run_stream(Scenario(5), GRIP + STREAM + ("--track-stream-clock", "link"), factory)
        self.assertEqual(code, 0, err)
        self.assertIn("clock=link", err)
        frames = [m for m in made[0].got if m["kind"] == "frame"]
        self.assertTrue(frames)
        self.assertTrue(all(m["publish_stamp_ns"] >= m["capture_stamp_ns"] > 0 for m in frames))
        block = self.summary(out)["world_tracking"]["track_stream"]
        self.assertEqual(block["clock_source"], "link")
        self.assertEqual((block["counts"]["published_stamp_zero"], block["counts"]["published_before_capture"]), (0, 0))

    def test_link_requires_track_stream(self):
        """--track-stream 없이 link면 종료 2(발행기 없음)."""
        code, err, _ = self.run_stream(Scenario(3), GRIP + ("--track-stream-clock", "link"),
                                       lambda c, now_ns=None: self.fail("no sender"))
        self.assertEqual(code, 2, err)   # 인자 오류 문구는 argparse가 sys.stderr로 낸다


if __name__ == "__main__":
    unittest.main()
