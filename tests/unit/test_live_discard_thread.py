"""#397: 운영자 데이터 폐기(C `~/discard_detections`)의 스레드 규칙 시험(C 5758ccf `clear_from_control_thread` 기준).

C 0d3d69b는 ROS 서비스 스레드에서 A→C 연결부 `clear`를 불러 `HandoffContractError`(생산자 스레드 고정)로 실패했다.
5758ccf는 서비스 스레드가 CLEAR 요청을 넣고 생산자(실행기 제어 흐름)가 다음 `check()`에서 적용한다.
모든 시험이 실제 스레드 두 개 이상(부르는 스레드 ≠ 생산자 스레드)을 쓴다.
"""
from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from eyetoarm.vision.handoff import AToCHandoff, HandoffContractError, MsgKind
from eyetoarm.vision.live_observation_dispatch import LiveObservationDispatcher
from test_live_sort_part import CliBase
from test_live_sort_run_source import MANAGED_CONFIG, MANAGED_EXTRA, TRACK_FRAMES, ManagedRuntime
from test_run_pipeline_live_observation import Scenario

LIMIT_S = 5.0   # 시험 스레드 기다림 상한(정상이면 훨씬 빨리 끝남)


class Producer(threading.Thread):
    """실행기 대역: 같은 스레드에서 start_run 뒤, 멈출 때까지 짧은 간격으로 check()를 부른다(`checking`이 꺼지면 부르지 않음)."""

    def __init__(self, handoff, pipeline, interval_s=0.005):
        """시작은 호출자가."""
        super().__init__(name="test-producer", daemon=True)
        self.handoff, self.pipeline, self.interval_s = handoff, pipeline, interval_s
        self.ready, self.halt, self.checking = threading.Event(), threading.Event(), threading.Event()
        self.checking.set()
        self.error = None

    def run(self):
        """start_run → check 고리."""
        try:
            self.handoff.start_run(self.pipeline)
            self.ready.set()

            while not self.halt.wait(self.interval_s):
                if self.checking.is_set():
                    self.handoff.check()
        except BaseException as exc:   # 시험이 확인한다
            self.error = exc


class ClearFromControlThreadTest(unittest.TestCase):
    """연결부 `clear_from_control_thread`: 실제 스레드 두 개, 한도 초과·취소, 생산자 멈춤·종료."""

    def setUp(self):
        """생산자 스레드에서 start_run까지 한 연결부."""
        from eyetoarm.vision.pipeline import VisionPipeline
        from test_ai_config import ai_config
        from test_pipeline import settings   # 기존 연결부 시험(test_vision_handoff)과 같은 설정 대역

        self.handoff = AToCHandoff()
        pipeline = VisionPipeline(settings(detector_config=ai_config(), profile=None), run_id="run-397",
                                  clock_ns=time.monotonic_ns)
        self.producer = Producer(self.handoff, pipeline)
        self.producer.start()
        self.assertTrue(self.producer.ready.wait(LIMIT_S))

    def tearDown(self):
        """생산자를 멈춘다."""
        self.producer.halt.set()
        self.producer.join(LIMIT_S)

    def kinds(self):
        """지금까지 넣은 메시지 종류(꺼내지 않고 대기열을 본다)."""
        with self.handoff._lock:
            return [(m.msg_kind, getattr(m, "reason", None)) for m in self.handoff._queue]

    def test_direct_clear_from_other_thread_is_refused(self):
        """고치기 전 경로: 생산자가 아닌 스레드에서 clear를 부르면 HandoffContractError."""
        with self.assertRaisesRegex(HandoffContractError, "producer methods must be called from one thread"):
            self.handoff.clear("operator_operator")

    def test_clear_applied_on_producer_at_next_check(self):
        """다른 스레드의 요청은 생산자의 다음 check()에서 CLEAR로 들어가고 msg_seq가 돌아온다."""
        seq = self.handoff.clear_from_control_thread("operator_operator", timeout_s=LIMIT_S)
        self.assertEqual(self.kinds()[-1], (MsgKind.CLEAR, "operator_operator"))
        self.assertEqual(seq, self.handoff.stats().published)
        self.assertIsNone(self.producer.error)

    def test_timeout_when_producer_does_not_check(self):
        """생산자가 check()를 부르지 않는 동안에는 한도 뒤 TimeoutError, 취소된 요청은 나중 check()에서 넣지 않는다."""
        self.producer.checking.clear()
        time.sleep(0.05)
        before = self.handoff.stats().published

        with self.assertRaises(TimeoutError):
            self.handoff.clear_from_control_thread("operator_operator", timeout_s=0.1)

        self.producer.checking.set()
        time.sleep(0.1)
        self.assertEqual(self.handoff.stats().published, before)
        self.assertNotIn((MsgKind.CLEAR, "operator_operator"), self.kinds())

    def test_after_producer_stops_waits_full_limit(self):
        """생산자가 끝난 뒤(정리 단계)의 요청은 바로 실패하지 않고 한도까지 기다린 뒤 TimeoutError(빠른 실패 없음)."""
        self.producer.halt.set()
        self.producer.join(LIMIT_S)
        started = time.monotonic()

        with self.assertRaises(TimeoutError):
            self.handoff.clear_from_control_thread("operator_operator", timeout_s=0.3)

        self.assertGreaterEqual(time.monotonic() - started, 0.3)

    def test_closed_handoff_also_waits_full_limit(self):
        """연결부가 닫힌 뒤에도 요청은 한도까지 기다린 뒤 TimeoutError(닫힘을 바로 알리지 않음)."""
        self.producer.checking.clear()
        self.handoff.close()

        with self.assertRaises(TimeoutError):
            self.handoff.clear_from_control_thread("operator_operator", timeout_s=0.2)

    def test_concurrent_requests_each_applied_once(self):
        """두 제어 스레드의 동시 요청은 각각 CLEAR 1건으로 한 번씩 들어간다."""
        out = []
        threads = [threading.Thread(target=lambda r=r: out.append(self.handoff.clear_from_control_thread(r, timeout_s=LIMIT_S)))
                   for r in ("operator_a", "operator_b")]

        for t in threads:
            t.start()

        for t in threads:
            t.join(LIMIT_S)

        clears = [r for k, r in self.kinds() if k is MsgKind.CLEAR and r.startswith("operator_")]
        self.assertEqual(sorted(clears), ["operator_a", "operator_b"])
        self.assertEqual(len(set(out)), 2)


class DiscardRuntime(ManagedRuntime):
    """가짜 관리 런타임 + C 폐기 서비스 자리(`set_discard_listener`). 서비스 응답 형식은 실제 런타임 함수로 만든다."""

    def __init__(self, *, call_at_set=False, **kw):
        """활성 실행(job-1, epoch 1). `call_at_set`: 받을 곳이 붙자마자(실행기 시작 전) 다른 스레드에서 한 번 부른다."""
        super().__init__((True, "job-1", 1), **kw)
        self.discard_listener = None
        self.call_at_set = call_at_set
        self.replies = []
        self.threads = []

    def set_discard_listener(self, listener):
        """받을 곳 기록(실행기 시작 전에 붙는다)."""
        self.discard_listener = listener

        if self.call_at_set:
            self.call_in_thread()

    def call_in_thread(self):
        """서비스 스레드 대역에서 실제 서비스 함수(`_discard_detections`)로 한 번 부른다."""
        def service():
            """C 서비스 호출 한 번."""
            self.replies.append(service_reply(self.discard_listener))

        t = threading.Thread(target=service, name="test-discard-service")
        self.threads.append(t)
        t.start()

        return t

    def join(self):
        """부른 스레드가 모두 끝날 때까지."""
        for t in self.threads:
            t.join(LIMIT_S * 2)


def service_reply(listener):
    """`_RosSortPartRuntime._discard_detections`를 rclpy 없이 부른다: (success, message, 부른 스레드)."""
    mod = _run_pipeline_module()
    ready = threading.Event()
    ready.set()
    fake = SimpleNamespace(_discard_ready=ready, _discard_listener=listener)
    resp = mod._RosSortPartRuntime._discard_detections(fake, None, SimpleNamespace(success=None, message=None))

    return resp.success, resp.message, threading.get_ident()


_RP = []


def _run_pipeline_module():
    """scripts/run_pipeline.py를 한 번만 적재한다."""
    if not _RP:
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "scripts" / "run_pipeline.py"
        spec = importlib.util.spec_from_file_location("rp_397", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _RP.append(mod)

    return _RP[0]


class RecordingHandoff(AToCHandoff):
    """start_run과 CLEAR 적용을 한 스레드를 남기는 연결부."""

    def __init__(self, **kw):
        """기록."""
        super().__init__(**kw)
        self.calls = []

    def start_run(self, pipeline):
        """시작 스레드 기록."""
        self.calls.append(("start_run", None, threading.get_ident()))
        return super().start_run(pipeline)

    def offer_result(self, outcome, **kw):
        """RESULT 순서 기록(사진 번호)."""
        self.calls.append(("result", outcome.context.frame.image_seq, threading.get_ident()))
        return super().offer_result(outcome, **kw)

    def _clear_on_producer(self, reason):
        """CLEAR를 실제로 넣는 스레드 기록(clear·clear_from_control_thread 공통 경로)."""
        self.calls.append(("clear", reason, threading.get_ident()))
        return super()._clear_on_producer(reason)


class CliDiscardTest(CliBase):
    """진입점 관리 실행: 서비스 스레드 폐기 → 실행기 제어 흐름에서 처리."""

    def run_with_handoff(self, scenario, rt):
        """기록 연결부로 관리 실행 → (종료 코드, stderr, 출력 폴더, 연결부)."""
        made = []

        def factory():
            """`_default_live_handoff` 대역."""
            from eyetoarm.vision.c_handoff_probe import HandoffProbe

            probe = HandoffProbe()
            h = RecordingHandoff(notify=probe.notify)
            made.append(h)
            return h, probe

        with mock.patch.object(self.mod, "_default_live_handoff", factory):
            code, err, out = self.run_sort(scenario, config=MANAGED_CONFIG, extra=MANAGED_EXTRA,
                                           factory=self.factory(rt))

        rt.join()

        return code, err, out, made[0]

    def test_discard_during_run_clears_on_runner_thread(self):
        """실행 중 다른 스레드의 폐기: 성공 응답, CLEAR는 start_run과 같은 스레드, 실행은 계속(종료 0)."""
        rt = DiscardRuntime()
        sc = Scenario(TRACK_FRAMES, during={2: lambda s: rt.call_in_thread()})
        code, err, out, handoff = self.run_with_handoff(sc, rt)
        self.assertEqual(code, 0, err)
        self.assertEqual(len(rt.replies), 1)
        success, message, caller = rt.replies[0]
        self.assertEqual((success, message), (True, "A/B detections discarded"))
        runner = [t for name, _, t in handoff.calls if name == "start_run"][0]
        clears = [(r, t) for name, r, t in handoff.calls if name == "clear" and r.startswith("operator_")]
        self.assertEqual(clears, [("operator_operator", runner)])
        self.assertNotEqual(caller, runner)
        self.assertGreater(len(self.accepts(out)), 0)

    def test_discard_before_runner_start_does_not_take_producer(self):
        """실행기 시작 전 폐기: 서비스 스레드가 연결부 생산자 자리를 차지하지 않고 실행기가 처리한다(종료 0)."""
        rt = DiscardRuntime(call_at_set=True)
        code, err, out, handoff = self.run_with_handoff(Scenario(4), rt)
        self.assertEqual(code, 0, err)
        self.assertEqual([r[:2] for r in rt.replies], [(True, "A/B detections discarded")])
        threads = {t for name, _, t in handoff.calls}
        self.assertEqual(len(threads), 1, handoff.calls)

    def test_inflight_b_result_after_discard_is_not_accepted(self):
        """폐기 때 계산 중이던 B 결과는 수락·송신되지 않는다(늦은 결과 버림). 폐기 뒤 새 프레임은 다시 수락된다."""
        rt = DiscardRuntime()
        orig = LiveObservationDispatcher._process
        held = {}

        def process(disp, item):
            """세 번째 B 계산 중에 폐기를 부르고, 토큰이 바뀐 뒤에야 계산을 끝낸다(한 번만)."""
            held["n"] = held.get("n", 0) + 1

            if held["n"] != 3:
                return orig(disp, item)

            seq = disp.state.token.token_seq
            held["stamp"] = item.frame.ros_stamp_ns
            rt.call_in_thread()
            end = time.monotonic() + LIMIT_S

            while disp.state.token.token_seq == seq and time.monotonic() < end:
                time.sleep(0.002)

            held["token_changed"] = disp.state.token.token_seq != seq
            return orig(disp, item)

        with mock.patch.object(LiveObservationDispatcher, "_process", process):
            code, err, out, _ = self.run_with_handoff(Scenario(TRACK_FRAMES), rt)

        self.assertEqual(code, 0, err)
        self.assertTrue(held.get("token_changed"))
        self.assertEqual([r[:2] for r in rt.replies], [(True, "A/B detections discarded")])
        stamps = [r["input"]["ros_stamp_ns"] for r in self.accepts(out)]
        self.assertNotIn(held["stamp"], stamps)
        self.assertNotIn(held["stamp"], {r.get("ros_stamp_ns") for r in self.sort_records(out)})
        self.assertTrue(any(st > held["stamp"] for st in stamps))   # 폐기 뒤 새 프레임은 다시 수락
        self.assertTrue(any(r["kind"] == "invalidation" and r["cause"] == "operator_discard:operator"
                            for r in self.records(out)))

    def test_clear_during_inflight_detection_then_old_frame_result(self):
        """검출 진행 중 폐기: CLEAR가 그 검출이 끝나기 전에 들어가 그 사진의 RESULT(표시용)는 CLEAR 뒤에 나가지만,
        폐기 전에 받은 그 사진은 추적·B에 넣지 않는다(고치기 전에는 B가 수락).

        C 5758ccf는 검출 대기 회차의 check()에서도 요청을 적용하므로 생기는 순서다(#397 결함 표 참조).
        """
        rt = DiscardRuntime()
        made = {}

        def during(sc):
            """검출 3 중: 폐기를 부르고 CLEAR가 들어갈 때까지 검출을 붙잡는다."""
            h = made["h"]
            rt.call_in_thread()
            end = time.monotonic() + LIMIT_S

            while time.monotonic() < end and not any(c[0] == "clear" and str(c[1]).startswith("operator_")
                                                     for c in h.calls):
                time.sleep(0.002)

            made["stamp"] = sc.stamp(3)

        sc = Scenario(TRACK_FRAMES, during={3: during})
        orig = self.mod._default_live_handoff

        def factory():
            """기록 연결부를 만들고 시험이 꺼낼 수 있게 둔다."""
            from eyetoarm.vision.c_handoff_probe import HandoffProbe

            probe = HandoffProbe()
            made["h"] = RecordingHandoff(notify=probe.notify)
            return made["h"], probe

        with mock.patch.object(self.mod, "_default_live_handoff", factory):
            code, err, out = self.run_sort(sc, config=MANAGED_CONFIG, extra=MANAGED_EXTRA, factory=self.factory(rt))

        rt.join()
        self.assertEqual(code, 0, err)
        self.assertEqual([r[:2] for r in rt.replies], [(True, "A/B detections discarded")])
        seq = [c[:2] for c in made["h"].calls if c[0] in ("clear", "result")]
        i_clear = seq.index(("clear", "operator_operator"))
        self.assertIn(("result", 3), seq[i_clear + 1:])   # 폐기 전에 찍힌 사진 3의 RESULT가 CLEAR 뒤
        self.assertNotIn(("result", 3), seq[:i_clear])
        stamps = [r["input"]["ros_stamp_ns"] for r in self.accepts(out)]
        self.assertNotIn(made["stamp"], stamps)   # 고친 뒤: 폐기 전에 받은 사진은 B에 넣지 않는다(고치기 전에는 수락됨)
        self.assertTrue(any(st > made["stamp"] for st in stamps))   # 폐기 뒤 새 사진은 다시 수락
        skips = [r for r in self.skipped(out) if r["reason"] == "operator_discard"]
        self.assertTrue(skips)

    def test_discard_after_run_end_fails_fast(self):
        """실행이 끝난 뒤의 폐기는 기다리지 않고 실패 응답(success=False)."""
        rt = DiscardRuntime()
        code, err, out, _ = self.run_with_handoff(Scenario(3), rt)
        self.assertEqual(code, 0, err)
        started = time.monotonic()
        success, message, _ = service_reply(rt.discard_listener)
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertFalse(success)



class MarkerPublisher:
    """`FakePublisher`(test_live_a_wiring)에 C가 부르던 `invalidate`(center marker)를 더한 기록용 발행기."""

    def __new__(cls, events, **kw):
        """FakePublisher를 감싸 invalidate 기록을 더한다."""
        from test_live_a_wiring import FakePublisher

        pub = FakePublisher(events, **kw)
        pub.invalidations = []
        pub.publish_times = []
        orig = pub.publish

        def publish(image, frame, keys=None):
            """발행 시각도 남긴다."""
            pub.publish_times.append(time.monotonic_ns())
            return orig(image, frame, keys)

        pub.publish = publish
        pub.invalidate = lambda run_id: pub.invalidations.append((run_id, time.monotonic_ns()))
        return pub


class CenterAfterDiscardTest(CliBase):
    """R9 (나, A 결정): 폐기 때 center 무효화 표지(marker)를 보내지 않고, 재개 뒤 center 발행이 이어진다."""

    def test_no_marker_and_center_continues_after_discard(self):
        """목록 입력 + center 발행 + 관리 송신: 실행 중 폐기 → 성공 응답, marker 0건, 폐기 뒤에도 center 발행이 이어진다."""
        from test_live_stage_l import three_bolts

        rt = DiscardRuntime()
        pubs = []
        seen = {}

        def during(sc):
            """검출 4 중: 폐기를 부르고 끝날 때까지 기다린 뒤 그 시각을 남긴다."""
            t = rt.call_in_thread()
            t.join(LIMIT_S * 2)
            seen["discard_done_ns"] = time.monotonic_ns()

        frames = 12
        sc = Scenario(frames, masks={i: three_bolts for i in range(frames)}, during={4: during})

        def pub_factory():
            """center 발행기 대역."""
            pub = MarkerPublisher([])
            pubs.append(pub)
            return pub

        real = self.mod.main

        def main(argv, **kw):
            """center 발행기 주입."""
            kw.setdefault("center_publisher_factory", pub_factory)
            return real(argv, **kw)

        with mock.patch.object(self.mod, "main", main):
            code, err, out = self.run_sort(sc, config=MANAGED_CONFIG,
                                           extra=("--b-input", "list", "--b-center-overlay", *MANAGED_EXTRA),
                                           factory=self.factory(rt))

        rt.join()
        self.assertEqual(code, 0, err)
        self.assertEqual([r[:2] for r in rt.replies], [(True, "A/B detections discarded")])
        pub = pubs[0]
        self.assertEqual(pub.invalidations, [])   # 폐기 때 marker를 보내지 않는다(고치기 전에는 1건)
        before = [t for t in pub.publish_times if t < seen["discard_done_ns"]]
        after = [t for t in pub.publish_times if t > seen["discard_done_ns"]]
        self.assertTrue(before)
        self.assertTrue(after, "center publishing must continue after the discard")

if __name__ == "__main__":
    unittest.main()
