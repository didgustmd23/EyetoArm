"""#406 ①: center 영상 구독자가 없으면 그리기·영상 발행을 건너뛰고, B 결과·`/b/results`·목표는 그대로인지 시험."""
from __future__ import annotations

import unittest
from unittest import mock

from eyetoarm.transforms.b_stage import BStage
from eyetoarm.vision.center_draw_gate import CenterDrawGate, gated_center_consumer, subscriber_total
from test_b_process_frame import bolt, envelope
from test_b_transforms import CAM, ROOT
from test_live_a_wiring import LIST, FakePublisher, WiringBase, no_lookup
from test_live_sort_part import FakeRuntime
from test_live_stage_l import three_bolts
from test_run_pipeline_live_observation import CONFIG, Scenario


class Clock:
    """시험이 움직이는 단조 시계(s)."""

    def __init__(self):
        """0에서 시작."""
        self.t = 0.0

    def __call__(self):
        """지금 시각."""
        return self.t


class CountingPublisher(FakePublisher):
    """구독자 수를 시험이 정하는 가짜 발행기. 결과 숫자 발행도 기록한다."""

    def __init__(self, events, subscribers=0, **kw):
        """구독자 수(영상 토픽)."""
        super().__init__(events, **kw)
        self.subscribers = subscribers
        self.results = []

    def subscriber_count(self):
        """영상 토픽 구독자 수."""
        return self.subscribers

    def publish_results(self, payload):
        """`/b/results` 기록."""
        self.results.append(payload)


def scene():
    """볼트 둘 봉투."""
    return envelope([bolt(0, 0.55, -0.12), bolt(2, 0.56, -0.05)])


def goal_view(rec):
    """목표 비교용(goal_id·시각처럼 매번 바뀌는 값 제외)."""
    g = rec["goal"]
    return None if g is None else (g["kind"], g["spec"], g["observed_position_mm"], g["part_track_id"])


class GateTest(unittest.TestCase):
    """문 자체: 구독자 수·확인 간격·늘 그림·셀 수 없음."""

    def test_counts_both_topics(self):
        """영상 토픽(`subscriber_count`)과 번호표 토픽(`_overlay_pub`)을 더한다. 둘 다 셀 수 없으면 None."""
        pub = CountingPublisher([], subscribers=0)
        pub._overlay_pub = mock.Mock(get_subscription_count=mock.Mock(return_value=1))
        self.assertEqual(subscriber_total(pub), 1)
        self.assertIsNone(subscriber_total(object()))

    def test_rechecks_only_after_interval(self):
        """구독자 수는 check_s마다 한 번만 묻고, 0 → 1이 되면 다음 확인부터 그린다. 전환 시각을 남긴다."""
        clock, pub = Clock(), CountingPublisher([], subscribers=0)
        gate = CenterDrawGate(pub, clock=clock, check_s=0.1)
        self.assertFalse(gate.wanted())
        pub.subscribers = 1
        clock.t = 0.05
        self.assertFalse(gate.wanted())   # 아직 간격 안
        clock.t = 0.11
        self.assertTrue(gate.wanted())
        self.assertEqual([w for _, w in gate.transitions], [False, True])

    def test_always_and_unknown_draw(self):
        """늘 그림 옵션, 셀 수 없음(None), 세기 예외는 모두 그린다(예외는 센다)."""
        self.assertTrue(CenterDrawGate(CountingPublisher([], 0), always=True).wanted())
        self.assertTrue(CenterDrawGate(object()).wanted())
        bad = CenterDrawGate(object(), count=mock.Mock(side_effect=RuntimeError("down")))
        self.assertTrue(bad.wanted())
        self.assertEqual(bad.count_errors, 1)


class ConsumerTest(unittest.TestCase):
    """감싸개: 실제 B 단계로 그릴 때·건너뛸 때 비교."""

    def setUp(self):
        """실제 B 단계 둘(그림·건너뜀 비교용)."""
        self.draw_stage = BStage.from_config(ROOT, CAM, no_lookup)
        self.skip_stage = BStage.from_config(ROOT, CAM, no_lookup)

    def test_no_subscriber_skips_render_but_keeps_results(self):
        """구독자 0: render_center·영상 발행 0, `/b/results`·반환 사전·목표는 그릴 때와 같다."""
        on, off = CountingPublisher([], subscribers=1), CountingPublisher([], subscribers=0)
        full = gated_center_consumer(self.draw_stage, on, CenterDrawGate(on))
        gate = CenterDrawGate(off)
        skip = gated_center_consumer(self.skip_stage, off, gate)

        with mock.patch.object(self.skip_stage, "render_center", wraps=self.skip_stage.render_center) as render:
            rec_skip = skip(scene())

        rec_full = full(scene())
        render.assert_not_called()
        self.assertEqual((off.published, on.published), (0, 1))
        self.assertEqual((gate.drawn, gate.skipped), (0, 1))
        self.assertEqual(goal_view(rec_skip), goal_view(rec_full))
        self.assertEqual(rec_skip["selected_instance_index"], rec_full["selected_instance_index"])
        strip = lambda p: [{k: v for k, v in r.items()} for r in p["results"]]  # noqa: E731
        self.assertEqual(strip(off.results[0]), strip(on.results[0]))

    def test_subscriber_appears_then_draws(self):
        """구독자가 생기면(간격 지난 뒤) 다음 건부터 그려 발행한다."""
        clock, pub = Clock(), CountingPublisher([], subscribers=0)
        gate = CenterDrawGate(pub, clock=clock, check_s=0.1)
        consume = gated_center_consumer(self.skip_stage, pub, gate)
        consume(scene())
        pub.subscribers = 1
        clock.t = 0.2
        consume(scene())
        self.assertEqual((pub.published, len(pub.results)), (1, 2))

    def test_results_publish_failure_does_not_break_b(self):
        """건너뛸 때 결과 발행이 실패해도 B 반환은 그대로이고 오류만 센다."""
        pub = CountingPublisher([], subscribers=0)
        pub.publish_results = mock.Mock(side_effect=RuntimeError("down"))
        errors = []
        consume = gated_center_consumer(self.skip_stage, pub, CenterDrawGate(pub), on_error=errors.append)
        rec = consume(scene())
        self.assertIsNotNone(rec["goal"])
        self.assertEqual((consume.errors, len(errors)), (1, 1))


class CliGateTest(WiringBase):
    """진입점: center 발행 + 목록 송신 실행에서 구독자 0이면 그리지 않고, 목표·수락은 늘 그림과 같다."""

    def run_with(self, subscribers, *extra):
        """구독자 수를 정한 발행기로 한 번 실행 → (종료 코드, stderr, 출력 폴더, 발행기, 송신 런타임)."""
        self.pub_kw = {}
        made = []

        def factory():
            """구독자 수를 정한 발행기."""
            pub = CountingPublisher(self.events, subscribers=subscribers)
            made.append(pub)
            return pub

        self.publisher_factory = factory
        rt = FakeRuntime()
        sc = Scenario(4, masks={i: three_bolts for i in range(4)})
        code, err, out = self.run_sort(sc, factory=self.factory(rt), out=self.out(f"live{len(self.handoffs)}"),
                                       extra=(*LIST, "--b-center-overlay", "--live-sort-part", *extra))
        return code, err, out, made[0], rt

    def test_zero_subscribers_skip_draw_same_goals(self):
        """구독자 0: 영상 발행 0, 결과 숫자는 수락마다, 보낸 목표는 늘 그림 실행과 같다. 시작 줄·요약에 문 상태."""
        code, err, out, pub, rt = self.run_with(0)
        code2, err2, out2, pub2, rt2 = self.run_with(0, "--b-center-always-draw")
        self.assertEqual((code, code2), (0, 0), err + err2)
        self.assertEqual(pub.published, 0)
        self.assertGreater(pub2.published, 0)
        self.assertEqual(len(pub.results), len(pub2.results))
        view = lambda r: [(g["kind"], g["spec"], g["observed_position_mm"]) for g in r.sent]  # noqa: E731
        self.assertEqual(view(rt), view(rt2))
        self.assertEqual(len(self.accepts(out)), len(self.accepts(out2)))
        self.assertIn("draw=only while a center subscriber exists (#406)", err)
        self.assertIn("draw=always", err2)
        gate = self.summary(out)["tracking"]["center_overlay"]["draw_gate"] if "tracking" in self.summary(out) \
            else self.summary(out)["center_overlay"]["draw_gate"]
        self.assertEqual((gate["always"], gate["drawn"]), (False, 0))
        self.assertGreater(gate["skipped"], 0)

    def test_subscriber_present_draws(self):
        """구독자 1: 사진마다 그려 발행한다(예전과 같음)."""
        code, err, out, pub, _ = self.run_with(1)
        self.assertEqual(code, 0, err)
        self.assertEqual(pub.published, len(pub.results))
        self.assertGreater(pub.published, 0)


if __name__ == "__main__":
    unittest.main()
