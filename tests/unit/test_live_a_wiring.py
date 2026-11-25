"""실시간 A→B 실행 하나로 detect(C 연결부)·B center 발행·목록 경로 목표 송신을 함께 잇는 시험.

`main()`부터 가짜 수신 → 실행기 → `AiSegDetector`(가짜 모델) → 연결부 → 가짜 TF → 실제 B로 돈다.
C 연결부는 실제 `AToCHandoff` + 가짜 probe, center 발행기·C 송신 런타임은 가짜다. ROS·Gazebo 없음.
"""
from __future__ import annotations

import io
import unittest
from pathlib import Path
from unittest import mock

from eyetoarm.transforms import b_stage as b_stage_mod
from eyetoarm.transforms.b_stage import BStage
from eyetoarm.transforms.center_overlay_pub import center_overlay_consumer
from eyetoarm.vision import live_observation_link as link_mod
from eyetoarm.vision.handoff import AToCHandoff
from test_b_process_frame import bolt, envelope
from test_b_transforms import CAM, ROOT
from test_live_sort_part import CliBase, FakeRuntime
from test_live_stage_l import LIST_ACCEPT_KEYS, LIST_SUMMARY_KEYS, three_bolts
from test_ros_image_adapter import info_msg
from test_run_pipeline_capture_observation import FRAME_ID, H, W
from test_run_pipeline_live_observation import CONFIG, FakeTf, Scenario

LIST = ("--b-input", "list")


def no_lookup(frame):
    """목록 입구는 봉투 TF만 쓴다(불리면 실패)."""
    raise AssertionError("process_frame must use the envelope TF")


class FakeProbe:
    """가짜 C probe(네트워크 없음): 시작·정지만 기록한다."""

    def __init__(self, events):
        """공유 사건 기록."""
        self.events = events
        self.started = None

    def notify(self):
        """연결부 알림 자리(아무것도 안 함)."""

    def start(self, handoff):
        """시작 기록."""
        self.started = handoff
        self.events.append("probe_start")

    def stop(self):
        """정지 기록."""
        self.events.append("probe_stop")


class RecordingHandoff(AToCHandoff):
    """실제 C 연결부 + 닫기·결과 제공 기록."""

    def __init__(self, events, **kw):
        """공유 사건 기록."""
        super().__init__(**kw)
        self.events = events
        self.offered_seqs = []

    def offer_result(self, outcome, **kw):
        """실행기 제어 흐름의 제공 순서를 기록한다."""
        self.events.append(("handoff_offer", outcome.context.frame.image_seq))
        self.offered_seqs.append(outcome.context.frame.image_seq)

        return super().offer_result(outcome, **kw)

    def close(self):
        """닫기 기록."""
        self.events.append("handoff_close")
        super().close()


class FakePublisher:
    """가짜 B center 발행기(`CenterOverlayPublisher`와 같은 공개 면)."""

    def __init__(self, events, *, fail=False, overlay_topic="/b/center_overlay"):
        """공유 사건 기록·실패 주입·번호표 토픽(None = 인터페이스 없음)."""
        self.events = events
        self.fail = fail
        self.topic = "/b/center_overlay/image"
        self.overlay_topic = overlay_topic
        self.published = 0
        self.frames = []
        self.closed = 0

    def publish(self, image, frame, keys=None):
        """발행 기록(실패 주입)."""
        if self.fail:
            raise RuntimeError("publish down")

        self.published += 1
        self.frames.append((frame.image_seq, keys))

    def close(self):
        """닫기 기록."""
        self.closed += 1
        self.events.append("center_close")


class NullCenterProcess:
    """center 그리기 자식 대역(#413 r3, 기본 켜짐): 프로세스를 띄우지 않고 요청을 받아 버린다."""

    def __init__(self, root, *, on_failed=None):
        """띄우지 않는다."""
        self.pid = None
        self.jobs = 0

    def start(self):
        """아무것도 안 함."""

    def configure(self, camera):
        """아무것도 안 함."""

    def submit(self, *args, **kwargs):
        """요청을 받은 것으로 센다."""
        self.jobs += 1
        return True

    def summary(self):
        """요약."""
        return {"requested": self.jobs, "failed": None}

    def close(self):
        """정상 종료."""
        return {"returncode": 0, "forced": False}


class OrderTf(FakeTf):
    """TF 닫기(연결부 정리 마지막 쪽 단계)를 공유 사건 기록에 남긴다."""

    def __init__(self, events, **kw):
        """공유 사건 기록."""
        super().__init__(**kw)
        self.events = events

    def close(self, limit_s):
        """닫기 시도 기록 뒤 원래 동작."""
        self.events.append("tf_close")
        super().close(limit_s)


class FindChosenRealStageTest(unittest.TestCase):
    """고른 결과 찾기(`o.goal is rec['goal']`)가 실제 B `process_frame` 반환과 맞는지."""

    def setUp(self):
        """실제 B 단계."""
        self.stage = BStage.from_config(ROOT, CAM, no_lookup)

    def test_finds_selected_output_of_real_stage(self):
        """볼트 둘 → B가 고른 instance 2의 결과를 찾고, 그 goal_id가 반환 사전 goal_id와 같다."""
        rec = self.stage.process_frame(envelope([bolt(0, 0.55, -0.12), bolt(2, 0.56, -0.05)]))
        chosen = link_mod.LiveObservationLink._find_chosen(self.stage, rec)

        self.assertEqual(rec["selected_instance_index"], 2)
        self.assertIn(chosen, self.stage.last_frame)
        self.assertIs(chosen.goal, rec["goal"])
        self.assertEqual(chosen.goal["goal_id"], rec["goal"]["goal_id"])
        self.assertEqual(chosen.goal["kind"], "PICK_PLACE")

    def test_center_wrapper_returns_same_dict_so_chosen_found(self):
        """B center 감싸개를 거쳐도 반환 사전이 같아 고른 결과를 찾는다."""
        consume = center_overlay_consumer(self.stage, FakePublisher([]))
        rec = consume(envelope([bolt(0, 0.55, -0.12), bolt(2, 0.56, -0.05)]))
        chosen = link_mod.LiveObservationLink._find_chosen(self.stage, rec)

        self.assertIs(chosen.goal, rec["goal"])

    def test_no_selection_ignores_stale_stage_last(self):
        """앞 프레임에서 고른 뒤 고른 것이 없는 프레임: `stage.last`는 남아도 찾기 결과는 None."""
        self.stage.process_frame(envelope([bolt(0, 0.55, -0.12)]))
        rec = self.stage.process_frame(envelope([bolt(0, 0.55, -0.12)], with_tf=False))

        self.assertIsNone(rec["goal"])
        self.assertIsNotNone(self.stage.last)
        self.assertIsNone(link_mod.LiveObservationLink._find_chosen(self.stage, rec))

    def test_goal_not_in_results_is_unmatched(self):
        """반환 goal이 결과 묶음에 없으면(계약 위반) 못 찾음 표시."""
        self.stage.process_frame(envelope([bolt(0, 0.55, -0.12)]))
        rec = {"goal": {"kind": "PICK_PLACE", "goal_id": "other"}}

        self.assertIs(link_mod.LiveObservationLink._find_chosen(self.stage, rec), link_mod._UNMATCHED)


class WiringBase(CliBase):
    """CLI 공통: 실제 C 연결부 + 가짜 probe·발행기·송신 런타임 주입."""

    def setUp(self):
        """사건 기록·주입 대역."""
        super().setUp()
        self.events = []
        self.handoffs = []
        self.probes = []
        self.pubs = []
        self.pub_kw = {}
        self.pub_error = None
        self.extra_kw = {}
        real = self.mod.main

        def main(argv, **kw):
            """C 연결부·발행기 주입을 더한다."""
            kw.setdefault("live_handoff_factory", self.handoff_factory)
            kw.setdefault("center_publisher_factory", self.publisher_factory)
            kw.setdefault("center_process_factory", NullCenterProcess)   # 단위시험에서 실제 자식(ROS)을 띄우지 않는다(#413 r3)
            kw.update(self.extra_kw)

            return real(argv, **kw)

        self.mod.main = main

    def handoff_factory(self):
        """`live_handoff_factory()` 대역: (실제 연결부, 가짜 probe)."""
        probe = FakeProbe(self.events)
        handoff = RecordingHandoff(self.events, notify=probe.notify)
        self.handoffs.append(handoff)
        self.probes.append(probe)

        return handoff, probe

    def publisher_factory(self):
        """`center_publisher_factory()` 대역."""
        if self.pub_error is not None:
            raise self.pub_error

        pub = FakePublisher(self.events, **self.pub_kw)
        self.pubs.append(pub)

        return pub


class AllOnTest(WiringBase):
    """한 실행에서 C 연결부·center 감싸개·목록 송신이 함께 만들어지고 동작한다."""

    def test_one_run_handoff_center_and_send(self):
        """볼트 3개 × 4장: handoff 제공·center 발행·고른 1개 송신이 한 main() 호출에서 모두 일어난다."""
        rt = FakeRuntime()
        sc = Scenario(4, masks={i: three_bolts for i in range(4)})
        # 같은 프로세스 발행기 경로를 본다(#413 r3부터 기본은 별도 프로세스라 끄는 옵션을 준다)
        code, err, out = self.run_sort(sc, factory=self.factory(rt), extra=(*LIST, "--b-center-overlay", "--no-b-center-process", "--live-sort-part"))

        self.assertEqual(code, 0, err)
        self.assertEqual((len(self.handoffs), len(self.pubs), len(self.factory_calls)), (1, 1, 1))
        self.assertIs(self.probes[0].started, self.handoffs[0])
        self.assertIn("consumer=center_overlay(BStage.process_frame)", err)
        addr = self.mod._handoff_address()   # 설정된 주소
        self.assertIn(f"outputs=c_handoff_detect({addr}),b_center_overlay(/b/center_overlay/image),"
                      "sort_part(C goals via /sort_part, accepted results only);", err)
        self.assertIn("B CENTER OVERLAY ON topic=/b/center_overlay/image overlay=/b/center_overlay", err)

        acc = self.accepts(out)
        s = self.summary(out)
        self.assertEqual(len(acc), 3)
        self.assertEqual((s["record_only"], s["outputs"]),
                         (False, [f"c_handoff_detect({addr})", "b_center_overlay(/b/center_overlay/image)",
                                  "sort_part(C goals via /sort_part, accepted results only)"]))

        # 실행기 → C 연결부: 결과 제공·실행 종료 보고
        self.assertTrue(self.handoffs[0].offered_seqs)
        self.assertEqual(s["handoff"]["state"], "open")
        self.assertTrue(s["handoff"]["bound"])
        self.assertTrue(s["handoff"]["run_end_sent"])

        # center: B 계산마다 1장(검사 ③ 전 발행), 번호표는 그 사진 값
        self.assertEqual(self.pubs[0].published, s["list_consumer_calls"])
        self.assertEqual([seq for seq, _ in self.pubs[0].frames], [r["input"]["image_seq"] for r in acc])
        self.assertTrue(all(k["image_seq"] == seq for seq, k in self.pubs[0].frames))
        self.assertEqual(s["center_overlay"]["published"], 3)
        self.assertEqual(s["center_overlay"]["errors"], 0)

        # 목표: 수락 프레임마다 B가 고른 1개, goal_id가 accept 기록의 B 목표와 같다
        self.assertEqual([g["goal_id"] for g in rt.sent], [r["consumer_result"]["goal"]["goal_id"] for r in acc])
        sent = [r for r in self.sort_records(out) if r["event"] == "sent"]
        self.assertEqual(sorted(r["accept_event_seq"] for r in sent), sorted(r["event_seq"] for r in acc))
        self.assertEqual((s["list_goals_offered"], s["list_no_selection"], s["list_goal_unmatched"]), (3, 0, 0))
        self.assertIn("center_published=3 center_errors=0", err)

    def test_runner_offers_handoff_before_link_for_same_frame(self):
        """두 sink를 함께 받은 실행기: 같은 프레임은 C 연결부 제공이 연결부 offer보다 먼저다."""
        real_offer = link_mod.LiveObservationLink.offer

        def offer(link, outcome, *a, **kw):
            """연결부 offer 순서를 기록한다."""
            self.events.append(("link_offer", outcome.context.frame.image_seq))

            return real_offer(link, outcome, *a, **kw)

        sc = Scenario(3, masks={i: three_bolts for i in range(3)})

        with mock.patch.object(link_mod.LiveObservationLink, "offer", offer):
            code, err, out = self.run_sort(sc, config=CONFIG, extra=LIST)

        self.assertEqual(code, 0, err)
        seqs = [e for e in self.events if isinstance(e, tuple)]
        offered = {seq for kind, seq in seqs if kind == "handoff_offer"}
        self.assertTrue(offered)

        for seq in offered:
            self.assertLess(seqs.index(("handoff_offer", seq)), seqs.index(("link_offer", seq)))

    def test_profile_change_rewraps_new_stage_with_one_publisher(self):
        """CameraInfo 변경으로 새 B 단계가 생겨도 감싸개가 다시 붙고 발행기는 하나다."""
        stages = []

        def change(s):
            """보정 번호가 바뀌는 CameraInfo."""
            s.src.handle_camera_info(info_msg(fx=310.0, width=W, height=H, frame_id=FRAME_ID))

        sc = Scenario(6, masks={i: three_bolts for i in range(6)}, during={2: change})
        code, err, out = self.run_sort(sc, config=CONFIG, extra=(*LIST, "--b-center-overlay", "--no-b-center-process"),
                                       hook=stages.append)   # 같은 프로세스 경로(#413 r3)

        self.assertEqual(code, 0, err)
        self.assertEqual(len(self.pubs), 1)
        self.assertEqual(len(stages), 2)
        link = self.controller.current
        self.assertEqual(len(link.stages), 2)
        self.assertTrue(all(st[2].__name__ == "consume" for st in link.stages))   # B 감싸개
        self.assertEqual(self.pubs[0].published, link.list_consumer_calls)
        self.assertGreater(len({k["calibration_id"] for _, k in self.pubs[0].frames}), 1)


class SelectionTest(WiringBase):
    """사진당 송신 1회·고른 것 없는 프레임."""

    def test_no_selection_frames_send_nothing(self):
        """B가 고르지 않으면(선택 없음) 수락돼도 송신 0, `list_no_selection`만 센다."""
        def hook(stage):
            """실제 B 단계의 고르기만 「없음」으로 바꾼다(계산·기록은 실제)."""
            stage._select = lambda outs, cands: (None, "NONE")

        rt = FakeRuntime()
        sc = Scenario(4, masks={i: three_bolts for i in range(4)})
        code, err, out = self.run_sort(sc, factory=self.factory(rt), extra=(*LIST, "--live-sort-part"), hook=hook)

        self.assertEqual(code, 0, err)
        s = self.summary(out)
        self.assertEqual(rt.sent, [])
        self.assertEqual((s["list_goals_offered"], s["list_no_selection"]), (0, len(self.accepts(out))))
        self.assertFalse([r for r in self.sort_records(out) if r["event"] == "sent" or r["event"].startswith("skip:")])

    def test_stale_stage_last_not_resent(self):
        """첫 계산만 고르고 다음은 고르지 않음: `stage.last`가 남아도 송신은 첫 프레임 1회뿐이다."""
        def hook(stage):
            """두 번째 계산부터 고르기 없음."""
            real, calls = stage._select, []

            def select(outs, cands):
                """첫 호출만 실제 고르기."""
                calls.append(1)

                return real(outs, cands) if len(calls) == 1 else (None, "NONE")

            stage._select = select

        rt = FakeRuntime()
        sc = Scenario(4, masks={i: three_bolts for i in range(4)})
        code, err, out = self.run_sort(sc, factory=self.factory(rt), extra=(*LIST, "--live-sort-part"), hook=hook)

        self.assertEqual(code, 0, err)
        acc = self.accepts(out)
        self.assertEqual(len(rt.sent), 1)
        self.assertEqual(rt.sent[0]["goal_id"], acc[0]["consumer_result"]["goal"]["goal_id"])
        self.assertEqual(self.summary(out)["list_no_selection"], len(acc) - 1)

    def test_one_send_per_frame_even_with_busy(self):
        """같은 종류가 결과 대기면 busy: 프레임마다 시도 1회, 보낸 것은 1개."""
        rt = FakeRuntime(busy_after=1)
        sc = Scenario(4, masks={i: three_bolts for i in range(4)})
        code, err, out = self.run_sort(sc, factory=self.factory(rt), extra=(*LIST, "--live-sort-part"))

        self.assertEqual(code, 0, err)
        acc = self.accepts(out)
        tried = [r for r in self.sort_records(out) if r["event"] in ("sent", "busy")]
        self.assertEqual(len(rt.sent), 1)
        self.assertEqual(sorted(r["accept_event_seq"] for r in tried), sorted(r["event_seq"] for r in acc))


class OffSameTest(WiringBase):
    """옵션을 끄면 단계 L 기록·요약이 지금과 같다(C 연결부만 항상)."""

    def test_list_without_options_unchanged(self):
        """center·송신 없음: 발행기 공장 0, 소비자 이름·accept 키·요약 목록 키가 기존 그대로."""
        sc = Scenario(3, masks={i: three_bolts for i in range(3)})
        code, err, out = self.run_sort(sc, config=CONFIG, extra=LIST)

        self.assertEqual(code, 0, err)
        self.assertEqual((self.pubs, self.factory_calls), ([], []))
        self.assertIn("consumer=BStage.process_frame", err)
        self.assertIn(f"outputs=c_handoff_detect({self.mod._handoff_address()});", err)   # 설정된 주소
        self.assertNotIn("CENTER OVERLAY ON", err)
        self.assertNotIn("center_published", err)
        s = self.summary(out)
        self.assertNotIn("center_overlay", s)
        self.assertEqual({k for k in s if k.startswith("list_") or k in ("b_input", "consumer_kind")}, LIST_SUMMARY_KEYS)

        for r in self.accepts(out):
            self.assertEqual(list(r), LIST_ACCEPT_KEYS)


class CleanupOrderTest(WiringBase):
    """정리 순서: 연결부 정리(TF 닫기 포함) → center 발행기 → probe 정지 → C 연결부 닫기."""

    def test_order(self):
        """정상 종료 사건 순서."""
        sc = Scenario(3, masks={i: three_bolts for i in range(3)})
        code, err, out = self.run_sort(sc, config=CONFIG, extra=(*LIST, "--b-center-overlay"), tf=OrderTf(self.events))

        self.assertEqual(code, 0, err)
        names = [e for e in self.events if isinstance(e, str)]
        self.assertEqual(names[0], "probe_start")
        self.assertEqual(names[-4:], ["tf_close", "center_close", "probe_stop", "handoff_close"])

    def test_unconfirmed_cleanup_leaves_publisher_open_exit_1(self):
        """연결부 정리 미확인(TF 닫기 실패): 발행기는 열어 두고 종료 1, C 연결부는 닫는다."""
        sc = Scenario(3, masks={i: three_bolts for i in range(3)})
        code, err, out = self.run_sort(sc, config=CONFIG, extra=(*LIST, "--b-center-overlay"),
                                       tf=OrderTf(self.events, fail_close=10))

        self.assertEqual(code, 1, err)
        self.assertIn("center publisher left open (dispatcher not confirmed stopped)", err)
        self.assertEqual(self.pubs[0].closed, 0)
        self.assertEqual([e for e in self.events if isinstance(e, str)][-2:], ["probe_stop", "handoff_close"])


class CenterIsolationTest(WiringBase):
    """center 발행 실패·인터페이스 없음 표시."""

    def test_publish_failure_does_not_change_exit_or_b(self):
        """발행 예외: 종료 0, B 수락·기록 그대로, 요약·마지막 줄에 오류 수."""
        self.pub_kw = {"fail": True}
        sc = Scenario(3, masks={i: three_bolts for i in range(3)})
        code, err, out = self.run_sort(sc, config=CONFIG, extra=(*LIST, "--b-center-overlay", "--no-b-center-process"))   # 같은 프로세스 경로(#413 r3)

        self.assertEqual(code, 0, err)
        s = self.summary(out)
        acc = self.accepts(out)
        self.assertEqual(len(acc), 2)
        self.assertEqual(s["center_overlay"]["errors"], s["list_consumer_calls"])
        self.assertEqual(s["center_overlay"]["first_error"], "RuntimeError: publish down")
        self.assertEqual(s["problems"], [])
        self.assertIn(f"center_errors={s['list_consumer_calls']}", err)

    def test_no_overlay_interface_shown(self):
        """번호표 토픽 없음(인터페이스 미빌드): 시작 줄에 Image만·재빌드 안내."""
        self.pub_kw = {"overlay_topic": None}
        sc = Scenario(2, masks={i: three_bolts for i in range(2)})
        code, err, out = self.run_sort(sc, config=CONFIG, extra=(*LIST, "--b-center-overlay"))

        self.assertEqual(code, 0, err)
        self.assertIn("overlay=none (eyetoarm_interfaces CenterOverlay not found: rebuild robot_ws; Image only)", err)


class ArgsTest(WiringBase):
    """인자·시작 거부."""

    def test_center_argument_errors(self):
        """단계 T(기본·명시)·기록 연결 없음은 2, TF·발행기·C 연결부 0."""
        for extra in (("--b-center-overlay",), ("--b-input", "single", "--b-center-overlay")):
            with self.subTest(extra=extra):
                tf = FakeTf()
                code, err, out = self.run_sort(Scenario(2), config=CONFIG, extra=extra, tf=tf)
                self.assertEqual(code, 2, err)
                self.assertEqual((tf.calls, self.pubs, self.handoffs), ([], [], []))

        err = io.StringIO()

        try:
            code = self.mod.main(["--ros-link", "--detector", "ai", "--ai-config", self.case.ai, "--b-center-overlay"],
                                 stderr=err, ros_available=lambda: True)
        except SystemExit as exc:
            code = exc.code

        self.assertEqual(code, 2)

    def test_publisher_start_failure_exit_1_before_tf_and_output(self):
        """발행기 시작 실패(ImportError): 종료 1, TF·출력 폴더·C 연결부 없음."""
        self.pub_error = ImportError("No module named 'sensor_msgs'")
        tf = FakeTf()
        code, err, out = self.run_sort(Scenario(2), config=CONFIG, extra=(*LIST, "--b-center-overlay"), tf=tf)

        self.assertEqual(code, 1, err)
        self.assertIn("ROS image publisher unavailable", err)
        self.assertFalse(Path(out).exists())
        self.assertEqual((tf.calls, self.handoffs), ([], []))

    def test_center_with_injected_list_consumer_refused(self):
        """시험 주입 목록 소비자와 center를 함께 주면 2."""
        self.extra_kw = {"list_consumer_factory": lambda stage: stage.process_frame}
        code, err, out = self.run_sort(Scenario(2), config=CONFIG, extra=(*LIST, "--b-center-overlay"))

        self.assertEqual(code, 2, err)
        self.assertIn("cannot be combined with an injected list consumer", err)
        self.assertEqual(self.pubs, [])


if __name__ == "__main__":
    unittest.main()
