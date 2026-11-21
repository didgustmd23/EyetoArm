"""#201 실시간 기록 전용 경로의 단계 L(다중 볼트 목록) 연결 시험.

근거: 지시 `_workspace/201_reviewer_instruction_live_stage_L.md`, 계획 `_workspace/201_builder2_live_stage_L_plan.md` r0과 그 검토
(`_workspace/201_review/201_reviewer_plan_r0_review.md`: Q1 주입·Q2 봉투 버전 키·b_list_skip_reason 차단 조건·D13 고정).

- 진입점 `main()`부터 돈다: 주입 `LiveLinkSource`(가짜 수신) → 실행기 → `AiSegDetector`(가짜 모델, 실제 convert·M1) → 코어 →
  `b_list_skip_reason` → `offer` → 디스패처 → 주입 TF(촬영 stamp 1회) → 목록 봉투 2 → **가짜 목록 소비자(시험 전용, 실제 B 아님)**.
- 실제 ROS·Gazebo는 쓰지 않는다. C 송신 감시는 `test_run_pipeline_live_observation.Base` 그대로다.
"""
from __future__ import annotations

import functools
import io
import json
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from eyetoarm.transforms import b_stage as b_stage_mod
from eyetoarm.vision import live_observation_link as link_mod
from eyetoarm.vision.list_envelope import LIST_ENVELOPE_VERSION, ListEnvelope, ListEnvelopeError
from eyetoarm.vision.pipeline_runner import b_list_skip_reason, b_skip_reason
from test_ai_axis_candidates import SCENARIOS, outcome_of
from test_ros_image_adapter import info_msg, stamp
from test_run_pipeline_capture_observation import FRAME_ID, H, W, bolt_mask, rect_mask
from test_run_pipeline_live_observation import Base, FakeTf, Scenario

SINGLE_ACCEPT_KEYS = ["event_seq", "kind", "token_seq", "enqueued_monotonic_ns", "cause", "compute_id", "input", "b_input",
                      "goal_kind", "b_reason", "b_valid", "result_dict", "goal", "tf_status", "tf_wait_ms", "k5_ms",
                      "annotation"]
LIST_ACCEPT_KEYS = ["event_seq", "kind", "token_seq", "enqueued_monotonic_ns", "cause", "compute_id", "input", "b_input",
                    "list_indices", "list_candidate_count", "consumer_result", "tf_status", "tf_wait_ms", "k5_ms",
                    "annotation"]
LIST_SUMMARY_KEYS = {"b_input", "consumer_kind", "list_consumer_calls", "list_candidates_sent", "list_length_hist"}


def three_bolts():
    """서로 떨어진 볼트 3개(위·가운데·아래)."""
    return [bolt_mask(y=50), bolt_mask(y=120), bolt_mask(y=190)]


class FakeListConsumer:
    """시험 전용 가짜 B 목록 소비자(실제 B 아님): 받은 봉투를 보관하고 후보별 결과·선택 번호를 돌려준다."""

    def __init__(self):
        """받은 봉투·호출 스레드."""
        self.envelopes = []

    def factory(self, stage):
        """`list_consumer_factory(stage)` 자리. stage의 `on_output`이 None인지 같이 본다."""
        assert stage.on_output is None

        return self

    def __call__(self, envelope):
        """봉투 하나 → 후보별 결과(원 인덱스 그대로)와 첫 후보 선택."""
        assert isinstance(envelope, ListEnvelope)
        self.envelopes.append(envelope)
        idx = [c.instance_index for c in envelope.observations.candidates]

        return {"per_candidate": [{"instance_index": i, "valid": True} for i in idx], "selected_index": idx[0]}

    def stamps(self):
        """소비한 프레임의 촬영 stamp."""
        return [e.observations.frame.ros_stamp_ns for e in self.envelopes]


class ListBase(Base):
    """`--b-input list` + 가짜 목록 소비자 주입(`main(list_consumer_factory=)`)."""

    def setUp(self):
        """Base 준비 + main에 소비자 공장 주입."""
        super().setUp()
        self.consumer = FakeListConsumer()
        self.inject = True
        real = self.mod.main

        def main(argv, **kw):
            """주입 여부에 따라 list_consumer_factory를 넘긴다."""
            if self.inject:
                kw["list_consumer_factory"] = self.consumer.factory

            return real(argv, **kw)

        self.mod.main = main

    def run_list(self, scenario, **kw):
        """`--b-input list`로 실행."""
        extra = ("--b-input", "list", *kw.pop("extra", ()))

        return self.run_cli(scenario, extra=extra, **kw)

    def accepts(self, out):
        """수락 기록."""
        return [r for r in self.records(out) if r["kind"] == "accept"]

    def skip_reasons(self, out, stamp_ns):
        """그 stamp 프레임의 건너뜀 사유들."""
        return [r["reason"] for r in self.skipped(out) if r["ros_stamp_ns"] == stamp_ns]

    def run_log(self, out):
        """실행기 기록(JSONL)."""
        p = Path(self.tmp.name) / f"{Path(out).name}.jsonl"

        return [json.loads(t) for t in p.read_text(encoding="utf-8").splitlines()]


class MultiBoltListTest(ListBase):
    """복수 볼트: 단계 T가 MULTIPLE_OBJECTS로 무효여도 목록은 촬영 stamp TF 하나로 넘어간다."""

    def test_three_bolts_sent_as_one_list(self):
        """4장 × 볼트 3: 열기 프레임 뒤 3장이 목록 3개로 소비되고, 기록·요약·실행기 기록이 맞는다."""
        sc = Scenario(4, masks={i: three_bolts for i in range(4)})
        code, err, out = self.run_list(sc)

        self.assertEqual(code, 0, err)
        self.assertIn("b_input=eyetoarm.b_input/2 consumer=injected_test_only", err)
        self.assertEqual(self.c_calls, [])
        self.assertEqual(self.consume_calls, [])            # 단일 관측 B 입구는 불리지 않음
        self.assertEqual(self.consumer.stamps(), [sc.stamp(1), sc.stamp(2), sc.stamp(3)])

        acc = self.accepts(out)
        self.assertEqual([r["input"]["ros_stamp_ns"] for r in acc], [sc.stamp(1), sc.stamp(2), sc.stamp(3)])

        for r in acc:
            s_ns = r["input"]["ros_stamp_ns"]
            self.assertEqual(list(r), LIST_ACCEPT_KEYS)
            self.assertEqual(r["b_input"]["version"], LIST_ENVELOPE_VERSION)
            self.assertEqual((r["list_indices"], r["list_candidate_count"]), ([0, 1, 2], 3))
            self.assertEqual(r["consumer_result"]["selected_index"], 0)
            self.assertEqual([c["instance_index"] for c in r["consumer_result"]["per_candidate"]], [0, 1, 2])
            self.assertEqual((r["b_input"]["tf"]["stamp_ns"], r["b_input"]["capture"]["stamp_ns"]), (s_ns, s_ns))
            self.assertEqual(r["tf_status"], "ok")
            self.assertIsNone(r["annotation"])
            self.assertEqual([c[2] for c in self.tf.calls].count(s_ns), 1)   # 프레임 공통 TF 조회 1회

        self.assertEqual(self.skip_reasons(out, sc.stamp(0)), ["not_ready"])

        # 단계 T 코어 관측은 그대로 MULTIPLE_OBJECTS(무효), 실행기 기록에 목록 키
        done = [r for r in self.run_log(out) if r["kind"] == "process"]
        self.assertEqual(len(done), 4)

        for r in done:
            self.assertEqual((r["valid"], r["reason"], r["list_indices"], r["list_candidate_count"]),
                             (False, "MULTIPLE_OBJECTS", [0, 1, 2], 3))

        s = self.summary(out)
        self.assertLessEqual(LIST_SUMMARY_KEYS, set(s))
        self.assertEqual((s["b_input"], s["consumer_kind"], s["list_consumer_calls"], s["list_candidates_sent"],
                          s["list_length_hist"]), (LIST_ENVELOPE_VERSION, "injected_test_only", 3, 9, {"3": 3}))
        self.assertEqual((s["cleanup_confirmed"], s["exit_code"]), (True, 0))

    def test_zero_one_mixed(self):
        """0개는 no_list_candidates(소비자 0), 1개는 길이 1, 혼합(볼트·모호·볼트)은 유효 원 인덱스 [0, 2]만."""
        sc = Scenario(4, masks={0: lambda: [bolt_mask()], 1: lambda: [bolt_mask()], 2: lambda: [rect_mask()],
                                3: lambda: [bolt_mask(y=50), rect_mask(), bolt_mask(y=190)]})
        code, err, out = self.run_list(sc)

        self.assertEqual(code, 0, err)
        self.assertEqual(self.consumer.stamps(), [sc.stamp(1), sc.stamp(3)])
        self.assertEqual(self.skip_reasons(out, sc.stamp(2)), ["no_list_candidates"])

        by = {r["input"]["ros_stamp_ns"]: r for r in self.accepts(out)}
        self.assertEqual(by[sc.stamp(1)]["list_indices"], [0])
        self.assertEqual(by[sc.stamp(3)]["list_indices"], [0, 2])
        self.assertEqual(self.summary(out)["list_length_hist"], {"1": 1, "2": 1})


class FrameCurrencyTest(ListBase):
    """프레임 단위 현재성: TF·보정·epoch·리셋·TF 수신 실패면 목록을 넘기지 않는다."""

    def test_tf_missing_late_mismatch(self):
        """TF 누락·상한 초과·stamp 불일치 프레임은 tf_* 사유, 소비자 0. 정상 프레임만 소비."""
        sc = Scenario(5, masks={i: three_bolts for i in range(5)})
        tf = FakeTf({sc.stamp(1): "missing", sc.stamp(2): "late", sc.stamp(3): "mismatch"})
        code, err, out = self.run_list(sc, tf=tf)

        self.assertEqual(code, 0, err)
        self.assertEqual(self.consumer.stamps(), [sc.stamp(4)])
        self.assertEqual(self.skip_reasons(out, sc.stamp(1)), ["tf_unavailable"])
        self.assertEqual(self.skip_reasons(out, sc.stamp(2)), ["tf_late"])
        self.assertEqual(self.skip_reasons(out, sc.stamp(3)), ["tf_stamp_mismatch"])

    def test_calibration_change_blocks_list(self):
        """검출 중 CameraInfo 보정 변경 → 그 프레임은 목록 미전달(세대 폐기·보정 불일치·옛 epoch 중 하나)."""
        def change(s):
            """보정 번호가 바뀌는 CameraInfo."""
            s.src.handle_camera_info(info_msg(fx=310.0, width=W, height=H, frame_id=FRAME_ID))

        sc = Scenario(4, masks={i: three_bolts for i in range(4)}, during={2: change})
        code, err, out = self.run_list(sc)

        self.assertEqual(code, 0, err)
        self.assertNotIn(sc.stamp(2), self.consumer.stamps())
        self.assertTrue(set(self.skip_reasons(out, sc.stamp(2))) & {"generation_discarded", "calibration_mismatch",
                                                                    "job_aborted", "rx_epoch_stale"})

    def test_reset_ends_run_no_list_after(self):
        """첫 목록 수락 뒤 /clock 되감기 → 실행 종료·철회, 그 뒤 목록 소비 0."""
        def rewind(s):
            """앞 프레임 수락을 기다린 뒤 /clock 역행."""
            link = self.controller.current
            end = time.monotonic() + 2.0

            while time.monotonic() < end and link.dispatcher.counts.get("accepted", 0) < 1:
                time.sleep(0.005)

            s.src.handle_clock(SimpleNamespace(clock=stamp(1, 0)))

        sc = Scenario(5, masks={i: three_bolts for i in range(5)}, during={2: rewind})
        code, err, out = self.run_list(sc)

        self.assertEqual(code, 3, err)   # #251 A-4: 되감기로 끝난 실행은 3(정상 0과 구분)
        self.assertIn("reason=reset_detected", err)
        self.assertEqual(self.consumer.stamps(), [sc.stamp(1)])
        kinds = [r["kind"] for r in self.records(out)]
        self.assertIn("revoke", kinds)
        self.assertEqual(kinds[-1], "run_end")
        s = self.summary(out)
        self.assertEqual((s["reset_detected"], s["cleanup_confirmed"], s["list_consumer_calls"]), (True, True, 1))

    def test_tf_receiver_failure_stops_lists(self):
        """TF 수신 스레드 실패 알림 뒤에는 목록을 넘기지 않고 종료 1(정리 미확인)."""
        class FailingTf(FakeTf):
            """실패 콜백을 받는 TF 대역."""

            def set_failure_callback(self, cb):
                """콜백 보관."""
                self.cb = cb

        tf = FailingTf()

        def fail(s):
            """앞 프레임 수락 뒤 TF 수신 실패."""
            link = self.controller.current
            end = time.monotonic() + 2.0

            while time.monotonic() < end and link.dispatcher.counts.get("accepted", 0) < 1:
                time.sleep(0.005)

            tf.cb("RuntimeError: spin died")

        sc = Scenario(5, masks={i: three_bolts for i in range(5)}, during={2: fail})
        code, err, out = self.run_list(sc, tf=tf)

        self.assertEqual(code, 1, err)
        self.assertEqual(self.consumer.stamps(), [sc.stamp(1)])
        s = self.summary(out)
        self.assertEqual((s["state_failed"], s["cleanup_confirmed"]), ("tf_receiver_failed", False))

    def test_list_envelope_error_recorded(self):
        """목록 봉투 오류는 b_input_rejected로 기록하고 소비자를 부르지 않는다(실행 오류 아님)."""
        with mock.patch.object(link_mod, "make_list_envelope", side_effect=ListEnvelopeError("bad parent")):
            sc = Scenario(3, masks={i: three_bolts for i in range(3)})
            code, err, out = self.run_list(sc)

        self.assertEqual(code, 0, err)
        self.assertEqual(self.consumer.envelopes, [])
        self.assertEqual(self.skip_reasons(out, sc.stamp(1)), ["b_input_rejected"])


class StartRefusalAndD13Test(ListBase):
    """시작 거부(단계 T로 바꾸지 않음)·D13 유지."""

    def test_no_process_frame_refused(self):
        """주입 없음 + 목록 입구 없는 B 단계 → 종료 1, 출력·기록·TF 없음, 단계 T로 돌지 않음.

        실제 B에는 `process_frame`이 생겼으므로(d4fab37), 그 입구가 없는 가짜 B 단계형(`NoListBStage`)을 끼워 거부를 확인한다.
        """
        self.inject = False
        self.assertTrue(callable(getattr(b_stage_mod.BStage, "process_frame", None)))   # 실제 B에는 있음

        class NoListBStage(b_stage_mod.BStage):
            """목록 입구(`process_frame`)가 없는 가짜 B 단계형(단계 T `consume` 등은 실제 그대로)."""

            process_frame = None

        sc = Scenario(3, masks={i: three_bolts for i in range(3)})

        with mock.patch.object(b_stage_mod, "BStage", NoListBStage):
            code, err, out = self.run_list(sc, tf=FakeTf())

        self.assertEqual(code, 1, err)
        self.assertIn("list consumer unavailable (BStage.process_frame missing); not falling back to stage T", err)
        self.assertFalse(Path(out).exists())
        self.assertFalse((Path(self.tmp.name) / f"{Path(out).name}.jsonl").exists())
        self.assertEqual((self.tf.calls, sc.detections, self.consume_calls), ([], 0, []))

    def test_stage_without_consumer_is_worker_error(self):
        """공장이 소비자를 못 주면(None) 단계 T로 바꾸지 않고 worker 오류로 종료 1."""
        self.consumer.factory = lambda stage: None
        sc = Scenario(3, masks={i: three_bolts for i in range(3)})
        code, err, out = self.run_list(sc)

        self.assertEqual(code, 1, err)
        self.assertIn("not falling back to stage T", err)
        self.assertEqual((self.accepts(out), self.consume_calls), ([], []))

    def test_on_output_refused(self):
        """B 단계 on_output이 있으면 목록 소비 없이 종료 1(C·D·로봇 송신 없음)."""
        hook = lambda stage: setattr(stage, "on_output", lambda *a: None)   # noqa: E731
        sc = Scenario(3, masks={i: three_bolts for i in range(3)})
        code, err, out = self.run_list(sc, hook=hook)

        self.assertEqual(code, 1, err)
        self.assertEqual(self.consumer.envelopes, [])

    def test_plain_ros_link_list_is_argument_error(self):
        """기록 전용 연결 없는 --ros-link --b-input list는 인자 오류 2(D13)."""
        argv = ["--ros-link", "--detector", "ai", "--ai-config", self.case.ai, "--log", str(Path(self.tmp.name) / "x.jsonl"),
                "--b-input", "list"]

        with self.assertRaises(SystemExit) as cm:
            self.mod.main(argv, stderr=io.StringIO(), ros_available=lambda: True)

        self.assertEqual(cm.exception.code, 2)

    def test_runner_live_consumers_still_refused(self):
        """실행기: 실시간 b_consumer·b_list_consumer는 그대로 거부, list 판정에 sink 없음은 live_observation_contract."""
        import test_pipeline_runner_live as live_mod

        # b_consumer는 Live 도우미 인자로, 나머지는 실행기 인자로 넣는다
        cases = {"b_consumer_not_allowed": ({"b_consumer": lambda f, o: None}, {}),
                 "b_list_consumer_not_allowed": ({}, {"b_list_consumer": lambda c, d: None}),
                 "live_observation_contract": ({}, {"live_observation_b_input": "list"})}

        for reason, (live_kw, run_kw) in cases.items():
            with self.subTest(reason=reason), mock.patch.object(live_mod, "run", functools.partial(live_mod.run, **run_kw)):
                live = live_mod.Live(feed=lambda s, c: s.handle_image(live_mod.image_msg(1)), **live_kw)
                res, _, err = live.run()

                self.assertEqual((res.exit_code, res.stop_reason, live.detect_calls), (1, reason, []), err)


class ListSkipReasonTest(unittest.TestCase):
    """`b_list_skip_reason`: MULTIPLE_OBJECTS 통과, 보정 불일치·옛 epoch 차단, 단계 T 판정은 그대로."""

    def test_multiple_objects_passes_list_only(self):
        """볼트 2개: 단계 T는 invalid_observation, 목록 판정은 통과(None)."""
        p, _, out = outcome_of(SCENARIOS["R4_two_valid"][0])

        self.assertEqual(out.observation.reason, "MULTIPLE_OBJECTS")
        self.assertEqual(b_skip_reason(p, out), "invalid_observation")
        self.assertIsNone(b_list_skip_reason(p, out))
        self.assertIsNone(b_list_skip_reason(p, out, SimpleNamespace(frame_current=lambda f: True)))

    def test_stale_epoch_blocked(self):
        """수신 epoch가 지난 프레임은 rx_epoch_stale."""
        p, _, out = outcome_of(SCENARIOS["R4_two_valid"][0])

        self.assertEqual(b_list_skip_reason(p, out, SimpleNamespace(frame_current=lambda f: False)), "rx_epoch_stale")

    def test_calibration_mismatch_and_empty(self):
        """프로필 없음/불일치는 calibration_mismatch, 유효 후보 0은 no_list_candidates, AI 아님은 not_ai_result."""
        p, _, out = outcome_of(SCENARIOS["R4_two_valid"][0], prof=False)
        self.assertEqual(b_list_skip_reason(p, out), "calibration_mismatch")

        p, _, out = outcome_of(SCENARIOS["R5_zero_valid"][0])
        self.assertEqual(b_list_skip_reason(p, out), "no_list_candidates")

        p, _, out = outcome_of(SCENARIOS["R4_two_valid"][0])
        no_display = type(out)(**{**out.__dict__, "display_result": None})
        self.assertEqual(b_list_skip_reason(p, no_display), "not_ai_result")


class StageTUnchangedTest(Base):
    """기본(single) 경로: accept 기록 키·순서와 요약 키가 #201 전과 같다(봉투 버전은 `b_input.version`)."""

    def test_single_records_unchanged(self):
        """단일 관측 accept 기록은 기존 키 그대로, 목록 키·요약 키 없음, 봉투 버전 eyetoarm.b_input/1."""
        code, err, out = self.run_cli(Scenario(3))

        self.assertEqual(code, 0, err)
        acc = [r for r in self.records(out) if r["kind"] == "accept"]
        self.assertTrue(acc)

        for r in acc:
            self.assertEqual(list(r), SINGLE_ACCEPT_KEYS)
            self.assertEqual(r["b_input"]["version"], "eyetoarm.b_input/1")

        self.assertFalse(LIST_SUMMARY_KEYS & set(self.summary(out)))
        self.assertNotIn("b_input=", err)


if __name__ == "__main__":
    unittest.main()
