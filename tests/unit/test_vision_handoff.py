"""A→C 같은 프로세스 내부 연결부(`AToCHandoff`) 시험 — 합성 프레임·실제 코어·가짜 수신 측.

모델·ROS·창·C 코드 없이 합성 영상과 시험용 AI 표시 결과로 실제 `VisionPipeline`을 돌린다.
"""
import dataclasses
import gc
import random
import threading
import unittest
import weakref

import numpy as np

from eyetoarm.common.types import FramePacket, InputStatus, InputStatusEvent, InvalidationKind
from eyetoarm.vision.handoff import (DEFAULT_CAPACITY, AToCHandoff, HandoffClosed, HandoffContractError, HandoffFailed,
                                     HandoffMessage, HandoffState, HandoffView, MsgKind, annotate)
from eyetoarm.vision.live_view import CLASS_COLORS_BGR
from eyetoarm.vision.pipeline import GenerationCause, VisionPipeline
from eyetoarm.vision.pipeline_types import JobContext, JobOutcome
from test_ai_config import ai_config
from test_ai_pipeline import ai_observation
from test_ai_result import seg_result
from test_live_view import POLY, big_instance
from test_pipeline import RUN_ID, FakeClock, frame, settings
from test_pipeline_terminal import INF, frameless_event, job_event

H, W = 200, 240


def image(value=40):
    """합성 원본(BGR uint8)."""
    return np.full((H, W, 3), value, np.uint8)


class HandoffTestBase(unittest.TestCase):
    """AI 설정 코어·가짜 시계·연결부를 준비한다."""

    def setUp(self):
        """코어와 연결부를 만들고 실행을 시작한다(RUN_START 1건)."""
        self.clock = FakeClock()
        self.p = self.pipeline()
        self.h = AToCHandoff(clock_ns=self.clock)
        self.h.start_run(self.p)

    def pipeline(self, run_id=RUN_ID):
        """AI 설정(프로파일 없음)의 새 코어."""
        return VisionPipeline(settings(detector_config=ai_config(), profile=None), run_id=run_id, clock_ns=self.clock)

    def start(self, image_seq, p=None, value=40):
        """합성 프레임을 넣고 작업을 시작한다."""
        p = p or self.p
        self.assertTrue(p.offer_frame(frame(image_seq, p.current_generation(), image=image(value), width=W, height=H,
                                            run_id=p.run_id)))
        job = p.start_next()
        self.assertIsNotNone(job)

        return job

    def complete(self, job, instances=None, p=None, **display_over):
        """시험용 AI 표시 결과로 실제 complete한 응답."""
        p = p or self.p
        instances = (big_instance(),) if instances is None else instances
        fields = {"config_sha256": job.detector_config_id, "model_id": job.settings.detector_config.model_id,
                  "crop_wh": (W, H)}
        fields.update(display_over)

        return p.complete(job, ai_observation(job), display_result=seg_result(instances, **fields))

    def run_job(self, image_seq, instances=None, p2=None, **display_over):
        """작업 하나를 완료해 연결부에 결과로 넘긴다(p2를 주면 그 코어). (응답, RESULT 메시지 seq)."""
        out = self.complete(self.start(image_seq, p2), instances, p2, **display_over)
        self.assertTrue(out.applied)
        seq = self.h.offer_result(out)

        return out, seq

    def drain(self):
        """제어 메시지를 모두 꺼낸다."""
        msgs, _ = self.h.take_control(10_000)

        return msgs

    def assert_balances(self):
        """보존식 네 개를 확인한다."""
        s = self.h.stats()
        self.assertEqual(s.published, s.taken + s.pending)
        self.assertEqual(s.views_put, s.views_taken + s.views_replaced + s.views_cleared + s.view_pending)
        self.assertEqual(s.events_enqueued + s.events_unsent, len(self.p.invalidation_records))
        self.assertEqual(s.results_enqueued + s.results_rejected, s.results_offered)


class ResultPairingTest(HandoffTestBase):
    """RESULT와 영상 칸이 같은 응답에서 나오고 무검출이 이전 윤곽을 대체한다."""

    def test_result_and_view_same_frame_and_display(self):
        """RESULT의 관측·표시 결과와 영상 칸의 프레임·표시 결과가 같은 응답의 객체이고 식별자가 프레임·작업 값 그대로다."""
        out, seq = self.run_job(0)
        (start, msg) = self.drain()
        view, drained = self.h.take_view()

        self.assertIs(start.msg_kind, MsgKind.RUN_START)
        self.assertIs(msg.msg_kind, MsgKind.RESULT)
        self.assertEqual(msg.msg_seq, seq)
        self.assertIs(msg.observation, out.observation)
        self.assertIs(msg.display, out.display_result)
        self.assertTrue(msg.display_only)                 # valid=False 표시용 그대로
        self.assertFalse(msg.observation.valid)
        f = out.context.frame
        self.assertEqual((msg.run_id, msg.source_id, msg.image_seq, msg.capture_generation, msg.timestamp_ms,
                          msg.timestamp_base, msg.ros_stamp_ns, msg.received_monotonic_ns, msg.calibration_id),
                         (f.run_id, f.source_id, f.image_seq, f.capture_generation, f.timestamp_ms, f.timestamp_base,
                          f.ros_stamp_ns, f.received_monotonic_ns, f.calibration_id))
        self.assertEqual((msg.generation, msg.job_seq, msg.detector_config_id),
                         (out.context.generation, out.context.job_seq, out.context.detector_config_id))
        self.assertIsInstance(view, HandoffView)
        self.assertEqual(view.msg_seq, seq)
        self.assertIs(view.frame, f)
        self.assertIs(view.display, out.display_result)
        self.assertTrue(drained)
        self.assertTrue(self.h.is_current(seq))
        self.assert_balances()

    def test_no_detection_replaces_previous_contours(self):
        """무검출 RESULT가 이전 결과·영상을 대체한다(이전 윤곽이 현재로 남지 않음)."""
        _, seq0 = self.run_job(0)
        out1, seq1 = self.run_job(1, instances=())
        view, _ = self.h.take_view()

        self.assertFalse(self.h.is_current(seq0))
        self.assertTrue(self.h.is_current(seq1))
        self.assertEqual(view.msg_seq, seq1)
        self.assertEqual(view.display.instances, ())
        self.assertEqual(self.h.stats().views_replaced, 1)
        self.assert_balances()

    def test_annotate_keeps_original_and_roi_coordinates(self):
        """annotate는 복사본에 그리고 원본(코어 소유 읽기 전용) 불변, ROI가 있어도 인스턴스 좌표를 그대로 쓴다."""
        out, _ = self.run_job(0, roi=(80, 80, 120, 100), crop_wh=(120, 100))
        view, _ = self.h.take_view()
        before = view.frame.image.copy()
        drawn = annotate(view)

        np.testing.assert_array_equal(view.frame.image, before)
        self.assertIsNot(drawn, view.frame.image)
        self.assertFalse(view.frame.image.flags.writeable)
        # 박스 오른쪽 변(원본 좌표)
        self.assertEqual(tuple(int(v) for v in drawn[125, 140]), CLASS_COLORS_BGR[0])
        self.assertEqual(tuple(int(v) for v in drawn[190, 200]), (40, 40, 40))  # ROI만큼 밀린 곳은 원본 그대로
        self.assertEqual(POLY[0], [100, 100])

    def test_offer_result_contract(self):
        """duplicate·현재 미반영·완료 아님·코어 현재 관측이 아닌 응답은 계약 오류이고 대기열·계수를 바꾸지 않는다."""
        job = self.start(0)
        out = self.complete(job)
        dup = self.complete(job)
        bads = (dup, dataclasses.replace(out, applied=False), self.p.fail(self.start(1), INF, "boom"), "x")
        before = self.h.stats()

        for bad in bads:
            with self.subTest(bad=bad):
                with self.assertRaises(HandoffContractError):
                    self.h.offer_result(bad)

        # out은 코어 현재 관측이 아니다(job1 실패가 지움)
        with self.assertRaises(HandoffContractError):
            self.h.offer_result(out)

        self.assertEqual(self.h.stats(), before)


class InvalidationScopeTest(HandoffTestBase):
    """무효화는 코어 규칙과 같은 범위만 현재 결과를 지운다."""

    def sync(self):
        """새 코어 사건을 연결부로 넘긴다."""
        return self.h.sync_invalidations(self.p)

    def test_job0_event_keeps_job1_result(self):
        """job1 현재 상태에서 job0 applied 사건은 job1 RESULT·영상을 유지한다."""
        out0, _ = self.run_job(0)
        out1, seq1 = self.run_job(1)
        self.assertTrue(self.p.apply_invalidation(job_event(out0.context)))
        self.assertEqual(self.sync(), 1)

        self.assertTrue(self.h.is_current(seq1))
        self.assertEqual(self.h.current_result, (seq1, 0, 1))
        self.assertEqual(self.h.take_view()[0].msg_seq, seq1)
        self.assertIs(self.p.current_observation, out1.observation)   # 코어와 같은 판정

    def test_same_or_later_job_event_clears(self):
        """현재 결과 작업과 같은(이하) job_seq 사건은 결과와 영상을 지운다."""
        out0, seq0 = self.run_job(0)
        self.p.apply_invalidation(job_event(out0.context))
        self.sync()

        self.assertFalse(self.h.is_current(seq0))
        self.assertIsNone(self.h.current_result)
        self.assertEqual(self.h.take_view(), (None, False))
        self.assertIsNone(self.p.current_observation)
        self.assert_balances()

    def test_later_job_failure_clears(self):
        """현재 결과보다 뒤 작업의 실패 사건(job_seq 큼)은 현재 결과를 지운다."""
        _, seq0 = self.run_job(0)
        self.p.fail(self.start(1), INF, "boom")
        self.sync()

        self.assertFalse(self.h.is_current(seq0))
        self.assertIsNone(self.p.current_observation)

    def test_frameless_event_clears_same_generation(self):
        """같은 세대의 프레임 없는 사건은 결과를 지운다."""
        _, seq0 = self.run_job(0)
        self.p.apply_invalidation(frameless_event(self.p))
        self.sync()

        self.assertFalse(self.h.is_current(seq0))

    def test_generation_change_clears_previous_generation(self):
        """GENERATION_CHANGED는 이전 세대 결과를 지운다."""
        _, seq0 = self.run_job(0)
        self.p.change_settings(self.p.settings, cause=GenerationCause.SEEK)
        self.sync()

        self.assertFalse(self.h.is_current(seq0))
        self.assertIsNone(self.h.current_result)

    def test_ignored_event_keeps_result_and_preserves_fields(self):
        """이전 세대 사건(applied=False)은 현재 결과를 그대로 두고, 원래 세대는 event 안·코어 세대는 바깥에."""
        out0, _ = self.run_job(0)
        self.p.change_settings(self.p.settings, cause=GenerationCause.SEEK)
        _, seq1 = self.run_job(1)
        self.assertFalse(self.p.apply_invalidation(job_event(out0.context)))
        self.sync()
        last = self.drain()[-1]

        self.assertIs(last.msg_kind, MsgKind.INVALIDATION)
        self.assertFalse(last.applied)
        self.assertEqual(last.event.generation, 0)
        self.assertEqual(last.core_generation, 1)
        self.assertTrue(self.h.is_current(seq1))

    def test_full_clears(self):
        """INPUT_STATUS·CLEAR·RUN_END는 결과를 지운다."""
        for name in ("input", "clear", "end"):
            with self.subTest(name=name):
                _, seq = self.run_job(self.p.counters.received)

                if name == "input":
                    ev = InputStatusEvent(run_id=RUN_ID, source_id="src0", status=InputStatus.FRAME_TIMEOUT,
                                          event_monotonic_ns=5, detail="t")
                    self.p.note_input_status(ev)
                    self.h.note_input_status(ev)
                elif name == "clear":
                    self.h.clear("result_withheld")
                else:
                    self.h.end_run(0, "eof")

                self.assertFalse(self.h.is_current(seq))
                self.assertEqual(self.h.take_view(), (None, False))

    def test_sync_forwards_each_record_once_in_order(self):
        """records 새 항목만 순서대로 1회 넘기고 applied 판정은 코어 기록 그대로다(같은 사건 재apply 무추가)."""
        out0, _ = self.run_job(0)
        ev = job_event(out0.context)
        self.p.apply_invalidation(ev)
        self.p.apply_invalidation(ev)
        self.assertEqual(self.sync(), 1)
        self.assertEqual(self.sync(), 0)
        msgs = [m for m in self.drain() if m.msg_kind is MsgKind.INVALIDATION]

        self.assertEqual([(m.event, m.applied) for m in msgs],
                         [(r.event, r.applied) for r in self.p.invalidation_records])
        self.assertEqual(self.h.stats().events_applied, 1)
        self.assertEqual(self.h.stats().events_ignored, 0)
        self.assert_balances()

    def test_sync_rejects_other_pipeline(self):
        """시작한 코어가 아닌 코어의 사건 동기화는 계약 오류다."""
        with self.assertRaises(HandoffContractError):
            self.h.sync_invalidations(self.pipeline())


class TakenResultRaceTest(HandoffTestBase):
    """이미 꺼낸 RESULT·영상도 뒤 경계로 현재가 아니게 된다(결정적 재현)."""

    def test_taken_result_becomes_stale(self):
        """RESULT·영상을 꺼낸 뒤 경계가 먼저 처리되면 is_current False, annotate 뒤 재확인도 False."""
        out0, seq0 = self.run_job(0)
        msgs = self.drain()
        view, _ = self.h.take_view()
        self.assertTrue(self.h.is_current(msgs[-1].msg_seq))

        self.p.apply_invalidation(job_event(out0.context))
        self.h.sync_invalidations(self.p)
        annotate(view)

        self.assertFalse(self.h.is_current(view.msg_seq))
        self.assertFalse(self.h.is_current(seq0))


    def test_queued_result_not_revived_after_boundary(self):
        """RESULT가 대기열에 있는 동안 경계가 먼저 처리되면 나중에 꺼내도 현재가 아니고 영상 칸도 비어 있다."""
        _, seq0 = self.run_job(0)
        ev = InputStatusEvent(run_id=RUN_ID, source_id="src0", status=InputStatus.FRAME_TIMEOUT,
                              event_monotonic_ns=5, detail=None)
        self.p.note_input_status(ev)
        self.h.note_input_status(ev)
        msgs = self.drain()

        self.assertEqual([m.msg_kind for m in msgs], [MsgKind.RUN_START, MsgKind.RESULT, MsgKind.INPUT_STATUS])
        self.assertFalse(self.h.is_current(msgs[1].msg_seq))
        self.assertEqual(self.h.take_view(), (None, True))   # 대기열까지 다 꺼낸 뒤라 drained


class FramelessMessageTest(HandoffTestBase):
    """프레임·작업 없는 메시지는 식별자 None(가짜 0 없음)."""

    def test_frameless_ids_none(self):
        """RUN_START·프레임 없는 INVALIDATION·INPUT_STATUS·CLEAR·RUN_END는 프레임·작업 식별자가 None."""
        self.p.apply_invalidation(frameless_event(self.p))
        self.h.sync_invalidations(self.p)
        ev = InputStatusEvent(run_id=RUN_ID, source_id="src0", status=InputStatus.FRAME_TIMEOUT,
                              event_monotonic_ns=5, detail=None)
        self.h.note_input_status(ev)
        self.h.clear("x")
        self.h.end_run(1, "model_load_failed")
        msgs = self.drain()

        self.assertEqual([m.msg_kind for m in msgs], [MsgKind.RUN_START, MsgKind.INVALIDATION, MsgKind.INPUT_STATUS,
                                                      MsgKind.CLEAR, MsgKind.RUN_END])
        self.assertEqual([m.msg_seq for m in msgs], [1, 2, 3, 4, 5])

        for m in msgs:
            self.assertEqual((m.image_seq, m.job_seq, m.capture_generation), (None, None, None))
            self.assertEqual((m.run_id, m.source_id), (RUN_ID, "src0"))

        self.assertIsNone(msgs[1].event.job_seq)
        self.assertEqual((msgs[2].status, msgs[2].detail, msgs[2].event_monotonic_ns),
                         (InputStatus.FRAME_TIMEOUT, None, 5))
        self.assertEqual((msgs[4].exit_code, msgs[4].stop_reason), (1, "model_load_failed"))
        self.assertEqual((msgs[0].method, msgs[0].model_id, msgs[0].detector_config_id),
                         ("ai", self.p.settings.detector_config.model_id, self.p.settings.detector_config_id))

    def test_msg_seq_not_reset_across_runs(self):
        """다음 실행의 RUN_START도 msg_seq를 이어서 발급한다(0으로 초기화하지 않음)."""
        self.h.end_run(0, "eof")
        p2 = self.pipeline("run-2")
        self.h.start_run(p2)
        msgs = self.drain()

        self.assertEqual([m.msg_seq for m in msgs], [1, 2, 3])
        self.assertEqual(msgs[2].run_id, "run-2")

    def test_input_status_other_run_rejected(self):
        """다른 run의 입력 사건은 계약 오류다."""
        ev = InputStatusEvent(run_id="other", source_id="src0", status=InputStatus.FRAME_TIMEOUT,
                              event_monotonic_ns=5, detail=None)

        with self.assertRaises(HandoffContractError):
            self.h.note_input_status(ev)


class RetentionTest(HandoffTestBase):
    """소비 정지 중 영상 1장, 이전 프레임 회수, 제어 메시지에 영상·작업 참조 없음."""

    def test_slow_consumer_keeps_one_view_and_releases_frames(self):
        """꺼내지 않아도 영상 칸은 1장이고 대체된 프레임·응답은 회수된다."""
        refs = []

        for i in range(4):
            out, _ = self.run_job(i)
            refs.append((weakref.ref(out.context.frame), weakref.ref(out)))
            del out

        gc.collect()
        alive = [(f() is not None, o() is not None) for f, o in refs]

        self.assertEqual(alive[:3], [(False, False)] * 3)
        self.assertEqual(self.h.stats().view_pending, 1)
        self.assertEqual(self.h.stats().views_replaced, 3)
        self.assert_balances()

    def test_control_messages_hold_no_frame_or_job(self):
        """제어 메시지 필드에 ndarray·FramePacket·JobContext·JobOutcome이 없다."""
        out, _ = self.run_job(0)
        self.p.apply_invalidation(job_event(out.context))
        self.h.sync_invalidations(self.p)

        for m in self.drain():
            for f in dataclasses.fields(m):
                value = getattr(m, f.name)
                self.assertNotIsInstance(value, (np.ndarray, FramePacket, JobContext, JobOutcome), (m.msg_kind, f.name))


class CapacityTest(HandoffTestBase):
    """용량 입력 검사와 넘침 fail-stop."""

    def test_capacity_validation(self):
        """bool·0·음수·실수·문자열 용량은 계약 오류, 기본값은 64."""
        self.assertEqual(DEFAULT_CAPACITY, 64)
        self.assertEqual(AToCHandoff().stats().capacity, 64)

        for bad in (True, 0, -1, 2.0, "3", None):
            with self.subTest(bad=bad):
                with self.assertRaises(HandoffContractError):
                    AToCHandoff(capacity=bad)

    def test_notify_must_be_callable(self):
        """notify는 None 또는 호출 가능 객체."""
        with self.assertRaises(HandoffContractError):
            AToCHandoff(notify=3)

    def test_overflow_latches_failed(self):
        """작은 용량에서 넘치면 넣지 않고 FAILED(overflow)·HandoffFailed, 이후 거부·is_current False·FIFO 무변."""
        h = AToCHandoff(capacity=2, clock_ns=self.clock)
        h.start_run(self.p)
        out, seq = None, None
        out = self.complete(self.start(0))
        seq = h.offer_result(out)

        with self.assertRaises(HandoffFailed):
            h.clear("x")

        status = h.state()
        self.assertIs(status.state, HandoffState.FAILED)
        self.assertEqual(status.reason, "overflow")
        self.assertFalse(h.is_current(seq))
        self.assertIsNone(h.current_result)
        self.assertEqual(h.take_view(), (None, False))

        with self.assertRaises(HandoffFailed):
            h.end_run(0, "eof")

        msgs, drained = h.take_control(10)
        self.assertEqual([m.msg_kind for m in msgs], [MsgKind.RUN_START, MsgKind.RESULT])
        self.assertTrue(drained)
        s = h.stats()
        self.assertEqual((s.published, s.rejected), (2, 2))

    def test_failed_rejects_results_and_events(self):
        """FAILED 뒤 결과·사건 넘김은 거부되고 결과·사건 보존식이 유지된다."""
        self.h.report_consumer_failure("send error")
        out = self.complete(self.start(0))

        with self.assertRaises(HandoffFailed):
            self.h.offer_result(out)

        self.p.apply_invalidation(job_event(out.context))

        with self.assertRaises(HandoffFailed):
            self.h.sync_invalidations(self.p)

        s = self.h.stats()
        self.assertEqual((s.results_offered, s.results_rejected, s.events_unsent), (1, 1, 1))
        self.assert_balances()


class NotifyTest(unittest.TestCase):
    """알림 병합·부분 take·영상만 남은 경우·교차·예외·잠금 밖 호출."""

    def setUp(self):
        """알림을 세는 연결부와 코어."""
        self.calls = 0
        self.clock = FakeClock()
        self.p = VisionPipeline(settings(detector_config=ai_config(), profile=None), run_id=RUN_ID, clock_ns=self.clock)

    def counting(self):
        """호출 수를 세는 notify."""
        def notify():
            """알림 1회."""
            self.calls += 1

        return notify

    def make(self, notify, capacity=DEFAULT_CAPACITY):
        """연결부를 만들고 실행을 시작한다."""
        h = AToCHandoff(capacity=capacity, notify=notify, clock_ns=self.clock)
        h.start_run(self.p)

        return h

    def result(self, h, image_seq):
        """작업 하나를 완료해 넘긴다."""
        self.assertTrue(self.p.offer_frame(frame(image_seq, self.p.current_generation(), image=image(), width=W,
                                                 height=H)))
        job = self.p.start_next()
        fields = {"config_sha256": job.detector_config_id, "model_id": job.settings.detector_config.model_id,
                  "crop_wh": (W, H)}
        out = self.p.complete(job, ai_observation(job), display_result=seg_result((big_instance(),), **fields))

        return h.offer_result(out)

    def test_coalesced_until_full_drain(self):
        """미처리 알림은 하나만: 모두 비기 전에는 다시 알리지 않고, 모두 꺼낸 뒤 새 데이터면 다시 알린다."""
        h = self.make(self.counting())
        h.clear("a")
        h.clear("b")
        self.assertEqual(self.calls, 1)

        msgs, drained = h.take_control(1)        # 부분 take
        self.assertFalse(drained)
        self.assertTrue(h.stats().notify_pending)
        h.clear("c")
        self.assertEqual(self.calls, 1)

        h.take_control(100)
        self.assertFalse(h.stats().notify_pending)
        h.clear("d")
        self.assertEqual(self.calls, 2)

    def test_view_only_remaining_keeps_pending(self):
        """제어 메시지를 다 꺼내도 영상 칸이 남으면 pending 유지, 영상까지 꺼내야 내린다."""
        h = self.make(self.counting())
        self.result(h, 0)
        _, drained = h.take_control(100)

        self.assertFalse(drained)
        self.assertTrue(h.stats().notify_pending)
        _, drained = h.take_view()
        self.assertTrue(drained)
        self.assertFalse(h.stats().notify_pending)

    def test_interleaving_never_data_without_pending(self):
        """생산과 drain을 무작위로 섞어도 「데이터 있음 + 알림 대기 아님」 상태가 없다."""
        h = self.make(self.counting(), capacity=10_000)
        rng = random.Random(7)
        image_seq = 0

        for _ in range(400):
            op = rng.randrange(4)

            if op == 0:
                h.clear("x")
            elif op == 1:
                self.result(h, image_seq)
                image_seq += 1
            elif op == 2:
                h.take_control(rng.randrange(1, 4))
            else:
                h.take_view()

            s = h.stats()

            if s.pending or s.view_pending:
                self.assertTrue(s.notify_pending)

    def test_notify_called_outside_lock(self):
        """notify 안에서 다른 스레드가 take해도 막히지 않는다(잠금 밖 호출)."""
        done = []

        def notify():
            """다른 스레드에서 take_control을 부르고 끝나길 기다린다."""
            t = threading.Thread(target=lambda: done.append(h.take_control(10)))
            t.start()
            t.join(2.0)

        h = AToCHandoff(notify=notify, clock_ns=self.clock)
        h.start_run(self.p)

        self.assertEqual(len(done), 1)
        self.assertEqual(done[0][0][0].msg_kind, MsgKind.RUN_START)

    def test_notify_exception_latches_failed(self):
        """notify 예외는 FAILED(notify: …)·HandoffFailed이고 이후 거부."""
        def notify():
            """항상 실패."""
            raise RuntimeError("loop closed")

        h = AToCHandoff(notify=notify, clock_ns=self.clock)

        with self.assertRaises(HandoffFailed):
            h.start_run(self.p)

        self.assertIs(h.state().state, HandoffState.FAILED)
        self.assertTrue(h.state().reason.startswith("notify: "))

        with self.assertRaises(HandoffFailed):
            h.clear("x")


class ConcurrencyTest(unittest.TestCase):
    """실제 스레드·asyncio 루프로 순서·보존식·알림 병합을 확인한다(실제 C 루프 아님)."""

    def setUp(self):
        """코어와 시계."""
        self.clock = FakeClock()
        self.p = VisionPipeline(settings(detector_config=ai_config(), profile=None), run_id=RUN_ID, clock_ns=self.clock)

    def test_threaded_consumer_receives_all_in_order(self):
        """생산자와 소비 스레드가 동시에 돌아도 msg_seq가 빠짐·중복 없이 순서대로 오고 보존식이 맞다."""
        h = AToCHandoff(capacity=100_000)
        got = []
        stop = threading.Event()

        def consume():
            """멈춤 신호 뒤 대기열이 빌 때까지 꺼낸다. 신호를 꺼내기 **전에** 읽어야 그 뒤 들어온 메시지를 놓치지 않는다."""
            while True:
                done = stop.is_set()
                msgs, drained = h.take_control(7)
                got.extend(m.msg_seq for m in msgs)

                if done and drained:
                    return

        t = threading.Thread(target=consume)
        t.start()
        h.start_run(self.p)

        for i in range(2000):
            h.clear(f"c{i}")

        stop.set()
        t.join(10.0)
        s = h.stats()

        self.assertFalse(t.is_alive())
        self.assertEqual(got, list(range(1, 2002)))
        self.assertEqual((s.published, s.taken, s.pending), (2001, 2001, 0))

    def test_asyncio_slow_loop_single_callback(self):
        """느린(아직 돌지 않는) asyncio 루프에는 call_soon_threadsafe가 한 번만 쌓이고, 루프가 돌면 모두 순서대로 꺼낸다."""
        import asyncio

        loop = asyncio.new_event_loop()
        ready = None   # 루프를 현재 루프로 둔 뒤 만든다(Python 3.8~3.9 호환)
        scheduled = []

        def notify():
            """루프에 깨움 콜백을 예약하고 그 수를 센다."""
            scheduled.append(1)
            loop.call_soon_threadsafe(ready.set)

        async def consumer(h):
            """깨움을 기다렸다가 모두 꺼낸다."""
            out = []

            while True:
                await asyncio.wait_for(ready.wait(), 2.0)
                ready.clear()

                while True:
                    msgs, drained = h.take_control(5)
                    out.extend(msgs)

                    if drained:
                        return out

        try:
            asyncio.set_event_loop(loop)
            ready = asyncio.Event()
            h = AToCHandoff(notify=notify)
            h.start_run(self.p)

            for i in range(50):
                h.clear(f"c{i}")

            self.assertEqual(len(scheduled), 1)
            out = loop.run_until_complete(consumer(h))
        finally:
            asyncio.set_event_loop(None)
            loop.close()

        self.assertEqual([m.msg_seq for m in out], list(range(1, 52)))


class MultiRunEventScopeTest(HandoffTestBase):
    """연결부를 다음 실행에 다시 쓰면 사건 계수는 실행별, 제어 계수·msg_seq는 수명 전체다."""

    def run1_with_events(self):
        """실행1: 결과·applied 사건·ignored 사건을 만들고 동기화한 뒤 RUN_END. (결과 seq, 실행1 사건 수)."""
        out0, _ = self.run_job(0)
        # applied GENERATION_CHANGED
        self.p.change_settings(self.p.settings, cause=GenerationCause.SEEK)
        _, seq1 = self.run_job(1)
        self.assertFalse(self.p.apply_invalidation(job_event(out0.context)))       # ignored(이전 세대)
        self.h.sync_invalidations(self.p)
        self.h.end_run(0, "done")

        return seq1, len(self.p.invalidation_records)

    def assert_run_scope(self, p):
        """현재 실행 기준 사건 보존식, 넘긴 기록의 applied/ignored 수(모두 넘겼으면 그 코어 계수 합과 같음)."""
        s = self.h.stats()
        c = p.counters
        sent = p.invalidation_records[:s.events_enqueued]

        self.assertEqual(s.run_id, p.run_id)
        self.assertEqual(s.events_enqueued + s.events_unsent, len(p.invalidation_records))
        self.assertEqual(s.events_applied, sum(r.applied for r in sent))
        self.assertEqual(s.events_ignored, sum(not r.applied for r in sent))

        if s.events_unsent == 0:
            self.assertEqual(s.events_applied, sum(c.invalidation_applied_by_kind.values()))
            self.assertEqual(s.events_ignored, sum(c.invalidation_ignored_by_kind.values()))
        self.assertEqual(s.published, s.taken + s.pending)

    def test_second_run_without_events_starts_at_zero(self):
        """실행2를 사건 0개로 시작하면 실행2 사건 계수는 0이고 실행1 사건을 다시 보내지 않는다(메시지가 큐에 남은 상태)."""
        _, n1 = self.run1_with_events()
        self.assertEqual(n1, 2)
        self.assert_run_scope(self.p)
        p2 = self.pipeline("run-two")
        self.h.start_run(p2)

        self.assertEqual(self.h.sync_invalidations(p2), 0)
        s = self.h.stats()
        self.assertEqual((s.events_enqueued, s.events_unsent, s.events_applied, s.events_ignored), (0, 0, 0, 0))
        self.assertEqual(s.events_enqueued_total, 2)
        self.assert_run_scope(p2)

    def test_second_run_events_sent_once_with_both_runs_queued(self):
        """실행2의 새 사건만 정확히 1회 넘기고(반복 sync 무추가), 두 실행 메시지가 큐에 남아도 run_id·msg_seq 순서가 맞다."""
        seq1, _ = self.run1_with_events()
        p2 = self.pipeline("run-two")
        self.h.start_run(p2)
        _, seq2 = self.run_job(0, p2=p2)
        p2.apply_invalidation(dataclasses.replace(frameless_event(p2), run_id=p2.run_id))

        self.assertEqual(self.h.sync_invalidations(p2), 1)
        self.assertEqual(self.h.sync_invalidations(p2), 0)
        self.assert_run_scope(p2)
        self.assertEqual(self.h.stats().events_enqueued_total, 3)
        self.assertFalse(self.h.is_current(seq1))
        self.assertFalse(self.h.is_current(seq2))
        msgs = self.drain()

        self.assertEqual([m.msg_seq for m in msgs], list(range(1, len(msgs) + 1)))
        kinds = [(m.run_id, m.msg_kind) for m in msgs]
        self.assertEqual(kinds.count((RUN_ID, MsgKind.INVALIDATION)), 2)
        self.assertEqual(kinds.count(("run-two", MsgKind.INVALIDATION)), 1)
        self.assertEqual(kinds[-3:], [("run-two", MsgKind.RUN_START), ("run-two", MsgKind.RESULT),
                                      ("run-two", MsgKind.INVALIDATION)])

    def test_unsent_fixed_at_producer_sync(self):
        """미전달 수는 생산자 sync 경계에서 확정한다: sync 전 새 사건은 아직 세지 않고, 실패 뒤 sync에서 미전달로 확정된다."""
        out0, _ = self.run_job(0)
        self.p.apply_invalidation(job_event(out0.context))

        self.assertEqual(self.h.stats().events_unsent, 0)        # 아직 sync 경계 전
        self.h.report_consumer_failure("x")

        with self.assertRaises(HandoffFailed):
            self.h.sync_invalidations(self.p)

        self.assertEqual(self.h.stats().events_unsent, 1)
        self.assert_run_scope(self.p)


class LifecycleTest(HandoffTestBase):
    """수신 측 실패·close 멱등·생산자 스레드 고정·check."""

    def test_consumer_failure_latches(self):
        """report_consumer_failure → FAILED(consumer: …), 꺼낸 결과도 현재가 아님, 이후 생산자 거부."""
        _, seq = self.run_job(0)
        self.h.report_consumer_failure("socket closed")
        status = self.h.state()

        self.assertIs(status.state, HandoffState.FAILED)
        self.assertEqual(status.reason, "consumer: socket closed")
        self.assertFalse(self.h.is_current(seq))

        with self.assertRaises(HandoffFailed):
            self.h.clear("x")

        self.h.report_consumer_failure("again")             # 첫 사유 유지
        self.assertEqual(self.h.state().reason, "consumer: socket closed")

    def test_close_idempotent_and_rejects_producer(self):
        """close는 멱등이고 이후 생산자는 HandoffClosed, 남은 메시지는 꺼낼 수 있다."""
        self.run_job(0)
        self.h.close()
        self.h.close()

        self.assertIs(self.h.state().state, HandoffState.CLOSED)
        self.assertEqual(self.h.take_view(), (None, False))

        with self.assertRaises(HandoffClosed):
            self.h.end_run(0, "eof")

        with self.assertRaises(HandoffClosed):
            self.h.check()

        self.assertEqual(len(self.drain()), 2)
        self.assert_balances()

    def test_close_keeps_failed_reason(self):
        """FAILED 뒤 close는 CLOSED로 바꾸되 첫 실패 사유를 유지한다."""
        self.h.report_consumer_failure("x")
        self.h.close()

        self.assertIs(self.h.state().state, HandoffState.CLOSED)
        self.assertEqual(self.h.state().reason, "consumer: x")

    def test_producer_thread_pinned(self):
        """첫 생산자 호출 스레드가 아닌 스레드의 생산자 호출은 계약 오류다."""
        errors = []

        def other():
            """다른 스레드에서 clear."""
            try:
                self.h.clear("x")
            except HandoffContractError as exc:
                errors.append(exc)

        t = threading.Thread(target=other)
        t.start()
        t.join(2.0)

        self.assertEqual(len(errors), 1)
        self.assertEqual(self.h.stats().published, 1)

    def test_check(self):
        """check는 OPEN이면 무동작(메시지·계수 무변), FAILED면 HandoffFailed."""
        before = self.h.stats()
        self.h.check()
        self.assertEqual(self.h.stats(), before)

        self.h.report_consumer_failure("x")

        with self.assertRaises(HandoffFailed):
            self.h.check()

        self.assertEqual(self.h.stats().published, before.published)

    def test_take_control_max_n_validation(self):
        """max_n은 양의 정수(bool 제외)."""
        for bad in (0, -1, True, 1.5):
            with self.subTest(bad=bad):
                with self.assertRaises(HandoffContractError):
                    self.h.take_control(bad)

    def test_producer_before_start_run(self):
        """start_run 전 생산자 호출은 계약 오류다."""
        h = AToCHandoff(clock_ns=self.clock)

        with self.assertRaises(HandoffContractError):
            h.clear("x")

    def test_message_type(self):
        """제어 메시지는 HandoffMessage이고 바뀌지 않는다."""
        (m,) = self.drain()
        self.assertIsInstance(m, HandoffMessage)

        with self.assertRaises(dataclasses.FrozenInstanceError):
            m.msg_seq = 9

    def test_invalidation_kind_untouched(self):
        """INVALIDATION의 event는 코어 사건 객체 그대로다(kind·cause 보존)."""
        job = self.start(0)
        out = self.p.fail(job, InvalidationKind.JOB_ABORTED, "detect_timeout")
        self.h.sync_invalidations(self.p)
        m = self.drain()[-1]

        self.assertIs(m.event, out.invalidation)


if __name__ == "__main__":
    unittest.main()
