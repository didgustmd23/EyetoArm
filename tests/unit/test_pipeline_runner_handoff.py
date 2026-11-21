"""실행기 A→C 연결부(`handoff_sink`) 시험: 전달 위치·순서·fail-stop·종료 정리·기본 None 회귀.

가짜 저장·실시간 소스와 실제 `LiveLinkSource`(가짜 수신 연결)만 쓴다. ROS·YOLO·창·C 코드 없음.
"""
import io
import threading
import time
import unittest

from eyetoarm.common.types import InputStatus, InputStatusEvent, InvalidationKind
from eyetoarm.vision.ai_result import ModelPrepareError
from eyetoarm.vision.handoff import AToCHandoff, HandoffState, MsgKind
from eyetoarm.vision.pipeline import GenerationCause
from eyetoarm.vision.pipeline_records import JsonlRecorder, PipelineLog
from eyetoarm.vision.pipeline_runner import RunnerConfig, run
from test_ai_config import ai_config
from test_ai_pipeline import ai_display, ai_observation
from test_pipeline import RUN_ID, FakeClock, frame, settings
from test_pipeline_runner_display import DisplayLive, RecordingSink
from test_pipeline_runner_live import S, SimpleClock, stop_after_jobs
from test_pipeline_watchdog import watch
from test_ros_image_adapter import image_msg, info_msg
from test_run_pipeline import FakeSource, lines

OFF = RunnerConfig(close_attempts=1, close_retry_interval_s=0.0, first_detect_timeout_s=None, detect_timeout_s=None)
H, W = 4, 6


def ai_settings():
    """AI 설정·프로파일 없음(표시 단계) 처리 설정."""
    return settings(detector_config=ai_config(), profile=None)


def stored_factory(n, holder):
    """n개 합성 프레임(실행 run_id) + 같은 run의 END_OF_STREAM 저장 소스 팩토리(holder["source"])."""
    def factory(issuer, provider):
        """issuer·세대 공급자를 받아 소스를 만든다."""
        items = [frame(issuer.next_seq(), provider(), run_id=issuer.run_id) for _ in range(n)]
        items.append(InputStatusEvent(run_id=issuer.run_id, source_id="src0", status=InputStatus.END_OF_STREAM,
                                      event_monotonic_ns=1, detail="end of list"))
        holder["source"] = FakeSource(items)
        return holder["source"]

    return factory


class FakeLiveSource(FakeSource):
    """poll_input이 있어 실시간으로 취급되는 가짜 소스. `frame_current`는 정해 둔 답을 차례로 돌려준다(다 쓰면 True)."""

    def __init__(self, items, current_answers=()):
        """항목·frame_current 답 목록."""
        super().__init__(items, eos=False)
        self.answers = list(current_answers)
        self.current_calls = 0

    def read(self):
        """항목을 다 쓰면 None(실시간 대기)."""
        if not self.items:
            time.sleep(0.002)
            return None

        return super().read()

    def poll_input(self):
        """입력 경계 없음."""
        return []

    def status_event(self, item):
        """상태 사건 없음."""
        return None

    def input_applied(self, item, generation):
        """무동작."""

    def frame_current(self, frame):
        """정해 둔 답(다 쓰면 True)."""
        self.current_calls += 1
        return self.answers.pop(0) if self.answers else True

    def stats(self):
        """빈 통계."""
        return {}


def drain(h):
    """연결부 제어 메시지를 모두 꺼낸다."""
    return h.take_control(100_000)[0]


def kinds(msgs):
    """메시지 종류 목록."""
    return [m.msg_kind for m in msgs]


class Runner:
    """저장·가짜 실시간 소스로 실행기를 한 번 돌리는 도구. detector는 AI 표시 관측(valid=False)과 표시 결과."""

    def __init__(self, test, *, handoff=True, capacity=64, notify=None, hooks=None):
        """test: 단언용 TestCase, handoff: 연결부를 넘기는지, hooks[job_seq](runner): 검출 중 동작."""
        self.test, self.hooks = test, hooks or {}
        self.h = AToCHandoff(capacity=capacity, notify=notify) if handoff else None
        self.holder = {}
        self.mirror_checks = 0
        self.mirror_errors = []
        self.detect_calls = []

    def detector(self, job):
        """비교 경계(직전 handle 끝)에서 거울을 확인하고 hook을 부른 뒤 결과를 돌려준다."""
        self.detect_calls.append(job.job_seq)
        self.check_mirror()
        hook = self.hooks.get(job.job_seq)

        if hook is not None:
            hook(self)

        return ai_observation(job), ai_display(job)

    def check_mirror(self):
        """코어 현재 관측이 없으면 거울도 None, 있으면 같은 작업이다(어긋남은 `mirror_errors`에 남김).

        worker 스레드에서 불릴 수 있어 단언하지 않는다.
        """
        if self.h is None or self.pipeline is None:
            return

        cur = self.h.current_result
        obs = self.pipeline.current_observation

        if obs is None and cur is not None:
            self.mirror_errors.append(("core None but mirror", cur))
        elif obs is not None and cur is not None and (cur[1], cur[2]) != (obs.generation, obs.job_seq):
            self.mirror_errors.append(("mismatch", cur, (obs.generation, obs.job_seq)))

        self.mirror_checks += 1

    pipeline = None

    def run(self, factory, *, config=OFF, prepare=None, should_stop=None, display_sink=None, run_id=RUN_ID,
            stream=None):
        """실행 한 번. (결과, 기록 목록, stderr)."""
        from unittest import mock

        from eyetoarm.vision import pipeline_runner as pr

        stream = stream if stream is not None else io.StringIO()
        err = io.StringIO()
        real = pr.VisionPipeline

        def capture(*a, **kw):
            """코어를 잡는다."""
            self.pipeline = real(*a, **kw)
            return self.pipeline

        kw = {} if self.h is None else {"handoff_sink": self.h}

        if display_sink is not None:
            kw["display_sink"] = display_sink

        with mock.patch.object(pr, "VisionPipeline", side_effect=capture):
            res = run(factory, ai_settings(), PipelineLog(JsonlRecorder(stream)), run_id=run_id,
                      detector=self.detector, config=config, stderr=err, clock_ns=FakeClock(), prepare=prepare,
                      should_stop=should_stop, **kw)

        return res, lines(stream), err.getvalue()

    def assert_event_balance(self, res):
        """실행 끝 사건 보존식(마지막 sync 확정): enqueued + unsent = len(records), applied/ignored는 넘긴 기록 기준."""
        s = self.h.stats()
        records = self.pipeline.invalidation_records
        sent = records[:s.events_enqueued]

        self.test.assertEqual(s.run_id, self.pipeline.run_id)
        self.test.assertEqual(s.records_seen, len(records))
        self.test.assertEqual(s.events_enqueued + s.events_unsent, len(records))
        self.test.assertEqual(s.events_applied, sum(r.applied for r in sent))
        self.test.assertEqual(s.events_ignored, sum(not r.applied for r in sent))
        self.test.assertEqual(res.handoff["events_unsent"], s.events_unsent)

    def invalidations(self, msgs):
        """INVALIDATION 메시지의 (kind, job_seq, cause, applied)."""
        return [(m.event.kind, m.event.job_seq, m.event.cause, m.applied) for m in msgs
                if m.msg_kind is MsgKind.INVALIDATION]


class DefaultOffTest(unittest.TestCase):
    """기본 handoff_sink=None: 기존 기록·계수·종료 코드 그대로."""

    def test_same_records_with_and_without_handoff(self):
        """연결부를 넘겨도 JSONL 기록·계수·종료 코드가 넘기지 않은 실행과 같다(연결부는 코어 시계를 쓰지 않음)."""
        off = Runner(self, handoff=False)
        res0, recs0, err0 = off.run(stored_factory(3, off.holder))
        on = Runner(self)
        res1, recs1, err1 = on.run(stored_factory(3, on.holder))

        self.assertEqual((res0.exit_code, res1.exit_code), (0, 0), (err0, err1))
        self.assertEqual(recs0, recs1)
        self.assertEqual(res0.counters, res1.counters)
        self.assertIsNone(res0.handoff)
        self.assertIsNotNone(res1.handoff)


class NormalFlowTest(unittest.TestCase):
    """정상 완료·무검출·비교 경계 거울·RUN_END."""

    def test_results_in_order_and_run_end(self):
        """저장 3장: RUN_START → RESULT×3 → INPUT_STATUS(EOS) → RUN_END, 경계마다 거울 = 현재 관측."""
        r = Runner(self)
        res, recs, err = r.run(stored_factory(3, r.holder))
        msgs = drain(r.h)

        self.assertEqual(res.exit_code, 0, err)
        self.assertEqual(kinds(msgs), [MsgKind.RUN_START] + [MsgKind.RESULT] * 3 + [MsgKind.INPUT_STATUS, MsgKind.RUN_END])
        self.assertIs(msgs[4].status, InputStatus.END_OF_STREAM)
        self.assertEqual([m.job_seq for m in msgs if m.msg_kind is MsgKind.RESULT], [0, 1, 2])
        self.assertTrue(all(m.display_only and not m.observation.valid for m in msgs if m.msg_kind is MsgKind.RESULT))
        self.assertEqual((msgs[-1].exit_code, msgs[-1].stop_reason), (0, "end_of_stream"))
        self.assertEqual(r.mirror_checks, 3)
        self.assertEqual(r.mirror_errors, [])
        self.assertIs(r.h.state().state, HandoffState.OPEN)
        self.assertIsNone(r.h.current_result)              # RUN_END가 해제
        self.assertEqual(res.handoff["run_end_sent"], True)
        r.assert_event_balance(res)

    def test_display_sink_together(self):
        """display_sink와 함께 써도 둘 다 같은 응답을 받는다(표시 1회·RESULT 1회)."""
        r = Runner(self)
        sink = RecordingSink(io.StringIO())
        res, _, err = r.run(stored_factory(1, r.holder), display_sink=sink)
        msgs = drain(r.h)

        self.assertEqual(res.exit_code, 0, err)
        self.assertEqual([c[3] for c in sink.calls if c[0] == "outcome"], [True])
        self.assertEqual(kinds(msgs).count(MsgKind.RESULT), 1)
        self.assertEqual(sink.bridge.shown, 1)

    def test_slow_consumer_does_not_block(self):
        """수신 측이 꺼내지 않아도 검출이 기다리지 않고 영상 칸은 1장, 대기열은 실행 길이만큼만(용량 안)."""
        r = Runner(self)
        res, _, err = r.run(stored_factory(20, r.holder))
        s = r.h.stats()

        self.assertEqual(res.exit_code, 0, err)
        self.assertEqual(len(r.detect_calls), 20)
        self.assertEqual(s.view_pending, 0)                # RUN_END가 해제
        self.assertEqual((s.views_put, s.views_replaced), (20, 19))
        self.assertEqual(s.pending, 23)                    # RUN_START + RESULT×20 + EOS + RUN_END


class WithheldTest(unittest.TestCase):
    """코어 complete 뒤 수신 epoch가 현재가 아니면 RESULT 대신 CLEAR(result_withheld)."""

    def test_withheld_clear(self):
        """job1: 지점 C 확인은 True, 공개 판정 frame_current는 False → CLEAR."""
        r = Runner(self)
        box = {}

        def factory(issuer, provider):
            """job0: [C True, 공개 True], job1: [C True, 공개 False]."""
            box["src"] = FakeLiveSource([frame(issuer.next_seq(), provider()) for _ in range(2)],
                                        current_answers=[True, True, True, False])
            return box["src"]

        def should_stop():
            """작업 2개가 끝나면 정지."""
            c = r.pipeline.counters
            return c.completed + c.failed + c.aborted >= 2

        res, _, err = r.run(factory, should_stop=should_stop)
        msgs = drain(r.h)

        self.assertEqual(res.exit_code, 0, err)
        self.assertEqual(kinds(msgs), [MsgKind.RUN_START, MsgKind.RESULT, MsgKind.CLEAR, MsgKind.RUN_END])
        self.assertEqual(msgs[2].reason, "result_withheld")
        self.assertFalse(r.h.is_current(msgs[1].msg_seq))
        self.assertEqual(box["src"].current_calls, 4)
        self.assertEqual(r.h.stats().results_offered, 1)
        self.assertEqual(r.mirror_errors, [])


class EventFlowTest(unittest.TestCase):
    """사건 전달 순서·정확히 1회."""

    def test_inference_error_invalidation(self):
        """검출 예외: RESULT(job0) → INVALIDATION(INFERENCE_ERROR) → INPUT_STATUS(EOS) → RUN_END(1)."""
        def boom(runner):
            """검출 실패."""
            raise RuntimeError("boom")

        r = Runner(self, hooks={1: boom})
        res, _, err = r.run(stored_factory(2, r.holder))
        msgs = drain(r.h)

        self.assertEqual(res.exit_code, 1)
        self.assertEqual(kinds(msgs), [MsgKind.RUN_START, MsgKind.RESULT, MsgKind.INVALIDATION, MsgKind.INPUT_STATUS,
                                       MsgKind.RUN_END])
        self.assertEqual(r.invalidations(msgs), [(InvalidationKind.INFERENCE_ERROR, 1, "RuntimeError: boom", True)])
        self.assertEqual((msgs[-1].exit_code, msgs[-1].stop_reason), (1, "end_of_stream"))
        r.assert_event_balance(res)

    def test_detect_timeout_and_late_result(self):
        """시간 초과: INVALIDATION(JOB_ABORTED detect_timeout) 1건, 늦은 결과는 RESULT로 오지 않는다."""
        r = Runner(self, hooks={0: lambda runner: time.sleep(0.15)})
        res, _, err = r.run(stored_factory(1, r.holder), config=watch(0.05, 0.05, grace=1.0, interval=0.01))
        msgs = drain(r.h)

        self.assertEqual((res.exit_code, res.timeouts, res.late_results), (1, 1, 1), err)
        self.assertEqual(r.invalidations(msgs), [(InvalidationKind.JOB_ABORTED, 0, "detect_timeout", True)])
        self.assertNotIn(MsgKind.RESULT, kinds(msgs))
        r.assert_event_balance(res)

    def test_model_prepare_failure_frameless(self):
        """모델 준비 실패: RUN_START → INVALIDATION(MODEL_LOAD_FAILED) → RUN_END(1). 프레임 없음."""
        def prepare():
            """준비 실패."""
            raise ModelPrepareError(InvalidationKind.MODEL_LOAD_FAILED, "weights missing")

        r = Runner(self)
        res, _, err = r.run(stored_factory(2, r.holder), prepare=prepare)
        msgs = drain(r.h)

        self.assertEqual((res.exit_code, res.stop_reason), (1, "model_load_failed"), err)
        self.assertEqual(kinds(msgs), [MsgKind.RUN_START, MsgKind.INVALIDATION, MsgKind.RUN_END])
        ev = msgs[1].event
        self.assertEqual((ev.kind, ev.job_seq, ev.image_seq, msgs[1].image_seq), (InvalidationKind.MODEL_LOAD_FAILED,
                                                                                None, None, None))
        self.assertEqual((msgs[2].exit_code, msgs[2].stop_reason), (1, "model_load_failed"))
        self.assertEqual(r.detect_calls, [])
        r.assert_event_balance(res)

    def test_close_abort_event_sent_once(self):
        """검출 대기 중 예외면 코어 close의 JOB_ABORTED(closed)를 정확히 1회 넘긴 뒤 RUN_END."""
        r = Runner(self, hooks={0: lambda runner: time.sleep(0.1)})
        def should_stop():
            """검출 대기 회차에서 예외."""
            if r.detect_calls:
                raise RuntimeError("stop check broke")

            return False

        res, _, err = r.run(stored_factory(1, r.holder), config=watch(5.0, 5.0, grace=1.0, interval=0.01),
                            should_stop=should_stop)
        msgs = drain(r.h)

        self.assertEqual((res.exit_code, res.stop_reason), (1, "error"), err)
        self.assertEqual(r.invalidations(msgs), [(InvalidationKind.JOB_ABORTED, 0, "closed", True)])
        self.assertEqual(kinds(msgs)[-1], MsgKind.RUN_END)
        self.assertEqual(len(r.pipeline.invalidation_records), 1)
        r.assert_event_balance(res)


class LiveBoundaryTest(unittest.TestCase):
    """실제 LiveLinkSource(가짜 수신): 입력 경계 → INPUT_STATUS·INVALIDATION 순서."""

    def run_live(self, live, h):
        """DisplayLive를 연결부와 함께 돌린다(display_sink 없음). DisplayLive가 부르는 run에 handoff_sink를 더한다."""
        from unittest import mock

        import test_pipeline_runner_display as tdisp

        def patched(*a, **kw):
            """handoff_sink를 더해 부른다."""
            kw["handoff_sink"] = h
            return run(*a, **kw)

        with mock.patch.object(tdisp, "run", side_effect=patched):
            return live.run()

    def test_frame_timeout_during_detection(self):
        """검출 중 수신 중단: INPUT_STATUS(FRAME_TIMEOUT) → INVALIDATION(input_boundary), RESULT 없음."""
        h = AToCHandoff()
        live = DisplayLive(with_sink=False, feed=lambda s, c: s.handle_image(image_msg(1)),
                           hooks={0: lambda lv: setattr(lv.clock, "now", lv.clock.now + 2 * S)})
        live.should_stop = stop_after_jobs(live, 1)
        res, _, err = self.run_live(live, h)
        msgs = drain(h)

        self.assertEqual(res.exit_code, 0, err)
        ks = [k for k in kinds(msgs) if k is not MsgKind.RUN_START and k is not MsgKind.RUN_END]
        self.assertEqual(ks, [MsgKind.INPUT_STATUS, MsgKind.INVALIDATION])
        inv = [m for m in msgs if m.msg_kind is MsgKind.INVALIDATION][0]
        self.assertEqual((inv.event.kind, inv.event.cause.split(":")[0]), (InvalidationKind.JOB_ABORTED,
                                                                         "input_boundary"))

    def test_reconnect_generation_changed_then_result(self):
        """수신 중단 → 재접속: GENERATION_CHANGED INVALIDATION 뒤 새 세대 RESULT만 현재."""
        def hook(lv):
            """검출 중 끊김 뒤 새 영상."""
            lv.clock.now += 2 * S
            lv.src.handle_image(image_msg(5))

        h = AToCHandoff()
        live = DisplayLive(with_sink=False, feed=lambda s, c: s.handle_image(image_msg(1)), hooks={0: hook})
        live.should_stop = stop_after_jobs(live, 2)
        res, _, err = self.run_live(live, h)
        msgs = drain(h)

        self.assertEqual(res.exit_code, 0, err)
        gen = [m for m in msgs if m.msg_kind is MsgKind.INVALIDATION
               and m.event.kind is InvalidationKind.GENERATION_CHANGED]
        results = [m for m in msgs if m.msg_kind is MsgKind.RESULT]
        self.assertEqual(len(gen), 1)
        self.assertEqual([m.generation for m in results], [1])
        self.assertLess(msgs.index(gen[0]), msgs.index(results[0]))


class HandoffLive(DisplayLive):
    """실제 LiveLinkSource(가짜 수신) + 연결부. `input_applied`를 감싸 비교 경계의 상태를 기록한다.

    실행기는 sync_records 뒤 input_applied를 부르므로 그 순간이 비교 경계다.
    """

    def __init__(self, h, **kw):
        """연결부 h를 받는다. display_sink는 쓰지 않는다."""
        super().__init__(with_sink=False, **kw)
        self.h = h
        self.boundaries = []
        self.taken_views = []

    def factory(self, issuer, generation_provider):
        """소스를 만들고 input_applied를 감싼다."""
        src = super().factory(issuer, generation_provider)
        original = src.input_applied

        def observed(item, generation):
            """반영 경계의 상태를 남긴 뒤 원래 동작."""
            p = self.pipeline
            self.boundaries.append({
                "kind": item.kind.value, "generation": generation, "current_result": self.h.current_result,
                "core_none": p.current_observation is None,
                "taken_current": [self.h.is_current(v.msg_seq) for v in self.taken_views]})
            original(item, generation)

        src.input_applied = observed
        return src

    def take_view_now(self):
        """수신 측처럼 지금 영상 칸을 꺼내 보관한다(검출 hook에서 부름)."""
        view, _ = self.h.take_view()

        if view is not None:
            self.taken_views.append(view)

        return view


class InputBoundaryHandoffTest(unittest.TestCase):
    """입력 반영 경계에서 되감기·프로필 변경·수신 중단 뒤 연결부·코어가 함께 비었는지와 새 RESULT."""

    def run_live(self, live):
        """HandoffLive를 실행한다(DisplayLive가 부르는 run에 handoff_sink를 더함)."""
        from unittest import mock

        import test_pipeline_runner_display as tdisp

        def patched(*a, **kw):
            """handoff_sink를 더해 부른다."""
            kw["handoff_sink"] = live.h
            return run(*a, **kw)

        with mock.patch.object(tdisp, "run", side_effect=patched):
            return live.run()

    def boundary(self, live, kind):
        """그 종류의 반영 경계 기록(정확히 1개)."""
        found = [b for b in live.boundaries if b["kind"] == kind]
        self.assertEqual(len(found), 1, live.boundaries)
        return found[0]

    def assert_cleared(self, live, b):
        """경계에서 연결부 현재 결과 None·코어 현재 None·이미 꺼낸 영상은 현재 아님(꺼낸 영상이 있어야 함)."""
        self.assertIsNone(b["current_result"], b)
        self.assertTrue(b["core_none"], b)
        self.assertTrue(b["taken_current"], b)
        self.assertEqual(set(b["taken_current"]), {False}, b)

    def assert_end_balance(self, live, res):
        """실행 끝 사건 보존식."""
        s = live.h.stats()
        records = live.pipeline.invalidation_records
        sent = records[:s.events_enqueued]

        self.assertEqual(s.events_enqueued + s.events_unsent, len(records))
        self.assertEqual(s.events_unsent, 0)
        self.assertEqual(s.events_applied, sum(r.applied for r in sent))
        self.assertEqual(res.handoff["run_end_sent"], True)

    def test_sim_time_rewind(self):
        """되감기: 경계에서 모두 비고 job1은 재공개 없음, job2는 GENERATION_CHANGED 뒤 세대 1."""
        def feed(src, clock):
            """sim 10 s 시계·영상."""
            src.handle_clock(SimpleClock(10))
            src.handle_image(image_msg(9, 900_000_000))

        def hook0(lv):
            """job1용 다음 영상."""
            lv.src.handle_image(image_msg(9, 950_000_000))

        def hook1(lv):
            """job0 영상을 꺼낸 뒤 되감기, 반영될 시간을 주고 새 영상."""
            self.assertIsNotNone(lv.take_view_now())
            lv.src.handle_clock(SimpleClock(1))
            time.sleep(0.1)
            lv.src.handle_image(image_msg(1, 100_000_000))

        live = HandoffLive(AToCHandoff(), feed=feed, hooks={0: hook0, 1: hook1})
        live.should_stop = stop_after_jobs(live, 3)
        res, _, err = self.run_live(live)
        msgs = drain(live.h)

        self.assertEqual(res.exit_code, 0, err)
        self.assert_cleared(live, self.boundary(live, "sim_time_rewind"))
        results = [m for m in msgs if m.msg_kind is MsgKind.RESULT]
        self.assertEqual([(m.job_seq, m.generation) for m in results], [(0, 0), (2, 1)])
        gen = [m for m in msgs if m.msg_kind is MsgKind.INVALIDATION
               and m.event.kind is InvalidationKind.GENERATION_CHANGED]
        self.assertEqual([(m.event.cause, m.event.generation) for m in gen], [(GenerationCause.SIM_TIME_REWIND.value, 1)])
        self.assertLess(msgs.index(results[0]), msgs.index(gen[0]))
        self.assertLess(msgs.index(gen[0]), msgs.index(results[1]))
        self.assertEqual((results[1].capture_generation, results[1].timestamp_ms), (1, 1100))
        self.assert_end_balance(live, res)

    def test_profile_changed(self):
        """프로필 변경: 경계에서 모두 비고 새 RESULT는 세대 1·새 calibration_id·같은 검출 설정 ID."""
        def hook0(lv):
            """job1용 다음 영상."""
            lv.src.handle_image(image_msg(2))

        def hook1(lv):
            """job0 영상을 꺼낸 뒤 프로필 변경, 반영될 시간을 주고 새 영상."""
            self.assertIsNotNone(lv.take_view_now())
            self.assertTrue(lv.src.handle_camera_info(info_msg(fx=101.0)))
            time.sleep(0.1)
            lv.src.handle_image(image_msg(3))

        live = HandoffLive(AToCHandoff(), feed=lambda s, c: s.handle_image(image_msg(1)), hooks={0: hook0, 1: hook1})
        live.should_stop = stop_after_jobs(live, 3)
        res, _, err = self.run_live(live)
        msgs = drain(live.h)

        self.assertEqual(res.exit_code, 0, err)
        self.assert_cleared(live, self.boundary(live, "profile_changed"))
        results = [m for m in msgs if m.msg_kind is MsgKind.RESULT]
        self.assertEqual([(m.job_seq, m.generation) for m in results], [(0, 0), (2, 1)])
        gen = [m for m in msgs if m.msg_kind is MsgKind.INVALIDATION
               and m.event.kind is InvalidationKind.GENERATION_CHANGED]
        self.assertEqual([m.event.cause.split("+")[0] for m in gen], [GenerationCause.PROFILE_CHANGED.value])
        self.assertLess(msgs.index(gen[0]), msgs.index(results[1]))
        settings_now = live.pipeline.settings
        self.assertEqual(results[1].calibration_id, settings_now.profile.calibration_id)
        self.assertIsNotNone(results[1].calibration_id)
        self.assertNotEqual(results[1].calibration_id, results[0].calibration_id)
        self.assertEqual(results[1].detector_config_id, settings_now.detector_config_id)
        self.assertEqual(results[1].detector_config_id, results[0].detector_config_id)
        self.assert_end_balance(live, res)

    def test_frame_timeout_after_result(self):
        """수신 중단: 경계에서 모두 비고 job1은 input_boundary 중단(재공개 없음)."""
        def hook0(lv):
            """job1용 다음 영상."""
            lv.src.handle_image(image_msg(2))

        def hook1(lv):
            """job0 영상을 꺼낸 뒤 수신 중단(2 s)."""
            self.assertIsNotNone(lv.take_view_now())
            lv.clock.now += 2 * S
            time.sleep(0.1)

        live = HandoffLive(AToCHandoff(), feed=lambda s, c: s.handle_image(image_msg(1)), hooks={0: hook0, 1: hook1})
        live.should_stop = stop_after_jobs(live, 2)
        res, _, err = self.run_live(live)
        msgs = drain(live.h)

        self.assertEqual(res.exit_code, 0, err)
        self.assert_cleared(live, self.boundary(live, "frame_timeout"))
        body = [m.msg_kind for m in msgs if m.msg_kind not in (MsgKind.RUN_START, MsgKind.RUN_END)]
        self.assertEqual(body, [MsgKind.RESULT, MsgKind.INPUT_STATUS, MsgKind.INVALIDATION])
        inv = [m for m in msgs if m.msg_kind is MsgKind.INVALIDATION][0]
        self.assertEqual((inv.event.job_seq, inv.event.cause.split(":")[0]), (1, "input_boundary"))
        self.assertEqual([m.job_seq for m in msgs if m.msg_kind is MsgKind.RESULT], [0])
        self.assert_end_balance(live, res)


class FailStopTest(unittest.TestCase):
    """fail-stop: 수신 측 실패·넘침·notify 예외 → handoff_failed 종료 1, 유한 정리, unsent 확정."""

    def assert_failed_cleanup(self, r, res, err, *, stop_reason="handoff_failed"):
        """종료 1·정지 사유·코어 닫힘·소스 close·RUN_END 없음·unsent 확정."""
        self.assertEqual((res.exit_code, res.stop_reason), (1, stop_reason), err)
        self.assertTrue(res.closed)
        self.assertTrue(any("handoff failed" in e for e in res.errors), res.errors)
        self.assertNotIn(MsgKind.RUN_END, kinds(drain(r.h)))
        self.assertFalse(res.handoff["run_end_sent"])
        r.assert_event_balance(res)

    def test_consumer_failure_without_input(self):
        """실시간 무입력 대기 중 수신 측 실패 → 입력 대기 지점 check로 유한 종료 1(handoff_failed)."""
        r = Runner(self)
        box = {}

        def factory(issuer, provider):
            """영상이 오지 않는 실시간 소스."""
            box["src"] = FakeLiveSource([])
            return box["src"]

        t = threading.Timer(0.1, lambda: r.h.report_consumer_failure("socket closed"))
        t.start()
        started = time.monotonic()
        res, _, err = r.run(factory)
        t.join()

        self.assertLess(time.monotonic() - started, 5.0)
        self.assertIn("consumer: socket closed", err)
        self.assertEqual(box["src"].closes, 1)
        self.assert_failed_cleanup(r, res, err)

    def test_consumer_failure_during_detection(self):
        """검출 대기 중 수신 측 실패 → on_idle check로 종료 1. 활성 작업은 close 중단 사건이 되지만 FAILED라 미전달(unsent 1)."""
        def hook(runner):
            """검출 중 수신 측 실패, 그 뒤 잠시 더 검출."""
            runner.h.report_consumer_failure("send error")
            time.sleep(0.1)

        r = Runner(self, hooks={0: hook})
        res, _, err = r.run(stored_factory(2, r.holder), config=watch(5.0, 5.0, grace=1.0, interval=0.01))

        self.assert_failed_cleanup(r, res, err)
        self.assertEqual(r.h.stats().events_unsent, 1)
        self.assertEqual(r.pipeline.invalidation_records[0].event.cause, "closed")
        self.assertEqual(r.holder["source"].closes, 1)

    def test_overflow_keeps_original_error(self):
        """모델 준비 실패 중 넘침: 정지 사유 model_load_failed 유지, handoff 실패도 errors에, unsent 1."""
        def prepare():
            """준비 실패."""
            raise ModelPrepareError(InvalidationKind.MODEL_LOAD_FAILED, "weights missing")

        r = Runner(self, capacity=1)
        res, _, err = r.run(stored_factory(1, r.holder), prepare=prepare)

        self.assert_failed_cleanup(r, res, err, stop_reason="model_load_failed")
        self.assertTrue(any("model prepare failed" in e for e in res.errors))
        self.assertEqual(r.h.state().reason, "overflow")
        self.assertEqual(r.h.stats().events_unsent, 1)
        self.assertEqual(r.holder["source"].closes, 1)

    def test_overflow_during_frames(self):
        """프레임 처리 중 넘침(용량 2): handoff_failed로 멈추고 남은 프레임은 처리하지 않는다."""
        r = Runner(self, capacity=2)
        res, _, err = r.run(stored_factory(5, r.holder))

        self.assert_failed_cleanup(r, res, err)
        self.assertEqual(len(r.detect_calls), 2)

    def test_notify_exception(self):
        """notify 예외(RUN_START에서) → 프레임을 읽지 않고 handoff_failed 종료 1."""
        def notify():
            """항상 실패."""
            raise RuntimeError("loop closed")

        r = Runner(self, notify=notify)
        res, _, err = r.run(stored_factory(2, r.holder))

        self.assert_failed_cleanup(r, res, err)
        self.assertEqual(r.detect_calls, [])

    def test_closed_handoff(self):
        """이미 닫힌 연결부로 실행하면 RUN_START에서 handoff_failed 종료 1(프레임 미처리)."""
        r = Runner(self)
        r.h.close()
        res, _, err = r.run(stored_factory(1, r.holder))

        self.assertEqual((res.exit_code, res.stop_reason), (1, "handoff_failed"), err)
        self.assertEqual(r.detect_calls, [])


class FinalStateTest(unittest.TestCase):
    """종료 보고 확정 시점(`_finish_handoff`)의 연결부가 FAILED·CLOSED면 종료 1.

    그 뒤 새로 생긴 외부 실패는 RunResult에 반영되지 않는다.
    """

    def run_with_close_hook(self, action):
        """소스 close 뒤 action(runner)을 부르는 실행. (runner, 결과, stderr)."""
        r = Runner(self)
        base = stored_factory(1, r.holder)

        def factory(issuer, provider):
            """소스 close를 감싼다."""
            src = base(issuer, provider)
            original = src.close

            def close():
                """원래 close 뒤 연결부 실패를 일으킨다."""
                original()
                action(r)

            src.close = close
            return src

        res, _, err = r.run(factory)

        return r, res, err

    def assert_final_failure(self, r, res, err, state, reason_part):
        """종료 1, 원래 정지 사유(end_of_stream) 유지, 연결부 사유가 errors·stderr·handoff 보고에 있음, 보존식 유지."""
        self.assertEqual((res.exit_code, res.stop_reason), (1, "end_of_stream"), err)
        self.assertEqual(res.handoff["state"], state)
        self.assertTrue(any(e.startswith("handoff failed") and reason_part in e for e in res.errors), res.errors)
        self.assertIn(reason_part, err)
        self.assertTrue(res.closed)
        self.assertEqual(r.holder["source"].closes, 1)
        r.assert_event_balance(res)

    def test_a_consumer_failure_after_source_close(self):
        """A: 소스 close 뒤 수신 측 실패(새 사건 없음) → 종료 1, RUN_END 미전달."""
        r, res, err = self.run_with_close_hook(lambda r: r.h.report_consumer_failure("late send failure"))

        self.assert_final_failure(r, res, err, "failed", "consumer: late send failure")
        self.assertFalse(res.handoff["run_end_sent"])

    def test_b_unexpected_close(self):
        """B: 실행 중 호출자가 연결부를 닫음(새 사건 없음) → 종료 1, RUN_END 미전달."""
        r, res, err = self.run_with_close_hook(lambda r: r.h.close())

        self.assert_final_failure(r, res, err, "closed", "closed")
        self.assertFalse(res.handoff["run_end_sent"])

    def test_c_run_end_delivery_failure(self):
        """C: RUN_END를 넣은 뒤 notify 안에서 수신 측이 전달 실패를 latch → run_end_sent True 그대로, 종료 1."""
        box = {}

        def notify():
            """모두 꺼내고 RUN_END를 받으면 전달 실패를 알린다."""
            h = box["r"].h
            msgs, _ = h.take_control(100)
            h.take_view()

            if any(m.msg_kind is MsgKind.RUN_END for m in msgs):
                h.report_consumer_failure("RUN_END delivery failed")

        r = Runner(self, notify=notify)
        box["r"] = r
        res, _, err = r.run(stored_factory(1, r.holder))

        self.assert_final_failure(r, res, err, "failed", "consumer: RUN_END delivery failed")
        self.assertTrue(res.handoff["run_end_sent"])

    def test_open_success_stays_zero(self):
        """정상 OPEN·RUN_END 성공은 종료 0, handoff 오류 없음."""
        r = Runner(self)
        res, _, err = r.run(stored_factory(1, r.holder))

        self.assertEqual(res.exit_code, 0, err)
        self.assertEqual(res.handoff["state"], "open")
        self.assertFalse(any("handoff" in e for e in res.errors), res.errors)


class ReuseTest(unittest.TestCase):
    """같은 연결부를 두 실행에 쓴다(실행별 사건 계수·msg_seq 단조)."""

    def test_two_runs(self):
        """실행1(실패 사건 1) → 실행2(정상): 두 RUN_START·RUN_END, 실행2 사건 계수 0, msg_seq 단조."""
        def boom(runner):
            """검출 실패."""
            raise RuntimeError("boom")

        r = Runner(self, hooks={0: boom})
        r.run(stored_factory(1, r.holder))
        r.hooks = {}
        res2, _, err2 = r.run(stored_factory(1, r.holder), run_id="run-two")
        msgs = drain(r.h)

        self.assertEqual(res2.exit_code, 0, err2)
        self.assertEqual([m.msg_seq for m in msgs], list(range(1, len(msgs) + 1)))
        self.assertEqual(kinds(msgs).count(MsgKind.RUN_START), 2)
        self.assertEqual(r.h.stats().events_enqueued, 0)
        self.assertEqual(r.h.stats().events_enqueued_total, 1)
        r.assert_event_balance(res2)


if __name__ == "__main__":
    unittest.main()
