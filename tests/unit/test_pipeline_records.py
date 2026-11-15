"""작업 5 처리 기록 단위시험 — JSONL 1회 기록·원본 시각/시작 설정 보존·eval_run_id·로그 오류 분리·입력 상태 지속 시간.

근거: 계획 r4 작업 5, A 상세설계 §9(처리 기록·입력 상태 로그·처리 건수·평가 실행 식별), §5.4,
작업 4 합의(103 §4: duplicate 응답은 재기록하지 않음). 관측은 시험이 만든 가짜 detector 결과다.
"""
import dataclasses
import io
import json
import unittest

from eyetoarm.common.types import CandidateInvalidationEvent, InputStatus, InputStatusEvent, InvalidationKind
from eyetoarm.vision.pipeline import GenerationCause, VisionPipeline
from eyetoarm.vision.pipeline_records import (
    FORMAT_VERSION,
    InputStateTracker,
    JsonlRecorder,
    PipelineLog,
    RecorderFailedError,
    process_record,
)
from eyetoarm.vision.pipeline_types import JobStatus, QueueDropReason
from test_pipeline import CHANGES, RUN_ID, FakeClock, frame, observation, settings
from test_pipeline_terminal import valid_observation

INF = InvalidationKind.INFERENCE_ERROR


def lines(stream):
    """스트림에 쓴 JSONL을 dict 목록으로 읽는다(엄격 JSON: NaN 등 거부)."""
    def reject(value):
        """비유한 상수를 거부한다."""
        raise ValueError(value)

    return [json.loads(text, parse_constant=reject) for text in stream.getvalue().splitlines()]


def kinds(stream):
    """기록 종류 목록."""
    return [r["kind"] for r in lines(stream)]


class FailingStream:
    """write가 OSError를 내는 스트림(디스크 가득 등 모사)."""

    def write(self, text):
        """항상 실패한다."""
        raise OSError("disk full")

    def flush(self):
        """아무것도 하지 않는다."""


class FlushFailStream(io.StringIO):
    """write는 줄을 받아 두지만 flush가 실패하는 스트림(기록 여부 불확실)."""

    def flush(self):
        """항상 실패한다."""
        raise OSError("flush failed")


class ShortWriteStream(io.StringIO):
    """줄 끝 5글자를 버리고 짧은 길이를 돌려주는 스트림."""

    def write(self, text):
        """일부만 쓴다."""
        super().write(text[:-5])
        return len(text) - 5


class FailOnNthWrite(io.StringIO):
    """n번째 write만 기록 전에 실패하는 스트림."""

    def __init__(self, n):
        """실패할 순번을 받는다."""
        super().__init__()
        self.n, self.calls = n, 0

    def write(self, text):
        """n번째 호출만 실패한다."""
        self.calls += 1
        if self.calls == self.n:
            raise OSError("write failed")
        return super().write(text)


class LogTestBase(unittest.TestCase):
    """파이프라인·기록기 준비."""

    def setUp(self):
        """가짜 시계 파이프라인과 메모리 스트림 기록기를 만든다."""
        self.p = VisionPipeline(settings(), run_id=RUN_ID, clock_ns=FakeClock())
        self.stream = io.StringIO()
        self.rec = JsonlRecorder(self.stream)
        self.log = PipelineLog(self.rec)

    def start(self, seq):
        """현재 세대 프레임으로 작업을 시작한다."""
        f = frame(seq, self.p.current_generation(), source_id=self.p.settings.source_id)
        self.assertTrue(self.p.offer_frame(f))
        return self.p.start_next()


class ProcessRecordTest(LogTestBase):
    """처리 기록 내용(A §9 「처리 기록」)."""

    def test_records_keep_original_time_and_start_config(self):
        """세대 폐기 기록은 시작 당시 설정 ID·세대·원본 시각을 쓰고, 무효 완료 기록은 reason·후보 수·처리 시간을 옮기며 명령 ID는 None이다."""
        job = self.start(0)
        old_cfg = job.detector_config_id
        self.p.change_settings(CHANGES[GenerationCause.DETECTOR_CONFIG_CHANGED](self.p.settings),
                               cause=GenerationCause.DETECTOR_CONFIG_CHANGED)
        self.assertTrue(self.log.log_outcome(self.p.complete(job, observation(job))))

        nxt = self.start(1)
        self.log.log_outcome(self.p.complete(nxt, observation(nxt)))

        recs = [r for r in lines(self.stream) if r["kind"] == "process"]
        disc, done = recs

        self.assertEqual(disc["status"], JobStatus.DISCARDED_GENERATION.value)
        self.assertEqual(disc["detector_config_id"], old_cfg)
        self.assertNotEqual(disc["detector_config_id"], self.p.settings.detector_config_id)
        self.assertEqual((disc["generation"], disc["capture_generation"], disc["image_seq"]), (0, 0, 0))
        self.assertEqual((disc["timestamp_ms"], disc["timestamp_base"]), (10_000, "SIM_TIME"))
        self.assertEqual((disc["valid"], disc["reason"], disc["candidate_count"]), (None, None, None))

        self.assertEqual(done["status"], "COMPLETED")
        self.assertEqual((done["valid"], done["reason"], done["reason_owner"]), (False, "NO_OBJECT", "A"))
        self.assertEqual((done["candidate_count"], done["process_ms"]), (0, 0.1))
        self.assertEqual(done["detector_config_id"], self.p.settings.detector_config_id)
        self.assertEqual((done["mode"], done["calibration_id"], done["reprocess"]), ("planar", "cal-1", False))
        self.assertIsNone(done["command_id"])

    def test_valid_completion_record_has_no_reason(self):
        """유효 완료 기록은 reason·reason_owner가 None이다."""
        job = self.start(0)
        self.log.log_outcome(self.p.complete(job, valid_observation(job)))
        rec = lines(self.stream)[-1]

        self.assertEqual((rec["valid"], rec["reason"], rec["reason_owner"]), (True, None, None))

    def test_process_record_carries_eval_run_id_when_evaluating(self):
        """평가 실행이면 처리 기록마다 같은 eval_run_id를 담고 실행 메타데이터에도 같은 값이 있다. 평가가 아니면 None이다."""
        stream = io.StringIO()
        log = PipelineLog(JsonlRecorder(stream), eval_run_id="eval-2026-11-15-a")
        log.write_run_meta(self.p, started_monotonic_ns=1)
        job = self.start(0)
        out = self.p.complete(job, observation(job))
        log.log_outcome(out)

        meta, rec = lines(stream)
        self.assertEqual(meta["kind"], "run_meta")
        self.assertEqual(meta["eval_run_id"], "eval-2026-11-15-a")
        self.assertEqual(rec["eval_run_id"], "eval-2026-11-15-a")
        self.assertEqual(meta["detector_config_id"], self.p.settings.detector_config_id)

        # 평가가 아니면 None, 빈 문자열·문자열 아닌 값은 거부(가짜 평가 ID를 만들지 않음)
        self.log.write_run_meta(self.p, started_monotonic_ns=1)
        self.assertIsNone(lines(self.stream)[-1]["eval_run_id"])
        self.assertIsNone(process_record(out).eval_run_id)
        self.assertEqual(process_record(out, eval_run_id="eval-x").eval_run_id, "eval-x")

        # duplicate 응답은 처리 기록으로 만들지 않는다
        with self.assertRaises(ValueError):
            process_record(self.p.complete(job, None))

        for bad in ("", 3):
            with self.subTest(eval_run_id=bad):
                with self.assertRaises(ValueError):
                    PipelineLog(JsonlRecorder(io.StringIO()), eval_run_id=bad)

    def test_jsonl_is_strict_and_has_no_image(self):
        """모든 줄은 엄격 JSON이며 format_version·kind가 있고 영상 배열·윤곽 배열을 담지 않는다."""
        self.log.write_run_meta(self.p, started_monotonic_ns=1)
        job = self.start(0)
        self.log.log_outcome(self.p.complete(job, valid_observation(job)))
        self.assertTrue(self.p.offer_frame(frame(1)))
        self.assertTrue(self.p.offer_frame(frame(2)))
        self.log.sync(self.p)

        for rec in lines(self.stream):
            self.assertEqual(rec["format_version"], FORMAT_VERSION)
            self.assertIn(rec["kind"], ("run_meta", "process", "event", "queue_drop"))
            self.assertNotIn("image", rec)
            self.assertNotIn("calc_contour", rec)

        self.assertLess(max(len(t) for t in self.stream.getvalue().splitlines()), 2_000)


class OnceOnlyTest(LogTestBase):
    """최초 terminal·사건·폐기 1회 기록(계획 r4 작업 5 「1회 기록」)."""

    def test_duplicate_outcome_is_not_rewritten(self):
        """같은 작업의 재통지(duplicate 응답)는 처리 기록·관측·사건으로 다시 쓰지 않는다."""
        job = self.start(0)
        self.assertTrue(self.log.log_outcome(self.p.complete(job, observation(job))))
        before = self.stream.getvalue()

        for again in (self.p.complete(job, observation(job)), self.p.fail(job, INF, "late")):
            self.assertTrue(again.duplicate)
            self.assertFalse(self.log.log_outcome(again))

        self.assertEqual(self.log.sync(self.p), 0)
        self.assertEqual(self.stream.getvalue(), before)

    def test_event_logged_once_across_outcome_and_event_list(self):
        """fail 응답의 사건과 코어 사건 목록의 같은 사건은 한 줄만 쓰고, apply→같은 9필드 fail도 한 줄이다."""
        job = self.start(0)
        out = self.p.fail(job, INF, "boom")
        self.log.log_outcome(out)
        self.assertEqual(self.log.sync(self.p), 0)
        self.assertEqual(kinds(self.stream), ["process", "event"])
        ev = lines(self.stream)[1]
        self.assertEqual((ev["invalidation_kind"], ev["job_seq"], ev["cause"]), ("INFERENCE_ERROR", 0, "boom"))

        # apply 경로로 먼저 들어온 같은 사건(고정 시계)
        p = VisionPipeline(settings(), run_id=RUN_ID, clock_ns=lambda: 7)
        stream = io.StringIO()
        log = PipelineLog(JsonlRecorder(stream))
        self.assertTrue(p.offer_frame(frame(0)))
        job = p.start_next()
        p.apply_invalidation(CandidateInvalidationEvent(
            run_id=RUN_ID, source_id="src0", kind=INF, cause="same", event_monotonic_ns=7, generation=0,
            detector_config_id=job.detector_config_id, job_seq=0, image_seq=0))
        log.sync(p)
        log.log_outcome(p.fail(job, INF, "same"))
        log.sync(p)

        self.assertEqual(kinds(stream), ["event", "process"])
        self.assertEqual(lines(stream)[1]["status"], "FAILED")

    def test_counts_balance_with_log_lines_on_close(self):
        """대기 교체·세대 폐기·실패·재처리·종료를 섞은 뒤 처리 기록 수·폐기 기록 수·사건 기록 수가 코어 계수와 같고 보존식이 성립한다."""
        p, log = self.p, self.log
        j0 = self.start(0)
        self.assertTrue(p.offer_frame(frame(1)))
        self.assertTrue(p.offer_frame(frame(2)))                      # 1 교체 폐기
        log.log_outcome(p.complete(j0, valid_observation(j0)))
        j1 = p.start_next()
        p.change_settings(CHANGES[GenerationCause.DETECTOR_CONFIG_CHANGED](p.settings),
                          cause=GenerationCause.DETECTOR_CONFIG_CHANGED)
        log.log_outcome(p.complete(j1, observation(j1)))              # 세대 폐기
        re_job = p.start_reprocess()
        log.log_outcome(p.fail(re_job, INF, "boom"))                  # 실패
        self.assertFalse(p.offer_frame(frame(3, 0)))                  # 오래된 캡처 폐기
        j4 = self.start(4)
        self.assertTrue(p.offer_frame(frame(5, 1)))
        for out in p.close():                                         # 중단 + 대기 closed 폐기
            log.log_outcome(out)
        log.log_outcome(p.complete(j4, observation(j4)))              # 종료 뒤 재통지(무기록)
        log.sync(p)

        c = p.counters
        k = kinds(self.stream)
        self.assertEqual(c.received, c.first_started + sum(c.queue_dropped.values()))
        self.assertEqual(c.first_started + c.reprocess_started,
                         c.completed + c.generation_discarded + c.failed + c.aborted)
        self.assertEqual(k.count("process"), c.first_started + c.reprocess_started)
        self.assertEqual(k.count("queue_drop"), sum(c.queue_dropped.values()))
        self.assertEqual(k.count("event"), len(p.invalidation_events))
        drop_reasons = sorted(r["reason"] for r in lines(self.stream) if r["kind"] == "queue_drop")
        self.assertEqual(drop_reasons, sorted([QueueDropReason.REPLACED_BY_NEWER.value,
                                               QueueDropReason.STALE_CAPTURE.value, QueueDropReason.CLOSED.value]))


class EnvelopeTest(unittest.TestCase):
    """기록 봉투 키 충돌 방지."""

    def test_envelope_key_clash_is_rejected(self):
        """기록 필드가 format_version·kind와 겹치면 쓰지 않고 ValueError이며 실패 수에도 넣지 않는다(직렬화 단계 오류)."""
        rec = JsonlRecorder(io.StringIO())

        with self.assertRaises(ValueError):
            rec.write("x", {"kind": "y"})

        self.assertEqual((rec.written, rec.failed), (0, 0))


class LogFailureTest(LogTestBase):
    """로그 오류는 기록 오류로 전파하고 영상 처리 실패로 이중 집계하지 않는다."""

    def test_log_failure_is_not_counted_as_inference_failure(self):
        """쓰기 실패는 OSError로 올라오고 코어의 completed·failed 계수는 바뀌지 않으며, 기록기의 쓰기 성공·실패 수가 따로 남는다."""
        rec = JsonlRecorder(FailingStream())
        log = PipelineLog(rec)
        job = self.start(0)
        out = self.p.complete(job, observation(job))
        before = self.p.counters

        with self.assertRaises(OSError):
            log.log_outcome(out)

        self.assertEqual(self.p.counters, before)
        self.assertEqual((before.completed, before.failed), (1, 0))
        self.assertEqual((rec.written, rec.failed), (0, 1))
        self.assertTrue(rec.broken)

    def test_uncertain_write_failure_stops_recording(self):
        """flush 실패·짧은 쓰기는 기록 여부가 불확실하므로 기록기가 실패 상태가 되고, 같은 기록을 다시 쓰지 않아 중복·잘린 줄을 더 만들지 않는다(fail-stop)."""
        for name, stream in (("flush", FlushFailStream()), ("short", ShortWriteStream())):
            with self.subTest(stream=name):
                rec = JsonlRecorder(stream)
                log = PipelineLog(rec)
                job = self.start(int(name == "short"))
                out = self.p.complete(job, observation(job))

                with self.assertRaises(OSError):
                    log.log_outcome(out)

                after_first = stream.getvalue()

                with self.assertRaises(RecorderFailedError) as ctx:
                    log.log_outcome(out)

                self.assertIsInstance(ctx.exception.cause, OSError)

                with self.assertRaises(RecorderFailedError):
                    log.sync(self.p)

                self.assertEqual(stream.getvalue(), after_first)   # 재시도로 줄이 늘지 않음
                self.assertLessEqual(after_first.count("\n"), 1)
                self.assertEqual((rec.written, rec.failed), (0, 1))
                self.assertEqual(log.missing(self.p)["process"], self.p.counters.completed)

    def test_event_failure_after_process_is_reported_not_hidden(self):
        """process 줄은 쓰고 event 쓰기만 기록 전에 실패하면, 재시도·sync는 거부되고 missing이 사건 1건 누락을 보고하며 스트림에는 process 1줄만 남는다."""
        stream = FailOnNthWrite(2)
        log = PipelineLog(JsonlRecorder(stream))
        job = self.start(0)
        out = self.p.fail(job, INF, "boom")

        with self.assertRaises(OSError):
            log.log_outcome(out)

        for call in (lambda: log.log_outcome(out), lambda: log.sync(self.p)):
            with self.assertRaises(RecorderFailedError):
                call()

        self.assertEqual(kinds(stream), ["process"])
        self.assertEqual(log.missing(self.p), {"process": 0, "event": 1, "queue_drop": 0})
        self.assertEqual(self.p.counters.failed, 1)   # 코어 계수는 다시 세지 않음

    def test_missing_is_zero_after_complete_logging(self):
        """실패 없이 모두 기록하면 missing은 전부 0이다."""
        job = self.start(0)
        self.assertTrue(self.p.offer_frame(frame(1)))
        self.log.log_outcome(self.p.fail(job, INF, "boom"))
        for out in self.p.close():
            self.log.log_outcome(out)
        self.log.sync(self.p)

        self.assertEqual(self.log.missing(self.p), {"process": 0, "event": 0, "queue_drop": 0})


class InputStateTest(unittest.TestCase):
    """입력 상태 로그(A §9 「입력 상태 로그」)."""

    def event(self, status, t):
        """입력 사건."""
        return InputStatusEvent(run_id=RUN_ID, source_id="src0", status=status, event_monotonic_ns=t, detail=None)

    def test_input_state_duration_uses_monotonic_clock(self):
        """지속 시간은 같은 PC 단조 시계(event_monotonic_ns·복귀 시각)의 차이이며, 같은 상태 반복은 한 구간, 상태 전환·정상 복귀·종료에서 확정한다."""
        tr = InputStateTracker(RUN_ID, "src0")

        self.assertEqual(tr.on_status(self.event(InputStatus.FRAME_TIMEOUT, 1_000)), [])
        self.assertEqual(tr.on_status(self.event(InputStatus.FRAME_TIMEOUT, 1_500)), [])   # 같은 구간
        (r1,) = tr.on_status(self.event(InputStatus.SOURCE_UNAVAILABLE, 4_000))          # 전환
        (r2,) = tr.on_frame(9_000)                                                       # 정상 복귀
        self.assertEqual(tr.on_frame(9_500), [])
        tr.on_status(self.event(InputStatus.END_OF_STREAM, 12_000))
        (r3,) = tr.finish(12_400)                                                        # 종료

        self.assertEqual((r1.status, r1.event_monotonic_ns, r1.duration_ns), (InputStatus.FRAME_TIMEOUT, 1_000, 3_000))
        self.assertEqual((r2.status, r2.duration_ns), (InputStatus.SOURCE_UNAVAILABLE, 5_000))
        self.assertEqual((r3.status, r3.duration_ns), (InputStatus.END_OF_STREAM, 400))
        self.assertEqual(tr.finish(13_000), [])

    def test_input_event_has_no_fake_frame(self):
        """입력 상태 기록 JSON에는 image_seq·job_seq·timestamp_ms가 없고 다른 run/source 사건은 거부한다."""
        tr = InputStateTracker(RUN_ID, "src0")
        tr.on_status(self.event(InputStatus.DECODE_FAILED, 10))
        stream = io.StringIO()
        log = PipelineLog(JsonlRecorder(stream))
        log.log_input_states(tr.finish(30))
        (rec,) = lines(stream)

        self.assertEqual((rec["kind"], rec["status"], rec["duration_ns"]), ("input_state", "DECODE_FAILED", 20))
        for key in ("image_seq", "job_seq", "timestamp_ms", "generation"):
            self.assertNotIn(key, rec)

        with self.assertRaises(ValueError):
            tr.on_status(dataclasses.replace(self.event(InputStatus.FRAME_TIMEOUT, 40), source_id="other"))


if __name__ == "__main__":
    unittest.main()
