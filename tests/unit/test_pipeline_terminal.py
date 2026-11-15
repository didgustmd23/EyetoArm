"""A 비전 파이프라인 작업 4 단위시험 — 완료·실패·중단·무효화 순서·멱등·종료·보존식·영상 보관 정책.

근거: 계획 `docs/superpowers/plans/2026-11-15-a-vision-pipeline.md` r3 작업 4,
A 상세설계 §5.2(예외·중단은 관측 없음)·§5.4(설정 스냅샷과 폐기·무효화 적용 규칙 1~5)·§9(처리 건수),
공통 자료형 관리 초안 §3.3·§3.5·§4, 작업 4 전달문과 보완(`_workspace/codex-reviews/2026-11-15-pipeline-task4-*.md`,
`_workspace/103_builder_task4_q13_policy.md` §4).

관측은 시험이 직접 만든 가짜 detector 결과다. 실제 B 계산·YOLO·로봇 동작·C/D 전달을 확인하지 않는다.
"""
import dataclasses
import gc
import unittest
import weakref

import numpy as np

from eyetoarm.common.types import (
    CandidateInvalidationEvent,
    FramePacket,
    InvalidationKind,
    PlanarObservation,
)
from eyetoarm.vision.pipeline import (
    ADAPTER_CONTRACT_VIOLATION,
    GenerationCause,
    PipelineContractError,
    VisionPipeline,
)
from eyetoarm.vision.pipeline_types import JobContext, JobOutcome, JobStatus, QueueDropReason
from eyetoarm.vision.planar import PlanarResult, detect_planar
from test_pipeline import (
    CHANGES,
    RUN_ID,
    FakeClock,
    PipelineTestBase,
    frame,
    observation,
    settings,
)

INF = InvalidationKind.INFERENCE_ERROR
ABORT = InvalidationKind.JOB_ABORTED
GEN = InvalidationKind.GENERATION_CHANGED

# 계산용 윤곽으로 쓰는 시험용 사각형(원본 픽셀, float32 (N, 2))
SQUARE = np.array([[1, 1], [4, 1], [4, 3], [1, 3]], dtype=np.float32)


def valid_observation(job):
    """작업 식별자를 그대로 옮긴 시험용 유효 관측(면·계산용 윤곽·선택 번호 있음)."""
    return dataclasses.replace(
        observation(job), observed_face="top_face", calc_contour=SQUARE, candidate_count=1,
        selected_index=0, valid=True, reason=None)


def job_event(job, kind=INF, cause="adapter_error", t=1):
    """작업의 실제 식별자를 가진 외부(어댑터) 사건."""
    return CandidateInvalidationEvent(
        run_id=job.frame.run_id, source_id=job.frame.source_id, kind=kind, cause=cause,
        event_monotonic_ns=t, generation=job.generation, detector_config_id=job.detector_config_id,
        job_seq=job.job_seq, image_seq=job.frame.image_seq)


def frameless_event(p, kind=InvalidationKind.MODEL_LOAD_FAILED, cause="weights missing", t=1):
    """현재 세대·설정의 프레임 없는 로딩·설정 오류 사건(가짜 job_seq·image_seq 없음)."""
    return CandidateInvalidationEvent(
        run_id=RUN_ID, source_id=p.settings.source_id, kind=kind, cause=cause, event_monotonic_ns=t,
        generation=p.current_generation(), detector_config_id=p.settings.detector_config_id,
        job_seq=None, image_seq=None)


def error_kind_counts(counts):
    """kind별 계수에서 세대 전이를 뺀 처리 오류 4종만 남긴다(A §9 「비전 처리 오류 이벤트 수」)."""
    return {k: v for k, v in counts.items() if k != GEN.value}


class TerminalTestBase(PipelineTestBase):
    """작업 4 시험 공통 도우미."""

    def start(self, image_seq, p=None, **frame_over):
        """프레임을 현재 세대·현재 소스로 offer하고 작업을 시작해 돌려준다."""
        p = p or self.p
        fields = {"source_id": p.settings.source_id}
        fields.update(frame_over)
        self.assertTrue(p.offer_frame(frame(image_seq, p.current_generation(), **fields)))
        job = p.start_next()
        self.assertIsNotNone(job)

        return job

    def run_valid(self, image_seq, p=None):
        """작업 하나를 시작해 유효 관측으로 완료한다. (작업, 응답)을 돌려준다."""
        p = p or self.p
        job = self.start(image_seq, p)

        return job, p.complete(job, valid_observation(job))

    def state(self, p=None):
        """현재 관측·계수·사건·폐기 기록·선택·대기의 비교용 묶음."""
        p = p or self.p

        return (p.current_observation, p.counters, p.invalidation_events, p.queue_drop_records,
                p.selected_frame, p.pending_frame, p.current_generation())

    def assert_job_balance(self, p=None):
        """종료(활성 없음) 상태의 작업 보존식과 완료 분해식을 확인한다(A §9)."""
        p = p or self.p
        c = p.counters

        self.assertEqual(c.first_started + c.reprocess_started,
                         c.completed + c.generation_discarded + c.failed + c.aborted)
        self.assertEqual(c.completed, c.valid_completed + sum(c.invalid_completed_by_reason.values()))


class CompletionTest(TerminalTestBase):
    """정상 완료·무검출·세대 폐기(계획 작업 4, A §5.4 「설정 스냅샷과 폐기」)."""

    def test_valid_then_no_object_does_not_reuse_previous(self):
        """유효 완료 뒤 NO_OBJECT 완료는 정상 완료(실패 아님)이며 현재 관측이 무효 관측으로 바뀌어 직전 유효 관측을 재사용하지 않는다."""
        job0, out0 = self.run_valid(0)

        self.assertIs(out0.status, JobStatus.COMPLETED)
        self.assertTrue(out0.applied)
        self.assertFalse(out0.duplicate)
        self.assertIs(self.p.current_observation, out0.observation)
        self.assertTrue(self.p.current_observation.valid)

        job1 = self.start(1)
        out1 = self.p.complete(job1, observation(job1))

        self.assertIs(out1.status, JobStatus.COMPLETED)
        self.assertTrue(out1.applied)
        self.assertIsNone(out1.error)
        self.assertIsNone(out1.invalidation)
        self.assertIs(self.p.current_observation, out1.observation)
        self.assertFalse(self.p.current_observation.valid)
        self.assertEqual(self.p.current_observation.reason, "NO_OBJECT")

        c = self.p.counters
        self.assertEqual((c.completed, c.valid_completed, c.failed), (2, 1, 0))
        self.assertEqual(c.invalid_completed_by_reason, {"NO_OBJECT": 1})
        self.assertEqual(error_kind_counts(c.invalidation_applied_by_kind), {})
        self.assertEqual(self.p.invalidation_events, ())

    def test_invalid_observation_still_requires_reason(self):
        """무효 관측의 reason 필수 규칙(F1)은 그대로이며 시험 fixture도 reason을 가진다."""
        job = self.start(0)

        with self.assertRaises(ValueError):
            dataclasses.replace(observation(job), reason=None)

        self.assertEqual(observation(job).reason, "NO_OBJECT")

    def test_display_result_passes_through_on_completion(self):
        """완료 응답은 실제 `detect_planar()` 표시 결과를 그대로 담고, 표시 결과가 없으면 None이다."""
        job = self.start(0)
        shown = detect_planar(job.frame, job.settings.detector_config)
        self.assertIsInstance(shown, PlanarResult)

        out = self.p.complete(job, observation(job), display_result=shown)
        self.assertIs(out.display_result, shown)

        job1 = self.start(1)
        self.assertIsNone(self.p.complete(job1, observation(job1)).display_result)

    def test_late_completion_is_discarded_with_start_config(self):
        """처리 중 세대가 바뀌면(모든 원인) 관측 발행 없이 세대 폐기로 정리하고, 시작 당시 설정·세대·프레임을 유지하며 generation_discarded를 한 번 올린다."""
        for cause in GenerationCause:
            with self.subTest(cause=cause):
                p = VisionPipeline(settings(), run_id=RUN_ID, clock_ns=FakeClock())
                job = self.start(0, p)
                start_settings, start_frame = job.settings, job.frame

                p.change_settings(CHANGES[cause](p.settings), cause=cause)
                out = p.complete(job, valid_observation(job))

                self.assertIs(out.status, JobStatus.DISCARDED_GENERATION)
                self.assertIsNone(out.observation)
                self.assertIsNone(out.display_result)
                self.assertIsNone(out.error)
                self.assertIsNone(out.invalidation)
                self.assertFalse(out.applied)

                # 기록은 시작 당시 값(새 설정 ID를 붙이지 않음)
                self.assertIs(out.context, job)
                self.assertIs(out.context.settings, start_settings)
                self.assertIs(out.context.frame, start_frame)
                self.assertEqual(out.context.generation, 0)

                self.assertIsNone(p.current_observation)
                c = p.counters
                self.assertEqual((c.generation_discarded, c.completed, c.failed), (1, 0, 0))
                self.assertEqual([e.kind for e in p.invalidation_events], [GEN])
                self.assert_job_balance(p)


class ObservationMismatchTest(TerminalTestBase):
    """관측 식별자·타입 계약 위반(계획 작업 4, 보완 Q7·Q15)."""

    # 필드별 어긋난 값. calibration_id는 None↔문자열과 다른 ID를 모두 본다
    WRONG = {
        "run_id": "run-other", "source_id": "src9", "image_seq": 99, "capture_generation": 5,
        "generation": 5, "job_seq": 99, "detector_config_id": "0" * 64,
    }

    def check_violation(self, out, job, fields):
        """INFERENCE_ERROR 실패 정리와 adapter_contract_violation cause를 확인한다."""
        self.assertIs(out.status, JobStatus.FAILED)
        self.assertIsNone(out.observation)
        self.assertFalse(out.applied)
        self.assertTrue(out.error.startswith(ADAPTER_CONTRACT_VIOLATION))

        for name in fields:
            self.assertIn(name, out.error)

        event = out.invalidation
        self.assertIs(event.kind, INF)
        self.assertEqual(event.cause, out.error)
        self.assertEqual((event.job_seq, event.image_seq, event.generation),
                         (job.job_seq, job.frame.image_seq, job.generation))
        self.assertEqual(event.detector_config_id, job.detector_config_id)

    def test_result_identity_mismatch_is_rejected(self):
        """관측 식별자(run·source·image·세대 2종·job_seq·설정·보정 ID)가 다르면 그 작업만 INFERENCE_ERROR 실패, 다른 작업·현재 관측 무변."""
        cases = dict((k, {k: v}) for k, v in self.WRONG.items())
        cases["calibration_id:none"] = {"calibration_id": None}
        cases["calibration_id:other"] = {"calibration_id": "cal-9"}

        for name, over in cases.items():
            with self.subTest(field=name):
                p = VisionPipeline(settings(), run_id=RUN_ID, clock_ns=FakeClock())
                prev, prev_out = self.run_valid(0, p)
                job = self.start(1, p)

                out = p.complete(job, dataclasses.replace(valid_observation(job), **over))

                self.check_violation(out, job, [name.split(":")[0]])
                c = p.counters
                self.assertEqual((c.completed, c.failed, c.valid_completed), (1, 1, 1))

                # 어긋난 관측은 현재 관측이 아니다. 이전 결과(job_seq 0)는 규칙 2로 무효화됐다
                self.assertIsNone(p.current_observation)

                # 다른 작업(이전 완료)은 그대로이고 재통지하면 최초 COMPLETED다
                self.assertIs(p.complete(prev, valid_observation(prev)).status, JobStatus.COMPLETED)
                self.assert_job_balance(p)

    def test_calibration_none_matches_none(self):
        """프레임과 관측의 calibration_id가 둘 다 None이면 일치이며 보정 값을 만들어 넣지 않는다."""
        job = self.start(0, calibration_id=None)
        out = self.p.complete(job, observation(job))

        self.assertIs(out.status, JobStatus.COMPLETED)
        self.assertIsNone(out.observation.calibration_id)
        self.assertIsNone(self.p.current_observation.calibration_id)

    def test_wrong_observation_or_display_type_fails_job(self):
        """관측이 PlanarObservation이 아니거나 표시 결과 타입이 틀리면 그 작업만 adapter_contract_violation 실패다."""
        for name, obs, shown, field in (("none", None, None, "observation_type"),
                                        ("dict", {"valid": False}, None, "observation_type"),
                                        ("display", "obs", "not-result", "display_result_type")):
            with self.subTest(case=name):
                p = VisionPipeline(settings(), run_id=RUN_ID, clock_ns=FakeClock())
                job = self.start(0, p)
                observed = observation(job) if obs == "obs" else obs

                out = p.complete(job, observed, display_result=shown)

                self.check_violation(out, job, [field])
                self.assertEqual(p.counters.failed, 1)

    def test_unknown_job_is_checked_before_observation(self):
        """알려지지 않은 작업은 관측 검사 전에 계약 오류로 거부하며 활성 작업을 실패시키지 않는다."""
        job = self.start(0)
        fake = dataclasses.replace(job)
        before = self.state()

        with self.assertRaises(PipelineContractError):
            self.p.complete(fake, None)

        self.assertEqual(self.state(), before)
        self.assertIs(self.p.complete(job, observation(job)).status, JobStatus.COMPLETED)

    def test_previous_generation_mismatch_is_failure_not_discard(self):
        """이전 세대 작업의 어긋난 관측은 세대 폐기가 아니라 FAILED 한 번이며 사건은 ignored로 기록한다."""
        job = self.start(0)
        self.p.change_settings(self.p.settings, cause=GenerationCause.SEEK)

        out = self.p.complete(job, dataclasses.replace(observation(job), image_seq=42))

        self.check_violation(out, job, ["image_seq"])
        c = self.p.counters
        self.assertEqual((c.failed, c.generation_discarded), (1, 0))
        self.assertEqual(c.invalidation_ignored_by_kind, {INF.value: 1})
        self.assertEqual(error_kind_counts(c.invalidation_applied_by_kind), {})
        self.assert_job_balance()


class ErrorOrderTest(TerminalTestBase):
    """무효화 적용 규칙 1~5(A §5.4 「비전 처리 오류와 이동 후보 무효화」)."""

    def test_old_error_does_not_invalidate_new_result(self):
        """이전 세대(같은 소스·소스 전환 뒤 옛 소스)의 지연 오류는 기록·ignored만 남기고 새 세대 결과를 지우지 않는다(규칙 1)."""
        for cause in (GenerationCause.RECONNECT, GenerationCause.SOURCE_CHANGED):
            with self.subTest(cause=cause):
                p = VisionPipeline(settings(), run_id=RUN_ID, clock_ns=FakeClock())
                old, _ = self.run_valid(0, p)
                p.change_settings(CHANGES[cause](p.settings), cause=cause)

                new = self.start(1, p)
                p.complete(new, valid_observation(new))
                current = p.current_observation

                late = job_event(old)
                self.assertEqual(late.source_id, "src0")
                self.assertFalse(p.apply_invalidation(late))

                self.assertIs(p.current_observation, current)
                self.assertEqual(p.invalidation_events[-1], late)
                c = p.counters
                self.assertEqual(c.invalidation_ignored_by_kind, {INF.value: 1})
                self.assertEqual(error_kind_counts(c.invalidation_applied_by_kind), {})

                # 다음 정상 작업도 막지 않는다
                self.assertTrue(self.run_valid(2, p)[1].applied)

    def test_job_error_watermark_protects_newer_results(self):
        """같은 세대 오류 job_seq=5는 결과 4·5에 적용하고 결과 6은 보호하며, 적용 수는 차단선 반영으로 1건이다(규칙 2)."""
        for current in (4, 5, 6):
            with self.subTest(current_result=current):
                p = VisionPipeline(settings(), run_id=RUN_ID, clock_ns=FakeClock())
                jobs = [self.run_valid(i, p)[0] for i in range(5)]

                job5 = self.start(5, p)

                if current >= 5:
                    p.complete(job5, valid_observation(job5))

                if current == 6:
                    self.run_valid(6, p)

                self.assertEqual(p.current_observation.job_seq, current)
                applied = p.apply_invalidation(job_event(job5))
                self.assertTrue(applied)

                if current == 6:
                    self.assertEqual(p.current_observation.job_seq, 6)
                else:
                    self.assertIsNone(p.current_observation)

                # 오류 작업이 아직 활성이었다면 늦은 완료는 COMPLETED지만 반영되지 않는다(M5)
                if current == 4:
                    out = p.complete(job5, valid_observation(job5))
                    self.assertIs(out.status, JobStatus.COMPLETED)
                    self.assertFalse(out.applied)
                    self.assertIsNone(p.current_observation)

                self.assertEqual(p.counters.invalidation_applied_by_kind, {INF.value: 1})
                self.assertEqual(len(jobs), 5)
                self.assert_job_balance(p)

    def test_active_job_survives_apply_invalidation(self):
        """apply_invalidation은 활성 작업을 끝내지 않는다: 그동안 새 작업을 시작할 수 없고, 늦은 완료는 COMPLETED·applied=False다(M5)."""
        job = self.start(0)
        self.assertTrue(self.p.offer_frame(frame(1)))
        self.assertTrue(self.p.apply_invalidation(job_event(job)))

        self.assertIsNone(self.p.start_next())

        out = self.p.complete(job, valid_observation(job))
        self.assertEqual((out.status, out.applied), (JobStatus.COMPLETED, False))
        self.assertEqual(self.p.counters.valid_completed, 1)

        nxt = self.p.start_next()
        self.assertTrue(self.p.complete(nxt, valid_observation(nxt)).applied)

    def test_frameless_error_blocks_all_jobs_started_so_far(self):
        """프레임 없는 로딩·설정 오류는 현재 세대 후보 전체를 무효화하고, 오류 당시 이미 시작한 작업의 늦은 완료도 복구시키지 않으며 가짜 ID를 만들지 않는다(규칙 3·4)."""
        for kind in (InvalidationKind.MODEL_LOAD_FAILED, InvalidationKind.MODEL_CONFIG_INVALID):
            with self.subTest(kind=kind):
                p = VisionPipeline(settings(), run_id=RUN_ID, clock_ns=FakeClock())
                self.run_valid(0, p)
                in_flight = self.start(1, p)
                event = frameless_event(p, kind=kind)

                self.assertTrue(p.apply_invalidation(event))
                self.assertIsNone(p.current_observation)

                recorded = p.invalidation_events[-1]
                self.assertEqual((recorded.job_seq, recorded.image_seq), (None, None))

                late = p.complete(in_flight, valid_observation(in_flight))
                self.assertEqual((late.status, late.applied), (JobStatus.COMPLETED, False))
                self.assertIsNotNone(late.observation)
                self.assertIsNone(p.current_observation)

                # 오류 이후 시작한 작업만 복구 가능
                _, after = self.run_valid(2, p)
                self.assertTrue(after.applied)
                self.assertIs(p.current_observation, after.observation)
                self.assertEqual(p.counters.invalidation_applied_by_kind, {kind.value: 1})

    def test_recovery_requires_a_newer_job(self):
        """현재 세대 작업 실패 뒤에는 그 이전 작업이 복구시키지 못하고, 실패 이후 시작한 정상 작업·재처리만 현재 관측이 되며 검출 설정을 바꾸지 않는다(규칙 4·5)."""
        self.run_valid(0)
        failed = self.start(1)
        out = self.p.fail(failed, INF, "runtime error")

        self.assertIs(out.status, JobStatus.FAILED)
        self.assertIsNone(self.p.current_observation)
        self.assertEqual(self.p.counters.invalidation_applied_by_kind, {INF.value: 1})

        # 규칙 5: 설정·검출 방식 자동 전환 없음
        self.assertIs(self.p.settings, self.settings)

        # 실패 이후 시작한 재처리(같은 선택 프레임, 새 job_seq)는 복구 가능
        self.p.change_settings(CHANGES[GenerationCause.DETECTOR_CONFIG_CHANGED](self.p.settings),
                               cause=GenerationCause.DETECTOR_CONFIG_CHANGED)
        re_job = self.p.start_reprocess()
        self.assertGreater(re_job.job_seq, failed.job_seq)
        self.assertTrue(self.p.complete(re_job, valid_observation(re_job)).applied)
        self.assert_job_balance()

    def test_generation_change_counts_but_is_not_a_watermark(self):
        """세대 전이는 GENERATION_CHANGED applied 1건(no-op 0건)으로 오류 4종과 따로 세고, 오류 차단선을 세우지 않아 새 세대 정상 작업이 반영된다."""
        self.run_valid(0)
        self.assertIsNone(self.p.change_settings(self.p.settings, cause=GenerationCause.MODE_CHANGED))
        self.assertEqual(self.p.counters.invalidation_applied_by_kind, {})

        event = self.p.change_settings(self.p.settings, cause=GenerationCause.SEEK)
        self.assertEqual(self.p.counters.invalidation_applied_by_kind, {GEN.value: 1})
        self.assertEqual(self.p.invalidation_events, (event,))
        self.assertEqual(self.p._error_watermarks, {})  # 차단선 내부 상태 확인

        _, out = self.run_valid(1)
        self.assertTrue(out.applied)

        # 자체 사건의 재수신(같은 객체·같은 payload)은 무변, 새 GENERATION_CHANGED는 거부
        before = self.state()
        self.assertFalse(self.p.apply_invalidation(event))
        self.assertFalse(self.p.apply_invalidation(dataclasses.replace(event)))
        self.assertEqual(self.state(), before)

        with self.assertRaises(PipelineContractError):
            self.p.apply_invalidation(dataclasses.replace(event, event_monotonic_ns=event.event_monotonic_ns + 1))

        self.assertEqual(self.state(), before)


class EventValidationTest(TerminalTestBase):
    """외부 사건의 출처 검증과 동일 통지 정의(보완 Q2·Q9·M3·M4)."""

    def test_invalid_origin_is_contract_error(self):
        """다른 run·미래 세대·세대와 다른 출처·발급하지 않은 job_seq·작업과 다른 image_seq/세대·사건 아닌 값은 계약 오류이며 상태가 변하지 않는다."""
        job, _ = self.run_valid(0)
        good = job_event(job)
        cases = {
            "run": dataclasses.replace(good, run_id="run-other"),
            "future generation": dataclasses.replace(good, generation=3),
            "source of generation": dataclasses.replace(good, source_id="src9"),
            "config of generation": dataclasses.replace(good, detector_config_id="f" * 64),
            "never issued job": dataclasses.replace(good, job_seq=7),
            "image of job": dataclasses.replace(good, image_seq=8),
            "frameless future": dataclasses.replace(frameless_event(self.p), generation=1),
            "not an event": {"kind": "INFERENCE_ERROR"},
        }
        before = self.state()

        for name, event in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(PipelineContractError):
                    self.p.apply_invalidation(event)

                self.assertEqual(self.state(), before)

    def test_same_payload_is_same_notification(self):
        """9개 필드가 모두 같은 새 객체는 같은 통지로 무변이고, job_seq=None이라도 시각·원인이 다른 별개 로딩 오류는 각각 기록한다."""
        first = frameless_event(self.p, t=10)
        self.assertTrue(self.p.apply_invalidation(first))
        before = self.state()

        self.assertFalse(self.p.apply_invalidation(dataclasses.replace(first)))
        self.assertEqual(self.state(), before)

        self.assertTrue(self.p.apply_invalidation(frameless_event(self.p, t=11)))
        self.assertTrue(self.p.apply_invalidation(frameless_event(self.p, t=11, cause="device lost")))

        self.assertEqual(len(self.p.invalidation_events), 3)
        self.assertEqual(self.p.counters.invalidation_applied_by_kind,
                         {InvalidationKind.MODEL_LOAD_FAILED.value: 3})

    def test_fail_records_own_event_and_redelivery_is_noop(self):
        """fail()이 만든 사건을 다시 apply해도 무변이며, fail은 프레임 없는 종류·모르는 종류·빈 원인을 거부한다."""
        job = self.start(0)

        for kind, cause in ((InvalidationKind.MODEL_LOAD_FAILED, "x"), (GEN, "x"), ("BOGUS", "x"), (INF, "")):
            with self.subTest(kind=kind, cause=cause):
                with self.assertRaises(PipelineContractError):
                    self.p.fail(job, kind, cause)

        out = self.p.fail(job, InvalidationKind.MODEL_CONFIG_INVALID, "class map mismatch")
        self.assertIs(out.status, JobStatus.FAILED)
        self.assertEqual(out.invalidation.job_seq, job.job_seq)

        before = self.state()
        self.assertFalse(self.p.apply_invalidation(out.invalidation))
        self.assertEqual(self.state(), before)


class EventPathDedupTest(TerminalTestBase):
    """apply 경로와 fail/close 경로 사이의 동일 사건 1회 반영(r1 F1). 시계는 같은 값을 돌려줄 수 있다."""

    def fixed_pipeline(self):
        """항상 1을 돌려주는 단조 시계(같은 값 반환은 허용된 계약)의 파이프라인."""
        return VisionPipeline(settings(), run_id=RUN_ID, clock_ns=lambda: 1)

    def test_apply_then_terminal_records_same_event_once(self):
        """외부 사건을 먼저 apply한 뒤 같은 9필드로 fail/close하면 사건·applied는 1건이고, 작업은 FAILED/ABORTED로 한 번 정상 종료한다."""
        for kind, cause, status in ((INF, "boom", JobStatus.FAILED), (ABORT, "closed", JobStatus.ABORTED)):
            with self.subTest(kind=kind):
                p = self.fixed_pipeline()
                job = self.start(0, p)
                event = job_event(job, kind=kind, cause=cause, t=1)
                self.assertTrue(p.apply_invalidation(event))

                if kind is ABORT:
                    (out,) = p.close()
                else:
                    out = p.fail(job, kind, cause)

                self.assertIs(out.status, status)
                self.assertEqual(out.invalidation, event)
                self.assertEqual(p.invalidation_events, (event,))

                c = p.counters
                self.assertEqual(c.invalidation_applied_by_kind, {kind.value: 1})
                self.assertEqual(c.failed + c.aborted, 1)
                self.assertIsNone(p.start_next())  # 활성 해제(대기 없음)
                self.assertTrue(p.complete(job, observation(job)).duplicate)
                self.assert_job_balance(p)

    def test_terminal_then_apply_is_noop(self):
        """fail(실패·중단)이 만든 사건을 같은 payload의 새 객체로 apply하면 무변이며, 이는 terminal 재통지와 별개로 확인한다."""
        for kind in (INF, ABORT):
            with self.subTest(kind=kind):
                p = self.fixed_pipeline()
                job = self.start(0, p)
                out = p.fail(job, kind, "late")
                before = self.state(p)

                self.assertFalse(p.apply_invalidation(dataclasses.replace(out.invalidation)))
                self.assertEqual(self.state(p), before)

                # terminal 재통지도 따로 무변
                self.assertTrue(p.fail(job, kind, "late").duplicate)
                self.assertEqual(self.state(p), before)

    def test_old_generation_event_then_terminal_is_ignored_once(self):
        """이전 세대 사건을 먼저 기록(ignored)한 뒤 같은 payload로 fail/close해도 ignored는 1건이고 현재 세대 상태를 바꾸지 않는다."""
        for kind, cause in ((INF, "boom"), (ABORT, "closed")):
            with self.subTest(kind=kind):
                p = self.fixed_pipeline()
                job = self.start(0, p)
                p.change_settings(p.settings, cause=GenerationCause.RECONNECT)
                event = job_event(job, kind=kind, cause=cause, t=1)

                self.assertFalse(p.apply_invalidation(event))
                (out,) = p.close() if kind is ABORT else (p.fail(job, kind, cause),)

                self.assertEqual(out.invalidation, event)
                c = p.counters
                self.assertEqual(c.invalidation_ignored_by_kind, {kind.value: 1})
                self.assertEqual(c.invalidation_applied_by_kind, {GEN.value: 1})
                self.assertEqual(p._error_watermarks, {})  # 현재 세대 차단선 없음
                self.assertEqual(len(p.invalidation_events), 2)

        # 현재 세대 후보 보호: 옛 사건 기록 뒤 실패해도 새 세대 결과는 유지된다
        p = self.fixed_pipeline()
        old = self.start(0, p)
        p.change_settings(p.settings, cause=GenerationCause.RECONNECT)
        stale = job_event(old, t=1)
        p.apply_invalidation(stale)
        p.fail(old, INF, stale.cause)
        _, out = self.run_valid(1, p)
        self.assertTrue(out.applied)
        self.assertIs(p.current_observation, out.observation)

    def test_different_payload_is_counted_separately(self):
        """원인이 다른 사건은 같은 작업이라도 별개 통지로 각각 기록·집계한다."""
        p = self.fixed_pipeline()
        job = self.start(0, p)

        self.assertTrue(p.apply_invalidation(job_event(job, cause="adapter says a", t=1)))
        out = p.fail(job, INF, "runtime says b")

        self.assertEqual(len(p.invalidation_events), 2)
        self.assertEqual(out.invalidation.cause, "runtime says b")
        self.assertEqual(p.counters.invalidation_applied_by_kind, {INF.value: 2})


class IdempotencyTest(TerminalTestBase):
    """중복 terminal 통지와 작업 판별(계획 작업 4, 보완 M1·M2·Q12)."""

    def check_duplicate(self, job, first, again):
        """재통지 응답이 duplicate=True·applied=False·데이터 없음·최초 status/error/사건인지 확인한다."""
        self.assertTrue(again.duplicate)
        self.assertFalse(again.applied)
        self.assertIsNone(again.observation)
        self.assertIsNone(again.display_result)
        self.assertIs(again.context, job)
        self.assertIs(again.status, first.status)
        self.assertEqual(again.error, first.error)
        self.assertEqual(again.invalidation, first.invalidation)

    def test_duplicate_terminal_callback_is_idempotent(self):
        """유효·무효·차단 완료, 실패·중단 뒤의 이중 complete/fail은 최초 status를 유지하고 현재 결과·모든 계수·사건 목록을 바꾸지 않는다."""

        def valid(p):
            """유효 완료."""
            job = self.start(0, p)
            return job, p.complete(job, valid_observation(job))

        def invalid(p):
            """무효(NO_OBJECT) 완료."""
            job = self.start(0, p)
            return job, p.complete(job, observation(job))

        def blocked(p):
            """프레임 없는 오류 뒤 늦은 완료(applied=False)."""
            job = self.start(0, p)
            p.apply_invalidation(frameless_event(p))
            return job, p.complete(job, valid_observation(job))

        def failed(p):
            """처리 실패."""
            job = self.start(0, p)
            return job, p.fail(job, INF, "boom")

        def aborted(p):
            """작업 중단."""
            job = self.start(0, p)
            return job, p.fail(job, ABORT, "timeout")

        expected = {"valid": (JobStatus.COMPLETED, True), "invalid": (JobStatus.COMPLETED, True),
                    "blocked": (JobStatus.COMPLETED, False), "failed": (JobStatus.FAILED, False),
                    "aborted": (JobStatus.ABORTED, False)}

        for name, make in (("valid", valid), ("invalid", invalid), ("blocked", blocked),
                           ("failed", failed), ("aborted", aborted)):
            with self.subTest(first=name):
                p = VisionPipeline(settings(), run_id=RUN_ID, clock_ns=FakeClock())
                job, first = make(p)
                self.assertEqual((first.status, first.applied), expected[name])
                self.assertFalse(first.duplicate)
                before = self.state(p)

                # 서로 다른 인자로 재통지해도 최초 status 유지
                for again in (p.complete(job, valid_observation(job)),
                              p.complete(job, None, display_result="junk"),
                              p.fail(job, INF, "late error"),
                              p.fail(job, "BOGUS", "")):
                    self.check_duplicate(job, first, again)

                self.assertEqual(self.state(p), before)
                self.assertEqual(p.close(), ())
                self.check_duplicate(job, first, p.complete(job, observation(job)))
                self.assert_job_balance(p)

    def test_copied_or_foreign_job_is_rejected(self):
        """복제·직접 생성·다른 파이프라인·발급하지 않은 작업·JobContext 아닌 값은 계약 오류이고 배열 비교 오류를 내지 않는다."""
        job = self.start(0)
        other = VisionPipeline(settings(), run_id=RUN_ID, clock_ns=FakeClock())
        foreign = self.start(0, other)
        cases = {
            "replace copy": dataclasses.replace(job),
            "manual same seq": JobContext(frame=job.frame, generation=0, job_seq=0, settings=job.settings,
                                          started_monotonic_ns=job.started_monotonic_ns, reprocess=False),
            "other pipeline": foreign,
            "never issued": dataclasses.replace(job, job_seq=5),
            "bool seq": dataclasses.replace(job, job_seq=False),
            "not a job": "job-0",
        }
        before = self.state()

        for name, fake in cases.items():
            with self.subTest(case=name):
                for call in (lambda: self.p.complete(fake, observation(job)),
                             lambda: self.p.fail(fake, INF, "x")):
                    with self.assertRaises(PipelineContractError):
                        call()

                self.assertEqual(self.state(), before)

        self.assertIs(self.p.complete(job, observation(job)).status, JobStatus.COMPLETED)

    def test_same_frame_jobs_are_not_mistaken_for_duplicates(self):
        """같은 프레임 객체를 쓰는 원작업과 재처리 작업은 서로 다른 작업이며 한쪽의 완료가 다른 쪽의 중복 통지가 되지 않는다."""
        first = self.start(0)
        self.p.complete(first, observation(first))
        self.p.change_settings(CHANGES[GenerationCause.DETECTOR_CONFIG_CHANGED](self.p.settings),
                               cause=GenerationCause.DETECTOR_CONFIG_CHANGED)
        re_job = self.p.start_reprocess()
        self.assertIs(re_job.frame, first.frame)

        out = self.p.complete(re_job, valid_observation(re_job))
        self.assertFalse(out.duplicate)
        self.assertTrue(out.applied)
        self.assertTrue(self.p.complete(first, observation(first)).duplicate)

    def test_job_outcome_keeps_six_argument_constructor(self):
        """JobOutcome은 기존 6인자 생성자를 그대로 받고 applied·duplicate 기본값은 False다."""
        job = self.start(0)
        out = JobOutcome(job, JobStatus.ABORTED, None, None, None, None)

        self.assertEqual((out.applied, out.duplicate), (False, False))
        self.assertEqual([f.name for f in dataclasses.fields(JobOutcome)][-2:], ["applied", "duplicate"])


class CloseTest(TerminalTestBase):
    """종료 처리와 종료 후 계약(계획 작업 4 close 문장, 전달 6)."""

    def test_close_aborts_current_generation_job_once(self):
        """현재 세대 활성 작업은 실제 식별자의 JOB_ABORTED로 한 번 정리해 applied 1건이고, 관측·선택을 비운 뒤 대기를 closed로 한 번 폐기한다."""
        self.run_valid(0)
        job = self.start(1)
        self.assertTrue(self.p.offer_frame(frame(2)))

        outs = self.p.close()

        self.assertEqual(len(outs), 1)
        out = outs[0]
        self.assertIs(out.context, job)
        self.assertIs(out.status, JobStatus.ABORTED)
        self.assertIsNone(out.observation)
        event = out.invalidation
        self.assertIs(event.kind, ABORT)
        self.assertEqual((event.job_seq, event.image_seq, event.generation), (1, 1, 0))
        self.assertEqual(self.p.invalidation_events, (event,))

        c = self.p.counters
        self.assertEqual(c.invalidation_applied_by_kind, {ABORT.value: 1})
        self.assertEqual(c.aborted, 1)
        self.assertEqual(c.queue_dropped, {QueueDropReason.CLOSED.value: 1})
        self.assertEqual(self.p.queue_drop_records[-1].image_seq, 2)
        self.assertIsNone(self.p.current_observation)
        self.assertIsNone(self.p.selected_frame)
        self.assertIsNone(self.p.pending_frame)
        self.assertEqual(self.p.current_generation(), 0)
        self.assertTrue(self.p.closed)

        # 재호출은 빈 tuple, 늦은 완료는 duplicate(재기록 없음)
        before = self.state()
        self.assertEqual(self.p.close(), ())
        late = self.p.complete(job, valid_observation(job))
        self.assertTrue(late.duplicate)
        self.assertIs(late.status, JobStatus.ABORTED)
        self.assertEqual(self.state(), before)
        self.assert_received_balance()
        self.assert_job_balance()

    def test_close_counts_applied_even_without_candidate(self):
        """현재 세대 작업의 중단 사건은 현재 후보가 이미 None이어도 applied 1건이다."""
        job = self.start(0)
        self.assertIsNone(self.p.current_observation)

        (out,) = self.p.close()

        self.assertIs(out.context, job)
        self.assertEqual(self.p.counters.invalidation_applied_by_kind, {ABORT.value: 1})
        self.assertEqual(self.p.counters.invalidation_ignored_by_kind, {})

    def test_close_previous_generation_job_is_ignored(self):
        """이전 세대 활성 작업의 중단 사건은 ignored 1건이며 세대나 작업 세대를 바꾸지 않는다."""
        job = self.start(0)
        self.p.change_settings(self.p.settings, cause=GenerationCause.RECONNECT)

        (out,) = self.p.close()

        self.assertEqual(out.invalidation.generation, 0)
        self.assertEqual(self.p.current_generation(), 1)
        c = self.p.counters
        self.assertEqual(c.invalidation_ignored_by_kind, {ABORT.value: 1})
        self.assertEqual(c.invalidation_applied_by_kind, {GEN.value: 1})
        self.assertIs(out.context, job)

    def test_close_without_active_job_creates_no_event(self):
        """활성 작업이 없으면 사건·무효화 계수를 만들지 않고 관측·선택·대기만 정리한다."""
        self.run_valid(0)
        self.assertTrue(self.p.offer_frame(frame(1)))

        self.assertEqual(self.p.close(), ())

        c = self.p.counters
        self.assertEqual(self.p.invalidation_events, ())
        self.assertEqual((c.invalidation_applied_by_kind, c.invalidation_ignored_by_kind), ({}, {}))
        self.assertEqual(c.queue_dropped, {QueueDropReason.CLOSED.value: 1})
        self.assertIsNone(self.p.current_observation)
        self.assertIsNone(self.p.selected_frame)

    def test_no_restart_after_close(self):
        """종료 후 수신·설정 전이·무효화는 계약 오류(수신 수 무변), 시작·재처리는 None, 위조 작업은 거부한다."""
        job, _ = self.run_valid(0)
        self.p.close()
        before = self.state()

        for call in (lambda: self.p.offer_frame(frame(1)),
                     lambda: self.p.change_settings(self.p.settings, cause=GenerationCause.SEEK),
                     lambda: self.p.apply_invalidation(frameless_event(self.p)),
                     lambda: self.p.complete(dataclasses.replace(job), observation(job))):
            with self.assertRaises(PipelineContractError):
                call()

        self.assertIsNone(self.p.start_next())
        self.assertIsNone(self.p.start_reprocess())
        self.assertEqual(self.state(), before)
        self.assertTrue(self.p.complete(job, observation(job)).duplicate)


class CounterBalanceTest(TerminalTestBase):
    """종료 후 보존식(A §9, 전달 7)."""

    def test_counters_balance_after_mixed_run(self):
        """대기 교체·오래된 캡처·세대 폐기·무효 완료·실패·재처리·중단·종료를 섞어도 보존식이 성립하고 중복 집계가 없다."""
        p = self.p
        self.run_valid(0)                                   # 유효 완료

        j1 = self.start(1)
        self.assertTrue(p.offer_frame(frame(2)))
        self.assertTrue(p.offer_frame(frame(3)))            # 2 교체 폐기
        p.complete(j1, observation(j1))                     # 무효 완료

        j3 = p.start_next()
        p.change_settings(CHANGES[GenerationCause.DETECTOR_CONFIG_CHANGED](p.settings),
                          cause=GenerationCause.DETECTOR_CONFIG_CHANGED)
        p.complete(j3, valid_observation(j3))               # 세대 폐기
        p.complete(j3, valid_observation(j3))               # 중복(무집계)

        re_job = p.start_reprocess()
        p.fail(re_job, INF, "boom")                         # 실패
        p.fail(re_job, INF, "boom")                         # 중복(무집계)

        self.assertFalse(p.offer_frame(frame(4, capture_generation=0)))   # 오래된 캡처 폐기
        j5 = self.start(5)
        self.assertTrue(p.offer_frame(frame(6, 1)))
        p.fail(j5, ABORT, "watchdog")                       # 중단

        j6 = p.start_next()
        self.assertTrue(p.offer_frame(frame(7, 1)))
        p.close()                                           # 활성 중단 + 대기 closed 폐기
        p.complete(j6, valid_observation(j6))               # 종료 후 늦은 완료(무집계)

        c = p.counters
        self.assertEqual(c.received, c.first_started + sum(c.queue_dropped.values()))
        self.assert_job_balance()
        self.assertEqual((c.received, c.first_started, c.reprocess_started), (8, 5, 1))
        self.assertEqual((c.completed, c.valid_completed, c.generation_discarded, c.failed, c.aborted),
                         (2, 1, 1, 1, 2))
        self.assertEqual(c.queue_dropped, {QueueDropReason.REPLACED_BY_NEWER.value: 1,
                                           QueueDropReason.STALE_CAPTURE.value: 1,
                                           QueueDropReason.CLOSED.value: 1})
        self.assertEqual(c.invalidation_applied_by_kind, {GEN.value: 1, INF.value: 1, ABORT.value: 2})


class RetentionTest(TerminalTestBase):
    """과거 영상·윤곽을 코어가 붙잡지 않는 보관 정책(보완 Q13, 103 §1·§4)."""

    def collect(self):
        """순환 참조까지 회수한다."""
        gc.collect()

    def test_old_images_and_contours_are_released(self):
        """N건 완료 뒤 시험 쪽 참조를 지우면 선택 프레임을 뺀 과거 영상과 현재 관측을 뺀 과거 윤곽이 회수되고, 선택·현재 관측은 보존되며 소스 전이·close로 비우면 회수된다."""
        image_refs, contour_refs = [], []

        for i in range(6):
            job, out = self.run_valid(i)
            image_refs.append(weakref.ref(job.frame.image))
            contour_refs.append(weakref.ref(out.observation.calc_contour))
            del job, out

        self.collect()

        self.assertEqual([r() is None for r in image_refs], [True] * 5 + [False])
        self.assertEqual([r() is None for r in contour_refs], [True] * 5 + [False])
        self.assertIs(self.p.selected_frame.image, image_refs[-1]())
        self.assertIs(self.p.current_observation.calc_contour, contour_refs[-1]())

        # 소스 전이는 선택·현재 관측을 비우므로 마지막 영상·윤곽도 회수된다
        self.p.change_settings(CHANGES[GenerationCause.SOURCE_CHANGED](self.p.settings),
                               cause=GenerationCause.SOURCE_CHANGED)
        self.collect()
        self.assertIsNone(image_refs[-1]())
        self.assertIsNone(contour_refs[-1]())

    def test_invalidation_and_close_release_buffers(self):
        """무효화로 지운 현재 관측의 윤곽과 close로 비운 선택·대기 영상이 회수된다."""
        out = self.run_valid(0)[1]  # 작업 객체를 시험 쪽 변수에 남기지 않는다
        contour = weakref.ref(out.observation.calc_contour)
        del out

        self.p.apply_invalidation(frameless_event(self.p))
        self.collect()
        self.assertIsNone(contour())

        self.assertTrue(self.p.offer_frame(frame(1)))
        pending = weakref.ref(self.p.pending_frame.image)
        selected = weakref.ref(self.p.selected_frame.image)

        self.p.close()
        self.collect()
        self.assertIsNone(pending())
        self.assertIsNone(selected())

    def test_held_old_job_can_still_renotify(self):
        """호출자가 계속 보관한 이전 작업의 늦은 재통지는 수락되어 duplicate 응답이 되고, 그 작업의 복제는 거부된다."""
        old, _ = self.run_valid(0)

        for i in range(1, 4):
            self.run_valid(i)

        self.collect()
        again = self.p.complete(old, valid_observation(old))
        self.assertTrue(again.duplicate)
        self.assertIs(again.status, JobStatus.COMPLETED)

        with self.assertRaises(PipelineContractError):
            self.p.complete(dataclasses.replace(old), valid_observation(old))

    def test_terminal_summaries_hold_no_heavy_objects(self):
        """terminal·작업 요약과 등록표가 JobContext·FramePacket·PlanarObservation·PlanarResult·배열·함수를 강하게 참조하지 않는다."""
        job, _ = self.run_valid(0)
        failed = self.start(1)
        self.p.fail(failed, INF, "boom")
        heavy = (JobContext, FramePacket, PlanarObservation, PlanarResult, np.ndarray)

        for summary in list(self.p._terminal.values()) + list(self.p._job_meta.values()):
            for ref in gc.get_referents(summary) + [getattr(summary, f.name) for f in dataclasses.fields(summary)]:
                self.assertNotIsInstance(ref, heavy)
                self.assertFalse(callable(ref) and not isinstance(ref, type), repr(ref))

        self.assertIsInstance(self.p._job_refs, weakref.WeakValueDictionary)


if __name__ == "__main__":
    unittest.main()
