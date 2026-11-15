"""A 비전 파이프라인 진행 보장(순환 정지 조건) 단위시험 — 활성 작업이 terminal 처리 뒤 반드시 풀리는지.

근거: 계획 r3 작업 4(관측 식별자·타입 불일치는 해당 작업만 adapter_contract_violation 실패 정리,
close 뒤 새 시작 금지, 최초 terminal 유지, 출처 검증), A 상세설계 §5.4·§9, reviewer 순환 정지 조건 점검 지시
(`_workspace/codex-reviews/2026-11-15-pipeline-liveness-probe.py` 입력).

분류:
- 코어 결함(수정함): 비스칼라·변조 관측이 비교 예외로 새어 활성 작업이 남던 경로, 반쯤 닫힌 close.
- 잘못된 호출의 정상 거부: 잘못된 fail 인자·위조 작업·변조 사건은 계약 오류이며 상태를 바꾸지 않는다.
- 후속 실행기 책임(작업 6): terminal 통지 자체가 없으면 코어는 스스로 풀지 않는다. 실행기가 worker 예외·
  시간 초과·종료 때 fail/close를 반드시 부른다. 자동 시간제한·강제 취소는 이 코어에 넣지 않는다.

관측은 시험이 만든 가짜 detector 결과다. 실제 worker·스레드·B/C/D 통신을 쓰지 않는다.
"""
import dataclasses
import gc
import random
import time
import unittest
import weakref

import numpy as np

from eyetoarm.common.types import InvalidationKind
from eyetoarm.vision.pipeline import GenerationCause, PipelineContractError, VisionPipeline
from eyetoarm.vision.pipeline_types import JobStatus
from test_pipeline import RUN_ID, FakeClock, config, frame, observation, settings
from test_pipeline_terminal import TerminalTestBase, frameless_event, job_event, valid_observation

INF = InvalidationKind.INFERENCE_ERROR
ABORT = InvalidationKind.JOB_ABORTED

# 관측 식별자 8필드(계획 r3 작업 4 + 보완 M7 calibration_id)
ID_FIELDS = ("run_id", "source_id", "image_seq", "capture_generation", "generation", "job_seq",
             "detector_config_id", "calibration_id")

# 무작위 순환 시험의 고정 조건(재현용)
SEEDS = range(20)
JOBS_PER_SEED = 250
TIME_LIMIT_S = 60.0


def tampered(obs, **fields):
    """frozen 관측을 생성 뒤 변조한다(생성자 검증을 우회한 잘못된 어댑터 출력 모사)."""
    for name, value in fields.items():
        object.__setattr__(obs, name, value)

    return obs


class LivenessTestBase(TerminalTestBase):
    """진행 확인 도우미."""

    def assert_next_job_runs(self, p, image_seq):
        """활성 슬롯이 비어 다음 정상 작업이 시작·반영되는지 확인한다."""
        self.assertIsNone(p._active)
        nxt = self.start(image_seq, p)
        out = p.complete(nxt, valid_observation(nxt))
        self.assertTrue(out.applied)
        self.assertIs(p.current_observation, out.observation)

    def assert_running_balance(self, p):
        """실행 중 보존식(활성 작업 0/1 포함)과 수신 보존식(대기 0/1 포함)을 확인한다(A §9)."""
        c = p.counters
        active = 0 if p._active is None else 1
        pending = 0 if p.pending_frame is None else 1

        self.assertEqual(c.first_started + c.reprocess_started,
                         c.completed + c.generation_discarded + c.failed + c.aborted + active)
        self.assertEqual(c.received, c.first_started + sum(c.queue_dropped.values()) + pending)


class BadObservationLivenessTest(LivenessTestBase):
    """잘못된 어댑터 관측 뒤에도 해당 작업만 실패 정리되고 다음 작업이 진행된다(코어 결함 수정)."""

    def test_non_scalar_identity_fails_job_and_allows_next(self):
        """8개 식별자에 배열이 들어오면 비교 예외 없이 그 작업만 adapter_contract_violation 실패이고 다음 작업이 시작된다."""
        for name in ID_FIELDS:
            with self.subTest(field=name):
                p = VisionPipeline(settings(), run_id=RUN_ID, clock_ns=FakeClock())
                job = self.start(0, p)
                bad = dataclasses.replace(valid_observation(job), **{name: np.array([999, 998])})

                out = p.complete(job, bad)

                self.assertIs(out.status, JobStatus.FAILED)
                self.assertIn(name, out.error)
                self.assertTrue(out.error.startswith("adapter_contract_violation"))
                self.assertEqual(p.counters.failed, 1)
                self.assert_next_job_runs(p, 1)

    def test_loosely_equal_identity_is_not_a_match(self):
        """`False == 0`·numpy 정수·같은 값의 다른 타입은 일치로 보지 않고 위반으로 정리한다(배열을 np.all로 받아들이지 않음)."""
        cases = {
            "job_seq bool": {"job_seq": False},
            "generation bool": {"generation": False},
            "image_seq numpy": {"image_seq": np.int64(0)},
            "capture_generation float": {"capture_generation": 0.0},
            "image_seq one-element array": {"image_seq": np.array([0])},
        }

        for name, over in cases.items():
            with self.subTest(case=name):
                p = VisionPipeline(settings(), run_id=RUN_ID, clock_ns=FakeClock())
                job = self.start(0, p)
                self.assertEqual((job.job_seq, job.generation, job.frame.image_seq), (0, 0, 0))

                out = p.complete(job, dataclasses.replace(valid_observation(job), **over))

                self.assertIs(out.status, JobStatus.FAILED)
                self.assertIn(next(iter(over)), out.error)
                self.assert_next_job_runs(p, 1)

    def test_tampered_verdict_fields_fail_job(self):
        """생성 뒤 valid·reason을 변조한 관측은 계수 키 오류로 새지 않고 해당 작업만 실패로 정리된다."""
        cases = {
            "valid array": (valid_observation, {"valid": np.array([1, 0])}, "valid"),
            "valid int": (valid_observation, {"valid": 1}, "valid"),
            "invalid reason list": (observation, {"reason": ["x"]}, "reason"),
            "invalid reason None": (observation, {"reason": None}, "reason"),
            "valid with reason": (valid_observation, {"reason": "NO_OBJECT"}, "reason"),
        }

        for name, (make, over, field) in cases.items():
            with self.subTest(case=name):
                p = VisionPipeline(settings(), run_id=RUN_ID, clock_ns=FakeClock())
                job = self.start(0, p)

                out = p.complete(job, tampered(make(job), **over))

                self.assertIs(out.status, JobStatus.FAILED)
                self.assertIn(field, out.error)
                c = p.counters
                self.assertEqual((c.completed, c.failed, c.invalid_completed_by_reason), (0, 1, {}))
                self.assert_next_job_runs(p, 1)

    def test_forged_job_with_bad_observation_does_not_release_active(self):
        """위조 작업의 잘못된 관측은 계약 오류이며 진짜 활성 작업을 해제하거나 실패시키지 않는다."""
        job = self.start(0)
        fake = dataclasses.replace(job)
        bad = dataclasses.replace(valid_observation(job), image_seq=np.array([1, 2]))
        before = self.state()

        with self.assertRaises(PipelineContractError):
            self.p.complete(fake, bad)

        self.assertEqual(self.state(), before)
        self.assertIs(self.p._active, job)
        self.assertTrue(self.p.offer_frame(frame(1)))
        self.assertIsNone(self.p.start_next())
        self.assertIs(self.p.complete(job, valid_observation(job)).status, JobStatus.COMPLETED)
        self.assertIsNotNone(self.p.start_next())


class CallContractLivenessTest(LivenessTestBase):
    """잘못된 호출의 정상 거부와 후속 실행기 책임의 경계."""

    def test_rejected_fail_call_keeps_job_until_valid_terminal(self):
        """잘못된 fail 인자는 계약 오류로 거부되어 상태가 그대로이고(활성 유지), 올바른 fail 뒤에는 다음 작업이 진행된다."""
        job = self.start(0)
        before = self.state()

        for kind, cause in (([1], "x"), (INF, np.array(["a", "b"])), (INF, ""),
                            (InvalidationKind.MODEL_LOAD_FAILED, "x")):
            with self.subTest(kind=kind, cause=cause):
                with self.assertRaises(PipelineContractError):
                    self.p.fail(job, kind, cause)

                self.assertEqual(self.state(), before)
                self.assertIs(self.p._active, job)

        self.p.fail(job, INF, "worker exception")
        self.assert_next_job_runs(self.p, 1)

    def test_tampered_event_is_contract_error_without_state_change(self):
        """생성 뒤 배열 등으로 변조한 사건은 비교 예외 대신 계약 오류이며 상태와 활성 작업을 바꾸지 않는다."""
        job = self.start(0)
        before = self.state()

        for name, value in (("run_id", np.array(["a", "b"])), ("generation", np.array([0, 1])),
                            ("job_seq", np.array([0, 1])), ("image_seq", True), ("kind", "INFERENCE_ERROR")):
            with self.subTest(field=name):
                event = job_event(job)
                object.__setattr__(event, name, value)

                with self.assertRaises(PipelineContractError):
                    self.p.apply_invalidation(event)

                self.assertEqual(self.state(), before)
                self.assertIs(self.p._active, job)

    def test_lost_terminal_notification_is_not_recovered_by_core(self):
        """terminal 통지가 없으면 세대 전이·무효화·새 프레임 100건으로도 풀리지 않고(현 계약), 명시적 JOB_ABORTED 뒤 최신 대기 프레임으로 진행한다."""
        original = self.start(0)
        self.p.change_settings(self.p.settings, cause=GenerationCause.RECONNECT)
        self.p.apply_invalidation(frameless_event(self.p))

        for seq in range(1, 101):
            self.assertTrue(self.p.offer_frame(frame(seq, self.p.current_generation())))
            self.assertIsNone(self.p.start_next())

        self.assertEqual(self.p.counters.first_started, 1)
        self.assert_running_balance(self.p)

        # 실행기(작업 6)가 시간 초과·worker 예외 때 해야 하는 명시적 정리
        out = self.p.fail(original, ABORT, "terminal notification timeout")
        self.assertIs(out.status, JobStatus.ABORTED)
        resumed = self.p.start_next()
        self.assertEqual(resumed.frame.image_seq, 100)
        self.assertTrue(self.p.complete(resumed, valid_observation(resumed)).applied)

    def test_close_failure_does_not_leave_half_closed_pipeline(self):
        """close 중 시계 계약 위반으로 중단 정리가 실패하면 닫힘 표시 없이 상태가 그대로이고, 시계를 고친 뒤 close가 한 번 중단 정리한다."""
        job = self.start(0)
        good_clock = self.p._clock_ns
        self.p._clock_ns = lambda: "not-an-int"  # 시험용: 주입 시계가 계약(정수)을 어김
        before = self.state()

        with self.assertRaises(ValueError):
            self.p.close()

        self.assertFalse(self.p.closed)
        self.assertIs(self.p._active, job)
        self.assertEqual(self.state(), before)

        self.p._clock_ns = good_clock
        (out,) = self.p.close()
        self.assertIs(out.status, JobStatus.ABORTED)
        self.assertTrue(self.p.closed)
        self.assertTrue(self.p.complete(job, valid_observation(job)).duplicate)


class CloseClockLivenessTest(LivenessTestBase):
    """close 도중 시계 예외의 예외 안전성과 중단 결과 인계(reviewer r1 L1)."""

    def make(self, clock):
        """활성 작업 1개와 대기 프레임 1개를 둔 파이프라인을 만든다."""
        p = VisionPipeline(settings(), run_id=RUN_ID, clock_ns=clock)
        job = self.start(0, p)
        self.assertTrue(p.offer_frame(frame(1)))

        return p, job

    def test_close_reads_clock_once_so_late_clock_failure_cannot_split_cleanup(self):
        """close는 시계를 한 번만 읽어 중단 사건과 대기 폐기에 같은 시각을 쓰므로, 그 뒤 시계가 예외를 내도 중단 결과를 반환하고 한 번만 정리한다."""
        armed = {"reads_left": None}
        base = FakeClock()

        def clock():
            """무장 뒤에는 남은 횟수만큼 정상 값을 주고, 그다음 조회는 예외(close 안의 두 번째 조회 실패 모사)."""
            left = armed["reads_left"]

            if left is not None:
                if left == 0:
                    raise RuntimeError("second clock read inside close")

                armed["reads_left"] = left - 1

            return base()

        p, job = self.make(clock)
        armed["reads_left"] = 1   # close에서 첫 조회만 성공
        calls_before = base.calls

        (out,) = p.close()

        self.assertEqual(base.calls - calls_before, 1)
        self.assertIs(out.status, JobStatus.ABORTED)
        self.assertIs(out.context, job)
        self.assertEqual(out.invalidation.event_monotonic_ns, p.queue_drop_records[-1].dropped_monotonic_ns)
        self.assertTrue(p.closed)

        c = p.counters
        self.assertEqual((c.aborted, len(p.invalidation_events)), (1, 1))
        self.assertEqual(c.queue_dropped, {"closed": 1})
        self.assertEqual(p.close(), ())
        self.assertEqual((p.counters.aborted, len(p.invalidation_events)), (1, 1))

    def test_close_clock_exception_keeps_state_and_retry_returns_abort_once(self):
        """close의 시계 조회가 예외면 닫힘·활성·대기·사건·계수가 그대로이고, 시계 정상화 뒤 재시도가 중단 결과를 정확히 한 번 반환한다."""
        broken = [False]
        base = FakeClock()

        def clock():
            """broken이면 예외, 아니면 정상 단조 시계."""
            if broken[0]:
                raise RuntimeError("clock unavailable")

            return base()

        p, job = self.make(clock)
        broken[0] = True
        before = self.state(p)

        with self.assertRaises(RuntimeError):
            p.close()

        self.assertFalse(p.closed)
        self.assertIs(p._active, job)
        self.assertEqual(self.state(p), before)
        self.assertNotIn(job.job_seq, p._terminal)

        broken[0] = False
        (out,) = p.close()

        self.assertIs(out.status, JobStatus.ABORTED)
        self.assertFalse(out.duplicate)
        c = p.counters
        self.assertEqual((c.aborted, len(p.invalidation_events)), (1, 1))
        self.assertEqual(c.queue_dropped, {"closed": 1})
        self.assertIsNone(p.pending_frame)
        self.assertEqual(p.close(), ())
        self.assert_job_balance(p)


class SeededCycleLivenessTest(LivenessTestBase):
    """고정 시드 반복 순환: 정상·무검출·실패·중단·차단·세대 변경·늦은/중복 통지·잘못된 관측·재처리를 섞는다."""

    MODES = ("valid", "no_object", "fail", "abort", "config_invalid", "generation_then_late",
             "detector_then_late_and_reprocess", "frameless_then_late", "job_event_then_complete",
             "apply_then_fail", "bad_identity", "bad_type", "tampered_reason", "pending_replaced")

    def run_cycle(self, p, rng, seq, configs):
        """작업 하나를 무작위 경로로 끝까지 정리한다. 시작한 작업 목록을 돌려준다."""
        job = self.start(seq, p)
        mode = rng.choice(self.MODES)
        jobs = [job]

        if mode == "valid":
            p.complete(job, valid_observation(job))
        elif mode == "no_object":
            p.complete(job, observation(job))
        elif mode == "fail":
            p.fail(job, INF, f"inference-{seq}")
        elif mode == "abort":
            p.fail(job, ABORT, f"timeout-{seq}")
        elif mode == "config_invalid":
            p.fail(job, InvalidationKind.MODEL_CONFIG_INVALID, f"class-map-{seq}")
        elif mode == "generation_then_late":
            p.change_settings(p.settings, cause=rng.choice((GenerationCause.SEEK, GenerationCause.RECONNECT)))
            self.assertIs(p.complete(job, valid_observation(job)).status, JobStatus.DISCARDED_GENERATION)
        elif mode == "detector_then_late_and_reprocess":
            new = configs[(configs.index(p.settings.detector_config) + 1) % len(configs)]
            p.change_settings(dataclasses.replace(p.settings, detector_config=new),
                              cause=GenerationCause.DETECTOR_CONFIG_CHANGED)
            p.complete(job, valid_observation(job))
            re_job = p.start_reprocess()
            self.assertIsNotNone(re_job)
            jobs.append(re_job)
            p.complete(re_job, valid_observation(re_job))
        elif mode == "frameless_then_late":
            p.apply_invalidation(frameless_event(p, t=seq + 1, cause=f"load-{seq}"))
            self.assertFalse(p.complete(job, valid_observation(job)).applied)
        elif mode == "job_event_then_complete":
            p.apply_invalidation(job_event(job, t=seq + 1))
            self.assertFalse(p.complete(job, valid_observation(job)).applied)
        elif mode == "apply_then_fail":
            event = job_event(job, cause=f"same-{seq}", t=seq + 1)
            p.apply_invalidation(event)
            p._clock_ns = lambda t=seq + 1: t   # 같은 9필드 사건이 되도록 시각을 맞춘다
            p.fail(job, INF, f"same-{seq}")
            p._clock_ns = FakeClock(start_ns=10_000_000 + seq * 1_000)
        elif mode == "bad_identity":
            name = rng.choice(ID_FIELDS)
            p.complete(job, dataclasses.replace(valid_observation(job), **{name: np.array([seq, seq])}))
        elif mode == "bad_type":
            p.complete(job, rng.choice((None, "obs", {"valid": True})))
        elif mode == "tampered_reason":
            p.complete(job, tampered(observation(job), reason=[seq]))
        else:  # pending_replaced: 처리 중 두 프레임이 와서 하나는 교체 폐기
            p.offer_frame(frame(seq + 10_000, p.current_generation(), source_id=p.settings.source_id))
            p.offer_frame(frame(seq + 20_000, p.current_generation(), source_id=p.settings.source_id))
            p.complete(job, valid_observation(job))
            nxt = p.start_next()
            jobs.append(nxt)
            p.complete(nxt, observation(nxt))

        # 늦은·중복 통지(무작위): 최초 terminal 유지, 아무것도 다시 반영하지 않음
        if rng.random() < 0.5:
            before = self.state(p)
            target = rng.choice(jobs)
            again = rng.choice((lambda: p.complete(target, valid_observation(target)),
                                lambda: p.fail(target, INF, "late")))()
            self.assertTrue(again.duplicate)
            self.assertEqual(self.state(p), before)

        # 매 순환 뒤 활성 작업 없음(다음 작업 시작 가능)
        self.assertIsNone(p._active)
        self.assert_running_balance(p)

        return jobs

    def test_seeded_mixed_cycles_keep_progress(self):
        """시드 0~19 × 250작업(총 5,000+)에서 모든 경로 뒤 활성 슬롯이 비고, 종료 후 보존식·사건 멱등이 성립하며 과거 영상이 회수되고 시간 상한 안에 끝난다."""
        configs = [config(), config(fixed_threshold=90)]
        started_at = time.monotonic()
        total = 0
        image_refs = []

        for seed in SEEDS:
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                p = VisionPipeline(settings(), run_id=RUN_ID, clock_ns=FakeClock())

                for seq in range(JOBS_PER_SEED):
                    jobs = self.run_cycle(p, rng, seq, configs)
                    total += len(jobs)

                    if seq % 50 == 0:
                        image_refs.append(weakref.ref(jobs[0].frame.image))

                    del jobs

                # 종료 후 보존식(활성·대기 없음)과 사건 목록의 중복 없음
                p.close()
                c = p.counters
                self.assertEqual(c.received, c.first_started + sum(c.queue_dropped.values()))
                self.assert_job_balance(p)
                events = p.invalidation_events
                self.assertEqual(len(events), len(set(events)))
                counted = sum(c.invalidation_applied_by_kind.values()) + sum(c.invalidation_ignored_by_kind.values())
                self.assertEqual(counted, len(events))
                self.assertGreaterEqual(c.first_started, JOBS_PER_SEED)
                del p

        gc.collect()
        elapsed = time.monotonic() - started_at

        self.assertGreaterEqual(total, len(SEEDS) * JOBS_PER_SEED)
        self.assertTrue(all(r() is None for r in image_refs), "old frames must be collectable after close")
        self.assertLess(elapsed, TIME_LIMIT_S)


if __name__ == "__main__":
    unittest.main()
