"""코어 읽기 전용 `invalidation_records`(#142 D1 안 A) 시험 — 사건마다 (event, applied) 1건, 기존 사건·판정·계수 의미 무변.

근거: `_workspace/142_builder2_a_c_handoff_contract.md` r3 §1·§4-3·§7 D1·§8, A 상세설계 §5.4 규칙 1~3.
"""
import dataclasses
import unittest

from eyetoarm.common.types import InvalidationKind
from eyetoarm.vision.pipeline import GenerationCause
from eyetoarm.vision.pipeline_types import InvalidationRecord
from test_pipeline import CHANGES
from test_pipeline_terminal import ABORT, INF, TerminalTestBase, frameless_event, job_event


class InvalidationRecordsTest(TerminalTestBase):
    """records와 events가 1:1이고 applied 판정이 계수와 같은지 확인한다."""

    def assert_records_match(self, p=None):
        """records의 사건 순서 = invalidation_events, applied/ignored kind별 수 = 코어 계수."""
        p = p or self.p
        records = p.invalidation_records
        c = p.counters

        self.assertEqual(tuple(r.event for r in records), p.invalidation_events)
        applied, ignored = {}, {}

        for r in records:
            bucket = applied if r.applied else ignored
            bucket[r.event.kind.value] = bucket.get(r.event.kind.value, 0) + 1

        self.assertEqual(applied, c.invalidation_applied_by_kind)
        self.assertEqual(ignored, c.invalidation_ignored_by_kind)

    def test_empty_at_start_and_read_only_tuple(self):
        """처음에는 빈 tuple이고, 돌려준 tuple을 바꿔도 코어 기록이 바뀌지 않는다(읽기 전용 조회)."""
        self.assertEqual(self.p.invalidation_records, ())
        self.assertIsInstance(self.p.invalidation_records, tuple)

    def test_record_is_frozen_pair(self):
        """기록 한 건은 고칠 수 없는 (event, applied) 쌍이다."""
        job = self.start(0)
        self.p.fail(job, INF, "boom")
        rec = self.p.invalidation_records[0]

        self.assertIsInstance(rec, InvalidationRecord)
        self.assertTrue(rec.applied)

        with self.assertRaises(dataclasses.FrozenInstanceError):
            rec.applied = False

    def test_current_generation_job_event_applied(self):
        """현재 세대 작업 실패 사건은 applied=True 1건이며 응답의 사건 객체와 같다."""
        job = self.start(0)
        out = self.p.fail(job, INF, "boom")

        self.assertEqual(self.p.invalidation_records, (InvalidationRecord(out.invalidation, True),))
        self.assert_records_match()

    def test_previous_generation_event_ignored(self):
        """세대 전이 뒤 이전 세대 작업 사건은 applied=False로 남고, 전이 사건은 applied=True다."""
        job = self.start(0)
        gen_event = self.p.change_settings(CHANGES[GenerationCause.SEEK](self.settings), cause=GenerationCause.SEEK)
        out = self.p.fail(job, ABORT, "late")

        self.assertEqual(self.p.invalidation_records,
                         (InvalidationRecord(gen_event, True), InvalidationRecord(out.invalidation, False)))
        self.assertEqual(out.invalidation.generation, 0)   # 원래 세대 보존
        self.assert_records_match()

    def test_duplicate_apply_adds_nothing(self):
        """9필드가 같은 사건을 다시 apply하면 records가 늘지 않는다."""
        job = self.start(0)
        event = job_event(job)

        self.assertTrue(self.p.apply_invalidation(event))
        self.assertFalse(self.p.apply_invalidation(event))
        self.assertEqual(len(self.p.invalidation_records), 1)
        self.assert_records_match()

    def test_apply_then_fail_same_event_once(self):
        """apply 뒤 같은 9필드로 fail하면 사건은 1건만 기록된다."""
        job = self.start(0)
        self.p.apply_invalidation(job_event(job, cause="boom", t=self.clock.now))
        self.p.fail(job, INF, "boom")

        self.assertEqual(len(self.p.invalidation_records), 1)
        self.assert_records_match()

    def test_close_abort_event(self):
        """close의 활성 작업 중단 사건이 JOB_ABORTED applied 1건으로 남는다."""
        self.start(0)
        self.p.close()
        (rec,) = self.p.invalidation_records

        self.assertIs(rec.event.kind, InvalidationKind.JOB_ABORTED)
        self.assertTrue(rec.applied)
        self.assert_records_match()

    def test_frameless_event(self):
        """프레임 없는 로딩 오류 사건은 job_seq·image_seq None 그대로 applied 1건이다."""
        event = frameless_event(self.p)
        self.p.apply_invalidation(event)

        self.assertEqual(self.p.invalidation_records, (InvalidationRecord(event, True),))
        self.assertIsNone(self.p.invalidation_records[0].event.job_seq)

    def test_job0_event_after_job1_current_keeps_core_semantics(self):
        """job1 현재 상태에서 job0 사건은 applied=True로 기록되지만 코어 현재 관측(job1)은 그대로다(판정 의미 무변)."""
        job0, _ = self.run_valid(0)
        job1, _ = self.run_valid(1)
        current = self.p.current_observation
        self.assertEqual(current.job_seq, job1.job_seq)

        self.assertTrue(self.p.apply_invalidation(job_event(job0)))
        self.assertIs(self.p.current_observation, current)
        self.assertTrue(self.p.invalidation_records[-1].applied)
        self.assert_records_match()


if __name__ == "__main__":
    unittest.main()
