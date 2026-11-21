"""`vision/live_observation_state.LiveObservationState` 단위시험."""
import unittest

from eyetoarm.vision.live_observation_state import LiveObservationConfig, LiveObservationState

REWIND, RECONNECT, PROFILE = "sim_time_rewind", "reconnect", "profile_changed"


def make_state(w=4, r=8, generation=0, cal="cal-a"):
    """시험 상태(열린 처음 토큰, 수신 epoch 0)."""
    return LiveObservationState(run_id="run1", generation=generation, calibration_id=cal,
                      config=LiveObservationConfig(write_capacity=w, reserved_capacity=r))


def snapshot(st):
    """비교용 상태 사본(토큰·경계·계수 제외)."""
    return st.token, st.boundary


class ConfigTest(unittest.TestCase):
    def test_capacities_required_positive(self):
        """용량은 양의 정수만 받는다(기본값 없음)."""
        for bad in (0, -1, 1.5, None):
            with self.assertRaises(ValueError):
                LiveObservationConfig(write_capacity=bad, reserved_capacity=1)
        with self.assertRaises(TypeError):
            LiveObservationConfig()  # noqa: 기본값이 없어야 한다


class AdmitTest(unittest.TestCase):
    def test_admit_matches_epoch_generation_calibration(self):
        """제출은 수신 epoch·세대·보정 번호가 현재 토큰과 같을 때만 토큰을 받는다."""
        st = make_state()
        self.assertIsNotNone(st.admit(rx_epoch=0, generation=0, calibration_id="cal-a"))
        self.assertIsNone(st.admit(rx_epoch=1, generation=0, calibration_id="cal-a"))
        self.assertIsNone(st.admit(rx_epoch=0, generation=1, calibration_id="cal-a"))
        self.assertIsNone(st.admit(rx_epoch=0, generation=0, calibration_id="cal-b"))
        self.assertIsNone(st.admit(rx_epoch=None, generation=0, calibration_id="cal-a"))

    def test_rx_boundary_closes_immediately(self):
        """수신 경계는 제어 흐름 반영 전에도 즉시 닫는다(제출·①·② 모두 실패)."""
        st = make_state()
        tok = st.token
        st.on_rx_boundary(1, RECONNECT)
        self.assertFalse(st.token.ready)
        self.assertFalse(st.is_current(tok))
        self.assertIsNone(st.begin_compute(tok))
        self.assertIsNone(st.admit(rx_epoch=1, generation=0, calibration_id="cal-a"))


class BoundaryReadinessTest(unittest.TestCase):
    def test_o1_tf_first_then_applied(self):
        """TF 준비가 먼저 와도 제어 반영 전에는 닫혀 있고, 반영에서 열린다."""
        st = make_state()
        st.on_rx_boundary(1, REWIND)
        self.assertTrue(st.on_tf_ready(1, st.begin_tf_attempt(1)))
        self.assertFalse(st.token.ready)
        self.assertTrue(st.on_applied(1, 1, "cal-a"))
        self.assertTrue(st.token.ready)
        self.assertEqual((st.token.rx_epoch, st.token.generation), (1, 1))

    def test_o2_applied_first_then_tf(self):
        """제어 반영이 먼저 와도 같은 결과(순서 무관)."""
        st = make_state()
        st.on_rx_boundary(1, REWIND)
        st.on_applied(1, 1, "cal-a")
        self.assertFalse(st.token.ready)
        st.on_tf_ready(1, st.begin_tf_attempt(1))
        self.assertTrue(st.token.ready)
        self.assertEqual((st.token.rx_epoch, st.token.generation), (1, 1))

    def test_o3_late_applied_of_previous_boundary_changes_nothing(self):
        """B1 → B2 → on_applied(B1)은 어떤 상태 필드도 바꾸지 않는다."""
        st = make_state()
        st.on_rx_boundary(1, RECONNECT)
        st.on_rx_boundary(2, RECONNECT)
        before = snapshot(st)
        self.assertFalse(st.on_applied(1, 5, "cal-z"))
        self.assertEqual(snapshot(st), before)
        self.assertEqual(st.counts["stale_applied_ignored"], 1)
        self.assertFalse(st.token.ready)

    def test_o4_late_tf_ready_of_previous_boundary_changes_nothing(self):
        """B1(되감기) → B2(되감기) → on_tf_ready(B1) 불변, B2는 TF 준비 뒤에만 열림."""
        st = make_state()
        st.on_rx_boundary(1, REWIND)
        a1 = st.begin_tf_attempt(1)
        st.on_rx_boundary(2, REWIND)
        before = snapshot(st)
        self.assertFalse(st.on_tf_ready(1, a1))
        self.assertIsNone(st.begin_tf_attempt(1))
        self.assertEqual(snapshot(st), before)
        self.assertEqual(st.counts["stale_tf_ready_ignored"], 1)
        st.on_applied(2, 2, "cal-a")
        self.assertFalse(st.token.ready)
        st.on_tf_ready(2, st.begin_tf_attempt(2))
        self.assertTrue(st.token.ready)

    def test_o5_non_rewind_opens_on_applied(self):
        """되감기가 아닌 경계는 TF 준비 없이 제어 반영만으로 열린다."""
        st = make_state()
        st.on_rx_boundary(1, RECONNECT)
        self.assertTrue(st.boundary.tf_ready)
        st.on_applied(1, 1, "cal-a")
        self.assertTrue(st.token.ready)

    def test_p_half_applied_does_not_open(self):
        """B1 → B2 → B1만 반영 → 열리지 않음."""
        st = make_state()
        st.on_rx_boundary(1, PROFILE)
        st.on_rx_boundary(2, RECONNECT)
        st.on_applied(1, 1, "cal-b")
        self.assertFalse(st.token.ready)
        self.assertFalse(st.boundary.applied)
        self.assertEqual(st.token.calibration_id, "cal-a")   # 이전 경계 값으로 바꾸지 않음

    def test_profile_change_updates_calibration_on_open(self):
        """보정 변경 경계의 반영 값(코어 현재 값)이 토큰에 들어간다."""
        st = make_state()
        st.on_rx_boundary(1, PROFILE)
        st.on_applied(1, 1, "cal-b")
        self.assertEqual((st.token.calibration_id, st.token.ready), ("cal-b", True))
        self.assertIsNotNone(st.admit(rx_epoch=1, generation=1, calibration_id="cal-b"))
        self.assertIsNone(st.admit(rx_epoch=1, generation=1, calibration_id="cal-a"))

    def test_non_monotonic_boundary_ignored(self):
        """같거나 작은 경계 번호는 무시한다."""
        st = make_state()
        st.on_rx_boundary(2, RECONNECT)
        before = snapshot(st)
        self.assertFalse(st.on_rx_boundary(2, RECONNECT))
        self.assertFalse(st.on_rx_boundary(1, RECONNECT))
        self.assertEqual(snapshot(st), before)

    def test_tf_anomaly_reverts_tf_ready(self):
        """준비 뒤 TF 이상은 그 경계 TF 준비를 되돌려 닫고, 다시 준비돼야 열린다."""
        st = make_state()
        st.on_rx_boundary(1, REWIND)
        st.on_applied(1, 1, "cal-a")
        a1 = st.begin_tf_attempt(1)
        st.on_tf_ready(1, a1)
        self.assertTrue(st.on_tf_anomaly(1, "future"))
        self.assertFalse(st.token.ready)
        self.assertFalse(st.on_tf_anomaly(0, "old"))
        self.assertFalse(st.on_tf_ready(1, a1))          # 이상 전 시도로는 다시 열 수 없다
        self.assertFalse(st.token.ready)
        st.on_tf_ready(1, st.begin_tf_attempt(1))
        self.assertTrue(st.token.ready)

    def test_f1_anomaly_during_attempt_blocks_that_attempt(self):
        """시도 중 이상 → 그 시도의 준비 완료는 무시, 새 시도로만 열림. 이전 시도 번호·0은 거부."""
        st = make_state()
        st.on_rx_boundary(1, REWIND)
        st.on_applied(1, 1, "cal-a")
        a1 = st.begin_tf_attempt(1)
        st.on_tf_anomaly(1, "future during lookup")
        self.assertFalse(st.on_tf_ready(1, a1))
        self.assertFalse(st.on_tf_ready(1, 0))
        self.assertFalse(st.token.ready)
        self.assertEqual(st.counts["stale_tf_attempt_ready_ignored"], 2)
        a2 = st.begin_tf_attempt(1)
        self.assertGreater(a2, a1)
        self.assertTrue(st.on_tf_ready(1, a2))
        self.assertTrue(st.token.ready)

    def test_tf_attempt_not_needed_when_ready(self):
        """TF 준비가 이미 참이면(비되감기 경계) 시도를 만들지 않는다(상태 무변)."""
        st = make_state()
        st.on_rx_boundary(1, RECONNECT)
        before = snapshot(st)
        self.assertIsNone(st.begin_tf_attempt(1))
        self.assertEqual(snapshot(st), before)

    def test_fail_blocks_everything(self):
        """fail()은 닫고 제출·계산·수락을 막으며 기록을 더 넣지 않는다."""
        st = make_state()
        tok = st.token
        cid = st.begin_compute(tok)
        st.fail("b_writer_error")
        st.fail("again")
        self.assertEqual((st.failed, st.closed), ("b_writer_error", True))
        self.assertIsNone(st.admit(rx_epoch=0, generation=0, calibration_id="cal-a"))
        self.assertIsNone(st.begin_compute(st.token))
        self.assertIsNone(st.accept(cid, tok, "x"))
        self.assertEqual(st.take_events(99), [])


class ComputeAcceptTest(unittest.TestCase):
    def test_invalid_before_begin_gives_no_compute(self):
        """② 전 무효화: begin_compute None(B 호출 금지)."""
        st = make_state()
        tok = st.token
        st.on_rx_boundary(1, RECONNECT)
        self.assertIsNone(st.begin_compute(tok))
        self.assertEqual(st.counts["stale_after_tf"], 1)

    def test_invalid_after_begin_blocks_accept(self):
        """② 후 무효화: 계산 중 표지 → 수락 0."""
        st = make_state()
        tok = st.token
        cid = st.begin_compute(tok)
        st.on_rx_boundary(1, RECONNECT)
        self.assertIsNone(st.accept(cid, tok, "out"))
        self.assertEqual(st.counts["stale_after_compute"], 1)
        kinds = [e.kind for e in st.take_events(99)]
        self.assertNotIn("accept", kinds)

    def test_token_update_without_cause_also_blocks(self):
        """경계 뒤 반영으로 세대·보정이 바뀌어도 계산 중이던 항목은 수락되지 않는다."""
        st = make_state()
        tok = st.token
        cid = st.begin_compute(tok)
        st.on_rx_boundary(1, PROFILE)
        st.on_applied(1, 1, "cal-b")
        self.assertIsNone(st.accept(cid, tok, "out"))

    def test_accept_then_invalidate_revokes(self):
        """수락 → 경계 → 철회가 event_seq 순서로 남고 is_current가 거짓이 된다."""
        st = make_state()
        tok = st.token
        seq = st.accept(st.begin_compute(tok), tok, "out")
        self.assertTrue(st.is_current_accept(seq))
        st.on_rx_boundary(1, RECONNECT)
        self.assertFalse(st.is_current_accept(seq))
        events = st.take_events(99)
        self.assertEqual([e.kind for e in events], ["accept", "revoke", "invalidation"])
        self.assertEqual([e.event_seq for e in events], sorted(e.event_seq for e in events))
        self.assertEqual(events[1].revoked_event_seq, seq)

    def test_newer_accept_supersedes_without_revoke(self):
        """더 새 수락은 이전 것을 현재에서 내리지만 철회 기록은 없다."""
        st = make_state()
        tok = st.token
        a = st.accept(st.begin_compute(tok), tok, "a")
        b = st.accept(st.begin_compute(tok), tok, "b")
        self.assertFalse(st.is_current_accept(a))
        self.assertTrue(st.is_current_accept(b))
        self.assertEqual([e.kind for e in st.take_events(99)], ["accept", "accept"])

    def test_unknown_compute_id_rejected(self):
        """모르는 compute_id·다른 토큰은 오류(조용히 수락하지 않음)."""
        st = make_state()
        with self.assertRaises(ValueError):
            st.accept(99, st.token, "x")

    def test_abandon_removes_without_output(self):
        """계산 예외는 abandon으로 출력 없이 끝난다."""
        st = make_state()
        cid = st.begin_compute(st.token)
        st.abandon(cid, "RuntimeError")
        self.assertEqual(st.take_events(99), [])
        with self.assertRaises(ValueError):
            st.accept(cid, st.token, "x")


class WriterCapacityTest(unittest.TestCase):
    def test_q_backpressure_refuses_accept(self):
        """쓰기 담당이 멈춘 상태에서 W+1번째 수락은 writer_backpressure(출력 0), 버림 없음."""
        st = make_state(w=2, r=8)
        tok = st.token
        seqs = [st.accept(st.begin_compute(tok), tok, i) for i in range(3)]
        self.assertIsNotNone(seqs[0])
        self.assertIsNotNone(seqs[1])
        self.assertIsNone(seqs[2])
        self.assertEqual(st.counts["writer_backpressure"], 1)
        events = st.take_events(99)
        self.assertEqual([e.payload for e in events], [0, 1])

    def test_q_reserved_overflow_fail_stop(self):
        """예약 칸이 차면 fail-stop(b_writer_overflow), 이후 제출·계산 불가, 마지막 기록 failed."""
        st = make_state(w=2, r=2)
        st.on_rx_boundary(1, RECONNECT)     # invalidation 1
        st.on_applied(1, 0, "cal-a")       # ready 1 → 예약 칸 2개 참
        st.on_rx_boundary(2, RECONNECT)    # invalidation → 넘침
        self.assertEqual(st.failed, "b_writer_overflow")
        self.assertTrue(st.closed)
        self.assertIsNone(st.admit(rx_epoch=2, generation=0, calibration_id="cal-a"))
        events = st.take_events(99)
        self.assertEqual(events[-1].kind, "failed")
        self.assertEqual([e.event_seq for e in events], sorted(e.event_seq for e in events))

    def test_space_frees_after_take(self):
        """꺼내면 칸이 비어 다시 수락한다."""
        st = make_state(w=1, r=8)
        tok = st.token
        st.accept(st.begin_compute(tok), tok, 0)
        self.assertIsNone(st.accept(st.begin_compute(tok), tok, 1))
        st.take_events(99)
        self.assertIsNotNone(st.accept(st.begin_compute(tok), tok, 2))

    def test_run_end_closes_and_records(self):
        """실행 종료: 닫힘, 계산 중 항목 수락 0, run_end 기록 마지막, 두 번 불러도 한 번."""
        st = make_state()
        tok = st.token
        cid = st.begin_compute(tok)
        st.run_end("end_of_stream")
        st.run_end("again")
        self.assertIsNone(st.accept(cid, tok, "x"))
        events = st.take_events(99)
        self.assertEqual([e.kind for e in events], ["invalidation", "run_end"])
        self.assertFalse(st.on_rx_boundary(5, RECONNECT))


if __name__ == "__main__":
    unittest.main()
