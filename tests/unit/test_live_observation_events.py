"""`vision/live_observation_events.LiveObservationEventWriter`(#172 r1 F2) 단위시험: 단일 쓰기 소유자, 쓰기 실패 보존·보고·fail-stop, 종료 상한.

근거: `_workspace/171_builder2_live_a_to_b_plan.md` r3 §3.3, reviewer #172 r1 F2. 쓰기 오류는 파일을 닫아 주입하고,
종료 경쟁은 쓰기 도중 정지시키는 장벽(Event)으로 재현한다. 상한 값은 결정 L-8 전 시험 후보다.
"""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from eyetoarm.vision.live_observation_events import LiveObservationEventWriter
from eyetoarm.vision.live_observation_state import LiveObservationConfig, LiveObservationState


def make(w=8, r=8):
    """상태와 쓰기 담당."""
    st = LiveObservationState(run_id="r", generation=0, calibration_id="c", config=LiveObservationConfig(write_capacity=w, reserved_capacity=r))
    wr = LiveObservationEventWriter(st, str(Path(tempfile.mkdtemp()) / "out"), lag_warn_s=2.0, batch=8)
    return st, wr


def accept_n(st, n):
    """수락 n건을 넣는다(payload 없음)."""
    tok = st.token
    return [st.accept(st.begin_compute(tok), tok, None) for _ in range(n)]


def lines(wr):
    """쓴 기록."""
    return [json.loads(x) for x in (wr.dir / "b_live_events.jsonl").read_text().splitlines()]


class WriteFailureTest(unittest.TestCase):
    def test_f2_closed_file_error_preserves_and_fail_stops(self):
        """F2 재현: 대기 2건 + 파일 닫힘 → _loop: error 기록, 2건 unwritten 보존(순서), state fail-stop, 입력 차단."""
        st, wr = make()
        seqs = accept_n(st, 2)
        wr._fh.close()
        wr._loop()
        self.assertIsNotNone(wr.error)
        self.assertEqual(wr.written, 0)
        self.assertEqual([e.event_seq for e in wr.unwritten], seqs)
        self.assertEqual((st.closed, st.failed), (True, "b_writer_error"))
        self.assertIsNone(st.admit(rx_epoch=0, generation=0, calibration_id="c"))
        rep = wr.close(1.0)
        self.assertEqual((rep.written, rep.unwritten, rep.error is not None), (0, 2, True))

    def test_failure_mid_batch_keeps_rest(self):
        """묶음 중간 실패: 앞 기록은 쓰고, 실패 기록부터 나머지는 unwritten에 순서대로 남는다."""
        st, wr = make()
        seqs = accept_n(st, 3)
        real = wr._write_one
        calls = []

        def flaky(event):
            calls.append(event.event_seq)
            if len(calls) == 2:
                raise OSError("disk full")
            real(event)

        wr._write_one = flaky
        self.assertEqual(wr.drain_once(), 1)
        self.assertEqual([e.event_seq for e in wr.unwritten], seqs[1:])
        self.assertEqual(st.failed, "b_writer_error")
        self.assertEqual(wr.drain_once(), 0)          # 실패 뒤에는 쓰지 않는다
        self.assertEqual([r["event_seq"] for r in lines(wr)], seqs[:1])

    def test_thread_failure_reported_by_close(self):
        """스레드 쓰기 실패도 close 보고에 error·unwritten으로 드러난다."""
        st, wr = make()
        accept_n(st, 2)
        wr._fh.close()
        wr.start()
        rep = wr.close(1.0)
        self.assertFalse(rep.timed_out)
        self.assertEqual(rep.unwritten, 2)
        self.assertTrue(st.closed)


class OwnershipAndCloseTest(unittest.TestCase):
    def test_caller_cannot_drain_while_thread_owns(self):
        """스레드 시작 뒤 다른 스레드의 drain_once는 RuntimeError(단일 쓰기 소유자)."""
        st, wr = make()
        wr.start()
        with self.assertRaises(RuntimeError):
            wr.drain_once()
        wr.close(1.0)

    def test_f2_close_timeout_does_not_race_live_writer(self):
        """F2 장벽: 쓰기 도중 멈춘 스레드 → close(0.1)은 상한 안에 timed_out 반환, 호출자는 파일·대기열에 손대지 않음, fail-stop.
        풀린 뒤 스레드가 스스로 파일을 닫는다."""
        st, wr = make()
        accept_n(st, 3)
        in_write, go = threading.Event(), threading.Event()
        real = wr._write_one
        writers = []

        def slow(event):
            writers.append(threading.current_thread().name)
            if event.event_seq == 1:
                in_write.set()
                go.wait(3.0)
            real(event)

        wr._write_one = slow
        wr.start()
        self.assertTrue(in_write.wait(2.0))
        t0 = time.monotonic()
        rep = wr.close(0.1)
        took = time.monotonic() - t0
        self.assertTrue(rep.timed_out)
        self.assertLess(took, 0.5)
        self.assertFalse(wr.file_closed)               # 호출자가 닫지 않음
        self.assertEqual(st.failed, "b_writer_close_timeout")
        go.set()
        wr._thread.join(2.0)
        self.assertFalse(wr._thread.is_alive())
        self.assertTrue(wr.file_closed)                # 스레드가 닫음
        self.assertEqual(set(writers), {"eyetoarm-live-observation-writer"})   # 쓴 것은 스레드뿐
        recs = lines(wr)
        self.assertEqual([r["event_seq"] for r in recs], sorted(r["event_seq"] for r in recs))
        self.assertGreaterEqual(len(recs), 1)
        self.assertEqual(len(recs) + wr.state.pending_events() + len(wr.unwritten), 3)

    def test_close_drains_within_limit(self):
        """정상 종료: 스레드가 기한 안에 남은 기록을 모두 쓰고 닫는다."""
        st, wr = make()
        wr.start()
        accept_n(st, 5)
        st.run_end("end")
        rep = wr.close(2.0)
        self.assertEqual((rep.timed_out, rep.pending, rep.unwritten, rep.error), (False, 0, 0, None))
        self.assertEqual([r["kind"] for r in lines(wr)][-1], "run_end")
        self.assertTrue(wr.file_closed)

    def test_close_without_thread_caller_owns(self):
        """스레드를 시작하지 않았으면 호출자가 소유자로서 비우고 닫는다."""
        st, wr = make()
        accept_n(st, 2)
        rep = wr.close(1.0)
        self.assertEqual((rep.written, rep.pending, rep.timed_out), (2, 0, False))
        self.assertTrue(wr.file_closed)


if __name__ == "__main__":
    unittest.main()
