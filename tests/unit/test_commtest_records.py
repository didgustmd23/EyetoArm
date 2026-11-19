"""#183 통신 시험 기록 순수 로직: EventLog 행 → 명령별 시간 추적, 기록 행, 실행 폴더·파일 쓰기.

근거: `_workspace/183_builder_comm_test_tool_plan.md` r0 §4, TCP 명세서 §8.4(수락 4분류는 D 값을 그대로 옮김), 규약 §4(단조 시계)·§10(경로).
"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from eyetoarm.commtest.records import (CommandTracker, client_command_row, injector_command_row, make_run_dir,
                                       write_jsonl)
from eyetoarm.commtest.scenarios import CASES, C_COOPERATION, cases_of


def clock(values):
    """정해 둔 ns 값을 차례로 돌려주는 시계."""
    it = iter(values)
    return lambda: next(it)


def status_row(cid, state=None, code=None, mono=0):
    """RX STATUS EventLog 행."""
    msg = {"version": "1.1", "type": "STATUS", "command_id": cid}
    msg.update({k: v for k, v in (("state", state), ("code", code)) if v is not None})
    return {"dir": "RX", "kind": "STATUS", "command_id": cid, "raw": json.dumps(msg), "mono_ms": mono}


class TrackerTest(unittest.TestCase):
    """CommandTracker: 시간·상태 수집과 예외 금지(D-07)."""

    def test_send_first_status_final(self):
        """첫 TX(sent)·첫 RX·최종 RX 시각을 남기고 재전송 TX는 횟수만 올린다."""
        t = CommandTracker(clock_ns=clock([10, 20, 30, 40, 50]))

        t.on_row({"dir": "TX", "kind": "MOVE", "command_id": "c", "send_state": "sent", "mono_ms": 1})
        t.on_row({"dir": "TX", "kind": "MOVE", "command_id": "c", "send_state": "sent", "mono_ms": 2})
        t.on_row(status_row("c", "ACK", mono=3))
        t.on_row(status_row("c", "MOVING", mono=4))
        t.on_row(status_row("c", "REACHED", mono=5))
        x = t.timings["c"]

        self.assertEqual((x.tx_count, x.tx_first_ns, x.rx_first_ns, x.rx_first_state), (2, 10, 30, "ACK"))
        self.assertEqual((x.final_ns, x.final_state, x.final_mono_ms), (50, "REACHED", 5))
        self.assertEqual(x.states, ["ACK", "MOVING", "REACHED"])

    def test_rejection_is_final(self):
        """state 없는 거부(code)는 첫 상태이자 최종이다."""
        t = CommandTracker(clock_ns=clock([1, 2]))

        t.on_row({"dir": "TX", "kind": "MOVE", "command_id": "c", "send_state": "sent", "mono_ms": 0})
        t.on_row(status_row("c", code="BUSY"))

        self.assertEqual((t.timings["c"].final_code, t.timings["c"].rx_first_state), ("BUSY", "REJECT:BUSY"))

    def test_unsent_recorded_not_timed(self):
        """미전송 TX는 unsent 목록에만 남고 송신 시각이 생기지 않는다."""
        t = CommandTracker(clock_ns=clock([1]))

        t.on_row({"dir": "TX", "kind": "MOVE", "command_id": "u", "send_state": "unsent", "note": "준비 안 됨"})

        self.assertEqual(t.unsent, [("u", "준비 안 됨")])
        self.assertNotIn("u", t.timings)

    def test_bad_row_never_raises(self):
        """깨진 raw·이상한 행에도 예외를 내지 않고 errors에 남긴다."""
        t = CommandTracker(clock_ns=clock([1, 2]))

        t.on_row({"dir": "RX", "kind": "STATUS", "command_id": "c", "raw": "{not json"})
        t.on_row(None)

        self.assertEqual(len(t.errors), 2)


class RowTest(unittest.TestCase):
    """기록 행: D 수락 분류 그대로, 지연은 ns 차이."""

    def test_client_row(self):
        """acceptance·attempts는 D 값, 지연 ms는 ns 차이, 재전송 수 = attempts-1."""
        t = CommandTracker(clock_ns=clock([1_000_000, 3_500_000, 9_000_000]))
        t.on_row({"dir": "TX", "kind": "MOVE", "command_id": "c", "send_state": "sent", "mono_ms": 1})
        t.on_row(status_row("c", "ACK"))
        t.on_row(status_row("c", "REACHED"))
        rec = SimpleNamespace(command_id="c", attempts=2, acceptance="accepted_after_resend", state="REACHED",
                              stage=None, code=None, rejected=False)
        row = client_command_row(case=CASES[0], record=rec, timing=t.timings["c"], expected="e", outcome="match",
                                 fault_kind=None, peer="local_mock", executor_calls_delta=1)

        self.assertEqual((row["ack_latency_ms"], row["final_latency_ms"]), (2.5, 8.0))
        self.assertEqual((row["acceptance"], row["resend_count"], row["fault_injected"]), ("accepted_after_resend", 1, False))

    def test_injector_row_always_fault(self):
        """주입기 행은 항상 fault_injected이고 거부 코드는 rejected_code로 간다."""
        row = injector_command_row(case=CASES[0], command_id="c", sent_ns=1_000_000, reply={"code": "ID_CONFLICT"},
                                   reply_ns=2_000_000, expected="e", outcome="match", fault_kind="resend_changed",
                                   peer="local_mock", executor_calls_delta=0)

        self.assertEqual((row["fault_injected"], row["rejected_code"], row["ack_latency_ms"]), (True, "ID_CONFLICT", 1.0))


class FilesTest(unittest.TestCase):
    """실행 폴더·파일: 새로 만들기만."""

    def test_run_dir_new_only_and_safe_name(self):
        """같은 run_id는 FileExistsError, 경로 구분자·`..`는 ValueError."""
        base = tempfile.mkdtemp()

        make_run_dir(base, "r1")

        with self.assertRaises(FileExistsError):
            make_run_dir(base, "r1")

        for bad in ("../x", "a/b", "..", ".hidden", ""):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                make_run_dir(base, bad)

    def test_jsonl_exclusive(self):
        """JSONL은 새 파일만 쓰고 NaN을 거부한다."""
        p = Path(tempfile.mkdtemp()) / "a.jsonl"

        write_jsonl(p, [{"a": 1}])

        with self.assertRaises(FileExistsError):
            write_jsonl(p, [{"a": 1}])

        with self.assertRaises(ValueError):
            write_jsonl(Path(tempfile.mkdtemp()) / "b.jsonl", [{"a": float("nan")}])


class ScenarioTableTest(unittest.TestCase):
    """시나리오 표: 근거·dedupe 대응·C 협력 정의."""

    def test_dedupe_ids_exist_in_vectors(self):
        """S2 케이스의 dedupe 사례 ID가 test_vectors.json에 있고 기대(same/conflict)가 같다."""
        from eyetoarm.commtest.scenarios import S2_VARIANTS

        root = Path(__file__).resolve().parents[2]
        vectors = {c["id"]: c["expect"] for c in json.loads((root / "docs/protocol/test_vectors.json").read_text())["dedupe"]}

        for case_id, dedupe_id, _, _, expect in S2_VARIANTS:
            with self.subTest(case=case_id):
                self.assertEqual(vectors[dedupe_id], expect)

    def test_cases_and_cooperation(self):
        """S1~S4 케이스에 근거가 있고, C 협력 항목은 정의만 있다. 모르는 시나리오는 ValueError."""
        self.assertTrue(all(c.basis for c in CASES))
        self.assertEqual([c.case_id for c in cases_of("S4")], ["S4a", "S4b", "S4c"])
        self.assertEqual(len(C_COOPERATION), 5)

        with self.assertRaises(ValueError):
            cases_of("S9")


if __name__ == "__main__":
    unittest.main()
