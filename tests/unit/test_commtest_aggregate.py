"""통신 시험 취합 순수 로직: 분위수(최근접 순위), 4분류 비율, 참조값(판정 없음), local_mock 표시·executor 설정 병기."""
import unittest

from eyetoarm.commtest.aggregate import percentile, summarize, to_markdown


def row(case="S1", via="client", acceptance="first_ack", ack=1.0, final=2.0, outcome="match", code=None, rej=None):
    """기록 행 하나."""
    return {"scenario": case[:2], "case": case, "via": via, "acceptance": acceptance, "ack_latency_ms": ack,
            "final_latency_ms": final, "outcome": outcome, "code": code, "rejected_code": rej,
            "fault_injected": via == "injector"}


class PercentileTest(unittest.TestCase):
    """최근접 순위 분위수."""

    def test_values(self):
        """빈 목록 None, 1개는 그 값, 1..20의 P95=19·P50=10·P100=20."""
        xs = list(range(1, 21))

        self.assertIsNone(percentile([], 95))
        self.assertEqual(percentile([7], 95), 7)
        self.assertEqual((percentile(xs, 95), percentile(xs, 50), percentile(xs, 100)), (19, 10, 20))
        self.assertEqual(percentile([3, 3, 3, 1], 50), 3)

    def test_bounds(self):
        """q 범위 밖은 ValueError, None 값은 뺀다."""
        with self.assertRaises(ValueError):
            percentile([1], 101)

        self.assertEqual(percentile([None, 4], 50), 4)


class SummaryTest(unittest.TestCase):
    """취합 사전·표."""

    def test_acceptance_ratio_client_rows_only(self):
        """4분류 분모는 클라이언트 행(acceptance 있음)만, 주입기 행은 빠진다."""
        rows = [row(), row(), row(acceptance="accepted_by_status"), row(case="S2a", via="injector", acceptance=None)]
        s = summarize(rows, peer="local_mock", executor_settings={"mode": "reach"})

        self.assertEqual(s["acceptance_denominator"], 3)
        self.assertAlmostEqual(s["acceptance_ratio"]["first_ack"], 2 / 3)
        self.assertEqual(s["fault_injected_rows"], 1)

    def test_reference_not_judged_and_mock_label(self):
        """참조값은 수치와 「판정하지 않음」만 있고 PASS/FAIL 키가 없다. local_mock 표시가 있다."""
        s = summarize([row()], peer="local_mock", executor_settings={})

        self.assertEqual((s["reference"]["comm_count_target"], s["reference"]["p95_ms_target"]), (100, 300))
        self.assertNotIn("pass", {k.lower() for k in s["reference"]})
        self.assertIn("실제 C 서버", s["peer_note"])

    def test_markdown_has_executor_next_to_latency(self):
        """S1 지연 표 바로 앞에 가짜 executor 설정이 있고, 실서버 비교 문장이 없다."""
        md = to_markdown(summarize([row(), row(ack=3.0)], peer="local_mock", executor_settings={"mode": "reach",
                                                                                                "moving_s": 0.02}))
        head, _, rest = md.partition("## S1 지연(ms)")

        self.assertIn("'moving_s': 0.02", rest.split("| 구간")[0])
        self.assertNotIn("실서버보다", md)

    def test_skipped_and_mismatch_counted(self):
        """건너뜀은 케이스 표에 들어가고, 불일치가 있으면 all_match가 거짓이다."""
        s = summarize([row(outcome="mismatch")], peer="local_mock", executor_settings={}, skipped=["S2a"])

        self.assertEqual(s["cases"]["S2a"]["skipped"], 1)
        self.assertFalse(s["all_match"])


if __name__ == "__main__":
    unittest.main()
