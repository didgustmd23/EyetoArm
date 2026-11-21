"""AI/OpenCV 동일 조건 비교 시험: 같은 조건 판정, 조건 불일치 표시, 같은 방식 거부, FAKE_INPUT 전파, CLI."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from eyetoarm.sorteval import aggregate as A
from eyetoarm.sorteval import compare as C
from eyetoarm.sorteval import records as R
from eyetoarm.sorteval import sequence as S


def summary(seq, method, outcomes=None, conditions=None):
    """가짜 입력 취합 결과."""
    records, errors = R.parse_events_jsonl(R.events_to_jsonl(R.fake_events(seq, outcomes or {})))
    assert errors == []
    meta, _ = R.adapt_run_meta(R.fake_run_meta(seq, method=method, conditions=conditions))

    return A.aggregate(seq, records, meta)


def seq_of(seed=5):
    """비교용 순서표."""
    return S.generate_sequence(seed=seed, count=8, created_at_utc="2026-11-20T00:00:00Z")


class CompareTest(unittest.TestCase):
    """비교 표."""

    def test_same_conditions(self):
        """같은 순서표·조건에서 방식만 다르면 same_conditions, 차이 열은 b − a."""
        seq = seq_of()
        c = C.compare(summary(seq, "opencv"), summary(seq, "ai", {1: "isolated"}))
        acc = next(r for r in c["table"] if r["metric"] == "분류 정확도")

        self.assertTrue(c["same_conditions"])
        self.assertEqual(c["condition_differences"], [])
        self.assertAlmostEqual(acc["b_minus_a"], -1 / 8)
        self.assertEqual(acc["a_denominator"], 8)

        iso = next(r for r in c["table"] if r["metric"] == "실패 C:REDETECT_UNSTABLE")
        self.assertEqual((iso["a"], iso["b"], iso["b_minus_a"]), (0, 1, 1))

    def test_different_sequence_invalid(self):
        """순서표가 다르면 불일치 항목을 나열하고 MD 머리에 「비교 무효」."""
        c = C.compare(summary(seq_of(5), "opencv"), summary(seq_of(6), "ai"))
        md = C.to_markdown(c)

        self.assertFalse(c["same_conditions"])
        self.assertTrue(any(d.startswith("sequence_sha256") for d in c["condition_differences"]))
        self.assertIn("조건 불일치, 비교 무효", md)

    def test_condition_field_difference(self):
        """실행 조건(예: B 설정) 하나만 달라도 불일치."""
        seq = seq_of()
        cond = {"world_id": "w", "b_config_id": "b1"}
        c = C.compare(summary(seq, "opencv", conditions=cond),
                      summary(seq, "ai", conditions={**cond, "b_config_id": "b2"}))

        self.assertEqual(c["condition_differences"], ["conditions.b_config_id: 'b1' != 'b2'"])

    def test_same_method_rejected(self):
        """두 실행의 검출 방식이 같으면 비교 조건 위반."""
        seq = seq_of()
        c = C.compare(summary(seq, "opencv"), summary(seq, "opencv"))

        self.assertFalse(c["same_conditions"])
        self.assertTrue(any("detector.method" in d for d in c["condition_differences"]))

    def test_fake_banner_propagates(self):
        """둘 중 하나라도 가짜 입력이면 비교 MD 머리에 FAKE_INPUT."""
        seq = seq_of()
        a = summary(seq, "opencv")
        b = summary(seq, "ai")
        b["header"]["fake_input"] = False
        b["header"]["status"] = "executed"
        c = C.compare(a, b)

        self.assertTrue(c["fake_input"])
        self.assertTrue(C.to_markdown(c).startswith("> **FAKE_INPUT"))

    def test_no_adoption_verdict(self):
        """비교 결과에 채택 판단 필드가 없다."""
        seq = seq_of()
        c = C.compare(summary(seq, "opencv"), summary(seq, "ai"))

        self.assertFalse({"adopt", "winner", "verdict"} & set(c))


class CliTest(unittest.TestCase):
    """scripts/aggregate_sort_test.py fake → aggregate → compare."""

    def test_end_to_end(self):
        """가짜 기록 두 벌 취합 후 비교 rc 0, 같은 출력 폴더 재사용은 rc 2(덮어쓰기 금지)."""
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location("aggregate_sort_test", root / "scripts" / "aggregate_sort_test.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            seq = seq_of()
            (t / "seq.json").write_text(S.to_json(seq), encoding="utf-8")

            for method, outcomes in (("opencv", "2=b_skip"), ("ai", "3=isolated")):
                self.assertEqual(mod.main(["fake", "--sequence", str(t / "seq.json"), "--outcomes", outcomes,
                                           "--method", method, "--out-dir", str(t / f"in_{method}")]), 0)
                self.assertEqual(mod.main(["aggregate", "--sequence", str(t / "seq.json"),
                                           "--events", str(t / f"in_{method}" / "events.jsonl"),
                                           "--meta", str(t / f"in_{method}" / "run_meta.json"),
                                           "--out-dir", str(t / f"out_{method}")]), 0)

            self.assertEqual(mod.main(["compare", "--a", str(t / "out_opencv" / "summary.json"),
                                       "--b", str(t / "out_ai" / "summary.json"), "--out-dir", str(t / "cmp")]), 0)

            c = json.loads((t / "cmp" / "compare.json").read_text(encoding="utf-8"))
            self.assertTrue(c["same_conditions"] and c["fake_input"])

            self.assertEqual(mod.main(["compare", "--a", str(t / "out_opencv" / "summary.json"),
                                       "--b", str(t / "out_ai" / "summary.json"), "--out-dir", str(t / "cmp")]), 2)

    def test_rejected_events_rc1(self):
        """SIM_TIME 아닌 사건이 섞인 기록은 aggregate rc 1."""
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location("aggregate_sort_test", root / "scripts" / "aggregate_sort_test.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            seq = seq_of()
            (t / "seq.json").write_text(S.to_json(seq), encoding="utf-8")
            events = R.fake_events(seq)
            events[0]["timestamp_base"] = "PC_MONOTONIC"
            (t / "events.jsonl").write_text(R.events_to_jsonl(events), encoding="utf-8")
            (t / "meta.json").write_text(json.dumps(R.fake_run_meta(seq)), encoding="utf-8")

            self.assertEqual(mod.main(["aggregate", "--sequence", str(t / "seq.json"), "--events",
                                       str(t / "events.jsonl"), "--meta", str(t / "meta.json"),
                                       "--out-dir", str(t / "out")]), 1)
            self.assertFalse((t / "out").exists())


if __name__ == "__main__":
    unittest.main()
