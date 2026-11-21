"""분류 시험 취합: 분모(순서표 전체)·볼트 정확도·혼동 행렬·정밀도/재현율·중복·재사용 창·사유·중단·
WRONG_BIN 관계, 어댑터 거부(SIM_TIME 전용·비유한·혼합). 기대값은 손계산이며 취합 코드로 역산하지 않는다.
"""
import json
import unittest

from eyetoarm.sorteval import aggregate as A
from eyetoarm.sorteval import records as R
from eyetoarm.sorteval import sequence as S


def make_seq(spec_list, include_nut=False):
    """정해진 규격 순서의 순서표(사람 작성 형식)."""
    rows = [{"supply_index": i, "part_spec": s, "scenario": "nut_distractor" if s == "NUT" else "normal",
             "interval_s": None, "start_pose": None} for i, s in enumerate(spec_list, start=1)]
    seq = {"format": S.FORMAT, "format_version": S.FORMAT_VERSION, "sequence_id": "t", "status": "planned",
           "meta": {"generator": None, "created_at_utc": "2026-11-20T00:00:00Z", "python": None, "author": None,
                    "confirmer": None, "notes": ""},
           "timing": {"basis": "SIM_TIME", "mode": "interval", "interval_s": None},
           "pose_ranges": {"lateral_offset_mm": None, "yaw_deg": None},
           "include_nut": include_nut, "supplies": rows}
    seq["content_sha256"] = S.content_sha256(seq)
    assert S.validate_sequence(seq) == []

    return seq


def run(seq, outcomes=None, events=None):
    """가짜 결과(또는 주어진 원본 사건) → JSONL 왕복 → 취합."""
    raws = events if events is not None else R.fake_events(seq, outcomes or {})
    records, errors = R.parse_events_jsonl(R.events_to_jsonl(raws))
    assert errors == [], errors
    meta, meta_errors = R.adapt_run_meta(R.fake_run_meta(seq))
    assert meta_errors == [], meta_errors

    return A.aggregate(seq, records, meta)


def row_of(summary, index):
    """공급 행."""
    return summary["per_supply"][index - 1]


class DenominatorTest(unittest.TestCase):
    """분모와 최종 판정."""

    def test_denominator_is_whole_sequence(self):
        """격리·미처리·사용자 중단·공급 누락도 분모에 남고, 판정 합 = 순서표 공급 수."""
        seq = make_seq(["BOLT_A", "BOLT_B", "BOLT_A", "BOLT_B", "BOLT_A", "BOLT_B"])
        s = run(seq, {2: "isolated", 3: "no_supply", 5: "user_abort"})

        self.assertEqual(s["denominator"]["total_supplies"], 6)
        self.assertEqual(sum(s["outcomes"].values()), 6)
        self.assertEqual(s["outcomes"]["CORRECT"], 2)
        self.assertEqual(s["outcomes"]["ISOLATED"], 1)
        self.assertEqual(s["outcomes"]["SUPPLY_MISSING"], 1)
        self.assertEqual(s["outcomes"]["USER_ABORTED"], 2)
        self.assertAlmostEqual(s["accuracy"], 2 / 6)

    def test_bolt_accuracy_always_present(self):
        """볼트 분류 정확도(분모 = 볼트 공급 수)는 너트 옵션과 관계없이 나온다."""
        s_off = run(make_seq(["BOLT_A", "BOLT_B"]), {2: "wrong_bin"})

        self.assertAlmostEqual(s_off["bolt_accuracy"], 0.5)
        self.assertIsNone(s_off["nut_arrival"])

        seq = make_seq(["BOLT_A", "NUT", "BOLT_B", "NUT"], include_nut=True)
        s_on = run(seq, {3: "wrong_bin", 4: "wrong_bin"})

        self.assertEqual(s_on["denominator"], {"total_supplies": 4, "bolt_supplies": 2, "nut_supplies": 2})
        self.assertAlmostEqual(s_on["accuracy"], 2 / 4)
        self.assertAlmostEqual(s_on["bolt_accuracy"], 1 / 2)
        self.assertEqual(s_on["nut_arrival"]["arrived_nut_bin"], 1)
        self.assertAlmostEqual(s_on["nut_arrival"]["ratio"], 0.5)
        self.assertIn("NUT", s_on["confusion"])

    def test_meta_must_match_sequence(self):
        """실행 메타의 순서표 해시가 다르면 취합을 거부한다."""
        seq = make_seq(["BOLT_A"])
        records, _ = R.parse_events_jsonl(R.events_to_jsonl(R.fake_events(seq)))
        meta_raw = R.fake_run_meta(seq)
        meta_raw["sequence_sha256"] = "0" * 64
        meta, _ = R.adapt_run_meta(meta_raw)

        with self.assertRaises(ValueError):
            A.aggregate(seq, records, meta)


class ConfusionTest(unittest.TestCase):
    """혼동 행렬·정밀도·재현율·너트 상자 유입."""

    def setUp(self):
        """A→A 정답, B→A 오분류, A→너트 상자, B→격리."""
        self.s = run(make_seq(["BOLT_A", "BOLT_B", "BOLT_A", "BOLT_B"]),
                     {2: "wrong_bin", 3: "bolt_to_nut_bin", 4: "isolated"})

    def test_matrix(self):
        """손계산 행렬."""
        c = self.s["confusion"]

        self.assertEqual(c["BOLT_A"], {"BOLT_A": 1, "BOLT_B": 0, "NUT": 1, "ISOLATION": 0, "NONE": 0})
        self.assertEqual(c["BOLT_B"], {"BOLT_A": 1, "BOLT_B": 0, "NUT": 0, "ISOLATION": 1, "NONE": 0})

    def test_precision_recall_with_zero_denominator(self):
        """A 정밀도 1/2·재현율 1/2, B 상자 도착 0 → 정밀도 None(0 아님), 재현율 0/2."""
        pr = self.s["precision_recall"]

        self.assertAlmostEqual(pr["BOLT_A"]["precision"], 0.5)
        self.assertAlmostEqual(pr["BOLT_A"]["recall"], 0.5)
        self.assertIsNone(pr["BOLT_B"]["precision"])
        self.assertEqual(pr["BOLT_B"]["precision_denominator"], 0)
        self.assertEqual(pr["BOLT_B"]["recall"], 0.0)
        self.assertIn("| BOLT_B | - | 0 |", A.to_markdown(self.s))

    def test_bolt_into_nut_bin_is_also_wrong_bin(self):
        """너트 상자에 들어간 볼트는 WRONG_BIN이자 너트 상자 유입 열에 함께 센다(보완 1)."""
        self.assertEqual(self.s["bolts_into_nut_bin"], 1)
        self.assertEqual(row_of(self.s, 3)["outcome"], "WRONG_BIN")
        self.assertEqual(self.s["outcomes"]["WRONG_BIN"], 2)
        self.assertIn("WRONG_BIN ⊇ bolts_into_nut_bin", self.s["notes"]["bolts_into_nut_bin"])

    def test_bin_placement(self):
        """집기 성공 2건(1·2번) 중 목표 상자 도착 1건(2번은 다른 상자)."""
        bp = self.s["bin_placement"]

        self.assertEqual((bp["pick_succeeded"], bp["placed_in_target_bin"]), (2, 1))


class DuplicateAndWindowTest(unittest.TestCase):
    """중복 도착과 풀 부품 재사용 창."""

    def test_duplicate_arrival_counted_once(self):
        """같은 부품·같은 트랙 두 번 도착 → 판정 1회(CORRECT), 중복 열 각 1."""
        s = run(make_seq(["BOLT_A", "BOLT_B"]), {1: "dup_arrival"})

        self.assertEqual(row_of(s, 1)["outcome"], "CORRECT")
        self.assertEqual(s["duplicates"], {"same_part": 1, "same_track": 1})
        self.assertEqual(sum(s["confusion"]["BOLT_A"].values()), 1)

    def test_pool_part_reuse_window(self):
        """1번과 4번이 같은 풀 부품(part_00): 4번 공급 뒤 도착은 1번에 붙지 않는다."""
        s = run(make_seq(["BOLT_A", "BOLT_B", "BOLT_A", "BOLT_B"]), {1: "not_arrived"})

        self.assertEqual(row_of(s, 1)["pool_part_id"], row_of(s, 4)["pool_part_id"])
        self.assertEqual(row_of(s, 1)["outcome"], "NOT_ARRIVED")
        self.assertEqual(row_of(s, 4)["outcome"], "CORRECT")
        self.assertEqual(s["unlinked"]["arrivals"], 0)

    def test_missed_counts_undetected(self):
        """검출(트랙 연결) 없이 통과한 공급 수."""
        s = run(make_seq(["BOLT_A", "BOLT_B", "BOLT_A"]), {2: "undetected"})

        self.assertEqual(s["missed"], 1)


class FailureSubjectTest(unittest.TestCase):
    """판정 주체별 첫 실패 사유."""

    def test_subjects(self):
        """A 검출 → B 판정 → C 실행 순으로 처음 끊긴 주체와 사유."""
        cases = {1: ("undetected", "A", "NOT_DETECTED"),
                 2: ("a_invalid", "A", "TOP_FACE_UNRESOLVED"),
                 3: ("b_unresolved", "B", "UNRESOLVED"),
                 4: ("b_skip", "B", "WORK_COORD_INVALID"),
                 5: ("b_wrong_class", "B", "CLASS_MISMATCH"),
                 6: ("c_failed", "C", "ATTACH_FAILED"),
                 7: ("supply_mismatch", "C", "SUPPLY_MISMATCH"),
                 8: ("isolated", "C", "REDETECT_UNSTABLE")}
        seq = make_seq(["BOLT_A", "BOLT_B"] * 5)
        s = run(seq, {i: c[0] for i, c in cases.items()})

        for i, (_, subject, reason) in cases.items():
            with self.subTest(i=i):
                r = row_of(s, i)

                self.assertEqual((r["failure_subject"], r["failure_reason"]), (subject, reason))

        self.assertIsNone(row_of(s, 9)["failure_subject"])
        self.assertEqual(sum(sum(v.values()) for v in s["failures_by_subject"].values()),
                         10 - s["outcomes"]["CORRECT"])

    def test_user_abort_only_in_flight(self):
        """사용자 정지 전 이미 끝난 미도착은 NOT_ARRIVED, 정지 뒤 공급 안 된 행만 USER_ABORTED."""
        s = run(make_seq(["BOLT_A", "BOLT_B", "BOLT_A", "BOLT_B"]), {1: "not_arrived", 3: "user_abort"})

        self.assertEqual(row_of(s, 1)["outcome"], "NOT_ARRIVED")
        self.assertEqual([row_of(s, i)["outcome"] for i in (3, 4)], ["USER_ABORTED", "USER_ABORTED"])
        self.assertEqual(row_of(s, 3)["failure_subject"], "USER")

    def test_supply_missing_in_middle_is_not_user_abort(self):
        """사용자 정지가 있어도 뒤 행이 공급됐으면 중간 누락은 SUPPLY_MISSING."""
        seq = make_seq(["BOLT_A", "BOLT_B", "BOLT_A"])
        events = R.fake_events(seq, {2: "no_supply"})
        events.insert(0, {"event": "run_event", "kind": "USER_STOP", "stamp_ns": 99_000_000_000,
                          "timestamp_base": "SIM_TIME"})
        s = run(seq, events=events)

        self.assertEqual(row_of(s, 2)["outcome"], "SUPPLY_MISSING")


class TimingTest(unittest.TestCase):
    """처리 시간 구간."""

    def test_intervals(self):
        """가짜 시각: 공급→첫 관측 500 ms, 관측→B goal 100 ms, 정지→재가동 5000 ms, 관측→도착 4500 ms."""
        s = run(make_seq(["BOLT_A", "BOLT_B"]))
        t = s["timings"]

        self.assertEqual((t["supply_to_first_seen"]["n"], t["supply_to_first_seen"]["p50_ms"]), (2, 500.0))
        self.assertEqual(t["first_seen_to_b_goal"]["p95_ms"], 100.0)
        self.assertEqual(t["belt_stop_to_restart"]["max_ms"], 5000.0)
        self.assertEqual(t["first_seen_to_arrival"]["p50_ms"], 4500.0)

    def test_negative_interval_flagged(self):
        """도착이 첫 관측보다 이르면 음수로 세고 통계에서 뺀다."""
        seq = make_seq(["BOLT_A"])
        events = R.fake_events(seq)

        for e in events:
            if e["event"] == "a_track":
                e["first_seen_ns"] += 10_000_000_000

        t = run(seq, events=events)["timings"]

        self.assertEqual(t["first_seen_to_arrival"]["negative"], 1)
        self.assertEqual(t["first_seen_to_arrival"]["n"], 0)


class AdapterRejectTest(unittest.TestCase):
    """어댑터 경계: SIM_TIME 전용·비유한·형식 위반 거부(보완 4)."""

    def base_line(self):
        """정상 공급 사건 한 줄."""
        return {"event": "supply", "supply_index": 1, "pool_part_id": "part_00", "supplied_spec": "BOLT_A",
                "stamp_ns": 5, "timestamp_base": "SIM_TIME"}

    def rejects(self, text):
        """거부 사유가 하나 이상."""
        _, errors = R.parse_events_jsonl(text)

        self.assertTrue(errors, text)

        return errors

    def test_only_sim_time(self):
        """SIM_TIME 이외 기준·기준 누락은 거부."""
        for base in ("PC_MONOTONIC", "ROS_TIME", None):
            line = self.base_line()
            line["timestamp_base"] = base

            with self.subTest(base=base):
                self.rejects(json.dumps(line))

    def test_mixed_bases_rejected(self):
        """한 기록 안에 SIM_TIME과 다른 기준이 섞이면 그 줄이 거부된다(취합 불가)."""
        good = self.base_line()
        bad = {"event": "arrival", "pool_part_id": "part_00", "bin": "BOLT_A", "stamp_ns": 9,
               "timestamp_base": "PC_MONOTONIC"}
        errors = self.rejects(json.dumps(good) + "\n" + json.dumps(bad))

        self.assertTrue(any("event 2" in e for e in errors))

    def test_non_finite_and_bad_stamps(self):
        """1e400·NaN·실수·bool·음수 시각, 거대 정수는 거부."""
        raw = json.dumps(self.base_line())

        for bad in ('1e400', 'NaN', '1.5', 'true', '-1', '1' + '0' * 30):
            with self.subTest(bad=bad):
                self.rejects(raw.replace('"stamp_ns": 5', f'"stamp_ns": {bad}'))

    def test_field_rules(self):
        """정의되지 않은 필드·모르는 사건·중복 공급 번호·이유 없는 SKIP은 거부."""
        line = self.base_line()
        line["extra"] = 1
        self.rejects(json.dumps(line))
        self.rejects(json.dumps({"event": "teleport", "timestamp_base": "SIM_TIME"}))
        self.rejects(json.dumps(self.base_line()) + "\n" + json.dumps(self.base_line()))
        self.rejects(json.dumps({"event": "b_decision", "part_track_id": "t", "part_class": "BOLT_A",
                                 "action": "SKIP", "skip_reason": None, "stamp_ns": 1, "timestamp_base": "SIM_TIME"}))

    def test_run_meta_fake_flag_consistency(self):
        """status fake_input ⇔ fake_input true. 어긋나면 거부."""
        seq = make_seq(["BOLT_A"])
        meta = R.fake_run_meta(seq)
        meta["status"] = "executed"

        self.assertIsNone(R.adapt_run_meta(meta)[0])

    def test_fake_outcome_names_checked(self):
        """모르는 가짜 결과 이름은 ValueError."""
        with self.assertRaises(ValueError):
            R.fake_events(make_seq(["BOLT_A"]), {1: "teleported"})


class OutputTest(unittest.TestCase):
    """MD·CSV 출력."""

    def test_fake_banner_and_csv(self):
        """가짜 입력 결과의 MD 머리에 FAKE_INPUT, CSV는 머리 + 공급 행 수."""
        s = run(make_seq(["BOLT_A", "BOLT_B", "BOLT_A"]))
        md = A.to_markdown(s)
        csv_lines = A.per_supply_csv(s).strip().splitlines()

        self.assertTrue(md.startswith("> **FAKE_INPUT"))
        self.assertEqual(len(csv_lines), 4)
        self.assertEqual(csv_lines[0].split(",")[:3], ["supply_index", "part_spec", "expected_bin"])
        json.dumps(s, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
