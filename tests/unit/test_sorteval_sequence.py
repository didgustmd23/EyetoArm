"""분류 시험 공급 순서표 시험: 고정 시드 재현성·패턴·너트 옵션, 검증기 거부, JSON/CSV 왕복, 덮어쓰기 금지.

분류 대상은 볼트만이다.
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from eyetoarm.sorteval import sequence as S


def gen(**kw):
    """생성 시각을 고정한 순서표."""
    kw.setdefault("seed", 42)
    kw.setdefault("count", 12)
    kw.setdefault("created_at_utc", "2026-11-20T00:00:00Z")

    return S.generate_sequence(**kw)


def specs(seq):
    """규격 열."""
    return [r["part_spec"] for r in seq["supplies"]]


def rehash(seq):
    """수정 뒤 해시를 다시 맞춘다(해시 외 사유만 보기 위해)."""
    seq["content_sha256"] = S.content_sha256(seq)

    return seq


class GeneratorTest(unittest.TestCase):
    """고정 시드 생성기."""

    def test_same_seed_same_content(self):
        """같은 seed·인자는 같은 공급 행과 해시를 낸다(생성 시각만 달라도 해시 같음)."""
        a = gen(pattern="mixed")
        b = gen(pattern="mixed", created_at_utc="2030-01-01T00:00:00Z")

        self.assertEqual(a["supplies"], b["supplies"])
        self.assertEqual(a["content_sha256"], b["content_sha256"])

    def test_different_seed_differs(self):
        """다른 seed는 다른 순서를 낸다."""
        self.assertNotEqual(specs(gen(seed=1, count=20)), specs(gen(seed=2, count=20)))

    def test_meta_records_seed_args(self):
        """메타에 시드·생성 인자·생성 시각·파이썬 버전이 남는다."""
        seq = gen(seed=7, count=10, pattern="blocks", block_len=2)
        g = seq["meta"]["generator"]

        self.assertEqual(g["seed"], 7)
        self.assertEqual(g["args"]["pattern"], "blocks")
        self.assertEqual(g["args"]["block_len"], 2)
        self.assertEqual(seq["meta"]["created_at_utc"], "2026-11-20T00:00:00Z")
        self.assertTrue(seq["meta"]["python"])

    def test_ratio_counts(self):
        """비율 1:2, 9개 → BOLT_A 3·BOLT_B 6(최대 나머지 배분)."""
        seq = gen(count=9, ratio={"BOLT_A": 1, "BOLT_B": 2})

        self.assertEqual(specs(seq).count("BOLT_A"), 3)
        self.assertEqual(specs(seq).count("BOLT_B"), 6)

    def test_default_excludes_nut(self):
        """기본(include_nut=False)은 너트 행 0, NUT 비율은 거부."""
        for pattern in S.PATTERNS:
            self.assertNotIn("NUT", specs(gen(count=12, pattern=pattern)))

        with self.assertRaises(ValueError):
            gen(ratio={"BOLT_A": 1, "NUT": 1})

    def test_include_nut_option(self):
        """옵션을 켜면 너트 행이 들어가고 시나리오는 nut_distractor, 기대 상자는 NUT."""
        seq = gen(count=9, include_nut=True)
        nuts = [r for r in seq["supplies"] if r["part_spec"] == "NUT"]

        self.assertEqual(len(nuts), 3)
        self.assertTrue(all(r["scenario"] == "nut_distractor" for r in nuts))
        self.assertEqual(S.expected_bin("NUT"), "NUT")

    def test_alternate_pattern(self):
        """alternate는 같은 수일 때 이웃이 모두 다르다."""
        s = specs(gen(count=10, pattern="alternate"))

        self.assertTrue(all(a != b for a, b in zip(s, s[1:])))

    def test_blocks_pattern(self):
        """blocks는 block_len 길이의 같은 규격 묶음으로 이루어진다."""
        s = specs(gen(count=12, pattern="blocks", block_len=3))

        self.assertTrue(all(len(set(s[i:i + 3])) == 1 for i in range(0, 12, 3)))

    def test_mixed_guarantees_segments(self):
        """mixed는 같은 규격 연속 구간과 교대 구간을 하나씩 보장한다."""
        seq = gen(count=14, pattern="mixed", block_len=3)
        scen = [r["scenario"] for r in seq["supplies"]]

        self.assertEqual(scen.count("consecutive_same"), 3)
        self.assertEqual(scen.count("alternate"), 4)

    def test_bad_args(self):
        """잘못된 인자(bool 시드, 0개, 모르는 패턴, block_len 1, mixed 부족)는 ValueError."""
        for kw in ({"seed": True}, {"count": 0}, {"pattern": "zigzag"}, {"block_len": 1},
                   {"count": 5, "pattern": "mixed"}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                gen(**kw)


class ValidatorTest(unittest.TestCase):
    """순서표 검증기 거부 사례."""

    def assertRejects(self, seq, needle):
        """검증 오류 중 needle을 포함하는 것이 있어야 한다."""
        errors = S.validate_sequence(seq)

        self.assertTrue(any(needle in e for e in errors), errors)

    def test_generated_passes(self):
        """생성 결과는 통과한다."""
        self.assertEqual(S.validate_sequence(gen()), [])

    def test_index_rules(self):
        """번호 비연속·중복·bool은 거부."""
        for value, needle in ((3, "contiguous"), (1, "contiguous"), (True, "bool")):
            seq = gen()
            seq["supplies"][1]["supply_index"] = value

            with self.subTest(value=value):
                self.assertRejects(rehash(seq), needle)

    def test_unknown_spec_and_nut_without_option(self):
        """모르는 규격, include_nut=false인데 NUT 행은 거부."""
        seq = gen()
        seq["supplies"][0]["part_spec"] = "BOLT_C"
        self.assertRejects(rehash(seq), "unknown part_spec")

        seq = gen()
        seq["supplies"][0].update(part_spec="NUT", scenario="nut_distractor")
        self.assertRejects(rehash(seq), "include_nut is false")

    def test_scenario_rules(self):
        """모르는 시나리오, 볼트 행의 nut_distractor는 거부."""
        seq = gen()
        seq["supplies"][0]["scenario"] = "weird"
        self.assertRejects(rehash(seq), "unknown scenario")

        seq = gen()
        seq["supplies"][0]["scenario"] = "nut_distractor"
        self.assertRejects(rehash(seq), "exactly for NUT")

    def test_interval_rules(self):
        """간격 0·음수·bool은 거부, null·양수는 허용."""
        for value in (0, -1.0, True):
            seq = gen()
            seq["supplies"][0]["interval_s"] = value

            with self.subTest(value=value):
                self.assertRejects(rehash(seq), "interval_s")

        seq = gen()
        seq["supplies"][0]["interval_s"] = 2.5
        self.assertEqual(S.validate_sequence(rehash(seq)), [])

    def test_start_pose_range(self):
        """범위 미선언 값·범위 밖 값은 거부, 범위 안은 통과."""
        seq = gen()
        seq["supplies"][0]["start_pose"] = {"lateral_offset_mm": 5.0, "yaw_deg": None}
        self.assertRejects(rehash(seq), "not declared")

        seq["pose_ranges"]["lateral_offset_mm"] = [-10.0, 10.0]
        self.assertEqual(S.validate_sequence(rehash(seq)), [])

        seq["supplies"][0]["start_pose"]["lateral_offset_mm"] = 11.0
        self.assertRejects(rehash(seq), "outside declared range")

    def test_bad_range_declaration(self):
        """lo > hi 범위 선언은 거부."""
        seq = gen()
        seq["pose_ranges"]["yaw_deg"] = [30, -30]

        self.assertRejects(rehash(seq), "pose_ranges.yaw_deg")

    def test_fields(self):
        """정의되지 않은 필드·누락 필드는 거부."""
        seq = gen()
        seq["supplies"][0]["expected_bin"] = "BOLT_A"
        self.assertRejects(seq, "undefined field")

        seq = gen()
        del seq["timing"]
        self.assertRejects(seq, "missing field")

    def test_timing_basis_and_version(self):
        """SIM_TIME 이외 기준, 모르는 format_version은 거부."""
        seq = gen()
        seq["timing"]["basis"] = "PC_MONOTONIC"
        self.assertRejects(rehash(seq), "SIM_TIME")

        seq = gen()
        seq["format_version"] = 2
        self.assertRejects(seq, "format_version")

    def test_empty_supplies(self):
        """빈 공급 목록은 거부."""
        seq = gen()
        seq["supplies"] = []

        self.assertRejects(seq, "non-empty list")

    def test_hash_mismatch(self):
        """내용이 바뀌었는데 해시가 그대로면 거부."""
        seq = gen()
        seq["supplies"][0]["part_spec"] = "BOLT_B" if seq["supplies"][0]["part_spec"] == "BOLT_A" else "BOLT_A"

        self.assertRejects(seq, "content_sha256")

    def test_strict_json(self):
        """중복 키·NaN·1e400·거대 정수는 해석 단계에서 거부."""
        text = S.to_json(gen())

        for bad in (text.replace('"status": "planned"', '"status": "planned", "status": "planned"'),
                    text.replace('"interval_s": null', '"interval_s": NaN', 1),
                    text.replace('"interval_s": null', '"interval_s": 1e400', 1),
                    text.replace('"seed": 42', '"seed": 1' + "0" * 30)):
            seq, errors = S.load_sequence_text(bad)

            with self.subTest(bad=bad[:0]):
                self.assertIsNone(seq)
                self.assertTrue(errors)

    def test_collects_all_errors(self):
        """첫 오류에서 멈추지 않고 모두 모은다."""
        seq = gen()
        seq["supplies"][0]["part_spec"] = "X"
        seq["supplies"][1]["scenario"] = "Y"

        self.assertGreaterEqual(len(S.validate_sequence(seq)), 2)


class RoundTripTest(unittest.TestCase):
    """JSON·CSV 왕복과 출력."""

    def test_json_roundtrip(self):
        """정본 JSON을 다시 읽으면 같은 순서표·통과."""
        seq = gen(pattern="mixed")
        back, errors = S.load_sequence_text(S.to_json(seq))

        self.assertEqual(errors, [])
        self.assertEqual(back, seq)

    def test_csv_roundtrip_same_supplies(self):
        """CSV 사본으로 다시 만든 순서표는 같은 공급 행·같은 해시."""
        seq = gen(pattern="blocks")
        back, errors = S.sequence_from_csv(S.to_csv(seq), sequence_id="x")

        self.assertEqual(errors, [])
        self.assertEqual(back["supplies"], seq["supplies"])
        self.assertEqual(back["content_sha256"], seq["content_sha256"])
        self.assertIsNone(back["meta"]["generator"])

    def test_human_csv_rejections(self):
        """사람이 쓴 CSV: 머리 다름·숫자 아님·번호 건너뜀·너트(옵션 꺼짐)는 거부."""
        head = "supply_index,part_spec,scenario,interval_s,lateral_offset_mm,yaw_deg\n"
        cases = ("idx,spec\n1,BOLT_A\n",
                 head + "1,BOLT_A,normal,abc,,\n",
                 head + "1,BOLT_A,normal,,,\n3,BOLT_B,normal,,,\n",
                 head + "1,NUT,nut_distractor,,,\n")

        for text in cases:
            with self.subTest(text=text[-30:]):
                seq, errors = S.sequence_from_csv(text, sequence_id="h")

                self.assertTrue(errors)

    def test_write_outputs_no_overwrite(self):
        """json·csv·md를 쓰고, 같은 이름이 있으면 덮어쓰지 않으며, 위험한 이름은 거부한다."""
        seq = gen()

        with tempfile.TemporaryDirectory() as tmp:
            paths = S.write_outputs(seq, tmp)

            self.assertEqual(json.loads(paths["json"].read_text(encoding="utf-8")), seq)
            self.assertIn("독립 정답", paths["md"].read_text(encoding="utf-8"))

            with self.assertRaises(FileExistsError):
                S.write_outputs(seq, tmp)

            with self.assertRaises(ValueError):
                S.write_outputs(seq, tmp, stem="../escape")

    def test_generator_does_not_mutate_ratio(self):
        """생성기는 넘겨받은 비율 사전을 바꾸지 않는다."""
        ratio = {"BOLT_A": 1, "BOLT_B": 1}
        before = copy.deepcopy(ratio)
        gen(ratio=ratio)

        self.assertEqual(ratio, before)


class CliTest(unittest.TestCase):
    """scripts/sort_test_sequence.py 진입점."""

    def test_generate_and_validate(self):
        """generate 후 validate rc 0, 손상 파일은 rc 1."""
        import importlib.util

        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location("sort_test_sequence", root / "scripts" / "sort_test_sequence.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(mod.main(["generate", "--seed", "3", "--count", "6", "--out-dir", tmp,
                                       "--sequence-id", "cli"]), 0)

            path = Path(tmp) / "cli.json"
            self.assertEqual(mod.main(["validate", str(path)]), 0)

            path.write_text(path.read_text(encoding="utf-8").replace('"part_spec": "BOLT_A"', '"part_spec": "BOLT_Z"', 1), encoding="utf-8")
            self.assertEqual(mod.main(["validate", str(path)]), 1)


if __name__ == "__main__":
    unittest.main()
