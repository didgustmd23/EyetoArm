"""학습기 I3 — 평가 순수 로직 시험(torch·ultralytics 없이 가짜 모델).

근거: 학습기 설계 r2 §6(평가 내용·cold 시간 순서·eval.json·요약 열·기록 실패), 구현 지시 I3(evaluate 재사용, cold 문제 복사 금지,
정답은 평가 경로에만), A 상세설계 §7.1 「1단계 실험 기록」 표 열 순서.
"""
import json
import math
import os
import random
import tempfile
import unittest

import numpy as np

from eyetoarm.training.config import ConfigError
from eyetoarm.training.evaluation import (PHASES, SUMMARY_COLUMNS, EvalError, build_eval_record,
                                          eval_config_from_dict, fine_recall, load_gz_parts, percentile, run_accuracy,
                                          run_timing, summary_markdown, summary_row, validate_accuracy,
                                          validate_timing)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def eval_doc(**over):
    """시험 평가 설정 문서."""
    d = {"format_version": 1, "conf": 0.25, "iou": 0.5, "device": "0", "edges_px": [10, 12, 15, 25],
         "public": {"data_dir": "data/ds", "split": "test", "imgsz": [640]},
         "gazebo": {"data_dir": "data/gz", "split": "test", "imgsz": [640, 1280], "manifest": "data/gz/manifest.jsonl"},
         "timing": {"imgsz": [640, 1280], "warmup_n": 2, "measure_n": 3}, "val": {"batch": 8, "workers": 0}}
    d.update(over)
    return d


class FakeModel:
    """호출 이력을 남기는 가짜 모델. warm_in_init이면 생성자에서 predict(준비 중 호출)."""

    def __init__(self, hist, warm_in_init=False):
        """생성 기록."""
        self.hist, self.calls = hist, 0
        hist.append("construct")
        if warm_in_init:
            self.predict("init")

    def predict(self, img):
        """호출 기록."""
        self.calls += 1
        self.hist.append(("predict", img))


def timing(hist, warm_in_init=False, warmup_n=2, measure_n=3):
    """가짜 모델로 run_timing."""
    box = {}

    def prepare():
        """모델 생성."""
        box["m"] = FakeModel(hist, warm_in_init)
        return box["m"]

    t = [0]

    def clock():
        """1 ms씩 흐르는 시계."""
        t[0] += 1_000_000
        return t[0]

    out = run_timing(prepare=prepare, images=["i0", "i1", "i2"], predict=lambda m, img: m.predict(img),
                     sync=lambda: hist.append("sync"), peak_reset=lambda: hist.append("reset"),
                     peak_read=lambda: 4096, clock_ns=clock, warmup_n=warmup_n, measure_n=measure_n,
                     calls_seen=lambda: box["m"].calls)
    out["imgsz"] = 640
    return out


class EvalConfigTest(unittest.TestCase):
    """평가 설정 검증."""

    def test_repository_config_loads(self):
        """저장소의 s1_eval.json은 유효하고, 계획은 공개 test 640 → Gazebo 640·1280 순서(A §7.1 평가 조건)."""
        from eyetoarm.training.evaluation import load_eval_config
        c = load_eval_config(os.path.join(ROOT, "config", "training", "s1_eval.json"))
        self.assertEqual((c.conf, c.iou), (0.25, 0.5))
        self.assertEqual([p[0] for p in c.accuracy_plan()], ["test640", "gz640", "gz1280"])
        self.assertEqual((c.timing["warmup_n"], c.timing["measure_n"]), (10, 200))

    def test_rejects_bad_values_and_unknown_keys(self):
        """알 수 없는 키·conf 0·중복/32배수 아닌 크기·절대 경로·감소하는 구간은 ConfigError, 값이 바뀌면 eval_config_id가 바뀐다."""
        for bad in ({"extra": 1}, {"conf": 0}, {"timing": {"imgsz": [640, 640], "warmup_n": 1, "measure_n": 1}},
                    {"timing": {"imgsz": [650], "warmup_n": 1, "measure_n": 1}},
                    {"public": {"data_dir": "/abs", "split": "test", "imgsz": [640]}},
                    {"edges_px": [12, 10]}, {"timing": {"imgsz": [640], "warmup_n": 1, "measure_n": 0}}):
            with self.subTest(bad=bad):
                with self.assertRaises(ConfigError):
                    eval_config_from_dict(eval_doc(**bad))
        a, b = eval_config_from_dict(eval_doc()), eval_config_from_dict(eval_doc(conf=0.3))
        self.assertNotEqual(a.eval_config_id, b.eval_config_id)
        self.assertEqual(a.eval_config_id, eval_config_from_dict(eval_doc()).eval_config_id)


class PercentileTest(unittest.TestCase):
    """시간 통계."""

    def test_matches_numpy_linear(self):
        """선형 보간 백분위가 numpy 기본값과 같다(무작위 20 묶음)."""
        rng = random.Random(0)
        for _ in range(20):
            xs = [rng.uniform(0, 50) for _ in range(rng.randint(1, 300))]
            for q in (50, 95, 99):
                self.assertTrue(math.isclose(percentile(xs, q), float(np.percentile(xs, q)), rel_tol=1e-12))


class TimingOrderTest(unittest.TestCase):
    """cold 측정 순서(r1 F3): 준비 → cold → 워밍업 → 본 측정, 단계마다 최대 메모리 초기화."""

    def test_order_and_counts(self):
        """생성 뒤 첫 predict가 cold이고, 단계 앞마다 reset, 호출 수 1 + 2 + 3, 기록 검사 통과."""
        hist = []
        out = timing(hist)

        self.assertEqual(hist[0], "construct")
        self.assertEqual(hist[1], "reset")                             # cold 단계 시작
        self.assertEqual(hist[2], ("predict", "i0"))                   # 첫 predict = cold
        self.assertEqual([i for i, h in enumerate(hist) if h == "reset"], [1, 4, 9])   # predict 뒤마다 sync
        self.assertEqual(out["phase_order"], list(PHASES))
        self.assertEqual((out["cold"]["n"], out["warmup"]["n"], out["measure"]["n"], out["predict_calls"]),
                         (1, 2, 3, 6))
        self.assertEqual(out["predict_calls_before_prepare_end"], 0)
        self.assertEqual(validate_timing(out, imgsz=640, warmup_n=2, measure_n=3), [])
        self.assertTrue(all(out[p]["peak_mem_bytes"] == 4096 for p in ("cold", "warmup", "measure")))

    def test_predict_during_prepare_is_rejected(self):
        """준비(모델 생성) 중 predict가 있으면 cold가 cold가 아니므로 기록 검사가 거부한다."""
        out = timing([], warm_in_init=True)
        self.assertIn("predict called before cold call", validate_timing(out, imgsz=640, warmup_n=2, measure_n=3))

    def test_partial_or_wrong_records_are_rejected(self):
        """단계 누락·호출 수 부족·순서 변경·크기 불일치·계수기 없음·fresh 아님·비유한 값은 각각 문제로 보고된다."""
        good = timing([])
        cases = {"missing measure": lambda d: d.pop("measure"),
                 "short warmup": lambda d: d["warmup"]["call_ms"].pop(),
                 "order": lambda d: d.update(phase_order=["prepare", "warmup", "cold", "measure"]),
                 "imgsz": lambda d: d.update(imgsz=1280),
                 "no counter": lambda d: d.update(predict_calls_before_prepare_end=None),
                 "not fresh": lambda d: d.update(fresh_model=False),
                 "nan": lambda d: d["measure"]["sync_ms"].__setitem__(0, float("nan")),
                 "no peak key": lambda d: d["cold"].pop("peak_mem_bytes")}
        for name, mutate in cases.items():
            with self.subTest(case=name):
                d = json.loads(json.dumps(good))
                mutate(d)
                self.assertTrue(validate_timing(d, imgsz=640, warmup_n=2, measure_n=3))
        self.assertTrue(validate_timing("x", imgsz=640, warmup_n=2, measure_n=3))


def acc_entry(split, imgsz, per_gt, n=(1, 1)):
    """정확도 한 항목(evaluate() 출력 형식)."""
    return {"split": split, "images": len(per_gt), "imgsz": imgsz, "conf": 0.25, "iou": 0.5, "scale": 1.0,
            "edges_px": [10, 12, 15, 25],
            "by_size": {"bolt": {"<10": {"n": 1, "tp": 1, "recall": 1.0}}, "nut": {"<10": {"n": 0, "tp": 0,
                                                                                         "recall": None}}},
            "total": {"bolt": {"n": n[0], "tp": n[0]}, "nut": {"n": n[1], "tp": 0}},
            "predictions": {"bolt": {"n": 2, "tp": 1, "fp": 1, "precision": 0.5},
                            "nut": {"n": 0, "tp": 0, "fp": 0, "precision": None}},
            "per_gt": per_gt}


GZ_PER_GT = [{"image": "g0.png", "matched": [True, False], "short_px": [20.0, 21.0]},
             {"image": "g1.png", "matched": [True], "short_px": [18.0]}]
GZ_PARTS = {"g0.png": ["BOLT_A", "NUT"], "g1.png": ["BOLT_B"]}


class AccuracyTest(unittest.TestCase):
    """정확도 절차·기록 검사."""

    def plan(self):
        """시험 계획."""
        return eval_config_from_dict(eval_doc()).accuracy_plan()

    def make(self):
        """가짜 evaluate·val로 run_accuracy."""
        seen = []

        def ev(model, data_dir, split, imgsz):
            """평가 호출 기록."""
            seen.append(("evaluate", data_dir, imgsz))
            return acc_entry(split, imgsz, GZ_PER_GT if "gz" in data_dir else [{"image": "a.png", "matched": [True]}])

        def val(model, data_dir, split, imgsz, key):
            """val 호출 기록."""
            seen.append(("val", key))
            return {"map50_box": 0.5, "map50_mask": 0.4}

        return run_accuracy(model="m", plan=self.plan(), evaluate_fn=ev, val_fn=val), seen

    def test_plan_order_and_valid(self):
        """계획 순서(공개 test → gz640 → gz1280)대로 같은 모델에 evaluate·val, 기록 검사 통과."""
        out, seen = self.make()
        self.assertEqual(seen, [("evaluate", "data/ds", 640), ("val", "test640"), ("evaluate", "data/gz", 640),
                                ("val", "gz640"), ("evaluate", "data/gz", 1280), ("val", "gz1280")])
        self.assertEqual(validate_accuracy(out, self.plan()), [])

    def test_partial_records_rejected(self):
        """키 누락·per_gt 수 불일치·mAP 비유한·tp > n·계획 불일치·빈 predictions·fp 불일치·잘못된 by_size·per_gt 항목 형식은 거부."""
        good, _ = self.make()
        cases = {"missing gz1280": lambda d: d.pop("gz1280"), "per_gt": lambda d: d["gz640"]["per_gt"].pop(),
                 "map nan": lambda d: d["test640"].update(map50_box=float("nan")),
                 "tp>n": lambda d: d["gz640"]["total"]["bolt"].update(tp=9), "plan": lambda d: d["plan"].pop(),
                 "no map key": lambda d: d["gz640"].pop("map50_mask"),
                 "empty predictions": lambda d: d["gz640"].update(predictions={}),
                 "fp mismatch": lambda d: d["gz640"]["predictions"]["bolt"].update(fp=0),
                 "bad by_size": lambda d: d["test640"].update(by_size={"bolt": {"<10": {"n": "x"}}}),
                 "per_gt entry": lambda d: d["gz640"]["per_gt"].__setitem__(0, "x")}
        for name, mutate in cases.items():
            with self.subTest(case=name):
                d = json.loads(json.dumps(good))
                mutate(d)
                self.assertTrue(validate_accuracy(d, self.plan()))


class FineRecallTest(unittest.TestCase):
    """Gazebo 세부 클래스 대응(manifest labeled 순서 = 라벨 줄 순서)."""

    def test_mapping_and_errors(self):
        """정상 대응 BOLT_A 1/1·BOLT_B 1/1·NUT 0/1, 개수 불일치·영상 누락·모르는 클래스는 EvalError."""
        r = fine_recall(GZ_PER_GT, GZ_PARTS)
        self.assertEqual({c: (v["n"], v["tp"]) for c, v in r.items()},
                         {"BOLT_A": (1, 1), "BOLT_B": (1, 1), "NUT": (1, 0)})
        for parts in ({"g0.png": ["BOLT_A"], "g1.png": ["BOLT_B"]}, {"g0.png": ["BOLT_A", "NUT"]},
                      {"g0.png": ["BOLT_A", "SCREW"], "g1.png": ["BOLT_B"]}):
            with self.subTest(parts=parts):
                with self.assertRaises(EvalError):
                    fine_recall(GZ_PER_GT, parts)

    def test_manifest_reader(self):
        """manifest.jsonl → labeled 부품만 순서대로, 중복 sample_id는 EvalError."""
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.jsonl")
            rows = [{"sample_id": "g0", "parts": [{"cls": "BOLT_A", "labeled": True}, {"cls": "NUT", "labeled": False},
                                                  {"cls": "NUT", "labeled": True}]}]
            with open(p, "w") as f:
                f.write("".join(json.dumps(r) + "\n" for r in rows))
            self.assertEqual(load_gz_parts(p), {"g0.png": ["BOLT_A", "NUT"]})
            with open(p, "a") as f:
                f.write(json.dumps(rows[0]) + "\n")
            with self.assertRaises(EvalError):
                load_gz_parts(p)


class RecordSummaryTest(unittest.TestCase):
    """eval.json·요약."""

    def test_columns_match_a_design_table(self):
        """요약 열은 A 상세설계 §7.1 「1단계 실험 기록」 표 머리 줄과 글자 그대로 같은 순서다(추적 문서에서 직접 읽음)."""
        path = os.path.join(ROOT, "docs", "design", "A담당_비전_상세설계.md")
        with open(path, encoding="utf-8") as f:
            header = next(line for line in f if line.startswith("| 번호 | 모델·가중치"))
        self.assertEqual(tuple(c.strip() for c in header.strip().strip("|").split("|")), SUMMARY_COLUMNS)

    def test_record_and_summary_columns(self):
        """정답별 매칭은 eval_per_gt로 분리되고 Gazebo에 세부 재현율이 붙으며, 요약 열은 A §7.1 표 순서다."""
        cfg = eval_config_from_dict(eval_doc())
        acc, _ = AccuracyTest.make(AccuracyTest())
        rec, per_gt = build_eval_record(exp_id="S1-01", weights={"sha256": "x"}, eval_cfg=cfg,
                                        fingerprint={"ok": True}, timing={640: timing([]), 1280: timing([])},
                                        accuracy=acc, plan=cfg.accuracy_plan(), parts=GZ_PARTS, loads=[])

        self.assertNotIn("per_gt", rec["accuracy"]["gz640"])
        self.assertEqual(per_gt["per_gt"]["gz640"], GZ_PER_GT)
        self.assertEqual(rec["accuracy"]["gz640"]["fine"]["NUT"]["tp"], 0)
        self.assertNotIn("fine", rec["accuracy"]["test640"])
        self.assertEqual(rec["conditions"]["eval_config_id"], cfg.eval_config_id)
        self.assertEqual(sorted(rec["timing"]), ["1280", "640"])
        self.assertEqual(rec["timing"]["640"]["measure_n"], 3)

        row = summary_row(exp_id="S1-01", train={"family": "yolov8n-seg", "notes": "a|b"}, record=rec)
        self.assertEqual(tuple(row), SUMMARY_COLUMNS)
        md = summary_markdown([row])
        self.assertEqual(md.splitlines()[0], "| " + " | ".join(SUMMARY_COLUMNS) + " |")
        self.assertEqual(md.splitlines()[2].count("|"), len(SUMMARY_COLUMNS) + 1)   # 값 안의 |는 바꿔 칸이 밀리지 않음


if __name__ == "__main__":
    unittest.main()
