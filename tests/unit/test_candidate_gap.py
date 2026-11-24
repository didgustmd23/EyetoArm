"""목록 후보 공백 기록(#401) 시험: 검출은 있는데 목록 후보 0이 오래 이어질 때의 줄·풀림·요약.

판정·송신은 바꾸지 않는다. 실제 AI·ROS 없이 `AiInstance`와 같은 속성을 가진 가짜 인스턴스를 쓴다.
"""
from __future__ import annotations

import unittest
import unittest.mock
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from eyetoarm.vision.candidate_gap import (CLEARED_EVENT, DEFAULT_CANDIDATE_GAP, GAP_EVENT, KIND_CANDIDATES,
                                           KIND_EMPTY, KIND_GAP, NO_LIST_CANDIDATES, CandidateGapConfig,
                                           CandidateGapConfigError, CandidateGapWatch, classify_instances,
                                           parse_candidate_gap)
from eyetoarm.vision.live_sort_part import SortPartConfigError, parse_sort_part
from test_live_sort_part import SenderBase, cfg
from test_live_world_tracking import WorldCli

S = 1_000_000_000   # 1 s(ns)


def base_doc() -> dict:
    """설정 블록의 필수 네 키만(요약 형식 `retry: off`는 설정 입력이 아님)."""
    return {k: v for k, v in cfg().to_json().items() if k != "retry"}


def bolt(axis_reason=None, *, clipped=False, valid=False):
    """축 경로가 다룬 범위 안 볼트(유효면 끝점, 아니면 사유)."""
    return SimpleNamespace(in_scope=True, axis_evaluated=True, rejected=None, clipped=clipped,
                           axis_overlap_with=None, axis_reason=None if valid else axis_reason,
                           axis_endpoints_px=object() if valid else None)


def rejected(code, in_scope=False):
    """축 경로가 다루지 않은 거부 인스턴스(팔 영역·벨트 밖·너트 등)."""
    return SimpleNamespace(in_scope=in_scope, axis_evaluated=False, rejected=code, clipped=False,
                           axis_overlap_with=None, axis_reason=None, axis_endpoints_px=None)


AMB = [bolt("AXIS_HEAD_AMBIGUOUS")]


class ClassifyTest(unittest.TestCase):
    """사진 하나의 종류 판정."""

    def test_candidates_when_no_skip(self):
        """실행기 건너뜀 사유가 없으면 후보 있는 사진."""
        self.assertEqual(classify_instances(AMB, None)[0], KIND_CANDIDATES)

    def test_other_skip_not_judged(self):
        """목록 후보 판정 전 건너뜀(세대·보정 등)은 대상이 아니다."""
        for reason in ("generation_discarded", "calibration_mismatch", "rx_epoch_stale", "duplicate"):
            with self.subTest(reason=reason):
                self.assertIsNone(classify_instances(AMB, reason))

    def test_gap_counts_axis_reasons(self):
        """범위 안 볼트의 축 제외 사유를 세고, 범위 밖 거부는 따로 센다. 너트·점수 미달은 세지 않는다."""
        kind, reasons, excluded = classify_instances(
            [bolt("AXIS_HEAD_AMBIGUOUS"), bolt(clipped=True), rejected("ARM_REGION"), rejected("OUTSIDE_BELT"),
             rejected("NON_TARGET_CLASS"), rejected("LOW_CONFIDENCE")], NO_LIST_CANDIDATES)

        self.assertEqual(kind, KIND_GAP)
        self.assertEqual(reasons, Counter({"AXIS_HEAD_AMBIGUOUS": 1, "CONTOUR_CLIPPED": 1}))
        self.assertEqual(excluded, Counter({"ARM_REGION": 1, "OUTSIDE_BELT": 1}))

    def test_empty_when_only_out_of_scope(self):
        """범위 안 볼트가 없으면(벨트 밖 볼트·너트만) 공백이 아닌 빈 사진."""
        kind, reasons, _ = classify_instances([rejected("OUTSIDE_BELT"), rejected("NON_TARGET_CLASS")],
                                              NO_LIST_CANDIDATES)
        self.assertEqual((kind, reasons), (KIND_EMPTY, Counter()))


class WatchTest(unittest.TestCase):
    """공백 구간의 알림·반복·풀림·요약."""

    def setUp(self):
        """10 s 알림·30 s 반복 감시와 기록 목록."""
        self.lines = []
        self.watch = CandidateGapWatch(CandidateGapConfig(alert_after_s=10.0, repeat_s=30.0, clear_after_s=2.0),
                                       lambda e, d: self.lines.append((e, d)), track_ids=lambda: [3])

    def feed(self, t0_s, t1_s, kind=KIND_GAP, reasons=None, step_s=0.5):
        """[t0, t1] 시뮬 s를 `step_s` 간격 사진으로 넣는다."""
        reasons = Counter({"AXIS_HEAD_AMBIGUOUS": 1}) if reasons is None and kind == KIND_GAP else (reasons or Counter())
        t = t0_s

        while t <= t1_s + 1e-9:
            self.watch.observe(int(round(t * S)), kind, reasons, Counter())
            t += step_s

    def events(self):
        """기록 사건 이름."""
        return [e for e, _ in self.lines]

    def test_short_gap_silent(self):
        """알림 시간보다 짧은 공백은 줄을 내지 않는다(풀림도 없음)."""
        self.feed(100, 109)
        self.feed(109.5, 110, KIND_CANDIDATES)

        self.assertEqual(self.lines, [])
        self.assertEqual(self.watch.summary()["gaps_short"], 1)

    def test_alert_repeat_and_clear(self):
        """10 s에 첫 줄, 30 s마다 반복, 후보가 다시 나오면 풀림 한 줄."""
        self.feed(100, 145)
        self.feed(146, 146, KIND_CANDIDATES)

        self.assertEqual(self.events(), [GAP_EVENT, GAP_EVENT, CLEARED_EVENT])
        first, second, cleared = (d for _, d in self.lines)
        self.assertEqual((first["duration_s"], first["alert_n"], first["main_reason"]), (10.0, 1, "AXIS_HEAD_AMBIGUOUS"))
        self.assertEqual(first["part_track_ids"], [3])
        self.assertEqual(second["duration_s"], 40.0)
        self.assertEqual((cleared["cleared_by"], cleared["duration_s"], cleared["alerts"]), ("candidates", 46.0, 2))

        s = self.watch.summary()
        self.assertEqual((s["alerts"], s["gaps_alerted"], s["cleared_by"], s["longest_s"]), (2, 1, {"candidates": 1}, 46.0))

    def test_empty_frames_do_not_open(self):
        """물체가 아예 없는 상태가 길어도 줄을 내지 않는다."""
        self.feed(0, 60, KIND_EMPTY)
        self.assertEqual(self.lines, [])

    def test_object_gone_closes(self):
        """알린 공백 뒤 범위 안 볼트가 `clear_after_s`(2 s) 안 보이면 `no_detection`으로 풀린다."""
        self.feed(0, 12)
        self.feed(12.5, 30, KIND_EMPTY)

        self.assertEqual(self.events(), [GAP_EVENT, CLEARED_EVENT])
        self.assertEqual(self.lines[-1][1]["cleared_by"], "no_detection")
        self.assertEqual(self.lines[-1][1]["ros_stamp_ns"], int(14.5 * S))

    def test_clipped_then_empty_silent(self):
        """잘린 조각 몇 장 뒤 빈 화면만 이어지면 알림 없이 닫힌다(H-C9b 78 s 사례)."""
        self.feed(78.5, 79.0, reasons=Counter({"CONTOUR_CLIPPED": 1}))
        self.feed(79.5, 95, KIND_EMPTY)

        self.assertEqual(self.lines, [])
        self.assertEqual(self.watch.summary()["gaps_short"], 1)

    def test_brief_empty_keeps_gap(self):
        """공백 중 잠깐 검출이 빠져도 이어진 공백으로 센다."""
        self.feed(0, 5)
        self.feed(5.5, 7, KIND_EMPTY)
        self.feed(7.5, 11)

        self.assertEqual(self.events(), [GAP_EVENT])
        self.assertEqual(self.lines[0][1]["empty_frames"], 4)

    def test_rewind_and_finish(self):
        """시각이 되감기면 `time_rewind`, 실행 끝이면 `run_end`로 풀리고 finish는 여러 번 불러도 된다."""
        self.feed(100, 111)
        self.feed(5, 16)
        self.watch.finish("run_end")
        self.watch.finish("run_end")

        self.assertEqual([d.get("cleared_by") for e, d in self.lines if e == CLEARED_EVENT], ["time_rewind", "run_end"])

    def test_stamp_missing_ignored(self):
        """시각 없는 사진은 무시한다."""
        self.watch.observe(None, KIND_GAP, Counter({"X": 1}), Counter())
        self.assertIsNone(self.watch.summary()["open_alerted_s"])


class ConfigTest(unittest.TestCase):
    """`sort_part.candidate_gap` 블록."""

    def test_default_and_format(self):
        """블록이 없으면 설정 형식 그대로이고 송신기는 기본값(10 s·30 s)을 쓴다."""
        doc = base_doc()
        self.assertNotIn("candidate_gap", cfg().to_json())
        self.assertIsNone(parse_sort_part(doc).candidate_gap)
        self.assertEqual(DEFAULT_CANDIDATE_GAP.to_json(), {"alert_after_s": 10.0, "repeat_s": 30.0, "clear_after_s": 2.0})

    def test_block_round_trip(self):
        """블록이 있으면 읽고 요약 형식에 그대로 남긴다."""
        doc = {**base_doc(), "candidate_gap": {"alert_after_s": 5, "repeat_s": 20, "clear_after_s": 1}}
        c = parse_sort_part(doc)
        self.assertEqual(c.candidate_gap, CandidateGapConfig(5.0, 20.0, 1.0))
        self.assertEqual(c.to_json()["candidate_gap"], {"alert_after_s": 5.0, "repeat_s": 20.0, "clear_after_s": 1.0})

    def test_bad_block(self):
        """모르는 키·누락·0·bool·inf는 거부."""
        ok = {"alert_after_s": 5, "repeat_s": 20, "clear_after_s": 1}

        for bad in ({"alert_after_s": 5, "repeat_s": 20}, {**ok, "x": 1}, {**ok, "alert_after_s": 0},
                    {**ok, "repeat_s": True}, {**ok, "clear_after_s": float("inf")}, []):
            with self.subTest(bad=bad):
                with self.assertRaises(CandidateGapConfigError):
                    parse_candidate_gap(bad)

                with self.assertRaises(SortPartConfigError):
                    parse_sort_part({**base_doc(), "candidate_gap": bad})


class SenderTest(SenderBase):
    """송신기 연결: 공백 줄이 `sort_part.jsonl`에 들어가고 송신·계수는 그대로다."""

    def frame(self, t_s):
        """시뮬 시각만 있는 가짜 사진."""
        return SimpleNamespace(ros_stamp_ns=int(round(t_s * S)), image_seq=int(t_s * 10))

    def test_records_and_summary(self):
        """AI 결과가 아니면 무시, 공백 12 s면 줄 하나, 실행 끝에 run_end 풀림. 송신은 0."""
        from eyetoarm.vision import live_sort_part as lsp

        sender = self.make()
        real = lsp.classify_frame

        # 가짜 AI 결과(인스턴스 목록만)를 분류하도록 바꿔 끼운다. AI 결과가 아니면 None은 실제 함수로 확인
        self.assertIsNone(real(object(), NO_LIST_CANDIDATES))

        with unittest.mock.patch.object(lsp, "classify_frame",
                                        lambda display, skip: classify_instances(display, skip)):
            for i in range(25):
                sender.note_list_frame(self.frame(100 + i * 0.5), AMB, NO_LIST_CANDIDATES)

        summary = sender.summary()["candidate_gap"]
        recs = self.done()
        gaps = [r for r in recs if r["event"] in (GAP_EVENT, CLEARED_EVENT)]

        self.assertEqual([r["event"] for r in gaps], [GAP_EVENT, CLEARED_EVENT])
        self.assertEqual(gaps[0]["reasons"], {"AXIS_HEAD_AMBIGUOUS": 10 * 2 + 1})
        self.assertIsNone(gaps[0]["goal_id"])
        self.assertIsNone(gaps[0]["part_track_ids"])   # 추적 없음
        self.assertEqual(gaps[1]["cleared_by"], "run_end")
        self.assertEqual((summary["source"], summary["alerts"]), ("default", 1))
        self.assertEqual(self.rt.sent, [])


if __name__ == "__main__":
    unittest.main()


class LinkWiringTest(WorldCli):
    """CLI 경로(목록 단계 + W 추적 + 송신): 연결부가 사진마다 감시에 넣고, 공백 줄에 W 번호가 실린다."""

    def test_cli_records_gap_with_world_ids(self):
        """목록 후보를 0으로 바꿔 끼우면 sort_part.jsonl에 공백 줄·풀림 줄, 요약에 계수, 송신 0."""
        import json as _json
        import tempfile

        from eyetoarm.vision import candidate_gap as cg
        from eyetoarm.vision import pipeline_runner
        from test_stage_l_tracking import LIST, WORLD_DEFAULT, three
        from test_tracking_send_fixes import TRACK_SORT_CONFIG

        # 알림을 짧게 한 설정 사본(사진 간격 66 ms, 10장 ≈ 0.6 s)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        doc = _json.loads(TRACK_SORT_CONFIG.read_text(encoding="utf-8"))
        doc["sort_part"]["candidate_gap"] = {"alert_after_s": 0.2, "repeat_s": 0.2, "clear_after_s": 1.0}
        config = Path(tmp.name) / "gap.json"
        config.write_text(_json.dumps(doc), encoding="utf-8")

        # 실행기는 목록 후보 0(no_list_candidates)으로, 감시는 그 볼트를 「머리 방향 모호」로 본다
        with unittest.mock.patch.object(pipeline_runner, "list_candidate_indices", lambda *a: ()), \
                unittest.mock.patch.object(cg, "axis_candidate_reason", lambda inst: "AXIS_HEAD_AMBIGUOUS"):
            code, err, out, rt = self.run_world(three(10), extra=(*WORLD_DEFAULT, *LIST, "--live-sort-part"),
                                                config=str(config))

        self.assertEqual(code, 0, err)
        recs = [_json.loads(t) for t in (Path(out) / "sort_part.jsonl").read_text(encoding="utf-8").splitlines()]
        gaps = [r for r in recs if r["event"] == GAP_EVENT]
        cleared = [r for r in recs if r["event"] == CLEARED_EVENT]

        self.assertTrue(gaps, [r["event"] for r in recs])
        self.assertEqual(gaps[0]["main_reason"], "AXIS_HEAD_AMBIGUOUS")
        self.assertTrue(any(r["part_track_ids"] for r in gaps), gaps)   # W 번호가 실림
        self.assertEqual([r["cleared_by"] for r in cleared], ["run_end"])
        self.assertEqual(rt.sent, [])

        summary = _json.loads((Path(out) / "summary.json").read_text(encoding="utf-8"))["sort_part"]["candidate_gap"]
        self.assertEqual((summary["source"], summary["alerts"], summary["gaps_alerted"]), ("config", len(gaps), 1))
