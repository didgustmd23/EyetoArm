"""단계 L(목록) 추적 시험: 후보별 확정 번호, 송신 실효 번호·불일치·release 규칙, B 한 줄 반영 모사.

ROS·Gazebo 없이 가짜 TF·런타임·모델과 실제 `BStage.process_frame`을 쓴다. 「B 수정 뒤」는
`BStage._candidate_view`를 바꾼 시험 대역으로 모사한다(B 파일 무변).
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from eyetoarm.common.types import ObjectObservation, PixelBasis
from eyetoarm.transforms import b_stage as b_stage_mod
from eyetoarm.vision import live_sort_part as spc
from eyetoarm.vision.handoff import AToCHandoff
from eyetoarm.vision.live_observation_link import LiveObservationLink
from eyetoarm.vision.live_world_tracking import LiveWorldTracking
from eyetoarm.vision.tracked_observation import (NO_ASSIGNMENT, NOT_UPDATED, OTHER_JOB, TrackedObjectObservation,
                                                 candidate_track_id, with_candidate_track)
from eyetoarm.vision.tracking import TrackAssignment, TrackFrame, TrackMode, TrackState
from test_live_a_wiring import WiringBase
from test_live_sort_part import FakeRuntime
from test_live_stage_l import three_bolts
from test_live_tracking_link import PIXEL, ReleasingRuntime, TrackCli, sort_config
from test_live_world_tracking import WORLD_CONFIG, WorldCli
from test_run_pipeline_live_observation import CONFIG, Scenario
from test_tracking_send_fixes import TRACK_SORT_CONFIG

LIST = ("--b-input", "list")
WORLD_DEFAULT = ("--tracking", "--world-tracker-config", str(WORLD_CONFIG))   # 공간 생략 = 실시간 기본 W


def accepts(test, out):
    """수락 기록."""
    return [r for r in test.records(out) if r["kind"] == "accept"]


def candidate_ids(rec):
    """수락 줄 봉투 후보의 (instance_index, part_track_id, track_state)."""
    return [(c["instance_index"], c.get("part_track_id"), c.get("track_state")) for c in rec["b_input"]["candidates"]]


@contextlib.contextmanager
def b_reads_ids(offset=0):
    """B 요청 한 줄 반영 모사: `_candidate_view`가 후보 번호를 읽는다(offset을 주면 일부러 어긋나게)."""
    real = b_stage_mod.BStage._candidate_view

    def view(c, fo):
        """원래 보기 + 후보 번호."""
        v = real(c, fo)
        tid = getattr(c, "part_track_id", None)
        v.part_track_id = None if tid is None else tid + offset

        return v

    with mock.patch.object(b_stage_mod.BStage, "_candidate_view", staticmethod(view)):
        yield


def three(n):
    """볼트 3개 × n장 장면."""
    return Scenario(n, masks={i: three_bolts for i in range(n)})


# ---------------------------------------------------------------------- 형·규칙

class CandidateTypeTest(TrackCli):
    """`TrackedObjectObservation`·`candidate_track_id`·봉투 기록."""

    def cand(self, **over):
        """볼트 후보 하나."""
        base = dict(instance_index=2, class_name="bolt", confidence=0.9, pixel_basis=PixelBasis.RAW, observed_face="axis",
                    axis_endpoints_px=np.array([[10.0, 10.0], [20.0, 30.0]]), calc_contour=None)
        base.update(over)

        return ObjectObservation(**base)

    def test_subtype_keeps_fields_and_checks_id(self):
        """하위 형은 상위 필드·검사를 그대로 거치고 번호는 양의 정수 또는 None(bool·0 거부)."""
        c = with_candidate_track(self.cand(), 3, "confirmed")
        self.assertIsInstance(c, ObjectObservation)
        self.assertEqual((c.instance_index, c.part_track_id, c.track_state), (2, 3, "confirmed"))
        np.testing.assert_array_equal(c.axis_endpoints_px, self.cand().axis_endpoints_px)

        for bad in (0, True, 1.0):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                with_candidate_track(self.cand(), bad, "confirmed")

    def test_candidate_track_id_rules(self):
        """확정만 번호. 다른 작업·갱신 안 함·배정 없음·tentative는 None + 상태."""
        f = SimpleNamespace(run_id="r", source_id="s", image_seq=1, capture_generation=0)
        ctx = SimpleNamespace(frame=f, generation=0, job_seq=1, detector_config_id="d")
        tf = TrackFrame(run_id="r", source_id="s", image_seq=1, capture_generation=0, generation=0, job_seq=1,
                        detector_config_id="d", timestamp_ms=1, mode=TrackMode.WORLD, track_namespace="n", tracker_epoch=1,
                        updated=True, event=None,
                        assignments=(TrackAssignment(0, 4, TrackState.CONFIRMED, 3, None),
                                     TrackAssignment(1, None, TrackState.TENTATIVE, 1, None)))

        self.assertEqual(candidate_track_id(tf, ctx, 0), (4, "confirmed"))
        self.assertEqual(candidate_track_id(tf, ctx, 1), (None, "tentative"))
        self.assertEqual(candidate_track_id(tf, ctx, 7), (None, NO_ASSIGNMENT))
        self.assertEqual(candidate_track_id(tf, SimpleNamespace(**{**vars(ctx), "job_seq": 2}), 0), (None, OTHER_JOB))
        self.assertEqual(candidate_track_id(TrackFrame(**{**vars(tf), "updated": False}), ctx, 0), (None, NOT_UPDATED))


# ---------------------------------------------------------------------- CLI: 픽셀

class PixelStageLTest(TrackCli):
    """단계 L + 픽셀 추적(+ 송신)."""

    def setUp(self):
        """B 단계 수집."""
        super().setUp()
        self.stages = []

    def run_l(self, n=8, *, require=True, runtime=None, extra=()):
        """단계 L + 픽셀 추적 + 송신 실행."""
        rt = runtime or ReleasingRuntime()
        code, err, out = self.run_sort(three(n), config=sort_config(self.tmp.name, require=require),
                                       extra=("--live-sort-part", "--tracking", *PIXEL, *LIST, *extra),
                                       factory=self.factory(rt), hook=self.stages.append)

        return code, err, out, rt

    def test_candidate_ids_on_envelope(self):
        """확정 뒤 후보마다 다른 양의 번호가 유지되고, 확정 전은 None + 상태, B 결과 번호는 None이다."""
        code, err, out, rt = self.run_l()
        self.assertEqual(code, 0, err)

        recs = accepts(self, out)
        cands = [candidate_ids(r) for r in recs]
        confirmed = [c for c in cands if all(t is not None for _, t, _ in c)]

        self.assertTrue(confirmed, cands)
        self.assertTrue(all(len({t for _, t, _ in c}) == 3 for c in confirmed))
        self.assertEqual(len({tuple(t for _, t, _ in c) for c in confirmed}), 1)   # 같은 볼트 같은 번호 유지
        self.assertTrue(any(t is None and s for c in cands for _, t, s in c))     # 확정 전 프레임
        self.assertTrue(all(res["part_track_id"] is None for r in recs for res in r["consumer_result"]["results"]))
        self.assertIn("(stage L: A candidate ids until B reads them)", err)

        s = self.summary(out)["tracking_link"]["counts"]
        self.assertGreater(s["list_candidates_with_track"], 0)

    def test_send_uses_chosen_a_id(self):
        """송신 기록 번호 = 고른 후보의 A 번호. require true면 번호 없는 볼트는 skip:no_track_id."""
        code, err, out, rt = self.run_l()
        self.assertEqual(code, 0, err)

        recs = self.sort_records(out)
        sent = [r for r in recs if r["event"] == "sent"]
        skips = [r for r in recs if r["event"] == "skip:no_track_id"]

        self.assertTrue(sent and all(r["track_source"] == "a" and r["goal_part_track_id"] is None
                                     and r["part_track_id"] == r["a_part_track_id"] is not None for r in sent))
        self.assertTrue(skips and all(r["a_part_track_id"] is None and r["track_state"] for r in skips))
        self.assertFalse(any(r["event"].startswith("release:") for r in recs))   # release 0
        self.assertFalse(any(r["event"] == "track_id_mismatch" for r in recs))
        self.assertTrue(all(st._pending == {} for st in self.stages))

    def test_track_keys_only_in_stage_l(self):
        """단계 L 전용 키는 단계 T 송신 기록에 없다(T 형식 그대로)."""
        rt = ReleasingRuntime()
        code, err, out = self.run_sort(Scenario(6), config=sort_config(self.tmp.name, require=False),
                                       extra=("--live-sort-part", "--tracking", *PIXEL), factory=self.factory(rt))
        self.assertEqual(code, 0, err)
        self.assertFalse(any("track_source" in r or "a_part_track_id" in r for r in self.sort_records(out)))

    def test_b_reads_ids_goal_wins_and_dedupes(self):
        """B 한 줄 반영 모사: 목표 번호 = A 번호, 뒤 프레임은 ALREADY_SENT, release는 목표 번호."""
        rt = ReleasingRuntime()

        with b_reads_ids():
            code, err, out, rt = self.run_l(10, runtime=rt)

        self.assertEqual(code, 0, err)
        recs = self.sort_records(out)
        sent = [r for r in recs if r["event"] == "sent"]

        self.assertTrue(sent and all(r["track_source"] == "goal" and r["goal_part_track_id"] == r["a_part_track_id"]
                                     for r in sent))
        self.assertTrue(any(r["event"] == "skip:ALREADY_SENT" for r in recs))
        self.assertFalse(any(r["event"] == "track_id_mismatch" for r in recs))

        gid = sent[0]["goal_id"]
        rt.finish(gid)
        self.assertIn(sent[0]["goal_part_track_id"], rt.released)

    def test_mismatch_counted_and_goal_followed(self):
        """목표 번호와 A 번호가 다르면 track_id_mismatch 기록·계수, 실효 번호는 목표 번호."""
        with b_reads_ids(offset=100):
            code, err, out, rt = self.run_l(8)

        self.assertEqual(code, 0, err)
        recs = self.sort_records(out)
        mism = [r for r in recs if r["event"] == "track_id_mismatch"]

        self.assertTrue(mism)
        self.assertTrue(all(r["part_track_id"] == r["goal_part_track_id"] == r["a_part_track_id"] + 100 for r in mism))
        self.assertEqual(self.summary(out)["sort_part"]["events"]["track_id_mismatch"], len(mism))

    def test_record_only_release_finds_list_goal(self):
        """기록 전용(송신 끔) + B 한 줄 모사: 단계 L 반환 사전의 goal로 B 기록을 푼다(`_goal_of`)."""
        with b_reads_ids():
            code, err, out = self.run_sort(three(8), config=CONFIG, extra=("--tracking", *PIXEL, *LIST),
                                           hook=self.stages.append)

        self.assertEqual(code, 0, err)
        self.assertGreater(self.summary(out)["tracking_link"]["counts"].get("released_record_only", 0), 0)
        self.assertTrue(all(st._pending == {} for st in self.stages))

    def test_not_accepted_release_finds_list_goal(self):
        """검사 ③ 거부(단계 L 사전 반환): `_goal_of`로 고른 목표를 찾아 목표 번호로 release."""
        released = []
        link = SimpleNamespace(sort_part=None, release_proxy=SimpleNamespace(release=released.append),
                               _count_tracking=lambda k: None, _goal_of=LiveObservationLink._goal_of)
        payload = SimpleNamespace(output={"goal": {"kind": "PICK_PLACE", "part_track_id": 7}})

        LiveObservationLink._on_not_accepted(link, SimpleNamespace(), payload)
        LiveObservationLink._on_not_accepted(link, SimpleNamespace(), SimpleNamespace(output={"goal": None}))

        self.assertEqual(released, [7])

    def test_unit_retry_limit_by_a_id(self):
        """번호별 재시도 제한이 A 번호로 걸려 실패 1회 뒤 같은 번호는 skip:track_retry_limit."""
        doc = json.loads(TRACK_SORT_CONFIG.read_text(encoding="utf-8"))["sort_part"]
        doc["retry"] = {"max_failures_per_track": 1, "failed_point_block_s": 1.0, "failed_point_block_mm": 1.0}
        rt = FakeRuntime()
        sender = spc.LiveSortPartSender("r", spc.parse_sort_part(doc), rt, self.tmp.name, tracking=True)
        rt.set_event_sink(sender.on_client_event)
        frame = SimpleNamespace(image_seq=1, ros_stamp_ns=10)
        ok = SimpleNamespace(valid=True)

        def out(gid):
            """번호 없는 PICK_PLACE 목표(관측 위치 없음 → 지점 차단 없음)."""
            return SimpleNamespace(goal={"kind": "PICK_PLACE", "goal_id": gid, "part_track_id": None}, result=ok)

        ids = {"job_seq": 1, "generation": 1, "a_part_track_id": 5, "track_state": "confirmed"}
        self.assertEqual(sender.offer(out("g1"), frame, ids, 1), "sent")
        rt.finish("g1", "FAILED")
        self.assertEqual(sender.offer(out("g2"), frame, ids, 2), "skip:track_retry_limit")
        self.assertEqual(sender.offer(out("g3"), frame, {**ids, "a_part_track_id": 6}, 3), "sent")
        sender.close_run("test")
        sender.close(2.0)


# ---------------------------------------------------------------------- CLI: W 기본

class WorldStageLTest(WorldCli):
    """단계 L + 실시간 기본 W 추적."""

    def test_world_default_candidate_ids_and_send(self):
        """공간 생략 = W: 후보 번호(W)·tracking_space default, require true 송신은 A 번호만, 계수."""
        code, err, out, rt = self.run_world(three(10), extra=(*WORLD_DEFAULT, *LIST, "--live-sort-part"),
                                            config=str(TRACK_SORT_CONFIG))
        self.assertEqual(code, 0, err)

        s = self.summary(out)
        self.assertEqual(s["tracking_space"], {"space": "world", "source": "default"})
        cands = [candidate_ids(r) for r in accepts(self, out)]
        self.assertTrue(any(all(t is not None for _, t, _ in c) for c in cands), cands)
        self.assertGreater(s["tracking_link"]["counts"].get("world_result", 0), 0)

        sent = [r for r in self.sort_records(out) if r["event"] == "sent"]
        self.assertTrue(sent and all(r["a_part_track_id"] is not None for r in sent))

    def test_world_wait_timeout_marks_pending(self):
        """W 결과를 못 받으면(대기 시간 초과) 후보 전부 None + track_pending, 계수 world_track_pending."""
        with mock.patch.object(LiveWorldTracking, "wait", lambda self, ctx, timeout_s=None: None):
            code, err, out, rt = self.run_world(three(4), extra=(*WORLD_DEFAULT, *LIST), config=str(CONFIG))

        self.assertEqual(code, 0, err)
        cands = [candidate_ids(r) for r in accepts(self, out)]
        self.assertTrue(cands and all(t is None and s == "track_pending" for c in cands for _, t, s in c))
        self.assertGreater(self.summary(out)["tracking_link"]["counts"]["world_track_pending"], 0)

    def test_no_list_candidates_frames_go_to_world(self):
        """단계 L은 볼트 없는 프레임도 W 추적 큐에 넣는다: offered = 전체 프레임."""
        sc = Scenario(6, masks={i: three_bolts for i in range(0, 6, 2)})   # 홀수 프레임은 볼트 없음
        code, err, out, rt = self.run_world(sc, extra=(*WORLD_DEFAULT, *LIST), config=str(CONFIG))
        self.assertEqual(code, 0, err)

        offered = self.summary(out)["world_tracking"]["counts"]["offered"]
        listed = self.summary(out)["list_consumer_calls"]
        self.assertGreater(offered, listed)

    def test_world_list_sets_peek(self):
        """단계 L W 실행도 detect 조회 함수(`peek`)를 연결하고 요약에 display_lookup이 있다."""
        made = []

        class Probe:
            """가짜 C probe."""

            def notify(self):
                """알림 자리."""

            def start(self, handoff):
                """시작 자리."""

            def stop(self):
                """정지 자리."""

        def factory():
            """실제 연결부 + 가짜 probe."""
            h = AToCHandoff(notify=Probe().notify)
            made.append(h)

            return h, Probe()

        with mock.patch.object(self.mod, "_default_live_handoff", factory):
            code, err, out, rt = self.run_world(three(4), extra=(*WORLD_DEFAULT, *LIST), config=str(CONFIG))

        self.assertEqual(code, 0, err)
        self.assertEqual(getattr(made[-1]._track_source, "__name__", None), "peek")
        self.assertIn("display_lookup", self.summary(out)["world_tracking"])

    def test_capture_meta_tracking_still_refused(self):
        """저장 입력 단계 L(--capture-meta) + --tracking은 여전히 인자 오류 2(범위 밖)."""
        args = self.mod.build_parser().parse_args(["--images", "a.png", "--log", "x.jsonl", "--detector", "ai", "--tracking",
                                                   "--capture-meta", "m.json", "--observation-out", "o", *LIST])

        with contextlib.redirect_stderr(io.StringIO()) as e, self.assertRaises(SystemExit):
            self.mod._check_source_args(args, None)

        self.assertIn("--capture-meta cannot be combined with --tracking", e.getvalue())


# ---------------------------------------------------------------------- CLI: center 발행 함께

class CenterStageLTest(WiringBase):
    """단계 L + 추적 + B center 발행 + 송신을 한 실행에서."""

    def test_all_options_together(self):
        """--b-input list --tracking(픽셀) --b-center-overlay --live-sort-part: 0, center·후보 번호."""
        code, err, out = self.run_sort(three(6), config=sort_config(self.tmp.name, require=False),
                                       extra=(*LIST, "--tracking", *PIXEL, "--b-center-overlay", "--live-sort-part"))

        self.assertEqual(code, 0, err)
        self.assertTrue(self.pubs and self.pubs[0].published)
        self.assertTrue(any(t is not None for r in accepts(self, out) for _, t, _ in candidate_ids(r)))
