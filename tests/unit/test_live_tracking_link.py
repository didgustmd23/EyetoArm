"""#215 추적 번호 → 실시간 A→B→C 연결 시험(T1~T12).

근거: 계획 `_workspace/215_builder2_tracking_live_plan_r0.md`(reviewer PASS 2026-11-20: Q1 (가) A 하위 형, Q2 `offer(tracks=)` 허용,
`not_accepted` release는 `PICK_PLACE`만, Q5 근방 금지는 번호가 있어도 적용, Q6 `retry` 선택·없으면 `retry=off`).
- 실제 ROS·Gazebo는 쓰지 않는다. CLI 시험은 기존 실시간 기록 연결 시험의 가짜 수신·가짜 모델(같은 볼트 마스크 → 같은 박스)·가짜 TF와
  #214 가짜 송신 런타임을 쓴다. 추적기는 실제 `PartTracker`(M2, 기본 임시값: 연속 3 hit 확정)다.
- B 계산은 실제 `BStage`다: 봉투 → B 입구 어댑터 → `BStage.consume`까지 하위 형 관측·번호가 남아 `goal.part_track_id`·`ALREADY_SENT`가
  나오는지 끝까지 본다(reviewer 조건).
"""
from __future__ import annotations

import dataclasses
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from eyetoarm.transforms import b_stage as b_stage_mod
from eyetoarm.vision import tracking as trk
from eyetoarm.vision.live_observation_link import _StageReleaseProxy
from eyetoarm.vision.live_sort_part import LiveSortPartSender, RetryConfig, SortPartConfig, parse_sort_part
from eyetoarm.vision.tracked_observation import (NO_ASSIGNMENT, NO_SELECTION, NO_TRACKS, NOT_UPDATED, OTHER_JOB,
                                                 TrackedPlanarObservation, confirmed_track_id, with_track)
from test_live_sort_part import SORT_CONFIG, CliBase, FakeRuntime, records
from test_run_pipeline_live_observation import CONFIG, Scenario
from test_view_bias import write_bias_config

ROOT = Path(__file__).resolve().parents[2]


class ReleasingRuntime(FakeRuntime):
    """B 클라이언트의 release 규칙까지 흉내내는 가짜 런타임: no_server·busy·결과 때 `stage.release(번호)`."""

    def __init__(self, **kw):
        """release 대상은 진입점이 `set_release_target`으로 준다."""
        super().__init__(**kw)
        self.target = None
        self.released = []

    def set_release_target(self, target):
        """B 클라이언트 `stage` 자리."""
        self.target = target

    def _release(self, goal):
        tid = goal.get("part_track_id")

        if self.target is not None and tid:
            self.released.append(tid)
            self.target.release(tid)

    def send(self, out):
        """B 규칙: 보내지 않으면(no_server·busy) release."""
        ok = super().send(out)

        if not ok and out.goal.get("kind") in ("PICK_PLACE", "PASS_THROUGH"):
            self._release(out.goal)

        return ok

    def finish(self, gid, status="SUCCEEDED", reason=""):
        """결과 → 기록 사건 → release(B `_finish` 순서: 대기 제거 → 기록 → release)."""
        goal = self.inflight.get(gid) or {}
        super().finish(gid, status, reason)
        self._release(goal)


# #239: 실시간 --tracking 기본 공간이 W가 되어, 픽셀 추적 시험은 공간을 명시한다
PIXEL = ("--tracking-space", "pixel")


def sort_config(tmp, retry=None, require=False) -> str:
    """송신 시험 설정(+ retry 블록 선택). #239: 추적 켬 송신은 `require_track_id`가 필수라 기본 false(번호 None도 보냄, 기존 동작)."""
    doc = json.loads(SORT_CONFIG.read_text(encoding="utf-8"))
    doc["sort_part"]["require_track_id"] = require

    if retry is not None:
        doc["sort_part"]["retry"] = retry

    path = Path(tmp) / f"sort_{'retry' if retry else 'plain'}_{'req' if require else 'noreq'}.json"
    path.write_text(json.dumps(doc), encoding="utf-8")

    return str(path)


class TrackCli(CliBase):
    """CLI 도우미: B 입력 관측 감시(실제 B 위에 한 겹)."""

    def setUp(self):
        """B가 받은 관측과 B 목표를 모은다."""
        super().setUp()
        self.b_seen = []
        prev = b_stage_mod.BStage.consume

        def consume(stage, frame, obs):
            """B 입력 관측(형·번호)과 B 목표."""
            out = prev(stage, frame, obs)
            self.b_seen.append((type(obs).__name__, getattr(obs, "part_track_id", None), out.goal.get("kind"),
                                out.goal.get("reason"), out.goal.get("part_track_id"), frame.ros_stamp_ns))
            return out

        p = mock.patch.object(b_stage_mod.BStage, "consume", consume)
        p.start()
        self.addCleanup(p.stop)

    def run_track(self, scenario, *, runtime=None, retry=None, extra=("--live-sort-part", "--tracking", *PIXEL), config=None,
                  max_frames=None):
        """추적 + 송신 실행."""
        cfg = config or sort_config(self.tmp.name, retry)
        rt = runtime or ReleasingRuntime()
        code, err, out = self.run_sort(scenario, config=cfg, extra=extra, factory=self.factory(rt), max_frames=max_frames)

        return code, err, out, rt


# ---------------------------------------------------------------------- 순수 규칙

def track_frame(assignments, *, updated=True, job=None):
    """TrackFrame 대역(작업 키는 `job`과 맞춤)."""
    job = job or ctx()
    f = job.frame

    return trk.TrackFrame(run_id=f.run_id, source_id=f.source_id, image_seq=f.image_seq, capture_generation=f.capture_generation,
                          generation=job.generation, job_seq=job.job_seq, detector_config_id=job.detector_config_id,
                          timestamp_ms=1, mode=trk.TrackMode.HEURISTIC, track_namespace="ns", tracker_epoch=1,
                          updated=updated, event=None, assignments=tuple(assignments))


def ctx(image_seq=5):
    """JobContext 대역."""
    return SimpleNamespace(frame=SimpleNamespace(run_id="r", source_id="s", image_seq=image_seq, capture_generation=0),
                           generation=1, job_seq=9, detector_config_id="d")


def asg(index, state, tid=None):
    """배정 대역."""
    return trk.TrackAssignment(instance_index=index, part_track_id=tid, state=state, hits=3, reason=None)


class ConfirmedRuleTest(unittest.TestCase):
    """T3(규칙): 확정 번호만."""

    def test_only_confirmed_selected_instance(self):
        """CONFIRMED + 선택 index만 번호, 나머지 상태·조건은 None과 사유."""
        obs = SimpleNamespace(selected_index=1)
        C, T, A, U = trk.TrackState.CONFIRMED, trk.TrackState.TENTATIVE, trk.TrackState.AMBIGUOUS, trk.TrackState.UNTRACKED
        cases = [
            (track_frame([asg(0, C, 4), asg(1, C, 7)]), (7, "confirmed")),
            (track_frame([asg(1, T)]), (None, "tentative")),
            (track_frame([asg(1, A)]), (None, "ambiguous")),
            (track_frame([asg(1, U)]), (None, "untracked")),
            (track_frame([asg(1, C, 7)], updated=False), (None, NOT_UPDATED)),
            (track_frame([asg(1, C, 7)], job=ctx(image_seq=6)), (None, OTHER_JOB)),
            (track_frame([asg(0, C, 4)]), (None, NO_ASSIGNMENT)),
            (track_frame([asg(1, C, 7), asg(1, C, 8)]), (None, NO_ASSIGNMENT)),
            (None, (None, NO_TRACKS)),
        ]

        for tf, expect in cases:
            with self.subTest(expect=expect):
                self.assertEqual(confirmed_track_id(tf, ctx(), obs), expect)

        self.assertEqual(confirmed_track_id(track_frame([asg(1, C, 7)]), ctx(), SimpleNamespace(selected_index=None)),
                         (None, NO_SELECTION))

    def test_subtype_keeps_fields_and_replace(self):
        """하위 형: 상위 필드 그대로·isinstance·replace(#207 보정) 뒤 번호 유지, bool·0 번호 거부."""
        import numpy as np

        from eyetoarm.common.types import PlanarObservation

        base = PlanarObservation(run_id="r", source_id="s", image_seq=1, capture_generation=0, generation=0, job_seq=1,
                                 detector_config_id="d", pixel_basis="raw", calibration_id="c", observed_face="axis",
                                 calc_contour=None, display_mask_contour=None, candidate_count=1, selected_index=0,
                                 valid=True, reason=None, process_ms=1.0, method="ai", class_name="bolt", confidence=0.9,
                                 model_id="m", axis_endpoints_px=np.array([[1.0, 2.0], [3.0, 4.0]]))
        t = with_track(base, 7, "confirmed")
        self.assertIsInstance(t, PlanarObservation)
        self.assertEqual({f.name: getattr(t, f.name) for f in dataclasses.fields(PlanarObservation) if f.name != "axis_endpoints_px"},
                         {f.name: getattr(base, f.name) for f in dataclasses.fields(PlanarObservation) if f.name != "axis_endpoints_px"})
        moved = dataclasses.replace(t, axis_endpoints_px=np.array([[1.5, 2.0], [3.5, 4.0]]))
        self.assertEqual((type(moved), moved.part_track_id, moved.track_state), (TrackedPlanarObservation, 7, "confirmed"))

        for bad in (True, 0, -1, 1.0):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                with_track(base, bad, "confirmed")

        self.assertIsNone(with_track(base, None, "tentative").part_track_id)


class ProxyTest(unittest.TestCase):
    """T8: 프록시는 연결부가 만든 모든 B 단계에 release."""

    def test_releases_every_stage(self):
        """이전 단계(프로필 변경 전)·현재 단계 모두 release, 호출 수."""
        stages = [mock.Mock(), mock.Mock()]
        link = SimpleNamespace(_lock=threading.Lock(), stages=[(stages[0], None), (stages[1], None)])
        proxy = _StageReleaseProxy(link)
        proxy.release(7)
        self.assertEqual([s.release.call_args_list for s in stages], [[mock.call(7)], [mock.call(7)]])
        self.assertEqual(proxy.calls, 1)


# ---------------------------------------------------------------------- 송신기 재시도·release(가짜 런타임)

def out_of(gid, track=None, kind="PICK_PLACE", pos=(400.0, 0.0, 105.0), stamp=1_000_000_000):
    """B 출력 대역."""
    return SimpleNamespace(goal={"kind": kind, "goal_id": gid, "spec": "BOLT_B", "reason": None, "part_track_id": track,
                                 "observed_position_mm": list(pos), "capture_stamp_ns": stamp},
                           result=SimpleNamespace(valid=True, reason=None))


class SenderRetryTest(unittest.TestCase):
    """T7·T9·T10(송신기)."""

    def make(self, retry=None, *, tracking=True):
        """송신기·가짜 런타임·release 기록."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.rt = FakeRuntime()
        self.target = mock.Mock()
        cfg = SortPartConfig(server_wait_s=0.01, result_wait_s=0.05, close_limit_s=2.0, spin_timeout_s=0.01, retry=retry)
        self.s = LiveSortPartSender("run", cfg, self.rt, self.tmp.name, tracking=tracking, release_target=self.target)
        self.rt.set_event_sink(self.s.on_client_event)
        self.s.start()

        return self.s

    def recs(self):
        """정리 후 기록."""
        self.assertTrue(self.s.close(2.0))

        return records(Path(self.tmp.name) / "sort_part.jsonl")

    def test_track_failure_limit_and_success_clears(self):
        """번호별 실패 2회 뒤 skip:track_retry_limit(+release), 성공하면 지움(근방 금지는 시간 창 밖으로 피함)."""
        s = self.make(RetryConfig(max_failures_per_track=2, failed_point_block_s=0.001, failed_point_block_mm=1.0))
        frame = SimpleNamespace(image_seq=1, ros_stamp_ns=1)

        for i, stamp in enumerate((1, 2)):
            self.assertEqual(s.offer(out_of(f"g{i}", 7, stamp=stamp * 10 ** 9), frame, {}, i), "sent")
            self.rt.finish(f"g{i}", "FAILED")

        self.assertEqual(s.offer(out_of("g2", 7, stamp=3 * 10 ** 9), frame, {}, 2), "skip:track_retry_limit")
        self.assertEqual(self.target.release.call_args_list, [mock.call(7)])
        self.assertEqual(s.offer(out_of("g3", 8, stamp=4 * 10 ** 9), frame, {}, 3), "sent")   # 다른 번호는 보냄
        self.rt.finish("g3", "SUCCEEDED")
        recs = self.recs()
        self.assertTrue(all("part_track_id" in r for r in recs))
        self.assertIn("release:skip:track_retry_limit", [r["event"] for r in recs])
        self.assertEqual(s.summary()["retry"]["track_failures"], {"7": 2})

    def test_point_block_with_and_without_track(self):
        """실패 집는 점 근방(거리·시간 창 안)은 번호가 있어도 없어도 막음, 창 밖·반경 밖은 보냄."""
        s = self.make(RetryConfig(max_failures_per_track=5, failed_point_block_s=10.0, failed_point_block_mm=15.0))
        frame = SimpleNamespace(image_seq=1, ros_stamp_ns=1)
        self.assertEqual(s.offer(out_of("g0", None, stamp=10 ** 9), frame, {}, 0), "sent")
        self.rt.finish("g0", "FAILED")
        cases = [(out_of("a", None, pos=(405.0, 5.0, 105.0), stamp=5 * 10 ** 9), "skip:failed_point_block"),
                 (out_of("b", 9, pos=(410.0, 0.0, 105.0), stamp=5 * 10 ** 9), "skip:failed_point_block"),
                 (out_of("c", 9, pos=(420.0, 0.0, 105.0), stamp=5 * 10 ** 9), "sent"),        # 20 mm 밖
                 ]

        for out, expect in cases:
            with self.subTest(gid=out.goal["goal_id"]):
                self.assertEqual(s.offer(out, frame, {}, 1), expect)

        self.rt.finish("c", "SUCCEEDED")
        self.assertEqual(s.offer(out_of("d", None, stamp=12 * 10 ** 9), frame, {}, 2), "sent")   # 창(10 s) 밖
        self.assertEqual(self.target.release.call_args_list, [mock.call(9)])                   # 번호 있는 것만 release
        self.recs()

    def test_no_call_release_cases(self):
        """클라이언트를 부르지 않은 PICK_PLACE(게이트 닫힘·상태 실패)는 release, PASS_THROUGH·번호 없음은 release 없음."""
        s = self.make()
        frame = SimpleNamespace(image_seq=1, ros_stamp_ns=1)
        s.close_run("reset")
        self.assertEqual(s.offer(out_of("g1", 3), frame, {}, 1), "gate_closed")
        self.assertEqual(s.offer(out_of("g2", 4, kind="PASS_THROUGH"), frame, {}, 2), "gate_closed")
        self.assertEqual(s.offer(out_of("g3", None), frame, {}, 3), "gate_closed")
        self.assertEqual(self.target.release.call_args_list, [mock.call(3)])
        self.recs()

    def test_already_sent_skip_not_released(self):
        """B ALREADY_SENT SKIP은 기록만 하고 release하지 않는다."""
        s = self.make()
        out = out_of("g1", 3, kind="SKIP")
        out.goal["reason"] = "ALREADY_SENT"
        self.assertEqual(s.offer(out, SimpleNamespace(image_seq=1, ros_stamp_ns=1), {}, 1), "skip:ALREADY_SENT")
        self.assertEqual(self.target.release.call_args_list, [])
        self.recs()

    def test_tracking_off_record_shape(self):
        """추적 끔: 기록에 part_track_id 키 없음(#214 형식 그대로), retry 없음이면 요약 retry=off."""
        s = self.make(tracking=False)
        s.offer(out_of("g1", None), SimpleNamespace(image_seq=1, ros_stamp_ns=1), {}, 1)
        recs = self.recs()
        self.assertFalse(any("part_track_id" in r for r in recs))
        self.assertEqual((s.summary()["retry"], s.summary()["config"]["retry"]), ("off", "off"))

    def test_retry_config_parse(self):
        """retry 블록: 선택, 키 정확히 3개, 정수 ≥ 1·양수만."""
        base = {"server_wait_s": 1, "result_wait_s": 1, "close_limit_s": 1, "spin_timeout_s": 1}
        self.assertIsNone(parse_sort_part(base).retry)
        good = {"max_failures_per_track": 2, "failed_point_block_s": 10.0, "failed_point_block_mm": 15.0}
        self.assertEqual(parse_sort_part({**base, "retry": good}).retry.max_failures_per_track, 2)

        for bad in ({**good, "max_failures_per_track": 0}, {**good, "max_failures_per_track": True},
                    {**good, "failed_point_block_s": 0}, {k: v for k, v in good.items() if k != "failed_point_block_mm"},
                    {**good, "x": 1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_sort_part({**base, "retry": bad})


# ---------------------------------------------------------------------- CLI(실제 추적기·실제 BStage)

class TrackingCliTest(TrackCli):
    """T1·T3·T4·T5·T6·T7·T10·T11·T12."""

    def test_confirmed_id_reaches_b_goal_and_dedupe(self):
        """T3·T4·T6: 확정 뒤 B 입력은 하위 형 + 번호, goal.part_track_id = 번호, 결과 전 두 번째부터 ALREADY_SENT, 송신 1회."""
        code, err, out, rt = self.run_track(Scenario(8))
        self.assertEqual(code, 0, err)
        self.assertIn("tracking ON M2(heuristic)", err)
        self.assertIn("retry=off tracking=M2(heuristic)", err)
        acc = self.accepts(out)
        ids = [r["input"]["part_track_id"] for r in acc]
        confirmed = [i for i in ids if i is not None]
        self.assertTrue(confirmed and len(set(confirmed)) == 1, ids)
        self.assertTrue(all(t == "TrackedPlanarObservation" for t, *_ in self.b_seen))

        for r in acc:
            self.assertEqual(r["goal"]["part_track_id"], r["input"]["part_track_id"])
            self.assertEqual(r["b_input"]["extensions"]["tracking"]["part_track_id"], r["input"]["part_track_id"])
            self.assertEqual(r["input"]["track_state"] == "confirmed", r["input"]["part_track_id"] is not None)

        with_id = [r for r in acc if r["input"]["part_track_id"] is not None]
        self.assertEqual(with_id[0]["goal_kind"], "PICK_PLACE")
        self.assertTrue(all((r["goal_kind"], r["b_reason"]) == ("SKIP", "ALREADY_SENT") for r in with_id[1:]), with_id)
        self.assertEqual([g["part_track_id"] for g in rt.sent if g["part_track_id"] is not None], [confirmed[0]])
        srec = self.sort_records(out)
        self.assertTrue(all("part_track_id" in r for r in srec))
        self.assertEqual(sum(1 for r in srec if r["event"] == "skip:ALREADY_SENT"), len(with_id) - 1)
        s = self.summary(out)
        self.assertGreaterEqual(s["tracking_link"]["counts"]["accepted_with_track"], 1)
        self.assertIsNotNone(s["tracking"])                       # 실행기 추적 보고

    def test_result_releases_then_sends_again(self):
        """T6: 결과(성공)가 오면 B 기록이 풀려(프록시) 같은 번호를 다시 PICK_PLACE로 보낸다."""
        holder = {}

        def finish(sc):
            """첫 번호 있는 송신의 결과를 낸다."""
            rt = holder["rt"]
            tracked = [g for g in rt.sent if g["part_track_id"] is not None]

            if tracked and tracked[0]["goal_id"] in rt.inflight:
                rt.finish(tracked[0]["goal_id"])

        rt = ReleasingRuntime()
        holder["rt"] = rt
        sc = Scenario(10, during={6: finish})
        code, err, out, rt = self.run_track(sc, runtime=rt)
        self.assertEqual(code, 0, err)
        tracked = [g for g in rt.sent if g["part_track_id"] is not None]
        self.assertEqual(len(tracked), 2, rt.sent)
        self.assertEqual(len({g["part_track_id"] for g in tracked}), 1)
        self.assertIn(tracked[0]["part_track_id"], rt.released)

    def test_not_accepted_releases_pick_place_only(self):
        """T7: 검사 ③ 거부가 PICK_PLACE면 release(다음 수락이 다시 PICK_PLACE), ALREADY_SENT 거부는 release 없음."""
        from eyetoarm.vision.live_observation_dispatch import LiveObservationDispatcher

        # 첫 번호 있는 결과(PICK_PLACE)를 거부 → release → 다음 번호 있는 수락은 PICK_PLACE(송신)
        calls = {"first": None}
        real = LiveObservationDispatcher._process

        def process_once(disp, item):
            """번호 있는 첫 계산만 거부."""
            if not hasattr(disp.state, "_orig_accept"):
                disp.state._orig_accept = disp.state.accept

                def accept(cid, token, payload):
                    goal = payload.output.goal

                    if goal.get("part_track_id") is not None and calls["first"] is None:
                        calls["first"] = goal["kind"]
                        return None

                    return disp.state._orig_accept(cid, token, payload)

                disp.state.accept = accept

            return real(disp, item)

        with mock.patch.object(LiveObservationDispatcher, "_process", process_once):
            code, err, out, rt = self.run_track(Scenario(8))

        self.assertEqual(code, 0, err)
        self.assertEqual(calls["first"], "PICK_PLACE")
        srec = self.sort_records(out)
        self.assertIn("release:not_accepted", [r["event"] for r in srec])
        acc = [r for r in self.accepts(out) if r["input"]["part_track_id"] is not None]
        self.assertEqual(acc[0]["goal_kind"], "PICK_PLACE")        # 풀렸으므로 다음 수락이 다시 PICK_PLACE
        self.assertEqual(len([g for g in rt.sent if g["part_track_id"] is not None]), 1)

    def test_already_sent_not_accepted_not_released(self):
        """T7: ALREADY_SENT(SKIP) 결과가 검사 ③에서 거부돼도 release하지 않는다(그 뒤도 ALREADY_SENT)."""
        from eyetoarm.vision.live_observation_dispatch import LiveObservationDispatcher

        seen = {"skip_rejected": 0}
        real = LiveObservationDispatcher._process

        def process(disp, item):
            """ALREADY_SENT 첫 결과만 거부."""
            if not hasattr(disp.state, "_orig_accept"):
                disp.state._orig_accept = disp.state.accept

                def accept(cid, token, payload):
                    goal = payload.output.goal

                    if goal.get("reason") == "ALREADY_SENT" and seen["skip_rejected"] == 0:
                        seen["skip_rejected"] = 1
                        return None

                    return disp.state._orig_accept(cid, token, payload)

                disp.state.accept = accept

            return real(disp, item)

        with mock.patch.object(LiveObservationDispatcher, "_process", process):
            code, err, out, rt = self.run_track(Scenario(9))

        self.assertEqual(code, 0, err)
        self.assertEqual(seen["skip_rejected"], 1)
        self.assertNotIn("release:not_accepted", [r["event"] for r in self.sort_records(out)])
        self.assertEqual(len([g for g in rt.sent if g["part_track_id"] is not None]), 1)

    def test_no_server_releases_each_time(self):
        """T7: 서버 없음이면 B 클라이언트가 매번 release → ALREADY_SENT 없이 매 수락 no_server."""
        code, err, out, rt = self.run_track(Scenario(7), runtime=ReleasingRuntime(server=False))
        self.assertEqual(code, 0, err)
        acc = [r for r in self.accepts(out) if r["input"]["part_track_id"] is not None]
        self.assertTrue(acc and all(r["goal_kind"] == "PICK_PLACE" for r in acc))
        self.assertTrue(rt.released)

    def test_record_only_tracking_releases_after_accept(self):
        """T7: 송신을 끈 기록 전용 + 추적이면 수락 뒤 바로 release(ALREADY_SENT로 굳지 않음), 기록에 번호."""
        code, err, out = self.run_sort(Scenario(7), config=CONFIG, extra=("--tracking", *PIXEL))
        self.assertEqual(code, 0, err)
        acc = [r for r in self.accepts(out) if r["input"]["part_track_id"] is not None]
        self.assertTrue(acc and all(r["goal_kind"] == "PICK_PLACE" for r in acc))
        self.assertGreaterEqual(self.summary(out)["tracking_link"]["counts"]["released_record_only"], len(acc))
        self.assertFalse((Path(out) / "sort_part.jsonl").exists())

    def test_profile_change_resets_ids(self):
        """T5: 프로필 변경(세대 변경) 뒤 이전 번호가 실리지 않고 새 번호로 다시 확정된다."""
        from test_ros_image_adapter import info_msg
        from test_run_pipeline_capture_observation import H, W, FRAME_ID

        def change(s):
            s.src.handle_camera_info(info_msg(fx=310.0, width=W, height=H, frame_id=FRAME_ID))

        code, err, out, rt = self.run_track(Scenario(12, during={5: change}), extra=("--tracking", "--live-sort-part", *PIXEL))
        self.assertEqual(code, 0, err)
        acc = self.accepts(out)
        cal = [r["input"]["calibration_id"] for r in acc]
        ids = [r["input"]["part_track_id"] for r in acc]
        first_cal = cal[0]
        before = {i for i, c in zip(ids, cal) if c == first_cal and i is not None}
        after = [i for i, c in zip(ids, cal) if c != first_cal]
        self.assertTrue(before and after, (ids, cal))
        self.assertFalse(set(a for a in after if a is not None) & before)      # 이전 번호 없음
        self.assertIsNone(after[0])                                            # 새 트랙은 다시 3 hit 필요

    def test_view_bias_keeps_track(self):
        """T4: 시야 보정 켬에서도 B 입력 하위 형·번호 유지, goal 번호 = 기록 번호."""
        self.case.ai = write_bias_config(self.case.ai, Path(self.tmp.name) / "ai_vb.json")
        code, err, out, rt = self.run_track(Scenario(6))
        self.assertEqual(code, 0, err)
        acc = self.accepts(out)
        self.assertTrue(all(r["input"]["view_bias"]["applied"] for r in acc))
        self.assertTrue(any(r["goal"]["part_track_id"] is not None for r in acc))
        self.assertTrue(all(t == "TrackedPlanarObservation" for t, *_ in self.b_seen))

    def test_retry_limit_end_to_end(self):
        """T9: 실패 서버 흉내 + retry → 같은 번호 실패 1회 뒤 skip:track_retry_limit, 시작 줄·요약 표기."""
        holder = {}

        def fail_first(sc):
            rt = holder["rt"]

            for g in list(rt.sent):
                if g["goal_id"] in rt.inflight:
                    rt.finish(g["goal_id"], "FAILED", "FAKE_FAIL")

        rt = ReleasingRuntime()
        holder["rt"] = rt
        retry = {"max_failures_per_track": 1, "failed_point_block_s": 0.001, "failed_point_block_mm": 0.001}
        code, err, out, rt = self.run_track(Scenario(10, during={i: fail_first for i in range(4, 10)}), runtime=rt,
                                            retry=retry)
        self.assertEqual(code, 0, err)
        self.assertIn("retry=max_failures_per_track=1", err)
        events = [r["event"] for r in self.sort_records(out)]
        self.assertIn("skip:track_retry_limit", events)
        tracked = [g for g in rt.sent if g["part_track_id"] is not None]
        self.assertEqual(len(tracked), 1)
        self.assertNotEqual(self.summary(out)["sort_part"]["retry"], "off")

    def test_tracking_failure_isolated(self):
        """T12: 추적 예외 → 실행기 latch(추적 끔), 이후 번호 None(no_tracks)으로 송신 규칙 계속, 종료 0."""
        with mock.patch.object(trk.PartTracker, "update", side_effect=RuntimeError("tracker boom")):
            code, err, out, rt = self.run_track(Scenario(4))

        self.assertEqual(code, 0, err)
        acc = self.accepts(out)
        self.assertTrue(acc and all(r["input"]["part_track_id"] is None for r in acc))
        self.assertTrue(all(r["input"]["track_state"] == NO_TRACKS for r in acc))
        self.assertTrue(rt.sent)
        self.assertIn("tracking_failed", err)

    def test_on_output_still_refused_with_tracking(self):
        """T11: 추적을 켜도 on_output 있는 B 단계는 거부(수락 0·송신 0·종료 1)."""
        def hook(stage):
            stage.on_output = lambda out: None

        rt = ReleasingRuntime()
        code, err, out = self.run_sort(Scenario(3), config=sort_config(self.tmp.name), factory=self.factory(rt),
                                       extra=("--live-sort-part", "--tracking", *PIXEL), hook=hook)
        self.assertEqual(code, 1)
        self.assertEqual((self.accepts(out), rt.sent), ([], []))


class TrackingOffTest(TrackCli):
    """T1·T2: 조합과 끔 동작."""

    def test_off_offer_has_no_tracks_and_records_unchanged(self):
        """T2: 추적 끔이면 offer에 tracks 인자 없음, accept input·sort_part 형식·요약이 #214와 같다."""
        from eyetoarm.vision.live_observation_link import LiveObservationLink

        kwargs = []
        real = LiveObservationLink.offer

        def spy(link, outcome, reason, epoch, **kw):
            kwargs.append(kw)
            return real(link, outcome, reason, epoch, **kw)

        with mock.patch.object(LiveObservationLink, "offer", spy):
            code, err, out, rt = self.run_track(Scenario(4), extra=("--live-sort-part",))

        self.assertEqual(code, 0, err)
        self.assertTrue(kwargs and all(kw == {} for kw in kwargs))
        for r in self.accepts(out):
            self.assertEqual(sorted(r["input"]), sorted(["run_id", "image_seq", "capture_generation", "ros_stamp_ns",
                                                         "calibration_id", "job_seq", "generation", "axis_endpoints_px"]))
            self.assertEqual(r["b_input"]["extensions"], {})
        self.assertFalse(any("part_track_id" in r for r in self.sort_records(out)))
        s = self.summary(out)
        self.assertNotIn("tracking_link", s)
        self.assertNotIn("tracking", s)
        self.assertTrue(all(t == "PlanarObservation" for t, *_ in self.b_seen))
        self.assertIn("tracking=off", err)

    def test_on_offer_gets_tracks(self):
        """T1: --live-observation-out --tracking 허용, offer에 tracks 인자(TrackFrame 또는 None)."""
        from eyetoarm.vision.live_observation_link import LiveObservationLink

        kwargs = []
        real = LiveObservationLink.offer

        def spy(link, outcome, reason, epoch, **kw):
            kwargs.append(kw)
            return real(link, outcome, reason, epoch, **kw)

        with mock.patch.object(LiveObservationLink, "offer", spy):
            code, err, out, rt = self.run_track(Scenario(4))

        self.assertEqual(code, 0, err)
        self.assertTrue(kwargs and all(set(kw) == {"tracks"} for kw in kwargs))
        self.assertTrue(any(isinstance(kw["tracks"], trk.TrackFrame) for kw in kwargs))

    def test_list_with_tracking_refused(self):
        """T1: 단계 L + 추적은 2."""
        code, err, out = self.run_sort(Scenario(2), config=CONFIG, extra=("--tracking", "--b-input", "list"))
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
