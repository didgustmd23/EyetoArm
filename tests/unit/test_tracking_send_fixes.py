"""#239 추적 기반 송신 보완 시험: 확정 번호 대기 송신(D4)·실시간 기본 W 추적·W 번호 detect 표시.

근거: 계획 `_workspace/239_builder2_tracking_send_fixes_plan_r0.md`(reviewer PASS: Q1 거부 2, Q2 PICK_PLACE만, Q3 실시간 기본 W +
W 설정 명시, Q4 start_run 전 setter, Q5 기다리지 않음·pending 계수, Q6 기록만). ROS·Gazebo는 쓰지 않는다(가짜 TF·런타임·모델).
"""
from __future__ import annotations

import contextlib
import dataclasses
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from eyetoarm.vision import live_sort_part as spc
from eyetoarm.vision.handoff import AToCHandoff, HandoffContractError, annotate
from eyetoarm.vision.live_view import draw_detections
from eyetoarm.vision.tracking import TrackMode
from test_live_sort_part import FakeRuntime
from test_live_tracking_link import PIXEL, ReleasingRuntime, TrackCli, sort_config
from test_live_world_tracking import WORLD_CONFIG, WorldCli
from test_run_pipeline_live_observation import CONFIG, Scenario
from test_tracking_outputs import Core
from test_world_tracking import LiveBase, ctx_of

ROOT = Path(__file__).resolve().parents[2]
TRACK_SORT_CONFIG = ROOT / "config" / "vision" / "live_observation_sort_part_track_test.json"


def put_texts(fn):
    """fn() 동안 live_view가 cv2.putText로 그린 문자열 목록과 fn 반환값."""
    seen = []
    real = cv2.putText

    def spy(img, text, *a, **kw):
        """그린 글자를 남긴다."""
        seen.append(text)
        return real(img, text, *a, **kw)

    with mock.patch("eyetoarm.vision.live_view.cv2.putText", side_effect=spy):
        out = fn()

    return out, seen


# ---------------------------------------------------------------------- 1. 설정 해석

class RequireConfigTest(TrackCli):
    """`sort_part.require_track_id` 해석과 진입점 짝 검사(Q1)."""

    def test_parse_bool_only(self):
        """bool만 받는다: true·false는 그대로, 키 없음은 None, 1·0·문자열은 설정 오류."""
        base = json.loads(TRACK_SORT_CONFIG.read_text(encoding="utf-8"))["sort_part"]
        self.assertIs(spc.parse_sort_part(base).require_track_id, True)
        self.assertIs(spc.parse_sort_part({**base, "require_track_id": False}).require_track_id, False)
        self.assertIsNone(spc.parse_sort_part({k: v for k, v in base.items() if k != "require_track_id"}).require_track_id)

        for bad in (1, 0, "true", None):
            with self.subTest(value=bad), self.assertRaises(spc.SortPartConfigError):
                spc.parse_sort_part({**base, "require_track_id": bad})

    def test_to_json_only_when_set(self):
        """요약 config에는 설정했을 때만 키가 들어간다(없으면 기존 형식 그대로)."""
        base = json.loads(TRACK_SORT_CONFIG.read_text(encoding="utf-8"))["sort_part"]
        unset = spc.parse_sort_part({k: v for k, v in base.items() if k != "require_track_id"})
        self.assertNotIn("require_track_id", unset.to_json())
        self.assertIs(spc.parse_sort_part(base).to_json()["require_track_id"], True)

    def test_tracking_without_key_is_2(self):
        """추적 켬 + 송신 + 키 없음(기존 송신 설정)은 설정 오류 2."""
        from test_live_sort_part import SORT_CONFIG

        code, err, out, rt = self.run_track(Scenario(3), config=str(SORT_CONFIG))
        self.assertEqual(code, 2, err)
        self.assertIn("sort_part.require_track_id is required with --tracking", err)
        self.assertEqual(rt.sent, [])

    def test_true_without_tracking_or_stage_l_is_2(self):
        """번호가 없는 실행(추적 끔·단계 L)의 true는 거부 2(볼트 송신 0 설정을 조용히 받지 않음)."""
        for i, extra in enumerate((("--live-sort-part",), ("--live-sort-part", "--b-input", "list"))):
            with self.subTest(extra=extra):
                code, err, out = self.run_sort(Scenario(3), config=TRACK_SORT_CONFIG, extra=extra, out=self.out(f"t{i}"))
                self.assertEqual(code, 2, err)
                self.assertIn("require_track_id=true needs --tracking stage T", err)

    def test_false_or_unset_without_tracking_runs(self):
        """추적 끔 + false·키 없음은 지금처럼 실행(0)."""
        from test_live_sort_part import SORT_CONFIG

        for i, cfg in enumerate((sort_config(self.tmp.name, require=False), str(SORT_CONFIG))):
            with self.subTest(config=cfg):
                code, err, out = self.run_sort(Scenario(3), config=cfg, out=self.out(f"off{i}"))
                self.assertEqual(code, 0, err)


# ---------------------------------------------------------------------- 1. 송신

class SendOnlyConfirmedTest(TrackCli):
    """확정 번호 대기 송신(D4): 번호 None PICK_PLACE는 보내지 않고 기록한다. B 기록은 남지 않는다."""

    def setUp(self):
        """B 단계 수집."""
        super().setUp()
        self.stages = []

    def run_req(self, scenario, require):
        """픽셀 추적 + 송신, require_track_id 지정."""
        cfg = sort_config(self.tmp.name, require=require)
        rt = ReleasingRuntime()
        code, err, out = self.run_sort(scenario, config=cfg, extra=("--live-sort-part", "--tracking", *PIXEL),
                                       factory=self.factory(rt), hook=self.stages.append)

        return code, err, out, rt

    def test_none_track_pick_place_not_sent(self):
        """true: 확정 전 프레임은 send 0·skip:no_track_id(track_state 함께), 확정 뒤 번호 있는 목표만 보낸다. B `_pending`에 None 없음."""
        code, err, out, rt = self.run_req(Scenario(9), True)
        self.assertEqual(code, 0, err)

        recs = self.sort_records(out)
        skips = [r for r in recs if r["event"] == "skip:no_track_id"]
        sent = [r for r in recs if r["event"] == "sent"]

        self.assertTrue(skips, recs)
        self.assertTrue(all(r["part_track_id"] is None and r["kind"] == "PICK_PLACE" and r["track_state"] for r in skips))
        self.assertTrue(sent and all(r["part_track_id"] is not None for r in sent))
        self.assertTrue(all(g.get("part_track_id") is not None for g in rt.sent if g.get("kind") == "PICK_PLACE"))
        self.assertLess(recs.index(skips[0]), recs.index(sent[0]))   # 번호 None 먼저, 확정 뒤 송신(D4의 반대)
        self.assertFalse(any(r["event"].startswith("release:") and r["part_track_id"] is None for r in recs))
        self.assertTrue(all(None not in st._pending for st in self.stages))
        self.assertTrue(all("track_state" not in r for r in recs if r["event"] != "skip:no_track_id"))

        s = self.summary(out)["sort_part"]
        self.assertIs(s["require_track_id"], True)
        self.assertEqual(sum(s["no_track_id_states"].values()), len(skips))
        self.assertEqual(s["events"]["skip:no_track_id"], len(skips))
        self.assertIn("require_track_id=true: PICK_PLACE without a confirmed part_track_id is not sent", err)

    def test_false_keeps_none_track_sends(self):
        """false(기존 동작): 번호 None PICK_PLACE도 보내고 skip:no_track_id는 없다. 요약에 no_track_id_states 빈 사전."""
        code, err, out, rt = self.run_req(Scenario(9), False)
        self.assertEqual(code, 0, err)

        recs = self.sort_records(out)
        self.assertFalse(any(r["event"] == "skip:no_track_id" for r in recs))
        self.assertTrue(any(r["event"] == "sent" and r["part_track_id"] is None for r in recs))
        self.assertEqual(self.summary(out)["sort_part"]["no_track_id_states"], {})

    def test_pass_through_not_blocked(self):
        """PASS_THROUGH(너트)는 번호 None이어도 그대로 보낸다(Q2: PICK_PLACE만)."""
        cfg = spc.parse_sort_part(json.loads(TRACK_SORT_CONFIG.read_text(encoding="utf-8"))["sort_part"])
        rt = FakeRuntime()
        sender = spc.LiveSortPartSender("r", cfg, rt, self.tmp.name, tracking=True)
        frame = SimpleNamespace(image_seq=1, ros_stamp_ns=10)
        res = SimpleNamespace(valid=True)
        nut = SimpleNamespace(goal={"kind": "PASS_THROUGH", "goal_id": "g1", "part_track_id": None}, result=res)
        bolt = SimpleNamespace(goal={"kind": "PICK_PLACE", "goal_id": "g2", "part_track_id": None}, result=res)

        self.assertEqual(sender.offer(nut, frame, {"job_seq": 1, "generation": 1, "track_state": None}, 1), "sent")
        self.assertEqual(sender.offer(bolt, frame, {"job_seq": 2, "generation": 1, "track_state": "tentative"}, 2),
                         "skip:no_track_id")
        self.assertEqual([g["goal_id"] for g in rt.sent], ["g1"])
        self.assertEqual(sender.summary()["no_track_id_states"], {"tentative": 1})
        sender.close_run("test")
        sender.close(2.0)


# ---------------------------------------------------------------------- 2. 기본 공간

class DefaultSpaceTest(WorldCli):
    """실시간 --tracking 기본 공간은 W, W 설정은 명시 필수(Q3). 저장 입력 기본은 픽셀 그대로."""

    def test_live_default_world_with_config(self):
        """공간 생략 + W 설정: W 추적(world_tracks 있음), 시작 줄·요약에 기본 출처, 번호 None PICK_PLACE 송신 0."""
        extra = ("--tracking", "--world-tracker-config", str(WORLD_CONFIG), "--live-sort-part")
        code, err, out, rt = self.run_world(Scenario(8), extra=extra, config=str(TRACK_SORT_CONFIG))

        self.assertEqual(code, 0, err)
        self.assertTrue((Path(out) / "world_tracks.jsonl").exists())
        self.assertIn("space=world(default for live;", err)
        self.assertNotIn("tracking ON M2(heuristic)", err)
        s = self.summary(out)
        self.assertEqual(s["tracking_space"], {"space": "world", "source": "default"})
        self.assertTrue(all(g.get("part_track_id") is not None for g in rt.sent if g.get("kind") == "PICK_PLACE"))

    def test_live_default_without_config_is_2_with_hint(self):
        """공간 생략 + W 설정 없음: 인자 오류 2, '--world-tracker-config 필요 또는 --tracking-space pixel' 안내."""
        with contextlib.redirect_stderr(io.StringIO()) as argerr:   # argparse 오류는 sys.stderr로 간다
            code, err, out, rt = self.run_world(Scenario(3), extra=("--tracking", "--live-sort-part"))

        err = argerr.getvalue()

        self.assertEqual(code, 2)
        self.assertIn("--world-tracker-config JSON is required", err)
        self.assertIn("or give --tracking-space pixel", err)

    def test_explicit_sources_recorded(self):
        """명시 world·pixel은 요약 출처 explicit."""
        for extra, space in (((*("--tracking", "--tracking-space", "world", "--world-tracker-config", str(WORLD_CONFIG)),
                               "--live-sort-part"), "world"),
                             (("--tracking", *PIXEL, "--live-sort-part"), "pixel")):
            with self.subTest(space=space):
                code, err, out, rt = self.run_world(Scenario(4), extra=extra, out=self.out(f"src_{space}"))
                self.assertEqual(code, 0, err)
                self.assertEqual(self.summary(out)["tracking_space"], {"space": space, "source": "explicit"})

    def test_stage_l_tracking_refusal_unchanged(self):
        """단계 L + --tracking 거부 문구는 기본 공간 결정보다 먼저(지금과 같음)."""
        with contextlib.redirect_stderr(io.StringIO()) as argerr:
            code, err, out, rt = self.run_world(Scenario(2), extra=("--tracking", "--b-input", "list"), config=str(CONFIG))

        err = argerr.getvalue()

        self.assertEqual(code, 2)
        self.assertIn("--live-observation-out --tracking is for stage T only", err)

    def test_stored_input_default_pixel(self):
        """저장 입력 경로의 공간 생략은 픽셀(인자 해석만, W 설정 없이 통과)."""
        args = self.mod.build_parser().parse_args(["--images", "a.png", "--log", "x.jsonl", "--detector", "ai", "--tracking"])
        self.mod._check_source_args(args, None)

        self.assertEqual((args.tracking_space, args.tracking_space_source), ("pixel", "default"))


# ---------------------------------------------------------------------- 3. W 번호 표시

class PeekTest(LiveBase):
    """`LiveWorldTracking.peek`: 기다리지 않음, 그 작업 결과만, 밀림 규칙 계수 무변."""

    def test_peek_does_not_touch_handoff_counts(self):
        """peek은 handoff_ok·pending·대기 창을 바꾸지 않고 display_counts만 센다. 없는 작업은 None(pending)."""
        lw = self.make()
        self.offer(0)
        self.assertIsNotNone(lw.wait(ctx_of(0), timeout_s=2.0))

        before = (lw.counts["handoff_ok"], lw.counts["handoff_pending"], list(lw._handoff_window))
        found = lw.peek(ctx_of(0))
        missing = lw.peek(ctx_of(99))

        self.assertIsNotNone(found)
        self.assertIsNone(missing)
        self.assertEqual((lw.counts["handoff_ok"], lw.counts["handoff_pending"], list(lw._handoff_window)), before)
        # #242: prev 계수·pending 사유가 더해졌다(이 시험 설정은 display_prev 없음 → 사유 off)
        self.assertEqual(lw.display_counts, {"found": 1, "prev": 0, "pending": 1, "reason": 0})
        self.assertEqual(lw.summary()["display_lookup"], {"found": 1, "prev": 0, "pending": 1, "reason": 0,
                                                          "prev_miss": {"off": 1}})


class WorldAnnotateTest(TrackCli):
    """handoff `track_source`·`annotate` W 표시."""

    def view(self, source, *, tracks=False):
        """연결부(조회 함수 설정)에서 꺼낸 영상 칸과 그 작업의 TrackFrame(W 모드로 바꾼 것)."""
        c = Core()
        h = AToCHandoff()
        h.set_track_source(source)
        h.start_run(c.p)
        out, frame_tracks = c.result(0)
        h.offer_result(out, **({"tracks": frame_tracks} if tracks else {}))

        return h, h.take_view()[0], dataclasses.replace(frame_tracks, mode=TrackMode.WORLD)

    def test_found_draws_world_ids(self):
        """같은 작업의 W 결과가 있으면 번호 라벨 + (world) 실험 배너."""
        box = {}
        h, view, wtracks = self.view(lambda ctx, display: box.get("v"))
        box["v"] = wtracks
        _, seen = put_texts(lambda: annotate(view))
        joined = " | ".join(seen)

        self.assertIn("#1", joined)
        self.assertIn("TRACKING EXPERIMENTAL (world) - ids display-only", joined)

    def test_pending_and_reason_banners(self):
        """결과 없음은 pending 배너, 사유 문자열은 사유 배너. 번호 라벨은 없다."""
        for value, text in ((None, "ids pending"), ("tf_timeout", "ids unavailable (tf_timeout)")):
            with self.subTest(value=value):
                h, view, _ = self.view(lambda ctx, display, v=value: v)
                _, seen = put_texts(lambda: annotate(view))
                joined = " | ".join(seen)

                self.assertIn(f"TRACKING EXPERIMENTAL (world) - {text}", joined)
                self.assertNotIn("#1", joined)

    def test_other_job_tracks_not_used(self):
        """다른 작업의 TrackFrame은 쓰지 않는다(현재 프레임 결과만): job_mismatch 배너, 번호 없음."""
        c = Core()
        _, other = c.result(5)
        other = dataclasses.replace(other, mode=TrackMode.WORLD)
        h, view, _ = self.view(lambda ctx, display: other)
        _, seen = put_texts(lambda: annotate(view))
        joined = " | ".join(seen)

        self.assertIn("ids unavailable (job_mismatch)", joined)
        self.assertNotIn("#1", joined)

    def test_no_source_unchanged_and_pixel_tracks_win(self):
        """조회 함수가 없으면 그림이 기존과 같다. 실행기 tracks(픽셀)가 있으면 조회를 부르지 않는다."""
        c = Core()
        h = AToCHandoff()
        h.start_run(c.p)
        out, _ = c.result(0)
        h.offer_result(out)
        plain = h.take_view()[0]
        self.assertIsNone(plain.track_lookup)
        np.testing.assert_array_equal(annotate(plain), draw_detections(plain.frame.image, plain.display)[0])

        calls = []
        h2, view, _ = self.view(lambda ctx, display: calls.append(ctx), tracks=True)
        _, seen = put_texts(lambda: annotate(view))

        self.assertEqual(calls, [])
        self.assertIsNone(view.track_lookup)
        self.assertIn("heuristic", " | ".join(seen))

    def test_setter_only_before_start_run(self):
        """start_run 뒤 set_track_source는 계약 오류. 호출 불가 값도 오류."""
        c = Core()
        h = AToCHandoff()

        with self.assertRaises(HandoffContractError):
            h.set_track_source(42)

        h.start_run(c.p)

        with self.assertRaises(HandoffContractError):
            h.set_track_source(lambda ctx, display: None)


class WorldHandoffWiringTest(WorldCli):
    """실시간 W 실행이 C 연결부에 W 조회를 연결한다(시작 전)."""

    def test_world_run_sets_peek(self):
        """W 기본 실행: 연결부의 조회 함수가 그 실행의 `peek`이고 요약에 조회 계수가 있다. 픽셀 실행은 연결하지 않는다."""
        made = []

        class Probe:
            """가짜 C probe(네트워크 없음)."""

            def notify(self):
                """알림 자리."""

            def start(self, handoff):
                """시작 자리."""

            def stop(self):
                """정지 자리."""

        def factory():
            """실제 연결부 + 가짜 probe."""
            probe = Probe()
            h = AToCHandoff(notify=probe.notify)
            made.append(h)

            return h, probe

        cases = ((("--tracking", "--world-tracker-config", str(WORLD_CONFIG), "--live-sort-part"), True),
                 (("--tracking", *PIXEL, "--live-sort-part"), False))

        for extra, world in cases:
            with self.subTest(world=world), mock.patch.object(self.mod, "_default_live_handoff", factory):
                code, err, out, rt = self.run_world(Scenario(6), extra=extra, config=str(TRACK_SORT_CONFIG),
                                                    out=self.out(f"wire_{world}"))
                self.assertEqual(code, 0, err)
                h = made[-1]
                self.assertEqual(h._track_source is not None, world)

                if world:
                    self.assertEqual(getattr(h._track_source, "__name__", None), "peek")
                    self.assertIn("display_lookup", self.summary(out)["world_tracking"])
