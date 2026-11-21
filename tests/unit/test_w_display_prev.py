"""#242 detect 영상 W 번호 표시 보완 시험: 같은 작업 결과가 없을 때 직전 W 결과를 위치(IoU 1:1)로 짝지어 화면에만 그린다.

근거: 지시 `_workspace/242_reviewer_instruction_w_display_prev.md`(A 결정 「권장으로 수정」, #240 F1). ROS·Gazebo는 쓰지 않는다.
"""
from __future__ import annotations

import dataclasses
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from eyetoarm.vision import world_tracking_display as wtd
from eyetoarm.vision.handoff import AToCHandoff, annotate
from eyetoarm.vision.live_world_tracking import LagConfigError, LiveWorldTracking, load_world_tracking_config
from eyetoarm.vision.tracking import FrameKey, TrackAssignment, TrackFrame, TrackMode, TrackState
from test_ai_result import instance, seg_result
from test_live_world_tracking import WORLD_CONFIG, WorldCli
from test_run_pipeline_live_observation import Scenario
from test_tracking_outputs import Core
from test_tracking_send_fixes import TRACK_SORT_CONFIG, put_texts
from test_world_tracking import LiveBase, ctx_of, live_cfg

CFG = wtd.DisplayPrevConfig(max_age_ms=300.0, min_iou=0.3, ambiguity_margin=0.1)
BOX_A = (100.0, 100.0, 140.0, 160.0)
BOX_B = (300.0, 100.0, 340.0, 160.0)


def shifted(box, dx=4.0, dy=0.0):
    """박스를 픽셀만큼 옮긴다(벨트 이동 모사)."""
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


def ctx(i, *, generation=0, ts=None, run="r"):
    """이번 작업 문맥(촬영 시각 기본 i×25 ms)."""
    f = SimpleNamespace(run_id=run, source_id="s", image_seq=i, capture_generation=0, timestamp_ms=i * 25 if ts is None else ts)

    return SimpleNamespace(frame=f, generation=generation, job_seq=i, detector_config_id="d")


def snap(i, confirmed, *, generation=0, ts=None):
    """직전 사본: confirmed = [(index, 번호, 박스)] (클래스 bolt, hits 3)."""
    key = FrameKey(run_id="r", source_id="s", image_seq=i, capture_generation=0, generation=generation, job_seq=i,
                   detector_config_id="d")

    return wtd.PrevSnapshot(key=key, timestamp_ms=i * 25 if ts is None else ts, tracker_epoch=1,
                            confirmed=tuple((idx, tid, 3, "bolt", box) for idx, tid, box in confirmed))


def display(*boxes, cls=None):
    """이번 AI 결과: 박스 순서대로 인스턴스 index 0, 1, …"""
    cls = cls or ["bolt"] * len(boxes)

    return seg_result([instance(i, box_xyxy=b, class_name=c) for i, (b, c) in enumerate(zip(boxes, cls))], crop_wh=(1280, 720))


def ids_of(shown):
    """표시 판정의 인스턴스 index → 번호(짝 없음 None)."""
    return {a.instance_index: a.part_track_id for a in shown.tracks.assignments}


# ---------------------------------------------------------------------- 짝짓기 규칙

class MatchPrevTest(LiveBase):
    """`match_prev`: 위치 1:1·확정 번호만·조건 불만족은 미표시."""

    def test_prev_within_limit_matches_by_position(self):
        """상한 안의 직전 결과: 옮겨 간 박스에 그 번호, 표시 판정은 이번 작업 키·WORLD·사유 display_prev."""
        shown, why = wtd.match_prev(snap(9, [(0, 7, BOX_A)]), ctx(10), display(shifted(BOX_A)), CFG, reset_pending=False)

        self.assertEqual(why, "prev")
        self.assertEqual(ids_of(shown), {0: 7})
        self.assertEqual((shown.prev_image_seq, shown.age_ms, shown.matched), (9, 25.0, 1))
        self.assertEqual((shown.tracks.image_seq, shown.tracks.mode, shown.tracks.assignments[0].reason),
                         (10, TrackMode.WORLD, wtd.PREV_REASON))

    def test_instance_order_swapped(self):
        """이번 프레임 인스턴스 순서가 뒤바뀌어도 번호는 위치를 따라간다(index 0 ↔ 1)."""
        prev = snap(9, [(0, 1, BOX_A), (1, 2, BOX_B)])
        shown, why = wtd.match_prev(prev, ctx(10), display(shifted(BOX_B), shifted(BOX_A)), CFG, reset_pending=False)

        self.assertEqual(why, "prev")
        self.assertEqual(ids_of(shown), {0: 2, 1: 1})

    def test_ambiguous_and_no_overlap_not_drawn(self):
        """1·2위 IoU 차가 작으면(같은 자리 두 박스) 그 번호는 그리지 않는다. 겹침 없음은 no_match."""
        twin, why = wtd.match_prev(snap(9, [(0, 1, BOX_A)]), ctx(10), display(shifted(BOX_A, 1), shifted(BOX_A, 2)), CFG,
                                   reset_pending=False)
        self.assertEqual((twin, why), (None, "no_match"))

        far, why = wtd.match_prev(snap(9, [(0, 1, BOX_A)]), ctx(10), display(BOX_B), CFG, reset_pending=False)
        self.assertEqual((far, why), (None, "no_match"))

    def test_partial_match_leaves_others_untracked(self):
        """한 인스턴스만 짝이 맞으면 그것만 번호, 나머지는 untracked(display_prev_unmatched)."""
        shown, _ = wtd.match_prev(snap(9, [(0, 1, BOX_A)]), ctx(10), display(shifted(BOX_A), BOX_B), CFG, reset_pending=False)

        self.assertEqual(ids_of(shown), {0: 1, 1: None})
        self.assertEqual(shown.tracks.assignments[1].state, TrackState.UNTRACKED)
        self.assertEqual(shown.tracks.assignments[1].reason, wtd.PREV_UNMATCHED)

    def test_conditions_rejected(self):
        """상한 초과·세대 변경·실행 변경·재설정 대기·더 새 결과·확정 없음·클래스 다름·사본 없음은 번호 없음."""
        cur = display(shifted(BOX_A))
        cases = [(snap(9, [(0, 1, BOX_A)], ts=0), ctx(10, ts=301), False, "too_old"),
                 (snap(9, [(0, 1, BOX_A)]), ctx(10, generation=1), False, "generation_changed"),
                 (snap(9, [(0, 1, BOX_A)]), ctx(10, run="other"), False, "scope_changed"),
                 (snap(9, [(0, 1, BOX_A)]), ctx(10), True, "reset_pending"),
                 (snap(9, [(0, 1, BOX_A)], ts=250), ctx(10, ts=250), False, "not_older"),
                 (snap(9, []), ctx(10), False, "no_confirmed"),
                 (None, ctx(10), False, "no_prev")]

        for prev, c, reset, why in cases:
            with self.subTest(why=why):
                self.assertEqual(wtd.match_prev(prev, c, cur, CFG, reset_pending=reset), (None, why))

        nut = display(shifted(BOX_A), cls=["nut"])
        self.assertEqual(wtd.match_prev(snap(9, [(0, 1, BOX_A)]), ctx(10), nut, CFG, reset_pending=False), (None, "no_match"))

    def test_age_limit_inclusive(self):
        """촬영 시각 차가 상한과 같으면 쓴다."""
        shown, why = wtd.match_prev(snap(9, [(0, 1, BOX_A)], ts=0), ctx(10, ts=300), display(shifted(BOX_A)), CFG,
                                    reset_pending=False)
        self.assertEqual(why, "prev")

    def test_snapshot_keeps_confirmed_only(self):
        """사본은 갱신한 결과의 확정 번호 인스턴스만(박스·클래스 함께). 갱신 안 한 결과는 None."""
        d = display(BOX_A, BOX_B)
        tf = TrackFrame(run_id="r", source_id="s", image_seq=3, capture_generation=0, generation=0, job_seq=3,
                        detector_config_id="d", timestamp_ms=75, mode=TrackMode.WORLD, track_namespace="n", tracker_epoch=2,
                        updated=True, event=None,
                        assignments=(TrackAssignment(0, 4, TrackState.CONFIRMED, 5, None),
                                     TrackAssignment(1, None, TrackState.TENTATIVE, 1, None)))
        s = wtd.snapshot(tf, d)

        self.assertEqual(s.confirmed, ((0, 4, 5, "bolt", BOX_A),))
        self.assertEqual((s.timestamp_ms, s.tracker_epoch, s.key.image_seq), (75, 2, 3))
        self.assertIsNone(wtd.snapshot(dataclasses.replace(tf, updated=False), d))

    def test_config_parse(self):
        """설정: 정확한 세 키, 범위 밖·bool·누락은 오류. 예시 파일은 읽힌다(300 ms, IoU 0.3, 차 0.1)."""
        good = {"max_age_ms": 300, "min_iou": 0.3, "ambiguity_margin": 0.1}
        self.assertEqual(wtd.parse_display_prev(good), CFG)

        for bad in ({**good, "max_age_ms": 0}, {**good, "min_iou": 0}, {**good, "min_iou": 1.5},
                    {**good, "ambiguity_margin": 1.0}, {**good, "max_age_ms": True}, {**good, "x": 1},
                    {k: v for k, v in good.items() if k != "min_iou"}):
            with self.subTest(bad=bad), self.assertRaises(wtd.DisplayPrevConfigError):
                wtd.parse_display_prev(bad)

        self.assertEqual(load_world_tracking_config(str(WORLD_CONFIG))[1].display_prev, CFG)

    def test_config_block_required_in_file(self):
        """설정 파일 live 블록에 display_prev가 없으면 설정 오류(기본값 없음)."""
        doc = json.loads(WORLD_CONFIG.read_text(encoding="utf-8"))
        del doc["live"]["display_prev"]
        p = Path(self.make_tmp()) / "w.json"
        p.write_text(json.dumps(doc), encoding="utf-8")

        with self.assertRaises(LagConfigError):
            load_world_tracking_config(str(p))

    def make_tmp(self):
        """임시 폴더(정리 등록)."""
        import tempfile

        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)

        return t.name


# ---------------------------------------------------------------------- 추적 스레드 peek

class PeekPrevTest(LiveBase):
    """`LiveWorldTracking.peek(context, display)`: 같은 작업 우선, 없으면 직전 짝, 재설정 뒤 무효. 송신 쪽 `wait`는 그대로."""

    def confirm(self, lw, n=3):
        """볼트 하나를 n장 넣고 마지막 결과를 기다린다(확정)."""
        for i in range(n):
            self.offer(i)

        return lw.wait(ctx_of(n - 1), timeout_s=2.0)

    def disp(self):
        """이번 프레임 AI 결과(시험 도구 볼트와 같은 박스)."""
        from test_world_tracking import result_of

        return result_of(self.lookup.T, [(0.50, 0.0, 0.040, 90.0, 0.9, False)])

    def test_same_then_prev_then_reset(self):
        """같은 작업이면 TrackFrame, 처리 안 된 다음 작업은 직전 짝(PrevDisplay), 재설정 요청 뒤에는 None(reset_pending)."""
        lw = self.make(display_prev=CFG)
        last = self.confirm(lw)
        self.assertTrue(any(a.state is TrackState.CONFIRMED for a in last.assignments))

        same = lw.peek(ctx_of(2), self.disp())
        self.assertIsInstance(same, TrackFrame)

        before = (lw.counts["handoff_ok"], lw.counts["handoff_pending"], list(lw._handoff_window))
        prev = lw.peek(ctx_of(3), self.disp())
        self.assertIsInstance(prev, wtd.PrevDisplay)
        self.assertEqual({a.part_track_id for a in prev.tracks.assignments}, {a.part_track_id for a in last.assignments})
        self.assertEqual((lw.counts["handoff_ok"], lw.counts["handoff_pending"], list(lw._handoff_window)), before)

        self.assertIsNone(lw.wait(ctx_of(3), timeout_s=0.05))   # 송신 쪽은 그 작업 결과만(직전 결과 안 씀)

        lw.request_reset("boundary:test")
        self.assertIsNone(lw.peek(ctx_of(3), self.disp()))

        # 요청이 아직 처리 전이면 reset_pending, 추적 스레드가 이미 처리했으면 사본이 지워져 no_prev(어느 쪽이든 번호 없음)
        s = lw.summary()["display_lookup"]
        self.assertEqual((s["found"], s["prev"], s["pending"]), (1, 1, 1))
        self.assertIn(s["prev_miss"], ({"reset_pending": 1}, {"no_prev": 1}))

    def test_reset_clears_snapshot(self):
        """재설정을 처리하면 사본을 지운다: 그 뒤 첫 프레임은 확정 번호가 없어 직전 짝 없음(no_confirmed)."""
        lw = self.make(display_prev=CFG)
        self.confirm(lw)
        lw.request_reset("boundary:test")
        self.offer(3)
        lw.wait(ctx_of(3), timeout_s=2.0)

        self.assertIsNone(lw.peek(ctx_of(4), self.disp()))
        self.assertEqual(lw.display_prev_miss, {"no_confirmed": 1})

    def test_off_without_config(self):
        """설정 없음(None)이면 직전 짝을 쓰지 않는다(사유 off)."""
        lw = self.make()
        self.confirm(lw)

        self.assertIsNone(lw.peek(ctx_of(3), self.disp()))
        self.assertEqual(lw.display_prev_miss, {"off": 1})


# ---------------------------------------------------------------------- 그리기

class AnnotatePrevTest(LiveBase):
    """`annotate`: PrevDisplay는 번호 + `(world, prev)` 배너, 같은 작업은 `(world)`."""

    def view_with(self, make_value):
        """조회 함수 `make_value(ctx, display)`를 둔 연결부의 영상 칸."""
        c = Core()
        h = AToCHandoff()
        h.set_track_source(make_value)
        h.start_run(c.p)
        out, tracks = c.result(0)
        h.offer_result(out)

        return h.take_view()[0], tracks

    def test_prev_banner_and_ids(self):
        """직전 짝: `#5`와 `TRACKING EXPERIMENTAL (world, prev)` 배너."""
        def source(c, d):
            """그 작업 직전(25 ms 전) 사본을 같은 박스로 만든다."""
            f = c.frame
            key = FrameKey(run_id=f.run_id, source_id=f.source_id, image_seq=f.image_seq - 1,
                           capture_generation=f.capture_generation, generation=c.generation, job_seq=c.job_seq - 1,
                           detector_config_id=c.detector_config_id)
            prev = wtd.PrevSnapshot(key=key, timestamp_ms=f.timestamp_ms - 25, tracker_epoch=1,
                                    confirmed=((0, 5, 3, d.instances[0].class_name, tuple(d.instances[0].box_xyxy)),))

            return wtd.match_prev(prev, c, d, CFG, reset_pending=False)[0]

        view, _ = self.view_with(source)
        _, seen = put_texts(lambda: annotate(view))
        joined = " | ".join(seen)

        self.assertIn("#5", joined)
        self.assertIn("TRACKING EXPERIMENTAL (world, prev) - ids display-only", joined)

    def test_same_job_still_world(self):
        """같은 작업 결과(TrackFrame)는 지금처럼 `(world)` 배너."""
        box = {}
        view, tracks = self.view_with(lambda c, d: box.get("v"))
        box["v"] = dataclasses.replace(tracks, mode=TrackMode.WORLD)
        _, seen = put_texts(lambda: annotate(view))
        joined = " | ".join(seen)

        self.assertIn("TRACKING EXPERIMENTAL (world) - ids display-only", joined)
        self.assertNotIn("prev", joined)


# ---------------------------------------------------------------------- 송신 경로 격리

class SendIsolationTest(WorldCli):
    """직전 결과는 화면 전용: 표시 판정이 엉뚱한 번호를 내도 송신·B 기록에는 나타나지 않는다."""

    def test_prev_ids_never_reach_send_or_b(self):
        """match_prev가 늘 번호 999 표시 판정을 내게 하고 C 소비자처럼 매 영상을 그려도, sort_part·B 수락에 999가 없다."""
        drawn = []

        def fake_match(prev, context, disp, config, *, reset_pending):
            """모든 인스턴스에 999를 붙인 표시 판정."""
            f = context.frame
            tracks = TrackFrame(run_id=f.run_id, source_id=f.source_id, image_seq=f.image_seq,
                                capture_generation=f.capture_generation, generation=context.generation, job_seq=context.job_seq,
                                detector_config_id=context.detector_config_id, timestamp_ms=int(f.timestamp_ms),
                                mode=TrackMode.WORLD, track_namespace="display_prev", tracker_epoch=1, updated=False,
                                event=wtd.PREV_REASON,
                                assignments=tuple(TrackAssignment(int(i.index), 999, TrackState.CONFIRMED, 3, wtd.PREV_REASON)
                                                  for i in disp.instances))

            return wtd.PrevDisplay(tracks=tracks, prev_image_seq=0, age_ms=1.0, matched=len(disp.instances)), "prev"

        class DrawingProbe:
            """C 소비자 대역: 알림마다 제어 메시지를 비우고 영상 칸을 꺼내 annotate한다(별도 스레드 없이 알림 자리에서).

            제어 메시지를 비워야 연결부가 다음 알림을 다시 부른다(알림 대기 latch).
            """

            def __init__(self):
                """연결부는 start에서 받는다."""
                self.h = None
                self.lock = threading.Lock()

            def notify(self):
                """영상 칸이 있으면 그린다."""
                if self.h is None:
                    return

                with self.lock:
                    try:
                        self.h.take_control(64)
                        view, _ = self.h.take_view()
                    except Exception:   # 닫힌 뒤 알림: 그릴 것 없음(시험 대역)
                        return

                if view is not None:
                    _, seen = put_texts(lambda: annotate(view))
                    drawn.append(" | ".join(seen))

            def start(self, handoff):
                """연결부 기억."""
                self.h = handoff

            def stop(self):
                """정지 자리."""

        def factory():
            """실제 연결부 + 그리는 probe."""
            probe = DrawingProbe()

            return AToCHandoff(notify=probe.notify), probe

        extra = ("--tracking", "--world-tracker-config", str(WORLD_CONFIG), "--live-sort-part")

        with mock.patch.object(self.mod, "_default_live_handoff", factory), \
                mock.patch("eyetoarm.vision.live_world_tracking.match_prev", fake_match):
            code, err, out, rt = self.run_world(Scenario(8), extra=extra, config=str(TRACK_SORT_CONFIG))

        self.assertEqual(code, 0, err)
        self.assertTrue(any("#999" in t for t in drawn), drawn[:3])
        live = Path(out)

        def track_ids(node):
            """기록 안의 모든 `part_track_id` 값(중첩 포함)."""
            if isinstance(node, dict):
                for k, v in node.items():
                    if k == "part_track_id":
                        yield v

                    yield from track_ids(v)
            elif isinstance(node, list):
                for v in node:
                    yield from track_ids(v)

        for name in ("sort_part.jsonl", "b_live_events.jsonl"):
            recs = [json.loads(t) for t in (live / name).read_text(encoding="utf-8").splitlines()]
            seen = set(track_ids(recs))
            self.assertTrue(seen, name)
            self.assertNotIn(999, seen, name)

        self.assertFalse(any(g.get("part_track_id") == 999 for g in rt.sent))
