"""추적 1차 출력 시험 — JSONL `ai_tracks`, 확인 창 표시, A→C handoff `tracks` 전파와 같은 작업 짝 유지.

가짜 소스·가짜 검출만 쓴다. 번호는 표시 실험 값이며 신원 정확도는 확인하지 않는다.
"""
import io
import unittest

import numpy as np

from eyetoarm.common.types import InputStatus
from eyetoarm.vision.handoff import AToCHandoff, HandoffContractError, MsgKind, annotate
from eyetoarm.vision.live_view import STATE_COLORS_BGR, CLASS_COLORS_BGR, DisplayBridge, _info_lines, draw_detections, \
    render_view, track_label
from eyetoarm.vision.pipeline import VisionPipeline
from eyetoarm.vision.pipeline_records import JsonlRecorder, PipelineLog
from eyetoarm.vision.tracking import FrameKey, PartTracker, PartTrackIdIssuer, TrackMode, TrackState, inputs_from_ai
from test_ai_config import ai_config
from test_ai_pipeline import ai_observation
from test_ai_result import seg_result
from test_live_view import big_instance
from test_pipeline import RUN_ID, FakeClock, frame, settings
from test_pipeline_runner_handoff import stored_factory
from test_pipeline_runner_tracking import run_stored, tracker
from test_run_pipeline import FakeSource, event

H, W = 200, 240


def image(value=40):
    """합성 원본."""
    return np.full((H, W, 3), value, np.uint8)


def key_of(job):
    """작업의 FrameKey."""
    f = job.frame
    return FrameKey(run_id=f.run_id, source_id=f.source_id, image_seq=f.image_seq, capture_generation=f.capture_generation,
                    generation=job.generation, job_seq=job.job_seq, detector_config_id=job.detector_config_id)


class Core:
    """실제 코어 + 추적기로 공개 결과·TrackFrame 쌍을 만드는 도구."""

    def __init__(self, min_hits=1):
        """AI 코어·M2 추적기(min_hits 1이면 첫 프레임부터 번호)."""
        from eyetoarm.vision.tracking import TrackerConfig

        self.p = VisionPipeline(settings(detector_config=ai_config(), profile=None), run_id=RUN_ID, clock_ns=FakeClock())
        self.trk = PartTracker(PartTrackIdIssuer(namespace="ns-out"), mode=TrackMode.HEURISTIC,
                               config=TrackerConfig(min_hits=min_hits))

    def result(self, image_seq):
        """작업 하나 완료 → (응답, TrackFrame)."""
        assert self.p.offer_frame(frame(image_seq, self.p.current_generation(), image=image(), width=W, height=H))
        job = self.p.start_next()
        display = seg_result((big_instance(),), config_sha256=job.detector_config_id,
                             model_id=job.settings.detector_config.model_id, crop_wh=(W, H))
        out = self.p.complete(job, ai_observation(job), display_result=display)

        return out, self.trk.update(key_of(job), job.frame.timestamp_ms, inputs_from_ai(display))


class LabelTest(unittest.TestCase):
    """표시 라벨·정보 줄."""

    def test_track_label(self):
        """confirmed → #ID, tentative → trk?, ambiguous → amb, untracked → 빈 문자열."""
        from eyetoarm.vision.tracking import TrackAssignment

        self.assertEqual(track_label(TrackAssignment(0, 7, TrackState.CONFIRMED, 3, None)), "#7")
        self.assertEqual(track_label(TrackAssignment(0, None, TrackState.TENTATIVE, 1, None)), "trk?")
        self.assertEqual(track_label(TrackAssignment(0, None, TrackState.AMBIGUOUS, 0, "ambiguous")), "amb")
        self.assertEqual(track_label(TrackAssignment(0, None, TrackState.UNTRACKED, 0, "not_in_scope")), "")

    def test_draw_with_tracks_changes_overlay_only(self):
        """번호가 있으면 오버레이에 라벨이 더해진다. 원본 불변, 초록 우세 색 없음(색표 그대로)."""
        c = Core()
        out, tracks = c.result(0)
        src = out.context.frame.image
        before = src.copy()
        plain, _ = draw_detections(src, out.display_result)
        labeled, drawn = draw_detections(src, out.display_result, tracks)

        np.testing.assert_array_equal(src, before)
        self.assertEqual(drawn, 1)
        self.assertFalse(np.array_equal(plain, labeled))
        for b, g, r in list(CLASS_COLORS_BGR) + list(STATE_COLORS_BGR.values()):
            self.assertFalse(g > b and g > r)

    def test_draw_rejects_other_job_tracks(self):
        """다른 결과의 인스턴스 수·index와 맞지 않는 TrackFrame은 그리지 않고 오류."""
        c = Core()
        out, _ = c.result(0)
        _, other = c.result(1)
        bad = seg_result((big_instance(0), big_instance(1)), crop_wh=(W, H))

        with self.assertRaises(ValueError):
            draw_detections(out.context.frame.image, bad, other)


class BridgeTest(unittest.TestCase):
    """DisplayBridge: 같은 작업의 TrackFrame만 칸에 짝지어 둔다."""

    def test_tracks_paired_in_item_and_info(self):
        """current 결과의 tracks가 같은 칸 항목에 들어가고 정보 줄에 EXPERIMENTAL 표기."""
        c, bridge = Core(), DisplayBridge()
        out, tracks = c.result(0)
        bridge.on_outcome(out, True, tracks=tracks)
        snap = bridge.snapshot()

        self.assertIs(snap.item.tracks, tracks)
        self.assertEqual(snap.item.tracks.job_seq, snap.item.job_seq)
        text = " ".join(_info_lines(snap))
        self.assertIn("TRACKING EXPERIMENTAL", text)
        self.assertIn("no identity guarantee", text)
        render_view(snap)

    def test_mismatched_tracks_rejected(self):
        """다른 작업의 TrackFrame을 붙이면 ValueError(칸 무변)."""
        c, bridge = Core(), DisplayBridge()
        out0, t0 = c.result(0)
        out1, t1 = c.result(1)
        bridge.on_outcome(out0, True, tracks=t0)
        version = bridge.version

        with self.assertRaises(ValueError):
            bridge.on_outcome(out1, True, tracks=t0)

        self.assertEqual(bridge.version, version)

    def test_no_tracks_default_unchanged(self):
        """tracks 없이 부르면 추적 없는 출력 그대로(tracks None, 정보 줄에 추적 표기 없음)."""
        c, bridge = Core(), DisplayBridge()
        out, _ = c.result(0)
        bridge.on_outcome(out, True)
        snap = bridge.snapshot()

        self.assertIsNone(snap.item.tracks)
        self.assertNotIn("TRACKING", " ".join(_info_lines(snap)))


class HandoffTracksTest(unittest.TestCase):
    """handoff RESULT·영상 칸에 같은 TrackFrame."""

    def test_result_and_view_share_tracks(self):
        """offer_result(tracks=)는 RESULT·영상 칸에 같은 TrackFrame을 싣고 annotate가 라벨을 그린다."""
        c = Core()
        h = AToCHandoff()
        h.start_run(c.p)
        out, tracks = c.result(0)
        seq = h.offer_result(out, tracks=tracks)
        msg = h.take_control(10)[0][-1]
        view, _ = h.take_view()

        self.assertIs(msg.tracks, tracks)
        self.assertIs(view.tracks, tracks)
        self.assertTrue(h.is_current(seq))
        self.assertFalse(np.array_equal(annotate(view), draw_detections(view.frame.image, view.display)[0]))

    def test_mismatched_tracks_contract_error(self):
        """다른 작업의 TrackFrame이면 계약 오류이고 계수 무변."""
        c = Core()
        h = AToCHandoff()
        h.start_run(c.p)
        _, t0 = c.result(0)
        out1, _ = c.result(1)
        before = h.stats()

        with self.assertRaises(HandoffContractError):
            h.offer_result(out1, tracks=t0)

        self.assertEqual(h.stats(), before)

    def test_default_tracks_none(self):
        """tracks 없이 넣으면 RESULT·영상 칸의 tracks는 None."""
        c = Core()
        h = AToCHandoff()
        h.start_run(c.p)
        out, _ = c.result(0)
        h.offer_result(out)

        self.assertIsNone(h.take_control(10)[0][-1].tracks)
        self.assertIsNone(h.take_view()[0].tracks)


class AnnotateMarkingTest(unittest.TestCase):
    """번호가 실린 annotate 영상에도 실험 표기·모드·신원 보장 없음을 그린다. 추적 끔 영상은 그대로."""

    def texts(self, fn):
        """fn() 동안 cv2.putText로 그린 문자열 목록."""
        import cv2
        from unittest import mock

        seen = []
        real = cv2.putText

        def spy(img, text, *a, **kw):
            """그린 글자를 남긴다."""
            seen.append(text)
            return real(img, text, *a, **kw)

        with mock.patch("eyetoarm.vision.live_view.cv2.putText", side_effect=spy):
            out = fn()

        return out, seen

    def view(self, with_tracks):
        """연결부에서 꺼낸 영상 칸 항목."""
        c = Core()
        h = AToCHandoff()
        h.start_run(c.p)
        out, tracks = c.result(0)
        h.offer_result(out, **({"tracks": tracks} if with_tracks else {}))

        return h.take_view()[0]

    def test_annotate_with_tracks_marks_experimental(self):
        """tracks가 있는 annotate: 번호 라벨과 EXPERIMENTAL·모드·no identity guarantee 문구, 원본 불변."""
        view = self.view(True)
        before = view.frame.image.copy()
        _, seen = self.texts(lambda: annotate(view))
        joined = " | ".join(seen)

        self.assertIn("#1", joined)
        self.assertIn("EXPERIMENTAL", joined)
        self.assertIn("heuristic", joined)
        self.assertIn("no identity guarantee", joined)
        np.testing.assert_array_equal(view.frame.image, before)

    def test_annotate_without_tracks_unchanged(self):
        """tracks 없는 annotate는 draw_detections 결과와 같고 실험 문구가 없다."""
        view = self.view(False)
        out, seen = self.texts(lambda: annotate(view))

        np.testing.assert_array_equal(out, draw_detections(view.frame.image, view.display)[0])
        self.assertFalse(any("EXPERIMENTAL" in t for t in seen))


class RunnerOutputTest(unittest.TestCase):
    """실행기 결합: ai_tracks 기록·표시·handoff가 같은 작업 TrackFrame, 경계 뒤 옛 번호 없음, 실패 뒤 번호 없음."""

    def run_all(self, factory, trk):
        """연결부·표시 칸과 함께 실행. (결과, 기록, stderr, 메시지, 브리지)."""
        h, bridge = AToCHandoff(capacity=1000), _Recording()
        res, recs, err = run_stored(factory, trk=trk, handoff_sink=h, display_sink=bridge)

        return res, recs, err, h.take_control(10_000)[0], bridge

    def test_ai_tracks_lines_after_process(self):
        """갱신마다 ai_tracks 1줄(그 작업 process 뒤), 필드(mode·namespace·epoch·인스턴스별 번호/상태/사유)."""
        res, recs, err, _, _ = self.run_all(stored_factory(4, {}), tracker())

        self.assertEqual(res.exit_code, 0, err)
        kinds = [(r["kind"], r.get("job_seq")) for r in recs if r["kind"] in ("process", "ai_tracks")]
        self.assertEqual(kinds, [(k, j) for j in range(4) for k in ("process", "ai_tracks")])
        t = [r for r in recs if r["kind"] == "ai_tracks"]
        self.assertEqual([r["tracks"][0]["part_track_id"] for r in t], [None, None, 1, 1])
        self.assertEqual((t[0]["mode"], t[0]["track_namespace"], t[0]["updated"]), ("heuristic", "ns-run", True))
        self.assertEqual(set(t[0]["tracks"][0]), {"instance_index", "part_track_id", "state", "hits", "reason"})

    def test_same_trackframe_everywhere_and_boundary(self):
        """RESULT·표시 칸·기록이 같은 작업의 TrackFrame. 저장 입력 오류 경계 뒤에는 옛 번호 1이 다시 나오지 않고 epoch가 오른다."""
        def factory(issuer, provider):
            """프레임 4 → 디코딩 실패 → 프레임 4."""
            items = [frame(issuer.next_seq(), provider()) for _ in range(4)] + [event(InputStatus.DECODE_FAILED)]
            items += [frame(issuer.next_seq(), provider()) for _ in range(4)]
            return FakeSource(items)

        res, recs, err, msgs, bridge = self.run_all(factory, tracker())
        results = [m for m in msgs if m.msg_kind is MsgKind.RESULT]

        self.assertTrue(all(m.tracks is not None and m.tracks.job_seq == m.job_seq and m.tracks.image_seq == m.image_seq
                            for m in results))
        ids = [m.tracks.assignments[0].part_track_id for m in results]
        self.assertEqual(ids, [None, None, 1, 1, None, None, 2, 2])
        self.assertGreater(results[4].tracks.tracker_epoch, results[3].tracks.tracker_epoch)
        self.assertTrue(all(t.job_seq == j for j, t in bridge.tracks))
        logged = [r["tracks"][0]["part_track_id"] for r in recs if r["kind"] == "ai_tracks"]
        self.assertEqual(logged, ids)

    def test_tracking_failure_then_no_ids_anywhere(self):
        """추적 실패 뒤에는 RESULT·표시 tracks가 None, ai_tracks 줄도 없다(이전 확정 번호를 싣지 않음)."""
        res, recs, err, msgs, bridge = self.run_all(stored_factory(5, {}), tracker(fail_at=4))
        results = [m for m in msgs if m.msg_kind is MsgKind.RESULT]

        self.assertEqual(res.exit_code, 0, err)
        self.assertEqual([m.tracks is None for m in results], [False, False, False, True, True])
        self.assertEqual([t is None for _, t in bridge.tracks], [False, False, False, True, True])
        self.assertEqual(len([r for r in recs if r["kind"] == "ai_tracks"]), 3)

    def test_default_off_no_new_fields(self):
        """추적 끔: ai_tracks 줄 없음, RESULT tracks None, 표시 칸은 tracks 키워드 없이 호출."""
        h, bridge = AToCHandoff(capacity=1000), _Recording()
        res, recs, err = run_stored(stored_factory(2, {}), handoff_sink=h, display_sink=bridge)
        msgs = h.take_control(100)[0]

        self.assertEqual(res.exit_code, 0, err)
        self.assertFalse(any(r["kind"] == "ai_tracks" for r in recs))
        self.assertTrue(all(m.tracks is None for m in msgs if m.msg_kind is MsgKind.RESULT))
        self.assertEqual(bridge.kw_seen, [set(), set()])

    def test_recorder_failure_on_ai_tracks_is_log_failure(self):
        """ai_tracks 기록 실패는 추적 예외로 삼키지 않고 log_failure(종료 1)."""
        class BrokenLog(PipelineLog):
            """ai_tracks 쓰기 실패."""

            def write_ai_tracks(self, frame_):
                """실패."""
                raise OSError("disk full")

        res, _, err = run_stored(stored_factory(2, {}), trk=tracker(), log=BrokenLog(JsonlRecorder(io.StringIO())))

        self.assertEqual((res.exit_code, res.stop_reason), (1, "log_failure"), err)


class _Recording(DisplayBridge):
    """on_outcome 키워드와 (job_seq, tracks)를 기록하는 브리지."""

    def __init__(self):
        """기록 목록."""
        super().__init__()
        self.tracks, self.kw_seen = [], []

    def on_outcome(self, outcome, current, **kw):
        """기록 뒤 원래 동작."""
        self.kw_seen.append(set(kw))

        if current:
            self.tracks.append((outcome.context.job_seq, kw.get("tracks")))

        super().on_outcome(outcome, current, **kw)


if __name__ == "__main__":
    unittest.main()
