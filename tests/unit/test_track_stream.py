"""A 추적 흐름 발행 시험(ROS 없음): 자료 변환·주기 상한·큐·발행 실패 격리·설정·추적 스레드 연결·꺼짐 회귀."""
from __future__ import annotations

import json
import math
import tempfile
import threading
import unittest
from pathlib import Path

import gripper_scene as gs
from eyetoarm.vision.live_world_tracking import GripperLive, LiveWorldTracking
from eyetoarm.vision.track_stream import (TrackStream, TrackStreamConfig, TrackStreamConfigError, fold_yaw_deg, frame_message,
                                          load_track_stream_config, marker_message, parse_track_stream, reset_message,
                                          track_entry, track_state)
from eyetoarm.vision.tracking import PartTrackIdIssuer
from eyetoarm.vision.world_tracking import WorldTracker
from test_gripper_depth import gcfg
from test_live_world_tracking_gripper import BOLT, DualLookup, LINK_T, grip_frame, link_ctx
from test_world_tracking import DIST, K, PLANE_Z, SPLIT, live_cfg, result_of, tcfg
from test_world_tracking_gripper import vis

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "vision" / "track_stream_test.json"


def cfg(**over):
    """시험 설정(설정 파일 값 + 덮어쓰기)."""
    doc = {"topic": "/a/track_states", "max_rate_hz": 20.0, "qos_depth": 10, "queue_capacity": 4}
    doc.update(over)

    return TrackStreamConfig(**doc)


def snap(pid=1, x=500.0, y=0.0, vy=20.0, source="TRACK", length=40.0, angle=90.0, last_hit_ms=100, camera="link",
         occluded=False, out_link=False, hold_pred=False):
    """`stream_snapshot` 항목 하나."""
    return {"part_track_id": pid, "x_mm": x, "y_mm": y, "vy_mm_per_s": vy, "velocity_source": source, "length_mm": length,
            "angle_deg": angle, "last_hit_ms": last_hit_ms, "hit_camera": camera, "occluded": occluded,
            "out_link": out_link, "hold_pred": hold_pred}


def frame_msg(capture_ns, tracks=(), **over):
    """프레임 사전(기본값 + 덮어쓰기)."""
    kw = dict(run_id="r", source_id="s", image_seq=1, generation=0, tracker_epoch=1, capture_stamp_ns=capture_ns,
              timestamp_ms=capture_ns // 1_000_000, snapshot=list(tracks), belt_velocity_mm_per_s=20.0, handoff_on=True,
              classes={}, default_plane_z_m=0.105)
    kw.update(over)

    return frame_message(**kw)


class Sink:
    """가짜 발행 함수: 받은 사전을 모은다. `gate`가 있으면 풀릴 때까지 막는다."""

    def __init__(self, gate: threading.Event | None = None, fail: bool = False) -> None:
        self.got: list[dict] = []
        self.gate, self.fail = gate, fail
        self.started = threading.Event()

    def __call__(self, msg: dict) -> None:
        """발행 한 번."""
        self.started.set()

        if self.gate is not None:
            self.gate.wait(5.0)

        if self.fail:
            raise RuntimeError("boom")

        self.got.append(msg)


class ConvertTest(unittest.TestCase):
    """자료 변환(단위·상태·축 접기)."""

    def test_units_and_fields(self):
        """mm·ns·SIM_TIME·base_link, capture_stamp_ns = 프레임 stamp, last_hit_stamp_ns = ms × 1e6, z = 투영 높이 mm."""
        m = frame_msg(1_234_000_000, [snap(last_hit_ms=1234)], classes={1: ("BOLT_B", 0.105)})
        self.assertEqual((m["kind"], m["timestamp_base"], m["frame_id"], m["capture_stamp_ns"]),
                         ("frame", "SIM_TIME", "base_link", 1_234_000_000))
        t = m["tracks"][0]
        self.assertEqual((t["part_track_id"], t["part_class"], t["last_hit_stamp_ns"], t["confirmed"]),
                         ("1", "BOLT_B", 1_234_000_000, True))
        self.assertAlmostEqual(t["position_mm"][2], 105.0)
        self.assertEqual(t["velocity_mm_per_s"], [0.0, 20.0, 0.0])

    def test_unknown_class_and_nan(self):
        """종류를 모르면 빈 문자열·기본 높이, 길이·각도가 없으면 끝점·축 NaN, 공통 속도 None은 NaN."""
        m = frame_msg(100_000_000, [snap(length=None, angle=None)], belt_velocity_mm_per_s=None)
        t = m["tracks"][0]
        self.assertEqual(t["part_class"], "")
        self.assertTrue(math.isnan(t["axis_yaw_deg"]) and math.isnan(t["length_mm"]) and math.isnan(t["endpoint_a_mm"][0]))
        self.assertTrue(math.isnan(m["belt_velocity_mm_per_s"]))

    def test_state_priority(self):
        """GRIPPER > OCCLUDED > PREDICTED > VISIBLE. 이 프레임에 갱신되지 않은 링크 트랙은 PREDICTED."""
        self.assertEqual(track_state(snap(camera="gripper", occluded=True, out_link=True), 100), "GRIPPER")
        self.assertEqual(track_state(snap(occluded=True, out_link=True), 100), "OCCLUDED")
        self.assertEqual(track_state(snap(out_link=True), 100), "PREDICTED")
        self.assertEqual(track_state(snap(hold_pred=True), 100), "PREDICTED")
        self.assertEqual(track_state(snap(last_hit_ms=66), 100), "PREDICTED")
        self.assertEqual(track_state(snap(), 100), "VISIBLE")

    def test_yaw_fold_and_endpoints(self):
        """축 각도 [0, 180) → [-90, 90), 끝점은 접은 축 방향으로 가운데 ± 길이/2."""
        for angle, want in ((0.0, 0.0), (45.0, 45.0), (89.9, 89.9), (90.0, -90.0), (135.0, -45.0), (179.9, -0.1),
                            (-90.0, -90.0), (270.0, -90.0)):
            with self.subTest(angle=angle):
                got = fold_yaw_deg(angle)
                self.assertAlmostEqual(got, want, places=9)
                self.assertTrue(-90.0 <= got < 90.0)
        t = track_entry(snap(x=500.0, y=10.0, length=40.0, angle=135.0), 100, "BOLT_B", 0.105)
        self.assertAlmostEqual(t["axis_yaw_deg"], -45.0)
        a, b = t["endpoint_a_mm"], t["endpoint_b_mm"]
        self.assertAlmostEqual(math.dist(a[:2], b[:2]), 40.0)
        self.assertAlmostEqual((a[0] + b[0]) / 2, 500.0)
        self.assertAlmostEqual(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])), -45.0)

    def test_reset_and_marker(self):
        """재설정·끝 알림은 트랙 없음, 사유는 재설정에만."""
        r = reset_message(run_id="r", source_id="s", reason="scope_changed", tracker_epoch=2, handoff_on=False)
        m = marker_message(run_id="r", source_id="s", tracker_epoch=2, handoff_on=False)
        self.assertEqual((r["kind"], r["reset_reason"], r["tracks"], r["tracker_epoch"]), ("reset", "scope_changed", [], 2))
        self.assertEqual((m["kind"], m["reset_reason"], m["tracks"]), ("marker", "", []))


class StreamTest(unittest.TestCase):
    """발행 큐·스레드."""

    def make(self, sink, now_ns=lambda: 5_000_000_000, **over):
        """발행기(정리 등록)."""
        s = TrackStream(cfg(**over), sink, now_ns)
        s.start()
        self.addCleanup(lambda: s.close(2.0))

        return s

    def test_rate_limit_keeps_reset_marker(self):
        """20 Hz 상한: 50 ms 안의 frame은 건너뛰고 reset·marker는 보존, reset 뒤 첫 frame은 낸다."""
        sink = Sink()
        s = self.make(sink, queue_capacity=16)
        out = [s.offer(frame_msg(t * 1_000_000)) for t in (0, 25, 50, 75)]
        out.append(s.offer(reset_message(run_id="r", source_id="s", reason="x", tracker_epoch=2, handoff_on=True)))
        out.append(s.offer(frame_msg(80_000_000)))
        out.append(s.offer(marker_message(run_id="r", source_id="s", tracker_epoch=2, handoff_on=True)))
        self.assertTrue(s.close(2.0))
        self.assertEqual(out, ["queued", "dropped_rate", "queued", "dropped_rate", "queued", "queued", "queued"])
        self.assertEqual([m["kind"] for m in sink.got], ["frame", "frame", "reset", "frame", "marker"])
        self.assertEqual(s.summary()["counts"]["dropped_rate"], 2)

    def test_queue_full_drops_oldest_frame(self):
        """발행이 막혀 큐가 차면 가장 오래된 frame부터 버리고 reset은 남긴다."""
        gate = threading.Event()
        sink = Sink(gate)
        s = self.make(sink, queue_capacity=2, max_rate_hz=1000.0)
        s.offer(frame_msg(0))                 # 발행 스레드가 잡고 막힘
        self.assertTrue(sink.started.wait(2.0))
        s.offer(reset_message(run_id="r", source_id="s", reason="x", tracker_epoch=2, handoff_on=True))
        s.offer(frame_msg(10_000_000, image_seq=2))
        s.offer(frame_msg(20_000_000, image_seq=3))   # 가득: frame seq 2를 버림
        gate.set()
        self.assertTrue(s.close(2.0))
        self.assertEqual([(m["kind"], m["image_seq"]) for m in sink.got], [("frame", 1), ("reset", 0), ("frame", 3)])
        self.assertEqual(s.summary()["counts"]["dropped_queue"], 1)

    def test_publish_error_isolated(self):
        """발행 예외는 계수·첫 오류로만 남고 offer는 계속 된다."""
        s = self.make(Sink(fail=True))
        self.assertEqual(s.offer(frame_msg(0)), "queued")
        self.assertTrue(s.close(2.0))
        summ = s.summary()
        self.assertEqual((summ["counts"]["publish_errors"], summ["counts"]["published"]), (1, 0))
        self.assertIn("RuntimeError: boom", summ["first_error"])
        self.assertEqual(s.offer(frame_msg(100_000_000)), "dropped_closed")

    def test_publish_stamp_and_latency(self):
        """발행 시각을 채우고 촬영 → 발행 SIM 지연을 잰다."""
        sink = Sink()
        s = self.make(sink, now_ns=lambda: 1_030_000_000)
        s.offer(frame_msg(1_000_000_000))
        self.assertTrue(s.close(2.0))
        self.assertEqual(sink.got[0]["publish_stamp_ns"], 1_030_000_000)
        self.assertEqual(s.summary()["capture_to_publish_sim_ms"]["p50"], 30.0)


class ConfigTest(unittest.TestCase):
    """설정 읽기."""

    def test_repo_file(self):
        """저장소 시험 설정을 읽는다."""
        c, sha = load_track_stream_config(str(CONFIG))
        self.assertEqual((c.topic, c.max_rate_hz, c.qos_depth, c.queue_capacity), ("/a/track_states", 20.0, 10, 4))
        self.assertEqual(len(sha), 64)

    def test_rejects(self):
        """모르는 키·status·토픽·범위·bool 거부."""
        base = json.loads(CONFIG.read_text(encoding="utf-8"))

        for over in ({"extra": 1}, {"status": "final"}, {"topic": "a/b"}, {"topic": "/a b"}, {"max_rate_hz": 0},
                     {"max_rate_hz": float("inf")}, {"max_rate_hz": True}, {"qos_depth": 0}, {"queue_capacity": 2.0},
                     {"queue_capacity": True}):
            with self.subTest(over=over), self.assertRaises(TrackStreamConfigError):
                parse_track_stream({**base, **over})

        with self.assertRaises(TrackStreamConfigError):
            parse_track_stream({k: v for k, v in base.items() if k != "topic"})


class LiveTest(unittest.TestCase):
    """추적 스레드 연결."""

    def make(self, *, stream: bool, handoff: bool = True, plane=PLANE_Z):
        """추적 스레드(+ 가짜 발행기)."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        gripper = GripperLive(gcfg(reorder_wait_ms=1.0), gs.ROI_XY_M) if handoff else None
        tracker = WorldTracker(PartTrackIdIssuer(), config=tcfg(), visibility=vis() if handoff else None,
                               gripper_gate_mm=(10.0, 15.0) if handoff else None, gripper_late_ms=500 if handoff else None)
        self.sink = Sink()
        ts = TrackStream(cfg(max_rate_hz=1000.0, queue_capacity=64), self.sink, lambda: 0) if stream else None
        self.lw = LiveWorldTracking("r", tracker, live_cfg(), tf_lookup=DualLookup(), plane_z_m=plane, out_dir=tmp.name,
                                    visibility=vis() if handoff else None, gripper=gripper, stream=ts)
        self.lw.start()
        self.addCleanup(lambda: self.lw.close(2.0))
        self.seq = 0

        return self.lw

    def link(self, t_ms, bolts=BOLT, generation=0):
        """링크 응답 하나(처리까지 기다림)."""
        self.seq += 1
        ctx = link_ctx(t_ms, self.seq, generation)
        self.lw.offer(ctx, result_of(LINK_T, list(bolts)), _Cam)

        return self.lw.wait(ctx, 2.0)

    def records(self):
        """닫고 world_tracks.jsonl."""
        self.lw.close(2.0)

        return [json.loads(t) for t in (Path(self.dir) / "world_tracks.jsonl").read_text().splitlines()]

    def test_confirmed_only_and_marker(self):
        """임시 단계는 트랙 없음, 확정 뒤 그 번호만, 끝에 marker. 기록 frame 줄에 stream 결과."""
        self.make(stream=True)
        tfs = [self.link(t) for t in (0, 66, 132, 198)]
        pid = tfs[-1].assignments[0].part_track_id
        self.assertIsNotNone(pid)
        recs = self.records()
        frames = [m for m in self.sink.got if m["kind"] == "frame"]
        self.assertEqual(frames[0]["tracks"], [])
        self.assertEqual([t["part_track_id"] for t in frames[-1]["tracks"]], [str(pid)])
        self.assertEqual(frames[-1]["tracks"][0]["state"], "VISIBLE")
        self.assertEqual(frames[-1]["capture_stamp_ns"], 198_000_000)
        self.assertTrue(frames[-1]["handoff_on"])
        self.assertEqual(self.sink.got[-1]["kind"], "marker")
        self.assertTrue(all(r["stream"]["result"] == "queued" for r in recs if r["kind"] == "frame"))
        self.assertEqual([r for r in recs if r["kind"] == "closed"][0]["stream"]["offered"], 5)   # frame 4 + marker

    def test_predicted_when_missed_and_class(self):
        """확정 뒤 못 본 프레임은 PREDICTED(가림 판정 없음), 종류별 높이면 종류·높이를 싣는다."""
        self.make(stream=True, handoff=False, plane=SPLIT)

        for t in (0, 66, 132, 198):
            self.link(t)

        self.link(264, bolts=())
        self.records()
        last = [m for m in self.sink.got if m["kind"] == "frame"][-1]
        t = last["tracks"][0]
        self.assertEqual((t["state"], t["part_class"]), ("PREDICTED", "BOLT_B"))
        self.assertAlmostEqual(t["position_mm"][2], 105.0)
        self.assertEqual(t["last_hit_stamp_ns"], 198_000_000)

    def test_gripper_state(self):
        """그리퍼 깊이로 위치가 갱신된 확정 트랙은 GRIPPER, last_hit_camera = gripper."""
        lw = self.make(stream=True)

        for t in (0, 66, 132):
            self.link(t)

        lw.offer_gripper(grip_frame(200))
        end = threading.Event()

        for _ in range(300):
            if lw.gripper_counts["processed"] == 1:
                break

            end.wait(0.01)

        self.link(220, bolts=())
        self.records()
        t = [m for m in self.sink.got if m["kind"] == "frame"][-1]["tracks"][0]
        self.assertEqual((t["state"], t["last_hit_camera"]), ("GRIPPER", "gripper"))

    def test_reset_and_generation(self):
        """재설정 요청과 세대 변화는 reset 알림을 내고 이후 frame은 새 epoch, 이전 번호 없음."""
        lw = self.make(stream=True)

        for t in (0, 66, 132):
            self.link(t)

        lw.request_reset("test")

        for _ in range(300):
            if lw.last_reset == "test":
                break

            threading.Event().wait(0.01)

        self.link(198)
        self.link(264, generation=1)
        self.records()
        kinds = [(m["kind"], m["reset_reason"]) for m in self.sink.got if m["kind"] != "frame"]
        self.assertEqual(kinds, [("reset", "test"), ("reset", "scope_changed"), ("marker", "")])
        after = [m for m in self.sink.got if m["kind"] == "frame"][-1]
        self.assertEqual(after["tracks"], [])
        self.assertEqual(after["generation"], 1)

    def test_off_regression(self):
        """끄면 기록 frame·closed 줄에 stream 키가 없고, 켬과 비교해 stream·시각 키를 빼면 같다. 요약에도 없다."""
        def run(stream):
            self.make(stream=stream)

            for t in (0, 66, 132, 198):
                self.link(t)

            self.link(264, bolts=())
            summary_has = "track_stream" in self.lw.summary()

            return self.records(), summary_has

        off, off_summary = run(False)
        on, on_summary = run(True)
        self.assertFalse(any("stream" in r for r in off))
        self.assertFalse(off_summary)
        self.assertTrue(on_summary)
        strip = lambda rows: [{k: v for k, v in r.items() if k not in ("stream", "t_mono_ns", "lag_ms")} for r in rows]
        self.assertEqual(strip(off), strip(on))


class _Cam:
    """링크 카메라 내부 값."""

    K, dist = K, DIST


if __name__ == "__main__":
    unittest.main()
