"""추적 스레드 그리퍼 넘기기 시험: 재정렬·재설정·기록·꺼짐 형식."""
from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import gripper_scene as gs
from eyetoarm.vision.live_observation_envelope import TfLookupResult
from eyetoarm.vision.live_world_tracking import GripperLive, LagConfigError, LiveWorldTracking, load_handoff_config
from eyetoarm.vision.observation_envelope import CaptureTf
from eyetoarm.vision.tracking import PartTrackIdIssuer
from eyetoarm.vision.world_tracking import WorldTracker
from test_gripper_depth import gcfg
from test_world_tracking import DIST, K, PLANE_Z, cam, live_cfg, result_of, tcfg
from test_world_tracking_gripper import CONFIG, vis

LINK_T = cam((0.0, 0.0, 0.5))
GRIP_T = gs.camera_above(x=0.50, y=0.0, height_m=0.12)
BOLT = ((0.50, 0.0, 0.040, 90.0, 0.9, False),)


class DualLookup:
    """링크·그리퍼 프레임에 각자의 TF를 주는 대역."""

    def __call__(self, frame):
        """조회."""
        T = GRIP_T if getattr(frame, "optical_frame", None) == "gripper_cam_optical_frame" else LINK_T

        return TfLookupResult(CaptureTf("world", "cam", frame.ros_stamp_ns, T), "ok", 0.1)


def link_ctx(t_ms, seq, generation=0):
    """촬영 시각 t_ms의 링크 작업 대역."""
    f = SimpleNamespace(run_id="r", source_id="s", image_seq=seq, capture_generation=0, ros_stamp_ns=t_ms * 1_000_000,
                        timestamp_ms=t_ms, width=1280, height=720)

    return SimpleNamespace(frame=f, generation=generation, job_seq=seq, detector_config_id="d")


def grip_frame(t_ms, bolts=(gs.Bolt(0.50, -0.017, 90.0, 0.005, 0.006, 0.003, 0.040),)):
    """촬영 시각 t_ms의 그리퍼 깊이 프레임(축 가운데가 링크 트랙 위치에 오게 둔 M6x40)."""
    return gs.render(GRIP_T, list(bolts), stamp_ns=t_ms * 1_000_000)


class Base(unittest.TestCase):
    """스레드 만들기·기록 읽기."""

    def make(self, handoff=True, reorder_wait_ms=5000.0):
        """추적 스레드(정리 등록)."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        gripper = GripperLive(gcfg(reorder_wait_ms=reorder_wait_ms), gs.ROI_XY_M) if handoff else None
        tracker = WorldTracker(PartTrackIdIssuer(), config=tcfg(), visibility=vis() if handoff else None,
                               gripper_gate_mm=(10.0, 15.0) if handoff else None, gripper_late_ms=500 if handoff else None)
        self.lw = LiveWorldTracking("r", tracker, live_cfg(), tf_lookup=DualLookup(), plane_z_m=PLANE_Z, out_dir=self.tmp.name,
                                    visibility=vis() if handoff else None, gripper=gripper)
        self.camera = SimpleNamespace(K=K, dist=DIST)
        self.seq = 0
        self.lw.start()
        self.addCleanup(lambda: self.lw.close(2.0))

        return self.lw

    def link(self, t_ms, bolts=BOLT, wait=True):
        """링크 응답 하나(처리될 때까지 기다림)."""
        self.seq += 1
        ctx = link_ctx(t_ms, self.seq)
        self.lw.offer(ctx, result_of(LINK_T, list(bolts)), self.camera)

        return self.lw.wait(ctx, 2.0) if wait else None

    def settle(self, cond, limit_s=3.0):
        """조건이 참이 될 때까지 기다린다."""
        end = time.monotonic() + limit_s

        while time.monotonic() < end and not cond():
            time.sleep(0.01)

    def records(self):
        """world_tracks.jsonl."""
        self.lw.close(2.0)

        return [json.loads(t) for t in (Path(self.tmp.name) / "world_tracks.jsonl").read_text().splitlines()]


class ReorderTest(Base):
    """시각 순서 재정렬."""

    def test_gripper_waits_for_later_link(self):
        """그리퍼 120이 먼저 와도 링크 100 → 그리퍼 120 → 링크 140 순서, stale 0."""
        self.make()
        self.link(100)
        self.lw.offer_gripper(grip_frame(120))
        time.sleep(0.1)
        self.link(140)
        recs = [r for r in self.records() if r["kind"] in ("frame", "gripper_frame")]
        self.assertEqual([(r["kind"], r["timestamp_ms"]) for r in recs],
                         [("frame", 100), ("gripper_frame", 120), ("frame", 140)])
        self.assertTrue(all(r.get("updated", True) for r in recs))

    def test_link_after_earlier_gripper_not_stale(self):
        """그리퍼(300)를 먼저 반영해도 추적기 시각은 링크만 앞당겨 뒤에 온 링크 250이 갱신된다."""
        lw = self.make(reorder_wait_ms=1.0)

        for t in (0, 66, 132):
            self.link(t)

        lw.offer_gripper(grip_frame(300))
        self.settle(lambda: lw.gripper_counts["processed"] == 1)
        self.assertTrue(self.link(250).updated)
        self.assertEqual(lw.summary()["handoff"]["gripper_counts"]["stale_after_gripper"], 0)

    def test_late_gripper_applied(self):
        """링크 400 뒤에 처리된 그리퍼 350(상한 안)은 늦은 관측으로 반영된다."""
        lw = self.make(reorder_wait_ms=1.0)

        for t in (0, 66, 132):
            self.link(t)

        self.link(400, bolts=())
        lw.offer_gripper(grip_frame(350))
        self.settle(lambda: lw.gripper_counts["processed"] == 1)
        g = [r for r in self.records() if r["kind"] == "gripper_frame"][0]
        self.assertTrue(g["updated"] and g["late"])
        self.assertEqual(g["components"][0]["outcome"], "gripper_hit")

    def test_gripper_without_track_keeps_link_time(self):
        """화면 안 확정 트랙이 없는 그리퍼 프레임은 추적기 시각을 앞당기지 않는다(뒤 링크 stale 없음)."""
        lw = self.make(reorder_wait_ms=1.0)
        self.link(100)
        lw.offer_gripper(grip_frame(200))
        self.settle(lambda: lw.gripper_counts["processed"] == 1)
        self.assertTrue(self.link(150).updated)
        self.assertEqual(lw.summary()["handoff"]["gripper_counts"]["stale_after_gripper"], 0)


class SlowLookup(DualLookup):
    """링크 132 프레임 조회에서 0.4 s 멈춰 추적 스레드를 붙잡는다."""

    def __call__(self, frame):
        """조회."""
        if getattr(frame, "timestamp_ms", None) == 132:
            time.sleep(0.4)

        return super().__call__(frame)


class SlowObserve:
    """관측을 0.5 s 붙잡는 대역(실제 관측 결과는 그대로)."""

    def __init__(self):
        self.calls = 0

    def __call__(self, *args):
        from eyetoarm.vision.gripper_depth import observe
        self.calls += 1
        time.sleep(0.5)
        return observe(*args)


class ObserveThreadTest(Base):
    """그리퍼 관측은 추적 스레드 밖(D3)."""

    def test_slow_observe_does_not_delay_link_results(self):
        """관측이 0.5 s 걸려도 링크 결과 대기(20 ms 안)는 실패하지 않는다."""
        lw = self.make(reorder_wait_ms=1.0)
        slow = SlowObserve()
        lw._observe = slow

        for t in (0, 66, 132):
            self.link(t)

        lw.offer_gripper(grip_frame(150))
        self.settle(lambda: slow.calls == 1)
        waits = []

        for k in range(8):
            ctx = link_ctx(200 + 66 * k, 100 + k)
            lw.offer(ctx, result_of(LINK_T, list(BOLT)), self.camera)
            t0 = time.monotonic()
            got = lw.wait(ctx, 0.02)
            waits.append((got is not None, time.monotonic() - t0))

        self.assertTrue(all(ok for ok, _ in waits), waits)
        self.assertEqual(lw.thread.name, "eyetoarm-live-world-tracking")
        self.settle(lambda: lw.gripper_counts["processed"] == 1)
        self.assertEqual(lw.gripper_counts["observed"], 1)


class InHandOrderTest(Base):
    """같은 반복에서 꺼낸 링크와 대기 만료 그리퍼의 순서."""

    def test_in_hand_earlier_link_not_stale(self):
        """꺼낸 링크 250이 대기 만료 그리퍼 300보다 먼저 처리된다(stale 0)."""
        lw = self.make(reorder_wait_ms=50.0)
        lw._tf_lookup = SlowLookup()

        for t in (0, 66):
            self.link(t)

        self.link(132, wait=False)
        time.sleep(0.05)
        lw.offer_gripper(grip_frame(300))
        self.link(250, wait=False)
        self.settle(lambda: lw.gripper_counts["processed"] == 1 and lw.summary()["tracker"]["updates"] >= 4, 3.0)
        self.assertEqual(lw.summary()["handoff"]["gripper_counts"]["stale_after_gripper"], 0)
        recs = [r for r in self.records() if r["kind"] in ("frame", "gripper_frame")]
        self.assertEqual([r["timestamp_ms"] for r in recs], [0, 66, 132, 250, 300])


class HandoffLiveTest(Base):
    """스레드 안 넘기기·기록."""

    def test_gripper_hit_and_events(self):
        """확정 트랙 위 그리퍼 깊이 → gripper_hit, 사건 camera_switch, 링크 기록에 camera."""
        lw = self.make(reorder_wait_ms=1.0)

        for t in (0, 66, 132):
            tf = self.link(t)

        pid = tf.assignments[0].part_track_id
        self.assertIsNotNone(pid)
        lw.offer_gripper(grip_frame(200))
        self.settle(lambda: lw.gripper_counts["processed"] == 1)
        self.assertEqual(self.link(266).assignments[0].part_track_id, pid)
        recs = self.records()
        g = [r for r in recs if r["kind"] == "gripper_frame"][0]
        self.assertEqual(g["tracks_in_view"], [pid])
        self.assertEqual([(c["outcome"], c["part_track_id"]) for c in g["components"]], [("gripper_hit", pid)])
        events = [(r["from"], r["to"]) for r in recs if r["kind"] == "track_event" and r["event"] == "camera_switch"]
        self.assertEqual(events, [("link", "gripper"), ("gripper", "link")])
        back = [r for r in recs if r["kind"] == "track_event" and r.get("to") == "link"][0]
        self.assertLess(back["link_return_offset_mm"]["dist"], 5.0)
        self.assertTrue(all(r["camera"] == "link" for r in recs if r["kind"] == "frame"))
        summary_counts = lw.summary()["handoff"]["gripper_counts"]
        self.assertEqual((summary_counts["observed"], summary_counts["updated"]), (1, 1))

    def test_no_track_in_view_skips_observer(self):
        """그리퍼 화면에 확정 트랙이 없으면 관측기를 돌리지 않는다."""
        lw = self.make(reorder_wait_ms=1.0)
        self.link(0)
        lw.offer_gripper(grip_frame(50))
        self.settle(lambda: lw.gripper_counts["processed"] == 1)
        self.assertEqual((lw.gripper_counts["no_track_in_view"], lw.gripper_counts["observed"]), (1, 0))

    def test_before_link_and_reset(self):
        """링크 전 그리퍼는 no_link_yet, 재설정은 그리퍼 큐를 비운다."""
        lw = self.make(reorder_wait_ms=1.0)
        lw.offer_gripper(grip_frame(10))
        self.settle(lambda: lw.gripper_counts["processed"] == 1)
        self.assertEqual(lw.gripper_counts["no_link_yet"], 1)
        lw.close(2.0)

        lw = self.make(reorder_wait_ms=60_000.0)
        self.link(100)
        lw.offer_gripper(grip_frame(120))
        lw.request_reset("test")
        self.settle(lambda: lw.last_reset == "test")
        self.link(200)
        self.assertEqual(lw.gripper_counts["processed"], 0)
        self.assertEqual(lw.summary()["handoff"]["gripper_counts"]["queue_left"], 0)

    def test_receiver_thread_offer(self):
        """다른 스레드에서 넣어도 처리된다."""
        lw = self.make(reorder_wait_ms=1.0)
        self.link(0)
        th = threading.Thread(target=lambda: lw.offer_gripper(grip_frame(30)))
        th.start()
        th.join()
        self.settle(lambda: lw.gripper_counts["processed"] == 1)
        self.assertEqual(lw.gripper_counts["processed"], 1)


class OffTest(Base):
    """꺼짐: 기록·요약 형식이 지금과 같음."""

    def test_off_schema(self):
        """꺼짐이면 offer_gripper는 False, 기록에 camera·새 종류가 없고 요약에 handoff가 없다."""
        lw = self.make(handoff=False)

        for t in (0, 66, 132):
            self.link(t)

        self.assertFalse(lw.offer_gripper(grip_frame(150)))
        self.assertNotIn("handoff", lw.summary())
        self.assertNotIn("handoff", lw.summary()["tracker"])
        recs = self.records()
        self.assertEqual({r["kind"] for r in recs}, {"frame", "closed"})
        self.assertFalse(any("camera" in r for r in recs))


class ConfigTest(unittest.TestCase):
    """선택 블록 읽기."""

    def test_load_blocks(self):
        """예시 파일은 두 블록, 기존 파일은 둘 다 None."""
        v, g = load_handoff_config(str(CONFIG))
        self.assertEqual((v.hold_limit_ms, g.max_rate_hz, g.reorder_wait_ms), (30000, 10.0, 250.0))
        self.assertEqual(load_handoff_config(str(CONFIG.parent / "world_tracker_test.json")), (None, None))

    def test_bad_block(self):
        """블록 오류는 LagConfigError."""
        with tempfile.TemporaryDirectory() as d:
            doc = json.loads(CONFIG.read_text(encoding="utf-8"))
            doc["gripper"]["near_m"] = -1
            p = Path(d) / "c.json"
            p.write_text(json.dumps(doc), encoding="utf-8")

            with self.assertRaises(LagConfigError):
                load_handoff_config(str(p))


if __name__ == "__main__":
    unittest.main()
