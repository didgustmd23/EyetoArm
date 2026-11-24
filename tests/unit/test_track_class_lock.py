"""확정 트랙 종류 고정 시험(#368, D-5 ⓑ): 팔 근처·급감 사진은 종류를 바꾸지 않고, 떨어진 사진 N장 연속이면 바꾼다."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eyetoarm.vision.live_world_tracking import LiveWorldTracking
from eyetoarm.vision.track_class_lock import ClassLockConfig, ClassLockConfigError, TrackClassLock, parse_class_lock
from eyetoarm.vision.track_stream import TrackStream, TrackStreamConfigError, parse_track_stream
from eyetoarm.vision.tracking import PartTrackIdIssuer
from eyetoarm.vision.world_tracking import WorldTracker
from test_live_world_tracking_gripper import DualLookup, LINK_T, link_ctx
from test_track_stream import CONFIG, Sink, _Cam, cfg
from test_world_tracking import SPLIT, live_cfg, result_of, tcfg
from test_world_tracking_gripper import vis

LOCK = ClassLockConfig(switch_frames=3, near_arm_mm=30.0, drop_mm=5.0)


def feed(lock, frames, pid=1):
    """(종류, 길이 mm, 팔 근처) 목록을 차례로 넣고 각 판정을 돌려준다(높이는 종류별 가짜 값)."""
    z = {"BOLT_A": 0.102, "BOLT_B": 0.105}

    return [lock.observe(pid, spec, z[spec], length, near) for spec, length, near in frames]


class LockLogicTest(unittest.TestCase):
    """`TrackClassLock` 판정."""

    def test_first_class_set_immediately(self):
        """처음 받은 종류는 바로 발행 종류가 된다."""
        lock = TrackClassLock(LOCK)
        d = feed(lock, [("BOLT_B", 42.0, True)])
        self.assertEqual(d[0]["decision"], "first")
        self.assertEqual(lock.classes[1], ("BOLT_B", 0.105))

    def test_near_arm_frames_do_not_change(self):
        """팔 근처 사진은 다른 종류가 몇 장 와도 바꾸지 않는다(#354 F-FEED 트랙 1: 42 → 32·30·27 mm)."""
        lock = TrackClassLock(LOCK)
        d = feed(lock, [("BOLT_B", 42.0, False), ("BOLT_B", 41.9, True)] + [("BOLT_A", L, True) for L in (32.2, 30.1, 27.3, 25.0)])
        self.assertEqual([x["decision"] for x in d[2:]], ["kept_near_arm"] * 4)
        self.assertEqual(lock.classes[1][0], "BOLT_B")
        self.assertTrue(all(x["class"] == "BOLT_B" for x in d))

    def test_sudden_drop_frame_does_not_change(self):
        """팔 근처가 아니어도 직전 팔 밖 사진보다 5 mm 넘게 짧은 한 장은 바꾸지 않는다(#354 F-AUTO 트랙 39: 39.99 → 32.32)."""
        lock = TrackClassLock(LOCK)
        d = feed(lock, [("BOLT_B", 40.0, False), ("BOLT_B", 39.99, False), ("BOLT_A", 32.32, False), ("BOLT_B", 41.36, False)])
        self.assertEqual([x["decision"] for x in d], ["first", "same", "kept_drop", "same"])
        self.assertEqual(lock.classes[1][0], "BOLT_B")

    def test_switch_after_n_clean_frames(self):
        """떨어진 사진에서 새 종류가 이어지면 첫 장은 급감으로 빠지고, 다음 3장째에 바뀐다(높이도 함께)."""
        lock = TrackClassLock(LOCK)
        d = feed(lock, [("BOLT_B", 40.0, False)] + [("BOLT_A", 22.0, False)] * 4)
        self.assertEqual([x["decision"] for x in d], ["first", "kept_drop", "pending", "pending", "switched"])
        self.assertEqual([x["streak"] for x in d[2:4]], [1, 2])
        self.assertEqual(lock.classes[1], ("BOLT_A", 0.102))

    def test_switch_without_drop(self):
        """길이가 급감하지 않은 새 종류 사진(경계 근처)은 첫 장부터 세어 3장째에 바뀐다."""
        lock = TrackClassLock(LOCK)
        d = feed(lock, [("BOLT_B", 37.0, False)] + [("BOLT_A", 35.0, False)] * 3)
        self.assertEqual([x["decision"] for x in d], ["first", "pending", "pending", "switched"])

    def test_current_class_breaks_streak(self):
        """새 종류 연속 중 지금 종류의 센 사진이 오면 연속이 끊겨 처음부터 다시 센다."""
        lock = TrackClassLock(LOCK)
        d = feed(lock, [("BOLT_B", 37.0, False), ("BOLT_A", 35.0, False), ("BOLT_A", 35.0, False), ("BOLT_B", 37.0, False),
                        ("BOLT_A", 35.0, False), ("BOLT_A", 35.0, False)])
        self.assertEqual([x["decision"] for x in d[1:]], ["pending", "pending", "same", "pending", "pending"])
        self.assertEqual(lock.classes[1][0], "BOLT_B")

    def test_near_and_drop_frames_neither_count_nor_break(self):
        """팔 근처·급감 사진은 연속을 늘리지도 끊지도 않는다."""
        lock = TrackClassLock(LOCK)
        d = feed(lock, [("BOLT_B", 37.0, False), ("BOLT_A", 35.0, False), ("BOLT_B", 37.0, True), ("BOLT_A", 35.0, True),
                        ("BOLT_A", 35.0, False), ("BOLT_B", 20.0, False), ("BOLT_A", 34.0, False)])
        self.assertEqual([x["decision"] for x in d[1:]], ["pending", "same", "kept_near_arm", "pending", "same", "switched"])

    def test_drop_reference_not_updated_by_near_frames(self):
        """팔 근처 사진의 짧은 길이는 급감 기준이 되지 않는다(F-STOP 끝: 팔 근처 22 mm 뒤 팔 밖 17.9 mm도 급감)."""
        lock = TrackClassLock(LOCK)
        d = feed(lock, [("BOLT_B", 44.0, False), ("BOLT_A", 22.0, True), ("BOLT_A", 17.9, False)])
        self.assertEqual(d[2]["decision"], "kept_drop")

    def test_two_frame_transient_does_not_switch(self):
        """팔 밖 짧은 사진 2장(급감 1 + 셈 1) 뒤 원래 길이로 돌아오면 바뀌지 않는다."""
        lock = TrackClassLock(LOCK)
        d = feed(lock, [("BOLT_B", 40.0, False), ("BOLT_A", 32.0, False), ("BOLT_A", 32.0, False), ("BOLT_B", 40.0, False)])
        self.assertEqual([x["decision"] for x in d], ["first", "kept_drop", "pending", "same"])
        self.assertEqual(lock.classes[1][0], "BOLT_B")

    def test_tracks_independent_and_clear(self):
        """트랙마다 따로 세고, `clear`는 종류를 모두 잊는다."""
        lock = TrackClassLock(LOCK)
        lock.observe(1, "BOLT_B", 0.105, 40.0, False)
        lock.observe(2, "BOLT_A", 0.102, 21.0, False)
        self.assertEqual(lock.observe(2, "BOLT_A", 0.102, 21.0, False)["decision"], "same")
        lock.clear()
        self.assertEqual(lock.classes, {})
        self.assertEqual(lock.observe(1, "BOLT_A", 0.102, 21.0, False)["decision"], "first")

    def test_off_overwrites_like_before(self):
        """설정이 없으면 이전 동작처럼 사진마다 덮어쓴다(팔 근처여도)."""
        lock = TrackClassLock(None)
        d = feed(lock, [("BOLT_B", 42.0, False), ("BOLT_A", 32.0, True)])
        self.assertEqual([x["decision"] for x in d], ["first", "overwrite"])
        self.assertEqual(lock.classes[1][0], "BOLT_A")

    def test_counts(self):
        """판정별 계수."""
        lock = TrackClassLock(LOCK)
        feed(lock, [("BOLT_B", 40.0, False), ("BOLT_A", 30.0, True), ("BOLT_A", 30.0, False)])
        self.assertEqual((lock.counts["first"], lock.counts["kept_near_arm"], lock.counts["kept_drop"]), (1, 1, 1))


class LockConfigTest(unittest.TestCase):
    """설정 검사."""

    def test_config_file_block(self):
        """시험 설정 파일의 class_lock 블록은 3장·30 mm·5 mm."""
        c = parse_track_stream(json.loads(CONFIG.read_text(encoding="utf-8")))
        self.assertEqual(c.class_lock, LOCK)
        self.assertEqual(c.to_json()["class_lock"], LOCK.to_json())

    def test_block_optional(self):
        """블록이 없으면 None(이전 동작)."""
        doc = {k: v for k, v in json.loads(CONFIG.read_text(encoding="utf-8")).items() if k != "class_lock"}
        self.assertIsNone(parse_track_stream(doc).class_lock)

    def test_rejects(self):
        """키 누락·추가, 0장, bool, 음수·비유한 값, drop 0을 거부한다."""
        base = LOCK.to_json()

        for over in ({"switch_frames": 0}, {"switch_frames": True}, {"switch_frames": 2.0}, {"near_arm_mm": -1},
                     {"near_arm_mm": float("nan")}, {"drop_mm": 0}, {"drop_mm": float("inf")}, {"drop_mm": True}, {"extra": 1}):
            with self.subTest(over=over), self.assertRaises(ClassLockConfigError):
                parse_class_lock({**base, **over})

        with self.assertRaises(ClassLockConfigError):
            parse_class_lock({"switch_frames": 3, "near_arm_mm": 30})

        doc = json.loads(CONFIG.read_text(encoding="utf-8"))

        with self.assertRaises(TrackStreamConfigError):
            parse_track_stream({**doc, "class_lock": {"switch_frames": 0, "near_arm_mm": 30, "drop_mm": 5}})

    def test_near_zero_allowed(self):
        """near_arm_mm 0은 허용(캡슐 그대로)."""
        self.assertEqual(parse_class_lock({**LOCK.to_json(), "near_arm_mm": 0}).near_arm_mm, 0.0)


class _NearLive(LiveWorldTracking):
    """가림 판정 함수를 시험에서 고르는 추적 스레드(`near` 집합의 사진 시각이면 모든 점이 팔 근처)."""

    near_ms: set = set()
    margins: list = []

    def _link_occluded(self, T_W_C, frame):
        """그 사진 시각이 `near_ms`에 있으면 여유 안이면 참(여유 30 mm 이상일 때만), 아니면 늘 거짓."""
        t = frame.timestamp_ms
        margins = self.margins

        def fn(_x, _y, margin_mm=0.0):
            margins.append(margin_mm)

            return t in self.near_ms and margin_mm >= 30.0

        return fn, "ok"


class LiveLockTest(unittest.TestCase):
    """추적 스레드에서 발행 `part_class`가 고정된다."""

    def make(self, lock, near_ms=()):
        """종류별 높이·종류 고정 설정·가짜 가림 판정의 추적 스레드."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        tracker = WorldTracker(PartTrackIdIssuer(), config=tcfg(), visibility=vis())
        self.sink = Sink()
        ts = TrackStream(cfg(max_rate_hz=1000.0, queue_capacity=64, class_lock=lock), self.sink, lambda: 0)
        self.lw = _NearLive("r", tracker, live_cfg(), tf_lookup=DualLookup(), plane_z_m=SPLIT, out_dir=tmp.name,
                            visibility=vis(), stream=ts)
        self.lw.near_ms, self.lw.margins = set(near_ms), []
        self.lw.start()
        self.addCleanup(lambda: self.lw.close(2.0))
        self.seq = 0

    def link(self, t_ms, length_m):
        """길이 `length_m` 볼트 하나의 링크 응답(처리까지 기다림)."""
        self.seq += 1
        ctx = link_ctx(t_ms, self.seq)
        self.lw.offer(ctx, result_of(LINK_T, [(0.50, 0.0, length_m, 90.0, 0.9, False)]), _Cam)

        return self.lw.wait(ctx, 2.0)

    def published(self):
        """닫고 발행된 frame마다 첫 트랙의 종류(트랙 없으면 None)와 기록 frame 줄."""
        self.lw.close(2.0)
        recs = [json.loads(t) for t in (Path(self.dir) / "world_tracks.jsonl").read_text().splitlines()]
        kinds = [m["tracks"][0]["part_class"] if m["tracks"] else None for m in self.sink.got if m["kind"] == "frame"]

        return kinds, [r for r in recs if r["kind"] == "frame"]

    def run_seq(self, lengths_m):
        """66 ms 간격 사진들."""
        for i, L in enumerate(lengths_m):
            self.link(i * 66, L)

    def test_near_arm_keeps_class(self):
        """확정(BOLT_B) 뒤 팔 근처 사진의 짧은 끝점(BOLT_A)은 발행 종류를 바꾸지 않고, 기록에 kept_near_arm을 남긴다."""
        self.make(LOCK, near_ms={264, 330, 396})
        self.run_seq([0.040] * 4 + [0.031] * 3)
        kinds, frames = self.published()
        self.assertEqual(kinds[-4:], ["BOLT_B"] * 4)
        locks = [d for r in frames for d in r["stream"].get("class_lock", [])]
        self.assertEqual([d["decision"] for d in locks], ["kept_near_arm"] * 3)
        self.assertIn(30.0, self.lw.margins)
        self.assertEqual(self.lw.summary()["track_stream"]["class_lock"]["kept_near_arm"], 3)

    def test_far_frames_switch(self):
        """팔에서 떨어진 짧은 사진(40 → 31 mm)이 이어지면 급감 1장 뒤 3장째에 BOLT_A로 바뀐다."""
        self.make(LOCK)
        self.run_seq([0.040] * 4 + [0.031] * 4)
        kinds, _ = self.published()
        self.assertEqual(kinds[-4:], ["BOLT_B", "BOLT_B", "BOLT_B", "BOLT_A"])

    def test_off_overwrites(self):
        """종류 고정 설정이 없으면 이전처럼 팔 근처 사진 한 장으로 바뀐다(기록에 class_lock 없음)."""
        self.make(None, near_ms={264})
        self.run_seq([0.040] * 4 + [0.031])
        kinds, frames = self.published()
        self.assertEqual(kinds[-1], "BOLT_A")
        self.assertFalse(any("class_lock" in r["stream"] for r in frames))

    def test_unconfirmed_not_published(self):
        """미확정 단계(첫 사진들)는 지금처럼 트랙·종류를 발행하지 않는다."""
        self.make(LOCK)
        self.run_seq([0.040])
        kinds, _ = self.published()
        self.assertEqual(kinds, [None])


if __name__ == "__main__":
    unittest.main()
