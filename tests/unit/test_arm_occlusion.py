"""팔·그리퍼 가림 판정 시험: 선분 거리, 캡슐 가림, 설정 검사, 추적기 가림 hold, 넘기기 켬의 잘린·마스크 입력."""
from __future__ import annotations

import json
import unittest

import numpy as np

from eyetoarm.vision.arm_occlusion import (Occluder, OccluderConfigError, occluded_fn, parse_occluders, segment_distance,
                                           world_capsules)
from eyetoarm.vision.live_observation_envelope import TfLookupResult
from eyetoarm.vision.observation_envelope import CaptureTf
from test_live_world_tracking_gripper import GRIP_T, LINK_T, Base
from test_world_tracking import winp
from test_world_tracking_gripper import CONFIG, LINK_DT, Scene, inside


def T_at(xyz):
    """회전 없는 4×4."""
    T = np.eye(4)
    T[:3, 3] = xyz
    return T


class GeometryTest(unittest.TestCase):
    """선분 거리·캡슐 가림."""

    def test_segment_distance(self):
        """평행·교차·끝점 최근접 거리가 해석값과 같다."""
        a = np.array
        self.assertAlmostEqual(segment_distance(a([0, 0, 0.]), a([1, 0, 0.]), a([0, 1, 0.]), a([1, 1, 0.])), 1.0)
        self.assertAlmostEqual(segment_distance(a([0, 0, 0.]), a([1, 0, 0.]), a([0.5, -1, 0.]), a([0.5, 1, 0.])), 0.0)
        self.assertAlmostEqual(segment_distance(a([0, 0, 0.]), a([1, 0, 0.]), a([2, 0, 0.]), a([3, 0, 0.])), 1.0)
        self.assertAlmostEqual(segment_distance(a([0, 0, 0.]), a([0, 0, 0.]), a([0, 3, 4.]), a([0, 3, 4.])), 5.0)

    def test_capsule_between_camera_and_point(self):
        """카메라–점 선분 위 캡슐은 가림, 옆으로 비킨 캡슐과 점 뒤 캡슐은 아님."""
        occ = (Occluder("l", (0.0, 0.0, 0.0), (0.0, 0.0, 0.02), 0.03),)
        cam = np.array([0.0, 0.0, 0.5])
        hit = occluded_fn(cam, world_capsules(occ, {"l": T_at([0.25, 0.0, 0.30])}), 0.105)
        side = occluded_fn(cam, world_capsules(occ, {"l": T_at([0.25, 0.2, 0.30])}), 0.105)
        behind = occluded_fn(cam, world_capsules(occ, {"l": T_at([0.6, 0.0, 0.0])}), 0.105)
        self.assertTrue(hit(500.0, 0.0))
        self.assertFalse(side(500.0, 0.0))
        self.assertFalse(behind(500.0, 0.0))

    def test_config_errors(self):
        """빈 목록·키 오류·반지름 0·좌표 길이 오류는 거부, 예시 파일은 읽힘."""
        good = {"link": "l", "p0_m": [0, 0, 0], "p1_m": [0, 0, 1], "radius_m": 0.1}

        for bad in ([], [{**good, "x": 1}], [{**good, "radius_m": 0}], [{**good, "p0_m": [0, 0]}], [{**good, "link": ""}]):
            with self.subTest(bad=bad), self.assertRaises(OccluderConfigError):
                parse_occluders(bad)

        occ = parse_occluders(json.loads(CONFIG.read_text(encoding="utf-8"))["visibility"]["occluders"])
        self.assertIn("forearm_link", {o.link for o in occ})


def occ_all(*_):
    """모두 가림."""
    return True


class OccludedHoldTest(unittest.TestCase):
    """추적기: 가림이면 공백을 세지 않고 occluded_limit까지 유지."""

    def test_occluded_holds_then_limit(self):
        """링크 화면 안이지만 가려진 트랙은 1.5 s 넘어도 살아 있고, 15 s를 넘으면 occluded_limit로 은퇴한다."""
        s = Scene(late_ms=500)
        pid, t = s.confirm()

        for k in range(1, 200):
            s.link(t + k * LINK_DT, [], visible=inside, occluded=occ_all)

        self.assertIn(pid, s.alive())
        self.assertEqual(s.tr.last_occluded, [pid])
        s.link(t + 15_100, [], visible=inside, occluded=occ_all)
        self.assertNotIn(pid, s.alive())
        self.assertEqual(s.tr.stats()["handoff"]["retire_by_reason"], {"occluded_limit": 1})

    def test_not_occluded_counts_gap(self):
        """가리지 않으면(또는 판정 함수 없음 = TF 실패) 기존대로 1.5 s에 in_view_missing."""
        for occ in (lambda *a: False, None):
            s = Scene(late_ms=500)
            pid, t = s.confirm()

            for k in range(1, 30):
                s.link(t + k * LINK_DT, [], visible=inside, occluded=occ)

            self.assertNotIn(pid, s.alive())
            self.assertEqual(s.tr.stats()["handoff"]["retire_by_reason"], {"in_view_missing": 1})

    def test_occluded_then_seen_resets(self):
        """가림 뒤 다시 보이면 가림 상태가 풀리고 같은 번호."""
        s = Scene(late_ms=500)
        pid, t = s.confirm()

        for k in range(1, 60):
            s.link(t + k * LINK_DT, [], visible=inside, occluded=occ_all)

        back = s.link(t + 60 * LINK_DT, [winp(0, 550.0, 0.0)], visible=inside, occluded=occ_all)
        self.assertEqual(back.assignments[0].part_track_id, pid)
        self.assertFalse(s.tr._tracks[0].occluded)


class ClippedMaskOnTest(unittest.TestCase):
    """넘기기 켬: 확정 트랙의 잘린·마스크 입력은 보임만."""

    def test_clipped_and_mask_keep_state(self):
        """잘린 입력·마스크 입력은 위치·속도·길이·각도를 바꾸지 않고 공백만 초기화한다."""
        s = Scene(late_ms=500)

        for k in range(4):
            s.link(k * LINK_DT, [winp(0, 550.0, 0.02 * k * LINK_DT)])

        tr = s.tr._tracks[0]
        before = (tr.x, tr.y, tr.vy, tr.length, tr.angle, tr.last_hit_ms)
        a = s.link(4 * LINK_DT, [winp(0, 545.0, 12.0, length=30.0, angle=80.0, clipped=True)]).assignments[0]
        self.assertEqual((a.part_track_id is not None, a.reason), (True, "clipped"))
        mask = winp(0, 544.0, 14.0)
        mask = type(mask)(**{**mask.__dict__, "kind": "mask", "endpoints_mm": None, "length_mm": None, "angle_deg": None})
        a = s.link(5 * LINK_DT, [mask]).assignments[0]
        self.assertEqual(a.reason, "mask_input")
        self.assertEqual((tr.x, tr.y, tr.vy, tr.length, tr.angle, tr.last_hit_ms), before)
        self.assertEqual((tr.seen_ms, tr.gap_ms), (5 * LINK_DT, 0))

    def test_off_clipped_still_updates(self):
        """꺼짐에서는 잘린 입력이 지금처럼 위치를 갱신한다."""
        s = Scene(visibility=False, gripper=False)

        for k in range(4):
            s.link(k * LINK_DT, [winp(0, 550.0, 0.0)])

        s.link(4 * LINK_DT, [winp(0, 545.0, 5.0, clipped=True)])
        self.assertEqual((s.tr._tracks[0].x, s.tr._tracks[0].y), (545.0, 5.0))


class NearConfirmedTest(unittest.TestCase):
    """넘기기 켬: 확정 근처 잘린 후보는 새 트랙 없음, 확정·임시가 겨루면 확정 유지(D9)."""

    def test_clipped_near_confirmed_no_new_track(self):
        """게이트 밖이지만 30 mm 안의 잘린 후보는 켬이면 near_confirmed, 꺼짐이면 새 임시 트랙."""
        for on in (True, False):
            s = Scene(late_ms=500) if on else Scene(visibility=False, gripper=False)
            pid, t = s.confirm()
            a = s.link(t + LINK_DT, [winp(0, 550.0, 0.0), winp(1, 562.0, 12.0, length=30.0, clipped=True)]).assignments[1]

            if on:
                self.assertEqual((a.part_track_id, a.reason), (None, "near_confirmed"))
                self.assertEqual(len(s.tr._tracks), 1)
                self.assertEqual(s.tr.stats()["handoff"]["near_confirmed_skipped"], 1)
            else:
                self.assertEqual(len(s.tr._tracks), 2)

    def test_confirmed_kept_when_competing_with_tentative(self):
        """한 후보를 확정·임시 트랙이 겨루면 켬은 확정을 유지(같은 번호)하고 임시를 버린다. 꺼짐은 둘 다 끝낸다."""
        for on in (True, False):
            s = Scene(late_ms=500) if on else Scene(visibility=False, gripper=False)
            pid, t = s.confirm()
            s.link(t + LINK_DT, [winp(0, 550.0, 0.0), winp(1, 558.0, 8.0)])
            a = s.link(t + 2 * LINK_DT, [winp(0, 554.0, 4.0)]).assignments[0]

            if on:
                self.assertEqual(a.part_track_id, pid)
                self.assertEqual(s.tr.stats()["handoff"]["tentative_dropped_for_confirmed"], 1)
            else:
                self.assertEqual(a.reason, "ambiguous")
                self.assertNotIn(pid, s.alive())


class LongVelocityTest(unittest.TestCase):
    """링크에 안 보이는 트랙은 긴 기준선 벨트 공통 속도로 예측(D10)."""

    def test_noisy_25ms_link_then_6s_out_of_view(self):
        """25 ms·±1 mm 잡음으로 20 mm/s인 두 볼트 중 하나가 6 s 화면 밖이어도 예측 오차 < 10 mm, 같은 번호로 다시 붙는다."""
        import random
        rng = random.Random(7)
        s = Scene(late_ms=500)
        dt = 25
        y = lambda t, y0: y0 + 0.02 * t + rng.uniform(-1.0, 1.0)   # noqa: E731

        for k in range(80):   # 2 s 둘 다 보임
            t = k * dt
            tf = s.link(t, [winp(0, 520.0, y(t, 0.0)), winp(1, 620.0, y(t, 0.0))])

        a_id = tf.assignments[0].part_track_id
        vis = lambda x, yy: x > 570   # noqa: E731  A(x 520) 화면 밖

        for k in range(80, 320):   # 6 s A 안 보임, B는 계속 보임
            t = k * dt
            s.link(t, [winp(1, 620.0, y(t, 0.0))], visible=vis)

        snap = [h for h in s.tr.held_snapshot(320 * dt) if h["part_track_id"] == a_id][0]
        self.assertLess(abs(snap["pred_mm"][1] - 0.02 * 320 * dt), 10.0)
        self.assertLess(abs(snap["vy_mm_per_s"] - 20.0), 2.0)
        back = s.link(320 * dt, [winp(0, 520.0, 0.02 * 320 * dt), winp(1, 620.0, 0.02 * 320 * dt)])
        self.assertEqual(back.assignments[0].part_track_id, a_id)


class ReentryVelocityTest(unittest.TestCase):
    """링크에서 놓친 트랙은 다시 화면 안으로 예측될 때도 다음 링크 위치 갱신까지 벨트 공통 속도(D10)."""

    def test_reentry_keeps_belt_velocity(self):
        """자기 속도(30 mm/s)가 아니라 벨트 공통 속도로 재진입 프레임을 예측한다(보이는 트랙이 1개라 직전 공통 속도 유지)."""
        s = Scene(late_ms=500)

        for k in range(60):   # B(x 620)는 20 mm/s로 1.5 s, A(x 520)는 30 mm/s
            t = k * 25
            s.link(t, [winp(0, 520.0, 0.03 * t), winp(1, 620.0, 0.02 * t)])

        a = [t for t in s.tr._tracks if t.x < 570][0]
        out = lambda x, y: x > 570   # noqa: E731

        for k in range(60, 260):   # 5 s A 화면 밖
            s.link(k * 25, [winp(1, 620.0, 0.02 * k * 25)], visible=out)

        s.link(260 * 25, [winp(1, 620.0, 0.02 * 260 * 25)], visible=inside)   # A 예측점이 화면 안으로(못 봄)
        self.assertFalse(a.out_link)
        self.assertTrue(a.hold_pred)
        self.assertEqual(s.tr._pred_vy(a, 0.0, None), s.tr._hold_vy)
        self.assertLess(s.tr._hold_vy * 1000.0, 27.0)
        s.link(261 * 25, [winp(0, 520.0, a.y + s.tr._hold_vy * (261 * 25 - a.last_hit_ms)), winp(1, 620.0, 0.02 * 261 * 25)])
        self.assertFalse(a.hold_pred)


class ArmFalseTrackVelocityTest(unittest.TestCase):
    """D14: 그리퍼 오검출(정지) 트랙이 벨트 공통 속도를 끌어내리지 않는다."""

    def scene(self):
        """두 볼트 20 mm/s(x 520·620)를 1.5 s 보인 뒤, 둘 다 가려지고 그리퍼 오검출 하나(x 511, 정지)만 1.5 s 보인다."""
        s = Scene(late_ms=500)

        for k in range(60):
            t = k * 25
            s.link(t, [winp(0, 520.0, 0.02 * t), winp(1, 620.0, 0.02 * t)])

        ids = {t.part_track_id for t in s.tr._tracks}
        return s, ids

    def run_false(self, s, occluded):
        """가림 + 정지 오검출(x 511, y 90) 60프레임."""
        for k in range(60, 120):
            s.link(k * 25, [winp(5, 511.0, 90.0, length=13.0)], visible=inside, occluded=occluded)

    def test_static_false_alone_keeps_velocity(self):
        """재현: 오검출 하나만 보이면(최소 2 미만) 공통 속도는 직전 20 mm/s를 유지한다."""
        s, ids = self.scene()
        self.run_false(s, lambda x, y, m=0.0: abs(x - 511.0) > 3.0)   # 볼트 둘은 가림, 오검출은 캡슐 밖으로 둠
        self.assertLess(abs(s.tr._hold_vy * 1000.0 - 20.0), 1.0)
        self.assertTrue(s.tr.last_velocity["kept_previous"])

    def test_near_arm_track_excluded(self):
        """캡슐(+여유) 근처 트랙은 공통 속도에서 빠진다(기록에 excluded_near_arm)."""
        s, ids = self.scene()
        near = lambda x, y, m=0.0: True   # noqa: E731  모두 캡슐 근처
        self.run_false(s, near)
        self.assertTrue(s.tr.last_velocity["excluded_near_arm"])
        self.assertLess(abs(s.tr._hold_vy * 1000.0 - 20.0), 1.0)
        self.assertGreater(s.tr.stats()["handoff"]["velocity_excluded_near_arm"], 0)

    def test_min_tracks_used_when_enough(self):
        """쓸 수 있는 트랙이 2개 이상이면 그 중앙값으로 갱신한다(정지 벨트 0)."""
        s = Scene(late_ms=500)

        for k in range(60):
            s.link(k * 25, [winp(0, 520.0, 0.0), winp(1, 620.0, 0.0)])

        self.assertFalse(s.tr.last_velocity["kept_previous"])
        self.assertEqual(sorted(s.tr.last_velocity["used_ids"]), sorted(t.part_track_id for t in s.tr._tracks))
        self.assertLess(abs(s.tr._hold_vy), 1e-6)


class OccluderLookup:
    """링크 카메라·그리퍼·팔 링크 TF 대역. `arm` = 팔 링크 위치(None이면 조회 실패)."""

    def __init__(self, arm):
        self.arm = arm

    def __call__(self, frame):
        child = getattr(frame, "optical_frame", None)

        if child == "gripper_cam_optical_frame":
            T = GRIP_T
        elif child is None:
            T = LINK_T
        elif self.arm is None:
            return TfLookupResult(None, "unavailable", 0.1, "fake")
        else:
            T = T_at(self.arm)

        return TfLookupResult(CaptureTf("world", "x", frame.ros_stamp_ns, T), "ok", 0.1)


class LiveOcclusionTest(Base):
    """실시간: 팔 TF로 가림 판정·기록."""

    def run_case(self, arm):
        """확정 뒤 2.5 s 동안 링크가 못 보는 장면 → 기록."""
        lw = self.make(reorder_wait_ms=1.0)
        lw._tf_lookup = OccluderLookup(arm)

        for t in (0, 66, 132):
            pid = self.link(t).assignments[0].part_track_id

        for k in range(1, 40):
            self.link(132 + k * LINK_DT, bolts=())

        alive = [p for p, *_ in lw._tracker.predicted_confirmed(10 ** 6)]

        return pid, alive, self.records(), lw

    def test_arm_between_camera_and_bolt_holds(self):
        """팔 링크가 링크 카메라와 볼트 사이에 있으면 occluded로 기록하고 번호를 유지한다."""
        pid, alive, recs, lw = self.run_case([0.25, 0.0, 0.30])
        frames = [r for r in recs if r["kind"] == "frame" and r.get("occlusion")]
        self.assertIn(pid, alive)
        self.assertEqual(frames[-1]["occluded_ids"], [pid])
        self.assertEqual(frames[-1]["occlusion"], "ok")
        held = frames[-1]["held"][0]
        self.assertEqual((held["part_track_id"], held["state"]), (pid, "occluded"))
        self.assertTrue({"pred_mm", "vy_mm_per_s", "own_vy_mm_per_s", "ref_ms", "seen_ms", "gap_ms"} <= set(held))
        self.assertIsNotNone(lw.summary()["handoff"]["occlusion_ms_p95"])

    def test_arm_tf_missing_is_not_occluded(self):
        """팔 TF 조회 실패면 가림 아님: 공백을 세어 1.5 s 뒤 은퇴, 사유 기록."""
        pid, alive, recs, lw = self.run_case(None)
        self.assertNotIn(pid, alive)
        self.assertTrue(any(str(r.get("occlusion", "")).startswith("tf_unavailable") for r in recs))
        self.assertGreater(lw.summary()["handoff"]["occlusion_tf_fail"], 0)


if __name__ == "__main__":
    unittest.main()
