"""W 추적 넘기기 시험: 그리퍼 갱신·가시성 만료·오연결 방지(합성 W 입력)."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from eyetoarm.vision.tracking import PartTrackIdIssuer, TrackingContractError, TrackingInputError
from eyetoarm.vision.world_tracking import GripperInput, VisibilityConfig, WorldTracker, WorldTrackerConfigError, parse_visibility
from test_world_tracking import key, tcfg, winp

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "vision" / "world_tracker_gripper_test.json"
LINK_DT, GRIP_DT = 66, 100


def vis():
    """예시 파일의 visibility 블록."""
    return parse_visibility(json.loads(CONFIG.read_text(encoding="utf-8"))["visibility"])


def ginp(i, x_mm, y_mm, length=40.0, angle=90.0, clipped=False):
    """그리퍼 입력 직접 생성."""
    return GripperInput(index=i, center_mm=(x_mm, y_mm), length_mm=length, angle_deg=angle, clipped=clipped)


def inside(_x, _y):
    """화면 안."""
    return True


def outside(_x, _y):
    """화면 밖."""
    return False


class Scene:
    """링크·그리퍼 프레임을 시각 순서로 넣는 도우미."""

    def __init__(self, visibility=True, gripper=True, late_ms=None):
        """추적기를 만든다."""
        self.tr = WorldTracker(PartTrackIdIssuer(), config=tcfg(), visibility=vis() if visibility else None,
                               gripper_gate_mm=(10.0, 15.0) if gripper else None, gripper_late_ms=late_ms)
        self.seq = 0

    def link(self, t_ms, inputs, visible=inside, occluded=None):
        """링크 프레임."""
        self.seq += 1

        return self.tr.update(key(self.seq), t_ms, inputs, visible=visible, occluded=occluded)

    def grip(self, t_ms, inputs, visible=inside, generation=0):
        """그리퍼 프레임."""
        return self.tr.update_gripper(generation, t_ms, inputs, visible=visible)

    def confirm(self, x=550.0, y=0.0, length=40.0, cls="bolt"):
        """링크 3프레임으로 확정하고 (번호, 마지막 시각)."""
        for k in range(3):
            tf = self.link(k * LINK_DT, [winp(0, x, y, length=length, cls=cls)])

        return tf.assignments[0].part_track_id, 2 * LINK_DT

    def alive(self):
        """살아 있는 확정 번호."""
        return {pid for pid, _, _ in self.tr.predicted_confirmed(10 ** 9)}


class HandoffTest(unittest.TestCase):
    """링크 → 그리퍼 → 링크."""

    def test_link_to_gripper_to_link_same_id(self):
        """링크 시야 밖 5 s 동안 그리퍼가 갱신하면 링크 복귀 때 같은 번호."""
        s = Scene()
        pid, t = s.confirm()
        self.assertIsNotNone(pid)
        grip_t = t + 1   # 링크 시각과 겹치지 않게

        for link_t in range(t + LINK_DT, t + 5000, LINK_DT):
            while grip_t + GRIP_DT < link_t:
                grip_t += GRIP_DT
                g = s.grip(grip_t, [ginp(0, 550.4, 0.3)])
                self.assertEqual(g.assignments[0].outcome, "gripper_hit")
                self.assertEqual(g.assignments[0].part_track_id, pid)

            s.link(link_t, [], visible=outside)

        back = s.link(t + 5000 + LINK_DT, [winp(0, 550.0, 0.0)])
        self.assertEqual(back.assignments[0].part_track_id, pid)
        switches = [(e["from"], e["to"]) for e in s.tr.drain_events() if e["kind"] == "camera_switch"]
        self.assertEqual(switches, [("link", "gripper"), ("gripper", "link")])

    def test_moving_belt_link_return(self):
        """벨트 20 mm/s: 그리퍼가 위치를 이어 주면 링크 복귀 때 같은 번호, 속도는 그리퍼로 바뀌지 않는다."""
        s = Scene()
        pid = None

        for k in range(6):
            tf = s.link(k * LINK_DT, [winp(0, 550.0, 0.02 * k * LINK_DT)])
            pid = tf.assignments[0].part_track_id

        t0 = 5 * LINK_DT
        vy = s.tr._tracks[0].vy

        for k in range(1, 41):
            t = t0 + k * GRIP_DT
            s.link(t - 1, [], visible=outside)
            g = s.grip(t, [ginp(0, 550.2, 0.02 * t)])
            self.assertEqual(g.assignments[0].part_track_id, pid)

        self.assertEqual(s.tr._tracks[0].vy, vy)
        t = t0 + 41 * GRIP_DT
        self.assertEqual(s.link(t, [winp(0, 550.0, 0.02 * t)]).assignments[0].part_track_id, pid)

    def test_link_return_offset(self):
        """링크 복귀 사건에 링크 W − 마지막 그리퍼 W 오프셋, 요약에 분포."""
        s = Scene()
        pid, t = s.confirm()
        s.link(t + 66, [], visible=outside)
        s.grip(t + 100, [ginp(0, 552.0, 1.0)])
        s.link(t + 200, [winp(0, 550.0, 0.0)])
        back = [e for e in s.tr.drain_events() if e["kind"] == "camera_switch" and e["to"] == "link"][0]
        self.assertEqual((back["link_return_offset_mm"]["x"], back["link_return_offset_mm"]["y"]), (-2.0, -1.0))
        self.assertAlmostEqual(back["link_return_offset_mm"]["dist"], 5 ** 0.5, places=3)
        self.assertEqual(back["link_return_offset_mm"]["dt_ms"], 100)
        offsets = s.tr.stats()["handoff"]["link_return_offset_mm"]
        self.assertEqual((offsets["n"], offsets["median"], offsets["max"]), (1, round(5 ** 0.5, 3), round(5 ** 0.5, 3)))

    def test_link_return_offset_none_after_clipped_only(self):
        """그리퍼가 걸린 성분으로만 봤으면 오프셋은 None, 분포에 넣지 않는다."""
        s = Scene()
        pid, t = s.confirm()
        s.grip(t + 100, [ginp(0, 553.0, 2.0, length=12.0, clipped=True)])
        s.link(t + 200, [winp(0, 550.0, 0.0)])
        back = [e for e in s.tr.drain_events() if e["kind"] == "camera_switch" and e["to"] == "link"][0]
        self.assertIsNone(back["link_return_offset_mm"])
        self.assertEqual(s.tr.stats()["handoff"]["link_return_offset_mm"]["n"], 0)

    def test_clipped_keeps_alive_without_moving(self):
        """가장자리 성분은 위치를 바꾸지 않고 「보임」으로만 센다."""
        s = Scene()
        pid, t = s.confirm()
        g = s.grip(t + 50, [ginp(0, 553.0, 2.0, length=12.0, clipped=True)])
        self.assertEqual((g.assignments[0].outcome, g.assignments[0].part_track_id), ("gripper_seen_clipped", pid))
        self.assertEqual((s.tr._tracks[0].x, s.tr._tracks[0].y), (550.0, 0.0))


class ExpiryTest(unittest.TestCase):
    """가시성 만료·hold 상한."""

    def test_hold_until_limit(self):
        """두 화면 모두 밖이면 30 s까지 유지, 넘으면 hold_limit 은퇴."""
        s = Scene()
        pid, t = s.confirm()

        for link_t in range(t + LINK_DT, t + 29_000, 500):
            s.link(link_t, [], visible=outside)
            s.grip(link_t + 1, [], visible=outside)

        self.assertIn(pid, s.alive())
        s.link(t + 30_001 + LINK_DT, [], visible=outside)
        self.assertNotIn(pid, s.alive())
        self.assertEqual([e["reason"] for e in s.tr.drain_events() if e["kind"] == "retire"], ["hold_limit"])

    def test_in_view_missing_regression(self):
        """링크 화면 안인데 못 보면 max_gap_ms(1.5 s)에 in_view_missing 은퇴."""
        s = Scene()
        pid, t = s.confirm()
        gone = None

        for link_t in range(t + LINK_DT, t + 3000, LINK_DT):
            s.link(link_t, [])

            if gone is None and pid not in s.alive():
                gone = link_t - t

        self.assertTrue(1500 < gone <= 1500 + 2 * LINK_DT, gone)
        self.assertEqual(s.tr.stats()["handoff"]["retire_by_reason"], {"in_view_missing": 1})

    def test_gap_attributed_per_camera_link_in_view(self):
        """섞인 열: 링크 화면 안·그리퍼 화면 밖이면 공백은 링크 프레임 간격으로만 쌓인다."""
        s = Scene()
        pid, t = s.confirm()
        gone, grip_t = None, t

        for link_t in range(t + LINK_DT, t + 3000, LINK_DT):
            s.link(link_t, [])

            while grip_t + GRIP_DT < link_t + LINK_DT:
                grip_t += GRIP_DT
                s.grip(grip_t, [], visible=outside)

            if gone is None and pid not in s.alive():
                gone = link_t - t

        self.assertTrue(1500 < gone <= 1500 + 2 * LINK_DT, gone)

    def test_gap_attributed_per_camera_gripper_in_view(self):
        """섞인 열: 링크 화면 밖·그리퍼 화면 안이면 공백은 그리퍼 프레임 간격으로만 쌓인다."""
        s = Scene()
        pid, t = s.confirm()
        gone, link_t = None, t

        for grip_t in range(t + GRIP_DT, t + 3000, GRIP_DT):
            s.grip(grip_t, [])

            while link_t + LINK_DT < grip_t + GRIP_DT:
                link_t += LINK_DT
                s.link(link_t, [], visible=outside)

            if gone is None and pid not in s.alive():
                gone = grip_t - t

        self.assertTrue(1500 < gone <= 1500 + 2 * GRIP_DT + LINK_DT, gone)

    def test_first_camera_frame_adds_no_gap(self):
        """그 카메라의 첫 프레임은 이전 구간을 공백으로 세지 않는다(링크 밖 3 s 뒤 그리퍼 첫 프레임에 바로 은퇴 안 함)."""
        s = Scene()
        pid, t = s.confirm()

        for link_t in range(t + LINK_DT, t + 3000, LINK_DT):
            s.link(link_t, [], visible=outside)

        s.grip(t + 3001, [])
        s.grip(t + 3101, [])
        self.assertIn(pid, s.alive())
        s.tr.note_camera("gripper", t + 3150)
        self.assertEqual(s.tr._cam_ts["gripper"], t + 3150)

    def test_lifted_bolt_no_ghost(self):
        """집어 올린 뒤(그리퍼 near 무효 = 화면 밖, 링크는 빈 벨트)는 1.5 s 뒤 은퇴, 30 s 유령 없음."""
        s = Scene()
        pid, t = s.confirm()

        for k in range(1, 11):
            s.link(t + k * GRIP_DT - 1, [], visible=outside)
            s.grip(t + k * GRIP_DT, [ginp(0, 550.0, 0.0)])

        lift = t + 10 * GRIP_DT
        gone = None

        for k in range(1, 60):
            now = lift + k * LINK_DT
            s.grip(now - 1, [], visible=outside)
            s.link(now, [], visible=inside)

            if gone is None and pid not in s.alive():
                gone = now - lift

        self.assertIsNotNone(gone)
        self.assertLessEqual(gone, 1500 + 2 * LINK_DT)
        self.assertEqual(s.tr.stats()["handoff"]["retire_by_reason"], {"in_view_missing": 1})


class MisassociationTest(unittest.TestCase):
    """오연결 방지."""

    def test_ambiguous_between_two_tracks(self):
        """두 트랙 사이 성분은 gripper_ambiguous, 두 트랙 모두 그대로."""
        s = Scene()

        for k in range(3):
            tf = s.link(k * LINK_DT, [winp(0, 545.0, 0.0), winp(1, 555.0, 10.0)])

        ids = {a.instance_index: a.part_track_id for a in tf.assignments}
        self.assertTrue(all(ids.values()))
        before = [(t.x, t.y) for t in s.tr._tracks]
        g = s.grip(300, [ginp(0, 550.0, 5.0)])
        self.assertEqual(g.assignments[0].outcome, "gripper_ambiguous")
        self.assertEqual([(t.x, t.y) for t in s.tr._tracks], before)

    def test_length_mismatch(self):
        """길이 비 초과는 갱신하지 않는다."""
        s = Scene()
        s.confirm()
        g = s.grip(300, [ginp(0, 550.0, 0.0, length=20.0)])
        self.assertEqual(g.assignments[0].outcome, "gripper_length_mismatch")

    def test_no_new_ids_or_class_change(self):
        """그리퍼로는 새 번호·새 트랙이 생기지 않고 클래스도 그대로."""
        s = Scene()
        pid, _ = s.confirm(cls="nut")
        issued = s.tr.stats()["issued"]
        g = s.grip(300, [ginp(0, 550.0, 0.0), ginp(1, 650.0, 100.0)])
        self.assertEqual([a.outcome for a in g.assignments], ["gripper_hit", "gripper_unmatched"])
        self.assertEqual((s.tr.stats()["issued"], len(s.tr._tracks), s.tr._tracks[0].cls), (issued, 1, "nut"))

    def test_tentative_not_updated(self):
        """임시 트랙은 그리퍼로 갱신하지 않는다."""
        s = Scene()
        s.link(0, [winp(0, 550.0, 0.0)])
        self.assertEqual(s.grip(50, [ginp(0, 550.0, 0.0)]).assignments[0].outcome, "gripper_unmatched")


class LateGripperTest(unittest.TestCase):
    """링크보다 늦게 처리되는 그리퍼 관측(D4)."""

    def test_late_updates_without_moving_tracker_time(self):
        """상한 안의 늦은 관측은 그 시각 예측으로 짝지어 반영하고, 추적기 시각은 그대로라 뒤 링크가 stale이 아니다."""
        s = Scene(late_ms=500)
        pid, t = s.confirm()
        s.link(t + 68, [], visible=outside)
        g = s.grip(t + 48, [ginp(0, 552.0, 1.0)])
        self.assertTrue(g.updated and g.late)
        self.assertEqual((g.assignments[0].outcome, g.assignments[0].part_track_id), ("gripper_hit", pid))
        self.assertEqual(s.tr._last_ts, t + 68)
        self.assertEqual((s.tr._tracks[0].x, s.tr._tracks[0].last_hit_ms), (552.0, t + 48))
        self.assertTrue(s.link(t + 134, [winp(0, 550.0, 0.0)]).updated)
        self.assertEqual(s.tr.stats()["handoff"]["gripper_late"], 1)

    def test_too_late_dropped(self):
        """상한보다 늦은 관측은 late_dropped로 센다."""
        s = Scene(late_ms=500)
        _, t = s.confirm()
        s.link(t + 1000, [], visible=outside)
        self.assertEqual(s.grip(t + 400, [ginp(0, 550.0, 0.0)]).reason, "late_dropped")

    def test_late_after_link_saw_it(self):
        """링크가 그 뒤에 이미 본 트랙은 늦은 그리퍼 관측으로 바꾸지 않는다."""
        s = Scene(late_ms=500)
        pid, t = s.confirm()
        s.link(t + 68, [winp(0, 550.0, 0.0)])
        g = s.grip(t + 48, [ginp(0, 553.0, 2.0)])
        self.assertEqual(g.assignments[0].outcome, "gripper_late_seen")
        self.assertEqual((s.tr._tracks[0].x, s.tr._tracks[0].last_hit_ms), (550.0, t + 68))

    def test_early_gripper_does_not_stale_link(self):
        """링크보다 앞선 그리퍼 관측 뒤에 온 더 이른 링크도 갱신된다(추적기 시각은 링크만 앞당김)."""
        s = Scene(late_ms=500)
        pid, t = s.confirm()
        s.grip(t + 60, [ginp(0, 551.0, 0.5)])
        back = s.link(t + 40, [winp(0, 550.0, 0.0)])
        self.assertTrue(back.updated)
        self.assertEqual(back.assignments[0].part_track_id, pid)
        self.assertEqual(s.tr._tracks[0].last_hit_ms, t + 60)


class VelocityHoldTest(unittest.TestCase):
    """속도는 링크 연속 hit로만(F6), hold 예측은 벨트 공통 속도(F5)."""

    def test_velocity_only_from_consecutive_link_hits(self):
        """그리퍼 갱신 직후 링크 hit는 속도를 바꾸지 않고, 그다음 링크 hit부터 갱신한다."""
        s = Scene(late_ms=500)

        for k in range(4):
            s.link(k * LINK_DT, [winp(0, 550.0, 0.02 * k * LINK_DT)])

        tr = s.tr._tracks[0]
        vy = tr.vy
        s.grip(4 * LINK_DT - 10, [ginp(0, 550.0, 0.02 * (4 * LINK_DT - 10) + 3.0)])
        s.link(4 * LINK_DT, [winp(0, 550.0, 0.02 * 4 * LINK_DT)])
        self.assertEqual(tr.vy, vy)
        s.link(5 * LINK_DT, [winp(0, 550.0, 0.02 * 5 * LINK_DT + 1.0)])
        self.assertNotEqual(tr.vy, vy)

    def test_hold_uses_belt_common_velocity(self):
        """두 화면 밖(hold) 트랙은 자기 속도 대신 보이는 트랙들의 공통 속도(여기선 0)로 예측해 3 s 뒤 같은 번호로 붙는다."""
        s = Scene(late_ms=500)
        link_vis = lambda x, y: x > 570   # noqa: E731  A(x 520)는 링크 화면 밖

        for k in range(5):
            tf = s.link(k * LINK_DT, [winp(0, 520.0, 0.02 * k * LINK_DT), winp(1, 620.0, 0.0)])

        a_id = tf.assignments[0].part_track_id
        t_end = 4 * LINK_DT

        for k in range(1, 46):
            s.link(t_end + k * LINK_DT, [winp(1, 620.0, 0.0)], visible=link_vis)
            s.grip(t_end + k * LINK_DT + 1, [], visible=outside)

        back = s.link(t_end + 46 * LINK_DT, [winp(0, 520.0, 0.02 * t_end), winp(1, 620.0, 0.0)])
        self.assertEqual(back.assignments[0].part_track_id, a_id)

    def test_hold_belt_stopped_predicts_zero(self):
        """belt_running False면 hold 예측 속도는 0."""
        s = Scene(late_ms=500)

        for k in range(5):
            s.link(k * LINK_DT, [winp(0, 520.0, 0.02 * k * LINK_DT)])

        t = s.tr._tracks[0]
        t.out_link = t.out_grip = True
        self.assertEqual(s.tr._predict(t, t.last_hit_ms + 3000, 0.02, False), (t.x, t.y))


class GuardTest(unittest.TestCase):
    """세대·시각·설정 검사."""

    def test_not_updated_reasons(self):
        """링크 전·다른 세대·지난 시각은 갱신하지 않는다."""
        s = Scene()
        self.assertEqual(s.grip(10, [ginp(0, 550.0, 0.0)]).reason, "no_link_scope")
        s.confirm()
        self.assertEqual(s.grip(300, [], generation=1).reason, "old_generation")
        self.assertEqual(s.grip(100, []).reason, "late_dropped")
        self.assertEqual(s.tr.stats()["handoff"]["gripper_not_updated"],
                         {"no_link_scope": 1, "old_generation": 1, "late_dropped": 1})

    def test_reset_clears(self):
        """reset 뒤에는 링크 전과 같다."""
        s = Scene()
        s.confirm()
        s.tr.reset("test")
        self.assertEqual(s.grip(1000, []).reason, "no_link_scope")

    def test_contract_errors(self):
        """그리퍼 게이트 없이 부르기·나쁜 입력·나쁜 설정은 거부."""
        s = Scene(gripper=False)

        with self.assertRaises(TrackingContractError):
            s.grip(10, [])

        with self.assertRaises(TrackingInputError):
            Scene().grip(10, [ginp(0, float("nan"), 0.0)])

        for bad in ({"margin_px": -1, "hold_limit_ms": 1}, {"margin_px": 0, "hold_limit_ms": 1.5},
                    {"margin_px": 0}, {"margin_px": 0, "hold_limit_ms": 1, "x": 1}):
            with self.subTest(bad=bad), self.assertRaises(WorldTrackerConfigError):
                parse_visibility(bad)

        with self.assertRaises(TrackingContractError):
            WorldTracker(PartTrackIdIssuer(), config=tcfg(), gripper_gate_mm=(0.0, 1.0))

    def test_off_has_no_handoff_stats(self):
        """기능이 꺼져 있으면 계수·사건이 없고 visible 인자는 무시된다."""
        s = Scene(visibility=False, gripper=False)
        pid, t = s.confirm()
        s.link(t + 1600, [], visible=outside)
        self.assertNotIn("handoff", s.tr.stats())
        self.assertNotIn(pid, s.alive())
        self.assertEqual(s.tr.drain_events(), [])
        self.assertIsInstance(vis(), VisibilityConfig)


if __name__ == "__main__":
    unittest.main()
