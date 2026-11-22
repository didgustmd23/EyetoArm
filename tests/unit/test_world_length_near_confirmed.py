"""W 추적 #279 시험: 길이 비로 막힌 끝점 후보는 확정 트랙 게이트 안이면 새 임시를 만들지 않음(F1′, 켬·끔 공통),
모호로 끝난 확정 트랙의 은퇴 사건(F5, 켬만).

합성 W 입력의 값은 #277 (가) 기록(`260_evidence/B_off` 66.8 s 등)과 #279 재생 사례의 길이·위치를 줄인 것이다.
"""
from __future__ import annotations

import unittest

from eyetoarm.vision.tracking import PartTrackIdIssuer
from eyetoarm.vision.world_tracking import WorldTracker
from test_world_tracking import key, tcfg, winp
from test_world_tracking_gripper import vis

DT = 25   # 링크 프레임 간격(ms), 기록과 같음


class Track:
    """링크 프레임만 넣는 도우미(켬은 가시성 설정, 화면 안·가림 없음)."""

    def __init__(self, on: bool):
        """켬/끔 추적기."""
        self.on = on
        self.tr = WorldTracker(PartTrackIdIssuer(), config=tcfg(), visibility=vis() if on else None)
        self.seq = 0

    def link(self, inputs):
        """링크 프레임 하나 → 배정 목록."""
        self.seq += 1
        kw = {"visible": lambda _x, _y: True} if self.on else {}

        return self.tr.update(key(self.seq), self.seq * DT, inputs, **kw).assignments

    def confirm(self, x, y, length):
        """같은 자리 3프레임으로 확정하고 번호."""
        for _ in range(3):
            a = self.link([winp(0, x, y, length=length)])

        return a[0].part_track_id

    def alive(self):
        """살아 있는 확정 번호."""
        return {pid for pid, _, _ in self.tr.predicted_confirmed(10 ** 9)}


class LengthNearConfirmedTest(unittest.TestCase):
    """F1′: 확정 트랙 자리의 길이만 다른 끝점 후보는 새 임시를 만들지 않는다."""

    def test_long_instance_keeps_confirmed_on_and_off(self):
        """(가) 재현: 긴 인스턴스(길이 비 1.42)는 임시가 되지 않고, 다음 보통 후보가 확정 번호를 잇는다(켬·끔)."""
        for on in (False, True):
            with self.subTest(on=on):
                s = Track(on)
                pid = s.confirm(520.0, -4.0, 19.2)
                long_ = s.link([winp(0, 523.2, -1.0, length=27.2)])[0]
                self.assertEqual((long_.state.value, long_.part_track_id, long_.reason),
                                 ("untracked", None, "length_near_confirmed"))
                a = s.link([winp(0, 521.6, -0.8, length=22.25)])[0]
                self.assertEqual((a.state.value, a.part_track_id), ("confirmed", pid))
                self.assertIn(pid, s.alive())
                self.assertEqual(s.tr.stats()["length_near_confirmed"], 1)

    def test_long_instance_outside_gate_makes_tentative(self):
        """확정 트랙 게이트 밖이면 길이와 상관없이 지금처럼 새 임시가 생긴다."""
        s = Track(False)
        s.confirm(520.0, -4.0, 19.2)
        a = s.link([winp(0, 540.0, -4.0, length=27.2)])[0]
        self.assertEqual(a.state.value, "tentative")
        self.assertEqual(s.tr.stats()["length_near_confirmed"], 0)

    def test_length_block_by_tentative_still_makes_tentative(self):
        """임시 트랙(번호 없음)과 길이만 다르면 막지 않는다(확정 트랙에만 적용)."""
        s = Track(False)
        s.link([winp(0, 520.0, -4.0, length=19.2)])
        a = s.link([winp(0, 522.0, -3.0, length=27.2)])[0]
        self.assertEqual(a.reason, None)
        self.assertEqual(s.tr.stats()["length_near_confirmed"], 0)

    def test_two_confirmed_still_end_both(self):
        """확정 둘이 겨루는 모호는 지금과 같다: 후보는 ambiguous, 두 번호 모두 끝남(켬·끔)."""
        for on in (False, True):
            with self.subTest(on=on):
                s = Track(on)

                for _ in range(3):
                    out = s.link([winp(0, 520.0, 0.0, length=20.0), winp(1, 526.0, 0.0, length=20.0)])

                ids = {a.part_track_id for a in out}
                a = s.link([winp(0, 523.0, 0.0, length=20.0)])[0]
                self.assertEqual(a.reason, "ambiguous")
                self.assertFalse(ids & s.alive())

    def test_same_length_confirmed_and_tentative_off_unchanged(self):
        """길이가 같은 확정·임시 다툼은 끔에서 지금처럼 둘 다 끝난다(F1은 적용하지 않음)."""
        s = Track(False)
        pid = s.confirm(550.0, 0.0, 40.0)
        s.link([winp(0, 550.0, 0.0), winp(1, 558.0, 8.0)])
        a = s.link([winp(0, 554.0, 4.0)])[0]
        self.assertEqual(a.reason, "ambiguous")
        self.assertNotIn(pid, s.alive())

    def test_known_side_effect_real_bolt_delayed_next_to_short_false_track(self):
        """알려진 부작용(재생 `r6 pre_d10fix B_off` 191.6 s): 짧은 오검출 확정 트랙(13.3 mm) 게이트 안의 진짜 볼트(20.6 mm)는
        게이트 안에 있는 동안 새 임시를 만들지 못해 번호가 늦어진다. 게이트를 벗어나면 지금처럼 임시가 생긴다."""
        s = Track(False)
        s.confirm(520.5, 91.7, 13.3)

        for y in (93.2, 94.0, 94.1):
            a = s.link([winp(0, 520.5, 91.7, length=13.3), winp(1, 519.9, y, length=20.6)])
            self.assertEqual(a[1].reason, "length_near_confirmed")

        a = s.link([winp(0, 520.5, 91.7, length=13.3), winp(1, 519.9, 110.0, length=20.6)])
        self.assertEqual(a[1].state.value, "tentative")

    def test_known_side_effect_real_bolt_takes_false_track_id(self):
        """알려진 부작용(기록 `r6 pre_d10fix B_off` 68.8 s): 짧은 오검출 확정 트랙(15.1 mm) 옆에서 진짜 볼트의 첫 끝점이
        길이 비로 막힌 뒤, 길이 비 안인 다음 후보는 오검출 트랙의 번호를 잇는다(이전에는 모호로 둘 다 끝나고 새 번호)."""
        s = Track(False)
        finger = s.confirm(520.7, 91.7, 15.1)
        first = s.link([winp(0, 519.85, 93.14, length=21.59)])[0]
        self.assertEqual(first.reason, "length_near_confirmed")
        a = s.link([winp(0, 519.88, 94.11, length=20.77)])[0]
        self.assertEqual((a.state.value, a.part_track_id), ("confirmed", finger))


class AmbiguousRetireEventTest(unittest.TestCase):
    """F5: 모호로 끝난 확정 트랙의 은퇴 사건."""

    def test_on_records_retire_ambiguous(self):
        """켬: 확정 둘 모호 → 번호마다 retire(reason ambiguous) 1건, retire_by_reason에 계수."""
        s = Track(True)

        for _ in range(3):
            out = s.link([winp(0, 520.0, 0.0, length=20.0), winp(1, 526.0, 0.0, length=20.0)])

        ids = sorted(a.part_track_id for a in out)
        s.tr.drain_events()
        s.link([winp(0, 523.0, 0.0, length=20.0)])
        ev = [e for e in s.tr.drain_events() if e.get("reason") == "ambiguous"]
        self.assertEqual(sorted(e["part_track_id"] for e in ev), ids)
        self.assertTrue(all(e["kind"] == "retire" and e["timestamp_ms"] == s.seq * DT for e in ev))
        self.assertEqual(s.tr.stats()["handoff"]["retire_by_reason"].get("ambiguous"), 2)

    def test_on_by_track_ambiguity_also_records(self):
        """켬: 한 확정 트랙 근처 후보 둘(트랙 쪽 모호, D13 모양)로 끝나도 retire ambiguous를 남긴다."""
        s = Track(True)
        pid = s.confirm(520.0, 0.0, 20.0)
        s.tr.drain_events()
        # 끝점 차 3 mm 초과·마스크 없음이라 중복 합치기 밖, 둘 다 비용 차 0.35 안
        s.link([winp(0, 520.0, 0.0, length=20.0), winp(1, 522.0, 3.0, length=26.0)])
        ev = [e for e in s.tr.drain_events() if e.get("reason") == "ambiguous"]
        self.assertEqual([e["part_track_id"] for e in ev], [pid])

    def test_off_records_no_event(self):
        """끔: 사건을 남기지 않고 넘기기 계수도 없다."""
        s = Track(False)

        for _ in range(3):
            s.link([winp(0, 520.0, 0.0, length=20.0), winp(1, 526.0, 0.0, length=20.0)])

        s.link([winp(0, 523.0, 0.0, length=20.0)])
        self.assertEqual(s.tr.drain_events(), [])
        self.assertNotIn("handoff", s.tr.stats())

    def test_tentative_ended_by_ambiguity_has_no_event(self):
        """임시 트랙은 번호가 없으므로 모호로 끝나도 사건을 남기지 않는다(켬)."""
        s = Track(True)
        s.link([winp(0, 520.0, 0.0, length=20.0), winp(1, 526.0, 0.0, length=20.0)])
        s.link([winp(0, 523.0, 0.0, length=20.0)])
        self.assertEqual([e for e in s.tr.drain_events() if e.get("reason") == "ambiguous"], [])


if __name__ == "__main__":
    unittest.main()
