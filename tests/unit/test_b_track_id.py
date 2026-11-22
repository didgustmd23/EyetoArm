"""B-11: 목록 후보의 A 추적 번호(`part_track_id`)를 받아 목표에 싣고, 고른 1개에 같은 부품 중복 방지를 적용한다."""
import dataclasses
import unittest

from eyetoarm.transforms.b_stage import BStage
from eyetoarm.vision.tracked_observation import TrackedObjectObservation
from test_b_process_frame import bolt, envelope
from test_b_transforms import CAM, ROOT


def no_lookup(frame):
    raise AssertionError("envelope TF only")


def tracked(c, track_id):
    """A가 확정 번호를 실은 후보(하위 형)."""
    return TrackedObjectObservation(**{f.name: getattr(c, f.name) for f in dataclasses.fields(c)},
                                    part_track_id=track_id, track_state=None if track_id else "unconfirmed")


class TrackIdTest(unittest.TestCase):
    def setUp(self):
        self.stage = BStage.from_config(ROOT, CAM, no_lookup)

    def test_track_id_reaches_goal_and_results(self):
        rec = self.stage.process_frame(envelope([tracked(bolt(0, 0.55, -0.12), 7), tracked(bolt(1, 0.56, 0.05), 9)]))
        self.assertEqual((rec["selected_instance_index"], rec["goal"]["part_track_id"]), (1, 9))   # y 최대 = 9번
        self.assertEqual([r["part_track_id"] for r in rec["results"]], [7, 9])

    def test_same_track_sent_once_until_release(self):
        env = lambda: envelope([tracked(bolt(0, 0.56, 0.05), 9)])
        first = self.stage.process_frame(env())["goal"]
        second = self.stage.process_frame(env())["goal"]
        self.assertEqual((first["kind"], second["kind"], second["reason"]), ("PICK_PLACE", "SKIP", "ALREADY_SENT"))
        self.stage.release(9)                                   # C 결과가 오면 다시 보낼 수 있음
        self.assertEqual(self.stage.process_frame(env())["goal"]["kind"], "PICK_PLACE")

    def test_untracked_candidate_not_deduped(self):
        env = lambda: envelope([bolt(0, 0.56, 0.05)])           # 번호 없는 후보(추적 끔·미확정)
        kinds = [self.stage.process_frame(env())["goal"]["kind"] for _ in range(2)]
        self.assertEqual(kinds, ["PICK_PLACE", "PICK_PLACE"])
        self.assertIsNone(self.stage.process_frame(envelope([tracked(bolt(0, 0.56, 0.05), None)]))["goal"]["part_track_id"])


if __name__ == "__main__":
    unittest.main()
