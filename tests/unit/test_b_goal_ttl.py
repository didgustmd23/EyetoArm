"""목표 유효기간 `valid_until_ns` = 촬영 시각 + `goal_ttl_s` (C-9, B 제안 2026-11-20)."""
import unittest

from eyetoarm.transforms.b_stage import BStage
from eyetoarm.transforms.sort_part_client import goal_to_msg
from test_b_process_frame import bolt, envelope
from test_b_transforms import CAM, ROOT


class Msg:
    pass


def no_lookup(frame):
    raise AssertionError("envelope TF only")


class GoalTtlTest(unittest.TestCase):
    def setUp(self):
        self.stage = BStage.from_config(ROOT, CAM, no_lookup)

    def test_valid_until_is_capture_plus_ttl(self):
        env = envelope([bolt(0, 0.55, -0.12)])
        rec = self.stage.process_frame(env)
        stamp = env.observations.frame.ros_stamp_ns
        self.assertEqual(self.stage.frames["goal_ttl_s"], 2.0)
        self.assertEqual(rec["goal"]["valid_until_ns"], stamp + 2_000_000_000)
        self.assertEqual(goal_to_msg(rec["goal"], Msg, Msg).valid_until_ns, stamp + 2_000_000_000)   # 액션 칸까지

    def test_no_ttl_sends_zero(self):
        self.stage.frames = dict(self.stage.frames, goal_ttl_s=None)
        rec = self.stage.process_frame(envelope([bolt(0, 0.55, -0.12)]))
        self.assertEqual(rec["goal"]["valid_until_ns"], 0)


if __name__ == "__main__":
    unittest.main()
