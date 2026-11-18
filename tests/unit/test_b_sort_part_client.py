"""B → C `/sort_part` 클라이언트 시험 — ROS 없이 주문서 변환과 보내기·받기 흐름만 본다."""
import threading
import unittest
from collections import Counter
from types import SimpleNamespace

from eyetoarm.transforms.sort_part_client import SortPartClient, goal_to_msg


class Msg:
    """ROS 메시지 대신 쓰는 빈 그릇."""


GOAL = {
    "kind": "PICK_PLACE", "goal_id": "3f0c2a8e-1111-4a2b-9c3d-000000000001", "spec": "BOLT_A",
    "sort_job_id": "job-1", "run_epoch": 2, "part_track_id": "t7", "source_id": "link_cam", "image_seq": 7,
    "capture_generation": 3, "generation": 4, "job_seq": 5, "capture_stamp_ns": 123, "timestamp_base": "SIM_TIME",
    "frame_id": "base_link", "observed_position_mm": [550.0, 50.0, 105.0],
    "grasp_candidates": [
        {"position_mm": [550.0, 53.0, 105.0], "orientation_xyzw": [1.0, 0.0, 0.0, 0.0], "closing_yaw_deg": 90.0},
        {"position_mm": [550.0, 53.0, 105.0], "orientation_xyzw": [0.0, 1.0, 0.0, 0.0], "closing_yaw_deg": -90.0},
    ],
    "gripper_opening_mm": 20.0, "approach_height_mm": 205.0, "lift_height_mm": 250.0,
    "place": {"bin": "BOLT_A", "position_mm": [720.0, -50.0, 180.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
    "calibration_id": "cal-1",
}


class Future:
    """바로 끝나는 가짜 future."""

    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value

    def add_done_callback(self, cb):
        cb(self)


class FakeActionClient:
    def __init__(self, ready=True, accepted=True, status="SUCCEEDED"):
        self.ready, self.accepted, self.status, self.sent = ready, accepted, status, []

    def server_is_ready(self):
        return self.ready

    def send_goal_async(self, msg, feedback_callback=None, goal_uuid=None):
        self.sent.append((msg, goal_uuid))
        feedback_callback(SimpleNamespace(feedback=SimpleNamespace(stage="EXECUTING")))
        result = SimpleNamespace(result=SimpleNamespace(status=self.status, reason=""))
        handle = SimpleNamespace(accepted=self.accepted, get_result_async=lambda: Future(result))
        return Future(handle)


class FakeStage:
    def __init__(self):
        self.released = []

    def release(self, tid):
        self.released.append(tid)


def make_client(action_client):
    """rclpy 없이 클라이언트를 만든다 (생성자 대신 필요한 속성만 채움)."""
    c = SortPartClient.__new__(SortPartClient)
    c._SortPart = SimpleNamespace(Goal=Msg)
    c._Cand = Msg
    c.node = SimpleNamespace(get_logger=lambda: SimpleNamespace(info=lambda *_: None))
    c.stage = FakeStage()
    c.client = action_client
    c.stats, c.inflight, c._lock = Counter(), {}, threading.Lock()
    return c


class GoalToMsgTest(unittest.TestCase):
    def test_pick_place_fields(self):
        m = goal_to_msg(GOAL, Msg, Msg)
        self.assertEqual((m.goal_kind, m.part_class, m.place_bin), ("PICK_PLACE", "BOLT_A", "BOLT_A"))
        self.assertEqual((m.run_epoch, m.image_seq, m.capture_stamp_ns), (2, 7, 123))
        self.assertEqual(len(m.grasp_pose_candidates), 2)
        self.assertEqual(m.grasp_pose_candidates[1].closing_yaw_deg, -90.0)
        self.assertEqual(m.place_position_mm, [720.0, -50.0, 180.0])
        self.assertEqual(m.opening_width_mm, 20.0)
        self.assertEqual(m.valid_until_ns, 0)          # C-9 답 전

    def test_pass_through_defaults(self):
        g = {"kind": "PASS_THROUGH", "spec": "NUT", "capture_stamp": {"base": "SIM_TIME", "ns": 9},
             "observed_position_mm": [580.0, 0.0, 105.7], "part_track_id": None, "image_seq": None}
        m = goal_to_msg(g, Msg, Msg)
        self.assertEqual((m.goal_kind, m.part_class, m.capture_stamp_ns), ("PASS_THROUGH", "NUT", 9))
        self.assertEqual((m.part_track_id, m.image_seq, m.place_bin), ("", -1, ""))
        self.assertEqual(m.grasp_pose_candidates, [])


class SendTest(unittest.TestCase):
    def setUp(self):
        import eyetoarm.transforms.sort_part_client as mod
        self._orig = mod._uuid_msg
        mod._uuid_msg = lambda gid: gid                 # unique_identifier_msgs 없이

    def tearDown(self):
        import eyetoarm.transforms.sort_part_client as mod
        mod._uuid_msg = self._orig

    def test_skip_is_not_sent(self):
        ac = FakeActionClient()
        c = make_client(ac)
        self.assertFalse(c.send({"kind": "SKIP", "reason": "OUT_OF_REACH"}))
        self.assertEqual((ac.sent, c.stats["skip"]), ([], 1))

    def test_no_server_does_not_block_and_releases(self):
        ac = FakeActionClient(ready=False)
        c = make_client(ac)
        self.assertFalse(c.send(SimpleNamespace(goal=GOAL)))
        self.assertEqual((ac.sent, c.stats["no_server"], c.stage.released), ([], 1, ["t7"]))

    def test_send_success_uses_goal_id_and_releases(self):
        ac = FakeActionClient()
        c = make_client(ac)
        self.assertTrue(c.send(SimpleNamespace(goal=GOAL)))
        msg, uuid_arg = ac.sent[0]
        self.assertEqual((msg.goal_id, uuid_arg), (GOAL["goal_id"], GOAL["goal_id"]))   # C-3
        self.assertEqual((c.stats["sent"], c.stats["SUCCEEDED"]), (1, 1))
        self.assertEqual((c.inflight, c.stage.released), ({}, ["t7"]))

    def test_rejected_releases(self):
        c = make_client(FakeActionClient(accepted=False))
        c.send(GOAL)
        self.assertEqual((c.stats["rejected"], c.stage.released), (1, ["t7"]))

    def test_busy_same_kind_not_sent_until_result(self):
        ac = PendingActionClient()
        c = make_client(ac)
        self.assertTrue(c.send(dict(GOAL)))
        second = dict(GOAL, goal_id="3f0c2a8e-1111-4a2b-9c3d-000000000002", part_track_id="t8")
        self.assertFalse(c.send(second))                       # 팔 1대: 집기 대기 중 → 안 보냄
        self.assertEqual((len(ac.sent), c.stats["busy"], c.stage.released), (1, 1, ["t8"]))
        ac.finish(0)                                           # C 결과 도착
        self.assertTrue(c.send(second))
        self.assertEqual(len(ac.sent), 2)

    def test_pass_through_not_blocked_by_pick(self):
        ac = PendingActionClient()
        c = make_client(ac)
        self.assertTrue(c.send(dict(GOAL)))
        nut = {"kind": "PASS_THROUGH", "goal_id": "3f0c2a8e-1111-4a2b-9c3d-000000000003", "spec": "NUT",
               "observed_position_mm": [550.0, 0.0, 105.7]}
        self.assertTrue(c.send(nut))                           # 종류가 다르면 보냄
        self.assertEqual((len(ac.sent), c.stats["busy"]), (2, 0))


class PendingActionClient(FakeActionClient):
    """수락만 하고 결과는 `finish(i)`를 부를 때 오는 가짜 서버."""

    def __init__(self):
        super().__init__()
        self._result_cbs = []

    def send_goal_async(self, msg, feedback_callback=None, goal_uuid=None):
        self.sent.append((msg, goal_uuid))
        result = SimpleNamespace(result=SimpleNamespace(status="SUCCEEDED", reason=""))
        pending = SimpleNamespace(add_done_callback=lambda cb: self._result_cbs.append((cb, result)))
        handle = SimpleNamespace(accepted=True, get_result_async=lambda: pending)
        return Future(handle)

    def finish(self, i):
        cb, result = self._result_cbs[i]
        cb(Future(result))


if __name__ == "__main__":
    unittest.main()
