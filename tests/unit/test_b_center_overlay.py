"""B center 주석 영상 발행 시험 (`transforms/center_overlay_pub.py`). ROS 없이 가짜 발행기로 확인한다."""
import unittest
from types import SimpleNamespace

import numpy as np

from eyetoarm.transforms.b_stage import BStage
from eyetoarm.transforms.center_overlay_pub import (center_overlay_consumer, make_image_msg, make_overlay_msg,
                                                    overlay_keys)
from test_b_process_frame import bolt, envelope
from test_b_transforms import CAM, ROOT


class FakeMsg:
    def __init__(self):
        self.header = SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=0), frame_id="")


class FakePublisher:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def publish(self, image, frame, keys=None):
        if self.fail:
            raise RuntimeError("publish down")
        self.calls.append((image, frame))
        self.keys = keys


def no_lookup(frame):
    raise AssertionError("process_frame must use the envelope TF")


class CenterOverlayTest(unittest.TestCase):
    def setUp(self):
        self.stage = BStage.from_config(ROOT, CAM, no_lookup)

    def test_same_result_and_one_image_per_frame(self):
        env = envelope([bolt(0, 0.55, -0.12), bolt(2, 0.56, -0.05)])
        pub = FakePublisher()
        rec = center_overlay_consumer(self.stage, pub)(env)
        self.assertEqual((rec["selected_instance_index"], rec["selection"]), (2, "BOLT_MAX_Y"))  # process_frame과 같은 결과
        self.assertEqual(len(pub.calls), 1)
        image, frame = pub.calls[0]
        src = env.observations.frame
        self.assertIs(frame, src)                                             # 같은 사진의 시각·frame_id로 발행
        self.assertEqual(image.shape, src.image.shape)
        self.assertFalse(np.array_equal(image, src.image))                    # 무언가 그려짐
        self.assertFalse(src.image.flags.writeable)                          # 원본(쓰기 금지)은 그대로

    def test_publish_failure_keeps_b_result(self):
        consume = center_overlay_consumer(self.stage, FakePublisher(fail=True), on_error=lambda e: None)
        rec = consume(envelope([bolt(0, 0.55, -0.12)]))
        self.assertEqual(rec["goal"]["kind"], "PICK_PLACE")
        self.assertEqual(consume.errors, 1)

    def test_image_msg_keeps_stamp_and_frame_id(self):
        img = np.zeros((4, 5, 3), np.uint8)
        msg = make_image_msg(img, 12_345_678_901, "link_cam_optical_frame", FakeMsg)
        self.assertEqual((msg.header.stamp.sec, msg.header.stamp.nanosec), (12, 345_678_901))
        self.assertEqual(msg.header.frame_id, "link_cam_optical_frame")
        self.assertEqual((msg.height, msg.width, msg.encoding, msg.step, len(msg.data)), (4, 5, "bgr8", 15, 60))

    def test_overlay_keys_match_sort_part_goal(self):
        """C 요청: 영상 번호표 = 같은 사진에서 나온 /sort_part 목표의 번호표 (C가 이걸로 짝을 맞춤)."""
        env = envelope([bolt(0, 0.55, -0.12), bolt(2, 0.56, -0.05)])
        pub = FakePublisher()
        rec = center_overlay_consumer(self.stage, pub)(env)
        goal = rec["goal"]
        for k in ("source_id", "image_seq", "capture_generation", "generation", "job_seq", "capture_stamp_ns",
                  "timestamp_base", "calibration_id"):
            self.assertEqual(pub.keys[k], goal[k], k)

    def test_overlay_msg_carries_keys_and_image(self):
        img_msg = make_image_msg(np.zeros((4, 5, 3), np.uint8), 7_000, "f", FakeMsg)
        keys = overlay_keys(envelope([bolt(0, 0.55, -0.12)]))
        m = make_overlay_msg(img_msg, keys, SimpleNamespace)
        self.assertEqual((m.kind, m.image_seq, m.capture_stamp_ns), ("accepted", keys["image_seq"], keys["capture_stamp_ns"]))
        self.assertIs(m.image, img_msg)

    def test_image_msg_rejects_gray(self):
        with self.assertRaises(ValueError):
            make_image_msg(np.zeros((4, 5), np.uint8), 0, "f", FakeMsg)


if __name__ == "__main__":
    unittest.main()
