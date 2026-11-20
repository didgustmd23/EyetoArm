"""B 목록 입구 `BStage.process_frame` 시험 (A 협업 요청 B-09, B 회신·A 동의 2026-11-19).

- 봉투는 A의 실제 자료형(`FrameObservations`·`ObjectObservation`)과 A 생성 함수(`make_list_envelope`)로 만든다.
  그래서 A가 보낼 형식과 B 입구가 맞는지도 같이 확인한다.
- 볼트 두 끝점은 정답 작업 좌표(높이 105 mm 축)를 카메라로 투영해 만든다(`test_b_transforms`와 같은 카메라·TF).
"""
import json
import unittest

import numpy as np

from eyetoarm.common.types import (FRAME_OBSERVATIONS_VERSION, CaptureInfo, CaptureTf, FrameObservations,
                                   ObjectObservation, PixelBasis)
from eyetoarm.transforms.b_stage import BStage, StaticTfLookup
from eyetoarm.transforms.planar_transform import w_to_image
from eyetoarm.vision.list_envelope import make_list_envelope
from test_b_transforms import CAM, ROOT, T_W_C, make_frame

BOLT_Z = 0.105          # 볼트 축 높이 (part_specs)
BOLT_B_LEN = 0.046


def read_only_frame():
    f = make_frame()
    f.image.setflags(write=False)                  # A 입력 소스는 쓰기 금지 영상을 넘긴다
    return f


def bolt(index, cx, cy, deg=0.0, length=BOLT_B_LEN):
    u = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
    head, tip = np.array([cx, cy]) - u * length / 2, np.array([cx, cy]) + u * length / 2
    px = w_to_image(np.array([[*head, BOLT_Z], [*tip, BOLT_Z]]), CAM.K, CAM.dist, T_W_C)
    return ObjectObservation(instance_index=index, class_name="bolt", confidence=0.9, pixel_basis=PixelBasis.RAW,
                             observed_face="axis", axis_endpoints_px=px, calc_contour=None)


def envelope(cands, with_tf=True):
    f = read_only_frame()
    cap = CaptureInfo(stamp_ns=f.ros_stamp_ns, optical_frame=f.optical_frame, calibration_id=f.calibration_id,
                      width=f.width, height=f.height)
    tf = CaptureTf("world", f.optical_frame, f.ros_stamp_ns, T_W_C) if with_tf else None
    fo = FrameObservations(version=FRAME_OBSERVATIONS_VERSION, frame=f, capture=cap, camera=CAM, tf=tf,
                           generation=0, job_seq=4, detector_config_id="det", candidates=tuple(cands))
    return make_list_envelope(fo)


class ProcessFrameTest(unittest.TestCase):
    def setUp(self):
        # tf_lookup은 쓰지 않아야 한다(목록 봉투의 프레임 공통 TF만 씀) → 불리면 실패하게 둔다
        def no_lookup(frame):
            raise AssertionError("process_frame must use the envelope TF, not tf_lookup")

        self.stage = BStage.from_config(ROOT, CAM, no_lookup)
        self.sent = []
        self.stage.on_output = self.sent.append

    def test_picks_bolt_with_largest_y_and_sends_one(self):
        rec = self.stage.process_frame(envelope([bolt(0, 0.55, -0.12), bolt(2, 0.56, -0.05)]))
        self.assertEqual((rec["selected_instance_index"], rec["selection"]), (2, "BOLT_MAX_Y"))
        self.assertEqual(len(self.sent), 1)                                   # C로는 1개만
        self.assertEqual((self.sent[0].goal["kind"], rec["goal"]["kind"]), ("PICK_PLACE", "PICK_PLACE"))
        by_index = {r["instance_index"]: r for r in rec["results"]}
        self.assertEqual((by_index[0]["selected"], by_index[0]["goal_kind"], by_index[0]["goal_reason"]),
                         (False, "SKIP", "NOT_SELECTED"))
        self.assertEqual((by_index[2]["selected"], by_index[2]["spec"]), (True, "BOLT_B"))
        np.testing.assert_allclose(by_index[2]["center_w_mm"], [560.0, -50.0], atol=0.2)   # mm로 변환된 위치
        json.dumps(rec)                                                         # A 기록기가 그대로 JSON으로 씀

    def test_tie_within_5mm_takes_smaller_index(self):
        rec = self.stage.process_frame(envelope([bolt(1, 0.55, -0.050), bolt(3, 0.57, -0.047)]))
        self.assertEqual(rec["selected_instance_index"], 1)                   # 3 mm 차이 → 같은 것으로 보고 작은 번호

    def test_no_pickable_bolt_sends_nothing(self):
        self.stage.frames = dict(self.stage.frames, pick_zone={"y_min": -0.45, "y_max": -0.20})
        rec = self.stage.process_frame(envelope([bolt(0, 0.55, -0.12)]))
        self.assertEqual((rec["selected_instance_index"], rec["selection"], rec["goal"]), (None, "NONE", None))
        self.assertEqual(rec["results"][0]["goal_reason"], "OUT_OF_REACH")
        self.assertEqual(self.sent, [])

    def test_only_first_8_computed(self):
        cands = [bolt(i, 0.52 + 0.006 * i, -0.10, deg=90.0) for i in range(10)]
        rec = self.stage.process_frame(envelope(cands))
        self.assertEqual((rec["candidate_count"], rec["computed"], rec["truncated"]), (10, 8, 2))
        self.assertEqual([r["instance_index"] for r in rec["results"]], list(range(8)))

    def test_missing_tf_skips_whole_frame(self):
        rec = self.stage.process_frame(envelope([bolt(0, 0.55, -0.12)], with_tf=False))
        self.assertEqual((rec["selection"], rec["results"][0]["reason"]), ("TF_UNAVAILABLE", "TF_UNAVAILABLE"))
        self.assertEqual(self.sent, [])

    def test_frame_time_over_budget_skips_chosen(self):
        self.stage.compute_budget_ms = -1.0                                   # 사진 한 장 전체 시간으로 판단
        rec = self.stage.process_frame(envelope([bolt(0, 0.55, -0.12)]))
        self.assertEqual((rec["goal"]["kind"], rec["goal"]["reason"]), ("SKIP", "B_TIMEOUT"))

    def test_goal_carries_same_correlation_keys_as_envelope(self):
        env = envelope([bolt(0, 0.55, -0.12)])
        rec = self.stage.process_frame(env)
        for k, v in env.correlation.items():
            self.assertEqual(rec["goal"][k], v, k)

    def test_wrong_envelope_version_raises(self):
        class V1:
            version = "eyetoarm.b_input/1"
        with self.assertRaises(ValueError):
            self.stage.process_frame(V1())

    def test_single_observation_process_unchanged(self):
        stage = BStage.from_config(ROOT, CAM, StaticTfLookup(T_W_C))
        self.assertTrue(callable(stage.process) and callable(stage.consume))  # 단계 T 입구 유지 (기존 시험이 동작 확인)


if __name__ == "__main__":
    unittest.main()
