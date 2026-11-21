"""검출·자료 계약 수정 시험: 폭 0 박스 추적, v2 봉투 TF 부모 대조, 거대 정수 거부, 끝점 오류 격리."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from eyetoarm.vision import ai_convert
from eyetoarm.vision.ai_result import AXIS_EXTRACTION_ERROR, AiInstance, AiSegResult, axis_candidate_reason
from eyetoarm.vision.list_envelope import ListEnvelopeError, make_list_envelope
from eyetoarm.vision.tracking import FrameKey, PartTracker, PartTrackIdIssuer, TrackMode, inputs_from_ai
from test_ai_axis_candidates import SCENARIOS, a8_like_small_frame
from test_ai_axis_path import H, W, axis_cfg
from test_ai_convert import raw
from test_frame_observations import frame_obs, tf

BASE = dict(index=0, class_index=0, class_name="bolt", confidence=0.9, box_xyxy=(10.0, 10.0, 30.0, 20.0),
            polygon=np.array([[10, 10], [30, 10], [30, 20]], np.float32), area_px=100.0, component_count=1,
            hole_count=0, clipped=False, rejected=None)


class ZeroWidthBoxTrackingTest(unittest.TestCase):
    """AiInstance가 허용하는 폭 0 박스를 추적기도 받는다(한 번으로 추적이 꺼지지 않음)."""

    def test_zero_width_box_accepted_by_tracker(self):
        """x1 == x2 박스가 섞인 결과로 추적을 갱신해도 TrackingInputError가 나지 않고 두 인스턴스 모두 배정된다."""
        thin = AiInstance(**{**BASE, "box_xyxy": (10.0, 10.0, 10.0, 20.0)})
        ok = AiInstance(**{**BASE, "index": 1})
        res = AiSegResult(instances=(thin, ok), model_id="m", config_sha256="a" * 64, roi=None, crop_wh=(64, 48),
                          infer_ms=1.0, post_ms=1.0, process_ms=2.0)
        tracker = PartTracker(PartTrackIdIssuer(), mode=TrackMode.HEURISTIC)
        frame = tracker.update(FrameKey("r", "s", 0, 0, 0, 0, "a" * 64), 100, inputs_from_ai(res))

        self.assertEqual(len(frame.assignments), 2)

    def test_inverted_box_still_rejected(self):
        """x2 < x1처럼 뒤집힌 박스는 두 쪽 모두 여전히 거부한다(AiInstance가 먼저 ValueError)."""
        with self.assertRaises(ValueError):
            AiInstance(**{**BASE, "box_xyxy": (10.0, 10.0, 9.0, 20.0)})


class ListEnvelopeWorldFrameTest(unittest.TestCase):
    """v2 봉투도 v1처럼 TF 부모 = world를 생성 때 대조한다."""

    def test_parent_must_be_world(self):
        """부모가 map인 목록은 ListEnvelopeError, world는 통과, world_frame 인자로 다른 이름을 줄 수 있다."""
        with self.assertRaises(ListEnvelopeError):
            make_list_envelope(frame_obs(tf=tf(parent="map")))

        self.assertEqual(make_list_envelope(frame_obs()).observations.tf.parent, "world")
        self.assertEqual(make_list_envelope(frame_obs(tf=tf(parent="map")), world_frame="map").observations.tf.parent,
                         "map")

    def test_tf_none_allowed(self):
        """TF가 없는 목록(B가 프레임 전체 TF_UNAVAILABLE)은 부모 대조 없이 통과한다."""
        self.assertIsNone(make_list_envelope(frame_obs(tf=None)).observations.tf)


class HugeIntegerTest(unittest.TestCase):
    """AiInstance 실수 필드의 거대 정수는 OverflowError가 아니라 ValueError."""

    def test_huge_int_is_value_error(self):
        """confidence·area_px·box·axis_max_iou에 10**400을 넣으면 ValueError."""
        for field, value in (("confidence", 10 ** 400), ("area_px", 10 ** 400),
                             ("box_xyxy", (0.0, 0.0, 10 ** 400, 1.0)), ("axis_max_iou", 10 ** 400)):
            with self.subTest(field=field):
                with self.assertRaises(ValueError) as cm:
                    AiInstance(**{**BASE, field: value})

                self.assertNotIsInstance(cm.exception, OverflowError)


class EndpointContractTest(unittest.TestCase):
    """잘못된 끝점은 물체 하나의 제외 사유가 되고, 같은 프레임의 다른 정상 볼트는 남는다."""

    def test_instance_rejects_same_point_endpoints(self):
        """AiInstance는 공통형과 같은 규칙으로 두 점이 같은 끝점·bool 배열을 거부한다."""
        for points in (np.array([[5.0, 5.0], [5.0, 5.0]]), np.array([[True, False], [False, True]])):
            with self.subTest(points=points.tolist()):
                with self.assertRaises(ValueError):
                    AiInstance(**{**BASE, "axis_endpoints_px": points})

    def run_with_bad_first(self, bad_points):
        """두 볼트 중 첫 볼트의 추출기 끝점만 bad_points(crop 좌표)로 바꿔 convert한다."""
        real = ai_convert.extract_m1
        calls = []

        def fake(mask):
            """첫 호출은 잘못된 끝점, 그 뒤는 실제 M1."""
            calls.append(1)

            if len(calls) == 1:
                est = real(mask)
                return SimpleNamespace(valid=True, reason=None, endpoints_px=np.array(bad_points, np.float64),
                                       diag=est.diag)

            return real(mask)

        frame = a8_like_small_frame()
        cfg = axis_cfg()

        with mock.patch.object(ai_convert, "extract_m1", fake):
            return ai_convert.convert(raw(SCENARIOS["R4_two_valid"][0], (W, H)), frame, cfg)

    def test_bad_endpoints_become_per_object_reason(self):
        """두 점 같음·프레임 밖 끝점은 그 볼트만 AXIS_EXTRACTION_ERROR이고 다른 볼트는 유효로 남는다."""
        for bad in ([[5.0, 5.0], [5.0, 5.0]], [[-3.0, 5.0], [40.0, 5.0]], [[5.0, 5.0], [float(W), 5.0]]):
            with self.subTest(bad=bad):
                res = self.run_with_bad_first(bad)
                first, second = res.instances[0], res.instances[1]

                self.assertIsNone(first.axis_endpoints_px)
                self.assertEqual(first.axis_reason, AXIS_EXTRACTION_ERROR)
                self.assertIn("endpoints_rejected", first.axis_diag)
                self.assertIsNone(axis_candidate_reason(second))
                self.assertEqual(res.axis_valid_indices, (1,))


if __name__ == "__main__":
    unittest.main()
