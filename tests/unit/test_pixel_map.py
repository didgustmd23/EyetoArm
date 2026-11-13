"""vision/pixel_map.py ROI·리사이즈 원본 픽셀 복원 단위시험.

근거: A 상세설계 §5.1(ROI·표시 축소 복원), 공통 자료형 관리 초안 §3.1
(PixelToSourceTransform = (scale_x, scale_y, offset_x, offset_y)).
기대값은 손으로 계산한 상수다. 좌표는 raw 픽셀이며 단순 배율 복원은 A 내부 잠정 규약이다.
"""
import unittest

import numpy as np

from eyetoarm.vision.pixel_map import (
    compose,
    from_source,
    inside_source,
    resize_transform,
    roi_transform,
    to_source,
    validate_roi,
)

SRC_WH = (640, 480)


def pts(*xy):
    """(x, y) 쌍들을 (N, 2) float32 배열로 만든다."""
    return np.array(xy, dtype=np.float32).reshape(-1, 2)


class ForwardMappingTest(unittest.TestCase):
    """처리 영상 좌표 → 원본 좌표 정방향 복원을 손 계산 값과 비교한다(A 상세설계 §5.1)."""

    def test_pixel_map_roi_offset(self):
        """ROI만 쓰면 배율 1에 ROI 원점만큼 더해 원본 좌표가 된다."""
        t = roi_transform((100, 50, 200, 100), SRC_WH)

        self.assertEqual(t, (1.0, 1.0, 100.0, 50.0))

        out = to_source(pts((0, 0), (199, 99)), t)

        np.testing.assert_array_equal(out, pts((100, 50), (299, 149)))

    def test_pixel_map_anisotropic_resize(self):
        """가로·세로 배율이 다른 축소(x 2, y 4)를 축별로 되돌리고, 두 배율을 바꿔 쓰지 않는다."""
        t = resize_transform((640, 480), (320, 120))  # x 2.0, y 4.0

        self.assertEqual(t, (2.0, 4.0, 0.0, 0.0))
        np.testing.assert_array_equal(to_source(pts((10, 10)), t), pts((20, 40)))

        # 배율을 바꿔 쓰면 (40, 20)이 되어야 하므로 위 결과와 다르다
        self.assertFalse(np.array_equal(to_source(pts((10, 10)), t), pts((40, 20))))

    def test_pixel_map_roi_then_resize_order(self):
        """ROI 뒤 축소한 영상은 축소를 먼저 되돌리고 ROI 오프셋을 더한다. 합성 순서를 뒤집으면 다른 값이 된다."""
        # 원본 640x480 → ROI (100,50,200,100) 크롭 → 100x25로 축소(배율 x 2, y 4)
        roi = roi_transform((100, 50, 200, 100), SRC_WH)
        resize = resize_transform((200, 100), (100, 25))

        t = compose(roi, resize)  # resize를 먼저 되돌리고 ROI 오프셋을 더함

        self.assertEqual(t, (2.0, 4.0, 100.0, 50.0))

        # 손 계산: (10*2+100, 5*4+50) = (120, 70)
        np.testing.assert_array_equal(to_source(pts((10, 5)), t), pts((120, 70)))

        # 순서를 뒤집으면 ((10+100)*2, (5+50)*4) = (220, 220)으로 달라야 함
        wrong = compose(resize, roi)

        np.testing.assert_array_equal(to_source(pts((10, 5)), wrong), pts((220, 220)))

    def test_compose_formula(self):
        """합성은 배율을 곱하고 오프셋은 outer 배율×inner 오프셋 + outer 오프셋이다."""
        outer = (2.0, 3.0, 5.0, 7.0)
        inner = (0.5, 4.0, 10.0, -1.0)

        # 배율 곱, 오프셋 = outer.scale*inner.offset + outer.offset
        self.assertEqual(compose(outer, inner), (1.0, 12.0, 25.0, 4.0))


class RoundTripTest(unittest.TestCase):
    """정방향·역방향 변환이 서로 되돌리는지 확인한다."""

    def test_pixel_map_round_trip(self):
        """ROI·축소 합성 변환으로 원본에 갔다가 돌아오면 소수 좌표까지 1e-4 안에서 제자리다."""
        t = compose(roi_transform((100, 50, 200, 100), SRC_WH),
                    resize_transform((200, 100), (64, 30)))
        p = pts((0, 0), (10.25, 5.5), (63.75, 29.9))

        back = from_source(to_source(p, t), t)

        np.testing.assert_allclose(back, p, atol=1e-4, rtol=0)

    def test_from_source_known_value(self):
        """역방향 변환이 손으로 계산한 값((120, 70) → (10, 5))과 같다."""
        # 정방향 상수와 독립된 역방향 손 계산 값
        t = (2.0, 4.0, 100.0, 50.0)
        np.testing.assert_array_equal(from_source(pts((120, 70)), t), pts((10, 5)))


class ValidationTest(unittest.TestCase):
    """잘못된 ROI·해상도·변환·점 입력을 ValueError로 거부하는지 확인한다."""

    def test_pixel_map_invalid_inputs(self):
        """음수·크기 0·원본 밖·정수 아닌 ROI와 잘못된 원본 크기는 ValueError이고, 원본 경계에 딱 맞는 ROI는 받는다."""
        bad_rois = [
            (-1, 0, 10, 10),     # x 음수
            (0, -1, 10, 10),     # y 음수
            (0, 0, 0, 10),       # 너비 0
            (0, 0, 10, 0),       # 높이 0
            (600, 0, 41, 10),    # x + w = 641 > 640
            (0, 470, 10, 11),    # y + h = 481 > 480
            (0.5, 0, 10, 10),    # 정수 아님
        ]

        for roi in bad_rois:
            with self.assertRaises(ValueError, msg=str(roi)):
                validate_roi(roi, SRC_WH)

            with self.assertRaises(ValueError, msg=str(roi)):
                roi_transform(roi, SRC_WH)

        validate_roi((630, 470, 10, 10), SRC_WH)  # 경계와 딱 맞음

        self.assertEqual(roi_transform((630, 470, 10, 10), SRC_WH), (1.0, 1.0, 630.0, 470.0))

        for wh in [(0, 480), (640, -1), (640.0, 480)]:
            with self.assertRaises(ValueError, msg=str(wh)):
                validate_roi((0, 0, 1, 1), wh)

    def test_resize_invalid_resolution(self):
        """축소 전후 해상도에 0·음수·정수 아닌 값이 있으면 ValueError다."""
        for src, dst in [((0, 480), (320, 240)), ((640, 480), (0, 240)),
                         ((640, -480), (320, 240)), ((640, 480), (320.5, 240))]:
            with self.assertRaises(ValueError, msg=f"{src}->{dst}"):
                resize_transform(src, dst)

    def test_invalid_transform_rejected(self):
        """배율이 0·음수·비유한이거나 오프셋이 비유한인 변환은 정방향·역방향·합성 어디서나 ValueError다."""
        p = pts((1, 1))
        bad = [(0.0, 1.0, 0.0, 0.0), (1.0, -2.0, 0.0, 0.0), (float("nan"), 1.0, 0.0, 0.0),
               (1.0, float("inf"), 0.0, 0.0), (1.0, 1.0, float("nan"), 0.0),
               (1.0, 1.0, 0.0, float("-inf"))]
        good = (1.0, 1.0, 0.0, 0.0)

        for t in bad:
            for fn in (lambda: to_source(p, t), lambda: from_source(p, t),
                       lambda: compose(t, good), lambda: compose(good, t)):
                with self.assertRaises(ValueError, msg=str(t)):
                    fn()

    def test_invalid_points_rejected(self):
        """모양이 (N, 2)가 아니거나 NaN·Inf가 있는 점 배열은 변환·범위 판정 모두 ValueError다."""
        t = (1.0, 1.0, 0.0, 0.0)
        bad_points = [
            np.zeros((3,), dtype=np.float32),
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((1, 2, 1), dtype=np.float32),
            pts((float("nan"), 0)),
            pts((0, float("inf"))),
        ]

        for p in bad_points:
            for fn in (to_source, from_source):
                with self.assertRaises(ValueError):
                    fn(p, t)

            with self.assertRaises(ValueError):
                inside_source(p, 640, 480)


class InsideAndArrayTest(unittest.TestCase):
    """원본 안쪽 판정과 출력 배열의 형·복사 규칙을 확인한다."""

    def test_pixel_map_outside_flag(self):
        """원본 안쪽은 0 ≤ x < 폭, 0 ≤ y < 높이이고(소수 좌표 포함), 결과는 (N,) bool 배열이다."""
        p = pts((0, 0), (639, 479), (-1, 0), (0, -0.5), (640, 0), (0, 480), (639.9, 479.9))

        np.testing.assert_array_equal(
            inside_source(p, 640, 480),
            np.array([True, True, False, False, False, False, True]),
        )
        self.assertEqual(inside_source(p, 640, 480).dtype, np.bool_)
        self.assertEqual(inside_source(p, 640, 480).shape, (7,))

    def test_input_preserved_and_float32_output(self):
        """변환은 입력 배열을 바꾸지 않고 메모리를 공유하지 않는 새 float32 배열을 돌려준다(항등 변환에서도)."""
        t = (2.0, 4.0, 100.0, 50.0)
        p64 = np.array([[10.0, 5.0], [1.0, 2.0]], dtype=np.float64)
        keep = p64.copy()
        out = to_source(p64, t)

        back = from_source(out, t)

        np.testing.assert_array_equal(p64, keep)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(back.dtype, np.float32)
        self.assertEqual(out.shape, (2, 2))
        self.assertFalse(np.shares_memory(out, p64))

        p32 = pts((10, 5))
        out32 = to_source(p32, (1.0, 1.0, 0.0, 0.0))

        self.assertFalse(np.shares_memory(out32, p32))

        out32[0, 0] = 999.0

        self.assertEqual(float(p32[0, 0]), 10.0)

    def test_empty_points(self):
        """빈 점 배열은 오류 없이 빈 (0, 2) float32 결과와 빈 (0,) bool 판정을 돌려준다."""
        t = (2.0, 4.0, 100.0, 50.0)
        empty = np.zeros((0, 2), dtype=np.float32)

        self.assertEqual(to_source(empty, t).shape, (0, 2))
        self.assertEqual(to_source(empty, t).dtype, np.float32)
        self.assertEqual(from_source(empty, t).shape, (0, 2))

        flags = inside_source(empty, 640, 480)

        self.assertEqual(flags.shape, (0,))
        self.assertEqual(flags.dtype, np.bool_)


if __name__ == "__main__":
    unittest.main()
