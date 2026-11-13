"""vision/planar.py 설정 로더·표시용 윤곽 추출 단위시험.

근거: A 상세설계 §5.1(원본 픽셀 복원)·§7(평면 검출 처리, 설정 식별자), vision/planar.py의 설정
필드·처리 순서·식별자 정규화(A 내부 잠정안), vision/pixel_map.py(단순 스케일 복원 잠정 규약).

- 입력은 NumPy로 그린 합성 영상이다. 기대 좌표·면적·경계 상자는 손으로 계산한 상수이며
  OpenCV 처리 결과로 기대값을 만들지 않는다.
- `cv2.findContours`의 직사각형 윤곽은 화소 중심 꼭짓점이다. 화소 x0..x1, y0..y1을 채운
  직사각형의 넓이는 (x1 - x0) * (y1 - y0)이다.
- 결과는 표시용이며 B 계산용 유효 결과·클래스·규격 판정이 아니다.
"""
import dataclasses
import json
import os
import tempfile
import unittest

import numpy as np

from eyetoarm.inputs.base import FramePacket, TimestampBase
from eyetoarm.vision.planar import PlanarConfig, PlanarResult, detect_planar, load_config

REPO_CONFIG = os.path.join(os.path.dirname(__file__), "..", "..", "config", "vision", "opencv_planar.json")

BASE = {
    "format_version": 1,
    "status": "provisional_display_only",
    "roi": None,
    "resize_wh": None,
    "resize_interpolation": "INTER_AREA",
    "gaussian_blur_ksize": 0,
    "polarity": "dark",
    "threshold_method": "fixed",
    "fixed_threshold": 128,
    "morph_open_ksize": 0,
    "contour_retrieval": "RETR_EXTERNAL",
    "contour_approximation": "CHAIN_APPROX_SIMPLE",
    "min_area_source_px2": 100,
    "max_area_source_px2": None,
    "border_margin_processed_px": 2,
}


def cfg(**over):
    fields = dict(BASE)
    fields.update(over)
    return PlanarConfig(**fields)


def frame(image, **over):
    h, w = image.shape[:2] if image.ndim >= 2 else (0, 0)
    fields = {
        "image": image, "run_id": "run-test", "source_id": "src0", "image_seq": 7, "timestamp_ms": 1234,
        "timestamp_base": TimestampBase.VIDEO_POS, "received_monotonic_ns": 555, "width": w, "height": h,
        "crop": None, "calibration_id": None, "capture_generation": 3, "optical_frame": None,
        "ros_stamp_ns": None, "source_ref": "synthetic", "source_index": 0,
    }
    fields.update(over)
    return FramePacket(**fields)


def canvas(w, h, value):
    return np.full((h, w, 3), value, dtype=np.uint8)


def fill(img, x0, y0, x1, y1, value):
    """화소 x0..x1, y0..y1(양끝 포함)을 채운다."""
    img[y0:y1 + 1, x0:x1 + 1] = value
    return img


class TempJson:
    def __init__(self, text):
        self.text = text

    def __enter__(self):
        fd, self.path = tempfile.mkstemp(suffix=".json")

        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(self.text)

        return self.path

    def __exit__(self, *exc):
        os.unlink(self.path)


def load_text(text):
    with TempJson(text) as p:
        return load_config(p)


def load_dict(d):
    return load_text(json.dumps(d))


# ---------------------------------------------------------------- 설정 로더


class ConfigLoadTest(unittest.TestCase):
    def test_repo_config_loads(self):
        c = load_config(REPO_CONFIG)

        self.assertEqual(c.polarity, "dark")
        self.assertEqual(c.threshold_method, "otsu")
        self.assertIsNone(c.fixed_threshold)
        self.assertIsNone(c.roi)
        self.assertIsNone(c.resize_wh)
        self.assertEqual(c.min_area_source_px2, 100.0)
        self.assertIsInstance(c.min_area_source_px2, float)
        self.assertEqual(c.border_margin_processed_px, 2)

        # 저장소 설정 config/vision/opencv_planar.json의 식별자(설정을 바꾸면 이 값도 바꾼다)
        self.assertEqual(c.config_sha256, "b54dec5729bf5dc4f5af1ae0921ede34a5db3124050e9dfe6c110571c1e51422")

    def test_pathlike_accepted(self):
        import pathlib
        self.assertEqual(load_config(pathlib.Path(REPO_CONFIG)).polarity, "dark")

    def test_same_hash_for_key_order_whitespace_int_float(self):
        a = load_dict(BASE)
        reordered = dict(reversed(list(BASE.items())))
        b = load_text(json.dumps(reordered, indent=4))

        c = load_dict({**BASE, "min_area_source_px2": 100.0, "fixed_threshold": 128.0})

        self.assertEqual(a.config_sha256, b.config_sha256)
        self.assertEqual(a.config_sha256, c.config_sha256)
        self.assertEqual(len(a.config_sha256), 64)

    def test_value_change_changes_hash(self):
        a = load_dict(BASE)

        for key, value in [("min_area_source_px2", 101), ("polarity", "bright"),
                           ("border_margin_processed_px", 3), ("roi", [0, 0, 10, 10]),
                           ("fixed_threshold", 127.5)]:
            with self.subTest(key=key):
                self.assertNotEqual(a.config_sha256, load_dict({**BASE, key: value}).config_sha256)

    def test_direct_construction_matches_load(self):
        self.assertEqual(cfg().config_sha256, load_dict(BASE).config_sha256)

    def test_sequences_stored_as_tuples(self):
        c = load_dict({**BASE, "roi": [1, 2, 30, 40], "resize_wh": [15, 20]})

        self.assertEqual(c.roi, (1, 2, 30, 40))
        self.assertEqual(c.resize_wh, (15, 20))
        self.assertIsInstance(c.roi, tuple)

    def test_config_is_frozen(self):
        c = cfg()

        with self.assertRaises(dataclasses.FrozenInstanceError):
            c.polarity = "bright"

        with self.assertRaises(dataclasses.FrozenInstanceError):
            c.config_sha256 = "x"

    def test_external_list_change_does_not_affect_config(self):
        roi = [1, 2, 30, 40]
        c = cfg(roi=roi)

        roi[0] = 99

        self.assertEqual(c.roi, (1, 2, 30, 40))

    def test_missing_file_is_oserror(self):
        with self.assertRaises(OSError):
            load_config("/nonexistent/dir/opencv_planar.json")


class ConfigRejectTest(unittest.TestCase):
    def assert_rejected(self, d):
        with self.assertRaises(ValueError):
            load_dict(d)

    def test_missing_key(self):
        for key in BASE:
            with self.subTest(key=key):
                d = dict(BASE)
                del d[key]

                self.assert_rejected(d)

    def test_unknown_key(self):
        self.assert_rejected({**BASE, "extra": 1})

    def test_duplicate_key(self):
        text = json.dumps(BASE)[:-1] + ', "polarity": "bright"}'
        with self.assertRaises(ValueError):
            load_text(text)

    def test_nan_infinity_and_overflow(self):
        for bad in ["NaN", "Infinity", "-Infinity", "1e400"]:
            with self.subTest(bad=bad):
                text = json.dumps({**BASE, "min_area_source_px2": 0}).replace(
                    '"min_area_source_px2": 0', '"min_area_source_px2": ' + bad)

                with self.assertRaises(ValueError):
                    load_text(text)

        self.assert_rejected({**BASE, "min_area_source_px2": 10 ** 400})

    def test_not_object(self):
        with self.assertRaises(ValueError):
            load_text("[1, 2]")

        with self.assertRaises(ValueError):
            load_text("{not json")

    def test_bool_as_number(self):
        for key in ["format_version", "gaussian_blur_ksize", "morph_open_ksize",
                    "border_margin_processed_px", "min_area_source_px2", "fixed_threshold"]:
            with self.subTest(key=key):
                self.assert_rejected({**BASE, key: True})

        self.assert_rejected({**BASE, "roi": [True, 0, 10, 10]})
        self.assert_rejected({**BASE, "resize_wh": [10, True]})

    def test_wrong_types(self):
        cases = [
            ("format_version", 1.0), ("format_version", 2), ("status", "final"),
            ("gaussian_blur_ksize", 3.0), ("morph_open_ksize", "3"),
            ("border_margin_processed_px", 2.0), ("border_margin_processed_px", -1),
            ("min_area_source_px2", "100"), ("max_area_source_px2", "x"),
            ("polarity", 1), ("roi", "0,0,1,1"), ("roi", {"x": 0}), ("resize_wh", [10.0, 10]),
        ]

        for key, value in cases:
            with self.subTest(key=key, value=value):
                self.assert_rejected({**BASE, key: value})

    def test_enums_and_fixed_constants(self):
        cases = [
            ("polarity", "auto"), ("polarity", "Dark"), ("threshold_method", "adaptive"),
            ("resize_interpolation", "INTER_LINEAR"), ("contour_retrieval", "RETR_TREE"),
            ("contour_approximation", "CHAIN_APPROX_NONE"),
        ]

        for key, value in cases:
            with self.subTest(key=key, value=value):
                self.assert_rejected({**BASE, key: value})

    def test_kernel_sizes(self):
        for key in ["gaussian_blur_ksize", "morph_open_ksize"]:
            for bad in [1, 2, 4, -3]:
                with self.subTest(key=key, bad=bad):
                    self.assert_rejected({**BASE, key: bad})

            for good in [0, 3, 5]:
                with self.subTest(key=key, good=good):
                    load_dict({**BASE, key: good})

    def test_roi_and_resize_shape(self):
        for bad in [[0, 0, 10], [0, 0, 10, 10, 1], [-1, 0, 10, 10], [0, -1, 10, 10],
                    [0, 0, 0, 10], [0, 0, 10, 0], [0, 0, 1.5, 10]]:
            with self.subTest(roi=bad):
                self.assert_rejected({**BASE, "roi": bad})

        for bad in [[10], [0, 10], [10, -1], [10, 10, 1]]:
            with self.subTest(resize=bad):
                self.assert_rejected({**BASE, "resize_wh": bad})

    def test_area_bounds(self):
        for bad in [0, -1]:
            with self.subTest(min=bad):
                self.assert_rejected({**BASE, "min_area_source_px2": bad})

        self.assert_rejected({**BASE, "max_area_source_px2": 100})   # max == min
        self.assert_rejected({**BASE, "max_area_source_px2": 50})    # max < min

        load_dict({**BASE, "max_area_source_px2": 100.5})

    def test_threshold_combination(self):
        self.assert_rejected({**BASE, "threshold_method": "otsu", "fixed_threshold": 128})
        self.assert_rejected({**BASE, "threshold_method": "fixed", "fixed_threshold": None})
        self.assert_rejected({**BASE, "fixed_threshold": -0.1})
        self.assert_rejected({**BASE, "fixed_threshold": 255.1})

        load_dict({**BASE, "fixed_threshold": 0})
        load_dict({**BASE, "fixed_threshold": 255})
        load_dict({**BASE, "threshold_method": "otsu", "fixed_threshold": None})

    def test_direct_construction_validates(self):
        with self.assertRaises(ValueError):
            cfg(polarity="auto")

        with self.assertRaises(ValueError):
            cfg(gaussian_blur_ksize=2)

        with self.assertRaises(ValueError):
            cfg(min_area_source_px2=float("nan"))

        with self.assertRaises(TypeError):
            PlanarConfig(**{k: v for k, v in BASE.items() if k != "roi"})


# ---------------------------------------------------------------- 검출


class ThresholdTest(unittest.TestCase):
    def test_uniform_background_has_no_candidates(self):
        r = detect_planar(frame(canvas(100, 80, 255)), cfg())

        self.assertEqual(r.raw_contour_count, 0)
        self.assertEqual(r.candidates, ())
        self.assertEqual(r.eligible_count, 0)
        self.assertEqual(r.threshold_used, 128.0)

    def test_dark_object_on_white(self):
        img = fill(canvas(100, 80, 255), 20, 10, 39, 29, 0)
        r = detect_planar(frame(img), cfg(polarity="dark"))

        self.assertEqual(r.raw_contour_count, 1)

        c = r.candidates[0]

        self.assertEqual(c.area_source_px2, 361.0)        # 19 * 19
        self.assertEqual(c.bbox_source, (20.0, 10.0, 39.0, 29.0))
        self.assertTrue(c.eligible)
        self.assertEqual(c.border_sides, ())
        self.assertEqual(c.clip_reasons, ())
        self.assertIsNone(c.area_reject_reason)

    def test_bright_object_on_black(self):
        img = fill(canvas(100, 80, 0), 20, 10, 39, 29, 255)
        r = detect_planar(frame(img), cfg(polarity="bright"))

        self.assertEqual(r.raw_contour_count, 1)
        self.assertEqual(r.candidates[0].bbox_source, (20.0, 10.0, 39.0, 29.0))

    def test_wrong_polarity_gives_background_contour(self):
        img = fill(canvas(100, 80, 255), 20, 10, 39, 29, 0)
        r = detect_planar(frame(img), cfg(polarity="bright"))

        # 밝은 배경이 물체가 되어 영상 테두리 전체 윤곽 하나가 나온다(공개 표본에서 관찰한 실패 사례와 같은 형태).
        self.assertEqual(r.raw_contour_count, 1)

        c = r.candidates[0]

        self.assertEqual(c.bbox_source, (0.0, 0.0, 99.0, 79.0))
        self.assertEqual(c.border_sides, ("left", "top", "right", "bottom"))
        self.assertFalse(c.eligible)

    def test_fixed_threshold_boundary(self):
        img = fill(canvas(100, 80, 200), 20, 10, 39, 29, 100)

        # 고정 임계값 100: 화소 > 100 이 배경. 100인 물체 화소는 dark에서 물체(255)가 된다.
        self.assertEqual(detect_planar(frame(img), cfg(fixed_threshold=100)).raw_contour_count, 1)

        # 고정 임계값 99: 물체 화소 100 > 99 → 배경과 같은 쪽. 전체가 0이 되어 윤곽 없음.
        self.assertEqual(detect_planar(frame(img), cfg(fixed_threshold=99)).raw_contour_count, 0)

    def test_otsu_path(self):
        img = fill(canvas(100, 80, 200), 20, 10, 39, 29, 40)
        r = detect_planar(frame(img), cfg(threshold_method="otsu", fixed_threshold=None))

        self.assertGreaterEqual(r.threshold_used, 40.0)
        self.assertLess(r.threshold_used, 200.0)
        self.assertEqual(r.raw_contour_count, 1)
        self.assertEqual(r.candidates[0].bbox_source, (20.0, 10.0, 39.0, 29.0))


class FilterStageTest(unittest.TestCase):
    def test_gaussian_removes_isolated_pixel(self):
        img = fill(canvas(100, 80, 255), 20, 10, 39, 29, 0)
        img[60, 70] = 0

        off = detect_planar(frame(img), cfg(min_area_source_px2=0.5))

        self.assertEqual(off.raw_contour_count, 2)

        # 3×3 Gaussian(sigma 자동) 뒤 고립 화소는 191 근처 → 128 초과로 배경.
        # 직사각형 안쪽 모서리 화소는 112 근처, 바깥 이웃은 191 이상이라 윤곽이 그대로다.
        on = detect_planar(frame(img), cfg(min_area_source_px2=0.5, gaussian_blur_ksize=3))

        self.assertEqual(on.raw_contour_count, 1)
        self.assertEqual(on.candidates[0].bbox_source, (20.0, 10.0, 39.0, 29.0))

    def test_open_removes_thin_noise(self):
        img = fill(canvas(100, 80, 255), 20, 10, 39, 29, 0)
        fill(img, 60, 60, 61, 61, 0)      # 2×2 점
        fill(img, 70, 40, 90, 40, 0)      # 1화소 두께 선

        off = detect_planar(frame(img), cfg(min_area_source_px2=0.5))

        self.assertEqual(off.raw_contour_count, 3)

        on = detect_planar(frame(img), cfg(min_area_source_px2=0.5, morph_open_ksize=3))

        self.assertEqual(on.raw_contour_count, 1)
        self.assertEqual(on.candidates[0].bbox_source, (20.0, 10.0, 39.0, 29.0))
        self.assertEqual(on.candidates[0].area_source_px2, 361.0)


class AreaFilterTest(unittest.TestCase):
    def setUp(self):
        self.img = fill(canvas(100, 80, 255), 20, 10, 39, 29, 0)   # 넓이 361
        fill(self.img, 60, 60, 61, 61, 0)                          # 넓이 1

    def by_area(self, r):
        return {c.area_source_px2: c for c in r.candidates}

    def test_small_noise_rejected_with_reason(self):
        r = detect_planar(frame(self.img), cfg(min_area_source_px2=100))

        self.assertEqual(r.raw_contour_count, 2)
        self.assertEqual(len(r.candidates), 2)          # 탈락 윤곽도 보존

        a = self.by_area(r)

        self.assertEqual(a[1.0].area_reject_reason, "area_below_min")
        self.assertFalse(a[1.0].eligible)
        self.assertTrue(a[361.0].eligible)
        self.assertEqual(r.eligible_count, 1)

    def test_inclusive_bounds(self):
        r = detect_planar(frame(self.img), cfg(min_area_source_px2=361, max_area_source_px2=361.5))

        self.assertTrue(self.by_area(r)[361.0].eligible)

        r = detect_planar(frame(self.img), cfg(min_area_source_px2=1, max_area_source_px2=361))

        self.assertTrue(self.by_area(r)[361.0].eligible)
        self.assertTrue(self.by_area(r)[1.0].eligible)

    def test_below_and_above(self):
        r = detect_planar(frame(self.img), cfg(min_area_source_px2=361.5))

        self.assertEqual(self.by_area(r)[361.0].area_reject_reason, "area_below_min")

        r = detect_planar(frame(self.img), cfg(min_area_source_px2=1, max_area_source_px2=360))

        self.assertEqual(self.by_area(r)[361.0].area_reject_reason, "area_above_max")
        self.assertIsNone(self.by_area(r)[1.0].area_reject_reason)


class MultipleAndShapeTest(unittest.TestCase):
    def test_two_objects_kept_without_selection(self):
        img = fill(canvas(120, 80, 255), 10, 10, 29, 29, 0)
        fill(img, 60, 20, 99, 59, 0)

        r = detect_planar(frame(img), cfg())

        self.assertEqual(r.eligible_count, 2)
        self.assertIsNone(r.selected_index)
        self.assertEqual(sorted(c.area_source_px2 for c in r.candidates), [361.0, 1521.0])
        self.assertEqual([c.index for c in r.candidates], [0, 1])

    def test_round_and_square_not_rejected_by_aspect(self):
        img = canvas(160, 80, 255)
        yy, xx = np.mgrid[0:80, 0:160]
        img[(xx - 40) ** 2 + (yy - 40) ** 2 <= 20 ** 2] = 0     # 원(장축비 1)
        fill(img, 100, 20, 139, 59, 0)                           # 정사각형(장축비 1)

        r = detect_planar(frame(img), cfg())

        self.assertEqual(r.raw_contour_count, 2)
        self.assertEqual(r.eligible_count, 2)

        for c in r.candidates:
            self.assertIsNone(c.area_reject_reason)
            self.assertTrue(c.eligible)


class BorderTest(unittest.TestCase):
    def test_source_edge(self):
        img = fill(canvas(100, 80, 255), 0, 20, 19, 39, 0)
        c = detect_planar(frame(img), cfg()).candidates[0]

        self.assertEqual(c.border_sides, ("left",))
        self.assertEqual(c.clip_reasons, ("source_edge",))
        self.assertIsNone(c.area_reject_reason)
        self.assertFalse(c.eligible)

    def test_margin_boundary(self):
        img = fill(canvas(100, 80, 255), 2, 20, 21, 39, 0)    # x_min = 2 = m

        self.assertEqual(detect_planar(frame(img), cfg()).candidates[0].border_sides, ("left",))

        img = fill(canvas(100, 80, 255), 3, 20, 22, 39, 0)    # x_min = 3 > m

        self.assertEqual(detect_planar(frame(img), cfg()).candidates[0].border_sides, ())

        img = fill(canvas(100, 80, 255), 50, 20, 97, 39, 0)   # x_max = 97 = W-1-m

        self.assertEqual(detect_planar(frame(img), cfg()).candidates[0].border_sides, ("right",))

        img = fill(canvas(100, 80, 255), 50, 40, 69, 77, 0)   # y_max = 77 = H-1-m

        self.assertEqual(detect_planar(frame(img), cfg()).candidates[0].border_sides, ("bottom",))

    def test_inner_roi_edge(self):
        # ROI (10, 10, 80, 60): 처리 영상 왼쪽 변은 원본 x = 10(원본 안쪽).
        img = fill(canvas(100, 80, 255), 10, 30, 29, 49, 0)
        c = detect_planar(frame(img), cfg(roi=[10, 10, 80, 60])).candidates[0]

        self.assertEqual(c.border_sides, ("left",))
        self.assertEqual(c.clip_reasons, ("roi_edge",))
        self.assertFalse(c.eligible)

    def test_both_edges_preserved(self):
        # ROI (10, 0, 80, 80): 위 변은 원본 위 변과 같고, 왼쪽 변은 원본 안쪽이다.
        img = fill(canvas(100, 80, 255), 10, 0, 29, 19, 0)
        c = detect_planar(frame(img), cfg(roi=[10, 0, 80, 80])).candidates[0]

        self.assertEqual(c.border_sides, ("left", "top"))
        self.assertEqual(c.clip_reasons, ("source_edge", "roi_edge"))

    def test_roi_reaching_source_right_edge(self):
        # ROI (40, 0, 60, 80): 오른쪽 변 x + w = 100 = W → 원본 변.
        img = fill(canvas(100, 80, 255), 80, 30, 99, 49, 0)
        c = detect_planar(frame(img), cfg(roi=[40, 0, 60, 80])).candidates[0]

        self.assertEqual(c.border_sides, ("right",))
        self.assertEqual(c.clip_reasons, ("source_edge",))

    def test_roi_outside_source_rejected(self):
        with self.assertRaises(ValueError):
            detect_planar(frame(canvas(100, 80, 255)), cfg(roi=[50, 0, 60, 80]))


class RestoreTest(unittest.TestCase):
    def test_roi_and_anisotropic_resize(self):
        # 원본 200×100, ROI (40, 20, 120, 60), 리사이즈 (60, 20) → sx = 2, sy = 3, 오프셋 (40, 20).
        # ROI 안 x 20..59(40화소, 2의 배수 정렬), y 15..44(30화소, 3의 배수 정렬)를 검게 칠하면
        # INTER_AREA 정수배 축소는 블록 평균이라 처리 영상 x 10..29, y 5..14가 정확히 검다.
        img = canvas(200, 100, 255)
        fill(img, 40 + 20, 20 + 15, 40 + 59, 20 + 44, 0)

        r = detect_planar(frame(img), cfg(roi=[40, 20, 120, 60], resize_wh=[60, 20]))

        self.assertEqual(r.processed_wh, (60, 20))
        self.assertEqual(r.transform, (2.0, 3.0, 40.0, 20.0))
        self.assertEqual(r.raw_contour_count, 1)

        c = r.candidates[0]

        # 처리 꼭짓점 (10,5)-(29,14) → 원본 x = 2u + 40, y = 3v + 20 (단순 스케일 잠정 규약)
        self.assertEqual(c.bbox_source, (60.0, 35.0, 98.0, 62.0))
        self.assertEqual(c.area_source_px2, 1026.0)   # (98-60)*(62-35) = 19*9 처리 넓이 × 6
        self.assertEqual(c.points_source.dtype, np.float32)
        self.assertEqual(c.points_source.shape[1], 2)
        self.assertEqual(c.border_sides, ())

    def test_area_filter_uses_source_area(self):
        img = canvas(200, 100, 255)
        fill(img, 60, 35, 99, 64, 0)

        # 처리 넓이 171, 원본 넓이 1026. 하한 500은 원본 기준이면 통과, 처리 기준이면 탈락.
        r = detect_planar(frame(img), cfg(roi=[40, 20, 120, 60], resize_wh=[60, 20],
                                          min_area_source_px2=500))

        self.assertTrue(r.candidates[0].eligible)

    def test_full_frame_identity(self):
        img = fill(canvas(100, 80, 255), 20, 10, 39, 29, 0)
        r = detect_planar(frame(img), cfg())

        self.assertEqual(r.transform, (1.0, 1.0, 0.0, 0.0))
        self.assertEqual(r.processed_wh, (100, 80))


class ResultContractTest(unittest.TestCase):
    def setUp(self):
        self.img = fill(canvas(100, 80, 255), 20, 10, 39, 29, 0)
        fill(self.img, 60, 60, 61, 61, 0)
        self.config = cfg()

    def test_metadata_and_fixed_fields(self):
        r = detect_planar(frame(self.img), self.config)

        self.assertIsInstance(r, PlanarResult)
        self.assertEqual((r.run_id, r.source_id, r.image_seq, r.capture_generation),
                         ("run-test", "src0", 7, 3))
        self.assertEqual((r.timestamp_ms, r.timestamp_base, r.received_monotonic_ns),
                         (1234, TimestampBase.VIDEO_POS, 555))
        self.assertEqual(r.config_sha256, self.config.config_sha256)
        self.assertTrue(r.display_only)
        self.assertFalse(r.valid)
        self.assertIsNone(r.observed_face)
        self.assertIsNone(r.calc_contour)
        self.assertIsNone(r.selected_index)
        self.assertGreaterEqual(r.process_ms, 0.0)

    def test_fixed_fields_not_settable(self):
        # 고정 필드는 init=False라 생성 인자로 바꿀 수 없고 재할당도 막힌다.
        r = detect_planar(frame(self.img), self.config)
        kwargs = {f.name: getattr(r, f.name) for f in dataclasses.fields(PlanarResult) if f.init}

        for name, value in [("valid", True), ("display_only", False), ("selected_index", 0),
                            ("observed_face", "top"), ("calc_contour", [])]:
            with self.subTest(name=name):
                with self.assertRaises(TypeError):
                    PlanarResult(**kwargs, **{name: value})

                with self.assertRaises(dataclasses.FrozenInstanceError):
                    setattr(r, name, value)

    def test_repeat_is_identical_except_time(self):
        a = detect_planar(frame(self.img), self.config)
        b = detect_planar(frame(self.img), self.config)
        skip = {"candidates", "process_ms"}

        for f in dataclasses.fields(PlanarResult):
            if f.name not in skip:
                self.assertEqual(getattr(a, f.name), getattr(b, f.name), f.name)

        self.assertEqual(len(a.candidates), len(b.candidates))

        for ca, cb in zip(a.candidates, b.candidates):
            np.testing.assert_array_equal(ca.points_source, cb.points_source)
            self.assertEqual((ca.index, ca.area_source_px2, ca.bbox_source, ca.border_sides,
                              ca.clip_reasons, ca.area_reject_reason, ca.eligible),
                             (cb.index, cb.area_source_px2, cb.bbox_source, cb.border_sides,
                              cb.clip_reasons, cb.area_reject_reason, cb.eligible))

    def test_input_not_modified_and_read_only_ok(self):
        img = self.img.copy()
        img.setflags(write=False)
        before = img.copy()
        f = frame(img)

        r = detect_planar(f, self.config)

        np.testing.assert_array_equal(img, before)
        self.assertIs(f.image, img)
        self.assertEqual((f.width, f.height, f.image_seq), (100, 80, 7))
        self.assertEqual(r.raw_contour_count, 2)

    def test_result_points_read_only(self):
        c = detect_planar(frame(self.img), self.config).candidates[0]
        with self.assertRaises(ValueError):
            c.points_source[0, 0] = 1.0

    def test_candidate_order_is_index_order(self):
        r = detect_planar(frame(self.img), self.config)
        self.assertEqual([c.index for c in r.candidates], list(range(r.raw_contour_count)))


class BadFrameTest(unittest.TestCase):
    def test_bad_images(self):
        good = canvas(100, 80, 255)
        cases = {
            "float": good.astype(np.float32),
            "gray": good[:, :, 0].copy(),
            "bgra": np.full((80, 100, 4), 255, np.uint8),
            "empty": np.zeros((0, 0, 3), np.uint8),
        }

        for name, img in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    detect_planar(frame(img), cfg())

    def test_size_mismatch(self):
        with self.assertRaises(ValueError):
            detect_planar(frame(canvas(100, 80, 255), width=99), cfg())

        with self.assertRaises(ValueError):
            detect_planar(frame(canvas(100, 80, 255), height=81), cfg())

    def test_not_array(self):
        with self.assertRaises(ValueError):
            detect_planar(dataclasses.replace(frame(canvas(100, 80, 255)), image=[[0]]), cfg())

    def test_config_type(self):
        with self.assertRaises(TypeError):
            detect_planar(frame(canvas(100, 80, 255)), dict(BASE))


if __name__ == "__main__":
    unittest.main()
