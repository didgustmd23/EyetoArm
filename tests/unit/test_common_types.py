"""common/types.py 공통 입력 자료형과 inputs/base.py A 확장의 호환 시험.

근거: 공통 자료형 관리 초안 §3.1·§3.2·§3.6, 사용자 확정(2026-11-11: 공통 자료형은 초안 기준). 기대 필드 목록은 초안 표에서 옮긴 상수다.
"""
import dataclasses
import inspect
import os
import subprocess
import sys
import unittest

import numpy as np

from eyetoarm.common import types as common
from eyetoarm.inputs import base
from eyetoarm.vision import planar

# 초안 §3.2 표 순서(`width`, `height`는 한 행에 함께 적힌 두 필드)
COMMON_FRAME_FIELDS = [
    ("image", "np.ndarray"),
    ("run_id", "str"),
    ("source_id", "str"),
    ("image_seq", "int"),
    ("timestamp_ms", "int"),
    ("timestamp_base", "TimestampBase"),
    ("received_monotonic_ns", "int"),
    ("width", "int"),
    ("height", "int"),
    ("crop", "tuple[int, int, int, int] | None"),
    ("calibration_id", "str | None"),
    ("capture_generation", "int"),
    ("optical_frame", "str | None"),
    ("ros_stamp_ns", "int | None"),
]
# 초안 §3.6 표
COMMON_EVENT_FIELDS = [
    ("run_id", "str"),
    ("source_id", "str"),
    ("status", "InputStatus"),
    ("event_monotonic_ns", "int"),
    ("detail", "str | None"),
]
A_FRAME_EXTRA = [("source_ref", "str"), ("source_index", "int")]
A_EVENT_EXTRA = [("source_ref", "str | None")]


def field_list(cls):
    return [(f.name, f.type) for f in dataclasses.fields(cls)]


def no_defaults(testcase, cls):
    for f in dataclasses.fields(cls):
        testcase.assertIs(f.default, dataclasses.MISSING, f.name)
        testcase.assertIs(f.default_factory, dataclasses.MISSING, f.name)


def common_frame_kwargs(image):
    h, w = image.shape[:2]
    return {
        "image": image, "run_id": "run-c", "source_id": "src0", "image_seq": 4, "timestamp_ms": 99,
        "timestamp_base": common.TimestampBase.VIDEO_POS, "received_monotonic_ns": 1000,
        "width": w, "height": h, "crop": None, "calibration_id": None, "capture_generation": 2,
        "optical_frame": None, "ros_stamp_ns": None,
    }


class CommonFrameTest(unittest.TestCase):
    def test_fields_match_draft(self):
        self.assertEqual(field_list(common.FramePacket), COMMON_FRAME_FIELDS)
        self.assertEqual(len(dataclasses.fields(common.FramePacket)), 14)

    def test_no_defaults_and_frozen(self):
        no_defaults(self, common.FramePacket)
        f = common.FramePacket(**common_frame_kwargs(np.zeros((2, 3, 3), np.uint8)))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            f.image_seq = 5
        with self.assertRaises(TypeError):
            common.FramePacket(image=np.zeros((2, 3, 3), np.uint8))

    def test_no_a_only_fields(self):
        names = {f.name for f in dataclasses.fields(common.FramePacket)}
        self.assertNotIn("source_ref", names)
        self.assertNotIn("source_index", names)


class CommonEventTest(unittest.TestCase):
    def test_fields_match_draft(self):
        self.assertEqual(field_list(common.InputStatusEvent), COMMON_EVENT_FIELDS)
        names = {n for n, _ in COMMON_EVENT_FIELDS}
        for absent in ("image_seq", "image", "timestamp_ms", "source_ref"):
            self.assertNotIn(absent, names)

    def test_no_defaults_frozen_detail_none(self):
        no_defaults(self, common.InputStatusEvent)
        e = common.InputStatusEvent(run_id="r", source_id="s", status=common.InputStatus.END_OF_STREAM,
                                    event_monotonic_ns=1, detail=None)
        self.assertIsNone(e.detail)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            e.detail = "x"


class EnumIdentityTest(unittest.TestCase):
    def test_same_enum_classes(self):
        self.assertIs(base.TimestampBase, common.TimestampBase)
        self.assertIs(base.InputStatus, common.InputStatus)

    def test_values_unchanged(self):
        self.assertEqual([m.value for m in common.TimestampBase],
                         ["PC_RECEIVE", "PHONE_CAPTURE", "SIM_TIME", "VIDEO_POS"])
        self.assertEqual([m.value for m in common.InputStatus],
                         ["SOURCE_UNAVAILABLE", "FRAME_TIMEOUT", "DECODE_FAILED", "END_OF_STREAM"])
        self.assertEqual(common.TimestampBase.SIM_TIME, "SIM_TIME")


class AExtensionTest(unittest.TestCase):
    def test_frame_extends_common(self):
        self.assertTrue(issubclass(base.FramePacket, common.FramePacket))
        self.assertIsNot(base.FramePacket, common.FramePacket)
        self.assertEqual(field_list(base.FramePacket), COMMON_FRAME_FIELDS + A_FRAME_EXTRA)
        no_defaults(self, base.FramePacket)
        params = list(inspect.signature(base.FramePacket).parameters)
        self.assertEqual(params, [n for n, _ in COMMON_FRAME_FIELDS + A_FRAME_EXTRA])

    def test_frame_instance(self):
        f = base.FramePacket(**common_frame_kwargs(np.zeros((2, 3, 3), np.uint8)),
                             source_ref="a.jpg", source_index=0)
        self.assertIsInstance(f, common.FramePacket)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            f.source_ref = "b.jpg"

    def test_event_extends_common(self):
        self.assertTrue(issubclass(base.InputStatusEvent, common.InputStatusEvent))
        self.assertEqual(field_list(base.InputStatusEvent), COMMON_EVENT_FIELDS + A_EVENT_EXTRA)
        no_defaults(self, base.InputStatusEvent)
        params = list(inspect.signature(base.InputStatusEvent).parameters)
        self.assertEqual(params, [n for n, _ in COMMON_EVENT_FIELDS + A_EVENT_EXTRA])
        e = base.InputStatusEvent("r", "s", base.InputStatus.DECODE_FAILED, 5, "bad", None)
        self.assertIsInstance(e, common.InputStatusEvent)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            e.source_ref = "x"

    def test_a_helpers_stay_in_inputs(self):
        self.assertTrue(hasattr(base, "FrameIdIssuer"))
        self.assertTrue(hasattr(base, "new_run_id"))
        self.assertFalse(hasattr(common, "FrameIdIssuer"))
        self.assertFalse(hasattr(common, "new_run_id"))


class PlanarWithCommonFrameTest(unittest.TestCase):
    def setUp(self):
        self.img = np.full((80, 100, 3), 255, np.uint8)
        self.img[10:30, 20:40] = 0       # 화소 20..39 × 10..29, 넓이 19*19 = 361
        self.config = planar.PlanarConfig(
            format_version=1, status="provisional_display_only", roi=None, resize_wh=None,
            resize_interpolation="INTER_AREA", gaussian_blur_ksize=0, polarity="dark",
            threshold_method="fixed", fixed_threshold=128, morph_open_ksize=0,
            contour_retrieval="RETR_EXTERNAL", contour_approximation="CHAIN_APPROX_SIMPLE",
            min_area_source_px2=100, max_area_source_px2=None, border_margin_processed_px=2)

    def test_planar_references_common_frame(self):
        self.assertIs(planar.FramePacket, common.FramePacket)

    def test_common_frame_without_a_fields(self):
        f = common.FramePacket(**common_frame_kwargs(self.img))
        r = planar.detect_planar(f, self.config)
        self.assertEqual((r.run_id, r.image_seq, r.capture_generation), ("run-c", 4, 2))
        self.assertEqual(r.raw_contour_count, 1)
        self.assertEqual(r.candidates[0].area_source_px2, 361.0)
        self.assertEqual(r.candidates[0].bbox_source, (20.0, 10.0, 39.0, 29.0))

    def test_a_frame_gives_same_result(self):
        rc = planar.detect_planar(common.FramePacket(**common_frame_kwargs(self.img)), self.config)
        ra = planar.detect_planar(base.FramePacket(**common_frame_kwargs(self.img),
                                                   source_ref="x", source_index=0), self.config)
        self.assertEqual(rc.raw_contour_count, ra.raw_contour_count)
        self.assertEqual(rc.candidates[0].bbox_source, ra.candidates[0].bbox_source)
        self.assertEqual(rc.config_sha256, ra.config_sha256)


class ImportIsolationTest(unittest.TestCase):
    def test_common_import_loads_no_qt_ros_cv2(self):
        code = (
            "import sys, eyetoarm.common.types, eyetoarm.inputs.base\n"
            "bad = [m for m in ('cv2', 'PySide6', 'PyQt5', 'PyQt6', 'rclpy', 'cv_bridge') if m in sys.modules]\n"
            "print(bad)\n"
            "sys.exit(1 if bad else 0)\n"
        )
        src = os.path.join(os.path.dirname(__file__), "..", "..", "src")
        env = dict(os.environ, PYTHONPATH=os.path.abspath(src), PYTHONDONTWRITEBYTECODE="1")
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertEqual(out.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
