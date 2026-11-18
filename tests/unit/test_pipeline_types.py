"""A 파이프라인 작업 1 자료형 단위시험(공통형 추가분과 A 내부 파이프라인 타입).

근거: 공통 자료형 관리 초안 §3.1(`PixelBasis`·`Contour`)·§3.3(`PlanarObservation`)·
§3.5(`CandidateInvalidationEvent`)·§4 불변조건 6·8, A 상세설계 §5.2·§5.4·§9,
계획 `docs/superpowers/plans/2026-11-15-a-vision-pipeline.md` 작업 1·「인터페이스 제안」.
디스크 이미지·ROS·Qt·실제 시계에 의존하지 않는다.
"""
import dataclasses
import unittest

import numpy as np

from eyetoarm.common import types as common
from eyetoarm.common.types import (
    CandidateInvalidationEvent,
    InvalidationKind,
    PixelBasis,
    PlanarObservation,
)
from eyetoarm.inputs import base
from eyetoarm.vision.planar import PlanarConfig
from eyetoarm.vision.pipeline_types import (
    InputStateRecord,
    JobContext,
    JobOutcome,
    JobStatus,
    PipelineCounters,
    PipelineSettings,
    ProcessRecord,
    QueueDropReason,
    QueueDropRecord,
    SourceProfile,
)

# 시험용 검출 설정(어두운 물체·고정 임계값 128). test_planar.py의 BASE와 같은 값이다
CONFIG_FIELDS = {
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


def _config(**over):
    """`CONFIG_FIELDS`에서 키워드로 준 값만 바꾼 `PlanarConfig`."""
    fields = dict(CONFIG_FIELDS)
    fields.update(over)

    return PlanarConfig(**fields)


def _common_frame_kwargs(**over):
    """공통 `FramePacket` 14개 필드를 모두 명시한 생성 인자."""
    fields = {
        "image": np.zeros((4, 6, 3), dtype=np.uint8),
        "run_id": "run-t",
        "source_id": "src0",
        "image_seq": 3,
        "timestamp_ms": 1234,
        "timestamp_base": common.TimestampBase.PC_RECEIVE,
        "received_monotonic_ns": 1_234_000_001,
        "width": 6,
        "height": 4,
        "crop": None,
        "calibration_id": None,
        "capture_generation": 2,
        "optical_frame": None,
        "ros_stamp_ns": None,
    }
    fields.update(over)

    return fields


def _observation_kwargs(**over):
    """무효 관측(사유 `NO_OBJECT`, 값이 없는 필드는 None) 생성 인자. 키워드로 필드를 바꿀 수 있다.

    `valid=True`로 바꾸면서 `reason`을 주지 않으면 사유를 None으로 둔다(유효 관측은 사유 없음).
    """
    fields = {
        "run_id": "run-t",
        "source_id": "src0",
        "image_seq": 3,
        "capture_generation": 2,
        "generation": 2,
        "job_seq": 5,
        "detector_config_id": "abc",
        "pixel_basis": PixelBasis.RAW,
        "calibration_id": None,
        "observed_face": None,
        "calc_contour": None,
        "display_mask_contour": None,
        "candidate_count": 0,
        "selected_index": None,
        "valid": False,
        "reason": "NO_OBJECT",
        "process_ms": 1.5,
        "method": "opencv",
        "class_name": None,
        "confidence": None,
        "model_id": None,
    }
    fields.update(over)

    # 유효 관측은 사유를 갖지 않는다(명시한 경우만 그대로 둠)
    if fields["valid"] is True and "reason" not in over:
        fields["reason"] = None

    return fields


def _square():
    """(4, 2) float32 사각형 윤곽."""
    return np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)


def _settings(**over):
    """기본 설정·프로파일 없음의 `PipelineSettings`."""
    fields = {"source_id": "src0", "mode": "planar", "detector_config": _config(), "profile": None}
    fields.update(over)

    return PipelineSettings(**fields)


class ExistingTypesTest(unittest.TestCase):
    """새 공통형 추가 뒤에도 기존 최소형·A 상속형이 그대로인지 확인한다(계획 작업 1)."""

    def test_existing_input_types_still_construct(self):
        """공통 `FramePacket`(14필드)·`InputStatusEvent`(5필드)와 A 상속형(+`source_ref`·`source_index`)의 생성 인자·필드 순서·동작이 바뀌지 않는다."""
        common_fields = [f.name for f in dataclasses.fields(common.FramePacket)]
        self.assertEqual(common_fields, [
            "image", "run_id", "source_id", "image_seq", "timestamp_ms", "timestamp_base",
            "received_monotonic_ns", "width", "height", "crop", "calibration_id",
            "capture_generation", "optical_frame", "ros_stamp_ns",
        ])
        self.assertEqual([f.name for f in dataclasses.fields(common.InputStatusEvent)],
                         ["run_id", "source_id", "status", "event_monotonic_ns", "detail"])

        # 기본값 없는 생성 인자 그대로
        for cls in (common.FramePacket, common.InputStatusEvent, base.FramePacket, base.InputStatusEvent):
            for f in dataclasses.fields(cls):
                self.assertIs(f.default, dataclasses.MISSING, f"{cls.__name__}.{f.name}")
                self.assertIs(f.default_factory, dataclasses.MISSING, f"{cls.__name__}.{f.name}")

        # 공통형 생성
        frame = common.FramePacket(**_common_frame_kwargs())
        self.assertEqual((frame.image_seq, frame.capture_generation), (3, 2))

        event = common.InputStatusEvent(run_id="run-t", source_id="src0",
                                        status=common.InputStatus.END_OF_STREAM,
                                        event_monotonic_ns=9, detail=None)
        self.assertIs(event.status, common.InputStatus.END_OF_STREAM)

        # A 상속형: 공통형 뒤에 확장 필드
        a_frame = base.FramePacket(**_common_frame_kwargs(), source_ref="a.png", source_index=4)
        self.assertIsInstance(a_frame, common.FramePacket)
        self.assertEqual([f.name for f in dataclasses.fields(base.FramePacket)],
                         common_fields + ["source_ref", "source_index"])
        self.assertEqual((a_frame.source_ref, a_frame.source_index), ("a.png", 4))

        a_event = base.InputStatusEvent(run_id="run-t", source_id="src0",
                                        status=base.InputStatus.DECODE_FAILED,
                                        event_monotonic_ns=1, detail="x", source_ref=None)
        self.assertIsInstance(a_event, common.InputStatusEvent)

        # frozen 동작 유지
        with self.assertRaises(dataclasses.FrozenInstanceError):
            a_frame.image_seq = 9

    def test_new_common_types_have_no_qt_ros_or_opencv_import(self):
        """`common.types`는 표준 라이브러리·NumPy만 import한다(초안 §3, 계획 파일 배치)."""
        import subprocess
        import sys

        code = (
            "import sys; import eyetoarm.common.types\n"
            "bad = [m for m in ('cv2', 'rclpy', 'PySide6', 'PyQt5') if m in sys.modules]\n"
            "print(','.join(bad))"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

        self.assertEqual(out.stdout.strip(), "")


class PixelBasisTest(unittest.TestCase):
    """`PixelBasis` 값 집합(초안 §3.1)."""

    def test_pixel_basis_values(self):
        """`PixelBasis`는 `raw`·`undistorted` 두 값의 문자열 열거다."""
        self.assertEqual({m.value for m in PixelBasis}, {"raw", "undistorted"})
        self.assertIsInstance(PixelBasis.RAW, str)


class PlanarObservationTest(unittest.TestCase):
    """`PlanarObservation` 필드·유효 조건·윤곽 소유(초안 §3.3, 계획 작업 1)."""

    def test_observation_fields_follow_draft(self):
        """관측 필드가 초안 §3.3 전 필드이며 기본값이 없다(`observed_face="top_face"` 기본값 금지). 예외는 A-8 끝점 필드 하나(마지막, 기본 None)."""
        names = [f.name for f in dataclasses.fields(PlanarObservation)]

        self.assertEqual(names, [
            "run_id", "source_id", "image_seq", "capture_generation", "generation", "job_seq",
            "detector_config_id", "pixel_basis", "calibration_id", "observed_face", "calc_contour",
            "display_mask_contour", "candidate_count", "selected_index", "valid", "reason",
            "process_ms", "method", "class_name", "confidence", "model_id", "axis_endpoints_px",
        ])

        for f in dataclasses.fields(PlanarObservation):
            if f.name == "axis_endpoints_px":
                self.assertIsNone(f.default, f.name)
            else:
                self.assertIs(f.default, dataclasses.MISSING, f.name)

    def test_valid_observation_requires_calc_face_and_selection(self):
        """`valid=True`인데 관측면·계산용 윤곽·선택 번호 중 하나가 None이면 거부하고, 무효 관측은 값이 없는 필드를 None으로 유지한다."""
        full = {"valid": True, "observed_face": "face_x", "calc_contour": _square(),
                "selected_index": 0, "candidate_count": 1}

        # 셋 다 있으면 생성된다
        ok = PlanarObservation(**_observation_kwargs(**full))
        self.assertTrue(ok.valid)

        # 하나라도 None이면 거부
        for missing in ("observed_face", "calc_contour", "selected_index"):
            with self.subTest(missing=missing):
                with self.assertRaises(ValueError):
                    PlanarObservation(**_observation_kwargs(**{**full, missing: None}))

        # 무효 관측: 값이 없는 필드는 None 그대로
        invalid = PlanarObservation(**_observation_kwargs())

        self.assertFalse(invalid.valid)

        for name in ("observed_face", "calc_contour", "display_mask_contour", "selected_index",
                     "calibration_id", "class_name", "confidence", "model_id"):
            self.assertIsNone(getattr(invalid, name), name)

        self.assertEqual(invalid.reason, "NO_OBJECT")

    def test_observation_rejects_bad_enums_and_flags(self):
        """`method`는 opencv/ai, `pixel_basis`는 열거 값, `valid`는 bool만 받는다."""
        with self.assertRaises(ValueError):
            PlanarObservation(**_observation_kwargs(method="yolo"))

        with self.assertRaises(ValueError):
            PlanarObservation(**_observation_kwargs(pixel_basis="rectified"))

        with self.assertRaises(ValueError):
            PlanarObservation(**_observation_kwargs(valid=1))

        # 문자열 값은 열거로 정규화
        obs = PlanarObservation(**_observation_kwargs(pixel_basis="undistorted"))
        self.assertIs(obs.pixel_basis, PixelBasis.UNDISTORTED)

    def test_contours_are_owned_readonly_copies(self):
        """외부 배열을 나중에 고쳐도 관측 윤곽이 변하지 않고, 윤곽은 읽기 전용 float32 (N,2)이며 클래스·보정·시간 값을 새로 만들어 넣지 않는다."""
        calc = _square()
        display = np.array([[1, 1], [20, 1], [20, 20]], dtype=np.float64)
        obs = PlanarObservation(**_observation_kwargs(
            valid=True, observed_face="face_x", calc_contour=calc, display_mask_contour=display,
            selected_index=0, candidate_count=1))

        calc[0, 0] = 99.0
        display[0, 0] = 99.0

        self.assertEqual(float(obs.calc_contour[0, 0]), 0.0)
        self.assertEqual(float(obs.display_mask_contour[0, 0]), 1.0)

        for contour in (obs.calc_contour, obs.display_mask_contour):
            self.assertEqual(contour.dtype, np.float32)
            self.assertEqual(contour.ndim, 2)
            self.assertEqual(contour.shape[1], 2)
            self.assertFalse(contour.flags.writeable)

        self.assertFalse(np.shares_memory(obs.calc_contour, calc))

        # 전체 실루엣을 계산용으로 승격하지 않는다(서로 다른 배열 그대로)
        self.assertEqual(obs.display_mask_contour.shape, (3, 2))
        self.assertEqual(obs.calc_contour.shape, (4, 2))

        # 생성자가 클래스·보정·시간 값을 만들어 넣지 않는다
        self.assertIsNone(obs.class_name)
        self.assertIsNone(obs.calibration_id)
        self.assertIsNone(obs.model_id)
        self.assertFalse(hasattr(obs, "timestamp_ms"))
        self.assertFalse(hasattr(obs, "received_monotonic_ns"))

        # 무효 관측도 표시용 윤곽만 둘 수 있다
        disp_only = PlanarObservation(**_observation_kwargs(display_mask_contour=display))
        self.assertIsNone(disp_only.calc_contour)

    def test_calc_contour_requires_observed_face(self):
        """관측면이 None이면 계산용 윤곽도 None이어야 하며, 무효 관측이라도 면 없는 계산용 윤곽은 거부한다(R2a, 초안 §3.3 유효 조건·§7 예시 1-b)."""
        with self.assertRaises(ValueError):
            PlanarObservation(**_observation_kwargs(calc_contour=_square()))

        # 관측면이 있으면 무효 관측에도 계산용 윤곽을 둘 수 있다
        obs = PlanarObservation(**_observation_kwargs(observed_face="face_x", calc_contour=_square()))
        self.assertEqual(obs.calc_contour.shape, (4, 2))

    def test_invalid_observation_requires_reason(self):
        """무효 관측은 비어 있지 않은 문자열 사유가 있어야 하고 None·빈 문자열·비문자열은 거부한다(초안 §3.3 reason 행)."""
        for bad in (None, "", 3):
            with self.subTest(reason=bad):
                with self.assertRaises(ValueError):
                    PlanarObservation(**_observation_kwargs(valid=False, reason=bad))

        # 관측면 미확정 무효 관측은 사유와 함께 calc_contour=None을 유지한다
        unresolved = PlanarObservation(**_observation_kwargs(reason="TOP_FACE_UNRESOLVED"))

        self.assertIsNone(unresolved.calc_contour)
        self.assertEqual(unresolved.reason, "TOP_FACE_UNRESOLVED")

    def test_conditional_fields_are_none_outside_their_condition(self):
        """유효 관측의 reason은 None이어야 하고, opencv 관측은 class_name·confidence·model_id가 None이어야 하며 ai 관측은 값을 가질 수 있다(R2b, 초안 §3.3)."""
        valid = {"valid": True, "observed_face": "face_x", "calc_contour": _square(),
                 "selected_index": 0, "candidate_count": 1}

        with self.assertRaises(ValueError):
            PlanarObservation(**_observation_kwargs(**valid, reason="NO_OBJECT"))

        # 무효 관측은 사유가 필수다
        self.assertEqual(PlanarObservation(**_observation_kwargs(reason="NO_OBJECT")).reason, "NO_OBJECT")

        for name, value in (("class_name", "bolt"), ("confidence", 0.9), ("model_id", "m1")):
            with self.subTest(field=name):
                with self.assertRaises(ValueError):
                    PlanarObservation(**_observation_kwargs(method="opencv", **{name: value}))

        ai = PlanarObservation(**_observation_kwargs(method="ai", class_name="bolt", confidence=0.9,
                                                     model_id="m1"))
        self.assertEqual((ai.class_name, ai.confidence, ai.model_id), ("bolt", 0.9, "m1"))

    def test_contour_shape_is_checked(self):
        """윤곽이 (N,2)가 아니면 거부한다."""
        for bad in (np.zeros((4,), np.float32), np.zeros((4, 3), np.float32), np.zeros((2, 4, 2), np.float32)):
            with self.subTest(shape=bad.shape):
                with self.assertRaises(ValueError):
                    PlanarObservation(**_observation_kwargs(display_mask_contour=bad))


def _endpoints():
    """머리 끝 → 축 끝 두 점(정수 목록, 전체 입력 영상 픽셀)."""
    return [[100, 200], [300, 400]]


def _axis_kwargs(**over):
    """A-8 볼트 축 유효 관측 생성 인자(AI, 클래스 bolt). 키워드로 필드를 바꿀 수 있다."""
    fields = {"valid": True, "observed_face": "axis", "calc_contour": None,
              "axis_endpoints_px": _endpoints(), "selected_index": 0, "candidate_count": 1,
              "method": "ai", "class_name": "bolt", "confidence": 0.9, "model_id": "m"}
    fields.update(over)

    return _observation_kwargs(**fields)


class AxisEndpointObservationTest(unittest.TestCase):
    """A-8 볼트 두 끝점 관측(#155 r1 §1~3, #157 T1). 기존 윤곽 경로는 그대로다."""

    def test_axis_observation_is_valid_owned_readonly_float64(self):
        """축 경로 유효 관측이 생성되고 끝점은 head→tip 순서의 float64 (2,2) 독립·읽기 전용 복사본이다(사례 #3)."""
        source = np.array(_endpoints(), dtype=np.int64)
        obs = PlanarObservation(**_axis_kwargs(axis_endpoints_px=source))

        self.assertTrue(obs.valid)
        self.assertIsNone(obs.calc_contour)
        self.assertEqual(obs.axis_endpoints_px.dtype, np.float64)
        self.assertEqual(obs.axis_endpoints_px.shape, (2, 2))
        self.assertEqual(obs.axis_endpoints_px.tolist(), [[100.0, 200.0], [300.0, 400.0]])
        self.assertFalse(obs.axis_endpoints_px.flags.writeable)
        self.assertFalse(np.shares_memory(obs.axis_endpoints_px, source))

        # 외부 배열을 나중에 고쳐도 관측은 변하지 않는다
        source[0, 0] = 999
        self.assertEqual(float(obs.axis_endpoints_px[0, 0]), 100.0)

        # 관측 배열에 직접 쓸 수 없다
        with self.assertRaises(ValueError):
            obs.axis_endpoints_px[0, 0] = 1.0

    def test_axis_accepts_bolt_names_or_none(self):
        """축 경로 클래스는 볼트 이름(대소문자·BOLT_A/B) 또는 None(OpenCV)만 받는다(사례 #3)."""
        for name in ("bolt", "Bolt", "BOLT_A", "BOLT_B"):
            with self.subTest(class_name=name):
                self.assertTrue(PlanarObservation(**_axis_kwargs(class_name=name)).valid)

        opencv = PlanarObservation(**_axis_kwargs(method="opencv", class_name=None, confidence=None,
                                                  model_id=None))
        self.assertTrue(opencv.valid)

        # 기존 method 규칙 유지: OpenCV 관측은 AI 전용 필드를 갖지 않는다
        with self.assertRaises(ValueError):
            PlanarObservation(**_axis_kwargs(method="opencv", class_name="bolt", confidence=None, model_id=None))

    def test_axis_rejects_nut_or_other_class(self):
        """너트·기타 클래스에 끝점을 주면 거부한다(사례 #5)."""
        for name in ("nut", "NUT", "washer", ""):
            with self.subTest(class_name=name):
                with self.assertRaises(ValueError):
                    PlanarObservation(**_axis_kwargs(class_name=name))

    def test_axis_rejects_mixed_paths_and_missing_face(self):
        """끝점+계산 윤곽, 관측면 없는 끝점, axis인데 끝점 없음(유효), axis+윤곽, 다른 관측면+끝점을 거부한다(사례 #6·#7·#7b·#7c)."""
        cases = {
            "endpoints_and_contour": _axis_kwargs(calc_contour=_square()),
            "endpoints_without_face": _axis_kwargs(observed_face=None),
            "axis_without_endpoints": _axis_kwargs(axis_endpoints_px=None),
            "axis_with_contour_no_endpoints": _axis_kwargs(axis_endpoints_px=None, calc_contour=_square()),
            "endpoints_with_top_face": _axis_kwargs(observed_face="top_face"),
            "invalid_endpoints_without_face": _observation_kwargs(axis_endpoints_px=_endpoints()),
        }

        for label, kwargs in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    PlanarObservation(**kwargs)

    def test_axis_requires_selected_index(self):
        """축 경로 유효 관측도 선택 번호가 필수다(사례 #8)."""
        with self.assertRaises(ValueError):
            PlanarObservation(**_axis_kwargs(selected_index=None))

    def test_bad_endpoint_values_are_value_errors(self):
        """형식·값이 잘못된 끝점은 OverflowError·경고 없이 ValueError로 거부한다(사례 #9a)."""
        bad = {
            "three_points": [[0, 0], [1, 1], [2, 2]],
            "three_coords": [[0, 0, 0], [1, 1, 1]],
            "flat": [0, 0, 1, 1],
            "one_point": [[0, 0]],
            "deep": [[[0], [0]], [[1], [1]]],
            "scalar": 5,
            "empty": [],
            "bool_all": [[True, False], [False, True]],
            "bool_mixed": [[True, 0], [10, 10]],
            "numpy_bool": np.array([[True, False], [False, True]]),
            "string": [["1", "2"], ["3", "4"]],
            "object": np.array([[1, 2], [3, None]], dtype=object),
            "object_numeric": np.array([[1, 2], [3, 4]], dtype=object),
            "string_array": np.array([["1", "2"], ["3", "4"]]),
            "complex_array": np.array([[1, 2], [3, 4]], dtype=np.complex128),
            "complex": [[1 + 1j, 2], [3, 4]],
            "nan": [[np.nan, 1.0], [2.0, 3.0]],
            "inf": [[np.inf, 1.0], [2.0, 3.0]],
            "neg_inf": [[1.0, 2.0], [-np.inf, 3.0]],
            "same_point": [[5, 5], [5, 5]],
            "huge_int": [[10 ** 400, 0], [1, 1]],
            "huge_neg_int": [[-(10 ** 400), 0], [1, 1]],
            "overflow_float": [[1e308 * 10, 0.0], [1.0, 1.0]],
            "same_after_float64": [[2 ** 53, 0], [2 ** 53 + 1, 0]],
        }

        for label, value in bad.items():
            with self.subTest(case=label):
                with np.errstate(all="raise"):
                    with self.assertRaises(ValueError):
                        PlanarObservation(**_axis_kwargs(axis_endpoints_px=value))

    def test_invalid_axis_observation_may_carry_endpoints(self):
        """무효 관측도 axis 관측면과 끝점을 진단용으로 가질 수 있고 사유 규칙은 그대로다(사례 #10)."""
        obs = PlanarObservation(**_axis_kwargs(valid=False, reason="ENDPOINT_CLIPPED"))

        self.assertFalse(obs.valid)
        self.assertEqual(obs.axis_endpoints_px.shape, (2, 2))

        with self.assertRaises(ValueError):
            PlanarObservation(**_axis_kwargs(valid=False, reason=None))

    def test_existing_contour_path_unchanged(self):
        """기존 무효 표시 관측·윤곽 유효 관측(자유 관측면 문자열)은 끝점 없이 그대로 생성된다(사례 #1·#2·#4, 실루엣 승격 허가 아님)."""
        invalid = PlanarObservation(**_observation_kwargs())
        self.assertIsNone(invalid.axis_endpoints_px)

        for face in ("face_x", "silhouette", "top_face"):
            with self.subTest(face=face):
                obs = PlanarObservation(**_observation_kwargs(
                    valid=True, observed_face=face, calc_contour=_square(), selected_index=0, candidate_count=1))
                self.assertIsNone(obs.axis_endpoints_px)

        nut = PlanarObservation(**_observation_kwargs(
            valid=True, observed_face="top_face", calc_contour=_square(), selected_index=0, candidate_count=1,
            method="ai", class_name="nut", confidence=0.8, model_id="m"))
        self.assertEqual(nut.class_name, "nut")

        # 윤곽 경로에 axis 관측면 문자열을 쓰지 않는다
        with self.assertRaises(ValueError):
            PlanarObservation(**_observation_kwargs(
                valid=True, observed_face="axis", calc_contour=_square(), selected_index=0, candidate_count=1))

    def test_replace_keeps_and_revalidates_endpoints(self):
        """`dataclasses.replace`는 끝점을 보존하고 새 값도 다시 검사한다."""
        obs = PlanarObservation(**_axis_kwargs())

        same = dataclasses.replace(obs, process_ms=2.0)
        self.assertEqual(same.axis_endpoints_px.tolist(), obs.axis_endpoints_px.tolist())
        self.assertFalse(same.axis_endpoints_px.flags.writeable)

        with self.assertRaises(ValueError):
            dataclasses.replace(obs, axis_endpoints_px=[[1, 1], [1, 1]])

        with self.assertRaises(ValueError):
            dataclasses.replace(obs, class_name="nut")


class InvalidationEventTest(unittest.TestCase):
    """`CandidateInvalidationEvent` 필드·kind·프레임 없는 사건 규칙(초안 §3.5·§4-6, A 상세설계 §5.4)."""

    def _kw(self, **over):
        """프레임 없는 세대 변경 사건 생성 인자."""
        fields = {"run_id": "run-t", "source_id": "src0", "kind": InvalidationKind.GENERATION_CHANGED,
                  "cause": "detector_config_changed", "event_monotonic_ns": 10, "generation": 1,
                  "detector_config_id": "abc", "job_seq": None, "image_seq": None}
        fields.update(over)

        return fields

    def test_event_fields_follow_draft(self):
        """사건 필드가 초안 §3.5의 9개이며 기본값이 없다."""
        self.assertEqual([f.name for f in dataclasses.fields(CandidateInvalidationEvent)], [
            "run_id", "source_id", "kind", "cause", "event_monotonic_ns", "generation",
            "detector_config_id", "job_seq", "image_seq",
        ])

        for f in dataclasses.fields(CandidateInvalidationEvent):
            self.assertIs(f.default, dataclasses.MISSING, f.name)

    def test_kind_has_exactly_five_values(self):
        """kind는 문서의 5개 값만 허용하고 다른 문자열은 거부한다."""
        self.assertEqual({k.value for k in InvalidationKind}, {
            "MODEL_LOAD_FAILED", "MODEL_CONFIG_INVALID", "INFERENCE_ERROR", "JOB_ABORTED",
            "GENERATION_CHANGED",
        })

        ev = CandidateInvalidationEvent(**self._kw(kind="GENERATION_CHANGED"))
        self.assertIs(ev.kind, InvalidationKind.GENERATION_CHANGED)

        with self.assertRaises(ValueError):
            CandidateInvalidationEvent(**self._kw(kind="POSE_AMBIGUOUS"))

    def test_frameless_events_have_no_frame_ids(self):
        """세대 변경·모델 로딩 실패는 job_seq·image_seq가 None이어야 하며 가상 값을 넣으면 거부한다."""
        for kind in (InvalidationKind.GENERATION_CHANGED, InvalidationKind.MODEL_LOAD_FAILED):
            ev = CandidateInvalidationEvent(**self._kw(kind=kind))
            self.assertIsNone(ev.job_seq)
            self.assertIsNone(ev.image_seq)

            for ids in ({"job_seq": 0}, {"image_seq": 0}, {"job_seq": 1, "image_seq": 2}):
                with self.subTest(kind=kind, ids=ids):
                    with self.assertRaises(ValueError):
                        CandidateInvalidationEvent(**self._kw(kind=kind, **ids))

    def test_required_fields_have_valid_types(self):
        """필수 문자열은 비어 있지 않고, 시각·세대·식별자는 bool 아닌 정수여야 한다(결함 4, 초안 §3.5 필수 필드)."""
        bad = [{"run_id": ""}, {"source_id": None}, {"cause": ""}, {"detector_config_id": ""},
               {"generation": "x"}, {"generation": True}, {"event_monotonic_ns": 1.5},
               {"kind": InvalidationKind.INFERENCE_ERROR, "job_seq": "1", "image_seq": 2},
               {"kind": InvalidationKind.INFERENCE_ERROR, "job_seq": 1, "image_seq": 2.0}]

        for over in bad:
            with self.subTest(over=over):
                with self.assertRaises(ValueError):
                    CandidateInvalidationEvent(**self._kw(**over))

    def test_job_events_need_job_and_frame_ids(self):
        """작업 오류·중단은 job_seq·image_seq를 함께 가지고, 설정 오류는 둘 다 None이거나 둘 다 값이다(A 상세설계 §5.4 표)."""
        for kind in (InvalidationKind.INFERENCE_ERROR, InvalidationKind.JOB_ABORTED):
            ev = CandidateInvalidationEvent(**self._kw(kind=kind, job_seq=4, image_seq=7))
            self.assertEqual((ev.job_seq, ev.image_seq), (4, 7))

            with self.assertRaises(ValueError):
                CandidateInvalidationEvent(**self._kw(kind=kind))

        cfg = InvalidationKind.MODEL_CONFIG_INVALID
        CandidateInvalidationEvent(**self._kw(kind=cfg))
        CandidateInvalidationEvent(**self._kw(kind=cfg, job_seq=1, image_seq=2))

        with self.assertRaises(ValueError):
            CandidateInvalidationEvent(**self._kw(kind=cfg, job_seq=1))


class PipelineTypesTest(unittest.TestCase):
    """A 내부 파이프라인 타입(계획 「인터페이스 제안」)의 불변성·유도 값·필드 경계."""

    def test_settings_reuse_planar_config_and_are_frozen(self):
        """`PipelineSettings`는 기존 `PlanarConfig`를 그대로 담고 frozen이며, 잘못된 값은 거부한다."""
        cfg = _config()
        s = _settings(detector_config=cfg)

        self.assertIs(s.detector_config, cfg)

        with self.assertRaises(dataclasses.FrozenInstanceError):
            s.mode = "other"

        with self.assertRaises(ValueError):
            _settings(detector_config={"roi": None})

        with self.assertRaises(ValueError):
            _settings(mode="")

        with self.assertRaises(ValueError):
            _settings(source_id="")

        with self.assertRaises(ValueError):
            _settings(profile="calib-1")

        # 표시 전용 설정(오버레이 색) 필드는 처리 설정에 두지 않는다
        self.assertEqual([f.name for f in dataclasses.fields(PipelineSettings)],
                         ["source_id", "mode", "detector_config", "profile"])

    def test_source_profile_applies_to_matching_frame_only(self):
        """프로파일은 source_id·해상도·크롭·calibration_id가 모두 같고 calibration_id가 있을 때만 그 프레임에 적용된다."""
        frame = common.FramePacket(**_common_frame_kwargs(calibration_id="cal-1", crop=(0, 0, 6, 4)))
        prof = SourceProfile(source_id="src0", width=6, height=4, crop=(0, 0, 6, 4), calibration_id="cal-1")

        self.assertTrue(prof.applies_to(frame))

        for over in ({"source_id": "src1"}, {"width": 7}, {"height": 5}, {"crop": None},
                     {"calibration_id": "cal-2"}):
            with self.subTest(over=over):
                self.assertFalse(dataclasses.replace(prof, **over).applies_to(frame))

        # 보정 누락(프로파일·프레임 모두 None)도 적용하지 않는다
        no_cal = SourceProfile(source_id="src0", width=6, height=4, crop=None, calibration_id=None)
        self.assertFalse(no_cal.applies_to(common.FramePacket(**_common_frame_kwargs())))

    def test_job_context_is_frozen_and_derives_config_id(self):
        """`JobContext`는 frozen이며 `detector_config_id`는 시작 설정의 `config_sha256`이다."""
        frame = common.FramePacket(**_common_frame_kwargs())
        s = _settings()
        job = JobContext(frame=frame, generation=2, job_seq=0, settings=s,
                         started_monotonic_ns=100, reprocess=False)

        self.assertEqual(job.detector_config_id, s.detector_config.config_sha256)

        with self.assertRaises(dataclasses.FrozenInstanceError):
            job.generation = 3

    def test_job_status_has_four_values(self):
        """`JobStatus`는 COMPLETED·DISCARDED_GENERATION·FAILED·ABORTED 네 값이며 대기열 폐기 상태가 없다."""
        self.assertEqual({s.name for s in JobStatus},
                         {"COMPLETED", "DISCARDED_GENERATION", "FAILED", "ABORTED"})

    def test_queue_drop_record_has_no_job_seq(self):
        """`QueueDropRecord`는 작업을 시작하지 않은 프레임의 기록이므로 job_seq·generation 필드가 없다."""
        names = {f.name for f in dataclasses.fields(QueueDropRecord)}

        self.assertNotIn("job_seq", names)
        self.assertNotIn("generation", names)
        self.assertTrue({"run_id", "source_id", "image_seq", "capture_generation", "timestamp_ms",
                         "timestamp_base", "reason", "dropped_monotonic_ns"} <= names)

        rec = QueueDropRecord(run_id="r", source_id="s", image_seq=1, capture_generation=0,
                              timestamp_ms=5, timestamp_base=common.TimestampBase.PC_RECEIVE,
                              received_monotonic_ns=5_000_000, ros_stamp_ns=None,
                              reason=QueueDropReason.REPLACED_BY_NEWER, dropped_monotonic_ns=9)

        with self.assertRaises(dataclasses.FrozenInstanceError):
            rec.reason = QueueDropReason.CLOSED

    def test_record_types_are_frozen_and_have_no_command_id_value(self):
        """`JobOutcome`·`ProcessRecord`·`InputStateRecord`는 frozen이고, 처리 기록의 명령 ID는 None만 허용한다."""
        frame = common.FramePacket(**_common_frame_kwargs())
        job = JobContext(frame=frame, generation=2, job_seq=0, settings=_settings(),
                         started_monotonic_ns=100, reprocess=False)
        outcome = JobOutcome(context=job, status=JobStatus.ABORTED, observation=None,
                             display_result=None, error=None, invalidation=None)

        self.assertIn("invalidation", {f.name for f in dataclasses.fields(JobOutcome)})

        with self.assertRaises(dataclasses.FrozenInstanceError):
            outcome.status = JobStatus.COMPLETED

        rec_kw = dict(run_id="r", source_id="s", image_seq=1, capture_generation=0, generation=0,
                      job_seq=0, mode="planar", timestamp_ms=1,
                      timestamp_base=common.TimestampBase.PC_RECEIVE, calibration_id=None,
                      detector_config_id="abc", reprocess=False, status=JobStatus.COMPLETED,
                      valid=False, reason="NO_OBJECT", reason_owner="A", candidate_count=0,
                      process_ms=1.0, command_id=None)
        rec = ProcessRecord(**rec_kw)
        self.assertIsNone(rec.command_id)

        with self.assertRaises(ValueError):
            ProcessRecord(**{**rec_kw, "command_id": "cmd-1"})

        state = InputStateRecord(run_id="r", source_id="s", status=common.InputStatus.FRAME_TIMEOUT,
                                 event_monotonic_ns=1, detail=None, duration_ns=None)

        with self.assertRaises(dataclasses.FrozenInstanceError):
            state.duration_ns = 3

    def test_counters_have_twelve_fields_starting_at_zero(self):
        """`PipelineCounters`는 계획의 12개 계수를 가지며 모두 0(사유·종류별 계수는 빈 사전)에서 시작한다."""
        c = PipelineCounters()

        self.assertEqual([f.name for f in dataclasses.fields(PipelineCounters)], [
            "received", "first_started", "reprocess_started", "completed", "valid_completed",
            "invalid_completed_by_reason", "queue_dropped", "generation_discarded", "failed",
            "aborted", "invalidation_applied_by_kind", "invalidation_ignored_by_kind",
        ])

        for f in dataclasses.fields(c):
            value = getattr(c, f.name)
            self.assertIn(value, (0, {}), f.name)

        # 사전 계수는 인스턴스마다 따로
        c.queue_dropped["x"] = 1
        self.assertEqual(PipelineCounters().queue_dropped, {})


if __name__ == "__main__":
    unittest.main()
