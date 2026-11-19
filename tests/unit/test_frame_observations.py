"""#189 단계 L 자료형(`ObjectObservation`·`FrameObservations`)과 목록 봉투 `eyetoarm.b_input/2` 계약 시험.

근거: 계획 `_workspace/189_builder2_stage_L_plan.md` r0 §2·§4·§8(L-1~L-9), A 상세설계 §5.2 B-09 협의안, reviewer #189 Q1(카메라 속성 대조).
"""
from __future__ import annotations

import dataclasses
import unittest
from types import SimpleNamespace

import numpy as np

from eyetoarm.common import types as common
from eyetoarm.common.types import (FRAME_OBSERVATIONS_VERSION, CaptureInfo, CaptureTf, FrameObservations,
                                   ObjectObservation, PixelBasis)
from eyetoarm.transforms.b_stage import correlation_keys
from eyetoarm.vision import observation_envelope
from eyetoarm.vision.list_envelope import LIST_ENVELOPE_VERSION, ListEnvelopeError, make_list_envelope
from test_pipeline import frame

W, H = 200, 120
CAL = "cal-list-test"
OPTICAL = "cam_optical"
STAMP = 77_000_000_000
T_W_C = np.array([[1.0, 0, 0, 0.4], [0, -1.0, 0, 0.0], [0, 0, -1.0, 0.5], [0, 0, 0, 1.0]])


def ro_frame(**over):
    """읽기 전용 영상의 200×120 프레임(촬영 시각 STAMP, 광학 프레임 OPTICAL)."""
    image = np.zeros((H, W, 3), np.uint8)
    image.setflags(write=False)
    fields = dict(image=image, width=W, height=H, calibration_id=CAL, optical_frame=OPTICAL, ros_stamp_ns=STAMP)
    fields.update(over)

    return frame(0, **fields)


def capture(**over):
    """프레임과 맞는 촬영 정보."""
    fields = dict(stamp_ns=STAMP, optical_frame=OPTICAL, calibration_id=CAL, width=W, height=H)
    fields.update(over)

    return CaptureInfo(**fields)


def camera(**over):
    """카메라 프로파일 대역(속성 대조만 하므로 같은 이름의 속성이면 된다)."""
    fields = dict(calibration_id=CAL, frame_id=OPTICAL, width=W, height=H)
    fields.update(over)

    return SimpleNamespace(**fields)


def tf(**over):
    """촬영 시각 TF."""
    fields = dict(parent="world", child=OPTICAL, stamp_ns=STAMP, T_W_C=T_W_C)
    fields.update(over)

    return CaptureTf(**fields)


def bolt(index=0, points=((10.0, 20.0), (60.0, 25.0)), **over):
    """볼트 축 후보."""
    fields = dict(instance_index=index, class_name="bolt", confidence=0.9, pixel_basis=PixelBasis.RAW,
                  observed_face="axis", axis_endpoints_px=points, calc_contour=None)
    fields.update(over)

    return ObjectObservation(**fields)


def frame_obs(**over):
    """유효 목록 하나(볼트 후보 1개)."""
    fields = dict(version=FRAME_OBSERVATIONS_VERSION, frame=ro_frame(), capture=capture(), camera=camera(), tf=tf(),
                  generation=0, job_seq=3, detector_config_id="a" * 64, candidates=(bolt(),))
    fields.update(over)

    return FrameObservations(**fields)


class ObjectFieldTest(unittest.TestCase):
    """L-1·L-2: 후보 필드 형·배타·클래스–관측면."""

    def test_valid_bolt(self):
        """정상 볼트 axis 후보는 만들어지고 끝점은 float64 (2,2) 읽기 전용이다."""
        b = bolt()

        self.assertEqual((b.axis_endpoints_px.dtype, b.axis_endpoints_px.shape), (np.float64, (2, 2)))
        self.assertFalse(b.axis_endpoints_px.flags.writeable)

    def test_bad_scalar_fields(self):
        """인덱스 음수·bool·실수, 빈 클래스, 신뢰도 bool·범위 밖·비유한·문자열, 기준 문자열, 관측면 미지는 ValueError."""
        bad = [dict(instance_index=-1), dict(instance_index=True), dict(instance_index=1.0), dict(class_name=""),
               dict(class_name=None), dict(confidence=True), dict(confidence=1.01), dict(confidence=-0.1),
               dict(confidence=float("nan")), dict(confidence="0.9"), dict(confidence=10 ** 400),
               dict(pixel_basis="raw"), dict(observed_face="face"), dict(observed_face=None)]

        for over in bad:
            with self.subTest(over=over):
                with self.assertRaises(ValueError):
                    bolt(**over)

    def test_exclusive_geometry_and_class(self):
        """axis 후보에 윤곽이 있거나 끝점이 없으면, 볼트 아닌 클래스의 axis 후보는 ValueError(너트 경로는 시험하지 않음, 사용자 지시)."""
        contour = np.array([[1.0, 1.0], [5.0, 1.0], [3.0, 4.0]])
        bad = [lambda: bolt(calc_contour=contour), lambda: bolt(axis_endpoints_px=None), lambda: bolt(class_name="nut")]

        for k, make in enumerate(bad):
            with self.subTest(case=k):
                with self.assertRaises(ValueError):
                    make()

    def test_bad_geometry_shapes(self):
        """끝점 (2,3)·같은 두 점·비유한·bool은 ValueError."""
        for points in ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[1.0, 2.0], [1.0, 2.0]], [[np.inf, 2.0], [3.0, 4.0]],
                       [[True, 2.0], [3.0, 4.0]]):
            with self.subTest(points=points):
                with self.assertRaises(ValueError):
                    bolt(points=points)


class OwnershipTest(unittest.TestCase):
    """L-4: 독립 복사·읽기 전용·메모리 비공유."""

    def test_input_change_does_not_leak(self):
        """입력 배열을 나중에 고쳐도 후보 값은 그대로이고, 후보 배열 쓰기는 ValueError, 입력과 메모리를 나누지 않는다."""
        src = np.array([[10.0, 20.0], [60.0, 25.0]])
        c = bolt(points=src)
        src[0, 0] = 99.0

        self.assertEqual(c.axis_endpoints_px[0, 0], 10.0)
        self.assertFalse(np.shares_memory(c.axis_endpoints_px, src))

        with self.assertRaises(ValueError):
            c.axis_endpoints_px[0, 0] = 1.0

    def test_tf_is_independent_copy(self):
        """목록의 T_W_C는 입력 행렬의 독립 읽기 전용 복사본이다."""
        src = T_W_C.copy()
        obs = frame_obs(tf=tf(T_W_C=src))
        src[0, 3] = 9.0

        self.assertEqual(obs.tf.T_W_C[0, 3], 0.4)
        self.assertFalse(np.shares_memory(obs.tf.T_W_C, src))
        self.assertFalse(obs.tf.T_W_C.flags.writeable)

    def test_frame_image_must_be_read_only(self):
        """쓰기 가능한 영상의 프레임은 거부한다(복사하지 않고 검사만)."""
        with self.assertRaises(ValueError):
            frame_obs(frame=ro_frame(image=np.zeros((H, W, 3), np.uint8)))

        obs = frame_obs()
        self.assertIs(obs.frame.image, obs.frame.image)


class CandidateTupleTest(unittest.TestCase):
    """L-3·L-5: 후보 tuple 규칙과 RAW 범위."""

    def test_tuple_rules(self):
        """빈 tuple·list·중복·역순·후보 아닌 원소는 ValueError, 오름차순 여러 개는 허용(상한 없음)."""
        for cands in ((), [bolt()], (bolt(1), bolt(1)), (bolt(2), bolt(1)), (bolt(0), "x")):
            with self.subTest(cands=cands):
                with self.assertRaises(ValueError):
                    frame_obs(candidates=cands)

        many = tuple(bolt(i) for i in range(0, 40, 3))
        self.assertEqual(frame_obs(candidates=many).instance_indices, tuple(range(0, 40, 3)))

    def test_raw_bounds(self):
        """RAW 좌표는 W−1·H−1까지 허용, W·H·음수는 ValueError. undistorted는 범위를 보지 않는다."""
        ok = bolt(points=((0.0, 0.0), (W - 1.0, H - 1.0)))
        self.assertEqual(frame_obs(candidates=(ok,)).candidates[0], ok)

        for points in (((0.0, 0.0), (float(W), 5.0)), ((0.0, 0.0), (5.0, float(H))), ((-0.5, 0.0), (5.0, 5.0))):
            with self.subTest(points=points):
                with self.assertRaises(ValueError):
                    frame_obs(candidates=(bolt(points=points),))

        out = bolt(points=((0.0, 0.0), (float(W), 5.0)), pixel_basis=PixelBasis.UNDISTORTED)
        self.assertEqual(frame_obs(candidates=(out,)).candidates[0].pixel_basis, PixelBasis.UNDISTORTED)


class BindingTest(unittest.TestCase):
    """L-1·L-6: 버전·형·capture–frame–camera 대조."""

    def test_version_and_types(self):
        """다른 버전·FramePacket 아님·CaptureInfo 아님·세대·순번 음수·빈 설정 ID는 ValueError."""
        bad = [dict(version="eyetoarm.frame_observations/2"), dict(frame="frame"),
               dict(capture=dict(stamp_ns=STAMP)), dict(generation=-1), dict(job_seq=True),
               dict(detector_config_id="")]

        for over in bad:
            with self.subTest(over=over):
                with self.assertRaises(ValueError):
                    frame_obs(**over)

    def test_capture_frame_camera_mismatch(self):
        """시각·광학 프레임·보정 번호·해상도가 하나라도 어긋나거나 카메라 속성이 없으면 ValueError."""
        bad = [dict(capture=capture(stamp_ns=STAMP + 1)), dict(capture=capture(optical_frame="other")),
               dict(capture=capture(calibration_id="other")), dict(capture=capture(width=W + 1)),
               dict(frame=ro_frame(ros_stamp_ns=None)), dict(camera=camera(calibration_id="other")),
               dict(camera=camera(frame_id="other")), dict(camera=camera(height=H - 1)),
               dict(camera=SimpleNamespace(calibration_id=CAL, frame_id=OPTICAL, width=W))]

        for over in bad:
            with self.subTest(over=over):
                with self.assertRaises(ValueError):
                    frame_obs(**over)

    def test_real_camera_profile_accepted(self):
        """실제 CameraProfile도 같은 속성 대조로 받는다(공통형은 calibration 모듈을 import하지 않음)."""
        from eyetoarm.calibration.profile import CameraProfile

        cam = CameraProfile(width=W, height=H, K=np.array([[300.0, 0, 100.0], [0, 300.0, 60.0], [0, 0, 1.0]]),
                            dist=np.zeros(5), frame_id=OPTICAL, calibration_id=CAL)
        self.assertIs(frame_obs(camera=cam).camera, cam)
        self.assertNotIn("eyetoarm.calibration", " ".join(
            line for line in open(common.__file__, encoding="utf-8") if line.startswith(("import", "from"))))


class TfTest(unittest.TestCase):
    """L-7·L-9: 프레임 공통 TF 하나."""

    def test_tf_none_allowed(self):
        """TF 없음(None)은 허용한다(B가 프레임 전체 SKIP TF_UNAVAILABLE)."""
        self.assertIsNone(frame_obs(tf=None).tf)

    def test_tf_mismatch_and_matrix(self):
        """시각·자식 프레임 불일치, 빈 부모, (3,4)·비유한·마지막 행·비강체·형 아님은 ValueError."""
        bad_rows = T_W_C.copy()
        bad_rows[3] = [0, 0, 0, 2]
        scaled = T_W_C.copy()
        scaled[:3, :3] *= 1.01
        nan = T_W_C.copy()
        nan[0, 0] = np.nan
        bad = [tf(stamp_ns=STAMP - 1), tf(child="other"), tf(parent=""), tf(T_W_C=T_W_C[:3]), tf(T_W_C=nan),
               tf(T_W_C=bad_rows), tf(T_W_C=scaled), {"parent": "world"}]

        for t in bad:
            with self.subTest(t=t):
                with self.assertRaises(ValueError):
                    frame_obs(tf=t)


class SharedClassTest(unittest.TestCase):
    """CaptureInfo·CaptureTf 이동: 봉투 1 모듈과 같은 클래스 객체."""

    def test_same_class_objects(self):
        """봉투 모듈의 CaptureInfo·CaptureTf는 공통형의 것과 같은 객체이고 필드가 바뀌지 않았다."""
        self.assertIs(observation_envelope.CaptureInfo, CaptureInfo)
        self.assertIs(observation_envelope.CaptureTf, CaptureTf)
        self.assertEqual([f.name for f in dataclasses.fields(CaptureInfo)],
                         ["stamp_ns", "optical_frame", "calibration_id", "width", "height"])
        self.assertEqual([f.name for f in dataclasses.fields(CaptureTf)], ["parent", "child", "stamp_ns", "T_W_C"])


class ListEnvelopeTest(unittest.TestCase):
    """봉투 v2: 버전·형·상관 키·기록."""

    def test_make_and_record(self):
        """v2 봉투는 B correlation_keys와 같은 키를 고정하고, 기록은 JSON 안전하며 후보 원 인덱스·끝점을 싣는다."""
        obs = frame_obs(candidates=(bolt(1), bolt(4, points=((5.0, 5.0), (30.0, 40.0)))))
        env = make_list_envelope(obs)
        view = SimpleNamespace(run_id=obs.frame.run_id, capture_generation=obs.frame.capture_generation, generation=0,
                               job_seq=3, detector_config_id="a" * 64)

        self.assertEqual(env.version, LIST_ENVELOPE_VERSION)
        self.assertEqual(dict(env.correlation), correlation_keys(view, obs.frame))
        rec = env.to_record()

        self.assertEqual(rec["version"], "eyetoarm.b_input/2")
        self.assertEqual(rec["observations_version"], FRAME_OBSERVATIONS_VERSION)
        self.assertEqual(rec["candidate_count"], 2)
        self.assertEqual([c["instance_index"] for c in rec["candidates"]], [1, 4])
        self.assertEqual(rec["candidates"][1]["axis_endpoints_px"], [[5.0, 5.0], [30.0, 40.0]])
        self.assertEqual(rec["tf"], {"parent": "world", "child": OPTICAL, "stamp_ns": STAMP})

        import json
        json.dumps(rec, allow_nan=False)

    def test_bad_version_and_type(self):
        """다른 버전·FrameObservations 아님은 ListEnvelopeError(ValueError)."""
        with self.assertRaises(ListEnvelopeError):
            make_list_envelope(frame_obs(), version="eyetoarm.b_input/1")

        with self.assertRaises(ListEnvelopeError):
            make_list_envelope("not a list")

    def test_rigid_tolerance_reported_values(self):
        """U-6: 정규직교 허용오차는 봉투 1 기준 그대로(R Rᵀ atol 1e-6 안은 허용, 밖은 거부)."""
        near = T_W_C.copy()
        near[:3, :3] *= 1 + 2e-7
        far = T_W_C.copy()
        far[:3, :3] *= 1 + 1e-5

        self.assertIsNotNone(frame_obs(tf=tf(T_W_C=near)).tf)

        with self.assertRaises(ValueError):
            frame_obs(tf=tf(T_W_C=far))


if __name__ == "__main__":
    unittest.main()
