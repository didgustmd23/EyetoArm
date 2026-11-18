"""A→B 입력 봉투(`vision/observation_envelope`, #168 「전달 자료의 확장성」) 단위시험.

근거: `_workspace/168_a_to_b_real_path_implementation_spec.md` 「전달 자료의 확장성」·「확장 봉투 시험」.
- 필수 필드 전달, 선택 확장의 추가·부재, 모르는 확장의 무해한 처리(이름만 남기고 내용 버림), 잘못된 버전·타입·좌표 기준·
  stamp 결합의 명시적 실패, B에는 frame·관측만 넘어가고 TF는 봉투 것만 쓰는지를 확인한다.
- 값은 합성(카메라 from_fov, 고정 자세)이며 정확도와 무관하다.
"""
import json
import unittest

import numpy as np

from eyetoarm.calibration.profile import from_fov
from eyetoarm.common.types import PixelBasis, PlanarObservation
from eyetoarm.transforms.b_stage import correlation_keys
from eyetoarm.transforms.planar_transform import look_at
from eyetoarm.transforms.observation_envelope_adapter import ObservationEnvelopeAdapter, EnvelopeTfLookup
from eyetoarm.vision.observation_envelope import (OBSERVATION_ENVELOPE_VERSION, ObservationEnvelopeError, CaptureInfo, CaptureTf, SUPPORTED_EXTENSIONS,
                                     make_observation_envelope)
from test_pipeline import RUN_ID, frame

W, H = 320, 240
CAM = from_fov(W, H, 1.0, "link_cam_optical_frame")
T = look_at((0.45, 0.0, 0.45), (0.5, 0.0, 0.10))
STAMP = 4_291_000_000


def make_frame(**over):
    """봉투 시험 프레임(보정 번호·광학 프레임·촬영 시각이 카메라와 맞음)."""
    fields = dict(image=np.zeros((H, W, 3), np.uint8), width=W, height=H, calibration_id=CAM.calibration_id,
                  optical_frame=CAM.frame_id, ros_stamp_ns=STAMP)
    fields.update(over)

    return frame(0, **fields)


def make_obs(f, **over):
    """그 프레임의 유효 볼트 축 관측."""
    fields = dict(run_id=f.run_id, source_id=f.source_id, image_seq=f.image_seq, capture_generation=f.capture_generation,
                  generation=0, job_seq=0, detector_config_id="c" * 64, pixel_basis=PixelBasis.RAW,
                  calibration_id=f.calibration_id, observed_face="axis", calc_contour=None, display_mask_contour=None,
                  candidate_count=1, selected_index=0, valid=True, reason=None, process_ms=1.0, method="ai",
                  class_name="bolt", confidence=0.9, model_id="m", axis_endpoints_px=[[100.0, 120.0], [140.0, 120.0]])
    fields.update(over)

    return PlanarObservation(**fields)


def capture(**over):
    """카메라와 맞는 촬영 정보."""
    fields = dict(stamp_ns=STAMP, optical_frame=CAM.frame_id, calibration_id=CAM.calibration_id, width=W, height=H)
    fields.update(over)

    return CaptureInfo(**fields)


def tf(**over):
    """촬영 시각 TF."""
    fields = dict(parent="world", child=CAM.frame_id, stamp_ns=STAMP, T_W_C=T)
    fields.update(over)

    return CaptureTf(**fields)


def binding(obs, stamp=STAMP):
    """확장 결합 키."""
    return {"image_seq": obs.image_seq, "job_seq": obs.job_seq, "generation": obs.generation, "capture_stamp_ns": stamp}


def build(f=None, obs=None, **kw):
    """기본값으로 봉투를 만든다."""
    f = f or make_frame()
    obs = obs or make_obs(f)
    args = dict(capture=capture(), camera=CAM, tf=tf())
    args.update(kw)

    return make_observation_envelope(f, obs, **args)


class RequiredFieldsTest(unittest.TestCase):
    """필수 필드 전달."""

    def test_envelope_carries_required_fields(self):
        """봉투는 같은 프레임·관측 객체, 촬영 정보, 읽기 전용 TF, correlation_keys와 같은 상관 키, 버전을 담고 끝점은 관측 안에 그대로다."""
        f = make_frame()
        obs = make_obs(f)
        env = build(f, obs)

        self.assertIs(env.frame, f)
        self.assertIs(env.observation, obs)
        self.assertEqual(env.version, OBSERVATION_ENVELOPE_VERSION)
        self.assertEqual(dict(env.correlation), correlation_keys(obs, f))
        self.assertFalse(env.tf.T_W_C.flags.writeable)
        np.testing.assert_array_equal(env.tf.T_W_C, T)
        self.assertEqual((dict(env.extensions), dict(env.unsupported)), ({}, {}))

    def test_record_is_json_without_image_bytes(self):
        """to_record는 JSON 직렬화 가능(allow_nan=False)하고 영상은 지문·모양만 담는다."""
        rec = build().to_record()
        text = json.dumps(rec, allow_nan=False)

        self.assertEqual(rec["image_shape"], [H, W, 3])
        self.assertEqual(len(rec["image_sha256"]), 64)
        self.assertLess(len(text), 2000)

    def test_tf_none_allowed(self):
        """그 시각 TF가 없으면 tf=None 봉투가 만들어진다(B가 TF_UNAVAILABLE로 판정)."""
        self.assertIsNone(build(tf=None).tf)


class BindingErrorTest(unittest.TestCase):
    """버전·타입·좌표 기준·stamp 결합 오류는 명시적 ObservationEnvelopeError."""

    def test_version_and_types(self):
        """다른 버전, 프레임·관측·촬영 정보 타입 오류는 거부된다."""
        f = make_frame()

        for name, kw in (("version", dict(version="eyetoarm.a_to_b_envelope/2")), ("capture_type", dict(capture={"stamp_ns": 1})),
                         ("tf_type", dict(tf={"T": T}))):
            with self.subTest(case=name):
                with self.assertRaises(ObservationEnvelopeError):
                    build(f, **kw)

        with self.assertRaises(ObservationEnvelopeError):
            make_observation_envelope("frame", make_obs(f), capture=capture(), camera=CAM, tf=tf())

        with self.assertRaises(ObservationEnvelopeError):
            make_observation_envelope(f, {"valid": True}, capture=capture(), camera=CAM, tf=tf())

    def test_frame_capture_camera_mismatches(self):
        """촬영 시각·광학 프레임·크기·보정 번호·카메라 프로파일·영상 모양·관측 식별자가 어긋나면 거부된다."""
        f = make_frame()
        cases = {
            "stamp": dict(capture=capture(stamp_ns=STAMP + 1)),
            "optical_frame": dict(capture=capture(optical_frame="other")),
            "size": dict(capture=capture(width=W + 1)),
            "calibration": dict(capture=capture(calibration_id="cal-x")),
            "camera": dict(camera=from_fov(W, H, 0.9, CAM.frame_id)),
        }

        for name, kw in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(ObservationEnvelopeError):
                    build(f, **kw)

        with self.assertRaises(ObservationEnvelopeError):
            build(make_frame(image=np.zeros((H, W + 1, 3), np.uint8)))

        g = make_frame()
        with self.assertRaises(ObservationEnvelopeError):
            build(g, make_obs(g, image_seq=5))

    def test_tf_binding_errors(self):
        """TF 부모·자식 프레임·시각이 촬영 정보와 다르거나 강체가 아니면 거부된다."""
        bad = np.eye(4)
        bad[0, 0] = 2.0

        for name, kw in (("parent", dict(parent="map")), ("child", dict(child="other")),
                         ("stamp", dict(stamp_ns=STAMP + 66_000_000)), ("rigid", dict(T_W_C=bad)),
                         ("shape", dict(T_W_C=np.eye(3)))):
            with self.subTest(case=name):
                with self.assertRaises(ObservationEnvelopeError):
                    build(tf=tf(**kw))


class ExtensionTest(unittest.TestCase):
    """확장 필드: 추가·부재·모르는 이름·형식 오류."""

    def test_supported_extensions_accepted_and_normalized(self):
        """ai_instances·tracking은 결합·좌표 기준을 검사해 받고(polygon 읽기 전용), 봉투 기록에 이름만 남는다."""
        f = make_frame()
        obs = make_obs(f)
        ext = {"ai_instances": {"binding": binding(obs), "pixel_basis": "raw",
                                "items": [{"index": 0, "class_name": "bolt", "confidence": 0.9, "rejected": None,
                                           "polygon_px": [[90, 110], [150, 110], [150, 130], [90, 130]]}]},
               "tracking": {"binding": binding(obs), "part_track_id": 7}}
        env = build(f, obs, extensions=ext)

        self.assertEqual(sorted(env.extensions), ["ai_instances", "tracking"])
        self.assertFalse(env.extensions["ai_instances"]["items"][0]["polygon_px"].flags.writeable)
        rec = env.to_record()
        self.assertEqual(rec["extensions"]["tracking"]["part_track_id"], 7)
        self.assertEqual(rec["extensions"]["ai_instances"]["items"][0]["polygon_px"][0], [90.0, 110.0])
        json.dumps(rec, allow_nan=False)
        self.assertEqual(set(SUPPORTED_EXTENSIONS), {"ai_instances", "tracking"})

    def test_unknown_extension_preserved_not_interpreted(self):
        """모르는 확장은 내용을 JSON 정규 문자열로 그대로 unsupported에 보존하고(해석 안 함), 핵심 필드·상관 키는 확장이 없을 때와 같다.
        JSON으로 표현할 수 없는 미지원 값은 거부한다."""
        f = make_frame()
        obs = make_obs(f)
        plain = build(f, obs)
        env = build(f, obs, extensions={"depth_map": {"anything": [1, 2, 3]}})

        self.assertEqual(dict(env.unsupported), {"depth_map": '{"anything":[1,2,3]}'})
        self.assertNotIn("depth_map", env.extensions)
        self.assertEqual(env.to_record()["unsupported"], {"depth_map": {"anything": [1, 2, 3]}})
        self.assertEqual(dict(env.correlation), dict(plain.correlation))
        self.assertIs(env.observation, plain.observation)

        for bad in ({"x": float("nan")}, {"x": object()}):
            with self.subTest(bad=bad):
                with self.assertRaises(ObservationEnvelopeError):
                    build(f, obs, extensions={"depth_map": bad})

    def test_malformed_known_extension_rejected(self):
        """결합 키 불일치(다른 stamp·job), 좌표 기준 불일치, 프레임 밖 polygon, 신뢰도 범위 밖, 추적 번호 형식 오류는 거부된다."""
        f = make_frame()
        obs = make_obs(f)

        def inst(**over):
            """ai_instances 확장 하나."""
            item = {"index": 0, "class_name": "bolt", "confidence": 0.9, "rejected": None,
                    "polygon_px": [[90, 110], [150, 110], [150, 130]]}
            item.update(over.pop("item", {}))
            value = {"binding": binding(obs), "pixel_basis": "raw", "items": [item]}
            value.update(over)
            return {"ai_instances": value}

        cases = {
            "stamp_binding": inst(binding=binding(obs, stamp=STAMP + 1)),
            "job_binding": inst(binding={**binding(obs), "job_seq": 9}),
            "basis": inst(pixel_basis="undistorted"),
            "out_of_frame": inst(item={"polygon_px": [[0, 0], [W, 0], [0, 10]]}),
            "confidence": inst(item={"confidence": 1.5}),
            "tracking": {"tracking": {"binding": binding(obs), "part_track_id": "7"}},
            "not_mapping": [("tracking", 1)],
        }

        for name, ext in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(ObservationEnvelopeError):
                    build(f, obs, extensions=ext)


class StrictNumberTest(unittest.TestCase):
    """확장·TF 수치는 변환 전에 형을 검사하고 변환 오류를 모두 ObservationEnvelopeError로 정리한다(#168 F3 실패 우선 회귀)."""

    def inst_ext(self, obs, **item_over):
        """ai_instances 확장 하나(항목 일부 덮어쓰기)."""
        item = {"index": 0, "class_name": "bolt", "confidence": 0.9, "rejected": None,
                "polygon_px": [[90, 110], [150, 110], [150, 130]]}
        item.update(item_over)

        return {"ai_instances": {"binding": binding(obs), "pixel_basis": "raw", "items": [item]}}

    def test_bad_confidence_values(self):
        """confidence가 거대 정수(10**400)·문자열·bool·복소수·NaN·None이면 OverflowError 등이 새지 않고 ObservationEnvelopeError다."""
        f = make_frame()
        obs = make_obs(f)

        for bad in (10 ** 400, "0.9", True, 0.5 + 0j, float("nan"), None):
            with self.subTest(confidence=repr(bad)[:20]):
                with self.assertRaises(ObservationEnvelopeError):
                    build(f, obs, extensions=self.inst_ext(obs, confidence=bad))

    def test_bad_polygon_values(self):
        """polygon 좌표가 문자열·bool·복소수·비유한·거대 정수·들쭉날쭉·문자열 dtype 배열이면 조용히 변환하지 않고 ObservationEnvelopeError다."""
        f = make_frame()
        obs = make_obs(f)
        cases = {
            "str": [["90", "110"], ["150", "110"], ["150", "130"]],
            "bool": [[True, 110], [150, 110], [150, 130]],
            "complex": [[90 + 1j, 110], [150, 110], [150, 130]],
            "nan": [[float("nan"), 110], [150, 110], [150, 130]],
            "inf": [[float("inf"), 110], [150, 110], [150, 130]],
            "huge": [[10 ** 400, 110], [150, 110], [150, 130]],
            "ragged": [[90, 110], [150], [150, 130]],
            "str_array": np.array([["90", "110"], ["150", "110"], ["150", "130"]]),
            "bool_array": np.ones((3, 2), bool),
        }

        for name, poly in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(ObservationEnvelopeError):
                    build(f, obs, extensions=self.inst_ext(obs, polygon_px=poly))

        # 정상 정수·실수·numpy 정수 배열은 받는다
        env = build(f, obs, extensions=self.inst_ext(obs, polygon_px=np.array([[90, 110], [150, 110], [150, 130]])))
        self.assertEqual(env.extensions["ai_instances"]["items"][0]["polygon_px"].dtype, np.float64)

    def test_bad_binding_and_tracking_types(self):
        """결합 키·추적 번호에 bool(True == 1)·문자열이 오면 ObservationEnvelopeError다."""
        f = make_frame()
        obs = make_obs(f)

        for name, ext in (("binding_bool", {"tracking": {"binding": {**binding(obs), "generation": False},
                                                         "part_track_id": 1}}),
                          ("binding_str", {"tracking": {"binding": {**binding(obs), "image_seq": "0"},
                                                        "part_track_id": 1}}),
                          ("track_bool", {"tracking": {"binding": binding(obs), "part_track_id": True}})):
            with self.subTest(case=name):
                with self.assertRaises(ObservationEnvelopeError):
                    build(f, obs, extensions=ext)

    def test_tf_matrix_strict(self):
        """TF 행렬에 문자열·bool 원소가 있으면 ObservationEnvelopeError다(float 자동 변환 없음)."""
        rows = T.tolist()
        rows[0][3] = str(rows[0][3])

        with self.assertRaises(ObservationEnvelopeError):
            build(tf=tf(T_W_C=rows))

        with self.assertRaises(ObservationEnvelopeError):
            build(tf=tf(T_W_C=np.eye(4, dtype=bool)))


class AdapterTest(unittest.TestCase):
    """B 입구 어댑터: 봉투 전체 수령, 계산은 frame·관측만, TF는 봉투 것만."""

    def stage(self, lookup, seen):
        """consume 인자와 그 시점 TF 조회를 기록하는 B 흉내."""
        class Stage:
            """B 단계 흉내."""

            tf_lookup = lookup

            def consume(self, frame_arg, obs_arg, *extra, **kw):
                """인자 기록."""
                seen.update(args=(frame_arg, obs_arg), extra=(extra, kw), tf=lookup(frame_arg),
                            other=lookup(make_frame()))
                return "out"

        return Stage()

    def test_adapter_receives_extensions_and_computes_with_frame_obs(self):
        """어댑터는 지원·미지원 확장을 그대로 결과에 돌려주고(used_in_compute 빈 튜플), 계산은 stage.consume(frame, obs)만 부르며 그 동안만 봉투 TF를 준다."""
        f = make_frame()
        obs = make_obs(f)
        env = build(f, obs, extensions={"tracking": {"binding": binding(obs), "part_track_id": 3},
                                        "depth_hint": [1, 2]})
        lookup = EnvelopeTfLookup()
        seen = {}
        adapter = ObservationEnvelopeAdapter(self.stage(lookup, seen), lookup)
        result = adapter.consume(env)

        self.assertEqual(result.output, "out")
        self.assertIs(seen["args"][0], f)
        self.assertIs(seen["args"][1], obs)
        self.assertEqual(seen["extra"], ((), {}))
        self.assertEqual(result.extensions["tracking"]["part_track_id"], 3)
        self.assertEqual(dict(result.unsupported), {"depth_hint": "[1,2]"})
        self.assertEqual(result.used_in_compute, ())
        self.assertEqual((adapter.received_extensions, adapter.received_unsupported), ({"tracking": 1}, {"depth_hint": 1}))
        np.testing.assert_array_equal(seen["tf"], T)
        self.assertIsNone(seen["other"])
        self.assertIsNone(lookup(f))   # 소비가 끝나면 봉투 TF를 주지 않는다

    def test_adapter_rejects_non_envelope_and_foreign_lookup(self):
        """봉투가 아닌 값은 거부(B 미호출)하고, stage의 tf_lookup이 어댑터 것이 아니면 생성 단계에서 거부한다."""
        lookup = EnvelopeTfLookup()
        seen = {}
        adapter = ObservationEnvelopeAdapter(self.stage(lookup, seen), lookup)

        with self.assertRaises(ObservationEnvelopeError):
            adapter.consume({"frame": None})

        self.assertEqual(seen, {})

        with self.assertRaises(ValueError):
            ObservationEnvelopeAdapter(self.stage(EnvelopeTfLookup(), {}), lookup)


if __name__ == "__main__":
    unittest.main()
