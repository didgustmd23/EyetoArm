"""YOLO 모델 출력·표시 결과 자료형(`RawSegOutput`·`AiInstance`·`AiSegResult`·`ModelPrepareError`) 시험."""
import unittest

import numpy as np

from eyetoarm.common.types import InvalidationKind
from eyetoarm.vision.ai_result import AiInstance, AiSegResult, ModelPrepareError, RawSegOutput

SQUARE = [[10, 5], [19, 5], [19, 14], [10, 14]]
CFG_SHA = "c" * 64


def instance(index=0, **over):
    """정상(eligible) 인스턴스 하나. 키워드로 필드를 바꾼다."""
    fields = {"index": index, "class_index": 0, "class_name": "bolt", "confidence": 0.9,
              "box_xyxy": (10.0, 5.0, 19.0, 14.0), "polygon": SQUARE, "area_px": 100.0,
              "component_count": 1, "hole_count": 0, "clipped": False, "rejected": None}
    fields.update(over)

    return AiInstance(**fields)


def seg_result(instances=(), **over):
    """표시 결과 하나."""
    fields = {"instances": tuple(instances), "model_id": "yolo11n-seg@" + "a" * 16, "config_sha256": CFG_SHA,
              "roi": None, "crop_wh": (64, 48), "infer_ms": 5.0, "post_ms": 1.0, "process_ms": 7.0}
    fields.update(over)

    return AiSegResult(**fields)


class AiInstanceTest(unittest.TestCase):
    """인스턴스 필드 검증·소유권·polygon 규칙."""

    def test_instance_polygon_owned_readonly_float32(self):
        """polygon은 float32 (N,2) 소유 복사·읽기 전용이라 외부 배열을 바꿔도 변하지 않는다."""
        src = np.array(SQUARE, dtype=np.int64)
        inst = instance(polygon=src)
        src[0, 0] = 999

        self.assertEqual((inst.polygon.dtype, inst.polygon.shape), (np.float32, (4, 2)))
        self.assertEqual(float(inst.polygon[0, 0]), 10.0)
        self.assertFalse(inst.polygon.flags.writeable)
        self.assertTrue(inst.in_scope and inst.eligible)

    def test_polygon_rejection_combinations(self):
        """거부 사유별 polygon 필수·금지·허용 조합을 지킨다."""
        instance(rejected="SMALL_MASK")
        instance(rejected="EMPTY_MASK", polygon=None, area_px=0.0, component_count=0)

        for rejected in ("NON_TARGET_CLASS", "LOW_CONFIDENCE"):
            instance(rejected=rejected)
            instance(rejected=rejected, polygon=None, area_px=1.0)

        for bad in ({"rejected": None, "polygon": None}, {"rejected": "SMALL_MASK", "polygon": None},
                    {"rejected": "EMPTY_MASK"}):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    instance(**bad)

    def test_instance_field_validation(self):
        """rejected·신뢰도·polygon·clipped·정수·bool·박스 형식 위반은 ValueError."""
        bads = [
            {"rejected": "MULTIPLE_OBJECTS"}, {"rejected": ""},
            {"confidence": 1.5}, {"confidence": -0.1}, {"confidence": float("nan")}, {"confidence": True},
            {"polygon": [[0, 0], [1, 1]]}, {"polygon": [[0, 0, 0], [1, 1, 1], [2, 2, 2]]},
            {"polygon": [[0, 0], [1, float("nan")], [2, 2]]},
            {"polygon": None, "rejected": "NON_TARGET_CLASS", "clipped": True},
            {"index": -1}, {"index": True}, {"class_index": -1}, {"class_name": ""},
            {"box_xyxy": (0.0, 0.0, 1.0)}, {"box_xyxy": (5.0, 0.0, 1.0, 1.0)},
            {"box_xyxy": (0.0, 0.0, float("inf"), 1.0)},
            {"area_px": -1.0}, {"component_count": -1}, {"hole_count": 1.0}, {"clipped": 1},
        ]

        for bad in bads:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    instance(**bad)

    def test_in_scope_and_eligible_flags(self):
        """in_scope = 비대상·점수 미달이 아님, eligible = 거부 없음이고 잘리지 않음."""
        cases = [
            ({}, (True, True)),
            ({"clipped": True}, (True, False)),
            ({"rejected": "SMALL_MASK"}, (True, False)),
            ({"rejected": "EMPTY_MASK", "polygon": None}, (True, False)),
            ({"rejected": "NON_TARGET_CLASS"}, (False, False)),
            ({"rejected": "LOW_CONFIDENCE", "polygon": None}, (False, False)),
        ]

        for over, expected in cases:
            with self.subTest(over=over):
                inst = instance(**over)
                self.assertEqual((inst.in_scope, inst.eligible), expected)


class AiSegResultTest(unittest.TestCase):
    """표시 결과 검증·파생 수."""

    def test_counts_in_scope_and_eligible(self):
        """여러 거부 사유가 섞인 목록: candidate_count 4(in_scope), eligible_count 1."""
        r = seg_result([
            instance(0, rejected="NON_TARGET_CLASS", class_index=2, class_name="washer"),
            instance(1, rejected="LOW_CONFIDENCE", polygon=None),
            instance(2, rejected="EMPTY_MASK", polygon=None, area_px=0.0, component_count=0),
            instance(3, rejected="SMALL_MASK", clipped=True),
            instance(4, clipped=True),
            instance(5),
        ])

        self.assertEqual((len(r.instances), r.candidate_count, r.eligible_count), (6, 4, 1))

    def test_mixed_non_target_low_conf_with_empty_or_degenerate(self):
        """비대상·점수 미달·1화소 혼합 목록 → instances 3, candidate 1, eligible 0."""
        r = seg_result([
            instance(0, rejected="NON_TARGET_CLASS", polygon=None, area_px=1.0, class_index=1, class_name="nut"),
            instance(1, rejected="LOW_CONFIDENCE", polygon=None, area_px=0.0, component_count=0, confidence=0.1),
            instance(2, rejected="EMPTY_MASK", polygon=None, area_px=1.0),
        ])

        self.assertEqual((len(r.instances), r.candidate_count, r.eligible_count), (3, 1, 0))
        self.assertEqual([(i.class_name, i.confidence) for i in r.instances],
                         [("nut", 0.9), ("bolt", 0.1), ("bolt", 0.9)])

    def test_result_validation(self):
        """인스턴스 번호·설정 해시·시간·crop_wh·roi 형식을 검사한다."""
        bads = [
            {"instances": (instance(1),)}, {"instances": (instance(0), instance(0))},
            {"instances": [instance(0)], "model_id": ""}, {"config_sha256": "x"},
            {"infer_ms": -1.0}, {"post_ms": float("nan")}, {"process_ms": True},
            {"crop_wh": (0, 10)}, {"crop_wh": (10,)}, {"roi": (0, 0, 10)},
            {"instances": ("not an instance",)},
        ]

        for bad in bads:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    seg_result(**{"instances": (), **bad})

        self.assertEqual(seg_result().candidate_count, 0)
        self.assertIsInstance(seg_result([instance(0)]).instances, tuple)


class RawSegOutputTest(unittest.TestCase):
    """모델 출력 원시 배열의 형식 계약."""

    def raw(self, k=2, h=8, w=10, **over):
        """K개 인스턴스의 정상 원시 출력."""
        fields = {"boxes_xyxy": np.zeros((k, 4), np.float32), "confidence": np.full(k, 0.5, np.float32),
                  "class_index": np.zeros(k, np.int64), "masks": np.zeros((k, h, w), bool), "crop_wh": (w, h),
                  "infer_ms": 3.0}
        fields.update(over)

        return RawSegOutput(**fields)

    def test_normalized_owned_arrays(self):
        """배열은 지정 dtype의 소유 복사·읽기 전용으로 정규화된다. K=0도 허용한다."""
        boxes = np.ones((2, 4), np.float64)
        r = self.raw(boxes_xyxy=boxes)
        boxes[0, 0] = 7

        self.assertEqual((r.boxes_xyxy.dtype, float(r.boxes_xyxy[0, 0])), (np.float32, 1.0))
        self.assertEqual((r.class_index.dtype, r.masks.dtype), (np.int64, np.bool_))
        self.assertFalse(r.masks.flags.writeable)
        self.assertEqual(self.raw(k=0).count, 0)

    def test_raw_output_shape_contract(self):
        """K 불일치·마스크 차원·crop 해상도 불일치·NaN·음수 클래스·bool 아닌 마스크는 ValueError."""
        bads = [
            {"confidence": np.zeros(3, np.float32)}, {"class_index": np.zeros(1, np.int64)},
            {"masks": np.zeros((2, 8), bool)}, {"masks": np.zeros((3, 8, 10), bool)},
            {"masks": np.zeros((2, 9, 10), bool)}, {"crop_wh": (10, 9)},
            {"boxes_xyxy": np.full((2, 4), np.nan, np.float32)}, {"boxes_xyxy": np.zeros((2, 5), np.float32)},
            {"confidence": np.array([0.5, np.inf], np.float32)}, {"class_index": np.array([0, -1])},
            {"class_index": np.array([0.0, 1.0])}, {"masks": np.zeros((2, 8, 10), np.uint8)},
            {"infer_ms": -1.0},
        ]

        for bad in bads:
            with self.subTest(bad=list(bad)):
                with self.assertRaises(ValueError):
                    self.raw(**bad)


class ModelPrepareErrorTest(unittest.TestCase):
    """준비 실패 분류 예외."""

    def test_model_prepare_error_kinds(self):
        """MODEL_LOAD_FAILED·MODEL_CONFIG_INVALID만 허용하고 cause는 비어 있지 않은 문자열이다."""
        e = ModelPrepareError(InvalidationKind.MODEL_LOAD_FAILED, "weights not found: x.pt")

        self.assertEqual((e.kind, e.cause), (InvalidationKind.MODEL_LOAD_FAILED, "weights not found: x.pt"))
        self.assertIn("weights not found", str(e))
        self.assertEqual(ModelPrepareError("MODEL_CONFIG_INVALID", "names").kind,
                         InvalidationKind.MODEL_CONFIG_INVALID)

        for kind, cause in ((InvalidationKind.INFERENCE_ERROR, "x"), (InvalidationKind.JOB_ABORTED, "x"),
                            (InvalidationKind.MODEL_LOAD_FAILED, ""), ("NOPE", "x")):
            with self.subTest(kind=kind, cause=cause):
                with self.assertRaises(ValueError):
                    ModelPrepareError(kind, cause)


class R1BoundaryTest(unittest.TestCase):
    """설정 해시 공백·저장 dtype 범위·면적 0 다각형 경계."""

    def test_result_config_sha_exact(self):
        """AiSegResult.config_sha256에 끝 개행·CRLF·공백이 붙으면 ValueError."""
        for suffix in ("\n", "\r\n", " "):
            with self.subTest(suffix=repr(suffix)):
                with self.assertRaises(ValueError):
                    seg_result(config_sha256=CFG_SHA + suffix)

    def test_raw_values_must_survive_storage_dtype(self):
        """float32 범위를 넘는 유한값·int64 범위를 넘는 uint64 클래스는 ValueError."""
        base = RawSegOutputTest()
        boxes = np.zeros((2, 4), np.float64)
        boxes[0, 2] = 1e39
        conf = np.array([0.5, 1e39], np.float64)
        cls = np.array([0, 2 ** 63], np.uint64)

        for name, value in (("boxes_xyxy", boxes), ("confidence", conf), ("class_index", cls)):
            with self.subTest(field=name):
                with self.assertRaises(ValueError):
                    base.raw(**{name: value})

        ok = base.raw(class_index=np.array([0, 3], np.uint64), boxes_xyxy=np.full((2, 4), 3.0e38, np.float64))
        self.assertEqual((ok.class_index.dtype, int(ok.class_index[1])), (np.int64, 3))
        self.assertTrue(np.all(np.isfinite(ok.boxes_xyxy)))

    def test_zero_area_polygon_is_not_geometry(self):
        """면적 0인 다각형은 polygon으로 받지 않고, 시계·반시계 정상 다각형은 받는다."""
        for name, poly in (("collinear", [[0, 0], [1, 1], [2, 2]]), ("repeated", [[1, 1], [1, 1], [1, 1]]),
                           ("back and forth", [[0, 0], [4, 0], [0, 0], [4, 0]])):
            for rejected in (None, "SMALL_MASK", "NON_TARGET_CLASS", "LOW_CONFIDENCE"):
                with self.subTest(poly=name, rejected=rejected):
                    with self.assertRaises(ValueError):
                        instance(polygon=poly, rejected=rejected)

        for name, poly in (("clockwise", SQUARE), ("counterclockwise", SQUARE[::-1]),
                           ("triangle", [[0, 0], [4, 0], [0, 3]])):
            with self.subTest(poly=name):
                self.assertTrue(instance(polygon=poly).eligible)

        self.assertIsNone(instance(polygon=None, rejected="NON_TARGET_CLASS").polygon)


if __name__ == "__main__":
    unittest.main()
