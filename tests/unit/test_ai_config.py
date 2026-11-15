"""YOLO 연결 T1 — AI 분할 설정(`AiSegConfig`)·엄격 JSON 읽기·안정 설정 ID·model_id 시험.

근거: 계획 `docs/superpowers/plans/2026-11-15-yolo-pipeline.md` r2 §4.1·§4.8·T1, A 상세설계 §5.4(결과에 영향을 주는
검출 설정: 방식·모델·가중치·대상 클래스·신뢰도 기준·입력 크기·추론 장치), §7.1(모델 설정 기록 항목).
torch·ultralytics 없이 돈다. 가중치 파일을 읽지 않는다(해시 대조는 준비 단계 T5).
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

from eyetoarm.vision.ai_config import AiSegConfig, load_ai_config
from test_pipeline import config as planar_config

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SHA_A = "a" * 64
SHA_B = "b" * 64

# 시험용 기본 AI 설정 필드(JSON 파일 형식, class_map 키는 문자열)
AI_FIELDS = {
    "format_version": 1,
    "method": "ai",
    "model_family": "yolo11n-seg",
    "weights_path": "outputs/pilot/weights/best.pt",
    "weights_sha256": SHA_A,
    "class_map": {"0": "bolt", "1": "nut"},
    "target_classes": ["bolt", "nut"],
    "conf_threshold": 0.25,
    "iou_threshold": 0.7,
    "imgsz": 1280,
    "max_det": 300,
    "device": "0",
    "half": False,
    "retina_masks": True,
    "roi": None,
    "min_mask_area_px": 20,
    "border_margin_px": 2,
}


def ai_config(**over):
    """`AI_FIELDS`에서 키워드로 준 값만 바꾼 `AiSegConfig`."""
    fields = dict(AI_FIELDS)
    fields.update(over)

    return AiSegConfig(**fields)


class AiConfigTest(unittest.TestCase):
    """생성자 검증·정규화·해시."""

    def test_valid_config_normalized(self):
        """정상 설정은 class_map을 (번호, 이름) 정렬 튜플, target을 정렬 튜플, 실수를 float로 정규화한다."""
        c = ai_config(class_map={1: "nut", "0": "bolt"}, target_classes=["nut", "bolt"], conf_threshold=1 / 4)

        self.assertEqual(c.class_map, ((0, "bolt"), (1, "nut")))
        self.assertEqual(c.target_classes, ("bolt", "nut"))
        self.assertEqual((c.conf_threshold, type(c.conf_threshold)), (0.25, float))
        self.assertEqual(c.roi, None)
        self.assertEqual(c.class_names, {0: "bolt", 1: "nut"})
        self.assertEqual(len(c.config_sha256), 64)

    def test_stable_hash_and_what_changes_it(self):
        """키 순서·class_map 형식이 달라도 같은 해시, weights_path만 바꾸면 같은 해시, 결과에 영향을 주는 값을 바꾸면 다른 해시."""
        base = ai_config()

        self.assertEqual(ai_config(class_map={1: "nut", 0: "bolt"}).config_sha256, base.config_sha256)
        self.assertEqual(ai_config(weights_path="/elsewhere/best.pt").config_sha256, base.config_sha256)

        for over in ({"weights_sha256": SHA_B}, {"imgsz": 640}, {"conf_threshold": 0.10}, {"device": "cpu"},
                     {"class_map": {"0": "bolt", "1": "washer"}, "target_classes": ["bolt"]},
                     {"target_classes": ["nut"]}, {"iou_threshold": 0.5}, {"max_det": 10}, {"half": True},
                     {"roi": [0, 0, 640, 360]}, {"min_mask_area_px": 0}, {"border_margin_px": 0},
                     {"model_family": "yolo26n-seg"}):
            with self.subTest(over=over):
                self.assertNotEqual(ai_config(**over).config_sha256, base.config_sha256)

    def test_conf_threshold_change_changes_config_id(self):
        """신뢰도 기준(0.10/0.25/0.50)마다 설정 ID가 달라진다(계획 §4.8: 조절은 설정 ID 변경·세대 전이로 이어짐)."""
        ids = {ai_config(conf_threshold=v).config_sha256 for v in (0.10, 0.25, 0.50)}

        self.assertEqual(len(ids), 3)

    def test_all_three_candidate_families_accepted(self):
        """사용자 확정 비교 후보 3개(YOLOv8n-seg·YOLO11n-seg·YOLO26n-seg)를 모두 받는다(정책 축소 없음)."""
        for family in ("yolov8n-seg", "yolo11n-seg", "yolo26n-seg"):
            with self.subTest(family=family):
                self.assertEqual(ai_config(model_family=family).model_family, family)

    def test_model_id_format(self):
        """model_id는 「모델 계열@가중치 sha256 앞 16자」다."""
        self.assertEqual(ai_config().model_id, "yolo11n-seg@" + "a" * 16)

    def test_hash_space_differs_from_planar(self):
        """AI 정규형은 method 키를 담아 PlanarConfig 해시와 겹치지 않는다."""
        c = ai_config()

        self.assertIn('"method":"ai"', c.canonical_json())
        self.assertNotIn('"weights_path"', c.canonical_json())
        self.assertNotEqual(c.config_sha256, planar_config().config_sha256)

    def test_rejects_bad_values(self):
        """형·범위가 어긋난 값, bool이 섞인 숫자, 비유한 수, target이 class_map 밖, retina_masks=false는 ValueError."""
        bads = [
            {"format_version": 2}, {"format_version": True}, {"method": "opencv"},
            {"model_family": "yolov5n-seg"}, {"model_family": ""},
            {"weights_path": ""}, {"weights_path": 3},
            {"weights_sha256": "A" * 64}, {"weights_sha256": "a" * 63}, {"weights_sha256": "g" * 64},
            {"class_map": {}}, {"class_map": {"-1": "bolt"}}, {"class_map": {"01": "bolt"}},
            {"class_map": {"x": "bolt"}}, {"class_map": {True: "bolt"}}, {"class_map": {"0": ""}},
            {"class_map": {"0": "bolt", "1": "bolt"}}, {"class_map": [["0", "bolt"]]},
            {"target_classes": []}, {"target_classes": ["washer"]}, {"target_classes": ["bolt", "bolt"]},
            {"target_classes": "bolt"},
            {"conf_threshold": 0}, {"conf_threshold": 1}, {"conf_threshold": float("nan")},
            {"conf_threshold": True}, {"conf_threshold": "0.25"},
            {"iou_threshold": 0}, {"iou_threshold": 1.5}, {"iou_threshold": float("inf")},
            {"imgsz": 0}, {"imgsz": 1000}, {"imgsz": 4128}, {"imgsz": 640.0}, {"imgsz": True},
            {"max_det": 0}, {"max_det": 1001}, {"max_det": 1.0},
            {"device": "cuda:0"}, {"device": ""}, {"device": 0}, {"device": "-1"},
            {"half": 0}, {"retina_masks": False}, {"retina_masks": 1},
            {"roi": [0, 0, 0, 10]}, {"roi": [-1, 0, 10, 10]}, {"roi": [0, 0, 10]}, {"roi": [0, 0, 10.0, 10]},
            {"min_mask_area_px": -1}, {"min_mask_area_px": 1.5}, {"border_margin_px": -1},
            {"border_margin_px": True},
        ]

        for bad in bads:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    ai_config(**bad)

    def test_rebuild_from_fields_gives_same_hash(self):
        """정규화된 필드로 다시 만들면 같은 해시다(코어 재검증 `_revalidated_config`가 쓰는 성질)."""
        c = ai_config(roi=[10, 20, 640, 360])

        self.assertEqual(AiSegConfig(**c.init_fields()).config_sha256, c.config_sha256)


class LoadAiConfigTest(unittest.TestCase):
    """엄격 JSON 읽기."""

    def write(self, text):
        """임시 파일에 쓰고 경로를 돌려준다."""
        d = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        path = os.path.join(d, "ai.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def test_round_trip(self):
        """파일로 쓴 설정을 읽으면 같은 해시다."""
        c = load_ai_config(self.write(json.dumps(AI_FIELDS)))

        self.assertEqual(c.config_sha256, ai_config().config_sha256)

    def test_rejects_malformed_json(self):
        """중복 키·NaN/Infinity·비유한 수·알 수 없는 키·빠진 키·객체 아님·구문 오류는 ValueError."""
        good = json.dumps(AI_FIELDS)
        texts = [
            good[:-1] + ', "imgsz": 640}',
            good.replace('"conf_threshold": 0.25', '"conf_threshold": NaN'),
            good.replace('"iou_threshold": 0.7', '"iou_threshold": 1e400'),
            json.dumps({**AI_FIELDS, "extra": 1}),
            json.dumps({k: v for k, v in AI_FIELDS.items() if k != "device"}),
            "[]",
            "{",
        ]

        for text in texts:
            with self.subTest(text=text[-40:]):
                with self.assertRaises(ValueError):
                    load_ai_config(self.write(text))

    def test_missing_file_is_oserror(self):
        """설정 파일이 없으면 OSError 그대로(CLI가 설정 오류 종료 2로 바꾼다)."""
        with self.assertRaises(OSError):
            load_ai_config(os.path.join(tempfile.gettempdir(), "no-such-ai-config.json"))

    def test_module_imports_no_torch_or_ultralytics(self):
        """ai_config를 불러도 torch·ultralytics가 로드되지 않는다(OpenCV 단독 실행 유지)."""
        code = ("import sys, eyetoarm.vision.ai_config\n"
                "print(','.join(k for k in sorted(sys.modules) if k.split('.')[0] in ('torch', 'ultralytics')) "
                "or 'none')")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                             env={"PYTHONPATH": os.path.join(ROOT, "src"), "PATH": ""}, cwd=ROOT)

        self.assertEqual(out.stdout.strip(), "none")


class ExactFormatTest(unittest.TestCase):
    """r1 F2: 해시·장치·클래스 키는 전체 문자열이 선언한 형식이어야 한다(끝 개행·CRLF·공백 거부)."""

    def test_trailing_newline_crlf_space_rejected_in_constructor(self):
        """생성자: weights_sha256·device·class_map 키에 끝 개행·CRLF·앞뒤 공백이 붙으면 ValueError."""
        bads = []
        for suffix in ("\n", "\r\n", " "):
            bads += [{"weights_sha256": SHA_A + suffix}, {"device": "0" + suffix}, {"device": "cpu" + suffix},
                     {"class_map": {"0" + suffix: "bolt", "1": "nut"}}]
        bads += [{"weights_sha256": " " + SHA_A}, {"device": " cpu"}]

        for bad in bads:
            with self.subTest(bad=repr(bad)):
                with self.assertRaises(ValueError):
                    ai_config(**bad)

    def test_trailing_newline_rejected_in_json_loader(self):
        """JSON 로더: 클래스 키 "0\\n"·해시 끝 개행·장치 끝 개행을 담은 파일은 ValueError."""
        loader = LoadAiConfigTest()
        loader.addCleanup = self.addCleanup

        for over in ({"class_map": {"0\n": "bolt", "1": "nut"}}, {"weights_sha256": SHA_A + "\n"},
                     {"device": "0\n"}):
            with self.subTest(over=repr(over)):
                with self.assertRaises(ValueError):
                    load_ai_config(loader.write(json.dumps({**AI_FIELDS, **over})))


if __name__ == "__main__":
    unittest.main()
