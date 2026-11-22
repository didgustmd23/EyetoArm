"""run_pipeline 종류별 시야 보정 시작 대조·전달(#270·#285).

B 규격은 시험용 사본(`part_spec_copies`: B-12 전·후)을 `PART_SPECS_PATH`·`specs_path`로 주입한다.
저장소 B 파일을 그대로 쓰는 시험은 「불러지고 시작 검사가 통과한다」만 본다.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from eyetoarm.transforms.work_result import load_specs
from eyetoarm.vision.ai_config import ViewBiasConfig
from eyetoarm.vision.view_bias import ViewBiasPlan
from part_spec_copies import post_b12, pre_b12, write_specs
from test_run_pipeline_capture_observation import Base as CaptureBase
from test_view_bias import model_id_of

ROOT = Path(__file__).resolve().parents[2]
SPECS = ROOT / "config" / "calibration" / "part_specs.json"


def write_spec_config(src_path, dst_path, **over) -> str:
    """기존 AI 설정 파일에 종류별 형식 `axis_view_bias`를 더한 새 파일."""
    doc = json.loads(Path(src_path).read_text(encoding="utf-8"))
    by_spec = {"BOLT_A": {"label": "M3x20", "k_mm": 0.62, "fit_plane_z_m": 0.102},
               "BOLT_B": {"label": "M6x40", "k_mm": 1.39, "fit_plane_z_m": 0.105}}
    by_spec.update(over)
    doc["axis_view_bias"] = {"model": "C1", "valid_for_model_id": model_id_of(doc["weights_path"]), "by_spec": by_spec}
    Path(dst_path).write_text(json.dumps(doc), encoding="utf-8")

    return str(dst_path)


class SpecStartTest(CaptureBase):
    """저장 입력 단계 T: 종류별 설정의 시작 대조·B 입력 기록."""

    def b12_specs_file(self) -> Path:
        """B-12 뒤 규격 사본(시험 임시 파일)."""
        return write_specs(post_b12(), Path(self.tmp.name) / "specs_b12.json")

    def pre_b12_specs_file(self) -> Path:
        """B-12 전 규격 사본(시험 임시 파일)."""
        return write_specs(pre_b12(), Path(self.tmp.name) / "specs_pre_b12.json")

    def test_runtime_legacy_passthrough_and_plan(self):
        """기존 형식은 같은 객체, 종류별 형식은 B 규격 지문을 가진 계획, B-12 전 규격이면 ValueError(경로 포함)."""
        from eyetoarm.vision.ai_config import load_ai_config

        legacy = ViewBiasConfig("C1", 1.39, 0.105, "m@x")
        self.assertIs(self.mod._view_bias_runtime(legacy), legacy)
        self.assertIsNone(self.mod._view_bias_runtime(None))

        c = self.case()
        cfg = load_ai_config(write_spec_config(c.ai, Path(self.tmp.name) / "ai_spec.json")).axis_view_bias
        plan = self.mod._view_bias_runtime(cfg, self.b12_specs_file())
        self.assertIsInstance(plan, ViewBiasPlan)
        self.assertEqual(len(plan.part_specs_sha256), 64)

        with self.assertRaisesRegex(ValueError, "label 'M3x20' != B 'M6x12'.*specs_pre_b12.json"):
            self.mod._view_bias_runtime(cfg, self.pre_b12_specs_file())

    def test_pre_b12_specs_refuse_start(self):
        """B-12 전 규격 + 종류별 설정 → 시작 종료 2, 사유에 라벨 불일치."""
        c = self.case()
        ai = write_spec_config(c.ai, Path(self.tmp.name) / "ai_spec.json")

        with mock.patch.object(self.mod, "PART_SPECS_PATH", self.pre_b12_specs_file()):
            code, err = self.cli(c, c.write_meta(c.meta()), ai=ai)

        self.assertEqual(code, 2, err)
        self.assertIn("label 'M3x20' != B 'M6x12'", err)

    def test_bad_range_in_specs_refuses_start(self):
        """규격 범위가 총길이를 품지 않으면(BOLT_B min 50 mm > 46 mm) 시작 종료 2, 사유에 종류 이름."""
        c = self.case()
        ai = write_spec_config(c.ai, Path(self.tmp.name) / "ai_spec.json")
        d = post_b12()
        d["specs"]["BOLT_B"]["length_min"] = 0.050

        with mock.patch.object(self.mod, "PART_SPECS_PATH", write_specs(d, Path(self.tmp.name) / "bad.json")):
            code, err = self.cli(c, c.write_meta(c.meta()), ai=ai)

        self.assertEqual(code, 2, err)
        self.assertIn("BOLT_B: needs length_min <= total_length <= length_max", err)

    def test_repo_specs_load_and_pass_start_check(self):
        """저장소 B 규격(B가 바꿀 수 있는 파일)은 시야 보정 대조·W 추적 높이 해석을 통과한다(값은 보지 않음)."""
        self.mod._bolt_planes(load_specs(SPECS))
        self.mod._check_view_bias_plane(ViewBiasConfig("C1", 1.39, 0.105, "m@x"))

    def test_b12_specs_start_and_record_spec(self):
        """B-12 뒤 규격 → 시작 줄에 종류별 값(길이는 B 범위), B 입력 기록에 종류 키, 요약에 규격 지문."""
        c = self.case()
        ai = write_spec_config(c.ai, Path(self.tmp.name) / "ai_spec.json")

        with mock.patch.object(self.mod, "PART_SPECS_PATH", self.b12_specs_file()):
            code, err = self.cli(c, c.write_meta(c.meta()), ai=ai)

        self.assertIn(code, (0, 1), err)
        self.assertIn("axis_view_bias by spec: BOLT_A(M3x20)=plane 102.0mm k 0.620mm len 17.0~27.0mm(range) "
                      "BOLT_B(M6x40)=plane 105.0mm k 1.390mm len 39.0~52.0mm(range)", err)
        rows = [r for r in c.b_outputs() if r["kind"] == "b_output"]
        self.assertEqual(len(rows), 1)
        self.assertIn("spec", rows[0]["input"]["view_bias"])
        s = c.summary()["view_bias"]
        self.assertEqual([r["name"] for r in s["specs"]], ["BOLT_A", "BOLT_B"])
        # 요약의 종류별 범위 = B 규격 값
        self.assertEqual([(r["length_min_m"], r["length_max_m"]) for r in s["specs"]], [(0.017, 0.027), (0.039, 0.052)])
        self.assertEqual(len(s["part_specs_sha256"]), 64)

    def test_bolt_planes_carry_tolerances(self):
        """W 추적 높이(`_bolt_planes`)에 B 길이 허용·범위가 함께 들어간다(시야 보정·B `_classify`와 같은 고르기).

        종류를 모르는 기본 높이는 B `class_plane_z(specs, "bolt")` = 최상위 `plane_z`(B-12 뒤 0.105).
        """
        planes = self.mod._bolt_planes(load_specs(self.b12_specs_file()))
        self.assertEqual(planes.specs, (("BOLT_A", 0.102, 0.023), ("BOLT_B", 0.105, 0.046)))
        self.assertEqual(planes.tolerances_m, (0.006, 0.006))
        self.assertEqual(planes.ranges_m, ((0.017, 0.027), (0.039, 0.052)))
        self.assertEqual(planes.default_z_m, 0.105)

        pre = self.mod._bolt_planes(load_specs(self.pre_b12_specs_file()))
        self.assertEqual((pre.ranges_m, pre.tolerances_m, pre.uniform), ((None, None), (0.004, 0.006), True))

        d = post_b12()
        d["specs"]["BOLT_A"]["length_max"] = 0.020
        with self.assertRaisesRegex(ValueError, "BOLT_A"):
            self.mod._bolt_planes(d)
