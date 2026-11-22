"""run_pipeline 종류별 시야 보정 시작 대조·전달(#270). `_workspace/270_k/run_pipeline.patch`와 함께 적용한다."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest import mock

from eyetoarm.transforms.work_result import load_specs
from eyetoarm.vision.view_bias import ViewBiasPlan
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
        """저장소 B 규격 사본 + B-12 예상 BOLT_A(시험 임시 파일)."""
        d = copy.deepcopy(load_specs(SPECS))
        d["specs"]["BOLT_A"].update(label="M3x20", plane_z=0.102, total_length=0.023, length_tolerance=0.006)
        p = Path(self.tmp.name) / "specs_b12.json"
        p.write_text(json.dumps(d), encoding="utf-8")

        return p

    def test_runtime_legacy_passthrough_and_plan(self):
        """기존 형식은 같은 객체, 종류별 형식은 B 규격 지문을 가진 계획, 지금 규격이면 ValueError(경로 포함)."""
        from eyetoarm.vision.ai_config import ViewBiasConfig, load_ai_config

        legacy = ViewBiasConfig("C1", 1.39, 0.105, "m@x")
        self.assertIs(self.mod._view_bias_runtime(legacy), legacy)
        self.assertIsNone(self.mod._view_bias_runtime(None))

        c = self.case()
        cfg = load_ai_config(write_spec_config(c.ai, Path(self.tmp.name) / "ai_spec.json")).axis_view_bias
        plan = self.mod._view_bias_runtime(cfg, self.b12_specs_file())
        self.assertIsInstance(plan, ViewBiasPlan)
        self.assertEqual(len(plan.part_specs_sha256), 64)

        with self.assertRaisesRegex(ValueError, "label 'M3x20' != B 'M6x12'.*part_specs.json"):
            self.mod._view_bias_runtime(cfg)

    def test_current_specs_refuse_start(self):
        """지금 저장소 규격 + 종류별 설정 → 시작 종료 2, 사유에 라벨 불일치."""
        c = self.case()
        ai = write_spec_config(c.ai, Path(self.tmp.name) / "ai_spec.json")
        code, err = self.cli(c, c.write_meta(c.meta()), ai=ai)
        self.assertEqual(code, 2, err)
        self.assertIn("label 'M3x20' != B 'M6x12'", err)

    def test_b12_specs_start_and_record_spec(self):
        """B-12 예상 규격 → 시작 줄에 종류별 값, B 입력 기록에 종류 키, 요약에 규격 지문."""
        c = self.case()
        ai = write_spec_config(c.ai, Path(self.tmp.name) / "ai_spec.json")

        with mock.patch.object(self.mod, "PART_SPECS_PATH", self.b12_specs_file()):
            code, err = self.cli(c, c.write_meta(c.meta()), ai=ai)

        self.assertIn(code, (0, 1), err)
        self.assertIn("axis_view_bias by spec: BOLT_A(M3x20)=plane 102.0mm k 0.620mm len 17.0~29.0mm "
                      "BOLT_B(M6x40)=plane 105.0mm k 1.390mm len 40.0~52.0mm", err)
        rows = [r for r in c.b_outputs() if r["kind"] == "b_output"]
        self.assertEqual(len(rows), 1)
        self.assertIn("spec", rows[0]["input"]["view_bias"])
        s = c.summary()["view_bias"]
        self.assertEqual([r["name"] for r in s["specs"]], ["BOLT_A", "BOLT_B"])
        self.assertEqual(len(s["part_specs_sha256"]), 64)

    def test_bolt_planes_carry_tolerances(self):
        """W 추적 높이(`_bolt_planes`)에 B 길이 허용이 함께 들어간다(시야 보정과 같은 고르기)."""
        planes = self.mod._bolt_planes(load_specs(self.b12_specs_file()))
        self.assertEqual(planes.specs, (("BOLT_A", 0.102, 0.023), ("BOLT_B", 0.105, 0.046)))
        self.assertEqual(planes.tolerances_m, (0.006, 0.006))
        self.assertEqual(planes.default_z_m, 0.102)
