"""`run_pipeline.py` 팔 영역 검출 제외 연결 시험(실시간 기록 연결, 가짜 수신·모델·TF).

확인: 인자 조합, 기본 켬 규칙(실제 TF일 때만), 끝까지의 제외(볼트를 가리는 캡슐 → ARM_REGION → B 수락 없음),
TF 실패 때 유지, 끄기 표시, 추론 프로세스 대리 객체(토큰 인자)와 함께 동작.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np

from eyetoarm.transforms.planar_transform import pixels_to_w
from test_run_pipeline_capture_observation import FRAME_ID, H, T_W_C, W
from test_run_pipeline_infer_process import LeakCheck, options
from test_run_pipeline_live_observation import CONFIG, Base, FakeTf, Scenario

ROOT = Path(__file__).resolve().parents[2]
ARM_CONFIG = ROOT / "config" / "vision" / "arm_region_filter.json"
K = np.array([[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]])
SCENARIO_BOLT_PX = (142.0, 120.0)   # test_run_pipeline_capture_observation.bolt_mask() 중심
CHILD_BOLT_PX = (42.0, 60.0)        # ai_process_fake 볼트(pos 0) 중심


def links():
    """설정 파일의 캡슐 링크 이름."""
    return sorted({o["link"] for o in json.loads(ARM_CONFIG.read_text(encoding="utf-8"))["occluders"]})


class ArmTf(FakeTf):
    """카메라 TF + 모든 팔 링크를 「카메라와 그 픽셀의 볼트 평면 점 사이 가운데」에 둔 TF(캡슐이 그 픽셀을 가림)."""

    def __init__(self, pixel, **kw):
        """가릴 픽셀."""
        super().__init__(**kw)
        w = pixels_to_w(np.array([pixel], float), K, np.zeros(5), T_W_C, 0.105)[0]
        mid = (T_W_C[:3, 3] + np.array([w[0], w[1], 0.105])) / 2.0
        self.link_T = np.eye(4)
        self.link_T[:3, 3] = mid

    def lookup(self, parent, child, stamp_ns, timeout_s):
        """팔 링크는 가운데 자세, 카메라는 기존 대역."""
        if child in links():
            self.calls.append((parent, child, stamp_ns))
            return self.link_T.copy(), stamp_ns

        return super().lookup(parent, child, stamp_ns, timeout_s)


class ArmBase(LeakCheck, Base):
    """실시간 기록 연결 CLI + 결과 도우미."""

    def run_arm(self, scenario, *, tf, extra=(), **kw):
        """실행 → (종료 코드, stderr, 출력 폴더)."""
        return self.run_cli(scenario, tf=tf, extra=extra, **kw)

    def run_meta(self, out):
        """run_meta 줄."""
        log = Path(self.tmp.name) / f"{Path(out).name}.jsonl"

        return next(json.loads(t) for t in log.read_text(encoding="utf-8").splitlines()
                    if json.loads(t)["kind"] == "run_meta")

    def processes(self, out):
        """process 줄."""
        log = Path(self.tmp.name) / f"{Path(out).name}.jsonl"

        rows = [json.loads(t) for t in log.read_text(encoding="utf-8").splitlines()]

        return [r for r in rows if r["kind"] == "process"]

    def accepts(self, out):
        """B 수락 수."""
        return sum(1 for r in self.records(out) if r["kind"] == "accept")

    def arm_lines(self, out):
        """arm_region.jsonl 줄."""
        p = Path(out) / "arm_region.jsonl"

        return [] if not p.exists() else [json.loads(t) for t in p.read_text(encoding="utf-8").splitlines()]


class ArgsTest(ArmBase):
    """인자 조합 오류는 종료 2."""

    def test_flag_combinations(self):
        """켬·끔 동시, 실시간 기록 연결 없이 지정은 오류 2."""
        base = ["--detector", "ai", "--ai-config", self.case.ai, "--log", str(Path(self.tmp.name) / "a.jsonl")]
        cases = [
            (["--ros-link", *base, "--live-observation-out", self.out("x"), "--live-observation-config", str(CONFIG),
              "--arm-region-filter", str(ARM_CONFIG), "--no-arm-region-filter"], "mutually exclusive"),
            (["--ros-link", *base, "--arm-region-filter", str(ARM_CONFIG)], "require --live-observation-out"),
            (["--ros-link", *base, "--no-arm-region-filter"], "require --live-observation-out"),
            (["--images", *self.case.images, *base, "--arm-region-filter", str(ARM_CONFIG)],
             "require --live-observation-out"),
        ]

        for argv, msg in cases:
            with self.subTest(msg=msg):
                buf = io.StringIO()

                with contextlib.redirect_stderr(buf):
                    try:
                        code = self.mod.main(argv, stderr=io.StringIO(), ros_available=lambda: True)
                    except SystemExit as exc:
                        code = exc.code

                self.assertEqual(code, 2)
                self.assertIn(msg, buf.getvalue())


class DefaultTest(ArmBase):
    """기본 켬 규칙과 설정 읽기."""

    def ns(self, **kw):
        """`_load_arm_region` 인자 대역."""
        return argparse.Namespace(**{"arm_region_filter": None, "no_arm_region_filter": False, **kw})

    def test_default_on_with_real_tf_only(self):
        """실제 TF면 기본 켬(source=default·기본 파일·sha256), 주입 TF면 기본 끔(표시 없음), 인자는 그대로."""
        cfg, mark = self.mod._load_arm_region(self.ns(), "world", injected_tf=False)
        self.assertIsNotNone(cfg)
        self.assertEqual((mark["enabled"], mark["source"], Path(mark["path"]).name, len(mark["sha256"])),
                         (True, "default", "arm_region_filter.json", 64))

        self.assertEqual(self.mod._load_arm_region(self.ns(), "world", injected_tf=True), (None, None))
        cfg, mark = self.mod._load_arm_region(self.ns(arm_region_filter=str(ARM_CONFIG)), "world", injected_tf=True)
        self.assertEqual((cfg is not None, mark["source"]), (True, "argument"))
        self.assertEqual(self.mod._load_arm_region(self.ns(no_arm_region_filter=True), "world", injected_tf=False),
                         (None, {"enabled": False, "source": "argument"}))

    def test_world_frame_mismatch_is_error(self):
        """설정 world_frame이 실시간 기록 연결과 다르면 ValueError(CLI 종료 2)."""
        with self.assertRaises(ValueError):
            self.mod._load_arm_region(self.ns(), "map", injected_tf=False)

    def test_injected_tf_default_adds_nothing(self):
        """주입 TF 기본(끔): run_meta·요약·stderr·기록 파일에 팔 영역 표시가 없다(기존 출력 그대로)."""
        code, err, out = self.run_arm(Scenario(2), tf=FakeTf())

        self.assertEqual(code, 0, err)
        self.assertNotIn("arm_region_filter", self.run_meta(out))
        self.assertNotIn("arm_region_filter", self.summary(out))
        self.assertNotIn("arm region", err.lower())
        self.assertFalse((Path(out) / "arm_region.jsonl").exists())


class EndToEndTest(ArmBase):
    """실시간 기록 연결 끝까지: 제외·유지·끄기."""

    def test_capsule_over_bolt_excludes_and_blocks_b(self):
        """캡슐이 볼트를 가리면 ARM_REGION으로 거부되고 관측이 무효라 B 수락이 없다. 끄면 같은 장면이 수락된다."""
        code_on, err_on, on = self.run_arm(Scenario(3), tf=ArmTf(SCENARIO_BOLT_PX),
                                           extra=("--arm-region-filter", str(ARM_CONFIG)))
        code_off, err_off, off = self.run_arm(Scenario(3), tf=ArmTf(SCENARIO_BOLT_PX), out=self.out("off"),
                                              extra=("--no-arm-region-filter",))

        self.assertEqual((code_on, code_off), (0, 0), err_on + err_off)
        self.assertEqual(self.accepts(on), 0)
        self.assertGreater(self.accepts(off), 0)   # 첫 프레임은 세션 열기라 수락되지 않는다(기존 동작)
        rejected = [i["rejected"] for p in self.processes(on) for i in (p.get("ai_instances") or [])]
        self.assertEqual(rejected, ["ARM_REGION"] * 3)

        s = self.summary(on)["arm_region_filter"]
        self.assertEqual((s["enabled"], s["source"], s["counts"]["excluded"], s["counts"]["reselected"]),
                         (True, "argument", 3, 3))
        self.assertIsNotNone(s["ms_p95"])
        meta = self.run_meta(on)["arm_region_filter"]
        self.assertEqual((meta["enabled"], meta["source"], meta["sha256"]), (True, "argument", s["sha256"]))
        lines = self.arm_lines(on)
        self.assertEqual(len(lines), 3)
        self.assertTrue(all(d["excluded"] and d["center_w_mm"] is not None for r in lines for d in r["decisions"]))
        self.assertIn("ARM REGION FILTER ON source=argument", err_on)

        self.assertEqual(self.run_meta(off)["arm_region_filter"], {"enabled": False, "source": "argument"})
        self.assertEqual(self.summary(off)["arm_region_filter"], {"enabled": False, "source": "argument"})
        self.assertIn("arm region filter off (--no-arm-region-filter)", err_off)

    def test_tf_failure_keeps_detection(self):
        """팔 링크 TF가 없으면 제외하지 않고(B 수락이 끔과 같음) TF 실패 수·사유를 남긴다."""
        code, err, out = self.run_arm(Scenario(3), tf=FakeTf(), extra=("--arm-region-filter", str(ARM_CONFIG)))
        code_off, err_off, off = self.run_arm(Scenario(3), tf=FakeTf(), out=self.out("off"),
                                              extra=("--no-arm-region-filter",))

        self.assertEqual((code, code_off), (0, 0), err + err_off)
        self.assertEqual(self.accepts(out), self.accepts(off))
        self.assertGreater(self.accepts(out), 0)
        s = self.summary(out)["arm_region_filter"]
        self.assertEqual((s["counts"]["excluded"], s["counts"]["tf_unavailable"]), (0, 3))
        self.assertTrue(all(r["status"].startswith("tf_unavailable:") for r in self.arm_lines(out)))


class InferProcessTest(ArmBase):
    """추론 프로세스 대리 객체(토큰 인자)를 감싸도 제외가 동작한다."""

    def test_filter_wraps_infer_proxy(self):
        """--infer-process + --arm-region-filter: 자식 결과의 볼트를 가리면 ARM_REGION, 자식은 정상 종료."""
        out = self.out()
        log = str(Path(self.tmp.name) / "live.jsonl")
        sc = Scenario(1)
        sc.link = lambda: self.controller.current
        argv = ["--ros-link", "--detector", "ai", "--ai-config", self.case.ai, "--log", log,
                "--live-observation-out", out, "--live-observation-config", str(CONFIG), "--max-frames", "1",
                "--infer-process", "--arm-region-filter", str(ARM_CONFIG)]
        err = io.StringIO()
        code = self.mod.main(argv, stderr=err, ros_available=lambda: True, live_source_factory=sc.factory_maker([]),
                             live_tf_factory=lambda c: ArmTf(CHILD_BOLT_PX), live_run_controller=self.controller,
                             infer_process_options=options())

        self.assertEqual(code, 0, err.getvalue())
        s = self.summary(out)
        self.assertEqual(s["arm_region_filter"]["counts"]["excluded"], 1)
        self.assertEqual((s["infer_process"]["results"], s["infer_process"]["exit"]),
                         (1, {"returncode": 0, "forced": False}))


assert (W, H, FRAME_ID) == (320, 240, "link_cam_optical_frame")   # 위 픽셀 상수의 전제
