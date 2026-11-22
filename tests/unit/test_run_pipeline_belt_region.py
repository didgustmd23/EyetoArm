"""`run_pipeline.py` 벨트 밖 검출 제외 연결 시험(실시간 기록 연결, 가짜 수신·모델·TF).

확인: 인자 조합, 기본 켬 규칙(실제 TF일 때만, 팔 필터와 같음), 끝까지의 제외(벨트 밖 볼트 → OUTSIDE_BELT → B 수락 없음),
벨트 위 볼트 유지, 끄기 표시, 팔 필터와 함께 켬.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np

from test_run_pipeline_arm_region import ARM_CONFIG, SCENARIO_BOLT_PX, ArmBase, ArmTf
from test_run_pipeline_capture_observation import T_W_C
from test_run_pipeline_live_observation import CONFIG, FakeTf, Scenario

ROOT = Path(__file__).resolve().parents[2]
BELT_CONFIG = ROOT / "config" / "vision" / "belt_region_filter.json"


class ShiftedTf(FakeTf):
    """카메라를 W −X로 0.2 m 옮긴 TF: 시나리오 볼트(원래 W x 0.50, 벨트 위)가 x 0.30(벨트 밖 로봇 쪽)이 된다."""

    def lookup(self, parent, child, stamp_ns, timeout_s):
        """기존 대역 + 평행 이동."""
        T, stamp = super().lookup(parent, child, stamp_ns, timeout_s)
        T = np.array(T, dtype=float)
        T[0, 3] -= 0.2

        return T, stamp


class BeltBase(ArmBase):
    """벨트 기록 도우미."""

    def belt_lines(self, out):
        """belt_region.jsonl 줄."""
        p = Path(out) / "belt_region.jsonl"
        return [] if not p.exists() else [json.loads(t) for t in p.read_text(encoding="utf-8").splitlines()]

    def rejected(self, out):
        """process 기록의 인스턴스 거부 사유 목록."""
        return [i["rejected"] for p in self.processes(out) for i in (p.get("ai_instances") or [])]


class ArgsTest(BeltBase):
    """인자 조합 오류는 종료 2."""

    def test_flag_combinations(self):
        """켬·끔 동시, 실시간 기록 연결 없이 지정은 오류 2."""
        base = ["--detector", "ai", "--ai-config", self.case.ai, "--log", str(Path(self.tmp.name) / "a.jsonl")]
        cases = [
            (["--ros-link", *base, "--live-observation-out", self.out("x"), "--live-observation-config", str(CONFIG),
              "--belt-region-filter", str(BELT_CONFIG), "--no-belt-region-filter"], "mutually exclusive"),
            (["--ros-link", *base, "--belt-region-filter", str(BELT_CONFIG)], "require --live-observation-out"),
            (["--ros-link", *base, "--no-belt-region-filter"], "require --live-observation-out"),
            (["--images", *self.case.images, *base, "--belt-region-filter", str(BELT_CONFIG)],
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

    def test_bad_config_is_exit_2(self):
        """읽을 수 없는 설정 파일은 시작 오류 2(사유 출력)."""
        bad = Path(self.tmp.name) / "bad.json"
        bad.write_text("{", encoding="utf-8")
        code, err, _ = self.run_arm(Scenario(1), tf=FakeTf(), extra=("--belt-region-filter", str(bad)))
        self.assertEqual(code, 2)
        self.assertIn("belt region filter config error", err)


class DefaultTest(BeltBase):
    """기본 켬 규칙과 설정 읽기."""

    def ns(self, **kw):
        """`_load_belt_region` 인자 대역."""
        return argparse.Namespace(**{"belt_region_filter": None, "no_belt_region_filter": False, **kw})

    def test_default_on_with_real_tf_only(self):
        """실제 TF면 기본 켬(source=default·기본 파일·sha256·frames sha256), 주입 TF면 기본 끔, 인자는 그대로."""
        cfg, mark = self.mod._load_belt_region(self.ns(), "world", injected_tf=False)
        self.assertIsNotNone(cfg)
        self.assertEqual((mark["enabled"], mark["source"], Path(mark["path"]).name, len(mark["sha256"]),
                          Path(mark["frames_path"]).name, len(mark["frames_sha256"])),
                         (True, "default", "belt_region_filter.json", 64, "frames.yaml", 64))

        self.assertEqual(self.mod._load_belt_region(self.ns(), "world", injected_tf=True), (None, None))
        cfg, mark = self.mod._load_belt_region(self.ns(belt_region_filter=str(BELT_CONFIG)), "world",
                                               injected_tf=True)
        self.assertEqual((cfg is not None, mark["source"]), (True, "argument"))
        self.assertEqual(self.mod._load_belt_region(self.ns(no_belt_region_filter=True), "world", injected_tf=False),
                         (None, {"enabled": False, "source": "argument"}))

    def test_world_frame_mismatch_is_error(self):
        """설정 world_frame이 실시간 기록 연결과 다르면 ValueError(CLI 종료 2)."""
        with self.assertRaises(ValueError):
            self.mod._load_belt_region(self.ns(), "map", injected_tf=False)

    def test_injected_tf_default_adds_nothing(self):
        """주입 TF 기본(끔): run_meta·요약·stderr·기록 파일에 벨트 표시가 없다(기존 출력 그대로)."""
        code, err, out = self.run_arm(Scenario(2), tf=FakeTf())

        self.assertEqual(code, 0, err)
        self.assertNotIn("belt_region_filter", self.run_meta(out))
        self.assertNotIn("belt_region_filter", self.summary(out))
        self.assertNotIn("belt region", err.lower())
        self.assertFalse((Path(out) / "belt_region.jsonl").exists())


class EndToEndTest(BeltBase):
    """실시간 기록 연결 끝까지: 제외·유지·끄기·팔 필터와 함께."""

    def test_outside_belt_excluded_and_blocks_b(self):
        """벨트 밖 볼트는 OUTSIDE_BELT로 거부되고 B 수락이 없다. 끄면 같은 장면이 수락된다."""
        code_on, err_on, on = self.run_arm(Scenario(3), tf=ShiftedTf(),
                                           extra=("--belt-region-filter", str(BELT_CONFIG)))
        code_off, err_off, off = self.run_arm(Scenario(3), tf=ShiftedTf(), out=self.out("off"),
                                              extra=("--no-belt-region-filter",))

        self.assertEqual((code_on, code_off), (0, 0), err_on + err_off)
        self.assertEqual(self.accepts(on), 0)
        self.assertGreater(self.accepts(off), 0)   # 첫 프레임은 세션 열기라 수락되지 않는다(기존 동작)
        self.assertEqual(self.rejected(on), ["OUTSIDE_BELT"] * 3)

        s = self.summary(on)["belt_region_filter"]
        self.assertEqual((s["enabled"], s["source"], s["counts"]["excluded"], s["counts"]["reselected"]),
                         (True, "argument", 3, 3))
        self.assertEqual(s["records_written"], 3)
        meta = self.run_meta(on)["belt_region_filter"]
        self.assertEqual((meta["enabled"], meta["source"], meta["sha256"]), (True, "argument", s["sha256"]))
        lines = self.belt_lines(on)
        self.assertEqual(len(lines), 3)
        self.assertTrue(all(d["excluded"] and d["inside_fraction"] == 0.0 and d["center_w_top_mm"][0] < 450.0
                            for r in lines for d in r["decisions"]))
        self.assertIn("BELT REGION FILTER ON source=argument", err_on)
        self.assertIn("rect=x[0.450,0.650] y[-0.630,0.630]", err_on)

        self.assertEqual(self.run_meta(off)["belt_region_filter"], {"enabled": False, "source": "argument"})
        self.assertEqual(self.summary(off)["belt_region_filter"], {"enabled": False, "source": "argument"})
        self.assertIn("belt region filter off (--no-belt-region-filter)", err_off)

    def test_on_belt_bolt_kept(self):
        """벨트 위 볼트(시나리오 기본 자세, W x 0.50)는 남아 B 수락이 끔과 같고 기록 파일은 비어 있다."""
        code, err, out = self.run_arm(Scenario(3), tf=FakeTf(), extra=("--belt-region-filter", str(BELT_CONFIG)))
        code_off, err_off, off = self.run_arm(Scenario(3), tf=FakeTf(), out=self.out("off"),
                                              extra=("--no-belt-region-filter",))

        self.assertEqual((code, code_off), (0, 0), err + err_off)
        self.assertEqual(self.accepts(out), self.accepts(off))
        self.assertGreater(self.accepts(out), 0)
        s = self.summary(out)["belt_region_filter"]
        self.assertEqual((s["counts"]["excluded"], s["counts"]["judged"], s["records_written"]), (0, 3, 0))
        self.assertEqual(self.belt_lines(out), [])

    def test_with_arm_filter(self):
        """팔 필터와 함께 켜면 팔이 먼저 ARM_REGION으로 거부하고 벨트는 그 인스턴스를 판정하지 않는다."""
        code, err, out = self.run_arm(Scenario(3), tf=ArmTf(SCENARIO_BOLT_PX),
                                      extra=("--arm-region-filter", str(ARM_CONFIG),
                                             "--belt-region-filter", str(BELT_CONFIG)))

        self.assertEqual(code, 0, err)
        self.assertEqual(self.rejected(out), ["ARM_REGION"] * 3)
        s = self.summary(out)
        self.assertEqual((s["arm_region_filter"]["counts"]["excluded"], s["belt_region_filter"]["counts"]["judged"]),
                         (3, 0))
        self.assertIn("ARM REGION FILTER ON", err)
        self.assertIn("BELT REGION FILTER ON", err)


assert T_W_C[0, 3] == 0.45   # ShiftedTf 주석(0.50 → 0.30)의 전제: 시나리오 카메라 x 0.45
