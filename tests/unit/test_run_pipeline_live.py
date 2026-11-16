"""CLI `--ros-link` 시험(L4) — 인자 검증·rclpy 확인·소스 ID 기본값·신호 → 정지 요청.

근거: 계획 `docs/superpowers/plans/2026-11-16-live-link-camera.md` r3 §3.8(CLI 신호)·§4.1 L4, 결정 D4(소스 ID `gz_link`, A 임시)·
D6(run_pipeline 확장, A 임시). ROS를 쓰지 않는다: rclpy 확인과 소스 팩토리를 주입한다.
"""
import importlib.util
import io
import json
import os
import signal
import sys
import tempfile
import unittest

from eyetoarm.inputs.ros_image import LiveLinkSource, LiveSourceConfig
from test_ros_image_adapter import FakeTransport, image_msg, view_convert

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_spec = importlib.util.spec_from_file_location("run_pipeline_cli", os.path.join(ROOT, "scripts", "run_pipeline.py"))
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)

PLANAR = os.path.join(ROOT, "config", "vision", "opencv_planar.json")


def fake_factory(made):
    """영상 하나를 넣어 두는 가짜 연결 실시간 소스 팩토리. 만든 (source_id, reliability, 소스)를 적는다."""
    def make(source_id, reliability):
        """CLI가 부르는 팩토리 생성기."""
        def factory(issuer, generation_provider):
            """소스를 만든다."""
            src = LiveLinkSource(source_id=source_id, issuer=issuer, generation_provider=generation_provider,
                                 config=LiveSourceConfig(frame_wait_s=0.005), transport_factory=lambda s: FakeTransport(),
                                 convert=view_convert)
            src.handle_image(image_msg(1, width=64, height=48))
            made.append((source_id, reliability, src))
            return src

        return factory

    return make


class CliArgsTest(unittest.TestCase):
    """인자 검증(종료 2)."""

    def run_cli(self, argv, **kw):
        """main을 부르고 (종료 코드, stderr)."""
        err = io.StringIO()

        try:
            code = cli.main(argv, stderr=err, **kw)
        except SystemExit as exc:
            code = exc.code

        return code, err.getvalue()

    def test_ros_link_requires_rclpy(self):
        """rclpy를 찾지 못하면 인자 오류 2."""
        with tempfile.TemporaryDirectory() as d:
            code, _ = self.run_cli(["--ros-link", "--detect-config", PLANAR, "--log", os.path.join(d, "x.jsonl")],
                                   ros_available=lambda: False)

        self.assertEqual(code, 2)

    def test_reliability_only_with_ros_link(self):
        """--ros-reliability는 --ros-link 전용(2)."""
        with tempfile.TemporaryDirectory() as d:
            img = os.path.join(d, "a.png")
            code, _ = self.run_cli(["--images", img, "--ros-reliability", "reliable", "--detect-config", PLANAR,
                                    "--log", os.path.join(d, "x.jsonl")])

        self.assertEqual(code, 2)

    def test_ros_link_exclusive_with_images(self):
        """--ros-link와 --images는 함께 쓸 수 없다(2)."""
        with tempfile.TemporaryDirectory() as d:
            code, _ = self.run_cli(["--ros-link", "--images", "a.png", "--detect-config", PLANAR,
                                    "--log", os.path.join(d, "x.jsonl")], ros_available=lambda: True)

        self.assertEqual(code, 2)

    def test_ros_link_runs_with_default_source_id(self):
        """주입한 실시간 소스로 1프레임 처리 후 0. 소스 ID 기본 gz_link, 신뢰성 기본 None(=reliable), 요약에 live 항목."""
        made = []

        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "run.jsonl")
            code, err = self.run_cli(["--ros-link", "--detect-config", PLANAR, "--log", log, "--max-frames", "1"],
                                     ros_available=lambda: True, live_source_factory=fake_factory(made))

            with open(log, encoding="utf-8") as f:
                recs = [json.loads(t) for t in f]

        self.assertEqual(code, 0, err)
        self.assertEqual([(m[0], m[1]) for m in made], [("gz_link", None)])
        self.assertEqual(recs[0]["source_id"], "gz_link")
        self.assertIn("live_inputs=", err)
        self.assertEqual([r["status"] for r in recs if r["kind"] == "process"], ["COMPLETED"])
        self.assertEqual(recs[-1]["kind"], "source_stats")

    def test_stored_default_source_id_unchanged(self):
        """저장 입력의 소스 ID 기본값은 기존 stored0 그대로."""
        args = cli.build_parser().parse_args(["--images", "a.png", "--detect-config", PLANAR, "--log", "x"])
        cli._check_source_args(args, None)

        self.assertIsNone(args.source_id)
        self.assertFalse(args.ros_link)


class StopSignalsTest(unittest.TestCase):
    """신호 처리: 첫 신호는 정지 요청, 두 번째는 KeyboardInterrupt, 끝나면 원래 처리기."""

    def test_first_signal_requests_stop_second_interrupts(self):
        """SIGTERM 한 번 → stop 표지, 두 번째 → KeyboardInterrupt. 나온 뒤 처리기가 원래대로."""
        before = signal.getsignal(signal.SIGTERM)

        with cli._StopSignals() as sig:
            signal.raise_signal(signal.SIGTERM)
            self.assertTrue(sig.stop.is_set())

            with self.assertRaises(KeyboardInterrupt):
                signal.raise_signal(signal.SIGINT)

        self.assertIs(signal.getsignal(signal.SIGTERM), before)


if __name__ == "__main__":
    unittest.main()
