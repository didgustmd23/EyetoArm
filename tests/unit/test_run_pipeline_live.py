"""CLI `--ros-link` 시험: 인자 검증·rclpy 확인·소스 ID 기본값·신호 → 정지 요청.
ROS를 쓰지 않고 rclpy 확인과 소스 팩토리를 주입한다."""
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
    """영상 하나를 넣어 둔 가짜 실시간 소스 팩토리. 만든 (source_id, reliability, 소스)를 적는다."""
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
        """주입 소스로 1프레임 처리 후 0, 소스 ID 기본 gz_link, 신뢰성 기본 None, 요약에 live 항목."""
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
        """저장 입력의 소스 ID 기본값은 stored0이다."""
        args = cli.build_parser().parse_args(["--images", "a.png", "--detect-config", PLANAR, "--log", "x"])
        cli._check_source_args(args, None)

        self.assertIsNone(args.source_id)
        self.assertFalse(args.ros_link)


class StopSignalsTest(unittest.TestCase):
    """신호 처리: 첫 신호는 정지 요청, 같은 사건의 중복은 무시, 그 뒤는 KeyboardInterrupt, 정리 중에는 무시."""

    def setUp(self):
        """시험이 움직이는 시계."""
        self.now = [100.0]

    def signals(self):
        """시험 시계를 쓰는 처리기(중복 창 1 s)."""
        return cli._StopSignals(repeat_window_s=1.0, clock=lambda: self.now[0])

    def test_first_signal_requests_stop_later_signal_interrupts(self):
        """SIGTERM 한 번 → stop 표지, 창(1 s) 뒤 두 번째 → KeyboardInterrupt. 나온 뒤 처리기가 원래대로."""
        before = signal.getsignal(signal.SIGTERM)

        with self.signals() as sig:
            signal.raise_signal(signal.SIGTERM)
            self.assertTrue(sig.stop.is_set())
            self.now[0] += 1.5

            with self.assertRaises(KeyboardInterrupt):
                signal.raise_signal(signal.SIGINT)

        self.assertIs(signal.getsignal(signal.SIGTERM), before)

    def test_repeat_within_window_is_ignored(self):
        """그룹 Ctrl+C와 감독 스크립트 전달처럼 거의 동시에 온 두 번째 신호는 중복으로 세고 무시한다."""
        with self.signals() as sig:
            signal.raise_signal(signal.SIGINT)
            self.now[0] += 0.2
            signal.raise_signal(signal.SIGINT)

        self.assertTrue(sig.stop.is_set())
        self.assertEqual(sig.ignored, {"repeat": 1, "cleanup": 0})

    def test_signal_during_cleanup_does_not_interrupt(self):
        """`begin_cleanup` 뒤에는 창이 지난 신호도 정리를 끊지 않는다(세기만 함)."""
        with self.signals() as sig:
            signal.raise_signal(signal.SIGINT)
            sig.begin_cleanup()
            self.now[0] += 5.0
            signal.raise_signal(signal.SIGTERM)
            signal.raise_signal(signal.SIGHUP)

        self.assertEqual(sig.ignored, {"repeat": 0, "cleanup": 2})

    def test_sighup_requests_stop_and_is_restored(self):
        """SIGHUP(창 닫기)도 정지 요청이고, 나온 뒤 원래 처리기로 돌아간다."""
        before = signal.getsignal(signal.SIGHUP)

        with self.signals() as sig:
            signal.raise_signal(signal.SIGHUP)

        self.assertTrue(sig.stop.is_set())
        self.assertIs(signal.getsignal(signal.SIGHUP), before)


if __name__ == "__main__":
    unittest.main()
