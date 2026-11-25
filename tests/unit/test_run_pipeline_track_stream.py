"""추적 흐름 발행 CLI 시험: 꺼짐이면 발행기 없음, 켬이면 시작 줄·발행·정리·요약, 옵션·설정·인터페이스 오류."""
from __future__ import annotations

import io
import json
import unittest
from pathlib import Path

from test_live_tracking_link import PIXEL, ReleasingRuntime, sort_config
from test_live_world_tracking import WORLD, WorldCli
from test_run_pipeline_capture_observation import model_with
from test_run_pipeline_gripper_handoff import GRIP, FakeTransport
from test_run_pipeline_live_observation import FakeTf, Scenario

ROOT = Path(__file__).resolve().parents[2]
STREAM_CONFIG = ROOT / "config" / "vision" / "track_stream_test.json"
STREAM = ("--track-stream", str(STREAM_CONFIG))


class FakeSender:
    """발행기 대역: send·now_ns·close."""

    def __init__(self, config, now_ns=None):
        """설정. 기본(link, #430)이면 CLI가 시각 함수 `now_ns`도 넘긴다(이 대역은 쓰지 않고 1을 낸다)."""
        self.config = config
        self.linked_now = now_ns
        self.got: list[dict] = []
        self.closed = False

    def send(self, msg):
        """발행 한 번."""
        self.got.append(msg)

    def now_ns(self):
        """SIM 시각 대역."""
        return 1

    def close(self):
        """정리."""
        self.closed = True


class StreamCli(WorldCli):
    """CLI 실행(발행기·그리퍼 수신 주입)."""

    def run_stream(self, scenario, extra, sender_factory):
        """main 실행 → (종료 코드, stderr, 출력 폴더)."""
        out = self.out()
        log = str(Path(self.tmp.name) / f"{Path(out).name}.jsonl")
        scenario.link = lambda: self.controller.current
        argv = ["--ros-link", "--detector", "ai", "--ai-config", self.case.ai, "--log", log, "--live-observation-out", out,
                "--live-observation-config", str(sort_config(self.tmp.name)), "--max-frames", str(scenario.n), *extra,
                "--live-sort-part"]
        err = io.StringIO()

        try:
            code = self.mod.main(argv, stderr=err, ros_available=lambda: True,
                                 model_factory=lambda path: model_with(scenario.masks_fn),
                                 live_source_factory=scenario.factory_maker([]), live_tf_factory=lambda c: FakeTf(),
                                 live_run_controller=self.controller, live_sort_part_factory=self.factory(ReleasingRuntime()),
                                 live_conveyor_probe=lambda: {"node": "/conveyor_controller", "speed_mps": None,
                                                              "status": "no_node"},
                                 live_gripper_factory=lambda c, s: FakeTransport(c, s),
                                 live_track_stream_factory=sender_factory)
        except SystemExit as exc:
            code = exc.code

        return code, err.getvalue(), out


class CliTest(StreamCli):
    """옵션 켬·끔·오류."""

    def test_off_never_creates_sender(self):
        """--track-stream이 없으면 발행기를 만들지 않고 요약·기록에 stream이 없다."""
        code, err, out = self.run_stream(Scenario(5), WORLD, lambda c, now_ns=None: self.fail("sender must not start"))
        self.assertEqual(code, 0, err)
        # 같은 실행: retry 블록 없는 송신 설정이면 기본 제한을 시작 줄·run_meta에 남긴다
        self.assertIn("retry=off tracking=world", err)
        self.assertIn("retry_default=failure_gap_1s", err)
        meta = json.loads((Path(self.tmp.name) / f"{Path(out).name}.jsonl").read_text().splitlines()[0])
        self.assertEqual(meta["sort_part_retry"], {"source": "default", "failure_gap_s": 1.0})
        self.assertNotIn("track_stream", self.summary(out)["world_tracking"])
        self.assertNotIn("TRACK STREAM", err)
        self.assertFalse(any("stream" in r for r in self.world_records(out)))

    def test_on_publishes_and_closes(self):
        """켬(+ 인계)이면 시작 줄, frame 발행·끝 marker, 발행기 정리, 요약 track_stream, 기록 frame 줄에 stream."""
        made = []

        def factory(config, now_ns=None):
            """대역 발행기."""
            made.append(FakeSender(config, now_ns))
            return made[-1]

        code, err, out = self.run_stream(Scenario(5), GRIP + STREAM, factory)
        self.assertEqual(code, 0, err)
        rows = [json.loads(t) for t in (Path(out) / "track_stream.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), len(made[0].got))
        self.assertTrue(all(r["published"] and r["publish_stamp_ns"] == 1 for r in rows))
        self.assertEqual(rows[-1]["kind"], "marker")
        self.assertIn("TRACK STREAM ON topic=/a/track_states", err)
        self.assertIn("handoff_on=true", err)
        self.assertNotIn("WARNING track stream", err)
        sender = made[0]
        self.assertTrue(sender.closed)
        kinds = [m["kind"] for m in sender.got]
        self.assertIn("frame", kinds)
        self.assertEqual(kinds[-1], "marker")
        self.assertTrue(all(m["handoff_on"] for m in sender.got))
        block = self.summary(out)["world_tracking"]["track_stream"]
        self.assertEqual(block["counts"]["publish_errors"], 0)
        self.assertEqual(block["counts"]["published"], len(sender.got))
        self.assertTrue(all("stream" in r for r in self.world_records(out) if r["kind"] == "frame"))

    def test_without_handoff_warns(self):
        """인계 없이 켜면 handoff_on=false를 싣고 경고 줄을 낸다."""
        made = []
        code, err, _ = self.run_stream(Scenario(3), WORLD + STREAM, lambda c, now_ns=None: made.append(FakeSender(c)) or made[-1])
        self.assertEqual(code, 0, err)
        self.assertIn("handoff_on=false", err)
        self.assertIn("WARNING track stream without --gripper-handoff", err)
        self.assertFalse(any(m["handoff_on"] for m in made[0].got))

    def test_errors(self):
        """pixel 추적이면 2, 설정 오류면 2, 인터페이스가 없으면 1(발행기 없음)."""
        bad = Path(self.tmp.name) / "bad_stream.json"
        bad.write_text(json.dumps({**json.loads(STREAM_CONFIG.read_text()), "extra": 1}))

        def missing(config, now_ns=None):
            """인터페이스 미빌드 대역."""
            raise ImportError("No module named 'eyetoarm_interfaces'")

        cases = ((("--tracking", *PIXEL, *STREAM), lambda c, now_ns=None: self.fail("no sender"), 2),
                 ((*WORLD, "--track-stream", str(bad)), lambda c, now_ns=None: self.fail("no sender"), 2),
                 ((*WORLD, *STREAM), missing, 1))

        for extra, factory, want in cases:
            with self.subTest(extra=extra):
                code, err, _ = self.run_stream(Scenario(3), extra, factory)
                self.assertEqual(code, want, err)

    def test_publish_error_is_problem(self):
        """발행 예외는 추적을 멈추지 않지만 종료 1과 사유로 드러난다."""
        class Failing(FakeSender):
            """늘 실패하는 발행기."""

            def send(self, msg):
                """실패."""
                raise RuntimeError("boom")

        code, err, out = self.run_stream(Scenario(3), WORLD + STREAM, Failing)
        self.assertEqual(code, 1, err)
        self.assertIn("track stream publish errors", err + json.dumps(self.summary(out)))


if __name__ == "__main__":
    unittest.main()
