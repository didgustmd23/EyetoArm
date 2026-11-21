"""그리퍼 넘기기 CLI·수신기 시험: 옵션 조합, 꺼짐이면 구독 0, 켜짐 요약, 깊이 짝짓기·해석."""
from __future__ import annotations

import io
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from eyetoarm.inputs.ros_gripper_depth import GripperDepthReceiver, belt_roi_xy_m, decode_depth
from test_gripper_depth import gcfg
from test_live_tracking_link import PIXEL, ReleasingRuntime, sort_config
from test_live_world_tracking import WORLD, WorldCli
from test_run_pipeline_capture_observation import model_with
from test_run_pipeline_live_observation import FakeTf, Scenario

ROOT = Path(__file__).resolve().parents[2]
GRIPPER_CONFIG = ROOT / "config" / "vision" / "world_tracker_gripper_test.json"
GRIP = ("--tracking", "--tracking-space", "world", "--world-tracker-config", str(GRIPPER_CONFIG), "--gripper-handoff")


class FakeTransport:
    """그리퍼 수신 대역: start·close·receiver.stats만."""

    def __init__(self, config, sink):
        """설정·받는 곳."""
        self.config, self.sink = config, sink
        self.started = self.closed = False
        self.receiver = SimpleNamespace(stats=lambda: {"emitted": 0})

    def start(self):
        """시작."""
        self.started = True

    def close(self, limit_s):
        """정리."""
        self.closed = True


class GripperCli(WorldCli):
    """CLI 실행(그리퍼 수신 주입)."""

    def run_grip(self, scenario, extra, gripper_factory):
        """main 실행 → (종료 코드, stderr, 출력 폴더)."""
        out = self.out()
        log = str(Path(self.tmp.name) / f"{Path(out).name}.jsonl")
        scenario.link = lambda: self.controller.current
        argv = ["--ros-link", "--detector", "ai", "--ai-config", self.case.ai, "--log", log, "--live-observation-out", out,
                "--live-observation-config", str(sort_config(self.tmp.name)), "--max-frames", str(scenario.n), *extra,
                "--live-sort-part"]
        err = io.StringIO()

        try:
            code = self.mod.main(argv, stderr=err, ros_available=lambda: True, model_factory=lambda path: model_with(scenario.masks_fn),
                                 live_source_factory=scenario.factory_maker([]), live_tf_factory=lambda c: FakeTf(),
                                 live_run_controller=self.controller, live_sort_part_factory=self.factory(ReleasingRuntime()),
                                 live_conveyor_probe=lambda: {"node": "/conveyor_controller", "speed_mps": None, "status": "no_node"},
                                 live_gripper_factory=gripper_factory)
        except SystemExit as exc:
            code = exc.code

        return code, err.getvalue(), out


class CliTest(GripperCli):
    """옵션 켬·끔."""

    def test_off_never_subscribes(self):
        """--gripper-handoff가 없으면 그리퍼 수신을 만들지 않고 요약에 handoff가 없다."""
        code, err, out = self.run_grip(Scenario(5), WORLD, lambda c, s: self.fail("gripper must not start"))
        self.assertEqual(code, 0, err)
        self.assertNotIn("handoff", self.summary(out)["world_tracking"])
        self.assertNotIn("GRIPPER HANDOFF", err)

    def test_on_starts_and_closes(self):
        """켜짐이면 수신을 시작·정리하고 요약·기록에 넘기기 정보가 있다."""
        made = []

        def factory(config, sink):
            """대역 수신."""
            made.append(FakeTransport(config, sink))
            return made[-1]

        code, err, out = self.run_grip(Scenario(5), GRIP, factory)
        self.assertEqual(code, 0, err)
        self.assertTrue(made[0].started and made[0].closed)
        self.assertEqual(made[0].config.depth_topic, "/gripper_cam/depth_image")
        self.assertIn("GRIPPER HANDOFF ON", err)
        world = self.summary(out)["world_tracking"]
        self.assertEqual(world["handoff"]["visibility"]["hold_limit_ms"], 30000)
        self.assertEqual(world["diagnostics"]["gripper_receiver"]["emitted"], 0)
        recs = self.world_records(out)
        self.assertTrue(all(r.get("camera") == "link" for r in recs if r["kind"] == "frame"))

    def test_argument_errors(self):
        """pixel 추적이나 블록 없는 설정에 --gripper-handoff는 2."""
        for extra in (("--tracking", *PIXEL, "--gripper-handoff"), WORLD + ("--gripper-handoff",)):
            with self.subTest(extra=extra):
                code, err, _ = self.run_grip(Scenario(3), extra, lambda c, s: self.fail("gripper must not start"))
                self.assertEqual(code, 2, err)


def depth_msg(stamp_ns, arr, frame="gripper_cam_optical_frame", encoding="32FC1", big=False):
    """깊이 Image 대역."""
    h, w = arr.shape
    data = arr.astype(">f4" if big else "<f4").tobytes()

    return SimpleNamespace(header=SimpleNamespace(frame_id=frame, stamp=SimpleNamespace(sec=stamp_ns // 10 ** 9,
                                                                                       nanosec=stamp_ns % 10 ** 9)),
                           encoding=encoding, height=h, width=w, step=w * 4, is_bigendian=int(big), data=data)


def info_msg(stamp_ns, w=4, h=3, frame="gripper_cam_optical_frame"):
    """CameraInfo 대역."""
    return SimpleNamespace(header=SimpleNamespace(frame_id=frame, stamp=SimpleNamespace(sec=stamp_ns // 10 ** 9,
                                                                                       nanosec=stamp_ns % 10 ** 9)),
                           width=w, height=h, k=[2.0, 0, 2, 0, 2.0, 1.5, 0, 0, 1])


class ReceiverTest(unittest.TestCase):
    """짝짓기·솎기·해석."""

    def setUp(self):
        """수신기와 받은 목록."""
        self.got = []
        self.rx = GripperDepthReceiver(gcfg(max_rate_hz=10), self.got.append)
        self.arr = np.arange(12, dtype=np.float32).reshape(3, 4) / 10

    def test_pair_any_order_and_rate(self):
        """같은 stamp 짝은 순서와 무관하게 넘기고, 10 Hz보다 촘촘하면 건너뛴다."""
        self.rx.handle_info(info_msg(1_000_000_000))
        self.rx.handle_depth(depth_msg(1_000_000_000, self.arr))
        self.rx.handle_depth(depth_msg(1_025_000_000, self.arr))
        self.rx.handle_info(info_msg(1_025_000_000))
        self.rx.handle_depth(depth_msg(1_100_000_000, self.arr))
        self.rx.handle_info(info_msg(1_100_000_000))
        self.assertEqual([f.timestamp_ms for f in self.got], [1000, 1100])
        self.assertEqual(self.rx.stats()["rate_skipped"], 1)
        np.testing.assert_array_equal(self.got[0].depth_m, self.arr)
        self.assertFalse(self.got[0].depth_m.flags.writeable)

    def test_rejects(self):
        """프레임 이름·인코딩·크기 불일치는 넘기지 않고 센다."""
        self.rx.handle_depth(depth_msg(1, self.arr, frame="other"))
        self.rx.handle_depth(depth_msg(2, self.arr, encoding="16UC1"))
        self.rx.handle_info(info_msg(2))
        self.rx.handle_depth(depth_msg(3, self.arr))
        self.rx.handle_info(info_msg(3, w=5))
        s = self.rx.stats()
        self.assertEqual((s["frame_mismatch"], s["bad_depth"], s["bad_info"], len(self.got)), (1, 1, 1, 0))

    def test_big_endian_and_roi(self):
        """빅엔디언 32FC1 해석, frames.yaml 벨트 사각형."""
        msg = depth_msg(5, self.arr, big=True)
        np.testing.assert_array_equal(decode_depth(msg), self.arr)
        roi = belt_roi_xy_m({"belt": {"center": [0.55, 0], "length": 1.26, "width": 0.20, "direction": [0.0, 1.0]}})
        np.testing.assert_allclose(roi, (0.45, 0.65, -0.63, 0.63))


if __name__ == "__main__":
    unittest.main()
