"""#217 W 추적 CLI 시험(W8·W10·W12·W14): 실시간 기록 연결 + `--tracking --tracking-space world`.

근거: 계획 `_workspace/217_builder2_world_tracking_plan_r0.md`·`_r1.md`, reviewer 보충 2(정답 격리: C `speed_mps` 조회값은 진단 기록만).
- 기존 실시간 CLI 시험의 가짜 수신·가짜 모델(같은 볼트 마스크)·가짜 TF(고정 `T_W_C`)·#214 가짜 송신 런타임을 쓴다. 실제 ROS·Gazebo 없음.
- B 계산은 실제 `BStage`다(W 번호 → 하위 형 관측 → B `_dedupe`).
"""
from __future__ import annotations

import io
import json
import re
import threading
import unittest
from pathlib import Path

from eyetoarm.vision.live_observation_link import LiveObservationLink
from test_live_sort_part import SORT_CONFIG, FakeRuntime
from test_live_tracking_link import ReleasingRuntime, TrackCli
from test_run_pipeline_capture_observation import model_with
from test_run_pipeline_live_observation import CONFIG, FakeTf, Scenario

ROOT = Path(__file__).resolve().parents[2]
WORLD_CONFIG = ROOT / "config" / "vision" / "world_tracker_test.json"
WORLD = ("--tracking", "--tracking-space", "world", "--world-tracker-config", str(WORLD_CONFIG))


class WorldCli(TrackCli):
    """CLI 실행 도우미(+ 진단 조회 주입)."""

    def run_world(self, scenario, *, extra=WORLD + ("--live-sort-part",), config=SORT_CONFIG, runtime=None, probe=None,
                  out=None):
        """main 실행 → (종료 코드, stderr, 출력 폴더, 런타임)."""
        out = out or self.out()
        log = str(Path(self.tmp.name) / f"{Path(out).name}.jsonl")
        rt = runtime or ReleasingRuntime()
        scenario.link = lambda: self.controller.current
        made = []
        argv = ["--ros-link", "--detector", "ai", "--ai-config", self.case.ai, "--log", log,
                "--live-observation-out", out, "--live-observation-config", str(config), "--max-frames", str(scenario.n),
                *extra]
        err = io.StringIO()
        probe = probe or (lambda: {"node": "/conveyor_controller", "speed_mps": None, "status": "no_node"})

        try:
            code = self.mod.main(argv, stderr=err, ros_available=lambda: True,
                                 model_factory=lambda path: model_with(scenario.masks_fn),
                                 live_source_factory=scenario.factory_maker(made), live_tf_factory=lambda c: FakeTf(),
                                 live_run_controller=self.controller, live_sort_part_factory=self.factory(rt),
                                 live_conveyor_probe=probe)
        except SystemExit as exc:
            code = exc.code

        return code, err.getvalue(), out, rt

    def world_records(self, out):
        """world_tracks.jsonl."""
        return [json.loads(t) for t in (Path(out) / "world_tracks.jsonl").read_text(encoding="utf-8").splitlines()]


class WorldCliTest(WorldCli):
    """W14·W10·W12."""

    def test_w14_world_ids_reach_b_and_dedupe(self):
        """W14: W 번호가 하위 형 관측 → 실제 B goal로 가고, 결과 전 두 번째부터 ALREADY_SENT, 송신은 번호당 1회."""
        code, err, out, rt = self.run_world(Scenario(10))
        self.assertEqual(code, 0, err)
        self.assertIn("tracking ON space=world", err)
        self.assertIn("tracking=world", err)
        self.assertNotIn("tracking ON M2(heuristic)", err)
        acc = self.accepts(out)
        with_id = [r for r in acc if r["input"]["part_track_id"] is not None]
        self.assertTrue(with_id, [r["input"]["track_state"] for r in acc])
        self.assertTrue(all(r["goal"]["part_track_id"] == r["input"]["part_track_id"] for r in acc))
        self.assertEqual(with_id[0]["goal_kind"], "PICK_PLACE")
        self.assertTrue(all((r["goal_kind"], r["b_reason"]) == ("SKIP", "ALREADY_SENT") for r in with_id[1:]))
        self.assertEqual(len([g for g in rt.sent if g["part_track_id"] is not None]), 1)
        self.assertTrue(all(t == "TrackedPlanarObservation" for t, *_ in self.b_seen))
        s = self.summary(out)
        self.assertEqual(s["tracking_link"]["mode"], "world")
        self.assertIn("world_tracking", s)
        self.assertTrue(s["cleanup_confirmed"])
        recs = self.world_records(out)
        self.assertTrue(recs and all(r["tf_status"] == "ok" for r in recs if r["kind"] == "frame"))
        self.assertTrue(any(t["part_track_id"] is not None for r in recs for t in r.get("tracks", [])))

    def test_w14_no_runner_pixel_tracker(self):
        """W8/W14: W 모드에서는 실행기 픽셀 추적기를 만들지 않는다(offer에 tracks 인자 없음, ai_tracks 없음)."""
        kwargs = []
        real = LiveObservationLink.offer

        def spy(link, outcome, reason, epoch, **kw):
            kwargs.append(kw)
            return real(link, outcome, reason, epoch, **kw)

        from unittest import mock

        with mock.patch.object(LiveObservationLink, "offer", spy):
            code, err, out, rt = self.run_world(Scenario(5))

        self.assertEqual(code, 0, err)
        self.assertTrue(kwargs and all(kw == {} for kw in kwargs))
        log = Path(self.tmp.name) / f"{Path(out).name}.jsonl"
        self.assertFalse(any('"ai_tracks"' in line for line in log.read_text(encoding="utf-8").splitlines()))

    def test_w10_speed_probe_does_not_change_tracking(self):
        """W10(정답 격리): C speed_mps 조회값을 0·0.2·99로 바꿔도 world_tracks 배정·번호와 B 목표가 같다. 진단 필드만 다르다."""
        outcomes = []

        for i, value in enumerate((0.0, 0.2, 99.0)):
            probe = (lambda v: (lambda: {"node": "/conveyor_controller", "speed_mps": v, "status": "ok"}))(value)
            code, err, out, rt = self.run_world(Scenario(8), probe=probe, out=self.out(f"w10_{i}"))
            self.assertEqual(code, 0, err)
            frames = [[(t["instance_index"], t["part_track_id"], t["state"], t["center_mm"]) for t in r["tracks"]]
                      for r in self.world_records(out) if r["kind"] == "frame"]
            goals = [(g["kind"], g["part_track_id"]) for g in rt.sent]
            diag = self.summary(out)["world_tracking"]["diagnostics"]["conveyor_speed_mps"]
            outcomes.append((frames, goals, diag["speed_mps"]))
            self.assertIn(f"diagnostic={value}", err)

        self.assertEqual(outcomes[0][0], outcomes[1][0])
        self.assertEqual(outcomes[1][0], outcomes[2][0])
        self.assertEqual(outcomes[0][1], outcomes[2][1])
        self.assertEqual([o[2] for o in outcomes], [0.0, 0.2, 99.0])

    def test_w10_tracker_has_no_probe_reference(self):
        """W10: 추적기 모듈은 진단 조회기·speed_mps를 참조하지 않는다(코드에 이름 없음)."""
        for name in ("world_tracking.py", "live_world_tracking.py"):
            code = (ROOT / "src" / "eyetoarm" / "vision" / name).read_text(encoding="utf-8").split('"""', 2)[2]
            code = re.sub(r"#.*$", "", code, flags=re.M)   # 주석(설명 문구)은 빼고 코드만 본다
            # 탐색 상한 이름 `max_belt_speed_mps`·진단 키 `conveyor_speed_mps` 말고는 speed_mps가 없어야 한다
            self.assertIsNone(re.search(r"(?<!max_belt_)(?<!conveyor_)speed_mps", code))

    def test_w12_control_flow_does_no_tf(self):
        """W12: 제어 흐름(MainThread)은 TF를 조회하지 않는다(추적 스레드·worker·열기 스레드만)."""
        tf = FakeTf()
        self.run_world_with_tf(Scenario(5), tf)
        self.assertNotIn("MainThread", tf.threads)
        self.assertIn("eyetoarm-live-world-tracking", tf.threads)

    def run_world_with_tf(self, scenario, tf):
        """가짜 TF를 주입해 실행."""
        out = self.out("tfw")
        log = str(Path(self.tmp.name) / "tfw.jsonl")
        scenario.link = lambda: self.controller.current
        made = []
        argv = ["--ros-link", "--detector", "ai", "--ai-config", self.case.ai, "--log", log, "--live-observation-out", out,
                "--live-observation-config", str(CONFIG), "--max-frames", str(scenario.n), *WORLD]
        code = self.mod.main(argv, stderr=io.StringIO(), ros_available=lambda: True,
                             model_factory=lambda path: model_with(scenario.masks_fn),
                             live_source_factory=scenario.factory_maker(made), live_tf_factory=lambda c: tf,
                             live_run_controller=self.controller,
                             live_conveyor_probe=lambda: {"node": "x", "speed_mps": None, "status": "no_node"})
        self.assertEqual(code, 0)

    def test_record_only_world_tracking(self):
        """W 추적 + 송신 끔(기록 전용): 번호가 accept 기록에 실리고 수락 뒤 release(ALREADY_SENT로 굳지 않음)."""
        code, err, out, rt = self.run_world(Scenario(8), extra=WORLD, config=CONFIG)
        self.assertEqual(code, 0, err)
        acc = [r for r in self.accepts(out) if r["input"]["part_track_id"] is not None]
        self.assertTrue(acc and all(r["goal_kind"] == "PICK_PLACE" for r in acc))


class WorldArgsTest(WorldCli):
    """W8(조합)·설정 오류."""

    def test_argument_errors(self):
        """world인데 --tracking 없음·기록 연결 없음·설정 없음, pixel인데 설정 있음, conveyor_state 비null은 2."""
        bad = Path(self.tmp.name) / "bad_world.json"
        doc = json.loads(WORLD_CONFIG.read_text(encoding="utf-8"))
        doc["conveyor_state"] = {"topic": "/conveyor/state"}
        bad.write_text(json.dumps(doc), encoding="utf-8")
        base = ["--detector", "ai", "--ai-config", self.case.ai, "--log", str(Path(self.tmp.name) / "x.jsonl")]
        live = ["--ros-link", *base, "--live-observation-out", self.out(), "--live-observation-config", str(CONFIG)]
        cases = [[*live, "--tracking-space", "world", "--world-tracker-config", str(WORLD_CONFIG)],
                 ["--ros-link", *base, "--tracking", "--tracking-space", "world", "--world-tracker-config", str(WORLD_CONFIG)],
                 [*live, "--tracking", "--tracking-space", "world"],
                 [*live, "--tracking", "--world-tracker-config", str(WORLD_CONFIG)],
                 [*live, "--tracking", "--tracking-space", "world", "--world-tracker-config", str(bad)]]

        for argv in cases:
            with self.subTest(argv=argv):
                err = io.StringIO()

                try:
                    code = self.mod.main(argv, stderr=err, ros_available=lambda: True,
                                         live_tf_factory=lambda c: self.fail("tf must not start"),
                                         live_run_controller=self.controller)
                except SystemExit as exc:
                    code = exc.code

                self.assertEqual(code, 2, err.getvalue())

    def test_pixel_default_unchanged(self):
        """W8: --tracking만(기본 pixel)이면 #215 동작 그대로(world_tracks 없음, 실행기 추적 tracks 인자 있음)."""
        code, err, out, rt = self.run_world(Scenario(5), extra=("--tracking", "--live-sort-part"))
        self.assertEqual(code, 0, err)
        self.assertFalse((Path(out) / "world_tracks.jsonl").exists())
        self.assertIn("tracking ON M2(heuristic)", err)
        self.assertNotIn("world_tracking", self.summary(out))


if __name__ == "__main__":
    unittest.main()
