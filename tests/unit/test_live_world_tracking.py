"""W 추적 CLI 시험: 실시간 기록 연결 + `--tracking --tracking-space world`.

가짜 수신·모델·TF(고정 `T_W_C`)·송신 런타임과 실제 `BStage`를 쓴다. 실제 ROS·Gazebo 없음.
C `speed_mps` 조회값은 진단 기록만(정답 격리).
"""
from __future__ import annotations

import io
import json
import re
import threading
import unittest
from pathlib import Path

from eyetoarm.vision.live_observation_link import LiveObservationLink
from part_spec_copies import post_b12, pre_b12, write_specs
from test_live_sort_part import FakeRuntime
from test_live_tracking_link import PIXEL, ReleasingRuntime, TrackCli, sort_config
from test_run_pipeline_capture_observation import model_with
from test_run_pipeline_live_observation import CONFIG, FakeTf, Scenario

ROOT = Path(__file__).resolve().parents[2]
WORLD_CONFIG = ROOT / "config" / "vision" / "world_tracker_test.json"
WORLD = ("--tracking", "--tracking-space", "world", "--world-tracker-config", str(WORLD_CONFIG))


class WorldCli(TrackCli):
    """CLI 실행 도우미(+ 진단 조회 주입)."""

    def run_world(self, scenario, *, extra=WORLD + ("--live-sort-part",), config=None, runtime=None, probe=None,
                  out=None):
        """main 실행 → (종료 코드, stderr, 출력 폴더, 런타임)."""
        out = out or self.out()
        config = config or sort_config(self.tmp.name)   # 추적 켬 송신은 require_track_id 필수
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
    """W 번호의 B 전달·정답 격리·TF 조회 스레드."""

    def test_world_planes_from_b_specs_per_spec(self):
        """W 투영 높이를 B 규격에서 읽는다(B-12 전·후 규격 사본 주입). B-12 전(모두 105 mm)이면 「uniform」·기록 높이
        0.105·종류 없음, B-12 뒤(BOLT_A 102 mm)면 시작 줄에 종류별 높이가 찍히고 기록 높이는 그 두 값 중 하나다.
        마스크 기본 높이는 B `class_plane_z(specs, "bolt")` = 최상위 plane_z(105 mm)."""
        self.mod.PART_SPECS_PATH = write_specs(pre_b12(), Path(self.tmp.name) / "specs_pre_b12.json")
        code, err, out, _ = self.run_world(Scenario(6))
        self.assertEqual(code, 0, err)
        self.assertIn("bolt planes BOLT_A=105.0mm BOLT_B=105.0mm, mask 105.0mm (uniform)", err)
        tracks = [t for r in self.world_records(out) if r.get("kind") == "frame" for t in r.get("tracks", [])]
        self.assertTrue(tracks)
        self.assertEqual({(t["plane_z_m"], t["plane_spec"]) for t in tracks if t["input"] != "none"}, {(0.105, None)})

        self.mod.PART_SPECS_PATH = write_specs(post_b12(), Path(self.tmp.name) / "specs_post_b12.json")
        code, err, out, _ = self.run_world(Scenario(6), out=self.out("split"))
        self.assertEqual(code, 0, err)
        self.assertIn("bolt planes BOLT_A=102.0mm BOLT_B=105.0mm, mask 105.0mm (per spec by W length)", err)
        planes = {t["plane_z_m"] for r in self.world_records(out) if r.get("kind") == "frame"
                  for t in r.get("tracks", []) if t["input"] == "endpoint"}
        self.assertTrue(planes and planes <= {0.102, 0.105}, planes)

    def test_w14_world_ids_reach_b_and_dedupe(self):
        """W 번호가 하위 형 관측 → 실제 B goal로 가고, 결과 전 두 번째부터 ALREADY_SENT, 송신은 번호당 1회."""
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
        """W 모드에서는 실행기 픽셀 추적기를 만들지 않는다(offer에 tracks 인자 없음, ai_tracks 없음)."""
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
        """정답 격리: C speed_mps 조회값을 0·0.2·99로 바꿔도 world_tracks 배정·번호와 B 목표가 같다. 진단 필드만 다르다."""
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
        """추적기 모듈은 진단 조회기·speed_mps를 참조하지 않는다(코드에 이름 없음)."""
        for name in ("world_tracking.py", "live_world_tracking.py"):
            code = (ROOT / "src" / "eyetoarm" / "vision" / name).read_text(encoding="utf-8").split('"""', 2)[2]
            code = re.sub(r"#.*$", "", code, flags=re.M)   # 주석(설명 문구)은 빼고 코드만 본다
            # 탐색 상한 이름 `max_belt_speed_mps`·진단 키 `conveyor_speed_mps` 말고는 speed_mps가 없어야 한다
            self.assertIsNone(re.search(r"(?<!max_belt_)(?<!conveyor_)speed_mps", code))

    def test_w12_control_flow_does_no_tf(self):
        """제어 흐름(MainThread)은 TF를 조회하지 않는다(추적 스레드·worker·열기 스레드만)."""
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
    """옵션 조합·설정 오류."""

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
                 # 공간 생략은 W라 pixel을 명시
                 [*live, "--tracking", *PIXEL, "--world-tracker-config", str(WORLD_CONFIG)],
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

    def test_pixel_explicit_unchanged(self):
        """--tracking-space pixel 명시는 픽셀 추적 그대로(world_tracks 없음, 실행기 tracks 인자 있음)."""
        code, err, out, rt = self.run_world(Scenario(5), extra=("--tracking", *PIXEL, "--live-sort-part"))
        self.assertEqual(code, 0, err)
        self.assertFalse((Path(out) / "world_tracks.jsonl").exists())
        self.assertIn("tracking ON M2(heuristic)", err)
        self.assertNotIn("world_tracking", self.summary(out))


if __name__ == "__main__":
    unittest.main()
