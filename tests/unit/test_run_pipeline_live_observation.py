"""`run_pipeline.py --ros-link --live-observation-out/--live-observation-config`(#173 실시간 A→B 기록 전용 시험) CLI 시험.

근거: reviewer 착수 명세(2026-11-18), `vision/live_observation_link.py`, `_workspace/173_builder2_reset_new_run_report.md`.
- 진입점 `main()`부터 실제 `BStage`까지 돈다: 주입한 `LiveLinkSource`(가짜 수신 연결) → 실행기 → `AiSegDetector`(가짜 모델, 실제
  convert·M1) → 코어 → `live_observation_sink.offer` → 전용 디스패처 → 주입 TF → 봉투 → B 입구 어댑터 → `BStage.consume` → 주석·기록.
- 실제 ROS·Gazebo는 쓰지 않는다. 운영값은 `config/vision/live_observation_test.json`(status=test_only) 시험 설정이다.
- C 송신 감시: `make_c_handoff`·`SortPartClient` 호출 0. 결과는 배선 확인이며 정확도가 아니다.
"""
import importlib.util
import io
import threading
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from eyetoarm.inputs.ros_image import LiveLinkSource, LiveSourceConfig
from eyetoarm.transforms import b_stage as b_stage_mod
from eyetoarm.transforms import sort_part_client
from eyetoarm.vision import pipeline_runner
from eyetoarm.vision.live_run_session import LiveRunController
from test_ros_image_adapter import FakeTransport, image_msg, info_msg, stamp, view_convert
from test_run_pipeline_capture_observation import FRAME_ID, H, T_W_C, W, Case, bolt_mask, model_with, rect_mask

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_pipeline.py"
CONFIG = ROOT / "config" / "vision" / "live_observation_test.json"
STAMP0 = 4_291_000_000
DT = 66_000_000


def ns_parts(ns):
    """ns → (sec, nanosec)."""
    return ns // 1_000_000_000, ns % 1_000_000_000


class FakeTf:
    """주입 TF 수신 묶음: 촬영 stamp별 동작(ok·missing·late·mismatch). 닫기 실패 횟수 주입."""

    def __init__(self, modes=None, *, fail_close=0, gate=None):
        """기본은 모든 stamp ok. `gate`(stamp → Event)면 그 stamp 조회를 Event까지 막는다(지연 장벽)."""
        self.modes = dict(modes or {})
        self.gate = dict(gate or {})
        self.calls = []
        self.threads = set()
        self.fail_close = fail_close
        self.closed = 0

    def lookup(self, parent, child, stamp_ns, timeout_s):
        """정확 stamp 조회 대역."""
        self.calls.append((parent, child, stamp_ns))
        self.threads.add(threading.current_thread().name)
        mode = self.modes.get(stamp_ns, "ok")

        if stamp_ns in self.gate:
            self.gate[stamp_ns].wait(5.0)

        if parent != "world" or child != FRAME_ID:
            raise LookupError(f"unexpected {parent}->{child}")

        if mode == "missing":
            raise LookupError(f"no tf at {stamp_ns}")

        if mode == "late":
            time.sleep(0.08)    # 시험 설정 상한 50 ms 초과

        return T_W_C.copy(), stamp_ns + (1 if mode == "mismatch" else 0)

    def has_static(self, parent, child):
        """정적 쌍 확인(시험 설정은 정적 쌍 없음)."""
        return True

    def clear(self):
        """연결부는 부르지 않는다."""
        raise AssertionError("live observation link must not clear a fresh buffer")

    def close(self, limit_s):
        """닫기(실패 주입)."""
        if self.fail_close > 0:
            self.fail_close -= 1
            raise RuntimeError("tf close failed")

        self.closed += 1


class Scenario:
    """프레임 계획: 프레임별 마스크·검출 중 동작. 검출 i 중에 i+1번째 영상을 넣는다(결정적 순서)."""

    def __init__(self, n, *, masks=None, during=None, sync=True):
        """n장, masks[i]() 마스크 목록, during[i](scenario) 검출 중 동작.

        `sync`: 검출마다 먼저 연결부의 열기 시도·디스패처가 쉴 때까지 기다린다(결정적 순서). 지연 장벽 시험은 False.
        """
        self.n = n
        self.masks = masks or {}
        self.during = during or {}
        self.sync = sync
        self.link = None   # 시험이 실행 중 연결부를 꺼내는 함수(Base가 넣음)
        self.src = None
        self.detections = 0

    def stamp(self, i):
        """i번째 촬영 stamp."""
        return STAMP0 + i * DT

    def push(self, i):
        """i번째 /clock·영상."""
        sec, nsec = ns_parts(self.stamp(i))
        self.src.handle_clock(SimpleNamespace(clock=stamp(sec, nsec)))
        self.src.handle_image(image_msg(sec, nsec, width=W, height=H, frame_id=FRAME_ID))

    def factory_maker(self, made):
        """CLI가 부르는 `(source_id, reliability, boundary_hook=)` 팩토리 생성기."""
        def make(source_id, reliability, boundary_hook=None):
            def factory(issuer, generation_provider):
                self.src = LiveLinkSource(source_id=source_id, issuer=issuer, generation_provider=generation_provider,
                                          config=LiveSourceConfig(frame_wait_s=0.01),
                                          transport_factory=lambda s: FakeTransport(), convert=view_convert,
                                          boundary_hook=boundary_hook)
                made.append(self.src)
                self.src.handle_camera_info(info_msg(fx=300.0, width=W, height=H, frame_id=FRAME_ID))
                self.push(0)
                return self.src

            return factory

        return make

    def masks_fn(self):
        """가짜 모델이 검출마다 부른다: 이번 프레임 동작 → 다음 영상 넣기 → 마스크."""
        i = self.detections
        self.detections += 1
        link = None if self.link is None else self.link()

        if self.sync and link is not None:
            end = time.monotonic() + 0.5

            while time.monotonic() < end and not link.probes_idle():
                time.sleep(0.002)

            link.dispatcher.drain(0.5)

        action = self.during.get(i)

        if action is not None:
            action(self)

        if i + 1 < self.n:
            self.push(i + 1)

        return self.masks.get(i, lambda: [bolt_mask()])()


class Base(unittest.TestCase):
    """CLI 도우미·C 송신 감시·B 호출 감시."""

    def setUp(self):
        """임시 폴더·CLI 모듈·감시."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.case = Case(self.tmp.name)
        spec = importlib.util.spec_from_file_location("run_pipeline_live_obs_under_test", SCRIPT)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self.consume_calls, self.c_calls = [], []
        real_consume = b_stage_mod.BStage.consume

        self.outputs = {}

        def consume(stage, frame, obs):
            """BStage.consume 감시(같은 프레임·관측 짝, 실제 BOutput 보관)."""
            self.consume_calls.append((frame.image_seq, obs.image_seq, frame.ros_stamp_ns))
            out = real_consume(stage, frame, obs)
            self.outputs[frame.ros_stamp_ns] = out
            return out

        def c_spy(*a, **k):
            """C 송신 경로가 불리면 실패."""
            self.c_calls.append(a)
            raise AssertionError("no C output in live record-only test")

        for p in (mock.patch.object(b_stage_mod.BStage, "consume", consume),
                  mock.patch.object(b_stage_mod, "make_c_handoff", c_spy),
                  mock.patch.object(sort_part_client.SortPartClient, "send", c_spy),
                  mock.patch.object(sort_part_client.SortPartClient, "__init__", c_spy)):
            p.start()
            self.addCleanup(p.stop)

        self.controller = LiveRunController(lambda rid: None, close_limit_s=1.0)

    def out(self, name="live"):
        """출력 폴더 경로(아직 없음)."""
        return str(Path(self.tmp.name) / name)

    def run_cli(self, scenario, *, tf=None, out=None, log=None, extra=(), config=CONFIG, max_frames=None,
                hook=None):
        """main 실행 → (종료 코드, stderr, 출력 폴더)."""
        out = out or self.out()
        log = log or str(Path(self.tmp.name) / f"{Path(out).name}.jsonl")
        made = []
        tf = tf if tf is not None else FakeTf()
        scenario.link = lambda: self.controller.current
        argv = ["--ros-link", "--detector", "ai", "--ai-config", self.case.ai, "--log", log,
                "--live-observation-out", out, "--live-observation-config", str(config),
                "--max-frames", str(max_frames or scenario.n), *extra]
        err = io.StringIO()

        try:
            code = self.mod.main(argv, stderr=err, ros_available=lambda: True,
                                 model_factory=lambda path: model_with(scenario.masks_fn),
                                 live_source_factory=scenario.factory_maker(made), live_tf_factory=lambda cfg: tf,
                                 live_run_controller=self.controller, b_stage_hook=hook)
        except SystemExit as exc:
            code = exc.code

        self.tf = tf
        return code, err.getvalue(), out

    def records(self, out):
        """공통 순번 기록."""
        p = Path(out) / "b_live_events.jsonl"
        return [json.loads(t) for t in p.read_text(encoding="utf-8").splitlines()]

    def skipped(self, out):
        """건너뜀 기록."""
        p = Path(out) / "skipped.jsonl"
        return [json.loads(t) for t in p.read_text(encoding="utf-8").splitlines()]

    def summary(self, out):
        """요약."""
        return json.loads((Path(out) / "summary.json").read_text(encoding="utf-8"))


class NormalPathTest(Base):
    """정상 좌표·같은 프레임 주석."""

    def test_accepts_with_exact_stamp_tf_and_same_frame_annotation(self):
        """3장: 첫 장으로 열리고, 각 수락 기록은 그 프레임의 image_seq·촬영 stamp·TF stamp가 같고 주석 PNG가 같은 크기다."""
        sc = Scenario(3)
        code, err, out = self.run_cli(sc)
        self.assertEqual(code, 0, err)
        self.assertIn("LIVE A->B RECORD-ONLY TEST", err)
        self.assertNotIn("MainThread", self.tf.threads)                   # r1 F1: A 제어 흐름은 TF를 조회하지 않음

        acc = [r for r in self.records(out) if r["kind"] == "accept"]
        self.assertGreaterEqual(len(acc), 1)
        self.assertEqual(self.c_calls, [])

        for r in acc:
            stamp_ns = r["input"]["ros_stamp_ns"]
            self.assertEqual(r["tf_status"], "ok")
            self.assertEqual(r["b_input"]["capture"]["stamp_ns"], stamp_ns)
            self.assertEqual(r["b_input"]["tf"]["stamp_ns"], stamp_ns)
            self.assertIn((r["input"]["image_seq"], r["input"]["image_seq"], stamp_ns), self.consume_calls)
            self.assertIsNotNone(r["annotation"])
            png = (Path(out) / r["annotation"]).read_bytes()
            self.assertEqual(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR).shape, (H, W, 3))

        self.assertEqual([r["input"]["ros_stamp_ns"] for r in acc], [sc.stamp(1), sc.stamp(2)])   # 마지막까지 마무리
        skipped = {r["ros_stamp_ns"]: r["reason"] for r in self.skipped(out)}
        self.assertEqual(skipped, {sc.stamp(0): "not_ready"})            # 열기에 쓴 첫 프레임은 계산하지 않음

        # r1 F2: 수락 기록의 B 좌표·각도·판정 사유가 실제 BStage 결과와 같다
        from eyetoarm.vision.live_observation_events import _plain

        for r in acc:
            real = self.outputs[r["input"]["ros_stamp_ns"]]
            self.assertEqual(r["result_dict"], json.loads(json.dumps(_plain(real.result_dict))))
            self.assertEqual(r["goal"], json.loads(json.dumps(_plain(real.goal))))
            self.assertEqual((r["goal_kind"], r["b_valid"]), (real.goal.get("kind"), real.result.valid))

        s = self.summary(out)
        # #233 r1: 실시간 A→B 실행은 C 연결부(detect)를 항상 출구로 가진다(C 목표·D·로봇 출력은 없음)
        self.assertEqual((s["record_only"], s["cleanup_confirmed"], s["reset_detected"], s["drained_before_end"]),
                         (False, True, False, True))
        self.assertEqual(s["outputs"], ["c_handoff_detect(127.0.0.1:47093)"])
        self.assertEqual(self.tf.closed, 1)
        self.assertTrue(all(c[2] in {sc.stamp(i) for i in range(3)} for c in self.tf.calls))


class FailurePathTest(Base):
    """실패 경로: 사유 기록, 계산·주석 0."""

    def assert_skipped_no_compute(self, out, stamp_ns, reasons):
        """그 stamp 프레임은 건너뜀 사유가 있고 B 계산·수락이 없다."""
        sk = [r for r in self.skipped(out) if r["ros_stamp_ns"] == stamp_ns]
        self.assertTrue(sk and sk[-1]["reason"] in reasons, (stamp_ns, sk))
        self.assertNotIn(stamp_ns, [c[2] for c in self.consume_calls])
        self.assertNotIn(stamp_ns, [r["input"]["ros_stamp_ns"] for r in self.records(out) if r["kind"] == "accept"])

    def test_tf_missing_late_mismatch(self):
        """TF 누락·50 ms 초과·stamp 불일치 프레임은 tf_* 사유로 건너뛰고, 정상 프레임은 수락된다."""
        sc = Scenario(5)
        tf = FakeTf({sc.stamp(1): "missing", sc.stamp(2): "late", sc.stamp(3): "mismatch"})
        code, err, out = self.run_cli(sc, tf=tf)
        self.assertEqual(code, 0, err)
        self.assert_skipped_no_compute(out, sc.stamp(1), {"tf_unavailable"})
        self.assert_skipped_no_compute(out, sc.stamp(2), {"tf_late"})
        self.assert_skipped_no_compute(out, sc.stamp(3), {"tf_stamp_mismatch"})
        self.assertEqual([r["input"]["ros_stamp_ns"] for r in self.records(out) if r["kind"] == "accept"], [sc.stamp(4)])

    def test_first_frame_without_tf_does_not_open(self):
        """첫 프레임 TF가 없으면 열리지 않고(not_ready) 다음 프레임 TF로 열린다."""
        sc = Scenario(3)
        code, err, out = self.run_cli(sc, tf=FakeTf({sc.stamp(0): "missing"}))
        self.assertEqual(code, 0, err)
        first = [r["reason"] for r in self.skipped(out) if r["ros_stamp_ns"] == sc.stamp(0)]
        self.assertEqual(first, ["not_ready", "open:dynamic_unavailable"])
        self.assertEqual([r["reason"] for r in self.skipped(out) if r["ros_stamp_ns"] == sc.stamp(1)], ["not_ready"])
        self.assertEqual([r["input"]["ros_stamp_ns"] for r in self.records(out) if r["kind"] == "accept"], [sc.stamp(2)])

    def test_invalid_observation(self):
        """무효 관측(M1 모호)은 invalid_observation으로 건너뛴다."""
        sc = Scenario(3, masks={2: lambda: [rect_mask()]})
        code, err, out = self.run_cli(sc)
        self.assertEqual(code, 0, err)
        self.assert_skipped_no_compute(out, sc.stamp(2), {"invalid_observation"})

    def test_calibration_change_discards_in_flight(self):
        """검출 중 CameraInfo 변경(보정 번호 바뀜) → 그 프레임은 세대 폐기·보정 불일치로 건너뛰고 B 없음."""
        def change(s):
            s.src.handle_camera_info(info_msg(fx=310.0, width=W, height=H, frame_id=FRAME_ID))

        sc = Scenario(4, during={2: change})
        code, err, out = self.run_cli(sc)
        self.assertEqual(code, 0, err)
        self.assert_skipped_no_compute(out, sc.stamp(2), {"generation_discarded", "calibration_mismatch",
                                                          "job_aborted", "rx_epoch_stale"})


class ResetAndRestartTest(Base):
    """리셋 감지 → 실행 종료·이전 결과 철회, 자동 재시작 없음, 정리 확인 뒤에만 명시 새 실행."""

    def rewind_at(self, i):
        """검출 i 중에 (앞 프레임 수락을 기다린 뒤) /clock 역행."""
        def act(s):
            link = self.controller.current
            end = time.monotonic() + 2.0

            while time.monotonic() < end and link.dispatcher.counts.get("accepted", 0) < 1:
                time.sleep(0.005)

            s.src.handle_clock(SimpleNamespace(clock=stamp(1, 0)))
        return {i: act}

    def test_reset_ends_run_and_revokes(self):
        """프레임 1 수락 뒤 리셋 → 실행 정지(종료 0, reset_detected), 수락 철회 기록, 정리 확인, 이후 수락 0."""
        sc = Scenario(5, during=self.rewind_at(2))
        code, err, out = self.run_cli(sc, max_frames=5)
        self.assertEqual(code, 0, err)
        self.assertIn("reason=reset_detected", err)
        self.assertIn("new run requires an explicit restart", err)
        recs = self.records(out)
        kinds = [r["kind"] for r in recs]
        acc = [r for r in recs if r["kind"] == "accept"]
        self.assertEqual([r["input"]["ros_stamp_ns"] for r in acc], [sc.stamp(1)])
        self.assertIn("revoke", kinds)
        self.assertEqual(kinds[-1], "run_end")
        self.assertTrue(recs[-1]["cause"].startswith("reset:sim_time_rewind"))
        s = self.summary(out)
        self.assertEqual((s["reset_detected"], s["cleanup_confirmed"]), (True, True))
        self.assertLess(sc.detections, 5)                     # 리셋 뒤 계속 처리하지 않음

    def test_explicit_new_run_after_cleanup(self):
        """리셋으로 끝난 뒤 다시 실행하면 새 run_id·새 출력으로 처음부터 열린다(이전 결과 없음)."""
        code1, err1, out1 = self.run_cli(Scenario(5, during=self.rewind_at(2)), out=self.out("a"))
        self.assertEqual(code1, 0, err1)
        code2, err2, out2 = self.run_cli(Scenario(2), out=self.out("b"))
        self.assertEqual(code2, 0, err2)
        s1, s2 = self.summary(out1), self.summary(out2)
        self.assertNotEqual(s1["run_id"], s2["run_id"])
        acc2 = [r for r in self.records(out2) if r["kind"] == "accept"]
        self.assertTrue(acc2)
        self.assertTrue(all(r["input"]["run_id"] == s2["run_id"] for r in acc2))

    def test_new_run_refused_until_cleanup_confirmed(self):
        """TF 닫기 실패 → 종료 1·정리 미확인 → 다음 실행 거부(1, 부품 생성 전) → 재정리 뒤 실행 성공."""
        code1, err1, _ = self.run_cli(Scenario(2), tf=FakeTf(fail_close=1), out=self.out("a"))
        self.assertEqual(code1, 1)
        self.assertIn("cleanup not confirmed", err1)

        tf2 = FakeTf()
        code2, err2, out2 = self.run_cli(Scenario(2), tf=tf2, out=self.out("b"))
        self.assertEqual(code2, 1)
        self.assertIn("live observation run refused", err2)
        self.assertFalse(Path(out2).exists())
        self.assertEqual(tf2.calls, [])

        self.assertEqual(self.controller.retry_cleanup().close_errors, ())
        code3, err3, _ = self.run_cli(Scenario(2), out=self.out("c"))
        self.assertEqual(code3, 0, err3)


class ControlFlowAndFailureTest(Base):
    """r1 F1·F3: A 제어 흐름 밖의 TF·디스크 쓰기, 시작 실패 정리, 쓰기·상태 실패의 종료 코드."""

    def test_tf_barrier_does_not_block_a_control_flow(self):
        """첫 열기 TF 조회를 장벽으로 막아도 A는 다음 프레임 검출·CameraInfo 경계 반영을 계속한다. TF·건너뜀 쓰기는 MainThread 밖."""
        gate = threading.Event()
        seen = {}

        def change(s):
            s.src.handle_camera_info(info_msg(fx=310.0, width=W, height=H, frame_id=FRAME_ID))

        def check_and_release(s):
            link = self.controller.current
            seen["blocked_probes_done"] = link.probes_done
            seen["detections"] = s.detections
            gate.set()

        sc = Scenario(4, sync=False, during={2: change, 3: check_and_release})
        tf = FakeTf(gate={STAMP0: gate})
        code, err, out = self.run_cli(sc, tf=tf)
        self.assertEqual(code, 0, err)
        self.assertEqual(seen, {"blocked_probes_done": 0, "detections": 4})   # 장벽 동안 A는 4번째 검출까지 진행
        s = self.summary(out)
        self.assertEqual(s["runner"]["frames"], 4)
        self.assertEqual(s["runner"]["live_inputs"].get("profile_changed"), 2)   # 장벽 중 경계 반영
        self.assertNotIn("MainThread", tf.threads)
        self.assertEqual(s["skip_write_threads"], ["eyetoarm-live-observation-skips"])

    def test_source_factory_build_failure_cleans_up(self):
        """소스 공장 구성 실패 → 종료 1, 세션·TF 정리 확인, 요약에 실패 기록, 다음 실행 허용."""
        sc = Scenario(2)
        tf = FakeTf()
        err = io.StringIO()

        def bad_maker(source_id, reliability, boundary_hook=None):
            raise RuntimeError("factory init failed")

        code = self.mod.main(["--ros-link", "--detector", "ai", "--ai-config", self.case.ai,
                              "--log", str(Path(self.tmp.name) / "f.jsonl"), "--live-observation-out", self.out("f"),
                              "--live-observation-config", str(CONFIG), "--max-frames", "2"],
                             stderr=err, ros_available=lambda: True, model_factory=lambda p: model_with(sc.masks_fn),
                             live_source_factory=bad_maker, live_tf_factory=lambda cfg: tf,
                             live_run_controller=self.controller)
        self.assertEqual(code, 1, err.getvalue())
        self.assertIn("factory init failed", err.getvalue())
        self.assertEqual(tf.closed, 1)
        self.assertTrue(self.controller.current.cleanup_confirmed)
        s = self.summary(self.out("f"))
        self.assertEqual((s["exit_code"], s["runner"], s["ended"]["reason"]), (1, None, "startup_failed"))
        self.assertTrue(any("factory init failed" in p for p in s["problems"]))
        code2, err2, _ = self.run_cli(Scenario(2), out=self.out("g"))
        self.assertEqual(code2, 0, err2)

    def test_source_build_failure_inside_run(self):
        """실행기 안 소스 생성 실패 → 종료 1, 정리 확인."""
        sc = Scenario(2)
        tf = FakeTf()
        err = io.StringIO()

        def maker(source_id, reliability, boundary_hook=None):
            def factory(issuer, generation_provider):
                raise RuntimeError("source start failed")
            return factory

        code = self.mod.main(["--ros-link", "--detector", "ai", "--ai-config", self.case.ai,
                              "--log", str(Path(self.tmp.name) / "h.jsonl"), "--live-observation-out", self.out("h"),
                              "--live-observation-config", str(CONFIG), "--max-frames", "2"],
                             stderr=err, ros_available=lambda: True, model_factory=lambda p: model_with(sc.masks_fn),
                             live_source_factory=maker, live_tf_factory=lambda cfg: tf,
                             live_run_controller=self.controller)
        self.assertEqual(code, 1, err.getvalue())
        self.assertEqual(tf.closed, 1)
        self.assertTrue(self.controller.current.cleanup_confirmed)

    def test_skip_write_error_fails_run(self):
        """건너뜀 기록 쓰기 실패는 실행 성공으로 숨지 않는다(종료 1, 요약 problems)."""
        def break_skip_file(s):
            self.controller.current._skip_fh.close()

        code, err, out = self.run_cli(Scenario(4, during={1: break_skip_file},
                                               masks={2: lambda: [rect_mask()], 3: lambda: [rect_mask()]}))
        self.assertEqual(code, 1, err)
        s = self.summary(out)
        self.assertIsNotNone(s["skip_write_error"])
        self.assertTrue(any(p.startswith("skip writer error") for p in s["problems"]))
        self.assertEqual(s["state_failed"], "skip_writer_error")       # r2: 쓰기 실패는 상태 fail-stop
        q = s["skip_queue"]
        self.assertEqual(q["total"], q["written"] + q["dropped"] + q["unwritten"])
        self.assertGreaterEqual(q["dropped"] + q["unwritten"], 1)

    def test_state_failure_fails_run(self):
        """상태 fail-stop(예: 기록기 넘침)은 종료 1로 드러난다."""
        def fail_state(s):
            self.controller.current.state.fail("injected_state_failure")

        code, err, out = self.run_cli(Scenario(3, during={2: fail_state}))
        self.assertEqual(code, 1, err)
        s = self.summary(out)
        self.assertEqual(s["state_failed"], "injected_state_failure")
        self.assertTrue(any(p.startswith("state failed") for p in s["problems"]))


class BlockingFile:
    """건너뜀 파일 쓰기 장벽: 첫 write가 Event까지 막힌다(느린·멈춘 디스크 대역)."""

    def __init__(self, real, gate):
        """실제 파일과 장벽."""
        self.real, self.gate = real, gate
        self.closed = False

    def write(self, text):
        """장벽 뒤 실제 쓰기."""
        self.gate.wait(5.0)
        return self.real.write(text)

    def flush(self):
        """flush."""
        self.real.flush()

    def close(self):
        """닫기."""
        self.closed = True
        self.real.close()


class SkipQueueBoundTest(Base):
    """r2: 건너뜀 기록 대기 상한·포화·종료 코드·먼저 온 기록 보존·누락 계수(쓰기 장벽)."""

    def config_with(self, capacity):
        """대기 상한만 바꾼 시험 설정."""
        doc = json.loads(CONFIG.read_text(encoding="utf-8"))
        doc["skip_queue_capacity"] = capacity
        p = Path(self.tmp.name) / f"cfg_{capacity}.json"
        p.write_text(json.dumps(doc), encoding="utf-8")
        return p

    def test_overflow_under_write_barrier(self):
        """쓰기가 멈춘 동안 상한 3에 이르면 4번째에서 포화 → 상태 fail-stop·실행 정지·종료 1. 먼저 온 3건만 순서대로 쓰이고,
        포화 전 수락(2건)은 남고 포화 뒤 수락 0. 계수: 전체 = 씀 + 누락. 제어 흐름은 막히지 않는다."""
        gate = threading.Event()
        snap = {}
        invalid = {i: (lambda: [rect_mask()]) for i in range(3, 12)}

        def install_barrier(s):
            link = self.controller.current
            link._skip_fh = BlockingFile(link._skip_fh, gate)

            def releaser():
                end = time.monotonic() + 5.0
                while time.monotonic() < end and link.skip_overflow_at is None:
                    time.sleep(0.002)
                snap["accepted_at_overflow"] = link.dispatcher.counts.get("accepted", 0)
                snap["detections_at_overflow"] = s.detections
                time.sleep(0.05)
                gate.set()

            threading.Thread(target=releaser, daemon=True).start()

        sc = Scenario(12, during={0: install_barrier}, masks=invalid)
        code, err, out = self.run_cli(sc, config=self.config_with(3))
        self.assertEqual(code, 1, err)
        s = self.summary(out)
        q = s["skip_queue"]
        self.assertEqual(s["state_failed"], "skip_queue_overflow")
        self.assertEqual((q["capacity"], q["written"], q["overflow_at"], q["max_depth"], q["unwritten"]), (3, 3, 4, 3, 0))
        self.assertEqual(q["total"], q["written"] + q["dropped"])
        self.assertGreaterEqual(q["dropped"], 1)
        self.assertTrue(any(p.startswith("skip records lost") for p in s["problems"]))
        self.assertTrue(any(p == "state failed: skip_queue_overflow" for p in s["problems"]))

        rows = self.skipped(out)
        self.assertEqual([(r["ros_stamp_ns"], r["reason"]) for r in rows],
                         [(sc.stamp(0), "not_ready"), (sc.stamp(3), "invalid_observation"),
                          (sc.stamp(4), "invalid_observation")])          # 먼저 온 기록 보존·순서

        acc = [r for r in self.records(out) if r["kind"] == "accept"]
        self.assertEqual([r["input"]["ros_stamp_ns"] for r in acc], [sc.stamp(1), sc.stamp(2)])
        self.assertEqual(snap["accepted_at_overflow"], 2)
        self.assertGreaterEqual(snap["detections_at_overflow"], 6)          # 쓰기 장벽 동안에도 A 검출은 진행
        self.assertEqual(s["dispatcher"].get("accepted", 0), 2)             # 포화 뒤 수락 0
        self.assertLess(sc.detections, 12)                                  # 실패를 보고 실행을 멈춤
        self.assertTrue(s["cleanup_confirmed"])

    def test_bounded_capacity_without_barrier_is_clean(self):
        """쓰기가 따라오면 같은 상한에서도 누락 0·종료 0."""
        sc = Scenario(6, masks={i: (lambda: [rect_mask()]) for i in range(2, 6)})
        code, err, out = self.run_cli(sc, config=self.config_with(3))
        self.assertEqual(code, 0, err)
        q = self.summary(out)["skip_queue"]
        self.assertEqual((q["dropped"], q["unwritten"], q["total"], q["written"]), (0, 0, 5, 5))


class D13AndArgsTest(Base):
    """D13 기본 거부·인자 조합·저장 경로 회귀."""

    def test_default_ros_link_has_no_b(self):
        """옵션 없는 --ros-link(AI)는 B를 부르지 않는다(D13 그대로)."""
        sc = Scenario(2)
        made = []
        err = io.StringIO()
        code = self.mod.main(["--ros-link", "--detector", "ai", "--ai-config", self.case.ai,
                              "--log", str(Path(self.tmp.name) / "plain.jsonl"), "--max-frames", "2"],
                             stderr=err, ros_available=lambda: True,
                             model_factory=lambda path: model_with(sc.masks_fn),
                             live_source_factory=lambda sid, rel: sc.factory_maker(made)(sid, rel))
        self.assertEqual(code, 0, err.getvalue())
        self.assertEqual(self.consume_calls, [])
        self.assertIsNone(made[0].slot._boundary_hook)

    def test_runner_still_rejects_live_b_consumer(self):
        """실행기: 실시간 + b_consumer는 b_consumer_not_allowed, 저장 입력 + 연결부는 live_observation_contract."""
        from test_live_observation_harness import D13RegressionTest

        D13RegressionTest("test_run_still_rejects_live_observation_consumer").test_run_still_rejects_live_observation_consumer()

        from eyetoarm.vision.pipeline_records import JsonlRecorder, PipelineLog
        from test_pipeline import settings

        res = pipeline_runner.run(pipeline_runner.stored_source_factory("images", self.case.images, "stored0"),
                                  settings(), PipelineLog(JsonlRecorder(io.StringIO())), stderr=io.StringIO(),
                                  live_observation_sink=object())
        self.assertEqual((res.exit_code, res.stop_reason), (1, "live_observation_contract"))

    def test_argument_errors(self):
        """조합 오류는 2: 짝 없음·--ros-link 없음·opencv·추적·status 아님·출력 존재."""
        base = ["--detector", "ai", "--ai-config", self.case.ai, "--log", str(Path(self.tmp.name) / "x.jsonl")]
        bad_cfg = Path(self.tmp.name) / "bad.json"
        doc = json.loads(CONFIG.read_text(encoding="utf-8"))
        doc["status"] = "approved"
        bad_cfg.write_text(json.dumps(doc), encoding="utf-8")
        no_cap = Path(self.tmp.name) / "no_cap.json"
        doc2 = json.loads(CONFIG.read_text(encoding="utf-8"))
        del doc2["skip_queue_capacity"]
        no_cap.write_text(json.dumps(doc2), encoding="utf-8")
        exists = Path(self.tmp.name) / "exists"
        exists.mkdir()
        cases = [
            ["--ros-link", *base, "--live-observation-out", self.out()],
            ["--images", *self.case.images, *base, "--live-observation-out", self.out(),
             "--live-observation-config", str(CONFIG)],
            ["--ros-link", "--detector", "opencv", "--detect-config", str(ROOT / "config/vision/opencv_planar.json"),
             "--log", str(Path(self.tmp.name) / "y.jsonl"), "--live-observation-out", self.out(),
             "--live-observation-config", str(CONFIG)],
            # #215: --tracking은 이제 허용(단계 T). 단계 L과 함께면 거부
            ["--ros-link", *base, "--tracking", "--b-input", "list", "--live-observation-out", self.out(),
             "--live-observation-config", str(CONFIG)],
            ["--ros-link", *base, "--live-observation-out", self.out(), "--live-observation-config", str(bad_cfg)],
            ["--ros-link", *base, "--live-observation-out", self.out(), "--live-observation-config", str(no_cap)],
            ["--ros-link", *base, "--live-observation-out", str(exists), "--live-observation-config", str(CONFIG)],
            ["--ros-link", "--detector", "ai", "--ai-config", self.case.ai_off, "--log", str(Path(self.tmp.name) / "z.jsonl"),
             "--live-observation-out", self.out(), "--live-observation-config", str(CONFIG)],
        ]

        for argv in cases:
            with self.subTest(argv=argv):
                err = io.StringIO()

                try:
                    code = self.mod.main(argv, stderr=err, ros_available=lambda: True,
                                         live_tf_factory=lambda cfg: self.fail("tf must not start"),
                                         live_run_controller=self.controller)
                except SystemExit as exc:
                    code = exc.code

                self.assertEqual(code, 2, err.getvalue())

        self.assertIsNone(self.controller.current)

    def test_example_config_is_test_only(self):
        """예시 설정은 status=test_only이고 읽힌다(TF 상한 0.05 s 시험값)."""
        from eyetoarm.vision.live_observation_link import load_live_observation_config

        cfg = load_live_observation_config(str(CONFIG))
        self.assertEqual((cfg.tf_wait_limit_s, cfg.world_frame), (0.05, "world"))


if __name__ == "__main__":
    unittest.main()
