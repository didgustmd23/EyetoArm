"""#171 R7 결합 하네스: `LiveObservationState` + `LiveObservationDispatcher` + 실시간 봉투 + B 입구 어댑터 + 순번 기록기를 가짜 입력·TF·B로 직접 구동한다.

근거: `_workspace/171_builder2_live_a_to_b_plan.md` r3 §3·§5.1 R7·§8.1·§8.2(T-a~T-f, T-h~T-j, T-n, T-q).
- `run()`·`scripts/run_pipeline.py`를 거치지 않는다(진입점 우회 경로 없음). D13 회귀는 마지막 시험에서 `run()`이 실시간 + b_consumer를
  여전히 거부하는지 확인한다.
- 경합은 worker 훅에서 장벽(Event)으로 순서를 강제한다. 용량·상한 값은 결정 K4·L-3·L-8 전 시험 후보다.
- 실제 BStage 결과 한 건(`RealBStageTest`)은 합성 카메라·자세로 B 계산·주석이 끝까지 도는지만 본다(정확도 무관).
"""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from eyetoarm.calibration.profile import from_fov
from eyetoarm.transforms.observation_envelope_adapter import ObservationEnvelopeAdapter, EnvelopeTfLookup
from eyetoarm.vision.live_observation_dispatch import LiveObservationDispatcher
from eyetoarm.vision.live_observation_envelope import LiveCaptureTfLookup, make_live_envelope
from eyetoarm.vision.live_observation_events import LiveObservationEventWriter
from eyetoarm.vision.live_observation_state import LiveObservationConfig, LiveObservationState
from test_observation_envelope import CAM, T, make_obs
from test_pipeline import frame as base_frame

STAMP0 = 4_291_000_000
WAIT = 2.0   # 시험 대기 상한(초)


def make_frame(seq, cam=CAM, **over):
    """seq번째 실시간 프레임(카메라와 맞는 보정 번호·광학 프레임·stamp)."""
    fields = dict(image=np.zeros((cam.height, cam.width, 3), np.uint8), width=cam.width, height=cam.height,
                  calibration_id=cam.calibration_id, optical_frame=cam.frame_id, ros_stamp_ns=STAMP0 + seq * 66_000_000,
                  received_monotonic_ns=time.monotonic_ns())
    fields.update(over)
    return base_frame(seq, **fields)


class FakeStage:
    """B 단계 대역: tf_lookup은 어댑터의 EnvelopeTfLookup, consume은 호출과 받은 TF를 기록한다."""

    def __init__(self, camera, harness):
        self.camera = camera
        self.tf_lookup = EnvelopeTfLookup()
        self.h = harness
        self.calls = []

    def consume(self, frame, obs):
        self.h.check_unlocked("compute")
        T_W_C = self.tf_lookup(frame)
        self.calls.append((frame.image_seq, None if T_W_C is None else T_W_C.copy()))
        self.h.compute_calls.append(frame.image_seq)
        return type("Out", (), {"goal": {"kind": "SKIP" if T_W_C is None else "PICK_PLACE",
                                         "reason": "TF_UNAVAILABLE" if T_W_C is None else None},
                                "result": type("R", (), {"valid": T_W_C is not None})(), "T_W_C": T_W_C})()


class Harness:
    """하네스 조립(가짜 TF 조회·가짜 B·실제 상태·봉투·어댑터·기록기)."""

    def __init__(self, *, w=8, r=16, tf_wait_s=0.0, hooks=None, writer=True, tmp=None):
        self.state = LiveObservationState(run_id=make_frame(0).run_id, generation=0, calibration_id=CAM.calibration_id,
                                config=LiveObservationConfig(write_capacity=w, reserved_capacity=r))
        self.violations = []
        self.compute_calls = []
        self.stages = []
        self.tf_wait_s = tf_wait_s
        self.clock_s = [0.0]
        self.tf = LiveCaptureTfLookup(self.fake_lookup, world_frame="world", wait_limit_s=0.05,
                                      clock=lambda: self.clock_s[0])
        self.disp = LiveObservationDispatcher(self.state, camera=CAM, stage_factory=self.stage_factory, tf_lookup=self.tf_call,
                                    build_envelope=self.build, compute=self.compute, render=self.render,
                                    hooks=hooks or {})
        self.writer = None

        if writer:
            self.tmp = tmp or tempfile.mkdtemp()
            self.writer = LiveObservationEventWriter(self.state, str(Path(self.tmp) / "out"), lag_warn_s=2.0)

    def check_unlocked(self, where):
        """T-e: 잠금을 쥔 채 무거운 일을 하면 위반으로 적는다."""
        if self.state.lock_held():
            self.violations.append(where)

    def stage_factory(self, camera):
        st = FakeStage(camera, self)
        self.stages.append(st)
        return (st, ObservationEnvelopeAdapter(st, st.tf_lookup))

    def fake_lookup(self, parent, child, stamp_ns, timeout_s):
        self.check_unlocked("tf")
        self.clock_s[0] += self.tf_wait_s
        return T, stamp_ns

    def tf_call(self, frame):
        return self.tf(frame)

    def build(self, frame, obs, camera, tf_result):
        self.check_unlocked("envelope")
        return make_live_envelope(frame, obs, camera=camera, tf_result=tf_result, world_frame="world")

    def compute(self, stage, envelope):
        return stage[1].consume(envelope).output

    def render(self, stage, frame, out):
        self.check_unlocked("render")
        return b"\x89PNG-fake" if out.T_W_C is not None else None

    def submit(self, seq, **over):
        f = make_frame(seq, **over)
        t = self.state.token
        return self.disp.submit(f, make_obs(f), rx_epoch=over.pop("rx_epoch", t.rx_epoch), generation=t.generation,
                                calibration_id=f.calibration_id)

    def wait_counts(self, pred):
        end = time.monotonic() + WAIT
        while time.monotonic() < end:
            if pred(self.disp.counts):
                return True
            time.sleep(0.005)
        return False

    def finish(self):
        self.disp.close(WAIT)
        self.state.run_end("test_end")
        report = None
        if self.writer is not None:
            self.writer.start()
            report = self.writer.close(WAIT)
        return report

    def records(self):
        path = Path(self.writer.dir) / "b_live_events.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]


def gate():
    """worker를 세우는 장벽: (훅, 도착 Event, 풀기 Event)."""
    arrived, release = threading.Event(), threading.Event()

    def hook():
        arrived.set()
        release.wait(WAIT)

    return hook, arrived, release


class NormalPathTest(unittest.TestCase):
    def test_accept_written_in_order_with_k5_and_tf(self):
        """정상: 제출 → TF → 봉투 → B → 수락 → 순번 기록. K5·TF 대기가 따로 기록되고 주석 PNG가 참조된다(T-i)."""
        h = Harness()
        h.disp.start()
        self.assertTrue(h.submit(1))
        self.assertTrue(h.wait_counts(lambda c: c.get("accepted") == 1))
        rep = h.finish()
        self.assertEqual((rep.pending, rep.unwritten, rep.timed_out, rep.error), (0, 0, False, None))
        recs = h.records()
        # 실행 종료도 무효화이므로 현재 수락 결과를 철회한다(계획 §3.3)
        self.assertEqual([r["kind"] for r in recs], ["accept", "revoke", "invalidation", "run_end"])
        acc = recs[0]
        self.assertEqual((acc["tf_status"], acc["goal_kind"], acc["input"]["image_seq"]), ("ok", "PICK_PLACE", 1))
        self.assertIsNotNone(acc["k5_ms"])
        self.assertIn("tf_wait_ms", acc)
        self.assertTrue((Path(h.writer.dir) / acc["annotation"]).exists())
        self.assertEqual(acc["b_input"]["tf"]["stamp_ns"], make_frame(1).ros_stamp_ns)
        self.assertEqual(h.violations, [])

    def test_h_late_tf_not_used(self):
        """T-h: TF 반환 60 ms(성공) → late, B는 TF 없이(SKIP TF_UNAVAILABLE) 계산, 주석 없음. 최신 대체 0."""
        h = Harness(tf_wait_s=0.06)
        h.disp.start()
        h.submit(1)
        self.assertTrue(h.wait_counts(lambda c: c.get("accepted") == 1))
        h.finish()
        acc = h.records()[0]
        self.assertEqual((acc["tf_status"], acc["goal_kind"], acc["annotation"]), ("late", "SKIP", None))
        self.assertIsNone(h.stages[0].calls[0][1])


class InvalidationBoundaryTest(unittest.TestCase):
    def run_with_gate(self, point):
        """point 훅에서 worker를 세우고 경계를 넣은 뒤 푼다."""
        hook, arrived, release = gate()
        h = Harness(hooks={point: hook})
        h.disp.start()
        h.submit(1)
        self.assertTrue(arrived.wait(WAIT))
        h.state.on_rx_boundary(1, "reconnect")
        release.set()
        self.assertTrue(h.wait_counts(lambda c: any(k in c for k in ("stale_before_b", "stale_after_tf",
                                                                     "not_accepted", "accepted"))))
        h.finish()
        return h

    def test_c_n1_invalid_before_begin_no_b_call(self):
        """T-c·T-n(i): TF 대기 뒤·② 전 무효화 → BStage 호출 0, 기록에 accept 없음."""
        h = self.run_with_gate("after_tf")
        self.assertEqual(h.compute_calls, [])
        self.assertEqual(h.disp.counts.get("stale_after_tf"), 1)
        self.assertNotIn("accept", [r["kind"] for r in h.records()])

    def test_after_take_invalid_no_tf_no_b(self):
        """① 전 무효화 → TF·B 모두 없음."""
        h = self.run_with_gate("after_take")
        self.assertEqual((h.compute_calls, h.disp.counts.get("stale_before_b")), ([], 1))

    def test_n2_invalid_after_begin_before_entry_output_zero(self):
        """T-n(ii): ② 전이 직후·BStage 진입 전 무효화 → 계산 중 분류, BStage는 불릴 수 있지만 수락 0."""
        h = self.run_with_gate("after_begin")
        self.assertEqual(h.compute_calls, [1])
        self.assertEqual(h.disp.counts.get("not_accepted"), 1)
        self.assertEqual(h.state.counts.get("stale_after_compute"), 1)
        self.assertNotIn("accept", [r["kind"] for r in h.records()])

    def test_d_n3_invalid_during_compute_output_zero(self):
        """T-d·T-n(iii): B 계산 중 무효화 → 수락 0."""
        h = self.run_with_gate("in_compute")
        self.assertEqual((h.compute_calls, h.disp.counts.get("not_accepted")), ([1], 1))

    def test_a_invalid_just_before_accept(self):
        """T-a: ③ 잠금 직전 무효화 → 수락 거부. 무효화 뒤 '현재'로 쓰인 기록 0."""
        h = self.run_with_gate("before_accept")
        kinds = [r["kind"] for r in h.records()]
        self.assertNotIn("accept", kinds)

    def test_a_b_accept_then_invalidate_revokes(self):
        """T-a·T-b: 수락 뒤 무효화 → 같은 순번열에서 뒤에 revoke, is_current 거짓."""
        h = Harness()
        h.disp.start()
        h.submit(1)
        self.assertTrue(h.wait_counts(lambda c: c.get("accepted") == 1))
        h.state.on_rx_boundary(1, "reconnect")
        h.finish()
        recs = h.records()
        # 경계 무효화(철회 포함) 뒤 종료 무효화가 한 번 더 남는다. 종료 때는 현재 수락이 없어 철회가 없다
        self.assertEqual([r["kind"] for r in recs], ["accept", "revoke", "invalidation", "invalidation", "run_end"])
        self.assertEqual(recs[1]["revoked_event_seq"], recs[0]["event_seq"])
        self.assertFalse(h.state.is_current_accept(recs[0]["event_seq"]))
        self.assertEqual([r["event_seq"] for r in recs], sorted(r["event_seq"] for r in recs))


class SlotAndProfileTest(unittest.TestCase):
    def test_f_new_frame_supersedes_pending_only(self):
        """T-f: 계산 중 새 프레임 2장 → 대기 칸만 교체(superseded 1), 진행 중 작업과 최신 1장은 수락."""
        hook, arrived, release = gate()
        h = Harness(hooks={"in_compute": hook})
        h.disp.start()
        h.submit(1)
        self.assertTrue(arrived.wait(WAIT))
        h.submit(2)
        h.submit(3)
        h.disp._hooks.clear()
        release.set()
        self.assertTrue(h.wait_counts(lambda c: c.get("accepted") == 2))
        h.finish()
        self.assertEqual(h.disp.counts.get("superseded"), 1)
        self.assertEqual([r["input"]["image_seq"] for r in h.records() if r["kind"] == "accept"], [1, 3])

    def test_j_profile_change_blocks_old_and_uses_new_camera(self):
        """T-j: 보정 변경 → 이전 보정 항목 제출 거부·대기 항목 ①에서 버림, 새 카메라로 새 B 단계."""
        hook, arrived, release = gate()
        h = Harness(hooks={"after_take": hook})
        h.disp.start()
        h.submit(1)
        self.assertTrue(arrived.wait(WAIT))
        cam2 = from_fov(CAM.width, CAM.height, 1.2, CAM.frame_id)
        h.state.on_rx_boundary(1, "profile_changed")
        h.disp.set_camera(cam2)
        h.state.on_applied(1, 1, cam2.calibration_id)
        h.disp._hooks.clear()
        release.set()
        self.assertTrue(h.wait_counts(lambda c: c.get("stale_before_b") == 1))
        f_old = make_frame(2)
        self.assertFalse(h.disp.submit(f_old, make_obs(f_old), rx_epoch=1, generation=1,
                                       calibration_id=f_old.calibration_id))
        f_new = make_frame(3, cam=cam2)
        self.assertTrue(h.disp.submit(f_new, make_obs(f_new), rx_epoch=1, generation=1,
                                      calibration_id=cam2.calibration_id))
        self.assertTrue(h.wait_counts(lambda c: c.get("accepted") == 1))
        h.finish()
        self.assertEqual([s.camera.calibration_id for s in h.stages], [CAM.calibration_id, cam2.calibration_id])
        self.assertEqual(h.compute_calls, [3])

    def test_rewind_not_ready_until_both(self):
        """되감기: 제어 반영만으로는 제출 거부, TF 준비까지 끝나야 제출·수락."""
        h = Harness()
        h.disp.start()
        h.state.on_rx_boundary(1, "sim_time_rewind")
        h.state.on_applied(1, 1, CAM.calibration_id)
        f = make_frame(1)
        self.assertFalse(h.disp.submit(f, make_obs(f), rx_epoch=1, generation=1, calibration_id=f.calibration_id))
        h.state.on_tf_ready(1, h.state.begin_tf_attempt(1))
        self.assertTrue(h.disp.submit(f, make_obs(f), rx_epoch=1, generation=1, calibration_id=f.calibration_id))
        self.assertTrue(h.wait_counts(lambda c: c.get("accepted") == 1))
        h.finish()
        self.assertEqual(h.compute_calls, [1])


class CapacityAndCloseTest(unittest.TestCase):
    def test_q_backpressure_and_close_drains(self):
        """T-q: 쓰기 담당이 멈춘 동안 W=1 → 두 번째 수락은 거부(출력 0). 종료 뒤 기록기가 모두 비운다."""
        h = Harness(w=1)
        h.disp.start()
        h.submit(1)
        self.assertTrue(h.wait_counts(lambda c: c.get("accepted") == 1))
        h.submit(2)
        self.assertTrue(h.wait_counts(lambda c: c.get("not_accepted") == 1))
        self.assertEqual(h.state.counts.get("writer_backpressure"), 1)
        rep = h.finish()
        self.assertEqual((rep.pending, rep.unwritten, rep.timed_out, rep.error), (0, 0, False, None))
        self.assertEqual([r["input"]["image_seq"] for r in h.records() if r["kind"] == "accept"], [1])

    def test_q_reserved_overflow_fail_stop_in_harness(self):
        """T-q: 예약 칸 넘침 → fail-stop, 이후 제출 거부, 기록 마지막은 failed."""
        h = Harness(r=1)
        h.disp.start()
        h.state.on_rx_boundary(1, "reconnect")   # invalidation(1칸)
        h.state.on_rx_boundary(2, "reconnect")   # 넘침
        self.assertEqual(h.state.failed, "b_writer_overflow")
        self.assertFalse(h.submit(1))
        h.finish()
        self.assertEqual(h.records()[-1]["kind"], "failed")

    def test_close_drops_pending_and_stops(self):
        """종료: 대기 항목은 버리고 worker는 상한 안에 멈춘다. 종료 뒤 제출 거부."""
        hook, arrived, release = gate()
        h = Harness(hooks={"after_take": hook})
        h.disp.start()
        h.submit(1)
        self.assertTrue(arrived.wait(WAIT))
        h.submit(2)
        h.state.run_end("stop")
        release.set()
        self.assertTrue(h.disp.close(WAIT))
        self.assertEqual(h.disp.counts.get("dropped_on_close"), 1)
        self.assertFalse(h.submit(3))
        h.writer.start()
        self.assertEqual(h.writer.close(WAIT).pending, 0)
        self.assertNotIn("accept", [r["kind"] for r in h.records()])

    def test_e_lock_discipline(self):
        """T-e: 여러 건 처리 동안 잠금 안 TF·봉투·B·렌더 호출 0."""
        h = Harness()
        h.disp.start()
        for i in range(1, 6):
            h.submit(i)
            h.wait_counts(lambda c, i=i: sum(c.get(k, 0) for k in ("accepted", "not_accepted")) >= 1)
        h.finish()
        self.assertEqual(h.violations, [])


class SubmitLatencyTest(unittest.TestCase):
    def test_submit_does_not_wait_for_worker(self):
        """제어 흐름 submit은 worker가 계산 중(장벽에서 정지)이어도 기다리지 않는다: 200회 중앙값 < 1 ms(시험 후보 기준)."""
        hook, arrived, release = gate()
        h = Harness(hooks={"in_compute": hook})
        h.disp.start()
        h.submit(0)
        self.assertTrue(arrived.wait(WAIT))
        durations = []
        for i in range(1, 201):
            f = make_frame(i)
            obs = make_obs(f)
            t0 = time.perf_counter()
            h.disp.submit(f, obs, rx_epoch=0, generation=0, calibration_id=f.calibration_id)
            durations.append(time.perf_counter() - t0)
        release.set()
        h.finish()
        durations.sort()
        self.assertLess(durations[len(durations) // 2], 0.001)
        self.assertEqual(h.disp.counts.get("superseded"), 199)


class RealBStageTest(unittest.TestCase):
    def test_real_bstage_annotation(self):
        """실제 BStage·render_center로 수락 → 같은 BOutput으로 주석 PNG(합성 값, 정확도 무관)."""
        import cv2
        from eyetoarm.transforms.b_stage import BStage
        root = Path(__file__).resolve().parents[2]
        cam = from_fov(640, 480, 1.0, "link_cam_optical_frame")
        state = LiveObservationState(run_id=make_frame(0).run_id, generation=0, calibration_id=cam.calibration_id,
                           config=LiveObservationConfig(write_capacity=4, reserved_capacity=8))

        def stage_factory(camera):
            lk = EnvelopeTfLookup()
            stage = BStage.from_config(root, camera, lk)
            return stage, ObservationEnvelopeAdapter(stage, lk)

        def render(stage, frame, out):
            if out.T_W_C is None:
                return None
            img = stage[0].render_center(frame.image, [out])
            return cv2.imencode(".png", img)[1].tobytes()

        tf = LiveCaptureTfLookup(lambda p, c, s, t: (T, s), world_frame="world", wait_limit_s=0.05)
        disp = LiveObservationDispatcher(state, camera=cam, stage_factory=stage_factory, tf_lookup=tf,
                               build_envelope=lambda f, o, c, r: make_live_envelope(f, o, camera=c, tf_result=r,
                                                                                    world_frame="world"),
                               compute=lambda st, env: st[1].consume(env).output, render=render)
        tmp = tempfile.mkdtemp()
        writer = LiveObservationEventWriter(state, str(Path(tmp) / "out"), lag_warn_s=2.0)
        disp.start()
        f = make_frame(1, cam=cam, image=np.zeros((480, 640, 3), np.uint8))
        obs = make_obs(f, axis_endpoints_px=[[300.0, 240.0], [360.0, 240.0]])
        self.assertTrue(disp.submit(f, obs, rx_epoch=0, generation=0, calibration_id=cam.calibration_id))
        end = time.monotonic() + WAIT
        while time.monotonic() < end and not disp.counts.get("accepted") and not disp.errors:
            time.sleep(0.005)
        disp.close(WAIT)
        state.run_end("end")
        writer.start()
        writer.close(WAIT)
        self.assertEqual(disp.errors, [])
        recs = [json.loads(x) for x in (Path(tmp) / "out" / "b_live_events.jsonl").read_text().splitlines()]
        acc = [r for r in recs if r["kind"] == "accept"]
        self.assertEqual(len(acc), 1)
        self.assertIn(acc[0]["goal_kind"], ("PICK_PLACE", "SKIP"))
        if acc[0]["annotation"] is not None:
            png = (Path(tmp) / "out" / acc[0]["annotation"]).read_bytes()
            self.assertEqual(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR).shape, (480, 640, 3))


class D13RegressionTest(unittest.TestCase):
    def test_run_still_rejects_live_observation_consumer(self):
        """D13 유지: 실시간 소스 + b_consumer(디스패처 제출로 감싸도)는 run()이 b_consumer_not_allowed로 거부, 제출 0."""
        from test_pipeline_runner_live import Live
        from test_ros_image_adapter import image_msg
        h = Harness(writer=False)
        submitted = []

        def consumer(frame, obs):
            submitted.append(frame.image_seq)
            h.disp.submit(frame, obs, rx_epoch=0, generation=0, calibration_id=frame.calibration_id)

        live = Live(feed=lambda s, c: s.handle_image(image_msg(1)), b_consumer=consumer)
        res, _recs, _err = live.run()
        self.assertEqual((res.exit_code, res.stop_reason), (1, "b_consumer_not_allowed"))
        self.assertEqual((submitted, live.detect_calls), ([], []))


if __name__ == "__main__":
    unittest.main()
