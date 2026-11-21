"""실행기 모델 준비 단계·AI 작업 분류·AI 기록·CLI 시험(가짜 모델·모듈, 실제 ultralytics 없음).

준비 상한 기본값 10 s는 첫 검출 10 s·이후 1 s와 따로이며, 준비는 첫 검출 기회를 쓰지 않는다.
관측은 항상 valid=False다.
"""
import dataclasses
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import cv2
import numpy as np

from eyetoarm.common.types import InvalidationKind
from eyetoarm.vision.ai_detector import AiSegDetector
from eyetoarm.vision.ai_result import AiJobConfigError, ModelPrepareError
from eyetoarm.vision.pipeline_records import JsonlRecorder, PipelineLog
from eyetoarm.vision.pipeline_runner import RunnerConfig, run
from test_ai_config import AI_FIELDS, ai_config
from test_ai_detector import FakeModel, result_for
from test_ai_pipeline import ai_observation
from test_pipeline import RUN_ID, FakeClock, frame, observation, profile, settings
from test_pipeline_terminal import valid_observation
from test_pipeline_watchdog import WORKER_NAME, WatchClock, detector_threads, watch
from test_run_pipeline import SCRIPT, FakeSource, lines

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FW, FH = 200, 120
OFF = dict(first_detect_timeout_s=None, detect_timeout_s=None)


class CountingSource(FakeSource):
    """read 호출 수를 세는 가짜 소스."""

    reads = 0

    def read(self):
        """호출 수를 센 뒤 다음 항목."""
        self.reads += 1
        return super().read()


def ai_frames(n, holder):
    """n개 200×120 프레임 소스 팩토리(holder["source"]에 소스)."""
    def factory(issuer, provider):
        """issuer·세대 공급자를 받아 소스를 만든다."""
        items = [frame(issuer.next_seq(), provider(), image=np.zeros((FH, FW, 3), np.uint8), width=FW, height=FH)
                 for _ in range(n)]
        holder["source"] = CountingSource(items)
        return holder["source"]

    return factory


def kinds(stream):
    """기록 종류 목록."""
    return [r["kind"] for r in lines(stream)]


def events(stream):
    """사건 (kind, job_seq, image_seq, cause) 목록."""
    return [(r["invalidation_kind"], r["job_seq"], r["image_seq"], r["cause"]) for r in lines(stream)
            if r["kind"] == "event"]


class Base(unittest.TestCase):
    """가짜 가중치 파일·AI detector·실행 도우미."""

    def setUp(self):
        """가짜 가중치 파일을 만든다."""
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.weights = os.path.join(self.dir.name, "best.pt")
        with open(self.weights, "wb") as f:
            f.write(b"fake weights for runner tests")
        import hashlib
        with open(self.weights, "rb") as f:
            self.sha = hashlib.sha256(f.read()).hexdigest()

    def cfg(self, **over):
        """가짜 가중치를 가리키는 AI 설정(cpu, min_mask_area 0)."""
        fields = {"weights_path": self.weights, "weights_sha256": self.sha, "device": "cpu", "min_mask_area_px": 0}
        fields.update(over)
        return ai_config(**fields)

    def run_ai(self, det, cfg, n=2, config=None, prepare="det", stream=None, log_geometry=False, **kw):
        """AI 설정으로 실행한다. (결과, 스트림, stderr, holder)."""
        holder = {}
        stream = stream if stream is not None else io.StringIO()
        err = io.StringIO()
        log = PipelineLog(JsonlRecorder(stream), log_geometry=log_geometry)
        res = run(ai_frames(n, holder), settings(detector_config=cfg, profile=profile(width=FW, height=FH)), log,
                  run_id=RUN_ID, detector=det, prepare=det.prepare if prepare == "det" else prepare,
                  config=config or watch(1.0, 1.0, grace=0.5), stderr=err, clock_ns=FakeClock(), **kw)
        return res, stream, err, holder

    def assert_balance(self, res):
        """종료 후 보존식."""
        c = res.counters
        self.assertEqual(c.received, c.first_started + sum(c.queue_dropped.values()))
        self.assertEqual(c.first_started + c.reprocess_started,
                         c.completed + c.generation_discarded + c.failed + c.aborted)


class PrepareStageTest(Base):
    """준비 단계 결과별 실행기 동작."""

    def test_prepare_success_writes_model_ready_then_frames(self):
        """준비 성공 → run_meta·model_ready 뒤 프레임 처리, 준비와 검출은 같은 worker 스레드."""
        model = FakeModel(make=lambda s: result_for(s, [(10, 5, 29, 24)]))
        cfg = self.cfg()
        det = AiSegDetector(cfg, model_factory=lambda p: model, environ={})

        res, stream, err, holder = self.run_ai(det, cfg)

        self.assertEqual((res.exit_code, res.stop_reason), (0, "end_of_stream"), err.getvalue())
        self.assertEqual(kinds(stream)[:2], ["run_meta", "model_ready"])
        ready = lines(stream)[1]
        self.assertEqual((ready["model_id"], ready["model_source"], ready["versions"], ready["device"]),
                         (cfg.model_id, "injected_factory", {}, "cpu"))
        self.assertGreaterEqual(ready["model_load_ms"], 0.0)
        self.assertEqual(kinds(stream).count("process"), 2)
        self.assertEqual(len({c[3] for c in model.calls}), 1)   # predict는 한 스레드(준비한 worker)
        self.assertNotEqual(model.calls[0][3], threading.get_ident())
        self.assertEqual(detector_threads(), [])

    def test_prepare_missing_weights_event_once_no_read_exit_1(self):
        """가중치 없음 → MODEL_LOAD_FAILED 사건 1건, read·process 0, 소스 close, 종료 1."""
        cfg = self.cfg(weights_path=os.path.join(self.dir.name, "missing.pt"))
        det = AiSegDetector(cfg, model_factory=lambda p: self.fail("factory called"), environ={})

        res, stream, err, holder = self.run_ai(det, cfg)

        self.assertEqual((res.exit_code, res.stop_reason), (1, "model_load_failed"))
        ev = events(stream)
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0][:3], ("MODEL_LOAD_FAILED", None, None))
        self.assertIn("missing.pt", ev[0][3])
        self.assertEqual((holder["source"].reads, holder["source"].closes), (0, 1))
        self.assertNotIn("process", kinds(stream))
        self.assertNotIn("model_ready", kinds(stream))
        self.assertIn("missing.pt", err.getvalue())

    def test_prepare_unreadable_weights_event_once(self):
        """디렉터리 경로(정규 파일 아님) → MODEL_LOAD_FAILED 1건·read 0·종료 1."""
        cfg = self.cfg(weights_path=self.dir.name)
        res, stream, _, holder = self.run_ai(AiSegDetector(cfg, model_factory=lambda p: None, environ={}), cfg)

        self.assertEqual([e[0] for e in events(stream)], ["MODEL_LOAD_FAILED"])
        self.assertEqual((res.exit_code, holder["source"].reads), (1, 0))

    def test_prepare_hash_mismatch_config_invalid_event_once(self):
        """가중치 sha256 불일치 → MODEL_CONFIG_INVALID 사건 1건·read 0·종료 1."""
        cfg = self.cfg(weights_sha256="0" * 64)
        res, stream, _, holder = self.run_ai(AiSegDetector(cfg, model_factory=lambda p: None, environ={}), cfg)

        ev = events(stream)
        self.assertEqual([e[:3] for e in ev], [("MODEL_CONFIG_INVALID", None, None)])
        self.assertIn("sha256 mismatch", ev[0][3])
        self.assertEqual((res.exit_code, res.stop_reason, holder["source"].reads), (1, "model_config_invalid", 0))

    def test_prepare_unclassified_exception_is_load_failed(self):
        """분류되지 않은 준비 예외는 MODEL_LOAD_FAILED(원인 「예외형: 메시지」)다. 감시 켬·끔 모두."""
        def prepare():
            """분류 없는 실패."""
            raise RuntimeError("driver exploded")

        for config in (watch(1.0, 1.0), RunnerConfig(close_retry_interval_s=0.0, **OFF)):
            with self.subTest(watchdog=config.watchdog):
                cfg = self.cfg()
                res, stream, _, holder = self.run_ai(AiSegDetector(cfg, environ={}), cfg, config=config,
                                                     prepare=prepare)
                self.assertEqual([(e[0], e[3]) for e in events(stream)],
                                 [("MODEL_LOAD_FAILED", "RuntimeError: driver exploded")])
                self.assertEqual((res.exit_code, res.stop_reason, holder["source"].reads), (1, "model_load_failed", 0))

    def test_prepare_timeout_then_late_completion_discarded(self):
        """준비가 상한을 넘기면 MODEL_LOAD_FAILED(prepare_timeout), 늦은 결과는 버리고 종료 1."""
        clock = WatchClock()
        calls = []

        def prepare():
            """시계를 11 s 돌리고 조금 뒤 반환."""
            clock.advance(11.0)
            time.sleep(0.15)
            return {"ok": True}

        def detector(job):
            """불리면 안 된다."""
            calls.append(job.job_seq)
            return ai_observation(job), None

        res, stream, _, holder = self._run_custom(prepare, detector, self.cfg(),
                                                  RunnerConfig(close_retry_interval_s=0.0, cancel_grace_s=1.0), clock)

        ev = events(stream)
        self.assertEqual([e[0] for e in ev], ["MODEL_LOAD_FAILED"])
        self.assertTrue(ev[0][3].startswith("prepare_timeout"))
        self.assertEqual((res.exit_code, res.stop_reason, res.stalled), (1, "model_load_timeout", False))
        self.assertEqual((holder["source"].reads, calls, res.late_results), (0, [], 1))
        self.assertNotIn("model_ready", kinds(stream))

    def _run_custom(self, prepare, detector, cfg, config, clock=None, should_stop=None, n=2):
        """임의 prepare·detector로 실행한다."""
        holder = {}
        stream, err = io.StringIO(), io.StringIO()
        kw = {"watch_clock": clock} if clock is not None else {}
        res = run(ai_frames(n, holder), settings(detector_config=cfg, profile=profile(width=FW, height=FH)),
                  PipelineLog(JsonlRecorder(stream)), run_id=RUN_ID, detector=detector, prepare=prepare,
                  config=config, stderr=err, clock_ns=FakeClock(), should_stop=should_stop, **kw)
        self.last_err = err
        return res, stream, err, holder

    def test_stop_during_prepare_exit_0_no_event(self):
        """준비 중 정지 요청 → 사건 없음·read 0·준비 반환 확인 뒤 stop_requested·종료 0."""
        state = {"started": False}

        def prepare():
            """시작 표시 후 0.1 s 뒤 반환."""
            state["started"] = True
            time.sleep(0.1)
            return {}

        res, stream, _, holder = self._run_custom(prepare, lambda job: self.fail("detector"), self.cfg(),
                                                  watch(1.0, 1.0, grace=1.0), should_stop=lambda: state["started"])

        self.assertEqual((res.exit_code, res.stop_reason, events(stream)), (0, "stop_requested", []))
        self.assertEqual((holder["source"].reads, res.late_results), (0, 1))

    def test_first_limit_applies_to_first_detection_not_prepare(self):
        """준비 9 s·첫 검출 9 s는 통과, 둘째 검출 2 s는 detect_timeout(준비는 첫 검출 기회와 별개)."""
        clock = WatchClock()
        plan = {0: 9.0, 1: 2.0}

        def prepare():
            """준비에 9 s."""
            clock.advance(9.0)
            return {}

        def detector(job, token):
            """작업별로 시계를 돌린다. 첫째는 0.2 s 더 머물러 제어 흐름이 기한을 여러 번 판정하게 하고, 둘째는 취소 표지를 기다린다."""
            clock.advance(plan[job.job_seq])
            if job.job_seq == 0:
                time.sleep(0.2)
            if job.job_seq == 1 and token.wait(2.0):
                from eyetoarm.vision.pipeline_runner import JobCancelled
                raise JobCancelled("token")
            return ai_observation(job), None

        holder = {}
        stream = io.StringIO()
        cfg = self.cfg()
        res = run(ai_frames(2, holder), settings(detector_config=cfg, profile=profile(width=FW, height=FH)),
                  PipelineLog(JsonlRecorder(stream)), run_id=RUN_ID, detector=detector, detector_takes_token=True,
                  prepare=prepare, config=RunnerConfig(close_retry_interval_s=0.0), stderr=io.StringIO(),
                  clock_ns=FakeClock(), watch_clock=clock)

        self.assertEqual((RunnerConfig().model_load_timeout_s, RunnerConfig().first_detect_timeout_s,
                          RunnerConfig().detect_timeout_s), (10.0, 10.0, 1.0))
        self.assertEqual([r["status"] for r in lines(stream) if r["kind"] == "process"], ["COMPLETED", "ABORTED"])
        self.assertEqual([(e[0], e[1], e[3]) for e in events(stream)], [("JOB_ABORTED", 1, "detect_timeout")])
        self.assertEqual(res.timeouts, 1)

    def test_worker_busy_during_prepare_no_read_no_job(self):
        """준비가 끝나기 전에는 프레임을 읽지 않고 detector도 부르지 않으며, 준비는 worker 스레드에서 돈다."""
        seen = {}
        holder_ref = {}

        def prepare():
            """준비 시점의 read 수·스레드를 기록."""
            seen["reads"] = holder_ref["h"]["source"].reads
            seen["thread"] = threading.current_thread().name
            time.sleep(0.05)
            return {}

        holder = {}
        holder_ref["h"] = holder
        stream = io.StringIO()
        cfg = self.cfg()
        res = run(ai_frames(1, holder), settings(detector_config=cfg, profile=profile(width=FW, height=FH)),
                  PipelineLog(JsonlRecorder(stream)), run_id=RUN_ID, detector=lambda job: (ai_observation(job), None),
                  prepare=prepare, config=watch(1.0, 1.0), stderr=io.StringIO(), clock_ns=FakeClock())

        self.assertEqual((seen["reads"], seen["thread"], res.exit_code), (0, WORKER_NAME, 0))

    def test_prepare_sync_when_watchdog_off(self):
        """감시 끔이면 준비를 제어 흐름에서 동기로 부르고 같은 분류를 쓴다(성공 시 model_ready, 실패 시 사건)."""
        cfg = self.cfg()
        model = FakeModel()
        det = AiSegDetector(cfg, model_factory=lambda p: model, environ={})
        res, stream, _, _ = self.run_ai(det, cfg, config=RunnerConfig(close_retry_interval_s=0.0, **OFF))

        self.assertEqual((res.exit_code, kinds(stream)[1]), (0, "model_ready"))
        self.assertEqual(model.calls[0][3], threading.get_ident())


class AiJobClassificationTest(Base):
    """작업 중 AI 오류 분류."""

    def test_job_config_error_is_model_config_invalid_and_continues(self):
        """AiJobConfigError는 그 작업만 MODEL_CONFIG_INVALID, ValueError는 INFERENCE_ERROR, 종료 1."""
        def detector(job):
            """작업 0: 설정 오류, 1: 출력 오류, 2: 정상."""
            if job.job_seq == 0:
                raise AiJobConfigError("roi does not fit frame")
            if job.job_seq == 1:
                raise ValueError("mask data must be finite")
            return ai_observation(job), None

        for config in (watch(1.0, 1.0), RunnerConfig(close_retry_interval_s=0.0, **OFF)):
            with self.subTest(watchdog=config.watchdog):
                res, stream, _, _ = self._run(detector, config)
                self.assertEqual([(e[0], e[1]) for e in events(stream)],
                                 [("MODEL_CONFIG_INVALID", 0), ("INFERENCE_ERROR", 1)])
                self.assertIn("roi does not fit frame", events(stream)[0][3])
                self.assertEqual((res.counters.failed, res.counters.completed, res.exit_code), (2, 1, 1))
                self.assert_balance(res)

    def _run(self, detector, config, n=3, b_consumer=None):
        """준비 없이 AI 설정으로 실행한다."""
        holder = {}
        stream = io.StringIO()
        cfg = self.cfg()
        res = run(ai_frames(n, holder), settings(detector_config=cfg, profile=profile(width=FW, height=FH)),
                  PipelineLog(JsonlRecorder(stream)), run_id=RUN_ID, detector=detector, config=config,
                  stderr=io.StringIO(), clock_ns=FakeClock(), b_consumer=b_consumer)
        return res, stream, io.StringIO(), holder

    def test_valid_then_no_detection_does_not_reuse(self):
        """시험용 유효 관측(실제 AI 유효 관측 아님) 뒤 무검출이면 B 지점은 유효 1회뿐이고 이전 유효 결과를 재사용하지 않는다."""
        calls = []

        def detector(job):
            """0: 시험용 유효(method ai), 1: 무효."""
            if job.job_seq == 0:
                return dataclasses.replace(valid_observation(job), method="ai", model_id="m"), None
            return ai_observation(job, reason="NO_OBJECT", candidate_count=0), None

        res, stream, _, _ = self._run(detector, watch(1.0, 1.0), n=2,
                                      b_consumer=lambda f, o: calls.append((f.image_seq, o.valid)))

        self.assertEqual(calls, [(0, True)])
        self.assertEqual([r["valid"] for r in lines(stream) if r["kind"] == "process"], [True, False])


class AiRecordsTest(Base):
    """필수 AI 기록·선택 geometry·run_meta·fail-stop."""

    def test_ai_process_has_instance_classes_scores_by_default(self):
        """옵션 없이도 process에 method·model_id·infer_ms·ai_instances가 그 작업 식별자로 남는다."""
        model = FakeModel(make=lambda s: result_for(s, [(10, 5, 29, 24), (50, 50, 69, 69)], cls=[0.0, 1.0],
                                                    conf=[0.9, 0.6]))
        cfg = self.cfg()
        res, stream, _, _ = self.run_ai(AiSegDetector(cfg, model_factory=lambda p: model, environ={}), cfg, n=1)
        proc = [r for r in lines(stream) if r["kind"] == "process"][0]

        self.assertEqual((proc["method"], proc["model_id"], proc["job_seq"], proc["image_seq"], proc["generation"],
                          proc["detector_config_id"], proc["run_id"], proc["source_id"]),
                         ("ai", cfg.model_id, 0, 0, 0, cfg.config_sha256, RUN_ID, "src0"))
        self.assertGreaterEqual(proc["infer_ms"], 0.0)
        self.assertEqual([(i["index"], i["class_name"], round(i["confidence"], 2), i["rejected"], i["clipped"],
                           i["area_px"]) for i in proc["ai_instances"]],
                         [(0, "bolt", 0.9, None, False, 400.0), (1, "nut", 0.6, None, False, 400.0)])
        self.assertEqual((proc["valid"], proc["reason"]), (False, "TOP_FACE_UNRESOLVED"))
        self.assertNotIn("ai_geometry", kinds(stream))

    def test_process_ai_fields_default_for_opencv(self):
        """OpenCV 작업 process는 method "opencv"이고 model_id·infer_ms·ai_instances는 None이다."""
        stream = io.StringIO()
        holder = {}
        run(ai_frames(1, holder), settings(), PipelineLog(JsonlRecorder(stream)), run_id=RUN_ID,
            detector=lambda job: (observation(job), None), config=watch(1.0, 1.0), stderr=io.StringIO(),
            clock_ns=FakeClock())
        proc = [r for r in lines(stream) if r["kind"] == "process"][0]
        meta = lines(stream)[0]

        self.assertEqual((proc["method"], proc["model_id"], proc["infer_ms"], proc["ai_instances"]),
                         ("opencv", None, None, None))
        self.assertEqual(meta["detector"], {"method": "opencv", "config_id": settings().detector_config_id})

    def test_ai_instances_not_rewritten_on_duplicate_late_or_error(self):
        """오류 작업의 ai_instances는 None, 시간 초과 뒤 늦은 결과는 process를 더 쓰지 않는다(작업마다 process 1줄)."""
        def make(source):
            """느린 결과(0.3 s)."""
            time.sleep(0.3)
            return result_for(source, [(10, 5, 29, 24)])

        cfg = self.cfg()
        res, stream, _, _ = self.run_ai(AiSegDetector(cfg, model_factory=lambda p: FakeModel(make=make),
                                                      environ={}), cfg, n=1, config=watch(0.05, 0.05, grace=1.0))
        procs = [r for r in lines(stream) if r["kind"] == "process"]

        self.assertEqual((len(procs), procs[0]["status"], procs[0]["ai_instances"], procs[0]["method"]),
                         (1, "ABORTED", None, "ai"))
        self.assertEqual(res.late_results, 1)

    def test_geometry_only_with_option(self):
        """log_geometry=True일 때만 process 다음에 ai_geometry(작업 식별자·박스·다각형·조각/구멍 수)가 1줄 있다."""
        model = FakeModel(make=lambda s: result_for(s, [(10, 5, 29, 24)]))
        cfg = self.cfg()
        _, stream, _, _ = self.run_ai(AiSegDetector(cfg, model_factory=lambda p: model, environ={}), cfg, n=1,
                                      log_geometry=True)
        recs = lines(stream)
        i = kinds(stream).index("process")

        self.assertEqual(recs[i + 1]["kind"], "ai_geometry")
        geo = recs[i + 1]
        self.assertEqual((geo["job_seq"], geo["image_seq"], geo["detector_config_id"]), (0, 0, cfg.config_sha256))
        inst = geo["instances"][0]
        self.assertEqual((inst["box_xyxy"], inst["component_count"], inst["hole_count"]),
                         ([10.0, 5.0, 29.0, 24.0], 1, 0))
        self.assertTrue({(10, 5), (29, 24)} <= {(int(x), int(y)) for x, y in inst["polygon"]})

    def test_run_meta_detector_block_for_ai(self):
        """AI run_meta의 detector 묶음에 방식·설정 ID·모델·해시·임계값 등이 있고 format_version 1."""
        cfg = self.cfg()
        _, stream, _, _ = self.run_ai(AiSegDetector(cfg, model_factory=lambda p: FakeModel(), environ={}), cfg, n=1)
        d = lines(stream)[0]["detector"]

        self.assertEqual((d["method"], d["config_id"], d["model_id"], d["weights_sha256"], d["class_map"],
                          d["target_classes"], d["conf_threshold"], d["imgsz"], d["device"], d["retina_masks"]),
                         ("ai", cfg.config_sha256, cfg.model_id, self.sha, {"0": "bolt", "1": "nut"},
                          ["bolt", "nut"], 0.25, 1280, "cpu", True))
        self.assertEqual({r["format_version"] for r in lines(stream)}, {1})

    def test_model_ready_write_failure_is_fail_stop(self):
        """model_ready 쓰기가 실패하면 기록을 멈추고 프레임을 읽지 않으며 종료 1(log_failure)·미확인 보고."""
        class BreakSecond(io.StringIO):
            """두 번째 write부터 실패."""

            n = 0

            def write(self, t):
                """두 번째부터 OSError."""
                self.n += 1
                if self.n >= 2:
                    raise OSError("disk full")
                return super().write(t)

        cfg = self.cfg()
        res, _, err, holder = self.run_ai(AiSegDetector(cfg, model_factory=lambda p: FakeModel(), environ={}), cfg,
                                          stream=BreakSecond())

        self.assertEqual((res.exit_code, res.stop_reason, holder["source"].reads), (1, "log_failure", 0))
        self.assertIn("log write failed", err.getvalue())


class PrepareFailureWithLogFailureTest(Base):
    """준비 실패와 사건 기록 실패가 겹쳐도 원래 준비 실패 원인이 결과·stderr에 남는다."""

    class BreakSecond(io.StringIO):
        """두 번째 write(=준비 실패 사건 줄)부터 실패."""

        n = 0

        def write(self, t):
            """두 번째부터 OSError."""
            self.n += 1
            if self.n >= 2:
                raise OSError("disk full")
            return super().write(t)

    def test_prepare_failure_cause_survives_event_write_failure(self):
        """준비 실패 세 종류 각각에서 사건 쓰기가 실패해도 원래 원인이 대표 정지 사유로 남는다."""
        clock = WatchClock()

        def slow_prepare():
            """주입 시계를 11 s 돌리고 조금 뒤 반환(시간 초과)."""
            clock.advance(11.0)
            time.sleep(0.15)
            return {}

        missing = self.cfg(weights_path=os.path.join(self.dir.name, "missing.pt"))
        bad_hash = self.cfg(weights_sha256="0" * 64)
        cases = [
            ("load", missing, AiSegDetector(missing, model_factory=lambda p: None, environ={}).prepare, None,
             "model_load_failed", "missing.pt"),
            ("config", bad_hash, AiSegDetector(bad_hash, model_factory=lambda p: None, environ={}).prepare, None,
             "model_config_invalid", "sha256 mismatch"),
            ("timeout", self.cfg(), slow_prepare, clock, "model_load_timeout", "prepare_timeout"),
        ]

        for name, cfg, prepare, wclock, reason, cause in cases:
            with self.subTest(case=name):
                holder = {}
                err = io.StringIO()
                kw = {"watch_clock": wclock} if wclock is not None else {}
                res = run(ai_frames(2, holder), settings(detector_config=cfg, profile=profile(width=FW, height=FH)),
                          PipelineLog(JsonlRecorder(self.BreakSecond())), run_id=RUN_ID,
                          detector=lambda job: self.fail("detector called"), prepare=prepare,
                          config=RunnerConfig(close_retry_interval_s=0.0, cancel_grace_s=1.0), stderr=err,
                          clock_ns=FakeClock(), **kw)
                joined = "\n".join(res.errors)

                self.assertEqual((res.exit_code, res.stop_reason), (1, reason), joined)
                self.assertIn(cause, joined)
                self.assertIn("log write failed", joined)
                self.assertIn(cause, err.getvalue())
                self.assertIn("log write failed", err.getvalue())
                self.assertEqual((holder["source"].reads, holder["source"].closes, res.closed), (0, 1, True))
                self.assertEqual(res.missing["event"], 1)
                c = res.counters
                self.assertEqual((c.first_started, c.failed, c.aborted, c.completed), (0, 0, 0, 0))


class AiCliTest(Base):
    """CLI 방식 선택·충돌·설정·준비 실패."""

    def load_cli(self):
        """scripts/run_pipeline.py를 모듈로 불러온다."""
        spec = importlib.util.spec_from_file_location("run_pipeline_ai_test", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def write_inputs(self, conf=0.4):
        """이미지 1장·AI 설정 JSON·OpenCV 설정 JSON을 쓴다."""
        img = os.path.join(self.dir.name, "a.png")
        cv2.imwrite(img, np.zeros((FH, FW, 3), np.uint8))
        ai = os.path.join(self.dir.name, "ai.json")
        with open(ai, "w") as f:
            json.dump({**AI_FIELDS, "weights_path": self.weights, "weights_sha256": self.sha, "device": "cpu",
                       "conf_threshold": conf}, f)
        planar = os.path.join(ROOT, "config", "vision", "opencv_planar.json")
        return img, ai, planar

    def test_cli_ai_with_fake_model_uses_config_conf(self):
        """--detector ai --ai-config 실행은 model_ready·AI process를 남기고 conf 0.4로 종료 0."""
        mod = self.load_cli()
        img, ai, _ = self.write_inputs(conf=0.4)
        out = os.path.join(self.dir.name, "run.jsonl")
        model = FakeModel()
        err = io.StringIO()

        code = mod.main(["--images", img, "--detector", "ai", "--ai-config", ai, "--log", out], stderr=err,
                        model_factory=lambda p: model)

        self.assertEqual(code, 0, err.getvalue())
        with open(out) as f:
            recs = [json.loads(t) for t in f]
        self.assertEqual([r["kind"] for r in recs][:3], ["run_meta", "model_ready", "process"])
        self.assertEqual(model.calls[0][2]["conf"], 0.4)
        self.assertIn("detector=ai", err.getvalue())
        self.assertIn("model_id=", err.getvalue())

    def test_cli_detector_choice_and_conflicts(self):
        """방식·설정 충돌, 준비 상한 0, 잘못된 AI 설정 파일은 종료 2."""
        mod = self.load_cli()
        img, ai, planar = self.write_inputs()
        bad_ai = os.path.join(self.dir.name, "bad.json")
        with open(bad_ai, "w") as f:
            f.write('{"format_version": 1}')
        out = os.path.join(self.dir.name, "o.jsonl")
        base = ["--images", img, "--log", out]
        cases = [
            ["--detector", "ai"],
            ["--detector", "ai", "--ai-config", ai, "--detect-config", planar],
            [],
            ["--detect-config", planar, "--ai-config", ai],
            ["--detector", "ai", "--ai-config", ai, "--model-load-timeout", "0"],
        ]

        for extra in cases:
            with self.subTest(extra=extra):
                with self.assertRaises(SystemExit) as ctx:
                    mod.main(base + extra, stderr=io.StringIO())
                self.assertEqual(ctx.exception.code, 2)

        self.assertEqual(mod.main(base + ["--detector", "ai", "--ai-config", bad_ai], stderr=io.StringIO()), 2)
        self.assertEqual(mod.main(base + ["--detector", "ai", "--ai-config", os.path.join(self.dir.name, "x.json")],
                                  stderr=io.StringIO()), 2)
        cfg = mod.build_runner_config(mod.build_parser().parse_args(
            base + ["--detector", "ai", "--ai-config", ai, "--model-load-timeout", "3.5"]))
        self.assertEqual(cfg.model_load_timeout_s, 3.5)

    def test_cli_ai_missing_dependency_is_exit_1_with_event(self):
        """ultralytics import 실패는 준비 실패: 종료 1, MODEL_LOAD_FAILED 사건 1, process 0."""
        mod = self.load_cli()
        img, ai, _ = self.write_inputs()
        out = os.path.join(self.dir.name, "run.jsonl")

        def no_module(name):
            """import 실패."""
            raise ImportError(f"No module named {name!r}")

        code = mod.main(["--images", img, "--detector", "ai", "--ai-config", ai, "--log", out],
                        stderr=io.StringIO(), import_module=no_module)

        self.assertEqual(code, 1)
        with open(out) as f:
            recs = [json.loads(t) for t in f]
        ev = [r for r in recs if r["kind"] == "event"]
        self.assertEqual([(e["invalidation_kind"], e["job_seq"]) for e in ev], [("MODEL_LOAD_FAILED", None)])
        self.assertIn("import ultralytics failed", ev[0]["cause"])
        self.assertNotIn("process", [r["kind"] for r in recs])

    def test_cli_opencv_run_imports_no_torch(self):
        """OpenCV 방식 CLI를 실제로 끝까지 돌려도 torch·ultralytics가 로드되지 않는다."""
        img, _, planar = self.write_inputs()
        out = os.path.join(self.dir.name, "run.jsonl")
        code = ("import sys, importlib.util, io\n"
                f"spec = importlib.util.spec_from_file_location('rp', {SCRIPT!r})\n"
                "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
                f"rc = m.main(['--images', {img!r}, '--detect-config', {planar!r}, '--log', {out!r}],"
                " stderr=io.StringIO())\n"
                "bad = [k for k in sys.modules if k.split('.')[0] in ('torch', 'ultralytics')]\n"
                "print(rc, ','.join(sorted(bad)) or 'none')")
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                             env={"PYTHONPATH": os.path.join(ROOT, "src"), "PATH": ""}, cwd=ROOT)

        self.assertEqual(res.stdout.strip(), "0 none", res.stderr[-2000:])


class PrepareContractTest(unittest.TestCase):
    """준비 설정 검증."""

    def test_model_load_timeout_validation(self):
        """model_load_timeout_s는 유한 양수(기본 10.0, A 임시 개발값)이고 0·음수·inf·nan·bool·None은 ValueError."""
        self.assertEqual(RunnerConfig().model_load_timeout_s, 10.0)

        for bad in (0, -1.0, float("inf"), float("nan"), True, None, "10"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    RunnerConfig(model_load_timeout_s=bad)

    def test_prepare_error_kind_mapping(self):
        """ModelPrepareError는 MODEL_LOAD_FAILED·MODEL_CONFIG_INVALID만 만든다(그 밖은 생성 거부)."""
        self.assertIs(ModelPrepareError("MODEL_LOAD_FAILED", "x").kind, InvalidationKind.MODEL_LOAD_FAILED)


if __name__ == "__main__":
    unittest.main()
