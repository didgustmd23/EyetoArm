"""추론 프로세스 분리(`ProcessAiSegDetector`·`ai_process_child`) 시험. 가짜 모델을 쓰는 실제 AiSegDetector를 자식에서
돌린다(GPU·torch 없음). 자식 동작은 영상 픽셀 값으로 고른다(`ai_process_fake`).
"""
import dataclasses
import enum
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import numpy as np

from eyetoarm.common.types import InvalidationKind
from eyetoarm.vision.ai_config import load_ai_config
from eyetoarm.vision.ai_process import (SHM_DIR, SHM_PREFIX, InferJobError, InferProcessError,
                                        ProcessAiSegDetector, sweep_orphan_slots)
from eyetoarm.vision.ai_result import AiJobConfigError, ModelPrepareError
from eyetoarm.vision.pipeline_records import JsonlRecorder, PipelineLog
from eyetoarm.vision.pipeline_runner import CancelToken, JobCancelled, run
from eyetoarm.vision.pipeline import GenerationCause, VisionPipeline
from eyetoarm.vision.pipeline_types import JobStatus

import ai_process_fake as fake
from test_ai_config import AI_FIELDS
from test_pipeline import RUN_ID, FakeClock, frame, profile, settings
from test_pipeline_watchdog import watch
from test_run_pipeline import FakeSource, lines

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
W, H = 200, 120
CAL = "cal-infer-process"
TIMING = {"infer_ms", "post_ms", "process_ms"}
LOAD, CONFIG = InvalidationKind.MODEL_LOAD_FAILED, InvalidationKind.MODEL_CONFIG_INVALID


def canon(value):
    """비교용 정규형: dataclass·배열·열거형을 풀고 시간 필드(infer/post/process ms)는 뺀다."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: canon(getattr(value, f.name)) for f in dataclasses.fields(value) if f.name not in TIMING}

    if isinstance(value, np.ndarray):
        return ("ndarray", str(value.dtype), value.shape, value.tolist())

    if isinstance(value, (list, tuple)):
        return [canon(v) for v in value]

    if isinstance(value, dict):
        return {k: canon(v) for k, v in value.items() if k not in TIMING}

    if isinstance(value, enum.Enum):
        return value.value

    return value


def image(code=fake.NORMAL, pos=0, w=W, h=H):
    """동작 코드(B)·볼트 위치(G)를 왼쪽 위 픽셀에 넣은 BGR 영상."""
    img = np.zeros((h, w, 3), np.uint8)
    img[0, 0, 0], img[0, 0, 1] = code, pos

    return img


def no_sweep():
    """남은 슬롯 정리 대역: 아무것도 지우지 않는다."""
    return {"removed": [], "kept": {}}


def shm_entries():
    """이 프로세스(부모 pid 이름)의 /dev/shm 슬롯. 동시에 도는 다른 pytest의 슬롯은 세지 않는다."""
    own = f"{SHM_PREFIX}{os.getpid()}_"
    return sorted(n for n in os.listdir(SHM_DIR) if n.startswith(own))


class Base(unittest.TestCase):
    """가짜 가중치·설정 파일·작업 도우미·대리 객체 정리."""

    def setUp(self):
        """임시 가중치·AI 설정 파일(M2 끝점 켬, cpu)과 정리 목록."""
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.shm_before = shm_entries()
        weights = os.path.join(self.dir.name, "best.pt")

        with open(weights, "wb") as f:
            f.write(b"fake weights for infer process tests")

        with open(weights, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()

        self.cfg_path = self.write_cfg(weights_path=weights, weights_sha256=sha)
        self.cfg = load_ai_config(self.cfg_path)
        self.weights, self.sha = weights, sha
        self.proxies = []
        self.addCleanup(self.close_all)

    def write_cfg(self, name="ai.json", **over):
        """AI_FIELDS + 덮어쓸 값으로 설정 파일을 쓴다(cpu, 면적 하한 0, M2)."""
        doc = dict(AI_FIELDS, device="cpu", min_mask_area_px=0, axis_extractor="m2")
        doc.update(over)
        path = os.path.join(self.dir.name, name)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f)

        return path

    def env(self, **extra):
        """자식이 src와 시험 폴더(가짜 공장)를 import하도록 PYTHONPATH를 준 환경."""
        env = {k: v for k, v in os.environ.items() if k not in ("YOLO_AUTOINSTALL", "YOLO_OFFLINE")}
        env["PYTHONPATH"] = os.pathsep.join([SRC, HERE, env.get("PYTHONPATH", "")])
        env.update(extra)

        return env

    def proxy(self, factory="ai_process_fake:make", cfg=None, cfg_path=None, env=None, **kw):
        """대리 객체(시험 종료 때 닫음)."""
        kw.setdefault("sweep", no_sweep)   # 실제 /dev/shm의 남은 파일을 시험이 지우지 않게 한다
        p = ProcessAiSegDetector(cfg or self.cfg, cfg_path or self.cfg_path, factory=factory,
                                 env=env if env is not None else self.env(), **kw)
        self.proxies.append(p)

        return p

    def close_all(self):
        """남은 대리 객체를 닫고 자식·슬롯이 남지 않았는지 확인한다."""
        for p in self.proxies:
            p.close()

            if p._proc is not None:
                self.assertIsNotNone(p._proc.poll(), "child left running")

        self.assertEqual(shm_entries(), self.shm_before)

    def job(self, img, seq=0, cfg=None):
        """영상 하나로 작업을 시작한다(보정 프로파일 있음). (pipeline, job)."""
        fr = frame(seq, image=img, width=img.shape[1], height=img.shape[0], calibration_id=CAL)
        prof = profile(width=img.shape[1], height=img.shape[0], calibration_id=CAL)
        p = VisionPipeline(settings(detector_config=cfg or self.cfg, profile=prof), run_id=RUN_ID, clock_ns=FakeClock())
        p.offer_frame(fr)

        return p, p.start_next()


class RoundTripTest(Base):
    """정상 왕복·동등성·준비 정보."""

    def test_same_result_and_observation_as_in_process(self):
        """같은 영상이면 단일 프로세스 AiSegDetector와 같은 AiSegResult·관측(시간 필드 제외), 슬롯 교대·크기 변경 포함."""
        local = fake.make(self.cfg)
        local.prepare()
        px = self.proxy()
        px.prepare()

        for i, img in enumerate([image(pos=0), image(pos=17), image(pos=33), image(pos=5, w=240, h=150)]):
            with self.subTest(i=i):
                _, job = self.job(img, seq=i)
                obs_l, res_l = local(job)
                obs_p, res_p = px(job, CancelToken())

                self.assertEqual(canon(res_p), canon(res_l))
                self.assertEqual(canon(obs_p), canon(obs_l))
                self.assertIsNotNone(res_p.instances[0].axis_endpoints_px)

        self.assertEqual(px.summary()["results"], 4)

    def test_prepare_info_keys_and_child_env(self):
        """model_ready 정보는 단일 프로세스와 같은 키 + infer_process이고, 자식 env에는 자동 설치 방지 변수가 있다."""
        local = fake.make(self.cfg)
        keys = set(local.prepare())
        info = self.proxy(torch_threads=3).prepare()

        self.assertEqual(set(info) - {"infer_process"}, keys)
        self.assertEqual(info["env"], {"YOLO_AUTOINSTALL": "false", "YOLO_OFFLINE": "1"})
        ip = info["infer_process"]
        self.assertEqual((ip["factory"], ip["torch_threads_requested"], ip["torch_threads"]),
                         ("ai_process_fake:make", 3, None))
        self.assertGreater(ip["pid"], 0)

    def test_close_stops_child_and_removes_slots(self):
        """정상 종료: 자식 종료 코드 0·강제 없음·슬롯 파일 삭제, 두 번 닫아도 같다."""
        px = self.proxy()
        px.prepare()
        _, job = self.job(image())
        px(job, CancelToken())

        self.assertEqual(len(shm_entries()), len(self.shm_before) + 1)
        self.assertEqual(px.close(), {"returncode": 0, "forced": False})
        self.assertEqual(px.close(), {"returncode": 0, "forced": False})
        self.assertEqual(shm_entries(), self.shm_before)


class CloseInterruptTest(Base):
    """닫는 도중 KeyboardInterrupt(정리 중 두 번째 신호)가 나도 소켓·슬롯 파일 정리는 끝난다(D-9)."""

    def test_interrupt_while_waiting_child_still_removes_slots(self):
        """자식 대기에서 KeyboardInterrupt: 예외는 올라가지만 슬롯 파일이 지워지고, 자식은 연결 끊김으로 끝난다."""
        px = self.proxy()
        px.prepare()
        _, job = self.job(image())
        px(job, CancelToken())
        self.assertEqual(len(shm_entries()), len(self.shm_before) + 1)

        proc = px._proc
        real_wait = proc.wait

        def interrupted(timeout=None):
            """첫 대기에서 신호가 온 것처럼 KeyboardInterrupt."""
            proc.wait = real_wait
            raise KeyboardInterrupt

        proc.wait = interrupted

        with self.assertRaises(KeyboardInterrupt):
            px.close()

        self.assertEqual(shm_entries(), self.shm_before)
        self.assertEqual(real_wait(5), 0)   # 연결이 닫혀 자식이 스스로 끝남


    def test_prepare_records_sweep_and_survives_sweep_error(self):
        """준비 때 정리를 한 번 부르고 요약에 남긴다. 정리 함수 오류는 준비를 막지 않는다."""
        calls = []

        def broken():
            """정리 실패 대역."""
            calls.append(1)
            raise OSError("boom")

        px = self.proxy(sweep=broken)
        px.prepare()

        self.assertEqual(calls, [1])
        self.assertEqual(px.summary()["orphan_shm"], {"removed": [], "kept": {}, "error": "OSError: boom"})


class SweepOrphanSlotsTest(unittest.TestCase):
    """남은 슬롯 정리: 죽은 pid·아무도 안 연·자기 소유·정해진 이름 형식만 지운다(D-9 안전망)."""

    def setUp(self):
        """임시 폴더(실제 /dev/shm 대신)."""
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def touch(self, name):
        """빈 파일 하나."""
        path = os.path.join(self.dir.name, name)
        open(path, "wb").close()

        return path

    def test_removes_only_dead_unused_own_files(self):
        """죽은 pid만 지우고 살아 있는 pid·자기 pid·열린 파일·다른 형식 이름은 남긴다."""
        dead = self.touch(f"{SHM_PREFIX}111_0123456789ab")
        alive = self.touch(f"{SHM_PREFIX}222_0123456789ab")
        own = self.touch(f"{SHM_PREFIX}333_0123456789ab")
        used = self.touch(f"{SHM_PREFIX}444_0123456789ab")
        other_format = self.touch(f"{SHM_PREFIX}362repro_555_x")
        unrelated = self.touch("other_program_shm")

        res = sweep_orphan_slots(self.dir.name, own_pid=333, is_alive=lambda pid: pid == 222,
                                 open_paths=lambda: {used})

        self.assertEqual(res["removed"], [os.path.basename(dead)])
        self.assertEqual(res["kept"], {os.path.basename(alive): "pid_alive", os.path.basename(own): "own_pid",
                                       os.path.basename(used): "in_use"})
        self.assertEqual([os.path.exists(p) for p in (dead, alive, own, used, other_format, unrelated)],
                         [False, True, True, True, True, True])

    def test_other_users_files_are_kept(self):
        """다른 사용자 소유(여기서는 uid를 바꿔 흉내)는 pid가 죽었어도 지우지 않는다."""
        path = self.touch(f"{SHM_PREFIX}111_0123456789ab")

        res = sweep_orphan_slots(self.dir.name, uid=os.getuid() + 1, own_pid=1, is_alive=lambda pid: False,
                                 open_paths=lambda: set())

        self.assertEqual((res["removed"], res["kept"]), ([], {os.path.basename(path): "other_owner"}))
        self.assertTrue(os.path.exists(path))

    def test_real_open_and_mapped_files_are_seen(self):
        """기본 `open_paths`: 이 프로세스가 열어 둔 파일은 사용 중으로 보고 남긴다."""
        path = self.touch(f"{SHM_PREFIX}111_0123456789ab")

        with open(path, "rb"):
            res = sweep_orphan_slots(self.dir.name, own_pid=1, is_alive=lambda pid: False)

        self.assertEqual(res["kept"], {os.path.basename(path): "in_use"})
        self.assertTrue(os.path.exists(path))

    def test_zombie_pid_counts_as_dead(self):
        """끝났지만 아직 회수하지 않은 자식(좀비)의 파일도 지운다(부모가 회수하기 전에 다음 A가 시작한 경우)."""
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        self.addCleanup(child.wait, 10)
        deadline = time.monotonic() + 10

        while time.monotonic() < deadline and open(f"/proc/{child.pid}/stat").read().rsplit(")", 1)[1].split()[0] != "Z":
            time.sleep(0.02)

        path = self.touch(f"{SHM_PREFIX}{child.pid}_0123456789ab")

        res = sweep_orphan_slots(self.dir.name)

        self.assertEqual(res["removed"], [os.path.basename(path)])

    def test_real_dead_pid_file_is_removed_and_live_pid_kept(self):
        """기본 `is_alive`: 끝난 자식 pid의 파일은 지우고, 살아 있는 이 프로세스의 부모 pid 파일은 남긴다."""
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait(10)
        dead = self.touch(f"{SHM_PREFIX}{child.pid}_0123456789ab")
        alive = self.touch(f"{SHM_PREFIX}{os.getppid()}_0123456789ab")

        res = sweep_orphan_slots(self.dir.name)

        self.assertEqual(res["removed"], [os.path.basename(dead)])
        self.assertEqual(res["kept"], {os.path.basename(alive): "pid_alive"})

class PrepareFailureTest(Base):
    """준비 실패 분류."""

    def test_unsafe_env_refused_without_starting_child(self):
        """YOLO_AUTOINSTALL=true면 자식을 띄우지 않고 MODEL_LOAD_FAILED."""
        px = self.proxy(env=self.env(YOLO_AUTOINSTALL="true"))

        with self.assertRaises(ModelPrepareError) as ctx:
            px.prepare()

        self.assertEqual(ctx.exception.kind, LOAD)
        self.assertIsNone(px.pid)

    def test_config_file_mismatch_is_config_invalid(self):
        """부모 설정과 자식이 읽은 설정 sha256이 다르면 MODEL_CONFIG_INVALID."""
        other = self.write_cfg("other.json", weights_path=self.weights, weights_sha256=self.sha, conf_threshold=0.5)
        px = self.proxy(cfg_path=other)

        with self.assertRaises(ModelPrepareError) as ctx:
            px.prepare()

        self.assertEqual(ctx.exception.kind, CONFIG)
        self.assertIn("!= parent", ctx.exception.cause)

    def test_child_prepare_error_keeps_kind(self):
        """자식 detector의 ModelPrepareError 분류·원인을 그대로 올리고 실패 상태가 된다."""
        px = self.proxy(factory="ai_process_fake:make_bad_prepare")

        with self.assertRaises(ModelPrepareError) as ctx:
            px.prepare()

        self.assertEqual((ctx.exception.kind, ctx.exception.cause), (CONFIG, "fake model names mismatch"))
        self.assertIsNotNone(px.failed)

    def test_child_death_in_prepare_is_load_failed(self):
        """준비 중 자식이 죽으면 MODEL_LOAD_FAILED(종료 코드 포함)."""
        px = self.proxy(factory="ai_process_fake:make_die_prepare")

        with self.assertRaises(ModelPrepareError) as ctx:
            px.prepare()

        self.assertEqual(ctx.exception.kind, LOAD)
        self.assertIn("returncode=9", ctx.exception.cause)


class JobFailureTest(Base):
    """작업 오류·취소·늦은 응답·죽음·멈춤."""

    def test_job_config_mismatch_raises_ai_job_config_error(self):
        """작업 시작 설정이 자식 detector 설정과 다르면 AiJobConfigError(→ MODEL_CONFIG_INVALID), 자식은 계속 산다."""
        px = self.proxy()
        px.prepare()
        other = load_ai_config(self.write_cfg("o.json", weights_path=self.weights, weights_sha256=self.sha, max_det=5))
        _, bad = self.job(image(), cfg=other)

        with self.assertRaises(AiJobConfigError):
            px(bad, CancelToken())

        _, good = self.job(image(), seq=1)
        self.assertEqual(px(good, CancelToken())[0].job_seq, good.job_seq)
        self.assertIsNone(px.failed)

    def test_job_exception_is_inference_error_and_child_survives(self):
        """자식 predict 예외는 InferJobError(원래 종류 포함)이고 다음 작업은 정상이다."""
        px = self.proxy()
        px.prepare()
        _, job = self.job(image(fake.RAISE))

        with self.assertRaises(InferJobError) as ctx:
            px(job, CancelToken())

        self.assertIn("ValueError: fake predict failure", str(ctx.exception))
        _, job2 = self.job(image(), seq=1)
        px(job2, CancelToken())
        self.assertEqual((px.failed, px.summary()["job_errors"]), (None, 1))

    def test_cancel_returns_within_grace_and_late_reply_is_dropped(self):
        """취소하면 짧게 JobCancelled, 다음 요청은 이전 요청의 늦은 응답을 버리고 자기 결과를 받는다."""
        px = self.proxy(env=self.env(FAKE_DELAY_S="0.6"))
        px.prepare()
        _, slow = self.job(image(fake.DELAY, pos=3))
        token = CancelToken()
        threading.Timer(0.1, token._cancel).start()
        t0 = time.monotonic()

        with self.assertRaises(JobCancelled):
            px(slow, token)

        self.assertLess(time.monotonic() - t0, 0.35)
        _, nxt = self.job(image(pos=40), seq=1)
        obs, res = px(nxt, CancelToken())

        self.assertEqual(obs.image_seq, nxt.frame.image_seq)
        local = fake.make(self.cfg)
        local.prepare()
        self.assertEqual(canon(res), canon(local(nxt)[1]))
        self.assertEqual((px.summary()["late_replies"], px.summary()["cancelled"], px.failed), (1, 1, None))

    def test_child_death_during_job_fails_and_refuses_later_calls(self):
        """작업 중 자식이 죽으면 InferProcessError(infer_process_died), 이후 호출은 바로 거부."""
        px = self.proxy()
        px.prepare()
        _, job = self.job(image(fake.DIE))

        with self.assertRaises(InferProcessError) as ctx:
            px(job, CancelToken())

        self.assertIn("infer_process_died", str(ctx.exception))
        self.assertIn("returncode=7", px.failed)
        _, job2 = self.job(image(), seq=1)

        with self.assertRaises(InferProcessError):
            px(job2, CancelToken())

    def test_unanswered_cancelled_request_becomes_stall(self):
        """취소한 요청이 stall_after_s 넘게 답이 없으면 멈춤으로 보고 자식을 끝낸다."""
        px = self.proxy(stall_after_s=0.4)
        px.prepare()
        _, hang = self.job(image(fake.HANG))
        token = CancelToken()
        token._cancel()

        with self.assertRaises(JobCancelled):
            px(hang, token)

        time.sleep(0.5)
        _, nxt = self.job(image(), seq=1)

        with self.assertRaises(InferProcessError) as ctx:
            px(nxt, CancelToken())

        self.assertIn("infer_process_stalled", str(ctx.exception))
        self.assertIsNotNone(px._proc.poll())

    def test_close_while_waiting_is_not_failure(self):
        """다른 스레드가 기다리는 중에 close()하면 대기는 InferProcessError로 끝나지만 실패 상태로 남지 않는다."""
        px = self.proxy(stop_grace_s=0.3)
        px.prepare()
        _, hang = self.job(image(fake.HANG))
        caught = []

        def call():
            """멈춘 작업을 기다린다."""
            try:
                px(hang, CancelToken())
            except InferProcessError as exc:
                caught.append(exc)

        t = threading.Thread(target=call)
        t.start()
        time.sleep(0.2)
        px.close()
        t.join(3)

        self.assertEqual(len(caught), 1)
        self.assertIsNone(px.failed)

    def test_close_forces_hung_child(self):
        """멈춘 자식은 종료 요청 유예 뒤 SIGTERM으로 끝내고 슬롯을 지운다."""
        px = self.proxy(stop_grace_s=0.3)
        px.prepare()
        _, hang = self.job(image(fake.HANG))
        token = CancelToken()
        token._cancel()

        with self.assertRaises(JobCancelled):
            px(hang, token)

        info = px.close()
        self.assertEqual((info["forced"], info["returncode"]), (True, -signal.SIGTERM))
        self.assertEqual(shm_entries(), self.shm_before)


class StallWindowTest(Base):
    """멈춤 판정은 취소한 요청에만 걸리고, 첫 검출 창(실행기 한도)의 긴 처리와 겹치지 않는다."""

    def test_slow_first_detection_without_cancel_is_not_stall(self):
        """취소 없이 stall_after_s보다 오래 걸린 첫 검출은 정상 결과다(멈춤 한도는 실행기 첫 검출 창을 대신하지 않음)."""
        px = self.proxy(stall_after_s=0.2, env=self.env(FAKE_DELAY_S="0.6"))
        px.prepare()
        _, slow = self.job(image(fake.DELAY))

        obs, _ = px(slow, CancelToken())

        self.assertEqual(obs.job_seq, slow.job_seq)
        self.assertIsNone(px.failed)

    def test_one_cancel_answered_late_is_not_stall(self):
        """첫 요청 취소 뒤 그 응답이 한도 안에 늦게 오면 멈춤이 아니고 다음 요청은 정상이다."""
        px = self.proxy(stall_after_s=2.0, env=self.env(FAKE_DELAY_S="0.6"))
        px.prepare()
        _, slow = self.job(image(fake.DELAY))
        token = CancelToken()
        threading.Timer(0.1, token._cancel).start()

        with self.assertRaises(JobCancelled):
            px(slow, token)

        _, nxt = self.job(image(), seq=1)
        px(nxt, CancelToken())

        self.assertEqual((px.failed, px.summary()["late_replies"]), (None, 1))

    def test_normal_then_two_unanswered_cancels_is_stall(self):
        """정상 응답 뒤 연달아 취소한 요청이 답 없이 한도를 넘으면 멈춤이다."""
        px = self.proxy(stall_after_s=0.4)
        px.prepare()
        _, ok = self.job(image())
        px(ok, CancelToken())

        for seq in (1, 2):
            _, hang = self.job(image(fake.HANG), seq=seq)
            token = CancelToken()
            token._cancel()

            with self.assertRaises(JobCancelled):
                px(hang, token)

        time.sleep(0.5)
        _, nxt = self.job(image(), seq=3)

        with self.assertRaises(InferProcessError) as ctx:
            px(nxt, CancelToken())

        self.assertIn("infer_process_stalled", str(ctx.exception))


class GenerationTest(Base):
    """세대 폐기는 실행기·파이프라인 규칙 그대로."""

    def test_generation_change_during_job_discards_result(self):
        """작업 중 검출 설정이 바뀌어 세대가 오르면 대리 객체 결과도 DISCARDED_GENERATION이다."""
        px = self.proxy()
        px.prepare()
        pipe, job = self.job(image())
        new_cfg = load_ai_config(self.write_cfg("g.json", weights_path=self.weights, weights_sha256=self.sha,
                                                conf_threshold=0.1))
        prof = profile(width=W, height=H, calibration_id=CAL)
        pipe.change_settings(settings(detector_config=new_cfg, profile=prof),
                             cause=GenerationCause.DETECTOR_CONFIG_CHANGED)
        obs, res = px(job, CancelToken())

        self.assertEqual(pipe.complete(job, obs, display_result=res).status, JobStatus.DISCARDED_GENERATION)


class RunnerTest(Base):
    """실행기 run()에 끼운 통합(감시 켬)."""

    def frames(self, images):
        """주어진 영상들로 소스 팩토리."""
        def factory(issuer, provider):
            """issuer·세대 공급자를 받아 소스를 만든다."""
            items = [frame(issuer.next_seq(), provider(), image=img, width=W, height=H, calibration_id=CAL)
                     for img in images]
            return FakeSource(items)

        return factory

    def run_px(self, px, images, normal_s=0.3):
        """대리 객체로 실행한다(첫 검출 5 s·이후 normal_s 감시). (결과, 기록 목록)."""
        stream, err = io.StringIO(), io.StringIO()
        log = PipelineLog(JsonlRecorder(stream))
        res = run(self.frames(images), settings(detector_config=self.cfg, profile=profile(width=W, height=H,
                                                                                         calibration_id=CAL)),
                  log, run_id=RUN_ID, detector=px, prepare=px.prepare, detector_takes_token=True,
                  config=watch(5.0, normal_s, grace=0.3), stderr=err, clock_ns=FakeClock())

        return res, lines(stream)

    def test_runner_completes_and_aborts_timeout_with_late_reply(self):
        """정상 작업은 COMPLETED, 시간 초과 작업은 ABORTED(detect_timeout)이고 다음 작업은 늦은 응답을 버린 뒤 완료.

        실행기 규칙대로 시간 초과가 있으면 종료 1이다. 자식은 취소된 작업을 끝까지 처리하므로 다음 작업이 그만큼 기다린다.
        """
        px = self.proxy(env=self.env(FAKE_DELAY_S="1.3"))
        res, recs = self.run_px(px, [image(), image(fake.DELAY), image(pos=9)], normal_s=1.0)
        status = [r["status"] for r in recs if r["kind"] == "process"]
        causes = [r["cause"] for r in recs if r["kind"] == "event"]

        self.assertEqual((res.exit_code, res.timeouts), (1, 1))
        self.assertEqual(status, ["COMPLETED", "ABORTED", "COMPLETED"])
        self.assertIn("detect_timeout", causes)
        self.assertTrue(any(r["kind"] == "model_ready" and "infer_process" in r for r in recs), recs)
        self.assertEqual((px.summary()["late_replies"], px.failed), (1, None))

    def test_runner_marks_job_failed_when_child_dies(self):
        """자식이 죽으면 그 작업은 INFERENCE_ERROR 실패이고 대리 객체는 실패 상태다(실행 종료 처리는 run_pipeline 몫)."""
        px = self.proxy()
        res, recs = self.run_px(px, [image(), image(fake.DIE), image()])
        events = [r.get("invalidation_kind") for r in recs if r["kind"] == "event"]

        self.assertIn("INFERENCE_ERROR", events)
        self.assertIn("infer_process_died", px.failed)


class ThreadLifetimeTest(Base):
    """자식을 띄운 스레드가 끝나도 자식은 살아 있다(실행기 worker 스레드에서 준비하는 경우)."""

    def test_child_survives_preparing_thread_exit(self):
        """다른 스레드에서 prepare한 뒤 그 스레드가 끝나도 작업·정상 종료(종료 코드 0)가 된다."""
        px = self.proxy()
        t = threading.Thread(target=px.prepare)
        t.start()
        t.join(30)
        time.sleep(0.3)

        self.assertIsNone(px._proc.poll())
        _, job = self.job(image())
        px(job, CancelToken())
        self.assertEqual(px.close(), {"returncode": 0, "forced": False})


class OrphanTest(Base):
    """부모가 강제로 죽어도 자식이 남지 않고, 부모는 torch를 import하지 않는다."""

    HELPER = r"""
import os, sys, time, json, numpy as np
from eyetoarm.vision.ai_config import load_ai_config
from eyetoarm.vision.ai_process import ProcessAiSegDetector
cfg_path = sys.argv[1]
px = ProcessAiSegDetector(load_ai_config(cfg_path), cfg_path, factory="ai_process_fake:make",
                          sweep=lambda: {"removed": [], "kept": {}})
px.prepare()
print(json.dumps({"child": px.pid, "torch": "torch" in sys.modules, "ultralytics": "ultralytics" in sys.modules}),
      flush=True)
time.sleep(60)
"""

    def test_parent_kill_terminates_child(self):
        """부모 SIGKILL 뒤 자식은 부모 감시 스레드로 끝나고, 부모 프로세스에는 torch·ultralytics가 없다."""
        helper = subprocess.Popen([sys.executable, "-c", self.HELPER, self.cfg_path], env=self.env(),
                                  stdout=subprocess.PIPE, text=True)
        self.addCleanup(lambda: helper.poll() is None and helper.kill())
        info = json.loads(helper.stdout.readline())
        child = info["child"]

        self.assertEqual((info["torch"], info["ultralytics"]), (False, False))
        helper.kill()
        helper.wait(5)
        deadline = time.monotonic() + 5

        while time.monotonic() < deadline and os.path.exists(f"/proc/{child}") and \
                not open(f"/proc/{child}/stat").read().split(") ")[1].startswith("Z"):
            time.sleep(0.05)

        alive = os.path.exists(f"/proc/{child}") and \
            not open(f"/proc/{child}/stat").read().split(") ")[1].startswith("Z")
        self.assertFalse(alive, f"child {child} still alive after parent SIGKILL")


if __name__ == "__main__":
    unittest.main()
