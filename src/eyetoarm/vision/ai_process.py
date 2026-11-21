"""AI 추론 프로세스 분리의 부모 쪽 대리 객체 `ProcessAiSegDetector`.

`AiSegDetector`와 같은 계약(`prepare() → dict`, `detector(job, token) → (관측, AiSegResult)`)으로 실행기에 끼운다.
crop·predict·변환(M2 끝점 포함)은 자식 프로세스(`ai_process_child`)가 하고, 부모는 `observation_from_ai`만 한다.

- 영상: 부모가 만든 `/dev/shm` 슬롯 2개에 한 번 복사하고 자식은 복사 없이 본다. 진행 중 요청은 1개다.
- 결과: 전용 socketpair로 받는다. pickle은 이 부모가 띄운 자식과의 연결에서만 쓰고, 받은 결과는 다시 검증한다.
- 요청 번호(`req_id`)가 진행 중 요청과 다른 응답(취소·시간 초과 뒤 늦은 응답)은 버린다.
- 자식 죽음·멈춤·통신 오류는 `InferProcessError`로 알리고 이후 호출을 거부한다(`failed`). 자동 재시작은 없다.
- 부모는 torch를 import하지 않는다. 호출은 실행기 detector worker 스레드 하나에서만 한다.
"""
from __future__ import annotations

import dataclasses
import mmap
import statistics
from collections import deque
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from multiprocessing.connection import Connection

import numpy as np

from eyetoarm.common.types import InvalidationKind, PlanarObservation
from eyetoarm.vision.ai_config import AiSegConfig
from eyetoarm.vision.ai_convert import observation_from_ai
from eyetoarm.vision.ai_result import AiJobConfigError, AiSegResult, ModelPrepareError
from eyetoarm.vision.pipeline_runner import JobCancelled
from eyetoarm.vision.pipeline_types import JobContext

DEFAULT_FACTORY = "eyetoarm.vision.ai_detector:AiSegDetector"
CHILD_MODULE = "eyetoarm.vision.ai_process_child"
REQUIRED_ENV = (("YOLO_AUTOINSTALL", "false"), ("YOLO_OFFLINE", "1"))   # ai_detector.REQUIRED_ENV와 같은 값
SHM_DIR = "/dev/shm"
SHM_PREFIX = "eyetoarm_infer_"
SLOT_COUNT = 2
LOAD = InvalidationKind.MODEL_LOAD_FAILED


class _TimeStat:
    """시간 통계(ms): 전체 개수·최댓값과 최근 window개의 중앙값·p95. 긴 실행에서도 메모리가 일정하다."""

    def __init__(self, window: int = 4096) -> None:
        """최근 window개를 보관한다."""
        self.n, self.max = 0, 0.0
        self._recent: deque[float] = deque(maxlen=window)

    def add(self, value_ms: float) -> None:
        """값 하나."""
        self.n += 1
        self.max = max(self.max, value_ms)
        self._recent.append(value_ms)

    def as_dict(self) -> dict:
        """기록용 요약(값이 없으면 n만)."""
        if not self._recent:
            return {"n": 0}

        ordered = sorted(self._recent)

        return {"n": self.n, "median": round(statistics.median(ordered), 3),
                "p95": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 3), "max": round(self.max, 3)}


class InferProcessError(Exception):
    """추론 프로세스를 더 쓸 수 없음(죽음·멈춤·통신 오류). 실행기에서는 그 작업이 INFERENCE_ERROR가 된다."""


class InferJobError(Exception):
    """자식이 작업 하나에서 낸 일반 오류(원래 예외 종류와 메시지를 담음). 실행기에서는 INFERENCE_ERROR."""


class ShmSlots:
    """`/dev/shm` 파일 하나에 같은 크기 영상 슬롯 N개. 부모가 만들고(`create`) 지우며, 자식은 `attach`만 한다.

    multiprocessing.shared_memory 대신 직접 열어 resource_tracker 프로세스·자식 종료 때 삭제를 피한다.
    """

    def __init__(self, name: str, shape: tuple[int, ...], count: int, *, create: bool) -> None:
        """이름·슬롯 모양(uint8)·개수. create면 새로 만들고(있으면 FileExistsError), 아니면 기존 것에 붙는다."""
        self.name, self.shape, self.count = name, tuple(int(v) for v in shape), int(count)
        self.slot_bytes = int(np.prod(self.shape))
        size = self.slot_bytes * self.count
        path = os.path.join(SHM_DIR, name)
        flags = os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0)
        fd = os.open(path, flags, 0o600)

        try:
            if create:
                os.ftruncate(fd, size)

            self._mmap = mmap.mmap(fd, size)
        except BaseException:
            os.close(fd)

            if create:
                os.unlink(path)

            raise

        os.close(fd)
        self.owner = create
        self.views = np.ndarray((self.count, *self.shape), dtype=np.uint8, buffer=self._mmap)

    def close(self) -> None:
        """보기·매핑을 닫고, 만든 쪽이면 파일을 지운다. 두 번 불러도 된다."""
        if self._mmap is None:
            return

        self.views = None

        try:
            self._mmap.close()
        except BufferError:   # 아직 보기를 잡은 배열이 있으면 매핑은 프로세스 종료 때 풀린다(파일 삭제는 계속)
            pass

        self._mmap = None

        if self.owner:
            try:
                os.unlink(os.path.join(SHM_DIR, self.name))
            except FileNotFoundError:   # 이미 지워짐(정리 경로)
                pass


class ProcessAiSegDetector:
    """자식 프로세스에서 추론하는 `AiSegDetector` 대리 객체. 호출은 실행기 detector worker 스레드 하나에서만 한다.

    `failed`·`summary()`는 다른 스레드에서 읽어도 된다(잠금으로 보호). `close()`는 실행이 끝난 뒤 부른다.
    """

    takes_token = True   # 실행기 `detector_takes_token=True`로 넘긴다(취소 토큰으로 유예 안에 반환)

    def __init__(self, config: AiSegConfig, config_path: str, *, factory: str = DEFAULT_FACTORY,
                 torch_threads: int | None = None, child_log: str | None = None,
                 env: dict[str, str] | None = None, python: str | None = None, poll_interval_s: float = 0.005,
                 stall_after_s: float = 5.0, stop_grace_s: float = 2.0, kill_grace_s: float = 1.0) -> None:
        """설정·설정 파일 경로(자식이 다시 읽어 sha256을 맞춘다)와 수명 한도. I/O·프로세스 시작은 하지 않는다."""
        if not isinstance(config, AiSegConfig):
            raise TypeError(f"config must be AiSegConfig: {type(config).__name__}")

        if torch_threads is not None and (isinstance(torch_threads, bool) or not isinstance(torch_threads, int)
                                          or torch_threads < 1):
            raise ValueError(f"torch_threads must be a positive int or None: {torch_threads!r}")

        for name, value in (("poll_interval_s", poll_interval_s), ("stall_after_s", stall_after_s),
                            ("stop_grace_s", stop_grace_s), ("kill_grace_s", kill_grace_s)):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not value > 0:
                raise ValueError(f"{name} must be > 0: {value!r}")

        self._config, self._config_path, self._factory = config, os.fspath(config_path), factory
        self._torch_threads, self._child_log = torch_threads, child_log
        self._env = dict(os.environ if env is None else env)
        self._python = python or sys.executable
        self._poll_s, self._stall_after_s = float(poll_interval_s), float(stall_after_s)
        self._stop_grace_s, self._kill_grace_s = float(stop_grace_s), float(kill_grace_s)

        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._conn: Connection | None = None
        self._log_fh = None
        self._slots: ShmSlots | None = None
        self._retired: list[ShmSlots] = []
        self._busy: dict[int, int] = {}            # 슬롯 번호 → 그 슬롯을 쓰는 요청 번호
        self._abandoned: dict[int, float] = {}     # 버린 요청 번호 → 버린 시각(monotonic)
        self._req = 0
        self._failed: str | None = None
        self._closed = False
        self._stats = {"jobs": 0, "results": 0, "job_errors": 0, "cancelled": 0, "late_replies": 0}
        self._times = {name: _TimeStat() for name in ("copy_ms", "roundtrip_ms", "child_process_ms", "ipc_overhead_ms")}
        self._ready: dict = {}
        self._exit: dict | None = None

    @property
    def config(self) -> AiSegConfig:
        """어댑터 설정."""
        return self._config

    @property
    def failed(self) -> str | None:
        """사용 불가 사유(없으면 None). 실행을 실패로 끝낼지 판단하는 데 쓴다."""
        with self._lock:
            return self._failed

    @property
    def pid(self) -> int | None:
        """자식 pid(시작 전 None)."""
        return None if self._proc is None else self._proc.pid

    def summary(self) -> dict:
        """요약 기록용 상태(계수·최대 시간·실패 사유·종료 정보)."""
        with self._lock:
            return {"pid": self.pid, "factory": self._factory, "torch_threads_requested": self._torch_threads,
                    **self._ready, "failed": self._failed, "exit": self._exit,
                    "pending_abandoned": len(self._abandoned), **dict(self._stats),
                    "times": {name: t.as_dict() for name, t in self._times.items()}}

    # ------------------------------------------------------------------ 준비

    def _child_env(self) -> dict[str, str]:
        """자식 환경: 필수 변수를 강제하고(다른 값이면 거부), 요청한 CPU 스레드 수를 넣는다."""
        env = dict(self._env)

        for key, want in REQUIRED_ENV:
            value = env.get(key)

            if value is not None and value != want:
                raise ModelPrepareError(LOAD, f"unsafe env {key}={value!r} (must be {want!r}; refuses auto-install)")

            env[key] = want

        if self._torch_threads is not None:
            env["OMP_NUM_THREADS"] = str(self._torch_threads)

        return env

    def prepare(self) -> dict:
        """자식을 띄우고 모델을 준비해 model_ready 정보를 돌려준다. 실패하면 ModelPrepareError."""
        if self._proc is not None:
            raise RuntimeError("ProcessAiSegDetector.prepare() called twice")

        t0 = time.monotonic_ns()
        env = self._child_env()
        parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)

        try:
            cmd = [self._python, "-m", CHILD_MODULE, "--fd", str(child_sock.fileno()), "--factory", self._factory,
                   "--parent-pid", str(os.getpid())]

            if self._child_log is not None:
                self._log_fh = open(self._child_log, "ab")

            try:
                self._proc = subprocess.Popen(cmd, pass_fds=[child_sock.fileno()], env=env, stdin=subprocess.DEVNULL,
                                              stdout=self._log_fh, stderr=self._log_fh)
            except OSError as exc:
                parent_sock.close()
                self._fail(f"start failed: {type(exc).__name__}: {exc}")
                raise ModelPrepareError(LOAD, f"infer process start failed: {type(exc).__name__}: {exc}") from exc
        finally:
            child_sock.close()

        self._conn = Connection(parent_sock.detach())

        try:
            self._conn.send(("prepare", {"config_path": self._config_path, "config_sha256": self._config.config_sha256,
                                         "torch_threads": self._torch_threads}))
        except OSError as exc:
            raise self._prepare_lost(f"send failed: {type(exc).__name__}: {exc}") from exc

        msg = self._recv_until(None, cancel=None)

        if msg is None:
            raise self._prepare_lost("no reply")

        kind, payload = msg[0], msg[1]

        if kind == "prepare_error":
            self._fail(f"prepare_error: {payload['cause']}")
            raise ModelPrepareError(payload["kind"], payload["cause"])

        if kind != "ready":
            raise self._prepare_lost(f"unexpected reply {kind!r}")

        info = dict(payload["info"])
        ready = {"start_to_ready_ms": round((time.monotonic_ns() - t0) / 1e6, 3),
                 "torch_threads": payload.get("torch_threads"), "device_name": info.get("device_name")}

        with self._lock:
            self._ready = ready

        info["infer_process"] = {"pid": self._proc.pid, "factory": self._factory,
                                 "torch_threads_requested": self._torch_threads, **ready}

        return info

    def _returncode(self) -> int | None:
        """자식 종료 코드. 연결이 먼저 끊긴 경우를 위해 잠깐(kill_grace_s, 최대 0.5 s) 기다린다."""
        if self._proc is None:
            return None

        try:
            return self._proc.wait(min(self._kill_grace_s, 0.5))
        except subprocess.TimeoutExpired:
            return None

    def _prepare_lost(self, detail: str) -> ModelPrepareError:
        """준비 중 자식을 잃음 → 실패 상태 + MODEL_LOAD_FAILED."""
        rc = self._returncode()
        reason = f"infer process lost during prepare ({detail}; returncode={rc})"
        self._fail(reason)

        return ModelPrepareError(LOAD, reason)

    # ------------------------------------------------------------------ 작업

    def __call__(self, job: JobContext, token=None) -> tuple[PlanarObservation, AiSegResult]:
        """작업 하나를 자식에게 넘기고 결과를 받아 (관측, 표시 결과)를 돌려준다."""
        failed = self.failed

        if failed is not None:
            raise InferProcessError(f"infer process unavailable: {failed}")

        if self._conn is None or self._closed:
            raise RuntimeError("ProcessAiSegDetector used before prepare() or after close()")

        self._check_stall(time.monotonic())
        frame = job.frame
        image = frame.image

        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3:
            raise ValueError(f"frame image must be uint8 (H, W, C): {getattr(image, 'dtype', None)} "
                             f"{getattr(image, 'shape', None)}")

        slots = self._slots_for(image.shape)
        slot = next((i for i in range(slots.count) if i not in self._busy), None)

        if slot is None:
            reason = "infer_process_stalled: no free image slot (earlier requests still unanswered)"
            self._fail(reason)
            self._terminate()
            raise InferProcessError(reason)

        t0 = time.perf_counter_ns()
        np.copyto(slots.views[slot], image)
        t1 = time.perf_counter_ns()

        self._req += 1
        req = self._req
        self._busy[slot] = req
        meta = dataclasses.replace(frame, image=None)

        try:
            self._conn.send(("job", {"req_id": req, "job_seq": job.job_seq, "shm": slots.name, "shape": slots.shape,
                                     "slot_count": slots.count, "slot": slot, "meta": meta,
                                     "detector_config_id": job.detector_config_id}))
        except OSError as exc:
            raise self._lost(f"send failed: {type(exc).__name__}: {exc}") from exc

        self._count("jobs")
        msg = self._recv_until(req, cancel=token)

        if msg is None:   # 취소: 이 요청을 버리고 실행기에 알린다
            self._abandoned[req] = time.monotonic()
            self._count("cancelled")
            raise JobCancelled(f"infer process request {req} abandoned")

        t2 = time.perf_counter_ns()
        kind, payload = msg[0], msg[1]

        if kind == "job_error":
            self._count("job_errors")

            if payload["kind"] == "config":
                raise AiJobConfigError(payload["message"])

            raise InferJobError(payload["message"])

        result = payload["result"]

        if not isinstance(result, AiSegResult) or payload.get("job_seq") != job.job_seq:
            raise self._lost(f"bad result for req {req}: {type(result).__name__} job_seq={payload.get('job_seq')}")

        result = dataclasses.replace(result)   # __post_init__ 검증을 다시 거친다
        self._count("results")
        roundtrip_ms = (t2 - t1) / 1e6
        self._note_times(copy_ms=(t1 - t0) / 1e6, roundtrip_ms=roundtrip_ms, child_process_ms=result.process_ms,
                         ipc_overhead_ms=max(0.0, roundtrip_ms - result.process_ms))

        return observation_from_ai(job, result), result

    def _slots_for(self, shape: tuple[int, ...]) -> ShmSlots:
        """영상 모양에 맞는 슬롯. 모양이 바뀌면 새로 만들고, 쓰던 것은 응답이 남았으면 닫을 때까지 보관한다."""
        if self._slots is not None and self._slots.shape == tuple(shape):
            return self._slots

        if self._slots is not None:
            if self._busy:
                self._retired.append(self._slots)
            else:
                self._slots.close()

            self._busy.clear()   # 새 슬롯 묶음에는 진행 중 요청이 없다(옛 요청 응답은 req로 버린다)

        name = f"{SHM_PREFIX}{os.getpid()}_{uuid.uuid4().hex[:12]}"
        self._slots = ShmSlots(name, shape, SLOT_COUNT, create=True)

        return self._slots

    def _recv_until(self, req: int | None, *, cancel):
        """응답을 기다린다. req가 None이면 첫 메시지, 아니면 그 req 응답. 취소면 None. 잃으면 예외.

        poll 대기 중에는 GIL을 놓는다. 늦은 응답은 슬롯을 풀고 버린다.
        """
        while True:
            try:
                ready = self._conn.poll(self._poll_s)
            except (OSError, EOFError) as exc:
                raise self._lost_any(req, f"poll failed: {type(exc).__name__}: {exc}") from exc

            if ready:
                try:
                    msg = self._conn.recv()
                except (OSError, EOFError) as exc:
                    raise self._lost_any(req, f"connection closed: {type(exc).__name__}: {exc}") from exc

                if req is None:
                    return msg

                got = msg[1].get("req_id") if isinstance(msg, tuple) and len(msg) == 2 and isinstance(msg[1], dict) \
                    else None
                self._release(got)

                if got == req:
                    return msg

                if got in self._abandoned:
                    del self._abandoned[got]
                    self._count("late_replies")
                    continue

                raise self._lost(f"unexpected reply req_id={got!r} while waiting for {req}")

            if self._proc.poll() is not None:
                raise self._lost_any(req, "child exited")

            if self._closed:
                raise self._lost_any(req, "closed")

            if cancel is not None and cancel.cancelled:
                return None

            if req is not None:
                self._check_stall(time.monotonic())

    def _release(self, req) -> None:
        """그 요청이 쓰던 슬롯을 푼다."""
        for slot, owner in list(self._busy.items()):
            if owner == req:
                del self._busy[slot]

    def _check_stall(self, now: float) -> None:
        """버린 요청이 stall_after_s 넘게 답이 없으면 멈춤으로 보고 자식을 끝낸다(첫 검출 창은 실행기가 정함)."""
        if self._abandoned and now - min(self._abandoned.values()) > self._stall_after_s:
            oldest = min(self._abandoned, key=self._abandoned.get)
            reason = (f"infer_process_stalled: request {oldest} unanswered {now - self._abandoned[oldest]:.1f} s "
                      f"after cancel (limit {self._stall_after_s} s)")
            self._fail(reason)
            self._terminate()
            raise InferProcessError(reason)

    def _lost_any(self, req, detail: str) -> Exception:
        """준비(req None)면 MODEL_LOAD_FAILED, 작업이면 InferProcessError. close() 중에 끊긴 것은 실패로 남기지 않는다."""
        if self._closed:
            return InferProcessError(f"infer process closed ({detail})")

        return self._prepare_lost(detail) if req is None else self._lost(detail)

    def _lost(self, detail: str) -> InferProcessError:
        """자식을 잃음 → 실패 상태(자식 종료 코드 포함)."""
        rc = self._returncode()
        reason = f"infer_process_died: {detail} (returncode={rc})"
        self._fail(reason)
        self._terminate()

        return InferProcessError(reason)

    def _fail(self, reason: str) -> None:
        """첫 실패 사유만 남긴다."""
        with self._lock:
            if self._failed is None:
                self._failed = reason

    def _count(self, key: str) -> None:
        """계수 하나 증가."""
        with self._lock:
            self._stats[key] += 1

    def _note_times(self, **values_ms: float) -> None:
        """정상 결과 하나의 단계 시간(ms)."""
        with self._lock:
            for name, value in values_ms.items():
                self._times[name].add(value)

    # ------------------------------------------------------------------ 종료

    def _terminate(self) -> None:
        """자식을 SIGTERM → kill_grace_s → SIGKILL로 끝낸다(이미 끝났으면 무변)."""
        proc = self._proc

        if proc is None or proc.poll() is not None:
            return

        proc.terminate()

        try:
            proc.wait(self._kill_grace_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(self._kill_grace_s)

    def close(self) -> dict:
        """자식에게 종료를 요청하고(stop_grace_s), 안 끝나면 SIGTERM·SIGKILL. 소켓·슬롯을 정리한다. 두 번 불러도 된다."""
        if self._closed:
            return dict(self._exit or {})

        self._closed = True
        forced = None
        proc = self._proc

        if proc is not None:
            if proc.poll() is None and self._conn is not None:
                try:
                    self._conn.send(("stop", {}))
                except OSError:   # 이미 끊긴 자식: 아래에서 신호로 끝낸다
                    pass

            try:
                proc.wait(self._stop_grace_s)
                forced = False
            except subprocess.TimeoutExpired:
                forced = True
                self._terminate()

        if self._conn is not None:
            self._conn.close()

        for slots in [self._slots, *self._retired]:
            if slots is not None:
                slots.close()

        if self._log_fh is not None:
            self._log_fh.close()

        exit_info = {"returncode": None if proc is None else proc.returncode, "forced": forced}

        with self._lock:
            self._exit = exit_info

        return dict(exit_info)


__all__ = ["DEFAULT_FACTORY", "InferJobError", "InferProcessError", "ProcessAiSegDetector", "ShmSlots"]
