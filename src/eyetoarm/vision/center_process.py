"""B center 그리기를 별도 프로세스로(#413): A 프로세스는 「그리기 요청」만 넘기고, 자식이 그려서 발행한다.

같은 프로세스에서 그리면 그리기(`render_center`)가 검출·추적과 GIL을 나눠 써서 A 전체가 느려진다(#372·#398).
그래서 B 계산·`/b/results`·수락·목표는 A에 남기고, 그리기와 center 영상 발행만 자식 프로세스로 보낸다.

- 넘기는 것: 사진 한 장(이름 없는 공유 메모리 memfd 한 칸), 그 사진의 B 결과(`WorkPlanarResult`·`T_W_C`), 번호표.
- 기다리지 않는다: `submit`은 가장 새 요청 하나만 남기고 바로 돌아온다. 자식이 앞 사진을 그리는 중이면 그 사이에
  들어온 것 가운데 마지막만 그리고 나머지는 버린다(표시 전용이라 최신 우선, 버린 수를 센다).
- 자식이 죽거나 멈추면 center 영상만 멈춘다. 다시 띄우지 않는다(`failed`에 사유, `on_failed`로 한 번 알림).
  A의 다른 일은 영향받지 않는다.
- 요청은 자식이 준비(ready)를 알린 뒤에만 보낸다. 멈춤 시계는 그 뒤부터 돈다. 준비가 `ready_timeout_s` 안에
  끝나지 않으면 그때 실패로 돌린다(부하가 큰 기계의 느린 준비를 멈춤으로 오인하지 않게, r2 D3).
- 통로는 전용 socketpair다. pickle은 이 부모가 띄운 자식과의 연결에서만 쓴다(#259 추론 프로세스와 같은 규칙).
- 공유 메모리는 이름이 없어(memfd) 부모·자식이 어떻게 끝나도 파일이 남지 않는다.

스레드: `submit`·`configure`는 B worker 스레드가, 보내기는 `eyetoarm-center-send`, 받기는 `eyetoarm-center-recv`가
한다. 셋이 나눠 쓰는 상태는 모두 `_cond`(잠금)로 보호하고, 연결에 쓰는 일은 `_send_lock`으로 한 번에 하나만 한다.
B 파일은 바꾸지 않는다. 자식 안에서 B 공개 함수(`BStage.render_center`·`CenterOverlayPublisher`)를 그대로 부른다.
"""
from __future__ import annotations

import mmap
import os
import socket
import subprocess
import sys
import threading
import time
from multiprocessing.connection import Connection
from typing import Callable

import numpy as np

CHILD_MODULE = "eyetoarm.vision.center_process_child"
DEFAULT_FACTORY = "eyetoarm.vision.center_process_child:ros_backend"
STALL_S = 5.0          # 준비된 자식이 한 장을 이보다 오래 붙들면 멈춘 것으로 본다(준비 전에는 재지 않음)
READY_TIMEOUT_S = 60.0  # 자식이 띄운 뒤 이 안에 준비(ready)를 알리지 않으면 실패로 본다
STOP_GRACE_S = 2.0     # stop 메시지 뒤 스스로 끝나기를 기다리는 시간
KILL_GRACE_S = 2.0     # SIGTERM 뒤 SIGKILL까지


class CenterDrawProcess:
    """center 그리기 자식 프로세스의 부모 쪽 대리. `start` → (`configure`·`submit` 여러 번) → `close`."""

    def __init__(self, root, *, factory: str = DEFAULT_FACTORY, python: str | None = None, env: dict | None = None,
                 child_log: str | None = None, stall_s: float = STALL_S, ready_timeout_s: float = READY_TIMEOUT_S,
                 stop_grace_s: float = STOP_GRACE_S, kill_grace_s: float = KILL_GRACE_S,
                 on_failed: Callable[[str], None] | None = None, clock: Callable[[], float] = time.monotonic) -> None:
        """`root`는 저장소 맨 위 폴더(자식이 B 설정을 읽는 곳). `factory`·`python`·`env`·`clock`은 시험 주입용.

        `on_failed(사유)`는 자식을 더 쓸 수 없게 된 때 한 번 불린다(받기·보내기 스레드에서, 잠금 밖에서).
        """
        self._root = str(root)
        self._factory = factory
        self._python = python or sys.executable
        self._env = dict(os.environ if env is None else env)
        self._child_log = child_log
        self._stall_s, self._stop_grace_s, self._kill_grace_s = float(stall_s), float(stop_grace_s), float(kill_grace_s)
        self._ready_timeout_s = float(ready_timeout_s)
        self._on_failed = on_failed
        self._clock = clock

        self._cond = threading.Condition()
        self._send_lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._conn: Connection | None = None
        self._log_fh = None
        self._mem_fd: int | None = None
        self._mem: mmap.mmap | None = None
        self._mem_size = 0
        self._threads: list[threading.Thread] = []

        # 아래는 _cond로 보호한다
        self._pending: dict | None = None      # 아직 안 보낸 가장 새 요청
        self._camera = None                    # 아직 안 보낸 카메라 설정
        self._busy_since: float | None = None  # 자식이 그리는 중이면 보낸 시각
        self._started_at: float | None = None  # 자식을 띄운 시각(준비 한도용)
        self._stopping = False
        self._closed = False
        self._ready: dict | None = None
        self._failed: str | None = None
        self._exit: dict | None = None
        self._req = 0
        self._counts = {"requested": 0, "sent": 0, "drawn": 0, "replaced": 0, "refused": 0, "child_errors": 0,
                        "send_errors": 0}
        self._first_child_error: str | None = None
        self._render_ms: list[float] = []
        self._publish_ms: list[float] = []

    # ------------------------------------------------------------------ 상태

    @property
    def failed(self) -> str | None:
        """자식을 더 쓸 수 없게 된 사유(정상이면 None)."""
        with self._cond:
            return self._failed

    @property
    def pid(self) -> int | None:
        """자식 pid(안 띄웠으면 None)."""
        return None if self._proc is None else self._proc.pid

    def summary(self) -> dict:
        """요약 블록: 요청·보냄·그림·버림 수, 자식 오류, 실패 사유, 그리기·발행 시간."""
        with self._cond:
            render, publish = sorted(self._render_ms), sorted(self._publish_ms)
            out = {"pid": self.pid, "factory": self._factory, **self._counts, "failed": self._failed,
                   "first_child_error": self._first_child_error, "ready": self._ready, "exit": self._exit}

        def mid(values, q):
            """분위수(없으면 None)."""
            return None if not values else round(values[min(len(values) - 1, int(q * len(values)))], 3)

        out["render_ms"] = {"p50": mid(render, .5), "p95": mid(render, .95)}
        out["publish_ms"] = {"p50": mid(publish, .5), "p95": mid(publish, .95)}

        return out

    # ------------------------------------------------------------------ 시작

    def start(self) -> None:
        """자식을 띄우고 주고받는 스레드를 시작한다. 띄우지 못하면 OSError(부르는 쪽이 center만 끄고 계속한다)."""
        if self._proc is not None:
            raise RuntimeError("CenterDrawProcess.start() called twice")

        parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        self._mem_fd = os.memfd_create("eyetoarm_center", os.MFD_CLOEXEC)

        try:
            cmd = [self._python, "-m", CHILD_MODULE, "--fd", str(child_sock.fileno()), "--mem-fd", str(self._mem_fd),
                   "--factory", self._factory, "--parent-pid", str(os.getpid()), "--root", self._root]

            if self._child_log is not None:
                self._log_fh = open(self._child_log, "ab")

            try:
                self._proc = subprocess.Popen(cmd, pass_fds=[child_sock.fileno(), self._mem_fd], env=self._env,
                                              stdin=subprocess.DEVNULL, stdout=self._log_fh, stderr=self._log_fh)
            except OSError as exc:
                parent_sock.close()
                self._set_failed(f"start failed: {type(exc).__name__}: {exc}")
                self._release_resources()
                raise
        finally:
            child_sock.close()

        self._conn = Connection(parent_sock.detach())

        with self._cond:
            self._started_at = self._clock()

        for name, target in (("eyetoarm-center-recv", self._recv_loop), ("eyetoarm-center-send", self._send_loop)):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

    # ------------------------------------------------------------------ 요청

    def configure(self, camera) -> None:
        """그리기에 쓸 카메라 값(B `CameraProfile`)을 자식에 알린다. 단계가 새로 만들어질 때마다 부른다."""
        with self._cond:
            self._camera = camera
            self._cond.notify_all()

    def submit(self, image: np.ndarray, stamp_ns: int | None, optical_frame: str | None, items: list, keys: dict,
               *, guides: bool = True) -> bool:
        """그리기 요청 하나를 맡긴다(기다리지 않음). 맡았으면 True, 자식을 못 쓰면 False.

        `items`는 그릴 결과 `(WorkPlanarResult, T_W_C)` 목록, `keys`는 번호표(`overlay_keys`).
        앞 요청이 아직 안 나갔으면 그것을 버리고 이 요청으로 바꾼다.
        """
        with self._cond:
            if self._failed is not None or self._stopping or self._proc is None:
                self._counts["refused"] += 1
                return False

            self._counts["requested"] += 1

            if self._pending is not None:
                self._counts["replaced"] += 1

            self._req += 1
            self._pending = {"req": self._req, "image": image, "stamp_ns": stamp_ns, "optical_frame": optical_frame,
                             "items": items, "keys": keys, "guides": bool(guides)}
            self._cond.notify_all()

        return True

    # ------------------------------------------------------------------ 스레드

    def _send_loop(self) -> None:
        """보내기 스레드: 자식이 준비된 뒤, 한가해지면 가장 새 요청 하나를 공유 메모리에 싣고 보낸다.

        멈춤(준비된 자식이 한 장을 stall_s 넘게 붙듦)과 준비 지연(ready_timeout_s 안에 준비를 알리지 않음)도 여기서 본다.
        """
        while True:
            with self._cond:
                while True:
                    if self._stopping or self._failed is not None:
                        return

                    now = self._clock()

                    if self._ready is None:
                        if self._started_at is not None and now - self._started_at > self._ready_timeout_s:
                            stalled = f"child not ready after {self._ready_timeout_s} s"
                            break
                    elif self._busy_since is not None and now - self._busy_since > self._stall_s:
                        stalled = f"child stalled: no answer for {self._stall_s} s"
                        break
                    elif self._busy_since is None and (self._camera is not None or self._pending is not None):
                        stalled = None
                        break

                    self._cond.wait(0.2)

                camera, job = (None, None) if stalled else (self._camera, self._pending)
                self._camera = None

                if job is not None:
                    self._pending = None
                    self._busy_since = self._clock()

            if stalled:
                self._set_failed(stalled)
                self._terminate()
                return

            try:
                if camera is not None:
                    self._send(("config", {"camera": camera}))

                if job is not None:
                    self._send(("draw", self._load(job)))

                    with self._cond:
                        self._counts["sent"] += 1
            except (OSError, ValueError) as exc:   # 끊긴 연결·영상 형식 오류: 이 요청만 잃고, 연결이 죽었으면 받기 쪽이 실패로 돌린다
                with self._cond:
                    self._counts["send_errors"] += 1
                    self._busy_since = None

                    if self._first_child_error is None:
                        self._first_child_error = f"send: {type(exc).__name__}: {exc}"

    def _load(self, job: dict) -> dict:
        """사진을 공유 메모리에 복사하고 자식에 보낼 메시지(영상 빼고)를 만든다."""
        image = np.ascontiguousarray(job["image"])

        if image.dtype != np.uint8 or image.ndim != 3:
            raise ValueError(f"center image must be uint8 HxWxC: dtype={image.dtype} shape={image.shape}")

        if image.nbytes > self._mem_size:
            if self._mem is not None:
                self._mem.close()

            os.ftruncate(self._mem_fd, image.nbytes)
            self._mem = mmap.mmap(self._mem_fd, image.nbytes)
            self._mem_size = image.nbytes

        np.frombuffer(self._mem, dtype=np.uint8, count=image.nbytes)[:] = image.reshape(-1)

        return {"req": job["req"], "shape": tuple(image.shape), "nbytes": image.nbytes, "stamp_ns": job["stamp_ns"],
                "optical_frame": job["optical_frame"], "items": job["items"], "keys": job["keys"],
                "guides": job["guides"]}

    def _send(self, message) -> None:
        """연결에 한 번에 하나만 쓴다."""
        with self._send_lock:
            self._conn.send(message)

    def _recv_loop(self) -> None:
        """받기 스레드: 자식의 답(`ready`·`done`·`failed`)을 반영한다. 연결이 끊기면 실패로 돌린다."""
        while True:
            try:
                kind, payload = self._conn.recv()
            except (EOFError, OSError):
                with self._cond:
                    stopping = self._stopping

                if not stopping:
                    rc = None if self._proc is None else self._proc.poll()
                    self._set_failed(f"child connection lost (returncode={rc})")

                return
            except Exception as exc:   # 깨진 답(풀 수 없는 pickle 등): 이 자식은 더 믿지 않는다
                self._set_failed(f"bad reply: {type(exc).__name__}: {exc}")
                return

            with self._cond:
                if kind == "ready":
                    self._ready = dict(payload)
                elif kind == "done":
                    self._busy_since = None

                    if payload.get("ok"):
                        self._counts["drawn"] += 1
                        self._render_ms.append(float(payload.get("render_ms") or 0.0))
                        self._publish_ms.append(float(payload.get("publish_ms") or 0.0))
                        del self._render_ms[:-4096], self._publish_ms[:-4096]
                    else:
                        self._counts["child_errors"] += 1

                        if self._first_child_error is None:
                            self._first_child_error = str(payload.get("error"))

                self._cond.notify_all()

            if kind == "failed":   # 자식이 준비·설정에 실패했다고 알림: 곧 끝난다
                self._set_failed(f"child failed: {payload.get('cause')}")

    def _set_failed(self, reason: str) -> None:
        """실패 사유를 한 번만 적고 기다리는 스레드를 깨운다. 처음 실패일 때만 `on_failed`를 부른다(잠금 밖에서)."""
        with self._cond:
            first = self._failed is None and not self._stopping

            if self._failed is None:
                self._failed = reason

            self._pending = None
            self._cond.notify_all()

        if first and self._on_failed is not None:
            try:
                self._on_failed(reason)
            except Exception as exc:   # noqa: BLE001 - 알림 실패가 보내기·받기 스레드를 죽이지 않게(요약에는 사유가 남음)
                print(f"center_process: on_failed callback failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    # ------------------------------------------------------------------ 종료

    def _terminate(self) -> None:
        """자식을 SIGTERM → kill_grace_s → SIGKILL로 끝낸다(이미 끝났으면 아무것도 안 함)."""
        proc = self._proc

        if proc is None or proc.poll() is not None:
            return

        proc.terminate()

        try:
            proc.wait(self._kill_grace_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(self._kill_grace_s)

    def _release_resources(self) -> None:
        """연결·공유 메모리·자식 기록 파일을 닫는다. 하나가 실패해도 나머지를 한다."""
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:   # 이미 닫힘
                pass

        if self._mem is not None:
            try:
                self._mem.close()
            except (BufferError, ValueError):   # 보내기 스레드가 아직 쥐고 있음: fd를 닫으면 프로세스가 끝날 때 함께 풀린다
                pass

            self._mem = None

        if self._mem_fd is not None:
            try:
                os.close(self._mem_fd)
            except OSError:   # 이미 닫힘
                pass

            self._mem_fd = None

        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None

    def close(self) -> dict:
        """자식에 종료를 알리고(stop_grace_s), 안 끝나면 SIGTERM·SIGKILL. 스레드·연결·공유 메모리를 정리한다. 두 번 불러도 된다."""
        with self._cond:
            if self._closed:
                return dict(self._exit or {})

            self._closed = True
            self._stopping = True
            self._pending = None
            self._cond.notify_all()

        forced = None
        proc = self._proc

        # 자식을 기다리다 KeyboardInterrupt가 나도 연결·공유 메모리 정리는 반드시 한다
        try:
            if proc is not None:
                if proc.poll() is None and self._conn is not None:
                    try:
                        self._send(("stop", {}))
                    except OSError:   # 이미 끊긴 자식: 아래에서 신호로 끝낸다
                        pass

                try:
                    proc.wait(self._stop_grace_s)
                    forced = False
                except subprocess.TimeoutExpired:
                    forced = True
                    self._terminate()
        finally:
            for thread in self._threads:
                thread.join(1.0)

            self._release_resources()

        exit_info = {"returncode": None if proc is None else proc.returncode, "forced": forced}

        with self._cond:
            self._exit = exit_info

        return dict(exit_info)


def process_center_consumer(stage, publisher, gate, drawer: CenterDrawProcess, *, guides: bool = True,
                            on_error: Callable[[Exception], None] | None = None) -> Callable:
    """B `center_overlay_consumer` 자리의 목록 소비자(별도 프로세스 판). 반환값은 `stage.process_frame`과 똑같다.

    B 계산 → (보는 쪽이 있으면) 그리기 요청을 자식에 맡김 → 결과 숫자(`/b/results`)는 A가 그대로 발행.
    요청·발행 실패는 B 결과에 영향을 주지 않는다(`on_error`). 자식을 못 쓰면 center 영상만 나가지 않는다.
    """
    from eyetoarm.transforms.center_overlay_pub import overlay_keys, results_payload

    drawer.configure(stage.camera)

    def consume(envelope) -> dict:
        rec = stage.process_frame(envelope)

        try:
            keys = overlay_keys(envelope)

            if gate.wanted():
                frame = envelope.observations.frame
                items = [(o.result, o.T_W_C) for o in stage.last_frame if o.T_W_C is not None]
                gate.note(drawer.submit(frame.image, frame.ros_stamp_ns, frame.optical_frame, items, keys,
                                        guides=guides))
            else:
                gate.note(False)

            send_results = getattr(publisher, "publish_results", None)

            if send_results is not None:
                send_results(results_payload(rec, keys))
        except Exception as exc:  # noqa: BLE001 - 표시 실패가 B 계산·기록을 막지 않게
            consume.errors += 1

            if on_error is not None:
                on_error(exc)

        return rec

    consume.errors = 0
    return consume


__all__ = ["CHILD_MODULE", "DEFAULT_FACTORY", "CenterDrawProcess", "process_center_consumer"]
