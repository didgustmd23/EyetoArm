"""B center 그리기를 별도 프로세스로(#413): A 프로세스는 「그리기 요청」만 넘기고, 자식이 그려서 발행한다.

같은 프로세스에서 그리면 그리기(`render_center`)가 검출·추적과 GIL을 나눠 써서 A 전체가 느려진다(#372·#398).
그래서 B 계산·`/b/results`·수락·목표는 A에 남기고, 그리기와 center 영상 발행만 자식 프로세스로 보낸다.

- 넘기는 것: 사진 한 장(이름 없는 공유 메모리 memfd 한 칸), 그 사진의 B 결과(`WorkPlanarResult`·`T_W_C`), 번호표.
- 기다리지 않는다: `submit`은 가장 새 요청 하나만 남기고 바로 돌아온다. 자식이 앞 사진을 그리는 중이면 그 사이에
  들어온 것 가운데 마지막만 그리고 나머지는 버린다(표시 전용이라 최신 우선, 버린 수를 센다).
- 요청은 자식이 준비(ready)를 알린 뒤에만 보낸다. 멈춤 시계는 그 뒤부터 돈다. 준비가 `ready_timeout_s` 안에
  끝나지 않으면 그때 그 자식을 잃은 것으로 본다(부하가 큰 기계의 느린 준비를 멈춤으로 오인하지 않게, r2 D3).
- **자식을 잃으면 다시 띄운다(#432).** 죽음(연결 끊김)·멈춤(답 없음)·준비 실패·준비 한도 초과 때, 그 자식을 끝내고
  `restart_delays_s`만큼(점점 늘림) 기다렸다가 새 자식을 띄우고 카메라 값을 다시 보낸다. 한도(간격 수)를 다 쓰면
  그만두고(`failed`에 사유, `on_failed`로 한 번 알림) center 영상만 멈춘다. 새 자식이 `healthy_s` 넘게 정상이었으면
  다음에 잃을 때 횟수를 다시 센다. 처음 띄우기 실패는 다시 시도하지 않는다(`start`가 OSError로 바로 알림).
  다시 띄우는 동안 `submit`은 거절로 바로 돌아온다. A의 다른 일은 영향받지 않는다.
- 시간은 모두 단조 시계(`time.monotonic`)로 잰다. 시스템 시계는 WSL에서 32초마다 2초씩 되돌아가므로 쓰지 않는다(#437).
- 통로는 전용 socketpair다. pickle은 이 부모가 띄운 자식과의 연결에서만 쓴다(#259 추론 프로세스와 같은 규칙).
- 공유 메모리는 이름이 없어(memfd) 부모·자식이 어떻게 끝나도 파일이 남지 않는다. 자식마다 새로 만들고 옛 것은 닫는다.

스레드: `submit`·`configure`는 B worker 스레드가, 보내기는 `eyetoarm-center-send`(하나), 받기는 자식마다
`eyetoarm-center-recv`, 다시 띄우기는 `eyetoarm-center-supervisor`가 한다. 넷이 나눠 쓰는 상태는 모두 `_cond`(잠금)로
보호하고, 지금 자식의 연결·공유 메모리를 쓰고 바꾸는 일은 `_send_lock`으로 한 번에 하나만 한다.
B 파일은 바꾸지 않는다. 자식 안에서 B 공개 함수(`BStage.render_center`·`CenterOverlayPublisher`)를 그대로 부른다.
"""
from __future__ import annotations

import collections
import mmap
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from typing import Callable

import numpy as np

CHILD_MODULE = "eyetoarm.vision.center_process_child"
DEFAULT_FACTORY = "eyetoarm.vision.center_process_child:ros_backend"
STALL_S = 5.0          # 준비된 자식이 한 장을 이보다 오래 붙들면 멈춘 것으로 본다(준비 전에는 재지 않음)
READY_TIMEOUT_S = 60.0  # 자식이 띄운 뒤 이 안에 준비(ready)를 알리지 않으면 잃은 것으로 본다
STOP_GRACE_S = 2.0     # stop 메시지 뒤 스스로 끝나기를 기다리는 시간
KILL_GRACE_S = 2.0     # SIGTERM 뒤 SIGKILL까지
# 다시 띄우기 간격(#432): 준비는 보통 0.3~0.75 s(#413 실측). 2 s는 죽은 자식의 ROS 노드가 사라질 틈, 5·10 s는 같은 원인이
# 되풀이될 때 덜 두드리려고. 셋을 다 써도 17 s(집기 한 번 약 34 s보다 짧음). 간격 수가 곧 횟수 한도다.
RESTART_DELAYS_S = (2.0, 5.0, 10.0)
HEALTHY_S = 60.0       # 새 자식이 이만큼 정상으로 돌면(집기 두 번 안팎) 다음에 잃을 때 횟수를 다시 센다


@dataclass
class _Child:
    """자식 하나의 자원. `_send_lock`을 쥔 쪽만 `conn`·`mem`을 쓰거나 바꾼다."""

    proc: subprocess.Popen
    conn: Connection
    mem_fd: int
    gen: int
    mem: mmap.mmap | None = None
    mem_size: int = 0
    recv_thread: threading.Thread | None = None
    started_at: float = 0.0
    returncodes: list = field(default_factory=list)


class CenterDrawProcess:
    """center 그리기 자식 프로세스의 부모 쪽 대리. `start` → (`configure`·`submit` 여러 번) → `close`."""

    def __init__(self, root, *, factory: str = DEFAULT_FACTORY, python: str | None = None, env: dict | None = None,
                 child_log: str | None = None, stall_s: float = STALL_S, ready_timeout_s: float = READY_TIMEOUT_S,
                 stop_grace_s: float = STOP_GRACE_S, kill_grace_s: float = KILL_GRACE_S,
                 restart_delays_s=RESTART_DELAYS_S, healthy_s: float = HEALTHY_S,
                 on_failed: Callable[[str], None] | None = None, on_notice: Callable[[str], None] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        """`root`는 저장소 맨 위 폴더(자식이 B 설정을 읽는 곳). `factory`·`python`·`env`·`clock`은 시험 주입용.

        `restart_delays_s`는 다시 띄우기 전 기다림(초) 목록이고 그 길이가 횟수 한도다(빈 목록이면 다시 띄우지 않음).
        `on_failed(사유)`는 더는 다시 띄우지 않기로 한 때 한 번, `on_notice(줄)`은 다시 띄울 때·다시 뜬 때 한 줄씩
        불린다(받기·보내기·다시 띄우기 스레드에서, 잠금 밖에서).
        """
        self._root = str(root)
        self._factory = factory
        self._python = python or sys.executable
        self._env = dict(os.environ if env is None else env)
        self._child_log = child_log
        self._stall_s, self._stop_grace_s, self._kill_grace_s = float(stall_s), float(stop_grace_s), float(kill_grace_s)
        self._ready_timeout_s = float(ready_timeout_s)
        self._delays = tuple(float(d) for d in restart_delays_s)

        if any(d < 0 for d in self._delays):
            raise ValueError(f"restart_delays_s must be >= 0: {restart_delays_s!r}")

        self._healthy_s = float(healthy_s)
        self._on_failed = on_failed
        self._on_notice = on_notice
        self._clock = clock

        self._cond = threading.Condition()
        self._send_lock = threading.Lock()
        self._log_fh = None
        self._threads: list[threading.Thread] = []

        # 아래는 _cond로 보호한다(_child는 바꿀 때 _send_lock도 쥔다)
        self._child: _Child | None = None
        self._gen = 0
        self._pending: dict | None = None      # 아직 안 보낸 가장 새 요청
        self._camera = None                    # 아직 안 보낸 카메라 설정
        self._camera_last = None               # 마지막 카메라 설정(다시 띄운 자식에 다시 보냄)
        self._busy_since: float | None = None  # 자식이 그리는 중이면 보낸 시각
        self._stopping = False
        self._closed = False
        self._ready: dict | None = None
        self._ready_at: float | None = None
        self._down: str | None = None          # 지금 자식을 잃은 사유(다시 띄우기를 기다리는 중)
        self._restart_due: float | None = None
        self._attempt = 0                      # 마지막 정상 구간 뒤 다시 띄운 횟수
        self._failed: str | None = None
        self._exit: dict | None = None
        self._req = 0
        self._counts = {"requested": 0, "sent": 0, "drawn": 0, "replaced": 0, "refused": 0, "child_errors": 0,
                        "send_errors": 0, "restarts": 0}
        self._restart_reasons = collections.deque(maxlen=20)
        self._child_exits = collections.deque(maxlen=20)
        self._first_child_error: str | None = None
        self._render_ms: list[float] = []
        self._publish_ms: list[float] = []

    # ------------------------------------------------------------------ 상태

    @property
    def failed(self) -> str | None:
        """더는 다시 띄우지 않기로 한 사유(정상·다시 띄우는 중이면 None)."""
        with self._cond:
            return self._failed

    @property
    def pid(self) -> int | None:
        """지금 자식 pid(안 띄웠으면 None)."""
        with self._cond:
            return None if self._child is None else self._child.proc.pid

    @property
    def _mem_fd(self) -> int | None:
        """지금 자식의 공유 메모리 번호(시험·진단용, 닫혔으면 None)."""
        with self._cond:
            return None if self._child is None else self._child.mem_fd

    def summary(self) -> dict:
        """요약 블록: 요청·보냄·그림·버림 수, 자식 오류, 다시 띄운 횟수·사유, 실패 사유, 그리기·발행 시간."""
        with self._cond:
            render, publish = sorted(self._render_ms), sorted(self._publish_ms)
            out = {"pid": None if self._child is None else self._child.proc.pid, "factory": self._factory,
                   **self._counts, "failed": self._failed, "first_child_error": self._first_child_error,
                   "ready": self._ready, "exit": self._exit, "waiting_restart": self._restart_due is not None,
                   "restart_delays_s": list(self._delays), "healthy_s": self._healthy_s,
                   "restart_reasons": list(self._restart_reasons), "child_exits": list(self._child_exits)}

        def mid(values, q):
            """분위수(없으면 None)."""
            return None if not values else round(values[min(len(values) - 1, int(q * len(values)))], 3)

        out["render_ms"] = {"p50": mid(render, .5), "p95": mid(render, .95)}
        out["publish_ms"] = {"p50": mid(publish, .5), "p95": mid(publish, .95)}

        return out

    # ------------------------------------------------------------------ 시작·다시 띄우기

    def start(self) -> None:
        """첫 자식을 띄우고 보내기·다시 띄우기 스레드를 시작한다. 띄우지 못하면 OSError(다시 시도하지 않음)."""
        with self._cond:
            if self._child is not None or self._gen:
                raise RuntimeError("CenterDrawProcess.start() called twice")

        if self._child_log is not None:
            self._log_fh = open(self._child_log, "ab")

        try:
            child = self._spawn()
        except OSError as exc:
            self._give_up(f"start failed: {type(exc).__name__}: {exc}", notify=True)
            self._release_log()
            raise

        with self._cond, self._send_lock:
            self._child = child

        for name, target in (("eyetoarm-center-send", self._send_loop),
                             ("eyetoarm-center-supervisor", self._supervise_loop)):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

    def _spawn(self) -> _Child:
        """새 자식 하나(연결·공유 메모리·받기 스레드). 실패하면 만든 것을 닫고 OSError."""
        parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        mem_fd = None

        try:
            mem_fd = os.memfd_create("eyetoarm_center", os.MFD_CLOEXEC)
            cmd = [self._python, "-m", CHILD_MODULE, "--fd", str(child_sock.fileno()), "--mem-fd", str(mem_fd),
                   "--factory", self._factory, "--parent-pid", str(os.getpid()), "--root", self._root]
            proc = subprocess.Popen(cmd, pass_fds=[child_sock.fileno(), mem_fd], env=self._env,
                                    stdin=subprocess.DEVNULL, stdout=self._log_fh, stderr=self._log_fh)
        except OSError:
            parent_sock.close()

            if mem_fd is not None:
                os.close(mem_fd)

            raise
        finally:
            child_sock.close()

        with self._cond:
            self._gen += 1
            gen = self._gen
            self._ready, self._ready_at, self._busy_since = None, None, None
            self._camera = self._camera_last   # 다시 띄운 자식에도 카메라 값을 보낸다

        child = _Child(proc=proc, conn=Connection(parent_sock.detach()), mem_fd=mem_fd, gen=gen,
                       started_at=self._clock())
        child.recv_thread = threading.Thread(target=self._recv_loop, args=(child,), name="eyetoarm-center-recv",
                                             daemon=True)
        child.recv_thread.start()

        with self._cond:
            self._threads.append(child.recv_thread)

        return child

    def _lost(self, gen: int, reason: str) -> None:
        """자식 `gen`을 잃음. 지금 자식이고 닫는 중이 아니면 다시 띄우기를 잡거나(한도 안) 그만둔다."""
        notice = None

        with self._cond:
            if self._stopping or self._failed is not None or self._down is not None or gen != self._gen:
                return

            lost_child = self._child
            now = self._clock()

            if self._ready_at is not None and now - self._ready_at >= self._healthy_s:
                self._attempt = 0   # 한참 정상이던 자식: 띄우자마자 죽는 것과 구분해 횟수를 다시 센다

            self._down = reason
            self._pending = None
            self._busy_since = None
            self._restart_reasons.append({"t_mono_s": round(now, 3), "reason": reason, "attempt": self._attempt + 1})

            if self._attempt < len(self._delays):
                delay = self._delays[self._attempt]
                self._attempt += 1
                self._restart_due = now + delay
                notice = (f"center draw process lost: {reason}; restarting in {delay:g} s "
                          f"(attempt {self._attempt}/{len(self._delays)})")

            self._cond.notify_all()

        if notice is None:
            self._give_up(reason, notify=True)
            self._retire(lost_child)   # 다시 띄우지 않으므로 여기서 끝낸다(멈춘 자식이 남지 않게)
        else:
            self._notify(notice)

    def _give_up(self, reason: str, *, notify: bool) -> None:
        """더는 다시 띄우지 않는다. 처음 한 번만 `on_failed`를 부른다(잠금 밖에서)."""
        with self._cond:
            first = self._failed is None and not self._stopping

            if self._failed is None:
                self._failed = reason

            self._restart_due = None
            self._pending = None
            self._cond.notify_all()

        if first and notify and self._on_failed is not None:
            try:
                self._on_failed(reason)
            except Exception as exc:   # noqa: BLE001 - 알림 실패가 스레드를 죽이지 않게(요약에는 사유가 남음)
                print(f"center_process: on_failed callback failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    def _notify(self, line: str) -> None:
        """안내 한 줄(잠금 밖에서). 알림 실패는 표준 오류에만 남긴다."""
        if self._on_notice is None:
            return

        try:
            self._on_notice(line)
        except Exception as exc:   # noqa: BLE001 - 알림 실패가 다시 띄우기를 막지 않게
            print(f"center_process: on_notice callback failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    def _supervise_loop(self) -> None:
        """다시 띄우기 스레드: 잃은 자식을 끝내고, 기다림이 지나면 새 자식을 띄운다."""
        while True:
            with self._cond:
                while not self._stopping and self._failed is None and self._down is None:
                    self._cond.wait(0.5)

                if self._stopping or self._failed is not None:
                    return

                old = self._child

            # 잃은 자식을 먼저 끝내고 자원을 닫는다(기다림 동안 남아 있지 않게)
            self._retire(old)

            with self._cond:
                while not self._stopping and self._restart_due is not None and self._clock() < self._restart_due:
                    self._cond.wait(min(0.2, max(0.0, self._restart_due - self._clock())))

                if self._stopping or self._failed is not None:
                    return

            try:
                child = self._spawn()
            except OSError as exc:
                with self._cond:
                    self._down = None
                    self._restart_due = None
                    gen = self._gen

                self._lost(gen, f"restart failed: {type(exc).__name__}: {exc}")
                continue

            with self._cond:
                if self._stopping:   # 띄우는 사이 닫기가 시작됨: 이 자식도 닫기가 끝낸다
                    stale = True
                else:
                    stale = False

                with self._send_lock:
                    self._child = child

                self._down = None
                self._restart_due = None
                self._counts["restarts"] += 1
                self._cond.notify_all()

            if stale:
                return

    def _retire(self, child: _Child | None) -> None:
        """자식 하나를 끝내고(SIGTERM → SIGKILL) 연결·공유 메모리를 닫는다. 이미 끝났으면 자원만 닫는다."""
        if child is None:
            return

        self._terminate(child.proc)

        with self._send_lock:
            self._close_child_resources(child)

        if child.recv_thread is not None and child.recv_thread is not threading.current_thread():
            child.recv_thread.join(1.0)

        with self._cond:
            self._child_exits.append({"pid": child.proc.pid, "returncode": child.proc.returncode})

    # ------------------------------------------------------------------ 요청

    def configure(self, camera) -> None:
        """그리기에 쓸 카메라 값(B `CameraProfile`)을 자식에 알린다. 단계가 새로 만들어질 때마다 부른다."""
        with self._cond:
            self._camera = camera
            self._camera_last = camera
            self._cond.notify_all()

    def submit(self, image: np.ndarray, stamp_ns: int | None, optical_frame: str | None, items: list, keys: dict,
               *, guides: bool = True) -> bool:
        """그리기 요청 하나를 맡긴다(기다리지 않음). 맡았으면 True, 자식을 못 쓰면(다시 띄우는 중 포함) False.

        `items`는 그릴 결과 `(WorkPlanarResult, T_W_C)` 목록, `keys`는 번호표(`overlay_keys`).
        앞 요청이 아직 안 나갔으면 그것을 버리고 이 요청으로 바꾼다.
        """
        with self._cond:
            if self._failed is not None or self._stopping or self._child is None or self._down is not None:
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

    # ------------------------------------------------------------------ 보내기·받기

    def _send_loop(self) -> None:
        """보내기 스레드: 지금 자식이 준비된 뒤, 한가해지면 가장 새 요청 하나를 공유 메모리에 싣고 보낸다.

        멈춤(준비된 자식이 한 장을 stall_s 넘게 붙듦)과 준비 지연(ready_timeout_s 안에 준비를 알리지 않음)도 여기서 본다.
        자식을 잃은 동안은 기다린다(다시 띄우기는 감독 스레드가 한다).
        """
        while True:
            with self._cond:
                while True:
                    if self._stopping or self._failed is not None:
                        return

                    child = self._child
                    now = self._clock()

                    if self._down is not None or child is None:
                        pass
                    elif self._ready is None:
                        if now - child.started_at > self._ready_timeout_s:
                            lost = f"child not ready after {self._ready_timeout_s} s"
                            break
                    elif self._busy_since is not None and now - self._busy_since > self._stall_s:
                        lost = f"child stalled: no answer for {self._stall_s} s"
                        break
                    elif self._busy_since is None and (self._camera is not None or self._pending is not None):
                        lost = None
                        break

                    self._cond.wait(0.2)

                camera, job = (None, None) if lost else (self._camera, self._pending)
                gen = child.gen

                if lost is None:
                    self._camera = None

                    if job is not None:
                        self._pending = None
                        self._busy_since = self._clock()

            if lost:
                self._lost(gen, lost)
                continue

            try:
                with self._send_lock:
                    if self._child is not child:   # 그 사이 자식이 바뀜: 이 요청은 버린다
                        continue

                    if camera is not None:
                        child.conn.send(("config", {"camera": camera}))

                    if job is not None:
                        child.conn.send(("draw", self._load(child, job)))

                if job is not None:
                    with self._cond:
                        self._counts["sent"] += 1
            except (OSError, ValueError) as exc:   # 끊긴 연결·영상 형식 오류: 이 요청만 잃고, 연결이 죽었으면 받기 쪽이 잃음으로 돌린다
                with self._cond:
                    self._counts["send_errors"] += 1
                    self._busy_since = None

                    if self._first_child_error is None:
                        self._first_child_error = f"send: {type(exc).__name__}: {exc}"

    def _load(self, child: _Child, job: dict) -> dict:
        """사진을 그 자식의 공유 메모리에 복사하고 자식에 보낼 메시지(영상 빼고)를 만든다. `_send_lock`을 쥔 채 부른다."""
        image = np.ascontiguousarray(job["image"])

        if image.dtype != np.uint8 or image.ndim != 3:
            raise ValueError(f"center image must be uint8 HxWxC: dtype={image.dtype} shape={image.shape}")

        if image.nbytes > child.mem_size:
            if child.mem is not None:
                child.mem.close()

            os.ftruncate(child.mem_fd, image.nbytes)
            child.mem = mmap.mmap(child.mem_fd, image.nbytes)
            child.mem_size = image.nbytes

        np.frombuffer(child.mem, dtype=np.uint8, count=image.nbytes)[:] = image.reshape(-1)

        return {"req": job["req"], "shape": tuple(image.shape), "nbytes": image.nbytes, "stamp_ns": job["stamp_ns"],
                "optical_frame": job["optical_frame"], "items": job["items"], "keys": job["keys"],
                "guides": job["guides"]}

    def _recv_loop(self, child: _Child) -> None:
        """받기 스레드(자식마다): 답(`ready`·`done`·`failed`)을 반영한다. 연결이 끊기면 그 자식을 잃은 것으로 돌린다."""
        while True:
            try:
                kind, payload = child.conn.recv()
            except (EOFError, OSError):
                try:   # 끊긴 직후라 아직 종료 코드가 안 잡혔을 수 있다: 잠깐만 기다려 사유에 남긴다
                    rc = child.proc.wait(0.5)
                except subprocess.TimeoutExpired:
                    rc = None

                self._lost(child.gen, f"child connection lost (returncode={rc})")
                return
            except Exception as exc:   # 깨진 답(풀 수 없는 pickle 등): 이 자식은 더 믿지 않는다
                self._lost(child.gen, f"bad reply: {type(exc).__name__}: {exc}")
                return

            notice = None

            with self._cond:
                if child.gen != self._gen:   # 옛 자식의 늦은 답: 버린다
                    continue

                if kind == "ready":
                    self._ready = dict(payload)
                    self._ready_at = self._clock()

                    if self._counts["restarts"]:
                        notice = (f"center draw process restarted pid={child.proc.pid} "
                                  f"(attempt {self._attempt}/{len(self._delays)})")
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

            if notice is not None:
                self._notify(notice)

            if kind == "failed":   # 자식이 준비·설정에 실패했다고 알림: 곧 끝난다
                self._lost(child.gen, f"child failed: {payload.get('cause')}")

    # ------------------------------------------------------------------ 종료

    def _terminate(self, proc: subprocess.Popen) -> None:
        """자식을 SIGTERM → kill_grace_s → SIGKILL로 끝낸다(이미 끝났으면 아무것도 안 함)."""
        if proc.poll() is not None:
            return

        proc.terminate()

        try:
            proc.wait(self._kill_grace_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(self._kill_grace_s)

    def _close_child_resources(self, child: _Child) -> None:
        """자식 하나의 연결·공유 메모리를 닫는다. 하나가 실패해도 나머지를 한다. `_send_lock`을 쥔 채 부른다."""
        try:
            child.conn.close()
        except OSError:   # 이미 닫힘
            pass

        if child.mem is not None:
            try:
                child.mem.close()
            except (BufferError, ValueError):   # 아직 쥔 배열이 있음: fd를 닫으면 프로세스가 끝날 때 함께 풀린다
                pass

            child.mem = None

        if child.mem_fd is not None:
            try:
                os.close(child.mem_fd)
            except OSError:   # 이미 닫힘
                pass

            child.mem_fd = None

    def _release_log(self) -> None:
        """자식 기록 파일을 닫는다."""
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None

    def close(self) -> dict:
        """지금 자식에 종료를 알리고(stop_grace_s), 안 끝나면 SIGTERM·SIGKILL. 다시 띄우기를 멈추고 스레드·연결·공유 메모리를
        정리한다. 다시 띄우기를 기다리는 중이어도 새 자식은 띄우지 않는다. 두 번 불러도 된다."""
        with self._cond:
            if self._closed:
                return dict(self._exit or {})

            self._closed = True
            self._stopping = True
            self._pending = None
            self._restart_due = None
            self._cond.notify_all()

        # 다시 띄우기 스레드가 새 자식을 들이는 중이면 끝날 때까지 기다린다(그 자식도 아래에서 닫는다)
        for thread in self._threads:
            if thread.name == "eyetoarm-center-supervisor":
                thread.join(self._kill_grace_s * 2 + self._stop_grace_s)

        with self._cond:
            child = self._child

        forced = None

        # 자식을 기다리다 KeyboardInterrupt가 나도 연결·공유 메모리 정리는 반드시 한다
        try:
            if child is not None:
                proc = child.proc

                if proc.poll() is None:
                    try:
                        with self._send_lock:
                            child.conn.send(("stop", {}))
                    except OSError:   # 이미 끊긴 자식: 아래에서 신호로 끝낸다
                        pass

                try:
                    proc.wait(self._stop_grace_s)
                    forced = False
                except subprocess.TimeoutExpired:
                    forced = True
                    self._terminate(proc)
        finally:
            if child is not None:
                with self._send_lock:
                    self._close_child_resources(child)

            for thread in list(self._threads):
                thread.join(1.0)

            self._release_log()

        exit_info = {"returncode": None if child is None else child.proc.returncode, "forced": forced}

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


__all__ = ["CHILD_MODULE", "DEFAULT_FACTORY", "HEALTHY_S", "RESTART_DELAYS_S", "CenterDrawProcess",
           "process_center_consumer"]
