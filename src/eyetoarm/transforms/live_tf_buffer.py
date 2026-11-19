"""실시간 TF 버퍼 부품(#171 R6, **B 담당 검토 대상 제안**): B 소유 노드·Buffer·Listener·전용 executor, 정확 stamp 조회, 되감기 준비 판정.

근거: `_workspace/171_builder2_live_a_to_b_plan.md` r3 §2(K-T 권고 T2)·§3.5·§4.1, `transforms/b_stage.py`(`RosTfLookup` 시각 의미).
B 파일 `b_stage.py`는 바꾸지 않고 새 파일로 둔다. **실시간 실행 경로에 연결하지 않는다**(D13 유지).

- `LiveTfRuntime(context, modules=)`: `start()`에서만 rclpy·tf2_ros를 불러(코드 규약 §8) `use_sim_time=True` 노드 → `Buffer(node=…)` →
  `TransformListener(buffer, node, spin_thread=False)` → 전용 `SingleThreadedExecutor` + 스레드 1개를 만든다. 컨텍스트는 주입받고
  종료하지 않는다. `close(limit_s)`는 executor 종료 → 스레드 join → 노드 해제 순서로, 단계 실패를 모아 마지막에 RuntimeError.
  `modules`(시험 주입): `{"rclpy", "Buffer", "TransformListener", "SingleThreadedExecutor", "Time", "Duration", "Parameter"}`.
- `lookup(parent, child, stamp_ns, timeout_s) → (T 4×4, stamp_ns)`: `Time(nanoseconds=stamp)` 정확 조회. 최신(`Time(0)`) 대체 없음.
  `vision.live_observation_envelope.LiveCaptureTfLookup`의 주입 함수 형식이다.
- `TfRewindReadiness`: 되감기 경계의 TF 준비(§4.1 2~4). `clear` → 정적 사슬 확인 → 경계 뒤 첫 프레임 stamp 정확 조회 성공이면
  `state.on_tf_ready(boundary_id)`. 순수 로직이며 조회·정리 함수는 주입한다. `observe_tf_stamp`는 준비 뒤 미래 stamp TF(이전 실행 잔여
  의심, §4.1-5)를 `state.on_tf_anomaly`로 알린다(휴리스틱, `/tf` 메시지 탭 연결은 아직 없음).
"""
from __future__ import annotations

import threading
from typing import Callable


from eyetoarm.transforms.b_stage import transform_to_matrix


def _ros_modules() -> dict:
    """실제 rclpy·tf2_ros(이 함수 안에서만 import)."""
    import rclpy
    from rclpy.duration import Duration
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.parameter import Parameter
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformListener

    return {"rclpy": rclpy, "Buffer": Buffer, "TransformListener": TransformListener,
            "SingleThreadedExecutor": SingleThreadedExecutor, "Time": Time, "Duration": Duration, "Parameter": Parameter}


class LiveTfRuntime:
    """B 소유 TF 수신 묶음."""

    THREAD_NAME = "eyetoarm-b-tf"

    def __init__(self, context, *, node_name: str = "eyetoarm_b_tf", cache_s: float = 30.0,
                 static_pairs: tuple = (), modules: dict | None = None) -> None:
        """주입 컨텍스트(종료하지 않음)·노드 이름·버퍼 보관 시간·정적 확인 허용 쌍(B4)·시험용 모듈."""
        self._context = context
        self._static_pairs = frozenset(tuple(p) for p in static_pairs)
        self.static_refused = 0
        self._node_name = node_name
        self._cache_s = cache_s
        self._m = modules
        self.node = None
        self.buffer = None
        self.listener = None
        self._executor = None
        self._thread: threading.Thread | None = None
        self._closing = False
        self.spin_error: str | None = None
        self._on_failure: Callable[[str], None] | None = None
        self._failure: str | None = None   # 실패로 알린(알릴) 사유. 정리 중 종료는 실패가 아니라 None으로 남는다(#196 r1 L1)
        self._failure_lock = threading.Lock()
        self._done: set[str] = set()

    def set_failure_callback(self, callback: Callable[[str], None] | None) -> None:
        """수신 스레드 실패 알림 함수(#196 A1-2). 스레드가 예외로 끝나거나 close 전에 반환하면 사유 문자열로 **TF 스레드에서** 한 번
        부른다. 이미 실패했으면 등록 즉시 부른다(등록 전 사망을 놓치지 않음). 호출 쪽은 잠금만 짧게 잡아야 한다."""
        with self._failure_lock:
            self._on_failure = callback
            error = self._failure

        if callback is not None and error is not None:
            callback(error)

    def start(self) -> None:
        """노드 → Buffer → Listener → executor 스레드. 실패하면 만든 것까지 정리하고 예외를 올린다."""
        m = self._m or _ros_modules()
        self._m = m

        try:
            self.node = m["rclpy"].create_node(self._node_name, context=self._context,
                                               parameter_overrides=[m["Parameter"]("use_sim_time", value=True)])
            self.buffer = m["Buffer"](cache_time=m["Duration"](seconds=self._cache_s), node=self.node)
            self.listener = m["TransformListener"](self.buffer, self.node, spin_thread=False)
            self._executor = m["SingleThreadedExecutor"](context=self._context)
            self._executor.add_node(self.node)
            self._thread = threading.Thread(target=self._spin, name=self.THREAD_NAME, daemon=True)
            self._thread.start()
        except BaseException:
            try:
                self.close(1.0)
            except RuntimeError:
                pass
            raise

    def _spin(self) -> None:
        """executor를 돌린다. 정리 요청(`close`) 없이 끝나면 예외든 반환이든 실패로 남기고 알린다(#196 A1-2).

        영상 수신기(`RclpyTransport._spin`)와 같은 규칙(#196 r1 L1): 정리 중에 끝나면 예외여도 `spin_error`에 원인만 남기고 알리지 않는다.
        """
        error = None

        try:
            self._executor.spin()
        except BaseException as exc:  # 스레드 밖으로 내보내지 않는다(아래에서 사유로 남김·알림)
            error = f"{type(exc).__name__}: {exc}"

        with self._failure_lock:
            self.spin_error = error

            if self._closing:
                return

            error = error or "spin returned before close"
            self.spin_error = error
            self._failure = error
            callback = self._on_failure

        if callback is None:
            return

        try:
            callback(error)
        except Exception as exc:  # noqa: BLE001 - 알림 실패도 사유에 남긴다(스레드 최상단, 규약 §10)
            with self._failure_lock:
                self.spin_error = f"{error}; failure callback failed: {type(exc).__name__}: {exc}"

    def lookup(self, parent: str, child: str, stamp_ns: int, timeout_s: float):
        """정확 stamp 조회. 실패는 tf2 예외 그대로(호출자가 None으로 바꾼다).

        B3(#173): stamp가 양의 정수가 아니면 Buffer를 부르기 전에 ValueError(시각 0 = 최신 조회 방지).
        """
        if type(stamp_ns) is not int or stamp_ns <= 0:
            raise ValueError(f"stamp_ns must be a positive int (0 means latest in tf2): {stamp_ns!r}")

        m = self._m
        tf = self.buffer.lookup_transform(parent, child, m["Time"](nanoseconds=stamp_ns),
                                          timeout=m["Duration"](seconds=timeout_s))
        stamp = tf.header.stamp
        got = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

        return transform_to_matrix(tf.transform), got

    def has_static(self, parent: str, child: str) -> bool:
        """정적 구간이 버퍼에 있는가(시각 0, 대기 없음 — 정적 변환은 시각과 무관).

        B4(#173): 시각 0(`Time()`)은 생성 때 등록한 정적 쌍(`static_pairs`)에만 쓴다. 목록 밖(동적 사슬 포함)은 확인하지 않고 False
        (`static_refused` 계수). 허용 목록 기본값은 비어 있다.
        """
        if (parent, child) not in self._static_pairs:
            self.static_refused += 1
            return False

        try:
            return bool(self.buffer.can_transform(parent, child, self._m["Time"]()))
        except Exception:
            return False

    def clear(self) -> None:
        """버퍼 비우기."""
        self.buffer.clear()

    def close(self, limit_s: float) -> None:
        """executor 종료 → 스레드 join → 노드 해제. 실패 단계는 모아서 RuntimeError, 성공 단계는 다시 하지 않는다."""
        self._closing = True
        errors = []

        def step(name, fn):
            if name in self._done:
                return
            try:
                fn()
                self._done.add(name)
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")

        if self._executor is not None:
            step("executor", lambda: self._executor.shutdown(timeout_sec=limit_s))

        if self._thread is not None:
            def join():
                self._thread.join(timeout=limit_s)
                if self._thread.is_alive():
                    raise RuntimeError("tf thread did not stop")
            step("thread", join)

        if self.node is not None:
            step("node", self.node.destroy_node)

        if errors:
            raise RuntimeError("; ".join(errors))


class TfRewindReadiness:
    """되감기 경계의 TF 준비 판정(§4.1). 조회·정리·정적 확인 함수는 주입.

    #172 r1 F1: 준비는 **시도 단위**다. `begin(boundary_id)`가 `LiveObservationState.begin_tf_attempt`로 시도 번호를 받은 **뒤** 버퍼를 비운다.
    `try_ready`는 그 시도 번호로만 `on_tf_ready`를 부르므로, 조회 도중 `observe_tf_stamp` 이상이 시도 번호를 올리면 조회가 정확 stamp를
    돌려줘도 열리지 않는다(같은 잠금에서 판정). 다시 열려면 `begin`(새 시도 = 버퍼 정리)과 정적·동적 재확인이 필요하다.
    """

    def __init__(self, state, *, clear: Callable[[], None], has_static: Callable[[str, str], bool],
                 lookup: Callable, world_frame: str, static_pairs: tuple, wait_limit_s: float,
                 future_tolerance_ns: int) -> None:
        """`static_pairs`: 확인할 정적 (parent, child) 쌍. 값들은 결정 전 시험 후보다."""
        self.state = state
        self._clear = clear
        self._has_static = has_static
        self._lookup = lookup
        self.world_frame = world_frame
        self.static_pairs = tuple(static_pairs)
        self.wait_limit_s = wait_limit_s
        self.future_tolerance_ns = int(future_tolerance_ns)
        self._attempt: tuple[int, int] | None = None   # (boundary_id, attempt_id)
        self._lock = threading.Lock()
        self.ready_boundary: int | None = None
        self.counts: dict[str, int] = {}

    def _count(self, key: str) -> None:
        with self._lock:
            self.counts[key] = self.counts.get(key, 0) + 1

    def begin(self, boundary_id: int) -> str:
        """새 준비 시도: 시도 번호를 먼저 받고(이후 이상은 이 시도를 무효로 함) 버퍼를 비운다."""
        attempt = self.state.begin_tf_attempt(boundary_id)

        if attempt is None:
            self._count("not_begun")
            return "not_begun"

        with self._lock:
            self._attempt = (boundary_id, attempt)
            self.ready_boundary = None

        self._clear()
        self._count("clear")

        return "begun"

    def try_ready(self, boundary_id: int, first_frame) -> str:
        """정적 사슬 → 첫 프레임 stamp 정확 조회 → 그 시도 번호로 `on_tf_ready`. 상태 문자열을 돌려준다."""
        with self._lock:
            attempt = self._attempt

        if attempt is None or attempt[0] != boundary_id:
            self._count("not_begun")
            return "not_begun"

        stamp = getattr(first_frame, "ros_stamp_ns", None)
        optical = getattr(first_frame, "optical_frame", None)

        if stamp is None or not optical:
            self._count("no_stamp")
            return "no_stamp"

        # B2(#173): 양의 정수가 아닌 stamp는 조회·준비 완료 없음(시각 0 = 최신 조회 방지)
        if type(stamp) is not int or stamp <= 0:
            self._count("invalid_stamp")
            return "invalid_stamp"

        # B4(#173): 정확 조회 대상(world → optical)을 정적 확인(시각 0)으로 대체하지 않는다
        if (self.world_frame, optical) in self.static_pairs:
            self._count("static_pair_is_dynamic_target")
            return "static_pair_is_dynamic_target"

        missing = [p for p in self.static_pairs if not self._has_static(*p)]

        if missing:
            self._count("static_missing")
            return f"static_missing:{missing}"

        try:
            _T, got = self._lookup(self.world_frame, first_frame.optical_frame, stamp, self.wait_limit_s)
        except Exception as exc:
            self._count("dynamic_unavailable")
            return f"dynamic_unavailable:{type(exc).__name__}"

        if got != stamp:
            self._count("stamp_mismatch")
            return "stamp_mismatch"

        accepted = self.state.on_tf_ready(boundary_id, attempt[1])
        self._count("ready" if accepted else "ready_ignored")

        if accepted:
            with self._lock:
                self.ready_boundary = boundary_id

        return "ready" if accepted else "ready_ignored"

    def observe_tf_stamp(self, boundary_id: int, tf_stamp_ns: int, clock_ns: int) -> bool:
        """`/clock` + 허용치보다 큰 stamp TF: 이전 실행 잔여 의심 → 그 경계 TF 준비·진행 중 시도 무효(휴리스틱). 알렸으면 True."""
        if tf_stamp_ns > clock_ns + self.future_tolerance_ns:
            self._count("future_tf")

            with self._lock:
                self._attempt = None
                self.ready_boundary = None

            return self.state.on_tf_anomaly(boundary_id, f"tf stamp {tf_stamp_ns} > clock {clock_ns}")

        return False


__all__ = ["LiveTfRuntime", "TfRewindReadiness"]
