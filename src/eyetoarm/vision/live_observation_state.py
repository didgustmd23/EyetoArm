"""실시간 관측 현재성 상태: 단일 잠금으로 토큰·경계 준비·계산 권한·결과 수락·공통 순번 기록을 관리한다.

잠금 안에서는 비교·상태 전이·순번·등록만 하고 TF·봉투·B 계산·렌더·쓰기는 호출자가 잠금 밖에서 한다.
경계는 `applied`·`tf_ready`가 같은 `boundary_id`에서 모두 참일 때만 열리고, 용량 값은 기본값이 없다.
ROS·Qt를 import하지 않는다.
"""
from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Callable

SIM_TIME_REWIND = "sim_time_rewind"   # `BoundaryKind.SIM_TIME_REWIND` 값(inputs를 import하지 않음)
RUN_START = "run_start"               # 새 실행의 처음 경계(start_closed)


def rewind_requires_tf(kind) -> bool:
    """기본 규칙: 되감기 경계만 TF 준비 완료를 요구한다."""
    return getattr(kind, "value", kind) == SIM_TIME_REWIND


@dataclass(frozen=True)
class LiveObservationConfig:
    """쓰기 대기열 용량(기본값 없음, 호출자가 반드시 준다)."""

    write_capacity: int     # 수락 기록 일반 칸(W)
    reserved_capacity: int  # 무효화·철회·준비·종료 예약 칸(R)

    def __post_init__(self) -> None:
        for name in ("write_capacity", "reserved_capacity"):
            value = getattr(self, name)

            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive int: {value!r}")


@dataclass(frozen=True)
class LiveObservationToken:
    """현재성 토큰. 같은 `token_seq`면 같은 토큰이다."""

    run_id: str
    rx_epoch: int
    generation: int
    calibration_id: str | None
    ready: bool
    token_seq: int


@dataclass(frozen=True)
class BoundaryReadiness:
    """현재 경계의 준비 상태."""

    boundary_id: int
    kind: str
    applied: bool
    tf_ready: bool
    tf_attempt: int = 0   # 현재 TF 준비 시도 번호(0 = 시작 안 함)


@dataclass(frozen=True)
class LiveObservationEvent:
    """공통 순번 기록 하나. `payload`는 accept의 결과 묶음(호출자 소유, 잠금 밖에서 만든 것)."""

    event_seq: int
    kind: str                 # accept | invalidation | revoke | ready | run_end | failed
    token_seq: int
    enqueued_monotonic_ns: int
    cause: str | None = None
    compute_id: int | None = None
    revoked_event_seq: int | None = None
    payload: object | None = field(default=None, repr=False)


class LiveObservationState:
    """단일 잠금 현재성 상태. 메서드는 어느 스레드에서 불러도 되며, 모두 짧게 잠근다."""

    def __init__(self, *, run_id: str, generation: int, calibration_id: str | None, config: LiveObservationConfig,
                 rx_epoch: int = 0, tf_ready_required: Callable[[object], bool] = rewind_requires_tf,
                 clock_ns: Callable[[], int] = time.monotonic_ns, start_closed: bool = False) -> None:
        """처음 토큰은 열린 상태(경계 없음).

        `start_closed=True`(새 실행): 닫힌 토큰 + 경계 0(`run_start`, 반영·TF 준비 모두 필요).
        """
        if not isinstance(config, LiveObservationConfig):
            raise TypeError("config must be LiveObservationConfig")

        self._lock = threading.Lock()
        self._owner: int | None = None
        self._config = config
        self._tf_required = tf_ready_required
        self._clock_ns = clock_ns
        self._token = LiveObservationToken(run_id, rx_epoch, generation, calibration_id, not start_closed, 1)
        self._boundary: BoundaryReadiness | None = \
            BoundaryReadiness(rx_epoch, RUN_START, False, False) if start_closed else None
        self._computing: dict[int, dict] = {}   # compute_id → {"token_seq", "invalidated", "cause"}
        self._next_compute = 1
        self._event_seq = 0
        self._normal: list[LiveObservationEvent] = []     # 쓰기 대기(accept)
        self._reserved: list[LiveObservationEvent] = []   # 쓰기 대기(그 밖)
        self._order: list[LiveObservationEvent] = []      # 두 칸을 합친 순번 순서(꺼낼 때 이 순서)
        self._current_accept: int | None = None
        self._closed = False
        self._failed: str | None = None
        self._stop_hold: str | None = None
        self._opened_once = not start_closed
        self.counts: dict[str, int] = {}

    # ------------------------------------------------------------------ 잠금

    @contextlib.contextmanager
    def _locked(self):
        """잠금과 보유 스레드 표시."""
        with self._lock:
            self._owner = threading.get_ident()

            try:
                yield
            finally:
                self._owner = None

    def lock_held(self) -> bool:
        """지금 이 스레드가 잠금을 쥐고 있는가(시험의 잠금 규율 확인용)."""
        return self._owner == threading.get_ident()

    # ------------------------------------------------------------------ 조회

    @property
    def token(self) -> LiveObservationToken:
        """현재 토큰."""
        with self._locked():
            return self._token

    @property
    def boundary(self) -> BoundaryReadiness | None:
        """현재 경계 준비 상태."""
        with self._locked():
            return self._boundary

    @property
    def failed(self) -> str | None:
        """fail-stop 사유(없으면 None)."""
        with self._locked():
            return self._failed

    @property
    def closed(self) -> bool:
        """닫혔는가(run_end 또는 fail-stop)."""
        with self._locked():
            return self._closed

    def is_current(self, token: LiveObservationToken) -> bool:
        """그 토큰이 지금 열린 현재 토큰인가(검사 ①)."""
        with self._locked():
            return self._is_open(token)

    def is_current_accept(self, event_seq: int) -> bool:
        """그 수락이 지금 현재 결과인가(마지막 수락이고 그 뒤 무효화 없음)."""
        with self._locked():
            return self._current_accept is not None and self._current_accept == event_seq

    @property
    def stop_hold(self) -> str | None:
        """사전 정지 보류 사유(없으면 None)."""
        with self._locked():
            return self._stop_hold

    def pending_events(self) -> int:
        """쓰기 대기 수."""
        with self._locked():
            return len(self._order)

    # ------------------------------------------------------------------ 제출(제어 흐름)

    def admit(self, *, rx_epoch: int | None, generation: int, calibration_id: str | None) -> LiveObservationToken | None:
        """제출 시점 판정: 열려 있고 그 프레임의 수신 epoch·세대·보정 번호가 현재 토큰과 같을 때만 토큰을 돌려준다."""
        with self._locked():
            t = self._token

            if self._closed or not t.ready:
                self._count("submit_not_ready" if not self._closed else "submit_closed")
                return None

            if rx_epoch is None or rx_epoch != t.rx_epoch or generation != t.generation \
                    or calibration_id != t.calibration_id:
                self._count("submit_stale")
                return None

            return t

    # ------------------------------------------------------------------ 경계(수신 쪽·제어 흐름·TF 준비)

    def on_rx_boundary(self, boundary_id: int, kind) -> bool:
        """수신 경계(slot 잠금 안에서 부름). 새 경계를 만들고 닫는다. 이전·같은 번호는 무시(False)."""
        with self._locked():
            if self._closed:
                self._count("boundary_after_close")
                return False

            if self._boundary is not None and boundary_id <= self._boundary.boundary_id:
                self._count("boundary_not_monotonic")
                return False

            kind_s = str(getattr(kind, "value", kind))
            tf_ready = not self._tf_required(kind) and self._opened_once
            self._boundary = BoundaryReadiness(boundary_id, kind_s, False, tf_ready)
            self._set_token(replace(self._token, rx_epoch=boundary_id, ready=False), cause=f"rx_boundary:{kind_s}")

            return True

    def on_applied(self, boundary_id: int, generation: int, calibration_id: str | None) -> bool:
        """제어 반영 완료(반영 뒤 코어 현재 값). 이전 경계면 아무것도 바꾸지 않는다(False)."""
        with self._locked():
            if self._closed or self._boundary is None or boundary_id != self._boundary.boundary_id:
                self._count("stale_applied_ignored")
                return False

            if self._boundary.applied:
                self._count("duplicate_applied")
                return True

            self._boundary = replace(self._boundary, applied=True)

            if (generation, calibration_id) != (self._token.generation, self._token.calibration_id):
                self._set_token(replace(self._token, generation=generation, calibration_id=calibration_id),
                                cause=None)

            self._maybe_open()

            return True

    def begin_tf_attempt(self, boundary_id: int) -> int | None:
        """그 경계의 새 TF 준비 시도 번호. 이전 경계·닫힘·이미 TF 준비됨이면 None(상태 무변)."""
        with self._locked():
            if self._closed or self._boundary is None or boundary_id != self._boundary.boundary_id:
                self._count("stale_tf_attempt_ignored")
                return None

            if self._boundary.tf_ready:
                self._count("tf_attempt_not_needed")
                return None

            self._boundary = replace(self._boundary, tf_attempt=self._boundary.tf_attempt + 1)

            return self._boundary.tf_attempt

    def on_tf_ready(self, boundary_id: int, attempt_id: int) -> bool:
        """TF 준비 완료. 이전 경계이거나 현재 시도 번호가 아니면 아무것도 바꾸지 않는다(False)."""
        with self._locked():
            if self._closed or self._boundary is None or boundary_id != self._boundary.boundary_id:
                self._count("stale_tf_ready_ignored")
                return False

            if attempt_id != self._boundary.tf_attempt or attempt_id <= 0:
                self._count("stale_tf_attempt_ready_ignored")
                return False

            if self._boundary.tf_ready:
                self._count("duplicate_tf_ready")
                return True

            self._boundary = replace(self._boundary, tf_ready=True)
            self._maybe_open()

            return True

    def on_tf_anomaly(self, boundary_id: int, detail: str) -> bool:
        """이전 실행 잔여로 보이는 TF: TF 준비를 되돌리고 시도 번호를 올려 진행 중 시도를 무효로 하고 닫는다.

        다시 열려면 새 `begin_tf_attempt`와 그 시도의 `on_tf_ready`가 필요하다. 이전 경계면 무시.
        """
        with self._locked():
            if self._closed or self._boundary is None or boundary_id != self._boundary.boundary_id:
                self._count("stale_tf_anomaly_ignored")
                return False

            self._boundary = replace(self._boundary, tf_ready=False, tf_attempt=self._boundary.tf_attempt + 1)
            self._set_token(replace(self._token, ready=False), cause=f"tf_anomaly:{detail}")

            return True

    def hold_for_stop(self, detail: str) -> int:
        """사전 정지: 현재 결과 철회·진행 중 수락 차단·보류 표지를 한 잠금 구간에서 하고 토큰 순번을 준다.

        정지 확인 응답은 이 함수가 돌아온 뒤 보낸다. 이미 닫혔으면 보류 표지만 세운다.
        """
        with self._locked():
            self._stop_hold = detail

            if not self._closed:
                self._set_token(replace(self._token, ready=False), cause=f"stop_hold:{detail}")
            else:
                self._current_accept = None

            self._count("stop_hold")

            return self._token.token_seq

    def release_stop_hold(self, detail: str) -> bool:
        """사전 정지 보류를 푼다(새 경계 fence 확정 뒤). 경계가 준비됐으면 연다. 보류가 없으면 False."""
        with self._locked():
            if self._stop_hold is None:
                self._count("release_without_hold")
                return False

            self._stop_hold = None
            self._count("stop_hold_released")

            if not self._closed:
                self._maybe_open()

            return True

    def fail(self, reason: str) -> None:
        """외부 fail-stop: 닫고 이후 제출·계산·수락을 막는다. 기록은 더 넣지 않고 두 번째부터 무시."""
        with self._locked():
            if self._failed is not None:
                return

            self._failed = reason
            self._closed = True
            self._token = replace(self._token, ready=False, token_seq=self._token.token_seq + 1)
            self._current_accept = None
            self._count(reason)

            for entry in self._computing.values():
                entry["invalidated"] = True
                entry["cause"] = reason

    def run_end(self, reason: str) -> None:
        """실행 종료: 닫고 무효화·종료 기록을 남긴다. 두 번째부터는 무시."""
        with self._locked():
            if self._closed:
                return

            self._set_token(replace(self._token, ready=False), cause=f"run_end:{reason}")

            if not self._closed:   # 위 무효화에서 fail-stop이면 이미 닫힘
                self._closed = True
                self._emit("run_end", reserved=True, cause=reason)

    def invalidate_current(self, reason: str) -> int:
        """현재 A/B 결과·계산을 무효화하되 실시간 입력은 계속 받는다."""
        if not isinstance(reason, str) or not reason:
            raise ValueError("reason must be a non-empty string")

        with self._locked():
            if self._closed:
                self._count("invalidate_after_close")
                return self._token.token_seq

            self._set_token(self._token, cause=f"operator_discard:{reason}")
            return self._token.token_seq

    # ------------------------------------------------------------------ 계산 권한·수락(worker)

    def begin_compute(self, token: LiveObservationToken) -> int | None:
        """검사 ②: 열린 현재 토큰이면 `COMPUTING`으로 전이하고 compute_id를 준다. 아니면 None(B 호출 금지)."""
        with self._locked():
            if not self._is_open(token):
                self._count("stale_after_tf")
                return None

            cid = self._next_compute
            self._next_compute += 1
            self._computing[cid] = {"token_seq": token.token_seq, "invalidated": False, "cause": None}

            return cid

    def accept(self, compute_id: int, token: LiveObservationToken, payload) -> int | None:
        """검사 ③: 계산 중 무효화가 없고 토큰이 현재이며 일반 칸에 자리가 있으면 수락하고 event_seq를 준다."""
        with self._locked():
            entry = self._computing.pop(compute_id, None)

            if entry is None or entry["token_seq"] != token.token_seq:
                raise ValueError(f"unknown compute_id {compute_id} for token_seq {token.token_seq}")

            if entry["invalidated"] or not self._is_open(token):
                self._count("stale_after_compute")
                return None

            if len(self._normal) >= self._config.write_capacity:
                self._count("writer_backpressure")
                return None

            seq = self._emit("accept", reserved=False, compute_id=compute_id, payload=payload)
            self._current_accept = seq

            return seq

    def abandon(self, compute_id: int, reason: str) -> None:
        """계산이 예외 등으로 끝나 수락하지 않는다(출력 0). 계산 중 표에서만 지운다."""
        with self._locked():
            if self._computing.pop(compute_id, None) is not None:
                self._count(f"abandoned:{reason}")

    # ------------------------------------------------------------------ 쓰기(쓰기 담당)

    def take_events(self, max_n: int) -> list[LiveObservationEvent]:
        """순번 순서대로 최대 max_n개를 꺼낸다(대기 없음)."""
        with self._locked():
            out = self._order[:max_n]
            del self._order[:max_n]
            taken = {id(e) for e in out}
            self._normal = [e for e in self._normal if id(e) not in taken]
            self._reserved = [e for e in self._reserved if id(e) not in taken]

            return out

    # ------------------------------------------------------------------ 내부(잠금 안에서 부름)

    def _count(self, key: str) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1

    def _is_open(self, token: LiveObservationToken) -> bool:
        return not self._closed and self._token.ready and token.token_seq == self._token.token_seq

    def _maybe_open(self) -> None:
        b = self._boundary

        if self._stop_hold is not None:
            if b is not None and b.applied and b.tf_ready and not self._token.ready:
                self._count("open_blocked_by_stop_hold")
            return

        if b is not None and b.applied and b.tf_ready and not self._token.ready:
            self._opened_once = True
            self._token = replace(self._token, ready=True, token_seq=self._token.token_seq + 1)
            self._emit("ready", reserved=True, cause=f"boundary:{b.boundary_id}")

    def _set_token(self, new: LiveObservationToken, *, cause: str | None) -> None:
        """토큰을 바꾼다. 계산 중 항목에 무효 표지, 현재 수락 철회, 무효화 기록(cause가 있으면)."""
        self._token = replace(new, token_seq=self._token.token_seq + 1)

        for entry in self._computing.values():
            if not entry["invalidated"]:
                entry["invalidated"] = True
                entry["cause"] = cause or "token_update"

        if cause is None:
            return

        if self._current_accept is not None:
            revoked = self._current_accept
            self._current_accept = None
            self._emit("revoke", reserved=True, cause=cause, revoked_event_seq=revoked)

        self._emit("invalidation", reserved=True, cause=cause)

    def _emit(self, kind: str, *, reserved: bool, cause: str | None = None, compute_id: int | None = None,
              revoked_event_seq: int | None = None, payload=None) -> int | None:
        """순번을 매겨 대기열에 넣는다. 예약 칸이 차면 fail-stop."""
        if self._failed is not None:
            return None

        if reserved and len(self._reserved) >= self._config.reserved_capacity:
            self._failed = "b_writer_overflow"
            self._closed = True
            self._token = replace(self._token, ready=False, token_seq=self._token.token_seq + 1)
            self._count("b_writer_overflow")
            self._current_accept = None

            for entry in self._computing.values():
                entry["invalidated"] = True
                entry["cause"] = "b_writer_overflow"

            # 넘침 자체는 대기열 밖 마지막 기록으로 남긴다(용량 무시, 1회)
            self._event_seq += 1
            self._order.append(LiveObservationEvent(self._event_seq, "failed", self._token.token_seq, self._clock_ns(),
                                          cause="b_writer_overflow"))
            return None

        self._event_seq += 1
        event = LiveObservationEvent(self._event_seq, kind, self._token.token_seq, self._clock_ns(), cause=cause,
                           compute_id=compute_id, revoked_event_seq=revoked_event_seq, payload=payload)
        (self._reserved if reserved else self._normal).append(event)
        self._order.append(event)

        return self._event_seq


__all__ = ["BoundaryReadiness", "LiveObservationConfig", "LiveObservationEvent", "LiveObservationState", "LiveObservationToken", "RUN_START", "SIM_TIME_REWIND",
           "rewind_requires_tf"]
