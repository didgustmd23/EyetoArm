"""실시간 관측 디스패처: 대기 칸 1개(최신 우선)와 worker 1개로 TF·봉투·B 계산·렌더를 잠금 밖에서 한다.

worker는 검사 ①(꺼낸 직후)·②(`begin_compute`)·③(`accept`)를 통과한 결과만 기록한다.
B 단계는 worker만 쓴다. ROS·Qt를 import하지 않는다.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

from eyetoarm.vision.observation_envelope import ObservationEnvelopeError
from eyetoarm.vision import stage_timing
from eyetoarm.vision.live_observation_state import LiveObservationState, LiveObservationToken


@dataclass(frozen=True)
class LiveObservationItem:
    """대기 칸 항목."""

    frame: object
    observation: object
    token: LiveObservationToken
    rx_epoch: int
    submitted_monotonic_ns: int


@dataclass(frozen=True)
class LiveObservationPayload:
    """수락 기록 묶음(잠금 밖에서 만든 결과)."""

    frame: object
    observation: object
    envelope: object
    output: object
    tf_status: str
    tf_wait_ms: float | None
    png: bytes | None
    received_monotonic_ns: int | None
    submitted_monotonic_ns: int
    compute_done_monotonic_ns: int
    view_bias: tuple | None = None   # 보정 기록 목록(보정 켬일 때만)


class LiveObservationDispatcher:
    """대기 칸 1개 + worker 1개."""

    def __init__(self, state: LiveObservationState, *, camera, stage_factory: Callable, tf_lookup: Callable,
                 build_envelope: Callable, compute: Callable, render: Callable | None = None,
                 clock_ns: Callable[[], int] = time.monotonic_ns, hooks: dict | None = None,
                 tf_lookup_with_epoch: bool = False, refuse_tf_statuses: frozenset = frozenset(),
                 on_skip: Callable | None = None,
                 envelope_errors: tuple = (ObservationEnvelopeError,),
                 view_bias: Callable | None = None, on_accepted: Callable | None = None,
                 on_not_accepted: Callable | None = None) -> None:
        """상태·처음 카메라·주입 함수(TF 조회·봉투 생성·계산·기록·훅)를 받는다.

        `on_accepted`·`on_not_accepted`는 검사 ③ 뒤 훅이며, 그 예외는 worker 오류가 아니라 `accept_hook_errors`로 센다.
        """
        self.state = state
        self._stage_factory = stage_factory
        self._tf_lookup = tf_lookup
        self._build = build_envelope
        self._compute = compute
        self._render = render
        self._clock_ns = clock_ns
        self._hooks = hooks or {}
        self._tf_with_epoch = bool(tf_lookup_with_epoch)
        self._refuse_tf = frozenset(refuse_tf_statuses)
        self._on_skip = on_skip
        self._envelope_errors = tuple(envelope_errors)
        self._view_bias = view_bias
        self._on_accepted = on_accepted
        self._on_not_accepted = on_not_accepted
        self.view_bias_applied = 0             # worker가 `_counts_lock` 안에서 씀
        self.view_bias_skipped: dict[str, int] = {}
        self._cond = threading.Condition()
        self._slot: LiveObservationItem | None = None
        self._pending_camera = camera
        self._camera = None
        self._stage = None
        self._stop = False
        self._thread: threading.Thread | None = None
        self._busy = False
        self.counts: dict[str, int] = {}
        self._counts_lock = threading.Lock()
        self.compute_calls = 0
        self.errors: list[str] = []

    # ------------------------------------------------------------------ 제어 흐름

    def submit(self, frame, observation, *, rx_epoch: int | None, generation: int, calibration_id: str | None) -> bool:
        """대기 없이 넣는다. 토큰을 못 받으면 False(B 미도달)."""
        token = self.state.admit(rx_epoch=rx_epoch, generation=generation, calibration_id=calibration_id)

        if token is None:
            self._count("submit_refused")
            return False

        item = LiveObservationItem(frame, observation, token, rx_epoch, self._clock_ns())

        with self._cond:
            if self._stop:
                self._count("submit_after_close")
                return False

            replaced = self._slot

            if replaced is not None:
                self._count("superseded")

            self._slot = item
            self._cond.notify_all()

        if replaced is not None:
            self._skip(replaced, "superseded")

        return True

    def set_camera(self, camera) -> None:
        """다음 항목 전에 worker가 교체할 카메라 프로파일."""
        with self._cond:
            self._pending_camera = camera
            self._cond.notify_all()

    def discard_pending(self, reason: str) -> None:
        """대기 칸을 비운다. 실행 중 계산은 상태 토큰 무효화로 수락되지 않는다."""
        with self._cond:
            dropped, self._slot = self._slot, None
            if dropped is not None:
                self._count(f"discarded:{reason}")
            self._cond.notify_all()

        if dropped is not None:
            self._skip(dropped, f"discarded:{reason}")

    def start(self) -> None:
        """worker를 시작한다."""
        if self._thread is not None:
            raise RuntimeError("dispatcher already started")

        self._thread = threading.Thread(target=self._loop, name="eyetoarm-live-observation", daemon=True)
        self._thread.start()

    def drain(self, limit_s: float) -> bool:
        """대기 칸·처리 중 항목이 빌 때까지 기다린다(새 제출은 막지 않음). 상한 안에 비면 True."""
        deadline = time.monotonic() + max(0.0, limit_s)

        with self._cond:
            while self._slot is not None or self._busy:
                if self._thread is None or not self._thread.is_alive():
                    return self._slot is None and not self._busy

                remaining = deadline - time.monotonic()

                if remaining <= 0:
                    return False

                self._cond.wait(min(remaining, 0.05))

            return True

    def close(self, limit_s: float) -> bool:
        """대기 항목을 버리고 worker를 멈춘다. 상한 안에 멈추면 True."""
        with self._cond:
            self._stop = True
            dropped = self._slot

            if dropped is not None:
                self._count("dropped_on_close")
                self._slot = None

            self._cond.notify_all()

        if dropped is not None:
            self._skip(dropped, "dropped_on_close")

        if self._thread is None:
            return True

        self._thread.join(timeout=limit_s)

        return not self._thread.is_alive()

    # ------------------------------------------------------------------ worker

    def view_bias_summary(self) -> dict:
        """보정 수 요약(잠금 안에서 복사)."""
        with self._counts_lock:
            return {"applied": self.view_bias_applied, "skipped_by_reason": dict(sorted(self.view_bias_skipped.items()))}

    def _count(self, key: str) -> None:
        """계수(별도 짧은 잠금, 대기 칸 조건변수와 섞지 않음)."""
        with self._counts_lock:
            self.counts[key] = self.counts.get(key, 0) + 1

    def _skip(self, item, reason: str, detail: str | None = None) -> None:
        """건너뜀 통지(선택)."""
        if self._on_skip is not None:
            self._on_skip(item, reason, detail)

    def _hook(self, name: str) -> None:
        fn = self._hooks.get(name)

        if fn is not None:
            fn()

    def _take(self) -> LiveObservationItem | None:
        with self._cond:
            while self._slot is None and not self._stop:
                self._cond.wait(0.05)

            if self._stop:
                return None

            item, self._slot = self._slot, None
            self._busy = True

            if self._pending_camera is not None:
                self._camera, self._pending_camera = self._pending_camera, None
                self._stage = None   # 아래(잠금 밖)에서 새로 만든다

            return item

    def _loop(self) -> None:
        while True:
            item = self._take()

            if item is None:
                return

            try:
                self._process(item)
            except Exception as exc:  # worker는 죽지 않는다. 오류는 기록한다
                self.errors.append(f"{type(exc).__name__}: {exc}")
                self._count("worker_error")
            finally:
                with self._cond:
                    self._busy = False
                    self._cond.notify_all()

    def _process(self, item: LiveObservationItem) -> None:
        seq = getattr(item.frame, "image_seq", None)
        stage_timing.mark("b_start", seq)

        try:
            self._process_body(item)
        finally:
            stage_timing.mark("b_done", seq)

    def _process_body(self, item: LiveObservationItem) -> None:
        if self._stage is None:
            self._stage = self._stage_factory(self._camera)

        self._hook("after_take")

        if not self.state.is_current(item.token):              # ①
            self._count("stale_before_b")
            self._skip(item, "stale_before_b")
            return

        if self._tf_with_epoch:
            tf_result = self._tf_lookup(item.frame, rx_epoch=item.rx_epoch)
        else:
            tf_result = self._tf_lookup(item.frame)

        self._count(f"tf_{tf_result.status}")
        self._hook("after_tf")

        if tf_result.status in self._refuse_tf:                  # 거부 집합의 TF 상태: 계산 권한 0
            self._count("tf_refused")
            self._skip(item, f"tf_{tf_result.status}", tf_result.detail)
            return

        try:
            envelope = self._build(item.frame, item.observation, self._camera, tf_result)
        except self._envelope_errors as exc:
            self._count("b_input_rejected")
            self.errors.append(f"envelope: {exc}")
            self._skip(item, "b_input_rejected", str(exc))
            return

        # 단계 T·L 봉투 공통 보정 지점(끔이면 부르지 않음)
        view_bias = None

        if self._view_bias is not None:
            envelope, records = self._view_bias(envelope, self._camera)
            view_bias = tuple(records)

            with self._counts_lock:
                for r in records:
                    if r["applied"]:
                        self.view_bias_applied += 1
                    else:
                        self.view_bias_skipped[r["reason"]] = self.view_bias_skipped.get(r["reason"], 0) + 1

        compute_id = self.state.begin_compute(item.token)       # ②
        self._hook("after_begin")

        if compute_id is None:
            self._count("stale_after_tf")
            self._skip(item, "stale_after_tf")
            return

        try:
            self.compute_calls += 1
            out = self._compute(self._stage, envelope)
            self._hook("in_compute")
            png = None if self._render is None else self._render(self._stage, item.frame, out)
        except Exception as exc:
            self.state.abandon(compute_id, type(exc).__name__)
            raise

        payload = LiveObservationPayload(item.frame, item.observation, envelope, out, tf_result.status, tf_result.wait_ms, png,
                               getattr(item.frame, "received_monotonic_ns", None), item.submitted_monotonic_ns,
                               self._clock_ns(), view_bias)
        self._hook("before_accept")

        seq = self.state.accept(compute_id, item.token, payload)  # ③
        self._count("accepted" if seq is not None else "not_accepted")

        if seq is None:
            self._skip(item, "not_accepted")

            if self._on_not_accepted is not None:
                try:
                    self._on_not_accepted(item, payload)
                except Exception as exc:   # B 결과·기록에 영향 없음(진입점이 종료 1로 드러냄)
                    self._count("accept_hook_errors")
                    self.errors.append(f"accept_hook: {type(exc).__name__}: {exc}")
        elif self._on_accepted is not None:
            try:
                self._on_accepted(item, payload, seq)
            except Exception as exc:   # B 결과·기록에 영향 없음(진입점이 종료 1로 드러냄)
                self._count("accept_hook_errors")
                self.errors.append(f"accept_hook: {type(exc).__name__}: {exc}")


__all__ = ["LiveObservationDispatcher", "LiveObservationItem", "LiveObservationPayload"]
