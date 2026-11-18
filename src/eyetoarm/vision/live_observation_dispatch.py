"""실시간 A→B 디스패처·worker(#171 R4): 대기 칸 1개(최신 우선), 검사 ①②③, TF·봉투·B·렌더는 잠금 밖.

근거: `_workspace/171_builder2_live_a_to_b_plan.md` r3 §1.1·§3.2·§3.4·§3.5. **준비 부품이며 실행기(`run()`)·진입점에 연결하지 않는다**(D13 유지).

흐름
- 제어 흐름 `submit(frame, obs, rx_epoch=, generation=, calibration_id=)`: `LiveObservationState.admit`로 토큰을 받고 대기 칸에 넣는다(대기 없음).
  칸에 이전 항목이 있으면 `superseded`로 바꾼다(무효 아님, 계획 §3.4).
- worker 1개
  1. 꺼낸 직후 검사 ①(`state.is_current`). 아니면 `stale_before_b`(TF·B 없음).
  2. TF 조회(주입 `tf_lookup(frame) → TfLookupResult`, 잠금 밖).
  3. 봉투 생성(주입 `build_envelope(frame, obs, camera, tf_result)`). 봉투 오류는 `b_input_rejected`(B 없음).
  4. 검사 ② `state.begin_compute(token)` — 토큰 비교와 `COMPUTING` 전이가 한 잠금 구간. 실패면 `stale_after_tf`(B 호출 0).
  5. B 계산(주입 `compute(stage, envelope)`)·렌더(주입 `render(stage, frame, out) → bytes | None`), 잠금 밖.
  6. 검사 ③ `state.accept(compute_id, token, payload)`. 거부면 출력 없음.
- 카메라 교체: 제어 흐름 `set_camera(camera)`는 대기 값만 둔다. worker가 다음 항목 전에 `stage_factory(camera)`로 B 단계를 새로 만든다
  (B 단계는 worker만 쓴다).
- 시험 훅 `hooks[name]()`: 지점 이름 `after_take`·`after_tf`·`after_begin`·`in_compute`·`before_accept`에서 부른다(경합 순서 강제용, 기본 없음).
- 되감기 epoch 보호 경로(#173 되감기 계획 r1, 선택): `tf_lookup_with_epoch=True`면 `tf_lookup(frame, rx_epoch=)`로 제출 때의 수신 epoch를
  넘긴다. `refuse_tf_statuses`에 든 TF 상태(예: `stale_overlap`·`epoch_mismatch`)는 봉투·검사 ② 전에 끝낸다(계산 권한 0, B 호출 0).
  기본값(False, 빈 집합)은 기존 동작 그대로다.
- 건너뜀 통지(#173, 선택): `on_skip(item, reason, detail)`을 worker가 B를 부르지 않고 끝내는 지점마다 부른다(사유: `stale_before_b`·
  `tf_<상태>`(거부 집합일 때)·`b_input_rejected`·`stale_after_tf`). 수락 거부는 `not_accepted`. 예외는 worker 오류로 센다.
- ROS·Qt를 import하지 않는다.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

from eyetoarm.vision.observation_envelope import ObservationEnvelopeError
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


class LiveObservationDispatcher:
    """대기 칸 1개 + worker 1개."""

    def __init__(self, state: LiveObservationState, *, camera, stage_factory: Callable, tf_lookup: Callable,
                 build_envelope: Callable, compute: Callable, render: Callable | None = None,
                 clock_ns: Callable[[], int] = time.monotonic_ns, hooks: dict | None = None,
                 tf_lookup_with_epoch: bool = False, refuse_tf_statuses: frozenset = frozenset(),
                 on_skip: Callable | None = None) -> None:
        """상태·처음 카메라·주입 함수들. 마지막 두 인자는 되감기 epoch 보호 경로용(기본은 기존 동작)."""
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

    def start(self) -> None:
        """worker를 시작한다."""
        if self._thread is not None:
            raise RuntimeError("dispatcher already started")

        self._thread = threading.Thread(target=self._loop, name="eyetoarm-live-observation", daemon=True)
        self._thread.start()

    def drain(self, limit_s: float) -> bool:
        """대기 칸이 비고 처리 중 항목이 끝날 때까지 기다린다(정상 종료 전, #173). 새 제출은 막지 않는다. 상한 안에 비면 True."""
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
        except ObservationEnvelopeError as exc:
            self._count("b_input_rejected")
            self.errors.append(f"envelope: {exc}")
            self._skip(item, "b_input_rejected", str(exc))
            return

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
                               self._clock_ns())
        self._hook("before_accept")

        seq = self.state.accept(compute_id, item.token, payload)  # ③
        self._count("accepted" if seq is not None else "not_accepted")

        if seq is None:
            self._skip(item, "not_accepted")


__all__ = ["LiveObservationDispatcher", "LiveObservationItem", "LiveObservationPayload"]
