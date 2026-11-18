"""실시간 A→B 실행 세션(#173 사용자 정정 2026-11-18): Gazebo 리셋·시간 역행이면 현재 실행을 끝내고, 새 실행은 명시적으로 처음부터 시작한다.

근거: reviewer 전달 사용자 정정(같은 실행 안 되감기 재개·C 정지 확인 연동을 이 방식으로 대체), `_workspace/171_builder2_live_a_to_b_plan.md` r3
(상태·디스패처·TF 준비 부품). **준비 부품이며 실시간 실행 경로·제품 진입점에 연결하지 않는다**(D13 유지). 자동 재시작은 없다.

구성
- `LiveRunSession`: 실행 하나의 묶음. `run_id`, 새 `LiveObservationState`(`start_closed=True`), 새 입력 칸(slot), 새 TF 수신 묶음,
  새 디스패처(B 단계는 디스패처가 새로 만든다), TF 준비(`TfRewindReadiness`, 경계 0). 실행끼리 어떤 객체도 공유하지 않는다.
- 열림 조건: 그 실행의 slot에서 꺼낸 영상(그 영상의 수신 epoch = 현재 경계)으로 (1) 보정 정보(`calibration_id`)가 있으면
  `state.on_applied(epoch, generation, calibration_id)`, (2) 그 영상 촬영 stamp의 TF 정확 조회가 그 실행의 새 버퍼에서 성공하면
  `state.on_tf_ready(epoch, 시도)`. 둘 다 끝나야 계산이 열린다. 첫 열림 전 경계(CameraInfo 첫 수신 등)도 TF 준비를 요구한다
  (`start_closed`). 새 버퍼는 실행 시작 때 비어 있으므로 준비 시도 때 버퍼를 비우지 않는다(비우면 이미 받은 촬영 시각 TF를 잃는다).
- 리셋 감지: slot 경계 훅(slot 잠금 안)에서 `SIM_TIME_REWIND`이면 `state.run_end("reset:…")`만 한다(상태 잠금만, slot·디스패처를
  건드리지 않음 = 교착 없음). 이것으로 이전 결과 현재성 해제(revoke)·계산 중 결과 수락 차단·이후 제출 거부가 즉시 일어난다. 대기 작업
  버리기·worker 종료·slot 닫기·TF 닫기는 제어 흐름이 `service()`(또는 `end_run`)에서 한다. 되감기가 아닌 경계는 `state.on_rx_boundary`로
  넘긴다(기존 규칙).
- 출처 표지: 세션의 `take()`는 영상을 `RunFrame(run_id, …)`으로 감싼다. `submit()`은 그 세션 `run_id`의 영상만 받는다(`foreign_run`).
  이전 실행 영상이 수신 epoch·세대·보정 번호가 우연히 같아도 새 실행에 들어가지 않는다.
- `LiveRunController`: 현재 세션 하나. `start_run(run_id)`는 명시 호출로만 새 세션을 만든다. 현재 세션이 살아 있거나, 끝났어도
  정리 완료(`cleanup_confirmed`: worker·slot·TF 정리 단계 모두 성공 + worker 스레드 종료)가 확인되지 않았거나, `run_id`를 다시 쓰면
  거부한다. 정리 실패는 `retry_cleanup()`으로 실패한 단계만 다시 한다(#173 r1 reviewer 재현 결함). 리셋 뒤 `service()`는 끝내기만 하고 새 실행을 만들지 않는다.
- 남는 한계(보장 아님): 새 실행의 TF 버퍼·구독은 새것이지만, 리셋 직전 이전 실행 발행자가 보낸 `/tf`·영상이 새 구독에 늦게 도착하면
  stamp만으로는 출처를 가를 수 없다(보고서 §4). 이 모듈은 이를 막지 않는다.
- ROS·Qt를 import하지 않는다(모든 수신·TF 객체는 주입).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from eyetoarm.transforms.live_tf_buffer import TfRewindReadiness
from eyetoarm.vision.live_observation_state import SIM_TIME_REWIND, LiveObservationConfig, LiveObservationState


@dataclass(frozen=True)
class RunFrame:
    """세션 slot에서 꺼낸 영상(출처 실행 표지 포함)."""

    run_id: str
    frame: object
    rx_epoch: int
    generation: int


@dataclass(frozen=True)
class RunEnd:
    """끝난 실행 기록."""

    run_id: str
    reason: str
    close_errors: tuple


class LiveRunSession:
    """실행 하나. 생성자에서 모든 부품을 새로 만든다."""

    def __init__(self, run_id: str, *, generation: int, state_config: LiveObservationConfig,
                 make_slot: Callable, make_tf: Callable, make_dispatcher: Callable, frame_of: Callable,
                 world_frame: str, static_pairs: tuple, wait_limit_s: float, future_tolerance_ns: int) -> None:
        """새 상태·slot·TF·디스패처·TF 준비를 만든다.

        - `make_slot(boundary_hook)` → slot(`take`·`close`·`on_*`).
        - `make_tf()` → TF 수신 묶음(`lookup`·`has_static`·`clear`·`close(limit_s)`), 시작된 상태로 돌려준다.
        - `make_dispatcher(state, tf)` → 디스패처(`submit`·`start`·`close(limit_s)`).
        - `frame_of(record, generation)` → 실시간 프레임(`ros_stamp_ns`·`optical_frame`·`calibration_id` 포함).
        """
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"run_id must be a non-empty string: {run_id!r}")

        self.run_id = run_id
        self.generation = generation
        self.state = LiveObservationState(run_id=run_id, generation=generation, calibration_id=None,
                                          config=state_config, start_closed=True)
        self._frame_of = frame_of
        self._lock = threading.Lock()
        self.reset_reason: str | None = None
        self.ended: RunEnd | None = None
        self._cleanup_done: set[str] = set()
        self.cleanup_attempts = 0
        self.counts: dict[str, int] = {}
        self.tf = make_tf()
        self.slot = make_slot(self._on_boundary)
        self.dispatcher = make_dispatcher(self.state, self.tf)
        self.readiness = TfRewindReadiness(self.state, clear=lambda: None, has_static=self.tf.has_static,
                                           lookup=self.tf.lookup, world_frame=world_frame, static_pairs=static_pairs,
                                           wait_limit_s=wait_limit_s, future_tolerance_ns=future_tolerance_ns)

    def _count(self, key: str) -> None:
        with self._lock:
            self.counts[key] = self.counts.get(key, 0) + 1

    # ------------------------------------------------------------------ 수신 쪽(slot 잠금 안)

    def _on_boundary(self, epoch: int, kind) -> None:
        """slot 경계 훅. 되감기면 실행 종료 표지와 상태 닫기만 한다(상태 잠금만 잡는다)."""
        if getattr(kind, "value", kind) == SIM_TIME_REWIND:
            with self._lock:
                if self.reset_reason is None:
                    self.reset_reason = f"reset:{SIM_TIME_REWIND}:epoch{epoch}"
                self.counts["reset_detected"] = self.counts.get("reset_detected", 0) + 1

            self.state.run_end(self.reset_reason)
            return

        self.state.on_rx_boundary(epoch, kind)

    # ------------------------------------------------------------------ 제어 흐름

    @property
    def reset_pending(self) -> bool:
        """리셋을 감지했고 아직 끝내지 않았는가."""
        with self._lock:
            return self.reset_reason is not None and self.ended is None

    def take(self, wait_s: float):
        """slot에서 영상을 꺼내 `RunFrame`으로 감싼다. 영상이 아니면 slot의 `Take`를 그대로 돌려준다."""
        took = self.slot.take(wait_s)

        if getattr(took, "record", None) is None or self.ended is not None:
            return took

        frame = self._frame_of(took.record, took.capture_generation)

        return RunFrame(self.run_id, frame, took.record.rx_epoch, took.capture_generation)

    def try_open(self, run_frame: RunFrame) -> str:
        """새 실행 열기: 그 실행 영상의 보정 정보 반영 → 촬영 stamp TF 준비. 상태 문자열을 돌려준다."""
        if not isinstance(run_frame, RunFrame) or run_frame.run_id != self.run_id:
            self._count("open_foreign_run")
            return "foreign_run"

        calibration_id = getattr(run_frame.frame, "calibration_id", None)

        if calibration_id is None:
            self._count("open_no_calibration")
            return "no_calibration"

        epoch = run_frame.rx_epoch
        boundary = self.state.boundary

        if boundary is None or boundary.boundary_id != epoch:
            self._count("open_stale_epoch")
            return "stale_epoch"

        self.state.on_applied(epoch, run_frame.generation, calibration_id)

        if self.state.boundary.tf_ready:   # 되감기가 아닌 경계(첫 열림 뒤)는 TF 준비가 필요 없다
            return "ready" if self.state.token.ready else "not_ready"

        if self.readiness.begin(epoch) != "begun":   # 새 시도 번호(버퍼는 비우지 않음)
            return "not_begun"

        return self.readiness.try_ready(epoch, run_frame.frame)

    def submit(self, run_frame: RunFrame, observation) -> str:
        """그 실행 영상만 디스패처에 넣는다. 결과: ok | refused | foreign_run | ended."""
        if self.ended is not None or self.reset_pending:
            self._count("submit_after_end")
            return "ended"

        if not isinstance(run_frame, RunFrame) or run_frame.run_id != self.run_id:
            self._count("submit_foreign_run")
            return "foreign_run"

        ok = self.dispatcher.submit(run_frame.frame, observation, rx_epoch=run_frame.rx_epoch,
                                    generation=run_frame.generation,
                                    calibration_id=getattr(run_frame.frame, "calibration_id", None))

        return "ok" if ok else "refused"

    def end(self, reason: str, *, limit_s: float) -> RunEnd:
        """실행을 끝낸다: 상태 닫기(현재성 해제·수락 차단) → 대기 작업 버리기·worker 종료 → slot 닫기 → TF 닫기.

        정리 단계가 실패하면(worker가 상한 안에 안 멈춤, 닫기 예외) 기록에 남기고 `cleanup_confirmed`는 거짓이다. 두 번째 호출부터는
        `retry_cleanup`과 같다(실패한 단계만 다시 하고 새 기록을 돌려준다). 상태 닫기는 처음 한 번만 한다.
        """
        with self._lock:
            first = self.ended is None

        if first:
            self.state.run_end(reason)

        return self._cleanup(reason if first else self.ended.reason, limit_s)

    def retry_cleanup(self, limit_s: float) -> RunEnd:
        """끝난 실행의 실패한 정리 단계만 다시 한다. 끝나지 않은 실행이면 ValueError."""
        with self._lock:
            ended = self.ended

        if ended is None:
            raise ValueError(f"run {self.run_id} has not ended")

        return self._cleanup(ended.reason, limit_s)

    @property
    def cleanup_confirmed(self) -> bool:
        """끝났고 모든 정리 단계가 성공했으며 worker 스레드가 살아 있지 않은가(다음 실행 허용 조건)."""
        with self._lock:
            done = self.ended is not None and self._cleanup_done >= {"dispatcher", "slot", "tf"}

        thread = getattr(self.dispatcher, "_thread", None)

        return done and not (thread is not None and thread.is_alive())

    def _cleanup(self, reason: str, limit_s: float) -> RunEnd:
        """아직 성공하지 않은 정리 단계를 순서대로 한다. 단계 실패는 다음 단계를 막지 않는다."""
        errors = []
        steps = (("dispatcher", lambda: self.dispatcher.close(limit_s)), ("slot", self.slot.close),
                 ("tf", lambda: self.tf.close(limit_s)))

        for name, fn in steps:
            with self._lock:
                if name in self._cleanup_done:
                    continue

            try:
                result = fn()
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
                continue

            if result is False:
                errors.append(f"{name}: did not stop within {limit_s}s")
                continue

            with self._lock:
                self._cleanup_done.add(name)

        record = RunEnd(self.run_id, reason, tuple(errors))

        with self._lock:
            self.ended = record
            self.cleanup_attempts += 1

        return record


class LiveRunController:
    """현재 실행 세션 하나를 관리한다. 새 실행은 `start_run` 명시 호출로만 만든다(자동 재시작 없음)."""

    def __init__(self, make_session: Callable[[str], LiveRunSession], *, close_limit_s: float) -> None:
        """`make_session(run_id)` → 새 세션(부품 모두 새로)."""
        self._make = make_session
        self.close_limit_s = close_limit_s
        self.current: LiveRunSession | None = None
        self.used_run_ids: set[str] = set()
        self.ended: list[RunEnd] = []

    def start_run(self, run_id: str) -> LiveRunSession:
        """새 실행. 현재 실행이 끝나지 않았거나, 끝났어도 정리 완료가 확인되지 않았거나, run_id를 다시 쓰면 ValueError."""
        if self.current is not None and self.current.ended is None:
            raise ValueError(f"run {self.current.run_id} is still active")

        if self.current is not None and not self.current.cleanup_confirmed:
            raise ValueError(f"run {self.current.run_id} cleanup not confirmed: {self.current.ended.close_errors}")

        if run_id in self.used_run_ids:
            raise ValueError(f"run_id {run_id!r} already used")

        self.used_run_ids.add(run_id)
        self.current = self._make(run_id)

        return self.current

    def service(self) -> RunEnd | None:
        """제어 흐름 주기 호출: 리셋이 감지된 현재 실행을 끝낸다. 새 실행은 만들지 않는다."""
        session = self.current

        if session is None or not session.reset_pending:
            return None

        record = session.end(session.reset_reason, limit_s=self.close_limit_s)
        self.ended.append(record)

        return record

    def retry_cleanup(self) -> RunEnd | None:
        """끝났지만 정리 완료가 확인되지 않은 현재 실행의 정리를 다시 한다. 해당 없으면 None."""
        session = self.current

        if session is None or session.ended is None or session.cleanup_confirmed:
            return None

        record = session.retry_cleanup(self.close_limit_s)
        self.ended.append(record)

        return record

    def stop(self, reason: str) -> RunEnd | None:
        """현재 실행을 명시적으로 끝낸다."""
        session = self.current

        if session is None or session.ended is not None:
            return None

        record = session.end(reason, limit_s=self.close_limit_s)
        self.ended.append(record)

        return record


__all__ = ["LiveRunController", "LiveRunSession", "RunEnd", "RunFrame"]
