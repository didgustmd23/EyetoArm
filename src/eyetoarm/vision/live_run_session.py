"""실시간 실행 세션: Gazebo 리셋·시간 역행이면 현재 실행을 끝내고, 새 실행은 명시 호출로만 처음부터 만든다.

실행마다 상태·slot·TF·디스패처를 새로 만들어 공유하지 않고, 영상은 `RunFrame(run_id, …)`으로 출처를 표시한다.
자동 재시작은 없다. ROS·Qt를 import하지 않는다(수신·TF 객체는 주입).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from eyetoarm.transforms.live_tf_buffer import TfRewindReadiness
from eyetoarm.vision.live_observation_state import SIM_TIME_REWIND, LiveObservationConfig, LiveObservationState


_INIT_FAILURE_CLOSE_LIMIT_S = 1.0   # 생성 실패 때 이미 만든 부품을 닫는 상한(s)


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
        """새 상태·slot·TF·디스패처·TF 준비를 주입 공장(`make_*`·`frame_of`)으로 만든다."""
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
        self.dispatcher = None
        self.slot = None

        try:
            self.slot = make_slot(self._on_boundary)
            self.dispatcher = make_dispatcher(self.state, self.tf)
            self.readiness = TfRewindReadiness(self.state, clear=lambda: None, has_static=self.tf.has_static,
                                               lookup=self.tf.lookup, world_frame=world_frame, static_pairs=static_pairs,
                                               wait_limit_s=wait_limit_s, future_tolerance_ns=future_tolerance_ns)
        except Exception:
            # 이미 만든 부품을 닫고 원래 예외를 올린다(정리 실패가 원래 예외를 가리지 않게)
            parts = ((self.dispatcher, True), (self.slot, False), (self.tf, True))

            for part, takes_limit in parts:
                if part is None:
                    continue

                try:
                    if takes_limit:
                        part.close(_INIT_FAILURE_CLOSE_LIMIT_S)
                    else:
                        part.close()
                except Exception:  # noqa: BLE001 - 정리 경로: 원래 생성 실패를 그대로 올리는 것이 우선
                    pass

            raise

    def _count(self, key: str) -> None:
        with self._lock:
            self.counts[key] = self.counts.get(key, 0) + 1

    # ------------------------------------------------------------------ 수신 쪽(slot 잠금 안)

    def _on_boundary(self, epoch: int, kind) -> None:
        """slot 경계 훅(slot 잠금 안). 되감기면 종료 표지와 상태 닫기만 한다(상태 잠금만, 교착 없음)."""
        if getattr(kind, "value", kind) == SIM_TIME_REWIND:
            with self._lock:
                if self.reset_reason is None:
                    self.reset_reason = f"reset:{SIM_TIME_REWIND}:epoch{epoch}"
                self.counts["reset_detected"] = self.counts.get("reset_detected", 0) + 1

            self.state.run_end(self.reset_reason)
            return

        try:
            self.state.on_rx_boundary(epoch, kind)
        except Exception:
            # slot 계수로만 두면 이전 수락이 현재로 남으므로 상태를 닫고 slot에 알린다
            self.state.fail("boundary_hook_error")
            raise

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
        """실행을 끝낸다: 상태 닫기 → worker 종료 → slot 닫기 → TF 닫기.

        정리 실패는 기록에 남기고 `cleanup_confirmed`는 거짓이다. 두 번째 호출부터는 `retry_cleanup`과 같다.
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

    def refusal(self, run_id: str) -> str | None:
        """새 실행을 거부할 사유(허용이면 None). 부품을 만들기 전에 확인할 때 쓴다."""
        if self.current is not None and self.current.ended is None:
            return f"run {self.current.run_id} is still active"

        if self.current is not None and not self.current.cleanup_confirmed:
            return f"run {self.current.run_id} cleanup not confirmed: {self.current.ended.close_errors}"

        if run_id in self.used_run_ids:
            return f"run_id {run_id!r} already used"

        return None

    def start_run(self, run_id: str, make_session: Callable[[str], object] | None = None):
        """새 실행. 현재 실행이 안 끝났거나 정리 미확인이거나 run_id를 다시 쓰면 ValueError.

        `make_session`을 주면 이번 실행만 그 공장으로 만든다.
        """
        reason = self.refusal(run_id)

        if reason is not None:
            raise ValueError(reason)

        self.used_run_ids.add(run_id)
        self.current = (make_session or self._make)(run_id)

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
