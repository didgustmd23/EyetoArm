"""`LiveObservationState`의 기록을 `event_seq` 순서대로 `b_live_events.jsonl` 한 파일에 쓴다.

출력 폴더는 새로 만들고(있으면 FileExistsError) accept PNG는 `annotated/{event_seq:08d}.png`로 따로 쓴다.
쓰기는 잠금 밖에서 꺼낸 순서대로 하며, 순번 역전·쓰기 실패는 fail-stop이다.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from eyetoarm.vision.list_envelope import LIST_ENVELOPE_VERSION
from eyetoarm.vision.live_observation_state import LiveObservationEvent, LiveObservationState


def _plain(value):
    """JSON 안전 값(넘파이 배열·스칼라 → 리스트·숫자)."""
    if hasattr(value, "tolist"):
        return value.tolist()

    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]

    return value


def _list_accept_fields(event: LiveObservationEvent, p, png_name: str | None) -> dict:
    """목록 봉투 accept 기록 키: 후보 원 인덱스·수와 `consumer_result`(A는 해석하지 않음).

    단일 관측 전용 키(끝점·goal·result_dict 등)는 두지 않는다."""
    f, item, envelope = p.frame, p.observation, p.envelope
    indices = [c.instance_index for c in envelope.observations.candidates]

    rec = {"compute_id": event.compute_id,
           "input": {"run_id": f.run_id, "image_seq": f.image_seq, "capture_generation": f.capture_generation,
                     "ros_stamp_ns": f.ros_stamp_ns, "calibration_id": f.calibration_id,
                     "job_seq": getattr(item, "job_seq", None), "generation": getattr(item, "generation", None)},
           "b_input": envelope.to_record(), "list_indices": indices, "list_candidate_count": len(indices),
           "consumer_result": _plain(p.output), "tf_status": p.tf_status, "tf_wait_ms": p.tf_wait_ms,
           "k5_ms": None if p.received_monotonic_ns is None
           else (event.enqueued_monotonic_ns - p.received_monotonic_ns) / 1e6,
           "annotation": png_name}

    # 보정 켬일 때만 후보별 보정 기록을 남긴다. b_input 후보 끝점은 B에 넘긴 보정값이다
    if getattr(p, "view_bias", None) is not None:
        rec["view_bias"] = _plain(list(p.view_bias))

    return rec


def event_record(event: LiveObservationEvent, png_name: str | None) -> dict:
    """기록 한 줄(사전). accept면 입력 키·B 출력 요약·시간 구간을 넣는다."""
    rec = {"event_seq": event.event_seq, "kind": event.kind, "token_seq": event.token_seq,
           "enqueued_monotonic_ns": event.enqueued_monotonic_ns, "cause": event.cause}

    if event.revoked_event_seq is not None:
        rec["revoked_event_seq"] = event.revoked_event_seq

    p = event.payload

    if event.kind == "accept" and p is not None and getattr(p.envelope, "version", None) == LIST_ENVELOPE_VERSION:
        rec.update(_list_accept_fields(event, p, png_name))
    elif event.kind == "accept" and p is not None:
        f, obs, out = p.frame, p.observation, p.output
        goal = getattr(out, "goal", None) or {}
        result = getattr(out, "result", None)
        rec.update({
            "compute_id": event.compute_id,
            "input": {"run_id": f.run_id, "image_seq": f.image_seq, "capture_generation": f.capture_generation,
                      "ros_stamp_ns": f.ros_stamp_ns, "calibration_id": f.calibration_id,
                      "job_seq": getattr(obs, "job_seq", None), "generation": getattr(obs, "generation", None),
                      "axis_endpoints_px": _plain(getattr(obs, "axis_endpoints_px", None))},
            "b_input": p.envelope.to_record() if hasattr(p.envelope, "to_record") else None,
            "goal_kind": goal.get("kind"), "b_reason": goal.get("reason"),
            "b_valid": getattr(result, "valid", None),
            # 저장 경로 기록기(`b_outputs.jsonl`)와 같은 키로 B 결과 전체를 남긴다
            "result_dict": _plain(getattr(out, "result_dict", None)), "goal": _plain(goal) if goal else None,
            "tf_status": p.tf_status, "tf_wait_ms": p.tf_wait_ms,
            "k5_ms": None if p.received_monotonic_ns is None
            else (event.enqueued_monotonic_ns - p.received_monotonic_ns) / 1e6,
            "annotation": png_name,
        })

        # 추적 켬일 때만 확정 번호·추적 상태를 남긴다. W 추적은 봉투를 만들 때 번호를 붙이므로 봉투 관측을 쓴다
        b_obs = getattr(p.envelope, "observation", obs)

        if hasattr(b_obs, "track_state") and hasattr(b_obs, "part_track_id"):
            rec["input"]["part_track_id"] = b_obs.part_track_id
            rec["input"]["track_state"] = b_obs.track_state

        # 보정 켬일 때만 B에 넘긴 끝점(봉투 관측)·원 끝점·보정 기록을 남긴다
        if getattr(p, "view_bias", None) is not None:
            vb = p.view_bias[0] if p.view_bias else None
            rec["input"]["axis_endpoints_px"] = _plain(getattr(p.envelope.observation, "axis_endpoints_px", None))
            rec["input"]["axis_endpoints_px_raw"] = None if vb is None else vb.get("axis_endpoints_px_raw")
            rec["input"]["view_bias"] = None if vb is None else {k: vb[k] for k in ("applied", "s", "delta_mm", "reason")}

    return rec


@dataclass(frozen=True)
class WriterCloseReport:
    """종료 보고. `unwritten`은 꺼냈지만 쓰지 못한 기록 수, `pending`은 상태 대기열에 남은 수."""

    written: int
    unwritten: int
    pending: int
    timed_out: bool
    error: str | None


class LiveObservationEventWriter:
    """기록 쓰기 담당: 단일 쓰기 소유자, 실패 시 기록 보존·fail-stop, 종료 상한.

    파일은 `start()` 전에는 호출자, 뒤에는 쓰기 스레드만 쓰고 닫는다(다른 스레드의 `drain_once`는 RuntimeError).
    쓰기 실패 시 남은 기록을 `unwritten`에 보존하고 `state.fail("b_writer_error")`로 입력을 막는다.
    `close`가 기한을 넘기면 파일에 손대지 않고 `timed_out=True`와 `b_writer_close_timeout` fail-stop.
    """

    def __init__(self, state: LiveObservationState, out_dir: str, *, lag_warn_s: float, batch: int = 64,
                 clock_ns: Callable[[], int] = time.monotonic_ns, poll_s: float = 0.01) -> None:
        """출력 폴더를 새로 만든다. `lag_warn_s`(밀림 경고 기준)는 기본값이 없다."""
        if not (isinstance(lag_warn_s, (int, float)) and lag_warn_s > 0):
            raise ValueError(f"lag_warn_s must be positive: {lag_warn_s!r}")

        self.state = state
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=False, exist_ok=False)
        (self.dir / "annotated").mkdir()
        self._fh = open(self.dir / "b_live_events.jsonl", "x", encoding="utf-8")
        self._lag_ns = int(lag_warn_s * 1e9)
        self._batch = batch
        self._clock_ns = clock_ns
        self._poll_s = poll_s
        self._stop = threading.Event()
        self._deadline: float | None = None
        self._thread: threading.Thread | None = None
        self.written = 0
        self.lag_warnings = 0
        self.max_lag_ms = 0.0
        self.last_seq = 0
        self.error: str | None = None
        self.unwritten: list[LiveObservationEvent] = []
        self.file_closed = False

    def _require_owner(self) -> None:
        """스레드가 있으면 그 스레드만 쓴다."""
        if self._thread is not None and threading.current_thread() is not self._thread:
            raise RuntimeError("writer thread owns the file")

    def _write_one(self, event: LiveObservationEvent) -> None:
        """기록 하나(순번 검사 → PNG → JSONL)."""
        if event.event_seq <= self.last_seq:
            raise RuntimeError(f"event_seq out of order: {event.event_seq} after {self.last_seq}")

        png_name = None
        payload = event.payload

        if event.kind == "accept" and payload is not None and getattr(payload, "png", None) is not None:
            png_name = f"annotated/{event.event_seq:08d}.png"

            with open(self.dir / png_name, "xb") as fh:
                fh.write(payload.png)

        self._fh.write(json.dumps(event_record(event, png_name), ensure_ascii=False, allow_nan=False) + "\n")
        self._fh.flush()
        lag = self._clock_ns() - event.enqueued_monotonic_ns
        self.max_lag_ms = max(self.max_lag_ms, lag / 1e6)

        if lag > self._lag_ns:
            self.lag_warnings += 1

        self.last_seq = event.event_seq
        self.written += 1

    def drain_once(self) -> int:
        """대기 기록을 한 번 꺼내 쓴다(소유자만). 쓴 수. 실패하면 남은 기록을 보존하고 fail-stop, 이후 0."""
        self._require_owner()

        if self.error is not None or self.file_closed:
            return 0

        events = self.state.take_events(self._batch)
        done = 0

        for i, event in enumerate(events):
            try:
                self._write_one(event)
            except Exception as exc:  # 쓰기 실패: 보존·보고·입력 차단
                self.unwritten.extend(events[i:])
                self.error = f"{type(exc).__name__}: {exc}"
                self.state.fail("b_writer_error")
                break

            done += 1

        return done

    def start(self) -> None:
        """쓰기 스레드 시작(이후 파일 소유자는 그 스레드)."""
        if self._thread is not None:
            raise RuntimeError("writer already started")

        self._thread = threading.Thread(target=self._loop, name="eyetoarm-live-observation-writer", daemon=True)
        self._thread.start()

    def _past_deadline(self) -> bool:
        return self._deadline is not None and time.monotonic() >= self._deadline

    def _loop(self) -> None:
        """쓰기 스레드 본체. 멈춤 요청 뒤에는 기한 안에서 비우고 끝낸다. 파일은 여기서 닫는다."""
        try:
            while self.error is None:
                n = self.drain_once()

                if self._stop.is_set():
                    if n == 0 or self._past_deadline():
                        break
                elif n == 0:
                    self._stop.wait(self._poll_s)
        except Exception as exc:  # 소유권·기타 예외도 보고하고 입력을 막는다
            self.error = self.error or f"{type(exc).__name__}: {exc}"
            self.state.fail("b_writer_error")
        finally:
            self._close_file()

    def _close_file(self) -> None:
        if not self.file_closed:
            self.file_closed = True

            try:
                self._fh.close()
            except Exception as exc:
                self.error = self.error or f"close: {type(exc).__name__}: {exc}"

    def close(self, limit_s: float) -> WriterCloseReport:
        """기한 안에서 비우고 닫는다. 스레드가 기한을 넘기면 손대지 않고 timed_out 보고 + fail-stop."""
        self._deadline = time.monotonic() + limit_s
        timed_out = False

        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=limit_s)

            if self._thread.is_alive():
                timed_out = True
                self.state.fail("b_writer_close_timeout")
        else:
            while not self._past_deadline() and self.drain_once():
                pass

            self._close_file()

        return WriterCloseReport(written=self.written, unwritten=len(self.unwritten),
                                 pending=self.state.pending_events(), timed_out=timed_out, error=self.error)


__all__ = ["LiveObservationEventWriter", "WriterCloseReport", "event_record"]
