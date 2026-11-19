"""통신 시험 기록(#183 계획 r0 §4): 명령별 시간 추적, 명령 기록 행, 실행 폴더·파일 쓰기. 순수(표준 라이브러리).

- 시간은 단조 시계만 쓴다(규약 §4). D `EventLog` 행의 `mono_ms`(1 ms)와 도구 listener가 받은 순간의 `time.monotonic_ns()`를
  함께 남긴다(reviewer Q5). 지연 계산은 `monotonic_ns`로 하고 그 사실을 취합에 적는다.
- 수락 4분류(`first_ack`·`accepted_after_resend`·`accepted_by_status`·`unconfirmed`)는 D 클라이언트가 판정한 값을 그대로
  옮긴다(TCP §8.4). 도구가 다시 판정하지 않는다.
- `CommandTracker.on_row`는 D `EventLog` listener로 등록된다. D-07(listener 예외가 재접속 작업을 끝냄) 때문에 이 함수는
  어떤 행에서도 예외를 내지 않고, 해석 실패는 `errors`에 남긴다.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

FINAL_STATES = ("REACHED", "FAILED")


@dataclass
class CommandTiming:
    """명령 하나의 송수신 시각(단조). ns는 도구 listener 시각, ms는 D EventLog 행 값."""

    command_id: str
    tx_count: int = 0
    tx_first_ns: int | None = None
    tx_first_mono_ms: int | None = None
    rx_first_ns: int | None = None
    rx_first_mono_ms: int | None = None
    rx_first_state: str | None = None
    final_ns: int | None = None
    final_mono_ms: int | None = None
    final_state: str | None = None
    final_code: str | None = None
    states: list = field(default_factory=list)


class CommandTracker:
    """D EventLog listener. 명령 ID별 TX MOVE·RX STATUS 시각을 모은다. 한 스레드(이벤트 루프)에서만 불리지만 잠금으로 보호한다."""

    def __init__(self, clock_ns=time.monotonic_ns) -> None:
        """단조 시계(시험에서 바꿀 수 있음)."""
        self._clock_ns = clock_ns
        self._lock = threading.Lock()
        self.timings: dict[str, CommandTiming] = {}
        self.unsent: list[tuple[str, str]] = []   # (command_id, D 기록 사유): 클라이언트가 보내지 않은 MOVE
        self.errors: list[str] = []

    def on_row(self, row) -> None:
        """EventLog 행 하나. 예외를 내지 않는다(D-07)."""
        try:
            self._apply(row)
        except Exception as exc:  # noqa: BLE001 - listener 최상단: 기록만 하고 D 클라이언트를 멈추지 않는다
            with self._lock:
                self.errors.append(f"{type(exc).__name__}: {exc}"[:200])

    def _apply(self, row: dict) -> None:
        """행 반영(해석 실패는 예외 → on_row가 기록)."""
        now = self._clock_ns()
        command_id = row.get("command_id")

        if command_id is None:
            return

        with self._lock:
            if row.get("dir") == "TX" and row.get("kind") == "MOVE" and row.get("send_state") == "unsent":
                self.unsent.append((command_id, row.get("note") or ""))   # 보내지 않은 MOVE는 시간 항목을 만들지 않는다
                return

            t = self.timings.setdefault(command_id, CommandTiming(command_id))

            if row.get("dir") == "TX" and row.get("kind") == "MOVE" and row.get("send_state") == "sent":
                t.tx_count += 1

                if t.tx_first_ns is None:
                    t.tx_first_ns, t.tx_first_mono_ms = now, row.get("mono_ms")

                return

            if row.get("dir") != "RX" or row.get("kind") != "STATUS" or not row.get("raw"):
                return

            status = json.loads(row["raw"])
            state = status.get("state") or ("REJECT:" + status["code"] if status.get("code") else None)
            t.states.append(state)

            if t.rx_first_ns is None:
                t.rx_first_ns, t.rx_first_mono_ms, t.rx_first_state = now, row.get("mono_ms"), state

            terminal = status.get("state") in FINAL_STATES or (status.get("state") is None and status.get("code"))

            if terminal and t.final_ns is None:
                t.final_ns, t.final_mono_ms = now, row.get("mono_ms")
                t.final_state, t.final_code = status.get("state"), status.get("code")


def _ms(start_ns: int | None, end_ns: int | None) -> float | None:
    """두 단조 ns 사이 ms(둘 중 하나라도 없으면 None)."""
    return None if start_ns is None or end_ns is None else (end_ns - start_ns) / 1e6


def client_command_row(*, case, record, timing: CommandTiming | None, expected: str, outcome: str,
                       fault_kind: str | None, peer: str, executor_calls_delta: int | None) -> dict:
    """D `CommandRecord`와 시간 추적으로 명령 기록 1행을 만든다(via=client)."""
    t = timing or CommandTiming(record.command_id)

    return {
        "scenario": case.scenario, "case": case.case_id, "command_id": record.command_id, "via": "client",
        "t_send_ns": t.tx_first_ns, "t_send_mono_ms": t.tx_first_mono_ms,
        "t_first_status_ns": t.rx_first_ns, "t_first_status_mono_ms": t.rx_first_mono_ms,
        "first_status_state": t.rx_first_state, "t_final_ns": t.final_ns, "t_final_mono_ms": t.final_mono_ms,
        "ack_latency_ms": _ms(t.tx_first_ns, t.rx_first_ns), "final_latency_ms": _ms(t.tx_first_ns, t.final_ns),
        "attempts": record.attempts, "resend_count": max(0, record.attempts - 1), "acceptance": record.acceptance,
        "final_state": record.state, "stage": record.stage, "code": record.code,
        "rejected_code": record.code if record.rejected else None,
        "expected": expected, "outcome": outcome, "fault_injected": fault_kind is not None, "fault_kind": fault_kind,
        "peer": peer, "executor_calls_delta": executor_calls_delta,
    }


def injector_command_row(*, case, command_id: str, sent_ns: int, reply: dict | None, reply_ns: int | None,
                         expected: str, outcome: str, fault_kind: str, peer: str,
                         executor_calls_delta: int | None) -> dict:
    """주입기 송신 1건의 기록 행(via=injector, 항상 fault_injected)."""
    reply = reply or {}

    return {
        "scenario": case.scenario, "case": case.case_id, "command_id": command_id, "via": "injector",
        "t_send_ns": sent_ns, "t_send_mono_ms": sent_ns // 1_000_000,
        "t_first_status_ns": reply_ns, "t_first_status_mono_ms": None if reply_ns is None else reply_ns // 1_000_000,
        "first_status_state": reply.get("state"), "t_final_ns": None, "t_final_mono_ms": None,
        "ack_latency_ms": _ms(sent_ns, reply_ns), "final_latency_ms": None,
        "attempts": 1, "resend_count": 0, "acceptance": None,
        "final_state": reply.get("state"), "stage": reply.get("stage"), "code": reply.get("code"),
        "rejected_code": reply.get("code") if reply and reply.get("state") is None else None,
        "expected": expected, "outcome": outcome, "fault_injected": True, "fault_kind": fault_kind,
        "peer": peer, "executor_calls_delta": executor_calls_delta,
    }


def make_run_dir(base: str | os.PathLike, run_id: str) -> Path:
    """새 실행 폴더(있으면 FileExistsError). run_id에 경로 구분자·`..`이 있으면 ValueError(규약 §10 경로)."""
    if not run_id or "/" in run_id or "\\" in run_id or run_id in (".", "..") or run_id.startswith("."):
        raise ValueError(f"invalid run_id {run_id!r}")

    path = Path(base) / run_id
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)

    return path


def write_jsonl(path: Path, rows) -> None:
    """새 JSONL 파일(있으면 FileExistsError)."""
    with open(path, "x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def write_json(path: Path, obj) -> None:
    """새 JSON 파일(있으면 FileExistsError)."""
    with open(path, "x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, allow_nan=False)
        f.write("\n")
