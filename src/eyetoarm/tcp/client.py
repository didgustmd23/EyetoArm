"""TCP 1.1 클라이언트 (D). TCP 패킷 정의서 §8의 클라이언트 규칙을 구현한다. 서버(C) 코드는 포함하지 않는다.

- 연결 직후와 대기 중에는 command_id 없는 GET_STATUS 로 준비를 확인한다. 한 번에 하나만 보낸다.
- MOVE 는 준비 확인 → 목표 재검사 → 새 UUID → 바이트 보관 → 송신 순서. 재전송은 같은 바이트만.
- 재접속 뒤에는 MOVE 를 다시 보내지 않고 조회만 한다.
- 받은 STATUS 는 검사 후 command_id 로 맞춰 적용하고, 앞 단계로 되돌아가는 응답은 기록만 한다.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .event_log import EventLog
from .protocol import (
    FINAL_STATES, STATE_RANK, MoveTarget, ProtocolError, build_cancel, build_get_status, build_move,
    new_command_id, parse_status, status_kind,
)


def now_ms() -> int:
    return time.monotonic_ns() // 1_000_000


@dataclass
class ClientConfig:
    """정의서 §9 초기값. 2주차 시험 뒤 C·D 가 확정한다."""

    host: str = "127.0.0.1"
    port: int = 47090
    max_line_bytes: int = 4096
    ack_timeout_ms: int = 1000
    max_resend: int = 2
    readiness_poll_ms: int = 1000
    readiness_timeout_ms: int = 1000
    status_watchdog_ms: int = 2000
    connect_timeout_ms: int = 2000
    reconnect_ms: int = 1500           # 재접속 간격. 구현 설정(정의서에 없음)
    target_validity_ms: int = 5000     # 수동 목표의 유효 시간. 구현 설정(소스별 값은 A·B 합의 전)


@dataclass
class CommandRecord:
    command_id: str
    target: MoveTarget
    raw: bytes
    conn_id: int
    created_ms: int
    attempts: int = 0
    state: str | None = None           # 서버가 알려 준 명령 상태
    stage: str | None = None
    code: str | None = None
    detail: str | None = None
    reach_error: dict[str, float] | None = None
    acceptance: str | None = None      # first_ack·accepted_after_resend·accepted_by_status·unconfirmed
    result: str = "송신 대기"          # 화면 문구
    rejected: bool = False
    cancel: str | None = None          # None·requested·unsent
    last_status_ms: int = 0
    first_final: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def finished(self) -> bool:
        return self.rejected or self.state in FINAL_STATES

    @property
    def accepted(self) -> bool:
        return self.state is not None


class TcpClient:
    def __init__(self, config: ClientConfig, log: EventLog, notify: Callable[[], None] = lambda: None) -> None:
        self.config = config
        self.log = log
        self.notify = notify
        self.phase = "연결 안 됨"            # 연결 안 됨·접속 중·연결됨·재접속 대기
        self.conn_id = 0
        self.robot_ready: bool | None = None  # None = 조회 중·미확인. TCP 연결만으로 준비로 보지 않는다
        self.not_ready_reasons: list[str] = []
        self.active_command_id: str | None = None
        self.server_time_ms: int | None = None
        self.ready_note = "미확인"
        self.current: CommandRecord | None = None
        self.history: list[CommandRecord] = []
        self.kpi = {"first_ack": 0, "accepted_after_resend": 0, "accepted_by_status": 0, "unconfirmed": 0,
                    "reached": 0, "failed": 0, "rejected": 0, "resend": 0, "unsent": 0, "invalid_rx": 0}
        self.last_message = ""
        self._want = False
        self._runner: asyncio.Task | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._ready_waiter: asyncio.Future | None = None
        self._ready_routine = False   # 대기 중인 준비 조회가 정기 조회인가
        self._last_ready: tuple | None = None
        self._executing = False
        self._tasks: set[asyncio.Task] = set()

    # ── 연결 ───────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self._writer is not None

    def connect(self, host: str | None = None, port: int | None = None) -> None:
        if host:
            self.config.host = host
        if port:
            self.config.port = port
        self._want = True
        if self._runner is None or self._runner.done():
            self._runner = asyncio.ensure_future(self._run())

    async def disconnect(self, reason: str = "사용자 연결 해제") -> None:
        """연결을 끊어도 로봇은 멈추지 않는다. 진행 중 명령은 '결과 미확인'으로 남는다(§8.5)."""
        self._want = False
        await self._drop(reason)
        self.phase = "연결 안 됨"
        self.notify()

    async def _run(self) -> None:
        while self._want:
            self.phase = "접속 중"
            self.notify()
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(self.config.host, self.config.port), self.config.connect_timeout_ms / 1000)
            except (OSError, asyncio.TimeoutError) as error:
                self.log.add("LOCAL", "연결 실패", note=f"{self.config.host}:{self.config.port} · {error or '시간 초과'}")
                self.phase = "재접속 대기"
                self.notify()
                await asyncio.sleep(self.config.reconnect_ms / 1000)
                continue
            self.conn_id += 1
            self._writer = writer
            self.phase = "연결됨"
            self.robot_ready, self.ready_note = None, "조회 중"
            self.log.add("LOCAL", "연결됨", note=f"{self.config.host}:{self.config.port} · 연결 {self.conn_id}")
            self.notify()
            poller = asyncio.ensure_future(self._poll_loop())
            try:
                await self._read_loop(reader)
            finally:
                poller.cancel()
                await self._drop("서버가 연결을 닫음" if self._want else "")
            if self._want:
                self.phase = "재접속 대기"
                self.notify()
                await asyncio.sleep(self.config.reconnect_ms / 1000)
        self.phase = "연결 안 됨"
        self.notify()

    async def _drop(self, reason: str) -> None:
        writer, self._writer = self._writer, None
        if writer is None:
            return
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            pass
        self.robot_ready, self.ready_note, self._last_ready = None, "미확인", None
        if self._ready_waiter is not None and not self._ready_waiter.done():
            self._ready_waiter.set_result(None)
        record = self.current
        if record is not None and not record.finished:
            record.result = "연결 끊김 · 결과 미확인 · 수락된 동작은 계속될 수 있음"
            record.notes.append("연결 끊김")
        if reason:
            self.log.add("LOCAL", "연결 끊김", note=reason)
        self.notify()

    # ── 송신 ───────────────────────────────────────────────────

    async def _send(self, raw: bytes, kind: str, command_id: str | None = None, note: str = "", **extra: Any) -> bool:
        """소켓에 쓴 것만 sent 로 기록한다. sent 는 서버 도착·수락을 뜻하지 않는다."""
        writer = self._writer
        if writer is None:
            self.log.add("TX", kind, raw=raw, command_id=command_id, send_state="unsent", note=note or "연결 없음", **extra)
            return False
        try:
            writer.write(raw)
            await writer.drain()
        except OSError as error:
            self.log.add("TX", kind, raw=raw, command_id=command_id, send_state="unsent", note=f"쓰기 실패: {error}", **extra)
            await self._drop("쓰기 실패")
            return False
        self.log.add("TX", kind, raw=raw, command_id=command_id, send_state="sent", note=note, conn_id=self.conn_id, **extra)
        return True

    async def query_ready(self, routine: bool = False) -> dict[str, Any] | None:
        """준비 조회. 응답에 어느 조회의 답인지 구분할 값이 없으므로 한 번에 하나만 보낸다(§8.1)."""
        if not self.connected:
            return None
        if self._ready_waiter is not None and not self._ready_waiter.done():
            return await self._ready_waiter
        waiter = self._ready_waiter = asyncio.get_running_loop().create_future()
        self._ready_routine = routine
        if not await self._send(build_get_status(), "GET_STATUS", note="준비 조회", routine=routine):
            return None
        try:
            return await asyncio.wait_for(asyncio.shield(waiter), self.config.readiness_timeout_ms / 1000)
        except asyncio.TimeoutError:
            # 늦게 온 답을 다음 조회의 답으로 오인하지 않도록 이 연결을 버린다
            self.ready_note = "준비 확인 불가"
            self.log.add("LOCAL", "준비 응답 시간 초과", note="새 MOVE 차단 · 연결 해제 후 재접속")
            await self._drop("준비 응답 없음")
            return None

    async def _poll_loop(self) -> None:
        conn = self.conn_id
        await self.query_ready()
        record = self.current
        if record is not None and not record.finished and record.attempts and self.conn_id == conn:
            # 재접속: MOVE 를 다시 보내지 않고 조회만 한다(§8.6)
            await self._send(build_get_status(record.command_id), "GET_STATUS", record.command_id, "재접속 후 명령 조회")
        while self.connected and self.conn_id == conn:
            await asyncio.sleep(self.config.readiness_poll_ms / 1000)
            if not self.connected or self.conn_id != conn:
                break
            await self.query_ready(routine=True)
            record = self.current
            if (record is not None and not record.finished and record.accepted
                    and now_ms() - record.last_status_ms > self.config.status_watchdog_ms):
                record.last_status_ms = now_ms()
                await self._send(build_get_status(record.command_id), "GET_STATUS", record.command_id, "알림 없음 · 명령 조회")

    async def execute(self, target: MoveTarget) -> str:
        """사용자 확정 1회 → 명령 ID 1개 → MOVE 1건. 결과 문구를 돌려준다."""
        if self._executing:
            return self._say("이전 실행 처리 중")
        if self.current is not None and not self.current.finished:
            return self._say("결과 미확인 명령 있음 · 새 MOVE 차단")
        self._executing = True
        try:
            if not self.connected:
                return self._unsent(target, "TCP 연결 없음")
            status = await self.query_ready()
            if status is None:
                return self._unsent(target, "준비 응답 없음 · 목표 폐기, 재접속 후 다시 확정")
            if not status["robot_ready"]:
                return self._unsent(target, "로봇 준비 안 됨: " + ",".join(status["not_ready_reasons"]))
            if now_ms() - target.created_monotonic_ms > self.config.target_validity_ms:
                return self._unsent(target, "목표 유효 시간 초과")
            command_id = new_command_id()
            raw = build_move(command_id, target)  # 이 바이트를 보관해 재전송에 그대로 쓴다
            record = CommandRecord(command_id, target, raw, self.conn_id, now_ms())
            self.current = record
            self.history.append(record)
            if not await self._send(raw, "MOVE", command_id, attempt=1):
                record.rejected, record.result = True, "미전송"
                self.kpi["unsent"] += 1
                return self._say("MOVE 미전송")
            record.attempts, record.result = 1, "송신함 · 수락(ACK) 대기"
            self._spawn(self._ack_timer(record))
            return self._say("MOVE 송신 · 수락 대기")
        finally:
            self._executing = False
            self.notify()

    def _unsent(self, target: MoveTarget, reason: str) -> str:
        """보내지 않은 MOVE 도 어떻게 작성됐는지 기록에 남긴다. 이 ID 는 다시 쓰지 않는다."""
        command_id = new_command_id()
        raw = build_move(command_id, target)
        self.log.add("TX", "MOVE", raw=raw, command_id=command_id, send_state="unsent", note=reason)
        self.kpi["unsent"] += 1
        return self._say(f"MOVE 미전송 · {reason}")

    async def _ack_timer(self, record: CommandRecord) -> None:
        while True:
            await asyncio.sleep(self.config.ack_timeout_ms / 1000)
            if record.accepted or record.rejected:
                return
            same_connection = self.connected and self.conn_id == record.conn_id
            valid = now_ms() - record.target.created_monotonic_ms <= self.config.target_validity_ms
            if not same_connection or not valid or record.attempts > self.config.max_resend:
                record.acceptance, record.result = "unconfirmed", "수락 미확인 · 새 ID 를 자동으로 만들지 않음"
                self.kpi["unconfirmed"] += 1
                why = "연결이 바뀜" if not same_connection else "목표 유효 시간 초과" if not valid else "재전송 상한"
                self.log.add("LOCAL", "수락 미확인", command_id=record.command_id, note=why)
                self.notify()
                return
            record.attempts += 1
            self.kpi["resend"] += 1
            await self._send(record.raw, "MOVE", record.command_id, "ACK 시간 초과 · 같은 바이트 재전송", attempt=record.attempts)
            self.notify()

    async def cancel(self) -> str:
        """정지 버튼. 요청 전송만으로 정지 완료로 표시하지 않는다(§8.7)."""
        record = self.current
        if record is None or record.finished or not record.attempts:
            return self._say("취소 대상 명령 없음")
        if not self.connected:
            record.cancel = "unsent"
            self.log.add("TX", "CANCEL", raw=build_cancel(record.command_id), command_id=record.command_id,
                         send_state="unsent", note="연결 끊김 — 취소 요청을 보낼 수 없음")
            self.notify()
            return self._say("연결 끊김 — 취소 요청을 보낼 수 없음 · 미전송")
        if await self._send(build_cancel(record.command_id), "CANCEL", record.command_id):
            record.cancel = "requested"
            self.notify()
            return self._say("취소 요청 전송 · 종료·정지 확인 중")
        return self._say("취소 요청 미전송")

    # ── 수신 ───────────────────────────────────────────────────

    async def _read_loop(self, reader: asyncio.StreamReader) -> None:
        buffer = b""
        while True:
            try:
                chunk = await reader.read(4096)
            except OSError:
                return
            if not chunk:
                return  # LF 로 끝나지 않은 마지막 조각은 버린다(§2.2)
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                self._on_line(line)
            if len(buffer) > self.config.max_line_bytes:
                buffer = b""
                self.kpi["invalid_rx"] += 1
                self.log.add("RX", "잘못된 응답", note="줄 길이 초과 · 다음 LF 까지 버림")

    def _on_line(self, line: bytes) -> None:
        try:
            if len(line) > self.config.max_line_bytes:
                raise ProtocolError("줄 길이 초과")
            status = parse_status(line)
        except ProtocolError as error:
            self.kpi["invalid_rx"] += 1
            self.log.add("RX", "잘못된 응답", raw=line, note=f"{error} · 기록만 하고 상태에 반영하지 않음")
            self.notify()
            return
        kind = status_kind(status)
        self.robot_ready = status["robot_ready"]
        self.not_ready_reasons = list(status["not_ready_reasons"])
        self.active_command_id = status["active_command_id"]
        self.server_time_ms = status["server_time_ms"]
        self.ready_note = "준비됨" if self.robot_ready else "준비 안 됨"
        note, routine = "", False
        if kind == "ready":
            if self._ready_waiter is not None and not self._ready_waiter.done():
                self._ready_waiter.set_result(status)
            ready = (status["robot_ready"], tuple(status["not_ready_reasons"]), status["active_command_id"])
            routine = self._ready_routine and ready == self._last_ready  # 내용이 바뀐 응답은 화면에도 남긴다
            self._last_ready = ready
            note = "준비 응답"
        elif kind == "command":
            note = self._apply_command(status)
        else:
            note = self._apply_reject(status)
        self.log.add("RX", "STATUS", raw=line, command_id=status.get("command_id"), note=note, conn_id=self.conn_id,
                     routine=routine)
        self.notify()

    def _apply_command(self, status: dict[str, Any]) -> str:
        record = self.current
        if record is None or record.command_id != status["command_id"]:
            return "추적 중인 명령이 아님 · 기록만"
        state = status["state"]
        record.last_status_ms = now_ms()
        if record.acceptance in (None, "unconfirmed"):
            if record.acceptance == "unconfirmed":
                self.kpi["unconfirmed"] -= 1
            if record.conn_id != self.conn_id:
                record.acceptance = "accepted_by_status"
            elif record.attempts > 1:
                record.acceptance = "accepted_after_resend"
            else:
                record.acceptance = "first_ack" if state == "ACK" else "accepted_by_status"
            self.kpi[record.acceptance] += 1
        if record.first_final is not None:
            if state != record.first_final:
                record.notes.append(f"종료 뒤 다른 상태 {state} 수신")
                return "종료 상태 불일치 · 처음 받은 종료 상태 유지 · C 에 알림 대상"
            return "같은 종료 상태 재수신"
        if record.state is not None and STATE_RANK[state] < STATE_RANK[record.state]:
            return "앞 단계 상태 · 기록만 하고 화면에 반영하지 않음"
        record.state, record.stage = state, status.get("stage")
        record.code, record.detail = status.get("code"), status.get("detail")
        record.reach_error = status.get("reach_error")
        if state in FINAL_STATES:
            record.first_final = state
            self.kpi["reached" if state == "REACHED" else "failed"] += 1
            if record.cancel == "requested" and (state == "REACHED" or record.code == "REACH_TOLERANCE"):
                record.result = "취소 요청 전에 이동이 완료됨"
            elif state == "REACHED":
                record.result = "도달 REACHED"
            else:
                record.result = f"실패 FAILED · {record.stage} · {record.code}"
            return record.result
        record.result = {"ACK": "수락 ACK (도달 아님)", "PLANNING": "계획 중", "MOVING": "이동 중"}[state]
        return record.result

    def _apply_reject(self, status: dict[str, Any]) -> str:
        code, command_id = status["code"], status.get("command_id")
        record = self.current
        if record is None or command_id != record.command_id:
            return f"거부 {code} · 추적 중인 명령과 무관 · 기존 명령 상태를 덮지 않음"
        if code == "UNKNOWN_COMMAND":
            record.rejected, record.result = True, "명령 기록 없음 · 실행 여부 미확인"
            return record.result
        if record.accepted:
            return f"거부 {code} · 이미 수락된 명령의 상태는 그대로 둠"
        record.rejected, record.code, record.detail = True, code, status.get("detail")
        record.result = f"거부 · {code}"
        self.kpi["rejected"] += 1
        return record.result

    # ── 보조 ───────────────────────────────────────────────────

    def _say(self, text: str) -> str:
        self.last_message = text
        self.notify()
        return text

    def _spawn(self, coroutine: Any) -> None:
        task = asyncio.ensure_future(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def snapshot(self) -> dict[str, Any]:
        record = self.current
        return {
            "phase": self.phase, "host": self.config.host, "port": self.config.port, "conn_id": self.conn_id,
            "robot_ready": self.robot_ready, "ready_note": self.ready_note, "not_ready_reasons": self.not_ready_reasons,
            "active_command_id": self.active_command_id, "server_time_ms": self.server_time_ms,
            "message": self.last_message, "kpi": dict(self.kpi), "commands": len(self.history),
            "current": None if record is None else {
                "command_id": record.command_id, "position_mm": list(record.target.position_mm),
                "theta_deg": record.target.theta_deg, "source_id": record.target.source_id,
                "image_seq": record.target.image_seq, "calibration_id": record.target.calibration_id,
                "attempts": record.attempts, "state": record.state, "stage": record.stage, "code": record.code,
                "result": record.result, "acceptance": record.acceptance, "cancel": record.cancel,
                "reach_error": record.reach_error, "finished": record.finished, "notes": record.notes,
            },
        }
