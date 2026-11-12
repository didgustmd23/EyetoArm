"""명령 ID, 실행 슬롯, 상태 전이와 영속 로그 색인을 관리합니다."""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .command_log import CommandLog, CommandLogError
from .protocol import ProtocolError, build_command_status, validate_move_request
from .readiness import ReadinessFields

_LOG = logging.getLogger("robot_server_package.command_manager")
TERMINAL_STATES = {"REACHED", "FAILED"}


@dataclass
class CommandRecord:
    """한 MOVE의 요청·상태·잠금과 TCP 구독 정보를 담습니다."""

    command_id: str
    request: dict[str, Any]
    status: dict[str, Any]
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    subscribers: set[int] = field(default_factory=set)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None
    goal_id: str | None = None
    dispatch_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _valid_logged_status(status: Any, command_id: str) -> bool:
    if not isinstance(status, dict):
        return False
    required = {
        "version", "type", "robot_ready", "not_ready_reasons",
        "active_command_id", "server_time_ms", "command_id", "state",
    }
    optional = {"stage", "code", "detail", "reach_error"}
    if not required <= set(status) or set(status) - required - optional:
        return False
    timestamp = status.get("server_time_ms")
    if type(timestamp) is not int or not 0 <= timestamp <= 9_007_199_254_740_991:
        return False
    if status.get("version") != "1.1" or status.get("type") != "STATUS":
        return False
    if status.get("command_id") != command_id:
        return False
    readiness = {
        "robot_ready": status.get("robot_ready"),
        "not_ready_reasons": status.get("not_ready_reasons"),
        "active_command_id": status.get("active_command_id"),
    }
    try:
        expected = build_command_status(
            command_id, status.get("state"), readiness,
            stage=status.get("stage"), code=status.get("code"),
            detail=status.get("detail"), reach_error=status.get("reach_error"),
        )
    except (TypeError, ValueError, KeyError):
        return False
    return set(expected) == set(status) and all(
        status[key] == value for key, value in expected.items() if key != "server_time_ms"
    )


class CommandManager:
    """명령 등록·중복 조회·상태 저장을 TCP 전송 계층에서 분리합니다.

    호출자는 ``commands_lock``을 잡은 상태에서 ``register``를 호출해 BUSY
    판정과 REGISTER 저장·슬롯 확보를 직렬화합니다. 알림은 TCP 계층이 등록한
    notifier를 통해 각 연결의 FIFO로 전달됩니다.
    """

    def __init__(
        self,
        command_log: CommandLog,
        *,
        record_retention: int = 1000,
        notify: Callable[[CommandRecord, dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        if type(record_retention) is not int or record_retention < 1:
            raise ValueError("record_retention은 1 이상인 정수여야 합니다")
        self.command_log = command_log
        self.record_retention = record_retention
        self.notify = notify
        self.records: dict[str, CommandRecord] = {}
        self.command_index: dict[str, tuple[dict[str, Any], int]] = {}
        self.recent_ids: list[str] = []
        self.active_command_id: str | None = None
        self.commands_lock = asyncio.Lock()
        self.log_unavailable = False
        self.pending_terminal_statuses: dict[str, dict[str, Any]] = {}
        self.recovery_failed = False
        self.recovering = True
        self._record_retention = record_retention
        self.load_log()

    def load_log(self) -> None:
        """점검 후 전체 ID 색인과 최근 상세 상태를 다시 만듭니다."""
        try:
            events = self.command_log.load_with_offsets()
        except CommandLogError as error:
            self.recovery_failed = True
            _LOG.error("명령 로그 복구 실패: %s", error)
            return

        for event_offset, event in events:
            command_id = event["command_id"]
            if event["event"] == "REGISTER":
                request = event.get("request")
                status = event.get("status")
                if command_id in self.command_index or not isinstance(request, dict):
                    self._fail_recovery("중복 REGISTER 또는 요청 손상", command_id)
                    return
                try:
                    validate_move_request(request)
                except ProtocolError:
                    self._fail_recovery("REGISTER 요청이 프로토콜과 다름", command_id)
                    return
                if not _valid_logged_status(status, command_id) or status.get("state") != "ACK":
                    self._fail_recovery("REGISTER ACK STATUS 손상", command_id)
                    return
                record = CommandRecord(command_id, copy.deepcopy(request), status)
                self.records[command_id] = record
                self.recent_ids.append(command_id)
                self.command_index[command_id] = (copy.deepcopy(request), event_offset)
                continue

            record = self.records.get(command_id)
            status = event.get("status")
            if record is None or not _valid_logged_status(status, command_id):
                self._fail_recovery("선행 REGISTER 누락 또는 STATE STATUS 손상", command_id)
                return
            state = status.get("state")
            if (state == "MOVING") != (event.get("goal_id") is not None):
                self._fail_recovery("MOVING goal_id 불일치", command_id)
                return
            previous = record.status.get("state")
            allowed = {
                "ACK": {"PLANNING", "FAILED"},
                "PLANNING": {"MOVING", "FAILED"},
                "MOVING": {"REACHED", "FAILED"},
            }
            if state not in allowed.get(previous, set()):
                self._fail_recovery(f"잘못된 상태 전이 {previous} -> {state}", command_id)
                return
            record.status = status
            record.goal_id = event.get("goal_id")
            self.command_index[command_id] = (record.request, event_offset)

        self.trim_recent_records()
        unfinished = [r for r in self.records.values() if r.status.get("state") not in TERMINAL_STATES]
        if unfinished:
            recovering_status: ReadinessFields = {
                "robot_ready": False,
                "not_ready_reasons": ["RECOVERING"],
                "active_command_id": None,
            }
            for record in unfinished:
                previous = record.status
                record.status = build_command_status(
                    record.command_id, previous["state"], recovering_status,
                    detail=previous.get("detail"),
                )
            self.recovering = True
            _LOG.warning("종료되지 않은 명령 %d건을 복구 대기 중", len(unfinished))
        else:
            self.recovering = False

    def _fail_recovery(self, reason: str, command_id: str) -> None:
        self.recovery_failed = True
        _LOG.error("명령 로그 색인 실패: %s, command_id=%s", reason, command_id)

    def find_record(self, command_id: str) -> CommandRecord | None:
        """최근 메모리 또는 전체 영속 색인에서 현재 상태를 반환합니다."""
        if self.recovery_failed:
            return None
        record = self.records.get(command_id)
        if record is not None:
            return record
        indexed = self.command_index.get(command_id)
        if indexed is None:
            return None
        request, status_offset = indexed
        try:
            event = self.command_log.read_record_at(status_offset)
        except CommandLogError as error:
            self.recovery_failed = True
            _LOG.error("색인 상태 읽기 실패: command_id=%s: %s", command_id, error)
            return None
        status = event.get("status")
        if event.get("command_id") != command_id or not _valid_logged_status(status, command_id):
            self.recovery_failed = True
            _LOG.error("색인 상태 내용 불일치: command_id=%s", command_id)
            return None
        return CommandRecord(command_id, request, status)

    def register(self, request: dict[str, Any], ack: dict[str, Any]) -> CommandRecord:
        """REGISTER 저장을 확정하고 새 명령 슬롯과 기록을 생성합니다."""
        command_id = request["command_id"]
        if self.active_command_id is not None:
            raise RuntimeError("실행 슬롯이 이미 점유되어 있습니다")
        self.active_command_id = command_id
        try:
            self.command_log.append("REGISTER", command_id, request=request, status=ack)
        except CommandLogError:
            self.active_command_id = None
            self.log_unavailable = True
            raise
        record = CommandRecord(command_id, copy.deepcopy(request), ack)
        self.records[command_id] = record
        self.command_index[command_id] = (copy.deepcopy(request), self.command_log.last_record_offset)
        self.recent_ids.append(command_id)
        self.trim_recent_records()
        return record

    async def transition(
        self,
        record: CommandRecord,
        state: str,
        readiness: ReadinessFields,
        *,
        stage: str | None = None,
        code: str | None = None,
        detail: str | None = None,
        reach_error: dict[str, float] | None = None,
        goal_id: str | None = None,
        retry_ms: int = 1000,
    ) -> None:
        """검증된 상태 전이를 로그에 저장하고 나서 구독자에게 알립니다."""
        async with record.lock:
            current = record.status.get("state")
            allowed = {
                "ACK": {"PLANNING"},
                "PLANNING": {"MOVING", "FAILED"},
                "MOVING": {"REACHED", "FAILED"},
            }
            recovery_terminal = state == "FAILED" and code == "SERVER_RESTARTED" and self.recovering
            cancel_before_motion = (
                current in {"ACK", "PLANNING"} and state == "FAILED"
                and code == "CANCELLED" and stage == "PLANNING"
            )
            log_failure_before_motion = (
                current in {"ACK", "PLANNING"} and state == "FAILED"
                and code == "LOG_WRITE_FAILED" and stage == "PLANNING"
            )
            if not (recovery_terminal or cancel_before_motion or log_failure_before_motion):
                if state not in allowed.get(current, set()):
                    raise ValueError(f"허용되지 않는 명령 상태 전이: {current} -> {state}")
            terminal = state in TERMINAL_STATES
            if state == "MOVING" and (not isinstance(goal_id, str) or not goal_id):
                raise ValueError("MOVING 상태에는 goal_id가 필요합니다")
            status = build_command_status(
                record.command_id, state, readiness,
                stage=stage, code=code, detail=detail, reach_error=reach_error,
            )
            try:
                fields = {"goal_id": goal_id} if state == "MOVING" else {}
                self.command_log.append("STATE", record.command_id, status=status, **fields)
                offset = self.command_log.last_record_offset
            except CommandLogError as error:
                self.log_unavailable = True
                _LOG.error("명령 상태 저장 실패: command_id=%s: %s", record.command_id, error)
                if terminal:
                    self.pending_terminal_statuses[record.command_id] = status
                    asyncio.create_task(self._retry_terminal(record, status, retry_ms))
                    return
                raise

            self.log_unavailable = bool(self.pending_terminal_statuses)
            record.status = status
            if goal_id is not None:
                record.goal_id = goal_id
            self.command_index[record.command_id] = (record.request, offset)
            if terminal and self.active_command_id == record.command_id:
                self.active_command_id = None
            if terminal:
                self.trim_recent_records()
            await self._notify(record, status)

    async def _retry_terminal(self, record: CommandRecord, status: dict[str, Any], retry_ms: int) -> None:
        while self.pending_terminal_statuses.get(record.command_id) is status:
            await asyncio.sleep(retry_ms / 1000.0)
            try:
                self.command_log.append("STATE", record.command_id, status=status)
                offset = self.command_log.last_record_offset
            except CommandLogError as error:
                _LOG.warning("종료 상태 저장 재시도 실패: command_id=%s: %s", record.command_id, error)
                continue
            async with record.lock:
                if self.pending_terminal_statuses.get(record.command_id) is not status:
                    return
                record.status = status
                self.command_index[record.command_id] = (record.request, offset)
                self.pending_terminal_statuses.pop(record.command_id, None)
                self.log_unavailable = bool(self.pending_terminal_statuses)
                if self.active_command_id == record.command_id:
                    self.active_command_id = None
                self.trim_recent_records()
                await self._notify(record, status)
            return

    async def _notify(self, record: CommandRecord, status: dict[str, Any]) -> None:
        if self.notify is not None:
            await self.notify(record, status)

    def trim_recent_records(self) -> None:
        while len(self.recent_ids) > self.record_retention:
            removable = next(
                (
                    command_id for command_id in self.recent_ids
                    if (record := self.records.get(command_id)) is not None
                    and record.status.get("state") in TERMINAL_STATES
                ),
                None,
            )
            if removable is None:
                return
            self.recent_ids.remove(removable)
            self.records.pop(removable, None)
