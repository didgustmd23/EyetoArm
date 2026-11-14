"""TCP 1.1 GUI client: one Qt connection for readiness, MOVE and command tracking."""

from __future__ import annotations

import ipaddress
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtNetwork import QAbstractSocket, QHostAddress, QTcpSocket

from eyetoarm.gui.diag import safe_text
from eyetoarm.gui.records import SessionLog
from eyetoarm.tcp import protocol

CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "gui" / "client.json"

# client.json 을 읽지 못할 때 쓰는 값. TCP 명세 §9 초기값과 같다
DEFAULTS = {"host": "127.0.0.1", "port": 47090, "max_line_bytes": 4096, "readiness_poll_ms": 1000,
            "status_watchdog_ms": 2000, "connect_timeout_ms": 2000, "reconnect_ms": 1500,
            "readiness_timeout_ms": 1000, "ack_timeout_ms": 1000, "max_resend": 2, "target_validity_ms": 5000}


def load_tcp_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """[공용 설정 읽기] client.json 의 tcp 항목에서 D가 쓰는 키만 꺼낸다. 파일은 바꾸지 않는다.

    서버 전용 키(record_retention · command_log_path 등)는 C 소유이므로 읽지 않는다.
    """
    config = dict(DEFAULTS)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        loaded = document.get("tcp", {}) if isinstance(document, dict) else {}
    except (OSError, ValueError):
        return config
    if not isinstance(loaded, dict):
        return config
    for key, default in DEFAULTS.items():
        if type(loaded.get(key)) is type(default):
            if isinstance(default, int) and (loaded[key] < (0 if key == "max_resend" else 1)
                                             or key == "port" and loaded[key] > 65535):
                continue
            config[key] = loaded[key]
    return config


def split_lines(buffer: bytearray, max_line_bytes: int) -> list[bytes]:
    """수신 버퍼에서 완성된 줄(LF 제외)을 모두 꺼내고, 미완성 꼬리는 buffer 에 남긴다.

    TCP 는 바이트 흐름이라 한 번의 수신에 줄이 쪼개지거나 여러 줄이 붙어 올 수 있다(명세 §2).
    한 줄이 max_line_bytes 를 넘으면 OverflowError. 호출 측은 연결을 끊고 다시 접속한다.
    """
    lines: list[bytes] = []
    while True:
        index = buffer.find(b"\n")
        if index < 0:
            if len(buffer) > max_line_bytes + 1:  # +1: LF 앞 CR 허용
                raise OverflowError(f"LF 없이 {len(buffer)}바이트 수신, 한도 {max_line_bytes}")
            return lines
        line = bytes(buffer[:index]).removesuffix(b"\r")
        if len(line) > max_line_bytes:
            raise OverflowError(f"한 줄이 {max_line_bytes}바이트를 넘음")
        del buffer[:index + 1]
        lines.append(line)


@dataclass(frozen=True)
class TrafficEntry:
    """GUI-03 화면 요약. 전체 원문은 세션 JSONL, 현재 보이는 표는 CSV로 저장한다."""
    wall_time: str      # 이 PC 벽시계 HH:MM:SS.mmm (서버 시각과 다른 시계)
    direction: str      # TX(보냄) · RX(받음) · EV(연결 사건)
    kind: str           # 준비 조회 · STATUS · 접속 · 끊김 등
    summary: str        # 사람이 읽는 한 줄
    raw: str = ""       # 실제 보낸/받은 줄(제어 문자 이스케이프, 길이 제한)


@dataclass
class CommandRecord:
    command_id: str
    target: protocol.MoveTarget
    raw: bytes
    conn_id: int
    attempts: int = 0
    state: str | None = None
    stage: str | None = None
    code: str | None = None
    detail: str | None = None
    reach_error: dict | None = None
    acceptance: str | None = None
    result: str = "준비 확인 중"
    rejected: bool = False
    cancel: str | None = None
    closed_unconfirmed: bool = False

    @property
    def finished(self):
        return self.closed_unconfirmed or self.rejected or self.state in protocol.FINAL_STATES


_DEFAULT_LOG = object()


class StatusLink(QObject):
    """One Qt connection for readiness and command tracking."""
    state_changed = Signal(str)
    status_received = Signal(dict)
    traffic = Signal(object)
    command_changed = Signal(object)
    log_error = Signal(str)

    def __init__(self, host=None, port=None, parent=None, *, log_path=_DEFAULT_LOG, stable_count=3, stable_ms=5000, config=None):
        super().__init__(parent)
        self.config = load_tcp_config()
        if config:
            self.config.update(config)
        self.host = self.config["host"] if host is None else host
        self.port = self.config["port"] if port is None else port
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("port must be an integer between 1 and 65535")
        ipaddress.ip_address(self.host)
        self.state = "disconnected"
        self.last_status = None
        self._last_status_at = None
        self.current_record = None
        self.log = SessionLog(SessionLog.default_path() if log_path is _DEFAULT_LOG else log_path)
        self.stable_count = max(1, int(stable_count))
        self.stable_ms = max(0, int(stable_ms))
        self.suppressed_count = 0
        self._fingerprint = None
        self._stable_since = 0.0
        self._unchanged = 0
        self._suppressing = False
        self._last_disconnect_reason = None
        self._buffer = bytearray()
        self._conn_id = 0
        self._want = False
        self._ready_pending = False
        self._pending_move = False
        self._target_invalid = False
        self._last_command_at = 0.0
        self._socket = QTcpSocket(self)
        self._socket.connected.connect(self._on_connected)
        self._socket.disconnected.connect(self._on_disconnected)
        self._socket.readyRead.connect(self._on_ready_read)
        self._socket.errorOccurred.connect(self._on_error)
        self._poll = QTimer(self, interval=self.config["readiness_poll_ms"])
        self._poll.timeout.connect(self._send_get_status)
        self._retry = QTimer(self, interval=self.config["reconnect_ms"], singleShot=True)
        self._retry.timeout.connect(self._open)
        self._connect_deadline = QTimer(self, interval=self.config["connect_timeout_ms"], singleShot=True)
        self._connect_deadline.timeout.connect(self._on_connect_timeout)
        self._ready_deadline = QTimer(self, interval=self.config["readiness_timeout_ms"], singleShot=True)
        self._ready_deadline.timeout.connect(self._ready_timeout)
        self._ack_deadline = QTimer(self, interval=self.config["ack_timeout_ms"], singleShot=True)
        self._ack_deadline.timeout.connect(self._ack_timeout)

    def start(self):
        self._want = True
        self._open()

    def stop(self):
        self._last_disconnect_reason = None
        if self._want or self.state != "disconnected":
            self._log("EV", "세션 종료", "사용자 종료 · C 서버 프로세스는 유지")
        self._want = False
        self.log.retry()
        if self._buffer:
            self._log("RX", "미완성 수신", "연결 종료 전 LF 없는 바이트", bytes(self._buffer))
            self._buffer.clear()
        for timer in (self._poll, self._retry, self._connect_deadline, self._ready_deadline, self._ack_deadline):
            timer.stop()
        self._ready_pending = False
        self._pending_move = False
        if self.current_record and not self.current_record.finished:
            self.current_record.result = "연결 종료 · 실행 결과 미확인"
            self.command_changed.emit(self.current_record)
        self._last_status_at = None
        self.last_status = None
        self._set_state("disconnected")
        self._socket.abort()

    def status_age_ms(self):
        return None if self._last_status_at is None else int((time.monotonic() - self._last_status_at) * 1000)

    def is_stale(self):
        age = self.status_age_ms()
        return self.state != "connected" or age is None or age > self.config["status_watchdog_ms"]

    def can_execute(self):
        return (self.state == "connected" and not self.is_stale() and bool(self.last_status and self.last_status["robot_ready"])
                and not self._pending_move and (self.current_record is None or self.current_record.finished))

    def _target_valid(self, target):
        age = time.monotonic_ns() // 1_000_000 - target.created_monotonic_ms
        return not self._target_invalid and target.created_monotonic_ms > 0 and 0 <= age <= self.config["target_validity_ms"]

    def execute(self, target):
        if not self.can_execute():
            return ""
        self._target_invalid = False
        if not self._target_valid(target):
            return ""
        command_id = protocol.new_command_id()
        try:
            raw = protocol.build_move(command_id, target)
        except (ValueError, TypeError, OverflowError):
            return ""
        self.current_record = CommandRecord(command_id, target, raw, self._conn_id)
        self._pending_move = True
        self.command_changed.emit(self.current_record)
        # An already outstanding readiness response predates this request: wait for it,
        # then send a new query to guarantee the response follows user confirmation.
        if not self._ready_pending:
            self._send_get_status()
        else:
            self._pending_move = "await_previous"
        return command_id

    def invalidate_target(self, reason):
        self._target_invalid = True
        record = self.current_record
        if record and self._pending_move:
            self._pending_move = False
            record.rejected = True
            record.result = "미전송 · " + str(reason)
            self._log("TX", "MOVE", record.result, record.raw, command_id=record.command_id, send_state="unsent")
            self.command_changed.emit(record)

    def confirm_new_command(self):
        """사용자가 미확인 결과를 보존하고 새 목표를 재확정한다. 재송신하지 않는다."""
        record = self.current_record
        if (not record or record.finished or record.acceptance != "unconfirmed" or self.is_stale()
                or not self.last_status or not self.last_status["robot_ready"]):
            return False
        self._ack_deadline.stop()
        self._pending_move = False
        record.closed_unconfirmed = True
        record.result = "이전 결과 미확인 보존 · 사용자 새 목표 재확정 필요"
        self._log("EV", "미확인 기록 보존", record.result, command_id=record.command_id)
        self.command_changed.emit(record)
        return True

    def cancel(self):
        record = self.current_record
        if not record or record.finished or not record.attempts:
            return False
        if self._send(protocol.build_cancel(record.command_id), "CANCEL", record.command_id):
            record.cancel = "requested"
            record.result = "취소 요청 전송 · 종료 미확인"
            self.command_changed.emit(record)
            return True
        record.cancel = "unsent"
        self.command_changed.emit(record)
        return False

    def query_command(self, command_id):
        record = self.current_record
        if not record or command_id != record.command_id or not record.attempts:
            return False
        return self._send(protocol.build_get_status(command_id), "GET_STATUS", command_id)

    def _open(self):
        if not self._want or self._socket.state() != QAbstractSocket.SocketState.UnconnectedState:
            return
        self._buffer.clear()
        self._set_state("connecting")
        self._connect_deadline.start()
        self._socket.connectToHost(QHostAddress(self.host), self.port)

    def _on_connected(self):
        self._connect_deadline.stop()
        self._last_disconnect_reason = None
        self._conn_id += 1
        self._set_state("connected")
        self._log("EV", "접속", f"{self.host}:{self.port} 연결됨")
        self._send_get_status()
        record = self.current_record
        if record and not record.finished and record.attempts:
            self.query_command(record.command_id)
        self._poll.start()

    def _on_disconnected(self):
        self._drop("서버가 연결을 닫음")

    def _on_error(self, _error):
        self._drop(safe_text(self._socket.errorString()))

    def _on_connect_timeout(self):
        self._drop("접속 기한 초과")

    def _ready_timeout(self):
        self._drop("준비 응답 시간 초과")

    def _drop(self, reason):
        if self.state == "disconnected":
            return
        if self._buffer:
            self._log("RX", "미완성 수신", "연결 종료 전 LF 없는 바이트", bytes(self._buffer))
            self._buffer.clear()
        for timer in (self._poll, self._connect_deadline, self._ready_deadline, self._ack_deadline):
            timer.stop()
        self._ready_pending = False
        record = self.current_record
        if record and not record.finished:
            if self._pending_move and not record.attempts:
                record.rejected = True
                record.result = "미전송 · 연결 끊김"
            else:
                record.acceptance = record.acceptance or "unconfirmed"
                record.result = "연결 끊김 · 실행 결과 미확인"
            self.command_changed.emit(record)
        self._pending_move = False
        self._last_status_at = None
        self.last_status = None
        self._fingerprint = None
        self._suppressing, self._unchanged = False, 0
        self._set_state("disconnected")
        self._socket.abort()
        self._log("EV", "끊김", reason, visible=reason != self._last_disconnect_reason)
        self._last_disconnect_reason = reason
        if self._want:
            self._retry.start()

    def _set_state(self, state):
        if state != self.state:
            self.state = state
            self.state_changed.emit(state)

    def _send(self, raw, kind, command_id=None, *, visible=True, note=""):
        written = self._socket.write(raw) if self.state == "connected" else -1
        sent = written == len(raw)
        self._log("TX", kind, note or ("소켓 기록" if sent else "쓰기 실패 또는 일부 기록"), raw,
                  visible=visible, command_id=command_id,
                  send_state="queued" if sent else "partial" if written > 0 else "unsent", written=written,
                  delivery="qt_buffered" if sent else "unconfirmed")
        if not sent and self.state == "connected":
            self._drop("소켓 쓰기 실패 또는 일부 기록 · 실행 여부 미확인")
        return sent

    def _send_get_status(self):
        if self.state != "connected" or self._ready_pending:
            return
        if self._send(protocol.build_get_status(), "준비 조회", visible=not self._suppressing):
            self._ready_pending = True
            self._ready_deadline.start()
        record = self.current_record
        if (record and not record.finished and record.attempts and
                (time.monotonic() - self._last_command_at) * 1000 > self.config["status_watchdog_ms"]):
            self._last_command_at = time.monotonic()
            self.query_command(record.command_id)

    def _ack_timeout(self):
        record = self.current_record
        if not record or record.finished or record.state:
            return
        if (self.state != "connected" or self._conn_id != record.conn_id or not self._target_valid(record.target)
                or record.attempts >= 1 + min(2, self.config["max_resend"])):
            record.acceptance = "unconfirmed"
            record.result = "수락 미확인 · 실행 여부 미확인"
            self._log("EV", "수락 미확인", record.result, command_id=record.command_id, attempts=record.attempts)
            self.command_changed.emit(record)
            return
        if self._send(record.raw, "MOVE", record.command_id, note="동일 바이트 재전송"):
            record.attempts += 1
            self.command_changed.emit(record)
            self._ack_deadline.start()

    def _on_ready_read(self):
        self._buffer += bytes(self._socket.readAll())
        while b"\n" in self._buffer:
            index = self._buffer.index(b"\n")
            raw = bytes(self._buffer[:index + 1])
            del self._buffer[:index + 1]
            if len(raw.removesuffix(b"\n").removesuffix(b"\r")) > self.config["max_line_bytes"]:
                self._log("RX", "형식 오류", "수신 한도 초과", raw)
                self._drop("수신 한도 초과")
                return
            self._handle_line(raw)
        if len(self._buffer) > self.config["max_line_bytes"] + 1:
            self._log("RX", "형식 오류", "수신 한도 초과", bytes(self._buffer))
            self._buffer.clear()
            self._drop("수신 한도 초과")

    def _handle_line(self, raw):
        try:
            message = protocol.parse_status(raw.removesuffix(b"\n"))
        except protocol.ProtocolError as error:
            self._log("RX", "형식 오류", safe_text(error), raw)
            return
        kind = protocol.status_kind(message)
        if kind == "ready":
            self._ready_deadline.stop()
            was_pending = self._ready_pending
            self._ready_pending = False
            self.last_status = message
            self._last_status_at = time.monotonic()
            fingerprint = self._ready_key(message)
            changed = fingerprint != self._fingerprint
            if changed:
                if self._suppressing:
                    self._log("EV", "표시 재개", f"반복 응답 {self.suppressed_count}건 생략")
                self._fingerprint, self._stable_since, self._unchanged, self._suppressing = fingerprint, time.monotonic(), 0, False
            else:
                self._unchanged += 1
                if not self._suppressing and (self._unchanged >= self.stable_count or (time.monotonic() - self._stable_since) * 1000 >= self.stable_ms):
                    self._suppressing = True
                    self._log("EV", "반복 표시 생략", "동일한 준비 응답")
            if self._suppressing:
                self.suppressed_count += 1
            self._log("RX", "STATUS", safe_text(f"robot_ready={message['robot_ready']} 사유={message['not_ready_reasons']}"), raw,
                      visible=not self._suppressing)
            if self._pending_move == "await_previous" and was_pending:
                self._pending_move = True
                self._send_get_status()
            elif self._pending_move and was_pending:
                self._complete_pending(message)
            if changed:
                self.status_received.emit(message)
            return
        self.last_status = message
        self._last_status_at = time.monotonic()
        ready_key = self._ready_key(message)
        if ready_key != self._fingerprint:
            self._fingerprint = ready_key
            self._unchanged, self._suppressing = 0, False
            self._stable_since = time.monotonic()
            self.status_received.emit(message)
        record = self.current_record
        repeated = (kind == "command" and record and not record.finished
                    and message.get("command_id") == record.command_id
                    and message.get("state") not in protocol.FINAL_STATES
                    and all(message.get(key) == getattr(record, key) for key in ("state", "stage", "code", "detail", "reach_error")))
        self._log("RX", "STATUS", safe_text(f"kind={kind} code={message.get('code')}"), raw,
                  command_id=message.get("command_id"), visible=not repeated)
        if repeated:
            self._last_command_at = time.monotonic()
            self.suppressed_count += 1
            return
        self._apply_command(message, kind)

    @staticmethod
    def _ready_key(message):
        return (message["robot_ready"], tuple(sorted(message["not_ready_reasons"])), message["active_command_id"])

    def _complete_pending(self, status):
        record = self.current_record
        self._pending_move = False
        if not record:
            return
        if not status["robot_ready"] or self.is_stale() or not self._target_valid(record.target):
            record.rejected = True
            record.result = "미전송 · 준비 또는 목표 유효성 확인 실패"
            self._log("TX", "MOVE", record.result, record.raw, command_id=record.command_id, send_state="unsent")
            self.command_changed.emit(record)
            return
        # A short socket write might have transmitted a prefix; mark outcome unknown.
        record.attempts = 1
        if self._send(record.raw, "MOVE", record.command_id):
            record.result = "송신 · 수락 대기"
            self.command_changed.emit(record)
            self._ack_deadline.start()

    def _apply_command(self, status, kind):
        record = self.current_record
        if not record or status.get("command_id") != record.command_id:
            return
        if record.finished:
            if kind == "command" and record.state in protocol.FINAL_STATES and status["state"] != record.state:
                self._log("EV", "종료 상태 불일치", "기존 종료 상태 유지", command_id=record.command_id)
            return
        if kind == "reject":
            code = status.get("code")
            if code == "UNKNOWN_COMMAND":
                self._ack_deadline.stop()
                record.acceptance = "unconfirmed"
                record.code, record.detail = code, status.get("detail")
                record.result = "서버 기록 없음 · 실행 여부 미확인"
            elif record.cancel == "requested" and record.state is None:
                self._ack_deadline.stop()
                record.acceptance = "unconfirmed"
                record.result = f"취소 또는 명령 거부 {code} · 실행 여부 미확인"
            elif record.state is None:
                record.rejected, record.code, record.detail = True, code, status.get("detail")
                record.result = f"거부 · {code}"
                self._ack_deadline.stop()
                self._send_get_status()
            self.command_changed.emit(record)
            return
        state = status["state"]
        self._last_command_at = time.monotonic()
        if record.state and protocol.STATE_RANK[state] < protocol.STATE_RANK[record.state]:
            return
        self._ack_deadline.stop()
        if record.acceptance in (None, "unconfirmed"):
            record.acceptance = ("accepted_by_status" if record.conn_id != self._conn_id
                                 else "accepted_after_resend" if record.attempts > 1
                                 else "first_ack" if state == "ACK" else "accepted_by_status")
        record.state, record.stage, record.code = state, status.get("stage"), status.get("code")
        record.detail, record.reach_error = status.get("detail"), status.get("reach_error")
        record.result = {"ACK": "수락 ACK · 도달 아님", "PLANNING": "계획 중", "MOVING": "이동 중",
                         "REACHED": "도달 REACHED", "FAILED": "실패 FAILED"}[state]
        self.command_changed.emit(record)
        if record.finished:
            self.last_status = None
            self._last_status_at = None
            self._fingerprint = None
            self._send_get_status()

    def _log(self, direction, kind, summary, raw=b"", *, visible=True, **fields):
        now = time.time()
        wall = time.strftime("%H:%M:%S", time.localtime(now)) + f".{int(now * 1000) % 1000:03d}"
        previous_error = self.log.write_error
        self.log.add(direction, kind, raw=raw, visible=visible, note=summary, **fields)
        if self.log.write_error != previous_error:
            self.log_error.emit(self.log.write_error)
        if visible:
            display = safe_text(raw.decode("utf-8", errors="replace").rstrip("\r\n"), 400) if raw else ""
            self.traffic.emit(TrafficEntry(wall, direction, kind, summary, display))
