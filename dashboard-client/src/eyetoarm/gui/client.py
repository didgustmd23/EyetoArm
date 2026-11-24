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

from eyetoarm.gui.diag import safe_text, validate_target
from eyetoarm.gui.records import SessionLog
from eyetoarm.tcp import protocol

# dashboard-client 폴더 자신의 config 만 쓴다. 저장소 안에서 실행해도 주변(저장소 공용) 설정을 찾지 않는다
_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = _ROOT / "config" / "gui" / "client.json"

# client.json 을 읽지 못할 때 쓰는 값. TCP 명세 §9 초기값과 같다
DEFAULTS = {"host": "127.0.0.1", "port": 47090, "max_line_bytes": 4096, "readiness_poll_ms": 1000,
            "status_watchdog_ms": 2000, "connect_timeout_ms": 2000, "reconnect_ms": 1500,
            "readiness_timeout_ms": 1000, "ack_timeout_ms": 1000, "max_resend": 2, "target_validity_ms": 5000,
            # TCP 1.2 후보(부록 A.6.11 후보값, C 미확정). --tcp-version 1.2 에서만 쓴다
            "run_reply_timeout_ms": 3000, "run_max_resend": 2,
            # E-Stop 응용 요청(D 제안). 보낸 뒤 이 시간 안에 C 최종 보고가 없으면 미확인으로 닫는다. 자동 재전송 없음
            "e_stop_timeout_ms": 10000}


def load_tcp_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """[설정 읽기] dashboard-client/config/gui/client.json 의 tcp 항목에서 D가 쓰는 키만 꺼낸다. 파일은 바꾸지 않는다.

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
            if isinstance(default, int) and (loaded[key] < (0 if key.endswith("max_resend") else 1)
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
    requested_mono_ms: int | None = None   # 이 PC monotonic. 유효한 execute() 수락 시각
    ack_latency_ms: int | None = None      # 첫 지표 한 번만 기록. 중복 ACK 로 바꾸지 않는다

    @property
    def finished(self):
        return self.closed_unconfirmed or self.rejected or self.state in protocol.FINAL_STATES

    def ack_metric(self) -> tuple[int | None, str]:
        """(사용자 실행→첫 ACK ms 또는 None, 숫자가 없는 사유). 정의는 ACK_METRIC_BOUNDARY."""
        if self.ack_latency_ms is not None:
            return self.ack_latency_ms, ""
        if self.rejected:
            return None, "미전송 또는 거부 · 측정 없음"
        return None, ACK_METRIC_EXCLUDED.get(self.acceptance, "ACK 대기 중")


@dataclass
class SortRequest:
    """TCP 1.2 후보 START·STOP·RECOVER 한 건. 요청 결과일 뿐 실행 상태가 아니다(상태는 스냅샷으로만)."""
    request_id: str
    action: str
    raw: bytes
    conn_id: int
    base: tuple | None            # 보낼 때 반영 중이던 (c_session_id, run_revision). 없으면 None
    attempts: int = 0
    deadline_ms: int = 0
    outcome: str | None = None    # 서버가 확정한 ACCEPTED·REJECTED·UNKNOWN, 또는 PENDING
    code: str | None = None
    detail: str | None = None
    final: bool = False           # 더 기다리지 않음(서버 확정 또는 D 가 미확인으로 닫음)
    result: str = "응답 대기"


HISTORY = "이전 연결 기록(현재 상태 아님) · "


@dataclass
class EStopRequest:
    """E-Stop 응용 요청 한 건(D 제안 · C 미합의). C 가 보고한 state 만 적는다. D 는 정지를 추정하지 않는다."""
    request_id: str
    raw: bytes
    conn_id: int
    deadline_ms: int
    state: str | None = None      # C 보고: REQUESTED(접수만)·CONFIRMED·REJECTED·UNCONFIRMED
    detail: str | None = None
    final: bool = False
    result: str = "응답 대기"


# 사용자 실행→ACK 지표의 경계. 화면 툴팁과 세션 JSONL 에 같은 문장을 쓴다
ACK_METRIC_BOUNDARY = ("시작: 사용자가 확정한 MOVE 실행 요청이 수락된 시각(새 GET_STATUS 준비 응답 대기 포함). "
                       "끝: 같은 연결·첫 송신(재전송 없음)에 대한 첫 ACK 수신. "
                       "REACHED 도달이나 서버 실행 시간이 아니며, 이 PC monotonic 시계로 잰다.")
ACK_METRIC_EXCLUDED = {
    "accepted_after_resend": "재전송 뒤 수락 · 측정 제외",
    "accepted_by_status": "ACK 없이 상태로 수락 확인(재접속·PLANNING 등) · 측정 제외",
    "unconfirmed": "수락 미확인 · 측정 없음",
}


def _now_ms() -> int:
    return time.monotonic_ns() // 1_000_000


_DEFAULT_LOG = object()


def endpoint_error(host, port):
    """접속 주소 검사. 문제가 없으면 "", 있으면 사유 문장을 돌려준다(host 빈칸은 거부).

    숫자 IP만 받고(diag.validate_target 재사용) 0.0.0.0·::·멀티캐스트·브로드캐스트는 접속 주소로 쓰지 않는다."""
    try:
        validate_target(host, port, 1.0)
    except ValueError as error:
        return str(error)
    address = ipaddress.ip_address(host)
    if address.is_unspecified or address.is_multicast or address == ipaddress.IPv4Address("255.255.255.255"):
        return f"접속 주소로 쓸 수 없는 IP: {host}"
    return ""


class StatusLink(QObject):
    """One Qt connection for readiness and command tracking."""
    state_changed = Signal(str)
    status_received = Signal(dict)
    traffic = Signal(object)
    command_changed = Signal(object)
    log_error = Signal(str)
    run_changed = Signal()   # 1.2 후보: 스냅샷·요청 결과·서버 확인 여부가 바뀜

    def __init__(self, host=None, port=None, parent=None, *, log_path=_DEFAULT_LOG, stable_count=3, stable_ms=5000, config=None,
                 version=protocol.VERSION):
        super().__init__(parent)
        if version not in (protocol.VERSION, protocol.VERSION_12):
            raise ValueError(f"지원하지 않는 TCP 버전 {version!r}")
        # "1.2" 는 부록 A.6 후보(미승인)를 고른 명시적 호환 모드. 현재 C 서버(1.1)는 이 줄을 거부한다
        self.version = version
        self.run_snapshot = None      # 마지막으로 반영한 유효 스냅샷(A.6.7 4). None = 유효 스냅샷 없음
        self._run_high = None         # 지금 연결에서 본 가장 높은 (c_session_id, run_revision)
        self._retired_sessions = set()  # 지금 연결에서 새 세션으로 교체된 c_session_id
        self.run_verified = False     # 지금 연결에서 검사를 통과한 1.2 응답을 받았는지(조작 관문)
        self.peer_mismatch = False    # 지금 연결의 상대가 1.1 STATUS 로 답함 = 1.2 후보 미지원
        self.sort_requests: dict[str, SortRequest] = {}
        self.last_request = None
        self.e_stop_available = False   # 지금 연결의 최신 유효 1.2 준비 STATUS 의 e_stop_available (생략 = false)
        self.e_stop = None              # 마지막 E-Stop 요청(EStopRequest)
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
        self._socket = self._new_socket()
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
        self._run_timer = QTimer(self, interval=50)   # 1.2 후보 요청 응답 기한 검사. 대기 요청이 있을 때만 돈다
        self._run_timer.timeout.connect(self._run_tick)

    def _new_socket(self):
        sock = QTcpSocket(self)
        sock.connected.connect(self._on_connected)
        sock.disconnected.connect(self._on_disconnected)
        sock.readyRead.connect(self._on_ready_read)
        sock.errorOccurred.connect(self._on_error)
        return sock

    def start(self):
        self._want = True
        self._open()

    def _unconfirmed(self):
        record = self.current_record
        return (record is not None and record.state not in protocol.FINAL_STATES and not record.rejected
                and (record.closed_unconfirmed or record.acceptance == "unconfirmed"))

    def endpoint_busy(self, host=None, port=None, keep_unconfirmed=False):
        """주소 적용을 거부할 이유(없으면 ""). host·port 가 지금과 다르면 다른 서버로 옮기는 것으로 보고 더 엄격히 본다.

        keep_unconfirmed: 사용자가 「미확인 보존 후 전환」을 명시한 경우. 이전 서버가 끊겨 유효 STATUS 가 없을 때만
        미확인 결과를 보존한 채 옮기도록 허용한다. 동작 중인 서버가 있으면 그 서버에서 결과를 확인해야 한다."""
        record = self.current_record
        if self._pending_move:
            return "명령 전송 대기 중 · 끝난 뒤 바꾸세요"
        changed = host is not None and (host, port) != (self.host, self.port)
        if self._unconfirmed():
            if not changed:
                return "" if record.finished else "명령 결과 대기 중 · 끝난 뒤 바꾸세요"
            if not keep_unconfirmed:
                return "지금 서버의 명령 결과가 미확인 · 확인하거나 「미확인 보존 후 전환」을 선택하세요"
            if self.state == "connected" or not self.is_stale():
                return "지금 서버가 응답 중 · 그 서버에서 결과를 확인한 뒤 바꾸세요(보존 전환은 끊긴 서버만)"
            return ""
        if record is not None and not record.finished:
            return "명령 처리 중이거나 결과 대기 중 · 끝난 뒤 바꾸세요"
        active = (self.last_status or {}).get("active_command_id") if changed else None
        if active:
            return f"지금 서버가 명령 {str(active)[:8]} 실행 중이라고 보고 · 끝난 뒤 바꾸세요"
        return ""

    def set_endpoint(self, host, port, keep_unconfirmed=False):
        """명시적 적용 때만 부른다. 검사 실패·명령 진행 중이면 연결과 주소를 그대로 두고 사유를 돌려준다."""
        error = endpoint_error(host, port) or self.endpoint_busy(host, port, keep_unconfirmed)
        if error:
            return error
        record = self.current_record
        if (host, port) != (self.host, self.port) and self._unconfirmed():
            # 기록은 지우지 않는다. 이전 서버 결과는 끝내 미확인으로 남기고 이 사실을 감사 기록에 남긴다
            record.closed_unconfirmed = True
            record.result = f"이전 서버 {self.host}:{self.port} 결과 미확인 보존 · 주소 변경"
            self._log("EV", "미확인 보존 · 주소 변경", f"{record.result} → {host}:{port}", command_id=record.command_id)
            self.command_changed.emit(record)
        old = f"{self.host}:{self.port}"
        self._drop("주소 적용 · 재접속")   # 준비 상태·버퍼를 비운다. 미확인 명령은 위에서 거부했다
        self._retry.stop()
        if (host, port) != (self.host, self.port):
            # 옛 소켓의 늦은 신호가 새 연결 상태를 바꾸지 못하게 소켓을 통째로 바꾼다
            old_socket, self._socket = self._socket, self._new_socket()
            old_socket.blockSignals(True)
            old_socket.abort()
            old_socket.deleteLater()
            self.host, self.port = host, port
        self._last_disconnect_reason = None
        self._log("EV", "주소 적용", f"{old} → {self.host}:{self.port}")
        if self._want:
            self._open()
        return ""

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
        self._reset_run("연결 종료")
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
        # 1.2 후보 모드에서는 수동 MOVE·CANCEL 을 쓰지 않는다: 1.2 명령 STATUS 의 slot_owner 일관성 규칙이 명세에 없다
        return (self.version == protocol.VERSION and self.state == "connected" and not self.is_stale() and bool(self.last_status and self.last_status["robot_ready"])
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
        self.current_record = CommandRecord(command_id, target, raw, self._conn_id, requested_mono_ms=_now_ms())
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
        self._reset_run("연결 끊김")
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

    def _send(self, raw, kind, command_id=None, *, visible=True, note="", **fields):
        written = self._socket.write(raw) if self.state == "connected" else -1
        sent = written == len(raw)
        self._log("TX", kind, note or ("소켓 기록" if sent else "쓰기 실패 또는 일부 기록"), raw,
                  visible=visible, command_id=command_id, **fields,
                  send_state="queued" if sent else "partial" if written > 0 else "unsent", written=written,
                  delivery="qt_buffered" if sent else "unconfirmed")
        if not sent and self.state == "connected":
            self._drop("소켓 쓰기 실패 또는 일부 기록 · 실행 여부 미확인")
        return sent

    def _send_get_status(self):
        if self.state != "connected" or self._ready_pending:
            return
        if self._send(protocol.build_get_status(version=self.version), "준비 조회", visible=not self._suppressing):
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
        line = raw.removesuffix(b"\n")
        v12 = self.version == protocol.VERSION_12
        try:
            message = protocol.parse_v12(line) if v12 else protocol.parse_status(line)
        except protocol.ProtocolError as error:
            # 응답 전체를 버린다. 요청 결과·스냅샷·준비 상태 어느 것도 일부 반영하지 않는다(A.6.6)
            self._log("RX", "형식 오류", safe_text(error), raw)
            if v12:
                self._note_peer_mismatch(line)
            return
        if message["type"] == "RUN_STATUS":
            self._on_run_status(message, raw)
            return
        if message["type"] == "E_STOP_STATUS":
            self._on_e_stop_status(message, raw)
            return
        kind = protocol.status_kind(message)
        if v12:   # parse_v12 는 무ID 준비 STATUS 만 통과시킨다. 아래 1.1 준비 경로를 그대로 쓴다
            self._mark_verified()
            self.e_stop_available = message.get("e_stop_available") is True
            self._apply_snapshot(message["run"])
            self.run_changed.emit()
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
        # 1.2 후보는 slot_owner·run 변화도 표시 변화다. 1.1 STATUS 에는 두 필드가 없어 키가 예전과 같다
        return (message["robot_ready"], tuple(sorted(message["not_ready_reasons"])), message["active_command_id"],
                json.dumps(message.get("slot_owner"), sort_keys=True), json.dumps(message.get("run"), sort_keys=True),
                message.get("e_stop_available"))

    # ── TCP 1.2 후보 START·STOP·RECOVER (부록 A.6, 미승인) ─────────────────

    def sort_allowed(self, action):
        """(보낼 수 있는지, 사유). 버튼은 스냅샷 allowed 로만 켠다. 유효 스냅샷이 없을 때 연결 중 STOP 만 예외(A.6.12)."""
        if self.version != protocol.VERSION_12:
            return False, "TCP 1.2 후보 모드 아님(--tcp-version 1.2)"
        if self.state != "connected":
            return False, "연결 끊김"
        if not self.run_verified:
            return False, ("상대가 1.1 STATUS 로 답함 · 1.2 후보 미지원 서버" if self.peer_mismatch
                           else "1.2 후보 응답 확인 전")
        snapshot = None if self.is_stale() else self.run_snapshot
        if action == "STOP":
            if any(r.action == "STOP" and not r.final for r in self.sort_requests.values()):
                return False, "앞선 STOP 응답 대기 중 · 같은 바이트 재전송만 함"
            if snapshot is None:
                return True, "유효 스냅샷 없음 · 연결 중 STOP 예외"
        elif snapshot is None:
            return False, "유효 스냅샷 없음"
        elif self._unresolved():
            return False, "이전 요청 결과 미확인 · 새 스냅샷 대기"
        if action in snapshot["allowed"]:
            return True, "allowed 에 있음"
        return False, f"allowed {snapshot['allowed']} 에 없음"

    def sort_request(self, action, confirmed=False):
        """사용자가 누른 분류 명령 한 건을 새 request_id 로 보낸다. 상태는 바꾸지 않는다(스냅샷으로만)."""
        if not self.sort_allowed(action)[0] or (action == "RECOVER" and confirmed is not True):
            return ""
        snapshot = None if self.is_stale() else self.run_snapshot
        base = (snapshot["c_session_id"], snapshot["run_revision"]) if snapshot else None
        request_id = protocol.new_command_id()
        raw = protocol.build_run_request(action, request_id, *(base or (None, None)), user_confirmed=confirmed is True)
        record = SortRequest(request_id, action, raw, self._conn_id, base, attempts=1,
                             deadline_ms=_now_ms() + self.config["run_reply_timeout_ms"])
        self.sort_requests[request_id] = record
        self.last_request = record
        if self._send(raw, action, request_id=request_id, note="1.2 후보 요청 · 응답 대기"):
            record.result = f"{action} 송신 · 응답 대기"
        self._update_run_timer()
        self.run_changed.emit()
        return request_id

    def _unresolved(self):
        """결과가 정해지지 않은 요청이 있는가. UNKNOWN·PENDING 은 그 뒤 스냅샷이 새로 오면 풀린다."""
        for record in self.sort_requests.values():
            if not record.final:
                return True
            if record.outcome not in ("ACCEPTED", "REJECTED") and not self._advanced(record.base):
                return True
        return False

    def _advanced(self, base):
        snapshot = self.run_snapshot
        return snapshot is not None and (base is None or snapshot["c_session_id"] != base[0]
                                         or snapshot["run_revision"] > base[1])

    def _mark_verified(self):
        if not self.run_verified:
            self.run_verified = True
            self._log("EV", "1.2 후보 응답 확인", "검사를 통과한 1.2 응답 수신 · 이 연결에서 분류 조작 관문 열림")

    def _note_peer_mismatch(self, line):
        try:
            old = protocol.parse_status(line)
        except protocol.ProtocolError:
            return
        if not self.peer_mismatch:
            self.peer_mismatch = True
            self._log("EV", "1.2 후보 미지원 서버",
                      f"상대가 TCP {old['version']} STATUS(code={old.get('code')})로 답함 · 분류 명령을 보내지 않음")
            self.run_changed.emit()

    def _apply_snapshot(self, snapshot):
        """A.6.7 4: 같은 세션에서 순번이 클 때만 반영, 같으면 무시, 작으면 기록만. 세션이 바뀌면 교체."""
        if snapshot is None:
            if self.run_snapshot is not None:
                self.run_snapshot = None
                self._log("EV", "실행 상태 미확인", "서버가 유효 스냅샷 없음(null)으로 보고 · 마지막 상태로 추정하지 않음")
            return
        key = (snapshot["c_session_id"], snapshot["run_revision"])
        high = self._run_high
        if key[0] in self._retired_sessions:
            # A → B 로 바뀐 뒤 늦게 온 A 스냅샷. 세션 ID 에는 순서가 없으므로 이 연결에서 버린 세션은 다시 받지 않는다
            self._log("EV", "이전 세션 스냅샷", f"세션 {key[0][:8]} rev {key[1]} · 이미 교체된 세션 · 반영 안 함")
            return
        if high is not None and key[0] != high[0]:
            self._retired_sessions.add(high[0])
        if high is not None and key[0] == high[0] and (key[1] < high[1] or key[1] == high[1] and self.run_snapshot is not None):
            if key[1] < high[1]:
                self._log("EV", "이전 순번 스냅샷", f"rev {key[1]} < {high[1]} · 반영 안 함")
            return
        self.run_snapshot, self._run_high = snapshot, key
        self._log("EV", "실행 상태", safe_text(f"{snapshot['run_state']} · rev {key[1]} · 세션 {key[0][:8]} · "
                                              f"보유 {snapshot['hold_state']} · allowed {snapshot['allowed']}"),
                  run_state=snapshot["run_state"], run_revision=key[1], c_session_id=key[0])

    def _on_run_status(self, message, raw):
        request_id = message["request_id"]
        self._log("RX", "RUN_STATUS", safe_text(f"{(request_id or '알림')[:8]} {message['outcome']} {message['code']}"),
                  raw, request_id=request_id)
        self._mark_verified()
        record = self.sort_requests.get(request_id) if request_id else None
        if request_id and record is None:
            self._log("EV", "대응 요청 없음", "이전 연결이나 다른 클라이언트의 요청 · 요청 결과에 반영 안 함", request_id=request_id)
        elif record is not None:
            self._apply_outcome(record, message)
        self._apply_snapshot(message["snapshot"])
        self._update_run_timer()
        self.run_changed.emit()

    def _apply_outcome(self, record, message):
        outcome, code = message["outcome"], message["code"]
        if outcome == "PENDING":   # 서버 기록만 있음. 기한·재전송은 그대로
            if record.outcome is None and not record.final:
                record.outcome, record.result = "PENDING", f"{record.action} 서버 기록됨 · 결과 대기(PENDING)"
            return
        if record.outcome in ("ACCEPTED", "REJECTED", "UNKNOWN"):   # 처음 확정한 값은 바뀌지 않는다
            if (record.outcome, record.code) != (outcome, code):
                self._log("EV", "요청 결과 불일치", f"기존 {record.outcome}/{record.code} 유지 · 받은 {outcome}/{code}",
                          request_id=record.request_id)
            return
        late = record.final
        record.outcome, record.code, record.detail, record.final = outcome, code, message["detail"], True
        text = f"{record.action} 요청 결과 {outcome} · {code}"
        if record.action == "STOP" and outcome == "ACCEPTED":
            text += " · 정지 확인 아님(상태로 확인)"
        elif outcome == "UNKNOWN":
            text += " · 실행되지 않았다는 뜻 아님"
        record.result = text + (" · 늦은 응답" if late else "")
        self._log("EV", "요청 결과", record.result, request_id=record.request_id)

    def _close_unknown(self, record, reason):
        record.final = True
        record.result = f"{record.action} {reason}"
        self._log("EV", "요청 결과 미확인", record.result, request_id=record.request_id, attempts=record.attempts)

    # ── E-Stop 응용 요청(D 제안 · C 미합의). 분류 스냅샷·allowed 와 별개 ───────────

    def e_stop_allowed(self):
        """(보낼 수 있는지, 사유). 지금 연결의 최신 유효 준비 STATUS 가 e_stop_available=true 일 때만. 분류 상태·robot_ready 는 보지 않는다."""
        if self.version != protocol.VERSION_12:
            return False, "TCP 1.2 후보 모드 아님"
        if self.state != "connected":
            return False, "연결 끊김"
        if self.is_stale() or not self.e_stop_available:
            return False, "C 가 E-Stop 요청 가능을 알리지 않음(e_stop_available)"
        if self.e_stop is not None and not self.e_stop.final:
            return False, "앞선 E-Stop 요청 결과 대기 중"
        return True, "C 에 E-Stop 요청 가능 · 실제 동작은 C 가 정함"

    def e_stop_request(self):
        """사용자가 누를 때 한 줄만 보낸다. 재전송·재접속 재송신 없음."""
        if not self.e_stop_allowed()[0]:
            return ""
        request_id = protocol.new_command_id()
        raw = protocol.build_e_stop(request_id)
        self.e_stop = EStopRequest(request_id, raw, self._conn_id, _now_ms() + self.config["e_stop_timeout_ms"])
        if self._send(raw, "E_STOP", request_id=request_id, note="E-Stop 응용 요청 · C 결과 대기"):
            self.e_stop.result = "E-Stop 요청 송신 · C 응답 대기 · 정지 확인 아님"
        self._update_run_timer()
        self.run_changed.emit()
        return request_id

    def _on_e_stop_status(self, message, raw):
        request_id, state = message["request_id"], message["state"]
        self._log("RX", "E_STOP_STATUS", f"{request_id[:8]} {state}", raw, request_id=request_id)
        record = self.e_stop
        if record is None or record.request_id != request_id or record.conn_id != self._conn_id:
            self._log("EV", "E-Stop 대응 요청 없음", "다른 ID 또는 이전 연결의 응답 · 반영 안 함", request_id=request_id)
            return
        if record.final and state == "REQUESTED":   # 기한 뒤 늦은 접수 통지로 미확인 결과를 가리지 않는다
            self._log("EV", "E-Stop 늦은 접수 무시", "이미 미확인으로 닫힌 요청의 REQUESTED · 반영 안 함", request_id=request_id)
            return
        terminal = record.state in ("CONFIRMED", "REJECTED", "UNCONFIRMED")
        if terminal or (state == "REQUESTED" and record.state == "REQUESTED"):
            if state != record.state:   # C 최종 보고는 되돌리지 않는다
                self._log("EV", "E-Stop 결과 불일치", f"기존 {record.state} 유지 · 받은 {state}", request_id=request_id)
            return
        late = record.final
        record.state, record.detail = state, message["detail"]
        record.result = {"REQUESTED": "E-Stop C 접수(REQUESTED) · 정지 확인 아님",
                         "CONFIRMED": "E-Stop C 확인 보고(CONFIRMED) · C 가 로봇·컨베이어 정지를 보고(D 제안 의미, D 검증 아님)",
                         "REJECTED": "E-Stop C 거부(REJECTED)",
                         "UNCONFIRMED": "E-Stop C 미확인 보고(UNCONFIRMED) · 정지 확인 아님"}[state]
        record.final = record.final or state != "REQUESTED"
        record.result += " · 늦은 응답" if late else ""
        self._log("EV", "E-Stop 결과", record.result, request_id=request_id)
        self._update_run_timer()
        self.run_changed.emit()

    def _update_run_timer(self):
        if any(not r.final for r in self.sort_requests.values()) or (self.e_stop is not None and not self.e_stop.final):
            if not self._run_timer.isActive():
                self._run_timer.start()
        else:
            self._run_timer.stop()

    def _run_tick(self):
        now = _now_ms()
        for record in list(self.sort_requests.values()):
            if record.final or now < record.deadline_ms or record.request_id not in self.sort_requests:
                continue
            if (self.state == "connected" and record.conn_id == self._conn_id
                    and record.attempts < 1 + self.config["run_max_resend"]):
                record.attempts += 1   # 같은 request_id·같은 바이트. 서버가 중복으로 판정한다(A.6.10)
                record.deadline_ms = now + self.config["run_reply_timeout_ms"]
                self._send(record.raw, record.action, note="동일 바이트 재전송", request_id=record.request_id)
            else:
                self._close_unknown(record, "응답 시간 초과 · 결과 미확인 · 재시도 끝")
            self.run_changed.emit()
        record = self.e_stop
        if record is not None and not record.final and now >= record.deadline_ms:
            self._close_e_stop(record, "C 최종 보고 없음 · 시간 초과 · 결과 미확인")
            self.run_changed.emit()
        self._update_run_timer()

    def _close_e_stop(self, record, reason):
        record.final = True
        record.result = f"E-Stop {reason}" + (" (마지막 C 보고 REQUESTED: 접수만)" if record.state == "REQUESTED" else "")
        self._log("EV", "E-Stop 결과 미확인", record.result, request_id=record.request_id)

    def _reset_run(self, reason):
        """끊김·종료·주소 변경: 조작 관문을 닫고 스냅샷을 버린다. 대기 요청은 미확인으로 닫고 다시 보내지 않는다."""
        for record in list(self.sort_requests.values()):
            if not record.final:
                self._close_unknown(record, f"{reason} · 결과 미확인 · 자동 재전송 없음")
        self.sort_requests.clear()
        if self.e_stop is not None:
            if not self.e_stop.final:
                self._close_e_stop(self.e_stop, f"{reason} · 결과 미확인 · 재전송 없음")
            if not self.e_stop.result.startswith(HISTORY):
                # 방금 닫은 것도, 이미 끝난 것도 이전 연결 기록이다. 새 연결·주소에서 현재 상태처럼 보이지 않게 한다
                self.e_stop.result = HISTORY + self.e_stop.result
        self.e_stop_available = False
        self._run_timer.stop()
        self.run_snapshot = self._run_high = None
        self._retired_sessions = set()
        self.run_verified = self.peer_mismatch = False
        if self.version == protocol.VERSION_12:
            self.run_changed.emit()

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
            self._note_ack_metric(record)
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

    def _note_ack_metric(self, record):
        """수락이 처음 정해질 때 한 번: 원래 연결·첫 송신의 문자 그대로 ACK 만 숫자로 남긴다(로컬 기록 전용)."""
        if record.ack_latency_ms is not None:
            return
        if record.acceptance == "first_ack" and record.attempts == 1 and record.requested_mono_ms is not None:
            record.ack_latency_ms = max(0, _now_ms() - record.requested_mono_ms)
            self._log("EV", "실행→ACK 지연", f"{record.ack_latency_ms} ms", command_id=record.command_id,
                      ack_latency_ms=record.ack_latency_ms, metric_boundary=ACK_METRIC_BOUNDARY)
        else:
            self._log("EV", "실행→ACK 지연", record.ack_metric()[1], command_id=record.command_id,
                      ack_latency_ms=None, acceptance=record.acceptance, metric_boundary=ACK_METRIC_BOUNDARY)

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
