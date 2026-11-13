"""D TCP 클라이언트(2일차 골격): C 서버 준비 상태를 주기 조회하는 지속 연결.

┌─ 파트별 경계 ──────────────────────────────────────────────────────────────┐
│ [C 연동] 상대는 C의 robot_server_package (WSL, 기본 127.0.0.1:47090).        │
│          이 파일은 C 코드를 읽거나 바꾸지 않는다. TCP 1.1 명세만 따른다.     │
│ [공용]   신호 작성·STATUS 검사는 eyetoarm.tcp.protocol 을 그대로 호출한다.   │
│          (build_get_status · parse_status · status_kind). 수정하지 않는다.   │
│ [공용]   주소·주기 값은 config/gui/client.json (C·D 공유)을 읽기만 한다.     │
│ [A 참고] A의 통신 시험 도구가 재사용할 클라이언트 API는 3일차 초안, 6일차    │
│          고정 예정이다(D 9일 계획). 이 2일차 클래스는 아직 그 API가 아니다.  │
└────────────────────────────────────────────────────────────────────────────┘

[2일차에 보내는 것] command_id 없는 GET_STATUS(준비 조회, 명세 §4.4) 한 종류뿐이다.
[2일차에 보내지 않는 것]
  - MOVE · CANCEL · ID 있는 GET_STATUS: 3일차 구현(ACK·재전송·타임아웃 포함).
  - START · STOP · RECOVER: 명세 부록 A 개정 제안이 미승인이다. C 승인 전에는 실제 서버로 보내지 않는다.
  로봇을 움직이는 신호는 이 파일에 없다.

[동작] Qt 이벤트 루프 위에서 QTcpSocket 으로 비동기 처리한다. GUI 스레드를 막지 않는다.
  접속 → readiness_poll_ms 마다 준비 조회 → 한 줄(LF)씩 STATUS 검사 → 신호로 화면에 전달.
  끊기면 reconnect_ms 뒤 다시 접속한다. 재접속 뒤에는 조회만 한다.
[표시 규칙] 마지막 유효 STATUS 가 status_watchdog_ms 보다 오래되면 화면은 그 값을 현재값처럼
  보여주지 않는다(is_stale). 이전 준비 상태를 살려 두지 않는다.
"""

from __future__ import annotations

import ipaddress
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtNetwork import QAbstractSocket, QHostAddress, QTcpSocket

from eyetoarm.d_client.diag import safe_text
from eyetoarm.tcp import protocol

CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "gui" / "client.json"

# client.json 을 읽지 못할 때 쓰는 값. TCP 명세 §9 초기값과 같다
DEFAULTS = {"host": "127.0.0.1", "port": 47090, "max_line_bytes": 4096, "readiness_poll_ms": 1000,
            "status_watchdog_ms": 2000, "connect_timeout_ms": 2000, "reconnect_ms": 1500}


def load_tcp_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """[공용 설정 읽기] client.json 의 tcp 항목에서 D가 쓰는 키만 꺼낸다. 파일은 바꾸지 않는다.

    서버 전용 키(record_retention · command_log_path 등)는 C 소유이므로 읽지 않는다.
    """
    config = dict(DEFAULTS)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8")).get("tcp", {})
    except (OSError, ValueError):
        return config
    for key, default in DEFAULTS.items():
        if type(loaded.get(key)) is type(default):
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
    """송수신 기록 한 건(GUI-03). 화면 표시용이며 파일 저장(CSV)은 3일차에 추가한다."""
    wall_time: str      # 이 PC 벽시계 HH:MM:SS.mmm (서버 시각과 다른 시계)
    direction: str      # TX(보냄) · RX(받음) · EV(연결 사건)
    kind: str           # 준비 조회 · STATUS · 접속 · 끊김 등
    summary: str        # 사람이 읽는 한 줄
    raw: str = ""       # 실제 보낸/받은 줄(제어 문자 이스케이프, 길이 제한)


class StatusLink(QObject):
    """C 서버와의 명령·상태 TCP 연결 하나. 앱 전체에서 하나만 만들고 모든 탭이 공유한다."""

    state_changed = Signal(str)        # "disconnected" · "connecting" · "connected"
    status_received = Signal(dict)     # 검사를 통과한 준비 응답(robot_ready · not_ready_reasons · ...)
    traffic = Signal(object)           # TrafficEntry

    def __init__(self, host: str | None = None, port: int | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.config = load_tcp_config()
        self.host = host or self.config["host"]
        self.port = port or self.config["port"]
        ipaddress.ip_address(self.host)  # 숫자 IP 만. localhost 는 Windows 에서 약 2초 늦다(README 실측)
        self.state = "disconnected"
        self.last_status: dict[str, Any] | None = None
        self._last_status_at: float | None = None   # time.monotonic()
        self._buffer = bytearray()

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

    # ── 바깥에서 쓰는 것 ────────────────────────────────────────────────

    def start(self) -> None:
        self._open()

    def stop(self) -> None:
        """창을 닫을 때 부른다. 타이머와 소켓을 정리한다."""
        for timer in (self._poll, self._retry, self._connect_deadline):
            timer.stop()
        self._set_state("disconnected")   # 먼저 바꿔야 abort() 뒤 재접속이 예약되지 않는다
        self._socket.abort()

    def status_age_ms(self) -> int | None:
        """마지막 유효 STATUS 를 받은 뒤 지난 시간(이 PC 단조 시계). 받은 적 없으면 None."""
        if self._last_status_at is None:
            return None
        return int((time.monotonic() - self._last_status_at) * 1000)

    def is_stale(self) -> bool:
        """유효 STATUS 가 없거나 status_watchdog_ms 보다 오래됐다. 참이면 준비 상태를 표시하지 않는다."""
        age = self.status_age_ms()
        return self.state != "connected" or age is None or age > self.config["status_watchdog_ms"]

    # ── 연결 수명 ───────────────────────────────────────────────────────

    def _open(self) -> None:
        if self._socket.state() != QAbstractSocket.SocketState.UnconnectedState:
            return
        self._buffer.clear()
        self._set_state("connecting")
        self._connect_deadline.start()
        self._socket.connectToHost(QHostAddress(self.host), self.port)

    def _on_connected(self) -> None:
        self._connect_deadline.stop()
        self._set_state("connected")
        self._log("EV", "접속", f"{self.host}:{self.port} 연결됨")
        self._send_get_status()   # 재접속 직후에는 조회만 한다
        self._poll.start()

    def _on_disconnected(self) -> None:
        self._drop("서버가 연결을 닫음")

    def _on_error(self, _error) -> None:
        self._drop(safe_text(self._socket.errorString()))

    def _on_connect_timeout(self) -> None:
        self._drop(f"접속 기한 {self.config['connect_timeout_ms']} ms 초과")

    def _drop(self, reason: str) -> None:
        """연결을 버리고 재접속을 예약한다. 받아 둔 STATUS 는 더 이상 현재값이 아니다."""
        if self.state == "disconnected":   # abort() 가 disconnected 신호를 다시 내도 한 번만 처리한다
            return
        self._poll.stop()
        self._connect_deadline.stop()
        self._last_status_at = None
        self.last_status = None
        self._set_state("disconnected")
        self._socket.abort()
        self._log("EV", "끊김", reason)
        self._retry.start()

    def _set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.state_changed.emit(state)

    # ── 송신·수신 ───────────────────────────────────────────────────────

    def _send_get_status(self) -> None:
        if self.state != "connected":
            return
        line = protocol.build_get_status()   # [공용] 무ID GET_STATUS 한 줄
        self._socket.write(line)
        self._log("TX", "준비 조회", "GET_STATUS (command_id 없음)", line)

    def _on_ready_read(self) -> None:
        self._buffer += bytes(self._socket.readAll())
        try:
            lines = split_lines(self._buffer, self.config["max_line_bytes"])
        except OverflowError as error:
            self._drop(f"수신 한도 초과: {error}")
            return
        for line in lines:
            self._handle_line(line)

    def _handle_line(self, line: bytes) -> None:
        try:
            message = protocol.parse_status(line)   # [공용] 스키마·정수 규칙 검사
        except protocol.ProtocolError as error:
            self._log("RX", "형식 오류", safe_text(error), line)
            return   # 잘못된 줄은 상태에 반영하지 않는다
        kind = protocol.status_kind(message)
        if kind != "ready" or "command_id" in message:
            # 2일차는 무ID 조회만 보내므로 명령 상태·거부 응답은 기대하지 않는다. 기록만 남긴다
            self._log("RX", "STATUS(예상 밖)", safe_text(f"kind={kind} code={message.get('code')}"), line)
            return
        self.last_status = message
        self._last_status_at = time.monotonic()
        reasons = ",".join(message["not_ready_reasons"]) or "-"
        self._log("RX", "STATUS", safe_text(f"robot_ready={str(message['robot_ready']).lower()} 사유={reasons}"), line)
        self.status_received.emit(message)

    def _log(self, direction: str, kind: str, summary: str, raw: bytes = b"") -> None:
        now = time.time()
        wall = time.strftime("%H:%M:%S", time.localtime(now)) + f".{int(now * 1000) % 1000:03d}"
        text = safe_text(raw.decode("utf-8", errors="replace").rstrip("\n"), 400) if raw else ""
        self.traffic.emit(TrafficEntry(wall, direction, kind, summary, text))
