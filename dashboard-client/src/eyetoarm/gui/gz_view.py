"""Gazebo 화면 수신(Windows 쪽): gz_view_relay.py 가 보내는 JPEG 프레임을 받아 QImage 로 넘긴다.

┌─ 파트별 경계 ──────────────────────────────────────────────────────────────┐
│ [D 임시 경로] 대시보드 메인 디스플레이에 Gazebo 로봇 구동 모습을 띄우기 위한 │
│   확인용 연결이다. 정식 "C 표시 연결"(TCP 명세 부록 B, 미승인)이 아니다.     │
│ [C 연동 예정] C 표시 연결의 방식·포트·직렬화가 정해지면 이 클래스 자리에     │
│   정식 수신기를 넣는다. 화면(DisplayView)은 frame_received(QImage) 신호만    │
│   받으므로 화면 코드는 그대로 둘 수 있다.                                    │
│ [A·B 연동 예정] 검출 주석·중심점 주석 영상과 결과·무효화 JSON 은 이 경로에   │
│   없다. 여기로 오는 것은 주석 없는 Gazebo 카메라 영상 하나뿐이다.            │
└────────────────────────────────────────────────────────────────────────────┘

명령·상태 TCP(client.py, 47090)와 소켓을 공유하지 않는다.
"""

from __future__ import annotations

import ipaddress
import time

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QImage
from PySide6.QtNetwork import QAbstractSocket, QHostAddress, QTcpSocket

from eyetoarm.gui.gz_view_relay import DEFAULT_PORT, FrameParser

RECONNECT_MS = 1500
STALE_MS = 3000   # 카메라가 2 Hz(500 ms 간격)이므로 6장 연속 누락이면 중단으로 본다


class GzViewReceiver(QObject):
    """중계 스크립트에 접속해 최신 프레임을 받는다. 앱에서 하나만 만든다."""

    state_changed = Signal(str)       # "disconnected" · "connecting" · "connected"
    frame_received = Signal(QImage)   # 해독에 성공한 프레임. 보간·합성 없이 받은 그대로

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, parent: QObject | None = None) -> None:
        super().__init__(parent)
        ipaddress.ip_address(host)
        self.host, self.port = host, port
        # 창(window.py)이 화면에 쓰는 이름표. GzWindowCapture · DemoView 도 같은 이름을 가진다
        self.title = f"카메라 중계 {host}:{port}"
        self.source_name = "appearance_check_cam"
        self.help_text = f"Gazebo 카메라 중계({host}:{port})에 연결되지 않았습니다. WSL 에서 브리지와 gz_view_relay.py 를 실행하세요."
        self.stale_label = "프레임 중단"
        self.state = "disconnected"
        self.frame_count = 0
        self.decode_errors = 0
        self._last_frame_at: float | None = None
        self._parser = FrameParser()

        self._socket = QTcpSocket(self)
        self._socket.connected.connect(lambda: self._set_state("connected"))
        self._socket.disconnected.connect(self._drop)
        self._socket.errorOccurred.connect(lambda _error: self._drop())
        self._socket.readyRead.connect(self._on_ready_read)
        self._retry = QTimer(self, interval=RECONNECT_MS, singleShot=True)
        self._retry.timeout.connect(self._open)

    def start(self) -> None:
        self._open()

    def stop(self) -> None:
        self._retry.stop()
        self._set_state("disconnected")   # 먼저 바꿔야 abort() 뒤 재접속이 예약되지 않는다
        self._socket.abort()

    def frame_age_ms(self) -> int | None:
        if self._last_frame_at is None:
            return None
        return int((time.monotonic() - self._last_frame_at) * 1000)

    def is_stale(self) -> bool:
        """프레임을 받은 적 없거나 STALE_MS 보다 오래됐다. 참이면 화면은 그 프레임을 현재 모습처럼 보여주지 않는다."""
        age = self.frame_age_ms()
        return age is None or age > STALE_MS

    def _open(self) -> None:
        if self._socket.state() != QAbstractSocket.SocketState.UnconnectedState:
            return
        self._parser = FrameParser()
        self._set_state("connecting")
        self._socket.connectToHost(QHostAddress(self.host), self.port)

    def _drop(self) -> None:
        if self.state == "disconnected":   # abort() 가 신호를 다시 내도 한 번만 처리한다
            return
        self._set_state("disconnected")
        self._socket.abort()
        if not self._retry.isActive():
            self._retry.start()

    def _set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.state_changed.emit(state)

    def _on_ready_read(self) -> None:
        try:
            frames = self._parser.feed(bytes(self._socket.readAll()))
        except ValueError:
            self._drop()   # 형식이 틀리면 경계를 다시 맞출 수 없다. 끊고 다시 접속한다
            return
        if not frames:
            return
        image = QImage.fromData(frames[-1], "JPG")   # 한 번에 여러 장이 왔으면 최신 한 장만 쓴다
        if image.isNull():
            self.decode_errors += 1
            return
        self.frame_count += 1
        self._last_frame_at = time.monotonic()
        self.frame_received.emit(image)
