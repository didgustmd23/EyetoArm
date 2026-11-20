"""C 3채널 원본 영상 수신(Windows 쪽, 시제품): C display_relay 의 ETV1 스트림을 받아 DisplayModel 로 넘긴다.

┌─ 경계 ────────────────────────────────────────────────────────────────────┐
│ [C 시제품] robot_server_package/display_relay.py (기본 포트 47092) 의 ETV1   │
│   패킷만 받는다. 승인된 C↔D 계약이 아니며 C README 도 "pending agreement".  │
│   명령·상태 TCP(47090)·D 임시 단일 영상(47091 EAV1)과 별개다.               │
│ [보내는 것] 사용자가 링크 raw/detect/center 를 누를 때만 이 연결로 한 줄   │
│   {"type":"SET_DISPLAY_VARIANT","channel":"link","variant":...} + LF 를    │
│   보낸다(임시 형식, C 와 합의 전). 현재 C relay 는 받은 쪽을 읽지 않으므로  │
│   송신은 적용 확인이 아니다. 해당 종류 영상이 와야 화면에 보인다.          │
│ [결과 없음] C 시제품은 영상만 보낸다. 결과·무효화 사건을 지어내지 않는다.   │
└────────────────────────────────────────────────────────────────────────────┘

패킷: b"ETV1" + 본문 길이(>I, 8 MiB 이하) + 본문[메타 길이(>I) + 메타 JSON + JPEG].
메타(C 현재): channel·variant·source_id·image_seq·generation·calibration_id·run_id·capture_generation·
detector_config_id·job_seq(+ stamp_sec·stamp_nanosec). DisplayModel 이 받는 키만 엄격히 검사해 넘긴다.
C 의 run_id 는 relay 프로세스마다 하나로 고정이다. 새 run_id 를 처음 보면 새 세션을 시작하고, 이미 지나간
run_id 는 같은 연결 안에서도 다시 받지 않는다(세션이 뒤로 가지 않게).
C 는 새 접속에 채널별 마지막 패킷(캐시)을 나이와 상관없이 바로 보낸다. 그래서 연결·run 마다 채널·종류별 첫 패킷은
「확인 대기」(confirmed=False)로 넘기고, 같은 run 에서 순번이 더 큰 패킷을 정상 해독한 뒤에야 실시간으로 본다.
"""
from __future__ import annotations

import ipaddress
import json
import struct
import time
from collections import deque

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QObject, QTimer, Signal
from PySide6.QtGui import QImage, QImageReader
from PySide6.QtNetwork import QAbstractSocket, QHostAddress, QTcpSocket

MAGIC = b"ETV1"
HEADER = struct.Struct(">4sI")
META_LENGTH = struct.Struct(">I")
MAX_PACKET = 8 * 1024 * 1024          # C display_relay.MAX_PACKET 과 같다
MAX_META = 16 * 1024                  # 메타 JSON 상한. C 현재 메타는 300바이트 안팎
MAX_PIXELS = 4096 * 4096              # 해독 전에 그림 크기를 확인해 큰 할당을 막는다
DEFAULT_PORT = 47092
CHANNELS = {"link": "gz_link", "gripper": "gz_gripper", "fixed": "gz_fixed"}
LINK_SOURCE = CHANNELS["link"]
VARIANTS = ("raw", "detect", "center")
INT_KEYS = ("image_seq", "generation", "capture_generation", "job_seq")
RECONNECT_MS = 1500
CONNECT_TIMEOUT_MS = 2000              # 명령 연결 connect_timeout_ms 초기값과 같다. 응답 없는 주소에 묶이지 않게
READ_CHUNK = 1024 * 1024               # 한 번에 읽어 파서에 넣는 최대 바이트
SOCKET_BUFFER = 2 * (MAX_PACKET + 8)  # Qt 소켓 내부 버퍼 상한. 차면 TCP 흐름 제어로 C 송신이 기다린다
RETIRED_RUNS = 32
VARIANT_REQUEST = "SET_DISPLAY_VARIANT"


class PacketError(ValueError):
    """프레임 경계는 맞지만 이 패킷 하나가 잘못됐다(건너뛰고 다음 패킷을 읽는다)."""


def _reject_constant(name):
    raise PacketError(f"JSON 상수 {name} 불허")


def _text(value, limit=128):
    return isinstance(value, str) and 0 < len(value) <= limit


def parse_metadata(raw: bytes) -> tuple[str, dict]:
    """메타 JSON 을 검사해 (channel, DisplayModel.ingest_frame 키워드)를 돌려준다. 그 외 키는 넘기지 않는다."""
    try:
        meta = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError, RecursionError) as error:   # 깊은 중첩도 패킷 오류로 처리
        raise PacketError(f"메타 JSON 오류: {type(error).__name__}") from None
    if not isinstance(meta, dict):
        raise PacketError("메타가 객체가 아님")
    channel, variant = meta.get("channel"), meta.get("variant")
    if not isinstance(channel, str) or channel not in CHANNELS:   # 리스트·객체는 해시 오류 전에 거른다
        raise PacketError(f"알 수 없는 channel {channel!r:.40}")
    if not isinstance(variant, str) or variant not in VARIANTS or (channel != "link" and variant != "raw"):
        raise PacketError(f"{channel} 의 variant {variant!r} 불허")
    if meta.get("source_id") != CHANNELS[channel]:
        raise PacketError(f"{channel} 의 source_id {meta.get('source_id')!r} 불일치")
    for key in INT_KEYS:
        value = meta.get(key)
        if type(value) is not int or not 0 <= value < 2 ** 53:   # bool 은 int 의 하위형이라 type 으로 거른다
            raise PacketError(f"{key} 가 0 이상 정수가 아님")
    if not _text(meta.get("run_id")) or not _text(meta.get("detector_config_id")):
        raise PacketError("run_id 또는 detector_config_id 오류")
    calibration = meta.get("calibration_id")
    if calibration is not None and not _text(calibration):
        raise PacketError("calibration_id 오류")
    if "calibration_id" not in meta:
        raise PacketError("calibration_id 키 없음")
    envelope = {key: meta[key] for key in ("source_id", *INT_KEYS, "run_id", "detector_config_id")}
    envelope.update(calibration_id=calibration, variant=variant)
    return channel, envelope


class EtvParser:
    """바이트 흐름에서 ETV1 패킷을 꺼낸다. 남는 버퍼는 미완성 한 패킷(8 MiB + 머리)과 마지막 feed 조각뿐이다.

    수신기는 READ_CHUNK 단위로만 feed 하므로 버퍼는 MAX_PACKET + HEADER + READ_CHUNK 를 넘지 않는다.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.packet_errors = 0
        self.last_error = ""

    def reset(self) -> None:
        """새 연결: 미완성 바이트만 버린다. 오류 수는 진단용으로 유지한다."""
        self._buffer.clear()

    def feed(self, data: bytes) -> list[tuple[str, dict, bytes]]:
        """완성된 패킷의 (channel, envelope, jpeg) 목록. 경계를 잃는 오류는 ValueError(연결을 끊어야 함)."""
        self._buffer.extend(data)
        packets = []
        while len(self._buffer) >= HEADER.size:
            magic, size = HEADER.unpack_from(self._buffer)
            if magic != MAGIC:
                raise ValueError(f"ETV1 아님: {bytes(magic)!r}")
            if not META_LENGTH.size < size <= MAX_PACKET:
                raise ValueError(f"본문 길이 {size} 허용 범위 밖")
            if len(self._buffer) < HEADER.size + size:
                break
            body = bytes(self._buffer[HEADER.size:HEADER.size + size])
            del self._buffer[:HEADER.size + size]
            try:
                (meta_length,) = META_LENGTH.unpack_from(body)
                if not 0 < meta_length <= MAX_META or META_LENGTH.size + meta_length >= len(body):
                    raise PacketError(f"메타 길이 {meta_length} 오류 또는 JPEG 없음")
                channel, envelope = parse_metadata(body[META_LENGTH.size:META_LENGTH.size + meta_length])
            except PacketError as error:
                self.packet_errors += 1
                self.last_error = str(error)[:80]
                continue
            packets.append((channel, envelope, body[META_LENGTH.size + meta_length:]))
        return packets


def decode_jpeg(data: bytes) -> QImage | None:
    """크기를 먼저 읽고 MAX_PIXELS 이하일 때만 해독한다. 실패하면 None."""
    array = QByteArray(data)
    buffer = QBuffer(array)
    if not buffer.open(QIODevice.OpenModeFlag.ReadOnly):
        return None
    try:
        reader = QImageReader(buffer, b"jpeg")
        size = reader.size()
        if not size.isValid() or size.width() <= 0 or size.height() <= 0 or size.width() * size.height() > MAX_PIXELS:
            return None
        image = reader.read()
        return None if image.isNull() else image
    finally:
        buffer.close()


class RelayViewReceiver(QObject):
    """C display_relay(47092)에 접속해 채널·종류별 최신 프레임만 DisplayModel 로 넘긴다. 앱에서 하나만 만든다."""

    is_relay = True   # 창이 C 3채널 중계 모드로 그리게 하는 표식(DemoView 등 다른 수신기와 구분)

    state_changed = Signal(str)              # "disconnected" · "connecting" · "connected"(TCP 만. 영상 수신과 별개)
    frame_received = Signal(QImage)          # 창 공통 인터페이스. 이 수신기는 내지 않는다(3채널은 display_frame)
    display_session = Signal(str, str)       # (run_id, 링크 source_id) 새 C relay 세션
    display_frame = Signal(str, object, object)   # (channel, QImage, ingest_frame 키워드)

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, parent: QObject | None = None) -> None:
        super().__init__(parent)
        ipaddress.ip_address(host)   # 숫자 IP 만. 이름 해석으로 다른 곳에 붙지 않는다
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("invalid relay port")
        self._set_address(host, port)
        self.source_name = "C display_relay"
        self.stale_label = "영상 노후"
        self.state = "disconnected"
        self.frame_count = 0
        self.decode_errors = 0
        self.stale_session_drops = 0
        self.framing_errors = 0
        self.socket_error = ""   # 마지막 연결 오류(Qt errorString). 오류 수에는 넣지 않는다
        self._last_error = ""
        self.run_id: str | None = None
        self.requested_variant: str | None = None   # 이 연결에서 마지막으로 보낸 링크 요청(송신만, C 적용 미확인)
        self._retired: deque[str] = deque(maxlen=RETIRED_RUNS)
        self._last_frame_at: float | None = None
        self._parser = EtvParser()
        self._pending: dict[tuple[str, str], tuple[str, dict, bytes]] = {}
        self._newest: dict[tuple[str, str], tuple[int, int]] = {}       # 이 연결·run 의 칸별 최고 (capture_generation, image_seq)
        self._confirmed: set[tuple[str, str]] = set()                   # 첫 패킷 뒤 더 새 패킷을 받은 칸

        self._socket = self._new_socket()
        self._retry = QTimer(self, interval=RECONNECT_MS, singleShot=True)
        self._retry.timeout.connect(self._open)
        self._flush_timer = QTimer(self, interval=0, singleShot=True)   # 한 번의 Qt 차례에 채널별 최신만 해독
        self._flush_timer.timeout.connect(self._flush)
        self._connect_deadline = QTimer(self, interval=CONNECT_TIMEOUT_MS, singleShot=True)
        self._connect_deadline.timeout.connect(self._on_connect_timeout)

    @property
    def packet_errors(self) -> int:
        return self._parser.packet_errors

    @property
    def last_error(self) -> str:
        """가장 최근 수신 오류 한 줄(80자 이내). 영상이 비는 이유가 카메라 부재인지 손상인지 구분하게 한다."""
        return (self._parser.last_error or self._last_error)[:80]

    def error_summary(self) -> str:
        counts = (("경계", self.framing_errors), ("메타/패킷", self.packet_errors), ("JPEG", self.decode_errors),
                  ("지난 세션", self.stale_session_drops))
        parts = [f"{name} {count}" for name, count in counts if count]
        return f"수신 오류 {', '.join(parts)} · 최근: {self.last_error}" if parts else ""

    def _set_address(self, host: str, port: int) -> None:
        self.host, self.port = host, port
        self.title = f"C 3채널 중계(시제품 ETV1) {host}:{port}"
        self.help_text = (f"C 영상 중계({host}:{port})에 연결되지 않았습니다. WSL 에서 "
                          "ros2 run robot_server_package display_relay 를 실행했는지 확인하세요.")

    def _new_socket(self) -> QTcpSocket:
        sock = QTcpSocket(self)
        sock.setReadBufferSize(SOCKET_BUFFER)
        sock.connected.connect(self._on_connected)
        sock.disconnected.connect(self._drop)
        sock.errorOccurred.connect(self._on_socket_error)
        sock.readyRead.connect(self._on_ready_read)
        return sock

    def start(self) -> None:
        self._open()

    def set_endpoint(self, host: str, port: int) -> bool:
        """명시적 적용. 끊고(표시 무효화) 다시 접속한다. 주소가 바뀌었으면 True.

        주소가 바뀌면 소켓을 새로 만들어 옛 연결의 늦은 신호를 막고, 이전 서버의 run_id 기록도 버린다
        (새 서버가 같은 run_id·낮은 순번으로 시작해도 막히지 않게). 호출자는 이때 DisplayModel.reset_session 을 부른다.
        같은 주소면 소켓과 run_id 기록을 그대로 두므로 기존 재접속 보호(지난 run_id 무시)가 유지된다."""
        ipaddress.ip_address(host)
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("invalid relay port")
        self.stop()
        changed = (host, port) != (self.host, self.port)
        if changed:
            old, self._socket = self._socket, self._new_socket()
            old.blockSignals(True)
            old.abort()
            old.deleteLater()
            self._set_address(host, port)
            self.socket_error = ""
            self._last_frame_at = None
            self.run_id = None
            self._retired.clear()
        self._open()
        return changed

    def stop(self) -> None:
        self._retry.stop()
        self._connect_deadline.stop()
        self._flush_timer.stop()
        self._pending.clear()
        self._set_state("disconnected")   # 먼저 바꿔야 abort() 뒤 재접속이 예약되지 않는다
        self._socket.abort()

    def request_variant(self, variant: str) -> bool:
        """연결된 이 영상 소켓으로 링크 표시 종류 요청 한 줄을 보낸다. 연결 안 됨·쓰기 실패면 False(재시도 없음).

        shortcut: C 와 합의된 요청 형식이 없어 UTF-8 compact JSON + LF 를 임시로 쓴다. 합의되면 형식만 바꾼다."""
        if variant not in VARIANTS:
            raise ValueError(f"variant {variant!r} 불허")
        self.requested_variant = None   # 실패하면 이전 요청을 「송신됨」으로 남기지 않는다
        if self.state != "connected":
            return False
        line = json.dumps({"type": VARIANT_REQUEST, "channel": "link", "variant": variant},
                          separators=(",", ":")).encode("utf-8") + b"\n"
        if self._socket.write(line) != len(line):
            # Qt 는 쓰기 버퍼에 통째로 넣거나 -1 이다. 그래도 일부만 들어갔다면 다음 줄과 붙지 않게 끊고 재접속한다
            self._last_error = f"링크 {variant} 요청 송신 실패"
            self._parser.last_error = ""
            self._drop()
            return False
        self.requested_variant = variant
        return True

    def frame_age_ms(self) -> int | None:
        return None if self._last_frame_at is None else int((time.monotonic() - self._last_frame_at) * 1000)

    def is_stale(self) -> bool:
        """채널 구분 없는 전체 값. 창은 채널별 신선도를 DisplayModel 로 따로 판단한다."""
        age = self.frame_age_ms()
        return age is None or age > 3000

    def _open(self) -> None:
        if self._socket.state() != QAbstractSocket.SocketState.UnconnectedState:
            return
        self._parser.reset()   # 재접속해도 누적 오류 수와 마지막 오류는 남긴다
        self._pending.clear()
        self._reset_confirmation()   # 새 연결의 첫 패킷은 C 캐시일 수 있다
        self.requested_variant = None   # 요청은 연결마다. 새 연결에 자동으로 다시 보내지 않는다
        self._set_state("connecting")
        self._connect_deadline.start()
        self._socket.connectToHost(QHostAddress(self.host), self.port)

    def _on_connected(self) -> None:
        self._connect_deadline.stop()
        self._set_state("connected")

    def _on_connect_timeout(self) -> None:
        if self.state == "connecting":
            self.socket_error = f"접속 기한 초과({CONNECT_TIMEOUT_MS} ms) · 중계 미기동 또는 차단 가능"
            self._drop()

    def _reset_confirmation(self) -> None:
        """새 연결·새 run: 칸별 확인과 「최근 확인 수신」 시각을 버린다(이전 연결의 확인을 들고 가지 않게)."""
        self._newest.clear()
        self._confirmed.clear()
        self._last_frame_at = None

    def _drop(self) -> None:
        if self.state == "disconnected":
            return
        self._connect_deadline.stop()
        self._flush_timer.stop()
        self._pending.clear()
        self._set_state("disconnected")
        self._socket.abort()
        if not self._retry.isActive():
            self._retry.start()

    def _on_socket_error(self, _error) -> None:
        self.socket_error = self._socket.errorString()[:80]
        self._drop()

    def _set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.state_changed.emit(state)

    def _on_ready_read(self) -> None:
        while self.state == "connected" and self._socket.bytesAvailable() > 0:
            try:
                packets = self._parser.feed(bytes(self._socket.read(READ_CHUNK)))
            except ValueError as error:
                self.framing_errors += 1
                self._last_error = f"경계 오류: {error}"
                self._parser.last_error = ""
                self._drop()   # 경계를 잃으면 다시 맞출 수 없다. 끊고 다시 접속한다
                return
            for channel, envelope, jpeg in packets:
                key = (channel, envelope["variant"])
                self._pending.pop(key, None)      # 도착 순서를 유지하며 같은 칸의 이전 패킷은 바로 버린다
                self._pending[key] = (channel, envelope, jpeg)
        if self._pending and not self._flush_timer.isActive():
            self._flush_timer.start()

    def _flush(self) -> None:
        pending, self._pending = self._pending, {}
        for channel, envelope, jpeg in pending.values():
            if self.state != "connected":
                return
            run_id = envelope["run_id"]
            if run_id != self.run_id:
                if run_id in self._retired:
                    self.stale_session_drops += 1
                    self._last_error = "지난 run_id 패킷 무시"
                    self._parser.last_error = ""
                    continue
                if self.run_id is not None:
                    self._retired.append(self.run_id)
                self.run_id = run_id
                self._reset_confirmation()   # 확인은 run 별
                self.display_session.emit(run_id, LINK_SOURCE)   # 첫 패킷이 fixed·gripper 여도 링크 소스를 먼저 정한다
            image = decode_jpeg(jpeg)
            if image is None:
                self.decode_errors += 1
                self._last_error = f"{channel} JPEG 해독 실패 또는 {MAX_PIXELS} 픽셀 초과"
                self._parser.last_error = ""
                continue
            self.frame_count += 1
            key, order = (channel, envelope["variant"]), (envelope["capture_generation"], envelope["image_seq"])
            newest = self._newest.get(key)
            advanced = newest is not None and order > newest   # 같은 연결·run 에서 순번이 실제로 앞섰다
            if newest is None or advanced:
                self._newest[key] = order
            if advanced:
                self._confirmed.add(key)
                self._last_frame_at = time.monotonic()   # 확인 대기·같은 순번 반복·역행은 「최근 수신」으로 치지 않는다
            confirmed = key in self._confirmed
            self.display_frame.emit(channel, image, dict(envelope, confirmed=confirmed))
