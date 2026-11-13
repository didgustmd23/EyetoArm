"""모의 데이터(--demo). C 서버·Gazebo 없이 화면 외형과 모션만 확인할 때 쓴다.

[중요] 여기서 나오는 값과 화면은 전부 가짜다. 실제 C 서버 응답이나 Gazebo 영상이 아니다.
  - 창 좌하단 출처 표시가 「모의 데이터」로 바뀌고, 모의 화면 안에도 「모의 화면」이라고 적힌다.
  - 이 모드에서 본 결과를 실제 통신·영상 검증 결과로 보고하지 않는다.
[구조] StatusLink · GzViewReceiver 와 같은 신호·메서드를 가진 대역이다. 창 코드는 둘을 구분하지 않는다.
  소켓을 열지 않는다. 아무 것도 보내지 않는다.
"""

from __future__ import annotations

import time

from PySide6.QtCore import QObject, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter

from eyetoarm.gui import theme
from eyetoarm.gui.client import DEFAULTS, TrafficEntry

# (연결 상태, robot_ready, not_ready_reasons). 4초마다 다음 장면으로 넘어간다
SCENES = [("connected", True, []), ("connected", False, ["NOT_STOPPED"]),
          ("connected", False, ["RECOVERING"]), ("disconnected", None, [])]


class DemoLink(QObject):
    """StatusLink 대역. 정상 → 미준비 → 복구 중 → 끊김 장면을 돌려 가며 내보낸다."""

    state_changed = Signal(str)
    status_received = Signal(dict)
    traffic = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.config = dict(DEFAULTS)
        self.host, self.port = "모의", 0
        self.state = "disconnected"
        self._scene = -1
        self._last_at: float | None = None
        self._scene_timer = QTimer(self, interval=4000)
        self._scene_timer.timeout.connect(self._next_scene)
        self._poll = QTimer(self, interval=1000)
        self._poll.timeout.connect(self._emit_status)

    def start(self) -> None:
        self._next_scene()
        self._scene_timer.start()
        self._poll.start()

    def stop(self) -> None:
        self._scene_timer.stop()
        self._poll.stop()

    def status_age_ms(self) -> int | None:
        return None if self._last_at is None else int((time.monotonic() - self._last_at) * 1000)

    def is_stale(self) -> bool:
        age = self.status_age_ms()
        return self.state != "connected" or age is None or age > self.config["status_watchdog_ms"]

    def _next_scene(self) -> None:
        self._scene = (self._scene + 1) % len(SCENES)
        state = SCENES[self._scene][0]
        if state != self.state:
            self.state = state
            if state != "connected":
                self._last_at = None
            self._log("EV", "모의 접속" if state == "connected" else "모의 끊김", "모의 장면 전환")
            self.state_changed.emit(state)
        self._emit_status()

    def _emit_status(self) -> None:
        state, ready, reasons = SCENES[self._scene]
        if state != "connected":
            return
        self._last_at = time.monotonic()
        self._log("TX", "준비 조회", "모의 · 실제로 보내지 않음")
        self._log("RX", "STATUS", f"모의 · robot_ready={str(ready).lower()} 사유={','.join(reasons) or '-'}")
        self.status_received.emit({"robot_ready": ready, "not_ready_reasons": list(reasons), "active_command_id": None})

    def _log(self, direction: str, kind: str, summary: str) -> None:
        now = time.time()
        wall = time.strftime("%H:%M:%S", time.localtime(now)) + f".{int(now * 1000) % 1000:03d}"
        self.traffic.emit(TrafficEntry(wall, direction, kind, summary))


class DemoView(QObject):
    """GzViewReceiver 대역. 2 Hz 로 「모의 화면」이라고 적힌 그림을 내보낸다. Gazebo 영상이 아니다."""

    state_changed = Signal(str)
    frame_received = Signal(QImage)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.title, self.source_name = "모의", "모의 화면"
        self.help_text, self.stale_label = "모의 화면 대기", "프레임 중단"
        self.state = "disconnected"
        self.frame_count = 0
        self._last_at: float | None = None
        self._timer = QTimer(self, interval=500)   # 실제 카메라 update_rate 2 Hz 와 같은 간격
        self._timer.timeout.connect(self._emit_frame)

    def start(self) -> None:
        self.state = "connected"
        self.state_changed.emit(self.state)
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def frame_age_ms(self) -> int | None:
        return None if self._last_at is None else int((time.monotonic() - self._last_at) * 1000)

    def is_stale(self) -> bool:
        age = self.frame_age_ms()
        return age is None or age > 3000

    def _emit_frame(self) -> None:
        self.frame_count += 1
        image = QImage(1280, 720, QImage.Format.Format_RGB32)
        image.fill(QColor("#101012"))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QColor(theme.DISPLAY_LINE))
        for x in range(0, 1280, 80):
            painter.drawLine(x, 0, x, 720)
        for y in range(0, 720, 80):
            painter.drawLine(0, y, 1280, y)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.DISPLAY_ACCENT))
        painter.drawRoundedRect(QRectF(80 + (self.frame_count * 40) % 1040, 520, 80, 24), 12, 12)
        painter.setPen(QColor(theme.DISPLAY_TEXT))
        font = painter.font()
        font.setPixelSize(56)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(image.rect(), Qt.AlignmentFlag.AlignCenter, "모의 화면 · Gazebo 영상 아님")
        painter.end()
        self._last_at = time.monotonic()
        self.frame_received.emit(image)
