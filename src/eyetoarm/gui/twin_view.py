"""위에서 본 셀 그림. 지금은 펜던트·수동 목표가 만든 "명령 목표"를 그린다(실제 로봇 상태 아님).

Gazebo 상태 스트림이 연결되면 set_command 대신 받은 값을 넣는다.
규칙: 받은 값을 그대로 그리고 표본 사이만 직선 보간한다. 가속·감속·튕김을 넣지 않는다.
"""

from __future__ import annotations

import math
import time
from typing import Any

from PySide6.QtCore import QPointF, QRectF, QTimer, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from .theme import ACC, BG, LINE, T1, T2, T3, mono

W, H = 760.0, 440.0
BASE = QPointF(380.0, 330.0)
L1, L2, MM = 150.0, 130.0, 2.0  # 1 px = 2 mm
MAX_MM_S = 250.0  # shortcut: 미리보기 이동 속도 상한(예시값), 실제 속도는 C·Gazebo 값으로 대체


def solve(tcp: QPointF) -> tuple[QPointF, QPointF, float, bool]:
    """2링크 평면 역기구학(그림용). 팔꿈치·끝점·손목 방향·도달 여부."""
    dx, dy = tcp.x() - BASE.x(), tcp.y() - BASE.y()
    real = math.hypot(dx, dy)
    d = min(L1 + L2 - 1, max(abs(L1 - L2) + 1, real))
    a = math.atan2(dy, dx)
    q2 = math.acos((d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2))
    q1 = a - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
    elbow = QPointF(BASE.x() + L1 * math.cos(q1), BASE.y() + L1 * math.sin(q1))
    tip = QPointF(elbow.x() + L2 * math.cos(q1 + q2), elbow.y() + L2 * math.sin(q1 + q2))
    return elbow, tip, q1 + q2, real <= L1 + L2 - 1


class TwinView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 190)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.goal: dict[str, float] | None = None
        self.shown: dict[str, float] | None = None
        self.grip = self.shown_grip = 85.0
        self.ratio = 0.25
        self._last = time.monotonic()
        self._timer = QTimer(self, interval=16)
        self._timer.timeout.connect(self._step)

    def set_command(self, pendant: dict[str, Any]) -> None:
        self.goal = dict(pendant["cart"])
        self.grip, self.ratio = float(pendant["grip_mm"]), float(pendant["speed_ratio"])
        if self.shown is None:
            self.shown, self.shown_grip = dict(self.goal), self.grip
        if self.isVisible() and not self._timer.isActive():
            self._last = time.monotonic()
            self._timer.start()
        self.update()

    def showEvent(self, _event) -> None:  # noqa: N802
        self._last = time.monotonic()
        self._timer.start()

    def hideEvent(self, _event) -> None:  # noqa: N802 - 안 보일 때는 돌리지 않는다(저사양 PC)
        self._timer.stop()

    def _step(self) -> None:
        now = time.monotonic()
        dt, self._last = min(0.1, now - self._last), now
        if self.goal is None or self.shown is None:
            return
        reach = MAX_MM_S * self.ratio * dt  # 일정한 속도로 직선 이동
        delta = [self.goal[k] - self.shown[k] for k in ("X", "Y", "Z")]
        dist = math.sqrt(sum(v * v for v in delta))
        grip_gap = self.grip - self.shown_grip
        if dist == 0 and grip_gap == 0 and self.shown["RZ"] == self.goal["RZ"]:
            self._timer.stop()  # 목표에 닿으면 멈춘다
            return
        if dist <= reach:
            self.shown.update(self.goal)
        else:
            for key, value in zip(("X", "Y", "Z"), delta):
                self.shown[key] += value * reach / dist
        self.shown["RZ"] = self.goal["RZ"]
        limit = 60.0 * dt
        self.shown_grip += grip_gap if abs(grip_gap) <= limit else math.copysign(limit, grip_gap)
        self.update()

    @staticmethod
    def _px(cart: dict[str, float]) -> QPointF:
        return QPointF(BASE.x() - cart["Y"] / MM, BASE.y() - cart["X"] / MM)

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(LINE), 1))
        painter.setBrush(QColor(BG))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(.5, .5, -.5, -.5), 8, 8)
        scale = min(self.width() / W, self.height() / H)
        painter.translate((self.width() - W * scale) / 2, (self.height() - H * scale) / 2)
        painter.scale(scale, scale)

        dashed = QPen(QColor("#d4d4d4"), 1, Qt.PenStyle.DashLine)
        painter.setPen(dashed)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(QRectF(40, 86, 620, 68), 34, 34)
        painter.setPen(QPen(QColor("#ececec"), 1, Qt.PenStyle.DotLine))
        painter.drawEllipse(BASE, 150, 150)
        painter.drawEllipse(BASE, 280, 280)
        painter.setPen(QColor(T3))
        painter.drawText(QPointF(52, 78), "컨베이어 자리 · 위치는 상태 스트림 연결 뒤 표시")
        painter.drawText(QPointF(336, 372), "로봇 베이스 (0, 0)")
        if self.goal is None or self.shown is None:
            painter.drawText(QRectF(0, 0, W, H), Qt.AlignmentFlag.AlignCenter, "명령 목표 없음")
            return

        goal = self._px(self.goal)
        painter.setPen(QPen(QColor(ACC), 1.5, Qt.PenStyle.DashLine))
        painter.drawEllipse(goal, 9, 9)
        painter.setPen(QPen(QColor(ACC), 1))
        painter.drawLine(goal + QPointF(-14, 0), goal + QPointF(14, 0))
        painter.drawLine(goal + QPointF(0, -14), goal + QPointF(0, 14))

        elbow, tip, _wrist, reachable = solve(self._px(self.shown))
        painter.setPen(QPen(QColor(T1), 12, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawLine(BASE, elbow)
        painter.setPen(QPen(QColor(T1), 9, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawLine(elbow, tip)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(T1))
        painter.drawEllipse(BASE, 20, 20)
        painter.setBrush(QColor(BG))
        painter.drawEllipse(BASE, 7, 7)
        painter.setPen(QPen(QColor(T1), 3))
        painter.drawEllipse(elbow, 7, 7)

        painter.save()  # 그리퍼: 장축각만큼 돌리고 개구폭만큼 벌린다
        painter.translate(tip)
        painter.rotate(-90 - self.shown["RZ"])
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(ACC))
        jaw = 3 + self.shown_grip / 9
        painter.drawRoundedRect(QRectF(-6, -12, 12, 24), 3, 3)
        painter.drawRoundedRect(QRectF(4, -jaw - 4, 16, 4), 2, 2)
        painter.drawRoundedRect(QRectF(4, jaw, 16, 4), 2, 2)
        painter.restore()

        painter.setPen(QColor(T2))
        painter.setFont(mono(9))
        s = self.shown
        text = f"명령 목표 X {s['X']:.1f}  Y {s['Y']:.1f}  Z {s['Z']:.1f} mm  θ {s['RZ']:.1f}°  개구폭 {self.shown_grip:.0f} mm"
        painter.drawText(QPointF(52, 424), text + ("" if reachable else "  · 그림의 팔 길이를 넘는 위치"))
