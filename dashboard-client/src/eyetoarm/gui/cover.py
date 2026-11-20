"""표지 화면: 앱을 켜면 대시보드 위에 잠깐 덮이고, UR5e 그림이 인사하듯 손을 흔든다.

[담당] D 전용. 사용자 지정 2026-11-20.
[꾸밈일 뿐이다] 이 로봇은 QPainter 로 그린 그림이다. Gazebo 로봇·실제 관절 자세·C 서버 상태와 무관하다.
  표지가 떠 있는 동안에도 아래 창의 연결·수신·기록은 평소대로 시작한다(표지가 막거나 늦추지 않는다).
  표지는 정해진 시간 뒤에 사라질 뿐이며 「불러오는 중」이나 준비 완료를 뜻하지 않는다.
[닫기] HOLD_MS 뒤 자동으로, 또는 클릭·Enter·Space·Esc 로 바로 닫힌다. 닫히면 finished 신호를 한 번 낸다.
[모션 줄이기] 팔을 든 자세로 멈춘 그림만 보이고 곧 닫힌다. 흔들림·흐려짐이 없다.
[그림 자산 없음] 로고·사진·3D 모델 파일을 쓰지 않는다. 선과 원으로만 그린다(회색 팔 + 파란 관절 덮개 + 손에 든 「Welcome」 피켓).
"""

from __future__ import annotations

import math

from PySide6.QtCore import QElapsedTimer, QEvent, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QGraphicsOpacityEffect, QWidget

from eyetoarm.gui import motion, theme

HOLD_MS = 3600        # 표지가 떠 있는 시간
RISE_MS = 900         # 접힌 팔이 일어서는 시간
FADE_MS = 360         # 닫힐 때 흐려지는 시간
STILL_MS = 1200       # 모션 줄이기일 때 멈춘 그림을 보여 주는 시간
WAVE_HZ = 1.7         # 손을 흔드는 빠르기(1초에 왕복 횟수)

_FOLDED = (-62.0, 128.0, 10.0)   # (어깨, 팔꿈치, 손목) 각도. 도. 위쪽이 0, 시계 방향이 +, 앞 마디 기준 상대각
_RAISED = (-14.0, 24.0, 0.0)
_LINK = "#D3D8DF"                # 팔 마디(밝은 회색)
_CAP = "#4DB5FF"                 # 관절 덮개(파랑)


def wave_pose(ms: float) -> tuple[float, float, float, float]:
    """표지를 연 뒤 ms 가 지났을 때의 (어깨, 팔꿈치, 손목) 각도와 흔들림 세기(0~1).

    0 → RISE_MS: 접힌 자세에서 든 자세로 일어선다(끝으로 갈수록 느려진다).
    그 뒤: 팔꿈치와 손목이 좌우로 흔들린다. 손목은 팔꿈치보다 조금 늦게 따라온다.
    """
    rise = max(0.0, min(1.0, ms / RISE_MS))
    eased = 1.0 - (1.0 - rise) ** 3
    shoulder, elbow, wrist = (a + (b - a) * eased for a, b in zip(_FOLDED, _RAISED))
    strength = max(0.0, min(1.0, (ms - RISE_MS * 0.7) / 400.0))
    phase = 2.0 * math.pi * WAVE_HZ * ms / 1000.0
    elbow += strength * 20.0 * math.sin(phase)
    wrist += strength * 30.0 * math.sin(phase - 0.9)
    return shoulder, elbow, wrist, strength


def _font(px: float, weight: int, mono: bool = False) -> QFont:
    font = QFont()
    font.setFamilies(theme.MONO_FAMILIES if mono else theme.FONT_FAMILIES)
    font.setPixelSize(max(9, round(px)))
    font.setWeight(QFont.Weight(weight))
    return font


class CoverScreen(QWidget):
    """parent 를 꽉 덮는 표지. 만들면 바로 보이고, 닫힌 뒤 스스로 지워진다."""

    finished = Signal()   # 표지가 닫혔다(한 번). 창이 이어서 카드 등장 모션을 건다

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._clock = QElapsedTimer()
        self._clock.start()
        self._closing = False
        self._ticker = QTimer(self, interval=16)
        self._ticker.timeout.connect(self._tick)
        self.setAccessibleName("표지 화면")
        self.setAccessibleDescription("UR5e 그림이 인사합니다. 클릭하거나 Enter 를 누르면 대시보드로 넘어갑니다.")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        parent.installEventFilter(self)   # 창 크기가 바뀌면 따라서 덮는다
        self.setGeometry(parent.rect())
        self.raise_()
        self.show()
        self.setFocus()
        if motion.is_reduced():
            QTimer.singleShot(STILL_MS, self, self.dismiss)
        else:
            self._ticker.start()

    def dismiss(self) -> None:
        """표지를 닫는다. 여러 번 불러도 한 번만 닫힌다."""
        if self._closing:
            return
        self._closing = True

        def end() -> None:
            self._ticker.stop()
            self.hide()
            self.finished.emit()
            self.deleteLater()

        effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(effect)
        motion.tween(self, 1.0, 0.0, FADE_MS, effect.setOpacity, done=end)   # 모션 줄이기면 즉시 닫힌다

    # ── 입력 ────────────────────────────────────────────────────────────

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if watched is self.parent() and event.type() == QEvent.Type.Resize:
            self.setGeometry(watched.rect())
        return False

    def mousePressEvent(self, _event) -> None:  # noqa: N802
        self.dismiss()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space, Qt.Key.Key_Escape):
            self.dismiss()
        else:
            super().keyPressEvent(event)

    def _tick(self) -> None:
        if not self._closing and self._clock.elapsed() >= HOLD_MS:
            self.dismiss()
        self.update()

    # ── 그리기 ──────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:  # noqa: N802
        ms = float(RISE_MS + 400) if motion.is_reduced() else float(self._clock.elapsed())
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width, height = self.width(), self.height()
        unit = min(width, height) / 820.0   # 창 크기에 맞춰 그림 전체를 키우거나 줄인다

        background = QLinearGradient(0, 0, width, height)
        background.setColorAt(0.0, QColor(theme.GLOW_A))
        background.setColorAt(0.25, QColor(theme.BG))
        background.setColorAt(0.75, QColor(theme.BG))
        background.setColorAt(1.0, QColor(theme.GLOW_B))
        painter.fillRect(self.rect(), background)

        base = QPointF(width / 2.0, height * 0.46 + 150.0 * unit)
        halo = QRadialGradient(QPointF(base.x(), base.y() - 150.0 * unit), 330.0 * unit)
        glow = QColor(theme.ACCENT)
        glow.setAlpha(34 if theme.name == "dark" else 70)
        halo.setColorAt(0.0, glow)
        halo.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.fillRect(self.rect(), halo)

        self._draw_robot(painter, base, unit, ms)
        self._draw_text(painter, base, unit, ms)

    def _draw_robot(self, painter: QPainter, base: QPointF, unit: float, ms: float) -> None:
        shoulder_angle, elbow_angle, wrist_angle, strength = wave_pose(ms)

        def step(origin: QPointF, degrees: float, length: float) -> QPointF:
            radians = math.radians(degrees)
            return QPointF(origin.x() + math.sin(radians) * length * unit, origin.y() - math.cos(radians) * length * unit)

        shoulder = QPointF(base.x(), base.y() - 46.0 * unit)
        elbow = step(shoulder, shoulder_angle, 128.0)
        wrist = step(elbow, shoulder_angle + elbow_angle, 112.0)
        hand_angle = shoulder_angle + elbow_angle + wrist_angle
        palm = step(wrist, hand_angle, 40.0)

        edge = QColor(theme.DIM)
        edge.setAlpha(150)
        link = QColor(_LINK)

        # 받침대
        painter.setPen(QPen(edge, 1.5 * unit))
        painter.setBrush(QColor(theme.SURFACE_2))
        painter.drawRoundedRect(QRectF(base.x() - 62.0 * unit, base.y() - 6.0 * unit, 124.0 * unit, 22.0 * unit), 8.0 * unit, 8.0 * unit)
        painter.setBrush(link)
        painter.drawRoundedRect(QRectF(base.x() - 26.0 * unit, base.y() - 46.0 * unit, 52.0 * unit, 42.0 * unit), 10.0 * unit, 10.0 * unit)

        # 마디: 굵은 둥근 선. 아래에 어두운 선을 한 번 더 그어 테두리를 만든다
        for start, end, thickness in ((shoulder, elbow, 30.0), (elbow, wrist, 24.0), (wrist, palm, 18.0)):
            for color, extra in ((edge, 3.0), (link, 0.0)):
                pen = QPen(color, (thickness + extra) * unit)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawLine(start, end)

        radians = math.radians(hand_angle)
        along = QPointF(math.sin(radians), -math.cos(radians))
        across = QPointF(along.y(), -along.x())

        # 피켓(사용자 지정 2026-11-20): 손에 쥔 막대와 「Welcome」 판. 손과 함께 기울어 흔들린다. 손가락보다 먼저 그려 쥔 것처럼 보이게 한다
        pole_end = QPointF(palm.x() + along.x() * 46.0 * unit, palm.y() + along.y() * 46.0 * unit)
        for color, thickness in ((edge, 9.0), (QColor(theme.SURFACE_2), 6.0)):
            pen = QPen(color, thickness * unit)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(palm, pole_end)
        painter.save()
        painter.translate(palm.x() + along.x() * 72.0 * unit, palm.y() + along.y() * 72.0 * unit)
        painter.rotate(hand_angle)
        board = QRectF(-82.0 * unit, -28.0 * unit, 164.0 * unit, 56.0 * unit)
        painter.setPen(QPen(edge, 1.5 * unit))
        painter.setBrush(QColor(theme.ACCENT))
        painter.drawRoundedRect(board, 12.0 * unit, 12.0 * unit)
        painter.setPen(QColor(theme.ON_ACCENT))
        painter.setFont(_font(30.0 * unit, 800))
        painter.drawText(board, Qt.AlignmentFlag.AlignCenter, "Welcome")
        painter.restore()

        # 손가락 두 개(2지 그리퍼)
        for side in (-1.0, 1.0):
            root = QPointF(palm.x() + across.x() * side * 11.0 * unit, palm.y() + across.y() * side * 11.0 * unit)
            tip = QPointF(root.x() + along.x() * 26.0 * unit, root.y() + along.y() * 26.0 * unit)
            pen = QPen(link, 7.0 * unit)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(root, tip)

        # 관절 덮개
        for center, radius in ((shoulder, 22.0), (elbow, 19.0), (wrist, 15.0)):
            painter.setPen(QPen(edge, 1.5 * unit))
            painter.setBrush(QColor(_CAP))
            painter.drawEllipse(center, radius * unit, radius * unit)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(theme.ON_ACCENT))
            painter.drawEllipse(center, radius * 0.32 * unit, radius * 0.32 * unit)

        # 피켓 양옆의 흔들림 선(인사 표시). 흔들기 시작하면 나타난다
        if strength > 0.0:
            mark = QColor(theme.ACCENT_INK)
            mark.setAlphaF(0.75 * strength)
            pen = QPen(mark, 4.0 * unit)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for reach in (150.0, 170.0):
                box = QRectF(wrist.x() - reach * unit, wrist.y() - reach * unit, 2 * reach * unit, 2 * reach * unit)
                for side in (-1.0, 1.0):   # Qt 의 호 각도: 3시 방향이 0, 반시계가 +
                    painter.drawArc(box, round((90.0 - hand_angle + side * 46.0 - 6.0) * 16), round(12.0 * 16))

    def _draw_text(self, painter: QPainter, base: QPointF, unit: float, ms: float) -> None:
        shown = max(0.0, min(1.0, (ms - 300.0) / 500.0))   # 글자는 팔이 일어서는 동안 올라오며 나타난다
        lift = (1.0 - shown) * 14.0 * unit
        painter.setOpacity(shown)
        painter.setPen(QColor(theme.DIM))
        painter.setFont(_font(12.0 * unit, 700, mono=True))
        painter.drawText(QRectF(0, base.y() + 46.0 * unit + lift, self.width(), 20.0 * unit), Qt.AlignmentFlag.AlignCenter, "U R 5 e  ·  안녕하세요")
        painter.setPen(QColor(theme.TEXT))
        painter.setFont(_font(46.0 * unit, 800))
        painter.drawText(QRectF(0, base.y() + 70.0 * unit + lift, self.width(), 62.0 * unit), Qt.AlignmentFlag.AlignCenter, "Eye To Arms")
        painter.setPen(QColor(theme.DIM))
        painter.setFont(_font(14.0 * unit, 500))
        painter.drawText(QRectF(0, base.y() + 136.0 * unit + lift, self.width(), 24.0 * unit), Qt.AlignmentFlag.AlignCenter, "팀명: 좌표로")
        painter.setOpacity(1.0)

        # 아래: 닫는 방법과 남은 시간 줄(시간 표시일 뿐 불러오기 진행률이 아니다)
        painter.setPen(QColor(theme.DIM))
        painter.setFont(_font(11.0, 600, mono=True))
        painter.drawText(QRectF(0, self.height() - 58.0, self.width(), 18.0), Qt.AlignmentFlag.AlignCenter, "클릭 · Enter 로 바로 시작")
        track = QRectF(self.width() / 2.0 - 90.0, self.height() - 30.0, 180.0, 3.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.LINE))
        painter.drawRoundedRect(track, 1.5, 1.5)
        if not motion.is_reduced():
            painter.setBrush(QColor(theme.ACCENT_INK))
            painter.drawRoundedRect(QRectF(track.left(), track.top(), track.width() * min(1.0, ms / HOLD_MS), track.height()), 1.5, 1.5)
