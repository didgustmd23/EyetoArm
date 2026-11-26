"""D 운영 GUI 화면 부품. 통신을 모른다. 값을 받아 그리기만 한다.

┌─ 부품과 연동 파트 ─────────────────────────────────────────────────────────┐
│ NavRail       상단 화면 선택(조작·설정 / 운영 대시보드 / 평가·기록)   [D]    │
│ StatusChip    상단 상태띠 한 칸                                      [C 값]  │
│ DisplayView   영상 표시면. 메인 = Gazebo 로봇 화면                   [D 임시]│
│               보조 2칸 = 그리퍼·링크 영상 자리                       [C 예정]│
│ DetailPopup   메인 디스플레이 위에 겹치는 측정·보정 상세             [A·B 예정]│
│ StageStrip    공정 단계 표시                                         [C 예정]│
│ TrafficTable  TX/RX 기록 표(GUI-03)                                  [C 값]  │
└────────────────────────────────────────────────────────────────────────────┘
[C 값]   = C 서버가 보낸 값을 그대로 표시.  [C 예정]·[A·B 예정] = 계약 미결이라 자리만 있음.
자리만 있는 칸에는 가짜 값을 넣지 않는다. 「—」와 사유를 표시한다(D 상세설계 §A.2).
"""

from __future__ import annotations

import math
import time
from collections import deque
from functools import lru_cache

from PySide6.QtCore import QEasingCurve, QPointF, QPropertyAnimation, QRect, QRectF, QSize, QSizeF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QKeySequence, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QFrame, QGraphicsOpacityEffect, QHBoxLayout, QHeaderView, QLabel, QPushButton, QSizePolicy, QStyle, QStyleOptionButton,
    QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from eyetoarm.gui import motion, theme
from eyetoarm.gui.icon import draw_icon


def _font(px: int, weight: int = 500, mono: bool = False) -> QFont:
    font = QFont()
    font.setFamilies(theme.MONO_FAMILIES if mono else theme.FONT_FAMILIES)
    font.setPixelSize(px)
    font.setWeight(QFont.Weight(weight))
    return font


def label(text: str, role: str = "Dim", wrap: bool = False) -> QLabel:
    """role 은 theme.QSS 의 objectName: Caption · Dim · Value · Mono · CardTitle · Tag · PageTitle."""
    widget = QLabel(text)
    widget.setObjectName(role)
    widget.setWordWrap(wrap)
    return widget


class FeedbackPill(QPushButton):
    """알약 버튼 + 누름·응답 대기·결과 표시. 화면 장식일 뿐 송신 여부·C 상태 판단과 무관하다(motion 규칙 1~4).

    누르는 동안: 버튼이 살짝 눌려 작아졌다가, 떼면 튀어 오르며 돌아온다.
    pulse(color): 누른 곳(키보드면 가운데)에서 색이 퍼지고, 같은 색의 선이 테두리를 한 바퀴 돈다. 반복하지 않는다.
    result(color): C 결과 색이 바탕에 퍼지고 테두리 전체가 그 색으로 한 번 밝아졌다 옅어진다.
    set_waiting(on): 응답을 기다리는 동안 테두리를 따라 주황 선이 돈다. 모션 줄이기면 테두리만 주황으로 고정.
    set_guide(color): 「다음에 누를 버튼」 안내. 테두리가 옅게 빛나고 네온 선 두 줄이 둘레를 돈다(응답 대기가 우선).
                      None 이면 끈다. 모션 줄이기면 돌지 않고 테두리만 그 색으로 빛난다.
    """

    PRESS_SCALE = 0.94   # 눌렸을 때 크기
    LINE = 1.5           # 효과 테두리 선 두께(Qt 논리 px)

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self._wash = 0.0
        self._wash_color = QColor()
        self._origin: QPointF | None = None
        self._press_at: QPointF | None = None
        self._press = 0.0          # 0 = 평소, 1 = 끝까지 눌림
        self._trace = -1.0         # 테두리 한 바퀴 진행(0~1). 음수 = 없음
        self._trace_color = QColor()
        self._ring = 0.0           # 결과 테두리 밝기(1→0)
        self._ring_color = QColor()
        self._waiting = False
        self._guide: QColor | None = None
        self._spin = 0.0
        self._spin_timer = QTimer(self)
        self._spin_timer.setInterval(30)
        self._spin_timer.timeout.connect(self._turn)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._press_at = event.position()
        if self.isEnabled():
            motion.tween(self, self._press, 1.0, theme.MS_PRESS_IN, self._set_press)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        motion.tween(self, self._press, 0.0, theme.MS_PRESS_OUT, self._set_press, motion.EASE_BACK)
        super().mouseReleaseEvent(event)

    def pulse(self, color: str) -> None:
        self._wash_color = self._trace_color = QColor(color)
        self._origin, self._press_at = self._press_at, None   # 마우스로 누른 한 번에만 그 위치를 쓴다
        if not motion.is_reduced():
            self._set_wash(1.0)    # 애니메이션 첫 값을 기다리지 않고 누른 즉시 보인다
            self._set_trace(0.0)
        motion.tween(self, 1.0, 0.0, theme.MS_FLASH, self._set_wash)
        motion.tween(self, 0.0, 1.0, theme.MS_TRACE, self._set_trace, motion.EASE_IO,
                     done=lambda: self._set_trace(-1.0))

    def result(self, color: str) -> None:
        self._wash_color = self._ring_color = QColor(color)
        self._origin = None
        if not motion.is_reduced():
            self._set_wash(1.0)
            self._set_ring(1.0)
        motion.tween(self, 1.0, 0.0, theme.MS_FLASH, self._set_wash)
        motion.tween(self, 1.0, 0.0, theme.MS_RING, self._set_ring)

    def set_waiting(self, on: bool) -> None:
        on = bool(on)
        if on == self._waiting:
            return
        self._waiting = on
        self._sync_spin()

    def is_waiting(self) -> bool:
        return self._waiting

    def set_guide(self, color: str | None) -> None:
        guide = QColor(color) if color else None
        if guide == self._guide:
            return
        self._guide = guide
        self._sync_spin()

    def guide(self) -> QColor | None:
        return self._guide

    def _sync_spin(self) -> None:
        """대기 선·안내 네온이 있을 때만 돈다. 모션 줄이기면 멈춘 채 테두리만 칠한다."""
        if (self._waiting or self._guide is not None) and not motion.is_reduced():
            self._spin_timer.start()
        else:
            self._spin_timer.stop()
        self.update()

    def _set_wash(self, value: float) -> None:
        self._wash = float(value)
        self.update()

    def _set_press(self, value: float) -> None:
        self._press = float(value)
        self.update()

    def _set_trace(self, value: float) -> None:
        self._trace = float(value)
        self.update()

    def _set_ring(self, value: float) -> None:
        self._ring = float(value)
        self.update()

    def _turn(self) -> None:
        if motion.is_reduced():   # 돌던 중 모션 줄이기를 켜면 바로 멈추고 고정 테두리로 그린다
            self._spin_timer.stop()
        else:
            self._spin = (self._spin + 0.018) % 1.0
        self.update()

    def _line(self, painter: QPainter, shape: QPainterPath, color: QColor, start: float | None) -> None:
        """테두리 선. start 가 있으면 둘레의 1/4 길이 선을 그 위치(0~1)에, 없으면 테두리 전체."""
        pen = QPen(color, self.LINE)
        if start is not None:
            length = shape.length() / self.LINE
            pen.setDashPattern([length * 0.25, length * 0.75])
            pen.setDashOffset(-length * start)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(shape)

    def _neon(self, painter: QPainter, shape: QPainterPath, rect: QRectF) -> None:
        """안내 네온: 옅게 빛나는 테두리 + 마주 보는 두 줄기 빛이 둘레를 돈다(번짐 → 심지 순서로 겹쳐 그린다)."""
        color = QColor(self._guide)
        halo = QColor(color)
        halo.setAlphaF(0.30)
        painter.setPen(QPen(halo, self.LINE * 0.7))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(shape)                             # 멈춰 있어도 보이는 은은한 테두리
        if not self._spin_timer.isActive():                 # 모션 줄이기: 테두리만 선명하게
            self._line(painter, shape, color, None)
            return
        length = shape.length()
        for offset in (0.0, 0.5):                           # 마주 보는 두 줄기
            start = (self._spin + offset) % 1.0
            for width, alpha in ((self.LINE * 3.2, 0.22), (self.LINE * 1.9, 0.45), (self.LINE, 1.0)):
                glow = QColor(color)
                glow.setAlphaF(alpha)
                pen = QPen(glow, width)
                units = length / width
                pen.setDashPattern([units * 0.18, units * 0.82])
                pen.setDashOffset(-units * start)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawPath(shape)

    def _draw_button(self, painter: QPainter) -> None:
        """QSS 바탕·테두리는 2배 해상도로 그려 줄인다(둥근 1px 테두리 계단 방지). 글자는 1배로 선명하게."""
        option = QStyleOptionButton()
        self.initStyleOption(option)
        ratio = self.devicePixelRatioF() * 2
        bevel = QImage((QSizeF(self.size()) * ratio).toSize(), QImage.Format.Format_ARGB32_Premultiplied)
        bevel.setDevicePixelRatio(ratio)
        bevel.fill(Qt.GlobalColor.transparent)
        with QPainter(bevel) as sub:
            self.style().drawControl(QStyle.ControlElement.CE_PushButtonBevel, option, sub, self)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(QRectF(self.rect()), bevel)
        painter.restore()
        self.style().drawControl(QStyle.ControlElement.CE_PushButtonLabel, option, painter, self)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if (self._press <= 0.0 and self._wash <= 0.0 and self._trace < 0.0 and self._ring <= 0.0 and not self._waiting
                and self._guide is None):
            self._draw_button(painter)
            return
        if self._press > 0.0:   # 가운데를 기준으로 살짝 작게 그린다(배치 크기는 그대로)
            scale = 1.0 - (1.0 - self.PRESS_SCALE) * self._press
            center = QRectF(self.rect()).center()
            painter.translate(center)
            painter.scale(scale, scale)
            painter.translate(-center)
        self._draw_button(painter)
        inset = self.LINE / 2
        rect = QRectF(self.rect()).adjusted(inset, inset, -inset, -inset)
        radius = rect.height() / 2
        shape = QPainterPath()
        shape.addRoundedRect(rect, radius, radius)
        if self._wash > 0.0:
            origin = self._origin or rect.center()
            reach = math.hypot(max(origin.x(), rect.width() - origin.x()), max(origin.y(), rect.height() - origin.y()))
            wash = QColor(self._wash_color)
            wash.setAlphaF(0.42 * self._wash)
            painter.save()
            painter.setClipPath(shape)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(wash)
            size = reach * (1.0 - self._wash * self._wash) + 6.0
            painter.drawEllipse(origin, size, size)
            painter.restore()
        if self._ring > 0.0:
            ring = QColor(self._ring_color)
            ring.setAlphaF(self._ring)
            self._line(painter, shape, ring, None)
        if self._waiting:
            self._line(painter, shape, QColor(theme.ACCENT), self._spin if self._spin_timer.isActive() else None)
        elif self._guide is not None:
            self._neon(painter, shape, rect)
        if not self._waiting and self._trace >= 0.0:   # 한 바퀴 돌고 마지막 1/5 구간에서 옅어진다
            trace = QColor(self._trace_color)
            trace.setAlphaF(min(1.0, (1.0 - self._trace) * 5.0))
            self._line(painter, shape, trace, self._trace)
        painter.end()


def pill(text: str, role: str = "", tooltip: str = "", checkable: bool = False) -> QPushButton:
    """알약 버튼. role: ""(기본) · "primary"(강조) · "stop"(정지 계열)."""
    button = FeedbackPill(text)
    button.setObjectName("Pill")
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setCheckable(checkable)
    if role:
        button.setProperty("role", role)
    if tooltip:
        button.setToolTip(tooltip)
    return button


class Card(QFrame):
    """둥근 카드. 제목 오른쪽 꼬리표(tag)에 데이터 출처·계약 상태를 적는다."""

    def __init__(self, title: str = "", tag: str = "") -> None:
        super().__init__()
        self.setObjectName("Card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(16, 13, 16, 13)
        self.body.setSpacing(8)
        if title:
            head = QHBoxLayout()
            parts = [label(title, "CardTitle")] + ([label(tag, "Tag")] if tag else [])
            for part in parts:   # 카드가 커져도 제목·꼬리표는 제 높이를 지킨다(꼬리표가 세로로 늘어나지 않게)
                part.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            head.addWidget(parts[0])
            head.addStretch(1)
            for part in parts[1:]:
                head.addWidget(part, 0, Qt.AlignmentFlag.AlignVCenter)
            self.body.addLayout(head)


# ── 상단 화면 선택 ────────────────────────────────────────────────────────

class NavRail(QFrame):
    """화면 선택 띠(창 맨 위 한 줄). 캡슐 안의 흰 알약이 누른 탭으로 미끄러져 간다(theme.MS_NAV).

    사용자 지정 2026-11-20: 참고 릴스처럼 좌측 세로 목록을 상단 탭으로 옮겼다. 고르는 방법(클릭·Ctrl+1~3)과
    신호(selected)는 그대로다. 이름 NavRail 은 다른 코드가 쓰고 있어 그대로 둔다.
    """

    selected = Signal(int)
    shortcut_used = Signal(int)   # Ctrl+1~3 으로 골랐다. 창이 단축키 표시(KeyHint)를 띄운다

    def __init__(self, items: list[str]) -> None:
        super().__init__()
        self.setObjectName("NavRail")
        self.setFixedHeight(50)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(10)
        mark = QLabel()                    # 상표 표식 = 프로그램 아이콘(icon.py, 좌표축과 점). 그림 파일을 쓰지 않는다
        mark.setPixmap(draw_icon(26))
        mark.setFixedSize(26, 26)
        layout.addWidget(mark)
        layout.addWidget(label("Eye To Arms", "Brand"))
        layout.addSpacing(10)

        self._list = QFrame()
        self._list.setObjectName("NavTabs")
        self._list.setFixedHeight(34)
        rows = QHBoxLayout(self._list)
        rows.setContentsMargins(3, 3, 3, 3)
        rows.setSpacing(2)
        self._indicator = QFrame(self._list)
        self._indicator.setObjectName("NavIndicator")
        self._group = QButtonGroup(self)
        self._buttons: list[QPushButton] = []
        for index, text in enumerate(items):
            button = QPushButton(text)
            button.setObjectName("NavButton")
            button.setCheckable(True)
            button.setFixedHeight(28)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setToolTip(f"Ctrl+{index + 1}")
            QShortcut(QKeySequence(f"Ctrl+{index + 1}"), self).activated.connect(lambda i=index: self._on_shortcut(i))
            self._group.addButton(button, index)
            self._buttons.append(button)
            rows.addWidget(button)
        self._indicator.lower()
        self._group.idClicked.connect(self._on_clicked)
        layout.addWidget(self._list)
        layout.addStretch(1)
        self.footer = QHBoxLayout()   # 창이 출처 표시·테마 선택·모션 줄이기 체크를 넣는다(오른쪽 끝)
        self.footer.setSpacing(12)
        layout.addLayout(self.footer)
        self._layout_timer = QTimer(self, singleShot=True)
        self._layout_timer.timeout.connect(self._snap)

    def select(self, index: int, animate: bool = True) -> None:
        self._buttons[index].setChecked(True)
        self._buttons[index].setFocus()   # 키보드 초점도 선택한 항목으로 옮긴다
        target = self._buttons[index].geometry()
        if animate:
            motion.move_geometry(self._indicator, target, theme.MS_NAV, motion.EASE_BACK)
        else:
            self._indicator.setGeometry(target)

    def _on_clicked(self, index: int) -> None:
        self.select(index)
        self.selected.emit(index)

    def _on_shortcut(self, index: int) -> None:
        self.shortcut_used.emit(index)
        self._on_clicked(index)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt 이름
        super().resizeEvent(event)
        self._layout_timer.start(0)   # 소유 위젯이 삭제되면 예약된 배치 보정도 종료한다.

    def _snap(self) -> None:
        index = self._group.checkedId()
        if index >= 0:
            self._indicator.setGeometry(self._buttons[index].geometry())


# ── 화면 전환 · 단축키 표시 ───────────────────────────────────────────────

class PageZoom(QWidget):
    """화면 전환 모션: 떠나는 화면의 「찍어 둔 그림」이 작아지거나 커지면서 옅어진다(theme.MS_PAGE).

    [값은 즉시] 새 화면은 이미 아래에 떠 있고 살아 있다. 이 위젯은 그 위에 옛 화면의 그림을 잠깐 겹칠 뿐이다.
      새 화면의 영상·상태·기록은 멈추거나 늦춰지지 않는다. 그림은 옛 화면의 것이며 새 값이 아니다.
    [방향] 다음 화면으로 가면(forward) 옛 화면이 뒤로 물러나듯 작아지고, 이전 화면으로 가면 앞으로 다가오듯 커진다.
    [모션 줄이기] 아무 것도 하지 않는다. 화면은 그냥 바뀐다.
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._progress = 0.0
        self._forward = True
        self._generation = 0
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)   # 전환 중에도 아래 화면을 바로 누를 수 있다
        self.hide()

    def play(self, pixmap: QPixmap, forward: bool) -> None:
        if motion.is_reduced():
            return
        self._generation += 1
        generation = self._generation
        self._pixmap, self._forward, self._progress = pixmap, forward, 0.0
        self.setGeometry(self.parentWidget().rect())
        self.raise_()
        self.show()

        def step(value: float) -> None:
            if generation == self._generation:
                self._progress = float(value)
                self.update()

        def finish() -> None:
            if generation == self._generation:
                self.hide()
                self._pixmap = None

        motion.tween(self, 0.0, 1.0, theme.MS_PAGE, step, done=finish)

    def paintEvent(self, _event) -> None:  # noqa: N802
        if self._pixmap is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        scale = 1.0 + (-0.07 if self._forward else 0.07) * self._progress
        painter.setOpacity(1.0 - self._progress)
        painter.translate(self.width() / 2.0, self.height() / 2.0)
        painter.scale(scale, scale)
        painter.translate(-self.width() / 2.0, -self.height() / 2.0)
        painter.drawPixmap(self.rect(), self._pixmap)


def _angle_gap(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


# 잠식 효과의 촉수: (방향 rad, 길이, 폭 rad). 실행할 때마다 같은 모양이 되도록 난수 대신 고정 식으로 만든다
_TENDRILS = [(k * 2.0 * math.pi / 11 + 0.4 * math.sin(k * 12.9898),
              0.16 + 0.30 * abs(math.sin(k * 78.233)),
              0.05 + 0.07 * abs(math.sin(k * 37.719))) for k in range(11)]
_BLOB_STEPS = 180


@lru_cache(maxsize=2)
def _ink_arrival(width: int, height: int, ox: int, oy: int) -> QImage:
    """Pixel index = when the new theme reaches it; both colours follow the same outward flow."""
    longest = max(width, height, 1)
    cols = max(4, 4 * math.ceil(100 * width / longest))   # 긴 변 400칸. 늘린 뒤 자르므로 이 정도면 경계가 매끄럽다
    rows = max(2, round(400 * height / longest))
    reach = max(math.hypot(ox-x, oy-y) for x in (0, width) for y in (0, height)) / longest
    data = bytearray(cols * rows)
    sin = math.sin
    for row in range(rows):
        y = ((row + .5) * height / rows - oy) / longest
        for col in range(cols):
            x = ((col + .5) * width / cols - ox) / longest
            distance = math.hypot(x, y)
            # Bend the fine veins around broad lobes so the moving edge splits and rejoins.
            u = x + .073 * sin(10*y + 2*sin(9*x)) + .028 * sin(25*y - 12*x)
            v = y + .080 * sin(9*x - 5*y) + .029 * sin(22*x + 10*y)
            fold = sin(23*u + 13*v + 2.7*sin(14*v - 6*u))
            fine = sin(49*v - 11*u + 4.9*fold)
            grain = (.058*sin(10*u - 8*v) + .034*fold
                     + .023*fine + .010*sin(104*x + 71*y + 4.6*fine) + .003*sin(191*v - 74*u + 3.3*fine))
            arrival = distance + min(1, 6*distance) * grain
            data[row*cols + col] = min(255, max(0, round(255 * arrival / (reach + .12))))
    return QImage(data, cols, rows, cols, QImage.Format.Format_Grayscale8).copy()


def _ink_full(field: QImage, width: int, height: int) -> QImage:
    """저해상도 도착 시각 지도를 화면 픽셀 크기로 부드럽게 키운 뒤 색표를 바꿀 수 있는 Indexed8 로 같은 바이트를 다시 읽는다.

    경계 자르기를 키운 뒤(화면 해상도)에 하므로 계단처럼 깨지지 않는다.
    """
    # Grayscale8 를 바로 늘리면 보간되지 않아 RGB32 로 늘린 뒤 다시 회색조로 바꾼다
    full = field.convertToFormat(QImage.Format.Format_RGB32).scaled(
        width, height, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation
    ).convertToFormat(QImage.Format.Format_Grayscale8)
    return QImage(full.constBits(), full.width(), full.height(), full.bytesPerLine(), QImage.Format.Format_Indexed8).copy()


class ThemeWipe(QWidget):
    """Reveal the live theme through the old snapshot, leaving the video stage uncovered."""

    STYLES = {"ink": "번짐", "venom": "잠식", "tiles": "타일"}
    TILE = 96   # 타일 한 변(px)

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.style = "ink"
        self._pixmap: QPixmap | None = None
        self._progress = 0.0
        self._spreading = True
        self._origin = QPointF()
        self._reach = 1.0
        self._keep: QRect | None = None
        self._animation = None
        self._ink_field = self._ink_base = None
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)   # 전환 중에도 아래 화면을 바로 누를 수 있다
        self.hide()

    def stop(self) -> None:
        if self._animation is not None:
            self._animation.stop()
        self._animation = self._pixmap = self._ink_field = self._ink_base = None
        self.hide()

    def play(self, pixmap: QPixmap, origin: QPointF, spreading: bool, keep: QRect | None = None) -> None:
        """pixmap: 옛 테마의 창 그림. origin: 누른 버튼의 가운데. spreading: 다크로 갈 때 True."""
        self.stop()
        if motion.is_reduced() or pixmap.isNull():
            return
        self._pixmap, self._origin, self._spreading, self._keep, self._progress = pixmap, QPointF(origin), spreading, keep, 0.0
        self.setGeometry(self.parentWidget().rect())
        self._active_style = self.style
        if self._active_style == "ink":
            field = _ink_arrival(self.width(), self.height(), round(origin.x()), round(origin.y()))
            self._ink_field = _ink_full(field, pixmap.width(), pixmap.height())   # 스냅샷과 같은 실제 픽셀 크기
            self._ink_field.setDevicePixelRatio(pixmap.devicePixelRatio())
            self._ink_base = pixmap.toImage().convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
        # 얼룩이 다 컸을 때 가장 먼 모서리까지 덮는 반지름(여유 2%)
        self._reach = 1.02 * max(math.hypot(origin.x() - x, origin.y() - y)
                                 for x in (0, self.width()) for y in (0, self.height()))
        self.raise_()
        self.show()

        def step(value: float) -> None:
            if self.parentWidget().size() != self.size() or motion.is_reduced():
                self.stop()
                return
            self._progress = float(value)
            self.update()

        duration = theme.MS_INK if self._active_style == "ink" else theme.MS_THEME
        curve = QEasingCurve.Type.InOutSine if self._active_style == "ink" else motion.EASE_IO
        self._animation = motion.tween(self, 0.0, 1.0, duration, step, curve, self.stop)

    def _paint_ink(self, painter: QPainter) -> None:
        if self._progress <= 0:
            painter.drawPixmap(self.rect(), self._pixmap)
            return
        if self._progress >= 1:
            return
        threshold = self._progress * 256
        band = 5.0   # 경계를 도착 시각 5단계에 걸쳐 부드럽게 넘긴다(안티앨리어싱)
        mask = QImage(self._ink_field)
        mask.setColorTable([min(255, max(0, round(255 * (index - threshold) / band))) << 24 for index in range(256)])
        layer = self._ink_base.copy()
        with QPainter(layer) as blend:   # mask 는 layer 와 같은 픽셀 크기라 늘이지 않는다
            blend.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
            blend.drawImage(QRectF(self.rect()), mask.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied))
        painter.drawImage(self.rect(), layer)

    def blob(self, size: float) -> QPainterPath:
        """어두운 얼룩의 외곽. size 0 이면 점, 1 이면 창 전체를 덮는다. 중간에서만 가장자리가 흐트러진다."""
        lead = math.sin(math.pi * size)   # 처음과 끝에서 0. 그래서 시작과 끝 모양은 두 효과가 같다
        path = QPainterPath()
        path.setFillRule(Qt.FillRule.WindingFill)
        for index in range(_BLOB_STEPS):
            angle = 2.0 * math.pi * index / _BLOB_STEPS
            ahead = sum(length * math.exp(-(_angle_gap(angle, at + 0.25 * math.sin(size * 7.0 + k)) / width) ** 2)
                        for k, (at, length, width) in enumerate(_TENDRILS))
            radius = size + lead * (ahead + 0.02 * math.sin(9.0 * angle + 12.0 * size))
            point = self._origin + QPointF(math.cos(angle), math.sin(angle)) * (radius * self._reach)
            path.lineTo(point) if index else path.moveTo(point)
        path.closeSubpath()
        return path

    def paintEvent(self, _event) -> None:  # noqa: N802
        if self._pixmap is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        shown = QPainterPath()          # 덮어도 되는 영역(창에서 영상 영역을 뺀 곳)
        shown.addRect(QRectF(self.rect()))
        if self._keep is not None:
            hole = QPainterPath()
            hole.addRect(QRectF(self._keep))
            shown = shown.subtracted(hole)
        painter.setClipPath(shown)
        if self._active_style == "ink":
            self._paint_ink(painter)
            return
        if self._active_style == "tiles":
            self._paint_tiles(painter)
            return
        blob = self.blob(self._progress if self._spreading else 1.0 - self._progress)
        # 옛 그림이 남는 곳: 다크로 갈 때는 얼룩 바깥(밝은 옛 화면), 라이트로 갈 때는 얼룩 안(어두운 옛 화면)
        painter.setClipPath(shown.subtracted(blob) if self._spreading else shown.intersected(blob))
        painter.drawPixmap(self.rect(), self._pixmap)
        painter.setClipPath(shown)
        painter.strokePath(blob, QPen(QColor("#060607"), 5.0))


    def tile_turn(self, column: int, row: int, progress: float) -> float:
        """타일 하나가 넘어간 정도(0 옛 면 그대로 ~ 1 다 넘어감).

        버튼에서 왼쪽 아래 모서리로 가는 대각선 위에서 얼마나 나아간 자리인지(far 0~1)로 시작 시각을 정한다.
        """
        tile = self.TILE
        across, down = -self._origin.x(), self.height() - self._origin.y()   # 버튼 → 왼쪽 아래 모서리
        far = (((column + 0.5) * tile - self._origin.x()) * across + ((row + 0.5) * tile - self._origin.y()) * down)             / max(1.0, across * across + down * down)
        delay = 0.72 * min(1.0, max(0.0, far)) + 0.06 * ((column + row) % 2)
        return min(1.0, max(0.0, (progress - delay) / 0.22))

    def _paint_tiles(self, painter: QPainter) -> None:
        tile = self.TILE
        ratio = self._pixmap.devicePixelRatio()
        for row in range(self.height() // tile + 1):
            for column in range(self.width() // tile + 1):
                turn = self.tile_turn(column, row, self._progress)
                if turn >= 1.0:
                    continue   # 다 넘어간 타일: 아래의 새 테마가 보인다
                cell = QRectF(column * tile, row * tile, tile, tile)
                height = tile * math.cos(turn * math.pi / 2.0)   # 아랫변을 축으로 아래로 넘어간다
                target = QRectF(cell.x(), cell.bottom() - height, tile, height)
                source = QRectF(cell.x() * ratio, cell.y() * ratio, tile * ratio, tile * ratio)
                painter.drawPixmap(target, self._pixmap, source)
                painter.fillRect(target, QColor(0, 0, 0, round(110 * turn)))   # 돌아가는 면의 그늘


class KeyHint(QLabel):
    """단축키 표시: 단축키를 누르면 창 아래 가운데에 「Ctrl+2 · 운영 대시보드」가 떠올랐다 사라진다(theme.MS_HINT).

    무엇이 눌렸는지 알려 줄 뿐이다. 누를 수 없고(마우스가 통과한다) 화면 전환을 늦추지 않는다.
    모션 줄이기면 떠오름·옅어짐 없이 나타났다 사라진다.
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("KeyHint")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._generation = 0
        self._hide_timer = QTimer(self, singleShot=True, interval=theme.MS_HINT)
        self._hide_timer.timeout.connect(self._leave)
        self.hide()

    def flash(self, text: str) -> None:
        self._generation += 1
        generation = self._generation
        self.setText(text)
        self.adjustSize()
        home_x = (self.parentWidget().width() - self.width()) // 2
        home_y = self.parentWidget().height() - self.height() - 26
        effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(effect)
        self.move(home_x, home_y)
        self.raise_()
        self.show()

        def step(value: float) -> None:
            if generation == self._generation:
                effect.setOpacity(value)
                self.move(home_x, home_y + round((1.0 - value) * 10))

        motion.tween(self, 0.0, 1.0, theme.MS_POPUP, step)
        self._hide_timer.start()

    def _leave(self) -> None:
        self._generation += 1
        generation = self._generation
        effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(effect)
        motion.tween(self, 1.0, 0.0, theme.MS_POPUP, lambda value: generation == self._generation and effect.setOpacity(value),
                     done=lambda: generation == self._generation and self.hide())


# ── 상단 상태띠 ───────────────────────────────────────────────────────────

class StatusChip(QWidget):
    """상태띠 한 칸: 색 점 + 항목 이름 + 현재 문구.

    문구는 set_state 즉시 바뀐다. 색 점의 색만 MS_STAGE 동안 전환되고, 수준이 바뀌면 그 색이 MS_FLASH 동안
    왼쪽에서 오른쪽으로 한 번 쓸고 지나가며 옅어지고, 점에서 고리가 한 번 퍼진다(반복 없음).
    level: ok · warn · fail · idle · accent.
    """

    def __init__(self, caption: str) -> None:
        super().__init__()
        self._caption = caption
        self._text = "—"
        self._level = "idle"
        self._dot = QColor(theme.IDLE)
        self._flash = 0.0
        self._dot_generation = 0
        self.setFixedHeight(46)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAccessibleName(caption)

    def set_state(self, level: str, text: str) -> None:
        if level == self._level and text == self._text:
            return
        level_changed = level != self._level
        self._level, self._text = level, text
        self.setAccessibleDescription(text)
        self.setToolTip(f"{self._caption}: {text}")
        if level_changed:
            # 앞선 색 전환이 아직 돌고 있을 수 있다. 세대 번호가 다른(옛) 전환의 값은 버려 옛 색으로 덮지 않게 한다
            self._dot_generation += 1
            generation = self._dot_generation
            motion.tween(self, self._dot, QColor(theme.LEVEL_COLOR[level]), theme.MS_STAGE,
                         lambda color: generation == self._dot_generation and self._set_dot(color))
            motion.tween(self, 1.0, 0.0, theme.MS_FLASH, self._set_flash)
        self.update()

    def refresh_theme(self) -> None:
        """테마가 바뀌면 보관해 둔 점 색을 새 테마 값으로 맞춘다."""
        self._dot_generation += 1   # 돌고 있던 색 전환을 무효로 한다
        self._dot = QColor(theme.LEVEL_COLOR[self._level])
        self.update()

    def _set_dot(self, color: QColor) -> None:
        self._dot = QColor(color)
        self.update()

    def _set_flash(self, value: float) -> None:
        self._flash = float(value)
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        shape = QPainterPath()
        shape.addRoundedRect(rect, theme.RADIUS_SMALL, theme.RADIUS_SMALL)
        painter.fillPath(shape, QColor(theme.SURFACE))
        if self._flash > 0.0:   # 색이 왼쪽에서 차오르며 옅어진다. 끝나면 흔적이 남지 않는다
            wash = QColor(self._dot)
            wash.setAlphaF(0.34 * self._flash)
            painter.save()
            painter.setClipPath(shape)
            painter.fillRect(QRectF(rect.left(), rect.top(), rect.width() * (1.0 - self._flash * self._flash), rect.height()), wash)
            painter.restore()
        painter.setPen(QPen(QColor(theme.LINE), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(shape)
        center = QPointF(15, 14)
        if self._flash > 0.0:   # 점에서 고리가 퍼지며 옅어진다
            ring = QColor(self._dot)
            ring.setAlphaF(0.7 * self._flash)
            painter.setPen(QPen(ring, 1.5))
            reach = 4.0 + 7.0 * (1.0 - self._flash)
            painter.drawEllipse(center, reach, reach)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._dot)
        painter.drawEllipse(center, 4.0, 4.0)
        painter.setPen(QColor(theme.DIM))
        painter.setFont(_font(10, 700))
        painter.drawText(QRectF(26, 5, rect.width() - 32, 18), Qt.AlignmentFlag.AlignVCenter, self._caption)
        painter.setPen(QColor(theme.TEXT))
        painter.setFont(_font(13, 700))
        text = painter.fontMetrics().elidedText(self._text, Qt.TextElideMode.ElideRight, int(rect.width() - 22))
        painter.drawText(QRectF(11, 22, rect.width() - 22, 20), Qt.AlignmentFlag.AlignVCenter, text)


# ── 영상 표시면 ───────────────────────────────────────────────────────────

class DisplayView(QWidget):
    """영상 한 칸. 받은 프레임을 비율 그대로 표시한다. 보간·합성·주석 추가를 하지 않는다.

    상태(mode):
      "empty"   받을 경로가 없음(자리만 있음). 사유를 표시한다.
      "waiting" 연결 중이거나 프레임을 기다림. 대기 무늬가 지나간다.
      "live"    최근 프레임을 표시 중. 우상단 점이 천천히 숨쉰다.
      "stale"   프레임이 끊김. 마지막 프레임을 어둡게 하고 「수신 중단」과 경과 시간을 겹쳐,
                옛 장면을 현재 모습으로 오인하지 않게 한다.
    """

    # 색은 theme.DISPLAY_* 만 쓴다. 다크·라이트 테마와 무관하게 표시면은 항상 어둡다(영상 밝기를 테마가 바꾸지 않게)

    clicked = Signal()   # set_clickable(True) 인 칸을 왼쪽 클릭하거나 포커스 상태에서 Enter·Space 를 눌렀다

    def __init__(self, title: str, compact: bool = False) -> None:
        super().__init__()
        self._title = title
        self._compact = compact
        self._video_radius = 0
        self._main_frame_width: int | None = None
        self._mode = "empty"
        self._reason = ""
        self._footer = ""
        self._image: QImage | None = None
        self._arrivals: deque[float] = deque(maxlen=12)   # 실측 수신 간격으로 fps 를 계산한다
        self._stale_ms = 0
        self._phase = 0.0
        self._ticker = QTimer(self, interval=33)
        self._ticker.timeout.connect(self._advance)
        self._clickable = False
        self.setMinimumSize(160, 100)
        self.setAccessibleName(title)

    # ── 바깥에서 쓰는 것 ────────────────────────────────────────────────

    def show_frame(self, image: QImage) -> None:
        self._image = image
        self._arrivals.append(time.monotonic())
        self._mode = "live"
        self.update()

    def set_placeholder(self, mode: str, reason: str) -> None:
        """mode 는 "empty" 또는 "waiting". 보관하던 프레임을 버린다."""
        if (mode, reason) == (self._mode, self._reason):
            return
        self._mode, self._reason, self._image = mode, reason, None
        self._arrivals.clear()
        self.update()

    def set_stale(self, age_ms: int) -> None:
        if self._image is None:
            return
        self._mode, self._stale_ms = "stale", age_ms
        self._arrivals.clear()
        self.update()

    def set_clickable(self, tooltip: str | None) -> None:
        """이 칸을 누를 수 있게 한다(보조 칸 ↔ 큰 칸 교체). 키보드로도 고를 수 있다. None 이면 끈다."""
        self._clickable = tooltip is not None
        if self._clickable:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.unsetCursor()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus if self._clickable else Qt.FocusPolicy.NoFocus)
        self.setToolTip(tooltip or "")

    def set_compact(self, compact: bool) -> None:
        """작은 칸용 글자 크기·모서리를 쓸지. 큰 칸 자리로 옮기면 큰 칸처럼 그린다."""
        if compact != self._compact:
            self._compact = compact
            self.update()

    def set_video_layout(self, radius: int, main_width: int | None) -> None:
        if (radius, main_width) != (self._video_radius, self._main_frame_width):
            self._video_radius, self._main_frame_width = radius, main_width
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._clickable and event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
        else:
            super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if self._clickable and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.clicked.emit()
        else:
            super().keyPressEvent(event)

    def set_footer(self, text: str) -> None:
        if text != self._footer:
            self._footer = text
            self.update()

    def measured_fps(self) -> float | None:
        if len(self._arrivals) < 2:
            return None
        span = self._arrivals[-1] - self._arrivals[0]
        return (len(self._arrivals) - 1) / span if span > 0 else None

    # ── 모션(대기 무늬·수신 점). 보이지 않을 때는 멈춘다 ─────────────────

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._ticker.start()

    def hideEvent(self, event) -> None:  # noqa: N802
        super().hideEvent(event)
        self._ticker.stop()

    def _advance(self) -> None:
        if motion.is_reduced() or self._mode in ("empty", "stale"):
            return
        self._phase = (self._phase + 0.033 / 2.4) % 1.0   # 2.4초에 한 바퀴
        if self._mode == "live":
            self.update(QRect(self.width() - 190, 0, 190, 48))   # 우상단 상태 표시만 다시 그린다
        else:
            self.update()

    # ── 그리기 ──────────────────────────────────────────────────────────

    def _frame_rect(self, image_size: QSize) -> QRect:
        # 출처 줄은 영상 밖에 둔다. Gazebo 창 아래쪽 재생 버튼 등을 가리지 않게.
        area = self.rect().adjusted(0, 0, 0, -36 if self._footer and not self._compact else 0)
        available = area.size()
        if self._main_frame_width is not None:
            available.setWidth(min(available.width(), self._main_frame_width))
        fitted = image_size.scaled(available, Qt.AspectRatioMode.KeepAspectRatio)
        target = QRect(0, 0, fitted.width(), fitted.height())
        target.moveCenter(area.center())
        if self._main_frame_width is not None:
            rightmost_left = area.left() + available.width() - target.width()
            target.moveLeft(max(area.left(), min(target.left() - 150, rightmost_left)))
        return target

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = (self._video_radius or theme.RADIUS_SMALL) if self._compact else theme.RADIUS
        clip = QPainterPath()
        clip.addRoundedRect(rect, radius, radius)
        painter.setClipPath(clip)
        painter.fillPath(clip, QColor(theme.DISPLAY_BG))

        if self._image is not None:
            target = self._frame_rect(self._image.size())
            painter.save()
            if self._video_radius:
                frame_clip = QPainterPath()
                frame_clip.addRoundedRect(QRectF(target), self._video_radius, self._video_radius)
                painter.setClipPath(frame_clip, Qt.ClipOperation.IntersectClip)
            painter.drawImage(target, self._image)
            painter.restore()
            if self._mode == "stale":
                painter.fillRect(self.rect(), QColor(6, 6, 7, 190))
                self._draw_center(painter, "수신 중단", f"마지막 프레임 {self._stale_ms / 1000:.1f}초 전 · 현재 모습 아님",
                                  target if self._main_frame_width is not None else None)
        else:
            if self._mode == "waiting" and not motion.is_reduced():
                self._draw_sweep(painter)
            self._draw_center(painter, "수신 대기" if self._mode == "waiting" else "미수신", self._reason)

        self._draw_badges(painter)
        painter.setClipping(False)
        painter.setPen(QPen(QColor(theme.DISPLAY_LINE), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect, radius, radius)

    def _draw_sweep(self, painter: QPainter) -> None:
        """대기 무늬: 옅은 라임 빛 띠가 왼쪽에서 오른쪽으로 지나간다. 영상이 아니라 「기다리는 중」 표시다."""
        width = self.width()
        center = -0.3 * width + self._phase * 1.6 * width
        gradient = QLinearGradient(center - 0.25 * width, 0, center + 0.25 * width, 0)
        glow = QColor(theme.DISPLAY_ACCENT)
        glow.setAlpha(26)
        gradient.setColorAt(0.0, QColor(0, 0, 0, 0))
        gradient.setColorAt(0.5, glow)
        gradient.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.fillRect(self.rect(), gradient)

    def _draw_center(self, painter: QPainter, head: str, detail: str, bounds: QRect | None = None) -> None:
        if bounds is None:
            bounds = self.rect()
            if self._main_frame_width is not None:
                bounds.setWidth(min(bounds.width(), self._main_frame_width))
        box = QRectF(bounds).adjusted(16, 0, -16, 0)
        painter.setPen(QColor(theme.DISPLAY_TEXT))
        painter.setFont(_font(13 if self._compact else 20, 800))
        painter.drawText(box.adjusted(0, 0, 0, -22), Qt.AlignmentFlag.AlignCenter, head)
        painter.setPen(QColor(theme.DISPLAY_DIM))
        painter.setFont(_font(11 if self._compact else 13, 500))
        painter.drawText(box.adjusted(0, 26 if self._compact else 34, 0, 0),
                         Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, detail)

    def _draw_badges(self, painter: QPainter) -> None:
        # 좌상단: 칸 이름
        painter.setFont(_font(11 if self._compact else 12, 700))
        metrics = painter.fontMetrics()
        name = QRectF(12, 12, metrics.horizontalAdvance(self._title) + 20, 24)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(10, 10, 11, 200))
        painter.drawRoundedRect(name, 12, 12)
        painter.setPen(QColor(theme.DISPLAY_TEXT))
        painter.drawText(name, Qt.AlignmentFlag.AlignCenter, self._title)

        # 우상단: 수신 상태. 문구가 항상 함께 있다(색만으로 구분하지 않음)
        if self._mode == "live":
            fps = self.measured_fps()
            text = "LIVE" + (f" · {fps:.1f} fps" if fps is not None else "")
            color = QColor(theme.DISPLAY_ACCENT)
            alpha = 255 if motion.is_reduced() else int(150 + 105 * (0.5 + 0.5 * math.sin(self._phase * 2 * math.pi)))
        else:
            text = {"waiting": "대기", "stale": "중단", "empty": "자리"}[self._mode]
            color = QColor({"waiting": theme.DISPLAY_WARN, "stale": theme.DISPLAY_FAIL, "empty": theme.DISPLAY_IDLE}[self._mode])
            alpha = 255
        width = metrics.horizontalAdvance(text) + 34
        state = QRectF(self.width() - width - 12, 12, width, 24)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(10, 10, 11, 200))
        painter.drawRoundedRect(state, 12, 12)
        dot = QColor(color)
        dot.setAlpha(alpha)
        painter.setBrush(dot)
        painter.drawEllipse(QPointF(state.left() + 13, state.center().y()), 4, 4)
        painter.setPen(QColor(theme.DISPLAY_TEXT))
        painter.drawText(state.adjusted(22, 0, -8, 0), Qt.AlignmentFlag.AlignVCenter, text)

        # 좌하단: 출처 한 줄(카메라 이름·해상도·프레임 수 등 실제 값)
        if self._footer and not self._compact:
            painter.setFont(_font(11, 500, mono=True))
            painter.setPen(QColor(theme.DISPLAY_DIM))
            painter.drawText(QRectF(16, self.height() - 30, self.width() - 32, 18), Qt.AlignmentFlag.AlignVCenter, self._footer)


class DetailPopup(Card):
    """메인 디스플레이 위에 겹쳐 뜨는 측정·보정 상세(골격 문서 「위 팝업」).

    [A·B 예정] px · mm · θ 와 보정 상태는 A·B 결과가 C 표시 연결로 와야 채워진다. 지금은 모두 「—」.
    열릴 때 MS_POPUP 동안 올라오며 나타나고, 닫힐 때 같은 시간 동안 사라진다.
    정지 버튼과 상태띠는 창 위쪽에 따로 있어 이 팝업에 가려지지 않는다.
    """

    def __init__(self, parent: QWidget, demo: bool = False) -> None:
        super().__init__("측정·보정 상세", "D 모의 · 실제 아님" if demo else "A·B 결과 미수신")
        self.setParent(parent)
        self.refresh_theme()
        self.values: dict[str, QLabel] = {}
        for key, name in (("px", "검출 중심 (u, v) px"), ("mm", "작업 좌표 (X, Y) mm"), ("theta", "장축각 θ °"),
                          ("calibration", "보정 상태"), ("frame", "소스 · 프레임"),
                          ("detail", "상태 · 규격")):
            row = QHBoxLayout()
            row.addWidget(label(name, "Dim"))
            row.addStretch(1)
            self.values[key] = label("—", "CardTitle")
            row.addWidget(self.values[key])
            self.body.addLayout(row)
        self.body.addWidget(label("일치하는 유효 결과만 표시합니다. D 모의 값은 실제 측정이 아닙니다. C 표시 연결은 미결입니다.", "Caption", wrap=True))
        self.hide()

    def refresh_theme(self) -> None:
        """카드 바탕을 현재 테마 색의 반투명으로 다시 정한다(영상 위에 겹치므로 살짝 비친다)."""
        surface = QColor(theme.SURFACE)
        self.setStyleSheet(f"#Card {{ background: rgba({surface.red()}, {surface.green()}, {surface.blue()}, 238); "
                           f"border: 1px solid {theme.LINE}; }}")

    def toggle(self, visible: bool) -> None:
        if visible:
            self.show()
            self.raise_()
            motion.appear(self, theme.MS_POPUP, rise_px=14)
        elif motion.is_reduced():
            self.hide()
        else:
            from PySide6.QtWidgets import QGraphicsOpacityEffect
            effect = QGraphicsOpacityEffect(self)
            self.setGraphicsEffect(effect)

            def finish() -> None:
                self.hide()
                self.setGraphicsEffect(None)

            motion.tween(self, 1.0, 0.0, theme.MS_POPUP, effect.setOpacity, done=finish)


class DisplayStage(QWidget):
    """탑뷰(큰 바탕) + 그 위 오른쪽 가장자리에 붙은 보조 영상 2칸(그리퍼·링크) + 상세 팝업.

    배치(골격 문서 2026-11-13 사용자 화면 방향):
      탑뷰가 큰 바탕이고, 보조 2칸은 그 안쪽 오른쪽에 세로로 고정, 팝업은 가운데~왼쪽.
      운영 탭에서는 큰 프레임을 왼쪽에 놓고 오른쪽 보조 2칸이 차지할 공간을 확보한다.
    교체(사용자 지정 2026-11-16): 보조 칸을 누르면 그 영상이 큰 칸 자리로 가고, 큰 칸에 있던 영상이 누른 보조 칸
      자리로 간다. 큰 칸을 가리는 겹친 확대는 없다. 위젯 자체를 자리만 바꾸므로 영상 수신·연결과
      main·gripper·link(영상 출처별 위젯)는 그대로다.
    [이름] 화면 이름 「탑뷰·그리퍼·링크」는 표시 이름이다. 탑뷰 칸에 지금 오는 것은 D 임시 경로
      (Gazebo 창 캡처 또는 카메라 중계)이며, 정식 영상 3개는 C 표시 연결이 정해진 뒤에 온다.
    """

    def __init__(self, demo: bool = False) -> None:
        super().__init__()
        self._operating_layout = False
        self.main = DisplayView("탑뷰")
        self.main.setParent(self)
        self.gripper = DisplayView("그리퍼", compact=True)
        self.link = DisplayView("링크", compact=True)
        self.slots: list[DisplayView] = [self.main, self.gripper, self.link]   # [큰 칸, 보조 위, 보조 아래]
        for view in (self.gripper, self.link):
            view.setParent(self)
            # [C 예정] 그리퍼·링크 영상은 C 표시 연결로만 온다. D 임시 경로에는 없다
            view.set_placeholder("empty", "C 표시 연결 미결")
        for view in self.slots:
            view.clicked.connect(lambda v=view: self.swap_to_main(v))
        self._apply_clickable()
        self.popup = DetailPopup(self, demo=demo)
        self.setMinimumHeight(max(380, 72 + self.popup.sizeHint().height()))

    def swap_to_main(self, view: DisplayView) -> None:
        """보조 칸의 view 를 큰 칸으로, 큰 칸에 있던 영상을 view 의 보조 칸 자리로 바꾼다. 큰 칸이면 무시."""
        index = self.slots.index(view)
        if index == 0:
            return
        previous = self.slots[0]
        self.slots[0], self.slots[index] = view, previous
        refocus = view.hasFocus()
        self._apply_clickable()
        self._place(animate=True)
        if refocus:   # 키보드로 바꿨으면 같은 보조 칸 자리(이제 이전 큰 영상)에 초점을 둔다
            previous.setFocus(Qt.FocusReason.OtherFocusReason)

    def _apply_clickable(self) -> None:
        self.slots[0].set_clickable(None)
        for view in self.slots[1:]:
            view.set_clickable("누르면 이 화면과 큰 화면의 자리를 바꿉니다.")

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._place(animate=False)

    def set_operating_layout(self, enabled: bool) -> None:
        if enabled != self._operating_layout:
            self._operating_layout = enabled
            self._place(animate=False)

    def _place(self, animate: bool) -> None:
        width, height = self.width(), self.height()
        if self._operating_layout:
            # 위쪽 52px, 두 칸 사이 10px, 출처 줄 36px, 아래 여유 8px.
            max_aux_height = (height - 106) // 2
            aux_width = max(170, min(336, width * 24 // 100, max_aux_height * 16 // 9))
            aux_height = max(100, aux_width * 9 // 16)
        else:
            aux_width = max(170, min(300, int(width * 0.21)))
            aux_height = min(int(aux_width * 9 / 16), (height - 86) // 2)
        left = width - aux_width - 16
        targets = [QRect(0, 0, width, height), QRect(left, 52, aux_width, aux_height),
                   QRect(left, 52 + aux_height + 10, aux_width, aux_height)]
        self.slots[0].lower()   # 큰 칸은 맨 아래, 보조 칸과 팝업은 그 위
        for index, (view, target) in enumerate(zip(self.slots, targets)):
            view.set_compact(index > 0)
            main_width = max(1, left - 16) if self._operating_layout and index == 0 else None
            view.set_video_layout(16 if self._operating_layout else 0, main_width)
            if animate:
                motion.move_geometry(view, target, theme.MS_POPUP)
            else:
                tween = view.findChild(QPropertyAnimation, "geometry_motion")
                if tween is not None:   # 교체 움직임 중 크기가 바뀌면 옛 목표로 가던 움직임을 멈춘다
                    tween.stop()
                    tween.setObjectName("")
                    tween.deleteLater()
                view.setGeometry(target)
            if index > 0:
                view.raise_()
        popup_width = min(380, max(280, width - 48))
        self.popup.setGeometry(24, 56, popup_width, self.popup.sizeHint().height())
        self.popup.raise_()


# ── 공정 단계 ─────────────────────────────────────────────────────────────

class StageStrip(QWidget):
    """공정 단계 표시. [C 예정] C가 확인해 알려 준 단계만 강조한다.

    2일차에는 단계를 알려 주는 계약이 없으므로 전부 「미확인」이다. 명령을 보냈다는 사실만으로
    단계를 진행시키지 않는다. set_stage(None) = 전부 미확인.
    """

    def __init__(self, stages: list[str]) -> None:
        super().__init__()
        self._stages = stages
        self._current: int | None = None
        self._glow = 0.0
        self.setFixedHeight(34)

    def set_stage(self, index: int | None) -> None:
        if index == self._current:
            return
        self._current = index
        motion.tween(self, 0.0, 1.0, theme.MS_STAGE, self._set_glow)   # 색이 차오른다(한 번)

    def _set_glow(self, value: float) -> None:
        self._glow = float(value)
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(_font(11, 700))
        gap = 5
        width = (self.width() - gap * (len(self._stages) - 1)) / len(self._stages)
        for index, name in enumerate(self._stages):
            box = QRectF(index * (width + gap), 3, width, 28)
            active = index == self._current
            shape = QPainterPath()
            shape.addRoundedRect(box, 14, 14)
            painter.fillPath(shape, QColor(theme.SURFACE_2))
            if active:   # 주황이 왼쪽부터 차오른다. 어느 칸이 확인됐는지는 그대로 한 칸뿐이다
                painter.save()
                painter.setClipPath(shape)
                painter.fillRect(QRectF(box.left(), box.top(), box.width() * self._glow, box.height()),
                                 QColor(theme.ACCENT))
                painter.restore()
            painter.setPen(QPen(QColor(theme.LINE), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(shape)
            painter.setPen(QColor(theme.ON_ACCENT if active and self._glow > 0.5 else theme.DIM))
            painter.drawText(box, Qt.AlignmentFlag.AlignCenter, name)


# ── TX/RX 기록 ────────────────────────────────────────────────────────────

class TrafficTable(QTableWidget):
    """송수신 기록 표(GUI-03). 최신이 맨 위. 실제로 보내고 받은 줄만 적는다.

    [C 값] RX 는 C 서버 응답 원문 요약이다. 원문 줄은 행에 마우스를 올리면 보인다.
    화면에서 지워도 세션 JSONL 파일은 지우지 않는다. CSV는 선택된 화면의 요약 기록을 내보낸다.
    """

    MAX_ROWS = 500

    @staticmethod
    def _direction_color(direction: str) -> QColor:
        """방향 글자색. 그릴 때마다 현재 테마에서 읽는다."""
        return QColor({"TX": theme.ACCENT_INK, "RX": theme.OK, "EV": theme.DIM}.get(direction, theme.TEXT))

    def __init__(self) -> None:
        super().__init__(0, 4)
        self.setHorizontalHeaderLabels(["시각", "방향", "종류", "내용"])
        self.verticalHeader().hide()
        self.setShowGrid(False)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setTabKeyNavigation(False)   # Tab 이 표 칸 사이에 갇히지 않고 다음 컨트롤로 간다. 행 이동은 방향키
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self._filter = ""

    def add_entry(self, entry) -> None:
        """entry 는 client.TrafficEntry."""
        self.insertRow(0)
        for column, text in enumerate((entry.wall_time, entry.direction, entry.kind, entry.summary)):
            item = QTableWidgetItem(text)
            if entry.raw:
                item.setToolTip(entry.raw)
            if column == 1:
                item.setForeground(self._direction_color(entry.direction))
            self.setItem(0, column, item)
        self.setRowHidden(0, bool(self._filter) and entry.direction != self._filter)
        if self.rowCount() > self.MAX_ROWS:
            self.removeRow(self.rowCount() - 1)
        self._flash_row([self.item(0, column) for column in range(4)])

    def _flash_row(self, items: list) -> None:
        """새로 들어온 줄의 바탕이 한 번 밝아졌다 사라진다(반복 없음). 글자와 값은 처음부터 최종 그대로다."""
        tint = QColor(self._direction_color(items[1].text()))

        def paint(value: float) -> None:
            color = QColor(tint)
            color.setAlphaF(0.20 * float(value))
            try:
                for item in items:
                    item.setBackground(color)
            except RuntimeError:   # 강조가 끝나기 전에 그 줄이 표에서 빠졌다(화면 비우기·500줄 초과)
                pass

        motion.tween(self, 1.0, 0.0, theme.MS_FLASH, paint)

    def refresh_theme(self) -> None:
        """이미 넣은 행의 방향 글자색을 새 테마 값으로 바꾼다."""
        for row in range(self.rowCount()):
            self.item(row, 1).setForeground(self._direction_color(self.item(row, 1).text()))

    def set_filter(self, direction: str) -> None:
        """direction: ""(전체) · "TX" · "RX" · "EV"."""
        self._filter = direction
        for row in range(self.rowCount()):
            self.setRowHidden(row, bool(direction) and self.item(row, 1).text() != direction)


class RingProgress(QWidget):
    """처리한 CSV 행 수의 비율. 실제 TCP/로봇 진행률과는 별개다."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.value = 0
        self.setFixedSize(52, 52)
        self.setAccessibleName("CSV 생성 진행률")

    def set_value(self, value: int) -> None:
        self.value = max(0, min(100, value))
        self.setAccessibleDescription(f"{self.value}%")
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(5, 5, 42, 42)
        painter.setPen(QPen(QColor(theme.LINE), 4))
        painter.drawEllipse(rect)
        pen = QPen(QColor(theme.ACCENT_INK), 4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(rect, 90 * 16, -int(self.value * 360 * 16 / 100))
        painter.setPen(QColor(theme.TEXT))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, f"{self.value}%")
