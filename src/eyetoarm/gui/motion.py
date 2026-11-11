"""화면 부품의 움직임. 들어올 때 ease-out, 나갈 때 ease-in, 0.2~0.4초.

스낵바: 아래에서 slide · 토스트: 제자리 fade · 팝업: 배경 어둡게 + 90% → 100% · 바텀시트: 아래에서 slide + 배경 fade
체크박스: 체크 선이 그려짐(ease in-out) · 로딩: 일정한 속도로 회전(linear) · 제목·부제: 차례로 등장 · 표시 버튼: 톡 튕김
실시간 셀 그림은 여기 것을 쓰지 않는다(받은 값을 직선 보간만 한다).
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import (
    Property, QAbstractAnimation, QEasingCurve, QParallelAnimationGroup, QPoint, QPropertyAnimation, QRect, QTimer, Qt, Signal,
)
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .theme import ACC, CARD, LINE2, MS_ENTER, MS_FAST, MS_SHEET, T1

EASE_OUT, EASE_IN, EASE_IO = QEasingCurve.Type.OutCubic, QEasingCurve.Type.InCubic, QEasingCurve.Type.InOutCubic
DELETE = QAbstractAnimation.DeletionPolicy.DeleteWhenStopped


def _animate(target, prop: bytes, start, end, ms: int, curve, done: Callable[[], None] | None = None) -> QPropertyAnimation:
    animation = QPropertyAnimation(target, prop, target)
    animation.setDuration(ms)
    animation.setStartValue(start)
    animation.setEndValue(end)
    animation.setEasingCurve(curve)
    if done is not None:
        animation.finished.connect(done)
    animation.start(DELETE)
    return animation


def fade(widget: QWidget, start: float, end: float, ms: int, curve, done: Callable[[], None] | None = None) -> None:
    effect = widget.graphicsEffect()
    if not isinstance(effect, QGraphicsOpacityEffect):
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
    effect.setOpacity(start)
    _animate(effect, b"opacity", start, end, ms, curve, done)


def stagger(widgets: list[QWidget], step_ms: int = 80) -> None:
    """제목 → 부제 순서로 한 줄씩 등장."""
    for index, widget in enumerate(widgets):
        effect = QGraphicsOpacityEffect(widget)
        effect.setOpacity(0.0)
        widget.setGraphicsEffect(effect)
        QTimer.singleShot(index * step_ms, lambda w=widget: fade(w, 0.0, 1.0, MS_ENTER, EASE_OUT))


def spring(widget: QWidget) -> None:
    """눌렀을 때 톡 튕긴다(60% → 100%, 살짝 넘쳤다 돌아옴)."""
    end = widget.geometry()
    start = QRect(0, 0, int(end.width() * .6), int(end.height() * .6))
    start.moveCenter(end.center())
    _animate(widget, b"geometry", start, end, 380, QEasingCurve.Type.OutBack)


class Spinner(QWidget):
    """로딩: 일정한 속도로 계속 회전."""

    def __init__(self, size: int = 14, color: str = T1, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._angle, self._color = 0, color
        self._timer = QTimer(self, interval=16)
        self._timer.timeout.connect(self._tick)

    def _tick(self) -> None:
        self._angle = (self._angle + 7) % 360  # 0.8초에 한 바퀴, 등속
        self.update()

    def showEvent(self, _event) -> None:  # noqa: N802
        self._timer.start()

    def hideEvent(self, _event) -> None:  # noqa: N802
        self._timer.stop()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(2, 2, -2, -2)
        painter.setPen(QPen(QColor(LINE2), 2))
        painter.drawEllipse(rect)
        painter.setPen(QPen(QColor(self._color), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawArc(rect, -self._angle * 16, 90 * 16)


class DrawCheck(QWidget):
    """체크박스: 켜면 체크 표시의 선이 그려진다."""

    toggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(18, 18)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.checked, self._progress = False, 0.0

    def _get(self) -> float:
        return self._progress

    def _set(self, value: float) -> None:
        self._progress = value
        self.update()

    progress = Property(float, _get, _set)

    def set_checked(self, on: bool) -> None:
        if on == self.checked:
            return
        self.checked = on
        _animate(self, b"progress", self._progress, 1.0 if on else 0.0, 250, EASE_IO)
        self.toggled.emit(on)

    def mousePressEvent(self, _event) -> None:  # noqa: N802
        self.set_checked(not self.checked)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return):
            self.set_checked(not self.checked)

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        fill = QColor(T1) if self._progress > 0 else QColor(CARD)
        painter.setPen(QPen(QColor(T1 if self._progress > 0 else LINE2), 1))
        painter.setBrush(fill)
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 5, 5)
        if self._progress > 0:
            path = QPainterPath()
            path.moveTo(4.5, 9.5)
            path.lineTo(7.5, 12.5)
            path.lineTo(13.5, 5.5)
            pen = QPen(QColor("#ffffff"), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            length = path.length()
            pen.setDashPattern([length * self._progress / 2, 1000])  # 선 길이만큼만 그린다
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)


class Overlays:
    """창 위에 겹치는 부품. host 는 창 전체를 덮는 부모 위젯."""

    def __init__(self, host: QWidget) -> None:
        self.host = host
        self._toast: QWidget | None = None
        self._snack: QWidget | None = None
        self._modal: list[QWidget] = []

    # 토스트: 제자리 fade in, fade out
    def toast(self, text: str) -> None:
        if self._toast is not None:
            self._toast.deleteLater()
        frame = QFrame(self.host, objectName="toast")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(16, 9, 16, 9)
        dot = QLabel("●")
        dot.setStyleSheet(f"color: {ACC}; font-size: 9px;")
        layout.addWidget(dot)
        layout.addWidget(QLabel(text))
        frame.adjustSize()
        frame.move((self.host.width() - frame.width()) // 2, 10)
        frame.show()
        frame.raise_()
        self._toast = frame
        fade(frame, 0.0, 1.0, MS_FAST, EASE_OUT)
        QTimer.singleShot(2400, lambda f=frame: self._leave_fade(f))

    def _leave_fade(self, frame: QWidget) -> None:
        if frame is self._toast:
            self._toast = None
        try:
            fade(frame, 1.0, 0.0, MS_FAST, EASE_IN, frame.deleteLater)
        except RuntimeError:  # 이미 지워짐
            pass

    # 스낵바: 화면 아래에서 slide in(ease-out), 나갈 때 ease-in
    def snack(self, text: str, action: str = "", run: Callable[[], None] | None = None) -> None:
        if self._snack is not None:
            self._snack.deleteLater()
        frame = QFrame(self.host, objectName="snack")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(16, 8, 8, 8)
        layout.setSpacing(16)
        layout.addWidget(QLabel(text))
        if action:
            button = QPushButton(action)
            button.clicked.connect(lambda: (run() if run else None, self._leave_snack(frame)))
            layout.addWidget(button)
        frame.adjustSize()
        x = (self.host.width() - frame.width()) // 2
        end = QPoint(x, self.host.height() - frame.height() - 24)
        frame.move(x, self.host.height() + 8)
        frame.show()
        frame.raise_()
        self._snack = frame
        _animate(frame, b"pos", frame.pos(), end, MS_ENTER, EASE_OUT)
        QTimer.singleShot(5000, lambda f=frame: self._leave_snack(f))

    def _leave_snack(self, frame: QWidget) -> None:
        if frame is not self._snack:
            return
        self._snack = None
        _animate(frame, b"pos", frame.pos(), QPoint(frame.x(), self.host.height() + 8), 250, EASE_IN, frame.deleteLater)

    def _dim(self, close: Callable[[], None]) -> QWidget:
        dim = QWidget(self.host, objectName="dim")
        dim.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        dim.setGeometry(self.host.rect())
        dim.mousePressEvent = lambda _e: close()  # 배경을 누르면 닫는다
        dim.show()
        dim.raise_()
        fade(dim, 0.0, 1.0, MS_ENTER, EASE_OUT)
        return dim

    # 팝업: 뒤 배경을 어둡게 하고 90% → 100% scale in
    def popup(self, title: str, body: str | QWidget, ok_text: str = "", on_ok: Callable[[], None] | None = None,
              width: int = 420) -> None:
        self.close_modal()
        dim = self._dim(self.close_modal)
        frame = QFrame(self.host, objectName="popup")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(24, 24, 24, 20)
        layout.setSpacing(10)
        layout.addWidget(QLabel(title, objectName="h1"))
        if isinstance(body, QWidget):
            layout.addWidget(body)
        else:
            label = QLabel(body, objectName="sub", wordWrap=True)
            layout.addWidget(label)
        row = QHBoxLayout()
        row.addStretch(1)
        cancel = QPushButton("닫기" if not ok_text else "취소")
        cancel.clicked.connect(self.close_modal)
        row.addWidget(cancel)
        if ok_text:
            ok = QPushButton(ok_text, objectName="primary")
            ok.clicked.connect(lambda: (self.close_modal(), on_ok() if on_ok else None))
            row.addWidget(ok)
        layout.addLayout(row)
        frame.setFixedWidth(width)
        frame.adjustSize()
        end = QRect(0, 0, width, frame.sizeHint().height())
        end.moveCenter(self.host.rect().center())
        start = QRect(0, 0, int(end.width() * .9), int(end.height() * .9))
        start.moveCenter(end.center())
        frame.setMinimumSize(0, 0)
        frame.setMaximumSize(16777215, 16777215)
        frame.setGeometry(start)
        frame.show()
        frame.raise_()
        self._modal = [dim, frame]
        group = QParallelAnimationGroup(frame)
        scale = QPropertyAnimation(frame, b"geometry")
        scale.setDuration(250)
        scale.setStartValue(start)
        scale.setEndValue(end)
        scale.setEasingCurve(EASE_OUT)
        group.addAnimation(scale)
        group.start(DELETE)
        (cancel if not ok_text else ok).setFocus()

    # 바텀시트: 화면 아래에서 slide in, 뒤 배경은 fade 로 어둡게
    def sheet(self, content: QWidget, height: int = 380) -> None:
        self.close_modal()
        dim = self._dim(self.close_modal)
        frame = QFrame(self.host, objectName="sheet")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(24, 10, 24, 20)
        grab = QFrame()
        grab.setFixedSize(36, 4)
        grab.setStyleSheet(f"background: {LINE2}; border-radius: 2px;")
        layout.addWidget(grab, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(content)
        width = self.host.width() - 48
        end = QRect(24, self.host.height() - height, width, height)
        frame.setGeometry(QRect(24, self.host.height(), width, height))
        frame.show()
        frame.raise_()
        self._modal = [dim, frame]
        _animate(frame, b"geometry", frame.geometry(), end, MS_SHEET, EASE_OUT)

    def close_modal(self) -> None:
        widgets, self._modal = self._modal, []
        for widget in widgets:
            if widget.objectName() == "sheet":
                end = QRect(widget.x(), self.host.height(), widget.width(), widget.height())
                _animate(widget, b"geometry", widget.geometry(), end, 250, EASE_IN, widget.deleteLater)
            else:
                fade(widget, 1.0, 0.0, MS_FAST, EASE_IN, widget.deleteLater)

    @property
    def modal_open(self) -> bool:
        return bool(self._modal)

    def relayout(self) -> None:
        """창 크기가 바뀌면 어두운 배경을 다시 맞춘다."""
        for widget in self._modal:
            if widget.objectName() == "dim":
                widget.setGeometry(self.host.rect())
