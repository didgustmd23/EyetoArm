"""D 운영 GUI 화면 부품. 통신을 모른다. 값을 받아 그리기만 한다.

┌─ 부품과 연동 파트 ─────────────────────────────────────────────────────────┐
│ NavRail       좌측 화면 선택(조작·설정 / 운영 대시보드 / 평가·기록)   [D]    │
│ StatusChip    상단 상태띠 한 칸                                      [C 값]  │
│ DisplayView   영상 표시면. 메인 = Gazebo 로봇 화면                   [D 임시]│
│               보조 2칸 = 그리퍼·고정 영상 자리                       [C 예정]│
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

from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QFrame, QHBoxLayout, QHeaderView, QLabel, QPushButton, QSizePolicy, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from eyetoarm.gui import motion, theme


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


def pill(text: str, role: str = "", tooltip: str = "", checkable: bool = False) -> QPushButton:
    """알약 버튼. role: ""(기본) · "primary"(강조) · "stop"(정지 계열)."""
    button = QPushButton(text)
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
        self.body.setContentsMargins(20, 18, 20, 18)
        self.body.setSpacing(10)
        if title:
            head = QHBoxLayout()
            head.addWidget(label(title, "CardTitle"))
            head.addStretch(1)
            if tag:
                head.addWidget(label(tag, "Tag"))
            self.body.addLayout(head)


# ── 좌측 화면 선택 ────────────────────────────────────────────────────────

class NavRail(QFrame):
    """화면 선택. 라임색 선택 표시가 누른 항목으로 미끄러져 간다(theme.MS_NAV)."""

    selected = Signal(int)

    def __init__(self, items: list[str]) -> None:
        super().__init__()
        self.setObjectName("NavRail")
        self.setFixedWidth(232)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 26, 20, 20)
        layout.setSpacing(6)
        layout.addWidget(label("Eye To Arms", "Brand"))
        layout.addSpacing(22)

        self._list = QWidget()
        rows = QVBoxLayout(self._list)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(6)
        self._indicator = QFrame(self._list)
        self._indicator.setObjectName("NavIndicator")
        self._group = QButtonGroup(self)
        self._buttons: list[QPushButton] = []
        for index, text in enumerate(items):
            button = QPushButton(text)
            button.setObjectName("NavButton")
            button.setCheckable(True)
            button.setFixedHeight(46)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setShortcut(f"Ctrl+{index + 1}")
            button.setToolTip(f"Ctrl+{index + 1}")
            self._group.addButton(button, index)
            self._buttons.append(button)
            rows.addWidget(button)
        self._indicator.lower()
        self._group.idClicked.connect(self._on_clicked)
        layout.addWidget(self._list)
        layout.addStretch(1)
        self.footer = QVBoxLayout()   # 창이 출처 표시·모션 줄이기 체크를 넣는다
        self.footer.setSpacing(10)
        layout.addLayout(self.footer)

    def select(self, index: int, animate: bool = True) -> None:
        self._buttons[index].setChecked(True)
        self._buttons[index].setFocus()   # 키보드 초점도 선택한 항목으로 옮긴다
        target = self._buttons[index].geometry()
        if animate:
            motion.move_geometry(self._indicator, target, theme.MS_NAV)
        else:
            self._indicator.setGeometry(target)

    def _on_clicked(self, index: int) -> None:
        self.select(index)
        self.selected.emit(index)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt 이름
        super().resizeEvent(event)
        QTimer.singleShot(0, self._snap)   # 레이아웃이 버튼 자리를 정한 뒤 선택 표시를 맞춘다

    def _snap(self) -> None:
        index = self._group.checkedId()
        if index >= 0:
            self._indicator.setGeometry(self._buttons[index].geometry())


# ── 상단 상태띠 ───────────────────────────────────────────────────────────

class StatusChip(QWidget):
    """상태띠 한 칸: 색 점 + 항목 이름 + 현재 문구.

    문구는 set_state 즉시 바뀐다. 색 점의 색만 MS_STAGE 동안 전환되고, 바탕이 MS_FLASH 동안
    한 번 밝아졌다 돌아온다(반복 없음). level: ok · warn · fail · idle · accent.
    """

    def __init__(self, caption: str) -> None:
        super().__init__()
        self._caption = caption
        self._text = "—"
        self._level = "idle"
        self._dot = QColor(theme.IDLE)
        self._flash = 0.0
        self._dot_generation = 0
        self.setFixedHeight(58)
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
        background = motion.blend(QColor(theme.SURFACE), self._dot, 0.22 * self._flash)
        painter.setPen(QPen(QColor(theme.LINE), 1))
        painter.setBrush(background)
        painter.drawRoundedRect(rect, theme.RADIUS_SMALL, theme.RADIUS_SMALL)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._dot)
        painter.drawEllipse(QPointF(18, 20), 4.5, 4.5)
        painter.setPen(QColor(theme.DIM))
        painter.setFont(_font(11, 700))
        painter.drawText(QRectF(30, 10, rect.width() - 38, 20), Qt.AlignmentFlag.AlignVCenter, self._caption)
        painter.setPen(QColor(theme.TEXT))
        painter.setFont(_font(14, 700))
        text = painter.fontMetrics().elidedText(self._text, Qt.TextElideMode.ElideRight, int(rect.width() - 24))
        painter.drawText(QRectF(14, 30, rect.width() - 24, 22), Qt.AlignmentFlag.AlignVCenter, text)


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

    def __init__(self, title: str, compact: bool = False) -> None:
        super().__init__()
        self._title = title
        self._compact = compact
        self._mode = "empty"
        self._reason = ""
        self._footer = ""
        self._image: QImage | None = None
        self._arrivals: deque[float] = deque(maxlen=12)   # 실측 수신 간격으로 fps 를 계산한다
        self._stale_ms = 0
        self._phase = 0.0
        self._ticker = QTimer(self, interval=33)
        self._ticker.timeout.connect(self._advance)
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

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = theme.RADIUS_SMALL if self._compact else theme.RADIUS
        clip = QPainterPath()
        clip.addRoundedRect(rect, radius, radius)
        painter.setClipPath(clip)
        painter.fillPath(clip, QColor(theme.DISPLAY_BG))

        if self._image is not None:
            # 아래 출처 줄(36px)은 영상 밖에 둔다. Gazebo 창 아래쪽 재생 버튼 등을 가리지 않게
            area = self.rect().adjusted(0, 0, 0, -36 if self._footer and not self._compact else 0)
            scaled = QSize(self._image.size()).scaled(area.size(), Qt.AspectRatioMode.KeepAspectRatio)
            target = QRect(0, 0, scaled.width(), scaled.height())
            target.moveCenter(area.center())
            painter.drawImage(target, self._image)
            if self._mode == "stale":
                painter.fillRect(self.rect(), QColor(6, 6, 7, 190))
                self._draw_center(painter, "수신 중단", f"마지막 프레임 {self._stale_ms / 1000:.1f}초 전 · 현재 모습 아님")
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

    def _draw_center(self, painter: QPainter, head: str, detail: str) -> None:
        box = QRectF(self.rect()).adjusted(16, 0, -16, 0)
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

    def __init__(self, parent: QWidget) -> None:
        super().__init__("측정·보정 상세", "A·B 결과 미수신")
        self.setParent(parent)
        self.refresh_theme()
        self.values: dict[str, QLabel] = {}
        for key, name in (("px", "검출 중심 (u, v) px"), ("mm", "작업 좌표 (X, Y) mm"), ("theta", "장축각 θ °"),
                          ("calibration", "보정 상태"), ("frame", "source_id · image_seq")):
            row = QHBoxLayout()
            row.addWidget(label(name, "Dim"))
            row.addStretch(1)
            self.values[key] = label("—", "CardTitle")
            row.addWidget(self.values[key])
            self.body.addLayout(row)
        self.body.addWidget(label("값이 없을 때 0이나 이전 값을 채우지 않습니다. 사유: C 표시 연결 미결(TCP 명세 부록 B).", "Caption", wrap=True))
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
    """메인 디스플레이 + 그 위 오른쪽 가장자리에 붙은 보조 영상 2칸 + 상세 팝업.

    배치(골격 문서 2026-11-13 사용자 화면 방향):
      메인이 큰 바탕이고, 보조 2칸은 메인 안쪽 오른쪽에 세로로 고정, 팝업은 가운데~왼쪽.
    """

    def __init__(self) -> None:
        super().__init__()
        self.main = DisplayView("메인 디스플레이 · Gazebo 로봇")
        self.main.setParent(self)
        self.gripper = DisplayView("그리퍼 영상", compact=True)
        self.fixed = DisplayView("고정(탑뷰) 영상", compact=True)
        for view in (self.gripper, self.fixed):
            view.setParent(self)
            # [C 예정] 그리퍼·고정 영상은 C 표시 연결로만 온다. D 임시 경로에는 없다
            view.set_placeholder("empty", "C 표시 연결 미결")
        self.popup = DetailPopup(self)
        self.setMinimumHeight(300)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        width, height = self.width(), self.height()
        self.main.setGeometry(0, 0, width, height)
        aux_width = max(170, min(300, int(width * 0.21)))
        aux_height = min(int(aux_width * 9 / 16), (height - 86) // 2)
        left = width - aux_width - 16
        self.gripper.setGeometry(left, 52, aux_width, aux_height)
        self.fixed.setGeometry(left, 52 + aux_height + 10, aux_width, aux_height)
        popup_width = min(380, max(280, int(width * 0.3)))
        self.popup.setGeometry(24, 56, popup_width, self.popup.sizeHint().height())
        self.gripper.raise_()
        self.fixed.raise_()
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
        self.setFixedHeight(40)

    def set_stage(self, index: int | None) -> None:
        if index == self._current:
            return
        self._current = index
        motion.tween(self, 0.0, 1.0, theme.MS_STAGE, self._set_glow)

    def _set_glow(self, value: float) -> None:
        self._glow = float(value)
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(_font(11, 700))
        gap = 6
        width = (self.width() - gap * (len(self._stages) - 1)) / len(self._stages)
        for index, name in enumerate(self._stages):
            box = QRectF(index * (width + gap), 4, width, 32)
            active = index == self._current
            fill = motion.blend(QColor(theme.SURFACE_2), QColor(theme.ACCENT), self._glow) if active else QColor(theme.SURFACE_2)
            painter.setPen(QPen(QColor(theme.LINE), 1))
            painter.setBrush(fill)
            painter.drawRoundedRect(box, 16, 16)
            painter.setPen(QColor(theme.ON_ACCENT if active and self._glow > 0.5 else theme.DIM))
            painter.drawText(box, Qt.AlignmentFlag.AlignCenter, name)


# ── TX/RX 기록 ────────────────────────────────────────────────────────────

class TrafficTable(QTableWidget):
    """송수신 기록 표(GUI-03). 최신이 맨 위. 실제로 보내고 받은 줄만 적는다.

    [C 값] RX 는 C 서버 응답 원문 요약이다. 원문 줄은 행에 마우스를 올리면 보인다.
    화면에서 지워도 원본 파일은 지우지 않는다(2일차에는 파일 저장 자체가 없다. CSV 는 3일차).
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

    def refresh_theme(self) -> None:
        """이미 넣은 행의 방향 글자색을 새 테마 값으로 바꾼다."""
        for row in range(self.rowCount()):
            self.item(row, 1).setForeground(self._direction_color(self.item(row, 1).text()))

    def set_filter(self, direction: str) -> None:
        """direction: ""(전체) · "TX" · "RX" · "EV"."""
        self._filter = direction
        for row in range(self.rowCount()):
            self.setRowHidden(row, bool(direction) and self.item(row, 1).text() != direction)
