"""Eye To Arm 키네틱 타이포 표지. 카메라 좌표와 로봇 작업 좌표의 정렬을 활자로 표현한다."""

from __future__ import annotations

import math

from PySide6.QtCore import QEasingCurve, QElapsedTimer, QEvent, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from eyetoarm.gui import motion, theme

HOLD_MS = 3600        # 표지가 떠 있는 시간
IRIS_MS = 1450        # 타이틀에서 운영 화면으로 조리개가 열리는 시간
FADE_MS = 360         # 수동 건너뛰기의 짧은 전환 시간
STILL_MS = 1200       # 모션 줄이기일 때 완성 타이틀을 보여 주는 시간
_SKIP_KEYS = (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space, Qt.Key.Key_Escape)

# 이전 호출자가 사용하는 자세 계산 API. 타이포 표지에는 쓰지 않는다.
RISE_MS = 900
WAVE_HZ = 1.7         # 손을 흔드는 빠르기(1초에 왕복 횟수)

_FOLDED = (-62.0, 128.0, 10.0)   # (어깨, 팔꿈치, 손목) 각도. 도. 위쪽이 0, 시계 방향이 +, 앞 마디 기준 상대각
_RAISED = (-14.0, 24.0, 0.0)


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


def _ease(ms: float, start: float, end: float) -> float:
    t = min(1.0, max(0.0, (ms - start) / (end - start)))
    return t * t * (3.0 - 2.0 * t)


def _mix(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lettering(word: str, weight: int) -> QPainterPath:
    font = QFont()
    font.setFamilies(["Bahnschrift", "Archivo Narrow", *theme.FONT_FAMILIES, "DejaVu Sans"])
    font.setPixelSize(200)
    font.setWeight(QFont.Weight(weight))
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, -7)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    path = QPainterPath()
    path.addText(QPointF(0, 0), font, word)
    path.translate(-path.boundingRect().left(), 0)
    return path


def _print_word(p: QPainter, path: QPainterPath, x: float, baseline: float, scale: float,
                shear: float, direction: int, ms: float, ink: QColor) -> None:
    p.save()
    p.translate(x, baseline)
    p.scale(scale, scale)
    p.shear(shear, 0)
    if ms >= 950 or direction == 0:
        p.fillPath(path, ink)
    else:
        # Move three horizontal exposures separately, then draw one continuous outline.
        bounds = path.boundingRect().adjusted(-1, -1, 1, 1)
        cuts = (0.0, .36, .69, 1.0)
        for i, offset in enumerate((135, -92, 63)):
            travel = 1 - _ease(ms, 75*i, 670+140*i)
            top = bounds.top() + bounds.height()*cuts[i]
            band = QRectF(-200, top, bounds.width()+400, bounds.height()*(cuts[i+1]-cuts[i]))
            p.save()
            p.setClipRect(band, Qt.ClipOperation.IntersectClip)
            p.translate(direction * offset * travel, 0)
            p.fillPath(path, ink)
            p.restore()
    p.restore()


class _SpiralIris(QWidget):
    """Only the curved blades are painted. The dashboard stays live inside the opening."""

    def __init__(self, parent: QWidget, title: QPixmap, duration: int) -> None:
        super().__init__(parent)
        self._title = title
        self._progress = 0.0
        self.setGeometry(parent.rect())
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setAccessibleName("대시보드 열기")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.show()
        self.raise_()
        self.setFocus()
        motion.tween(self, 0.0, 1.0, duration, self._advance, QEasingCurve.Type.Linear, self.close)

    def _advance(self, progress: float) -> None:
        self._progress = progress
        if progress >= .18 and not self._title.isNull():
            self._title = QPixmap()
        self.setGeometry(self.parentWidget().rect())
        self.update()

    def mousePressEvent(self, _event) -> None:  # noqa: N802
        self.close()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in _SKIP_KEYS:
            self.close()
        else:
            super().keyPressEvent(event)

    def paintEvent(self, _event) -> None:  # noqa: N802
        t = self._progress
        opening = _ease(t, .18, .99)
        if opening >= 1:
            return
        w, h = self.width(), self.height()
        centre = QPointF(w * _mix(.60, .53, opening), h * .5)
        reach = math.hypot(max(centre.x(), w-centre.x()), h/2)
        radius, outer = 1.12 * reach * opening, 1.30 * reach
        span = outer - radius
        angle = -1.4 + 1.45 * _ease(t, 0, 1)
        step = math.tau / 8

        def radial(a):
            return QPointF(math.cos(a), math.sin(a))

        hole = QPainterPath()
        hole.moveTo(centre + radial(angle)*radius)
        for i in range(8):
            a = angle + i*step
            unit, tangent = radial(a), radial(a + math.pi/2)
            tip = centre + unit*radius
            next_tip = centre + radial(a+step)*radius
            hole.cubicTo(tip + (tangent*.40 + unit*.05)*radius,
                         next_tip - radial(a+step+math.pi/2)*(.22*radius), next_tip)

        hole.closeSubpath()
        # Radial speed starts at zero: each tapered seam leaves the aperture tangentially.
        previous = QPointF(radius, 0)
        seam = QPainterPath(previous)
        far_edge = []
        for j in range(1, 41):
            s = j / 40
            distance = radius + span * s*s*(2-s)
            twist = 1.72 * s*(1.65-.65*s)
            point = radial(twist)*distance
            delta = point - previous
            length = math.hypot(delta.x(), delta.y())
            width = min(w, h) * .029 * s*(2-s)
            seam.lineTo(point)
            far_edge.append(point + QPointF(delta.y(), -delta.x()) * (width / max(length, .001)))
            previous = point
        for point in reversed(far_edge):
            seam.lineTo(point)
        seam.closeSubpath()
        blades = QPainterPath()
        blades.setFillRule(Qt.FillRule.OddEvenFill)
        blades.addRect(QRectF(self.rect()))
        blades.addPath(hole)
        with QPainter(self) as p:
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.fillPath(blades, QColor("#161817"))
            p.save()
            p.translate(centre)
            p.rotate(math.degrees(angle))
            edge = QColor("#e9e9e2")
            for _ in range(8):
                p.fillPath(seam, edge)
                p.rotate(45)
            p.restore()
            if not self._title.isNull():
                p.setOpacity(1 - _ease(t, 0, .18))
                p.drawPixmap(self.rect(), self._title)


class CoverScreen(QWidget):
    """parent 를 꽉 덮는 표지. 만들면 바로 보이고, 닫힌 뒤 스스로 지워진다."""

    finished = Signal()   # 표지가 닫혔다(한 번). 창이 이어서 카드 등장 모션을 건다

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._words = tuple(_lettering(word, weight) for word, weight in (("Eye", 700), ("To", 350), ("Arm", 700)))
        self._widths = tuple(path.boundingRect().width() for path in self._words)
        self._clock = QElapsedTimer()
        self._clock.start()
        self._closing = False
        self._ticker = QTimer(self, interval=16)
        self._ticker.timeout.connect(self._tick)
        self.setAccessibleName("Eye To Arm")
        self.setAccessibleDescription("영상 좌표와 로봇 작업 좌표를 뜻하는 글자가 Eye To Arm으로 정렬됩니다. 클릭하거나 Enter를 눌러 건너뛸 수 있습니다.")
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
        self._ticker.stop()
        if not motion.is_reduced():
            duration = IRIS_MS if self._clock.elapsed() >= HOLD_MS else FADE_MS
            _SpiralIris(self.parentWidget(), self.grab(), duration)
        self.hide()
        # Start the dashboard's staggered entrance behind the iris, before it becomes visible.
        self.finished.emit()
        self.deleteLater()

    # ── 입력 ────────────────────────────────────────────────────────────

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if watched is self.parent() and event.type() == QEvent.Type.Resize:
            self.setGeometry(watched.rect())
        return False

    def mousePressEvent(self, _event) -> None:  # noqa: N802
        self.dismiss()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in _SKIP_KEYS:
            self.dismiss()
        else:
            super().keyPressEvent(event)

    def _tick(self) -> None:
        if not self._closing and self._clock.elapsed() >= HOLD_MS:
            self.dismiss()
        self.update()

    # ── 그리기 ──────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:  # noqa: N802
        with QPainter(self) as painter:
            self._paint_frame(painter, HOLD_MS if motion.is_reduced() else self._clock.elapsed())

    def _paint_frame(self, p: QPainter, ms: float) -> None:
        dark = theme.name == "dark"
        paper = QColor("#191b1a" if dark else "#eeece5")
        ink = QColor("#eeece4" if dark else "#1d201e")
        muted = QColor("#b3b5aa" if dark else "#50574e")
        hairline = QColor("#4e534c" if dark else "#b9bcb0")
        copper = QColor("#dc946c" if dark else "#a74f30")
        p.fillRect(self.rect(), paper)
        scale = min(self.width() / 1000, self.height() / 580)
        if scale <= 0:
            return
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        p.translate((self.width() - 1000*scale)/2, (self.height() - 580*scale)/2)
        p.scale(scale, scale)
        left, right = 52, 948
        p.setPen(QPen(hairline, 1))
        p.drawLine(QPointF(left, 65), QPointF(right, 65))
        p.drawLine(QPointF(left, 515), QPointF(right, 515))
        p.setFont(_font(13, 600, True))
        p.setPen(muted)
        p.drawText(QPointF(left, 53), "MACHINE VISION")

        eye, to, arm = self._words
        gap = 30
        fit = (right-left - 2*gap) / sum(self._widths)
        eye_w, to_w, arm_w = self._widths
        centre_x = left + eye_w*fit + gap
        arm_x = right - arm_w*fit
        acquire = _ease(ms, 0, 980)
        align = _ease(ms, 1300, 2610)
        transfer = _ease(ms, 930, 2510)
        baseline = 339

        # The same registration line travels from the image plane to the working plane.
        # It contracts to a short calibration mark below the connecting word.
        datum_y = _mix(278, 403, align)
        arrival_x = _mix(550, centre_x+to_w*fit, transfer)
        depart_x = _mix(450, centre_x, transfer)
        p.save()
        p.setOpacity(_ease(ms, 500, 980))
        p.setPen(QPen(copper, 1.4))
        p.drawLine(QPointF(depart_x, datum_y), QPointF(arrival_x, datum_y))
        p.drawLine(QPointF(depart_x, datum_y-4), QPointF(depart_x, datum_y+4))
        p.drawLine(QPointF(arrival_x, datum_y-4), QPointF(arrival_x, datum_y+4))
        p.setFont(_font(12, 500, True))
        p.setPen(muted)
        p.drawText(QRectF(depart_x-20, datum_y+8, arrival_x-depart_x+40, 23),
                   Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, "px → mm")
        p.restore()

        p.save()
        p.setClipRect(QRectF(left-4, 77, right-left+8, 423))
        opening_scale = fit * 1.26
        eye_start = _mix(239, left+7, acquire)
        arm_start = _mix(432, right-arm_w*opening_scale-7, acquire)
        lean = -.105 * (1-align)
        _print_word(p, eye, _mix(eye_start, left, align), _mix(268, baseline, align),
                    _mix(opening_scale, fit, align), lean, 1, ms, ink)
        _print_word(p, arm, _mix(arm_start, arm_x, align), _mix(446, baseline, align),
                    _mix(opening_scale, fit, align), lean, -1, ms, ink)
        small = fit * .41
        p.setOpacity(_ease(ms, 450, 960))
        _print_word(p, to, _mix(500-to_w*small/2, centre_x, transfer),
                    _mix(264, baseline, transfer), _mix(small, fit, transfer), 0, 0, ms, ink)
        p.restore()
        p.restore()
