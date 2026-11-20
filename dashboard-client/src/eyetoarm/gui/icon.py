"""프로그램 아이콘: 좌표를 표시하는 느낌(X·Y 축 + 찍힌 점 하나). 팀명 「좌표로」에서 따왔다.

[담당] D 전용. 사용자 지정 2026-11-20.
[그림 자산 없음] 파일을 읽지 않고 QPainter 로 그린다. 창 제목줄·작업 표시줄·상단 상표 표식이 같은 그림을 쓴다.
[뜻] 주황 타일 위에 흰 X·Y 축, 축에서 점까지의 점선 안내선, 그리고 찍힌 점. 꾸밈이며 실제 측정 좌표를 나타내지 않는다.
[크기] 16~256px 을 크기마다 따로 그린다(줄여 쓰면 작은 크기에서 선이 뭉개진다). 24px 이하에서는 안내선을 뺀다.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF

SIZES = (16, 24, 32, 48, 64, 128, 256)
_TILE = "#F26B2A"     # theme 의 ACCENT 와 같은 주황. 아이콘은 테마를 따라 바뀌지 않는다
_INK = "#FFFFFF"
_POINT = "#14110F"


def draw_icon(size: int) -> QPixmap:
    """size×size 아이콘 한 장. 좌표는 256 기준으로 적고 unit 으로 줄인다."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    unit = size / 256.0

    def at(x: float, y: float) -> QPointF:
        return QPointF(x * unit, y * unit)

    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(_TILE))
    painter.drawRoundedRect(QRectF(0, 0, size, size), 58 * unit, 58 * unit)

    origin, top, right, point = at(66, 192), at(66, 54), at(204, 192), at(152, 104)

    if size > 24:   # 축에서 점까지의 점선 안내선(이 점의 X·Y 를 읽는 선)
        guide = QColor(_INK)
        guide.setAlpha(170)
        pen = QPen(guide, 7 * unit, Qt.PenStyle.CustomDashLine, Qt.PenCapStyle.RoundCap)
        pen.setDashPattern([0.1, 2.6])
        painter.setPen(pen)
        painter.drawLine(at(152, 192), point)
        painter.drawLine(at(66, 104), point)

    axis = QPen(QColor(_INK), 18 * unit, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    painter.setPen(axis)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPolyline(QPolygonF([top, origin, right]))
    # 화살촉
    painter.drawPolyline(QPolygonF([at(44, 76), top, at(88, 76)]))
    painter.drawPolyline(QPolygonF([at(182, 170), right, at(182, 214)]))

    # 찍힌 점: 흰 고리 안의 어두운 점
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(_INK))
    painter.drawEllipse(point, 30 * unit, 30 * unit)
    painter.setBrush(QColor(_POINT))
    painter.drawEllipse(point, 17 * unit, 17 * unit)
    painter.end()
    return pixmap


def app_icon() -> QIcon:
    """창·작업 표시줄에 쓰는 아이콘. QApplication 을 만든 뒤에 부른다."""
    icon = QIcon()
    for size in SIZES:
        icon.addPixmap(draw_icon(size))
    return icon
