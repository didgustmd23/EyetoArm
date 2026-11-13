"""D 운영 GUI 공통 모션. Qt 내장 애니메이션만 쓴다(새 의존성·웹뷰·영상 렌더러 없음).

[담당] D 전용.
[규칙] docs/design/D_2일차_대시보드_골격.md 「모션그래픽 규칙」
  1. 모션은 "표시가 바뀌었다"는 사실만 알린다. 측정값·좌표·시각·수량을 보간해 중간값을 만들지 않는다.
  2. 상태 문구와 수치는 즉시 바뀐다. 모션이 끝날 때까지 기다리지 않는다.
  3. 정지·오류 표시는 깜박이거나 반복하지 않는다.
  4. 「모션 줄이기」를 켜면 모든 함수가 움직임 없이 최종 상태를 바로 적용한다.
[팀 참고] 여기의 움직임은 화면 장식이다. Gazebo 로봇의 실제 움직임이나 C가 보고한 단계와 무관하다.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QAbstractAnimation, QEasingCurve, QObject, QPoint, QPropertyAnimation, QRect, QTimer, QVariantAnimation
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QGraphicsOpacityEffect, QWidget

EASE_OUT = QEasingCurve.Type.OutCubic
EASE_IO = QEasingCurve.Type.InOutCubic
_DELETE = QAbstractAnimation.DeletionPolicy.DeleteWhenStopped

_reduced = False  # 「모션 줄이기」. 창의 체크박스와 --no-motion 이 바꾼다


def set_reduced(value: bool) -> None:
    global _reduced
    _reduced = bool(value)


def is_reduced() -> bool:
    return _reduced


def tween(owner: QObject, start, end, ms: int, on_value: Callable, curve=EASE_OUT,
          done: Callable[[], None] | None = None) -> QVariantAnimation | None:
    """start→end 값을 ms 동안 on_value 로 흘려보낸다. 숫자·QColor 를 받는다.

    모션 줄이기 상태면 on_value(end) 만 한 번 부르고 None 을 돌려준다.
    """
    if _reduced or ms <= 0:
        on_value(end)
        if done is not None:
            done()
        return None
    animation = QVariantAnimation(owner)
    animation.setDuration(ms)
    animation.setStartValue(start)
    animation.setEndValue(end)
    animation.setEasingCurve(curve)
    animation.valueChanged.connect(on_value)
    if done is not None:
        animation.finished.connect(done)
    animation.start(_DELETE)
    return animation


def move_geometry(widget: QWidget, target: QRect, ms: int) -> None:
    """위젯을 target 위치·크기로 옮긴다(탭 선택선 이동에 사용)."""
    previous = widget.findChild(QPropertyAnimation, "geometry_motion")
    if previous is not None:  # 빠르게 다시 누르면 이전 움직임을 정리하고 새로 시작한다
        previous.stop()
        previous.setObjectName("")
        previous.deleteLater()
    if _reduced or not widget.isVisible():
        widget.setGeometry(target)
        return
    animation = QPropertyAnimation(widget, b"geometry", widget)
    animation.setObjectName("geometry_motion")
    animation.setDuration(ms)
    animation.setStartValue(widget.geometry())
    animation.setEndValue(target)
    animation.setEasingCurve(EASE_IO)
    animation.start(_DELETE)


def appear(widget: QWidget, ms: int, rise_px: int = 12, delay_ms: int = 0) -> None:
    """위젯이 아래에서 살짝 올라오며 나타난다. 끝나면 효과 객체를 떼어 그리기 비용을 남기지 않는다.

    위치 이동은 레이아웃이 정한 자리로 끝나므로 최종 배치는 모션 유무와 같다.
    """
    if _reduced:
        return
    effect = QGraphicsOpacityEffect(widget)
    effect.setOpacity(0.0)
    widget.setGraphicsEffect(effect)

    def start() -> None:
        home = widget.pos()

        def step(value: float) -> None:
            effect.setOpacity(value)
            widget.move(home + QPoint(0, round((1.0 - value) * rise_px)))

        def finish() -> None:
            widget.move(home)
            widget.setGraphicsEffect(None)  # effect 는 Qt 가 함께 지운다

        tween(widget, 0.0, 1.0, ms, step, EASE_OUT, finish)

    QTimer.singleShot(delay_ms, widget, start)


def stagger(widgets: list[QWidget], ms: int, gap_ms: int) -> None:
    """여러 카드가 차례로 등장한다."""
    for index, widget in enumerate(widgets):
        appear(widget, ms, delay_ms=index * gap_ms)


def blend(a: QColor, b: QColor, t: float) -> QColor:
    """두 색을 t(0~1) 비율로 섞는다. 바탕 강조(flash)에 쓴다."""
    return QColor(round(a.red() + (b.red() - a.red()) * t), round(a.green() + (b.green() - a.green()) * t),
                  round(a.blue() + (b.blue() - a.blue()) * t), round(a.alpha() + (b.alpha() - a.alpha()) * t))
