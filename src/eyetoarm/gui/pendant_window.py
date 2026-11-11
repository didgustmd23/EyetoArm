"""수동 펜던트 창(PySide6). 외형은 현대로보틱스 Hi5a 티치 펜던트를 참고했다:
넓은 짙은 회색 몸체, 왼쪽 화면, 오른쪽 조그 키, 오른쪽 위 정지 버튼. 상표·로고·화면 그림은 쓰지 않는다.

키·버튼 조작 → Bridge.call("pendant_action") → 세션이 신호를 작성·기록한다.
TCP 1.1 로 실제 나가는 것은 START(MOVE)와 STOP·정지 버튼(CANCEL)뿐이다. 조그 신호는 D 제안 형식으로 작성·기록만 한다.
"""

from __future__ import annotations

import time
from typing import Any

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .bridge import Bridge
from .motion import Overlays, spring
from .theme import mono

CART, JOINT = ("X", "Y", "Z", "RZ"), ("J1", "J2", "J3", "J4", "J5", "J6")
UNIT = {"X": "mm", "Y": "mm", "Z": "mm", "RZ": "°"}
REPEAT_S = 0.15  # 키 자동 반복 최소 간격
HELP = (
    ("E", "인에이블 켜기/끄기. 꺼져 있으면 조그·그리퍼·START 가 동작하지 않음"),
    ("W  S", "직교: X + / −   ·   관절: 선택한 축 + / −"),
    ("A  D  ·  ←  →", "직교: Y + / −   ·   관절: 축 선택"),
    ("↑  ↓", "직교: Z + / −   ·   관절: 선택한 축 + / −"),
    ("Q  X", "직교: RZ − / +"),
    ("숫자패드 8  2", "그리퍼 닫기 / 열기 (5 mm씩)"),
    ("−  +", "속도 내림 / 올림 (5·10·25·50·100 %)"),
    ("Tab  ·  R", "좌표계 전환  ·  이동 단위 전환"),
    ("Home", "홈 자세 목표"),
    ("Enter", "START: 수동 목표를 TCP 1.1 MOVE 로 보냄"),
    ("Space  ·  Esc", "STOP: 진행 중 명령에 TCP 1.1 CANCEL 요청"),
    ("H", "이 도움말"),
)
STYLE = """
* { font-family: 'Noto Sans CJK KR', 'Noto Sans KR', 'Malgun Gothic', sans-serif; font-size: 13px; color: #171717; }
QWidget#pendant { background: #e9eaec; }
QLabel#top { color: #d7d9dc; font-size: 12px; }
QLabel#lamp { background: #1f2124; border: 1px solid #5a5e64; border-radius: 6px; }
QLabel#lamp[state="on"] { background: #4ade80; }
QLabel#lamp[state="wait"] { background: #fbbf24; }
QFrame#lcd { background: #f6f7f8; border: 6px solid #1f2124; border-radius: 10px; }
QLabel#bar { background: #1d1f22; color: #eceef0; font-size: 12px; padding: 6px 12px; }
QLabel#axis { color: #6b7078; }
QLabel#axis[sel="true"] { color: #0f766e; font-weight: 500; }
QLabel#value { font-size: 18px; }
QLabel#small { color: #6b7078; font-size: 12px; }
QLabel#raw { background: #eceef0; border-radius: 6px; padding: 8px 10px; color: #2f3237; }
QLabel#msg { color: #2f3237; border-top: 1px solid #e4e5e7; padding: 8px 12px; }
QLabel#foot { color: #a9adb3; font-size: 12px; }
QPushButton { background: #f1f1f0; color: #1d1f22; border: 0; border-bottom: 3px solid #b9bbbd; border-radius: 8px; padding: 4px; }
QPushButton:pressed, QPushButton[down="true"] { border-bottom: 1px solid #b9bbbd; margin-top: 2px; }
QPushButton:disabled { color: #a0a3a8; background: #dcdddd; }
QPushButton:focus { border: 2px solid #5eead4; border-bottom: 3px solid #5eead4; }
QPushButton#jog { font-size: 15px; font-weight: 500; }
QPushButton#jog[sel="true"] { border: 2px solid #0f766e; border-bottom: 3px solid #0b5a54; }
QPushButton#enable[on="true"] { background: #d9f5e3; border-bottom-color: #7fbf97; }
QPushButton#start { background: #15803d; color: #ffffff; border-bottom-color: #0f5a2b; font-weight: 500; }
QPushButton#stopkey { background: #c81e1e; color: #ffffff; border-bottom-color: #8a1414; font-weight: 500; }
QPushButton#estop { background: #c81e1e; color: #ffffff; border: 6px solid #e2b90f; border-radius: 32px; font-weight: 500; }
QPushButton#estop[locked="true"] { background: #8f1515; }
QFrame#toast { background: #ffffff; border: 1px solid #dcdcdc; border-radius: 10px; }
QFrame#popup { background: #ffffff; border-radius: 16px; }
QFrame#popup QPushButton { background: #171717; color: #ffffff; border: 0; border-radius: 8px; padding: 6px 16px; }
QLabel#h1 { font-size: 16px; font-weight: 500; }
QLabel#sub { color: #525252; font-size: 12px; }
QLabel#key { color: #6b7078; }
QWidget#dim { background: rgba(23, 23, 23, 115); }
"""


def _restyle(widget: QWidget, name: str, value: Any) -> None:
    if widget.property(name) != value:
        widget.setProperty(name, value)
        widget.style().unpolish(widget)
        widget.style().polish(widget)


class PendantWindow(QWidget):
    def __init__(self, bridge: Bridge) -> None:
        super().__init__(objectName="pendant")
        self.bridge = bridge
        self.state: dict[str, Any] | None = None
        self.last_signal: dict[str, Any] | None = None
        self.selected = 0
        self._last_key = 0.0
        self.setWindowTitle("EyetoArm 수동 펜던트")
        self.setStyleSheet(STYLE)
        self.setFixedSize(1120, 700)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        body = QVBoxLayout(self)
        body.setContentsMargins(76, 40, 76, 34)
        body.setSpacing(12)
        body.addLayout(self._build_top())
        middle = QHBoxLayout()
        middle.setSpacing(20)
        middle.addWidget(self._build_lcd(), 1)
        middle.addWidget(self._build_keys())
        body.addLayout(middle, 1)
        foot = QHBoxLayout()
        foot.addWidget(QLabel("EyetoArm 수동 펜던트", objectName="foot"))
        foot.addStretch(1)
        self.foot_note = QLabel(objectName="foot")
        foot.addWidget(self.foot_note)
        body.addLayout(foot)
        self.overlays = Overlays(self)

        bridge.state_changed.connect(self.apply_state)
        bridge.log_added.connect(self._on_log)
        for row in bridge.rows():
            self._on_log(row)
        self.apply_state(bridge.snapshot())

    # ── 구성 ───────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:  # noqa: N802 - 몸체와 양옆 손잡이
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#e9eaec"))
        body = QRectF(20, 20, self.width() - 40, self.height() - 40)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#25282b"))
        painter.drawRoundedRect(body.translated(0, 3), 36, 36)
        painter.setBrush(QColor("#3a3d42"))
        painter.drawRoundedRect(body, 36, 36)
        painter.setBrush(QColor("#2c2f33"))
        for x in (30.0, self.width() - 64.0):
            painter.drawRoundedRect(QRectF(x, 116, 34, self.height() - 232), 17, 17)

    def _build_top(self) -> QHBoxLayout:
        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(QLabel("수동 조작", objectName="top"))
        top.addSpacing(16)
        self.lamps: dict[str, QLabel] = {}
        for key, caption in (("enable", "인에이블"), ("tcp", "TCP"), ("ready", "준비")):
            lamp = QLabel(objectName="lamp")
            lamp.setFixedSize(12, 12)
            self.lamps[key] = lamp
            top.addWidget(lamp)
            top.addWidget(QLabel(caption, objectName="top"))
            top.addSpacing(10)
        top.addStretch(1)
        self.estop = QPushButton("정지", objectName="estop")
        self.estop.setFixedSize(64, 64)
        self.estop.setToolTip("소프트웨어 정지입니다. 실물 비상정지 장치가 아닙니다")
        self.estop.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.estop.clicked.connect(lambda: self.act("estop"))
        top.addWidget(self.estop)
        return top

    def _build_lcd(self) -> QFrame:
        lcd = QFrame(objectName="lcd")
        layout = QVBoxLayout(lcd)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        bar = QHBoxLayout()
        bar.setSpacing(1)
        self.bar: dict[str, QLabel] = {}
        for key in ("frame", "step", "speed", "link"):
            label = QLabel(objectName="bar")
            self.bar[key] = label
            bar.addWidget(label, 1 if key == "link" else 0)
        layout.addLayout(bar)
        body = QHBoxLayout()
        body.setContentsMargins(12, 12, 12, 8)
        body.setSpacing(16)
        axes = QGridLayout()
        axes.setVerticalSpacing(6)
        self.axis_names, self.axis_values = [], []
        for index in range(6):
            name, value = QLabel(objectName="axis"), QLabel(objectName="value")
            value.setFont(mono(14))
            value.setAlignment(Qt.AlignmentFlag.AlignRight)
            axes.addWidget(name, index, 0)
            axes.addWidget(value, index, 1)
            self.axis_names.append(name)
            self.axis_values.append(value)
        axes.setColumnStretch(1, 1)
        body.addLayout(axes, 1)
        side = QVBoxLayout()
        side.setSpacing(2)
        self.side: dict[str, QLabel] = {}
        for key, caption in (("grip", "그리퍼 개구폭"), ("command", "명령 진행"), ("signal", "마지막 신호")):
            side.addWidget(QLabel(caption, objectName="small"))
            value = QLabel("—", wordWrap=True)
            value.setFixedWidth(210)
            self.side[key] = value
            side.addWidget(value)
            side.addSpacing(6)
        side.addStretch(1)
        body.addLayout(side)
        layout.addLayout(body)
        self.raw = QLabel(objectName="raw", wordWrap=True)
        self.raw.setFont(mono(8))
        self.raw.setFixedHeight(96)
        self.raw.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.raw.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        wrap = QHBoxLayout()
        wrap.setContentsMargins(12, 0, 12, 8)
        wrap.addWidget(self.raw)
        layout.addLayout(wrap)
        layout.addStretch(1)
        self.message = QLabel(objectName="msg", wordWrap=True)
        layout.addWidget(self.message)
        return lcd

    def _key(self, text: str, hint: str = "", name: str = "") -> QPushButton:
        button = QPushButton(f"{text}\n{hint}" if hint else text, objectName=name)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # 키보드 입력이 창으로 가게 한다
        button.setMinimumHeight(40)
        return button

    def _build_keys(self) -> QWidget:
        keys = QWidget()
        keys.setFixedWidth(400)
        grid = QGridLayout(keys)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(8)
        functions = (("좌표계", "Tab", lambda: self.act("frame")), ("이동 단위", "R", lambda: self.act("step")),
                     ("홈", "Home", lambda: self.act("home")), ("도움말", "H", self.show_help),
                     ("잠금 해제", "", lambda: self.act("reset")), ("속도 −", "−", lambda: self.act("speed", sign=-1)),
                     ("속도 +", "+", lambda: self.act("speed", sign=1)))
        for row, (text, hint, run) in enumerate(functions):
            button = self._key(text, hint)
            button.clicked.connect(run)
            grid.addWidget(button, row, 0)
            if text == "잠금 해제":
                self.reset_key = button
        self.jog_buttons: list[tuple[QPushButton, QPushButton]] = []
        for row in range(6):
            minus, plus = self._key("", name="jog"), self._key("", name="jog")
            minus.clicked.connect(lambda _c=False, r=row: self._jog_row(r, -1))
            plus.clicked.connect(lambda _c=False, r=row: self._jog_row(r, 1))
            grid.addWidget(minus, row, 1)
            grid.addWidget(plus, row, 2)
            self.jog_buttons.append((minus, plus))
        self.grip_close, self.grip_open = self._key("그리퍼 닫기", "Num 8"), self._key("그리퍼 열기", "Num 2")
        self.grip_close.clicked.connect(lambda: self.act("grip", open=False))
        self.grip_open.clicked.connect(lambda: self.act("grip", open=True))
        grid.addWidget(self.grip_close, 6, 1)
        grid.addWidget(self.grip_open, 6, 2)
        self.enable = self._key("인에이블 켜기", "E", "enable")
        self.enable.clicked.connect(lambda: self.act("enable", on=not self.state["pendant"]["enabled"]))
        self.start = self._key("START", "Enter · 수동 목표 MOVE", "start")
        self.start.clicked.connect(lambda: self.act("start"))
        self.stop = self._key("STOP", "Space · CANCEL", "stopkey")
        self.stop.clicked.connect(lambda: self.act("stop"))
        for column, button in enumerate((self.enable, self.start, self.stop)):
            button.setMinimumHeight(52)
            grid.addWidget(button, 7, column)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        return keys

    # ── 조작 ───────────────────────────────────────────────────

    def act(self, action: str, **params: Any) -> None:
        # 인에이블이 꺼져 있으면 왜 안 움직이는지 바로 알려 준다(버튼은 눌리게 두고 안내한다)
        pendant = self.state["pendant"] if self.state else None
        if pendant is not None and action in ("jog", "grip", "start", "home"):
            if pendant["estop"]:
                self.overlays.toast("정지 잠금 중입니다 · '잠금 해제'를 누른 뒤 인에이블(E)을 켜세요")
                spring(self.reset_key)
            elif not pendant["enabled"]:
                self.overlays.toast("인에이블이 꺼져 있습니다 · E 키 또는 '인에이블 켜기'를 누르세요")
                spring(self.enable)
        self.bridge.call("pendant_action", action, **params)

    def _axes(self) -> tuple[str, ...]:
        return CART if self.state["pendant"]["frame"] == "cart" else JOINT

    def _jog_row(self, row: int, sign: int) -> None:
        axes = self._axes()
        if row < len(axes):
            self.act("jog", axis=axes[row], sign=sign)

    def _flash(self, button: QPushButton) -> None:
        button.setDown(True)
        from PySide6.QtCore import QTimer

        QTimer.singleShot(120, lambda: button.setDown(False))

    def show_help(self) -> None:
        if self.overlays.modal_open:
            self.overlays.close_modal()
            return
        box = QWidget()
        grid = QGridLayout(box)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setVerticalSpacing(6)
        grid.setHorizontalSpacing(20)
        for row, (key, text) in enumerate(HELP):
            name = QLabel(key, objectName="key")
            name.setFont(mono(9))
            grid.addWidget(name, row, 0)
            grid.addWidget(QLabel(text), row, 1)
        note = QLabel("조그·그리퍼·속도 신호는 D 제안 형식(pendant-draft/0.1)으로 작성해 기록만 합니다. 보낼 경로는 C와 합의한 뒤 연결합니다. "
                      "빨간 정지 버튼은 소프트웨어 정지이며 실물 비상정지 장치가 아닙니다.", objectName="sub", wordWrap=True)
        grid.addWidget(note, len(HELP), 0, 1, 2)
        self.overlays.popup("펜던트 사용 방법", box, width=620)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if self.state is None or event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier):
            return
        key, pad = event.key(), bool(event.modifiers() & Qt.KeyboardModifier.KeypadModifier)
        K = Qt.Key
        cart = self.state["pendant"]["frame"] == "cart"
        run, flash = None, None
        if pad and key in (K.Key_8, K.Key_Up):
            run, flash = (lambda: self.act("grip", open=False)), self.grip_close
        elif pad and key in (K.Key_2, K.Key_Down):
            run, flash = (lambda: self.act("grip", open=True)), self.grip_open
        elif key in (K.Key_Minus, K.Key_Underscore):
            run = lambda: self.act("speed", sign=-1)  # noqa: E731
        elif key in (K.Key_Plus, K.Key_Equal):
            run = lambda: self.act("speed", sign=1)  # noqa: E731
        elif key == K.Key_Tab:
            run = lambda: self.act("frame")  # noqa: E731
        elif key == K.Key_R:
            run = lambda: self.act("step")  # noqa: E731
        elif key == K.Key_E:
            run = lambda: self.act("enable", on=not self.state["pendant"]["enabled"])  # noqa: E731
        elif key in (K.Key_H, K.Key_F1):
            run = self.show_help
        elif key == K.Key_Home:
            run = lambda: self.act("home")  # noqa: E731
        elif key in (K.Key_Return, K.Key_Enter):
            run, flash = (lambda: self.act("start")), self.start
        elif key in (K.Key_Space, K.Key_Escape):
            if key == K.Key_Escape and self.overlays.modal_open:
                run = self.overlays.close_modal
            else:
                run, flash = (lambda: self.act("stop")), self.stop
        else:
            jog = None
            if cart:
                jog = {K.Key_W: (0, 1), K.Key_S: (0, -1), K.Key_A: (1, 1), K.Key_Left: (1, 1), K.Key_D: (1, -1), K.Key_Right: (1, -1),
                       K.Key_Up: (2, 1), K.Key_Down: (2, -1), K.Key_Q: (3, -1), K.Key_X: (3, 1)}.get(key)
            elif key in (K.Key_A, K.Key_Left, K.Key_D, K.Key_Right):
                step = -1 if key in (K.Key_A, K.Key_Left) else 1
                run = lambda: self._select((self.selected + step) % len(JOINT))  # noqa: E731
            elif key in (K.Key_W, K.Key_Up):
                jog = (self.selected, 1)
            elif key in (K.Key_S, K.Key_Down):
                jog = (self.selected, -1)
            if jog is not None:
                row, sign = jog
                run, flash = (lambda: self._jog_row(row, sign)), self.jog_buttons[row][0 if sign < 0 else 1]
        if run is None:
            super().keyPressEvent(event)
            return
        now = time.monotonic()
        if event.isAutoRepeat():
            if flash is None or now - self._last_key < REPEAT_S:
                return  # 눌러 두어도 신호가 쌓이지 않게 한다. 새로 누른 키는 항상 받는다
        self._last_key = now
        if flash is not None:
            self._flash(flash)
        run()

    def focusNextPrevChild(self, _next: bool) -> bool:  # noqa: N802 - Tab 을 좌표계 전환 키로 쓴다
        return False

    def _select(self, index: int) -> None:
        self.selected = index
        self.apply_state(self.state)

    # ── 표시 ───────────────────────────────────────────────────

    def _on_log(self, row: dict[str, Any]) -> None:
        if row["dir"] == "TX" and not (row["kind"] == "GET_STATUS" and not row["command_id"]):
            self.last_signal = row
            if self.state is not None:
                self.apply_state(self.state)

    def apply_state(self, state: dict[str, Any]) -> None:
        self.state = state
        p, t = state["pendant"], state["tcp"]
        cart = p["frame"] == "cart"
        axes = CART if cart else JOINT
        self.selected = min(self.selected, len(axes) - 1)
        self.bar["frame"].setText("좌표계 " + ("직교" if cart else "관절"))
        self.bar["step"].setText(f"단위 {p['step']:g} " + ("mm" if cart else "°"))
        self.bar["speed"].setText(f"속도 {round(p['speed_ratio'] * 100)} %")
        self.bar["link"].setText(f"TCP {t['phase']} · 준비 {t['ready_note']}")
        for index in range(6):
            shown = index < len(axes)
            self.axis_names[index].setVisible(shown)
            self.axis_values[index].setVisible(shown)
            minus, plus = self.jog_buttons[index]
            minus.setVisible(shown)
            plus.setVisible(shown)
            if not shown:
                continue
            axis = axes[index]
            value = p["cart"][axis] if cart else p["joint"][axis]
            self.axis_names[index].setText(axis)
            self.axis_values[index].setText(f"{value:.1f} {UNIT[axis] if cart else '°'}")
            selected = (not cart) and index == self.selected
            _restyle(self.axis_names[index], "sel", selected)
            minus.setText(f"{axis} −")
            plus.setText(f"{axis} +")
            for button in (minus, plus):
                _restyle(button, "sel", selected)
        self.side["grip"].setText(f"{p['grip_mm']:.0f} mm")
        self.side["command"].setText(t["current"]["result"] if t["current"] else "명령 없음")
        last = self.last_signal
        self.side["signal"].setText(f"{last['kind']} · {'송신함' if last['send_state'] == 'sent' else '미전송'}" if last else "없음")
        # 긴 JSON 이 화면 폭에서 줄바꿈되도록 쉼표 뒤에 폭 없는 공백을 넣는다(표시 전용, 기록 원문은 그대로)
        self.raw.setText(last["raw"].replace(",", ",​") if last else "조작하면 작성한 신호 원문이 여기에 표시됩니다")
        self.message.setText(p["message"])
        _restyle(self.lamps["enable"], "state", "on" if p["enabled"] else "")
        _restyle(self.lamps["tcp"], "state", "on" if t["phase"] == "연결됨" else "" if t["phase"] == "연결 안 됨" else "wait")
        _restyle(self.lamps["ready"], "state", "on" if t["robot_ready"] else "wait" if t["robot_ready"] is False else "")
        _restyle(self.enable, "on", p["enabled"])
        self.enable.setText("인에이블 끄기\nE" if p["enabled"] else "인에이블 켜기\nE")
        _restyle(self.estop, "locked", p["estop"])
        self.foot_note.setText("소프트웨어 정지 잠금 · 잠금 해제를 누르세요" if p["estop"] else "조그 신호는 작성·기록만 함 · 전송 경로 미합의")
