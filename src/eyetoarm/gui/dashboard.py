"""대시보드 창(PySide6). 상태와 기록은 Bridge 시그널로 받고, 조작은 Bridge.call 로 보낸다. 화면이 결과를 추정하지 않는다.

화면: 조작·설정 / 운영 현황 / 실시간 셀 / 분류 / 사람 제어 / 평가·기록. D 상세설계 §A.2·§A.6·§A.11.
"""

from __future__ import annotations

import json
import time
from typing import Any

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMainWindow, QPlainTextEdit, QPushButton, QSizePolicy, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .bridge import Bridge
from .motion import DrawCheck, Overlays, Spinner, spring, stagger
from .theme import ACC, LINE2, NEG, STYLE, T1, T3, WARN, mono
from .twin_view import TwinView

PAGES = (
    ("ops", "조작·설정", "필수 화면 4개를 한 화면에 · 신호 작성 → 전송 → 기록"),
    ("status", "운영 현황", "상태를 주고받는 흐름 한눈에"),
    ("live", "실시간 셀", "Gazebo 값을 받으면 같은 움직임으로 표시 · 지금은 명령 목표만"),
    ("sort", "분류", "볼트 2규격은 상자로, 너트는 벨트 끝으로"),
    ("human", "사람 제어", "정해 둔 자세를 인식하면 미리 정의한 동작 하나를 실행"),
    ("records", "평가·기록", "화면과 같은 원시 기록으로 표·CSV를 만듦"),
)
RANK = {"ACK": 0, "PLANNING": 1, "MOVING": 2, "REACHED": 3, "FAILED": 3}
SORT_STAGES = ("RUNNING", "BELT_STOP", "REDETECT", "TARGET", "PICK", "TRANSPORT", "DROP", "ARRIVAL", "HOME", "RESTART")
KPI_ROWS = (("고유 최초 ACK", "first_ack"), ("재전송 뒤 수락", "accepted_after_resend"), ("조회·알림으로 수락 확인", "accepted_by_status"),
            ("수락 미확인", "unconfirmed"), ("재전송 횟수", "resend"), ("거부", "rejected"), ("도달", "reached"), ("실패", "failed"),
            ("작성했으나 미전송", "unsent"), ("잘못된 응답", "invalid_rx"))
MAX_ROWS = 500


def clock(wall_ms: int) -> str:
    return time.strftime("%H:%M:%S", time.localtime(wall_ms / 1000)) + f".{wall_ms % 1000:03d}"


def short_id(command_id: str | None) -> str:
    return f"{command_id[:4]}…{command_id[-4:]}" if command_id else ""


def row_text(row: dict[str, Any]) -> str:
    """기록 한 줄을 사람이 읽는 내용으로 줄인다. 원문은 상세 창에서 그대로 보여 준다."""
    if row["dir"] == "LOCAL":
        return row["note"]
    text = ""
    try:
        m = json.loads(row["raw"])
        if m.get("type") == "MOVE":
            text = f"{m['mode']} ({', '.join(str(v) for v in m['position_mm'])}) mm" + (f" θ {m['theta_deg']}°" if "theta_deg" in m else "")
        elif m.get("type") == "STATUS":
            if "state" in m:
                text = m["state"] + (f" · {m['code']}" if "code" in m else "")
            elif "code" in m:
                text = "거부 " + m["code"]
            else:
                text = f"robot_ready {str(m['robot_ready']).lower()}" + (" · " + ",".join(m["not_ready_reasons"]) if m["not_ready_reasons"] else "")
        elif "contract" in m:
            parts = [m.get("axis", "")]
            if "delta" in m:
                parts.append(f"{m['delta']:+g} {m['unit']}")
            if "opening_mm" in m:
                parts.append(f"개구폭 {m['opening_mm']:g} mm")
            parts.append(f"속도 {round(m['speed_ratio'] * 100)}%")
            text = " · ".join(p for p in parts if p)
    except (TypeError, ValueError, KeyError):
        pass
    return " · ".join(p for p in (text, row["note"]) if p)


def card(title: str, meta: str = "") -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame(objectName="card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(8)
    head = QHBoxLayout()
    head.addWidget(QLabel(title, objectName="h2"))
    head.addStretch(1)
    if meta:
        head.addWidget(QLabel(meta, objectName="meta"))
    layout.addLayout(head)
    return frame, layout


class KeyValues(QWidget):
    """이름 → 값 줄. 값은 set 으로 바꾼다."""

    def __init__(self, rows: tuple[tuple[str, str], ...], label_width: int = 112) -> None:
        super().__init__()
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setVerticalSpacing(5)
        grid.setColumnMinimumWidth(0, label_width)
        grid.setColumnStretch(1, 1)
        self.values: dict[str, QLabel] = {}
        for index, (key, caption) in enumerate(rows):
            grid.addWidget(QLabel(caption, objectName="muted"), index, 0, Qt.AlignmentFlag.AlignTop)
            value = QLabel("—", wordWrap=True)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            grid.addWidget(value, index, 1)
            self.values[key] = value

    def set(self, key: str, text: str, color: str | None = None, fixed: bool = False) -> None:
        label = self.values[key]
        label.setText(text)
        label.setStyleSheet(f"color: {color};" if color else "")
        if fixed:
            label.setFont(mono(10))


class Rail(QWidget):
    """단계 줄. 수락(ACK)과 도달을 섞지 않고 한 줄로 보여 준다."""

    def __init__(self, steps: tuple[str, ...], small: bool = False) -> None:
        super().__init__()
        self.steps, self.now, self.done_all, self.small = list(steps), -1, False, small
        self.setFixedHeight(40)

    def set(self, now: int, done_all: bool = False, last_label: str | None = None) -> None:
        self.now, self.done_all = now, done_all
        if last_label:
            self.steps[-1] = last_label
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        count = len(self.steps)
        width = self.width() / count
        if self.small:
            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)
        for index, label in enumerate(self.steps):
            x = index * width
            done = index < self.now or (index == self.now and self.done_all)
            current = index == self.now and not self.done_all
            if index < count - 1:
                painter.setPen(QPen(QColor(T1 if index < self.now else LINE2), 2))
                painter.drawLine(QPointF(x + 5, 6), QPointF(x + width + 5, 6))
            color = QColor(ACC if current else T1 if done else "#ffffff")
            painter.setPen(QPen(QColor(ACC if current else T1 if done else LINE2), 2))
            painter.setBrush(color)
            painter.drawEllipse(QPointF(x + 5, 6), 5, 5)
            painter.setPen(QColor(ACC if current else T1 if done else T3))
            painter.drawText(QRectF(x, 16, width, 22), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)


class CommandBlock(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.empty = QLabel("명령 없음 · 보낸 MOVE 가 있으면 여기에 진행 상태를 표시합니다", objectName="muted", wordWrap=True)
        self.top = KeyValues((("id", "command_id"), ("target", "목표"), ("basis", "근거")))
        self.rail = Rail(("수락 ACK", "계획", "이동", "도달"))
        self.bottom = KeyValues((("result", "진행"), ("accept", "수락 구분"), ("cancel", "취소"), ("error", "도달 오차")))
        for widget in (self.empty, self.top, self.rail, self.bottom):
            layout.addWidget(widget)
        layout.addStretch(1)

    def show_command(self, c: dict[str, Any] | None) -> None:
        for widget in (self.top, self.rail, self.bottom):
            widget.setVisible(c is not None)
        self.empty.setVisible(c is None)
        if c is None:
            return
        self.top.set("id", c["command_id"], fixed=True)
        self.top.set("target", f"({', '.join(str(v) for v in c['position_mm'])}) mm θ {c['theta_deg']}°", fixed=True)
        self.top.set("basis", f"{c['source_id']} #{c['image_seq']} · {c['calibration_id']}", fixed=True)
        self.rail.set(RANK.get(c["state"], -1), c["finished"], "실패" if c["state"] == "FAILED" else "도달")
        self.bottom.set("result", c["result"] + (f" · 송신 {c['attempts']}회" if c["attempts"] > 1 else ""))
        self.bottom.set("accept", c["acceptance"] or "—", fixed=True)
        cancel = {"requested": ("요청 전송 · 종료·정지 확인 중", None), "unsent": ("미전송 · 연결 끊김", WARN)}.get(c["cancel"], ("요청 없음", T3))
        self.bottom.set("cancel", *cancel)
        error = c["reach_error"]
        self.bottom.set("error", f"{error['position_mm']} mm / {error['rotation_deg']}°" if error else "도달 후 표시", None if error else T3)


class LogTable(QTableWidget):
    def __init__(self, wide: bool) -> None:
        headers = ("시각", "방향", "종류", "명령 ID", "전송 상태", "", "내용") if wide else ("시각", "방향", "종류", "명령 ID", "내용")
        super().__init__(0, len(headers))
        self.wide = wide
        self.setHorizontalHeaderLabels(headers)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setShowGrid(False)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(30)
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        for column, width in enumerate((112, 48, 150, 96) + ((84, 32) if wide else ())):
            self.setColumnWidth(column, width)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def add(self, row: dict[str, Any], limit: int) -> None:
        self.insertRow(0)  # 최신이 위
        state = {"unsent": "미전송", "sent": "송신함"}.get(row["send_state"], "")
        ident = short_id(row["command_id"]) or ("준비 조회" if row["kind"] == "GET_STATUS" else "")
        text = row_text(row) + (" · 미전송" if not self.wide and row["send_state"] == "unsent" else "")
        cells = [clock(row["wall_ms"]), row["dir"] if row["dir"] != "LOCAL" else "—", row["kind"], ident]
        cells += [state, "", text] if self.wide else [text]
        for column, value in enumerate(cells):
            item = QTableWidgetItem(value)
            if column in (0, 3):
                item.setFont(mono(9))
            if column == 1 and row["dir"] == "RX":
                item.setForeground(QColor(ACC))
            if (self.wide and column == 4 and row["send_state"] == "unsent"):
                item.setForeground(QColor(WARN))
            self.setItem(0, column, item)
        self.item(0, 0).setData(Qt.ItemDataRole.UserRole, row)
        if self.wide:
            pin = QPushButton("◆", objectName="pin")
            pin.setToolTip("표시해 두기")
            pin.setFixedSize(24, 24)
            pin.clicked.connect(lambda _c=False, b=pin: self._toggle(b))
            self.setCellWidget(0, 5, pin)
        while self.rowCount() > limit:
            self.removeRow(self.rowCount() - 1)

    @staticmethod
    def _toggle(button: QPushButton) -> None:
        button.setProperty("on", not button.property("on"))
        button.style().unpolish(button)
        button.style().polish(button)
        spring(button)  # 눌렀을 때 톡 튕긴다

    def row_data(self, index: int) -> dict[str, Any] | None:
        item = self.item(index, 0)
        return item.data(Qt.ItemDataRole.UserRole) if item else None


class Dashboard(QMainWindow):
    def __init__(self, bridge: Bridge) -> None:
        super().__init__()
        self.bridge = bridge
        self.state: dict[str, Any] | None = None
        self.pendant_window = None
        self._last_phase: str | None = None
        self.setWindowTitle("EyetoArm 대시보드")
        self.setStyleSheet(STYLE)
        self.resize(1440, 900)
        self.setMinimumSize(1180, 700)

        root = QWidget(objectName="root")
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_side())
        main = QVBoxLayout()
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)
        main.addWidget(self._build_top())
        self.stack = QStackedWidget()
        for build in (self._page_ops, self._page_status, self._page_live, self._page_sort, self._page_human, self._page_records):
            self.stack.addWidget(build())
        main.addWidget(self.stack, 1)
        foot = QLabel("D 클라이언트 · C 서버와 Gazebo는 포함하지 않음", objectName="meta")
        foot.setAlignment(Qt.AlignmentFlag.AlignRight)
        foot.setContentsMargins(0, 0, 24, 6)
        main.addWidget(foot)
        outer.addLayout(main, 1)
        self.overlays = Overlays(root)

        bridge.state_changed.connect(self.apply_state)
        bridge.log_added.connect(self.add_row)
        bridge.replied.connect(self.overlays.toast)
        for row in bridge.rows():
            self.add_row(row)
        self.apply_state(bridge.snapshot())
        self.go(0)

    # ── 구성 ───────────────────────────────────────────────────

    def _build_side(self) -> QWidget:
        side = QWidget(objectName="side")
        side.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        side.setFixedWidth(248)
        layout = QVBoxLayout(side)
        layout.setContentsMargins(12, 16, 12, 16)
        layout.setSpacing(2)
        brand = QHBoxLayout()
        brand.setContentsMargins(8, 0, 0, 16)
        brand.addWidget(QLabel("EyetoArm", objectName="brand"))
        brand.addWidget(QLabel("project09", objectName="meta"))
        brand.addStretch(1)
        layout.addLayout(brand)
        caption = QLabel("화면", objectName="muted")
        caption.setContentsMargins(8, 0, 0, 6)
        layout.addWidget(caption)
        self.nav = QButtonGroup(self)
        self.nav.setExclusive(True)
        for index, (_key, label, _sub) in enumerate(PAGES):
            button = QPushButton(label, objectName="nav", checkable=True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            self.nav.addButton(button, index)
            layout.addWidget(button)
        self.nav.idClicked.connect(self.go)
        caption = QLabel("수동 조작", objectName="muted")
        caption.setContentsMargins(8, 18, 0, 6)
        layout.addWidget(caption)
        pendant = QPushButton("펜던트 열기", objectName="nav")
        pendant.clicked.connect(self.open_pendant)
        layout.addWidget(pendant)
        layout.addStretch(1)

        links, body = card("연결")
        self.side = KeyValues((("tcp", "TCP 1.1"), ("ready", "로봇 준비"), ("exec", "실행 채널"), ("gz", "Gazebo 상태")), 84)
        self.side.set("exec", "계약 미확정", T3)
        self.side.set("gz", "연결 전", T3)
        body.addWidget(self.side)
        endpoint = QHBoxLayout()
        self.host_edit, self.port_edit = QLineEdit(), QLineEdit()
        self.host_edit.setFont(mono(9))
        self.port_edit.setFont(mono(9))
        self.port_edit.setFixedWidth(64)
        endpoint.addWidget(self.host_edit, 1)
        endpoint.addWidget(self.port_edit)
        body.addLayout(endpoint)
        row = QHBoxLayout()
        row.addWidget(QLabel("C 서버 주소", objectName="meta"))
        row.addStretch(1)
        self.connect_button = QPushButton("접속", objectName="link")
        self.connect_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.connect_button.clicked.connect(self.on_connect)
        row.addWidget(self.connect_button)
        body.addLayout(row)
        layout.addWidget(links)
        return side

    def _build_top(self) -> QWidget:
        top = QWidget(objectName="top")
        top.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        top.setFixedHeight(56)
        layout = QHBoxLayout(top)
        layout.setContentsMargins(24, 0, 24, 0)
        layout.setSpacing(12)
        self.title, self.subtitle = QLabel(objectName="h1"), QLabel(objectName="meta")
        layout.addWidget(self.title)
        layout.addWidget(self.subtitle)
        layout.addStretch(1)
        layout.addWidget(QLabel("실행 모드 · C 보고", objectName="meta"))
        for mode in ("IDLE", "MOVE_TEST", "AUTO_SORT", "HUMAN"):
            chip = QLabel(mode, objectName="seg")
            chip.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            chip.setToolTip("모드는 C가 보고한 값만 표시합니다. 지금은 수신 전입니다")
            layout.addWidget(chip)
        layout.addWidget(QLabel("수신 전", objectName="meta"))
        stop = QPushButton("정지 / 취소", objectName="stop")
        stop.setToolTip("진행 중 명령에 TCP 1.1 CANCEL 을 요청합니다. 실물 비상정지 장치가 아닙니다")
        stop.clicked.connect(lambda: self.bridge.call("cancel"))
        layout.addWidget(stop)
        return top

    @staticmethod
    def _page(columns: tuple[int, ...]) -> tuple[QWidget, QGridLayout]:
        page = QWidget()
        grid = QGridLayout(page)
        grid.setContentsMargins(24, 20, 24, 8)
        grid.setSpacing(14)
        for index, stretch in enumerate(columns):
            grid.setColumnStretch(index, stretch)
        return page, grid

    def _page_ops(self) -> QWidget:
        page, grid = self._page((1, 0))
        chips = QHBoxLayout()
        chips.addWidget(QLabel("입력", objectName="muted"))
        chips.addWidget(QLabel("A·B 입력 연결 전", objectName="tag"))
        self.chip_source = QLabel(objectName="tag")
        self.chip_source.setFont(mono(9))
        chips.addWidget(self.chip_source)
        for text in ("PLANAR", "manual_test"):
            chip = QLabel(text, objectName="tag")
            chip.setFont(mono(9))
            chips.addWidget(chip)
        chips.addStretch(1)
        self.chip_valid = QLabel(objectName="meta")
        chips.addWidget(self.chip_valid)
        grid.addLayout(chips, 0, 0, 1, 2)

        video, body = card("영상 · 검출 영역 · 장축 화살표", "GUI-01 · A 입력이 연결되면 영상과 검출 표시가 여기에 겹칩니다")
        self.twin_ops = TwinView()
        body.addWidget(self.twin_ops, 1)
        body.addWidget(QLabel("영상 입력 없음 · 위 그림은 펜던트·수동 목표의 명령 목표 미리보기이며 실제 로봇 상태가 아닙니다", objectName="meta"))
        grid.addWidget(video, 1, 0)

        right = QVBoxLayout()
        right.setSpacing(12)
        measure, body = card("측정", "GUI-02")
        values = KeyValues((("px", "영상 좌표"), ("mm", "작업 좌표 W"), ("theta", "장축각 θ")))
        values.set("px", "— 검출 결과 없음", T3)
        values.set("mm", "— (보정 필요)", T3)
        values.set("theta", "— (보정 필요)", T3)
        body.addWidget(values)
        right.addWidget(measure)
        calib, body = card("캘리브레이션 상태", "GUI-04 · 유효 전에는 비전 목표 이동 차단")
        values = KeyValues((("state", "상태"), ("profile", "프로파일")))
        values.set("state", "미설정")
        values.set("profile", "— B 프로파일 수신 전", T3)
        body.addWidget(values)
        right.addWidget(calib)
        manual, body = card("수동 목표와 실행", "시험용 · source_id pendant_manual")
        fields = QHBoxLayout()
        self.target_edits: dict[str, QLineEdit] = {}
        for axis, caption in (("X", "X mm"), ("Y", "Y mm"), ("RZ", "θ °")):
            box = QVBoxLayout()
            box.setSpacing(2)
            box.addWidget(QLabel(caption, objectName="meta"))
            edit = QLineEdit()
            edit.setFont(mono(10))
            self.target_edits[axis] = edit
            box.addWidget(edit)
            fields.addLayout(box)
        apply_button = QPushButton("적용")
        apply_button.clicked.connect(self.on_apply_target)
        fields.addWidget(apply_button, 0, Qt.AlignmentFlag.AlignBottom)
        body.addLayout(fields)
        run = QHBoxLayout()
        self.execute_button = QPushButton("현재 목표로 이동", objectName="primary")
        self.execute_button.clicked.connect(lambda: self.bridge.call("execute"))
        run.addWidget(self.execute_button, 1)
        self.exec_spinner = Spinner()
        self.exec_spinner.hide()
        run.addWidget(self.exec_spinner)
        run.addWidget(QLabel("확정 1회 → 명령 ID 1개\n→ MOVE 1건", objectName="meta"))
        body.addLayout(run)
        self.exec_note = QLabel(objectName="meta", wordWrap=True)
        body.addWidget(self.exec_note)
        right.addWidget(manual)
        right.addStretch(1)
        holder = QWidget()
        holder.setFixedWidth(392)
        holder.setLayout(right)
        right.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(holder, 1, 1)

        log, body = card("가상 로봇 송수신 기록", "GUI-03 · 행을 누르면 실제 송수신 필드를 표시")
        self.log_ops = LogTable(False)
        self.log_ops.cellClicked.connect(lambda r, _c: self.show_detail(self.log_ops.row_data(r)))
        body.addWidget(self.log_ops)
        grid.addWidget(log, 2, 0)
        command, body = card("확정 목표 · 명령 진행", "송신 시점 값 · 이후 새 입력과 분리")
        self.command_ops = CommandBlock()
        body.addWidget(self.command_ops)
        command.setFixedWidth(392)
        grid.addWidget(command, 2, 1)
        grid.setRowStretch(1, 3)
        grid.setRowStretch(2, 2)
        return page

    def _page_status(self) -> QWidget:
        page, grid = self._page((1, 1, 0))
        strip = QFrame(objectName="card")
        row = QHBoxLayout(strip)
        row.setContentsMargins(16, 10, 16, 10)
        row.setSpacing(32)
        self.strip: dict[str, QLabel] = {}
        for key, caption in (("source", "입력 소스"), ("tcp", "TCP 연결"), ("ready", "robot_ready"), ("reasons", "not_ready_reasons"),
                             ("active", "active_command_id"), ("time", "서버 시각")):
            box = QVBoxLayout()
            box.setSpacing(0)
            box.addWidget(QLabel(caption, objectName="meta"))
            value = QLabel("—")
            value.setFont(mono(10))
            self.strip[key] = value
            box.addWidget(value)
            row.addLayout(box)
        row.addStretch(1)
        grid.addWidget(strip, 0, 0, 1, 3)
        stats = QHBoxLayout()
        stats.setSpacing(12)
        self.stats: dict[str, QLabel] = {}
        self.stat_notes: dict[str, QLabel] = {}
        for key, caption, note in (("first_ack", "최초 ACK", "재전송 없이 받은 수락만"), ("reached", "도달 REACHED", "ACK와 따로 셈"),
                                   ("failed", "실패 FAILED", "새 이동 가능을 뜻하지 않음"), ("rejected", "거부", "state 없는 응답"),
                                   ("unconfirmed", "수락 미확인", "새 ID를 자동으로 만들지 않음"), ("resend", "재전송", "")):
            frame = QFrame(objectName="card")
            box = QVBoxLayout(frame)
            box.setContentsMargins(16, 10, 16, 10)
            box.setSpacing(2)
            box.addWidget(QLabel(caption, objectName="meta"))
            figure = QLabel("0", objectName="fig")
            self.stats[key] = figure
            box.addWidget(figure)
            hint = QLabel(note, objectName="meta")
            self.stat_notes[key] = hint
            box.addWidget(hint)
            stats.addWidget(frame)
        grid.addLayout(stats, 1, 0, 1, 3)
        command, body = card("현재 명령", "서버 STATUS를 command_id에 맞춰 적용")
        self.command_status = CommandBlock()
        body.addWidget(self.command_status)
        grid.addWidget(command, 2, 0)
        twin, body = card("명령 목표 미리보기", "실제 로봇 상태 아님")
        self.twin_status = TwinView()
        body.addWidget(self.twin_status, 1)
        grid.addWidget(twin, 2, 1)
        alarms, body = card("최근 오류·알람", "색만으로 구분하지 않고 글자로 표시")
        self.alarms = QTableWidget(0, 2)
        self.alarms.horizontalHeader().setVisible(False)
        self.alarms.verticalHeader().setVisible(False)
        self.alarms.setShowGrid(False)
        self.alarms.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.alarms.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.alarms.setColumnWidth(0, 88)
        self.alarms.horizontalHeader().setStretchLastSection(True)
        self.alarms.setWordWrap(True)
        body.addWidget(self.alarms)
        alarms.setFixedWidth(392)
        grid.addWidget(alarms, 2, 2)
        grid.setRowStretch(2, 1)
        return page

    def _page_live(self) -> QWidget:
        page, grid = self._page((1, 0))
        twin, body = card("실시간 셀 · 위에서 본 모습", "Gazebo 상태 수신 전 · 명령 목표만 표시")
        self.twin_live = TwinView()
        body.addWidget(self.twin_live, 1)
        body.addWidget(QLabel("받은 값을 그대로 그림 · 표본 사이만 직선 보간 · 끊기면 그 자리에 멈춤 · 벨트·부품 위치는 상태 스트림 연결 뒤 표시", objectName="meta"))
        grid.addWidget(twin, 0, 0, 3, 1)
        stream, body = card("상태 스트림", "전달 경로는 C와 합의 필요")
        values = KeyValues((("rate", "수신 주기"), ("last", "마지막 수신"), ("stale", "끊김 판정")))
        values.set("rate", "미측정 · 수신 없음", T3)
        values.set("last", "없음", T3)
        values.set("stale", '250 ms 넘으면 멈춤 · "수신 끊김"')
        body.addWidget(values)
        stream.setFixedWidth(372)
        grid.addWidget(stream, 0, 1)
        goal, body = card("명령 목표", "펜던트·수동 목표가 만든 값")
        self.joints = KeyValues(tuple((f"J{i}", f"J{i}") for i in range(1, 7)) + (("tcp", "끝단 목표"), ("grip", "그리퍼 개구폭"), ("speed", "속도")))
        body.addWidget(self.joints)
        body.addStretch(1)
        grid.addWidget(goal, 1, 1)
        belt, body = card("벨트 · 부품", "C·Gazebo 값")
        body.addWidget(QLabel("수신 없음 · 0으로 채우지 않음", objectName="muted"))
        grid.addWidget(belt, 2, 1)
        grid.setRowStretch(1, 1)
        return page

    def _page_sort(self) -> QWidget:
        page, grid = self._page((1, 0))
        main, body = card("분류", "컨베이어 분류 1~3주차")
        body.addWidget(QLabel("C의 실행 상태 수신 전", objectName="h1"))
        body.addWidget(QLabel("분류 시작·정지·재개, 분류 단계, 상자별 수량, 격리 수, 실패 사유는 C가 보고한 값만 표시합니다. "
                              "확장 명령 계약(ext-cmd/0.1)이 정해지면 이 화면에 연결합니다.", objectName="sub", wordWrap=True))
        body.addSpacing(12)
        body.addWidget(Rail(SORT_STAGES, small=True))
        body.addStretch(1)
        grid.addWidget(main, 0, 0)
        bins, body = card("상자별 수량", "도착 확인 이벤트로만 증가")
        values = KeyValues((("a", "볼트 A 상자"), ("b", "볼트 B 상자"), ("n", "너트 상자"), ("i", "격리")))
        for key in ("a", "b", "n", "i"):
            values.set(key, "미측정", T3)
        body.addWidget(values)
        body.addStretch(1)
        bins.setFixedWidth(360)
        grid.addWidget(bins, 0, 1)
        return page

    def _page_human(self) -> QWidget:
        page, grid = self._page((1, 0))
        main, body = card("사람 제어", "4~5주차")
        body.addWidget(QLabel("자세 입력(A)과 실행 상태(C) 수신 전", objectName="h1"))
        body.addWidget(QLabel("동의와 활성화, 관절점 표시, 의도 판정, 하트비트, 유실 표시는 입력이 연결된 뒤 동작합니다. "
                              "추적 유실·오래된 입력·하트비트 만료이면 자동으로 다시 활성화하지 않습니다.", objectName="sub", wordWrap=True))
        body.addStretch(1)
        grid.addWidget(main, 0, 0)
        consent, body = card("동의와 활성화", "활성화 상태가 곧 승인")
        row = QHBoxLayout()
        self.consent = DrawCheck()
        row.addWidget(self.consent)
        row.addWidget(QLabel("카메라로 내 자세를 인식해 로봇 동작을 제어하는 데 동의", wordWrap=True), 1)
        body.addLayout(row)
        body.addWidget(QLabel("동의해도 HUMAN 모드와 C 확인 전에는 활성화되지 않습니다", objectName="meta", wordWrap=True))
        body.addStretch(1)
        consent.setFixedWidth(392)
        grid.addWidget(consent, 0, 1)
        return page

    def _page_records(self) -> QWidget:
        page, grid = self._page((0, 1))
        kpi, body = card("통신", "최초 ACK와 그 밖을 따로 집계")
        self.kpi = KeyValues(tuple((key, caption) for caption, key in KPI_ROWS), 170)
        body.addWidget(self.kpi)
        kpi.setFixedWidth(320)
        grid.addWidget(kpi, 0, 0)
        settings, body = card("기록 파일과 설정", "정의서 §9 초기값")
        self.settings = KeyValues((("log", "기록 파일"), ("pendant", "펜던트 조그 신호"), ("timers", "시간 설정"), ("limits", "재전송·줄 길이")), 150)
        body.addWidget(self.settings)
        buttons = QHBoxLayout()
        self.export_button = QPushButton("CSV 내보내기")
        self.export_button.clicked.connect(self.on_export)
        buttons.addWidget(self.export_button)
        self.export_spinner = Spinner()
        self.export_spinner.hide()
        buttons.addWidget(self.export_spinner)
        clear = QPushButton("화면 비우기")
        clear.clicked.connect(self.on_clear)
        buttons.addWidget(clear)
        buttons.addWidget(QLabel("화면을 비워도 기록 파일은 지우지 않습니다", objectName="meta"))
        buttons.addStretch(1)
        body.addStretch(1)
        body.addLayout(buttons)
        grid.addWidget(settings, 0, 1)
        log, body = card("명령·이벤트 기록")
        self.log_count = QLabel(objectName="meta")
        body.itemAt(0).layout().addWidget(self.log_count)
        self.log_all = LogTable(True)
        self.log_all.cellClicked.connect(lambda r, c: self.show_detail(self.log_all.row_data(r)) if c != 5 else None)
        body.addWidget(self.log_all)
        grid.addWidget(log, 1, 0, 1, 2)
        grid.setRowStretch(1, 1)
        return page

    # ── 수신 → 표시 ─────────────────────────────────────────────

    def add_row(self, row: dict[str, Any]) -> None:
        if row["dir"] != "LOCAL":
            self.log_ops.add(row, 80)
        self.log_all.add(row, MAX_ROWS)
        alarm = row["send_state"] == "unsent" or row["kind"] == "잘못된 응답" or (
            row["dir"] == "LOCAL" and any(word in row["kind"] for word in ("실패", "끊김", "초과", "미확인")))
        if alarm:
            self.alarms.insertRow(0)
            when = QTableWidgetItem(clock(row["wall_ms"])[:8])
            when.setFont(mono(9))
            when.setForeground(QColor(T3))
            self.alarms.setItem(0, 0, when)
            self.alarms.setItem(0, 1, QTableWidgetItem(f"{row['kind']}{' 미전송' if row['send_state'] == 'unsent' else ''} · {row['note']}"))
            while self.alarms.rowCount() > 40:
                self.alarms.removeRow(40)
        self._count()

    def _count(self) -> None:
        routine = self.state["routine_count"] if self.state else 0
        self.log_count.setText(f"{self.log_all.rowCount()}행 표시 · 최근 {MAX_ROWS}행까지 · 정기 준비 조회 {routine}건은 파일에만 기록")

    def apply_state(self, state: dict[str, Any]) -> None:
        self.state = state
        t, p, c = state["tcp"], state["pendant"], state["tcp"]["current"]
        live = t["phase"] == "연결됨"
        self.side.set("tcp", t["phase"], ACC if live else T3)
        self.side.set("ready", t["ready_note"], ACC if t["robot_ready"] else WARN if t["robot_ready"] is False else T3)
        self.connect_button.setText("접속" if t["phase"] == "연결 안 됨" else "연결 해제")
        if not self.host_edit.hasFocus():
            self.host_edit.setText(t["host"])
        if not self.port_edit.hasFocus():
            self.port_edit.setText(str(t["port"]))
        if self._last_phase is not None and self._last_phase != t["phase"] and "연결됨" in (t["phase"], self._last_phase):
            text = "TCP 1.1 연결됨 · 로봇 준비를 조회합니다" if live else "연결 끊김 · MOVE 를 자동으로 다시 보내지 않습니다"
            self.overlays.snack(text, "기록 보기", lambda: self.go(5))
        self._last_phase = t["phase"]

        self.chip_source.setText(f"pendant_manual #{p['target_seq']}")
        self.chip_valid.setText(f"수동 목표 유효 시간 {state['config']['target_validity_ms']} ms · 시험용이며 비전 근거가 아님")
        for axis, edit in self.target_edits.items():
            if not edit.hasFocus():
                edit.setText(f"{p['cart'][axis]:g}")
        waiting = bool(c and not c["finished"] and c["attempts"] and not c["state"])
        self.exec_spinner.setVisible(waiting)
        self.execute_button.setEnabled(not (c and not c["finished"]))
        if t["message"]:
            note = t["message"]
        elif not live:
            note = "TCP 연결 없음 · 누르면 신호를 작성해 기록만 하고 보내지 않음"
        elif t["robot_ready"]:
            note = "로봇 준비됨 · 보낼 수 있음"
        else:
            reasons = ": " + ",".join(t["not_ready_reasons"]) if t["not_ready_reasons"] else ""
            note = f"로봇 {t['ready_note']}{reasons} · 누르면 작성만 하고 보내지 않음"
        self.exec_note.setText(note)
        self.command_ops.show_command(c)
        self.command_status.show_command(c)
        for twin in (self.twin_ops, self.twin_status, self.twin_live):
            twin.set_command(p)

        self.strip["source"].setText("A·B 입력 연결 전")
        self.strip["tcp"].setText(t["phase"])
        self.strip["ready"].setText("미확인" if t["robot_ready"] is None else str(t["robot_ready"]).lower())
        self.strip["ready"].setStyleSheet(f"color: {NEG};" if t["robot_ready"] is False else "")
        self.strip["reasons"].setText(", ".join(t["not_ready_reasons"]) or "—")
        self.strip["active"].setText(short_id(t["active_command_id"]) or "—")
        self.strip["time"].setText("미측정" if t["server_time_ms"] is None else str(t["server_time_ms"]))
        kpi = t["kpi"]
        for key, label in self.stats.items():
            label.setText(str(kpi[key]))
        self.stat_notes["resend"].setText(f"복구된 ACK {kpi['accepted_after_resend']}")
        for _caption, key in KPI_ROWS:
            self.kpi.set(key, str(kpi[key]), fixed=True)
        config = state["config"]
        self.settings.set("log", (state["log_path"] or "저장 안 함") + (f" · 기록 실패: {state['log_error']}" if state["log_error"] else ""),
                          NEG if state["log_error"] else None, fixed=True)
        self.settings.set("pendant", state["pendant_channel"])
        self.settings.set("timers", f"ACK {config['ack_timeout_ms']} ms · 준비 조회 {config['readiness_poll_ms']} ms · 준비 응답 "
                                    f"{config['readiness_timeout_ms']} ms · 알림 감시 {config['status_watchdog_ms']} ms", fixed=True)
        self.settings.set("limits", f"재전송 최대 {config['max_resend']}회 · 한 줄 {config['max_line_bytes']}바이트 · 수동 목표 유효 {config['target_validity_ms']} ms", fixed=True)
        for name, value in p["joint"].items():
            self.joints.set(name, f"{value:.1f}°", fixed=True)
        self.joints.set("tcp", f"({p['cart']['X']:g}, {p['cart']['Y']:g}, {p['cart']['Z']:g}) mm", fixed=True)
        self.joints.set("grip", f"{p['grip_mm']:g} mm", fixed=True)
        self.joints.set("speed", f"{round(p['speed_ratio'] * 100)} %", fixed=True)
        self._count()

    # ── 조작 ───────────────────────────────────────────────────

    def go(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        self.nav.button(index).setChecked(True)
        self.title.setText(PAGES[index][1])
        self.subtitle.setText(PAGES[index][2])
        stagger([self.title, self.subtitle])  # 제목 → 부제 순서로 등장

    def on_connect(self) -> None:
        if self.state["tcp"]["phase"] == "연결 안 됨":
            try:
                port = int(self.port_edit.text())
            except ValueError:
                self.overlays.toast("포트는 숫자로 입력하세요")
                return
            self.bridge.call("connect", self.host_edit.text().strip() or None, port)
            return
        c = self.state["tcp"]["current"]
        if c and not c["finished"] and c["attempts"]:
            self.overlays.popup("연결을 해제할까요", "진행 중인 명령이 있습니다. 연결을 끊어도 로봇은 멈추지 않으며 결과는 미확인으로 남습니다.",
                                "연결 해제", lambda: self.bridge.call("disconnect"))
        else:
            self.bridge.call("disconnect")

    def on_apply_target(self) -> None:
        try:
            x, y, theta = (float(self.target_edits[axis].text()) for axis in ("X", "Y", "RZ"))
        except ValueError:
            self.overlays.toast("숫자를 입력하세요")
            return
        for edit in self.target_edits.values():
            edit.clearFocus()
        self.bridge.call("set_target", x, y, theta)

    def on_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "CSV 내보내기", "eyetoarm-events.csv", "CSV (*.csv)")
        if path:
            self.export_to(path)

    def export_to(self, path: str) -> None:
        self.export_spinner.show()
        try:
            with open(path, "w", encoding="utf-8-sig", newline="") as handle:
                handle.write(self.bridge.session.log.to_csv())
            self.overlays.snack(f"CSV 내보내기 완료 · {len(self.bridge.rows())}행")
        except OSError as error:
            self.overlays.toast(f"내보내기 실패: {error}")
        finally:
            self.export_spinner.hide()

    def on_clear(self) -> None:
        for table in (self.log_ops, self.log_all, self.alarms):
            table.setRowCount(0)
        self._count()
        self.overlays.toast("화면만 비움 · 기록 파일은 그대로")

    def show_detail(self, row: dict[str, Any] | None) -> None:
        if row is None:
            return
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        head = QHBoxLayout()
        what = {"RX": "실제 수신 필드", "TX": "실제 작성한 신호"}.get(row["dir"], "사건")
        head.addWidget(QLabel(f"{row['dir']} · {row['kind']} · {what}", objectName="h2"))
        when = QLabel(clock(row["wall_ms"]), objectName="meta")
        head.addWidget(when)
        head.addStretch(1)
        state = {"unsent": "작성했으나 보내지 않음", "sent": "소켓에 씀 · 서버 수락(ACK)은 별도 사건"}.get(
            row["send_state"], "수신" if row["dir"] == "RX" else "로컬 사건")
        head.addWidget(QLabel(state, objectName="tag"))
        copy = QPushButton("복사")
        copy.clicked.connect(lambda: (QGuiApplication.clipboard().setText(row["raw"] or ""), self.overlays.toast("기록 1행 복사됨")))
        head.addWidget(copy)
        close = QPushButton("닫기")
        close.clicked.connect(self.overlays.close_modal)
        head.addWidget(close)
        layout.addLayout(head)
        try:
            pretty = json.dumps(json.loads(row["raw"]), ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            pretty = row["raw"] or "(원문 없음)"
        text = QPlainTextEdit(pretty, readOnly=True)
        text.setFont(mono(10))
        layout.addWidget(text, 1)
        info = KeyValues((("channel", "채널"), ("bytes", "바이트 수"), ("note", "비고")), 96)
        info.set("channel", row["channel"], fixed=True)
        info.set("bytes", f"{row['bytes']} (LF 포함)" if row["bytes"] else "—", fixed=True)
        info.set("note", row["note"] or "—")
        layout.addWidget(info)
        self.overlays.sheet(box, 400)

    def open_pendant(self) -> None:
        from .pendant_window import PendantWindow

        if self.pendant_window is None:
            self.pendant_window = PendantWindow(self.bridge)
        self.pendant_window.show()
        self.pendant_window.raise_()
        self.pendant_window.activateWindow()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape and self.overlays.modal_open:
            self.overlays.close_modal()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.overlays.relayout()

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.pendant_window is not None:
            self.pendant_window.close()
        self.bridge.stop()
        super().closeEvent(event)
