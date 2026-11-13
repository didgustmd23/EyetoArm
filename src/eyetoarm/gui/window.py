"""D 운영 GUI 창(2일차 골격): 세 화면 + 공통 상태띠 + 메인 디스플레이.

실행 (Windows, conda 환경 eyetoarm, 저장소 루트에서):
  set PYTHONPATH=src
  python -m eyetoarm.gui.window            실제 C 서버(127.0.0.1:47090) + 화면에 떠 있는 Gazebo 창
  python -m eyetoarm.gui.window --view camera  Gazebo 창 대신 카메라 중계(127.0.0.1:47091, gz_view_relay.py 필요)
  python -m eyetoarm.gui.window --demo     모의 데이터로 외형·모션만 확인 (아무 것도 보내지 않음)
  python -m eyetoarm.gui.window --no-motion  모션 줄이기 상태로 시작
  python -m eyetoarm.gui.window --theme light  라이트 테마로 시작 (좌측 아래 「다크 / 라이트」로 실행 중 전환)
1일차 진단 CLI(python -m eyetoarm.gui)는 그대로다. 이 창은 별도 진입점이다.

┌─ 화면 구역과 연동 파트 ────────────────────────────────────────────────────┐
│ 구역              │ 지금 표시하는 것              │ 연동 파트 · 계약 상태    │
│───────────────────┼───────────────────────────────┼──────────────────────────│
│ 상태띠 명령 TCP   │ 연결 상태(실제)               │ C · TCP 1.1 확정         │
│ 상태띠 유효STATUS │ 최근 준비 응답 수신 여부(실제)│ C · TCP 1.1 확정         │
│ 상태띠 로봇 준비  │ robot_ready · 사유(실제)      │ C · TCP 1.1 확정         │
│ 상태띠 분류 상태  │ —                             │ C · 부록 A 미승인        │
│ 상태띠 표시 연결  │ Gazebo 화면 수신 상태(실제)   │ D 임시 경로(정식은 C·부록 B 미승인)│
│ 메인 디스플레이   │ Gazebo 창 실제 화면(기본)     │ D 임시 경로 · gz_window.py│
│                   │ 또는 카메라 중계(--view camera)│ D 임시 경로 · gz_view.py │
│ 보조 영상 2칸     │ 자리만                        │ C 표시 연결 · 미결       │
│ 측정·보정 상세    │ —                             │ A·B 결과 → C 중계 · 미결 │
│ 공정 단계 · 수량  │ 미확인 · —                    │ C 도착 확인 · 미결       │
│ START/STOP/RECOVER│ 비활성                        │ C · 부록 A 미승인        │
│ CANCEL(수동 이동) │ 비활성                        │ D 3일차 구현(TCP 1.1)    │
│ TX/RX 기록        │ 실제 송수신(준비 조회)        │ C · TCP 1.1 확정         │
└────────────────────────────────────────────────────────────────────────────┘
「—」와 비활성은 고장이 아니다. 계약이 정해지지 않아 값을 만들지 않은 자리다.

[공통 상태] 연결 객체(StatusLink 와 화면 수신기 GzWindowCapture/GzViewReceiver)는 창에 하나씩만 있고 세 화면이 함께 쓴다.
  화면을 바꿔도 연결·수신·기록이 초기화되지 않는다(D 상세설계 §A.2).
[보내는 신호] 무ID GET_STATUS 뿐이다. 이 창의 어떤 버튼도 로봇을 움직이지 않는다.
"""

from __future__ import annotations

import argparse
import sys

if not __package__:
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PySide6.QtCore import QSettings, QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QGridLayout, QHBoxLayout, QMainWindow, QStackedWidget, QVBoxLayout, QWidget,
)

from eyetoarm.gui import motion, theme
from eyetoarm.gui.gz_view_relay import DEFAULT_PORT as VIEW_PORT
from eyetoarm.gui.widgets import Card, DisplayStage, NavRail, StageStrip, StatusChip, TrafficTable, label, pill

PAGES = ["조작·설정", "운영 대시보드", "평가·기록"]
NOT_APPROVED = "TCP 명세 부록 A(분류 명령 개정 제안)가 C 승인 전입니다. 승인 전에는 실제 서버로 보내지 않습니다."
DAY3 = "수동 이동(MOVE)과 취소(CANCEL)는 D 3일차에 구현합니다. 지금은 보낼 대상 명령이 없습니다."
NO_VALUE = "—"


class MainWindow(QMainWindow):
    def __init__(self, link, view, demo: bool = False) -> None:
        super().__init__()
        self.link, self.view, self.demo = link, view, demo
        # 송수신 카드 꼬리표. 모의 모드에서는 「실제」라고 적지 않는다
        self._live_tag = "모의 · 실제 아님" if demo else "실제 송수신"
        self.setWindowTitle("Eye To Arms · D 운영 GUI (2일차 골격)")
        self.resize(1440, 960)
        self.setMinimumSize(1180, 760)

        root = QWidget(objectName="Root")
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── 좌측: 화면 선택 ────────────────────────────────────────────
        self.nav = NavRail(PAGES)
        self.source = label("", "Tag")
        self.reduce = QCheckBox("모션 줄이기")
        self.reduce.setChecked(motion.is_reduced())
        self.reduce.toggled.connect(motion.set_reduced)
        # 테마 선택(다크·라이트). 고르면 즉시 바뀌고 다음 실행 때도 유지된다
        theme_row = QHBoxLayout()
        theme_row.setSpacing(6)
        self.theme_buttons: dict[str, QWidget] = {}
        for key, text in theme.THEME_LABELS.items():
            button = pill(text, "small", f"{text} 테마로 바꿉니다. 영상 표시면은 항상 어둡습니다.", checkable=True)
            button.setAutoExclusive(True)
            button.setChecked(key == theme.name)
            button.clicked.connect(lambda _checked, k=key: self.set_theme(k))
            self.theme_buttons[key] = button
            theme_row.addWidget(button)
        theme_row.addStretch(1)
        self.nav.footer.addWidget(self.source)
        self.nav.footer.addLayout(theme_row)
        self.nav.footer.addWidget(self.reduce)
        outer.addWidget(self.nav)

        # ── 우측: 제목줄 · 상태띠 · 화면 ───────────────────────────────
        right = QVBoxLayout()
        right.setContentsMargins(28, 22, 28, 24)
        right.setSpacing(16)
        outer.addLayout(right, 1)
        right.addLayout(self._build_header())
        right.addLayout(self._build_status_strip())

        self.traffic_full = TrafficTable()      # 평가·기록 화면의 전체 기록
        self.traffic_recent = TrafficTable()    # 운영 대시보드의 최근 기록(같은 내용을 함께 받는다)
        self.stage = DisplayStage()             # 메인 디스플레이. 창에 하나만 있고 화면 사이를 옮겨 다닌다
        self.pages = QStackedWidget()
        self._page_cards: list[list[QWidget]] = []
        for build in (self._build_setup_page, self._build_operate_page, self._build_record_page):
            page, cards = build()
            self.pages.addWidget(page)
            self._page_cards.append(cards)
        right.addWidget(self.pages, 1)

        # ── 연결과 화면을 잇는다 ───────────────────────────────────────
        self.nav.selected.connect(self._show_page)
        self.link.traffic.connect(self.traffic_full.add_entry)
        self.link.traffic.connect(self.traffic_recent.add_entry)
        self.link.status_received.connect(self._on_status)
        self.view.frame_received.connect(self._on_frame)
        self._status: dict | None = None
        self._frame_size = ""

        self._refresh = QTimer(self, interval=250)   # 경과 시간·노후 판정을 다시 그린다. 통신은 하지 않는다
        self._refresh.timeout.connect(self._update_status)
        self._refresh.start()

        self._show_page(1, animate=False)
        self._update_status()
        self.link.start()
        self.view.start()

    # ══ 제목줄: 화면 이름 + 분류 조작 버튼 ══════════════════════════════════

    def _build_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)
        self.title = label("", "PageTitle")
        row.addWidget(self.title)
        row.addStretch(1)
        # [C 연동 · 부록 A 미승인] 분류 시작·정지·복구. 계약 승인 뒤 4일차에 연결한다
        self.start_button = pill("START 시작", "primary", NOT_APPROVED)
        self.stop_button = pill("STOP 분류 정지", "stop", NOT_APPROVED + " STOP 은 분류 작업 정지이며 수동 이동 취소(CANCEL)와 다릅니다.")
        self.recover_button = pill("RECOVER 복구", "", NOT_APPROVED)
        # [C 연동 · TCP 1.1] 수동 이동 취소. 3일차 구현
        self.cancel_button = pill("CANCEL 수동 이동 취소", "", DAY3)
        for button in (self.start_button, self.stop_button, self.recover_button, self.cancel_button):
            button.setEnabled(False)   # 비활성 사유는 툴팁과 「조작·설정」의 명령 계약 표에 있다
            row.addWidget(button)
        return row

    # ══ 상태띠: 항목마다 따로 표시한다(연결됨 ≠ 준비됨 ≠ 분류 중) ═══════════

    def _build_status_strip(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)
        self.chip_tcp = StatusChip("명령 TCP")        # [C] 47090 연결
        self.chip_view = StatusChip("표시 연결")      # [D 임시] Gazebo 창 캡처 또는 카메라 중계
        self.chip_status = StatusChip("유효 STATUS")  # [C] 검사를 통과한 최근 준비 응답
        self.chip_ready = StatusChip("로봇 준비")     # [C] robot_ready · not_ready_reasons
        self.chip_sort = StatusChip("분류 상태")      # [C 예정] 부록 A 미승인
        self.chip_age = StatusChip("마지막 수신")     # 이 PC 시계 기준 경과
        for chip in (self.chip_tcp, self.chip_view, self.chip_status, self.chip_ready, self.chip_sort, self.chip_age):
            row.addWidget(chip)
        self.chip_sort.set_state("idle", "— 계약 미승인")
        return row

    # ══ 화면 1: 조작·설정 ═══════════════════════════════════════════════════

    def _build_setup_page(self) -> tuple[QWidget, list[QWidget]]:
        page = QWidget()
        grid = QGridLayout(page)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(14)

        self.setup_display_slot = QVBoxLayout()   # 이 화면이 보일 때 메인 디스플레이가 여기로 온다
        display_card = Card("영상", "모의 · 실제 아님" if self.demo else "D 임시 경로")
        display_card.body.addLayout(self.setup_display_slot, 1)
        channel = QHBoxLayout()
        channel.addWidget(label("링크 채널", "Caption"))
        for name in ("raw 원본", "detect 검출", "center 중심점"):
            # [C·A·B 예정] 링크 채널 선택 요청. 요청 형식이 미결이라 비활성
            button = pill(name, tooltip="링크 채널 선택 요청 형식이 미결입니다(C·D 협의, TCP 명세 부록 B).")
            button.setEnabled(False)
            channel.addWidget(button)
        channel.addStretch(1)
        display_card.body.addLayout(channel)
        grid.addWidget(display_card, 0, 0, 2, 2)

        link_card = Card("연결", "client.json 읽기 전용")
        self._rows(link_card, [
            ("명령 TCP (C 서버)", f"{self.link.host}:{self.link.port}"),
            ("준비 조회 주기", f"{self.link.config['readiness_poll_ms']} ms"),
            ("STATUS 유효 한도", f"{self.link.config['status_watchdog_ms']} ms"),
            ("재접속 간격", f"{self.link.config['reconnect_ms']} ms"),
            ("Gazebo 화면 원본", self.view.title),
        ])
        grid.addWidget(link_card, 0, 2)

        command_card = Card("명령 계약 상태", "D ↔ C")
        self._rows(command_card, [
            ("GET_STATUS (준비 조회)", "사용 중 · TCP 1.1"),
            ("MOVE · CANCEL", "3일차 구현 · TCP 1.1"),
            ("START · STOP · RECOVER", "부록 A 미승인 · 미전송"),
            ("링크 채널 선택", "부록 B 미승인 · 미전송"),
        ])
        command_card.body.addStretch(1)
        grid.addWidget(command_card, 1, 2)

        input_card = Card("입력·보정", "A·B 결과 미수신")
        self._rows(input_card, [("입력 소스 · 모드", NO_VALUE), ("보정 프로파일", NO_VALUE), ("보정 상태 · 무효 사유", NO_VALUE)])
        input_card.body.addWidget(label("A·B 값은 C 표시 연결로 옵니다. 연결 방식이 정해지기 전에는 비워 둡니다.", "Caption", wrap=True))
        grid.addWidget(input_card, 2, 0)

        target_card = Card("명령 목표", "3일차")
        self._rows(target_card, [("화면 후보 → 확정 목표", NO_VALUE), ("command_id", NO_VALUE), ("허용·거부 사유", "수동 이동 미구현")])
        grid.addWidget(target_card, 2, 1)

        shape_card = Card("로봇 외형", "D 자산 · C 적용 확인 전")
        self._rows(shape_card, [("모델 이름 · 버전", NO_VALUE), ("Gazebo 적용 확인", "미실행")])
        shape_card.body.addWidget(label("정적 미리보기는 실제 관절 자세가 아닙니다. 움직임은 메인 디스플레이의 Gazebo 화면으로 봅니다.", "Caption", wrap=True))
        grid.addWidget(shape_card, 2, 2)

        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 3)
        grid.setColumnStretch(2, 4)
        grid.setRowStretch(0, 2)
        grid.setRowStretch(1, 2)
        grid.setRowStretch(2, 1)
        return page, [display_card, link_card, command_card, input_card, target_card, shape_card]

    # ══ 화면 2: 운영 대시보드 ═══════════════════════════════════════════════

    def _build_operate_page(self) -> tuple[QWidget, list[QWidget]]:
        page = QWidget()
        column = QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(14)

        self.operate_display_slot = QVBoxLayout()
        holder = QWidget()
        holder_layout = QVBoxLayout(holder)
        holder_layout.setContentsMargins(0, 0, 0, 0)
        holder_layout.setSpacing(10)
        tools = QHBoxLayout()
        self.detail_button = pill("측정·보정 상세", checkable=True, tooltip="메인 디스플레이 위에 상세 팝업을 펼칩니다.")
        self.detail_button.toggled.connect(self.stage.popup.toggle)
        tools.addWidget(self.detail_button)
        tools.addWidget(label(f"메인 디스플레이 원본: {self.view.title}. 검출 주석이 없는 화면입니다.", "Caption"))
        tools.addStretch(1)
        holder_layout.addLayout(tools)
        holder_layout.addLayout(self.operate_display_slot, 1)
        column.addWidget(holder, 1)

        cards = QHBoxLayout()
        cards.setSpacing(14)

        measure_card = Card("현재 측정", "A·B 미수신")
        self._rows(measure_card, [("(u, v) px", NO_VALUE), ("(X, Y) mm", NO_VALUE), ("θ °", NO_VALUE)])
        measure_card.body.addStretch(1)

        stage_card = Card("공정 단계", "C 확인 전")
        stage_card.body.addWidget(label("볼트", "Caption"))
        self.bolt_stages = StageStrip(["정지·재검출", "집기", "도착", "복귀"])
        stage_card.body.addWidget(self.bolt_stages)
        stage_card.body.addWidget(label("너트", "Caption"))
        self.nut_stages = StageStrip(["통과", "도착"])
        stage_card.body.addWidget(self.nut_stages)
        stage_card.body.addWidget(label("C가 확인한 단계만 강조합니다. 지금은 전부 미확인입니다.", "Caption", wrap=True))
        stage_card.body.addStretch(1)

        count_card = Card("수량", "C 도착 확인 전")
        self._rows(count_card, [("볼트 A 상자", NO_VALUE), ("볼트 B 상자", NO_VALUE), ("너트 상자", NO_VALUE), ("격리", NO_VALUE)])
        count_card.body.addStretch(1)

        recent_card = Card("최근 TX/RX", self._live_tag)
        recent_card.body.addWidget(self.traffic_recent, 1)

        for card, stretch in ((measure_card, 2), (stage_card, 4), (count_card, 2), (recent_card, 5)):
            card.setFixedHeight(236)
            cards.addWidget(card, stretch)
        column.addLayout(cards)
        return page, [holder, measure_card, stage_card, count_card, recent_card]

    # ══ 화면 3: 평가·기록 ═══════════════════════════════════════════════════

    def _build_record_page(self) -> tuple[QWidget, list[QWidget]]:
        page = QWidget()
        row = QHBoxLayout(page)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(14)

        log_card = Card("TX/RX 기록", "GUI-03 · " + self._live_tag)
        tools = QHBoxLayout()
        filters = []
        for text, direction in (("전체", ""), ("TX 보냄", "TX"), ("RX 받음", "RX"), ("EV 연결 사건", "EV")):
            button = pill(text, checkable=True)
            button.setAutoExclusive(True)
            button.toggled.connect(lambda on, d=direction: on and self.traffic_full.set_filter(d))
            filters.append(button)
            tools.addWidget(button)
        filters[0].setChecked(True)
        tools.addStretch(1)
        clear = pill("화면 비우기", tooltip="화면의 표만 비웁니다. 파일은 지우지 않습니다(2일차에는 파일 저장이 없습니다).")
        clear.clicked.connect(lambda: self.traffic_full.setRowCount(0))
        tools.addWidget(clear)
        log_card.body.addLayout(tools)
        log_card.body.addWidget(self.traffic_full, 1)
        row.addWidget(log_card, 5)

        side = QVBoxLayout()
        side.setSpacing(14)
        metric_card = Card("평가", "측정 전")
        self._rows(metric_card, [("사용자 실행 → ACK 지연", NO_VALUE), ("명령 성공률", NO_VALUE), ("위치 오차", NO_VALUE)])
        metric_card.body.addWidget(label("정답·관측이 없는 지표는 계산하지 않고 —로 둡니다.", "Caption", wrap=True))
        side.addWidget(metric_card)
        error_card = Card("오류 상세", "3일차")
        error_card.body.addWidget(label("명령별 state · stage · code · detail 을 3일차부터 연결합니다.", "Dim", wrap=True))
        error_card.body.addStretch(1)
        side.addWidget(error_card, 1)
        row.addLayout(side, 2)
        return page, [log_card, metric_card, error_card]

    @staticmethod
    def _rows(card: Card, rows: list[tuple[str, str]]) -> None:
        """「이름 ……… 값」 줄을 카드에 넣는다."""
        for name, value in rows:
            line = QHBoxLayout()
            line.addWidget(label(name, "Dim"))
            line.addStretch(1)
            line.addWidget(label(value, "CardTitle" if value != NO_VALUE else "Dim"))
            card.body.addLayout(line)

    # ══ 화면 전환 ═══════════════════════════════════════════════════════════

    def _show_page(self, index: int, animate: bool = True) -> None:
        """본문은 즉시 바뀐다. 그 뒤 카드가 차례로 등장한다. 연결·기록은 건드리지 않는다."""
        if index != 1 and self.detail_button.isChecked():
            self.detail_button.setChecked(False)
        # 메인 디스플레이는 하나뿐이다. 영상 칸이 있는 화면으로 옮긴다(수신은 끊기지 않는다)
        slot = {0: self.setup_display_slot, 1: self.operate_display_slot}.get(index)
        if slot is not None and slot.indexOf(self.stage) < 0:
            slot.addWidget(self.stage)
        self.pages.setCurrentIndex(index)
        self.title.setText(PAGES[index])
        self.nav.select(index, animate)
        if animate:
            motion.stagger(self._page_cards[index], theme.MS_PAGE, theme.MS_STAGGER)

    # ══ 테마 ═══════════════════════════════════════════════════════════════

    def set_theme(self, name: str, save: bool = True) -> None:
        """다크·라이트 전환. 연결·수신·기록은 건드리지 않는다. 표시 문구와 값은 그대로이고 색만 바뀐다.

        1) theme.apply 로 색 상수와 QSS 를 바꾼다
        2) 앱 전체 스타일시트를 다시 건다(QSS 로 꾸민 위젯)
        3) 색을 보관해 둔 위젯(refresh_theme 가 있는 것)을 맞추고, 직접 그리는 위젯을 다시 그린다
        """
        theme.apply(name)
        QApplication.instance().setStyleSheet(theme.QSS)
        for widget in self.findChildren(QWidget):
            if hasattr(widget, "refresh_theme"):
                widget.refresh_theme()
            widget.update()
        self.theme_buttons[theme.name].setChecked(True)
        if save:
            settings().setValue("theme", theme.name)

    # ══ 값 반영 ═════════════════════════════════════════════════════════════

    def _on_status(self, message: dict) -> None:
        self._status = message
        self._update_status()

    def _on_frame(self, image: QImage) -> None:
        self._frame_size = f"{image.width()}×{image.height()}"
        self.stage.main.show_frame(image)

    def _update_status(self) -> None:
        """250 ms 마다와 값 수신 때 부른다. 오래된 값은 현재값처럼 남기지 않는다."""
        link, view = self.link, self.view
        self.source.setText("모의 데이터 · 실제 아님" if self.demo else
                            ("실제 C 서버" if link.state == "connected" else "C 서버 미연결"))

        # 명령 TCP [C]
        self.chip_tcp.set_state(*{"connected": ("ok", "연결됨"), "connecting": ("warn", "접속 중"),
                                  "disconnected": ("fail", "끊김 · 재접속 대기")}[link.state])
        # 유효 STATUS · 로봇 준비 [C]
        if link.is_stale() or self._status is None:
            self.chip_status.set_state("idle" if link.state != "connected" else "fail",
                                       "없음" if link.state != "connected" else "응답 없음 · 한도 초과")
            self.chip_ready.set_state("idle", "— STATUS 없음")   # 이전 준비 상태를 살려 두지 않는다
        else:
            self.chip_status.set_state("ok", "수신 중")
            if self._status["robot_ready"]:
                self.chip_ready.set_state("ok", "준비됨")
            else:
                reasons = ", ".join(self._status["not_ready_reasons"]) or "사유 없음"
                self.chip_ready.set_state("warn", f"미준비 · {reasons}")
        age = link.status_age_ms()
        self.chip_age.set_state("idle" if age is None else "accent", NO_VALUE if age is None else f"{age / 1000:.1f}초 전")

        # 표시 연결 · 메인 디스플레이 [D 임시 경로]
        main = self.stage.main
        if view.state != "connected":
            self.chip_view.set_state("idle" if view.state == "disconnected" else "warn",
                                     "Gazebo 화면 없음" if view.state == "disconnected" else "접속 중")
            main.set_placeholder("waiting" if view.state == "connecting" else "empty", view.help_text)
        elif view.frame_age_ms() is None:
            self.chip_view.set_state("warn", "연결됨 · 화면 대기")
            main.set_placeholder("waiting", "Gazebo 화면 원본에 연결됐습니다. 첫 화면을 기다립니다.")
        elif view.is_stale():
            self.chip_view.set_state("fail", view.stale_label)
            main.set_stale(view.frame_age_ms())
        else:
            self.chip_view.set_state("ok", "Gazebo 수신 중")
        main.set_footer(f"{view.source_name} · {self._frame_size or NO_VALUE} · 프레임 {view.frame_count} · D 임시 확인 경로(C 표시 연결 아님)")

    def closeEvent(self, event) -> None:  # noqa: N802
        """[자원 정리] 이 창이 만든 타이머·소켓만 닫는다. C 서버·Gazebo·중계 프로세스는 건드리지 않는다."""
        self._refresh.stop()
        self.link.stop()
        self.view.stop()
        super().closeEvent(event)


def settings() -> QSettings:
    """사용자 화면 설정 저장소(Windows 레지스트리 HKCU/Software/EyeToArms/d_client). 테마 선택만 저장한다."""
    # 폴더명 변경 뒤에도 기존 테마 선택을 읽도록 저장 키는 유지한다.
    return QSettings("EyeToArms", "d_client")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eyetoarm.gui.window", description="D 운영 GUI (2일차 골격)")
    parser.add_argument("--host", default=None, help="C 서버 숫자 IP (기본: config/gui/client.json)")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--view", choices=("window", "camera"), default="window",
                        help="메인 디스플레이 원본. window(기본): 화면에 떠 있는 Gazebo 창을 그대로. "
                             "camera: Gazebo 카메라 영상 중계(2 Hz, WSL 에서 브리지와 gz_view_relay.py 필요)")
    parser.add_argument("--view-host", default="127.0.0.1", help="--view camera 일 때 중계 주소")
    parser.add_argument("--view-port", type=int, default=VIEW_PORT)
    parser.add_argument("--demo", action="store_true", help="모의 데이터로 외형·모션만 확인. 소켓을 열지 않음")
    parser.add_argument("--no-motion", action="store_true", help="모션 줄이기 상태로 시작")
    parser.add_argument("--theme", choices=tuple(theme.PALETTES), default=None,
                        help="시작 테마. 생략하면 지난번에 고른 테마(없으면 dark). 이 인자로 준 값은 저장하지 않음")
    parser.add_argument("--shot", default="", help="(개발용) 창을 띄우고 --shot-after ms 뒤 이 경로에 PNG 저장 후 종료")
    parser.add_argument("--shot-after", type=int, default=2500)
    parser.add_argument("--page", type=int, default=1, choices=(0, 1, 2), help="(개발용) 시작 화면")
    parser.add_argument("--popup", action="store_true", help="(개발용) 상세 팝업을 연 채 시작")
    args = parser.parse_args(argv)

    app = QApplication(sys.argv[:1])
    theme.apply(args.theme or str(settings().value("theme", "dark")))
    app.setStyleSheet(theme.QSS)
    motion.set_reduced(args.no_motion)
    if args.demo:
        from eyetoarm.gui.demo import DemoLink, DemoView
        link, view = DemoLink(), DemoView()
    else:
        from eyetoarm.gui.client import StatusLink
        link = StatusLink(args.host, args.port)
        if args.view == "window":
            from eyetoarm.gui.gz_window import GzWindowCapture   # Win32 호출을 쓰므로 Windows 에서만 불러온다
            view = GzWindowCapture()
        else:
            from eyetoarm.gui.gz_view import GzViewReceiver
            view = GzViewReceiver(args.view_host, args.view_port)
    window = MainWindow(link, view, demo=args.demo)
    window.show()
    if args.page != 1:
        window._show_page(args.page)
    if args.popup:
        window.detail_button.setChecked(True)
    if args.shot:
        def shoot() -> None:
            window.grab().save(args.shot)
            window.close()
        QTimer.singleShot(args.shot_after, shoot)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
