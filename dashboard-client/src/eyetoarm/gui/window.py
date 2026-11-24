"""D 운영 GUI 창: TCP 1.1 + 소켓 없는 4일차 분류 대역.

실행 (Windows, conda 환경 eyetoarm, 저장소 루트에서):
  set PYTHONPATH=dashboard-client/src
  python -m eyetoarm.gui.window --host <C PC 숫자 IP>
                                           실제 C 서버(명령 47090) + C 3채널 원본 영상(기본 view relay: 같은 IP 의
                                           47092, 시제품 ETV1, C display_relay 필요. 다르면 --view-host/--view-port).
  python -m eyetoarm.gui.window --view window  영상 대신 화면에 떠 있는 Gazebo 창(같은 PC 일 때만)
  python -m eyetoarm.gui.window --view camera  D 임시 단일 카메라 중계(127.0.0.1:47091, gz_view_relay.py 필요)
                                           링크 raw/detect/center 를 누르면 영상 연결(47092)로 SET_DISPLAY_VARIANT 요청을
                                           보낸다(임시 형식). 해당 영상 수신으로 확인하며, 그 종류의 C 프레임이 오기 전에는 수신 대기다
  python -m eyetoarm.gui.window --demo     모의 분류 조작·상태 확인 (아무 것도 보내지 않음)
  python -m eyetoarm.gui.window --tcp-version 1.2  TCP 1.2 후보(명세 부록 A.6, 미승인)로 실제 START/STOP/RECOVER 송수신.
                                           같은 명령 TCP 연결을 쓴다. 현재 C 서버(1.1)는 1.2 를 거부하므로 조작이 열리지 않는다.
                                           이 모드에서 수동 MOVE·CANCEL 은 비활성(1.2 명령 STATUS 규칙 미정)
  python -m eyetoarm.gui.window --no-motion  모션 줄이기 상태로 시작
  python -m eyetoarm.gui.window --theme light  라이트 테마로 시작 (오른쪽 위 「다크 / 라이트」로 실행 중 전환)
  python -m eyetoarm.gui.window --no-cover  표지 화면(UR5e 인사, cover.py) 없이 바로 연다
1일차 진단 CLI(python -m eyetoarm.gui)는 그대로다. 이 창은 별도 진입점이다.

┌─ 화면 구역과 연동 파트 ────────────────────────────────────────────────────┐
│ 구역              │ 지금 표시하는 것              │ 연동 파트 · 계약 상태    │
│───────────────────┼───────────────────────────────┼──────────────────────────│
│ 상태띠 명령 TCP   │ 연결 상태(실제)               │ C · TCP 1.1 확정         │
│ 상태띠 유효STATUS │ 최근 준비 응답 수신 여부(실제)│ C · TCP 1.1 확정         │
│ 상태띠 로봇 준비  │ robot_ready · 사유(실제)      │ C · TCP 1.1 확정         │
│ 상태띠 분류 상태  │ 실제 모드 — / 모의 내부 상태   │ C · 부록 A 미승인        │
│ 상태띠 표시 연결  │ 3채널 각각의 신선도(relay)    │ C 시제품 ETV1(계약 미승인)│
│ 메인·보조 3칸     │ 탑뷰·그리퍼·링크 raw(기본 relay)│ C display_relay · 47092  │
│                   │ 또는 Gazebo 창(--view window)  │ D 임시 경로 · gz_window.py│
│                   │ 또는 단일 중계(--view camera)  │ D 임시 경로 · gz_view.py │
│ 링크 detect/center│ 요청 송신 · 그 종류 프레임 대기│ 영상 수신으로 확인 · 미합의│
│ 측정·보정 상세    │ 실제 — / D 모의 값            │ A·B 결과 → C 중계 · 미결 │
│ 공정 단계 · 수량  │ 실제 미확인 · — / 모의 사건 집계│ C 도착 확인 · 미결       │
│ START/STOP/RECOVER│ 실제 비활성 / 모의에서만 조작  │ C · 부록 A 미승인        │
│ CANCEL(수동 이동) │ 현재 명령 추적 중 활성         │ C · TCP 1.1              │
│ TX/RX 기록        │ 실제 송수신(준비 조회)        │ C · TCP 1.1 확정         │
└────────────────────────────────────────────────────────────────────────────┘
「—」와 비활성은 고장이 아니다. 계약이 정해지지 않아 값을 만들지 않은 자리다.

[공통 상태] 연결 객체(StatusLink 와 화면 수신기 GzWindowCapture/GzViewReceiver)는 창에 하나씩만 있고 세 화면이 함께 쓴다.
  화면을 바꿔도 연결·수신·기록이 초기화되지 않는다(D 상세설계 §A.2).
[보내는 신호] 준비/명령 GET_STATUS, 사용자 확정 수동 MOVE, 기존 ID의 CANCEL. 모의 분류는 TCP를 사용하지 않는다.
  영상 연결(relay)에는 링크 종류 선택 시와 재접속 시 마지막 명시 선택을 SET_DISPLAY_VARIANT로 보낸다.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time

if not __package__:
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PySide6.QtCore import QIODevice, QPoint, QPointF, QRect, QSaveFile, QSettings, Qt, QTimer
from PySide6.QtGui import QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QScrollArea, QStackedWidget,
    QVBoxLayout, QWidget,
)

from eyetoarm.gui import motion, theme
from eyetoarm.gui.client import ACK_METRIC_BOUNDARY
from eyetoarm.gui.gz_view_relay import DEFAULT_PORT as VIEW_PORT
from eyetoarm.gui.relay_view import DEFAULT_PORT as RELAY_PORT
from eyetoarm.gui.widgets import (
    Card, DisplayStage, KeyHint, NavRail, PageZoom, RingProgress, StageStrip, StatusChip, ThemeWipe, TrafficTable, label, pill,
)
from eyetoarm.tcp.protocol import MoveTarget

PAGES = ["조작·설정", "운영 대시보드", "평가·기록"]
NOT_APPROVED = "TCP 명세 부록 A(분류 명령 개정 제안)가 C 승인 전입니다. 승인 전에는 실제 서버로 보내지 않습니다."
NO_VALUE = "—"
ACTION_NAMES = {"START": "START", "STOP": "Stop 정지", "RECOVER": "RECOVER"}
REQUEST_IDLE_11 = "분류·E-Stop 버튼: 누를 수 있으나 미전송 · C TCP 1.2 필요"
REQUEST_IDLE_12 = "분류·E-Stop 버튼: 누름 = 요청 시도 · 준비·정지 확인 아님"
# 분류 START·Stop 정지(STOP)·E-Stop 은 로봇과 컨베이어 벨트를 함께 다룬다(사용자 지정 2026-11-24). 실제 동작은 C 가 정한다
START_HINT = "START: 로봇+컨베이어 분류 시작 요청. C 순서대로 운전하며 벨트 연속 운전을 약속하지 않습니다."
STOP_HINT = "Stop 정지(STOP): 로봇+컨베이어 분류 정지 요청. 부품 해제·정지 완료가 아닙니다(정지 확인은 C 상태로만)."
E_STOP_HINT = ("E-Stop 응용 요청(D 제안 · C 미합의): 로봇+컨베이어 비상 정지를 C 에 요청만 하고 실제 동작은 C 가 정합니다. "
               "현재 C(1.1)는 이 요청을 처리하지 않으며, 이 화면은 안전 장치가 아닙니다.")
CHANNEL_NAMES = {"fixed": "탑뷰", "gripper": "그리퍼", "link": "링크"}


class MainWindow(QMainWindow):
    def __init__(self, link, view, demo: bool = False, store=None) -> None:
        super().__init__()
        self.link, self.view, self.demo = link, view, demo
        self.store = None if demo else store   # 적용한 서버 주소 저장소(QSettings). 없으면 저장하지 않는다
        self.display = None
        # 3채널 표시 모델: 모의 영상(--demo) 또는 C 3채널 중계(--view relay, RelayViewReceiver)
        self.relay = not demo and getattr(view, "is_relay", False) is True
        if demo or self.relay:
            from eyetoarm.gui.display import DisplayModel
            self.display = DisplayModel(self)
        self.mode_buttons: list = []
        self.sort_demo = self.tally = self.supplier = None
        self._sort_logged = None
        # TCP 1.2 후보 실제 송수신(--tcp-version 1.2). 모의(--demo)와 함께 쓰지 않는다
        self.live_run = not demo and getattr(link, "version", None) == "1.2"
        if demo:
            from eyetoarm.gui.sort_demo import DemoProcessSupplier, DemoSortController, SortTally
            self.sort_demo = DemoSortController(self)
            # 5일차 모의 공정: 공급자는 자기 사건으로만 권위 집합을 만들고, 집계는 조작 상태와 따로 둔다
            self.supplier = DemoProcessSupplier()
            self.tally = SortTally(self.supplier.scope)
        self._cycle_timer = QTimer(self, interval=250)   # 모의 1주기 사건 공급. 실제 도착 근거가 아니다
        self._cycle_timer.timeout.connect(self._cycle_step)
        # 송수신 카드 꼬리표. 모의 모드에서는 「실제」라고 적지 않는다
        self._live_tag = "모의 · 실제 아님" if demo else "실제 송수신"
        self.setWindowTitle("Eye To Arms · D 운영 GUI (모의 분류)" if demo else
                            "Eye To Arms · D 운영 GUI (TCP 1.2 후보 · 미승인 · C 1.1 미지원)" if self.live_run else
                            "Eye To Arms · D 운영 GUI (TCP 1.1)")
        self.resize(1440, 960)
        self.setMinimumSize(1180, 760)

        root = QWidget(objectName="Root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── 맨 위: 화면 선택 띠(상표 · 탭 · 출처 · 테마 · 모션 줄이기) ─────
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
        # 테마 전환 효과 선택(번짐 ↔ 잠식). 누를 때마다 바뀌고 다음 실행 때도 유지된다
        self.theme_wipe = ThemeWipe(root)
        self.reduce.toggled.connect(lambda reduced: self.theme_wipe.stop() if reduced else None)
        self.fx_button = pill("", "small", "테마를 바꿀 때의 전환 효과를 고릅니다(번짐 → 잠식 → 타일).")
        styles = list(ThemeWipe.STYLES)
        self.fx_button.clicked.connect(
            lambda: self.set_theme_fx(styles[(styles.index(self.theme_wipe.style) + 1) % len(styles)]))
        theme_row.addWidget(self.fx_button)
        self.set_theme_fx("ink", save=False)
        self.nav.footer.addWidget(self.source, 0, Qt.AlignmentFlag.AlignVCenter)
        self.nav.footer.addLayout(theme_row)
        self.nav.footer.addWidget(self.reduce)
        outer.addWidget(self.nav)

        # ── 아래: 제목줄 · 상태띠 · 화면 ───────────────────────────────
        right = QVBoxLayout()
        right.setContentsMargins(20, 10, 20, 14)
        right.setSpacing(10)
        outer.addLayout(right, 1)
        right.addLayout(self._build_header())
        right.addWidget(self.request_note)
        self.request_note.setVisible(not demo)
        right.addLayout(self._build_status_strip())

        self.traffic_full = TrafficTable()      # 평가·기록 화면의 전체 기록
        self.traffic_recent = TrafficTable()    # 운영 대시보드의 최근 기록(같은 내용을 함께 받는다)
        self.stage = DisplayStage(demo=demo)    # 메인 디스플레이. 창에 하나만 있고 화면 사이를 옮겨 다닌다
        self.pages = QStackedWidget()
        self.page_zoom = PageZoom(self.pages)   # 화면 전환 모션(떠나는 화면의 그림이 줌되며 옅어진다)
        self.key_hint = KeyHint(root)           # 단축키 표시(Ctrl+1~3)
        self._page_cards: list[list[QWidget]] = []
        for build in (self._build_setup_page, self._build_operate_page, self._build_record_page):
            page, cards = build()
            self.pages.addWidget(page)
            self._page_cards.append(cards)
        right.addWidget(self.pages, 1)

        # ── 연결과 화면을 잇는다 ───────────────────────────────────────
        self.nav.selected.connect(self._show_page)
        self.nav.shortcut_used.connect(lambda index: self.key_hint.flash(f"Ctrl+{index + 1}  ·  {PAGES[index]}"))
        self.link.traffic.connect(self.traffic_full.add_entry)
        self.link.traffic.connect(self.traffic_recent.add_entry)
        self.link.status_received.connect(self._on_status)
        if hasattr(self.link, "command_changed"):
            self.link.command_changed.connect(self._on_command)
            self.link.log_error.connect(self._on_log_error)
        self._csv_file = None
        self._export_timer = QTimer(self, interval=0)
        self._export_timer.timeout.connect(self._export_chunk)
        self.view.frame_received.connect(self._on_frame)
        if self.display is not None:
            self.view.display_session.connect(self.display.begin_session)
            self.view.display_frame.connect(self._on_display_frame)
            if hasattr(self.view, "display_result"):   # C 중계는 center 프레임에 실린 B 결과만 보낸다. 결과를 지어내지 않는다
                self.view.display_result.connect(self._on_display_result)
            self.display.changed.connect(self._update_display)
            self.view.state_changed.connect(lambda state: self.display.set_connected(state == "connected"))
        if self.sort_demo is not None:
            self.sort_demo.changed.connect(self._update_sort_controls)
            self.link.state_changed.connect(self._on_demo_connection)
        if not demo:
            # 실제 모드에서는 연결되면 네 버튼을 누를 수 있다. 1.1 은 미전송 사유만 보이고, 1.2 는 계약 검사를 통과할 때만 보낸다
            self.estop_button.clicked.connect(self._request_e_stop)
            self.request_note.setText(REQUEST_IDLE_12 if self.live_run else REQUEST_IDLE_11)
        if self.live_run:
            self.link.run_changed.connect(self._update_live_sort)
            self.link.state_changed.connect(self._update_live_sort)
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
        row.setSpacing(8)
        self.title = label("", "PageTitle")
        row.addWidget(self.title)
        row.addStretch(1)
        # 분류·E-Stop 버튼 누름 결과(송신/미전송 사유). 누를 수 있다는 것은 준비·정지·비상정지 확인이 아니다
        self.request_note = label("", "Caption", wrap=True)   # 제목줄 아래 별도 줄(작은 창에서 버튼을 밀지 않게)
        self.request_note.setAccessibleName("분류·E-Stop 요청 결과")
        self.request_note.setMinimumWidth(0)
        # [C 연동 · 부록 A 미승인] 분류 시작·정지·복구. 계약 승인 뒤 4일차에 연결한다
        self.start_button = pill("START 시작", "primary", NOT_APPROVED + " " + START_HINT)
        self.start_button.setAccessibleDescription(START_HINT)
        # 「Stop 정지」 = 기존 분류 STOP 요청 그대로. 부품 해제·정지 완료를 뜻하지 않는다(정지 확인은 C 상태로만)
        self.stop_button = pill("Stop 정지", "stop", NOT_APPROVED + " " + STOP_HINT + " 수동 이동 취소(CANCEL)와 다릅니다.")
        self.stop_button.setAccessibleDescription(STOP_HINT)
        # E-Stop 자리만 둔다. 저장소에 C 안전(비상정지) 인터페이스가 없어 실제·모의 모두 항상 비활성이며 어떤 신호도 보내지 않는다
        self.estop_button = pill("E-Stop", "stop", "E-Stop 사용 불가 · C 안전 연동 없음(로봇+컨베이어 비상정지 인터페이스 미정의). "
                                 "이 버튼은 아무것도 보내지 않으며 Stop 정지(STOP)으로 대신하지 않습니다.")
        self.estop_button.setAccessibleDescription("사용 불가: C 안전 연동 없음. 로봇·컨베이어에 아무것도 보내지 않습니다.")
        self.recover_button = pill("RECOVER 복구", "", NOT_APPROVED)
        for action, button in (("START", self.start_button), ("STOP", self.stop_button), ("RECOVER", self.recover_button)):
            button.clicked.connect(lambda _checked=False, command=action: self._request_sort(command))
        # [C 연동 · TCP 1.1] 현재 명령 ID의 취소 요청. 정지 확인과 구분한다.
        self.cancel_button = pill("CANCEL 취소 · 미구현", "", "현재 ID로 취소 요청. 전송만으로 정지 완료가 되지 않습니다.")
        self.cancel_button.clicked.connect(lambda: self.link.cancel() if hasattr(self.link, "cancel") else None)
        for button in (self.start_button, self.stop_button, self.estop_button, self.recover_button, self.cancel_button):
            button.setEnabled(False)   # 비활성 사유는 툴팁과 「조작·설정」의 명령 계약 표에 있다. E-Stop 은 다시 켜지 않는다
            row.addWidget(button)
        return row

    # ══ 상태띠: 항목마다 따로 표시한다(연결됨 ≠ 준비됨 ≠ 분류 중) ═══════════

    def _build_status_strip(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
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
        grid.setSpacing(10)

        self.setup_display_slot = QVBoxLayout()   # 이 화면이 보일 때 메인 디스플레이가 여기로 온다
        display_card = Card("영상", "모의 · 실제 아님" if self.demo else "C 3채널 중계 시제품" if self.relay else "D 임시 경로")
        display_card.body.addLayout(self.setup_display_slot, 1)
        channel = QHBoxLayout()
        channel.addWidget(label("링크 채널", "Caption"))
        self._add_mode_buttons(channel)
        channel.addStretch(1)
        display_card.body.addLayout(channel)
        grid.addWidget(display_card, 0, 0, 2, 2)

        link_card = Card("연결", "입력 주소 적용·로컬 저장")
        self.link_value, *_rest, source_value = self._rows(link_card, [
            ("명령 TCP (C 서버)", f"{self.link.host}:{self.link.port}"),
            ("준비 조회 주기", f"{self.link.config['readiness_poll_ms']} ms"),
            ("STATUS 유효 한도", f"{self.link.config['status_watchdog_ms']} ms"),
            ("재접속 간격", f"{self.link.config['reconnect_ms']} ms"),
            ("Gazebo 화면 원본", self.view.title),
        ])
        # 긴 중계 주소(제목)가 카드를 넓히지 않게 줄임표로 맞춘다. 전체 문장은 툴팁에 둔다(_fit_line)
        self.view_value = self._elidable(source_value, self.view.title)
        self._fit_labels = [self.view_value]
        self._build_endpoint_inputs(link_card)
        link_card.body.addStretch(1)
        grid.addWidget(link_card, 0, 2)

        command_card = Card("모의 분류 조작" if self.demo else "명령 계약 상태", "실제 전송 없음" if self.demo else "D ↔ C")
        if self.live_run:
            rows = [
                ("GET_STATUS (준비 조회)", "TCP 1.2 후보 · 실행 상태 포함"),
                ("MOVE · CANCEL", "1.2 후보 모드에서 비활성 · 명세 미정"),
                ("START · Stop 정지(STOP) · RECOVER", "1.2 후보 · 로봇+컨베이어 · C 1.1 은 거부"),
                ("E-Stop (응용 요청)", "미지원 · C 알림 없음"),
                ("링크 채널 선택", "누를 때 영상 연결로 요청 · 임시 형식 · 프레임 도착으로 확인" if self.relay else "부록 B 미승인 · 미전송"),
            ]
            values = self._rows(command_card, rows)
            for value, (_name, text) in zip(values, rows):
                self._fit_labels.append(self._elidable(value, text))
            self.estop_value = values[3]   # E-Stop 지원 여부·현재 결과(C 보고)
            self.sort_detail = label("C 실행 상태 미확인", "Caption", wrap=True)
            self.sort_confirm = QCheckBox("복구 조건을 확인했습니다 (RECOVER 확정)")
            self.sort_confirm.setToolTip("체크한 뒤 RECOVER 를 누를 때만 user_confirmed=true 로 보냅니다. 보낸 뒤 해제됩니다.")
            self.sort_confirm.toggled.connect(self._update_live_sort)
            command_card.body.addWidget(self.sort_detail)
            command_card.body.addWidget(self.sort_confirm)
            command_card.body.addWidget(label("부록 A.6 후보이며 C 승인 전입니다. 실행 상태는 C 스냅샷으로만 표시합니다.", "Caption", wrap=True))
        elif not self.demo:
            rows = [
                ("GET_STATUS (준비 조회)", "사용 중 · TCP 1.1"),
                ("MOVE · CANCEL", "미구현"),
                ("START · STOP · RECOVER", "미전송 · TCP 1.2 필요"),
                ("E-Stop", "미전송 · TCP 1.2 필요"),
                ("링크 채널 선택", "누를 때 영상 연결로 요청 · 임시 형식 · 프레임 도착으로 확인" if self.relay else "부록 B 미승인 · 미전송"),
            ]
            for value, (_name, text) in zip(self._rows(command_card, rows), rows):
                self._fit_labels.append(self._elidable(value, text))   # 1180 폭에서도 칸을 넓히지 않게
        grid.addWidget(command_card, 1, 2)
        if self.demo:
            self.sort_detail = label("모의 분류 상태 대기", "Caption", wrap=True)
            self.sort_confirm = QCheckBox("모의 복구 조건을 확인했습니다")
            self.sort_confirm.toggled.connect(self._update_sort_controls)
            command_card.body.addWidget(self.sort_detail)
            command_card.body.addWidget(self.sort_confirm)
            self.sort_fault_button = pill("모의 보유 상태 불명", tooltip="실제 장치에 영향 없이 복구 잠금 상태를 재현합니다.")
            self.sort_fault_button.clicked.connect(self._inject_demo_recovery)
            self.sort_connection_button = pill("모의 연결 끊기", tooltip="실제 TCP 연결이 없는 내부 대역의 연결 상태를 바꿉니다.")
            self.sort_connection_button.clicked.connect(lambda: self.link.set_connected(self.link.state != "connected"))
            self.sort_cycle_button = pill("모의 1주기", tooltip="모의 RUNNING에서 볼트 A·볼트 B·너트 정상 단계·도착 사건을 내부에서만 재현합니다. 실제 전송 없음.")
            self.sort_cycle_button.clicked.connect(self._start_cycle)
            scenarios = QVBoxLayout()
            scenarios.addWidget(self.sort_fault_button)
            scenarios.addWidget(self.sort_connection_button)
            scenarios.addWidget(self.sort_cycle_button)
            command_card.body.addLayout(scenarios)
            self.process_detail = label("모의 공정 대기", "Caption", wrap=True)   # 인식·로봇·벨트·그리퍼·보유·실패
            command_card.body.addWidget(self.process_detail)
            command_card.body.addWidget(label("실제 모드 분류 명령은 부록 A 승인 대기입니다.", "Caption", wrap=True))
        command_card.body.addStretch(1)

        input_card = Card("입력·보정", "D 모의" if self.demo else "A·B 결과 미수신")
        self.input_values = self._rows(input_card, [("소스 · 모드", NO_VALUE), ("보정", NO_VALUE), ("상태 · 사유", NO_VALUE)])
        input_card.body.addWidget(label("D 내부 모의 값입니다. 실제 A·B 측정값이 아닙니다." if self.demo else "B 값은 C 중계 center 영상에 실려 옵니다(시제품). 그 프레임과 일치할 때만 보이고 그 외에는 비워 둡니다.", "Caption", wrap=True))
        input_card.body.addStretch(1)
        grid.addWidget(input_card, 2, 0)

        target_card = Card("수동 이동", "미구현")
        self.target_inputs = []
        for name, lower, upper in (("X mm", -10000, 10000), ("Y mm", -10000, 10000), ("θ °", 0, 179.999)):
            line = QHBoxLayout()
            line.addWidget(label(name, "Dim"))
            spin = QDoubleSpinBox()
            spin.setRange(lower, upper)
            spin.setDecimals(3)
            spin.setAccessibleName(name)
            spin.valueChanged.connect(self._target_edited)
            self.target_inputs.append(spin)
            line.addWidget(spin)
            target_card.body.addLayout(line)
        self.target_confirm = QCheckBox("수동 시험 목표를 확인했습니다")
        self.target_confirm.setToolTip("A/B 측정값이 아닙니다. C 작업 범위를 확인한 수동 목표만 사용하세요.")
        self.move_button = pill("MOVE 목표 전송", "primary", "새 준비 조회 통과 후 확정한 수동 목표를 보냅니다.")
        self.move_button.clicked.connect(self._execute_manual)
        self.target_confirm.toggled.connect(self._update_command_controls)
        self.command_label = label("명령 없음", "Caption", wrap=True)
        target_card.body.addWidget(self.target_confirm)
        target_card.body.addWidget(self.move_button)
        target_card.body.addWidget(self.command_label)
        self.resolve_button = pill("미확인 기록 보존·새 목표 확인", tooltip="C가 준비됨을 보고한 경우에만 이전 결과 미확인을 보존하고 새 목표를 확인할 수 있습니다. 자동 재송신하지 않습니다.")
        self.resolve_button.clicked.connect(self._resolve_unconfirmed)
        target_card.body.addWidget(self.resolve_button)
        target_card.body.addStretch(1)
        grid.addWidget(target_card, 2, 1)

        shape_card = Card("로봇 외형")
        self._rows(shape_card, [("모델명", "UR5e")])
        shape_card.body.addStretch(1)
        grid.addWidget(shape_card, 2, 2)

        grid.setColumnStretch(0, 4)   # 영상 송출이 주인공이다. 영상 쪽 두 열이 폭의 8/11 을 쓴다(사용자 지정 2026-11-20)
        grid.setColumnStretch(1, 4)
        grid.setColumnStretch(2, 3)   # 오른쪽 카드(연결·조작·로봇 외형)는 좁게
        grid.setRowStretch(0, 1)   # 남는 높이는 영상과 오른쪽 두 카드가 나눠 갖는다
        grid.setRowStretch(1, 1)
        grid.setRowStretch(2, 0)   # 아래 줄(입력·보정 / 수동 명령 목표 / 로봇 외형)은 내용 높이만 쓴다
        # 최소 창(1180x760)에서는 카드 최소 높이 합이 본문보다 커서 글자·입력칸이 잘린다. 줄이지 않고 세로로 넘긴다
        scroll = QScrollArea()
        scroll.setWidget(page)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }")
        return scroll, [display_card, link_card, command_card, input_card, target_card, shape_card]

    # ══ 화면 2: 운영 대시보드 ═══════════════════════════════════════════════

    def _build_operate_page(self) -> tuple[QWidget, list[QWidget]]:
        page = QWidget()
        column = QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(10)

        self.operate_display_slot = QVBoxLayout()
        holder = QWidget()
        holder_layout = QVBoxLayout(holder)
        holder_layout.setContentsMargins(0, 0, 0, 0)
        holder_layout.setSpacing(8)
        tools = QHBoxLayout()
        self.detail_button = pill("측정·보정 상세", checkable=True, tooltip="탑뷰 화면 위에 상세 팝업을 펼칩니다.")
        self.detail_button.toggled.connect(self.stage.popup.toggle)
        # 팝업이 열려 있을 때만 켠다. 닫혀 있으면 Escape 는 입력칸 등 다른 위젯이 그대로 받는다
        self.close_popup_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self.close_popup_shortcut.setEnabled(False)
        self.close_popup_shortcut.activated.connect(self._close_detail_popup)
        self.detail_button.toggled.connect(self.close_popup_shortcut.setEnabled)
        tools.addWidget(self.detail_button)
        if self.relay:   # 링크가 큰 칸이든 보조 칸이든 운영 화면에서도 표시 종류를 고른다
            tools.addWidget(label("링크 표시", "Caption"))
            self._add_mode_buttons(tools)
        view_caption = label(f"탑뷰 원본: {self.view.title}. 검출 주석이 없는 화면입니다.", "Caption")
        self._fit_labels.append(self._elidable(view_caption, view_caption.text()))
        tools.addWidget(view_caption, 1)
        tools.addStretch(1)
        holder_layout.addLayout(tools)
        holder_layout.addLayout(self.operate_display_slot, 1)
        column.addWidget(holder, 1)

        cards = QHBoxLayout()
        cards.setSpacing(10)

        measure_card = Card("현재 측정", "D 모의" if self.demo else "미구현")
        self.measure_values = self._rows(measure_card, [("(u, v) px", NO_VALUE), ("(X, Y) mm", NO_VALUE), ("θ °", NO_VALUE)])
        measure_card.body.addStretch(1)

        stage_card = Card("공정 단계", "모의 · 실제 아님" if self.demo else "미구현")
        stage_card.body.setSpacing(8)
        self.bolt_caption = label("볼트 · 단계 미확인", "Caption")
        self.bolt_caption.setMinimumWidth(40)   # 긴 부품 ID가 카드를 넓히지 않게. 말줄임은 _fit_line
        stage_card.body.addWidget(self.bolt_caption)
        self.bolt_stages = StageStrip(["집기", "운반", "도착", "복귀"])
        self.bolt_stages.setToolTip("종류별 가장 최근 부품 하나의 단계입니다. 각 단계는 C가 확인한 사건이 있을 때만 강조하며 순서를 강제하지 않습니다.\n"
                                    "도착 확인과 팔 복귀는 별개입니다. 같은 부품의 도착 확인이 복귀 뒤에 와도 수량만 늘고 단계는 되돌아가지 않습니다.\n"
                                    "정상 집기에서 벨트를 멈추는 뜻이 아닙니다.")
        stage_card.body.addWidget(self.bolt_stages)
        self.nut_caption = label("너트 · 단계 미확인", "Caption")
        self.nut_caption.setMinimumWidth(40)
        stage_card.body.addWidget(self.nut_caption)
        self.nut_stages = StageStrip(["통과", "도착"])
        stage_card.body.addWidget(self.nut_stages)
        # 전체 분류 상태(상단 「분류 상태」)·부품별 진행(이 카드)·수동 MOVE(TCP 1.1)는 서로 다른 근거다
        self.stage_scope = label("모의 · 동기화 전" if self.demo else "분류 진행 계약 미승인 · 실제 C 미확인", "Caption", wrap=True)
        self.stage_scope.setToolTip("부품별 진행의 출처·범위·세대·사건 경계. 전체 분류 상태는 상단 「분류 상태」, 수동 MOVE는 TCP 1.1 STATUS로 따로 봅니다.")
        stage_card.body.addWidget(self.stage_scope)
        self.stage_note = label("모의 사건으로 확인된 단계만 강조합니다." if self.demo
                                else "C가 확인한 단계만 강조합니다. 지금은 전부 미확인입니다.", "Caption", wrap=True)
        stage_card.body.addWidget(self.stage_note)
        stage_card.body.addStretch(1)

        count_card = Card("수량", "모의 · 미확인" if self.demo else "미구현")
        self.count_tag = count_card.findChild(QLabel, "Tag")
        self.count_values = dict(zip(("BOLT_A", "BOLT_B", "NUT", "QUARANTINE", "EXCEPTION"), self._rows(count_card, [
            ("볼트 A 상자", NO_VALUE), ("볼트 B 상자", NO_VALUE), ("너트 상자", NO_VALUE), ("격리", NO_VALUE),
            ("오분류 / 미판정", NO_VALUE)])))
        self.count_values["EXCEPTION"].setToolTip("예외 도착: 오분류(WRONG_BIN) / 미판정(UNRESOLVED). 정상 상자 수량에는 포함하지 않습니다.")
        self.count_note = label("도착 확인 사건만 셉니다. 팔 복귀와 별개입니다.", "Caption", wrap=True)
        count_card.body.addWidget(self.count_note)
        count_card.body.addStretch(1)

        recent_card = Card("최근 TX/RX", self._live_tag)
        recent_card.body.addWidget(self.traffic_recent, 1)

        for card, stretch in ((measure_card, 2), (stage_card, 4), (count_card, 2), (recent_card, 5)):
            card.setFixedHeight(232)   # 단계 ID·실패 줄과 수량 5줄이 최소 창에서도 잘리지 않는 높이
            cards.addWidget(card, stretch)
        column.addLayout(cards)
        scroll = QScrollArea()
        scroll.setWidget(page)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }")
        return scroll, [holder, measure_card, stage_card, count_card, recent_card]

    # ══ 화면 3: 평가·기록 ═══════════════════════════════════════════════════

    def _build_record_page(self) -> tuple[QWidget, list[QWidget]]:
        page = QWidget()
        row = QHBoxLayout(page)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)

        log_card = Card("TX/RX 기록", "GUI-03 · " + self._live_tag)
        tools = QHBoxLayout()
        filters = []
        self.record_filters = filters
        for text, direction in (("전체", ""), ("TX 보냄", "TX"), ("RX 받음", "RX"), ("EV 연결 사건", "EV")):
            button = pill(text, checkable=True)
            button.setAutoExclusive(True)
            button.toggled.connect(lambda on, d=direction: on and self.traffic_full.set_filter(d))
            filters.append(button)
            tools.addWidget(button)
        filters[0].setChecked(True)
        tools.addStretch(1)
        log_card.body.addLayout(tools)
        tools = QHBoxLayout()   # 동작 버튼은 둘째 줄. 한 줄이면 1180 폭에서 필터 버튼 글자가 잘린다
        tools.addStretch(1)
        clear = pill("화면 비우기", tooltip="화면 표만 비웁니다. 세션 JSONL 파일과 모의 수량 집계는 유지합니다.")
        self.clear_records_button = clear
        clear.clicked.connect(lambda: self.traffic_full.setRowCount(0))
        tools.addWidget(clear)
        self.export_button = pill("CSV 내보내기", tooltip="선택한 현재 화면의 요약 기록을 저장합니다. 전체 원문은 세션 JSONL에 있습니다.")
        self.export_button.clicked.connect(self._choose_export)
        self.export_progress = RingProgress()
        self.export_progress.hide()
        tools.addWidget(self.export_button)
        tools.addWidget(self.export_progress)
        log_card.body.addLayout(tools)
        session_log = getattr(self.link, "log", None)
        log_path = getattr(session_log, "path", None)
        self.log_location = label(f"세션 JSONL: {log_path}" if log_path else "모의/시험: 파일 기록 없음", "Caption", wrap=True)
        self.export_message = label("CSV: 현재 화면의 최대 500건 요약", "Caption", wrap=True)
        self.log_notice = label(getattr(session_log, "write_error", ""), "Caption", wrap=True)
        log_card.body.addWidget(self.log_location)
        log_card.body.addWidget(self.log_notice)
        log_card.body.addWidget(self.export_message)
        log_card.body.addWidget(self.traffic_full, 1)
        row.addWidget(log_card, 5)

        side = QVBoxLayout()
        side.setSpacing(10)
        metric_card = Card("평가", "측정 전")
        self.ack_metric_value, _rate, self.reach_error_value = self._rows(
            metric_card, [("사용자 실행 → ACK 지연", NO_VALUE), ("명령 성공률", NO_VALUE), ("도달 오차", NO_VALUE)])
        # 현재 추적 MOVE 의 C 보고 reach_error 만 쓴다. D 가 계산하지 않는다(TCP 1.1 §4.6)
        self.reach_error_value.setToolTip("C 가 STATUS reach_error 로 보낸 현재 MOVE 의 위치·회전 오차. 없으면 —")
        self.ack_metric_note = label("모의: 실제 ACK 없음 · 측정 안 함" if self.demo else "명령 없음", "Caption", wrap=True)
        tip = "모의 분류는 TCP 를 쓰지 않아 측정하지 않습니다." if self.demo else ACK_METRIC_BOUNDARY
        self.ack_metric_value.setToolTip(tip)
        self.ack_metric_note.setToolTip(tip)
        metric_card.body.addWidget(self.ack_metric_note)
        metric_card.body.addWidget(label("정답·관측이 없는 지표는 계산하지 않고 —로 둡니다.", "Caption", wrap=True))
        side.addWidget(metric_card)
        error_card = Card("명령·오류 상세", "TCP 1.1")
        self.error_detail = label("명령 없음", "Dim", wrap=True)
        error_card.body.addWidget(self.error_detail)
        error_card.body.addStretch(1)
        side.addWidget(error_card, 1)
        row.addLayout(side, 2)
        return page, [log_card, metric_card, error_card]

    def _build_endpoint_inputs(self, card: Card) -> None:
        """서버 주소 입력. 적용 버튼을 누를 때만 연결을 바꾼다. 영상 IP 빈칸은 명령 IP를 따른다."""
        relay_port = getattr(self.view, "port", RELAY_PORT) if self.relay else RELAY_PORT
        view_host = getattr(self.view, "host", "") if self.relay else ""
        self.endpoint_edits = {}
        for key, name, value, hint, width in (
                ("host", "명령 IP", self.link.host, "숫자 IP", 80), ("port", "포트", str(self.link.port), "47090", 56),
                ("view_host", "영상 IP", "" if view_host == self.link.host else view_host, "빈칸 = 명령 IP", 80),
                ("view_port", "포트", str(relay_port), str(RELAY_PORT), 56)):
            if key in ("host", "view_host"):
                line = QHBoxLayout()
                card.body.addLayout(line)
            edit = QLineEdit(value)
            edit.setPlaceholderText(hint)
            edit.setAccessibleName(name if key in ("host", "view_host") else f"{'명령' if key == 'port' else '영상'} {name}")
            if key in ("host", "view_host"):
                edit.setMinimumWidth(width)   # 남는 폭은 IP 칸이 쓴다. 카드(열) 폭을 넓히지 않는다
                edit.setMaxLength(45)         # IPv6 문자열 최대 길이
            else:
                edit.setFixedWidth(width)
                edit.setMaxLength(5)
            edit.returnPressed.connect(self._apply_endpoint)
            self.endpoint_edits[key] = edit
            line.addWidget(label(name, "Dim"))
            line.addWidget(edit, 1 if key in ("host", "view_host") else 0)
        self.endpoint_keep = QCheckBox("결과 미확인 보존 후 다른 서버로 전환")
        self.endpoint_keep.setToolTip("이전 서버가 끊겨 결과를 확인할 수 없을 때만 씁니다. 그 명령은 미확인으로 기록에 남고 다시 보내지 않습니다.")
        card.body.addWidget(self.endpoint_keep)
        self.endpoint_button = pill("적용·재접속", tooltip="입력한 주소를 검사한 뒤 지금 연결을 끊고 그 주소로 다시 접속합니다. 앱을 다시 켤 필요가 없습니다.")
        self.endpoint_button.clicked.connect(self._apply_endpoint)
        card.body.addWidget(self.endpoint_button)
        self.endpoint_note = label("", "Caption", wrap=True)
        card.body.addWidget(self.endpoint_note)
        if self.demo:
            note = "모의 모드: 실제 서버 주소를 쓰지 않습니다(소켓 없음)."
        elif not self.relay:
            note = "영상 주소 입력은 C 3채널 중계(--view relay)에서만 씁니다."
        else:
            note = ""
        for key, edit in self.endpoint_edits.items():
            edit.setEnabled(not self.demo and (self.relay or key in ("host", "port")))
        self.endpoint_button.setEnabled(not self.demo)
        self.endpoint_keep.setEnabled(not self.demo)
        self.endpoint_note.setText(note)

    def _apply_endpoint(self) -> None:
        """입력 검사 → 명령 진행 검사 → 명령·영상 재접속 → 저장. 앞 단계에서 막히면 아무것도 바꾸지 않는다."""
        if self.demo:
            return
        from eyetoarm.gui.client import endpoint_error

        def port_of(key: str) -> int | None:
            text = self.endpoint_edits[key].text().strip()
            return int(text) if text.isascii() and text.isdigit() else None

        host, port = self.endpoint_edits["host"].text().strip(), port_of("port")
        error = endpoint_error(host, port) if port is not None else "명령 포트는 1~65535 정수"
        view_host, view_port = "", None
        if not error and self.relay:
            view_host = self.endpoint_edits["view_host"].text().strip()
            view_port = port_of("view_port")
            error = (endpoint_error(view_host or host, view_port) if view_port is not None
                     else "영상 포트는 1~65535 정수")
            error = f"영상: {error}" if error else ""
        # 명령·영상 어느 쪽도 바꾸기 전에 막는다(명령 결과를 지키는 쪽이 우선)
        keep = self.endpoint_keep.isChecked()
        error = (error or self.link.endpoint_busy(host, port, keep)
                 or self.link.set_endpoint(host, port, keep_unconfirmed=keep))
        if error:
            self.endpoint_note.setText(f"적용 안 함 · {error}. 지금 연결과 저장값은 그대로입니다.")
            return
        self.endpoint_keep.setChecked(False)   # 한 번의 전환에만 쓴다
        self._status = None   # 이전 서버의 준비 상태를 새 연결의 값처럼 보이지 않게 한다
        if self.relay:
            if self.view.set_endpoint(view_host or host, view_port):
                # 다른 서버: 이전 서버의 run_id·순번·결과 기록을 버린다. 같은 주소 재접속은 보존한다
                self.display.reset_session("주소 변경 · 새 서버 영상 대기")
            self.view_value._full = self.view.title
            self.view_value.setToolTip(self.view.title)
        self.link_value.setText(f"{self.link.host}:{self.link.port}")
        saved = "저장 안 함(이 실행만)"
        if self.store is not None:
            values = {"host": host, "port": port}
            if self.relay:
                values.update(view_host=view_host, view_port=view_port)
            for key, value in values.items():
                self.store.setValue(f"endpoint/{key}", value)
            self.store.sync()
            saved = "저장함" if self.store.status() == QSettings.Status.NoError else "저장 실패(다음 실행에는 이전 주소)"
        # 요청만 했다는 뜻이다. 실제 수신·로봇 준비는 상태띠(명령 TCP·유효 STATUS·표시 연결)로 확인한다
        self.endpoint_note.setText(f"연결 요청함 · 명령 {self.link.host}:{self.link.port}"
                                   + (f" · 영상 {self.view.host}:{self.view.port}" if self.relay else "")
                                   + f" · {saved}. 수신·준비 여부는 상태띠에서 확인")
        self._update_status()

    @staticmethod
    def _rows(card: Card, rows: list[tuple[str, str]]) -> list[QLabel]:
        """「이름 ……… 값」 줄을 카드에 넣고 값 라벨을 돌려준다(동적 갱신용)."""
        values = []
        for name, value in rows:
            line = QHBoxLayout()
            line.addWidget(label(name, "Dim"))
            line.addStretch(1)
            values.append(label(value, "CardTitle" if value != NO_VALUE else "Dim"))
            line.addWidget(values[-1])
            card.body.addLayout(line)
        return values

    # ══ 화면 전환 ═══════════════════════════════════════════════════════════

    def _target_edited(self, *_args) -> None:
        if hasattr(self, "target_confirm"):
            self.target_confirm.setChecked(False)
            if hasattr(self.link, "invalidate_target"):
                self.link.invalidate_target("수동 목표 변경")

    def _update_command_controls(self, *_args) -> None:
        available = not self.demo and hasattr(self.link, "can_execute") and self.link.can_execute()
        self.move_button.setEnabled(bool(available and self.target_confirm.isChecked()))
        record = getattr(self.link, "current_record", None)
        self.cancel_button.setEnabled(bool(not self.demo and self.link.state == "connected" and record
                                           and record.attempts and not record.finished))
        self.resolve_button.setEnabled(bool(not self.demo and record and not record.finished
                                            and record.acceptance == "unconfirmed" and not self.link.is_stale()
                                            and (getattr(self.link, "last_status", None) or {}).get("robot_ready")))

    def _resolve_unconfirmed(self) -> None:
        if self.resolve_button.isEnabled() and self.link.confirm_new_command():
            self.target_confirm.setChecked(False)

    def _execute_manual(self) -> None:
        if not self.move_button.isEnabled():
            return
        x, y, theta = (spin.value() for spin in self.target_inputs)
        now = time.monotonic_ns() // 1_000_000
        target = MoveTarget((x, y), theta, "manual_gui", 0, now, "PC_RECEIVE", "manual_test", created_monotonic_ms=now)
        self.link.execute(target)
        self.target_confirm.setChecked(False)
        self._update_command_controls()

    def _on_command(self, record) -> None:
        reach = getattr(record, "reach_error", None)   # 새 명령이면 None 이라 이전 명령 값이 남지 않는다
        reach_text = (NO_VALUE if not reach else   # 0 은 유효값이므로 is None 으로만 비운다
                      f"{NO_VALUE if reach.get('position_mm') is None else reach['position_mm']} mm · "
                      f"{NO_VALUE if reach.get('rotation_deg') is None else reach['rotation_deg']} °")
        text = (f"ID: {record.command_id}\n{record.result}\n"
                f"state={record.state or '—'} / stage={record.stage or '—'}\n"
                f"code={record.code or '—'} / 시도={record.attempts}\n"
                f"reach_error={reach_text}\n{record.detail or ''}")
        self.command_label.setText(text)
        self.error_detail.setText(text)
        self.reach_error_value.setText(reach_text)
        self.reach_error_value.setObjectName("Dim" if not reach else "CardTitle")
        self.reach_error_value.style().polish(self.reach_error_value)
        latency, reason = record.ack_metric()
        self.ack_metric_value.setText(NO_VALUE if latency is None else f"{latency} ms")
        self.ack_metric_value.setObjectName("Dim" if latency is None else "CardTitle")
        self.ack_metric_value.style().polish(self.ack_metric_value)
        self.ack_metric_note.setText(f"{record.command_id[:8]} · {reason or '같은 연결 첫 송신의 첫 ACK'}")
        self._update_command_controls()

    def _add_mode_buttons(self, row) -> None:
        """링크 raw/detect/center 선택. 모의와 C 중계에서만 켠다. 중계면 누를 때 영상 연결로 요청을 보낸다."""
        for mode, name in (("raw", "raw 원본"), ("detect", "detect 검출"), ("center", "center 중심점")):
            tip = ("D 내부 모의 영상 선택" if self.demo else
                   "영상 연결로 C 에 링크 표시 요청을 보냅니다(임시 형식, 해당 영상 수신으로 확인). 그 종류 영상이 오기 전에는 수신 대기로 보입니다."
                   if self.relay else "C 표시 연결 형식 미결")
            button = pill(name, tooltip=tip, checkable=True)   # 눌린 상태가 지금 고른 표시 종류다
            button.setEnabled(self.display is not None)
            button.setChecked(self.display is not None and self.display.mode == mode)
            button._mode = mode
            if self.display is not None:
                button.clicked.connect(lambda _checked=False, m=mode: self._select_link_mode(m))
            row.addWidget(button)
            self.mode_buttons.append(button)

    def _select_link_mode(self, mode: str) -> None:
        if self.relay and mode != self.display.mode:
            # 실제 전환: 이전 링크 프레임·순번·확인 기록을 버린다. 고른 종류의 새 패킷이 와야 다시 보인다
            # (종류마다 순번·세대·설정이 따로라 이전 기록이 새 종류를 막지 않게도 한다). 보조 칸은 그대로다
            self.view.reset_link()
            if self.display.link_source_id:
                self.display.reset_source(self.display.link_source_id, f"링크 {mode} 선택 · 해당 영상 대기")
        self.display.set_mode(mode)   # 화면은 고른 종류만 보이고, 그 영상이 올 때까지 수신 대기다
        if self.relay:
            self.view.request_variant(mode)
        for button in self.mode_buttons:   # 조작·설정과 운영 화면의 선택을 같게 맞춘다
            button.setChecked(button._mode == mode)

    def _close_detail_popup(self) -> None:
        if self.detail_button.isChecked():
            self.detail_button.setChecked(False)
            self.detail_button.setFocus(Qt.FocusReason.ShortcutFocusReason)

    def _on_log_error(self, text: str) -> None:
        self.log_notice.setText(f"로그 저장 오류: {text}" if text else "")

    def _choose_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "현재 화면 기록 저장", "records.csv", "CSV (*.csv)")
        if path:
            self._start_export(path)

    def _start_export(self, path: str) -> bool:
        if self._csv_file is not None:
            return False
        table = self.traffic_full
        self._csv_rows = []
        for row in range(table.rowCount()):
            if not table.isRowHidden(row):
                values = [table.item(row, col).text() for col in range(4)]
                values.append(table.item(row, 0).toolTip())
                self._csv_rows.append(values)
        output = QSaveFile(path, self)
        if not output.open(QIODevice.OpenModeFlag.WriteOnly):
            self.export_message.setText(f"CSV 저장 실패: {output.errorString()}")
            output.deleteLater()
            return False
        self._csv_file = output
        self._csv_index = 0
        self._csv_header = True
        self._export_path = path
        self._export_lock(True)
        self.export_progress.set_value(0)
        self.export_progress.show()
        self.export_message.setText("현재 화면 요약 CSV 생성 중 · 원문 전체는 JSONL 참조")
        self._export_timer.start()
        return True

    def _export_lock(self, busy: bool) -> None:
        for widget in [*self.record_filters, self.clear_records_button, self.export_button]:
            widget.setEnabled(not busy)

    def _export_chunk(self) -> None:
        if self._csv_file is None:
            return
        try:
            buffer = io.StringIO(newline="")
            writer = csv.writer(buffer, lineterminator="\r\n")
            if self._csv_header:
                buffer.write("\ufeff")
                writer.writerow(["시각", "방향", "종류", "내용", "원문 요약"])
                self._csv_header = False
            end = min(self._csv_index + 50, len(self._csv_rows))
            for row in self._csv_rows[self._csv_index:end]:
                writer.writerow(["'" + cell if cell.lstrip().startswith(("=", "+", "-", "@"))
                                 or cell.startswith(("\t", "\r")) else cell for cell in row])
            data = buffer.getvalue().encode("utf-8")
            if self._csv_file.write(data) != len(data):
                raise OSError(self._csv_file.errorString())
            self._csv_index = end
            self.export_progress.set_value(int(100 * end / max(1, len(self._csv_rows))))
            if end == len(self._csv_rows):
                if not self._csv_file.commit():
                    raise OSError(self._csv_file.errorString())
                self.export_progress.set_value(100)
                self.export_message.setText(f"CSV 저장 완료: {self._export_path} ({end}건)")
                self._finish_export()
        except (OSError, ValueError) as error:
            self._csv_file.cancelWriting()
            self.export_message.setText(f"CSV 저장 실패: {error}")
            self.export_progress.hide()
            self._finish_export()

    def _finish_export(self) -> None:
        self._export_timer.stop()
        self._csv_file.deleteLater()
        self._csv_file = None
        self._export_lock(False)

    def _show_page(self, index: int, animate: bool = True) -> None:
        """본문은 즉시 바뀐다. 떠나는 화면의 그림이 줌되며 옅어지고, 새 화면의 카드가 차례로 등장한다. 연결·기록은 건드리지 않는다."""
        previous = self.pages.currentIndex()
        if index != previous:
            self.theme_wipe.stop()
        # 떠나는 화면을 그림으로 찍어 둔다(영상 칸을 옮기기 전에). 새 화면은 그 아래에서 바로 살아 있다
        snapshot = (self.pages.grab() if animate and index != previous and self.pages.isVisible() and not motion.is_reduced()
                    else None)
        if index != 1 and self.detail_button.isChecked():
            self.detail_button.setChecked(False)
        # 메인 디스플레이는 하나뿐이다. 영상 칸이 있는 화면으로 옮긴다(수신은 끊기지 않는다)
        slot = {0: self.setup_display_slot, 1: self.operate_display_slot}.get(index)
        if slot is not None and slot.indexOf(self.stage) < 0:
            slot.addWidget(self.stage)
        self.pages.setCurrentIndex(index)
        if snapshot is not None:
            self.page_zoom.play(snapshot, forward=index > previous)
        self.title.setText(PAGES[index])
        self.nav.select(index, animate)
        if animate:
            motion.stagger(self._page_cards[index], theme.MS_PAGE, theme.MS_STAGGER)

    def play_entrance(self) -> None:
        """표지가 닫힌 뒤 한 번: 상태띠 칸과 지금 화면의 카드가 차례로 떠오른다. 값·연결은 건드리지 않는다."""
        chips = [self.chip_tcp, self.chip_view, self.chip_status, self.chip_ready, self.chip_sort, self.chip_age]
        motion.stagger(chips + self._page_cards[self.pages.currentIndex()], theme.MS_PAGE, theme.MS_STAGGER)

    # ══ 테마 ═══════════════════════════════════════════════════════════════

    def set_theme(self, name: str, save: bool = True) -> None:
        """다크·라이트 전환. 연결·수신·기록은 건드리지 않는다. 표시 문구와 값은 그대로이고 색만 바뀐다.

        1) theme.apply 로 색 상수와 QSS 를 바꾼다
        2) 앱 전체 스타일시트를 다시 건다(QSS 로 꾸민 위젯)
        3) 색을 보관해 둔 위젯(refresh_theme 가 있는 것)을 맞추고, 직접 그리는 위젯을 다시 그린다
        """
        previous = theme.name
        root = self.centralWidget()
        # 옛 테마의 창 그림. 전환 효과(ThemeWipe)가 새 테마 위에 잠깐 겹쳐 쓴다
        snapshot = root.grab() if name != previous and self.isVisible() and not motion.is_reduced() else None
        theme.apply(name)
        QApplication.instance().setStyleSheet(theme.QSS)
        if snapshot is not None and theme.name != previous:
            button = self.theme_buttons[theme.name]
            keep = QRect(self.stage.mapTo(root, QPoint(0, 0)), self.stage.size()) if self.stage.isVisible() else None
            self.theme_wipe.play(snapshot, QPointF(button.mapTo(root, button.rect().center())), theme.name == "dark", keep)
        for widget in self.findChildren(QWidget):
            if hasattr(widget, "refresh_theme"):
                widget.refresh_theme()
            widget.update()
        self.theme_buttons[theme.name].setChecked(True)
        if save:
            settings().setValue("theme", theme.name)

    def set_theme_fx(self, style: str, save: bool = True) -> None:
        """테마 전환 효과를 고른다(ink 번짐 · venom 잠식 · tiles 타일). 모르는 이름은 ink."""
        self.theme_wipe.style = style if style in ThemeWipe.STYLES else "ink"
        self.fx_button.setText(f"전환: {ThemeWipe.STYLES[self.theme_wipe.style]}")
        if save:
            settings().setValue("theme_fx", self.theme_wipe.style)

    # ══ 값 반영 ═════════════════════════════════════════════════════════════

    def _on_demo_connection(self, state):
        self.sort_demo.set_connected(state == "connected")
        if state != "connected":
            self.sort_confirm.setChecked(False)
            return
        snapshot = self.sort_demo.snapshot or dict(epoch=1, seq=0, state="IDLE", holding="DETACHED", job_id=None, detail="모의 초기 상태")
        if snapshot["state"] in {"STARTING", "RUNNING", "STOPPING", "RECOVERING"}:
            snapshot.update(state="RECOVERY_REQUIRED", holding="HOLD_UNKNOWN", detail="모의 재접속 · 실행 결과 불명 · 복구 확인 필요")
        snapshot["seq"] += 1
        self.sort_demo.apply_snapshot(snapshot, resync=True)

    def _request_sort(self, action):
        name = ACTION_NAMES[action]
        if self.live_run:
            reason = self._sort_block(action)
            if reason:
                self._note(f"{name} 미전송 · {reason}")
            elif self.link.sort_request(action, confirmed=action == "RECOVER" and self.sort_confirm.isChecked()):
                if action == "RECOVER":
                    self.sort_confirm.setChecked(False)   # 확정은 한 번의 송신에만 쓴다
                self._note(f"{name} 요청 시도 · 소켓 기록·결과는 TX/RX 기록과 C 보고로(준비·정지 확인 아님)")
            else:
                self._note(f"{name} 미전송 · 송신 조건 불충족")
            self._update_live_sort()
            return
        if self.sort_demo is None:
            if not self.demo:   # 1.1 실제 모드: 보내지 않고 이유만 보인다
                self._note(f"{name} 미전송 · C TCP 1.2 필요(--tcp-version 1.2 와 C 지원)")
            return
        confirmed = action != "RECOVER" or self.sort_confirm.isChecked()
        self.sort_demo.request(action, confirmed=confirmed)
        if action == "RECOVER":
            self.sort_confirm.setChecked(False)

    def _inject_demo_recovery(self):
        snapshot = self.sort_demo.snapshot
        if snapshot and self.sort_demo.connected and self.sort_demo.synchronized:
            snapshot.update(seq=snapshot["seq"] + 1, state="RECOVERY_REQUIRED", holding="HOLD_UNKNOWN", detail="모의 보유 상태 불명 · 시작 잠금")
            self.sort_confirm.setChecked(False)
            self.sort_demo.apply_snapshot(snapshot)

    # ── 5일차 모의 공정·집계 ──────────────────────────────────────────────

    def _start_cycle(self) -> bool:
        if self.sort_demo is None or not self.sort_cycle_button.isEnabled() or not self.supplier.start_cycle():
            return False
        self.link.record_event("EV", "모의 공정", f"모의 · 실제 아님 · 범위 {self.supplier.scope} · 1주기 시작(볼트 A·볼트 B·너트)")
        self._cycle_timer.start()
        self._update_sort_controls()
        return True

    def _cycle_step(self) -> None:
        snapshot = self.sort_demo.snapshot if self.sort_demo is not None else None
        if snapshot is None or not (self.sort_demo.connected and self.sort_demo.synchronized and snapshot["state"] == "RUNNING"):
            self._stop_cycle("RUNNING 아님")
            return
        event = self.supplier.next_event(snapshot["epoch"])
        if event is not None:
            self.feed_demo_event(event)
        if not self.supplier.active:
            self._cycle_timer.stop()
            self.link.record_event("EV", "모의 공정", f"모의 · 실제 아님 · 범위 {self.supplier.scope} · 1주기 사건 공급 끝")
            self._update_sort_controls()

    def _stop_cycle(self, reason: str) -> None:
        """추가 시나리오만 멈춘다. 조작 상태·권한·수량은 건드리지 않는다. 남은 사건은 만들지 않는다."""
        self._cycle_timer.stop()
        if self.supplier is not None and self.supplier.active:
            dropped = self.supplier.abort()
            self.tally.mark_stage_unknown()
            self.link.record_event("EV", "모의 공정", f"모의 · 실제 아님 · 1주기 중단 · {reason} · 남은 사건 {dropped}건 미생성 · 단계 미확인")

    def feed_demo_event(self, event):
        """모의 사건 하나를 집계에 넣고 처분을 기록한다. 실제 모드에는 연결하지 않는다."""
        if self.tally is None:
            return None
        outcome = self.tally.process(event)
        get = event.get if isinstance(event, dict) else (lambda _key, default="?": default)
        self.link.record_event("EV", "모의 공정", " · ".join(str(part) for part in (
            "모의 · 실제 아님", f"범위 {get('scope', '?')}", f"세대 {get('epoch', '?')}", f"#{get('seq', '?')}", get("part", "?"),
            get("kind", "?"), get("spec", get("reason", "")) or "-", outcome.code, outcome.note)))
        self._update_process_view()
        return outcome

    def _sync_tally(self) -> None:
        outcome = self.tally.apply_snapshot(self.supplier.snapshot(self.sort_demo.snapshot["epoch"]))
        self.link.record_event("EV", "모의 집계", f"모의 · 실제 아님 · 범위 {self.supplier.scope} · 집계 스냅샷 {outcome.code} · {outcome.note}")

    def _update_process_view(self) -> None:
        tally, controller = self.tally, self.sort_demo
        snapshot = controller.snapshot
        trusted = controller.connected and controller.synchronized and tally.synchronized
        show_stage = trusted and tally.stage_known and snapshot is not None and snapshot["state"] != "RECOVERY_REQUIRED"
        for group, caption, strip in (("bolt", self.bolt_caption, self.bolt_stages), ("nut", self.nut_caption, self.nut_stages)):
            current = tally.display[group]
            name = "볼트" if group == "bolt" else "너트"
            if show_stage and current is not None:
                final = tally.finals.get(current["part"])   # 이 부품의 최종 범주만. 다른 부품의 도착으로 추정하지 않음
                arrival = ("도착 미확인" if final is None else "도착 확인" if final in ("BOLT_A", "BOLT_B", "NUT") and final == current["spec"]
                           else {"UNRESOLVED": "미판정 도착", "WRONG_BIN": "오분류 도착", "QUARANTINE": "격리"}.get(final, final))
                outcome = current.get("outcome")   # 같은 부품의 실패·격리. 단계는 미확인으로 둔다
                caption._full = f"{name} {current['part']} · {current['spec']} · " + (outcome.split(" · ")[0] if outcome else arrival)
                caption.setToolTip(f"모의 사건 · {current['part']} · {current['spec']} · {outcome or arrival}")
                strip.set_stage(current["stage"])
            else:
                caption._full = f"{name} · 단계 미확인" + (f" · 마지막 {current['part']}" if current else "")
                caption.setToolTip(caption._full)
                strip.set_stage(None)
            self._fit_line(caption)
        counts = tally.counts()
        for key, value in self.count_values.items():
            value.setText(f"{counts['WRONG_BIN']} / {counts['UNRESOLVED']}" if key == "EXCEPTION" else str(counts[key]))
        if not trusted:
            self.count_tag.setText("모의 · 마지막 값 · 미확인")
        else:
            self.count_tag.setText("모의 · 불일치 · 집계 미확인" if tally.conflict else "모의 · 동기화됨")
        failures = tally.failure_counts()
        failure_text = ("실패: 최근 " + (tally.last_failure or "—") + " · 사유별 "
                        + ", ".join(f"{reason} {count}" for reason, count in sorted(failures.items()))) if failures else "실패: 없음"
        summary = f"실패 {sum(failures.values())}건 · 사유 {len(failures)}종" if failures else "실패: 없음"
        self.stage_note.setText(summary if trusted else "단계 미확인 · " + summary)
        self.stage_note.setToolTip(failure_text)
        fields = {key: (value or NO_VALUE) if show_stage else "미확인" for key, value in tally.fields.items()}
        holding = snapshot["holding"] if controller.connected and controller.synchronized and snapshot else "미확인"
        self.process_detail.setText(f"모의 공정 · 인식 {fields['recognition']} · 로봇 {fields['robot']} · 벨트 {fields['belt']}"
                                    f" · 그리퍼 {fields['gripper']} · 보유 {holding}\n{failure_text}")
        # 운영 화면에도 같은 출처·상태·실패를 둔다(조작·설정 화면에만 있지 않게)
        if not controller.connected:
            scope = "모의 · 연결 끊김 · 단계 미확인"
        elif not trusted or snapshot is None:
            scope = "모의 · 동기화 전 · 단계 미확인"
        elif snapshot["state"] == "RECOVERY_REQUIRED":
            scope = "모의 · 복구 필요 · 단계 미확인"
        else:
            scope = f"모의 · 마지막 확인 단계 · 세대 {tally.epoch} · 경계 {tally.boundary}"
        self.stage_scope.setText(scope)
        self.stage_scope.setToolTip(f"모의 사건 출처 · 범위 {tally.scope} · 세대 {tally.epoch} · 스냅샷 경계 {tally.boundary}\n"
                                    f"{self.process_detail.text()}\n"
                                    "전체 분류 상태는 상단 「분류 상태」, 수동 MOVE는 TCP 1.1 STATUS로 따로 봅니다.")

    def _update_sort_controls(self):
        if self.sort_demo is None:
            return
        controller = self.sort_demo
        snapshot = controller.snapshot
        synced = controller.connected and controller.synchronized and snapshot is not None
        if synced and snapshot["epoch"] != self.tally.epoch:
            self._stop_cycle("세대 전환")
            self.tally.desync()
        running = synced and snapshot["state"] == "RUNNING"
        if self.supplier.active and not running:
            self._stop_cycle("단절" if not controller.connected else ("동기화 상실·세대 전환" if not synced else f"상태 {snapshot['state']}"))
        if not synced:
            self.tally.desync()
        elif not self.tally.synchronized:
            self._sync_tally()
        self.sort_cycle_button.setEnabled(running and self.tally.synchronized and not self.supplier.active)
        self._update_process_view()
        for action, button in (("START", self.start_button), ("STOP", self.stop_button), ("RECOVER", self.recover_button)):
            allowed, reason = controller.allowed(action)
            if action == "RECOVER" and not self.sort_confirm.isChecked():
                allowed, reason = False, "복구 조건 확인란을 선택하세요."
            button.setEnabled(allowed)
            button.setToolTip("모의 · 실제 전송 없음. " + reason)
        self.sort_fault_button.setEnabled(controller.connected and controller.synchronized)
        self.sort_connection_button.setText("모의 연결 끊기" if controller.connected else "모의 재접속")
        if not controller.connected or not controller.synchronized or snapshot is None:
            self.chip_sort.set_state("warn", "모의 · 상태 미확인")
            self.sort_detail.setText("모의 연결·스냅샷 동기화 전에는 분류 조작을 허용하지 않습니다.")
            return
        state = snapshot["state"]
        self.chip_sort.set_state("warn" if state in {"STOPPING", "RECOVERY_REQUIRED", "RECOVERING"} else "accent", "모의 · " + state)
        self.sort_detail.setText(f"모의 · {state}\n보유: {snapshot['holding']}\n{snapshot['detail']}")
        key = (snapshot["epoch"], snapshot["seq"])
        if key != self._sort_logged:
            self._sort_logged = key
            self.link.record_event("EV", "모의 분류", f"실제 전송 없음 · {state} · 보유 {snapshot['holding']} · {snapshot['detail']}")

    RUN_LABELS = {"IDLE": "대기", "RUNNING": "분류 실행 중", "STOP_REQUESTED": "정지 요청 접수 — 정지 확인 전",
                  "STOPPED": "정지 확인됨", "RECOVERY_REQUIRED": "복구 필요", "RECOVERING": "복구 진행 중 — 자동 재개 없음",
                  "UNAVAILABLE": "C 실행 관리 없음"}

    def _update_live_sort(self, *_args) -> None:
        """TCP 1.2 후보: 버튼은 link.sort_allowed(스냅샷 allowed·STOP 예외), 상태 문구는 C 스냅샷으로만 쓴다."""
        if not self.live_run:
            return
        link = self.link
        connected = link.state == "connected"
        plans = []   # 누르기 전에 보이는 「지금 누르면」 결과. 누를 수 있음 ≠ 보냄 ≠ 준비·정지 확인
        for action, button in (("START", self.start_button), ("STOP", self.stop_button), ("RECOVER", self.recover_button)):
            reason = self._sort_block(action)
            button.setEnabled(connected)
            hint = {"START": START_HINT, "STOP": STOP_HINT}.get(action, "")
            button.setToolTip("TCP 1.2 후보(부록 A.6, 미승인) · " + (hint + " · " if hint else "")
                              + ("누르면 송신" if not reason else "누르면 미전송 · " + reason))
            plans.append(f"{ACTION_NAMES[action]} " + ("송신" if not reason else f"미전송({reason})"))
        allowed, reason = link.e_stop_allowed()
        self._estop_face(blocked=connected and not allowed)
        plans.append("E-Stop " + ("송신" if allowed else f"미전송({reason})"))   # 지난 결과(아래 줄)와 별개로 지금 가능 여부
        self.estop_button.setEnabled(connected)
        self.estop_button.setToolTip(E_STOP_HINT + " · " + ("누르면 C 에 요청 송신" if allowed else "누르면 미전송 · " + reason))
        e_stop = link.e_stop
        if e_stop is not None:   # C 보고와 D 판단(기한·끊김 = 결과 미확인)을 구분하고, 지난 연결 기록은 그렇게 적는다
            if e_stop.state in ("CONFIRMED", "REJECTED", "UNCONFIRMED"):
                status = f"C 보고 {e_stop.state}"
            elif e_stop.final:
                status = "결과 미확인" + (" (C 접수 REQUESTED 뒤)" if e_stop.state == "REQUESTED" else "")
            elif e_stop.state == "REQUESTED":
                status = "C 보고 REQUESTED · 접수만"
            else:
                status = "응답 대기"
            status = ("이전 연결 · " if e_stop.result.startswith("이전 연결") else "") + status
        else:
            status = "요청 가능 · C 알림" if allowed else f"요청 불가 · {reason}"
        self.estop_value._full = status
        self._fit_line(self.estop_value)
        self.estop_button.setAccessibleDescription(E_STOP_HINT)
        snapshot = None if link.is_stale() else link.run_snapshot
        if link.state != "connected":
            kind, text = "fail", "연결 끊김 — 실행 상태 미확인"
        elif not link.run_verified:
            kind, text = (("fail", "1.2 후보 미지원 서버(1.1 응답)") if link.peer_mismatch
                          else ("warn", "1.2 응답 확인 전"))
        elif snapshot is None:
            kind, text = "warn", "C 실행 상태 미확인"
        else:
            state = snapshot["run_state"]
            text = self.RUN_LABELS[state]
            if snapshot["hold_state"] == "HELD":
                text += " · 부품 보유 중 — 해제 명령 미제공"
            elif snapshot["hold_state"] == "HOLD_UNKNOWN":
                text += " · 보유 불명"
            kind = "warn" if state in {"STOP_REQUESTED", "RECOVERY_REQUIRED", "RECOVERING", "UNAVAILABLE"} else "accent"
        self.chip_sort.set_state(kind, "1.2 후보 · " + text)
        lines = [f"실행 상태(C 스냅샷): {text}", "지금 누르면: " + " · ".join(plans)]
        if snapshot is not None:
            lines.append(f"rev {snapshot['run_revision']} · 작업 {(snapshot['sort_job_id'] or NO_VALUE)[:8]} · "
                         f"epoch {snapshot['run_epoch']} · 정지 사유 {snapshot['stop_cause'] or NO_VALUE}")
        request = link.last_request
        if request is not None:   # 요청 결과는 실행 상태와 따로 적는다
            lines.append(f"마지막 요청(결과일 뿐 상태 아님): {request.result}" + (f" · {request.detail}" if request.detail else ""))
        e_stop = link.e_stop
        if e_stop is not None:   # 분류 상태와 따로: C 가 보고한 E-Stop 요청 결과만 적는다
            lines.append(f"E-Stop 응용 요청(안전 장치 아님): {e_stop.result}" + (f" · {e_stop.detail}" if e_stop.detail else ""))
        self.sort_detail.setText("\n".join(lines))

    def _sort_block(self, action) -> str:
        """1.2 후보에서 지금 누르면 보내지 않는 사유. 빈 문자열 = 보낸다. link 검사를 그대로 쓴다."""
        allowed, reason = self.link.sort_allowed(action)
        if not allowed:
            return reason
        if action == "RECOVER" and not self.sort_confirm.isChecked():
            return "복구 조건 확인란을 먼저 선택하세요"
        return ""

    def _estop_face(self, blocked: bool) -> None:
        """연결됐지만 보낼 수 없으면 버튼 글자에 미전송을 적고 정지(빨강) 강조를 뺀다. 보낼 수 있을 때만 「E-Stop」."""
        text, role = ("E-Stop · 미전송", "") if blocked else ("E-Stop", "stop")
        if self.estop_button.text() != text:
            self.estop_button.setText(text)
            self.estop_button.setProperty("role", role)
            self.estop_button.style().unpolish(self.estop_button)
            self.estop_button.style().polish(self.estop_button)

    def _note(self, text: str) -> None:
        self.request_note.setText(text)
        self.request_note.setToolTip(text)

    def _request_e_stop(self) -> None:
        if not self.live_run:
            self._note("E-Stop 미전송 · C TCP 1.2 필요(--tcp-version 1.2 와 C 지원)")
            return
        allowed, reason = self.link.e_stop_allowed()
        if not allowed:
            self._note(f"E-Stop 미전송 · {reason}")
        elif self.link.e_stop_request():
            self._note("E-Stop 요청 시도 · 소켓 기록·결과는 TX/RX 기록과 C 보고로 · 정지 확인 아님")
        else:
            self._note("E-Stop 미전송 · 송신 조건 불충족")
        self._update_live_sort()

    def _on_status(self, message: dict) -> None:
        self._status = message
        self._update_status()

    def _on_frame(self, image: QImage) -> None:
        self._frame_size = f"{image.width()}×{image.height()}"
        if not self.demo:
            self.stage.main.show_frame(image)

    def _on_display_frame(self, channel, image, envelope):
        if self.relay and channel == "link":
            if envelope["variant"] != self.display.mode:
                return   # 고르지 않은 종류의 링크 프레임은 순서·표시에 넣지 않는다(전환 직전 이동 중 프레임 포함)
        self.display.ingest_frame(channel, image, **envelope)

    def _on_display_result(self, result, generation, order):
        if self.relay and self.display.mode != "center":
            return   # 고르지 않은 종류의 결과는 받지 않는다(전환 직전 이동 중 center 결과 포함)
        self.display.ingest_result(result, generation=generation, order=order)

    def _update_display(self):
        if self.display is None:
            return
        tag = "C 중계 시제품" if self.relay else "D 모의"
        for channel, view in (("fixed", self.stage.main), ("gripper", self.stage.gripper), ("link", self.stage.link)):
            frame = self.display.frame(channel)
            variant = self.display.mode if channel == "link" else "raw"
            if self.display.fresh(frame):
                if getattr(view, "_display_item", None) is not frame:
                    view.show_frame(frame["image"])
                    view._display_item = frame
                view.set_footer(f"{tag} · {frame['source_id']} {variant} · 프레임 {frame['image_seq']} · "
                                f"보정 {frame['calibration_id'] or NO_VALUE}")
            elif self.relay and frame is not None and not frame.get("confirmed", True):
                if getattr(view, "_display_item", None) is not frame:   # 확인 대기: 보이되 실시간이 아닌 노후 모습으로
                    view.show_frame(frame["image"])
                    view._display_item = frame
                view.set_stale(int(max(0.0, self.display.clock() - frame["received"]) * 1000))
                view.set_footer(f"{tag} · {frame['source_id']} {variant} · 프레임 {frame['image_seq']} · "
                                "확인 대기(접속 직후 C 캐시일 수 있음 · 새 프레임 수신 전)")
            elif self.relay and frame is not None and getattr(view, "_display_item", None) is frame:
                view.set_stale(int(max(0.0, self.display.clock() - frame["received"]) * 1000))   # 마지막 모습은 노후로 표시
            else:
                view.set_placeholder(*self._display_wait(channel, variant))
                view._display_item = None
        result = self.display.matched_result()
        values = self.stage.popup.values
        def show(value):
            return NO_VALUE if value is None else str(value)
        contents = {
            "px": show(result.get("center_px")) if result else NO_VALUE,
            "mm": show(result.get("center_w_mm")) if result else NO_VALUE,
            "theta": show(result.get("theta_deg")) if result and result.get("theta_valid") else NO_VALUE,
            "calibration": show(result.get("calibration_id")) if result else NO_VALUE,
            "frame": f"{result['source_id']} · {result['image_seq']}" if result else NO_VALUE,
            "detail": (f"{'B · C 중계' if self.relay else 'D 모의'} · {show(result.get('spec'))}" if result else
                       "링크 center 를 고르면 B 결과 표시(C 시제품)" if self.relay and self.display.mode != "center"
                       else self.display.reason),
        }
        for key, text in contents.items():
            self._set_elided(values[key], text)
        for output, key in zip(self.measure_values, ("px", "mm", "theta")):
            self._set_elided(output, contents[key])
        inputs = (f"{result['source_id']} · {self.display.mode}" if result else NO_VALUE,
                  contents["calibration"], contents["detail"])
        for output, text in zip(self.input_values, inputs):
            self._set_elided(output, text)

    def _display_wait(self, channel: str, variant: str) -> tuple[str, str]:
        """3채널 칸의 빈 화면 사유. 다른 종류 영상으로 대신 채우지 않는다."""
        if not self.relay:
            return "empty", "D 모의 영상 미수신 또는 노후"
        if self.view.state != "connected":
            return "empty", "C 영상 중계 연결 안 됨"
        if channel == "link" and variant != "raw":
            sent = ("요청 송신됨 · C 적용 미확인" if self.view.requested_variant == variant
                    else "요청 송신 대기")
            return "waiting", f"링크 {variant} {sent} · 수신 대기(raw 로 대신 보이지 않음)"
        return "waiting", f"{CHANNEL_NAMES[channel]} 새 프레임 수신 대기 또는 노후"

    def _relay_status(self) -> None:
        """표시 연결 칩: TCP 연결만으로 정상 표시하지 않는다. 세 칸 각각의 신선한 프레임으로 판단한다."""
        self._update_display()
        errors = self.view.error_summary()   # 손상된 스트림이 카메라 없음처럼 보이지 않게 한다
        note = " · 수신 오류 있음" if errors else ""
        missing = [CHANNEL_NAMES[channel] for channel in ("fixed", "gripper", "link")
                   if not self.display.fresh(self.display.frame(channel))]
        if self.view.state != "connected":
            self.chip_view.set_state("idle" if self.view.state == "disconnected" else "warn",
                                     ("C 영상 중계 없음" if self.view.state == "disconnected" else "접속 중") + note)
        elif not missing:
            self.chip_view.set_state("ok", f"3채널 수신 · 링크 {self.display.mode}{note}")
        else:
            self.chip_view.set_state("warn" if len(missing) < 3 else "fail",
                                     f"연결됨 · 대기/노후: {', '.join(missing)}{note}")
        lines = [f"표시 연결: {self.chip_view._text}", f"C 영상 중계 {self.view.host}:{self.view.port}"]
        if errors:
            lines.append(errors)
        if self.view.socket_error:
            lines.append(f"마지막 연결 오류: {self.view.socket_error}")
        self.chip_view.setToolTip("\n".join(lines))   # 모든 상태에서 갱신한다(끊김·접속 중 포함)

    @staticmethod
    def _set_elided(value: QLabel, text: str) -> None:
        """「이름 … 값」 줄에서 이름을 뺀 실제 남은 폭으로 말줄임하고 전체 값은 툴팁에 둔다."""
        row = getattr(value, "_row", None)
        if row is None:
            row = value._row = next(r for r in value.parentWidget().findChildren(QHBoxLayout) if r.indexOf(value) >= 0)
        name = row.itemAt(0).widget()
        width = row.geometry().width() - name.sizeHint().width() - 2 * max(row.spacing(), 6)
        # 아직 배치 전이면 이전 고정 폭을 쓴다. 250 ms 갱신 때 실제 폭으로 다시 맞춘다.
        value.setText(value.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, width if width > 0 else 110))
        value.setToolTip(text)

    @staticmethod
    def _elidable(value: QLabel, full: str) -> QLabel:
        """한 줄 라벨이 레이아웃 최소 폭을 늘리지 않게 한다. 표시는 _fit_line 이 폭에 맞춰 줄인다."""
        value._full = full
        value.setToolTip(full)
        value.setMinimumWidth(40)
        return value

    @staticmethod
    def _fit_line(value: QLabel) -> None:
        """한 줄 라벨을 지금 폭에 맞춰 말줄임한다. 전체 문장은 value._full(툴팁에도 있음)."""
        full = getattr(value, "_full", value.text())
        width = value.width() if value.isVisible() else 0
        value.setText(value.fontMetrics().elidedText(full, Qt.TextElideMode.ElideMiddle, width) if width > 0 else full)

    def _update_status(self) -> None:
        """250 ms 마다와 값 수신 때 부른다. 오래된 값은 현재값처럼 남기지 않는다."""
        link, view = self.link, self.view
        for caption in (self.bolt_caption, self.nut_caption, *self._fit_labels):   # 창 크기가 바뀌어도 폭에 다시 맞춘다
            self._fit_line(caption)
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
        age_text = NO_VALUE if age is None else ("정상 · 단절 감시" if not link.is_stale() else f"{age // 1000}초 전")
        self.chip_age.set_state("idle" if age is None else "accent", age_text)
        self._update_command_controls()
        self._update_live_sort()   # 오래된 스냅샷을 현재 상태처럼 두지 않는다
        if not self.demo and not self.live_run:   # 1.1 실제 모드: 연결되면 누를 수 있으나 보내지 않는다(누르면 미전송 사유)
            for button in (self.start_button, self.stop_button, self.estop_button, self.recover_button):
                button.setEnabled(link.state == "connected")
            self._estop_face(blocked=link.state == "connected")
            self.estop_button.setToolTip("E-Stop · 누르면 미전송 · C TCP 1.2 필요. 로봇+컨베이어 비상 정지 요청은 "
                                         "--tcp-version 1.2 와 C 지원이 있을 때만 C 에 보냅니다. 이 화면은 안전 장치가 아닙니다.")

        if self.demo:
            self._update_display()
            self.chip_view.set_state("ok" if self.view.state == "connected" and not self.view.is_stale() else "fail", "D 모의 화면")
            return
        if self.relay:
            self._relay_status()
            return

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
        self._export_timer.stop()
        self._stop_cycle("창 닫기")
        if self._csv_file is not None:
            self._csv_file.cancelWriting()
            self._csv_file.deleteLater()
            self._csv_file = None
        self.link.stop()
        if self.sort_demo is not None:
            self.sort_demo.close()
        self.view.stop()
        if self.display is not None:
            self.display.set_connected(False)
        super().closeEvent(event)


def settings() -> QSettings:
    """사용자 화면 설정 저장소(Windows 레지스트리 HKCU/Software/EyeToArms/d_client). 테마와 적용한 서버 주소(endpoint/*)를 저장한다."""
    # 폴더명 변경 뒤에도 기존 테마 선택을 읽도록 저장 키는 유지한다.
    return QSettings("EyeToArms", "d_client")


def saved_endpoints(store) -> dict:
    """「적용·재접속」으로 저장한 주소. 검사를 통과한 값만 돌려준다(손상·옛 값은 무시하고 client.json 으로)."""
    from eyetoarm.gui.client import endpoint_error
    found = {}
    for host_key, port_key in (("host", "port"), ("view_host", "view_port")):
        host, port = store.value(f"endpoint/{host_key}"), store.value(f"endpoint/{port_key}")
        # 정수 또는 숫자만 있는 문자열(INI 저장형)만 받는다. bool·실수·"47090.5" 는 손상으로 보고 무시한다
        if isinstance(port, str) and port.isascii() and port.isdigit():
            port = int(port)
        elif port is not None and type(port) is not int:
            continue
        host = "" if host is None else str(host)
        if port is None or (host and endpoint_error(host, port)) or (not host and host_key == "host"):
            continue
        if not host and endpoint_error("127.0.0.1", port):   # 영상 IP 빈칸(명령 IP 따름)도 포트는 검사한다
            continue
        found[host_key], found[port_key] = host, port
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eyetoarm.gui.window", description="D 운영 GUI (TCP 1.1 · 모의 분류)")
    parser.add_argument("--host", default=None, help="C 서버 숫자 IP (기본: config/gui/client.json)")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--view", choices=("relay", "window", "camera"), default="relay",
                        help="메인 디스플레이 원본. relay(기본): C display_relay 의 3채널 원본 영상(시제품 ETV1, "
                             "center 프레임에 실린 B 결과만 표시). window: 화면에 떠 있는 Gazebo 창을 그대로. "
                             "camera: Gazebo 카메라 영상 중계(2 Hz, WSL 에서 브리지와 gz_view_relay.py 필요)")
    parser.add_argument("--view-host", default=None,
                        help="중계 숫자 IP. 생략하면 relay 는 명령 서버 주소(--host 또는 client.json), camera 는 127.0.0.1")
    parser.add_argument("--view-port", type=int, default=None,
                        help=f"중계 포트 (기본: camera {VIEW_PORT}, relay {RELAY_PORT})")
    parser.add_argument("--demo", action="store_true", help="모의 분류 조작·상태·화면 확인. 소켓을 열지 않음")
    parser.add_argument("--tcp-version", choices=("1.1", "1.2"), default="1.1",
                        help="명령 TCP 버전. 1.2 는 부록 A.6 후보(미승인): 실제 START/STOP/RECOVER 송수신, 수동 MOVE 비활성. "
                             "현재 C 서버(1.1)는 1.2 를 거부한다")
    parser.add_argument("--no-motion", action="store_true", help="모션 줄이기 상태로 시작")
    parser.add_argument("--no-cover", action="store_true", help="표지 화면(UR5e 인사) 없이 바로 대시보드를 연다")
    parser.add_argument("--theme", choices=tuple(theme.PALETTES), default=None,
                        help="시작 테마. 생략하면 지난번에 고른 테마(없으면 dark). 이 인자로 준 값은 저장하지 않음")
    parser.add_argument("--shot", default="", help="(개발용) 창을 띄우고 --shot-after ms 뒤 이 경로에 PNG 저장 후 종료")
    parser.add_argument("--shot-after", type=int, default=2500)
    parser.add_argument("--page", type=int, default=1, choices=(0, 1, 2), help="(개발용) 시작 화면")
    parser.add_argument("--popup", action="store_true", help="(개발용) 상세 팝업을 연 채 시작")
    parser.add_argument("--log-dir", default=None, help="세션 JSONL 폴더 (기본 logs/gui)")
    parser.add_argument("--stable-count", type=int, default=3, help="동일 상태 반복 표시 억제 횟수 (기본 3)")
    parser.add_argument("--stable-ms", type=int, default=5000, help="동일 상태 반복 표시 억제 시간 ms (기본 5000)")
    args = parser.parse_args(argv)
    if args.stable_count < 1 or args.stable_ms < 0:
        parser.error("stable-count는 1 이상, stable-ms는 0 이상이어야 합니다")
    if args.demo and args.tcp_version != "1.1":
        parser.error("--demo 는 소켓을 쓰지 않으므로 --tcp-version 1.2 와 함께 쓸 수 없습니다")
    if not args.demo and args.view != "window":   # 창·소켓을 만들기 전에 중계 주소를 검사한다
        import ipaddress
        try:
            if args.view_host is not None:
                ipaddress.ip_address(args.view_host)
        except ValueError:
            parser.error(f"--view-host 는 숫자 IP 여야 합니다: {args.view_host!r}")
        if args.view_port is not None and not 1 <= args.view_port <= 65535:
            parser.error(f"--view-port 범위 밖: {args.view_port}")

    if sys.platform == "win32":
        # 작업 표시줄이 python.exe 아이콘 대신 이 앱의 아이콘을 쓰게 한다(Windows 는 프로세스 ID 로 묶는다)
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("EyeToArms.DashboardClient")
    app = QApplication(sys.argv[:1])
    from eyetoarm.gui.icon import app_icon
    app.setWindowIcon(app_icon())   # 좌표축과 점(icon.py). 창 제목줄·작업 표시줄에 쓴다
    store = settings()
    theme.apply(args.theme or str(store.value("theme", theme.DEFAULT_THEME)))
    app.setStyleSheet(theme.QSS)
    motion.set_reduced(args.no_motion)
    if args.demo:
        from eyetoarm.gui.demo import DemoLink, DemoView
        link, view = DemoLink(), DemoView()
    else:
        # 명시한 CLI 인자 > 「적용·재접속」으로 저장한 주소 > client.json. 칸마다 따로 고른다
        saved = saved_endpoints(store)
        args.host = args.host if args.host is not None else saved.get("host")
        args.port = args.port if args.port is not None else saved.get("port")
        if args.view == "relay":
            args.view_host = args.view_host if args.view_host is not None else (saved.get("view_host") or None)
            args.view_port = args.view_port if args.view_port is not None else saved.get("view_port")
        from eyetoarm.gui.client import StatusLink
        log_options = {}
        if args.log_dir:
            from pathlib import Path
            from eyetoarm.gui.records import SessionLog
            log_options["log_path"] = Path(args.log_dir) / SessionLog.default_path().name
        link = StatusLink(args.host, args.port, stable_count=args.stable_count, stable_ms=args.stable_ms,
                          version=args.tcp_version, **log_options)
        if args.view == "window":
            from eyetoarm.gui.gz_window import GzWindowCapture   # Win32 호출을 쓰므로 Windows 에서만 불러온다
            view = GzWindowCapture()
        elif args.view == "camera":
            from eyetoarm.gui.gz_view import GzViewReceiver
            view = GzViewReceiver(args.view_host or "127.0.0.1", args.view_port or VIEW_PORT)
        else:
            from eyetoarm.gui.relay_view import RelayViewReceiver
            # C 영상 중계는 보통 명령 서버와 같은 C PC 에서 돈다. 따로 주면 그 주소를 쓴다
            view = RelayViewReceiver(args.view_host or link.host, args.view_port or RELAY_PORT)
    window = MainWindow(link, view, demo=args.demo, store=store)
    window.set_theme_fx(str(store.value("theme_fx", "ink")), save=False)
    window.show()
    if args.page != 1:
        window._show_page(args.page)
    if args.popup:
        window.detail_button.setChecked(True)
    if not args.shot and not args.no_cover:
        from eyetoarm.gui.cover import CoverScreen
        # 표지는 덮개일 뿐이다. 아래 창의 연결·수신은 이미 시작했고 표지가 막지 않는다
        CoverScreen(window.centralWidget()).finished.connect(window.play_entrance)
    if args.shot:
        def shoot() -> None:
            window.grab().save(args.shot)
            window.close()
        QTimer.singleShot(args.shot_after, shoot)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
