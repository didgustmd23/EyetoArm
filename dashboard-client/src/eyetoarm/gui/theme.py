"""D 운영 GUI 시각 토큰과 전역 스타일(QSS). 다크·라이트 두 테마를 실행 중에 바꿀 수 있다.

[담당] D 전용. 다른 파트(A·B·C) 코드·설정을 읽거나 바꾸지 않는다.
[방향] 사용자 지정 2026-11-13: Higgsfield 계열 인터페이스의 "느낌"을 따른다.
       - 바탕 위에 형광 라임 한 가지 강조색
       - 큰 굵은 제목, 넉넉한 여백, 크게 둥근 카드, 알약(pill) 버튼
       로고·이미지·아이콘·글꼴 파일 등 자산은 가져오지 않는다. 색·모양 규칙만 새로 작성했다.
[방향 갱신] 사용자 지정 2026-11-20: 참고 릴스(습관 기록 대시보드 시안)의 디자인·모션을 따른다.
       - 상단 한 줄에 상표와 캡슐형 탭. 고른 탭은 흰 알약이 미끄러져 가서 덮는다
       - 더 깊은 검정 바탕 + 모서리의 옅은 색 번짐, 위가 살짝 밝은 큰 둥근 카드
       - 한 가지 라임 대신 여러 선명한 색(HUES)을 자리마다 나눠 쓴다. 상태 색(OK·WARN·FAIL)의 뜻은 그대로다
       - 작은 구분 글자는 고정폭·넓은 자간
       참고물의 로고·아이콘·글꼴·문구·화면 구성은 가져오지 않았다. 색·모양·움직임 규칙만 새로 작성했다.
       기능·표시 문구·영상 3채널 표시 방식은 바꾸지 않는다.
[컴팩트] 사용자 지정 2026-11-20(2차): 전체를 깔끔하고 촘촘하게. 글자·버튼·카드 여백·모서리를 한 단계씩 줄였다.
       표지 화면(cover.py)과 등장·강조 모션을 더했다.
[방향 갱신 3차] 사용자 지정 2026-11-20: 두 번째 참고 릴스(물류 관제 화면 시안)의 전체 느낌을 따른다.
       - 밝은 테마가 기본. 차가운 옅은 회색 바탕 위에 흰 카드, 가는 옅은 테두리, 번짐 없는 평평한 면
       - 검은 알약이 「고른 것·주요 버튼」, 주황 한 가지가 「눈여겨볼 것」. 여러 색(HUES)과 라임은 걷어 냈다
       - 꼬리표·표 머리글은 고정폭을 버리고 작은 보통 글자로
       이번에도 참고물의 로고·아이콘·글꼴·문구·지도·화면 구성은 가져오지 않았다. 기능과 영상 3채널 표시 방식은 그대로다.
[이전 초안과의 차이] docs/design/D_2일차_대시보드_골격.md 의 Midnight Navy(파란 강조) 토큰을
       위 사용자 지정으로 대체한다.
[접근성] 상태는 색만으로 구분하지 않는다. 모든 상태 표시에 문구를 함께 쓴다(D 상세설계 §A.2).

[테마 구조] 사용자 지정 2026-11-13: 다크·라이트 선택.
  - PALETTES 에 테마별 색을 둔다. 색을 고치거나 테마를 추가할 때는 이 표만 바꾼다.
  - apply(name) 이 이 모듈의 색 상수(BG · TEXT · ...)와 QSS 를 그 테마 값으로 바꾼다.
  - 다른 파일은 색을 복사해 두지 말고 그릴 때마다 theme.BG 처럼 읽는다. 그래야 전환이 즉시 반영된다.
  - 영상 표시면(DISPLAY_*)은 테마와 무관하게 항상 어둡다. 영상의 밝기·색을 테마가 바꾸지 않게 하기 위해서다.
"""

from __future__ import annotations

PALETTES: dict[str, dict[str, str]] = {
    "light": {                    # 기본 테마. 참고 릴스(2026-11-20, 3차)의 밝은 운영 도구 느낌
        "BG": "#ECEEF1",          # 창 바탕(차가운 옅은 회색)
        "SURFACE": "#FFFFFF",     # 카드
        "SURFACE_HI": "#FFFFFF",  # 카드 위쪽(밝은 테마에서는 카드와 같다)
        "SURFACE_2": "#F3F4F6",   # 카드 안 구획, 입력, 꼬리표
        "LINE": "#E2E4E9",        # 구획선
        "TEXT": "#111318",        # 본문
        "DIM": "#6B7280",         # 보조 글자
        "ACCENT": "#F26B2A",      # 강조 면(주황 한 가지: 상표 표식·차오르는 단계 칸·체크·피켓)
        "ON_ACCENT": "#14110F",   # 강조 면 위 글자(주황 위 흰 글자는 대비가 모자라 어두운 색을 쓴다)
        "ACCENT_INK": "#D9480F",  # 강조 글자·점(흰 바탕 위에 직접 올라가는 주황)
        "SELECT": "#111318",      # 고른 탭·켜진 선택 버튼·주요 버튼의 면(검은 알약)
        "ON_SELECT": "#FFFFFF",   # 그 위 글자
        "GLOW_A": "#ECEEF1",      # 창 모서리 번짐. 지금은 바탕과 같아 평평하다
        "GLOW_B": "#ECEEF1",
        "OK": "#12855A", "WARN": "#A86A00", "FAIL": "#D03030",
        "IDLE": "#A3A8B3",        # 미연결·미확인·자리만 있음
    },
    "dark": {                     # 같은 규칙의 어두운 판
        "BG": "#0E0F12",
        "SURFACE": "#17181C",
        "SURFACE_HI": "#1B1C21",
        "SURFACE_2": "#202228",
        "LINE": "#2A2C33",
        "TEXT": "#F3F4F6",
        "DIM": "#9097A3",
        "ACCENT": "#F26B2A",
        "ON_ACCENT": "#14110F",
        "ACCENT_INK": "#FF8A4C",
        "SELECT": "#F3F4F6",      # 어두운 바탕에서는 흰 알약
        "ON_SELECT": "#0E0F12",
        "GLOW_A": "#0E0F12",
        "GLOW_B": "#0E0F12",
        "OK": "#3DDC97", "WARN": "#F7C948", "FAIL": "#FF6B6B",
        "IDLE": "#5A5F6B",
    },
}
DEFAULT_THEME = "light"   # 저장된 선택도 --theme 도 없을 때. 모르는 이름을 받았을 때의 대체값은 apply() 의 dark 그대로다
THEME_LABELS = {"dark": "다크", "light": "라이트"}

# 영상 표시면 전용. 테마가 바뀌어도 그대로다
DISPLAY_BG = "#060607"
DISPLAY_TEXT = "#F4F4F5"
DISPLAY_DIM = "#8C8C95"
DISPLAY_LINE = "#26262B"
DISPLAY_ACCENT = "#FF8A4C"
DISPLAY_WARN, DISPLAY_FAIL, DISPLAY_IDLE = "#F2C66D", "#FF6B6B", "#5A5A63"

# ── 모양·글자(테마 공통) ──────────────────────────────────────────────
RADIUS = 14             # 카드 모서리
RADIUS_SMALL = 10
FONT_FAMILIES = ["Segoe UI Variable Display", "Segoe UI", "Malgun Gothic"]  # Windows 기본 글꼴만 사용
MONO_FAMILIES = ["Cascadia Mono", "Consolas"]

# ── 모션 시간(ms). 골격 문서의 모션 규칙 표를 따른다 ─────────────────────
MS_NAV = 260            # 탭의 흰 알약 이동(끝에서 살짝 넘쳤다 돌아온다)
MS_POPUP = 180          # 상세 팝업 펼침·접힘
MS_STAGE = 420          # 단계 칸에 색이 차오르는 시간 · 상태 점 색 전환
MS_FLASH = 520          # 상태 칸에 색이 한 번 쓸고 지나가는 시간
MS_PAGE = 300           # 화면 전환 시 카드 등장
MS_STAGGER = 55         # 카드가 차례로 등장할 때 간격
MS_HINT = 900           # 단축키 표시가 떠 있는 시간

_FONT = ", ".join(f'"{name}"' for name in FONT_FAMILIES)
_MONO = ", ".join(f'"{name}"' for name in MONO_FAMILIES)

# apply() 가 채운다. 아래 이름들이 다른 파일에서 theme.BG 처럼 읽는 현재 테마 값이다
name = ""
BG = SURFACE = SURFACE_HI = SURFACE_2 = LINE = TEXT = DIM = ACCENT = ON_ACCENT = ACCENT_INK = ""
SELECT = ON_SELECT = GLOW_A = GLOW_B = OK = WARN = FAIL = IDLE = ""
LEVEL_COLOR: dict[str, str] = {}   # StatusChip 의 level → 색
QSS = ""


def _build_qss() -> str:
    return f"""
* {{ font-family: {_FONT}; color: {TEXT}; }}
QMainWindow {{ background: {BG}; }}
#Root {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {GLOW_A}, stop:0.22 {BG}, stop:0.78 {BG}, stop:1 {GLOW_B}); }}
QToolTip {{ background: {SURFACE_2}; color: {TEXT}; border: 1px solid {LINE}; padding: 6px 8px; }}

#NavRail {{ background: {SURFACE}; border: none; border-bottom: 1px solid {LINE}; }}
#BrandMark {{ background: {ACCENT}; color: #FFFFFF; border-radius: 8px; font-size: 12px; font-weight: 900; }}
#Brand {{ font-size: 15px; font-weight: 800; }}
#NavTabs {{ background: {SURFACE_2}; border: 1px solid {LINE}; border-radius: 17px; }}
#NavButton {{ background: transparent; border: 1px solid transparent; padding: 0 14px;
             font-size: 12px; font-weight: 700; color: {DIM}; border-radius: 13px; }}
#NavButton:hover {{ color: {TEXT}; }}
#NavButton:checked {{ color: {ON_SELECT}; }}
#NavButton:focus {{ border-color: {LINE}; }}
#NavIndicator {{ background: {SELECT}; border-radius: 13px; }}

#KeyHint {{ background: {SELECT}; color: {ON_SELECT}; border-radius: 12px; padding: 7px 14px;
            font-size: 12px; font-weight: 700; }}
#PageTitle {{ font-size: 22px; font-weight: 800; }}
#Caption {{ color: {DIM}; font-size: 11px; font-weight: 600; }}
#Dim {{ color: {DIM}; font-size: 12px; }}
#Value {{ font-size: 22px; font-weight: 700; }}
#Mono {{ font-family: {_MONO}; font-size: 12px; color: {DIM}; }}

#Card {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {SURFACE_HI}, stop:1 {SURFACE});
         border: 1px solid {LINE}; border-radius: {RADIUS}px; }}
#CardTitle {{ font-size: 14px; font-weight: 700; }}
#Tag {{ background: {SURFACE_2}; color: {DIM}; border: 1px solid {LINE}; border-radius: 9px; padding: 2px 8px;
        font-size: 10px; font-weight: 600; }}

QPushButton#Pill {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 16px;
                   padding: 0 14px; min-height: 32px; font-size: 12px; font-weight: 700; }}
QPushButton#Pill:hover {{ border-color: {DIM}; }}
QPushButton#Pill:hover {{ background: {SURFACE_2}; }}
QPushButton#Pill:pressed {{ background: {LINE}; }}
QPushButton#Pill:focus {{ border-color: {ACCENT_INK}; }}
QPushButton#Pill:checked {{ background: {SELECT}; color: {ON_SELECT}; border-color: {SELECT}; }}
QPushButton#Pill:disabled {{ color: {IDLE}; border-color: {LINE}; background: transparent; }}
QPushButton#Pill[role="primary"] {{ background: {SELECT}; color: {ON_SELECT}; border-color: {SELECT}; }}
QPushButton#Pill[role="primary"]:disabled {{ background: transparent; color: {IDLE}; border-color: {LINE}; }}
QPushButton#Pill[role="stop"] {{ border-color: {FAIL}; color: {FAIL}; }}
QPushButton#Pill[role="stop"]:disabled {{ border-color: {LINE}; color: {IDLE}; }}
QPushButton#Pill[role="small"] {{ min-height: 26px; border-radius: 13px; padding: 0 12px; font-size: 11px; }}

QTableWidget {{ background: transparent; border: none; gridline-color: transparent;
               font-family: {_MONO}; font-size: 12px; selection-background-color: {SURFACE_2}; selection-color: {TEXT}; }}
QTableWidget::item {{ padding: 2px 8px; border-bottom: 1px solid {LINE}; }}
QHeaderView::section {{ background: transparent; color: {DIM}; border: none; border-bottom: 1px solid {LINE};
                       padding: 4px 8px; font-size: 10px; font-weight: 600; }}
QScrollBar:vertical {{ background: transparent; width: 8px; }}
QScrollBar::handle:vertical {{ background: {LINE}; border-radius: 4px; min-height: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QCheckBox {{ color: {DIM}; font-size: 12px; spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 5px; border: 1px solid {LINE}; background: {SURFACE_2}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT_INK}; }}
QLineEdit {{ background: {SURFACE_2}; color: {TEXT}; border: 1px solid {LINE}; border-radius: 8px; padding: 4px 8px;
            font-size: 13px; selection-background-color: {ACCENT}; selection-color: {ON_ACCENT}; }}
QLineEdit:focus {{ border-color: {ACCENT_INK}; }}
QLineEdit:disabled {{ color: {IDLE}; background: transparent; }}
"""


def apply(theme_name: str) -> None:
    """테마를 바꾼다. 모듈 상수와 QSS 를 새 값으로 채운다.

    부른 쪽은 이어서 QApplication.setStyleSheet(theme.QSS) 와 직접 그리는 위젯의 다시 그리기를 해야 한다
    (window.MainWindow.set_theme 가 한다).
    """
    global name, QSS
    if theme_name not in PALETTES:
        theme_name = "dark"
    name = theme_name
    globals().update(PALETTES[theme_name])
    LEVEL_COLOR.clear()
    LEVEL_COLOR.update({"ok": OK, "warn": WARN, "fail": FAIL, "idle": IDLE, "accent": ACCENT_INK})
    QSS = _build_qss()


apply("dark")   # 기본값. 창이 저장된 선택이나 --theme 로 다시 부른다
