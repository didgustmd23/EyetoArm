"""D 운영 GUI 시각 토큰과 전역 스타일(QSS). 다크·라이트 두 테마를 실행 중에 바꿀 수 있다.

[담당] D 전용. 다른 파트(A·B·C) 코드·설정을 읽거나 바꾸지 않는다.
[방향] 사용자 지정 2026-11-13: Higgsfield 계열 인터페이스의 "느낌"을 따른다.
       - 바탕 위에 형광 라임 한 가지 강조색
       - 큰 굵은 제목, 넉넉한 여백, 크게 둥근 카드, 알약(pill) 버튼
       로고·이미지·아이콘·글꼴 파일 등 자산은 가져오지 않는다. 색·모양 규칙만 새로 작성했다.
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
    "dark": {
        "BG": "#0A0A0B",          # 창 바탕
        "SURFACE": "#131315",     # 카드
        "SURFACE_2": "#1B1B1E",   # 카드 안 구획, 입력
        "LINE": "#26262B",        # 구획선
        "TEXT": "#F4F4F5",        # 본문
        "DIM": "#8C8C95",         # 보조 글자
        "ACCENT": "#D1FE17",      # 강조 면(선택 표시·주요 버튼 바탕)
        "ON_ACCENT": "#0A0A0B",   # 강조 면 위 글자
        "ACCENT_INK": "#D1FE17",  # 강조 글자·점(바탕 위에 직접 올라가는 강조색)
        "OK": "#7CE38B", "WARN": "#F2C66D", "FAIL": "#FF6B6B",
        "IDLE": "#5A5A63",        # 미연결·미확인·자리만 있음
    },
    "light": {
        "BG": "#F3F3EF",
        "SURFACE": "#FFFFFF",
        "SURFACE_2": "#ECECE6",
        "LINE": "#D8D8D0",
        "TEXT": "#121214",
        "DIM": "#666670",
        "ACCENT": "#D1FE17",      # 면은 같은 라임. 위 글자는 어두운 색이라 밝은 바탕에서도 읽힌다
        "ON_ACCENT": "#0A0A0B",
        "ACCENT_INK": "#4F6B00",  # 흰 바탕 위 라임 글자는 안 보이므로 어두운 올리브로 바꾼다
        "OK": "#1E8E44", "WARN": "#A86A00", "FAIL": "#D03030",
        "IDLE": "#A3A3AB",
    },
}
THEME_LABELS = {"dark": "다크", "light": "라이트"}

# 영상 표시면 전용. 테마가 바뀌어도 그대로다
DISPLAY_BG = "#060607"
DISPLAY_TEXT = "#F4F4F5"
DISPLAY_DIM = "#8C8C95"
DISPLAY_LINE = "#26262B"
DISPLAY_ACCENT = "#D1FE17"
DISPLAY_WARN, DISPLAY_FAIL, DISPLAY_IDLE = "#F2C66D", "#FF6B6B", "#5A5A63"

# ── 모양·글자(테마 공통) ──────────────────────────────────────────────
RADIUS = 18             # 카드 모서리
RADIUS_SMALL = 12
FONT_FAMILIES = ["Segoe UI Variable Display", "Segoe UI", "Malgun Gothic"]  # Windows 기본 글꼴만 사용
MONO_FAMILIES = ["Cascadia Mono", "Consolas"]

# ── 모션 시간(ms). 골격 문서의 모션 규칙 표를 따른다 ─────────────────────
MS_NAV = 160            # 탭 선택선 이동
MS_POPUP = 180          # 상세 팝업 펼침·접힘
MS_STAGE = 200          # 상태 강조 전환
MS_FLASH = 240          # 새 값이 들어온 칸의 바탕 강조
MS_PAGE = 220           # 화면 전환 시 본문 등장
MS_STAGGER = 45         # 카드가 차례로 등장할 때 간격

_FONT = ", ".join(f'"{name}"' for name in FONT_FAMILIES)
_MONO = ", ".join(f'"{name}"' for name in MONO_FAMILIES)

# apply() 가 채운다. 아래 이름들이 다른 파일에서 theme.BG 처럼 읽는 현재 테마 값이다
name = ""
BG = SURFACE = SURFACE_2 = LINE = TEXT = DIM = ACCENT = ON_ACCENT = ACCENT_INK = OK = WARN = FAIL = IDLE = ""
LEVEL_COLOR: dict[str, str] = {}   # StatusChip 의 level → 색
QSS = ""


def _build_qss() -> str:
    return f"""
* {{ font-family: {_FONT}; color: {TEXT}; }}
QMainWindow, #Root {{ background: {BG}; }}
QToolTip {{ background: {SURFACE_2}; color: {TEXT}; border: 1px solid {LINE}; padding: 6px 8px; }}

#NavRail {{ background: {BG}; border-right: 1px solid {LINE}; }}
#Brand {{ font-size: 22px; font-weight: 800; }}
#NavButton {{ background: transparent; border: 1px solid transparent; text-align: left; padding: 0 16px;
             font-size: 15px; font-weight: 600; color: {DIM}; border-radius: {RADIUS_SMALL}px; }}
#NavButton:hover {{ color: {TEXT}; }}
#NavButton:checked {{ color: {ON_ACCENT}; }}
#NavButton:focus {{ border-color: {LINE}; }}
#NavIndicator {{ background: {ACCENT}; border-radius: {RADIUS_SMALL}px; }}

#PageTitle {{ font-size: 34px; font-weight: 800; }}
#Caption {{ color: {DIM}; font-size: 12px; font-weight: 600; }}
#Dim {{ color: {DIM}; font-size: 13px; }}
#Value {{ font-size: 26px; font-weight: 700; }}
#Mono {{ font-family: {_MONO}; font-size: 12px; color: {DIM}; }}

#Card {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: {RADIUS}px; }}
#CardTitle {{ font-size: 16px; font-weight: 700; }}
#Tag {{ background: {SURFACE_2}; color: {DIM}; border-radius: 9px; padding: 2px 8px; font-size: 11px; font-weight: 700; }}

QPushButton#Pill {{ background: {SURFACE_2}; border: 1px solid {LINE}; border-radius: 19px;
                   padding: 0 18px; min-height: 38px; font-size: 13px; font-weight: 700; }}
QPushButton#Pill:hover {{ border-color: {DIM}; }}
QPushButton#Pill:focus {{ border-color: {ACCENT_INK}; }}
QPushButton#Pill:checked {{ background: {ACCENT}; color: {ON_ACCENT}; border-color: {ACCENT}; }}
QPushButton#Pill:disabled {{ color: {IDLE}; border-color: {LINE}; background: transparent; }}
QPushButton#Pill[role="primary"] {{ background: {ACCENT}; color: {ON_ACCENT}; border-color: {ACCENT}; }}
QPushButton#Pill[role="primary"]:disabled {{ background: transparent; color: {IDLE}; border-color: {LINE}; }}
QPushButton#Pill[role="stop"] {{ border-color: {FAIL}; color: {FAIL}; }}
QPushButton#Pill[role="stop"]:disabled {{ border-color: {LINE}; color: {IDLE}; }}
QPushButton#Pill[role="small"] {{ min-height: 30px; border-radius: 15px; padding: 0 14px; font-size: 12px; }}

QTableWidget {{ background: transparent; border: none; gridline-color: transparent;
               font-family: {_MONO}; font-size: 12px; selection-background-color: {SURFACE_2}; selection-color: {TEXT}; }}
QTableWidget::item {{ padding: 4px 8px; border-bottom: 1px solid {LINE}; }}
QHeaderView::section {{ background: transparent; color: {DIM}; border: none; border-bottom: 1px solid {LINE};
                       padding: 6px 8px; font-size: 11px; font-weight: 700; }}
QScrollBar:vertical {{ background: transparent; width: 8px; }}
QScrollBar::handle:vertical {{ background: {LINE}; border-radius: 4px; min-height: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QCheckBox {{ color: {DIM}; font-size: 12px; spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 5px; border: 1px solid {LINE}; background: {SURFACE_2}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT_INK}; }}
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
