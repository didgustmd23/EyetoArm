"""색·글꼴·QSS. 화면 기준은 unslop(Damir Sokolovsky, CC BY 4.0)을 따랐다:
회색 단일 램프, 강조색 하나, 4px 간격, 문장형 표기, 굵기 400·500 두 가지."""

from __future__ import annotations

from PySide6.QtGui import QFont, QFontDatabase

BG, CARD, RAISED = "#fafafa", "#ffffff", "#f4f4f5"
LINE, LINE2 = "#ebebeb", "#dcdcdc"
T1, T2, T3 = "#171717", "#525252", "#737373"
ACC, ACC_SOFT = "#0f766e", "#e7f1f0"
NEG, POS, WARN = "#dc2626", "#16a34a", "#b45309"

# 움직임: 들어올 때 ease-out, 나갈 때 ease-in, 0.2~0.4초
MS_FAST, MS_ENTER, MS_SHEET = 200, 300, 350


def mono(size: int = 10) -> QFont:
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    font.setPointSize(size)
    return font


STYLE = f"""
* {{ font-family: 'Noto Sans CJK KR', 'Noto Sans KR', 'Malgun Gothic', sans-serif; font-size: 13px; color: {T1}; }}
QMainWindow, QWidget#root {{ background: {BG}; }}
QWidget#side {{ background: {RAISED}; border-right: 1px solid {LINE}; }}
QWidget#top {{ background: {CARD}; border-bottom: 1px solid {LINE}; }}
QFrame#card {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 12px; }}
QLabel#h1 {{ font-size: 17px; font-weight: 500; }}
QLabel#h2 {{ font-weight: 500; }}
QLabel#muted, QLabel#meta {{ color: {T3}; font-size: 12px; }}
QLabel#sub {{ color: {T2}; }}
QLabel#fig {{ font-size: 26px; font-weight: 500; }}
QLabel#brand {{ font-size: 17px; font-weight: 500; }}
QLabel#tag {{ background: {RAISED}; color: {T2}; border-radius: 6px; padding: 3px 8px; font-size: 12px; }}
QLabel#seg {{ color: {T3}; font-size: 12px; padding: 4px 10px; background: {RAISED}; border-radius: 6px; }}
QLabel#seg[on="true"] {{ color: {T1}; font-weight: 500; background: {CARD}; border: 1px solid {LINE2}; }}
QPushButton {{ background: {CARD}; border: 1px solid {LINE2}; border-radius: 8px; padding: 6px 12px; min-height: 18px; }}
QPushButton:hover {{ background: {RAISED}; }}
QPushButton:focus {{ border: 2px solid {ACC}; }}
QPushButton:disabled {{ color: {T3}; background: {RAISED}; border-color: {LINE}; }}
QPushButton#primary {{ background: {T1}; border-color: {T1}; color: #ffffff; min-height: 26px; font-weight: 500; }}
QPushButton#primary:hover {{ background: #2a2a2a; }}
QPushButton#primary:disabled {{ background: #a3a3a3; border-color: #a3a3a3; color: #ffffff; }}
QPushButton#stop {{ background: {NEG}; border-color: {NEG}; color: #ffffff; font-weight: 500; padding: 6px 16px; }}
QPushButton#stop:hover {{ background: #b91c1c; }}
QPushButton#nav {{ border: 0; background: transparent; text-align: left; padding: 8px 10px; color: {T2}; border-radius: 8px; }}
QPushButton#nav:hover {{ background: #ececed; }}
QPushButton#nav:checked {{ background: {CARD}; color: {T1}; font-weight: 500; border: 1px solid {LINE}; }}
QPushButton#link {{ border: 0; background: transparent; text-decoration: underline; padding: 0; color: {T1}; }}
QPushButton#pin {{ border: 0; background: transparent; padding: 0; color: {T3}; }}
QPushButton#pin[on="true"] {{ color: {ACC}; }}
QLineEdit {{ background: {CARD}; border: 1px solid {LINE2}; border-radius: 6px; padding: 4px 8px; }}
QLineEdit:focus {{ border: 2px solid {ACC}; }}
QTableWidget {{ background: {CARD}; border: 0; gridline-color: {LINE}; selection-background-color: {RAISED}; selection-color: {T1}; }}
QTableWidget::item {{ padding: 0 6px; border-bottom: 1px solid {LINE}; }}
QHeaderView::section {{ background: {CARD}; color: {T3}; border: 0; border-bottom: 1px solid {LINE}; padding: 4px 6px; font-size: 12px; }}
QPlainTextEdit {{ background: {RAISED}; border: 0; border-radius: 8px; padding: 10px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: {LINE2}; border-radius: 5px; min-height: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QFrame#toast {{ background: {CARD}; border: 1px solid {LINE2}; border-radius: 10px; }}
QFrame#snack {{ background: {T1}; border-radius: 10px; }}
QFrame#snack QLabel {{ color: #fafafa; }}
QFrame#snack QPushButton {{ background: #3a3a3a; border: 0; color: #fafafa; padding: 4px 10px; border-radius: 6px; }}
QFrame#popup {{ background: {CARD}; border-radius: 16px; }}
QFrame#sheet {{ background: {CARD}; border-top-left-radius: 16px; border-top-right-radius: 16px; }}
QWidget#dim {{ background: rgba(23, 23, 23, 110); }}
"""
