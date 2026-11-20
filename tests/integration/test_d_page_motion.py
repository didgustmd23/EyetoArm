"""D 화면 전환 줌·단축키 표시 모션 시험. 모의 연결(DemoLink)만 쓰고 소켓을 열지 않는다.

  QT_QPA_PLATFORM=offscreen python -m unittest tests.integration.test_d_page_motion
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dashboard-client" / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from eyetoarm.gui import motion, theme  # noqa: E402
from eyetoarm.gui.demo import DemoLink, DemoView  # noqa: E402
from eyetoarm.gui.window import MainWindow  # noqa: E402

SETTLE_MS = theme.MS_PAGE + theme.MS_STAGGER * 8 + 300   # 줌과 카드 등장이 모두 끝나는 시간


class PageMotionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        motion.set_reduced(False)
        theme.apply("light")
        self.app.setStyleSheet(theme.QSS)
        self.window = MainWindow(DemoLink(), DemoView(), demo=True)
        self.window.show()
        QTest.qWait(100)

    def tearDown(self) -> None:
        self.window.close()
        motion.set_reduced(False)
        theme.apply("dark")

    def _press(self, digit: int) -> None:
        QTest.keyClick(self.window, Qt.Key(Qt.Key.Key_0 + digit), Qt.KeyboardModifier.ControlModifier)

    def test_shortcut_switches_at_once_and_motion_ends(self) -> None:
        window = self.window
        self._press(3)
        self.assertEqual(window.pages.currentIndex(), 2)                 # 화면은 모션을 기다리지 않고 즉시 바뀐다
        self.assertEqual(window.title.text(), "평가·기록")
        self.assertTrue(window.page_zoom.isVisible())                    # 떠나는 화면의 그림이 겹쳐 있다
        self.assertTrue(window.key_hint.isVisible())
        self.assertIn("Ctrl+3", window.key_hint.text())
        QTest.qWait(SETTLE_MS)
        self.assertFalse(window.page_zoom.isVisible())                   # 줌이 끝나면 겹친 그림이 사라진다
        QTest.qWait(theme.MS_HINT)
        self.assertFalse(window.key_hint.isVisible())

    def _positions(self) -> list:
        """세 화면을 차례로 열어 각 화면이 보일 때의 카드 자리를 모은다(숨은 화면은 배치가 갱신되지 않는다)."""
        found = []
        for digit in (1, 3, 2):
            self._press(digit)
            QTest.qWait(SETTLE_MS)
            found.append([card.pos() for card in self.window._page_cards[digit - 1]])
        return found

    def test_fast_switching_leaves_cards_in_place(self) -> None:
        window = self.window
        home = self._positions()
        for digit in (1, 3, 2, 1, 2, 3, 2):                              # 등장 모션 도중에 계속 화면을 바꾼다
            self._press(digit)
            QTest.qWait(60)
        QTest.qWait(SETTLE_MS)
        self.assertEqual(window.pages.currentIndex(), 1)
        self.assertEqual(self._positions(), home)                        # 자리가 밀리지 않는다
        for cards in window._page_cards:
            for card in cards:
                self.assertIsNone(card.graphicsEffect())                 # 등장 효과가 남지 않는다

    def test_theme_wipe_both_styles_cover_then_end(self) -> None:
        window = self.window
        root = window.centralWidget()
        corners = [(0, 0), (root.width() - 1, 0), (0, root.height() - 1), (root.width() - 1, root.height() - 1)]
        for style, name in (("ink", "dark"), ("venom", "light"), ("tiles", "dark")):
            window.set_theme_fx(style, save=False)
            window.set_theme(name, save=False)
            self.assertEqual(theme.name, name)                           # 색은 효과를 기다리지 않고 즉시 바뀐다
            wipe = window.theme_wipe
            self.assertTrue(wipe.isVisible())
            for x, y in corners:                                         # 다 큰 얼룩은 창 전체를 덮는다
                self.assertTrue(wipe.blob(1.0).contains(QPointF(x, y)), (style, x, y))
            self.assertFalse(wipe.blob(0.0).contains(QPointF(*corners[3])))
            for column, row in ((0, 0), (1, 0), (9, 5)):                 # 타일은 처음엔 옛 면, 끝엔 다 넘어간다
                self.assertEqual(wipe.tile_turn(column, row, 0.0), 0.0)
                self.assertEqual(wipe.tile_turn(column, row, 1.0), 1.0)
            QTest.qWait((theme.MS_INK if style == "ink" else theme.MS_THEME) + 300)
            self.assertFalse(wipe.isVisible())                           # 끝나면 겹친 그림이 사라진다

    def test_ink_reveals_from_either_button_and_releases_snapshot(self) -> None:
        window = self.window
        wipe = window.theme_wipe
        for name in ("dark", "light"):
            window.set_theme(name, save=False)
            wipe._animation.pause()
            wipe._animation.setCurrentTime(theme.MS_INK // 3)
            overlay = QImage(wipe.size(), QImage.Format.Format_ARGB32_Premultiplied)
            overlay.fill(Qt.GlobalColor.transparent)
            with QPainter(overlay) as painter:
                wipe._paint_ink(painter)
            self.assertEqual(overlay.pixelColor(wipe._origin.toPoint()).alpha(), 0)
            self.assertEqual(overlay.pixelColor(2, wipe.height() - 3).alpha(), 255)
        window.reduce.setChecked(True)
        self.assertFalse(wipe.isVisible())
        self.assertIsNone(wipe._pixmap)
        self.assertEqual(theme.name, "light")

    def test_reduced_motion_has_no_theme_wipe(self) -> None:
        motion.set_reduced(True)
        self.window.set_theme("dark", save=False)
        self.assertEqual(theme.name, "dark")
        self.assertFalse(self.window.theme_wipe.isVisible())

    def test_reduced_motion_has_no_zoom(self) -> None:
        motion.set_reduced(True)
        self._press(1)
        self.assertEqual(self.window.pages.currentIndex(), 0)
        self.assertFalse(self.window.page_zoom.isVisible())


if __name__ == "__main__":
    unittest.main()
