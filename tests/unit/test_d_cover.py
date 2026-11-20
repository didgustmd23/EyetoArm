"""D 표지 화면(UR5e 인사 그림)의 자세 계산과 닫힘 동작 시험. 소켓을 열지 않는다.

  QT_QPA_PLATFORM=offscreen python -m unittest tests.unit.test_d_cover
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dashboard-client" / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from eyetoarm.gui import cover, motion  # noqa: E402


class WavePoseTest(unittest.TestCase):
    def test_rises_then_waves(self) -> None:
        self.assertEqual(cover.wave_pose(0), (*cover._FOLDED, 0.0))           # 처음은 접힌 자세, 흔들림 없음
        shoulder, _elbow, _wrist, strength = cover.wave_pose(cover.RISE_MS + 400)
        self.assertAlmostEqual(shoulder, cover._RAISED[0])                     # 어깨는 든 자세에서 멈춘다
        self.assertEqual(strength, 1.0)
        elbows = [cover.wave_pose(cover.RISE_MS + 400 + ms)[1] for ms in range(0, 1200, 20)]
        self.assertLess(min(elbows), cover._RAISED[1] - 15)                    # 팔꿈치가 좌우로 흔들린다
        self.assertGreater(max(elbows), cover._RAISED[1] + 15)
        self.assertLessEqual(max(elbows) - cover._RAISED[1], 20.0 + 1e-9)      # 정한 폭을 넘지 않는다


class CoverScreenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def tearDown(self) -> None:
        motion.set_reduced(False)

    def _cover(self):
        host = QWidget()
        host.resize(900, 600)
        host.show()
        screen = cover.CoverScreen(host)
        done: list[int] = []
        screen.finished.connect(lambda: done.append(1))
        return host, screen, done

    def test_covers_parent_and_closes_once_on_key(self) -> None:
        host, screen, done = self._cover()
        self.assertEqual(screen.geometry(), host.rect())
        host.resize(1000, 700)
        self.app.processEvents()
        self.assertEqual(screen.geometry(), host.rect())                       # 창 크기를 따라간다
        screen.grab()                                                          # 그리기에서 예외가 없어야 한다
        QTest.keyClick(screen, Qt.Key.Key_Return)
        QTest.keyClick(screen, Qt.Key.Key_Return)                              # 두 번 눌러도 한 번만 닫힌다
        QTest.qWait(cover.FADE_MS + 200)
        self.assertEqual(done, [1])

    def test_reduced_motion_closes_without_fade(self) -> None:
        motion.set_reduced(True)
        _host, screen, done = self._cover()
        screen.grab()
        QTest.mouseClick(screen, Qt.MouseButton.LeftButton)
        self.assertEqual(done, [1])                                            # 흐려짐 없이 즉시


if __name__ == "__main__":
    unittest.main()
