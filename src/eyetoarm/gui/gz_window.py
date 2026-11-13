"""Gazebo 창 화면 수신(Windows 쪽): 화면에 떠 있는 Gazebo GUI 창을 그대로 떠서 QImage 로 넘긴다.

┌─ 파트별 경계 ──────────────────────────────────────────────────────────────┐
│ [D 임시 경로 · 기본값] 사용자 지정 2026-11-13: 카메라 이미지가 아니라          │
│   Gazebo 의 실제 화면을 메인 디스플레이에 내보낸다.                          │
│   WSL 의 Gazebo GUI 는 WSLg 를 통해 Windows 창("Gazebo Sim (...)")으로 뜬다.  │
│   그 창의 그림을 Windows 가 그려 준 그대로 읽는다. 초당 15장.                │
│ [C·A 무변경] WSL 쪽에 추가로 띄울 것이 없다. 브리지·중계·월드·launch·C 코드를 │
│   쓰지도 바꾸지도 않는다. `ros2 launch scripts/sim_sort_cell.launch.py` 로    │
│   Gazebo 창만 떠 있으면 된다.                                                │
│ [보기 전용] 창을 읽기만 한다. Gazebo 창에 입력(클릭·키)을 보내지 않고, 창을   │
│   옮기거나 크기를 바꾸지 않는다. 시점 회전·확대는 Gazebo 창에서 직접 한다.   │
│ [정식 경로 아님] 설계의 "C 표시 연결"(영상 3개 + 결과 JSON, TCP 명세 부록 B, │
│   미승인)이 아니며 그것을 대체하지 않는다. 검출 주석·측정값은 여기로 오지    │
│   않는다. 카메라 중계 방식(gz_view.py)은 `--view camera` 로 남겨 두었다.     │
└────────────────────────────────────────────────────────────────────────────┘

[제약]
  - Gazebo 창이 최소화되면 Windows 가 그림을 주지 않는다. 이때는 「수신 중단」으로 표시한다.
    다른 창에 가려지는 것은 괜찮다(대시보드가 위에 있어도 된다).
  - 같은 PC 화면에 Gazebo 창이 떠 있어야 한다. 다른 PC 의 Gazebo 는 이 방식으로 볼 수 없다.
  - 창 테두리 밖 여백(그림자 영역, 검은색)은 잘라 낸다. 제목줄과 Gazebo 도구 패널은 화면의 일부라 그대로 둔다.
[방식] Win32 PrintWindow(PW_RENDERFULLCONTENT). 표준 라이브러리 ctypes 만 쓴다(새 의존성 없음).
  한 장에 약 10 ms(실측 2026-11-13, 1076x942 창). GUI 스레드의 타이머에서 부른다.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

from PySide6.QtCore import QObject, QRect, QTimer, Signal
from PySide6.QtGui import QImage

TITLE_KEYWORD = "gazebo"      # 창 제목에 이 말이 들어 있으면 Gazebo 창으로 본다(대소문자 무시)
CAPTURE_MS = 66               # 초당 약 15장
SEARCH_MS = 1500              # 창이 없을 때 다시 찾는 간격
STALE_MS = 1000               # 이보다 오래 새 그림이 없으면 중단으로 본다
PW_RENDERFULLCONTENT = 2      # GPU 로 그린 창도 내용까지 받는다

_user32 = ctypes.windll.user32
_gdi32 = ctypes.windll.gdi32
_user32.GetWindowDC.restype = wintypes.HDC
_user32.GetWindowDC.argtypes = [wintypes.HWND]
_user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
_user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
_user32.IsWindow.argtypes = [wintypes.HWND]
_user32.IsIconic.argtypes = [wintypes.HWND]
_gdi32.CreateCompatibleDC.restype = wintypes.HDC
_gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
_gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
_gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
_gdi32.SelectObject.restype = wintypes.HGDIOBJ
_gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
_gdi32.GetDIBits.argtypes = [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT,
                             ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT]
_gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
_gdi32.DeleteDC.argtypes = [wintypes.HDC]


class _BitmapInfoHeader(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG), ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]


def find_gazebo_window() -> tuple[int, str] | None:
    """보이는 최상위 창 가운데 제목에 TITLE_KEYWORD 가 든 첫 창의 (핸들, 제목). 없으면 None."""
    found: list[tuple[int, str]] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(handle, _param):
        length = _user32.GetWindowTextLengthW(handle)
        if length and _user32.IsWindowVisible(handle):
            buffer = ctypes.create_unicode_buffer(length + 1)
            _user32.GetWindowTextW(handle, buffer, length + 1)
            # 이 대시보드 자신(제목에 Gazebo 가 없다)은 걸리지 않는다
            if TITLE_KEYWORD in buffer.value.lower():
                found.append((handle, buffer.value))
        return True

    _user32.EnumWindows(visit, 0)
    return found[0] if found else None


def grab_window(handle: int) -> QImage | None:
    """창 하나를 떠서 QImage 로 돌려준다. 실패하면 None. 창에는 아무 것도 보내지 않는다."""
    rect = wintypes.RECT()
    if not _user32.GetWindowRect(handle, ctypes.byref(rect)):
        return None
    width, height = rect.right - rect.left, rect.bottom - rect.top
    if width <= 0 or height <= 0:
        return None
    window_dc = _user32.GetWindowDC(handle)
    if not window_dc:
        return None
    memory_dc = _gdi32.CreateCompatibleDC(window_dc)
    bitmap = _gdi32.CreateCompatibleBitmap(window_dc, width, height)
    previous = _gdi32.SelectObject(memory_dc, bitmap)
    try:   # GDI 자원은 성공·실패와 무관하게 아래 finally 에서 반납한다
        if not _user32.PrintWindow(handle, memory_dc, PW_RENDERFULLCONTENT):
            return None
        pixels = ctypes.create_string_buffer(width * height * 4)
        info = _BitmapInfoHeader(ctypes.sizeof(_BitmapInfoHeader), width, -height, 1, 32, 0, 0, 0, 0, 0, 0)  # 음수 높이 = 위에서 아래로
        if not _gdi32.GetDIBits(memory_dc, bitmap, 0, height, pixels, ctypes.byref(info), 0):
            return None
        return QImage(pixels, width, height, width * 4, QImage.Format.Format_RGB32).copy()   # copy: pixels 수명과 분리
    finally:
        _gdi32.SelectObject(memory_dc, previous)
        _gdi32.DeleteObject(bitmap)
        _gdi32.DeleteDC(memory_dc)
        _user32.ReleaseDC(handle, window_dc)


def content_rect(image: QImage) -> QRect:
    """창 테두리 밖의 검은 여백(WSLg 그림자 영역)을 뺀 영역. 가운데 가로·세로 줄을 따라 안쪽으로 훑는다."""
    width, height = image.width(), image.height()
    middle_x, middle_y = width // 2, height // 2
    limit_x, limit_y = width // 4, height // 4   # 여백은 가장자리 근처에만 있다. 화면 안쪽의 검은 그림은 자르지 않는다

    def dark(x: int, y: int) -> bool:
        return (image.pixel(x, y) & 0xFFFFFF) == 0

    left = next((x for x in range(limit_x) if not dark(x, middle_y)), 0)
    right = next((x for x in range(width - 1, width - 1 - limit_x, -1) if not dark(x, middle_y)), width - 1)
    top = next((y for y in range(limit_y) if not dark(middle_x, y)), 0)
    bottom = next((y for y in range(height - 1, height - 1 - limit_y, -1) if not dark(middle_x, y)), height - 1)
    return QRect(left, top, right - left + 1, bottom - top + 1)


class GzWindowCapture(QObject):
    """Gazebo 창을 찾아 주기적으로 떠서 내보낸다. GzViewReceiver 와 같은 신호·메서드를 가진다(창 코드가 구분하지 않는다)."""

    state_changed = Signal(str)       # "disconnected"(창 없음) · "connected"(창 찾음)
    frame_received = Signal(QImage)   # 떠 온 화면. 보간·합성 없이 그대로

    title = "Gazebo 창 캡처"
    source_name = "Gazebo Sim 창"
    help_text = ("Gazebo 창을 찾지 못했습니다. WSL 에서 ros2 launch scripts/sim_sort_cell.launch.py 로 "
                 "Gazebo 를 띄우세요. 창이 뜨면 자동으로 붙습니다.")
    stale_label = "Gazebo 창 최소화됨"

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.state = "disconnected"
        self.frame_count = 0
        self.window_title = ""
        self._handle = 0
        self._crop: QRect | None = None
        self._crop_size = (0, 0)
        self._last_frame_at: float | None = None
        self._timer = QTimer(self, interval=SEARCH_MS)
        self._timer.timeout.connect(self._tick)

    def start(self) -> None:
        self._tick()
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        self._release()

    def frame_age_ms(self) -> int | None:
        if self._last_frame_at is None:
            return None
        return int((time.monotonic() - self._last_frame_at) * 1000)

    def is_stale(self) -> bool:
        age = self.frame_age_ms()
        return age is None or age > STALE_MS

    def _tick(self) -> None:
        if not self._handle:
            match = find_gazebo_window()
            if match is None:
                return
            self._handle, self.window_title = match
            self._last_frame_at = None
            self._timer.setInterval(CAPTURE_MS)
            self._set_state("connected")
        if not _user32.IsWindow(self._handle):   # Gazebo 를 끄면 창이 사라진다. 다시 찾는다
            self._release()
            return
        if _user32.IsIconic(self._handle):       # 최소화 중에는 그림이 없다. 옛 화면을 새 것처럼 보내지 않는다
            return
        image = grab_window(self._handle)
        if image is None:
            return
        if (image.width(), image.height()) != self._crop_size:   # 창 크기가 바뀔 때만 여백을 다시 잰다
            self._crop_size = (image.width(), image.height())
            self._crop = content_rect(image)
        self.frame_count += 1
        self._last_frame_at = time.monotonic()
        self.frame_received.emit(image.copy(self._crop))

    def _release(self) -> None:
        self._handle = 0
        self._timer.setInterval(SEARCH_MS)
        self._set_state("disconnected")

    def _set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.state_changed.emit(state)
