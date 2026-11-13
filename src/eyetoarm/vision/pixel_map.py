"""ROI·리사이즈 처리 좌표를 원본 픽셀 좌표로 되돌리는 함수 — A 내부 잠정 규약.

근거: A 상세설계 §5.1, 공통 자료형 관리 초안 §3.1 `PixelToSourceTransform`.
표준 라이브러리와 NumPy만 쓴다. 공개 함수는 잘못된 인자에 `ValueError`를 낸다.

좌표 규약
- Transform = (scale_x, scale_y, offset_x, offset_y).
  x_source = scale_x * x + offset_x, y_source = scale_y * y + offset_y.
- ROI = (x, y, width, height), 해상도 = (width, height). `source_wh`는 ROI를 자르기 전
  원본 프레임의 해상도다.
- 점은 raw 픽셀이며 왜곡 보정·작업 좌표(m/mm) 변환은 하지 않는다.
- 리사이즈 복원은 배율만 곱하는 단순 스케일이다. `cv2.resize`가 쓰는 화소 중심 역사상
  `(u + 0.5) * scale - 0.5`와 같지 않다(배율 2에서 0.5 px 차이). 이 규약은 A 협업 요청 초안
  B-01·B-02 확인 전의 A 내부 잠정 규약이며 B 계산용 확정 계약이 아니다.
- 이 변환을 관측에 붙일지 따로 기록할지는 미결이므로 함수 반환값으로만 다룬다.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np

Transform = tuple[float, float, float, float]


def _is_int(value) -> bool:
    """bool을 뺀 정수인지 확인한다."""
    return isinstance(value, Integral) and not isinstance(value, bool)


def _check_wh(wh, name: str) -> tuple[int, int]:
    """(폭, 높이)가 양의 정수 두 개인지 확인해 int 튜플로 돌려준다."""
    if len(wh) != 2 or not all(_is_int(v) and v > 0 for v in wh):
        raise ValueError(f"{name} must be two positive integers (width, height): {wh!r}")
    return int(wh[0]), int(wh[1])


def _check_transform(t) -> Transform:
    """변환 (sx, sy, ox, oy)가 유한 실수이고 배율이 양수인지 확인한다."""
    if len(t) != 4 or not all(isinstance(v, Real) and not isinstance(v, bool) for v in t):
        raise ValueError(f"transform must be four real numbers: {t!r}")

    sx, sy, ox, oy = (float(v) for v in t)

    if not (math.isfinite(sx) and math.isfinite(sy) and sx > 0 and sy > 0):
        raise ValueError(f"transform scales must be positive finite numbers: {t!r}")

    if not (math.isfinite(ox) and math.isfinite(oy)):
        raise ValueError(f"transform offsets must be finite: {t!r}")

    return sx, sy, ox, oy


def _check_points(points) -> np.ndarray:
    """점 배열이 (N, 2) 유한 실수인지 확인해 돌려준다."""
    arr = np.asarray(points)

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2): {arr.shape}")

    if not np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.complexfloating):
        raise ValueError(f"points must be real numbers: {arr.dtype}")

    if not np.all(np.isfinite(arr)):
        raise ValueError("points must be finite")

    return arr


def validate_roi(roi: tuple[int, int, int, int], source_wh: tuple[int, int]) -> None:
    """ROI가 원본 안에 있는지 검사한다. 0 <= x, 0 <= y, w > 0, h > 0, x + w <= W, y + h <= H.

    Raises:
        ValueError: ROI·해상도가 정수가 아니거나 ROI가 비었거나 원본을 벗어남.
    """
    width, height = _check_wh(source_wh, "source_wh")

    if len(roi) != 4 or not all(_is_int(v) for v in roi):
        raise ValueError(f"roi must be four integers (x, y, width, height): {roi!r}")

    x, y, w, h = (int(v) for v in roi)

    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
        raise ValueError(f"roi {roi!r} is outside source {width}x{height} or empty")


def roi_transform(roi: tuple[int, int, int, int], source_wh: tuple[int, int]) -> Transform:
    """ROI 안 좌표 → 원본 좌표 변환 (1, 1, x, y)."""
    validate_roi(roi, source_wh)
    return 1.0, 1.0, float(roi[0]), float(roi[1])


def resize_transform(src_wh: tuple[int, int], dst_wh: tuple[int, int]) -> Transform:
    """리사이즈한 영상(dst) 좌표 → 리사이즈 전(src) 좌표 변환. 가로·세로 배율은 따로 계산한다."""
    sw, sh = _check_wh(src_wh, "src_wh")
    dw, dh = _check_wh(dst_wh, "dst_wh")
    return sw / dw, sh / dh, 0.0, 0.0


def compose(outer: Transform, inner: Transform) -> Transform:
    """inner를 먼저, outer를 나중에 적용하는 합성 변환."""
    osx, osy, oox, ooy = _check_transform(outer)
    isx, isy, iox, ioy = _check_transform(inner)
    return osx * isx, osy * isy, osx * iox + oox, osy * ioy + ooy


def to_source(points: np.ndarray, transform: Transform) -> np.ndarray:
    """처리 좌표 (N, 2) → 원본 좌표 (N, 2) float32. 입력 배열은 바꾸지 않는다."""
    sx, sy, ox, oy = _check_transform(transform)
    arr = _check_points(points).astype(np.float64)
    out = np.empty(arr.shape, dtype=np.float32)
    out[:, 0] = arr[:, 0] * sx + ox
    out[:, 1] = arr[:, 1] * sy + oy
    return out


def from_source(points: np.ndarray, transform: Transform) -> np.ndarray:
    """원본 좌표 (N, 2) → 처리 좌표 (N, 2) float32. `to_source`의 역변환."""
    sx, sy, ox, oy = _check_transform(transform)
    arr = _check_points(points).astype(np.float64)
    out = np.empty(arr.shape, dtype=np.float32)
    out[:, 0] = (arr[:, 0] - ox) / sx
    out[:, 1] = (arr[:, 1] - oy) / sy
    return out


def inside_source(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """원본 영상 안(0 <= x < width, 0 <= y < height)인 점은 True인 (N,) bool 배열. 점을 고치지 않는다."""
    w, h = _check_wh((width, height), "width/height")
    arr = _check_points(points)
    return (arr[:, 0] >= 0) & (arr[:, 0] < w) & (arr[:, 1] >= 0) & (arr[:, 1] < h)
