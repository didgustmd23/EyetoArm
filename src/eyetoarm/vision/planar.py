"""OpenCV 전처리 설정 로더와 표시용(raw) 평면 윤곽 후보 추출 — A 내부 잠정안.

근거: A 상세설계 §5.1(공통 프레임)·§7(평면 검출 처리 1~3단계, 설정 식별자),
`vision/pixel_map.py`(단순 스케일 복원 잠정 규약). 설정 필드·처리 순서·식별자 정규화는
A 내부 잠정안이며, 이 모듈의 `PlanarConfig`·`detect_planar`가 그 정본이다.

- 결과는 표시용이다. `eligible`은 영상 필터(면적 범위·경계 미접촉) 통과라는 뜻일 뿐이며
  B 계산용 유효 판정·관측면·클래스·규격·보정 좌표가 아니다. 후보를 자동 선정하지 않는다.
- 입력은 공통 `common.types.FramePacket`이다(A 확장 `inputs.base.FramePacket`도 받는다).
- 이 모듈의 설정·결과 자료형(`PlanarConfig`·`PlanarCandidate`·`PlanarResult`)은 A 내부 표시용이다.
  D API·B 계약이나 공통 `PlanarObservation`·`WorkPlanarResult`가 아니다.
- 표본 경로·정답·라벨을 읽지 않는다. 모든 입력에 같은 설정을 적용한다.
- Qt·ROS를 import하지 않는다.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from numbers import Integral, Real

import cv2
import numpy as np

from eyetoarm.common.types import FramePacket, TimestampBase
from eyetoarm.vision.pixel_map import (
    Transform,
    compose,
    resize_transform,
    roi_transform,
    to_source,
    validate_roi,
)

FORMAT_VERSION = 1
STATUS = "provisional_display_only"
_FIXED = {
    "resize_interpolation": "INTER_AREA",
    "contour_retrieval": "RETR_EXTERNAL",
    "contour_approximation": "CHAIN_APPROX_SIMPLE",
}
_POLARITY_FLAG = {"dark": cv2.THRESH_BINARY_INV, "bright": cv2.THRESH_BINARY}
_THRESHOLD_METHODS = ("otsu", "fixed")
_REAL_FIELDS = ("fixed_threshold", "min_area_source_px2", "max_area_source_px2")
_SIDES = ("left", "top", "right", "bottom")
_CLIP_ORDER = ("source_edge", "roi_edge")


def _is_int(v) -> bool:
    return isinstance(v, Integral) and not isinstance(v, bool)


def _to_finite_float(v, name: str) -> float:
    if not isinstance(v, Real) or isinstance(v, bool):
        raise ValueError(f"{name} must be a number: {v!r}")

    try:
        f = float(v)
    except OverflowError as exc:
        raise ValueError(f"{name} is out of float range") from exc

    if not math.isfinite(f):
        raise ValueError(f"{name} must be finite: {v!r}")

    return f


def _int_tuple(v, n: int, name: str, positive_from: int) -> tuple[int, ...]:
    """정수 n개 배열. 인덱스 `positive_from` 이후는 양수, 그 앞은 0 이상."""
    if not isinstance(v, (list, tuple)) or len(v) != n or not all(_is_int(x) for x in v):
        raise ValueError(f"{name} must be {n} integers: {v!r}")

    out = tuple(int(x) for x in v)

    for i, x in enumerate(out):
        if (i >= positive_from and x <= 0) or x < 0:
            raise ValueError(f"{name} has an out-of-range value: {v!r}")

    return out


def _kernel(v, name: str) -> int:
    if not _is_int(v) or not (v == 0 or (v >= 3 and v % 2 == 1)):
        raise ValueError(f"{name} must be 0 or an odd integer >= 3: {v!r}")
    return int(v)


@dataclass(frozen=True)
class PlanarConfig:
    """검증한 전처리·필터 설정의 변경 불가 스냅샷(A 상세설계 §7 「소스별 설정」).

    생성자가 모든 값을 검증하고 정규화한다(직접 생성해도 검증을 우회하지 않는다). 잘못된 값은 ValueError.
    배열은 튜플, 실수 필드는 float로 보관한다. `config_sha256`은 정규화한 값에서 계산한다.
    값을 바꾸려면 새 설정을 로드하거나 새로 생성한다.
    """

    format_version: int
    status: str
    roi: tuple[int, int, int, int] | None
    resize_wh: tuple[int, int] | None
    resize_interpolation: str
    gaussian_blur_ksize: int
    polarity: str
    threshold_method: str
    fixed_threshold: float | None
    morph_open_ksize: int
    contour_retrieval: str
    contour_approximation: str
    min_area_source_px2: float
    max_area_source_px2: float | None
    border_margin_processed_px: int
    config_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        put = lambda k, v: object.__setattr__(self, k, v)  # noqa: E731

        if not _is_int(self.format_version) or self.format_version != FORMAT_VERSION:
            raise ValueError(f"format_version must be {FORMAT_VERSION}: {self.format_version!r}")

        if self.status != STATUS:
            raise ValueError(f"status must be {STATUS!r}: {self.status!r}")

        for key, value in _FIXED.items():
            if getattr(self, key) != value:
                raise ValueError(f"{key} must be {value!r}: {getattr(self, key)!r}")

        put("roi", None if self.roi is None else _int_tuple(self.roi, 4, "roi", 2))
        put("resize_wh", None if self.resize_wh is None else _int_tuple(self.resize_wh, 2, "resize_wh", 0))
        put("gaussian_blur_ksize", _kernel(self.gaussian_blur_ksize, "gaussian_blur_ksize"))
        put("morph_open_ksize", _kernel(self.morph_open_ksize, "morph_open_ksize"))

        if not isinstance(self.polarity, str) or self.polarity not in _POLARITY_FLAG:
            raise ValueError(f"polarity must be 'dark' or 'bright': {self.polarity!r}")

        if not isinstance(self.threshold_method, str) or self.threshold_method not in _THRESHOLD_METHODS:
            raise ValueError(f"threshold_method must be 'otsu' or 'fixed': {self.threshold_method!r}")

        if self.threshold_method == "otsu":
            if self.fixed_threshold is not None:
                raise ValueError("fixed_threshold must be null when threshold_method is 'otsu'")
        else:
            if self.fixed_threshold is None:
                raise ValueError("fixed_threshold is required when threshold_method is 'fixed'")

            t = _to_finite_float(self.fixed_threshold, "fixed_threshold")

            if not 0.0 <= t <= 255.0:
                raise ValueError(f"fixed_threshold must be in [0, 255]: {self.fixed_threshold!r}")

            put("fixed_threshold", t)

        lo = _to_finite_float(self.min_area_source_px2, "min_area_source_px2")

        if lo <= 0.0:
            raise ValueError(f"min_area_source_px2 must be > 0: {self.min_area_source_px2!r}")

        put("min_area_source_px2", lo)

        if self.max_area_source_px2 is not None:
            hi = _to_finite_float(self.max_area_source_px2, "max_area_source_px2")

            if not hi > lo:
                raise ValueError("max_area_source_px2 must be greater than min_area_source_px2")

            put("max_area_source_px2", hi)

        m = self.border_margin_processed_px

        if not _is_int(m) or m < 0:
            raise ValueError(f"border_margin_processed_px must be an integer >= 0: {m!r}")

        put("border_margin_processed_px", int(m))
        put("config_sha256", _canonical_sha256(self))


def _config_keys() -> tuple[str, ...]:
    return tuple(f.name for f in PlanarConfig.__dataclass_fields__.values() if f.init)


def _canonical_sha256(c: PlanarConfig) -> str:
    """키 정렬·구분자 고정·allow_nan=False JSON(UTF-8)의 sha256. 실수 필드는 이미 float."""
    doc = {k: (list(v) if isinstance(v, tuple) else v) for k in _config_keys() for v in [getattr(c, k)]}
    text = json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _reject_duplicates(pairs):
    out = {}

    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate key: {k!r}")

        out[k] = v

    return out


def _reject_constant(name):
    raise ValueError(f"non-standard JSON constant: {name}")


def _parse_float(text: str) -> float:
    f = float(text)

    if not math.isfinite(f):
        raise ValueError(f"non-finite number: {text}")

    return f


def load_config(path: str | os.PathLike) -> PlanarConfig:
    """JSON 설정 파일을 읽어 검증한 `PlanarConfig`를 돌려준다.

    파일 열기·읽기 오류는 OSError 그대로 올린다. 구문·키·타입·값 오류는 ValueError.
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()

    try:
        doc = json.loads(text, object_pairs_hook=_reject_duplicates,
                         parse_constant=_reject_constant, parse_float=_parse_float)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc

    if not isinstance(doc, dict):
        raise ValueError("config must be a JSON object")

    keys = set(_config_keys())
    missing, unknown = keys - doc.keys(), doc.keys() - keys

    if missing or unknown:
        raise ValueError(f"config keys mismatch: missing={sorted(missing)} unknown={sorted(unknown)}")

    return PlanarConfig(**doc)


@dataclass(frozen=True)
class PlanarCandidate:
    """윤곽 하나(표시용). 좌표·면적·경계 상자는 원본 raw 픽셀이다.

    - `index`: `cv2.findContours` 출력 순번(정렬하지 않음).
    - `points_source`: 원본 raw 좌표 float32 (N, 2), 읽기 전용.
    - `bbox_source`: (x_min, y_min, x_max, y_max) 원본 px.
    - `border_sides`: 처리 영상 기준 접촉 변(left, top, right, bottom 순).
    - `clip_reasons`: 접촉 변이 원본 변이면 `source_edge`, 원본 안쪽 ROI 변이면 `roi_edge`(이 순서).
    - `area_reject_reason`: None, `area_below_min`, `area_above_max`.
    - `eligible`: 면적 통과이고 경계 미접촉. B 유효 판정이 아니다.
    """

    index: int
    points_source: np.ndarray
    area_source_px2: float
    bbox_source: tuple[float, float, float, float]
    border_sides: tuple[str, ...]
    clip_reasons: tuple[str, ...]
    area_reject_reason: str | None
    eligible: bool


@dataclass(frozen=True)
class PlanarResult:
    """프레임 하나의 표시용 윤곽 추출 결과. 후보 0개·복수·잘림을 그대로 담는다."""

    run_id: str
    source_id: str
    image_seq: int
    capture_generation: int
    timestamp_ms: int
    timestamp_base: TimestampBase
    received_monotonic_ns: int
    config_sha256: str
    processed_wh: tuple[int, int]
    transform: Transform
    threshold_used: float
    raw_contour_count: int
    candidates: tuple[PlanarCandidate, ...]
    eligible_count: int
    process_ms: float
    display_only: bool = field(default=True, init=False)
    observed_face: None = field(default=None, init=False)
    calc_contour: None = field(default=None, init=False)
    valid: bool = field(default=False, init=False)
    selected_index: None = field(default=None, init=False)


def _check_frame(frame: FramePacket) -> np.ndarray:
    img = frame.image

    if not isinstance(img, np.ndarray):
        raise ValueError(f"frame.image must be a numpy array: {type(img).__name__}")

    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"frame.image must be BGR uint8 (H, W, 3): {img.dtype} {img.shape}")

    h, w = img.shape[:2]

    if h == 0 or w == 0:
        raise ValueError("frame.image is empty")

    if frame.width != w or frame.height != h:
        raise ValueError(f"frame width/height {frame.width}x{frame.height} != image {w}x{h}")

    return img


def detect_planar(frame: FramePacket, config: PlanarConfig) -> PlanarResult:
    """아래 1)~10) 순서로 표시용 윤곽 후보를 추출한다. 입력 배열·메타데이터를 바꾸지 않는다.

    잘못된 프레임·원본 밖 ROI는 ValueError, `PlanarConfig`가 아닌 설정은 TypeError다
    (입력 사건이나 무검출로 바꾸지 않는다).
    """
    if not isinstance(config, PlanarConfig):
        raise TypeError("config must be a PlanarConfig")

    t0 = time.monotonic_ns()
    img = _check_frame(frame)
    src_h, src_w = img.shape[:2]

    # 1) ROI
    roi = config.roi if config.roi is not None else (0, 0, src_w, src_h)
    validate_roi(roi, (src_w, src_h))
    rx, ry, rw, rh = roi
    work = img[ry:ry + rh, rx:rx + rw]
    t_roi = roi_transform(roi, (src_w, src_h))

    # 2) 리사이즈(INTER_AREA)
    if config.resize_wh is not None:
        work = cv2.resize(work, config.resize_wh, interpolation=cv2.INTER_AREA)
        t_resize = resize_transform((rw, rh), config.resize_wh)
    else:
        t_resize = (1.0, 1.0, 0.0, 0.0)

    transform = compose(t_roi, t_resize)

    # 3) 회색
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    ph, pw = gray.shape[:2]

    # 4) Gaussian blur
    if config.gaussian_blur_ksize:
        k = config.gaussian_blur_ksize
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    # 5) 이진화
    flag = _POLARITY_FLAG[config.polarity]

    if config.threshold_method == "otsu":
        used, binary = cv2.threshold(gray, 0, 255, flag | cv2.THRESH_OTSU)
    else:
        used, binary = cv2.threshold(gray, config.fixed_threshold, 255, flag)

    # 6) morphology open
    if config.morph_open_ksize:
        k = config.morph_open_ksize
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((k, k), np.uint8), iterations=1)

    # 7) 윤곽
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    m = config.border_margin_processed_px
    side_is_source = {"left": rx == 0, "top": ry == 0, "right": rx + rw == src_w, "bottom": ry + rh == src_h}
    candidates = []

    for i, cnt in enumerate(contours):
        p = cnt.reshape(-1, 2)

        # 8) 원본 좌표 복원
        pts = to_source(p, transform)
        pts.setflags(write=False)

        # 9) 원본 면적
        area = float(cv2.contourArea(pts))

        if area < config.min_area_source_px2:
            reject = "area_below_min"
        elif config.max_area_source_px2 is not None and area > config.max_area_source_px2:
            reject = "area_above_max"
        else:
            reject = None

        # 10) 처리 영상 경계 접촉(전체 점)
        touch = {
            "left": bool(np.any(p[:, 0] <= m)),
            "top": bool(np.any(p[:, 1] <= m)),
            "right": bool(np.any(p[:, 0] >= pw - 1 - m)),
            "bottom": bool(np.any(p[:, 1] >= ph - 1 - m)),
        }
        sides = tuple(s for s in _SIDES if touch[s])
        kinds = {"source_edge" if side_is_source[s] else "roi_edge" for s in sides}
        clips = tuple(k for k in _CLIP_ORDER if k in kinds)
        bbox = (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))
        candidates.append(PlanarCandidate(
            index=i, points_source=pts, area_source_px2=area, bbox_source=bbox,
            border_sides=sides, clip_reasons=clips, area_reject_reason=reject,
            eligible=reject is None and not sides,
        ))

    process_ms = (time.monotonic_ns() - t0) / 1e6
    return PlanarResult(
        run_id=frame.run_id, source_id=frame.source_id, image_seq=frame.image_seq,
        capture_generation=frame.capture_generation, timestamp_ms=frame.timestamp_ms,
        timestamp_base=frame.timestamp_base, received_monotonic_ns=frame.received_monotonic_ns,
        config_sha256=config.config_sha256, processed_wh=(pw, ph), transform=transform,
        threshold_used=float(used), raw_contour_count=len(contours), candidates=tuple(candidates),
        eligible_count=sum(c.eligible for c in candidates), process_ms=process_ms,
    )
