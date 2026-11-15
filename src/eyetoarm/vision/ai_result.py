"""AI 분할 모델 출력·표시 결과 자료형(YOLO 연결 T2). A 내부형이며 공통 `PlanarObservation`을 바꾸지 않는다.

근거: 계획 `docs/superpowers/plans/2026-11-15-yolo-pipeline.md` r2 §4.3.1(필드·소유권·파생 수·polygon 규칙)·§4.4
(candidate_count = in_scope 수), A 상세설계 §5.2(표시용 전체 마스크와 계산용 윤곽 구분), §5.4(모델 오류 종류).

- `RawSegOutput`: 어댑터가 모델 결과에서 뽑아 형식 계약을 확인한 원시 배열(crop 좌표). 변환 뒤 버린다.
- `AiInstance`: 모델이 낸 인스턴스 하나의 표시 정보. polygon 유무는 **기하 기준**이며 대표 거부 사유와 따로 정한다
  (r2 P4). 가짜 다각형을 만들지 않는다. `index`는 모델 출력 순서이고 B의 최종 후보 선택을 뜻하지 않는다.
- `AiSegResult`: 작업 하나의 표시 결과. 다각형과 숫자만 들고 영상·마스크 원본·모델 Results를 붙잡지 않는다.
- `ModelPrepareError`: 모델 준비 실패의 분류(MODEL_LOAD_FAILED·MODEL_CONFIG_INVALID만).
- `AiJobConfigError`: 작업 중 발견한 설정 불일치(MODEL_CONFIG_INVALID, 작업 값).
- 거부 이름(`NON_TARGET_CLASS` 등)은 표시 결과 안의 A 내부 값이며 A §5.4 판정 사유 표에 넣지 않는다.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from eyetoarm.common.types import Contour, InvalidationKind, _owned_contour

# 대표 거부 사유(판정 순서대로, 계획 §4.3 5단계)
NON_TARGET_CLASS = "NON_TARGET_CLASS"
LOW_CONFIDENCE = "LOW_CONFIDENCE"
EMPTY_MASK = "EMPTY_MASK"
SMALL_MASK = "SMALL_MASK"
REJECTIONS = (NON_TARGET_CLASS, LOW_CONFIDENCE, EMPTY_MASK, SMALL_MASK)

# polygon 허용 조합(계획 §4.3.1 표): 필수 / 금지 / 둘 다 허용
_POLYGON_REQUIRED = (None, SMALL_MASK)
_POLYGON_FORBIDDEN = (EMPTY_MASK,)

_OUT_OF_SCOPE = (NON_TARGET_CLASS, LOW_CONFIDENCE)
_PREPARE_KINDS = (InvalidationKind.MODEL_LOAD_FAILED, InvalidationKind.MODEL_CONFIG_INVALID)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")  # fullmatch로만 쓴다(끝 개행 거부, r1 F2)


def _is_int(v) -> bool:
    """bool을 뺀 정수인지 확인한다."""
    return isinstance(v, Integral) and not isinstance(v, bool)


def _nonneg_int(v, name: str) -> int:
    """0 이상 정수."""
    if not _is_int(v) or v < 0:
        raise ValueError(f"{name} must be an integer >= 0: {v!r}")

    return int(v)


def _finite(v, name: str, *, lo: float | None = 0.0, hi: float | None = None) -> float:
    """bool이 아닌 유한 실수(범위 [lo, hi])를 float로."""
    if not isinstance(v, Real) or isinstance(v, bool) or not math.isfinite(float(v)):
        raise ValueError(f"{name} must be a finite number: {v!r}")

    f = float(v)

    if (lo is not None and f < lo) or (hi is not None and f > hi):
        raise ValueError(f"{name} out of range [{lo}, {hi}]: {v!r}")

    return f


def _wh(v, name: str) -> tuple[int, int]:
    """(폭, 높이) 양의 정수."""
    if not isinstance(v, (list, tuple)) or len(v) != 2 or not all(_is_int(x) and x > 0 for x in v):
        raise ValueError(f"{name} must be 2 positive integers: {v!r}")

    return (int(v[0]), int(v[1]))


def _owned(array, dtype, name: str) -> np.ndarray:
    """지정 dtype의 C 순서 소유 복사·읽기 전용 배열."""
    out = np.array(array, dtype=dtype, copy=True, order="C")
    out.setflags(write=False)

    return out


def _shoelace_area(points: np.ndarray) -> float:
    """닫힌 다각형의 부호 없는 면적(신발끈 공식, float64로 계산)."""
    p = points.astype(np.float64)
    x, y = p[:, 0], p[:, 1]

    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2.0


@dataclass(frozen=True)
class RawSegOutput:
    """모델 출력 원시 배열(crop 좌표). 어댑터가 결과 객체에서 뽑은 뒤 형식 계약을 여기서 확인한다.

    - K개 인스턴스: `boxes_xyxy` (K,4) float32, `confidence` (K,) float32, `class_index` (K,) int64 ≥ 0,
      `masks` (K,h,w) bool. retina 계약상 (h,w)는 crop (높이, 폭)과 같아야 한다(`crop_wh`=(폭, 높이)).
    - 값은 모두 유한해야 한다. K=0(무검출)은 정상이다. 어긋나면 ValueError(실행기가 INFERENCE_ERROR로 정리).
    """

    boxes_xyxy: np.ndarray
    confidence: np.ndarray
    class_index: np.ndarray
    masks: np.ndarray
    crop_wh: tuple[int, int]
    infer_ms: float

    def __post_init__(self) -> None:
        """형식 계약을 검사하고 배열을 소유 복사로 정규화한다."""
        put = lambda k, v: object.__setattr__(self, k, v)  # noqa: E731
        w, h = _wh(self.crop_wh, "crop_wh")
        put("crop_wh", (w, h))
        put("infer_ms", _finite(self.infer_ms, "infer_ms"))

        boxes, conf = np.asarray(self.boxes_xyxy), np.asarray(self.confidence)
        cls, masks = np.asarray(self.class_index), np.asarray(self.masks)

        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(f"boxes_xyxy must have shape (K, 4): {boxes.shape}")

        k = boxes.shape[0]

        if conf.shape != (k,) or cls.shape != (k,):
            raise ValueError(f"confidence/class_index must have shape ({k},): {conf.shape} {cls.shape}")

        if masks.ndim != 3 or masks.shape[0] != k:
            raise ValueError(f"masks must have shape ({k}, h, w): {masks.shape}")

        if masks.shape[1:] != (h, w):
            raise ValueError(f"mask resolution {masks.shape[1:]} != crop (h, w) {(h, w)} (retina contract)")

        if masks.dtype != np.bool_:
            raise ValueError(f"masks must be bool: {masks.dtype}")

        if not np.issubdtype(cls.dtype, np.integer) or (k and int(cls.min()) < 0):
            raise ValueError(f"class_index must be integers >= 0: {cls.dtype}")

        # int64 저장에서 값이 바뀌지 않아야 한다(uint64 2**63 등, r1 F3)
        if k and int(cls.max()) > np.iinfo(np.int64).max:
            raise ValueError("class_index exceeds int64 range")

        if not (np.all(np.isfinite(boxes)) and np.all(np.isfinite(conf))):
            raise ValueError("boxes_xyxy/confidence must be finite")

        # float32로 바꾸기 전에 범위를 본다(넘는 값은 변환 시 inf가 됨, r1 F3)
        f32_max = float(np.finfo(np.float32).max)

        for name, arr in (("boxes_xyxy", boxes), ("confidence", conf)):
            if arr.size and float(np.max(np.abs(arr.astype(np.float64)))) > f32_max:
                raise ValueError(f"{name} exceeds float32 range")

        stored_boxes = _owned(boxes, np.float32, "boxes_xyxy")
        stored_conf = _owned(conf, np.float32, "confidence")

        # float32 저장에서도 유한해야 한다(1e39 → inf 방지, r1 F3). 검사는 저장한 값으로 한다
        if not (np.all(np.isfinite(stored_boxes)) and np.all(np.isfinite(stored_conf))):
            raise ValueError("boxes_xyxy/confidence must be finite in float32")

        put("boxes_xyxy", stored_boxes)
        put("confidence", stored_conf)
        put("class_index", _owned(cls, np.int64, "class_index"))
        put("masks", _owned(masks, np.bool_, "masks"))

    @property
    def count(self) -> int:
        """인스턴스 수 K."""
        return int(self.boxes_xyxy.shape[0])


@dataclass(frozen=True)
class AiInstance:
    """모델 인스턴스 하나의 표시 정보(원본 픽셀 좌표, 계획 §4.3.1).

    polygon 규칙(r2 P4, 대표 거부 사유와 독립):
    - 기하 있음(바깥 윤곽 점 3개 이상·면적 > 0)일 때만 polygon이 있고, 없으면 None·`clipped=False`다.
    - `rejected` None·`SMALL_MASK` → polygon 필수, `EMPTY_MASK` → None 필수, `NON_TARGET_CLASS`·`LOW_CONFIDENCE`
      → 둘 다 허용(기하 판정보다 앞서 걸림). 비대상·점수 미달도 클래스·점수·사유를 보존한다.
    """

    index: int
    class_index: int
    class_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]
    polygon: Contour | None
    area_px: float
    component_count: int
    hole_count: int
    clipped: bool
    rejected: str | None

    def __post_init__(self) -> None:
        """필드를 검증·정규화하고 polygon 허용 조합을 검사한다."""
        put = lambda k, v: object.__setattr__(self, k, v)  # noqa: E731
        put("index", _nonneg_int(self.index, "index"))
        put("class_index", _nonneg_int(self.class_index, "class_index"))

        if not isinstance(self.class_name, str) or not self.class_name:
            raise ValueError(f"class_name must be a non-empty string: {self.class_name!r}")

        put("confidence", _finite(self.confidence, "confidence", lo=0.0, hi=1.0))

        box = self.box_xyxy

        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(f"box_xyxy must be 4 numbers: {box!r}")

        x1, y1, x2, y2 = (_finite(v, "box_xyxy", lo=None) for v in box)

        if x2 < x1 or y2 < y1:
            raise ValueError(f"box_xyxy must satisfy x1 <= x2 and y1 <= y2: {box!r}")

        put("box_xyxy", (x1, y1, x2, y2))
        put("area_px", _finite(self.area_px, "area_px"))
        put("component_count", _nonneg_int(self.component_count, "component_count"))
        put("hole_count", _nonneg_int(self.hole_count, "hole_count"))

        if not isinstance(self.clipped, bool):
            raise ValueError(f"clipped must be a bool: {self.clipped!r}")

        if self.rejected is not None and self.rejected not in REJECTIONS:
            raise ValueError(f"rejected must be None or one of {REJECTIONS}: {self.rejected!r}")

        # 다각형: 소유 복사(float32 (N,2), 읽기 전용) + 점 3개 이상 + 유한
        polygon = _owned_contour(self.polygon, "polygon")

        if polygon is not None and (polygon.shape[0] < 3 or not np.all(np.isfinite(polygon))):
            raise ValueError(f"polygon must have >= 3 finite points: {polygon.shape}")

        # 면적 0(일직선·같은 점 반복)은 기하 없음이다(계획 §4.3.1 「점 3개 이상·면적 > 0」, r1 F4). 임의 epsilon 없음
        if polygon is not None and _shoelace_area(polygon) == 0.0:
            raise ValueError("polygon must have a non-zero area (degenerate geometry is polygon=None)")

        put("polygon", polygon)

        if polygon is None and self.clipped:
            raise ValueError("clipped requires a polygon")

        if self.rejected in _POLYGON_REQUIRED and polygon is None:
            raise ValueError(f"rejected={self.rejected!r} requires a polygon (geometry present)")

        if self.rejected in _POLYGON_FORBIDDEN and polygon is not None:
            raise ValueError(f"rejected={self.rejected!r} must not carry a polygon")

    @property
    def in_scope(self) -> bool:
        """대상 클래스이고 점수 기준을 넘었는가(기하 거부·잘림 포함). candidate_count가 세는 범위."""
        return self.rejected not in _OUT_OF_SCOPE

    @property
    def eligible(self) -> bool:
        """거부가 없고 잘리지 않았는가."""
        return self.rejected is None and not self.clipped


@dataclass(frozen=True)
class AiSegResult:
    """작업 하나의 AI 표시 결과(계획 §4.3.1). `instances`는 모델이 낸 전부(비대상 포함)이며 index 순이다.

    `config_sha256`은 계산에 쓴 설정(= 작업 시작 설정 `job.detector_config_id`)이다. 코어가 대조한다.
    시간: `infer_ms`는 predict 호출만, `post_ms`는 변환, `process_ms`는 어댑터 전체(ms).
    """

    instances: tuple[AiInstance, ...]
    model_id: str
    config_sha256: str
    roi: tuple[int, int, int, int] | None
    crop_wh: tuple[int, int]
    infer_ms: float
    post_ms: float
    process_ms: float

    def __post_init__(self) -> None:
        """필드를 검증한다. 인스턴스 번호는 0부터 빠짐없이 순서대로여야 한다."""
        put = lambda k, v: object.__setattr__(self, k, v)  # noqa: E731

        if not isinstance(self.instances, (list, tuple)) or not all(isinstance(i, AiInstance) for i in self.instances):
            raise ValueError("instances must be a sequence of AiInstance")

        instances = tuple(self.instances)

        if [i.index for i in instances] != list(range(len(instances))):
            raise ValueError(f"instance indexes must be 0..K-1 in order: {[i.index for i in instances]}")

        put("instances", instances)

        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError(f"model_id must be a non-empty string: {self.model_id!r}")

        if not isinstance(self.config_sha256, str) or not _SHA256_RE.fullmatch(self.config_sha256):
            raise ValueError(f"config_sha256 must be 64 lowercase hex characters: {self.config_sha256!r}")

        roi = self.roi

        if roi is not None:
            if not isinstance(roi, (list, tuple)) or len(roi) != 4 or not all(_is_int(v) for v in roi):
                raise ValueError(f"roi must be 4 integers or None: {roi!r}")

            put("roi", tuple(int(v) for v in roi))

        put("crop_wh", _wh(self.crop_wh, "crop_wh"))

        for name in ("infer_ms", "post_ms", "process_ms"):
            put(name, _finite(getattr(self, name), name))

    @property
    def candidate_count(self) -> int:
        """in_scope 인스턴스 수(대상 클래스·점수 통과, 기하 거부·잘림 포함). `len(instances)`가 아니다."""
        return sum(1 for i in self.instances if i.in_scope)

    @property
    def eligible_count(self) -> int:
        """eligible 인스턴스 수."""
        return sum(1 for i in self.instances if i.eligible)


class ModelPrepareError(Exception):
    """모델 준비(파일 확인·해시·import·생성·검사) 실패의 분류. kind는 MODEL_LOAD_FAILED·MODEL_CONFIG_INVALID만."""

    def __init__(self, kind: InvalidationKind | str, cause: str) -> None:
        """분류와 원인을 받는다. 그 밖의 kind·빈 원인은 ValueError."""
        try:
            kind = InvalidationKind(kind)
        except ValueError as exc:
            raise ValueError(f"unknown invalidation kind: {kind!r}") from exc

        if kind not in _PREPARE_KINDS:
            raise ValueError(f"ModelPrepareError kind must be one of {[k.value for k in _PREPARE_KINDS]}: {kind}")

        if not isinstance(cause, str) or not cause:
            raise ValueError(f"cause must be a non-empty string: {cause!r}")

        super().__init__(f"{kind.value}: {cause}")
        self.kind = kind
        self.cause = cause


class AiJobConfigError(Exception):
    """작업 중 발견한 검출 설정 불일치(A §5.4 `MODEL_CONFIG_INVALID`, 「해당 작업의 값」).

    예: 설정 ROI가 작업 프레임 밖, 작업 시작 설정 해시가 어댑터 설정과 다름, 모델이 class_map 밖 클래스를 냄.
    실행기는 이 예외를 `fail(job, MODEL_CONFIG_INVALID, cause)`로 정리한다(R3). 일반 예외(INFERENCE_ERROR)와 구분한다.
    """

    def __init__(self, cause: str) -> None:
        """원인 문자열(비어 있지 않음)."""
        if not isinstance(cause, str) or not cause:
            raise ValueError(f"cause must be a non-empty string: {cause!r}")

        super().__init__(cause)
        self.cause = cause


__all__ = ["AiInstance", "AiJobConfigError", "AiSegResult", "EMPTY_MASK", "LOW_CONFIDENCE", "ModelPrepareError", "NON_TARGET_CLASS",
           "REJECTIONS", "RawSegOutput", "SMALL_MASK"]
