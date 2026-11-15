"""AI 분할 검출 설정(`AiSegConfig`) — 검증·정규화·안정 설정 ID·model_id·엄격 JSON 읽기(YOLO 연결 T1).

근거: 계획 `docs/superpowers/plans/2026-11-15-yolo-pipeline.md` r2 §4.1·§4.8, A 상세설계 §5.4(결과에 영향을 주는 검출 설정:
검출 방식(OpenCV/AI), AI 모델·가중치·대상 클래스·신뢰도 기준·입력 크기·추론 장치), §7.1(모델 설정 기록 항목·AI 의존성 선택적 구성).

- 순수 Python이다. torch·ultralytics를 import하지 않고 가중치 파일도 읽지 않는다(파일 확인·해시 대조는 준비 단계).
- 생성자가 모든 값을 검증·정규화한다. 잘못된 값은 ValueError. `config_sha256`은 정규 JSON(키 정렬·구분자 고정·
  `weights_path` 제외·`method` 포함)의 sha256이다. 같은 가중치 파일이 다른 경로에 있어도 같은 설정이다.
- `method` 키가 정규형에 있으므로 `PlanarConfig`(정규형에 method 없음)와 설정 ID 공간이 겹치지 않는다.
- 필드 이름·범위는 계획의 **A 내부 제안**이다. `conf_threshold` 등 값은 운영 확정값이 아니며 설정 파일로 조절한다.
- 1단계(표시용)는 `retina_masks=True`만 받는다. 마스크가 predict에 넘긴 crop 해상도로 나오므로 A는 ROI 오프셋만 더한다.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from numbers import Integral, Real

FORMAT_VERSION = 1
METHOD = "ai"

# 사용자 확정 비교 후보 3개(A 상세설계 §7.1, 2026-11-12). 라벨이며 실제 확인은 로드 때 task·names로 한다
MODEL_FAMILIES = ("yolov8n-seg", "yolo11n-seg", "yolo26n-seg")

# 전체 문자열 검사는 fullmatch로 한다(`$`는 끝 개행 앞에서도 맞으므로 쓰지 않음, r1 F2)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_DEVICE_RE = re.compile(r"cpu|[0-9]+")
_CLASS_KEY_RE = re.compile(r"0|[1-9][0-9]*")
_IMGSZ_STEP, _IMGSZ_MIN, _IMGSZ_MAX = 32, 32, 4096
_MAX_DET_MAX = 1000


def _is_int(v) -> bool:
    """bool을 뺀 정수인지 확인한다."""
    return isinstance(v, Integral) and not isinstance(v, bool)


def _finite_float(v, name: str) -> float:
    """bool이 아닌 유한 실수를 float로 바꾼다. 아니면 ValueError."""
    if not isinstance(v, Real) or isinstance(v, bool):
        raise ValueError(f"{name} must be a number: {v!r}")

    try:
        f = float(v)
    except OverflowError as exc:
        raise ValueError(f"{name} is out of float range") from exc

    if not math.isfinite(f):
        raise ValueError(f"{name} must be finite: {v!r}")

    return f


def _text(v, name: str) -> str:
    """비어 있지 않은 문자열."""
    if not isinstance(v, str) or not v:
        raise ValueError(f"{name} must be a non-empty string: {v!r}")

    return v


def _class_key(k) -> int:
    """class_map 키: 0 이상 정수 또는 그 십진 문자열("0", "12"; "01"·"-1"·bool 거부)."""
    if _is_int(k) and k >= 0:
        return int(k)

    if isinstance(k, str) and _CLASS_KEY_RE.fullmatch(k):
        return int(k)

    raise ValueError(f"class_map key must be an integer >= 0 (or its decimal string): {k!r}")


def _normalize_class_map(v) -> tuple[tuple[int, str], ...]:
    """dict(번호→이름) 또는 이미 정규화된 (번호, 이름) 튜플을 번호 순 튜플로 만든다. 이름 중복·빈 맵 거부."""
    if isinstance(v, dict):
        items = list(v.items())
    elif isinstance(v, tuple) and all(isinstance(p, tuple) and len(p) == 2 for p in v):
        items = list(v)
    else:
        raise ValueError(f"class_map must be an object of index -> name: {v!r}")

    if not items:
        raise ValueError("class_map must not be empty")

    out: dict[int, str] = {}

    for k, name in items:
        idx = _class_key(k)

        if idx in out:
            raise ValueError(f"class_map has a duplicate index: {k!r}")

        out[idx] = _text(name, f"class_map[{k!r}]")

    if len(set(out.values())) != len(out):
        raise ValueError(f"class_map names must be unique: {sorted(out.values())}")

    return tuple(sorted(out.items()))


def _roi(v) -> tuple[int, int, int, int] | None:
    """[x, y, w, h] 정수(x·y ≥ 0, w·h > 0) 또는 None. 프레임 해상도와의 대조는 작업 때(pixel_map.validate_roi)."""
    if v is None:
        return None

    if not isinstance(v, (list, tuple)) or len(v) != 4 or not all(_is_int(x) for x in v):
        raise ValueError(f"roi must be 4 integers or null: {v!r}")

    x, y, w, h = (int(a) for a in v)

    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError(f"roi has an out-of-range value: {v!r}")

    return (x, y, w, h)


@dataclass(frozen=True)
class AiSegConfig:
    """검증한 AI 분할 검출 설정의 변경 불가 스냅샷(계획 §4.1). 직접 생성해도 검증을 우회하지 않는다.

    - `class_map`은 (모델 클래스 번호, 이름) 번호 순 튜플, `target_classes`는 이름 정렬 튜플로 보관한다.
    - `weights_path`는 기록용이며 설정 ID에 넣지 않는다. 가중치 동일성은 `weights_sha256`으로 판단한다.
    - `model_id` = 「모델 계열@가중치 sha256 앞 16자」(계획 §4.1 제안). 가짜 ID를 만들지 않는다.
    """

    format_version: int
    method: str
    model_family: str
    weights_path: str
    weights_sha256: str
    class_map: tuple[tuple[int, str], ...]
    target_classes: tuple[str, ...]
    conf_threshold: float
    iou_threshold: float
    imgsz: int
    max_det: int
    device: str
    half: bool
    retina_masks: bool
    roi: tuple[int, int, int, int] | None
    min_mask_area_px: int
    border_margin_px: int
    config_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        """모든 필드를 검증·정규화하고 `config_sha256`을 채운다. 잘못된 값은 ValueError."""
        put = lambda k, v: object.__setattr__(self, k, v)  # noqa: E731

        if not _is_int(self.format_version) or self.format_version != FORMAT_VERSION:
            raise ValueError(f"format_version must be {FORMAT_VERSION}: {self.format_version!r}")

        if self.method != METHOD:
            raise ValueError(f"method must be {METHOD!r}: {self.method!r}")

        if self.model_family not in MODEL_FAMILIES:
            raise ValueError(f"model_family must be one of {MODEL_FAMILIES}: {self.model_family!r}")

        _text(self.weights_path, "weights_path")

        if not isinstance(self.weights_sha256, str) or not _SHA256_RE.fullmatch(self.weights_sha256):
            raise ValueError(f"weights_sha256 must be 64 lowercase hex characters: {self.weights_sha256!r}")

        # 클래스: 번호·이름 정규화, 대상은 class_map 이름의 중복 없는 비어 있지 않은 부분집합
        class_map = _normalize_class_map(self.class_map)
        put("class_map", class_map)
        names = {name for _, name in class_map}

        targets = self.target_classes

        if isinstance(targets, str) or not isinstance(targets, (list, tuple)) or not targets:
            raise ValueError(f"target_classes must be a non-empty list of class names: {targets!r}")

        if len(set(targets)) != len(targets) or not all(isinstance(t, str) and t in names for t in targets):
            raise ValueError(f"target_classes must be unique names from class_map: {targets!r}")

        put("target_classes", tuple(sorted(targets)))

        # 임계값: 0 < conf < 1, 0 < iou <= 1
        conf = _finite_float(self.conf_threshold, "conf_threshold")

        if not 0.0 < conf < 1.0:
            raise ValueError(f"conf_threshold must be in (0, 1): {self.conf_threshold!r}")

        iou = _finite_float(self.iou_threshold, "iou_threshold")

        if not 0.0 < iou <= 1.0:
            raise ValueError(f"iou_threshold must be in (0, 1]: {self.iou_threshold!r}")

        put("conf_threshold", conf)
        put("iou_threshold", iou)

        # 입력 크기·최대 검출 수
        if (not _is_int(self.imgsz) or self.imgsz % _IMGSZ_STEP or not _IMGSZ_MIN <= self.imgsz <= _IMGSZ_MAX):
            raise ValueError(f"imgsz must be a multiple of {_IMGSZ_STEP} in [{_IMGSZ_MIN}, {_IMGSZ_MAX}]: "
                             f"{self.imgsz!r}")

        if not _is_int(self.max_det) or not 1 <= self.max_det <= _MAX_DET_MAX:
            raise ValueError(f"max_det must be an integer in [1, {_MAX_DET_MAX}]: {self.max_det!r}")

        put("imgsz", int(self.imgsz))
        put("max_det", int(self.max_det))

        # 장치·정밀도·마스크 방식
        if not isinstance(self.device, str) or not _DEVICE_RE.fullmatch(self.device):
            raise ValueError(f"device must be 'cpu' or a CUDA index string ('0', '1', ...): {self.device!r}")

        if not isinstance(self.half, bool):
            raise ValueError(f"half must be a bool: {self.half!r}")

        if self.retina_masks is not True:
            raise ValueError("retina_masks must be true (stage 1 coordinate path: masks at crop resolution)")

        # ROI·마스크 필터
        put("roi", _roi(self.roi))

        for name in ("min_mask_area_px", "border_margin_px"):
            value = getattr(self, name)

            if not _is_int(value) or value < 0:
                raise ValueError(f"{name} must be an integer >= 0: {value!r}")

            put(name, int(value))

        put("config_sha256", hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest())

    @property
    def class_names(self) -> dict[int, str]:
        """모델 클래스 번호 → 이름(새 dict)."""
        return dict(self.class_map)

    @property
    def model_id(self) -> str:
        """관측·기록에 쓰는 모델 식별자: 「모델 계열@가중치 sha256 앞 16자」."""
        return f"{self.model_family}@{self.weights_sha256[:16]}"

    def init_fields(self) -> dict:
        """생성자 인자(정규화된 값). 같은 값으로 다시 만들면 같은 해시다."""
        return {f: getattr(self, f) for f in _config_keys()}

    def canonical_json(self) -> str:
        """설정 ID 계산용 정규 JSON(키 정렬·구분자 고정·allow_nan=False, `weights_path` 제외)."""
        doc = {}

        for key in _config_keys():
            if key == "weights_path":
                continue

            value = getattr(self, key)

            if key == "class_map":
                value = {str(i): name for i, name in value}
            elif isinstance(value, tuple):
                value = list(value)

            doc[key] = value

        return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _config_keys() -> tuple[str, ...]:
    """설정 파일에 쓰는 필드 이름(생성자 인자, `config_sha256` 제외)."""
    return tuple(f.name for f in AiSegConfig.__dataclass_fields__.values() if f.init)


def _reject_duplicates(pairs):
    """JSON 객체의 중복 키를 거부한다(`object_pairs_hook`)."""
    out = {}

    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate key: {k!r}")

        out[k] = v

    return out


def _reject_constant(name):
    """JSON 비표준 상수(NaN·Infinity)를 거부한다(`parse_constant`)."""
    raise ValueError(f"non-standard JSON constant: {name}")


def _parse_float(text: str) -> float:
    """JSON 실수를 float로 읽고 비유한 값(`1e400` 등)을 거부한다(`parse_float`)."""
    f = float(text)

    if not math.isfinite(f):
        raise ValueError(f"non-finite number: {text}")

    return f


def load_ai_config(path: str | os.PathLike) -> AiSegConfig:
    """JSON 설정 파일을 읽어 검증한 `AiSegConfig`를 돌려준다.

    파일 열기·읽기 오류는 OSError 그대로 올린다. 구문·키·타입·값 오류는 ValueError. 가중치 파일은 읽지 않는다.
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

    return AiSegConfig(**doc)


__all__ = ["AiSegConfig", "FORMAT_VERSION", "MODEL_FAMILIES", "load_ai_config"]
