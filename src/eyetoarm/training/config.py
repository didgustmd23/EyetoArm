"""학습기 공통·후보 설정 검증: 엄격 JSON, 후보별 학습 인자 거부, 저장소 상대 경로, config_id.

후보 설정은 출처·해시만 담고 학습 인자 키가 있으면 ConfigError다(같은 조건 보장).
`kind`는 `comparison` 또는 `smoke`(`fraction` 허용, 비교 묶음 제외). torch·ultralytics를 import하지 않는다.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any

FORMAT_VERSION = 1
KINDS = ("comparison", "smoke")
FAMILIES = ("yolov8n-seg", "yolo11n-seg", "yolo26n-seg")  # 비교 후보 계열
# 계열 ↔ 공식 nano 분할 가중치 파일. 다른 크기·과제 모델은 공식 url이어도 받지 않는다
FAMILY_WEIGHTS = {"yolov8n-seg": "yolov8n-seg.pt", "yolo11n-seg": "yolo11n-seg.pt", "yolo26n-seg": "yolo26n-seg.pt"}
TRAIN_KEYS = ("imgsz", "epochs", "seed", "deterministic", "batch", "device", "workers", "cache", "plots")
SMOKE_ONLY_KEYS = ("fraction",)
# 명시할 때만 ultralytics에 넘기는 선택 학습 키. 생략하면 ultralytics 기본값, config_id 불변
OPTIONAL_TRAIN_KEYS = ("hsv_v",)
WATCH_KEYS = ("watch_interval_s", "mem_available_stop_kib", "kill_grace_s", "train_deadline_s", "eval_deadline_s",
              "reap_timeout_s", "monitor_cmd_timeout_s")
REQUIRED_ENV = {"YOLO_AUTOINSTALL": "false", "YOLO_OFFLINE": "1"}
# 후보 선택 모드: 생략과 "single"을 구분하려고 기본값은 객체로 둔다
_MODE_DEFAULT = object()
SELECTION_MODES = (_MODE_DEFAULT, "single")
OFFICIAL_URL = "https://github.com/ultralytics/assets/releases/download/"
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_EXP_RE = re.compile(r"S1-[0-9]{2}")
_FILE_RE = re.compile(r"[A-Za-z0-9._-]+\.pt")


class ConfigError(ValueError):
    """설정 형식·값 오류."""


def _freeze(v):
    """검증된 값을 바꿀 수 없게 고정한다(dict → MappingProxyType, list → tuple, 재귀)."""
    if isinstance(v, dict):
        return MappingProxyType({k: _freeze(x) for k, x in v.items()})
    if isinstance(v, (list, tuple)):
        return tuple(_freeze(x) for x in v)
    return v


def thaw(v) -> Any:
    """고정된 값을 JSON용 dict·list로 되돌린다(새 사본)."""
    if isinstance(v, MappingProxyType) or isinstance(v, dict):
        return {k: thaw(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [thaw(x) for x in v]
    return v


def _is_int(v) -> bool:
    """bool 아닌 정수."""
    return isinstance(v, Integral) and not isinstance(v, bool)


def _pos_int(v, name: str, lo: int = 1, hi: int | None = None) -> int:
    """범위 안 정수."""
    if not _is_int(v) or v < lo or (hi is not None and v > hi):
        raise ConfigError(f"{name} must be an integer in [{lo}, {hi}]: {v!r}")
    return int(v)


def _pos_num(v, name: str) -> float:
    """유한 양수. 거대 정수(float 변환 넘침)도 ConfigError."""
    f = _as_float(v)
    if f is None or not math.isfinite(f) or f <= 0:
        raise ConfigError(f"{name} must be a finite number > 0: {v!r}")
    return f


def _as_float(v) -> float | None:
    """bool이 아닌 실수를 float로. 변환할 수 없으면(거대 정수의 OverflowError 포함) None."""
    if isinstance(v, bool) or not isinstance(v, Real):
        return None
    try:
        return float(v)
    except (OverflowError, ValueError):
        return None


def _rel_path(v, name: str) -> str:
    """저장소 상대 경로(절대 경로·`..` 거부)."""
    if not isinstance(v, str) or not v or os.path.isabs(v) or ".." in v.replace("\\", "/").split("/"):
        raise ConfigError(f"{name} must be a repository-relative path without '..': {v!r}")
    return v


def _exact_keys(d, required, name: str, optional=()) -> None:
    """dict의 키가 정확히 required(+optional)인지."""
    if not isinstance(d, dict):
        raise ConfigError(f"{name} must be an object")
    missing = set(required) - d.keys()
    unknown = d.keys() - set(required) - set(optional)
    if missing or unknown:
        raise ConfigError(f"{name} keys mismatch: missing={sorted(missing)} unknown={sorted(unknown)}")


@dataclass(frozen=True)
class CommonConfig:
    """검증한 공통 설정. dict·list 필드는 고정(MappingProxyType·tuple)되어 바꿀 수 없다."""

    kind: str
    train: dict
    data: dict
    watch: dict
    weights_dir: str
    outputs_dir: str
    required_env: dict
    raw: dict


@dataclass(frozen=True)
class Candidate:
    """후보 하나(출처·해시만). 학습 인자를 갖지 않는다."""

    exp_id: str
    family: str
    weights_file: str
    source: dict
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        """필드 검증 뒤 source를 고정한다."""
        if not isinstance(self.exp_id, str) or not _EXP_RE.fullmatch(self.exp_id):
            raise ConfigError(f"exp_id must look like S1-01: {self.exp_id!r}")

        if self.family not in FAMILIES:
            raise ConfigError(f"family must be one of {FAMILIES}: {self.family!r}")

        if not isinstance(self.weights_file, str) or not _FILE_RE.fullmatch(self.weights_file):
            raise ConfigError(f"weights_file must be a bare .pt file name: {self.weights_file!r}")

        if self.weights_file != FAMILY_WEIGHTS[self.family]:
            raise ConfigError(f"family {self.family} requires weights_file {FAMILY_WEIGHTS[self.family]}: "
                              f"{self.weights_file!r}")

        _exact_keys(self.source, ("repo", "tag", "url"), "source")

        url = self.source["url"]

        if (self.source["repo"] != "ultralytics/assets" or not isinstance(self.source["tag"], str)
                or not isinstance(url, str) or url != f"{OFFICIAL_URL}{self.source['tag']}/{self.weights_file}"):
            raise ConfigError(f"source must be the official ultralytics/assets release url: {self.source!r}")

        if not isinstance(self.sha256, str) or not _SHA_RE.fullmatch(self.sha256):
            raise ConfigError(f"sha256 must be 64 lowercase hex: {self.sha256!r}")

        _pos_int(self.size_bytes, "size_bytes")
        object.__setattr__(self, "source", _freeze(dict(self.source)))


@dataclass(frozen=True)
class TrainingConfig:
    """공통 + 후보와 그 정규 JSON sha256(`config_id`). 실행 인자는 `canonical` 스냅샷에서만 만든다."""

    common: CommonConfig
    candidates: tuple[Candidate, ...]
    config_id: str
    canonical: str = ""

    def train_args(self) -> dict:
        """config_id를 계산한 스냅샷에서 새 학습 인자 dict를 만든다(호출자가 바꿔도 설정은 그대로)."""
        return dict(json.loads(self.canonical)["common"]["train"])

    def verify(self) -> bool:
        """스냅샷이 config_id와 맞고, 다시 검증한 공통·후보 값이 지금 객체와 같은가."""
        if hashlib.sha256(self.canonical.encode()).hexdigest() != self.config_id:
            return False
        doc = json.loads(self.canonical)
        again, cands = common_from_dict(doc["common"]), candidates_from_dict(doc["candidates"])
        return (all(thaw(getattr(again, k)) == thaw(getattr(self.common, k)) for k in ("train", "data", "watch"))
                and [(c.exp_id, c.weights_file, c.sha256, thaw(c.source)) for c in cands]
                == [(c.exp_id, c.weights_file, c.sha256, thaw(c.source)) for c in self.candidates])


def common_from_dict(d) -> CommonConfig:
    """공통 설정 dict를 검증한다."""
    _exact_keys(d, ("format_version", "kind", "train", "data", "watch", "weights_dir", "outputs_dir",
                    "required_env"), "common")

    if not _is_int(d["format_version"]) or d["format_version"] != FORMAT_VERSION:
        raise ConfigError(f"format_version must be {FORMAT_VERSION}")
    if d["kind"] not in KINDS:
        raise ConfigError(f"kind must be one of {KINDS}: {d['kind']!r}")

    t = d["train"]
    _exact_keys(t, TRAIN_KEYS, "train", optional=SMOKE_ONLY_KEYS + OPTIONAL_TRAIN_KEYS)
    imgsz = _pos_int(t["imgsz"], "imgsz", 32, 4096)
    if imgsz % 32:
        raise ConfigError(f"imgsz must be a multiple of 32: {imgsz}")
    train = {"imgsz": imgsz, "epochs": _pos_int(t["epochs"], "epochs", 1, 10000),
             "seed": _pos_int(t["seed"], "seed", 0), "batch": _pos_int(t["batch"], "batch", 1, 1024),
             "workers": _pos_int(t["workers"], "workers", 0, 64)}
    for k in ("deterministic", "cache", "plots"):
        if not isinstance(t[k], bool):
            raise ConfigError(f"{k} must be a bool: {t[k]!r}")
        train[k] = t[k]
    if not isinstance(t["device"], str) or not re.fullmatch(r"cpu|[0-9]+", t["device"]):
        raise ConfigError(f"device must be 'cpu' or a CUDA index string: {t['device']!r}")
    train["device"] = t["device"]
    if "fraction" in t:
        if d["kind"] != "smoke":
            raise ConfigError("fraction is allowed only in smoke configs")
        f = _pos_num(t["fraction"], "fraction")
        if f > 1:
            raise ConfigError(f"fraction must be in (0, 1]: {f}")
        train["fraction"] = f
    if "hsv_v" in t:
        # 밝기(V) 증강 폭: bool 제외 유한 실수, ultralytics 분수 키와 같은 [0, 1].
        # 범위 비교를 isfinite보다 먼저 한다: 거대 정수(10**400)는 float 변환 없이 걸러진다
        v = t["hsv_v"]
        if isinstance(v, bool) or not isinstance(v, Real) or not 0 <= v <= 1 or not math.isfinite(v):
            raise ConfigError(f"hsv_v must be a finite number in [0, 1]: {v!r}")
        train["hsv_v"] = float(v)

    data = d["data"]
    _exact_keys(data, ("data_yaml", "split_lists", "class_names", "expected_fingerprint"), "data")
    lists = data["split_lists"]
    if not isinstance(lists, list) or not lists:
        raise ConfigError("split_lists must be a non-empty list")
    names = data["class_names"]
    if not isinstance(names, list) or not names or not all(isinstance(n, str) and n for n in names):
        raise ConfigError("class_names must be a non-empty list of names")
    fp = data["expected_fingerprint"]
    if fp is not None and not isinstance(fp, dict):
        raise ConfigError("expected_fingerprint must be an object or null")
    data_out = {"data_yaml": _rel_path(data["data_yaml"], "data_yaml"),
                "split_lists": [_rel_path(p, "split_lists[]") for p in lists], "class_names": list(names),
                "expected_fingerprint": fp}

    w = d["watch"]
    _exact_keys(w, WATCH_KEYS, "watch")
    watch = {k: _pos_num(w[k], k) for k in WATCH_KEYS}
    watch["mem_available_stop_kib"] = _pos_int(w["mem_available_stop_kib"], "mem_available_stop_kib")

    if d["required_env"] != REQUIRED_ENV:
        raise ConfigError(f"required_env must be {REQUIRED_ENV}: {d['required_env']!r}")

    return CommonConfig(kind=d["kind"], train=_freeze(train), data=_freeze(data_out), watch=_freeze(watch),
                        weights_dir=_rel_path(d["weights_dir"], "weights_dir"),
                        outputs_dir=_rel_path(d["outputs_dir"], "outputs_dir"), required_env=_freeze(REQUIRED_ENV),
                        raw=_freeze(json.loads(json.dumps(d))))


def candidates_from_dict(d) -> tuple[Candidate, ...]:
    """후보 설정 dict를 검증한다. exp_id 유일.

    `selection_mode` 생략은 3계열이 정확히 한 번씩, `"single"`은 후보 정확히 1개. 그 밖의 값은 거부한다.
    """
    _exact_keys(d, ("format_version", "candidates"), "candidates file", optional=("selection_mode",))
    if not _is_int(d["format_version"]) or d["format_version"] != FORMAT_VERSION:
        raise ConfigError(f"format_version must be {FORMAT_VERSION}")

    mode = d["selection_mode"] if "selection_mode" in d else _MODE_DEFAULT
    if mode not in SELECTION_MODES:
        raise ConfigError(f"selection_mode must be one of {SELECTION_MODES[1:]} or omitted: {mode!r}")

    items = d["candidates"]
    if not isinstance(items, list):
        raise ConfigError("candidates must be a list")

    out = []
    for i, c in enumerate(items):
        if not isinstance(c, dict):
            raise ConfigError(f"candidates[{i}] must be an object")
        extra = c.keys() - {"exp_id", "family", "weights_file", "source", "sha256", "size_bytes"}
        if extra & set(TRAIN_KEYS + SMOKE_ONLY_KEYS + OPTIONAL_TRAIN_KEYS) or extra:
            raise ConfigError(f"candidates[{i}] must not carry training args or unknown keys: {sorted(extra)}")
        try:
            out.append(Candidate(**c))
        except TypeError as exc:
            raise ConfigError(f"candidates[{i}]: {exc}") from exc

    if len({c.exp_id for c in out}) != len(out):
        raise ConfigError("exp_id must be unique")

    if mode == "single":
        if len(out) != 1:
            raise ConfigError(f"selection_mode single requires exactly one candidate: {len(out)}")
    elif sorted(c.family for c in out) != sorted(FAMILIES):
        raise ConfigError(f"candidates must contain each of {FAMILIES} exactly once")

    return tuple(out)


def _reject_duplicates(pairs):
    """중복 키 거부."""
    out = {}
    for k, v in pairs:
        if k in out:
            raise ConfigError(f"duplicate key: {k!r}")
        out[k] = v
    return out


def _reject_constant(name):
    """NaN·Infinity 거부."""
    raise ConfigError(f"non-standard JSON constant: {name}")


def _parse_float(text):
    """비유한 수(1e400) 거부."""
    f = float(text)
    if not math.isfinite(f):
        raise ConfigError(f"non-finite number: {text}")
    return f


def _load_json(path):
    """엄격 JSON 읽기. 파일 오류는 OSError 그대로."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        doc = json.loads(text, object_pairs_hook=_reject_duplicates, parse_constant=_reject_constant,
                         parse_float=_parse_float)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ConfigError(f"{path} must contain a JSON object")
    return doc


def load_common(path) -> CommonConfig:
    """공통 설정 파일."""
    return common_from_dict(_load_json(path))


def load_candidates(path) -> tuple[Candidate, ...]:
    """후보 설정 파일."""
    return candidates_from_dict(_load_json(path))


def training_config_from_dicts(common_doc: dict, cand_doc: dict) -> TrainingConfig:
    """두 문서를 검증하고 config_id(정규 JSON sha256)와 스냅샷을 붙인다."""
    common, cands = common_from_dict(common_doc), candidates_from_dict(cand_doc)
    text = json.dumps({"common": common_doc, "candidates": cand_doc}, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)

    return TrainingConfig(common=common, candidates=cands, config_id=hashlib.sha256(text.encode()).hexdigest(),
                          canonical=text)


def load_training_config(common_path, candidates_path) -> TrainingConfig:
    """두 파일을 읽어 검증한다."""
    return training_config_from_dicts(_load_json(common_path), _load_json(candidates_path))


__all__ = ["Candidate", "CommonConfig", "ConfigError", "FAMILIES", "FAMILY_WEIGHTS", "TrainingConfig", "candidates_from_dict",
           "common_from_dict", "load_candidates", "load_common", "load_training_config", "thaw",
           "training_config_from_dicts"]
