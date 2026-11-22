"""B 규격의 볼트 종류 고르기(#270): 끝점 두 점을 종류마다 그 종류 높이로 투영해 길이로 종류를 고른다.

규칙은 B `_classify`와 같다(#285, B-12). 규격에 `length_min`·`length_max`가 있으면 `length_min ≤ L ≤ length_max`,
없으면 `|L − total_length| ≤ length_tolerance`인 종류 중 `|L − total_length|`가 가장 작은 종류이고,
차가 같으면 B 규격 파일 순서가 앞선 종류다. 다만 길이 L은 B와 달리 **그 종류의 `plane_z`로 투영한 길이**다.
시야 보정(`view_bias`)과 W 추적 높이 고르기(`world_tracking`)가 이 모듈 하나를 쓴다. B 파일은 읽기만 한다.
ROS·Qt·정답을 import하지 않는다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from eyetoarm.transforms.planar_transform import pixels_to_w


@dataclass(frozen=True)
class BoltSpec:
    """B 볼트 종류 하나: 이름, 라벨, 기준 높이(m), 총길이(m), 길이 허용(m), 길이 범위(m, 선택).

    범위(`length_min_m`·`length_max_m`)가 있으면 판정은 범위로 하고 허용은 쓰지 않는다(B `_classify`, B-12).
    범위가 없으면 허용이 있어야 한다.
    """

    name: str
    label: str
    plane_z_m: float
    total_length_m: float
    length_tolerance_m: float | None
    length_min_m: float | None = None
    length_max_m: float | None = None

    def __post_init__(self) -> None:
        """값 검사(아니면 ValueError): 이름·라벨은 비어 있지 않은 문자열, 높이·총길이는 유한 양수, 허용은 유한 0 이상.

        범위는 둘 다 있거나 둘 다 없어야 하고, 있으면 유한 양수이며 `length_min ≤ total_length ≤ length_max`.
        범위가 없으면 허용이 있어야 한다.
        """
        for field_name in ("name", "label"):
            v = getattr(self, field_name)

            if not isinstance(v, str) or not v:
                raise ValueError(f"BoltSpec.{field_name} must be a non-empty string: {v!r}")

        ranged = (self.length_min_m is not None, self.length_max_m is not None)

        if ranged[0] != ranged[1]:
            raise ValueError(f"BoltSpec {self.name}: length_min and length_max must be given together: "
                             f"{self.length_min_m!r}, {self.length_max_m!r}")

        if not ranged[0] and self.length_tolerance_m is None:
            raise ValueError(f"BoltSpec {self.name}: needs length_tolerance or length_min/length_max")

        checks = [("plane_z_m", False), ("total_length_m", False)]
        checks += [("length_tolerance_m", True)] if self.length_tolerance_m is not None else []
        checks += [("length_min_m", False), ("length_max_m", False)] if ranged[0] else []

        for field_name, zero_ok in checks:
            v = getattr(self, field_name)

            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or (v == 0 and not zero_ok):
                raise ValueError(f"BoltSpec {self.name}.{field_name} must be a finite positive number: {v!r}")

            object.__setattr__(self, field_name, float(v))

        if ranged[0] and not self.length_min_m <= self.total_length_m <= self.length_max_m:
            raise ValueError(f"BoltSpec {self.name}: needs length_min <= total_length <= length_max: "
                             f"{self.length_min_m!r} <= {self.total_length_m!r} <= {self.length_max_m!r}")

    @property
    def ranged(self) -> bool:
        """길이 범위로 판정하는가."""
        return self.length_min_m is not None

    def accept_range_m(self) -> tuple[float, float]:
        """판정 길이 범위(m): 범위가 있으면 (min, max), 없으면 총길이 ± 허용."""
        if self.ranged:
            return self.length_min_m, self.length_max_m

        return self.total_length_m - self.length_tolerance_m, self.total_length_m + self.length_tolerance_m

    def accepts(self, length_m: float) -> bool:
        """B `_classify`와 같은 판정: 범위가 있으면 min ≤ L ≤ max, 없으면 |L − total| ≤ 허용."""
        if self.ranged:
            return self.length_min_m <= length_m <= self.length_max_m

        return abs(length_m - self.total_length_m) <= self.length_tolerance_m

    def to_json(self) -> dict:
        """기록용 사전."""
        return {"name": self.name, "label": self.label, "plane_z_m": self.plane_z_m,
                "total_length_m": self.total_length_m, "length_tolerance_m": self.length_tolerance_m,
                "length_min_m": self.length_min_m, "length_max_m": self.length_max_m}


def bolt_specs_from(specs: dict) -> tuple[BoltSpec, ...]:
    """B 규격 사전(`load_specs` 결과) → 볼트 종류들(파일 순서). 높이는 종류의 `plane_z`, 없으면 최상위 `plane_z`.

    길이 범위(`length_min`·`length_max`)는 있으면 읽는다(B-12). 볼트 종류가 없거나, 필요한 값이 빠지거나,
    값이 `BoltSpec` 검사에 걸리면 ValueError(시작 거부용).
    """
    try:
        rows = specs["specs"]
        default_z = specs["plane_z"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"part specs need 'specs' and 'plane_z': {exc}") from exc

    out = []

    if not isinstance(rows, dict):
        raise ValueError(f"part specs 'specs' must be an object: {type(rows).__name__}")

    for name, spec in rows.items():
        if not isinstance(spec, dict):
            raise ValueError(f"part specs {name} is not an object: {type(spec).__name__}")

        if str(spec.get("class_name", "")).strip().lower() != "bolt":
            continue

        try:
            out.append(BoltSpec(name=name, label=spec.get("label", name), plane_z_m=spec.get("plane_z", default_z),
                                total_length_m=spec["total_length"], length_tolerance_m=spec.get("length_tolerance"),
                                length_min_m=spec.get("length_min"), length_max_m=spec.get("length_max")))
        except KeyError as exc:
            raise ValueError(f"part specs {name} lacks {exc}") from exc

    if not out:
        raise ValueError("part specs have no bolt spec")

    return tuple(out)


def length_at(endpoints_px, K, dist, T_W_C, plane_z_m: float) -> float | None:
    """끝점 (2, 2) 픽셀 → `plane_z_m` 평면 투영 길이(m). 유한하지 않으면 None."""
    with np.errstate(divide="ignore", invalid="ignore"):   # 특이 투영의 0 나눗셈은 아래 유한성 검사로 거른다
        w = pixels_to_w(np.asarray(endpoints_px, dtype=np.float64).reshape(2, 2), K, dist, T_W_C, plane_z_m)

    if not np.all(np.isfinite(w)):
        return None

    return float(np.hypot(*(w[1] - w[0])))   # B `compute_bolt_from_endpoints`와 같은 계산(경계값 일치)


@dataclass(frozen=True)
class SpecChoice:
    """고르기 결과. `name`은 허용 안 종류(없으면 None), `nearest`는 허용과 무관하게 차가 가장 작은 종류.

    `lengths_m`은 종류별 (이름, 그 종류 높이로 잰 길이 m 또는 None).
    """

    name: str | None
    nearest: str | None
    lengths_m: tuple[tuple[str, float | None], ...]

    def lengths_mm(self) -> dict:
        """기록용 {이름: 길이 mm(소수 3자리) 또는 None}."""
        return {n: None if L is None else round(L * 1000.0, 3) for n, L in self.lengths_m}


def choose_spec(lengths_m, specs: tuple[BoltSpec, ...]) -> SpecChoice:
    """종류별 길이(m, `specs`와 같은 순서, 투영 실패는 None) → 고르기 결과(B `_classify`와 같은 비교)."""
    best, best_err, near, near_err = None, None, None, None

    for spec, L in zip(specs, lengths_m):
        if L is None:
            continue

        err = abs(L - spec.total_length_m)

        if spec.accepts(L) and (best_err is None or err < best_err):
            best, best_err = spec.name, err

        if near_err is None or err < near_err:
            near, near_err = spec.name, err

    return SpecChoice(best, near, tuple((s.name, L) for s, L in zip(specs, lengths_m)))


def select_spec(endpoints_px, K, dist, T_W_C, specs: tuple[BoltSpec, ...]) -> SpecChoice:
    """끝점 → 종류마다 그 종류 높이로 투영한 길이 → `choose_spec`."""
    return choose_spec([length_at(endpoints_px, K, dist, T_W_C, s.plane_z_m) for s in specs], specs)


__all__ = ["BoltSpec", "SpecChoice", "bolt_specs_from", "choose_spec", "length_at", "select_spec"]
