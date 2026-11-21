"""작업 좌표(W) 기준 객체 추적: 볼트 끝점(또는 마스크)을 촬영 시각 TF로 벨트 평면(W, mm)에 올려 짝짓는다.

픽셀 추적과 같은 표시·실험 번호이며 분열·스왑을 없앤다는 보장이 아니다. W 위치는 B와 같은 투영이라 TF·보정이 틀리면
같이 틀린다. 단일 소유(추적 스레드 하나)이며 설정값은 모두 개발 임시값이다. ROS·Qt를 import하지 않는다.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field, fields

import cv2
import numpy as np

from eyetoarm.transforms.planar_transform import pixels_to_w
from eyetoarm.vision.arm_occlusion import Occluder, OccluderConfigError, parse_occluders
from eyetoarm.vision.ai_result import AiSegResult
from eyetoarm.vision.tracking import (FrameKey, PartTrackIdIssuer, TrackAssignment, TrackFrame, TrackingContractError,
                                      TrackingInputError, TrackMode, TrackState)

_GEOMETRY_REJECTIONS = ("EMPTY_MASK", "SMALL_MASK")
WORLD_JUMP = "world_jump_suspected"


class WorldTrackerConfigError(ValueError):
    """W 추적 설정 오류."""


@dataclass(frozen=True)
class WorldTrackerConfig:
    """W 짝짓기 시험값(mm·ms·m/s, 이름에 단위). 모두 필수."""

    gate_x_mm: float
    gate_y_mm: float
    length_ratio_max: float
    angle_max_deg: float
    ambiguity_margin: float
    min_hits: int
    max_gap_ms: int
    new_track_min_conf: float
    max_tracks: int
    velocity_alpha: float
    max_belt_speed_mps: float
    dup_merge: bool
    dup_endpoint_mm: float
    jump_min_tracks: int
    coherent_jump_mm: float
    jump_coherence_mm: float
    jump_search_mm: float
    mask_fallback: bool         # 끝점이 없는 볼트를 마스크 W 평균으로 짝짓기
    mask_dup_contain: float     # 마스크가 낀 이중 검출 합치기 포함률 기준

    def to_json(self) -> dict:
        """요약용 사전."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


_INT_KEYS = ("min_hits", "max_gap_ms", "max_tracks", "jump_min_tracks")


def parse_world_tracker(doc) -> WorldTrackerConfig:
    """설정 사전 → 설정. 모르는 키·누락·형·범위 오류는 `WorldTrackerConfigError`."""
    names = {f.name for f in fields(WorldTrackerConfig)}

    if not isinstance(doc, dict):
        raise WorldTrackerConfigError("world tracker config must be an object")

    unknown, missing = set(doc) - names, names - set(doc)

    if unknown or missing:
        raise WorldTrackerConfigError(f"world tracker: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")

    values = {}

    for name in names:
        v = doc[name]

        if name in ("dup_merge", "mask_fallback"):
            if not isinstance(v, bool):
                raise WorldTrackerConfigError(f"{name} must be a bool: {v!r}")
        elif name in _INT_KEYS:
            if type(v) is not int or v < 1:
                raise WorldTrackerConfigError(f"{name} must be an int >= 1: {v!r}")
        elif isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise WorldTrackerConfigError(f"{name} must be a positive number: {v!r}")
        else:
            v = float(v)

        values[name] = v

    if values["velocity_alpha"] > 1.0 or values["new_track_min_conf"] > 1.0 or values["mask_dup_contain"] > 1.0:
        raise WorldTrackerConfigError("velocity_alpha, new_track_min_conf and mask_dup_contain must be <= 1")

    return WorldTrackerConfig(**values)


@dataclass(frozen=True)
class VisibilityConfig:
    """가시성 기반 만료: 화면(여백 `margin_px`) 안인데 못 본 시간만 `max_gap_ms`로 세고, 안 보인 채 `hold_limit_ms`까지 유지.

    팔·그리퍼에 가려진(`occluders` 캡슐) 동안도 공백을 세지 않고 `occluded_limit_ms`까지 유지한다.
    """

    margin_px: float
    hold_limit_ms: int
    occluded_limit_ms: int
    occluders: tuple[Occluder, ...]
    clipped_near_mm: float      # 확정 트랙 예측에서 이 거리 안의 잘린·마스크 후보는 새 트랙을 만들지 않는다
    velocity_window_ms: int     # 링크에 안 보이는 트랙 예측용 벨트 공통 속도의 기준선(이만큼 떨어진 링크 끝점 hit 차)

    def to_json(self) -> dict:
        """요약용 사전."""
        return {"margin_px": self.margin_px, "hold_limit_ms": self.hold_limit_ms, "occluded_limit_ms": self.occluded_limit_ms,
                "occluders": [o.to_json() for o in self.occluders], "clipped_near_mm": self.clipped_near_mm,
                "velocity_window_ms": self.velocity_window_ms}


_VIS_KEYS = {"margin_px", "hold_limit_ms", "occluded_limit_ms", "occluders", "clipped_near_mm", "velocity_window_ms"}


def parse_visibility(doc) -> VisibilityConfig:
    """`visibility` 블록 → 설정. 오류는 `WorldTrackerConfigError`."""
    if not isinstance(doc, dict) or set(doc) != _VIS_KEYS:
        raise WorldTrackerConfigError(f"visibility must have exactly {sorted(_VIS_KEYS)}")

    margin, hold, occ, near = doc["margin_px"], doc["hold_limit_ms"], doc["occluded_limit_ms"], doc["clipped_near_mm"]

    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin < 0:
        raise WorldTrackerConfigError(f"margin_px must be a number >= 0: {margin!r}")

    if isinstance(near, bool) or not isinstance(near, (int, float)) or not math.isfinite(near) or near <= 0:
        raise WorldTrackerConfigError(f"clipped_near_mm must be a positive number: {near!r}")

    for name, v in (("hold_limit_ms", hold), ("occluded_limit_ms", occ), ("velocity_window_ms", doc["velocity_window_ms"])):
        if type(v) is not int or v < 1:
            raise WorldTrackerConfigError(f"{name} must be an int >= 1: {v!r}")

    try:
        occluders = parse_occluders(doc["occluders"])
    except OccluderConfigError as exc:
        raise WorldTrackerConfigError(str(exc)) from exc

    return VisibilityConfig(float(margin), hold, occ, occluders, float(near), doc["velocity_window_ms"])


@dataclass(frozen=True)
class GripperInput:
    """그리퍼 깊이 성분 하나(W, mm). 클래스 없음. `clipped`면 위치 갱신 없이 「보임」으로만 쓴다."""

    index: int
    center_mm: tuple[float, float]
    length_mm: float
    angle_deg: float
    clipped: bool


@dataclass(frozen=True)
class GripperAssignment:
    """그리퍼 성분 배정. `outcome`: gripper_hit | gripper_seen_clipped | gripper_ambiguous | gripper_length_mismatch | gripper_unmatched."""

    index: int
    part_track_id: int | None
    outcome: str


@dataclass(frozen=True)
class GripperUpdate:
    """그리퍼 갱신 결과. 갱신하지 않았으면 `reason`이 있다."""

    timestamp_ms: int
    updated: bool
    reason: str | None
    assignments: tuple[GripperAssignment, ...]
    late: bool = False          # 추적기 시각(마지막 링크)보다 이른 관측: 그 시각 예측으로 짝짓고 시각은 그대로


@dataclass(frozen=True)
class WorldInput:
    """W 추적 입력 하나(스칼라·작은 튜플만). `candidate=False`면 `untracked_reason`을 그대로 낸다."""

    index: int
    class_name: str
    confidence: float
    candidate: bool
    untracked_reason: str | None
    clipped: bool = False
    center_mm: tuple[float, float] | None = None        # 추적에 쓰는 W 중점(보정 뒤, 보정 없으면 원)
    center_raw_mm: tuple[float, float] | None = None    # 보정 전 W 중점(기록용)
    endpoints_mm: tuple[tuple[float, float], tuple[float, float]] | None = None
    length_mm: float | None = None
    angle_deg: float | None = None
    kind: str = "endpoint"                                   # endpoint | mask | none
    mask: tuple | None = field(default=None, compare=False)  # (x0, y0, bool 배열), 포함률 판정용
    plane_z_m: float | None = None                           # W로 올린 높이(m). 후보가 아니면 None
    plane_spec: str | None = None                            # 높이를 고른 B 규격 이름(끝점 입력만)


@dataclass(frozen=True)
class BoltPlanes:
    """B 규격의 볼트 종류별 투영 높이. `specs` = ((이름, 기준 높이 m, 총길이 m), …).

    `default_z_m`은 마스크 입력용이다. 종류는 높이 고르기에만 쓰며 B 판정이 아니다.
    """

    default_z_m: float
    specs: tuple[tuple[str, float, float], ...]

    def __post_init__(self) -> None:
        """값 검사: 유한 양수, 이름은 빈 문자열이 아니고 겹치지 않음, 종류 1개 이상."""
        if not self.specs:
            raise WorldTrackerConfigError("BoltPlanes needs at least one bolt spec")

        names = [n for n, _, _ in self.specs]

        if len(set(names)) != len(names) or not all(isinstance(n, str) and n for n in names):
            raise WorldTrackerConfigError(f"BoltPlanes spec names must be unique non-empty strings: {names}")

        for v in (self.default_z_m, *(z for _, z, _ in self.specs), *(L for _, _, L in self.specs)):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                raise WorldTrackerConfigError(f"BoltPlanes values must be finite positive numbers: {v!r}")

    @property
    def uniform(self) -> bool:
        """종류 높이가 모두 기본 높이와 같은가(그러면 고르기 없이 기본 높이 하나)."""
        return all(z == self.default_z_m for _, z, _ in self.specs)


def mask_of(polygon) -> tuple | None:
    """다각형(원본 픽셀, 3점 이상) → (x0, y0, bool 배열). 정수 반올림 뒤 채움. 없으면 None."""
    if polygon is None or len(polygon) < 3:
        return None

    p = np.round(np.asarray(polygon, dtype=np.float64)).astype(np.int32)
    x0, y0 = p.min(axis=0)
    w, h = p.max(axis=0) - (x0, y0) + 1
    m = np.zeros((int(h), int(w)), dtype=np.uint8)
    cv2.fillPoly(m, [p - (x0, y0)], 1)

    return int(x0), int(y0), m.astype(bool)


def mask_center_w(mask, K, dist, T_W_C, plane_z_m) -> tuple[float, float] | None:
    """마스크 픽셀 전부 → W(mm) 평균. 유한하지 않으면 None."""
    x0, y0, m = mask
    ys, xs = np.nonzero(m)

    if len(xs) == 0:
        return None

    px = np.stack([xs + x0, ys + y0], axis=1).astype(np.float64)
    w = pixels_to_w(px, K, dist, T_W_C, plane_z_m) * 1000.0

    return (float(w[:, 0].mean()), float(w[:, 1].mean())) if np.all(np.isfinite(w)) else None


def containment(a: tuple | None, b: tuple | None) -> float:
    """두 마스크 교집합 / 작은 쪽 면적. 마스크가 없으면 0."""
    if a is None or b is None:
        return 0.0

    (ax, ay, am), (bx, by, bm) = a, b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + am.shape[1], bx + bm.shape[1]), min(ay + am.shape[0], by + bm.shape[0])

    if x1 <= x0 or y1 <= y0:
        return 0.0

    inter = np.logical_and(am[y0 - ay:y1 - ay, x0 - ax:x1 - ax], bm[y0 - by:y1 - by, x0 - bx:x1 - bx]).sum()
    small = min(am.sum(), bm.sum())

    return float(inter / small) if small else 0.0


def _is_bolt(class_name) -> bool:
    return isinstance(class_name, str) and class_name.strip().lower().startswith("bolt")


def _project(endpoints_px, K, dist, T_W_C, plane_z_m) -> np.ndarray | None:
    """끝점 (2, 2) 픽셀 → W (2, 2) mm. 유한하지 않으면 None."""
    w = pixels_to_w(np.asarray(endpoints_px, dtype=np.float64).reshape(2, 2), K, dist, T_W_C, plane_z_m) * 1000.0

    return w if np.all(np.isfinite(w)) else None


def _pick_plane(endpoints_px, K, dist, T_W_C, planes: BoltPlanes) -> tuple[float, str | None]:
    """끝점 볼트의 투영 높이: 그 높이로 올린 W 길이가 종류 총길이에 가장 가까운 종류의 높이.

    높이가 모두 같거나 투영이 안 되면 기본 높이(이름 None). 차가 같으면 `specs` 순서가 앞선 종류다.
    """
    if planes.uniform:
        return planes.default_z_m, None

    best = None

    for name, z_m, total_m in planes.specs:
        w = _project(endpoints_px, K, dist, T_W_C, z_m)

        if w is None:
            continue

        err_mm = abs(float(math.hypot(*(w[1] - w[0]))) - total_m * 1000.0)

        if best is None or err_mm < best[0]:
            best = (err_mm, z_m, name)

    return (planes.default_z_m, None) if best is None else (best[1], best[2])


def world_inputs(result: AiSegResult, *, K, dist, T_W_C, plane_z_m: float | BoltPlanes, view_bias=None,
                 width: int | None = None, height: int | None = None, mask_fallback: bool = False) -> list[WorldInput]:
    """AI 결과 + 촬영 시각 `T_W_C` → W 추적 입력(인스턴스 순서 그대로). 후보는 끝점 있는 축 경로 볼트다.

    `mask_fallback`이면 끝점 없는 볼트를 마스크 W 평균 입력(`kind="mask"`)으로 넣는다.
    `plane_z_m`이 `BoltPlanes`이면 끝점 볼트마다 종류별 높이를 고르고, 실수면 모든 볼트가 그 높이다.
    """
    if not isinstance(result, AiSegResult):
        raise TrackingContractError(f"world_inputs needs AiSegResult: {type(result).__name__}")

    planes = plane_z_m if isinstance(plane_z_m, BoltPlanes) else None
    base_z = planes.default_z_m if planes is not None else float(plane_z_m)

    out = []

    for inst in result.instances:
        base = dict(index=int(inst.index), class_name=str(inst.class_name), confidence=float(inst.confidence))

        if not inst.in_scope:
            out.append(WorldInput(**base, candidate=False, untracked_reason="not_in_scope", kind="none"))
            continue

        if inst.rejected in _GEOMETRY_REJECTIONS:
            out.append(WorldInput(**base, candidate=False, untracked_reason="geometry_rejected", kind="none"))
            continue

        if not _is_bolt(inst.class_name):
            out.append(WorldInput(**base, candidate=False, untracked_reason="not_bolt", kind="none"))
            continue

        m = mask_of(inst.polygon) if mask_fallback else None

        if inst.axis_endpoints_px is None:
            c = None if m is None else mask_center_w(m, K, dist, T_W_C, base_z)

            if c is not None:   # 마스크 W 평균(짝짓기 전용). 길이가 없어 기본 높이
                out.append(WorldInput(**base, candidate=True, untracked_reason=None, clipped=bool(inst.clipped), center_mm=c,
                                      center_raw_mm=c, kind="mask", mask=m, plane_z_m=base_z))
            else:
                out.append(WorldInput(**base, candidate=False, untracked_reason="no_endpoints", kind="none", mask=m))

            continue

        z_m, spec = (base_z, None) if planes is None else _pick_plane(inst.axis_endpoints_px, K, dist, T_W_C, planes)
        raw = _project(inst.axis_endpoints_px, K, dist, T_W_C, z_m)
        used = raw

        if view_bias is not None and raw is not None:
            from eyetoarm.vision.view_bias import correct_endpoints

            vb = correct_endpoints(inst.axis_endpoints_px, K, dist, T_W_C, plane_z_m=view_bias.plane_z_m,
                                   k_mm=view_bias.k_mm, width=width, height=height)

            if vb.applied:
                used = _project(vb.endpoints_px, K, dist, T_W_C, z_m)

        if used is None or raw is None:
            out.append(WorldInput(**base, candidate=False, untracked_reason="projection_failed", kind="none", mask=m))
            continue

        d = used[1] - used[0]
        out.append(WorldInput(**base, candidate=True, untracked_reason=None, clipped=bool(inst.clipped),
                              center_mm=(float(used[:, 0].mean()), float(used[:, 1].mean())),
                              center_raw_mm=(float(raw[:, 0].mean()), float(raw[:, 1].mean())),
                              endpoints_mm=((float(used[0, 0]), float(used[0, 1])), (float(used[1, 0]), float(used[1, 1]))),
                              length_mm=float(math.hypot(d[0], d[1])),
                              angle_deg=float(math.degrees(math.atan2(d[1], d[0])) % 180.0), kind="endpoint", mask=m,
                              plane_z_m=z_m, plane_spec=spec))

    return out


def _dist_summary(values: list[float]) -> dict:
    """거리 분포 요약(개수·중앙값·최대)."""
    if not values:
        return {"n": 0, "median": None, "max": None}

    return {"n": len(values), "median": round(float(statistics.median(values)), 3), "max": round(max(values), 3)}


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0

    return min(d, 180.0 - d)


def _same_bolt(a: WorldInput, b: WorldInput, limit_mm: float) -> bool:
    """두 끝점 쌍이 모두 limit 안(순서 무관)."""
    (a0, a1), (b0, b1) = a.endpoints_mm, b.endpoints_mm
    straight = math.dist(a0, b0) <= limit_mm and math.dist(a1, b1) <= limit_mm
    swapped = math.dist(a0, b1) <= limit_mm and math.dist(a1, b0) <= limit_mm

    return straight or swapped


class _WTrack:
    """트랙 하나(스칼라만)."""

    __slots__ = ("cls", "x", "y", "vy", "has_v", "length", "angle", "last_hit_ms", "seen", "clean", "part_track_id",
                 "gap_ms", "seen_ms", "last_camera", "grip_xy", "grip_ms", "hit_camera", "out_link", "out_grip",
                 "occluded", "vhist", "hold_pred")

    def __init__(self, c: WorldInput, t_ms: int) -> None:
        self.cls = c.class_name
        self.x, self.y = c.center_mm
        self.vy = 0.0
        self.has_v = False
        self.length, self.angle = c.length_mm, c.angle_deg
        self.last_hit_ms = t_ms
        self.seen = 1
        self.clean = 0 if c.clipped else 1
        self.part_track_id = None
        self.gap_ms = 0             # 화면 안인데 못 본 누적(가시성 만료일 때만 씀)
        self.seen_ms = t_ms         # 마지막으로 본 시각(위치 갱신 없는 「보임」 포함)
        self.last_camera = "link"
        self.grip_xy: tuple[float, float] | None = None   # 마지막 그리퍼 위치 갱신(W mm)과 그 시각
        self.grip_ms: int | None = None
        self.hit_camera = "link"    # 마지막 위치 갱신 카메라(속도는 링크 연속 hit로만)
        self.out_link = False       # 마지막 링크 프레임에서 화면 밖이었나
        self.out_grip = True        # 마지막 그리퍼 프레임에서 화면 밖이었나(평소 그리퍼는 다른 곳을 봄)
        self.occluded = False       # 마지막 링크 프레임에서 링크 화면 안이지만 팔·그리퍼에 가려졌나
        self.vhist: list[tuple[int, float]] = []   # 넘기기 켬: 링크 끝점 hit (시각, y) — 긴 기준선 속도용
        self.hold_pred = False      # 링크에서 놓친 뒤 다음 링크 위치 갱신까지 벨트 공통 속도로 예측


class WorldTracker:
    """W 추적 상태기(단일 소유: 추적 스레드). 결과 형식은 픽셀 추적과 같은 `TrackFrame`(`mode=WORLD`)."""

    def __init__(self, issuer: PartTrackIdIssuer, *, config: WorldTrackerConfig, visibility: VisibilityConfig | None = None,
                 gripper_gate_mm: tuple[float, float] | None = None, gripper_late_ms: int | None = None) -> None:
        """주입된 발급기(프로세스 하나)·설정. `visibility`·`gripper_gate_mm`(x, y)이 None이면 그 기능은 꺼짐.

        `gripper_late_ms`: 마지막 링크보다 이만큼(ms) 넘게 이른 그리퍼 관측은 버린다(None이면 이른 관측을 모두 버림).
        """
        if not isinstance(issuer, PartTrackIdIssuer):
            raise TrackingContractError("WorldTracker needs an injected PartTrackIdIssuer (one per process)")

        if not isinstance(config, WorldTrackerConfig):
            raise TrackingContractError("config must be WorldTrackerConfig")

        if visibility is not None and not isinstance(visibility, VisibilityConfig):
            raise TrackingContractError("visibility must be VisibilityConfig or None")

        if gripper_gate_mm is not None and (len(gripper_gate_mm) != 2 or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0 for v in gripper_gate_mm)):
            raise TrackingContractError(f"gripper_gate_mm must be two positive numbers: {gripper_gate_mm!r}")

        if gripper_late_ms is not None and (type(gripper_late_ms) is not int or gripper_late_ms < 0):
            raise TrackingContractError(f"gripper_late_ms must be an int >= 0: {gripper_late_ms!r}")

        self._issuer, self._config = issuer, config
        self._visibility = visibility
        self._gripper_late_ms = gripper_late_ms
        self.last_occluded: list = []          # 마지막 링크 프레임에서 가림으로 판정한 트랙 번호(기록용)
        self._hold_vy = 0.0                    # hold 예측용 벨트 공통 속도(mm/ms): 보이는 확정 트랙 중앙값, 없으면 직전 값
        self._gripper_gate = None if gripper_gate_mm is None else (float(gripper_gate_mm[0]), float(gripper_gate_mm[1]))
        self._cam_ts: dict[str, int] = {}     # 카메라별 직전 프레임 시각(공백 Δt 귀속)
        self._events: list[dict] = []          # 넘기기·만료 사건(가시성·그리퍼가 켜졌을 때만)
        self._handoff = {"gripper_updates": 0, "gripper_late": 0, "near_confirmed_skipped": 0, "tentative_dropped_for_confirmed": 0,
                         "gripper_not_updated": {}, "gripper_outcomes": {}, "retire_by_reason": {},
                         "camera_switches": 0}
        self._return_offsets: list[float] = []   # 링크 복귀 오프셋 거리(mm), 요약 분포용
        self._epoch = issuer.next_tracker_epoch()
        self._tracks: list[_WTrack] = []
        self._scope: tuple | None = None
        self._generation: int | None = None
        self._last_ts: int | None = None
        self.last_belt_vy: float | None = None   # mm/ms, 마지막 갱신의 벨트 공통 추정(없으면 None)
        self._stats = {"updates": 0, "not_updated": {}, "tentative_dropped": 0, "ids_retired": 0, "resets": {},
                       "events": {}, "dup_merged": 0, "dup_merged_with_mask": 0}

    @property
    def tracker_epoch(self) -> int:
        """현재 epoch."""
        return self._epoch

    @property
    def config(self) -> WorldTrackerConfig:
        """설정(읽기 전용)."""
        return self._config

    def stats(self) -> dict:
        """계수 스냅샷(복사)."""
        s = self._stats

        return {"updates": s["updates"], "not_updated": dict(s["not_updated"]), "tentative_dropped": s["tentative_dropped"],
                "ids_retired": s["ids_retired"], "resets": dict(s["resets"]), "events": dict(s["events"]),
                "dup_merged": s["dup_merged"], "dup_merged_with_mask": s["dup_merged_with_mask"], "tracks": len(self._tracks), "issued": self._issuer.issued,
                "tracker_epoch": self._epoch,
                "belt_vy_mm_per_s": None if self.last_belt_vy is None else round(self.last_belt_vy * 1000.0, 3),
                **({} if not self._extended else {"handoff": self._handoff_stats()})}

    @property
    def _extended(self) -> bool:
        """가시성 만료나 그리퍼 입력이 켜졌는가(꺼짐이면 기록·계수를 지금과 같게 둔다)."""
        return self._visibility is not None or self._gripper_gate is not None

    def _handoff_stats(self) -> dict:
        """넘기기 계수 복사."""
        h = self._handoff

        return {"gripper_updates": h["gripper_updates"], "gripper_late": h["gripper_late"],
                "near_confirmed_skipped": h["near_confirmed_skipped"], "tentative_dropped_for_confirmed": h["tentative_dropped_for_confirmed"],
                "gripper_not_updated": dict(h["gripper_not_updated"]),
                "gripper_outcomes": dict(h["gripper_outcomes"]), "retire_by_reason": dict(h["retire_by_reason"]),
                "camera_switches": h["camera_switches"], "link_return_offset_mm": _dist_summary(self._return_offsets)}

    def drain_events(self) -> list[dict]:
        """쌓인 넘기기·만료 사건을 꺼낸다(추적 스레드)."""
        events, self._events = self._events, []

        return events

    def predicted_confirmed(self, timestamp_ms: int) -> list[tuple[int, float, float]]:
        """확정 트랙의 예측 W 위치 [(번호, x mm, y mm)]."""
        belt_vy = self._belt_vy(None)

        return [(t.part_track_id, *self._predict(t, timestamp_ms, belt_vy, None)) for t in self._tracks
                if t.part_track_id is not None]

    def reset(self, reason: str) -> None:
        """전 트랙 종료·새 epoch·시각 기준 해제."""
        if not isinstance(reason, str) or not reason:
            raise TrackingContractError(f"reset reason must be a non-empty string: {reason!r}")

        self._terminate_all()
        self._scope = self._generation = self._last_ts = None
        self._cam_ts = {}
        self._stats["resets"][reason] = self._stats["resets"].get(reason, 0) + 1

    def update(self, key: FrameKey, timestamp_ms: int, inputs, *, belt_running: bool | None = None,
               visible=None, occluded=None) -> TrackFrame:
        """링크 프레임 하나를 반영한다. `visible(x_mm, y_mm)`은 가시성 만료일 때 그 화면 안인지. 입력 오류는 `TrackingInputError`."""
        inputs = self._validated(key, timestamp_ms, inputs)
        scope = (key.run_id, key.source_id, key.detector_config_id)
        event = None

        if self._scope is not None and scope == self._scope and key.generation < self._generation:
            return self._not_updated(key, timestamp_ms, inputs, "old_generation")

        if self._scope is not None and (scope != self._scope or key.generation > self._generation):
            self.reset("scope_changed")
            event = "scope_changed"

        if self._last_ts is not None and timestamp_ms <= self._last_ts:
            return self._not_updated(key, timestamp_ms, inputs, "stale_frame")

        self._scope, self._generation, self._last_ts = scope, key.generation, timestamp_ms
        self._stats["updates"] += 1
        cfg = self._config

        if self._visibility is None:
            for t in [t for t in self._tracks if timestamp_ms - t.last_hit_ms > cfg.max_gap_ms]:
                self._retire(t)
        else:
            self._expire(timestamp_ms)

        cands, merged = self._merge_duplicates([i for i in inputs if i.candidate])

        if self._visibility is not None:
            self._update_hold_vy()

        belt_vy = self._belt_vy(belt_running)
        self.last_belt_vy = belt_vy

        if self._jump_suspected(cands, timestamp_ms, belt_vy, belt_running):
            self._terminate_all()
            event = WORLD_JUMP
            self._stats["events"][WORLD_JUMP] = self._stats["events"].get(WORLD_JUMP, 0) + 1

        result = self._associate(cands, timestamp_ms, belt_vy, belt_running)

        if self._visibility is not None:
            self._count_gap("link", timestamp_ms, visible, belt_vy, belt_running, occluded)

        result.update({i: TrackAssignment(i, None, TrackState.UNTRACKED, 0, "duplicate_merged") for i in merged})
        out = [result[i.index] if i.index in result else TrackAssignment(i.index, None, TrackState.UNTRACKED, 0,
                                                                          i.untracked_reason) for i in inputs]

        return self._frame(key, timestamp_ms, True, event, out)

    # ------------------------------------------------------------------ 내부

    def _validated(self, key, timestamp_ms, inputs) -> list[WorldInput]:
        if not isinstance(key, FrameKey):
            raise TrackingInputError(f"key must be FrameKey: {type(key).__name__}")

        if type(timestamp_ms) is not int:
            raise TrackingInputError(f"timestamp_ms must be an int: {timestamp_ms!r}")

        inputs = list(inputs)
        seen = set()

        for i in inputs:
            if not isinstance(i, WorldInput):
                raise TrackingInputError(f"inputs must be WorldInput: {type(i).__name__}")

            if i.index in seen:
                raise TrackingInputError(f"duplicate input index {i.index}")

            seen.add(i.index)

            if i.candidate and i.kind == "mask":
                if not self._config.mask_fallback:
                    raise TrackingInputError(f"mask candidate {i.index} given but mask_fallback is off")

                if i.center_mm is None or not all(math.isfinite(v) for v in i.center_mm):
                    raise TrackingInputError(f"mask candidate {i.index} needs a finite W center")
            elif i.candidate and (i.center_mm is None or i.endpoints_mm is None or i.length_mm is None
                                  or not all(math.isfinite(v) for v in i.center_mm)):
                raise TrackingInputError(f"candidate {i.index} needs finite W center/endpoints/length")

        return inputs

    def _merge_duplicates(self, cands: list[WorldInput]) -> tuple[list[WorldInput], list[int]]:
        """이중 검출이면 하나만 남긴다(중심만 가까운 이웃 볼트는 합치지 않음). (남은 후보, 빠진 index)."""
        if not self._config.dup_merge:
            return cands, []

        # 남기는 순서: 끝점 입력 먼저, 다음 점수 높은 순
        order = sorted(cands, key=lambda c: (c.kind != "endpoint", -c.confidence, c.index))
        kept: list[WorldInput] = []
        dropped: list[int] = []

        for c in order:
            if any(k.class_name == c.class_name and self._dup(k, c) for k in kept):
                dropped.append(c.index)
                self._stats["dup_merged_with_mask"] += int(c.kind == "mask" or any(k.kind == "mask" for k in kept))
            else:
                kept.append(c)

        self._stats["dup_merged"] += len(dropped)

        return sorted(kept, key=lambda c: c.index), dropped

    def _dup(self, a: WorldInput, b: WorldInput) -> bool:
        """이중 검출 판정. 둘 다 끝점이면 끝점 쌍 규칙, 마스크가 끼면 중심 거리 또는 마스크 포함률."""
        if a.kind == "endpoint" and b.kind == "endpoint":
            return _same_bolt(a, b, self._config.dup_endpoint_mm)

        return (math.dist(a.center_mm, b.center_mm) <= self._config.dup_endpoint_mm
                or containment(a.mask, b.mask) >= self._config.mask_dup_contain)

    def _belt_vy(self, belt_running: bool | None) -> float:
        """벨트 공통 Y 속도(mm/ms): 확정·속도 있는 트랙 Y 속도 중앙값, 없으면 0. 정지 신호면 0."""
        if belt_running is False:
            return 0.0

        v = [t.vy for t in self._tracks if t.part_track_id is not None and t.has_v]

        return float(statistics.median(v)) if v else 0.0

    def _predict(self, t: _WTrack, t_ms: int, belt_vy: float, belt_running) -> tuple[float, float]:
        """예측 위치(mm). 속도를 모르면 벨트 공통 추정. 정지 신호면 0. `max_belt_speed_mps`는 쓰지 않는다."""
        return t.x, t.y + self._pred_vy(t, belt_vy, belt_running) * (t_ms - t.last_hit_ms)

    def _pred_vy(self, t: _WTrack, belt_vy: float, belt_running) -> float:
        """예측에 쓰는 Y 속도(mm/ms). hold(두 화면 밖)면 트랙 자체 속도 대신 벨트 공통 속도."""
        if belt_running is False:
            return 0.0

        if self._visibility is not None and (t.out_link or t.hold_pred):
            # 링크에 안 보임(화면 밖·가림), 또는 그 뒤 아직 링크 위치 갱신 전: 긴 기준선 벨트 공통 속도
            return self._hold_vy

        return t.vy if t.has_v else belt_vy

    def held_snapshot(self, timestamp_ms: int, belt_running=None) -> list[dict]:
        """링크에 안 보이는(hold·가림) 확정 트랙의 예측 위치·사용 속도·기준 시각(기록용, 켬일 때만 씀)."""
        belt_vy = self._belt_vy(belt_running)
        out = []

        for t in self._tracks:
            if t.part_track_id is None or not t.out_link:
                continue

            x, y = self._predict(t, timestamp_ms, belt_vy, belt_running)
            out.append({"part_track_id": t.part_track_id, "state": "occluded" if t.occluded else ("held" if self._held(t) else "out_link"),
                        "pred_mm": [round(x, 2), round(y, 2)],
                        "vy_mm_per_s": round(self._pred_vy(t, belt_vy, belt_running) * 1000.0, 3),
                        "own_vy_mm_per_s": round(t.vy * 1000.0, 3) if t.has_v else None,
                        "long_vy_mm_per_s": None if self._long_vy(t) is None else round(self._long_vy(t) * 1000.0, 3),
                        "ref_ms": t.last_hit_ms, "seen_ms": t.seen_ms, "gap_ms": t.gap_ms})

        return out

    def _held(self, t: _WTrack) -> bool:
        """가시성 만료가 켜져 있고 마지막 판정에서 링크·그리퍼 화면 모두 밖인 트랙."""
        return self._visibility is not None and t.out_link and t.out_grip

    def _update_hold_vy(self) -> None:
        """링크에 안 보이는 트랙 예측 속도: 보이는 확정 트랙의 긴 기준선 Y 속도 중앙값(없으면 직전 값 유지).

        링크 프레임 간격(25 ms) 차분 속도는 1 mm 잡음이 40 mm/s가 되어 긴 hold에서 수십 mm 어긋난다(r6 D10 실측).
        """
        v = [w for w in (self._long_vy(t) for t in self._tracks
                         if t.part_track_id is not None and not t.out_link and not t.hold_pred) if w is not None]

        if v:
            self._hold_vy = float(statistics.median(v))

    def _long_vy(self, t: _WTrack) -> float | None:
        """링크 끝점 hit 중 `velocity_window_ms` 이상 떨어진 가장 최근 쌍의 Y 속도(mm/ms). 기준선이 짧으면 None."""
        if not t.vhist:
            return None

        t1, y1 = t.vhist[-1]
        older = [(t0, y0) for t0, y0 in t.vhist if t1 - t0 >= self._visibility.velocity_window_ms]

        if not older:
            return None

        t0, y0 = older[-1]

        return (y1 - y0) / (t1 - t0)

    def _gate_y(self, t: _WTrack, t_ms: int) -> float:
        """Y 반경(mm): 속도를 모르는 트랙만 탐색 상한 × Δt를 더한다(m/s × ms = mm)."""
        extra = 0.0 if t.has_v else self._config.max_belt_speed_mps * (t_ms - t.last_hit_ms)

        return self._config.gate_y_mm + extra

    def _jump_suspected(self, cands, t_ms: int, belt_vy: float, belt_running) -> bool:
        """확정 트랙들이 예측 자리를 잃고 하나의 큰 공통 이동으로 설명되면 TF·보정 이상 의심.
        이웃 볼트가 더 가까울 수 있어 가장 가까운 후보 대응을 쓰지 않는다."""
        cfg = self._config
        confirmed = [t for t in self._tracks if t.part_track_id is not None]

        if len(confirmed) < cfg.jump_min_tracks:
            return False

        preds = [self._predict(t, t_ms, belt_vy, belt_running) for t in confirmed]
        same = [[c for c in cands if c.class_name == t.cls] for t in confirmed]

        for t, p, cs in zip(confirmed, preds, same):
            gy = self._gate_y(t, t_ms)

            if not cs or any(math.hypot((c.center_mm[0] - p[0]) / cfg.gate_x_mm, (c.center_mm[1] - p[1]) / gy) <= 1.0
                             for c in cs):
                return False   # 예측 자리에서 이어지는 트랙이 있으면 공통 점프가 아님

        hypotheses = [(c.center_mm[0] - p[0], c.center_mm[1] - p[1]) for p, cs in zip(preds, same) for c in cs]

        for sx, sy in hypotheses:
            if not cfg.coherent_jump_mm < math.hypot(sx, sy) <= cfg.jump_search_mm:
                continue

            if all(any(math.hypot(c.center_mm[0] - p[0] - sx, c.center_mm[1] - p[1] - sy) <= cfg.jump_coherence_mm for c in cs)
                   for p, cs in zip(preds, same)):
                return True

        return False

    def _associate(self, cands, t_ms: int, belt_vy: float, belt_running) -> dict[int, TrackAssignment]:
        cfg = self._config
        tracks = self._tracks
        preds = [self._predict(t, t_ms, belt_vy, belt_running) for t in tracks]
        elig: dict[int, list[tuple[float, int]]] = {c.index: [] for c in cands}
        by_track: dict[int, list[tuple[float, int]]] = {k: [] for k in range(len(tracks))}
        other_class_near: set[int] = set()

        for c in cands:
            cx, cy = c.center_mm

            for k, t in enumerate(tracks):
                gx, gy = cfg.gate_x_mm, self._gate_y(t, t_ms)
                dx, dy = (cx - preds[k][0]) / gx, (cy - preds[k][1]) / gy
                cost = math.hypot(dx, dy)

                if cost > 1.0:
                    continue

                if c.class_name != t.cls:
                    other_class_near.add(c.index)
                    continue

                # 길이·각도 검사는 둘 다 값이 있을 때만(마스크 입력에는 없음)
                if c.length_mm is not None and t.length is not None:
                    ratio = max(c.length_mm, t.length) / max(min(c.length_mm, t.length), 1e-9)

                    if ratio > cfg.length_ratio_max:
                        continue

                if c.angle_deg is not None and t.angle is not None and _angle_diff(c.angle_deg, t.angle) > cfg.angle_max_deg:
                    continue

                elig[c.index].append((cost, k))
                by_track[k].append((cost, c.index))

        if self._extended:
            self._drop_competing_tentatives(elig, by_track, tracks)

        ambiguous: set[int] = set()
        ended: set[int] = set()

        for ci, lst in elig.items():
            lst.sort()
            close = [k for cost, k in lst if cost - lst[0][0] < cfg.ambiguity_margin] if lst else []

            if len(close) >= 2:
                ambiguous.add(ci)
                ended.update(close)

        for k, lst in by_track.items():
            lst.sort()
            close = [ci for cost, ci in lst if cost - lst[0][0] < cfg.ambiguity_margin] if lst else []

            if len(close) >= 2:
                ambiguous.update(close)
                ended.add(k)

        pairs = sorted((cost, ci, k) for ci, lst in elig.items() if ci not in ambiguous for cost, k in lst if k not in ended)
        matched_c: dict[int, int] = {}
        matched_t: set[int] = set()

        for cost, ci, k in pairs:
            if ci in matched_c or k in matched_t:
                continue

            matched_c[ci] = k
            matched_t.add(k)

        by_index = {c.index: c for c in cands}
        out: dict[int, TrackAssignment] = {ci: self._hit(tracks[k], by_index[ci], t_ms) for ci, k in matched_c.items()}
        survivors = []

        for k, t in enumerate(tracks):
            if k in ended:
                self._retire_counted(t)
            elif k in matched_t or t.part_track_id is not None:
                survivors.append(t)
            else:
                self._stats["tentative_dropped"] += 1

        self._tracks = survivors

        for ci in ambiguous:
            out[ci] = TrackAssignment(ci, None, TrackState.AMBIGUOUS, 0, "ambiguous")

        for c in cands:
            if c.index in out:
                continue

            if c.confidence < cfg.new_track_min_conf:
                out[c.index] = TrackAssignment(c.index, None, TrackState.UNTRACKED, 0, "low_confidence")
                continue

            if self._extended and self._visibility is not None and (c.clipped or c.kind == "mask") and \
                    self._near_confirmed(c, t_ms, belt_vy, belt_running):
                # 넘기기 켬: 확정 트랙 근처의 잘린·마스크 후보는 새 트랙을 만들지 않는다(가림 경계 편향 후보)
                out[c.index] = TrackAssignment(c.index, None, TrackState.UNTRACKED, 0, "near_confirmed")
                self._handoff["near_confirmed_skipped"] += 1
                continue

            if len(self._tracks) >= cfg.max_tracks and not self._retire_oldest_lost(t_ms):
                out[c.index] = TrackAssignment(c.index, None, TrackState.UNTRACKED, 0, "track_capacity")
                continue

            t = _WTrack(c, t_ms)

            if cfg.min_hits == 1 and not c.clipped:
                t.part_track_id = self._issuer.next_id()

            self._tracks.append(t)
            reason = "class_changed" if c.index in other_class_near else ("clipped" if c.clipped else None)
            state = TrackState.CONFIRMED if t.part_track_id is not None else TrackState.TENTATIVE
            out[c.index] = TrackAssignment(c.index, t.part_track_id, state, t.clean, reason)

        return out

    def _drop_competing_tentatives(self, elig, by_track, tracks) -> None:
        """넘기기 켬: 한 후보를 확정 트랙 하나와 임시 트랙이 겨루면 임시 트랙을 짝짓기에서 빼서(이후 버려짐) 확정을 끝내지 않는다."""
        margin = self._config.ambiguity_margin
        drop: set[int] = set()

        for lst in elig.values():
            if len(lst) < 2:
                continue

            best = min(cost for cost, _ in lst)
            close = [k for cost, k in lst if cost - best < margin]
            confirmed = [k for k in close if tracks[k].part_track_id is not None]
            tentative = [k for k in close if tracks[k].part_track_id is None]

            if len(confirmed) == 1 and tentative:
                drop.update(tentative)

        if not drop:
            return

        for ci in elig:
            elig[ci] = [(cost, k) for cost, k in elig[ci] if k not in drop]

        for k in drop:
            by_track[k] = []

        self._handoff["tentative_dropped_for_confirmed"] += len(drop)

    def _near_confirmed(self, c: WorldInput, t_ms: int, belt_vy: float, belt_running) -> bool:
        """후보 중심이 확정 트랙 예측 위치에서 `clipped_near_mm` 안인가."""
        near = self._visibility.clipped_near_mm

        return any(math.dist(c.center_mm, self._predict(t, t_ms, belt_vy, belt_running)) <= near
                   for t in self._tracks if t.part_track_id is not None)

    def _hit(self, t: _WTrack, c: WorldInput, t_ms: int) -> TrackAssignment:
        cx, cy = c.center_mm
        dt = t_ms - t.last_hit_ms
        offset = self._return_offset(t, cx, cy, t_ms) if t.last_camera == "gripper" else None

        if self._extended and t.part_track_id is not None and (c.clipped or c.kind == "mask"):
            # 넘기기 켬: 확정 트랙의 잘린·마스크 입력은 「보임」만(위치·속도·길이·각도 불변). 가림 직전 끌림 방지
            self._seen(t, t_ms, "link", offset)
            t.seen += 1

            return TrackAssignment(c.index, t.part_track_id, TrackState.CONFIRMED, t.clean,
                                   "clipped" if c.clipped else "mask_input")

        if dt < 0:   # 더 늦은 그리퍼 갱신이 이미 있음: 위치·속도는 두고 본 것으로만
            self._seen(t, t_ms, "link")
            t.seen += 1
            state = TrackState.CONFIRMED if t.part_track_id is not None else TrackState.TENTATIVE

            return TrackAssignment(c.index, t.part_track_id, state, t.clean, "clipped" if c.clipped else None)

        if dt > 0 and t.hit_camera == "link":   # 속도는 링크 연속 hit로만(직전이 그리퍼 갱신이면 다음 링크 hit부터)
            vy = (cy - t.y) / dt
            t.vy = vy if not t.has_v else self._config.velocity_alpha * vy + (1 - self._config.velocity_alpha) * t.vy
            t.has_v = True

        t.x, t.y, t.last_hit_ms = cx, cy, t_ms
        t.hit_camera = "link"
        t.hold_pred = False

        if self._visibility is not None and c.kind == "endpoint" and not c.clipped:
            w = self._visibility.velocity_window_ms
            t.vhist.append((t_ms, cy))
            t.vhist = [(a, b) for a, b in t.vhist if t_ms - a <= 2 * w]
        self._seen(t, t_ms, "link", offset)

        if c.length_mm is not None:   # 마스크 hit는 길이·각도를 바꾸지 않는다
            t.length, t.angle = c.length_mm, c.angle_deg

        t.seen += 1
        t.clean = 0 if c.clipped else t.clean + 1

        if t.part_track_id is None and t.clean >= self._config.min_hits:
            t.part_track_id = self._issuer.next_id()

        state = TrackState.CONFIRMED if t.part_track_id is not None else TrackState.TENTATIVE
        reason = "clipped" if c.clipped else ("mask_input" if c.kind == "mask" else None)

        return TrackAssignment(c.index, t.part_track_id, state, t.clean, reason)

    def update_gripper(self, generation: int, timestamp_ms: int, inputs, *, visible=None) -> GripperUpdate:
        """그리퍼 깊이 프레임 하나로 확정 트랙의 위치만 갱신한다. 새 번호·클래스·속도는 바꾸지 않는다."""
        if self._gripper_gate is None:
            raise TrackingContractError("update_gripper needs gripper_gate_mm")

        if type(timestamp_ms) is not int or type(generation) is not int:
            raise TrackingInputError(f"generation/timestamp_ms must be ints: {generation!r}, {timestamp_ms!r}")

        inputs = list(inputs)

        for i in inputs:
            if not isinstance(i, GripperInput) or not all(math.isfinite(v) for v in (*i.center_mm, i.length_mm, i.angle_deg)):
                raise TrackingInputError(f"gripper inputs must be finite GripperInput: {i!r}")

        if self._scope is None:
            return self._gripper_not_updated(timestamp_ms, inputs, "no_link_scope")

        if generation != self._generation:
            return self._gripper_not_updated(timestamp_ms, inputs, "old_generation")

        late = self._last_ts is not None and timestamp_ms <= self._last_ts

        if late and (self._gripper_late_ms is None or self._last_ts - timestamp_ms > self._gripper_late_ms):
            return self._gripper_not_updated(timestamp_ms, inputs, "late_dropped")

        # 추적기 시각(_last_ts)은 링크만 앞당긴다: 그리퍼가 앞서도 뒤 링크 프레임이 stale이 되지 않는다
        self._handoff["gripper_updates"] += 1
        self._handoff["gripper_late"] += int(late)

        if not late:
            if self._visibility is None:
                for t in [t for t in self._tracks if timestamp_ms - t.last_hit_ms > self._config.max_gap_ms]:
                    self._retire(t)
            else:
                self._expire(timestamp_ms)

        if self._visibility is not None:
            self._update_hold_vy()

        belt_vy = self._belt_vy(None)
        out = self._associate_gripper(inputs, timestamp_ms, belt_vy)

        if self._visibility is not None:
            self._count_gap("gripper", timestamp_ms, visible, belt_vy, None)

        for a in out:
            o = self._handoff["gripper_outcomes"]
            o[a.outcome] = o.get(a.outcome, 0) + 1

        return GripperUpdate(timestamp_ms, True, None, tuple(out), late)

    def _associate_gripper(self, inputs, t_ms: int, belt_vy: float) -> list[GripperAssignment]:
        """확정 트랙과 짝짓기(게이트·길이·각도, 모호하면 둘 다 미갱신)."""
        cfg = self._config
        gx, gy = self._gripper_gate
        tracks = [t for t in self._tracks if t.part_track_id is not None]
        preds = [self._predict(t, t_ms, belt_vy, None) for t in tracks]
        elig: dict[int, list[tuple[float, int]]] = {c.index: [] for c in inputs}
        by_track: dict[int, list[tuple[float, int]]] = {k: [] for k in range(len(tracks))}
        mismatch: set[int] = set()

        for c in inputs:
            for k, t in enumerate(tracks):
                cost = math.hypot((c.center_mm[0] - preds[k][0]) / gx, (c.center_mm[1] - preds[k][1]) / gy)

                if cost > 1.0:
                    continue

                # 걸린 성분은 길이·각도가 잘려 있어 게이트만 본다
                if not c.clipped and t.length is not None:
                    ratio = max(c.length_mm, t.length) / max(min(c.length_mm, t.length), 1e-9)

                    if ratio > cfg.length_ratio_max or (t.angle is not None and
                                                        _angle_diff(c.angle_deg, t.angle) > cfg.angle_max_deg):
                        mismatch.add(c.index)
                        continue

                elig[c.index].append((cost, k))
                by_track[k].append((cost, c.index))

        ambiguous: set[int] = set()
        blocked: set[int] = set()

        for ci, lst in elig.items():
            lst.sort()
            close = [k for cost, k in lst if cost - lst[0][0] < cfg.ambiguity_margin] if lst else []

            if len(close) >= 2:
                ambiguous.add(ci)
                blocked.update(close)

        for k, lst in by_track.items():
            lst.sort()
            close = [ci for cost, ci in lst if cost - lst[0][0] < cfg.ambiguity_margin] if lst else []

            if len(close) >= 2:
                ambiguous.update(close)
                blocked.add(k)

        pairs = sorted((cost, ci, k) for ci, lst in elig.items() if ci not in ambiguous for cost, k in lst if k not in blocked)
        matched: dict[int, int] = {}
        used: set[int] = set()

        for cost, ci, k in pairs:
            if ci in matched or k in used:
                continue

            matched[ci] = k
            used.add(k)

        out = []

        for c in inputs:
            if c.index in matched:
                t = tracks[matched[c.index]]

                if t_ms <= t.seen_ms:   # 링크가 그 뒤에 이미 봄(늦은 관측): 바꾸지 않는다
                    out.append(GripperAssignment(c.index, t.part_track_id, "gripper_late_seen"))
                elif c.clipped:
                    self._seen(t, t_ms, "gripper")
                    out.append(GripperAssignment(c.index, t.part_track_id, "gripper_seen_clipped"))
                else:
                    t.x, t.y, t.last_hit_ms = c.center_mm[0], c.center_mm[1], t_ms
                    t.grip_xy, t.grip_ms = c.center_mm, t_ms
                    t.hit_camera = "gripper"
                    t.seen += 1
                    self._seen(t, t_ms, "gripper")
                    out.append(GripperAssignment(c.index, t.part_track_id, "gripper_hit"))
            elif c.index in ambiguous:
                out.append(GripperAssignment(c.index, None, "gripper_ambiguous"))
            elif c.index in mismatch:
                out.append(GripperAssignment(c.index, None, "gripper_length_mismatch"))
            else:
                out.append(GripperAssignment(c.index, None, "gripper_unmatched"))

        return out

    def _return_offset(self, t: _WTrack, cx: float, cy: float, t_ms: int) -> dict | None:
        """링크 복귀 오프셋: 링크 W − 마지막 그리퍼 W(트랙 속도로 복귀 시각까지 옮김). 그리퍼 위치 갱신이 없었으면 None."""
        if t.grip_xy is None:
            return None

        dt = t_ms - t.grip_ms
        gx, gy = t.grip_xy[0], t.grip_xy[1] + (t.vy * dt if t.has_v else 0.0)
        dx, dy = cx - gx, cy - gy

        return {"x": round(dx, 3), "y": round(dy, 3), "dist": round(math.hypot(dx, dy), 3), "dt_ms": dt}

    def _seen(self, t: _WTrack, t_ms: int, camera: str, offset: dict | None = None) -> None:
        """본 트랙: 공백·hold 초기화, 카메라가 바뀌면 사건(링크 복귀면 오프셋 포함)."""
        if self._extended and camera != t.last_camera and t.part_track_id is not None:
            self._handoff["camera_switches"] += 1
            event = {"kind": "camera_switch", "part_track_id": t.part_track_id, "from": t.last_camera,
                     "to": camera, "timestamp_ms": t_ms, "held_ms": t_ms - t.seen_ms}

            if camera == "link":
                event["link_return_offset_mm"] = offset

                if offset is not None:
                    self._return_offsets.append(offset["dist"])

            self._events.append(event)

        t.gap_ms, t.seen_ms, t.last_camera = 0, t_ms, camera

        t.occluded = False

        if camera == "link":
            t.out_link = False
        else:
            t.out_grip = False

    def _count_gap(self, camera: str, t_ms: int, visible, belt_vy: float, belt_running, occluded=None) -> None:
        """이 프레임에서 못 본 트랙 중 이 카메라 화면 안인 것에 그 카메라의 직전 프레임 이후 시간을 더한다."""
        prev = self._cam_ts.get(camera)
        self._cam_ts[camera] = t_ms

        if camera == "link":
            self.last_occluded = []

        for t in self._tracks:
            if t.seen_ms == t_ms:
                continue

            x, y = self._predict(t, t_ms, belt_vy, belt_running)
            out = visible is not None and not visible(x, y)

            if camera == "link":
                # 링크 화면 안이지만 팔·그리퍼에 가려짐: 링크에는 안 보이는 것으로 두고 공백을 세지 않는다
                t.occluded = not out and occluded is not None and occluded(x, y)
                out = out or t.occluded
                t.out_link = out
                t.hold_pred = t.hold_pred or out

                if t.occluded:
                    self.last_occluded.append(t.part_track_id)
            else:
                t.out_grip = out

            if out or prev is None:   # 화면 밖, 또는 이 카메라의 첫 프레임(귀속할 구간 없음)
                continue

            t.gap_ms += max(0, t_ms - max(prev, t.seen_ms))

    def note_camera(self, camera: str, t_ms: int) -> None:
        """화면 안 트랙이 없는 프레임: 그 카메라의 직전 프레임 시각만 남긴다(공백 귀속용)."""
        if self._visibility is not None and (self._cam_ts.get(camera) is None or t_ms > self._cam_ts[camera]):
            self._cam_ts[camera] = t_ms

    def _expire(self, t_ms: int) -> None:
        """가시성 만료: 화면 안 공백 > max_gap_ms 또는 안 보인 지 > hold_limit_ms."""
        for t in list(self._tracks):
            unseen = t_ms - t.seen_ms
            reason = ("in_view_missing" if t.gap_ms > self._config.max_gap_ms else
                      "occluded_limit" if t.occluded and unseen > self._visibility.occluded_limit_ms else
                      "hold_limit" if unseen > self._visibility.hold_limit_ms else None)

            if reason is None:
                continue

            self._retire(t)
            r = self._handoff["retire_by_reason"]
            r[reason] = r.get(reason, 0) + 1

            if t.part_track_id is not None:
                self._events.append({"kind": "retire", "part_track_id": t.part_track_id, "reason": reason, "timestamp_ms": t_ms,
                                     "last_camera": t.last_camera, "unseen_ms": t_ms - t.seen_ms, "gap_ms": t.gap_ms})

    def _gripper_not_updated(self, timestamp_ms: int, inputs, reason: str) -> GripperUpdate:
        n = self._handoff["gripper_not_updated"]
        n[reason] = n.get(reason, 0) + 1

        return GripperUpdate(timestamp_ms, False, reason, tuple(GripperAssignment(i.index, None, reason) for i in inputs))

    def _retire_oldest_lost(self, t_ms: int) -> bool:
        lost = [t for t in self._tracks if t.last_hit_ms != t_ms]

        if not lost:
            return False

        self._retire(min(lost, key=lambda t: t.last_hit_ms))

        return True

    def _retire(self, t: _WTrack) -> None:
        self._tracks.remove(t)
        self._retire_counted(t)

    def _retire_counted(self, t: _WTrack) -> None:
        if t.part_track_id is not None:
            self._stats["ids_retired"] += 1
        else:
            self._stats["tentative_dropped"] += 1

    def _terminate_all(self) -> bool:
        if not self._tracks:
            return False

        for t in self._tracks:
            self._retire_counted(t)

        self._tracks = []
        self._epoch = self._issuer.next_tracker_epoch()

        return True

    def _not_updated(self, key, timestamp_ms, inputs, reason: str) -> TrackFrame:
        n = self._stats["not_updated"]
        n[reason] = n.get(reason, 0) + 1

        return self._frame(key, timestamp_ms, False, None,
                           [TrackAssignment(i.index, None, TrackState.UNTRACKED, 0, reason) for i in inputs])

    def _frame(self, key: FrameKey, timestamp_ms: int, updated: bool, event, out) -> TrackFrame:
        return TrackFrame(run_id=key.run_id, source_id=key.source_id, image_seq=key.image_seq,
                          capture_generation=key.capture_generation, generation=key.generation, job_seq=key.job_seq,
                          detector_config_id=key.detector_config_id, timestamp_ms=timestamp_ms, mode=TrackMode.WORLD,
                          track_namespace=self._issuer.namespace, tracker_epoch=self._epoch, updated=updated, event=event,
                          assignments=tuple(out))


__all__ = ["WORLD_JUMP", "BoltPlanes", "GripperAssignment", "GripperInput", "GripperUpdate", "Occluder", "VisibilityConfig",
           "WorldInput",
           "WorldTracker", "WorldTrackerConfig", "WorldTrackerConfigError", "containment", "mask_center_w", "mask_of",
           "parse_visibility", "parse_world_tracker", "world_inputs"]
