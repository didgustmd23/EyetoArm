"""작업 좌표(W) 기준 객체 추적(#217 1단계): 볼트 끝점을 촬영 시각 TF로 벨트 평면(W, mm)에 올려 짝짓는다.

근거: 계획 `_workspace/217_builder2_world_tracking_plan_r0.md`·`_r1.md`(reviewer PASS, 사용자 승인 2026-11-20; Q2 보정 뒤 끝점, Q3 이중 검출
3 mm, Q4 `max_belt_speed_mps`는 짝짓기 탐색 상한만, Q6 `TrackMode.WORLD`).

**보장 범위.** 픽셀 추적(#151)과 같은 표시·실험 번호다(A-6). 분열·스왑을 없앤다는 보장이 아니다. W 위치는 B와 같은 투영
(`transforms.planar_transform.pixels_to_w`, 볼트 평면 높이)이므로 TF·보정이 틀리면 W 위치도 틀린다.

- **입력(`world_inputs`)**: AI 인스턴스 중 축 경로 볼트(`axis_evaluated`)이고 끝점이 있는 것만 후보다. 너트·비대상·점수 미달·기하
  거부는 `untracked`(`not_bolt`·`not_in_scope`·`geometry_rejected`), 끝점 없음·추출기 무효는 `no_endpoints`. 잘린 볼트는 후보이되
  `clipped`(트랙 유지에만 쓰고 연속 hit를 끊음). 겹침 당사자(`MULTIPLE_CANDIDATES`)도 끝점이 있으면 후보로 넣고 이중 검출 합치기가
  정리한다. W 위치 = 두 끝점을 각각 W로 올린 뒤의 중점(mm, B 중심과 같음). 시야 보정(#207)을 주면 보정 뒤 끝점으로 올리고 보정 전
  위치도 함께 남긴다. W 길이·축 각도(180° 주기)도 만든다.
- **짝짓기**: 예측 = 위치 + Y 속도 × Δt(X 속도 0, 벨트는 ±Y 한 축). Y 속도는 트랙별 차분 지수 평균이고, 속도를 아직 모르는 트랙은
  확정 트랙 Y 속도 중앙값(벨트 공통 추정, 없으면 0)으로 예측한다. 벨트 정지 신호(`belt_running=False`)면 예측 속도 0.
  반경은 X `gate_x_mm`, Y `gate_y_mm`이고, 속도를 모르는 트랙만 Y 반경에 `max_belt_speed_mps × Δt`를 더한다(**탐색 상한이며 예측
  위치에는 쓰지 않음**). 길이 비·축 각도 일관 검사, 모호(1·2위 비용 차 < 여유)면 번호 None·관련 트랙 종료, 탐욕 1:1 배정.
- **이중 검출 합치기**: 같은 클래스이고 두 끝점 쌍이 모두 `dup_endpoint_mm` 안(순서 무관)이면 점수 높은 하나만 남긴다
  (나란히 붙은 두 볼트처럼 중심만 가까운 경우는 합치지 않음).
- **확정·만료**: 잘리지 않은 연속 hit `min_hits`면 번호 발급(발급기 공유, 재사용 없음). 마지막 hit 뒤 `max_gap_ms`면 은퇴.
- **카메라 이동 의심은 두지 않는다**(W에서 팔 이동은 위치를 바꾸지 않는다). 대신 확정 트랙 `jump_min_tracks`개 이상이 같은 방향으로
  `coherent_jump_mm` 넘게 함께 튀면 TF·보정 이상 의심(`world_jump_suspected`)으로 전 트랙 종료.
- 시각·범위·세대 규칙은 픽셀 추적과 같다(같은·이른 시각은 `stale_frame`, 이전 세대는 `old_generation`, 범위·세대 변경은 전 종료).
- **마스크 입력(#226 W+M, 설정 `mask_fallback`)**: 켜면 끝점이 없는 볼트(`no_endpoints`)도 다각형(3점 이상)을 채운 마스크 픽셀 전부를
  촬영 시각 TF로 볼트 평면에 올린 평균을 W 위치로 넣는다(`kind="mask"`, 시야 보정 없음, 길이·각도 없음). 짝짓기에서 길이 비·축 각도 검사는
  어느 쪽이든 값이 없으면 건너뛰고, 마스크 hit는 트랙의 길이·각도를 바꾸지 않는다. 이중 검출 합치기는 마스크 입력이 낀 쌍에 한해 「W 중심
  거리 ≤ `dup_endpoint_mm` 또는 마스크 포함률(교집합/작은 면적) ≥ `mask_dup_contain`」을 쓰고, 남기는 순서는 끝점 입력 먼저·다음 점수 순이다
  (#224 W+M 규칙 그대로). 마스크 입력은 **짝짓기에만** 쓴다: B 입력·번호 규칙(#215 확정·선택 인스턴스)은 바뀌지 않는다. 끄면 #217과 같다.
- **종류별 투영 높이(#250 A-3 P1)**: `plane_z_m`에 `BoltPlanes`(B 규격의 볼트 종류별 기준 높이·총길이)를 주면, 끝점 볼트는 종류마다
  그 높이로 두 끝점을 올려 W 길이가 그 종류 총길이에 가장 가까운 종류의 높이를 쓴다(종류는 이 고르기에만 쓰고 B 판정이 아니다).
  종류 높이가 모두 같으면(지금 B 규격) 그 높이 하나라 결과가 실수 `plane_z_m`과 같다. 마스크 입력은 길이가 없어 기본 높이(B `class_plane_z`
  「bolt」)를 쓴다. 시야 보정은 자기 설정 높이(`view_bias.plane_z_m`)로 보정량을 정하고, 보정 뒤 끝점은 고른 높이로 올린다.
- 단일 소유(추적 스레드 하나). ROS·Qt를 import하지 않는다. 설정값은 모두 A 개발 임시값(시험 설정 파일, 기본값 없음).
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field, fields

import cv2
import numpy as np

from eyetoarm.transforms.planar_transform import pixels_to_w
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
    mask_fallback: bool         # #226 W+M: 끝점이 없는 볼트를 마스크 W 평균으로 짝짓기(끄면 #217)
    mask_dup_contain: float     # #226 W+M: 마스크가 낀 이중 검출 합치기 포함률 기준(#224 값 0.85)

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
    kind: str = "endpoint"                                   # endpoint | mask | none(#226)
    mask: tuple | None = field(default=None, compare=False)  # (x0, y0, bool 배열) 원본 픽셀, 이중 검출 포함률 판정용
    plane_z_m: float | None = None                           # #250: W로 올린 높이(m). 후보가 아니면 None
    plane_spec: str | None = None                            # #250: 높이를 고른 B 규격 이름(BoltPlanes 끝점 입력만, 그 밖 None)


@dataclass(frozen=True)
class BoltPlanes:
    """B 규격의 볼트 종류별 투영 높이(#250 A-3 P1). `specs` = ((이름, 기준 높이 m, 총길이 m), …), `default_z_m` = 마스크 입력용.

    B 규격 파일을 고치지 않고 읽은 값을 그대로 담는다(진입점이 만든다). 값은 유한하고 높이·길이는 양수여야 한다.
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
    """다각형(원본 픽셀, 3점 이상) → (x0, y0, bool 배열). 정수 반올림 뒤 채움(#224 규칙). 없으면 None."""
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
    """끝점 볼트의 투영 높이(#250 P1): 종류마다 그 높이로 올린 W 길이와 그 종류 총길이의 차가 가장 작은 종류의 높이.

    높이가 모두 같으면 기본 높이(이름 None). 어느 종류로도 투영이 안 되면 기본 높이(이름 None, 뒤의 투영이 실패를 낸다).
    차가 같으면 `specs` 순서가 앞선 종류다.
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
    """AI 결과 + 촬영 시각 `T_W_C` → W 추적 입력(인스턴스 순서 그대로). `view_bias`는 `ViewBiasConfig`(#207) 또는 None.

    `mask_fallback`(#226)이면 볼트 입력에 마스크(포함률용)를 붙이고, 끝점 없는 볼트를 마스크 W 평균 입력(`kind="mask"`)으로 넣는다.
    `plane_z_m`이 `BoltPlanes`이면 끝점 볼트마다 종류별 높이를 고른다(#250 P1, 모듈 설명). 실수면 모든 볼트가 그 높이다(기존과 같음).
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

            if c is not None:   # #226 W+M: 끝점 없는 볼트 → 마스크 W 평균(짝짓기 전용). 길이가 없어 기본 높이
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

    __slots__ = ("cls", "x", "y", "vy", "has_v", "length", "angle", "last_hit_ms", "seen", "clean", "part_track_id")

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


class WorldTracker:
    """W 추적 상태기(단일 소유: 추적 스레드). 결과 형식은 픽셀 추적과 같은 `TrackFrame`(`mode=WORLD`)."""

    def __init__(self, issuer: PartTrackIdIssuer, *, config: WorldTrackerConfig) -> None:
        """주입된 발급기(프로세스 하나)·설정."""
        if not isinstance(issuer, PartTrackIdIssuer):
            raise TrackingContractError("WorldTracker needs an injected PartTrackIdIssuer (one per process)")

        if not isinstance(config, WorldTrackerConfig):
            raise TrackingContractError("config must be WorldTrackerConfig")

        self._issuer, self._config = issuer, config
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
                "belt_vy_mm_per_s": None if self.last_belt_vy is None else round(self.last_belt_vy * 1000.0, 3)}

    def reset(self, reason: str) -> None:
        """전 트랙 종료·새 epoch·시각 기준 해제."""
        if not isinstance(reason, str) or not reason:
            raise TrackingContractError(f"reset reason must be a non-empty string: {reason!r}")

        self._terminate_all()
        self._scope = self._generation = self._last_ts = None
        self._stats["resets"][reason] = self._stats["resets"].get(reason, 0) + 1

    def update(self, key: FrameKey, timestamp_ms: int, inputs, *, belt_running: bool | None = None) -> TrackFrame:
        """프레임 하나를 반영한다. 입력 오류는 `TrackingInputError`(상태 무변)."""
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

        for t in [t for t in self._tracks if timestamp_ms - t.last_hit_ms > cfg.max_gap_ms]:
            self._retire(t)

        cands, merged = self._merge_duplicates([i for i in inputs if i.candidate])
        belt_vy = self._belt_vy(belt_running)
        self.last_belt_vy = belt_vy

        if self._jump_suspected(cands, timestamp_ms, belt_vy, belt_running):
            self._terminate_all()
            event = WORLD_JUMP
            self._stats["events"][WORLD_JUMP] = self._stats["events"].get(WORLD_JUMP, 0) + 1

        result = self._associate(cands, timestamp_ms, belt_vy, belt_running)
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
        """같은 클래스·끝점 쌍 dup_endpoint_mm 안이면 점수 높은 하나만 남긴다. (남은 후보, 합쳐져 빠진 index)."""
        if not self._config.dup_merge:
            return cands, []

        # 남기는 순서: 끝점 입력 먼저(#226), 다음 점수 높은 순. 끝점 입력만 있으면 #217과 같다
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
        """이중 검출 판정. 둘 다 끝점이면 끝점 쌍 규칙(#217), 마스크가 끼면 중심 거리 또는 마스크 포함률(#224)."""
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
        dt = t_ms - t.last_hit_ms
        vy = 0.0 if belt_running is False else (t.vy if t.has_v else belt_vy)

        return t.x, t.y + vy * dt

    def _gate_y(self, t: _WTrack, t_ms: int) -> float:
        """Y 반경(mm): 속도를 모르는 트랙만 탐색 상한 × Δt를 더한다(m/s × ms = mm)."""
        extra = 0.0 if t.has_v else self._config.max_belt_speed_mps * (t_ms - t.last_hit_ms)

        return self._config.gate_y_mm + extra

    def _jump_suspected(self, cands, t_ms: int, belt_vy: float, belt_running) -> bool:
        """확정 트랙 `jump_min_tracks`개 이상이 모두 예측 자리(반경 안)에 같은 클래스 후보가 없고, 크기 `coherent_jump_mm` 넘는 공통
        이동 하나로 모든 트랙의 후보를 `jump_coherence_mm` 안에서 설명할 수 있으면 TF·보정 이상 의심. 가장 가까운 후보 대응을 쓰지 않는다
        (이웃 볼트가 더 가까울 수 있음)."""
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

                # 길이·각도 검사는 둘 다 값이 있을 때만(마스크 입력·마스크로 시작한 트랙은 없음, #226)
                if c.length_mm is not None and t.length is not None:
                    ratio = max(c.length_mm, t.length) / max(min(c.length_mm, t.length), 1e-9)

                    if ratio > cfg.length_ratio_max:
                        continue

                if c.angle_deg is not None and t.angle is not None and _angle_diff(c.angle_deg, t.angle) > cfg.angle_max_deg:
                    continue

                elig[c.index].append((cost, k))
                by_track[k].append((cost, c.index))

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

    def _hit(self, t: _WTrack, c: WorldInput, t_ms: int) -> TrackAssignment:
        cx, cy = c.center_mm
        dt = t_ms - t.last_hit_ms

        if dt > 0:
            vy = (cy - t.y) / dt
            t.vy = vy if not t.has_v else self._config.velocity_alpha * vy + (1 - self._config.velocity_alpha) * t.vy
            t.has_v = True

        t.x, t.y, t.last_hit_ms = cx, cy, t_ms

        if c.length_mm is not None:   # 마스크 hit는 길이·각도를 바꾸지 않는다(#226)
            t.length, t.angle = c.length_mm, c.angle_deg

        t.seen += 1
        t.clean = 0 if c.clipped else t.clean + 1

        if t.part_track_id is None and t.clean >= self._config.min_hits:
            t.part_track_id = self._issuer.next_id()

        state = TrackState.CONFIRMED if t.part_track_id is not None else TrackState.TENTATIVE
        reason = "clipped" if c.clipped else ("mask_input" if c.kind == "mask" else None)

        return TrackAssignment(c.index, t.part_track_id, state, t.clean, reason)

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


__all__ = ["WORLD_JUMP", "BoltPlanes", "WorldInput", "WorldTracker", "WorldTrackerConfig", "WorldTrackerConfigError", "containment",
           "mask_center_w", "mask_of", "parse_world_tracker", "world_inputs"]
