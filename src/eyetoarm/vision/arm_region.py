"""팔·그리퍼 투영 영역 검출 제외: 링크 카메라에 걸린 팔·그리퍼를 AI가 볼트로 검출한 인스턴스를 `ARM_REGION`으로 거부한다.

마스크 내부 표본 픽셀마다 「카메라 → 그 픽셀의 볼트 평면 교점」 선분이 팔 캡슐(촬영 stamp TF)을 지나면 그 픽셀에
보이는 것은 팔이다. 그런 표본 비율이 문턱 이상이면 제외하고 관측을 다시 고른다. ROS·Qt를 import하지 않는다.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import cv2
import numpy as np

from eyetoarm.transforms.planar_transform import pixels_to_w
from eyetoarm.vision.ai_result import ARM_REGION, AiSegResult
from eyetoarm.vision.arm_occlusion import (Occluder, OccluderConfigError, parse_occluders,
                                           world_capsules)


class ArmRegionConfigError(ValueError):
    """팔 영역 제외 설정 오류."""


@dataclass(frozen=True)
class ArmRegionConfig:
    """판정 설정(이름에 단위). 모두 필수."""

    occluders: tuple[Occluder, ...]
    margin_mm: float            # 캡슐을 이만큼 부풀려 판정
    min_fraction: float         # 팔에 가린 표본 비율이 이 이상이면 제외
    max_samples: int            # 마스크 내부 픽셀 표본 상한(고르게 솎음)
    plane_z_m: float            # 픽셀 광선을 자를 볼트 평면 높이
    world_frame: str

    def to_json(self) -> dict:
        """요약용 사전."""
        return {"occluders": [o.to_json() for o in self.occluders], "margin_mm": self.margin_mm,
                "min_fraction": self.min_fraction, "max_samples": self.max_samples, "plane_z_m": self.plane_z_m,
                "world_frame": self.world_frame}


_KEYS = {"status", "occluders", "margin_mm", "min_fraction", "max_samples", "plane_z_m", "world_frame"}


def parse_arm_region(doc) -> ArmRegionConfig:
    """설정 사전 → 설정(`status`는 반드시 `test_only`, `note` 선택). 오류는 `ArmRegionConfigError`."""
    if not isinstance(doc, dict) or set(doc) - {"note"} != _KEYS:
        raise ArmRegionConfigError(f"arm region config must have exactly {sorted(_KEYS)} (+ optional note)")

    if doc["status"] != "test_only":
        raise ArmRegionConfigError("arm region config status must be 'test_only' (development values)")

    def num(name, lo, hi=None, positive=False):
        """범위 검사한 실수."""
        v = doc[name]

        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < lo \
                or (positive and v <= 0) or (hi is not None and v > hi):
            raise ArmRegionConfigError(f"{name} out of range: {v!r}")

        return float(v)

    n = doc["max_samples"]

    if isinstance(n, bool) or not isinstance(n, int) or n < 3:
        raise ArmRegionConfigError(f"max_samples must be an int >= 3: {n!r}")

    if not isinstance(doc["world_frame"], str) or not doc["world_frame"]:
        raise ArmRegionConfigError("world_frame must be a non-empty string")

    try:
        occluders = parse_occluders(doc["occluders"])
    except OccluderConfigError as exc:
        raise ArmRegionConfigError(str(exc)) from exc

    return ArmRegionConfig(occluders, num("margin_mm", 0.0), num("min_fraction", 0.0, 1.0, positive=True), n,
                           num("plane_z_m", 0.0, positive=True), doc["world_frame"])


def load_arm_region_config(path: str) -> tuple[ArmRegionConfig, str]:
    """설정 파일 → (설정, sha256)."""
    raw = Path(path).read_bytes()

    try:
        doc = json.loads(raw)
    except ValueError as exc:
        raise ArmRegionConfigError(f"arm region config is not JSON: {exc}") from exc

    return parse_arm_region(doc), hashlib.sha256(raw).hexdigest()


def sample_points(polygon: np.ndarray, max_samples: int) -> np.ndarray:
    """다각형 내부 픽셀을 고르게 솎은 표본 (N ≤ max_samples, 2) 픽셀(면적 비율을 잰다). 내부 픽셀이 없으면 꼭짓점."""
    poly = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    x0, y0 = np.floor(poly.min(axis=0)).astype(int)
    x1, y1 = np.ceil(poly.max(axis=0)).astype(int)
    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(poly - [x0, y0]).astype(np.int32)], 1)
    ys, xs = np.nonzero(mask)

    if len(xs) == 0:
        return poly[:max_samples]

    pts = np.stack([xs + x0, ys + y0], axis=1).astype(np.float64)
    idx = np.linspace(0, len(pts) - 1, num=min(max_samples, len(pts))).round().astype(int)

    return pts[idx]


def segment_distances(c: np.ndarray, P: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """선분 c–P[i]와 A[j]–B[j] 사이 최소 거리 (N, M). `arm_occlusion.segment_distance`와 같은 분기를 배열로 푼다."""
    eps = 1e-15
    d1 = P - c                                   # (N, 3)
    d2 = B - A                                   # (M, 3)
    r = c - A                                    # (M, 3)

    a = np.einsum("ij,ij->i", d1, d1)[:, None]   # (N, 1)
    e = np.einsum("ij,ij->i", d2, d2)[None, :]   # (1, M)
    f = np.einsum("ij,ij->i", d2, r)[None, :]    # (1, M)
    cc = d1 @ r.T                                # (N, M)
    b = d1 @ d2.T                                # (N, M)

    a_zero, e_zero = a <= eps, e <= eps

    with np.errstate(divide="ignore", invalid="ignore"):
        s_c = np.clip(-cc / a, 0.0, 1.0)         # t=0일 때 s
        denom = a * e - b * b
        s = np.where(denom > eps, np.clip((b * f - cc * e) / denom, 0.0, 1.0), 0.0)
        t = (b * s + f) / e
        lo, hi = t < 0.0, t > 1.0
        s = np.where(lo, s_c, np.where(hi, np.clip((b - cc) / a, 0.0, 1.0), s))
        t = np.where(lo, 0.0, np.where(hi, 1.0, t))

        # 퇴화 선분: P가 c와 같으면 s=0·t=f/e, 캡슐 축 길이 0이면 t=0·s=s_c
        t = np.where(a_zero, np.clip(f / e, 0.0, 1.0), np.where(e_zero, 0.0, t))
        s = np.where(a_zero, 0.0, np.where(e_zero, s_c, s))
        both = a_zero & e_zero

    s, t = np.nan_to_num(s), np.nan_to_num(t)
    gap = (c + d1[:, None, :] * s[..., None]) - (A[None, :, :] + d2[None, :, :] * t[..., None])
    dist = np.sqrt(np.einsum("nmk,nmk->nm", gap, gap))

    return np.where(both, np.sqrt(np.einsum("ij,ij->i", r, r))[None, :], dist)


def arm_fraction(polygon, K, dist, T_W_C, capsules, config: ArmRegionConfig) -> tuple[float, int, list | None]:
    """표본 픽셀 중 팔 캡슐에 가린 비율, 표본 수, 표본의 볼트 평면 위 중심 W [mm](평면과 만나는 표본이 없으면 None).

    평면과 만나지 않는 표본은 가린 것으로 세지 않는다. 표본×캡슐 거리를 한 번에 계산한다.
    """
    pts = sample_points(polygon, config.max_samples)
    w = pixels_to_w(pts, K, dist, T_W_C, config.plane_z_m) * 1000.0
    finite = w[np.isfinite(w).all(axis=1)]

    if len(finite) == 0:
        return 0.0, len(pts), None

    hits = 0

    if capsules:
        A = np.array([cap[0] for cap in capsules], dtype=np.float64).reshape(-1, 3)
        B = np.array([cap[1] for cap in capsules], dtype=np.float64).reshape(-1, 3)
        radius = np.array([cap[2] for cap in capsules], dtype=np.float64) + config.margin_mm / 1000.0
        P = np.column_stack([finite / 1000.0, np.full(len(finite), config.plane_z_m)])
        d = segment_distances(np.asarray(T_W_C, dtype=np.float64)[:3, 3], P, A, B)
        hits = int(np.count_nonzero((d < radius[None, :]).any(axis=1)))

    center = [round(float(np.mean(finite[:, 0])), 1), round(float(np.mean(finite[:, 1])), 1)]

    return hits / len(pts), len(pts), center


def filter_result(result: AiSegResult, *, K, dist, T_W_C, T_W_L_by_link: dict,
                  config: ArmRegionConfig) -> tuple[AiSegResult, list[dict]]:
    """팔 영역 인스턴스를 `ARM_REGION`으로 거부한 결과와 인스턴스별 판정 근거."""
    capsules = world_capsules(config.occluders, T_W_L_by_link)
    decisions, removed = [], set()

    for inst in result.instances:
        if not inst.in_scope or inst.polygon is None:
            continue

        frac, n, center = arm_fraction(inst.polygon, K, dist, T_W_C, capsules, config)
        excluded = frac >= config.min_fraction
        decisions.append({"index": inst.index, "class_name": inst.class_name, "confidence": round(inst.confidence, 4),
                          "samples": n, "arm_fraction": round(frac, 4), "excluded": excluded,
                          "center_w_mm": center})

        if excluded:
            removed.add(inst.index)

    return reject_instances(result, removed, ARM_REGION), decisions


def reject_instances(result: AiSegResult, removed: set, code: str) -> AiSegResult:
    """`removed` 번호의 인스턴스를 범위 밖 사유 `code`로 거부한 결과(없으면 그대로). 팔·벨트 영역 필터가 같이 쓴다."""
    if not removed:
        return result

    instances = []

    for inst in result.instances:
        if inst.index in removed:
            # 범위 밖 거부: 축 경로 필드를 비운다(축 후보·추적·목록에서 빠짐)
            instances.append(dataclasses.replace(inst, rejected=code, axis_endpoints_px=None,
                                                 axis_reason=None, axis_max_iou=None, axis_diag=None,
                                                 axis_overlap_with=None))
        elif inst.axis_overlap_with and removed & set(inst.axis_overlap_with):
            # 겹침 짝이 제외됐으면 짝 목록에서 뺀다(남은 짝이 없으면 겹침 아님)
            rest = tuple(i for i in inst.axis_overlap_with if i not in removed)
            instances.append(dataclasses.replace(inst, axis_overlap_with=rest or None))
        else:
            instances.append(inst)

    return dataclasses.replace(result, instances=tuple(instances))


class ArmRegionFilter:
    """검출기 감싸개(worker 스레드에서 호출). 결과의 팔 영역 인스턴스를 거부하고 관측을 다시 고른다.

    `tf_lookup(frame_like)`는 촬영 stamp 정확 조회(`LiveCaptureTfLookup`), `camera_for(frame)`은 그 프레임의
    `CameraProfile`(없으면 None)을 돌려준다. TF·카메라가 없거나 판정 중 오류가 나면 제외하지 않고 사유만 남긴다
    (필터가 진짜 볼트 검출을 막지 않게). 계수는 잠금 안.
    """

    def __init__(self, detector, *, tf_lookup: Callable, camera_for: Callable, config: ArmRegionConfig,
                 select: Callable, on_decision: Callable | None = None,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        """감쌀 검출기·TF 조회·카메라 조회·설정·관측 재선택 함수(`observation_from_ai`)·판정 기록 콜백."""
        self._inner = detector
        self._tf = tf_lookup
        self._camera_for = camera_for
        self.config = config
        self._select = select
        self._on_decision = on_decision
        self._clock = clock
        self._lock = threading.Lock()
        self.counts = {"frames": 0, "judged": 0, "excluded": 0, "reselected": 0, "no_camera": 0, "tf_unavailable": 0,
                       "errors": 0}
        self._ms: list[float] = []
        self._tf_ms: list[float] = []   # 그중 TF 조회(카메라 + 링크) 시간

    def __getattr__(self, name):
        """감싼 검출기의 다른 속성(prepare·config 등)은 그대로 넘긴다."""
        return getattr(self._inner, name)

    def __call__(self, job, *args):
        """감싼 검출기를 부르고(토큰 인자 그대로) 결과를 거른다."""
        value = self._inner(job, *args)

        if not (isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], AiSegResult)):
            return value

        obs, result = value
        t0 = self._clock()

        try:
            out, record = self._filter(job, result)
        except Exception as exc:   # 판정 오류는 제외 없이 사유만 남긴다(진짜 볼트 검출을 막지 않음)
            self._count("errors")
            out, record = result, {"image_seq": job.frame.image_seq, "status": f"error:{type(exc).__name__}: {exc}"}

        if out is not result:
            obs = self._select(job, out)
            self._count("reselected")

        self._timed(t0)

        if self._on_decision is not None and record is not None:
            self._on_decision(record)

        return obs, out

    def _filter(self, job, result: AiSegResult):
        """(결과, 판정 기록 | None)."""
        frame = job.frame
        self._count("frames")
        cands = [i for i in result.instances if i.in_scope and i.polygon is not None]

        if not cands:
            return result, None

        camera = self._camera_for(frame)
        base = {"image_seq": frame.image_seq, "ros_stamp_ns": getattr(frame, "ros_stamp_ns", None)}

        if camera is None:
            self._count("no_camera")
            return result, {**base, "status": "no_camera"}

        t_tf = self._clock()
        tf_cam = self._tf(frame)

        if tf_cam.status != "ok":
            self._count("tf_unavailable")
            return result, {**base, "status": f"tf_{tf_cam.status}:camera"}

        poses = {}

        for link in dict.fromkeys(o.link for o in self.config.occluders):
            res = self._tf(SimpleNamespace(ros_stamp_ns=getattr(frame, "ros_stamp_ns", None), optical_frame=link))

            if res.status != "ok":
                self._count("tf_unavailable")
                return result, {**base, "status": f"tf_{res.status}:{link}"}

            poses[link] = res.tf.T_W_C

        self._timed(t_tf, self._tf_ms)
        out, decisions = filter_result(result, K=camera.K, dist=camera.dist, T_W_C=tf_cam.tf.T_W_C,
                                       T_W_L_by_link=poses, config=self.config)
        self._count("judged")
        self._count("excluded", sum(d["excluded"] for d in decisions))

        return out, {**base, "status": "ok", "decisions": decisions}

    def _count(self, key: str, n: int = 1) -> None:
        """계수 증가."""
        with self._lock:
            self.counts[key] += n

    def _timed(self, t0: float, into: list[float] | None = None) -> None:
        """처리 시간(ms) 하나(기본 전체 처리 목록, 최근 4096개 보관)."""
        into = self._ms if into is None else into

        with self._lock:
            into.append((self._clock() - t0) * 1000.0)
            del into[:-4096]

    def summary(self) -> dict:
        """계수·처리 ms 요약."""
        with self._lock:
            ms, tf_ms, counts = sorted(self._ms), sorted(self._tf_ms), dict(self.counts)

        def pct(values, q):
            """분위수(ms)."""
            return None if not values else round(values[min(len(values) - 1, int(q * (len(values) - 1)))], 3)

        return {"config": self.config.to_json(), "counts": counts, "ms_median": pct(ms, 0.5), "ms_p95": pct(ms, 0.95),
                "ms_max": None if not ms else round(ms[-1], 3), "tf_ms_median": pct(tf_ms, 0.5),
                "tf_ms_p95": pct(tf_ms, 0.95)}


__all__ = ["ArmRegionConfig", "ArmRegionConfigError", "ArmRegionFilter", "arm_fraction", "filter_result",
           "load_arm_region_config", "parse_arm_region", "reject_instances", "sample_points", "segment_distances"]
