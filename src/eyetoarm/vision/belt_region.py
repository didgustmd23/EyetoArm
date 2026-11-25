"""벨트 밖 검출 제외: 바닥·상자 등 벨트 윗면 밖에 있는 물체의 검출을 `OUTSIDE_BELT`로 거부한다.

마스크 내부 표본 픽셀의 광선을 벨트 윗면 높이(top_z)에서 자르고, 그 교점이 벨트 사각형(+여유) 안에 드는 비율을 잰다.
보이는 바닥 점의 광선은 벨트 윗면을 지나지 않으므로(지나면 벨트에 가려 보이지 않는다) 교점이 사각형 밖이다. 이 판정은
「영상에 투영한 벨트 윗면 다각형 안」 판정과 같다(같은 평면 사영의 앞·뒤 방향). 판정 평면이 볼트 평면이 아니라 벨트 윗면
높이여야 이 성질이 선다. 벨트 위 공중에 들린 볼트는 교점이 사각형 안이라 통과한다(팔 영역 필터 몫).
ROS·Qt를 import하지 않는다.

사진 하나의 판정은 모든 후보의 표본을 모아 교점 계산을 한 번 한다(#434, 값은 부품별 `inside_fraction`과 같음).
안쪽 감싸개가 팔 영역 필터이고 같은 사진을 방금 판정했으면, 그 카메라 위치 조회 결과와 (표본 상한이 같을 때) 표본을
다시 쓴다. 다시 쓰지 못하면 지금처럼 직접 조회·표본 추출한다.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from eyetoarm.vision.ai_result import OUTSIDE_BELT, AiSegResult
from eyetoarm.vision.arm_region import ArmRegionFilter, judge_candidates, reject_instances, sample_points


class BeltRegionConfigError(ValueError):
    """벨트 영역 제외 설정 오류."""


@dataclass(frozen=True)
class BeltRegionConfig:
    """판정 설정(이름에 단위). 사각형·높이는 frames.yaml `belt`, 나머지는 설정 파일."""

    rect_xy_m: tuple[float, float, float, float]   # W 축 정렬 벨트 윗면 (x_min, x_max, y_min, y_max)
    top_z_m: float              # 벨트 윗면 높이(판정 평면)
    margin_mm: float            # 사각형을 사방으로 이만큼 넓혀 판정
    min_inside_fraction: float  # 안에 든 표본 비율이 이보다 작으면 제외
    max_samples: int            # 마스크 내부 픽셀 표본 상한(고르게 솎음)
    world_frame: str

    def to_json(self) -> dict:
        """요약용 사전."""
        return {"rect_xy_m": list(self.rect_xy_m), "top_z_m": self.top_z_m, "margin_mm": self.margin_mm,
                "min_inside_fraction": self.min_inside_fraction, "max_samples": self.max_samples,
                "world_frame": self.world_frame}


_KEYS = {"status", "margin_mm", "min_inside_fraction", "max_samples", "world_frame"}


def belt_from_frames(frames_doc) -> tuple[tuple[float, float, float, float], float]:
    """frames.yaml 사전 → (벨트 사각형 (x_min, x_max, y_min, y_max) m, top_z m). 오류는 `BeltRegionConfigError`."""
    from eyetoarm.inputs.ros_gripper_depth import belt_roi_xy_m   # 그 모듈은 맨 위에서 rclpy를 import하지 않는다

    try:
        rect = tuple(float(v) for v in belt_roi_xy_m(frames_doc))
        top_z = frames_doc["belt"]["top_z"]
    except (KeyError, TypeError, ValueError) as exc:
        raise BeltRegionConfigError(f"frames belt block invalid: {type(exc).__name__}: {exc}") from exc

    if isinstance(top_z, bool) or not isinstance(top_z, (int, float)) or not math.isfinite(top_z):
        raise BeltRegionConfigError(f"frames belt.top_z invalid: {top_z!r}")

    if not all(math.isfinite(v) for v in rect) or not (rect[0] < rect[1] and rect[2] < rect[3]):
        raise BeltRegionConfigError(f"frames belt rectangle invalid: {rect!r}")

    return rect, float(top_z)


def parse_belt_region(doc, frames_doc) -> BeltRegionConfig:
    """설정 사전(`status`는 반드시 `test_only`, `note` 선택) + frames.yaml 사전 → 설정."""
    if not isinstance(doc, dict) or set(doc) - {"note"} != _KEYS:
        raise BeltRegionConfigError(f"belt region config must have exactly {sorted(_KEYS)} (+ optional note)")

    if doc["status"] != "test_only":
        raise BeltRegionConfigError("belt region config status must be 'test_only' (development values)")

    def num(name, lo, hi=None, positive=False):
        """범위 검사한 실수."""
        v = doc[name]

        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < lo \
                or (positive and v <= 0) or (hi is not None and v > hi):
            raise BeltRegionConfigError(f"{name} out of range: {v!r}")

        return float(v)

    n = doc["max_samples"]

    if isinstance(n, bool) or not isinstance(n, int) or n < 3:
        raise BeltRegionConfigError(f"max_samples must be an int >= 3: {n!r}")

    if not isinstance(doc["world_frame"], str) or not doc["world_frame"]:
        raise BeltRegionConfigError("world_frame must be a non-empty string")

    rect, top_z = belt_from_frames(frames_doc)

    return BeltRegionConfig(rect, top_z, num("margin_mm", 0.0), num("min_inside_fraction", 0.0, 1.0, positive=True),
                            n, doc["world_frame"])


def load_belt_region_config(path: str, frames_path: str) -> tuple[BeltRegionConfig, str, str]:
    """설정 파일 + frames.yaml(읽기만) → (설정, 설정 sha256, frames sha256)."""
    import yaml

    raw = Path(path).read_bytes()
    frames_raw = Path(frames_path).read_bytes()

    try:
        doc = json.loads(raw)
    except ValueError as exc:
        raise BeltRegionConfigError(f"belt region config is not JSON: {exc}") from exc

    try:
        frames_doc = yaml.safe_load(frames_raw)
    except yaml.YAMLError as exc:
        raise BeltRegionConfigError(f"frames file is not YAML: {exc}") from exc

    return (parse_belt_region(doc, frames_doc), hashlib.sha256(raw).hexdigest(),
            hashlib.sha256(frames_raw).hexdigest())


def rays_to_plane(px, K, dist, T_W_C, plane_z_m: float) -> np.ndarray:
    """원본 픽셀 (N, 2) → 광선과 z = plane_z_m 평면의 교점 W (N, 2) m. 카메라 앞에서 만나지 않으면 NaN."""
    pts = np.asarray(px, dtype=np.float64).reshape(-1, 1, 2)
    norm = cv2.undistortPoints(pts, np.asarray(K, dtype=np.float64), np.asarray(dist, dtype=np.float64)).reshape(-1, 2)
    T = np.asarray(T_W_C, dtype=np.float64)
    d_w = np.column_stack([norm, np.ones(len(norm))]) @ T[:3, :3].T   # 광학 프레임 광선(Z 전방) → W 방향
    c = T[:3, 3]

    with np.errstate(divide="ignore", invalid="ignore"):
        t = (plane_z_m - c[2]) / d_w[:, 2]

    ok = np.isfinite(t) & (t > 0)
    out = np.full((len(norm), 2), np.nan)
    out[ok] = c[:2] + d_w[ok, :2] * t[ok, None]

    return out


def inside_fraction(polygon, K, dist, T_W_C, config: BeltRegionConfig) -> tuple[float, int, list | None]:
    """표본 픽셀 중 top_z 교점이 넓힌 사각형 안인 비율, 표본 수, 교점 중심 W [mm](교점이 없으면 None).

    평면과 만나지 않는 표본은 밖으로 센다.
    """
    pts = sample_points(polygon, config.max_samples)
    w = rays_to_plane(pts, K, dist, T_W_C, config.top_z_m)
    m = config.margin_mm / 1000.0
    x0, x1, y0, y1 = config.rect_xy_m

    with np.errstate(invalid="ignore"):
        inside = (w[:, 0] >= x0 - m) & (w[:, 0] <= x1 + m) & (w[:, 1] >= y0 - m) & (w[:, 1] <= y1 + m)

    finite = w[np.isfinite(w).all(axis=1)] * 1000.0
    center = None if len(finite) == 0 else [round(float(np.mean(finite[:, 0])), 1),
                                            round(float(np.mean(finite[:, 1])), 1)]

    return int(np.count_nonzero(inside)) / len(pts), len(pts), center


def inside_fractions(samples: list[np.ndarray], K, dist, T_W_C,
                     config: BeltRegionConfig) -> list[tuple[float, int, list | None]]:
    """부품별 표본 목록 → 부품마다 `inside_fraction`과 같은 (안 비율, 표본 수, 교점 중심 W [mm] | None).

    모든 표본을 이어 붙여 교점(`rays_to_plane`)과 사각형 안 판정을 한 번씩 한 뒤 부품별로 나눈다(점마다 독립이라 값이 같다).
    """
    if not samples:
        return []

    sizes = [len(pts) for pts in samples]

    # 교점·사각형 안: 모든 표본을 한 번에(평면과 만나지 않는 NaN 표본은 비교가 거짓이라 밖으로 센다)
    w_all = rays_to_plane(np.concatenate(samples), K, dist, T_W_C, config.top_z_m)
    m = config.margin_mm / 1000.0
    x0, x1, y0, y1 = config.rect_xy_m

    with np.errstate(invalid="ignore"):
        inside_all = (w_all[:, 0] >= x0 - m) & (w_all[:, 0] <= x1 + m) & (w_all[:, 1] >= y0 - m) & (w_all[:, 1] <= y1 + m)

    ok_all = np.isfinite(w_all).all(axis=1)

    # 부품별로 나눔: 안 비율과 교점 중심(부품별 계산처럼 m → mm 뒤 평균)
    out, start = [], 0

    for n in sizes:
        part = slice(start, start + n)
        start += n
        finite = w_all[part][ok_all[part]] * 1000.0
        center = None if len(finite) == 0 else [round(float(np.mean(finite[:, 0])), 1),
                                                round(float(np.mean(finite[:, 1])), 1)]
        out.append((int(np.count_nonzero(inside_all[part])) / n, n, center))

    return out


def judge_belt(result: AiSegResult, *, K, dist, T_W_C, config: BeltRegionConfig,
               reuse: dict | None = None) -> tuple[AiSegResult, list[dict]]:
    """`filter_result`와 같다. `reuse`({인스턴스 번호: (다각형 객체, 표본)}, 같은 상한으로 뽑은 것)에 같은 다각형 객체가
    있으면 그 표본을 쓰고, 없으면 새로 뽑는다."""
    cands = judge_candidates(result)
    samples = []

    for inst in cands:
        got = None if reuse is None else reuse.get(inst.index)
        samples.append(got[1] if got is not None and got[0] is inst.polygon
                       else sample_points(inst.polygon, config.max_samples))

    fractions = inside_fractions(samples, K, dist, T_W_C, config)
    decisions, removed = [], set()

    for inst, (frac, n, center) in zip(cands, fractions):
        excluded = frac < config.min_inside_fraction
        decisions.append({"index": inst.index, "class_name": inst.class_name, "confidence": round(inst.confidence, 4),
                          "samples": n, "inside_fraction": round(frac, 4), "excluded": excluded,
                          "center_w_top_mm": center})

        if excluded:
            removed.add(inst.index)

    return reject_instances(result, removed, OUTSIDE_BELT), decisions


def filter_result(result: AiSegResult, *, K, dist, T_W_C, config: BeltRegionConfig) -> tuple[AiSegResult, list[dict]]:
    """벨트 밖 인스턴스를 `OUTSIDE_BELT`로 거부한 결과와 인스턴스별 판정 근거(이미 범위 밖인 것은 판정하지 않음)."""
    return judge_belt(result, K=K, dist=dist, T_W_C=T_W_C, config=config)


class BeltRegionFilter(ArmRegionFilter):
    """검출기 감싸개(worker 스레드에서 호출). 벨트 밖 인스턴스를 거부하고 관측을 다시 고른다.

    감싸기·재선택·계수·요약은 `ArmRegionFilter`와 같고, 판정만 다르다(카메라 TF 하나만 조회). TF·카메라가 없거나 판정
    중 오류가 나면 제외하지 않고 사유만 남긴다. 계수는 잠금 안.

    안쪽 감싸개가 같은 스레드에서 같은 사진을 판정했으면(`region_share`) 그 카메라 위치 조회 결과를 다시 쓰고(조회 시간은
    그만큼 짧게 기록됨), 표본 상한이 같으면 표본도 다시 쓴다(#434).
    """

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

        # 안쪽 팔 영역 필터가 같은 사진에서 얻은 카메라 위치·표본(없으면 None: 직접 조회)
        share_of = getattr(self._inner, "region_share", None)
        share = share_of(frame) if callable(share_of) else None
        t_tf = self._clock()
        tf_cam = share.tf_cam if share is not None else self._tf(frame)

        if tf_cam.status != "ok":
            self._count("tf_unavailable")
            return result, {**base, "status": f"tf_{tf_cam.status}:camera"}

        self._timed(t_tf, self._tf_ms)
        reuse = share.samples if share is not None and share.max_samples == self.config.max_samples else None
        out, decisions = judge_belt(result, K=camera.K, dist=camera.dist, T_W_C=tf_cam.tf.T_W_C,
                                    config=self.config, reuse=reuse)
        self._count("judged")
        self._count("excluded", sum(d["excluded"] for d in decisions))

        return out, {**base, "status": "ok", "decisions": decisions}


__all__ = ["BeltRegionConfig", "BeltRegionConfigError", "BeltRegionFilter", "belt_from_frames", "filter_result",
           "inside_fraction", "inside_fractions", "judge_belt", "load_belt_region_config", "parse_belt_region",
           "rays_to_plane"]
