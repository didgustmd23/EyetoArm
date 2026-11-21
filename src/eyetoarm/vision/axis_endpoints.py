"""볼트 마스크 하나에서 축선 위 두 끝점 픽셀(행 0 머리 끝, 행 1 축 끝)을 구하는 순수 추출기(M0·M1·M2).

NumPy·OpenCV만 쓰며 정답·카메라 자세·world 기하·크기(mm)를 쓰지 않는다. 두 볼트가 같은 치수 비의 축소형이라
끝점은 머리 지름 px와 비만으로 정해지고, 종류·길이(mm)는 끝점을 작업면에 투영하는 B가 정한다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AxisModel:
    """볼트 모델 치수의 비에서 나온 고정값(무차원)과 각도 deg, 화소 수. 길이(mm)는 쓰지 않는다.

    치수 근거: `bolt_m6x40` 머리 ⌀10 × 6, 축 ⌀6 × 40, 축 끝 면취 안쪽 r 2. `bolt_m3x20`은 그 0.5배.
    """

    tip_to_head_radius: float = 0.5     # 축 끝 돌출 반지름 / 머리 반지름 = 2.5 / 5
    band_frac: float = 0.12             # 규칙 K와 같음(머리 길이 비 6/46 = 3/23 ≈ 0.13보다 작게)
    ambiguity_ratio: float = 1.2        # 머리 지름 / 축 지름 = 10/6 = 5/3 ≈ 1.67보다 충분히 작게
    min_pixels: int = 20
    max_tilt_deg: float = 60.0          # 이보다 서면 누운 자세(축 높이 평면) 가정이 깨진다고 본다
    max_head_step_frac: float = 0.7     # 머리 계단 위치 상한(머리 길이 비 ≈ 0.13에 여유 있게)
    vote_band_frac: float = 0.25        # M2 머리 투표 띠. M1은 쓰지 않는다
    vote_min_ratio: float = 1.1         # M2 투표 띠 폭 비 하한. M1은 쓰지 않는다


MODEL_DERIVED = AxisModel()


@dataclass(frozen=True)
class AxisEstimate:
    """추출 결과. `endpoints_px`는 유효일 때만 float64 (2, 2) 읽기 전용(행 0 머리 끝, 행 1 축 끝), 무효면 None."""

    method: str
    valid: bool
    endpoints_px: np.ndarray | None
    reason: str | None
    diag: dict


def _check_mask(mask) -> np.ndarray:
    """입력 경계: 2차원 bool 배열만 받는다(그 밖은 ValueError)."""
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("mask must be a 2-D numpy bool array")

    return mask


def _invalid(method: str, reason: str, diag: dict) -> AxisEstimate:
    """무효 결과(끝점 None)."""
    return AxisEstimate(method, False, None, reason, diag)


def _frozen(points) -> np.ndarray:
    """float64 (2, 2) 읽기 전용 복사본."""
    out = np.array(points, dtype=np.float64).reshape(2, 2)
    out.setflags(write=False)
    return out


def _axis_frame(mask: np.ndarray, model: AxisModel, method: str):
    """공통 단계: 화소 수·테두리·주축·양 끝 폭. (frame 사전, None) 또는 (None, 무효 결과)."""
    ys, xs = np.nonzero(mask)
    count = int(xs.size)
    diag = {"pixels": count}

    if count < model.min_pixels:
        return None, _invalid(method, "NO_OBJECT", diag)

    if mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any():
        return None, _invalid(method, "CONTOUR_CLIPPED", diag)

    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    center = pts.mean(axis=0)
    rel = pts - center
    evals, evecs = np.linalg.eigh(rel.T @ rel / count)

    if np.isclose(evals[1], evals[0], rtol=1e-9, atol=1e-12):
        return None, _invalid(method, "AXIS_DEGENERATE", diag)

    e1 = evecs[:, 1]

    if e1[0] < 0 or (e1[0] == 0 and e1[1] < 0):
        e1 = -e1

    e2 = np.array([-e1[1], e1[0]])
    s, t = rel @ e1, rel @ e2
    s_min, s_max = float(s.min()), float(s.max())
    band = model.band_frac * (s_max - s_min)
    lo, hi = s <= s_min + band, s >= s_max - band
    width_lo = float(t[lo].max() - t[lo].min() + 1.0)
    width_hi = float(t[hi].max() - t[hi].min() + 1.0)
    ratio = max(width_lo, width_hi) / min(width_lo, width_hi)
    diag["width_ratio"] = ratio

    if ratio < model.ambiguity_ratio:
        return None, _invalid(method, "AXIS_HEAD_AMBIGUOUS", diag)

    head_at_min = width_lo > width_hi
    return {"center": center, "e1": e1, "e2": e2, "s": s, "t": t, "s_min": s_min, "s_max": s_max,
            "head_at_min": head_at_min, "diag": diag}, None


def extract_m0(mask, model: AxisModel = MODEL_DERIVED) -> AxisEstimate:
    """M0: 주축 위 화소 중심 극값. 진단 기준선 전용(축선 위 기준점이 아님)."""
    mask = _check_mask(mask)
    f, bad = _axis_frame(mask, model, "M0")

    if bad is not None:
        return bad

    lo_pt = f["center"] + f["s_min"] * f["e1"]
    hi_pt = f["center"] + f["s_max"] * f["e1"]
    head, tip = (lo_pt, hi_pt) if f["head_at_min"] else (hi_pt, lo_pt)
    return AxisEstimate("M0", True, _frozen([head, tip]), None, f["diag"])


def extract_m1(mask, model: AxisModel = MODEL_DERIVED) -> AxisEstimate:
    """M1: 축선 위 머리 끝·축 끝 기준점 근사(모델 치수 비로 끝면 테두리 돌출을 안쪽 보정)."""
    mask = _check_mask(mask)
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    pre = {"pixels": int(xs.size)}

    if xs.size >= model.min_pixels and not (mask[0, :].any() or mask[-1, :].any()
                                            or mask[:, 0].any() or mask[:, -1].any()):
        m8 = mask.astype(np.uint8)
        n_labels, _ = cv2.connectedComponents(m8, connectivity=8)
        contours, hierarchy = cv2.findContours(m8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        holes = 0 if hierarchy is None else int((hierarchy[0][:, 3] != -1).sum())
        pre.update(components=int(n_labels - 1), holes=holes)

        if n_labels - 1 > 1:
            return _invalid("M1", "MASK_FRAGMENTED", pre)

        if holes > 0:
            return _invalid("M1", "MASK_HOLE", pre)

    f, bad = _axis_frame(mask, model, "M1")

    if bad is not None:
        return AxisEstimate("M1", False, None, bad.reason, {**pre, **bad.diag})

    diag = {**pre, **f["diag"]}

    # 1. 머리 쪽을 0으로 하는 축 좌표와 1 px 단면
    s, t = f["s"], f["t"]
    s_head = (s - f["s_min"]) if f["head_at_min"] else (f["s_max"] - s)
    span = f["s_max"] - f["s_min"]
    n_bins = int(math.floor(span)) + 1
    b = np.minimum(np.floor(s_head).astype(int), n_bins - 1)
    tmin = np.full(n_bins, np.inf)
    tmax = np.full(n_bins, -np.inf)
    np.minimum.at(tmin, b, t)
    np.maximum.at(tmax, b, t)
    filled = np.isfinite(tmin)
    width = np.where(filled, tmax - tmin + 1.0, np.nan)
    mid = np.where(filled, (tmax + tmin) / 2.0, np.nan)
    pos = np.arange(n_bins) + 0.5

    head_bins = filled & (pos <= model.band_frac * span + 0.5)
    shaft_far = filled & (pos >= 0.5 * span)

    if not head_bins.any() or not shaft_far.any():
        return _invalid("M1", "AXIS_ORDER_INCONSISTENT", diag)

    head_w = float(np.median(width[head_bins]))
    shaft_w = float(np.median(width[shaft_far]))
    diag.update(head_width_px=head_w, shaft_width_px=shaft_w, length_px=span + 1.0)

    # 2. 머리 계단: 첫 넓은 단면부터 끊김 없이 이어진 넓은 구간의 끝 + 1(빈 단면도 끊김).
    #    축 끝 쪽의 고립된 넓은 단면(1 px 양자화)이 계단을 축 끝까지 밀지 않게 한다
    thresh = (head_w + shaft_w) / 2.0
    wide = filled & (width >= thresh)
    wide_idx = np.nonzero(wide)[0]

    if wide_idx.size == 0:
        step = 0
    else:
        gaps = np.nonzero(np.diff(wide_idx) != 1)[0]
        run_end = int(wide_idx[gaps[0]]) if gaps.size else int(wide_idx[-1])
        step = run_end + 1

    step_frac = step / n_bins
    diag["head_step_frac"] = step_frac

    if shaft_w >= head_w or step == 0 or step_frac > model.max_head_step_frac:
        return _invalid("M1", "AXIS_ORDER_INCONSISTENT", diag)

    # 3. 머리 지름 px: 계단 앞 단면 폭의 최댓값(직교 투영에서 끝면 타원 장축은 기울기와 무관)
    head_region = filled & (np.arange(n_bins) < step)
    head_d_px = float(np.max(width[head_region]))
    r_head_px = head_d_px / 2.0

    # 4. 기울기: 끝면 반타원의 축 방향 반지름 c = 머리 지름에 닿기 전 폭 부족 면적 / (b·(2 − π/2)),
    #    sinα = c / 머리 반지름 px. 실루엣 길이만으로는 α = 0과 큰 α를 구분하지 못해서다
    reach = np.nonzero(head_region & (width >= head_d_px - 1.0))[0]
    first_full = int(reach[0]) if reach.size else 0
    front = filled & (np.arange(n_bins) < first_full)
    deficit = float(np.clip(head_d_px - width[front], 0.0, None).sum())
    cap_px = deficit / (r_head_px * (2.0 - math.pi / 2.0))
    sin_raw = cap_px / r_head_px
    clamped = sin_raw >= 1.0
    sin_a = 1.0 if clamped else sin_raw
    alpha = math.asin(sin_a)
    diag.update(head_diameter_px=head_d_px, head_cap_px=cap_px, tilt_deg=math.degrees(alpha), tilt_clamped=clamped)

    if math.degrees(alpha) > model.max_tilt_deg:
        return _invalid("M1", "AXIS_VIEW_OBLIQUE", diag)

    # 5. 중심선과 안쪽 보정
    shaft_fit = filled & (np.arange(n_bins) > step) & (np.arange(n_bins) < n_bins - 1)

    if shaft_fit.sum() >= 3:
        slope, intercept = np.polyfit(pos[shaft_fit], mid[shaft_fit], 1)
    else:
        slope, intercept = 0.0, 0.0

    d_head = r_head_px * sin_a
    d_tip = model.tip_to_head_radius * r_head_px * sin_a
    diag.update(centerline_slope=float(slope), centerline_offset_px=float(intercept),
                protrusion_head_px=d_head, protrusion_tip_px=d_tip)

    def point(sh: float) -> np.ndarray:
        """머리 기준 축 좌표 sh의 중심선 위 화소 좌표."""
        s_abs = (f["s_min"] + sh) if f["head_at_min"] else (f["s_max"] - sh)
        return f["center"] + s_abs * f["e1"] + (intercept + slope * sh) * f["e2"]

    head = point(0.0 + d_head)
    tip = point(span - d_tip)
    pts = np.array([head, tip])

    if (not np.isfinite(pts).all() or float(np.hypot(*(head - tip))) < 1.0
            or (pts[:, 0] < 0).any() or (pts[:, 0] > w - 1).any() or (pts[:, 1] < 0).any() or (pts[:, 1] > h - 1).any()):
        return _invalid("M1", "ENDPOINTS_INVALID", diag)

    return AxisEstimate("M1", True, _frozen(pts), None, diag)


def _band_widths(s: np.ndarray, t: np.ndarray, s_min: float, s_max: float, frac: float) -> tuple[float, float]:
    """주축 양 끝 frac 띠(길이 비)의 단면 폭(lo 쪽, hi 쪽). 폭 = t 범위 + 1 px."""
    band = frac * (s_max - s_min)
    lo, hi = s <= s_min + band, s >= s_max - band

    return float(t[lo].max() - t[lo].min() + 1.0), float(t[hi].max() - t[hi].min() + 1.0)


def _principal_frame(mask: np.ndarray, model: AxisModel, method: str):
    """M2 공통 단계: 화소 수·테두리·주축(머리 판정 없음). (frame 사전, None) 또는 (None, 무효 결과).

    M1 `_axis_frame` 앞부분과 같지만 띠 모호 관문이 없다. M1을 바꾸지 않으려고 따로 둔다.
    """
    ys, xs = np.nonzero(mask)
    count = int(xs.size)
    diag = {"pixels": count}

    if count < model.min_pixels:
        return None, _invalid(method, "NO_OBJECT", diag)

    if mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any():
        return None, _invalid(method, "CONTOUR_CLIPPED", diag)

    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    center = pts.mean(axis=0)
    rel = pts - center
    evals, evecs = np.linalg.eigh(rel.T @ rel / count)

    if np.isclose(evals[1], evals[0], rtol=1e-9, atol=1e-12):
        return None, _invalid(method, "AXIS_DEGENERATE", diag)

    e1 = evecs[:, 1]

    if e1[0] < 0 or (e1[0] == 0 and e1[1] < 0):
        e1 = -e1

    e2 = np.array([-e1[1], e1[0]])
    s, t = rel @ e1, rel @ e2

    return {"center": center, "e1": e1, "e2": e2, "s": s, "t": t, "s_min": float(s.min()), "s_max": float(s.max()),
            "diag": diag}, None


def extract_m2(mask, model: AxisModel = MODEL_DERIVED) -> AxisEstimate:
    """M2: M1과 같은 끝점 근사에 띠 폭·화소 질량 투표 머리 판별, 먼 쪽 중심선, 기울기 자르기를 쓴다.

    순서 관문·기울기 초과는 거부하지 않으므로 선 볼트를 기울기로 거부하지 못한다.
    """
    mask = _check_mask(mask)
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    pre = {"pixels": int(xs.size)}

    # 위상(조각·구멍)은 M1과 같은 순서·같은 사유
    if xs.size >= model.min_pixels and not (mask[0, :].any() or mask[-1, :].any()
                                            or mask[:, 0].any() or mask[:, -1].any()):
        m8 = mask.astype(np.uint8)
        n_labels, _ = cv2.connectedComponents(m8, connectivity=8)
        _, hierarchy = cv2.findContours(m8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        holes = 0 if hierarchy is None else int((hierarchy[0][:, 3] != -1).sum())
        pre.update(components=int(n_labels - 1), holes=holes)

        if n_labels - 1 > 1:
            return _invalid("M2", "MASK_FRAGMENTED", pre)

        if holes > 0:
            return _invalid("M2", "MASK_HOLE", pre)

    f, bad = _principal_frame(mask, model, "M2")

    if bad is not None:
        return AxisEstimate("M2", False, None, bad.reason, {**pre, **bad.diag})

    diag = {**pre, **f["diag"]}
    s, t, s_min, s_max = f["s"], f["t"], f["s_min"], f["s_max"]

    # 1. 머리 판별 투표: 25% 띠 폭과 화소 질량(주축 중점의 양쪽 화소 수, 중점 위 화소는 세지 않음)
    width_lo12, width_hi12 = _band_widths(s, t, s_min, s_max, model.band_frac)
    vote_lo, vote_hi = _band_widths(s, t, s_min, s_max, model.vote_band_frac)
    vote_ratio = max(vote_lo, vote_hi) / min(vote_lo, vote_hi)
    width_head_at_min = vote_lo > vote_hi
    mid_s = (s_min + s_max) / 2.0
    mass_lo, mass_hi = int((s < mid_s).sum()), int((s > mid_s).sum())
    mass_head_at_min = mass_lo > mass_hi
    agree = width_head_at_min == mass_head_at_min
    diag.update(width_ratio=max(width_lo12, width_hi12) / min(width_lo12, width_hi12), vote_width_ratio=vote_ratio,
                vote_mass_lo=mass_lo, vote_mass_hi=mass_hi, head_vote="agree" if agree else "disagree")

    if not agree or vote_ratio < model.vote_min_ratio:
        return _invalid("M2", "AXIS_HEAD_AMBIGUOUS", diag)

    head_at_min = width_head_at_min

    # 2. 머리 쪽을 0으로 하는 축 좌표와 1 px 단면(M1 1단계와 같음)
    s_head = (s - s_min) if head_at_min else (s_max - s)
    span = s_max - s_min
    n_bins = int(math.floor(span)) + 1
    b = np.minimum(np.floor(s_head).astype(int), n_bins - 1)
    tmin = np.full(n_bins, np.inf)
    tmax = np.full(n_bins, -np.inf)
    np.minimum.at(tmin, b, t)
    np.maximum.at(tmax, b, t)
    filled = np.isfinite(tmin)
    width = np.where(filled, tmax - tmin + 1.0, np.nan)
    mid = np.where(filled, (tmax + tmin) / 2.0, np.nan)
    pos = np.arange(n_bins) + 0.5
    idx = np.arange(n_bins)

    head_bins = filled & (pos <= model.band_frac * span + 0.5)
    shaft_far = filled & (pos >= 0.5 * span)

    if not head_bins.any() or not shaft_far.any():
        return _invalid("M2", "AXIS_ORDER_INCONSISTENT", diag)

    head_w = float(np.median(width[head_bins]))
    shaft_w = float(np.median(width[shaft_far]))
    diag.update(head_width_px=head_w, shaft_width_px=shaft_w, length_px=span + 1.0)

    # 3. 머리 계단(M1 2단계와 같은 정의). 순서 관문은 거부 대신 먼 쪽 중심선으로 돌린다
    thresh = (head_w + shaft_w) / 2.0
    wide_idx = np.nonzero(filled & (width >= thresh))[0]

    if wide_idx.size == 0:
        step = 0
    else:
        gaps = np.nonzero(np.diff(wide_idx) != 1)[0]
        step = (int(wide_idx[gaps[0]]) if gaps.size else int(wide_idx[-1])) + 1

    step_frac = step / n_bins
    order_gates = [name for name, hit in (("shaft_ge_head", shaft_w >= head_w), ("step_zero", step == 0),
                                          ("step_frac_hi", step_frac > model.max_head_step_frac)) if hit]
    diag.update(head_step_frac=step_frac, order_gate=",".join(order_gates) if order_gates else "none")

    # 4. 축척·기울기(M1 3·4단계와 같은 식). 계단 0이면 머리 구간을 12% 띠 길이로 대신한다
    step_eff = step if step > 0 else max(1, int(round(model.band_frac * span)))
    head_region = filled & (idx < step_eff)

    if not head_region.any():
        head_region = head_bins

    head_d_px = float(np.max(width[head_region]))
    r_head_px = head_d_px / 2.0
    reach = np.nonzero(head_region & (width >= head_d_px - 1.0))[0]
    first_full = int(reach[0]) if reach.size else 0
    front = filled & (idx < first_full)
    deficit = float(np.clip(head_d_px - width[front], 0.0, None).sum())
    cap_px = deficit / (r_head_px * (2.0 - math.pi / 2.0))
    sin_raw = cap_px / r_head_px
    clamped = sin_raw >= 1.0
    sin_a = 1.0 if clamped else sin_raw
    alpha = math.asin(sin_a)

    # 5. 기울기 자르기: 둥근 머리 앞면이 기울기를 부풀리므로 초과 추정은 거부 대신 상한으로 자른다
    sin_cap = math.sin(math.radians(model.max_tilt_deg))
    tilt_capped = math.degrees(alpha) > model.max_tilt_deg
    sin_used = min(sin_a, sin_cap)
    diag.update(head_diameter_px=head_d_px, head_cap_px=cap_px, tilt_deg=math.degrees(alpha), tilt_clamped=clamped,
                tilt_capped=tilt_capped, tilt_used_deg=math.degrees(math.asin(sin_used)))

    # 6. 중심선: 순서 관문에 걸리면 계단을 믿지 않고 축 먼 쪽 절반을 쓴다(단면 3개 미만이면 앞의 것)
    shaft_fit = filled & (idx > step_eff) & (idx < n_bins - 1)

    if shaft_fit.sum() >= 3:
        slope, intercept = np.polyfit(pos[shaft_fit], mid[shaft_fit], 1)
    else:
        slope, intercept = 0.0, 0.0

    centerline = "shaft"

    if order_gates:
        far_fit = shaft_far & (idx < n_bins - 1)
        centerline = "far_half"

        if far_fit.sum() >= 3:
            slope, intercept = np.polyfit(pos[far_fit], mid[far_fit], 1)

    d_head = r_head_px * sin_used
    d_tip = model.tip_to_head_radius * r_head_px * sin_used
    diag.update(centerline=centerline, centerline_slope=float(slope), centerline_offset_px=float(intercept),
                protrusion_head_px=d_head, protrusion_tip_px=d_tip)

    def point(sh: float) -> np.ndarray:
        """머리 기준 축 좌표 sh의 중심선 위 화소 좌표."""
        s_abs = (s_min + sh) if head_at_min else (s_max - sh)
        return f["center"] + s_abs * f["e1"] + (intercept + slope * sh) * f["e2"]

    head = point(0.0 + d_head)
    tip = point(span - d_tip)
    pts = np.array([head, tip])

    if (not np.isfinite(pts).all() or float(np.hypot(*(head - tip))) < 1.0
            or (pts[:, 0] < 0).any() or (pts[:, 0] > w - 1).any() or (pts[:, 1] < 0).any() or (pts[:, 1] > h - 1).any()):
        return _invalid("M2", "ENDPOINTS_INVALID", diag)

    return AxisEstimate("M2", True, _frozen(pts), None, diag)


__all__ = ["AxisEstimate", "AxisModel", "MODEL_DERIVED", "extract_m0", "extract_m1", "extract_m2"]
