"""볼트 마스크 하나 → 축선 위 두 끝점 픽셀(머리 끝·축 끝) 순수 추출기(#163 단계 (i), 진단용).

근거: #163 명세 r1 §2·§4(`_workspace/163_builder_axis_endpoint_b_input_spec.md`), A-8 점 의미 계약
(`PlanarObservation.axis_endpoints_px` 행 0 = 머리 끝, 행 1 = 축 끝, 축선 위 기준점, RAW 전체 프레임 픽셀).

이 모듈은 아직 실제 AI 경로(`ai_convert.convert`·`observation_from_ai`)에 연결하지 않는다(단계 (ii)는 별도 승인).
결과는 진단용이며 관측을 유효로 만들지 않는다. NumPy·OpenCV만 쓰고 정답·카메라 자세·world 기하를 쓰지 않는다.

방법
- M0(`extract_m0`): 규칙 K와 같은 정의. 화소 주성분 축 위 극값(화소 중심), 양 끝 폭이 넓은 쪽이 머리.
  머리 끝면 **테두리의 가장 바깥 투영점**에 가까워 축선 위 기준점과 다르다(#159 X0). 비교 기준선 전용.
- M1(`extract_m1`): 축선 위 기준점 근사.
  1. M0과 같은 주축·머리 판정 뒤, 축을 따라 1 px 단면마다 폭·중점을 잰다.
  2. 머리 계단(폭이 머리·축 폭 중간값 아래로 내려가는 곳)을 찾아 순서 일관성을 본다. 계단은 **머리 쪽 첫 넓은 단면부터 끊김 없이
     이어진 넓은 구간의 끝 + 1**이다(#177 C1). 축 끝 쪽의 고립된 넓은 단면은 계단으로 보지 않는다(#176: 1 px 양자화로 생긴 고립 단면
     하나가 계단을 축 끝까지 밀어 `AXIS_ORDER_INCONSISTENT`가 되던 결함).
  3. 머리 지름 px = 머리 구간 단면 폭 최댓값. 직교 투영에서 끝면 타원의 장축은 기울기와 무관하다. 머리 반지름 px = 그 절반.
  4. 기울기 α(축과 화면 평면 사이 각): 머리 끝면은 축 방향 반지름 c = r_head_px·sinα인 반타원으로 보인다.
     폭이 머리 지름에 처음 닿기 전 단면들의 폭 부족 면적 = c·b·(2 − π/2)(b = 지름 px/2)에서 c를 잰다.
     (실루엣 길이 L·cosα + R·sinα만으로는 α = 0과 α ≈ 2·atan(R/L)을 구분하지 못한다.)
     c가 머리 반지름(px) 이상이라 sinα를 1로 자른 경우(#177 T1)는 `tilt_clamped=True`이고, 판정은 기울기 90°로 `AXIS_VIEW_OBLIQUE`다.
  5. 축 중심선(축 구간 단면 중점의 직선 적합) 위에서, 실루엣 양 끝을 끝면 테두리 돌출만큼 안쪽으로 옮긴다.
     머리 쪽 r_head_px·sinα, 축 끝 쪽 `tip_to_head_radius`·r_head_px·sinα.
     반화소 보정은 하지 않는다(규칙 K·G1과 같은 「정수 = 화소 중심」 규약, 극값 화소 중심을 실루엣 끝으로 본다).
- M2(`extract_m2`, #225): #223 정책 P7. M1과 같은 단면·축척·기울기 식에 (1) 25% 띠 폭 + 화소 질량 투표 머리 판별,
  (2) 순서 관문 대신 축 먼 쪽 절반 중심선, (3) 기울기 60° 자르기를 쓴다. M1은 바뀌지 않는다(#207 시야 보정 계수가 M1 기준).
- 원근(끝마다 다른 축척)과 끝면 면취의 세부는 무시한다. 오차는 단계 (i) 오프라인 진단에서 따로 잰다.

고정값(`MODEL_DERIVED`)은 볼트 모델 치수의 **비**에서만 정했다(simulation 모델). #159 결과를 보고 바꾸지 않는다.
바꿔야 하면 함수 인자로 다른 `AxisModel`을 넘긴다.

크기(mm)를 쓰지 않는 이유(#245 E1): 월드 볼트 BOLT_A `bolt_m3x20`은 BOLT_B `bolt_m6x40` 메시를 0.5배로 줄인 모델이라(#209)
머리 지름 : 축 지름 : 축 끝 돌출 반지름 비가 같다. 그래서 끝점은 머리 지름 px와 비만으로 정해지고, 영상 하나로는 두 볼트의 크기를
구분할 수 없다(같은 모양의 축소형). 종류·길이(mm)는 끝점을 작업면에 투영하는 B가 정한다(사용자 결정 2026-11-18: 변환·mm는 B 책임).
이 가정은 두 볼트가 같은 비율일 때만 맞다. 비율이 다른 볼트가 들어오면 종류별 비가 필요하다(#245 계획 r0 §2.4).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AxisModel:
    """모델 치수의 비에서 나온 고정값(무차원)과 각도 deg, 화소 수. 길이(mm)는 쓰지 않는다(모듈 설명 #245 E1).

    치수 근거(simulation 모델): BOLT_B `bolt_m6x40` 머리 ⌀10 × 6, 축 ⌀6 × 40, 축 끝 면취 안쪽 r 2.
    BOLT_A `bolt_m3x20`은 그 0.5배(머리 ⌀5 × 3, 축 ⌀3 × 20). 두 볼트의 비는 같다.
    """

    tip_to_head_radius: float = 0.5     # 축 끝 돌출 반지름 / 머리 반지름: (면취 안쪽 r 2와 축 r 3의 중간 2.5) / 5 = 1.25 / 2.5
    band_frac: float = 0.12             # 규칙 K와 같음(머리 길이 비 6/46 = 3/23 ≈ 0.13보다 작게)
    ambiguity_ratio: float = 1.2        # 머리 지름 / 축 지름 = 10/6 = 5/3 ≈ 1.67보다 충분히 작게
    min_pixels: int = 20
    max_tilt_deg: float = 60.0          # 이보다 서면 누운 자세(축 높이 평면) 가정이 깨진다고 본다
    max_head_step_frac: float = 0.7     # 머리 계단 위치 상한(#177 값 그대로). 근거였던 M6×12 머리 비(α ≤ 60°에서 ≈ 0.67)는
    #                                     월드에서 빠졌고, 지금 두 볼트(머리 길이 비 ≈ 0.13)에는 여유가 더 크다
    vote_band_frac: float = 0.25        # M2 머리 투표 띠(#223 P7). M1은 쓰지 않는다
    vote_min_ratio: float = 1.1         # M2 투표 띠 폭 비 하한(#223 P7 r ≥ 1.1). M1은 쓰지 않는다


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
    """M0(규칙 K 정의): 주축 위 화소 중심 극값. 진단 기준선 전용(축선 위 기준점이 아님)."""
    mask = _check_mask(mask)
    f, bad = _axis_frame(mask, model, "M0")

    if bad is not None:
        return bad

    lo_pt = f["center"] + f["s_min"] * f["e1"]
    hi_pt = f["center"] + f["s_max"] * f["e1"]
    head, tip = (lo_pt, hi_pt) if f["head_at_min"] else (hi_pt, lo_pt)
    return AxisEstimate("M0", True, _frozen([head, tip]), None, f["diag"])


def extract_m1(mask, model: AxisModel = MODEL_DERIVED) -> AxisEstimate:
    """M1: 축선 위 머리 끝·축 끝 기준점 근사(모델 치수 기반 테두리 돌출 보정)."""
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

    # 2. 머리 계단(#177 C1): 폭이 (머리+축)/2 이상인 단면 중 머리 쪽 첫 단면부터 끊김 없이 이어진 구간의 끝 + 1.
    #    이어짐 = 인덱스가 1씩 이어지는 넓은 단면. 빈 단면(filled=False)과 문턱 미만 단면은 모두 끊김이며 허용 끊김은 0개다.
    #    첫 넓은 단면이 0번이 아니어도(머리 끝 둥근 테두리 단면) 그 단면부터 센다. 문턱·max_head_step_frac은 바꾸지 않는다
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

    # 3. 머리 지름 px: 머리 구간(계단 앞) 단면 폭의 최댓값(직교 투영에서 끝면 타원 장축은 기울기와 무관)
    head_region = filled & (np.arange(n_bins) < step)
    head_d_px = float(np.max(width[head_region]))
    r_head_px = head_d_px / 2.0

    # 4. 기울기: 머리 끝 타원의 축 방향 반지름 c(px) = 앞쪽(폭이 머리 지름에 처음 닿기 전) 단면 폭 부족 면적
    #    / (b·(2 − π/2)), b = 머리 반지름 px. sinα = c / 머리 반지름 px. 크기(mm)는 쓰지 않는다(#245 E1)
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
    """주축 양 끝 frac 띠(길이 비)의 단면 폭(lo 쪽, hi 쪽). 폭 = t 범위 + 1 px(M1 `_axis_frame`과 같은 정의)."""
    band = frac * (s_max - s_min)
    lo, hi = s <= s_min + band, s >= s_max - band

    return float(t[lo].max() - t[lo].min() + 1.0), float(t[hi].max() - t[hi].min() + 1.0)


def _principal_frame(mask: np.ndarray, model: AxisModel, method: str):
    """M2 공통 단계: 화소 수·테두리·주축(머리 판정 없음). (frame 사전, None) 또는 (None, 무효 결과).

    M1 `_axis_frame`의 앞부분과 같은 계산·같은 사유 순서다(12% 띠 모호 관문만 없다). M1을 바꾸지 않으려고 따로 둔다.
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
    """M2(#223 P7, #225): M1과 같은 축선 위 기준점 근사에 머리 판별·순서·기울기 처리를 바꾼 추출기.

    M1과 다른 점(근거 `_workspace/223_builder3_axis_failure_report.md` §5, 정책 P7 r ≥ 1.1):
    1. 머리 판별: 양 끝 `vote_band_frac`(25%) 띠 폭의 넓은 쪽과 주축 중점 기준 화소 질량이 많은 쪽이 같고, 띠 폭 비가
       `vote_min_ratio`(1.1) 이상이어야 한다. 아니면 `AXIS_HEAD_AMBIGUOUS`. 이 투표가 배경·너트처럼 머리 계단이 없는 형상의 관문이다.
    2. 순서 관문(축이 머리보다 넓음·계단 0·계단 비 > `max_head_step_frac`)에 걸리면 거부하지 않고, 계단 위치를 믿지 않는
       축 먼 쪽 절반 단면의 중심선을 쓴다(`centerline="far_half"`). 계단이 0이면 머리 구간을 12% 띠 길이로 대신한다.
    3. 기울기 추정이 `max_tilt_deg`(60°)를 넘으면 거부하지 않고 sin을 sin 60°로 잘라 안쪽 보정에 쓴다(`tilt_capped`).
       예측 마스크의 둥근 머리 앞면이 기울기를 크게 부풀리기 때문이다(실제 ~38° → 추정 90°, #223 §3).
    그 밖(위상·잘림·퇴화·끝점 무효 사유, 머리 지름 px·폭 부족 면적 기울기, 안쪽 보정 식)은 M1과 같다. 새 사유 코드는 없다.

    위험(#223 §6): 기울기 관문이 사라져 선 볼트를 기울기로 거부하지 못한다. 저장 자료에는 선 볼트가 없어 시험하지 못했다.
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

    # 5. 기울기 자르기: 60° 초과 추정은 sin 60°로 보정한다(거부하지 않음)
    sin_cap = math.sin(math.radians(model.max_tilt_deg))
    tilt_capped = math.degrees(alpha) > model.max_tilt_deg
    sin_used = min(sin_a, sin_cap)
    diag.update(head_diameter_px=head_d_px, head_cap_px=cap_px, tilt_deg=math.degrees(alpha), tilt_clamped=clamped,
                tilt_capped=tilt_capped, tilt_used_deg=math.degrees(math.asin(sin_used)))

    # 6. 중심선: 기본은 계단 뒤 축 구간, 순서 관문에 걸리면 축 먼 쪽 절반(마지막 단면 제외). 단면 3개 미만이면 앞의 것을 쓴다
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
