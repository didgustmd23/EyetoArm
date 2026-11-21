"""볼트 축 끝점 순수 추출기(`vision/axis_endpoints.py`) 시험.

정답은 직교 투영한 원기둥 실루엣과 축선 위 기준점을 같은 투영식으로 계산한 합성 기하다.
`dim_scale`은 모든 치수를 그 배수로 줄인다(0.5 = 월드 BOLT_A `bolt_m3x20`).
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from eyetoarm.vision import axis_endpoints as ax

SHIFT = 4                      # cv2 고정소수점 꼭짓점(1/16 px)


def _hull_of_discs(discs, k_px_mm, origin, u, v, tilt_rad):
    """원판 목록 [(z_mm, r_mm)]의 직교 투영 타원들의 볼록 껍질 꼭짓점(화소 좌표)."""
    pts = []
    for z, r in discs:
        c = origin + u * (z * math.cos(tilt_rad) * k_px_mm)
        for a in np.linspace(0.0, 2 * math.pi, 96, endpoint=False):
            p = c + v * (r * math.cos(a) * k_px_mm) + u * (r * math.sin(a) * math.sin(tilt_rad) * k_px_mm)
            pts.append(p)
    pts = np.asarray(pts, np.float64)
    hull = cv2.convexHull((pts * (1 << SHIFT)).round().astype(np.int32))
    return hull


def render_bolt(*, shank_mm=12.0, theta_deg=30.0, tilt_deg=0.0, k_px_mm=4.0, size=(240, 320), center=(160.0, 120.0),
                head_d_mm=10.0, dim_scale=1.0):
    """합성 볼트 마스크와 축선 위 정답 두 점(행 0 머리 끝면 중심, 행 1 축 끝면 중심).

    `shank_mm`은 실제 축 길이, `dim_scale`은 머리 지름·높이·축 반지름·면취(반지름·길이)에 곱하는 배수다.
    """
    h, w = size
    th, tl = math.radians(theta_deg), math.radians(tilt_deg)
    u = np.array([math.cos(th), math.sin(th)])          # 화면 위 축 방향(머리 쪽 +)
    v = np.array([-u[1], u[0]])
    origin = np.asarray(center, float)                   # 머리·축 경계(z 0)
    mask = np.zeros((h, w), np.uint8)
    head_r = head_d_mm / 2.0 * dim_scale
    head_h, shaft_r, tip_r, chamfer = 6.0 * dim_scale, 3.0 * dim_scale, 2.5 * dim_scale, 0.5 * dim_scale

    for discs in ([(0.0, head_r), (head_h, head_r)],
                  [(-shank_mm + chamfer, shaft_r), (0.0, shaft_r)],
                  [(-shank_mm, tip_r), (-shank_mm + chamfer, shaft_r)]):
        cv2.fillConvexPoly(mask, _hull_of_discs(discs, k_px_mm, origin, u, v, tl), 1, lineType=cv2.LINE_8, shift=SHIFT)

    head = origin + u * (head_h * math.cos(tl) * k_px_mm)
    tip = origin - u * (shank_mm * math.cos(tl) * k_px_mm)
    return mask.astype(bool), np.array([head, tip])


def _err(est, truth):
    """머리·끝 오차(px)."""
    p = np.asarray(est.endpoints_px)
    return float(np.hypot(*(p[0] - truth[0]))), float(np.hypot(*(p[1] - truth[1])))


@pytest.mark.parametrize("method", [ax.extract_m0, ax.extract_m1])
def test_output_contract_on_lying_bolt(method):
    """누운 볼트: 유효, float64 (2,2) 읽기 전용, 행 0 = 넓은 머리 쪽, 두 점 구분, raw 범위 안."""
    mask, truth = render_bolt()
    est = method(mask)
    assert est.valid and est.reason is None
    p = est.endpoints_px
    assert p.dtype == np.float64 and p.shape == (2, 2) and not p.flags.writeable
    assert np.hypot(*(p[0] - truth[0])) < np.hypot(*(p[0] - truth[1]))       # 머리가 머리 쪽
    assert (p[:, 0] >= 0).all() and (p[:, 0] <= mask.shape[1] - 1).all()
    assert (p[:, 1] >= 0).all() and (p[:, 1] <= mask.shape[0] - 1).all()


@pytest.mark.parametrize("theta", [0.0, 37.0, 90.0, 133.0, 180.0, 251.0, 300.0])
def test_m1_follows_rotation_lying(theta):
    """누운 볼트(기울기 0)를 화면에서 돌려도 M1 두 점이 축선 기준점 1.5 px 안이고 순서가 맞다."""
    mask, truth = render_bolt(theta_deg=theta)
    est = ax.extract_m1(mask)
    assert est.valid, est.reason
    head_err, tip_err = _err(est, truth)
    assert head_err <= 1.5 and tip_err <= 1.5


@pytest.mark.parametrize("shank,tilt", [(12.0, 25.0), (12.0, 40.0), (40.0, 30.0), (40.0, 50.0)])
def test_m1_corrects_rim_protrusion_when_tilted(shank, tilt):
    """축이 화면 평면에서 기울면 실루엣 끝은 끝면 테두리만큼 바깥이다. M1은 축선 기준점 2 px 안, M0보다 머리가 가깝다."""
    mask, truth = render_bolt(shank_mm=shank, tilt_deg=tilt, theta_deg=20.0)
    m0, m1 = ax.extract_m0(mask), ax.extract_m1(mask)
    assert m0.valid and m1.valid, (m0.reason, m1.reason)
    h0, _ = _err(m0, truth)
    h1, t1 = _err(m1, truth)
    assert h1 <= 2.0 and t1 <= 2.0
    assert h0 > h1 + 2.0                                  # 기울기가 있으면 극값(M0)은 머리 테두리만큼 바깥


def test_m1_tilt_diag_without_size_estimates():
    """진단값: 추정 기울기가 합성 조건(30°)과 맞고 크기(mm) 진단은 남기지 않는다."""
    mask, _ = render_bolt(shank_mm=40.0, tilt_deg=30.0)
    est = ax.extract_m1(mask)
    assert abs(est.diag["tilt_deg"] - 30.0) <= 6.0
    assert not {"scale_px_per_mm", "observed_length_mm", "total_length_est_mm", "class_est"} & est.diag.keys()


@pytest.mark.parametrize("method", [ax.extract_m1, ax.extract_m2])
@pytest.mark.parametrize("tilt,theta", [(0.0, 20.0), (25.0, 20.0), (40.0, 133.0), (30.0, 251.0)])
def test_m3x20_endpoints_follow_axis_reference(method, tilt, theta):
    """월드 BOLT_A M3×20(0.5배 축소)도 같은 비 상수로 축선 기준점 2 px 안이다(k 8 px/mm)."""
    mask, truth = render_bolt(shank_mm=20.0, dim_scale=0.5, tilt_deg=tilt, theta_deg=theta, k_px_mm=8.0,
                              size=(320, 320), center=(160.0, 160.0))  # 약 184 px: 어느 방향이든 화면 안
    est = method(mask)
    assert est.valid, est.reason
    head_err, tip_err = _err(est, truth)
    assert head_err <= 2.0 and tip_err <= 2.0


@pytest.mark.parametrize("method", [ax.extract_m1, ax.extract_m2])
@pytest.mark.parametrize("tilt", [0.0, 30.0, 45.0])
def test_scaled_bolt_gives_same_endpoints_as_full_size_at_double_scale(method, tilt):
    """M3×20(k 8)과 M6×40(k 4) 마스크는 같은 모양이라 끝점도 같다(크기 무관)."""
    small, _ = render_bolt(shank_mm=20.0, dim_scale=0.5, tilt_deg=tilt, k_px_mm=8.0)
    full, _ = render_bolt(shank_mm=40.0, tilt_deg=tilt, k_px_mm=4.0)
    assert np.array_equal(small, full)
    a, b = method(small), method(full)
    assert a.valid and b.valid, (a.reason, b.reason)
    np.testing.assert_array_equal(a.endpoints_px, b.endpoints_px)


def test_head_ambiguous_when_both_ends_same_width():
    """양 끝 폭이 같은 막대(머리 계단 없음) → AXIS_HEAD_AMBIGUOUS, 끝점 None."""
    mask = np.zeros((120, 200), bool)
    mask[50:62, 40:150] = True
    for method in (ax.extract_m0, ax.extract_m1):
        est = method(mask)
        assert not est.valid and est.endpoints_px is None and est.reason == "AXIS_HEAD_AMBIGUOUS"


def test_order_inconsistent_when_wide_part_is_long():
    """넓은 쪽이 전체의 절반을 넘게 이어지면(머리 계단 위치 이상, 역순·가림 의심) M1은 AXIS_ORDER_INCONSISTENT."""
    mask = np.zeros((120, 220), bool)
    mask[44:68, 30:170] = True        # 넓은 부분 140 px
    mask[50:62, 170:200] = True       # 좁은 부분 30 px → 계단 위치 0.82 > 0.7
    est = ax.extract_m1(mask)
    assert not est.valid and est.reason == "AXIS_ORDER_INCONSISTENT" and est.endpoints_px is None


def test_view_oblique_when_tilt_exceeds_limit():
    """머리 끝면 타원이 둥글수록(축이 시선 쪽으로 70° 기울면) 누운 자세 가정이 깨져 AXIS_VIEW_OBLIQUE."""
    mask, _ = render_bolt(shank_mm=40.0, tilt_deg=70.0, theta_deg=10.0)
    est = ax.extract_m1(mask)
    assert not est.valid and est.reason == "AXIS_VIEW_OBLIQUE" and est.endpoints_px is None


def test_standing_bolt_is_not_valid():
    """거의 선 볼트(기울기 80°)는 어떤 사유로든 무효(끝점 None)."""
    mask, _ = render_bolt(shank_mm=12.0, tilt_deg=80.0)
    est = ax.extract_m1(mask)
    assert not est.valid and est.endpoints_px is None
    assert est.reason in {"AXIS_VIEW_OBLIQUE", "AXIS_HEAD_AMBIGUOUS", "AXIS_ORDER_INCONSISTENT"}


def test_fragmented_and_hole_rejected_by_m1():
    """조각 2개 → MASK_FRAGMENTED, 머리 안 구멍(소켓·그림자 의심) → MASK_HOLE."""
    mask, _ = render_bolt()
    frag = mask.copy()
    frag[5:15, 5:15] = True
    assert ax.extract_m1(frag).reason == "MASK_FRAGMENTED"

    holed, truth = render_bolt(theta_deg=0.0)
    hx, hy = (truth[0] + np.array([-10.0, 0.0])).round().astype(int)
    holed[hy - 2:hy + 3, hx - 2:hx + 3] = False
    assert ax.extract_m1(holed).reason == "MASK_HOLE"


@pytest.mark.parametrize("method", [ax.extract_m0, ax.extract_m1])
def test_clipped_and_tiny_rejected(method):
    """화면 테두리에 닿으면 CONTOUR_CLIPPED, 화소가 너무 적으면 NO_OBJECT."""
    mask, _ = render_bolt(center=(20.0, 120.0), theta_deg=0.0)
    assert method(mask).reason == "CONTOUR_CLIPPED"
    tiny = np.zeros((50, 50), bool)
    tiny[20:23, 20:24] = True
    assert method(tiny).reason == "NO_OBJECT"


def test_bad_input_raises():
    """2차원 bool이 아닌 입력은 ValueError(입력 경계 오류는 무효 사유로 숨기지 않음)."""
    with pytest.raises(ValueError):
        ax.extract_m1(np.zeros((4, 4, 3), bool))
    with pytest.raises(ValueError):
        ax.extract_m1(np.zeros((10, 10), np.uint8))


def test_deterministic():
    """같은 마스크 → 같은 결과(seed 없음)."""
    mask, _ = render_bolt(tilt_deg=35.0, theta_deg=70.0)
    a, b = ax.extract_m1(mask), ax.extract_m1(mask)
    assert a.endpoints_px is not None and b.endpoints_px is not None
    assert np.array_equal(a.endpoints_px, b.endpoints_px) and a.diag == b.diag


def test_m0_matches_rule_k_definition():
    """M0은 규칙 K와 같은 정의: 주성분 축 위 극값(화소 중심), 넓은 쪽이 머리, 폭 비 1.2 미만 무효."""
    mask = np.zeros((100, 200), bool)
    mask[40:60, 30:60] = True          # 머리 폭 20
    mask[45:55, 60:150] = True         # 축 폭 10
    est = ax.extract_m0(mask)
    assert est.valid and est.endpoints_px is not None
    np.testing.assert_allclose(est.endpoints_px[0][0], 30.0, atol=0.6)
    np.testing.assert_allclose(est.endpoints_px[1][0], 149.0, atol=0.6)


def test_model_constants_are_model_derived():
    """고정값은 모델 치수의 비에서 나온 값이고 크기(mm) 상수는 없다."""
    m = ax.MODEL_DERIVED
    assert m.tip_to_head_radius == 2.5 / 5.0 == 1.25 / 2.5   # M6×40 r 2.5 / 5 = M3×20 r 1.25 / 2.5
    assert (m.band_frac, m.ambiguity_ratio, m.min_pixels) == (0.12, 1.2, 20)
    assert (m.max_tilt_deg, m.max_head_step_frac) == (60.0, 0.7)
    assert (m.vote_band_frac, m.vote_min_ratio) == (0.25, 1.1)
    fields = set(ax.AxisModel.__dataclass_fields__)
    assert not fields & {"head_diameter_mm", "head_height_mm", "tip_radius_mm", "shank_lengths_mm", "class_split_mm"}
