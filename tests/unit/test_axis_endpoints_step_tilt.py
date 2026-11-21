"""`vision.axis_endpoints.extract_m1`의 머리 계단 정의(#177 C1)와 기울기 sin 잘림 진단(#177 T1) 단위시험.

근거: `_workspace/177_reviewer_instruction_m1_c1_t1.md`, 원인 분석 `_workspace/176_builder2_m1_axis_order_report.md` §2~§5.
- C1: 계단 = 머리 쪽 첫 넓은 단면부터 끊김 없이 이어진 넓은 구간의 끝 + 1(고립된 넓은 단면 무시). 문턱·`max_head_step_frac`는 그대로.
- T1: sinα를 1로 자르면 `tilt_clamped=True`, 총길이·종류 None, 판정은 `AXIS_VIEW_OBLIQUE` 그대로.
- 실기록 자료: `tests/unit/data/m1_live175_bolt_polygon.json`(#175 실시간 S1-08 볼트 윤곽, 출처는 파일 안 `source`).
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from eyetoarm.vision.ai_convert import rasterize_polygon
from eyetoarm.vision.axis_endpoints import MODEL_DERIVED, extract_m1

DATA = Path(__file__).resolve().parent / "data" / "m1_live175_bolt_polygon.json"


def flat_bolt(*, bump_at=None, head_r=10, ramp=0, head_len=20, shaft_end=160):
    """가로로 누운 합성 볼트(머리 왼쪽): 머리 지름 2·head_r, 축 폭 12 px. bump_at 열에 폭 18 px 고립 단면, ramp px 길이의 뾰족한 머리 앞."""
    m = np.zeros((240, 320), np.uint8)
    y, x0 = 120, 40

    if ramp:
        cv2.fillPoly(m, [np.array([[x0, y], [x0 + ramp, y - head_r], [x0 + head_len, y - head_r],
                                   [x0 + head_len, y + head_r - 1], [x0 + ramp, y + head_r - 1]], np.int32)], 1)
    else:
        m[y - head_r:y + head_r, x0:x0 + head_len] = 1

    m[y - 6:y + 6, x0 + head_len:shaft_end] = 1

    if bump_at is not None:
        m[y - 9:y + 9, bump_at] = 1

    return m.astype(bool)


def old_step_frac(mask):
    """#177 이전 정의(문턱 이상 **마지막** 단면 + 1)의 계단 비율. 이 축 정렬 합성 볼트(머리 왼쪽)에서만 쓴다."""
    xs = np.nonzero(mask.any(axis=0))[0]
    widths = mask[:, xs].sum(axis=0).astype(float)
    n = len(widths)
    span = n - 1
    pos = np.arange(n) + 0.5
    head_w = float(np.median(widths[pos <= MODEL_DERIVED.band_frac * span + 0.5]))
    shaft_w = float(np.median(widths[pos >= 0.5 * span]))
    wide = np.nonzero(widths >= (head_w + shaft_w) / 2.0)[0]

    return (int(wide.max()) + 1) / n


def test_c1_isolated_wide_section_near_tip_is_ignored():
    """축 끝 쪽 고립된 넓은 단면 하나는 계단이 아니다: 이전 정의라면 0.7을 넘지만 C1에서는 머리 끝에서 계단을 잡아 유효."""
    mask = flat_bolt(bump_at=130)

    est = extract_m1(mask)

    assert old_step_frac(mask) > MODEL_DERIVED.max_head_step_frac
    assert est.valid and est.reason is None
    assert est.diag["head_step_frac"] == pytest.approx(20 / 120)


def test_c1_long_contiguous_wide_part_still_order_inconsistent():
    """머리 쪽부터 이어진 넓은 구간이 0.7을 넘으면(역순·가림 의심) C1에서도 AXIS_ORDER_INCONSISTENT."""
    mask = flat_bolt(head_len=100)

    est = extract_m1(mask)

    assert not est.valid and est.reason == "AXIS_ORDER_INCONSISTENT" and est.endpoints_px is None
    assert est.diag["head_step_frac"] > MODEL_DERIVED.max_head_step_frac


def test_c1_head_no_wider_than_shaft_still_order_inconsistent():
    """머리 띠 폭 중앙값이 축 폭 이하이면(머리 없음) 계단 정의와 무관하게 AXIS_ORDER_INCONSISTENT."""
    mask = flat_bolt(head_r=10, ramp=18)

    est = extract_m1(mask)

    assert not est.valid and est.reason == "AXIS_ORDER_INCONSISTENT"


def test_c1_live175_polygon_becomes_valid_with_head_first():
    """#175 실시간 윤곽(이전 0.857로 실패)은 C1에서 유효이고 행 0이 머리 끝(오른쪽), 같은 장면 오프라인 정답과 6 px 안."""
    doc = json.loads(DATA.read_text(encoding="utf-8"))
    mask = rasterize_polygon(np.asarray(doc["polygon"], float), np.zeros(2), tuple(doc["frame_wh"]))
    ref = np.asarray(doc["reference_endpoints_px"], float)

    est = extract_m1(mask)

    assert est.valid and est.reason is None
    assert est.diag["head_step_frac"] < MODEL_DERIVED.max_head_step_frac
    assert est.diag["tilt_clamped"] is False
    assert est.endpoints_px[0][0] > est.endpoints_px[1][0]
    assert np.max(np.hypot(*(est.endpoints_px - ref).T)) < 6.0


def test_t1_clamped_tilt_has_no_length_or_class():
    """머리 앞 부족 면적이 반지름을 넘어 sinα를 1로 자르면 tilt_clamped=True, 사유 AXIS_VIEW_OBLIQUE. 총길이·종류 진단은 없다(#245 E1)."""
    mask = flat_bolt(ramp=5)

    est = extract_m1(mask)
    d = est.diag

    assert not est.valid and est.reason == "AXIS_VIEW_OBLIQUE" and est.endpoints_px is None
    assert d["tilt_clamped"] is True and d["tilt_deg"] == pytest.approx(90.0)
    assert not {"total_length_est_mm", "class_est", "observed_length_mm", "scale_px_per_mm"} & d.keys()
    assert d["head_cap_px"] >= d["head_diameter_px"] / 2.0
    assert math.isfinite(d["head_cap_px"]) and math.isfinite(d["head_diameter_px"])


def test_t1_unclamped_tilt_keeps_finite_values():
    """잘리지 않으면 tilt_clamped=False이고 기울기·머리 진단은 유한, 크기(mm) 진단은 없다(#245 E1)."""
    mask = flat_bolt(ramp=3)

    est = extract_m1(mask)
    d = est.diag

    assert est.valid
    assert d["tilt_clamped"] is False and d["tilt_deg"] < 90.0
    assert d["head_cap_px"] < d["head_diameter_px"] / 2.0 and math.isfinite(d["tilt_deg"])
    assert not {"total_length_est_mm", "class_est", "observed_length_mm", "scale_px_per_mm"} & d.keys()


def test_t1_none_values_are_dropped_by_record_diag():
    """기록용 진단 선택(`ai_convert._diag`)은 추출기에 없는 총길이·종류 키(#245 E1)를 깨지지 않고 빼고, 유한 수는 남긴다."""
    from eyetoarm.vision.ai_convert import _diag

    est = extract_m1(flat_bolt(ramp=5))

    out = _diag(est.diag)

    assert "total_length_est_mm" not in out and "class_est" not in out
    assert out["tilt_deg"] == pytest.approx(90.0)
    json.dumps(out, allow_nan=False)
