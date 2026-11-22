"""시선 기반 볼트 끝점 보정(C1): B 입력 직전에 끝점 픽셀을 작업면에서 축 수직으로 카메라 쪽으로 k·s만큼 옮긴다.

AI 마스크가 카메라에서 먼 쪽으로 치우치는 것을 보정하며, 두 점을 같은 양만큼 옮겨 길이·축 각도는 바뀌지 않는다.
실패하면 원 끝점과 사유를 돌려주고 원 봉투·관측은 바꾸지 않는다. ROS·Qt·정답을 import하지 않는다.

종류별 형식(#270)은 `build_view_bias_plan`으로 B 규격과 대조한 `ViewBiasPlan`을 만들어 쓴다. 끝점 볼트마다
B 규격 종류를 길이로 고르고(`bolt_spec.select_spec`), 그 종류의 `plane_z`로 투영·재투영하며 그 종류의 k를 쓴다.
기존 형식(`ViewBiasConfig.legacy`)은 #270 전과 같은 평면·k 하나로 동작한다.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from eyetoarm.transforms.planar_transform import pixels_to_w, w_to_image
from eyetoarm.vision.ai_config import ViewBiasConfig
from eyetoarm.vision.bolt_spec import BoltSpec, bolt_specs_from, select_spec
from eyetoarm.vision.list_envelope import ListEnvelope, make_list_envelope
from eyetoarm.vision.observation_envelope import ObservationEnvelope

_MIN_AXIS_M = 1e-4          # 두 끝점 투영 거리 하한(B `compute_bolt_from_endpoints`와 같은 값)
_MIN_HORIZONTAL_M = 1e-6    # 카메라 바로 위(수평 거리 0)면 먼 쪽 방향이 없다
_ROUNDTRIP_TOL_PX = 1e-3    # 원 끝점 투영 → 재투영 왕복 허용(평면 뒤·호모그래피 특이 검출)
# k를 잰 평면과 B 규격 기준 높이의 허용 차(m). 1 mm 높이 차는 중심 약 1.2~1.4 mm(k 효과와 비슷)라 0.5 mm로 둔다
FIT_PLANE_TOL_M = 0.0005


@dataclass(frozen=True)
class ViewBiasResult:
    """보정 결과. `applied`가 False면 `endpoints_px`는 원 끝점이고 `reason`이 사유다.

    종류별 보정이면 `spec`(고른 B 종류, 없으면 None)·`plane_z_m`·`k_mm`·`lengths_mm`(종류별 길이)을 채운다.
    기존 형식이면 이 넷은 None이고 기록도 #270 전과 같다.
    """

    endpoints_px: np.ndarray
    applied: bool
    s: float | None
    delta_mm: float | None
    reason: str | None
    spec: str | None = None
    plane_z_m: float | None = None
    k_mm: float | None = None
    lengths_mm: dict | None = None

    def to_record(self) -> dict:
        """기록용 사전(원 끝점은 호출부가 따로 넣는다). 종류별 값은 종류별 보정일 때만 넣는다."""
        rec = {"applied": self.applied, "s": self.s, "delta_mm": self.delta_mm, "reason": self.reason}

        if self.lengths_mm is not None:
            rec.update({"spec": self.spec, "plane_z_m": self.plane_z_m, "k_mm": self.k_mm, "lengths_mm": self.lengths_mm})

        return rec


def _skip(endpoints: np.ndarray, reason: str, s: float | None = None) -> ViewBiasResult:
    """보정하지 않은 결과(원 끝점·사유)."""
    return ViewBiasResult(endpoints, False, s, None, reason)


def correct_endpoints(endpoints_px, K: np.ndarray, dist: np.ndarray, T_W_C: np.ndarray | None, *,
                      plane_z_m: float, k_mm: float, width: int | None = None, height: int | None = None) -> ViewBiasResult:
    """원본 픽셀 끝점 (2, 2)[머리, 축 끝] → C1 보정 결과. `width`·`height`를 주면 프레임 밖 결과는 버린다."""
    raw = np.array(endpoints_px, dtype=np.float64).reshape(2, 2)

    if T_W_C is None:
        return _skip(raw, "no_tf")

    T = np.asarray(T_W_C, dtype=np.float64)

    if not (np.all(np.isfinite(raw)) and np.all(np.isfinite(T)) and np.all(np.isfinite(np.asarray(K, float)))):
        return _skip(raw, "non_finite")

    # 1. 작업면 투영과 왕복 확인(평면 뒤·호모그래피 특이는 왕복이 어긋나거나 비유한)
    try:
        # 특이 호모그래피의 0 나눗셈은 아래 유한성 검사로 거른다
        with np.errstate(divide="ignore", invalid="ignore"):
            pts = pixels_to_w(raw, K, dist, T, plane_z_m)
            back = w_to_image(np.column_stack([pts, np.full(2, plane_z_m)]), K, dist, T)
    except (np.linalg.LinAlgError, ValueError):  # 특이 호모그래피·형식 오류: 보정하지 않고 사유만 남긴다
        return _skip(raw, "projection_failed")

    if not (np.all(np.isfinite(pts)) and np.all(np.isfinite(back))) or np.max(np.abs(back - raw)) > _ROUNDTRIP_TOL_PX:
        return _skip(raw, "projection_failed")

    # 2. 축·수직·먼 쪽 방향
    axis = pts[1] - pts[0]
    length = math.hypot(*axis)

    if length < _MIN_AXIS_M:
        return _skip(raw, "degenerate_axis")

    u = axis / length
    n = np.array([-u[1], u[0]])
    mid = pts.mean(axis=0)
    horizontal = mid - T[:2, 3]
    dist_h = math.hypot(*horizontal)

    if dist_h < _MIN_HORIZONTAL_M:
        return _skip(raw, "camera_above_point")

    s = float(n @ (horizontal / dist_h))

    # 3. 두 점을 같은 양만큼 카메라 쪽으로
    delta_m = k_mm / 1000.0 * s
    moved = pts - delta_m * n

    # 4. 원본 픽셀로 재투영
    out = w_to_image(np.column_stack([moved, np.full(2, plane_z_m)]), K, dist, T)

    if not np.all(np.isfinite(out)):
        return _skip(raw, "non_finite", s)

    if np.array_equal(out[0], out[1]):
        return _skip(raw, "degenerate_result", s)

    if width is not None and height is not None:
        if np.any(out < 0.0) or np.any(out[:, 0] > width - 1) or np.any(out[:, 1] > height - 1):
            return _skip(raw, "out_of_frame", s)

    return ViewBiasResult(out, True, s, delta_m * 1000.0, None)


@dataclass(frozen=True)
class ViewBiasPlan:
    """종류별 보정 실행 값: 설정(`ViewBiasConfig`, 종류별 형식) + B 규격 볼트 종류 + 종류별 k.

    `build_view_bias_plan`만 만든다(라벨·평면 대조를 거친 값). `part_specs_sha256`은 기록용(B 규격 파일 지문).
    """

    config: ViewBiasConfig
    specs: tuple[BoltSpec, ...]
    k_by_spec: tuple[tuple[str, float], ...]
    part_specs_sha256: str | None = None

    def k_of(self, name: str) -> float:
        """종류 이름 → k(mm)."""
        return dict(self.k_by_spec)[name]

    def plane_of(self, name: str) -> float:
        """종류 이름 → B 기준 높이(m)."""
        return next(s.plane_z_m for s in self.specs if s.name == name)

    def to_json(self) -> dict:
        """기록용 사전: 설정 + 종류별 (B 라벨·높이·길이 범위, k, k를 잰 평면) + B 규격 지문."""
        fit = dict(self.config.by_spec)
        rows = [{**s.to_json(), "k_mm": self.k_of(s.name), "fit_plane_z_m": fit[s.name].fit_plane_z_m} for s in self.specs]

        return {**self.config.to_json(), "specs": rows, "part_specs_sha256": self.part_specs_sha256}


def build_view_bias_plan(cfg: ViewBiasConfig, part_specs: dict, *, part_specs_sha256: str | None = None,
                         fit_tol_m: float = FIT_PLANE_TOL_M) -> ViewBiasPlan:
    """종류별 형식 설정 + B 규격 → `ViewBiasPlan`. 맞지 않으면 ValueError(실행 시작 거부용 문구).

    대조: 설정 종류 집합 = B 볼트 종류 집합, 종류마다 라벨 같음, `|fit_plane_z_m − plane_z| ≤ fit_tol_m`.
    """
    if not isinstance(cfg, ViewBiasConfig) or cfg.legacy:
        raise ValueError("build_view_bias_plan needs an axis_view_bias with by_spec")

    specs = bolt_specs_from(part_specs)
    entries = dict(cfg.by_spec)
    b_names, c_names = [s.name for s in specs], set(entries)

    if set(b_names) != c_names:
        raise ValueError(f"axis_view_bias.by_spec {sorted(c_names)} != B part_specs bolt specs {sorted(b_names)}: "
                         "add or remove entries to match the current B specs")

    problems = []

    for s in specs:
        e = entries[s.name]

        if e.label != s.label:
            problems.append(f"{s.name} label {e.label!r} != B {s.label!r}")

        if abs(e.fit_plane_z_m - s.plane_z_m) > fit_tol_m:
            problems.append(f"{s.name} fit_plane_z_m {e.fit_plane_z_m * 1000:.1f}mm != B plane_z {s.plane_z_m * 1000:.1f}mm "
                            f"(±{fit_tol_m * 1000:.1f} mm)")

    if problems:
        raise ValueError("axis_view_bias does not match B part_specs: " + "; ".join(problems)
                         + ": re-fit k for the current B specs")

    return ViewBiasPlan(cfg, specs, tuple((s.name, entries[s.name].k_mm) for s in specs), part_specs_sha256)


def correct_bolt(endpoints_px, K, dist, T_W_C, plan: ViewBiasPlan, *, width: int | None = None,
                 height: int | None = None) -> ViewBiasResult:
    """끝점 볼트 하나를 종류별로 보정: B 종류 고르기 → 그 종류 높이·k로 `correct_endpoints`.

    TF·끝점이 비유한하면 고르기 전에 그 사유로 건너뛴다. 어느 종류의 길이 범위에도 들지 않으면 `no_spec`
    (원 끝점, 종류별 길이 기록). 종류를 고른 뒤의 실패 사유는 `correct_endpoints`와 같다.
    """
    raw = np.array(endpoints_px, dtype=np.float64).reshape(2, 2)

    if T_W_C is None:
        return dataclasses.replace(_skip(raw, "no_tf"), lengths_mm={})

    T = np.asarray(T_W_C, dtype=np.float64)

    if not (np.all(np.isfinite(raw)) and np.all(np.isfinite(T)) and np.all(np.isfinite(np.asarray(K, float)))):
        return dataclasses.replace(_skip(raw, "non_finite"), lengths_mm={})

    choice = select_spec(raw, K, dist, T, plan.specs)
    lengths = choice.lengths_mm()

    if choice.name is None:
        return dataclasses.replace(_skip(raw, "no_spec"), lengths_mm=lengths)

    z, k = plan.plane_of(choice.name), plan.k_of(choice.name)
    res = correct_endpoints(raw, K, dist, T, plane_z_m=z, k_mm=k, width=width, height=height)

    return dataclasses.replace(res, spec=choice.name, plane_z_m=z, k_mm=k, lengths_mm=lengths)


def _correct_with(endpoints, camera, tf, capture, cfg) -> ViewBiasResult:
    """봉투의 카메라·TF·촬영 크기로 끝점 하나를 보정한다(기존 형식 설정 또는 종류별 계획)."""
    T = None if tf is None else tf.T_W_C

    if isinstance(cfg, ViewBiasPlan):
        return correct_bolt(endpoints, camera.K, camera.dist, T, cfg, width=capture.width, height=capture.height)

    return correct_endpoints(endpoints, camera.K, camera.dist, T,
                             plane_z_m=cfg.plane_z_m, k_mm=cfg.k_mm, width=capture.width, height=capture.height)


def check_runtime(cfg) -> None:
    """보정 인자 형 검사: None·기존 형식 설정·`ViewBiasPlan`만. 종류별 형식 설정을 계획 없이 넘기면 ValueError."""
    if cfg is None or isinstance(cfg, ViewBiasPlan):
        return

    if not isinstance(cfg, ViewBiasConfig):
        raise TypeError(f"view bias must be ViewBiasConfig or ViewBiasPlan: {type(cfg).__name__}")

    if not cfg.legacy:
        raise ValueError("axis_view_bias with by_spec needs build_view_bias_plan (B part_specs) before use")


def _record(result: ViewBiasResult, raw) -> dict:
    """기록 한 줄(결과 + 원 끝점)."""
    return {**result.to_record(), "axis_endpoints_px_raw": np.asarray(raw, dtype=np.float64).tolist()}


def apply_view_bias(envelope, camera, cfg: ViewBiasConfig | None):
    """봉투(1·2)의 볼트 끝점을 C1로 보정한 새 봉투와 기록 목록을 돌려준다. `cfg`가 None이면 (같은 봉투, []).

    봉투 2는 끝점 있는 후보마다 기록하고 목록 자체의 `camera`를 쓴다. 그 밖의 형은 TypeError.
    """
    if cfg is None:
        return envelope, []

    check_runtime(cfg)

    if isinstance(envelope, ObservationEnvelope):
        obs = envelope.observation

        if obs.axis_endpoints_px is None:
            return envelope, [{"instance_index": None, "applied": False, "s": None, "delta_mm": None,
                               "reason": "no_endpoints", "axis_endpoints_px_raw": None}]

        res = _correct_with(obs.axis_endpoints_px, camera, envelope.tf, envelope.capture, cfg)
        rec = {"instance_index": None, **_record(res, obs.axis_endpoints_px)}

        if not res.applied:
            return envelope, [rec]

        new_obs = dataclasses.replace(obs, axis_endpoints_px=res.endpoints_px)

        return dataclasses.replace(envelope, observation=new_obs), [rec]

    if isinstance(envelope, ListEnvelope):
        fo = envelope.observations
        records, cands, changed = [], [], False

        for c in fo.candidates:
            if c.axis_endpoints_px is None:
                cands.append(c)
                continue

            res = _correct_with(c.axis_endpoints_px, fo.camera, fo.tf, fo.capture, cfg)
            records.append({"instance_index": c.instance_index, **_record(res, c.axis_endpoints_px)})

            if res.applied:
                cands.append(dataclasses.replace(c, axis_endpoints_px=res.endpoints_px))
                changed = True
            else:
                cands.append(c)

        if not changed:
            return envelope, records

        new_fo = dataclasses.replace(fo, candidates=tuple(cands))
        world = fo.tf.parent if fo.tf is not None else "world"

        return make_list_envelope(new_fo, world_frame=world, version=envelope.version), records

    raise TypeError(f"apply_view_bias: unsupported envelope type {type(envelope).__name__}")


def summarize(records_list) -> dict:
    """기록 목록들 → {applied, skipped_by_reason} 수."""
    applied, skipped = 0, {}

    for records in records_list:
        for r in records:
            if r["applied"]:
                applied += 1
            else:
                skipped[r["reason"]] = skipped.get(r["reason"], 0) + 1

    return {"applied": applied, "skipped_by_reason": dict(sorted(skipped.items()))}


__all__ = ["FIT_PLANE_TOL_M", "ViewBiasPlan", "ViewBiasResult", "apply_view_bias", "build_view_bias_plan", "check_runtime",
           "correct_bolt", "correct_endpoints", "summarize"]
