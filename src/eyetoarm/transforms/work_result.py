"""B 판정 — 윤곽(원본 픽셀) → W 중심점·장축 각도·규격(BOLT_A/BOLT_B/NUT).

근거: B 상세설계 §7(중심·각도), §15(규격 판정), 공통 자료형 초안 §3.4(WorkPlanarResult).

처리 순서
1. 원본 픽셀 윤곽 → W 평면(`plane_z`) 윤곽 [m]  (`planar_transform.pixels_to_w`)
2. 중심 = W 윤곽의 면적 중심(moments). 픽셀에서 구한 중심을 옮기면 원근 때문에 어긋나므로
   반드시 W로 옮긴 다음 구한다.
3. 면적 주축 = 장축 각도 θ, 그 축 방향 범위 = 길이, 직각 방향 범위 = 폭.
   θ는 W의 +X축에서 반시계 방향, [0°, 180°)로 접는다. 너트는 θ를 쓰지 않는다.
4. 규격: 볼트는 머리 포함 전체 길이로 BOLT_A(약 18 mm) / BOLT_B(약 46 mm),
   너트는 맞변 크기로 NUT. 기준을 벗어나면 `UNRESOLVED`(무효).

ROS를 import하지 않는다.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from eyetoarm.calibration.profile import CameraProfile
from eyetoarm.transforms.planar_transform import pixels_to_w

UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class WorkPlanarResult:
    """부품 하나의 B 결과. 길이는 m, 각도는 도(deg)."""

    center_w_m: tuple[float, float]
    plane_z_m: float
    theta_deg: float | None
    theta_valid: bool
    spec: str
    class_name: str | None
    measured_length_m: float
    measured_width_m: float
    aspect_ratio: float
    area_w_m2: float
    contour_w_m: np.ndarray
    valid: bool
    reason: str | None
    reason_owner: str | None
    calibration_id: str
    camera_frame: str
    compute_ms: float
    part_track_id: int | None = None
    image_seq: int | None = None
    grasp_w_m: tuple[float, float] | None = None   # 집는 점 (볼트 두 끝점 입력: 축 가운데 Pb). None이면 center_w_m
    method: str = "contour"                         # "contour" | "axis_endpoints"


def load_specs(path: str | Path) -> dict:
    """`config/calibration/part_specs.json`을 읽는다."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _principal_axes(contour_mm: np.ndarray, m: dict) -> tuple[float, float, float]:
    """W 윤곽(mm) → (장축 방향 길이, 직각 방향 폭, 장축 각도 deg [0,180)).

    각도는 면적 2차 모멘트(주축)로 구한다. 볼트처럼 한쪽만 굵은(머리) 모양도 축에 대칭이면
    주축이 정확히 축과 맞는다. 최소 외접 사각형은 머리 모서리 때문에 몇 도 기울 수 있다.
    길이·폭은 그 축으로 윤곽을 투영한 범위다.
    """
    if abs(m["m00"]) < 1e-9:
        theta = 0.0
    else:
        mu20, mu02, mu11 = m["mu20"] / m["m00"], m["mu02"] / m["m00"], m["mu11"] / m["m00"]
        theta = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)

    u = np.array([math.cos(theta), math.sin(theta)])
    v = np.array([-u[1], u[0]])
    pu, pv = contour_mm @ u, contour_mm @ v

    return float(pu.max() - pu.min()), float(pv.max() - pv.min()), math.degrees(theta) % 180.0


def normalize_class(class_name: str | None) -> str | None:
    """A(OpenCV·YOLO) 클래스 이름 → "bolt" / "nut" / None.

    YOLO 2클래스(`bolt`·`nut`, `scripts/coco_to_yolo_seg.py`)와 3클래스 안(`BOLT_A`·`BOLT_B`·`NUT`)을 함께 받는다.
    3클래스라도 규격은 B가 길이로 다시 판정한다(A 이름은 볼트/너트 구분에만 쓴다). 모르는 이름은 None(모양으로 추정).
    """
    if not class_name:
        return None

    name = str(class_name).strip().lower()

    if name.startswith("bolt"):
        return "bolt"

    if name.startswith("nut"):
        return "nut"

    return None


def _classify(class_name, length_m, width_m, aspect, specs) -> tuple[str, str | None]:
    """규격 판정. (spec, 무효 사유) — 사유가 None이면 유효."""
    s = specs["specs"]
    class_name = normalize_class(class_name)
    is_bolt = class_name == "bolt" or (class_name is None and aspect >= specs["bolt_min_aspect"])

    if is_bolt:
        if aspect < specs["bolt_min_aspect"]:
            return UNRESOLVED, "ASPECT_TOO_LOW"

        best, best_err = None, None

        for name in ("BOLT_A", "BOLT_B"):
            err = abs(length_m - s[name]["total_length"])

            if err <= s[name]["length_tolerance"] and (best_err is None or err < best_err):
                best, best_err = name, err

        return (best, None) if best else (UNRESOLVED, "LENGTH_OUT_OF_RANGE")

    nut = s["NUT"]
    lo = nut["across_flats"] - nut["size_tolerance"]
    hi = nut["across_corners"] + nut["size_tolerance"]

    if lo <= length_m <= hi and lo <= width_m <= hi:
        return "NUT", None

    return UNRESOLVED, "NUT_SIZE_OUT_OF_RANGE"


def class_plane_z(specs: dict, class_name: str | None, spec: str | None = None) -> float:
    """종류별 기준 높이 (B 선택 2026-11-17). 볼트 축 105.0 mm, 너트 윗면 105.2 mm (2026-11-20 Gazebo 실측). 모르면 기본 plane_z."""
    key = spec if spec in specs["specs"] else {"bolt": "BOLT_A", "nut": "NUT"}.get(normalize_class(class_name) or "")

    return float(specs["specs"].get(key, {}).get("plane_z", specs["plane_z"])) if key else float(specs["plane_z"])


def compute_bolt_from_endpoints(head_px, tip_px, camera: CameraProfile, T_W_C: np.ndarray, specs: dict,
                                class_name: str | None = "bolt", observed_calibration_id: str | None = None,
                                part_track_id: int | None = None, image_seq: int | None = None) -> WorkPlanarResult:
    """볼트 머리 끝·축 끝 픽셀 두 점 → `WorkPlanarResult` (B 선택 2026-11-17, A-2).

    - 두 점은 축 위의 점이라 높이 = 볼트 축 높이(105.0 mm). 그 평면으로 투영한다.
    - 길이 = 두 점 거리(머리 포함 총길이), 각도 = 축 방향 [0, 180).
    - 위치 `center_w_m` = 총길이 중점 Pa. 집는 점 `grasp_w_m` = 축 가운데 Pb = Pa에서 축 끝 쪽으로 머리 높이/2(3 mm).
    """
    t0 = time.perf_counter()
    plane_z = class_plane_z(specs, "bolt")
    reason, owner = None, None

    if observed_calibration_id is not None and observed_calibration_id != camera.calibration_id:
        reason, owner = "CALIBRATION_MISMATCH", "A"

    pts = pixels_to_w(np.array([head_px, tip_px], dtype=np.float64).reshape(2, 2), camera.K, camera.dist, T_W_C, plane_z)
    head, tip = pts[0], pts[1]
    axis = tip - head
    length = float(np.hypot(*axis))

    if not np.isfinite(length) or length < 1e-4:
        reason, owner = reason or "ENDPOINTS_INVALID", owner or "A"
        unit = np.array([1.0, 0.0])
    else:
        unit = axis / length

    pa = (head + tip) / 2.0
    head_h = float(specs["specs"]["BOLT_A"].get("head_height", 0.006))
    pb = pa + unit * head_h / 2.0
    theta = math.degrees(math.atan2(unit[1], unit[0])) % 180.0
    spec, cls_reason = _classify("bolt", length, 0.0, float("inf"), specs)

    if reason is None and cls_reason is not None:
        reason, owner = cls_reason, "B"

    return WorkPlanarResult(
        center_w_m=(float(pa[0]), float(pa[1])), plane_z_m=plane_z,
        theta_deg=float(theta) if reason is None else None, theta_valid=reason is None,
        spec=spec if reason is None else UNRESOLVED, class_name=class_name,
        measured_length_m=length, measured_width_m=float(specs["specs"]["BOLT_A"]["shaft_diameter"]),
        aspect_ratio=float("inf"), area_w_m2=0.0, contour_w_m=pts, valid=reason is None,
        reason=reason, reason_owner=owner, calibration_id=camera.calibration_id, camera_frame=camera.frame_id,
        compute_ms=(time.perf_counter() - t0) * 1000.0, part_track_id=part_track_id, image_seq=image_seq,
        grasp_w_m=(float(pb[0]), float(pb[1])), method="axis_endpoints",
    )


def compute_work_result(contour_px, camera: CameraProfile, T_W_C: np.ndarray, specs: dict,
                        class_name: str | None = None, plane_z: float | None = None,
                        observed_calibration_id: str | None = None,
                        part_track_id: int | None = None, image_seq: int | None = None) -> WorkPlanarResult:
    """원본 픽셀 윤곽 (N, 2) 하나 → `WorkPlanarResult`.

    - `class_name`: A의 클래스("bolt"/"nut"). None이면 모양(가로세로비)으로 추정한다.
    - `observed_calibration_id`: A 관측에 붙은 보정 식별자. 다르면 무효(책임 A).
    """
    t0 = time.perf_counter()
    plane_z = class_plane_z(specs, class_name) if plane_z is None else plane_z    # 너트 105.2, 볼트·모름 105.0 mm
    pts = np.asarray(contour_px, dtype=np.float64).reshape(-1, 2)

    reason, owner = None, None

    if observed_calibration_id is not None and observed_calibration_id != camera.calibration_id:
        reason, owner = "CALIBRATION_MISMATCH", "A"

    if pts.shape[0] < 3:
        reason, owner = "CONTOUR_TOO_SMALL", "A"
        pts = np.vstack([pts, np.zeros((3 - pts.shape[0], 2))]) if pts.size else np.zeros((3, 2))

    cw = pixels_to_w(pts, camera.K, camera.dist, T_W_C, plane_z)
    cmm = cw * 1000.0
    m = cv2.moments(cmm.astype(np.float32))

    if abs(m["m00"]) < 1e-9:
        cx, cy = float(cw[:, 0].mean()), float(cw[:, 1].mean())
        area = 0.0
    else:
        cx, cy = m["m10"] / m["m00"] / 1000.0, m["m01"] / m["m00"] / 1000.0
        area = abs(m["m00"]) / 1e6

    long_mm, short_mm, theta = _principal_axes(cmm, m)
    length_m, width_m = long_mm / 1000.0, short_mm / 1000.0
    aspect = long_mm / short_mm if short_mm > 1e-9 else float("inf")

    spec, cls_reason = _classify(class_name, length_m, width_m, aspect, specs)

    if reason is None and cls_reason is not None:
        reason, owner = cls_reason, "B"

    is_bolt = spec in ("BOLT_A", "BOLT_B") or (spec == UNRESOLVED and normalize_class(class_name) == "bolt")
    theta_valid = bool(is_bolt and aspect >= specs["bolt_min_aspect"])

    return WorkPlanarResult(
        center_w_m=(float(cx), float(cy)), plane_z_m=float(plane_z),
        theta_deg=float(theta) if theta_valid else None, theta_valid=theta_valid,
        spec=spec if reason is None else UNRESOLVED, class_name=class_name,
        measured_length_m=length_m, measured_width_m=width_m, aspect_ratio=float(aspect),
        area_w_m2=float(area), contour_w_m=cw, valid=reason is None,
        reason=reason, reason_owner=owner, calibration_id=camera.calibration_id,
        camera_frame=camera.frame_id, compute_ms=(time.perf_counter() - t0) * 1000.0,
        part_track_id=part_track_id, image_seq=image_seq,
    )


def from_observation(obs, frame, camera: CameraProfile, T_W_C: np.ndarray, specs: dict,
                     plane_z: float | None = None) -> WorkPlanarResult:
    """A의 `PlanarObservation` + 그 `FramePacket` → B 결과 (A 협업 요청 B-07 수락).

    - `T_W_C`는 호출자가 **촬영 시각**(`frame.ros_stamp_ns`, SIM_TIME)의 TF
      `world` → `frame.optical_frame`으로 구해서 넣는다. 링크 카메라는 팔과 같이 움직인다.
    - A가 무효로 준 관측은 계산하지 않고 A의 사유를 그대로 둔다 (책임 A).
    - 윤곽이 이미 왜곡 보정된 것(`pixel_basis=undistorted`)이면 B는 다시 보정하지 않는다.
    """
    ids = {"part_track_id": getattr(obs, "part_track_id", None), "image_seq": obs.image_seq}

    if obs.source_id != frame.source_id or obs.image_seq != frame.image_seq:
        raise ValueError("관측과 프레임의 source_id·image_seq가 다르다")

    endpoints = getattr(obs, "axis_endpoints_px", None)     # B 선택: 볼트는 [머리 끝, 축 끝] 픽셀 (새 입력 계약, A 협의 중)

    if obs.valid and endpoints is not None:
        cam = camera

        if str(getattr(obs.pixel_basis, "value", obs.pixel_basis)) == "undistorted" and np.any(camera.dist):
            cam = CameraProfile(camera.width, camera.height, camera.K, np.zeros(5), camera.frame_id,
                                camera.calibration_id)

        head, tip = endpoints
        return compute_bolt_from_endpoints(head, tip, cam, T_W_C, specs, class_name=obs.class_name or "bolt",
                                           observed_calibration_id=obs.calibration_id, **ids)

    if not obs.valid or obs.calc_contour is None:
        empty = np.zeros((0, 2))
        return WorkPlanarResult(
            center_w_m=(float("nan"), float("nan")), plane_z_m=float(specs["plane_z"] if plane_z is None else plane_z),
            theta_deg=None, theta_valid=False, spec=UNRESOLVED, class_name=obs.class_name,
            measured_length_m=0.0, measured_width_m=0.0, aspect_ratio=0.0, area_w_m2=0.0,
            contour_w_m=empty, valid=False, reason=obs.reason or "OBSERVATION_INVALID", reason_owner="A",
            calibration_id=camera.calibration_id, camera_frame=camera.frame_id, compute_ms=0.0, **ids)

    cam = camera

    if str(getattr(obs.pixel_basis, "value", obs.pixel_basis)) == "undistorted" and np.any(camera.dist):
        cam = CameraProfile(camera.width, camera.height, camera.K, np.zeros(5), camera.frame_id,
                            camera.calibration_id)

    return compute_work_result(obs.calc_contour, cam, T_W_C, specs, class_name=obs.class_name,
                               plane_z=plane_z, observed_calibration_id=obs.calibration_id, **ids)


def to_output_dict(r: WorkPlanarResult, source_id: str) -> dict:
    """B 결과 → C·D로 내보내는 사전. 위치·길이 mm, 각도 deg (B가 한 번만 변환, DB-01)."""
    mm = lambda v: None if v is None or not math.isfinite(v) else round(v * 1000.0, 1)

    return {
        "source_id": source_id, "image_seq": r.image_seq, "part_track_id": r.part_track_id,
        "calibration_id": r.calibration_id, "spec": r.spec, "class_name": r.class_name,
        "valid": r.valid, "reason": r.reason, "reason_owner": r.reason_owner,
        "center_w_mm": [mm(r.center_w_m[0]), mm(r.center_w_m[1])],
        "theta_deg": None if r.theta_deg is None else round(r.theta_deg, 1),
        "theta_valid": r.theta_valid,
        "length_mm": mm(r.measured_length_m), "width_mm": mm(r.measured_width_m),
    }
