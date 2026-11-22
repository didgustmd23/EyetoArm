"""AI 분할 모델 원시 출력을 원본 픽셀 좌표의 표시 결과(`AiSegResult`)와 관측·단계 L 목록으로 바꾼다.

A는 ROI만 자르고 ROI 오프셋만 더한다(`retina_masks=True`, letterbox 역변환은 라이브러리). `axis_extractor`를 켜면
볼트마다 표시 polygon 래스터로 축 끝점을 구하고 물체별 자격으로 유효 후보를 정한다. 무효를 유효로 바꾸지 않는다.
"""
from __future__ import annotations

import dataclasses
import time

import cv2
import numpy as np

from eyetoarm.common.types import FramePacket, PixelBasis, PlanarObservation
from eyetoarm.common.types import (FRAME_OBSERVATIONS_VERSION, OBSERVED_FACE_AXIS, CaptureInfo, CaptureTf,
                                   FrameObservations, ObjectObservation, owned_axis_endpoints, raw_points_in_frame)
from eyetoarm.vision.ai_config import AiSegConfig
from eyetoarm.vision.axis_endpoints import MODEL_DERIVED, WINDOW_MARGIN_PX, extract_m1, extract_m2_window, mask_window
from eyetoarm.vision.ai_result import (AXIS_EXTRACTION_ERROR, CONTOUR_CLIPPED, EMPTY_MASK, LOW_CONFIDENCE,
                                       MULTIPLE_CANDIDATES, NON_TARGET_CLASS, SMALL_MASK, AiInstance, AiJobConfigError,
                                       AiSegResult, RawSegOutput, axis_candidate_reason)
from eyetoarm.vision.pipeline_types import JobContext, SourceProfile
from eyetoarm.vision.pixel_map import validate_roi
from eyetoarm.vision.planar import _check_frame

CALIBRATION_MISMATCH = "CALIBRATION_MISMATCH"
TOP_FACE_UNRESOLVED = "TOP_FACE_UNRESOLVED"
NO_OBJECT = "NO_OBJECT"

# 나머지 축 경로 사유(CONTOUR_CLIPPED 등)는 물체별 자격 함수와 함께 ai_result에 있다
MULTIPLE_OBJECTS = "MULTIPLE_OBJECTS"
AXIS_OVERLAP_IOU = 0.5   # 볼트 마스크끼리 IoU ≥ 0.5면 겹침(명세 상수)

# 기록에 남길 추출기 진단 키(수·문자열만)
_DIAG_KEYS = ("pixels", "components", "holes", "width_ratio", "head_step_frac", "tilt_deg",
              # M2만 내는 값(M1 기록에는 없음)
              "vote_width_ratio", "head_vote", "order_gate", "centerline", "tilt_used_deg")


def _is_bolt(name: str) -> bool:
    """볼트 클래스 이름인가(공통형 규칙: 소문자화 후 `bolt`로 시작)."""
    return name.strip().lower().startswith("bolt")


def _diag(diag: dict) -> dict:
    """추출기 진단에서 기록용 수·문자열만 고른다."""
    out = {}

    for key in _DIAG_KEYS:
        value = diag.get(key)

        if isinstance(value, str) or (isinstance(value, (int, float, np.integer, np.floating))
                                      and not isinstance(value, bool) and np.isfinite(value)):
            out[key] = value.item() if isinstance(value, np.generic) else value

    return out


def rasterize_polygon(polygon_px: np.ndarray, offset: np.ndarray, crop_wh: tuple[int, int]) -> np.ndarray:
    """표시 polygon(원본 프레임 픽셀)을 경계 화소까지 채운 crop 크기 bool 마스크로 그린다.

    꼭짓점은 정수 화소 중심이어야 하며 아니면 ValueError.
    """
    local = np.asarray(polygon_px, dtype=np.float64) - np.asarray(offset, dtype=np.float64)
    points = np.rint(local)

    if not np.array_equal(points, local):
        raise ValueError("polygon vertices must be integer pixel centers for rasterization")

    w, h = crop_wh
    canvas = np.zeros((h, w), np.uint8)
    cv2.drawContours(canvas, [points.astype(np.int32).reshape(-1, 1, 2)], -1, 1, thickness=cv2.FILLED)

    return canvas.astype(bool)


@dataclasses.dataclass(frozen=True)
class _Window:
    """crop 좌표 (x0, y0)에 놓인 bool 창. 창 밖 crop 화소는 모두 0이다(겹침 IoU·M2 입력용)."""

    mask: np.ndarray
    x0: int
    y0: int


def rasterize_polygon_window(polygon_px: np.ndarray, offset: np.ndarray, crop_wh: tuple[int, int]) -> _Window:
    """`rasterize_polygon`과 같은 래스터를 꼭짓점 범위(+`WINDOW_MARGIN_PX`, crop 안으로 자름) 창에만 그린다.

    채운 다각형은 꼭짓점 범위 밖에 화소를 두지 않으므로 창 밖 crop 화소는 0이고, 정수 평행 이동이라 창 안
    화소는 전체 래스터와 같다. 꼭짓점은 정수 화소 중심이어야 하며 아니면 ValueError.
    """
    local = np.asarray(polygon_px, dtype=np.float64) - np.asarray(offset, dtype=np.float64)
    points = np.rint(local)

    if not np.array_equal(points, local):
        raise ValueError("polygon vertices must be integer pixel centers for rasterization")

    w, h = crop_wh
    ints = points.astype(np.int64).reshape(-1, 2)
    lo, hi = ints.min(axis=0) - WINDOW_MARGIN_PX, ints.max(axis=0) + WINDOW_MARGIN_PX
    x0, y0 = min(max(int(lo[0]), 0), w - 1), min(max(int(lo[1]), 0), h - 1)
    x1, y1 = max(min(int(hi[0]), w - 1), x0), max(min(int(hi[1]), h - 1), y0)
    canvas = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    shifted = (ints - np.array([x0, y0])).astype(np.int32).reshape(-1, 1, 2)
    cv2.drawContours(canvas, [shifted], -1, 1, thickness=cv2.FILLED)

    return _Window(canvas.astype(bool), x0, y0)


def _m1_on_window(win: _Window, crop_wh: tuple[int, int]):
    """M1은 전체 crop 마스크로 계산한다(이번 창 계산 범위 밖). 창을 crop 크기로 펼쳐 넣는다."""
    w, h = crop_wh
    full = np.zeros((h, w), bool)
    sh, sw = win.mask.shape
    full[win.y0:win.y0 + sh, win.x0:win.x0 + sw] = win.mask

    return extract_m1(full)


def _m2_on_window(win: _Window, crop_wh: tuple[int, int]):
    """M2는 창 그대로 계산한다(결과는 전체 crop 마스크와 같음)."""
    return extract_m2_window(win.mask, (win.x0, win.y0), (crop_wh[1], crop_wh[0]))


def _extract_axis(inst: AiInstance, offset: np.ndarray, crop_wh: tuple[int, int], extractor: str = "m1"):
    """비거부·비잘림 볼트 하나 → (래스터 창, 끝점, 사유, 진단). 없는 값은 None.

    위상(조각·구멍)을 먼저 판정하고, 추출기·래스터 예외는 AXIS_EXTRACTION_ERROR로 막는다.
    """
    # 호출 때 모듈 이름으로 찾는다(시험의 mock.patch가 닿도록: M1은 extract_m1, M2는 extract_m2_window)
    extract = {"m1": _m1_on_window, "m2": _m2_on_window}[extractor]
    topo = {"pixels": int(inst.area_px), "components": inst.component_count, "holes": inst.hole_count}

    if inst.area_px >= MODEL_DERIVED.min_pixels:
        if inst.component_count > 1:
            return None, None, "MASK_FRAGMENTED", topo

        if inst.hole_count > 0:
            return None, None, "MASK_HOLE", topo

    try:
        mask = rasterize_polygon_window(inst.polygon, offset, crop_wh)
        est = extract(mask, crop_wh)
    except Exception as exc:  # noqa: BLE001 - 관측 예외로 새지 않게
        return None, None, AXIS_EXTRACTION_ERROR, {"error": f"{type(exc).__name__}: {exc}"[:200]}

    diag = {**_diag(est.diag), **topo}

    if not est.valid:
        return mask, None, est.reason or AXIS_EXTRACTION_ERROR, diag

    return mask, est.endpoints_px + offset.astype(np.float64), None, diag


def _checked_endpoints(endpoints, reason, diag, frame: FramePacket):
    """추출기 끝점을 공통형 규칙과 RAW 프레임 범위로 검사한다.

    어긋나면 그 물체만 `AXIS_EXTRACTION_ERROR`로 바꿔, 관측·목록 생성에서 프레임 전체를 잃지 않게 한다.
    """
    if endpoints is None:
        return endpoints, reason, diag

    try:
        owned = owned_axis_endpoints(endpoints)
    except ValueError as exc:
        return None, AXIS_EXTRACTION_ERROR, {**diag, "endpoints_rejected": str(exc)[:200]}

    if not raw_points_in_frame(owned, frame.width, frame.height):
        return None, AXIS_EXTRACTION_ERROR, {**diag, "endpoints_rejected": f"out of frame {(frame.width, frame.height)}"}

    return owned, reason, diag


def overlaps(iou: float) -> bool:
    """겹침 당사자 판정 술어(IoU ≥ 0.5, 경계 포함)."""
    return iou >= AXIS_OVERLAP_IOU


def _iou(a: _Window, b: _Window) -> float:
    """두 창 마스크의 IoU(같은 crop 좌표계). 창 밖은 0이라 겹치는 사각형에서만 교집합을 센다.

    합집합 = |a| + |b| − |a∩b|라 전체 crop 마스크로 센 정수와 같고, 나눗셈도 같다.
    """
    ah, aw = a.mask.shape
    bh, bw = b.mask.shape
    x0, y0 = max(a.x0, b.x0), max(a.y0, b.y0)
    x1, y1 = min(a.x0 + aw, b.x0 + bw), min(a.y0 + ah, b.y0 + bh)
    inter = 0

    if x0 < x1 and y0 < y1:
        inter = int(np.count_nonzero(a.mask[y0 - a.y0:y1 - a.y0, x0 - a.x0:x1 - a.x0]
                                     & b.mask[y0 - b.y0:y1 - b.y0, x0 - b.x0:x1 - b.x0]))

    union = int(np.count_nonzero(a.mask)) + int(np.count_nonzero(b.mask)) - inter

    return 0.0 if union == 0 else float(inter) / union


def crop_frame(frame: FramePacket, config: AiSegConfig) -> tuple[np.ndarray, tuple[int, int]]:
    """프레임을 검사하고 설정 ROI로 자른 뷰(복사·변환 없음)와 (x, y) 오프셋을 돌려준다.

    잘못된 프레임은 ValueError, ROI가 프레임 밖이면 AiJobConfigError.
    """
    img = _check_frame(frame)
    h, w = img.shape[:2]

    if config.roi is None:
        return img, (0, 0)

    try:
        validate_roi(config.roi, (w, h))
    except ValueError as exc:
        raise AiJobConfigError(f"roi does not fit frame {w}x{h}: {exc}") from exc

    x, y, rw, rh = config.roi

    return img[y:y + rh, x:x + rw], (x, y)


def _geometry(mask: np.ndarray):
    """이진 마스크의 (가장 큰 바깥 윤곽 또는 None, 바깥 윤곽 수, 구멍 수). 점 3개 미만·면적 0이면 윤곽 None.

    윤곽은 전경 범위 창(`mask_window`, 사방 여유 1 px)에서 찾고 창 원점을 더한다. 프레임 경계가 아닌 창
    가장자리가 0이라 윤곽 점·순서·계층이 전체 마스크와 같다.
    """
    sub, (x0, y0) = mask_window(mask)
    contours, hierarchy = cv2.findContours(sub.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE,
                                           offset=(x0, y0))

    if not contours:
        return None, 0, 0

    parents = hierarchy[0][:, 3]
    outer = [c for c, parent in zip(contours, parents) if parent == -1]
    holes = len(contours) - len(outer)
    best = max(outer, key=cv2.contourArea)

    if len(best) < 3 or cv2.contourArea(best) <= 0.0:
        return None, len(outer), holes

    return best.reshape(-1, 2), len(outer), holes


def _clipped(points: np.ndarray, crop_wh: tuple[int, int], margin: int) -> bool:
    """crop 좌표 다각형이 crop 경계에서 margin 화소 안에 닿는가(원본 경계 닿음도 포함된다)."""
    w, h = crop_wh
    xs, ys = points[:, 0], points[:, 1]

    return bool(xs.min() <= margin or ys.min() <= margin or xs.max() >= w - 1 - margin or ys.max() >= h - 1 - margin)


def convert(raw: RawSegOutput, frame: FramePacket, config: AiSegConfig, *,
            started_ns: int | None = None) -> AiSegResult:
    """원시 출력을 원본 픽셀 좌표의 표시 결과로 바꾼다.

    거부는 처음 걸린 하나(비대상 → 점수 미달 → EMPTY → SMALL). class_map 밖 클래스는 AiJobConfigError,
    crop 크기 불일치는 ValueError. `started_ns`가 있으면 process_ms를 그때부터 잰다.
    """
    t0 = time.perf_counter_ns()
    crop, (ox, oy) = crop_frame(frame, config)
    crop_wh = (crop.shape[1], crop.shape[0])

    if raw.crop_wh != crop_wh:
        raise ValueError(f"raw output crop {raw.crop_wh} != actual crop {crop_wh} (retina contract)")

    names = config.class_names
    targets = set(config.target_classes)
    offset = np.array([ox, oy], np.float32)
    instances = []
    rasters: dict[int, _Window] = {}   # 겹침 IoU용, 결과에 남기지 않음

    for i in range(raw.count):
        cls = int(raw.class_index[i])

        if cls not in names:
            raise AiJobConfigError(f"model class index {cls} is not in class_map {sorted(names)}")

        name = names[cls]
        conf = float(raw.confidence[i])
        mask = raw.masks[i]
        points, components, holes = _geometry(mask)
        area = float(np.count_nonzero(mask))
        clipped = points is not None and _clipped(points, crop_wh, config.border_margin_px)

        if name not in targets:
            rejected = NON_TARGET_CLASS
        elif conf < config.conf_threshold:
            rejected = LOW_CONFIDENCE
        elif points is None:
            rejected = EMPTY_MASK
        elif area < config.min_mask_area_px:
            rejected = SMALL_MASK
        else:
            rejected = None

        x1, y1, x2, y2 = (float(v) for v in raw.boxes_xyxy[i])

        inst = AiInstance(
            index=i, class_index=cls, class_name=name, confidence=conf,
            box_xyxy=(x1 + ox, y1 + oy, x2 + ox, y2 + oy),
            polygon=None if points is None else points.astype(np.float32) + offset,
            area_px=area, component_count=components, hole_count=holes, clipped=clipped, rejected=rejected)

        # 잘린 볼트는 추출하지 않고 CONTOUR_CLIPPED
        if config.axis_extractor is not None and rejected is None and _is_bolt(name):
            if clipped:
                inst = dataclasses.replace(inst, axis_reason=CONTOUR_CLIPPED)
            else:
                raster, endpoints, reason, diag = _extract_axis(inst, offset, crop_wh, config.axis_extractor)
                endpoints, reason, diag = _checked_endpoints(endpoints, reason, diag, frame)
                inst = dataclasses.replace(inst, axis_endpoints_px=endpoints, axis_reason=reason, axis_diag=diag)

                if raster is not None:
                    rasters[i] = raster

        instances.append(inst)

    # 겹침 쌍은 양쪽 모두 상대 index를 남긴다. 잘린 볼트도 쌍에 든다
    if config.axis_extractor is not None:
        bolts = [k for k, inst in enumerate(instances) if inst.axis_evaluated]
        best = {k: 0.0 for k in bolts}
        partners: dict[int, list[int]] = {k: [] for k in bolts}
        shapes = {k: rasters[k] if k in rasters else rasterize_polygon_window(instances[k].polygon, offset, crop_wh)
                  for k in bolts}

        for a_pos, a in enumerate(bolts):
            for b in bolts[a_pos + 1:]:
                iou = _iou(shapes[a], shapes[b])
                best[a], best[b] = max(best[a], iou), max(best[b], iou)

                if overlaps(iou):
                    partners[a].append(b)
                    partners[b].append(a)

        for k in bolts:
            instances[k] = dataclasses.replace(instances[k], axis_max_iou=best[k],
                                               axis_overlap_with=tuple(sorted(partners[k])) or None)

    t1 = time.perf_counter_ns()
    post_ms = (t1 - t0) / 1e6
    process_ms = (t1 - started_ns) / 1e6 if started_ns is not None else raw.infer_ms + post_ms

    return AiSegResult(instances=tuple(instances), model_id=config.model_id, config_sha256=config.config_sha256,
                       roi=config.roi, crop_wh=crop_wh, infer_ms=raw.infer_ms, post_ms=post_ms,
                       process_ms=max(process_ms, raw.infer_ms + post_ms))


def display_reason(result: AiSegResult, frame: FramePacket, profile: SourceProfile | None) -> str:
    """표시용 무효 사유(보정 불일치 > 윗면 미해결 > 잘림 > 물체 없음)."""
    if profile is None or not profile.applies_to(frame):
        return CALIBRATION_MISMATCH

    if result.eligible_count > 0:
        return TOP_FACE_UNRESOLVED

    # 거부 없이 잘린 대상만 CONTOUR_CLIPPED(작은 잘림 잡음은 제외)
    if any(i.rejected is None and i.clipped for i in result.instances):
        return CONTOUR_CLIPPED

    return NO_OBJECT


def axis_selection(result: AiSegResult, frame: FramePacket, profile: SourceProfile | None):
    """축 끝점 경로의 프레임 판정. (선택 인스턴스 또는 None, 무효 사유 또는 None)."""
    if profile is None or not profile.applies_to(frame):
        return None, CALIBRATION_MISMATCH

    bolts = [i for i in result.instances if i.axis_evaluated]

    if not bolts:
        return None, display_reason(result, frame, profile)

    # 집계(`result.axis_valid_indices`)와 같은 자격 함수 결과만 쓴다
    reasons = [(i, axis_candidate_reason(i)) for i in bolts]
    valid = [i for i, r in reasons if r is None]

    if len(valid) == 1:
        return valid[0], None

    if len(valid) >= 2:
        return None, MULTIPLE_OBJECTS   # 임의로 고르지 않는다

    # 유효 0개: 대표 사유(진단용). 표 순서 앞선 것, 추출기 사유끼리는 가장 작은 index
    found = [r for _, r in reasons]

    for first in (CONTOUR_CLIPPED, MULTIPLE_CANDIDATES):
        if first in found:
            return None, first

    return None, found[0]


def observation_from_ai(job: JobContext, result: AiSegResult) -> PlanarObservation:
    """작업과 AI 표시 결과로 관측을 만든다. 설정 해시가 작업 시작 설정과 다르면 AiJobConfigError.

    `axis_extractor`가 꺼져 있으면 표시용 무효 관측, 켜져 있으면 유일한 유효 볼트만 `axis` 관측이 된다.
    """
    if result.config_sha256 != job.detector_config_id:
        raise AiJobConfigError(f"display result config {result.config_sha256[:12]} != job start config "
                               f"{job.detector_config_id[:12]}")

    f = job.frame

    if getattr(job.settings.detector_config, "axis_extractor", None) is not None:
        chosen, reason = axis_selection(result, f, job.settings.profile)
        valid = chosen is not None

        return PlanarObservation(
            run_id=f.run_id, source_id=f.source_id, image_seq=f.image_seq,
            capture_generation=f.capture_generation, generation=job.generation, job_seq=job.job_seq,
            detector_config_id=job.detector_config_id, pixel_basis=PixelBasis.RAW,
            calibration_id=f.calibration_id, observed_face=OBSERVED_FACE_AXIS if valid else None, calc_contour=None,
            display_mask_contour=None, candidate_count=result.candidate_count,
            selected_index=chosen.index if valid else None, valid=valid, reason=reason,
            process_ms=result.process_ms, method="ai", class_name=chosen.class_name if valid else None,
            confidence=chosen.confidence if valid else None, model_id=result.model_id,
            axis_endpoints_px=chosen.axis_endpoints_px if valid else None)

    return PlanarObservation(
        run_id=f.run_id, source_id=f.source_id, image_seq=f.image_seq,
        capture_generation=f.capture_generation, generation=job.generation, job_seq=job.job_seq,
        detector_config_id=job.detector_config_id, pixel_basis=PixelBasis.RAW,
        calibration_id=f.calibration_id, observed_face=None, calc_contour=None, display_mask_contour=None,
        candidate_count=result.candidate_count, selected_index=None, valid=False,
        reason=display_reason(result, f, job.settings.profile), process_ms=result.process_ms,
        method="ai", class_name=None, confidence=None, model_id=result.model_id)


def list_candidate_indices(result: AiSegResult, frame: FramePacket, profile: SourceProfile | None) -> tuple[int, ...]:
    """단계 L 목록에 실을 원 인덱스. 보정 불일치로 프레임이 막히면 빈 tuple."""
    if profile is None or not profile.applies_to(frame):
        return ()

    return result.axis_valid_indices


def frame_observations_from_ai(job: JobContext, result: AiSegResult, *, capture: CaptureInfo, camera,
                               tf: CaptureTf | None, world_frame: str = "world") -> FrameObservations | None:
    """작업과 AI 표시 결과로 단계 L 목록을 만든다. 후보 0개면 None(B 계산 호출 생략).

    설정 해시 불일치는 `AiJobConfigError`, TF 부모가 `world_frame`이 아니면 ValueError.
    """
    if result.config_sha256 != job.detector_config_id:
        raise AiJobConfigError(f"display result config {result.config_sha256[:12]} != job start config "
                               f"{job.detector_config_id[:12]}")

    indices = list_candidate_indices(result, job.frame, job.settings.profile)

    if not indices:
        return None

    candidates = tuple(
        ObjectObservation(instance_index=inst.index, class_name=inst.class_name, confidence=inst.confidence,
                          pixel_basis=PixelBasis.RAW, observed_face=OBSERVED_FACE_AXIS,
                          axis_endpoints_px=inst.axis_endpoints_px, calc_contour=None)
        for inst in (result.instances[i] for i in indices))

    observations = FrameObservations(version=FRAME_OBSERVATIONS_VERSION, frame=job.frame, capture=capture,
                                     camera=camera, tf=tf, generation=job.generation, job_seq=job.job_seq,
                                     detector_config_id=job.detector_config_id, candidates=candidates)

    # 생성자가 TF 형·시각·광학 프레임을 먼저 검사한 뒤 부모(world) 이름을 대조한다
    if observations.tf is not None and observations.tf.parent != world_frame:
        raise ValueError(f"tf.parent {observations.tf.parent!r} != world frame {world_frame!r}")

    return observations


__all__ = ["AXIS_EXTRACTION_ERROR", "AXIS_OVERLAP_IOU", "MULTIPLE_CANDIDATES", "MULTIPLE_OBJECTS", "axis_selection", "convert", "overlaps",
           "crop_frame", "display_reason", "frame_observations_from_ai", "list_candidate_indices", "observation_from_ai",
           "rasterize_polygon"]
