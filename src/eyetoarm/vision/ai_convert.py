"""AI 분할 모델 원시 출력 → 표시 결과 → 표시용 무효 관측 변환(YOLO 연결 T4). 순수 NumPy·OpenCV, torch 없음.

근거: 계획 `docs/superpowers/plans/2026-11-15-yolo-pipeline.md` r2 §2·§4.3·§4.3.1·§4.4, A 상세설계 §5.2(전체 마스크는 표시·후보
선택용, 계산용 윤곽 아님), §5.4(판정 사유), §7·§7.1(ROI → 추론, 원본 좌표 복원, 최고 신뢰도 후보 자동 선택 금지).

좌표 경로(설치된 ultralytics 8.4.211 원문 확인, 계획 §2)
- A는 ROI만 자른다(`crop_frame`, 원본 BGR 뷰·색 변환 없음). letterbox와 그 역변환은 라이브러리가 한다.
- `retina_masks=True`이면 마스크가 predict에 넘긴 crop 해상도로 나온다. 그래서 A는 **ROI 오프셋만** 더한다. 마스크 해상도가
  crop과 다르면 `RawSegOutput`·`convert`가 거부한다(이중 역변환 방지).
- 윤곽은 `Masks.xy`(구멍 제거·조각 연결선) 대신 이진 마스크에서 직접 `cv2.findContours(RETR_CCOMP, CHAIN_APPROX_NONE)`로
  구한다. 표시 다각형은 가장 큰 바깥 윤곽이고 조각·구멍 수를 함께 남긴다.
- 원점은 원본 영상 왼쪽 위 화소 인덱스(0,0)이고 점은 화소 인덱스 좌표(RAW, 왜곡 보정 없음)다. `pixel_map` 잠정 규약과 같으며
  B-01·B-02 확인 전이다.

관측(1단계 표시용, 계획 §4.4): 항상 `valid=False`. 관측면·계산 윤곽·선택·표시 마스크·관측 단위 class_name·confidence는 None이다.
인스턴스별 클래스·점수는 `AiSegResult`에 있다. reason 우선순위는 CALIBRATION_MISMATCH > TOP_FACE_UNRESOLVED(eligible이 있으면
개수와 무관, Q2 개발 구현안) > CONTOUR_CLIPPED(거부 없이 잘림) > NO_OBJECT.

볼트 축 끝점 opt-in(#168, 설정 `axis_extractor="m1"`, 기본 꺼짐). 근거: `_workspace/168_a_to_b_real_path_implementation_spec.md`,
#163 명세 r1 §3~§5(`_workspace/163_builder_axis_endpoint_b_input_spec.md`), #167 계획 r2 K0 사용자 결정.
- 끝점은 `convert` 안에서 한 번만 계산한다(#163 §5 위치). M1 입력은 **표시 polygon(`AiInstance.polygon`)의 래스터**다(#168 명세 §1,
  #167 r2 W1). 래스터 규약(`rasterize_polygon`): polygon에서 ROI 오프셋을 빼 crop 좌표 정수 꼭짓점(화소 중심, `pixel_map` 잠정 규약)으로
  두고 `cv2.drawContours(..., thickness=FILLED)`로 경계 화소를 포함해 채운다. polygon은 가장 큰 바깥 윤곽 하나라 조각·구멍이 없으므로,
  위상 판정은 같은 마스크에서 잰 `component_count`·`hole_count`(RETR_CCOMP, M1 내부 판정과 같은 기준)로 M1과 같은 순서에서 먼저 한다:
  화소 수(`area_px`) ≥ M1 `min_pixels`이면 조각 2개 이상 → MASK_FRAGMENTED, 구멍 1개 이상 → MASK_HOLE. 볼트 겹침 IoU도 래스터로 잰다.
  대상: 볼트 클래스, 거부 없음, 잘리지 않음. 끝점에는 ROI 오프셋만 더한다(원본 프레임 픽셀, RAW). 마스크는 결과·관측에 남기지 않는다.
- 관측(`observation_from_ai`)의 사유 이름과 순서는 #163 §4 표를 따른다(#174부터 프레임 전체가 아니라 물체별로 적용, 아래 단계 T): CALIBRATION_MISMATCH > 비거부 볼트 0개면 기존 표시 사유 >
  CONTOUR_CLIPPED > MULTIPLE_CANDIDATES(볼트 마스크 IoU ≥ 0.5) > MULTIPLE_OBJECTS(유효 후보 2개 이상) >
  추출기 사유(MASK_FRAGMENTED·MASK_HOLE·AXIS_HEAD_AMBIGUOUS·AXIS_ORDER_INCONSISTENT·AXIS_VIEW_OBLIQUE·ENDPOINTS_INVALID 등) >
  AXIS_EXTRACTION_ERROR. 남은 경우(유일한 볼트의 추출 성공)만 `valid=True, observed_face="axis"`이다.
- **#174 전환 단계 T(물체별 후보 자격)**: 근거 `_workspace/174_builder_a_multi_object_candidate_plan.md` r2 §5(r1 §2).
  위 표 순서를 프레임 전체가 아니라 볼트마다 적용한다(`ai_result.axis_candidate_reason` 하나로 판정). 잘림·겹침 당사자(IoU ≥ 0.5 쌍의
  양쪽)·추출 실패 볼트만 빠지고 떨어진 독립 볼트는 남는다. 보정 불일치는 지금처럼 프레임 단위로 먼저 막는다.
  - 유효 후보 1개: 다른 물체가 보여도 그 볼트로 기존 단일 관측(`selected_index` = 원 index)을 만든다.
  - 2개 이상: `MULTIPLE_OBJECTS` 무효(최종 목록 형식 B-09 합의 전이라 B를 부르지 않고 임의로 고르지 않는다). 후보는 기록에 남는다.
  - 0개: 대표 사유 하나를 무효 사유로 쓴다(CONTOUR_CLIPPED > MULTIPLE_CANDIDATES > 가장 작은 index의 추출기 사유). **대표 사유는
    진단·표시용**이며 유효 판정·B 호출은 후보별 자격으로만 정한다. 후보별 사유는 인스턴스 기록에 있다.
- 이 `valid=True`는 **추출기 형식 조건 통과** 뜻이다. 정확도·운영 승인이 아니다(K0). 무효를 유효로 바꾸거나 polygon·이전 결과로
  대체하지 않는다. `#163`의 `AXIS_PROFILE_MISMATCH`는 M1에 구현되어 있지 않아 쓰지 않는다. 너트는 기존 사유(윗면 미구현) 그대로다.
"""
from __future__ import annotations

import dataclasses
import time

import cv2
import numpy as np

from eyetoarm.common.types import FramePacket, PixelBasis, PlanarObservation
from eyetoarm.common.types import OBSERVED_FACE_AXIS
from eyetoarm.vision.ai_config import AiSegConfig
from eyetoarm.vision.axis_endpoints import MODEL_DERIVED, extract_m1
from eyetoarm.vision.ai_result import (AXIS_EXTRACTION_ERROR, CONTOUR_CLIPPED, EMPTY_MASK, LOW_CONFIDENCE,
                                       MULTIPLE_CANDIDATES, NON_TARGET_CLASS, SMALL_MASK, AiInstance, AiJobConfigError,
                                       AiSegResult, RawSegOutput, axis_candidate_reason)
from eyetoarm.vision.pipeline_types import JobContext, SourceProfile
from eyetoarm.vision.pixel_map import validate_roi
from eyetoarm.vision.planar import _check_frame

CALIBRATION_MISMATCH = "CALIBRATION_MISMATCH"
TOP_FACE_UNRESOLVED = "TOP_FACE_UNRESOLVED"
NO_OBJECT = "NO_OBJECT"

# 축 끝점 경로 프레임 사유(#163 §4 표)
# (CONTOUR_CLIPPED·MULTIPLE_CANDIDATES·AXIS_EXTRACTION_ERROR는 물체별 자격 함수와 함께 ai_result에 있다)
MULTIPLE_OBJECTS = "MULTIPLE_OBJECTS"
AXIS_OVERLAP_IOU = 0.5   # #163 §4 「볼트 마스크끼리 IoU ≥ 0.5」(명세 상수, #159 결과로 정하지 않음)

# 기록에 남길 추출기 진단 키(수·문자열만)
_DIAG_KEYS = ("pixels", "components", "holes", "width_ratio", "head_step_frac", "tilt_deg", "class_est",
              "total_length_est_mm", "scale_px_per_mm")


def _is_bolt(name: str) -> bool:
    """볼트 클래스 이름인가(공통형 A-8 규칙과 같음: 소문자화 후 `bolt`로 시작)."""
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
    """표시 polygon(원본 프레임 픽셀) → crop 크기 bool 마스크(#168 래스터 규약, 모듈 설명).

    꼭짓점은 정수 화소 중심이어야 한다(convert가 정수 윤곽에 정수 ROI 오프셋을 더해 만든 값). 정수가 아니면 ValueError.
    """
    local = np.asarray(polygon_px, dtype=np.float64) - np.asarray(offset, dtype=np.float64)
    points = np.rint(local)

    if not np.array_equal(points, local):
        raise ValueError("polygon vertices must be integer pixel centers for rasterization")

    w, h = crop_wh
    canvas = np.zeros((h, w), np.uint8)
    cv2.drawContours(canvas, [points.astype(np.int32).reshape(-1, 1, 2)], -1, 1, thickness=cv2.FILLED)

    return canvas.astype(bool)


def _extract_axis(inst: AiInstance, offset: np.ndarray, crop_wh: tuple[int, int]):
    """비거부·비잘림 볼트 인스턴스 하나 → (래스터 마스크 또는 None, 끝점 또는 None, 사유 또는 None, 진단).

    위상(조각·구멍)은 인스턴스 계수로 M1과 같은 순서에서 먼저 판정한다. 추출기·래스터 예외는 AXIS_EXTRACTION_ERROR로 막는다.
    """
    topo = {"pixels": int(inst.area_px), "components": inst.component_count, "holes": inst.hole_count}

    if inst.area_px >= MODEL_DERIVED.min_pixels:
        if inst.component_count > 1:
            return None, None, "MASK_FRAGMENTED", topo

        if inst.hole_count > 0:
            return None, None, "MASK_HOLE", topo

    try:
        mask = rasterize_polygon(inst.polygon, offset, crop_wh)
        est = extract_m1(mask)
    except Exception as exc:  # noqa: BLE001 - 관측 예외로 새지 않게(#163 §4)
        return None, None, AXIS_EXTRACTION_ERROR, {"error": f"{type(exc).__name__}: {exc}"[:200]}

    diag = {**_diag(est.diag), **topo}

    if not est.valid:
        return mask, None, est.reason or AXIS_EXTRACTION_ERROR, diag

    return mask, est.endpoints_px + offset.astype(np.float64), None, diag


def overlaps(iou: float) -> bool:
    """겹침 당사자 판정 술어(#163 §4 「IoU ≥ 0.5」, 경계 0.5 포함)."""
    return iou >= AXIS_OVERLAP_IOU


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """두 bool 마스크의 IoU."""
    union = np.count_nonzero(a | b)

    return 0.0 if union == 0 else float(np.count_nonzero(a & b)) / union


def crop_frame(frame: FramePacket, config: AiSegConfig) -> tuple[np.ndarray, tuple[int, int]]:
    """프레임 검사(BGR uint8 (H,W,3), 크기 메타데이터 일치) 뒤 설정 ROI로 자른 뷰와 (x, y) 오프셋을 돌려준다.

    잘못된 프레임은 ValueError(INFERENCE_ERROR로 정리). ROI가 프레임 밖이면 AiJobConfigError(작업 설정 오류).
    영상을 복사·변환하지 않으며 쓰지 않는다.
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
    """이진 마스크의 (가장 큰 바깥 윤곽 또는 None, 바깥 윤곽 수, 구멍 수). 점 3개 미만·면적 0이면 윤곽 None."""
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

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
    """crop 좌표 다각형이 crop 경계(=ROI 경계, ROI 없으면 원본 경계)에서 margin 화소 안에 닿는가.

    crop은 원본 안에 있으므로 원본 경계에 닿는 점은 crop 경계에도 닿는다. 그래서 crop 경계만 보면 둘 다 포함된다.
    """
    w, h = crop_wh
    xs, ys = points[:, 0], points[:, 1]

    return bool(xs.min() <= margin or ys.min() <= margin or xs.max() >= w - 1 - margin or ys.max() >= h - 1 - margin)


def convert(raw: RawSegOutput, frame: FramePacket, config: AiSegConfig, *,
            started_ns: int | None = None) -> AiSegResult:
    """원시 출력을 원본 픽셀 좌표의 표시 결과로 바꾼다(계획 §4.3 5단계, 판정 순서 고정).

    거부는 처음 걸린 하나: 비대상 → 점수 미달 → 기하 없음(EMPTY) → 작은 마스크(SMALL). 기하(polygon·clipped)는 거부와 따로
    먼저 계산한다. class_map 밖 클래스는 AiJobConfigError, crop 크기 불일치는 ValueError. `started_ns`(어댑터 시작,
    `time.perf_counter_ns`)가 있으면 process_ms를 그때부터 잰다.
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
    rasters: dict[int, np.ndarray] = {}   # 축 경로 볼트의 polygon 래스터(겹침 IoU용, 결과에 남기지 않음)

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

        # 축 끝점(#168 opt-in): 비거부 볼트만, 입력은 표시 polygon 래스터. 잘린 볼트는 추출하지 않고 CONTOUR_CLIPPED
        if config.axis_extractor is not None and rejected is None and _is_bolt(name):
            if clipped:
                inst = dataclasses.replace(inst, axis_reason=CONTOUR_CLIPPED)
            else:
                raster, endpoints, reason, diag = _extract_axis(inst, offset, crop_wh)
                inst = dataclasses.replace(inst, axis_endpoints_px=endpoints, axis_reason=reason, axis_diag=diag)

                if raster is not None:
                    rasters[i] = raster

        instances.append(inst)

    # 볼트끼리 겹침(#163 §4 MULTIPLE_CANDIDATES): 비거부 볼트 polygon 래스터 쌍의 최대 IoU(래스터가 없는 볼트는 polygon을 새로 그림).
    # #174: IoU ≥ AXIS_OVERLAP_IOU인 쌍은 양쪽 모두 상대 index를 `axis_overlap_with`에 남긴다(겹침 당사자). 잘린 볼트도 쌍에 든다
    if config.axis_extractor is not None:
        bolts = [k for k, inst in enumerate(instances) if inst.axis_evaluated]
        best = {k: 0.0 for k in bolts}
        partners: dict[int, list[int]] = {k: [] for k in bolts}
        shapes = {k: rasters[k] if k in rasters else rasterize_polygon(instances[k].polygon, offset, crop_wh)
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
    """표시용 무효 사유(계획 §4.4 우선순위). MULTIPLE_OBJECTS는 1단계에서 쓰지 않는다(Q2 개발 구현안)."""
    if profile is None or not profile.applies_to(frame):
        return CALIBRATION_MISMATCH

    if result.eligible_count > 0:
        return TOP_FACE_UNRESOLVED

    # 클래스·점수·기하·면적을 모두 통과했지만 잘린 대상만 CONTOUR_CLIPPED(작은 잘림 잡음은 제외, r1 C1)
    if any(i.rejected is None and i.clipped for i in result.instances):
        return CONTOUR_CLIPPED

    return NO_OBJECT


def axis_selection(result: AiSegResult, frame: FramePacket, profile: SourceProfile | None):
    """축 끝점 경로의 프레임 판정(#163 §4 표 순서). (선택 인스턴스 또는 None, 무효 사유 또는 None)."""
    if profile is None or not profile.applies_to(frame):
        return None, CALIBRATION_MISMATCH

    bolts = [i for i in result.instances if i.axis_evaluated]

    if not bolts:
        return None, display_reason(result, frame, profile)

    # 물체별 자격(#174): 집계(`result.axis_valid_indices`)와 같은 함수 결과만 쓴다
    reasons = [(i, axis_candidate_reason(i)) for i in bolts]
    valid = [i for i, r in reasons if r is None]

    if len(valid) == 1:
        return valid[0], None

    if len(valid) >= 2:
        return None, MULTIPLE_OBJECTS   # B-09 합의 전: 기록만, 임의 선택·B 호출 없음

    # 유효 0개: 대표 사유(진단용). 표 순서 앞선 것, 추출기 사유끼리는 가장 작은 index
    found = [r for _, r in reasons]

    for first in (CONTOUR_CLIPPED, MULTIPLE_CANDIDATES):
        if first in found:
            return None, first

    return None, found[0]


def observation_from_ai(job: JobContext, result: AiSegResult) -> PlanarObservation:
    """작업과 AI 표시 결과로 관측을 만든다. 결과 설정 해시가 작업 시작 설정과 다르면 AiJobConfigError.

    `axis_extractor`가 꺼진 설정은 표시용 무효 관측(기존과 같음). 켠 설정은 `axis_selection`으로 유일한 볼트의 추출기 출력만
    계약 관측(`observed_face="axis"`)으로 만들고, 나머지는 첫 사유를 실은 무효 관측이다.
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


__all__ = ["AXIS_EXTRACTION_ERROR", "AXIS_OVERLAP_IOU", "MULTIPLE_CANDIDATES", "MULTIPLE_OBJECTS", "axis_selection", "convert", "overlaps",
           "crop_frame", "display_reason", "observation_from_ai", "rasterize_polygon"]
