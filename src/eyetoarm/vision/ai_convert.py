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
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from eyetoarm.common.types import FramePacket, PixelBasis, PlanarObservation
from eyetoarm.vision.ai_config import AiSegConfig
from eyetoarm.vision.ai_result import (EMPTY_MASK, LOW_CONFIDENCE, NON_TARGET_CLASS, SMALL_MASK, AiInstance,
                                       AiJobConfigError, AiSegResult, RawSegOutput)
from eyetoarm.vision.pipeline_types import JobContext, SourceProfile
from eyetoarm.vision.pixel_map import validate_roi
from eyetoarm.vision.planar import _check_frame

CALIBRATION_MISMATCH = "CALIBRATION_MISMATCH"
TOP_FACE_UNRESOLVED = "TOP_FACE_UNRESOLVED"
CONTOUR_CLIPPED = "CONTOUR_CLIPPED"
NO_OBJECT = "NO_OBJECT"


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

        instances.append(AiInstance(
            index=i, class_index=cls, class_name=name, confidence=conf,
            box_xyxy=(x1 + ox, y1 + oy, x2 + ox, y2 + oy),
            polygon=None if points is None else points.astype(np.float32) + offset,
            area_px=area, component_count=components, hole_count=holes, clipped=clipped, rejected=rejected))

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


def observation_from_ai(job: JobContext, result: AiSegResult) -> PlanarObservation:
    """작업과 AI 표시 결과로 표시용 무효 관측을 만든다. 결과 설정 해시가 작업 시작 설정과 다르면 AiJobConfigError."""
    if result.config_sha256 != job.detector_config_id:
        raise AiJobConfigError(f"display result config {result.config_sha256[:12]} != job start config "
                               f"{job.detector_config_id[:12]}")

    f = job.frame

    return PlanarObservation(
        run_id=f.run_id, source_id=f.source_id, image_seq=f.image_seq,
        capture_generation=f.capture_generation, generation=job.generation, job_seq=job.job_seq,
        detector_config_id=job.detector_config_id, pixel_basis=PixelBasis.RAW,
        calibration_id=f.calibration_id, observed_face=None, calc_contour=None, display_mask_contour=None,
        candidate_count=result.candidate_count, selected_index=None, valid=False,
        reason=display_reason(result, f, job.settings.profile), process_ms=result.process_ms,
        method="ai", class_name=None, confidence=None, model_id=result.model_id)


__all__ = ["convert", "crop_frame", "display_reason", "observation_from_ai"]
