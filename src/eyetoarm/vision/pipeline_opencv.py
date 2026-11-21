"""OpenCV 표시용 검출 결과를 파이프라인 관측으로 바꾸는 어댑터.

표시용이라 관측은 항상 `valid=False`이고 계산용 윤곽·선택 후보는 None이다.
사유 우선순위: 프로파일 불일치 → `CALIBRATION_MISMATCH`, eligible 후보 → `TOP_FACE_UNRESOLVED`,
면적 통과·잘림 후보 → `CONTOUR_CLIPPED`, 나머지 → `NO_OBJECT`. 검출 예외는 그대로 올린다.
"""
from __future__ import annotations

from eyetoarm.common.types import FramePacket, PixelBasis, PlanarObservation
from eyetoarm.vision.pipeline_types import JobContext, SourceProfile
from eyetoarm.vision.planar import PlanarResult, detect_planar

# 판정 사유 이름(판정 주체 A)
CALIBRATION_MISMATCH = "CALIBRATION_MISMATCH"
TOP_FACE_UNRESOLVED = "TOP_FACE_UNRESOLVED"
CONTOUR_CLIPPED = "CONTOUR_CLIPPED"
NO_OBJECT = "NO_OBJECT"


def display_reason(result: PlanarResult, frame: FramePacket, profile: SourceProfile | None) -> str:
    """표시용 결과의 무효 사유를 우선순위대로 정한다(위 모듈 설명)."""
    if profile is None or not profile.applies_to(frame):
        return CALIBRATION_MISMATCH

    if result.eligible_count > 0:
        return TOP_FACE_UNRESOLVED

    # 면적은 통과했지만 경계에 걸린 후보(면적·잘림 동시 탈락은 제외)
    if any(c.area_reject_reason is None and c.clip_reasons for c in result.candidates):
        return CONTOUR_CLIPPED

    return NO_OBJECT


def observation_from_result(job: JobContext, result: PlanarResult) -> PlanarObservation:
    """작업과 표시용 결과로 무효 관측을 만든다. 식별자는 작업·원본 프레임 값을 그대로 옮긴다."""
    f = job.frame

    return PlanarObservation(
        run_id=f.run_id, source_id=f.source_id, image_seq=f.image_seq,
        capture_generation=f.capture_generation, generation=job.generation, job_seq=job.job_seq,
        detector_config_id=job.detector_config_id, pixel_basis=PixelBasis.RAW,
        calibration_id=f.calibration_id, observed_face=None, calc_contour=None, display_mask_contour=None,
        candidate_count=len(result.candidates), selected_index=None, valid=False,
        reason=display_reason(result, f, job.settings.profile), process_ms=result.process_ms,
        method="opencv", class_name=None, confidence=None, model_id=None)


def process_opencv(job: JobContext) -> tuple[PlanarObservation, PlanarResult]:
    """작업 스냅샷 설정으로 detect_planar를 실행해 (무효 관측, 표시 결과)를 돌려준다."""
    result = detect_planar(job.frame, job.settings.detector_config)

    return observation_from_result(job, result), result
