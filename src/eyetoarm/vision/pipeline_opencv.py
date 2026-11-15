"""현재 OpenCV 표시용 검출을 파이프라인 관측으로 바꾸는 어댑터(계획 r4 작업 5).

근거: 계획 `docs/superpowers/plans/2026-11-15-a-vision-pipeline.md` r4 작업 5, A 상세설계 §5.2(관측면·계산용 윤곽)·
§5.4(판정 사유 표), 공통 자료형 관리 초안 §3.3(PlanarObservation 유효 조건·무효 시 reason 필수).

- 이 어댑터는 **표시용**이다. 관측은 항상 `valid=False`이고 `calc_contour`·`observed_face`·`selected_index`·
  `display_mask_contour`는 None이다. eligible 후보를 자동 선택하거나 전체 윤곽을 계산용으로 승격하지 않는다.
  클래스·윗면 추출이 아니며 운영 A 판정 전체를 대체하지 않는다.
- 사유 우선순위(계획 「제안 사유 정책」): 프로파일 없음·적용 조건 불일치 → `CALIBRATION_MISMATCH`.
  프로파일 일치 시 eligible 후보가 있으면 `TOP_FACE_UNRESOLVED`, 없고 면적 통과·잘림 후보가 있으면
  `CONTOUR_CLIPPED`, 나머지(후보 없음·면적 탈락만)는 `NO_OBJECT`. 후보별 원인은 PlanarResult에 그대로 남는다.
- `candidate_count`는 전체 표시 후보 수 `len(result.candidates)`다(eligible 수와 구분).
- 검출 예외는 잡지 않고 그대로 올린다. 실행기가 `fail(job, INFERENCE_ERROR, ...)`로 정리한다. 관측을 위조하지 않는다.
- 작업 스냅샷 설정(`job.settings`)으로 계산한다. 처리 중 설정이 바뀌어도 시작 설정을 쓴다.
"""
from __future__ import annotations

from eyetoarm.common.types import FramePacket, PixelBasis, PlanarObservation
from eyetoarm.vision.pipeline_types import JobContext, SourceProfile
from eyetoarm.vision.planar import PlanarResult, detect_planar

# A 상세설계 §5.4 판정 사유 이름(판정 주체 A)
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
    """작업과 표시용 결과로 무효 관측을 만든다. 식별자는 작업·원본 프레임 값을 같은 타입 그대로 옮긴다."""
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
    """작업 스냅샷 설정으로 detect_planar를 실행해 (무효 관측, 표시 결과)를 돌려준다. 예외는 그대로 올린다."""
    result = detect_planar(job.frame, job.settings.detector_config)

    return observation_from_result(job, result), result
