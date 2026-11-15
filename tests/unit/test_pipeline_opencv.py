"""작업 5 OpenCV 어댑터 단위시험 — 실제 detect_planar 결과를 표시용 무효 관측으로만 변환하는지.

근거: 계획 r4 작업 5(표시용 관측 고정·사유 우선순위·candidate_count), A 상세설계 §5.2·§5.4(사유 표),
공통 자료형 초안 §3.3(무효 관측 reason 필수, 계산용 윤곽 승격 금지).
합성 영상(흰 배경 위 검은 사각형)을 numpy로 그려 정답을 아는 입력을 만든다. 시험 정답을 같은 검출 코드로 역산하지 않는다.
"""
import dataclasses
import unittest

import numpy as np

from eyetoarm.common.types import PlanarObservation, TimestampBase
from eyetoarm.inputs import base
from eyetoarm.vision.pipeline import VisionPipeline
from eyetoarm.vision.pipeline_opencv import (
    CALIBRATION_MISMATCH,
    CONTOUR_CLIPPED,
    NO_OBJECT,
    TOP_FACE_UNRESOLVED,
    display_reason,
    process_opencv,
)
from eyetoarm.vision.pipeline_types import JobStatus
from eyetoarm.vision.planar import detect_planar
from test_pipeline import RUN_ID, FakeClock, config, profile, settings

# 합성 영상 크기와 보정 ID
W, H, CAL = 64, 48, "cal-1"


def scene(*boxes):
    """흰 배경(255)에 검은 사각형(x0, y0, x1, y1, 끝 미포함)을 그린 BGR 영상."""
    img = np.full((H, W, 3), 255, np.uint8)

    for x0, y0, x1, y1 in boxes:
        img[y0:y1, x0:x1] = 0

    return img


def frame(image, image_seq=0, capture_generation=0, **over):
    """합성 영상의 A 확장 프레임."""
    fields = {
        "image": image, "run_id": RUN_ID, "source_id": "src0", "image_seq": image_seq,
        "timestamp_ms": 5_000 + image_seq, "timestamp_base": TimestampBase.VIDEO_POS,
        "received_monotonic_ns": 900_000 + image_seq, "width": W, "height": H, "crop": None,
        "calibration_id": CAL, "capture_generation": capture_generation, "optical_frame": None,
        "ros_stamp_ns": None, "source_ref": "synthetic.mp4", "source_index": image_seq,
    }
    fields.update(over)

    return base.FramePacket(**fields)


def pipeline(prof="match"):
    """합성 영상 크기에 맞는 프로파일(또는 None·불일치)의 파이프라인."""
    if prof == "match":
        prof = profile(width=W, height=H)

    return VisionPipeline(settings(profile=prof, detector_config=config()), run_id=RUN_ID, clock_ns=FakeClock())


def start(p, image, **over):
    """프레임을 넣고 작업을 시작한다."""
    assert p.offer_frame(frame(image, capture_generation=p.current_generation(), **over))
    return p.start_next()


# 사각형 배치(면적 기준 100 px², 경계 여유 2 px)
CENTER = (20, 14, 40, 30)      # 320 px², 경계 미접촉 → eligible
EDGE = (0, 10, 16, 30)         # 320 px², 왼쪽 원본 경계 접촉 → 면적 통과·잘림
TINY = (30, 20, 34, 24)        # 16 px², 면적 미달
TINY_EDGE = (0, 2, 4, 6)       # 면적 미달 + 잘림


class DisplayObservationTest(unittest.TestCase):
    """표시용 관측 고정(계획 r4 작업 5 첫 항목)."""

    def test_display_result_never_becomes_valid_calculation(self):
        """eligible 후보가 있어도 valid=False·calc_contour·observed_face·selected_index=None이고 eligible 후보를 자동 선택하지 않는다."""
        p = pipeline()
        job = start(p, scene(CENTER))
        obs, result = process_opencv(job)

        self.assertEqual(result.eligible_count, 1)
        self.assertIsInstance(obs, PlanarObservation)
        self.assertFalse(obs.valid)
        self.assertIsNone(obs.calc_contour)
        self.assertIsNone(obs.observed_face)
        self.assertIsNone(obs.selected_index)
        self.assertIsNone(obs.display_mask_contour)
        self.assertEqual(obs.reason, TOP_FACE_UNRESOLVED)
        self.assertEqual(obs.method, "opencv")
        self.assertEqual(obs.process_ms, result.process_ms)

    def test_identity_fields_copied_with_same_types(self):
        """관측 식별자 8필드가 작업·원본 프레임 값과 같은 타입·값이며 코어가 그대로 완료로 받는다."""
        p = pipeline()
        job = start(p, scene(CENTER), image_seq=7)
        obs, _ = process_opencv(job)
        f = job.frame
        expected = {"run_id": f.run_id, "source_id": f.source_id, "image_seq": f.image_seq,
                    "capture_generation": f.capture_generation, "generation": job.generation,
                    "job_seq": job.job_seq, "detector_config_id": job.detector_config_id,
                    "calibration_id": f.calibration_id}

        for name, value in expected.items():
            self.assertIs(type(getattr(obs, name)), type(value), name)
            self.assertEqual(getattr(obs, name), value, name)

        out = p.complete(job, obs)
        self.assertIs(out.status, JobStatus.COMPLETED)

    def test_calibration_id_is_not_invented(self):
        """원본 프레임의 calibration_id가 None이면 관측도 None이다(값을 만들어 넣지 않음)."""
        p = pipeline(prof=None)
        job = start(p, scene(CENTER), calibration_id=None)
        obs, _ = process_opencv(job)

        self.assertIsNone(obs.calibration_id)
        self.assertEqual(obs.reason, CALIBRATION_MISMATCH)


class ReasonPriorityTest(unittest.TestCase):
    """사유 우선순위와 candidate_count(계획 작업 5 「제안 사유 정책」)."""

    def check(self, boxes, expected, prof="match"):
        """장면을 처리해 사유와 후보 수를 확인한다."""
        p = pipeline(prof)
        job = start(p, scene(*boxes))
        obs, result = process_opencv(job)

        self.assertEqual(obs.reason, expected)
        self.assertEqual(obs.candidate_count, len(result.candidates))
        self.assertEqual(display_reason(result, job.frame, job.settings.profile), expected)

        return result

    def test_profile_mismatch_wins(self):
        """프로파일 없음·적용 조건 불일치는 eligible 후보가 있어도 CALIBRATION_MISMATCH가 우선이다."""
        for name, prof in (("none", None), ("calibration", profile(width=W, height=H, calibration_id="cal-9")),
                           ("size", profile(width=W + 1, height=H))):
            with self.subTest(profile=name):
                self.check([CENTER], CALIBRATION_MISMATCH, prof)

    def test_eligible_wins_over_clipped(self):
        """eligible 후보가 있으면 잘림 후보가 함께 있어도 TOP_FACE_UNRESOLVED다."""
        result = self.check([CENTER], TOP_FACE_UNRESOLVED)
        self.assertEqual(result.eligible_count, 1)
        self.check([CENTER, EDGE], TOP_FACE_UNRESOLVED)

    def test_area_passing_clipped_candidate_gives_contour_clipped(self):
        """eligible 없음, 면적 통과·잘림 후보가 있으면 CONTOUR_CLIPPED다."""
        result = self.check([EDGE], CONTOUR_CLIPPED)
        self.assertEqual(result.eligible_count, 0)
        self.assertTrue(any(c.area_reject_reason is None and c.clip_reasons for c in result.candidates))

    def test_no_candidate_or_area_rejects_give_no_object(self):
        """후보 없음·면적 탈락 후보만·면적과 잘림 동시 탈락 후보만이면 NO_OBJECT이고 후보 수는 전체 후보 수다."""
        self.assertEqual(len(self.check([], NO_OBJECT).candidates), 0)
        self.assertEqual(len(self.check([TINY], NO_OBJECT).candidates), 1)
        result = self.check([TINY_EDGE], NO_OBJECT)
        self.assertEqual(len(result.candidates), 1)
        self.assertIsNotNone(result.candidates[0].area_reject_reason)

    def test_area_and_clip_reject_with_separate_clipped_candidate(self):
        """면적·잘림 동시 탈락 후보와 면적 통과·잘림 후보가 함께 있으면 CONTOUR_CLIPPED다."""
        self.check([TINY_EDGE, EDGE], CONTOUR_CLIPPED)


class DetectorErrorTest(unittest.TestCase):
    """검출 예외는 관측을 만들지 않고 fail 경로로 간다."""

    def test_detector_exception_has_no_observation(self):
        """영상 크기와 프레임 메타데이터가 어긋나 detect_planar가 예외를 내면 관측 없이 예외가 올라오고, 호출자가 fail(INFERENCE_ERROR)로 정리한다."""
        p = pipeline()
        bad = scene(CENTER)[:, :32]   # 실제 영상 너비 32 ≠ frame.width 64
        job = start(p, bad)

        with self.assertRaises(ValueError):
            process_opencv(job)

        out = p.fail(job, "INFERENCE_ERROR", "detect_planar: ValueError")
        self.assertIs(out.status, JobStatus.FAILED)
        self.assertIsNone(out.observation)
        self.assertIsNone(p.current_observation)
        self.assertEqual(p.counters.failed, 1)

    def test_uses_job_snapshot_config(self):
        """처리 중 검출 설정이 바뀌어도 어댑터는 작업 스냅샷 설정으로 계산하고 결과 해시가 시작 설정과 같다."""
        p = pipeline()
        job = start(p, scene(CENTER))
        p.change_settings(dataclasses.replace(p.settings, detector_config=config(fixed_threshold=90)),
                          cause="detector_config_changed")
        obs, result = process_opencv(job)

        self.assertEqual(result.config_sha256, job.detector_config_id)
        self.assertEqual(obs.detector_config_id, job.detector_config_id)
        self.assertNotEqual(obs.detector_config_id, p.settings.detector_config_id)
        self.assertEqual(result.config_sha256, detect_planar(job.frame, job.settings.detector_config).config_sha256)


if __name__ == "__main__":
    unittest.main()
