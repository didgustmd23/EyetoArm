"""A 비전 파이프라인 상태 코어(작업 2·3) 단위시험 — 세대·설정 전이·최신 대기 프레임·작업 스냅샷·재처리.

근거: 계획 `docs/superpowers/plans/2026-11-15-a-vision-pipeline.md` r3 작업 2·3·「인터페이스 제안」,
A 상세설계 §3(최신 프레임 1개·버퍼 비덮어쓰기)·§5.1(image_seq 수명)·§5.4(세대 증가 조건·설정 스냅샷·
동일 프레임 재처리 조건)·§9(처리 건수·보존식), 공통 자료형 관리 초안 §3.5·§4.
시계는 가짜 `clock_ns`를 주입한다. ROS·Qt·실제 입력 장치를 쓰지 않는다.

작업 4 이전 임시 사항: `complete()`가 아직 없어 활성 작업을 끝내는 공개 경로가 없다. 재처리 시험은
골격 헬퍼 `_release_active(job)`로 활성 슬롯만 비우고, 현재 관측 무효화 시험은 `_current_observation`에
시험용 관측을 직접 넣는다. 작업 4에서 `complete()`로 바꾼다.
"""
import dataclasses
import subprocess
import sys
import unittest

import numpy as np

from eyetoarm.common import types as common
from eyetoarm.common.types import InvalidationKind, PixelBasis, PlanarObservation, TimestampBase
from eyetoarm.inputs import base
from eyetoarm.vision.pipeline import GenerationCause, PipelineContractError, VisionPipeline
from eyetoarm.vision.pipeline_types import (
    JobContext,
    PipelineSettings,
    QueueDropReason,
    SourceProfile,
)
from eyetoarm.vision.planar import PlanarConfig

RUN_ID = "run-p"

# test_planar.py BASE와 같은 시험용 검출 설정
CONFIG_FIELDS = {
    "format_version": 1,
    "status": "provisional_display_only",
    "roi": None,
    "resize_wh": None,
    "resize_interpolation": "INTER_AREA",
    "gaussian_blur_ksize": 0,
    "polarity": "dark",
    "threshold_method": "fixed",
    "fixed_threshold": 128,
    "morph_open_ksize": 0,
    "contour_retrieval": "RETR_EXTERNAL",
    "contour_approximation": "CHAIN_APPROX_SIMPLE",
    "min_area_source_px2": 100,
    "max_area_source_px2": None,
    "border_margin_processed_px": 2,
}

# 영상 크기(4×6)와 보정 ID. 프로파일 적용 조건과 맞춘다
WIDTH, HEIGHT, CAL_ID = 6, 4, "cal-1"


class FakeClock:
    """호출마다 1000 ns씩 늘어나는 가짜 단조 시계. 호출 수를 센다."""

    def __init__(self, start_ns=1_000):
        """시작 시각을 받는다."""
        self.now = start_ns
        self.calls = 0

    def __call__(self):
        """현재 값을 돌려주고 1000 ns 늘린 뒤 호출 수를 1 올린다."""
        self.calls += 1
        value = self.now
        self.now += 1_000
        return value


def config(**over):
    """`CONFIG_FIELDS`에서 키워드로 준 값만 바꾼 `PlanarConfig`."""
    fields = dict(CONFIG_FIELDS)
    fields.update(over)

    return PlanarConfig(**fields)


def profile(**over):
    """프레임 기본값(src0·6×4·크롭 없음·cal-1)과 맞는 `SourceProfile`."""
    fields = {"source_id": "src0", "width": WIDTH, "height": HEIGHT, "crop": None, "calibration_id": CAL_ID}
    fields.update(over)

    return SourceProfile(**fields)


def settings(**over):
    """src0·planar 모드·기본 검출 설정·일치 프로파일의 `PipelineSettings`."""
    fields = {"source_id": "src0", "mode": "planar", "detector_config": config(), "profile": profile()}
    fields.update(over)

    return PipelineSettings(**fields)


def frame(image_seq=0, capture_generation=0, *, common_type=False, image=None, **over):
    """식별자·시각이 서로 다른 값인 A 확장(또는 공통) 프레임. 영상은 image_seq로 채운 쓰기 가능 배열."""
    if image is None:
        image = np.full((HEIGHT, WIDTH, 3), image_seq % 256, dtype=np.uint8)

    fields = {
        "image": image, "run_id": RUN_ID, "source_id": "src0", "image_seq": image_seq,
        "timestamp_ms": 10_000 + image_seq, "timestamp_base": TimestampBase.SIM_TIME,
        "received_monotonic_ns": 500_000 + image_seq, "width": WIDTH, "height": HEIGHT,
        "crop": None, "calibration_id": CAL_ID, "capture_generation": capture_generation,
        "optical_frame": "cam_optical", "ros_stamp_ns": 77_000_000_000 + image_seq,
    }

    if not common_type:
        fields.update(source_ref=f"img{image_seq}.png", source_index=100 + image_seq)

    fields.update(over)
    cls = common.FramePacket if common_type else base.FramePacket

    return cls(**fields)


def observation(job):
    """작업 식별자를 그대로 옮긴 시험용 무효 관측(현재 관측 무효화 확인용)."""
    return PlanarObservation(
        run_id=job.frame.run_id, source_id=job.frame.source_id, image_seq=job.frame.image_seq,
        capture_generation=job.frame.capture_generation, generation=job.generation,
        job_seq=job.job_seq, detector_config_id=job.detector_config_id, pixel_basis=PixelBasis.RAW,
        calibration_id=None, observed_face=None, calc_contour=None, display_mask_contour=None,
        candidate_count=0, selected_index=None, valid=False, reason="NO_OBJECT", process_ms=0.1,
        method="opencv", class_name=None, confidence=None, model_id=None)


# 원인별로 바꾼 설정을 만드는 함수(A 상세설계 §5.4 세대 증가 조건). 탐색·재접속·시각 되감기는 설정 불변
CHANGES = {
    GenerationCause.SOURCE_CHANGED: lambda s: dataclasses.replace(
        s, source_id="src1", profile=profile(source_id="src1")),
    GenerationCause.PROFILE_CHANGED: lambda s: dataclasses.replace(s, profile=profile(calibration_id="cal-2")),
    GenerationCause.MODE_CHANGED: lambda s: dataclasses.replace(s, mode="planar_alt"),
    GenerationCause.SEEK: lambda s: s,
    GenerationCause.RECONNECT: lambda s: s,
    GenerationCause.SIM_TIME_REWIND: lambda s: s,
    GenerationCause.DETECTOR_CONFIG_CHANGED: lambda s: dataclasses.replace(
        s, detector_config=config(roi=(1, 1, 3, 2))),
}

# 검출 설정 변경의 세부 항목(ROI·이진화·노이즈·면적·경계 여유)
DETECTOR_VARIANTS = {
    "roi": {"roi": (1, 1, 3, 2)},
    "threshold": {"fixed_threshold": 90},
    "binarization": {"threshold_method": "otsu", "fixed_threshold": None},
    "noise": {"morph_open_ksize": 3},
    "area": {"min_area_source_px2": 50},
    "border": {"border_margin_processed_px": 5},
}


class PipelineTestBase(unittest.TestCase):
    """가짜 시계와 기본 설정의 파이프라인을 준비하는 공통 기반."""

    def setUp(self):
        """가짜 시계·기본 설정·파이프라인을 만든다."""
        self.clock = FakeClock()
        self.settings = settings()
        self.p = VisionPipeline(self.settings, run_id=RUN_ID, clock_ns=self.clock)

    def assert_received_balance(self, p=None):
        """보존식 `received = first_started + queue_dropped + 대기(0/1)`를 확인한다(A 상세설계 §9)."""
        p = p or self.p
        c = p.counters
        pending = 0 if p.pending_frame is None else 1

        self.assertEqual(c.received, c.first_started + sum(c.queue_dropped.values()) + pending)

    def snapshot(self, p=None):
        """식별자·계수·선택·대기·폐기 기록의 비교용 상태 묶음."""
        p = p or self.p

        return (p.current_generation(), p.counters, p.selected_frame, p.pending_frame,
                p.queue_drop_records, p.current_observation, p.settings)

    def start_and_release(self, f, p=None):
        """프레임을 offer·start하고 활성 슬롯만 비운다(작업 4 전 임시 경로). 시작한 작업을 돌려준다."""
        p = p or self.p
        self.assertTrue(p.offer_frame(f))
        job = p.start_next()
        self.assertIsNotNone(job)
        p._release_active(job)

        return job


class GenerationTransitionTest(PipelineTestBase):
    """세대 증가와 설정 전이(계획 작업 2, A 상세설계 §5.4)."""

    def test_generation_and_job_seq_start_at_zero(self):
        """세대와 job_seq는 0에서 시작하고 `current_generation()`은 공급자로 쓸 수 있는 호출 가능 객체다."""
        self.assertEqual(self.p.current_generation(), 0)
        self.assertIsNone(self.p.current_observation)
        self.assertIsNone(self.p.selected_frame)

        job = self.p.offer_frame(frame()) and self.p.start_next()
        self.assertEqual((job.generation, job.job_seq), (0, 0))

    def test_generation_changes_invalidate_immediately(self):
        """소스·보정·모드·탐색·재접속·시각 되감기·검출 설정 변경마다 세대가 정확히 1 오르고 현재 결과가 즉시 None이며 사건은 새 세대·식별자 None이다."""
        cases = [(c.value, c, CHANGES[c]) for c in GenerationCause]
        cases += [(f"detector:{k}", GenerationCause.DETECTOR_CONFIG_CHANGED,
                   lambda s, o=o: dataclasses.replace(s, detector_config=config(**o)))
                  for k, o in DETECTOR_VARIANTS.items()]

        for name, cause, change in cases:
            with self.subTest(change=name):
                clock = FakeClock()
                p = VisionPipeline(settings(), run_id=RUN_ID, clock_ns=clock)

                # 현재 관측이 있는 상태(작업 4 전 임시로 직접 설정)
                job = self.start_and_release(frame(0), p)
                p._current_observation = observation(job)
                self.assertTrue(p.offer_frame(frame(1)))

                new_settings = change(p.settings)
                expected_time = clock.now
                event = p.change_settings(new_settings, cause=cause)

                self.assertEqual(p.current_generation(), 1)
                self.assertIsNone(p.current_observation)
                self.assertIs(p.settings, new_settings)

                self.assertIs(event.kind, InvalidationKind.GENERATION_CHANGED)
                self.assertEqual(event.generation, 1)
                self.assertIsNone(event.job_seq)
                self.assertIsNone(event.image_seq)
                self.assertEqual(event.run_id, RUN_ID)
                self.assertEqual(event.source_id, new_settings.source_id)
                self.assertEqual(event.detector_config_id, new_settings.detector_config.config_sha256)
                self.assertEqual(event.event_monotonic_ns, expected_time)
                self.assertTrue(event.cause.startswith(cause.value))

                # 남은 대기 프레임은 세대 전이 사유로 한 번 폐기
                self.assertIsNone(p.pending_frame)
                self.assertEqual(p.counters.queue_dropped, {QueueDropReason.GENERATION_CHANGED.value: 1})
                self.assertEqual(p.queue_drop_records[-1].image_seq, 1)
                self.assert_received_balance(p)

                # 한 번 더 같은 종류로 바꾸면 다시 정확히 1 증가
                if cause in (GenerationCause.SEEK, GenerationCause.RECONNECT, GenerationCause.SIM_TIME_REWIND):
                    p.change_settings(p.settings, cause=cause)
                    self.assertEqual(p.current_generation(), 2)

    def test_presentation_change_does_not_bump_generation(self):
        """오버레이 색은 처리 설정에 없고, 같은 검출 설정 해시의 재적용은 세대·현재 결과·대기·선택을 바꾸지 않고 None을 돌려준다."""
        # 표시 전용 설정은 처리 설정 필드가 아니다
        with self.assertRaises(TypeError):
            PipelineSettings(source_id="src0", mode="planar", detector_config=config(),
                             profile=profile(), overlay_color=(0, 255, 0))

        job = self.start_and_release(frame(0))
        self.p._current_observation = obs = observation(job)
        self.assertTrue(self.p.offer_frame(frame(1)))
        before = self.snapshot()
        clock_calls = self.clock.calls

        # 새 객체지만 같은 값(같은 config_sha256)의 검출 설정
        same = dataclasses.replace(self.settings, detector_config=config())
        self.assertIsNot(same.detector_config, self.settings.detector_config)

        for cause in (GenerationCause.DETECTOR_CONFIG_CHANGED, GenerationCause.MODE_CHANGED,
                      GenerationCause.SOURCE_CHANGED, GenerationCause.PROFILE_CHANGED):
            with self.subTest(cause=cause):
                self.assertIsNone(self.p.change_settings(same, cause=cause))

        self.assertEqual(self.p.current_generation(), 0)
        self.assertIs(self.p.current_observation, obs)
        self.assertEqual(self.snapshot()[:5], before[:5])
        self.assertEqual(self.clock.calls, clock_calls)

    def test_invalid_settings_change_is_atomic(self):
        """검증 실패(설정 형식·변조된 PlanarConfig·원인 불일치·모르는 원인)는 예외이며 세대·현재 결과·선택·대기·기록이 변하지 않는다."""
        job = self.start_and_release(frame(0))
        self.p._current_observation = observation(job)
        self.assertTrue(self.p.offer_frame(frame(1)))
        before = self.snapshot()

        # 생성 뒤 값을 바꿔 검증을 우회한 설정(재검증으로 잡아야 함)
        tampered = config(roi=(1, 1, 3, 2))
        object.__setattr__(tampered, "fixed_threshold", 999.0)

        bad_calls = [
            ("not settings", lambda: self.p.change_settings({"source_id": "src0"},
                                                           cause=GenerationCause.SOURCE_CHANGED)),
            ("tampered config", lambda: self.p.change_settings(
                dataclasses.replace(self.settings, detector_config=tampered),
                cause=GenerationCause.DETECTOR_CONFIG_CHANGED)),
            ("cause mismatch", lambda: self.p.change_settings(
                dataclasses.replace(self.settings, detector_config=config(roi=(1, 1, 3, 2))),
                cause=GenerationCause.MODE_CHANGED)),
            ("unknown cause", lambda: self.p.change_settings(self.settings, cause="overlay_color")),
        ]

        for name, call in bad_calls:
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    call()

                self.assertEqual(self.snapshot(), before)

        # 잘못된 검출 값은 PlanarConfig 생성 단계에서 이미 거부된다
        with self.assertRaises(ValueError):
            config(fixed_threshold=300)

    def test_job_seq_is_not_reset_by_generation_change(self):
        """job_seq는 세대 변경으로 초기화되지 않고 실행 안에서 계속 증가한다(A 상세설계 §5.4)."""
        first = self.start_and_release(frame(0))
        self.p.change_settings(self.settings, cause=GenerationCause.RECONNECT)
        second = self.start_and_release(frame(1, capture_generation=1))

        self.assertEqual((first.job_seq, second.job_seq), (0, 1))
        self.assertEqual((first.generation, second.generation), (0, 1))


class ReprocessConditionTest(PipelineTestBase):
    """선택 프레임 재처리 허용 조건(계획 작업 2, A 상세설계 §5.4 「동일 프레임 재처리의 조건」 1~3)."""

    def test_only_detector_changes_allow_selected_frame_reprocessing(self):
        """선택 프레임 수신 뒤 세대 증가가 모두 검출 설정 변경일 때만 재처리를 허용하고, 소스→검출 설정처럼 중간에 다른 전이가 있으면 거부한다."""
        # 검출 설정 변경만 두 번: 허용
        self.start_and_release(frame(0))
        self.p.change_settings(CHANGES[GenerationCause.DETECTOR_CONFIG_CHANGED](self.p.settings),
                               cause=GenerationCause.DETECTOR_CONFIG_CHANGED)
        self.p.change_settings(dataclasses.replace(self.p.settings, detector_config=config(fixed_threshold=90)),
                               cause=GenerationCause.DETECTOR_CONFIG_CHANGED)

        job = self.p.start_reprocess()
        self.assertIsNotNone(job)
        self.assertEqual(job.generation, 2)

        # 다른 전이 뒤 검출 설정 변경: 마지막이 검출 설정이어도 거부
        for cause in GenerationCause:
            if cause is GenerationCause.DETECTOR_CONFIG_CHANGED:
                continue

            with self.subTest(first=cause):
                clock = FakeClock()
                p = VisionPipeline(settings(), run_id=RUN_ID, clock_ns=clock)
                self.start_and_release(frame(0), p)
                p.change_settings(CHANGES[cause](p.settings), cause=cause)
                p.change_settings(dataclasses.replace(p.settings, detector_config=config(fixed_threshold=90)),
                                  cause=GenerationCause.DETECTOR_CONFIG_CHANGED)
                before = self.snapshot(p)
                calls = clock.calls

                self.assertIsNone(p.start_reprocess())
                self.assertEqual(self.snapshot(p), before)
                self.assertEqual(clock.calls, calls)

    def test_cumulative_history_rejects_even_if_selection_survives(self):
        """선택 비우기가 빠져도(방어선 단독) 수신 뒤 비검출 전이 이력이 있으면 재처리를 거부한다(누적 이력 판정)."""
        for cause in GenerationCause:
            if cause in (GenerationCause.DETECTOR_CONFIG_CHANGED, GenerationCause.SOURCE_CHANGED):
                continue  # 소스 변경은 source_id 불일치로도 거부되므로 이력 판정만 보려고 뺀다

            with self.subTest(first=cause):
                p = VisionPipeline(settings(), run_id=RUN_ID, clock_ns=FakeClock())
                p._clear_selection = lambda: None  # 선택 비우기를 막아 이력 판정만 남긴다
                self.start_and_release(frame(0), p)

                # 프로파일 변경은 없음→원래 프로파일로 되돌려 프로파일 조건은 통과시키고 이력만 남긴다
                if cause is GenerationCause.PROFILE_CHANGED:
                    p.change_settings(dataclasses.replace(p.settings, profile=None), cause=cause)
                    p.change_settings(dataclasses.replace(p.settings, profile=profile()), cause=cause)
                else:
                    p.change_settings(CHANGES[cause](p.settings), cause=cause)

                p.change_settings(dataclasses.replace(p.settings, detector_config=config(fixed_threshold=90)),
                                  cause=GenerationCause.DETECTOR_CONFIG_CHANGED)

                self.assertIsNotNone(p.selected_frame)
                self.assertIsNone(p.start_reprocess())

    def test_combined_change_is_not_treated_as_detector_only(self):
        """한 호출에서 검출 설정과 다른 설정이 함께 바뀌면 비검출 전이로 보고 선택을 비운다(M3 임시 결정)."""
        self.start_and_release(frame(0))
        both = dataclasses.replace(self.settings, mode="planar_alt", detector_config=config(fixed_threshold=90))

        event = self.p.change_settings(both, cause=GenerationCause.DETECTOR_CONFIG_CHANGED)

        self.assertIn(GenerationCause.MODE_CHANGED.value, event.cause)
        self.assertIsNone(self.p.selected_frame)
        self.assertIsNone(self.p.start_reprocess())

    def test_reprocess_requires_profile_that_applies_to_frame(self):
        """프로파일 없음·해상도·크롭·source_id·calibration_id 불일치·보정 누락이면 재처리를 시작하지 않고 식별자·계수가 변하지 않는다."""
        cases = {
            "no profile": (None, {}),
            "width": (profile(width=WIDTH + 1), {}),
            "height": (profile(height=HEIGHT + 1), {}),
            "crop": (profile(crop=(0, 0, WIDTH, HEIGHT)), {}),
            "source_id": (profile(source_id="other"), {}),
            "calibration_id": (profile(calibration_id="cal-9"), {}),
            "calibration missing": (profile(calibration_id=None), {"calibration_id": None}),
            "frame without calibration": (profile(), {"calibration_id": None}),
        }

        for name, (prof, frame_over) in cases.items():
            with self.subTest(case=name):
                clock = FakeClock()
                p = VisionPipeline(settings(profile=prof), run_id=RUN_ID, clock_ns=clock)
                self.start_and_release(frame(0, **frame_over), p)
                before = self.snapshot(p)
                calls = clock.calls

                self.assertIsNone(p.start_reprocess())
                self.assertEqual(self.snapshot(p), before)
                self.assertEqual(clock.calls, calls)

        # 대조: 일치하는 프로파일이면 시작
        p = VisionPipeline(settings(), run_id=RUN_ID, clock_ns=FakeClock())
        self.start_and_release(frame(0), p)
        self.assertIsNotNone(p.start_reprocess())

    def test_reprocess_needs_selection_and_no_active_job(self):
        """선택 프레임이 없거나 활성 작업이 있으면 재처리는 None이고 계수가 변하지 않는다."""
        self.assertIsNone(self.p.start_reprocess())

        self.assertTrue(self.p.offer_frame(frame(0)))
        self.p.start_next()
        before = self.snapshot()

        self.assertIsNone(self.p.start_reprocess())
        self.assertEqual(self.snapshot(), before)


class PendingQueueTest(PipelineTestBase):
    """최신 대기 프레임 1개와 수신 계약(계획 작업 3, A 상세설계 §3·§9)."""

    def test_latest_pending_frame_replaces_only_pending(self):
        """A 처리 중 B·C를 받으면 처리 중 A는 보존되고 대기는 C 하나, B는 폐기 1건이며, source 버퍼를 덮어써도 A 처리 버퍼는 그대로다."""
        img_a = np.full((HEIGHT, WIDTH, 3), 11, np.uint8)
        self.assertTrue(self.p.offer_frame(frame(0, image=img_a)))
        job_a = self.p.start_next()

        self.assertTrue(self.p.offer_frame(frame(1)))
        self.assertTrue(self.p.offer_frame(frame(2)))

        # 처리 중 A 보존, 대기 C 하나
        self.assertEqual(job_a.frame.image_seq, 0)
        self.assertEqual(self.p.pending_frame.image_seq, 2)
        self.assertIs(self.p.selected_frame, job_a.frame)

        # B 폐기 1건(작업 식별자 없음)
        drops = self.p.queue_drop_records
        self.assertEqual(len(drops), 1)
        self.assertEqual((drops[0].image_seq, drops[0].reason), (1, QueueDropReason.REPLACED_BY_NEWER))
        self.assertEqual(drops[0].capture_generation, 0)
        self.assertEqual(drops[0].timestamp_ms, 10_001)
        self.assertEqual(drops[0].timestamp_base, TimestampBase.SIM_TIME)
        self.assertEqual(drops[0].ros_stamp_ns, 77_000_000_001)
        self.assertEqual(drops[0].received_monotonic_ns, 500_001)
        self.assertFalse(hasattr(drops[0], "job_seq"))
        self.assertEqual(self.p.counters.queue_dropped, {QueueDropReason.REPLACED_BY_NEWER.value: 1})

        # 생산자가 A 버퍼를 덮어써도 처리 버퍼는 그대로
        img_a[:] = 200
        self.assertTrue(np.all(job_a.frame.image == 11))

        # 다음 시작은 활성 작업이 끝난 뒤 C
        self.assertIsNone(self.p.start_next())
        self.p._release_active(job_a)
        self.assertEqual(self.p.start_next().frame.image_seq, 2)
        self.assert_received_balance()

    def test_offer_copies_and_owns_frame_buffer(self):
        """offer 시점에 영상을 독립 복사·읽기 전용으로 소유하고, type·source_ref·source_index·식별자·시각을 보존하며 생산자 덮어쓰기(대기·처리 중)에 영향받지 않는다."""
        for common_type in (False, True):
            with self.subTest(common_type=common_type):
                p = VisionPipeline(settings(), run_id=RUN_ID, clock_ns=FakeClock())
                producer = np.full((HEIGHT, WIDTH, 3), 7, np.uint8)
                original = frame(5, common_type=common_type, image=producer)

                self.assertTrue(p.offer_frame(original))
                owned = p.pending_frame

                # 독립 복사·읽기 전용
                self.assertIsNot(owned, original)
                self.assertIs(type(owned), type(original))
                self.assertFalse(owned.image.flags.writeable)
                self.assertFalse(np.shares_memory(owned.image, producer))

                # 식별자·시각·A 확장 필드 보존
                for f in dataclasses.fields(original):
                    if f.name != "image":
                        self.assertEqual(getattr(owned, f.name), getattr(original, f.name), f.name)

                # 대기 중 덮어쓰기
                producer[:] = 50
                self.assertTrue(np.all(owned.image == 7))

                # 처리 중 덮어쓰기
                job = p.start_next()
                self.assertIs(job.frame, owned)
                producer[:] = 90
                self.assertTrue(np.all(job.frame.image == 7))

        # 소스가 이미 읽기 전용으로 넘긴 버퍼도 다시 복사한다
        p = VisionPipeline(settings(), run_id=RUN_ID, clock_ns=FakeClock())
        ro = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        ro.setflags(write=False)
        self.assertTrue(p.offer_frame(frame(0, image=ro)))
        self.assertFalse(np.shares_memory(p.pending_frame.image, ro))

    def test_old_capture_is_not_accepted_as_new_frame(self):
        """현재 source의 오래된 캡처 세대 프레임은 시작하지 않고 폐기 사유·건수를 남기며, 다른 run/source는 계약 오류로 거부하고 수신 수에 넣지 않는다."""
        self.p.change_settings(self.settings, cause=GenerationCause.SEEK)
        self.assertTrue(self.p.offer_frame(frame(1, capture_generation=1)))

        # 오래된 캡처: received·queue_dropped 각 +1, 대기 프레임은 그대로
        self.assertFalse(self.p.offer_frame(frame(0, capture_generation=0)))
        self.assertEqual(self.p.pending_frame.image_seq, 1)
        self.assertEqual(self.p.counters.received, 2)
        self.assertEqual(self.p.counters.queue_dropped, {QueueDropReason.STALE_CAPTURE.value: 1})
        self.assertEqual(self.p.queue_drop_records[-1].image_seq, 0)
        self.assertEqual(self.p.queue_drop_records[-1].capture_generation, 0)
        self.assert_received_balance()

        # 다른 run·source: 예외, 상태 무변
        before = self.snapshot()

        for over in ({"run_id": "run-other"}, {"source_id": "src9"}):
            with self.subTest(over=over):
                with self.assertRaises(PipelineContractError):
                    self.p.offer_frame(frame(3, capture_generation=1, **over))

                self.assertEqual(self.snapshot(), before)

        self.assertTrue(issubclass(PipelineContractError, ValueError))

        # 시작은 대기 중이던 새 프레임
        self.assertEqual(self.p.start_next().frame.image_seq, 1)

    def test_future_capture_generation_is_contract_error(self):
        """현재 세대보다 앞선 capture_generation은 계약 오류로 거부한다(M4 임시 결정)."""
        before = self.snapshot()

        with self.assertRaises(PipelineContractError):
            self.p.offer_frame(frame(0, capture_generation=1))

        self.assertEqual(self.snapshot(), before)

    def test_non_frame_offer_is_contract_error(self):
        """FramePacket이 아닌 입력(입력 사건 등)은 계약 오류이며 수신 수에 넣지 않는다."""
        event = common.InputStatusEvent(run_id=RUN_ID, source_id="src0",
                                        status=common.InputStatus.END_OF_STREAM,
                                        event_monotonic_ns=1, detail=None)

        with self.assertRaises(PipelineContractError):
            self.p.offer_frame(event)

        self.assertEqual(self.p.counters.received, 0)

    def test_received_balance_holds_across_drop_reasons(self):
        """대기 교체·오래된 캡처·세대 전이 폐기 독립·혼합 사례에서 보존식이 매 단계 성립하고, 시작한 프레임·선택·재처리를 폐기로 다시 세지 않는다."""
        p = self.p

        # 대기 교체만
        p.offer_frame(frame(0))
        self.assert_received_balance()
        p.offer_frame(frame(1))
        self.assert_received_balance()

        # 시작(선택) 후 세대 전이: 시작한 프레임은 폐기로 세지 않음, 대기만 폐기
        job = p.start_next()
        p.offer_frame(frame(2))
        p.change_settings(p.settings, cause=GenerationCause.RECONNECT)
        self.assert_received_balance()

        # 오래된 캡처
        self.assertFalse(p.offer_frame(frame(3, capture_generation=0)))
        self.assert_received_balance()

        # 혼합: 새 대기, 교체, 재처리(받기 아님)
        p._release_active(job)
        p.offer_frame(frame(4, capture_generation=1))
        p.offer_frame(frame(5, capture_generation=1))
        job = p.start_next()
        p._release_active(job)
        p.change_settings(dataclasses.replace(p.settings, detector_config=config(fixed_threshold=90)),
                          cause=GenerationCause.DETECTOR_CONFIG_CHANGED)
        self.assertIsNotNone(p.start_reprocess())
        self.assert_received_balance()

        c = p.counters
        self.assertEqual(c.received, 6)
        self.assertEqual(c.first_started, 2)
        self.assertEqual(c.reprocess_started, 1)
        self.assertEqual(c.queue_dropped, {
            QueueDropReason.REPLACED_BY_NEWER.value: 2,
            QueueDropReason.GENERATION_CHANGED.value: 1,
            QueueDropReason.STALE_CAPTURE.value: 1,
        })
        self.assertEqual(len(p.queue_drop_records), 4)

    def test_counters_property_is_a_snapshot(self):
        """`counters` 조회 결과를 고쳐도 파이프라인 계수는 변하지 않는다."""
        self.p.offer_frame(frame(0))
        c = self.p.counters
        c.received = 99
        c.queue_dropped["x"] = 1

        self.assertEqual(self.p.counters.received, 1)
        self.assertEqual(self.p.counters.queue_dropped, {})


class JobSnapshotTest(PipelineTestBase):
    """작업 시작·선택 프레임·설정 스냅샷(계획 작업 3, A 상세설계 §5.4 「설정 스냅샷과 폐기」)."""

    def test_start_next_is_noop_while_active_or_empty(self):
        """대기 프레임이 없거나 활성 작업이 있으면 `start_next()`는 None이고 job_seq·계수·시계 호출이 변하지 않는다."""
        before = self.snapshot()
        calls = self.clock.calls
        self.assertIsNone(self.p.start_next())
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.clock.calls, calls)

        self.p.offer_frame(frame(0))
        job = self.p.start_next()
        self.p.offer_frame(frame(1))
        before = self.snapshot()
        calls = self.clock.calls

        self.assertIsNone(self.p.start_next())
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.clock.calls, calls)

        # 활성 작업을 끝낸 뒤 다음 job_seq는 직전+1
        self.p._release_active(job)
        self.assertEqual(self.p.start_next().job_seq, job.job_seq + 1)

    def test_job_keeps_original_configuration(self):
        """시작 뒤 설정을 바꿔도 이미 만든 작업의 설정·해시·generation·시작 시각은 바뀌지 않는다."""
        self.p.offer_frame(frame(0))
        start_ns = self.clock.now
        job = self.p.start_next()
        old_id = job.detector_config_id

        self.assertEqual(job.started_monotonic_ns, start_ns)
        self.assertFalse(job.reprocess)
        self.assertEqual(self.p.counters.first_started, 1)

        new = dataclasses.replace(self.settings, detector_config=config(fixed_threshold=90))
        self.p.change_settings(new, cause=GenerationCause.DETECTOR_CONFIG_CHANGED)
        self.p.change_settings(CHANGES[GenerationCause.MODE_CHANGED](self.p.settings),
                               cause=GenerationCause.MODE_CHANGED)

        self.assertIs(job.settings, self.settings)
        self.assertEqual(job.detector_config_id, old_id)
        self.assertNotEqual(job.detector_config_id, new.detector_config.config_sha256)
        self.assertEqual(job.generation, 0)
        self.assertEqual(job.started_monotonic_ns, start_ns)

        with self.assertRaises(dataclasses.FrozenInstanceError):
            job.settings = new

    def test_selection_starts_with_job_and_survives_detector_change(self):
        """선택은 실제 작업 시작 때 소유 복사 프레임으로 정해지고, offer·대기 교체·재처리 시작·검출 설정 변경은 선택을 바꾸지 않으며, 다른 세대 전이는 선택을 비운다."""
        # offer만으로는 선택되지 않음
        self.p.offer_frame(frame(0))
        self.assertIsNone(self.p.selected_frame)

        job = self.p.start_next()
        self.assertIs(self.p.selected_frame, job.frame)
        self.assertIs(self.p.selected_frame, self.p.selected_frame)

        # 처리 중 offer·대기 교체는 선택을 바꾸지 않음
        self.p.offer_frame(frame(1))
        self.p.offer_frame(frame(2))
        self.assertIs(self.p.selected_frame, job.frame)

        # 처리 중 검출 설정만 변경: 선택 유지(완료 폐기 경로는 작업 4)
        self.p.change_settings(dataclasses.replace(self.settings, detector_config=config(fixed_threshold=90)),
                               cause=GenerationCause.DETECTOR_CONFIG_CHANGED)
        self.assertIs(self.p.selected_frame, job.frame)

        # 조건을 만족하면 재처리 가능, 재처리 시작은 선택을 바꾸지 않음
        self.p._release_active(job)
        re_job = self.p.start_reprocess()
        self.assertIs(re_job.frame, job.frame)
        self.assertIs(self.p.selected_frame, job.frame)

        # 다음 실제 start_next는 선택을 새 프레임으로 교체
        self.p._release_active(re_job)
        self.p.offer_frame(frame(3, capture_generation=1))
        new_job = self.p.start_next()
        self.assertIs(self.p.selected_frame, new_job.frame)
        self.assertEqual(new_job.frame.image_seq, 3)

        # 검출 설정 이외의 모든 전이는 선택을 비운다
        for cause in GenerationCause:
            if cause is GenerationCause.DETECTOR_CONFIG_CHANGED:
                continue

            with self.subTest(cause=cause):
                p = VisionPipeline(settings(), run_id=RUN_ID, clock_ns=FakeClock())
                self.start_and_release(frame(0), p)
                self.assertIsNotNone(p.selected_frame)

                p.change_settings(CHANGES[cause](p.settings), cause=cause)

                self.assertIsNone(p.selected_frame)
                self.assertIsNone(p.start_reprocess())

    def test_reprocess_keeps_original_identity_and_time(self):
        """재처리는 source.read·issuer를 부르지 않고 선택 프레임(같은 객체)을 쓰며, 식별자·원본/수신 시각을 보존하고 새 job_seq·현재 generation만 붙이며 received는 늘지 않는다."""

        class CountingSource:
            """read 호출 수를 세는 가짜 입력 소스."""

            def __init__(self, frames):
                """돌려줄 프레임 목록을 받는다."""
                self.frames = list(frames)
                self.reads = 0

            def read(self):
                """다음 프레임을 돌려주고 호출 수를 센다."""
                self.reads += 1
                return self.frames.pop(0)

        class CountingIssuer(base.FrameIdIssuer):
            """next_seq 호출 수를 세는 발급기."""

            def __init__(self, run_id):
                """부모 발급기를 만들고 호출 수를 0으로 둔다."""
                super().__init__(run_id)
                self.calls = 0

            def next_seq(self):
                """호출 수를 센 뒤 부모 값을 돌려준다."""
                self.calls += 1
                return super().next_seq()

        issuer = CountingIssuer(RUN_ID)
        source = CountingSource([frame(issuer.next_seq())])
        self.assertTrue(self.p.offer_frame(source.read()))
        first = self.p.start_next()
        self.p._release_active(first)
        reads, seqs = source.reads, issuer.calls

        self.p.change_settings(dataclasses.replace(self.settings, detector_config=config(fixed_threshold=90)),
                               cause=GenerationCause.DETECTOR_CONFIG_CHANGED)
        received = self.p.counters.received
        start_ns = self.clock.now
        re_job = self.p.start_reprocess()

        self.assertEqual((source.reads, issuer.calls), (reads, seqs))
        self.assertIs(re_job.frame, first.frame)

        f = re_job.frame
        self.assertEqual((f.image_seq, f.capture_generation), (0, 0))
        self.assertEqual((f.timestamp_ms, f.timestamp_base, f.ros_stamp_ns), (10_000, TimestampBase.SIM_TIME,
                                                                             77_000_000_000))
        self.assertEqual(f.received_monotonic_ns, 500_000)

        self.assertEqual(re_job.job_seq, first.job_seq + 1)
        self.assertEqual(re_job.generation, 1)
        self.assertTrue(re_job.reprocess)
        self.assertEqual(re_job.started_monotonic_ns, start_ns)
        self.assertEqual(re_job.detector_config_id, self.p.settings.detector_config.config_sha256)
        self.assertEqual(self.p.counters.received, received)
        self.assertEqual(self.p.counters.reprocess_started, 1)
        self.assertEqual(self.p.counters.first_started, 1)


class SkeletonBoundaryTest(PipelineTestBase):
    """작업 4로 미룬 API와 의존성 경계(계획 Global Constraints·작업 3)."""

    def test_terminal_api_is_not_implemented_yet(self):
        """complete·fail·apply_invalidation·close는 작업 4 전이라 NotImplementedError를 낸다."""
        self.p.offer_frame(frame(0))
        job = self.p.start_next()

        for call in (lambda: self.p.complete(job, observation(job)),
                     lambda: self.p.fail(job, InvalidationKind.INFERENCE_ERROR, "x"),
                     lambda: self.p.apply_invalidation(None),
                     lambda: self.p.close()):
            with self.assertRaises(NotImplementedError):
                call()

    def test_pipeline_imports_no_ros_qt_tcp_or_other_roles(self):
        """파이프라인 모듈을 import해도 ROS·Qt·ultralytics·TCP·GUI·B/C/D 모듈이 로드되지 않고 스레드를 만들지 않는다."""
        code = (
            "import sys, threading\n"
            "import eyetoarm.vision.pipeline, eyetoarm.vision.pipeline_types\n"
            "bad = [m for m in sys.modules if m.split('.')[0] in ('rclpy', 'PySide6', 'PyQt5', 'ultralytics')\n"
            "       or m.startswith(('eyetoarm.tcp', 'eyetoarm.gui', 'eyetoarm.calibration',\n"
            "                        'eyetoarm.transforms', 'eyetoarm.pendant', 'eyetoarm.d_client'))]\n"
            "print(','.join(sorted(bad)) or 'none', threading.active_count())"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                             env={"PYTHONPATH": "src", "PATH": ""}, cwd=_repo_root())

        self.assertEqual(out.stdout.split(), ["none", "1"])

    def test_job_context_type(self):
        """시작 결과는 `JobContext`이며 설정은 파이프라인의 그 시점 설정 객체다."""
        self.p.offer_frame(frame(0))
        job = self.p.start_next()

        self.assertIsInstance(job, JobContext)
        self.assertIs(job.settings, self.p.settings)


def _repo_root():
    """저장소 루트 경로(이 파일 기준 두 단계 위)."""
    import os

    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


if __name__ == "__main__":
    unittest.main()
