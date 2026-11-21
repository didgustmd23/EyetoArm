"""코어 설정형·표시 결과형 합집합(OpenCV·AI) 시험.

검출 방식·AI 설정 변경은 세대 증가, 작업은 시작 설정 스냅샷으로 계산·기록한다.
관측은 시험용 표시 관측(method="ai", valid=False)이며 실제 모델 결과가 아니다.
"""
import dataclasses
import unittest

import numpy as np

from eyetoarm.common.types import InvalidationKind, PlanarObservation
from eyetoarm.vision.pipeline import GenerationCause, PipelineContractError, VisionPipeline
from eyetoarm.vision.pipeline_runner import b_gate
from eyetoarm.vision.pipeline_types import JobStatus, PipelineSettings
from eyetoarm.vision.planar import detect_planar
from test_ai_config import ai_config
from test_ai_result import instance, seg_result
from test_pipeline import RUN_ID, FakeClock, config, frame, observation, settings
from test_pipeline_terminal import valid_observation


def ai_settings(**over):
    """AI 설정을 쓰는 처리 설정."""
    return settings(detector_config=ai_config(**over))


def ai_observation(job, **over):
    """작업 식별자를 옮긴 시험용 AI 표시 관측(무효)."""
    fields = {"method": "ai", "model_id": job.settings.detector_config.model_id, "reason": "TOP_FACE_UNRESOLVED",
              "candidate_count": 1}
    fields.update(over)

    return dataclasses.replace(observation(job), **fields)


def ai_display(job, **over):
    """작업 시작 설정 해시를 가진 표시 결과."""
    fields = {"config_sha256": job.detector_config_id, "model_id": job.settings.detector_config.model_id}
    fields.update(over)

    return seg_result([instance(0)], **fields)


class AiSettingsTest(unittest.TestCase):
    """설정형 합집합."""

    def pipe(self, sett=None):
        """AI 설정 파이프라인."""
        return VisionPipeline(sett or ai_settings(), run_id=RUN_ID, clock_ns=FakeClock())

    def test_settings_accept_ai_config_and_id_is_its_hash(self):
        """detector_config_id는 AiSegConfig의 config_sha256이고 작업 스냅샷도 같은 ID다."""
        s = ai_settings()
        p = self.pipe(s)
        p.offer_frame(frame(0))
        job = p.start_next()

        self.assertEqual(s.detector_config_id, s.detector_config.config_sha256)
        self.assertEqual(job.detector_config_id, s.detector_config.config_sha256)
        self.assertIs(job.settings.detector_config, s.detector_config)

    def test_settings_still_reject_other_types(self):
        """PlanarConfig·AiSegConfig가 아닌 검출 설정은 계속 ValueError다."""
        for bad in ({"imgsz": 640}, "config", None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    PipelineSettings(source_id="src0", mode="planar", detector_config=bad, profile=None)

    def test_switch_opencv_to_ai_bumps_generation_and_drops_current(self):
        """OpenCV → AI 전환은 세대 전이 1건·관측 무효·대기 폐기, 같은 AI 설정 재적용은 변경 아님."""
        p = VisionPipeline(settings(), run_id=RUN_ID, clock_ns=FakeClock())
        p.offer_frame(frame(0))
        j = p.start_next()
        p.complete(j, valid_observation(j))
        p.offer_frame(frame(1))
        target = ai_settings()

        event = p.change_settings(target, cause=GenerationCause.DETECTOR_CONFIG_CHANGED)

        self.assertEqual((event.kind.value, event.generation, event.detector_config_id),
                         ("GENERATION_CHANGED", 1, target.detector_config_id))
        self.assertIsNone(p.current_observation)
        self.assertIsNone(p.pending_frame)
        self.assertIsNone(p.change_settings(ai_settings(), cause=GenerationCause.DETECTOR_CONFIG_CHANGED))

    def test_conf_change_bumps_generation_and_blocks_old_result(self):
        """conf 0.25로 시작한 작업 도중 conf를 0.10으로 바꾸면 세대가 오르고, 그 작업 완료는 세대 폐기라 관측·B 지점 없음."""
        p = self.pipe()
        p.offer_frame(frame(0))
        job = p.start_next()

        p.change_settings(ai_settings(conf_threshold=0.10), cause=GenerationCause.DETECTOR_CONFIG_CHANGED)
        outcome = p.complete(job, ai_observation(job), display_result=ai_display(job))

        self.assertEqual(outcome.status, JobStatus.DISCARDED_GENERATION)
        self.assertIsNone(p.current_observation)
        self.assertFalse(b_gate(p, outcome))

    def test_job_snapshot_keeps_start_config(self):
        """설정을 바꾼 뒤에도 시작한 작업은 시작 AI 설정(conf 0.25)과 그 ID를 그대로 들고 있다."""
        p = self.pipe()
        p.offer_frame(frame(0))
        job = p.start_next()
        start_id = job.detector_config_id

        p.change_settings(ai_settings(conf_threshold=0.5), cause=GenerationCause.DETECTOR_CONFIG_CHANGED)

        self.assertEqual((job.settings.detector_config.conf_threshold, job.detector_config_id), (0.25, start_id))
        self.assertNotEqual(p.settings.detector_config_id, start_id)

    def test_modified_ai_config_is_contract_error(self):
        """생성 뒤 값이 바뀐 AI 설정은 코어 재검증에서 계약 오류이고 상태가 바뀌지 않는다(생성자·change_settings 모두)."""
        tampered = ai_config(conf_threshold=0.5)
        object.__setattr__(tampered, "imgsz", 640)
        p = self.pipe()

        with self.assertRaises(PipelineContractError):
            p.change_settings(settings(detector_config=tampered), cause=GenerationCause.DETECTOR_CONFIG_CHANGED)

        self.assertEqual(p.current_generation(), 0)

        with self.assertRaises(PipelineContractError):
            VisionPipeline(settings(detector_config=tampered), run_id=RUN_ID, clock_ns=FakeClock())


class AiDisplayResultTest(unittest.TestCase):
    """표시 결과형 합집합과 방식·설정 일치 검사."""

    def start(self, sett):
        """작업 하나를 시작한다."""
        p = VisionPipeline(sett, run_id=RUN_ID, clock_ns=FakeClock())
        p.offer_frame(frame(0))
        return p, p.start_next()

    def test_display_result_accepts_ai_result(self):
        """AI 작업은 AiSegResult(시작 설정 해시)와 method="ai" 관측으로 정상 완료되고 응답에 표시 결과가 그대로 실린다."""
        p, job = self.start(ai_settings())
        display = ai_display(job)

        outcome = p.complete(job, ai_observation(job), display_result=display)

        self.assertEqual(outcome.status, JobStatus.COMPLETED)
        self.assertIs(outcome.display_result, display)
        self.assertIsInstance(outcome.observation, PlanarObservation)

    def test_mismatched_display_or_method_is_adapter_contract_violation(self):
        """방식·설정 해시·형이 어긋난 표시 결과는 INFERENCE_ERROR(adapter_contract_violation)다."""
        cases = []

        p, job = self.start(ai_settings())
        cases.append(("ai job + opencv observation", p, job, observation(job), None, "method"))

        p, job = self.start(ai_settings())
        cases.append(("ai job + planar display", p, job, ai_observation(job),
                      detect_planar(frame(0), config()), "display_result_type"))

        p, job = self.start(ai_settings())
        cases.append(("ai job + other config display", p, job, ai_observation(job),
                      ai_display(job, config_sha256="d" * 64), "display_result_config"))

        p, job = self.start(ai_settings())
        cases.append(("ai job + not a result", p, job, ai_observation(job), "x", "display_result_type"))

        p, job = self.start(settings())
        cases.append(("opencv job + ai display", p, job, observation(job),
                      seg_result(config_sha256=job.detector_config_id), "display_result_type"))

        p, job = self.start(settings())
        cases.append(("opencv job + ai observation", p, job,
                      dataclasses.replace(observation(job), method="ai", model_id="m"), None, "method"))

        for name, pipe, j, obs, display, field in cases:
            with self.subTest(name):
                outcome = pipe.complete(j, obs, display_result=display)
                self.assertEqual(outcome.status, JobStatus.FAILED)
                self.assertIn("adapter_contract_violation", outcome.error)
                self.assertIn(field, outcome.error)

    def test_opencv_path_unchanged(self):
        """기존 OpenCV 작업은 PlanarConfig·opencv 관측·표시 결과 None으로 이전과 같이 완료된다."""
        p, job = self.start(settings())

        self.assertEqual(p.complete(job, observation(job)).status, JobStatus.COMPLETED)


class AiTamperLivenessTest(unittest.TestCase):
    """생성 뒤 변조한 method·config_sha256도 예외로 새지 않고 그 작업만 INFERENCE_ERROR가 된다."""

    def tamper(self, obj, **fields):
        """frozen 객체를 생성 뒤 변조한다(생성자 검증을 우회한 잘못된 어댑터 출력 모사)."""
        for name, value in fields.items():
            object.__setattr__(obj, name, value)

        return obj

    def assert_failed_once_then_next_runs(self, p, job, obs, display, field):
        """한 번 FAILED(INFERENCE_ERROR·필드 이름)로 정리되고, 변조 관측이 현재가 되지 않으며, 다음 작업이 시작·반영된다."""
        out = p.complete(job, obs, display_result=display)

        self.assertIs(out.status, JobStatus.FAILED)
        self.assertIs(out.invalidation.kind, InvalidationKind.INFERENCE_ERROR)
        self.assertIn("adapter_contract_violation", out.error)
        self.assertIn(field, out.error)
        self.assertIsNot(p.current_observation, obs)
        self.assertTrue(p.complete(job, obs, display_result=display).duplicate)
        c = p.counters
        self.assertEqual((c.completed, c.failed), (0, 1))
        self.assertIsNone(p._active)

        p.offer_frame(frame(1))
        nxt = p.start_next()
        self.assertIsNotNone(nxt)
        done = p.complete(nxt, ai_observation(nxt) if isinstance(nxt.settings.detector_config, type(ai_config()))
                          else observation(nxt))
        self.assertTrue(done.applied)
        self.assertIs(p.current_observation, done.observation)
        c = p.counters
        self.assertEqual(c.first_started + c.reprocess_started,
                         c.completed + c.generation_discarded + c.failed + c.aborted)

    def test_tampered_method_arrays_and_non_strings(self):
        """AI·OpenCV 작업 모두에서 method가 다원소 배열·한 원소 배열·비문자열이면 method 계약 위반으로 1회 실패하고 진행이 이어진다."""
        for sett_name, make_sett, make_obs, ok in (("ai", ai_settings, ai_observation, "ai"),
                                                   ("opencv", settings, observation, "opencv")):
            for value in (np.array([ok, "wrong"]), np.array([ok]), 1, None):
                with self.subTest(settings=sett_name, value=repr(value)):
                    p = VisionPipeline(make_sett(), run_id=RUN_ID, clock_ns=FakeClock())
                    p.offer_frame(frame(0))
                    job = p.start_next()
                    obs = self.tamper(make_obs(job), method=value)

                    self.assert_failed_once_then_next_runs(p, job, obs, None, "method")

    def test_tampered_display_config_sha_arrays_and_non_strings(self):
        """config_sha256이 배열·비문자열이면 display_result_config 위반으로 1회 실패하고 진행한다."""
        for kind in ("multi", "single", "int"):
            with self.subTest(kind=kind):
                p = VisionPipeline(ai_settings(), run_id=RUN_ID, clock_ns=FakeClock())
                p.offer_frame(frame(0))
                job = p.start_next()
                value = {"multi": np.array([job.detector_config_id, "x"]),
                         "single": np.array([job.detector_config_id]), "int": 5}[kind]
                display = self.tamper(ai_display(job), config_sha256=value)

                self.assert_failed_once_then_next_runs(p, job, ai_observation(job), display, "display_result_config")


if __name__ == "__main__":
    unittest.main()
