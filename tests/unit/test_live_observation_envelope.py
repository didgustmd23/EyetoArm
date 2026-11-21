"""`vision/live_observation_envelope` 단위시험: 정확 stamp TF 조회·반환 뒤 상한 판정·봉투 결합 검사.

CameraInfo 짝 진단도 본다. 카메라·자세는 합성 값이며 정확도와 무관하다.
"""
import unittest

import numpy as np

from eyetoarm.vision.observation_envelope import ObservationEnvelopeError
from eyetoarm.vision.live_observation_envelope import LiveCaptureTfLookup, camera_info_pairing, make_live_envelope
from test_observation_envelope import CAM, STAMP, T, make_frame, make_obs


class FakeClock:
    """조회 함수가 앞으로 돌리는 단조 시계(초)."""

    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def lookup_returning(clock, wait_s, stamp=None, exc=None, calls=None):
    """wait_s만큼 시계를 돌리고 TF를 돌려주는 가짜 조회."""
    def fn(parent, child, stamp_ns, timeout_s):
        if calls is not None:
            calls.append((parent, child, stamp_ns, timeout_s))
        clock.t += wait_s
        if exc is not None:
            raise exc
        return T, stamp_ns if stamp is None else stamp
    return fn


class TfLookupTest(unittest.TestCase):
    def test_exact_stamp_requested_once(self):
        """프레임 stamp·광학 프레임 그대로 한 번 조회한다(Time(0) 대체 없음)."""
        clock, calls = FakeClock(), []
        res = LiveCaptureTfLookup(lookup_returning(clock, 0.01, calls=calls), world_frame="world", wait_limit_s=0.05,
                                  clock=clock)(make_frame())
        self.assertEqual(calls, [("world", CAM.frame_id, STAMP, 0.05)])
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.tf.stamp_ns, STAMP)
        np.testing.assert_allclose(res.tf.T_W_C, T)

    def test_h_49ms_ok_51ms_late(self):
        """반환 49 ms는 사용, 51 ms는 성공이어도 late(tf None)."""
        for wait, status in ((0.049, "ok"), (0.051, "late")):
            clock = FakeClock()
            res = LiveCaptureTfLookup(lookup_returning(clock, wait), world_frame="world", wait_limit_s=0.05,
                                      clock=clock)(make_frame())
            self.assertEqual(res.status, status, wait)
            self.assertAlmostEqual(res.wait_ms, wait * 1000, places=6)
            self.assertEqual(res.tf is None, status == "late")

    def test_h_window_excludes_prior_wait(self):
        """구간은 호출 직전부터다(조회 전에 흐른 시간은 포함하지 않음)."""
        clock = FakeClock()
        lk = LiveCaptureTfLookup(lookup_returning(clock, 0.02), world_frame="world", wait_limit_s=0.05, clock=clock)
        clock.t += 5.0   # 대기열 대기 흉내
        self.assertEqual(lk(make_frame()).status, "ok")

    def test_stamp_mismatch_and_exception_and_no_stamp(self):
        """다른 stamp 반환·예외·stamp 없는 프레임은 tf None."""
        clock = FakeClock()
        mk = lambda fn: LiveCaptureTfLookup(fn, world_frame="world", wait_limit_s=0.05, clock=clock)
        self.assertEqual(mk(lookup_returning(clock, 0.0, stamp=STAMP + 1))(make_frame()).status, "stamp_mismatch")
        self.assertEqual(mk(lookup_returning(clock, 0.0, exc=RuntimeError("x")))(make_frame()).status, "unavailable")
        self.assertEqual(mk(lookup_returning(clock, 0.0))(make_frame(ros_stamp_ns=None)).status, "no_stamp")

    def test_limit_required(self):
        """상한은 기본값 없이 양수만."""
        with self.assertRaises(TypeError):
            LiveCaptureTfLookup(lambda *a: None, world_frame="world")  # noqa
        with self.assertRaises(ValueError):
            LiveCaptureTfLookup(lambda *a: None, world_frame="world", wait_limit_s=0)


class EnvelopeTest(unittest.TestCase):
    def ok_tf(self):
        """성공 TF 결과."""
        clock = FakeClock()
        return LiveCaptureTfLookup(lookup_returning(clock, 0.0), world_frame="world", wait_limit_s=0.05,
                                   clock=clock)(make_frame())

    def test_builds_from_frame(self):
        """촬영 정보는 프레임 값으로 만들고 TF·관측은 그대로 담긴다."""
        f = make_frame()
        env = make_live_envelope(f, make_obs(f), camera=CAM, tf_result=self.ok_tf(), world_frame="world")
        self.assertEqual((env.capture.stamp_ns, env.capture.calibration_id), (STAMP, CAM.calibration_id))
        self.assertIs(env.frame, f)
        self.assertEqual(env.tf.stamp_ns, STAMP)

    def test_late_tf_gives_envelope_without_tf(self):
        """TF late/없음이면 tf=None 봉투(B가 TF_UNAVAILABLE로 처리)."""
        clock = FakeClock()
        late = LiveCaptureTfLookup(lookup_returning(clock, 0.06), world_frame="world", wait_limit_s=0.05,
                                   clock=clock)(make_frame())
        f = make_frame()
        self.assertIsNone(make_live_envelope(f, make_obs(f), camera=CAM, tf_result=late, world_frame="world").tf)

    def test_l_binding_errors(self):
        """관측·카메라와 프레임의 보정 번호·식별자·해상도·frame이 어긋나면 봉투 오류."""
        from eyetoarm.calibration.profile import from_fov
        f = make_frame()
        tf = self.ok_tf()
        with self.assertRaises(ObservationEnvelopeError):
            make_live_envelope(f, make_obs(f, calibration_id="cal-other"), camera=CAM, tf_result=tf, world_frame="world")
        with self.assertRaises(ObservationEnvelopeError):
            make_live_envelope(f, make_obs(f, image_seq=9), camera=CAM, tf_result=tf, world_frame="world")
        other = from_fov(640, 480, 1.0, CAM.frame_id)
        with self.assertRaises(ObservationEnvelopeError):
            make_live_envelope(f, make_obs(f), camera=other, tf_result=tf, world_frame="world")
        wrong_frame = from_fov(f.width, f.height, 1.0, "other_optical")
        with self.assertRaises(ObservationEnvelopeError):
            make_live_envelope(f, make_obs(f), camera=wrong_frame, tf_result=tf, world_frame="world")
        f2 = make_frame(calibration_id=None)
        with self.assertRaises(ObservationEnvelopeError):
            make_live_envelope(f2, make_obs(f2), camera=CAM, tf_result=tf, world_frame="world")

    def test_pairing_diagnostic_only(self):
        """CameraInfo 짝 진단은 true/false/unknown만 내고 봉투 판정과 무관하다."""
        f = make_frame()
        self.assertEqual(camera_info_pairing(f, None), "unknown")
        self.assertEqual(camera_info_pairing(f, {STAMP}), "true")
        self.assertEqual(camera_info_pairing(f, {STAMP + 1}), "false")
        env = make_live_envelope(f, make_obs(f), camera=CAM, tf_result=self.ok_tf(), world_frame="world")
        self.assertNotIn("camera_info_same_stamp", env.to_record())


if __name__ == "__main__":
    unittest.main()
