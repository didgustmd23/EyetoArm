"""#173 B1~B4 단위시험: 동적 정확 조회·되감기 준비에서 stamp 0·음수·None·bool·float를 조회 호출 전에 막고, 시각 0(`Time()`)은 정적 쌍
확인에서만 쓴다.

근거: `_workspace/173_builder2_live_tf_ros_verification_spec.md` r1 §2.1(B1~B4). tf2에서 시각 0은 "최신" 조회라, 0이 조회 함수까지 가면
반환 stamp 검사로 나중에 거부하더라도 최신 대체 금지 조건 위반이다. 실제 ROS는 쓰지 않는다(가짜 모듈).
"""
import types
import unittest

import numpy as np

from eyetoarm.transforms.live_tf_buffer import LiveTfRuntime, TfRewindReadiness
from eyetoarm.vision.live_observation_envelope import LiveCaptureTfLookup
from eyetoarm.vision.live_observation_state import LiveObservationConfig, LiveObservationState
from test_live_tf_buffer import Log, fake_modules

BAD_STAMPS = (0, -1, -5_000_000_000, True, False, 1.5, 0.0, 2_000.0, "2000")
OPTICAL = "link_cam_optical_frame"


def frame(stamp, optical=OPTICAL):
    """조회 대상 프레임."""
    return types.SimpleNamespace(ros_stamp_ns=stamp, optical_frame=optical)


class B1ExactLookupTest(unittest.TestCase):
    def test_bad_stamps_never_reach_lookup(self):
        """B1: 0·음수·bool·float·문자열 stamp는 조회 함수 호출 0건, tf None, 상태 invalid_stamp."""
        calls = []
        lk = LiveCaptureTfLookup(lambda *a: calls.append(a) or (np.eye(4), a[2]), world_frame="world",
                                 wait_limit_s=0.05)
        for s in BAD_STAMPS:
            with self.subTest(stamp=s):
                res = lk(frame(s))
                self.assertIsNone(res.tf)
                self.assertEqual(res.status, "invalid_stamp")
        self.assertEqual(calls, [])

    def test_none_stamp_still_no_stamp_and_no_call(self):
        """None(stamp 없음)은 기존 상태 no_stamp 그대로, 조회 호출 0건."""
        calls = []
        lk = LiveCaptureTfLookup(lambda *a: calls.append(a), world_frame="world", wait_limit_s=0.05)
        self.assertEqual(lk(frame(None)).status, "no_stamp")
        self.assertEqual(calls, [])

    def test_positive_int_still_exact(self):
        """양의 정수 stamp는 그대로 한 번 조회한다(회귀)."""
        calls = []
        lk = LiveCaptureTfLookup(lambda *a: calls.append(a) or (np.eye(4), a[2]), world_frame="world",
                                 wait_limit_s=0.05)
        self.assertEqual(lk(frame(1)).status, "ok")
        self.assertEqual(calls, [("world", OPTICAL, 1, 0.05)])


class B2RewindReadinessTest(unittest.TestCase):
    def setup(self):
        """되감기 경계 1, 제어 반영 끝, 시도 시작된 상태."""
        st = LiveObservationState(run_id="r", generation=0, calibration_id="c",
                                  config=LiveObservationConfig(write_capacity=4, reserved_capacity=16))
        st.on_rx_boundary(1, "sim_time_rewind")
        st.on_applied(1, 1, "c")
        self.calls = []
        rd = TfRewindReadiness(st, clear=lambda: None, has_static=lambda p, c: True,
                               lookup=lambda *a: self.calls.append(a) or (np.eye(4), a[2]), world_frame="world",
                               static_pairs=(("upper_arm_link", "link_cam_link"),), wait_limit_s=0.05,
                               future_tolerance_ns=500_000_000)
        rd.begin(1)
        return st, rd

    def test_bad_stamps_no_lookup_no_open(self):
        """B2: 첫 프레임 stamp가 0·음수·bool·float·문자열이면 조회 0건·on_tf_ready 0건·준비 안 열림."""
        st, rd = self.setup()
        for s in BAD_STAMPS:
            with self.subTest(stamp=s):
                self.assertEqual(rd.try_ready(1, frame(s)), "invalid_stamp")
                self.assertFalse(st.token.ready)
        self.assertEqual(self.calls, [])
        self.assertNotIn("duplicate_tf_ready", st.counts)

    def test_none_no_lookup_no_open(self):
        """None stamp도 조회 0건·준비 안 열림(no_stamp)."""
        st, rd = self.setup()
        self.assertEqual(rd.try_ready(1, frame(None)), "no_stamp")
        self.assertEqual((self.calls, st.token.ready), ([], False))

    def test_valid_stamp_opens(self):
        """양의 정수 stamp는 조회 뒤 연다(회귀)."""
        st, rd = self.setup()
        self.assertEqual(rd.try_ready(1, frame(2_000)), "ready")
        self.assertTrue(st.token.ready)

    def test_world_to_optical_cannot_be_static_pair(self):
        """B4: 정확 조회 대상(world→optical)을 정적 쌍으로 두면 준비하지 않는다(시각 0 확인으로 대체 금지)."""
        st = LiveObservationState(run_id="r", generation=0, calibration_id="c",
                                  config=LiveObservationConfig(write_capacity=4, reserved_capacity=16))
        st.on_rx_boundary(1, "sim_time_rewind")
        st.on_applied(1, 1, "c")
        rd = TfRewindReadiness(st, clear=lambda: None, has_static=lambda p, c: True,
                               lookup=lambda *a: (np.eye(4), a[2]), world_frame="world",
                               static_pairs=(("world", OPTICAL),), wait_limit_s=0.05, future_tolerance_ns=1)
        rd.begin(1)
        self.assertEqual(rd.try_ready(1, frame(2_000)), "static_pair_is_dynamic_target")
        self.assertFalse(st.token.ready)


class B3RuntimeLookupTest(unittest.TestCase):
    def start(self, **kw):
        """가짜 rclpy 런타임."""
        self.log = Log()
        rt = LiveTfRuntime(object(), modules=fake_modules(self.log), **kw)
        rt.start()
        self.addCleanup(rt.close, 1.0)
        return rt

    def test_bad_stamp_raises_before_buffer(self):
        """B3: runtime.lookup은 0·음수·bool·float이면 buffer.lookup_transform 전에 ValueError."""
        rt = self.start()
        for s in (0, -1, True, 1.5, 0.0):
            with self.subTest(stamp=s):
                with self.assertRaises(ValueError):
                    rt.lookup("world", OPTICAL, s, 0.05)
        self.assertFalse(any(e[0] == "lookup" for e in self.log))

    def test_valid_stamp_uses_exact_time(self):
        """양의 정수는 Time(nanoseconds=stamp)로 조회(시각 0 아님)."""
        rt = self.start()
        rt.buffer.dynamic[7] = 7
        self.assertEqual(rt.lookup("world", OPTICAL, 7, 0.05)[1], 7)
        self.assertIn(("lookup", "world", OPTICAL, 7, 0.05), self.log)


class B4StaticOnlyTest(unittest.TestCase):
    def test_has_static_only_for_allowed_pairs(self):
        """B4: has_static(시각 0)은 런타임에 등록한 정적 쌍에만 쓰고, 목록 밖(동적 사슬 포함)은 확인하지 않고 False."""
        log = Log()
        rt = LiveTfRuntime(object(), modules=fake_modules(log),
                           static_pairs=(("upper_arm_link", "link_cam_link"),))
        rt.start()
        self.addCleanup(rt.close, 1.0)
        rt.buffer.static = {("upper_arm_link", "link_cam_link"), ("world", OPTICAL)}
        self.assertTrue(rt.has_static("upper_arm_link", "link_cam_link"))
        self.assertFalse(rt.has_static("world", OPTICAL))
        self.assertEqual(rt.static_refused, 1)

    def test_static_pairs_required(self):
        """정적 쌍 목록 없이 has_static을 쓰면 모두 거부(허용 목록 기본값 없음)."""
        rt = LiveTfRuntime(object(), modules=fake_modules(Log()))
        rt.start()
        self.addCleanup(rt.close, 1.0)
        rt.buffer.static = {("a", "b")}
        self.assertFalse(rt.has_static("a", "b"))


if __name__ == "__main__":
    unittest.main()
