"""#171 R1 단위시험: 수신 슬롯 경계 훅(잠금 안 호출), CameraProfileInfo의 CameraProfile 보관, LiveLinkSource `frame_epoch`.

근거: `_workspace/171_builder2_live_a_to_b_plan.md` r3 §0.1·§3.1-1·§5.1 R1. 훅이 없으면 기존 동작과 같다(기존 slot·adapter 시험이 회귀).
"""
import dataclasses
import types
import unittest

from eyetoarm.inputs.live_slot import BoundaryKind, CameraProfileInfo, GenerationMirror, LiveFrameSlot, LiveSlotConfig
from eyetoarm.inputs.ros_image import LiveLinkSource, LiveSourceConfig, profile_info
from eyetoarm.vision.live_observation_state import LiveObservationConfig, LiveObservationState
from test_ros_image_adapter import CountingIssuer, FakeTransport, SourceBase, image_msg, view_convert
from test_ros_image_adapter import info_msg as adapter_info_msg


def info_msg(fx=500.0, frame_id="link_cam_optical_frame"):
    """가짜 CameraInfo."""
    return types.SimpleNamespace(k=[fx, 0, 320, 0, fx, 240, 0, 0, 1], d=[], width=640, height=480,
                                 header=types.SimpleNamespace(frame_id=frame_id))


class ProfileInfoTest(unittest.TestCase):
    def test_camera_profile_kept_and_not_compared(self):
        """profile_info는 K·왜곡 원본을 camera로 싣고, 같음 비교는 식별 정보만 본다."""
        a, b = profile_info(info_msg()), profile_info(info_msg())
        self.assertEqual(a.camera.calibration_id, a.calibration_id)
        self.assertEqual(a.camera.K[0, 0], 500.0)
        self.assertEqual(a, b)
        self.assertIsNot(a.camera, b.camera)
        self.assertNotEqual(a, profile_info(info_msg(fx=501.0)))
        self.assertNotIn("camera", repr(a))

    def test_default_camera_none(self):
        """camera 없이 만든 식별 정보(기존 호출)도 그대로 쓸 수 있다."""
        info = CameraProfileInfo("cal-x", 640, 480, "f")
        self.assertIsNone(info.camera)


class SlotHookTest(unittest.TestCase):
    def make_slot(self, hook):
        """훅을 단 슬롯."""
        return LiveFrameSlot(LiveSlotConfig(), generation=GenerationMirror(0), clock_ns=lambda: 1_000, boundary_hook=hook)

    def test_hook_called_inside_slot_lock_with_epoch(self):
        """경계마다 epoch·종류로 훅을 부르며, 그때 slot 잠금을 쥐고 있다."""
        seen = []
        slot = None

        def hook(epoch, kind):
            seen.append((epoch, kind, slot._cond._is_owned()))

        slot = self.make_slot(hook)
        slot.on_camera_info(CameraProfileInfo("cal-a", 640, 480, "f"))
        slot.on_camera_info(CameraProfileInfo("cal-b", 640, 480, "f"))
        self.assertEqual(seen, [(1, BoundaryKind.PROFILE_CHANGED, True), (2, BoundaryKind.PROFILE_CHANGED, True)])
        self.assertEqual([i.epoch for i in slot.poll_inputs()], [1, 2])

    def test_hook_error_counted_not_raised(self):
        """훅 예외는 수신을 멈추지 않고 boundary_hook_errors로 세며, 경계 뒤에 치명 알림 BOUNDARY_HOOK_FAILED를 넣는다(#180 #5)."""
        def bad(epoch, kind):
            raise RuntimeError("boom")

        slot = self.make_slot(bad)
        slot.on_camera_info(CameraProfileInfo("cal-a", 640, 480, "f"))
        self.assertEqual(slot.stats()["boundary_hook_errors"], 1)
        self.assertEqual([i.kind.value for i in slot.poll_inputs()], ["profile_changed", "boundary_hook_failed"])

    def test_hook_closes_live_observation_state(self):
        """T-k 결합: 실제 슬롯 경계가 LiveObservationState를 즉시 닫는다(제어 흐름 반영 전)."""
        st = LiveObservationState(run_id="r", generation=0, calibration_id=None,
                        config=LiveObservationConfig(write_capacity=2, reserved_capacity=4))
        slot = self.make_slot(st.on_rx_boundary)
        tok = st.token
        slot.on_camera_info(CameraProfileInfo("cal-a", 640, 480, "f"))
        self.assertFalse(st.token.ready)
        self.assertFalse(st.is_current(tok))
        self.assertEqual(st.boundary.boundary_id, 1)

    def test_no_hook_default_unchanged(self):
        """훅이 없으면 기존과 같고 오류 수 0."""
        slot = LiveFrameSlot(LiveSlotConfig(), generation=GenerationMirror(0), clock_ns=lambda: 1_000)
        slot.on_camera_info(CameraProfileInfo("cal-a", 640, 480, "f"))
        self.assertEqual(slot.stats()["boundary_hook_errors"], 0)


class FrameEpochTest(SourceBase):
    def test_frame_epoch_known_and_unknown(self):
        """frame_epoch는 이 실행이 조립한 프레임의 수신 epoch, 모르는 프레임·다른 실행은 None."""
        self.make()
        self.assertTrue(self.source.handle_camera_info(adapter_info_msg()))
        self.apply_all()
        self.source.handle_image(image_msg(1))
        frame = self.source.read()
        self.assertIsNotNone(frame.calibration_id)
        self.assertEqual(self.source.frame_epoch(frame), 1)
        self.assertIsNone(self.source.frame_epoch(dataclasses.replace(frame, image_seq=999)))
        self.assertIsNone(self.source.frame_epoch(dataclasses.replace(frame, run_id="other")))

    def test_profile_item_carries_camera(self):
        """PROFILE_CHANGED 항목의 식별 정보에 같은 CameraInfo의 CameraProfile이 실린다."""
        self.make()
        self.source.handle_camera_info(adapter_info_msg())
        items = self.source.poll_input()
        self.assertEqual(items[0].kind, BoundaryKind.PROFILE_CHANGED)
        cam = items[0].profile.camera
        self.assertEqual(cam.calibration_id, items[0].profile.calibration_id)

    def test_source_boundary_hook_passthrough(self):
        """LiveLinkSource에 준 훅이 슬롯 경계에서 불린다."""
        seen = []
        self.transport = FakeTransport()
        src = LiveLinkSource(source_id="gz_link", issuer=CountingIssuer(), generation_provider=lambda: 0,
                             config=LiveSourceConfig(frame_wait_s=0.01), transport_factory=lambda s: self.transport,
                             convert=view_convert, clock_ns=lambda: 5_000,
                             boundary_hook=lambda e, k: seen.append((e, k)))
        src.handle_camera_info(adapter_info_msg())
        self.assertEqual(seen, [(1, BoundaryKind.PROFILE_CHANGED)])


if __name__ == "__main__":
    unittest.main()
