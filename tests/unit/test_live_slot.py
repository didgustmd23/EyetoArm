"""실시간 수신 상태기(`LiveFrameSlot`) 시험 — 최신 칸·수신 epoch·경계·되감기 필터(ROS 없음).

근거: 계획 `docs/superpowers/plans/2026-11-16-live-link-camera.md` r3 §3.3·§3.4(R1~R7)·§4.2(S1~S18), A 상세설계 §5.1·§5.4.
가짜 시계(시험이 직접 움직임)와 가짜 메시지 객체를 쓴다. rclpy를 import하지 않는다.
"""
import threading
import time
import unittest

from eyetoarm.common.types import InputStatus
from eyetoarm.inputs.live_slot import (BoundaryKind, CameraProfileInfo, GenerationMirror, LiveFrameSlot,
                                       LiveInputOverflow, LiveSlotConfig, NoticeKind, TakeStatus)

S = 1_000_000_000  # 1초(ns)
FRAME_ID = "link_cam_optical_frame"


class ManualClock:
    """시험이 직접 움직이는 PC 단조 시계(ns). 호출 수를 센다."""

    def __init__(self, now=10 * S):
        """시작 시각을 받는다."""
        self.now, self.calls = now, 0

    def __call__(self):
        """현재 값."""
        self.calls += 1
        return self.now


def info(cal="cal-a", width=1280, height=720, frame_id=FRAME_ID):
    """프로필 식별 정보."""
    return CameraProfileInfo(calibration_id=cal, width=width, height=height, frame_id=frame_id)


class SlotBase(unittest.TestCase):
    """상태기·시계·세대 사본 준비와 공용 동작."""

    def make(self, **over):
        """기본 설정(기대 frame_id 있음)으로 상태기를 만든다."""
        fields = {"expected_frame_id": FRAME_ID}
        fields.update(over)
        self.clock = ManualClock()
        self.mirror = GenerationMirror(0)
        self.slot = LiveFrameSlot(LiveSlotConfig(**fields), generation=self.mirror, clock_ns=self.clock)
        return self.slot

    def image(self, stamp_s, msg=None, frame_id=FRAME_ID, width=1280, height=720):
        """영상 하나를 넣는다(stamp 초 단위)."""
        return self.slot.on_image(msg if msg is not None else f"m{stamp_s}", stamp_ns=int(stamp_s * S),
                                  frame_id=frame_id, width=width, height=height)

    def apply_all(self, generation_for=None):
        """대기열을 꺼내 모두 반영한다. generation_for(item) → 반영 세대(기본: 경계면 +1)."""
        items = self.slot.poll_inputs()
        gen = self.mirror.get()

        for item in items:
            if generation_for is not None:
                gen = generation_for(item)
            elif item.kind in (BoundaryKind.SIM_TIME_REWIND, BoundaryKind.RECONNECT, BoundaryKind.PROFILE_CHANGED):
                gen += 1

            self.slot.input_applied(item, gen)

        return items


class LatestSlotTest(SlotBase):
    """최신 칸 1개·중복·조립 정보."""

    def test_s1_overwrite_keeps_latest_only(self):
        """S1: 안 읽은 영상 위에 새 영상이 오면 덮어쓰고 overwritten을 센다. take는 마지막 것만 준다."""
        self.make()
        self.image(1.0)
        self.image(1.1)
        self.image(1.2)

        take = self.slot.take(0)
        self.assertEqual((take.status, take.record.message), (TakeStatus.FRAME, "m1.2"))
        self.assertEqual(self.slot.stats()["overwritten"], 2)
        self.assertEqual(self.slot.take(0).status, TakeStatus.EMPTY)

    def test_s2_duplicate_stamp_dropped(self):
        """S2: 같은 stamp는 받지 않는다(duplicate)."""
        self.make()
        self.assertEqual(self.image(1.0), "accepted")
        self.slot.take(0)
        self.assertEqual(self.image(1.0), "duplicate")
        self.assertEqual(self.slot.take(0).status, TakeStatus.EMPTY)
        self.assertEqual(self.slot.stats()["duplicate_stamp"], 1)

    def test_s15_clock_read_once_per_image(self):
        """S15: 영상 콜백마다 시계는 한 번, 그 값이 received_monotonic_ns다."""
        self.make()
        before = self.clock.calls
        self.clock.now = 42 * S
        self.image(1.0)

        self.assertEqual(self.clock.calls - before, 1)
        self.assertEqual(self.slot.take(0).record.received_monotonic_ns, 42 * S)

    def test_source_index_counts_received_messages(self):
        """source_index는 받은 영상 순번이라 덮어쓴 만큼 건너뛴다."""
        self.make()
        self.image(1.0)
        self.image(1.1)

        self.assertEqual(self.slot.take(0).record.source_index, 1)


class ProfileTest(SlotBase):
    """프로필 귀속(R4)·D7·S8~S10."""

    def test_s10_before_first_camera_info_no_calibration(self):
        """S10: 첫 CameraInfo 전 영상은 profile None(calibration_id 없음)."""
        self.make()
        self.image(1.0)

        self.assertIsNone(self.slot.take(0).record.profile)

    def test_s8_profile_then_image(self):
        """S8: 프로필 경계 반영 뒤 받은 영상은 새 프로필·새 세대."""
        self.make()
        self.slot.on_camera_info(info("cal-a"))
        items = self.apply_all()
        self.image(1.0)

        take = self.slot.take(0)
        self.assertEqual([i.kind for i in items], [BoundaryKind.PROFILE_CHANGED])
        self.assertEqual((take.record.profile.calibration_id, take.capture_generation), ("cal-a", 1))

    def test_s9_image_then_profile_change_is_not_promoted(self):
        """S9: 영상이 먼저 오고 조립 전에 프로필이 바뀌면, 그 영상은 이전 프로필로 기록돼 R1로 폐기(새 ID로 둔갑 안 함)."""
        self.make()
        self.slot.on_camera_info(info("cal-a"))
        self.apply_all()
        self.image(1.0)
        self.slot.on_camera_info(info("cal-b"))

        self.assertEqual(self.slot.take(0).status, TakeStatus.NEEDS_INPUT)
        self.apply_all()
        self.assertEqual(self.slot.take(0).status, TakeStatus.EMPTY)
        self.assertEqual(self.slot.stats()["stale_epoch"], 1)

        self.image(1.1)
        take = self.slot.take(0)
        self.assertEqual((take.record.profile.calibration_id, take.capture_generation), ("cal-b", 2))

    def test_same_camera_info_is_not_a_boundary(self):
        """같은 CameraInfo가 반복되면 경계를 만들지 않는다."""
        self.make()
        self.slot.on_camera_info(info())
        self.slot.on_camera_info(info())

        self.assertEqual(len(self.slot.poll_inputs()), 1)

    def test_frame_id_mismatch_gets_no_calibration(self):
        """D7: 헤더 frame_id가 기대값과 다르면 프로필을 싣지 않고 수를 센다(영상은 받음)."""
        self.make()
        self.slot.on_camera_info(info())
        self.apply_all()
        self.image(1.0, frame_id="other_frame")

        self.assertIsNone(self.slot.take(0).record.profile)
        self.assertEqual(self.slot.stats()["frame_id_mismatch"], 1)

    def test_resolution_mismatch_gets_no_calibration(self):
        """영상 해상도가 프로필과 다르면 프로필을 싣지 않는다."""
        self.make()
        self.slot.on_camera_info(info())
        self.apply_all()
        self.image(1.0, width=640, height=480)

        self.assertIsNone(self.slot.take(0).record.profile)


class EpochTest(SlotBase):
    """수신 epoch·세대(R1~R3, S6·S7·S11)."""

    def test_s6_previous_epoch_frame_is_dropped(self):
        """S6: 경계 이전 epoch 영상이 남은 채 경계를 반영하면 조립하지 않는다(stale_epoch)."""
        self.make()
        self.image(5.0)
        self.slot.on_camera_info(info())
        self.apply_all()

        self.assertEqual(self.slot.take(0).status, TakeStatus.EMPTY)
        self.assertEqual(self.slot.stats()["stale_epoch"], 1)

    def test_s7_new_epoch_frame_requires_input_first(self):
        """S7: 반영 안 된 경계 뒤 영상은 꺼내지 않고 NEEDS_INPUT. 반영 뒤에는 새 세대로 꺼낸다."""
        self.make()
        self.slot.on_clock(int(9 * S))
        self.slot.on_clock(int(1 * S))
        self.image(1.0)

        self.assertEqual(self.slot.take(0).status, TakeStatus.NEEDS_INPUT)
        self.apply_all()
        take = self.slot.take(0)
        self.assertEqual((take.status, take.capture_generation), (TakeStatus.FRAME, 1))

    def test_s7b_polled_but_not_applied_still_needs_input(self):
        """S7(R3): 경계를 꺼냈지만 아직 반영하지 않은 사이에도 새 epoch 영상은 꺼내지 않는다(NEEDS_INPUT)."""
        self.make()
        self.slot.on_clock(int(9 * S))
        self.slot.on_clock(int(1 * S))
        self.image(1.0)
        items = self.slot.poll_inputs()

        self.assertEqual(self.slot.take(0).status, TakeStatus.NEEDS_INPUT)
        self.slot.input_applied(items[0], 1)
        self.assertEqual(self.slot.take(0).capture_generation, 1)

    def test_s11_generation_after_receipt_not_attached(self):
        """S11: capture_generation = max(gen_at_rx, gen_of_epoch). 수신 뒤 오른 세대(예: 검출 설정 변경)는 붙지 않는다."""
        self.make()
        self.image(1.0)
        self.mirror.set(7)

        take = self.slot.take(0)
        self.assertEqual(take.capture_generation, 0)

        self.image(1.1)
        self.assertEqual(self.slot.take(0).capture_generation, 7)

    def test_is_current_epoch(self):
        """공개 판정: 반영된 최신 epoch만 현재이고, 대기 중인 경계가 있으면 현재가 아니다."""
        self.make()
        self.assertTrue(self.slot.is_current_epoch(0))
        self.slot.on_camera_info(info())

        self.assertFalse(self.slot.is_current_epoch(0))
        self.apply_all()
        self.assertFalse(self.slot.is_current_epoch(0))
        self.assertTrue(self.slot.is_current_epoch(1))

    def test_out_of_order_apply_rejected(self):
        """경계를 꺼낸 순서와 다르게 반영하면 ValueError이고 상태가 그대로다."""
        self.make()
        self.slot.on_camera_info(info("a"))
        self.slot.on_camera_info(info("b"))
        first, second = self.slot.poll_inputs()

        with self.assertRaises(ValueError):
            self.slot.input_applied(second, 5)

        self.assertEqual(self.slot.stats()["applied_epoch"], 0)
        self.slot.input_applied(first, 1)
        self.slot.input_applied(second, 2)
        self.assertEqual(self.slot.stats()["applied_epoch"], 2)


class RewindTest(SlotBase):
    """되감기(R5·R6, S3~S5). R5는 휴리스틱이며 한계 시험(S3b·S3c)을 함께 둔다."""

    def start_run(self):
        """정상 진행: clock 10 s까지, 영상 9.9 s까지 받고 꺼낸다."""
        self.make()

        for t in (9.0, 9.5, 10.0):
            self.slot.on_clock(int(t * S))
            self.image(t - 0.1)

        self.slot.take(0)

    def test_s3_late_old_frame_beyond_tolerance_dropped(self):
        """S3: clock 되감기(10 s → 1 s) 뒤 stamp 9.9 s 늦은 영상은 1 s + 0.5 s 허용 밖이라 버린다. 그 뒤 새 영상은 새 세대."""
        self.start_run()
        self.slot.on_clock(1 * S)

        self.assertEqual(self.image(9.9), "stale_after_rewind")
        self.assertEqual(self.image(9.8), "stale_after_rewind")
        self.apply_all()

        self.assertEqual(self.image(1.0), "accepted")
        take = self.slot.take(0)
        self.assertEqual((take.record.stamp_ns, take.capture_generation), (1 * S, 1))

    def test_s3a_tolerance_moves_with_current_clock(self):
        """S3a: 허용 상한은 현재 clock 기준이라, 같은 stamp라도 clock이 진행한 뒤에는 받는다(고정값 판정 아님)."""
        self.start_run()
        self.slot.on_clock(1 * S)

        self.assertEqual(self.image(3.0), "stale_after_rewind")
        self.slot.on_clock(int(2.6 * S))
        self.assertEqual(self.image(3.0), "accepted")

    def test_s3b_limit_old_frame_within_tolerance_is_accepted(self):
        """S3b(한계 1): 허용 구간 안의 늦은 이전 영상(stamp 1.2 s ≤ clock 1 s + 0.5 s)은 새 영상으로 받아들여진다.

        차단 보장 시험이 아니라 한계를 그대로 확인한다. 그 뒤 진짜 새 영상(1.1 s)은 stamp가 줄어 되감기 경계를 하나 더 만든다.
        """
        self.start_run()
        self.slot.on_clock(1 * S)
        self.apply_all()

        self.assertEqual(self.image(1.2, msg="old"), "accepted")
        take = self.slot.take(0)
        self.assertEqual((take.record.message, take.capture_generation), ("old", 1))

        self.image(1.1, msg="new")
        self.assertEqual([i.kind for i in self.apply_all()], [BoundaryKind.SIM_TIME_REWIND])

    def test_s3c_limit_jitter_misjudged_then_filter_released(self):
        """S3c(한계 2): 새 실행 영상이 clock보다 허용 넘게 앞서면 버려진다. 연속 폐기가 상한을 넘으면 알림과 함께 필터를 푼다."""
        self.start_run()
        self.slot.on_clock(1 * S)
        self.apply_all()

        results = [self.image(2.0 + 0.01 * k) for k in range(31)]
        self.assertTrue(all(r == "stale_after_rewind" for r in results))
        items = self.slot.poll_inputs()
        self.assertEqual([i.kind for i in items], [NoticeKind.REWIND_FILTER_RELEASED])
        self.assertIsNone(items[0].status)
        self.assertEqual(self.image(2.5), "accepted")

    def test_s3d_filter_released_when_clock_passes_old_max(self):
        """S3d: clock이 되감기 전 최대 stamp(9.9 s)를 넘으면 필터가 풀린다."""
        self.start_run()
        self.slot.on_clock(1 * S)
        self.assertTrue(self.slot.stats()["rewind_filter_active"])

        self.slot.on_clock(10 * S)
        self.assertFalse(self.slot.stats()["rewind_filter_active"])
        self.assertEqual(self.image(20.0), "accepted")

    def test_s4_image_stamp_rewind_creates_boundary(self):
        """S4: clock 경계 없이 영상 stamp가 줄면 그 영상에서 경계 1회, 그 영상은 새 epoch."""
        self.make()
        self.image(5.0)
        self.slot.take(0)
        self.image(1.0)

        self.assertEqual(self.slot.take(0).status, TakeStatus.NEEDS_INPUT)
        self.assertEqual([i.kind for i in self.apply_all()], [BoundaryKind.SIM_TIME_REWIND])
        self.assertEqual(self.slot.take(0).record.rx_epoch, 1)

    def test_s5_clock_then_image_single_boundary(self):
        """S5: clock 되감기 뒤 작은 stamp 영상이 와도 경계는 1회."""
        self.make()
        self.slot.on_clock(10 * S)
        self.image(9.9)
        self.slot.on_clock(1 * S)
        self.image(0.9)

        self.assertEqual(len(self.slot.poll_inputs()), 1)

    def test_s5_image_then_clock_single_boundary(self):
        """S5: 영상 stamp 되감기 뒤 clock 되감기가 와도 경계는 1회."""
        self.make()
        self.slot.on_clock(10 * S)
        self.image(9.9)
        self.image(0.9)
        self.slot.on_clock(1 * S)

        self.assertEqual(len(self.slot.poll_inputs()), 1)


class TimeoutAndPublisherTest(SlotBase):
    """수신 중단·첫 영상·재접속·QoS·대기열(S12~S14·S17)."""

    def test_s12_frame_timeout_once_then_resume(self):
        """S12: 마지막 수신 뒤 1 s가 지나면 FRAME_TIMEOUT 경계 1회. 재개 영상은 RECONNECT 경계 뒤 새 세대(사용자 확정 대안 0,
        이전 「재개 세대 유지」 기대를 대체)."""
        self.make()
        self.image(1.0)
        self.slot.take(0)
        self.clock.now += 2 * S
        items = self.slot.poll_inputs()

        self.assertEqual([(i.kind, i.status) for i in items], [(BoundaryKind.FRAME_TIMEOUT, InputStatus.FRAME_TIMEOUT)])
        self.assertEqual(self.slot.poll_inputs(), [])
        self.slot.input_applied(items[0], 0)
        self.image(1.1)

        self.assertEqual([i.kind for i in self.apply_all()], [BoundaryKind.RECONNECT])
        take = self.slot.take(0)
        self.assertEqual((take.record.rx_epoch, take.capture_generation), (2, 1))

    def test_s13_no_first_frame_notice_once(self):
        """S13: 시작 뒤 10 s 동안 영상이 없으면 SOURCE_UNAVAILABLE 알림 1회(epoch 없음)."""
        self.make()
        self.clock.now += 11 * S
        items = self.slot.poll_inputs()

        self.assertEqual([(i.kind, i.epoch, i.status) for i in items],
                         [(NoticeKind.NO_FIRST_FRAME, None, InputStatus.SOURCE_UNAVAILABLE)])
        self.clock.now += 11 * S
        self.assertEqual(self.slot.poll_inputs(), [])

    def test_s14_reconnect_by_gid_set(self):
        """S14(R9): 처음 집합은 확정만, 빈 집합 뒤 같은 집합은 경계 없음, 수신 중 다른 집합이면 RECONNECT 1회."""
        self.make()
        self.slot.on_publisher_snapshot(frozenset())
        self.slot.on_publisher_snapshot(frozenset({"A"}))
        self.image(1.0)
        self.slot.take(0)
        self.slot.on_publisher_snapshot(frozenset())
        self.slot.on_publisher_snapshot(frozenset({"A"}))
        self.assertEqual(self.slot.poll_inputs(), [])

        self.slot.on_publisher_snapshot(frozenset({"A", "B"}))
        self.assertEqual([i.kind for i in self.slot.poll_inputs()], [BoundaryKind.RECONNECT])

    def test_qos_notice_once_per_detail(self):
        """QoS 비호환 알림은 내용마다 한 번(SOURCE_UNAVAILABLE)."""
        self.make()
        self.slot.on_qos_incompatible("qos_incompatible: x")
        self.slot.on_qos_incompatible("qos_incompatible: x")

        items = self.slot.poll_inputs()
        self.assertEqual([(i.kind, i.status) for i in items],
                         [(NoticeKind.QOS_INCOMPATIBLE, InputStatus.SOURCE_UNAVAILABLE)])

    def test_s17_queue_order_and_overflow(self):
        """S17: 대기열은 순서를 보존하고, 상한을 넘으면 poll이 LiveInputOverflow를 올린다."""
        self.make(max_pending_inputs=2)
        self.slot.on_camera_info(info("a"))
        self.slot.on_camera_info(info("b"))
        self.assertEqual([i.profile.calibration_id for i in self.slot.poll_inputs()], ["a", "b"])

        for cal in ("c", "d", "e"):
            self.slot.on_camera_info(info(cal))

        self.assertEqual(self.slot.take(0).status, TakeStatus.NEEDS_INPUT)
        with self.assertRaises(LiveInputOverflow):
            self.slot.poll_inputs()


class ResumePolicyTest(SlotBase):
    """사용자 확정 대안 0(#124, 2026-11-16): 영상 중단 뒤 첫 영상은 발행자 확인 없이 새 경계(RECONNECT)·새 epoch.

    발행자 그래프는 GID 집합으로 비교해 중단 없이 일어난 교체만 보완한다. 같은 복구의 첫 영상 경계와 뒤늦은 그래프 변경은
    흡수 창(`graph_absorb_s`, A 임시) 안에서 한 번만 세대를 바꾼다. 완전 차단 보장이 아니다(짧은 끊김 미감지·그래프 지연·출처 미확정).
    """

    def stream(self, *stamps):
        """영상을 0.1 s 간격으로 넣고 모두 꺼낸다."""
        for t in stamps:
            self.image(t)
            self.slot.take(0)
            self.clock.now += S // 10

    def test_a1_pause_resume_gets_new_generation(self):
        """A1: 중단 반영 뒤 첫 영상은 NEEDS_INPUT, RECONNECT(origin 미확인) 반영 뒤 새 세대로 꺼낸다(재개 세대 유지 정책 대체)."""
        self.make()
        self.stream(1.0)
        self.clock.now += 2 * S
        self.assertEqual([i.kind for i in self.apply_all()], [BoundaryKind.FRAME_TIMEOUT])
        self.image(1.1)

        self.assertEqual(self.slot.take(0).status, TakeStatus.NEEDS_INPUT)
        items = self.apply_all()
        self.assertEqual([i.kind for i in items], [BoundaryKind.RECONNECT])
        self.assertIn("frame_timeout", items[0].detail)
        take = self.slot.take(0)
        self.assertEqual((take.status, take.capture_generation), (TakeStatus.FRAME, 1))

    def test_a2_gap_detected_at_image_time(self):
        """A2: 제어 흐름이 중단을 아직 판정하지 못했어도, 간격 ≥ frame_timeout인 영상은 콜백에서 FRAME_TIMEOUT→RECONNECT 순서로 경계."""
        self.make()
        self.stream(1.0)
        self.clock.now += 2 * S
        self.image(1.1)

        self.assertEqual([i.kind for i in self.apply_all()], [BoundaryKind.FRAME_TIMEOUT, BoundaryKind.RECONNECT])
        self.assertEqual(self.slot.take(0).capture_generation, 1)

    def test_a3_continuous_stream_keeps_generation(self):
        """A3: 끊김 없는 연속 수신은 경계 없이 세대 그대로."""
        self.make()
        self.slot.on_publisher_snapshot(frozenset({"A"}))
        self.stream(*[1.0 + 0.066 * k for k in range(20)])

        self.assertEqual(self.slot.poll_inputs(), [])
        self.image(3.0)
        self.assertEqual(self.slot.take(0).capture_generation, 0)

    def test_a4_gid_set_change_while_streaming(self):
        """A4: 수신 중 GID 집합 변화({A}→{B})는 RECONNECT 1회. 반복 {B}·빈 집합·빈 뒤 같은 {B}는 경계 없음, {C}는 1회."""
        self.make()
        self.slot.on_publisher_snapshot(frozenset({"A"}))
        self.stream(1.0)
        kinds = []

        for gids in ({"B"}, {"B"}, set(), {"B"}, {"C"}):
            self.slot.on_publisher_snapshot(frozenset(gids))
            kinds += [i.kind for i in self.apply_all()]

        self.assertEqual(kinds, [BoundaryKind.RECONNECT, BoundaryKind.RECONNECT])

    def test_a5_same_recovery_late_graph_absorbed(self):
        """A5(#123 순서): 중단 → 새 영상 먼저(경계) → 뒤늦은 {B}는 흡수 창 안이라 추가 경계 없음. 반복 {B}도 없음."""
        self.make()
        self.slot.on_publisher_snapshot(frozenset({"A"}))
        self.stream(1.0)
        self.clock.now += 2 * S
        self.image(5.0)
        first = [i.kind for i in self.apply_all()]
        self.clock.now += S // 2
        self.slot.on_publisher_snapshot(frozenset({"B"}))
        self.slot.on_publisher_snapshot(frozenset({"B"}))

        self.assertEqual(first, [BoundaryKind.FRAME_TIMEOUT, BoundaryKind.RECONNECT])
        self.assertEqual(self.apply_all(), [])
        self.assertEqual(self.slot.stats()["graph_change_absorbed"], 1)

    def test_a6_graph_change_during_gap_no_extra_boundary(self):
        """A6: 끊긴 동안 보인 {}→{B}는 경계를 만들지 않고, 첫 새 영상의 경계 1회만 생긴다."""
        self.make()
        self.slot.on_publisher_snapshot(frozenset({"A"}))
        self.stream(1.0)
        self.clock.now += 2 * S
        self.apply_all()
        self.slot.on_publisher_snapshot(frozenset())
        self.slot.on_publisher_snapshot(frozenset({"B"}))
        self.assertEqual(self.apply_all(), [])

        self.image(5.0)
        self.assertEqual([i.kind for i in self.apply_all()], [BoundaryKind.RECONNECT])
        self.slot.on_publisher_snapshot(frozenset({"B"}))
        self.assertEqual(self.apply_all(), [])

    def test_a7_late_graph_after_absorb_window_is_new_boundary(self):
        """A7(한계): 흡수 창(graph_absorb_s) 뒤에 보인 변화는 새 RECONNECT(같은 복구의 두 번째 세대 전이가 될 수 있음)."""
        self.make(graph_absorb_s=1.0)
        self.slot.on_publisher_snapshot(frozenset({"A"}))
        self.stream(1.0)
        self.clock.now += 2 * S
        self.image(5.0)
        self.apply_all()
        self.slot.take(0)
        self.stream(*[5.1 + 0.066 * k for k in range(15)])  # 끊김 없이 1.5 s 진행(창 1.0 s 초과)
        self.slot.on_publisher_snapshot(frozenset({"B"}))

        self.assertEqual([i.kind for i in self.apply_all()], [BoundaryKind.RECONNECT])

    def test_a8_limit_short_replacement_frames_before_snapshot(self):
        """A8(한계): 끊김 없는 짧은 교체에서 그래프 확인 전에 받은 영상은 이전 epoch로 남고, 확인 뒤에야 경계가 생긴다."""
        self.make()
        self.slot.on_publisher_snapshot(frozenset({"A"}))
        self.stream(1.0)
        self.image(1.1, msg="from-B-before-graph")
        take = self.slot.take(0)
        self.slot.on_publisher_snapshot(frozenset({"B"}))

        self.assertEqual((take.record.message, take.capture_generation), ("from-B-before-graph", 0))
        self.assertEqual([i.kind for i in self.apply_all()], [BoundaryKind.RECONNECT])

    def test_first_snapshot_only_confirms(self):
        """첫 비지 않은 집합은 확정만 하고 경계를 만들지 않는다(시작 전 0도 경계 아님)."""
        self.make()
        self.slot.on_publisher_snapshot(frozenset())
        self.slot.on_publisher_snapshot(frozenset({"A"}))

        self.assertEqual(self.slot.poll_inputs(), [])
        self.assertEqual(self.slot.stats()["publisher_count"], 1)


class CloseAndWaitTest(SlotBase):
    """닫기·대기(S18)."""

    def test_s18_closed_slot(self):
        """S18: 닫은 뒤 take는 CLOSED, 수신은 closed, 두 번 닫아도 된다."""
        self.make()
        self.image(1.0)
        self.slot.close()
        self.slot.close()

        self.assertEqual(self.slot.take(0).status, TakeStatus.CLOSED)
        self.assertEqual(self.image(2.0), "closed")

    def test_take_wakes_on_new_image(self):
        """take는 기다리는 동안 다른 스레드의 새 영상에 깨어난다(상한 안)."""
        self.make()
        timer = threading.Timer(0.05, lambda: self.image(1.0))
        timer.start()
        t0 = time.monotonic()
        take = self.slot.take(2.0)
        timer.join()

        self.assertEqual(take.status, TakeStatus.FRAME)
        self.assertLess(time.monotonic() - t0, 1.5)

    def test_close_wakes_waiting_take(self):
        """close는 기다리는 take를 깨운다."""
        self.make()
        timer = threading.Timer(0.05, self.slot.close)
        timer.start()
        take = self.slot.take(2.0)
        timer.join()

        self.assertEqual(take.status, TakeStatus.CLOSED)


class ConfigTest(unittest.TestCase):
    """설정 검증."""

    def test_invalid_values_rejected(self):
        """양수 아닌 시간·음수 허용·0 상한·빈 frame_id는 ValueError."""
        for over in ({"frame_timeout_s": 0}, {"startup_timeout_s": float("inf")}, {"future_tolerance_ns": -1},
                     {"stale_after_rewind_limit": 0}, {"max_pending_inputs": True}, {"expected_frame_id": ""}):
            with self.subTest(over=over), self.assertRaises(ValueError):
                LiveSlotConfig(**over)


if __name__ == "__main__":
    unittest.main()
