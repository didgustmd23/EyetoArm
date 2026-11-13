"""inputs/base.py의 A 내부 잠정 자료형·식별자 발급기 단위시험.

근거: A 상세설계 §5.1(FramePacket, image_seq 수명, 시각 관계), §5.4(입력 상태),
공통 자료형 관리 초안 §3.1·§3.2·§3.6.
디스크 이미지·ROS·Qt·네트워크·실제 시계에 의존하지 않는다.
"""
import dataclasses
import unittest
import uuid

import numpy as np

from eyetoarm.inputs.base import (
    FrameIdIssuer,
    FramePacket,
    InputStatus,
    InputStatusEvent,
    TimestampBase,
    new_run_id,
)


def _make_packet(image_seq=7, **overrides):
    """합성 4x6 BGR 배열로 모든 필드를 명시한 프레임을 만든다."""
    fields = {
        "image": np.zeros((4, 6, 3), dtype=np.uint8),
        "run_id": "run-test",
        "source_id": "stored0",
        "image_seq": image_seq,
        "timestamp_ms": 1234,
        "timestamp_base": TimestampBase.PC_RECEIVE,
        "received_monotonic_ns": 1_234_567_890,
        "width": 6,
        "height": 4,
        "crop": None,
        "calibration_id": None,
        "capture_generation": 0,
        "optical_frame": None,
        "ros_stamp_ns": None,
        "source_ref": "/tmp/sample.png",
        "source_index": 2,
    }
    fields.update(overrides)
    return FramePacket(**fields)


class EnumContractTest(unittest.TestCase):
    """입력 열거형 값 집합 계약을 확인한다."""

    def test_enum_contracts(self):
        """`TimestampBase`·`InputStatus` 값이 문서의 4개씩이고, 각 멤버는 이름과 값이 같은 문자열이다(TCP 명세서 §4.3, A 상세설계 §5.4)."""
        # TCP 명세서 §4.3, 공통 자료형 초안 §3.1
        self.assertEqual(
            {m.value for m in TimestampBase},
            {"PC_RECEIVE", "PHONE_CAPTURE", "SIM_TIME", "VIDEO_POS"},
        )

        # A 상세설계 §5.4 입력 상태
        self.assertEqual(
            {m.value for m in InputStatus},
            {"SOURCE_UNAVAILABLE", "FRAME_TIMEOUT", "DECODE_FAILED", "END_OF_STREAM"},
        )

        for enum_cls in (TimestampBase, InputStatus):
            for member in enum_cls:
                self.assertIsInstance(member, str)
                self.assertEqual(member.name, member.value)


class FramePacketTest(unittest.TestCase):
    """A `FramePacket`의 필드·순서·불변성을 확인한다(A 상세설계 §5.1)."""

    def test_frame_packet_metadata(self):
        """넘긴 값이 모든 필드에 그대로 들어가고, 영상 배열은 복사 없이 같은 객체다."""
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        packet = _make_packet(
            image=image,
            crop=(10, 20, 6, 4),
            timestamp_base=TimestampBase.VIDEO_POS,
        )

        self.assertIs(packet.image, image)
        self.assertEqual(packet.run_id, "run-test")
        self.assertEqual(packet.source_id, "stored0")
        self.assertEqual(packet.image_seq, 7)
        self.assertEqual(packet.timestamp_ms, 1234)
        self.assertIs(packet.timestamp_base, TimestampBase.VIDEO_POS)
        self.assertEqual(packet.received_monotonic_ns, 1_234_567_890)
        self.assertEqual((packet.width, packet.height), (6, 4))
        self.assertEqual(packet.crop, (10, 20, 6, 4))
        self.assertIsNone(packet.calibration_id)
        self.assertEqual(packet.capture_generation, 0)
        self.assertIsNone(packet.optical_frame)
        self.assertIsNone(packet.ros_stamp_ns)

        # A 내부 재현용 필드(공통 초안에 없음)
        self.assertEqual(packet.source_ref, "/tmp/sample.png")
        self.assertEqual(packet.source_index, 2)

    def test_frame_packet_field_order_and_no_defaults(self):
        """필드 순서는 공통 14개 뒤에 `source_ref`·`source_index`이고, 기본값이 하나도 없다."""
        expected = [
            "image", "run_id", "source_id", "image_seq", "timestamp_ms",
            "timestamp_base", "received_monotonic_ns", "width", "height", "crop",
            "calibration_id", "capture_generation", "optical_frame", "ros_stamp_ns",
            "source_ref", "source_index",
        ]
        fields = dataclasses.fields(FramePacket)

        self.assertEqual([f.name for f in fields], expected)

        for f in fields:
            self.assertIs(f.default, dataclasses.MISSING, f.name)
            self.assertIs(f.default_factory, dataclasses.MISSING, f.name)

    def test_frame_packet_reassignment_rejected(self):
        """필드 재할당은 FrozenInstanceError이지만, 배열 내용의 쓰기 가능 여부는 자료형이 바꾸지 않는다."""
        packet = _make_packet()

        for name, value in (("image_seq", 8), ("timestamp_ms", 0), ("image", None)):
            with self.assertRaises(dataclasses.FrozenInstanceError):
                setattr(packet, name, value)

        # frozen은 필드 재할당만 막는다. 배열 내용 불변은 이 자료형이 보장하지 않는다
        # (버퍼 소유권·독립 복사는 저장 소스 `stored_images.py`의 책임).
        self.assertTrue(packet.image.flags.writeable)


class InputStatusEventTest(unittest.TestCase):
    """입력 사건 자료형을 확인한다(A 상세설계 §5.4)."""

    def test_status_event_is_frameless(self):
        """입력 사건에는 `image_seq`·영상·시각 같은 프레임 필드가 없고, 모든 필드에 기본값이 없다."""
        event = InputStatusEvent(
            run_id="run-test",
            source_id="stored0",
            status=InputStatus.SOURCE_UNAVAILABLE,
            event_monotonic_ns=42,
            detail="file not found",
            source_ref=None,
        )

        self.assertIs(event.status, InputStatus.SOURCE_UNAVAILABLE)
        self.assertIsNone(event.source_ref)

        names = {f.name for f in dataclasses.fields(InputStatusEvent)}

        self.assertEqual(
            names,
            {"run_id", "source_id", "status", "event_monotonic_ns", "detail", "source_ref"},
        )

        for absent in ("image_seq", "image", "timestamp_ms"):
            self.assertFalse(hasattr(event, absent), absent)

        for f in dataclasses.fields(InputStatusEvent):
            self.assertIs(f.default, dataclasses.MISSING, f.name)
            self.assertIs(f.default_factory, dataclasses.MISSING, f.name)


class FrameIdIssuerTest(unittest.TestCase):
    """`image_seq` 발급기와 `new_run_id`를 확인한다(A 상세설계 §5.1 「image_seq 수명」)."""

    def test_issuer_sequence(self):
        """발급기는 0, 1, 2 순으로 내주고 생성 때 받은 `run_id`를 유지한다."""
        issuer = FrameIdIssuer("run-a")
        self.assertEqual([issuer.next_seq() for _ in range(3)], [0, 1, 2])
        self.assertEqual(issuer.run_id, "run-a")

    def test_issuer_run_id_read_only(self):
        """발급기의 `run_id`는 읽기 전용이라 대입하면 AttributeError이고 값이 그대로다."""
        issuer = FrameIdIssuer("run-a")

        with self.assertRaises(AttributeError):
            issuer.run_id = "run-b"

        self.assertEqual(issuer.run_id, "run-a")

    def test_shared_issuer_between_consumers(self):
        """두 소비자가 발급기 하나를 번갈아 쓰면 순번이 겹치지 않고 0~3으로 이어진다."""
        # 실제 소스가 아닌 가상 소비자 두 개가 같은 발급기를 교대로 쓴다(A §5.1 image_seq 수명).
        issuer = FrameIdIssuer("run-a")
        consumer_a = issuer
        consumer_b = issuer

        got = [consumer_a.next_seq(), consumer_b.next_seq(),
               consumer_a.next_seq(), consumer_b.next_seq()]

        self.assertEqual(got, [0, 1, 2, 3])

    def test_reusing_packet_does_not_allocate(self):
        """같은 프레임을 다시 쓰면 순번·시각·세대가 그대로이고, 발급기의 다음 순번도 소비되지 않는다."""
        # 정지 화면 재처리는 같은 FramePacket을 그대로 쓰고 새 순번을 받지 않는다(A §5.1·§5.4).
        issuer = FrameIdIssuer("run-a")
        packet = _make_packet(image_seq=issuer.next_seq())

        first, second = packet, packet

        self.assertIs(first, second)
        self.assertEqual(
            (second.image_seq, second.timestamp_ms, second.received_monotonic_ns,
             second.capture_generation),
            (0, 1234, 1_234_567_890, 0),
        )
        self.assertEqual(issuer.next_seq(), 1)

    def test_new_run_id_format(self):
        """`new_run_id()`는 하이픈 없는 32자리 UUID4 16진 문자열이고, 부를 때마다 다르다."""
        value = new_run_id()

        self.assertIsInstance(value, str)
        self.assertEqual(len(value), 32)

        parsed = uuid.UUID(hex=value)

        self.assertEqual(parsed.version, 4)
        self.assertEqual(parsed.hex, value)
        self.assertNotEqual(new_run_id(), value)


if __name__ == "__main__":
    unittest.main()
