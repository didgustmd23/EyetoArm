"""B center 주석 영상 발행 (B-08, C 답 C-6: 토픽 `/b/center_overlay/image`).

목적: A 실시간 목록 경로(`run_pipeline --b-input list`)에서 B 계산과 **같은 사진**에 B 결과(중심·장축·규격)를 그려
ROS 토픽으로 낸다. C `display_relay`가 이 토픽을 받아 D에 링크 채널 `variant:"center"`로 보낸다.

- 헤더: 원본 사진의 촬영 시각(`frame.ros_stamp_ns`)과 `frame.optical_frame`을 그대로 쓴다 (B 협업 요청 §DB-01,
  C 요청 "원본 프레임의 촬영 시각과 frame_id 유지").
- `b_stage.py`는 ROS 없이 그대로 둔다. ROS는 이 파일에서만, 그리고 `CenterOverlayPublisher`를 만들 때만 불러온다.
- C 목표(`/sort_part`)는 보내지 않는다. 화면 표시용 영상만 낸다.

A 쪽 연결 (A 확인 뒤, run_pipeline 실시간 목록 경로의 목록 소비자 자리 한 곳):

    from eyetoarm.transforms.center_overlay_pub import CenterOverlayPublisher, center_overlay_consumer
    pub = CenterOverlayPublisher()                       # 프로세스에 하나. 끝날 때 pub.close()
    consume = center_overlay_consumer(stage, pub)        # stage.process_frame 과 같은 반환값
"""
from __future__ import annotations

import threading
from typing import Callable

import numpy as np

TOPIC = "/b/center_overlay/image"


def make_image_msg(image: np.ndarray, stamp_ns: int | None, frame_id: str | None, msg_cls=None):
    """BGR 영상 → `sensor_msgs/Image`(bgr8). cv_bridge 없이 만든다. `msg_cls`는 시험 주입용."""
    if msg_cls is None:
        from sensor_msgs.msg import Image as msg_cls

    img = np.ascontiguousarray(image, dtype=np.uint8)

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"bgr8 영상이 아님: shape={img.shape}")

    msg = msg_cls()
    ns = int(stamp_ns or 0)
    msg.header.stamp.sec, msg.header.stamp.nanosec = ns // 1_000_000_000, ns % 1_000_000_000
    msg.header.frame_id = frame_id or ""
    msg.height, msg.width = int(img.shape[0]), int(img.shape[1])
    msg.encoding, msg.is_bigendian, msg.step = "bgr8", 0, int(img.shape[1] * 3)
    msg.data = img.tobytes()
    return msg


class CenterOverlayPublisher:
    """`/b/center_overlay/image` 발행기. A의 ROS 입력과 섞이지 않게 자기 rclpy context를 쓴다 (A `ros_image.py`와 같은 방식).

    QoS: RELIABLE·KEEP_LAST 2 (`b_live_view.py --publish`와 같음). RELIABLE 발행은 RELIABLE·BEST_EFFORT 구독 모두와 맞는다.
    """

    def __init__(self, topic: str = TOPIC, node_name: str = "b_center_overlay", depth: int = 2) -> None:
        import rclpy
        from rclpy.context import Context
        from rclpy.signals import SignalHandlerOptions
        from sensor_msgs.msg import Image

        self._rclpy, self._Image = rclpy, Image
        self._context = Context()
        rclpy.init(context=self._context, signal_handler_options=SignalHandlerOptions.NO)
        self._node = rclpy.create_node(node_name, context=self._context)
        self._pub = self._node.create_publisher(Image, topic, depth)
        self._lock = threading.Lock()
        self.topic = topic
        self.published = 0

    def publish(self, image: np.ndarray, frame) -> None:
        """영상 한 장 발행. `frame`은 원본 FramePacket(`ros_stamp_ns`·`optical_frame`)."""
        msg = make_image_msg(image, frame.ros_stamp_ns, frame.optical_frame, self._Image)

        with self._lock:
            self._pub.publish(msg)
            self.published += 1

    def subscriber_count(self) -> int:
        return self._pub.get_subscription_count()

    def close(self) -> None:
        try:
            self._node.destroy_node()
        finally:
            if self._context.ok():
                self._rclpy.shutdown(context=self._context)


def center_overlay_consumer(stage, publisher, *, guides: bool = True,
                            on_error: Callable[[Exception], None] | None = None) -> Callable:
    """`stage.process_frame`을 감싼 목록 소비자. 반환값은 `process_frame`과 **똑같다**.

    순서: B 계산 → 같은 사진(`envelope.observations.frame.image`)에 같은 프레임 결과 전부(`stage.last_frame`,
    선택·비선택 모두)를 그림 → 발행. 그리기·발행 실패는 B 결과에 영향을 주지 않는다(`on_error`로만 알림, 오류 수는
    `consume.errors`).
    """
    def consume(envelope) -> dict:
        rec = stage.process_frame(envelope)

        try:
            frame = envelope.observations.frame
            publisher.publish(stage.render_center(frame.image, stage.last_frame, guides=guides), frame)
        except Exception as exc:  # noqa: BLE001 - 표시 실패가 B 계산·기록을 막지 않게
            consume.errors += 1
            if on_error is not None:
                on_error(exc)

        return rec

    consume.errors = 0
    return consume
