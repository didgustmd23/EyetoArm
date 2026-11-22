"""B center 주석 영상 발행 (B-08, C 답 C-6: 토픽 `/b/center_overlay/image`).

목적: A 실시간 목록 경로(`run_pipeline --b-input list`)에서 B 계산과 **같은 사진**에 B 결과(중심·장축·규격)를 그려
ROS 토픽으로 낸다. C `display_relay`가 이 토픽을 받아 D에 링크 채널 `variant:"center"`로 보낸다.

- 헤더: 원본 사진의 촬영 시각(`frame.ros_stamp_ns`)과 `frame.optical_frame`을 그대로 쓴다 (B 협업 요청 §DB-01,
  C 요청 "원본 프레임의 촬영 시각과 frame_id 유지").
- `b_stage.py`는 ROS 없이 그대로 둔다. ROS는 이 파일에서만, 그리고 `CenterOverlayPublisher`를 만들 때만 불러온다.
- C 목표(`/sort_part`)는 보내지 않는다. 화면 표시용 영상만 낸다.
- C 요청(2026-11-20): 같은 영상을 `eyetoarm_interfaces/msg/CenterOverlay`로 `/b/center_overlay`에도 낸다.
  영상 + 같은 사진 번호표(`source_id`·`image_seq`·`capture_generation`·`generation`·`job_seq`·`capture_stamp_ns`·
  `timestamp_base`·`calibration_id`, 값은 `/sort_part` 목표와 똑같음). 잡기·놓기 좌표는 넣지 않는다(목표로만).
  인터페이스 패키지가 없으면 이 토픽만 건너뛰고 `/b/center_overlay/image`는 그대로 낸다.
- C 요청(2026-11-22): 같은 사진의 B 결과 숫자(규격·중심 mm·각도·픽셀 중심·유효/사유)를 `/b/results`
  (`std_msgs/String`, JSON 한 줄)로도 낸다. 번호표는 CenterOverlay와 같아 C가 영상과 짝지을 수 있다.
  D 「현재 측정」 칸 표시용. 집기 명령은 아니다(목표는 `/sort_part`로만).

A 쪽 연결 (A 확인 뒤, run_pipeline 실시간 목록 경로의 목록 소비자 자리 한 곳):

    from eyetoarm.transforms.center_overlay_pub import CenterOverlayPublisher, center_overlay_consumer
    pub = CenterOverlayPublisher()                       # 프로세스에 하나. 끝날 때 pub.close()
    consume = center_overlay_consumer(stage, pub)        # stage.process_frame 과 같은 반환값
"""
from __future__ import annotations

import json
import threading
from typing import Callable

import numpy as np

TOPIC = "/b/center_overlay/image"
OVERLAY_TOPIC = "/b/center_overlay"          # CenterOverlay (영상 + 같은 사진 번호표, C 요청 2026-11-20)
RESULTS_TOPIC = "/b/results"                 # B 결과 숫자 JSON (std_msgs/String, C 요청 2026-11-22)
RESULTS_VERSION = "eyetoarm.b_results/1"


def overlay_keys(envelope) -> dict:
    """목록 봉투 → 번호표. `b_stage.correlation_keys`·`_candidate_view`와 같은 출처라 `/sort_part` 목표와 값이 같다."""
    fo = envelope.observations
    f = fo.frame
    base = getattr(f, "timestamp_base", "SIM_TIME")
    return {
        "run_id": str(f.run_id), "source_id": str(f.source_id), "image_seq": int(f.image_seq),
        "capture_generation": int(f.capture_generation), "generation": int(fo.generation), "job_seq": int(fo.job_seq),
        "capture_stamp_ns": int(f.ros_stamp_ns or 0), "timestamp_base": str(getattr(base, "value", base)),
        "calibration_id": str(getattr(getattr(fo, "capture", None), "calibration_id", None) or ""),
    }


def results_payload(rec: dict, keys: dict) -> dict:
    """`process_frame` 반환값 + 번호표 → `/b/results` JSON 사전. 단위 mm·deg (B가 한 번만 변환, DB-01).

    부품마다: 규격·클래스·유효/사유·중심 (X, Y) mm·각도·길이·픽셀 중심 (u, v)·추적 번호·선택 여부·목표 종류/사유.
    """
    fields = ("instance_index", "part_track_id", "spec", "class_name", "valid", "reason", "reason_owner",
              "center_w_mm", "theta_deg", "theta_valid", "length_mm", "width_mm", "center_px",
              "selected", "goal_kind", "goal_reason")
    return {
        "version": RESULTS_VERSION, **keys,
        "candidate_count": rec.get("candidate_count"), "selected_instance_index": rec.get("selected_instance_index"),
        "results": [{k: r.get(k) for k in fields} for r in rec.get("results", [])],
    }


def make_overlay_msg(image_msg, keys: dict, overlay_cls, kind: str = "accepted"):
    """영상 메시지 + 번호표 → `CenterOverlay`. `overlay_cls`는 시험 주입용."""
    m = overlay_cls()
    m.kind = kind
    for k, v in keys.items():
        setattr(m, k, v)
    m.image = image_msg
    return m


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

    def __init__(self, topic: str = TOPIC, node_name: str = "b_center_overlay", depth: int = 2,
                 overlay_topic: str | None = OVERLAY_TOPIC, results_topic: str | None = RESULTS_TOPIC) -> None:
        import rclpy
        from rclpy.context import Context
        from rclpy.signals import SignalHandlerOptions
        from sensor_msgs.msg import Image

        self._rclpy, self._Image = rclpy, Image
        self._context = Context()
        rclpy.init(context=self._context, signal_handler_options=SignalHandlerOptions.NO)
        self._node = rclpy.create_node(node_name, context=self._context)
        self._pub = self._node.create_publisher(Image, topic, depth)
        self._overlay_cls, self._overlay_pub = None, None

        if overlay_topic:
            try:
                from eyetoarm_interfaces.msg import CenterOverlay
            except ImportError:                      # robot_ws 빌드·source 전이면 번호표 토픽만 건너뜀
                CenterOverlay = None

            if CenterOverlay is not None:
                self._overlay_cls = CenterOverlay
                self._overlay_pub = self._node.create_publisher(CenterOverlay, overlay_topic, depth)

        self._results_pub, self._String = None, None

        if results_topic:
            from std_msgs.msg import String

            self._String = String
            self._results_pub = self._node.create_publisher(String, results_topic, depth)

        self._lock = threading.Lock()
        self.topic, self.overlay_topic = topic, overlay_topic if self._overlay_pub is not None else None
        self.results_topic = results_topic if self._results_pub is not None else None
        self.published = 0

    def publish(self, image: np.ndarray, frame, keys: dict | None = None) -> None:
        """영상 한 장 발행. `frame`은 원본 FramePacket(`ros_stamp_ns`·`optical_frame`). `keys`가 있으면 CenterOverlay도."""
        msg = make_image_msg(image, frame.ros_stamp_ns, frame.optical_frame, self._Image)

        with self._lock:
            self._pub.publish(msg)

            if keys is not None and self._overlay_pub is not None:
                self._overlay_pub.publish(make_overlay_msg(msg, keys, self._overlay_cls))

            self.published += 1

    def publish_results(self, payload: dict) -> None:
        """B 결과 숫자 한 묶음(`results_payload`)을 JSON 한 줄로 발행. 토픽이 꺼져 있으면 아무것도 안 함."""
        if self._results_pub is None:
            return

        msg = self._String()
        msg.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)

        with self._lock:
            self._results_pub.publish(msg)

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
            keys = overlay_keys(envelope)
            publisher.publish(stage.render_center(frame.image, stage.last_frame, guides=guides), frame, keys)
            send_results = getattr(publisher, "publish_results", None)

            if send_results is not None:
                send_results(results_payload(rec, keys))     # 같은 사진의 숫자 (C 요청 2026-11-22)
        except Exception as exc:  # noqa: BLE001 - 표시 실패가 B 계산·기록을 막지 않게
            consume.errors += 1
            if on_error is not None:
                on_error(exc)

        return rec

    consume.errors = 0
    return consume
