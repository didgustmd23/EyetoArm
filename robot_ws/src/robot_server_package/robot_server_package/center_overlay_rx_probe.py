"""C-side diagnostic subscriber for B's center-overlay video and metadata.

This node only subscribes and reports compact frame summaries. It does not
control the robot or forward frames to the dashboard.
"""
from __future__ import annotations

import time

import rclpy
from eyetoarm_interfaces.msg import CenterOverlay
from rclpy.node import Node
from sensor_msgs.msg import Image

IMAGE_TOPIC = "/b/center_overlay/image"
DATA_TOPIC = "/b/center_overlay"


def _stamp_ns(image: Image) -> int:
    return int(image.header.stamp.sec) * 1_000_000_000 + int(image.header.stamp.nanosec)


class CenterOverlayRxProbe(Node):
    """Observe B's image-only and image-plus-identifiers topics from C."""

    def __init__(self) -> None:
        super().__init__("c_center_overlay_rx_probe")
        self._image_count = 0
        self._data_count = 0
        self._last_image_report = time.monotonic()
        self._last_image_count = 0
        self.create_subscription(Image, IMAGE_TOPIC, self._on_image, 10)
        self.create_subscription(CenterOverlay, DATA_TOPIC, self._on_data, 10)
        self.get_logger().info(
            f"B 영상/데이터 수신 감시 시작: image={IMAGE_TOPIC}, metadata+image={DATA_TOPIC}"
        )

    def _on_image(self, msg: Image) -> None:
        self._image_count += 1
        now = time.monotonic()
        first = self._image_count == 1
        if first or now - self._last_image_report >= 2.0:
            elapsed = max(now - self._last_image_report, 1e-6)
            delta = self._image_count - self._last_image_count
            fps = delta / elapsed if not first else 0.0
            self.get_logger().info(
                f"B_IMAGE_RX count={self._image_count} fps={fps:.1f} stamp_ns={_stamp_ns(msg)} "
                f"frame={msg.header.frame_id!r} size={msg.width}x{msg.height} "
                f"encoding={msg.encoding} bytes={len(msg.data)}"
            )
            self._last_image_report = now
            self._last_image_count = self._image_count

    def _on_data(self, msg: CenterOverlay) -> None:
        self._data_count += 1
        image = msg.image
        stamp_ns = _stamp_ns(image)
        stamp_matches = msg.capture_stamp_ns == stamp_ns
        keys_ok = (
            bool(msg.run_id)
            and bool(msg.source_id)
            and msg.image_seq >= 0
            and msg.capture_generation >= 0
            and msg.generation >= 0
            and msg.job_seq >= 0
            and stamp_matches
            and bool(msg.timestamp_base)
            and bool(msg.calibration_id)
        )
        self.get_logger().info(
            f"B_DATA_RX count={self._data_count} kind={msg.kind} run_id={msg.run_id} "
            f"source_id={msg.source_id} image_seq={msg.image_seq} "
            f"capture_generation={msg.capture_generation} generation={msg.generation} "
            f"job_seq={msg.job_seq} capture_stamp_ns={msg.capture_stamp_ns} "
            f"stamp_matches_image={stamp_matches} timestamp_base={msg.timestamp_base} "
            f"calibration_id={msg.calibration_id} frame={image.header.frame_id!r} "
            f"size={image.width}x{image.height} encoding={image.encoding} "
            f"bytes={len(image.data)} keys_ok={keys_ok}"
        )


def main(args=None) -> int:
    rclpy.init(args=args)
    node = CenterOverlayRxProbe()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f"B_RX_SUMMARY image_frames={node._image_count} metadata_frames={node._data_count}"
        )
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())