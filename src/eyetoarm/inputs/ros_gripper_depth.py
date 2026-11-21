"""그리퍼 깊이 수신: `/gripper_cam/depth_image`(32FC1)·`camera_info`를 같은 stamp로 짝지어 소유 사본 프레임을 넘긴다.

짝짓기·해석·솎기는 `GripperDepthReceiver`(ROS 없음), rclpy 구독·수신 스레드는 `GripperDepthTransport`가 맡는다.
"""
from __future__ import annotations

import collections
import math
import threading
from typing import Any, Callable

import numpy as np

from eyetoarm.vision.gripper_depth import GripperConfig, GripperDepthFrame

_KEEP = 6


def _stamp_ns(header) -> int:
    """header.stamp → ns."""
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def decode_depth(message) -> np.ndarray:
    """32FC1 Image → 읽기 전용 float32 (H, W) 소유 사본. 형식이 다르면 ValueError."""
    if message.encoding != "32FC1":
        raise ValueError(f"depth encoding must be 32FC1: {message.encoding!r}")

    h, w, step = int(message.height), int(message.width), int(message.step)

    if h <= 0 or w <= 0 or step < w * 4 or len(message.data) < h * step:
        raise ValueError(f"depth size mismatch: {w}x{h} step {step} data {len(message.data)}")

    dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
    raw = np.frombuffer(bytes(message.data[:h * step]), dtype=np.uint8).reshape(h, step)[:, :w * 4]
    out = raw.copy().view(dtype).reshape(h, w).astype(np.float32)
    out.setflags(write=False)

    return out


class GripperDepthReceiver:
    """같은 stamp 깊이·camera_info 짝짓기와 솎기(최대 `max_rate_hz`). 콜백 스레드 하나가 부르고 계수는 잠금 안."""

    def __init__(self, config: GripperConfig, sink: Callable[[GripperDepthFrame], object]) -> None:
        """설정·받는 곳(`LiveWorldTracking.offer_gripper`)."""
        self._config = config
        self._sink = sink
        self._lock = threading.Lock()
        self._depth: collections.OrderedDict[int, Any] = collections.OrderedDict()
        self._info: collections.OrderedDict[int, Any] = collections.OrderedDict()
        self._last_emit_ns: int | None = None
        self._min_gap_ns = int(1e9 / config.max_rate_hz)
        self.counts = {k: 0 for k in ("depth", "info", "emitted", "rate_skipped", "frame_mismatch", "bad_depth", "bad_info",
                                      "old_stamp")}

    def handle_depth(self, message) -> None:
        """깊이 콜백."""
        self._put(self._depth, message, "depth")

    def handle_info(self, message) -> None:
        """camera_info 콜백."""
        self._put(self._info, message, "info")

    def _put(self, table, message, kind: str) -> None:
        """한쪽을 넣고 짝이 생기면 넘긴다."""
        with self._lock:
            self.counts[kind] += 1

            if message.header.frame_id != self._config.optical_frame:
                self.counts["frame_mismatch"] += 1
                return

            stamp = _stamp_ns(message.header)
            table[stamp] = message

            while len(table) > _KEEP:
                table.popitem(last=False)

            if stamp not in self._depth or stamp not in self._info:
                return

            depth, info = self._depth.pop(stamp), self._info.pop(stamp)

            if self._last_emit_ns is not None and stamp <= self._last_emit_ns:
                self.counts["old_stamp"] += 1
                return

            if self._last_emit_ns is not None and stamp - self._last_emit_ns < self._min_gap_ns:
                self.counts["rate_skipped"] += 1
                return

            frame = self._frame(stamp, depth, info)

            if frame is None:
                return

            self._last_emit_ns = stamp
            self.counts["emitted"] += 1

        self._sink(frame)

    def _frame(self, stamp: int, depth, info) -> GripperDepthFrame | None:
        """잠금 안: 해석. 실패는 계수만 올리고 None."""
        K = np.asarray(info.k, dtype=np.float64).reshape(3, 3) if len(info.k) == 9 else None

        if K is None or not np.all(np.isfinite(K)) or K[0, 0] <= 0 or K[1, 1] <= 0:
            self.counts["bad_info"] += 1
            return None

        try:
            d = decode_depth(depth)
        except ValueError:
            self.counts["bad_depth"] += 1
            return None

        if d.shape != (int(info.height), int(info.width)):
            self.counts["bad_info"] += 1
            return None

        K.setflags(write=False)

        return GripperDepthFrame(ros_stamp_ns=stamp, timestamp_ms=stamp // 1_000_000, optical_frame=self._config.optical_frame,
                                 depth_m=d, K=K, width=d.shape[1], height=d.shape[0])

    def stats(self) -> dict:
        """계수 복사."""
        with self._lock:
            return dict(self.counts)


class GripperDepthTransport:
    """rclpy 수신: 자기 Context·노드·구독 2개·수신 스레드 1개. 콜백은 `GripperDepthReceiver`만 부른다."""

    THREAD_NAME = "eyetoarm-gripper-rx"

    def __init__(self, config: GripperConfig, receiver: GripperDepthReceiver) -> None:
        """설정·수신기."""
        self._config = config
        self.receiver = receiver
        self._context: Any = None
        self._node: Any = None
        self._executor: Any = None
        self._thread: threading.Thread | None = None
        self.spin_error: str | None = None

    def start(self) -> None:
        """rclpy를 지연 import해 노드·구독·executor 스레드를 만든다."""
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from rclpy.signals import SignalHandlerOptions
        from sensor_msgs.msg import CameraInfo, Image

        cfg = self._config
        self._context = Context()
        rclpy.init(context=self._context, signal_handler_options=SignalHandlerOptions.NO)

        try:
            self._node = rclpy.create_node("eyetoarm_gripper_depth", context=self._context)
            depth_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=2, reliability=ReliabilityPolicy.BEST_EFFORT,
                                   durability=DurabilityPolicy.VOLATILE)
            info_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5, reliability=ReliabilityPolicy.RELIABLE,
                                  durability=DurabilityPolicy.VOLATILE)
            self._node.create_subscription(Image, cfg.depth_topic, self.receiver.handle_depth, depth_qos)
            self._node.create_subscription(CameraInfo, cfg.camera_info_topic, self.receiver.handle_info, info_qos)
            self._executor = SingleThreadedExecutor(context=self._context)
            self._executor.add_node(self._node)
            self._thread = threading.Thread(target=self._spin, name=self.THREAD_NAME, daemon=True)
            self._thread.start()
        except BaseException:
            self._teardown()
            raise

    def _spin(self) -> None:
        """수신 스레드 본체. 예외는 `spin_error`에 남긴다(그리퍼 입력이 끊기면 추적은 링크만으로 계속)."""
        try:
            self._executor.spin()
        except Exception as exc:  # 스레드 최상단: 원인을 남긴다
            self.spin_error = f"{type(exc).__name__}: {exc}"

    def close(self, limit_s: float) -> None:
        """executor 종료 → 스레드 join → 노드·Context 정리. 실패는 모아서 RuntimeError."""
        errors = []

        if self._executor is not None:
            try:
                self._executor.shutdown(timeout_sec=limit_s)
            except Exception as exc:
                errors.append(f"executor shutdown failed: {type(exc).__name__}: {exc}")

        if self._thread is not None:
            self._thread.join(limit_s)

            if self._thread.is_alive():
                errors.append(f"gripper receive thread did not stop within {limit_s} s")

        errors += self._teardown()

        if errors:
            raise RuntimeError("; ".join(errors))

    def _teardown(self) -> list[str]:
        """노드 해제·Context 종료."""
        errors = []

        if self._node is not None:
            try:
                self._node.destroy_node()
                self._node = None
            except Exception as exc:
                errors.append(f"node destroy failed: {type(exc).__name__}: {exc}")

        if self._context is not None:
            try:
                self._context.try_shutdown()
                self._context = None
            except Exception as exc:
                errors.append(f"context shutdown failed: {type(exc).__name__}: {exc}")

        return errors


def belt_roi_xy_m(frames_doc: dict) -> tuple[float, float, float, float]:
    """frames.yaml `belt`(center·length·width·direction) → W 축 정렬 사각형 (x_min, x_max, y_min, y_max) m."""
    b = frames_doc["belt"]
    cx, cy = (float(v) for v in b["center"])
    dx, dy = (float(v) for v in b["direction"])
    n = math.hypot(dx, dy)

    if not n > 0:
        raise ValueError("belt direction must be non-zero")

    dx, dy = dx / n, dy / n
    half_l, half_w = float(b["length"]) / 2, float(b["width"]) / 2
    corners = [(cx + sl * half_l * dx - sw * half_w * dy, cy + sl * half_l * dy + sw * half_w * dx)
               for sl in (-1, 1) for sw in (-1, 1)]
    xs, ys = [c[0] for c in corners], [c[1] for c in corners]

    return min(xs), max(xs), min(ys), max(ys)


__all__ = ["GripperDepthReceiver", "GripperDepthTransport", "belt_roi_xy_m", "decode_depth"]
