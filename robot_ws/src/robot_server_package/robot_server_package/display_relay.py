"""Relay selected link variants and raw auxiliary camera topics to D over TCP."""
from __future__ import annotations

import argparse
import json
import socket
import struct
import threading
import time
import uuid

MAGIC = b"ETV1"
HEADER = struct.Struct(">4sI")
MAX_PACKET = 8 * 1024 * 1024
MAX_META = 16 * 1024
MAX_REQUEST_LINE = 4096
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 47092
CAMERAS = {"link": "/link_cam/image", "gripper": "/gripper_cam/image", "fixed": "/overview_cam/image"}
LINK_VARIANTS = {"raw", "detect", "center"}


def recv_exact(sock, size):
    """Read exactly one bounded ETV1 field or return None on a closed stream."""
    chunks = bytearray()
    while len(chunks) < size:
        part = sock.recv(size - len(chunks))
        if not part:
            return None
        chunks.extend(part)
    return bytes(chunks)


class HandoffIngress:
    """Receive A's selected-frame JPEG packets from the in-process C handoff consumer."""

    def __init__(self, relay, host, port):
        self.relay = relay
        self._stop = threading.Event()
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((host, port))
        self._listener.listen(1)
        self._listener.settimeout(0.5)
        threading.Thread(target=self._accept, name="a-handoff-ingress", daemon=True).start()

    def _accept(self):
        while not self._stop.is_set():
            try:
                client, address = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            print(f"[C 영상] A handoff 접속 {address}", flush=True)
            threading.Thread(target=self._receive, args=(client,), daemon=True).start()

    def _receive(self, client):
        try:
            while not self._stop.is_set():
                header = recv_exact(client, HEADER.size)
                if header is None:
                    return
                magic, body_size = HEADER.unpack(header)
                if magic != MAGIC or not 4 < body_size <= MAX_PACKET:
                    raise ValueError("invalid ETV1 handoff header")
                body = recv_exact(client, body_size)
                if body is None:
                    return
                meta_size = struct.unpack_from(">I", body)[0]
                if not 0 < meta_size <= MAX_META or 4 + meta_size >= len(body):
                    raise ValueError("invalid ETV1 handoff metadata length")
                metadata = json.loads(body[4:4 + meta_size].decode("utf-8"))
                jpeg = body[4 + meta_size:]
                if (not isinstance(metadata, dict) or metadata.get("channel") != "link"
                        or metadata.get("variant") != "detect" or metadata.get("source_id") != "gz_link"
                        or not jpeg or any(type(metadata.get(key)) is not int or metadata[key] < 0
                                           for key in ("image_seq", "generation", "capture_generation", "job_seq"))
                        or not isinstance(metadata.get("run_id"), str) or not metadata["run_id"]
                        or not isinstance(metadata.get("detector_config_id"), str)
                        or not metadata["detector_config_id"] or "calibration_id" not in metadata):
                    raise ValueError("A handoff only accepts link/detect JPEG frames")
                metadata["source_run_id"] = metadata["run_id"]
                metadata["run_id"] = self.relay.run_id
                self.relay.offer("link", pack_packet(metadata, jpeg), variant="detect")
                self.relay.note_handoff_received()
        except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            if not self._stop.is_set():
                print(f"[C 영상] A handoff 패킷 거부: {exc}", flush=True)
        finally:
            try:
                client.close()
            except OSError:
                pass

    def close(self):
        self._stop.set()
        self._listener.close()


def pack_packet(metadata, jpeg):
    header = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    body = struct.pack(">I", len(header)) + header + jpeg
    if not jpeg or len(body) > MAX_PACKET:
        raise ValueError("invalid or oversized camera packet")
    return HEADER.pack(MAGIC, len(body)) + body


class LatestFrameServer:
    """Keep only each channel/variant's newest packet; accept D's link-variant request."""

    def __init__(self, host, port, run_id=None):
        self._lock = threading.Lock()
        self._latest: dict[tuple[str, str], bytes] = {}
        self._revision: dict[tuple[str, str], int] = {}
        self._client = None
        self._stop = threading.Event()
        self._selected_link_variant = "raw"
        self.run_id = run_id or str(uuid.uuid4())
        self._rx_count = {name: 0 for name in CAMERAS}
        self._handoff_rx_count = 0
        self._tx_count = {name: 0 for name in CAMERAS}
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((host, port))
        self._listener.listen(1)
        self._listener.settimeout(0.5)
        threading.Thread(target=self._accept, daemon=True).start()
        threading.Thread(target=self._send, daemon=True).start()

    def offer(self, channel, packet, variant="raw"):
        if channel not in CAMERAS:
            raise ValueError(f"unknown camera channel: {channel}")
        if channel != "link" and variant != "raw":
            raise ValueError("only the link channel supports detect/center variants")
        if channel == "link" and variant not in LINK_VARIANTS:
            raise ValueError(f"unknown link variant: {variant}")
        key = (channel, variant)
        with self._lock:
            self._latest[key] = packet
            self._revision[key] = self._revision.get(key, 0) + 1

    def note_received(self, channel):
        with self._lock:
            self._rx_count[channel] += 1

    def note_handoff_received(self):
        with self._lock:
            self._handoff_rx_count += 1

    def select_link_variant(self, variant):
        if variant not in LINK_VARIANTS:
            return False
        with self._lock:
            changed = variant != self._selected_link_variant
            self._selected_link_variant = variant
            key = ("link", variant)
            # Re-send the cached frame immediately when switching back to a cached variant.
            if changed and key in self._latest:
                self._revision[key] = self._revision.get(key, 0) + 1
        print(f"[C 영상] 링크 표시 선택 적용 variant={variant}", flush=True)
        return True

    def _handle_request(self, line, address):
        try:
            request = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            print(f"[C 영상] 잘못된 표시 요청 무시 address={address}", flush=True)
            return
        if (not isinstance(request, dict)
                or request.get("type") != "SET_DISPLAY_VARIANT"
                or request.get("channel") != "link"
                or not isinstance(request.get("variant"), str)
                or request.get("variant") not in LINK_VARIANTS
                or set(request) != {"type", "channel", "variant"}):
            print(f"[C 영상] 지원하지 않는 표시 요청 무시 address={address}", flush=True)
            return
        self.select_link_variant(request["variant"])

    def _receive_requests(self, client, address):
        pending = bytearray()
        try:
            while not self._stop.is_set():
                try:
                    chunk = client.recv(2048)
                except socket.timeout:
                    continue
                if not chunk:
                    break
                pending.extend(chunk)
                if len(pending) > MAX_REQUEST_LINE and b"\n" not in pending:
                    pending.clear()
                    print(f"[C 영상] 표시 요청 길이 초과 무시 address={address}", flush=True)
                    continue
                while b"\n" in pending:
                    line, _, rest = pending.partition(b"\n")
                    pending = bytearray(rest)
                    if len(line) > MAX_REQUEST_LINE:
                        print(f"[C 영상] 표시 요청 길이 초과 무시 address={address}", flush=True)
                        continue
                    if line.strip():
                        self._handle_request(line.strip(), address)
        except OSError:
            pass
        finally:
            with self._lock:
                if self._client is client:
                    self._client = None
            try:
                client.close()
            except OSError:
                pass

    def report_rates(self):
        previous = time.monotonic()
        while not self._stop.wait(1.0):
            now = time.monotonic()
            elapsed = now - previous
            previous = now
            with self._lock:
                rx, self._rx_count = self._rx_count, {name: 0 for name in CAMERAS}
                handoff_rx, self._handoff_rx_count = self._handoff_rx_count, 0
                tx, self._tx_count = self._tx_count, {name: 0 for name in CAMERAS}
            rates = " ".join(f"{name}:rx={rx[name]/elapsed:.1f},tx={tx[name]/elapsed:.1f}" for name in CAMERAS)
            print(f"[C video FPS] {rates} A_detect_rx={handoff_rx/elapsed:.1f}", flush=True)

    def _accept(self):
        while not self._stop.is_set():
            try:
                client, address = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            client.settimeout(1.0)
            with self._lock:
                old, self._client = self._client, client
            if old:
                try:
                    old.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                old.close()
            print(f"[C 영상] D 접속 {address}", flush=True)
            threading.Thread(target=self._receive_requests, args=(client, address), daemon=True).start()

    def _send(self):
        sent_revision = {}
        active_client = None
        while not self._stop.wait(0.01):
            with self._lock:
                client = self._client
                keys = [("link", self._selected_link_variant), ("gripper", "raw"), ("fixed", "raw")]
                packets = [(key, self._latest[key], self._revision[key]) for key in keys
                           if key in self._latest and self._revision[key] != sent_revision.get(key, 0)]
            if client is None:
                active_client = None
                sent_revision.clear()
                continue
            if client is not active_client:
                active_client = client
                sent_revision.clear()
                with self._lock:
                    keys = [("link", self._selected_link_variant), ("gripper", "raw"), ("fixed", "raw")]
                    packets = [(key, self._latest[key], self._revision[key]) for key in keys if key in self._latest]
            for key, packet, revision in packets:
                try:
                    client.sendall(packet)
                    sent_revision[key] = revision
                    with self._lock:
                        self._tx_count[key[0]] += 1
                except OSError:
                    with self._lock:
                        if self._client is client:
                            self._client = None
                    try:
                        client.close()
                    except OSError:
                        pass
                    active_client = None
                    sent_revision.clear()
                    break

    def close(self):
        self._stop.set()
        self._listener.close()
        with self._lock:
            client, self._client = self._client, None
        if client:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            client.close()


def main():
    parser = argparse.ArgumentParser(description="Relay selected link variant and raw auxiliary cameras to D")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--handoff-host", default="127.0.0.1",
                        help="local A-to-C display-frame ingress bind address")
    parser.add_argument("--handoff-port", type=int, default=47093,
                        help="local A-to-C display-frame ingress port")
    parser.add_argument("--quality", type=int, default=80)
    args = parser.parse_args()
    if not 1 <= args.quality <= 100:
        parser.error("--quality must be between 1 and 100")
    import cv2
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    run_id = str(uuid.uuid4())
    relay = LatestFrameServer(args.host, args.port, run_id)
    try:
        handoff_ingress = HandoffIngress(relay, args.handoff_host, args.handoff_port)
    except OSError:
        relay.close()
        raise
    threading.Thread(target=relay.report_rates, daemon=True).start()
    bridge = CvBridge()
    sequences = {channel: 0 for channel in CAMERAS}
    rclpy.init()
    node = rclpy.create_node("c_camera_display_relay")

    def subscribe(channel, topic):
        def on_image(message):
            relay.note_received(channel)
            try:
                image = bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
                ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                if not ok:
                    return
                sequences[channel] += 1
                seq = sequences[channel]
                meta = {"channel": channel, "variant": "raw", "source_id": f"gz_{channel}", "image_seq": seq,
                        "generation": 0, "calibration_id": None, "run_id": run_id, "capture_generation": 0,
                        "detector_config_id": "raw-camera-v1", "job_seq": seq,
                        "stamp_sec": int(message.header.stamp.sec), "stamp_nanosec": int(message.header.stamp.nanosec)}
                relay.offer(channel, pack_packet(meta, encoded.tobytes()), variant="raw")
            except Exception as exc:
                node.get_logger().warning(f"{channel} frame dropped: {exc}")
        return node.create_subscription(Image, topic, on_image, qos_profile_sensor_data)

    subscriptions = [subscribe(channel, topic) for channel, topic in CAMERAS.items()]
    print(f"[C 영상] {CAMERAS} -> {args.host}:{args.port} 대기 중 · 기본 link variant=raw", flush=True)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        handoff_ingress.close()
        relay.close()
        subscriptions.clear()
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
