"""Relay selected link variants and raw auxiliary camera topics to D over TCP."""
from __future__ import annotations

import argparse
from collections import deque
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
DEFAULT_CENTER_TOPIC = "/b/center_overlay"
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
        self._client_lock = threading.Lock()
        self._active_client = None
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
            with self._client_lock:
                old, self._active_client = self._active_client, client
            if old is not None:
                try:
                    old.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                old.close()
            print(f"[C 영상] A handoff 접속 {address}", flush=True)
            threading.Thread(target=self._receive, args=(client,), daemon=True).start()

    def _receive(self, client):
        source_run_id = None
        try:
            while not self._stop.is_set():
                with self._client_lock:
                    if self._active_client is not client:
                        return
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
                source_run_id = metadata["run_id"]
                changed = self.relay.update_detect_source(
                    source_run_id, metadata["generation"], metadata["capture_generation"])
                if changed:
                    print("[C 영상] A detect 이전 캐시 무효화: 실행 또는 세대 변경", flush=True)
                metadata["source_run_id"] = source_run_id
                metadata["run_id"] = self.relay.run_id
                self.relay.note_handoff_received()
                # center 대체(#379)용 번호는 선택과 관계없이 최신 것을 남긴다(영상은 버림)
                self.relay.fallback.note_detect(metadata)
                # A의 단방향 handoff는 계속 들어오므로, 선택되지 않은 detect는 표시 캐시에 넣지 않는다.
                if self.relay.selected_link_variant() != "detect":
                    continue
                self.relay.offer("link", pack_packet(metadata, jpeg), variant="detect")
                self.relay.note_handoff_selected()
        except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            if not self._stop.is_set():
                print(f"[C 영상] A handoff 패킷 거부: {exc}", flush=True)
        finally:
            with self._client_lock:
                was_active = self._active_client is client
                if was_active:
                    self._active_client = None
            if was_active:
                self.relay.invalidate_detect_source(source_run_id)
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


FALLBACK_PRODUCER = "relay_fallback"
FALLBACK_NUMBER_KEYS = ("image_seq", "job_seq", "generation", "capture_generation")


class CenterFallback:
    """Decide when the relay substitutes the raw link image under the ``center`` label (#379, A decision).

    B draws center only for frames A hands to it. When the arm hides every candidate, A hands nothing and the
    center view freezes for ~10-25 s. While D has selected ``center`` and no center frame has gone out for
    ``stale_s`` (monotonic clock), the relay sends the raw link image labelled ``center``.

    D drops a link frame whose (generation, capture_generation, image_seq, job_seq) does not move forward, so the
    substitute borrows the numbers of the newest A detect frame. A keeps handing detect frames to the relay even
    when they are not selected, and B numbers center frames with the same A numbers. A center frame that is not
    ahead of the last substitute is held back (D would drop it); the first center ahead of it ends the fallback.
    The detector config is the last center's (same generation must keep one config in D), else A detect's.
    All state is guarded by one lock: the handoff thread, the D request thread and the ROS executor call it.
    """

    def __init__(self, stale_s=1.0, clock=time.monotonic):
        if not stale_s >= 0:
            raise ValueError("stale_s must be >= 0")
        self.stale_s = float(stale_s)
        self._clock = clock
        self._lock = threading.Lock()
        self._selected = False
        self._ref = None             # selection time or last center sent (monotonic s)
        self._numbers = None         # newest A detect numbering
        self._center_config = None
        self._last_order = None      # order of the last substitute sent
        self._active = False
        self._no_numbers_episode = False
        self._detect_at = None       # 마지막 A detect 번호를 받은 단조 시각(진단)
        self.detect_gaps = deque(maxlen=256)   # 1 s 이상 A detect 공백: (벽시계, 공백 s, 앞 순서, 뒤 순서), 진단 기록용
        self.counts = {"fallback_tx": 0, "fallback_starts": 0, "fallback_no_numbers": 0, "center_behind_fallback": 0,
                       "fallback_ends": 0, "detect_gaps_ge1s": 0}

    @staticmethod
    def order_of(meta):
        """D's forward order for one link frame."""
        return tuple(meta[key] for key in ("generation", "capture_generation", "image_seq", "job_seq"))

    def set_selected(self, variant):
        """D chose a link variant. Selecting center starts a fresh wait; anything else stops the fallback."""
        with self._lock:
            selected = variant == "center"
            if selected and not self._selected:
                self._ref = self._clock()
                self._last_order = None
                self._active = False
                self._no_numbers_episode = False
            self._selected = selected
            if not selected:
                self._active = False

    def note_detect(self, metadata):
        """Remember the newest A detect numbering (handoff thread), whether or not detect is selected."""
        numbers = {key: metadata[key] for key in FALLBACK_NUMBER_KEYS}
        numbers.update(calibration_id=metadata.get("calibration_id"), detector_config_id=metadata["detector_config_id"])
        now = self._clock()
        with self._lock:
            previous, at = self._numbers, self._detect_at
            if at is not None and now - at >= 1.0:   # 대체가 멈출 수 있는 A detect 공백(진단)
                self.counts["detect_gaps_ge1s"] += 1
                self.detect_gaps.append((time.time(), round(now - at, 3),
                                         None if previous is None else self.order_of(previous), self.order_of(numbers)))
            self._numbers, self._detect_at = numbers, now

    def forget_detect(self):
        """A detect source changed or disconnected: its numbers can no longer be compared."""
        with self._lock:
            self._numbers = None
            self._last_order = None

    def center_allowed(self, metadata):
        """Called before a center frame goes to D. False holds it back (not ahead of the last substitute)."""
        order = self.order_of(metadata)
        with self._lock:
            if self._last_order is not None and order <= self._last_order:
                self.counts["center_behind_fallback"] += 1
                return False
            if self._active:
                self.counts["fallback_ends"] += 1
            self._active = False
            self._no_numbers_episode = False
            self._last_order = None
            self._center_config = metadata["detector_config_id"]
            self._ref = self._clock()
            return True

    def next_fallback(self):
        """Numbers for one substitute frame now, or None (not due, no numbers, or nothing newer)."""
        with self._lock:
            if not self._selected or self.stale_s <= 0 or self._ref is None:
                return None
            if self._clock() - self._ref < self.stale_s:
                return None
            if self._numbers is None:
                if not self._no_numbers_episode:
                    self.counts["fallback_no_numbers"] += 1
                    self._no_numbers_episode = True
                return None
            numbers = dict(self._numbers)
            order = self.order_of(numbers)
            if self._last_order is not None and order <= self._last_order:
                return None
            if not self._active:
                self.counts["fallback_starts"] += 1
            self._active = True
            self._last_order = order
            self.counts["fallback_tx"] += 1
            if self._center_config is not None:
                numbers["detector_config_id"] = self._center_config
            return numbers

    def take_counts(self):
        """Snapshot of the cumulative counters."""
        with self._lock:
            return dict(self.counts)


def fallback_metadata(run_id, numbers, stamp_sec, stamp_nanosec, frame_id):
    """Packet metadata for one substitute frame (D keys of a center frame, no B result or producer run)."""
    meta = {"channel": "link", "variant": "center", "source_id": "gz_link",
            "image_seq": numbers["image_seq"], "generation": numbers["generation"],
            "calibration_id": numbers.get("calibration_id"), "run_id": run_id,
            "capture_generation": numbers["capture_generation"], "detector_config_id": numbers["detector_config_id"],
            "job_seq": numbers["job_seq"], "producer": FALLBACK_PRODUCER,
            "capture_stamp_ns": int(stamp_sec) * 1_000_000_000 + int(stamp_nanosec), "timestamp_base": "SIM_TIME",
            "stamp_sec": int(stamp_sec), "stamp_nanosec": int(stamp_nanosec)}
    if frame_id:
        meta["frame_id"] = frame_id
    return meta


class LatestFrameServer:
    """Keep only each channel/variant's newest packet; accept D's link-variant request."""

    def __init__(self, host, port, run_id=None, fallback=None):
        self.fallback = fallback or CenterFallback(0.0)
        self._lock = threading.Lock()
        self._latest: dict[tuple[str, str], bytes] = {}
        self._revision: dict[tuple[str, str], int] = {}
        self._client = None
        self._stop = threading.Event()
        self._selected_link_variant = "raw"
        self.run_id = run_id or str(uuid.uuid4())
        self._detect_source = None
        self._center_source = None
        self._center_order = None
        self._retired_center_runs = deque(maxlen=32)
        self._rx_count = {name: 0 for name in CAMERAS}
        self._center_rx_count = 0
        self._handoff_rx_count = 0
        self._handoff_selected_count = 0
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

    def clear_cached_frame(self, channel, variant):
        """Forget a frame so a later selection or D connection cannot replay it."""
        key = (channel, variant)
        with self._lock:
            existed = self._latest.pop(key, None) is not None
            if existed:
                self._revision[key] = self._revision.get(key, 0) + 1
        return existed

    def update_detect_source(self, source_run_id, generation, capture_generation):
        """Invalidate A detect cache when its producer run or simulation epoch changes."""
        identity = (source_run_id, generation, capture_generation)
        with self._lock:
            previous = self._detect_source
            if previous == identity:
                return False
            self._detect_source = identity
            if previous is not None:
                self.fallback.forget_detect()
                key = ("link", "detect")
                if self._latest.pop(key, None) is not None:
                    self._revision[key] = self._revision.get(key, 0) + 1
            return previous is not None

    def invalidate_detect_source(self, source_run_id=None):
        """Clear A detect cache when its active producer disconnects."""
        with self._lock:
            if self._detect_source is None:
                return False
            if source_run_id is not None and self._detect_source[0] != source_run_id:
                return False
            self._detect_source = None
            self.fallback.forget_detect()
            key = ("link", "detect")
            existed = self._latest.pop(key, None) is not None
            if existed:
                self._revision[key] = self._revision.get(key, 0) + 1
            return existed

    def update_center_source(self, producer_run_id, generation, capture_generation, image_seq):
        """Accept only forward B frames; retire superseded producer runs."""
        identity = (producer_run_id, generation, capture_generation)
        order = (generation, capture_generation, image_seq)
        with self._lock:
            if producer_run_id in self._retired_center_runs:
                return None
            previous = self._center_source
            if previous is not None and previous[0] == producer_run_id:
                if order <= self._center_order:
                    return None
            elif previous is not None:
                self._retired_center_runs.append(previous[0])
            if previous == identity:
                self._center_order = order
                return False
            self._center_source = identity
            self._center_order = order
            if previous is not None:
                key = ("link", "center")
                if self._latest.pop(key, None) is not None:
                    self._revision[key] = self._revision.get(key, 0) + 1
            return previous is not None

    def invalidate_center_source(self, producer_run_id=None):
        """Clear B cache; retire a matching run on its end marker."""
        with self._lock:
            if producer_run_id is not None and (self._center_source is None
                                                or self._center_source[0] != producer_run_id):
                return False
            if producer_run_id is not None:
                self._retired_center_runs.append(producer_run_id)
                self._center_source = None
                self._center_order = None
            key = ("link", "center")
            existed = self._latest.pop(key, None) is not None
            if existed:
                self._revision[key] = self._revision.get(key, 0) + 1
            return existed

    def note_received(self, channel):
        with self._lock:
            self._rx_count[channel] += 1

    def note_handoff_received(self):
        with self._lock:
            self._handoff_rx_count += 1

    def note_handoff_selected(self):
        with self._lock:
            self._handoff_selected_count += 1

    def note_center_received(self):
        with self._lock:
            self._center_rx_count += 1

    def select_link_variant(self, variant):
        if variant not in LINK_VARIANTS:
            return False
        with self._lock:
            changed = variant != self._selected_link_variant
            self._selected_link_variant = variant
            self.fallback.set_selected(variant)
            key = ("link", variant)
            # Re-send the cached frame immediately when switching back to a cached variant.
            if changed and key in self._latest:
                self._revision[key] = self._revision.get(key, 0) + 1
        print(f"[C 영상] 링크 표시 선택 적용 variant={variant}", flush=True)
        return True

    def selected_link_variant(self):
        with self._lock:
            return self._selected_link_variant

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
        fallback_prev = 0
        while not self._stop.wait(1.0):
            now = time.monotonic()
            elapsed = now - previous
            previous = now
            with self._lock:
                rx, self._rx_count = self._rx_count, {name: 0 for name in CAMERAS}
                center_rx, self._center_rx_count = self._center_rx_count, 0
                handoff_rx, self._handoff_rx_count = self._handoff_rx_count, 0
                handoff_selected, self._handoff_selected_count = self._handoff_selected_count, 0
                tx, self._tx_count = self._tx_count, {name: 0 for name in CAMERAS}
            rates = " ".join(f"{name}:rx={rx[name]/elapsed:.1f},tx={tx[name]/elapsed:.1f}" for name in CAMERAS)
            fb = self.fallback.take_counts()
            fb_tx = fb["fallback_tx"] - fallback_prev
            fallback_prev = fb["fallback_tx"]
            print(f"[C video FPS] {rates} A_detect_rx={handoff_rx/elapsed:.1f} "
                  f"A_detect_selected={handoff_selected/elapsed:.1f} B_center_selected={center_rx/elapsed:.1f} "
                  f"B_center_fallback={fb_tx/elapsed:.1f} fallback_starts={fb['fallback_starts']} "
                  f"fallback_ends={fb['fallback_ends']} fallback_no_numbers={fb['fallback_no_numbers']} "
                  f"center_behind_fallback={fb['center_behind_fallback']} "
                  f"detect_gaps_ge1s={fb['detect_gaps_ge1s']}", flush=True)
            while self.fallback.detect_gaps:
                wall, gap, before, after = self.fallback.detect_gaps.popleft()
                print(f"[C 영상] A detect 번호 공백 wall={wall:.3f} gap_s={gap} before={before} after={after}", flush=True)

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
    parser.add_argument("--center-topic", default=DEFAULT_CENTER_TOPIC,
                        help="B CenterOverlay message topic (contains the image and B frame identity)")
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument("--center-fallback-s", type=float, default=1.0,
                        help="send the raw link image labelled center after this many seconds without a center "
                             "frame while D selects center (#379); 0 disables")
    args = parser.parse_args()
    if not 1 <= args.quality <= 100:
        parser.error("--quality must be between 1 and 100")
    if not args.center_fallback_s >= 0:
        parser.error("--center-fallback-s must be >= 0")
    import cv2
    import rclpy
    from cv_bridge import CvBridge
    from eyetoarm_interfaces.msg import CenterOverlay
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    run_id = str(uuid.uuid4())
    relay = LatestFrameServer(args.host, args.port, run_id, fallback=CenterFallback(args.center_fallback_s))
    try:
        handoff_ingress = HandoffIngress(relay, args.handoff_host, args.handoff_port)
    except OSError:
        relay.close()
        raise
    threading.Thread(target=relay.report_rates, daemon=True).start()
    bridge = CvBridge()
    sequences = {channel: 0 for channel in CAMERAS}
    # /b/results and CenterOverlay use separate ROS topics. Join them by the complete
    # B frame identity and retain only a small bounded set while the topics race.
    center_result_cache = {}
    center_frame = {"key": None, "metadata": None, "jpeg": None, "counted_key": None}
    latest_link = {"message": None}   # center 선택 중 마지막 링크 원본(변환 전)

    def result_key(payload):
        return tuple(payload.get(key) for key in (
            "run_id", "source_id", "image_seq", "capture_generation", "generation", "job_seq",
            "detector_config_id", "capture_stamp_ns", "timestamp_base", "calibration_id"))

    def publish_center_frame(result_payload=None):
        current = center_frame
        if current["metadata"] is None or current["jpeg"] is None:
            return
        metadata = dict(current["metadata"])
        if result_payload is not None:
            selected = next((item for item in result_payload.get("results", [])
                             if isinstance(item, dict) and item.get("selected") is True), None)
            if selected is not None and type(selected.get("valid")) is bool:
                # D's display result uses the relay run as its session ID. Preserve B's
                # producer run separately while keeping all frame keys identical.
                metadata["detector_config_id"] = result_payload["detector_config_id"]
                metadata["result"] = {
                    **{key: metadata[key] for key in (
                        "source_id", "image_seq", "capture_generation", "generation", "job_seq",
                        "capture_stamp_ns", "timestamp_base", "calibration_id")},
                    **selected,
                    "run_id": metadata["run_id"],
                    "producer_run_id": metadata["producer_run_id"],
                    "detector_config_id": result_payload["detector_config_id"],
                }
        if not relay.fallback.center_allowed(metadata):
            return   # 마지막 대체보다 앞서지 않은 center는 D가 버리므로 보내지 않는다(#379)
        relay.offer("link", pack_packet(metadata, current["jpeg"]), variant="center")
        # Results and CenterOverlay arrive on separate topics and either can arrive first.
        # Count the matched frame here, where both arrival paths converge, exactly once.
        if current["counted_key"] != current["key"]:
            relay.note_center_received()
            current["counted_key"] = current["key"]

    def on_results(message):
        try:
            payload = json.loads(message.data)
            required = ("run_id", "source_id", "image_seq", "capture_generation", "generation", "job_seq",
                        "capture_stamp_ns", "timestamp_base", "calibration_id", "detector_config_id", "results")
            if (not isinstance(payload, dict) or payload.get("version") != "eyetoarm.b_results/1"
                    or any(key not in payload for key in required)
                    or payload["source_id"] != "gz_link"
                    or any(type(payload[key]) is not int or payload[key] < 0 for key in (
                        "image_seq", "capture_generation", "generation", "job_seq", "capture_stamp_ns"))
                    or not isinstance(payload["run_id"], str) or not payload["run_id"]
                    or not isinstance(payload["timestamp_base"], str) or not payload["timestamp_base"]
                    or not isinstance(payload["detector_config_id"], str) or not payload["detector_config_id"]
                    or not isinstance(payload["results"], list)):
                return
            key = result_key(payload)
            center_result_cache[key] = payload
            while len(center_result_cache) > 8:
                center_result_cache.pop(next(iter(center_result_cache)))
            if key == center_frame["key"]:
                publish_center_frame(payload)
        except (ValueError, TypeError, KeyError) as exc:
            node.get_logger().warning(f"B result dropped: {exc}")
    rclpy.init()
    node = rclpy.create_node("c_camera_display_relay")

    def subscribe(channel, topic):
        def on_image(message):
            relay.note_received(channel)
            if channel == "link" and relay.selected_link_variant() != "raw":
                latest_link["message"] = message   # center 대체용: 보낼 때만 변환한다
                return
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

    def on_center_overlay(message):
        # CenterOverlay carries the annotated image and the B frame identity in one ROS message.
        # Only accepted frame messages have a displayable image; status/end markers have no frame.
        if message.kind != "accepted":
            if message.kind == "marker":
                relay.invalidate_center_source(str(message.run_id))
                if center_frame["metadata"] is not None and center_frame["metadata"].get("producer_run_id") == str(message.run_id):
                    center_frame.update(key=None, metadata=None, jpeg=None, counted_key=None)
            return
        try:
            image_message = message.image
            stamp_ns = int(image_message.header.stamp.sec) * 1_000_000_000 + int(image_message.header.stamp.nanosec)
            keys = {
                "source_id": str(message.source_id),
                "image_seq": int(message.image_seq),
                "capture_generation": int(message.capture_generation),
                "generation": int(message.generation),
                "job_seq": int(message.job_seq),
                "capture_stamp_ns": int(message.capture_stamp_ns),
                "timestamp_base": str(message.timestamp_base),
                "calibration_id": str(message.calibration_id),
                "detector_config_id": str(message.detector_config_id),
            }
            if (keys["source_id"] != "gz_link"
                    or any(keys[key] < 0 for key in ("image_seq", "capture_generation", "generation", "job_seq"))
                    or keys["capture_stamp_ns"] < 0
                    or keys["capture_stamp_ns"] != stamp_ns
                    or not message.run_id
                    or not keys["timestamp_base"]
                    or not keys["detector_config_id"]):
                raise ValueError("invalid B frame identity or image timestamp")
            image = bridge.imgmsg_to_cv2(image_message, desired_encoding="bgr8")
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            if not ok:
                return
            seq = keys["image_seq"]
            changed = relay.update_center_source(
                str(message.run_id), keys["generation"], keys["capture_generation"], keys["image_seq"])
            if changed is None:
                return
            if changed:
                print("[C 영상] B center 이전 캐시 무효화: 실행 또는 세대 변경", flush=True)
            # Keep the C relay run_id stable across raw/detect/center switches so D's
            # session guard remains usable. Carry B's original run and correlation keys
            # alongside the display envelope; B sequence/generation values are preserved.
            meta = {
                "channel": "link", "variant": "center",
                "source_id": keys["source_id"], "image_seq": seq,
                "generation": keys["generation"],
                "calibration_id": keys["calibration_id"] or None,
                "run_id": run_id, "capture_generation": keys["capture_generation"],
                "detector_config_id": keys["detector_config_id"], "job_seq": keys["job_seq"],
                "producer": "B", "producer_run_id": str(message.run_id),
                "producer_topic": args.center_topic, "kind": message.kind,
                "capture_stamp_ns": keys["capture_stamp_ns"],
                "timestamp_base": keys["timestamp_base"],
                "frame_id": image_message.header.frame_id,
                "stamp_sec": int(image_message.header.stamp.sec),
                "stamp_nanosec": int(image_message.header.stamp.nanosec),
            }
            frame_key = result_key({**keys, "run_id": str(message.run_id)})
            center_frame.update(key=frame_key, metadata=meta, jpeg=encoded.tobytes(), counted_key=None)
            result_payload = center_result_cache.pop(frame_key, None)
            # Publish only after its matching B result is available. D rejects a repeated
            # (capture_generation, image_seq, job_seq), so a later enriched resend is lost.
            if result_payload is not None:
                publish_center_frame(result_payload)
        except Exception as exc:
            node.get_logger().warning(f"B center frame dropped: {exc}")

    # Raw is subscribed only for the raw selection; detect uses A handoff and center uses B CenterOverlay.
    # Subscription changes stay inside the ROS executor timer.
    link_subscription = {"value": subscribe("link", CAMERAS["link"])}
    applied_link_variant = {"value": "raw"}
    center_subscription = {"value": None}
    results_subscription = {"value": None}
    applied_center_enabled = {"value": False}
    subscriptions = [link_subscription["value"]]
    subscriptions.extend(subscribe(channel, topic) for channel, topic in CAMERAS.items()
                         if channel != "link")

    def update_link_input_subscription():
        selected = relay.selected_link_variant()
        if selected == applied_link_variant["value"]:
            return
        # raw와 center(대체용, #379)는 링크 원본을 받고 detect만 끊는다
        want_raw = selected in ("raw", "center") and (selected == "raw" or args.center_fallback_s > 0)
        if want_raw and link_subscription["value"] is None:
            link_subscription["value"] = subscribe("link", CAMERAS["link"])
            subscriptions.append(link_subscription["value"])
            print(f"[C 영상] link raw 입력 구독 재개 variant={selected}", flush=True)
        elif not want_raw and link_subscription["value"] is not None:
            current = link_subscription["value"]
            node.destroy_subscription(current)
            subscriptions.remove(current)
            link_subscription["value"] = None
            print(f"[C 영상] link raw 입력 구독 중지 variant={selected}", flush=True)
        latest_link["message"] = None
        applied_link_variant["value"] = selected

        center_enabled = selected == "center"
        if center_enabled != applied_center_enabled["value"]:
            if center_enabled:
                center_subscription["value"] = node.create_subscription(
                    CenterOverlay, args.center_topic, on_center_overlay, qos_profile_sensor_data)
                subscriptions.append(center_subscription["value"])
                from std_msgs.msg import String
                results_subscription["value"] = node.create_subscription(
                    String, "/b/results", on_results, qos_profile_sensor_data)
                subscriptions.append(results_subscription["value"])
                print("[C 영상] B center 입력 구독 재개", flush=True)
            else:
                current = center_subscription["value"]
                if current is not None:
                    node.destroy_subscription(current)
                    subscriptions.remove(current)
                    center_subscription["value"] = None
                current_results = results_subscription["value"]
                if current_results is not None:
                    node.destroy_subscription(current_results)
                    subscriptions.remove(current_results)
                    results_subscription["value"] = None
                relay.invalidate_center_source()
                center_frame.update(key=None, metadata=None, jpeg=None)
                center_result_cache.clear()
                print(f"[C 영상] B center 입력 구독 중지 variant={selected}", flush=True)
            applied_center_enabled["value"] = center_enabled

    def send_center_fallback():
        """center 대체(#379): 기한이 지났고 새 A 번호와 링크 원본이 있으면 원본을 center 이름표로 보낸다."""
        message = latest_link["message"]
        if message is None or relay.selected_link_variant() != "center":
            return
        numbers = relay.fallback.next_fallback()
        if numbers is None:
            return
        try:
            image = bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            if not ok:
                return
            meta = fallback_metadata(run_id, numbers, message.header.stamp.sec, message.header.stamp.nanosec,
                                     message.header.frame_id)
            relay.offer("link", pack_packet(meta, encoded.tobytes()), variant="center")
        except Exception as exc:
            node.get_logger().warning(f"center fallback frame dropped: {exc}")

    subscription_control_timer = node.create_timer(0.1, update_link_input_subscription)
    fallback_timer = node.create_timer(0.1, send_center_fallback)
    print(f"[C 영상] {CAMERAS} + B center={args.center_topic} -> {args.host}:{args.port} 대기 중 · "
          "link variant 기본값=raw", flush=True)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_timer(subscription_control_timer)
        node.destroy_timer(fallback_timer)
        handoff_ingress.close()
        relay.close()
        subscriptions.clear()
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
