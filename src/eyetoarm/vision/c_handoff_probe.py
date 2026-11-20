"""C-side AToCHandoff consumer; logs metadata and forwards current AI display frames to C relay."""
from __future__ import annotations

import json
import os
import queue
import socket
import struct
import threading
import time
from typing import Any

from eyetoarm.vision.ai_result import AiSegResult
from eyetoarm.vision.handoff import HandoffView, annotate

MAGIC = b"ETV1"
PACKET_HEADER = struct.Struct(">4sI")
DEFAULT_INGRESS_HOST = "127.0.0.1"
DEFAULT_INGRESS_PORT = 47093


class _DisplayFrameForwarder:
    """Best-effort latest-frame TCP sender to C display_relay; never blocks A's producer."""

    def __init__(self, host: str, port: int) -> None:
        self.host, self.port = host, port
        self._pending: queue.Queue[HandoffView | None] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._last_error_at = 0.0
        self._thread = threading.Thread(target=self._run, name="c-display-frame-forwarder", daemon=True)
        self._thread.start()

    def offer(self, view: HandoffView) -> bool:
        if not isinstance(view.display, AiSegResult):
            return False
        try:
            self._pending.put_nowait(view)
        except queue.Full:
            try:
                self._pending.get_nowait()
            except queue.Empty:
                pass
            self._pending.put_nowait(view)
        return True

    @staticmethod
    def _packet(view: HandoffView) -> bytes:
        import cv2

        frame = view.frame
        metadata = {
            "channel": "link", "variant": "detect", "source_id": frame.source_id,
            "image_seq": frame.image_seq, "generation": view.generation,
            "calibration_id": frame.calibration_id, "run_id": frame.run_id,
            "capture_generation": frame.capture_generation,
            "detector_config_id": view.display.config_sha256, "job_seq": view.job_seq,
        }
        image = annotate(view)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise ValueError("JPEG encoding failed")
        header = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
        body = struct.pack(">I", len(header)) + header + encoded.tobytes()
        if len(body) > 8 * 1024 * 1024:
            raise ValueError("A display frame exceeds ETV1 packet limit")
        return PACKET_HEADER.pack(MAGIC, len(body)) + body

    def _run(self) -> None:
        client = None
        try:
            while not self._stop.is_set():
                item = self._pending.get()
                if item is None or self._stop.is_set():
                    break
                try:
                    packet = self._packet(item)
                    if client is None:
                        client = socket.create_connection((self.host, self.port), timeout=0.5)
                        client.settimeout(1.0)
                    client.sendall(packet)
                except (OSError, ValueError) as exc:
                    if client is not None:
                        try:
                            client.close()
                        except OSError:
                            pass
                        client = None
                    now = time.monotonic()
                    if now - self._last_error_at >= 5.0:
                        print(f"[C handoff] detect 영상 전송 대기/실패: {exc}", flush=True)
                        self._last_error_at = now
        finally:
            if client is not None:
                try:
                    client.close()
                except OSError:
                    pass

    def close(self) -> None:
        self._stop.set()
        try:
            self._pending.put_nowait(None)
        except queue.Full:
            try:
                self._pending.get_nowait()
            except queue.Empty:
                pass
            self._pending.put_nowait(None)
        self._thread.join(timeout=2.0)


class HandoffProbe:
    """Drain A-to-C handoff data on a worker thread and print concise summaries."""

    def __init__(self, *, display_host: str | None = None, display_port: int | None = None) -> None:
        self.wakeup = threading.Event()
        self._stop = threading.Event()
        self._handoff: Any = None
        self._thread: threading.Thread | None = None
        self._forwarder = _DisplayFrameForwarder(
            display_host or os.environ.get("EYETOARM_DISPLAY_INGRESS_HOST", DEFAULT_INGRESS_HOST),
            display_port or int(os.environ.get("EYETOARM_DISPLAY_INGRESS_PORT", DEFAULT_INGRESS_PORT)),
        )

    def notify(self) -> None:
        """Nonblocking callback passed to AToCHandoff."""
        self.wakeup.set()

    def start(self, handoff: Any) -> None:
        if self._thread is not None:
            raise RuntimeError("handoff probe already started")
        self._handoff = handoff
        self._thread = threading.Thread(target=self._consume, name="c-handoff-probe", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.wakeup.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._forwarder.close()

    @staticmethod
    def _display_summary(display: Any) -> str:
        if display is None:
            return "none"
        parts = [type(display).__name__]
        for key in ("model_id", "config_sha256", "infer_ms", "post_ms", "process_ms",
                    "raw_contour_count", "eligible_count"):
            value = getattr(display, key, None)
            if value is not None:
                parts.append(f"{key}={value}")
        instances = getattr(display, "instances", None)
        if instances is not None:
            parts.append("instances=" + repr([
                {k: getattr(item, k, None) for k in ("index", "class_name", "confidence", "box_xyxy", "rejected")}
                for item in instances
            ]))
        candidates = getattr(display, "candidates", None)
        if candidates is not None:
            parts.append("candidates=" + repr([
                {k: getattr(item, k, None) for k in ("index", "area_source_px2", "bbox_source", "eligible", "area_reject_reason")}
                for item in candidates
            ]))
        return " ".join(parts)

    def _log_control(self, msg: Any) -> None:
        kind = getattr(msg.msg_kind, "value", str(msg.msg_kind))
        common = (f"kind={kind} msg_seq={msg.msg_seq} run_id={msg.run_id} source_id={msg.source_id} "
                  f"core_generation={msg.core_generation}")
        if kind == "RESULT":
            obs = msg.observation
            print("[C handoff] " + common +
                  f" image_seq={msg.image_seq} capture_generation={msg.capture_generation}"
                  f" generation={msg.generation} job_seq={msg.job_seq}"
                  f" timestamp_ms={msg.timestamp_ms} timestamp_base={msg.timestamp_base}"
                  f" calibration_id={msg.calibration_id} valid={getattr(obs, 'valid', None)}"
                  f" reason={getattr(obs, 'reason', None)} observed_face={getattr(obs, 'observed_face', None)}"
                  f" candidate_count={getattr(obs, 'candidate_count', None)}"
                  f" selected_index={getattr(obs, 'selected_index', None)}"
                  f" method={getattr(obs, 'method', None)} class_name={getattr(obs, 'class_name', None)}"
                  f" confidence={getattr(obs, 'confidence', None)} process_ms={getattr(obs, 'process_ms', None)}"
                  f" display_only={msg.display_only} display=({self._display_summary(msg.display)})", flush=True)
        else:
            extras = {key: getattr(msg, key, None) for key in
                      ("detector_config_id", "method", "model_id", "status", "detail", "reason", "exit_code", "stop_reason", "applied")
                      if getattr(msg, key, None) is not None}
            if msg.event is not None:
                extras["event"] = repr(msg.event)
            print(f"[C handoff] {common} data={extras}", flush=True)

    def _consume(self) -> None:
        while True:
            # Clear before draining so a producer notification during the drain is retained.
            self.wakeup.clear()
            while True:
                messages, controls_empty = self._handoff.take_control(64)
                for msg in messages:
                    self._log_control(msg)
                view, all_empty = self._handoff.take_view()
                if view is not None:
                    if self._handoff.is_current(view.msg_seq):
                        frame = view.frame.image
                        forwarded = self._forwarder.offer(view)
                        print("[C handoff] FRAME"
                              f" msg_seq={view.msg_seq} generation={view.generation} job_seq={view.job_seq}"
                              f" shape={getattr(frame, 'shape', None)} dtype={getattr(frame, 'dtype', None)}"
                              f" display=({self._display_summary(view.display)})"
                              f" detect_forward_queued={forwarded}", flush=True)
                    else:
                        print(f"[C handoff] FRAME_STALE msg_seq={view.msg_seq} dropped", flush=True)
                if controls_empty and all_empty:
                    break
            if self._stop.is_set():
                break
            self.wakeup.wait(timeout=0.5)
