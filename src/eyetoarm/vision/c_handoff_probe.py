"""C-side diagnostic consumer for AToCHandoff; logs received metadata without forwarding it."""
from __future__ import annotations

import threading
from typing import Any


class HandoffProbe:
    """Drain A-to-C handoff data on a worker thread and print concise summaries."""

    def __init__(self) -> None:
        self.wakeup = threading.Event()
        self._stop = threading.Event()
        self._handoff: Any = None
        self._thread: threading.Thread | None = None

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
                        print("[C handoff] FRAME"
                              f" msg_seq={view.msg_seq} generation={view.generation} job_seq={view.job_seq}"
                              f" shape={getattr(frame, 'shape', None)} dtype={getattr(frame, 'dtype', None)}"
                              f" display=({self._display_summary(view.display)})", flush=True)
                    else:
                        print(f"[C handoff] FRAME_STALE msg_seq={view.msg_seq} dropped", flush=True)
                if controls_empty and all_empty:
                    break
            if self._stop.is_set():
                break
            self.wakeup.wait(timeout=0.5)