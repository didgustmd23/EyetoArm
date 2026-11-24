"""Append structured runtime events for each /sort_part execution."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock


class SortEventLog:
    """Thread-safe JSONL log. A write failure is surfaced to the caller."""

    def __init__(self, path: Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def record(self, event: str, **fields) -> None:
        row = {"time_utc": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
        line = json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()


__all__ = ["SortEventLog"]
