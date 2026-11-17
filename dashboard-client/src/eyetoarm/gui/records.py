"""Exact-byte JSONL audit, bounded display and explicit write-failure backlog."""
from __future__ import annotations

import csv
import base64
import io
import json
import time
from collections import deque
from pathlib import Path
from uuid import uuid4


class SessionLog:
    def __init__(self, path: Path | str | None, keep: int = 500):
        self.path = Path(path) if path is not None else None
        self._rows = deque(maxlen=keep)
        self.write_error = ""
        self._seq = 0
        self._pending = deque()
        self.dropped_count = 0
        self._tail_untrusted = False

    @staticmethod
    def default_path() -> Path:
        return Path(__file__).resolve().parents[3] / "logs" / "gui" / f"{time.time_ns()}-{uuid4().hex}.jsonl"

    def add(self, direction, kind, *, raw=b"", visible=True, **fields):
        self._seq += 1
        row = {"seq": self._seq, "wall_ms": time.time_ns() // 1_000_000,
               "mono_ms": time.monotonic_ns() // 1_000_000, "dir": direction, "kind": kind,
               "raw": raw.decode("utf-8", errors="replace"), "raw_b64": base64.b64encode(raw).decode("ascii"),
               "raw_hex": raw.hex(), **fields}
        if visible:
            self._rows.append(row)
        if self.path is not None:
            if len(self._pending) >= 1024:
                self.retry()
            if len(self._pending) >= 1024:
                self.dropped_count += 1
                self.write_error = f"Audit backlog full; {self.dropped_count} rows not retained"
            else:
                self._pending.append(row)
                self.retry()
        return row

    def retry(self):
        if not self.path or not self._pending or self._tail_untrusted:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = b"".join((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                               for row in self._pending)
            with self.path.open("a+b") as handle:
                offset = handle.tell()
                try:
                    if handle.write(payload) != len(payload):
                        raise OSError("Short audit write")
                    handle.flush()
                except OSError:
                    try:
                        handle.seek(offset)
                        handle.truncate()
                        handle.flush()
                    except OSError:
                        self._tail_untrusted = True
                    raise
            self._pending.clear()
            self.write_error = f"Audit lost {self.dropped_count} rows during earlier failure" if self.dropped_count else ""
        except OSError as error:
            self.write_error = str(error) + ("; audit tail untrusted" if self._tail_untrusted else "")

    def rows(self, direction=None):
        return [dict(row) for row in self._rows if not direction or direction in ("ALL", "전체") or row["dir"] == direction]

    def to_csv(self, direction=None):
        rows = self.rows(direction)
        columns = ("seq", "wall_ms", "mono_ms", "dir", "kind", "command_id", "send_state", "note", "raw_hex")
        output = io.StringIO(newline="")
        writer = csv.writer(output, lineterminator="\r\n")
        writer.writerow(columns)
        for row in rows:
            cells = []
            for column in columns:
                value = "" if row.get(column) is None else str(row[column])
                if value.lstrip().startswith(("=", "+", "-", "@")) or value.startswith(("\t", "\r")):
                    value = "'" + value
                cells.append(value)
            writer.writerow(cells)
        return "\ufeff" + output.getvalue()
