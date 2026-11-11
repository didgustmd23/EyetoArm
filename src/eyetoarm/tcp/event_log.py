"""송수신·로컬 사건 기록. JSONL 파일과 화면용 최근 목록을 함께 유지한다. D 상세설계 §A.7.

한 줄 = 사건 1개. 실제 송수신 바이트(raw)를 그대로 남겨 화면 상세·CSV 가 같은 원시 기록을 쓴다.
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable


class EventLog:
    def __init__(self, path: Path | None, keep: int = 500) -> None:
        self.path = path
        self.rows: deque[dict[str, Any]] = deque(maxlen=keep)
        self.listeners: list[Callable[[dict[str, Any]], None]] = []
        self.write_error = ""
        self.routine_count = 0  # 화면 목록에서 뺀 정기 준비 조회 수. 파일에는 모두 남는다
        self._seq = 0
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, direction: str, kind: str, *, raw: bytes | str | None = None, command_id: str | None = None,
            channel: str = "tcp1.1", send_state: str | None = None, note: str = "", routine: bool = False,
            **extra: Any) -> dict[str, Any]:
        """direction: TX(송신 시도)·RX(수신)·LOCAL(화면·연결 사건).

        send_state: sent(소켓에 씀)·unsent(보내지 않음). 수락·도달과는 별개의 사건이다.
        routine: 1초마다 반복되는 준비 조회와 내용이 같은 응답. 파일에는 쓰고 화면 목록에는 넣지 않는다.
        """
        self._seq += 1
        text = raw.decode("utf-8", "replace").rstrip("\n") if isinstance(raw, bytes) else raw
        row = {
            "seq": self._seq,
            "wall_ms": time.time_ns() // 1_000_000,          # 사람이 읽는 시각
            "mono_ms": time.monotonic_ns() // 1_000_000,     # 지연 계산용. 벽시계와 섞어 빼지 않는다
            "dir": direction, "kind": kind, "channel": channel, "command_id": command_id,
            "send_state": send_state, "note": note, "raw": text,
            "bytes": len(raw) if isinstance(raw, bytes) else None, **extra,
        }
        if routine:
            row["routine"] = True
            self.routine_count += 1
        else:
            self.rows.append(row)
        if self.path is not None:
            try:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                self.write_error = ""
            except OSError as error:  # 기록 실패를 숨기지 않는다
                self.write_error = str(error)
        if not routine:
            for listener in list(self.listeners):
                listener(row)
        return row

    def to_csv(self) -> str:
        """화면과 같은 원시 기록으로 CSV 를 만든다."""
        columns = ("seq", "wall_ms", "mono_ms", "dir", "kind", "channel", "command_id", "send_state", "note", "raw")
        lines = [",".join(columns)]
        for row in self.rows:
            cells = []
            for column in columns:
                value = "" if row.get(column) is None else str(row[column])
                cells.append('"' + value.replace('"', '""') + '"')
            lines.append(",".join(cells))
        return "\n".join(lines) + "\n"
