"""A 처리 단계별 시각 진단(기본 꺼짐, #372 ③).

켜면(`enable(path)`) 각 단계가 `mark(단계, image_seq, **값)`으로 단조 시각(ns)과 스레드를 남기고, 기록 스레드가 0.5 s마다
JSONL로 쓴다. 5 s마다 살아 있는 스레드의 OS 번호(native_id)와 이름을 함께 남겨, 바깥 표본기(`/proc/<pid>/task`)의
스레드별 CPU를 이름에 맞출 수 있게 한다. 꺼져 있으면 `mark`는 바로 돌아온다(전역 하나를 읽는 비용뿐).

판정·기록 형식·출력에는 영향이 없다. 쓰기 실패는 진단만 멈춘다(실행을 실패시키지 않음).
"""
from __future__ import annotations

import json
import threading
import time

_rec: "_Recorder | None" = None


class _Recorder:
    """표시를 모아 두었다가 기록 스레드가 쓴다."""

    def __init__(self, path: str, flush_s: float = 0.5, threads_s: float = 5.0) -> None:
        """새 파일을 연다(있으면 FileExistsError)."""
        self._fh = open(path, "x", encoding="utf-8")
        self._lock = threading.Lock()
        self._buf: list[dict] = []
        self._stop = threading.Event()
        self._flush_s, self._threads_s = flush_s, threads_s
        self.error: str | None = None
        self.marks = 0
        self._thread = threading.Thread(target=self._loop, name="eyetoarm-stage-timing", daemon=True)
        self._thread.start()

    def add(self, rec: dict) -> None:
        """한 줄을 쌓는다."""
        with self._lock:
            self._buf.append(rec)
            self.marks += 1

    def _threads(self) -> dict:
        """스레드 이름표 한 줄."""
        return {"kind": "threads", "t_ns": time.monotonic_ns(),
                "threads": {str(t.native_id): t.name for t in threading.enumerate() if t.native_id is not None}}

    def _write(self, extra: list[dict] | None = None) -> None:
        with self._lock:
            buf, self._buf = self._buf, []

        if self.error is not None:
            return

        try:
            for rec in [*(extra or []), *buf]:
                self._fh.write(json.dumps(rec, separators=(",", ":")) + "\n")

            self._fh.flush()
        except (OSError, ValueError) as exc:   # 진단만 멈춘다
            self.error = f"{type(exc).__name__}: {exc}"

    def _loop(self) -> None:
        last_threads = 0.0

        while not self._stop.wait(self._flush_s):
            now = time.monotonic()
            extra = None

            if now - last_threads >= self._threads_s:
                extra, last_threads = [self._threads()], now

            self._write(extra)

    def close(self) -> dict:
        """남은 것을 쓰고 닫는다. 요약(표시 수·오류)."""
        self._stop.set()
        self._thread.join(2.0)
        self._write([self._threads()])

        try:
            self._fh.close()
        except OSError as exc:
            self.error = self.error or f"{type(exc).__name__}: {exc}"

        return {"marks": self.marks, "error": self.error}


def enable(path: str) -> None:
    """진단을 켠다(한 프로세스에 하나). 이미 켜져 있으면 RuntimeError."""
    global _rec

    if _rec is not None:
        raise RuntimeError("stage timing already enabled")

    _rec = _Recorder(path)


def disable() -> dict | None:
    """진단을 끄고 요약을 돌려준다(꺼져 있었으면 None)."""
    global _rec

    rec, _rec = _rec, None

    return None if rec is None else rec.close()


def enabled() -> bool:
    """켜져 있는가."""
    return _rec is not None


def mark(stage: str, image_seq, **values) -> None:
    """단계 하나의 시각. 꺼져 있으면 아무것도 하지 않는다."""
    rec = _rec

    if rec is None:
        return

    rec.add({"kind": "mark", "stage": stage, "image_seq": image_seq, "t_ns": time.monotonic_ns(),
             "tid": threading.get_native_id(), **values})


__all__ = ["disable", "enable", "enabled", "mark"]
