"""통신 스레드와 Qt 화면 사이의 다리.

TCP 처리(asyncio)는 별도 스레드에서 돌고, 화면 갱신은 GUI 스레드에서만 한다(A 협업 요청 D-01).
통신 스레드 → 화면: Qt 시그널(스레드 사이에서는 대기열로 전달됨). 화면 → 통신: run_coroutine_threadsafe.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, Signal

from eyetoarm.tcp.client import ClientConfig

from .session import ClientSession


class Bridge(QObject):
    state_changed = Signal(dict)   # 전체 상태 스냅샷
    log_added = Signal(dict)       # 기록 1행
    replied = Signal(str)          # 조작 결과 문구

    def __init__(self, config: ClientConfig, log_path: Path | None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._loop = asyncio.new_event_loop()
        self._dirty = False
        self.session = ClientSession(config, log_path, self._on_change)
        self.session.log.listeners.append(self.log_added.emit)
        self._thread = threading.Thread(target=self._run, name="eyetoarm-tcp", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _on_change(self) -> None:
        """통신 스레드에서 불린다. 한 번의 처리에서 여러 번 바뀌어도 화면에는 한 번만 보낸다."""
        if not self._dirty:
            self._dirty = True
            self._loop.call_soon(self._flush)

    def _flush(self) -> None:
        self._dirty = False
        self.state_changed.emit(self.session.state())

    def call(self, method: str, *args: Any, on_done: Callable[[str], None] | None = None, **kwargs: Any) -> None:
        """세션의 조작 함수를 통신 스레드에서 실행한다. 결과 문구는 replied 시그널로 돌아온다."""
        async def run() -> None:
            text = await getattr(self.session, method)(*args, **kwargs)
            self._flush()
            if text:
                self.replied.emit(text)
        asyncio.run_coroutine_threadsafe(run(), self._loop)

    def snapshot(self) -> dict[str, Any]:
        return self.session.state()

    def rows(self) -> list[dict[str, Any]]:
        return list(self.session.log.rows)

    def stop(self) -> None:
        """창을 닫을 때. 연결을 끊어도 로봇은 멈추지 않는다."""
        if not self._thread.is_alive():
            return
        async def shutdown() -> None:
            await self.session.client.disconnect("앱 종료")
            tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
            for task in tasks:  # 재접속 대기·타이머가 남지 않게 정리한다
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        future = asyncio.run_coroutine_threadsafe(shutdown(), self._loop)
        try:
            future.result(2)
        except Exception:  # noqa: BLE001 - 종료 중 오류는 삼킨다
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(2)
