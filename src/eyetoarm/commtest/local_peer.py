"""로컬 모의 상대: 수정하지 않은 C `TcpServer`를 별도 스레드의 asyncio 루프에서 띄운다.

서버 루프는 이 객체의 스레드 소유이고, 다른 스레드는 `set_executor`·`executions`(잠금)와
`run_coroutine_threadsafe`로만 접근한다.
"""
from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

WORKSPACE_MM = {"x": (-1000.0, 1000.0), "y": (-1000.0, 1000.0)}


@dataclass(frozen=True)
class ExecutorSettings:
    """가짜 executor 동작. mode: reach(지연 뒤 REACHED) | hold(취소를 hold_s까지 기다림, 취소면 FAILED CANCELLED)."""

    mode: str = "reach"
    moving_s: float = 0.02
    hold_s: float = 5.0

    def __post_init__(self) -> None:
        """값 검사."""
        if self.mode not in ("reach", "hold"):
            raise ValueError(f"mode must be reach or hold: {self.mode!r}")

        if not (0 <= self.moving_s <= 30 and 0 < self.hold_s <= 60):
            raise ValueError("moving_s must be in [0, 30] and hold_s in (0, 60]")

    def describe(self) -> dict:
        """기록용 사전."""
        return asdict(self)


class LocalPeer:
    """C TcpServer 로컬 모의 상대. `start()` → (host, port), `close()`는 정리 오류 목록을 돌려준다(빈 목록 = 성공)."""

    def __init__(self, repo_root: str | Path, settings: ExecutorSettings | None = None) -> None:
        """저장소 루트와 처음 executor 설정."""
        self.repo_root = Path(repo_root)
        self._lock = threading.Lock()
        self._settings = settings or ExecutorSettings()
        self._executions = 0
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self.temp: Path | None = None
        self.server = None
        self.listener = None
        self.port: int | None = None

    # ------------------------------------------------------------------ 다른 스레드에서 쓰는 조절·관측
    def set_executor(self, settings: ExecutorSettings) -> None:
        """다음 MOVE부터 쓸 executor 설정."""
        with self._lock:
            self._settings = settings

    @property
    def settings(self) -> ExecutorSettings:
        """현재 executor 설정."""
        with self._lock:
            return self._settings

    @property
    def executions(self) -> int:
        """가짜 executor가 불린 수(= 서버가 새로 실행한 명령 수)."""
        with self._lock:
            return self._executions

    # ------------------------------------------------------------------ 서버 루프 안
    async def _execute(self, request, report, cancel) -> None:
        """가짜 move_executor(서버 루프에서 불림)."""
        with self._lock:
            self._executions += 1
            settings = self._settings

        async def dispatch():
            """goal 전송 대역(아무것도 보내지 않음)."""
            return None

        await report("MOVING", goal_id=str(uuid.uuid4()), dispatch=dispatch)

        if settings.mode == "hold":
            try:
                await asyncio.wait_for(cancel.wait(), settings.hold_s)
            except asyncio.TimeoutError:
                await report("REACHED", reach_error={"position_mm": 0.1, "rotation_deg": 0.2})
                return

            await report("FAILED", stage="EXECUTION", code="CANCELLED")
            return

        await asyncio.sleep(settings.moving_s)
        await report("REACHED", reach_error={"position_mm": 0.1, "rotation_deg": 0.2})

    async def _start(self) -> None:
        """서버·리스너 시작(포트 0)."""
        sys.path.insert(0, str(self.repo_root / "robot_ws" / "src" / "robot_server_package"))
        from robot_server_package.tcp_server import TcpServer   # C 원본, 수정 없음

        log_path = self.temp / "c-command.jsonl"
        log_path.touch(exist_ok=False)

        def readiness():
            """항상 준비됨(활성 명령·로그 상태는 서버가 덧붙인다)."""
            return {"robot_ready": True, "not_ready_reasons": [], "active_command_id": None}

        self.server = TcpServer(readiness_provider=readiness, command_log_path=log_path, workspace_mm=WORKSPACE_MM,
                                move_executor=self._execute)
        self.listener = await asyncio.start_server(self.server._handle_client, "127.0.0.1", 0)
        self.port = self.listener.sockets[0].getsockname()[1]

    async def _close(self) -> None:
        """리스너·세션·서버·남은 태스크 정리."""
        if self.listener is not None:
            self.listener.close()
            await self.listener.wait_closed()

        if self.server is not None:
            for session in list(self.server._sessions.values()):
                session.writer.close()

            await self.server.close()

        await asyncio.sleep(0)
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

        for t in tasks:
            t.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------ 수명
    def start(self, timeout_s: float = 5.0) -> tuple[str, int]:
        """스레드·루프를 띄우고 서버를 시작한다. (host, port)."""
        self.temp = Path(tempfile.mkdtemp(prefix="eyetoarm-commtest-peer-"))
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="commtest-local-peer", daemon=True)
        self.thread.start()
        asyncio.run_coroutine_threadsafe(self._start(), self.loop).result(timeout=timeout_s)

        return "127.0.0.1", self.port

    def close(self, timeout_s: float = 5.0) -> list[str]:
        """끝까지 정리하고 실패한 단계의 오류 문자열 목록을 돌려준다(호출자가 실행 실패로 처리)."""
        errors: list[str] = []

        if self.loop is not None and self.loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(self._close(), self.loop).result(timeout=timeout_s)
            except Exception as exc:  # noqa: BLE001 - 정리 경로: 다음 단계를 계속하고 실패를 보고한다
                errors.append(f"peer close: {type(exc).__name__}: {exc}")

            self.loop.call_soon_threadsafe(self.loop.stop)

        if self.thread is not None:
            self.thread.join(timeout_s)

            if self.thread.is_alive():
                errors.append("peer thread did not stop")

        if self.loop is not None and not self.loop.is_running() and not self.loop.is_closed():
            self.loop.close()

        if self.temp is not None:
            shutil.rmtree(self.temp, ignore_errors=True)

            if self.temp.exists():
                errors.append(f"peer temp not removed: {self.temp}")

        return errors
