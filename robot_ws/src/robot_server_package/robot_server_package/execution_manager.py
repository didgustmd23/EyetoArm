"""TCP 서버와 로봇 실행 백엔드 사이의 연결 관리자.

이 모듈은 TCP 프로토콜 처리나 MoveIt 동작을 구현하지 않습니다. 백엔드의
준비 상태를 TCP 서버가 사용하는 형식으로 바꾸고, MOVE 요청·상태 보고·취소
이벤트를 백엔드에 전달합니다.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .protocol import NOT_READY_REASONS
from .readiness import ReadinessFields
from .robot_backend import ReportStatus, RobotBackend


_LOG = logging.getLogger("robot_server_package.execution_manager")


class ExecutionManager:
    """백엔드 하나를 TCP 서버의 readiness/executor 인터페이스에 연결합니다.

    사용 예:

    .. code-block:: python

        manager = ExecutionManager(backend)
        server = TcpServer(
            readiness_provider=manager.readiness_provider,
            move_executor=manager.execute_move,
        )

    백엔드가 아직 구성되지 않았을 때는 이 관리자를 생성하지 말고 TCP 서버에
    실행기를 전달하지 않아야 합니다. 이 경우 tcp_server가
    ``ROS_DISCONNECTED``로 MOVE를 거절합니다.
    """

    def __init__(self, backend: RobotBackend) -> None:
        if not isinstance(backend, RobotBackend):
            raise TypeError("backend는 RobotBackend 구현체여야 합니다")
        self.backend = backend

    def readiness_provider(self) -> ReadinessFields:
        """백엔드 준비 상태를 TCP STATUS용 필드로 반환합니다.

        준비 상태 확인 중 오류가 나면 안전하게 준비되지 않은 것으로
        처리합니다. TcpServer가 active command와 복구 상태 필드를 덧붙입니다.
        """
        try:
            ready, reasons = self.backend.readiness()
            if type(ready) is not bool or not isinstance(reasons, list):
                raise ValueError("backend readiness 형식이 올바르지 않습니다")
            if any(not isinstance(reason, str) or reason not in NOT_READY_REASONS
                   for reason in reasons):
                raise ValueError("backend가 정의되지 않은 NOT_READY 사유를 반환했습니다")
            if len(set(reasons)) != len(reasons):
                raise ValueError("backend readiness 사유가 중복되었습니다")
            if ready and reasons:
                raise ValueError("준비 완료 상태에는 NOT_READY 사유가 없어야 합니다")
            if not ready and not reasons:
                raise ValueError("준비되지 않은 상태에는 사유가 필요합니다")
            return {
                "robot_ready": ready,
                "not_ready_reasons": list(reasons),
                "active_command_id": None,
            }
        except Exception:
            _LOG.exception("Backend readiness 확인 실패; 준비되지 않은 상태로 처리합니다")
            return {
                "robot_ready": False,
                "not_ready_reasons": ["ROS_DISCONNECTED"],
                "active_command_id": None,
            }

    async def execute_move(
        self,
        request: dict[str, Any],
        report: ReportStatus,
        cancel_event: asyncio.Event,
    ) -> None:
        """TCP 서버가 검증한 MOVE와 상태 콜백을 백엔드에 전달합니다.

        상태 전이와 영속 기록은 tcp_server.py/command_manager.py가 맡습니다.
        실제 계획·실행 및 REACHED 판정은 백엔드가 실제 로봇 피드백으로
        수행해야 합니다.
        """
        await self.backend.execute_move(request, report, cancel_event)

    async def close(self) -> None:
        """백엔드의 ROS/액션 자원을 정리합니다."""
        await self.backend.close()


__all__ = ["ExecutionManager"]
