"""로봇 실행 백엔드의 공통 인터페이스.

TCP 서버는 프로토콜과 연결 관리만 맡고, 이 모듈의 백엔드가 실제 로봇
실행 계층(추후 ROS 2 / MoveIt)에 명령을 전달합니다. 현재는 인터페이스만
정의하며, 실행 구현이 연결되지 않은 상태를 성공으로 보고하지 않습니다.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, Protocol


class Dispatch(Protocol):
    """계획한 로봇 목표를 실제 액션 서버에 전송하는 함수."""

    def __call__(self) -> Awaitable[Any]:
        """목표 전송이 완료될 때까지 기다립니다."""
        ...


class ReportStatus(Protocol):
    """실행 상태를 TCP 서버에 전달하는 콜백."""

    def __call__(
        self,
        state: str,
        *,
        stage: str | None = None,
        code: str | None = None,
        detail: str | None = None,
        reach_error: dict[str, float] | None = None,
        goal_id: str | None = None,
        dispatch: Dispatch | None = None,
    ) -> Awaitable[Any]:
        """상태를 기록하고, MOVING이면 dispatch를 통해 목표를 전송합니다."""
        ...


MoveRequest = dict[str, Any]
MoveExecutor = Callable[[MoveRequest, ReportStatus, asyncio.Event], Awaitable[None]]


class RobotBackend(ABC):
    """TCP MOVE 실행 계층이 구현해야 하는 인터페이스.

    구현체는 요청 검증을 다시 수행하기보다, protocol.py가 검증한 MOVE를
    로봇 실행 계층에 맞게 변환하고 계획·실행·도달 확인을 담당합니다.
    위치 단위와 작업 프레임 의미는 TCP 프로토콜 계약을 유지해야 합니다.
    """

    @abstractmethod
    def readiness(self) -> tuple[bool, list[str]]:
        """실행 가능 여부와 NOT_READY 사유 코드를 반환합니다.

        ROS/MoveIt이 준비되지 않았다면 ``(False, ["ROS_DISCONNECTED"])``처럼
        명세에 정의된 사유를 반환해야 합니다. 준비되지 않은 실행기를
        준비 완료로 표시하면 안 됩니다.
        """

    @abstractmethod
    async def execute_move(
        self,
        request: MoveRequest,
        report: ReportStatus,
        cancel_event: asyncio.Event,
    ) -> None:
        """MOVE를 계획·실행하고, terminal 상태까지 report로 알려줍니다.

        정상 완료는 실제 로봇 피드백으로 목표 도달 오차를 확인한 뒤
        ``REACHED``와 ``reach_error``를 보고해야 합니다. 계획 또는 실행
        실패는 TCP 명세의 FAILED stage/code 조합을 사용합니다.
        """

    async def close(self) -> None:
        """백엔드가 소유한 ROS 노드·액션 클라이언트 등을 정리합니다.

        자원이 없는 구현체는 기본 동작을 그대로 사용할 수 있습니다.
        """
        return None


class UnconfiguredRobotBackend(RobotBackend):
    """실제 ROS/MoveIt 백엔드 연결 전의 안전한 기본 구현.

    GET_STATUS는 백엔드 미연결 상태를 보고할 수 있습니다. MOVE는 readiness
    확인에서 거절되어야 하며, 이 메서드가 호출되면 실행하지 않고 예외를
    발생시킵니다. 시험용 성공 응답이나 가짜 로봇 동작은 만들지 않습니다.
    """

    def readiness(self) -> tuple[bool, list[str]]:
        return False, ["ROS_DISCONNECTED"]

    async def execute_move(
        self,
        request: MoveRequest,
        report: ReportStatus,
        cancel_event: asyncio.Event,
    ) -> None:
        del request, report, cancel_event
        raise RuntimeError("실제 ROS/MoveIt 로봇 백엔드가 연결되지 않았습니다")


__all__ = [
    "Dispatch",
    "MoveExecutor",
    "MoveRequest",
    "ReportStatus",
    "RobotBackend",
    "UnconfiguredRobotBackend",
]
