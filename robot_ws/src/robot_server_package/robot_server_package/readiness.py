"""GET_STATUS 응답에 사용할 로봇 준비 상태 정보를 제공합니다.

ROS, Gazebo, 컨트롤러 상태 연결 전까지는 안전하게 준비되지 않은 상태를 반환합니다.
protocol.py가 이 필드에 version, type, server_time_ms를 더해 TCP STATUS로 직렬화합니다.
"""

from typing import TypedDict


class ReadinessFields(TypedDict):
    """TCP STATUS에 포함될 준비 상태 필드."""

    robot_ready: bool
    not_ready_reasons: list[str]
    active_command_id: str | None


def get_readiness_fields() -> ReadinessFields:
    """현재 초기 구현의 준비 상태를 반환합니다.

    실제 ROS/Gazebo readiness 판정이 연결되기 전에는 새 이동을 허용하지 않습니다.
    """
    return {
        "robot_ready": False,
        "not_ready_reasons": ["RECOVERING"],
        "active_command_id": None,
    }