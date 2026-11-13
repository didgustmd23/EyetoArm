"""GET_STATUS 응답에 사용할 기본 로봇 준비 상태를 제공합니다.

실제 ROS/MoveIt 백엔드가 연결되기 전의 안전한 기본값입니다. 복구 중임을
확인할 수 없는 상태에서 ``RECOVERING``으로 표시하지 않고, 실행 백엔드가
아직 연결되지 않았음을 ``ROS_DISCONNECTED``로 보고합니다. 백엔드 연결 후에는
ExecutionManager가 백엔드의 실제 준비 상태를 TCP STATUS 형식으로 제공합니다.
"""

from typing import TypedDict


class ReadinessFields(TypedDict):
    """TCP STATUS에 포함될 준비 상태 필드."""

    robot_ready: bool
    not_ready_reasons: list[str]
    active_command_id: str | None


def get_readiness_fields() -> ReadinessFields:
    """실행 백엔드 미연결 상태를 반환합니다.

    이 기본 provider는 실제 ROS/Gazebo 상태를 감시하지 않습니다. 백엔드가
    연결되기 전에는 새 MOVE를 허용하지 않으며, 복구 상태는 TCP 서버의
    CommandManager가 실제 recovery 진행 여부에 따라 별도로 관리합니다.
    """
    return {
        "robot_ready": False,
        "not_ready_reasons": ["ROS_DISCONNECTED"],
        "active_command_id": None,
    }
