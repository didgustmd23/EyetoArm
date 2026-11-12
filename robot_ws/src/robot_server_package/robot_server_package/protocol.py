"""TCP 1.1 요청 파싱과 STATUS 응답 구성.

요청 검증과 STATUS 메시지 생성만 담당합니다. 소켓 프레이밍·전송은
tcp_server.py, 명령 ID·상태 전이·로그 색인은 command_manager.py가 담당합니다.
"""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from typing import Any

from .readiness import ReadinessFields

VERSION = "1.1"
MAX_SAFE_INT = 9_007_199_254_740_991
IDENT_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
TIMESTAMP_BASES = {"PC_RECEIVE", "PHONE_CAPTURE", "SIM_TIME", "VIDEO_POS"}
NOT_READY_REASONS = {
    "COMMAND_ACTIVE", "ROS_DISCONNECTED", "SIM_PAUSED", "CONTROLLER_INACTIVE",
    "TRAJECTORY_ACTIVE", "ACTION_STATUS_UNKNOWN", "FEEDBACK_STALE", "NOT_STOPPED",
    "SCENE_NOT_READY", "RECOVERING", "RECOVERY_FAILED", "LOG_UNAVAILABLE",
}
COMMAND_STATES = {"ACK", "PLANNING", "MOVING", "REACHED", "FAILED"}
REJECTION_CODES = {
    "BAD_FORMAT", "BAD_VALUE", "OUT_OF_WORKSPACE", "ID_CONFLICT", "BUSY",
    "NOT_READY", "UNKNOWN_COMMAND",
}
FAILURE_CODES = {
    "PLAN_FAILED", "EXEC_FAILED", "FEEDBACK_LOST", "REACH_TOLERANCE", "CANCELLED",
    "CANCEL_UNCONFIRMED", "SERVER_RESTARTED", "LOG_WRITE_FAILED",
}


class ProtocolError(ValueError):
    """수신한 JSON 또는 요청 형식이 TCP 명세를 따르지 않습니다."""

    code = "BAD_FORMAT"


class BadValueError(ProtocolError):
    """형식은 맞지만 프로토콜상 허용 값 범위를 벗어난 요청입니다."""

    code = "BAD_VALUE"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"중복 JSON 키: {key}")
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> None:
    raise ProtocolError(f"허용되지 않는 JSON 숫자: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ProtocolError("유한하지 않은 실수")
    return parsed


def _parse_safe_integer(value: str) -> int:
    # JSON 정수의 문법은 여기서 보존하고, 필드별 범위는 스키마 규칙에
    # 따라 validate_move_request에서 BAD_VALUE로 분류합니다.
    return int(value)


def _validate_uuid_v4(value: Any, field: str = "command_id") -> str:
    if not isinstance(value, str):
        raise ProtocolError(f"{field}는 UUID v4 문자열이어야 합니다")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as error:
        raise ProtocolError(f"{field}가 유효한 UUID가 아닙니다") from error
    if parsed.version != 4 or str(parsed) != value:
        raise ProtocolError(f"{field}는 소문자 하이픈 UUID v4 형식이어야 합니다")
    return value


def valid_command_id_from_request(request: dict[str, Any]) -> str | None:
    """오류 응답에 포함할 수 있는 유효한 command_id만 반환합니다."""
    try:
        return _validate_uuid_v4(request.get("command_id"))
    except ProtocolError:
        return None


def _require_finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{field}는 숫자여야 합니다")
    try:
        number = float(value)
    except OverflowError as error:
        raise ProtocolError(f"{field}가 binary64 범위를 벗어납니다") from error
    if not math.isfinite(number):
        raise ProtocolError(f"{field}는 유한한 숫자여야 합니다")
    return number


def _validate_header(request: dict[str, Any], expected_type: str) -> None:
    if request.get("version") != VERSION or request.get("type") != expected_type:
        raise ProtocolError(f"version은 {VERSION}, type은 {expected_type}이어야 합니다")


def validate_request(request: dict[str, Any]) -> dict[str, Any]:
    """요청 type에 맞는 검증기를 선택하고 검증된 객체를 반환합니다."""
    request_type = request.get("type")
    if request_type == "GET_STATUS":
        validate_get_status_request(request)
    elif request_type == "MOVE":
        validate_move_request(request)
    elif request_type == "CANCEL":
        validate_cancel_request(request)
    else:
        raise ProtocolError("지원하지 않는 요청 type")
    return request


def parse_request(line: bytes) -> dict[str, Any]:
    """LF를 제거한 요청 한 줄을 UTF-8 JSON 객체로 파싱합니다."""
    if line.endswith(b"\r"):
        line = line[:-1]

    try:
        text = line.decode("utf-8")
        request = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
            parse_float=_parse_finite_float,
            parse_int=_parse_safe_integer,
        )
    except ProtocolError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ProtocolError(f"요청 JSON 파싱 실패: {error}") from error

    if not isinstance(request, dict):
        raise ProtocolError("요청 최상위 값은 JSON 객체여야 합니다")
    return request


def validate_get_status_request(request: dict[str, Any]) -> str | None:
    """GET_STATUS 형식을 확인하고 command_id가 있으면 반환합니다."""
    allowed_fields = {"version", "type", "command_id"}
    if set(request) - allowed_fields:
        raise ProtocolError("GET_STATUS에 정의되지 않은 필드가 있습니다")
    _validate_header(request, "GET_STATUS")

    if "command_id" not in request:
        return None
    return _validate_uuid_v4(request["command_id"])


def validate_cancel_request(request: dict[str, Any]) -> str:
    """CANCEL 요청의 스키마를 확인하고 대상 command_id를 반환합니다."""
    if set(request) != {"version", "type", "command_id"}:
        raise ProtocolError("CANCEL 필수 필드 누락 또는 정의되지 않은 필드")
    _validate_header(request, "CANCEL")
    return _validate_uuid_v4(request["command_id"])


def validate_move_request(request: dict[str, Any]) -> dict[str, Any]:
    """TCP 1.1 §4.2에 따라 MOVE 요청을 검증합니다.

    문법·필드 위반은 ProtocolError(BAD_FORMAT), 값 범위 위반은
    BadValueError(BAD_VALUE)로 구분합니다.
    """
    # 명세 §7.2.1: 한 요청에 형식 위반과 값 위반이 함께 있으면
    # 형식 위반(BAD_FORMAT)이 우선하므로 먼저 형식만 전부 확인합니다.
    _validate_move_format(request)

    common_required = {
        "version", "type", "command_id", "mode", "frame", "source_id",
        "image_seq", "timestamp_ms", "timestamp_base", "calibration_id", "valid",
        "position_mm",
    }
    allowed = common_required | {"theta_deg", "orientation_xyzw"}
    if set(request) - allowed:
        raise ProtocolError("MOVE에 정의되지 않은 필드가 있습니다")
    if common_required - set(request):
        raise ProtocolError("MOVE 필수 필드가 누락되었습니다")
    _validate_header(request, "MOVE")
    _validate_uuid_v4(request["command_id"])

    mode = request["mode"]
    if not isinstance(mode, str) or mode not in {"PLANAR", "POSE_6D"}:
        raise ProtocolError("mode는 PLANAR 또는 POSE_6D이어야 합니다")
    if request["frame"] != "work":
        raise ProtocolError("frame은 work이어야 합니다")
    for key in ("source_id", "calibration_id"):
        if not isinstance(request[key], str) or not IDENT_PATTERN.fullmatch(request[key]):
            raise ProtocolError(f"{key} 식별 문자열 형식이 올바르지 않습니다")
    if not isinstance(request["timestamp_base"], str) or request["timestamp_base"] not in TIMESTAMP_BASES:
        raise ProtocolError("timestamp_base 값이 정의되어 있지 않습니다")
    if not isinstance(request["valid"], bool):
        raise ProtocolError("valid는 불리언이어야 합니다")
    if request["valid"] is False:
        raise BadValueError("valid=false인 MOVE는 허용되지 않습니다")

    for key in ("image_seq", "timestamp_ms"):
        value = request[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ProtocolError(f"{key}는 정수여야 합니다")
        if value < 0 or value > MAX_SAFE_INT:
            raise BadValueError(f"{key} 허용 범위를 벗어났습니다")

    position = request["position_mm"]
    if not isinstance(position, list):
        raise ProtocolError("position_mm는 배열이어야 합니다")
    expected_position_len = 2 if mode == "PLANAR" else 3
    if len(position) != expected_position_len:
        raise BadValueError(f"{mode}의 position_mm 길이는 {expected_position_len}이어야 합니다")
    for index, value in enumerate(position):
        _require_finite_number(value, f"position_mm[{index}]")

    if mode == "PLANAR":
        if "orientation_xyzw" in request:
            raise ProtocolError("PLANAR에는 orientation_xyzw를 넣을 수 없습니다")
        if "theta_deg" not in request:
            raise ProtocolError("PLANAR에는 theta_deg가 필요합니다")
        theta = _require_finite_number(request["theta_deg"], "theta_deg")
        if not 0 <= theta < 180:
            raise BadValueError("theta_deg는 0 이상 180 미만이어야 합니다")
    else:
        if "theta_deg" in request:
            raise ProtocolError("POSE_6D에는 theta_deg를 넣을 수 없습니다")
        if "orientation_xyzw" not in request:
            raise ProtocolError("POSE_6D에는 orientation_xyzw가 필요합니다")
        quaternion = request["orientation_xyzw"]
        if not isinstance(quaternion, list):
            raise ProtocolError("orientation_xyzw는 배열이어야 합니다")
        if len(quaternion) != 4:
            raise BadValueError("orientation_xyzw 길이는 4여야 합니다")
        values = [
            _require_finite_number(value, f"orientation_xyzw[{index}]")
            for index, value in enumerate(quaternion)
        ]
        if any(value < -1 or value > 1 for value in values[:3]) or not 0 <= values[3] <= 1:
            raise BadValueError("orientation_xyzw 성분 범위를 벗어났습니다")
        norm = math.sqrt(sum(value * value for value in values))
        if abs(norm - 1.0) > 1e-6:
            raise BadValueError("orientation_xyzw는 단위 쿼터니언이어야 합니다")
    return request


def _validate_move_format(request: dict[str, Any]) -> None:
    """MOVE의 키·타입·열거형·모드 조합을 값 범위 검사 전에 확인합니다."""
    common_required = {
        "version", "type", "command_id", "mode", "frame", "source_id",
        "image_seq", "timestamp_ms", "timestamp_base", "calibration_id", "valid",
        "position_mm",
    }
    allowed = common_required | {"theta_deg", "orientation_xyzw"}
    if set(request) - allowed or common_required - set(request):
        raise ProtocolError("MOVE 필수 필드 누락 또는 정의되지 않은 필드")
    _validate_header(request, "MOVE")
    _validate_uuid_v4(request["command_id"])

    mode = request["mode"]
    if not isinstance(mode, str) or mode not in {"PLANAR", "POSE_6D"}:
        raise ProtocolError("mode는 PLANAR 또는 POSE_6D이어야 합니다")
    if request["frame"] != "work":
        raise ProtocolError("frame은 work이어야 합니다")
    for key in ("source_id", "calibration_id"):
        if not isinstance(request[key], str) or not IDENT_PATTERN.fullmatch(request[key]):
            raise ProtocolError(f"{key} 식별 문자열 형식이 올바르지 않습니다")
    if not isinstance(request["timestamp_base"], str) or request["timestamp_base"] not in TIMESTAMP_BASES:
        raise ProtocolError("timestamp_base 값이 정의되어 있지 않습니다")
    if not isinstance(request["valid"], bool):
        raise ProtocolError("valid는 불리언이어야 합니다")
    for key in ("image_seq", "timestamp_ms"):
        if isinstance(request[key], bool) or not isinstance(request[key], int):
            raise ProtocolError(f"{key}는 정수여야 합니다")

    position = request["position_mm"]
    if not isinstance(position, list):
        raise ProtocolError("position_mm는 배열이어야 합니다")
    for index, value in enumerate(position):
        _require_finite_number(value, f"position_mm[{index}]")

    if mode == "PLANAR":
        if "theta_deg" not in request or "orientation_xyzw" in request:
            raise ProtocolError("PLANAR은 theta_deg만 사용하고 orientation_xyzw는 넣지 않습니다")
        _require_finite_number(request["theta_deg"], "theta_deg")
    else:
        if "orientation_xyzw" not in request or "theta_deg" in request:
            raise ProtocolError("POSE_6D은 orientation_xyzw만 사용하고 theta_deg는 넣지 않습니다")
        quaternion = request["orientation_xyzw"]
        if not isinstance(quaternion, list):
            raise ProtocolError("orientation_xyzw는 배열이어야 합니다")
        for index, value in enumerate(quaternion):
            _require_finite_number(value, f"orientation_xyzw[{index}]")


def same_move_content(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """TCP 1.1 §7.3에 따라 두 검증된 MOVE의 내용이 같은지 비교합니다.

    command_id는 호출자가 같은지 먼저 확인해야 합니다. 이 함수는 요청의
    나머지 내용만 비교하며, 잘못된 MOVE가 들어오면 ProtocolError를 냅니다.
    """
    validate_move_request(first)
    validate_move_request(second)

    exact_fields = (
        "mode", "frame", "source_id", "image_seq", "timestamp_ms",
        "timestamp_base", "calibration_id", "valid",
    )
    if any(first[field] != second[field] for field in exact_fields):
        return False

    if any(
        abs(float(left) - float(right)) > 0.01
        for left, right in zip(first["position_mm"], second["position_mm"])
    ):
        return False

    if first["mode"] == "PLANAR":
        angle_difference = abs(float(first["theta_deg"]) - float(second["theta_deg"]))
        if min(angle_difference, 180.0 - angle_difference) > 0.01:
            return False
        return True

    q1 = [float(value) for value in first["orientation_xyzw"]]
    q2 = [float(value) for value in second["orientation_xyzw"]]
    norm1 = math.hypot(*q1)
    norm2 = math.hypot(*q2)
    q1 = [value / norm1 for value in q1]
    q2 = [value / norm2 for value in q2]
    dot = sum(left * right for left, right in zip(q1, q2))
    sign = -1.0 if dot < 0 else 1.0
    difference_norm = math.hypot(*(left - sign * right for left, right in zip(q1, q2)))
    sum_norm = math.hypot(*(left + sign * right for left, right in zip(q1, q2)))
    rotation_difference_deg = math.degrees(4.0 * math.atan2(difference_norm, sum_norm))
    return rotation_difference_deg <= 0.01


def _common_status_fields(readiness: ReadinessFields) -> dict[str, Any]:
    ready = readiness["robot_ready"]
    reasons = readiness["not_ready_reasons"]
    active_command_id = readiness["active_command_id"]

    if not isinstance(ready, bool) or not isinstance(reasons, list):
        raise ValueError("readiness 필드 형식이 올바르지 않습니다")
    if any(not isinstance(reason, str) or reason not in NOT_READY_REASONS for reason in reasons):
        raise ValueError("정의되지 않은 not_ready_reasons가 있습니다")
    if len(set(reasons)) != len(reasons):
        raise ValueError("not_ready_reasons는 중복될 수 없습니다")
    if active_command_id is not None:
        _validate_uuid_v4(active_command_id, "active_command_id")
    if ready and reasons:
        raise ValueError("robot_ready=true이면 not_ready_reasons는 비어 있어야 합니다")
    if not ready and not reasons:
        raise ValueError("robot_ready=false이면 not_ready_reasons가 필요합니다")
    if active_command_id is not None and ready:
        raise ValueError("active command가 있으면 robot_ready는 false여야 합니다")
    if active_command_id is not None and "COMMAND_ACTIVE" not in reasons:
        raise ValueError("active command가 있으면 COMMAND_ACTIVE 사유가 필요합니다")
    if (active_command_id is None) == ("COMMAND_ACTIVE" in reasons):
        raise ValueError("COMMAND_ACTIVE와 active_command_id는 함께 설정되어야 합니다")

    return {
        "version": VERSION,
        "type": "STATUS",
        "robot_ready": ready,
        "not_ready_reasons": list(reasons),
        "active_command_id": active_command_id,
        "server_time_ms": time.monotonic_ns() // 1_000_000,
    }


def build_ready_status(readiness: ReadinessFields) -> dict[str, Any]:
    """command_id 없는 GET_STATUS에 대한 준비 상태 응답을 만듭니다."""
    return _common_status_fields(readiness)


def build_command_status(
    command_id: str,
    state: str,
    readiness: ReadinessFields,
    *,
    stage: str | None = None,
    code: str | None = None,
    detail: str | None = None,
    reach_error: dict[str, float] | None = None,
) -> dict[str, Any]:
    """명령 상태 STATUS를 만듭니다. 실제 상태 전이는 호출자가 결정합니다."""
    _validate_uuid_v4(command_id)
    if state not in COMMAND_STATES:
        raise ValueError(f"지원하지 않는 command state: {state}")
    if state == "FAILED":
        if code not in FAILURE_CODES or stage not in {"PLANNING", "EXECUTION", "REACH_CHECK", "RECOVERY"}:
            raise ValueError("FAILED에는 명세에 맞는 code와 stage가 필요합니다")
        expected_stages = {
            "PLAN_FAILED": {"PLANNING"},
            "CANCELLED": {"PLANNING", "EXECUTION"},
            "EXEC_FAILED": {"EXECUTION"},
            "FEEDBACK_LOST": {"EXECUTION"},
            "CANCEL_UNCONFIRMED": {"EXECUTION"},
            "REACH_TOLERANCE": {"REACH_CHECK"},
            "SERVER_RESTARTED": {"RECOVERY"},
            "LOG_WRITE_FAILED": {"PLANNING"},
        }
        if stage not in expected_stages[code]:
            raise ValueError(f"{code}는 stage={stage}와 조합할 수 없습니다")
    elif stage is not None or code is not None:
        raise ValueError("FAILED가 아닌 상태에는 stage/code를 넣을 수 없습니다")
    if (state == "REACHED" or stage == "REACH_CHECK") != (reach_error is not None):
        raise ValueError("REACHED 및 REACH_CHECK 실패에는 reach_error가 필요합니다")
    if reach_error is not None:
        if set(reach_error) != {"position_mm", "rotation_deg"}:
            raise ValueError("reach_error 필드가 올바르지 않습니다")
        if any(_require_finite_number(value, key) < 0 for key, value in reach_error.items()):
            raise ValueError("reach_error 값은 0 이상이어야 합니다")
    response = _common_status_fields(readiness)
    response.update(command_id=command_id, state=state)
    if stage is not None:
        response["stage"] = stage
    if code is not None:
        response["code"] = code
    if detail is not None:
        if not isinstance(detail, str) or len(detail) > 256:
            raise ValueError("detail은 최대 256자 문자열이어야 합니다")
        response["detail"] = detail
    if reach_error is not None:
        response["reach_error"] = reach_error
    return response


def build_rejection_status(
    code: str,
    readiness: ReadinessFields,
    command_id: str | None = None,
    *,
    detail: str | None = None,
) -> dict[str, Any]:
    """명령을 등록하지 않는 거부 STATUS를 만듭니다."""
    if code not in REJECTION_CODES:
        raise ValueError(f"거부 응답에 사용할 수 없는 code: {code}")
    if code != "BAD_FORMAT" and command_id is None:
        raise ValueError(f"{code} 거부에는 command_id가 필요합니다")
    if command_id is not None:
        _validate_uuid_v4(command_id)
    response = _common_status_fields(readiness)
    if command_id is not None:
        response["command_id"] = command_id
    response["code"] = code
    if detail is not None:
        if not isinstance(detail, str) or len(detail) > 256:
            raise ValueError("detail은 최대 256자 문자열이어야 합니다")
        response["detail"] = detail
    return response



def build_get_status_response(
    request: dict[str, Any], readiness: ReadinessFields
) -> dict[str, Any]:
    """GET_STATUS 응답을 구성합니다.

    ID 지정 조회는 이 함수 호출 전에 tcp_server.py가 command_manager에서
    기록을 찾습니다. 이 함수는 그 조회가 없는 경우 UNKNOWN_COMMAND를 만듭니다.
    """
    command_id = validate_get_status_request(request)
    if command_id is None:
        return build_ready_status(readiness)
    return build_unknown_command_status(command_id, readiness)

def build_unknown_command_status(
    command_id: str, readiness: ReadinessFields
) -> dict[str, Any]:
    """기록에 없는 ID를 조회했을 때의 거부 STATUS를 만듭니다."""
    return build_rejection_status("UNKNOWN_COMMAND", readiness, command_id)


def build_bad_format_status(
    readiness: ReadinessFields, command_id: str | None = None
) -> dict[str, Any]:
    """BAD_FORMAT 응답을 만듭니다. 읽을 수 있는 UUID가 있으면 포함합니다."""
    return build_rejection_status("BAD_FORMAT", readiness, command_id)


def encode_response(response: dict[str, Any]) -> bytes:
    """STATUS 객체를 UTF-8 JSON 한 줄과 LF로 직렬화합니다."""
    try:
        return (
            json.dumps(
                response,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ProtocolError(f"STATUS 직렬화 실패: {error}") from error
