"""TCP 1.1 요청 파싱과 STATUS 응답 구성.

이번 초기 구현은 GET_STATUS 왕복에 필요한 프로토콜 유틸리티만 제공합니다.
소켓 프레이밍·전송은 tcp_server.py, 준비 상태는 readiness.py가 담당합니다.
MOVE와 CANCEL의 처리는 후속 단계에서 command_manager와 연결합니다.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from typing import Any

from .readiness import ReadinessFields

VERSION = "1.1"
MAX_SAFE_INT = 9_007_199_254_740_991


class ProtocolError(ValueError):
    """수신한 JSON 또는 요청 형식이 TCP 명세를 따르지 않습니다."""


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
    parsed = int(value)
    if abs(parsed) > MAX_SAFE_INT:
        raise ProtocolError("표현 가능한 범위를 넘는 정수")
    return parsed


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
    if request.get("version") != VERSION or request.get("type") != "GET_STATUS":
        raise ProtocolError("지원하지 않는 version 또는 type")

    if "command_id" not in request:
        return None
    command_id = request["command_id"]
    if not isinstance(command_id, str):
        raise ProtocolError("command_id는 UUID 문자열이어야 합니다")
    try:
        parsed_id = uuid.UUID(command_id)
    except (ValueError, AttributeError) as error:
        raise ProtocolError("command_id가 유효한 UUID가 아닙니다") from error
    if str(parsed_id) != command_id:
        raise ProtocolError("command_id는 소문자 하이픈 UUID 형식이어야 합니다")
    return command_id


def _common_status_fields(readiness: ReadinessFields) -> dict[str, Any]:
    ready = readiness["robot_ready"]
    reasons = readiness["not_ready_reasons"]
    active_command_id = readiness["active_command_id"]

    if not isinstance(ready, bool) or not isinstance(reasons, list):
        raise ValueError("readiness 필드 형식이 올바르지 않습니다")
    if ready and reasons:
        raise ValueError("robot_ready=true이면 not_ready_reasons는 비어 있어야 합니다")
    if not ready and not reasons:
        raise ValueError("robot_ready=false이면 not_ready_reasons가 필요합니다")
    if active_command_id is not None and ready:
        raise ValueError("active command가 있으면 robot_ready는 false여야 합니다")
    if active_command_id is not None and "COMMAND_ACTIVE" not in reasons:
        raise ValueError("active command가 있으면 COMMAND_ACTIVE 사유가 필요합니다")

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



def build_get_status_response(
    request: dict[str, Any], readiness: ReadinessFields
) -> dict[str, Any]:
    """GET_STATUS 응답을 구성합니다.

    현재 단계에서는 command_id 조회용 기록이 아직 없으므로, ID가 지정된
    조회는 UNKNOWN_COMMAND를 반환합니다. command_manager 연결 뒤 이 경로를
    해당 명령의 저장 상태 조회로 확장합니다.
    """
    command_id = validate_get_status_request(request)
    if command_id is None:
        return build_ready_status(readiness)
    return build_unknown_command_status(command_id, readiness)

def build_unknown_command_status(
    command_id: str, readiness: ReadinessFields
) -> dict[str, Any]:
    """기록에 없는 ID를 조회했을 때의 거부 STATUS를 만듭니다."""
    response = _common_status_fields(readiness)
    response["command_id"] = command_id
    response["code"] = "UNKNOWN_COMMAND"
    return response


def build_bad_format_status(
    readiness: ReadinessFields, command_id: str | None = None
) -> dict[str, Any]:
    """BAD_FORMAT 응답을 만듭니다. 읽을 수 있는 UUID가 있으면 포함합니다."""
    response = _common_status_fields(readiness)
    if command_id is not None:
        response["command_id"] = command_id
    response["code"] = "BAD_FORMAT"
    return response


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