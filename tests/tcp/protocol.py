import json
import time


def handle_request(request: dict) -> dict:
    """요청 하나를 받아 응답 하나를 반환합니다."""

    if request.get("version") != "1.1":
        return make_error("BAD_FORMAT")

    if request.get("type") != "GET_STATUS":
        return make_error("BAD_FORMAT")

    # 통신만 시험하는 단계라 로봇 상태를 아직 확인하지 않습니다.
    return {
        "version": "1.1",
        "type": "STATUS",
        "robot_ready": False,
        "not_ready_reasons": ["RECOVERING"],
        "active_command_id": None,
        "server_time_ms": int(time.time() * 1000),
    }


def make_error(code: str) -> dict:
    """간단한 오류 STATUS를 만듭니다."""
    return {
        "version": "1.1",
        "type": "STATUS",
        "code": code,
        "robot_ready": False,
        "not_ready_reasons": ["RECOVERING"],
        "active_command_id": None,
        "server_time_ms": int(time.time() * 1000),
    }


def encode_response(response: dict) -> bytes:
    """STATUS를 JSON 한 줄(LF 포함)로 바꿉니다."""
    return (
        json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")