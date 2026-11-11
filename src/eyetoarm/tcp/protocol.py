"""TCP 1.1 신호 작성과 수신 검사 (D 클라이언트). TCP 패킷 정의서 §2·§4·§8.

송신: MOVE·GET_STATUS·CANCEL 을 JSON 한 줄 + LF 바이트로 만든다.
수신: STATUS 한 줄을 §2.3 해석 규칙과 스키마로 검사한다. 통과하지 못한 줄은 기록만 하고 상태에 반영하지 않는다.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VERSION = "1.1"
SCHEMA_DIR = Path(__file__).resolve().parents[3] / "docs" / "protocol"
MAX_SAFE_INT = 9007199254740991
STATE_RANK = {"ACK": 1, "PLANNING": 2, "MOVING": 3, "REACHED": 4, "FAILED": 4}
FINAL_STATES = ("REACHED", "FAILED")


class ProtocolError(ValueError):
    """수신 줄이 정의서를 지키지 않음. 메시지는 화면·기록에 그대로 쓴다."""


@dataclass(frozen=True)
class MoveTarget:
    """사용자 실행 시점에 고정한 목표(스냅샷). 이후 새 입력으로 바뀌지 않는다."""

    position_mm: tuple[float, ...]
    theta_deg: float | None  # PLANAR
    source_id: str
    image_seq: int
    timestamp_ms: int
    timestamp_base: str
    calibration_id: str
    mode: str = "PLANAR"
    orientation_xyzw: tuple[float, float, float, float] | None = None  # POSE_6D
    created_monotonic_ms: int = 0  # 유효 시간 판정용. 전송하지 않는다


def new_command_id() -> str:
    return str(uuid.uuid4())


def encode(message: dict[str, Any]) -> bytes:
    """JSON 객체 1개 + LF 1개. 들여쓰기 없음, NaN·Infinity 금지(§2.2·§2.3)."""
    return (json.dumps(message, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def build_get_status(command_id: str | None = None) -> bytes:
    """command_id 없음 = 준비 조회, 있음 = 명령 상태 조회(§4.4)."""
    message: dict[str, Any] = {"version": VERSION, "type": "GET_STATUS"}
    if command_id is not None:
        message["command_id"] = command_id
    return encode(message)


def build_cancel(command_id: str) -> bytes:
    return encode({"version": VERSION, "type": "CANCEL", "command_id": command_id})


def build_move(command_id: str, target: MoveTarget) -> bytes:
    """MOVE 한 줄. 보낸 바이트를 보관해 재전송 때 그대로 다시 보낸다(§8.2)."""
    message: dict[str, Any] = {
        "version": VERSION, "type": "MOVE", "command_id": command_id, "mode": target.mode, "frame": "work",
        "source_id": target.source_id, "image_seq": int(target.image_seq), "timestamp_ms": int(target.timestamp_ms),
        "timestamp_base": target.timestamp_base, "calibration_id": target.calibration_id, "valid": True,
        "position_mm": [float(v) for v in target.position_mm],
    }
    if target.mode == "PLANAR":
        if target.theta_deg is None or not 0 <= target.theta_deg < 180 or len(target.position_mm) != 2:
            raise ValueError("PLANAR 목표는 position_mm 2개와 0 ≤ theta_deg < 180 이 필요하다")
        message["theta_deg"] = float(target.theta_deg)
    else:
        if target.orientation_xyzw is None or len(target.position_mm) != 3:
            raise ValueError("POSE_6D 목표는 position_mm 3개와 orientation_xyzw 가 필요하다")
        message["orientation_xyzw"] = [float(v) for v in target.orientation_xyzw]
    return encode(message)


# ── 수신 ─────────────────────────────────────────────────────────────

def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ProtocolError("중복 키")
    return dict(pairs)


def _reject_constant(name: str) -> Any:
    raise ProtocolError(f"허용하지 않는 숫자 {name}")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ProtocolError(f"표현 범위를 넘는 숫자 {text}")
    return value


def _safe_int(text: str) -> int:
    value = int(text)
    if abs(value) > MAX_SAFE_INT:
        raise ProtocolError(f"표현 범위를 넘는 정수 {text}")
    return value


def _load_schema() -> Any:
    try:
        import jsonschema  # 있으면 저장소 스키마 원본으로 검사한다

        schema = json.loads((SCHEMA_DIR / "s2c.schema.json").read_text(encoding="utf-8"))
        return jsonschema.Draft202012Validator(schema)
    except Exception:  # noqa: BLE001 - 라이브러리·파일이 없으면 아래 핵심 검사로 대신한다
        return None


_VALIDATOR = _load_schema()
_ALLOWED = {"version", "type", "command_id", "state", "stage", "code", "detail", "robot_ready", "not_ready_reasons",
            "active_command_id", "server_time_ms", "reach_error"}
_REQUIRED = ("version", "type", "robot_ready", "not_ready_reasons", "active_command_id", "server_time_ms")


def _core_check(message: dict[str, Any]) -> None:
    """jsonschema 가 없을 때 쓰는 최소 검사. 스키마 전체를 대신하지는 못한다."""
    unknown = set(message) - _ALLOWED
    if unknown:
        raise ProtocolError(f"정의되지 않은 필드 {sorted(unknown)}")
    missing = [key for key in _REQUIRED if key not in message]
    if missing:
        raise ProtocolError(f"필수 필드 없음 {missing}")
    if message["version"] != VERSION or message["type"] != "STATUS":
        raise ProtocolError("version·type 불일치")
    if not isinstance(message["robot_ready"], bool) or not isinstance(message["not_ready_reasons"], list):
        raise ProtocolError("robot_ready·not_ready_reasons 형식 오류")
    if message["robot_ready"] == bool(message["not_ready_reasons"]):
        raise ProtocolError("robot_ready 와 not_ready_reasons 가 서로 맞지 않음")
    if isinstance(message["server_time_ms"], bool) or not isinstance(message["server_time_ms"], int):
        raise ProtocolError("server_time_ms 는 정수여야 함")
    state = message.get("state")
    if state is not None and (state not in STATE_RANK or "command_id" not in message):
        raise ProtocolError("명령 상태 형식 오류")
    if state == "FAILED" and not ("stage" in message and "code" in message):
        raise ProtocolError("FAILED 에는 stage 와 code 가 필요함")


def parse_status(line: bytes) -> dict[str, Any]:
    """LF 를 뗀 한 줄을 STATUS 로 해석한다. 위반이면 ProtocolError."""
    if line.endswith(b"\r"):
        line = line[:-1]
    try:
        text = line.decode("utf-8")
        message = json.loads(text, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant,
                             parse_float=_finite_float, parse_int=_safe_int)
    except ProtocolError:
        raise
    except (UnicodeDecodeError, ValueError) as error:
        raise ProtocolError(f"JSON 해석 실패: {error}") from error
    if not isinstance(message, dict):
        raise ProtocolError("최상위가 객체가 아님")
    if _VALIDATOR is not None:
        errors = sorted(_VALIDATOR.iter_errors(message), key=lambda e: list(e.path))
        if errors:
            raise ProtocolError("스키마 위반: " + errors[0].message[:160])
    else:
        _core_check(message)
    # 정수 필드는 소수점·지수 표기 없는 JSON 정수여야 한다(§2.3). jsonschema 는 1.0 을 정수로 받아들이므로 따로 본다
    if type(message.get("server_time_ms")) is not int:
        raise ProtocolError("server_time_ms 는 소수점 없는 정수여야 함")
    return message


def status_kind(message: dict[str, Any]) -> str:
    """§4.7: state 있음 = 명령 상태, state 없고 code 있음 = 거부, 둘 다 없음 = 준비 응답."""
    if "state" in message:
        return "command"
    return "reject" if "code" in message else "ready"
