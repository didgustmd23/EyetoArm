"""D 1일차 진단: 실행 환경 점검과 C 서버 무ID GET_STATUS 1회 조회.

동작 명령(MOVE·CANCEL·START·STOP 등)은 보내지 않는다. 보내는 것은 TCP 1.1 §4.4의
command_id 없는 GET_STATUS 한 줄뿐이다. 신호 작성·STATUS 검사는 팀 공통
eyetoarm.tcp.protocol(build_get_status·parse_status·status_kind)을 그대로 쓴다.

결과는 세 단계로 나눠 판정한다(결과 데이터 ProbeResult 는 models.py, 여기서 재수출).
  connection: 접속 성공 여부(refused·connect_timeout·os_error)
  protocol:   STATUS 한 줄 수신·검사 결과(ok·timeout·eof·line_too_long·invalid·reject·unexpected)
  readiness:  robot_ready 값과 not_ready_reasons (protocol=ok 일 때만)

host 는 숫자 IP 리터럴(IPv4·IPv6)만 받는다. 이름 해석(DNS)은 총 기한 밖에서 막힐 수 있어 1일차에는
쓰지 않는다. localhost 대신 127.0.0.1 을 쓴다.
총 기한 하나(접속+송신+수신)는 진단 전용이다. 운영 클라이언트(eyetoarm.tcp.client)의
connect_timeout_ms·readiness_timeout_ms 분리 타이머와 다르다.

probe 는 동기·단발 진단이며 소켓에서 최대 timeout_s 동안 막힌다. GUI 스레드에서 직접 부르지 않는다.
실제 GUI 의 지속 통신은 별도 비동기 연결을 쓴다.
"""

from __future__ import annotations

import ipaddress
import socket
import sys
import time
from dataclasses import asdict
from typing import Any

if __name__ == "__main__":
    if not __package__:
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from eyetoarm.gui.__main__ import main
    raise SystemExit(main())

from eyetoarm.gui.models import EXIT_CONNECTION, EXIT_OK, EXIT_PROTOCOL, ProbeResult  # noqa: F401 - 재수출
from eyetoarm.tcp import protocol

MAX_LINE_BYTES = 4096   # TCP 명세서 §9 max_line_bytes (LF 제외)
MAX_TIMEOUT_S = 60.0    # 진단 한 번의 총 기한 상한. 잘못된 입력으로 오래 붙잡히지 않게 한다
EXPECTED = {"python": (3, 12), "PySide6": "6.12.0", "jsonschema": "4.26.0"}  # environment.yml 고정값

EXIT_ENV = 4
EXIT_USAGE = 5
# 기본 총 기한 3 s = config/gui/client.json connect_timeout_ms 2000 + readiness_timeout_ms 1000
DEFAULT_TIMEOUT_S = 3.0


def safe_text(value: Any, limit: int = 200) -> str:
    """서버가 보낸 문자열을 터미널·기록에 그대로 쓰지 않는다: 제어 문자 이스케이프, 길이 제한."""
    text = str(value)
    text = "".join(ch if ch.isprintable() else repr(ch)[1:-1] for ch in text)
    return text if len(text) <= limit else text[:limit] + "…"


# ── 환경 점검 ─────────────────────────────────────────────────────────

def check_environment() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    py = sys.version_info
    checks["python"] = {"found": f"{py.major}.{py.minor}.{py.micro}", "expected": "3.12.x",
                        "ok": (py.major, py.minor) == EXPECTED["python"]}
    for name in ("PySide6", "jsonschema"):
        try:
            if name == "PySide6":
                import PySide6 as module  # noqa: N813
                found = module.__version__
            else:
                from importlib.metadata import version
                import jsonschema  # noqa: F401
                found = version("jsonschema")  # jsonschema.__version__ 는 4.x 에서 폐기 예정
            checks[name] = {"found": found, "expected": EXPECTED[name], "ok": found == EXPECTED[name]}
        except Exception as error:  # noqa: BLE001 - 없거나 깨진 설치 모두 실패로 보고
            checks[name] = {"found": None, "expected": EXPECTED[name], "ok": False, "error": safe_text(error)}
    schema = protocol.SCHEMA_DIR / "s2c.schema.json"
    checks["s2c_schema"] = {"path": str(schema), "ok": schema.is_file() and protocol._VALIDATOR is not None,
                            "validator_loaded": protocol._VALIDATOR is not None}
    return {"ok": all(item["ok"] for item in checks.values()), "checks": checks}


# ── 접속 진단 ─────────────────────────────────────────────────────────

def read_line(sock: socket.socket, deadline: float, max_line_bytes: int = MAX_LINE_BYTES) -> tuple[bytes, int]:
    """LF 하나가 올 때까지 읽는다. (LF 뗀 줄, 줄 뒤 남은 바이트 수). 바이트로 모아 UTF-8 분할에 안전하다.

    실패: TimeoutError(총 기한), EOFError(LF 전 종료), OverflowError(LF 전 max_line_bytes 초과).
    """
    buffer = bytearray()
    while True:
        index = buffer.find(b"\n")
        if index >= 0:
            line = bytes(buffer[:index])
            if len(line.removesuffix(b"\r")) > max_line_bytes:
                raise OverflowError(f"한 줄이 {max_line_bytes}바이트를 넘음")
            return line, len(buffer) - index - 1
        if len(buffer) > max_line_bytes + 1:  # +1: LF 앞 CR 허용
            raise OverflowError(f"LF 없이 {len(buffer)}바이트 수신, 한도 {max_line_bytes}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("총 수신 기한 초과")
        sock.settimeout(remaining)
        try:
            chunk = sock.recv(4096)
        except socket.timeout as error:
            raise TimeoutError("총 수신 기한 초과") from error
        if not chunk:
            raise EOFError(f"LF 전에 연결 종료 (미완성 {len(buffer)}바이트 폐기)")
        buffer += chunk


def validate_target(host: str, port: int, timeout_s: float) -> None:
    """호출 측 입력 검사. CLI 밖에서 probe 를 직접 불러도 같은 범위를 지킨다."""
    try:
        ipaddress.ip_address(host)  # 숫자 IP 만. 이름 해석(DNS)은 기한 밖에서 막힐 수 있어 거부
    except (TypeError, ValueError):
        raise ValueError("host 는 숫자 IP 리터럴(예: 127.0.0.1, ::1)만 허용") from None
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("port 는 1~65535 정수")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not 0 < timeout_s <= MAX_TIMEOUT_S:
        raise ValueError(f"timeout 은 0 초과 {MAX_TIMEOUT_S:g} 이하 (NaN 불가)")


def probe(host: str, port: int, timeout_s: float) -> ProbeResult:
    """무ID GET_STATUS 1회. 접속·수신 모두 timeout_s 하나의 총 기한 안에서 끝난다."""
    validate_target(host, port, timeout_s)
    result = ProbeResult(host=host, port=port, timeout_s=timeout_s)
    start = time.monotonic()
    deadline = start + timeout_s
    try:
        try:
            sock = socket.create_connection((host, port), timeout=timeout_s)
        except ConnectionRefusedError as error:
            result.connection, result.detail = "refused", safe_text(error)
            return result
        except (socket.timeout, TimeoutError) as error:
            result.connection, result.detail = "connect_timeout", safe_text(error)
            return result
        except OSError as error:
            result.connection, result.detail = "os_error", safe_text(error)
            return result
        result.connection = "ok"
        with sock:
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:  # 접속에 기한을 다 썼으면 보내지 않는다
                    raise TimeoutError("접속 후 기한 초과, 요청 미전송")
                sock.settimeout(remaining)
                sock.sendall(protocol.build_get_status())
                line, result.trailing_bytes = read_line(sock, deadline)
            except TimeoutError as error:
                result.protocol, result.detail = "timeout", safe_text(error)
                return result
            except socket.timeout as error:
                result.protocol, result.detail = "timeout", safe_text(error)
                return result
            except EOFError as error:
                result.protocol, result.detail = "eof", safe_text(error)
                return result
            except OverflowError as error:
                result.protocol, result.detail = "line_too_long", safe_text(error)
                return result
            except OSError as error:
                result.protocol, result.detail = "io_error", safe_text(error)
                return result
        try:
            message = protocol.parse_status(line)
        except protocol.ProtocolError as error:
            result.protocol, result.detail = "invalid", safe_text(error)
            return result
        kind = protocol.status_kind(message)
        if kind == "reject":
            result.protocol = "reject"
            result.detail = safe_text(f"code={message.get('code')} detail={message.get('detail', '')}")
        elif kind != "ready" or "command_id" in message:
            result.protocol, result.detail = "unexpected", "무ID 조회에 명령 상태 응답이 옴"
        else:
            result.protocol = "ok"
        # 거부 응답도 STATUS 필수 필드를 갖고 있으므로 준비 정보는 기록한다(판정은 protocol=ok 만)
        result.robot_ready = message["robot_ready"]
        result.not_ready_reasons = list(message["not_ready_reasons"])
        result.active_command_id = message["active_command_id"]
        result.server_time_ms = message["server_time_ms"]
        return result
    finally:
        result.elapsed_ms = int((time.monotonic() - start) * 1000)


def to_record(env: dict[str, Any] | None, result: ProbeResult | None) -> dict[str, Any]:
    record: dict[str, Any] = {"tool": "eyetoarm.d_client", "check": "day1", "wall_time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if env is not None:
        record["environment"] = env
    if result is not None:
        record["probe"] = asdict(result)
    record["exit_code"] = overall_exit(env, result)
    return record


def overall_exit(env: dict[str, Any] | None, result: ProbeResult | None) -> int:
    """환경 불일치가 우선(접속 결과는 그대로 출력). 그다음 접속·프로토콜 결과."""
    if env is not None and not env["ok"]:
        return EXIT_ENV
    return result.exit_code if result is not None else EXIT_OK


def format_human(env: dict[str, Any] | None, result: ProbeResult | None) -> str:
    lines = []
    if env is not None:
        lines.append(f"[환경] {'정상' if env['ok'] else '불일치'}")
        for name, item in env["checks"].items():
            mark = "OK " if item["ok"] else "NG "
            found = item.get("found", item.get("path"))
            lines.append(f"  {mark}{name}: {found} (기대 {item.get('expected', '존재·검증기 로드')})"
                         + (f" {item['error']}" if item.get("error") else ""))
    if result is not None:
        lines.append(f"[접속] {result.host}:{result.port} -> {result.connection}"
                     + (f" ({result.detail})" if result.connection != "ok" else ""))
        if result.connection == "ok":
            lines.append(f"[프로토콜] STATUS -> {result.protocol}"
                         + (f" ({result.detail})" if result.protocol != "ok" else ""))
        if result.protocol == "ok":
            reasons = ",".join(result.not_ready_reasons) or "-"
            lines.append(f"[준비] robot_ready={str(result.robot_ready).lower()} not_ready_reasons={reasons}"
                         f" active_command_id={result.active_command_id}")
        if result.protocol == "ok":
            lines.append("[판정] 유효 STATUS 수신 = 접속 진단 성공"
                         + ("" if result.robot_ready else " (robot_ready=false 는 C 준비 상태이며 접속 실패가 아님)"))
        lines.append(f"[시간] {result.elapsed_ms} ms")
    lines.append(f"[종료 코드] {overall_exit(env, result)}")
    return "\n".join(lines)
