"""D 클라이언트 결과 데이터. CLI·socket·PySide6 없이 import 할 수 있다(GUI 재사용용).

공유 스냅샷은 dataclasses.asdict(result) 를 그대로 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 실제 C 서버는 robot_ready=false(RECOVERING 등)가 정상일 수 있으므로 유효 STATUS 수신이면 성공(0)이다.
EXIT_OK = 0
EXIT_CONNECTION = 1
EXIT_PROTOCOL = 2


@dataclass
class ProbeResult:
    host: str
    port: int
    timeout_s: float
    connection: str = "not_attempted"
    protocol: str = "not_attempted"
    robot_ready: bool | None = None
    not_ready_reasons: list[str] = field(default_factory=list)
    active_command_id: str | None = None
    server_time_ms: int | None = None
    detail: str = ""
    elapsed_ms: int = 0
    trailing_bytes: int = 0   # 첫 줄 뒤에 함께 온 바이트 수(기록만)

    @property
    def exit_code(self) -> int:
        if self.connection != "ok":
            return EXIT_CONNECTION
        if self.protocol != "ok":
            return EXIT_PROTOCOL
        return EXIT_OK
