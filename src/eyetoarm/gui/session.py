"""D 클라이언트 세션. 화면(Qt)과 무관하게 조작 → 신호 작성·전송·기록을 묶는다.

로봇 쪽으로 나가는 것은 TcpClient 의 TCP 1.1 연결 하나뿐이다. C 서버·로봇·Gazebo 를 흉내 내지 않는다.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from eyetoarm.pendant.model import PendantModel
from eyetoarm.tcp.client import ClientConfig, TcpClient, now_ms
from eyetoarm.tcp.event_log import EventLog
from eyetoarm.tcp.protocol import MoveTarget, encode

PENDANT_NOTE = "전송 경로 미합의 · 작성·기록만 함 (D 제안 형식, TCP 1.1 아님)"
MANUAL_SOURCE = "pendant_manual"   # 수동 목표의 source_id. 비전 근거가 아님을 드러낸다
MANUAL_CALIBRATION = "manual_test"


class ClientSession:
    def __init__(self, config: ClientConfig, log_path: Path | None, notify: Callable[[], None] = lambda: None) -> None:
        self.log = EventLog(log_path)
        self.notify = notify
        self.client = TcpClient(config, self.log, self._changed)
        self.pendant = PendantModel()
        self.started_wall_ms = time.time_ns() // 1_000_000

    def _changed(self) -> None:
        self.notify()

    def state(self) -> dict[str, Any]:
        return {"tcp": self.client.snapshot(), "pendant": self.pendant.snapshot(), "pendant_channel": PENDANT_NOTE,
                "log_path": str(self.log.path) if self.log.path else None, "log_error": self.log.write_error,
                "routine_count": self.log.routine_count,
                "config": {key: getattr(self.client.config, key) for key in (
                    "ack_timeout_ms", "max_resend", "readiness_poll_ms", "readiness_timeout_ms", "status_watchdog_ms",
                    "target_validity_ms", "max_line_bytes")}}

    def manual_target(self) -> MoveTarget:
        cart = self.pendant.cart
        return MoveTarget(position_mm=(cart["X"], cart["Y"]), theta_deg=cart["RZ"] % 180.0, source_id=MANUAL_SOURCE,
                          image_seq=self.pendant.target_seq, timestamp_ms=now_ms(), timestamp_base="PC_RECEIVE",
                          calibration_id=MANUAL_CALIBRATION, created_monotonic_ms=now_ms())

    # ── 조작 → 신호. 모두 통신 루프 안에서 부른다 ─────────────────────────

    async def connect(self, host: str | None = None, port: int | None = None) -> str:
        self.client.connect(host, port)
        return ""

    async def disconnect(self) -> str:
        await self.client.disconnect()
        return ""

    async def execute(self) -> str:
        """대시보드의 '현재 목표로 이동'. 수동 목표를 TCP 1.1 MOVE 로 보낸다."""
        return await self.client.execute(self.manual_target())

    async def cancel(self) -> str:
        return await self.client.cancel()

    async def set_target(self, x: float, y: float, theta: float) -> str:
        self.pendant.cart.update(X=float(x), Y=float(y), RZ=float(theta) % 180.0)
        self.pendant.target_seq += 1
        self.log.add("LOCAL", "수동 목표 변경", note=json.dumps(self.pendant.cart, ensure_ascii=False))
        self.notify()
        return "수동 목표 적용"

    async def pendant_action(self, action: str, **params: Any) -> str:
        message = ""
        if action == "start":  # 펜던트 START = 수동 목표를 TCP 1.1 MOVE 로 보냄
            if self.pendant.estop or not self.pendant.enabled:
                self.pendant.message = "START 불가 · 인에이블 꺼짐 또는 정지 잠금"
            else:
                self.pendant.message = message = await self.client.execute(self.manual_target())
        else:
            signal = self.pendant.act(action, **params)
            if signal is not None:  # 조그·그리퍼·속도: 제안 형식으로 작성하고 기록한다. 보낼 경로는 아직 없다
                self.log.add("TX", "PENDANT " + signal["type"], raw=encode(signal), channel="pendant-draft",
                             send_state="unsent", note=PENDANT_NOTE)
            if action in ("stop", "estop"):  # 정지는 TCP 1.1 CANCEL 로도 보낸다
                message = await self.client.cancel()
        self.notify()
        return message  # TCP 1.1 로 나간 조작(START·STOP)의 결과만 돌려준다. 조그 안내는 펜던트 화면이 직접 보여 준다
