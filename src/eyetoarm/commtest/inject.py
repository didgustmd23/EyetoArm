"""시험 모드 전용 장애 주입. 운영 경로와 분리된 시험 모드에서만 쓴다.

`Injector`는 별도 연결로 D `protocol`이 만든 바이트를 보내 같은 ID 재송신·서버 BUSY를 확인하고,
`FaultRelay`는 클라이언트 ↔ 서버 중계로 줄을 삼키거나 끊어 D의 재전송·재접속 경로를 타게 한다.
둘 다 `test_mode=True`가 아니면 ValueError이고, 하나의 asyncio 루프(실행기 소유)에서만 쓴다.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Callable

from eyetoarm.tcp.protocol import ProtocolError, parse_status


def require_test_mode(test_mode: bool, what: str) -> None:
    """시험 모드가 아니면 ValueError(운영 경로에서 주입 금지)."""
    if test_mode is not True:
        raise ValueError(f"{what} is a fault injector and requires test_mode=True (TCP spec §8.6)")


class Injector:
    """별도 연결 주입기. `open()` → `send(raw)` → 응답 dict(또는 None), `close()`."""

    def __init__(self, host: str, port: int, *, test_mode: bool, log: list, timeout_s: float = 2.0) -> None:
        """대상 주소·시험 모드·사건 목록."""
        require_test_mode(test_mode, "Injector")
        self.host, self.port, self.log, self.timeout_s = host, port, log, timeout_s
        self._reader = self._writer = None

    async def open(self) -> None:
        """연결."""
        self._reader, self._writer = await asyncio.wait_for(asyncio.open_connection(self.host, self.port), self.timeout_s)
        self.log.append({"kind": "injector_open", "fault_injected": True, "mono_ns": time.monotonic_ns()})

    async def send(self, raw: bytes, *, label: str) -> tuple[int, dict | None, int | None]:
        """한 줄 보내고 다음 한 줄(직접 응답)을 읽는다. (송신 ns, 응답 또는 None, 응답 ns 또는 None)."""
        sent = time.monotonic_ns()
        self._writer.write(raw)
        await self._writer.drain()
        reply, got, error = None, None, None

        try:
            line = await asyncio.wait_for(self._reader.readline(), self.timeout_s)
            got = time.monotonic_ns()
            reply = parse_status(line.rstrip(b"\n"))
        except (asyncio.TimeoutError, ProtocolError, ConnectionError) as exc:
            error = f"{type(exc).__name__}: {exc}"[:200]

        self.log.append({"kind": "injector_send", "label": label, "fault_injected": True, "bytes": len(raw),
                         "sent_ns": sent, "reply_ns": got, "reply_state": None if reply is None else reply.get("state"),
                         "reply_code": None if reply is None else reply.get("code"), "error": error})

        return sent, reply, got

    async def close(self) -> None:
        """닫는다."""
        if self._writer is not None:
            self._writer.close()

            try:
                await self._writer.wait_closed()
            except ConnectionError:   # 이미 끊긴 소켓: 닫힘 확인만 실패, 결과 영향 없음
                pass

            self._writer = None


def _parse_object(line: bytes) -> dict:
    """중계기가 들여다볼 JSON 객체. 해석할 수 없거나 객체가 아닌 줄은 빈 사전(줄은 그대로 넘긴다)."""
    try:
        msg = json.loads(line)
    except ValueError:
        return {}

    return msg if isinstance(msg, dict) else {}


class FaultRelay:
    """클라이언트 ↔ 서버 중계기. `start()` → 들을 포트. 규칙은 실행 중에 바꿀 수 있다(같은 루프에서만)."""

    def __init__(self, upstream_host: str, upstream_port: int, *, test_mode: bool, log: list) -> None:
        """상류 서버 주소·시험 모드·사건 목록."""
        require_test_mode(test_mode, "FaultRelay")
        self.upstream = (upstream_host, upstream_port)
        self.log = log
        self.swallow_rule: Callable[[dict], bool] | None = None   # 서버→클라이언트 줄 삼킴 규칙
        self.cut_after_swallow = False                             # 삼킨 직후 그 연결 쌍을 끊음(1회)
        self.stop_swallow_on_resend: str | None = None             # 이 ID("*"=아무)의 MOVE 재통과 시 삼킴 해제
        self._move_seen: dict[str, int] = {}
        self._server: asyncio.AbstractServer | None = None
        self._pairs: list = []
        self._tasks: set = set()

    def _event(self, kind: str, **fields) -> None:
        """사건 기록."""
        self.log.append({"kind": kind, "fault_injected": True, "mono_ns": time.monotonic_ns(), **fields})

    async def start(self) -> int:
        """127.0.0.1 포트 0에서 듣는다."""
        self._server = await asyncio.start_server(self._accept, "127.0.0.1", 0)

        return self._server.sockets[0].getsockname()[1]

    async def _accept(self, c_reader, c_writer) -> None:
        """클라이언트 연결 하나 → 상류 연결 하나."""
        try:
            s_reader, s_writer = await asyncio.open_connection(*self.upstream)
        except OSError as exc:
            self._event("relay_upstream_failed", error=str(exc)[:200])
            c_writer.close()
            return

        pair = (c_writer, s_writer)
        self._pairs.append(pair)
        self._event("relay_pair_open", pair=len(self._pairs))
        up = asyncio.ensure_future(self._pump_c2s(c_reader, s_writer))
        down = asyncio.ensure_future(self._pump_s2c(s_reader, c_writer, pair))

        for t in (up, down):
            self._tasks.add(t)
            t.add_done_callback(self._tasks.discard)

        await asyncio.wait({up, down}, return_when=asyncio.FIRST_COMPLETED)
        self._close_pair(pair)

    async def _pump_c2s(self, reader, writer) -> None:
        """클라이언트→서버 펌프. 연결 끊김은 정상 종료, 그 밖의 예외는 `relay_c2s_failed`로 기록한다."""
        try:
            await self._c2s(reader, writer)
        except (ConnectionError, OSError) as exc:
            self._event("relay_c2s_closed", error=f"{type(exc).__name__}: {exc}"[:200])
        except Exception as exc:  # noqa: BLE001 - 작업 최상단: 버려진 작업에 묻히지 않게 기록한다
            self._event("relay_c2s_failed", error=f"{type(exc).__name__}: {exc}"[:200])

    async def _pump_s2c(self, reader, writer, pair) -> None:
        """서버→클라이언트 펌프. 연결 끊김은 정상 종료, 그 밖의 예외는 `relay_s2c_failed`로 기록한다."""
        try:
            await self._s2c(reader, writer, pair)
        except (ConnectionError, OSError) as exc:
            self._event("relay_s2c_closed", error=f"{type(exc).__name__}: {exc}"[:200])
        except Exception as exc:  # noqa: BLE001 - 작업 최상단: 버려진 작업에 묻히지 않게 기록한다
            self._event("relay_s2c_failed", error=f"{type(exc).__name__}: {exc}"[:200])

    async def _c2s(self, reader, writer) -> None:
        """클라이언트→서버: 그대로 넘기고 MOVE command_id 통과 수를 센다."""
        while line := await reader.readline():
            msg = _parse_object(line)

            if msg.get("type") == "MOVE":
                cid = msg.get("command_id")
                self._move_seen[cid] = self._move_seen.get(cid, 0) + 1

                if self.stop_swallow_on_resend in (cid, "*") and self._move_seen[cid] >= 2:
                    self.swallow_rule, self.stop_swallow_on_resend = None, None
                    self._event("relay_swallow_released", command_id=cid, reason="resend passed")

            writer.write(line)
            await writer.drain()

    async def _s2c(self, reader, writer, pair) -> None:
        """서버→클라이언트: 규칙에 맞는 줄은 삼키고(기록), 필요하면 끊는다."""
        while line := await reader.readline():
            msg = _parse_object(line)

            rule = self.swallow_rule

            if rule is not None and rule(msg):
                self._event("relay_swallow", command_id=msg.get("command_id"), state=msg.get("state"),
                            code=msg.get("code"))

                if self.cut_after_swallow:
                    self.cut_after_swallow = False
                    self.swallow_rule = None
                    self._event("relay_cut", command_id=msg.get("command_id"))
                    self._close_pair(pair)
                    return

                continue

            writer.write(line)
            await writer.drain()

    def _close_pair(self, pair) -> None:
        """두 쪽 소켓을 닫는다(여러 번 불려도 됨)."""
        for w in pair:
            if not w.is_closing():
                w.close()

    def move_count(self, command_id: str) -> int:
        """그 ID의 MOVE가 중계기를 지나간 수(클라이언트 재전송 확인용)."""
        return self._move_seen.get(command_id, 0)

    async def close(self) -> None:
        """리스너·연결·펌프 정리."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

        for pair in self._pairs:
            self._close_pair(pair)

        for t in list(self._tasks):
            t.cancel()

        await asyncio.gather(*list(self._tasks), return_exceptions=True)
