"""TCP 1.1 서버의 초기 GET_STATUS 왕복 구현.

이 모듈은 소켓 연결과 LF 프레이밍만 담당합니다. JSON 파싱과 STATUS 구성은
protocol.py, 초기 준비 상태는 readiness.py에 위임합니다.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from collections.abc import Callable
from typing import Any

from .protocol import (
    ProtocolError,
    build_bad_format_status,
    build_get_status_response,
    encode_response,
    parse_request,
)
from .readiness import ReadinessFields, get_readiness_fields

HOST = "127.0.0.1"
PORT = 47090
MAX_LINE_BYTES = 4096
READ_CHUNK_BYTES = 4096

_LOG = logging.getLogger("robot_server_package.tcp_server")


class TcpServer:
    """연결별 LF 버퍼를 유지하며 초기 TCP 1.1 요청을 처리합니다."""

    def __init__(
        self,
        host: str = HOST,
        port: int = PORT,
        max_line_bytes: int = MAX_LINE_BYTES,
        readiness_provider: Callable[[], ReadinessFields] = get_readiness_fields,
    ) -> None:
        if max_line_bytes < 1:
            raise ValueError("max_line_bytes는 1 이상이어야 합니다")
        self.host = host
        self.port = port
        self.max_line_bytes = max_line_bytes
        self.readiness_provider = readiness_provider
        self._server: asyncio.AbstractServer | None = None
        self._connection_ids = itertools.count(1)

    async def start(self) -> None:
        """TCP 수신을 시작하고 종료될 때까지 대기합니다."""
        self._server = await asyncio.start_server(
            self._handle_client,
            self.host,
            self.port,
            limit=self.max_line_bytes + 1,
        )
        addresses = ", ".join(str(sock.getsockname()) for sock in (self._server.sockets or []))
        _LOG.info("TCP 1.1 서버 대기 중: %s", addresses)
        async with self._server:
            await self._server.serve_forever()

    async def close(self) -> None:
        """서버 listener를 닫습니다."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        connection_id = next(self._connection_ids)
        peer = writer.get_extra_info("peername")
        _LOG.info("클라이언트 연결: id=%s peer=%s", connection_id, peer)

        buffer = bytearray()
        discarding_oversized_line = False
        try:
            while chunk := await reader.read(READ_CHUNK_BYTES):
                cursor = 0
                while cursor < len(chunk):
                    if discarding_oversized_line:
                        newline = chunk.find(b"\n", cursor)
                        if newline < 0:
                            break
                        cursor = newline + 1
                        discarding_oversized_line = False
                        continue

                    newline = chunk.find(b"\n", cursor)
                    if newline < 0:
                        buffer.extend(chunk[cursor:])
                        if len(buffer) > self.max_line_bytes:
                            _LOG.warning(
                                "요청 줄 길이 초과: connection_id=%s limit=%s",
                                connection_id,
                                self.max_line_bytes,
                            )
                            await self._send_bad_format(writer, connection_id)
                            buffer.clear()
                            discarding_oversized_line = True
                        break

                    line_part = chunk[cursor:newline]
                    if len(buffer) + len(line_part) > self.max_line_bytes:
                        _LOG.warning(
                            "요청 줄 길이 초과: connection_id=%s limit=%s",
                            connection_id,
                            self.max_line_bytes,
                        )
                        await self._send_bad_format(writer, connection_id)
                        buffer.clear()
                    else:
                        line = bytes(buffer) + line_part
                        buffer.clear()
                        keep_connection = await self._handle_line(line, writer, connection_id)
                        if not keep_connection:
                            return
                    cursor = newline + 1
        except (ConnectionError, asyncio.CancelledError):
            raise
        except Exception:
            _LOG.exception("클라이언트 처리 오류: connection_id=%s", connection_id)
        finally:
            # LF 없이 연결된 마지막 조각은 명세에 따라 버립니다.
            if buffer:
                _LOG.info(
                    "미완성 요청 조각 폐기: connection_id=%s bytes=%s",
                    connection_id,
                    len(buffer),
                )
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass
            _LOG.info("클라이언트 연결 종료: id=%s peer=%s", connection_id, peer)

    async def _handle_line(
        self, line: bytes, writer: asyncio.StreamWriter, connection_id: int
    ) -> bool:
        _LOG.info("RX connection_id=%s bytes=%s raw=%r", connection_id, len(line), line)
        try:
            request = parse_request(line)
            if request.get("type") in {"MOVE", "CANCEL"}:
                _LOG.warning("요청 유형은 아직 구현되지 않음: connection_id=%s type=%s", connection_id, request.get("type"))
                return False
            response = build_get_status_response(request, self.readiness_provider())
        except ProtocolError as error:
            _LOG.warning("요청 거부: connection_id=%s reason=%s", connection_id, error)
            response = build_bad_format_status(self.readiness_provider())

        await self._send_response(response, writer, connection_id)
        return True

    async def _send_bad_format(
        self, writer: asyncio.StreamWriter, connection_id: int
    ) -> None:
        response = build_bad_format_status(self.readiness_provider())
        await self._send_response(response, writer, connection_id)

    async def _send_response(
        self,
        response: dict[str, Any],
        writer: asyncio.StreamWriter,
        connection_id: int,
    ) -> None:
        payload = encode_response(response)
        writer.write(payload)
        await writer.drain()
        _LOG.info("TX connection_id=%s bytes=%s raw=%r", connection_id, len(payload), payload)


async def serve(host: str = HOST, port: int = PORT) -> None:
    server = TcpServer(host=host, port=port)
    try:
        await server.start()
    finally:
        await server.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        _LOG.info("TCP 서버 종료 요청")


if __name__ == "__main__":
    main()