"""TcpClient 를 시험용 상대(이 파일 안의 대본대로 답하는 소켓)와 묶어 §8 규칙을 확인한다.

상대는 C 서버 구현이 아니라 시험이 정한 줄을 그대로 돌려주는 대역이다. 실제 C 서버·로봇 검증이 아니다.
"""

import asyncio
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eyetoarm.tcp.client import ClientConfig, TcpClient, now_ms  # noqa: E402
from eyetoarm.tcp.event_log import EventLog  # noqa: E402
from eyetoarm.tcp.protocol import MoveTarget  # noqa: E402

FAST = dict(ack_timeout_ms=80, readiness_poll_ms=60, readiness_timeout_ms=150, status_watchdog_ms=120,
            reconnect_ms=60, connect_timeout_ms=300, target_validity_ms=5000)


def status(**fields):
    base = {"version": "1.1", "type": "STATUS", "robot_ready": True, "not_ready_reasons": [], "active_command_id": None,
            "server_time_ms": 1}
    return {**base, **fields}


def busy(command_id, **fields):
    return status(robot_ready=False, not_ready_reasons=["COMMAND_ACTIVE"], active_command_id=command_id,
                  command_id=command_id, **fields)


class Peer:
    """받은 요청을 기록하고, reply(요청) 이 돌려준 줄들을 그대로 보낸다."""

    def __init__(self, reply):
        self.reply = reply
        self.requests: list[dict] = []
        self.writers: list[asyncio.StreamWriter] = []
        self.server = None

    async def start(self) -> int:
        self.server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
        return self.server.sockets[0].getsockname()[1]

    async def _serve(self, reader, writer):
        self.writers.append(writer)
        while line := await reader.readline():
            request = json.loads(line)
            self.requests.append(request)
            for answer in self.reply(request) or ():
                writer.write(answer if isinstance(answer, bytes) else json.dumps(answer, separators=(",", ":")).encode() + b"\n")
            await writer.drain()

    def push(self, message):
        self.writers[-1].write(json.dumps(message, separators=(",", ":")).encode() + b"\n")

    def moves(self):
        return [r for r in self.requests if r["type"] == "MOVE"]

    async def drop_clients(self):
        for writer in self.writers:
            writer.close()

    async def stop(self):
        self.server.close()
        await self.drop_clients()


def target():
    return MoveTarget((470.2, 160.4), 35.2, "pendant_manual", 1, now_ms(), "PC_RECEIVE", "manual_test",
                      created_monotonic_ms=now_ms())


async def until(predicate, timeout=2.0):
    end = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < end:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


class ClientCase(unittest.IsolatedAsyncioTestCase):
    async def start(self, reply, **config):
        self.peer = Peer(reply)
        port = await self.peer.start()
        self.log = EventLog(None)
        self.client = TcpClient(ClientConfig(port=port, **{**FAST, **config}), self.log)
        self.client.connect()
        self.assertTrue(await until(lambda: self.client.connected))

    async def asyncTearDown(self):
        await self.client.disconnect()
        await self.peer.stop()

    def rows(self, direction=None, kind=None):
        return [r for r in self.log.rows if (direction is None or r["dir"] == direction) and (kind is None or r["kind"] == kind)]


class Readiness(ClientCase):
    async def test_connection_alone_is_not_ready_and_poll_updates_it(self):
        await self.start(lambda r: [status(robot_ready=False, not_ready_reasons=["RECOVERING"])] if r["type"] == "GET_STATUS" else None)
        self.assertTrue(await until(lambda: self.client.robot_ready is False))
        self.assertEqual(self.client.not_ready_reasons, ["RECOVERING"])
        self.assertEqual(self.client.phase, "연결됨")

    async def test_move_is_composed_but_not_sent_while_not_ready(self):
        await self.start(lambda r: [status(robot_ready=False, not_ready_reasons=["RECOVERING"])])
        text = await self.client.execute(target())
        self.assertIn("로봇 준비 안 됨: RECOVERING", text)
        self.assertEqual(self.peer.moves(), [])
        unsent = [r for r in self.rows("TX", "MOVE") if r["send_state"] == "unsent"]
        self.assertEqual(len(unsent), 1)
        self.assertEqual(json.loads(unsent[0]["raw"])["position_mm"], [470.2, 160.4])  # 어떻게 작성됐는지는 기록에 남는다

    async def test_readiness_timeout_drops_the_connection_and_reconnects(self):
        await self.start(lambda r: None)  # 답하지 않는 상대
        first = self.client.conn_id
        self.assertTrue(await until(lambda: self.client.conn_id > first, 3))
        self.assertTrue(self.rows("LOCAL", "준비 응답 시간 초과"))

    async def test_invalid_reply_is_logged_only(self):
        await self.start(lambda r: [b'{"version":"1.1","type":"STATUS","robot_ready":true}\n'])
        self.assertTrue(await until(lambda: self.client.kpi["invalid_rx"] >= 1))
        self.assertIsNone(self.client.robot_ready)  # 잘못된 응답을 준비됨으로 읽지 않는다


class Move(ClientCase):
    def script(self, on_move):
        def reply(request):
            if request["type"] == "MOVE":
                return on_move(request)
            if request["type"] == "GET_STATUS" and "command_id" not in request:
                return [status()]
            return None
        return reply

    async def test_normal_flow_and_first_ack(self):
        await self.start(self.script(lambda r: [busy(r["command_id"], state="ACK")]))
        self.assertIn("MOVE 송신", await self.client.execute(target()))
        record = self.client.current
        self.assertTrue(await until(lambda: record.state == "ACK"))
        self.assertEqual((record.acceptance, record.result, len(self.peer.moves())), ("first_ack", "수락 ACK (도달 아님)", 1))
        for state in ("PLANNING", "MOVING"):
            self.peer.push(busy(record.command_id, state=state))
        self.peer.push(status(command_id=record.command_id, state="REACHED", reach_error={"position_mm": 0.8, "rotation_deg": 0.2}))
        self.assertTrue(await until(lambda: record.finished))
        self.assertEqual((record.result, record.reach_error["position_mm"], self.client.kpi["reached"]), ("도달 REACHED", 0.8, 1))
        self.assertEqual(self.client.kpi["first_ack"], 1)

    async def test_second_click_does_not_create_second_command(self):
        await self.start(self.script(lambda r: [busy(r["command_id"], state="ACK")]))
        await self.client.execute(target())
        self.assertIn("결과 미확인 명령 있음", await self.client.execute(target()))
        self.assertTrue(await until(lambda: len(self.peer.moves()) == 1))
        await asyncio.sleep(0.05)
        self.assertEqual(len(self.peer.moves()), 1)

    async def test_ack_loss_resends_same_bytes_then_marks_unconfirmed(self):
        await self.start(self.script(lambda r: None))  # MOVE 에 답하지 않음
        await self.client.execute(target())
        record = self.client.current
        self.assertTrue(await until(lambda: record.acceptance == "unconfirmed", 3))
        moves = self.peer.moves()
        self.assertEqual(len(moves), 3)  # 첫 송신 + 재전송 2회(max_resend)
        self.assertEqual(len({json.dumps(m, sort_keys=True) for m in moves}), 1)  # 같은 ID·같은 내용
        self.assertEqual((self.client.kpi["resend"], self.client.kpi["unconfirmed"]), (2, 1))
        self.assertIn("새 ID 를 자동으로 만들지 않음", record.result)

    async def test_ack_after_resend_is_counted_separately(self):
        seen = []

        def on_move(request):
            seen.append(request)
            return [busy(request["command_id"], state="ACK")] if len(seen) == 2 else None
        await self.start(self.script(on_move))
        await self.client.execute(target())
        record = self.client.current
        self.assertTrue(await until(lambda: record.accepted))
        self.assertEqual((record.acceptance, self.client.kpi["first_ack"], self.client.kpi["accepted_after_resend"]),
                         ("accepted_after_resend", 0, 1))

    async def test_backward_state_is_logged_but_not_applied(self):
        await self.start(self.script(lambda r: [busy(r["command_id"], state="ACK"), busy(r["command_id"], state="MOVING")]))
        await self.client.execute(target())
        record = self.client.current
        self.assertTrue(await until(lambda: record.state == "MOVING"))
        self.peer.push(busy(record.command_id, state="PLANNING"))
        self.assertTrue(await until(lambda: any("앞 단계" in r["note"] for r in self.rows("RX", "STATUS"))))
        self.assertEqual(record.state, "MOVING")

    async def test_reject_marks_command_rejected(self):
        await self.start(self.script(lambda r: [status(robot_ready=False, not_ready_reasons=["SCENE_NOT_READY"],
                                                       command_id=r["command_id"], code="NOT_READY")]))
        await self.client.execute(target())
        record = self.client.current
        self.assertTrue(await until(lambda: record.rejected))
        self.assertEqual((record.result, self.client.kpi["rejected"], record.state), ("거부 · NOT_READY", 1, None))

    async def test_reconnect_queries_instead_of_resending_move(self):
        await self.start(self.script(lambda r: [busy(r["command_id"], state="ACK")]))
        await self.client.execute(target())
        record = self.client.current
        self.assertTrue(await until(lambda: record.state == "ACK"))
        first = self.client.conn_id
        await self.peer.drop_clients()
        self.assertTrue(await until(lambda: self.client.conn_id > first and self.client.connected, 3))
        self.assertTrue(await until(lambda: any(r["type"] == "GET_STATUS" and r.get("command_id") == record.command_id
                                                for r in self.peer.requests)))
        self.assertEqual(len(self.peer.moves()), 1)  # MOVE 자동 재전송 없음

    async def test_cancel_only_requests_and_reports_completion_before_cancel(self):
        await self.start(self.script(lambda r: [busy(r["command_id"], state="ACK")]))
        self.assertEqual(await self.client.cancel(), "취소 대상 명령 없음")
        await self.client.execute(target())
        record = self.client.current
        await until(lambda: record.state == "ACK")
        self.assertEqual(await self.client.cancel(), "취소 요청 전송 · 종료·정지 확인 중")
        self.assertFalse(record.finished)  # 요청 전송만으로 정지 완료가 아니다
        self.assertTrue(await until(lambda: any(r["type"] == "CANCEL" for r in self.peer.requests)))
        self.peer.push(status(command_id=record.command_id, state="REACHED", reach_error={"position_mm": 0.1, "rotation_deg": 0.0}))
        self.assertTrue(await until(lambda: record.finished))
        self.assertEqual(record.result, "취소 요청 전에 이동이 완료됨")

    async def test_cancel_while_disconnected_is_unsent(self):
        await self.start(self.script(lambda r: [busy(r["command_id"], state="ACK")]))
        await self.client.execute(target())
        await until(lambda: self.client.current.state == "ACK")
        await self.client.disconnect()
        self.assertIn("미전송", await self.client.cancel())
        self.assertEqual(self.client.current.cancel, "unsent")
        self.assertFalse(any(r["type"] == "CANCEL" for r in self.peer.requests))


if __name__ == "__main__":
    unittest.main()
