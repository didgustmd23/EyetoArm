"""통신 시험 실행기(#183 계획 r0 §3·§6). D `TcpClient`를 그대로 재사용해 S1~S4를 실행한다.

- 실행기가 asyncio 루프를 소유하고 루프 안에서 `TcpClient.connect()`, 끝에 `await disconnect()`(D-06 D 답신).
- 시나리오 케이스마다 새 `TcpClient`·`EventLog`(파일 `events_<케이스>.jsonl`)를 쓴다. 미확인 명령이 다음 케이스의 송신을 막지 않게
  하려는 것이다(D 클라이언트는 결과 미확인 명령이 있으면 새 MOVE를 막는다).
- notify·listener는 예외를 내지 않는다(D-07). 콜백 오류는 계수·첫 사유만 남긴다.
- 모든 대기에 상한이 있다. 케이스가 시간 안에 끝나지 않으면 불일치 행(`note=timeout`)으로 남기고 정리한다.
- 정리(`finally`): 세션 disconnect, 주입기·중계기 닫기. 정리 실패는 `cleanup_errors`에 남고 실행 실패가 된다(reviewer 보완 4).
- 로컬 모의 상대 조절(`peer.set_executor`·`peer.executions`)은 스레드 안전한 메서드만 쓴다. 상대 루프에 직접 손대지 않는다.
- 블로킹 파일 쓰기는 하지 않는다(결과 파일은 호출자가 루프 밖에서 쓴다). D `EventLog`의 줄 단위 동기 쓰기는 D 코드다.
"""
from __future__ import annotations

import asyncio
import dataclasses
import time
from dataclasses import dataclass, field
from pathlib import Path

from eyetoarm.commtest.inject import FaultRelay, Injector
from eyetoarm.commtest.local_peer import ExecutorSettings
from eyetoarm.commtest.records import CommandTracker, client_command_row, injector_command_row
from eyetoarm.commtest.scenarios import CASES, S2_VARIANTS, Case
from eyetoarm.tcp.client import ClientConfig, TcpClient, now_ms
from eyetoarm.tcp.event_log import EventLog
from eyetoarm.tcp.protocol import MoveTarget, build_get_status, build_move, new_command_id

# 시험용 단축 시간(TCP §9 초기값과 다름. 실행 메타에 그대로 기록한다)
TEST_CLIENT_TIMING = dict(ack_timeout_ms=300, max_resend=2, readiness_poll_ms=200, readiness_timeout_ms=800,
                          status_watchdog_ms=600, connect_timeout_ms=1000, reconnect_ms=150, target_validity_ms=10000)

_CASE = {c.case_id: c for c in CASES}


def test_client_config(host: str, port: int) -> ClientConfig:
    """시험용 D ClientConfig(단축 시간)."""
    return ClientConfig(host=host, port=port, **TEST_CLIENT_TIMING)


@dataclass
class RunOutcome:
    """실행 결과(파일 쓰기 전)."""

    rows: list = field(default_factory=list)
    injector_log: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    callback_errors: list = field(default_factory=list)
    tracker_errors: list = field(default_factory=list)
    cleanup_errors: list = field(default_factory=list)
    executor_settings: dict = field(default_factory=dict)


def _target(seq: int, *, theta: float = 35.2, pos=(120.0, 80.0)) -> MoveTarget:
    """시험 목표(로컬 모의 상대 작업 범위 안)."""
    return MoveTarget(position_mm=tuple(pos), theta_deg=theta, source_id="commtest", image_seq=seq,
                      timestamp_ms=1000 + seq, timestamp_base="SIM_TIME", calibration_id="commtest_cal",
                      created_monotonic_ms=now_ms())


async def wait_until(pred, timeout_s: float, step_s: float = 0.01) -> bool:
    """조건이 참이 될 때까지 기다린다(상한). 참이면 True."""
    end = time.monotonic() + timeout_s

    while time.monotonic() < end:
        if pred():
            return True

        await asyncio.sleep(step_s)

    return bool(pred())


class _Session:
    """D TcpClient 하나(케이스 전용)."""

    def __init__(self, out: RunOutcome, events_dir: Path | None, tag: str, host: str, port: int) -> None:
        """기록 파일·추적기·클라이언트를 만든다(연결은 start)."""
        self.out, self.tag = out, tag
        self.tracker = CommandTracker()
        self.log = EventLog(None if events_dir is None else events_dir / f"events_{tag}.jsonl")
        self.log.listeners.append(self._listener)
        self.client = TcpClient(test_client_config(host, port), self.log, notify=self._notify)

    def _notify(self) -> None:
        """D notify. 아무것도 하지 않지만 예외를 내지 않음을 보장한다(D-07)."""
        try:
            return None
        except Exception as exc:  # noqa: BLE001 - 콜백 최상단(D-07)
            self.out.callback_errors.append(f"{self.tag} notify: {type(exc).__name__}: {exc}")

    def _listener(self, row) -> None:
        """D EventLog listener. 추적기 오류도 밖으로 내지 않는다(D-07)."""
        try:
            self.tracker.on_row(row)
        except Exception as exc:  # noqa: BLE001 - 콜백 최상단(D-07)
            self.out.callback_errors.append(f"{self.tag} listener: {type(exc).__name__}: {exc}")

    async def start(self, timeout_s: float = 5.0) -> bool:
        """루프 안에서 connect, 준비됨까지 기다린다."""
        self.client.connect()

        return await wait_until(lambda: self.client.robot_ready is True, timeout_s)

    async def execute_and_wait(self, target: MoveTarget, timeout_s: float):
        """MOVE 1건 → 그 CommandRecord(미전송이면 None)와 종료 여부."""
        before = len(self.client.history)
        await self.client.execute(target)

        if len(self.client.history) == before:
            return None, False

        record = self.client.history[-1]

        return record, await wait_until(lambda: record.finished, timeout_s)

    async def close(self) -> None:
        """disconnect 대기. 실패는 cleanup_errors."""
        try:
            await asyncio.wait_for(self.client.disconnect(), 3.0)
        except Exception as exc:  # noqa: BLE001 - 정리 경로: 실패를 실행 결과에 남긴다
            self.out.cleanup_errors.append(f"{self.tag} disconnect: {type(exc).__name__}: {exc}")

        self.out.tracker_errors += [f"{self.tag}: {e}" for e in self.tracker.errors]


def _row(case: Case, s: _Session, record, expected: str, ok: bool, *, fault: str | None, peer: str, delta) -> dict:
    """클라이언트 명령 기록 행."""
    return client_command_row(case=case, record=record, timing=s.tracker.timings.get(record.command_id),
                              expected=expected, outcome="match" if ok else "mismatch", fault_kind=fault, peer=peer,
                              executor_calls_delta=delta)


def _missing_row(case: Case, expected: str, note: str, peer: str) -> dict:
    """명령 기록을 만들 수 없을 때(미전송·시간 초과)의 불일치 행."""
    keys = ("command_id", "t_send_ns", "t_send_mono_ms", "t_first_status_ns", "t_first_status_mono_ms",
            "first_status_state", "t_final_ns", "t_final_mono_ms", "ack_latency_ms", "final_latency_ms", "attempts",
            "resend_count", "acceptance", "final_state", "stage", "code", "rejected_code", "fault_kind",
            "executor_calls_delta")

    return {"scenario": case.scenario, "case": case.case_id, "via": "client", **{k: None for k in keys},
            "expected": expected, "outcome": "mismatch", "fault_injected": False, "peer": peer, "note": note}


class Runner:
    """한 실행. `await run()`이 RunOutcome을 돌려준다."""

    def __init__(self, *, host: str, port: int, peer, peer_kind: str, scenarios, n: int, test_mode: bool,
                 events_dir: Path | None, case_timeout_s: float = 20.0) -> None:
        """상대 주소·조절 객체·시나리오·N·시험 모드·EventLog 폴더."""
        if n < 1:
            raise ValueError(f"n must be >= 1: {n}")

        self.host, self.port, self.peer, self.peer_kind = host, port, peer, peer_kind
        self.scenarios, self.n, self.test_mode = tuple(scenarios), n, test_mode
        self.events_dir, self.case_timeout_s = events_dir, case_timeout_s
        self.out = RunOutcome()
        self._closers: list = []
        self._seq = 0

    def _next_seq(self) -> int:
        """목표 image_seq."""
        self._seq += 1
        return self._seq

    def _set(self, settings: ExecutorSettings) -> None:
        """상대 executor 설정을 바꾸고 기록한다."""
        self.peer.set_executor(settings)
        self.out.executor_settings.setdefault("history", []).append(settings.describe())

    async def _session(self, tag: str, *, via_relay: FaultRelay | None = None, relay_port: int | None = None):
        """세션을 만들고 정리 목록에 넣는다."""
        port = relay_port if via_relay is not None else self.port
        s = _Session(self.out, self.events_dir, tag, self.host, port)
        self._closers.append(s.close)

        if not await s.start():
            raise RuntimeError(f"{tag}: client not ready within 5 s")

        return s

    async def _injector(self) -> Injector:
        """주입기(시험 모드 전용)."""
        inj = Injector(self.host, self.port, test_mode=self.test_mode, log=self.out.injector_log)
        await inj.open()
        self._closers.append(inj.close)

        return inj

    async def _relay(self) -> tuple[FaultRelay, int]:
        """중계기(시험 모드 전용)."""
        relay = FaultRelay(self.host, self.port, test_mode=self.test_mode, log=self.out.injector_log)
        port = await relay.start()
        self._closers.append(relay.close)

        return relay, port

    async def _inj_row(self, inj: Injector, case: Case, raw: bytes, command_id: str, expected: str, check, fault: str):
        """주입기 송신 1건 → 기록 행(check(reply)로 일치 판정)."""
        before = self.peer.executions
        sent, reply, got = await inj.send(raw, label=case.case_id)
        await asyncio.sleep(0.05)   # 새 실행이 생겼다면 executor 호출이 잡히도록 짧게 기다린다
        delta = self.peer.executions - before
        ok = reply is not None and check(reply, delta)
        self.out.rows.append(injector_command_row(case=case, command_id=command_id, sent_ns=sent, reply=reply,
                                                  reply_ns=got, expected=expected, outcome="match" if ok else "mismatch",
                                                  fault_kind=fault, peer=self.peer_kind, executor_calls_delta=delta))

    # ------------------------------------------------------------------ 시나리오
    async def s1(self) -> None:
        """S1 정상 N회: first_ack + REACHED."""
        case = _CASE["S1"]
        self._set(ExecutorSettings(mode="reach", moving_s=0.02))
        s = await self._session("S1")

        for _ in range(self.n):
            before = self.peer.executions
            record, done = await s.execute_and_wait(_target(self._next_seq()), 5.0)

            if record is None:
                self.out.rows.append(_missing_row(case, case.expected, "unsent", self.peer_kind))
                continue

            ok = done and record.acceptance == "first_ack" and record.state == "REACHED"
            self.out.rows.append(_row(case, s, record, case.expected, ok, fault=None, peer=self.peer_kind,
                                      delta=self.peer.executions - before))

    async def s2(self) -> None:
        """S2 같은 ID 재송신: dedupe 사례별 same → 저장 상태·실행 없음, conflict → ID_CONFLICT·실행 없음."""
        self._set(ExecutorSettings(mode="reach", moving_s=0.02))
        s = await self._session("S2")
        inj = await self._injector()
        setup = Case("S2_setup", "S2", "원본 명령", "first_ack + REACHED", "TCP §8.4", False)
        originals = {}

        for theta in (35.2, 179.998):
            before = self.peer.executions
            record, done = await s.execute_and_wait(_target(self._next_seq(), theta=theta), 5.0)

            if record is None:
                self.out.rows.append(_missing_row(setup, setup.expected, "unsent", self.peer_kind))
                return

            ok = done and record.acceptance == "first_ack" and record.state == "REACHED"
            self.out.rows.append(_row(setup, s, record, setup.expected, ok, fault=None, peer=self.peer_kind,
                                      delta=self.peer.executions - before))
            originals[theta] = record

        for case_id, _, field_name, value, expect in S2_VARIANTS:
            case = _CASE[case_id]
            base = originals[179.998 if field_name == "theta_deg" else 35.2]
            t = base.target

            if field_name is None:
                raw = base.raw                                   # 완전히 같은 바이트(dedupe-001)
            else:
                change = {"position_dx_mm": lambda: {"position_mm": (t.position_mm[0] + value, t.position_mm[1])},
                          "theta_deg": lambda: {"theta_deg": value},
                          "theta_dd": lambda: {"theta_deg": t.theta_deg + value},
                          "image_seq_d": lambda: {"image_seq": t.image_seq + value},
                          "calibration_id": lambda: {"calibration_id": value},
                          "timestamp_base": lambda: {"timestamp_base": value}}[field_name]()
                raw = build_move(base.command_id, dataclasses.replace(t, **change))

            if expect == "same":
                check = lambda r, d: r.get("state") == "REACHED" and r.get("command_id") == base.command_id and d == 0  # noqa: E731
            else:
                check = lambda r, d: r.get("code") == "ID_CONFLICT" and r.get("state") is None and d == 0  # noqa: E731

            await self._inj_row(inj, case, raw, base.command_id, case.expected, check,
                                "resend_same" if expect == "same" else "resend_changed")

    async def s3(self) -> None:
        """S3 단절·재접속(a)·재전송(b)·미확인(c)·재접속 후 같은 MOVE(d)."""
        relay, rport = await self._relay()
        cfg = test_client_config(self.host, self.port)

        # a: 직접 ACK를 삼키고 끊는다 → D가 재접속 뒤 GET_STATUS(ID)로 수락 확인
        case = _CASE["S3a"]
        self._set(ExecutorSettings(mode="reach", moving_s=0.6))
        s = await self._session("S3a", via_relay=relay, relay_port=rport)
        relay.swallow_rule = lambda m: m.get("state") == "ACK" and m.get("command_id") is not None
        relay.cut_after_swallow = True
        before = self.peer.executions
        record, done = await s.execute_and_wait(_target(self._next_seq()), 6.0)

        if record is None:
            self.out.rows.append(_missing_row(case, case.expected, "unsent", self.peer_kind))
            return

        ok = (done and record.acceptance == "accepted_by_status" and relay.move_count(record.command_id) == 1
              and record.state == "REACHED" and s.client.conn_id >= 2)
        self.out.rows.append(_row(case, s, record, case.expected, ok, fault="swallow_ack_then_cut", peer=self.peer_kind,
                                  delta=self.peer.executions - before))

        # d: 재접속 뒤 같은 MOVE를 주입기로 다시 보낸다(서버 중복 방어)
        inj = await self._injector()
        await self._inj_row(inj, _CASE["S3d"], record.raw, record.command_id, _CASE["S3d"].expected,
                            lambda r, d: r.get("state") == "REACHED" and d == 0, "resend_after_reconnect")

        # b: 그 명령 응답을 재전송이 지나갈 때까지 삼킨다(연결 유지) → accepted_after_resend
        case = _CASE["S3b"]
        self._set(ExecutorSettings(mode="reach", moving_s=1.0))
        s = await self._session("S3b", via_relay=relay, relay_port=rport)
        relay.swallow_rule = lambda m: m.get("command_id") is not None
        relay.stop_swallow_on_resend = "*"
        before = self.peer.executions
        record, done = await s.execute_and_wait(_target(self._next_seq()), 6.0)
        relay.swallow_rule, relay.stop_swallow_on_resend = None, None

        if record is None:
            self.out.rows.append(_missing_row(case, case.expected, "unsent", self.peer_kind))
        else:
            ok = (done and record.acceptance == "accepted_after_resend" and record.attempts >= 2
                  and self.peer.executions - before == 1 and record.state == "REACHED")
            self.out.rows.append(_row(case, s, record, case.expected, ok, fault="swallow_until_resend",
                                      peer=self.peer_kind, delta=self.peer.executions - before))

        # c: 그 명령 응답을 전부 삼킨다(연결 유지) → 재전송 상한 뒤 unconfirmed
        case = _CASE["S3c"]
        moving = cfg.ack_timeout_ms * (cfg.max_resend + 2) / 1000 + 0.5
        self._set(ExecutorSettings(mode="reach", moving_s=moving))
        s = await self._session("S3c", via_relay=relay, relay_port=rport)
        relay.swallow_rule = lambda m: m.get("command_id") is not None
        before = self.peer.executions
        await s.client.execute(_target(self._next_seq()))
        record = s.client.history[-1] if s.client.history else None

        if record is None:
            relay.swallow_rule = None
            self.out.rows.append(_missing_row(case, case.expected, "unsent", self.peer_kind))
            return

        got = await wait_until(lambda: record.acceptance == "unconfirmed", moving + 2.0)
        relay.swallow_rule = None
        ok = (got and record.attempts == cfg.max_resend + 1 and relay.move_count(record.command_id) == cfg.max_resend + 1
              and self.peer.executions - before == 1)
        self.out.rows.append(_row(case, s, record, case.expected, ok, fault="swallow_all", peer=self.peer_kind,
                                  delta=self.peer.executions - before))

        # 서버 쪽 실행이 끝날 때까지 주입기로 조회(다음 케이스가 BUSY를 받지 않게)
        async def finished() -> bool:
            """주입기 GET_STATUS(ID)가 종료 상태인가."""
            _, reply, _ = await inj.send(build_get_status(record.command_id), label="S3c_drain")
            return reply is not None and reply.get("state") in ("REACHED", "FAILED")

        end = time.monotonic() + moving + 3.0

        while time.monotonic() < end and not await finished():
            await asyncio.sleep(0.1)

    async def s4(self) -> None:
        """S4 이동 중 CANCEL(a), 실행 중 다른 ID → BUSY(b), 자기 명령 재전송 → 현재 상태(c)."""
        # a: 취소
        case = _CASE["S4a"]
        self._set(ExecutorSettings(mode="hold", hold_s=5.0))
        s = await self._session("S4a")
        before = self.peer.executions
        await s.client.execute(_target(self._next_seq()))
        record = s.client.history[-1] if s.client.history else None

        if record is None or not await wait_until(lambda: record.state == "MOVING", 3.0):
            self.out.rows.append(_missing_row(case, case.expected, "not moving", self.peer_kind))
            return

        await s.client.cancel()
        done = await wait_until(lambda: record.finished, 5.0)
        ok = (done and record.state == "FAILED" and record.code == "CANCELLED" and record.stage == "EXECUTION"
              and record.acceptance == "first_ack")
        self.out.rows.append(_row(case, s, record, case.expected, ok, fault=None, peer=self.peer_kind,
                                  delta=self.peer.executions - before))

        if not self.test_mode:
            self.out.skipped += ["S4b", "S4c"]
            return

        # b·c: 실행 중(hold) 명령 A
        s = await self._session("S4bc_A")
        await s.client.execute(_target(self._next_seq()))
        a = s.client.history[-1] if s.client.history else None

        if a is None or not await wait_until(lambda: a.state == "MOVING", 3.0):
            self.out.rows.append(_missing_row(_CASE["S4b"], _CASE["S4b"].expected, "A not moving", self.peer_kind))
            return

        inj = await self._injector()
        new_id = new_command_id()
        await self._inj_row(inj, _CASE["S4b"], build_move(new_id, _target(self._next_seq())), new_id,
                            "BUSY", lambda r, d: r.get("code") == "BUSY" and d == 0, "busy_other_id")

        # 같은 조건에서 두 번째 D 클라이언트는 준비 조회에서 막혀 보내지 않는다(클라이언트 정상 동작)
        b = await self._session_unready("S4b_clientB")
        await b.client.execute(_target(self._next_seq()))
        unsent = b.tracker.unsent
        row = _missing_row(_CASE["S4b"], "D 클라이언트 미전송(준비 안 됨)", "client-side block", self.peer_kind)
        row.update(command_id=unsent[-1][0] if unsent else None, outcome="match" if len(unsent) == 1 else "mismatch",
                   note=unsent[-1][1] if unsent else "no unsent row")
        self.out.rows.append(row)

        await self._inj_row(inj, _CASE["S4c"], a.raw, a.command_id, "현재 상태 MOVING",
                            lambda r, d: r.get("state") == "MOVING" and d == 0, "resend_while_active")

        await s.client.cancel()
        await wait_until(lambda: a.finished, 5.0)

    async def _session_unready(self, tag: str) -> _Session:
        """준비되지 않은 상태(활성 명령)에서 쓰는 세션: 연결·첫 준비 응답까지만 기다린다."""
        s = _Session(self.out, self.events_dir, tag, self.host, self.port)
        self._closers.append(s.close)
        s.client.connect()
        await wait_until(lambda: s.client.robot_ready is not None, 5.0)

        return s

    # ------------------------------------------------------------------ 실행
    async def run(self) -> RunOutcome:
        """선택한 시나리오를 차례로 실행하고 끝까지 정리한다."""
        table = {"S1": self.s1, "S2": self.s2, "S3": self.s3, "S4": self.s4}

        try:
            for name in self.scenarios:
                if name in ("S2", "S3") and not self.test_mode:
                    self.out.skipped += [c.case_id for c in CASES if c.scenario == name]
                    continue

                try:
                    await asyncio.wait_for(table[name](), self.case_timeout_s)
                except asyncio.TimeoutError:
                    case = next(c for c in CASES if c.scenario == name)
                    self.out.rows.append(_missing_row(case, case.expected, f"{name} timeout {self.case_timeout_s}s",
                                                      self.peer_kind))
                finally:
                    await self._close_all()
        finally:
            await self._close_all()

        return self.out

    async def _close_all(self) -> None:
        """등록된 정리 함수를 모두 부른다(역순). 실패는 cleanup_errors."""
        closers, self._closers = self._closers[::-1], []

        for close in closers:
            try:
                await asyncio.wait_for(close(), 5.0)
            except Exception as exc:  # noqa: BLE001 - 정리 경로: 다음 정리를 계속하고 실행 실패로 남긴다
                self.out.cleanup_errors.append(f"{getattr(close, '__qualname__', close)}: {type(exc).__name__}: {exc}")
