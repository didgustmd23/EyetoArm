"""TCP 1.2 후보(명세 부록 A.6, 미승인) D 클라이언트 시험. 루프백 대역 상대만 쓴다(실제 C·로봇·Gazebo 없음).

대역 상대는 이 파일이 정한 줄을 그대로 보내는 시험용 소켓이다. C 구현이 아니다.
마지막 묶음은 1.1 만 아는 상대(모든 요청에 1.1 STATUS BAD_FORMAT, 이전 C 1.1 동작)를 루프백 대역으로 두고
1.2 요청이 거부되고 조작 관문이 열리지 않음을 본다(최신 C 구현 버전에 의존하지 않음).
"""
import asyncio
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "dashboard-client" / "src")]
from PySide6.QtCore import QCoreApplication, QEvent  # noqa: E402
from PySide6.QtNetwork import QAbstractSocket  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from eyetoarm.gui.client import StatusLink  # noqa: E402
from eyetoarm.tcp import protocol  # noqa: E402

FAST = dict(readiness_poll_ms=30, readiness_timeout_ms=300, status_watchdog_ms=400, reconnect_ms=40,
            connect_timeout_ms=300, run_reply_timeout_ms=150, run_max_resend=2, e_stop_timeout_ms=400)
SESSION = str(uuid.uuid4())
JOB = str(uuid.uuid4())
MAX = protocol.MAX_SAFE_INT


def until(predicate, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    QCoreApplication.processEvents()
    return predicate()


def snap(state="IDLE", rev=12, allowed=("START",), session=SESSION, hold="DETACHED", **extra):
    owner = {"kind": "SORT_RUN", "id": JOB} if state in ("RUNNING", "STOP_REQUESTED") else None
    base = {"c_session_id": session, "run_revision": rev, "run_state": state,
            "sort_job_id": JOB if state != "IDLE" else None, "run_epoch": 7, "hold_state": hold,
            "stop_cause": "USER_STOP" if state in ("STOP_REQUESTED", "STOPPED") else None,
            "slot_owner": owner, "allowed": list(allowed)}
    return {**base, **extra}


def ready(run, e_stop_available=None):
    owner = run["slot_owner"] if run else None
    reasons = (["COMMAND_ACTIVE"] if owner else []) + ([] if run else ["RUN_STATE_UNKNOWN"])
    message = {"version": "1.2", "type": "STATUS", "robot_ready": not reasons, "not_ready_reasons": reasons,
               "active_command_id": None, "server_time_ms": 10, "slot_owner": owner, "run": run}
    if e_stop_available is not None:   # 생략 = false
        message["e_stop_available"] = e_stop_available
    return message


def e_stop_status(request_id, state, detail=None):
    return {"version": "1.2", "type": "E_STOP_STATUS", "request_id": request_id, "state": state, "detail": detail,
            "server_time_ms": 12}


def run_status(request_id=None, outcome=None, code=None, snapshot=None, detail=None):
    return {"version": "1.2", "type": "RUN_STATUS", "request_id": request_id, "outcome": outcome, "code": code,
            "detail": detail, "snapshot": snapshot, "server_time_ms": 11}


def line(message):
    return json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"


class Peer:
    """1.2 후보 대역. GET_STATUS 에는 현재 스냅샷의 준비 STATUS, 분류 요청에는 on_request 가 돌려준 줄을 보낸다."""

    def __init__(self):
        self.snapshot = snap()
        self.e_stop_available = None   # 준비 STATUS 에 실을 값. None = 생략
        self.legacy = False            # True = 1.1 만 아는 상대: 모든 요청에 1.1 STATUS BAD_FORMAT
        self.requests = []          # 받은 원문 줄(bytes)
        self.on_request = None      # dict -> list[bytes|dict] | None(무응답)
        self.writers = []
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="v12-peer-loop", daemon=True)
        self.thread.start()
        self.port = self.call(self._start())

    def call(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout=3)

    async def _start(self):
        self.server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
        return self.server.sockets[0].getsockname()[1]

    async def _serve(self, reader, writer):
        self.writers.append(writer)
        try:
            await self._answer(reader, writer)
        except ConnectionError:   # 클라이언트가 abort 로 끊으면 정상 종료로 본다
            pass

    async def _answer(self, reader, writer):
        while raw := await reader.readline():
            self.requests.append(raw)
            request = json.loads(raw)
            if self.legacy:   # 이전 C 1.1 처럼 지원하지 않는 version 을 1.1 거부 STATUS 로 답한다
                answers = [{"version": "1.1", "type": "STATUS", "code": "BAD_FORMAT", "detail": "version은 1.1이어야 합니다",
                            "robot_ready": True, "not_ready_reasons": [], "active_command_id": None, "server_time_ms": 1}]
            elif request["type"] == "GET_STATUS":
                answers = [ready(self.snapshot, self.e_stop_available)]
            else:
                answers = self.on_request(request) if self.on_request else []
            for answer in answers or ():
                writer.write(answer if isinstance(answer, bytes) else line(answer))
            await writer.drain()

    def push(self, message):
        async def send():
            writer = self.writers[-1]
            writer.write(message if isinstance(message, bytes) else line(message))
            await writer.drain()
        self.call(send())

    def drop(self):
        async def close():
            for writer in self.writers:
                writer.close()
        self.call(close())

    def sort_requests(self):
        return [json.loads(raw) for raw in self.requests if json.loads(raw)["type"] != "GET_STATUS"]

    def close(self):
        """writer 를 먼저 닫는다(3.12 wait_closed 는 남은 연결 처리기를 기다린다). 실패해도 loop 정지·join 은 한다."""
        if self.loop.is_closed():
            return
        async def stop():
            for writer in self.writers:
                writer.close()
            self.server.close()
            await asyncio.wait_for(self.server.wait_closed(), 2)
        try:
            self.call(stop())
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=3)
            assert not self.thread.is_alive()
            self.loop.close()


class V12Case(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="eyetoarm-v12-")
        self.addCleanup(self._remove_temp)
        self.peer = Peer()
        self.addCleanup(self.peer.close)   # setUp 단언이 실패해도 대역 스레드를 남기지 않는다
        self.link = StatusLink("127.0.0.1", self.peer.port, log_path=Path(self.temp.name) / "d.jsonl",
                               config=FAST, version="1.2")
        self.addCleanup(self._close_link)
        self.link.start()
        self.assertTrue(until(lambda: self.link.run_snapshot is not None and not self.link.is_stale()))

    def _remove_temp(self):
        folder = self.temp.name
        self.temp.cleanup()
        self.assertFalse(Path(folder).exists())

    def _close_link(self):
        self.link.stop()
        # 종료 정리: 타이머·소켓이 모두 멈췄는지 확인한다
        for timer in (self.link._poll, self.link._retry, self.link._ready_deadline, self.link._run_timer):
            self.assertFalse(timer.isActive())
        self.assertEqual(self.link._socket.state(), QAbstractSocket.SocketState.UnconnectedState)
        self.link.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()

    def log_rows(self):
        return self.link.log.rows()


class Wire(V12Case):
    def test_get_status_and_requests_are_exact_1_2_candidate_lines(self):
        self.assertEqual(json.loads(self.peer.requests[0]), {"version": "1.2", "type": "GET_STATUS"})
        self.peer.on_request = lambda r: [run_status(r["request_id"], "ACCEPTED", "RUN_STARTED",
                                                     snap("RUNNING", 13, ("STOP",)))]
        start_id = self.link.sort_request("START")
        self.assertTrue(until(lambda: self.link.run_snapshot["run_state"] == "RUNNING"))
        start = self.peer.sort_requests()[0]
        self.assertEqual(start, {"version": "1.2", "type": "START", "request_id": start_id,
                                 "expected_session": SESSION, "expected_revision": 12})
        self.assertEqual(uuid.UUID(start_id).version, 4)
        self.peer.snapshot = snap("RUNNING", 13, ("STOP",))
        stop_id = self.link.sort_request("STOP")
        self.assertTrue(until(lambda: len(self.peer.sort_requests()) == 2))
        self.assertEqual(self.peer.sort_requests()[1], {"version": "1.2", "type": "STOP", "request_id": stop_id})
        self.assertNotEqual(start_id, stop_id)

    def test_recover_needs_confirmation_and_carries_latest_revision(self):
        self.peer.snapshot = snap("RECOVERY_REQUIRED", 20, ("STOP", "RECOVER"), hold="HOLD_UNKNOWN")
        self.peer.push(run_status(snapshot=self.peer.snapshot))
        self.assertTrue(until(lambda: self.link.run_snapshot["run_revision"] == 20))
        self.assertEqual(self.link.sort_request("RECOVER"), "")
        self.assertEqual(self.link.sort_request("RECOVER", confirmed="true"), "")
        until(lambda: False, 0.1)
        self.assertEqual(self.peer.sort_requests(), [])
        request_id = self.link.sort_request("RECOVER", confirmed=True)
        self.assertTrue(until(lambda: self.peer.sort_requests()))
        self.assertEqual(self.peer.sort_requests()[0], {"version": "1.2", "type": "RECOVER", "request_id": request_id,
                                                        "expected_session": SESSION, "expected_revision": 20,
                                                        "user_confirmed": True})

    def test_manual_move_is_disabled_in_candidate_mode(self):
        self.assertFalse(self.link.can_execute())


class Validation(V12Case):
    def assert_dropped(self, message):
        before = dict(self.link.run_snapshot)
        errors = len([r for r in self.log_rows() if r.get("kind") == "형식 오류"])
        self.peer.push(message)
        self.assertTrue(until(lambda: len([r for r in self.log_rows() if r.get("kind") == "형식 오류"]) > errors))
        self.assertEqual(self.link.run_snapshot, before)

    def test_integer_lexical_types_and_ranges_drop_whole_response(self):
        for bad in (b"12.0", b"1e1", b"true", b'"12"', b"null", b"-1", str(MAX + 1).encode(), b"9" * 400,
                    b"NaN", b"Infinity", b"-Infinity", b"1e400"):
            with self.subTest(run_revision=bad):
                text = line(run_status(snapshot=snap(rev=99))).replace(b'"run_revision":99', b'"run_revision":' + bad)
                self.assert_dropped(text)
        for field in (b"run_epoch", b"server_time_ms"):
            with self.subTest(field=field):
                text = line(run_status(snapshot=snap(rev=99)))
                text = text.replace(b'"run_epoch":7', b'"run_epoch":7.0') if field == b"run_epoch" else \
                    text.replace(b'"server_time_ms":11', b'"server_time_ms":-1')
                self.assert_dropped(text)
        self.peer.push(run_status(snapshot=snap(rev=MAX)))   # 경계값 2^53-1 은 유효
        self.assertTrue(until(lambda: self.link.run_snapshot["run_revision"] == MAX))

    def test_allowed_order_duplicates_and_state_consistency(self):
        for allowed, state in ((["STOP", "START"], "RECOVERY_REQUIRED"), (["STOP", "STOP"], "RUNNING"),
                               (["MOVE"], "IDLE"), (["START"], "RUNNING"), (["RECOVER"], "RECOVERY_REQUIRED"),
                               (["STOP"], "RECOVERING")):
            with self.subTest(allowed=allowed, state=state):
                self.assert_dropped(run_status(snapshot=snap(state, 50, allowed)))

    def test_detail_code_points_and_outcome_shape(self):
        request_id = str(uuid.uuid4())
        self.link.last_request = None
        self.assert_dropped(run_status(request_id, "REJECTED", "BUSY", snap(rev=60), detail="a" * 257))
        self.assert_dropped(run_status(None, "ACCEPTED", "RUN_STARTED", snap(rev=60)))   # 알림인데 결과 필드
        self.assert_dropped(run_status(request_id, "ACCEPTED", "STALE_REVISION", snap(rev=60)))
        self.assert_dropped(run_status(request_id, "ACCEPTED", None, snap(rev=60)))
        # 보조 평면 문자를 \u 서로게이트 쌍으로 적어도 해석 뒤 코드 포인트 1개다: 256개는 반영
        self.peer.push(run_status(snapshot=snap(rev=61)))
        self.assertTrue(until(lambda: self.link.run_snapshot["run_revision"] == 61))
        self.peer.on_request = lambda r: [line(run_status(r["request_id"], "REJECTED", "BUSY", snap(rev=61)))
                                          .replace(b'"detail":null', b'"detail":"' + b"\\ud83d\\ude00" * 256 + b'"')]
        stop_id = self.link.sort_request("START")
        self.assertTrue(until(lambda: self.link.sort_requests[stop_id].final))
        self.assertEqual(self.link.sort_requests[stop_id].detail, "\U0001F600" * 256)

    def test_enum_fields_with_object_or_array_values_are_dropped_not_crashing(self):
        for key in ("run_state", "hold_state"):
            for value in ({}, [], 1, None):
                with self.subTest(key=key, value=value):
                    self.assert_dropped(run_status(snapshot=snap(rev=80, **{key: value})))
        self.assert_dropped(run_status(snapshot=snap(rev=80, slot_owner={"kind": [], "id": None})))
        self.assert_dropped(run_status(str(uuid.uuid4()), {}, "BUSY", snap(rev=80)))
        self.assert_dropped(run_status(str(uuid.uuid4()), "REJECTED", ["BUSY"], snap(rev=80)))
        self.assert_dropped({**run_status(snapshot=snap(rev=80)), "type": {"x": 1}})
        self.assert_dropped({**run_status(snapshot=snap(rev=80)), "version": ["1.2"]})

    def test_start_allowed_requires_detached_and_free_slot(self):
        self.assert_dropped(run_status(snapshot=snap(rev=90, hold="HELD")))
        self.assert_dropped(run_status(snapshot=snap(rev=90, hold="HOLD_UNKNOWN")))
        self.assert_dropped(run_status(snapshot=snap(rev=90, slot_owner={"kind": "MOVE", "id": str(uuid.uuid4())})))
        self.peer.push(run_status(snapshot=snap(rev=90, allowed=(), hold="HELD")))   # START 없이 HELD 는 유효
        self.assertTrue(until(lambda: self.link.run_snapshot["run_revision"] == 90))

    def test_deeply_nested_json_is_a_format_error(self):
        self.assert_dropped(b"[" * 2000 + b"]" * 2000 + b"\n")   # max_line_bytes 안, 재귀 한도 초과

    def test_ready_status_extension_is_validated_atomically(self):
        bad = ready(snap(rev=70))
        bad["run"]["run_revision"] = -1
        self.assert_dropped(bad)
        bad = ready(snap(rev=70, allowed=("START", "START")))
        self.assert_dropped(bad)
        bad = ready(snap(rev=70))
        bad["slot_owner"] = {"kind": "MOVE"}
        self.assert_dropped(bad)
        bad = ready(None)
        bad["not_ready_reasons"] = []
        bad["robot_ready"] = True
        self.assert_dropped(bad)   # run=null 인데 RUN_STATE_UNKNOWN 없음


class Snapshot(V12Case):
    def test_revision_order_session_change_and_null(self):
        self.peer.push(run_status(snapshot=snap(rev=30, allowed=())))
        self.assertTrue(until(lambda: self.link.run_snapshot["run_revision"] == 30))
        self.peer.push(run_status(snapshot=snap(rev=29)))           # 작으면 기록만
        self.peer.push(run_status(snapshot=snap(rev=30, allowed=("START",))))   # 같으면 무시
        until(lambda: False, 0.1)
        self.assertEqual((self.link.run_snapshot["run_revision"], self.link.run_snapshot["allowed"]), (30, []))
        other = str(uuid.uuid4())
        self.peer.snapshot = snap(rev=1, session=other)
        self.peer.push(run_status(snapshot=self.peer.snapshot))   # 세션이 바뀌면 순번과 관계없이 교체
        self.assertTrue(until(lambda: self.link.run_snapshot["c_session_id"] == other))
        self.peer.push(run_status(snapshot=snap(rev=99, session=SESSION)))   # A → B → 늦은 A: 되돌리지 않음
        self.assertTrue(until(lambda: any(r.get("kind") == "이전 세션 스냅샷" for r in self.log_rows())))
        self.assertEqual(self.link.run_snapshot["c_session_id"], other)
        self.peer.snapshot = None                                   # 유효 스냅샷 없음
        self.peer.push(run_status(snapshot=None))
        self.assertTrue(until(lambda: self.link.run_snapshot is None))
        self.assertEqual(self.link.sort_allowed("START")[0], False)
        self.assertEqual(self.link.sort_allowed("RECOVER")[0], False)
        self.assertTrue(self.link.sort_allowed("STOP")[0])          # 연결 중 STOP 예외(A.6.12)
        stop_id = self.link.sort_request("STOP")
        self.assertTrue(until(lambda: self.peer.sort_requests()))
        self.assertEqual(self.peer.sort_requests()[0], {"version": "1.2", "type": "STOP", "request_id": stop_id})

    def test_stop_request_outcome_is_not_stopped_state(self):
        self.peer.snapshot = snap("RUNNING", 13, ("STOP",))
        self.peer.push(run_status(snapshot=self.peer.snapshot))
        self.assertTrue(until(lambda: self.link.run_snapshot["run_state"] == "RUNNING"))
        self.peer.on_request = lambda r: [run_status(r["request_id"], "ACCEPTED", "STOP_REQUESTED",
                                                     snap("STOP_REQUESTED", 14, ("STOP",)))]
        stop_id = self.link.sort_request("STOP")
        self.assertTrue(until(lambda: self.link.sort_requests[stop_id].final))
        self.assertIn("정지 확인 아님", self.link.sort_requests[stop_id].result)
        self.assertEqual(self.link.run_snapshot["run_state"], "STOP_REQUESTED")
        self.peer.push(run_status(snapshot=snap("STOPPED", 15, ("START",))))   # 권위 STOPPED 는 스냅샷으로만
        self.assertTrue(until(lambda: self.link.run_snapshot["run_state"] == "STOPPED"))

    def test_states_are_not_inferred_from_requests(self):
        self.peer.on_request = lambda r: []   # 무응답
        self.link.sort_request("START")
        until(lambda: False, 0.05)
        self.assertEqual(self.link.run_snapshot["run_state"], "IDLE")


class TimeoutAndCorrelation(V12Case):
    def test_timeout_resends_same_bytes_then_unknown_and_blocks_until_new_snapshot(self):
        self.peer.on_request = lambda r: []
        request_id = self.link.sort_request("START")
        self.assertTrue(until(lambda: self.link.sort_requests[request_id].final, 3))
        raws = [raw for raw in self.peer.requests if b'"START"' in raw]
        self.assertEqual(len(raws), 1 + FAST["run_max_resend"])
        self.assertEqual(len(set(raws)), 1)                      # 같은 request_id·같은 바이트
        self.assertIn("결과 미확인", self.link.sort_requests[request_id].result)
        self.assertEqual(self.link.sort_allowed("START"), (False, "이전 요청 결과 미확인 · 새 스냅샷 대기"))
        self.peer.snapshot = snap(rev=13)
        self.peer.push(run_status(snapshot=self.peer.snapshot))
        self.assertTrue(until(lambda: self.link.sort_allowed("START")[0]))

    def test_unknown_request_id_and_stale_reply_do_not_touch_request(self):
        self.peer.on_request = lambda r: []
        request_id = self.link.sort_request("START")
        self.peer.push(run_status(str(uuid.uuid4()), "ACCEPTED", "RUN_STARTED", snap(rev=12)))
        self.assertTrue(until(lambda: any(r.get("kind") == "대응 요청 없음" for r in self.log_rows())))
        self.assertIsNone(self.link.sort_requests[request_id].outcome)
        self.peer.push(run_status(request_id, "PENDING", None, snap(rev=12)))
        self.assertTrue(until(lambda: self.link.sort_requests[request_id].outcome == "PENDING"))
        self.peer.push(run_status(request_id, "REJECTED", "STALE_REVISION", snap(rev=12)))
        self.assertTrue(until(lambda: self.link.sort_requests[request_id].outcome == "REJECTED"))
        self.peer.push(run_status(request_id, "ACCEPTED", "RUN_STARTED", snap(rev=12)))   # 처음 확정값 유지
        self.assertTrue(until(lambda: any(r.get("kind") == "요청 결과 불일치" for r in self.log_rows())))
        self.assertEqual(self.link.sort_requests[request_id].code, "STALE_REVISION")

    def test_duplicate_stop_click_is_not_a_second_physical_request(self):
        self.peer.snapshot = snap("RUNNING", 13, ("STOP",))
        self.peer.push(run_status(snapshot=self.peer.snapshot))
        self.assertTrue(until(lambda: self.link.run_snapshot["run_state"] == "RUNNING"))
        self.peer.on_request = lambda r: []
        first = self.link.sort_request("STOP")
        self.assertEqual(self.link.sort_request("STOP"), "")
        until(lambda: False, 0.05)
        self.assertEqual({json.loads(raw)["request_id"] for raw in self.peer.requests if b'"STOP"' in raw}, {first})

    def test_disconnect_closes_pending_and_never_replays(self):
        self.peer.on_request = lambda r: []
        request_id = self.link.sort_request("START")
        self.assertTrue(until(lambda: self.peer.sort_requests()))
        self.peer.drop()
        self.assertTrue(until(lambda: self.link.state == "disconnected"))
        self.assertFalse(self.link.run_verified)
        self.assertIsNone(self.link.run_snapshot)
        self.assertEqual(self.link.sort_allowed("STOP"), (False, "연결 끊김"))
        self.assertIn("자동 재전송 없음", self.link.last_request.result)
        self.assertTrue(until(lambda: self.link.state == "connected" and self.link.run_snapshot is not None))
        until(lambda: False, 0.3)
        self.assertEqual([r["request_id"] for r in self.peer.sort_requests()], [request_id])   # 재접속 뒤 자동 재전송 없음
        self.peer.push(run_status(request_id, "ACCEPTED", "RUN_STARTED", snap(rev=12)))   # 옛 연결의 늦은 답
        self.assertTrue(until(lambda: any(r.get("kind") == "대응 요청 없음" for r in self.log_rows())))

    def test_endpoint_change_clears_gate(self):
        self.assertTrue(self.link.run_verified)
        other = Peer()
        try:
            self.assertEqual(self.link.set_endpoint("127.0.0.1", other.port), "")
            self.assertFalse(self.link.run_verified)
            self.assertIsNone(self.link.run_snapshot)
            self.assertTrue(until(lambda: self.link.run_verified and self.link.port == other.port))
        finally:
            self.link.stop()
            other.close()


class Window(V12Case):
    def setUp(self):
        super().setUp()
        from eyetoarm.gui.demo import DemoView
        from eyetoarm.gui.window import MainWindow
        self.window = MainWindow(self.link, DemoView())

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        super().tearDown()

    def test_buttons_clickable_but_send_only_when_allowed_and_confirmed(self):
        w = self.window
        self.assertEqual(w.windowTitle(), "Eye To Arms · D 운영 GUI (TCP 1.2 실행 제어)")
        buttons = (w.start_button, w.stop_button, w.estop_button, w.recover_button)
        self.assertTrue(until(lambda: all(b.isEnabled() for b in buttons)))   # 연결되면 누를 수 있음
        self.assertFalse(w.move_button.isEnabled())
        self.assertIn("STOP 정지 미전송(allowed", w.sort_detail.text())        # 누르기 전 보이는 미전송 사유
        w.stop_button.click()                                    # IDLE allowed=["START"]: STOP 미전송
        until(lambda: False, 0.1)
        self.assertEqual(self.peer.sort_requests(), [])
        self.assertIn("STOP 정지 미전송 · allowed", w.request_note.text())
        self.peer.snapshot = snap("RECOVERY_REQUIRED", 20, ("STOP", "RECOVER"), hold="HOLD_UNKNOWN")
        self.peer.push(run_status(snapshot=self.peer.snapshot))
        self.assertTrue(until(lambda: "복구 필요" in w.sort_detail.text()))
        w.recover_button.click()                                 # 확인란 전: 미전송
        until(lambda: False, 0.1)
        self.assertEqual(self.peer.sort_requests(), [])
        self.assertIn("확인란", w.request_note.text())
        w.sort_confirm.setChecked(True)
        w.recover_button.click()
        self.assertTrue(until(lambda: self.peer.sort_requests()))
        self.assertEqual(self.peer.sort_requests()[0]["user_confirmed"], True)
        self.assertFalse(w.sort_confirm.isChecked())

    def test_start_button_click_sends_exact_start_and_shows_c_reply(self):
        w = self.window
        self.assertTrue(until(lambda: "START 송신" in w.sort_detail.text()))   # IDLE allowed=["START"] 반영 대기
        self.peer.on_request = lambda r: [run_status(r["request_id"], "ACCEPTED", "RUN_STARTED",
                                                     snap("RUNNING", 13, ("STOP",)))]
        w.start_button.click()                                   # 실제 버튼 → StatusLink 소켓
        self.assertTrue(until(lambda: self.peer.sort_requests()))
        sent = self.peer.sort_requests()
        self.assertEqual(len(sent), 1)
        request_id = sent[0]["request_id"]
        self.assertEqual(uuid.UUID(request_id).version, 4)
        self.assertEqual(sent[0], {"version": "1.2", "type": "START", "request_id": request_id,
                                   "expected_session": SESSION, "expected_revision": 12})
        self.assertIn("START 요청 시도", w.request_note.text())
        self.assertTrue(until(lambda: "분류 실행 중" in w.sort_detail.text()))   # 상태는 C 스냅샷으로만
        self.assertIn("RUN_STARTED", w.sort_detail.text())                       # C 요청 결과(상태와 별도 줄)

    def test_cleanup_button_sends_stop_and_estop_is_inert(self):
        w = self.window
        def holder(layout):   # STOP 버튼이 직접 들어 있는 레이아웃(제목줄)
            if layout.indexOf(w.stop_button) >= 0:
                return layout
            for i in range(layout.count()):
                child = layout.itemAt(i).layout()
                found = child is not None and holder(child)
                if found:
                    return found
            return None
        row = holder(w.centralWidget().layout())
        self.assertEqual(w.stop_button.text(), "STOP 정지")
        self.assertTrue(until(lambda: w.estop_button.text() == "E-STOP · 미전송"))   # C 알림 없음 = 미전송 표시
        self.assertEqual(row.indexOf(w.estop_button), row.indexOf(w.stop_button) + 1)   # STOP 버튼 바로 오른쪽
        self.assertIn("e_stop_available", w.estop_button.toolTip())   # C 가 알리지 않음 = 비활성
        self.peer.snapshot = snap("RUNNING", 13, ("STOP",))
        self.peer.push(run_status(snapshot=self.peer.snapshot))
        self.assertTrue(until(lambda: "STOP 정지 송신" in w.sort_detail.text()))
        self.assertTrue(w.estop_button.isEnabled())             # 누를 수 있으나 C 알림 없음 → 미전송
        self.assertIn("요청 불가", w.estop_value._full)
        w.estop_button.click()
        until(lambda: False, 0.1)
        self.assertEqual(self.peer.sort_requests(), [])
        self.assertIn("E-STOP 미전송", w.request_note.text())
        w.stop_button.click()                                   # STOP 버튼 = 기존 STOP 요청 그대로
        self.assertTrue(until(lambda: self.peer.sort_requests()))
        self.assertEqual([r["type"] for r in self.peer.sort_requests()], ["STOP"])
        self.assertIn("STOP 정지 요청 시도", w.request_note.text())
        self.assertIn("E-STOP 미전송(C 가 E-STOP 요청 가능을 알리지 않음", w.sort_detail.text())

    def test_stop_requested_label_and_disconnect_disables_all(self):
        w = self.window
        self.peer.snapshot = snap("STOP_REQUESTED", 14, ("STOP",))
        self.peer.push(run_status(snapshot=self.peer.snapshot))
        self.assertTrue(until(lambda: "정지 확인 전" in w.sort_detail.text()))
        self.assertNotIn("정지 확인됨", w.sort_detail.text())
        self.link.stop()
        self.assertTrue(until(lambda: "연결 끊김" in w.sort_detail.text()))
        self.assertFalse(any(b.isEnabled() for b in (w.start_button, w.stop_button, w.recover_button)))


class EStop(V12Case):
    def enable(self, value=True):
        self.peer.e_stop_available = value
        self.assertTrue(until(lambda: self.link.e_stop_available is value))

    def e_stops(self):
        return [r for r in self.peer.sort_requests() if r["type"] == "E_STOP"]

    def test_no_send_without_capability(self):
        for value in (None, False):
            self.peer.e_stop_available = value
            until(lambda: False, 0.1)
            self.assertFalse(self.link.e_stop_allowed()[0])
            self.assertEqual(self.link.e_stop_request(), "")
        bad = ready(snap(), True)
        bad["e_stop_available"] = "true"
        self.peer.push(bad)                        # bool 아님 = 응답 전체 폐기
        until(lambda: False, 0.1)
        self.assertFalse(self.link.e_stop_available)
        self.assertEqual(self.e_stops(), [])

    def test_one_e_stop_packet_separate_from_stop_and_c_reports_only(self):
        self.peer.snapshot = snap("RUNNING", 13, ("STOP",))
        self.enable()
        self.assertTrue(until(lambda: self.link.run_snapshot["run_state"] == "RUNNING"))
        before = dict(self.link.run_snapshot)
        request_id = self.link.e_stop_request()
        self.assertEqual(self.link.e_stop_request(), "")            # 결과 전 두 번째 요청 없음
        self.assertTrue(until(lambda: self.e_stops()))
        self.assertEqual(self.e_stops(), [{"version": "1.2", "type": "E_STOP", "request_id": request_id}])
        stop_id = self.link.sort_request("STOP")                     # 일반 STOP 은 별개 요청
        self.assertTrue(until(lambda: len(self.peer.sort_requests()) == 2))
        self.assertEqual(self.peer.sort_requests()[1], {"version": "1.2", "type": "STOP", "request_id": stop_id})
        self.peer.push(e_stop_status(str(uuid.uuid4()), "CONFIRMED"))   # 다른 ID: 무시
        self.peer.push(e_stop_status(request_id, "STOPPED"))            # 형식 위반: 폐기
        self.peer.push(e_stop_status(request_id, "REQUESTED"))
        self.assertTrue(until(lambda: self.link.e_stop.state == "REQUESTED"))
        self.assertFalse(self.link.e_stop.final)
        self.assertIn("정지 확인 아님", self.link.e_stop.result)
        self.peer.push(e_stop_status(request_id, "CONFIRMED", "C 보고"))
        self.assertTrue(until(lambda: self.link.e_stop.state == "CONFIRMED"))
        self.peer.push(e_stop_status(request_id, "REJECTED"))           # 최종 보고는 되돌리지 않음
        self.assertTrue(until(lambda: any(r.get("kind") == "E-STOP 결과 불일치" for r in self.log_rows())))
        self.assertEqual((self.link.e_stop.state, self.link.e_stop.detail), ("CONFIRMED", "C 보고"))
        self.assertTrue(any(r.get("kind") == "E-STOP 대응 요청 없음" for r in self.log_rows()))
        self.assertTrue(any(r.get("kind") == "형식 오류" for r in self.log_rows()))
        self.assertEqual(self.link.run_snapshot["run_state"], before["run_state"])   # 분류 스냅샷을 바꾸지 않음
        self.assertEqual(len(self.e_stops()), 1)

    def test_capability_drop_keeps_pending_result_but_blocks_new(self):
        self.enable()
        request_id = self.link.e_stop_request()
        self.enable(False)
        self.peer.push(e_stop_status(request_id, "REJECTED", "C 판단"))
        self.assertTrue(until(lambda: self.link.e_stop.state == "REJECTED"))
        self.assertFalse(self.link.e_stop_allowed()[0])
        self.assertEqual(self.link.e_stop_request(), "")

    def test_requested_then_silence_times_out_unknown_without_retry(self):
        self.enable()
        request_id = self.link.e_stop_request()
        self.peer.push(e_stop_status(request_id, "REQUESTED"))
        self.assertTrue(until(lambda: self.link.e_stop.final, 2))
        self.assertEqual(self.link.e_stop.state, "REQUESTED")
        self.assertIn("결과 미확인", self.link.e_stop.result)
        self.assertEqual(len(self.e_stops()), 1)
        self.assertTrue(self.link.e_stop_allowed()[0])               # 영구히 막히지 않음

    def test_disconnect_closes_unknown_no_replay_and_old_reply_ignored(self):
        self.enable()
        request_id = self.link.e_stop_request()
        self.assertTrue(until(lambda: self.e_stops()))
        self.peer.drop()
        self.assertTrue(until(lambda: self.link.state == "disconnected"))
        self.assertTrue(self.link.e_stop.final)
        self.assertIn("재전송 없음", self.link.e_stop.result)
        self.assertFalse(self.link.e_stop_allowed()[0])
        self.assertTrue(until(lambda: self.link.state == "connected" and self.link.e_stop_available))
        self.peer.push(e_stop_status(request_id, "CONFIRMED"))          # 이전 연결 요청의 답
        self.assertTrue(until(lambda: any(r.get("kind") == "E-STOP 대응 요청 없음" for r in self.log_rows())))
        self.assertIsNone(self.link.e_stop.state)
        self.assertEqual(len(self.e_stops()), 1)

    def test_confirmed_result_becomes_history_after_reconnect(self):
        self.enable()
        request_id = self.link.e_stop_request()
        self.peer.push(e_stop_status(request_id, "CONFIRMED"))
        self.assertTrue(until(lambda: self.link.e_stop.state == "CONFIRMED"))
        self.assertIn("로봇·컨베이어", self.link.e_stop.result)
        self.assertIn("D 검증 아님", self.link.e_stop.result)
        self.peer.drop()
        self.assertTrue(until(lambda: self.link.state == "disconnected"))   # 끊김을 본 뒤 재접속을 기다린다
        self.assertTrue(until(lambda: self.link.state == "connected" and self.link.e_stop_available, 3))
        self.assertTrue(self.link.e_stop.result.startswith("이전 연결 기록(현재 상태 아님)"))
        self.assertEqual(self.link.e_stop.result.count("이전 연결 기록"), 1)

    def test_late_requested_after_timeout_is_ignored_but_terminal_is_kept(self):
        self.enable()
        request_id = self.link.e_stop_request()
        self.assertTrue(until(lambda: self.link.e_stop.final, 2))     # 무응답 → 미확인
        self.peer.push(e_stop_status(request_id, "REQUESTED"))
        self.assertTrue(until(lambda: any(r.get("kind") == "E-STOP 늦은 접수 무시" for r in self.log_rows())))
        self.assertIsNone(self.link.e_stop.state)
        self.assertIn("결과 미확인", self.link.e_stop.result)
        self.peer.push(e_stop_status(request_id, "CONFIRMED"))         # 같은 현재 요청의 C 최종 보고는 반영
        self.assertTrue(until(lambda: self.link.e_stop.state == "CONFIRMED"))
        self.assertIn("늦은 응답", self.link.e_stop.result)
        self.assertEqual(len(self.e_stops()), 1)

    def test_robot_and_conveyor_wording(self):
        from eyetoarm.gui.demo import DemoView
        from eyetoarm.gui.window import MainWindow
        window = MainWindow(self.link, DemoView())
        try:
            self.assertIn("요청 불가", window.estop_value._full)
            self.enable()
            self.assertTrue(until(window.estop_button.isEnabled))
            self.assertIn("요청 가능", window.estop_value._full)
            for button in (window.start_button, window.stop_button, window.estop_button):
                self.assertIn("로봇+컨베이어", button.toolTip())
                self.assertIn("로봇+컨베이어", button.accessibleDescription())
            self.assertIn("벨트 연속 운전을 약속하지 않습니다", window.start_button.toolTip())
            self.assertIn("현재 C(1.1)는 이 요청을 처리하지 않으며", window.estop_button.toolTip())
            self.assertTrue(until(lambda: window.estop_button.text() == "E-STOP"))   # 보낼 수 있을 때만 정확히 E-STOP
            self.assertEqual(window.stop_button.text(), "STOP 정지")
        finally:
            window.close()
            window.deleteLater()

    def test_status_row_requested_then_timeout_or_disconnect_is_unknown(self):
        from eyetoarm.gui.demo import DemoView
        from eyetoarm.gui.window import MainWindow
        window = MainWindow(self.link, DemoView())
        try:
            self.enable()
            row = lambda: window.estop_value._full   # noqa: E731
            request_id = self.link.e_stop_request()
            self.peer.push(e_stop_status(request_id, "REQUESTED"))
            self.assertTrue(until(lambda: row() == "C 보고 REQUESTED · 접수만"))
            self.assertTrue(until(lambda: row() == "결과 미확인 (C 접수 REQUESTED 뒤)", 2))   # 기한 초과
            self.assertTrue(until(lambda: self.link.e_stop_allowed()[0]))
            request_id = self.link.e_stop_request()
            self.peer.push(e_stop_status(request_id, "REQUESTED"))
            self.assertTrue(until(lambda: row() == "C 보고 REQUESTED · 접수만"))
            self.peer.drop()                                                          # 끊김
            self.assertTrue(until(lambda: row() == "이전 연결 · 결과 미확인 (C 접수 REQUESTED 뒤)"))
            self.assertTrue(self.link.e_stop.result.startswith("이전 연결 기록(현재 상태 아님)"))
            self.assertIn("재전송 없음", self.link.e_stop.result)
        finally:
            window.close()
            window.deleteLater()

    def test_window_renders_c_reports_and_demo_stays_inert(self):
        from eyetoarm.gui.demo import DemoLink, DemoView
        from eyetoarm.gui.window import MainWindow
        window = MainWindow(self.link, DemoView())
        try:
            self.enable()
            self.assertTrue(until(lambda: "요청 가능" in window.estop_value._full))
            window.estop_button.click()
            self.assertTrue(until(lambda: self.e_stops()))
            request_id = self.e_stops()[0]["request_id"]
            window.estop_button.click()                          # 대기 중 두 번째 누름: 미전송
            until(lambda: False, 0.1)
            self.assertEqual(len(self.e_stops()), 1)
            self.assertIn("E-STOP 미전송 · 앞선 E-STOP 요청 결과 대기 중", window.request_note.text())
            self.peer.push(e_stop_status(request_id, "REQUESTED"))
            self.assertTrue(until(lambda: "REQUESTED" in window.sort_detail.text()))
            self.assertIn("정지 확인 아님", window.sort_detail.text())
            self.peer.push(e_stop_status(request_id, "REJECTED"))
            self.assertTrue(until(lambda: "거부(REJECTED)" in window.sort_detail.text()))
        finally:
            window.close()
            window.deleteLater()
        demo = MainWindow(DemoLink(), DemoView(), demo=True)
        try:
            demo.estop_button.click()
            self.assertFalse(demo.estop_button.isEnabled())
            self.assertIn("C 안전 연동 없음", demo.estop_button.toolTip())
        finally:
            demo.close()
            demo.deleteLater()
        self.assertEqual(len(self.e_stops()), 1)


class AgainstLegacy11Peer(unittest.TestCase):
    """1.1 만 아는 상대(대역). 실제 D 소켓으로 1.2 조회를 보내고, 1.1 거부를 받으면 조작 관문이 닫힌 채인지 본다."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_candidate_is_rejected_and_gate_stays_closed(self):
        temp = tempfile.TemporaryDirectory(prefix="eyetoarm-v12-c11-")
        self.addCleanup(temp.cleanup)
        peer = Peer()
        self.addCleanup(peer.close)
        peer.legacy = True
        peer.e_stop_available = True   # 1.1 상대는 이 값을 보내지 않는다(legacy 응답이 우선)
        link = StatusLink("127.0.0.1", peer.port, log_path=Path(temp.name) / "d.jsonl", config=FAST, version="1.2")
        try:
            link.start()
            self.assertTrue(until(lambda: link.peer_mismatch, 3))
            self.assertFalse(link.run_verified)
            self.assertIsNone(link.run_snapshot)
            for action in ("START", "STOP", "RECOVER"):
                self.assertFalse(link.sort_allowed(action)[0])
                self.assertEqual(link.sort_request(action, confirmed=True), "")
            self.assertFalse(link.e_stop_allowed()[0])
            self.assertEqual(link.e_stop_request(), "")
            self.assertFalse(link.can_execute())
            sent = {json.loads(r["raw"])["type"] for r in link.log.rows("TX") if r.get("raw")}
            self.assertEqual(sent, {"GET_STATUS"})
            received = [json.loads(raw) for raw in peer.requests]
            self.assertTrue(received)
            self.assertEqual({(r["version"], r["type"]) for r in received}, {("1.2", "GET_STATUS")})   # 실제 소켓으로 받은 것
        finally:
            link.stop()
            link.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

if __name__ == "__main__":
    unittest.main()
