"""D 1일차 진단(eyetoarm.d_client) 단위시험.

시험 안에서 127.0.0.1 임시 포트에 일회용 대역 서버를 띄운다. 합성 응답이며 실제 C 서버 검증이 아니다.
"""

import io
import json
import socket
import sys
import threading
import time
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eyetoarm.d_client import diag  # noqa: E402
from eyetoarm.d_client.__main__ import main  # noqa: E402

READY = {"version": "1.1", "type": "STATUS", "robot_ready": True, "not_ready_reasons": [],
         "active_command_id": None, "server_time_ms": 12}
NOT_READY = dict(READY, robot_ready=False, not_ready_reasons=["RECOVERING"])


def line(message) -> bytes:
    return json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"


class FakeServer:
    """연결 하나를 받아 요청 한 줄을 기록하고 chunks 를 delay 간격으로 보낸다. close=False 면 연결 유지."""

    def __init__(self, chunks, delay=0.0, close=True, hold_s=0.0):
        self.chunks, self.delay, self.close, self.hold_s = chunks, delay, close, hold_s
        self.received = b""
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            self.sock.settimeout(5)
            conn, _ = self.sock.accept()
            with conn:
                conn.settimeout(5)
                while b"\n" not in self.received:
                    data = conn.recv(4096)
                    if not data:
                        return
                    self.received += data
                for chunk in self.chunks:
                    if self.delay:
                        time.sleep(self.delay)
                    conn.sendall(chunk)
                if not self.close:
                    time.sleep(self.hold_s)
        except OSError:
            pass

    def stop(self):
        self.sock.close()
        self.thread.join(timeout=6)
        assert not self.thread.is_alive(), "대역 서버 스레드가 끝나지 않음"


class ProbeTest(unittest.TestCase):
    def run_probe(self, chunks, timeout=2.0, **kwargs):
        server = FakeServer(chunks, **kwargs)
        try:
            result = diag.probe("127.0.0.1", server.port, timeout)
        finally:
            server.stop()
        return result, server

    def test_sends_only_no_id_get_status(self):
        result, server = self.run_probe([line(READY)])
        self.assertEqual(json.loads(server.received), {"version": "1.1", "type": "GET_STATUS"})
        self.assertEqual((result.connection, result.protocol, result.robot_ready), ("ok", "ok", True))
        self.assertEqual(result.exit_code, diag.EXIT_OK)

    def test_not_ready_is_separate_from_protocol(self):
        result, _ = self.run_probe([line(NOT_READY)])
        self.assertEqual(result.protocol, "ok")
        self.assertFalse(result.robot_ready)
        self.assertEqual(result.not_ready_reasons, ["RECOVERING"])
        self.assertEqual(result.exit_code, diag.EXIT_OK)

    def test_split_utf8_and_crlf(self):
        data = line(dict(NOT_READY, not_ready_reasons=["RECOVERING"])).replace(b"\n", b"\r\n")
        # detail 없는 준비 응답에는 한글이 없으므로 거부 응답으로 UTF-8 경계 분할을 시험한다
        reject = line({**NOT_READY, "code": "BAD_FORMAT", "detail": "한글 설명"})
        cut = reject.index("한".encode()) + 1  # 3바이트 문자의 중간
        result, _ = self.run_probe([reject[:cut], reject[cut:]], delay=0.05)
        self.assertEqual(result.protocol, "reject")
        self.assertIn("한글 설명", result.detail)
        result, _ = self.run_probe([bytes([b]) for b in data])
        self.assertEqual(result.protocol, "ok")

    def test_line_limit_4096(self):
        filler = "x" * 4096
        over = line({**NOT_READY, "code": "BAD_FORMAT", "detail": filler})
        result, _ = self.run_probe([over], close=False, hold_s=1)
        self.assertEqual(result.protocol, "line_too_long")
        # LF 없는 4097바이트도 LF 를 기다리지 않고 끊는다
        result, _ = self.run_probe([b"{" * 5000], close=False, hold_s=1)
        self.assertEqual(result.protocol, "line_too_long")

    def test_exact_4096_accepted_by_framing(self):
        body = line(READY)[:-1]
        padded = body + b" " * (4096 - len(body)) + b"\n"
        result, _ = self.run_probe([padded])
        self.assertEqual(result.protocol, "ok")

    def test_eof_before_lf(self):
        result, _ = self.run_probe([line(READY)[:-1]])
        self.assertEqual(result.protocol, "eof")
        self.assertEqual(result.exit_code, diag.EXIT_PROTOCOL)
        result, _ = self.run_probe([])
        self.assertEqual(result.protocol, "eof")

    def test_total_deadline_with_trickle(self):
        # 바이트를 계속 조금씩 보내도 총 기한에서 끝난다
        data = line(READY)
        start = time.monotonic()
        result, _ = self.run_probe([bytes([b]) for b in data], timeout=0.5, delay=0.1)
        self.assertEqual(result.protocol, "timeout")
        self.assertLess(time.monotonic() - start, 3)

    def test_silent_server_timeout(self):
        result, _ = self.run_probe([], timeout=0.3, close=False, hold_s=1)
        self.assertEqual(result.protocol, "timeout")

    def test_invalid_responses(self):
        cases = [
            b"not json\n",
            b"[1]\n",
            line({**READY, "robot_ready": False}),               # 준비 사유 불일치
            line({**READY, "extra": 1}),                         # 정의되지 않은 필드
            b'{"version":"1.1","version":"1.1"}\n',              # 중복 키
            line({**READY, "server_time_ms": 1.5}),
            b"\xff\xfe\n",                                       # UTF-8 아님
            b"\xef\xbb\xbf" + line(READY),                       # BOM
            line(READY).replace(b'"server_time_ms":12', b'"server_time_ms":NaN'),
            line(READY).replace(b'"server_time_ms":12', b'"server_time_ms":Infinity'),
            line(READY).replace(b'"server_time_ms":12', b'"server_time_ms":1e400'),
        ]
        for data in cases:
            with self.subTest(data=data[:40]):
                result, _ = self.run_probe([data])
                self.assertEqual(result.protocol, "invalid")
                self.assertIsNone(result.robot_ready)

    def test_command_state_reply_is_unexpected(self):
        cid = str(uuid.uuid4())
        busy = {**READY, "robot_ready": False, "not_ready_reasons": ["COMMAND_ACTIVE"], "active_command_id": cid,
                "command_id": cid, "state": "MOVING"}
        result, _ = self.run_probe([line(busy)])
        self.assertEqual(result.protocol, "unexpected")

    def test_trailing_bytes_recorded_not_parsed(self):
        result, _ = self.run_probe([line(READY) + b"garbage"])
        self.assertEqual((result.protocol, result.trailing_bytes), ("ok", 7))

    def test_connection_refused(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()  # 닫힌 포트
        result = diag.probe("127.0.0.1", port, 1.0)
        self.assertIn(result.connection, ("refused", "connect_timeout"))
        self.assertEqual(result.protocol, "not_attempted")
        self.assertEqual(result.exit_code, diag.EXIT_CONNECTION)

    def test_late_connect_sends_nothing(self):
        fake = mock.MagicMock()

        def slow_connect(*_args, **_kwargs):
            time.sleep(0.15)  # 접속은 성공했지만 총 기한(0.1 s)을 넘김
            return fake

        with mock.patch.object(diag.socket, "create_connection", side_effect=slow_connect):
            result = diag.probe("127.0.0.1", 47090, 0.1)
        self.assertEqual((result.connection, result.protocol), ("ok", "timeout"))
        fake.sendall.assert_not_called()
        fake.recv.assert_not_called()


class SafetyAndCliTest(unittest.TestCase):
    def test_safe_text_escapes_control_chars(self):
        self.assertNotIn("\x1b", diag.safe_text("a\x1b[2Jb\n"))
        self.assertTrue(diag.safe_text("x" * 500).endswith("…"))

    def test_probe_rejects_bad_inputs(self):
        for host, port, timeout in [("", 1, 1), ("a\x1b", 1, 1), ("localhost", 1, 1), ("no-such.invalid", 1, 1),
                                    ("xn--a", 1, 1), (None, 1, 1), ("127.0.0.1", 0, 1), ("127.0.0.1", True, 1),
                                    ("127.0.0.1", 1, 0), ("127.0.0.1", 1, float("nan")), ("127.0.0.1", 1, 61)]:
            with self.subTest(host=host, port=port, timeout=timeout), self.assertRaises(ValueError):
                diag.probe(host, port, timeout)

    def test_ip_literals_accepted(self):
        diag.validate_target("127.0.0.1", 47090, 3)
        diag.validate_target("::1", 47090, 3)

    def test_no_socket_for_env_only_and_bad_host(self):
        with mock.patch.object(diag.socket, "create_connection", side_effect=AssertionError("소켓 사용")) as conn, \
                redirect_stdout(io.StringIO()), mock.patch("sys.stderr", io.StringIO()):
            main(["--env-only"])
            self.assertEqual(main(["--host", "localhost"]), diag.EXIT_USAGE)
            self.assertEqual(main(["--host", "bad。host"]), diag.EXIT_USAGE)
        conn.assert_not_called()

    def test_cli_help_exit_zero(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--help"]), 0)

    def test_cli_env_failure_exit_matches_json(self):
        server = FakeServer([line(READY)])
        out = io.StringIO()
        try:
            with mock.patch.object(diag, "check_environment", return_value={"ok": False, "checks": {}}), \
                    redirect_stdout(out):
                code = main(["--check-env", "--json", "--port", str(server.port), "--timeout", "2"])
        finally:
            server.stop()
        printed = json.loads(out.getvalue())
        self.assertEqual((code, printed["exit_code"], printed["probe"]["protocol"]), (diag.EXIT_ENV, diag.EXIT_ENV, "ok"))

    def test_env_failure_exit_matches_json(self):
        bad_env = {"ok": False, "checks": {}}
        server = FakeServer([line(READY)])
        try:
            result = diag.probe("127.0.0.1", server.port, 2)
        finally:
            server.stop()
        record = diag.to_record(bad_env, result)
        self.assertEqual(record["exit_code"], diag.EXIT_ENV)
        self.assertEqual(record["probe"]["protocol"], "ok")

    def test_env_check_shape(self):
        env = diag.check_environment()
        self.assertEqual(set(env["checks"]), {"python", "PySide6", "jsonschema", "s2c_schema"})
        self.assertEqual(env["ok"], all(item["ok"] for item in env["checks"].values()))

    def test_bad_args(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--port", "0"]), diag.EXIT_USAGE)
            self.assertEqual(main(["--timeout", "0"]), diag.EXIT_USAGE)
            self.assertEqual(main(["--timeout", "nan"]), diag.EXIT_USAGE)
            self.assertEqual(main(["--timeout", "61"]), diag.EXIT_USAGE)
            self.assertEqual(main(["--port", "abc"]), diag.EXIT_USAGE)

    def test_cli_json_not_ready_still_success(self):
        server = FakeServer([line(NOT_READY)])
        try:
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(["--port", str(server.port), "--timeout", "2", "--json"])
        finally:
            server.stop()
        self.assertEqual(code, diag.EXIT_OK)
        printed = json.loads(out.getvalue())
        self.assertEqual((printed["probe"]["protocol"], printed["probe"]["robot_ready"]), ("ok", False))

    def test_cli_human_output(self):
        server = FakeServer([line(NOT_READY)])
        try:
            out = io.StringIO()
            with redirect_stdout(out):
                main(["--port", str(server.port), "--timeout", "2"])
        finally:
            server.stop()
        text = out.getvalue()
        self.assertIn("[접속]", text)
        self.assertIn("robot_ready=false", text)


if __name__ == "__main__":
    unittest.main()
