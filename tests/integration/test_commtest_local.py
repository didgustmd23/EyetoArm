"""#183 통신 시험 도구 통합 시험: 수정하지 않은 C TcpServer(로컬 모의 상대, 포트 0) + D TcpClient 재사용으로 S1~S4.

근거: `_workspace/183_builder_comm_test_tool_plan.md` r0 §3·§7, TCP 명세서 §7.2~7.3·§8.3~8.7·§13.1, 협업 요청 D-07.
- 로컬 모의 상대 결과다(실제 C 서버·로봇 아님). 실서버·47090·ROS·Gazebo를 쓰지 않는다.
- 각 시나리오는 Runner의 케이스 상한(20 s) 안에서 끝나고, 정리 실패는 실패로 본다.
"""
import asyncio
import tempfile
import unittest
from pathlib import Path

from eyetoarm.commtest.inject import FaultRelay, Injector
from eyetoarm.commtest.local_peer import LocalPeer
from eyetoarm.commtest.runner import Runner, _Session, RunOutcome

ROOT = Path(__file__).resolve().parents[2]


class LocalPeerBase(unittest.TestCase):
    """로컬 모의 상대를 띄우고 정리한다."""

    def setUp(self):
        """상대 시작(포트 0)."""
        self.peer = LocalPeer(ROOT)
        self.host, self.port = self.peer.start()
        self.assertNotEqual(self.port, 47090)

    def tearDown(self):
        """상대 정리. 정리 실패는 시험 실패."""
        self.assertEqual(self.peer.close(), [])

    def run_scenarios(self, names, *, n=1, test_mode=True):
        """Runner 실행 → RunOutcome."""
        runner = Runner(host=self.host, port=self.port, peer=self.peer, peer_kind="local_mock", scenarios=names, n=n,
                        test_mode=test_mode, events_dir=Path(tempfile.mkdtemp()))
        out = asyncio.run(runner.run())

        self.assertEqual((out.cleanup_errors, out.callback_errors, out.tracker_errors), ([], [], []))

        return out

    def assert_all_match(self, out, cases):
        """기대 케이스가 모두 있고 모든 행이 일치한다."""
        bad = [(r["case"], r.get("note"), r["acceptance"], r["final_state"], r["code"]) for r in out.rows
               if r["outcome"] != "match"]

        self.assertEqual(bad, [])
        self.assertTrue(set(cases) <= {r["case"] for r in out.rows}, {r["case"] for r in out.rows})


class ScenarioTest(LocalPeerBase):
    """S1~S4 기대 결과."""

    def test_s1_normal(self):
        """S1: MOVE 20회 모두 first_ack·REACHED, 실행 20회, 지연이 측정된다."""
        out = self.run_scenarios(["S1"], n=20)

        self.assert_all_match(out, ["S1"])
        self.assertEqual(len(out.rows), 20)
        self.assertEqual(self.peer.executions, 20)
        self.assertTrue(all(r["ack_latency_ms"] is not None and r["final_latency_ms"] >= r["ack_latency_ms"]
                            for r in out.rows))

    def test_s2_dedupe(self):
        """S2: 같은 내용(001·002·004)은 저장 상태·실행 없음, 다른 내용(003·005~008)은 ID_CONFLICT·실행 없음."""
        out = self.run_scenarios(["S2"])

        self.assert_all_match(out, ["S2a", "S2c_pos", "S2b_pos", "S2c_theta_wrap", "S2b_theta", "S2b_image_seq",
                                    "S2b_calibration", "S2b_timestamp_base"])
        self.assertEqual(self.peer.executions, 2)          # 원본 2건만 실행

    def test_s3_reconnect_resend_unconfirmed(self):
        """S3: accepted_by_status·accepted_after_resend·unconfirmed가 실제로 나오고 재접속 후 같은 MOVE는 실행되지 않는다."""
        out = self.run_scenarios(["S3"])

        self.assert_all_match(out, ["S3a", "S3b", "S3c", "S3d"])
        got = {r["case"]: r["acceptance"] for r in out.rows if r["via"] == "client"}
        self.assertEqual(got, {"S3a": "accepted_by_status", "S3b": "accepted_after_resend", "S3c": "unconfirmed"})
        self.assertEqual(self.peer.executions, 3)
        self.assertTrue(all(e["fault_injected"] for e in out.injector_log))

    def test_s4_cancel_busy(self):
        """S4: 이동 중 CANCEL → FAILED CANCELLED, 다른 ID → BUSY(두 번째 D 클라이언트는 미전송), 자기 재전송 → MOVING."""
        out = self.run_scenarios(["S4"])

        self.assert_all_match(out, ["S4a", "S4b", "S4c"])
        self.assertEqual(self.peer.executions, 2)

    def test_without_test_mode_skips_injection(self):
        """시험 모드가 아니면 S2·S3·S4b/c는 실행하지 않고 건너뜀, 주입 사건 0."""
        out = self.run_scenarios(["S1", "S2", "S3", "S4"], test_mode=False)

        self.assertIn("S2a", out.skipped)
        self.assertIn("S3a", out.skipped)
        self.assertIn("S4b", out.skipped)
        self.assertEqual(out.injector_log, [])
        self.assertEqual({r["case"] for r in out.rows}, {"S1", "S4a"})


class SafetyTest(unittest.TestCase):
    """주입기·중계기 안전장치와 콜백 예외 격리(D-07)."""

    def test_injectors_require_test_mode(self):
        """test_mode가 True가 아니면 주입기·중계기 생성이 ValueError."""
        for make in (lambda: Injector("127.0.0.1", 1, test_mode=False, log=[]),
                     lambda: FaultRelay("127.0.0.1", 1, test_mode=False, log=[]),
                     lambda: Injector("127.0.0.1", 1, test_mode=1, log=[])):
            with self.subTest(), self.assertRaises(ValueError):
                make()

    def test_listener_error_contained(self):
        """추적기가 예외를 내도 세션 listener는 예외를 내지 않고 callback_errors에 남긴다."""
        out = RunOutcome()
        s = _Session(out, None, "t", "127.0.0.1", 1)

        def boom(row):
            """항상 실패."""
            raise RuntimeError("listener bug")

        s.tracker.on_row = boom
        s.log.add("LOCAL", "x")

        self.assertEqual(len(out.callback_errors), 1)
        self.assertIn("listener bug", out.callback_errors[0])

    def test_cli_refuses_c_server(self):
        """CLI는 --peer c_server를 rc 2로 거부한다(실서버 범위 밖)."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("rct", ROOT / "scripts" / "run_comm_test.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        self.assertEqual(mod.main(["--peer", "c_server", "--out-root", tempfile.mkdtemp()]), 2)


if __name__ == "__main__":
    unittest.main()
