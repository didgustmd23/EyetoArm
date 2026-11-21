"""시험 handoff 격리: 시험 기본 연결부는 송신하지 않고, 실제 연결부 확인은 빈 포트로만 한다."""
from __future__ import annotations

import importlib.util
import os
import socket
import unittest
from unittest import mock

from eyetoarm.vision.c_handoff_probe import HandoffProbe
from eyetoarm.vision.handoff import AToCHandoff
from handoff_isolation import IsolatedHandoffProbe, NullForwarder, isolate_cli, isolated_live_handoff
from test_run_pipeline import SCRIPT
from test_run_pipeline_live_observation import Base


def free_port() -> int:
    """지금 비어 있는 로컬 포트(닫은 뒤 돌려줌, 아무도 듣지 않음)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def load_cli():
    """격리하지 않은 run_pipeline 모듈."""
    spec = importlib.util.spec_from_file_location("run_pipeline_isolation_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


class IsolatedProbeTest(unittest.TestCase):
    """송신 없는 대역."""

    def test_isolated_probe_has_null_forwarder_and_real_one_closed(self):
        """대역 probe는 `NullForwarder`를 쓰고(생성 때 만든 실제 forwarder는 생성자 안에서 닫음), 정지하면 대역도 닫힌다."""
        probe = IsolatedHandoffProbe()

        self.assertIsInstance(probe._forwarder, NullForwarder)
        probe.stop()
        self.assertTrue(probe._forwarder.closed)

    def test_isolate_cli_replaces_both_defaults(self):
        """`isolate_cli`는 실시간 기록 경로 공장과 기본 --ros-link 경로 probe를 모두 바꾼다(그 모듈 객체만)."""
        mod, other = load_cli(), load_cli()
        isolate_cli(mod)

        handoff, probe = mod._default_live_handoff()
        self.addCleanup(probe.stop)

        self.assertIsInstance(handoff, AToCHandoff)
        self.assertIsInstance(probe, IsolatedHandoffProbe)
        self.assertIs(mod.HandoffProbe, IsolatedHandoffProbe)
        self.assertIs(other.HandoffProbe, HandoffProbe)   # 다른 모듈 객체에는 새지 않음

    def test_live_base_is_isolated(self):
        """실시간 CLI 시험 공통 `Base`는 격리된 기본 연결부를 쓴다(9개 시험 파일이 이 Base를 거침)."""
        class T(Base):
            def runTest(self):
                """자리."""

        t = T()
        t.setUp()
        self.addCleanup(t.doCleanups)

        self.assertIs(t.mod._default_live_handoff, isolated_live_handoff)
        self.assertIs(t.mod.HandoffProbe, IsolatedHandoffProbe)


class RealHandoffOnFreePortTest(unittest.TestCase):
    """실제 연결부를 일부러 확인하는 시험: 운영 포트 대신 빈 포트(환경 변수)로만."""

    def test_default_live_handoff_targets_configured_port(self):
        """격리하지 않은 `_default_live_handoff`는 실제 `HandoffProbe`이고, forwarder 목적지·표시 주소가 환경 변수 포트다."""
        port = free_port()

        with mock.patch.dict(os.environ, {"EYETOARM_DISPLAY_INGRESS_PORT": str(port)}):
            mod = load_cli()
            handoff, probe = mod._default_live_handoff()
            address = mod._handoff_address()

        self.addCleanup(probe.stop)

        self.assertIs(type(probe), HandoffProbe)
        self.assertEqual((probe._forwarder.host, probe._forwarder.port), ("127.0.0.1", port))
        self.assertEqual(address, f"127.0.0.1:{port}")
        self.assertNotEqual(port, 47093)


if __name__ == "__main__":
    unittest.main()
