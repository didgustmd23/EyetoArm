"""#196 수정 시험(학습·시험 도구·스크립트): #193 A4-1~A4-4.

근거: `_workspace/196_reviewer_instruction_a_fixes_from_193.md`, `_workspace/193_builder2_full_source_review.md`(fd1a7942).
"""
from __future__ import annotations

import asyncio
import importlib.util
import io
import re
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from eyetoarm.commtest.inject import FaultRelay
from eyetoarm.training import config as tconfig
from eyetoarm.training import evaluation

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_pipeline.py"


def load_cli():
    """run_pipeline.py를 모듈로 불러온다."""
    spec = importlib.util.spec_from_file_location("run_pipeline_196_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


class DefaultLogOverwriteTest(unittest.TestCase):
    """A4-1: 기본 저장 입력 경로도 기존 --log를 덮어쓰지 않고 종료 2."""

    def test_existing_log_refused(self):
        """이미 있는 기록 파일이면 종료 2이고 내용은 그대로다. 새 경로면 기존대로 0."""
        with tempfile.TemporaryDirectory() as d:
            img = Path(d) / "a.png"
            cv2.imwrite(str(img), np.zeros((48, 64, 3), np.uint8))
            log = Path(d) / "run.jsonl"
            log.write_text('{"kind":"previous evidence"}\n', encoding="utf-8")
            argv = ["--images", str(img), "--detect-config", str(ROOT / "config/vision/opencv_planar.json"),
                    "--no-detect-timeout"]
            err = io.StringIO()
            code = load_cli().main([*argv, "--log", str(log)], stderr=err)

            self.assertEqual(code, 2, err.getvalue())
            self.assertEqual(log.read_text(encoding="utf-8"), '{"kind":"previous evidence"}\n')
            self.assertIn("refusing to overwrite", err.getvalue())

            fresh = Path(d) / "new.jsonl"
            self.assertEqual(load_cli().main([*argv, "--log", str(fresh)], stderr=io.StringIO()), 0)
            self.assertTrue(fresh.exists())


class HugeIntegerConfigTest(unittest.TestCase):
    """A4-2: training 설정 숫자 검사의 거대 정수는 ConfigError(유효하지 않은 기록은 False)."""

    def test_config_validators(self):
        """_pos_num·_unit은 10**400에 ConfigError, _finite는 False를 돌려준다(OverflowError 없음)."""
        with self.assertRaises(tconfig.ConfigError):
            tconfig._pos_num(10 ** 400, "lr")

        with self.assertRaises(tconfig.ConfigError):
            evaluation._unit(10 ** 400, "conf")

        self.assertFalse(evaluation._finite(10 ** 400))
        self.assertEqual(tconfig._pos_num(3, "lr"), 3.0)
        self.assertTrue(evaluation._finite(1.5))


async def _relay_round(server_line: bytes, client_line: bytes, *, rule=None):
    """상류 서버 하나·중계기 하나로 한 줄씩 주고받는다. (클라이언트가 받은 줄, 서버가 받은 줄, 사건 목록)."""
    got_server = []

    async def handle(reader, writer):
        """클라이언트 줄을 받고 서버 줄을 보낸다."""
        got_server.append(await reader.readline())
        writer.write(server_line)
        await writer.drain()
        await asyncio.sleep(0.05)
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    log = []
    relay = FaultRelay("127.0.0.1", port, test_mode=True, log=log)
    relay.swallow_rule = rule
    rport = await relay.start()
    reader, writer = await asyncio.open_connection("127.0.0.1", rport)
    writer.write(client_line)
    await writer.drain()

    try:
        got_client = await asyncio.wait_for(reader.readline(), 1.0)
    except asyncio.TimeoutError:
        got_client = None

    writer.close()
    await relay.close()
    server.close()
    await server.wait_closed()

    return got_client, got_server, log


class RelayNonObjectTest(unittest.TestCase):
    """A4-3: 중계기는 객체가 아닌 JSON 줄에서 죽지 않고, 펌프 예외는 사건으로 남긴다."""

    def test_non_object_json_passes_both_ways(self):
        """[1]·5 같은 줄도 그대로 양방향으로 지나가고 펌프 실패 사건이 없다."""
        got_client, got_server, log = asyncio.run(_relay_round(b"5\n", b"[1]\n"))

        self.assertEqual(got_server, [b"[1]\n"])
        self.assertEqual(got_client, b"5\n")
        self.assertFalse([e for e in log if e["kind"].endswith("_failed")], log)

    def test_pump_exception_recorded(self):
        """삼킴 규칙이 예외를 내면 relay_s2c_failed 사건에 예외형·내용이 남는다(버려진 작업에 묻히지 않음)."""
        def bad_rule(msg):
            """규칙 오류."""
            raise KeyError("rule bug")

        _, _, log = asyncio.run(_relay_round(b'{"type":"STATUS"}\n', b"{}\n", rule=bad_rule))
        failed = [e for e in log if e["kind"] == "relay_s2c_failed"]

        self.assertEqual(len(failed), 1, log)
        self.assertIn("KeyError", failed[0]["error"])


class UsageExampleTest(unittest.TestCase):
    """A4-4: run_pipeline.py 사용 예의 설정 경로가 실제로 있다."""

    def test_docstring_config_paths_exist(self):
        """모듈 설명에 적힌 config/vision/*.json 경로가 모두 저장소에 있다."""
        text = SCRIPT.read_text(encoding="utf-8").split('"""')[1]
        paths = set(re.findall(r"config/vision/[\w.-]+\.json", text))

        self.assertTrue(paths)

        for p in sorted(paths):
            with self.subTest(path=p):
                self.assertTrue((ROOT / p).exists(), p)


if __name__ == "__main__":
    unittest.main()
