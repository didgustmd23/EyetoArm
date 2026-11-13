"""직접 실행과 패키지 실행이 같은 CLI로 연결되는지 확인한다."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]


class EntryPointTest(unittest.TestCase):
    def run_cli(self, args, module=False):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        if module:
            env["PYTHONPATH"] = str(ROOT / "src")
        return subprocess.run(
            [sys.executable, "-B", "-X", "utf8", *args], cwd=ROOT / "docs",
            env=env, capture_output=True, text=True, encoding="utf-8", timeout=15,
        )

    def test_direct_files_without_pythonpath_from_other_directory(self):
        for name in ("diag.py", "__main__.py"):
            with self.subTest(name=name):
                result = self.run_cli([str(ROOT / "src/eyetoarm/gui" / name), "--env-only", "--json"])
                self.assertIn(result.returncode, (0, 4), result.stderr)
                record = json.loads(result.stdout)
                self.assertEqual(record["exit_code"], result.returncode)
                self.assertEqual(record["environment"]["ok"], result.returncode == 0)
                self.assertIn("s2c_schema", record["environment"]["checks"])

    def test_direct_files_share_help(self):
        for name in ("diag.py", "__main__.py"):
            with self.subTest(name=name):
                result = self.run_cli([str(ROOT / "src/eyetoarm/gui" / name), "--help"])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--env-only", result.stdout)

    def test_package_entry_points(self):
        for module in ("eyetoarm.gui", "eyetoarm.gui.diag"):
            with self.subTest(module=module):
                result = self.run_cli(["-m", module, "--env-only", "--json"], module=True)
                self.assertIn(result.returncode, (0, 4), result.stderr)
                self.assertEqual(json.loads(result.stdout)["exit_code"], result.returncode)

    def test_gui_can_import_result_without_transport_or_qt(self):
        script = "\n".join([
            "import sys, json",
            "sys.modules['socket'] = None",
            "sys.modules['PySide6'] = None",
            "sys.modules['eyetoarm.tcp.protocol'] = None",
            "from dataclasses import asdict",
            "from eyetoarm.gui.models import ProbeResult",
            "assert 'eyetoarm.gui.diag' not in sys.modules",
            "print(json.dumps(asdict(ProbeResult('127.0.0.1', 47090, 3))))",
        ])
        result = self.run_cli(["-c", script], module=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        snapshot = json.loads(result.stdout)
        self.assertEqual(snapshot["connection"], "not_attempted")
        self.assertIsNone(snapshot["robot_ready"])


if __name__ == "__main__":
    unittest.main()
