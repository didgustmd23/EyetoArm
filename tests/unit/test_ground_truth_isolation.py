"""#186 정답 격리: 런타임 패키지가 평가 패키지·정답 경로를 참조하지 않고, 평가 패키지가 런타임 패키지를 import하지 않음.

근거: 코드 규약 §5(정답은 평가 경로에서만 읽는다), 계획서 §2.1, A 상세설계 §3·§9, reviewer 계획 검토 보완 3.
AST로 import 문과 문자열 상수를 본다. 런타임이 정답을 import·구독하면 평가 전체가 무효가 되므로 결합 자체를 막는다.
"""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "eyetoarm"
# D GUI 소스는 dashboard-client 아래에 있다(2026-11-19 기준 src/eyetoarm/gui에는 .py가 없음). 둘 다 본다.
EXTRA_ROOTS = (ROOT / "dashboard-client" / "src" / "eyetoarm",)
RUNTIME_PACKAGES = ("vision", "inputs", "calibration", "transforms", "tcp", "gui", "common", "commtest")
FORBIDDEN_STRINGS = ("ground_truth", "evaluation/", "sorteval")
SORTEVAL_ALLOWED_STDLIB = {"__future__", "csv", "hashlib", "io", "json", "math", "platform", "random", "datetime",
                          "pathlib", "collections", "dataclasses"}


def py_files(package: str):
    """패키지 아래 .py 파일(src/eyetoarm와 dashboard-client 양쪽)."""
    files = []

    for base in (SRC,) + EXTRA_ROOTS:
        files.extend((base / package).rglob("*.py"))

    return sorted(files)


def rel(path: Path) -> str:
    """저장소 기준 상대 경로."""
    return str(path.relative_to(ROOT))


def imported_modules(tree: ast.AST, package: str) -> set[str]:
    """import 대상 모듈 이름(상대 import는 eyetoarm.<package> 기준 절대 이름으로)."""
    names = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = ["eyetoarm", package][: max(0, 2 - (node.level - 1))]
                base = ".".join(parts + ([node.module] if node.module else []))
                names.add(base)
                names.update(f"{base}.{a.name}" for a in node.names)
            else:
                names.add(node.module or "")
                names.update(f"{node.module}.{a.name}" for a in node.names)

    return names


def string_constants(tree: ast.AST):
    """문자열 상수(docstring 포함)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node


class GroundTruthIsolationTest(unittest.TestCase):
    """정답 격리 양방향."""

    def test_runtime_packages_exist(self):
        """검사 대상 런타임 패키지가 실제로 있고 파일이 있다(빈 검사로 통과하지 않게)."""
        for pkg in RUNTIME_PACKAGES:
            with self.subTest(pkg=pkg):
                self.assertTrue(py_files(pkg), pkg)

    def test_runtime_does_not_import_sorteval(self):
        """런타임 패키지의 어떤 파일도 eyetoarm.sorteval을 import하지 않는다."""
        offenders = []

        for pkg in RUNTIME_PACKAGES:
            for path in py_files(pkg):
                tree = ast.parse(path.read_text(encoding="utf-8"))

                if any("sorteval" in m for m in imported_modules(tree, pkg)):
                    offenders.append(rel(path))

        self.assertEqual(offenders, [])

    def test_runtime_has_no_ground_truth_paths(self):
        """런타임 패키지 문자열 상수에 ground_truth·evaluation/·sorteval이 없다(경로를 만들어 읽는 결합 방지)."""
        offenders = []

        for pkg in RUNTIME_PACKAGES:
            for path in py_files(pkg):
                tree = ast.parse(path.read_text(encoding="utf-8"))

                for node in string_constants(tree):
                    hit = [s for s in FORBIDDEN_STRINGS if s in node.value]

                    if hit:
                        offenders.append(f"{rel(path)}:{node.lineno} {hit}")

        self.assertEqual(offenders, [])

    def test_sorteval_imports_only_stdlib_and_itself(self):
        """평가 패키지는 표준 라이브러리 허용 목록과 eyetoarm.sorteval만 import한다(런타임 상태 변경 모듈 미import)."""
        offenders = []

        for path in py_files("sorteval"):
            tree = ast.parse(path.read_text(encoding="utf-8"))

            for name in imported_modules(tree, "sorteval"):
                top = name.split(".")[0]

                if name.startswith("eyetoarm.sorteval") or top in SORTEVAL_ALLOWED_STDLIB:
                    continue

                offenders.append(f"{rel(path)}: {name}")

        self.assertEqual(offenders, [])

    def test_detector_catches_planted_import(self):
        """검사기 자체 확인: 심은 import·경로 문자열을 잡는다."""
        tree = ast.parse("from eyetoarm.sorteval import sequence\nfrom ..sorteval import records\n"
                         "P = 'evaluation/ground_truth/x.json'\n")

        self.assertTrue(any("sorteval" in m for m in imported_modules(tree, "vision")))
        self.assertIn("eyetoarm.sorteval.records", imported_modules(tree, "vision"))
        self.assertTrue(any("ground_truth" in n.value for n in string_constants(tree)))


if __name__ == "__main__":
    unittest.main()
