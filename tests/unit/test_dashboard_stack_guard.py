"""scripts/dashboard_stack.launch.py 단위시험: 남은 Gazebo 처리(run_mode·표식·시작 전 검사)가 다른 프로세스보다 먼저 오는지 확인한다.

ROS launch 패키지가 없는 환경에서는 건너뛴다(실제 동작은 작업 사본 실제 시험으로 확인).
"""
import importlib.util
import sys
from pathlib import Path

import pytest

_actions = pytest.importorskip("launch.actions")   # 다른 시험이 넣어 둔 가짜 launch 모듈이면 여기서 건너뜀
try:
    from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,  # noqa: E402
                                OpaqueFunction, SetEnvironmentVariable, TimerAction)
except ImportError:
    pytest.skip("ROS launch 패키지가 온전하지 않음", allow_module_level=True)

_ROOT = Path(__file__).resolve().parents[2]


def _load():
    """런처 모듈을 불러 LaunchDescription 항목 목록을 돌려준다."""
    spec = importlib.util.spec_from_file_location("dashboard_stack_under_test", _ROOT / "scripts" / "dashboard_stack.launch.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod, mod.generate_launch_description().entities


def test_run_mode_argument_defaults_to_single():
    """run_mode 인자가 있고 기본값이 single이다."""
    _, ents = _load()
    args = {e.name: e for e in ents if isinstance(e, DeclareLaunchArgument)}

    assert "run_mode" in args
    assert [t.text for t in args["run_mode"].default_value] == ["single"]


def test_marker_and_guard_come_before_any_process():
    """표식 환경 변수와 시작 전 검사가 C 서버·Gazebo·중계기보다 먼저 온다(거부하면 아무것도 안 띄움)."""
    mod, ents = _load()
    kinds = [type(e) for e in ents]
    first_proc = min(i for i, k in enumerate(kinds) if k in (ExecuteProcess, IncludeLaunchDescription, TimerAction))
    marker_i = next(i for i, e in enumerate(ents) if isinstance(e, SetEnvironmentVariable)
                    and [t.text for t in e.name] == [mod.stack_guard.MARKER_ENV])
    guard_i = next(i for i, e in enumerate(ents) if isinstance(e, OpaqueFunction))

    assert marker_i < guard_i < first_proc


def test_uses_committed_stack_guard():
    """c_stack과 같은 scripts/stack_guard.py를 쓴다(따로 복사본 없음)."""
    mod, _ = _load()

    assert Path(mod.GUARD_SCRIPT) == _ROOT / "scripts" / "stack_guard.py"
