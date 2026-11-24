"""scripts/c_stack.launch.py 단위시험: ai_config 옵션(기본 S1-08, 상대·절대 경로, 없는 파일 거부, A 명령 전달)을 확인한다.

ROS launch 패키지가 없는 환경에서는 건너뛴다(실제 동작은 작업 사본 격리 실행으로 확인).
"""
import importlib.util
import sys
from pathlib import Path

import pytest

_actions = pytest.importorskip("launch.actions")   # 다른 시험이 넣어 둔 가짜 launch 모듈이면 여기서 건너뜀
try:
    from launch import LaunchContext  # noqa: E402
    from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction  # noqa: E402
    from ament_index_python.packages import PackageNotFoundError  # noqa: E402
except ImportError:
    pytest.skip("ROS launch 패키지가 온전하지 않음", allow_module_level=True)

_ROOT = Path(__file__).resolve().parents[2]
_S1_08 = _ROOT / "config" / "vision" / "ai_s1_08_axis_m2_viewbias.json"
_S1_09_REL = "config/vision/ai_s1_09_axis_m2_viewbias.json"


def _module():
    """런처 모듈을 불러 돌려준다(LaunchDescription은 만들지 않음)."""
    spec = importlib.util.spec_from_file_location("c_stack_under_test", _ROOT / "scripts" / "c_stack.launch.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _entities(mod):
    """LaunchDescription 항목 목록. C 패키지 환경을 불러오지 않은 셸이면 건너뛴다."""
    try:
        return mod.generate_launch_description().entities
    except PackageNotFoundError as exc:
        pytest.skip(f"robot_ws 환경 없음: {exc}")


def _context(**values):
    """런처 인자 값을 넣은 LaunchContext."""
    ctx = LaunchContext()
    base = {"start_a": "true", "start_b": "true", "auto_sort": "true", "a_python": "python3",
            "ai_config": str(_S1_08)}
    base.update(values)
    ctx.launch_configurations.update(base)
    return ctx


def _a_command(mod, ctx):
    """_live_pipeline_actions가 만든 A 실행 명령(문자열 목록)."""
    acts = mod._live_pipeline_actions(ctx, _ROOT, mod.LIVE_CONFIG, Path("/tmp/a.jsonl"), "stamp")
    proc = [a for a in acts if isinstance(a, ExecuteProcess)]
    assert len(proc) == 1
    return [t.text if hasattr(t, "text") else str(t) for part in proc[0].cmd for t in part]


def test_ai_config_default_is_s1_08():
    """ai_config 인자가 있고 기본값이 운영 모델 S1-08 설정 파일이다."""
    mod = _module()
    args = {e.name: e for e in _entities(mod) if isinstance(e, DeclareLaunchArgument)}

    assert "ai_config" in args
    assert [t.text for t in args["ai_config"].default_value] == [str(_S1_08)]
    assert mod.AI_CONFIG == _S1_08 and _S1_08.is_file()


def test_ai_config_check_comes_before_any_process():
    """ai_config 확인이 정리 guard·Gazebo·C 노드보다 먼저 온다(없는 파일이면 아무것도 안 띄움)."""
    mod = _module()
    ents = _entities(mod)
    check_i = next(i for i, e in enumerate(ents) if isinstance(e, OpaqueFunction) and e._OpaqueFunction__function is mod._check_ai_config)
    first_proc = min(i for i, e in enumerate(ents) if isinstance(e, (ExecuteProcess, IncludeLaunchDescription)))
    first_opaque = next(i for i, e in enumerate(ents) if isinstance(e, OpaqueFunction))

    assert check_i == first_opaque < first_proc


def test_default_value_goes_to_a_command():
    """옵션을 주지 않으면 A 명령의 --ai-config가 S1-08 경로다(지금과 같은 동작)."""
    mod = _module()
    cmd = _a_command(mod, _context())

    assert cmd[cmd.index("--ai-config") + 1] == str(_S1_08)


def test_relative_path_resolves_from_repo_top():
    """상대 경로는 저장소 맨 위 기준으로 풀려 A 명령의 --ai-config에 그대로 들어간다."""
    mod = _module()
    cmd = _a_command(mod, _context(ai_config=_S1_09_REL))

    assert cmd[cmd.index("--ai-config") + 1] == str(_ROOT / _S1_09_REL)


def test_absolute_path_is_kept(tmp_path):
    """절대 경로는 바꾸지 않고 A 명령의 --ai-config에 그대로 들어간다."""
    mod = _module()
    cfg = tmp_path / "other_ai.json"
    cfg.write_text("{}")
    cmd = _a_command(mod, _context(ai_config=str(cfg)))

    assert cmd[cmd.index("--ai-config") + 1] == str(cfg)


@pytest.mark.parametrize("value", ["config/vision/no_such_ai.json", "/nonexistent/ai.json", "", "  "])
def test_missing_file_is_rejected_with_path(value):
    """없는 파일·빈 값은 받은 값과 찾아본 경로를 담은 FileNotFoundError로 거부한다."""
    mod = _module()

    with pytest.raises(FileNotFoundError) as exc:
        mod.resolve_ai_config(value, _ROOT)

    if value.strip():
        assert value in str(exc.value)
        assert "찾아본 경로" in str(exc.value)


def test_check_rejects_missing_file_and_logs_existing():
    """확인 단계는 없는 파일이면 멈추고, 있으면 어떤 설정으로 띄우는지 로그 한 줄을 남긴다."""
    mod = _module()

    with pytest.raises(FileNotFoundError):
        mod._check_ai_config(_context(ai_config="config/vision/no_such_ai.json"))

    acts = mod._check_ai_config(_context(ai_config=_S1_09_REL))
    assert len(acts) == 1
    assert str(_ROOT / _S1_09_REL) in "".join(t.text for t in acts[0].msg)


def test_start_a_false_skips_check():
    """start_a:=false면 ai_config를 쓰지 않으므로 없는 파일이어도 확인하지 않는다."""
    mod = _module()

    assert mod._check_ai_config(_context(start_a="false", start_b="false", auto_sort="false",
                                         ai_config="config/vision/no_such_ai.json")) == []
    assert mod._live_pipeline_actions(_context(start_a="false", start_b="false", auto_sort="false",
                                               ai_config="config/vision/no_such_ai.json"),
                                      _ROOT, mod.LIVE_CONFIG, Path("/tmp/a.jsonl"), "stamp") == []
