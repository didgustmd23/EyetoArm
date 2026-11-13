"""scripts/gz_eval_capture.py 단위 시험(Gazebo·ROS 실행 없음)."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))
_SPEC = importlib.util.spec_from_file_location("gz_eval_capture", _ROOT / "scripts" / "gz_eval_capture.py")
cap = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = cap
_SPEC.loader.exec_module(cap)
_ESPEC = importlib.util.spec_from_file_location("eval_size_recall_for_gz", _ROOT / "scripts" / "eval_size_recall.py")
ev = importlib.util.module_from_spec(_ESPEC)
sys.modules[_ESPEC.name] = ev
_ESPEC.loader.exec_module(ev)


def test_build_env_sets_partition_and_domain_without_mutating_base():
    """실행 환경에 Gazebo 파티션·ROS 도메인·모델 경로를 넣되 원래 환경 dict는 바꾸지 않고, 빈 파티션은 ValueError다."""
    base = {"PATH": "/usr/bin", "ROS_DOMAIN_ID": "0"}
    env = cap.build_env(base, "eyetoarm_gz_eval_1", 61, "/models")

    assert env["GZ_PARTITION"] == "eyetoarm_gz_eval_1" and env["ROS_DOMAIN_ID"] == "61"
    assert env["GZ_SIM_RESOURCE_PATH"] == "/models" and base["ROS_DOMAIN_ID"] == "0" and "GZ_PARTITION" not in base
    with pytest.raises(ValueError):
        cap.build_env(base, "", 61, "/m")

    assert cap.default_partition(123) == "eyetoarm_gz_eval_123"


def test_resource_parsing_and_refusal():
    """`/proc/meminfo`·nvidia-smi 출력을 읽고, 여유 메모리 부족·GPU 사용량 초과·GPU 값 없음을 거부 사유로 돌려준다."""
    assert cap.parse_meminfo_available_kib("MemTotal: 1 kB\nMemAvailable:    8849344 kB\n") == 8849344
    assert cap.parse_gpu_used_mib("2463\n") == 2463
    assert cap.check_resources(8 * 1048576, 2500) == []

    r = cap.check_resources(4 * 1048576, 9000)

    assert len(r) == 2 and "MemAvailable" in r[0] and "GPU" in r[1]
    assert cap.check_resources(8 * 1048576, None) == ["GPU 사용량을 읽지 못함"]


def test_other_gz_processes_warns_but_excludes_own():
    """다른 `gz sim` 프로세스는 경고 목록에 넣고, 자기가 띄운 PID와 gz가 아닌 프로세스는 넣지 않는다."""
    lines = ["  10 gz sim -r -s --headless-rendering /x/ax_warehouse.world",
             "  11 ruby /opt/ros/jazzy/opt/gz_tools_vendor/bin/gz sim -s world.sdf",
             "  12 python3 something", "  99 gz sim -s mine.sdf"]
    out = cap.other_gz_processes(lines, {99})

    assert len(out) == 2 and all("99 gz" not in x for x in out)


def test_stop_group_only_kills_own_new_session(tmp_path):
    """`launch_group`으로 새 세션에 띄운 그룹만 종료하고, 새 세션이 아닌 프로세스는 RuntimeError로 거부한다."""
    p = cap.launch_group(["sleep", "30"], dict(os.environ), tmp_path / "s.log")

    assert os.getpgid(p.pid) == p.pid != os.getpgrp()
    assert cap.stop_group(p) and p.poll() is not None

    # 같은 그룹(새 세션 아님)으로 띄운 프로세스는 종료를 거부
    q = subprocess.Popen(["sleep", "30"])

    try:
        with pytest.raises(RuntimeError):
            cap.stop_group(q)
    finally:
        q.kill()  # 이 시험이 띄운 자식 PID만
        q.wait()


def test_launch_group_closes_parent_log_handle(tmp_path):
    """`launch_group`은 자식 출력을 로그에 남기고, 부모의 로그 파일 핸들을 닫는다(ResourceWarning 없음, 열린 fd 없음)."""
    import gc
    import warnings
    log_path = tmp_path / "echo.log"

    # 부모가 핸들을 닫지 않고 버리면 CPython은 GC 때 닫으며 ResourceWarning(unclosed file)을 낸다
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        p = cap.launch_group(["sh", "-c", "echo hi"], dict(os.environ), log_path)
        gc.collect()

    assert not [w for w in caught if issubclass(w.category, ResourceWarning) and str(log_path) in str(w.message)]

    try:
        assert p.wait(timeout=10) == 0
    finally:
        if p.poll() is None:  # 이 시험이 띄운 자식 PID만 정리
            p.kill()
            p.wait()

    assert log_path.read_text() == "hi\n"

    # 부모 프로세스에 그 로그 파일을 가리키는 열린 디스크립터가 없어야 한다
    targets = []

    for fd in os.listdir("/proc/self/fd"):
        try:
            targets.append(os.readlink(f"/proc/self/fd/{fd}"))
        except OSError:  # listdir가 연 디렉터리 fd처럼 이미 닫힌 항목
            pass

    assert str(log_path.resolve()) not in targets


def test_pose_vector_request_covers_all_slots():
    """자세 요청은 슬롯 18개를 모두 담고, 계획한 부품은 그 위치·쿼터니언, 나머지는 대기 위치로 보낸다. 명령은 `set_pose_vector` 서비스다."""
    parts = [{"slot": "NUT_0", "xyz": [0.5, -0.2, 0.1005], "rpy": [0.0, 0.0, 1.5707963]}]
    req = cap.pose_vector_request(parts)

    assert req.count("pose {") == 18
    assert 'name: "NUT_0" position { x: 0.500000 y: -0.200000 z: 0.100500 }' in req

    import re
    m = re.search(r'name: "NUT_0".*?orientation \{ x: (\S+) y: (\S+) z: (\S+) w: (\S+) \}', req)

    assert [float(v) for v in m.groups()] == pytest.approx([0, 0, 0.7071068, 0.7071068], abs=1e-6)
    assert 'name: "BOLT_A_0" position { x: -2.000000' in req

    cmd = cap.set_pose_cmd(req)

    assert cmd[:4] == ["gz", "service", "-s", "/world/sort_cell/set_pose_vector"]


def _seg(h=60, w=80, panoptic=False):
    """분할 카메라 영상 흉내. 너트·볼트 라벨 영역과 계획에 없는 라벨 99를 그린 의미 분할(3채널 같음) 또는 파놉틱(3번째 채널) 영상."""
    lab = np.zeros((h, w), np.uint8)
    lab[10:20, 10:40] = 13   # NUT_0 (짧은 변 10)
    lab[30:36, 50:75] = 7    # BOLT_B_0 (짧은 변 6)
    lab[50:52, 0:3] = 99     # 계획에 없는 라벨

    if panoptic:
        return np.dstack([np.full_like(lab, 5), np.full_like(lab, 1), lab])

    return np.dstack([lab, lab, lab])


PARTS = [
    {"slot": "NUT_0", "label": 13, "cls": "NUT", "yolo_cls": 1, "xyz": [0, 0, 0], "rpy": [0, 0, 0],
     "cam_depth_m": 0.5, "expected_px": [25, 15]},
    {"slot": "BOLT_B_0", "label": 7, "cls": "BOLT_B", "yolo_cls": 0, "xyz": [0, 0, 0], "rpy": [0, 0, 0],
     "cam_depth_m": 0.6, "expected_px": [62, 33]},
    {"slot": "NUT_1", "label": 14, "cls": "NUT", "yolo_cls": 1, "xyz": [0, 0, 0], "rpy": [0, 0, 0],
     "cam_depth_m": 0.7, "expected_px": [5, 5]},
]


@pytest.mark.parametrize("panoptic", [False, True])
def test_labels_for_sample_polygons_and_manifest(panoptic, tmp_path):
    """보이는 부품만 YOLO 다각형 라벨과 짧은 변·넓이 기록을 받고, 안 보이는 부품은 보이지 않음, 계획 밖 라벨은 따로 나온다.

    만든 라벨은 `eval_size_recall.py`가 그대로 읽는다.
    """
    lines, infos, mode, unknown = cap.labels_for_sample(_seg(panoptic=panoptic), PARTS)

    assert mode == ("panoptic" if panoptic else "semantic") and unknown == [99]
    assert len(lines) == 2 and lines[0].startswith("1 ") and lines[1].startswith("0 ")

    nut, bolt, hidden = infos

    assert nut["visible"] and nut["labeled"] and nut["short_px"] == pytest.approx(9.0, abs=1.01)
    assert bolt["short_px"] == pytest.approx(5.0, abs=1.01) and bolt["area_px"] == 150
    assert hidden == {**{k: PARTS[2][k] for k in ("slot", "label", "cls", "yolo_cls", "xyz", "rpy", "cam_depth_m",
                                                  "expected_px")},
                      "visible": False, "area_px": 0, "fragments": 0, "labeled": False}

    # eval_size_recall.py가 그대로 읽는다
    f = tmp_path / "a.txt"
    f.write_text("".join(x + "\n" for x in lines), encoding="utf-8")

    cls, polys = ev.read_yolo_labels(f, 80, 60)

    assert cls == [1, 0]
    assert polys[0][:, 0].min() == pytest.approx(10, abs=0.01) and polys[0][:, 0].max() == pytest.approx(39, abs=0.01)


def test_label_channel_single_and_dataset_yaml():
    """1채널 분할 영상은 `single` 방식이고, `dataset.yaml`에는 절대 경로가 없이 test 경로와 클래스 이름이 있다."""
    lab = np.zeros((4, 4), np.uint8)

    assert cap.label_channel(lab)[1] == "single"

    y = cap.dataset_yaml()

    assert "path:" not in y and "test: images/test" in y and "1: nut" in y


def test_memory_low_threshold():
    """여유 메모리가 기준(GiB)보다 작으면 부족, 크면 부족이 아니다."""
    assert cap.memory_low(int(1.9 * 1048576), 2.0) and not cap.memory_low(int(2.1 * 1048576), 2.0)


def test_reset_latest_clears_previous_batch_state():
    """묶음 사이에 최근 RGB·분할·시계 메시지를 모두 None으로 비운다."""
    latest = {"rgb": object(), "seg": object(), "clock": object()}
    cap.reset_latest(latest)
    assert latest == {"rgb": None, "seg": None, "clock": None}
