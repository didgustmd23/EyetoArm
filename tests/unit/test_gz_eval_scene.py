"""scripts/gz_eval_scene.py 단위 시험(Gazebo 실행 없음)."""
import hashlib
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))
_SPEC = importlib.util.spec_from_file_location("gz_eval_scene", _ROOT / "scripts" / "gz_eval_scene.py")
sc = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = sc
_SPEC.loader.exec_module(sc)


def test_eval_seed_space_is_separated():
    assert sc.eval_seed(0) == sc.EVAL_SEED_BASE and sc.is_eval_seed(sc.eval_seed(123))
    assert not sc.is_eval_seed(0) and not sc.is_eval_seed(20261112)  # 재분할·학습 시드와 겹치지 않음
    with pytest.raises(ValueError):
        sc.eval_seed(-1)


def test_plan_is_deterministic_and_batches_differ():
    a, b = sc.plan_batch(3, 5), sc.plan_batch(3, 5)
    assert sc.batch_to_json(a) == sc.batch_to_json(b)
    assert sc.batch_to_json(sc.plan_batch(4, 5)) != sc.batch_to_json(a)


def test_parts_on_belt_in_view_and_separated():
    for batch in range(5):
        b = sc.plan_batch(batch, 8)
        for s in b.samples:
            assert 1 <= len(s.parts) <= 6
            slots = [p.slot for p in s.parts]
            assert len(set(slots)) == len(slots)
            for p in s.parts:
                assert sc.BELT_X[0] < p.xyz[0] < sc.BELT_X[1] and sc.BELT_Y[0] < p.xyz[1] < sc.BELT_Y[1]
                u, v, _ = sc.project(b.cam_xyz, b.cam_rpy, p.xyz)
                assert 0 <= u < sc.IMG_W and 0 <= v < sc.IMG_H
                assert p.yolo_cls == (1 if p.cls == "NUT" else 0)
            for i, p in enumerate(s.parts):
                for q in s.parts[i + 1:]:
                    assert math.hypot(p.xyz[0] - q.xyz[0], p.xyz[1] - q.xyz[1]) >= 0.035 - 1e-9


def test_camera_variation_and_aim_center():
    b = sc.plan_batch(0, 1)
    u, v, _ = sc.project(b.cam_xyz, b.cam_rpy, b.cam_aim)
    assert u == pytest.approx(640, abs=1.0) and v == pytest.approx(360, abs=1.0)
    assert 0.40 <= b.cam_xyz[2] <= 0.65 and 0.75 <= b.light_diffuse <= 1.0


def test_quat_matches_matrix():
    r, p, y = 0.3, -0.7, 2.1
    qx, qy, qz, qw = sc.rpy_to_quat(r, p, y)
    m = sc.rpy_to_matrix(r, p, y)
    # 쿼터니언 → 행렬
    mq = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                   [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                   [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
    assert np.allclose(m, mq, atol=1e-9)


def test_slot_labels_unique():
    t = sc.slot_table()
    assert len(t) == 18 and sorted(lab for _, lab in t.values()) == list(range(1, 19))


def test_build_world_copy_keeps_source(tmp_path):
    src_hash = hashlib.sha256(sc.WORLD_SRC.read_bytes()).hexdigest()
    w = sc.build_world(sc.plan_batch(1, 2))
    assert hashlib.sha256(sc.WORLD_SRC.read_bytes()).hexdigest() == src_hash
    assert w.count('name="gz::sim::systems::Label"') == 18
    assert 'type="segmentation"' in w and "<segmentation_type>semantic</segmentation_type>" in w
    assert "part_00" not in w and "appearance_check_cam" not in w
    assert w.count("</world>") == 1 and "<static>true</static>" in w
