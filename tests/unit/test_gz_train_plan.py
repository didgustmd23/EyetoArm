"""scripts/gz_train_plan.py 단위 시험(Gazebo·ROS 실행 없음)."""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))
_SPEC = importlib.util.spec_from_file_location("gz_train_plan", _ROOT / "scripts" / "gz_train_plan.py")
tp = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = tp
_SPEC.loader.exec_module(tp)

# 링크 카메라 HOME 자세(#146 gz_eval_green_link_v2 setup/camera.json, #267 TF와 같음)
HOME_T = np.array([0.060424, 0.14, 0.504538])
HOME_Q = (-0.6491866716634059, 0.6491876837400284, -0.28027964392714944, 0.2802772516900301)
K = [1171.5120697021484, 0.0, 640.0, 0.0, 1171.51216506958, 360.0, 0.0, 0.0, 1.0]


def _R(q):
    """xyzw 쿼터니언 → 회전행렬(시험용, 모듈과 독립 구현)."""
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


HOME_R = _R(HOME_Q)


def test_train_and_eval_seed_ranges_are_disjoint():
    """학습 시드는 학습 구간에만 있고, 평가 구간과 겹치지 않으며, 범위 밖 묶음 번호는 ValueError다."""
    assert tp.is_train_seed(tp.train_seed(0)) and not tp.scene.is_eval_seed(tp.train_seed(999_999))
    assert not tp.is_train_seed(tp.scene.eval_seed(0))
    with pytest.raises(ValueError):
        tp.train_seed(-1)
    with pytest.raises(ValueError):
        tp.train_seed(tp.TRAIN_SEED_SPAN)


def test_plan_rejects_seed_of_other_purpose():
    """학습 용도에 평가 시드를, 평가 용도에 학습 시드를 주면 계획을 만들지 않는다(자료 섞임 방지)."""
    with pytest.raises(ValueError):
        tp.plan_samples(1, HOME_T, HOME_R, K, 1280, 720, tp.scene.eval_seed(0), "train")
    with pytest.raises(ValueError):
        tp.plan_samples(1, HOME_T, HOME_R, K, 1280, 720, tp.train_seed(0), "eval")
    with pytest.raises(ValueError):
        tp.check_seed(tp.train_seed(0), "other")


def test_slot_table_labels_unique_and_cover_all_kinds():
    """슬롯 라벨은 1부터 연속·고유이고, 종류마다 SLOTS_PER_KIND개다."""
    st = tp.slot_table()
    labels = [lab for _, lab in st.values()]
    assert labels == list(range(1, len(tp.PARTS) * tp.SLOTS_PER_KIND + 1))
    for kind in tp.PARTS:
        assert sum(1 for k, _ in st.values() if k == kind) == tp.SLOTS_PER_KIND


def test_plan_is_deterministic_and_respects_constraints():
    """같은 시드면 계획이 같고, 부품 중점 투영이 여백 안·부품 간격 3.5 cm 이상이다."""
    a = tp.plan_samples(30, HOME_T, HOME_R, K, 1280, 720, tp.train_seed(5), "train")
    b = tp.plan_samples(30, HOME_T, HOME_R, K, 1280, 720, tp.train_seed(5), "train")
    assert a == b
    for s in a:
        assert 1 <= len(s["parts"]) <= 6
        for p in s["parts"]:
            u, v = p["expected_px"]
            assert 24 <= u < 1280 - 24 and 24 <= v < 720 - 24
        xy = [p["xyz"][:2] for p in s["parts"]]
        for i in range(len(xy)):
            for j in range(i + 1, len(xy)):
                assert math.dist(xy[i], xy[j]) >= 0.035 - 1e-3  # 기록값은 4자리 반올림


def test_horizontal_yaw_maps_to_image_horizontal_at_home():
    """HOME 링크 카메라에서 yaw 90°로 누운 볼트는 영상 각도 0°(가로), yaw 0°는 세로에 가깝다(#267 D0 자세와 같음)."""
    xyz = (0.55, 0.10, 0.1025)
    for yaw_deg, lo, hi in ((90.0, 0.0, 2.0), (0.0, 70.0, 90.0)):
        rpy = (0.0, math.pi / 2, math.radians(yaw_deg))
        ang = tp.image_axis_angle_deg(HOME_T, HOME_R, K, tp.part_mid_world("BOLT_M3X20", xyz, rpy), tp.axis_dir_world(rpy))
        assert lo <= ang <= hi


def test_horizontal_weight_raises_share_of_horizontal_bolts_and_class_mix():
    """기본 가중이면 균등(가중 1)보다 가로 볼트 비율이 높고, 가로 중 완전 가로(< 7°)가 절반 이상이며,
    M3×20이 종류 중 가장 많다(비중 0.40)."""
    def frac(hw, nw):
        s = tp.plan_samples(200, HOME_T, HOME_R, K, 1280, 720, tp.train_seed(1), "train", horiz_weight=hw, narrow_weight=nw)
        ang = np.array([p["image_angle_deg"] for x in s for p in x["parts"] if p["yolo_cls"] == 0])
        kinds = [p["cls"] for x in s for p in x["parts"]]
        return np.mean(ang < 22.5), np.sum(ang < 7.0) / np.sum(ang < 22.5), kinds, s

    f_def, narrow_share, kinds, s = frac(1.3, 4.0)
    f_uni, narrow_uni, _, _ = frac(1.0, 1.0)
    assert f_def > f_uni + 0.1
    assert narrow_share >= 0.5 > narrow_uni
    d = tp.yaw_distribution(s)
    assert d["narrow_share_of_horizontal"] == pytest.approx(narrow_share, abs=1e-3)
    assert sum(d["image_angle_bins"].values()) == d["bolts"] == sum(d["yaw_mod180_hist_22.5deg"])
    counts = {k: kinds.count(k) for k in tp.PARTS}
    assert max(counts, key=counts.get) == "BOLT_M3X20"
    assert counts["BOLT_M6X12"] < counts["NUT_M6"]


def test_yaw_weight_tiers():
    """완전 가로 < 가로 < 그 밖 순서로 가중이 크고, 각도를 모르면 1이다."""
    assert tp.yaw_weight(3.0) == 4.0 and tp.yaw_weight(10.0) == 1.3 and tp.yaw_weight(40.0) == 1.0
    assert tp.yaw_weight(None) == 1.0


def test_pose_vector_request_places_planned_and_parks_others():
    """계획한 슬롯은 그 자리에, 나머지 슬롯은 PARK 줄에 놓는 요청을 만든다(슬롯 수 = 종류 × 6)."""
    req = tp.pose_vector_request([{"slot": "BOLT_M3X20_0", "xyz": [0.5, 0.1, 0.1025], "rpy": [0.0, math.pi / 2, 0.0]}])
    assert req.count("pose {") == len(tp.PARTS) * tp.SLOTS_PER_KIND
    assert 'name: "BOLT_M3X20_0" position { x: 0.500000 y: 0.100000 z: 0.102500 }' in req
    assert f'name: "NUT_M6_0" position {{ x: {tp.PARK[0]:.6f}' in req


def test_build_world_drops_part_includes_and_adds_labeled_slots(tmp_path):
    """월드 사본은 part_0x include를 빼고, 라벨 플러그인이 붙은 슬롯을 모두 넣는다. </world>가 없으면 ValueError."""
    src = tmp_path / "w.sdf"
    src.write_text("<sdf><world>\n    <include><uri>model://x</uri><name>part_00</name></include>\n"
                   "    <include><uri>model://bin</uri><name>bin_a</name></include>\n  </world></sdf>\n", encoding="utf-8")
    text = tp.build_world(src)
    assert "part_00" not in text and "bin_a" in text
    assert text.count("gz-sim-label-system") == len(tp.PARTS) * tp.SLOTS_PER_KIND
    assert "model://bolt_m3x20" in text
    bad = tmp_path / "bad.sdf"
    bad.write_text("<sdf></sdf>", encoding="utf-8")
    with pytest.raises(ValueError):
        tp.build_world(bad)


def test_any_part_in_view_detects_visible_and_hidden_parts():
    """벨트 위 보이는 볼트는 True, 카메라 뒤(바닥 먼 곳) 부품만 있으면 False다."""
    q_lying = (0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4))  # rpy (0, π/2, 0)
    assert tp.any_part_in_view(HOME_T, HOME_R, K, 1280, 720, [("BOLT_M3X20", (0.55, 0.10, 0.1025), q_lying)])
    assert not tp.any_part_in_view(HOME_T, HOME_R, K, 1280, 720, [("NUT_M6", (-1.5, 0.0, 0.05), (0, 0, 0, 1))])
    assert not tp.any_part_in_view(HOME_T, HOME_R, K, 1280, 720, [])


def test_clip_polygon_and_area():
    """사각형 다각형을 조각 경계로 자르면 넓이가 겹친 부분만 남고, 겹치지 않으면 빈 배열이다."""
    sq = np.array([[600, 100], [700, 100], [700, 150], [600, 150]], float)
    assert tp.polygon_area(sq) == pytest.approx(5000.0)
    cut = tp.clip_polygon_rect(sq, 0, 0, 640, 640)
    assert tp.polygon_area(cut) == pytest.approx(2000.0)
    assert tp.clip_polygon_rect(sq, 800, 0, 900, 640).size == 0


def test_tile_origins_and_bounds():
    """1280×720 기본 조각은 가로 3개(0·320·640), 세로 가운데(40)이고, 영상 밖 조각은 ValueError다."""
    assert tp.tile_origins(1280, 720) == [(0, 40), (320, 40), (640, 40)]
    with pytest.raises(ValueError):
        tp.tile_origins(1280, 720, xs=(700,))


def test_tile_labels_keep_cut_and_reject_ambiguous():
    """조각에 충분히 남은 부품은 잘린 다각형(조각 좌표)으로 두고, 조금만 걸친 부품이 있으면 조각을 버린다(None)."""
    inside = (0, np.array([[100, 100], [140, 100], [140, 110], [100, 110]], float))
    half = (0, np.array([[620, 100], [660, 100], [660, 110], [620, 110]], float))   # 절반이 조각 안
    sliver = (0, np.array([[636, 300], [700, 300], [700, 310], [636, 310]], float))  # 4/64만 조각 안
    out = tp.tile_labels([inside, half], 0, 40)
    assert len(out) == 2 and out[0][1][0].tolist() == [100.0, 60.0]
    assert tp.polygon_area(out[1][1]) == pytest.approx(200.0)
    assert tp.tile_labels([inside, sliver], 0, 40) is None
    assert tp.tile_labels([inside], 640, 40) == []  # 조각 밖 부품은 무시
