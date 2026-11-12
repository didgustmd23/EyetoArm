"""scripts/resplit_dataset.py 단위 시험 (지시서 _workspace/46 §1-2, A 상세설계 §12 데이터 분리 행).

합성 영상과 작은 COCO 주석으로 정답을 아는 입력을 만든다.
"""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "resplit_dataset", Path(__file__).resolve().parents[2] / "scripts" / "resplit_dataset.py")
rd = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = rd  # dataclass가 모듈을 찾을 수 있게
_SPEC.loader.exec_module(rd)

SIZE = 96


def _pattern(seed):
    rng = np.random.RandomState(seed)
    small = rng.randint(0, 256, (8, 8), dtype=np.uint8)
    img = cv2.resize(small, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _name(stem, h):
    return f"{stem}_jpg.rf.{h:032x}.jpg"


def _make_raw(root: Path):
    """줄기 20개. 정답:
    - scene_a(train 원본 폴더, 사본 3장: 원본·좌우 뒤집기·90° 회전)
    - scene_b(valid 폴더)는 scene_a와 거의 같은 영상(잡음 조금) → 같은 묶음이어야 함
    - scene_c(test 폴더)는 scene_a의 좌우 뒤집기 + 잡음 → dihedral이면 같은 묶음
    - 나머지 17개는 서로 다른 무늬
    """
    layout = {"train": [], "valid": [], "test": []}
    base_a = _pattern(1)
    rng = np.random.RandomState(99)

    def noisy(img):
        n = rng.randint(-3, 4, img.shape)
        return np.clip(img.astype(int) + n, 0, 255).astype(np.uint8)

    layout["train"].append(("scene_a", [base_a, base_a[:, ::-1], np.rot90(base_a).copy()]))
    layout["valid"].append(("scene_b", [noisy(base_a)]))
    layout["test"].append(("scene_c", [noisy(np.ascontiguousarray(base_a[:, ::-1]))]))
    for k in range(17):
        folder = ("train", "train", "valid", "test")[k % 4]
        img = _pattern(100 + k)
        copies = [img, img[::-1, :].copy()] if folder == "train" else [img]
        layout[folder].append((f"img{k:03d}", copies))
    h = 1
    for folder, items in layout.items():
        d = root / folder
        d.mkdir(parents=True)
        coco = {"categories": [{"id": 0, "name": "bolt-nut"}, {"id": 1, "name": "Bolt"}, {"id": 2, "name": "Nut"}],
                "images": [], "annotations": []}
        aid = 0
        for stem, imgs in items:
            for img in imgs:
                fn = _name(stem, h)
                h += 1
                cv2.imwrite(str(d / fn), np.ascontiguousarray(img))
                iid = len(coco["images"])
                coco["images"].append({"id": iid, "file_name": fn, "width": SIZE, "height": SIZE})
                for cat in (1, 2) if stem.endswith(("0", "2", "4", "6", "8")) else (1,):
                    coco["annotations"].append({"id": aid, "image_id": iid, "category_id": cat})
                    aid += 1
        (d / rd.ANNOTATION_FILE).write_text(json.dumps(coco), encoding="utf-8")
    return layout


def _snapshot(root: Path):
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = (hashlib.sha256(p.read_bytes()).hexdigest(), os.stat(p).st_mtime_ns)
    return out


@pytest.fixture
def raw(tmp_path):
    root = tmp_path / "raw"
    _make_raw(root)
    return root


def _run(raw, out, **kw):
    kw.setdefault("ratios", (0.6, 0.2, 0.2))
    return rd.resplit(raw, out, **kw)


def _split_of(lists):
    m = {}
    for s, paths in lists.items():
        for p in paths:
            m.setdefault(rd.stem_of(p), set()).add(s)
    return m


def test_stem_and_family_rules():
    assert rd.stem_of("051_png_jpg.rf.43a3decdae13a68bec0369fd9fdf2932.jpg") == "051_png_jpg"
    assert rd.family_key("051_png_jpg") == ("#", (51,))
    assert rd.family_key("M6_frame_33_jpg") == ("M#_frame_#", (6, 33))
    assert rd.family_key("z4773562275197_881e074d102aea120bfb01fdcf0fc21e_jpg") == ("z#", (4773562275197,))


def test_same_stem_copies_in_one_split(raw, tmp_path):
    res = _run(raw, tmp_path / "out")
    for stem, splits in _split_of(res["lists"]).items():
        assert len(splits) == 1, stem
    # train에 배정된 줄기는 사본 모두
    train = res["lists"]["train"]
    for st in res["stems"]:
        if rd.stem_of(st.images[0].rel_path) in {rd.stem_of(p) for p in train}:
            assert all(r.rel_path in train for r in st.images)


def test_similar_images_grouped_together(raw, tmp_path):
    res = _run(raw, tmp_path / "out", dihedral=True)
    where = _split_of(res["lists"])
    assert where["scene_a_jpg"] == where["scene_b_jpg"] == where["scene_c_jpg"]
    stem_names = [s.stem for s in res["stems"]]
    group_of = {stem_names[i]: gi for gi, g in enumerate(res["groups"]) for i in g}
    assert group_of["scene_a_jpg"] == group_of["scene_b_jpg"] == group_of["scene_c_jpg"]
    # 뒤집기를 고려하지 않으면 scene_c는 scene_a 원본과 직접 닮지 않았지만 scene_a의 뒤집힌 사본과 닮아 묶인다
    res_plain = _run(raw, tmp_path / "out_plain", dihedral=False)
    names = [s.stem for s in res_plain["stems"]]
    g2 = {names[i]: gi for gi, g in enumerate(res_plain["groups"]) for i in g}
    assert g2["scene_a_jpg"] == g2["scene_b_jpg"]


def test_valid_test_one_image_per_stem(raw, tmp_path):
    res = _run(raw, tmp_path / "out")
    for s in ("valid", "test"):
        stems = [rd.stem_of(p) for p in res["lists"][s]]
        assert len(stems) == len(set(stems))
        assert stems  # 비어 있지 않음
    m = res["manifest"]["splits"]
    assert m["valid"]["images"] == m["valid"]["stems"]
    assert m["test"]["images"] == m["test"]["stems"]


def test_same_seed_same_output(raw, tmp_path):
    _run(raw, tmp_path / "o1", seed=7)
    _run(raw, tmp_path / "o2", seed=7)
    for f in ("train.txt", "valid.txt", "test.txt", "manifest.json"):
        assert (tmp_path / "o1" / f).read_bytes() == (tmp_path / "o2" / f).read_bytes()


def test_raw_files_unchanged(raw, tmp_path):
    before = _snapshot(raw)
    _run(raw, tmp_path / "out")
    assert _snapshot(raw) == before


def test_no_cross_split_pair_above_threshold(raw, tmp_path):
    thr = 0.95
    for dihedral in (True, False):
        res = _run(raw, tmp_path / f"out_{dihedral}", threshold=thr, dihedral=dihedral)
        cross = rd.cross_split_similarity(raw, res["lists"], dihedral)
        for pair, v in cross.items():
            assert v[">=0.95"] == 0, (dihedral, pair, v)


def test_manifest_records_sources(raw, tmp_path):
    res = _run(raw, tmp_path / "out")
    m = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert set(m["annotation_sha256"]) == {f"{s}/{rd.ANNOTATION_FILE}" for s in rd.SOURCE_SPLITS}
    assert m["args"]["seed"] == res["manifest"]["args"]["seed"]
    assert any("Stretch" in n for n in m["notes"])
    lines = (tmp_path / "out" / "train.txt").read_text(encoding="utf-8").splitlines()
    assert lines == sorted(lines)
    assert all((raw / p).is_file() for p in lines)


def test_missing_annotation_raises(tmp_path):
    (tmp_path / "raw" / "train").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        rd.resplit(tmp_path / "raw", tmp_path / "out")


def test_adjacency_join_needs_family_rank_and_similarity():
    # 줄기 3개(같은 계열 s#): s001–s002 상관 0.9(임계 0.95 미만), s001–s010 상관 0
    rng = np.random.RandomState(3)
    base = rng.randn(1024)
    other = rng.randn(1024)
    other -= other @ base / (base @ base) * base  # base와 직교
    def unit(v):
        v = v - v.mean()
        return v / np.linalg.norm(v)
    b, o = unit(base), unit(other)
    near = unit(0.9 * b + np.sqrt(1 - 0.81) * o)
    stems = [rd.StemRecord(s, [rd.ImageRecord(f"train/{s}.jpg", s, {"Bolt": 0, "Nut": 0})])
             for s in ("s001", "s002", "s010")]
    records = [st.images[0] for st in stems]
    feats = np.stack([b, near, o])[:, None, :]
    assert 0.85 < float(b @ near) < 0.95
    no_adj = rd.build_groups(stems, records, feats, 0.95, adj_k=0, adj_min_sim=0.85)
    assert no_adj == [[0], [1], [2]]
    with_adj = rd.build_groups(stems, records, feats, 0.95, adj_k=3, adj_min_sim=0.85)
    assert with_adj == [[0, 1], [2]]
    strict = rd.build_groups(stems, records, feats, 0.95, adj_k=3, adj_min_sim=0.95)
    assert strict == [[0], [1], [2]]


def test_annotation_short_px_rotated_rectangle():
    # 30×10 px 직사각형을 30° 돌린 다각형 → 최소 외접 사각형 짧은 변 10 px
    c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
    pts = [(-15, -5), (15, -5), (15, 5), (-15, 5)]
    poly = [v for x, y in pts for v in (100 + c * x - s * y, 100 + s * x + c * y)]
    assert rd.annotation_short_px({"segmentation": [poly]}) == pytest.approx(10.0, abs=0.05)
    assert rd.annotation_short_px({"segmentation": {"counts": "rle"}}) is None


def _size_stems(n, small_idx):
    stems = []
    for i in range(n):
        short = [8.0] if i in small_idx else [40.0]
        rec = rd.ImageRecord(f"train/s{i:03d}.jpg", f"s{i:03d}", {"Bolt": 0, "Nut": 1}, {"Bolt": [], "Nut": short})
        stems.append(rd.StemRecord(rec.stem, [rec]))
    return stems


@pytest.mark.parametrize("seed", range(5))
def test_size_balance_puts_small_objects_into_eval(seed):
    # 줄기 20개(각 한 묶음, 너트 1개). 작은 너트(8 px) 4개. 목표 60/20/20.
    stems = _size_stems(20, {0, 1, 2, 3})
    groups = [[i] for i in range(20)]
    small = {0, 1, 2, 3}
    a = rd.assign_groups(groups, stems, (0.6, 0.2, 0.2), seed, size_edges=(15.0, 25.0), w_size=10.0)
    per = {s: sum(1 for g, sp in zip(groups, a) if sp == s and g[0] in small) for s in rd.OUT_SPLITS}
    # 목표 60/20/20에 가장 가까운 2/1/1 (w_size=0이면 시드 1~3에서 valid 또는 test가 0개가 됨)
    assert per == {"train": 2, "valid": 1, "test": 1}, per
    # 줄기 비율은 그대로 유지
    assert [a.count(s) for s in rd.OUT_SPLITS] == [12, 4, 4]


def test_size_balance_zero_weight_matches_v1_cost():
    # w_size=0이면 크기 정보가 배정에 영향을 주지 않는다
    stems_a = _size_stems(20, {0, 1, 2, 3})
    stems_b = _size_stems(20, set())
    groups = [[i] for i in range(20)]
    assert rd.assign_groups(groups, stems_a, (0.6, 0.2, 0.2), 3, w_size=0.0) == \
        rd.assign_groups(groups, stems_b, (0.6, 0.2, 0.2), 3, w_size=0.0)
