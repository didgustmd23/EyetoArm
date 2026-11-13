"""scripts/coco_to_yolo_seg.py 단위 시험 (A 상세설계 §7.1 AI 파일럿 학습 자료 변환)."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "coco_to_yolo_seg", Path(__file__).resolve().parents[2] / "scripts" / "coco_to_yolo_seg.py")
cy = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = cy
_SPEC.loader.exec_module(cy)


def _write_raw(root: Path):
    """train·valid·test 폴더에 영상 하나씩과 COCO 주석을 쓴다. 상위 분류·다중 다각형·주석 없는 영상을 포함한다."""
    cats = [{"id": 0, "name": "bolt-nut"}, {"id": 1, "name": "Bolt"}, {"id": 2, "name": "Nut"}]
    spec = {
        "train": [("a.jpg", 200, 100, [(1, [[20, 10, 60, 10, 60, 50]]),
                                       (2, [[100, 50, 150, 50, 150, 100, 100, 100]])])],
        "valid": [("b.jpg", 100, 100, [(0, [[0, 0, 10, 0, 10, 10]]),
                                       (2, [[0, 0, 10, 0, 10, 10], [20, 20, 80, 20, 80, 80, 20, 80]])])],
        "test": [("c.jpg", 100, 100, [])],
    }

    for split, items in spec.items():
        d = root / split
        d.mkdir(parents=True)
        coco = {"categories": cats, "images": [], "annotations": []}

        for name, w, h, anns in items:
            # 변환은 영상을 디코딩하지 않고 링크만 만들므로 영상 대신 구분되는 바이트를 쓴다.
            (d / name).write_bytes(b"jpegdata-" + name.encode())
            iid = len(coco["images"])
            coco["images"].append({"id": iid, "file_name": name, "width": w, "height": h})

            for cid, seg in anns:
                coco["annotations"].append({"id": len(coco["annotations"]), "image_id": iid,
                                            "category_id": cid, "segmentation": seg})

        (d / cy.ANNOTATION_FILE).write_text(json.dumps(coco), encoding="utf-8")


def _snapshot(root):
    """폴더 안 모든 파일의 (sha256, 수정 시각)을 경로별로 모은다."""
    return {str(p): (hashlib.sha256(p.read_bytes()).hexdigest(), os.stat(p).st_mtime_ns)
            for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture
def setup(tmp_path):
    """합성 원본과, 원래 valid·test 영상을 서로 바꾼 재분할 목록을 만들어 (원본, 목록 폴더, 출력 폴더)를 돌려주는 fixture."""
    raw = tmp_path / "raw"
    _write_raw(raw)
    lists = tmp_path / "lists"
    lists.mkdir()

    # 재분할 목록: 원래 valid 영상을 test로, 원래 test 영상을 val로 옮긴다
    (lists / "train.txt").write_text("train/a.jpg\n", encoding="utf-8")
    (lists / "valid.txt").write_text("test/c.jpg\n", encoding="utf-8")
    (lists / "test.txt").write_text("valid/b.jpg\n", encoding="utf-8")
    return raw, lists, tmp_path / "out"


def test_class_mapping_and_normalization(setup):
    """Bolt는 0, Nut는 1로 바뀌고, 다각형 좌표는 영상 폭·높이로 나눈 0~1 값이다."""
    raw, lists, out = setup
    cy.convert(raw, lists, out)

    lines = (out / "labels/train/a.txt").read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2

    bolt = [float(v) for v in lines[0].split()]
    nut = [float(v) for v in lines[1].split()]

    assert bolt[0] == 0 and nut[0] == 1  # Bolt→0, Nut→1

    # 폭 200·높이 100으로 정규화
    assert bolt[1:] == pytest.approx([0.1, 0.1, 0.3, 0.1, 0.3, 0.5])
    assert nut[1:] == pytest.approx([0.5, 0.5, 0.75, 0.5, 0.75, 1.0, 0.5, 1.0])


def test_split_lists_are_followed_and_images_are_symlinks(setup):
    """원래 폴더가 아니라 재분할 목록대로 나뉘고, 영상은 원본을 가리키는 상대 심볼릭 링크다. 주석 없는 영상은 빈 라벨 파일을 받는다."""
    raw, lists, out = setup
    stats = cy.convert(raw, lists, out)

    assert sorted(os.listdir(out / "images/val")) == ["c.jpg"]
    assert sorted(os.listdir(out / "images/test")) == ["b.jpg"]

    link = out / "images/test/b.jpg"

    assert link.is_symlink() and not os.path.isabs(os.readlink(link))
    assert link.resolve() == (raw / "valid/b.jpg").resolve()
    assert (out / "labels/val/c.txt").read_text(encoding="utf-8") == ""  # 주석 없는 영상은 빈 라벨
    assert stats["splits"]["val"]["images"] == 1


def test_superclass_ignored_and_multi_polygon_keeps_largest(setup):
    """상위 분류(bolt-nut) 주석은 세고 버리며, 다각형이 여러 개인 주석은 가장 큰 것만 남긴다."""
    raw, lists, out = setup
    stats = cy.convert(raw, lists, out)

    assert stats["ignored_categories"] == {"bolt-nut": 1}
    assert stats["multi_polygon"] == 1 and stats["dropped_polygons"] == 1

    (line,) = (out / "labels/test/b.txt").read_text(encoding="utf-8").splitlines()
    vals = [float(v) for v in line.split()]

    assert vals[0] == 1 and vals[1:] == pytest.approx([0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8])


def test_dataset_yaml_has_no_absolute_path(setup):
    """`dataset.yaml`에 절대 경로·`path:`가 없고 상대 분할 경로와 두 클래스 이름이 있다."""
    raw, lists, out = setup
    cy.convert(raw, lists, out)

    text = (out / "dataset.yaml").read_text(encoding="utf-8")

    assert "path:" not in text and str(out) not in text
    assert "train: images/train" in text and "0: bolt" in text and "1: nut" in text


def test_raw_unchanged_and_rerun_identical(setup):
    """변환은 원본을 바꾸지 않고, 다시 실행해도 출력 내용이 같다."""
    raw, lists, out = setup
    before = _snapshot(raw)
    cy.convert(raw, lists, out)
    first = {p: v[0] for p, v in _snapshot(out).items()}

    cy.convert(raw, lists, out)

    assert _snapshot(raw) == before
    assert {p: v[0] for p, v in _snapshot(out).items()} == first


def test_unknown_list_entry_raises(setup):
    """목록에 원본 주석에 없는 영상이 있으면 KeyError다."""
    raw, lists, out = setup
    (lists / "train.txt").write_text("train/missing.jpg\n", encoding="utf-8")

    with pytest.raises(KeyError):
        cy.convert(raw, lists, out)
