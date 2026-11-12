#!/usr/bin/env python3
"""COCO 분할 주석 + 재분할 목록 → ultralytics YOLO 분할 형식(파일럿용 2클래스).

- 입력: `data/raw/{train,valid,test}/_annotations.coco.json`, 분할 목록 `data/splits/resplit_v2/{train,valid,test}.txt`
- 출력: `data/derived/bolt_nut_v2/{images,labels}/{train,val,test}/`, `dataset.yaml`
  - 영상은 심볼릭 링크(복사·수정하지 않음). 원본 파일명 그대로.
  - 라벨 한 줄: `<클래스> x1 y1 x2 y2 ...` (영상 폭·높이로 나눈 [0,1] 좌표)
- 클래스(파일럿용 2클래스, 클래스 구성은 B와 미결): `Bolt`→0 `bolt`, `Nut`→1 `nut`.
  범주 `bolt-nut`(상위 범주)은 무시하고 건수만 보고한다.
- 다중 다각형 주석: 면적이 가장 큰 다각형 하나만 쓰고 나머지 개수를 보고한다.
- `dataset.yaml`에는 `path`를 쓰지 않는다(절대 경로 금지). ultralytics 8.4는 `path`가 없으면 yaml 파일이
  있는 폴더를 데이터 루트로 쓰고, `path: .`은 실행 위치(CWD) 기준으로 풀리므로 쓰지 않는다.

의존성: 표준 라이브러리, NumPy. 같은 입력이면 같은 출력.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

SOURCE_SPLITS = ("train", "valid", "test")
LIST_TO_OUT = {"train": "train", "valid": "val", "test": "test"}
CLASS_MAP = {"Bolt": 0, "Nut": 1}
CLASS_NAMES = ["bolt", "nut"]
ANNOTATION_FILE = "_annotations.coco.json"


def polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def annotation_to_line(ann: dict, width: int, height: int, cls: int, stats: dict):
    seg = ann.get("segmentation")
    if not isinstance(seg, list):
        stats["skipped_non_polygon"] += 1
        return None
    polys = [np.asarray(p, float).reshape(-1, 2) for p in seg if len(p) >= 6]
    if not polys:
        stats["skipped_empty"] += 1
        return None
    if len(polys) > 1:
        stats["multi_polygon"] += 1
        stats["dropped_polygons"] += len(polys) - 1
    poly = max(polys, key=polygon_area)
    norm = poly / np.array([width, height], float)
    norm = np.clip(norm, 0.0, 1.0)
    return f"{cls} " + " ".join(f"{v:.6f}" for v in norm.ravel())


def load_coco(raw_root: Path) -> dict:
    """rel_path → (width, height, [(class_name, ann), ...])."""
    out = {}
    for split in SOURCE_SPLITS:
        with open(raw_root / split / ANNOTATION_FILE, encoding="utf-8") as f:
            coco = json.load(f)
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        images = {im["id"]: im for im in coco["images"]}
        per = {i: [] for i in images}
        for ann in coco["annotations"]:
            per[ann["image_id"]].append((cats.get(ann["category_id"]), ann))
        for iid, im in images.items():
            out[f"{split}/{im['file_name']}"] = (im["width"], im["height"], per[iid])
    return out


def convert(raw_root: Path, list_dir: Path, out_root: Path) -> dict:
    raw_root, list_dir, out_root = Path(raw_root), Path(list_dir), Path(out_root)
    coco = load_coco(raw_root)
    stats = {"skipped_non_polygon": 0, "skipped_empty": 0, "multi_polygon": 0, "dropped_polygons": 0,
             "ignored_categories": {}, "splits": {}}
    for list_name, out_split in LIST_TO_OUT.items():
        paths = [ln.strip() for ln in (list_dir / f"{list_name}.txt").read_text(encoding="utf-8").splitlines()
                 if ln.strip()]
        img_dir = out_root / "images" / out_split
        lbl_dir = out_root / "labels" / out_split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        counts = {"images": 0, "bolt": 0, "nut": 0}
        names_seen = set()
        for rel in sorted(paths):
            if rel not in coco:
                raise KeyError(f"주석에 없는 목록 항목: {rel}")
            name = Path(rel).name
            if name in names_seen:
                raise ValueError(f"같은 분할에 같은 파일명 중복: {name}")
            names_seen.add(name)
            width, height, anns = coco[rel]
            lines = []
            for cname, ann in anns:
                if cname not in CLASS_MAP:
                    stats["ignored_categories"][cname] = stats["ignored_categories"].get(cname, 0) + 1
                    continue
                line = annotation_to_line(ann, width, height, CLASS_MAP[cname], stats)
                if line is not None:
                    lines.append(line)
                    counts[CLASS_NAMES[CLASS_MAP[cname]]] += 1
            link = img_dir / name
            target = os.path.relpath((raw_root / rel).resolve(), img_dir.resolve())
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target)
            (lbl_dir / (Path(name).stem + ".txt")).write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")
            counts["images"] += 1
        stats["splits"][out_split] = counts
    yaml = ("# 파일럿용 2클래스(클래스 구성은 B와 미결). coco_to_yolo_seg.py가 생성\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)))
    (out_root / "dataset.yaml").write_text(yaml, encoding="utf-8")
    (out_root / "convert_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--lists", default="data/splits/resplit_v2")
    p.add_argument("--out", default="data/derived/bolt_nut_v2")
    a = p.parse_args(argv)
    print(json.dumps(convert(Path(a.raw), Path(a.lists), Path(a.out)), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
