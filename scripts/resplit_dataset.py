#!/usr/bin/env python3
"""공개 데이터(Bolt nut segmentation v1) 장면 묶음 단위 재분할.

A 상세설계 §12 데이터 분리 행(A 제안 방식, 사용자 확정 2026-11-12)을 구현한다.

- 원본(`data/raw/{train,valid,test}`)은 읽기만 한다. 출력은 분할 목록 파일과 manifest뿐이다.
- 묶음 1단계: 같은 원본 줄기(파일명 `.rf.` 앞)의 사본은 한 묶음.
- 묶음 2단계: 줄기 간 유사 영상 결합(회색조 32×32 정규화 상관 ≥ 임계값). `--dihedral`이면
  뒤집기·90° 회전 8가지 중 최댓값을 쓴다(Roboflow 증강 사본이 뒤집기·회전을 포함하므로).
  `--adj-k > 0`이면 같은 파일명 계열에서 번호 순위 차가 k 이하이고 상관이 `--adj-min-sim`
  이상인 줄기도 결합한다(보조 결합 조건).
- 배정: 묶음 단위, 목표 비율은 줄기 수 기준, Bolt·Nut 주석 수 비율과 클래스별 크기 구간(주석 마스크
  최소 외접 사각형 짧은 변, 기본 <15 / 15~25 / ≥25 px) 비율이 분할 간 비슷하도록 탐욕 배정(1.1, v2).
  `--w-size 0`이면 1.0(v1)과 같은 배정이다.
- valid·test는 줄기당 1장(파일명 정렬 첫 사본), train은 사본 모두.
- 같은 인자·같은 원본이면 출력은 바이트 단위로 같다(시각·경로 같은 실행 환경 값은 기록하지 않음).

의존성: 표준 라이브러리, NumPy, OpenCV(4.6에서 확인). 네트워크를 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

SCRIPT_VERSION = "resplit_dataset 1.1"
SOURCE_SPLITS = ("train", "valid", "test")
OUT_SPLITS = ("train", "valid", "test")
ANNOTATION_FILE = "_annotations.coco.json"
CLASS_NAMES = ("Bolt", "Nut")
FEATURE_SIZE = 32

_HASH_RE = re.compile(r"_[0-9a-f]{32}")
_EXT_SUFFIX_RE = re.compile(r"(_(jpe?g|png|bmp|webp))+$", re.IGNORECASE)
_DIGITS_RE = re.compile(r"\d+")


@dataclass
class ImageRecord:
    """원본 영상 한 장의 주석 요약(클래스별 개수·짧은 변)."""

    rel_path: str  # data/raw 기준 상대 경로 (예: "train/xxx.jpg")
    stem: str
    counts: dict  # {"Bolt": n, "Nut": n}
    short_px: dict = field(default_factory=dict)  # {"Bolt": [짧은 변 px, ...], "Nut": [...]}


@dataclass
class StemRecord:
    """같은 원본 줄기에서 나온 사본들."""

    stem: str
    images: list = field(default_factory=list)  # ImageRecord, rel_path 정렬

    @property
    def representative(self) -> ImageRecord:
        """valid·test에 넣는 대표 사본(rel_path 정렬 첫 사본)."""
        return self.images[0]


# ---------------------------------------------------------------- 파일명 규칙
def stem_of(file_name: str) -> str:
    """Roboflow 내보내기 파일명 `<원본>_<확장자>_jpg.rf.<해시>.jpg`의 원본 줄기."""
    name = Path(file_name).name
    if ".rf." in name:
        return name.split(".rf.", 1)[0]
    return Path(name).stem


def family_key(stem: str) -> tuple[str, tuple[int, ...]]:
    """파일명 계열(숫자를 #로 바꾼 무늬)과 번호 튜플. 확장자 꼬리와 32자리 해시는 뺀다."""
    base = _EXT_SUFFIX_RE.sub("", stem)
    base = _HASH_RE.sub("", base)
    numbers = tuple(int(d) for d in _DIGITS_RE.findall(base))
    pattern = _DIGITS_RE.sub("#", base)
    return pattern, numbers


# ---------------------------------------------------------------- 자료 읽기
def sha256_file(path: Path) -> str:
    """파일 내용의 sha256 16진 문자열."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_records(raw_root: Path) -> tuple[list[ImageRecord], dict]:
    """세 원본 분할의 COCO 주석을 읽어 (영상 기록 목록(rel_path 정렬), 주석 파일별 sha256)을 돌려준다.

    주석·영상 파일이 없으면 FileNotFoundError.
    """
    records: list[ImageRecord] = []
    ann_sha: dict = {}
    for split in SOURCE_SPLITS:
        ann_path = raw_root / split / ANNOTATION_FILE
        if not ann_path.is_file():
            raise FileNotFoundError(f"주석 파일 없음: {ann_path}")
        ann_sha[f"{split}/{ANNOTATION_FILE}"] = sha256_file(ann_path)
        with open(ann_path, encoding="utf-8") as f:
            coco = json.load(f)
        cat_name = {c["id"]: c["name"] for c in coco["categories"]}
        per_image: dict = {im["id"]: {n: 0 for n in CLASS_NAMES} for im in coco["images"]}
        per_short: dict = {im["id"]: {n: [] for n in CLASS_NAMES} for im in coco["images"]}
        for ann in coco["annotations"]:
            name = cat_name.get(ann["category_id"])
            if name in CLASS_NAMES:
                per_image[ann["image_id"]][name] += 1
                s = annotation_short_px(ann)
                if s is not None:
                    per_short[ann["image_id"]][name].append(s)
        for im in coco["images"]:
            rel = f"{split}/{im['file_name']}"
            if not (raw_root / rel).is_file():
                raise FileNotFoundError(f"영상 파일 없음: {raw_root / rel}")
            records.append(ImageRecord(rel, stem_of(im["file_name"]), per_image[im["id"]], per_short[im["id"]]))
    records.sort(key=lambda r: r.rel_path)
    return records, ann_sha


def annotation_short_px(ann: dict):
    """COCO 다각형 분할 주석의 최소 외접 사각형 짧은 변(px). 다각형이 없으면 None(RLE 등)."""
    seg = ann.get("segmentation")
    if not isinstance(seg, list):
        return None
    polys = [np.asarray(p, np.float32).reshape(-1, 2) for p in seg if len(p) >= 6]
    if not polys:
        return None
    (_, _), (w, h), _ = cv2.minAreaRect(np.concatenate(polys))
    return float(min(w, h))


def size_bin_counts(short_px: dict, edges) -> list:
    """클래스별 크기 구간 개수를 [Bolt 구간..., Nut 구간...] 순서로."""
    out = []
    for name in CLASS_NAMES:
        idx = np.digitize(np.asarray(short_px.get(name, []), float), edges)
        out.extend(int((idx == b).sum()) for b in range(len(edges) + 1))
    return out


def size_bin_labels(edges) -> list:
    """`size_bin_counts`와 같은 순서의 구간 이름(예: "Bolt:<15")."""
    names = [f"<{edges[0]:g}"] + [f"{a:g}-{b:g}" for a, b in zip(edges[:-1], edges[1:])] + [f">={edges[-1]:g}"]
    return [f"{c}:{n}" for c in CLASS_NAMES for n in names]


def group_stems(records: list[ImageRecord]) -> list[StemRecord]:
    """영상 기록을 줄기별로 모은다(줄기 이름 정렬, 줄기 안은 rel_path 정렬)."""
    by_stem: dict = {}
    for r in records:
        by_stem.setdefault(r.stem, StemRecord(r.stem)).images.append(r)
    stems = [by_stem[k] for k in sorted(by_stem)]
    for s in stems:
        s.images.sort(key=lambda r: r.rel_path)
    return stems


# ---------------------------------------------------------------- 영상 특징
def thumbnail(img_bgr: np.ndarray) -> np.ndarray:
    """회색조 32×32(INTER_AREA) float64."""
    gray = img_bgr if img_bgr.ndim == 2 else cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (FEATURE_SIZE, FEATURE_SIZE), interpolation=cv2.INTER_AREA)
    return small.astype(np.float64)


def normalize(small: np.ndarray) -> np.ndarray:
    """평균 0·노름 1 벡터. 균일 영상은 0 벡터."""
    v = np.ascontiguousarray(small).ravel().copy()
    v -= v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros_like(v)


def image_feature(img_bgr: np.ndarray) -> np.ndarray:
    """회색조 32×32 정규화 특징(1024)."""
    return normalize(thumbnail(img_bgr))


def dihedral_features(img_bgr: np.ndarray) -> np.ndarray:
    """뒤집기·90° 회전 8가지 특징(8×1024). 0번이 원본.

    32×32로 줄인 뒤 변형한다(정사각형 원본 640×640에서는 변형 후 축소와 같다).
    """
    small = thumbnail(img_bgr)
    out = []
    for k in range(4):
        rot = np.rot90(small, k)
        out.append(normalize(rot))
        out.append(normalize(rot[:, ::-1]))
    return np.stack(out)


def load_features(raw_root: Path, records: list[ImageRecord], dihedral: bool) -> np.ndarray:
    """(N, T, 1024). T = 8(dihedral) 또는 1."""
    feats = []
    for r in records:
        img = cv2.imread(str(raw_root / r.rel_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"영상 해석 실패: {r.rel_path}")
        feats.append(dihedral_features(img) if dihedral else image_feature(img)[None, :])
    return np.stack(feats)


def max_similarity(feats_a: np.ndarray, feats_b: np.ndarray) -> np.ndarray:
    """A의 원본 특징과 B의 모든 변형 특징의 상관 최댓값 (Na, Nb)."""
    base = feats_a[:, 0, :]
    best = np.full((feats_a.shape[0], feats_b.shape[0]), -np.inf)
    for t in range(feats_b.shape[1]):
        np.maximum(best, base @ feats_b[:, t, :].T, out=best)
    return best


# ---------------------------------------------------------------- 묶음
class UnionFind:
    """줄기 색인 묶음용 합집합-찾기. 대표는 묶음의 가장 작은 색인이다."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        """`a`가 속한 묶음의 대표 색인(경로 절반 압축)."""
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        """두 묶음을 합친다. 작은 대표를 남긴다."""
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def build_groups(stems, records, feats, threshold, adj_k, adj_min_sim, block=512):
    """줄기 색인 묶음 목록(각 묶음은 정렬된 줄기 색인 목록, 묶음은 첫 색인 순)."""
    stem_index = {s.stem: i for i, s in enumerate(stems)}
    img_stem = np.array([stem_index[r.stem] for r in records])
    n_img = len(records)
    uf = UnionFind(len(stems))
    # 줄기 쌍 최대 상관(보조 결합용). 줄기 수가 작아 dense로 둔다.
    stem_sim = np.full((len(stems), len(stems)), -np.inf) if adj_k > 0 else None
    # records는 줄기 색인 순으로 정렬돼 있어야 한다(열 방향 reduceat).
    if np.any(np.diff(img_stem) < 0):
        raise ValueError("records must be ordered by stem index")
    starts = np.flatnonzero(np.r_[True, np.diff(img_stem) != 0])
    for a0 in range(0, n_img, block):
        sim = max_similarity(feats[a0:a0 + block], feats)
        rows, cols = np.nonzero(sim >= threshold)
        for i, j in zip(rows + a0, cols):
            if img_stem[i] != img_stem[j]:
                uf.union(int(img_stem[i]), int(img_stem[j]))
        if stem_sim is not None:
            red = np.maximum.reduceat(sim, starts, axis=1)  # (block, 줄기 수)
            np.maximum.at(stem_sim, img_stem[a0:a0 + sim.shape[0]], red)
    if adj_k > 0:
        fam: dict = {}
        for i, s in enumerate(stems):
            pattern, numbers = family_key(s.stem)
            fam.setdefault(pattern, []).append((numbers, s.stem, i))
        for members in fam.values():
            members.sort()
            for a in range(len(members)):
                for b in range(a + 1, min(a + adj_k, len(members) - 1) + 1):
                    i, j = members[a][2], members[b][2]
                    if max(stem_sim[i, j], stem_sim[j, i]) >= adj_min_sim:
                        uf.union(i, j)
    groups: dict = {}
    for i in range(len(stems)):
        groups.setdefault(uf.find(i), []).append(i)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


# ---------------------------------------------------------------- 배정
def assign_groups(groups, stems, ratios, seed, big_frac=0.25, size_edges=(15.0, 25.0),
                  w_stems=4.0, w_class=1.0, w_size=1.0):
    """묶음마다 분할 이름을 정해 묶음 순서대로 목록으로 돌려준다(탐욕 배정).

    비용은 줄기 수·Bolt·Nut·클래스별 크기 구간(대표 사본 기준)의 분할 몫 편차 제곱합에 가중치를
    곱한 값이며, 비용이 가장 작은 분할을 고른다.

    가장 작은 분할 목표 줄기 수의 `big_frac`보다 큰 묶음은 먼저(큰 순) 배정하고, 나머지는 시드로 섞은
    순서로 배정한다. 크기 순으로만 배정하면 valid·test가 한 장짜리 묶음으로만 채워진다.
    """
    ratios = np.asarray(ratios, dtype=float)
    ratios = ratios / ratios.sum()
    rng = np.random.RandomState(seed)
    order = list(range(len(groups)))
    tie = rng.permutation(len(groups))
    vec = []
    for g in groups:
        reps = [stems[i].representative.counts for i in g]
        bins = np.sum([size_bin_counts(stems[i].representative.short_px, size_edges) for i in g], axis=0)
        head = [len(g), sum(c["Bolt"] for c in reps), sum(c["Nut"] for c in reps)]
        vec.append(np.concatenate([head, bins]).astype(float))
    total = np.sum(vec, axis=0)
    total[total == 0] = 1.0
    big_limit = big_frac * float(ratios.min()) * float(total[0])
    big = sorted((k for k in order if vec[k][0] > big_limit), key=lambda k: (-vec[k][0], tie[k]))
    rest = sorted((k for k in order if vec[k][0] <= big_limit), key=lambda k: tie[k])
    order = big + rest
    acc = np.zeros((len(OUT_SPLITS), len(vec[0])))
    assign = [None] * len(groups)
    for k in order:
        best, best_cost = None, None
        for s in range(len(OUT_SPLITS)):
            trial = acc.copy()
            trial[s] += vec[k]
            frac = trial / total  # 각 분할이 차지하는 몫
            # 줄기 수 편차를 가장 무겁게, 주석 비율 편차는 보조
            dev = frac - ratios[:, None]
            cost = (w_stems * np.sum(dev[:, 0] ** 2) + w_class * np.sum(dev[:, 1:3] ** 2)
                    + w_size * np.sum(dev[:, 3:] ** 2))
            if best_cost is None or cost < best_cost - 1e-15:
                best, best_cost = s, cost
        acc[best] += vec[k]
        assign[k] = OUT_SPLITS[best]
    return assign


# ---------------------------------------------------------------- 실행
def resplit(raw_root: Path, out_dir: Path, *, threshold=0.95, dihedral=True, adj_k=3, adj_min_sim=0.85,
            ratios=(0.8, 0.1, 0.1), seed=20261112, big_frac=0.25, size_edges=(15.0, 25.0),
            w_stems=4.0, w_class=1.0, w_size=1.0, write=True) -> dict:
    """원본을 읽어 묶음·배정을 계산하고, `write`면 `out_dir`에 분할 목록 3개와 manifest.json을 쓴다.

    돌려주는 dict: manifest, lists(분할별 rel_path), groups, stems, records(줄기 순), feats.
    """
    raw_root = Path(raw_root)
    records, ann_sha = load_records(raw_root)
    stems = group_stems(records)
    stem_index = {s.stem: i for i, s in enumerate(stems)}
    rec_order = sorted(records, key=lambda r: (stem_index[r.stem], r.rel_path))
    feats = load_features(raw_root, rec_order, dihedral)
    groups = build_groups(stems, rec_order, feats, threshold, adj_k, adj_min_sim)
    size_edges = tuple(float(e) for e in size_edges)
    assign = assign_groups(groups, stems, ratios, seed, big_frac, size_edges, w_stems, w_class, w_size)

    lists = {s: [] for s in OUT_SPLITS}
    stats = {s: {"images": 0, "stems": 0, "groups": 0, "Bolt": 0, "Nut": 0} for s in OUT_SPLITS}
    shorts = {s: {n: [] for n in CLASS_NAMES} for s in OUT_SPLITS}
    for g, split in zip(groups, assign):
        stats[split]["groups"] += 1
        for i in g:
            st = stems[i]
            chosen = st.images if split == "train" else [st.representative]
            stats[split]["stems"] += 1
            for r in chosen:
                lists[split].append(r.rel_path)
                stats[split]["images"] += 1
                stats[split]["Bolt"] += r.counts["Bolt"]
                stats[split]["Nut"] += r.counts["Nut"]
                for n in CLASS_NAMES:
                    shorts[split][n].extend(r.short_px.get(n, []))
    for s in OUT_SPLITS:
        lists[s].sort()
        stats[s]["size_bins"] = dict(zip(size_bin_labels(size_edges), size_bin_counts(shorts[s], size_edges)))
        stats[s]["short_px_pct"] = {
            n: ({f"p{q}": round(float(np.percentile(shorts[s][n], q)), 1) for q in (5, 10, 50)}
                if shorts[s][n] else None) for n in CLASS_NAMES}
    sizes = sorted((len(g) for g in groups), reverse=True)
    manifest = {
        "script_version": SCRIPT_VERSION,
        "source": "data/raw (Bolt nut segmentation v1, Roboflow, CC BY 4.0) — 원본 무수정, 목록만",
        "args": {"threshold": threshold, "dihedral": dihedral, "adj_k": adj_k, "adj_min_sim": adj_min_sim,
                 "ratios_stems": list(ratios), "seed": seed, "big_frac": big_frac,
                 "size_edges_px": list(size_edges), "w_stems": w_stems, "w_class": w_class, "w_size": w_size,
                 "feature": "gray 32x32 INTER_AREA, zero-mean unit-norm"},
        "rules": {
            "stem": "파일명 '.rf.' 앞",
            "valid_test_one_per_stem": "rel_path 정렬 첫 사본",
            "train": "줄기의 사본 모두",
            "size": "주석 다각형 minAreaRect 짧은 변(640x640 기준 px), 배정 균형은 대표 사본 기준",
        },
        "source_counts": {"images": len(records), "stems": len(stems)},
        "groups": {"count": len(groups), "max_size_stems": sizes[0] if sizes else 0,
                   "top10_sizes": sizes[:10], "singletons": sum(1 for x in sizes if x == 1)},
        "splits": stats,
        "annotation_sha256": ann_sha,
        "notes": [
            "Roboflow 전처리: 640x640 Stretch 리사이즈(원본 종횡비 손실). 화소 크기별 검출률 해석 시 원본 크기와 다름",
            "동봉 README.roboflow.txt는 'No image augmentation'이라 적었으나 train 줄기당 최대 3사본(뒤집기·회전·잡음 등)이 있어 증강 사본으로 추정",
            "형식 변환(YOLO 라벨)은 하지 않음",
        ],
    }
    if write:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for s in OUT_SPLITS:
            (out_dir / f"{s}.txt").write_text("".join(p + "\n" for p in lists[s]), encoding="utf-8")
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"manifest": manifest, "lists": lists, "groups": groups, "stems": stems,
            "records": rec_order, "feats": feats}


def cross_split_similarity(raw_root: Path, lists: dict, dihedral: bool) -> dict:
    """분할 간 최대 상관과 ≥0.98·≥0.95 쌍 수(검증용)."""
    out = {}
    feats = {s: load_features(Path(raw_root), [ImageRecord(p, "", {}) for p in lists[s]], dihedral)
             for s in OUT_SPLITS}
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        sim = np.maximum(max_similarity(feats[a], feats[b]), max_similarity(feats[b], feats[a]).T)
        out[f"{a}-{b}"] = {"max": float(sim.max()) if sim.size else None,
                           ">=0.98": int((sim >= 0.98).sum()), ">=0.95": int((sim >= 0.95).sum())}
    return out


def main(argv=None) -> int:
    """CLI 진입점. 통계(와 `--verify`면 분할 간 상관)를 stdout에 JSON으로 쓰고 0을 돌려준다."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--out", default="data/splits/resplit_v2")
    p.add_argument("--threshold", type=float, default=0.95)
    p.add_argument("--no-dihedral", dest="dihedral", action="store_false")
    p.add_argument("--adj-k", type=int, default=3, help="같은 파일명 계열 번호 순위 차 k 이하 줄기 결합(0이면 끔)")
    p.add_argument("--adj-min-sim", type=float, default=0.85, help="보조 결합에 필요한 줄기 간 최대 상관")
    p.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    p.add_argument("--seed", type=int, default=20261112)
    p.add_argument("--big-frac", type=float, default=0.25,
                   help="가장 작은 분할 목표 줄기 수의 이 비율보다 큰 묶음을 먼저 배정")
    p.add_argument("--size-edges", type=float, nargs="+", default=(15.0, 25.0), help="크기 구간 경계(px)")
    p.add_argument("--w-stems", type=float, default=4.0)
    p.add_argument("--w-class", type=float, default=1.0)
    p.add_argument("--w-size", type=float, default=1.0, help="크기 구간 균형 가중치(0이면 v1 배정)")
    p.add_argument("--dry-run", action="store_true", help="목록을 쓰지 않고 통계만 출력")
    p.add_argument("--verify", action="store_true", help="분할 간 상관 검사 출력")
    a = p.parse_args(argv)
    res = resplit(Path(a.raw), Path(a.out), threshold=a.threshold, dihedral=a.dihedral, adj_k=a.adj_k,
                  adj_min_sim=a.adj_min_sim, ratios=tuple(a.ratios), seed=a.seed, big_frac=a.big_frac,
                  size_edges=tuple(a.size_edges), w_stems=a.w_stems, w_class=a.w_class, w_size=a.w_size,
                  write=not a.dry_run)
    m = res["manifest"]
    print(json.dumps({"groups": m["groups"], "splits": m["splits"]}, ensure_ascii=False))
    if a.verify:
        print(json.dumps({"cross_plain": cross_split_similarity(Path(a.raw), res["lists"], False),
                          "cross_dihedral": cross_split_similarity(Path(a.raw), res["lists"], True)},
                         ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
