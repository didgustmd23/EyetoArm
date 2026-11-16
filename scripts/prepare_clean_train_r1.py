#!/usr/bin/env python3
"""검수 train 후보(#137 r2) + 기존 v2 val/test로 학습용 데이터 구성(clean_train_r1)을 만든다.

- train: #137 r2 최종 목록(final_usable_list.csv) 1420장만 연결한다. 영상은 raw 원본, 라벨은 bolt_nut_v2 train 라벨에 링크한다.
- val/test: bolt_nut_v2의 val/test 영상·라벨을 그대로 링크한다(재분할·라벨 수정 없음, 기존 미보정 평가 분할).
- 분할 목록(data/splits/clean_train_r1/)은 resplit_v2와 같은 형식(data/raw 기준 상대 경로)이다. valid.txt·test.txt는
  resplit_v2 파일과 내용이 같아야 한다.
- 원본(raw)·v2·조사 근거 파일은 읽기만 한다. 출력 폴더가 이미 있으면 덮어쓰지 않고 종료 2로 끝난다.
- 줄기('.rf.' 앞)·파일 sha256이 분할을 가로지르면 재분할하지 않고 manifest와 종료 코드 3으로 보고한다.

사용(저장소 루트): python3 scripts/prepare_clean_train_r1.py [--check-only]
종료 코드: 0 생성 완료(또는 --check-only 검사 통과), 2 입력·출력 오류, 3 분할 누수 발견(생성하지 않음).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CAND_CSV = "_workspace/evidence/2026-11-16-nonwin-clean-inventory/recheck_session7_evidence/final_usable_list.csv"
LEDGER_CSV = "_workspace/evidence/2026-11-16-nonwin-clean-inventory/recheck_ledger.csv"
ANCHOR_CSV = "_workspace/evidence/2026-11-16-nonwin-clean-inventory/interim_judgments.csv"
V2 = "data/derived/bolt_nut_v2"
RAW = "data/raw"
SRC_SPLITS = "data/splits/resplit_v2"
OUT_DATA = "data/derived/bolt_nut_clean_train_r1"
OUT_SPLITS = "data/splits/clean_train_r1"
EXPECTED_ROWS = 1420

# v2 분할 폴더 이름 ↔ 분할 목록 파일 이름
SPLITS = (("train", "train.txt"), ("val", "valid.txt"), ("test", "test.txt"))


def sha256_file(path: str) -> str:
    """파일 sha256(심볼릭 링크는 가리키는 내용)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stem_of(name: str) -> str:
    """Roboflow 줄기(파일명 '.rf.' 앞)."""
    return name.split(".rf.")[0]


def p(rel: str) -> str:
    """저장소 기준 절대 경로."""
    return os.path.join(ROOT, rel)


def raw_rel(v2_split: str, name: str) -> str:
    """v2 영상 링크가 가리키는 data/raw 기준 상대 경로(예: train/xxx.jpg)."""
    target = os.path.realpath(p(f"{V2}/images/{v2_split}/{name}"))
    rel = os.path.relpath(target, p(RAW))
    if rel.startswith(".."):
        raise ValueError(f"v2 image is not under data/raw: {name}")
    return rel


def excluded_images() -> dict:
    """사용 목록에 들어가면 안 되는 영상: 원장 후보 중 통과 아닌 행(강등·보류), 비후보, 비-desk 앵커 중 사용가능 아닌 것."""
    out = {}

    for r in csv.DictReader(open(p(LEDGER_CSV), encoding="utf-8")):
        if r["subagent_verdict"] != "사용가능후보":
            out[r["image"]] = f"ledger {r['case']}: 하위 {r['subagent_verdict']}"
        elif r["builder_recheck"] != "통과":
            out[r["image"]] = f"ledger {r['case']}: builder {r['builder_recheck']}/{r['builder_verdict']}"

    seen = set()
    for r in csv.DictReader(open(p(ANCHOR_CSV), encoding="utf-8")):
        if r["case"] in seen:
            continue
        seen.add(r["case"])
        if r["verdict"] != "사용가능 후보":
            out[r["image"]] = f"anchor {r['case']}: {r['verdict']}"

    return out


def collect() -> dict:
    """입력을 읽고 검증한다. 오류는 ValueError."""
    rows = list(csv.DictReader(open(p(CAND_CSV), encoding="utf-8")))

    # 1) 후보 목록 자체
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"candidate rows {len(rows)} != {EXPECTED_ROWS}")
    if len({r["image"] for r in rows}) != len(rows) or len({r["sha256"] for r in rows}) != len(rows):
        raise ValueError("duplicate image name or sha256 in candidate list")

    # 2) 제외 대상이 섞이지 않았는가
    excl = excluded_images()
    mixed = sorted(r["image"] for r in rows if r["image"] in excl)
    if mixed:
        raise ValueError(f"excluded images in candidate list: {mixed[:5]}")

    # 3) 후보마다 v2 train 영상·라벨, raw 원본, sha256 일치
    train = []
    for r in rows:
        name = r["image"]
        lab = os.path.splitext(name)[0] + ".txt"
        if not os.path.exists(p(f"{V2}/images/train/{name}")) or not os.path.isfile(p(f"{V2}/labels/train/{lab}")):
            raise ValueError(f"missing in v2 train: {name}")
        rel = raw_rel("train", name)
        if sha256_file(p(f"{RAW}/{rel}")) != r["sha256"]:
            raise ValueError(f"sha256 mismatch: {name}")
        train.append({"name": name, "label": lab, "raw_rel": rel, "sha256": r["sha256"], "source": r["source"]})

    # 4) val/test는 v2 폴더 그대로, 목록은 resplit_v2와 같아야 한다
    evals = {}
    for split, lst in SPLITS[1:]:
        names = sorted(os.listdir(p(f"{V2}/images/{split}")))
        items = [{"name": n, "label": os.path.splitext(n)[0] + ".txt", "raw_rel": raw_rel(split, n),
                  "sha256": sha256_file(p(f"{V2}/images/{split}/{n}"))} for n in names]
        src_lines = sorted(open(p(f"{SRC_SPLITS}/{lst}"), encoding="utf-8").read().split())
        if sorted(i["raw_rel"] for i in items) != src_lines:
            raise ValueError(f"v2 {split} folder differs from {SRC_SPLITS}/{lst}")
        evals[split] = items

    return {"train": sorted(train, key=lambda i: i["raw_rel"]), **evals, "excluded": excl}


def leakage(data: dict) -> dict:
    """줄기·sha256이 분할을 가로지르는지."""
    stems = {s: {stem_of(i["name"]) for i in data[s]} for s, _ in SPLITS}
    shas = {s: {i["sha256"] for i in data[s]} for s, _ in SPLITS}
    out = {}

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        out[f"{a}-{b}"] = {"stems": sorted(stems[a] & stems[b]), "sha256": sorted(shas[a] & shas[b])}

    # 같은 분할 안의 sha256 중복(사본이 같은 바이트인 경우)
    for s, _ in SPLITS:
        c = {}
        for i in data[s]:
            c.setdefault(i["sha256"], []).append(i["name"])
        out[f"within-{s}"] = {"dup_sha256": {k: v for k, v in c.items() if len(v) > 1}}

    return out


def has_leak(lk: dict) -> bool:
    """분할 사이 누수 또는 분할 안 sha 중복이 있는가."""
    cross = any(v["stems"] or v["sha256"] for k, v in lk.items() if not k.startswith("within"))
    within = any(v["dup_sha256"] for k, v in lk.items() if k.startswith("within"))
    return cross or within


def build(data: dict) -> None:
    """링크·목록·dataset.yaml을 만든다(출력 폴더가 없을 때만)."""
    for split, lst in SPLITS:
        img_dir, lab_dir = p(f"{OUT_DATA}/images/{split}"), p(f"{OUT_DATA}/labels/{split}")
        os.makedirs(img_dir)
        os.makedirs(lab_dir)

        for i in data[split]:
            # 영상은 raw 원본, 라벨은 v2 라벨(같은 분할)로 상대 링크
            os.symlink(os.path.relpath(p(f"{RAW}/{i['raw_rel']}"), img_dir), os.path.join(img_dir, i["name"]))
            os.symlink(os.path.relpath(p(f"{V2}/labels/{split}/{i['label']}"), lab_dir),
                       os.path.join(lab_dir, i["label"]))

        # train은 정렬 목록, val/test는 resplit_v2 파일을 바이트 그대로 복사(줄 순서까지 동일)
        os.makedirs(p(OUT_SPLITS), exist_ok=True)
        if split == "train":
            text = "".join(i["raw_rel"] + "\n" for i in data[split]).encode("utf-8")
        else:
            with open(p(f"{SRC_SPLITS}/{lst}"), "rb") as f:
                text = f.read()
        with open(p(f"{OUT_SPLITS}/{lst}"), "wb") as f:
            f.write(text)

    with open(p(f"{OUT_DATA}/dataset.yaml"), "w", encoding="utf-8") as f:
        f.write("# clean_train_r1: train=#137 r2 검수 후보 1420장, val/test=bolt_nut_v2 기존 분할(미보정).\n"
                "# scripts/prepare_clean_train_r1.py가 생성. 정제된 전체 v3가 아니다.\n"
                "train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: bolt\n  1: nut\n")


def manifest(data: dict, lk: dict) -> dict:
    """출처·해시·제외·불변 조건."""
    inst = {}
    for split, _ in SPLITS:
        c = {"bolt": 0, "nut": 0}
        for i in data[split]:
            for line in open(p(f"{V2}/labels/{split}/{i['label']}"), encoding="utf-8"):
                if line.strip():
                    c["bolt" if line.split()[0] == "0" else "nut"] += 1
        inst[split] = c

    excl_ids = sorted(v for k, v in data["excluded"].items() if "보류" in v or "강등" in v or "anchor" in v)

    return {
        "name": "clean_train_r1",
        "script": "scripts/prepare_clean_train_r1.py",
        "description": "train = #137 r2 builder 검수 후보(사용자 미승인), val/test = bolt_nut_v2 기존 분할·라벨(미보정). "
                       "정제된 전체 v3가 아니다. Gazebo 평가 데이터는 포함하지 않는다.",
        "inputs": {
            "candidate_csv": {"path": CAND_CSV, "sha256": sha256_file(p(CAND_CSV)), "rows": len(data["train"])},
            "ledger_csv": {"path": LEDGER_CSV, "sha256": sha256_file(p(LEDGER_CSV))},
            "anchor_csv": {"path": ANCHOR_CSV, "sha256": sha256_file(p(ANCHOR_CSV))},
            "source_splits": {f"{SRC_SPLITS}/{lst}": sha256_file(p(f"{SRC_SPLITS}/{lst}"))
                              for lst in ("train.txt", "valid.txt", "test.txt", "manifest.json")},
            "v2_dataset_yaml": {"path": f"{V2}/dataset.yaml", "sha256": sha256_file(p(f"{V2}/dataset.yaml"))},
        },
        "counts": {s: {"images": len(data[s]), "stems": len({stem_of(i["name"]) for i in data[s]}),
                       "instances": inst[s]} for s, _ in SPLITS},
        "train_by_source": {src: sum(i["source"] == src for i in data["train"])
                            for src in sorted({i["source"] for i in data["train"]})},
        "excluded": {"note": "원장 후보 중 강등·보류, 비후보(하위 수정필요·판단불가), 사용가능 아닌 사전 앵커는 train에 넣지 않았다. "
                             "desk 612장·나머지 v2 train(WIN 418 등)은 검토 범위 밖이라 넣지 않았다.",
                     "ledger_or_anchor_excluded_in_v2_train": len(data["excluded"]),
                     "held_or_demoted": excl_ids},
        "leakage": lk,
        "invariants": ["data/raw, data/derived/bolt_nut_v2, data/splits/resplit_v2, _workspace 조사 근거는 읽기만 함",
                       "val/test 목록은 resplit_v2와 내용 동일(재분할 없음)",
                       "라벨은 v2 라벨 링크(수정 없음), 영상은 raw 원본 링크",
                       "학습·추론 실행 없음"],
        "known_limits": ["val/test 라벨은 미보정(알려진 결함 포함) → best.pt 선택·평가 수치의 한계",
                         "train은 z-hash 단일 장면 편중(약 81%), 너트 140개, 소형·밝은 물체 조건 부족",
                         "builder 검수이며 사용자 승인 아님"],
    }


def main(argv=None) -> int:
    """진입점."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check-only", action="store_true", help="검사만 하고 아무것도 만들지 않는다")
    args = ap.parse_args(argv)

    if not args.check_only and (os.path.exists(p(OUT_DATA)) or os.path.exists(p(OUT_SPLITS))):
        print(json.dumps({"ok": False, "error": f"output exists: {OUT_DATA} or {OUT_SPLITS} (덮어쓰지 않음)"},
                         ensure_ascii=False))
        return 2

    try:
        data = collect()
    except (ValueError, OSError) as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 2

    lk = leakage(data)
    man = manifest(data, lk)

    if has_leak(lk):
        print(json.dumps({"ok": False, "error": "split leakage found (재분할하지 않음)", "leakage": lk},
                         ensure_ascii=False, indent=1))
        return 3

    if not args.check_only:
        build(data)
        with open(p(f"{OUT_SPLITS}/manifest.json"), "w", encoding="utf-8") as f:
            json.dump(man, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")

    print(json.dumps({"ok": True, "check_only": args.check_only, "counts": man["counts"],
                      "train_by_source": man["train_by_source"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
