#!/usr/bin/env bash
# S1-01·S1-07 conf 재평가 스윕(#140). 기존 평가 CLI(scripts/eval_size_recall.py)를 조건마다 한 번씩 순차로 부른다.
#
# - 조건: 모델 2(S1-01, S1-07) × conf 7 × 데이터 2(Gazebo gz_eval_v1/test imgsz 1280, 공개 bolt_nut_v2/test imgsz 640) = 28.
#   고정: device 0, scale 1.0, 마스크 매칭 IoU 0.5, 크기 구간 10 12 15 25(공식 평가 s1_eval.json과 같음), --map 생략.
#   predict는 eval_size_recall.evaluate()가 공식 평가와 같은 인자(imgsz·conf·device·retina_masks=True)로 부르고,
#   NMS iou·max_det 등은 ultralytics 기본값 그대로다(이 스크립트가 바꾸지 않는다).
# - 출력: outputs/conf_sweep/s101_s107_<접미사>/ 를 새로 만든다(이미 있으면 거부). 조건마다 <조건>.json(결과)·<조건>.log(stderr).
# - 추론 전에 manifest.json에 가중치 sha256·평가 입력 지문·도구 sha256·조건 목록·git 상태를 기록한다.
#   가중치 sha256이 기대값과 다르면 추론하지 않고 끝낸다. 끝나면 가중치를 다시 해시해 바뀌지 않았는지 확인한다.
# - 한 조건이라도 실패하면 즉시 중단한다(set -e). 병렬 실행 없음.
#
# 사용(저장소 루트):
#   scripts/conf_sweep_s101_s107.sh [접미사]        실제 실행(GPU 추론). 접미사 생략 시 날짜_시각_난수
#   scripts/conf_sweep_s101_s107.sh --dry-run       추론 없이 경로·해시 확인, manifest 미리보기와 28개 명령만 출력
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
TOOL="scripts/eval_size_recall.py"
CONFS=(0.084 0.10 0.15 0.25 0.35 0.50 0.70)

# 모델: 이름|가중치 경로|기대 sha256(평가 기록 eval.json의 weights.sha256과 같다)
MODELS=(
  "S1-01|outputs/s1_candidates/S1-01_yolov8n-seg/weights/best.pt|68ed4d1139d4d19fa009e20b037e052924574d6429c61f20ec7bdd81dc26f2c6"
  "S1-07|outputs/training/s1_clean_r1/S1-07_yolov8n-seg/weights/best.pt|f921ae2dc9d2bc70b24672b3c892716d5c1a6e5562cbdf14cceea5f4f8ee4766"
)

# 데이터: 이름|데이터 폴더|분할|imgsz
DATASETS=(
  "gz1280|data/derived/gz_eval_v1|test|1280"
  "pub640|data/derived/bolt_nut_v2|test|640"
)

DRY=0
SUFFIX=""
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY=1
elif [[ -n "${1:-}" ]]; then
  SUFFIX="$1"
fi
if [[ -n "$SUFFIX" && ! "$SUFFIX" =~ ^[A-Za-z0-9_.-]{1,40}$ ]]; then
  echo "접미사는 영문·숫자·_.- 1~40자: $SUFFIX" >&2
  exit 2
fi
[[ -z "$SUFFIX" ]] && SUFFIX="$(date +%Y%m%d_%H%M%S)_$(od -An -N3 -tx1 /dev/urandom | tr -d ' \n')"

OUT="outputs/conf_sweep/s101_s107_${SUFFIX}"

# 1) 사전 확인: 인터프리터·도구·가중치·데이터
[[ -x "$PY" ]] || { echo "인터프리터 없음: $PY" >&2; exit 2; }
[[ -f "$TOOL" ]] || { echo "도구 없음: $TOOL" >&2; exit 2; }

for m in "${MODELS[@]}"; do
  IFS='|' read -r name w sha <<< "$m"
  [[ -f "$w" ]] || { echo "가중치 없음: $w" >&2; exit 2; }
  got="$(sha256sum "$w" | cut -d' ' -f1)"
  [[ "$got" == "$sha" ]] || { echo "가중치 sha256 불일치 $name: $got != $sha" >&2; exit 2; }
done

for d in "${DATASETS[@]}"; do
  IFS='|' read -r dname dir split imgsz <<< "$d"
  [[ -d "$dir/images/$split" && -d "$dir/labels/$split" ]] || { echo "데이터 없음: $dir ($split)" >&2; exit 2; }
done

# 2) 조건 목록(28)과 명령
CMDS=()
NAMES=()
for m in "${MODELS[@]}"; do
  IFS='|' read -r name w sha <<< "$m"
  for c in "${CONFS[@]}"; do
    for d in "${DATASETS[@]}"; do
      IFS='|' read -r dname dir split imgsz <<< "$d"
      cond="${name}_${dname}_conf${c}"
      NAMES+=("$cond")
      CMDS+=("$PY $TOOL --weights $w --data $dir --split $split --imgsz $imgsz --conf $c --iou 0.5 --scale 1.0 --edges 10 12 15 25 --device 0 --out $OUT/$cond.json")
    done
  done
done

# 3) manifest(추론 전): 가중치·입력 지문·도구·조건·환경
write_manifest() {
  # $1: 출력 경로
  MODELS_S="$(printf '%s\n' "${MODELS[@]}")" DATASETS_S="$(printf '%s\n' "${DATASETS[@]}")" \
  NAMES_S="$(printf '%s\n' "${NAMES[@]}")" CMDS_S="$(printf '%s\n' "${CMDS[@]}")" CONFS_S="${CONFS[*]}" \
  "$PY" - "$1" "$OUT" <<'PYEOF'
"""manifest 작성(모델 로드·추론 없음). 평가 입력 지문은 fingerprint 모듈과 같은 방식(이름 + 파일 sha256 누적)."""
import hashlib, json, os, subprocess, sys, importlib.metadata as md

def sha(p):
    """파일 sha256(링크는 가리키는 내용)."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def split_fp(d, split):
    """분할의 영상·라벨 지문과 수."""
    fp = {}
    for kind in ("images", "labels"):
        folder = os.path.join(d, kind, split)
        names = sorted(os.listdir(folder))
        h = hashlib.sha256()
        for n in names:
            h.update(n.encode()); h.update(sha(os.path.join(folder, n)).encode())
        fp[kind] = {"count": len(names), "sha256": h.hexdigest()}
    return fp

def git(*a):
    """git 출력(실패하면 None)."""
    try:
        return subprocess.run(["git", *a], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return None

def ver(p):
    """패키지 버전(import 없이)."""
    try:
        return md.version(p)
    except md.PackageNotFoundError:
        return None

models = [dict(zip(("name", "path", "sha256"), l.split("|"))) for l in os.environ["MODELS_S"].split("\n") if l]
datasets = [dict(zip(("name", "dir", "split", "imgsz"), l.split("|"))) for l in os.environ["DATASETS_S"].split("\n") if l]
for d in datasets:
    d["imgsz"] = int(d["imgsz"])
    d["fingerprint"] = split_fp(d["dir"], d["split"])
    yaml = os.path.join(d["dir"], "dataset.yaml")
    d["dataset_yaml_sha256"] = sha(yaml) if os.path.exists(yaml) else None
for m in models:
    m["sha256_checked"] = sha(m["path"]) == m["sha256"]

doc = {
    "purpose": "#140 S1-01·S1-07 conf 재평가(분석용). 미보정 공개 test·반복 사용한 Gazebo test라 미사용 시험·일반화 보장이 아니며 conf 자동 채택 근거가 아니다.",
    "out_dir": sys.argv[2],
    "tool": {"path": "scripts/eval_size_recall.py", "sha256": sha("scripts/eval_size_recall.py"),
             "runner": "scripts/conf_sweep_s101_s107.sh", "runner_sha256": sha("scripts/conf_sweep_s101_s107.sh")},
    "fixed": {"device": "0", "scale": 1.0, "match_mask_iou": 0.5, "edges_px": [10, 12, 15, 25], "map": False,
              "predict": "evaluate(): model.predict(img, imgsz, conf, device, retina_masks=True); NMS iou·max_det는 ultralytics 기본값",
              "official_eval_ref": "config/training/s1_eval.json (conf 0.25, iou 0.5, edges 10 12 15 25)"},
    "confs": [float(c) for c in os.environ["CONFS_S"].split()],
    "models": models, "datasets": datasets,
    "conditions": [{"name": n, "cmd": c} for n, c in zip(os.environ["NAMES_S"].split("\n"), os.environ["CMDS_S"].split("\n")) if n],
    "env": {"python": sys.version.split()[0], "ultralytics": ver("ultralytics"), "torch": ver("torch"),
            "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain"))},
    "baseline_conf025_gz1280": {"S1-01": {"bolt": {"tp": 266, "n": 444, "fp": 5}, "nut": {"tp": 177, "n": 219, "fp": 1}},
                                "S1-07": {"bolt": {"tp": 366, "n": 444, "fp": 77}, "nut": {"tp": 212, "n": 219, "fp": 1}},
                                "source": "outputs/training/s1_legacy/S1-01_yolov8n-seg/eval/eval.json, outputs/training/s1_clean_r1/S1-07_yolov8n-seg/eval/eval.json"},
}
doc["n_conditions"] = len(doc["conditions"])
assert doc["n_conditions"] == 28, doc["n_conditions"]
assert all(m["sha256_checked"] for m in models)
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(doc, f, ensure_ascii=False, indent=1)
    f.write("\n")
print(f"manifest ok: {sys.argv[1]} ({doc['n_conditions']} conditions)")
PYEOF
}

if [[ "$DRY" == 1 ]]; then
  TMP="$(mktemp -d)"
  write_manifest "$TMP/manifest.json"
  echo "dry-run: 추론 없음. manifest 미리보기 $TMP/manifest.json, 실제 실행 시 출력 폴더 예: $OUT"
  printf '%s\n' "${CMDS[@]}" | sed "s#$OUT#<OUT>#"
  exit 0
fi

# 4) 실제 실행: 새 폴더(있으면 거부) → manifest → 조건 순차 실행 → 가중치 재확인 → 요약
mkdir -p outputs/conf_sweep
mkdir "$OUT" || { echo "출력 폴더가 이미 있음(덮어쓰지 않음): $OUT" >&2; exit 2; }
write_manifest "$OUT/manifest.json"

for i in "${!CMDS[@]}"; do
  cond="${NAMES[$i]}"
  [[ -e "$OUT/$cond.json" ]] && { echo "이미 있음(덮어쓰지 않음): $OUT/$cond.json" >&2; exit 2; }
  echo "[$((i + 1))/${#CMDS[@]}] $cond"
  # stdout(결과 JSON 사본)은 버리고 결과는 --out 파일, 경고·오류는 .log
  ${CMDS[$i]} > /dev/null 2> "$OUT/$cond.log"
done

for m in "${MODELS[@]}"; do
  IFS='|' read -r name w sha <<< "$m"
  got="$(sha256sum "$w" | cut -d' ' -f1)"
  [[ "$got" == "$sha" ]] || { echo "실행 중 가중치가 바뀜 $name: $got" >&2; exit 2; }
done

# 5) 요약(TP/FP/FN·precision/recall, 추론 없음)
"$PY" - "$OUT" <<'PYEOF'
"""조건별 JSON을 읽어 summary.tsv를 쓴다."""
import json, os, sys
out = sys.argv[1]
man = json.load(open(os.path.join(out, "manifest.json"), encoding="utf-8"))
rows = ["model\tdata\tconf\tclass\tgt_n\ttp\tfn\tpred_n\tfp\tprecision\trecall"]
for c in man["conditions"]:
    r = json.load(open(os.path.join(out, c["name"] + ".json"), encoding="utf-8"))
    model, data, conf = c["name"].split("_", 2)
    for cls in ("bolt", "nut"):
        t, p = r["total"][cls], r["predictions"][cls]
        prec = "" if p["precision"] is None else f"{p['precision']:.4f}"
        rec = f"{t['tp'] / t['n']:.4f}" if t["n"] else ""
        rows.append(f"{model}\t{data}\t{r['conf']}\t{cls}\t{t['n']}\t{t['tp']}\t{t['n'] - t['tp']}\t{p['n']}\t{p['fp']}\t{prec}\t{rec}")
open(os.path.join(out, "summary.tsv"), "w", encoding="utf-8").write("\n".join(rows) + "\n")
print(f"summary: {os.path.join(out, 'summary.tsv')} ({len(rows) - 1} rows)")
PYEOF

( cd "$OUT" && sha256sum *.json *.log summary.tsv > SHA256SUMS )
echo "완료: $OUT"
