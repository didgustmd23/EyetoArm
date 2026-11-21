"""AI 후보 평가 순수 로직: 평가 설정, 시간 측정·정확도 절차, 기록 검사, 요약. torch·ultralytics 없음.

평가 설정(`eval_config_id`)은 학습 설정과 분리해 평가 조건만 바꿔도 학습 기록이 흔들리지 않게 한다.
시간 측정은 준비 → cold → 워밍업 → 본 측정 순서를 고정하고, 크기마다 새 자식·새 모델은 실행기가 보장한다.
정답(manifest·라벨)은 평가 경로에서만 쓴다.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from numbers import Real
from typing import Any, Callable

from eyetoarm.training.config import (ConfigError, _as_float, _exact_keys, _freeze, _is_int, _load_json, _rel_path,
                                      thaw)

PHASES = ("prepare", "cold", "warmup", "measure")
ACC_REQUIRED = ("split", "images", "imgsz", "conf", "iou", "by_size", "total", "predictions", "per_gt",
                "map50_box", "map50_mask")
FINE_CLASSES = ("BOLT_A", "BOLT_B", "NUT")
SUMMARY_COLUMNS = (   # 1단계 실험 기록 표의 열 순서
    "번호", "모델·가중치(출처·해시)", "학습 설정(imgsz·에포크·시드·증강 등)", "데이터 목록", "라이브러리 버전",
    "공개 test mAP50 박스/마스크·크기 구간 재현율", "Gazebo 재현율 볼트/너트·BOLT_A/BOLT_B/NUT(추론 640·1280)", "오검출",
    "추론 P95·GPU 최대·학습 시간", "해석·실패·중단")


class EvalError(ValueError):
    """평가 입력·기록 오류."""


# ---------------------------------------------------------------- 설정
@dataclass(frozen=True)
class EvalConfig:
    """검증된 평가 설정(고정값)."""

    conf: float
    iou: float
    device: str
    edges_px: tuple
    public: Any
    gazebo: Any
    timing: Any
    val: Any
    eval_config_id: str

    def accuracy_plan(self) -> list:
        """정확도 계획 [(키, 데이터 폴더, 분할, imgsz)]: 공개 test → Gazebo 크기 순."""
        out = [(f"test{s}", self.public["data_dir"], self.public["split"], s) for s in self.public["imgsz"]]
        out += [(f"gz{s}", self.gazebo["data_dir"], self.gazebo["split"], s) for s in self.gazebo["imgsz"]]
        return out


def _unit(v, name: str) -> float:
    """(0, 1] 실수. 거대 정수(float 변환 넘침)도 ConfigError."""
    f = _as_float(v)
    if f is None or not math.isfinite(f) or not 0 < f <= 1:
        raise ConfigError(f"{name} must be in (0, 1]: {v!r}")
    return f


def _sizes(v, name: str) -> tuple:
    """서로 다른 양의 정수 imgsz 목록(32의 배수)."""
    if not isinstance(v, list) or not v or len(set(map(str, v))) != len(v):
        raise ConfigError(f"{name} must be a non-empty list without duplicates: {v!r}")
    for s in v:
        if not _is_int(s) or s < 32 or s % 32:
            raise ConfigError(f"{name} entries must be integers >= 32 and multiples of 32: {s!r}")
    return tuple(int(s) for s in v)


def eval_config_from_dict(d: dict) -> EvalConfig:
    """평가 설정 검증."""
    _exact_keys(d, ("format_version", "conf", "iou", "device", "edges_px", "public", "gazebo", "timing", "val"),
                "eval")
    if d["format_version"] != 1:
        raise ConfigError(f"eval format_version must be 1: {d['format_version']!r}")
    if not isinstance(d["device"], str) or not d["device"]:
        raise ConfigError("eval device must be a non-empty string")
    edges = d["edges_px"]
    if (not isinstance(edges, list) or not edges or any(isinstance(e, bool) or not isinstance(e, Real) for e in edges)
            or list(edges) != sorted(set(edges)) or edges[0] <= 0):
        raise ConfigError(f"edges_px must be strictly increasing positive numbers: {edges!r}")

    pub = d["public"]
    _exact_keys(pub, ("data_dir", "split", "imgsz"), "public")
    gz = d["gazebo"]
    _exact_keys(gz, ("data_dir", "split", "imgsz", "manifest"), "gazebo")
    tm = d["timing"]
    _exact_keys(tm, ("imgsz", "warmup_n", "measure_n"), "timing")
    va = d["val"]
    _exact_keys(va, ("batch", "workers"), "val")
    for part in (pub, gz):
        if part["split"] not in ("test", "val"):
            raise ConfigError(f"eval split must be test or val: {part['split']!r}")
    for k, lo in (("warmup_n", 0), ("measure_n", 1)):
        if not _is_int(tm[k]) or tm[k] < lo:
            raise ConfigError(f"timing.{k} must be an integer >= {lo}: {tm[k]!r}")
    for k, lo in (("batch", 1), ("workers", 0)):
        if not _is_int(va[k]) or va[k] < lo:
            raise ConfigError(f"val.{k} must be an integer >= {lo}: {va[k]!r}")

    text = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return EvalConfig(
        conf=_unit(d["conf"], "conf"), iou=_unit(d["iou"], "iou"), device=d["device"],
        edges_px=tuple(float(e) for e in edges),
        public=_freeze({"data_dir": _rel_path(pub["data_dir"], "public.data_dir"), "split": pub["split"],
                        "imgsz": _sizes(pub["imgsz"], "public.imgsz")}),
        gazebo=_freeze({"data_dir": _rel_path(gz["data_dir"], "gazebo.data_dir"), "split": gz["split"],
                        "imgsz": _sizes(gz["imgsz"], "gazebo.imgsz"),
                        "manifest": _rel_path(gz["manifest"], "gazebo.manifest")}),
        timing=_freeze({"imgsz": _sizes(tm["imgsz"], "timing.imgsz"), "warmup_n": int(tm["warmup_n"]),
                        "measure_n": int(tm["measure_n"])}),
        val=_freeze({"batch": int(va["batch"]), "workers": int(va["workers"])}),
        eval_config_id=hashlib.sha256(text.encode()).hexdigest())


def load_eval_config(path) -> EvalConfig:
    """평가 설정 파일(엄격 JSON)."""
    return eval_config_from_dict(_load_json(path))


# ---------------------------------------------------------------- 통계
def percentile(values, q: float) -> float:
    """선형 보간 백분위(numpy 기본 방식과 같음)."""
    xs = sorted(float(v) for v in values)
    if not xs:
        raise EvalError("percentile of empty list")
    pos = (len(xs) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def timing_stats(values) -> dict:
    """중앙값·P95·P99·최대·평균(ms)."""
    xs = [float(v) for v in values]
    return {"n": len(xs), "median": percentile(xs, 50), "p95": percentile(xs, 95), "p99": percentile(xs, 99),
            "max": max(xs), "mean": sum(xs) / len(xs)}


# ---------------------------------------------------------------- 시간 측정 절차
def run_timing(*, prepare: Callable[[], Any], images: list, predict: Callable[[Any, Any], Any],
               sync: Callable[[], None], peak_reset: Callable[[], None], peak_read: Callable[[], int | None],
               clock_ns: Callable[[], int], warmup_n: int, measure_n: int,
               calls_seen: Callable[[], int] | None = None) -> dict:
    """준비 → cold → 워밍업 → 본 측정. 호출마다 predict 반환까지(call)와 sync 뒤까지(sync)를 따로 잰다.

    images는 미리 읽어 둔 영상(디코딩 제외), prepare는 predict를 부르지 않는 모델 생성이다.
    `calls_seen`은 실제 predict 호출 누계이며, 준비 끝 시점 값으로 cold 이전 호출을 검사한다(없으면 실패).
    """
    if not images:
        raise EvalError("timing needs at least one preloaded image")

    t0 = clock_ns()
    model = prepare()
    prepare_ms = (clock_ns() - t0) / 1e6
    order = ["prepare"]
    out = {"prepare_ms": prepare_ms, "fresh_model": True,
           "predict_calls_before_prepare_end": calls_seen() if calls_seen is not None else None}

    def one(img):
        """predict 1회(반환, 동기화) ms."""
        a = clock_ns()
        predict(model, img)
        b = clock_ns()
        sync()
        c = clock_ns()
        return (b - a) / 1e6, (c - a) / 1e6

    k = 0
    for phase, n in (("cold", 1), ("warmup", warmup_n), ("measure", measure_n)):
        peak_reset()
        calls = []
        for _ in range(n):
            calls.append(one(images[k % len(images)]))
            k += 1
        order.append(phase)
        out[phase] = {"n": n, "call_ms": [c for c, _ in calls], "sync_ms": [s for _, s in calls],
                      "peak_mem_bytes": peak_read()}

    out["phase_order"] = order
    out["predict_calls"] = k
    return out


def summarize_timing(raw: dict) -> dict:
    """원시 시간 기록 → 요약(cold 1회 값, 워밍업·본 측정 통계, 단계별 최대 메모리)."""
    m = raw["measure"]
    return {"imgsz": raw.get("imgsz"), "prepare_ms": raw["prepare_ms"],
            "cold_call_ms": raw["cold"]["call_ms"][0], "cold_sync_ms": raw["cold"]["sync_ms"][0],
            "warmup_n": raw["warmup"]["n"], "measure_n": m["n"],
            "call_ms": timing_stats(m["call_ms"]), "sync_ms": timing_stats(m["sync_ms"]),
            "peak_mem_bytes": {p: raw[p]["peak_mem_bytes"] for p in ("cold", "warmup", "measure")}}


def _finite(v) -> bool:
    """유한 실수. 거대 정수는 OverflowError 대신 False(유한 실수로 보지 않음)."""
    f = _as_float(v)
    return f is not None and math.isfinite(f)


def validate_timing(d, *, imgsz: int, warmup_n: int, measure_n: int) -> list[str]:
    """시간 자식 기록 검사. 빈 목록이면 유효."""
    if not isinstance(d, dict):
        return ["timing record is not an object"]
    r = []
    if d.get("imgsz") != imgsz:
        r.append(f"imgsz {d.get('imgsz')!r} != {imgsz}")
    if d.get("phase_order") != list(PHASES):
        r.append(f"phase_order {d.get('phase_order')!r} != {list(PHASES)}")
    if d.get("fresh_model") is not True:
        r.append("model was not created in this child (fresh_model is not true)")
    if d.get("predict_calls_before_prepare_end") != 0:
        r.append("predict called before cold call")
    if not _finite(d.get("prepare_ms")):
        r.append("prepare_ms missing")
    for phase, n in (("cold", 1), ("warmup", warmup_n), ("measure", measure_n)):
        p = d.get(phase)
        if not isinstance(p, dict):
            r.append(f"{phase} missing")
            continue
        if p.get("n") != n or not isinstance(p.get("call_ms"), list) or len(p["call_ms"]) != n \
                or not isinstance(p.get("sync_ms"), list) or len(p["sync_ms"]) != n:
            r.append(f"{phase}: expected {n} calls")
        elif not all(_finite(x) for x in p["call_ms"] + p["sync_ms"]):
            r.append(f"{phase}: non-finite durations")
        if "peak_mem_bytes" not in p:
            r.append(f"{phase}: peak_mem_bytes key missing")
    if d.get("predict_calls") != 1 + warmup_n + measure_n:
        r.append(f"predict_calls {d.get('predict_calls')!r} != {1 + warmup_n + measure_n}")
    return r


# ---------------------------------------------------------------- 정확도 절차
def run_accuracy(*, model, plan: list, evaluate_fn: Callable, val_fn: Callable) -> dict:
    """계획 순서대로 같은 모델에 evaluate_fn(model, data_dir, split, imgsz)·val_fn(…)을 부른다."""
    out = {}
    for key, data_dir, split, imgsz in plan:
        res = dict(evaluate_fn(model, data_dir, split, imgsz))
        res.update(val_fn(model, data_dir, split, imgsz, key))
        out[key] = res
    out["plan"] = [list(p) for p in plan]
    return out


def validate_accuracy(d, plan: list) -> list[str]:
    """정확도 자식 기록 검사."""
    if not isinstance(d, dict):
        return ["accuracy record is not an object"]
    r = []
    if d.get("plan") != [list(p) for p in plan]:
        r.append("accuracy plan differs from the configured plan")
    for key, _, split, imgsz in plan:
        a = d.get(key)
        if not isinstance(a, dict):
            r.append(f"{key} missing")
            continue
        missing = [k for k in ACC_REQUIRED if k not in a]
        if missing:
            r.append(f"{key}: missing {missing}")
            continue
        if a["imgsz"] != imgsz or a["split"] != split:
            r.append(f"{key}: imgsz/split {a['imgsz']}/{a['split']} != {imgsz}/{split}")
        if not _is_int(a["images"]) or a["images"] <= 0 or len(a["per_gt"]) != a["images"]:
            r.append(f"{key}: images {a['images']!r} vs per_gt {len(a['per_gt'])}")
        if not (_finite(a["map50_box"]) and _finite(a["map50_mask"])):
            r.append(f"{key}: mAP50 not finite")
        for cls in ("bolt", "nut"):
            t = a["total"].get(cls) if isinstance(a["total"], dict) else None
            if not _count_ok(t):
                r.append(f"{key}: total.{cls} invalid")
            p = a["predictions"].get(cls) if isinstance(a["predictions"], dict) else None
            if not _count_ok(p) or not _is_int(p.get("fp")) or p["fp"] != p["n"] - p["tp"]:
                r.append(f"{key}: predictions.{cls} invalid")
            bins = a["by_size"].get(cls) if isinstance(a["by_size"], dict) else None
            if not isinstance(bins, dict) or not bins or not all(_count_ok(b) for b in bins.values()):
                r.append(f"{key}: by_size.{cls} invalid")
        if not isinstance(a["per_gt"], list) or not all(
                isinstance(g, dict) and isinstance(g.get("image"), str) and isinstance(g.get("matched"), list)
                for g in a["per_gt"]):
            r.append(f"{key}: per_gt entries invalid")
    return r


def _count_ok(t) -> bool:
    """{n, tp} 정수이고 0 ≤ tp ≤ n."""
    return (isinstance(t, dict) and _is_int(t.get("n")) and _is_int(t.get("tp")) and 0 <= t["tp"] <= t["n"])


# ---------------------------------------------------------------- 실제 평가 입력 fingerprint·완료 수용
def _file_sha(path: str) -> str:
    """파일 sha256."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dir_digest(path: str) -> dict:
    """폴더의 파일 수와 (이름, 파일 sha256) 누적 해시. 내용이 하나라도 바뀌면 달라진다."""
    h = hashlib.sha256()
    names = sorted(n for n in os.listdir(path) if os.path.isfile(os.path.join(path, n)))
    for n in names:
        h.update(f"{n}\0{_file_sha(os.path.join(path, n))}\n".encode())
    return {"count": len(names), "sha256": h.hexdigest()}


def eval_inputs_fingerprint(repo_root: str, eval_cfg: "EvalConfig") -> dict:
    """정확도 계획의 실제 평가 입력 fingerprint: 영상·라벨 누적 해시, dataset.yaml, Gazebo manifest.

    `eval_config_id`는 경로·설정 식별자일 뿐이라, 같은 경로의 내용이 바뀌면 이 값으로 잡는다.
    """
    out = {}
    for key, data_dir, split, imgsz in eval_cfg.accuracy_plan():
        base = os.path.join(repo_root, data_dir)
        e = {"data_dir": data_dir, "split": split, "images": _dir_digest(os.path.join(base, "images", split)),
             "labels": _dir_digest(os.path.join(base, "labels", split)),
             "dataset_yaml": {"sha256": _file_sha(os.path.join(base, "dataset.yaml"))}}
        if data_dir == eval_cfg.gazebo["data_dir"]:
            mpath = os.path.join(repo_root, eval_cfg.gazebo["manifest"])
            with open(mpath, "rb") as f:
                lines = f.read().count(b"\n")
            e["manifest"] = {"path": eval_cfg.gazebo["manifest"], "sha256": _file_sha(mpath), "lines": lines}
        out[key] = e
    return out


def canonical_sha(obj) -> str:
    """정규 JSON(키 정렬·공백 없음)의 sha256. 완료 수용 조건의 결합 값."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def accepted_eval(eval_dir: str) -> tuple[dict | None, list[str]]:
    """완료로 받아들일 수 있는 eval.json인가(게시 계약).

    eval.json이 읽히고, 그 `attempt_<n>/status.json`이 evaluated이며 `eval_sha256`이 eval.json 정규 해시와
    같고, 요약 행을 만들 수 있어야 한다. 필수 기록을 eval.json보다 먼저 쓰므로 eval.json만 남거나
    바뀐 경우는 해시가 맞지 않아 완료가 아니다.
    """
    path = os.path.join(eval_dir, "eval.json")
    try:
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
    except (OSError, ValueError) as exc:
        return None, [f"eval.json unreadable: {type(exc).__name__}: {exc}"]
    if not isinstance(rec, dict) or not _is_int(rec.get("attempt")):
        return None, ["eval.json has no attempt number"]
    try:
        with open(os.path.join(eval_dir, f"attempt_{rec['attempt']}", "status.json"), encoding="utf-8") as f:
            st = json.load(f)
    except (OSError, ValueError) as exc:
        return None, [f"attempt status unreadable: {type(exc).__name__}: {exc}"]
    if st.get("status") != "evaluated" or st.get("eval_sha256") != canonical_sha(rec):
        return None, ["eval.json does not match its attempt status (not a published result)"]
    try:
        summary_row(exp_id=rec["exp_id"], train=rec.get("train") or {}, record=rec)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        return None, [f"eval.json fields invalid: {type(exc).__name__}: {exc}"]
    return rec, []


# ---------------------------------------------------------------- Gazebo 세부 클래스
def load_gz_parts(manifest_path: str) -> dict:
    """manifest.jsonl → {영상 파일 이름: labeled 부품 cls 목록(라벨 줄 순서)}."""
    out = {}
    with open(manifest_path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            d = json.loads(line)
            name = f"{d['sample_id']}.png"
            if name in out:
                raise EvalError(f"duplicate sample_id in manifest line {n}: {d['sample_id']}")
            out[name] = [p["cls"] for p in d["parts"] if p.get("labeled")]
    return out


def fine_recall(per_gt: list, parts: dict) -> dict:
    """정답별 매칭 → 세부 클래스(BOLT_A/BOLT_B/NUT) n·tp·재현율. 영상 누락·개수 불일치는 EvalError."""
    tally = {c: {"n": 0, "tp": 0} for c in FINE_CLASSES}
    for g in per_gt:
        if g["image"] not in parts:
            raise EvalError(f"{g['image']}: not in manifest")
        cls = parts[g["image"]]
        if len(cls) != len(g["matched"]):
            raise EvalError(f"{g['image']}: manifest labeled parts {len(cls)} != label lines {len(g['matched'])}")
        for c, m in zip(cls, g["matched"]):
            if c not in tally:
                raise EvalError(f"{g['image']}: unknown fine class {c!r}")
            tally[c]["n"] += 1
            tally[c]["tp"] += int(bool(m))
    return {c: {**v, "recall": (v["tp"] / v["n"]) if v["n"] else None} for c, v in tally.items()}


# ---------------------------------------------------------------- 기록·요약
def _recall(t: dict) -> float | None:
    """n·tp → 재현율."""
    return (t["tp"] / t["n"]) if t.get("n") else None


def build_eval_record(*, exp_id: str, weights: dict, eval_cfg: EvalConfig, fingerprint: dict, timing: dict,
                      accuracy: dict, plan: list, parts: dict, loads: list) -> tuple[dict, dict]:
    """(eval.json, eval_per_gt.json). accuracy·timing은 검사를 통과한 자식 기록이다. Gazebo 키에는 세부 재현율을 붙인다."""
    results, per_gt = {}, {}
    for key, *_ in plan:
        a = dict(accuracy[key])
        per_gt[key] = a.pop("per_gt")
        if key.startswith("gz"):
            a["fine"] = fine_recall(per_gt[key], parts)
        a["recall"] = {c: _recall(a["total"][c]) for c in ("bolt", "nut")}
        results[key] = a
    record = {"exp_id": exp_id, "weights": weights,
              "conditions": {"eval_config_id": eval_cfg.eval_config_id, "conf": eval_cfg.conf, "iou": eval_cfg.iou,
                             "device": eval_cfg.device, "edges_px": list(eval_cfg.edges_px),
                             "public": thaw(eval_cfg.public), "gazebo": thaw(eval_cfg.gazebo),
                             "timing": thaw(eval_cfg.timing), "data_fingerprint": fingerprint},
              "accuracy": results,
              "timing": {str(s): summarize_timing({**raw, "imgsz": int(s)}) for s, raw in timing.items()},
              "loads": loads}
    return record, {"exp_id": exp_id, "per_gt": per_gt}


def _f(v, nd: int = 3) -> str:
    """숫자 → 문자열(없으면 -)."""
    return "-" if v is None else f"{v:.{nd}f}"


def summary_row(*, exp_id: str, train: dict, record: dict) -> dict:
    """1단계 실험 기록 표 열 순서의 한 행(사실 값만, 해석 칸은 사람이 채운다)."""
    acc, tim = record["accuracy"], record["timing"]
    pub = next((v for k, v in acc.items() if k.startswith("test")), None)
    gz = {k: v for k, v in acc.items() if k.startswith("gz")}

    def gz_text(k, v):
        """한 크기의 Gazebo 재현율."""
        f = v.get("fine", {})
        return (f"{k[2:]}: {_f(v['recall']['bolt'], 2)}/{_f(v['recall']['nut'], 2)} · "
                + "/".join(_f((f.get(c) or {}).get("recall"), 2) for c in FINE_CLASSES))

    sizes = (" · ".join(f"{c} " + ", ".join(f"{b} {_f(x['recall'], 2)}({x['n']})" for b, x in pub["by_size"][c].items())
                        for c in ("bolt", "nut")) if pub else "-")
    fp = " · ".join(f"{k}: {v['predictions']['bolt']['fp']}/{v['predictions']['nut']['fp']}" for k, v in acc.items())
    perf = " · ".join(f"{s}: P95 {_f(t['call_ms']['p95'], 1)} ms(sync {_f(t['sync_ms']['p95'], 1)}), "
                      f"GPU 최대 {_f(max((x or 0) for x in t['peak_mem_bytes'].values()) / 2 ** 20, 0)} MiB"
                      for s, t in tim.items())
    row = {
        SUMMARY_COLUMNS[0]: exp_id,
        SUMMARY_COLUMNS[1]: f"{train.get('family')} · {train.get('weights_source')} · {train.get('weights_sha256')}",
        SUMMARY_COLUMNS[2]: train.get("train_args_text", "-"),
        SUMMARY_COLUMNS[3]: train.get("data_text", "-"),
        SUMMARY_COLUMNS[4]: train.get("versions_text", "-"),
        SUMMARY_COLUMNS[5]: (f"{_f(pub['map50_box'])}/{_f(pub['map50_mask'])} · {sizes}" if pub else "-"),
        SUMMARY_COLUMNS[6]: " · ".join(gz_text(k, v) for k, v in gz.items()) or "-",
        SUMMARY_COLUMNS[7]: fp,
        SUMMARY_COLUMNS[8]: f"{perf} · 학습 {train.get('train_wall_text', '-')}",
        SUMMARY_COLUMNS[9]: train.get("notes", "") + " (해석은 사람이 작성)",
    }
    return row


def summary_markdown(rows: list) -> str:
    """요약 표(Markdown)."""
    out = ["| " + " | ".join(SUMMARY_COLUMNS) + " |", "|" + "---|" * len(SUMMARY_COLUMNS)]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]).replace("|", "/") for c in SUMMARY_COLUMNS) + " |")
    return "\n".join(out) + "\n"


def images_in(data_root: str, split: str) -> list:
    """분할의 영상 경로(이름순)."""
    d = os.path.join(data_root, "images", split)
    return [os.path.join(d, n) for n in sorted(os.listdir(d))]


__all__ = ["ACC_REQUIRED", "EvalConfig", "EvalError", "FINE_CLASSES", "PHASES", "SUMMARY_COLUMNS",
           "accepted_eval", "build_eval_record", "canonical_sha", "eval_inputs_fingerprint", "eval_config_from_dict", "fine_recall", "images_in", "load_eval_config",
           "load_gz_parts", "percentile", "run_accuracy", "run_timing", "summarize_timing", "summary_markdown",
           "summary_row", "timing_stats", "validate_accuracy", "validate_timing"]
