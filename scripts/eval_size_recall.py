#!/usr/bin/env python3
"""YOLO 분할 모델의 클래스별·크기 구간별 재현율 평가(파일럿, 지시서 _workspace/54 §3).

- 정답: `data/derived/<데이터>/labels/<split>/*.txt`(YOLO 분할, coco_to_yolo_seg.py 출력)
- 크기: 정답 다각형의 최소 외접 사각형(minAreaRect) **짧은 변**(평가 영상 화소 기준).
  기본 구간 <10 · 10~12 · 12~15 · 15~25 · ≥25 px. `_workspace/49` 통계와 같은 정의.
- 매칭: 같은 클래스에서 마스크 IoU ≥ `--iou`(기본 0.5)인 쌍을 신뢰도 높은 예측부터 일대일로 잡는다.
- 신뢰도 임계값 `--conf`(기본 0.25).
- `--scale s`(< 1): 영상을 s배로 줄여 같은 크기의 회색(114) 캔버스 가운데에 놓고 정답도 같이 변환한다.
  작은 물체 표본을 늘리는 대략 추정용이다(Gazebo 영상 대체가 아님). 크기 구간은 캔버스 화소 기준이며,
  모델 입력 화소는 캔버스 화소 × imgsz / 캔버스 긴 변이다.
- `--map`: ultralytics `val`로 해당 분할의 mAP50(박스·마스크)도 출력한다.

매칭·구간 함수는 ultralytics 없이 시험할 수 있다(ultralytics는 main에서만 import).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

DEFAULT_EDGES = (10.0, 12.0, 15.0, 25.0)
CLASS_NAMES = ("bolt", "nut")


# ---------------------------------------------------------------- 순수 함수
def bin_labels(edges) -> list:
    return ([f"<{edges[0]:g}"] + [f"{a:g}-{b:g}" for a, b in zip(edges[:-1], edges[1:])]
            + [f">={edges[-1]:g}"])


def bin_index(size_px: float, edges) -> int:
    return int(np.digitize([size_px], edges)[0])


def short_side(poly_px: np.ndarray) -> float:
    (_, _), (w, h), _ = cv2.minAreaRect(np.asarray(poly_px, np.float32))
    return float(min(w, h))


def polygon_mask(poly_px: np.ndarray, shape) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(poly_px)).astype(np.int32)], 1)
    return m.astype(bool)


def mask_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a (Na,H,W), b (Nb,H,W) bool → (Na,Nb) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    fa = a.reshape(len(a), -1).astype(np.float32)
    fb = b.reshape(len(b), -1).astype(np.float32)
    inter = fa @ fb.T
    union = fa.sum(1)[:, None] + fb.sum(1)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def match_full(gt_masks, gt_cls, pred_masks, pred_cls, pred_conf, iou_thr=0.5):
    """(정답별 매칭 여부, 예측별 매칭 여부). 같은 클래스, IoU ≥ iou_thr, 신뢰도 높은 예측부터 일대일."""
    gt_cls = np.asarray(gt_cls)
    pred_cls = np.asarray(pred_cls)
    matched = np.zeros(len(gt_cls), bool)
    pred_ok = np.zeros(len(pred_cls), bool)
    if len(gt_cls) == 0 or len(pred_cls) == 0:
        return matched, pred_ok
    iou = mask_iou_matrix(np.asarray(pred_masks), np.asarray(gt_masks))
    for p in np.argsort(-np.asarray(pred_conf), kind="stable"):
        cand = np.flatnonzero((gt_cls == pred_cls[p]) & ~matched & (iou[p] >= iou_thr))
        if cand.size:
            g = cand[np.argmax(iou[p, cand])]
            matched[g] = True
            pred_ok[p] = True
    return matched, pred_ok


def match(gt_masks, gt_cls, pred_masks, pred_cls, pred_conf, iou_thr=0.5) -> np.ndarray:
    """정답별 매칭 여부(bool, 길이 Ngt)."""
    return match_full(gt_masks, gt_cls, pred_masks, pred_cls, pred_conf, iou_thr)[0]


def recall_table(sizes, classes, matched, edges) -> dict:
    """{클래스: {구간: {"n": n, "tp": tp, "recall": r 또는 None}}}"""
    labels = bin_labels(edges)
    out = {}
    for c, name in enumerate(CLASS_NAMES):
        out[name] = {}
        for b, lab in enumerate(labels):
            sel = [i for i in range(len(sizes)) if classes[i] == c and bin_index(sizes[i], edges) == b]
            n = len(sel)
            tp = int(sum(bool(matched[i]) for i in sel))
            out[name][lab] = {"n": n, "tp": tp, "recall": (tp / n) if n else None}
    return out


def scale_to_canvas(img: np.ndarray, polys_px: list, scale: float):
    """영상을 scale배로 줄여 같은 크기 회색 캔버스 가운데에 놓고 다각형도 같이 옮긴다."""
    h, w = img.shape[:2]
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full_like(img, 114)
    ox, oy = (w - nw) // 2, (h - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = small
    sx, sy = nw / w, nh / h
    return canvas, [np.asarray(p) * [sx, sy] + [ox, oy] for p in polys_px]


def read_yolo_labels(path: Path, w: int, h: int):
    cls, polys = [], []
    if path.is_file():
        for ln in path.read_text(encoding="utf-8").splitlines():
            v = ln.split()
            if len(v) >= 7:
                cls.append(int(v[0]))
                polys.append(np.asarray(v[1:], float).reshape(-1, 2) * [w, h])
    return cls, polys


# ---------------------------------------------------------------- 실행
def evaluate(model, data_dir: Path, split: str, imgsz: int, conf: float, iou_thr: float, scale: float,
             edges, device) -> dict:
    img_dir = data_dir / "images" / split
    lbl_dir = data_dir / "labels" / split
    sizes, classes, matched_all = [], [], []
    preds = {c: {"n": 0, "tp": 0} for c in range(len(CLASS_NAMES))}
    per_gt = []
    names = sorted(os.listdir(img_dir))
    for name in names:
        img = cv2.imread(str(img_dir / name), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"영상 해석 실패: {name}")
        h, w = img.shape[:2]
        gcls, gpolys = read_yolo_labels(lbl_dir / (Path(name).stem + ".txt"), w, h)
        if scale != 1.0:
            img, gpolys = scale_to_canvas(img, gpolys, scale)
        gmasks = np.stack([polygon_mask(p, img.shape) for p in gpolys]) if gpolys else np.zeros((0, h, w), bool)
        res = model.predict(img, imgsz=imgsz, conf=conf, device=device, retina_masks=True, verbose=False)[0]
        if res.masks is not None and len(res.boxes):
            pm = res.masks.data.cpu().numpy() > 0.5
            pc = res.boxes.cls.cpu().numpy().astype(int)
            pf = res.boxes.conf.cpu().numpy()
        else:
            pm, pc, pf = np.zeros((0, h, w), bool), np.zeros(0, int), np.zeros(0)
        m, pok = match_full(gmasks, gcls, pm, pc, pf, iou_thr)
        for c, ok in zip(pc, pok):
            if int(c) in preds:
                preds[int(c)]["n"] += 1
                preds[int(c)]["tp"] += int(ok)
        per_gt.append({"image": name, "matched": [bool(x) for x in m],
                       "short_px": [round(short_side(p), 2) for p in gpolys]})
        sizes += [short_side(p) for p in gpolys]
        classes += gcls
        matched_all += list(m)
    table = recall_table(sizes, classes, matched_all, edges)
    total = {CLASS_NAMES[c]: {"n": int(sum(1 for k in classes if k == c)),
                              "tp": int(sum(1 for k, mm in zip(classes, matched_all) if k == c and mm))}
             for c in range(len(CLASS_NAMES))}
    return {"split": split, "images": len(names), "imgsz": imgsz, "conf": conf, "iou": iou_thr, "scale": scale,
            "edges_px": list(edges), "by_size": table, "total": total,
            "predictions": {CLASS_NAMES[c]: {"n": v["n"], "tp": v["tp"], "fp": v["n"] - v["tp"],
                                             "precision": (v["tp"] / v["n"]) if v["n"] else None}
                            for c, v in preds.items()},
            "per_gt": per_gt}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default="data/derived/bolt_nut_v2")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--edges", type=float, nargs="+", default=DEFAULT_EDGES)
    p.add_argument("--device", default="0")
    p.add_argument("--map", action="store_true", help="ultralytics val로 mAP50도 출력(scale 1.0에서만)")
    p.add_argument("--out", help="결과 JSON 경로")
    p.add_argument("--workers", type=int, default=0, help="ultralytics val 데이터로더 worker 수")
    p.add_argument("--val-project", default="outputs/eval_val", help="ultralytics val 출력 폴더(Git 제외 위치)")
    a = p.parse_args(argv)
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_OFFLINE", "1")
    from ultralytics import YOLO  # 지연 import: 순수 함수 시험은 ultralytics 없이 돈다
    model = YOLO(a.weights)
    out = evaluate(model, Path(a.data), a.split, a.imgsz, a.conf, a.iou, a.scale, tuple(a.edges), a.device)
    if a.map and a.scale == 1.0:
        r = model.val(data=str(Path(a.data) / "dataset.yaml"), split=a.split, imgsz=a.imgsz, device=a.device,
                      plots=False, verbose=False, batch=8, workers=a.workers, project=str(Path(a.val_project).resolve()), name="val",
                      exist_ok=True)  # 상대 경로면 ultralytics가 runs/<task>/ 아래에 만든다
        out["map50_box"] = float(r.box.map50)
        out["map50_mask"] = float(r.seg.map50)
    text = json.dumps(out, ensure_ascii=False, indent=2)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
