"""90° 돌린 추론(#417·#425)의 순수 함수: crop 돌리기, 돌린 결과를 원래 좌표로 되돌리기, 두 결과 합치기(R5).

torch·ultralytics를 쓰지 않는다. 좌표 약속(원래 crop 높이 H, 폭 W, 시계 방향 90°):
돌린 영상은 (W, H) 모양이고 `rotated[r, c] = crop[H-1-c, r]`이다. 원래 화소 (x, y)는 돌린 영상의 (c = H-1-y, r = x)에
있다. 상자는 화소 경계 좌표라 돌린 상자 (xr1, yr1, xr2, yr2)는 원래 (yr1, H-xr2, yr2, H-xr1)이다.

합치기 R5(#417에서 잰 규칙, A 결정 #429: 그대로끼리의 겹침도 합침):
1. 돌린 쪽 볼트가 그대로 쪽 너트와 마스크 IoU ≥ `nut_veto_iou`면 그 볼트를 버린다(너트 우선).
2. 남은 모두를 신뢰도 높은 순으로 보며, 이미 남긴 같은 종류와 마스크 IoU ≥ `nms_iou`면 버린다.
3. 남은 것을 「그대로 쪽 원래 순서 → 돌린 쪽 원래 순서」로 놓는다.
마스크 IoU는 두 상자가 겹치는 쌍만, 두 상자를 합친 창 안에서 센다(모델 마스크는 자기 상자 안에 있다).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from eyetoarm.vision.ai_result import SOURCE_ROT90, RawSegOutput
_BOX_MARGIN_PX = 1       # 상자 경계 반올림 여유(마스크가 상자 밖 1화소에 걸쳐도 놓치지 않게)


def _is_bolt(name: str) -> bool:
    """볼트 클래스 이름인가(ai_convert와 같은 규칙: 소문자화 후 `bolt`로 시작)."""
    return name.strip().lower().startswith("bolt")


def _is_nut(name: str) -> bool:
    """너트 클래스 이름인가(같은 방식: 소문자화 후 `nut`로 시작)."""
    return name.strip().lower().startswith("nut")


def rotate_crop(crop: np.ndarray, direction: str = "cw") -> np.ndarray:
    """crop을 시계 방향으로 90° 돌린 새 배열. 방향은 `"cw"`만 받는다."""
    if direction != "cw":
        raise ValueError(f"unsupported rotate90 direction: {direction!r}")

    return cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)


def rotate_back(raw_rot: RawSegOutput, crop_wh: tuple[int, int]) -> RawSegOutput:
    """시계 방향으로 돌린 crop의 원시 출력을 원래 crop 좌표로 되돌린다.

    `crop_wh`는 원래 crop의 (폭 W, 높이 H)다. 돌린 출력의 crop 크기가 (H, W)가 아니면 ValueError.
    """
    w, h = crop_wh

    if raw_rot.crop_wh != (h, w):
        raise ValueError(f"rotated output crop {raw_rot.crop_wh} != rotated crop {(h, w)}")

    # 마스크: 반시계 방향으로 90° 돌리면 원래 좌표(연속 배열로 소유)
    masks = np.ascontiguousarray(np.rot90(raw_rot.masks, k=1, axes=(1, 2)))

    # 상자: 화소 경계 좌표 변환
    b = raw_rot.boxes_xyxy.astype(np.float64)
    boxes = np.stack([b[:, 1], h - b[:, 2], b[:, 3], h - b[:, 0]], axis=1) if len(b) else np.zeros((0, 4))

    return RawSegOutput(boxes_xyxy=boxes, confidence=raw_rot.confidence, class_index=raw_rot.class_index,
                        masks=masks, crop_wh=(w, h), infer_ms=raw_rot.infer_ms)


def _window(box, wh: tuple[int, int]) -> tuple[int, int, int, int]:
    """실수 상자 → 정수 창 (x1, y1, x2, y2), 여유를 두고 crop 안으로 자른다."""
    w, h = wh
    x1, y1, x2, y2 = (float(v) for v in box)

    return (max(0, math.floor(x1) - _BOX_MARGIN_PX), max(0, math.floor(y1) - _BOX_MARGIN_PX),
            min(w, math.ceil(x2) + _BOX_MARGIN_PX), min(h, math.ceil(y2) + _BOX_MARGIN_PX))


def _pair_iou(ma: np.ndarray, mb: np.ndarray, wa, wb) -> float:
    """두 마스크의 IoU. 창이 겹치지 않으면 0, 겹치면 두 창을 합친 범위만 센다."""
    if wa[0] >= wb[2] or wb[0] >= wa[2] or wa[1] >= wb[3] or wb[1] >= wa[3]:
        return 0.0

    x1, y1 = min(wa[0], wb[0]), min(wa[1], wb[1])
    x2, y2 = max(wa[2], wb[2]), max(wa[3], wb[3])
    a, b = ma[y1:y2, x1:x2], mb[y1:y2, x1:x2]
    inter = int(np.count_nonzero(a & b))

    if inter == 0:
        return 0.0

    return inter / (int(np.count_nonzero(a)) + int(np.count_nonzero(b)) - inter)


def merge_r5(plain: RawSegOutput, rot: RawSegOutput, class_names: dict[int, str], *, nms_iou: float,
             nut_veto_iou: float) -> tuple[RawSegOutput, tuple[str | None, ...]]:
    """그대로 결과와 (원래 좌표로 되돌린) 돌린 결과를 R5로 합친다. (합친 출력, 인스턴스별 출처)를 돌려준다.

    두 출력의 crop 크기가 다르면 ValueError. 합친 출력의 `infer_ms`는 두 추론의 합이다.
    class_map 밖 클래스는 종류 판정에서 볼트·너트가 아닌 것으로 보고, 오류는 뒤의 `convert`가 낸다.
    """
    if plain.crop_wh != rot.crop_wh:
        raise ValueError(f"merge crop mismatch: plain {plain.crop_wh} != rotated {rot.crop_wh}")

    wh = plain.crop_wh
    name = lambda c: class_names.get(int(c), "")  # noqa: E731

    # 항목: (출처, 원래 번호, 마스크, 창, 신뢰도, 클래스)
    items = [(None, i, plain.masks[i], _window(plain.boxes_xyxy[i], wh), float(plain.confidence[i]),
              int(plain.class_index[i])) for i in range(plain.count)]
    rot_items = [(SOURCE_ROT90, j, rot.masks[j], _window(rot.boxes_xyxy[j], wh), float(rot.confidence[j]),
                  int(rot.class_index[j])) for j in range(rot.count)]

    # 1. 너트 우선: 그대로 쪽 너트와 겹치는 돌린 쪽 볼트를 버린다
    nuts = [it for it in items if _is_nut(name(it[5]))]
    rot_items = [it for it in rot_items
                 if not _is_bolt(name(it[5])) or all(_pair_iou(it[2], n[2], it[3], n[3]) < nut_veto_iou for n in nuts)]
    items += rot_items

    # 2. 신뢰도 순(같으면 앞 순서 먼저)으로 같은 종류 IoU ≥ nms_iou를 버린다
    order = sorted(range(len(items)), key=lambda k: -items[k][4])
    kept: list[int] = []

    for k in order:
        it = items[k]

        if all(items[q][5] != it[5] or _pair_iou(it[2], items[q][2], it[3], items[q][3]) < nms_iou for q in kept):
            kept.append(k)

    # 3. 그대로 쪽 → 돌린 쪽, 각자 원래 순서
    kept.sort()
    sel = [items[k] for k in kept]
    srcs = tuple(it[0] for it in sel)

    def pick(src_plain, src_rot):
        """선택한 항목의 배열 값을 출처에 따라 모은다."""
        return [src_plain[it[1]] if it[0] is None else src_rot[it[1]] for it in sel]

    w, h = wh
    boxes = np.asarray(pick(plain.boxes_xyxy, rot.boxes_xyxy), np.float32).reshape(-1, 4)
    conf = np.asarray(pick(plain.confidence, rot.confidence), np.float32)
    cls = np.asarray(pick(plain.class_index, rot.class_index), np.int64)
    masks = np.stack(pick(plain.masks, rot.masks)) if sel else np.zeros((0, h, w), bool)
    merged = RawSegOutput(boxes_xyxy=boxes, confidence=conf, class_index=cls, masks=masks, crop_wh=wh,
                          infer_ms=plain.infer_ms + rot.infer_ms)

    return merged, srcs


__all__ = ["SOURCE_ROT90", "merge_r5", "rotate_back", "rotate_crop"]
