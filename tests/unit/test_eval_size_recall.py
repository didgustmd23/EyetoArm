"""scripts/eval_size_recall.py 매칭·구간 로직 단위 시험(합성 마스크, ultralytics 불필요)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "eval_size_recall", Path(__file__).resolve().parents[2] / "scripts" / "eval_size_recall.py")
ev = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = ev
_SPEC.loader.exec_module(ev)

EDGES = ev.DEFAULT_EDGES


def _rect(x0, y0, x1, y1):
    """축 정렬 사각형 꼭짓점 4개를 (4, 2) float 배열로 만든다."""
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def _mask(poly, shape=(100, 100)):
    """다각형을 (높이, 폭) bool 마스크로 채운다(기본 100×100)."""
    return ev.polygon_mask(poly, shape)


def test_bins_and_labels():
    """크기 구간 이름과 경계 배정이 맞다(하한 포함·상한 제외, 25 이상은 마지막 구간)."""
    assert ev.bin_labels(EDGES) == ["<10", "10-12", "12-15", "15-25", ">=25"]
    assert [ev.bin_index(s, EDGES) for s in (9.9, 10.0, 11.9, 12.0, 14.99, 15.0, 24.9, 25.0, 80)] == \
        [0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_short_side_rotated():
    """회전한 40×12 사각형의 짧은 변은 12 px다(최소 외접 사각형 기준)."""
    c, s = np.cos(0.4), np.sin(0.4)
    pts = np.array([[-20, -6], [20, -6], [20, 6], [-20, 6]], float) @ np.array([[c, s], [-s, c]]) + 50
    assert ev.short_side(pts) == pytest.approx(12.0, abs=0.05)


def test_match_one_to_one_class_and_iou():
    """정답 하나에 예측 하나만 짝지어지고, 클래스가 다른 예측은 겹쳐도 짝이 되지 않는다."""
    g0, g1 = _rect(10, 10, 30, 30), _rect(60, 60, 80, 80)
    gt = np.stack([_mask(g0), _mask(g1)])

    # 예측: g0와 크게 겹치는 nut(클래스 1) 2개, g1과 겹치지만 클래스가 다른 것 1개
    p_a = _mask(_rect(11, 11, 31, 31))
    p_b = _mask(_rect(10, 10, 29, 30))
    p_c = _mask(_rect(60, 60, 80, 80))

    m = ev.match(gt, [1, 0], np.stack([p_a, p_b, p_c]), [1, 1, 1], [0.6, 0.9, 0.99])

    assert m.tolist() == [True, False]  # g1은 클래스 불일치로 미검출, g0은 한 번만


def test_match_iou_threshold():
    """IoU 1/3 예측은 임계값 0.5에서는 미검출, 0.3에서는 검출이다."""
    gt = np.stack([_mask(_rect(10, 10, 30, 30))])
    shifted = np.stack([_mask(_rect(20, 10, 40, 30))])  # IoU 1/3

    assert ev.match(gt, [0], shifted, [0], [0.9], 0.5).tolist() == [False]
    assert ev.match(gt, [0], shifted, [0], [0.9], 0.3).tolist() == [True]


def test_match_empty():
    """예측이 없으면 모든 정답이 미검출이고, 정답이 없으면 빈 결과다."""
    gt = np.stack([_mask(_rect(10, 10, 30, 30))])
    assert ev.match(gt, [0], np.zeros((0, 100, 100), bool), [], []).tolist() == [False]
    assert ev.match(np.zeros((0, 100, 100), bool), [], gt, [0], [0.5]).tolist() == []


def test_recall_table_counts():
    """클래스·크기 구간별 정답 수·검출 수·재현율을 세고, 정답이 없는 구간의 재현율은 None이다."""
    sizes = [8, 9, 11, 13, 20, 30, 30]
    classes = [1, 1, 1, 1, 0, 0, 1]
    matched = [False, True, True, False, True, False, True]

    t = ev.recall_table(sizes, classes, matched, EDGES)

    assert t["nut"]["<10"] == {"n": 2, "tp": 1, "recall": 0.5}
    assert t["nut"]["10-12"]["recall"] == 1.0 and t["nut"]["12-15"]["recall"] == 0.0
    assert t["bolt"]["15-25"]["n"] == 1 and t["bolt"]["<10"]["recall"] is None
    assert t["bolt"][">=25"] == {"n": 1, "tp": 0, "recall": 0.0}


def test_scale_to_canvas_halves_sizes_and_centers():
    """0.5배 축소 영상을 원래 크기 회색(114) 캔버스 가운데에 놓고, 다각형도 같이 줄이고 옮긴다."""
    img = np.zeros((100, 100, 3), np.uint8)
    poly = _rect(0, 0, 40, 20)

    canvas, (p2,) = ev.scale_to_canvas(img, [poly], 0.5)

    assert canvas.shape == img.shape and canvas[0, 0, 0] == 114 and canvas[50, 50, 0] == 0
    assert ev.short_side(p2) == pytest.approx(ev.short_side(poly) / 2)
    assert p2.min(axis=0).tolist() == [25, 25]


def test_read_yolo_labels(tmp_path):
    """YOLO 분할 라벨의 정규화 좌표를 영상 화소로 되돌리고, 라벨 파일이 없으면 빈 목록이다."""
    f = tmp_path / "a.txt"
    f.write_text("1 0.1 0.1 0.3 0.1 0.3 0.5\n", encoding="utf-8")

    cls, polys = ev.read_yolo_labels(f, 200, 100)

    assert cls == [1] and polys[0].tolist() == [[20, 10], [60, 10], [60, 50]]
    assert ev.read_yolo_labels(tmp_path / "none.txt", 10, 10) == ([], [])


def test_match_full_counts_false_positives():
    """`match_full`은 정답과 짝지어지지 않은 예측(다른 위치·중복 예측)을 오검출로 표시한다."""
    gt = np.stack([_mask(_rect(10, 10, 30, 30))])
    preds = np.stack([_mask(_rect(10, 10, 30, 30)), _mask(_rect(60, 60, 80, 80)), _mask(_rect(11, 11, 30, 30))])

    g, p = ev.match_full(gt, [1], preds, [1, 1, 1], [0.9, 0.8, 0.7])

    assert g.tolist() == [True] and p.tolist() == [True, False, False]  # 중복 예측은 오검출
