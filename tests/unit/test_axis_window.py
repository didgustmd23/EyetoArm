"""M2 창(부분 마스크) 계산과 convert 창 경로(래스터·IoU·윤곽) 단위시험(#265 C1·C2).

기준은 같은 마스크를 프레임 전체로 계산한 결과다. 창 계산은 이것과 끝점·진단·사유가 비트까지 같아야 한다.
"""
import json
import unittest
from unittest import mock

import cv2
import numpy as np

import eyetoarm.vision.ai_convert as ai_convert
import eyetoarm.vision.axis_endpoints as ax
from eyetoarm.vision.ai_result import AXIS_EXTRACTION_ERROR
from test_ai_axis_path import bolt_mask, run_job
from test_ai_config import ai_config
from test_axis_endpoints_m2 import drawn_bolt


def full_frame_m2(mask):
    """기준: 창 없이 프레임 전체를 창으로 본 M2(바꾸기 전 계산과 같은 경로)."""
    h, w = mask.shape

    return ax._extract_m2(mask, 0, 0, h, w, ax.MODEL_DERIVED)


def canon(est):
    """비교용 정규형(실수는 repr로 비트 비교)."""
    return json.dumps({"method": est.method, "valid": est.valid, "reason": est.reason,
                       "endpoints": None if est.endpoints_px is None else est.endpoints_px.tolist(),
                       "diag": {k: repr(v) for k, v in sorted(est.diag.items())}})


def shift_to(mask, side, gap):
    """전경을 프레임 한 변(left/top/right/bottom)에서 gap 화소 떨어지게 옮긴다(음수면 그만큼 잘림)."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    dx = {"left": gap - xs.min(), "right": (w - 1 - gap) - xs.max()}.get(side, 0)
    dy = {"top": gap - ys.min(), "bottom": (h - 1 - gap) - ys.max()}.get(side, 0)
    out = np.zeros_like(mask)
    nx, ny = xs + dx, ys + dy
    keep = (nx >= 0) & (nx < w) & (ny >= 0) & (ny < h)
    out[ny[keep], nx[keep]] = True

    return out


def sample_masks():
    """(이름, 마스크): 각도별 볼트, 네 변 0·1·2 px·잘림, 모서리, 구멍, 조각, 작은 마스크, 직선, 정사각형, 빈 마스크."""
    out = []

    for angle in range(0, 180, 15):
        m = drawn_bolt(angle)[0]
        out.append((f"angle{angle}", m))

        for side in ("left", "top", "right", "bottom"):
            for gap in (-3, 0, 1, 2):
                out.append((f"angle{angle}_{side}{gap}", shift_to(m, side, gap)))

    base = drawn_bolt(30)[0]
    out.append(("corner", shift_to(shift_to(base, "left", 0), "top", 0)))
    holed = base.copy()
    ys, xs = np.nonzero(holed)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    holed[cy - 1:cy + 1, cx - 1:cx + 1] = False
    out.append(("hole", holed))
    frag = base.copy()
    frag[5:8, 200:203] = True
    out.append(("fragment", frag))
    tiny = np.zeros((240, 240), bool)
    tiny[100:103, 100:105] = True
    out.append(("tiny", tiny))
    line = np.zeros((240, 240), bool)
    line[120, 50:150] = True
    out.append(("line", line))
    square = np.zeros((240, 240), bool)
    square[60:80, 60:80] = True
    out.append(("square", square))
    out.append(("empty", np.zeros((240, 240), bool)))

    return out


class MaskWindowTest(unittest.TestCase):
    """`mask_window`: 전경 범위 + 여유, 프레임 안으로 자름, 복사하지 않음."""

    def test_margin_and_clamp(self):
        """안쪽 마스크는 사방 1 px 여유, 프레임 경계에 닿은 쪽은 경계에서 자른다."""
        m = np.zeros((50, 60), bool)
        m[10:20, 30:45] = True
        sub, (x0, y0) = ax.mask_window(m)
        self.assertEqual(((x0, y0), sub.shape), ((29, 9), (12, 17)))
        self.assertTrue(np.shares_memory(sub, m))
        self.assertFalse(sub[0].any() or sub[-1].any() or sub[:, 0].any() or sub[:, -1].any())

        m2 = np.zeros((50, 60), bool)
        m2[0:5, 55:60] = True
        sub, (x0, y0) = ax.mask_window(m2)
        self.assertEqual(((x0, y0), sub.shape), ((54, 0), (6, 6)))

    def test_empty_mask(self):
        """전경이 없으면 (0, 0)의 1×1 빈 창이다."""
        sub, origin = ax.mask_window(np.zeros((5, 7), bool))
        self.assertEqual((origin, sub.shape, bool(sub.any())), ((0, 0), (1, 1), False))

    def test_rejects_non_bool(self):
        """bool 2차원 배열이 아니면 ValueError."""
        for bad in (np.zeros((4, 4), np.uint8), np.zeros(4, bool), [[True]]):
            with self.subTest(bad=type(bad).__name__), self.assertRaises(ValueError):
                ax.mask_window(bad)


class WindowEquivalenceTest(unittest.TestCase):
    """창 계산 결과는 전체 프레임 계산과 비트까지 같다."""

    def test_extract_m2_matches_full_frame(self):
        """공개 extract_m2(내부 창)와 extract_m2_window가 전체 프레임 기준과 같다(경계·잘림·위상 사례 포함)."""
        reasons = set()

        for name, m in sample_masks():
            with self.subTest(case=name):
                ref = full_frame_m2(m)
                sub, origin = ax.mask_window(m)
                self.assertEqual(canon(ax.extract_m2(m)), canon(ref))
                self.assertEqual(canon(ax.extract_m2_window(sub, origin, m.shape)), canon(ref))
                reasons.add(ref.reason)

        # 사례가 유효·잘림·구멍·조각·작은 마스크·퇴화(정사각형)·머리 모호(직선) 경로를 실제로 지났는지
        self.assertTrue({None, "CONTOUR_CLIPPED", "MASK_HOLE", "MASK_FRAGMENTED", "NO_OBJECT",
                         "AXIS_DEGENERATE", "AXIS_HEAD_AMBIGUOUS"} <= reasons, reasons)

    def test_border_gap_decides_clipping(self):
        """경계에 닿으면(0 px) CONTOUR_CLIPPED, 1 px 떨어지면 잘림이 아니다(창 여유가 경계에서 잘려도 같다)."""
        m = drawn_bolt(0)[0]

        for side in ("left", "top", "right", "bottom"):
            with self.subTest(side=side):
                self.assertEqual(ax.extract_m2(shift_to(m, side, 0)).reason, "CONTOUR_CLIPPED")
                self.assertNotEqual(ax.extract_m2(shift_to(m, side, 1)).reason, "CONTOUR_CLIPPED")


class CheckWindowTest(unittest.TestCase):
    """`extract_m2_window` 입력 경계."""

    def test_rejects_bad_windows(self):
        """프레임 밖 창·빈 창·정수 아닌 원점·여유 없는 안쪽 가장자리는 ValueError."""
        m = drawn_bolt(0)[0]
        sub, (x0, y0) = ax.mask_window(m)
        tight = sub[1:-1, 1:-1]
        bad = [(sub, (x0, y0), (10, 10)), (sub, (-1, y0), m.shape), (sub[:0], (x0, y0), m.shape),
               (sub, (True, y0), m.shape), (sub, (float(x0), y0), m.shape), (tight, (x0 + 1, y0 + 1), m.shape),
               (sub.astype(np.uint8), (x0, y0), m.shape)]

        for k, (s, origin, hw) in enumerate(bad):
            with self.subTest(k=k), self.assertRaises(ValueError):
                ax.extract_m2_window(s, origin, hw)

    def test_frame_edge_may_hold_foreground(self):
        """프레임 경계와 겹친 창 가장자리는 전경이 있어도 받는다(잘림 판정 대상)."""
        m = shift_to(drawn_bolt(0)[0], "left", 0)
        sub, origin = ax.mask_window(m)
        self.assertEqual(origin[0], 0)
        self.assertEqual(ax.extract_m2_window(sub, origin, m.shape).reason, "CONTOUR_CLIPPED")


class ConvertWindowTest(unittest.TestCase):
    """convert 창 경로: 다각형 래스터 창·창 IoU·창 윤곽이 전체 crop 계산과 같다."""

    def test_raster_window_matches_full_raster(self):
        """다각형 창 래스터를 crop에 펼치면 전체 래스터와 같고, 창 밖은 0이다(경계 근처 포함)."""
        for name, m in sample_masks():
            pts, _, _ = ai_convert._geometry(m)

            if pts is None:
                continue

            with self.subTest(case=name):
                h, w = m.shape
                full = ai_convert.rasterize_polygon(pts.astype(np.float32), np.zeros(2), (w, h))
                win = ai_convert.rasterize_polygon_window(pts.astype(np.float32), np.zeros(2), (w, h))
                pasted = np.zeros_like(full)
                sh, sw = win.mask.shape
                pasted[win.y0:win.y0 + sh, win.x0:win.x0 + sw] = win.mask
                np.testing.assert_array_equal(pasted, full)

    def test_raster_window_rejects_non_integer_vertices(self):
        """정수 화소 중심이 아닌 꼭짓점은 전체 래스터와 같이 ValueError."""
        with self.assertRaises(ValueError):
            ai_convert.rasterize_polygon_window(np.array([[1.5, 1.0], [5.0, 1.0], [5.0, 5.0]]), np.zeros(2), (10, 10))

    def test_window_iou_matches_full_iou(self):
        """창 IoU는 전체 마스크 IoU와 같다(겹침·접함·떨어짐·포함·빈 합집합)."""
        h, w = 60, 80

        def win(box):
            """사각형 (x0, y0, x1, y1) 전경의 창과 전체 마스크."""
            full = np.zeros((h, w), bool)
            full[box[1]:box[3], box[0]:box[2]] = True
            sub, (x0, y0) = ax.mask_window(full)

            return ai_convert._Window(sub, x0, y0), full

        boxes = [((10, 10, 30, 30), (20, 20, 40, 40)), ((10, 10, 20, 20), (20, 10, 30, 20)),
                 ((0, 0, 5, 5), (70, 50, 80, 60)), ((10, 10, 50, 50), (20, 20, 30, 30)), ((10, 10, 30, 30),) * 2]

        for a_box, b_box in boxes:
            with self.subTest(a=a_box, b=b_box):
                (a, fa), (b, fb) = win(a_box), win(b_box)
                union = np.count_nonzero(fa | fb)
                ref = float(np.count_nonzero(fa & fb)) / union
                self.assertEqual(ai_convert._iou(a, b), ref)
                self.assertEqual(ai_convert._iou(b, a), ref)

        empty = ai_convert._Window(np.zeros((1, 1), bool), 0, 0)
        self.assertEqual(ai_convert._iou(empty, empty), 0.0)

    def test_window_geometry_matches_full_contours(self):
        """창 윤곽(점·순서·바깥 수·구멍 수)이 전체 마스크 findContours 결과와 같다(경계에 닿은 마스크 포함)."""
        for name, m in sample_masks():
            with self.subTest(case=name):
                contours, hierarchy = cv2.findContours(m.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                pts, outer, holes = ai_convert._geometry(m)

                if not contours:
                    self.assertEqual((pts, outer, holes), (None, 0, 0))
                    continue

                parents = hierarchy[0][:, 3]
                ref_outer = [c for c, p in zip(contours, parents) if p == -1]
                best = max(ref_outer, key=cv2.contourArea)
                self.assertEqual((outer, holes), (len(ref_outer), len(contours) - len(ref_outer)))

                if len(best) < 3 or cv2.contourArea(best) <= 0.0:
                    self.assertIsNone(pts)
                else:
                    np.testing.assert_array_equal(pts, best.reshape(-1, 2))

    def test_m2_convert_uses_window_extractor(self):
        """m2 설정 convert 끝점은 전체 마스크 M2와 같고, 창 추출기 예외는 AXIS_EXTRACTION_ERROR 무효다."""
        mask = bolt_mask()
        _, result, obs = run_job(ai_config(axis_extractor="m2"), [mask])
        self.assertTrue(obs.valid)
        np.testing.assert_array_equal(result.instances[0].axis_endpoints_px, full_frame_m2(mask).endpoints_px)

        with mock.patch.object(ai_convert, "extract_m2_window", side_effect=RuntimeError("boom")):
            _, result, obs = run_job(ai_config(axis_extractor="m2"), [mask])

        self.assertEqual((obs.valid, obs.reason), (False, AXIS_EXTRACTION_ERROR))
        self.assertIn("boom", result.instances[0].axis_diag["error"])

    def test_overlap_iou_kept_for_m1_and_m2(self):
        """겹친 두 볼트의 axis_max_iou는 전체 마스크로 센 IoU와 같다(m1·m2 모두)."""
        a, b = bolt_mask(), bolt_mask(x0=42)
        ref = float(np.count_nonzero(a & b)) / np.count_nonzero(a | b)

        for extractor in ("m1", "m2"):
            with self.subTest(extractor=extractor):
                _, result, _ = run_job(ai_config(axis_extractor=extractor), [a, b])
                self.assertEqual([i.axis_max_iou for i in result.instances], [ref, ref])


if __name__ == "__main__":
    unittest.main()
