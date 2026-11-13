#!/usr/bin/env python3
"""OpenCV ArUco 검출·IPPE_SQUARE PnP 동작 점검 (독립 시험 스크립트).

현재 Python 환경의 OpenCV가 마커 생성 → 검출 → 모서리 대응 → PnP 후보 출력까지
실제로 동작하는지 합성 입력으로 확인한다. ROS·PySide6·카메라·네트워크를 쓰지 않는다.

  python3 scripts/check_opencv.py                 # 사람이 읽는 출력
  python3 scripts/check_opencv.py --json          # 기계가 읽는 출력
  python3 scripts/check_opencv.py --output-dir outputs/opencv_check   # 합성 이미지·오버레이 저장

종료 코드: 0 = 필수 시험 전체 PASS, 1 = FAIL이 하나 이상(실행됐지만 기대 불일치),
2 = FAIL은 없지만 UNVERIFIED가 하나 이상(의존성·API 부재로 필수 시험 실행 불가).
--output-dir를 주면 이미지 저장 결과(image_save)도 별도로 기록하고 종료 코드에 포함한다:
파일 쓰기 실패·imwrite False → FAIL, 디렉터리 생성·권한 등 환경 제약 → UNVERIFIED.

기준 문서: A 상세설계 §8(마커 좌표계 M·모서리 대응·OpenCV 환경).
한계: 이 시험이 통과해도 실제 촬영 정확도, 카메라 보정, B의 최종 후보 선택·모호성 판정,
ROS 영상 변환, GUI·TCP·로봇 통합은 검증되지 않는다.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

PASS, FAIL, UNVERIFIED = "PASS", "FAIL", "UNVERIFIED"

# ── 시험 조건 (A 상세설계 §8과 사용자 지정 시험 조건) ─────────────────────────────
DICT_NAME = "DICT_4X4_50"
MARKER_ID = 7
MARKER_PX = 200                  # 생성 마커 영상 한 변(픽셀)
CANVAS_W, CANVAS_H = 640, 480    # 캔버스 = 카메라 해상도
MARKER_LENGTH_M = 0.08           # 마커 변 길이 L (m)
FX = FY = 800.0
CX, CY = 320.0, 240.0

# 합성 PnP 시험 전용 수치 허용치. 렌더링·검출 오차가 없는 정확한 투영점에 대한 값이며
# 프로젝트의 검출·자세 품질 임계값(4주차 파일럿에서 결정)이 아니다.
PNP_POS_TOL_M = 1e-6
PNP_ROT_TOL_DEG = 1e-4
# 회전 대응 시험의 기본 픽셀 허용치(시험용 값, --corner-tolerance-px로 조절).
DEFAULT_CORNER_TOL_PX = 1.0

API_NAMES = [
    "cv2.aruco",
    "cv2.aruco.getPredefinedDictionary",
    "cv2.aruco.ArucoDetector",
    "cv2.aruco.DetectorParameters",
    "cv2.aruco.DetectorParameters_create",
    "cv2.aruco.generateImageMarker",
    "cv2.aruco.drawMarker",
    "cv2.aruco.detectMarkers",
    "cv2.solvePnPGeneric",
    "cv2.SOLVEPNP_IPPE_SQUARE",
]


# ── 순수 계산 도우미 (cv2 없이 단위시험 가능) ────────────────────────────────────

def marker_object_points(length_m: float):
    """SOLVEPNP_IPPE_SQUARE 점 순서. 마커 고유 방향의 좌상·우상·우하·좌하, M 좌표(m)."""
    import numpy as np
    h = length_m / 2.0
    return np.array([[-h, +h, 0.0], [+h, +h, 0.0], [+h, -h, 0.0], [-h, -h, 0.0]], dtype=np.float64)


def rot_x(deg: float):
    """X축 회전행렬(3×3, 오른손 법칙, 각도는 도)."""
    import numpy as np
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def transform_points(R_C_M, t_C_M_m, pts_M):
    """M 좌표 점을 C 좌표로: X_C = R_C_M · X_M + t_C_M."""
    import numpy as np
    return (np.asarray(R_C_M) @ np.asarray(pts_M).T).T + np.asarray(t_C_M_m).reshape(1, 3)


def project_pinhole(pts_C, fx, fy, cx, cy):
    """왜곡 없는 핀홀 투영: u = fx·X/Z + cx, v = fy·Y/Z + cy. PnP 결과와 독립적으로 정답 영상점을 만든다."""
    import numpy as np
    P = np.asarray(pts_C, dtype=np.float64)
    return np.stack([fx * P[:, 0] / P[:, 2] + cx, fy * P[:, 1] / P[:, 2] + cy], axis=1)


def rotation_angle_deg(R_a, R_b) -> float:
    """두 회전행렬 사이 상대 회전각(°). trace 공식은 0° 근처에서 정밀도가 낮아 atan2 형식을 쓴다."""
    import numpy as np
    R = np.asarray(R_a).T @ np.asarray(R_b)
    s = 0.5 * math.sqrt(max(0.0, (R[2, 1] - R[1, 2]) ** 2 + (R[0, 2] - R[2, 0]) ** 2 + (R[1, 0] - R[0, 1]) ** 2))
    c = 0.5 * (float(np.trace(R)) - 1.0)
    return math.degrees(math.atan2(s, c))


def rot90_points(pts_xy, k: int, size_px: int, origin_xy):
    """np.rot90(marker, k)로 회전한 정사각 영상(size_px)에서 점이 옮겨 가는 위치.

    OpenCV 픽셀 중심 좌표(정수 = 픽셀 중심) 기준. np.rot90(k=1)은 화면 기준 반시계 90°이며
    마커 내부 좌표 (x, y) → (y, N−1−x). origin_xy는 캔버스에서 마커 영상의 좌상단 위치.
    """
    import numpy as np
    o = np.asarray(origin_xy, dtype=np.float64)
    p = np.asarray(pts_xy, dtype=np.float64) - o
    n1 = size_px - 1

    for _ in range(k % 4):
        p = np.stack([p[:, 1], n1 - p[:, 0]], axis=1)

    return p + o


def all_finite(*arrays) -> bool:
    """주어진 배열의 모든 값이 유한하면 True."""
    import numpy as np
    return all(bool(np.all(np.isfinite(np.asarray(a, dtype=np.float64)))) for a in arrays)


def overall_exit_code(results) -> int:
    """FAIL이 있으면 1, FAIL 없이 UNVERIFIED가 있으면 2, 모두 PASS면 0."""
    statuses = {r["status"] for r in results}

    if FAIL in statuses:
        return 1

    if UNVERIFIED in statuses:
        return 2

    return 0


def _r(x, nd=9):
    """JSON·출력용 숫자. 비유한 값은 문자열로 남겨 숨기지 않는다(판정은 별도로 FAIL)."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return x

    return round(f, nd) if math.isfinite(f) else repr(f)


def _lst(a, nd=6):
    import numpy as np
    return [[_r(v, nd) for v in row] for row in np.asarray(a, dtype=np.float64).reshape(-1, np.asarray(a).shape[-1])]


# ── 환경·API 확인 ───────────────────────────────────────────────────────────────

def _lookup(cv2, dotted: str):
    obj = cv2

    for part in dotted.split(".")[1:]:
        if not hasattr(obj, part):
            return False

        obj = getattr(obj, part)

    return True


def environment(cv2, np):
    """Python·OpenCV·NumPy 버전과 `API_NAMES` 각 API의 존재 여부(dict)."""
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "opencv_version": cv2.__version__ if cv2 else None,
        "cv2_file": getattr(cv2, "__file__", None) if cv2 else None,
        "numpy_version": np.__version__ if np else None,
        "api": {name: (_lookup(cv2, name) if cv2 else False) for name in API_NAMES},
    }


class ArucoBackend:
    """버전 문자열이 아니라 실제 기능 존재로 생성·검출 API를 고른다."""

    def __init__(self, cv2):
        self.cv2 = cv2
        a = getattr(cv2, "aruco", None)
        self.reasons = []
        self.dictionary = None
        self.generate_api = None
        self.detect_api = None

        if a is None:
            self.reasons.append("cv2.aruco 없음 (opencv-contrib 미포함 빌드일 수 있음)")
            return

        if hasattr(a, "getPredefinedDictionary"):
            self.dictionary = a.getPredefinedDictionary(getattr(a, DICT_NAME))
            self.dictionary_api = "cv2.aruco.getPredefinedDictionary"
        else:
            self.reasons.append("cv2.aruco.getPredefinedDictionary 없음")

        if hasattr(a, "generateImageMarker"):
            self.generate_api = "cv2.aruco.generateImageMarker"
        elif hasattr(a, "drawMarker"):
            self.generate_api = "cv2.aruco.drawMarker"
        else:
            self.reasons.append("마커 생성 API 없음 (generateImageMarker·drawMarker 모두 없음)")

        if hasattr(a, "ArucoDetector") and hasattr(a, "DetectorParameters"):
            self.detect_api = "cv2.aruco.ArucoDetector + DetectorParameters"
        elif hasattr(a, "detectMarkers") and hasattr(a, "DetectorParameters_create"):
            self.detect_api = "cv2.aruco.detectMarkers + DetectorParameters_create"
        else:
            self.reasons.append("검출 API 조합 없음 (ArucoDetector+DetectorParameters, "
                                "detectMarkers+DetectorParameters_create 모두 불가)")

        self._detector = None

    @property
    def usable(self) -> bool:
        """사전·생성·검출 API가 모두 골라졌으면 True."""
        return self.dictionary is not None and self.generate_api is not None and self.detect_api is not None

    def generate(self, marker_id: int, size_px: int):
        """마커 영상(uint8, size_px×size_px)을 고른 생성 API로 만든다."""
        a = self.cv2.aruco

        if self.generate_api == "cv2.aruco.generateImageMarker":
            return a.generateImageMarker(self.dictionary, marker_id, size_px)

        return a.drawMarker(self.dictionary, marker_id, size_px)

    def detect(self, gray):
        """(corners 목록[각 (4,2)], ids 목록). 반환 모서리 순서는 그대로 둔다(화면 기준 재정렬 금지)."""
        import numpy as np
        a = self.cv2.aruco

        if self.detect_api.startswith("cv2.aruco.ArucoDetector"):
            if self._detector is None:
                self._detector = a.ArucoDetector(self.dictionary, a.DetectorParameters())

            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = a.detectMarkers(gray, self.dictionary, parameters=a.DetectorParameters_create())

        ids_list = [] if ids is None else [int(i) for i in np.asarray(ids).ravel()]
        return [np.asarray(c, dtype=np.float64).reshape(4, 2) for c in corners], ids_list


# ── 시험 항목 ────────────────────────────────────────────────────────────────────

def _result(name, status, reason, **data):
    return {"name": name, "status": status, "reason": reason, **data}


def make_canvas(np, marker_img):
    """흰 캔버스 가운데에 마커를 붙인다. (캔버스, 마커 좌상단 (x, y))를 돌려준다."""
    canvas = np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8)
    ox, oy = (CANVAS_W - MARKER_PX) // 2, (CANVAS_H - MARKER_PX) // 2
    canvas[oy:oy + MARKER_PX, ox:ox + MARKER_PX] = marker_img
    return canvas, (ox, oy)


def test_api_selection(backend):
    """생성·검출 API 조합이 있으면 PASS, 없으면 UNVERIFIED."""
    if backend.usable:
        return _result("api_selection", PASS, "마커 생성·검출에 쓸 수 있는 API 조합이 있음",
                       generate_api=backend.generate_api, detect_api=backend.detect_api,
                       dictionary_api=getattr(backend, "dictionary_api", None))

    return _result("api_selection", UNVERIFIED, "; ".join(backend.reasons) or "API 조합 없음",
                   generate_api=backend.generate_api, detect_api=backend.detect_api)


def test_synthetic_detection(np, backend, ctx):
    """합성 마커 1개를 검출한다. 성공하면 `ctx`에 마커·캔버스·기준 모서리(corners0)를 남긴다."""
    name = "synthetic_detection"

    if not backend.usable:
        return _result(name, UNVERIFIED, "생성·검출 API 조합이 없어 실행 불가")

    try:
        marker = backend.generate(MARKER_ID, MARKER_PX)
        canvas, origin = make_canvas(np, marker)
        corners, ids = backend.detect(canvas)
    except Exception as e:  # noqa: BLE001 — 예외를 숨기지 않고 FAIL 근거로 남긴다
        return _result(name, FAIL, f"생성·검출 중 예외: {type(e).__name__}: {e}")

    ctx.update(marker=marker, canvas=canvas, origin=origin)

    if ids.count(MARKER_ID) != 1 or len(ids) != 1:
        return _result(name, FAIL, f"ID {MARKER_ID}가 정확히 1개 검출되어야 함 (검출 ID: {ids})", ids=ids)

    c = corners[ids.index(MARKER_ID)]

    if c.shape != (4, 2) or not all_finite(c):
        return _result(name, FAIL, f"모서리가 4개의 유한 좌표가 아님 (shape {c.shape})", corners_px=_lst(c))

    ctx["corners0"] = c
    return _result(name, PASS, f"ID {MARKER_ID} 1개, 유한한 모서리 4개", ids=ids, corners_px=_lst(c, 3),
                   marker_origin_px=list(origin))


def test_blank_negative(np, backend):
    """흰 영상에서 아무것도 검출하지 않아야 PASS(음성 시험)."""
    name = "blank_negative"

    if not backend.usable:
        return _result(name, UNVERIFIED, "검출 API 조합이 없어 실행 불가")

    try:
        _, ids = backend.detect(np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8))
    except Exception as e:  # noqa: BLE001
        return _result(name, FAIL, f"흰 영상 검출 중 예외: {type(e).__name__}: {e}")

    if ids:
        return _result(name, FAIL, f"흰 영상에서 마커가 검출됨: {ids}", ids=ids)

    return _result(name, PASS, "흰 영상에서 검출 없음 (음성 시험 통과)", ids=ids)


def test_rotation_correspondence(np, backend, ctx, tol_px: float):
    """마커를 0·90·180·270° 돌려 검출하고, 모서리 인덱스가 회전 기대 위치와 `tol_px` 안에서 대응하는지 본다."""
    name = "rotation_correspondence"

    if not backend.usable:
        return _result(name, UNVERIFIED, "생성·검출 API 조합이 없어 실행 불가")

    if "corners0" not in ctx:
        return _result(name, UNVERIFIED, "기준(0°) 검출이 실패해 기대 모서리를 만들 수 없음")

    c0, origin = ctx["corners0"], ctx["origin"]
    rows, ok, notes = [], True, []
    ctx["rotations"] = {}

    for k in range(4):
        deg = 90 * k

        try:
            rotated = np.ascontiguousarray(np.rot90(ctx["marker"], k))
            canvas, _ = make_canvas(np, rotated)
            corners, ids = backend.detect(canvas)
        except Exception as e:  # noqa: BLE001
            return _result(name, FAIL, f"{deg}° 검출 중 예외: {type(e).__name__}: {e}", rotations=rows)

        ctx["rotations"][deg] = (canvas, corners, ids)

        if ids.count(MARKER_ID) != 1:
            ok = False
            notes.append(f"{deg}°: ID {MARKER_ID} 1개 검출 실패 (검출 ID {ids})")
            rows.append({"deg_ccw": deg, "ids": ids})
            continue

        got = corners[ids.index(MARKER_ID)]
        expected = rot90_points(c0, k, MARKER_PX, origin)
        max_diff = float(np.max(np.linalg.norm(got - expected, axis=1)))

        # 인덱스 대응이 의미 있는지: 다른 순환 대응(인덱스를 1~3칸 민 경우)과의 차이가 커야 한다.
        alt = min(float(np.max(np.linalg.norm(np.roll(got, s, axis=0) - expected, axis=1))) for s in (1, 2, 3))
        row_ok = all_finite(got) and max_diff <= tol_px
        ok &= row_ok

        if not row_ok:
            notes.append(f"{deg}°: 최대 차이 {max_diff:.3f}px > 허용치 {tol_px}px")

        rows.append({"deg_ccw": deg, "ids": ids, "max_corner_diff_px": _r(max_diff, 4),
                     "min_diff_if_index_shifted_px": _r(alt, 2), "corners_px": _lst(got, 3)})

    status = PASS if ok else FAIL
    reason = (f"0·90·180·270° 모두 ID 유지, 인덱스별 모서리가 회전 기대 위치와 {tol_px}px 이내 (시험용 허용치)"
              if ok else "; ".join(notes))
    return _result(name, status, reason, corner_tolerance_px=tol_px, rotation_direction="np.rot90, 화면 기준 반시계",
                   rotations=rows)


def _pnp_candidates(cv2, np, obj_M, img_px, K, dist):
    retval, rvecs, tvecs, reproj = cv2.solvePnPGeneric(obj_M, img_px, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    cands = []
    reproj = None if reproj is None else np.asarray(reproj, dtype=np.float64).ravel()

    for i, (rv, tv) in enumerate(zip(rvecs, tvecs)):
        R, _ = cv2.Rodrigues(np.asarray(rv, dtype=np.float64))
        t = np.asarray(tv, dtype=np.float64).reshape(3)
        z = transform_points(R, t, obj_M)[:, 2]
        cands.append({"R": R, "t": t, "z": z,
                      "reproj": float(reproj[i]) if reproj is not None and i < len(reproj) else float("nan")})

    return int(retval), cands


def _cand_report(c, extra=None):
    d = {"t_C_M_m": [_r(v, 9) for v in c["t"]], "R_C_M": _lst(c["R"], 9),
         "reprojection_error_px": _r(c["reproj"], 9),
         "corner_z_m": [_r(v, 6) for v in c["z"]], "in_front": bool(all_finite(c["z"]) and (c["z"] > 0).all())}
    d.update(extra or {})
    return d


def test_pnp_synthetic(cv2, np):
    """정확한 투영점으로 IPPE_SQUARE PnP를 풀어 정답 허용치 안의 후보가 있는지 본다."""
    name = "pnp_synthetic"

    if not (_lookup(cv2, "cv2.solvePnPGeneric") and _lookup(cv2, "cv2.SOLVEPNP_IPPE_SQUARE")):
        return _result(name, UNVERIFIED, "cv2.solvePnPGeneric 또는 cv2.SOLVEPNP_IPPE_SQUARE 없음")

    obj_M = marker_object_points(MARKER_LENGTH_M)
    R_true = rot_x(150.0)
    t_true = np.array([0.02, -0.01, 0.50])
    z_true = transform_points(R_true, t_true, obj_M)[:, 2]

    if not (z_true > 0).all():
        return _result(name, FAIL, f"정답 자세의 모서리가 카메라 뒤에 있음: z={z_true}")

    img = project_pinhole(transform_points(R_true, t_true, obj_M), FX, FY, CX, CY)
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5)

    try:
        n, cands = _pnp_candidates(cv2, np, obj_M, img, K, dist)
    except Exception as e:  # noqa: BLE001
        return _result(name, FAIL, f"solvePnPGeneric 예외: {type(e).__name__}: {e}")

    reports, finite_ok, match = [], True, False

    for c in cands:
        pos_err = float(np.linalg.norm(c["t"] - t_true))
        rot_err = rotation_angle_deg(c["R"], R_true)
        finite_ok &= all_finite(c["R"], c["t"], c["z"], [c["reproj"]])

        # 정답과의 비교는 시험 판정용일 뿐, 제품의 후보 선택 방식이 아니다.
        within = pos_err <= PNP_POS_TOL_M and rot_err <= PNP_ROT_TOL_DEG
        match |= within
        reports.append(_cand_report(c, {"position_error_m": _r(pos_err, 12), "rotation_error_deg": _r(rot_err, 9),
                                        "within_synthetic_tolerance": within}))

    data = {"retval": n, "candidate_count": len(cands), "candidates": reports,
            "truth": {"R_C_M": "Rx(150°)", "t_C_M_m": t_true.tolist(), "corner_z_m": [_r(v, 6) for v in z_true]},
            "image_points_px": _lst(img, 6),
            "tolerance": {"position_m": PNP_POS_TOL_M, "rotation_deg": PNP_ROT_TOL_DEG, "kind": "합성 시험용 수치 허용치"},
            "note": "모든 후보를 출력함. 이 시험은 B의 모호성 판정·최종 후보 선택을 검증하지 않음"}

    if not cands:
        return _result(name, FAIL, "후보가 0개", **data)

    if not finite_ok:
        return _result(name, FAIL, "후보 출력에 NaN·Infinity가 있음", **data)

    if not match:
        return _result(name, FAIL, "정답 허용치 안의 후보가 없음", **data)

    return _result(name, PASS, f"후보 {len(cands)}개 출력, 정답 허용치 안의 후보 존재, 모든 값 유한", **data)


def test_detect_to_pnp(cv2, np, backend, ctx):
    """검출한 모서리를 PnP에 넣어 유한한 후보가 나오는지 본다(정답 허용치는 적용하지 않음)."""
    name = "detect_to_pnp"

    if not backend.usable:
        return _result(name, UNVERIFIED, "생성·검출 API 조합이 없어 실행 불가")

    if not (_lookup(cv2, "cv2.solvePnPGeneric") and _lookup(cv2, "cv2.SOLVEPNP_IPPE_SQUARE")):
        return _result(name, UNVERIFIED, "cv2.solvePnPGeneric 또는 cv2.SOLVEPNP_IPPE_SQUARE 없음")

    if "corners0" not in ctx:
        return _result(name, UNVERIFIED, "합성 마커 검출이 실패해 연결 시험을 할 수 없음")

    obj_M = marker_object_points(MARKER_LENGTH_M)
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)

    try:
        n, cands = _pnp_candidates(cv2, np, obj_M, ctx["corners0"], K, np.zeros(5))
    except Exception as e:  # noqa: BLE001
        return _result(name, FAIL, f"검출 모서리로 solvePnPGeneric 호출 중 예외: {type(e).__name__}: {e}")

    reports = [_cand_report(c) for c in cands]
    finite = [all_finite(c["R"], c["t"], c["z"], [c["reproj"]]) for c in cands]
    data = {"retval": n, "candidate_count": len(cands), "candidates": reports,
            "reference_only": {"expected_z_m_if_fronto_parallel": _r(FX * MARKER_LENGTH_M / MARKER_PX, 4),
                               "note": "정면 배치·200px 마커에서의 참고값. 판정에 쓰지 않음"},
            "note": "검출 API와 PnP API의 연결 동작 확인. 렌더링·검출 픽셀 오차가 있어 정답 허용치를 적용하지 않음"}

    if not any(finite):
        return _result(name, FAIL, "유한한 후보가 없음", **data)

    return _result(name, PASS, f"검출 모서리 → IPPE_SQUARE 후보 {len(cands)}개, 유한한 후보 {sum(finite)}개", **data)


# ── 이미지 저장 (--output-dir 요청 시에만) ────────────────────────────────────────
# 저장은 필수 영상처리 시험 6개와 별개인 선택 기능이다. 결과는 report["image_save"]에 따로 남기고
# 종료 코드에는 반영한다: 파일 쓰기 실패·False 반환 → FAIL(1), 디렉터리 생성·권한 등 환경 제약 → UNVERIFIED(2).

def save_images(cv2, np, ctx, out_dir: Path):
    """합성 이미지·오버레이를 저장하고 저장 결과(dict)를 돌려준다. 예외를 밖으로 던지지 않는다.

    성공 판정은 이번 호출의 cv2.imwrite 반환값(True)과 쓴 뒤 파일 존재로 한다.
    같은 이름의 기존 파일이 있다는 사실만으로 성공으로 보지 않는다.
    """
    import os
    name = "image_save"

    try:
        targets = _image_targets(cv2, np, ctx)
    except Exception as e:  # noqa: BLE001
        return _result(name, FAIL, f"저장할 오버레이 영상을 만들지 못함: {type(e).__name__}: {e}",
                       output_dir=str(out_dir), saved_files=[], failed_files=[])

    if not targets:
        return _result(name, UNVERIFIED, "저장할 합성 이미지가 없음 (합성 마커 검출 시험이 실행·통과되지 않음)",
                       output_dir=str(out_dir), saved_files=[], failed_files=[])

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return _result(name, UNVERIFIED, f"출력 디렉터리를 만들 수 없음 (환경 제약): {type(e).__name__}: {e}",
                       output_dir=str(out_dir), saved_files=[], failed_files=[])

    if not out_dir.is_dir() or not os.access(out_dir, os.W_OK | os.X_OK):
        return _result(name, UNVERIFIED, "출력 디렉터리에 쓸 권한이 없음 (환경 제약)",
                       output_dir=str(out_dir), saved_files=[], failed_files=[])

    saved, failed = [], []

    for fname, img in targets:
        path = out_dir / fname

        try:
            ok = cv2.imwrite(str(path), img)
        except Exception as e:  # noqa: BLE001 — 원인을 기록하고 다음 파일을 계속 시도한다
            failed.append({"path": str(path), "error": f"{type(e).__name__}: {e}"})
            continue

        if ok is not True:
            failed.append({"path": str(path), "error": f"cv2.imwrite 반환값 {ok!r}"})
        elif not path.is_file():
            failed.append({"path": str(path), "error": "cv2.imwrite가 True를 반환했으나 파일이 없음"})
        else:
            saved.append(str(path))

    if failed:
        return _result(name, FAIL, f"{len(targets)}개 중 {len(failed)}개 저장 실패",
                       output_dir=str(out_dir), saved_files=saved, failed_files=failed)

    return _result(name, PASS, f"{len(saved)}개 저장", output_dir=str(out_dir), saved_files=saved, failed_files=[])


def _image_targets(cv2, np, ctx):
    """저장할 (파일명, 영상) 목록. 합성 마커가 만들어지지 않았으면 빈 목록."""
    if "canvas" not in ctx:
        return []

    def overlay(gray, corners_list):
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        for c in corners_list:
            cv2.polylines(img, [np.round(c).astype(np.int32).reshape(-1, 1, 2)], True, (0, 200, 0), 2)

            for i, (x, y) in enumerate(c):
                cv2.circle(img, (int(round(x)), int(round(y))), 5, (0, 0, 255), -1)
                cv2.putText(img, str(i), (int(round(x)) + 6, int(round(y)) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 0, 0), 2)

        return img

    targets = [("marker.png", ctx["marker"]), ("canvas_0deg.png", ctx["canvas"])]

    for deg, (canvas, corners, _) in ctx.get("rotations", {}).items():
        targets.append((f"overlay_{deg:03d}deg.png", overlay(canvas, corners)))

    return targets


# ── 실행 ─────────────────────────────────────────────────────────────────────────

def run(cv2, np, corner_tol_px: float, output_dir: str | None = None):
    """필수 시험 6개(와 요청 시 이미지 저장)를 실행해 보고서 dict를 돌려준다. `exit_code` 키 포함."""
    env = environment(cv2, np)
    backend = ArucoBackend(cv2)
    ctx: dict = {}
    results = [
        test_api_selection(backend),
        test_synthetic_detection(np, backend, ctx),
        test_blank_negative(np, backend),
        test_rotation_correspondence(np, backend, ctx, corner_tol_px),
        test_pnp_synthetic(cv2, np),
        test_detect_to_pnp(cv2, np, backend, ctx),
    ]
    report = {"environment": env,
              "selected_api": {"dictionary": getattr(backend, "dictionary_api", None),
                               "generate": backend.generate_api, "detect": backend.detect_api},
              "results": results, "saved_files": [],
              "limits": "실제 촬영 정확도, 카메라 보정, B의 최종 후보 선택·모호성 판정, ROS 영상 변환, "
                        "GUI·TCP·로봇 통합은 이 점검으로 검증되지 않음"}
    judged = list(results)

    if output_dir:  # 요청했을 때만 저장 결과를 추가한다
        save = save_images(cv2, np, ctx, Path(output_dir))
        report["image_save"] = save
        report["saved_files"] = save["saved_files"]
        judged.append(save)

    report["exit_code"] = overall_exit_code(judged)
    return report


def _print_human(report):
    env = report["environment"]
    print("== 환경")
    print(f"Python {env['python_version']}  ({env['python_executable']})")
    print(f"OpenCV {env['opencv_version']}  ({env['cv2_file']})")
    print(f"NumPy  {env['numpy_version']}")
    print("== API 존재 여부")

    for k, v in env["api"].items():
        print(f"  {'있음' if v else '없음':4} {k}")

    sa = report["selected_api"]
    print("== 선택한 API")
    print(f"  사전: {sa['dictionary']}\n  생성: {sa['generate']}\n  검출: {sa['detect']}")
    print("== 시험 결과")

    for r in report["results"]:
        print(f"[{r['status']:10}] {r['name']}: {r['reason']}")

        if r["name"] == "rotation_correspondence":
            for row in r.get("rotations", []):
                if "max_corner_diff_px" in row:
                    print(f"      {row['deg_ccw']:3d}°: 최대 차이 {row['max_corner_diff_px']} px "
                          f"(인덱스를 민 대응이면 최소 {row['min_diff_if_index_shifted_px']} px)")

        if r["name"] in ("pnp_synthetic", "detect_to_pnp"):
            for i, c in enumerate(r.get("candidates", [])):
                extra = ""

                if "position_error_m" in c:
                    extra = (f", 위치 오차 {c['position_error_m']} m, 회전 오차 {c['rotation_error_deg']}°, "
                             f"허용치 안 {c['within_synthetic_tolerance']}")

                print(f"      후보 {i}: t={c['t_C_M_m']} m, 재투영 {c['reprojection_error_px']} px, "
                      f"카메라 앞 {c['in_front']}{extra}")
                print(f"              R={c['R_C_M']}")

    if "image_save" in report:
        s = report["image_save"]
        print("== 이미지 저장 (선택 기능, 필수 시험과 별도)")
        print(f"[{s['status']:10}] {s['name']}: {s['reason']}")

        for f in s["saved_files"]:
            print(f"      저장: {f}")

        for f in s["failed_files"]:
            print(f"      실패: {f['path']} — {f['error']}")

    print(f"== 종료 코드 {report['exit_code']}  (0 전체 PASS, 1 FAIL 있음, 2 UNVERIFIED 있음)")
    print(f"한계: {report['limits']}")


def main(argv=None) -> int:
    """CLI 진입점. 종료 코드(0 PASS, 1 FAIL 있음, 2 UNVERIFIED 있음)를 돌려준다."""
    ap = argparse.ArgumentParser(description="OpenCV ArUco 검출·IPPE_SQUARE PnP 동작 점검 (합성 입력)")
    ap.add_argument("--json", action="store_true", help="기계가 읽을 수 있는 JSON으로 출력")
    ap.add_argument("--output-dir", help="합성 이미지·오버레이를 저장할 디렉터리 (지정할 때만 저장)")
    ap.add_argument("--corner-tolerance-px", type=float, default=DEFAULT_CORNER_TOL_PX,
                    help=f"회전 대응 시험의 픽셀 허용치, 시험용 값 (기본 {DEFAULT_CORNER_TOL_PX})")
    args = ap.parse_args(argv)

    if not (args.corner_tolerance_px >= 0 and math.isfinite(args.corner_tolerance_px)):
        ap.error("--corner-tolerance-px는 0 이상의 유한한 값이어야 함")

    missing = []

    try:
        import numpy as np
    except ImportError as e:
        np, missing = None, missing + [f"numpy: {e}"]

    try:
        import cv2
    except ImportError as e:
        cv2, missing = None, missing + [f"cv2: {e}"]

    if missing:
        report = {"environment": {"python_version": platform.python_version(), "python_executable": sys.executable},
                  "results": [_result("import", UNVERIFIED, "; ".join(missing))], "exit_code": 2}
        print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else
              f"[UNVERIFIED] import: {'; '.join(missing)}\n== 종료 코드 2")
        return 2

    report = run(cv2, np, args.corner_tolerance_px, args.output_dir)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        _print_human(report)

    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
