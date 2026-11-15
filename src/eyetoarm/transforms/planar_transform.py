"""B 위치 변환 — 사진 픽셀 ↔ 작업 좌표 W(벨트 위 평면).

근거: B 상세설계 §6.2(원본 픽셀 → 왜곡 보정 1회 → W), §7(높이별 Homography), B2 실습.

표기: `T_A_B` = A 기준에서 본 B의 4×4 변환. `T_W_C`는 W에서 본 카메라 광학 프레임이다.
팔 카메라는 움직이므로 `T_W_C`는 **사진 찍은 시각의 TF**(`world` → 광학 프레임)를 넣는다.

- 평면 높이 h(m)는 W의 Z값이다. 부품 윤곽은 `part_plane_z`(벨트 윗면 + 5 mm)에서 본다.
- 입력 픽셀은 원본(raw) 픽셀이다. 왜곡 보정은 여기서 한 번만 한다.
- ROS를 import하지 않는다.
"""
from __future__ import annotations

import cv2
import numpy as np


def rot_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """RPY(rad, 고정축 X→Y→Z) → 3×3 회전행렬. ROS·URDF와 같은 규약."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


def quat_to_rot(x: float, y: float, z: float, w: float) -> np.ndarray:
    """쿼터니언(xyzw, TF 순서) → 3×3 회전행렬."""
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n

    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def make_T(R: np.ndarray, t) -> np.ndarray:
    """회전 R(3×3)과 이동 t(3)로 4×4 변환을 만든다."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)

    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    """4×4 강체 변환의 역 (Rᵀ, -Rᵀt)."""
    R, t = T[:3, :3], T[:3, 3]

    return make_T(R.T, -R.T @ t)


def look_at(cam_pos, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """카메라 위치·바라볼 점으로 `T_W_C`(광학 프레임: Z 앞·X 오른쪽·Y 아래)를 만든다. 시험·데모용."""
    cam_pos = np.asarray(cam_pos, dtype=np.float64)
    z = np.asarray(target, dtype=np.float64) - cam_pos
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, dtype=np.float64))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    return make_T(np.column_stack([x, y, z]), cam_pos)


def homography_w_to_image(K: np.ndarray, T_W_C: np.ndarray, plane_z: float) -> np.ndarray:
    """평면 Z = plane_z 위의 점 (X, Y) → 왜곡 없는 픽셀 (u, v) Homography.

    H = K [r1 r2 (plane_z·r3 + t)],  여기서 [R | t] = T_C_W.
    """
    T_C_W = inv_T(T_W_C)
    R, t = T_C_W[:3, :3], T_C_W[:3, 3]
    H = K @ np.column_stack([R[:, 0], R[:, 1], plane_z * R[:, 2] + t])

    return H / H[2, 2]


def undistort_pixels(px, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """원본 픽셀 (N, 2) → 왜곡을 뺀 픽셀 (N, 2). dist가 0이면 그대로."""
    pts = np.asarray(px, dtype=np.float64).reshape(-1, 1, 2)

    if not np.any(dist):
        return pts.reshape(-1, 2)

    return cv2.undistortPoints(pts, K, dist, P=K).reshape(-1, 2)


def pixels_to_w(px, K: np.ndarray, dist: np.ndarray, T_W_C: np.ndarray, plane_z: float) -> np.ndarray:
    """원본 픽셀 (N, 2) → W 평면 위의 점 (N, 2) [m]. 왜곡 보정은 여기서 한 번만 한다."""
    und = undistort_pixels(px, K, dist)
    H_inv = np.linalg.inv(homography_w_to_image(K, T_W_C, plane_z))
    out = cv2.perspectiveTransform(und.reshape(-1, 1, 2), H_inv)

    return out.reshape(-1, 2)


def w_to_image(points_w, K: np.ndarray, dist: np.ndarray, T_W_C: np.ndarray) -> np.ndarray:
    """W의 3D 점 (N, 3) [m] → 원본 픽셀 (N, 2). 중심점 주석 영상·안내선 그리기용.

    카메라 뒤(Z ≤ 0)에 있는 점은 NaN으로 돌려준다.
    """
    P = np.asarray(points_w, dtype=np.float64).reshape(-1, 3)
    T_C_W = inv_T(T_W_C)
    Pc = (T_C_W[:3, :3] @ P.T).T + T_C_W[:3, 3]
    rvec, _ = cv2.Rodrigues(T_C_W[:3, :3])
    px, _ = cv2.projectPoints(P.reshape(-1, 1, 3), rvec, T_C_W[:3, 3].reshape(3, 1), K, dist)
    px = px.reshape(-1, 2)
    px[Pc[:, 2] <= 1e-6] = np.nan

    return px


def tf_matrix_from_echo(text: str) -> np.ndarray:
    """`ros2 run tf2_ros tf2_echo` 출력의 `- Matrix:` 4줄을 읽어 4×4로 돌려준다(마지막 블록)."""
    lines = text.strip().splitlines()
    idx = max(i for i, s in enumerate(lines) if s.strip().startswith("- Matrix"))
    rows = [list(map(float, lines[idx + k].split())) for k in range(1, 5)]

    return np.array(rows)
