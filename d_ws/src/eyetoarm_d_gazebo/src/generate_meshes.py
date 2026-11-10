#!/usr/bin/env python3
"""컨베이어 벨트와 상자 메시(binary STL, 단위 m)를 만든다. 외부 의존성 없음.

치수는 B가 확정하기 전의 임시값이다(D 상세설계 §A.11.1 5). 값이 바뀌면 이 파일의
상수만 고치고 다시 실행한다:
    python3 src/generate_meshes.py            # 패키지 루트에서
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "models"

# 컨베이어(임시값, m). 벨트는 X축 방향으로 진행한다
BELT_LENGTH = 1.20       # 롤러 중심 사이
BELT_WIDTH = 0.20
ROLLER_R = 0.030
BELT_T = 0.004           # 벨트 두께
BELT_TOP_Z = 0.100       # 바닥에서 벨트 윗면까지
RAIL_T, RAIL_H = 0.020, 0.060
LEG = 0.040

# 상자(임시값, m)
BIN_X, BIN_Y, BIN_Z, BIN_WALL = 0.160, 0.120, 0.080, 0.004

Tri = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


def box(cx: float, cy: float, cz: float, sx: float, sy: float, sz: float) -> list[Tri]:
    x0, x1, y0, y1, z0, z1 = cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2, cz - sz / 2, cz + sz / 2
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return [t for a, b, c, d in faces for t in ((v[a], v[b], v[c]), (v[a], v[c], v[d]))]


def cylinder_y(cx: float, cz: float, r: float, y0: float, y1: float, n: int = 32) -> list[Tri]:
    """Y축 방향 원기둥(롤러)."""
    tris: list[Tri] = []
    ring = [(cx + r * math.cos(2 * math.pi * i / n), cz + r * math.sin(2 * math.pi * i / n)) for i in range(n)]
    for i in range(n):
        (xa, za), (xb, zb) = ring[i], ring[(i + 1) % n]
        tris += [((xa, y0, za), (xb, y0, zb), (xb, y1, zb)), ((xa, y0, za), (xb, y1, zb), (xa, y1, za)),
                 ((cx, y0, cz), (xb, y0, zb), (xa, y0, za)), ((cx, y1, cz), (xa, y1, za), (xb, y1, zb))]
    return tris


def band(length: float, r_in: float, t: float, width: float, cz: float, n_arc: int = 24) -> list[Tri]:
    """두 롤러를 감싸는 벨트 고리. 경기장 모양 경로를 폭 방향(Y)으로 밀어 낸다."""
    def path(r: float) -> list[tuple[float, float]]:
        pts = []
        for i in range(n_arc + 1):  # 오른쪽 반원(아래 → 위)
            a = -math.pi / 2 + math.pi * i / n_arc
            pts.append((length / 2 + r * math.cos(a), cz + r * math.sin(a)))
        for i in range(n_arc + 1):  # 왼쪽 반원(위 → 아래)
            a = math.pi / 2 + math.pi * i / n_arc
            pts.append((-length / 2 + r * math.cos(a), cz + r * math.sin(a)))
        return pts

    inner, outer = path(r_in), path(r_in + t)
    y0, y1 = -width / 2, width / 2
    tris: list[Tri] = []
    m = len(outer)
    for i in range(m):
        j = (i + 1) % m
        (oa, ob), (ia, ib) = (outer[i], outer[j]), (inner[i], inner[j])
        tris += [((oa[0], y0, oa[1]), (ob[0], y0, ob[1]), (ob[0], y1, ob[1])), ((oa[0], y0, oa[1]), (ob[0], y1, ob[1]), (oa[0], y1, oa[1])),
                 ((ia[0], y0, ia[1]), (ia[0], y1, ia[1]), (ib[0], y1, ib[1])), ((ia[0], y0, ia[1]), (ib[0], y1, ib[1]), (ib[0], y0, ib[1])),
                 ((oa[0], y0, oa[1]), (ia[0], y0, ia[1]), (ib[0], y0, ib[1])), ((oa[0], y0, oa[1]), (ib[0], y0, ib[1]), (ob[0], y0, ob[1])),
                 ((oa[0], y1, oa[1]), (ob[0], y1, ob[1]), (ib[0], y1, ib[1])), ((oa[0], y1, oa[1]), (ib[0], y1, ib[1]), (ia[0], y1, ia[1]))]
    return tris


def write_stl(path: Path, tris: list[Tri]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = [b"EyetoArm D generated mesh (m)".ljust(80, b" "), struct.pack("<I", len(tris))]
    for a, b, c in tris:
        u = [b[k] - a[k] for k in range(3)]
        w = [c[k] - a[k] for k in range(3)]
        nx, ny, nz = u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]
        norm = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
        out.append(struct.pack("<12fH", nx / norm, ny / norm, nz / norm, *a, *b, *c, 0))
    path.write_bytes(b"".join(out))


def conveyor() -> dict[str, list[Tri]]:
    roller_z = BELT_TOP_Z - BELT_T - ROLLER_R
    half_w = BELT_WIDTH / 2
    rail_y = half_w + 0.010 + RAIL_T / 2
    frame = []
    for s in (-1, 1):  # 옆 레일과 다리 4개, 하부 가로대
        frame += box(0, s * rail_y, roller_z, BELT_LENGTH + 2 * ROLLER_R + 0.04, RAIL_T, RAIL_H)
        for xs in (-1, 1):
            leg_h = roller_z - RAIL_H / 2
            frame += box(xs * (BELT_LENGTH / 2 - 0.05), s * rail_y, leg_h / 2, LEG, LEG, leg_h)
    for xs in (-1, 1):
        frame += box(xs * (BELT_LENGTH / 2 - 0.05), 0, 0.03, LEG, 2 * rail_y, 0.02)
    rollers = []
    for xs in (-1, 1):
        rollers += cylinder_y(xs * BELT_LENGTH / 2, roller_z, ROLLER_R, -half_w - 0.010, half_w + 0.010)
    return {"frame": frame, "rollers": rollers, "belt": band(BELT_LENGTH, ROLLER_R, BELT_T, BELT_WIDTH, roller_z)}


def open_bin() -> list[Tri]:
    w = BIN_WALL
    return (box(0, 0, w / 2, BIN_X, BIN_Y, w)
            + box(0, (BIN_Y - w) / 2, BIN_Z / 2, BIN_X, w, BIN_Z) + box(0, -(BIN_Y - w) / 2, BIN_Z / 2, BIN_X, w, BIN_Z)
            + box((BIN_X - w) / 2, 0, BIN_Z / 2, w, BIN_Y, BIN_Z) + box(-(BIN_X - w) / 2, 0, BIN_Z / 2, w, BIN_Y, BIN_Z))


def main() -> None:
    for name, tris in conveyor().items():
        write_stl(ROOT / "conveyor_belt" / "meshes" / f"{name}.stl", tris)
    write_stl(ROOT / "sort_bin" / "meshes" / "bin.stl", open_bin())
    # 자체 확인: 벨트 윗면 높이가 설정값과 일치
    top = max(v[2] for t in conveyor()["belt"] for v in t)
    assert abs(top - BELT_TOP_Z) < 1e-9, top
    print("meshes written under", ROOT)


if __name__ == "__main__":
    main()
