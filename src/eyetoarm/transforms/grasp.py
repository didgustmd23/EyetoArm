"""B 분류 목표 — B 결과 하나를 C에게 넘길 목표(집기·놓기)로 바꾼다.

근거: B 협업 요청 §0 확정 표(C 회신), §6 DB-02 결정(2026-11-16).

결정 요약
1. 단위: 위치 mm, 방향 쿼터니언 [x, y, z, w].
2. 기준 좌표: `base_link` (= 월드, 로봇 바닥이 원점).
3. 움직이는 벨트: B는 **찍은 위치 + 찍은 시각(SIM_TIME)**만 준다. 잡는 시점 예측은 C.
4. 잡는 방향: 집게 가운데 프레임 G — z = 아래, x = 손가락이 닫히는 방향.
   닫히는 방향 = 볼트 축(θ)에 직각 = θ + 90°. 180° 다른 후보 2개, C가 고른다.
5. 잡는 점: G 기준(볼트 무게중심, 높이 = 부품 평면). G → tool0 오프셋은 C가 적용.
6. 너트: 집지 않는다 → `PASS_THROUGH` (벨트 끝 너트 상자로 흘러감).
7. 놓는 곳: 상자 가운데, 상자 윗면 위 10 cm, 집게 벌림은 규격별(20 mm).

관절 각도(IK)·경로·충돌 확인은 C가 MoveIt으로 한다 (IK-2). ROS를 import하지 않는다.
"""
from __future__ import annotations

import math

from eyetoarm.transforms.work_result import UNRESOLVED, WorkPlanarResult


def down_quat_xyzw(closing_yaw_rad: float) -> list[float]:
    """G 프레임 방향: z가 아래(-Z), x가 `closing_yaw` 방향(수평)인 쿼터니언 [x, y, z, w].

    회전 = Rz(ψ) · Rx(π). 곱을 풀면 [cos(ψ/2), sin(ψ/2), 0, 0].
    """
    h = closing_yaw_rad / 2.0

    return [round(math.cos(h), 6), round(math.sin(h), 6), 0.0, 0.0]


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def build_sort_goal(r: WorkPlanarResult, *, source_id: str, stamp_ns: int | None,
                    frames: dict, specs: dict) -> dict:
    """B 결과 → C 목표 사전 (DB-02 형식).

    `kind`
    - `PICK_PLACE`: 볼트(BOLT_A/B). `grasp_candidates` 2개 + `place`.
    - `PASS_THROUGH`: 너트. 잡지 않음.
    - `SKIP`: 무효·판정 불가·집는 구간 밖(`OUT_OF_REACH`). 잡지 않음 (`reason` 포함).
    """
    mm = lambda v: round(v * 1000.0, 1)
    goal = {
        "kind": "SKIP",
        "frame_id": frames["world"].get("output_frame_id", "base_link"),
        "source_id": source_id,
        "image_seq": r.image_seq,
        "part_track_id": r.part_track_id,
        "capture_stamp": {"base": "SIM_TIME", "ns": stamp_ns},
        "calibration_id": r.calibration_id,
        "spec": r.spec,
        "reason": r.reason,
    }

    if not r.valid or r.spec == UNRESOLVED:
        goal["reason"] = r.reason or "UNRESOLVED"
        return goal

    spec = specs["specs"][r.spec]
    observed = [mm(r.center_w_m[0]), mm(r.center_w_m[1]), mm(r.plane_z_m)]
    goal["observed_position_mm"] = observed

    if spec.get("action") == "PASS_THROUGH":
        goal["kind"] = "PASS_THROUGH"
        return goal

    if not r.theta_valid or r.theta_deg is None:
        goal["reason"] = "THETA_INVALID"
        return goal

    zone = frames.get("pick_zone")

    if zone and not (zone["y_min"] <= r.center_w_m[1] <= zone["y_max"]):
        goal["reason"] = "OUT_OF_REACH"          # 집는 구간 밖 (B 결정 2026-11-17)
        return goal

    closing = _wrap_deg(r.theta_deg + 90.0)
    candidates = []
    gx, gy = r.grasp_w_m if getattr(r, "grasp_w_m", None) else r.center_w_m   # 볼트 두 끝점 입력이면 축 가운데 Pb
    # 집는 높이 = 축 중심 높이 + 종류별 grasp_z_offset (part_specs.json, 지금은 모두 0. 2026-11-24 B 결정: 육각 M3x20
    # −2 mm 보정은 C 실행기 GRASP_HEIGHT_OFFSET 한 곳에만 둔다. 보고 위치 observed_position_mm 높이는 그대로 plane_z)
    grasp_z = r.plane_z_m + float(spec.get("grasp_z_offset", 0.0))
    grasp_pos = [mm(gx), mm(gy), mm(grasp_z)]

    for yaw in (closing, _wrap_deg(closing + 180.0)):
        candidates.append({"position_mm": grasp_pos, "orientation_xyzw": down_quat_xyzw(math.radians(yaw)),
                           "closing_yaw_deg": round(yaw, 1)})

    bin_ = frames["bins"][spec["bin"]]
    place_z = bin_["size"][2] + frames["place"]["height_above_bin"]
    place_yaw = math.degrees(bin_["yaw"])

    goal.update({
        "kind": "PICK_PLACE",
        "grasp_candidates": candidates,
        "gripper_opening_mm": mm(spec["grasp_opening"]),
        "approach_height_mm": mm(frames.get("heights", {}).get("approach", r.plane_z_m + 0.10)),
        "lift_height_mm": mm(frames.get("heights", {}).get("lift", 0.25)),
        "place": {
            "bin": spec["bin"],
            "position_mm": [mm(bin_["center"][0]), mm(bin_["center"][1]), mm(place_z)],
            "orientation_xyzw": down_quat_xyzw(math.radians(place_yaw)),
        },
    })

    return goal
