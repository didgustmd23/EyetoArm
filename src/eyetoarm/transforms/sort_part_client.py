"""B → C 분류 주문 보내기 — ROS 액션 `/sort_part` 클라이언트 (B 담당).

하는 일
1. B 출력(`BOutput.goal` 사전)을 `SortPart` 주문서(Goal)로 바꾼다  → `goal_to_msg`
2. `/sort_part`로 C에게 보낸다 (C 서버가 없으면 기다리지 않고 건너뜀)
3. C가 보내는 진행 상황(`stage`)과 결과(`status`·`reason`)를 받아 기록한다
4. 결과가 오면 `BStage.release(part_track_id)`로 같은 부품 다시 보내기를 허용한다

연결 (A 실행기와 같은 프로세스, C 답 C-1·C-6):
    client = SortPartClient(node, stage)
    stage.on_output = client.send

이름: 액션 `/sort_part`, 양식 `eyetoarm_interfaces/action/SortPart` (B·C 합의, 혼자 바꾸지 않음)
"""
from __future__ import annotations

import threading
import uuid as _uuid
from collections import Counter
from typing import Any

ACTION_NAME = "/sort_part"
SEND_KINDS = ("PICK_PLACE", "PASS_THROUGH")   # SKIP은 보내지 않음


def _int(v: Any, default: int = -1) -> int:
    return default if v is None else int(v)


def goal_to_msg(goal: dict, goal_cls, candidate_cls):
    """B 목표 사전 → SortPart.Goal. ROS 없이도 시험할 수 있게 클래스를 받아 쓴다."""
    m = goal_cls()
    m.goal_id = str(goal.get("goal_id") or "")
    m.goal_kind = goal["kind"]
    m.sort_job_id = str(goal.get("sort_job_id") or "")
    m.run_epoch = _int(goal.get("run_epoch"), 0)
    m.part_track_id = str(goal.get("part_track_id") or "")
    m.part_class = str(goal.get("spec") or "")
    m.source_id = str(goal.get("source_id") or "")
    m.image_seq = _int(goal.get("image_seq"))
    m.capture_generation = _int(goal.get("capture_generation"))
    m.generation = _int(goal.get("generation"))
    m.job_seq = _int(goal.get("job_seq"))
    stamp = goal.get("capture_stamp_ns") or (goal.get("capture_stamp") or {}).get("ns")
    m.capture_stamp_ns = _int(stamp, 0)
    m.timestamp_base = str(goal.get("timestamp_base") or "SIM_TIME")
    m.frame_id = str(goal.get("frame_id") or "base_link")
    m.observed_position_mm = [float(v) for v in goal.get("observed_position_mm", [0.0, 0.0, 0.0])]

    cands = []
    for c in goal.get("grasp_candidates", []):
        gc = candidate_cls()
        gc.position_mm = [float(v) for v in c["position_mm"]]
        gc.orientation_xyzw = [float(v) for v in c["orientation_xyzw"]]
        gc.closing_yaw_deg = float(c["closing_yaw_deg"])
        cands.append(gc)
    m.grasp_pose_candidates = cands

    m.opening_width_mm = float(goal.get("gripper_opening_mm") or 0.0)
    m.approach_height_mm = float(goal.get("approach_height_mm") or 0.0)
    m.lift_height_mm = float(goal.get("lift_height_mm") or 0.0)
    place = goal.get("place") or {}
    m.place_bin = str(place.get("bin") or "")
    m.place_position_mm = [float(v) for v in place.get("position_mm", [0.0, 0.0, 0.0])]
    m.place_orientation_xyzw = [float(v) for v in place.get("orientation_xyzw", [0.0, 0.0, 0.0, 1.0])]
    m.calibration_id = str(goal.get("calibration_id") or "")
    m.valid_until_ns = _int(goal.get("valid_until_ns"), 0)   # 0 = 아직 안 정함 (C-9)
    return m


class SortPartClient:
    """`stage.on_output = client.send` 로 붙인다. A 실행기 스레드에서 불려도 막히지 않는다.

    같은 종류(`PICK_PLACE` / `PASS_THROUGH`)의 목표가 C 결과를 기다리는 중이면 새 목표를 보내지 않는다(`busy`).
    팔이 1대라 집기는 한 번에 하나(B 상세설계 §16.3 사이클). `part_track_id`(A-6)가 없어도 같은 볼트를 연달아 보내지 않는다.
    """

    def __init__(self, node, stage=None, action_name: str = ACTION_NAME):
        from rclpy.action import ActionClient
        from eyetoarm_interfaces.action import SortPart
        from eyetoarm_interfaces.msg import GraspCandidate

        self._SortPart, self._Cand = SortPart, GraspCandidate
        self.node, self.stage = node, stage
        self.client = ActionClient(node, SortPart, action_name)
        self.stats: Counter = Counter()          # sent / no_server / rejected / SUCCEEDED / FAILED ...
        self.inflight: dict[str, dict] = {}      # goal_id → {part_track_id, stage, status}
        self._lock = threading.Lock()

    # ---------- 보내기 ----------
    def send(self, out) -> bool:
        goal = out.goal if hasattr(out, "goal") else out
        kind = goal.get("kind")
        if kind not in SEND_KINDS:
            self.stats["skip"] += 1
            return False
        if not self.client.server_is_ready():        # C 서버 없음 → 기다리지 않음
            self.stats["no_server"] += 1
            self._release(goal)
            return False

        msg = goal_to_msg(goal, self._SortPart.Goal, self._Cand)
        gid = msg.goal_id or str(_uuid.uuid4())
        msg.goal_id = gid
        with self._lock:
            busy = any(v["kind"] == kind for v in self.inflight.values())
            if not busy:
                self.inflight[gid] = {"kind": kind, "part_track_id": goal.get("part_track_id"),
                                      "stage": "SENT", "status": None}
        if busy:                                       # 같은 종류가 결과 대기 중 → 보내지 않음
            self.stats["busy"] += 1
            self._release(goal)                        # B 쪽 기억도 풀어 다음 사진에서 다시 보낼 수 있게
            return False
        self.stats["sent"] += 1

        fut = self.client.send_goal_async(msg, feedback_callback=lambda fb: self._on_feedback(gid, fb),
                                          goal_uuid=_uuid_msg(gid))
        fut.add_done_callback(lambda f: self._on_accepted(gid, goal, f))
        return True

    # ---------- 받기 ----------
    def _on_feedback(self, gid: str, fb) -> None:
        with self._lock:
            if gid in self.inflight:
                self.inflight[gid]["stage"] = fb.feedback.stage

    def _on_accepted(self, gid: str, goal: dict, fut) -> None:
        handle = fut.result()
        if not handle.accepted:
            self.stats["rejected"] += 1
            self._finish(gid, goal, "REJECTED", "")
            return
        handle.get_result_async().add_done_callback(lambda f: self._on_result(gid, goal, f))

    def _on_result(self, gid: str, goal: dict, fut) -> None:
        res = fut.result().result
        self.stats[res.status or "UNKNOWN"] += 1
        self._finish(gid, goal, res.status, res.reason)

    def _finish(self, gid: str, goal: dict, status: str, reason: str) -> None:
        with self._lock:
            self.inflight.pop(gid, None)
        self.node.get_logger().info(f"[B→C] {goal.get('spec')} {gid[:8]} → {status} {reason}".rstrip())
        self._release(goal)

    def _release(self, goal: dict) -> None:
        tid = goal.get("part_track_id")
        if self.stage is not None and tid:
            self.stage.release(tid)                  # 같은 부품 다시 보낼 수 있게


def _uuid_msg(gid: str):
    """goal_id 문자열 → ROS goal UUID (C-3: goal_id = ROS goal UUID)."""
    from unique_identifier_msgs.msg import UUID
    return UUID(uuid=list(_uuid.UUID(gid).bytes))
