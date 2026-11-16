"""B 입구 — A 파이프라인이 부를 B 함수 하나 (`BStage.process`).

A는 관측 하나마다 이렇게 부른다 (A 협업 요청 B-07, B 상세설계 §16):

    stage = BStage.from_config(root, camera_profile, tf_lookup)
    out = stage.process(observation, frame)

A 실행기(`vision/pipeline_runner.run`)의 B 자리 `b_consumer(frame, obs)`에는 순서가 같은 `stage.consume`을 넣는다:

    run(..., b_consumer=stage.consume)      # A 검사(b_gate)를 통과한 유효 관측만 들어온다
    stage.on_output = c_queue.put           # (선택) B 출력을 C 쪽으로 넘길 곳. 없으면 stage.last 에만 남긴다
    out.result        # WorkPlanarResult (m, B 내부)
    out.result_dict   # C·D 표시용 결과 (mm·deg, DB-01)
    out.goal          # C 목표 사전 (PICK_PLACE / PASS_THROUGH / SKIP, DB-02)
    stage.render_center(frame.image, [out])   # center 주석 영상 (같은 원본 프레임 위)

카메라 위치(`T_W_C`)는 `tf_lookup(frame)`이 **촬영 시각**의 TF로 준다.
- ROS 안에서는 `RosTfLookup(buffer)` (tf2 Buffer를 넘김)
- 시험·저장 영상에서는 `StaticTfLookup(T_W_C)`
TF를 못 읽으면 계산하지 않고 `SKIP`(`TF_UNAVAILABLE`, 책임 B)으로 둔다.

이 모듈은 ROS를 import하지 않는다. ROS 메시지는 같은 이름의 속성만 읽는다.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from eyetoarm.calibration.profile import CameraProfile
from eyetoarm.transforms.grasp import build_sort_goal
from eyetoarm.transforms.overlay import draw_guides, draw_results
from eyetoarm.transforms.planar_transform import make_T, quat_to_rot
from eyetoarm.transforms.work_result import (
    UNRESOLVED, WorkPlanarResult, from_observation, load_specs, to_output_dict,
)

TfLookup = Callable[[object], "np.ndarray | None"]


# ---------- 카메라 위치(TF) 읽기 ----------
def transform_to_matrix(transform) -> np.ndarray:
    """ROS `geometry_msgs/Transform`(translation, rotation) → 4×4 `T_W_C`."""
    t, q = transform.translation, transform.rotation

    return make_T(quat_to_rot(q.x, q.y, q.z, q.w), [t.x, t.y, t.z])


class StaticTfLookup:
    """고정된 `T_W_C`를 돌려준다. 시험·합성 영상·고정 카메라용."""

    def __init__(self, T_W_C: np.ndarray) -> None:
        self.T_W_C = np.asarray(T_W_C, dtype=np.float64)

    def __call__(self, frame) -> np.ndarray:
        return self.T_W_C


class RosTfLookup:
    """tf2 `Buffer`에서 촬영 시각(`frame.ros_stamp_ns`)의 `world` → 광학 프레임을 읽는다.

    링크 카메라는 팔과 같이 움직이므로 '최신' TF가 아니라 **촬영 시각** TF를 쓴다.
    그 시각 TF가 없으면 None (B는 SKIP).
    """

    def __init__(self, buffer, world_frame: str = "world", timeout_s: float = 0.05) -> None:
        self.buffer = buffer
        self.world_frame = world_frame
        self.timeout_s = timeout_s

    def __call__(self, frame) -> np.ndarray | None:
        from rclpy.duration import Duration   # ROS 안에서만 import
        from rclpy.time import Time

        if frame.ros_stamp_ns is None or not frame.optical_frame:
            return None

        try:
            tf = self.buffer.lookup_transform(self.world_frame, frame.optical_frame,
                                              Time(nanoseconds=frame.ros_stamp_ns),
                                              timeout=Duration(seconds=self.timeout_s))
        except Exception:  # noqa: BLE001 - TF 없음·시각 밖
            return None

        return transform_to_matrix(tf.transform)


# ---------- B 입구 ----------
@dataclass(frozen=True)
class BOutput:
    """관측 하나에 대한 B 출력 묶음."""

    result: WorkPlanarResult
    result_dict: dict
    goal: dict
    T_W_C: np.ndarray | None


class BStage:
    def __init__(self, camera: CameraProfile, tf_lookup: TfLookup, frames: dict, specs: dict) -> None:
        self.camera = camera
        self.tf_lookup = tf_lookup
        self.frames = frames
        self.specs = specs
        self.sort_job_id: str | None = None    # C가 START 수락 때 발급 → 파이프라인이 넣어 줌
        self.run_epoch: int | None = None      # C 실행 세대 (C 상세설계 §16.5.2)
        self.on_output: Callable[[BOutput], None] | None = None   # B 출력 받을 곳 (C 대기열 등)
        self.last: BOutput | None = None

    @classmethod
    def from_config(cls, root: str | Path, camera: CameraProfile, tf_lookup: TfLookup) -> "BStage":
        """저장소 맨 위 폴더(`root`)의 `config/robot/frames.yaml`·`config/calibration/part_specs.json`을 읽는다."""
        root = Path(root)

        with open(root / "config/robot/frames.yaml", encoding="utf-8") as f:
            frames = yaml.safe_load(f)

        return cls(camera, tf_lookup, frames, load_specs(root / "config/calibration/part_specs.json"))

    def process(self, obs, frame) -> BOutput:
        """A 관측 + 그 프레임 → B 결과·표시 사전·C 목표."""
        T_W_C = self.tf_lookup(frame) if obs.valid else None

        if obs.valid and T_W_C is None:
            result = self._failed(obs, "TF_UNAVAILABLE", "B")
        else:
            result = from_observation(obs, frame, self.camera, T_W_C if T_W_C is not None else np.eye(4),
                                      self.specs)

        goal = build_sort_goal(result, source_id=frame.source_id, stamp_ns=frame.ros_stamp_ns,
                               frames=self.frames, specs=self.specs)
        goal.update({                                   # C 상세설계 「goal 필드 (안)」의 연결 키
            "goal_id": str(uuid.uuid4()),
            "sort_job_id": self.sort_job_id,
            "run_epoch": self.run_epoch,
            "run_id": obs.run_id,
            "capture_generation": obs.capture_generation,
            "generation": obs.generation,
            "job_seq": obs.job_seq,
            "timestamp_ms": frame.timestamp_ms,
            "timestamp_base": str(getattr(frame.timestamp_base, "value", frame.timestamp_base)),
        })

        return BOutput(result, to_output_dict(result, frame.source_id), goal, T_W_C)

    def consume(self, frame, obs) -> BOutput:
        """A 실행기 `b_consumer(frame, obs)` 자리용. `process`를 부르고 결과를 `last`·`on_output`으로 넘긴다."""
        out = self.process(obs, frame)
        self.last = out

        if self.on_output is not None:
            self.on_output(out)

        return out

    def render_center(self, image: np.ndarray, outputs: list[BOutput], guides: bool = True) -> np.ndarray:
        """center 주석 영상: 원본 프레임 복사본 위에 B 결과(+작업 좌표 안내선)를 그린다 (DB-01).

        원본 배열은 바꾸지 않는다. 같은 프레임의 결과만 넘긴다(같은 `T_W_C`).
        """
        img = np.ascontiguousarray(image).copy()
        drawable = [o for o in outputs if o.T_W_C is not None]

        if not drawable:
            return img

        T_W_C = drawable[0].T_W_C

        if guides:
            draw_guides(img, self.camera, T_W_C, self.frames)

        draw_results(img, self.camera, T_W_C, [o.result for o in drawable if len(o.result.contour_w_m)])

        return img

    def _failed(self, obs, reason: str, owner: str) -> WorkPlanarResult:
        return WorkPlanarResult(
            center_w_m=(float("nan"), float("nan")), plane_z_m=float(self.specs["plane_z"]),
            theta_deg=None, theta_valid=False, spec=UNRESOLVED, class_name=obs.class_name,
            measured_length_m=0.0, measured_width_m=0.0, aspect_ratio=0.0, area_w_m2=0.0,
            contour_w_m=np.zeros((0, 2)), valid=False, reason=reason, reason_owner=owner,
            calibration_id=self.camera.calibration_id, camera_frame=self.camera.frame_id, compute_ms=0.0,
            part_track_id=getattr(obs, "part_track_id", None), image_seq=obs.image_seq)


def goal_to_json(goal: dict) -> str:
    """C 목표 사전 → 한 줄 JSON (로그·TCP 시험용)."""
    return json.dumps(goal, ensure_ascii=False, separators=(",", ":"))
