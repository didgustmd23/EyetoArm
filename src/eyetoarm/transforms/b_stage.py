"""B 입구 — A 파이프라인이 부를 B 함수 하나 (`BStage.process`).

A는 관측 하나마다 이렇게 부른다 (A 협업 요청 B-07, B 상세설계 §16):

    stage = BStage.from_config(root, camera_profile, tf_lookup)
    out = stage.process(observation, frame)

A 실행기(`vision/pipeline_runner.run`)의 B 자리 `b_consumer(frame, obs)`에는 순서가 같은 `stage.consume`을 넣는다:

    run(..., b_consumer=stage.consume)      # A 검사(b_gate)를 통과한 유효 관측만 들어온다
    stage.on_output = make_c_handoff(loop, sort_goal_queue)   # (선택) B 목표를 C asyncio 대기열로. 없으면 stage.last 에만 남긴다
    out.result        # WorkPlanarResult (m, B 내부)
    out.result_dict   # C·D 표시용 결과 (mm·deg, DB-01)
    out.goal          # C 목표 사전 (PICK_PLACE / PASS_THROUGH / SKIP, DB-02)
    stage.render_center(frame.image, [out])   # center 주석 영상 (같은 원본 프레임 위)

같은 사진의 볼트 여러 개(A 단계 L, B-09 합의)는 목록 봉투 하나로 받는다:

    rec = stage.process_frame(list_envelope)   # 앞 8개 계산 → 1개 골라 on_output으로, 결과는 JSON 사전
    rec["selected_instance_index"], rec["goal"], rec["results"]   # 후보별 결과에 instance_index·selected

카메라 위치(`T_W_C`)는 `tf_lookup(frame)`이 **촬영 시각**의 TF로 준다.
- ROS 안에서는 `RosTfLookup(buffer)` (tf2 Buffer를 넘김)
- 시험·저장 영상에서는 `StaticTfLookup(T_W_C)`
TF를 못 읽으면 계산하지 않고 `SKIP`(`TF_UNAVAILABLE`, 책임 B)으로 둔다.

이 모듈은 ROS를 import하지 않는다. ROS 메시지는 같은 이름의 속성만 읽는다.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import yaml

from eyetoarm.calibration.profile import CameraProfile
from eyetoarm.transforms.grasp import build_sort_goal
from eyetoarm.transforms.overlay import draw_guides, draw_results
from eyetoarm.transforms.planar_transform import make_T, quat_to_rot, w_to_image
from eyetoarm.transforms.work_result import (
    UNRESOLVED, WorkPlanarResult, from_observation, load_specs, to_output_dict,
)

TfLookup = Callable[[object], "np.ndarray | None"]

# 목록 입구 `process_frame` (A 협업 요청 B-09, B 회신 2026-11-19, A 동의 2026-11-19)
LIST_ENVELOPE_VERSION = "eyetoarm.b_input/2"   # A `vision/list_envelope`와 같은 값 (그 모듈이 이 파일을 import하므로 글자로 둠)
FRAME_OUTPUT_VERSION = "eyetoarm.b_frame_output/1"
MAX_FRAME_CANDIDATES = 8                       # 앞 8개(`instance_index` 순)만 계산, 잘린 수는 기록
TIE_Y_M = 0.005                                # y 차이 5 mm 이내면 같은 것으로 보고 작은 `instance_index`


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
        self.last_frame: list[BOutput] = []    # `process_frame`의 마지막 사진 후보별 결과
        self.compute_budget_ms = 20.0          # B 계산 시간 상한 (A-7). 넘으면 C로 안 보냄(SKIP B_TIMEOUT). TF 대기는 RosTfLookup 0.05 s
        self._pending: dict[object, str] = {}   # part_track_id → 보낸 goal_id (결과 받을 때까지 다시 안 보냄)
        self._pending_epoch: int | None = None

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
        out = self._build(obs, frame, T_W_C)

        if out.result.compute_ms > self.compute_budget_ms and out.goal.get("kind") != "SKIP":
            out.goal["kind"], out.goal["reason"] = "SKIP", "B_TIMEOUT"

        self._dedupe(out.goal)

        return out

    def _build(self, obs, frame, T_W_C: np.ndarray | None) -> BOutput:
        """계산 → C 목표 → 표시 사전. 시간 상한·같은 부품 거르기는 부르는 쪽(`process`·`process_frame`)이 한다."""
        if obs.valid and T_W_C is None:
            result = self._failed(obs, "TF_UNAVAILABLE", "B")
        else:
            result = from_observation(obs, frame, self.camera, T_W_C if T_W_C is not None else np.eye(4),
                                      self.specs)

        goal = build_sort_goal(result, source_id=frame.source_id, stamp_ns=frame.ros_stamp_ns,
                               frames=self.frames, specs=self.specs)
        keys = correlation_keys(obs, frame)              # A 관측 → B → C·D 연결 키 (CB-05, D DB-01)
        goal.update(keys)
        goal.update({"goal_id": str(uuid.uuid4()), "sort_job_id": self.sort_job_id, "run_epoch": self.run_epoch})
        ttl, stamp = self.frames.get("goal_ttl_s"), frame.ros_stamp_ns
        goal["valid_until_ns"] = int(stamp + round(float(ttl) * 1e9)) if ttl and stamp else 0   # C-9 (B 제안 2 s)

        result_dict = to_output_dict(result, frame.source_id)
        result_dict.update(keys)
        result_dict["center_px"] = self._center_px(result, T_W_C)

        return BOutput(result, result_dict, goal, T_W_C)

    def _dedupe(self, goal: dict) -> None:
        """같은 부품(`part_track_id`)은 결과를 받을 때까지 한 번만 PICK_PLACE로 보낸다 (B 결정 2026-11-17).

        `part_track_id`가 없으면(A-6 미정) 거르지 않는다. 영업 회차(`run_epoch`)가 바뀌면 기억을 비운다.
        """
        if self.run_epoch != self._pending_epoch:
            self._pending.clear()
            self._pending_epoch = self.run_epoch

        track = goal.get("part_track_id")

        if goal.get("kind") != "PICK_PLACE" or track is None:
            return

        if track in self._pending:
            goal["kind"] = "SKIP"
            goal["reason"] = "ALREADY_SENT"
            return

        self._pending[track] = goal["goal_id"]

    def release(self, part_track_id) -> None:
        """C 결과(성공·실패·만료·MISSED)를 받으면 부른다. 그 부품을 다시 보낼 수 있게 된다."""
        self._pending.pop(part_track_id, None)

    def _center_px(self, result: WorkPlanarResult, T_W_C: np.ndarray | None) -> list[float] | None:
        """작업 좌표 중심 → 원본 사진 픽셀 [u, v] (D 표시용). 계산 못 하면 None."""
        x, y = result.center_w_m

        if T_W_C is None or not (np.isfinite(x) and np.isfinite(y)):
            return None

        uv = w_to_image(np.array([[x, y, result.plane_z_m]]), self.camera.K, self.camera.dist, T_W_C)[0]

        return [round(float(uv[0]), 1), round(float(uv[1]), 1)] if np.isfinite(uv).all() else None

    def consume(self, frame, obs) -> BOutput:
        """A 실행기 `b_consumer(frame, obs)` 자리용. `process`를 부르고 결과를 `last`·`on_output`으로 넘긴다."""
        out = self.process(obs, frame)
        self.last = out

        if self.on_output is not None:
            self.on_output(out)

        return out

    # ---------- 목록 입구 (A 단계 L, B-09) ----------
    def process_frame(self, envelope) -> dict:
        """A 목록 봉투(`eyetoarm.b_input/2`) → 후보별 B 결과 + 고른 C 목표 1개 (B-09 합의).

        1. 앞 `MAX_FRAME_CANDIDATES`개만 계산한다(넘는 수는 `truncated`).
        2. TF는 봉투의 프레임 공통 `tf` 하나만 쓴다. 없거나 촬영 시각·광학 프레임이 다르면 모두 `SKIP TF_UNAVAILABLE`.
        3. 고르기: 집을 수 있는 볼트(`PICK_PLACE`) 중 y 최대(`TIE_Y_M` 이내 동률은 작은 `instance_index`),
           없으면 너트(`PASS_THROUGH`) 중 y 최대, 없으면 고르지 않는다.
        4. 안 고른 후보의 보낼 수 있는 목표는 `SKIP NOT_SELECTED`로 바꾼다. C(`on_output`)로는 고른 1개만 넘긴다.
        5. 계산 시간 상한(`compute_budget_ms`)은 사진 한 장 전체 시간으로 본다. 넘으면 고른 목표를 `SKIP B_TIMEOUT`.
        돌려주는 값은 A 기록기가 그대로 JSON으로 쓰는 사전이다. 봉투 버전이 다르면 ValueError.
        """
        if getattr(envelope, "version", None) != LIST_ENVELOPE_VERSION:
            raise ValueError(f"process_frame requires {LIST_ENVELOPE_VERSION} envelope: "
                             f"{getattr(envelope, 'version', type(envelope).__name__)!r}")

        fo = envelope.observations
        frame, tf = fo.frame, fo.tf
        T_W_C = (tf.T_W_C if tf is not None and tf.child == frame.optical_frame
                 and tf.stamp_ns == frame.ros_stamp_ns else None)

        t0 = time.perf_counter()
        computed = fo.candidates[:MAX_FRAME_CANDIDATES]
        outs = [self._build(self._candidate_view(c, fo), frame, T_W_C) for c in computed]
        selected, selection = self._select(outs, computed)

        chosen = None

        for o, c in zip(outs, computed):
            if c.instance_index == selected:
                chosen = o
            elif o.goal.get("kind") != "SKIP":
                o.goal["kind"], o.goal["reason"] = "SKIP", "NOT_SELECTED"

        frame_ms = (time.perf_counter() - t0) * 1000.0

        if chosen is not None:
            if frame_ms > self.compute_budget_ms:
                chosen.goal["kind"], chosen.goal["reason"] = "SKIP", "B_TIMEOUT"

            self._dedupe(chosen.goal)
            self.last = chosen

            if self.on_output is not None:
                self.on_output(chosen)

        self.last_frame = outs                       # 같은 프레임 결과 묶음 (`render_center(frame.image, stage.last_frame)`)

        return {
            "version": FRAME_OUTPUT_VERSION,
            "candidate_count": len(fo.candidates),
            "computed": len(computed),
            "truncated": len(fo.candidates) - len(computed),
            "selected_instance_index": selected,
            "selection": selection,
            "frame_compute_ms": round(frame_ms, 3),
            "goal": None if chosen is None else chosen.goal,
            "results": [dict(o.result_dict, instance_index=c.instance_index, selected=c.instance_index == selected,
                             goal_kind=o.goal.get("kind"), goal_reason=o.goal.get("reason"))
                        for o, c in zip(outs, computed)],
        }

    @staticmethod
    def _candidate_view(c, fo) -> SimpleNamespace:
        """후보 하나 + 프레임 공통 값 → 기존 계산(`from_observation`·`correlation_keys`)이 읽는 관측 모양.

        상관 키 값은 A `list_envelope._correlation`과 같다(프레임의 run_id·capture_generation, 목록의 generation·job_seq).
        """
        f = fo.frame

        return SimpleNamespace(
            valid=True, reason=None, class_name=c.class_name, pixel_basis=c.pixel_basis,
            axis_endpoints_px=c.axis_endpoints_px, calc_contour=c.calc_contour,
            calibration_id=fo.capture.calibration_id, source_id=f.source_id, image_seq=f.image_seq,
            part_track_id=getattr(c, "part_track_id", None),   # A 확정 추적 번호(없으면 None, B-11 2026-11-22)
            run_id=f.run_id, capture_generation=f.capture_generation,
            generation=fo.generation, job_seq=fo.job_seq, detector_config_id=fo.detector_config_id)

    @staticmethod
    def _select(outs: list[BOutput], cands) -> tuple[int | None, str]:
        """고를 후보의 `instance_index`와 이유(`BOLT_MAX_Y` / `NUT_MAX_Y` / `TF_UNAVAILABLE` / `NONE`)."""
        for kind, why in (("PICK_PLACE", "BOLT_MAX_Y"), ("PASS_THROUGH", "NUT_MAX_Y")):
            pool = [(o.result.center_w_m[1], c.instance_index) for o, c in zip(outs, cands)
                    if o.goal.get("kind") == kind]

            if pool:
                y_max = max(y for y, _ in pool)
                return min(i for y, i in pool if y >= y_max - TIE_Y_M), why

        if outs and all(o.result.reason == "TF_UNAVAILABLE" for o in outs):
            return None, "TF_UNAVAILABLE"

        return None, "NONE"

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


def correlation_keys(obs, frame) -> dict:
    """같은 사진·같은 세대인지 확인하는 번호표. C 목표와 D 표시 사전에 똑같이 붙인다."""
    return {
        "run_id": obs.run_id,
        "capture_generation": obs.capture_generation,
        "generation": obs.generation,
        "job_seq": obs.job_seq,
        "detector_config_id": getattr(obs, "detector_config_id", None),
        "timestamp_ms": frame.timestamp_ms,
        "timestamp_base": str(getattr(frame.timestamp_base, "value", frame.timestamp_base)),
        "capture_stamp_ns": frame.ros_stamp_ns,
    }


SEND_KINDS = ("PICK_PLACE", "PASS_THROUGH")


def make_c_handoff(loop, queue, kinds: tuple[str, ...] = SEND_KINDS) -> Callable[[BOutput], None]:
    """B 출력 → C asyncio 대기열(`sort_goal_queue`)로 넘기는 함수를 만든다 (B→C 제안 1).

        stage.on_output = make_c_handoff(loop, sort_goal_queue)

    - A 검출 스레드에서 불려도 안전하게 `loop.call_soon_threadsafe`로 C 루프 안에서 넣는다.
    - `kinds`에 없는 목표(`SKIP`)는 넣지 않는다.
    - 대기열이 꽉 차면 가장 오래된 목표를 버리고 새 목표를 넣는다(벨트가 계속 움직이므로 새 것이 중요).
    - 넘긴 수·버린 수는 돌려준 함수의 `stats`에 남는다.
    """
    import asyncio

    stats = {"sent": 0, "skipped": 0, "dropped_old": 0}

    def _put(goal: dict) -> None:
        if queue.full():
            try:
                queue.get_nowait()
                stats["dropped_old"] += 1
            except asyncio.QueueEmpty:
                pass                                 # 확인과 꺼내기 사이에 다른 쪽이 먼저 비움 → 버릴 것이 없으니 그대로 넣는다 (B-10 관찰 2)

        queue.put_nowait(goal)
        stats["sent"] += 1

    def handoff(out: BOutput) -> None:
        if out.goal.get("kind") not in kinds:
            stats["skipped"] += 1
            return

        loop.call_soon_threadsafe(_put, out.goal)

    handoff.stats = stats  # type: ignore[attr-defined]

    return handoff


def goal_to_json(goal: dict) -> str:
    """C 목표 사전 → 한 줄 JSON (로그·TCP 시험용)."""
    return json.dumps(goal, ensure_ascii=False, separators=(",", ":"))
