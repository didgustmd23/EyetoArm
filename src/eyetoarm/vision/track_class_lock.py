"""확정 트랙의 볼트 종류 고정(#368, D-5 ⓑ). ROS 없음.

추적 흐름(`/a/track_states`)의 `part_class`는 끝점 입력마다 고른 B 규격(`plane_spec`)이다. 팔·그리퍼가 볼트를
가린 사진에서는 끝점 길이가 짧게 재져 M6×40(BOLT_B)이 M3×20(BOLT_A)으로 바뀌고, C가 `TRACK_CLASS_MISMATCH`로
집기를 멈춘다(#354 기록 4건). 이 모듈은 한 번 정한 종류를 아래 사진으로는 바꾸지 않는다.

- 팔 근처 사진: 링크 카메라 → 후보 중심 선분이 가림 캡슐 + `near_arm_mm` 안을 지남(호출자가 판정해 넘김).
- 급감 사진: 직전 팔 밖 끝점 사진(급감 사진 포함)의 길이보다 `drop_mm` 넘게 짧음. 급감 사진도 기준을 갱신하므로
  정말 짧은 볼트로 바뀌었다면 첫 장만 빠지고 다음 장부터 센다. 팔 근처 사진의 길이는 기준이 되지 않는다.

이 둘이 아닌 끝점 사진이 `switch_frames`장 연속 새 종류여야 바꾼다. 지금 종류의 센 사진이 오면 연속이 끊긴다.
팔 근처·급감 사진은 연속을 늘리지도 끊지도 않는다. 설정이 없으면(`None`) 사진마다 덮어쓰는 이전 동작과 같다.

소유: 추적 스레드 하나(`LiveWorldTracking`)만 쓴다. 잠금 없음.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


class ClassLockConfigError(ValueError):
    """종류 고정 설정 오류."""


@dataclass(frozen=True)
class ClassLockConfig:
    """`switch_frames`: 바꾸는 데 필요한 연속 센 사진 수, `near_arm_mm`: 가림 캡슐 여유(mm), `drop_mm`: 급감 기준(mm)."""

    switch_frames: int
    near_arm_mm: float
    drop_mm: float

    def to_json(self) -> dict:
        """요약용 사전."""
        return {"switch_frames": self.switch_frames, "near_arm_mm": self.near_arm_mm, "drop_mm": self.drop_mm}


def parse_class_lock(doc) -> ClassLockConfig:
    """설정 사전 → `ClassLockConfig`. 키는 정확히 `switch_frames`(int ≥ 1)·`near_arm_mm`(≥ 0)·`drop_mm`(> 0)."""
    keys = {"switch_frames", "near_arm_mm", "drop_mm"}

    if not isinstance(doc, dict) or set(doc) != keys:
        raise ClassLockConfigError(f"class_lock must have exactly {sorted(keys)}: {doc!r}")

    n = doc["switch_frames"]

    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ClassLockConfigError(f"switch_frames must be an int >= 1: {n!r}")

    for name, low_ok in (("near_arm_mm", True), ("drop_mm", False)):
        v = doc[name]

        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or (v == 0 and not low_ok):
            raise ClassLockConfigError(f"{name} must be a finite number {'>= 0' if low_ok else '> 0'}: {v!r}")

    return ClassLockConfig(switch_frames=n, near_arm_mm=float(doc["near_arm_mm"]), drop_mm=float(doc["drop_mm"]))


# 판정 이름(기록·계수 키)
FIRST, SAME, KEPT_NEAR_ARM, KEPT_DROP, PENDING, SWITCHED, OVERWRITE = (
    "first", "same", "kept_near_arm", "kept_drop", "pending", "switched", "overwrite")


class _Entry:
    """트랙 하나의 고정 상태."""

    __slots__ = ("spec", "z_m", "ref_length_mm", "cand_spec", "streak")

    def __init__(self, spec: str, z_m: float, length_mm: float | None) -> None:
        self.spec, self.z_m = spec, z_m
        self.ref_length_mm = length_mm   # 직전 팔 밖 끝점 사진 길이(급감 기준)
        self.cand_spec: str | None = None   # 바꿀 후보 종류와 그 연속 센 사진 수
        self.streak = 0


class TrackClassLock:
    """확정 번호 → (종류, 투영 높이 m). `observe`로 끝점 사진 하나씩 넣고 `classes`를 발행에 쓴다."""

    def __init__(self, config: ClassLockConfig | None) -> None:
        self.config = config
        self._entries: dict = {}
        self.classes: dict = {}   # 번호 → (종류, 높이 m), frame_message `classes`에 그대로 넘긴다
        self.counts = {k: 0 for k in (FIRST, SAME, KEPT_NEAR_ARM, KEPT_DROP, PENDING, SWITCHED, OVERWRITE)}

    def clear(self) -> None:
        """재설정: 모든 트랙의 종류를 잊는다(계수는 남김)."""
        self._entries = {}
        self.classes = {}

    def observe(self, part_track_id, spec: str, z_m: float, length_mm: float | None, near_arm: bool) -> dict:
        """확정 트랙의 끝점 사진 하나. 판정 사전 {`decision`, `spec`(이번 사진), `class`(발행할 종류), `streak`}을 돌려준다."""
        decision = self._decide(part_track_id, spec, z_m, length_mm, near_arm)
        self.counts[decision] += 1
        e = self._entries.get(part_track_id)
        published = self.classes[part_track_id][0]

        return {"part_track_id": part_track_id, "decision": decision, "spec": spec, "class": published,
                "streak": 0 if e is None else e.streak}

    def _decide(self, pid, spec: str, z_m: float, length_mm: float | None, near_arm: bool) -> str:
        cfg = self.config

        if cfg is None:   # 이전 동작: 사진마다 덮어쓴다
            first = pid not in self.classes
            self.classes[pid] = (spec, z_m)

            return FIRST if first else OVERWRITE

        e = self._entries.get(pid)

        if e is None:   # 처음 받은 종류는 바로 정한다(미확정 트랙은 호출하지 않음)
            self._entries[pid] = _Entry(spec, z_m, length_mm)
            self.classes[pid] = (spec, z_m)

            return FIRST

        if near_arm:
            return SAME if spec == e.spec else KEPT_NEAR_ARM

        drop = length_mm is not None and e.ref_length_mm is not None and length_mm < e.ref_length_mm - cfg.drop_mm

        if length_mm is not None:   # 급감 기준은 팔 밖 사진 길이(급감 사진 포함): 정말 짧아졌다면 다음 장부터 센다
            e.ref_length_mm = length_mm

        if drop:
            return SAME if spec == e.spec else KEPT_DROP

        if spec == e.spec:
            e.cand_spec, e.streak = None, 0

            return SAME

        if spec != e.cand_spec:   # 다른 새 종류면 연속을 새로 센다
            e.cand_spec, e.streak = spec, 0

        e.streak += 1

        if e.streak < cfg.switch_frames:
            return PENDING

        e.spec, e.z_m = spec, z_m
        e.cand_spec, e.streak = None, 0
        self.classes[pid] = (spec, z_m)

        return SWITCHED


__all__ = ["ClassLockConfig", "ClassLockConfigError", "TrackClassLock", "parse_class_lock"]
