"""detect 영상 W 번호 표시 보완: 같은 작업의 W 결과가 아직 없을 때 직전 W 결과의 확정 번호를 위치로 짝지어 화면에만 그린다.

C 목표·B 중복 방지 판정은 그 프레임의 W 결과만 쓰며, 기다림을 넣지 않는다. 짝짓기는 박스 IoU가 서로 1위이고
`min_iou`·`ambiguity_margin`을 넘을 때만 1:1로 붙인다. ROS·Qt를 import하지 않는다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

from eyetoarm.vision.tracking import FrameKey, TrackAssignment, TrackFrame, TrackMode, TrackState

PREV_REASON = "display_prev"                    # 짝지어 그린 인스턴스의 판정 사유
PREV_UNMATCHED = "display_prev_unmatched"       # 짝이 없거나 모호해 번호를 그리지 않은 인스턴스


class DisplayPrevConfigError(ValueError):
    """`display_prev` 설정 오류."""


@dataclass(frozen=True)
class DisplayPrevConfig:
    """직전 결과 표시 시험값(기본값 없음, 모두 필수)."""

    max_age_ms: float          # 직전 결과 촬영 시각 차 상한
    min_iou: float             # 짝으로 인정할 최소 IoU(0 초과 1 이하)
    ambiguity_margin: float    # 1·2위 IoU 차 하한(0 이상 1 미만)

    def to_json(self) -> dict:
        """요약용 사전."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


def parse_display_prev(doc) -> DisplayPrevConfig:
    """`live.display_prev` 블록. 모르는 키·누락·범위 오류는 `DisplayPrevConfigError`."""
    keys = {f.name for f in fields(DisplayPrevConfig)}

    if not isinstance(doc, dict) or set(doc) != keys:
        raise DisplayPrevConfigError(f"display_prev must have exactly {sorted(keys)}")

    def num(name):
        v = doc[name]

        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise DisplayPrevConfigError(f"display_prev.{name} must be a finite number: {v!r}")

        return float(v)

    age, iou, margin = num("max_age_ms"), num("min_iou"), num("ambiguity_margin")

    if age <= 0:
        raise DisplayPrevConfigError(f"display_prev.max_age_ms must be > 0: {age!r}")

    if not 0.0 < iou <= 1.0:
        raise DisplayPrevConfigError(f"display_prev.min_iou must be in (0, 1]: {iou!r}")

    if not 0.0 <= margin < 1.0:
        raise DisplayPrevConfigError(f"display_prev.ambiguity_margin must be in [0, 1): {margin!r}")

    return DisplayPrevConfig(max_age_ms=age, min_iou=iou, ambiguity_margin=margin)


@dataclass(frozen=True)
class PrevSnapshot:
    """직전 W 결과의 표시용 사본(불변). `confirmed`: (인스턴스 index, 번호, hits, 클래스 이름, 박스 xyxy 원본 픽셀)."""

    key: FrameKey
    timestamp_ms: int
    tracker_epoch: int
    confirmed: tuple[tuple[int, int, int, str, tuple[float, float, float, float]], ...]


@dataclass(frozen=True)
class PrevDisplay:
    """직전 결과로 만든 표시 전용 판정. `tracks`는 이번 작업 인스턴스와 1:1, 짝 없는 인스턴스는 untracked."""

    tracks: TrackFrame
    prev_image_seq: int
    age_ms: float
    matched: int


def snapshot(tracks: TrackFrame, display) -> PrevSnapshot | None:
    """추적 스레드: 갱신한 W 결과 + 그 프레임 AI 표시 결과 → 표시용 사본(확정 번호만). 미갱신은 None."""
    if not tracks.updated:
        return None

    boxes = {int(i.index): (str(i.class_name), tuple(float(v) for v in i.box_xyxy)) for i in display.instances}
    confirmed = tuple((a.instance_index, a.part_track_id, a.hits, *boxes[a.instance_index])
                      for a in tracks.assignments
                      if a.state is TrackState.CONFIRMED and a.part_track_id is not None and a.instance_index in boxes)
    key = FrameKey(run_id=tracks.run_id, source_id=tracks.source_id, image_seq=tracks.image_seq,
                   capture_generation=tracks.capture_generation, generation=tracks.generation, job_seq=tracks.job_seq,
                   detector_config_id=tracks.detector_config_id)

    return PrevSnapshot(key=key, timestamp_ms=int(tracks.timestamp_ms), tracker_epoch=int(tracks.tracker_epoch),
                        confirmed=confirmed)


def iou(a, b) -> float:
    """두 xyxy 박스의 IoU(겹침 없음·넓이 0이면 0)."""
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])

    if ix <= 0 or iy <= 0:
        return 0.0

    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter

    return inter / union if union > 0 else 0.0


def _second(values: list[float], best: int) -> float:
    """1위를 뺀 최댓값(없으면 0)."""
    return max((v for i, v in enumerate(values) if i != best), default=0.0)


def match_prev(prev: PrevSnapshot | None, context, display, config: DisplayPrevConfig, *,
               reset_pending: bool) -> tuple[PrevDisplay | None, str]:
    """직전 사본을 이번 작업(`context`, AI 결과 `display`)에 짝짓는다. (표시 판정 | None, 사유).

    같은 실행·소스·설정·세대이고 `0 < 촬영 시각 차 ≤ max_age_ms`일 때만 쓴다.
    """
    if prev is None:
        return None, "no_prev"

    if reset_pending:
        return None, "reset_pending"

    f, k = context.frame, prev.key

    if (k.run_id, k.source_id, k.detector_config_id) != (f.run_id, f.source_id, context.detector_config_id):
        return None, "scope_changed"

    if (k.capture_generation, k.generation) != (f.capture_generation, context.generation):
        return None, "generation_changed"

    age_ms = float(f.timestamp_ms) - float(prev.timestamp_ms)

    if age_ms <= 0:
        return None, "not_older"

    if age_ms > config.max_age_ms:
        return None, "too_old"

    if not prev.confirmed:
        return None, "no_confirmed"

    cur = [i for i in display.instances if i.in_scope]
    scores = [[iou(p[4], tuple(float(v) for v in c.box_xyxy)) if p[3] == c.class_name else 0.0 for c in cur]
              for p in prev.confirmed]
    pairs: dict[int, tuple[int, int]] = {}   # 이번 index → (번호, hits)

    for pi, row in enumerate(scores):
        if not row:
            break

        ci = max(range(len(row)), key=row.__getitem__)
        col = [scores[r][ci] for r in range(len(scores))]
        best = row[ci]

        if (best >= config.min_iou and max(range(len(col)), key=col.__getitem__) == pi
                and best - _second(row, ci) >= config.ambiguity_margin and best - _second(col, pi) >= config.ambiguity_margin):
            pairs[int(cur[ci].index)] = (prev.confirmed[pi][1], prev.confirmed[pi][2])

    if not pairs:
        return None, "no_match"

    assignments = tuple(TrackAssignment(int(i.index), pairs[int(i.index)][0], TrackState.CONFIRMED, pairs[int(i.index)][1],
                                        PREV_REASON) if int(i.index) in pairs else
                        TrackAssignment(int(i.index), None, TrackState.UNTRACKED, 0, PREV_UNMATCHED)
                        for i in display.instances)
    tracks = TrackFrame(run_id=f.run_id, source_id=f.source_id, image_seq=f.image_seq, capture_generation=f.capture_generation,
                        generation=context.generation, job_seq=context.job_seq,
                        detector_config_id=context.detector_config_id, timestamp_ms=int(f.timestamp_ms), mode=TrackMode.WORLD,
                        track_namespace="display_prev", tracker_epoch=prev.tracker_epoch, updated=False, event=PREV_REASON,
                        assignments=assignments)

    return PrevDisplay(tracks=tracks, prev_image_seq=k.image_seq, age_ms=age_ms, matched=len(pairs)), "prev"


__all__ = ["DisplayPrevConfig", "DisplayPrevConfigError", "PREV_REASON", "PREV_UNMATCHED", "PrevDisplay", "PrevSnapshot",
           "iou", "match_prev", "parse_display_prev", "snapshot"]
