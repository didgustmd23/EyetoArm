"""추적 번호를 실은 A 관측(#215): 실시간 A→B→C 경로에서 확정 추적 번호만 B에 넘긴다.

근거: 계획 `_workspace/215_builder2_tracking_live_plan_r0.md`(reviewer PASS 2026-11-20, Q1 (가): 공통형 무변, A 하위 형).

- `TrackedPlanarObservation`: `common.types.PlanarObservation`의 하위 형(frozen). 끝에 `part_track_id`(양의 정수 | None)와
  `track_state`(기록용 문자열 | None)만 더한다. 상위 형 검사(`isinstance`)·봉투 검사·`dataclasses.replace`(#207 시야 보정)를 그대로
  통과하고, B는 `getattr(obs, "part_track_id", None)`으로 읽으므로 B·공통형 파일은 바뀌지 않는다. 실시간 기록 연결(`--tracking`)만 쓴다.
- `confirmed_track_id(track_frame, context, obs)`: **확정(CONFIRMED) 번호만** 돌려준다. 그 작업의 TrackFrame(`matches_job`)이고
  상태를 갱신한 프레임(`updated`)이며 관측의 `selected_index`와 같은 `instance_index` 배정이 정확히 1개일 때만이다. 그 밖(tentative·
  ambiguous·untracked·갱신 안 함·다른 작업·선택 없음·추적 결과 없음)은 번호 None이고 사유 문자열을 함께 준다.
- 번호는 A-6 1차 실험 번호다(M2 휴리스틱, 분열은 알려진 한계 #156). 중복 집기 방지 보증이 아니다.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

from eyetoarm.common.types import PlanarObservation
from eyetoarm.vision.tracking import TrackFrame, TrackState, matches_job

NO_TRACKS = "no_tracks"            # 추적 결과 없음(추적 실패 latch·OpenCV·현재 아님)
OTHER_JOB = "other_job"            # 다른 작업의 TrackFrame
NOT_UPDATED = "not_updated"        # 이 프레임으로 상태를 바꾸지 않음
NO_SELECTION = "no_selection"      # 관측에 선택 인스턴스 없음
NO_ASSIGNMENT = "no_assignment"    # 선택 인스턴스의 배정이 없거나 둘 이상


@dataclass(frozen=True)
class TrackedPlanarObservation(PlanarObservation):
    """추적 번호를 더한 A 관측(하위 형). 번호는 확정일 때만 양의 정수, 아니면 None."""

    part_track_id: int | None = None
    track_state: str | None = None

    def __post_init__(self) -> None:
        """상위 형 검사 뒤 번호 형을 확인한다(bool 거부, 양의 정수 또는 None)."""
        super().__post_init__()

        if self.part_track_id is not None and (type(self.part_track_id) is not int or self.part_track_id < 1):
            raise ValueError(f"part_track_id must be a positive int or None: {self.part_track_id!r}")

        if self.track_state is not None and not isinstance(self.track_state, str):
            raise ValueError(f"track_state must be a string or None: {self.track_state!r}")


def confirmed_track_id(track_frame, context, obs) -> tuple[int | None, str]:
    """(확정 번호 | None, 상태 문자열). 상태는 배정 상태 값(`confirmed`·`tentative`·`ambiguous`·`untracked`) 또는 위 사유 상수."""
    if not isinstance(track_frame, TrackFrame):
        return None, NO_TRACKS

    if not matches_job(track_frame, context):
        return None, OTHER_JOB

    if not track_frame.updated:
        return None, NOT_UPDATED

    index = getattr(obs, "selected_index", None)

    if index is None:
        return None, NO_SELECTION

    hits = [a for a in track_frame.assignments if a.instance_index == index]

    if len(hits) != 1:
        return None, NO_ASSIGNMENT

    a = hits[0]

    if a.state is TrackState.CONFIRMED and a.part_track_id is not None:
        return a.part_track_id, a.state.value

    return None, a.state.value


def with_track(obs: PlanarObservation, part_track_id: int | None, track_state: str | None) -> TrackedPlanarObservation:
    """같은 관측 필드에 번호를 더한 새 관측(생성자 검사를 다시 거침). 상위 형의 모든 필드를 그대로 옮긴다."""
    values = {f.name: getattr(obs, f.name) for f in fields(PlanarObservation)}

    return TrackedPlanarObservation(**values, part_track_id=part_track_id, track_state=track_state)


__all__ = ["NO_ASSIGNMENT", "NO_SELECTION", "NO_TRACKS", "NOT_UPDATED", "OTHER_JOB", "TrackedPlanarObservation",
           "confirmed_track_id", "with_track"]
