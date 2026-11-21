"""추적 번호를 실은 A 관측·단계 L 후보(공통형의 하위 형)와 확정 추적 번호 조회.

번호는 확정(CONFIRMED)일 때만 넘기고 그 밖은 None과 사유 문자열을 준다. B는 `getattr(..., "part_track_id", None)`으로
읽으므로 공통형·봉투 파일은 바뀌지 않는다. 번호는 실험 번호이며 중복 집기 방지 보증이 아니다.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

from eyetoarm.common.types import ObjectObservation, PlanarObservation
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


@dataclass(frozen=True)
class TrackedObjectObservation(ObjectObservation):
    """추적 번호를 더한 단계 L 후보(하위 형). 번호는 확정일 때만 양의 정수, 아니면 None."""

    part_track_id: int | None = None
    track_state: str | None = None

    def __post_init__(self) -> None:
        """상위 형 검사(좌표 복사 포함) 뒤 번호 형을 확인한다(bool 거부, 양의 정수 또는 None)."""
        super().__post_init__()

        if self.part_track_id is not None and (type(self.part_track_id) is not int or self.part_track_id < 1):
            raise ValueError(f"part_track_id must be a positive int or None: {self.part_track_id!r}")

        if self.track_state is not None and not isinstance(self.track_state, str):
            raise ValueError(f"track_state must be a string or None: {self.track_state!r}")


def with_candidate_track(c: ObjectObservation, part_track_id: int | None, track_state: str | None) -> TrackedObjectObservation:
    """같은 후보 필드에 번호를 더한 새 후보(생성자 검사를 다시 거침)."""
    values = {f.name: getattr(c, f.name) for f in fields(ObjectObservation)}

    return TrackedObjectObservation(**values, part_track_id=part_track_id, track_state=track_state)


def confirmed_track_id(track_frame, context, obs) -> tuple[int | None, str]:
    """선택 인스턴스의 (확정 번호 | None, 상태 문자열). 상태는 배정 상태 값 또는 위 사유 상수."""
    return candidate_track_id(track_frame, context, getattr(obs, "selected_index", None))


def candidate_track_id(track_frame, context, index) -> tuple[int | None, str]:
    """인스턴스 index 하나의 (확정 번호 | None, 상태). index가 None이면 `no_selection`."""
    if not isinstance(track_frame, TrackFrame):
        return None, NO_TRACKS

    if not matches_job(track_frame, context):
        return None, OTHER_JOB

    if not track_frame.updated:
        return None, NOT_UPDATED

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
    """같은 관측 필드에 번호를 더한 새 관측(생성자 검사를 다시 거침)."""
    values = {f.name: getattr(obs, f.name) for f in fields(PlanarObservation)}

    return TrackedPlanarObservation(**values, part_track_id=part_track_id, track_state=track_state)


__all__ = ["NO_ASSIGNMENT", "NO_SELECTION", "NO_TRACKS", "NOT_UPDATED", "OTHER_JOB", "TrackedObjectObservation",
           "TrackedPlanarObservation", "candidate_track_id", "confirmed_track_id", "with_candidate_track", "with_track"]
