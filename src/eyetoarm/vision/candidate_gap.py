"""목록 후보 공백 감시(#401): 볼트 검출은 있는데 목록 후보가 0인 상태가 오래 이어지면 기록 줄을 낸다.

판정·송신은 바꾸지 않고 기록만 만든다. 시간은 사진의 시뮬 시각(`ros_stamp_ns`) 기준이다.
물체가 아예 없는 정상 상태(축 경로가 다룬 범위 안 볼트 0)는 공백을 열지 않는다.
순수 로직이라 ROS 없이 시험한다.
"""

from __future__ import annotations

import math
import threading
from collections import Counter
from dataclasses import dataclass
from typing import Callable

from eyetoarm.vision.ai_result import LOW_CONFIDENCE, NON_TARGET_CLASS, AiSegResult, axis_candidate_reason

GAP_EVENT = "candidate_gap"                  # 공백이 `alert_after_s`를 넘을 때와 그 뒤 `repeat_s`마다
CLEARED_EVENT = "candidate_gap_cleared"      # 알린 공백이 끝날 때 한 번
NO_LIST_CANDIDATES = "no_list_candidates"    # 실행기 `b_list_skip_reason`의 목록 후보 없음 사유

KIND_CANDIDATES = "candidates"   # 목록 후보 있음(공백 끝)
KIND_GAP = "gap"                 # 범위 안 볼트 검출은 있는데 후보 0
KIND_EMPTY = "empty"             # 후보 0이고 범위 안 볼트 검출도 0

# 공백 사유 계수에서 뺄 거부(볼트가 아니거나 점수 미달은 「검출」로 치지 않음)
_NOT_DETECTION = (NON_TARGET_CLASS, LOW_CONFIDENCE)


class CandidateGapConfigError(ValueError):
    """`candidate_gap` 설정 오류."""


@dataclass(frozen=True)
class CandidateGapConfig:
    """`alert_after_s`: 공백이 이만큼(시뮬 s) 이어지면 첫 줄. `repeat_s`: 그 뒤 이어지면 이 간격으로 다시.

    `clear_after_s`: 범위 안 볼트가 이만큼 연달아 안 보이면 물체가 사라진 것으로 보고 공백을 닫는다
    (팔이 지나가거나 잘린 조각이 잠깐 잡힌 뒤 빈 화면만 이어지는 경우를 공백으로 세지 않음).
    """

    alert_after_s: float
    repeat_s: float
    clear_after_s: float

    def to_json(self) -> dict:
        """요약용 사전."""
        return {"alert_after_s": self.alert_after_s, "repeat_s": self.repeat_s, "clear_after_s": self.clear_after_s}


DEFAULT_CANDIDATE_GAP = CandidateGapConfig(alert_after_s=10.0, repeat_s=30.0, clear_after_s=2.0)
_KEYS = {"alert_after_s", "repeat_s", "clear_after_s"}


def parse_candidate_gap(doc, name: str = "candidate_gap") -> CandidateGapConfig:
    """설정 블록 검사: 두 키 모두, bool 아닌 유한 양수. 어긋나면 `CandidateGapConfigError`."""
    if not isinstance(doc, dict) or set(doc) != _KEYS:
        raise CandidateGapConfigError(f"{name} must have exactly {sorted(_KEYS)}")

    values = {}

    for key in sorted(_KEYS):
        v = doc[key]

        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise CandidateGapConfigError(f"{name}.{key} must be a positive number: {v!r}")

        values[key] = float(v)

    return CandidateGapConfig(**values)


def classify_instances(instances, skip_reason: str | None) -> tuple[str, Counter, Counter] | None:
    """사진 하나의 (종류, 축 후보 제외 사유 계수, 범위 밖·기하 거부 계수). 판정 대상이 아니면 None.

    `skip_reason`이 None이면 목록 후보가 있는 사진, `no_list_candidates`면 후보 0인 사진이다.
    그 밖의 건너뜀(세대·보정·현재성 등)은 후보를 판정하지 않은 사진이라 None.
    인스턴스는 `AiInstance`와 같은 속성(`in_scope`·`axis_evaluated`·`rejected`)을 가져야 한다.
    """
    if skip_reason is None:
        return KIND_CANDIDATES, Counter(), Counter()

    if skip_reason != NO_LIST_CANDIDATES:
        return None

    reasons: Counter = Counter()
    excluded: Counter = Counter()

    for inst in instances:
        if inst.in_scope and inst.axis_evaluated:
            # 목록 후보 0인 사진이므로 사유가 늘 있다. 없으면(계약 밖) 따로 센다
            reasons[axis_candidate_reason(inst) or "AXIS_VALID_UNLISTED"] += 1
        elif inst.rejected is not None and inst.rejected not in _NOT_DETECTION:
            excluded[inst.rejected] += 1

    return (KIND_GAP if reasons else KIND_EMPTY), reasons, excluded


def classify_frame(display, skip_reason: str | None) -> tuple[str, Counter, Counter] | None:
    """`classify_instances`의 AI 결과판. AI 결과가 아니면 None."""
    if not isinstance(display, AiSegResult):
        return None

    return classify_instances(display.instances, skip_reason)


class CandidateGapWatch:
    """공백 구간 상태. `observe`·`finish`는 어느 스레드에서 불러도 되고(잠금 `_lock`), `emit`은 잠금 안에서 부른다.

    `emit(event, detail)`: 기록 줄 하나(송신기 기록 큐에 넣기만 해야 함). `track_ids()`: 최근 추적 결과의 번호 목록.
    """

    def __init__(self, config: CandidateGapConfig, emit: Callable[[str, dict], None], *,
                 track_ids: Callable[[], list] | None = None, source: str = "default") -> None:
        """설정·기록 함수를 받는다. `source`는 요약에 남길 설정 출처(config | default)."""
        self.config = config
        self.source = source
        self._emit = emit
        self._track_ids = track_ids
        self._lock = threading.Lock()
        self._gap: dict | None = None           # 열린 공백(없으면 None)
        self._last_ns: int | None = None        # 마지막으로 본 사진 시각
        self.counts = {"alerts": 0, "gaps_alerted": 0, "gaps_short": 0}
        self.cleared_by: dict[str, int] = {}
        self.longest_s = 0.0                    # 알린 공백 중 가장 긴 것(시뮬 s)

    def observe(self, stamp_ns, kind: str, reasons: Counter, excluded: Counter) -> None:
        """사진 하나. 시각이 없으면 무시하고, 시각이 되감기면 열린 공백을 `time_rewind`로 닫는다."""
        if not isinstance(stamp_ns, int) or isinstance(stamp_ns, bool):
            return

        with self._lock:
            if self._last_ns is not None and stamp_ns < self._last_ns:
                self._close("time_rewind", self._last_ns)

            self._last_ns = stamp_ns
            gap = self._gap

            if kind == KIND_CANDIDATES:
                if gap is not None:
                    self._close("candidates", stamp_ns)
                return

            if kind == KIND_GAP:
                if gap is None:
                    gap = self._gap = {"since_ns": stamp_ns, "frames": 0, "empty_frames": 0, "empty_since_ns": None,
                                       "reasons": Counter(), "excluded": Counter(), "alerts": 0,
                                       "next_ns": stamp_ns + _ns(self.config.alert_after_s)}

                gap["frames"] += 1
                gap["reasons"].update(reasons)
                gap["excluded"].update(excluded)
                gap["empty_since_ns"] = None
            elif kind == KIND_EMPTY:
                if gap is None:
                    return   # 물체 없음: 정상 상태

                gap["empty_frames"] += 1
                gap["excluded"].update(excluded)

                if gap["empty_since_ns"] is None:
                    gap["empty_since_ns"] = stamp_ns

                # 범위 안 볼트가 `clear_after_s`만큼 안 보이면 물체가 사라진 것으로 보고 닫는다
                if stamp_ns - gap["empty_since_ns"] >= _ns(self.config.clear_after_s):
                    self._close("no_detection", stamp_ns)

                return   # 알림은 볼트가 보인 사진에서만 판정한다
            else:
                raise ValueError(f"unknown frame kind {kind!r}")

            if stamp_ns >= gap["next_ns"]:
                gap["alerts"] += 1
                self.counts["alerts"] += 1
                gap["next_ns"] = stamp_ns + _ns(self.config.repeat_s)
                self._emit(GAP_EVENT, {**self._detail(gap, stamp_ns), "alert_n": gap["alerts"],
                                       "part_track_ids": self._recent_ids()})

    def finish(self, reason: str) -> None:
        """실행 끝: 열린 공백을 `reason`으로 닫는다(여러 번 불러도 됨)."""
        with self._lock:
            if self._gap is not None:
                self._close(reason, self._last_ns)

    def summary(self) -> dict:
        """요약 블록: 설정·계수·끝난 사유·가장 긴 알린 공백·지금 열린 알린 공백 길이."""
        with self._lock:
            gap = self._gap
            open_s = (None if gap is None or not gap["alerts"] or self._last_ns is None
                      else _s(self._last_ns - gap["since_ns"]))

            return {"source": self.source, **self.config.to_json(), **self.counts,
                    "cleared_by": dict(sorted(self.cleared_by.items())), "longest_s": round(self.longest_s, 3),
                    "open_alerted_s": open_s}

    # ------------------------------------------------------------------ 잠금 안

    def _close(self, by: str, stamp_ns: int | None) -> None:
        """열린 공백을 닫는다. 알린 공백이면 「풀림」 줄을 낸다."""
        gap, self._gap = self._gap, None

        if gap is None:
            return

        if not gap["alerts"]:
            self.counts["gaps_short"] += 1
            return

        end_ns = gap["since_ns"] if stamp_ns is None else stamp_ns
        detail = self._detail(gap, end_ns)
        self.counts["gaps_alerted"] += 1
        self.cleared_by[by] = self.cleared_by.get(by, 0) + 1
        self.longest_s = max(self.longest_s, detail["duration_s"])
        self._emit(CLEARED_EVENT, {**detail, "cleared_by": by, "alerts": gap["alerts"]})

    def _detail(self, gap: dict, stamp_ns: int) -> dict:
        """기록 줄 공통 키."""
        reasons = gap["reasons"]
        main = None if not reasons else sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

        return {"since_ros_stamp_ns": gap["since_ns"], "ros_stamp_ns": stamp_ns,
                "duration_s": _s(stamp_ns - gap["since_ns"]), "frames": gap["frames"],
                "empty_frames": gap["empty_frames"], "main_reason": main,
                "reasons": dict(sorted(reasons.items())), "excluded": dict(sorted(gap["excluded"].items()))}

    def _recent_ids(self) -> list | None:
        """최근 추적 결과의 번호(추적이 없으면 None)."""
        if self._track_ids is None:
            return None

        return list(self._track_ids())


def _ns(seconds: float) -> int:
    """초 → ns(정수)."""
    return int(round(seconds * 1e9))


def _s(ns: int) -> float:
    """ns → 초(소수 3자리, 기록용)."""
    return round(ns / 1e9, 3)


__all__ = ["CLEARED_EVENT", "DEFAULT_CANDIDATE_GAP", "GAP_EVENT", "KIND_CANDIDATES", "KIND_EMPTY", "KIND_GAP",
           "NO_LIST_CANDIDATES", "CandidateGapConfig", "CandidateGapConfigError", "CandidateGapWatch",
           "classify_frame", "classify_instances", "parse_candidate_gap"]
