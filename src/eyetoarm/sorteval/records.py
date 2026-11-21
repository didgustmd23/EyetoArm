"""분류 시험 실행 기록: 정규화 사건 자료형, 원본 → 정규화 어댑터, 가짜 입력 생성.

C·B 원본 형식이 정해지지 않아 어댑터는 초안 JSONL(한 줄 = `{"event": 종류, ...}`)을 읽고, 형식이 정해지면
어댑터만 바꾼다. 시각은 SIM_TIME ns만 받고, 정의되지 않은·누락 필드는 거부하며 사유를 모두 모은다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from eyetoarm.sorteval.sequence import BOLT_SPECS, EXPECTED_BIN, SequenceFormatError, loads_strict

TIMESTAMP_BASE = "SIM_TIME"
BINS = ("BOLT_A", "BOLT_B", "NUT", "ISOLATION")
CYCLE_STATUSES = ("SUCCEEDED", "FAILED", "CANCELED", "EXPIRED")
GOAL_KINDS = ("PICK_PLACE", "PASS_THROUGH", "STOP_FOR_PICK", "ABORT_CYCLE")
B_CLASSES = BOLT_SPECS + ("UNRESOLVED",)
B_ACTIONS = ("GOAL_SENT", "SKIP")
RUN_EVENT_KINDS = ("USER_STOP", "ISOLATE_STOP", "RUN_END")
RUN_STATUSES = ("planned", "executed", "fake_input")
LINK_SOURCES = ("c_cycle", "eval_matcher")
_MAX_INT_DIGITS = 19


@dataclass(frozen=True)
class SupplyEvent:
    """C 공급 사건(실행 정답 기록): 순서표 행 ↔ 풀 부품 ID ↔ 실제 공급 규격·시각."""

    supply_index: int
    pool_part_id: str
    supplied_spec: str
    stamp_ns: int


@dataclass(frozen=True)
class ArrivalEvent:
    """C 도착 확인 사건(런타임 피드백). 수량 집계의 유일한 근거."""

    pool_part_id: str
    bin: str
    stamp_ns: int
    part_track_id: str | None = None


@dataclass(frozen=True)
class CycleEvent:
    """C 사이클 결과(`SortPart` result 등). `part_track_id`↔`pool_part_id` 연결을 C가 남긴다."""

    part_track_id: str
    pool_part_id: str | None
    goal_kind: str
    status: str
    reason: str | None
    stamp_ns: int
    place_bin: str | None = None
    belt_stop_ns: int | None = None
    belt_restart_ns: int | None = None


@dataclass(frozen=True)
class BDecision:
    """B 판정 기록: 규격 판정과 goal 송신/건너뜀."""

    part_track_id: str
    part_class: str
    action: str
    skip_reason: str | None
    stamp_ns: int


@dataclass(frozen=True)
class ATrack:
    """A 트랙 요약: 첫 관측 시각과 무효 사유별 건수(A 처리 기록에서 만든다)."""

    part_track_id: str
    first_seen_ns: int
    invalid_reasons: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TrackLink:
    """A 트랙 ↔ 풀 부품 연결. B가 건너뛴 부품은 C 사이클 기록이 없어 평가 측 대응기
    (`eval_matcher`, 평가 경로 전용)나 C 기록(`c_cycle`)이 남긴다."""

    part_track_id: str
    pool_part_id: str
    stamp_ns: int
    source: str


@dataclass(frozen=True)
class RunEvent:
    """실행 사건: 사용자 정지·격리 정지·실행 끝."""

    kind: str
    stamp_ns: int


@dataclass(frozen=True)
class RunMeta:
    """실행 메타: 평가 실행 식별·상태·검출 방식·동일 조건 비교 항목·확인 기록."""

    eval_run_id: str
    status: str
    fake_input: bool
    sequence_id: str
    sequence_sha256: str
    detector: dict
    conditions: dict
    confirmer: str | None
    date: str | None
    procedure: str | None
    evidence_paths: tuple


@dataclass
class RunRecords:
    """정규화된 실행 기록 전체. 취합의 유일한 입력."""

    supplies: list = field(default_factory=list)
    arrivals: list = field(default_factory=list)
    cycles: list = field(default_factory=list)
    b_decisions: list = field(default_factory=list)
    a_tracks: list = field(default_factory=list)
    track_links: list = field(default_factory=list)
    run_events: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# 필드 검사 도우미
# ---------------------------------------------------------------------------

class _Reject(Exception):
    """한 사건을 거부한다(사유 문자열)."""


def _keys(raw: dict, required: tuple, optional: tuple) -> None:
    """누락·미정의 필드 거부."""
    missing = [k for k in required if k not in raw]
    unknown = [k for k in raw if k not in required and k not in optional]

    if missing:
        raise _Reject(f"missing field(s) {missing}")

    if unknown:
        raise _Reject(f"undefined field(s) {unknown}")


def _str(raw: dict, key: str, *, optional: bool = False) -> str | None:
    """비어 있지 않은 문자열(optional이면 None 허용)."""
    value = raw.get(key)

    if value is None and optional:
        return None

    if not isinstance(value, str) or not value:
        raise _Reject(f"{key}: must be a non-empty string")

    return value


def _enum(raw: dict, key: str, allowed: tuple, *, optional: bool = False) -> str | None:
    """허용 값 중 하나."""
    value = raw.get(key)

    if value is None and optional:
        return None

    if value not in allowed:
        raise _Reject(f"{key}: {value!r} not in {allowed}")

    return value


def _stamp(raw: dict, key: str, *, optional: bool = False) -> int | None:
    """SIM_TIME ns 정수(bool·실수·음수·비유한 거부)."""
    value = raw.get(key)

    if value is None and optional:
        return None

    if isinstance(value, bool) or not isinstance(value, int):
        raise _Reject(f"{key}: must be an integer ns (got {type(value).__name__})")

    if value < 0:
        raise _Reject(f"{key}: negative time {value}")

    return value


def _base(raw: dict) -> None:
    """시각 기준은 SIM_TIME만."""
    if raw.get("timestamp_base") != TIMESTAMP_BASE:
        raise _Reject(f"timestamp_base: only {TIMESTAMP_BASE} is accepted (got {raw.get('timestamp_base')!r})")


def _positive_int(raw: dict, key: str) -> int:
    """1 이상 정수(bool 거부)."""
    value = raw.get(key)

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise _Reject(f"{key}: must be an int >= 1")

    return value


# ---------------------------------------------------------------------------
# 어댑터(초안 JSONL 형식 → 정규화 사건). C·B 형식 확정 시 이 함수들만 교체한다.
# ---------------------------------------------------------------------------

def adapt_c_supply(raw: dict) -> SupplyEvent:
    """C 공급 사건(초안)."""
    _keys(raw, ("event", "supply_index", "pool_part_id", "supplied_spec", "stamp_ns", "timestamp_base"), ())
    _base(raw)

    return SupplyEvent(_positive_int(raw, "supply_index"), _str(raw, "pool_part_id"),
                       _enum(raw, "supplied_spec", tuple(EXPECTED_BIN)), _stamp(raw, "stamp_ns"))


def adapt_c_arrival(raw: dict) -> ArrivalEvent:
    """C 도착 확인 사건(초안)."""
    _keys(raw, ("event", "pool_part_id", "bin", "stamp_ns", "timestamp_base"), ("part_track_id",))
    _base(raw)

    return ArrivalEvent(_str(raw, "pool_part_id"), _enum(raw, "bin", BINS), _stamp(raw, "stamp_ns"),
                        _str(raw, "part_track_id", optional=True))


def adapt_c_cycle(raw: dict) -> CycleEvent:
    """C 사이클 결과(초안, `SortPart` result + 연결 필드)."""
    _keys(raw, ("event", "part_track_id", "pool_part_id", "goal_kind", "status", "reason", "stamp_ns",
                "timestamp_base"), ("place_bin", "belt_stop_ns", "belt_restart_ns"))
    _base(raw)

    return CycleEvent(_str(raw, "part_track_id"), _str(raw, "pool_part_id", optional=True),
                      _enum(raw, "goal_kind", GOAL_KINDS), _enum(raw, "status", CYCLE_STATUSES),
                      _str(raw, "reason", optional=True), _stamp(raw, "stamp_ns"),
                      _enum(raw, "place_bin", BOLT_SPECS, optional=True),
                      _stamp(raw, "belt_stop_ns", optional=True), _stamp(raw, "belt_restart_ns", optional=True))


def adapt_b_decision(raw: dict) -> BDecision:
    """B 판정·건너뜀 기록(초안)."""
    _keys(raw, ("event", "part_track_id", "part_class", "action", "skip_reason", "stamp_ns", "timestamp_base"), ())
    _base(raw)

    action = _enum(raw, "action", B_ACTIONS)
    skip_reason = _str(raw, "skip_reason", optional=True)

    if (action == "SKIP") != (skip_reason is not None):
        raise _Reject("skip_reason: required exactly when action is SKIP")

    return BDecision(_str(raw, "part_track_id"), _enum(raw, "part_class", B_CLASSES), action, skip_reason,
                     _stamp(raw, "stamp_ns"))


def adapt_a_track(raw: dict) -> ATrack:
    """A 트랙 요약(A 처리 기록에서 만든 초안)."""
    _keys(raw, ("event", "part_track_id", "first_seen_ns", "timestamp_base"), ("invalid_reasons",))
    _base(raw)

    reasons = raw.get("invalid_reasons") or {}

    if not isinstance(reasons, dict) or not all(
            isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool) and v >= 0
            for k, v in reasons.items()):
        raise _Reject("invalid_reasons: must map reason -> non-negative int")

    return ATrack(_str(raw, "part_track_id"), _stamp(raw, "first_seen_ns"), dict(reasons))


def adapt_track_link(raw: dict) -> TrackLink:
    """A 트랙 ↔ 풀 부품 연결(초안)."""
    _keys(raw, ("event", "part_track_id", "pool_part_id", "stamp_ns", "source", "timestamp_base"), ())
    _base(raw)

    return TrackLink(_str(raw, "part_track_id"), _str(raw, "pool_part_id"), _stamp(raw, "stamp_ns"),
                     _enum(raw, "source", LINK_SOURCES))


def adapt_run_event(raw: dict) -> RunEvent:
    """실행 사건(초안)."""
    _keys(raw, ("event", "kind", "stamp_ns", "timestamp_base"), ())
    _base(raw)

    return RunEvent(_enum(raw, "kind", RUN_EVENT_KINDS), _stamp(raw, "stamp_ns"))


_ADAPTERS = {
    "supply": (adapt_c_supply, "supplies"),
    "arrival": (adapt_c_arrival, "arrivals"),
    "cycle": (adapt_c_cycle, "cycles"),
    "b_decision": (adapt_b_decision, "b_decisions"),
    "a_track": (adapt_a_track, "a_tracks"),
    "track_link": (adapt_track_link, "track_links"),
    "run_event": (adapt_run_event, "run_events"),
}


def adapt_events(raws: list) -> tuple[RunRecords, list[str]]:
    """원본 사건 목록 → (정규화 기록, 거부 사유). 같은 supply_index 공급 사건 중복과 같은 트랙 A 요약 중복도 거부."""
    records = RunRecords()
    errors: list[str] = []

    for pos, raw in enumerate(raws, start=1):
        if not isinstance(raw, dict) or raw.get("event") not in _ADAPTERS:
            errors.append(f"event {pos}: unknown event kind {raw.get('event') if isinstance(raw, dict) else raw!r}")
            continue

        adapter, bucket = _ADAPTERS[raw["event"]]

        try:
            getattr(records, bucket).append(adapter(raw))
        except _Reject as exc:
            errors.append(f"event {pos} ({raw['event']}): {exc}")

    seen_idx = [s.supply_index for s in records.supplies]
    dup_idx = sorted({i for i in seen_idx if seen_idx.count(i) > 1})

    if dup_idx:
        errors.append(f"supply: duplicate supply_index {dup_idx}")

    seen_tracks = [t.part_track_id for t in records.a_tracks]
    dup_tracks = sorted({t for t in seen_tracks if seen_tracks.count(t) > 1})

    if dup_tracks:
        errors.append(f"a_track: duplicate part_track_id {dup_tracks}")

    return records, errors


def parse_events_jsonl(text: str) -> tuple[RunRecords, list[str]]:
    """초안 JSONL 텍스트 → (정규화 기록, 거부 사유). 빈 줄은 건너뛴다."""
    raws = []
    errors: list[str] = []

    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue

        try:
            raws.append(loads_strict(line))
        except SequenceFormatError as exc:
            errors.append(f"line {line_no}: {exc}")
            raws.append({"event": None})

    records, adapt_errors = adapt_events(raws)

    return records, errors + adapt_errors


def adapt_run_meta(raw) -> tuple[RunMeta | None, list[str]]:
    """실행 메타(초안) → (RunMeta, 거부 사유). `status == fake_input` ⇔ `fake_input == true`."""
    required = ("eval_run_id", "status", "fake_input", "sequence_id", "sequence_sha256", "detector", "conditions",
                "confirmer", "date", "procedure", "evidence_paths")

    try:
        if not isinstance(raw, dict):
            raise _Reject("run meta must be an object")

        _keys(raw, required, ())
        status = _enum(raw, "status", RUN_STATUSES)

        if not isinstance(raw["fake_input"], bool) or raw["fake_input"] != (status == "fake_input"):
            raise _Reject("fake_input: must be true exactly when status is fake_input")

        detector = raw["detector"]

        if not isinstance(detector, dict) or not isinstance(detector.get("method"), str) or not detector["method"]:
            raise _Reject("detector: must be an object with a non-empty 'method'")

        if not isinstance(raw["conditions"], dict):
            raise _Reject("conditions: must be an object")

        paths = raw["evidence_paths"]

        if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
            raise _Reject("evidence_paths: must be a list of strings")

        for key in ("confirmer", "date", "procedure"):
            _str(raw, key, optional=True)

        return RunMeta(_str(raw, "eval_run_id"), status, raw["fake_input"], _str(raw, "sequence_id"),
                       _str(raw, "sequence_sha256"), dict(detector), dict(raw["conditions"]), raw["confirmer"],
                       raw["date"], raw["procedure"], tuple(paths)), []
    except _Reject as exc:
        return None, [f"run meta: {exc}"]


# ---------------------------------------------------------------------------
# 가짜 입력(시험·시연용)
# ---------------------------------------------------------------------------

FAKE_OUTCOMES = ("correct", "wrong_bin", "bolt_to_nut_bin", "isolated", "not_arrived", "undetected",
                 "a_invalid", "b_unresolved", "b_wrong_class", "b_skip", "c_failed", "no_supply",
                 "supply_mismatch", "dup_arrival", "user_abort")


def _other_bolt(spec: str) -> str:
    """다른 볼트 규격."""
    return "BOLT_B" if spec == "BOLT_A" else "BOLT_A"


def fake_events(sequence: dict, outcomes: dict | None = None, *, interval_ns: int = 10_000_000_000,
                t0_ns: int = 1_000_000_000) -> list[dict]:
    """순서표와 공급별 가짜 결과(`FAKE_OUTCOMES`, 기본 correct)로 초안 JSONL 원본 사건 목록을 만든다.

    풀 부품 ID는 3개를 돌려 쓴다(회수·재공급 재현). 시각은 공급 i의 기준 t0 + (i-1)·interval 뒤 고정 오프셋.
    `user_abort`는 그 행 직전에 USER_STOP을 넣고 그 행부터 공급 사건을 만들지 않는다.
    """
    outcomes = outcomes or {}
    unknown = sorted(set(outcomes.values()) - set(FAKE_OUTCOMES))

    if unknown:
        raise ValueError(f"unknown fake outcome(s) {unknown}")

    events: list[dict] = []
    base = {"timestamp_base": TIMESTAMP_BASE}
    stopped = False
    last_t = t0_ns

    for row in sequence["supplies"]:
        idx = row["supply_index"]
        spec = row["part_spec"]
        kind = outcomes.get(idx, "correct")
        t = t0_ns + (idx - 1) * interval_ns
        last_t = t
        pool = f"part_{(idx - 1) % 3:02d}"
        track = f"trk_{idx:04d}"

        if kind == "user_abort" and not stopped:
            events.append({"event": "run_event", "kind": "USER_STOP", "stamp_ns": t - 1, **base})
            stopped = True

        if stopped or kind == "no_supply":
            continue

        supplied = _other_bolt(spec) if kind == "supply_mismatch" and spec in BOLT_SPECS else spec
        events.append({"event": "supply", "supply_index": idx, "pool_part_id": pool, "supplied_spec": supplied,
                       "stamp_ns": t, **base})

        if kind == "undetected":
            continue

        events.append({"event": "a_track", "part_track_id": track, "first_seen_ns": t + 500_000_000,
                       **({"invalid_reasons": {"TOP_FACE_UNRESOLVED": 3}} if kind == "a_invalid" else {}), **base})
        events.append({"event": "track_link", "part_track_id": track, "pool_part_id": pool,
                       "stamp_ns": t + 500_000_000, "source": "eval_matcher", **base})

        if kind == "a_invalid":
            continue

        if spec == "NUT":
            arrival_bin = "NUT" if kind != "wrong_bin" else "ISOLATION"
            events.append({"event": "arrival", "pool_part_id": pool, "bin": arrival_bin, "stamp_ns": t + 6_000_000_000,
                           "part_track_id": track, **base})
            continue

        judged = _other_bolt(supplied) if kind == "b_wrong_class" else supplied

        if kind == "b_unresolved":
            events.append({"event": "b_decision", "part_track_id": track, "part_class": "UNRESOLVED",
                           "action": "SKIP", "skip_reason": "UNRESOLVED", "stamp_ns": t + 600_000_000, **base})
            continue

        if kind == "b_skip":
            events.append({"event": "b_decision", "part_track_id": track, "part_class": judged, "action": "SKIP",
                           "skip_reason": "WORK_COORD_INVALID", "stamp_ns": t + 600_000_000, **base})
            continue

        events.append({"event": "b_decision", "part_track_id": track, "part_class": judged, "action": "GOAL_SENT",
                       "skip_reason": None, "stamp_ns": t + 600_000_000, **base})

        if kind == "isolated":
            events.append({"event": "cycle", "part_track_id": track, "pool_part_id": pool, "goal_kind": "ABORT_CYCLE",
                           "status": "SUCCEEDED", "reason": "REDETECT_UNSTABLE", "stamp_ns": t + 4_000_000_000,
                           "belt_stop_ns": t + 1_000_000_000, "belt_restart_ns": t + 7_000_000_000, **base})
            events.append({"event": "arrival", "pool_part_id": pool, "bin": "ISOLATION", "stamp_ns": t + 5_000_000_000,
                           "part_track_id": track, **base})
            continue

        if kind == "c_failed":
            events.append({"event": "cycle", "part_track_id": track, "pool_part_id": pool, "goal_kind": "PICK_PLACE",
                           "status": "FAILED", "reason": "ATTACH_FAILED", "stamp_ns": t + 4_000_000_000,
                           "place_bin": judged, "belt_stop_ns": t + 1_000_000_000, **base})
            continue

        if kind == "not_arrived":
            events.append({"event": "cycle", "part_track_id": track, "pool_part_id": pool, "goal_kind": "PICK_PLACE",
                           "status": "SUCCEEDED", "reason": None, "stamp_ns": t + 4_000_000_000, "place_bin": judged,
                           "belt_stop_ns": t + 1_000_000_000, "belt_restart_ns": t + 6_000_000_000, **base})
            continue

        if kind == "bolt_to_nut_bin":
            events.append({"event": "arrival", "pool_part_id": pool, "bin": "NUT", "stamp_ns": t + 6_000_000_000,
                           "part_track_id": track, **base})
            continue

        arrival_bin = _other_bolt(judged) if kind == "wrong_bin" else judged
        events.append({"event": "cycle", "part_track_id": track, "pool_part_id": pool, "goal_kind": "PICK_PLACE",
                       "status": "SUCCEEDED", "reason": None, "stamp_ns": t + 4_000_000_000, "place_bin": judged,
                       "belt_stop_ns": t + 1_000_000_000, "belt_restart_ns": t + 6_000_000_000, **base})
        events.append({"event": "arrival", "pool_part_id": pool, "bin": arrival_bin, "stamp_ns": t + 5_000_000_000,
                       "part_track_id": track, **base})

        if kind == "dup_arrival":
            events.append({"event": "arrival", "pool_part_id": pool, "bin": arrival_bin,
                           "stamp_ns": t + 5_100_000_000, "part_track_id": track, **base})

    events.append({"event": "run_event", "kind": "RUN_END", "stamp_ns": last_t + interval_ns, **base})

    return events


def fake_run_meta(sequence: dict, *, method: str = "opencv", eval_run_id: str | None = None,
                  conditions: dict | None = None) -> dict:
    """가짜 입력 실행 메타(초안 형식). status·fake_input이 강제로 가짜 표시."""
    return {
        "eval_run_id": eval_run_id or f"fake_{sequence['sequence_id']}_{method}",
        "status": "fake_input",
        "fake_input": True,
        "sequence_id": sequence["sequence_id"],
        "sequence_sha256": sequence["content_sha256"],
        "detector": {"method": method, "config_id": f"{method}_fake"},
        "conditions": dict(conditions) if conditions is not None else {
            "world_id": "fake_world", "belt_config_id": "fake_belt", "supply_config_id": "fake_supply",
            "b_config_id": "fake_b", "c_commit": "fake_c", "calibration_id": "fake_calib",
        },
        "confirmer": None,
        "date": None,
        "procedure": "fake_events (eyetoarm.sorteval.records)",
        "evidence_paths": [],
    }


def events_to_jsonl(events: list[dict]) -> str:
    """원본 사건 목록 → JSONL 텍스트."""
    return "".join(json.dumps(e, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n" for e in events)
