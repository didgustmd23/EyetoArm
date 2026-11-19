"""A-6 1차 인스턴스 추적(#151 S1) — 프로세스 수명 번호 발급기, 불변 `TrackFrame`, 검출 기반 픽셀+SIM_TIME 연결 상태기.

근거: `_workspace/150_builder2_part_track_id_plan.md` r2(독립 검토 PASS) §2(TrackFrame 분리)·§3.1~3.6·§4·§6, reviewer 구현 명세 #151(D1~D11).

**보장 범위(반드시 함께 읽을 것).**
- 1차는 **표시·기록 실험**이다. M2(휴리스틱) 모드는 팔(카메라) 이동을 넘는 신원 연속성, 운영 중복 집기 방지, 벨트 이동과 카메라 이동의 구분을
  보장하지 않는다. 결과를 B 목표·공통 관측 필드로 넘기지 않는다(A-6 전체 완료 아님).
- M1(정지 확인)은 인터페이스와 가짜 신호 시험까지다. 실제 정지 신호(C) 연결은 없다. M0(끔)은 추적기를 만들지 않는 것이다.
- 번호는 「A가 같은 물리 부품이라고 판단한 동안」의 표시용 번호다. 모르면 None이다. 모호하면 None으로 내고 관련 트랙을 끝낸다(표시 실험 정책, 운영 안전 판단 아님).

**규약.**
- 번호: 양의 정수, `PartTrackIdIssuer` 하나가 프로세스 수명 동안 발급(재사용·재설정 없음). 발급기는 통합 진입점이 만들어 주입한다(추적기는 만들지 않음, D11).
  (track_namespace, part_track_id)로 유일하다. 프로세스 재시작·발급기 재생성은 새 namespace다.
- `tracker_epoch`: 추적기 생성 때, 그리고 **트랙(tentative·confirmed·lost)이 하나라도 있던 전체 종료**(reset·disable·M1 정지 아님·M2 카메라 이동 의심·
  범위 변경)마다 발급기의 별도 순번에서 받는다(번호를 소비하지 않음). 트랙이 없는 반복 종료는 epoch를 올리지 않는다(멱등). 신원 주장은 한 epoch 안에서만 한다.
- 확정 근거: **잘리지 않은 연속 hit**만 센다. 잘림 hit는 트랙 유지(이미 확정된 번호 유지)에만 쓰고 연속 수를 0으로 끊는다.
- 시각: 예측·만료는 프레임 촬영 시각 `timestamp_ms`(SIM_TIME 등 프레임 기준)만 쓴다. PC 단조 시각과 섞지 않는다. 같은·이른 시각 입력은 갱신하지 않는다.
- 갱신: 한 프레임을 한 번만 반영한다(같은·이른 시각 → `stale_frame`, 이전 세대 → `old_generation`). run·source·검출 설정이 바뀌거나 세대가 오르면 전 트랙 종료.
- 상태: tentative(번호 없음) → confirmed(번호) → lost(번호 유지, 누락 허용 시간 안) → retired(계수만). 트랙은 스칼라만 갖고 영상·마스크·다각형·JobContext를
  붙잡지 않는다. active+lost 수에 상한이 있다.
- 단일 소유: A 제어 흐름 한 곳에서만 부른다(잠금 없음). 발급기만 작은 잠금을 둔다.

기본값(`TrackerConfig`)은 **A 개발 임시값**이며 운영 채택값이 아니다(연속 자료 실측 전). 표·한계는 구현 보고서에 적는다.
ROS·Qt·torch·ultralytics를 import하지 않는다.
"""
from __future__ import annotations

import math
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum

from eyetoarm.vision.ai_result import AiSegResult

_GEOMETRY_REJECTIONS = ("EMPTY_MASK", "SMALL_MASK")


class TrackingContractError(ValueError):
    """호출 계약 위반(발급기 미주입·모드 오류·M2에 정지 신호 등)."""


class TrackingInputError(ValueError):
    """프레임 입력이 잘못됨(시각·박스·점수·index). 추적 상태를 바꾸지 않는다."""


class TrackMode(str, Enum):
    """추적 모드(계획 §3.6). OFF는 「추적기를 만들지 않음」을 기록할 때만 쓰며 추적기 모드로 받지 않는다."""

    OFF = "off"                              # M0
    CONFIRMED_STATIC = "confirmed_static"    # M1: 외부 정지 확인 신호가 True인 동안만 연결
    HEURISTIC = "heuristic"                  # M2: 신호 없음, 공통 큰 이동이면 카메라 이동 의심 → 전 종료(보장 없음)


class TrackState(str, Enum):
    """인스턴스 하나의 추적 판정."""

    TENTATIVE = "tentative"    # 트랙 있음, 번호 미발급
    CONFIRMED = "confirmed"    # 번호 있음
    AMBIGUOUS = "ambiguous"    # 합쳐짐·분리·교차 의심 → 번호 None, 관련 트랙 종료
    UNTRACKED = "untracked"    # 후보 아님·갱신 안 함·상한·끔 등


def _pos_int(value, name: str, minimum: int = 1) -> int:
    """bool 아닌 정수 ≥ minimum."""
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}: {value!r}")
    return value


def _pos_float(value, name: str, *, allow_zero: bool = False) -> float:
    """bool 아닌 유한 실수 > 0(allow_zero면 ≥ 0)."""
    ok = (not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
          and (value >= 0 if allow_zero else value > 0))
    if not ok:
        raise ValueError(f"{name} must be a finite number {'>= 0' if allow_zero else '> 0'}: {value!r}")
    return float(value)


@dataclass(frozen=True)
class TrackerConfig:
    """연결 파라미터. 모두 **A 개발 임시값**(연속 자료 실측 전, 운영 채택값 아님). 단위는 이름에 둔다.

    - `min_hits`(3): 잘리지 않은 **연속** hit 수가 이만큼이면 번호 발급(잘림 hit는 연속을 0으로 끊음).
    - `max_gap_ms`(500, 촬영 시각 ms): 마지막 hit 뒤 이 시간을 넘으면 트랙 은퇴(재등장해도 새 트랙).
    - `gate_min_px`(8)·`gate_box_ratio`(1.0): 연결 반경 = max(gate_min_px, ratio × 트랙 박스 대각선). 예측 중심과 검출 중심 거리.
    - `size_ratio_max`(2.5): 박스 넓이 비가 이 배를 넘으면 연결하지 않음.
    - `ambiguity_margin`(0.2): 정규화 비용(거리/반경) 1·2위 차가 이보다 작으면 모호(합쳐짐·분리·교차).
    - `new_track_min_conf`(0.25): 이 점수 미만 검출은 기존 트랙 유지에만 쓰고 새 트랙을 만들지 않는다(운영 conf 0.05와 별개).
    - `max_tracks`(64): active+lost 상한. 차면 가장 오래된 lost부터 은퇴, 그래도 차면 새 후보는 `track_capacity`.
    - `velocity_alpha`(0.5): 속도(px/ms) 지수 평균 계수.
    - M2 카메라 이동 의심: 예측에서 가장 가까운 같은 클래스 검출(`camera_search_px` 300 안)까지의 이동이 `camera_min_tracks`(2)개 이상 트랙에서
      모두 `camera_jump_px`(40)를 넘고 중앙값에서 `camera_coherence_px`(15) 안으로 모이면 의심. 미탐(게이트 안 작은 공통 이동, 트랙 1개)·
      오탐(벨트 급가속)이 있다.
    """

    min_hits: int = 3
    max_gap_ms: int = 500
    gate_min_px: float = 8.0
    gate_box_ratio: float = 1.0
    size_ratio_max: float = 2.5
    ambiguity_margin: float = 0.2
    new_track_min_conf: float = 0.25
    max_tracks: int = 64
    velocity_alpha: float = 0.5
    camera_min_tracks: int = 2
    camera_jump_px: float = 40.0
    camera_coherence_px: float = 15.0
    camera_search_px: float = 300.0

    def __post_init__(self) -> None:
        """값 범위를 검사한다. 어긋나면 ValueError."""
        for name in ("min_hits", "max_gap_ms", "max_tracks", "camera_min_tracks"):
            _pos_int(getattr(self, name), name)

        for name in ("gate_min_px", "gate_box_ratio", "camera_jump_px", "camera_coherence_px", "camera_search_px"):
            _pos_float(getattr(self, name), name)

        _pos_float(self.ambiguity_margin, "ambiguity_margin", allow_zero=True)

        if _pos_float(self.size_ratio_max, "size_ratio_max") < 1.0:
            raise ValueError(f"size_ratio_max must be >= 1: {self.size_ratio_max!r}")

        if not 0.0 <= _pos_float(self.new_track_min_conf, "new_track_min_conf", allow_zero=True) <= 1.0:
            raise ValueError(f"new_track_min_conf must be within [0, 1]: {self.new_track_min_conf!r}")

        if not 0.0 < _pos_float(self.velocity_alpha, "velocity_alpha") <= 1.0:
            raise ValueError(f"velocity_alpha must be within (0, 1]: {self.velocity_alpha!r}")


DEFAULT_TRACKER_CONFIG = TrackerConfig()


class PartTrackIdIssuer:
    """프로세스 수명 번호 발급기(계획 §3.5). 통합 진입점이 프로세스당 1개 만들어 추적기에 주입한다.

    - `next_id()`: 1부터 단조 증가, 되돌리기·재설정 없음. `next_tracker_epoch()`: 번호와 별개인 epoch 순번.
    - `namespace`: 발급기마다 다른 문자열(기본 uuid4 hex). 프로세스 재시작·재생성 구분용.
    - 작은 잠금으로 보호한다(현재는 A 제어 흐름 한 곳에서만 부름).
    """

    def __init__(self, namespace: str | None = None) -> None:
        """namespace를 받거나 새로 만든다. 빈 문자열·비문자열은 계약 오류."""
        if namespace is None:
            namespace = uuid.uuid4().hex
        elif not isinstance(namespace, str) or not namespace:
            raise TrackingContractError(f"namespace must be a non-empty string: {namespace!r}")

        self._namespace = namespace
        self._lock = threading.Lock()
        self._last_id = 0
        self._last_epoch = 0

    @property
    def namespace(self) -> str:
        """발급기 namespace."""
        return self._namespace

    @property
    def issued(self) -> int:
        """지금까지 발급한 번호 수(= 마지막 번호)."""
        with self._lock:
            return self._last_id

    def next_id(self) -> int:
        """새 번호."""
        with self._lock:
            self._last_id += 1
            return self._last_id

    def next_tracker_epoch(self) -> int:
        """새 tracker_epoch(번호와 별개 순번)."""
        with self._lock:
            self._last_epoch += 1
            return self._last_epoch


@dataclass(frozen=True)
class FrameKey:
    """추적 결과를 작업·프레임과 대조하는 식별자 묶음(복사값)."""

    run_id: str
    source_id: str
    image_seq: int
    capture_generation: int
    generation: int
    job_seq: int
    detector_config_id: str


@dataclass(frozen=True)
class TrackInput:
    """인스턴스 하나의 추적 입력(스칼라만). `candidate=False`면 `untracked_reason`을 그대로 낸다."""

    index: int
    class_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]
    clipped: bool
    candidate: bool
    untracked_reason: str | None


@dataclass(frozen=True)
class TrackAssignment:
    """인스턴스 하나의 추적 판정. `hits`는 확정 근거인 잘리지 않은 연속 hit 수(잘림 hit 뒤 0)."""

    instance_index: int
    part_track_id: int | None
    state: TrackState
    hits: int
    reason: str | None


@dataclass(frozen=True)
class TrackFrame:
    """작업 하나의 추적 결과(불변). `assignments`는 입력 인스턴스 순서·index와 1:1이다.

    `updated=False`면 이 프레임으로 상태를 바꾸지 않았고 모든 인스턴스가 untracked다(이전 번호를 싣지 않음).
    `event`: 이 프레임에서 일어난 전 트랙 종료 사유(`camera_motion_suspected`·`camera_not_static`·`scope_changed`) 또는 None.
    """

    run_id: str
    source_id: str
    image_seq: int
    capture_generation: int
    generation: int
    job_seq: int
    detector_config_id: str
    timestamp_ms: int
    mode: TrackMode
    track_namespace: str
    tracker_epoch: int
    updated: bool
    event: str | None
    assignments: tuple[TrackAssignment, ...]


def matches_job(tracks: TrackFrame, job) -> bool:
    """TrackFrame이 그 작업(JobContext 형태: frame·generation·job_seq·detector_config_id)의 것인지 7개 식별자로 대조한다."""
    f = job.frame

    return (isinstance(tracks, TrackFrame)
            and (tracks.run_id, tracks.source_id, tracks.image_seq, tracks.capture_generation, tracks.generation,
                 tracks.job_seq, tracks.detector_config_id)
            == (f.run_id, f.source_id, f.image_seq, f.capture_generation, job.generation, job.job_seq,
                job.detector_config_id))


def track_record(tracks: TrackFrame) -> dict:
    """JSONL `ai_tracks` 한 줄의 내용(스칼라·열거 값 문자열)."""
    return {"run_id": tracks.run_id, "source_id": tracks.source_id, "image_seq": tracks.image_seq,
            "capture_generation": tracks.capture_generation, "generation": tracks.generation, "job_seq": tracks.job_seq,
            "detector_config_id": tracks.detector_config_id, "timestamp_ms": tracks.timestamp_ms,
            "mode": tracks.mode.value, "track_namespace": tracks.track_namespace, "tracker_epoch": tracks.tracker_epoch,
            "updated": tracks.updated, "event": tracks.event,
            "tracks": [{"instance_index": a.instance_index, "part_track_id": a.part_track_id, "state": a.state.value,
                        "hits": a.hits, "reason": a.reason} for a in tracks.assignments]}


def inputs_from_ai(result: AiSegResult) -> list[TrackInput]:
    """AI 표시 결과 → 추적 입력(D8). 원래 검출·거부를 바꾸지 않고 스칼라만 복사한다.

    후보 = 대상 범위(`in_scope`: 대상 클래스·점수 통과)이고 기하 거부가 없는 인스턴스(잘림은 후보이되 `clipped`).
    그 밖: 비대상·점수 미달 → `not_in_scope`, `EMPTY_MASK`·`SMALL_MASK` → `geometry_rejected`.
    """
    if not isinstance(result, AiSegResult):
        raise TrackingContractError(f"inputs_from_ai needs AiSegResult: {type(result).__name__}")

    out = []

    for inst in result.instances:
        if not inst.in_scope:
            candidate, reason = False, "not_in_scope"
        elif inst.rejected in _GEOMETRY_REJECTIONS:
            candidate, reason = False, "geometry_rejected"
        else:
            candidate, reason = True, None

        out.append(TrackInput(index=int(inst.index), class_name=str(inst.class_name), confidence=float(inst.confidence),
                              box_xyxy=tuple(float(v) for v in inst.box_xyxy), clipped=bool(inst.clipped),
                              candidate=candidate, untracked_reason=reason))

    return out


@dataclass
class _Track:
    """트랙 하나(스칼라만)."""

    cls: str
    cx: float
    cy: float
    w: float
    h: float
    last_hit_ms: int
    clean: int = 1          # 잘리지 않은 연속 hit(확정 근거)
    seen: int = 1           # 전체 hit(속도 추정용)
    vx: float = 0.0
    vy: float = 0.0
    part_track_id: int | None = None

    def predict(self, t_ms: int) -> tuple[float, float]:
        """t 시각 예측 중심."""
        dt = t_ms - self.last_hit_ms
        return self.cx + self.vx * dt, self.cy + self.vy * dt

    @property
    def diag(self) -> float:
        """박스 대각선."""
        return math.hypot(self.w, self.h)

    @property
    def area(self) -> float:
        """박스 넓이."""
        return self.w * self.h


def _center(inp: TrackInput) -> tuple[float, float, float, float]:
    """(cx, cy, w, h)."""
    x1, y1, x2, y2 = inp.box_xyxy
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


class PartTracker:
    """인스턴스 추적 상태기(단일 소유: A 제어 흐름). 계약은 모듈 설명 참조."""

    def __init__(self, issuer: PartTrackIdIssuer, *, mode: TrackMode, config: TrackerConfig = DEFAULT_TRACKER_CONFIG) -> None:
        """주입된 발급기·모드(M1/M2)·설정. 발급기가 없거나 모드가 틀리면 계약 오류(D11)."""
        if not isinstance(issuer, PartTrackIdIssuer):
            raise TrackingContractError("PartTracker needs an injected PartTrackIdIssuer (one per process)")

        if mode not in (TrackMode.HEURISTIC, TrackMode.CONFIRMED_STATIC):
            raise TrackingContractError(f"mode must be HEURISTIC or CONFIRMED_STATIC (OFF = no tracker): {mode!r}")

        if not isinstance(config, TrackerConfig):
            raise TrackingContractError("config must be TrackerConfig")

        self._issuer, self._mode, self._config = issuer, TrackMode(mode), config
        self._epoch = issuer.next_tracker_epoch()
        self._tracks: list[_Track] = []
        self._scope: tuple[str, str, str] | None = None
        self._generation: int | None = None
        self._last_ts: int | None = None
        self._disabled: str | None = None
        self._stats = {"updates": 0, "not_updated": {}, "tentative_dropped": 0, "ids_retired": 0, "resets": {},
                       "events": {}}

    # ------------------------------------------------------------------ 조회

    @property
    def mode(self) -> TrackMode:
        """추적 모드."""
        return self._mode

    @property
    def tracker_epoch(self) -> int:
        """현재 epoch."""
        return self._epoch

    @property
    def disabled(self) -> bool:
        """disable 되었는가."""
        return self._disabled is not None

    @property
    def track_namespace(self) -> str:
        """발급기 namespace."""
        return self._issuer.namespace

    def stats(self) -> dict:
        """계수 스냅샷(복사)."""
        s = self._stats
        active = sum(1 for t in self._tracks if t.last_hit_ms == self._last_ts)

        return {"updates": s["updates"], "not_updated": dict(s["not_updated"]), "tentative_dropped": s["tentative_dropped"],
                "ids_retired": s["ids_retired"], "resets": dict(s["resets"]), "events": dict(s["events"]),
                "tracks": len(self._tracks), "active": active, "lost": len(self._tracks) - active,
                "issued": self._issuer.issued, "tracker_epoch": self._epoch, "disabled": self._disabled}

    # ------------------------------------------------------------------ 제어

    def reset(self, reason: str) -> None:
        """전 트랙 종료(은퇴 계수)·새 epoch·시각 기준 해제. 경계(세대 전이·입력 사건·종료)에서 실행기가 부른다."""
        if not isinstance(reason, str) or not reason:
            raise TrackingContractError(f"reset reason must be a non-empty string: {reason!r}")

        self._terminate_all()
        self._scope = self._generation = self._last_ts = None
        self._stats["resets"][reason] = self._stats["resets"].get(reason, 0) + 1

    def disable(self, reason: str) -> None:
        """추적 끔(추적 예외 격리, D10). 트랙을 모두 끝내고 이후 프레임은 untracked(`tracking_disabled`). 재호출은 무동작(첫 사유 유지)."""
        if self._disabled is not None:
            return

        self._terminate_all()
        self._disabled = str(reason) or "disabled"

    def update(self, key: FrameKey, timestamp_ms: int, inputs, *, camera_static: bool | None = None) -> TrackFrame:
        """프레임 하나를 반영해 TrackFrame을 돌려준다. 입력 오류는 TrackingInputError(상태 무변)."""
        inputs = self._validated(key, timestamp_ms, inputs)

        if self._mode is TrackMode.HEURISTIC and camera_static is not None:
            raise TrackingContractError("camera_static is only for CONFIRMED_STATIC mode (do not fake M1 in M2)")

        if self._disabled is not None:
            return self._not_updated(key, timestamp_ms, inputs, "tracking_disabled")

        # 범위·세대(§4.1): run·source·설정이 바뀌거나 세대가 오르면 전 종료, 이전 세대는 갱신 안 함
        scope = (key.run_id, key.source_id, key.detector_config_id)
        event = None

        if self._scope is not None and scope == self._scope and key.generation < self._generation:
            return self._not_updated(key, timestamp_ms, inputs, "old_generation")

        if self._scope is not None and (scope != self._scope or key.generation > self._generation):
            self.reset("scope_changed")
            event = "scope_changed"

        if self._last_ts is not None and timestamp_ms <= self._last_ts:
            return self._not_updated(key, timestamp_ms, inputs, "stale_frame")

        self._scope, self._generation, self._last_ts = scope, key.generation, timestamp_ms
        self._stats["updates"] += 1

        # M1: 정지 확인 신호가 True가 아니면 연결하지 않는다(전 종료, 트랙 생성 없음)
        if self._mode is TrackMode.CONFIRMED_STATIC and camera_static is not True:
            self._terminate_all()

            self._count_event("camera_not_static")
            out = [self._untracked(i, "camera_not_static" if i.candidate else i.untracked_reason) for i in inputs]
            return self._frame(key, timestamp_ms, True, "camera_not_static", out)

        cfg = self._config

        # 만료: 마지막 hit 뒤 max_gap_ms 초과
        for t in [t for t in self._tracks if timestamp_ms - t.last_hit_ms > cfg.max_gap_ms]:
            self._retire(t)

        cands = [i for i in inputs if i.candidate]

        # M2 카메라 이동 의심(보장 없음)
        if self._mode is TrackMode.HEURISTIC and self._camera_motion_suspected(cands, timestamp_ms):
            self._terminate_all()
            event = "camera_motion_suspected"
            self._count_event(event)

        result = self._associate(cands, timestamp_ms)
        out = [result[i.index] if i.candidate else self._untracked(i, i.untracked_reason) for i in inputs]

        return self._frame(key, timestamp_ms, True, event, out)

    # ------------------------------------------------------------------ 내부

    def _validated(self, key, timestamp_ms, inputs) -> list[TrackInput]:
        """모든 입력을 상태 변경 전에 검사한다."""
        if not isinstance(key, FrameKey):
            raise TrackingInputError(f"key must be FrameKey: {type(key).__name__}")

        for name in ("run_id", "source_id", "detector_config_id"):
            if not isinstance(getattr(key, name), str) or not getattr(key, name):
                raise TrackingInputError(f"key.{name} must be a non-empty string")

        for name in ("image_seq", "capture_generation", "generation", "job_seq"):
            if type(getattr(key, name)) is not int or getattr(key, name) < 0:
                raise TrackingInputError(f"key.{name} must be an int >= 0")

        if type(timestamp_ms) is not int or timestamp_ms < 0:
            raise TrackingInputError(f"timestamp_ms must be an int >= 0: {timestamp_ms!r}")

        inputs = list(inputs)
        seen = set()

        for i in inputs:
            if not isinstance(i, TrackInput):
                raise TrackingInputError(f"inputs must be TrackInput: {type(i).__name__}")

            if type(i.index) is not int or i.index < 0 or i.index in seen:
                raise TrackingInputError(f"instance index must be a unique int >= 0: {i.index!r}")

            seen.add(i.index)

            if not isinstance(i.class_name, str) or not i.class_name:
                raise TrackingInputError(f"class_name must be a non-empty string: {i.class_name!r}")

            conf = i.confidence
            if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not math.isfinite(conf) or not 0 <= conf <= 1:
                raise TrackingInputError(f"confidence must be a finite number in [0, 1]: {conf!r}")

            box = i.box_xyxy
            if (not isinstance(box, tuple) or len(box) != 4
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)
                    or not (box[2] >= box[0] and box[3] >= box[1])):
                # #196 A3-1: AiInstance와 같은 규칙(x1 <= x2, y1 <= y2). 폭·높이 0 박스(1열 마스크)로 추적이 꺼지지 않게 한다.
                # 정합 계산은 넓이 비 하한(1e-9)·반경 하한(gate_min_px)이 있어 0 크기에서도 나눗셈 오류가 없다
                raise TrackingInputError(f"box_xyxy must be 4 finite numbers with x2>=x1, y2>=y1: {box!r}")

            if type(i.clipped) is not bool or type(i.candidate) is not bool:
                raise TrackingInputError("clipped and candidate must be bool")

        return inputs

    def _camera_motion_suspected(self, cands: list[TrackInput], t_ms: int) -> bool:
        """M2 휴리스틱: 여러 트랙이 같은 방향·크기로 크게 튀었는가(설정 설명 참조)."""
        cfg = self._config

        if len(self._tracks) < cfg.camera_min_tracks:
            return False

        moves = []

        for t in self._tracks:
            px, py = t.predict(t_ms)
            best = None

            for c in cands:
                if c.class_name != t.cls:
                    continue

                cx, cy, _, _ = _center(c)
                d = math.hypot(cx - px, cy - py)

                if d <= cfg.camera_search_px and (best is None or d < best[0]):
                    best = (d, cx - px, cy - py)

            if best is not None:
                moves.append(best)

        if len(moves) < cfg.camera_min_tracks or any(d <= cfg.camera_jump_px for d, _, _ in moves):
            return False

        mx = sorted(m[1] for m in moves)[len(moves) // 2]
        my = sorted(m[2] for m in moves)[len(moves) // 2]

        return all(math.hypot(dx - mx, dy - my) <= cfg.camera_coherence_px for _, dx, dy in moves)

    def _associate(self, cands: list[TrackInput], t_ms: int) -> dict[int, TrackAssignment]:
        """연결·모호성·생성. 후보 index → 판정."""
        cfg = self._config
        tracks = self._tracks
        preds = [t.predict(t_ms) for t in tracks]
        geo = {c.index: _center(c) for c in cands}

        # 적격 쌍(같은 클래스, 반경 안, 넓이 비)
        elig: dict[int, list[tuple[float, int]]] = {c.index: [] for c in cands}
        by_track: dict[int, list[tuple[float, int]]] = {k: [] for k in range(len(tracks))}
        other_class_near: set[int] = set()

        for c in cands:
            cx, cy, w, h = geo[c.index]

            for k, t in enumerate(tracks):
                gate = max(cfg.gate_min_px, cfg.gate_box_ratio * t.diag)
                d = math.hypot(cx - preds[k][0], cy - preds[k][1])

                if d > gate:
                    continue

                if c.class_name != t.cls:
                    other_class_near.add(c.index)
                    continue

                ratio = max(w * h, t.area) / max(min(w * h, t.area), 1e-9)

                if ratio > cfg.size_ratio_max:
                    continue

                cost = d / gate
                elig[c.index].append((cost, k))
                by_track[k].append((cost, c.index))

        # 모호성: 검출 하나에 트랙 둘(합쳐짐·교차), 트랙 하나에 검출 둘(분리)
        ambiguous: set[int] = set()
        ended: set[int] = set()

        for ci, lst in elig.items():
            lst.sort()
            close = [k for cost, k in lst if cost - lst[0][0] < cfg.ambiguity_margin] if lst else []

            if len(close) >= 2:
                ambiguous.add(ci)
                ended.update(close)

        for k, lst in by_track.items():
            lst.sort()
            close = [ci for cost, ci in lst if cost - lst[0][0] < cfg.ambiguity_margin] if lst else []

            if len(close) >= 2:
                ambiguous.update(close)
                ended.add(k)

        # 탐욕 1:1 배정(모호·종료 제외)
        pairs = sorted((cost, ci, k) for ci, lst in elig.items() if ci not in ambiguous
                       for cost, k in lst if k not in ended)
        matched_c: dict[int, int] = {}
        matched_t: set[int] = set()

        for cost, ci, k in pairs:
            if ci in matched_c or k in matched_t:
                continue

            matched_c[ci] = k
            matched_t.add(k)

        out: dict[int, TrackAssignment] = {}
        by_index = {c.index: c for c in cands}

        for ci, k in matched_c.items():
            out[ci] = self._hit(tracks[k], by_index[ci], geo[ci], t_ms)

        # 종료·미배정 트랙 정리
        survivors = []

        for k, t in enumerate(tracks):
            if k in ended:
                self._retire_counted(t)
            elif k in matched_t or t.part_track_id is not None:
                survivors.append(t)        # confirmed 미배정은 lost로 남김
            else:
                self._stats["tentative_dropped"] += 1

        self._tracks = survivors

        for ci in ambiguous:
            out[ci] = TrackAssignment(ci, None, TrackState.AMBIGUOUS, 0, "ambiguous")

        # 새 트랙
        for c in cands:
            if c.index in out:
                continue

            if c.confidence < cfg.new_track_min_conf:
                out[c.index] = TrackAssignment(c.index, None, TrackState.UNTRACKED, 0, "low_confidence")
                continue

            if len(self._tracks) >= cfg.max_tracks and not self._retire_oldest_lost(t_ms):
                out[c.index] = TrackAssignment(c.index, None, TrackState.UNTRACKED, 0, "track_capacity")
                continue

            cx, cy, w, h = geo[c.index]
            t = _Track(cls=c.class_name, cx=cx, cy=cy, w=w, h=h, last_hit_ms=t_ms, clean=0 if c.clipped else 1)

            if cfg.min_hits == 1 and not c.clipped:
                t.part_track_id = self._issuer.next_id()

            self._tracks.append(t)
            reason = "class_changed" if c.index in other_class_near else ("clipped" if c.clipped else None)
            state = TrackState.CONFIRMED if t.part_track_id is not None else TrackState.TENTATIVE
            out[c.index] = TrackAssignment(c.index, t.part_track_id, state, t.clean, reason)

        return out

    def _hit(self, t: _Track, c: TrackInput, g, t_ms: int) -> TrackAssignment:
        """배정된 검출로 트랙을 갱신한다. 잘리지 않은 연속 hit가 min_hits에 닿으면 번호 발급, 잘림 hit는 연속을 0으로 끊는다."""
        cx, cy, w, h = g
        dt = t_ms - t.last_hit_ms

        if dt > 0:
            vx, vy = (cx - t.cx) / dt, (cy - t.cy) / dt
            a = 1.0 if t.seen == 1 else self._config.velocity_alpha
            t.vx, t.vy = a * vx + (1 - a) * t.vx, a * vy + (1 - a) * t.vy

        t.cx, t.cy, t.w, t.h, t.last_hit_ms = cx, cy, w, h, t_ms
        t.seen += 1
        t.clean = 0 if c.clipped else t.clean + 1

        if t.part_track_id is None and t.clean >= self._config.min_hits:
            t.part_track_id = self._issuer.next_id()

        state = TrackState.CONFIRMED if t.part_track_id is not None else TrackState.TENTATIVE

        return TrackAssignment(c.index, t.part_track_id, state, t.clean, "clipped" if c.clipped else None)

    def _retire_oldest_lost(self, t_ms: int) -> bool:
        """가장 오래된 lost 트랙 하나를 은퇴시킨다. 없으면 False."""
        lost = [t for t in self._tracks if t.last_hit_ms != t_ms]

        if not lost:
            return False

        self._retire(min(lost, key=lambda t: t.last_hit_ms))

        return True

    def _retire(self, t: _Track) -> None:
        """트랙 하나 은퇴(목록에서 제거, 계수)."""
        self._tracks.remove(t)
        self._retire_counted(t)

    def _retire_counted(self, t: _Track) -> None:
        """은퇴 계수만 올린다(목록 정리는 호출자)."""
        if t.part_track_id is not None:
            self._stats["ids_retired"] += 1
        else:
            self._stats["tentative_dropped"] += 1

    def _terminate_all(self) -> bool:
        """전체 종료: 트랙이 하나라도 있으면 모두 은퇴시키고 새 epoch를 받는다(True). 트랙이 없으면 무동작(False, epoch 그대로)."""
        if not self._tracks:
            return False

        for t in self._tracks:
            self._retire_counted(t)

        self._tracks = []
        self._epoch = self._issuer.next_tracker_epoch()

        return True

    def _count_event(self, name: str) -> None:
        """event 계수."""
        self._stats["events"][name] = self._stats["events"].get(name, 0) + 1

    @staticmethod
    def _untracked(i: TrackInput, reason: str | None) -> TrackAssignment:
        """untracked 판정."""
        return TrackAssignment(i.index, None, TrackState.UNTRACKED, 0, reason)

    def _not_updated(self, key, timestamp_ms, inputs, reason: str) -> TrackFrame:
        """갱신하지 않은 프레임: 모든 인스턴스 untracked(사유), 이전 번호를 싣지 않는다."""
        n = self._stats["not_updated"]
        n[reason] = n.get(reason, 0) + 1

        return self._frame(key, timestamp_ms, False, None, [self._untracked(i, reason) for i in inputs])

    def _frame(self, key: FrameKey, timestamp_ms: int, updated: bool, event, out) -> TrackFrame:
        """TrackFrame 생성."""
        return TrackFrame(run_id=key.run_id, source_id=key.source_id, image_seq=key.image_seq,
                          capture_generation=key.capture_generation, generation=key.generation, job_seq=key.job_seq,
                          detector_config_id=key.detector_config_id, timestamp_ms=timestamp_ms, mode=self._mode,
                          track_namespace=self._issuer.namespace, tracker_epoch=self._epoch, updated=updated, event=event,
                          assignments=tuple(out))


__all__ = ["DEFAULT_TRACKER_CONFIG", "FrameKey", "PartTrackIdIssuer", "PartTracker", "TrackAssignment", "TrackFrame",
           "TrackInput", "TrackMode", "TrackState", "TrackerConfig", "TrackingContractError", "TrackingInputError",
           "inputs_from_ai", "matches_job", "track_record"]
