"""실시간 A→B 관측 기록 연결(#173 다음 단계): 실행기의 실시간 terminal 응답을 전용 디스패처로 B에 넘기고 결과를 기록만 한다.

근거: reviewer 착수 명세(2026-11-18, 준비 코드 ad932da + 리셋 = 새 실행 결정), `_workspace/171_builder2_live_a_to_b_plan.md` r3 §3,
`_workspace/173_builder2_reset_new_run_report.md` §2·§8. **기록 전용 시험 연결**이다: `stage.on_output`을 쓰지 않으며 C·D 송신·로봇
목표 발행이 없다. D13은 그대로다(실행기는 실시간 `b_consumer`를 계속 거부하고, 이 연결부는 `live_observation_sink` 자리만 쓴다).

흐름
- 실행기 제어 흐름: `offer(outcome, skip_reason, rx_epoch)`. **TF 조회·디스크 쓰기를 하지 않는다**(r1 F1): 상태 비교·대기 칸 넣기·
  메모리 큐 넣기만 한다. `skip_reason`(실행기 `b_skip_reason`: 무효 관측·세대 폐기·보정 불일치·수신 epoch 지남 등)이 있으면 그 사유를
  건너뜀 큐에 넣고 넘기지 않는다. 열려 있지 않으면 그 프레임을 열기 시도 칸(1개, 최신 우선)에 넣고 그 프레임은 `not_ready`로
  건너뛴다. 열려 있으면 디스패처 대기 칸 1개(최신 우선)에 넣는다. 검사 ①②③·잠금 규율은 `LiveObservationDispatcher`·
  `LiveObservationState` 그대로다.
- 열기(새 실행): 상태는 `start_closed`. 수신 경계(CameraInfo 첫 수신 등)가 코어에 반영되면 `on_live_input`이 그 경계의 보정 번호·세대로
  `on_applied`를 부른다. **열기 스레드**(`eyetoarm-live-observation-open`)가 시도 칸의 프레임으로 그 경계의 촬영 stamp TF 정확 조회를
  하고, 성공하면 `on_tf_ready`로 연다(버퍼는 비우지 않음). 열기에 쓴 프레임 자체는 계산하지 않는다(다음 프레임부터).
- 건너뜀 기록: 메모리 큐에 넣고 **기록 스레드**(`eyetoarm-live-observation-skips`)가 `skipped.jsonl`에 쓴다. 쓰기 실패는
  `skip_write_error`로 남고 진입점은 종료 1로 본다.
- 건너뜀 대기 상한(r2): 대기 + 쓰는 중 기록 수가 `skip_queue_capacity`(시험 설정, 필수)에 이르면 그 뒤 기록은 넣지 않고 누락으로 센다.
  처음 포화되거나 쓰기가 실패하면 상태를 fail-stop한다(`skip_queue_overflow`·`skip_writer_error`: 이후 제출·계산·수락 0). 제어 흐름은
  기다리지 않는다(넣기 실패는 계수만). 진입점은 상태 실패를 보고 실행을 멈추고 종료 1로 끝낸다. 이미 큐에 들어간 기록은 순서대로 쓰고
  (먼저 들어온 것 보존), 쓰기 실패 뒤에는 더 넣지 않는다. 계수: `skip_total = skip_written + skip_dropped + skip_unwritten`.
- B 계산 조건: 정확 촬영 stamp TF가 `ok`일 때만 계산·주석한다. TF 누락(`unavailable`)·상한 초과(`late`, 상한은 설정값)·stamp
  불일치·stamp 없음/무효는 계산 없이 사유를 기록한다(디스패처 `refuse_tf_statuses`). 봉투 오류(보정 번호·크기·프레임 불일치)는
  `b_input_rejected`로 기록한다. 주석은 같은 `BOutput`·같은 프레임 영상으로 `render_center`를 그린다.
- 리셋: 소스 경계 훅에서 `sim_time_rewind`이면 상태를 즉시 닫는다(`run_end`: 이전 결과 철회·계산 중 결과 수락 차단·제출 거부).
  `reset_pending`이 참이 되며, 진입점은 실행을 멈추고 `LiveRunController.service()`로 정리한다. 자동 재시작은 없다.
- 정리: `end`/`retry_cleanup`은 상태 닫기(1회) → 디스패처 → 기록기 → TF → 건너뜀 기록 순서, 실패 단계만 재시도.
  `cleanup_confirmed`가 참이어야 `LiveRunController.start_run`이 다음 실행을 허용한다.
- 기록: 출력 폴더(새로 만듦)에 `b_live_events.jsonl`(공통 순번, 기존 `LiveObservationEventWriter`)·`annotated/`·`skipped.jsonl`(건너뜀
  사유)·`summary.json`.
- 운영값(TF 상한·용량·종료 상한 등)은 아직 미확정이라 설정 파일에 `"status": "test_only"`로 명시해야 한다(기본값 없음).
- 단계 L(#201, `b_input="list"`): 실행기가 `b_list_skip_reason`으로 판정한 프레임을 같은 프레임 단위 검사(종료·epoch·준비·보정·세대)
  뒤 대기 칸에 `LiveListInput(context, display)`로 넣는다. worker가 촬영 stamp TF를 한 번 조회해 그 TF 하나로 `FrameObservations`·
  목록 봉투(`eyetoarm.b_input/2`)를 만들고 목록 소비자(`list_consumer_factory(stage)` 또는 `stage.process_frame`)를 부른다. 소비자를 못
  구하면 RuntimeError(worker 오류, 단계 T로 바꾸지 않음). 목록 봉투 오류(ValueError 계열)는 `b_input_rejected`. 주석 영상은 만들지
  않는다(B 후보 결과 형식 U-3 전). 단일 관측 경로(`"single"`, 기본)는 바꾸지 않는다.
- C 목표 송신(#214 D13 해제, 선택): `sort_part_runtime`을 주면(설정 `sort_part` 필수) 검사 ③이 수락한 결과의 C 목표를
  `LiveSortPartSender`가 B `SortPartClient`로 보낸다(`vision/live_sort_part`). 단계 L(#233)은 B `process_frame`이 고른 1개(`chosen`)만
  보낸다: worker가 소비자를 부른 직후 `stage.last_frame`에서 반환 사전 `goal`과 같은 객체를 가진 결과를 찾아 그 봉투와 짝으로 두고, 수락
  훅은 같은 봉투일 때만 그 결과를 송신기에 넘긴다(사진당 최대 1회). 고른 것이 없으면 송신하지 않고 `list_no_selection`만 센다.
  못 찾으면(반환 계약 위반) `list_goal_unmatched`로 세고 진입점이 종료 1로 드러낸다. `stage.last`는 쓰지 않는다(고른 것이 없는
  프레임에 앞 프레임 값이 남음). `stage.on_output`은 여전히 None이어야 한다(수락 전 송신
  경로를 만들지 않음). 실행 닫힘(리셋·종료·fail-stop)에 송신 게이트를 먼저 닫고, 정리 단계 `sort_part`는 dispatcher 다음이다. 실행기
  D13 가드(실시간 `b_consumer` 거부)는 그대로다. 주지 않으면 기록·요약이 기존과 같다.
- 추적 번호(#215, 선택 `tracking=True`, 진입점 `--tracking`): 실행기가 `offer(..., tracks=TrackFrame | None)`로 같은 작업의 추적 결과를
  넘기면 확정 번호만(`vision/tracked_observation.confirmed_track_id`) A 하위 형 관측(`TrackedPlanarObservation`)과 봉투 확장
  `tracking`에 싣는다(공통형·B 파일 무변, B는 관측의 `part_track_id`로 `_dedupe`). B 중복 방지 기록 해제는 `release_proxy`(이 연결부가
  만든 모든 B 단계에 `release`)가 맡는다: B 클라이언트 결과·`no_server`·`busy`, 검사 ③ 거부(`on_not_accepted`, `PICK_PLACE`만), 송신기가
  클라이언트를 부르지 않은 경우, 송신이 꺼진 기록 전용 실행의 수락 뒤. `ALREADY_SENT` SKIP은 풀지 않는다. 단계 T만.
- W 추적(#217, 선택 `world_tracker`·`world_config`, 진입점 `--tracking-space world`): 실행기 픽셀 추적 대신 이 연결부가 추적
  스레드(`vision/live_world_tracking`)를 둔다. `offer`가 AI 결과가 있는 현재 응답(건너뜀 사유 없음·`invalid_observation`)을 추적 큐에
  넣고(제어 흐름 TF 조회 없음), worker는 봉투를 만들기 직전 그 작업 결과를 상한까지 기다려 #215와 같은 규칙(확정 번호만)으로 싣는다.
  실행 닫힘·보정/프로필 경계에서 추적기 초기화를 요청한다. 정리 단계 `world_tracking`은 dispatcher 다음(TF를 닫기 전).
- ROS·Qt를 import하지 않는다(TF 수신 묶음·B 단계·송신 런타임은 주입).
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from eyetoarm.common.types import CaptureInfo
from eyetoarm.transforms.live_tf_buffer import TfRewindReadiness
from eyetoarm.vision.ai_convert import frame_observations_from_ai
from eyetoarm.vision.list_envelope import make_list_envelope
from eyetoarm.vision.live_observation_dispatch import LiveObservationDispatcher
from eyetoarm.vision.live_observation_envelope import LiveCaptureTfLookup, make_live_envelope
from eyetoarm.vision.live_observation_events import LiveObservationEventWriter
from eyetoarm.vision.live_observation_state import (SIM_TIME_REWIND, LiveObservationConfig, LiveObservationState)
from eyetoarm.vision.live_sort_part import LiveSortPartSender, SortPartConfig, SortPartConfigError, parse_sort_part
from eyetoarm.vision.tracked_observation import TrackedPlanarObservation, confirmed_track_id, with_track
from eyetoarm.vision.ai_result import AiSegResult
from eyetoarm.vision.live_world_tracking import LiveWorldTracking

WORLD_TRACKABLE_SKIPS = (None, "invalid_observation")   # #217: 추적 큐에 넣는 실행기 판정(그 밖은 현재 응답이 아님)
TRACK_PENDING = "track_pending"
from eyetoarm.vision.view_bias import apply_view_bias

PROFILE_CHANGED = "profile_changed"             # `inputs.live_slot.BoundaryKind.PROFILE_CHANGED` 값(#180 r1 L1)
BOUNDARY_HOOK_FAILED = "boundary_hook_failed"   # `inputs.live_slot.NoticeKind.BOUNDARY_HOOK_FAILED` 값(#180 #5, 입력 모듈 import 없이 비교)

LIVE_OBSERVATION_CONFIG_STATUS = "test_only"
NON_OK_TF = frozenset({"unavailable", "late", "stamp_mismatch", "no_stamp", "invalid_stamp"})
_CONFIG_KEYS = {"status", "world_frame", "tf_wait_limit_s", "static_pairs", "future_tolerance_ns", "write_capacity",
                "reserved_capacity", "lag_warn_s", "close_limit_s", "skip_queue_capacity", "note"}
_OPTIONAL_KEYS = {"sort_part"}   # #214 송신을 켤 때만(진입점이 옵션과 짝을 확인)


B_INPUT_SINGLE = "single"
B_INPUT_LIST = "list"
_UNMATCHED = object()   # #233: 고른 목표를 결과 묶음에서 못 찾음 표시


@dataclass(frozen=True)
class LiveListInput:
    """단계 L 대기 칸 항목의 관측 자리(#201): 작업 문맥과 AI 표시 결과. 건너뜀·기록 키(`job_seq`·`generation`)를 단일 관측과 같게 준다."""

    context: object
    display: object

    @property
    def job_seq(self):
        """작업 순번."""
        return self.context.job_seq

    @property
    def generation(self):
        """작업 세대."""
        return self.context.generation

    @property
    def image_seq(self):
        """프레임 순번."""
        return self.context.frame.image_seq


class _StageReleaseProxy:
    """B 클라이언트 `stage` 자리(#215): 연결부가 만든 모든 B 단계에 `release(번호)`를 부른다(프로필 변경으로 새로 만든 단계 포함)."""

    def __init__(self, link) -> None:
        """연결부(단계 목록 소유자)."""
        self._link = link
        self.calls = 0

    def release(self, part_track_id) -> None:
        """모든 B 단계의 중복 방지 기록에서 그 번호를 지운다(없으면 아무 일 없음)."""
        with self._link._lock:
            stages = [st[0] for st in self._link.stages]
            self.calls += 1

        for stage in stages:
            stage.release(part_track_id)


class LiveObservationConfigError(ValueError):
    """설정 파일 오류."""


@dataclass(frozen=True)
class LiveObservationLinkConfig:
    """실시간 관측 기록 연결 설정. 값은 모두 미확정 운영값의 **시험 설정**이다(`status`는 반드시 `test_only`)."""

    world_frame: str
    tf_wait_limit_s: float
    static_pairs: tuple
    future_tolerance_ns: int
    write_capacity: int
    reserved_capacity: int
    lag_warn_s: float
    close_limit_s: float
    skip_queue_capacity: int = 0   # 건너뜀 기록 대기 상한(쓰는 중 포함). 설정 파일에서는 필수(r2)
    sha256: str | None = None
    sort_part: SortPartConfig | None = None             # #214 선택 블록(없으면 C 송신 불가)


def _positive(value, name, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, int if integer else (int, float)) \
            or (not integer and not math.isfinite(value)) or value <= 0:
        raise LiveObservationConfigError(f"{name} must be a positive {'int' if integer else 'number'}: {value!r}")

    return value


def load_live_observation_config(path: str) -> LiveObservationLinkConfig:
    """설정 JSON을 읽고 검사한다. 모르는 키·누락·`status != test_only`는 오류."""
    raw = Path(path).read_bytes()

    try:
        doc = json.loads(raw)
    except ValueError as exc:
        raise LiveObservationConfigError(f"not JSON: {exc}") from exc

    if not isinstance(doc, dict):
        raise LiveObservationConfigError("top level must be an object")

    unknown = set(doc) - _CONFIG_KEYS - _OPTIONAL_KEYS
    missing = (_CONFIG_KEYS - {"note"}) - set(doc)

    if unknown or missing:
        raise LiveObservationConfigError(f"unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")

    if doc["status"] != LIVE_OBSERVATION_CONFIG_STATUS:
        raise LiveObservationConfigError(f"status must be {LIVE_OBSERVATION_CONFIG_STATUS!r} (operating values undecided)")

    if not isinstance(doc["world_frame"], str) or not doc["world_frame"]:
        raise LiveObservationConfigError("world_frame must be a non-empty string")

    pairs = doc["static_pairs"]

    if not isinstance(pairs, list) or not all(isinstance(p, list) and len(p) == 2 and all(isinstance(x, str) and x for x in p)
                                              for p in pairs):
        raise LiveObservationConfigError("static_pairs must be a list of [parent, child] string pairs")

    if isinstance(doc["future_tolerance_ns"], bool) or not isinstance(doc["future_tolerance_ns"], int) \
            or doc["future_tolerance_ns"] < 0:
        raise LiveObservationConfigError("future_tolerance_ns must be an int >= 0")

    sort_part = None

    if "sort_part" in doc:
        try:
            sort_part = parse_sort_part(doc["sort_part"])
        except SortPartConfigError as exc:
            raise LiveObservationConfigError(str(exc)) from exc

    return LiveObservationLinkConfig(
        world_frame=doc["world_frame"], tf_wait_limit_s=float(_positive(doc["tf_wait_limit_s"], "tf_wait_limit_s")),
        static_pairs=tuple(tuple(p) for p in pairs), future_tolerance_ns=doc["future_tolerance_ns"],
        write_capacity=_positive(doc["write_capacity"], "write_capacity", integer=True),
        reserved_capacity=_positive(doc["reserved_capacity"], "reserved_capacity", integer=True),
        lag_warn_s=float(_positive(doc["lag_warn_s"], "lag_warn_s")),
        close_limit_s=float(_positive(doc["close_limit_s"], "close_limit_s")),
        skip_queue_capacity=_positive(doc["skip_queue_capacity"], "skip_queue_capacity", integer=True),
        sha256=hashlib.sha256(raw).hexdigest(), sort_part=sort_part)


def _tracking_extension(frame, obs):
    """봉투 확장 `tracking`(#215, 추적 하위 형 관측일 때만). 결합 키는 봉투 검사 규칙과 같다."""
    if not isinstance(obs, TrackedPlanarObservation):
        return None

    return {"tracking": {"binding": {"image_seq": obs.image_seq, "job_seq": obs.job_seq, "generation": obs.generation,
                                     "capture_stamp_ns": frame.ros_stamp_ns},
                         "part_track_id": obs.part_track_id}}


def _join(thread: threading.Thread | None, limit_s: float) -> bool:
    """스레드를 상한까지 기다린다. 시작 안 했으면 True."""
    if thread is None:
        return True

    thread.join(timeout=limit_s)

    return not thread.is_alive()


class LiveObservationLink:
    """실행 하나의 실시간 관측 기록 연결. `LiveRunController`의 세션 규약(`run_id`·`ended`·`reset_pending`·`reset_reason`·
    `cleanup_confirmed`·`end`·`retry_cleanup`)을 따른다."""

    def __init__(self, run_id: str, out_dir: str, *, config: LiveObservationLinkConfig, tf, stage_factory: Callable,
                 render: Callable | None, clock_ns: Callable[[], int] = time.monotonic_ns,
                 b_input: str = B_INPUT_SINGLE, list_consumer_factory: Callable | None = None,
                 view_bias=None, sort_part_runtime=None, tracking: bool = False,
                 world_tracker=None, world_config=None, world_plane_z_m: float | None = None,
                 list_consumer_kind: str | None = None, run_outputs: tuple | None = None) -> None:
        """출력 폴더(새로)·상태(start_closed)·기록기·디스패처·TF 준비를 만든다.

        - `b_input`(#201): `"single"`(기본, 단계 T 봉투 1) | `"list"`(단계 L 목록 봉투 2). `"list"`면 `render`는 쓰지 않는다.
        - `list_consumer_factory(stage)`(#201, 시험 주입): 목록 봉투를 받는 함수. None이면 `stage.process_frame`.
        - `run_outputs`(#233 r1, 선택): 이 실행의 실제 출구 이름 목록(진입점이 C 연결부·B center·C 목표를 나열). 주면 요약
          `record_only`는 목록이 비었을 때만 참이고 `outputs`는 그 목록이다. None이면 기존 규칙(송신기 유무).
        - `list_consumer_kind`(#233, 선택): 운영 소비자(B center 감싸개 등)를 줄 때 기록할 소비자 이름. None이면 기존 이름 규칙.
        - `view_bias`(#207): `ViewBiasConfig` 또는 None(기본, 꺼짐). 켜면 디스패처의 공통 지점(봉투 생성 직후)에서 단계 T·L 끝점을
          보정하고 accept 기록·요약에 보정 필드를 남긴다. None이면 기록·요약이 기존과 같다.

        - `tf`: 시작된 TF 수신 묶음(`lookup(parent, child, stamp_ns, timeout_s)`·`has_static`·`close(limit_s)`).
        - `stage_factory(camera)` → `(BStage, ObservationEnvelopeAdapter)`. B 단계의 `on_output`은 None이어야 한다.
        - `render(stage, frame, out)` → PNG bytes | None(같은 프레임 주석).
        - `sort_part_runtime`(#214, 선택): C 송신 런타임(`send`·`stats`·`inflight_ids`·`server_seen`·`close`). 주면 `config.sort_part`가
          있어야 한다. 단계 T, 또는 단계 L(#233, 고른 1개만). 연결부가 소유해 정리 단계 `sort_part`에서 닫는다. None(기본)이면 송신 없음.
        - `tracking`(#215, 기본 False): 추적 번호 연결. 단계 T만. 끄면 `offer`의 `tracks`를 쓰지 않고 기록·요약이 기존과 같다.
        - `world_tracker`·`world_config`·`world_plane_z_m`(#217, 선택): W 추적 스레드. `tracking`이 참이어야 하고 단계 T만.
        """
        # 설정 검사는 파일을 열기 전에 한다(#180 #7: 검사 실패가 기록 파일 핸들을 남기지 않게)
        if type(config.skip_queue_capacity) is not int or config.skip_queue_capacity <= 0:
            raise ValueError(f"skip_queue_capacity must be a positive int: {config.skip_queue_capacity!r}")

        if b_input not in (B_INPUT_SINGLE, B_INPUT_LIST):
            raise ValueError(f"b_input must be {B_INPUT_SINGLE!r} or {B_INPUT_LIST!r}: {b_input!r}")

        if tracking and b_input != B_INPUT_SINGLE:
            raise ValueError("tracking link is for stage T (b_input single) only")

        if world_tracker is not None and (not tracking or world_config is None or world_plane_z_m is None):
            raise ValueError("world tracking requires tracking=True, world_config and world_plane_z_m")

        if sort_part_runtime is not None:
            if config.sort_part is None:
                raise ValueError("sort_part sending requires config block sort_part")

        self.b_input = b_input
        self.run_outputs = None if run_outputs is None else tuple(run_outputs)
        self._list_consumer_factory = list_consumer_factory
        self.consumer_kind = (None if b_input == B_INPUT_SINGLE else list_consumer_kind if list_consumer_kind is not None
                              else "injected_test_only" if list_consumer_factory is not None else "BStage.process_frame")
        self.list_consumer_calls = 0              # 목록 소비자 호출 수(worker만 씀, 잠금으로 읽음)
        self.list_candidates_sent = 0
        self.list_length_hist: dict[int, int] = {}
        self._list_chosen: tuple | None = None    # #233: (봉투, 고른 BOutput | None) worker 전용(같은 worker의 수락 훅이 읽음)
        self.list_no_selection = 0                # #233: 수락됐지만 B가 고른 목표가 없는 프레임(송신 없음)
        self.list_goal_unmatched = 0              # #233: 반환 `goal`을 결과 묶음에서 못 찾은 프레임(계약 위반)
        self.list_goals_offered = 0               # #233: 송신기에 넘긴 고른 목표 수

        self.run_id = run_id
        self.config = config
        self.tf = tf
        self.dir = Path(out_dir)
        self.state = LiveObservationState(run_id=run_id, generation=0, calibration_id=None, start_closed=True,
                                          config=LiveObservationConfig(write_capacity=config.write_capacity,
                                                                       reserved_capacity=config.reserved_capacity),
                                          clock_ns=clock_ns)
        self.writer = LiveObservationEventWriter(self.state, str(self.dir), lag_warn_s=config.lag_warn_s,
                                                 clock_ns=clock_ns)
        self._skip_fh = open(self.dir / "skipped.jsonl", "x", encoding="utf-8")
        self._skip_lock = threading.Lock()
        self._skip_cond = threading.Condition(self._skip_lock)
        self._skip_queue: list[dict] = []
        self._skip_capacity = config.skip_queue_capacity
        self._skip_inflight = 0                   # 기록 스레드가 꺼내 쓰는 중인 수
        self.skip_total = 0
        self.skip_written = 0
        self.skip_dropped = 0
        self.skip_max_depth = 0
        self.skip_overflow_at: int | None = None  # 처음 포화된 기록 번호(1부터)
        self._skip_stop = False
        self._skip_thread: threading.Thread | None = None
        self.skip_crash: str | None = None        # 기록 스레드의 예기치 않은 예외(#180 #6)
        self.skip_write_threads: set[str] = set()
        self._probe_cond = threading.Condition()
        self._probe: tuple | None = None          # (frame, epoch) 최신 우선 1칸
        self._probe_stop = False
        self._probe_busy = False
        self._probe_thread: threading.Thread | None = None
        self.probes_requested = 0
        self.probes_done = 0
        self.open_attempts: list[tuple] = []      # (image_seq, epoch, 결과)
        self._lock = threading.Lock()
        self._stage_factory = stage_factory
        self.stages: list = []
        capture = LiveCaptureTfLookup(tf.lookup, world_frame=config.world_frame, wait_limit_s=config.tf_wait_limit_s)

        if b_input == B_INPUT_LIST:
            list_kw = dict(build_envelope=self._build_list_envelope, compute=self._compute_list, render=None,
                           envelope_errors=(ValueError,))
        else:
            list_kw = dict(build_envelope=self._build_single,
                           compute=lambda st, env: st[1].consume(env).output,
                           render=None if render is None else (lambda st, f, out: render(st[0], f, out)))

        self.view_bias = view_bias
        bias_fn = None if view_bias is None else (lambda env, cam: apply_view_bias(env, cam, view_bias))
        self.tracking = bool(tracking)
        self.world = (None if world_tracker is None else
                      LiveWorldTracking(run_id, world_tracker, world_config, tf_lookup=capture, plane_z_m=world_plane_z_m,
                                        out_dir=str(self.dir), view_bias=view_bias))
        self._track_camera = None                  # #217: 추적 스레드에 넘길 현재 카메라(제어 흐름만 씀)
        self.release_proxy = _StageReleaseProxy(self) if self.tracking else None
        self.tracking_counts: dict[str, int] = {}   # 상태별 제출 수·해제 수(잠금 `_lock`)

        # 송신 기록 파일은 출력 폴더(기록기가 만듦) 안에 둔다
        self.sort_part = (None if sort_part_runtime is None else
                          LiveSortPartSender(run_id, config.sort_part, sort_part_runtime, str(self.dir),
                                             is_failed=lambda: self.state.failed is not None, tracking=self.tracking,
                                             release_target=self.release_proxy))
        hooked = self.sort_part is not None or self.tracking
        hook_kw = {} if not hooked else {"on_accepted": self._on_accepted}

        if self.tracking:
            hook_kw["on_not_accepted"] = self._on_not_accepted
        self.dispatcher = LiveObservationDispatcher(
            self.state, camera=None, stage_factory=self._make_stage, tf_lookup=capture,
            clock_ns=clock_ns, refuse_tf_statuses=NON_OK_TF, on_skip=self._worker_skip, view_bias=bias_fn,
            **hook_kw, **list_kw)
        self.readiness = TfRewindReadiness(self.state, clear=lambda: None, has_static=tf.has_static, lookup=tf.lookup,
                                           world_frame=config.world_frame, static_pairs=config.static_pairs,
                                           wait_limit_s=config.tf_wait_limit_s,
                                           future_tolerance_ns=config.future_tolerance_ns)
        self._calibration_id: str | None = None
        self.camera_cleared = False                # 프로필 없음 경계 뒤 카메라 「없음」 표시(#180 r1 L1)
        self.reset_reason: str | None = None
        self.ended = None
        self._cleanup_done: set[str] = set()
        self.cleanup_attempts = 0
        self.skips: dict[str, int] = {}
        self.offers = 0
        self.submitted = 0
        self.writer_report = None
        self.skip_write_error: str | None = None
        self.drained: bool | None = None   # 정상 종료 전 대기 작업 마무리 결과(진입점이 기록)
        self.tf_receiver_error: str | None = None   # TF 수신 스레드 실패 사유(#196 A1-2)

        # TF 수신 스레드가 죽으면 상태를 실패로 닫는다(영상 수신기 실패와 같은 fail-stop). 등록 함수가 없는 대역은 건너뛴다
        register = getattr(tf, "set_failure_callback", None)

        if callable(register):
            register(self._tf_receiver_failed)

    def _tf_receiver_failed(self, error: str) -> None:
        """TF 스레드에서 불린다: 사유를 남기고 상태 fail-stop(이후 제출·계산·수락 차단). 정리 확인은 `end`의 tf 단계가 막는다."""
        with self._lock:
            if self.tf_receiver_error is None:
                self.tf_receiver_error = error

        self.state.fail("tf_receiver_failed")
        self._close_outputs("tf_receiver_failed")

    def _build_single(self, frame, obs, camera, tf_result):
        """단계 T 봉투(worker). W 추적이면 그 작업의 추적 결과를 상한까지 기다려 번호를 싣는다(#217)."""
        if self.world is not None and isinstance(obs, TrackedPlanarObservation) and obs.track_state == TRACK_PENDING:
            obs = self._attach_world(frame, obs)

        return make_live_envelope(frame, obs, camera=camera, tf_result=tf_result, world_frame=self.config.world_frame,
                                  extensions=_tracking_extension(frame, obs))

    def _attach_world(self, frame, obs):
        """worker: 추적 스레드 결과로 확정 번호만 싣는다. 못 받으면 `track_pending`, 갱신 못 한 사유는 그 사유."""
        from types import SimpleNamespace

        ctx = SimpleNamespace(frame=frame, generation=obs.generation, job_seq=obs.job_seq,
                              detector_config_id=obs.detector_config_id)
        result = self.world.wait(ctx)

        if result is None:
            track_id, state = None, TRACK_PENDING
        elif isinstance(result, str):
            track_id, state = None, result
        else:
            track_id, state = confirmed_track_id(result, ctx, obs)

        self._count_tracking(f"world_{state}")

        return with_track(obs, track_id, state)

    def _count_tracking(self, key: str) -> None:
        with self._lock:
            self.tracking_counts[key] = self.tracking_counts.get(key, 0) + 1

    def _on_not_accepted(self, item, payload) -> None:
        """worker(#215): 검사 ③ 거부. B가 이 목표로 남긴 중복 방지 기록(`PICK_PLACE`일 때만)을 푼다. `ALREADY_SENT` SKIP은 풀지 않는다."""
        goal = getattr(payload.output, "goal", None) or {}
        track = goal.get("part_track_id")

        if goal.get("kind") != "PICK_PLACE" or track is None:
            return

        if self.sort_part is not None:
            obs = item.observation
            keys = {"run_id": self.run_id, "image_seq": getattr(item.frame, "image_seq", None),
                    "ros_stamp_ns": getattr(item.frame, "ros_stamp_ns", None), "accept_event_seq": None,
                    "kind": goal.get("kind"), "spec": goal.get("spec"), "job_seq": getattr(obs, "job_seq", None),
                    "generation": getattr(obs, "generation", None), "part_track_id": track}
            self.sort_part.release(track, "not_accepted", goal.get("goal_id"), keys)
        else:
            self.release_proxy.release(track)

        self._count_tracking("released_not_accepted")

    def _on_accepted(self, item, payload, event_seq) -> None:
        """worker 수락 훅: 추적 계수(#215)·C 송신(#214)."""
        obs = item.observation
        ids = {"job_seq": getattr(obs, "job_seq", None), "generation": getattr(obs, "generation", None)}

        if self.tracking:
            # #239: 송신기가 skip:no_track_id 줄에 남길 관측 추적 상태(그 밖의 기록 줄에는 넣지 않음)
            ids["track_state"] = getattr(obs, "track_state", None)
            goal = getattr(payload.output, "goal", None) or {}
            self._count_tracking("accepted_with_track" if goal.get("part_track_id") is not None else "accepted_without_track")

            if goal.get("kind") == "SKIP" and goal.get("reason") == "ALREADY_SENT":
                self._count_tracking("b_already_sent")

            # 송신이 꺼진 기록 전용 실행: 보내지 않은 목표의 B 기록을 바로 푼다(결과가 오지 않으므로)
            if self.sort_part is None and goal.get("kind") == "PICK_PLACE" and goal.get("part_track_id") is not None:
                self.release_proxy.release(goal["part_track_id"])
                self._count_tracking("released_record_only")

        if self.sort_part is not None and self.b_input == B_INPUT_LIST:   # #233: 고른 1개만
            self._offer_list_goal(item, payload, event_seq, ids)
        elif self.sort_part is not None:   # 송신 예외는 디스패처가 수락 훅 오류로 센다
            self.sort_part.offer(payload.output, item.frame, ids, event_seq)

    def _close_outputs(self, reason: str) -> None:
        """실행 닫힘: 송신 게이트를 닫는다(그 뒤 송신 0). W 추적은 초기화 요청."""
        if self.world is not None:
            self.world.request_reset(f"run_end:{reason}")

        if self.sort_part is not None:
            self.sort_part.close_run(reason)

    def _close_tf(self, limit_s: float):
        """TF 묶음을 닫는다. 수신 스레드가 실패했던 실행은 닫혀도 정리 확인으로 치지 않는다(#196 A1-2)."""
        self.tf.close(limit_s)

        with self._lock:
            error = self.tf_receiver_error

        if error is not None:
            raise RuntimeError(f"tf receiver failed: {error}")

        return True

    # ------------------------------------------------------------------ 수명

    def start(self) -> None:
        """건너뜀 기록 스레드·열기 스레드·(송신 스레드)·순번 기록기·worker를 시작한다."""
        self._skip_thread = threading.Thread(target=self._skip_loop, name="eyetoarm-live-observation-skips", daemon=True)
        self._skip_thread.start()
        self._probe_thread = threading.Thread(target=self._probe_loop, name="eyetoarm-live-observation-open",
                                              daemon=True)
        self._probe_thread.start()

        if self.sort_part is not None:
            self.sort_part.start()

        if self.world is not None:
            self.world.start()

        self.writer.start()
        self.dispatcher.start()

    def probes_idle(self) -> bool:
        """요청한 열기 시도를 모두 끝냈는가(시험 동기화용)."""
        with self._probe_cond:
            return self._probe is None and not self._probe_busy

    def _make_stage(self, camera):
        st = self._stage_factory(camera)

        if getattr(st[0], "on_output", None) is not None:
            raise RuntimeError("live observation link requires stage.on_output None (no C/D/robot output)")

        if self.b_input == B_INPUT_LIST:
            consume = (self._list_consumer_factory(st[0]) if self._list_consumer_factory is not None
                       else getattr(st[0], "process_frame", None))

            # 목록 소비자가 없으면 단계 T로 바꾸지 않는다(#201): worker 오류로 드러나고 진입점은 종료 1
            if not callable(consume):
                raise RuntimeError("list consumer unavailable (BStage.process_frame missing); not falling back to stage T")

            st = (st[0], st[1], consume)

        self.stages.append(st)

        return st

    # ------------------------------------------------------------------ 단계 L(worker, 잠금 밖)

    def _build_list_envelope(self, frame, item: LiveListInput, camera, tf_result):
        """촬영 stamp TF 하나(`tf_result.tf`)로 목록·봉투 2를 만든다. 대조 오류는 ValueError(디스패처가 `b_input_rejected`)."""
        capture = CaptureInfo(stamp_ns=frame.ros_stamp_ns, optical_frame=frame.optical_frame,
                              calibration_id=frame.calibration_id, width=frame.width, height=frame.height)
        observations = frame_observations_from_ai(item.context, item.display, capture=capture, camera=camera,
                                                  tf=tf_result.tf, world_frame=self.config.world_frame)

        # 실행기 `b_list_skip_reason`이 같은 함수로 비어 있지 않음을 확인했다. None이면 계약 위반(worker 오류)
        if observations is None:
            raise RuntimeError("b_list_skip_reason passed but frame_observations_from_ai built no list")

        return make_list_envelope(observations, world_frame=self.config.world_frame)

    def _compute_list(self, st, envelope):
        """목록 소비자를 한 번 부르고 길이 분포를 센다. 송신을 켰으면 고른 결과를 이 봉투와 짝으로 둔다(#233)."""
        self._list_chosen = None
        out = st[2](envelope)
        n = len(envelope.observations.candidates)

        if self.sort_part is not None:
            self._list_chosen = (envelope, self._find_chosen(st[0], out))

        with self._lock:
            self.list_consumer_calls += 1
            self.list_candidates_sent += n
            self.list_length_hist[n] = self.list_length_hist.get(n, 0) + 1

        return out

    @staticmethod
    def _find_chosen(stage, out):
        """B 반환 사전의 `goal`(고른 결과의 목표 객체 그 자체)을 가진 결과를 이번 사진 결과 묶음에서 찾는다.

        고른 것이 없으면 None. `goal`이 있는데 못 찾으면 `_UNMATCHED`(수락 훅이 계약 위반으로 센다).
        """
        goal = out.get("goal") if isinstance(out, dict) else None

        if goal is None:
            return None

        for o in getattr(stage, "last_frame", None) or ():
            if getattr(o, "goal", None) is goal:
                return o

        return _UNMATCHED

    def _offer_list_goal(self, item, payload, event_seq, ids) -> None:
        """단계 L 수락 훅(#233): 같은 봉투의 고른 결과 하나만 송신기에 넘긴다."""
        pair = self._list_chosen
        chosen = pair[1] if pair is not None and pair[0] is payload.envelope else _UNMATCHED

        if chosen is None:
            with self._lock:
                self.list_no_selection += 1
            return

        if chosen is _UNMATCHED:
            with self._lock:
                self.list_goal_unmatched += 1
            return

        with self._lock:
            self.list_goals_offered += 1

        self.sort_part.offer(chosen, item.frame, ids, event_seq)

    # ------------------------------------------------------------------ 수신 경계(소스 slot 잠금 안)

    def boundary_hook(self, epoch: int, kind) -> None:
        """소스 경계 훅. 되감기면 실행 종료(상태 닫기)만, 아니면 상태 경계. 상태 잠금만 잡는다."""
        if getattr(kind, "value", kind) == SIM_TIME_REWIND:
            with self._lock:
                if self.reset_reason is None:
                    self.reset_reason = f"reset:{SIM_TIME_REWIND}:epoch{epoch}"

            self.state.run_end(self.reset_reason)
            self._close_outputs(self.reset_reason)   # 이전 실행 목표·주석을 더 보내지 않는다
            return

        try:
            self.state.on_rx_boundary(epoch, kind)
        except Exception:
            # 현재성 갱신 실패를 계수로만 두면 이전 수락이 계속 현재로 남는다(#180 #5). 상태를 닫고 slot에 알린다(slot이 치명 알림)
            self.state.fail("boundary_hook_error")
            raise

    @property
    def reset_pending(self) -> bool:
        """리셋을 감지했고 아직 끝내지 않았는가."""
        with self._lock:
            return self.reset_reason is not None and self.ended is None

    # ------------------------------------------------------------------ 실행기 제어 흐름

    def on_live_input(self, item, generation: int) -> None:
        """코어에 반영된 실시간 입력 항목. 경계면 그 경계의 보정 번호·세대로 반영 완료를 알린다(되감기 제외)."""
        kind = getattr(item.kind, "value", item.kind)

        if kind == BOUNDARY_HOOK_FAILED:   # slot 경계 훅 실패(#180 #5): 현재성을 믿을 수 없으므로 fail-stop
            self.state.fail("boundary_hook_error")
            return

        if item.epoch is None or kind == SIM_TIME_REWIND:
            return

        profile = getattr(item, "profile", None)

        if self.world is not None:   # #217: 경계(프로필·보정·입력 상태)마다 추적기 초기화 요청(번호 재사용 없음)
            self.world.request_reset(f"boundary:{kind}")

        if profile is not None:
            self._calibration_id = profile.calibration_id
            self.camera_cleared = False
            self.dispatcher.set_camera(profile.camera)
            self._track_camera = profile.camera
        elif kind == PROFILE_CHANGED:
            self._track_camera = None
            # 프로필 없음 경계(해석할 수 없는 CameraInfo, #180 r1 L1): 이전 보정 번호를 현재로 두지 않는다. 디스패처의 대기 카메라
            # 교체는 None을 받지 않으므로 카메라는 「없음」으로 표시하고, 그동안 `offer`가 제출을 막는다(유효 프로필이 오면 해제)
            self._calibration_id = None
            self.camera_cleared = True

        self.state.on_applied(item.epoch, generation, self._calibration_id)

    def offer(self, outcome, skip_reason: str | None, rx_epoch: int | None, tracks=None) -> str:
        """terminal 응답 하나. 결과: submitted | 건너뜀 사유. `tracks`(#215)는 추적을 켠 실행의 같은 작업 TrackFrame(없으면 None)."""
        self.offers += 1
        frame, obs = outcome.context.frame, outcome.observation
        ids = {"job_seq": outcome.context.job_seq, "generation": outcome.context.generation, "rx_epoch": rx_epoch}

        # #217: 현재 응답(실행기 판정 통과·무효 관측)이면 추적 큐에 넣는다(넣기만, TF 없음). B로 보내지 않는 프레임도 추적은 본다
        if (self.world is not None and skip_reason in WORLD_TRACKABLE_SKIPS and self._track_camera is not None
                and isinstance(outcome.display_result, AiSegResult) and frame.calibration_id == self._calibration_id
                and self.ended is None and not self.reset_pending):
            self.world.offer(outcome.context, outcome.display_result, self._track_camera)

        if skip_reason is not None:
            return self._record_skip("runner", skip_reason, frame, ids)

        if self.ended is not None or self.reset_pending or self.state.closed:
            return self._record_skip("link", "run_ended", frame, ids)

        if rx_epoch is None:
            return self._record_skip("link", "rx_epoch_unknown", frame, ids)

        if not self.state.token.ready:
            with self._probe_cond:
                replaced = self._probe is not None
                self._probe = (frame, rx_epoch)
                self.probes_requested += 1
                last = self.open_attempts[-1][2] if self.open_attempts else None
                self._probe_cond.notify_all()

            return self._record_skip("link", "not_ready", frame, ids,
                                     detail=f"open attempt queued (last={last}, replaced={replaced})")

        token = self.state.token

        if self.camera_cleared or token.calibration_id is None:   # 프로필 없음(L1): 보정 없는 프레임끼리 일치해도 제출하지 않는다
            return self._record_skip("link", "calibration_unavailable", frame, ids)

        if frame.calibration_id != token.calibration_id:
            return self._record_skip("link", "calibration_mismatch", frame, ids,
                                     detail=f"frame {frame.calibration_id} != current {token.calibration_id}")

        if outcome.context.generation != token.generation:
            return self._record_skip("link", "generation_discarded", frame, ids)

        if self.b_input == B_INPUT_LIST:
            obs = LiveListInput(outcome.context, outcome.display_result)
        elif self.world is not None:   # #217: 번호는 worker가 추적 스레드 결과로 싣는다(지금은 대기 표시만)
            obs = with_track(obs, None, TRACK_PENDING)
            self._count_tracking("submitted_world")
        elif self.tracking:   # #215: 확정 번호만 하위 형 관측에 싣는다(그 밖은 번호 None)
            track_id, state = confirmed_track_id(tracks, outcome.context, obs)
            obs = with_track(obs, track_id, state)
            self._count_tracking(f"submitted_{state}")

        if not self.dispatcher.submit(frame, obs, rx_epoch=rx_epoch, generation=outcome.context.generation,
                                      calibration_id=frame.calibration_id):
            return self._record_skip("link", "submit_refused", frame, ids)

        self.submitted += 1

        return "submitted"

    def _probe_loop(self) -> None:
        """열기 스레드: 시도 칸의 프레임으로 TF 준비를 시도한다(A 제어 흐름 밖)."""
        while True:
            with self._probe_cond:
                while self._probe is None and not self._probe_stop:
                    self._probe_cond.wait(0.05)

                if self._probe_stop:
                    return

                (frame, epoch), self._probe = self._probe, None
                self._probe_busy = True

            try:
                status = self._try_open(frame, epoch)
            except Exception as exc:   # 열기 실패는 스레드를 멈추지 않는다
                status = f"error:{type(exc).__name__}: {exc}"

            with self._probe_cond:
                self.open_attempts.append((getattr(frame, "image_seq", None), epoch, status))
                self.probes_done += 1
                self._probe_busy = False
                self._probe_cond.notify_all()

            if status != "ready":
                self._record_skip("open", f"open:{status.split(':')[0]}", frame, {"rx_epoch": epoch}, detail=status)

    def _try_open(self, frame, epoch: int) -> str:
        """그 경계 영상으로 열기: 반영 완료 확인 → 새 TF 시도 → 촬영 stamp 정확 조회."""
        boundary = self.state.boundary

        if boundary is None or boundary.boundary_id != epoch:
            return "stale_epoch"

        if not boundary.applied:
            return "not_applied"

        if boundary.tf_ready:
            return "ready" if self.state.token.ready else "closed"

        if self.readiness.begin(epoch) != "begun":
            return "not_begun"

        return self.readiness.try_ready(epoch, frame)

    # ------------------------------------------------------------------ 건너뜀 기록

    def _worker_skip(self, item, reason: str, detail: str | None) -> None:
        """디스패처 건너뜀(worker·제어 흐름)."""
        obs = item.observation
        ids = {"job_seq": getattr(obs, "job_seq", None), "generation": getattr(obs, "generation", None),
               "rx_epoch": item.rx_epoch}
        self._record_skip("worker", reason, item.frame, ids, detail=detail)

    def _record_skip(self, stage: str, reason: str, frame, ids: dict, *, detail: str | None = None) -> str:
        rec = {"stage": stage, "reason": reason, "detail": detail, "run_id": getattr(frame, "run_id", None),
               "image_seq": getattr(frame, "image_seq", None), "ros_stamp_ns": getattr(frame, "ros_stamp_ns", None),
               "calibration_id": getattr(frame, "calibration_id", None), **ids}

        fail = None

        with self._skip_cond:   # 메모리 큐에만 넣는다(디스크 쓰기는 기록 스레드). 기다리지 않는다
            self.skips[reason] = self.skips.get(reason, 0) + 1
            self.skip_total += 1
            depth = len(self._skip_queue) + self._skip_inflight

            if self.skip_write_error is not None or self._skip_stop and self._skip_thread is None:
                self.skip_dropped += 1
            elif depth >= self._skip_capacity:
                self.skip_dropped += 1

                if self.skip_overflow_at is None:
                    self.skip_overflow_at = self.skip_total
                    fail = "skip_queue_overflow"
            else:
                self._skip_queue.append(rec)
                self.skip_max_depth = max(self.skip_max_depth, depth + 1)
                self._skip_cond.notify_all()

        if fail is not None:   # 상태 잠금은 건너뜀 잠금 밖에서
            self.state.fail(fail)

        return reason

    def _skip_loop(self) -> None:
        """건너뜀 기록 스레드 최상단(#180 #6): 예기치 않은 예외도 `skip_write_error`·`state.fail`로 드러내고 파일을 닫는다.

        이 경우 `skip_crash`가 남아 정리 단계 「skips」가 성공으로 처리되지 않는다(`cleanup_confirmed` 거짓).
        """
        try:
            self._skip_loop_body()
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"

            with self._skip_cond:
                self.skip_crash = msg
                self.skip_write_error = self.skip_write_error or msg
                self._skip_inflight = 0

            self.state.fail("skip_writer_error")
        finally:
            if not self._skip_fh.closed:
                try:
                    self._skip_fh.close()
                except OSError as exc:
                    self.skip_write_error = self.skip_write_error or f"close: {type(exc).__name__}: {exc}"

    def _skip_loop_body(self) -> None:
        """큐를 꺼내 쓰고, 멈춤 요청 뒤 남은 것을 비우고 파일을 닫는다."""
        while True:
            with self._skip_cond:
                while not self._skip_queue and not self._skip_stop:
                    self._skip_cond.wait(0.05)

                batch, self._skip_queue = self._skip_queue, []
                self._skip_inflight = len(batch)
                stop = self._skip_stop

            failed_now = False

            if batch and self.skip_write_error is None:
                self.skip_write_threads.add(threading.current_thread().name)
                written = 0

                try:
                    for rec in batch:
                        self._skip_fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
                        written += 1

                    self._skip_fh.flush()
                except (OSError, ValueError) as exc:
                    with self._skip_cond:
                        self.skip_write_error = f"{type(exc).__name__}: {exc}"
                    failed_now = True

                with self._skip_cond:
                    self.skip_written += written
                    self._skip_inflight = 0
            else:
                with self._skip_cond:
                    self._skip_inflight = 0

            if failed_now:
                self.state.fail("skip_writer_error")

            if stop:
                with self._skip_cond:
                    if self._skip_queue:
                        continue

                try:
                    self._skip_fh.close()
                except OSError as exc:
                    self.skip_write_error = self.skip_write_error or f"close: {type(exc).__name__}: {exc}"

                return

    # ------------------------------------------------------------------ 정리

    def end(self, reason: str, *, limit_s: float):
        """상태 닫기(1회) → 실패하지 않은 정리 단계 진행. 두 번째부터는 실패 단계만 재시도."""
        from eyetoarm.vision.live_run_session import RunEnd

        with self._lock:
            first = self.ended is None

        if first:
            self.state.run_end(reason)
            self._close_outputs(reason)
        else:
            reason = self.ended.reason

        errors = []

        def writer_close():
            report = self.writer.close(limit_s)
            self.writer_report = report

            if report.timed_out:
                return False

            return True

        def probe_close():
            with self._probe_cond:
                self._probe_stop = True
                self._probe_cond.notify_all()

            return _join(self._probe_thread, limit_s)

        def skip_close():
            with self._skip_cond:
                self._skip_stop = True
                self._skip_cond.notify_all()

            if self._skip_thread is None:   # 시작 전 실패: 호출자가 소유자로 남은 것을 쓰고 닫는다
                with self._skip_cond:
                    batch, self._skip_queue = self._skip_queue, []

                if not self._skip_fh.closed:
                    for rec in batch:
                        self._skip_fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")

                    self._skip_fh.close()

                    with self._skip_cond:
                        self.skip_written += len(batch)

                return True

            if not _join(self._skip_thread, limit_s):
                return False

            if self.skip_crash is not None:   # 스레드는 끝났지만 비정상 종료(#180 #6): 정리 확인으로 치지 않는다
                raise RuntimeError(f"skip writer crashed: {self.skip_crash}")

            return True

        world_step = (() if self.world is None else (("world_tracking", lambda: self.world.close(limit_s)),))
        sort_step = (() if self.sort_part is None else
                     (("sort_part", lambda: self.sort_part.close(min(limit_s, self.sort_part.config.close_limit_s))),))
        steps = (("probe", probe_close), ("dispatcher", lambda: self.dispatcher.close(limit_s)), *world_step, *sort_step,
                 ("writer", writer_close), ("tf", lambda: self._close_tf(limit_s)), ("skips", skip_close))

        for name, fn in steps:
            with self._lock:
                if name in self._cleanup_done:
                    continue

            try:
                result = fn()
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
                continue

            if result is False:
                errors.append(f"{name}: did not stop within {limit_s}s")
                continue

            with self._lock:
                self._cleanup_done.add(name)

        record = RunEnd(self.run_id, reason, tuple(errors))

        with self._lock:
            self.ended = record
            self.cleanup_attempts += 1

        return record

    def retry_cleanup(self, limit_s: float):
        """끝난 실행의 실패 단계만 다시."""
        if self.ended is None:
            raise ValueError(f"run {self.run_id} has not ended")

        return self.end(self.ended.reason, limit_s=limit_s)

    @property
    def cleanup_confirmed(self) -> bool:
        """끝났고 모든 정리 단계 성공 + worker 스레드 종료."""
        with self._lock:
            need = ({"probe", "dispatcher", "writer", "tf", "skips"}
                    | (set() if self.sort_part is None else {"sort_part"})
                    | (set() if self.world is None else {"world_tracking"}))
            done = self.ended is not None and self._cleanup_done >= need

        threads = (getattr(self.dispatcher, "_thread", None), self._probe_thread, self._skip_thread,
                   None if self.sort_part is None else self.sort_part.thread,
                   None if self.world is None else self.world.thread)

        return done and not any(t is not None and t.is_alive() for t in threads)

    def skip_counts(self) -> dict:
        """건너뜀 기록 계수: 전체 = 쓴 것 + 누락 + 쓰지 못한 것(쓰기 실패 묶음·정리 시간 초과)."""
        with self._skip_cond:
            unwritten = self.skip_total - self.skip_written - self.skip_dropped

            return {"capacity": self._skip_capacity, "total": self.skip_total, "written": self.skip_written,
                    "dropped": self.skip_dropped, "unwritten": unwritten, "max_depth": self.skip_max_depth,
                    "overflow_at": self.skip_overflow_at}

    def summary(self) -> dict:
        """종료 요약(사전)."""
        report = self.writer_report

        if self.run_outputs is not None:   # #233 r1: 진입점이 준 실제 출구(C 연결부는 항상)
            outputs = {"record_only": not self.run_outputs, "outputs": list(self.run_outputs)}
        else:
            outputs = ({"record_only": True, "outputs": "none (no C/D/robot output)"} if self.sort_part is None else
                       {"record_only": False, "outputs": "C goals via /sort_part (accepted results only, #214)"})

        return {"run_id": self.run_id, **outputs,
                "config_status": LIVE_OBSERVATION_CONFIG_STATUS, "config_sha256": self.config.sha256,
                "tf_wait_limit_s": self.config.tf_wait_limit_s, "offers": self.offers, "submitted": self.submitted,
                "skips": dict(sorted(self.skips.items())), "dispatcher": dict(sorted(self.dispatcher.counts.items())),
                "dispatcher_errors": list(self.dispatcher.errors), "compute_calls": self.dispatcher.compute_calls,
                "state": dict(sorted(self.state.counts.items())), "state_failed": self.state.failed,
                "reset_reason": self.reset_reason,
                "ended": None if self.ended is None else {"reason": self.ended.reason,
                                                          "close_errors": list(self.ended.close_errors)},
                "cleanup_confirmed": self.cleanup_confirmed, "cleanup_attempts": self.cleanup_attempts,
                "writer": None if report is None else {"written": report.written, "unwritten": report.unwritten,
                                                       "pending": report.pending, "timed_out": report.timed_out,
                                                       "error": report.error,
                                                       "lag_warnings": self.writer.lag_warnings},
                "skip_write_error": self.skip_write_error, "drained_before_end": self.drained,
                "tf_receiver_error": self.tf_receiver_error,
                **({} if self.b_input == B_INPUT_SINGLE else self._list_summary()),
                **({} if self.view_bias is None else {"view_bias": {"enabled": True, **self.view_bias.to_json(),
                                                                     **self.dispatcher.view_bias_summary()}}),
                **({} if self.sort_part is None else {"sort_part": self.sort_part.summary()}),
                **({} if not self.tracking else {"tracking_link": self._tracking_summary()}),
                **({} if self.world is None else {"world_tracking": self.world.summary()}),
                "skip_queue": self.skip_counts(),
                "open_attempts": [list(a) for a in self.open_attempts],
                "skip_write_threads": sorted(self.skip_write_threads)}

    def _tracking_summary(self) -> dict:
        """추적 연결 요약 키(#215). 추적을 끈 요약에는 넣지 않는다."""
        with self._lock:
            counts = dict(sorted(self.tracking_counts.items()))
            calls = self.release_proxy.calls

        return {"mode": "world" if self.world is not None else "heuristic", "counts": counts, "release_proxy_calls": calls}

    def _list_summary(self) -> dict:
        """단계 L 요약 키(#201). 단일 관측 요약에는 넣지 않는다."""
        with self._lock:
            return {"b_input": "eyetoarm.b_input/2", "consumer_kind": self.consumer_kind,
                    "list_consumer_calls": self.list_consumer_calls, "list_candidates_sent": self.list_candidates_sent,
                    "list_length_hist": {str(k): v for k, v in sorted(self.list_length_hist.items())},
                    **({} if self.sort_part is None else {
                        "list_goals_offered": self.list_goals_offered, "list_no_selection": self.list_no_selection,
                        "list_goal_unmatched": self.list_goal_unmatched})}

    def write_summary(self, extra: dict) -> None:
        """`summary.json`을 새로 쓴다."""
        with open(self.dir / "summary.json", "x", encoding="utf-8") as fh:
            json.dump({**self.summary(), **extra}, fh, ensure_ascii=False, indent=2, default=str)


__all__ = ["B_INPUT_LIST", "B_INPUT_SINGLE", "LIVE_OBSERVATION_CONFIG_STATUS", "LiveListInput", "LiveObservationConfigError", "LiveObservationLink",
           "LiveObservationLinkConfig", "NON_OK_TF", "load_live_observation_config"]
