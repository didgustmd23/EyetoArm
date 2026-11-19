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
- ROS·Qt를 import하지 않는다(TF 수신 묶음·B 단계는 주입).
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

PROFILE_CHANGED = "profile_changed"             # `inputs.live_slot.BoundaryKind.PROFILE_CHANGED` 값(#180 r1 L1)
BOUNDARY_HOOK_FAILED = "boundary_hook_failed"   # `inputs.live_slot.NoticeKind.BOUNDARY_HOOK_FAILED` 값(#180 #5, 입력 모듈 import 없이 비교)

LIVE_OBSERVATION_CONFIG_STATUS = "test_only"
NON_OK_TF = frozenset({"unavailable", "late", "stamp_mismatch", "no_stamp", "invalid_stamp"})
_CONFIG_KEYS = {"status", "world_frame", "tf_wait_limit_s", "static_pairs", "future_tolerance_ns", "write_capacity",
                "reserved_capacity", "lag_warn_s", "close_limit_s", "skip_queue_capacity", "note"}


B_INPUT_SINGLE = "single"
B_INPUT_LIST = "list"


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

    unknown = set(doc) - _CONFIG_KEYS
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

    return LiveObservationLinkConfig(
        world_frame=doc["world_frame"], tf_wait_limit_s=float(_positive(doc["tf_wait_limit_s"], "tf_wait_limit_s")),
        static_pairs=tuple(tuple(p) for p in pairs), future_tolerance_ns=doc["future_tolerance_ns"],
        write_capacity=_positive(doc["write_capacity"], "write_capacity", integer=True),
        reserved_capacity=_positive(doc["reserved_capacity"], "reserved_capacity", integer=True),
        lag_warn_s=float(_positive(doc["lag_warn_s"], "lag_warn_s")),
        close_limit_s=float(_positive(doc["close_limit_s"], "close_limit_s")),
        skip_queue_capacity=_positive(doc["skip_queue_capacity"], "skip_queue_capacity", integer=True),
        sha256=hashlib.sha256(raw).hexdigest())


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
                 b_input: str = B_INPUT_SINGLE, list_consumer_factory: Callable | None = None) -> None:
        """출력 폴더(새로)·상태(start_closed)·기록기·디스패처·TF 준비를 만든다.

        - `b_input`(#201): `"single"`(기본, 단계 T 봉투 1) | `"list"`(단계 L 목록 봉투 2). `"list"`면 `render`는 쓰지 않는다.
        - `list_consumer_factory(stage)`(#201, 시험 주입): 목록 봉투를 받는 함수. None이면 `stage.process_frame`.

        - `tf`: 시작된 TF 수신 묶음(`lookup(parent, child, stamp_ns, timeout_s)`·`has_static`·`close(limit_s)`).
        - `stage_factory(camera)` → `(BStage, ObservationEnvelopeAdapter)`. B 단계의 `on_output`은 None이어야 한다.
        - `render(stage, frame, out)` → PNG bytes | None(같은 프레임 주석).
        """
        # 설정 검사는 파일을 열기 전에 한다(#180 #7: 검사 실패가 기록 파일 핸들을 남기지 않게)
        if type(config.skip_queue_capacity) is not int or config.skip_queue_capacity <= 0:
            raise ValueError(f"skip_queue_capacity must be a positive int: {config.skip_queue_capacity!r}")

        if b_input not in (B_INPUT_SINGLE, B_INPUT_LIST):
            raise ValueError(f"b_input must be {B_INPUT_SINGLE!r} or {B_INPUT_LIST!r}: {b_input!r}")

        self.b_input = b_input
        self._list_consumer_factory = list_consumer_factory
        self.consumer_kind = (None if b_input == B_INPUT_SINGLE
                              else "injected_test_only" if list_consumer_factory is not None else "BStage.process_frame")
        self.list_consumer_calls = 0              # 목록 소비자 호출 수(worker만 씀, 잠금으로 읽음)
        self.list_candidates_sent = 0
        self.list_length_hist: dict[int, int] = {}

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
            list_kw = dict(build_envelope=lambda f, o, cam, r: make_live_envelope(f, o, camera=cam, tf_result=r,
                                                                                  world_frame=config.world_frame),
                           compute=lambda st, env: st[1].consume(env).output,
                           render=None if render is None else (lambda st, f, out: render(st[0], f, out)))

        self.dispatcher = LiveObservationDispatcher(
            self.state, camera=None, stage_factory=self._make_stage, tf_lookup=capture,
            clock_ns=clock_ns, refuse_tf_statuses=NON_OK_TF, on_skip=self._worker_skip, **list_kw)
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
        """건너뜀 기록 스레드·열기 스레드·순번 기록기·worker를 시작한다."""
        self._skip_thread = threading.Thread(target=self._skip_loop, name="eyetoarm-live-observation-skips", daemon=True)
        self._skip_thread.start()
        self._probe_thread = threading.Thread(target=self._probe_loop, name="eyetoarm-live-observation-open",
                                              daemon=True)
        self._probe_thread.start()
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
        """목록 소비자를 한 번 부르고 길이 분포를 센다."""
        out = st[2](envelope)
        n = len(envelope.observations.candidates)

        with self._lock:
            self.list_consumer_calls += 1
            self.list_candidates_sent += n
            self.list_length_hist[n] = self.list_length_hist.get(n, 0) + 1

        return out

    # ------------------------------------------------------------------ 수신 경계(소스 slot 잠금 안)

    def boundary_hook(self, epoch: int, kind) -> None:
        """소스 경계 훅. 되감기면 실행 종료(상태 닫기)만, 아니면 상태 경계. 상태 잠금만 잡는다."""
        if getattr(kind, "value", kind) == SIM_TIME_REWIND:
            with self._lock:
                if self.reset_reason is None:
                    self.reset_reason = f"reset:{SIM_TIME_REWIND}:epoch{epoch}"

            self.state.run_end(self.reset_reason)
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

        if profile is not None:
            self._calibration_id = profile.calibration_id
            self.camera_cleared = False
            self.dispatcher.set_camera(profile.camera)
        elif kind == PROFILE_CHANGED:
            # 프로필 없음 경계(해석할 수 없는 CameraInfo, #180 r1 L1): 이전 보정 번호를 현재로 두지 않는다. 디스패처의 대기 카메라
            # 교체는 None을 받지 않으므로 카메라는 「없음」으로 표시하고, 그동안 `offer`가 제출을 막는다(유효 프로필이 오면 해제)
            self._calibration_id = None
            self.camera_cleared = True

        self.state.on_applied(item.epoch, generation, self._calibration_id)

    def offer(self, outcome, skip_reason: str | None, rx_epoch: int | None) -> str:
        """terminal 응답 하나. 결과: submitted | 건너뜀 사유."""
        self.offers += 1
        frame, obs = outcome.context.frame, outcome.observation
        ids = {"job_seq": outcome.context.job_seq, "generation": outcome.context.generation, "rx_epoch": rx_epoch}

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

        steps = (("probe", probe_close), ("dispatcher", lambda: self.dispatcher.close(limit_s)),
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
            done = self.ended is not None and self._cleanup_done >= {"probe", "dispatcher", "writer", "tf", "skips"}

        threads = (getattr(self.dispatcher, "_thread", None), self._probe_thread, self._skip_thread)

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

        return {"run_id": self.run_id, "record_only": True, "outputs": "none (no C/D/robot output)",
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
                "skip_queue": self.skip_counts(),
                "open_attempts": [list(a) for a in self.open_attempts],
                "skip_write_threads": sorted(self.skip_write_threads)}

    def _list_summary(self) -> dict:
        """단계 L 요약 키(#201). 단일 관측 요약에는 넣지 않는다."""
        with self._lock:
            return {"b_input": "eyetoarm.b_input/2", "consumer_kind": self.consumer_kind,
                    "list_consumer_calls": self.list_consumer_calls, "list_candidates_sent": self.list_candidates_sent,
                    "list_length_hist": {str(k): v for k, v in sorted(self.list_length_hist.items())}}

    def write_summary(self, extra: dict) -> None:
        """`summary.json`을 새로 쓴다."""
        with open(self.dir / "summary.json", "x", encoding="utf-8") as fh:
            json.dump({**self.summary(), **extra}, fh, ensure_ascii=False, indent=2, default=str)


__all__ = ["B_INPUT_LIST", "B_INPUT_SINGLE", "LIVE_OBSERVATION_CONFIG_STATUS", "LiveListInput", "LiveObservationConfigError", "LiveObservationLink",
           "LiveObservationLinkConfig", "NON_OK_TF", "load_live_observation_config"]
