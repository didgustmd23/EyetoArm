"""실시간 W 추적 스레드: 제어 흐름 밖에서 W 추적기를 돌리고 worker에 작업별 결과(`TrackFrame` | 사유)를 넘긴다.

`offer`는 유한 큐에 넣기만 하고(가득이면 새 항목을 버림), 추적기·`world_tracks.jsonl`은 추적 스레드가 소유한다.
밀림이 이어지면 최신 1개만 남기고 전 트랙을 종료한다. C `speed_mps` 등 진단 값은 추적기에 넣지 않는다(정답 격리).
ROS·Qt를 import하지 않는다.
"""
from __future__ import annotations

import collections
import json
import math
import threading
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable

import numpy as np

from types import SimpleNamespace

from eyetoarm.transforms.planar_transform import w_to_image
from eyetoarm.vision.arm_occlusion import occluded_fn, world_capsules
from eyetoarm.vision.gripper_depth import GripperConfig, GripperConfigError, GripperDepthFrame, observe, parse_gripper, project_w
from eyetoarm.vision.track_class_lock import TrackClassLock
from eyetoarm.vision.track_stream import TrackStream, frame_message, marker_message, reset_message
from eyetoarm.vision.tracking import FrameKey, TrackFrame
from eyetoarm.vision.world_tracking import (BoltPlanes, GripperInput, VisibilityConfig, WorldTracker, WorldTrackerConfigError,
                                            parse_visibility, world_inputs)
from eyetoarm.vision.world_tracking_display import (DisplayPrevConfig, DisplayPrevConfigError, PrevSnapshot, match_prev,
                                                    parse_display_prev, snapshot)

RESULT_KEEP = 64
GRIPPER_KEEP = 8


class LagConfigError(ValueError):
    """밀림 규칙 설정 오류."""


@dataclass(frozen=True)
class LiveWorldConfig:
    """추적 스레드 설정. 모두 필수.

    추적 큐 대기 `lag_ms` > `lag_limit_ms`인 밀림이 `lag_persist_n`회 이어지면 전 트랙을 종료한다
    (`lag_cooldown_ms` 동안 다시 하지 않음). 밀림은 추적기 자신의 처리 지연으로만 판정한다.

    `pending_ratio_limit`은 진단 한도다. B가 결과를 기다리다 못 받은 비율(`wait` 실패, 창 `window_n`)이 이 값을
    넘는 구간을 `pending_high`로 세기만 하고 트랙은 건드리지 않는다. B가 추적기보다 먼저 물어도 생기는 값이라
    밀림의 근거로 쓰지 않는다(#372: 기록된 밀림 초기화가 모두 이 값 때문이었고 큐 대기는 최대 22 ms).
    """

    queue_capacity: int
    handoff_wait_s: float
    lag_limit_ms: float
    window_n: int
    pending_ratio_limit: float
    lag_persist_n: int
    lag_cooldown_ms: float
    conveyor_probe_timeout_s: float   # C speed_mps 진단 조회 상한(추적기에 전달 안 함)
    # detect 영상 직전 결과 표시(화면 전용). 설정 파일에서는 필수, None이면 직전 결과를 쓰지 않는다
    display_prev: DisplayPrevConfig | None = None

    def to_json(self) -> dict:
        """요약용 사전."""
        doc = {f.name: getattr(self, f.name) for f in fields(self)}
        doc["display_prev"] = None if self.display_prev is None else self.display_prev.to_json()

        return doc


def parse_live_world(doc) -> LiveWorldConfig:
    """`queue_capacity`·`handoff_wait_s` + `lag` 블록. 모르는 키·누락·범위 오류는 `LagConfigError`."""
    if not isinstance(doc, dict) or set(doc) != {"queue_capacity", "handoff_wait_s", "lag", "conveyor_probe_timeout_s",
                                                 "display_prev"}:
        raise LagConfigError("live world block must have exactly queue_capacity, handoff_wait_s, lag, conveyor_probe_timeout_s, "
                             "display_prev")

    lag = doc["lag"]
    keys = {"lag_limit_ms", "window_n", "pending_ratio_limit", "lag_persist_n", "lag_cooldown_ms"}

    if not isinstance(lag, dict) or set(lag) != keys:
        raise LagConfigError(f"lag must have exactly {sorted(keys)}")

    def pos(v, name, integer=False):
        if isinstance(v, bool) or not isinstance(v, int if integer else (int, float)) or \
                (not integer and not math.isfinite(v)) or v <= 0:
            raise LagConfigError(f"{name} must be a positive {'int' if integer else 'number'}: {v!r}")

        return v if integer else float(v)

    ratio = pos(lag["pending_ratio_limit"], "pending_ratio_limit")

    if ratio > 1.0:
        raise LagConfigError("pending_ratio_limit must be <= 1")

    return LiveWorldConfig(queue_capacity=pos(doc["queue_capacity"], "queue_capacity", True),
                           handoff_wait_s=pos(doc["handoff_wait_s"], "handoff_wait_s"),
                           lag_limit_ms=pos(lag["lag_limit_ms"], "lag_limit_ms"), window_n=pos(lag["window_n"], "window_n", True),
                           pending_ratio_limit=ratio, lag_persist_n=pos(lag["lag_persist_n"], "lag_persist_n", True),
                           lag_cooldown_ms=pos(lag["lag_cooldown_ms"], "lag_cooldown_ms"),
                           conveyor_probe_timeout_s=pos(doc["conveyor_probe_timeout_s"], "conveyor_probe_timeout_s"),
                           display_prev=_display_prev(doc["display_prev"]))


def _display_prev(doc) -> DisplayPrevConfig:
    """`display_prev` 블록. 오류는 `LagConfigError`로 바꾼다(진입점이 같은 설정 오류로 다룸)."""
    try:
        return parse_display_prev(doc)
    except DisplayPrevConfigError as exc:
        raise LagConfigError(str(exc)) from exc


def load_world_tracking_config(path: str):
    """W 추적 시험 설정 파일 → (`WorldTrackerConfig`, `LiveWorldConfig`, sha256).

    키: `status`(반드시 `test_only`)·`note`(선택)·`tracker`·`live`·`conveyor_state`(지금은 null만).
    """
    import hashlib

    from eyetoarm.vision.world_tracking import WorldTrackerConfigError, parse_world_tracker

    raw = Path(path).read_bytes()

    try:
        doc = json.loads(raw)
    except ValueError as exc:
        raise LagConfigError(f"world tracking config is not JSON: {exc}") from exc

    keys = {"status", "tracker", "live", "conveyor_state"}

    if not isinstance(doc, dict) or not keys <= set(doc) or set(doc) - keys - {"note", "visibility", "gripper"}:
        raise LagConfigError(f"world tracking config must have {sorted(keys)} (+ optional note, visibility, gripper)")

    if doc["status"] != "test_only":
        raise LagConfigError("world tracking config status must be 'test_only' (development values)")

    if doc["conveyor_state"] is not None:
        raise LagConfigError("conveyor_state topic is not supported yet (C topic name/type undecided); keep it null")

    try:
        tracker = parse_world_tracker(doc["tracker"])
    except WorldTrackerConfigError as exc:
        raise LagConfigError(str(exc)) from exc

    load_handoff_config(path)   # 선택 블록도 같은 시점에 검사한다

    return tracker, parse_live_world(doc["live"]), hashlib.sha256(raw).hexdigest()


def load_handoff_config(path: str) -> tuple[VisibilityConfig | None, GripperConfig | None]:
    """같은 설정 파일의 선택 블록 `visibility`·`gripper` → (설정 | None, 설정 | None). 오류는 `LagConfigError`."""
    try:
        doc = json.loads(Path(path).read_bytes())
    except ValueError as exc:
        raise LagConfigError(f"world tracking config is not JSON: {exc}") from exc

    try:
        visibility = parse_visibility(doc["visibility"]) if "visibility" in doc else None
        gripper = parse_gripper(doc["gripper"]) if "gripper" in doc else None
    except (WorldTrackerConfigError, GripperConfigError) as exc:
        raise LagConfigError(str(exc)) from exc

    return visibility, gripper


@dataclass(frozen=True)
class GripperLive:
    """추적 스레드의 그리퍼 입력 묶음: 설정·벨트 사각형(W m, 설정값)."""

    config: GripperConfig
    roi_xy_m: tuple[float, float, float, float]


@dataclass(frozen=True)
class GripperItem:
    """그리퍼 입력 항목(관측 스레드 앞): 프레임·넣을 때의 재설정 회차·넣은 시각."""

    frame: GripperDepthFrame
    epoch: int
    enqueued_ns: int


@dataclass(frozen=True)
class GripperResult:
    """관측 스레드 결과(추적 스레드가 짝짓기만 함). `enqueued_ns`는 결과가 준비된 시각."""

    frame: GripperDepthFrame
    epoch: int
    enqueued_ns: int
    tf_status: str
    T_W_C: object
    in_view: tuple
    obs: object
    observe_ms: float


@dataclass(frozen=True)
class WorldTrackItem:
    """추적 큐 항목(참조만)."""

    key: FrameKey
    frame: object
    display: object
    camera: object
    enqueued_ns: int


def frame_key(context) -> FrameKey:
    """JobContext → 작업 키."""
    f = context.frame

    return FrameKey(run_id=f.run_id, source_id=f.source_id, image_seq=f.image_seq, capture_generation=f.capture_generation,
                    generation=context.generation, job_seq=context.job_seq, detector_config_id=context.detector_config_id)


def _pct(values, q: float):
    if not values:
        return None

    v = sorted(values)

    return round(v[min(len(v) - 1, int(q * (len(v) - 1)))], 3)


class LiveWorldTracking:
    """실행 하나의 W 추적 스레드. 큐·결과 표·계수는 조건변수 하나가, 추적기와 기록 파일은 추적 스레드가 소유한다."""

    def __init__(self, run_id: str, tracker: WorldTracker, config: LiveWorldConfig, *, tf_lookup: Callable,
                 plane_z_m: float | BoltPlanes,
                 out_dir: str, view_bias=None, clock_ns: Callable[[], int] = time.monotonic_ns,
                 visibility: VisibilityConfig | None = None, gripper: GripperLive | None = None,
                 stream: TrackStream | None = None) -> None:
        """새 기록 파일 `world_tracks.jsonl`을 연다. `tf_lookup(frame)`은 촬영 stamp 정확 조회다.

        `plane_z_m`은 볼트 투영 높이 하나(m) 또는 종류별 높이 `BoltPlanes`다. `visibility`·`gripper`가 None이면 지금과 같다.
        `stream`(시작 전 상태로 받음)이 있으면 링크 프레임마다 확정 트랙을 발행하고, 이 객체가 시작·종료한다. None이면 지금과 같다.
        """
        self.run_id = run_id
        self._tracker = tracker
        self.config = config
        self._tf_lookup = tf_lookup
        self._plane_z = plane_z_m if isinstance(plane_z_m, BoltPlanes) else float(plane_z_m)
        self._view_bias = view_bias
        self._clock_ns = clock_ns
        self._fh = open(Path(out_dir) / "world_tracks.jsonl", "x", encoding="utf-8")
        self._cond = threading.Condition()
        self._queue: collections.deque[WorldTrackItem] = collections.deque()
        self._results: collections.OrderedDict[tuple, object] = collections.OrderedDict()
        self._reset_reason: str | None = None
        self._stop = False
        self._thread: threading.Thread | None = None
        self._handoff_window: collections.deque[bool] = collections.deque(maxlen=config.window_n)
        self._lag_streak = 0
        self._last_lag_reset_ns: int | None = None
        self.crash: str | None = None
        self.write_error: str | None = None
        self.counts = {k: 0 for k in ("offered", "dropped_full", "processed", "updated", "handoff_ok", "handoff_pending",
                                      "stale_skipped", "lag_resets", "reset_requests", "pending_high")}
        self._pending_high = False   # 추적 스레드: 지금 B 기다림 실패 비율이 진단 한도를 넘는 구간인가
        self.not_updated: dict[str, int] = {}
        # `peek` 계수. 밀림 규칙 입력(handoff_*·대기 창)과 섞지 않는다
        self.display_counts = {"found": 0, "prev": 0, "pending": 0, "reason": 0}
        self.display_prev_miss: dict[str, int] = {}
        self._display_prev: PrevSnapshot | None = None   # 조건변수 안, 추적 스레드가 바꿈
        self.lag_reset_log: list[dict] = []
        self.last_reset: str | None = None      # 마지막으로 처리한 초기화 사유
        self._queue_len_samples: collections.deque[int] = collections.deque(maxlen=4096)
        self._lag_ms_samples: collections.deque[float] = collections.deque(maxlen=4096)
        self._tf_wait_samples: collections.deque[float] = collections.deque(maxlen=4096)
        self.queue_len_max = 0
        self.diagnostics: dict = {}   # 진단 전용(C speed_mps 조회 등). 추적기에 전달하지 않는다
        self._visibility = visibility
        self._gripper = gripper
        self._grip_in: collections.deque[GripperItem] = collections.deque()       # 관측 스레드 입력(조건변수 안)
        self._grip_queue: collections.deque[GripperResult] = collections.deque()  # 관측 결과, 추적 스레드가 재정렬(조건변수 안)
        self._snapshot: list = []                # 확정 트랙 예측 [(번호, x, y)] 사본(조건변수 안), 관측 스레드가 화면 판정에 씀
        self._grip_thread: threading.Thread | None = None
        self._observe = observe                  # 관측 함수(시험 주입용)
        self._grip_epoch = 0                     # 재설정 회차(조건변수 안). 넣을 때와 다르면 버린다
        self._link_generation: int | None = None   # 마지막 링크 항목 세대(추적 스레드)
        self._last_gripper_ts: int | None = None   # 마지막 그리퍼 처리 시각(추적 스레드)
        self.gripper_counts = {k: 0 for k in ("offered", "dropped_old", "processed", "old_epoch", "no_link_yet", "no_track_in_view",
                                              "observed", "updated", "late", "stale_after_gripper")}
        self._observe_ms: collections.deque[float] = collections.deque(maxlen=4096)
        self._occl_ms: collections.deque[float] = collections.deque(maxlen=4096)   # 가림 판정 준비(TF 조회 포함) ms
        self.occlusion_tf_fail = 0
        self.gripper_reasons: dict[str, int] = {}
        self._gripper_ms: collections.deque[float] = collections.deque(maxlen=4096)
        self._stream = stream

        if stream is not None:   # 발행 기록(촬영·발행 SIM 시각)은 같은 실행 폴더에
            stream.set_record(Path(out_dir) / "track_stream.jsonl")
        # 확정 번호 → (종류, 투영 높이 m), 끝점 입력에서 고른 값. 설정이 있으면 팔 근처·급감 사진으로 바꾸지 않는다(추적 스레드)
        self._class_lock = TrackClassLock(None if stream is None else stream.config.class_lock)
        self._class_counts = dict(self._class_lock.counts)   # 요약용 사본(조건변수 안, 추적 스레드가 바꿈)
        self._stream_source = ""                 # 마지막 링크 항목의 source_id(재설정·끝 알림용, 추적 스레드)

    @property
    def handoff_on(self) -> bool:
        """가시성 만료나 그리퍼 입력이 켜졌는가."""
        return self._visibility is not None or self._gripper is not None

    # ------------------------------------------------------------------ 제어 흐름

    def offer(self, context, display, camera) -> bool:
        """현재 응답 하나를 큐에 넣는다(기다리지 않음). 가득이면 새 것을 버리고 False."""
        item = WorldTrackItem(frame_key(context), context.frame, display, camera, self._clock_ns())

        with self._cond:
            self.counts["offered"] += 1
            n = len(self._queue)
            self._queue_len_samples.append(n)
            self.queue_len_max = max(self.queue_len_max, n)

            if self._stop or n >= self.config.queue_capacity:
                self.counts["dropped_full"] += 1
                return False

            self._queue.append(item)
            self._cond.notify_all()

        return True

    def offer_gripper(self, frame: GripperDepthFrame) -> bool:
        """그리퍼 깊이 프레임을 넣는다(수신 스레드). 꺼져 있으면 False. 가득이면 가장 오래된 것을 버린다."""
        if self._gripper is None:
            return False

        with self._cond:
            if self._stop:
                return False

            self.gripper_counts["offered"] += 1

            if len(self._grip_in) >= GRIPPER_KEEP:
                self._grip_in.popleft()
                self.gripper_counts["dropped_old"] += 1

            self._grip_in.append(GripperItem(frame, self._grip_epoch, self._clock_ns()))
            self._cond.notify_all()

        return True

    def request_reset(self, reason: str) -> None:
        """다음 항목 전에 전 트랙 종료를 요청한다(추적 스레드가 수행). 메모리 작업만."""
        with self._cond:
            self._reset_reason = reason
            self.counts["reset_requests"] += 1
            self._cond.notify_all()

    # ------------------------------------------------------------------ worker

    def wait(self, context, timeout_s: float | None = None):
        """그 작업 결과(`TrackFrame` | 사유 문자열)를 상한까지 기다린다. 못 받으면 None(`track_pending`)."""
        key = self._k(frame_key(context))
        limit = self.config.handoff_wait_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + limit

        with self._cond:
            while key not in self._results and not self._stop:
                remaining = deadline - time.monotonic()

                if remaining <= 0:
                    break

                self._cond.wait(remaining)

            found = key in self._results
            value = self._results.get(key)
            self.counts["handoff_ok" if found else "handoff_pending"] += 1
            self._handoff_window.append(found)

        return value if found else None

    def peek(self, context, display=None):
        """detect 영상 그리기용(C forwarder 스레드): 그 작업 결과를 기다리지 않고 읽는다.

        결과가 없으면 `display_prev` 설정 시 직전 W 결과를 위치로 짝지은 `PrevDisplay`, 그것도 안 되면 None.
        밀림 규칙 입력과 추적 판정은 바꾸지 않는다.
        """
        key = self._k(frame_key(context))

        with self._cond:
            value = self._results.get(key)
            prev, reset_pending = self._display_prev, self._reset_reason is not None

        cfg = self.config.display_prev
        miss = None

        if value is None and display is not None and cfg is not None:
            shown, why = match_prev(prev, context, display, cfg, reset_pending=reset_pending)
            value, miss = shown, (None if shown is not None else why)
        elif value is None:
            miss = "off" if cfg is None else "no_display"

        kind = ("pending" if value is None else "reason" if isinstance(value, str) else
                "found" if isinstance(value, TrackFrame) else "prev")

        with self._cond:
            self.display_counts[kind] += 1

            if miss is not None:
                self.display_prev_miss[miss] = self.display_prev_miss.get(miss, 0) + 1

        return value

    @staticmethod
    def _k(key: FrameKey) -> tuple:
        return tuple(getattr(key, f.name) for f in fields(key))

    # ------------------------------------------------------------------ 수명

    def start(self) -> None:
        """추적 스레드(그리퍼가 켜져 있으면 관측 스레드도)를 시작한다."""
        self._thread = threading.Thread(target=self._loop, name="eyetoarm-live-world-tracking", daemon=True)
        self._thread.start()

        if self._gripper is not None:
            self._grip_thread = threading.Thread(target=self._grip_loop, name="eyetoarm-gripper-observe", daemon=True)
            self._grip_thread.start()

        if self._stream is not None:
            self._stream.start()

    @property
    def thread(self) -> threading.Thread | None:
        """추적 스레드(정리 확인용)."""
        return self._thread

    def close(self, limit_s: float) -> bool:
        """멈춤 → join → 기록 닫기. 재시도 가능. 스레드 비정상 종료는 RuntimeError(정리 확인 아님)."""
        with self._cond:
            self._stop = True
            self._cond.notify_all()

        for th in (self._grip_thread, self._thread):
            if th is not None:
                th.join(timeout=limit_s)

                if th.is_alive():
                    return False

        # 추적 스레드가 끝난 뒤(끝 알림을 넣은 뒤) 남은 발행을 보낸다
        if self._stream is not None and not self._stream.close(limit_s):
            return False

        if not self._fh.closed:
            try:
                self._fh.close()
            except OSError as exc:
                self.write_error = self.write_error or f"close: {type(exc).__name__}: {exc}"

        if self.crash is not None:
            raise RuntimeError(f"world tracking thread crashed: {self.crash}")

        return True

    # ------------------------------------------------------------------ 추적 스레드

    def _loop(self) -> None:
        try:
            self._loop_body()
            self._finish()
        except Exception as exc:   # 추적 스레드 최상단: 비정상 종료를 남긴다(이후 결과 없음 → 번호 None)
            with self._cond:
                self.crash = f"{type(exc).__name__}: {exc}"
                self._stop = True
                self._cond.notify_all()

    def _loop_body(self) -> None:
        while True:
            with self._cond:
                while (not self._queue and not self._stop and self._reset_reason is None
                       and not self._gripper_due(self._clock_ns())):
                    self._cond.wait(0.05)

                if self._stop:
                    break

                reset, self._reset_reason = self._reset_reason, None
                item = self._queue.popleft() if self._queue else None
                now = self._clock_ns()
                lag_reset = None
                grips = self._take_gripper(item, now)

                if item is not None:
                    lag_ms = (now - item.enqueued_ns) / 1e6
                    self._lag_ms_samples.append(lag_ms)
                    window = list(self._handoff_window)
                    pending_ratio = (window.count(False) / len(window)) if len(window) >= self.config.window_n else 0.0
                    lagging = lag_ms > self.config.lag_limit_ms   # 추적 큐 대기만(B 기다림 실패는 진단)
                    pending_high = pending_ratio > self.config.pending_ratio_limit

                    if pending_high and not self._pending_high:   # 진단: 한도를 넘는 구간 시작마다 1회
                        self.counts["pending_high"] += 1

                    self._pending_high = pending_high
                    self._lag_streak = self._lag_streak + 1 if lagging else 0
                    cooled = (self._last_lag_reset_ns is None
                              or (now - self._last_lag_reset_ns) / 1e6 >= self.config.lag_cooldown_ms)

                    if self._lag_streak >= self.config.lag_persist_n and cooled:
                        # 밀림 지속: 최신 1개만 남기고 다시 시작(꺼낸 항목도 오래된 것으로 버림)
                        newest = self._queue[-1] if self._queue else item
                        # 버린 수 = 꺼낸 항목 + 큐에서 최신을 뺀 나머지 = 큐 길이
                        skipped = len(self._queue)
                        self._queue.clear()
                        self.counts["stale_skipped"] += skipped
                        self.counts["lag_resets"] += 1
                        self._last_lag_reset_ns = now
                        self._lag_streak = 0
                        self._handoff_window.clear()
                        self._pending_high = False
                        lag_reset = {"t_mono_ns": now, "lag_ms": round(lag_ms, 3), "pending_ratio": round(pending_ratio, 3),
                                     "queue_len": skipped, "skipped": skipped}
                        self.lag_reset_log.append(lag_reset)
                        item = newest
                        self._grip_queue.clear()
                        self._grip_in.clear()
                        self._grip_epoch += 1
                        grips = []

            if reset is not None:
                self._apply_reset(reset)

            if lag_reset is not None:
                self._clear_display_prev()
                self._tracker.reset("lag")
                self._stream_reset("lag")
                self._write({"kind": "lag_reset", **lag_reset})

            # stamp 순서: 링크보다 이른 그리퍼 → 링크 → 남은(대기 만료) 그리퍼
            limit = getattr(item.frame, "ros_stamp_ns", None) if item is not None else None
            before = [g for g in grips if limit is None or g.frame.ros_stamp_ns < limit]

            for g in before:
                self._process_gripper(g)

            if item is not None:
                self._process(item, lag_ms if lag_reset is None else (now - item.enqueued_ns) / 1e6)

            for g in grips[len(before):]:
                self._process_gripper(g)

    def _gripper_due(self, now_ns: int) -> bool:
        """조건변수 안: 대기 상한을 넘긴 그리퍼 항목이 있는가."""
        if not self._grip_queue:
            return False

        return (now_ns - self._grip_queue[0].enqueued_ns) / 1e6 >= self._gripper.config.reorder_wait_ms

    def _take_gripper(self, item, now_ns: int) -> list[GripperItem]:
        """조건변수 안: 링크 항목보다 이른 그리퍼 항목과 대기 상한을 넘긴 항목을 시각 순서로 꺼낸다."""
        if not self._grip_queue:
            return []

        limit = getattr(item.frame, "ros_stamp_ns", None) if item is not None else None
        wait_ns = self._gripper.config.reorder_wait_ms * 1e6
        take = [g for g in self._grip_queue
                if (limit is not None and g.frame.ros_stamp_ns < limit) or now_ns - g.enqueued_ns >= wait_ns]

        for g in take:
            self._grip_queue.remove(g)

        return sorted(take, key=lambda g: g.frame.ros_stamp_ns)

    def _clear_display_prev(self) -> None:
        """추적 스레드: 재설정 전 결과를 표시에 쓰지 않게 직전 사본을 지운다."""
        with self._cond:
            self._display_prev = None

    def _apply_reset(self, reason: str) -> None:
        """추적 스레드: 전 트랙 종료 + 기록."""
        self._clear_display_prev()
        self._bump_gripper_epoch()
        self._tracker.reset(reason)
        self.last_reset = reason
        self._stream_reset(reason)
        self._write({"kind": "reset", "reason": reason, "t_mono_ns": self._clock_ns()})

    def _finish(self) -> None:
        """추적 스레드 끝: 남은 초기화 요청을 처리하고 종료 사건을 한 번 쓴다."""
        with self._cond:
            pending, self._reset_reason = self._reset_reason, None

        if pending is not None:
            self._apply_reset(pending)

        rec = {"kind": "closed", "pending_reset_applied": pending is not None, "last_reset": self.last_reset,
               "t_mono_ns": self._clock_ns()}

        if self._stream is not None:
            self._stream.offer(marker_message(run_id=self.run_id, source_id=self._stream_source,
                                              tracker_epoch=self._tracker.tracker_epoch, handoff_on=self.handoff_on))
            rec["stream"] = self._stream.summary()["counts"]

        self._write(rec)

    def _process(self, item: WorldTrackItem, lag_ms: float) -> None:
        frame, key = item.frame, item.key
        tf_result = self._tf_lookup(frame)

        if tf_result.wait_ms is not None:
            self._tf_wait_samples.append(float(tf_result.wait_ms))

        rec = {"kind": "frame", "image_seq": key.image_seq, "job_seq": key.job_seq, "generation": key.generation,
               "ros_stamp_ns": getattr(frame, "ros_stamp_ns", None), "timestamp_ms": getattr(frame, "timestamp_ms", None),
               "lag_ms": round(lag_ms, 3), "tf_status": tf_result.status}

        if tf_result.status != "ok" or item.camera is None:
            reason = f"tf_{tf_result.status}" if tf_result.status != "ok" else "no_camera"
            value: object = reason
            self.not_updated[reason] = self.not_updated.get(reason, 0) + 1
            rec["result"] = reason
        else:
            cam = item.camera
            inputs = world_inputs(item.display, K=cam.K, dist=cam.dist, T_W_C=tf_result.tf.T_W_C, plane_z_m=self._plane_z,
                                  view_bias=self._view_bias, width=getattr(frame, "width", None),
                                  height=getattr(frame, "height", None), mask_fallback=self._tracker.config.mask_fallback)
            ts = int(frame.timestamp_ms)
            occluded = None   # 가림 판정 함수(가시성 켬·팔 TF 있음일 때만). 종류 고정의 팔 근처 판정에도 쓴다

            if self._visibility is None:
                track = self._tracker.update(key, ts, inputs)
            else:
                occluded, occ_status = self._link_occluded(tf_result.tf.T_W_C, frame)
                track = self._tracker.update(key, ts, inputs, visible=self._link_visible(cam, tf_result.tf.T_W_C, frame),
                                             occluded=occluded)
                rec["occlusion"] = occ_status
                rec["occluded_ids"] = list(self._tracker.last_occluded)
                rec["held"] = self._tracker.held_snapshot(ts)   # hold·가림 트랙 예측(재연결 분석용)
                rec["belt_velocity"] = self._tracker.last_velocity

            if self.handoff_on:
                self._link_generation = key.generation
                self._publish_snapshot(ts)

                if track.event == "scope_changed":
                    self._bump_gripper_epoch()

                if (not track.updated and self._last_gripper_ts is not None and ts <= self._last_gripper_ts):
                    self._gcount("stale_after_gripper")

            value = track
            self.counts["updated"] += int(track.updated)
            by_index = {i.index: i for i in inputs}
            belt = self._tracker.last_belt_vy
            rec.update({"updated": track.updated, "event": track.event, "tracker_epoch": track.tracker_epoch,
                        "belt_vy_mm_per_s": None if belt is None else round(belt * 1000.0, 3),
                        "tracks": [{"instance_index": a.instance_index, "part_track_id": a.part_track_id,
                                    "state": a.state.value, "hits": a.hits, "reason": a.reason,
                                    "input": by_index[a.instance_index].kind,
                                    "center_mm": _r(by_index[a.instance_index].center_mm),
                                    "center_raw_mm": _r(by_index[a.instance_index].center_raw_mm),
                                    "length_mm": _r1(by_index[a.instance_index].length_mm),
                                    "plane_z_m": by_index[a.instance_index].plane_z_m,   # W로 올린 높이
                                    "plane_spec": by_index[a.instance_index].plane_spec} for a in track.assignments]})

            if self.handoff_on:
                rec["camera"] = "link"

            if self._stream is not None:
                rec["stream"] = self._stream_frame(key, frame, ts, track, inputs, occluded)

        # 갱신한 결과면 확정 번호의 표시용 박스 사본을 둔다(추적 판정·송신에는 쓰지 않음)
        snap = snapshot(value, item.display) if isinstance(value, TrackFrame) else None

        with self._cond:
            self.counts["processed"] += 1
            self._results[self._k(key)] = value

            if snap is not None:
                self._display_prev = snap

            while len(self._results) > RESULT_KEEP:
                self._results.popitem(last=False)

            self._cond.notify_all()

        self._write(rec)
        self._write_events()

    # ------------------------------------------------------------------ 추적 흐름 발행(추적 스레드)

    def _stream_reset(self, reason: str) -> None:
        """재설정 알림을 발행 큐에 넣고 종류 기억을 비운다(켜졌을 때만)."""
        if self._stream is None:
            return

        self._class_lock.clear()
        self._stream.offer(reset_message(run_id=self.run_id, source_id=self._stream_source, reason=reason,
                                         tracker_epoch=self._tracker.tracker_epoch, handoff_on=self.handoff_on))

    def _stream_frame(self, key: FrameKey, frame, ts: int, track: TrackFrame, inputs, occluded=None) -> dict:
        """링크 프레임 결과 → 확정 트랙 발행. 범위가 바뀌어 추적기가 비워졌으면 재설정 알림을 먼저 낸다. 기록용 결과를 돌려준다.

        `occluded(x_mm, y_mm, margin_mm)`은 이 사진의 가림 판정 함수(없으면 None). 종류 고정이 켜져 있으면 후보 중심이
        캡슐 + `near_arm_mm` 안인 사진을 팔 근처로 보고 종류를 바꾸지 않는다(#368).
        """
        t0 = self._clock_ns()
        self._stream_source = key.source_id

        if track.event == "scope_changed":
            self._stream_reset("scope_changed")

        by_index = {i.index: i for i in inputs}
        lock = self._class_lock
        decisions = []

        for a in track.assignments:
            c = by_index[a.instance_index]

            if a.part_track_id is None or c.plane_spec is None:
                continue

            near = False

            if lock.config is not None and occluded is not None and c.center_mm is not None:
                near = bool(occluded(c.center_mm[0], c.center_mm[1], lock.config.near_arm_mm))

            d = lock.observe(a.part_track_id, c.plane_spec, c.plane_z_m, c.length_mm, near)

            if lock.config is not None and d["decision"] not in ("first", "same"):
                decisions.append(d)

        velocity = self._tracker.last_velocity if self._visibility is not None else None
        belt = self._tracker.last_belt_vy

        if velocity is not None and velocity.get("vy_mm_per_s") is not None:
            belt_mm_s = velocity["vy_mm_per_s"]   # 긴 기준선 공통 속도(hold 예측에 쓰는 값)
        else:
            belt_mm_s = None if belt is None else belt * 1000.0

        snap = self._tracker.stream_snapshot(ts)
        msg = frame_message(run_id=self.run_id, source_id=key.source_id, image_seq=key.image_seq, generation=key.generation,
                            tracker_epoch=track.tracker_epoch, capture_stamp_ns=int(getattr(frame, "ros_stamp_ns", 0) or 0),
                            timestamp_ms=ts, snapshot=snap, belt_velocity_mm_per_s=belt_mm_s, handoff_on=self.handoff_on,
                            classes=lock.classes, default_plane_z_m=self._plane_z_m())
        result = self._stream.offer(msg)
        out = {"result": result, "tracks": len(msg["tracks"]), "build_ms": round((self._clock_ns() - t0) / 1e6, 3)}

        with self._cond:
            self._class_counts = dict(lock.counts)

        if decisions:   # 종류 고정이 이 사진의 종류를 받아들이지 않았거나 바꾼 트랙(기록용)
            out["class_lock"] = decisions

        return out

    # ------------------------------------------------------------------ 그리퍼(추적 스레드)

    def _gcount(self, key: str, n: int = 1) -> None:
        """그리퍼 계수(조건변수 안)."""
        with self._cond:
            self.gripper_counts[key] += n

    def _greason(self, reason: str) -> None:
        """그리퍼 무효 사유 계수(조건변수 안)."""
        with self._cond:
            self.gripper_reasons[reason] = self.gripper_reasons.get(reason, 0) + 1

    def _bump_gripper_epoch(self) -> None:
        """재설정: 그리퍼 큐를 비우고 회차를 올린다."""
        with self._cond:
            self._grip_queue.clear()
            self._grip_in.clear()
            self._grip_epoch += 1

    def _plane_z_m(self) -> float:
        """화면 판정에 쓰는 볼트 높이(m)."""
        return self._plane_z.default_z_m if isinstance(self._plane_z, BoltPlanes) else self._plane_z

    def _link_occluded(self, T_W_C, frame):
        """링크 프레임 촬영 stamp의 팔 링크 TF로 가림 판정 함수를 만든다. TF가 하나라도 없으면 (None, 사유) = 가림 아님."""
        t0 = time.perf_counter()
        poses, status = {}, "ok"

        for link in dict.fromkeys(o.link for o in self._visibility.occluders):
            res = self._tf_lookup(SimpleNamespace(ros_stamp_ns=getattr(frame, "ros_stamp_ns", None), optical_frame=link))

            if res.status != "ok":
                status = f"tf_{res.status}:{link}"
                break

            poses[link] = res.tf.T_W_C

        fn = None

        if status == "ok":
            caps = world_capsules(self._visibility.occluders, poses)
            fn = occluded_fn(np.asarray(T_W_C)[:3, 3], caps, self._plane_z_m())
        else:
            self.occlusion_tf_fail += 1

        self._occl_ms.append((time.perf_counter() - t0) * 1000.0)

        return fn, status

    def _link_visible(self, cam, T_W_C, frame):
        """링크 화면 판정 함수: 예측 W(mm)를 볼트 평면에 놓고 투영해 화면(여백) 안·카메라 앞인지."""
        margin = self._visibility.margin_px
        width, height = getattr(frame, "width", None), getattr(frame, "height", None)
        z_m = self._plane_z_m()

        def visible(x_mm: float, y_mm: float) -> bool:
            """링크 화면 안인가."""
            p = np.array([[x_mm / 1000.0, y_mm / 1000.0, z_m]])
            _, zc = project_w(p * 1000.0, T_W_C, cam.K)

            if not zc[0] > 0 or width is None or height is None:
                return bool(zc[0] > 0)

            u, v = w_to_image(p, cam.K, cam.dist, T_W_C)[0]

            return bool(-margin <= u < width + margin and -margin <= v < height + margin)

        return visible

    def _gripper_visible(self, frame: GripperDepthFrame, T_W_C):
        """그리퍼 화면 판정 함수: 투영 화소가 화면 안이고 그 화소 깊이가 유효(near 이상)인지. 근접 무효는 화면 밖으로 본다."""
        margin = self._visibility.margin_px if self._visibility is not None else 0.0
        near = self._gripper.config.near_m
        z_mm = self._plane_z_m() * 1000.0
        depth = frame.depth_m

        def visible(x_mm: float, y_mm: float) -> bool:
            """그리퍼 화면 안인가."""
            uv, zc = project_w([[x_mm, y_mm, z_mm]], T_W_C, frame.K)
            u, v = uv[0]

            if not zc[0] > 0 or not (-margin <= u < frame.width + margin and -margin <= v < frame.height + margin):
                return False

            ui, vi = min(max(int(u), 0), frame.width - 1), min(max(int(v), 0), frame.height - 1)
            d = float(depth[vi, ui])

            return math.isfinite(d) and d >= near

        return visible

    def _grip_loop(self) -> None:
        """관측 스레드: 그리퍼 프레임마다 TF·화면 판정(트랙 사본)·깊이 관측을 계산해 결과 큐에 넣는다."""
        try:
            while True:
                with self._cond:
                    while not self._grip_in and not self._stop:
                        self._cond.wait(0.05)

                    if self._stop:
                        return

                    item = self._grip_in.popleft()
                    snapshot = list(self._snapshot)

                self._grip_result(item, snapshot)
        except Exception as exc:   # 관측 스레드 최상단: 비정상 종료를 남기고 추적 스레드도 멈춘다
            with self._cond:
                self.crash = f"gripper observe thread: {type(exc).__name__}: {exc}"
                self._stop = True
                self._cond.notify_all()

    def _grip_result(self, item: GripperItem, snapshot: list) -> None:
        """관측 스레드: TF → 사본 트랙이 화면 안이면 관측 → 결과."""
        t0 = time.perf_counter()
        frame = item.frame
        tf_result = self._tf_lookup(frame)
        T, in_view, obs = None, (), None

        if tf_result.status == "ok":
            T = tf_result.tf.T_W_C
            visible = self._gripper_visible(frame, T)
            in_view = tuple(pid for pid, x, y in snapshot if visible(x, y))

            if in_view:
                obs = self._observe(frame, T, self._gripper.config, self._gripper.roi_xy_m)

        result = GripperResult(frame, item.epoch, self._clock_ns(), tf_result.status, T, in_view, obs,
                               (time.perf_counter() - t0) * 1000.0)

        with self._cond:
            if obs is not None:
                self.gripper_counts["observed"] += 1

            if len(self._grip_queue) >= GRIPPER_KEEP:
                self._grip_queue.popleft()
                self.gripper_counts["dropped_old"] += 1

            self._grip_queue.append(result)
            self._cond.notify_all()

    def _publish_snapshot(self, timestamp_ms: int) -> None:
        """추적 스레드: 관측 스레드의 화면 판정용 확정 트랙 예측 사본."""
        if self._gripper is None:
            return

        snap = self._tracker.predicted_confirmed(timestamp_ms)

        with self._cond:
            self._snapshot = snap

    def _process_gripper(self, g: GripperResult) -> None:
        """추적 스레드: 관측 결과 하나를 확정 트랙과 짝짓고 기록한다(무거운 계산 없음)."""
        t0 = time.perf_counter()
        frame = g.frame
        rec = {"kind": "gripper_frame", "ros_stamp_ns": frame.ros_stamp_ns, "timestamp_ms": frame.timestamp_ms,
               "tf_status": g.tf_status, "observe_ms": round(g.observe_ms, 3)}

        with self._cond:
            current = self._grip_epoch
            self.gripper_counts["processed"] += 1

        if g.epoch != current:
            self._gcount("old_epoch")
            self._write({**rec, "result": "old_epoch"})
            return

        if self._link_generation is None:
            self._gcount("no_link_yet")
            self._write({**rec, "result": "no_link_yet"})
            return

        if g.tf_status != "ok":
            reason = f"tf_{g.tf_status}"
            self._greason(reason)
            self._write({**rec, "result": reason})
            return

        rec["tracks_in_view"] = list(g.in_view)

        if g.obs is None:
            self._gcount("no_track_in_view")
            self._tracker.note_camera("gripper", frame.timestamp_ms)
            rec["result"] = "no_track_in_view"
            self._write(rec)
            return

        obs = g.obs
        rec.update(valid=obs.valid, reason=obs.reason, diag=obs.diag)

        if not obs.valid:
            self._greason(obs.reason)

        comps = list(obs.components)
        inputs = [GripperInput(index=c.index, center_mm=c.center_mm, length_mm=c.length_mm, angle_deg=c.angle_deg,
                               clipped=c.clipped) for c in comps]
        visible = self._gripper_visible(frame, g.T_W_C) if self._visibility is not None else None
        up = self._tracker.update_gripper(self._link_generation, frame.timestamp_ms, inputs, visible=visible)
        self._gcount("updated", int(up.updated))
        self._gcount("late", int(up.updated and up.late))

        if up.updated:
            self._last_gripper_ts = max(self._last_gripper_ts or 0, frame.timestamp_ms)
        rec.update(updated=up.updated, late=up.late, not_updated=up.reason,
                   components=[{"index": c.index, "center_mm": _r(c.center_mm), "length_mm": _r1(c.length_mm),
                                "angle_deg": _r1(c.angle_deg), "top_z_mm": _r1(c.top_z_mm), "n_px": c.n_px, "clipped": c.clipped,
                                "part_track_id": a.part_track_id, "outcome": a.outcome}
                               for c, a in zip(comps, up.assignments)])
        ms = (time.perf_counter() - t0) * 1000.0
        self._gripper_ms.append(ms)
        self._observe_ms.append(g.observe_ms)
        rec["proc_ms"] = round(ms, 3)
        self._write(rec)
        self._write_events()
        self._publish_snapshot(frame.timestamp_ms)

    def _write_events(self) -> None:
        """추적기의 넘기기·만료 사건을 기록한다(켜졌을 때만)."""
        if not self.handoff_on:
            return

        for e in self._tracker.drain_events():
            self._write({"kind": "track_event", "event": e["kind"], **{k: v for k, v in e.items() if k != "kind"}})

    def _write(self, rec: dict) -> None:
        if self.write_error is not None:
            return

        try:
            self._fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
            self._fh.flush()
        except (OSError, ValueError) as exc:
            self.write_error = f"{type(exc).__name__}: {exc}"

    # ------------------------------------------------------------------ 요약

    def problem(self) -> str | None:
        """진입점이 종료 1로 드러낼 문제(스레드 비정상 종료·기록 실패). 밀림·번호 없음은 문제가 아니다(지표로만)."""
        if self.crash is not None:
            return f"world tracking crashed: {self.crash}"

        if self.write_error is not None:
            return f"world_tracks write error: {self.write_error}"

        return None

    def summary(self) -> dict:
        """요약 블록 `world_tracking`."""
        with self._cond:
            counts = dict(self.counts)
            total = counts["handoff_ok"] + counts["handoff_pending"]
            lag = {"queue_len_max": self.queue_len_max, "queue_len_p95": _pct(list(self._queue_len_samples), 0.95),
                   "lag_ms_median": _pct(list(self._lag_ms_samples), 0.5), "lag_ms_p95": _pct(list(self._lag_ms_samples), 0.95),
                   "lag_ms_max": _pct(list(self._lag_ms_samples), 1.0),
                   "tf_wait_ms_median": _pct(list(self._tf_wait_samples), 0.5),
                   "tf_wait_ms_p95": _pct(list(self._tf_wait_samples), 0.95),
                   "handoff_pending_ratio": None if total == 0 else round(counts["handoff_pending"] / total, 4),
                   "dropped_full": counts["dropped_full"], "stale_skipped": counts["stale_skipped"],
                   "lag_resets": counts["lag_resets"], "lag_reset_log": list(self.lag_reset_log),
                   "pending_high": counts["pending_high"]}
            # detect 영상 W 번호 조회 계수(found·prev·pending·reason)와 prev 실패 사유
            display = {**self.display_counts, "prev_miss": dict(sorted(self.display_prev_miss.items()))}
            counts["queue_left"] = len(self._queue)   # offered = processed + dropped + stale + left

            handoff = None if not self.handoff_on else {
                "visibility": None if self._visibility is None else self._visibility.to_json(),
                "gripper": None if self._gripper is None else self._gripper.config.to_json(),
                "gripper_counts": {**self.gripper_counts, "queue_left": len(self._grip_queue) + len(self._grip_in)},
                "gripper_reasons": dict(sorted(self.gripper_reasons.items())),
                "gripper_proc_ms_median": _pct(list(self._gripper_ms), 0.5),
                "gripper_proc_ms_p95": _pct(list(self._gripper_ms), 0.95),
                "observe_ms_median": _pct(list(self._observe_ms), 0.5), "observe_ms_p95": _pct(list(self._observe_ms), 0.95),
                "occlusion_ms_median": _pct(list(self._occl_ms), 0.5), "occlusion_ms_p95": _pct(list(self._occl_ms), 0.95),
                "occlusion_tf_fail": self.occlusion_tf_fail}

        return {"config": self.config.to_json(), "counts": counts, "not_updated": dict(sorted(self.not_updated.items())),
                "lag": lag, "tracker": self._tracker.stats(), "diagnostics": dict(self.diagnostics),
                "display_lookup": display,
                "crash": self.crash, "write_error": self.write_error, **({} if handoff is None else {"handoff": handoff}),
                **({} if self._stream is None else {"track_stream": {**self._stream.summary(),
                                                                      "class_lock": dict(self._class_counts)}})}


def _r(p):
    return None if p is None else [round(p[0], 2), round(p[1], 2)]


def _r1(v):
    return None if v is None else round(v, 2)


__all__ = ["GripperItem", "GripperLive", "LagConfigError", "LiveWorldConfig", "LiveWorldTracking", "WorldTrackItem", "frame_key",
           "load_handoff_config", "load_world_tracking_config", "parse_live_world"]
