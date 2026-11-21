"""실시간 W 추적 스레드(#217 1단계): 실시간 기록 연결 안에서 W 추적기를 제어 흐름 밖에서 돌리고, worker에 작업별 결과를 넘긴다.

근거: 계획 `_workspace/217_builder2_world_tracking_plan_r0.md`(Q1 (c) 별도 추적 스레드) + `_r1.md`(밀림 지표·초기화 규칙),
reviewer 보충 1~3(정답 격리: C `speed_mps`·컨베이어 토픽 속도 값은 예측에 넣지 않음).

- **넣기(제어 흐름, `offer`)**: 항목(프레임 참조·AI 결과 참조·작업 키·카메라)을 유한 큐에 넣기만 한다(O(1), TF 조회 없음).
  큐가 가득이면 **새 항목을 버린다**(`dropped_full`, 시간 순서 유지).
- **추적 스레드**(`eyetoarm-live-world-tracking`): 큐를 순서대로 꺼내 촬영 stamp TF 정확 조회(주입 `tf_lookup(frame)`, 최신 TF·픽셀
  대체 없음) → `world_inputs`(B와 같은 투영, 볼트 평면, 시야 보정 뒤 끝점) → `WorldTracker.update`. TF가 `ok`가 아니면 갱신하지 않고
  사유(`tf_<상태>`)를 결과로 둔다. 결과는 작업 키 → `TrackFrame | 사유 문자열` 작은 표(최근 `RESULT_KEEP`개)에 둔다.
- **넘기기(worker, `wait`)**: 그 작업 키 결과를 최대 `handoff_wait_s` 기다린다. 못 받으면 None(`track_pending`).
- **밀림 규칙(r1 L2)**: 꺼낸 항목의 `lag_ms`(넣은 시각부터) > `lag_limit_ms`이거나 최근 `window_n`개 넘김의 대기 실패 비율 >
  `pending_ratio_limit`이면 「밀림」. `lag_persist_n`개 연속이면 큐에 최신 1개만 남기고 전 트랙 종료(`reset("lag")`, 번호 재사용 없음)
  후 그 항목부터 다시 한다. 초기화 뒤 `lag_cooldown_ms` 동안은 다시 하지 않는다. 추적을 끄지는 않는다.
- **초기화 요청(`request_reset`)**: 실행 닫힘·세대/보정 변경 때 연결부가 부른다. 다음 항목 전에 추적 스레드가 `reset`한다(단일 소유).
- **기록**: 출력 폴더 `world_tracks.jsonl`(추적 스레드가 직접 씀, 프레임마다 배정·W 위치·보정 전후·벨트 추정 속도·지연).
  진단 전용 값(`diagnostics`, C `speed_mps` 조회 등)은 요약에만 넣고 **추적기에는 전달하지 않는다**.
- **마스크 입력(#226)**: 추적기 설정 `mask_fallback`이면 `world_inputs(..., mask_fallback=True)`로 끝점 없는 볼트의 마스크 W 평균을 넣고,
  `world_tracks.jsonl` 배정마다 입력 종류(`input`: endpoint | mask | none)를 남긴다.
- **리셋·종료 기록(#226 D3)**: 초기화 요청을 처리할 때 `{"kind": "reset", "reason"}`을 쓰고, 닫을 때 처리 못 한 초기화 요청이 남아 있으면
  그 자리에서 처리한 뒤 `{"kind": "closed", "pending_reset_applied", "last_reset"}`을 한 번 쓴다.
- ROS·Qt를 import하지 않는다.
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

from eyetoarm.vision.tracking import FrameKey, TrackFrame
from eyetoarm.vision.world_tracking import BoltPlanes, WorldTracker, world_inputs
from eyetoarm.vision.world_tracking_display import (DisplayPrevConfig, DisplayPrevConfigError, PrevSnapshot, match_prev,
                                                    parse_display_prev, snapshot)

RESULT_KEEP = 64


class LagConfigError(ValueError):
    """밀림 규칙 설정 오류."""


@dataclass(frozen=True)
class LiveWorldConfig:
    """추적 스레드 시험값(r0·r1). 모두 필수."""

    queue_capacity: int
    handoff_wait_s: float
    lag_limit_ms: float
    window_n: int
    pending_ratio_limit: float
    lag_persist_n: int
    lag_cooldown_ms: float
    conveyor_probe_timeout_s: float   # #226 D2: C speed_mps 진단 조회 상한(진단 전용, 추적기에 전달 안 함)
    # #242: detect 영상 직전 결과 표시(화면 전용). 설정 파일에서는 필수, 코드에서 직접 만들 때 None이면 직전 결과를 쓰지 않는다
    display_prev: DisplayPrevConfig | None = None

    def to_json(self) -> dict:
        """요약용 사전."""
        doc = {f.name: getattr(self, f.name) for f in fields(self)}
        doc["display_prev"] = None if self.display_prev is None else self.display_prev.to_json()

        return doc


def parse_live_world(doc) -> LiveWorldConfig:
    """`queue_capacity`·`handoff_wait_s` + `lag` 블록(r1). 모르는 키·누락·범위 오류는 `LagConfigError`."""
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
    """`display_prev` 블록(#242). 오류는 `LagConfigError`로 바꾼다(진입점이 같은 설정 오류로 다룸)."""
    try:
        return parse_display_prev(doc)
    except DisplayPrevConfigError as exc:
        raise LagConfigError(str(exc)) from exc


def load_world_tracking_config(path: str):
    """W 추적 시험 설정 파일 → (`WorldTrackerConfig`, `LiveWorldConfig`, sha256).

    키: `status`(반드시 `test_only`)·`note`(선택)·`tracker`·`live`·`conveyor_state`(지금은 null만: 토픽 이름·형식 확정 전 구독 자리).
    """
    import hashlib

    from eyetoarm.vision.world_tracking import WorldTrackerConfigError, parse_world_tracker

    raw = Path(path).read_bytes()

    try:
        doc = json.loads(raw)
    except ValueError as exc:
        raise LagConfigError(f"world tracking config is not JSON: {exc}") from exc

    keys = {"status", "tracker", "live", "conveyor_state"}

    if not isinstance(doc, dict) or not keys <= set(doc) or set(doc) - keys - {"note"}:
        raise LagConfigError(f"world tracking config must have {sorted(keys)} (+ optional note)")

    if doc["status"] != "test_only":
        raise LagConfigError("world tracking config status must be 'test_only' (development values)")

    if doc["conveyor_state"] is not None:
        raise LagConfigError("conveyor_state topic is not supported yet (C topic name/type undecided); keep it null")

    try:
        tracker = parse_world_tracker(doc["tracker"])
    except WorldTrackerConfigError as exc:
        raise LagConfigError(str(exc)) from exc

    return tracker, parse_live_world(doc["live"]), hashlib.sha256(raw).hexdigest()


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
                 out_dir: str, view_bias=None, clock_ns: Callable[[], int] = time.monotonic_ns) -> None:
        """`tf_lookup(frame) → TfLookupResult`(촬영 stamp 정확 조회). 기록 파일(`world_tracks.jsonl`, 새 파일)을 연다.

        `plane_z_m`은 볼트 투영 높이 하나(m) 또는 종류별 높이 `BoltPlanes`(#250 P1, `world_inputs`가 끝점 볼트마다 고름)다.
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
                                      "stale_skipped", "lag_resets", "reset_requests")}
        self.not_updated: dict[str, int] = {}
        # #239: detect 영상 그리기 조회(`peek`) 계수. 밀림 규칙 입력(handoff_*·대기 창)과 섞지 않는다
        # #242: prev = 같은 작업 결과가 없어 직전 결과를 위치로 짝지어 그림, pending = 그것도 못 함(사유는 display_prev_miss)
        self.display_counts = {"found": 0, "prev": 0, "pending": 0, "reason": 0}
        self.display_prev_miss: dict[str, int] = {}
        self._display_prev: PrevSnapshot | None = None   # 직전 갱신 W 결과의 표시용 사본(조건변수 안, 추적 스레드가 바꿈)
        self.lag_reset_log: list[dict] = []
        self.last_reset: str | None = None      # 마지막으로 처리한 초기화 사유(D3)
        self._queue_len_samples: collections.deque[int] = collections.deque(maxlen=4096)
        self._lag_ms_samples: collections.deque[float] = collections.deque(maxlen=4096)
        self._tf_wait_samples: collections.deque[float] = collections.deque(maxlen=4096)
        self.queue_len_max = 0
        self.diagnostics: dict = {}   # 진단 전용(C speed_mps 조회 등). 추적기에 전달하지 않는다

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
        """detect 영상 그리기용(#239, C forwarder 스레드): 그 작업 결과를 **기다리지 않고** 읽는다.

        그 작업 키(FrameKey 전체)의 값이 있으면 그것(`TrackFrame` | 사유 문자열). 없고(아직 없음·표에서 밀려남) `display`(그 작업 AI 결과)와
        설정 `display_prev`가 있으면(#242) 직전 W 결과를 위치로 짝지은 표시 전용 `PrevDisplay`, 그것도 안 되면 None.
        `handoff_*` 계수·대기 창(밀림 규칙 입력)과 추적 판정은 바꾸지 않고 `display_counts`·`display_prev_miss`만 센다.
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
        """추적 스레드를 시작한다."""
        self._thread = threading.Thread(target=self._loop, name="eyetoarm-live-world-tracking", daemon=True)
        self._thread.start()

    @property
    def thread(self) -> threading.Thread | None:
        """추적 스레드(정리 확인용)."""
        return self._thread

    def close(self, limit_s: float) -> bool:
        """멈춤 → join → 기록 닫기. 재시도 가능. 스레드 비정상 종료는 RuntimeError(정리 확인 아님)."""
        with self._cond:
            self._stop = True
            self._cond.notify_all()

        if self._thread is not None:
            self._thread.join(timeout=limit_s)

            if self._thread.is_alive():
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
                while not self._queue and not self._stop and self._reset_reason is None:
                    self._cond.wait(0.05)

                if self._stop:
                    break

                reset, self._reset_reason = self._reset_reason, None
                item = self._queue.popleft() if self._queue else None
                now = self._clock_ns()
                lag_reset = None

                if item is not None:
                    lag_ms = (now - item.enqueued_ns) / 1e6
                    self._lag_ms_samples.append(lag_ms)
                    window = list(self._handoff_window)
                    pending_ratio = (window.count(False) / len(window)) if len(window) >= self.config.window_n else 0.0
                    lagging = lag_ms > self.config.lag_limit_ms or pending_ratio > self.config.pending_ratio_limit
                    self._lag_streak = self._lag_streak + 1 if lagging else 0
                    cooled = (self._last_lag_reset_ns is None
                              or (now - self._last_lag_reset_ns) / 1e6 >= self.config.lag_cooldown_ms)

                    if self._lag_streak >= self.config.lag_persist_n and cooled:
                        # 밀림 지속: 최신 1개만 남기고 다시 시작(꺼낸 항목도 오래된 것으로 버림)
                        newest = self._queue[-1] if self._queue else item
                        # 버린 수 = 꺼낸 항목(최신이 아니면) + 큐에서 최신을 뺀 나머지 = 큐 길이(큐가 비면 꺼낸 것을 그대로 씀)
                        skipped = len(self._queue)
                        self._queue.clear()
                        self.counts["stale_skipped"] += skipped
                        self.counts["lag_resets"] += 1
                        self._last_lag_reset_ns = now
                        self._lag_streak = 0
                        self._handoff_window.clear()
                        lag_reset = {"t_mono_ns": now, "lag_ms": round(lag_ms, 3), "pending_ratio": round(pending_ratio, 3),
                                     "queue_len": skipped, "skipped": skipped}
                        self.lag_reset_log.append(lag_reset)
                        item = newest

            if reset is not None:
                self._apply_reset(reset)

            if lag_reset is not None:
                self._clear_display_prev()
                self._tracker.reset("lag")
                self._write({"kind": "lag_reset", **lag_reset})

            if item is not None:
                self._process(item, lag_ms if lag_reset is None else (now - item.enqueued_ns) / 1e6)

    def _clear_display_prev(self) -> None:
        """추적 스레드: 재설정 전 결과를 표시에 쓰지 않게 직전 사본을 지운다(#242)."""
        with self._cond:
            self._display_prev = None

    def _apply_reset(self, reason: str) -> None:
        """추적 스레드: 전 트랙 종료 + 기록(D3)."""
        self._clear_display_prev()
        self._tracker.reset(reason)
        self.last_reset = reason
        self._write({"kind": "reset", "reason": reason, "t_mono_ns": self._clock_ns()})

    def _finish(self) -> None:
        """추적 스레드 끝: 처리 못 한 초기화 요청이 있으면 지금 처리하고 종료 사건을 한 번 쓴다(D3)."""
        with self._cond:
            pending, self._reset_reason = self._reset_reason, None

        if pending is not None:
            self._apply_reset(pending)

        self._write({"kind": "closed", "pending_reset_applied": pending is not None, "last_reset": self.last_reset,
                     "t_mono_ns": self._clock_ns()})

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
            track = self._tracker.update(key, int(frame.timestamp_ms), inputs)
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
                                    "plane_z_m": by_index[a.instance_index].plane_z_m,     # #250: W로 올린 높이
                                    "plane_spec": by_index[a.instance_index].plane_spec} for a in track.assignments]})

        # #242: 갱신한 결과면 확정 번호의 표시용 박스 사본을 둔다(화면 전용, 추적 판정·송신에는 쓰지 않음)
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
                   "lag_resets": counts["lag_resets"], "lag_reset_log": list(self.lag_reset_log)}
            # #239·#242: detect 영상 W 번호 조회(found·prev(직전 결과 짝)·pending 배너·사유 배너), pending 사유
            display = {**self.display_counts, "prev_miss": dict(sorted(self.display_prev_miss.items()))}
            counts["queue_left"] = len(self._queue)   # 닫힐 때 처리하지 못하고 남은 수(합: offered = processed + dropped + stale + left)

        return {"config": self.config.to_json(), "counts": counts, "not_updated": dict(sorted(self.not_updated.items())),
                "lag": lag, "tracker": self._tracker.stats(), "diagnostics": dict(self.diagnostics),
                "display_lookup": display,
                "crash": self.crash, "write_error": self.write_error}


def _r(p):
    return None if p is None else [round(p[0], 2), round(p[1], 2)]


def _r1(v):
    return None if v is None else round(v, 2)


__all__ = ["LagConfigError", "LiveWorldConfig", "LiveWorldTracking", "WorldTrackItem", "frame_key", "load_world_tracking_config",
           "parse_live_world"]
