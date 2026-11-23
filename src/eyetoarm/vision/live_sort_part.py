"""실시간 기록 연결이 검사 ③에서 수락한 B 결과의 C 목표를 B `SortPartClient`로 `/sort_part`에 보낸다.

프레임당 1회 시도(목표 최대 1개)이고, 게이트 잠금으로 `close_run`이 돌아온 뒤 시작하는 송신은 0이다.
사건은 메모리 큐를 거쳐 기록 스레드가 `sort_part.jsonl`에 쓰며, 판정은 클라이언트 `stats`가 아니라 이 기록으로 한다.
rclpy를 import하지 않는다(B 클라이언트 생성자만 지연 import).
"""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from eyetoarm.transforms.sort_part_client import SEND_KINDS, SortPartClient

_CONFIG_KEYS = {"server_wait_s", "result_wait_s", "close_limit_s", "spin_timeout_s"}
_RETRY_KEYS = {"max_failures_per_track", "failed_point_block_s", "failed_point_block_mm"}
_FAIL_STATUSES = ("FAILED", "EXPIRED", "ERROR", "REJECTED")
_RUN_SOURCE_KEYS = {"service", "poll_s", "call_timeout_s", "stale_s"}
_RUN_BLOCK_EVENTS = ("skip:run_unknown", "skip:run_inactive", "skip:run_changed")
_NO_CALL_RELEASE = ("gate_closed", "state_failed", "skip:track_retry_limit", "skip:failed_point_block",
                    "skip:failure_gap", *_RUN_BLOCK_EVENTS)
# `retry` 블록이 없을 때의 기본 제한: 실패한 목표의 촬영 시각부터 이 SIM 시간 안에 찍힌 같은 번호(번호 없으면 모든)
# `PICK_PLACE`는 보내지 않는다. 실패가 곧바로 돌아올 때 프레임마다 다시 보내는 폭주를 막는다(블록이 있으면 블록 값)
DEFAULT_FAILURE_GAP_S = 1.0
RUN_MISMATCH_REASON = "SORT_RUN_MISMATCH"   # C 결과 사유: 이것을 받으면 실행 번호를 바로 다시 묻는다


def retry_policy(config) -> dict:
    """기록용 재시도 정책: 설정 블록이 있으면 그 값, 없으면 기본 제한 상수."""
    if config.retry is not None:
        return {"source": "config", **config.retry.to_json()}

    return {"source": "default", "failure_gap_s": DEFAULT_FAILURE_GAP_S}


class SortPartConfigError(ValueError):
    """`sort_part` 설정 오류."""


@dataclass(frozen=True)
class RetryConfig:
    """실패 재시도 제한(기본값 없음).

    번호별 실패가 `max_failures_per_track`회면 그 번호를, 실패 위치 근방 `failed_point_block_mm` 안은
    `failed_point_block_s`(SIM 시간) 동안 `PICK_PLACE`를 보내지 않는다. 성공이면 그 번호의 실패 수를 지운다.
    """

    max_failures_per_track: int
    failed_point_block_s: float
    failed_point_block_mm: float

    def to_json(self) -> dict:
        """요약용 사전."""
        return {k: getattr(self, k) for k in sorted(_RETRY_KEYS)}


@dataclass(frozen=True)
class RunSourceConfig:
    """C 실행 번호 조회(`GetSortRun`) 설정(기본값 없음). 있으면 관리 실행으로 보낸다(#301 P2).

    `poll_s`마다 조회하고, `call_timeout_s` 안에 답이 없으면 실패로 센다.
    마지막 답이 `stale_s`보다 오래되면 실행 번호를 모른다고 보고 보내지 않는다.
    """

    service: str
    poll_s: float
    call_timeout_s: float
    stale_s: float

    def to_json(self) -> dict:
        """요약용 사전."""
        return {k: getattr(self, k) for k in sorted(_RUN_SOURCE_KEYS)}


@dataclass(frozen=True)
class SortPartConfig:
    """송신 설정(기본값 없음, 모두 필수)."""

    server_wait_s: float     # 시작 때 서버 대기(보이지 않아도 실행)
    result_wait_s: float     # 닫을 때 결과 대기(기록용)
    close_limit_s: float     # spin 스레드·기록 스레드 join 상한
    spin_timeout_s: float    # spin_once 대기
    retry: RetryConfig | None = None   # 없으면 제한 없음
    require_track_id: bool | None = None   # 참이면 확정 번호 없는 PICK_PLACE를 보내지 않음
    run_source: RunSourceConfig | None = None   # 없으면 비관리(빈 job·epoch 0, 지금 동작)

    def to_json(self) -> dict:
        """요약용 사전. `require_track_id`는 설정에 있을 때만 넣는다(없으면 형식 그대로)."""
        doc = {**{k: getattr(self, k) for k in sorted(_CONFIG_KEYS)},
               "retry": "off" if self.retry is None else self.retry.to_json()}

        if self.require_track_id is not None:
            doc["require_track_id"] = self.require_track_id

        if self.run_source is not None:   # 없으면 형식 그대로
            doc["run_source"] = self.run_source.to_json()

        return doc


def parse_sort_part(doc) -> SortPartConfig:
    """`sort_part` 블록을 검사한다. 모르는 키·누락·양수 아닌 값은 `SortPartConfigError`."""
    if not isinstance(doc, dict):
        raise SortPartConfigError("sort_part must be an object")

    unknown, missing = set(doc) - _CONFIG_KEYS - {"retry", "require_track_id", "run_source"}, _CONFIG_KEYS - set(doc)

    if unknown or missing:
        raise SortPartConfigError(f"sort_part: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")

    for key in _CONFIG_KEYS:
        _positive(doc[key], f"sort_part.{key}")

    retry = None

    if "retry" in doc:
        r = doc["retry"]

        if not isinstance(r, dict) or set(r) != _RETRY_KEYS:
            raise SortPartConfigError(f"sort_part.retry must have exactly {sorted(_RETRY_KEYS)}")

        n = r["max_failures_per_track"]

        if type(n) is not int or n < 1:
            raise SortPartConfigError(f"sort_part.retry.max_failures_per_track must be an int >= 1: {n!r}")

        retry = RetryConfig(max_failures_per_track=n,
                            failed_point_block_s=_positive(r["failed_point_block_s"], "sort_part.retry.failed_point_block_s"),
                            failed_point_block_mm=_positive(r["failed_point_block_mm"], "sort_part.retry.failed_point_block_mm"))

    require = doc.get("require_track_id")

    # bool만 받는다(1·0·문자열 거부). 추적 켬·끔과의 짝은 진입점이 확인한다
    if "require_track_id" in doc and not isinstance(require, bool):
        raise SortPartConfigError(f"sort_part.require_track_id must be true or false: {require!r}")

    run_source = _parse_run_source(doc["run_source"]) if "run_source" in doc else None

    return SortPartConfig(**{k: float(doc[k]) for k in _CONFIG_KEYS}, retry=retry, require_track_id=require,
                          run_source=run_source)


def _parse_run_source(r) -> RunSourceConfig:
    """`sort_part.run_source`: 네 키 모두 필요, 서비스 이름은 `/`로 시작하는 문자열, 나머지는 양수."""
    if not isinstance(r, dict) or set(r) != _RUN_SOURCE_KEYS:
        raise SortPartConfigError(f"sort_part.run_source must have exactly {sorted(_RUN_SOURCE_KEYS)}")

    service = r["service"]

    if not isinstance(service, str) or not service.startswith("/") or len(service) < 2 or any(c.isspace() for c in service):
        raise SortPartConfigError(f"sort_part.run_source.service must be an absolute ROS name: {service!r}")

    return RunSourceConfig(service=service,
                           **{k: _positive(r[k], f"sort_part.run_source.{k}") for k in ("poll_s", "call_timeout_s", "stale_s")})


def _positive(v, name: str) -> float:
    """bool 아닌 유한 양수."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
        raise SortPartConfigError(f"{name} must be a positive number: {v!r}")

    return float(v)


class RecordingSortPartClient(SortPartClient):
    """B 클라이언트의 비공개 콜백을 덮어써 사건을 `event_sink(event, goal_id, detail)`로 넘긴다.

    사건은 spin 스레드(콜백)나 worker(송신 오류)에서 온다.
    """

    def __init__(self, node, event_sink: Callable[[str, str, dict], None], stage=None, **kw) -> None:
        """B 생성자(rclpy·인터페이스 지연 import) 뒤 기록 함수를 둔다."""
        super().__init__(node, stage, **kw)
        self.event_sink = event_sink
        self._arrived: dict[str, int] = {}

    def _on_feedback(self, gid: str, fb) -> None:
        super()._on_feedback(gid, fb)
        self.event_sink(f"feedback:{fb.feedback.stage}", gid, {})

    def _on_accepted(self, gid: str, goal: dict, fut) -> None:
        try:
            accepted = bool(fut.result().accepted)
        except Exception:   # 실패는 super가 `_finish(ERROR)`로 남긴다
            accepted = False

        if accepted:
            self.event_sink("accepted", gid, {})

        super()._on_accepted(gid, goal, fut)

    def _on_result(self, gid: str, goal: dict, fut) -> None:
        try:
            self._arrived[gid] = int(fut.result().result.arrived_stamp_ns)
        except Exception:   # 결과 받기 실패는 super가 `_finish(ERROR)`로 남긴다
            pass

        super()._on_result(gid, goal, fut)

    def _finish(self, gid: str, goal: dict, status: str, reason: str) -> None:
        super()._finish(gid, goal, status, reason)
        self.event_sink(f"result:{status}", gid, {"reason": reason, "arrived_stamp_ns": self._arrived.pop(gid, None)})

    def inflight_ids(self) -> list[str]:
        """결과 대기 중인 goal_id(잠금 안에서 복사)."""
        with self._lock:
            return list(self.inflight)


class LiveSortPartSender:
    """실행 하나의 C 목표 송신기. 게이트는 worker·닫는 쪽이 같이 쓰고, 기록 파일은 기록 스레드가 소유한다.

    `runtime`: `send(out) → bool`, `stats`, `inflight_ids()`, `server_seen`, `close(limit_s)`.
    """

    def __init__(self, run_id: str, config: SortPartConfig, runtime, out_dir: str, *,
                 is_failed: Callable[[], bool] | None = None, clock_ns: Callable[[], int] = time.monotonic_ns,
                 tracking: bool = False, release_target=None,
                 run_target: Callable[[str | None, int | None], None] | None = None) -> None:
        """새 기록 파일 `sort_part.jsonl`을 연다. `tracking`이면 모든 줄에 `part_track_id`를 붙인다.

        클라이언트를 부르지 않은 번호 있는 `PICK_PLACE`는 `release_target.release(번호)`로 B 기록을 푼다.
        관리 실행(`config.run_source`)이면 실행 번호가 바뀔 때 `run_target(job, epoch)`로 B 단계에 넣는다
        (비활성·모름이면 둘 다 None)."""
        self.run_id = run_id
        self.tracking = bool(tracking)
        self.release_target = release_target
        self._track_failures: dict[int, int] = {}      # 번호 → 실패 수(기록 조건변수 안)
        self._failed_points: list[tuple] = []          # (x_mm, y_mm, z_mm, 촬영 stamp ns)
        self._goal_meta: dict[str, tuple] = {}         # goal_id → (번호, 관측 위치 mm, 촬영 stamp ns)
        self.retry_counts = {"track_limit": 0, "point_block": 0, "failures_recorded": 0, "points_recorded": 0}
        self._default_failures: dict = {}              # retry 블록 없음: 번호(없으면 None) → 실패한 목표의 촬영 stamp ns
        self.default_gap_counts = {"failures_recorded": 0, "blocked": 0}
        self.no_track_id_states: dict[str, int] = {}   # skip:no_track_id의 track_state 분포(기록 조건변수 안)
        self.config = config
        self.runtime = runtime
        self._is_failed = is_failed or (lambda: False)
        self._clock_ns = clock_ns
        self._gate = threading.Lock()
        self._closed = False
        self.close_reason: str | None = None
        self.gate_wait_max_ms = 0.0
        self._fh = open(Path(out_dir) / "sort_part.jsonl", "x", encoding="utf-8")
        self._qcond = threading.Condition()
        self._queue: list[dict] = []
        self._stop = False
        self._thread: threading.Thread | None = None
        self.write_error: str | None = None
        self.crash: str | None = None
        self.counts: dict[str, int] = {}
        self.goal_keys: dict[str, dict] = {}     # goal_id → 프레임 키(콜백 기록에 붙임)
        self.unresolved: list[str] = []
        self._runtime_closed = False

        # 관리 실행 상태. `_run_lock`이 지킨다. 쓰는 쪽은 런타임 조회 콜백(spin 스레드, 시작 때 진입점)
        self.run_target = run_target
        self._run_lock = threading.Lock()
        self._run: tuple | None = None           # (active, sort_job_id, run_epoch) 마지막 답
        self._run_t_ns: int | None = None        # 그 답을 받은 시각(clock_ns)
        self.applied_run: tuple = (None, None)   # B 단계에 넣은 (job, epoch). 한 번에 통째로 바꿈(잠금 없이 읽음)
        self.run_counts = {"snapshots": 0, "query_failed": 0, "state_changes": 0, "run_changes": 0,
                           "retry_cleared": 0, "refresh_requests": 0}
        self.run_query_failures: dict[str, int] = {}

    @property
    def managed(self) -> bool:
        """관리 실행(C 실행 번호를 받아 실음)인가."""
        return self.config.run_source is not None

    # ------------------------------------------------------------------ 실행 번호(런타임 콜백)

    def on_run_snapshot(self, active: bool, sort_job_id: str, run_epoch: int) -> None:
        """`GetSortRun` 답 하나. 상태가 바뀌면 기록하고, (job, epoch)가 바뀌면 재시도 기억을 비우고 B 단계에 넣는다."""
        new = (bool(active), str(sort_job_id), int(run_epoch))

        with self._run_lock:
            old = self._run
            self._run, self._run_t_ns = new, self._clock_ns()
            self.run_counts["snapshots"] += 1

            if new == old:
                return

            self.run_counts["state_changes"] += 1
            changed = old is not None and old[1:] != new[1:]

            if changed:
                self.run_counts["run_changes"] += 1
                self._clear_retry_memory()

            applied = new[1:] if new[0] else (None, None)
            apply = applied != self.applied_run
            self.applied_run = applied

            # B 단계 쓰기도 이 잠금 안: 답 순서대로 넣는다(연결부 잠금은 이 잠금을 잡지 않음)
            if apply and self.run_target is not None:
                self.run_target(*applied)

        detail = {"old": None if old is None else _run_json(old), "new": _run_json(new)}
        self._record("run_state", None, None, **detail)

        if changed:
            self._record("run_changed", None, None, **detail)

    def on_run_query_failed(self, reason: str) -> None:
        """조회 실패(서비스 없음·시간 초과·오류). 마지막 답은 그대로 두고(`stale_s`가 지나면 모름) 수만 센다."""
        with self._run_lock:
            self.run_counts["query_failed"] += 1
            self.run_query_failures[reason] = self.run_query_failures.get(reason, 0) + 1

    def _clear_retry_memory(self) -> None:
        """실행이 바뀌면 번호별 실패 수·실패 위치를 비운다(C 중복 기억도 새 실행에서 비워짐)."""
        with self._qcond:
            if self._track_failures or self._failed_points:
                self.run_counts["retry_cleared"] += 1

            self._track_failures.clear()
            self._failed_points.clear()
            self._default_failures.clear()

    def _run_block(self, goal: dict) -> str | None:
        """관리 실행 송신 조건(#301 P2). 막으면 사건 이름, 보낼 수 있으면 None."""
        rs = self.config.run_source

        if rs is None:
            return None

        with self._run_lock:
            run, t_ns = self._run, self._run_t_ns

        if run is None or t_ns is None or self._clock_ns() - t_ns > rs.stale_s * 1e9:
            return "skip:run_unknown"

        if not run[0]:
            return "skip:run_inactive"

        # 목표를 만든 때의 B 단계 값이 지금 실행과 다르면(옛 실행·비활성 때 만든 목표) 보내지 않는다
        if goal.get("sort_job_id") != run[1] or goal.get("run_epoch") != run[2]:
            return "skip:run_changed"

        return None

    # ------------------------------------------------------------------ 기록

    def _record(self, event: str, goal_id: str | None, keys: dict | None = None, **detail) -> None:
        """메모리 큐에 넣는다(기다리지 않음). 계수는 같은 잠금."""
        rec = {"t_mono_ns": self._clock_ns(), "event": event, "goal_id": goal_id, **(keys or {}), **detail}

        if self.tracking:   # 목표 없는 줄·모르는 목표는 null
            rec.setdefault("part_track_id", None)

        with self._qcond:
            self.counts[event] = self.counts.get(event, 0) + 1
            self._queue.append(rec)
            self._qcond.notify_all()

    def on_client_event(self, event: str, goal_id: str, detail: dict) -> None:
        """클라이언트 콜백 사건(spin 스레드). 송신 때 남긴 프레임 키를 붙이고, 결과면 재시도 제한 상태를 갱신한다."""
        with self._qcond:
            keys = dict(self.goal_keys.get(goal_id, {}))

            track, pos, stamp, run_ids = self._goal_meta.get(goal_id, (None, None, None, None))

            # 관리 실행: 다른 실행에서 보낸 목표의 결과는 지금 실행의 재시도 기억에 넣지 않는다
            same_run = not self.managed or run_ids == self.applied_run

            if event.startswith("result:") and self.config.retry is not None and same_run:
                status = event.split(":", 1)[1]

                if status in _FAIL_STATUSES:
                    if track is not None:
                        self._track_failures[track] = self._track_failures.get(track, 0) + 1
                        self.retry_counts["failures_recorded"] += 1

                    if pos is not None and stamp is not None:
                        self._failed_points.append((*pos, stamp))
                        self.retry_counts["points_recorded"] += 1
                elif status == "SUCCEEDED" and track is not None:
                    self._track_failures.pop(track, None)

            # 기본 제한(retry 블록 없음): 실패한 목표의 촬영 시각을 번호별로 남긴다. 성공하면 지운다
            if event.startswith("result:") and self.config.retry is None and same_run:
                status = event.split(":", 1)[1]

                if status in _FAIL_STATUSES and stamp is not None:
                    self._default_failures[track] = max(stamp, self._default_failures.get(track, stamp))
                    self.default_gap_counts["failures_recorded"] += 1
                elif status == "SUCCEEDED":
                    self._default_failures.pop(track, None)

        self._record(event, goal_id, keys, **detail)

        # C가 실행 번호가 다르다고 거부했으면 다음 주기를 기다리지 않고 다시 묻는다
        if self.managed and event.startswith("result:") and detail.get("reason") == RUN_MISMATCH_REASON:
            refresh = getattr(self.runtime, "request_run_refresh", None)

            if callable(refresh):
                with self._run_lock:
                    self.run_counts["refresh_requests"] += 1

                refresh()

    def start(self) -> None:
        """기록 스레드를 시작한다."""
        self._thread = threading.Thread(target=self._loop, name="eyetoarm-live-sort-part-records", daemon=True)
        self._thread.start()

    @property
    def thread(self) -> threading.Thread | None:
        """기록 스레드(정리 확인용)."""
        return self._thread

    def _loop(self) -> None:
        try:
            while True:
                with self._qcond:
                    while not self._queue and not self._stop:
                        self._qcond.wait(0.05)

                    batch, self._queue = self._queue, []
                    stop = self._stop

                if batch and self.write_error is None:
                    try:
                        for rec in batch:
                            self._fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")

                        self._fh.flush()
                    except (OSError, ValueError) as exc:
                        self.write_error = f"{type(exc).__name__}: {exc}"

                if stop:
                    with self._qcond:
                        if self._queue:
                            continue

                    return
        except Exception as exc:   # 기록 스레드 최상단: 비정상 종료를 남긴다(정리 확인 거짓)
            self.crash = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                self._fh.close()
            except OSError as exc:
                self.write_error = self.write_error or f"close: {type(exc).__name__}: {exc}"

    # ------------------------------------------------------------------ 송신(worker)

    def offer(self, out, frame, ids: dict, accept_event_seq: int | None) -> str:
        """수락된 B 결과 하나를 보낸다. 사건 이름(sent | skip:… | gate_closed | state_failed | …)을 준다.

        `valid`이고 `goal.kind`가 송신 대상이며 게이트 열림·상태 실패 아님·재시도 제한 통과일 때만 보낸다."""
        goal = getattr(out, "goal", None) or {}
        res = getattr(out, "result", None)
        gid = goal.get("goal_id")
        keys = {"run_id": self.run_id, "image_seq": getattr(frame, "image_seq", None),
                "ros_stamp_ns": getattr(frame, "ros_stamp_ns", None), "accept_event_seq": accept_event_seq,
                "kind": goal.get("kind"), "spec": goal.get("spec"),
                # track_state는 no_track_id 줄에만 남긴다
                **{k: v for k, v in ids.items() if k not in ("track_state", "a_part_track_id")}}
        if self.managed:   # 목표를 만든 때 실린 실행 번호(비관리면 줄 형식 그대로)
            keys.update(sort_job_id=goal.get("sort_job_id"), run_epoch=goal.get("run_epoch"))

        goal_track = goal.get("part_track_id")
        stage_l = "a_part_track_id" in ids   # 단계 L 추적이면 연결부가 고른 후보의 A 번호를 넘긴다
        a_track = ids.get("a_part_track_id")

        # 실효 번호: 목표 번호 우선, 없으면 고른 후보의 A 번호. B 기록 해제는 목표 번호로만 한다
        track = goal_track if goal_track is not None else a_track

        if self.tracking:
            keys["part_track_id"] = track

        if stage_l:
            keys.update(goal_part_track_id=goal_track, a_part_track_id=a_track,
                        track_source="goal" if goal_track is not None else "a" if a_track is not None else None)

            if goal_track is not None and a_track is not None and goal_track != a_track:
                self._record("track_id_mismatch", gid, keys)   # 송신은 목표 번호를 따른다

        if res is None or getattr(res, "valid", False) is not True:
            event = f"skip:{getattr(res, 'reason', None) or 'invalid'}"
            self._record(event, gid, keys)
            return event

        if goal.get("kind") not in SEND_KINDS:
            event = f"skip:{goal.get('reason') or goal.get('kind')}"   # B ALREADY_SENT 포함: 풀지 않는다
            self._record(event, gid, keys)
            return event

        # 같은 볼트를 번호 없이 한 번, 확정 뒤 한 번 더 보내지 않게 한다. PASS_THROUGH(너트)는 그대로 보낸다
        if self.config.require_track_id and goal.get("kind") == "PICK_PLACE" and track is None:
            state = ids.get("track_state")

            with self._qcond:
                key = str(state)
                self.no_track_id_states[key] = self.no_track_id_states.get(key, 0) + 1

            event = "skip:no_track_id"
            self._record(event, gid, keys, track_state=state)
            return event

        pos = _position(goal)
        stamp = goal.get("capture_stamp_ns")
        stamp = stamp if isinstance(stamp, int) and not isinstance(stamp, bool) else getattr(frame, "ros_stamp_ns", None)

        with self._gate:
            if self._closed:
                event = "gate_closed"
            elif self._is_failed():
                event = "state_failed"
            elif (blocked := self._run_block(goal)) is not None:
                event = blocked
            elif (limited := self._retry_block(goal.get("kind"), track, pos, stamp)) is not None:
                event = limited
            else:
                run_ids = (goal.get("sort_job_id"), goal.get("run_epoch")) if self.managed else None

                with self._qcond:
                    if gid:
                        self.goal_keys[gid] = keys
                        self._goal_meta[gid] = (track, pos, stamp, run_ids)

                before = _stats_copy(self.runtime.stats)
                ok = bool(self.runtime.send(out))
                after = _stats_copy(self.runtime.stats)
                event = "sent" if ok else _refusal(before, after)

        self._record(event, gid, keys)

        # B 클라이언트를 부르지 않았으면 B 클라이언트가 풀지 않으므로 여기서 `_dedupe` 기록을 푼다
        if event in _NO_CALL_RELEASE and goal.get("kind") == "PICK_PLACE":
            self.release(goal_track, event, gid, keys)   # A 번호로 B 기록을 풀지 않는다

        return event

    def release(self, track, reason: str, gid: str | None = None, keys: dict | None = None) -> bool:
        """B 중복 방지 기록 해제(번호·대상이 있을 때만). `release:<사유>`를 기록한다."""
        if track is None or self.release_target is None:
            return False

        self.release_target.release(track)
        self._record(f"release:{reason}", gid, keys)

        return True

    def _retry_block(self, kind, track, pos, stamp) -> str | None:
        """재시도 제한(설정이 있을 때, PICK_PLACE만). 막으면 사건 이름."""
        r = self.config.retry

        if kind != "PICK_PLACE":
            return None

        if r is None:
            return self._default_gap_block(track, stamp)

        with self._qcond:
            if track is not None and self._track_failures.get(track, 0) >= r.max_failures_per_track:
                self.retry_counts["track_limit"] += 1
                return "skip:track_retry_limit"

            if pos is not None and stamp is not None:
                window_ns = r.failed_point_block_s * 1e9

                for fx, fy, fz, fstamp in self._failed_points:
                    if 0 <= stamp - fstamp <= window_ns and math.dist(pos, (fx, fy, fz)) <= r.failed_point_block_mm:
                        self.retry_counts["point_block"] += 1
                        return "skip:failed_point_block"

        return None

    def _default_gap_block(self, track, stamp) -> str | None:
        """기본 제한: 같은 번호의 마지막 실패 목표 촬영 뒤 `DEFAULT_FAILURE_GAP_S` 안에 찍힌 목표는 막는다."""
        if stamp is None:
            return None

        with self._qcond:
            failed = self._default_failures.get(track)

            if failed is not None and 0 <= stamp - failed < DEFAULT_FAILURE_GAP_S * 1e9:
                self.default_gap_counts["blocked"] += 1
                return "skip:failure_gap"

        return None

    # ------------------------------------------------------------------ 닫기

    def close_run(self, reason: str) -> None:
        """게이트를 닫는다. 진행 중 송신 한 번이 끝날 때까지 기다린 뒤 돌아온다(그 뒤 송신 0)."""
        t0 = time.monotonic_ns()

        with self._gate:
            waited_ms = (time.monotonic_ns() - t0) / 1e6
            self.gate_wait_max_ms = max(self.gate_wait_max_ms, waited_ms)

            if self._closed:
                return

            self._closed = True
            self.close_reason = reason

        self._record("gate_closed_by", None, None, reason=reason, waited_ms=round(waited_ms, 3))

    def close(self, limit_s: float) -> bool:
        """게이트 닫기 → 결과 대기 → 남은 목표 기록 → 런타임 정리 → 기록 스레드 멈춤(재시도 가능).

        취소 API가 없어 남은 목표는 `unresolved_at_close`로만 기록한다. 런타임 정리 실패는 다시 던진다.
        """
        self.close_run(self.close_reason or "close")
        runtime_error = None

        if not self._runtime_closed:
            deadline = time.monotonic() + self.config.result_wait_s

            while self.runtime.inflight_ids() and time.monotonic() < deadline:
                time.sleep(0.02)

            for gid in self.runtime.inflight_ids():
                if gid not in self.unresolved:
                    self.unresolved.append(gid)
                    self.on_client_event("unresolved_at_close", gid, {})

            try:
                self.runtime.close(limit_s)
                self._runtime_closed = True
            except Exception as exc:   # 기록 스레드를 멈춘 뒤 다시 던진다(정리 확인 거짓, 재시도 대상)
                runtime_error = exc

        with self._qcond:
            self._stop = True
            self._qcond.notify_all()

        if self._thread is not None:
            self._thread.join(timeout=limit_s)

            if self._thread.is_alive():
                return False
        elif not self._fh.closed:   # 시작 전 실패: 남은 것을 이 자리에서 쓰고 닫는다
            self._stop = True
            self._loop()

        if runtime_error is not None:
            raise runtime_error

        if self.crash is not None:
            raise RuntimeError(f"sort_part record writer crashed: {self.crash}")

        return True

    # ------------------------------------------------------------------ 요약

    def _run_summary(self) -> dict:
        """요약 `sort_part.run_source`: 설정·런타임 시작 상태·계수·마지막 답."""
        status = getattr(self.runtime, "run_source_status", None)

        with self._run_lock:
            return {"status": status() if callable(status) else None,
                    "counts": dict(self.run_counts), "query_failures": dict(sorted(self.run_query_failures.items())),
                    "last": None if self._run is None else _run_json(self._run),
                    "applied": {"sort_job_id": self.applied_run[0], "run_epoch": self.applied_run[1]}}

    def problem(self) -> str | None:
        """진입점이 종료 1로 드러낼 문제(기록 실패). 서버 없음·busy는 문제가 아니다."""
        if self.crash is not None:
            return f"sort_part record writer crashed: {self.crash}"

        if self.write_error is not None:
            return f"sort_part record write error: {self.write_error}"

        return None

    def summary(self) -> dict:
        """요약 블록 `sort_part`."""
        with self._qcond:
            counts = dict(sorted(self.counts.items()))

        with self._qcond:
            retry = ("off" if self.config.retry is None else
                     {**self.retry_counts, "track_failures": {str(k): v for k, v in sorted(self._track_failures.items())},
                      "failed_points": len(self._failed_points)})
            no_track_states = dict(sorted(self.no_track_id_states.items()))
            default_gap = None if self.config.retry is not None else {**retry_policy(self.config), **self.default_gap_counts}

        return {"config": self.config.to_json(), "server_seen": getattr(self.runtime, "server_seen", None),
                "tracking": self.tracking, "retry": retry,
                **({} if default_gap is None else {"retry_default": default_gap}),
                **({} if self.config.require_track_id is None else
                   {"require_track_id": self.config.require_track_id,
                    "no_track_id_states": no_track_states}),
                **({} if not self.managed else {"run_source": self._run_summary()}),
                "events": counts, "unresolved_at_close": list(self.unresolved), "close_reason": self.close_reason,
                "gate_wait_max_ms": round(self.gate_wait_max_ms, 3), "write_error": self.write_error, "crash": self.crash,
                "client_stats_reference_only": _stats_copy(getattr(self.runtime, "stats", {}))}


def _run_json(run: tuple) -> dict:
    """(active, job, epoch) → 기록용 사전."""
    return {"active": run[0], "sort_job_id": run[1], "run_epoch": run[2]}


def _stats_copy(stats: Any) -> dict:
    """클라이언트 Counter 사본(다른 스레드가 쓰는 중이면 한 번 다시)."""
    for _ in range(3):
        try:
            return {str(k): int(v) for k, v in dict(stats).items()}
        except RuntimeError:   # 복사 중 크기 변경(spin 스레드): 다시 시도
            continue

    return {}


def _position(goal: dict) -> tuple | None:
    """목표의 관측 위치(base_link mm 세 값, 유한) 또는 None."""
    p = goal.get("observed_position_mm")

    try:
        v = tuple(float(x) for x in p)
    except (TypeError, ValueError):
        return None

    return v if len(v) == 3 and all(math.isfinite(x) for x in v) else None


def _refusal(before: dict, after: dict) -> str:
    """`send`가 False일 때 클라이언트 사유(증가한 계수)."""
    for key, event in (("no_server", "no_server"), ("busy", "busy"), ("skip", "client_skip"), ("error", "send_error")):
        if after.get(key, 0) > before.get(key, 0):
            return event

    return "send_refused"


__all__ = ["DEFAULT_FAILURE_GAP_S", "LiveSortPartSender", "RecordingSortPartClient", "RetryConfig", "RunSourceConfig",
           "SortPartConfig", "SortPartConfigError", "parse_sort_part", "retry_policy"]
