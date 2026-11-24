"""A 추적 흐름 발행: W 추적 확정 트랙을 링크 프레임마다 `TrackStateArray`(`/a/track_states`)로 낸다.

- 자료 변환(`frame_message`·`reset_message`·`marker_message`)은 ROS 없이 dict를 만든다(단위시험 대상).
- `TrackStream`은 유한 큐와 발행 스레드 하나다. 추적 스레드는 `offer`로 넣기만 하고 막히지 않는다.
  주기 상한을 넘는 `frame`은 건너뛰고, 큐가 차면 가장 오래된 `frame`을 버린다. `reset`·`marker`는 버리지 않는다.
- `RosTrackStreamSender`는 생성 시점에 rclpy·`eyetoarm_interfaces`를 import한다(ROS 의존성 분리).

단위·좌표·시각은 `SortPart.action`과 같다(position mm, SIM_TIME ns, frame base_link = world).
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from eyetoarm.vision.track_class_lock import ClassLockConfig, ClassLockConfigError, parse_class_lock
from eyetoarm.vision import stage_timing

TOPIC = "/a/track_states"
STATES = ("VISIBLE", "PREDICTED", "OCCLUDED", "GRIPPER")
VELOCITY_SOURCES = ("TRACK", "BELT_COMMON", "NONE")
KINDS = ("frame", "reset", "marker")
NAN = float("nan")
LATENCY_KEEP = 4096


class TrackStreamConfigError(ValueError):
    """추적 흐름 설정 오류."""


@dataclass(frozen=True)
class TrackStreamConfig:
    """발행 설정. `max_rate_hz`: `frame` 상한(촬영 시각 기준), `qos_depth`: ROS KEEP_LAST 깊이, `queue_capacity`: 발행 큐.

    `class_lock`: 확정 트랙 종류 고정(#368). `None`이면 사진마다 종류를 덮어쓰는 이전 동작.
    """

    topic: str
    max_rate_hz: float
    qos_depth: int
    queue_capacity: int
    class_lock: ClassLockConfig | None = None

    def to_json(self) -> dict:
        """요약용 사전."""
        return {"topic": self.topic, "max_rate_hz": self.max_rate_hz, "qos_depth": self.qos_depth,
                "queue_capacity": self.queue_capacity,
                "class_lock": None if self.class_lock is None else self.class_lock.to_json()}


def parse_track_stream(doc) -> TrackStreamConfig:
    """설정 사전 → `TrackStreamConfig`. 키는 `status`(반드시 `test_only`)·`note`(선택)·`topic`·`max_rate_hz`·`qos_depth`·`queue_capacity`
    ·`class_lock`(선택, `track_class_lock.parse_class_lock`)."""
    keys = {"status", "topic", "max_rate_hz", "qos_depth", "queue_capacity"}

    if not isinstance(doc, dict) or not keys <= set(doc) or set(doc) - keys - {"note", "class_lock"}:
        raise TrackStreamConfigError(f"track stream config must have exactly {sorted(keys)} (+ optional note, class_lock)")

    if doc["status"] != "test_only":
        raise TrackStreamConfigError("track stream config status must be 'test_only' (development values)")

    topic = doc["topic"]

    if not isinstance(topic, str) or not topic.startswith("/") or any(c.isspace() for c in topic):
        raise TrackStreamConfigError(f"topic must be an absolute ROS topic name: {topic!r}")

    rate = doc["max_rate_hz"]

    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise TrackStreamConfigError(f"max_rate_hz must be a positive number: {rate!r}")

    for name in ("qos_depth", "queue_capacity"):
        v = doc[name]

        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise TrackStreamConfigError(f"{name} must be an int >= 1: {v!r}")

    lock = None

    if "class_lock" in doc:
        try:
            lock = parse_class_lock(doc["class_lock"])
        except ClassLockConfigError as exc:
            raise TrackStreamConfigError(str(exc)) from exc

    return TrackStreamConfig(topic=topic, max_rate_hz=float(rate), qos_depth=doc["qos_depth"],
                             queue_capacity=doc["queue_capacity"], class_lock=lock)


def load_track_stream_config(path: str) -> tuple[TrackStreamConfig, str]:
    """설정 파일 → (설정, sha256). JSON이 아니거나 값이 틀리면 `TrackStreamConfigError`."""
    raw = Path(path).read_bytes()

    try:
        doc = json.loads(raw)
    except ValueError as exc:
        raise TrackStreamConfigError(f"track stream config is not JSON: {exc}") from exc

    return parse_track_stream(doc), hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------- 자료 변환(ROS 없음)

def fold_yaw_deg(angle_deg) -> float:
    """축 각도(180° 주기) → [-90, 90). 모르면 NaN."""
    if angle_deg is None or not math.isfinite(angle_deg):
        return NAN

    a = (float(angle_deg) + 90.0) % 180.0 - 90.0

    return a if a < 90.0 else a - 180.0


def track_state(t: dict, timestamp_ms: int) -> str:
    """상태 우선순위 GRIPPER > OCCLUDED > PREDICTED > VISIBLE. 이 프레임에 링크로 갱신된 트랙만 VISIBLE."""
    if t["hit_camera"] == "gripper":
        return "GRIPPER"

    if t["occluded"]:
        return "OCCLUDED"

    if t["out_link"] or t["hold_pred"] or t["last_hit_ms"] != timestamp_ms:
        return "PREDICTED"

    return "VISIBLE"


def _num(v) -> float:
    """유한 실수, 아니면 NaN."""
    return float(v) if v is not None and math.isfinite(v) else NAN


def track_entry(t: dict, timestamp_ms: int, part_class: str, plane_z_m: float) -> dict:
    """`WorldTracker.stream_snapshot` 항목 하나 → `TrackState` 사전(mm·ns·도)."""
    z_mm = plane_z_m * 1000.0
    x, y = float(t["x_mm"]), float(t["y_mm"])
    yaw = fold_yaw_deg(t["angle_deg"])
    length = _num(t["length_mm"])

    if math.isfinite(yaw) and math.isfinite(length):
        dx, dy = math.cos(math.radians(yaw)) * length / 2.0, math.sin(math.radians(yaw)) * length / 2.0
        end_a, end_b = [x - dx, y - dy, z_mm], [x + dx, y + dy, z_mm]
    else:
        end_a, end_b = [NAN, NAN, NAN], [NAN, NAN, NAN]

    return {"part_track_id": str(t["part_track_id"]), "part_class": part_class, "state": track_state(t, timestamp_ms),
            "position_mm": [x, y, z_mm], "velocity_mm_per_s": [0.0, float(t["vy_mm_per_s"]), 0.0],
            "velocity_source": t["velocity_source"], "axis_yaw_deg": yaw, "length_mm": length,
            "endpoint_a_mm": end_a, "endpoint_b_mm": end_b, "last_hit_stamp_ns": int(t["last_hit_ms"]) * 1_000_000,
            "last_hit_camera": t["hit_camera"], "confirmed": True}


def _base(kind: str, run_id: str, source_id: str, *, image_seq: int = 0, generation: int = -1, tracker_epoch: int,
          capture_stamp_ns: int = 0, belt_velocity_mm_per_s: float = NAN, reset_reason: str = "",
          handoff_on: bool, tracks=()) -> dict:
    """공통 머리(발행 시각은 발행 스레드가 채운다)."""
    return {"kind": kind, "run_id": run_id, "source_id": source_id, "image_seq": int(image_seq), "generation": int(generation),
            "tracker_epoch": int(tracker_epoch), "capture_stamp_ns": int(capture_stamp_ns), "timestamp_base": "SIM_TIME",
            "frame_id": "base_link", "publish_stamp_ns": 0, "belt_velocity_mm_per_s": _num(belt_velocity_mm_per_s),
            "reset_reason": reset_reason, "handoff_on": bool(handoff_on), "tracks": list(tracks)}


def frame_message(*, run_id: str, source_id: str, image_seq: int, generation: int, tracker_epoch: int, capture_stamp_ns: int,
                  timestamp_ms: int, snapshot: list[dict], belt_velocity_mm_per_s, handoff_on: bool,
                  classes: dict, default_plane_z_m: float) -> dict:
    """링크 프레임 하나의 `TrackStateArray` 사전. `classes`는 번호 → (종류, 투영 높이 m), 없으면 ("", 기본 높이)."""
    tracks = []

    for t in snapshot:
        part_class, z_m = classes.get(t["part_track_id"], ("", default_plane_z_m))
        tracks.append(track_entry(t, timestamp_ms, part_class, z_m))

    return _base("frame", run_id, source_id, image_seq=image_seq, generation=generation, tracker_epoch=tracker_epoch,
                 capture_stamp_ns=capture_stamp_ns, belt_velocity_mm_per_s=belt_velocity_mm_per_s, handoff_on=handoff_on,
                 tracks=tracks)


def reset_message(*, run_id: str, source_id: str, reason: str, tracker_epoch: int, handoff_on: bool) -> dict:
    """재설정 알림(받는 쪽은 이전 번호를 모두 버린다)."""
    return _base("reset", run_id, source_id, tracker_epoch=tracker_epoch, reset_reason=reason, handoff_on=handoff_on)


def marker_message(*, run_id: str, source_id: str, tracker_epoch: int, handoff_on: bool) -> dict:
    """실행 끝 알림."""
    return _base("marker", run_id, source_id, tracker_epoch=tracker_epoch, handoff_on=handoff_on)


# ---------------------------------------------------------------------- 발행 큐·스레드

def _pct(values, q: float):
    if not values:
        return None

    v = sorted(values)

    return round(v[min(len(v) - 1, int(q * (len(v) - 1)))], 3)


class TrackStream:
    """발행 큐 + 발행 스레드. 큐·계수는 잠금 하나가 보호하고, `send`는 발행 스레드만 부른다.

    `send(msg_dict)`: 실제 발행(ROS면 `RosTrackStreamSender.send`). `now_ns()`: 발행 시각(ROS면 SIM 시계).
    `send`의 예외는 계수·첫 오류로만 남기고 추적 스레드로 번지지 않는다.
    """

    def __init__(self, config: TrackStreamConfig, send: Callable[[dict], None], now_ns: Callable[[], int],
                 clock_ns: Callable[[], int] = time.monotonic_ns) -> None:
        if not isinstance(config, TrackStreamConfig):
            raise TrackStreamConfigError("config must be TrackStreamConfig")

        self.config = config
        self._send, self._now_ns, self._clock_ns = send, now_ns, clock_ns
        self._cond = threading.Condition()
        self._queue: collections.deque[tuple[dict, int]] = collections.deque()
        self._stop = False
        self._thread: threading.Thread | None = None
        self._last_capture_ns: int | None = None
        self._min_gap_ns = int(1e9 / config.max_rate_hz)
        self.counts = {k: 0 for k in ("offered", "published", "dropped_rate", "dropped_queue", "dropped_closed",
                                      "publish_errors")}
        self.first_error: str | None = None
        self._latency_sim_ms: collections.deque[float] = collections.deque(maxlen=LATENCY_KEEP)
        self._queue_ms: collections.deque[float] = collections.deque(maxlen=LATENCY_KEEP)
        self._record = None                 # 발행 기록 파일(발행 스레드만 씀)
        self.record_error: str | None = None

    def set_record(self, path) -> None:
        """시작 전에 발행 기록 파일(새 파일)을 정한다. 줄마다 촬영·발행 SIM 시각과 지연을 남긴다."""
        if self._thread is not None:
            raise TrackStreamConfigError("set_record must be called before start")

        self._record = open(path, "x", encoding="utf-8")

    def _write_record(self, rec: dict) -> None:
        """발행 스레드: 기록 한 줄. 실패는 첫 오류만 남기고 이후 쓰지 않는다(발행은 계속)."""
        if self._record is None or self.record_error is not None:
            return

        try:
            self._record.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
            self._record.flush()
        except (OSError, ValueError) as exc:
            self.record_error = f"{type(exc).__name__}: {exc}"

    def start(self) -> None:
        """발행 스레드 시작."""
        self._thread = threading.Thread(target=self._loop, name="eyetoarm-track-stream", daemon=True)
        self._thread.start()

    def offer(self, msg: dict) -> str:
        """넣기만 한다(기다리지 않음). 결과: `queued` | `dropped_rate` | `dropped_closed`."""
        with self._cond:
            self.counts["offered"] += 1

            if self._stop:
                self.counts["dropped_closed"] += 1
                return "dropped_closed"

            if msg["kind"] == "frame":
                cap = msg["capture_stamp_ns"]

                if self._last_capture_ns is not None and 0 <= cap - self._last_capture_ns < self._min_gap_ns:
                    self.counts["dropped_rate"] += 1
                    return "dropped_rate"

                self._last_capture_ns = cap
            else:
                self._last_capture_ns = None   # 재설정·끝 뒤 첫 프레임은 주기 상한과 무관하게 낸다

            if len(self._queue) >= self.config.queue_capacity:
                self._drop_oldest_frame()

            self._queue.append((msg, self._clock_ns()))
            self._cond.notify_all()

        return "queued"

    def _drop_oldest_frame(self) -> None:
        """잠금 안: 가장 오래된 `frame`을 버린다. `frame`이 없으면(모두 reset·marker) 버리지 않고 넘긴다."""
        for i, (m, _) in enumerate(self._queue):
            if m["kind"] == "frame":
                del self._queue[i]
                self.counts["dropped_queue"] += 1
                return

    def _loop(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._stop:
                    self._cond.wait()

                if not self._queue:
                    return

                msg, queued_ns = self._queue.popleft()

            self._publish(msg, queued_ns)

    def _publish(self, msg: dict, queued_ns: int) -> None:
        """발행 스레드: 발행 시각을 채워 보낸다. 실패는 계수와 첫 오류로 남긴다."""
        out = dict(msg)

        try:
            out["publish_stamp_ns"] = int(self._now_ns())
            self._send(out)
            stage_timing.mark("published", out.get("image_seq"), msg_kind=out.get("kind"),
                              publish_sim_ns=out.get("publish_stamp_ns"), capture_ns=out.get("capture_stamp_ns"))
        except Exception as exc:   # 발행 스레드 최상단: 추적 결과에 영향 없이 실패만 드러낸다
            with self._cond:
                self.counts["publish_errors"] += 1
                self.first_error = self.first_error or f"{type(exc).__name__}: {exc}"

            self._write_record({**_record_keys(out), "published": False, "error": f"{type(exc).__name__}: {exc}"})
            return

        queue_ms = (self._clock_ns() - queued_ns) / 1e6
        latency = None

        if out["kind"] == "frame" and out["capture_stamp_ns"] > 0 and out["publish_stamp_ns"] > 0:
            latency = (out["publish_stamp_ns"] - out["capture_stamp_ns"]) / 1e6

        with self._cond:
            self.counts["published"] += 1
            self._queue_ms.append(queue_ms)

            if latency is not None:
                self._latency_sim_ms.append(latency)

        self._write_record({**_record_keys(out), "published": True, "capture_to_publish_sim_ms": latency,
                            "queue_wait_ms": round(queue_ms, 3)})

    def close(self, limit_s: float) -> bool:
        """남은 항목을 보내고 끝낸다. 제한 안에 끝나면 True."""
        with self._cond:
            self._stop = True
            self._cond.notify_all()

        if self._thread is not None:
            self._thread.join(limit_s)

            if self._thread.is_alive():
                return False

        if self._record is not None and not self._record.closed:
            try:
                self._record.close()
            except OSError as exc:
                self.record_error = self.record_error or f"close: {type(exc).__name__}: {exc}"

        return True

    def summary(self) -> dict:
        """계수·지연(촬영 → 발행 SIM ms, 큐 대기 ms) 요약."""
        with self._cond:
            lat, qms = list(self._latency_sim_ms), list(self._queue_ms)

            return {"config": self.config.to_json(), "counts": dict(self.counts), "first_error": self.first_error,
                    "record_error": self.record_error,
                    "queue_left": len(self._queue),
                    "capture_to_publish_sim_ms": {"n": len(lat), "p50": _pct(lat, 0.5), "p95": _pct(lat, 0.95),
                                                  "max": _pct(lat, 1.0)},
                    "queue_wait_ms": {"n": len(qms), "p50": _pct(qms, 0.5), "p95": _pct(qms, 0.95), "max": _pct(qms, 1.0)}}


def _record_keys(msg: dict) -> dict:
    """발행 기록 줄의 공통 키(SIM ns)."""
    return {"kind": msg["kind"], "image_seq": msg["image_seq"], "generation": msg["generation"],
            "tracker_epoch": msg["tracker_epoch"], "capture_stamp_ns": msg["capture_stamp_ns"],
            "publish_stamp_ns": msg["publish_stamp_ns"], "tracks": len(msg["tracks"]),
            "reset_reason": msg["reset_reason"] or None}


# ---------------------------------------------------------------------- ROS 발행기

class RosTrackStreamSender:
    """`TrackStateArray` 발행기(rclpy·`eyetoarm_interfaces`는 여기서만 import). 노드는 SIM 시계(`use_sim_time`)를 쓴다.

    `/clock`을 받으려고 노드를 별도 스레드에서 돌린다. `send`는 `TrackStream` 발행 스레드만 부른다.
    """

    def __init__(self, config: TrackStreamConfig, node_name: str = "a_track_stream") -> None:
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.parameter import Parameter
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from rclpy.signals import SignalHandlerOptions

        from eyetoarm_interfaces.msg import TrackState, TrackStateArray

        self._rclpy, self._State, self._Array = rclpy, TrackState, TrackStateArray
        self._context = Context()
        rclpy.init(context=self._context, signal_handler_options=SignalHandlerOptions.NO)
        self._node = rclpy.create_node(node_name, context=self._context,
                                       parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=config.qos_depth,
                         durability=DurabilityPolicy.VOLATILE)
        self._pub = self._node.create_publisher(TrackStateArray, config.topic, qos)
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        self._spin = threading.Thread(target=self._executor.spin, name="eyetoarm-track-stream-clock", daemon=True)
        self._spin.start()
        self.topic = config.topic

    def now_ns(self) -> int:
        """노드 SIM 시각(ns). `/clock`을 아직 못 받았으면 0."""
        return self._node.get_clock().now().nanoseconds

    def send(self, msg: dict) -> None:
        """사전 → `TrackStateArray` 발행."""
        out = self._Array()

        for name in ("kind", "run_id", "source_id", "image_seq", "generation", "tracker_epoch", "capture_stamp_ns",
                     "timestamp_base", "frame_id", "publish_stamp_ns", "belt_velocity_mm_per_s", "reset_reason", "handoff_on"):
            setattr(out, name, msg[name])

        for t in msg["tracks"]:
            s = self._State()

            for name, value in t.items():
                setattr(s, name, value)

            out.tracks.append(s)

        self._pub.publish(out)

    def close(self, limit_s: float = 2.0) -> bool:
        """시계 스레드를 멈추고 노드·문맥을 정리한다. 시계 스레드가 제한 안에 끝나면 True."""
        self._executor.shutdown(timeout_sec=limit_s)
        self._spin.join(limit_s)
        self._node.destroy_node()

        if self._context.ok():
            self._rclpy.shutdown(context=self._context)

        return not self._spin.is_alive()


__all__ = ["KINDS", "STATES", "TOPIC", "VELOCITY_SOURCES", "RosTrackStreamSender", "TrackStream", "TrackStreamConfig",
           "TrackStreamConfigError", "fold_yaw_deg", "frame_message", "load_track_stream_config", "marker_message",
           "parse_track_stream", "reset_message", "track_entry", "track_state"]
