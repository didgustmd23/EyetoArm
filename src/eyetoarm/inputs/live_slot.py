"""실시간 입력 수신 상태기(ROS 없음): 최신 칸 1개, 수신 epoch, 입력 경계 대기열.

수신 스레드는 `on_*`만, 제어 흐름은 `poll_inputs`·`input_applied`·`take`·`is_current_epoch`·`close`를 부른다.
입력 경계마다 잠금 안에서 `rx_epoch`를 올리고, 반영 epoch보다 오래된 영상은 버린다(규칙 R1~R9).
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from eyetoarm.common.types import InputStatus


class BoundaryKind(str, Enum):
    """수신 epoch를 올리는 입력 경계. 값은 `GenerationCause`의 같은 이름 문자열과 같다."""

    FRAME_TIMEOUT = "frame_timeout"
    SIM_TIME_REWIND = "sim_time_rewind"
    RECONNECT = "reconnect"
    PROFILE_CHANGED = "profile_changed"


class NoticeKind(str, Enum):
    """epoch를 올리지 않는 입력 알림."""

    NO_FIRST_FRAME = "no_first_frame"
    START_FAILED = "start_failed"
    RECEIVER_FAILED = "receiver_failed"
    QOS_INCOMPATIBLE = "qos_incompatible"
    REWIND_FILTER_RELEASED = "rewind_filter_released"
    BOUNDARY_HOOK_FAILED = "boundary_hook_failed"  # 경계 훅 예외(치명, 연결부가 fail-stop)


# 경계·알림 → 코어에 알릴 입력 상태(None이면 세대 전이 또는 기록만)
_STATUS_OF = {
    BoundaryKind.FRAME_TIMEOUT: InputStatus.FRAME_TIMEOUT,
    NoticeKind.NO_FIRST_FRAME: InputStatus.SOURCE_UNAVAILABLE,
    NoticeKind.START_FAILED: InputStatus.SOURCE_UNAVAILABLE,
    NoticeKind.RECEIVER_FAILED: InputStatus.SOURCE_UNAVAILABLE,
    NoticeKind.QOS_INCOMPATIBLE: InputStatus.SOURCE_UNAVAILABLE,
    NoticeKind.BOUNDARY_HOOK_FAILED: InputStatus.SOURCE_UNAVAILABLE,
}


@dataclass(frozen=True)
class CameraProfileInfo:
    """수신한 `CameraInfo`의 식별 정보. `calibration_id`는 B `from_camera_info`가 만든 값이다.

    `camera`(선택)는 같은 CameraInfo의 B `CameraProfile` 원본이며, 같음 비교·repr에는 넣지 않는다.
    """

    calibration_id: str
    width: int
    height: int
    frame_id: str
    camera: object | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class LiveInput:
    """입력 대기열 항목. 경계면 `epoch`가 새 epoch, 알림이면 None. `status`는 코어에 알릴 입력 상태."""

    kind: BoundaryKind | NoticeKind
    epoch: int | None
    detail: str
    event_monotonic_ns: int
    profile: CameraProfileInfo | None

    @property
    def status(self) -> InputStatus | None:
        """코어 `note_input_status`로 알릴 상태. 세대 전이·기록 전용 항목은 None."""
        return _STATUS_OF.get(self.kind)


@dataclass(frozen=True)
class RxRecord:
    """영상 수신 기록. `message`는 변환 전 원본 메시지 참조이며 이 모듈은 내용을 읽지 않는다."""

    message: object
    rx_epoch: int
    gen_at_rx: int
    received_monotonic_ns: int
    stamp_ns: int
    frame_id: str
    width: int
    height: int
    profile: CameraProfileInfo | None
    source_index: int


class TakeStatus(str, Enum):
    """`take` 결과 종류."""

    FRAME = "frame"
    EMPTY = "empty"
    NEEDS_INPUT = "needs_input"
    CLOSED = "closed"


@dataclass(frozen=True)
class Take:
    """`take` 결과. FRAME이면 기록과 수신 당시 세대(R2)가 있다."""

    status: TakeStatus
    record: RxRecord | None = None
    capture_generation: int | None = None


class LiveInputOverflow(RuntimeError):
    """입력 대기열 상한을 넘었다(경계·알림을 잃지 않으려고 실행을 실패로 끝낸다)."""


def _positive_finite(value) -> bool:
    """bool이 아닌 유한한 양수인지."""
    return (not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
            and value > 0)


@dataclass(frozen=True)
class LiveSlotConfig:
    """상태기 설정. 기본값은 모두 A 임시 개발값이다.

    `frame_timeout_s` 중단 판정, `startup_timeout_s` 첫 영상 제한, `future_tolerance_ns` 되감기 필터 허용 폭,
    `stale_after_rewind_limit` 필터 해제 수, `max_pending_inputs` 대기열 상한, `graph_absorb_s` R9 흡수 창,
    `expected_frame_id` 영상 헤더 기대값(다르면 `calibration_id`를 싣지 않음, None이면 검사 안 함).
    """

    frame_timeout_s: float = 1.0
    startup_timeout_s: float = 10.0
    future_tolerance_ns: int = 500_000_000
    stale_after_rewind_limit: int = 30
    max_pending_inputs: int = 64
    expected_frame_id: str | None = None
    graph_absorb_s: float = 5.0

    def __post_init__(self) -> None:
        """값 범위를 검사한다. 어긋나면 ValueError."""
        for name in ("frame_timeout_s", "startup_timeout_s", "graph_absorb_s"):
            if not _positive_finite(getattr(self, name)):
                raise ValueError(f"{name} must be a finite number > 0: {getattr(self, name)!r}")

        for name in ("future_tolerance_ns", "stale_after_rewind_limit", "max_pending_inputs"):
            value = getattr(self, name)
            low = 0 if name == "future_tolerance_ns" else 1

            if type(value) is not int or value < low:
                raise ValueError(f"{name} must be an int >= {low}: {value!r}")

        if self.expected_frame_id is not None and (not isinstance(self.expected_frame_id, str)
                                                   or not self.expected_frame_id):
            raise ValueError(f"expected_frame_id must be None or a non-empty string: {self.expected_frame_id!r}")


class GenerationMirror:
    """코어 세대의 잠금 보호 사본. 제어 흐름이 쓰고 수신 콜백은 읽기만 한다."""

    def __init__(self, value: int) -> None:
        """초기 세대를 받는다."""
        self._lock = threading.Lock()
        self._value = value

    def get(self) -> int:
        """현재 사본 값."""
        with self._lock:
            return self._value

    def set(self, value: int) -> None:
        """사본을 바꾼다(제어 흐름 전용)."""
        with self._lock:
            self._value = value


_STATS_KEYS = ("received", "accepted", "overwritten", "duplicate_stamp", "stale_epoch", "stale_after_rewind",
               "rewind_filter_released", "frame_id_mismatch", "boundaries", "notices", "taken", "resume_boundaries",
               "graph_change_absorbed", "graph_change_during_gap", "rewind_merged_into_resume",
               "boundary_hook_errors", "camera_info_invalid", "rewind_after_gap", "invalid_header", "invalid_clock")


class LiveFrameSlot:
    """수신 스레드와 제어 흐름 사이의 최신 칸 1개·입력 대기열.

    최신 칸은 새 영상으로 덮어쓰고(`overwritten` 셈), 경계·알림 대기열은 순서를 보존하고 버리지 않는다.
    """

    def __init__(self, config: LiveSlotConfig, *, generation: GenerationMirror,
                 clock_ns: Callable[[], int] = time.monotonic_ns,
                 boundary_hook: Callable[[int, BoundaryKind], None] | None = None) -> None:
        """설정·세대 사본·PC 단조 시계를 받고, 시작 시각을 한 번 읽는다.

        `boundary_hook(epoch, kind)`는 경계마다 이 잠금 안에서 부른다(잠금 순서 slot → 훅). 훅은 이 잠금을
        다시 잡으면 안 되며, 예외는 세고 치명 알림 `BOUNDARY_HOOK_FAILED`를 넣는다.
        """
        self._config = config
        self._generation = generation
        self._clock_ns = clock_ns
        self._boundary_hook = boundary_hook
        self._cond = threading.Condition()

        # epoch(R1~R3)
        self._rx_epoch = 0
        self._applied_epoch = 0
        self._gen_of_epoch: dict[int, int] = {0: generation.get()}

        # 최신 칸·대기열
        self._latest: RxRecord | None = None
        self._inputs: deque[LiveInput] = deque()
        self._overflow = False

        # 시각 검사(R5~R7)
        self._last_stamp_ns: int | None = None
        self._last_clock_ns: int | None = None
        self._filter_until_ns: int | None = None
        self._filter_drops = 0

        # 프로필·발행자·수신 상태
        self._profile: CameraProfileInfo | None = None
        self._publisher_count: int | None = None
        self._confirmed_gids: frozenset | None = None
        self._absorb_until_ns: int | None = None
        self._clock_seen = False
        self._state = "waiting_first"
        self._start_ns = clock_ns()
        self._last_rx_ns: int | None = None
        self._startup_reported = False
        self._qos_reported: set[str] = set()
        self._receiver_failed = False
        self._next_index = 0
        self._closed = False
        self._stats = {k: 0 for k in _STATS_KEYS}

    # ------------------------------------------------------------------ 수신 스레드 쪽

    def on_image(self, message: object, *, stamp_ns: int, frame_id: str, width: int, height: int) -> str:
        """영상 하나를 받는다. 결과: accepted | duplicate | stale_after_rewind | closed."""
        now = self._clock_ns()

        with self._cond:
            if self._closed:
                return "closed"

            index = self._next_index
            self._next_index += 1
            self._stats["received"] += 1

            # R5: 되감기 필터(현재 /clock 기준 허용, 휴리스틱)
            if self._filter_until_ns is not None:
                clock = self._last_clock_ns

                if clock is None or stamp_ns > clock + self._config.future_tolerance_ns:
                    self._stats["stale_after_rewind"] += 1
                    self._filter_drops += 1

                    if self._filter_drops > self._config.stale_after_rewind_limit:
                        self._release_filter(now, f"consecutive drops {self._filter_drops} > "
                                                  f"{self._config.stale_after_rewind_limit}")

                    return "stale_after_rewind"

                self._filter_drops = 0

            # R7: 같은 stamp
            if self._last_stamp_ns is not None and stamp_ns == self._last_stamp_ns:
                self._stats["duplicate_stamp"] += 1
                return "duplicate"

            # R8: 중단 뒤 첫 새 영상은 발행자 확인 없이 RECONNECT 경계
            if self._in_gap(now):
                if self._state == "streaming":
                    self._state = "timed_out"
                    self._boundary(BoundaryKind.FRAME_TIMEOUT,
                                   f"no frame for {(now - self._last_rx_ns) / 1e9:.3f} s (detected at image)", now)

                # R8a: 재개 첫 영상 stamp가 중단 전보다 작으면 중단 중 리셋이다. /clock 되감기(R5)와 같게
                # SIM_TIME_REWIND 경계 + 되감기 필터(기준 = 중단 전 마지막 stamp)를 만든다. 늦은 낮은 /clock은
                # 기준을 비웠으므로 경계를 또 만들지 않는다. 필터는 다음 /clock 전까지 영상을 넘기지 않는다.
                if self._last_stamp_ns is not None and stamp_ns < self._last_stamp_ns:
                    pre_max = self._last_stamp_ns
                    self._boundary(BoundaryKind.SIM_TIME_REWIND,
                                   f"image stamp {stamp_ns} < {pre_max} after frame_timeout (reset during gap)", now)
                    self._stats["rewind_after_gap"] += 1
                    self._filter_until_ns = pre_max
                    self._filter_drops = 0
                    self._last_clock_ns = None

                self._boundary(BoundaryKind.RECONNECT, "first frame after frame_timeout (origin not checked)", now)
                self._stats["resume_boundaries"] += 1
                self._absorb_until_ns = now + int(self._config.graph_absorb_s * 1e9)

            # R6: 영상 stamp 되감기. 같은 되감기의 /clock 경계를 막으려고 clock 기준을 비운다
            elif self._last_stamp_ns is not None and stamp_ns < self._last_stamp_ns:
                self._boundary(BoundaryKind.SIM_TIME_REWIND, f"image stamp {stamp_ns} < {self._last_stamp_ns}", now)
                self._last_clock_ns = None

            self._last_stamp_ns = stamp_ns

            # R4: 수신 시점 프로필. 헤더가 기대값과 다르거나 해상도가 다르면 싣지 않는다
            expected = self._config.expected_frame_id
            mismatch = expected is not None and frame_id != expected

            if mismatch:
                self._stats["frame_id_mismatch"] += 1

            profile = self._profile

            if mismatch or profile is None or (profile.width, profile.height, profile.frame_id) != (width, height,
                                                                                                    frame_id):
                profile = None

            # 수신 상태(재개 경계는 위 R8에서 이미 만들었다)
            self._state = "streaming"
            self._last_rx_ns = now

            record = RxRecord(message=message, rx_epoch=self._rx_epoch, gen_at_rx=self._generation.get(),
                              received_monotonic_ns=now, stamp_ns=stamp_ns, frame_id=frame_id, width=width,
                              height=height, profile=profile, source_index=index)

            if self._latest is not None:
                self._stats["overwritten"] += 1

            self._latest = record
            self._stats["accepted"] += 1
            self._cond.notify_all()

            return "accepted"

    def on_clock(self, clock_ns: int) -> None:
        """`/clock` 값(시뮬레이션 ns)을 받는다. 줄면 되감기 경계를 만들고 필터를 켠다(R5)."""
        with self._cond:
            if self._closed:
                return

            self._clock_seen = True
            last = self._last_clock_ns

            if last is not None and clock_ns < last:
                pre_max = self._last_stamp_ns
                self._boundary(BoundaryKind.SIM_TIME_REWIND, f"clock {clock_ns} < {last}", self._clock_ns())
                self._filter_until_ns = pre_max
                self._filter_drops = 0
                self._last_stamp_ns = None

            self._last_clock_ns = clock_ns

            # 필터 해제: 새 실행의 clock이 되감기 전 최대 stamp를 넘으면 더 가를 수 없다
            if self._filter_until_ns is not None and clock_ns > self._filter_until_ns:
                self._filter_until_ns = None
                self._filter_drops = 0

    def on_invalid_input(self, kind: str) -> None:
        """해석할 수 없는 영상 헤더·`/clock`을 센다. 상태·epoch는 바꾸지 않는다."""
        if kind not in ("invalid_header", "invalid_clock"):
            raise ValueError(f"unknown invalid input kind: {kind!r}")

        with self._cond:
            self._stats[kind] += 1

    def on_camera_info(self, info: CameraProfileInfo) -> None:
        """`CameraInfo` 식별 정보를 받는다. 처음이거나 달라지면 `PROFILE_CHANGED` 경계(R4)."""
        with self._cond:
            if self._closed or info == self._profile:
                return

            self._profile = info
            self._boundary(BoundaryKind.PROFILE_CHANGED, f"calibration_id {info.calibration_id}", self._clock_ns(),
                           profile=info)

    def on_camera_info_invalid(self, detail: str) -> None:
        """해석할 수 없는 `CameraInfo`를 받았다.

        프로필이 있으면 지우고 프로필 없음 `PROFILE_CHANGED` 경계를 만든다. 이미 없으면 계수만 올린다.
        """
        with self._cond:
            if self._closed:
                return

            self._stats["camera_info_invalid"] += 1

            if self._profile is None:
                return

            self._profile = None
            self._boundary(BoundaryKind.PROFILE_CHANGED, f"camera_info invalid (profile cleared): {detail}",
                           self._clock_ns(), profile=None)

    def on_publisher_snapshot(self, gids: frozenset) -> None:
        """영상 발행자 GID 집합을 받는다(R9). 수신 중 확정 집합과 다르면 RECONNECT(흡수 창 안이면 흡수)."""
        with self._cond:
            if self._closed:
                return

            gids = frozenset(gids)
            self._publisher_count = len(gids)

            if not gids:
                return

            if self._confirmed_gids is None or gids == self._confirmed_gids:
                self._confirmed_gids = gids
                return

            self._confirmed_gids = gids
            now = self._clock_ns()

            # 끊긴 동안(또는 아직 첫 영상 전)의 변화는 경계가 아니다. 첫 새 영상이 R8 경계를 만든다
            if self._state != "streaming" or self._in_gap(now):
                self._stats["graph_change_during_gap"] += 1
                return

            # 같은 복구의 R8 경계 뒤 처음 보인 변화는 흡수(세대 중복 전이 방지)
            if self._absorb_until_ns is not None and now <= self._absorb_until_ns:
                self._absorb_until_ns = None
                self._stats["graph_change_absorbed"] += 1
                return

            self._absorb_until_ns = None
            self._boundary(BoundaryKind.RECONNECT, f"publisher set changed ({len(gids)} publisher(s))", now)

    def on_receiver_failed(self, detail: str) -> None:
        """수신기(executor)가 비정상으로 멈췄다. 자동 회복되지 않으므로 치명 알림 1회."""
        with self._cond:
            if self._closed or self._receiver_failed:
                return

            self._receiver_failed = True
            self._notice(NoticeKind.RECEIVER_FAILED, detail, self._clock_ns())

    def on_qos_incompatible(self, detail: str) -> None:
        """QoS 비호환 진단을 받는다. 같은 내용은 한 번만 알린다."""
        with self._cond:
            if self._closed or detail in self._qos_reported:
                return

            self._qos_reported.add(detail)
            self._notice(NoticeKind.QOS_INCOMPATIBLE, detail, self._clock_ns())

    # ------------------------------------------------------------------ 제어 흐름 쪽

    def poll_inputs(self) -> list[LiveInput]:
        """제한시간 판정 뒤 대기열을 모두 꺼낸다(순서 보존). 상한을 넘었으면 `LiveInputOverflow`."""
        now = self._clock_ns()

        with self._cond:
            if self._overflow:
                raise LiveInputOverflow(f"live input queue exceeded {self._config.max_pending_inputs}")

            timeout_ns = int(self._config.frame_timeout_s * 1e9)

            if self._state == "streaming" and self._last_rx_ns is not None and now - self._last_rx_ns >= timeout_ns:
                self._state = "timed_out"
                self._boundary(BoundaryKind.FRAME_TIMEOUT,
                               f"no frame for {(now - self._last_rx_ns) / 1e9:.3f} s", now)

            if (self._state == "waiting_first" and not self._startup_reported
                    and now - self._start_ns >= int(self._config.startup_timeout_s * 1e9)):
                self._startup_reported = True
                self._notice(NoticeKind.NO_FIRST_FRAME,
                             f"no frame within {self._config.startup_timeout_s} s "
                             f"(publishers={self._publisher_count}, clock_seen={self._clock_seen})", now)

            if self._overflow:
                raise LiveInputOverflow(f"live input queue exceeded {self._config.max_pending_inputs}")

            items = list(self._inputs)
            self._inputs.clear()

            return items

    def input_applied(self, item: LiveInput, generation: int) -> None:
        """제어 흐름이 항목을 코어에 반영했다고 알린다. 경계면 `applied_epoch`·`gen_of_epoch`를 갱신한다.

        경계는 꺼낸 순서대로 반영해야 하며, 어긋나면 ValueError(상태 무변).
        """
        with self._cond:
            if item.epoch is not None:
                if item.epoch != self._applied_epoch + 1:
                    raise ValueError(f"boundary epoch {item.epoch} applied out of order "
                                     f"(applied {self._applied_epoch})")

                self._applied_epoch = item.epoch
                self._gen_of_epoch[item.epoch] = generation

            self._generation.set(generation)
            self._cond.notify_all()

    def take(self, wait_s: float) -> Take:
        """새 영상을 꺼낸다. 없으면 최대 `wait_s`초 기다린다.

        이전 epoch 영상은 버리고(R1), 미반영 경계 뒤 영상이나 대기열 항목이 있으면 `NEEDS_INPUT`(R3).
        """
        deadline = time.monotonic() + max(0.0, wait_s)

        with self._cond:
            while True:
                if self._closed:
                    return Take(TakeStatus.CLOSED)

                record = self._latest

                if record is not None and record.rx_epoch < self._applied_epoch:
                    self._latest = None
                    self._stats["stale_epoch"] += 1
                    continue

                if self._inputs or self._overflow or (record is not None and record.rx_epoch > self._applied_epoch):
                    return Take(TakeStatus.NEEDS_INPUT)

                if record is not None:
                    self._latest = None
                    self._stats["taken"] += 1
                    generation = max(record.gen_at_rx, self._gen_of_epoch[record.rx_epoch])

                    return Take(TakeStatus.FRAME, record, generation)

                remaining = deadline - time.monotonic()

                if remaining <= 0:
                    return Take(TakeStatus.EMPTY)

                self._cond.wait(remaining)

    def is_current_epoch(self, rx_epoch: int) -> bool:
        """그 epoch가 반영된 최신 epoch이고 미반영 경계가 없는지(결과 공개 판정)."""
        with self._cond:
            pending = any(item.epoch is not None for item in self._inputs)

            return rx_epoch == self._applied_epoch and rx_epoch == self._rx_epoch and not pending

    def close(self) -> None:
        """닫는다. 최신 칸을 비우고 기다리는 `take`를 깨운다. 두 번 불러도 된다."""
        with self._cond:
            self._closed = True
            self._latest = None
            self._cond.notify_all()

    @property
    def closed(self) -> bool:
        """닫혔는가."""
        with self._cond:
            return self._closed

    def stats(self) -> dict:
        """통계 사본(수·epoch·상태·필터)."""
        with self._cond:
            out: dict[str, object] = dict(self._stats)
            out.update(rx_epoch=self._rx_epoch, applied_epoch=self._applied_epoch, state=self._state,
                       rewind_filter_active=self._filter_until_ns is not None,
                       publisher_count=self._publisher_count, clock_seen=self._clock_seen,
                       receiver_failed=self._receiver_failed,
                       pending_inputs=len(self._inputs))

            return out

    # ------------------------------------------------------------------ 내부(잠금 안에서만)

    def _in_gap(self, now: int) -> bool:
        """영상이 끊긴 상태인가(중단 판정됐거나, 마지막 수신 뒤 `frame_timeout_s` 이상 지남)."""
        if self._state == "timed_out":
            return True

        return (self._state == "streaming" and self._last_rx_ns is not None
                and now - self._last_rx_ns >= int(self._config.frame_timeout_s * 1e9))

    def _push(self, item: LiveInput) -> None:
        """대기열에 넣는다. 상한이면 넣지 않고 넘침을 표시한다."""
        if len(self._inputs) >= self._config.max_pending_inputs:
            self._overflow = True
        else:
            self._inputs.append(item)

        self._cond.notify_all()

    def _boundary(self, kind: BoundaryKind, detail: str, now: int, *, profile: CameraProfileInfo | None = None) -> None:
        """epoch를 올리고 같은 잠금 안에서 경계를 대기열에 넣는다."""
        self._rx_epoch += 1
        self._stats["boundaries"] += 1

        hook_error = None

        if self._boundary_hook is not None:
            try:
                self._boundary_hook(self._rx_epoch, kind)
            except Exception as exc:  # 훅 결함이 수신을 멈추지 않게 한다. 수를 세고 아래에서 치명 알림으로 올린다
                self._stats["boundary_hook_errors"] += 1
                hook_error = f"{type(exc).__name__}: {exc}"

        self._push(LiveInput(kind=kind, epoch=self._rx_epoch, detail=detail, event_monotonic_ns=now, profile=profile))

        # 훅이 현재성 갱신에 실패했을 수 있으므로 경계 뒤에 치명 알림을 넣는다
        if hook_error is not None:
            self._notice(NoticeKind.BOUNDARY_HOOK_FAILED,
                         f"boundary hook failed at epoch {self._rx_epoch} ({kind.value}): {hook_error}", now)

    def _notice(self, kind: NoticeKind, detail: str, now: int) -> None:
        """epoch 없는 알림을 넣는다."""
        self._stats["notices"] += 1
        self._push(LiveInput(kind=kind, epoch=None, detail=detail, event_monotonic_ns=now, profile=None))

    def _release_filter(self, now: int, detail: str) -> None:
        """되감기 필터를 풀고 알린다(정상 영상을 오판 중일 수 있음)."""
        self._filter_until_ns = None
        self._filter_drops = 0
        self._stats["rewind_filter_released"] += 1
        self._notice(NoticeKind.REWIND_FILTER_RELEASED, detail, now)


__all__ = ["BoundaryKind", "CameraProfileInfo", "GenerationMirror", "LiveFrameSlot", "LiveInput", "LiveInputOverflow",
           "LiveSlotConfig", "NoticeKind", "RxRecord", "Take", "TakeStatus"]
