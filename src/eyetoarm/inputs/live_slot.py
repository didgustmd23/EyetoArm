"""실시간 입력 수신 상태기(ROS 없음) — 최신 칸 1개, 수신 epoch, 입력 경계 대기열.

근거: 계획 `docs/superpowers/plans/2026-11-16-live-link-camera.md` r3 §3.3(최신 프레임 우선·소유권), §3.4(수신 epoch·
규칙 R1~R7), §3.5(결과 공개 전 입력 반영), A 상세설계 §5.1(`received_monotonic_ns`는 수신 순간 한 번), §5.4(세대 증가 조건:
재접속·시뮬레이션 시각 되감기·프로필 변경), §6(Gazebo: stamp·시뮬레이션 시각·PC 수신 중단을 함께 검사).

역할 분담
- **수신 스레드**(ROS 콜백)는 `on_image`·`on_clock`·`on_camera_info`·`on_publisher_snapshot`·`on_qos_incompatible`·
  `on_receiver_failed`만 부른다.
  이 메서드들은 잠금 안에서 상태를 바꾸고 최신 칸을 교체할 뿐, 변환·검출·기록을 하지 않는다.
- **제어 흐름**(실행기)은 `poll_inputs`·`input_applied`·`take`·`is_current_epoch`·`close`를 부른다. 코어(`VisionPipeline`)는
  이 모듈이 부르지 않는다.

수신 epoch(계획 §3.4)
- 입력 쪽 경계(되감기·재접속·프로필 변경·수신 중단)를 감지할 때마다 잠금 안에서 `rx_epoch`를 1 올리고, 같은 잠금 안에서 그
  경계를 대기열에 넣는다. 영상 수신 기록은 받은 순간의 `rx_epoch`를 갖는다.
- R1: 제어 흐름이 반영한 마지막 경계 epoch(`applied_epoch`)보다 작은 epoch의 영상은 넘기지 않는다(`stale_epoch`).
- R2: `capture_generation = max(gen_at_rx, gen_of_epoch[rx_epoch])`. 수신 뒤에 생긴 세대를 붙이지 않는다.
- R3: 반영되지 않은 경계 뒤의 영상은 넘기지 않고 먼저 입력 반영을 요구한다(`TakeStatus.NEEDS_INPUT`).
- R4: 프로필(`calibration_id`)은 수신 기록을 만들 때의 현재 프로필에서 정한다.
- R5: `/clock` 되감기 뒤에는 휴리스틱 필터를 켠다. **이전 실행 영상을 완전히 막는다는 보장이 아니다**(계획 r2 한계 1·2).
- R6: 영상 stamp가 줄면 그 영상에서 되감기 경계를 만든다. 같은 되감기에서 `/clock` 경계가 또 생기지 않게 기준을 비운다.
- R7: 같은 stamp는 넘기지 않는다(`duplicate_stamp`).
- R8(사용자 확정 2026-11-16, #124 대안 0): 영상 중단(`frame_timeout_s`) 뒤 첫 새 영상은 **발행자 변경 여부를 구분하지 않고** 그 콜백에서
  `RECONNECT` 경계를 만들어 새 epoch로 받는다(일시 정지 재개도 세대 +1, 이전 「재개 세대 유지」 정책을 이 범위에서 대체).
  제어 흐름이 중단을 아직 판정하지 못했어도 간격이 `frame_timeout_s` 이상이면 같은 콜백에서 `FRAME_TIMEOUT` → `RECONNECT` 순서로 만든다.
- R9: 발행자 그래프는 GID 집합으로 비교한다. 수신 중(끊김 아님) 확정 집합과 다른 비지 않은 집합이 보이면 `RECONNECT`(중단 없는 교체 보완).
  빈 집합·끊긴 동안의 변화는 경계가 아니다(첫 새 영상이 경계를 만든다). R8 경계 뒤 `graph_absorb_s` 안에 처음 보인 변화 1회는 같은
  복구로 흡수해 세대를 또 올리지 않는다. 창 뒤의 변화는 새 경계다(같은 복구의 두 번째 전이가 될 수 있음).
- 한계(보장 아님): 짧은 끊김(`frame_timeout_s` 미만) 미감지, 그래프 지연(타이머 주기 동안 새 발행자 영상이 이전 epoch로 처리될 수
  있음), 메시지별 출처 미확정.

값(`LiveSlotConfig` 기본값)은 모두 **A 임시 개발값**(계획 결정 D5 기본안)이며 사용자 확정값이 아니다.
표준 라이브러리만 쓴다. ROS·Qt·OpenCV를 import하지 않는다.
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
    """수신 epoch를 올리는 입력 경계. 세대 전이 값은 `GenerationCause`의 같은 이름 문자열과 같다."""

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
    BOUNDARY_HOOK_FAILED = "boundary_hook_failed"   # #180 #5: 경계 훅 예외(치명, 연결부가 상태를 fail-stop)


# 경계·알림 → 코어에 알릴 입력 상태(None이면 상태 사건이 아님: 세대 전이 또는 기록만)
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

    `camera`(#171 R1, 선택): 같은 CameraInfo에서 만든 B `CameraProfile`(K·왜곡) 원본. 같음 비교·repr에는 넣지 않는다(경계 판정은
    지금처럼 식별 정보로만 한다. `calibration_id`가 K·왜곡·해상도 해시라 내용이 다르면 식별 정보도 다르다).
    """

    calibration_id: str
    width: int
    height: int
    frame_id: str
    camera: object | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class LiveInput:
    """입력 대기열 항목. 경계면 `epoch`가 새 epoch, 알림이면 None. `status`는 코어에 알릴 입력 상태(없으면 None)."""

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
    """입력 대기열 상한을 넘었다(경계·알림을 잃지 않으려고 실행을 실패로 끝내게 한다)."""


def _positive_finite(value) -> bool:
    """bool이 아닌 유한한 양수인지."""
    return (not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
            and value > 0)


@dataclass(frozen=True)
class LiveSlotConfig:
    """상태기 설정. 기본값은 모두 A 임시 개발값(계획 결정 D5 기본안)이다.

    - `frame_timeout_s`(1.0): 영상이 이만큼 안 오면 `FRAME_TIMEOUT` 경계(PC 단조 시각).
    - `startup_timeout_s`(10.0): 시작 뒤 첫 영상이 이만큼 안 오면 `NO_FIRST_FRAME` 알림 1회.
    - `future_tolerance_ns`(0.5 s): 되감기 필터에서 영상 stamp가 최신 `/clock`보다 앞서도 받는 허용 폭(시뮬레이션 시간).
    - `stale_after_rewind_limit`(30): 되감기 필터가 연속으로 버린 영상이 이 수를 넘으면 필터를 풀고 알림.
    - `max_pending_inputs`(64): 입력 대기열 상한. 넘으면 `LiveInputOverflow`.
    - `graph_absorb_s`(5.0): R8 경계 뒤 그래프 변화 1회를 같은 복구로 흡수하는 창(PC 단조, A 임시 개발값).
    - `expected_frame_id`: 영상 헤더 기대값. 다르면 그 영상에 `calibration_id`를 싣지 않는다(결정 D7 기본안). None이면 검사 안 함.
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
    """코어 세대의 잠금 보호 사본(계획 결정 D10 기본안). 제어 흐름이 쓰고 수신 콜백은 읽기만 한다."""

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
               "boundary_hook_errors", "camera_info_invalid")


class LiveFrameSlot:
    """수신 스레드와 제어 흐름 사이의 최신 칸 1개·입력 대기열(계획 §3.3·§3.4).

    - 최신 칸에는 아직 넘기지 않은 마지막 영상 기록 하나만 둔다. 새 영상이 오면 덮어쓰고 `overwritten`을 센다.
    - 경계·알림 대기열은 순서를 보존하고 버리지 않는다. 상한을 넘으면 다음 `poll_inputs`가 `LiveInputOverflow`를 올린다.
    - 시계(`clock_ns`)는 영상 콜백마다 한 번 읽는다(수신 시각). 그 밖에는 경계 시각·제한시간 판정에만 읽는다.
    """

    def __init__(self, config: LiveSlotConfig, *, generation: GenerationMirror,
                 clock_ns: Callable[[], int] = time.monotonic_ns,
                 boundary_hook: Callable[[int, BoundaryKind], None] | None = None) -> None:
        """설정·세대 사본·PC 단조 시계를 받는다. 시작 시각을 한 번 읽는다(첫 영상 제한시간 기준).

        `boundary_hook(epoch, kind)`(#171 R1, 선택, 기본 None): 경계마다 **이 잠금 안에서** epoch를 올린 직후 부른다(잠금 순서
        slot → 훅 쪽 잠금). 훅은 이 slot의 잠금을 다시 잡거나 오래 막으면 안 된다. 예외는 잡아서 `boundary_hook_errors`로 세고,
        그 경계 뒤에 치명 알림 `BOUNDARY_HOOK_FAILED`(입력 상태 SOURCE_UNAVAILABLE)를 넣는다(#180 #5).
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

            # R8: 중단 뒤 첫 새 영상은 발행자 확인 없이 RECONNECT 경계(사용자 확정 대안 0)
            if self._in_gap(now):
                if self._state == "streaming":
                    self._state = "timed_out"
                    self._boundary(BoundaryKind.FRAME_TIMEOUT,
                                   f"no frame for {(now - self._last_rx_ns) / 1e9:.3f} s (detected at image)", now)

                self._boundary(BoundaryKind.RECONNECT, "first frame after frame_timeout (origin not checked)", now)
                self._stats["resume_boundaries"] += 1
                self._absorb_until_ns = now + int(self._config.graph_absorb_s * 1e9)

                # 같은 영상의 stamp 되감기는 이미 새 경계 안에 있으므로 따로 세대를 올리지 않는다
                if self._last_stamp_ns is not None and stamp_ns < self._last_stamp_ns:
                    self._stats["rewind_merged_into_resume"] += 1
                    self._last_clock_ns = None

            # R6: 영상 stamp 되감기. 같은 되감기의 /clock 경계를 막으려고 clock 기준을 비운다
            elif self._last_stamp_ns is not None and stamp_ns < self._last_stamp_ns:
                self._boundary(BoundaryKind.SIM_TIME_REWIND, f"image stamp {stamp_ns} < {self._last_stamp_ns}", now)
                self._last_clock_ns = None

            self._last_stamp_ns = stamp_ns

            # R4·D7: 수신 시점 프로필. 헤더가 기대값과 다르거나 해상도가 다르면 싣지 않는다
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

    def on_camera_info(self, info: CameraProfileInfo) -> None:
        """`CameraInfo` 식별 정보를 받는다. 처음이거나 달라지면 `PROFILE_CHANGED` 경계(R4)."""
        with self._cond:
            if self._closed or info == self._profile:
                return

            self._profile = info
            self._boundary(BoundaryKind.PROFILE_CHANGED, f"calibration_id {info.calibration_id}", self._clock_ns(),
                           profile=info)

    def on_camera_info_invalid(self, detail: str) -> None:
        """해석할 수 없는 `CameraInfo`(잘못된 K 등)를 받았다(#180 #1, #178 결함 1).

        이전 프로필로 계속 `calibration_id`를 달지 않도록, 프로필이 있으면 지우고 프로필 없음 `PROFILE_CHANGED` 경계를 만든다
        (A 상세설계 §5.4: 보정 변경은 세대 전이, 이후 프레임은 보정 없음). 이미 프로필이 없으면 경계 없이 계수만 올린다
        (연속 무효가 경계를 쏟아내지 않게). 다시 유효한 CameraInfo가 오면 `on_camera_info`가 새 경계를 만든다.
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
        """영상 발행자 GID 집합을 받는다(R9). 수신 중 확정 집합과 다른 비지 않은 집합이면 RECONNECT(흡수 창 안이면 흡수)."""
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
        """수신기(executor)가 비정상으로 멈췄다. 자동 회복되지 않으므로 치명 알림 1회(실행기가 실패로 끝낸다)."""
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

        경계는 꺼낸 순서대로 반영해야 한다. 순서가 어긋나면 ValueError(상태 무변). 세대 사본도 갱신한다.
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
        """새 영상을 꺼낸다. 없으면 최대 `wait_s`초 기다린다(실제 단조 시계).

        - 이전 epoch 영상은 버리고(R1) 계속 기다린다.
        - 반영 안 된 경계 뒤 영상이 있거나 대기열에 항목이 있으면 `NEEDS_INPUT`(R3).
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
        """그 epoch가 반영된 최신 epoch이고 반영 안 된 경계가 없는지(결과 공개 판정, 계획 §3.5 C)."""
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
        """영상이 끊긴 상태인가: 중단으로 판정됐거나, 수신 중인데 마지막 수신 뒤 `frame_timeout_s` 이상 지남."""
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
            except Exception as exc:  # 훅 결함이 수신을 멈추지 않게 한다. 수를 세고 아래에서 치명 알림으로 올린다(#180 #5)
                self._stats["boundary_hook_errors"] += 1
                hook_error = f"{type(exc).__name__}: {exc}"

        self._push(LiveInput(kind=kind, epoch=self._rx_epoch, detail=detail, event_monotonic_ns=now, profile=profile))

        # 훅이 상태(현재성) 갱신에 실패했으면 이전 수락이 현재로 남을 수 있다. 실행기·연결부가 알도록 경계 뒤에 알린다
        if hook_error is not None:
            self._notice(NoticeKind.BOUNDARY_HOOK_FAILED,
                         f"boundary hook failed at epoch {self._rx_epoch} ({kind.value}): {hook_error}", now)

    def _notice(self, kind: NoticeKind, detail: str, now: int) -> None:
        """epoch 없는 알림을 넣는다."""
        self._stats["notices"] += 1
        self._push(LiveInput(kind=kind, epoch=None, detail=detail, event_monotonic_ns=now, profile=None))

    def _release_filter(self, now: int, detail: str) -> None:
        """되감기 필터를 풀고 알린다(정상 영상을 오판 중일 수 있음, 계획 r2 한계 2)."""
        self._filter_until_ns = None
        self._filter_drops = 0
        self._stats["rewind_filter_released"] += 1
        self._notice(NoticeKind.REWIND_FILTER_RELEASED, detail, now)


__all__ = ["BoundaryKind", "CameraProfileInfo", "GenerationMirror", "LiveFrameSlot", "LiveInput", "LiveInputOverflow",
           "LiveSlotConfig", "NoticeKind", "RxRecord", "Take", "TakeStatus"]
