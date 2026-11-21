"""Gazebo 링크 카메라 실시간 입력(ROS 2): `/link_cam/image`·`camera_info`·`/clock` → `FramePacket`.

`LiveLinkSource`가 실행기용 소스이고 수신 상태는 `LiveFrameSlot`에 있다. rclpy·메시지 패키지는
`RclpyTransport.start()` 안에서만 import한다. 설정 기본값은 A 임시 개발값이다.
"""
from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from eyetoarm.inputs.base import FrameIdIssuer, FramePacket, InputStatus, InputStatusEvent, TimestampBase
from eyetoarm.inputs.live_slot import (BoundaryKind, CameraProfileInfo, GenerationMirror, LiveFrameSlot, LiveInput, LiveSlotConfig,
                                       NoticeKind, TakeStatus)

# 허용 영상 인코딩. 그 밖은 DECODE_FAILED
ALLOWED_ENCODINGS = ("rgb8", "bgr8")

# 영상 번호 → epoch 대응을 보관하는 최근 프레임 수(공개 판정용)
_EPOCH_MEMORY = 256

_RELIABILITY_NAMES = ("reliable", "best_effort")


@dataclass(frozen=True)
class LiveSourceConfig:
    """실시간 소스 설정. 토픽·광학 프레임 기본값은 `config/robot/frames.yaml`의 `cameras.link`와 같다.

    `reliability`(영상 구독 신뢰성, 자동으로 바꾸지 않음), `frame_wait_s`(read 대기 상한),
    `graph_check_s`(발행자 확인 주기), `close_limit_s`(정리 상한), `slot`(상태기 설정).
    """

    image_topic: str = "/link_cam/image"
    camera_info_topic: str = "/link_cam/camera_info"
    clock_topic: str = "/clock"
    reliability: str = "reliable"
    frame_wait_s: float = 0.2
    graph_check_s: float = 0.5
    close_limit_s: float = 2.0
    node_name: str = "eyetoarm_a_link_input"
    slot: LiveSlotConfig = field(default_factory=lambda: LiveSlotConfig(expected_frame_id="link_cam_optical_frame"))

    def __post_init__(self) -> None:
        """값 범위를 검사한다. 어긋나면 ValueError."""
        for name in ("image_topic", "camera_info_topic", "clock_topic", "node_name"):
            value = getattr(self, name)

            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string: {value!r}")

        if self.reliability not in _RELIABILITY_NAMES:
            raise ValueError(f"reliability must be one of {_RELIABILITY_NAMES}: {self.reliability!r}")

        for name in ("frame_wait_s", "graph_check_s", "close_limit_s"):
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0 or value == float("inf"):
                raise ValueError(f"{name} must be a finite number > 0: {value!r}")

        if not isinstance(self.slot, LiveSlotConfig):
            raise ValueError("slot must be a LiveSlotConfig")


# ---------------------------------------------------------------------- 순수 변환 도구

def stamp_to_ns(stamp) -> int:
    """ROS 시각(`sec`·`nanosec`)을 정수 ns로. 음수·범위 밖 nanosec는 ValueError."""
    sec, nanosec = int(stamp.sec), int(stamp.nanosec)

    if sec < 0 or not 0 <= nanosec < 1_000_000_000:
        raise ValueError(f"invalid stamp sec={sec} nanosec={nanosec}")

    return sec * 1_000_000_000 + nanosec


def owned_bgr(image) -> np.ndarray:
    """변환 결과를 독립·연속·읽기 전용 BGR uint8 (H, W, 3) 배열로 만든다. 버퍼를 공유하면 복사한다."""
    array = np.asarray(image)

    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"converted image must be uint8 (H, W, 3): {array.dtype} {array.shape}")

    if array.base is not None or not array.flags.owndata or not array.flags.c_contiguous:
        array = np.array(array, copy=True, order="C")

    array.setflags(write=False)

    return array


def default_convert(message) -> np.ndarray:
    """`cv_bridge`로 `bgr8` 배열을 만든다(지연 import). bgr8은 버퍼 공유일 수 있다."""
    from cv_bridge import CvBridge

    return CvBridge().imgmsg_to_cv2(message, desired_encoding="bgr8")


def qos_incompatibility(requested: str, offered: list[str]) -> str | None:
    """요청 신뢰성과 발행자 offered 신뢰성 목록을 비교한다. 비호환이면 진단 문구, 아니면 None.

    reliable 요청은 best-effort 발행자와 연결되지 않는다.
    """
    if requested == "reliable" and "best_effort" in offered:
        return "reliability offered=BEST_EFFORT requested=RELIABLE"

    return None


def profile_info(message) -> CameraProfileInfo:
    """`CameraInfo`(또는 같은 속성의 객체)에서 식별 정보를 만든다(B `from_camera_info` 사용)."""
    from eyetoarm.calibration.profile import from_camera_info

    profile = from_camera_info(message)

    return CameraProfileInfo(calibration_id=profile.calibration_id, width=profile.width, height=profile.height, camera=profile,
                             frame_id=profile.frame_id)


# ---------------------------------------------------------------------- 소스

class LiveLinkSource:
    """링크 카메라 실시간 소스. 실행기의 단일 발급기·세대 공급자를 받는다.

    `read()`는 새 영상이면 `FramePacket`, 없거나 입력 반영이 먼저면 None, 닫혔으면 `SOURCE_UNAVAILABLE`.
    생성자가 연결을 시작하고, 시작 실패는 첫 `poll_input` 알림으로 알린다.
    제어 흐름 메서드는 한 스레드에서 차례로, `handle_*`는 수신 스레드에서 부른다.
    """

    def __init__(self, *, source_id: str, issuer: FrameIdIssuer, generation_provider: Callable[[], int],
                 config: LiveSourceConfig | None = None,
                 transport_factory: Callable[["LiveLinkSource"], object] | None = None,
                 convert: Callable[[object], np.ndarray] = default_convert,
                 clock_ns: Callable[[], int] = time.monotonic_ns,
                 boundary_hook: Callable[[int, BoundaryKind], None] | None = None) -> None:
        """설정을 받아 상태기를 만들고 수신 연결을 시작한다."""
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"source_id must be a non-empty string: {source_id!r}")

        self._source_id = source_id
        self._issuer = issuer
        self._config = config or LiveSourceConfig()
        self._convert = convert
        self._clock_ns = clock_ns
        self._mirror = GenerationMirror(generation_provider())
        self._slot = LiveFrameSlot(self._config.slot, generation=self._mirror, clock_ns=clock_ns,
                                   boundary_hook=boundary_hook)
        self._epochs: collections.OrderedDict[int, int] = collections.OrderedDict()
        self._callback_ns: collections.deque[int] = collections.deque(maxlen=1000)
        self._callback_lock = threading.Lock()
        self._offered: list[str] = []
        self._start_error: str | None = None
        self._receiver_error: str | None = None
        self._close_done = False
        self._start_reported = False
        self._closed = False
        self._decode_failures = 0
        self._frames = 0

        factory = transport_factory or (lambda src: RclpyTransport(src.config))
        self._transport: Any = None

        try:
            self._transport = factory(self)
            self._transport.start(self)
        except Exception as exc:  # rclpy 없음·초기화 실패 등: 사건으로 알리고 닫힌 소스로 둔다
            self._start_error = f"{type(exc).__name__}: {exc}"
            self._slot.close()

    # ------------------------------------------------------------------ 조회

    @property
    def source_id(self) -> str:
        """소스 ID."""
        return self._source_id

    @property
    def config(self) -> LiveSourceConfig:
        """소스 설정."""
        return self._config

    @property
    def slot(self) -> LiveFrameSlot:
        """수신 상태기(시험·통계용)."""
        return self._slot

    # ------------------------------------------------------------------ 수신 스레드 쪽(메시지 속성만 읽음)

    def handle_image(self, message) -> str:
        """영상 메시지를 상태기에 넘기고 콜백 시간을 잰다. 헤더가 잘못되면 받지 않는다."""
        t0 = time.perf_counter_ns()

        try:
            stamp_ns = stamp_to_ns(message.header.stamp)
            outcome = self._slot.on_image(message, stamp_ns=stamp_ns, frame_id=str(message.header.frame_id),
                                          width=int(message.width), height=int(message.height))
        except (AttributeError, TypeError, ValueError):
            outcome = "invalid_header"
            self._slot.on_invalid_input("invalid_header")  # 버리되 수신 진단에서 사라지지 않게 센다

        with self._callback_lock:
            self._callback_ns.append(time.perf_counter_ns() - t0)

        return outcome

    def handle_clock(self, message) -> None:
        """`/clock` 메시지를 넘긴다. 잘못된 값은 버리고 `invalid_clock`으로 센다."""
        try:
            clock_ns = stamp_to_ns(message.clock)
        except (AttributeError, TypeError, ValueError):
            self._slot.on_invalid_input("invalid_clock")
            return

        self._slot.on_clock(clock_ns)

    def handle_camera_info(self, message) -> bool:
        """`CameraInfo`를 식별 정보로 바꿔 넘긴다. 만들 수 없으면 False.

        해석 실패는 slot이 이전 프로필을 지우게 해 이후 영상이 이전 `calibration_id`를 달지 않게 한다.
        """
        try:
            info = profile_info(message)
        except (AttributeError, TypeError, ValueError) as exc:
            self._slot.on_camera_info_invalid(f"{type(exc).__name__}: {exc}")
            return False

        self._slot.on_camera_info(info)

        return True

    def handle_publishers(self, gids, offered: list[str]) -> None:
        """발행자 GID 목록과 offered 신뢰성 목록을 넘긴다(R9). 비호환이면 QoS 진단 알림."""
        self._offered = list(offered)
        self._slot.on_publisher_snapshot(frozenset(gids))
        problem = qos_incompatibility(self._config.reliability, offered)

        if problem is not None:
            self.handle_qos_incompatible(problem)

    def handle_receiver_failed(self, detail: str) -> None:
        """수신 스레드가 비정상으로 멈췄다(spin 예외·정리 요청 없는 반환). 치명 알림으로 넘긴다."""
        self._receiver_error = detail
        self._slot.on_receiver_failed(detail)

    def handle_qos_incompatible(self, detail: str) -> None:
        """QoS 비호환 사건(구독 이벤트 콜백 또는 발행자 QoS 비교)을 알린다."""
        self._slot.on_qos_incompatible(f"qos_incompatible: {detail}")

    # ------------------------------------------------------------------ 제어 흐름 쪽

    def poll_input(self) -> list[LiveInput]:
        """경계·알림을 꺼낸다. 시작 실패는 첫 호출에서 알림 하나로 돌려준다."""
        if self._start_error is not None:
            if self._start_reported:
                return []

            self._start_reported = True

            return [LiveInput(kind=NoticeKind.START_FAILED, epoch=None, detail=f"start failed: {self._start_error}",
                              event_monotonic_ns=self._clock_ns(), profile=None)]

        return self._slot.poll_inputs()

    def status_event(self, item: LiveInput) -> InputStatusEvent | None:
        """항목의 입력 상태 사건(코어 `note_input_status`용). 상태가 없는 항목은 None."""
        status = item.status

        if status is None:
            return None

        return InputStatusEvent(run_id=self._issuer.run_id, source_id=self._source_id, status=status,
                                event_monotonic_ns=item.event_monotonic_ns, detail=item.detail,
                                source_ref=self._config.image_topic)

    def input_applied(self, item: LiveInput, generation: int) -> None:
        """실행기가 항목을 반영했다고 알린다(경계 epoch·세대 사본 갱신)."""
        if item.epoch is None and self._start_error is not None:
            return

        self._slot.input_applied(item, generation)

    def read(self) -> FramePacket | InputStatusEvent | None:
        """새 영상을 조립해 돌려준다(최대 `frame_wait_s` 대기)."""
        if self._closed or self._start_error is not None:
            return self._event(InputStatus.SOURCE_UNAVAILABLE, "closed")

        take = self._slot.take(self._config.frame_wait_s)

        if take.status is TakeStatus.CLOSED:
            return self._event(InputStatus.SOURCE_UNAVAILABLE, "closed")

        if take.status is not TakeStatus.FRAME:
            return None

        record = take.record
        generation = take.capture_generation
        assert record is not None and generation is not None  # FRAME이면 둘 다 있다
        message = record.message
        encoding = getattr(message, "encoding", None)

        if encoding not in ALLOWED_ENCODINGS:
            return self._decode_failed(f"unsupported encoding {encoding!r}")

        try:
            image = owned_bgr(self._convert(message))
        except Exception as exc:  # 변환기 예외 종류는 cv_bridge 버전마다 달라 모두 해석 실패로 본다
            return self._decode_failed(f"convert failed: {type(exc).__name__}: {exc}")

        if image.shape[:2] != (record.height, record.width):
            return self._decode_failed(f"image shape {image.shape[:2]} != header {(record.height, record.width)}")

        # 성공한 프레임에만 발급기 1회(시계는 콜백에서 이미 1회 읽음)
        seq = self._issuer.next_seq()
        profile = record.profile
        frame = FramePacket(image=image, run_id=self._issuer.run_id, source_id=self._source_id, image_seq=seq,
                            timestamp_ms=record.stamp_ns // 1_000_000, timestamp_base=TimestampBase.SIM_TIME,
                            received_monotonic_ns=record.received_monotonic_ns, width=record.width,
                            height=record.height, crop=None,
                            calibration_id=None if profile is None else profile.calibration_id,
                            capture_generation=generation, optical_frame=record.frame_id,
                            ros_stamp_ns=record.stamp_ns, source_ref=self._config.image_topic,
                            source_index=record.source_index)

        self._epochs[seq] = record.rx_epoch

        while len(self._epochs) > _EPOCH_MEMORY:
            self._epochs.popitem(last=False)

        self._frames += 1

        return frame

    def frame_current(self, frame: FramePacket) -> bool:
        """그 프레임의 수신 epoch가 지금도 현재인지. 모르는 프레임은 False."""
        epoch = self._epochs.get(frame.image_seq) if frame.run_id == self._issuer.run_id else None

        return epoch is not None and self._slot.is_current_epoch(epoch)

    def frame_epoch(self, frame: FramePacket) -> int | None:
        """그 프레임을 받은 수신 epoch(모르면 None). 제어 흐름 전용."""
        if frame.run_id != self._issuer.run_id:
            return None

        return self._epochs.get(frame.image_seq)

    def stats(self) -> dict:
        """수신·조립 통계(콜백 시간 최대·P95 ns, 발행자 offered 신뢰성 포함)."""
        out = self._slot.stats()

        with self._callback_lock:
            samples = sorted(self._callback_ns)

        out.update(frames=self._frames, decode_failures=self._decode_failures, offered_reliability=list(self._offered),
                   callback_ns_max=samples[-1] if samples else None,
                   callback_ns_p95=samples[min(len(samples) - 1, int(0.95 * len(samples)))] if samples else None,
                   start_error=self._start_error, receiver_error=self._receiver_error)

        return out

    def close(self) -> None:
        """수신 연결을 끊고 상태기를 닫는다.

        연결 정리가 실패하면 RuntimeError를 올리고 완료로 표시하지 않아, 다시 부르면 남은 단계를 재시도한다.
        """
        if self._close_done:
            return

        self._closed = True
        self._slot.close()

        if self._transport is not None and self._start_error is None:
            self._transport.close(self._config.close_limit_s)

        self._close_done = True

    # ------------------------------------------------------------------ 내부

    def _event(self, status: InputStatus, detail: str) -> InputStatusEvent:
        """입력 사건 하나(시계 1회)."""
        return InputStatusEvent(run_id=self._issuer.run_id, source_id=self._source_id, status=status,
                                event_monotonic_ns=self._clock_ns(), detail=detail,
                                source_ref=self._config.image_topic)

    def _decode_failed(self, detail: str) -> InputStatusEvent:
        """해석 실패 사건. 발급기를 부르지 않는다."""
        self._decode_failures += 1

        return self._event(InputStatus.DECODE_FAILED, detail)


# ---------------------------------------------------------------------- 실제 rclpy 연결

class RclpyTransport:
    """rclpy 수신 연결. 자기 `Context`(또는 주입받은 것)·노드·구독 3개·확인 타이머·수신 스레드 1개.

    콜백은 `LiveLinkSource.handle_*`만 부른다. 주입받은 `Context`는 종료하지 않는다.
    """

    THREAD_NAME = "eyetoarm-ros-rx"

    def __init__(self, config: LiveSourceConfig, *, context=None) -> None:
        """설정과 선택 주입 컨텍스트를 받는다. 주입 컨텍스트는 이 객체가 종료하지 않는다."""
        self._config = config
        self._context = context
        self._owns_context = context is None
        self._node: Any = None
        self._executor: Any = None
        self._thread: threading.Thread | None = None
        self._source: Any = None
        self._closing = False
        self._done: set[str] = set()
        self.spin_error: str | None = None
        self.teardown_errors: list[str] = []

    def start(self, source: LiveLinkSource) -> None:
        """rclpy를 지연 import해 노드·구독·타이머·executor 스레드를 만든다."""
        import rclpy
        from rclpy.context import Context
        from rclpy.event_handler import SubscriptionEventCallbacks
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from rclpy.signals import SignalHandlerOptions
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import CameraInfo, Image

        cfg = self._config
        self._source = source

        if self._owns_context:
            self._context = Context()
            rclpy.init(context=self._context, signal_handler_options=SignalHandlerOptions.NO)

        try:
            node = rclpy.create_node(cfg.node_name, context=self._context)
            self._node = node

            reliability = ReliabilityPolicy.RELIABLE if cfg.reliability == "reliable" else ReliabilityPolicy.BEST_EFFORT
            image_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=reliability,
                                   durability=DurabilityPolicy.VOLATILE)
            info_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                  durability=DurabilityPolicy.VOLATILE)
            clock_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.RELIABLE,
                                   durability=DurabilityPolicy.VOLATILE)

            def on_incompatible(event) -> None:
                """구독 QoS 비호환 사건."""
                source.handle_qos_incompatible(f"requested incompatible qos (policy={event.last_policy_kind}, "
                                               f"total={event.total_count})")

            events = SubscriptionEventCallbacks(incompatible_qos=on_incompatible)
            def on_image(message) -> None:
                """영상 콜백."""
                source.handle_image(message)

            def on_info(message) -> None:
                """CameraInfo 콜백."""
                source.handle_camera_info(message)

            node.create_subscription(Image, cfg.image_topic, on_image, image_qos, event_callbacks=events)
            node.create_subscription(CameraInfo, cfg.camera_info_topic, on_info, info_qos)
            node.create_subscription(Clock, cfg.clock_topic, source.handle_clock, clock_qos)

            def check_publishers() -> None:
                """영상 발행자 수와 offered 신뢰성을 확인한다."""
                infos = node.get_publishers_info_by_topic(cfg.image_topic)
                offered = ["best_effort" if i.qos_profile.reliability == ReliabilityPolicy.BEST_EFFORT else "reliable"
                           for i in infos]
                source.handle_publishers([bytes(i.endpoint_gid) for i in infos], offered)

            node.create_timer(cfg.graph_check_s, check_publishers)

            self._executor = SingleThreadedExecutor(context=self._context)
            self._executor.add_node(node)
            self._thread = threading.Thread(target=self._spin, name=self.THREAD_NAME, daemon=True)
            self._thread.start()
        except BaseException:
            self._teardown()
            raise

    def _spin(self) -> None:
        """수신 스레드 본체. `close` 요청 없이 끝나면 예외든 정상 반환이든 비정상 종료로 알린다."""
        error = None

        try:
            self._executor.spin()
        except BaseException as exc:  # 스레드 밖으로 내보내지 않고 원인을 남긴다
            error = f"{type(exc).__name__}: {exc}"

        self.spin_error = error

        if not self._closing and self._source is not None:
            try:
                self._source.handle_receiver_failed(f"receiver stopped unexpectedly: {error or 'spin returned'}")
            except Exception as exc:
                # 무시해도 결과가 유효로 바뀌지 않는다: 원인은 `spin_error`에 남았고, 영상이 끊기면 FRAME_TIMEOUT
                # 경계가 나온다. 알림 실패 사유만 원인 뒤에 덧붙인다
                self.spin_error = f"{self.spin_error or 'spin returned'}; notify failed: {type(exc).__name__}: {exc}"

    def close(self, limit_s: float) -> None:
        """executor 종료 → 수신 스레드 join → 노드 해제 → (소유한) 컨텍스트 종료를 `limit_s` 안에서 한다.

        실패해도 뒤 단계를 계속하고 모아서 RuntimeError로 올린다. 실패한 단계만 다음 `close`에서 재시도한다.
        """
        self._closing = True
        errors: list[str] = []

        if self._executor is not None and "shutdown" not in self._done:
            try:
                self._executor.shutdown(timeout_sec=limit_s)
                self._done.add("shutdown")
            except Exception as exc:
                errors.append(f"executor shutdown failed: {type(exc).__name__}: {exc}")

        if self._thread is not None:
            self._thread.join(limit_s)

            if self._thread.is_alive():
                errors.append(f"ros receive thread did not stop within {limit_s} s")

        errors += self._teardown()

        if errors:
            raise RuntimeError("; ".join(errors))

    def _teardown(self) -> list[str]:
        """노드 해제·(소유한) 컨텍스트 종료. 실패해도 다음 단계를 계속하고 실패 목록을 돌려준다."""
        errors: list[str] = []

        if self._node is not None:
            try:
                self._node.destroy_node()
                self._node = None
            except Exception as exc:
                errors.append(f"node destroy failed: {type(exc).__name__}: {exc}")

        if self._owns_context and self._context is not None:
            try:
                self._context.try_shutdown()
                self._context = None
            except Exception as exc:
                errors.append(f"context shutdown failed: {type(exc).__name__}: {exc}")

        self.teardown_errors += errors

        return errors


__all__ = ["ALLOWED_ENCODINGS", "LiveLinkSource", "LiveSourceConfig", "RclpyTransport", "default_convert",
           "owned_bgr", "profile_info", "qos_incompatibility", "stamp_to_ns"]
