"""공통 입력·관측 자료형(`FramePacket`·`PlanarObservation`·`FrameObservations` 등)과 공통 검사 함수.

담당별 확장(예: A 입력의 `source_ref`·`source_index`)은 각 모듈에서 상속해 둔다.
표준 라이브러리와 NumPy만 쓰고 Qt·ROS·OpenCV를 import하지 않는다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

import numpy as np


class TimestampBase(str, Enum):
    """`timestamp_ms`의 시간 기준. 기준이 다른 시각끼리 빼지 않는다."""

    PC_RECEIVE = "PC_RECEIVE"
    PHONE_CAPTURE = "PHONE_CAPTURE"
    SIM_TIME = "SIM_TIME"
    VIDEO_POS = "VIDEO_POS"


class InputStatus(str, Enum):
    """프레임이 없는 입력 사건의 상태. 검출 판정 사유와 섞지 않는다."""

    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    FRAME_TIMEOUT = "FRAME_TIMEOUT"
    DECODE_FAILED = "DECODE_FAILED"
    END_OF_STREAM = "END_OF_STREAM"


@dataclass(frozen=True)
class FramePacket:
    """성공적으로 받은 프레임 하나(14개 필드, 기본값 없음).

    `image`는 BGR `uint8` (H, W, 3) 계약이지만 여기서 검사하지 않고, 시계·순번도 만들지 않는다.
    `frozen=True`는 필드 재할당만 막는다. 버퍼 독립·쓰기 제한은 프레임을 만드는 입력 소스 책임이다.
    """

    image: np.ndarray
    run_id: str
    source_id: str
    image_seq: int
    timestamp_ms: int
    timestamp_base: TimestampBase
    received_monotonic_ns: int
    width: int
    height: int
    crop: tuple[int, int, int, int] | None
    calibration_id: str | None
    capture_generation: int
    optical_frame: str | None
    ros_stamp_ns: int | None


@dataclass(frozen=True)
class InputStatusEvent:
    """프레임을 받지 못했거나 받기를 끝낸 사건(가칭).

    프레임이 없어 `image_seq`·`image`·`timestamp_ms`를 두지 않고, 시각은 PC 단조 시각
    (`event_monotonic_ns`)이다. 모든 필드에 기본값이 없다.
    """

    run_id: str
    source_id: str
    status: InputStatus
    event_monotonic_ns: int
    detail: str | None


class PixelBasis(str, Enum):
    """윤곽 점 좌표의 기준. 왜곡 보정 전(`raw`)·후(`undistorted`)."""

    RAW = "raw"
    UNDISTORTED = "undistorted"


# 윤곽 점 배열: numpy (N, 2) float32. ROI·표시 축소를 되돌린 원본 해상도·크롭 기준 픽셀이며,
# 왜곡 보정 여부는 함께 담는 `PixelBasis`로 구분한다.
Contour: TypeAlias = np.ndarray

# 볼트 축 두 끝점: numpy (2, 2) float64, 행 0 = 머리 끝, 행 1 = 축 끝. FramePacket 영상 전체 기준
# 픽셀(ROI 오프셋·표시 축소 복원)이며 왜곡 보정 여부는 `PixelBasis`로 구분한다.
AxisEndpoints: TypeAlias = np.ndarray

# 축 끝점 경로의 관측면 예약어. 끝점 경로에서만 쓰고 윤곽 경로에는 쓰지 않는다
OBSERVED_FACE_AXIS = "axis"

# `PlanarObservation.method` 허용 값
_OBSERVATION_METHODS = ("opencv", "ai")

# AI 관측에서만 값을 갖는 조건부 필드
_AI_ONLY_FIELDS = ("class_name", "confidence", "model_id")


def _is_int(value) -> bool:
    """bool을 뺀 정수인지 확인한다."""
    return isinstance(value, int) and not isinstance(value, bool)


def _owned_contour(points, name: str) -> Contour | None:
    """윤곽을 소유 복사본(float32 (N, 2), 읽기 전용)으로 만든다. None은 그대로, 형태가 다르면 ValueError."""
    if points is None:
        return None

    # 외부 배열과 메모리를 나누지 않도록 항상 새로 복사한다
    owned = np.array(points, dtype=np.float32, copy=True, order="C")

    if owned.ndim != 2 or owned.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2): {owned.shape}")

    owned.setflags(write=False)

    return owned


def _finite_coordinate(value) -> float:
    """끝점 좌표 하나를 유한 float로 바꾼다. bool·문자열·복소수·None·변환 불가·비유한 값은 ValueError."""
    # bool은 정수의 하위형이라 숫자로 승격되기 전에 먼저 거른다
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"axis_endpoints_px coordinate must not be bool: {value!r}")

    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"axis_endpoints_px coordinate must be a real number: {value!r}")

    # 거대 정수는 float 변환에서 OverflowError가 난다 → ValueError로 통일
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"axis_endpoints_px coordinate is not representable as float64: {value!r}") from error

    if not math.isfinite(number):
        raise ValueError(f"axis_endpoints_px coordinate must be finite: {value!r}")

    return number


def _owned_endpoints(points) -> AxisEndpoints | None:
    """축 끝점을 소유 복사본(float64 (2, 2), 읽기 전용)으로 만든다. None은 그대로, 어긋나면 ValueError.

    NumPy 배열은 정수·실수 dtype만, 목록은 원소마다 검사한다(bool 승격·거대 정수 넘침 방지).
    두 점이 같으면(길이 0) 거부한다. 픽셀 범위는 프레임 크기를 아는 A 코어가 검사한다.
    """
    if points is None:
        return None

    # NumPy 배열은 변환 전에 dtype부터 본다: 정수·실수만 허용
    if isinstance(points, np.ndarray) and points.dtype.kind not in ("i", "u", "f"):
        raise ValueError(f"axis_endpoints_px array dtype must be integer or float: {points.dtype}")

    # 원래 원소를 보존한 object 배열로 형태를 먼저 본다(숫자 승격 전)
    try:
        cells = np.array(points, dtype=object)
    except (ValueError, TypeError) as error:
        raise ValueError(f"axis_endpoints_px must be a (2, 2) array: {error}") from error

    if cells.shape != (2, 2):
        raise ValueError(f"axis_endpoints_px must have shape (2, 2): {cells.shape}")

    owned = np.array([[_finite_coordinate(cells[r, c]) for c in range(2)] for r in range(2)], dtype=np.float64)

    if np.array_equal(owned[0], owned[1]):
        raise ValueError("axis_endpoints_px head and tip must differ")

    owned.setflags(write=False)

    return owned


def owned_axis_endpoints(points) -> AxisEndpoints | None:
    """`PlanarObservation`과 같은 규칙으로 축 끝점을 검사해 소유 복사본을 돌려준다(None은 None).

    A 코어가 생성 뒤 변조된 끝점을 다시 확인할 때 쓴다. 어긋나면 ValueError.
    """
    return _owned_endpoints(points)


def check_axis_path(*, observed_face, calc_contour, axis_endpoints_px, class_name, valid) -> AxisEndpoints | None:
    """축 끝점 경로 조합 규칙을 검사하고 끝점 소유 복사본을 돌려준다. 어긋나면 ValueError.

    끝점이 있으면 관측면 `axis`·클래스 None 또는 볼트. 관측면 `axis`이면 `calc_contour`는 None이고,
    유효(`valid is True`)이면 끝점 필수(무효 axis의 끝점 None은 허용).
    """
    endpoints = _owned_endpoints(axis_endpoints_px)
    is_axis = isinstance(observed_face, str) and observed_face == OBSERVED_FACE_AXIS

    if endpoints is not None:
        if not is_axis:
            raise ValueError(f"axis_endpoints_px requires observed_face={OBSERVED_FACE_AXIS!r}")

        if not _is_bolt_class(class_name):
            raise ValueError(f"axis_endpoints_px is only for bolts: class_name={class_name!r}")

    if is_axis and calc_contour is not None:
        raise ValueError("axis observation must not carry calc_contour")

    if is_axis and valid is True and endpoints is None:
        raise ValueError("valid axis observation requires axis_endpoints_px")

    return endpoints


def _is_bolt_class(class_name) -> bool:
    """축 끝점 경로에 쓸 수 있는 클래스인지: None(OpenCV·미지정) 또는 볼트 이름(소문자화 후 `bolt`로 시작)."""
    if class_name is None:
        return True

    return isinstance(class_name, str) and class_name.strip().lower().startswith("bolt")


@dataclass(frozen=True)
class PlanarObservation:
    """A 평면 관측 하나. 규칙을 어기면 ValueError.

    유효이면 `observed_face`·`selected_index`·기하(`calc_contour` 또는 axis 경로의 `axis_endpoints_px`)가 있고,
    무효이면 `reason`이 있다. 끝점 행 0 = 머리 끝, 행 1 = 축 끝. 기하는 읽기 전용 소유 복사본이다.
    """

    run_id: str
    source_id: str
    image_seq: int
    capture_generation: int
    generation: int
    job_seq: int
    detector_config_id: str
    pixel_basis: PixelBasis
    calibration_id: str | None
    observed_face: str | None
    calc_contour: Contour | None
    display_mask_contour: Contour | None
    candidate_count: int
    selected_index: int | None
    valid: bool
    reason: str | None
    process_ms: float
    method: str
    class_name: str | None
    confidence: float | None
    model_id: str | None
    axis_endpoints_px: AxisEndpoints | None = None

    def __post_init__(self) -> None:
        """열거 값 정규화·유효 조건 검사·윤곽·끝점 소유 복사를 한다. 어긋나면 ValueError."""
        put = lambda k, v: object.__setattr__(self, k, v)  # noqa: E731

        # 열거·플래그 값 검사
        put("pixel_basis", PixelBasis(self.pixel_basis))

        if self.method not in _OBSERVATION_METHODS:
            raise ValueError(f"method must be 'opencv' or 'ai': {self.method!r}")

        if not isinstance(self.valid, bool):
            raise ValueError(f"valid must be bool: {self.valid!r}")

        # 윤곽 소유 복사(형태 검사 포함)
        put("calc_contour", _owned_contour(self.calc_contour, "calc_contour"))
        put("display_mask_contour", _owned_contour(self.display_mask_contour, "display_mask_contour"))

        # 관측면이 없으면 계산용 윤곽도 없다(전체 실루엣 승격 방지)
        if self.observed_face is None and self.calc_contour is not None:
            raise ValueError("calc_contour requires observed_face")

        # 축 끝점 경로: 끝점 ⇔ 관측면 axis, 계산용 윤곽과 함께 쓰지 않음, 볼트(또는 미지정)만
        put("axis_endpoints_px", check_axis_path(
            observed_face=self.observed_face, calc_contour=self.calc_contour,
            axis_endpoints_px=self.axis_endpoints_px, class_name=self.class_name, valid=self.valid))
        is_axis = isinstance(self.observed_face, str) and self.observed_face == OBSERVED_FACE_AXIS

        # 유효 조건(축 경로는 calc_contour 대신 axis_endpoints_px)
        if self.valid:
            geometry = "axis_endpoints_px" if is_axis else "calc_contour"
            missing = [k for k in ("observed_face", geometry, "selected_index")
                       if getattr(self, k) is None]

            if missing:
                raise ValueError(f"valid observation requires {missing}")

            if self.reason is not None:
                raise ValueError("valid observation must not carry reason")

        # 무효 관측의 사유는 필수(이름 목록은 미결이라 형식만 검사)
        elif not isinstance(self.reason, str) or not self.reason:
            raise ValueError(f"invalid observation requires non-empty str reason: {self.reason!r}")

        # AI 전용 필드는 opencv 관측에서 None
        if self.method == "opencv":
            present = [k for k in _AI_ONLY_FIELDS if getattr(self, k) is not None]

            if present:
                raise ValueError(f"opencv observation must not carry {present}")


class InvalidationKind(str, Enum):
    """후보 무효화 사건의 종류. 이 5개 외의 값을 임의로 추가하지 않는다."""

    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    MODEL_CONFIG_INVALID = "MODEL_CONFIG_INVALID"
    INFERENCE_ERROR = "INFERENCE_ERROR"
    JOB_ABORTED = "JOB_ABORTED"
    GENERATION_CHANGED = "GENERATION_CHANGED"


# 프레임 없는 사건: job_seq·image_seq가 반드시 None
_FRAMELESS_KINDS = (InvalidationKind.GENERATION_CHANGED, InvalidationKind.MODEL_LOAD_FAILED)

# 작업에서 생긴 사건: job_seq·image_seq가 반드시 값
_JOB_KINDS = (InvalidationKind.INFERENCE_ERROR, InvalidationKind.JOB_ABORTED)


@dataclass(frozen=True)
class CandidateInvalidationEvent:
    """이동 후보 무효화 사건(가칭). 모든 필드에 기본값이 없고, 어긋나면 ValueError.

    - `GENERATION_CHANGED`·`MODEL_LOAD_FAILED`는 `job_seq`·`image_seq`가 None(`generation`은 새 세대).
    - `INFERENCE_ERROR`·`JOB_ABORTED`는 둘 다 값, `MODEL_CONFIG_INVALID`는 둘 다 None 또는 둘 다 값.
    - `event_monotonic_ns`는 PC 단조 시각이며 이 생성자는 시계를 읽지 않는다.
    """

    run_id: str
    source_id: str
    kind: InvalidationKind
    cause: str
    event_monotonic_ns: int
    generation: int
    detector_config_id: str
    job_seq: int | None
    image_seq: int | None

    def __post_init__(self) -> None:
        """필드 형식·kind 정규화·kind별 프레임 식별자 규칙을 검사한다. 어긋나면 ValueError."""
        # 필수 필드 형식
        for name in ("run_id", "source_id", "cause", "detector_config_id"):
            value = getattr(self, name)

            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string: {value!r}")

        for name in ("event_monotonic_ns", "generation"):
            if not _is_int(getattr(self, name)):
                raise ValueError(f"{name} must be an integer: {getattr(self, name)!r}")

        for name in ("job_seq", "image_seq"):
            value = getattr(self, name)

            if value is not None and not _is_int(value):
                raise ValueError(f"{name} must be an integer or None: {value!r}")

        kind = InvalidationKind(self.kind)
        object.__setattr__(self, "kind", kind)

        has_job = self.job_seq is not None
        has_frame = self.image_seq is not None

        if kind in _FRAMELESS_KINDS and (has_job or has_frame):
            raise ValueError(f"{kind.value} must not carry job_seq/image_seq")

        if kind in _JOB_KINDS and not (has_job and has_frame):
            raise ValueError(f"{kind.value} requires job_seq and image_seq")

        if has_job != has_frame:
            raise ValueError("job_seq and image_seq must be both set or both None")


# ---------------------------------------------------------------------------
# 촬영 정보·촬영 시각 TF와 A→B 다중 물체 목록
#
# 검사 보조 함수는 봉투(`eyetoarm.b_input/1`)와 목록이 같은 규칙을 쓰도록 한 곳에 두고 ValueError를 낸다.
# 카메라 프로파일은 B 모듈을 import하지 않고 속성으로 대조한다(공통형은 최소 의존).
# ---------------------------------------------------------------------------

FRAME_OBSERVATIONS_VERSION = "eyetoarm.frame_observations/1"

# 목록 후보의 관측면. 볼트 축 끝점 또는 너트 윗면 윤곽
OBSERVED_FACE_TOP = "top_face"
_OBJECT_FACES = (OBSERVED_FACE_AXIS, OBSERVED_FACE_TOP)


@dataclass(frozen=True)
class CaptureInfo:
    """촬영 정보: 원본 stamp(ns, SIM_TIME)·광학 프레임·보정 번호·CameraInfo 해상도."""

    stamp_ns: int
    optical_frame: str
    calibration_id: str
    width: int
    height: int


@dataclass(frozen=True)
class CaptureTf:
    """촬영 시각의 `parent → child`(world ← optical) 4×4. `T_W_C`는 읽기 전용 float64."""

    parent: str
    child: str
    stamp_ns: int
    T_W_C: np.ndarray


def nonneg_int(value, where: str) -> int:
    """bool이 아닌 0 이상 정수(NumPy 정수 포함). 아니면 ValueError."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{where} must be an integer >= 0: {value!r}")

    return int(value)


def real_number(value, where: str) -> float:
    """유한 실수 하나. bool·문자열·복소수·None 등은 거부하고, 거대 정수·비유한 값도 ValueError."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{where} must be a real number (not bool/str/complex): {value!r}")

    try:
        number = float(value)
    except (OverflowError, ValueError, TypeError) as exc:
        raise ValueError(f"{where} is not representable as float64: {value!r}") from exc

    if not math.isfinite(number):
        raise ValueError(f"{where} must be finite: {value!r}")

    return number


def real_array(value, where: str) -> np.ndarray:
    """실수 배열(float64 새 복사본). ndarray는 정수·실수 dtype만, 목록은 원소마다 `real_number`로 검사."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "iuf":
            raise ValueError(f"{where} must have an integer/float dtype: {value.dtype}")

        with np.errstate(over="ignore"):
            arr = value.astype(np.float64, copy=True)
    elif isinstance(value, (list, tuple)):
        def walk(v, path):
            """중첩 목록을 원소 검사하며 그대로 옮긴다."""
            if isinstance(v, (list, tuple)):
                return [walk(x, f"{path}[{i}]") for i, x in enumerate(v)]

            if isinstance(v, np.ndarray):
                return real_array(v, path).tolist()

            return real_number(v, path)

        try:
            arr = np.array(walk(value, where), dtype=np.float64)
        except ValueError as exc:   # 들쭉날쭉한 중첩
            raise ValueError(f"{where} must be a rectangular numeric array: {exc}") from exc
    else:
        raise ValueError(f"{where} must be a numeric array: {type(value).__name__}")

    if not np.isfinite(arr).all():
        raise ValueError(f"{where} must be finite")

    return arr


def nonempty_text(value, where: str) -> str:
    """비어 있지 않은 문자열. 아니면 ValueError."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a non-empty string: {value!r}")

    return value


def rigid_transform(T, where: str) -> np.ndarray:
    """4×4 유한 강체변환의 읽기 전용 복사본(원소 형 엄격 검사).

    허용오차: 마지막 행 [0,0,0,1] atol 1e-9, `R Rᵀ = I` atol 1e-6, det(R) = 1 abs 1e-6.
    """
    arr = real_array(T, where)

    R = arr[:3, :3] if arr.shape == (4, 4) else None

    if (R is None or not np.isfinite(arr).all() or not np.allclose(arr[3], [0, 0, 0, 1], atol=1e-9)
            or not np.allclose(R @ R.T, np.eye(3), atol=1e-6) or not math.isclose(float(np.linalg.det(R)), 1.0,
                                                                                   abs_tol=1e-6)):
        raise ValueError(f"{where} must be a finite 4x4 rigid transform")

    arr.setflags(write=False)

    return arr


def raw_points_in_frame(points: np.ndarray, width: int, height: int) -> bool:
    """RAW 픽셀 점이 모두 0 ≤ u ≤ width − 1, 0 ≤ v ≤ height − 1 안인가(목록·검출 변환 공통 규칙)."""
    return not ((points[:, 0] < 0).any() or (points[:, 0] > width - 1).any()
                or (points[:, 1] < 0).any() or (points[:, 1] > height - 1).any())


def check_frame_image(frame: FramePacket) -> None:
    """프레임 영상이 uint8 (H, W, 3)이고 frame.width·height와 맞는지. 아니면 ValueError."""
    img = frame.image

    if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3 \
            or img.shape[:2] != (frame.height, frame.width):
        raise ValueError(f"frame.image must be uint8 (H, W, 3) matching width/height: "
                         f"{getattr(img, 'shape', None)} vs {(frame.height, frame.width)}")


def check_frame_capture(frame: FramePacket, capture: CaptureInfo) -> None:
    """촬영 정보 필드와 프레임 ↔ 촬영 정보(시각·광학 프레임·크기·보정 번호)를 대조한다."""
    nonneg_int(capture.stamp_ns, "capture.stamp_ns")
    nonempty_text(capture.optical_frame, "capture.optical_frame")
    nonempty_text(capture.calibration_id, "capture.calibration_id")

    if frame.ros_stamp_ns != capture.stamp_ns:
        raise ValueError(f"frame.ros_stamp_ns {frame.ros_stamp_ns!r} != capture.stamp_ns {capture.stamp_ns}")

    if frame.optical_frame != capture.optical_frame:
        raise ValueError(f"frame.optical_frame {frame.optical_frame!r} != capture.optical_frame {capture.optical_frame!r}")

    if (frame.width, frame.height) != (capture.width, capture.height):
        raise ValueError(f"frame size {(frame.width, frame.height)} != capture size {(capture.width, capture.height)}")

    if frame.calibration_id != capture.calibration_id:
        raise ValueError(f"frame.calibration_id {frame.calibration_id!r} != capture {capture.calibration_id!r}")


_MISSING = object()


def check_camera_capture(camera, capture: CaptureInfo) -> None:
    """카메라 프로파일 ↔ 촬영 정보(보정 번호·광학 프레임·해상도)를 속성으로 대조한다."""
    got = tuple(getattr(camera, name, _MISSING) for name in ("calibration_id", "frame_id", "width", "height"))

    if got != (capture.calibration_id, capture.optical_frame, capture.width, capture.height):
        raise ValueError("camera profile (calibration_id/frame_id/size) does not match capture info")


def check_capture_tf(tf, capture: CaptureInfo, world_frame: str | None) -> CaptureTf | None:
    """촬영 시각 TF 결합을 대조하고 읽기 전용 복사본으로 정규화한다(None은 None).

    `world_frame`이 None이면 부모 프레임은 비어 있지 않은지만 본다.
    """
    if tf is None:
        return None

    if not isinstance(tf, CaptureTf):
        raise ValueError(f"tf must be CaptureTf or None: {type(tf).__name__}")

    if world_frame is None:
        nonempty_text(tf.parent, "tf.parent")
    elif tf.parent != world_frame:
        raise ValueError(f"tf.parent {tf.parent!r} != world frame {world_frame!r}")

    if tf.child != capture.optical_frame:
        raise ValueError(f"tf.child {tf.child!r} != capture.optical_frame {capture.optical_frame!r}")

    if nonneg_int(tf.stamp_ns, "tf.stamp_ns") != capture.stamp_ns:
        raise ValueError(f"tf.stamp_ns {tf.stamp_ns} != capture.stamp_ns {capture.stamp_ns}")

    return CaptureTf(tf.parent, tf.child, tf.stamp_ns, rigid_transform(tf.T_W_C, "tf.T_W_C"))


@dataclass(frozen=True)
class ObjectObservation:
    """같은 프레임의 물체 후보 하나. 좌표는 독립 복사·읽기 전용이다.

    `instance_index`는 그 프레임의 원래 검출 인덱스(추적 ID 아님). `observed_face`는 볼트 `"axis"`
    (끝점만) 또는 너트 등 `"top_face"`(윤곽만)로 클래스와 맞아야 한다. A 생성 함수는 볼트만 넣는다.
    RAW 픽셀 범위는 프레임 크기를 아는 `FrameObservations`가 검사한다.
    """

    instance_index: int
    class_name: str
    confidence: float
    pixel_basis: PixelBasis
    observed_face: str
    axis_endpoints_px: AxisEndpoints | None
    calc_contour: Contour | None

    def __post_init__(self) -> None:
        """필드를 검사하고 좌표를 독립 복사·읽기 전용으로 바꾼다. 어긋나면 ValueError."""
        put = lambda k, v: object.__setattr__(self, k, v)  # noqa: E731

        nonneg_int(self.instance_index, "instance_index")
        put("instance_index", int(self.instance_index))
        nonempty_text(self.class_name, "class_name")

        confidence = real_number(self.confidence, "confidence")

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1]: {self.confidence!r}")

        put("confidence", confidence)

        if not isinstance(self.pixel_basis, PixelBasis):
            raise ValueError(f"pixel_basis must be PixelBasis: {self.pixel_basis!r}")

        if not isinstance(self.observed_face, str) or self.observed_face not in _OBJECT_FACES:
            raise ValueError(f"observed_face must be one of {_OBJECT_FACES}: {self.observed_face!r}")

        bolt = _is_bolt_class(self.class_name)

        # 관측면에 맞는 기하 하나만(배타). 볼트 ↔ axis, 볼트 아님 ↔ top_face
        if self.observed_face == OBSERVED_FACE_AXIS:
            if not bolt:
                raise ValueError(f"observed_face 'axis' is only for bolts: class_name={self.class_name!r}")

            if self.calc_contour is not None:
                raise ValueError("axis candidate must not carry calc_contour")

            endpoints = _owned_endpoints(self.axis_endpoints_px)

            if endpoints is None:
                raise ValueError("axis candidate requires axis_endpoints_px")

            put("axis_endpoints_px", endpoints)
        else:
            if bolt:
                raise ValueError(f"observed_face 'top_face' is not for bolts: class_name={self.class_name!r}")

            if self.axis_endpoints_px is not None:
                raise ValueError("top_face candidate must not carry axis_endpoints_px")

            if self.calc_contour is None:
                raise ValueError("top_face candidate requires calc_contour")

            values = real_array(self.calc_contour, "calc_contour")

            # float64 유한 값도 float32로 넘치면 inf가 된다(변환 전에 거부)
            if (np.abs(values) > np.finfo(np.float32).max).any():
                raise ValueError("calc_contour must be finite in float32")

            contour = _owned_contour(values, "calc_contour")

            if contour.shape[0] < 3:
                raise ValueError(f"calc_contour needs at least 3 points: {contour.shape}")

            put("calc_contour", contour)

    @property
    def points(self) -> np.ndarray:
        """관측면의 좌표 배열(끝점 또는 윤곽)."""
        return self.axis_endpoints_px if self.observed_face == OBSERVED_FACE_AXIS else self.calc_contour


@dataclass(frozen=True)
class FrameObservations:
    """같은 프레임의 유효 물체 후보 목록(A→B). 생성 때 검사하고 어긋나면 ValueError.

    `frame.image`는 쓰기 금지 uint8, `tf`는 None 또는 촬영 시각·광학 프레임이 같은 강체,
    `candidates`는 1개 이상·`instance_index` 엄격 오름차순·RAW 좌표는 프레임 안.
    """

    version: str
    frame: FramePacket
    capture: CaptureInfo
    camera: object
    tf: CaptureTf | None
    generation: int
    job_seq: int
    detector_config_id: str
    candidates: tuple[ObjectObservation, ...]

    def __post_init__(self) -> None:
        """필드와 결합을 검사하고 TF를 독립 복사로 바꾼다. 어긋나면 ValueError."""
        if self.version != FRAME_OBSERVATIONS_VERSION:
            raise ValueError(f"unsupported frame observations version {self.version!r} "
                             f"(expected {FRAME_OBSERVATIONS_VERSION!r})")

        if not isinstance(self.frame, FramePacket):
            raise ValueError(f"frame must be FramePacket: {type(self.frame).__name__}")

        check_frame_image(self.frame)

        if self.frame.image.flags.writeable:
            raise ValueError("frame.image must be read-only (owned by the input source)")

        if not isinstance(self.capture, CaptureInfo):
            raise ValueError(f"capture must be CaptureInfo: {type(self.capture).__name__}")

        check_frame_capture(self.frame, self.capture)
        check_camera_capture(self.camera, self.capture)
        object.__setattr__(self, "tf", check_capture_tf(self.tf, self.capture, None))

        nonneg_int(self.generation, "generation")
        nonneg_int(self.job_seq, "job_seq")
        nonempty_text(self.detector_config_id, "detector_config_id")

        cands = self.candidates

        if not isinstance(cands, tuple) or not cands:
            raise ValueError("candidates must be a non-empty tuple of ObjectObservation")

        previous = -1

        for c in cands:
            if not isinstance(c, ObjectObservation):
                raise ValueError(f"candidates must contain ObjectObservation: {type(c).__name__}")

            if c.instance_index <= previous:
                raise ValueError(f"candidate instance_index must be strictly increasing: {[x.instance_index for x in cands]}")

            previous = c.instance_index

            # RAW만 범위를 본다(undistorted 범위 규칙은 아직 없음)
            if c.pixel_basis is PixelBasis.RAW:
                if not raw_points_in_frame(c.points, self.capture.width, self.capture.height):
                    raise ValueError(f"candidate {c.instance_index} raw coordinates out of frame "
                                     f"{(self.capture.width, self.capture.height)}")

    @property
    def instance_indices(self) -> tuple[int, ...]:
        """목록에 실린 원 인덱스(오름차순)."""
        return tuple(c.instance_index for c in self.candidates)
