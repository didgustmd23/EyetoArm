"""저장 입력 녹화 메타(#168): 원본 CameraInfo·프레임별 촬영 시각·그 시각의 `world → optical_frame` TF.

근거: `_workspace/168_a_to_b_real_path_implementation_spec.md` §2, #167 계획 r2 W2, B 협업 요청 §8.2 A-7(TF 50 ms·최신 TF 대체 금지),
`transforms/b_stage.py`(`RosTfLookup` 시각 의미), `calibration/profile.from_camera_info`(실시간과 같은 보정 번호 계산).

이 모듈의 역할과 한계
- **관측은 다루지 않는다.** 관측은 실제 AI 검출과 끝점 추출에서 나온다(#166 합성 fixture와 다른 점). 메타는 저장 영상에 없는 촬영
  기하(카메라 내부값·촬영 시각·카메라 자세)만 준다.
- `CaptureMetaSource`는 저장 입력 프레임에 메타와 **모두 일치할 때만** `calibration_id`·`optical_frame`·`ros_stamp_ns`를 붙인다.
  메타 없음·해상도·frame id·TF 시각/자식 프레임 불일치 프레임은 붙이지 않고 그대로 넘긴다. 이런 프레임은 코어·`b_gate`가 B로 보내지
  않으며(`calibration_id` 없음 → 프로파일 미적용), 불일치 사유는 `mismatches`에 남는다. 임의 보정·임의 고정 TF를 붙이지 않는다.
- `CaptureMetaTfLookup`은 프레임의 메타 레코드 TF를 **촬영 시각·자식 프레임이 정확히 같을 때만** 돌려준다. 그 시각 TF가 녹화되지
  않았으면(`tf: null`) None이다(B는 `SKIP/TF_UNAVAILABLE`). 다른 프레임·최신 TF로 대체하지 않는다. 조회가 50 ms를 넘기면 None이다
  (반환 뒤 판정).
- 실시간 링크 카메라 TF(`RosTfLookup`·Buffer 소유)는 이 모듈 범위가 아니다(#167 K-T 미결).
- ROS·Qt를 import하지 않는다.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable

import numpy as np

from eyetoarm.calibration.profile import CameraProfile, from_camera_info
from eyetoarm.common.types import FramePacket
from eyetoarm.transforms.planar_transform import make_T, quat_to_rot
from eyetoarm.vision.pipeline_types import SourceProfile

CAPTURE_META_FORMAT = "eyetoarm.capture_meta/1"
TF_WAIT_LIMIT_S = 0.05   # B 시간 상한: TF 대기 50 ms (B 협업 요청 §8.2 A-7)

_TOP_KEYS = {"capture_meta_format", "capture_id", "source_id", "world_frame", "inputs_sha256", "camera_info", "frames",
             "provenance"}
_CAMERA_KEYS = {"width", "height", "k", "d", "distortion_model", "frame_id"}
_FRAME_KEYS = {"source_index", "stamp_ns", "frame_id", "width", "height", "tf"}
_TF_KEYS = {"parent", "child", "stamp_ns", "translation_m", "rotation_xyzw"}


class CaptureMetaError(ValueError):
    """메타 형식·내용 오류. 실행 전에 종료 2로 끝낸다."""


def _reject_duplicates(pairs):
    """중복 키 거부."""
    out = {}

    for key, value in pairs:
        if key in out:
            raise CaptureMetaError(f"duplicate key: {key!r}")

        out[key] = value

    return out


def _reject_constant(name):
    """NaN·Infinity 거부."""
    raise CaptureMetaError(f"non-finite constant: {name}")


def _parse_float(text):
    """비유한 실수 거부."""
    value = float(text)

    if not math.isfinite(value):
        raise CaptureMetaError(f"non-finite number: {text}")

    return value


def _keys(obj, allowed: set, where: str) -> None:
    """객체이고 키가 정확히 `allowed`인지."""
    if not isinstance(obj, dict):
        raise CaptureMetaError(f"{where} must be an object")

    missing, unknown = allowed - obj.keys(), obj.keys() - allowed

    if missing or unknown:
        raise CaptureMetaError(f"{where} keys mismatch: missing={sorted(missing)} unknown={sorted(unknown)}")


def _int(value, where: str, minimum: int = 0) -> int:
    """bool이 아닌 정수(최솟값 이상)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CaptureMetaError(f"{where} must be an integer >= {minimum}: {value!r}")

    return value


def _text(value, where: str) -> str:
    """비어 있지 않은 문자열."""
    if not isinstance(value, str) or not value:
        raise CaptureMetaError(f"{where} must be a non-empty string: {value!r}")

    return value


def _reals(value, n: int | None, where: str) -> list[float]:
    """bool이 아닌 유한 실수 목록(길이 n, None이면 자유)."""
    if not isinstance(value, list) or (n is not None and len(value) != n) or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
        raise CaptureMetaError(f"{where} must be a list of {n if n is not None else 'N'} real numbers")

    try:
        out = [float(v) for v in value]
    except OverflowError as exc:
        raise CaptureMetaError(f"{where} has a number out of float range") from exc

    if not all(math.isfinite(v) for v in out):
        raise CaptureMetaError(f"{where} must be finite")

    return out


@dataclass(frozen=True)
class FrameMeta:
    """입력 프레임 하나의 촬영 메타."""

    source_index: int
    stamp_ns: int
    frame_id: str
    width: int
    height: int
    tf_child: str | None
    tf_stamp_ns: int | None
    T_W_C: np.ndarray | None   # world ← optical (B `T_W_C`), 그 시각 TF가 녹화되지 않았으면 None


@dataclass(frozen=True)
class CaptureMeta:
    """검사를 마친 녹화 메타."""

    path: str
    sha256: str
    capture_id: str
    source_profile: SourceProfile
    camera: CameraProfile
    world_frame: str
    inputs_sha256: tuple[str, ...]
    frames: dict[int, FrameMeta]
    provenance: dict


def load_capture_meta(path: str) -> CaptureMeta:
    """메타 JSON을 읽어 검사한다. 형식·내용 오류는 CaptureMetaError, 파일 오류는 OSError."""
    with open(path, "rb") as f:
        data = f.read()

    try:
        doc = json.loads(data.decode("utf-8"), object_pairs_hook=_reject_duplicates,
                         parse_constant=_reject_constant, parse_float=_parse_float)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CaptureMetaError(f"invalid JSON: {exc}") from exc

    _keys(doc, _TOP_KEYS, "capture_meta")

    if doc["capture_meta_format"] != CAPTURE_META_FORMAT:
        raise CaptureMetaError(f"capture_meta_format must be {CAPTURE_META_FORMAT!r}")

    inputs = doc["inputs_sha256"]

    if not isinstance(inputs, list) or not inputs or any(
            not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in inputs):
        raise CaptureMetaError("inputs_sha256 must be a non-empty list of lowercase sha256 hex")

    if not isinstance(doc["provenance"], dict):
        raise CaptureMetaError("provenance must be an object")

    # 원본 CameraInfo → 실시간과 같은 경로(from_camera_info)로 CameraProfile·보정 번호
    cam = doc["camera_info"]
    _keys(cam, _CAMERA_KEYS, "camera_info")

    # B CameraProfile은 plumb_bob 5계수만 안다. 왜곡이 0이 아닌데 다른 모델이면 거부한다
    if not isinstance(cam["distortion_model"], str) or not isinstance(cam["d"], list):
        raise CaptureMetaError("camera_info.distortion_model must be a string and d a list")

    if cam["distortion_model"] != "plumb_bob" and any(cam["d"]):
        raise CaptureMetaError("only plumb_bob distortion is supported by B (CameraProfile k1 k2 p1 p2 k3)")

    msg = SimpleNamespace(k=_reals(cam["k"], 9, "camera_info.k"), d=_reals(cam["d"], None, "camera_info.d"),
                          width=_int(cam["width"], "camera_info.width", 1), height=_int(cam["height"], "camera_info.height", 1),
                          header=SimpleNamespace(frame_id=_text(cam["frame_id"], "camera_info.frame_id")))

    if len(msg.d) not in (0, 5):
        raise CaptureMetaError("camera_info.d must have 0 or 5 coefficients")

    try:
        camera = from_camera_info(msg)
    except ValueError as exc:
        raise CaptureMetaError(f"camera_info invalid: {exc}") from exc

    source_id = _text(doc["source_id"], "source_id")
    profile = SourceProfile(source_id=source_id, width=camera.width, height=camera.height, crop=None,
                            calibration_id=camera.calibration_id)
    world = _text(doc["world_frame"], "world_frame")

    # 프레임별 메타: source_index 중복 금지
    if not isinstance(doc["frames"], list) or not doc["frames"]:
        raise CaptureMetaError("frames must be a non-empty list")

    frames: dict[int, FrameMeta] = {}

    for i, raw in enumerate(doc["frames"]):
        where = f"frames[{i}]"
        _keys(raw, _FRAME_KEYS, where)
        index = _int(raw["source_index"], f"{where}.source_index")

        if index in frames:
            raise CaptureMetaError(f"{where}: duplicate source_index {index}")

        tf = raw["tf"]
        child = tf_stamp = T = None

        if tf is not None:
            _keys(tf, _TF_KEYS, f"{where}.tf")

            if tf["parent"] != world:
                raise CaptureMetaError(f"{where}.tf.parent {tf['parent']!r} != world_frame {world!r}")

            child = _text(tf["child"], f"{where}.tf.child")
            tf_stamp = _int(tf["stamp_ns"], f"{where}.tf.stamp_ns")
            t = _reals(tf["translation_m"], 3, f"{where}.tf.translation_m")
            q = _reals(tf["rotation_xyzw"], 4, f"{where}.tf.rotation_xyzw")
            norm = math.hypot(*q)

            if not math.isclose(norm, 1.0, abs_tol=1e-4):
                raise CaptureMetaError(f"{where}.tf.rotation_xyzw must be a unit quaternion (|q|={norm})")

            T = make_T(quat_to_rot(*(v / norm for v in q)), t)
            T.setflags(write=False)

        frames[index] = FrameMeta(source_index=index, stamp_ns=_int(raw["stamp_ns"], f"{where}.stamp_ns"),
                                  frame_id=_text(raw["frame_id"], f"{where}.frame_id"),
                                  width=_int(raw["width"], f"{where}.width", 1),
                                  height=_int(raw["height"], f"{where}.height", 1),
                                  tf_child=child, tf_stamp_ns=tf_stamp, T_W_C=T)

    return CaptureMeta(path=str(path), sha256=hashlib.sha256(data).hexdigest(),
                       capture_id=_text(doc["capture_id"], "capture_id"), source_profile=profile, camera=camera,
                       world_frame=world, inputs_sha256=tuple(inputs), frames=frames, provenance=dict(doc["provenance"]))


def frame_mismatch(meta: CaptureMeta, frame: FramePacket) -> str | None:
    """프레임과 메타가 어긋난 첫 사유(없으면 None). 이 사유가 있는 프레임에는 촬영 메타를 붙이지 않는다."""
    record = meta.frames.get(getattr(frame, "source_index", -1))

    if record is None:
        return "META_MISSING"

    if (frame.width, frame.height) != (record.width, record.height) or \
            (frame.width, frame.height) != (meta.camera.width, meta.camera.height):
        return "META_SIZE_MISMATCH"

    if record.frame_id != meta.camera.frame_id:
        return "META_FRAME_ID_MISMATCH"

    if record.tf_child is not None and (record.tf_child != record.frame_id or record.tf_stamp_ns != record.stamp_ns):
        return "META_TF_STAMP_MISMATCH"

    return None


class CaptureMetaSource:
    """저장 입력 소스를 감싸 메타와 일치하는 프레임에만 촬영 메타를 붙인다(`read`·`current`·`close`만, 실시간 표지 없음)."""

    def __init__(self, inner, meta: CaptureMeta) -> None:
        """감쌀 소스와 메타."""
        self._inner = inner
        self._meta = meta
        self.mismatches: list[dict] = []   # 메타를 붙이지 않은 프레임과 사유

    def read(self):
        """다음 항목. 일치하는 프레임이면 `calibration_id`·`optical_frame`·`ros_stamp_ns`를 붙인 새 프레임(크기·순번·세대 그대로)."""
        item = self._inner.read()

        if not isinstance(item, FramePacket):
            return item

        reason = frame_mismatch(self._meta, item)

        if reason is not None:
            self.mismatches.append({"image_seq": item.image_seq, "source_index": getattr(item, "source_index", None),
                                    "reason": reason})
            return item

        record = self._meta.frames[item.source_index]

        return dataclasses.replace(item, calibration_id=self._meta.camera.calibration_id, optical_frame=record.frame_id,
                                   ros_stamp_ns=record.stamp_ns)

    def current(self):
        """감싼 소스의 현재 프레임."""
        return self._inner.current()

    def close(self) -> None:
        """감싼 소스를 닫는다."""
        self._inner.close()


class CaptureMetaTfLookup:
    """메타 TF 조회: 프레임 레코드의 TF를 촬영 시각·자식 프레임이 정확히 같을 때만, 50 ms 안에 돌려준다."""

    def __init__(self, meta: CaptureMeta, *, timeout_s: float = TF_WAIT_LIMIT_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        """메타와 상한."""
        self.timeout_s = timeout_s
        self._clock = clock
        self.calls = 0
        self.unavailable = 0
        self.timed_out = 0
        self.wait_ms: list[float] = []

        def at_capture_time(frame):
            """그 프레임 촬영 시각의 world → optical. 시각·프레임이 다르거나 녹화되지 않았으면 None."""
            record = meta.frames.get(getattr(frame, "source_index", -1))

            if (record is None or record.T_W_C is None or frame.ros_stamp_ns is None
                    or frame.ros_stamp_ns != record.tf_stamp_ns or frame.optical_frame != record.tf_child):
                return None

            return record.T_W_C

        self.inner = at_capture_time

    def __call__(self, frame) -> np.ndarray | None:
        """조회. 상한 초과·없음은 None."""
        self.calls += 1
        t0 = self._clock()
        T = self.inner(frame)
        waited = self._clock() - t0
        self.wait_ms.append(waited * 1000.0)

        if waited > self.timeout_s:
            self.timed_out += 1
            self.unavailable += 1
            return None

        if T is None:
            self.unavailable += 1

        return T


__all__ = ["CAPTURE_META_FORMAT", "CaptureMeta", "CaptureMetaError", "CaptureMetaSource", "CaptureMetaTfLookup", "check_capture_inputs",
           "FrameMeta", "frame_mismatch", "load_capture_meta"]


def _file_sha256(path: str) -> str:
    """입력 파일 지문."""
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    return h.hexdigest()


def check_capture_inputs(meta: CaptureMeta, paths: list[str], source_id: str) -> list[str]:
    """입력 파일 순서·지문과 소스 ID가 메타와 같은지 확인하고 지문 목록을 돌려준다. 다르면 CaptureMetaError."""
    if source_id != meta.source_profile.source_id:
        raise CaptureMetaError(f"--source-id {source_id!r} != capture meta source_id {meta.source_profile.source_id!r}")

    actual = [_file_sha256(p) for p in paths]

    if tuple(actual) != meta.inputs_sha256:
        raise CaptureMetaError("input files (order/sha256) do not match capture meta inputs_sha256")

    return actual
