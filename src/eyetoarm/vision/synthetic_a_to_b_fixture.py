"""A→B 저장 입력 **시험 전용** 연결 fixture(#166). 합성 관측을 실제 실행기 `run()` → `b_gate` → `b_consumer`로 흘린다.

근거: `_workspace/166_a_b_pipeline_connection_spec.md` §2~§5, B 협업 요청 §8.2 A-7·A-8, A 상세설계 §7.1·D13,
`transforms/b_stage.py`(B 입구 `BStage.consume`, TF 대기 50 ms·계산 20 ms).

이 모듈이 하는 일과 하지 않는 일
- **synthetic/test-only.** fixture 관측은 사람이 정한 합성 값이다. 실제 RGB 검출·끝점 추출·mm 정확도·실시간 B 허용의
  증거가 아니다. 제품 AI/OpenCV 생산자(`process_opencv`·`AiSegDetector`)와 `b_gate`·공통형 유효 조건은 바꾸지 않는다.
- 관측 주입 경계는 실행기의 **detector 자리**(`run(detector=FixtureDetector)`)다. 실행기가 작업을 시작해 넘긴 `JobContext`마다
  한 번 불리고, 반환한 관측은 코어 `complete()`의 식별자·끝점 범위 검사 → 실행기 `handle()`의 `b_gate` → `b_consumer` 순서를
  그대로 지난다. B를 직접 부르지 않는다.
- 저장 입력 프레임에는 보정 번호·광학 프레임·촬영 시각이 없다. `FixtureStampedSource`가 fixture에 명시된 값만 프레임에 붙인다
  (`calibration_id`·`optical_frame`·`ros_stamp_ns`). 크기·crop·순번·세대는 손대지 않는다(불일치는 `b_gate`가 거른다).
- TF: 고정 카메라(`camera_motion="fixed"`)만 `StaticTfLookup`. 움직이는 카메라는 프레임별 촬영 시각 TF 표(`per_frame`)를
  **정확한 시각 일치**로만 읽는다(가장 가까운·최신 TF로 대체하지 않음). 대기 상한 50 ms를 넘긴 조회는 None(B `TF_UNAVAILABLE`).
- 재처리 작업·이미 쓴 레코드·대응 레코드가 없는 프레임에는 관측을 만들지 않고 `FixtureError`를 올린다(실행기가
  `INFERENCE_ERROR`로 정리, B 호출 없음).
- ROS·Qt·모델을 import하지 않는다. 출력 파일은 덮어쓰지 않는다(`open(..., "x")`).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from eyetoarm.calibration.profile import CameraProfile
from eyetoarm.common.types import OBSERVED_FACE_AXIS, FramePacket, PixelBasis, PlanarObservation
from eyetoarm.transforms.b_stage import BOutput, StaticTfLookup
from eyetoarm.vision.ai_config import AiSegConfig
from eyetoarm.vision.pipeline_types import JobContext, SourceProfile

FIXTURE_FORMAT = "eyetoarm.b_test_fixture/1"
TEST_MODE_LABEL = "synthetic/test-only"
TF_WAIT_LIMIT_S = 0.05   # B 시간 상한: TF 대기 50 ms (B 협업 요청 §8.2 A-7)

_TOP_KEYS = {"fixture_format", "fixture_id", "test_only", "run_id", "inputs_sha256", "source", "camera",
             "camera_motion", "tf", "records"}
_SOURCE_KEYS = {"source_id", "width", "height", "crop", "calibration_id", "pixel_basis"}
_CAMERA_KEYS = {"width", "height", "K", "dist", "frame_id"}
_RECORD_KEYS = {"source_index", "image_seq", "ros_stamp_ns", "T_W_C", "observation"}
_OBS_KEYS = {"run_id", "source_id", "image_seq", "capture_generation", "generation", "job_seq",
             "detector_config_id", "calibration_id", "pixel_basis", "class_name", "selected_index",
             "candidate_count", "valid", "reason", "observed_face", "axis_endpoints_px", "calc_contour"}


class FixtureError(ValueError):
    """fixture 형식·내용 오류 또는 실행 중 fixture 대응 실패. B를 부르지 않고 시험 실패로 끝낸다."""


# ---------- JSON 읽기(중복 키·NaN 거부) ----------
def _reject_duplicates(pairs):
    """중복 키가 있으면 FixtureError."""
    out = {}

    for key, value in pairs:
        if key in out:
            raise FixtureError(f"duplicate key: {key!r}")

        out[key] = value

    return out


def _reject_constant(name):
    """NaN·Infinity 상수 거부."""
    raise FixtureError(f"non-finite constant: {name}")


def _parse_float(text):
    """비유한 실수(1e400 등) 거부."""
    value = float(text)

    if not math.isfinite(value):
        raise FixtureError(f"non-finite number: {text}")

    return value


def _keys(obj, allowed: set, where: str) -> None:
    """객체이고 키가 정확히 `allowed`인지 확인한다."""
    if not isinstance(obj, dict):
        raise FixtureError(f"{where} must be an object")

    missing, unknown = allowed - obj.keys(), obj.keys() - allowed

    if missing or unknown:
        raise FixtureError(f"{where} keys mismatch: missing={sorted(missing)} unknown={sorted(unknown)}")


def _int(value, where: str, *, minimum: int | None = 0, allow_none: bool = False):
    """bool이 아닌 정수(기본 0 이상)."""
    if value is None and allow_none:
        return None

    if isinstance(value, bool) or not isinstance(value, int) or (minimum is not None and value < minimum):
        raise FixtureError(f"{where} must be an integer >= {minimum}: {value!r}")

    return value


def _text(value, where: str, *, allow_none: bool = False):
    """비어 있지 않은 문자열."""
    if value is None and allow_none:
        return None

    if not isinstance(value, str) or not value:
        raise FixtureError(f"{where} must be a non-empty string: {value!r}")

    return value


def _matrix(value, shape: tuple[int, ...], where: str) -> np.ndarray:
    """실수 행렬(bool 금지)을 float64로."""
    def check(v):
        """원소 하나가 bool이 아닌 실수인지."""
        if isinstance(v, list):
            for x in v:
                check(x)
        elif isinstance(v, bool) or not isinstance(v, (int, float)):
            raise FixtureError(f"{where} must contain real numbers: {v!r}")

    check(value)

    try:
        arr = np.array(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FixtureError(f"{where} is not a numeric array: {exc}") from exc

    if arr.shape != shape or not np.isfinite(arr).all():
        raise FixtureError(f"{where} must be finite with shape {shape}: {arr.shape}")

    return arr


def _rigid(value, where: str) -> np.ndarray:
    """4×4 강체변환(회전 직교·det 1, 마지막 행 0 0 0 1)."""
    T = _matrix(value, (4, 4), where)
    R = T[:3, :3]

    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-9) or not np.allclose(R @ R.T, np.eye(3), atol=1e-6) \
            or not math.isclose(float(np.linalg.det(R)), 1.0, abs_tol=1e-6):
        raise FixtureError(f"{where} must be a rigid transform")

    return T


# ---------- fixture 자료 ----------
@dataclass(frozen=True)
class FixtureRecord:
    """입력 프레임 하나에 귀속된 합성 관측 레코드."""

    source_index: int
    image_seq: int
    ros_stamp_ns: int | None
    T_W_C: np.ndarray | None
    observation: dict   # PlanarObservation 생성 인자(식별자는 fixture 값 그대로, 실행 중 값으로 덮지 않음)


@dataclass(frozen=True)
class SyntheticAToBFixture:
    """검사를 마친 fixture 전체."""

    path: str
    sha256: str
    fixture_id: str
    run_id: str
    inputs_sha256: tuple[str, ...]
    source_profile: SourceProfile
    pixel_basis: PixelBasis
    camera: CameraProfile
    camera_motion: str
    tf_mode: str
    static_T_W_C: np.ndarray | None
    records: tuple[FixtureRecord, ...]

    @property
    def expected_b_calls(self) -> int:
        """`b_gate`를 통과해야 하는 레코드 수(valid=True 레코드 수). 실행 뒤 실제 B 호출 수와 비교한다."""
        return sum(1 for r in self.records if r.observation["valid"])

    def record_for_index(self, source_index: int) -> FixtureRecord | None:
        """소스 위치의 레코드(없으면 None)."""
        return next((r for r in self.records if r.source_index == source_index), None)


def _check_point(point, width: int, height: int, where: str, basis: PixelBasis) -> list[float]:
    """픽셀 한 점: 실수 2개, raw면 0 ≤ u ≤ W−1, 0 ≤ v ≤ H−1."""
    arr = _matrix(point, (2,), where)

    if basis is PixelBasis.RAW and not (0 <= arr[0] <= width - 1 and 0 <= arr[1] <= height - 1):
        raise FixtureError(f"{where} out of frame {width}x{height}: {arr.tolist()}")

    return arr.tolist()


def _observation(raw, source: SourceProfile, basis: PixelBasis, where: str) -> dict:
    """관측 레코드를 검사해 `PlanarObservation` 생성 인자로 바꾼다(method·confidence·model_id 등은 실행 때 붙인다)."""
    _keys(raw, _OBS_KEYS, where)
    obs = dict(raw)

    for k in ("run_id", "source_id", "detector_config_id"):
        _text(obs[k], f"{where}.{k}")

    for k in ("image_seq", "capture_generation", "generation", "job_seq", "candidate_count"):
        _int(obs[k], f"{where}.{k}")

    _text(obs["calibration_id"], f"{where}.calibration_id", allow_none=True)
    _int(obs["selected_index"], f"{where}.selected_index", allow_none=True)
    _text(obs["class_name"], f"{where}.class_name", allow_none=True)
    _text(obs["observed_face"], f"{where}.observed_face", allow_none=True)
    _text(obs["reason"], f"{where}.reason", allow_none=True)

    if not isinstance(obs["valid"], bool):
        raise FixtureError(f"{where}.valid must be bool")

    if obs["pixel_basis"] != basis.value:
        raise FixtureError(f"{where}.pixel_basis {obs['pixel_basis']!r} != source.pixel_basis {basis.value!r}")

    obs["pixel_basis"] = basis
    face, endpoints, contour = obs["observed_face"], obs["axis_endpoints_px"], obs["calc_contour"]

    # 볼트 끝점: 순서가 이름으로 드러나는 {"head", "tip"}만 받는다(순서 모를 2×2 목록은 거부)
    if endpoints is not None:
        _keys(endpoints, {"head", "tip"}, f"{where}.axis_endpoints_px")
        head = _check_point(endpoints["head"], source.width, source.height, f"{where}.axis_endpoints_px.head", basis)
        tip = _check_point(endpoints["tip"], source.width, source.height, f"{where}.axis_endpoints_px.tip", basis)
        obs["axis_endpoints_px"] = np.array([head, tip], dtype=np.float64)

    # 너트 윤곽: top_face 윗면 바깥 테두리만(전체 실루엣 대용 금지 — 관측면 top_face가 아니면 거부)
    if contour is not None:
        if face != "top_face":
            raise FixtureError(f"{where}.calc_contour is only for observed_face='top_face': {face!r}")

        if not isinstance(contour, list) or len(contour) < 3:
            raise FixtureError(f"{where}.calc_contour needs >= 3 points")

        pts = [_check_point(p, source.width, source.height, f"{where}.calc_contour[{i}]", basis)
               for i, p in enumerate(contour)]
        obs["calc_contour"] = np.array(pts, dtype=np.float32)

    if face is not None and face not in (OBSERVED_FACE_AXIS, "top_face"):
        raise FixtureError(f"{where}.observed_face must be 'axis', 'top_face' or null: {face!r}")

    return obs


def load_fixture(path: str) -> SyntheticAToBFixture:
    """fixture JSON을 읽어 검사한다. 어긋나면 FixtureError(형식·내용), 파일 오류는 OSError."""
    with open(path, "rb") as f:
        data = f.read()

    try:
        doc = json.loads(data.decode("utf-8"), object_pairs_hook=_reject_duplicates,
                         parse_constant=_reject_constant, parse_float=_parse_float)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FixtureError(f"invalid JSON: {exc}") from exc

    _keys(doc, _TOP_KEYS, "fixture")

    if doc["fixture_format"] != FIXTURE_FORMAT:
        raise FixtureError(f"fixture_format must be {FIXTURE_FORMAT!r}")

    if doc["test_only"] is not True:
        raise FixtureError("test_only must be true (synthetic fixture only)")

    fixture_id, run_id = _text(doc["fixture_id"], "fixture_id"), _text(doc["run_id"], "run_id")
    inputs = doc["inputs_sha256"]

    if not isinstance(inputs, list) or not inputs or any(
            not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in inputs):
        raise FixtureError("inputs_sha256 must be a non-empty list of lowercase sha256 hex")

    # 소스 적용 조건(SourceProfile)
    src = doc["source"]
    _keys(src, _SOURCE_KEYS, "source")
    crop = src["crop"]

    if crop is not None:
        if not isinstance(crop, list) or len(crop) != 4:
            raise FixtureError("source.crop must be null or [x, y, w, h]")

        crop = tuple(_int(v, "source.crop") for v in crop)

    try:
        basis = PixelBasis(src["pixel_basis"])
    except ValueError as exc:
        raise FixtureError(f"source.pixel_basis invalid: {src['pixel_basis']!r}") from exc

    profile = SourceProfile(source_id=_text(src["source_id"], "source.source_id"),
                            width=_int(src["width"], "source.width", minimum=1),
                            height=_int(src["height"], "source.height", minimum=1), crop=crop,
                            calibration_id=_text(src["calibration_id"], "source.calibration_id"))

    # 카메라 프로파일(B 입력). 보정 번호는 K·dist·해상도에서 계산되며 source와 같아야 한다
    cam = doc["camera"]
    _keys(cam, _CAMERA_KEYS, "camera")

    try:
        camera = CameraProfile(_int(cam["width"], "camera.width", minimum=1),
                               _int(cam["height"], "camera.height", minimum=1),
                               _matrix(cam["K"], (3, 3), "camera.K"), _matrix(cam["dist"], (5,), "camera.dist"),
                               _text(cam["frame_id"], "camera.frame_id"))
    except ValueError as exc:
        raise FixtureError(f"camera invalid: {exc}") from exc

    if (camera.width, camera.height) != (profile.width, profile.height):
        raise FixtureError("camera width/height must equal source width/height")

    if camera.calibration_id != profile.calibration_id:
        raise FixtureError(f"source.calibration_id {profile.calibration_id!r} != camera calibration_id "
                           f"{camera.calibration_id!r}")

    # TF: 고정 카메라만 정적 자세. 움직이는 카메라는 프레임별 촬영 시각 TF
    motion, tf = doc["camera_motion"], doc["tf"]

    if motion not in ("fixed", "moving"):
        raise FixtureError("camera_motion must be 'fixed' or 'moving'")

    if not isinstance(tf, dict) or tf.get("mode") not in ("static", "per_frame"):
        raise FixtureError("tf.mode must be 'static' or 'per_frame'")

    static_T = None

    if tf["mode"] == "static":
        _keys(tf, {"mode", "T_W_C"}, "tf")

        if motion != "fixed":
            raise FixtureError("tf.mode 'static' is only allowed for camera_motion 'fixed'")

        static_T = _rigid(tf["T_W_C"], "tf.T_W_C")
    else:
        _keys(tf, {"mode"}, "tf")

    # 레코드: 프레임 키(source_index·image_seq) 중복 금지
    if not isinstance(doc["records"], list) or not doc["records"]:
        raise FixtureError("records must be a non-empty list")

    records, seen_index, seen_seq, seen_stamp = [], set(), set(), set()

    for i, raw in enumerate(doc["records"]):
        where = f"records[{i}]"
        _keys(raw, _RECORD_KEYS, where)
        index, seq = _int(raw["source_index"], f"{where}.source_index"), _int(raw["image_seq"], f"{where}.image_seq")

        if index in seen_index or seq in seen_seq:
            raise FixtureError(f"{where}: duplicate frame key source_index={index} image_seq={seq}")

        seen_index.add(index)
        seen_seq.add(seq)
        stamp = _int(raw["ros_stamp_ns"], f"{where}.ros_stamp_ns", allow_none=True)
        T = None

        if tf["mode"] == "per_frame":
            # 촬영 시각은 필수. 그 시각 TF가 저장되지 않았으면 T_W_C null(B는 TF_UNAVAILABLE로 SKIP, 다른 시각 TF로 대체 안 함)
            if stamp is None:
                raise FixtureError(f"{where}: per_frame TF needs ros_stamp_ns (capture time)")

            if stamp in seen_stamp:
                raise FixtureError(f"{where}: duplicate ros_stamp_ns {stamp}")

            seen_stamp.add(stamp)
            T = None if raw["T_W_C"] is None else _rigid(raw["T_W_C"], f"{where}.T_W_C")
        elif raw["T_W_C"] is not None:
            raise FixtureError(f"{where}.T_W_C must be null with tf.mode 'static'")

        records.append(FixtureRecord(index, seq, stamp, T, _observation(raw["observation"], profile, basis,
                                                                        f"{where}.observation")))

    return SyntheticAToBFixture(path=str(path), sha256=hashlib.sha256(data).hexdigest(), fixture_id=fixture_id, run_id=run_id,
                        inputs_sha256=tuple(inputs), source_profile=profile, pixel_basis=basis, camera=camera,
                        camera_motion=motion, tf_mode=tf["mode"], static_T_W_C=static_T, records=tuple(records))


def file_sha256(path: str) -> str:
    """입력 파일 지문."""
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    return h.hexdigest()


def check_inputs(fixture: SyntheticAToBFixture, paths: list[str], source_id: str) -> list[str]:
    """입력 파일 순서·지문과 소스 ID가 fixture와 같은지 확인하고 지문 목록을 돌려준다. 다르면 FixtureError."""
    if source_id != fixture.source_profile.source_id:
        raise FixtureError(f"--source-id {source_id!r} != fixture source.source_id "
                           f"{fixture.source_profile.source_id!r}")

    actual = [file_sha256(p) for p in paths]

    if tuple(actual) != fixture.inputs_sha256:
        raise FixtureError(f"input files (order/sha256) do not match fixture inputs_sha256: {actual}")

    return actual


# ---------- 실행기에 넣는 조각 ----------
class FixtureStampedSource:
    """저장 입력 소스를 감싸 fixture에 명시된 보정 번호·광학 프레임·촬영 시각만 프레임에 붙인다.

    `read`·`current`·`close`만 제공한다(실시간 소스 표지 `poll_input`이 없어 실행기는 저장 입력으로 다룬다).
    대응 레코드가 없는 프레임은 촬영 시각 없이 보정 번호만 붙이고 넘긴다(detector가 누락으로 실패시킨다).
    """

    def __init__(self, inner, fixture: SyntheticAToBFixture) -> None:
        """감쌀 소스와 fixture."""
        self._inner = inner
        self._fixture = fixture

    def read(self):
        """다음 항목. 프레임이면 fixture 값을 붙인 새 프레임(같은 크기·순번·세대)."""
        item = self._inner.read()

        if not isinstance(item, FramePacket):
            return item

        record = self._fixture.record_for_index(getattr(item, "source_index", -1))

        return dataclasses.replace(item, calibration_id=self._fixture.source_profile.calibration_id,
                                   optical_frame=self._fixture.camera.frame_id,
                                   ros_stamp_ns=None if record is None else record.ros_stamp_ns)

    def current(self):
        """감싼 소스의 현재 프레임(fixture 값을 붙이지 않은 원본)."""
        return self._inner.current()

    def close(self) -> None:
        """감싼 소스를 닫는다."""
        self._inner.close()


class FixtureDetector:
    """detector 자리(`run(detector=...)`)에 넣는 합성 관측 공급기. 작업마다 레코드를 정확히 한 번 쓴다."""

    def __init__(self, fixture: SyntheticAToBFixture) -> None:
        """fixture와 사용 기록."""
        self._fixture = fixture
        self.used: list[int] = []   # 쓴 레코드의 image_seq

    def __call__(self, job: JobContext):
        """작업 프레임의 레코드로 `(관측, None)`을 돌려준다. 재처리·누락·재사용·프레임 키 불일치는 FixtureError."""
        f = job.frame

        if job.reprocess:
            raise FixtureError(f"fixture observation is not attached to reprocess jobs (image_seq={f.image_seq})")

        record = self._fixture.record_for_index(getattr(f, "source_index", -1))

        if record is None:
            raise FixtureError(f"no fixture record for source_index={getattr(f, 'source_index', None)}")

        if record.image_seq != f.image_seq:
            raise FixtureError(f"fixture record image_seq {record.image_seq} != frame image_seq {f.image_seq}")

        if record.image_seq in self.used:
            raise FixtureError(f"fixture record image_seq {record.image_seq} already used")

        self.used.append(record.image_seq)

        # 식별자는 fixture 값 그대로 넣는다. 작업 값과 다르면 코어가 adapter_contract_violation으로 거부한다
        is_ai = isinstance(job.settings.detector_config, AiSegConfig)
        obs = PlanarObservation(display_mask_contour=None, process_ms=0.0, method="ai" if is_ai else "opencv",
                                confidence=None, model_id=None, **record.observation)

        return obs, None


class FixtureTfLookup:
    """fixture TF 조회. 고정 카메라는 `StaticTfLookup`, 움직이는 카메라는 촬영 시각 정확 일치 표만.

    `inner`(조회 함수)를 부른 시간이 `timeout_s`(기본 50 ms)를 넘으면 None을 돌려 B가 `TF_UNAVAILABLE`로 SKIP하게 한다.
    이는 반환 뒤 판정이다(블로킹 조회를 중간에 끊지 않음, 저장 fixture 조회는 즉시 끝남).
    """

    def __init__(self, fixture: SyntheticAToBFixture, *, timeout_s: float = TF_WAIT_LIMIT_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        """fixture로 조회 함수를 고른다."""
        self.timeout_s = timeout_s
        self._clock = clock
        self.calls = 0
        self.unavailable = 0
        self.timed_out = 0

        if fixture.tf_mode == "static":
            self.inner = StaticTfLookup(fixture.static_T_W_C)
        else:
            table = {r.ros_stamp_ns: r.T_W_C for r in fixture.records if r.T_W_C is not None}
            frame_id = fixture.camera.frame_id

            def at_capture_time(frame):
                """촬영 시각(`frame.ros_stamp_ns`)과 정확히 같은 시각의 `world → 광학 프레임`. 없으면 None."""
                if frame.ros_stamp_ns is None or frame.optical_frame != frame_id:
                    return None

                return table.get(frame.ros_stamp_ns)

            self.inner = at_capture_time

    def __call__(self, frame) -> np.ndarray | None:
        """조회하고 상한을 넘겼거나 없으면 None."""
        self.calls += 1
        t0 = self._clock()
        T = self.inner(frame)

        if self._clock() - t0 > self.timeout_s:
            self.timed_out += 1
            self.unavailable += 1
            return None

        if T is None:
            self.unavailable += 1

        return T


def _plain(value):
    """JSON으로 쓸 수 있는 값으로(numpy·열거)."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]

    if isinstance(value, np.ndarray):
        return _plain(value.tolist())

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return getattr(value, "value", value) if not isinstance(value, (str, int, float, bool, type(None))) else value


class SyntheticAToBRecorder:
    """B 시험 출력 JSONL(덮어쓰지 않음). 줄마다 `mode: synthetic/test-only`."""

    def __init__(self, path: str, fixture: SyntheticAToBFixture) -> None:
        """새 파일을 만든다(이미 있으면 FileExistsError)."""
        self._stream = open(path, "x", encoding="utf-8")
        self._fixture = fixture
        self.calls = 0
        self.kinds: dict[str, int] = {}

    def _write(self, record: dict) -> None:
        """한 줄 쓴다."""
        record = {"mode": TEST_MODE_LABEL, "fixture_id": self._fixture.fixture_id,
                  "fixture_sha256": self._fixture.sha256, **record}
        self._stream.write(json.dumps(_plain(record), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                           + "\n")
        self._stream.flush()

    def wrap(self, consume: Callable[[FramePacket, PlanarObservation], BOutput]):
        """`b_consumer` 자리 함수: B(`stage.consume`)를 부르고 그 출력을 입력 키와 함께 기록한다. B 예외는 그대로 올린다."""
        def consumer(frame, obs):
            """실행기가 `b_gate` 통과 뒤 부른다."""
            out = consume(frame, obs)
            self.calls += 1
            kind = out.goal.get("kind")
            self.kinds[kind] = self.kinds.get(kind, 0) + 1
            self._write({"kind": "b_test_output", "b_call": self.calls,
                         "input": {"run_id": obs.run_id, "source_id": obs.source_id, "image_seq": obs.image_seq,
                                   "source_index": getattr(frame, "source_index", None),
                                   "capture_generation": obs.capture_generation, "generation": obs.generation,
                                   "job_seq": obs.job_seq, "detector_config_id": obs.detector_config_id,
                                   "calibration_id": obs.calibration_id, "observed_face": obs.observed_face,
                                   "timestamp_ms": frame.timestamp_ms, "capture_stamp_ns": frame.ros_stamp_ns},
                         "result_dict": out.result_dict, "goal": out.goal,
                         "b_reason": out.result.reason, "b_reason_owner": out.result.reason_owner})

            return out

        return consumer

    def summary(self, record: dict) -> None:
        """끝 요약 한 줄."""
        self._write({"kind": "b_test_summary", "b_outputs": self.calls, "goal_kinds": self.kinds, **record})

    def close(self) -> None:
        """파일을 닫는다."""
        self._stream.close()


__all__ = ["SyntheticAToBFixture", "SyntheticAToBRecorder", "FIXTURE_FORMAT", "FixtureDetector", "FixtureError",
           "FixtureStampedSource", "FixtureTfLookup", "TEST_MODE_LABEL", "check_inputs", "load_fixture"]
