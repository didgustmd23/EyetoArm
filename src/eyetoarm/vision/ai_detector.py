"""ultralytics 분할 모델 어댑터 `AiSegDetector`. 가중치 확인·모델 준비·추론을 맡는다.

생성자는 I/O·import를 하지 않고, torch·ultralytics는 `prepare()` 안에서만 import한다. predict는 준비한
같은 스레드에서만 하며, OpenCV 대체 실행·가짜 ID·이전 결과 재사용은 하지 않는다.
"""
from __future__ import annotations

import hashlib
import importlib
import os
import threading
import time
from typing import Callable, MutableMapping

import numpy as np

from eyetoarm.common.types import FramePacket, InvalidationKind, PlanarObservation
from eyetoarm.vision.ai_config import AiSegConfig
from eyetoarm.vision.ai_convert import convert, crop_frame, observation_from_ai
from eyetoarm.vision.ai_result import AiJobConfigError, AiSegResult, ModelPrepareError, RawSegOutput
from eyetoarm.vision.ai_rot90 import merge_r5, rotate_back, rotate_crop
from eyetoarm.vision.pipeline_types import JobContext

# import 전에 설정해야 하는 환경 변수(자동 설치·온라인 접근 방지)
REQUIRED_ENV = (("YOLO_AUTOINSTALL", "false"), ("YOLO_OFFLINE", "1"))
_READ_CHUNK = 1 << 20
_FLOAT_EXACT_INT = 2 ** 53          # float64가 모든 정수를 정확히 담는 범위(넘으면 클래스 값을 믿을 수 없음)
_INT64_MAX = int(np.iinfo(np.int64).max)
LOAD, CONFIG = InvalidationKind.MODEL_LOAD_FAILED, InvalidationKind.MODEL_CONFIG_INVALID


def _numpy(value) -> np.ndarray:
    """텐서(cpu()·numpy())나 배열을 numpy 배열로."""
    if hasattr(value, "cpu"):
        value = value.cpu()

    if hasattr(value, "numpy"):
        value = value.numpy()

    return np.asarray(value)


def raw_from_results(results, crop_shape: tuple[int, ...], infer_ms: float) -> RawSegOutput:
    """predict 결과 목록에서 원시 배열을 뽑고 출력 계약을 확인한다. 위반은 ValueError.

    결과는 정확히 1개, K>0이면 masks가 있고, 클래스 값은 정수, 마스크는 crop 해상도여야 한다.
    """
    if not isinstance(results, (list, tuple)) or len(results) != 1:
        raise ValueError(f"predict must return exactly one result: {type(results).__name__}"
                         f"{'' if not hasattr(results, '__len__') else f' len={len(results)}'}")

    r = results[0]
    h, w = crop_shape[:2]
    orig = getattr(r, "orig_shape", None)

    if orig is not None and tuple(orig)[:2] != (h, w):
        raise ValueError(f"result orig_shape {tuple(orig)} != crop {(h, w)}")

    boxes, masks = r.boxes, r.masks
    k = 0 if boxes is None else len(boxes)

    if k == 0:
        if masks is not None and _numpy(masks.data).shape[0] != 0:
            raise ValueError("masks present without boxes")

        return RawSegOutput(boxes_xyxy=np.zeros((0, 4), np.float32), confidence=np.zeros(0, np.float32),
                            class_index=np.zeros(0, np.int64), masks=np.zeros((0, h, w), bool), crop_wh=(w, h),
                            infer_ms=infer_ms)

    if masks is None:
        raise ValueError(f"{k} boxes without masks")

    return RawSegOutput(boxes_xyxy=_numpy(boxes.xyxy), confidence=_numpy(boxes.conf),
                        class_index=_class_values(boxes.cls), masks=_binary_masks(masks.data), crop_wh=(w, h),
                        infer_ms=infer_ms)


def _class_values(value) -> np.ndarray:
    """클래스 값을 int64로 바꾼다. 실수는 유한·정수·2**53 이하일 때만 받고, 위반은 ValueError.

    음수는 여기서 거부하지 않는다(RawSegOutput이 거부).
    """
    a = _numpy(value)

    if a.dtype == np.bool_ or not (np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.floating)):
        raise ValueError(f"class values must be numeric: {a.dtype}")

    if np.issubdtype(a.dtype, np.integer):
        if a.size and int(a.max()) > _INT64_MAX:
            raise ValueError("class values exceed int64 range")

        return a.astype(np.int64)

    f = a.astype(np.float64)

    if not np.all(np.isfinite(f)) or not np.all(f == np.round(f)):
        raise ValueError(f"class values must be finite integers: {f.tolist()}")

    if f.size and float(np.max(np.abs(f))) > _FLOAT_EXACT_INT:
        raise ValueError("float class values beyond 2**53 cannot be represented exactly")

    return f.astype(np.int64)


def _binary_masks(value) -> np.ndarray:
    """마스크 데이터를 bool로 바꾼다. 숫자는 > 0.5를 전경으로 보고, NaN·inf는 ValueError."""
    data = _numpy(value)

    if data.dtype == np.bool_:
        return data

    if not (np.issubdtype(data.dtype, np.integer) or np.issubdtype(data.dtype, np.floating)):
        raise ValueError(f"mask data must be numeric or bool: {data.dtype}")

    if not np.all(np.isfinite(data)):
        raise ValueError("mask data must be finite (NaN/inf in model masks)")

    return data > 0.5


def _names_dict(names) -> dict[int, str]:
    """모델 names(dict 또는 list)를 {번호: 이름}으로."""
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}

    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}

    raise TypeError(f"unsupported names type: {type(names).__name__}")


class AiSegDetector:
    """ultralytics 분할 모델 어댑터. 설정 하나·모델 하나를 소유한다."""

    def __init__(self, config: AiSegConfig, *, model_factory: Callable[[str], object] | None = None,
                 environ: MutableMapping[str, str] | None = None,
                 import_module: Callable[[str], object] = importlib.import_module,
                 cuda_device_count: Callable[[], int] | None = None) -> None:
        """설정과 주입 의존성을 받는다. I/O·import를 하지 않는다."""
        if not isinstance(config, AiSegConfig):
            raise TypeError(f"config must be AiSegConfig: {type(config).__name__}")

        self._config = config
        self._model_factory = model_factory
        self._environ = os.environ if environ is None else environ
        self._import_module = import_module
        self._cuda_device_count = cuda_device_count
        self._model = None
        self._torch = None
        self._owner_thread: int | None = None
        self._infer_calls = 0   # infer 호출 수(돌린 추론 차례 `every_n` 계산용, 준비한 스레드만 씀)

    @property
    def config(self) -> AiSegConfig:
        """어댑터 설정."""
        return self._config

    def _check_weights(self) -> None:
        """가중치 파일 존재·정규 파일·읽기·sha256 대조."""
        path = self._config.weights_path

        if not os.path.exists(path):
            raise ModelPrepareError(LOAD, f"weights not found: {path}")

        if not os.path.isfile(path):
            raise ModelPrepareError(LOAD, f"weights is not a regular file: {path}")

        digest = hashlib.sha256()

        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(_READ_CHUNK), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise ModelPrepareError(LOAD, f"weights unreadable: {path}: {type(exc).__name__}: {exc}") from exc

        actual = digest.hexdigest()

        if actual != self._config.weights_sha256:
            raise ModelPrepareError(CONFIG, f"weights sha256 mismatch: expected {self._config.weights_sha256}, "
                                            f"got {actual} ({path})")

    def _guard_env(self) -> dict[str, str]:
        """필수 환경 변수를 확인·설정한다. 다른 값이 있으면 덮지 않고 거부한다(자동 설치 방지)."""
        for key, want in REQUIRED_ENV:
            value = self._environ.get(key)

            if value is not None and value != want:
                raise ModelPrepareError(LOAD, f"unsafe env {key}={value!r} (must be {want!r}; refuses auto-install)")

        for key, want in REQUIRED_ENV:
            self._environ.setdefault(key, want)

        return {key: self._environ[key] for key, _ in REQUIRED_ENV}

    def prepare(self) -> dict:
        """모델을 준비해 model_ready 기록 정보를 돌려준다. 실패하면 ModelPrepareError."""
        t0 = time.perf_counter_ns()
        cfg = self._config
        self._check_weights()
        env = self._guard_env()
        versions: dict[str, str] = {}
        factory = self._model_factory
        source = "injected_factory"   # 주입 factory는 라이브러리 버전을 알 수 없어 꾸며 넣지 않는다

        if factory is None:
            source = "ultralytics"

            try:
                ultralytics = self._import_module("ultralytics")
                factory = ultralytics.YOLO
            except Exception as exc:  # 설치 안 됨·import 실패를 로딩 실패로 분류
                raise ModelPrepareError(LOAD, f"import ultralytics failed: {type(exc).__name__}: {exc}") from exc

            # 기본 경로는 장치와 관계없이 torch 버전도 기록한다
            try:
                self._torch = self._import_module("torch")
            except Exception as exc:
                raise ModelPrepareError(LOAD, f"import torch failed: {type(exc).__name__}: {exc}") from exc

            versions["ultralytics"] = str(getattr(ultralytics, "__version__", "unknown"))
            versions["torch"] = str(getattr(self._torch, "__version__", "unknown"))

        try:
            model = factory(cfg.weights_path)
        except Exception as exc:
            raise ModelPrepareError(LOAD, f"model construction failed: {type(exc).__name__}: {exc}") from exc

        task = getattr(model, "task", None)

        if task != "segment":
            raise ModelPrepareError(CONFIG, f"model task must be 'segment': {task!r}")

        try:
            names = _names_dict(getattr(model, "names", None))
        except (TypeError, ValueError) as exc:
            raise ModelPrepareError(CONFIG, f"model names unreadable: {exc}") from exc

        if names != cfg.class_names:
            raise ModelPrepareError(CONFIG, f"model names {names} != class_map {cfg.class_names}")

        device_name = self._check_device(versions)
        self._model = model
        self._owner_thread = threading.get_ident()

        return {"model_load_ms": (time.perf_counter_ns() - t0) / 1e6, "model_id": cfg.model_id,
                "weights_path": cfg.weights_path, "weights_sha256": cfg.weights_sha256, "task": task,
                "names": {str(k): v for k, v in sorted(names.items())}, "device": cfg.device,
                "device_name": device_name, "versions": versions, "env": env, "model_source": source}

    def _check_device(self, versions: dict) -> str | None:
        """CUDA 장치 번호가 있는지 확인한다(cpu는 확인 없음). 장치 이름을 돌려준다."""
        device = self._config.device

        if device == "cpu":
            return None

        count_fn, name = self._cuda_device_count, None

        if count_fn is None:
            torch = self._torch

            if torch is None:   # 주입 factory + 실제 CUDA 확인: 이때만 torch를 따로 가져온다
                try:
                    torch = self._import_module("torch")
                except Exception as exc:
                    raise ModelPrepareError(LOAD, f"import torch failed: {type(exc).__name__}: {exc}") from exc

                versions["torch"] = str(getattr(torch, "__version__", "unknown"))

            count_fn = lambda: torch.cuda.device_count() if torch.cuda.is_available() else 0  # noqa: E731
            name_fn = torch.cuda.get_device_name
        else:
            name_fn = None

        count = int(count_fn())
        index = int(device)

        if index >= count:
            raise ModelPrepareError(CONFIG, f"CUDA device {device} unavailable (count={count})")

        if name_fn is not None:
            name = str(name_fn(index))

        return name

    def __call__(self, job: JobContext) -> tuple[PlanarObservation, AiSegResult]:
        """작업 하나를 추론해 (표시용 무효 관측, 표시 결과)를 돌려준다."""
        result = self.infer(job.frame, job.detector_config_id)

        return observation_from_ai(job, result), result

    def infer(self, frame: FramePacket, detector_config_id: str) -> AiSegResult:
        """프레임 하나를 crop·추론·변환해 표시 결과만 돌려준다. 추론 프로세스 자식도 이것을 부른다."""
        if self._model is None:
            raise RuntimeError("AiSegDetector used before prepare()")

        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("AiSegDetector must be called from the thread that ran prepare()")

        cfg = self._config

        if detector_config_id != cfg.config_sha256:
            raise AiJobConfigError(f"job start config {detector_config_id[:12]} != detector config "
                                   f"{cfg.config_sha256[:12]}")

        t0 = time.perf_counter_ns()
        crop, _ = crop_frame(frame, cfg)

        # predict에는 폐기된 `half` 대신 `quantize`로 넘긴다(False → None(FP32), True → 16(FP16))
        quantize = 16 if cfg.half else None

        raw = self._predict(crop, quantize)

        # 90° 돌린 추론: 켜져 있고 이번이 차례면(호출 수 % every_n == 0) 한 번 더 추론해 R5로 합친다
        rot = cfg.rotate90_pass
        turn = rot is not None and self._infer_calls % rot.every_n == 0
        self._infer_calls += 1

        if not turn:
            return convert(raw, frame, cfg, started_ns=t0)

        raw_rot = rotate_back(self._predict(rotate_crop(crop, rot.direction), quantize), (crop.shape[1], crop.shape[0]))
        merged, sources = merge_r5(raw, raw_rot, cfg.class_names, nms_iou=rot.nms_iou, nut_veto_iou=rot.nut_veto_iou)

        return convert(merged, frame, cfg, started_ns=t0, sources=sources, rot_infer_ms=raw_rot.infer_ms)

    def _predict(self, image, quantize) -> RawSegOutput:
        """영상 하나를 운영 인자로 predict해 원시 출력으로 바꾼다(계약 위반은 ValueError)."""
        cfg = self._config

        t1 = time.perf_counter_ns()
        results = self._model.predict(image, imgsz=cfg.imgsz, conf=cfg.conf_threshold, iou=cfg.iou_threshold,
                                      max_det=cfg.max_det, device=cfg.device, quantize=quantize, retina_masks=True,
                                      verbose=False)
        infer_ms = (time.perf_counter_ns() - t1) / 1e6

        raw = raw_from_results(results, image.shape, infer_ms)
        del results  # 결과 객체(마스크 텐서)를 붙잡지 않는다

        return raw


__all__ = ["AiSegDetector", "REQUIRED_ENV", "raw_from_results"]
