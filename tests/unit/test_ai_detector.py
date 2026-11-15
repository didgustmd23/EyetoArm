"""YOLO 연결 T5 — ultralytics 분할 어댑터(`AiSegDetector`) 시험. 가짜 model_factory·import·CUDA 확인을 주입한다.

근거: 계획 `docs/superpowers/plans/2026-11-15-yolo-pipeline.md` r2 §2(BGR 입력·retina 마스크 계약), §4.2(준비 호출 순서:
파일 존재 → 읽기·sha256 → 환경 변수 → import → 생성 → task·names·장치, 생성자는 I/O 없음), §4.3 4단계(출력 계약 확인), §4.5
(분류), T5, A 상세설계 §7.1 ①(`YOLO_AUTOINSTALL=false`). 실제 ultralytics·torch·가중치 없이 돈다.
"""
import hashlib
import os
import subprocess
import sys
import tempfile
import threading
import unittest

import numpy as np

from eyetoarm.common.types import InvalidationKind
from eyetoarm.vision.ai_detector import AiSegDetector, raw_from_results
from eyetoarm.vision.ai_result import AiJobConfigError, AiSegResult, ModelPrepareError
from test_ai_config import ai_config
from test_ai_convert import W, H, big_frame, job_for, rect_mask
from test_pipeline import profile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOAD, CONFIG = InvalidationKind.MODEL_LOAD_FAILED, InvalidationKind.MODEL_CONFIG_INVALID


class FakeTensor:
    """torch 텐서 흉내: cpu()·numpy()."""

    def __init__(self, array):
        """배열을 감싼다."""
        self.array = np.asarray(array)

    def cpu(self):
        """자기 자신."""
        return self

    def numpy(self):
        """배열."""
        return self.array


class FakeBoxes:
    """Results.boxes 흉내(xyxy·conf·cls, cls는 실수 텐서)."""

    def __init__(self, xyxy, conf, cls):
        """K개 박스."""
        self.xyxy, self.conf, self.cls = FakeTensor(xyxy), FakeTensor(conf), FakeTensor(cls)

    def __len__(self):
        """K."""
        return len(self.xyxy.array)


class FakeMasks:
    """Results.masks 흉내(data: (K,h,w) 실수 0/1)."""

    def __init__(self, data):
        """마스크 데이터."""
        self.data = FakeTensor(data)


class FakeResult:
    """Results 하나 흉내."""

    def __init__(self, boxes, masks, orig_shape):
        """박스·마스크·원본(=넘긴 crop) 크기."""
        self.boxes, self.masks, self.orig_shape = boxes, masks, orig_shape


def result_for(source, rects, cls=None, conf=None):
    """넘긴 crop(source) 해상도의 retina 마스크로 가짜 결과를 만든다. rects는 crop 좌표 (x0,y0,x1,y1)."""
    h, w = source.shape[:2]
    masks = [rect_mask((w, h), *r).astype(np.float32) for r in rects]
    k = len(rects)
    boxes = FakeBoxes(np.array([list(r) for r in rects], np.float32).reshape(k, 4),
                      np.array(conf or [0.9] * k, np.float32), np.array(cls or [0.0] * k, np.float32))

    return [FakeResult(boxes, FakeMasks(np.array(masks, np.float32).reshape(k, h, w)) if k else None, (h, w))]


class FakeModel:
    """YOLO 객체 흉내. predict 인자를 기록하고 make(source)가 만든 결과를 돌려준다."""

    def __init__(self, task="segment", names=None, make=None):
        """task·names·결과 생성 함수."""
        self.task = task
        self.names = names if names is not None else {0: "bolt", 1: "nut"}
        self.make = make or (lambda source: result_for(source, []))
        self.calls = []

    def predict(self, source, **kwargs):
        """호출 기록 후 결과."""
        self.calls.append((source.copy(), source, kwargs, threading.get_ident()))
        return self.make(source)


class WeightsMixin:
    """임시 가중치 파일과 detector 생성 도우미."""

    def setUp(self):
        """가짜 가중치 파일(내용 고정)을 만든다."""
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.weights = os.path.join(self.dir.name, "best.pt")
        with open(self.weights, "wb") as f:
            f.write(b"fake weights for tests")
        with open(self.weights, "rb") as f:
            self.sha = hashlib.sha256(f.read()).hexdigest()

    def make(self, model=None, factory=None, cfg_over=None, environ=None, cuda_count=1, import_module=None):
        """주입 가능한 의존성으로 detector를 만든다. 기본 장치 cpu."""
        self.model = model or FakeModel()
        cfg = ai_config(**{"weights_path": self.weights, "weights_sha256": self.sha, "device": "cpu",
                           **(cfg_over or {})})
        self.environ = {} if environ is None else environ
        det = AiSegDetector(cfg, model_factory=factory or (lambda path: self.model), environ=self.environ,
                            cuda_device_count=lambda: cuda_count,
                            import_module=import_module or (lambda name: (_ for _ in ()).throw(
                                AssertionError(f"unexpected import {name}"))))
        return det, cfg


class PrepareTest(WeightsMixin, unittest.TestCase):
    """준비 단계(§4.2 ③-a)의 순서·분류."""

    def assert_prepare_error(self, det, kind, text):
        """prepare가 kind·원인 문자열을 담은 ModelPrepareError를 낸다."""
        with self.assertRaises(ModelPrepareError) as ctx:
            det.prepare()
        self.assertIs(ctx.exception.kind, kind)
        self.assertIn(text, ctx.exception.cause)

    def test_constructor_does_no_io(self):
        """생성자는 파일·import·factory에 손대지 않는다(없는 경로로도 생성 성공, factory 호출 0)."""
        called = []
        cfg = ai_config(weights_path=os.path.join(self.dir.name, "missing.pt"))

        AiSegDetector(cfg, model_factory=lambda p: called.append(p), environ={},
                      import_module=lambda n: called.append(n), cuda_device_count=lambda: called.append("cuda"))

        self.assertEqual(called, [])
        with self.assertRaises(TypeError):
            AiSegDetector("not a config")

    def test_prepare_success_info(self):
        """정상 준비는 모델을 한 번 만들고 model_ready 기록용 정보(로딩 ms·model_id·해시·task·names·장치·환경)를 돌려준다."""
        made = []
        det, cfg = self.make(factory=lambda path: made.append(path) or self.model)

        info = det.prepare()

        self.assertEqual(made, [self.weights])
        self.assertEqual((info["model_id"], info["weights_sha256"], info["task"], info["names"], info["device"]),
                         (cfg.model_id, self.sha, "segment", {"0": "bolt", "1": "nut"}, "cpu"))
        self.assertEqual(info["env"], {"YOLO_AUTOINSTALL": "false", "YOLO_OFFLINE": "1"})
        self.assertGreaterEqual(info["model_load_ms"], 0.0)

    def test_prepare_missing_weights_is_load_failed(self):
        """가중치 파일이 없으면 MODEL_LOAD_FAILED이고 원인에 경로가 있으며 factory를 부르지 않는다."""
        det, _ = self.make(cfg_over={"weights_path": os.path.join(self.dir.name, "nope.pt")},
                           factory=lambda p: self.fail("factory called"))

        self.assert_prepare_error(det, LOAD, "nope.pt")

    def test_prepare_unreadable_weights_is_load_failed(self):
        """디렉터리 경로·읽기 권한 없는 파일은 MODEL_LOAD_FAILED다."""
        det, _ = self.make(cfg_over={"weights_path": self.dir.name})
        self.assert_prepare_error(det, LOAD, "not a regular file")

        os.chmod(self.weights, 0)
        self.addCleanup(os.chmod, self.weights, 0o600)
        det, _ = self.make()
        self.assert_prepare_error(det, LOAD, "unreadable")

    def test_prepare_hash_mismatch_is_config_invalid(self):
        """가중치 sha256이 설정과 다르면 MODEL_CONFIG_INVALID이고 factory를 부르지 않는다."""
        det, _ = self.make(cfg_over={"weights_sha256": "0" * 64}, factory=lambda p: self.fail("factory called"))

        self.assert_prepare_error(det, CONFIG, "sha256 mismatch")

    def test_prepare_import_or_factory_error_is_load_failed(self):
        """기본 factory의 ultralytics import 실패와 factory 예외는 MODEL_LOAD_FAILED다."""
        def no_module(name):
            """import 실패."""
            raise ImportError(f"No module named {name!r}")

        cfg = ai_config(weights_path=self.weights, weights_sha256=self.sha, device="cpu")
        det = AiSegDetector(cfg, environ={}, import_module=no_module, cuda_device_count=lambda: 0)
        self.assert_prepare_error(det, LOAD, "import ultralytics failed")

        det, _ = self.make(factory=lambda p: (_ for _ in ()).throw(RuntimeError("bad checkpoint")))
        self.assert_prepare_error(det, LOAD, "bad checkpoint")

    def test_prepare_task_names_device_mismatch_is_config_invalid(self):
        """task가 segment가 아니거나 names가 class_map과 다르거나, CUDA 장치가 없거나 번호가 범위 밖이면 MODEL_CONFIG_INVALID다."""
        cases = [
            ({"model": FakeModel(task="detect")}, "task"),
            ({"model": FakeModel(names={0: "bolt"})}, "names"),
            ({"model": FakeModel(names={0: "bolt", 1: "washer"})}, "names"),
            ({"cfg_over": {"device": "0"}, "cuda_count": 0}, "CUDA"),
            ({"cfg_over": {"device": "1"}, "cuda_count": 1}, "CUDA"),
        ]

        for kw, text in cases:
            with self.subTest(text=text, kw=list(kw)):
                det, _ = self.make(**kw)
                self.assert_prepare_error(det, CONFIG, text)

        det, _ = self.make(model=FakeModel(names=["bolt", "nut"]), cfg_over={"device": "0"}, cuda_count=1)
        self.assertEqual(det.prepare()["names"], {"0": "bolt", "1": "nut"})

    def test_env_guard(self):
        """필수 환경 변수는 import·생성 전에 설정되고, 다른 값이 이미 있으면 덮지 않고 MODEL_LOAD_FAILED(unsafe env)다."""
        seen = []
        env = {}
        det, _ = self.make(environ=env, factory=lambda p: seen.append(dict(env)) or self.model)
        det.prepare()
        self.assertEqual(seen, [{"YOLO_AUTOINSTALL": "false", "YOLO_OFFLINE": "1"}])

        for key, value in (("YOLO_AUTOINSTALL", "true"), ("YOLO_OFFLINE", "0"), ("YOLO_AUTOINSTALL", "False")):
            with self.subTest(key=key, value=value):
                env = {key: value}
                det, _ = self.make(environ=env, factory=lambda p: self.fail("factory called"))
                self.assert_prepare_error(det, LOAD, "unsafe env")
                self.assertEqual(env, {key: value})

    def test_no_ultralytics_import_until_prepare(self):
        """ai_detector를 불러 detector를 만들어도 torch·ultralytics가 로드되지 않는다(OpenCV 단독 실행 유지)."""
        code = ("import sys\n"
                "from eyetoarm.vision.ai_detector import AiSegDetector\n"
                "from eyetoarm.vision.ai_config import AiSegConfig\n"
                "c = AiSegConfig(format_version=1, method='ai', model_family='yolo11n-seg', weights_path='x.pt',"
                " weights_sha256='a'*64, class_map={0: 'bolt'}, target_classes=['bolt'], conf_threshold=0.25,"
                " iou_threshold=0.7, imgsz=640, max_det=10, device='cpu', half=False, retina_masks=True, roi=None,"
                " min_mask_area_px=0, border_margin_px=0)\n"
                "AiSegDetector(c)\n"
                "print(','.join(k for k in sorted(sys.modules) if k.split('.')[0] in ('torch', 'ultralytics')) "
                "or 'none')")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                             env={"PYTHONPATH": os.path.join(ROOT, "src"), "PATH": ""}, cwd=ROOT)

        self.assertEqual(out.stdout.strip(), "none")


class InferenceTest(WeightsMixin, unittest.TestCase):
    """추론 호출(§4.3·§4.4)."""

    def ready(self, make=None, cfg_over=None, prof=True):
        """준비를 마친 detector와 작업."""
        det, cfg = self.make(model=FakeModel(make=make), cfg_over=cfg_over)
        det.prepare()
        img = np.zeros((H, W, 3), np.uint8)
        img[..., 0], img[..., 1], img[..., 2] = 10, 20, 30          # B·G·R 채널 값을 달리해 색 순서 변환을 드러낸다
        img[:, :, 0] += (np.arange(W, dtype=np.uint8) % 50)[None, :]
        fr = big_frame()
        fr = type(fr)(**{**{f: getattr(fr, f) for f in fr.__dataclass_fields__}, "image": img})
        job = job_for(cfg, fr, profile(width=W, height=H) if prof else None)
        return det, cfg, job

    def test_passes_bgr_crop_unchanged_and_kwargs(self):
        """predict는 원본 BGR crop을 값·shape 그대로(색 변환·복사 수정 없이) 받고 설정의 imgsz·conf·iou·max_det·device·half·retina_masks를 받는다. 원본 프레임은 바뀌지 않는다."""
        det, cfg, job = self.ready(cfg_over={"roi": [100, 50, 64, 48], "imgsz": 640})
        img = job.frame.image
        before = img.copy()

        det(job)
        copy, source, kwargs, _ = self.model.calls[0]

        self.assertTrue(np.array_equal(copy, img[50:98, 100:164]))
        self.assertEqual(tuple(int(v) for v in copy[0, 0]), (10 + 100 % 50, 20, 30))   # BGR 순서 그대로
        self.assertEqual((source.shape, source.dtype), ((48, 64, 3), np.uint8))
        self.assertEqual(kwargs, {"imgsz": 640, "conf": cfg.conf_threshold, "iou": cfg.iou_threshold,
                                  "max_det": cfg.max_det, "device": "cpu", "half": False, "retina_masks": True,
                                  "verbose": False})
        self.assertTrue(np.array_equal(img, before))

    def test_call_returns_invalid_observation_and_offset_polygons(self):
        """결과는 표시용 무효 관측(TOP_FACE_UNRESOLVED·candidate 2)과 ROI 오프셋만 더한 다각형의 AiSegResult다."""
        det, cfg, job = self.ready(make=lambda s: result_for(s, [(10, 5, 19, 14), (30, 5, 39, 14)], cls=[0.0, 1.0]),
                                   cfg_over={"roi": [100, 50, 64, 48]})

        obs, res = det(job)

        self.assertIsInstance(res, AiSegResult)
        self.assertEqual((obs.valid, obs.reason, obs.candidate_count, obs.method), (False, "TOP_FACE_UNRESOLVED", 2,
                                                                                   "ai"))
        pts = {(int(x), int(y)) for x, y in res.instances[0].polygon}
        self.assertTrue({(110, 55), (119, 64)} <= pts)
        self.assertEqual([i.class_name for i in res.instances], ["bolt", "nut"])

    def test_empty_result_is_no_object(self):
        """K=0(masks None)은 정상 무검출 NO_OBJECT다."""
        det, _, job = self.ready()

        obs, res = det(job)

        self.assertEqual((obs.reason, obs.candidate_count, res.instances), ("NO_OBJECT", 0, ()))

    def test_results_contract_violations_raise(self):
        """출력 계약 위반(결과 수≠1, K>0인데 masks None, K 불일치, 비retina 해상도, NaN 박스, 비정수 클래스)은 ValueError, class_map 밖 클래스는 AiJobConfigError다."""
        def bad(kind):
            """종류별 잘못된 결과 생성 함수."""
            def make(s):
                """잘못된 가짜 결과."""
                r = result_for(s, [(10, 5, 19, 14)])
                res = r[0]
                if kind == "two results":
                    return r + r
                if kind == "masks None":
                    res.masks = None
                if kind == "K mismatch":
                    res.masks = FakeMasks(np.zeros((2,) + s.shape[:2], np.float32))
                if kind == "non-retina":
                    res.masks = FakeMasks(np.zeros((1, 32, 32), np.float32))
                if kind == "nan box":
                    res.boxes.xyxy = FakeTensor(np.full((1, 4), np.nan, np.float32))
                if kind == "fractional cls":
                    res.boxes.cls = FakeTensor(np.array([0.5], np.float32))
                if kind == "unknown cls":
                    res.boxes.cls = FakeTensor(np.array([7.0], np.float32))
                return r
            return make

        for kind in ("two results", "masks None", "K mismatch", "non-retina", "nan box", "fractional cls"):
            with self.subTest(kind=kind):
                det, _, job = self.ready(make=bad(kind))
                with self.assertRaises(ValueError):
                    det(job)

        det, _, job = self.ready(make=bad("unknown cls"))
        with self.assertRaises(AiJobConfigError):
            det(job)

    def test_not_prepared_or_other_thread_is_error(self):
        """준비 전 호출과 준비한 스레드가 아닌 스레드의 호출은 RuntimeError다(모델 사용 주체는 하나)."""
        det, cfg = self.make()
        job = job_for(cfg, big_frame())

        with self.assertRaises(RuntimeError):
            det(job)

        det.prepare()
        box = {}
        t = threading.Thread(target=lambda: box.setdefault("err", self._call(det, job)))
        t.start()
        t.join(5)

        self.assertIsInstance(box["err"], RuntimeError)
        self.assertEqual(self.model.calls, [])

    @staticmethod
    def _call(det, job):
        """호출하고 예외를 돌려준다(없으면 None)."""
        try:
            det(job)
        except Exception as exc:  # 스레드 경계 넘어 확인용
            return exc
        return None

    def test_snapshot_config_mismatch_fails_job_without_predict(self):
        """작업 시작 설정 해시가 detector 설정과 다르면 predict 없이 AiJobConfigError다(숨은 설정으로 계산하지 않음)."""
        det, cfg = self.make()
        det.prepare()
        job = job_for(ai_config(weights_path=self.weights, weights_sha256=self.sha, device="cpu",
                                conf_threshold=0.5), big_frame())

        with self.assertRaises(AiJobConfigError):
            det(job)

        self.assertEqual(self.model.calls, [])

    def test_roi_outside_frame_is_job_config_error(self):
        """설정 ROI가 작업 프레임 밖이면 predict 없이 AiJobConfigError다."""
        det, _, job = self.ready(cfg_over={"roi": [150, 0, 64, 48]})

        with self.assertRaises(AiJobConfigError):
            det(job)

        self.assertEqual(self.model.calls, [])


class FakeModule:
    """import_module이 돌려줄 가짜 모듈(버전·YOLO·cuda)."""

    def __init__(self, **attrs):
        """속성을 받는다."""
        self.__dict__.update(attrs)


class R2BoundaryTest(WeightsMixin, unittest.TestCase):
    """R2 r1 F1·F2·L1 경계."""

    def crop(self):
        """가짜 crop(48×64 BGR)."""
        return np.zeros((48, 64, 3), np.uint8)

    def test_mask_nan_or_inf_is_contract_error(self):
        """마스크 데이터의 NaN·+inf·-inf는 이진화로 숨기지 않고 ValueError(INFERENCE_ERROR)다(F1). 0/1 실수·bool 마스크와 K=0은 정상."""
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(value=bad):
                r = result_for(self.crop(), [(10, 5, 19, 14)])
                data = r[0].masks.data.array.copy()
                data[0, 7, 12] = bad
                r[0].masks = FakeMasks(data)
                with self.assertRaises(ValueError):
                    raw_from_results(r, (48, 64, 3), 1.0)

        ok = result_for(self.crop(), [(10, 5, 19, 14)])
        self.assertEqual(int(raw_from_results(ok, (48, 64, 3), 1.0).masks.sum()), 100)
        ok[0].masks = FakeMasks(ok[0].masks.data.array > 0.5)
        self.assertEqual(int(raw_from_results(ok, (48, 64, 3), 1.0).masks.sum()), 100)
        self.assertEqual(raw_from_results(result_for(self.crop(), []), (48, 64, 3), 1.0).count, 0)

        bad_type = result_for(self.crop(), [(10, 5, 19, 14)])
        bad_type[0].masks = FakeMasks(np.full((1, 48, 64), "x", object))
        with self.assertRaises(ValueError):
            raw_from_results(bad_type, (48, 64, 3), 1.0)

    def test_detector_call_with_nan_mask_raises_without_observation(self):
        """detector 호출에서도 NaN 마스크는 관측 없이 ValueError로 올라간다(NO_OBJECT·EMPTY_MASK로 바꾸지 않음)."""
        def make(source):
            """NaN이 섞인 마스크."""
            r = result_for(source, [(10, 5, 19, 14)])
            data = r[0].masks.data.array.copy()
            data[0, 7, 12] = np.nan
            r[0].masks = FakeMasks(data)
            return r

        det, cfg = self.make(model=FakeModel(make=make))
        det.prepare()

        with self.assertRaises(ValueError):
            det(job_for(cfg, big_frame()))

    def test_class_values_preserved_or_rejected(self):
        """클래스 값: 정수 배열은 float을 거치지 않고 보존(uint64 2**53+1), int64 범위 밖(uint64 2**63)·float의 2**53 초과(정밀도 불확실)·2**63·비정수는 ValueError(L1)."""
        def with_cls(values):
            """클래스 값을 바꾼 결과."""
            r = result_for(self.crop(), [(10, 5, 19, 14)])
            r[0].boxes.cls = FakeTensor(values)
            return r

        kept = raw_from_results(with_cls(np.array([2 ** 53 + 1], np.uint64)), (48, 64, 3), 1.0)
        self.assertEqual(int(kept.class_index[0]), 2 ** 53 + 1)
        self.assertEqual(int(raw_from_results(with_cls(np.array([1.0], np.float32)), (48, 64, 3), 1.0)
                             .class_index[0]), 1)

        for values in (np.array([2 ** 63], np.uint64), np.array([2.0 ** 63]), np.array([2.0 ** 53 + 2]),
                       np.array([0.5]), np.array([np.nan]), np.array([True])):
            with self.subTest(values=repr(values)):
                with self.assertRaises(ValueError):
                    raw_from_results(with_cls(values), (48, 64, 3), 1.0)

    def default_path(self, device, cuda_available=True, count=1):
        """기본 라이브러리 경로(가짜 import_module로 ultralytics·torch 주입) detector."""
        imported = []
        model = FakeModel()
        cuda = FakeModule(is_available=lambda: cuda_available, device_count=lambda: count,
                          get_device_name=lambda i: f"Fake GPU {i}")
        modules = {"ultralytics": FakeModule(__version__="8.4.211", YOLO=lambda path: model),
                   "torch": FakeModule(__version__="2.11.0+cu128", cuda=cuda)}

        def import_module(name):
            """가짜 import(기록)."""
            imported.append(name)
            return modules[name]

        cfg = ai_config(weights_path=self.weights, weights_sha256=self.sha, device=device)
        return AiSegDetector(cfg, environ={}, import_module=import_module), imported

    def test_default_path_records_torch_version_for_cpu_and_gpu(self):
        """기본 라이브러리 경로는 CPU·GPU 모두 ultralytics·torch 버전을 기록하고 model_source는 ultralytics다. GPU는 장치 이름도 기록한다(F2)."""
        det, imported = self.default_path("cpu")
        info = det.prepare()
        self.assertEqual(info["versions"], {"ultralytics": "8.4.211", "torch": "2.11.0+cu128"})
        self.assertEqual((info["model_source"], info["device_name"]), ("ultralytics", None))
        self.assertEqual(imported, ["ultralytics", "torch"])

        det, imported = self.default_path("0")
        info = det.prepare()
        self.assertEqual(info["versions"], {"ultralytics": "8.4.211", "torch": "2.11.0+cu128"})
        self.assertEqual(info["device_name"], "Fake GPU 0")
        self.assertEqual(imported, ["ultralytics", "torch"])

        det, _ = self.default_path("0", cuda_available=False)
        with self.assertRaises(ModelPrepareError) as ctx:
            det.prepare()
        self.assertIs(ctx.exception.kind, CONFIG)

    def test_default_path_torch_import_failure_is_load_failed(self):
        """기본 경로에서 torch import가 실패하면 MODEL_LOAD_FAILED다."""
        cfg = ai_config(weights_path=self.weights, weights_sha256=self.sha, device="cpu")

        def import_module(name):
            """ultralytics만 있음."""
            if name == "torch":
                raise ImportError("no torch")
            return FakeModule(__version__="8.4.211", YOLO=lambda path: FakeModel())

        with self.assertRaises(ModelPrepareError) as ctx:
            AiSegDetector(cfg, environ={}, import_module=import_module).prepare()
        self.assertIs(ctx.exception.kind, LOAD)
        self.assertIn("torch", ctx.exception.cause)

    def test_injected_factory_does_not_fake_versions(self):
        """주입 factory 경로는 실제 버전을 꾸며 넣지 않는다: versions {}, model_source "injected_factory"."""
        det, _ = self.make()
        info = det.prepare()

        self.assertEqual((info["versions"], info["model_source"]), ({}, "injected_factory"))


if __name__ == "__main__":
    unittest.main()
