"""추론 프로세스 시험용 가짜 detector 공장(자식 프로세스에서 `--factory ai_process_fake:이름`으로 불린다).

실제 `AiSegDetector`에 가짜 모델(`FakeModel`)을 주입한다. crop 왼쪽 위 픽셀 B 값이 동작을 정한다:
0 정상, 1 멈춤(오래 잠), 2 자식 죽음(os._exit), 3 예외(ValueError), 4 지연(`FAKE_DELAY_S`) 뒤 정상.
정상 결과는 G 값으로 위치를 정한 가로 볼트 마스크 1개다(같은 영상이면 같은 결과).
"""
import os
import time

import numpy as np

from eyetoarm.common.types import InvalidationKind
from eyetoarm.vision.ai_detector import AiSegDetector
from eyetoarm.vision.ai_result import ModelPrepareError

NORMAL, HANG, DIE, RAISE, DELAY = 0, 1, 2, 3, 4


class _Tensor:
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


class _Boxes:
    """Results.boxes 흉내."""

    def __init__(self, xyxy, conf, cls):
        """K개 박스."""
        self.xyxy, self.conf, self.cls = _Tensor(xyxy), _Tensor(conf), _Tensor(cls)

    def __len__(self):
        """K."""
        return len(self.xyxy.array)


class _Masks:
    """Results.masks 흉내."""

    def __init__(self, data):
        """마스크 데이터(K,h,w)."""
        self.data = _Tensor(data)


class _Result:
    """Results 하나 흉내."""

    def __init__(self, boxes, masks, orig_shape):
        """박스·마스크·원본 크기."""
        self.boxes, self.masks, self.orig_shape = boxes, masks, orig_shape


def bolt_mask(h, w, x0, y=60, head_len=10, head_w=18, shaft_len=34, shaft_w=10):
    """가로 볼트 합성 마스크(머리 왼쪽)."""
    m = np.zeros((h, w), np.float32)
    m[y - head_w // 2:y - head_w // 2 + head_w, x0:x0 + head_len] = 1.0
    m[y - shaft_w // 2:y - shaft_w // 2 + shaft_w, x0 + head_len:x0 + head_len + shaft_len] = 1.0

    return m


class FakeModel:
    """ultralytics YOLO 흉내. predict는 crop 픽셀 값으로 동작을 고른다."""

    task = "segment"
    names = {0: "bolt", 1: "nut"}

    def __init__(self, weights_path):
        """가중치 경로(읽지 않음)."""
        self.weights_path = weights_path

    def predict(self, crop, **kwargs):
        """가짜 추론."""
        code, pos = int(crop[0, 0, 0]), int(crop[0, 0, 1])

        if code == HANG:
            time.sleep(600)
        elif code == DIE:
            os._exit(7)
        elif code == RAISE:
            raise ValueError("fake predict failure")
        elif code == DELAY:
            time.sleep(float(os.environ.get("FAKE_DELAY_S", "0.5")))

        h, w = crop.shape[:2]
        x0 = 20 + pos % 60
        mask = bolt_mask(h, w, x0)
        ys, xs = np.nonzero(mask)
        boxes = _Boxes(np.array([[xs.min(), ys.min(), xs.max(), ys.max()]], np.float32),
                       np.array([0.9], np.float32), np.array([0.0], np.float32))

        return [_Result(boxes, _Masks(mask[None]), (h, w))]


def make(cfg):
    """가짜 모델을 쓰는 실제 AiSegDetector."""
    return AiSegDetector(cfg, model_factory=FakeModel, cuda_device_count=lambda: 1)


class _BadPrepare:
    """준비에서 설정 오류를 내는 detector."""

    def __init__(self, cfg):
        """설정을 받는다."""
        self.cfg = cfg

    def prepare(self):
        """MODEL_CONFIG_INVALID."""
        raise ModelPrepareError(InvalidationKind.MODEL_CONFIG_INVALID, "fake model names mismatch")


class _DiePrepare(_BadPrepare):
    """준비 중 프로세스가 죽는 detector."""

    def prepare(self):
        """os._exit."""
        os._exit(9)


def make_bad_prepare(cfg):
    """준비 실패 detector."""
    return _BadPrepare(cfg)


def make_die_prepare(cfg):
    """준비 중 죽는 detector."""
    return _DiePrepare(cfg)
