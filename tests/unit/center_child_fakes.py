"""#413 시험 전용: center 그리기 자식 프로세스에 넣는 가짜 발행기(ROS 없이 파일로 남김).

자식은 `--factory center_child_fakes:file_backend`로 뜬다. 그리기는 실제 B `BStage.render_center`를 그대로 쓰고,
발행만 파일로 바꾼다. 동작은 환경 변수로 정한다.
- `EYETOARM_TEST_CENTER_OUT`: 그린 사진(`<image_seq>.npy`)과 번호표(`published.jsonl`)를 남길 폴더
- `EYETOARM_TEST_CENTER_SLEEP_S`: 한 장마다 쉬는 시간(느린 자식)
- `EYETOARM_TEST_CENTER_DIE_AT`: 이 image_seq(쉼표로 여럿)에서 바로 죽음(종료 코드 9)
- `EYETOARM_TEST_CENTER_HANG_AT`: 이 image_seq에서 멈춤(답하지 않음)
- `EYETOARM_TEST_CENTER_FAIL_AT`: 이 image_seq에서 발행 예외
- `EYETOARM_TEST_CENTER_PREPARE_S`: 준비(발행기 만들기)에 걸리는 시간(`slow_backend`, 부하가 큰 기계의 느린 ROS 준비 흉내)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from eyetoarm.vision.center_process_child import DrawBackend


def _at(name: str) -> int | None:
    """환경 변수의 image_seq(없으면 None)."""
    value = os.environ.get(name)

    return None if value is None else int(value)


def _all_at(name: str) -> set[int]:
    """환경 변수의 image_seq 여럿(쉼표로 나눔, 없으면 빈 집합)."""
    return {int(v) for v in os.environ.get(name, "").split(",") if v.strip()}


class FilePublisher:
    """B `CenterOverlayPublisher`의 `publish`·`close` 자리. 받은 것을 파일로 남긴다."""

    def __init__(self) -> None:
        """출력 폴더를 만든다."""
        self.out = Path(os.environ["EYETOARM_TEST_CENTER_OUT"])
        self.out.mkdir(parents=True, exist_ok=True)

    def publish(self, image, frame, keys=None) -> None:
        """사진과 번호표를 남긴다(주입된 죽음·멈춤·실패·느림 포함)."""
        seq = int(keys["image_seq"])

        if seq in _all_at("EYETOARM_TEST_CENTER_DIE_AT"):
            os._exit(9)

        if seq == _at("EYETOARM_TEST_CENTER_HANG_AT"):
            time.sleep(600)

        if seq == _at("EYETOARM_TEST_CENTER_FAIL_AT"):
            raise RuntimeError("publish down")

        time.sleep(float(os.environ.get("EYETOARM_TEST_CENTER_SLEEP_S", "0")))
        np.save(self.out / f"{seq}.npy", image)

        with open(self.out / "published.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"keys": keys, "stamp_ns": frame.ros_stamp_ns, "optical_frame": frame.optical_frame,
                                "shape": list(image.shape)}) + "\n")

    def close(self) -> None:
        """정상 종료 표시를 남긴다."""
        (self.out / "closed").write_text("closed")


def file_backend(root: str) -> DrawBackend:
    """실제 B 그리기 + 파일 발행기."""
    return DrawBackend(root, FilePublisher())


def slow_backend(root: str) -> DrawBackend:
    """`EYETOARM_TEST_CENTER_PREPARE_S`초 뒤에 준비되는 실제 B 그리기 + 파일 발행기."""
    time.sleep(float(os.environ.get("EYETOARM_TEST_CENTER_PREPARE_S", "0")))

    return file_backend(root)


def broken_backend(root: str):
    """준비 실패(ROS 없음 같은 경우) 대역."""
    raise ImportError("no ros here")
