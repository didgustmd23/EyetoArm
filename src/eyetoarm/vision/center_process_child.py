"""B center 그리기 자식 진입점(#413). `CenterDrawProcess`(부모)만 띄운다: `python -m eyetoarm.vision.center_process_child`.

메시지(부모↔자식, 전용 socketpair):
- 시작하면 `ready`(pid·준비 시간) 또는 `failed`(원인)를 보낸다.
- `config`(카메라 값) → 그리기용 B 단계를 만든다.
- `draw`(공유 메모리의 사진 + B 결과 + 번호표) → 그려서 발행하고 `done`(성공 여부·그리기·발행 시간)을 보낸다.
- `stop` 또는 연결 끊김 → 정리하고 끝낸다.

SIGINT는 무시하고(부모가 정리 순서를 정함), 부모가 죽으면 감시 스레드가 0.5 s 안에 끝낸다.
그리기·발행은 B 공개 함수(`BStage.render_center`·`CenterOverlayPublisher`)를 그대로 부른다. `--factory`는 시험 주입용이다.
"""
from __future__ import annotations

import argparse
import importlib
import mmap
import os
import signal
import sys
import threading
import time
from multiprocessing.connection import Connection
from types import SimpleNamespace

import numpy as np

PARENT_POLL_S = 0.5
NODE_NAME = "b_center_draw"   # A 프로세스에 남는 B 발행기 노드(`b_center_overlay`)와 이름이 겹치지 않게


def _watch_parent(parent_pid: int) -> None:
    """부모가 바뀌면(부모 죽음 → 재부모화) 바로 끝낸다. 그리는 중에도 동작하도록 데몬 스레드로 돈다."""
    if os.getppid() != parent_pid:
        os._exit(0)

    def loop():
        """부모 pid를 주기적으로 확인한다."""
        while True:
            time.sleep(PARENT_POLL_S)

            if os.getppid() != parent_pid:
                os._exit(0)

    threading.Thread(target=loop, name="center-child-parent-watch", daemon=True).start()


def _resolve(spec: str):
    """"모듈:이름" 문자열을 객체로."""
    module, _, name = spec.partition(":")

    if not module or not name:
        raise ValueError(f"factory must be 'module:name': {spec!r}")

    return getattr(importlib.import_module(module), name)


def _no_tf(frame):
    """그리기용 단계는 TF를 묻지 않는다(결과와 `T_W_C`를 부모에게서 받는다)."""
    return None


class DrawBackend:
    """그리기 + 발행. `publisher`는 B `CenterOverlayPublisher`와 같은 `publish(image, frame, keys)`·`close()`를 가진다."""

    def __init__(self, root: str, publisher) -> None:
        """`root`는 B 설정을 읽을 저장소 맨 위 폴더."""
        self._root = root
        self._publisher = publisher
        self._stage = None

    def configure(self, camera) -> None:
        """카메라 값으로 그리기용 B 단계를 (다시) 만든다."""
        from eyetoarm.transforms.b_stage import BStage

        self._stage = BStage.from_config(self._root, camera, _no_tf)

    def draw(self, image: np.ndarray, items: list, guides: bool, stamp_ns, optical_frame, keys: dict) -> tuple[float, float]:
        """한 장을 그려 발행한다. (그리기 ms, 발행 ms)를 돌려준다."""
        from eyetoarm.transforms.b_stage import BOutput

        if self._stage is None:
            raise RuntimeError("draw before config")

        t0 = time.monotonic_ns()
        outputs = [BOutput(result=result, result_dict={}, goal={}, T_W_C=T_W_C) for result, T_W_C in items]
        drawn = self._stage.render_center(image, outputs, guides=guides)
        t1 = time.monotonic_ns()

        # B 발행기는 원본 프레임에서 촬영 시각과 광학 프레임 이름만 읽는다
        self._publisher.publish(drawn, SimpleNamespace(ros_stamp_ns=stamp_ns, optical_frame=optical_frame), keys)
        t2 = time.monotonic_ns()

        return (t1 - t0) / 1e6, (t2 - t1) / 1e6

    def close(self) -> None:
        """발행기를 닫는다."""
        self._publisher.close()


def ros_backend(root: str) -> DrawBackend:
    """운영 구성: B `CenterOverlayPublisher`(영상·번호표 토픽만, 결과 숫자는 부모가 냄)."""
    from eyetoarm.transforms.center_overlay_pub import CenterOverlayPublisher

    return DrawBackend(root, CenterOverlayPublisher(node_name=NODE_NAME, results_topic=None))


def _view(mem: dict, mem_fd: int, nbytes: int, shape) -> np.ndarray:
    """공유 메모리의 사진을 배열로 본다(크기가 바뀌면 다시 붙인다). 복사하지 않는다."""
    if mem.get("size", 0) < nbytes:
        if mem.get("map") is not None:
            mem["map"].close()

        mem["map"] = mmap.mmap(mem_fd, nbytes, prot=mmap.PROT_READ)
        mem["size"] = nbytes

    return np.frombuffer(mem["map"], dtype=np.uint8, count=nbytes).reshape(shape)


def main(argv=None) -> int:
    """자식 루프. 정상 종료 0, 준비 실패 2."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    parser = argparse.ArgumentParser(description="EyetoArm B center draw child (started by CenterDrawProcess)")
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--mem-fd", type=int, required=True)
    parser.add_argument("--factory", required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args(argv)

    _watch_parent(args.parent_pid)
    conn = Connection(args.fd)
    mem: dict = {}
    backend = None
    t0 = time.monotonic_ns()

    try:
        try:
            backend = _resolve(args.factory)(args.root)
        except Exception as exc:   # 준비 실패(ROS 없음 등)는 부모에 알리고 끝낸다
            conn.send(("failed", {"cause": f"{type(exc).__name__}: {exc}"}))
            return 2

        conn.send(("ready", {"pid": os.getpid(), "start_to_ready_ms": round((time.monotonic_ns() - t0) / 1e6, 3)}))

        while True:
            try:
                kind, payload = conn.recv()
            except EOFError:   # 부모가 연결을 닫음: 정리하고 끝낸다
                return 0

            if kind == "stop":
                return 0

            if kind == "config":
                try:
                    backend.configure(payload["camera"])
                except Exception as exc:   # 설정을 못 읽으면 그릴 수 없다: 부모에 알리고 끝낸다
                    conn.send(("failed", {"cause": f"config: {type(exc).__name__}: {exc}"}))
                    return 2
            elif kind == "draw":
                try:
                    image = _view(mem, args.mem_fd, payload["nbytes"], payload["shape"])
                    render_ms, publish_ms = backend.draw(image, payload["items"], payload["guides"],
                                                         payload["stamp_ns"], payload["optical_frame"], payload["keys"])
                    answer = {"req": payload["req"], "ok": True, "render_ms": render_ms, "publish_ms": publish_ms}
                except Exception as exc:   # 한 장 실패는 그 장만 잃는다(부모가 오류로 셈)
                    answer = {"req": payload["req"], "ok": False, "error": f"{type(exc).__name__}: {exc}"}

                image = None   # 공유 메모리에 붙은 배열을 놓는다(크기가 바뀌면 다시 붙일 수 있게)
                conn.send(("done", answer))
            else:
                print(f"center_process_child: unknown message {kind!r}", file=sys.stderr)
                return 2
    except (BrokenPipeError, ConnectionResetError):   # 부모가 답을 받기 전에 끊음: 받을 쪽이 없다
        return 0
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:   # 닫기 실패는 종료를 막지 않는다(기록만)
                print(f"center_process_child: backend close failed: {type(exc).__name__}: {exc}", file=sys.stderr)

        if mem.get("map") is not None:
            try:
                mem["map"].close()
            except BufferError:   # 마지막 사진 배열이 아직 붙어 있음: 프로세스가 끝나면 함께 풀린다
                pass

        conn.close()


if __name__ == "__main__":
    sys.exit(main())
