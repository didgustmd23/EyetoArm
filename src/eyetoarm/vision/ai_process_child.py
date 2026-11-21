"""AI 추론 프로세스 자식 진입점. `ProcessAiSegDetector`(부모)만 띄운다: `python -m eyetoarm.vision.ai_process_child`.

메시지(부모↔자식, 전용 socketpair):
- `prepare` → `ready`(model_ready 정보·실제 torch CPU 스레드 수) 또는 `prepare_error`(분류·원인)
- `job` → `result`(AiSegResult) 또는 `job_error`(config: 설정 불일치, error: 그 밖)
- `stop` 또는 연결 끊김 → 정리하고 끝냄

SIGINT는 무시하고(부모가 정리 순서를 정함), 부모가 죽으면 감시 스레드가 0.5 s 안에 끝낸다.
rclpy는 import하지 않는다. `--factory`는 시험 주입용이며 운영값은 `AiSegDetector`다.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import os
import signal
import sys
import threading
import time
from multiprocessing.connection import Connection

from eyetoarm.common.types import InvalidationKind
from eyetoarm.vision.ai_config import load_ai_config
from eyetoarm.vision.ai_process import ShmSlots
from eyetoarm.vision.ai_result import AiJobConfigError, ModelPrepareError

PARENT_POLL_S = 0.5


def _watch_parent(parent_pid: int) -> None:
    """부모가 바뀌면(부모 죽음 → 재부모화) 바로 끝낸다. 작업 중(멈춘 predict 포함)에도 동작하도록 데몬 스레드로 돈다.

    PR_SET_PDEATHSIG는 자식을 띄운 *스레드*가 끝나도 신호를 보내므로(대리 객체는 실행기 worker 스레드에서 띄움) 쓰지 않는다.
    """
    if os.getppid() != parent_pid:
        os._exit(0)

    def loop():
        """부모 pid를 주기적으로 확인한다."""
        while True:
            time.sleep(PARENT_POLL_S)

            if os.getppid() != parent_pid:
                os._exit(0)

    threading.Thread(target=loop, name="infer-child-parent-watch", daemon=True).start()


def _resolve(spec: str):
    """"모듈:이름" 문자열을 객체로."""
    module, _, name = spec.partition(":")

    if not module or not name:
        raise ValueError(f"factory must be 'module:name': {spec!r}")

    return getattr(importlib.import_module(module), name)


def _torch_threads(requested: int | None) -> int | None:
    """torch가 import됐으면 CPU 스레드 수를 정하고(요청 시) 실제 값을 돌려준다. torch가 없으면 None."""
    torch = sys.modules.get("torch")

    if torch is None:
        return None

    if requested is not None:
        torch.set_num_threads(requested)

    return int(torch.get_num_threads())


def _prepare(conn: Connection, factory_spec: str, payload: dict):
    """설정을 다시 읽어 sha256을 맞추고 detector를 준비한다. 성공이면 detector, 실패면 None(오류는 보냄)."""
    def refuse(kind, cause):
        conn.send(("prepare_error", {"kind": InvalidationKind(kind).value, "cause": cause}))

    try:
        cfg = load_ai_config(payload["config_path"])
    except (OSError, ValueError) as exc:
        refuse(InvalidationKind.MODEL_CONFIG_INVALID, f"infer child cannot load AI config: {type(exc).__name__}: {exc}")
        return None

    if cfg.config_sha256 != payload["config_sha256"]:
        refuse(InvalidationKind.MODEL_CONFIG_INVALID, f"infer child config {cfg.config_sha256[:12]} != parent "
                                                      f"{payload['config_sha256'][:12]}")
        return None

    try:
        detector = _resolve(factory_spec)(cfg)
        info = detector.prepare()
    except ModelPrepareError as exc:
        refuse(exc.kind, exc.cause)
        return None
    except Exception as exc:   # 준비 단계의 그 밖 오류는 로딩 실패로 분류해 부모에 알린다
        refuse(InvalidationKind.MODEL_LOAD_FAILED, f"infer child prepare failed: {type(exc).__name__}: {exc}")
        return None

    conn.send(("ready", {"info": info, "torch_threads": _torch_threads(payload.get("torch_threads"))}))

    return detector


def _run_job(conn: Connection, detector, slots: dict, payload: dict) -> None:
    """작업 하나: 슬롯을 복사 없이 보고 추론해 결과나 오류를 보낸다."""
    name, shape = payload["shm"], tuple(payload["shape"])
    shm = slots.get(name)

    if shm is None or shm.shape != shape:
        shm = slots[name] = ShmSlots(name, shape, payload.get("slot_count", 2), create=False)

    frame = dataclasses.replace(payload["meta"], image=shm.views[payload["slot"]])
    ids = {"req_id": payload["req_id"], "job_seq": payload["job_seq"]}

    try:
        result = detector.infer(frame, payload["detector_config_id"])
    except AiJobConfigError as exc:
        conn.send(("job_error", {**ids, "kind": "config", "message": exc.cause}))
        return
    except Exception as exc:   # 작업 하나의 오류는 부모가 INFERENCE_ERROR로 분류한다. 자식은 계속 돈다
        conn.send(("job_error", {**ids, "kind": "error", "message": f"{type(exc).__name__}: {exc}"}))
        return
    finally:
        del frame

    conn.send(("result", {**ids, "result": result}))


def main(argv=None) -> int:
    """자식 루프. 정상 종료 0, 준비 실패 2."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    parser = argparse.ArgumentParser(description="EyetoArm AI inference child (started by ProcessAiSegDetector)")
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--factory", required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args(argv)

    _watch_parent(args.parent_pid)
    conn = Connection(args.fd)
    slots: dict[str, ShmSlots] = {}

    try:
        try:
            kind, payload = conn.recv()
        except EOFError:
            return 0

        if kind != "prepare":
            print(f"ai_process_child: expected prepare, got {kind!r}", file=sys.stderr)
            return 2

        detector = _prepare(conn, args.factory, payload)

        if detector is None:
            return 2

        while True:
            try:
                kind, payload = conn.recv()
            except EOFError:   # 부모가 연결을 닫음: 정리하고 끝낸다
                return 0

            if kind == "stop":
                return 0

            if kind == "job":
                try:
                    _run_job(conn, detector, slots, payload)
                except (BrokenPipeError, ConnectionResetError):   # 부모가 응답 전에 끊음: 받을 쪽이 없다
                    return 0
            else:
                print(f"ai_process_child: unknown message {kind!r}", file=sys.stderr)
                return 2
    finally:
        for shm in slots.values():
            shm.close()

        conn.close()


if __name__ == "__main__":
    sys.exit(main())
