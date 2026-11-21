"""AI 추론 프로세스 자식 진입점. `ProcessAiSegDetector`(부모)만 띄운다: `python -m eyetoarm.vision.ai_process_child`.

메시지(부모↔자식, 전용 socketpair):
- `prepare` → `ready`(model_ready 정보·실제 torch CPU 스레드 수) 또는 `prepare_error`(분류·원인)
- `job` → `result`(AiSegResult) 또는 `job_error`(config: 설정 불일치, error: 그 밖)
- `stop` 또는 연결 끊김 → 정리하고 끝냄

SIGINT는 무시하고(부모가 정리 순서를 정함), 부모가 죽으면 SIGTERM을 받도록 PR_SET_PDEATHSIG를 건다.
rclpy는 import하지 않는다. `--factory`는 시험 주입용이며 운영값은 `AiSegDetector`다.
"""
from __future__ import annotations

import argparse
import ctypes
import dataclasses
import importlib
import os
import signal
import sys
from multiprocessing.connection import Connection

from eyetoarm.common.types import InvalidationKind
from eyetoarm.vision.ai_config import load_ai_config
from eyetoarm.vision.ai_process import ShmSlots
from eyetoarm.vision.ai_result import AiJobConfigError, ModelPrepareError

_PR_SET_PDEATHSIG = 1


def _set_parent_death_signal(parent_pid: int) -> None:
    """부모가 죽으면 SIGTERM을 받게 한다(Linux). 걸기 전에 이미 부모가 바뀌었으면 바로 끝낸다."""
    try:
        ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)
    except (OSError, AttributeError) as exc:   # prctl 없는 환경: 연결 끊김(EOF)으로만 끝난다
        print(f"ai_process_child: prctl unavailable: {exc}", file=sys.stderr)

    if os.getppid() != parent_pid:
        sys.exit(0)


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

    _set_parent_death_signal(args.parent_pid)
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
