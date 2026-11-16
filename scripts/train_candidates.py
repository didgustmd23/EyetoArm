#!/usr/bin/env python3
"""AI 후보 학습기 CLI(학습기 설계 r2). 부모는 torch·ultralytics를 import하지 않는다(학습·검사 자식만).

하위 명령(저장소 루트 기준, 설정 기본값 `config/training/s1_common.json`·`s1_candidates.json`):
- `check`: 설정 검증·공식 가중치 확인(내려받지 않음)·데이터 fingerprint 비교. 종료 0 정상, 1 가중치 문제, 2 설정·데이터 오류.
- `prepare-weights [--allow-download]`: 공식 가중치 확인·(명시할 때만) 내려받기. 부모가 전체 기한을 쥔 자식 다운로드.
- `train RUN_SET`: 후보를 순서대로 새로 학습(`outputs/training/RUN_SET/`). 종료 0/1/2(설계 §5).
- `resume RUN_SET EXP_ID`: 그 실험의 weights/last.pt에서 이어 학습(재개 (a)~(d) 확인).
- `status RUN_SET`: lock·job_state·자기 작업 프로세스·에포크 수(JSON).
- `recover RUN_SET [--kill-own]`: 명시적 복구(토큰으로 확인한 자기 작업만 신호, 조건이 맞으면 lock 해제).
- `register-legacy RUN_SET [--source outputs/s1_candidates]`: 기존 S1 결과를 검증해 새 run_set에 등록(원본 무변, 학습 없음).
- `eval RUN_SET`: 완료 후보의 best.pt를 평가(`--eval-config`, 기본 `config/training/s1_eval.json`). 크기별 시간 자식 → 정확도 자식.
숨은 명령 `_child-train`·`_child-inspect`·`_child-eval-timing`·`_child-eval-accuracy`는 부모가 띄우는 자식이다.

주의: ultralytics의 `exist_ok`는 True로 넘긴다. 실험 폴더는 부모가 `os.mkdir`로 원자적으로 먼저 만들고(있으면 거부),
ultralytics가 그 폴더에 쓰게 하려는 것이다(False면 `name2`로 새 폴더를 만든다). 덮어쓰기 방지는 부모의 mkdir이 맡는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT = os.path.abspath(__file__)
sys.path.insert(0, os.path.join(ROOT, "src"))

from eyetoarm.training.config import ConfigError, load_training_config, thaw  # noqa: E402
from eyetoarm.training.fingerprint import compare_fingerprint, compute_fingerprint  # noqa: E402
from eyetoarm.training.weights import WeightsError, check_weights, prepare_weights  # noqa: E402

INSPECT_TIMEOUT_S = 120


def build_parser() -> argparse.ArgumentParser:
    """명령행 해석기."""
    p = argparse.ArgumentParser(description="AI 후보 학습기(공식 가중치·공통 조건·순차 실행·감시)")
    p.add_argument("--repo-root", default=ROOT)
    p.add_argument("--common", default="config/training/s1_common.json")
    p.add_argument("--candidates", default="config/training/s1_candidates.json")
    p.add_argument("--eval-config", default="config/training/s1_eval.json")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    pw = sub.add_parser("prepare-weights")
    pw.add_argument("--allow-download", action="store_true")
    pw.add_argument("--timeout", type=float, default=300.0)
    t = sub.add_parser("train")
    t.add_argument("run_set")
    r = sub.add_parser("resume")
    r.add_argument("run_set")
    r.add_argument("exp_id")
    s = sub.add_parser("status")
    s.add_argument("run_set")
    rc = sub.add_parser("recover")
    rc.add_argument("run_set")
    rc.add_argument("--kill-own", action="store_true")
    rl = sub.add_parser("register-legacy")
    rl.add_argument("run_set")
    rl.add_argument("--source", default="outputs/s1_candidates")
    ev = sub.add_parser("eval")
    ev.add_argument("run_set")
    et = sub.add_parser("_child-eval-timing")
    et.add_argument("--weights", required=True)
    et.add_argument("--imgsz", type=int, required=True)
    et.add_argument("--out", required=True)
    ea = sub.add_parser("_child-eval-accuracy")
    ea.add_argument("--weights", required=True)
    ea.add_argument("--out", required=True)
    ea.add_argument("--val-dir", required=True)
    ct = sub.add_parser("_child-train")
    ct.add_argument("--exp-id", required=True)
    ct.add_argument("--exp-dir", required=True)
    ct.add_argument("--resume-from")
    ci = sub.add_parser("_child-inspect")
    ci.add_argument("path")
    return p


def _path(root: str, rel: str) -> str:
    """저장소 기준 경로."""
    return rel if os.path.isabs(rel) else os.path.join(root, rel)


def load(args):
    """설정을 읽는다."""
    return load_training_config(_path(args.repo_root, args.common), _path(args.repo_root, args.candidates))


def git_info(root: str) -> dict:
    """git commit·dirty(제한시간 5 s, 실패하면 None)."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5)
        st = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=5)
        return {"commit": head.stdout.strip() or None, "dirty": bool(st.stdout.strip())}
    except (OSError, subprocess.TimeoutExpired):
        return {"commit": None, "dirty": None}


def child_argv(args, cand, exp_dir: str, resume_from: str | None) -> list:
    """학습 자식 argv."""
    argv = [sys.executable, SCRIPT, "--repo-root", args.repo_root, "--common", args.common, "--candidates",
            args.candidates, "_child-train", "--exp-id", cand.exp_id, "--exp-dir", exp_dir]
    if resume_from:
        argv += ["--resume-from", resume_from]
    return argv


def eval_child_argv(args, kind: str, weights: str, imgsz, out_path: str, attempt_dir: str) -> list:
    """평가 자식 argv(시간: 크기 하나, 정확도: 계획 전체)."""
    base = [sys.executable, SCRIPT, "--repo-root", args.repo_root, "--common", args.common, "--candidates",
            args.candidates, "--eval-config", args.eval_config]
    if kind == "timing":
        return base + ["_child-eval-timing", "--weights", weights, "--imgsz", str(imgsz), "--out", out_path]
    return base + ["_child-eval-accuracy", "--weights", weights, "--out", out_path, "--val-dir",
                   os.path.join(attempt_dir, "val")]


def inspect_checkpoint(path: str, env: dict | None = None) -> dict:
    """체크포인트 메타를 자식(torch)에서 읽는다(제한시간). 없으면 exists False."""
    if not os.path.exists(path):
        return {"exists": False}
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    try:
        out = subprocess.run([sys.executable, SCRIPT, "_child-inspect", path], capture_output=True, text=True,
                             timeout=INSPECT_TIMEOUT_S, env={**os.environ, **(env or {})})
        meta = json.loads(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 else {}
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        meta = {}
    return {"exists": True, "sha256": sha, "loadable": bool(meta.get("loadable")), "epoch": meta.get("epoch"),
            "optimizer_none": meta.get("optimizer_none"), "train_args": meta.get("train_args", {})}


def cmd_check(args) -> int:
    """설정·가중치·fingerprint 확인."""
    try:
        cfg = load(args)
    except (ConfigError, OSError) as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 2
    weights_dir = _path(args.repo_root, cfg.common.weights_dir)
    weights = {c.exp_id: check_weights(c, weights_dir).status for c in cfg.candidates}
    fp = cfg.common.data["expected_fingerprint"]
    diffs = [] if fp is None else compare_fingerprint(
        thaw(fp), compute_fingerprint(args.repo_root, cfg.common.data["data_yaml"],
                                      list(cfg.common.data["split_lists"])))
    print(json.dumps({"ok": not diffs and all(v == "ok" for v in weights.values()), "config_id": cfg.config_id,
                      "kind": cfg.common.kind, "weights": weights, "fingerprint_diffs": diffs[:20]},
                     ensure_ascii=False, indent=1))
    return 2 if diffs else (0 if all(v == "ok" for v in weights.values()) else 1)


def cmd_prepare_weights(args) -> int:
    """공식 가중치 준비."""
    from eyetoarm.training.runner import StopFlag
    cfg = load(args)
    stop = StopFlag(install=True)
    rc, out = 0, {}
    try:
        for c in cfg.candidates:
            try:
                out[c.exp_id] = prepare_weights(c, _path(args.repo_root, cfg.common.weights_dir),
                                                allow_download=args.allow_download, timeout_s=args.timeout,
                                                should_stop=stop.is_set)
            except WeightsError as exc:
                out[c.exp_id] = {"error": str(exc), "child": exc.child}
                rc = 1
            if stop.is_set():
                rc = 1
                break
    finally:
        stop.restore()
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return rc


def _runner_kwargs(args, cfg):
    """train·resume 공통 인자."""
    from eyetoarm.training.runner import StopFlag
    env = thaw(cfg.common.required_env)
    return {"repo_root": args.repo_root, "argv": sys.argv[1:],
            "child_factory": lambda cand, exp_dir, resume_from: child_argv(args, cand, exp_dir, resume_from),
            "inspect_fn": lambda path: inspect_checkpoint(path, env), "stop": StopFlag(install=True),
            "versions": {"python": platform.python_version()}, "git": git_info(args.repo_root)}


def _report(res) -> int:
    """결과 출력."""
    print(json.dumps({"rc": res.rc, "run_set_dir": res.run_set_dir, "candidates": res.candidates,
                      "errors": res.errors, "stop_reason": res.stop_reason,
                      "unreaped": [p.pid for p in res.unreaped]}, ensure_ascii=False, indent=1))
    return res.rc


def cmd_train(args) -> int:
    """새 학습."""
    from eyetoarm.training.runner import run_train
    cfg = load(args)
    kw = _runner_kwargs(args, cfg)
    try:
        return _report(run_train(cfg, run_set=args.run_set, **kw))
    finally:
        kw["stop"].restore()


def cmd_resume(args) -> int:
    """재개."""
    from eyetoarm.training.runner import run_resume
    cfg = load(args)
    kw = _runner_kwargs(args, cfg)
    try:
        return _report(run_resume(cfg, run_set=args.run_set, exp_id=args.exp_id, **kw))
    finally:
        kw["stop"].restore()


def cmd_status(args) -> int:
    """상태."""
    from eyetoarm.training.runner import status
    cfg = load(args)
    print(json.dumps(status(os.path.join(args.repo_root, cfg.common.outputs_dir, args.run_set)),
                     ensure_ascii=False, indent=1))
    return 0


def cmd_recover(args) -> int:
    """복구."""
    from eyetoarm.training.runner import recover
    cfg = load(args)
    rep = recover(os.path.join(args.repo_root, cfg.common.outputs_dir, args.run_set), kill_own=args.kill_own,
                  grace_s=cfg.common.watch["kill_grace_s"])
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    return 0 if rep["lock_released"] else 1


def child_train(args) -> int:
    """학습 자식(여기서만 ultralytics를 import). 적용값(applied.json)·진행(progress.json)을 콜백으로 쓴다."""
    cfg = load(args)
    cand = next(c for c in cfg.candidates if c.exp_id == args.exp_id)
    for k, v in thaw(cfg.common.required_env).items():
        os.environ[k] = v
    import torch
    import ultralytics
    from ultralytics import YOLO

    exp_dir = os.path.abspath(args.exp_dir)

    def write(name, obj):
        """기록 파일(원자 교체)."""
        from eyetoarm.training.layout import write_json_atomic
        write_json_atomic(os.path.join(exp_dir, name), obj)

    def on_train_start(trainer):
        """실제 적용값."""
        opt = trainer.optimizer
        g = opt.param_groups[0] if opt is not None and opt.param_groups else {}
        write("applied.json", {"optimizer": type(opt).__name__ if opt is not None else None,
                               "lr": g.get("lr"), "momentum": g.get("momentum", g.get("betas")),
                               "param_groups": len(opt.param_groups) if opt is not None else None,
                               "amp": bool(getattr(trainer, "amp", False)), "batch": trainer.batch_size,
                               "imgsz": trainer.args.imgsz, "epochs": trainer.epochs,
                               "start_epoch": trainer.start_epoch, "torch": torch.__version__,
                               "ultralytics": ultralytics.__version__,
                               "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})

    def on_train_end(trainer):
        """진행(마지막 epoch)."""
        write("progress.json", {"final_epoch": trainer.epoch + 1, "epochs": trainer.epochs,
                                "finished": trainer.epoch + 1 == trainer.epochs})

    model = YOLO(args.resume_from or os.path.join(args.repo_root, cfg.common.weights_dir, cand.weights_file))
    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_end", on_train_end)
    if args.resume_from:
        model.train(resume=True, data=cfg.common.data["data_yaml"])
    else:
        model.train(data=cfg.common.data["data_yaml"], project=os.path.dirname(exp_dir),
                    name=os.path.basename(exp_dir), exist_ok=True, verbose=True, **cfg.train_args())
    return 0


def child_inspect(args) -> int:
    """체크포인트 메타 자식(torch.load, CPU)."""
    import torch
    keys = ("data", "imgsz", "batch", "epochs", "seed", "deterministic", "workers", "device", "project", "name",
            "model", "hsv_v")
    try:
        ck = torch.load(args.path, map_location="cpu", weights_only=False)
        ta = ck.get("train_args") or {}
        meta = {"loadable": True, "epoch": ck.get("epoch"), "optimizer_none": ck.get("optimizer") is None,
                "train_args": {k: ta.get(k) for k in keys}}
    except Exception as exc:
        meta = {"loadable": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(meta, default=str))
    return 0


def cmd_register_legacy(args) -> int:
    """기존 결과 등록."""
    from eyetoarm.training.eval_runner import register_legacy
    cfg = load(args)
    env = thaw(cfg.common.required_env)
    res = register_legacy(cfg, repo_root=args.repo_root, source_dir=args.source, run_set=args.run_set,
                          inspect_fn=lambda path: inspect_checkpoint(path, env), argv=sys.argv[1:])
    print(json.dumps({"rc": res.rc, "run_set_dir": res.run_set_dir, "candidates": res.candidates,
                      "errors": res.errors}, ensure_ascii=False, indent=1))
    return res.rc


def cmd_eval(args) -> int:
    """평가."""
    from eyetoarm.training.eval_runner import run_eval
    from eyetoarm.training.evaluation import load_eval_config
    from eyetoarm.training.runner import StopFlag
    cfg = load(args)
    ecfg = load_eval_config(_path(args.repo_root, args.eval_config))
    stop = StopFlag(install=True)
    try:
        return _report(run_eval(cfg, ecfg, repo_root=args.repo_root, run_set=args.run_set, argv=sys.argv[1:],
                                child_factory=lambda kind, w, sz, out, adir: eval_child_argv(args, kind, w, sz, out,
                                                                                             adir),
                                stop=stop))
    finally:
        stop.restore()


def _eval_env(args):
    """평가 자식 공통: 설정·환경 변수."""
    cfg = load(args)
    for k, v in thaw(cfg.common.required_env).items():
        os.environ[k] = v
    from eyetoarm.training.evaluation import load_eval_config
    return load_eval_config(_path(args.repo_root, args.eval_config))


def child_eval_timing(args) -> int:
    """시간 자식(크기 하나): 영상을 미리 읽고 → 준비(import·YOLO 생성) → cold → 워밍업 → 본 측정. 정확도 평가는 하지 않는다."""
    ecfg = _eval_env(args)
    import time as _time
    import cv2
    from eyetoarm.training.evaluation import images_in, run_timing
    from eyetoarm.training.layout import write_json_atomic

    gz = ecfg.gazebo
    images = [cv2.imread(p) for p in images_in(os.path.join(args.repo_root, gz["data_dir"]), gz["split"])]
    if any(im is None for im in images):
        raise ValueError("unreadable timing image")
    state, calls = {}, [0]

    def prepare():
        """모델 준비(측정 대상): torch·ultralytics import와 YOLO 생성. predict 호출 계수기를 붙인다."""
        import torch
        from ultralytics import YOLO
        state["torch"] = torch
        model = YOLO(args.weights)
        orig = model.predict

        def counted(*a, **k):
            """호출 누계."""
            calls[0] += 1
            return orig(*a, **k)

        model.predict = counted
        return model

    def cuda():
        """CUDA 사용 여부."""
        return state["torch"].cuda.is_available()

    out = run_timing(
        prepare=prepare, images=images,
        predict=lambda m, img: m.predict(img, imgsz=args.imgsz, conf=ecfg.conf, device=ecfg.device,
                                         retina_masks=True, verbose=False),
        sync=lambda: state["torch"].cuda.synchronize() if cuda() else None,
        peak_reset=lambda: state["torch"].cuda.reset_peak_memory_stats() if cuda() else None,
        peak_read=lambda: int(state["torch"].cuda.max_memory_allocated()) if cuda() else None,
        clock_ns=_time.perf_counter_ns, warmup_n=ecfg.timing["warmup_n"], measure_n=ecfg.timing["measure_n"],
        calls_seen=lambda: calls[0])
    import ultralytics
    out.update(imgsz=args.imgsz, pid=os.getpid(), weights=args.weights,
               env={"torch": state["torch"].__version__, "ultralytics": ultralytics.__version__,
                    "gpu": state["torch"].cuda.get_device_name(0) if cuda() else None})
    write_json_atomic(args.out, out)
    return 0


def child_eval_accuracy(args) -> int:
    """정확도 자식: 모델 하나로 계획(공개 test → Gazebo 640·1280) 순서대로 evaluate()·val mAP50."""
    ecfg = _eval_env(args)
    from pathlib import Path
    sys.path.insert(0, os.path.join(args.repo_root, "scripts"))
    from eval_size_recall import evaluate
    from ultralytics import YOLO
    from eyetoarm.training.evaluation import run_accuracy
    from eyetoarm.training.layout import write_json_atomic

    root = Path(args.repo_root)
    model = YOLO(args.weights)

    def ev(m, data_dir, split, imgsz):
        """기존 일대일 매칭 평가."""
        return evaluate(m, root / data_dir, split, imgsz, ecfg.conf, ecfg.iou, 1.0, ecfg.edges_px, ecfg.device)

    def val(m, data_dir, split, imgsz, key):
        """ultralytics val mAP50(절대 경로 project)."""
        v = m.val(data=str(root / data_dir / "dataset.yaml"), split=split, imgsz=imgsz, device=ecfg.device,
                  plots=False, verbose=False, batch=ecfg.val["batch"], workers=ecfg.val["workers"],
                  project=os.path.abspath(args.val_dir), name=key, exist_ok=True)
        return {"map50_box": float(v.box.map50), "map50_mask": float(v.seg.map50)}

    out = run_accuracy(model=model, plan=ecfg.accuracy_plan(), evaluate_fn=ev, val_fn=val)
    write_json_atomic(args.out, out)
    return 0


COMMANDS = {"check": cmd_check, "prepare-weights": cmd_prepare_weights, "train": cmd_train, "resume": cmd_resume,
            "status": cmd_status, "recover": cmd_recover, "register-legacy": cmd_register_legacy, "eval": cmd_eval,
            "_child-train": child_train, "_child-inspect": child_inspect, "_child-eval-timing": child_eval_timing,
            "_child-eval-accuracy": child_eval_accuracy}


def main(argv=None) -> int:
    """진입점. 설정·사전 입력 오류는 traceback 대신 error JSON과 종료 2(학습 중 예외는 실행기가 정리·보고한다)."""
    args = build_parser().parse_args(argv)
    args.repo_root = os.path.abspath(args.repo_root)
    try:
        return COMMANDS[args.cmd](args)
    except (ConfigError, OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    sys.exit(main())
