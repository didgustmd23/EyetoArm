#!/usr/bin/env python3
"""A 통신 시험 도구 진입점. D `TcpClient`로 로컬 모의 상대(C `TcpServer`, 포트 0)에서 S1~S4를 실행한다.

사용: PYTHONPATH=src python3 scripts/run_comm_test.py --peer local_mock --scenario all --test-mode

`--test-mode`가 없으면 장애 주입 케이스는 건너뛴다. 종료 코드: 0 기대 일치, 1 불일치·정리 실패, 2 인자 거부.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eyetoarm.commtest.aggregate import summarize, to_markdown  # noqa: E402
from eyetoarm.commtest.local_peer import LocalPeer  # noqa: E402
from eyetoarm.commtest.records import make_run_dir, write_json, write_jsonl  # noqa: E402
from eyetoarm.commtest.runner import TEST_CLIENT_TIMING, Runner  # noqa: E402
from eyetoarm.commtest.scenarios import C_COOPERATION, SCENARIOS  # noqa: E402


def _git() -> dict:
    """커밋·dirty(실패하면 None)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True,
                                    timeout=10).stdout.strip())
        return {"commit": commit or None, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _sha256sums(run_dir: Path) -> None:
    """폴더 안 파일의 SHA256SUMS."""
    lines = []

    for p in sorted(run_dir.iterdir()):
        if p.is_file() and p.name != "SHA256SUMS":
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")

    (run_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--peer", choices=("local_mock", "c_server"), default="local_mock")
    ap.add_argument("--scenario", default="all", help="S1|S2|S3|S4|all(쉼표로 여러 개)")
    ap.add_argument("--n", type=int, default=100, help="S1 MOVE 횟수(기본 100)")
    ap.add_argument("--test-mode", action="store_true", help="장애 주입(같은 ID 재송신·줄 삼킴·단절) 허용")
    ap.add_argument("--out-root", default=str(ROOT / "outputs" / "comm_test"))
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)

    if args.peer == "c_server":
        print("--peer c_server is out of scope for #183 (real C server/robot); refused", file=sys.stderr)
        return 2

    names = SCENARIOS if args.scenario == "all" else tuple(x.strip() for x in args.scenario.split(","))

    if any(n not in SCENARIOS for n in names) or not 1 <= args.n <= 10000:
        print(f"bad --scenario {args.scenario!r} or --n {args.n}", file=sys.stderr)
        return 2

    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + uuid.uuid4().hex[:6]
    run_dir = make_run_dir(args.out_root, run_id)
    peer = LocalPeer(ROOT)
    host, port = peer.start()
    started = time.time()
    peer_errors: list[str] = []

    try:
        runner = Runner(host=host, port=port, peer=peer, peer_kind="local_mock", scenarios=names, n=args.n,
                        test_mode=args.test_mode, events_dir=run_dir)
        out = asyncio.run(runner.run())
    finally:
        peer_errors = peer.close()

    out.cleanup_errors += peer_errors
    meta = {"run_id": run_id, "started_wall": started, **_git(), "python": sys.version.split()[0],
            "platform": platform.platform(), "protocol": "1.1", "peer": "local_mock",
            "peer_note": "수정하지 않은 C TcpServer + 가짜 readiness·executor, 127.0.0.1 포트 0(C 비공개 _handle_client)",
            "peer_port": port, "client_timing_ms": TEST_CLIENT_TIMING,
            "client_timing_note": "시험용 단축값(TCP §9 초기값과 다름)", "executor_settings": out.executor_settings,
            "scenarios": list(names), "n": args.n, "test_mode": args.test_mode, "argv": sys.argv[1:],
            "c_cooperation_defined_not_run": [list(c) for c in C_COOPERATION],
            "skipped_cases": out.skipped, "callback_errors": out.callback_errors, "tracker_errors": out.tracker_errors,
            "cleanup_errors": out.cleanup_errors}
    summary = summarize(out.rows, peer="local_mock", executor_settings=out.executor_settings, skipped=out.skipped)
    summary.update(cleanup_errors=out.cleanup_errors, callback_errors=out.callback_errors)
    write_json(run_dir / "run_meta.json", meta)
    write_jsonl(run_dir / "commands.jsonl", out.rows)
    write_jsonl(run_dir / "injector.jsonl", out.injector_log)
    write_json(run_dir / "summary.json", summary)
    (run_dir / "summary.md").write_text(to_markdown(summary), encoding="utf-8")
    _sha256sums(run_dir)
    ok = summary["all_match"] and not out.cleanup_errors and not out.callback_errors and not out.tracker_errors
    print(json.dumps({"run_dir": str(run_dir), "rows": len(out.rows), "all_match": summary["all_match"],
                      "skipped": out.skipped, "cleanup_errors": out.cleanup_errors}, ensure_ascii=False))

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
