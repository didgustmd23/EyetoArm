#!/usr/bin/env python3
"""분류 시험 결과 취합(aggregate)·AI/OpenCV 비교(compare)·가짜 입력 생성(fake) 도구.

사용: PYTHONPATH=src python3 scripts/aggregate_sort_test.py aggregate --sequence seq.json \\
      --events events.jsonl --meta run_meta.json --out-dir evaluation/results/sort/<run_id>
종료 코드: 0 성공, 1 입력 거부(검증 실패), 2 인자·파일 오류. 출력 파일은 덮어쓰지 않는다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eyetoarm.sorteval import aggregate as agg  # noqa: E402
from eyetoarm.sorteval import compare as cmp  # noqa: E402
from eyetoarm.sorteval import records as rec  # noqa: E402
from eyetoarm.sorteval import sequence as seqmod  # noqa: E402


def _write_new(out_dir: Path, files: dict[str, str]) -> None:
    """새 파일만 쓴다(있으면 FileExistsError)."""
    existing = [str(out_dir / n) for n in files if (out_dir / n).exists()]

    if existing:
        raise FileExistsError(f"refusing to overwrite: {existing}")

    out_dir.mkdir(parents=True, exist_ok=True)

    for name, text in files.items():
        (out_dir / name).write_text(text, encoding="utf-8")
        print(f"wrote {out_dir / name}")


def _dump(obj) -> str:
    """JSON(들여쓰기 2)."""
    return json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _aggregate(args) -> int:
    """aggregate 하위 명령."""
    try:
        seq = seqmod.load_sequence(args.sequence)
        records, errors = rec.parse_events_jsonl(Path(args.events).read_text(encoding="utf-8"))
        meta, meta_errors = rec.adapt_run_meta(seqmod.loads_strict(Path(args.meta).read_text(encoding="utf-8")))
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    errors += meta_errors

    if errors:
        print("REJECTED input:")

        for e in errors:
            print(f"  - {e}")

        return 1

    try:
        summary = agg.aggregate(seq, records, meta)
        _write_new(Path(args.out_dir), {"summary.json": _dump(summary), "summary.md": agg.to_markdown(summary),
                                        "per_supply.csv": agg.per_supply_csv(summary)})
    except ValueError as exc:
        print(f"REJECTED: {exc}")
        return 1
    except (FileExistsError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 0


def _compare(args) -> int:
    """compare 하위 명령."""
    try:
        a = seqmod.loads_strict(Path(args.a).read_text(encoding="utf-8"))
        b = seqmod.loads_strict(Path(args.b).read_text(encoding="utf-8"))

        for s in (a, b):
            if s.get("format") != agg.FORMAT:
                raise ValueError(f"not a sort summary: format={s.get('format')!r}")

        c = cmp.compare(a, b)
        _write_new(Path(args.out_dir), {"compare.json": _dump(c), "compare.md": cmp.to_markdown(c)})
    except (ValueError, KeyError, FileExistsError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"same_conditions={c['same_conditions']} fake_input={c['fake_input']}")

    return 0


def _fake(args) -> int:
    """fake 하위 명령: 가짜 events.jsonl·run_meta.json."""
    try:
        seq = seqmod.load_sequence(args.sequence)
        outcomes = {}

        for item in filter(None, (args.outcomes or "").split(",")):
            idx, _, kind = item.partition("=")
            outcomes[int(idx)] = kind.strip()

        events = rec.fake_events(seq, outcomes)
        meta = rec.fake_run_meta(seq, method=args.method)
        _write_new(Path(args.out_dir), {"events.jsonl": rec.events_to_jsonl(events), "run_meta.json": _dump(meta)})
    except (ValueError, FileExistsError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 0


def main(argv=None) -> int:
    """CLI 진입점."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("aggregate")
    a.add_argument("--sequence", required=True)
    a.add_argument("--events", required=True)
    a.add_argument("--meta", required=True)
    a.add_argument("--out-dir", required=True)
    a.set_defaults(func=_aggregate)

    c = sub.add_parser("compare")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--out-dir", required=True)
    c.set_defaults(func=_compare)

    f = sub.add_parser("fake")
    f.add_argument("--sequence", required=True)
    f.add_argument("--outcomes", help=f"예: 3=isolated,5=b_skip (종류: {', '.join(rec.FAKE_OUTCOMES)})")
    f.add_argument("--method", default="opencv")
    f.add_argument("--out-dir", required=True)
    f.set_defaults(func=_fake)

    args = parser.parse_args(argv)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
