#!/usr/bin/env python3
"""분류 시험 공급 순서표 생성(generate)·검증(validate) 도구. 순서표는 평가 경로에서만 읽는다.

사용: PYTHONPATH=src python3 scripts/sort_test_sequence.py generate --seed 42 --count 20 \\
      --pattern mixed --out-dir evaluation/ground_truth/sort_sequences
종료 코드: 0 통과, 1 검증 실패, 2 인자·파일 오류. 같은 이름 파일은 덮어쓰지 않는다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eyetoarm.sorteval import sequence as seqmod  # noqa: E402

DEFAULT_OUT = ROOT / "evaluation" / "ground_truth" / "sort_sequences"


def _ratio(text: str | None) -> dict | None:
    """`BOLT_A=1,BOLT_B=2` → dict."""
    if text is None:
        return None

    out = {}

    for part in text.split(","):
        spec, _, weight = part.partition("=")

        if not weight.strip().isdigit():
            raise ValueError(f"bad ratio item {part!r}; expected SPEC=INT")

        out[spec.strip()] = int(weight)

    return out


def _generate(args) -> int:
    """generate 하위 명령."""
    try:
        seq = seqmod.generate_sequence(seed=args.seed, count=args.count, pattern=args.pattern,
                                       ratio=_ratio(args.ratio), include_nut=args.include_nut,
                                       block_len=args.block_len, sequence_id=args.sequence_id,
                                       interval_s=args.interval_s)
        paths = seqmod.write_outputs(seq, args.out_dir)
    except (ValueError, FileExistsError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for ext, path in paths.items():
        print(f"{ext}: {path}")

    print(f"content_sha256: {seq['content_sha256']}")

    return 0


def _validate(args) -> int:
    """validate 하위 명령(JSON 정본 또는 사람이 쓴 CSV)."""
    path = Path(args.path)

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if path.suffix.lower() == ".csv":
        if not args.sequence_id:
            print("error: --sequence-id is required for CSV input", file=sys.stderr)
            return 2

        seq, errors = seqmod.sequence_from_csv(text, sequence_id=args.sequence_id, include_nut=args.include_nut)
    else:
        seq, errors = seqmod.load_sequence_text(text)

    if errors:
        print(f"INVALID {path}")

        for e in errors:
            print(f"  - {e}")

        return 1

    print(f"OK {path} supplies={len(seq['supplies'])} content_sha256={seq['content_sha256']}")

    if path.suffix.lower() == ".csv" and args.out_dir:
        try:
            for ext, p in seqmod.write_outputs(seq, args.out_dir).items():
                print(f"{ext}: {p}")
        except (ValueError, FileExistsError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    return 0


def main(argv=None) -> int:
    """CLI 진입점."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="고정 시드 순서표 생성")
    g.add_argument("--seed", type=int, required=True)
    g.add_argument("--count", type=int, required=True)
    g.add_argument("--pattern", choices=seqmod.PATTERNS, default="shuffled")
    g.add_argument("--ratio", help="예: BOLT_A=1,BOLT_B=1 (기본 1:1, --include-nut이면 NUT=1 추가)")
    g.add_argument("--include-nut", action="store_true", help="너트 행 포함(기본 제외, 사용자 결정 항목)")
    g.add_argument("--block-len", type=int, default=3)
    g.add_argument("--interval-s", type=float, default=None, help="공급 간격 s(기본 null = C 기본값)")
    g.add_argument("--sequence-id")
    g.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    g.set_defaults(func=_generate)

    v = sub.add_parser("validate", help="순서표 검증(JSON 정본 또는 사람이 쓴 CSV)")
    v.add_argument("path")
    v.add_argument("--sequence-id", help="CSV 입력일 때 필수")
    v.add_argument("--include-nut", action="store_true")
    v.add_argument("--out-dir", type=Path, help="CSV 입력이 통과하면 정본 JSON·MD를 여기에 쓴다")
    v.set_defaults(func=_validate)

    args = parser.parse_args(argv)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
