"""AI/OpenCV 동일 조건 비교 표: 검출 방식만 다른 두 취합 결과를 같은 표로 나란히 낸다.

순서표·시드·`include_nut`·시간 기준·실행 메타 `conditions`가 같고 `detector.method`는 달라야 한다.
조건이 다르면 표 머리에 「조건 불일치, 비교 무효」를, 가짜 입력이면 FAKE_INPUT을 붙인다.
운영 채택 판단·가중치는 넣지 않는다.
"""
from __future__ import annotations

from eyetoarm.sorteval.aggregate import OUTCOMES, fmt_cell, fake_banner

_HEADER_KEYS = ("sequence_id", "sequence_sha256", "sequence_seed", "include_nut", "timestamp_base")


def condition_differences(a: dict, b: dict) -> list[str]:
    """동일 조건 위반 목록(빈 목록이면 같은 조건)."""
    ha, hb = a["header"], b["header"]
    diffs = [f"{k}: {ha.get(k)!r} != {hb.get(k)!r}" for k in _HEADER_KEYS if ha.get(k) != hb.get(k)]

    for key in sorted(set(ha["conditions"]) | set(hb["conditions"])):
        va, vb = ha["conditions"].get(key), hb["conditions"].get(key)

        if va != vb:
            diffs.append(f"conditions.{key}: {va!r} != {vb!r}")

    if ha["detector"].get("method") == hb["detector"].get("method"):
        diffs.append(f"detector.method: both {ha['detector'].get('method')!r} (detection method must differ)")

    return diffs


def _metric_rows(s: dict) -> list[tuple[str, object, object]]:
    """(지표, 값, 분모) 목록."""
    d = s["denominator"]
    pr = s["precision_recall"]
    rows = [("분류 정확도", s["accuracy"], d["total_supplies"]),
            ("볼트 분류 정확도", s["bolt_accuracy"], d["bolt_supplies"])]

    for spec, v in pr.items():
        rows.append((f"{spec} 정밀도", v["precision"], v["precision_denominator"]))
        rows.append((f"{spec} 재현율", v["recall"], v["recall_denominator"]))

    rows.append(("실제 상자 투입", s["bin_placement"]["ratio"], s["bin_placement"]["pick_succeeded"]))
    rows += [(f"판정 {o}", s["outcomes"][o], d["total_supplies"]) for o in OUTCOMES]
    rows += [("검출 없이 통과", s["missed"], d["total_supplies"]),
             ("중복 도착(같은 부품)", s["duplicates"]["same_part"], None),
             ("실패 볼트 너트 상자 유입", s["bolts_into_nut_bin"], d["bolt_supplies"])]

    for subject, reasons in s["failures_by_subject"].items():
        rows += [(f"실패 {subject}:{reason}", n, None) for reason, n in reasons.items()]

    for key, v in s["timings"].items():
        rows.append((f"{key} P95 ms", v["p95_ms"], v["n"]))

    return rows


def compare(a: dict, b: dict) -> dict:
    """두 취합 결과 → 비교 사전. 지표 이름 합집합 순서는 a 먼저."""
    diffs = condition_differences(a, b)
    ra = {name: (value, den) for name, value, den in _metric_rows(a)}
    rb = {name: (value, den) for name, value, den in _metric_rows(b)}
    names = list(ra) + [n for n in rb if n not in ra]
    table = []

    for name in names:
        # 실패 사유 건수는 한쪽에 없으면 0건이다(값이 없는 것이 아님).
        absent = (0, None) if name.startswith("실패 ") else (None, None)
        va, da = ra.get(name, absent)
        vb, db = rb.get(name, absent)
        diff = vb - va if isinstance(va, (int, float)) and isinstance(vb, (int, float)) else None
        table.append({"metric": name, "a": va, "a_denominator": da, "b": vb, "b_denominator": db, "b_minus_a": diff})

    return {
        "fake_input": bool(a["header"]["fake_input"] or b["header"]["fake_input"]),
        "same_conditions": not diffs,
        "condition_differences": diffs,
        "a": {"eval_run_id": a["header"]["eval_run_id"], "detector": a["header"]["detector"],
              "status": a["header"]["status"]},
        "b": {"eval_run_id": b["header"]["eval_run_id"], "detector": b["header"]["detector"],
              "status": b["header"]["status"]},
        "sequence_id": a["header"]["sequence_id"],
        "sequence_sha256": a["header"]["sequence_sha256"],
        "table": table,
        "note": "비교 표만 낸다. 운영 채택 판단은 이 도구의 범위가 아니다(A 상세설계 §7.1, 3주차 말 결정).",
    }


def to_markdown(c: dict) -> str:
    """비교 MD 표. 가짜 입력·조건 불일치 표시를 머리에 둔다."""
    ma, mb = c["a"]["detector"].get("method"), c["b"]["detector"].get("method")
    lines = fake_banner(c["fake_input"])

    if not c["same_conditions"]:
        lines += ["> **조건 불일치, 비교 무효** — 아래 항목이 다르다.", ""]
        lines += [f"> - {d}" for d in c["condition_differences"]]
        lines.append("")

    lines += [
        f"# 검출 방식 비교: {ma} vs {mb}",
        "",
        f"- 순서표 `{c['sequence_id']}` sha256 `{c['sequence_sha256'][:16]}…`",
        f"- A열 `{c['a']['eval_run_id']}` ({c['a']['status']}), B열 `{c['b']['eval_run_id']}` ({c['b']['status']})",
        f"- 같은 조건: {'예' if c['same_conditions'] else '아니오'} · {c['note']}",
        "",
        f"| 지표 | {ma} | 분모 | {mb} | 분모 | 차이({mb}−{ma}) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines += [f"| {r['metric']} | {fmt_cell(r['a'])} | {fmt_cell(r['a_denominator'])} | {fmt_cell(r['b'])} | "
              f"{fmt_cell(r['b_denominator'])} | {fmt_cell(r['b_minus_a'])} |" for r in c["table"]]

    return "\n".join(lines) + "\n"
