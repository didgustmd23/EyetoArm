"""통신 시험 취합(#183 계획 r0 §5). 순수.

- 케이스별 기대 일치·불일치·건너뜀, TCP §8.4 수락 4분류 건수·비율, 지연 분위수(최근접 순위), 거부·실패 코드별 건수.
- 계획서 목표(통신 100회, P95 300 ms)는 **참조값**으로만 옆에 둔다. 판정하지 않는다.
- `local_mock` 결과의 시간은 가짜 executor 설정에 좌우되므로 그 설정을 시간 표 바로 옆에 적는다(reviewer 보완 2). 실서버
  시간과 비교하는 문장을 쓰지 않는다. A 영상처리 시간·GUI 응답 시간과 섞지 않는다(A §10).
"""
from __future__ import annotations

import math
from collections import Counter

ACCEPTANCE = ("first_ack", "accepted_after_resend", "accepted_by_status", "unconfirmed")
REFERENCE = {"comm_count_target": 100, "p95_ms_target": 300,
             "note": "계획서 목표 참조값(판정하지 않음). 이 결과는 로컬 모의 상대 측정이다"}
LATENCY_SOURCE = "도구 listener time.monotonic_ns(송신 TX 행 기록 시점 → 그 ID의 첫/최종 RX STATUS 행 기록 시점)"


def percentile(values, q: float) -> float | None:
    """최근접 순위 분위수(q는 0~100). 빈 목록이면 None. 순위 = ceil(q/100·n), 최소 1."""
    xs = sorted(v for v in values if v is not None)

    if not xs:
        return None

    if not 0 <= q <= 100:
        raise ValueError(f"q must be within [0, 100]: {q}")

    rank = max(1, math.ceil(q / 100 * len(xs)))

    return xs[rank - 1]


def _stats(values) -> dict:
    """N·P50·P95·최대."""
    xs = [v for v in values if v is not None]

    return {"n": len(xs), "p50": percentile(xs, 50), "p95": percentile(xs, 95), "max": max(xs) if xs else None}


def summarize(rows: list[dict], *, peer: str, executor_settings: dict, skipped: list[str] | None = None) -> dict:
    """명령 기록 행 → 취합 사전."""
    cases: dict[str, dict] = {}

    for r in rows:
        c = cases.setdefault(r["case"], {"scenario": r["scenario"], "rows": 0, "match": 0, "mismatch": 0, "skipped": 0})
        c["rows"] += 1
        c[r["outcome"]] = c.get(r["outcome"], 0) + 1

    for case_id in skipped or []:
        cases.setdefault(case_id, {"scenario": case_id[:2], "rows": 0, "match": 0, "mismatch": 0, "skipped": 0})
        cases[case_id]["skipped"] += 1

    client = [r for r in rows if r["via"] == "client"]
    acc = Counter(r["acceptance"] for r in client if r["acceptance"] is not None)
    denom = sum(acc[k] for k in ACCEPTANCE)
    s1 = [r for r in client if r["scenario"] == "S1"]

    return {
        "peer": peer,
        "peer_note": "로컬 모의 상대(C TcpServer + 가짜 executor). 실제 C 서버·로봇 결과가 아니다" if peer == "local_mock" else peer,
        "cases": cases,
        "acceptance": {k: acc.get(k, 0) for k in ACCEPTANCE},
        "acceptance_ratio": {k: (acc.get(k, 0) / denom if denom else None) for k in ACCEPTANCE},
        "acceptance_denominator": denom,
        "first_ack_kpi_note": "통신 KPI 「최초 수락 ACK」는 first_ack만 센다(TCP §8.4)",
        "latency_ms": {"S1_ack": _stats(r["ack_latency_ms"] for r in s1),
                       "S1_final": _stats(r["final_latency_ms"] for r in s1)},
        "latency_source": LATENCY_SOURCE,
        "percentile_method": "nearest-rank: rank = ceil(q/100·n)",
        "executor_settings": executor_settings,
        "codes": dict(Counter(r["code"] for r in rows if r["code"])),
        "rejected_codes": dict(Counter(r["rejected_code"] for r in rows if r["rejected_code"])),
        "fault_injected_rows": sum(1 for r in rows if r["fault_injected"]),
        "reference": REFERENCE,
        "all_match": all(r["outcome"] == "match" for r in rows) and bool(rows),
    }


def to_markdown(s: dict) -> str:
    """사람이 읽는 취합표."""
    lines = [f"# 통신 시험 취합 — {s['peer']}", "", f"> {s['peer_note']}", ""]
    lines += ["| 케이스 | 시나리오 | 행 | 일치 | 불일치 | 건너뜀 |", "| --- | --- | --- | --- | --- | --- |"]

    for cid, c in sorted(s["cases"].items()):
        lines.append(f"| {cid} | {c['scenario']} | {c['rows']} | {c['match']} | {c['mismatch']} | {c['skipped']} |")

    lines += ["", "## 수락 분류(TCP §8.4)", "", "| 분류 | 건수 | 비율 |", "| --- | --- | --- |"]

    for k in ACCEPTANCE:
        ratio = s["acceptance_ratio"][k]
        lines.append(f"| {k} | {s['acceptance'][k]} | {'-' if ratio is None else f'{ratio:.3f}'} |")

    ex = s["executor_settings"]
    lines += ["", f"- {s['first_ack_kpi_note']}", "", "## S1 지연(ms)", "",
              f"가짜 executor 설정(시간은 이 값에 좌우됨): {ex}", "",
              "| 구간 | N | P50 | P95 | 최대 | 참조값(판정 없음) |", "| --- | --- | --- | --- | --- | --- |"]

    for key, label in (("S1_ack", "송신 → 첫 상태"), ("S1_final", "송신 → 최종 상태")):
        st = s["latency_ms"][key]
        fmt = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
        ref = f"P95 {s['reference']['p95_ms_target']} ms, 통신 {s['reference']['comm_count_target']}회"
        lines.append(f"| {label} | {st['n']} | {fmt(st['p50'])} | {fmt(st['p95'])} | {fmt(st['max'])} | {ref} |")

    lines += ["", f"- 지연 출처: {s['latency_source']}", f"- 분위수: {s['percentile_method']}",
              f"- 참조값: {s['reference']['note']}", "", "## 코드", "",
              f"- 상태 코드: {s['codes'] or '-'}", f"- 거부 코드: {s['rejected_codes'] or '-'}",
              f"- 장애 주입 행: {s['fault_injected_rows']}", ""]

    return "\n".join(lines)
