"""분류 시험 취합 — 공급별 최종 판정과 §10.3 지표 표(#186 계획 r0 §5, reviewer 보완 1·2).

입력은 순서표(독립 정답, `sequence`)·정규화 실행 기록(`records.RunRecords`)·실행 메타(`records.RunMeta`)뿐이다.

- **분모는 순서표 전체 공급 수.** 무효·미처리·격리·사용자 중단은 분모에서 빼지 않고 열로 따로 낸다(계획서 §10.3).
- 볼트 분류 정확도(분모 = 순서표 볼트 공급 수)는 `include_nut`과 관계없이 항상 낸다(보완 2, 사용자 2026-11-19).
- 상자 수량은 도착 확인 사건으로만 센다. 같은 부품의 두 번째 도착은 판정에 쓰지 않고 중복 열로 센다(§2.6.4).
- 정답과 결과의 연결 키는 풀 부품 ID다. 풀 부품은 회수·재공급되므로 「그 공급 시각 ~ 같은 풀 부품의 다음 공급 시각」
  창 안의 사건만 그 공급에 붙인다. `part_track_id`는 런타임 추정치라 정답 연결에 쓰지 않는다.
- 볼트가 너트 상자에 도착하면 `WRONG_BIN`이고, 「실패 볼트의 너트 상자 유입 수」에도 함께 센다(보완 1).
- 비율의 분모가 0이면 `None`(표에서는 「-」)으로 두고 0으로 쓰지 않는다.
"""
from __future__ import annotations

import csv
import io
import math
from collections import Counter

from eyetoarm.sorteval.records import BINS, RunMeta, RunRecords
from eyetoarm.sorteval.sequence import BOLT_SPECS, NUT_SPEC, expected_bin

OUTCOMES = ("CORRECT", "WRONG_BIN", "ISOLATED", "NOT_ARRIVED", "USER_ABORTED", "SUPPLY_MISSING", "SUPPLY_MISMATCH")
CONFUSION_COLUMNS = BINS + ("NONE",)
B_CONFUSION_COLUMNS = BOLT_SPECS + ("UNRESOLVED", "NONE")
SUBJECTS = ("A", "B", "C", "USER", "UNKNOWN")
TIMING_KEYS = ("supply_to_first_seen", "first_seen_to_b_goal", "belt_stop_to_restart", "first_seen_to_arrival")
FORMAT = "eyetoarm.sort_summary"
FORMAT_VERSION = 1

RELATION_NOTES = {
    "bolts_into_nut_bin": "볼트가 너트 상자(벨트 끝)에 도착한 행. 이 행들은 모두 outcome WRONG_BIN에도 포함된다"
                          "(WRONG_BIN ⊇ bolts_into_nut_bin). 목표 0",
    "duplicates": "same_part = 같은 풀 부품의 같은 공급 창 안 두 번째 이후 도착, same_track = 같은 part_track_id의 두 번째 "
                  "이후 도착. 한 중복 도착이 두 열에 함께 잡힐 수 있다",
    "missed": "공급됐지만 어떤 A 트랙에도 연결되지 않은 행(검출 없이 통과). 트랙 연결 출처는 records.TrackLink",
    "precision_recall": "도착 상자 기준. 재현율 분모 = 순서표의 그 규격 공급 수(미처리 포함), 정밀도 분모 = 그 상자 전체 도착 수",
}


def percentile(values, q: float) -> float | None:
    """최근접 순위 분위수(q는 0~100). 빈 목록이면 None. 순위 = ceil(q/100·n), 최소 1."""
    xs = sorted(values)

    if not xs:
        return None

    if not 0 <= q <= 100:
        raise ValueError(f"q must be within [0, 100]: {q}")

    return xs[max(1, math.ceil(q / 100 * len(xs))) - 1]


def _ratio(num: int, den: int) -> float | None:
    """분모 0이면 None."""
    return num / den if den else None


def _ms(ns: int) -> float:
    """ns → ms."""
    return ns / 1e6


class _Index:
    """공급별 사건 창을 만들기 위한 색인."""

    def __init__(self, records: RunRecords):
        """정규화 기록에서 공급 번호·풀 부품별 공급 시각·A 트랙·첫 사용자 정지 시각 색인을 만든다."""
        self.records = records
        self.supply_by_index = {s.supply_index: s for s in records.supplies}

        # 같은 풀 부품의 공급 시각 목록(창 끝 계산용).
        self.pool_supply_times: dict[str, list[int]] = {}

        for s in records.supplies:
            self.pool_supply_times.setdefault(s.pool_part_id, []).append(s.stamp_ns)

        for times in self.pool_supply_times.values():
            times.sort()

        self.a_track = {t.part_track_id: t for t in records.a_tracks}
        self.user_stop_ns = min((e.stamp_ns for e in records.run_events if e.kind == "USER_STOP"), default=None)

    def in_flight_at_stop(self, supply, cycles: list) -> bool:
        """사용자 정지 때 이 공급이 아직 처리 중이었나(잠정 규칙, C 사건 형식 확정 전).

        정지가 공급 뒤에 있고, 정지 전에 끝난 사이클 결과가 없고, 정지 전에 다른 부품이 더 공급되지 않았으면 처리 중으로
        본다. 이미 끝났거나 뒤 부품이 공급된 뒤의 미도착은 사용자 중단이 아니라 NOT_ARRIVED다.
        """
        stop = self.user_stop_ns

        if stop is None or stop < supply.stamp_ns:
            return False

        if any(c.stamp_ns <= stop for c in cycles):
            return False

        return not any(supply.stamp_ns < s.stamp_ns <= stop for s in self.records.supplies)

    def window(self, supply) -> tuple[int, float]:
        """[공급 시각, 같은 풀 부품의 다음 공급 시각) 창."""
        later = [t for t in self.pool_supply_times[supply.pool_part_id] if t > supply.stamp_ns]

        return supply.stamp_ns, (min(later) if later else math.inf)


def _in(stamp: int, win: tuple[int, float]) -> bool:
    """창 안 여부."""
    return win[0] <= stamp < win[1]


def _first_failure(row: dict, spec: str, ctx: dict, a_index: dict) -> tuple[str, str] | None:
    """처리 순서(A 검출 → B 판정 → C 실행)에서 처음 끊긴 판정 주체와 사유. CORRECT면 None."""
    outcome = row["outcome"]

    if outcome == "CORRECT":
        return None

    if outcome in ("SUPPLY_MISSING", "SUPPLY_MISMATCH"):
        return "C", outcome

    if outcome == "USER_ABORTED":
        return "USER", "USER_STOP"

    if spec == NUT_SPEC:
        return "C", f"NUT_{outcome}"

    tracks = ctx["tracks"]

    if not tracks:
        return "A", "NOT_DETECTED"

    decisions = ctx["b_decisions"]

    if not decisions:
        reasons = Counter()

        for track_id in tracks:
            if track_id in a_index:
                reasons.update(a_index[track_id].invalid_reasons)

        if reasons:
            top = sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            return "A", top

        return "B", "NO_B_DECISION"

    last = decisions[-1]

    if last.part_class == "UNRESOLVED":
        return "B", "UNRESOLVED"

    if last.action == "SKIP":
        return "B", last.skip_reason

    if last.part_class != spec:
        return "B", "CLASS_MISMATCH"

    failed = [c for c in ctx["cycles"] if c.status != "SUCCEEDED" or c.goal_kind == "ABORT_CYCLE"]

    if failed:
        c = failed[0]
        return "C", c.reason or f"{c.goal_kind}_{c.status}"

    return "C", outcome


def _per_supply(sequence: dict, records: RunRecords, idx: _Index) -> tuple[list[dict], dict]:
    """공급 행마다 최종 판정·연결 사건·시간을 만든다. 연결되지 않은 사건 수도 함께 센다."""
    rows = []
    used_arrivals: set[int] = set()
    used_cycles: set[int] = set()
    used_tracks: set[str] = set()
    max_supplied = max(idx.supply_by_index, default=0)

    for truth in sequence["supplies"]:
        n = truth["supply_index"]
        spec = truth["part_spec"]
        supply = idx.supply_by_index.get(n)
        row = {"supply_index": n, "part_spec": spec, "expected_bin": expected_bin(spec), "scenario": truth["scenario"],
               "pool_part_id": None, "supplied_spec": None, "arrival_bin": None, "duplicate_arrivals": 0,
               "tracks": [], "b_class": None, "b_action": None, "timings_ms": {},
               "_cycles": [], "_first_arrival": None}
        ctx = {"tracks": [], "b_decisions": [], "cycles": []}

        if supply is None:
            aborted = idx.user_stop_ns is not None and n > max_supplied
            row["outcome"] = "USER_ABORTED" if aborted else "SUPPLY_MISSING"
            row["failure"] = _first_failure(row, spec, ctx, idx.a_track)
            rows.append(row)
            continue

        win = idx.window(supply)
        row["pool_part_id"] = supply.pool_part_id
        row["supplied_spec"] = supply.supplied_spec

        arrivals = sorted(((i, a) for i, a in enumerate(records.arrivals)
                           if a.pool_part_id == supply.pool_part_id and _in(a.stamp_ns, win)),
                          key=lambda ia: ia[1].stamp_ns)
        cycles = sorted(((i, c) for i, c in enumerate(records.cycles)
                         if c.pool_part_id == supply.pool_part_id and _in(c.stamp_ns, win)),
                        key=lambda ic: ic[1].stamp_ns)
        used_arrivals.update(i for i, _ in arrivals)
        used_cycles.update(i for i, _ in cycles)

        tracks = {lk.part_track_id for lk in records.track_links
                  if lk.pool_part_id == supply.pool_part_id and _in(lk.stamp_ns, win)}
        tracks |= {c.part_track_id for _, c in cycles}
        tracks |= {a.part_track_id for _, a in arrivals if a.part_track_id}
        used_tracks |= tracks

        decisions = sorted((d for d in records.b_decisions if d.part_track_id in tracks), key=lambda d: d.stamp_ns)
        ctx = {"tracks": sorted(tracks), "b_decisions": decisions, "cycles": [c for _, c in cycles]}
        row["tracks"] = sorted(tracks)
        row["duplicate_arrivals"] = max(0, len(arrivals) - 1)

        if decisions:
            row["b_class"] = decisions[-1].part_class
            row["b_action"] = decisions[-1].action

        first_arrival = arrivals[0][1] if arrivals else None
        row["arrival_bin"] = first_arrival.bin if first_arrival else None

        if supply.supplied_spec != spec:
            row["outcome"] = "SUPPLY_MISMATCH"
        elif first_arrival is not None:
            if first_arrival.bin == expected_bin(spec):
                row["outcome"] = "CORRECT"
            elif first_arrival.bin == "ISOLATION":
                row["outcome"] = "ISOLATED"
            else:
                row["outcome"] = "WRONG_BIN"
        elif idx.in_flight_at_stop(supply, ctx["cycles"]):
            row["outcome"] = "USER_ABORTED"
        else:
            row["outcome"] = "NOT_ARRIVED"

        row["failure"] = _first_failure(row, spec, ctx, idx.a_track)
        row["timings_ms"] = _timings(supply, ctx, first_arrival, idx.a_track)
        row["_cycles"] = ctx["cycles"]
        row["_first_arrival"] = first_arrival
        rows.append(row)

    unlinked = {
        "arrivals": len(records.arrivals) - len(used_arrivals),
        "cycles": len(records.cycles) - len(used_cycles),
        "a_tracks": sum(1 for t in records.a_tracks if t.part_track_id not in used_tracks),
        "b_decisions": sum(1 for d in records.b_decisions if d.part_track_id not in used_tracks),
    }

    return rows, unlinked


def _timings(supply, ctx: dict, first_arrival, a_index: dict) -> dict:
    """구간별 시간(ms). 구할 수 없는 구간은 빠진다. 음수는 그대로 두고 집계에서 이상치로 센다."""
    out = {}
    seen = [a_index[t].first_seen_ns for t in ctx["tracks"] if t in a_index]
    first_seen = min(seen) if seen else None

    if first_seen is not None:
        out["supply_to_first_seen"] = _ms(first_seen - supply.stamp_ns)

        goals = [d.stamp_ns for d in ctx["b_decisions"] if d.action == "GOAL_SENT"]

        if goals:
            out["first_seen_to_b_goal"] = _ms(min(goals) - first_seen)

        if first_arrival is not None:
            out["first_seen_to_arrival"] = _ms(first_arrival.stamp_ns - first_seen)

    stops = [(c.belt_stop_ns, c.belt_restart_ns) for c in ctx["cycles"]
             if c.belt_stop_ns is not None and c.belt_restart_ns is not None]

    if stops:
        out["belt_stop_to_restart"] = _ms(stops[0][1] - stops[0][0])

    return out


def _timing_stats(rows: list[dict]) -> dict:
    """구간별 n·중앙값·P95·최대·결측·음수(이상치, 통계 제외)."""
    out = {}

    for key in TIMING_KEYS:
        values = [r["timings_ms"][key] for r in rows if key in r["timings_ms"]]
        good = [v for v in values if v >= 0]
        out[key] = {"n": len(good), "p50_ms": percentile(good, 50), "p95_ms": percentile(good, 95),
                    "max_ms": max(good) if good else None, "missing": len(rows) - len(values),
                    "negative": len(values) - len(good)}

    return out


def _confusion(rows: list[dict], specs: tuple, columns: tuple, key) -> dict:
    """행 = 정답 규격, 열 = key(row) 값(None → NONE)."""
    table = {s: {c: 0 for c in columns} for s in specs}

    for r in rows:
        if r["part_spec"] in table:
            table[r["part_spec"]][key(r) or "NONE"] += 1

    return table


def _precision_recall(confusion: dict, specs: tuple) -> dict:
    """도착 상자 기준 볼트 규격별 정밀도·재현율(분모 공개)."""
    out = {}

    for spec in BOLT_SPECS:
        tp = confusion.get(spec, {}).get(spec, 0)
        supplied = sum(confusion.get(spec, {}).values())
        arrived = sum(confusion[s][spec] for s in specs)
        out[spec] = {"tp": tp, "recall": _ratio(tp, supplied), "recall_denominator": supplied,
                     "precision": _ratio(tp, arrived), "precision_denominator": arrived}

    return out


def aggregate(sequence: dict, records: RunRecords, meta: RunMeta) -> dict:
    """순서표·정규화 기록·메타 → 취합 사전. 메타의 순서표 식별이 다르면 ValueError."""
    if meta.sequence_sha256 != sequence["content_sha256"] or meta.sequence_id != sequence["sequence_id"]:
        raise ValueError("run meta does not refer to this sequence (sequence_id/sequence_sha256 mismatch)")

    idx = _Index(records)
    rows, unlinked = _per_supply(sequence, records, idx)
    total = len(rows)
    include_nut = sequence["include_nut"]
    specs = BOLT_SPECS + ((NUT_SPEC,) if include_nut else ())
    outcome_counts = {o: sum(1 for r in rows if r["outcome"] == o) for o in OUTCOMES}

    bolts = [r for r in rows if r["part_spec"] in BOLT_SPECS]
    nuts = [r for r in rows if r["part_spec"] == NUT_SPEC]
    confusion = _confusion(rows, specs, CONFUSION_COLUMNS, lambda r: r["arrival_bin"])
    b_confusion = _confusion([r for r in bolts if r["outcome"] not in ("SUPPLY_MISSING", "SUPPLY_MISMATCH",
                                                                       "USER_ABORTED")],
                             BOLT_SPECS, B_CONFUSION_COLUMNS, lambda r: r["b_class"])

    picks = [(r, c) for r in bolts for c in r["_cycles"]
             if c.goal_kind == "PICK_PLACE" and c.status == "SUCCEEDED"]
    placed = sum(1 for r, c in picks if r["_first_arrival"] is not None and r["_first_arrival"].bin == c.place_bin)

    failures = Counter((r["failure"][0], r["failure"][1]) for r in rows if r["failure"] is not None)
    by_subject = {s: {reason: n for (subj, reason), n in sorted(failures.items()) if subj == s} for s in SUBJECTS}

    track_arrivals = Counter(a.part_track_id for a in records.arrivals if a.part_track_id)
    isolated = [r for r in bolts if r["outcome"] == "ISOLATED"]
    iso_restart = []

    for r in isolated:
        restarts = [c.belt_restart_ns for c in r["_cycles"] if c.belt_restart_ns is not None]

        if restarts and r["_first_arrival"] is not None:
            iso_restart.append(_ms(min(restarts) - r["_first_arrival"].stamp_ns))

    stop_by_type = {}

    for outcome in ("ISOLATED", "WRONG_BIN", "NOT_ARRIVED"):
        group = [r for r in bolts if r["outcome"] == outcome]
        stop_by_type[outcome] = {"rows": len(group),
                                 "belt_stopped": sum(1 for r in group
                                                     if any(c.belt_stop_ns is not None for c in r["_cycles"]))}

    correct = outcome_counts["CORRECT"]
    bolt_correct = sum(1 for r in bolts if r["outcome"] == "CORRECT")

    summary = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "header": {
            "eval_run_id": meta.eval_run_id, "status": meta.status, "fake_input": meta.fake_input,
            "sequence_id": sequence["sequence_id"], "sequence_sha256": sequence["content_sha256"],
            "sequence_seed": (sequence["meta"]["generator"] or {}).get("seed"), "include_nut": include_nut,
            "timestamp_base": "SIM_TIME", "detector": meta.detector, "conditions": meta.conditions,
            "confirmer": meta.confirmer, "date": meta.date, "procedure": meta.procedure,
            "evidence_paths": list(meta.evidence_paths),
        },
        "denominator": {"total_supplies": total, "bolt_supplies": len(bolts), "nut_supplies": len(nuts)},
        "outcomes": outcome_counts,
        "accuracy": _ratio(correct, total),
        "bolt_accuracy": _ratio(bolt_correct, len(bolts)),
        "nut_arrival": ({"arrived_nut_bin": sum(1 for r in nuts if r["outcome"] == "CORRECT"), "nut_supplies": len(nuts),
                         "ratio": _ratio(sum(1 for r in nuts if r["outcome"] == "CORRECT"), len(nuts))}
                        if include_nut else None),
        "confusion": confusion,
        "precision_recall": _precision_recall(confusion, specs),
        "b_confusion": b_confusion,
        "bin_placement": {"pick_succeeded": len(picks), "placed_in_target_bin": placed,
                          "ratio": _ratio(placed, len(picks))},
        "missed": sum(1 for r in rows if r["pool_part_id"] is not None and r["outcome"] != "SUPPLY_MISMATCH"
                      and not r["tracks"]),
        "duplicates": {"same_part": sum(r["duplicate_arrivals"] for r in rows),
                       "same_track": sum(n - 1 for n in track_arrivals.values() if n > 1)},
        "bolts_into_nut_bin": sum(1 for r in bolts if r["arrival_bin"] == NUT_SPEC),
        "failures_by_subject": by_subject,
        "timings": _timing_stats(rows),
        "stop_recovery": {"by_outcome": stop_by_type,
                          "isolate_to_restart_ms": {"n": len(iso_restart), "p50_ms": percentile(iso_restart, 50),
                                                    "max_ms": max(iso_restart) if iso_restart else None}},
        "unlinked": unlinked,
        "notes": RELATION_NOTES,
        "per_supply": [_public_row(r) for r in rows],
    }

    return summary


def _public_row(r: dict) -> dict:
    """공급 행에서 내부 참조를 뺀 출력용 사본."""
    out = {k: v for k, v in r.items() if not k.startswith("_")}
    out["failure_subject"], out["failure_reason"] = r["failure"] if r["failure"] else (None, None)
    del out["failure"]

    return out


# ---------------------------------------------------------------------------
# 출력
# ---------------------------------------------------------------------------

def fmt_cell(value) -> str:
    """표 칸: None은 「-」, 비율은 소수 3자리."""
    if value is None:
        return "-"

    if isinstance(value, float):
        return f"{value:.3f}"

    return str(value)


def fake_banner(fake: bool) -> list[str]:
    """가짜 입력 표시 줄(표 머리)."""
    return ["> **FAKE_INPUT — 가짜 입력으로 만든 결과. 실제 분류 성능이 아니다.**", ""] if fake else []


def to_markdown(s: dict) -> str:
    """취합 MD 표."""
    h = s["header"]
    d = s["denominator"]
    lines = fake_banner(h["fake_input"]) + [
        f"# 분류 시험 취합 `{h['eval_run_id']}`",
        "",
        f"- 상태 {h['status']} · 순서표 `{h['sequence_id']}` sha256 `{h['sequence_sha256'][:16]}…` · 시드 "
        f"{fmt_cell(h['sequence_seed'])} · 너트 행 {h['include_nut']} · 시간 기준 {h['timestamp_base']}",
        f"- 검출 방식 {h['detector'].get('method')} (설정 {h['detector'].get('config_id', '-')})",
        f"- 확인자 {fmt_cell(h['confirmer'])} · 날짜 {fmt_cell(h['date'])} · 절차 {fmt_cell(h['procedure'])} · 증거 "
        f"{', '.join(h['evidence_paths']) or '-'}",
        "",
        "## 정확도(분모 = 순서표 공급 수)",
        "",
        "| 지표 | 값 | 분모 |",
        "|---|---:|---:|",
        f"| 분류 정확도 | {fmt_cell(s['accuracy'])} | {d['total_supplies']} |",
        f"| 볼트 분류 정확도 | {fmt_cell(s['bolt_accuracy'])} | {d['bolt_supplies']} |",
    ]

    if s["nut_arrival"] is not None:
        lines.append(f"| 너트 도착 | {fmt_cell(s['nut_arrival']['ratio'])} | {d['nut_supplies']} |")

    bp = s["bin_placement"]
    lines += [f"| 실제 상자 투입(집기 성공 중) | {fmt_cell(bp['ratio'])} | {bp['pick_succeeded']} |", "",
              "## 공급별 최종 판정(합 = 분모)", "", "| " + " | ".join(OUTCOMES) + " | 합 |",
              "|" + "---:|" * (len(OUTCOMES) + 1),
              "| " + " | ".join(str(s["outcomes"][o]) for o in OUTCOMES) + f" | {sum(s['outcomes'].values())} |", "",
              "## 혼동 행렬(행 = 정답 규격, 열 = 첫 도착 상자)", "",
              "| 정답 \\ 도착 | " + " | ".join(CONFUSION_COLUMNS) + " |", "|---|" + "---:|" * len(CONFUSION_COLUMNS)]
    lines += [f"| {spec} | " + " | ".join(str(row[c]) for c in CONFUSION_COLUMNS) + " |"
              for spec, row in s["confusion"].items()]
    lines += ["", "| 규격 | 정밀도 | 정밀도 분모 | 재현율 | 재현율 분모 |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {spec} | {fmt_cell(v['precision'])} | {v['precision_denominator']} | {fmt_cell(v['recall'])} | "
              f"{v['recall_denominator']} |" for spec, v in s["precision_recall"].items()]
    lines += ["", "## B 판정 혼동 행렬(보조, 공급·연결된 볼트 행)", "",
              "| 정답 \\ B | " + " | ".join(B_CONFUSION_COLUMNS) + " |", "|---|" + "---:|" * len(B_CONFUSION_COLUMNS)]
    lines += [f"| {spec} | " + " | ".join(str(row[c]) for c in B_CONFUSION_COLUMNS) + " |"
              for spec, row in s["b_confusion"].items()]
    lines += ["", "## 누락·중복·너트 상자 유입", "",
              f"- 검출 없이 통과: {s['missed']}",
              f"- 중복 도착: 같은 부품 {s['duplicates']['same_part']}, 같은 트랙 {s['duplicates']['same_track']}",
              f"- 실패 볼트의 너트 상자 유입: {s['bolts_into_nut_bin']} (WRONG_BIN에 포함, 목표 0)",
              "- 연결 안 된 사건: " + ", ".join(f"{k} {v}" for k, v in s["unlinked"].items()), "",
              "## 판정 주체별 실패 사유(공급 행마다 처음 끊긴 주체 1건)", "", "| 주체 | 사유 | 건수 |", "|---|---|---:|"]

    for subj, reasons in s["failures_by_subject"].items():
        for reason, n in reasons.items():
            lines.append(f"| {subj} | {reason} | {n} |")

    lines += ["", "## 처리 시간(SIM_TIME, ms)", "", "| 구간 | n | 중앙값 | P95 | 최대 | 결측 | 음수 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {k} | {v['n']} | {fmt_cell(v['p50_ms'])} | {fmt_cell(v['p95_ms'])} | {fmt_cell(v['max_ms'])} | {v['missing']} | "
              f"{v['negative']} |" for k, v in s["timings"].items()]
    sr = s["stop_recovery"]
    lines += ["", "## 정지·복구", "", "| 최종 판정 | 행 | 벨트 정지 기록 |", "|---|---:|---:|"]
    lines += [f"| {k} | {v['rows']} | {v['belt_stopped']} |" for k, v in sr["by_outcome"].items()]
    lines += ["", f"- 격리 도착 → 재가동: n {sr['isolate_to_restart_ms']['n']}, 중앙값 "
                  f"{fmt_cell(sr['isolate_to_restart_ms']['p50_ms'])} ms, 최대 {fmt_cell(sr['isolate_to_restart_ms']['max_ms'])} ms",
              "", "## 열 관계", ""]
    lines += [f"- `{k}`: {v}" for k, v in s["notes"].items()]

    return "\n".join(lines) + "\n"


PER_SUPPLY_COLUMNS = ("supply_index", "part_spec", "expected_bin", "scenario", "pool_part_id", "supplied_spec",
                      "arrival_bin", "outcome", "failure_subject", "failure_reason", "b_class", "b_action",
                      "duplicate_arrivals", "tracks") + tuple(f"{k}_ms" for k in TIMING_KEYS)


def per_supply_csv(s: dict) -> str:
    """공급 행별 CSV."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(PER_SUPPLY_COLUMNS)

    for r in s["per_supply"]:
        cells = [r[c] for c in PER_SUPPLY_COLUMNS[:13]] + [";".join(r["tracks"])]
        cells += [r["timings_ms"].get(k) for k in TIMING_KEYS]
        writer.writerow("" if c is None else c for c in cells)

    return buf.getvalue()
