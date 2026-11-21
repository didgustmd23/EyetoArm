"""분류 시험 공급 순서표(독립 정답): 자료 형식·고정 시드 생성기·검증기·출력.

JSON이 정본이고 CSV·MD는 사본이다. 기대 결과는 행에 쓰지 않고 규격에서 유도한다(`expected_bin`).
생성기는 `random.Random(seed)` 하나만 써서 같은 seed·인자면 같은 내용이 나온다.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

FORMAT = "eyetoarm.sort_sequence"
FORMAT_VERSION = 1
GENERATOR_NAME = "eyetoarm.sorteval.sequence"
GENERATOR_VERSION = 1

BOLT_SPECS = ("BOLT_A", "BOLT_B")
NUT_SPEC = "NUT"
ALL_SPECS = BOLT_SPECS + (NUT_SPEC,)

SCENARIOS = ("normal", "consecutive_same", "alternate", "boundary", "nut_distractor")
PATTERNS = ("shuffled", "blocks", "alternate", "mixed")
SEQUENCE_STATUSES = ("draft", "planned")

# 규격 → 기대 도착 상자. 너트는 집지 않고 벨트 끝 너트 상자로 흘러간다(NUT_PASS).
EXPECTED_BIN = {"BOLT_A": "BOLT_A", "BOLT_B": "BOLT_B", "NUT": "NUT"}

_TOP_REQUIRED = ("format", "format_version", "sequence_id", "status", "meta", "timing", "pose_ranges",
                 "include_nut", "supplies")
_TOP_OPTIONAL = ("content_sha256",)
_META_KEYS = ("generator", "created_at_utc", "python", "author", "confirmer", "notes")
_GENERATOR_KEYS = ("name", "version", "seed", "args")
_TIMING_KEYS = ("basis", "mode", "interval_s")
_POSE_RANGE_KEYS = ("lateral_offset_mm", "yaw_deg")
_ROW_KEYS = ("supply_index", "part_spec", "scenario", "interval_s", "start_pose")
_CSV_COLUMNS = ("supply_index", "part_spec", "scenario", "interval_s", "lateral_offset_mm", "yaw_deg")
_MAX_INT_DIGITS = 18


class SequenceFormatError(ValueError):
    """순서표 JSON 자체를 해석할 수 없다(중복 키·비유한 수·문법 오류)."""


def expected_bin(part_spec: str) -> str:
    """규격의 기대 도착 상자."""
    return EXPECTED_BIN[part_spec]


# ---------------------------------------------------------------------------
# 엄격한 JSON 해석
# ---------------------------------------------------------------------------

def _reject_duplicate_keys(pairs):
    """object_pairs_hook: 같은 키가 두 번 나오면 거부한다."""
    out = {}

    for key, value in pairs:
        if key in out:
            raise SequenceFormatError(f"duplicate key: {key!r}")

        out[key] = value

    return out


def _reject_constant(name):
    """parse_constant: NaN·Infinity·-Infinity를 거부한다."""
    raise SequenceFormatError(f"non-finite constant: {name}")


def _parse_float(text):
    """parse_float: `1e400`처럼 float로 넘치는 값을 거부한다."""
    value = float(text)

    if not math.isfinite(value):
        raise SequenceFormatError(f"non-finite number: {text}")

    return value


def _parse_int(text):
    """parse_int: 지나치게 긴 정수를 거부한다(시드·번호에 18자리를 넘는 값은 오류로 본다)."""
    digits = text.lstrip("-")

    if len(digits) > _MAX_INT_DIGITS:
        raise SequenceFormatError(f"integer too large: {text[:24]}…")

    return int(text)


def loads_strict(text: str):
    """중복 키·비유한 수·거대 정수를 거부하는 JSON 해석."""
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant,
                          parse_float=_parse_float, parse_int=_parse_int)
    except json.JSONDecodeError as exc:
        raise SequenceFormatError(f"invalid JSON: {exc}") from exc


def dumps_canonical(obj) -> str:
    """해시용 정규 JSON(키 정렬·구분자 고정·비유한 수 금지)."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_sha256(seq: dict) -> str:
    """순서표 내용 해시. `supplies`·`timing`·`pose_ranges`·`include_nut`만 넣는다(메타·생성 시각 제외)."""
    body = {k: seq[k] for k in ("supplies", "timing", "pose_ranges", "include_nut")}

    return hashlib.sha256(dumps_canonical(body).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 생성기
# ---------------------------------------------------------------------------

def _allocate(count: int, ratio: dict[str, int]) -> dict[str, int]:
    """최대 나머지 방식으로 count를 비율대로 나눈다. 동률은 규격 이름 순(결정적)."""
    total = sum(ratio.values())
    exact = {k: count * v / total for k, v in ratio.items()}
    out = {k: int(math.floor(x)) for k, x in exact.items()}
    rest = count - sum(out.values())

    order = sorted(ratio, key=lambda k: (-(exact[k] - out[k]), k))

    for k in order[:rest]:
        out[k] += 1

    return out


def _check_ratio(ratio: dict, include_nut: bool) -> dict[str, int]:
    """비율 인자 검사. 음이 아닌 정수, 합 > 0, include_nut=False면 NUT 없음."""
    if not isinstance(ratio, dict) or not ratio:
        raise ValueError("ratio must be a non-empty dict")

    out = {}

    for spec, weight in ratio.items():
        if spec not in ALL_SPECS:
            raise ValueError(f"unknown spec in ratio: {spec!r}")

        if spec == NUT_SPEC and not include_nut:
            raise ValueError("NUT in ratio requires include_nut=True")

        if isinstance(weight, bool) or not isinstance(weight, int) or weight < 0:
            raise ValueError(f"ratio weight must be a non-negative int: {spec}={weight!r}")

        out[spec] = weight

    if sum(out.values()) <= 0:
        raise ValueError("ratio weights sum to zero")

    return out


def _take_alternate(pool: dict[str, int], specs: list[str], start: int, length: int | None) -> list[str]:
    """남은 수가 있는 규격을 정해진 순서로 돌아가며 뽑는다. length=None이면 모두 소진할 때까지."""
    out: list[str] = []
    i = start

    while (length is None or len(out) < length) and any(pool[s] > 0 for s in specs):
        spec = specs[i % len(specs)]
        i += 1

        if pool[spec] > 0:
            pool[spec] -= 1
            out.append(spec)

    return out


def _blocks(pool: dict[str, int], specs: list[str], block_len: int, rng: random.Random) -> list[list[str]]:
    """규격별로 block_len 묶음을 만들고 묶음 순서를 섞는다(마지막 묶음은 짧을 수 있음)."""
    blocks = []

    for spec in specs:
        n = pool[spec]

        while n > 0:
            k = min(block_len, n)
            blocks.append([spec] * k)
            n -= k

        pool[spec] = 0

    rng.shuffle(blocks)

    return blocks


def _scenario_for(spec: str, base: str) -> str:
    """너트 행은 항상 nut_distractor, 볼트 행은 패턴 구간의 시나리오."""
    return "nut_distractor" if spec == NUT_SPEC else base


def generate_sequence(*, seed: int, count: int, pattern: str = "shuffled", ratio: dict | None = None,
                      include_nut: bool = False, block_len: int = 3, sequence_id: str | None = None,
                      interval_s: float | None = None, created_at_utc: str | None = None) -> dict:
    """고정 시드 순서표를 만든다. 같은 인자면 같은 내용이 나온다."""
    if isinstance(seed, bool) or not isinstance(seed, int) or len(str(abs(seed))) > _MAX_INT_DIGITS:
        raise ValueError(f"seed must be an int of at most {_MAX_INT_DIGITS} digits: {seed!r}")

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(f"count must be a positive int: {count!r}")

    if pattern not in PATTERNS:
        raise ValueError(f"unknown pattern {pattern!r}; expected one of {PATTERNS}")

    if isinstance(block_len, bool) or not isinstance(block_len, int) or block_len < 2:
        raise ValueError(f"block_len must be an int >= 2: {block_len!r}")

    if ratio is None:
        ratio = {"BOLT_A": 1, "BOLT_B": 1, **({NUT_SPEC: 1} if include_nut else {})}

    ratio = _check_ratio(ratio, include_nut)

    rng = random.Random(seed)
    pool = _allocate(count, ratio)
    specs = [s for s in ALL_SPECS if pool.get(s, 0) > 0]
    pool = {s: pool.get(s, 0) for s in ALL_SPECS}

    rows: list[tuple[str, str]] = []

    if pattern == "shuffled":
        flat = [s for s in specs for _ in range(pool[s])]
        rng.shuffle(flat)
        rows = [(s, "normal") for s in flat]

    elif pattern == "blocks":
        for block in _blocks(pool, specs, block_len, rng):
            rows.extend((s, "consecutive_same") for s in block)

    elif pattern == "alternate":
        start = rng.randrange(len(specs))
        rows = [(s, "alternate") for s in _take_alternate(pool, specs, start, None)]

    else:
        rows = _mixed_rows(pool, specs, block_len, rng)

    supplies = [
        {"supply_index": i, "part_spec": spec, "scenario": _scenario_for(spec, base),
         "interval_s": None, "start_pose": None}
        for i, (spec, base) in enumerate(rows, start=1)
    ]

    args = {"count": count, "pattern": pattern, "ratio": dict(sorted(ratio.items())), "include_nut": include_nut,
            "block_len": block_len, "interval_s": interval_s}

    seq = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "sequence_id": sequence_id or f"seq_s{seed}_n{count}_{pattern}",
        "status": "planned",
        "meta": {
            "generator": {"name": GENERATOR_NAME, "version": GENERATOR_VERSION, "seed": seed, "args": args},
            "created_at_utc": created_at_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "python": platform.python_version(),
            "author": None,
            "confirmer": None,
            "notes": "",
        },
        "timing": {"basis": "SIM_TIME", "mode": "interval", "interval_s": interval_s},
        "pose_ranges": {"lateral_offset_mm": None, "yaw_deg": None},
        "include_nut": include_nut,
        "supplies": supplies,
    }
    seq["content_sha256"] = content_sha256(seq)

    errors = validate_sequence(seq)

    if errors:
        raise ValueError("generator produced an invalid sequence: " + "; ".join(errors))

    return seq


def _mixed_rows(pool: dict[str, int], specs: list[str], block_len: int, rng: random.Random) -> list[tuple[str, str]]:
    """섞기 구간 + 같은 볼트 규격 연속 구간 + 볼트 교대 구간을 보장하고 세 구간 순서를 섞는다."""
    bolts = [s for s in BOLT_SPECS if s in specs]
    can_block = [s for s in bolts if pool[s] >= block_len]

    if not can_block or len(bolts) < 2:
        raise ValueError("pattern 'mixed' needs both bolt specs and one with at least block_len supplies")

    block_spec = rng.choice(can_block)
    pool[block_spec] -= block_len
    block = [(block_spec, "consecutive_same")] * block_len

    if min(pool[s] for s in BOLT_SPECS) < 2:
        raise ValueError("pattern 'mixed' needs at least 2 of each bolt spec left for the alternate segment")

    start = rng.randrange(2)
    alt = [(s, "alternate") for s in _take_alternate(pool, list(BOLT_SPECS), start, 4)]

    flat = [s for s in specs for _ in range(pool[s])]
    rng.shuffle(flat)
    normal = [(s, "normal") for s in flat]

    segments = [block, alt, normal]
    rng.shuffle(segments)

    return [row for seg in segments for row in seg]


# ---------------------------------------------------------------------------
# 검증기
# ---------------------------------------------------------------------------

def _is_int(value) -> bool:
    """bool이 아닌 정수인가."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value) -> bool:
    """bool이 아닌 유한 실수(정수 포함)인가."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False

    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _check_keys(obj, required, optional, where: str, errors: list[str]) -> bool:
    """사전 여부와 필드 집합(누락·미정의)을 검사한다."""
    if not isinstance(obj, dict):
        errors.append(f"{where}: must be an object")
        return False

    missing = [k for k in required if k not in obj]
    unknown = [k for k in obj if k not in required and k not in optional]

    if missing:
        errors.append(f"{where}: missing field(s) {missing}")

    if unknown:
        errors.append(f"{where}: undefined field(s) {unknown}")

    return not missing


def _check_range(value, where: str, errors: list[str]) -> tuple[float, float] | None:
    """[lo, hi] 유한 수 쌍(lo ≤ hi) 또는 None."""
    if value is None:
        return None

    if (not isinstance(value, list) or len(value) != 2 or not all(_is_finite_number(v) for v in value)
            or value[0] > value[1]):
        errors.append(f"{where}: must be null or [lo, hi] finite numbers with lo <= hi")
        return None

    return float(value[0]), float(value[1])


def _check_interval(value, where: str, errors: list[str]) -> None:
    """간격은 null 또는 유한 양수(s)."""
    if value is not None and (not _is_finite_number(value) or value <= 0):
        errors.append(f"{where}: interval_s must be null or a finite number > 0")


def validate_sequence(seq) -> list[str]:
    """순서표 검사. 오류 사유를 모두 모아 돌려준다(빈 목록이면 통과)."""
    errors: list[str] = []

    if not _check_keys(seq, _TOP_REQUIRED, _TOP_OPTIONAL, "sequence", errors):
        return errors

    if seq["format"] != FORMAT:
        errors.append(f"format: expected {FORMAT!r}")

    if not _is_int(seq["format_version"]) or seq["format_version"] != FORMAT_VERSION:
        errors.append(f"format_version: unsupported {seq['format_version']!r}")

    if not isinstance(seq["sequence_id"], str) or not seq["sequence_id"].strip():
        errors.append("sequence_id: must be a non-empty string")

    if seq["status"] not in SEQUENCE_STATUSES:
        errors.append(f"status: expected one of {SEQUENCE_STATUSES}")

    if not isinstance(seq["include_nut"], bool):
        errors.append("include_nut: must be true or false")

    _validate_meta(seq["meta"], errors)

    if _check_keys(seq["timing"], _TIMING_KEYS, (), "timing", errors):
        if seq["timing"]["basis"] != "SIM_TIME":
            errors.append("timing.basis: only SIM_TIME is supported")

        if seq["timing"]["mode"] != "interval":
            errors.append("timing.mode: only 'interval' is supported")

        _check_interval(seq["timing"]["interval_s"], "timing", errors)

    ranges = {}

    if _check_keys(seq["pose_ranges"], _POSE_RANGE_KEYS, (), "pose_ranges", errors):
        for key in _POSE_RANGE_KEYS:
            ranges[key] = _check_range(seq["pose_ranges"][key], f"pose_ranges.{key}", errors)

    _validate_supplies(seq["supplies"], seq["include_nut"] is True, ranges, errors)

    if "content_sha256" in seq and not errors and seq["content_sha256"] != content_sha256(seq):
        errors.append("content_sha256: does not match content")

    return errors


def _validate_meta(meta, errors: list[str]) -> None:
    """메타 필드 검사. 생성기 메타가 있으면 시드·인자를 요구한다(사람이 쓴 순서표는 generator=null)."""
    if not _check_keys(meta, _META_KEYS, (), "meta", errors):
        return

    gen = meta["generator"]

    if gen is not None and _check_keys(gen, _GENERATOR_KEYS, (), "meta.generator", errors):
        if not _is_int(gen["seed"]):
            errors.append("meta.generator.seed: must be an int")

        if not isinstance(gen["args"], dict):
            errors.append("meta.generator.args: must be an object")

    if not isinstance(meta["created_at_utc"], str) or not meta["created_at_utc"]:
        errors.append("meta.created_at_utc: must be a non-empty string")

    for key in ("python", "author", "confirmer"):
        if meta[key] is not None and not isinstance(meta[key], str):
            errors.append(f"meta.{key}: must be null or a string")

    if not isinstance(meta["notes"], str):
        errors.append("meta.notes: must be a string")


def _validate_supplies(rows, include_nut: bool, ranges: dict, errors: list[str]) -> None:
    """공급 행 검사: 1부터 연속 번호, 규격·시나리오, 간격, 시작 자세 범위."""
    if not isinstance(rows, list) or not rows:
        errors.append("supplies: must be a non-empty list")
        return

    for pos, row in enumerate(rows, start=1):
        where = f"supplies[{pos}]"

        if not _check_keys(row, _ROW_KEYS, (), where, errors):
            continue

        idx = row["supply_index"]

        if not _is_int(idx):
            errors.append(f"{where}: supply_index must be an int (not bool)")
        elif idx != pos:
            errors.append(f"{where}: supply_index {idx} breaks 1..N contiguous order (expected {pos})")

        spec = row["part_spec"]

        if spec not in ALL_SPECS:
            errors.append(f"{where}: unknown part_spec {spec!r}")
        elif spec == NUT_SPEC and not include_nut:
            errors.append(f"{where}: NUT row but include_nut is false")

        if row["scenario"] not in SCENARIOS:
            errors.append(f"{where}: unknown scenario {row['scenario']!r}")
        elif spec in ALL_SPECS and (row["scenario"] == "nut_distractor") != (spec == NUT_SPEC):
            errors.append(f"{where}: scenario nut_distractor must be used exactly for NUT rows")

        _check_interval(row["interval_s"], where, errors)
        _validate_start_pose(row["start_pose"], ranges, where, errors)


def _validate_start_pose(pose, ranges: dict, where: str, errors: list[str]) -> None:
    """시작 자세는 null 또는 {lateral_offset_mm, yaw_deg}. 값이 있으면 선언된 범위 안이어야 한다."""
    if pose is None:
        return

    if not _check_keys(pose, _POSE_RANGE_KEYS, (), f"{where}.start_pose", errors):
        return

    for key in _POSE_RANGE_KEYS:
        value = pose[key]

        if value is None:
            continue

        if not _is_finite_number(value):
            errors.append(f"{where}.start_pose.{key}: must be null or a finite number")
            continue

        rng = ranges.get(key)

        if rng is None:
            errors.append(f"{where}.start_pose.{key}: value given but pose_ranges.{key} is not declared")
        elif not rng[0] <= value <= rng[1]:
            errors.append(f"{where}.start_pose.{key}: {value} outside declared range {list(rng)}")


def load_sequence_text(text: str) -> tuple[dict | None, list[str]]:
    """JSON 텍스트 → (순서표, 오류 목록). 해석 자체가 실패하면 (None, [사유])."""
    try:
        seq = loads_strict(text)
    except SequenceFormatError as exc:
        return None, [str(exc)]

    return seq, validate_sequence(seq)


def load_sequence(path: str | Path) -> dict:
    """파일에서 순서표를 읽고 검증한다. 실패하면 ValueError(사유 전부)."""
    seq, errors = load_sequence_text(Path(path).read_text(encoding="utf-8"))

    if errors:
        raise ValueError(f"{path}: invalid sort sequence: " + "; ".join(errors))

    return seq


# ---------------------------------------------------------------------------
# 출력·CSV 입구
# ---------------------------------------------------------------------------

def to_json(seq: dict) -> str:
    """정본 JSON(들여쓰기 2, 비유한 수 금지)."""
    return json.dumps(seq, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _cell(value) -> str:
    """CSV 칸: None은 빈칸."""
    return "" if value is None else str(value)


def to_csv(seq: dict) -> str:
    """공급 행 CSV(사람용 사본)."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(_CSV_COLUMNS)

    for row in seq["supplies"]:
        pose = row["start_pose"] or {}
        writer.writerow([row["supply_index"], row["part_spec"], row["scenario"], _cell(row["interval_s"]),
                         _cell(pose.get("lateral_offset_mm")), _cell(pose.get("yaw_deg"))])

    return buf.getvalue()


def to_markdown(seq: dict) -> str:
    """사람이 읽는 MD 표. 정답 사본이므로 런타임에 넘기지 않는다."""
    gen = seq["meta"]["generator"]
    counts = {s: sum(1 for r in seq["supplies"] if r["part_spec"] == s) for s in ALL_SPECS}

    lines = [
        f"# 분류 시험 공급 순서표 `{seq['sequence_id']}`",
        "",
        f"- 상태: {seq['status']} · 형식 {seq['format']} v{seq['format_version']}",
        f"- content_sha256: `{seq.get('content_sha256', content_sha256(seq))}`",
        f"- 생성: {('seed=' + str(gen['seed']) + ' args=' + dumps_canonical(gen['args'])) if gen else '사람 작성'}"
        f" · {seq['meta']['created_at_utc']}",
        f"- 시간 기준 {seq['timing']['basis']}, 간격 {_cell(seq['timing']['interval_s']) or 'C 기본값(null)'} s",
        f"- 너트 행 포함: {seq['include_nut']} · 규격별 수: " + ", ".join(f"{k} {v}" for k, v in counts.items()),
        "- 이 표는 **독립 정답**이다. 평가 경로에서만 읽는다(코드 규약 §5).",
        "",
        "| # | 규격 | 기대 상자 | 시나리오 | 간격 s | 가로 mm | yaw ° |",
        "|---:|---|---|---|---:|---:|---:|",
    ]

    for row in seq["supplies"]:
        pose = row["start_pose"] or {}
        lines.append(f"| {row['supply_index']} | {row['part_spec']} | {expected_bin(row['part_spec'])} | "
                     f"{row['scenario']} | {_cell(row['interval_s'])} | {_cell(pose.get('lateral_offset_mm'))} | "
                     f"{_cell(pose.get('yaw_deg'))} |")

    return "\n".join(lines) + "\n"


def _csv_number(text: str, where: str, errors: list[str]):
    """CSV 숫자 칸: 빈칸은 None, 그 밖은 유한 실수."""
    if text.strip() == "":
        return None

    try:
        value = float(text)
    except ValueError:
        errors.append(f"{where}: not a number {text!r}")
        return None

    if not math.isfinite(value):
        errors.append(f"{where}: non-finite number {text!r}")
        return None

    return value


def sequence_from_csv(text: str, *, sequence_id: str, include_nut: bool = False,
                      created_at_utc: str | None = None) -> tuple[dict | None, list[str]]:
    """사람이 쓴 CSV → (순서표, 오류 목록). 범위 선언이 없어 시작 자세 값이 있으면 검증기가 거부한다."""
    errors: list[str] = []
    reader = csv.DictReader(io.StringIO(text))

    if tuple(reader.fieldnames or ()) != _CSV_COLUMNS:
        return None, [f"csv header must be {list(_CSV_COLUMNS)}"]

    rows = []

    for line_no, rec in enumerate(reader, start=2):
        where = f"csv line {line_no}"
        idx_text = rec["supply_index"].strip()

        if not idx_text.isdigit():
            errors.append(f"{where}: supply_index must be a positive integer")
            continue

        lateral = _csv_number(rec["lateral_offset_mm"], where, errors)
        yaw = _csv_number(rec["yaw_deg"], where, errors)
        pose = None if lateral is None and yaw is None else {"lateral_offset_mm": lateral, "yaw_deg": yaw}

        rows.append({"supply_index": int(idx_text), "part_spec": rec["part_spec"].strip(),
                     "scenario": rec["scenario"].strip(), "interval_s": _csv_number(rec["interval_s"], where, errors),
                     "start_pose": pose})

    if errors:
        return None, errors

    seq = {
        "format": FORMAT, "format_version": FORMAT_VERSION, "sequence_id": sequence_id, "status": "draft",
        "meta": {"generator": None,
                 "created_at_utc": created_at_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "python": None, "author": None, "confirmer": None, "notes": "csv"},
        "timing": {"basis": "SIM_TIME", "mode": "interval", "interval_s": None},
        "pose_ranges": {"lateral_offset_mm": None, "yaw_deg": None},
        "include_nut": include_nut,
        "supplies": rows,
    }

    errors = validate_sequence(seq)

    if not errors:
        seq["content_sha256"] = content_sha256(seq)

    return seq, errors


def write_outputs(seq: dict, out_dir: str | Path, stem: str | None = None) -> dict[str, Path]:
    """`{stem}.json`·`.csv`·`.md`를 쓴다. 이미 있으면 덮어쓰지 않고 FileExistsError."""
    out = Path(out_dir)
    stem = stem or seq["sequence_id"]

    if not stem or "/" in stem or "\\" in stem or stem in (".", "..") or stem.startswith("."):
        raise ValueError(f"unsafe output stem: {stem!r}")

    paths = {ext: out / f"{stem}.{ext}" for ext in ("json", "csv", "md")}
    existing = [str(p) for p in paths.values() if p.exists()]

    if existing:
        raise FileExistsError(f"refusing to overwrite: {existing}")

    out.mkdir(parents=True, exist_ok=True)
    paths["json"].write_text(to_json(seq), encoding="utf-8")
    paths["csv"].write_text(to_csv(seq), encoding="utf-8")
    paths["md"].write_text(to_markdown(seq), encoding="utf-8")

    return paths
