"""시험용 B 규격 사본(#285): 저장소 `config/calibration/part_specs.json`(B 파일)의 지금 값에 기대지 않도록 고정 값을 둔다.

- `pre_b12()`: B-12 전(git 7e97617, 2026-11-17 판). BOLT_A M6x12 18 ± 4 mm, 높이 모두 0.105 m, 길이 범위 없음.
- `post_b12()`: B-12 뒤(git 94c8590, 2026-11-22 판). BOLT_A M3x20 0.102 m, 길이 범위 17~27·39~52 mm(허용 6 mm도 남음).
값은 각 판에서 `note`만 뺀 것이다. 부르는 쪽이 고쳐도 되도록 매번 새 사본을 돌려준다.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

_NUT = {"class_name": "nut", "label": "M6", "across_flats": 0.01, "across_corners": 0.01105, "size_tolerance": 0.003,
        "height": 0.0052, "grasp_opening": 0.02, "bin": "NUT", "action": "PASS_THROUGH", "plane_z": 0.1052}
_BOLT_B_PRE = {"class_name": "bolt", "label": "M6x40", "total_length": 0.046, "length_tolerance": 0.006,
               "head_diameter": 0.01, "shaft_diameter": 0.006, "grasp_opening": 0.02, "bin": "BOLT_B",
               "action": "PICK_PLACE", "plane_z": 0.105, "head_height": 0.006}

_PRE_B12 = {
    "format_version": 1, "status": "draft-2026-11-17", "plane_z": 0.105, "bolt_min_aspect": 1.5,
    "specs": {
        "BOLT_A": {"class_name": "bolt", "label": "M6x12", "total_length": 0.018, "length_tolerance": 0.004,
                   "head_diameter": 0.01, "shaft_diameter": 0.006, "grasp_opening": 0.02, "bin": "BOLT_A",
                   "action": "PICK_PLACE", "plane_z": 0.105, "head_height": 0.006},
        "BOLT_B": _BOLT_B_PRE,
        "NUT": _NUT,
    },
}

_POST_B12 = {
    "format_version": 1, "status": "draft-2026-11-22", "plane_z": 0.105, "bolt_min_aspect": 1.5,
    "specs": {
        "BOLT_A": {"class_name": "bolt", "label": "M3x20", "total_length": 0.023, "length_min": 0.017, "length_max": 0.027,
                   "length_tolerance": 0.006, "head_diameter": 0.005, "shaft_diameter": 0.003, "grasp_opening": 0.02,
                   "bin": "BOLT_A", "action": "PICK_PLACE", "plane_z": 0.102, "head_height": 0.003},
        "BOLT_B": {**_BOLT_B_PRE, "length_min": 0.039, "length_max": 0.052},
        "NUT": _NUT,
    },
}


def pre_b12() -> dict:
    """B-12 전 규격 사본(허용 판정만)."""
    return copy.deepcopy(_PRE_B12)


def post_b12() -> dict:
    """B-12 뒤 규격 사본(범위 판정)."""
    return copy.deepcopy(_POST_B12)


def write_specs(d: dict, path) -> Path:
    """규격 사전을 JSON 파일로 쓴다(`run_pipeline.PART_SPECS_PATH` 주입용)."""
    p = Path(path)
    p.write_text(json.dumps(d), encoding="utf-8")

    return p
