"""통신 시험 시나리오 정의·기대 결과·근거 절. 실행 로직 없음(순수).

`CASES`는 로컬 모의 상대에서 실행할 케이스, `C_COOPERATION`은 C 협력이 필요해 정의만 하는 항목이다.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    """시나리오 케이스 하나."""

    case_id: str
    scenario: str
    title: str
    expected: str
    basis: str
    fault: bool            # 시험 모드 주입·중계기가 필요한가
    dedupe_id: str | None = None


# S2 변형: (케이스 ID, dedupe 사례, 바꿀 필드, 값 또는 위치 차이, 기대 same/conflict)
S2_VARIANTS = (
    ("S2a", "dedupe-001", None, None, "same"),
    ("S2c_pos", "dedupe-002", "position_dx_mm", 0.005, "same"),
    ("S2b_pos", "dedupe-003", "position_dx_mm", 0.02, "conflict"),
    ("S2c_theta_wrap", "dedupe-004", "theta_deg", 0.001, "same"),  # θ 179.998 → 0.001(180° 주기)
    ("S2b_theta", "dedupe-005", "theta_dd", 0.05, "conflict"),
    ("S2b_image_seq", "dedupe-006", "image_seq_d", 1, "conflict"),
    ("S2b_calibration", "dedupe-007", "calibration_id", "commtest_cal_v2", "conflict"),
    ("S2b_timestamp_base", "dedupe-008", "timestamp_base", "PHONE_CAPTURE", "conflict"),
)

CASES = (
    Case("S1", "S1", "정상 MOVE N회", "직접 ACK → first_ack, 최종 REACHED", "TCP §8.1~8.4, A §10, 계획서 §7(참조값)", False),
    *(Case(cid, "S2", f"같은 ID 재송신 ({dd})",
           "새 실행 없음 · 저장 상태 응답" if exp == "same" else "ID_CONFLICT 거부 · 실행 없음",
           "TCP §7.2 4번·§7.3", True, dd) for cid, dd, _, _, exp in S2_VARIANTS),
    Case("S3a", "S3", "ACK 삼킨 뒤 단절 → 재접속 GET_STATUS", "accepted_by_status, MOVE 재송신 없음", "TCP §8.4~8.6", True),
    Case("S3b", "S3", "ACK·알림 삼킴(연결 유지) → 재전송", "accepted_after_resend, 새 실행 없음", "TCP §8.3·§7.2", True),
    Case("S3c", "S3", "그 명령 응답 전부 삼킴(연결 유지)", "재전송 max_resend회 뒤 unconfirmed, 새 ID 자동 발급 없음", "TCP §8.3", True),
    Case("S3d", "S3", "재접속 후 같은 MOVE(서버 중복 방어)", "새 실행 없음 · 저장 상태 응답", "TCP §8.6, §13 연결 복구 시험", True),
    Case("S4a", "S4", "이동 중 CANCEL", "FAILED(CANCELLED, EXECUTION)", "TCP §8.7, §13.1", False),
    Case("S4b", "S4", "실행 중 다른 ID MOVE", "BUSY 거부. D 클라이언트 execute는 준비 조회에서 미전송", "TCP §7.2 5번, §13.1", True),
    Case("S4c", "S4", "실행 중 자기 명령 재전송", "현재 상태(BUSY 아님)", "TCP §7.2 4번, §13.1", True),
)

C_COOPERATION = (
    ("C1", "서버 재시작 후 재시작 전 ID로 같은 MOVE", "새 실행 없이 저장된 최종 상태"),
    ("C2", "명령 로그 손상 뒤 서버 시작", "RECOVERY_FAILED, 새 MOVE는 NOT_READY"),
    ("C3", "ACK 직후 서버 kill -9 → 재시작", "그 ID GET_STATUS가 UNKNOWN_COMMAND 아님"),
    ("C4", "REACHED 직후 서버 kill -9 → 재시작", "GET_STATUS가 REACHED, reach_error 같음"),
    ("C5", "기록 실패·복구·초기화 관련 §13.1 행", "§13.1 표 기대 결과"),
)

SCENARIOS = ("S1", "S2", "S3", "S4")


def cases_of(scenario: str) -> tuple[Case, ...]:
    """시나리오의 케이스 목록. 모르는 이름은 ValueError."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r} (expected one of {SCENARIOS})")

    return tuple(c for c in CASES if c.scenario == scenario)
