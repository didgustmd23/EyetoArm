"""C 중계기 center 대체 시험(#379, ROS 없음).

D가 center를 고른 상태에서 center 영상이 1 s 오지 않으면, 중계기가 링크 원본을 center 이름표로 보낸다.
번호는 최신 A detect 메타에서 빌리고, 마지막 대체보다 앞서지 않는 center는 보내지 않는다(D가 버리므로).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

from robot_server_package.display_relay import (  # noqa: E402
    FALLBACK_PRODUCER, CenterFallback, LatestFrameServer, fallback_metadata)


class Clock:
    """단조 시계 대역(시험이 시각을 옮긴다)."""

    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def detect(seq, job=None, gen=1, cg=1, config="det-cfg", cal="cal-1"):
    """A detect 메타 대역."""
    return {"image_seq": seq, "job_seq": seq if job is None else job, "generation": gen, "capture_generation": cg,
            "calibration_id": cal, "detector_config_id": config}


def center(seq, job=None, gen=1, cg=1, config="b-cfg"):
    """B center 메타 대역."""
    return {"image_seq": seq, "job_seq": seq if job is None else job, "generation": gen, "capture_generation": cg,
            "detector_config_id": config}


@pytest.fixture
def fb():
    """center를 고른 대체 판정기(1 s)."""
    clock = Clock()
    f = CenterFallback(1.0, clock=clock)
    f.clock = clock
    f.set_selected("center")
    return f


def test_fallback_after_stale_with_detect_numbers(fb):
    """① center가 1 s 끊기면 대체하고, 번호는 최신 A detect 값·설정은 마지막 center 값이다."""
    assert fb.center_allowed(center(10))
    fb.note_detect(detect(12))
    fb.clock.t += 0.9
    assert fb.next_fallback() is None            # 아직 1 s 전
    fb.clock.t += 0.2
    numbers = fb.next_fallback()
    assert numbers["image_seq"] == 12 and numbers["job_seq"] == 12
    assert numbers["detector_config_id"] == "b-cfg" and numbers["calibration_id"] == "cal-1"
    assert fb.take_counts()["fallback_starts"] == 1


def test_same_numbers_not_resent(fb):
    """② 새 A 번호가 없으면 같은 번호로 다시 보내지 않는다. 새 번호가 오면 또 보낸다."""
    fb.note_detect(detect(5))
    fb.clock.t += 1.5
    assert fb.next_fallback() is not None
    assert fb.next_fallback() is None
    fb.note_detect(detect(6))
    assert fb.next_fallback()["image_seq"] == 6
    counts = fb.take_counts()
    assert (counts["fallback_tx"], counts["fallback_starts"]) == (2, 1)


def test_center_behind_fallback_held_then_resumes(fb):
    """③ 대체 중 center가 오면, 마지막 대체보다 앞서지 않은 것은 보내지 않고 앞선 것부터 보내며 대체를 끝낸다."""
    assert fb.center_allowed(center(10))
    fb.note_detect(detect(20))
    fb.clock.t += 1.5
    assert fb.next_fallback()["image_seq"] == 20
    assert not fb.center_allowed(center(18))     # B가 늦어 번호가 뒤
    assert not fb.center_allowed(center(20))     # 같은 번호도 D가 버림
    assert fb.center_allowed(center(21))
    counts = fb.take_counts()
    assert (counts["center_behind_fallback"], counts["fallback_ends"]) == (2, 1)
    fb.note_detect(detect(25))
    assert fb.next_fallback() is None            # 방금 center가 나갔으므로 다시 1 s 기다림


def test_first_center_never_arrived(fb):
    """④ center를 고른 뒤 한 번도 center가 없으면 고른 때부터 1 s 뒤 대체한다(설정은 A detect 값)."""
    fb.note_detect(detect(3))
    fb.clock.t += 1.0
    numbers = fb.next_fallback()
    assert numbers["image_seq"] == 3 and numbers["detector_config_id"] == "det-cfg"


def test_no_detect_numbers_no_fallback(fb):
    """⑤ A detect 번호가 없으면 대체하지 않고, 그 구간을 한 번만 센다."""
    fb.clock.t += 2.0
    assert fb.next_fallback() is None
    assert fb.next_fallback() is None
    assert fb.take_counts()["fallback_no_numbers"] == 1


def test_detect_source_change_forgets_numbers():
    """⑥ A detect 실행·세대가 바뀌면 중계기가 번호와 마지막 대체 기준을 지운다(옛 번호로 보내지 않음)."""
    clock = Clock()
    relay = LatestFrameServer("127.0.0.1", 0, "relay-run", fallback=CenterFallback(1.0, clock=clock))
    try:
        relay.select_link_variant("center")
        relay.update_detect_source("a-run-1", 1, 1)
        relay.fallback.note_detect(detect(40))
        clock.t += 1.5
        assert relay.fallback.next_fallback()["image_seq"] == 40
        assert relay.update_detect_source("a-run-2", 1, 1) is True
        assert relay.fallback.next_fallback() is None
        assert relay.fallback.center_allowed(center(5))   # 기준이 지워져 작은 번호의 center도 보냄
        relay.invalidate_detect_source("a-run-2")
        clock.t += 1.5
        assert relay.fallback.next_fallback() is None
    finally:
        relay.close()


def test_disabled_and_other_variants():
    """⑦ 0이면 끄고, center가 아닌 종류를 고르면 대체하지 않는다."""
    clock = Clock()
    off = CenterFallback(0.0, clock=clock)
    off.set_selected("center")
    off.note_detect(detect(1))
    clock.t += 5
    assert off.next_fallback() is None

    on = CenterFallback(1.0, clock=clock)
    on.set_selected("raw")
    on.note_detect(detect(1))
    clock.t += 5
    assert on.next_fallback() is None
    with pytest.raises(ValueError):
        CenterFallback(-1.0)


def test_select_through_server_resets_wait():
    """중계기 선택 요청이 판정기에 전해지고, center를 다시 고르면 기다림을 새로 시작한다."""
    clock = Clock()
    relay = LatestFrameServer("127.0.0.1", 0, "relay-run", fallback=CenterFallback(1.0, clock=clock))
    try:
        relay.fallback.note_detect(detect(7))
        relay.select_link_variant("center")
        clock.t += 0.5
        relay.select_link_variant("raw")
        relay.select_link_variant("center")
        clock.t += 0.6
        assert relay.fallback.next_fallback() is None    # 다시 고른 뒤 0.6 s
        clock.t += 0.5
        assert relay.fallback.next_fallback()["image_seq"] == 7
    finally:
        relay.close()


def test_fallback_metadata_keys():
    """대체 패킷 메타: center 이름표·A 번호·원본 시각·진단 키, B 결과·생산자 실행 키는 없다."""
    numbers = dict(detect(9), detector_config_id="b-cfg")
    meta = fallback_metadata("relay-run", numbers, 12, 500, "link_cam_optical_frame")
    assert (meta["channel"], meta["variant"], meta["source_id"], meta["run_id"]) == ("link", "center", "gz_link", "relay-run")
    assert (meta["image_seq"], meta["job_seq"], meta["generation"], meta["capture_generation"]) == (9, 9, 1, 1)
    assert meta["capture_stamp_ns"] == 12_000_000_500 and meta["producer"] == FALLBACK_PRODUCER
    assert not {"result", "producer_run_id", "source_run_id"} & set(meta)


def test_detect_gap_recorded(fb):
    """진단: A detect 번호가 1 s 이상 끊겼다가 오면 공백 길이와 앞뒤 순서를 남긴다(세대 변경 판별용)."""
    fb.note_detect(detect(1))
    fb.clock.t += 0.5
    fb.note_detect(detect(2))
    fb.clock.t += 1.2
    fb.note_detect(detect(9, gen=2))
    assert fb.take_counts()["detect_gaps_ge1s"] == 1
    wall, gap, before, after = fb.detect_gaps[0]
    assert gap == pytest.approx(1.2) and before == (1, 1, 2, 2) and after == (2, 1, 9, 9)
