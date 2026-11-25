"""C 중계기 center 대체 — 사진마다 보내기(#426 나, ROS 없음).

`--center-fallback-every-frame`을 켜면:
- 대체 원본을 0.1 s마다 한 장이 아니라 A가 번호를 붙인 링크 사진마다 보낸다(주기 0.02 s 타이머).
- 대체 사진의 번호표는 **그 사진의 것**이다: 원본 사진과 A detect 번호를 촬영 시각(capture_stamp_ns)으로 짝짓는다.
- B center 영상이 「마지막 대체보다 뒤 번호」로 보류되면 대체 구간을 끝내고 다시 기다린다. 그래야 다음 B 영상이 앞서서 나간다.
#426 적용(추가 지시 1): 중계기 기본값은 기다림 0.3 s·사진마다 보내기 켬. `--center-fallback-legacy`는 예전 동작(1.0 s·0.1 s 주기).
판정기(`CenterFallback`) 자체의 기본값은 예전 그대로라 `test_display_relay_fallback.py`는 예전 동작을 계속 본다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

from robot_server_package.display_relay import CenterFallback, build_parser, fallback_period_s, fallback_settings  # noqa: E402

FRAME_NS = 25_000_000   # 시뮬 40장/s


class Clock:
    """단조 시계 대역(시험이 시각을 옮긴다)."""

    def __init__(self) -> None:
        """100 s에서 시작."""
        self.t = 100.0

    def __call__(self) -> float:
        """지금 시각."""
        return self.t


def detect(seq, gen=1, cg=1, config="det-cfg", cal="cal-1"):
    """A detect 메타 대역(촬영 시각 = 번호 × 25 ms)."""
    return {"image_seq": seq, "job_seq": seq, "generation": gen, "capture_generation": cg, "calibration_id": cal,
            "detector_config_id": config, "capture_stamp_ns": seq * FRAME_NS}


def center(seq, gen=1, cg=1, config="b-cfg"):
    """B center 메타 대역."""
    return {"image_seq": seq, "job_seq": seq, "generation": gen, "capture_generation": cg, "detector_config_id": config}


def stamps(*seqs):
    """중계기가 가진 원본 사진 촬영 시각(새것 먼저)."""
    return [s * FRAME_NS for s in sorted(seqs, reverse=True)]


@pytest.fixture
def fb():
    """center를 고르고 사진마다 보내기를 켠 판정기(기다림 0.3 s)."""
    clock = Clock()
    f = CenterFallback(0.3, clock=clock, every_frame=True)
    f.clock = clock
    f.set_selected("center")
    return f


def test_pairs_image_with_its_own_numbers(fb):
    """원본 사진은 같은 촬영 시각의 A 번호로만 나간다. 번호가 아직 없는 새 사진은 건너뛰고 짝이 있는 가장 새 사진을 보낸다."""
    for seq in (1, 2, 3):
        fb.note_detect(detect(seq))

    fb.clock.t += 0.35
    stamp, numbers = fb.next_fallback_for(stamps(1, 2, 3, 4))   # 4번은 A 번호가 아직 없음
    assert (stamp, numbers["image_seq"], numbers["job_seq"]) == (3 * FRAME_NS, 3, 3)
    assert fb.next_fallback_for(stamps(1, 2, 3, 4)) is None      # 같은 사진을 다시 보내지 않음
    fb.note_detect(detect(4))
    stamp, numbers = fb.next_fallback_for(stamps(2, 3, 4))
    assert (stamp, numbers["image_seq"]) == (4 * FRAME_NS, 4)
    counts = fb.take_counts()
    assert (counts["fallback_tx"], counts["fallback_starts"]) == (2, 1)


def test_sends_every_new_frame_not_every_100_ms(fb):
    """사진마다: 새 짝이 생길 때마다 곧바로 다음 대체가 나간다(0.1 s를 기다리지 않음). 타이머 주기는 0.02 s."""
    fb.note_detect(detect(1))
    fb.clock.t += 0.35
    assert fb.next_fallback_for(stamps(1)) is not None

    for seq in range(2, 8):
        fb.clock.t += 0.025 / 0.64          # 실제 시간으로 사진 한 장 간격(배율 0.64)
        fb.note_detect(detect(seq))
        assert fb.next_fallback_for(stamps(seq, seq - 1))[1]["image_seq"] == seq

    assert fallback_period_s(every_frame=True) == pytest.approx(0.02)
    assert fallback_period_s(every_frame=False) == pytest.approx(0.1)


def test_held_center_ends_fallback_then_b_returns(fb):
    """대체 중 B 영상이 뒤 번호로 보류되면 대체 구간을 끝낸다. 기다림 동안 대체를 더 보내지 않아, 다음 B 영상이 앞서 나간다."""
    assert fb.center_allowed(center(10))

    for seq in range(11, 21):
        fb.note_detect(detect(seq))

    fb.clock.t += 0.35
    assert fb.next_fallback_for(stamps(20))[1]["image_seq"] == 20
    assert not fb.center_allowed(center(19))     # B가 늦어 뒤 번호: 보류하고 대체 구간을 끝냄
    fb.note_detect(detect(21))
    fb.clock.t += 0.05
    assert fb.next_fallback_for(stamps(21)) is None   # 다시 기다리는 중: 대체가 B를 앞지르지 않음
    assert fb.center_allowed(center(21))         # 마지막 대체(20)보다 앞선 B가 나감
    counts = fb.take_counts()
    assert (counts["center_behind_fallback"], counts["fallback_ended_by_center"], counts["fallback_ends"]) == (1, 1, 1)


def test_never_goes_backwards(fb):
    """대체·B 영상을 섞어도 나가는 번호는 늘 앞으로만 간다."""
    sent = []
    assert fb.center_allowed(center(5))
    sent.append(5)

    for seq in range(6, 15):
        fb.note_detect(detect(seq))

    fb.clock.t += 0.35
    sent.append(fb.next_fallback_for(stamps(*range(6, 15)))[1]["image_seq"])   # 14

    for seq in (12, 13, 14, 15, 16):
        if fb.center_allowed(center(seq)):
            sent.append(seq)

    fb.note_detect(detect(15))
    assert fb.next_fallback_for(stamps(13, 15)) is None   # B가 방금 나갔으니 다시 기다림
    assert sent == sorted(set(sent)) == [5, 14, 15, 16]


def test_numbers_without_stamp_are_not_paired(fb):
    """촬영 시각이 없는 A 번호는 짝을 지을 수 없어 쓰지 않는다(다른 사진의 번호를 붙이지 않음)."""
    meta = detect(3)
    del meta["capture_stamp_ns"]
    fb.note_detect(meta)
    fb.clock.t += 0.5
    assert fb.next_fallback_for(stamps(3)) is None
    assert fb.take_counts()["fallback_unmatched"] == 1


def test_default_mode_unchanged_after_held_center():
    """끄면(기본) 지금처럼 보류된 B 영상이 대체 구간을 끝내지 않고, 다음 대체가 바로 나간다."""
    clock = Clock()
    f = CenterFallback(1.0, clock=clock)
    f.set_selected("center")
    assert f.center_allowed(center(10))
    f.note_detect(detect(20))
    clock.t += 1.5
    assert f.next_fallback()["image_seq"] == 20
    assert not f.center_allowed(center(19))
    f.note_detect(detect(22))
    assert f.next_fallback()["image_seq"] == 22
    assert f.take_counts().get("fallback_ended_by_center", 0) == 0


def test_shorter_wait():
    """가: 기다림 0.3 s — 0.25 s에는 아직, 0.35 s에는 대체한다."""
    clock = Clock()
    f = CenterFallback(0.3, clock=clock)
    f.set_selected("center")
    f.note_detect(detect(4))
    clock.t += 0.25
    assert f.next_fallback() is None
    clock.t += 0.1
    assert f.next_fallback()["image_seq"] == 4


def test_relay_defaults_are_new_behaviour():
    """적용: 인자 없이 띄우면 기다림 0.3 s, 사진마다 보내기 켬(타이머 0.02 s)."""
    stale_s, every_frame = fallback_settings(build_parser().parse_args([]))
    assert (stale_s, every_frame) == (0.3, True)
    assert fallback_period_s(every_frame) == pytest.approx(0.02)


def test_legacy_option_restores_old_behaviour():
    """`--center-fallback-legacy`: 예전 동작(1.0 s, 0.1 s마다 한 장, 가장 새 A 번호를 빌림)."""
    stale_s, every_frame = fallback_settings(build_parser().parse_args(["--center-fallback-legacy"]))
    assert (stale_s, every_frame) == (1.0, False)
    assert fallback_period_s(every_frame) == pytest.approx(0.1)


def test_each_switch_alone():
    """두 수정은 따로 끄고 켠다: 사진마다만 끔(기다림 0.3 s 유지), 기다림만 바꿈(사진마다 유지), 예전 동작 + 값 지정."""
    parse = lambda *a: fallback_settings(build_parser().parse_args(list(a)))  # noqa: E731
    assert parse("--no-center-fallback-every-frame") == (0.3, False)
    assert parse("--center-fallback-s", "0.5") == (0.5, True)
    assert parse("--center-fallback-legacy", "--center-fallback-s", "0.3") == (0.3, False)
    assert parse("--center-fallback-legacy", "--center-fallback-every-frame") == (1.0, True)
    assert parse("--center-fallback-s", "0") == (0.0, True)       # 0이면 대체를 끔(예전과 같음)


def test_fallback_not_equal_to_last_center(fb):
    """B 영상이 나간 뒤 대체는 그 B 영상보다 앞선 사진만 쓴다(같은 번호를 다시 내면 D가 버림)."""
    for seq in (29, 30, 31):
        fb.note_detect(detect(seq))

    assert fb.center_allowed(center(30))
    fb.clock.t += 0.35
    assert fb.next_fallback_for(stamps(30)) is None                # 방금 나간 B와 같은 사진
    assert fb.next_fallback_for(stamps(30, 31))[1]["image_seq"] == 31
