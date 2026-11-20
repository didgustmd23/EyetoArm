"""#213 실시간 B 중심점 주석 영상 표시 전용 송출 시험(U1~U17).

근거: 계획 `_workspace/213_builder2_live_center_display_plan_r0.md`·`_r1.md`(reviewer PASS, 사용자 승인 2026-11-20, Q7 = 사유 화면 낮은
빈도: status_enabled true, hold 1.0 s, 최대 1 Hz). 실제 ROS·Gazebo는 쓰지 않는다: 표시 발행자는 가짜(`FakePublisher`), CLI 시험은 기존
실시간 기록 전용 시험의 가짜 수신 연결·가짜 모델·가짜 TF를 그대로 쓴다.
- 표시 핵심 시험은 주입 시계(`FakeClock`)로 hold·주기 판정을 정한다. 표시 스레드는 진짜 스레드이며 `loop_count`로 한 바퀴를 기다린다.
"""
from __future__ import annotations

import io
import json
import math
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from eyetoarm.vision import live_center_display as lcd
from eyetoarm.vision.live_center_display import (CenterDisplayConfig, CenterDisplayConfigError, LiveCenterDisplay,
                                                 NO_STATUS_REASONS, parse_center_display)
from test_run_pipeline_capture_observation import model_with
from test_run_pipeline_live_observation import CONFIG, FRAME_ID, FakeTf, Scenario
from test_run_pipeline_live_observation import Base as LiveBase
from test_view_bias import write_bias_config

ROOT = Path(__file__).resolve().parents[2]
CENTER_CONFIG = ROOT / "config" / "vision" / "live_observation_center_test.json"
BANNER_H = 44
S = 1_000_000_000


def cfg(**over) -> CenterDisplayConfig:
    """시험 설정(기본 = 승인 값, graph 확인은 사실상 끔)."""
    base = dict(discovery_wait_s=0.01, graph_check_s=1e6, publish_fail_limit=3, qos_depth=2, close_limit_s=2.0,
                status_enabled=True, status_hold_s=1.0, status_max_hz=1.0)
    base.update(over)

    return CenterDisplayConfig(**base)


class FakeClock:
    """주입 단조 시계(ns). 시험이 직접 옮긴다."""

    def __init__(self):
        """0에서 시작."""
        self.now = 0
        self.lock = threading.Lock()

    def __call__(self):
        """현재 ns."""
        with self.lock:
            return self.now

    def set_s(self, t_s: float):
        """초 단위로 옮긴다."""
        with self.lock:
            self.now = int(round(t_s * S))


class FakePublisher:
    """가짜 표시 발행자: 받은 프레임·발행 시각·닫기 기록. 막기·실패·발행자 수 주입."""

    def __init__(self, clock=None, *, fail=0, publishers=1, gate=None, on_close=None):
        """`fail`: 처음 n회 예외(-1이면 항상). `gate`: 발행 때 기다릴 Event. `on_close()`: 닫을 때 부를 함수."""
        self.clock = clock or time.monotonic_ns
        self.frames = []
        self.times = []
        self.fail = fail
        self.publishers = publishers
        self.gate = gate
        self.entered = threading.Event()
        self.closed = 0
        self.on_close = on_close
        self.close_notes = []
        self.startup_check = "fake"

    def publish(self, frame):
        """발행 대역."""
        self.entered.set()

        if self.gate is not None:
            self.gate.wait(5.0)

        if self.fail:
            self.fail -= 1 if self.fail > 0 else 0
            raise RuntimeError("publish failed (injected)")

        self.frames.append(frame)
        self.times.append(self.clock())

    def count_publishers(self):
        """같은 토픽 발행자 수(자기 포함)."""
        return self.publishers

    def close(self, limit_s):
        """닫기 기록."""
        if self.on_close is not None:
            self.close_notes.append(self.on_close())

        self.closed += 1


def frame(stamp_ns: int, *, w=64, h=48, value=90):
    """읽기 전용 원본 영상을 가진 프레임 대역(`ros_image.owned_bgr`처럼 write=False)."""
    img = np.full((h, w, 3), value, np.uint8)
    img[h // 2:, :, 1] = 200
    img.setflags(write=False)

    return SimpleNamespace(image=img, ros_stamp_ns=stamp_ns, optical_frame=FRAME_ID, image_seq=stamp_ns)


def annotated_of(fr):
    """주석 배열 대역(원본 복사 + 표시)."""
    img = np.array(fr.image, copy=True)
    img[-5:, :, 2] = 255

    return img


IDS = {"job_seq": 7, "generation": 1}


def balance(c: dict) -> int:
    """넣은 항목의 행방 합계(발행·버림·덮어씀·닫힘 뒤·비움)."""
    return sum(c[k] for k in ("published_accepted", "published_status", "accepted_replaced", "accepted_older_dropped",
                              "status_replaced", "status_suppressed_stale", "status_older_ignored", "cleared_on_reset",
                              "offered_after_close", "dropped_not_live"))


class DisplayBase(unittest.TestCase):
    """표시 스레드를 돌리고 한 바퀴씩 기다리는 도우미."""

    def make(self, clock=None, publisher=None, **over):
        """표시 송출기를 만들고 시작한다(정리 등록)."""
        self.clock = clock or FakeClock()
        self.pub = publisher or FakePublisher(self.clock)
        self.disp = LiveCenterDisplay("run-test-0001", cfg(**over), self.pub, clock_ns=self.clock)
        self.disp.start()
        self.addCleanup(lambda: self.disp.close(1.0) if self.disp.thread.is_alive() else None)

        return self.disp

    def settle(self, n=3):
        """표시 스레드가 n 바퀴 더 돌 때까지 기다린다(주입 시계는 그대로)."""
        start = self.disp.loop_count
        end = time.monotonic() + 2.0

        while self.disp.loop_count < start + n and time.monotonic() < end:
            time.sleep(0.002)

    def kinds(self):
        """발행한 (종류, stamp) 목록."""
        return [(f.kind, f.stamp_ns) for f in self.pub.frames]


# ---------------------------------------------------------------------- 설정

class ConfigTest(unittest.TestCase):
    """`center_display` 블록 검사와 예시 설정."""

    def test_repo_config_has_approved_values(self):
        """예시 설정은 status=test_only이고 Q7 승인값(켬·hold 1.0 s·1 Hz)이다."""
        from eyetoarm.vision.live_observation_link import load_live_observation_config

        c = load_live_observation_config(str(CENTER_CONFIG))
        self.assertEqual((c.center_display.status_enabled, c.center_display.status_hold_s, c.center_display.status_max_hz),
                         (True, 1.0, 1.0))
        self.assertIsNone(load_live_observation_config(str(CONFIG)).center_display)

    def test_bad_blocks_refused(self):
        """모르는 키·누락·0 이하·bool 아닌 status_enabled·정수 아닌 한도는 오류."""
        good = cfg().to_json()

        for bad in ({**good, "x": 1}, {k: v for k, v in good.items() if k != "status_hold_s"},
                    {**good, "status_max_hz": 0}, {**good, "status_enabled": 1}, {**good, "publish_fail_limit": 2.5},
                    {**good, "qos_depth": True}, {**good, "graph_check_s": float("inf")}, []):
            with self.subTest(bad=bad), self.assertRaises(CenterDisplayConfigError):
                parse_center_display(bad)

        self.assertEqual(parse_center_display(good), cfg())


# ---------------------------------------------------------------------- 표시 핵심(가짜 시계)

class AcceptedTest(DisplayBase):
    """U2·U4·U17: 수락 주석 발행, 수락끼리 단조, 원본 불변."""

    def test_accepted_published_with_stamp_and_frame_id(self):
        """U2: 수락 1건 → 1회 발행, stamp·frame_id 복사, 띠 밖 픽셀은 주석 배열과 같다."""
        d = self.make()
        fr = frame(5 * S + 123)
        ann = annotated_of(fr)
        d.offer_accepted(fr, ann, None, IDS)
        self.settle()
        self.assertEqual(self.kinds(), [("accepted", 5 * S + 123)])
        out = self.pub.frames[0]
        self.assertEqual(out.frame_id, FRAME_ID)
        self.assertTrue(np.array_equal(out.image[BANNER_H + 1:], ann[BANNER_H + 1:]))
        self.assertFalse(np.array_equal(out.image[:BANNER_H], ann[:BANNER_H]))   # 띠가 그려짐

    def test_accepted_monotonic_only_among_accepted(self):
        """U4: 발행한 수락 stamp 이하 수락은 버림. 상태 화면 stamp는 수락을 버리게 하지 않는다."""
        d = self.make(status_hold_s=0.001, status_max_hz=1000.0)
        self.clock.set_s(10.0)
        d.offer_status(frame(9 * S), "invalid_observation", IDS)    # 상태 화면(stamp 9 s) 먼저 발행
        self.settle()
        d.offer_accepted(frame(8 * S), annotated_of(frame(8 * S)), None, IDS)   # 더 이른 stamp 수락도 발행된다
        self.settle()
        d.offer_accepted(frame(7 * S), annotated_of(frame(7 * S)), None, IDS)   # 수락 8 s보다 이전 → 버림
        self.settle()
        self.assertEqual(self.kinds(), [("status", 9 * S), ("accepted", 8 * S)])
        self.assertEqual(d.counts["accepted_older_dropped"], 1)

    def test_latest_accepted_replaces_pending(self):
        """U4: 발행 전 칸의 수락은 더 새 수락이 덮어쓴다(최신 우선)."""
        gate = threading.Event()
        clock = FakeClock()
        d = self.make(clock, FakePublisher(clock, gate=gate))
        d.offer_accepted(frame(1 * S), annotated_of(frame(1 * S)), None, IDS)   # 발행 중(막힘)
        self.assertTrue(self.pub.entered.wait(2.0))
        d.offer_accepted(frame(2 * S), annotated_of(frame(2 * S)), None, IDS)
        d.offer_accepted(frame(3 * S), annotated_of(frame(3 * S)), None, IDS)
        gate.set()
        self.settle()
        self.assertEqual(self.kinds(), [("accepted", 1 * S), ("accepted", 3 * S)])
        self.assertEqual(d.counts["accepted_replaced"], 1)

    def test_source_arrays_not_modified(self):
        """U17: 발행 뒤에도 원본(읽기 전용)과 주석 배열은 그대로다."""
        d = self.make(status_hold_s=0.001, status_max_hz=1000.0)
        self.clock.set_s(5.0)
        fr, fr2 = frame(1 * S), frame(2 * S)
        ann = annotated_of(fr2)
        before, before_ann = fr.image.copy(), ann.copy()
        d.offer_status(fr, "tf_unavailable", IDS)
        self.settle()
        d.offer_accepted(fr2, ann, None, IDS)
        self.settle()
        self.assertEqual([k for k, _ in self.kinds()], ["status", "accepted"])
        self.assertTrue(np.array_equal(fr.image, before))
        self.assertFalse(fr.image.flags.writeable)
        self.assertTrue(np.array_equal(ann, before_ann))


class StatusScreenTest(DisplayBase):
    """U3·U14·U15·U16: 상태 화면 규칙(r1 F1)."""

    def test_excluded_reasons_never_offered(self):
        """U3: not_accepted·superseded·dropped_on_close·run_ended는 사유 자리에 들어가지 않고, 나머지는 들어간다."""
        d = self.make(status_hold_s=0.001, status_max_hz=1000.0)
        self.clock.set_s(5.0)

        for i, reason in enumerate(sorted(NO_STATUS_REASONS)):
            d.offer_status(frame((i + 1) * S), reason, IDS)

        self.settle()
        self.assertEqual((self.kinds(), d.counts["status_offered"]), ([], 0))

        for i, reason in enumerate(("invalid_observation", "tf_late", "b_input_rejected", "stale_after_tf", "not_ready",
                                    "generation_discarded", "calibration_mismatch")):
            self.clock.set_s(6.0 + i)                # 주기 상한을 넘기도록 시계를 옮긴다
            d.offer_status(frame((10 + i) * S), reason, IDS)
            self.settle()

        self.assertEqual(d.counts["status_offered"], 7)
        self.assertEqual([k for k, _ in self.kinds()], ["status"] * 7)

    def test_alternating_valid_invalid_loses_no_accepted(self):
        """U14: V I V I … 40장(제어 흐름 0.1 s 간격, 수락은 0.3 s 늦게) → 수락 20건 모두 발행, 버림 0, 상태 화면 0."""
        d = self.make()
        events = []

        for i in range(40):
            if i % 2:
                events.append((0.1 * i, "status", i))
            else:
                events.append((0.1 * i + 0.3, "accepted", i))

        for t, kind, i in sorted(events):
            self.clock.set_s(t)
            fr = frame(i * S // 10)

            if kind == "status":
                d.offer_status(fr, "invalid_observation", IDS)
            else:
                d.offer_accepted(fr, annotated_of(fr), None, IDS)

            self.settle()

        self.assertEqual(d.counts["published_accepted"], 20)
        self.assertEqual((d.counts["accepted_older_dropped"], d.counts["accepted_replaced"]), (0, 0))
        self.assertEqual(d.counts["published_status"], 0)
        self.assertTrue(d.close(1.0))
        self.assertEqual(balance(d.counts), d.counts["accepted_offered"] + d.counts["status_offered"])

    def test_rate_limit_and_hold_restart(self):
        """U15: 무효만 5 s → 첫 상태 화면은 hold(1 s) 뒤, 발행 수 ≤ ⌈5×1 Hz⌉. 수락 1건 뒤 1 s 안에는 상태 화면 0."""
        d = self.make()
        step = 0.05

        for k in range(1, int(5.0 / step) + 1):
            t = k * step
            self.clock.set_s(t)
            d.offer_status(frame(int(t * S)), "invalid_observation", IDS)
            self.settle()

        st_times = [t / S for t, f in zip(self.pub.times, self.pub.frames) if f.kind == "status"]
        self.assertTrue(st_times)
        self.assertGreaterEqual(st_times[0], 1.0)
        self.assertLessEqual(len(st_times), math.ceil(5.0 * 1.0))
        self.assertTrue(all(b - a >= 1.0 - 1e-9 for a, b in zip(st_times, st_times[1:])))

        self.clock.set_s(5.0)
        d.offer_accepted(frame(int(5.0 * S) + 1), annotated_of(frame(int(5.0 * S) + 1)), None, IDS)
        self.settle()
        n_status = d.counts["published_status"]

        for k in range(1, 19):                      # 5.05 … 5.90 s: hold 안
            t = 5.0 + k * step
            self.clock.set_s(t)
            d.offer_status(frame(int(t * S) + 2), "invalid_observation", IDS)
            self.settle()

        self.assertEqual(d.counts["published_status"], n_status)
        self.clock.set_s(6.05)                      # hold 지남 → 다시 허용
        self.settle()
        self.assertEqual(d.counts["published_status"], n_status + 1)

    def test_status_disabled(self):
        """U16: status_enabled=false면 상태 화면 0, 수락 주석은 그대로 발행."""
        d = self.make(status_enabled=False, status_hold_s=0.001, status_max_hz=1000.0)
        self.clock.set_s(10.0)
        d.offer_status(frame(1 * S), "invalid_observation", IDS)
        self.settle()
        d.offer_accepted(frame(2 * S), annotated_of(frame(2 * S)), None, IDS)
        self.settle()
        self.assertEqual(self.kinds(), [("accepted", 2 * S)])
        self.assertEqual(d.counts["status_offered"], 0)


class CloseAndFailureTest(DisplayBase):
    """U5·U7·U8(실행 중)·U9·U10."""

    def test_close_run_clears_and_marker_once(self):
        """U5: close_run 뒤 넣기는 버리고(발행 0), 끝 표시 1회(stamp 0), 그 뒤 발행 0."""
        d = self.make()
        d.offer_accepted(frame(1 * S), annotated_of(frame(1 * S)), None, IDS)
        self.settle()
        d.close_run("reset:sim_time_rewind:epoch3")
        d.offer_accepted(frame(2 * S), annotated_of(frame(2 * S)), None, IDS)
        d.offer_status(frame(3 * S), "invalid_observation", IDS)
        self.settle()
        self.assertTrue(d.close(1.0))
        self.assertEqual(self.kinds(), [("accepted", 1 * S), ("marker", 0)])
        self.assertEqual(d.counts["offered_after_close"], 2)
        self.assertEqual(self.pub.frames[-1].image.shape, (48, 64, 3))
        self.assertEqual(self.pub.closed, 1)

    def test_publish_in_flight_at_close_is_the_only_one(self):
        """U5 경합: 닫힘 순간 발행 중이던 1장만 나가고, 칸에 있던 다음 수락은 버린다(닫힌 뒤 시작한 발행 0)."""
        gate = threading.Event()
        clock = FakeClock()
        d = self.make(clock, FakePublisher(clock, gate=gate))
        d.offer_accepted(frame(1 * S), annotated_of(frame(1 * S)), None, IDS)
        self.assertTrue(self.pub.entered.wait(2.0))
        d.offer_accepted(frame(2 * S), annotated_of(frame(2 * S)), None, IDS)   # 칸에 대기
        d.close_run("reset")
        gate.set()
        self.assertTrue(d.close(1.0))
        self.assertEqual(self.kinds(), [("accepted", 1 * S), ("marker", 0)])
        self.assertEqual((d.counts["publish_in_flight_at_close"], d.counts["cleared_on_reset"]), (1, 1))

    def test_no_marker_when_nothing_published(self):
        """끝 표시는 보낸 영상이 있을 때만(크기·프레임을 모르면 보내지 않음)."""
        d = self.make()
        self.assertTrue(d.close(1.0))
        self.assertEqual(self.kinds(), [])

    def test_publish_failures_stop_display(self):
        """U7: 연속 3회 실패 → 표시 멈춤(publish_failed), problem() 있음, 그 뒤 발행 시도 없음."""
        clock = FakeClock()
        d = self.make(clock, FakePublisher(clock, fail=-1))

        for i in range(5):
            d.offer_accepted(frame((i + 1) * S), annotated_of(frame((i + 1) * S)), None, IDS)
            self.settle()

        self.assertEqual((d.stopped, d.counts["publish_errors"]), ("publish_failed", 3))
        self.assertIn("publish_failed", d.problem())
        self.assertTrue(d.close(1.0))
        self.assertEqual(d.counts["published_marker"], 0)

    def test_success_resets_failure_streak(self):
        """U7: 실패 2회 뒤 성공하면 연속 수가 0으로 돌아가 멈추지 않는다."""
        clock = FakeClock()
        d = self.make(clock, FakePublisher(clock, fail=2))

        for i in range(4):
            d.offer_accepted(frame((i + 1) * S), annotated_of(frame((i + 1) * S)), None, IDS)
            self.settle()

        self.assertIsNone(d.stopped)
        self.assertEqual(d.counts["published_accepted"], 2)

    def test_foreign_publisher_stops_display(self):
        """U8(실행 중): 같은 토픽 발행자가 자기 말고 더 보이면 표시 멈춤(foreign_publisher)."""
        clock = FakeClock()
        pub = FakePublisher(clock, publishers=2)
        d = self.make(clock, pub, graph_check_s=1.0)
        d.offer_accepted(frame(1 * S), annotated_of(frame(1 * S)), None, IDS)
        self.settle()
        self.clock.set_s(1.5)
        self.settle()
        d.offer_accepted(frame(2 * S), annotated_of(frame(2 * S)), None, IDS)
        self.settle()
        self.assertEqual((d.stopped, d.counts["foreign_publishers_seen"]), ("foreign_publisher", 1))
        self.assertEqual(self.kinds(), [("accepted", 1 * S)])

    def test_blocked_publisher_does_not_block_offers(self):
        """U9: 발행이 막혀 있어도 넣기는 바로 돌아온다(칸만 덮어씀)."""
        gate = threading.Event()
        clock = FakeClock()
        d = self.make(clock, FakePublisher(clock, gate=gate))
        d.offer_accepted(frame(1 * S), annotated_of(frame(1 * S)), None, IDS)
        self.assertTrue(self.pub.entered.wait(2.0))
        t0 = time.monotonic()

        for i in range(2, 52):
            d.offer_accepted(frame(i * S), annotated_of(frame(i * S)), None, IDS)
            d.offer_status(frame(i * S + 1), "invalid_observation", IDS)

        self.assertLess(time.monotonic() - t0, 0.5)
        self.assertEqual(d.counts["accepted_replaced"], 49)
        gate.set()

    def test_thread_crash_is_not_confirmed(self):
        """U10: 표시 스레드 비정상 종료 → crash 기록, close가 RuntimeError(정리 확인 아님), 다시 불러도 같음."""
        clock = FakeClock()
        pub = FakePublisher(clock)
        disp = LiveCenterDisplay("r", cfg(), pub, clock_ns=clock, is_live=mock.Mock(side_effect=ValueError("boom")))
        disp.start()
        disp.thread.join(2.0)
        self.assertIn("ValueError: boom", disp.crash)

        for _ in range(2):
            with self.assertRaises(RuntimeError):
                disp.close(1.0)

        self.assertIn("crashed", disp.problem())
        self.assertEqual(pub.closed, 1)

    def test_not_live_drops_without_publish(self):
        """연결부가 실행 끝(상태 fail-stop 등)을 알리면 칸을 버리고 발행하지 않는다."""
        live = {"v": True}
        clock = FakeClock()
        pub = FakePublisher(clock)
        disp = LiveCenterDisplay("r", cfg(), pub, clock_ns=clock, is_live=lambda: live["v"])
        live["v"] = False
        disp.start()
        disp.offer_accepted(frame(1 * S), annotated_of(frame(1 * S)), None, IDS)
        time.sleep(0.15)
        self.assertTrue(disp.close(1.0))
        self.assertEqual(pub.frames, [])
        self.assertEqual(disp.counts["dropped_not_live"], 1)


class BannerTextTest(unittest.TestCase):
    """U13(문구): 시야 보정 표기."""

    def test_view_bias_label(self):
        """보정 적용·건너뜀·끔 표기."""
        out = SimpleNamespace(result=SimpleNamespace(spec="BOLT_A", measured_length_m=0.0434, theta_deg=104.4))
        self.assertEqual(lcd.accepted_text(out, None), "OK BOLT_A L=43.4mm th=104.4")
        self.assertTrue(lcd.accepted_text(out, ({"applied": True},)).endswith("vb=applied"))
        self.assertTrue(lcd.accepted_text(out, ({"applied": False, "reason": "no_tf"},)).endswith("vb=skip:no_tf"))
        bad = SimpleNamespace(result=SimpleNamespace(spec="UNRESOLVED", measured_length_m=float("nan"), theta_deg=None))
        self.assertEqual(lcd.accepted_text(bad, None), "OK UNRESOLVED L=? th=?")


# ---------------------------------------------------------------------- CLI(가짜 수신·가짜 모델·가짜 TF·가짜 발행자)

def write_center_config(dst, **over) -> str:
    """예시 설정의 center_display 블록을 바꾼 새 설정 파일."""
    doc = json.loads(CENTER_CONFIG.read_text(encoding="utf-8"))
    doc["center_display"].update(over)
    Path(dst).write_text(json.dumps(doc), encoding="utf-8")

    return str(dst)


class CliBase(LiveBase):
    """기존 실시간 기록 전용 CLI 시험에 표시 발행자 주입을 더한다."""

    def setUp(self):
        """발행자 공장 기록."""
        super().setUp()
        self.factory_calls = []
        self.pub = None

    def factory(self, publisher=None):
        """`live_display_factory(topic, cfg)` 대역."""
        def make(topic, center_cfg):
            self.factory_calls.append((topic, center_cfg))
            self.pub = publisher or FakePublisher()
            return self.pub

        return make

    def run_display(self, scenario, *, config=None, extra=("--live-center-topic",), factory=None, tf=None, max_frames=None,
                    out=None, tf_factory=None):
        """main 실행(표시 주입) → (종료 코드, stderr, 출력 폴더)."""
        out = out or self.out()
        log = str(Path(self.tmp.name) / f"{Path(out).name}.jsonl")
        made = []
        tf = tf if tf is not None else FakeTf()
        scenario.link = lambda: self.controller.current
        config = config or write_center_config(Path(self.tmp.name) / "center_hold_long.json", status_hold_s=100.0)
        argv = ["--ros-link", "--detector", "ai", "--ai-config", self.case.ai, "--log", log,
                "--live-observation-out", out, "--live-observation-config", str(config),
                "--max-frames", str(max_frames or scenario.n), *extra]
        err = io.StringIO()

        try:
            code = self.mod.main(argv, stderr=err, ros_available=lambda: True,
                                 model_factory=lambda path: model_with(scenario.masks_fn),
                                 live_source_factory=scenario.factory_maker(made), live_tf_factory=tf_factory or (lambda c: tf),
                                 live_run_controller=self.controller, live_display_factory=factory or self.factory())
        except SystemExit as exc:
            code = exc.code

        self.tf = tf
        return code, err.getvalue(), out

    def accepts(self, out):
        """수락 기록."""
        return [r for r in self.records(out) if r["kind"] == "accept"]


class CliOnTest(CliBase):
    """U2·U3·U5·U6·U7·U8·U10·U11·U13 CLI 경로."""

    def test_accepted_frames_match_png_annotations(self):
        """U2: 발행 수락 stamp = accept 기록 stamp, 띠 밖 픽셀 = annotated PNG, frame_id 광학 프레임, 끝 표시가 마지막."""
        sc = Scenario(3)
        code, err, out = self.run_display(sc)
        self.assertEqual(code, 0, err)
        self.assertIn("center display ON topic=/b/center_overlay/image", err)
        self.assertEqual(self.factory_calls[0][0], "/b/center_overlay/image")
        acc = {r["input"]["ros_stamp_ns"]: r for r in self.accepts(out)}
        sent = [f for f in self.pub.frames if f.kind == "accepted"]
        self.assertTrue(sent)
        self.assertEqual([f.stamp_ns for f in sent], sorted(f.stamp_ns for f in sent))

        for f in sent:
            png = cv2.imdecode(np.frombuffer((Path(out) / acc[f.stamp_ns]["annotation"]).read_bytes(), np.uint8),
                               cv2.IMREAD_COLOR)
            self.assertEqual(f.frame_id, FRAME_ID)
            self.assertTrue(np.array_equal(f.image[BANNER_H + 1:], png[BANNER_H + 1:]))

        self.assertEqual([f.kind for f in self.pub.frames][-1], "marker")
        s = self.summary(out)
        c = s["center_display"]["counts"]
        # 정상 종료 때 아직 발행 전이던 마지막 수락은 close_run이 비운다(r0 §4): 수락 수 = 발행 + 비움 + 덮어씀
        self.assertEqual(c["accepted_offered"], len(acc))
        self.assertEqual(c["published_accepted"] + c["cleared_on_reset"] + c["accepted_replaced"], len(acc))
        self.assertEqual(balance(c), c["accepted_offered"] + c["status_offered"])
        self.assertEqual((s["center_display_topic"], s["center_display_startup_check"]), ("/b/center_overlay/image", "fake"))
        self.assertEqual(self.c_calls, [])                                   # U11: C 송신 0

    def test_status_screens_only_for_eligible_reasons(self):
        """U3(CLI): hold를 짧게 하면 무효 프레임이 상태 화면으로 나가고, 그 stamp의 건너뜀 사유는 보내는 사유다."""
        from test_run_pipeline_capture_observation import rect_mask

        sc = Scenario(5, masks={2: lambda: [rect_mask()]})
        tf = FakeTf({sc.stamp(3): "missing"})
        cfg_path = write_center_config(Path(self.tmp.name) / "short.json", status_hold_s=0.001, status_max_hz=1000.0)
        code, err, out = self.run_display(sc, config=cfg_path, tf=tf)
        self.assertEqual(code, 0, err)
        reasons = {}

        for r in self.skipped(out):
            reasons.setdefault(r["ros_stamp_ns"], set()).add(r["reason"])

        status = [f.stamp_ns for f in self.pub.frames if f.kind == "status"]
        self.assertTrue(status)   # 적어도 첫 프레임(not_ready) 상태 화면은 나간다

        for st in status:
            self.assertTrue(reasons.get(st, set()) - NO_STATUS_REASONS, (st, reasons.get(st)))

        acc_stamps = {r["input"]["ros_stamp_ns"] for r in self.accepts(out)}
        self.assertTrue(set(f.stamp_ns for f in self.pub.frames if f.kind == "accepted") <= acc_stamps)

    def test_reset_stops_display_and_marker_last(self):
        """U5(CLI): 리셋 → 수락 [프레임 1]만, 끝 표시가 마지막(그 뒤 발행 0), 닫힘 사유 reset, 정리 확인·발행자 닫힘."""
        from test_run_pipeline_live_observation import ResetAndRestartTest

        sc = Scenario(5, during=ResetAndRestartTest.rewind_at(self, 2))
        code, err, out = self.run_display(sc, max_frames=5)
        self.assertEqual(code, 0, err)
        self.assertIn("reason=reset_detected", err)
        kinds = [(f.kind, f.stamp_ns) for f in self.pub.frames]
        s = self.summary(out)
        c = s["center_display"]["counts"]

        # 프레임 1 수락 직후 리셋: 발행 전이면 칸이 비워지고(cleared_on_reset) 끝 표시도 없다(보낸 영상 없음). 어느 쪽이든 리셋 뒤 발행 0
        if c["published_accepted"]:
            self.assertEqual(kinds, [("accepted", sc.stamp(1)), ("marker", 0)])
        else:
            self.assertEqual(kinds, [])
            self.assertGreaterEqual(c["cleared_on_reset"], 1)   # 수락(+ hold 중이던 상태 화면) 비움

        self.assertEqual(balance(c), c["accepted_offered"] + c["status_offered"])   # 넣은 것은 모두 어딘가에 셈
        self.assertTrue(s["center_display"]["close_reason"].startswith("reset:sim_time_rewind"))
        self.assertEqual((s["cleanup_confirmed"], self.pub.closed), (True, 1))

    def test_display_closed_after_dispatcher(self):
        """U6: 정상 종료에도 끝 표시 1회, 표시 정리는 dispatcher 다음(닫을 때 worker 멈춤), 정리 확인."""
        holder = {}

        def note():
            """닫는 순간 worker가 살아 있는지."""
            return holder["link"].dispatcher._thread.is_alive()

        pub = FakePublisher(on_close=note)
        sc = Scenario(3)
        orig = sc.masks_fn

        def masks():
            """실행 중 연결부를 잡아 둔다."""
            holder["link"] = self.controller.current
            return orig()

        sc.masks_fn = masks
        code, err, out = self.run_display(sc, factory=self.factory(pub))
        self.assertEqual(code, 0, err)
        self.assertEqual(pub.close_notes, [False])
        self.assertEqual([f.kind for f in pub.frames].count("marker"), 1)
        self.assertTrue(self.summary(out)["cleanup_confirmed"])

    def test_publish_failure_exit_1_but_b_records_continue(self):
        """U7(CLI): 발행이 늘 실패해도 B 수락 기록은 남고 종료 1, problems에 표시 멈춤."""
        code, err, out = self.run_display(Scenario(5), factory=self.factory(FakePublisher(fail=-1)))
        self.assertEqual(code, 1, err)
        self.assertGreaterEqual(len(self.accepts(out)), 3)
        s = self.summary(out)
        self.assertTrue(any("center display stopped: publish_failed" in p for p in s["problems"]), s["problems"])
        self.assertIsNone(s["state_failed"])
        self.assertTrue(s["cleanup_confirmed"])

    def test_existing_publisher_refused_before_anything(self):
        """U8(시작): 같은 토픽 발행자가 보이면 2, 출력 폴더·TF·세션 없음."""
        from eyetoarm.vision.live_observation_link import LiveObservationLink

        def refuse(topic, c):
            """시작 확인 거부 대역."""
            raise self.mod.ExistingPublisherError(f"topic {topic} already has 1 publisher(s)")

        with mock.patch.object(LiveObservationLink, "__init__", side_effect=AssertionError("no session")):
            code, err, out = self.run_display(Scenario(2), factory=refuse,
                                              tf_factory=lambda c: self.fail("tf must not start"))

        self.assertEqual(code, 2, err)
        self.assertIn("center display refused", err)
        self.assertFalse(Path(out).exists())
        self.assertIsNone(self.controller.current)

    def test_foreign_publisher_during_run_exit_1(self):
        """U8(실행 중): 실행 중 발행자 2개가 보이면 표시 멈춤·종료 1, B 기록은 계속."""
        cfg_path = write_center_config(Path(self.tmp.name) / "graph.json", graph_check_s=0.001, status_hold_s=100.0)
        code, err, out = self.run_display(Scenario(5), config=cfg_path, factory=self.factory(FakePublisher(publishers=2)))
        self.assertEqual(code, 1, err)
        self.assertIn("foreign_publisher", err)
        self.assertTrue(self.accepts(out))

    def test_display_crash_blocks_cleanup_confirmation(self):
        """U10(CLI): 표시 스레드가 죽으면 정리 확인 거짓·종료 1."""
        with mock.patch.object(LiveCenterDisplay, "_loop_body", side_effect=ValueError("boom")):
            code, err, out = self.run_display(Scenario(3))

        self.assertEqual(code, 1, err)
        s = self.summary(out)
        self.assertFalse(s["cleanup_confirmed"])
        self.assertTrue(any("crashed" in e for e in s["ended"]["close_errors"]), s["ended"])

    def test_view_bias_annotation_uses_corrected_output(self):
        """U13: 보정 켬 → 발행 수락 = 보정 끝점으로 만든 accept 기록의 PNG와 같은 픽셀, 띠에 vb=applied."""
        self.case.ai = write_bias_config(self.case.ai, Path(self.tmp.name) / "ai_vb.json")
        texts = []
        real = lcd._draw_banner

        def spy(image, line1, ok):
            """띠 문구 기록."""
            texts.append(line1)
            return real(image, line1, ok)

        with mock.patch.object(lcd, "_draw_banner", spy):
            code, err, out = self.run_display(Scenario(3))

        self.assertEqual(code, 0, err)
        acc = {r["input"]["ros_stamp_ns"]: r for r in self.accepts(out)}
        self.assertTrue(acc and all(r["input"]["view_bias"]["applied"] for r in acc.values()))
        sent = [f for f in self.pub.frames if f.kind == "accepted"]
        self.assertTrue(sent)

        for f in sent:
            png = cv2.imdecode(np.frombuffer((Path(out) / acc[f.stamp_ns]["annotation"]).read_bytes(), np.uint8),
                               cv2.IMREAD_COLOR)
            self.assertTrue(np.array_equal(f.image[BANNER_H + 1:], png[BANNER_H + 1:]))

        self.assertTrue([t for t in texts if " OK " in t] and all(t.endswith("vb=applied") for t in texts if " OK " in t))


class CliOffAndArgsTest(CliBase):
    """U1·U11·U12."""

    def test_off_creates_nothing_and_keeps_record_shape(self):
        """U1: 옵션 없으면 발행자 공장 0, 표시 스레드 0, 요약에 표시 키 없음, 기록 키 집합은 켬과 같다."""
        code, err, out_off = self.run_display(Scenario(3), extra=(), config=str(CONFIG), out=self.out("off"))
        self.assertEqual(code, 0, err)
        self.assertEqual(self.factory_calls, [])
        self.assertNotIn("center display", err)
        s_off = self.summary(out_off)
        self.assertFalse({"center_display", "center_display_topic", "center_display_startup_check"} & set(s_off))
        self.assertFalse(any(t.name == "eyetoarm-live-center-display" for t in threading.enumerate()))

        code, err, out_on = self.run_display(Scenario(3), out=self.out("on"))
        self.assertEqual(code, 0, err)
        s_on = self.summary(out_on)
        self.assertEqual(set(s_on) - set(s_off), {"center_display", "center_display_topic", "center_display_startup_check"})

        def shape(out):
            """기록 종류별 키 집합."""
            return sorted({(r["kind"], tuple(sorted(r))) for r in self.records(out)})

        self.assertEqual(shape(out_off), shape(out_on))
        self.assertEqual(sorted({tuple(sorted(r)) for r in self.skipped(out_off)}),
                         sorted({tuple(sorted(r)) for r in self.skipped(out_on)}))

    def test_off_render_returns_bytes(self):
        """U1: 옵션 없으면 render는 기존처럼 PNG bytes(RenderedCenter 아님)를 돌려 payload에 넣는다."""
        from eyetoarm.vision import live_observation_dispatch as dispatch

        seen = []
        real = dispatch.LiveObservationPayload

        def spy(*a, **k):
            """payload의 png 형 기록."""
            seen.append(type(a[6]))
            return real(*a, **k)

        with mock.patch.object(dispatch, "LiveObservationPayload", spy):
            code, err, _ = self.run_display(Scenario(3), extra=(), config=str(CONFIG))

        self.assertEqual(code, 0, err)
        self.assertTrue(seen and all(t is bytes for t in seen))

    def test_argument_and_config_errors(self):
        """U12: 기록 전용 연결 없이·단계 L·잘못된 토픽·블록 없음/불필요/잘못됨은 2, 발행자·TF 0."""
        base = ["--detector", "ai", "--ai-config", self.case.ai, "--log", str(Path(self.tmp.name) / "x.jsonl")]
        bad_block = Path(self.tmp.name) / "bad_block.json"
        doc = json.loads(CENTER_CONFIG.read_text(encoding="utf-8"))
        del doc["center_display"]["status_hold_s"]
        bad_block.write_text(json.dumps(doc), encoding="utf-8")
        live = ["--live-observation-out", self.out(), "--live-observation-config"]
        cases = [
            ["--ros-link", *base, "--live-center-topic"],
            ["--ros-link", *base, *live, str(CENTER_CONFIG), "--b-input", "list", "--live-center-topic"],
            ["--ros-link", *base, *live, str(CENTER_CONFIG), "--live-center-topic", "no_slash"],
            ["--ros-link", *base, *live, str(CENTER_CONFIG), "--live-center-topic", "/b//x"],
            ["--ros-link", *base, *live, str(CONFIG), "--live-center-topic"],
            ["--ros-link", *base, *live, str(CENTER_CONFIG)],
            ["--ros-link", *base, *live, str(bad_block), "--live-center-topic"],
        ]

        for argv in cases:
            with self.subTest(argv=argv):
                err = io.StringIO()

                try:
                    code = self.mod.main(argv, stderr=err, ros_available=lambda: True,
                                         live_tf_factory=lambda c: self.fail("tf must not start"),
                                         live_run_controller=self.controller,
                                         live_display_factory=lambda t, c: self.fail("publisher must not start"))
                except SystemExit as exc:
                    code = exc.code

                self.assertEqual(code, 2, err.getvalue())

        self.assertIsNone(self.controller.current)

    def test_custom_topic_is_used(self):
        """토픽 이름은 옵션 값으로 바꿀 수 있다."""
        code, err, _ = self.run_display(Scenario(2), extra=("--live-center-topic", "/eyetoarm/display/link/center"))
        self.assertEqual(code, 0, err)
        self.assertEqual(self.factory_calls[0][0], "/eyetoarm/display/link/center")

    def test_d13_and_no_send_path(self):
        """U11: 실행기는 실시간 b_consumer를 계속 거부하고, 표시 모듈은 ROS·C 송신 경로를 쓰지 않는다."""
        from test_live_observation_harness import D13RegressionTest

        D13RegressionTest("test_run_still_rejects_live_observation_consumer").test_run_still_rejects_live_observation_consumer()
        src = (ROOT / "src" / "eyetoarm" / "vision" / "live_center_display.py").read_text(encoding="utf-8")

        for word in ("import rclpy", "sort_part", "SortPartClient", "make_c_handoff", "on_output", "handoff"):
            self.assertNotIn(word, src.split('"""', 2)[2])   # 모듈 docstring 밖 코드에 없음

    def test_stage_with_on_output_still_refused(self):
        """U11: 표시를 켜도 on_output이 있는 B 단계는 거부(RuntimeError → worker 오류·종료 1, 수락 0)."""
        def hook(stage):
            """C 출력 연결 흉내."""
            stage.on_output = lambda out: None

        sc = Scenario(3)
        out = self.out()
        log = str(Path(self.tmp.name) / "hook.jsonl")
        sc.link = lambda: self.controller.current
        made = []
        argv = ["--ros-link", "--detector", "ai", "--ai-config", self.case.ai, "--log", log, "--live-observation-out", out,
                "--live-observation-config", write_center_config(Path(self.tmp.name) / "c.json"), "--max-frames", "3",
                "--live-center-topic"]
        code = self.mod.main(argv, stderr=io.StringIO(), ros_available=lambda: True,
                             model_factory=lambda path: model_with(sc.masks_fn), live_source_factory=sc.factory_maker(made),
                             live_tf_factory=lambda c: FakeTf(), live_run_controller=self.controller, b_stage_hook=hook,
                             live_display_factory=self.factory())
        self.assertEqual(code, 1)
        self.assertEqual(self.accepts(out), [])
        self.assertFalse([f for f in self.pub.frames if f.kind == "accepted"])


if __name__ == "__main__":
    unittest.main()
