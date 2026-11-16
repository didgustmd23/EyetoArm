"""실시간 검출 확인 창 진입점(#138 `scripts/view_live_detection.py`) 시험 — 창 루프·종료·실패 정리.

근거: reviewer 지시 138 §3(메인 스레드 창 이벤트 지속, runner 별도 스레드, 그리기 직전 버전 확인)·§4(ESC/q·창 닫기·Ctrl+C,
유한 정리, 창 실패·디스플레이 없음 진단, 종료 요청 뒤 새 검출 미표시)·§5. 실제 창·ROS·GPU·모델 없이 가짜 창·가짜 runner·가짜 신호를 넣는다.
"""
import importlib.util
import io
import os
import tempfile
import threading
import time
import unittest

import numpy as np

from eyetoarm.vision.live_view import DisplayBridge
from eyetoarm.vision.pipeline_runner import RunResult
from test_live_view import big_instance, completed, job

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_spec = importlib.util.spec_from_file_location("view_live_detection", os.path.join(ROOT, "scripts",
                                                                                     "view_live_detection.py"))
viewer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(viewer)

AI_CONFIG = os.path.join(ROOT, "config", "vision", "ai_s1_01_live.json")


class FakeWindow:
    """가짜 창. `keys`는 poll 회차별로 돌려줄 키(없으면 None), `close_at`·`fail_show`·`interrupt_at`으로 사건을 만든다."""

    def __init__(self, keys=None, close_at=None, fail_show=False, fail_open=False, interrupt_at=None, on_poll=None):
        """사건 설정."""
        self.keys, self.close_at, self.fail_show, self.fail_open = dict(keys or {}), close_at, fail_show, fail_open
        self.interrupt_at, self.on_poll = interrupt_at, on_poll
        self.polls, self.shown, self.opened, self.closed_called = 0, [], False, False

    def open(self):
        """창 생성(실패 가능)."""
        if self.fail_open:
            raise RuntimeError("cannot connect to display")

        self.opened = True

    def show(self, image):
        """그린 영상을 적는다(실패 가능)."""
        if self.fail_show:
            raise RuntimeError("imshow failed")

        self.shown.append(image)

    def poll(self, interval_ms):
        """한 회차 이벤트 처리(실제 대기 대신 짧게 쉼)."""
        self.polls += 1

        if self.on_poll is not None:
            self.on_poll(self.polls)

        if self.interrupt_at is not None and self.polls == self.interrupt_at:
            raise KeyboardInterrupt

        time.sleep(0.002)

        return self.keys.get(self.polls)

    def closed(self):
        """close_at 회차부터 닫힘."""
        return self.close_at is not None and self.polls >= self.close_at

    def close(self):
        """창 없앰."""
        self.closed_called = True


class FakeRunner:
    """가짜 runner. 시작하면 결과 하나를 표시 칸에 넣고, 정지 요청까지 기다린다(또는 스스로 끝남·정지 무시)."""

    def __init__(self, *, publish=True, end_after_s=None, exit_code=0, reason="stop_requested", ignore_stop=False):
        """동작 설정."""
        self.publish, self.end_after_s, self.exit_code, self.reason = publish, end_after_s, exit_code, reason
        self.ignore_stop = ignore_stop
        self.release = threading.Event()   # 정지 무시 runner를 시험 끝에 풀어 줌
        self.calls, self.bridge, self.saw_stop, self.after_stop_shown = 0, None, False, None

    def __call__(self, source_factory, settings, log, *, display_sink, should_stop, **kw):
        """run 대체."""
        self.calls += 1
        self.bridge = display_sink
        self.kw = kw

        if self.publish:
            display_sink.on_outcome(completed(job(1), [big_instance()]), current=True)

        end = None if self.end_after_s is None else time.monotonic() + self.end_after_s

        while True:
            if self.ignore_stop:
                self.release.wait(10)
                break

            if should_stop():
                self.saw_stop = True
                before = display_sink.shown
                display_sink.on_outcome(completed(job(2, job_seq=1), [big_instance()]), current=True)
                self.after_stop_shown = display_sink.shown - before
                break

            if end is not None and time.monotonic() >= end:
                break

            time.sleep(0.002)

        return RunResult(exit_code=self.exit_code, stop_reason="stop_requested" if self.saw_stop else self.reason,
                         run_id="run-fake")


class FakeSignals:
    """가짜 신호 처리기(`_StopSignals` 대체): stop 표지와 with 진입·종료 기록."""

    def __init__(self):
        """정지 표지."""
        self.stop = threading.Event()
        self.entered = self.exited = False

    def __enter__(self):
        """설치."""
        self.entered = True
        return self

    def __exit__(self, *exc):
        """복원."""
        self.exited = True


class ViewerMainTest(unittest.TestCase):
    """main: 종료 경로별 코드·정리."""

    def setUp(self):
        """임시 기록 폴더."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def main(self, window, runner, extra=(), **kw):
        """가짜를 넣어 main 실행. (코드, stderr)."""
        err = io.StringIO()
        argv = ["--ai-config", AI_CONFIG, "--log", os.path.join(self.tmp.name, "run.jsonl"), "--key-interval-ms", "1",
                *extra]
        kw.setdefault("display_check", lambda: None)
        kw.setdefault("ros_available", lambda: True)
        kw.setdefault("signals", FakeSignals())
        code = viewer.main(argv, stderr=err, window=window, run_fn=runner, **kw)

        return code, err.getvalue()

    def test_quit_keys_and_window_close(self):
        """ESC·q·창 닫기 → 정지 요청, runner가 정지를 보고 반환, 종료 0, 창 닫힘, 종료 요청 뒤 새 검출 미표시."""
        for name, window in (("esc", FakeWindow(keys={5: 27})), ("q", FakeWindow(keys={5: ord("q")})),
                             ("close", FakeWindow(close_at=5))):
            with self.subTest(name=name):
                runner = FakeRunner()
                code, err = self.main(window, runner)

                self.assertEqual(code, 0, err)
                self.assertTrue(runner.saw_stop)
                self.assertEqual(runner.after_stop_shown, 0)
                self.assertTrue(window.closed_called)
                self.assertGreaterEqual(len(window.shown), 1)
                self.assertIn("end=user", err)

    def test_events_processed_without_frames(self):
        """프레임(결과)이 오지 않아도 창 이벤트를 계속 처리한다(poll 회차가 늘고 q로 끝남)."""
        window = FakeWindow(keys={50: ord("q")})
        runner = FakeRunner(publish=False)
        code, err = self.main(window, runner)

        self.assertEqual(code, 0, err)
        self.assertGreaterEqual(window.polls, 50)
        self.assertEqual(runner.bridge.shown, 0)

    def test_first_signal_is_stop_request(self):
        """첫 SIGINT/SIGTERM(신호 표지) → runner 정지 요청이 바로 보이고 창 루프 종료, 종료 0, 신호 처리기 복원."""
        signals = FakeSignals()
        window = FakeWindow(on_poll=lambda n: signals.stop.set() if n == 5 else None)
        runner = FakeRunner()
        code, err = self.main(window, runner, signals=signals)

        self.assertEqual(code, 0, err)
        self.assertTrue(runner.saw_stop)
        self.assertTrue(signals.entered and signals.exited)
        self.assertIn("end=signal", err)

    def test_second_signal_interrupt(self):
        """두 번째 신호(KeyboardInterrupt) → 정지 요청·유한 join 뒤 종료 1."""
        window = FakeWindow(interrupt_at=5)
        runner = FakeRunner()
        code, err = self.main(window, runner)

        self.assertEqual(code, 1, err)
        self.assertTrue(runner.saw_stop)
        self.assertIn("end=interrupted", err)

    def test_draw_failure_stops_runner(self):
        """그리기 실패 → 진단, 정지 요청으로 runner 정리, 창 닫기 시도, 종료 1."""
        window = FakeWindow(fail_show=True)
        runner = FakeRunner()
        started = time.monotonic()
        code, err = self.main(window, runner)

        self.assertEqual(code, 1)
        self.assertTrue(runner.saw_stop)
        self.assertIn("display failed", err)
        self.assertTrue(window.closed_called)
        self.assertLess(time.monotonic() - started, 5)

    def test_no_display_or_open_failure_before_runner(self):
        """디스플레이 없음·창 생성 실패는 runner(ROS 연결)를 시작하기 전에 진단하고 2."""
        runner = FakeRunner()
        code, err = self.main(FakeWindow(), runner, display_check=lambda: "no display: DISPLAY is not set")
        self.assertEqual((code, runner.calls), (2, 0))
        self.assertIn("no display", err)

        window = FakeWindow(fail_open=True)
        code, err = self.main(window, runner)
        self.assertEqual((code, runner.calls), (2, 0))
        self.assertIn("cannot create window", err)

    def test_runner_not_finishing_is_bounded(self):
        """정지를 무시하는 runner: join 상한 뒤 미종료를 보고하고 1(무한 대기 없음)."""
        runner = FakeRunner(ignore_stop=True)
        started = time.monotonic()
        code, err = self.main(FakeWindow(keys={3: 27}), runner, extra=("--join-timeout", "0.2"))
        runner.release.set()

        self.assertEqual(code, 1)
        self.assertLess(time.monotonic() - started, 3)
        self.assertIn("did not finish", err)

    def test_runner_ends_by_itself(self):
        """runner가 스스로 실패로 끝나면 마지막 상태를 그리고, --exit-on-end면 바로 끝난다(runner 코드 1)."""
        window = FakeWindow()
        runner = FakeRunner(end_after_s=0.05, exit_code=1, reason="receiver_failed")
        code, err = self.main(window, runner, extra=("--exit-on-end",))

        self.assertEqual(code, 1, err)
        self.assertIn("end=ended", err)
        self.assertIn("reason=receiver_failed", err)

    def test_runner_ends_hold_until_user(self):
        """기본(hold)은 runner가 끝나도 사용자가 닫을 때까지 창을 유지한다."""
        window = FakeWindow(keys={200: ord("q")})
        runner = FakeRunner(end_after_s=0.01, exit_code=0, reason="max_frames")
        code, err = self.main(window, runner)

        self.assertEqual(code, 0, err)
        self.assertGreaterEqual(window.polls, 200)
        self.assertIn("end=user", err)

    def test_runner_wired_like_cli(self):
        """runner에 AI 검출기·준비 함수·정지 판정·표시 전달 지점을 넘긴다(b_consumer 없음)."""
        runner = FakeRunner()
        code, err = self.main(FakeWindow(keys={3: 27}), runner)

        self.assertEqual(code, 0, err)
        self.assertIn("detector", runner.kw)
        self.assertIn("prepare", runner.kw)
        self.assertNotIn("b_consumer", runner.kw)
        self.assertIsInstance(runner.bridge, DisplayBridge)

    def test_rclpy_missing_and_config_error(self):
        """rclpy가 없거나 AI 설정을 읽지 못하면 창·runner 없이 2."""
        runner, window = FakeRunner(), FakeWindow()
        code, err = self.main(window, runner, ros_available=lambda: False)
        self.assertEqual((code, runner.calls, window.opened), (2, 0, False))

        code, err = self.main(window, runner, extra=("--ai-config", os.path.join(self.tmp.name, "missing.json")))
        self.assertEqual((code, runner.calls, window.opened), (2, 0, False))


class GuiLoopTest(unittest.TestCase):
    """창 루프 단독: 그리기 직전 버전 확인."""

    def test_stale_render_discarded_before_show(self):
        """그리는 동안 경계가 오면 그 영상은 보이지 않고, 경계 뒤 상태를 다시 그려 보인다."""
        bridge = DisplayBridge()
        bridge.on_outcome(completed(job(1), [big_instance()]), current=True)
        stale_version = bridge.version
        rendered = []

        def render(snap, max_width):
            """첫 그리기 중에 경계가 도착한다. 영상에 버전을 표시해 둔다."""
            rendered.append(snap.version)

            if len(rendered) == 1:
                bridge.on_input("frame_timeout", "gap", 0, False)

            return np.full((2, 2, 3), snap.version, np.uint8)

        window = FakeWindow(keys={3: ord("q")})
        how = viewer.gui_loop(window, bridge, lambda: False, lambda: True, interval_ms=1, max_width=100,
                              hold_on_end=True, render=render)
        shown_versions = [int(img[0, 0, 0]) for img in window.shown]

        self.assertEqual(how, "user")
        self.assertNotIn(stale_version, shown_versions)
        self.assertEqual(shown_versions, [bridge.version])


if __name__ == "__main__":
    unittest.main()
