"""#413: center 그리기를 별도 프로세스로 — 같은 그림·같은 번호표, 밀리면 최신만, 자식이 죽어도 B는 그대로, 끝나면 남는 것 없음."""
from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from eyetoarm.transforms.b_stage import BStage
from eyetoarm.transforms.center_overlay_pub import overlay_keys
from eyetoarm.vision.center_draw_gate import CenterDrawGate, gated_center_consumer
from eyetoarm.vision.center_process import CenterDrawProcess, process_center_consumer
from test_b_process_frame import bolt, envelope
from test_b_transforms import CAM, ROOT
from test_center_draw_gate import CountingPublisher, goal_view
from test_live_a_wiring import LIST, WiringBase, no_lookup
from test_live_sort_part import FakeRuntime
from test_live_stage_l import three_bolts
from test_run_pipeline_live_observation import Scenario

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src"
FILE_BACKEND = "center_child_fakes:file_backend"


def child_env(out, **extra) -> dict:
    """자식 환경: 시험 폴더·src를 import 경로에 넣고 출력 폴더와 주입 값을 정한다."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(HERE), str(SRC), env.get("PYTHONPATH", "")])
    env["EYETOARM_TEST_CENTER_OUT"] = str(out)
    env.update({k: str(v) for k, v in extra.items()})

    return env


def scene(seq=0):
    """볼트 둘 봉투. `seq`로 사진 번호(image_seq)만 바꾼다."""
    env = envelope([bolt(0, 0.55, -0.12), bolt(2, 0.56, -0.05)])

    if seq:
        fo = env.observations
        fo = dataclasses.replace(fo, frame=dataclasses.replace(fo.frame, image_seq=seq))
        env = dataclasses.replace(env, observations=fo)

    return env


def wait_for(cond, limit_s=10.0) -> bool:
    """조건이 참이 될 때까지 기다린다(못 되면 False)."""
    end = time.monotonic() + limit_s

    while time.monotonic() < end:
        if cond():
            return True

        time.sleep(0.02)

    return bool(cond())


def published(out) -> list[dict]:
    """자식이 남긴 발행 기록."""
    path = Path(out) / "published.jsonl"

    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def pid_gone(pid: int) -> bool:
    """그 pid의 프로세스가 없거나 좀비인가."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError):
        return True

    return state == "Z"


class ProcessBase(unittest.TestCase):
    """자식 프로세스를 띄우고 시험 뒤 반드시 닫는다."""

    def setUp(self):
        """임시 출력 폴더와 실제 B 단계."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "child"
        self.stage = BStage.from_config(ROOT, CAM, no_lookup)

    def drawer(self, factory=FILE_BACKEND, start=True, **kw) -> CenterDrawProcess:
        """자식 대리를 만들어 띄운다. `kw`의 대문자 키는 자식 환경 변수로 간다."""
        env = child_env(self.out, **{k: kw.pop(k) for k in list(kw) if k.isupper()})
        proc = CenterDrawProcess(ROOT, factory=factory, env=env, child_log=str(Path(self.tmp.name) / "child.log"), **kw)
        self.addCleanup(proc.close)

        if start:
            proc.start()

        return proc

    def consumer(self, proc, subscribers=1):
        """별도 프로세스 판 감싸개와 A 쪽 가짜 발행기·구독자 확인."""
        pub = CountingPublisher([], subscribers=subscribers)
        gate = CenterDrawGate(pub)

        return process_center_consumer(self.stage, pub, gate, proc), pub, gate


class SameOutputTest(ProcessBase):
    """자식이 그린 것이 같은 프로세스에서 그린 것과 같다."""

    def test_same_pixels_and_number_tags(self):
        """같은 입력: 자식의 그림은 화소까지 같고, 번호표·촬영 시각·프레임 이름·B 반환값도 같다."""
        proc = self.drawer()
        consume, pub, gate = self.consumer(proc)
        env = scene()
        rec = consume(env)
        self.assertTrue(wait_for(lambda: proc.summary()["drawn"] == 1), proc.summary())

        other = BStage.from_config(ROOT, CAM, no_lookup)
        on = CountingPublisher([], subscribers=1)
        rec_same = gated_center_consumer(other, on, CenterDrawGate(on))(scene())
        frame = env.observations.frame
        expected = other.render_center(frame.image, other.last_frame, guides=True)
        self.assertFalse(np.array_equal(expected, frame.image))   # 실제로 무언가 그려진 그림이어야 비교가 뜻이 있다

        got = np.load(self.out / f"{frame.image_seq}.npy")
        np.testing.assert_array_equal(got, expected)
        line = published(self.out)[0]
        self.assertEqual(line["keys"], overlay_keys(env))
        self.assertEqual((line["stamp_ns"], line["optical_frame"]), (frame.ros_stamp_ns, frame.optical_frame))
        self.assertEqual(goal_view(rec), goal_view(rec_same))
        self.assertEqual(pub.results[0]["results"], on.results[0]["results"])   # 결과 숫자는 A가 그대로 발행
        self.assertEqual((pub.published, gate.drawn), (0, 1))                   # A 프로세스는 영상을 발행하지 않음

    def test_no_viewer_sends_no_request(self):
        """보는 쪽이 없으면 자식에 요청을 보내지 않는다. 결과 숫자는 그대로 나간다."""
        proc = self.drawer()
        consume, pub, gate = self.consumer(proc, subscribers=0)
        rec = consume(scene())
        self.assertIsNotNone(rec["goal"])
        self.assertEqual((proc.summary()["requested"], len(pub.results), gate.skipped), (0, 1, 1))


class BacklogTest(ProcessBase):
    """자식이 밀리면 오래된 사진을 버리고 최신 것만 그린다."""

    def test_slow_child_draws_only_latest(self):
        """느린 자식에 5장을 잇달아 맡기면 처음 것과 마지막 것만 그려지고 사이의 3장은 버려진다(쌓이지 않음)."""
        proc = self.drawer(EYETOARM_TEST_CENTER_SLEEP_S=0.4)
        consume, pub, _ = self.consumer(proc)
        consume(scene(1))
        self.assertTrue(wait_for(lambda: proc.summary()["sent"] == 1))

        for seq in (2, 3, 4, 5):
            consume(scene(seq))

        self.assertTrue(wait_for(lambda: proc.summary()["drawn"] == 2), proc.summary())
        summary = proc.summary()
        self.assertEqual([p["keys"]["image_seq"] for p in published(self.out)], [1, 5])
        self.assertEqual((summary["requested"], summary["sent"], summary["replaced"]), (5, 2, 3))
        self.assertEqual(len(pub.results), 5)   # 결과 숫자는 사진마다


class ChildFailureTest(ProcessBase):
    """자식이 죽거나 멈추거나 한 장을 못 그려도 B 계산·결과 숫자·목표는 그대로다."""

    def test_child_death_keeps_b_running(self):
        """자식이 죽으면 실패 사유가 남고, 그 뒤에도 B 반환·결과 숫자는 그대로이며 요청은 거절로만 센다."""
        proc = self.drawer(EYETOARM_TEST_CENTER_DIE_AT=2)
        consume, pub, _ = self.consumer(proc)
        consume(scene(1))
        self.assertTrue(wait_for(lambda: proc.summary()["drawn"] == 1))
        consume(scene(2))
        self.assertTrue(wait_for(lambda: proc.failed is not None), proc.summary())
        self.assertIn("connection lost", proc.failed)

        rec = consume(scene(3))
        self.assertIsNotNone(rec["goal"])
        self.assertEqual((len(pub.results), consume.errors, proc.summary()["refused"]), (3, 0, 1))
        self.assertEqual(proc.close()["returncode"], 9)

    def test_stalled_child_is_killed(self):
        """자식이 답하지 않으면 멈춘 것으로 보고 끝낸다. B는 계속 돈다."""
        proc = self.drawer(EYETOARM_TEST_CENTER_HANG_AT=1, stall_s=0.5, kill_grace_s=1.0)
        consume, pub, _ = self.consumer(proc)
        consume(scene(1))
        self.assertTrue(wait_for(lambda: proc.failed is not None), proc.summary())
        self.assertIn("stalled", proc.failed)
        self.assertTrue(wait_for(lambda: pid_gone(proc.pid)))
        self.assertIsNotNone(consume(scene(2))["goal"])

    def test_one_failed_frame_does_not_stop_the_child(self):
        """한 장의 발행 실패는 그 장만 잃는다(자식 오류로 셈). 다음 장은 그린다."""
        proc = self.drawer(EYETOARM_TEST_CENTER_FAIL_AT=1)
        consume, _, _ = self.consumer(proc)
        consume(scene(1))
        self.assertTrue(wait_for(lambda: proc.summary()["child_errors"] == 1), proc.summary())
        consume(scene(2))
        self.assertTrue(wait_for(lambda: proc.summary()["drawn"] == 1), proc.summary())
        self.assertIn("publish down", proc.summary()["first_child_error"])
        self.assertIsNone(proc.failed)

    def test_child_that_cannot_prepare(self):
        """자식이 준비에 실패하면(ROS 없음 등) 실패 사유만 남고 요청은 거절된다."""
        proc = self.drawer(factory="center_child_fakes:broken_backend")
        self.assertTrue(wait_for(lambda: proc.failed is not None))
        self.assertIn("no ros here", proc.failed)
        consume, pub, _ = self.consumer(proc)
        self.assertIsNotNone(consume(scene())["goal"])
        self.assertEqual(len(pub.results), 1)

    def test_start_failure_raises_and_refuses(self):
        """자식을 띄우지 못하면 OSError. 그 뒤 요청은 거절되고 닫기는 문제없이 된다."""
        proc = self.drawer(start=False, python="/nonexistent/python")

        with self.assertRaises(OSError):
            proc.start()

        self.assertFalse(proc.submit(np.zeros((2, 2, 3), np.uint8), 0, "cam", [], {"image_seq": 0}))
        self.assertEqual(proc.close(), {"returncode": None, "forced": None})


class SlowStartTest(ProcessBase):
    """r2 D3: 자식 준비가 오래 걸려도 멈춤으로 보지 않는다. 끝내 준비되지 않으면 그때 실패로 돌린다."""

    def test_slow_prepare_is_not_a_stall(self):
        """준비가 멈춤 기준(0.5초)보다 오래(1.5초) 걸려도 준비가 끝난 뒤 맡긴 사진을 그린다. 멈춤 시계는 준비 뒤부터 돈다."""
        proc = self.drawer("center_child_fakes:slow_backend", EYETOARM_TEST_CENTER_PREPARE_S=1.5, stall_s=0.5)
        consume, _, _ = self.consumer(proc)
        consume(scene(1))
        self.assertTrue(wait_for(lambda: proc.summary()["drawn"] == 1, 10.0), proc.summary())
        self.assertIsNone(proc.failed)
        self.assertEqual(proc.summary()["sent"], 1)

    def test_never_ready_fails_after_ready_limit(self):
        """준비가 준비 한도(1초) 안에 끝나지 않으면 그때 「준비 안 됨」으로 실패하고 자식을 끝낸다. 요청은 보내지 않는다."""
        proc = self.drawer("center_child_fakes:slow_backend", EYETOARM_TEST_CENTER_PREPARE_S=60, ready_timeout_s=1.0,
                           kill_grace_s=1.0)
        consume, pub, _ = self.consumer(proc)
        consume(scene(1))
        self.assertTrue(wait_for(lambda: proc.failed is not None, 5.0), proc.summary())
        self.assertIn("not ready", proc.failed)
        self.assertEqual(proc.summary()["sent"], 0)
        self.assertTrue(wait_for(lambda: pid_gone(proc.pid)))
        self.assertEqual(len(pub.results), 1)


class FailureNoticeTest(ProcessBase):
    """r2 D2: 자식을 더 쓸 수 없게 되면 그 사유를 알림 함수로 한 번 알린다."""

    def test_death_is_announced_once(self):
        """자식이 죽으면 알림 함수가 사유와 함께 정확히 한 번 불린다(그 뒤 요청이 더 와도 다시 불리지 않음)."""
        notices = []
        proc = self.drawer(EYETOARM_TEST_CENTER_DIE_AT=1, on_failed=notices.append)
        consume, _, _ = self.consumer(proc)
        consume(scene(1))
        self.assertTrue(wait_for(lambda: notices, 5.0), proc.summary())
        consume(scene(2))
        time.sleep(0.3)
        self.assertEqual(len(notices), 1)
        self.assertIn("connection lost", notices[0])
        self.assertEqual(notices[0], proc.failed)

    def test_prepare_failure_is_announced(self):
        """자식이 준비에 실패해도 알림 함수가 불린다."""
        notices = []
        proc = self.drawer(factory="center_child_fakes:broken_backend", on_failed=notices.append)
        self.assertTrue(wait_for(lambda: notices, 5.0))
        self.assertIn("no ros here", notices[0])


class ShutdownTest(ProcessBase):
    """끝낼 때 자식과 공유 메모리가 남지 않는다."""

    def test_close_leaves_nothing(self):
        """정상 종료: 자식이 스스로 끝나고(강제 아님), 스레드·공유 메모리 fd가 닫힌다. 두 번 닫아도 된다."""
        proc = self.drawer()
        consume, _, _ = self.consumer(proc)
        consume(scene())
        self.assertTrue(wait_for(lambda: proc.summary()["drawn"] == 1))
        pid, mem_fd, threads = proc.pid, proc._mem_fd, list(proc._threads)

        self.assertEqual(proc.close(), {"returncode": 0, "forced": False})
        self.assertTrue(pid_gone(pid))
        self.assertTrue((self.out / "closed").exists())
        self.assertFalse(any(t.is_alive() for t in threads))
        self.assertIsNone(proc._mem_fd)

        with self.assertRaises(OSError):
            os.fstat(mem_fd)

        self.assertEqual(proc.close(), {"returncode": 0, "forced": False})
        self.assertFalse(proc.submit(np.zeros((2, 2, 3), np.uint8), 0, "cam", [], {"image_seq": 0}))

    def test_sigint_to_child_is_ignored(self):
        """터미널 Ctrl+C(SIGINT)가 자식에 가도 자식은 버티고, 부모가 닫을 때 정상 종료한다."""
        proc = self.drawer()
        consume, _, _ = self.consumer(proc)
        self.assertTrue(wait_for(lambda: proc.summary()["ready"] is not None))
        os.kill(proc.pid, signal.SIGINT)
        consume(scene())
        self.assertTrue(wait_for(lambda: proc.summary()["drawn"] == 1), proc.summary())
        self.assertEqual(proc.close(), {"returncode": 0, "forced": False})

    def test_hung_child_is_forced_on_close(self):
        """닫을 때 자식이 멈춰 있으면 신호로 끝낸다(남기지 않음)."""
        proc = self.drawer(EYETOARM_TEST_CENTER_HANG_AT=1, stop_grace_s=0.3, kill_grace_s=1.0, stall_s=60.0)
        consume, _, _ = self.consumer(proc)
        consume(scene(1))
        self.assertTrue(wait_for(lambda: proc.summary()["sent"] == 1))
        self.assertTrue(proc.close()["forced"])
        self.assertTrue(pid_gone(proc.pid))

    def test_parent_killed_child_exits(self):
        """부모가 강제 종료(SIGKILL)되면 자식이 스스로 끝난다."""
        code = ("import sys, time\n"
                "from eyetoarm.vision.center_process import CenterDrawProcess\n"
                f"p = CenterDrawProcess({str(ROOT)!r}, factory={FILE_BACKEND!r})\n"
                "p.start()\n"
                "print(p.pid, flush=True)\n"
                "time.sleep(60)\n")
        parent = subprocess.Popen([sys.executable, "-c", code], env=child_env(self.out), stdout=subprocess.PIPE, text=True)
        self.addCleanup(parent.wait)
        self.addCleanup(parent.kill)
        child_pid = int(parent.stdout.readline())
        self.assertFalse(pid_gone(child_pid))
        parent.kill()
        parent.wait()
        self.assertTrue(wait_for(lambda: pid_gone(child_pid), 5.0))
        parent.stdout.close()


class FakeDrawer:
    """진입점 시험용 자식 대리(프로세스를 띄우지 않고 요청만 기록)."""

    def __init__(self, root, *, on_failed=None, fail_at=None):
        """요청 기록. `fail_at`번째 요청에서 자식이 죽은 것처럼 `on_failed`를 부른다(그 뒤 요청은 거절)."""
        self.root, self.on_failed, self.fail_at = root, on_failed, fail_at
        self.started = self.closed = 0
        self.cameras, self.jobs = [], []
        self.pid = 4242
        self.failed = None

    def start(self):
        """시작 기록."""
        self.started += 1

    def configure(self, camera):
        """카메라 기록."""
        self.cameras.append(camera)

    def submit(self, image, stamp_ns, optical_frame, items, keys, *, guides=True):
        """요청 기록(실패 뒤에는 거절)."""
        if self.failed is not None:
            return False

        self.jobs.append(keys)

        if self.fail_at is not None and len(self.jobs) == self.fail_at:
            self.failed = "child connection lost (returncode=-9)"

            if self.on_failed is not None:
                self.on_failed(self.failed)

        return True

    def summary(self):
        """요약."""
        return {"requested": len(self.jobs), "failed": self.failed}

    def close(self):
        """닫기 기록."""
        self.closed += 1
        return {"returncode": 0, "forced": False}


class CliProcessTest(WiringBase):
    """진입점: `--b-center-process`를 주면 그리기 요청이 자식으로 가고, 목표·수락은 지금 방식과 같다."""

    def run_with(self, *extra, fail_at=None):
        """구독자 1인 발행기로 한 번 실행 → (종료 코드, stderr, 출력 폴더, 발행기, 송신 런타임, 자식 대리들)."""
        made, drawers = [], []

        def publisher():
            """구독자 1인 발행기."""
            made.append(CountingPublisher(self.events, subscribers=1))
            return made[-1]

        def drawer(root, **kw):
            """가짜 자식 대리."""
            drawers.append(FakeDrawer(root, fail_at=fail_at, **kw))
            return drawers[-1]

        self.publisher_factory = publisher
        self.extra_kw = {"center_process_factory": drawer}
        rt = FakeRuntime()
        sc = Scenario(4, masks={i: three_bolts for i in range(4)})
        code, err, out = self.run_sort(sc, factory=self.factory(rt), out=self.out(f"live{len(self.handoffs)}"),
                                       extra=(*LIST, "--b-center-overlay", "--live-sort-part", *extra))

        self.assertTrue(made, err)

        return code, err, out, made[0], rt, drawers

    def test_process_option_moves_drawing_out(self):
        """옵션을 주면 A는 영상을 발행하지 않고 요청만 넘긴다. 번호표·결과 숫자·보낸 목표는 같은 프로세스 방식과 같다. 끝나면 닫는다."""
        code, err, out, pub, rt, drawers = self.run_with("--b-center-process")
        code2, err2, out2, pub2, rt2, drawers2 = self.run_with("--no-b-center-process")
        self.assertEqual((code, code2), (0, 0), err + err2)
        self.assertEqual((len(drawers), len(drawers2)), (1, 0))
        drawer = drawers[0]
        self.assertEqual((drawer.started, drawer.closed, pub.published), (1, 1, 0))
        self.assertGreater(pub2.published, 0)
        self.assertEqual(len(drawer.jobs), pub2.published)
        tags = lambda keys: {k: v for k, v in keys.items() if k != "run_id"}  # noqa: E731 - run_id는 실행마다 새로 만든다
        self.assertEqual([tags(k) for k in drawer.jobs], [tags(k) for _, k in pub2.frames])   # 번호표가 지금 방식의 발행과 같다
        self.assertEqual({k["run_id"] for k in drawer.jobs}, {r["run_id"] for r in pub.results})
        self.assertEqual(len(pub.results), len(pub2.results))
        view = lambda r: [(g["kind"], g["spec"], g["observed_position_mm"]) for g in r.sent]  # noqa: E731
        self.assertEqual(view(rt), view(rt2))
        self.assertTrue(drawer.cameras)
        self.assertIn("draw_process=on pid=4242 (#413)", err)
        self.assertIn("draw_process=off", err2)
        summary = self.summary(out)
        center = (summary.get("tracking") or summary)["center_overlay"]
        self.assertEqual(center["draw_process"]["requested"], len(drawer.jobs))
        self.assertEqual(center["draw_process"]["exit"], {"returncode": 0, "forced": False})

    def test_default_is_separate_process(self):
        """r3: 옵션 없이 `--b-center-overlay`만 주면 별도 프로세스가 기본으로 켜진다(시작 줄 `draw_process=on`)."""
        code, err, _, pub, _, drawers = self.run_with()
        self.assertEqual(code, 0, err)
        self.assertEqual((len(drawers), drawers[0].started, drawers[0].closed, pub.published), (1, 1, 1, 0))
        self.assertTrue(drawers[0].jobs)
        self.assertIn("draw_process=on pid=4242 (#413)", err)

    def test_off_option_draws_in_process(self):
        """r3: `--no-b-center-process`를 주면 자식을 띄우지 않고 예전처럼 같은 프로세스에서 그려 발행한다."""
        code, err, _, pub, _, drawers = self.run_with("--no-b-center-process")
        self.assertEqual(code, 0, err)
        self.assertEqual(drawers, [])
        self.assertGreater(pub.published, 0)
        self.assertIn("draw_process=off", err)

    def test_on_and_off_together_refused(self):
        """r3: 켜기·끄기를 함께 주면 시작을 거부한다(종료 2)."""
        rt = FakeRuntime()
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            code, err, _ = self.run_sort(Scenario(1, masks={0: three_bolts}), factory=self.factory(rt), out=self.out("both"),
                                         extra=(*LIST, "--b-center-overlay", "--live-sort-part", "--b-center-process",
                                                "--no-b-center-process"))

        self.assertEqual(code, 2, err)
        self.assertIn("not allowed with argument", stderr.getvalue())

    def test_without_overlay_nothing_happens(self):
        """r3: `--b-center-overlay` 없이는 기본값 때문에 거부되지 않고, 자식도 띄우지 않는다(끄는 옵션만 준 경우도 같음)."""
        for extra in ((), ("--no-b-center-process",)):
            drawers = []
            self.extra_kw = {"center_process_factory": lambda root, **kw: drawers.append(FakeDrawer(root, **kw)) or drawers[-1]}
            rt = FakeRuntime()
            code, err, _ = self.run_sort(Scenario(2, masks={i: three_bolts for i in range(2)}), factory=self.factory(rt),
                                         out=self.out(f"plain{len(extra)}"), extra=(*LIST, "--live-sort-part", *extra))
            self.assertEqual(code, 0, err)
            self.assertEqual(drawers, [])
            self.assertNotIn("draw_process", err)
            self.assertTrue(rt.sent)

    def test_process_option_needs_overlay(self):
        """r2 D1: 다른 인자는 모두 맞고 `--b-center-overlay`만 빠지면 바로 그 규칙(사유 문구)으로 시작을 거부한다(종료 2)."""
        rt = FakeRuntime()
        sc = Scenario(1, masks={0: three_bolts})
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            code, err, _ = self.run_sort(sc, factory=self.factory(rt), out=self.out("refused"),
                                         extra=(*LIST, "--live-sort-part", "--b-center-process"))

        self.assertEqual(code, 2, err)
        self.assertIn("--b-center-process requires --b-center-overlay", stderr.getvalue())
        self.assertEqual(rt.sent, [])

    def test_child_failure_is_shown_during_and_at_end(self):
        """r2 D2: 실행 중 자식이 죽으면 그때 표준 오류에 안내가 나오고, 끝 줄에도 실패 사유가 나온다. 목표는 그대로 나간다."""
        code, err, out, _, rt, drawers = self.run_with("--b-center-process", fail_at=1)
        code2, _, _, _, rt2, _ = self.run_with("--b-center-process")
        self.assertEqual((code, code2), (0, 0), err)
        self.assertIn("run_pipeline: center draw process stopped: child connection lost (returncode=-9); "
                      "center image off (relay sends the raw image), run continues", err)
        self.assertEqual(err.count("center draw process stopped"), 1)
        self.assertIn("center_process=failed(child connection lost (returncode=-9))", err.splitlines()[-1] + err)
        exit_line = next(line for line in err.splitlines() if line.startswith("run_pipeline: exit="))
        self.assertIn("center_process=failed(child connection lost (returncode=-9))", exit_line)
        view = lambda r: [(g["kind"], g["spec"], g["observed_position_mm"]) for g in r.sent]  # noqa: E731
        self.assertEqual(view(rt), view(rt2))
        center = (self.summary(out).get("tracking") or self.summary(out))["center_overlay"]
        self.assertEqual(center["draw_process"]["failed"], "child connection lost (returncode=-9)")


if __name__ == "__main__":
    unittest.main()
