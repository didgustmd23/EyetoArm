"""#432: center 그리기 자식이 죽거나 멈추거나 준비에 실패하면 잠시 뒤 다시 띄운다(횟수 한도·늘어나는 간격).

실제 자식(시험 발행기 `center_child_fakes`)으로 확인한다. 간격은 시험이 짧게 준다.
"""
from __future__ import annotations

import os
import time
import unittest

import numpy as np

from test_center_process import CountingPublisher, FakeDrawer, ProcessBase, pid_gone, published, scene, wait_for
from test_live_a_wiring import LIST, WiringBase
from test_live_sort_part import FakeRuntime
from test_live_stage_l import three_bolts
from test_run_pipeline_live_observation import Scenario


def tiny_image():
    """작은 사진(번호표만 보는 시험용)."""
    return np.zeros((4, 4, 3), np.uint8)


class RestartTest(ProcessBase):
    """다시 띄우기: 죽음·멈춤·준비 실패 뒤 새 자식, 한도, 정상 뒤 초기화."""

    def notices(self):
        """안내 줄 모음과 받는 함수."""
        lines = []

        return lines, lines.append

    def test_death_then_new_child_draws(self):
        """자식이 죽으면 잠시 뒤 새 자식을 띄우고, 카메라 값을 다시 보내 그 뒤 사진을 그린다. 안내는 다시 띄움·성공 두 줄."""
        lines, note = self.notices()
        failed = []
        proc = self.drawer(EYETOARM_TEST_CENTER_DIE_AT=1, restart_delays_s=(0.1, 0.1, 0.1), on_notice=note,
                           on_failed=failed.append)
        consume, pub, _ = self.consumer(proc)
        first_pid = proc.pid
        consume(scene(1))
        self.assertTrue(wait_for(lambda: proc.summary()["restarts"] == 1 and proc.summary()["ready"] is not None, 10.0),
                        proc.summary())
        self.assertNotEqual(proc.pid, first_pid)
        self.assertTrue(pid_gone(first_pid))

        consume(scene(2))
        self.assertTrue(wait_for(lambda: proc.summary()["drawn"] == 1, 10.0), proc.summary())
        self.assertEqual([p["keys"]["image_seq"] for p in published(self.out)], [2])
        self.assertEqual(proc.summary()["child_errors"], 0)   # 카메라 값을 다시 받지 못했으면 'draw before config'
        self.assertIsNone(proc.failed)
        self.assertEqual(failed, [])
        self.assertEqual(len(lines), 2, lines)
        self.assertIn("restarting in 0.1 s (attempt 1/3)", lines[0])
        self.assertIn("connection lost", lines[0])
        self.assertIn(f"restarted pid={proc.pid} (attempt 1/3)", lines[1])
        self.assertEqual(len(pub.results), 2)
        reasons = proc.summary()["restart_reasons"]
        self.assertEqual(len(reasons), 1)
        self.assertIn("connection lost", reasons[0]["reason"])

    def test_gives_up_after_limit(self):
        """준비에 계속 실패하면 한도(3번)까지 다시 시도하고 그만둔다. 마지막에 실패 알림 한 번, 그 뒤 요청은 거절."""
        lines, note = self.notices()
        failed = []
        proc = self.drawer(factory="center_child_fakes:broken_backend", restart_delays_s=(0.05, 0.05, 0.05),
                           on_notice=note, on_failed=failed.append)
        self.assertTrue(wait_for(lambda: proc.failed is not None, 20.0), proc.summary())
        self.assertEqual(proc.summary()["restarts"], 3)
        self.assertEqual(len(failed), 1)
        self.assertIn("no ros here", failed[0])
        self.assertEqual(len([line for line in lines if "restarting" in line]), 3, lines)
        self.assertFalse(any("restarted pid" in line for line in lines))   # 한 번도 준비되지 않음
        self.assertFalse(proc.submit(tiny_image(), 0, "cam", [], {"image_seq": 0}))
        time.sleep(0.3)
        self.assertEqual(proc.summary()["restarts"], 3)   # 그만둔 뒤에는 더 띄우지 않음

    def test_stall_then_restart(self):
        """자식이 답하지 않으면(멈춤) 끝내고 다시 띄운다. 새 자식은 다음 사진을 그린다."""
        proc = self.drawer(EYETOARM_TEST_CENTER_HANG_AT=1, stall_s=0.5, kill_grace_s=1.0, restart_delays_s=(0.1,))
        consume, _, _ = self.consumer(proc)
        first_pid = proc.pid
        consume(scene(1))
        self.assertTrue(wait_for(lambda: proc.summary()["restarts"] == 1 and proc.summary()["ready"] is not None, 10.0),
                        proc.summary())
        self.assertTrue(wait_for(lambda: pid_gone(first_pid), 5.0))
        self.assertIn("stalled", proc.summary()["restart_reasons"][0]["reason"])
        consume(scene(2))
        self.assertTrue(wait_for(lambda: proc.summary()["drawn"] == 1, 10.0), proc.summary())

    def test_count_resets_after_healthy_spell(self):
        """한참(시험 0.3 s) 정상으로 그린 뒤에 죽으면 횟수를 다시 센다. 띄우자마자 또 죽으면 한도로 그만둔다."""
        failed = []
        proc = self.drawer(EYETOARM_TEST_CENTER_DIE_AT="1,3,4", restart_delays_s=(0.05,), healthy_s=0.3,
                           on_failed=failed.append)
        consume, _, _ = self.consumer(proc)
        consume(scene(1))   # 죽음 → 1/1번째 다시 띄우기
        self.assertTrue(wait_for(lambda: proc.summary()["ready"] is not None and proc.summary()["restarts"] == 1, 10.0))
        consume(scene(2))   # 정상으로 그림
        self.assertTrue(wait_for(lambda: proc.summary()["drawn"] == 1, 10.0))
        time.sleep(0.4)     # 한참 정상
        consume(scene(3))   # 죽음 → 횟수를 다시 세므로 또 다시 띄움
        self.assertTrue(wait_for(lambda: proc.summary()["restarts"] == 2 and proc.summary()["ready"] is not None, 10.0),
                        proc.summary())
        self.assertIsNone(proc.failed)
        consume(scene(4))   # 띄우자마자 또 죽음 → 한도(1번)를 다 씀
        self.assertTrue(wait_for(lambda: proc.failed is not None, 10.0), proc.summary())
        self.assertEqual((proc.summary()["restarts"], len(failed)), (2, 1))

    def test_close_while_waiting_to_restart(self):
        """다시 띄우기를 기다리는 중에 닫으면 바로 끝나고, 새 자식을 띄우지 않으며 남는 것이 없다."""
        proc = self.drawer(EYETOARM_TEST_CENTER_DIE_AT=1, restart_delays_s=(5.0,))
        consume, _, _ = self.consumer(proc)
        first_pid = proc.pid
        consume(scene(1))
        self.assertTrue(wait_for(lambda: proc.summary()["waiting_restart"], 10.0), proc.summary())
        threads = list(proc._threads)
        t0 = time.monotonic()
        exit_info = proc.close()
        self.assertLess(time.monotonic() - t0, 3.0)
        self.assertEqual(proc.pid, first_pid)          # 새 자식을 띄우지 않았다
        self.assertTrue(pid_gone(first_pid))
        self.assertEqual(exit_info["returncode"], 9)
        self.assertFalse(any(t.is_alive() for t in threads))
        self.assertIsNone(proc._mem_fd)
        self.assertEqual(proc.summary()["restarts"], 0)

    def test_submit_not_blocked_while_restarting(self):
        """다시 띄우는 동안 submit은 기다리지 않고 바로 돌아온다(거절로 셈). B 결과 숫자는 그대로 나간다."""
        proc = self.drawer(EYETOARM_TEST_CENTER_DIE_AT=1, restart_delays_s=(1.0,))
        consume, pub, _ = self.consumer(proc)
        consume(scene(1))
        self.assertTrue(wait_for(lambda: proc.summary()["waiting_restart"], 10.0))
        t0 = time.monotonic()
        rec = consume(scene(2))
        self.assertLess(time.monotonic() - t0, 0.5)
        self.assertIsNotNone(rec["goal"])
        self.assertEqual(len(pub.results), 2)
        self.assertGreaterEqual(proc.summary()["refused"], 1)

    def test_restart_spawns_fresh_memory_and_connection(self):
        """다시 띄우면 공유 메모리·연결을 새로 만들고 옛 것은 닫는다."""
        proc = self.drawer(EYETOARM_TEST_CENTER_DIE_AT=1, restart_delays_s=(0.05,))
        consume, _, _ = self.consumer(proc)
        old_fd = proc._mem_fd
        consume(scene(1))
        self.assertTrue(wait_for(lambda: proc.summary()["restarts"] == 1 and proc.summary()["ready"] is not None, 10.0))
        self.assertNotEqual(proc._mem_fd, None)

        with self.assertRaises(OSError):   # 옛 memfd 번호는 닫혔다(새 번호가 같은 값을 다시 쓸 수는 있음)
            if proc._mem_fd == old_fd:
                raise OSError("reused number")
            os.fstat(old_fd)


class StartFailureTest(ProcessBase):
    """처음 띄우기 실패는 다시 시도하지 않는다(실행 시작 때 바로 드러나게, #413과 같음)."""

    def test_first_start_failure_is_not_retried(self):
        """처음 띄우지 못하면 OSError, 다시 띄우기 없음."""
        proc = self.drawer(start=False, python="/nonexistent/python", restart_delays_s=(0.05,))

        with self.assertRaises(OSError):
            proc.start()

        time.sleep(0.2)
        self.assertEqual(proc.summary()["restarts"], 0)


class RestartingDrawer(FakeDrawer):
    """진입점 시험용: 첫 요청 때 자식을 잃고 다시 띄운 것처럼 안내 두 줄을 내고, 요약에 다시 띄운 횟수 1."""

    def submit(self, image, stamp_ns, optical_frame, items, keys, *, guides=True):
        """첫 요청에서 다시 띄우기 안내."""
        if not self.jobs and self.on_notice is not None:
            self.on_notice("center draw process lost: child connection lost (returncode=-9); restarting in 2 s (attempt 1/3)")
            self.on_notice("center draw process restarted pid=4343 (attempt 1/3)")

        return super().submit(image, stamp_ns, optical_frame, items, keys, guides=guides)

    def summary(self):
        """다시 띄운 횟수 1."""
        return {**super().summary(), "restarts": 1}


class CliRestartTest(WiringBase):
    """진입점: 다시 띄우기 안내가 표준 오류로 나가고 끝 줄에 다시 띄운 횟수가 붙는다. 목표·종료 코드는 그대로."""

    def test_notices_and_end_line(self):
        """안내 두 줄이 그대로 나가고, 끝 줄에 `center_process=ok center_restarts=1`. 종료 0, 목표는 나감."""
        made = []

        def drawer(root, **kw):
            """다시 띄우기를 흉내 내는 자식 대리."""
            made.append(RestartingDrawer(root, **kw))
            return made[-1]

        self.publisher_factory = lambda: CountingPublisher(self.events, subscribers=1)
        self.extra_kw = {"center_process_factory": drawer}
        rt = FakeRuntime()
        sc = Scenario(4, masks={i: three_bolts for i in range(4)})
        code, err, _ = self.run_sort(sc, factory=self.factory(rt), out=self.out("restart"),
                                     extra=(*LIST, "--b-center-overlay", "--live-sort-part"))
        self.assertEqual(code, 0, err)
        self.assertIn("run_pipeline: center draw process lost: child connection lost (returncode=-9); restarting in 2 s "
                      "(attempt 1/3)", err)
        self.assertIn("run_pipeline: center draw process restarted pid=4343 (attempt 1/3)", err)
        exit_line = next(line for line in err.splitlines() if line.startswith("run_pipeline: exit="))
        self.assertIn("center_process=ok center_restarts=1", exit_line)
        self.assertTrue(rt.sent)


if __name__ == "__main__":
    unittest.main()
