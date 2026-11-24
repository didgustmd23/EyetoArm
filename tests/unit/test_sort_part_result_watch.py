"""진행 중 목표 결과 감시 시험(#373 D-14, ROS 없음).

C가 실행 중에 죽었다 다시 뜨면 원래 결과 future는 끝나지 않는다(가짜 서버 재현). 그래서 A는 오래된 목표의 결과를
다시 묻고, 새 서버의 「모름」(상태 0) 답·실행 번호 변경·대기 한도로 목표를 「모름」으로 끝내 `busy`를 푼다.
끝낸 번호는 그 실행 동안 다시 보내지 않는다.
"""
from __future__ import annotations

import inspect
import json
import tempfile
import threading
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from eyetoarm.transforms import sort_part_client as spc
from eyetoarm.vision.live_sort_part import (DEFAULT_RESULT_WATCH, GOAL_STATUS_UNKNOWN, LiveSortPartSender,
                                            RecordingSortPartClient, ResultWatchConfig, RunSourceConfig, SortPartConfig,
                                            SortPartConfigError, parse_sort_part, result_watch_of)
from test_live_sort_part import FakeRuntime, frame_of, records

S = 1_000_000_000   # ns
WATCH = ResultWatchConfig(check_s=1.0, reask_after_s=30.0, result_limit_s=120.0)
BASE = dict(server_wait_s=0.01, result_wait_s=0.05, close_limit_s=2.0, spin_timeout_s=0.01)


class FakeFuture:
    """rclpy Future 대역: 시험이 `resolve`로 끝낸다."""

    def __init__(self) -> None:
        self._done, self._value, self._cbs = False, None, []

    def add_done_callback(self, cb) -> None:
        """끝났으면 바로, 아니면 끝날 때 부른다."""
        if self._done:
            cb(self)
        else:
            self._cbs.append(cb)

    def result(self):
        """값."""
        return self._value

    def resolve(self, value) -> None:
        """끝낸다."""
        self._done, self._value = True, value

        for cb in self._cbs:
            cb(self)


def answer(status: int, result_status: str = "", reason: str = ""):
    """GetResult 답 대역(`status` = 목표 상태 번호, `result` = SortPart 결과)."""
    return SimpleNamespace(status=status, result=SimpleNamespace(status=result_status, reason=reason, arrived_stamp_ns=0))


class FakeHandle:
    """목표 핸들 대역: `get_result_async`마다 새 future를 만든다."""

    def __init__(self) -> None:
        self.accepted = True
        self.futures: list[FakeFuture] = []

    def get_result_async(self) -> FakeFuture:
        """결과 묻기."""
        f = FakeFuture()
        self.futures.append(f)

        return f


class Stage:
    """B 단계 대역: 풀린 번호를 모은다."""

    def __init__(self) -> None:
        self.released: list = []

    def release(self, tid) -> None:
        """B 중복 기억 해제."""
        self.released.append(tid)


def client(watch=WATCH):
    """rclpy 없이 만든 상속 클라이언트(B 생성자 우회). 실제 B `send`를 쓰도록 액션 클라이언트·메시지를 대역으로 둔다."""
    events = []
    c = object.__new__(RecordingSortPartClient)
    c.node = mock.Mock()
    c.stage = Stage()
    c.stats = Counter()
    c.inflight = {}
    c._lock = threading.Lock()
    c.event_sink = lambda e, g, d: events.append((e, g, d))
    c._arrived = {}
    c._init_watch(watch)
    c.client = mock.Mock()
    c.client.server_is_ready.return_value = True
    c._handles = []

    def send_goal_async(msg, feedback_callback=None, goal_uuid=None):
        f = FakeFuture()
        c._handles.append(f)

        return f

    c.client.send_goal_async.side_effect = send_goal_async
    c._SortPart = SimpleNamespace(Goal=lambda: SimpleNamespace())
    c._Cand = lambda: SimpleNamespace()

    return c, events


def out_of(gid: str, track=7):
    """B 출력 대역(PICK_PLACE)."""
    return SimpleNamespace(goal={"kind": "PICK_PLACE", "goal_id": gid, "spec": "BOLT_B", "part_track_id": track})


def send_and_accept(c, gid: str, track=7) -> FakeHandle:
    """실제 B `send` → 수락까지. 핸들을 돌려준다."""
    with mock.patch.object(spc, "_uuid_msg", lambda g: g):
        assert c.send(out_of(gid, track))

    h = FakeHandle()
    c._handles[-1].resolve(h)

    return h


def names(events):
    """사건 이름만."""
    return [e[0] for e in events]


class BInternalsTest(unittest.TestCase):
    """B `SortPartClient` 안쪽에 기대는 연결을 고정한다(B가 바꾸면 여기서 실패)."""

    def test_b_signatures_and_attrs(self):
        """감시가 쓰는 B 메서드 이름·인자와 생성자 속성(`inflight`·`_lock`·`stage`·`client`)."""
        expect = {"_on_accepted": ["self", "gid", "goal", "fut"], "_on_result": ["self", "gid", "goal", "fut"],
                  "_finish": ["self", "gid", "goal", "status", "reason"], "_release": ["self", "goal"], "send": ["self", "out"]}

        for name, params in expect.items():
            with self.subTest(name=name):
                self.assertEqual(list(inspect.signature(getattr(spc.SortPartClient, name)).parameters), params)

        for attr in ("inflight", "_lock", "stage", "client"):
            self.assertIn(f"self.{attr}", inspect.getsource(spc.SortPartClient.__init__))

    def test_b_finish_pops_inflight_and_releases(self):
        """B `_finish`가 진행 중 표시를 지우고 B 단계 번호를 푼다(「모름」으로 끝낼 때 이것에 기댄다)."""
        c, _ = client()
        send_and_accept(c, "g1", track=7)
        self.assertEqual(c.inflight_ids(), ["g1"])
        spc.SortPartClient._finish(c, "g1", {"spec": "BOLT_B", "part_track_id": 7}, "UNKNOWN", "x")
        self.assertEqual((c.inflight_ids(), c.stage.released), ([], [7]))

    def test_b_accepted_asks_result_once(self):
        """B `_on_accepted`는 수락 때 결과를 한 번만 묻는다(그 future가 끝나지 않으면 영원히 대기)."""
        c, _ = client()
        h = send_and_accept(c, "g1")
        self.assertEqual(len(h.futures), 1)

    def test_b_send_busy_while_inflight(self):
        """진행 중 목표가 있으면 B `send`는 같은 종류를 busy로 거부한다(풀어야 다음을 보낸다)."""
        c, _ = client()
        send_and_accept(c, "g1")

        with mock.patch.object(spc, "_uuid_msg", lambda g: g):
            self.assertFalse(c.send(out_of("g2", track=8)))

        self.assertEqual(c.stats["busy"], 1)


class WatchTest(unittest.TestCase):
    """`RecordingSortPartClient.watch`."""

    def setUp(self):
        self.c, self.ev = client()
        self.h = send_and_accept(self.c, "g1")
        self.t0 = self.c._watch["g1"]["since_ns"]

    def test_no_reask_before_30s(self):
        """30 s 전에는 다시 묻지 않는다."""
        self.c.watch(self.t0 + 29 * S, True)
        self.assertEqual(len(self.h.futures), 1)
        self.assertNotIn("reask_result", names(self.ev))

    def test_reask_unknown_releases_busy(self):
        """30 s 뒤 다시 묻고, 답이 상태 0(모름)이면 result:UNKNOWN/SERVER_LOST로 끝나 busy가 풀리고 번호를 푼다."""
        self.c.watch(self.t0 + 30 * S, True)
        self.assertEqual(len(self.h.futures), 2)
        self.h.futures[1].resolve(answer(GOAL_STATUS_UNKNOWN))
        self.assertIn(("result:UNKNOWN", "g1", {"reason": "SERVER_LOST", "arrived_stamp_ns": None}), self.ev)
        self.assertEqual((self.c.inflight_ids(), self.c.stage.released), ([], [7]))

    def test_pending_reask_not_duplicated(self):
        """다시 묻기 답이 미뤄지면(살아 있는 서버) 그대로 두고, 미해결인 동안 또 묻지 않는다."""
        self.c.watch(self.t0 + 30 * S, True)
        self.c.watch(self.t0 + 65 * S, True)
        self.assertEqual(len(self.h.futures), 2)
        self.assertEqual(self.c.inflight_ids(), ["g1"])

    def test_server_back_triggers_reask(self):
        """서버가 사라졌다(server_lost) 다시 보이면(server_back) 30 s를 기다리지 않고 다시 묻는다."""
        self.c.watch(self.t0 + 1 * S, False)
        self.c.watch(self.t0 + 5 * S, True)
        self.assertEqual(len(self.h.futures), 2)
        self.assertEqual([e[0] for e in self.ev if e[0] in ("server_lost", "server_back", "reask_result")],
                         ["server_lost", "server_back", "reask_result"])
        self.assertEqual([e for e in self.ev if e[0] == "reask_result"][0][2], {"why": "server_back"})

    def test_no_reask_while_server_gone(self):
        """서버가 안 보이는 동안은 다시 묻지 않는다(물어도 답할 곳이 없음)."""
        self.c.watch(self.t0 + 40 * S, False)
        self.assertEqual(len(self.h.futures), 1)

    def test_result_limit(self):
        """120 s(단조 시계) 넘게 결과가 없으면 미해결 다시 묻기가 있어도 RESULT_TIMEOUT으로 끝낸다."""
        self.c.watch(self.t0 + 30 * S, True)
        self.c.watch(self.t0 + 120 * S, True)
        self.assertIn(("result:UNKNOWN", "g1", {"reason": "RESULT_TIMEOUT", "arrived_stamp_ns": None}), self.ev)
        self.assertEqual(self.c.inflight_ids(), [])

    def test_late_original_result_logged_only(self):
        """「모름」 뒤 원래 결과가 늦게 오면 late_result로 기록만 하고 결과를 두 번 내지 않는다."""
        self.c.abandon("g1", "RUN_CHANGED")
        self.h.futures[0].resolve(answer(4, "SUCCEEDED"))
        self.assertEqual(names(self.ev).count("result:UNKNOWN"), 1)
        self.assertNotIn("result:SUCCEEDED", names(self.ev))
        self.assertIn("late_result:SUCCEEDED", names(self.ev))
        self.assertEqual(self.c.watch_counts["late_results"], 1)

    def test_real_answer_and_original_once(self):
        """다시 묻기 답이 진짜 결과이고 원래 결과도 오면 결과 사건은 한 번만(늦은 쪽은 무시)."""
        self.c.watch(self.t0 + 30 * S, True)
        self.h.futures[1].resolve(answer(4, "SUCCEEDED"))
        self.h.futures[0].resolve(answer(4, "SUCCEEDED"))
        self.assertEqual(names(self.ev).count("result:SUCCEEDED"), 1)
        self.assertNotIn("late_result:SUCCEEDED", names(self.ev))

    def test_normal_result_clears_watch(self):
        """정상 결과가 오면 감시 대상에서 빠져 다시 묻지 않는다."""
        self.h.futures[0].resolve(answer(4, "SUCCEEDED"))
        self.c.watch(self.t0 + 60 * S, True)
        self.assertEqual(len(self.h.futures), 1)
        self.assertEqual(self.c._watch, {})

    def test_abandon_after_finish_is_noop(self):
        """이미 끝난 목표를 「모름」으로 끝내려 하면 거짓이고 사건이 없다."""
        self.h.futures[0].resolve(answer(4, "SUCCEEDED"))
        self.assertFalse(self.c.abandon("g1", "RUN_CHANGED"))
        self.assertNotIn("result:UNKNOWN", names(self.ev))


class SenderUnknownTest(unittest.TestCase):
    """송신기: 「모름」 번호 막기·실행 번호 변경 해제."""

    def make(self, managed=False):
        """가짜 런타임 + 송신기."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        rs = RunSourceConfig(service="/c/get_run", poll_s=1.0, call_timeout_s=1.0, stale_s=10.0) if managed else None
        self.rt = FakeRuntime()
        self.rt.abandoned = []

        def abandon(gid, reason):
            if gid not in self.rt.inflight:
                return False

            self.rt.abandoned.append((gid, reason))
            self.rt.finish(gid, "UNKNOWN", reason)

            return True

        self.rt.abandon = abandon
        self.stage = Stage()
        self.sender = LiveSortPartSender("r", SortPartConfig(**BASE, run_source=rs), self.rt, self.dir, tracking=True,
                                         release_target=self.stage)
        self.rt.set_event_sink(self.sender.on_client_event)
        self.sender.start()

        return self.sender

    def out(self, gid, track, run=("", 0)):
        """B 출력(번호·실행 번호)."""
        goal = {"kind": "PICK_PLACE", "goal_id": gid, "spec": "BOLT_B", "part_track_id": track,
                "sort_job_id": run[0], "run_epoch": run[1]}

        return SimpleNamespace(goal=goal, result=SimpleNamespace(valid=True, reason=None))

    def test_unknown_track_not_resent(self):
        """「모름」으로 끝난 번호는 다시 보내지 않고(skip:result_unknown_track), 다른 번호는 보낸다."""
        s = self.make()
        self.assertEqual(s.offer(self.out("g1", 7), frame_of(1), {}, 1), "sent")
        self.rt.finish("g1", "UNKNOWN", "SERVER_LOST")
        self.assertEqual(s.offer(self.out("g2", 7), frame_of(2), {}, 2), "skip:result_unknown_track")
        self.assertEqual(s.offer(self.out("g3", 8), frame_of(3), {}, 3), "sent")
        summary = s.summary()["result_watch"]
        self.assertEqual((summary["released"], summary["blocked"], summary["unknown_tracks"]), (1, 1, {"7": "SERVER_LOST"}))
        self.assertEqual(summary["source"], "default")
        s.close(2.0)
        self.assertIn("release:skip:result_unknown_track", [r["event"] for r in records(Path(self.dir) / "sort_part.jsonl")])
        self.assertEqual(self.stage.released, [7])   # 막은 번호도 B 중복 기억은 푼다(다른 막기와 같음)

    def test_failure_not_blocked_by_unknown_rule(self):
        """실패 결과는 「모름」 막기에 들어가지 않는다(기존 재시도 제한만)."""
        s = self.make()
        s.offer(self.out("g1", 7), frame_of(1), {}, 1)
        self.rt.finish("g1", "FAILED", "MISSED")
        self.assertNotEqual(s.offer(self.out("g2", 7), SimpleNamespace(image_seq=2, ros_stamp_ns=5 * S), {}, 2),
                            "skip:result_unknown_track")

    def test_run_change_abandons_old_and_clears(self):
        """관리 실행: 실행 번호가 바뀌면 이전 실행 목표를 RUN_CHANGED로 끝내고, 새 실행에서는 그 번호를 다시 허용한다."""
        s = self.make(managed=True)
        s.on_run_snapshot(True, "job1", 3)
        self.assertEqual(s.offer(self.out("g1", 7, ("job1", 3)), frame_of(1), {}, 1), "sent")
        s.on_run_snapshot(False, "", 4)   # C 재시작: epoch 올림, job 비움
        self.assertEqual(self.rt.abandoned, [("g1", "RUN_CHANGED")])
        self.assertEqual(s.summary()["result_watch"]["unknown_tracks"], {})   # 새 실행 기억은 비어 있음
        s.on_run_snapshot(True, "job2", 4)
        self.assertEqual(s.offer(self.out("g2", 7, ("job2", 4)), frame_of(2), {}, 2), "sent")
        self.assertEqual(s.summary()["result_watch"]["run_changed_abandoned"], 1)

    def test_same_run_goal_not_abandoned(self):
        """실행 상태만 바뀌고(job·epoch 같음) 번호가 같으면 진행 중 목표를 끝내지 않는다."""
        s = self.make(managed=True)
        s.on_run_snapshot(True, "job1", 3)
        s.offer(self.out("g1", 7, ("job1", 3)), frame_of(1), {}, 1)
        s.on_run_snapshot(False, "job1", 3)
        self.assertEqual(self.rt.abandoned, [])


class ConfigTest(unittest.TestCase):
    """`sort_part.result_watch` 설정."""

    def test_default_when_absent(self):
        """블록이 없으면 기본값(1 s·30 s·120 s)으로 켠다(A 결정 Q1). 설정 요약 형식은 그대로."""
        c = parse_sort_part(dict(BASE))
        self.assertIsNone(c.result_watch)
        self.assertEqual(result_watch_of(c), DEFAULT_RESULT_WATCH)
        self.assertEqual((DEFAULT_RESULT_WATCH.check_s, DEFAULT_RESULT_WATCH.reask_after_s,
                          DEFAULT_RESULT_WATCH.result_limit_s), (1.0, 30.0, 120.0))
        self.assertNotIn("result_watch", c.to_json())

    def test_block(self):
        """블록 값을 읽고 요약에 남긴다."""
        doc = {**BASE, "result_watch": {"check_s": 0.5, "reask_after_s": 10, "result_limit_s": 60}}
        c = parse_sort_part(doc)
        self.assertEqual(result_watch_of(c), ResultWatchConfig(check_s=0.5, reask_after_s=10.0, result_limit_s=60.0))
        self.assertEqual(c.to_json()["result_watch"], {"check_s": 0.5, "reask_after_s": 10.0, "result_limit_s": 60.0})

    def test_rejects(self):
        """키 누락·추가, 0·음수·bool·inf, 다시 묻기 ≥ 한도는 거부."""
        good = {"check_s": 1, "reask_after_s": 30, "result_limit_s": 120}

        for bad in ({k: v for k, v in good.items() if k != "check_s"}, {**good, "x": 1}, {**good, "check_s": 0},
                    {**good, "reask_after_s": -1}, {**good, "result_limit_s": True}, {**good, "check_s": float("inf")},
                    {**good, "reask_after_s": 120}, "x"):
            with self.subTest(bad=bad), self.assertRaises(SortPartConfigError):
                parse_sort_part({**BASE, "result_watch": bad})


class RuntimeWatchTest(unittest.TestCase):
    """`_RosSortPartRuntime` 감시 타이머 부분(ROS 없이 메서드만)."""

    def setUp(self):
        import importlib.util

        path = Path(__file__).resolve().parents[2] / "scripts" / "run_pipeline.py"
        spec = importlib.util.spec_from_file_location("run_pipeline_373", path)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def runtime(self, watch_side_effect=None):
        """생성자 우회 런타임 + 가짜 클라이언트."""
        rt = self.mod._RosSortPartRuntime.__new__(self.mod._RosSortPartRuntime)
        rt._sink = None
        rt.watch_errors = 0
        rt.client = mock.Mock()
        rt.client.client.server_is_ready.return_value = False
        rt.client.watch.side_effect = watch_side_effect
        rt.client.watch_counts = {"checks": 0}
        rt.events = []
        rt.set_event_sink(lambda e, g, d: rt.events.append((e, g, d)))

        return rt

    def test_timer_passes_monotonic_and_ready(self):
        """타이머는 단조 시계 ns와 서버 보임 여부를 넘긴다."""
        rt = self.runtime()

        with mock.patch.object(self.mod.time, "monotonic_ns", return_value=123):
            rt._watch_results()

        rt.client.watch.assert_called_once_with(123, False)

    def test_timer_error_counted_not_raised(self):
        """감시 중 예외는 spin 스레드를 죽이지 않고 세어 result_watch_error 사건으로 남긴다."""
        rt = self.runtime(RuntimeError("boom"))
        rt._watch_results()
        self.assertEqual(rt.watch_errors, 1)
        self.assertEqual(rt.events[0][0], "result_watch_error")
        self.assertEqual(rt.watch_counts, {"checks": 0, "errors": 1})


if __name__ == "__main__":
    unittest.main()
