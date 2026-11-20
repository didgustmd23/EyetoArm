"""#214 D13 해제: 실시간 A→B→C(`/sort_part`) 목표 송신 시험(U1~U14).

근거: 계획 `_workspace/214_builder2_d13_release_plan_r0.md`(reviewer PASS 2026-11-20: Q1 검사 ③ 뒤 송신, Q5 서버 없어도 실행,
Q7 별도 `sort_part.jsonl`, Q8 클라이언트 stats는 참고만). 실제 ROS·Gazebo·`eyetoarm_interfaces`는 쓰지 않는다: 송신 런타임은 가짜
(`FakeRuntime`, B 클라이언트 `send` 규칙 흉내), CLI 시험은 기존 실시간 기록 연결 시험의 가짜 수신·가짜 모델·가짜 TF를 그대로 쓴다.
"""
from __future__ import annotations

import inspect
import io
import json
import subprocess
import threading
import time
import unittest
from collections import Counter
from typing import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from eyetoarm.transforms import sort_part_client as spc
from eyetoarm.vision.live_sort_part import (LiveSortPartSender, RecordingSortPartClient, SortPartConfig,
                                            SortPartConfigError, parse_sort_part)
from test_live_center_display import FakePublisher, write_center_config
from test_run_pipeline_capture_observation import model_with
from test_run_pipeline_live_observation import CONFIG, FakeTf, Scenario
from test_run_pipeline_live_observation import Base as LiveBase

ROOT = Path(__file__).resolve().parents[2]
SORT_CONFIG = ROOT / "config" / "vision" / "live_observation_sort_part_test.json"


def cfg(**over) -> SortPartConfig:
    """시험 설정(결과 대기 짧게)."""
    base = dict(server_wait_s=0.01, result_wait_s=0.05, close_limit_s=2.0, spin_timeout_s=0.01)
    base.update(over)

    return SortPartConfig(**base)


class FakeRuntime:
    """가짜 C 송신 런타임: B `SortPartClient.send`의 반환·계수 규칙(SKIP·no_server·busy)을 흉내낸다. 막기·예외 주입."""

    def __init__(self, *, server=True, busy_after=None, gate=None, raise_on_send=None):
        """`busy_after`: 대기 중 목표가 이 수 이상이면 busy. `gate`: send 안에서 기다릴 Event."""
        self.server = server
        self.server_seen = server
        self.stats = Counter()
        self.inflight = {}
        self.sent = []
        self.busy_after = busy_after
        self.gate = gate
        self.entered = threading.Event()
        self.raise_on_send = raise_on_send
        self.closed = 0
        self.sink: Callable | None = None
        self.close_notes = []
        self.on_close: Callable | None = None

    def set_event_sink(self, sink):
        """사건 받을 곳."""
        self.sink = sink

    def send(self, out):
        """B 클라이언트 규칙 흉내."""
        self.entered.set()

        if self.gate is not None:
            self.gate.wait(5.0)

        if self.raise_on_send is not None:
            raise self.raise_on_send

        goal = out.goal

        if goal.get("kind") not in spc.SEND_KINDS:
            self.stats["skip"] += 1
            return False

        if not self.server:
            self.stats["no_server"] += 1
            return False

        if self.busy_after is not None and len(self.inflight) >= self.busy_after:
            self.stats["busy"] += 1
            return False

        self.stats["sent"] += 1
        self.inflight[goal["goal_id"]] = goal
        self.sent.append(goal)

        return True

    def finish(self, gid, status="SUCCEEDED", reason=""):
        """시험이 C 결과를 흉내낸다(spin 스레드 콜백 대역)."""
        self.inflight.pop(gid, None)
        assert self.sink is not None
        self.sink(f"result:{status}", gid, {"reason": reason, "arrived_stamp_ns": None})

    def inflight_ids(self):
        """결과 대기 goal_id."""
        return list(self.inflight)

    def close(self, limit_s):
        """닫기 기록."""
        if self.on_close is not None:
            self.close_notes.append(self.on_close())

        self.closed += 1


def out_of(kind="PICK_PLACE", valid=True, gid="g-1", reason=None, spec="BOLT_B"):
    """B 출력 대역."""
    return SimpleNamespace(goal={"kind": kind, "goal_id": gid, "spec": spec, "reason": reason},
                           result=SimpleNamespace(valid=valid, reason=reason))


def frame_of(i):
    """프레임 대역."""
    return SimpleNamespace(image_seq=i, ros_stamp_ns=1_000_000 * i)


def records(path):
    """`sort_part.jsonl` 기록."""
    return [json.loads(t) for t in Path(path).read_text(encoding="utf-8").splitlines()]


class SenderBase(unittest.TestCase):
    """송신기 단독 도우미."""

    def make(self, runtime=None, failed=None, **over):
        """임시 폴더에 송신기를 만들고 시작한다."""
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.rt = runtime or FakeRuntime()
        self.failed = failed or {"v": False}
        self.sender = LiveSortPartSender("run-A", cfg(**over), self.rt, self.tmp.name, is_failed=lambda: self.failed["v"])
        self.rt.set_event_sink(self.sender.on_client_event)
        self.sender.start()

        return self.sender

    def done(self):
        """정리 후 기록."""
        self.assertTrue(self.sender.close(2.0))

        return records(Path(self.tmp.name) / "sort_part.jsonl")


# ---------------------------------------------------------------------- 설정·B 클라이언트 연결

class ConfigTest(unittest.TestCase):
    """`sort_part` 블록과 예시 설정."""

    def test_repo_config(self):
        """예시 설정은 test_only이고 sort_part 블록을 읽는다. 기존 설정에는 블록이 없다."""
        from eyetoarm.vision.live_observation_link import load_live_observation_config

        c = load_live_observation_config(str(SORT_CONFIG))
        assert c.sort_part is not None
        self.assertEqual((c.sort_part.server_wait_s, c.sort_part.result_wait_s), (2.0, 3.0))
        self.assertIsNone(load_live_observation_config(str(CONFIG)).sort_part)

    def test_bad_blocks(self):
        """모르는 키·누락·0·bool·inf·형 오류는 거부."""
        good = cfg().to_json()

        for bad in ({**good, "x": 1}, {k: v for k, v in good.items() if k != "result_wait_s"}, {**good, "close_limit_s": 0},
                    {**good, "spin_timeout_s": True}, {**good, "server_wait_s": float("inf")}, "x"):
            with self.subTest(bad=bad), self.assertRaises(SortPartConfigError):
                parse_sort_part(bad)


class RecordingClientTest(unittest.TestCase):
    """U9·U10: A 쪽 상속 클라이언트가 B 비공개 콜백 이름·인자에 맞고, 사건을 기록 함수로 넘긴다."""

    def test_b_methods_exist_with_expected_signatures(self):
        """U9: B `SortPartClient`의 덮어쓴 메서드가 그대로 있다(B가 바꾸면 이 시험이 실패)."""
        expect = {"_on_feedback": ["self", "gid", "fb"], "_on_accepted": ["self", "gid", "goal", "fut"],
                  "_on_result": ["self", "gid", "goal", "fut"], "_finish": ["self", "gid", "goal", "status", "reason"],
                  "send": ["self", "out"]}

        for name, params in expect.items():
            with self.subTest(name=name):
                self.assertEqual(list(inspect.signature(getattr(spc.SortPartClient, name)).parameters), params)

        for attr in ("inflight", "_lock", "stats", "client"):
            self.assertIn(f"self.{attr}", inspect.getsource(spc.SortPartClient.__init__))

    def client(self):
        """rclpy 없이 만든 상속 클라이언트(B 생성자 우회, 같은 속성)."""
        events = []
        c = object.__new__(RecordingSortPartClient)
        c.node = mock.Mock()
        c.stage = None
        c.stats = Counter()
        c.inflight = {"g1": {"kind": "PICK_PLACE"}}
        c._lock = threading.Lock()
        c.event_sink = lambda e, g, d: events.append((e, g, d))
        c._arrived = {}

        return c, events

    def test_events_flow(self):
        """U10: feedback → accepted → result(도착 시각 포함) 순으로 사건, 결과 뒤 대기 목표 없음."""
        c, ev = self.client()
        goal = {"spec": "BOLT_B", "part_track_id": None}
        c._on_feedback("g1", SimpleNamespace(feedback=SimpleNamespace(stage="ACCEPTED")))
        handle = mock.Mock(accepted=True)
        c._on_accepted("g1", goal, mock.Mock(result=mock.Mock(return_value=handle)))
        result = SimpleNamespace(result=SimpleNamespace(status="SUCCEEDED", reason="", arrived_stamp_ns=123))
        c._on_result("g1", goal, mock.Mock(result=mock.Mock(return_value=result)))
        self.assertEqual([e[0] for e in ev], ["feedback:ACCEPTED", "accepted", "result:SUCCEEDED"])
        self.assertEqual(ev[-1][2], {"reason": "", "arrived_stamp_ns": 123})
        self.assertEqual(c.inflight_ids(), [])

    def test_rejected_and_error(self):
        """U10: 거절 → result:REJECTED, 송신 future 예외 → result:ERROR(accepted 없음)."""
        c, ev = self.client()
        c._on_accepted("g1", {}, mock.Mock(result=mock.Mock(return_value=mock.Mock(accepted=False))))
        c.inflight["g2"] = {"kind": "PICK_PLACE"}
        c._on_accepted("g2", {}, mock.Mock(result=mock.Mock(side_effect=RuntimeError("x"))))
        self.assertEqual([(e[0], e[1]) for e in ev], [("result:REJECTED", "g1"), ("result:ERROR", "g2")])


# ---------------------------------------------------------------------- 송신기(가짜 런타임)

class SenderTest(SenderBase):
    """U2·U4·U6·U7·U8·U10."""

    def test_valid_accept_sends_once_with_keys(self):
        """U2: 유효 수락 1건 → send 1회, `sent` 기록에 프레임 키·accept 순번·goal_id."""
        s = self.make()
        self.assertEqual(s.offer(out_of(gid="g-1"), frame_of(3), {"job_seq": 7, "generation": 2}, 41), "sent")
        recs = self.done()
        self.assertEqual(len(self.rt.sent), 1)
        sent = [r for r in recs if r["event"] == "sent"][0]
        self.assertEqual({k: sent[k] for k in ("goal_id", "run_id", "image_seq", "ros_stamp_ns", "accept_event_seq",
                                               "job_seq", "generation", "kind", "spec")},
                         {"goal_id": "g-1", "run_id": "run-A", "image_seq": 3, "ros_stamp_ns": 3_000_000,
                          "accept_event_seq": 41, "job_seq": 7, "generation": 2, "kind": "PICK_PLACE", "spec": "BOLT_B"})

    def test_invalid_and_skip_not_sent(self):
        """U4: 무효·UNRESOLVED·SKIP(B_TIMEOUT·OUT_OF_REACH) → send 0, skip:<사유>."""
        s = self.make()
        events = [s.offer(out_of(valid=False, reason="AXIS_AMBIGUOUS"), frame_of(1), {}, 1),
                  s.offer(out_of(valid=False, reason=None), frame_of(2), {}, 2),
                  s.offer(out_of(kind="SKIP", reason="B_TIMEOUT"), frame_of(3), {}, 3),
                  s.offer(out_of(kind="SKIP", reason="OUT_OF_REACH"), frame_of(4), {}, 4)]
        self.assertEqual(events, ["skip:AXIS_AMBIGUOUS", "skip:invalid", "skip:B_TIMEOUT", "skip:OUT_OF_REACH"])
        self.done()
        self.assertEqual(self.rt.sent, [])

    def test_close_run_then_zero_sends(self):
        """U6: close_run 뒤 송신 0(gate_closed), 닫힘 사유 기록."""
        s = self.make()
        s.close_run("reset:sim_time_rewind:epoch2")
        self.assertEqual(s.offer(out_of(), frame_of(1), {}, 1), "gate_closed")
        recs = self.done()
        self.assertEqual(self.rt.sent, [])
        self.assertIn({"event": "gate_closed_by", "reason": "reset:sim_time_rewind:epoch2"},
                      [{k: r.get(k) for k in ("event", "reason")} for r in recs])

    def test_close_waits_for_in_flight_send(self):
        """U6 경합: send 진행 중 close_run은 그 송신이 끝날 때까지 기다리고, 그 뒤 송신 0. 대기 시간 기록."""
        gate = threading.Event()
        s = self.make(FakeRuntime(gate=gate))
        t = threading.Thread(target=lambda: s.offer(out_of(gid="g-a"), frame_of(1), {}, 1))
        t.start()
        self.assertTrue(self.rt.entered.wait(2.0))
        closed = threading.Event()
        threading.Thread(target=lambda: (s.close_run("reset"), closed.set())).start()
        time.sleep(0.1)
        self.assertFalse(closed.is_set())                      # 진행 중 송신을 기다림
        gate.set()
        self.assertTrue(closed.wait(2.0))
        t.join(2.0)
        self.assertEqual(s.offer(out_of(gid="g-b"), frame_of(2), {}, 2), "gate_closed")
        self.done()
        self.assertEqual([g["goal_id"] for g in self.rt.sent], ["g-a"])
        self.assertGreaterEqual(s.gate_wait_max_ms, 90.0)

    def test_state_failed_blocks(self):
        """U7: 상태 fail-stop이면 송신 0(state_failed)."""
        failed = {"v": False}
        s = self.make(failed=failed)
        failed["v"] = True
        self.assertEqual(s.offer(out_of(), frame_of(1), {}, 1), "state_failed")
        self.done()
        self.assertEqual(self.rt.sent, [])

    def test_no_server_and_busy_recorded_then_retry(self):
        """U8: 서버 없음 → no_server, 결과 대기 중 → busy, 결과 뒤 다음 수락에서 다시 보냄."""
        rt = FakeRuntime(server=False)
        s = self.make(rt)
        self.assertEqual(s.offer(out_of(gid="g0"), frame_of(0), {}, 0), "no_server")
        rt.server = True
        rt.busy_after = 1
        self.assertEqual(s.offer(out_of(gid="g1"), frame_of(1), {}, 1), "sent")
        self.assertEqual(s.offer(out_of(gid="g2"), frame_of(2), {}, 2), "busy")
        rt.finish("g1")
        self.assertEqual(s.offer(out_of(gid="g3"), frame_of(3), {}, 3), "sent")
        rt.finish("g3")
        recs = self.done()
        self.assertEqual([g["goal_id"] for g in rt.sent], ["g1", "g3"])
        res = [r for r in recs if r["event"] == "result:SUCCEEDED"]
        self.assertEqual([(r["goal_id"], r["image_seq"]) for r in res], [("g1", 1), ("g3", 3)])   # 결과에 프레임 키

    def test_unresolved_at_close(self):
        """U10: 닫을 때 결과 대기 목표는 result_wait 뒤 unresolved_at_close, 런타임 닫힘 1회."""
        s = self.make()
        s.offer(out_of(gid="g-x"), frame_of(5), {"job_seq": 9}, 5)
        recs = self.done()
        un = [r for r in recs if r["event"] == "unresolved_at_close"]
        self.assertEqual([(r["goal_id"], r["image_seq"], r["job_seq"]) for r in un], [("g-x", 5, 9)])
        self.assertEqual((s.summary()["unresolved_at_close"], self.rt.closed), (["g-x"], 1))

    def test_send_exception_propagates_without_record_loss(self):
        """송신 예외는 호출자(디스패처 수락 훅 오류)로 올라가고, 다음 송신은 계속 판단한다."""
        rt = FakeRuntime(raise_on_send=RuntimeError("boom"))
        s = self.make(rt)

        with self.assertRaises(RuntimeError):
            s.offer(out_of(gid="g1"), frame_of(1), {}, 1)

        rt.raise_on_send = None
        self.assertEqual(s.offer(out_of(gid="g2"), frame_of(2), {}, 2), "sent")
        self.done()


# ---------------------------------------------------------------------- CLI(가짜 수신·가짜 모델·가짜 TF·가짜 송신 런타임)

class CliBase(LiveBase):
    """기존 실시간 기록 연결 CLI 시험에 송신 런타임 주입을 더한다."""

    def setUp(self):
        """런타임 기록."""
        super().setUp()
        self.rt = None
        self.factory_calls = []

    def factory(self, runtime=None):
        """`live_sort_part_factory(cfg)` 대역."""
        def make(c):
            self.factory_calls.append(c)
            self.rt = runtime or FakeRuntime()
            return self.rt

        return make

    def run_sort(self, scenario, *, config: Path | str = SORT_CONFIG, extra=("--live-sort-part",), factory=None, tf=None, max_frames=None,
                 out=None, display_factory=None, hook=None):
        """main 실행 → (종료 코드, stderr, 출력 폴더)."""
        out = out or self.out()
        log = str(Path(self.tmp.name) / f"{Path(out).name}.jsonl")
        made = []
        tf = tf if tf is not None else FakeTf()
        scenario.link = lambda: self.controller.current
        argv = ["--ros-link", "--detector", "ai", "--ai-config", self.case.ai, "--log", log,
                "--live-observation-out", out, "--live-observation-config", str(config),
                "--max-frames", str(max_frames or scenario.n), *extra]
        err = io.StringIO()

        try:
            code = self.mod.main(argv, stderr=err, ros_available=lambda: True,
                                 model_factory=lambda path: model_with(scenario.masks_fn),
                                 live_source_factory=scenario.factory_maker(made), live_tf_factory=lambda c: tf,
                                 live_run_controller=self.controller, b_stage_hook=hook,
                                 live_sort_part_factory=factory or self.factory(),
                                 live_display_factory=display_factory)
        except SystemExit as exc:
            code = exc.code

        return code, err.getvalue(), out

    def accepts(self, out):
        """수락 기록."""
        return [r for r in self.records(out) if r["kind"] == "accept"]

    def sort_records(self, out):
        """송신 기록."""
        return records(Path(out) / "sort_part.jsonl")


class CliOnTest(CliBase):
    """U2·U3·U5·U6·U11·U13 CLI 경로."""

    def test_each_accept_tried_once_and_goal_matches_b(self):
        """U2·U3: 수락마다 송신 판단 1회(같은 프레임 두 번 없음), 보낸 목표 = accept 기록의 B 목표(goal_id·spec·kind)."""
        rt = FakeRuntime()
        code, err, out = self.run_sort(Scenario(4), factory=self.factory(rt))
        self.assertEqual(code, 0, err)
        self.assertIn("LIVE A->B->C SORT_PART ON action=/sort_part server_seen=true", err)
        self.assertIn("C goals via /sort_part", err)
        acc = self.accepts(out)
        recs = self.sort_records(out)
        tried = [r for r in recs if r["event"] in ("sent", "busy", "no_server", "gate_closed", "state_failed")
                 or r["event"].startswith("skip:")]
        self.assertEqual(sorted(r["accept_event_seq"] for r in tried), sorted(r["event_seq"] for r in acc))
        self.assertEqual(len({r["image_seq"] for r in tried}), len(tried))
        by_gid = {r["goal"]["goal_id"]: r for r in acc}

        for g in rt.sent:
            self.assertEqual((g["kind"], g["spec"]), (by_gid[g["goal_id"]]["goal"]["kind"], by_gid[g["goal_id"]]["goal"]["spec"]))

        self.assertEqual(len(rt.sent), sum(1 for r in recs if r["event"] == "sent"))
        s = self.summary(out)
        self.assertEqual((s["record_only"], s["cleanup_confirmed"], s["sort_part"]["server_seen"]), (False, True, True))
        self.assertEqual(rt.closed, 1)

    def test_not_accepted_never_sent(self):
        """U5: 계산 중 무효화(not_accepted)된 결과는 송신 0. on_output 경로가 없다(B 단계 on_output은 None)."""
        from eyetoarm.vision.live_observation_dispatch import LiveObservationDispatcher

        stages = []
        real_accept = None

        def hook(stage):
            """B 단계 기록."""
            stages.append(stage)

        rt = FakeRuntime()
        sc = Scenario(4)
        orig = LiveObservationDispatcher._process

        def process(disp, item):
            """두 번째 수락 시도는 검사 ③을 거부한다(계산 중 무효화 흉내)."""
            nonlocal real_accept

            if real_accept is None:
                real_accept = disp.state.accept
                calls = {"n": 0}

                def accept(cid, token, payload):
                    calls["n"] += 1
                    return None if calls["n"] == 2 else real_accept(cid, token, payload)

                disp.state.accept = accept

            return orig(disp, item)

        with mock.patch.object(LiveObservationDispatcher, "_process", process):
            code, err, out = self.run_sort(sc, factory=self.factory(rt), hook=hook)

        self.assertEqual(code, 0, err)
        self.assertTrue(stages and all(st.on_output is None for st in stages))
        skipped = {r["ros_stamp_ns"] for r in self.skipped(out) if r["reason"] == "not_accepted"}
        self.assertEqual(len(skipped), 1)
        self.assertFalse({r.get("ros_stamp_ns") for r in self.sort_records(out)} & skipped)
        self.assertEqual(len(rt.sent), len(self.accepts(out)))

    def test_reset_then_zero_sends(self):
        """U6(CLI): 리셋 → 게이트 닫힘(사유 reset), 그 뒤 송신 0, 정리 확인."""
        from test_run_pipeline_live_observation import ResetAndRestartTest

        rt = FakeRuntime()
        sc = Scenario(5, during=ResetAndRestartTest.rewind_at(self, 2))
        code, err, out = self.run_sort(sc, factory=self.factory(rt), max_frames=5)
        self.assertEqual(code, 0, err)
        self.assertIn("reason=reset_detected", err)
        recs = self.sort_records(out)
        closed_at = [i for i, r in enumerate(recs) if r["event"] == "gate_closed_by"]
        self.assertTrue(closed_at)
        self.assertTrue(recs[closed_at[0]]["reason"].startswith("reset:sim_time_rewind"))
        self.assertFalse([r for r in recs[closed_at[0]:] if r["event"] == "sent"])
        self.assertEqual(len(rt.sent), sum(1 for r in recs if r["event"] == "sent"))
        self.assertTrue(self.summary(out)["cleanup_confirmed"])

    def test_cleanup_order_after_dispatcher(self):
        """U11: 송신 런타임을 닫는 순간 worker는 멈춰 있다(dispatcher 다음), 정리 확인. 닫기 실패면 확인 거짓·재시도."""
        holder = {}
        rt = FakeRuntime()
        rt.on_close = lambda: holder["link"].dispatcher._thread.is_alive()
        sc = Scenario(3)
        orig = sc.masks_fn

        def masks():
            """실행 중 연결부를 잡아 둔다."""
            holder["link"] = self.controller.current
            return orig()

        sc.masks_fn = masks
        code, err, out = self.run_sort(sc, factory=self.factory(rt))
        self.assertEqual(code, 0, err)
        self.assertEqual(rt.close_notes, [False])

    def test_runtime_close_failure_not_confirmed(self):
        """U11: 런타임 닫기가 실패하면 정리 확인 거짓·종료 1·close_errors에 sort_part."""
        rt = FakeRuntime()

        def bad_close(limit_s):
            raise RuntimeError("spin thread did not stop")

        rt.close = bad_close
        code, err, out = self.run_sort(Scenario(3), factory=self.factory(rt))
        self.assertEqual(code, 1, err)
        s = self.summary(out)
        self.assertFalse(s["cleanup_confirmed"])
        self.assertTrue(any(e.startswith("sort_part:") for e in s["ended"]["close_errors"]), s["ended"])

    def test_no_server_still_runs(self):
        """Q5: 서버가 없어도 실행하고 server_seen=false 경고, 기록은 no_server, 종료 0."""
        rt = FakeRuntime(server=False)
        code, err, out = self.run_sort(Scenario(3), factory=self.factory(rt))
        self.assertEqual(code, 0, err)
        self.assertIn("server_seen=false", err)
        self.assertIn("warning: /sort_part server not seen", err)
        self.assertTrue(all(r["event"] in ("no_server", "gate_closed_by") for r in self.sort_records(out)))

    def test_with_center_display_both_once_send_first(self):
        """U13: 표시와 함께 켜면 같은 수락에 송신 판단 1회·표시 1회, 송신이 먼저. 표시 실패는 송신에 영향 없음."""
        order = []
        rt = FakeRuntime()
        real_send = rt.send

        def send(out):
            """순서 기록."""
            order.append(("send", out.goal["goal_id"]))
            return real_send(out)

        rt.send = send
        from eyetoarm.vision import live_center_display as lcd

        real_offer = lcd.LiveCenterDisplay.offer_accepted

        def offer(disp, frame, annotated, out, ids, vb=None):
            """순서 기록."""
            order.append(("display", out.goal["goal_id"]))
            return real_offer(disp, frame, annotated, out, ids, vb)

        cfg_path = write_center_config(Path(self.tmp.name) / "both.json", status_hold_s=100.0, publish_fail_limit=1)
        doc = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        doc["sort_part"] = json.loads(SORT_CONFIG.read_text(encoding="utf-8"))["sort_part"]
        Path(cfg_path).write_text(json.dumps(doc), encoding="utf-8")
        pub = FakePublisher(fail=-1)

        with mock.patch.object(lcd.LiveCenterDisplay, "offer_accepted", offer):
            code, err, out = self.run_sort(Scenario(4), config=cfg_path, factory=self.factory(rt),
                                           extra=("--live-sort-part", "--live-center-topic"),
                                           display_factory=lambda t, c: pub)

        self.assertEqual(code, 1, err)                          # 표시 실패는 종료 1(#213 Q6)
        acc = self.accepts(out)
        sends = [g for k, g in order if k == "send"]
        displays = [g for k, g in order if k == "display"]
        self.assertEqual(sends, displays)                       # 수락마다 둘 다 1회
        self.assertEqual(len(sends), len(acc))
        self.assertTrue(all(order[2 * i] == ("send", sends[i]) for i in range(len(sends))))
        self.assertEqual(len(rt.sent), len(acc))                # 표시 실패와 무관하게 송신
        self.assertIn("center display stopped: publish_failed", err)


class CliOffAndArgsTest(CliBase):
    """U1·U12·U14."""

    def test_off_creates_nothing_and_keeps_summary(self):
        """U1: 옵션 없으면 런타임 공장 0, sort_part.jsonl 없음, 요약에 sort_part 없음·record_only 참, 시작 줄 그대로."""
        code, err, out = self.run_sort(Scenario(3), extra=(), config=CONFIG)
        self.assertEqual(code, 0, err)
        self.assertEqual(self.factory_calls, [])
        self.assertFalse((Path(out) / "sort_part.jsonl").exists())
        s = self.summary(out)
        self.assertNotIn("sort_part", s)
        self.assertEqual((s["record_only"], s["outputs"]), (True, "none (no C/D/robot output)"))
        self.assertIn("no C/D/robot output, no auto restart", err)
        self.assertNotIn("SORT_PART", err)

    def test_argument_and_config_errors(self):
        """U12: 기록 연결 없이·단계 L·블록 없음·블록만 있음·잘못된 블록은 2, 런타임·TF 0."""
        base = ["--detector", "ai", "--ai-config", self.case.ai, "--log", str(Path(self.tmp.name) / "x.jsonl")]
        bad = Path(self.tmp.name) / "bad.json"
        doc = json.loads(SORT_CONFIG.read_text(encoding="utf-8"))
        doc["sort_part"]["spin_timeout_s"] = 0
        bad.write_text(json.dumps(doc), encoding="utf-8")
        live = ["--live-observation-out", self.out(), "--live-observation-config"]
        cases = [["--ros-link", *base, "--live-sort-part"],
                 ["--ros-link", *base, *live, str(SORT_CONFIG), "--b-input", "list", "--live-sort-part"],
                 ["--ros-link", *base, *live, str(CONFIG), "--live-sort-part"],
                 ["--ros-link", *base, *live, str(SORT_CONFIG)],
                 ["--ros-link", *base, *live, str(bad), "--live-sort-part"]]

        for argv in cases:
            with self.subTest(argv=argv):
                err = io.StringIO()

                try:
                    code = self.mod.main(argv, stderr=err, ros_available=lambda: True,
                                         live_tf_factory=lambda c: self.fail("tf must not start"),
                                         live_run_controller=self.controller,
                                         live_sort_part_factory=lambda c: self.fail("runtime must not start"))
                except SystemExit as exc:
                    code = exc.code

                self.assertEqual(code, 2, err.getvalue())

        self.assertIsNone(self.controller.current)

    def test_missing_interfaces_exit_1_before_session(self):
        """인터페이스가 없으면(ImportError) 1, 출력 폴더·TF 없음."""
        def missing(c):
            raise ImportError("No module named 'eyetoarm_interfaces'")

        code, err, out = self.run_sort(Scenario(2), factory=missing, tf=None)
        self.assertEqual(code, 1, err)
        self.assertIn("eyetoarm_interfaces not found", err)
        self.assertFalse(Path(out).exists())

    def test_runner_d13_guard_unchanged(self):
        """U14: 실행기는 실시간 b_consumer를 계속 거부하고(옵션과 무관), pipeline_runner.py는 HEAD와 같다."""
        from test_live_observation_harness import D13RegressionTest

        D13RegressionTest("test_run_still_rejects_live_observation_consumer").test_run_still_rejects_live_observation_consumer()
        diff = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src/eyetoarm/vision/pipeline_runner.py"], cwd=ROOT)
        self.assertEqual(diff.returncode, 0)

    def test_on_output_stage_still_refused_with_sending(self):
        """U5·U14: 송신을 켜도 on_output이 있는 B 단계는 거부(수락 0·송신 0·종료 1)."""
        def hook(stage):
            """C 출력 직접 연결 흉내."""
            stage.on_output = lambda out: None

        rt = FakeRuntime()
        code, err, out = self.run_sort(Scenario(3), factory=self.factory(rt), hook=hook)
        self.assertEqual(code, 1)
        self.assertEqual((self.accepts(out), rt.sent), ([], []))

    def test_sender_module_has_no_ros_import(self):
        """송신 게이트 모듈은 rclpy를 import하지 않는다(B 클라이언트 생성자만 지연 import)."""
        src = (ROOT / "src" / "eyetoarm" / "vision" / "live_sort_part.py").read_text(encoding="utf-8")
        code = src.split('"""', 2)[2]
        self.assertNotIn("import rclpy", code)
        self.assertNotIn("on_output", code)


if __name__ == "__main__":
    unittest.main()
