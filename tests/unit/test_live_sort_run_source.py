"""#301 P2: 관리 실행 송신(C `GetSortRun` 실행 번호 → B 목표 `sort_job_id`·`run_epoch`) 시험.

실제 ROS·Gazebo·`eyetoarm_interfaces` 없이 가짜 런타임·가짜 서비스 클라이언트를 쓴다.
근거: C 상세설계 §16.5.2(run_epoch 다르면 폐기), `sort_manager.py` e6cb4e1(서비스 이름·MISMATCH 판정),
계획 `_workspace/301_builder2_p2_plan_r0.md`(사용자 승인 Q1·Q2·Q3·Q6·Q7).
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from eyetoarm.transforms.b_stage import BStage
from eyetoarm.vision.live_sort_part import (LiveSortPartSender, RetryConfig, RunSourceConfig, SortPartConfig,
                                            SortPartConfigError, parse_sort_part)
from test_live_sort_part import CliBase, FakeRuntime, frame_of, records
from test_live_tracking_link import PIXEL
from test_run_pipeline_live_observation import Scenario

ROOT = Path(__file__).resolve().parents[2]
MANAGED_CONFIG = ROOT / "config" / "vision" / "live_observation_sort_part_managed_test.json"
PLAIN_CONFIG = ROOT / "config" / "vision" / "live_observation_sort_part_test.json"
SERVICE = "/c_sort_goal_receiver/get_run"   # sort_manager.py:65·85 노드 이름 + ~/get_run (Q7)
RUN_SOURCE = RunSourceConfig(service=SERVICE, poll_s=0.5, call_timeout_s=1.0, stale_s=2.0)
BASE = dict(server_wait_s=0.01, result_wait_s=0.05, close_limit_s=2.0, spin_timeout_s=0.01)
SEC = 1_000_000_000
# 관리 설정은 require_track_id=true라 --tracking이 필요하다(run_pipeline 짝 검사). 진입점 시험은 픽셀 추적으로 돌린다
MANAGED_EXTRA = ("--live-sort-part", "--tracking", *PIXEL)
TRACK_FRAMES = 9   # 픽셀 추적이 번호를 확정하고 송신까지 가는 프레임 수(test_tracking_send_fixes와 같음)


class Clock:
    """시험이 움직이는 단조 시계(ns)."""

    def __init__(self):
        """0에서 시작."""
        self.ns = 0

    def __call__(self):
        """지금 시각."""
        return self.ns


class RefreshRuntime(FakeRuntime):
    """즉시 재조회 요청 수를 세는 가짜 런타임."""

    def __init__(self, **kw):
        """요청 수 0."""
        super().__init__(**kw)
        self.refreshes = 0

    def request_run_refresh(self):
        """재조회 요청 기록."""
        self.refreshes += 1


def goal_out(gid, job=None, epoch=None, kind="PICK_PLACE", track=None, pos=(10.0, 20.0, 30.0), stamp=5):
    """B 출력 대역(목표를 만든 때의 실행 번호 포함)."""
    goal = {"kind": kind, "goal_id": gid, "spec": "BOLT_B", "reason": None, "sort_job_id": job, "run_epoch": epoch,
            "part_track_id": track, "observed_position_mm": list(pos), "capture_stamp_ns": stamp}

    return SimpleNamespace(goal=goal, result=SimpleNamespace(valid=True, reason=None))


class ReleaseTarget:
    """B 기록 해제 대역."""

    def __init__(self):
        """호출 기록."""
        self.calls = []

    def release(self, track):
        """해제 기록."""
        self.calls.append(track)


class ConfigTest(unittest.TestCase):
    """`sort_part.run_source` 설정 검사."""

    def test_absent_means_unmanaged_and_same_json(self):
        """블록이 없으면 비관리이고, 요약 설정 형식이 예전과 같다(run_source 키 없음)."""
        c = parse_sort_part(dict(BASE))
        self.assertIsNone(c.run_source)
        self.assertNotIn("run_source", c.to_json())

    def test_repo_managed_config(self):
        """저장소 관리 시험 설정: 서비스 이름·시험값(Q6·Q7)·require_track_id=true, 다른 값은 원본 시험 설정과 같다."""
        doc = json.loads(MANAGED_CONFIG.read_text(encoding="utf-8"))
        plain = json.loads(PLAIN_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(doc["status"], "test_only")
        c = parse_sort_part(doc["sort_part"])
        self.assertEqual(c.run_source, RUN_SOURCE)
        self.assertIs(c.require_track_id, True)   # 관리 실행은 추적 번호 대상(#307)
        self.assertEqual(c.to_json()["run_source"],
                         {"call_timeout_s": 1.0, "poll_s": 0.5, "service": SERVICE, "stale_s": 2.0})
        rest = {k: v for k, v in doc.items() if k not in ("note", "sort_part")}
        self.assertEqual(rest, {k: v for k, v in plain.items() if k not in ("note", "sort_part")})
        self.assertEqual({k: v for k, v in doc["sort_part"].items() if k not in ("run_source", "require_track_id")},
                         plain["sort_part"])

    def test_bad_run_source_refused(self):
        """키 누락·모르는 키·0 이하·bool·상대 이름·공백 이름은 거부한다."""
        good = {"service": SERVICE, "poll_s": 0.5, "call_timeout_s": 1.0, "stale_s": 2.0}
        bad = [
            "x",
            {k: v for k, v in good.items() if k != "stale_s"},
            {**good, "extra": 1},
            {**good, "poll_s": 0},
            {**good, "call_timeout_s": -1.0},
            {**good, "stale_s": True},
            {**good, "stale_s": float("nan")},
            {**good, "service": "c_sort_goal_receiver/get_run"},
            {**good, "service": "/a b"},
            {**good, "service": "/"},
            {**good, "service": 3},
        ]

        for r in bad:
            with self.subTest(r=r), self.assertRaises(SortPartConfigError):
                parse_sort_part({**BASE, "run_source": r})


class SenderBase(unittest.TestCase):
    """관리 송신기 도우미."""

    def make(self, runtime=None, retry=None, run_source=RUN_SOURCE, run_target=None):
        """임시 폴더에 송신기를 만들고 시작한다."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.clock = Clock()
        self.rt = runtime or RefreshRuntime()
        self.release = ReleaseTarget()
        self.applied = []
        config = SortPartConfig(**BASE, retry=retry, run_source=run_source)
        self.sender = LiveSortPartSender("run-A", config, self.rt, self.tmp.name, clock_ns=self.clock,
                                         release_target=self.release,
                                         run_target=run_target or (lambda j, e: self.applied.append((j, e))))
        self.rt.set_event_sink(self.sender.on_client_event)
        self.sender.start()
        self.addCleanup(self._close_sender)   # 기록 스레드를 남기지 않는다(다른 시험의 스레드 수 검사)

        return self.sender

    def _close_sender(self):
        """송신기를 닫는다(이미 닫혔으면 아무 일 없음). 기록 스레드가 끝났는지 확인한다."""
        self.assertTrue(self.sender.close(2.0))
        self.assertFalse(self.sender.thread.is_alive())

    def records(self):
        """닫고 기록을 읽는다."""
        self.assertTrue(self.sender.close(2.0))

        return records(Path(self.tmp.name) / "sort_part.jsonl")


class SendConditionTest(SenderBase):
    """송신 조건(Q1·Q2, 계획 §2.3)."""

    def test_unknown_before_any_answer(self):
        """답이 없으면 보내지 않고(skip:run_unknown) B 기록을 푼다."""
        s = self.make()
        self.assertEqual(s.offer(goal_out("g1", track=4), frame_of(1), {}, 1), "skip:run_unknown")
        self.assertEqual(self.rt.sent, [])
        self.assertEqual(self.release.calls, [4])

    def test_stale_answer_is_unknown(self):
        """마지막 답이 stale_s보다 오래되면 모름. 경계(=stale_s)는 아직 보낸다."""
        s = self.make()
        s.on_run_snapshot(True, "job-1", 7)
        self.clock.ns = 2 * SEC
        self.assertEqual(s.offer(goal_out("g1", "job-1", 7), frame_of(1), {}, 1), "sent")
        self.clock.ns = 2 * SEC + 1
        self.assertEqual(s.offer(goal_out("g2", "job-1", 7), frame_of(2), {}, 2), "skip:run_unknown")

    def test_inactive_blocks(self):
        """active=false(실행 전·RECOVERY_REQUIRED)면 skip:run_inactive."""
        s = self.make()
        s.on_run_snapshot(False, "", 3)
        self.assertEqual(s.offer(goal_out("g1", None, None, track=2), frame_of(1), {}, 1), "skip:run_inactive")
        self.assertEqual(self.rt.sent, [])
        self.assertEqual(self.release.calls, [2])

    def test_goal_built_for_other_run_blocked(self):
        """목표에 실린 값이 지금 실행과 다르면(job·epoch 하나라도) skip:run_changed."""
        s = self.make()
        s.on_run_snapshot(True, "job-2", 8)

        for gid, job, epoch in (("a", None, None), ("b", "job-1", 8), ("c", "job-2", 7)):
            with self.subTest(gid=gid):
                self.assertEqual(s.offer(goal_out(gid, job, epoch), frame_of(1), {}, 1), "skip:run_changed")

        self.assertEqual(self.rt.sent, [])

    def test_active_sends_with_run_numbers_recorded(self):
        """활성·일치면 보내고, 기록 줄에 sort_job_id·run_epoch가 붙는다."""
        s = self.make()
        s.on_run_snapshot(True, "job-1", 7)
        self.assertEqual(s.offer(goal_out("g1", "job-1", 7), frame_of(1), {}, 1), "sent")
        self.assertEqual([(g["sort_job_id"], g["run_epoch"]) for g in self.rt.sent], [("job-1", 7)])
        sent = [r for r in self.records() if r["event"] == "sent"]
        self.assertEqual([(r["sort_job_id"], r["run_epoch"]) for r in sent], [("job-1", 7)])

    def test_unmanaged_unchanged(self):
        """run_source 없음: 실행 번호를 보지 않고 보내며, 기록 줄에 실행 번호 키를 붙이지 않는다."""
        s = self.make(run_source=None)
        self.assertFalse(s.managed)
        self.assertEqual(s.offer(goal_out("g1"), frame_of(1), {}, 1), "sent")
        recs = self.records()
        self.assertTrue(all("sort_job_id" not in r and "run_epoch" not in r for r in recs))
        self.assertNotIn("run_source", s.summary())
        self.assertEqual(self.applied, [])


class RunChangeTest(SenderBase):
    """실행이 바뀔 때(계획 §2.5, Q3)."""

    def test_state_and_change_records_and_stage_apply(self):
        """첫 답은 run_state만, (job, epoch) 변화는 run_changed. B 단계에는 활성일 때 값, 비활성이면 None."""
        s = self.make()
        s.on_run_snapshot(False, "", 3)        # C 기동 직후(job 없음)
        s.on_run_snapshot(False, "", 3)        # 같은 답: 기록 없음
        s.on_run_snapshot(True, "job-1", 3)    # 첫 start_run: epoch 그대로
        s.on_run_snapshot(True, "job-2", 4)    # 두 번째 start_run: epoch +1
        self.assertEqual(self.applied, [("job-1", 3), ("job-2", 4)])
        recs = self.records()
        runs = [r for r in recs if r["event"].startswith("run_")]
        self.assertEqual([r["event"] for r in runs], ["run_state", "run_state", "run_changed", "run_state", "run_changed"])
        self.assertEqual((runs[-1]["old"]["sort_job_id"], runs[-1]["new"]),
                         ("job-1", {"active": True, "sort_job_id": "job-2", "run_epoch": 4}))
        counts = s.summary()["run_source"]["counts"]
        self.assertEqual((counts["snapshots"], counts["state_changes"], counts["run_changes"]), (4, 3, 2))

    def test_recovery_inactive_then_active_reapplies(self):
        """같은 job·epoch에서 비활성(복구 필요)→활성: run_changed 없이 None을 넣었다가 다시 값을 넣는다."""
        s = self.make()
        s.on_run_snapshot(True, "job-1", 3)
        s.on_run_snapshot(False, "job-1", 3)
        s.on_run_snapshot(True, "job-1", 3)
        self.assertEqual(self.applied, [("job-1", 3), (None, None), ("job-1", 3)])
        self.assertEqual(s.summary()["run_source"]["counts"]["run_changes"], 0)

    def test_retry_memory_cleared_on_change(self):
        """실행이 바뀌면 번호별 실패 수·실패 위치 차단을 비운다."""
        retry = RetryConfig(max_failures_per_track=1, failed_point_block_s=100.0, failed_point_block_mm=5.0)
        s = self.make(retry=retry)
        s.on_run_snapshot(True, "job-1", 3)
        self.assertEqual(s.offer(goal_out("g1", "job-1", 3, track=9), frame_of(1), {}, 1), "sent")
        self.rt.finish("g1", "FAILED", "EMPTY_GRASP")
        self.assertEqual(s.offer(goal_out("g2", "job-1", 3, track=9), frame_of(2), {}, 2), "skip:track_retry_limit")
        s.on_run_snapshot(True, "job-2", 4)
        self.assertEqual(s.offer(goal_out("g3", "job-2", 4, track=9), frame_of(3), {}, 3), "sent")
        self.assertEqual(s.summary()["run_source"]["counts"]["retry_cleared"], 1)

    def test_old_run_result_not_counted(self):
        """옛 실행에서 보낸 목표의 실패 결과는 새 실행의 재시도 기억에 넣지 않는다. 결과 줄은 송신 때 값을 남긴다."""
        retry = RetryConfig(max_failures_per_track=1, failed_point_block_s=100.0, failed_point_block_mm=5.0)
        s = self.make(retry=retry)
        s.on_run_snapshot(True, "job-1", 3)
        self.assertEqual(s.offer(goal_out("g1", "job-1", 3, track=9), frame_of(1), {}, 1), "sent")
        s.on_run_snapshot(True, "job-2", 4)
        self.rt.finish("g1", "FAILED", "SORT_RUN_MISMATCH")
        self.assertEqual(s.offer(goal_out("g2", "job-2", 4, track=9), frame_of(2), {}, 2), "sent")
        result = [r for r in self.records() if r["event"] == "result:FAILED"]
        self.assertEqual([(r["sort_job_id"], r["run_epoch"]) for r in result], [("job-1", 3)])

    def test_mismatch_result_requests_refresh(self):
        """SORT_RUN_MISMATCH 결과면 즉시 재조회 1회. 다른 사유는 요청하지 않는다."""
        s = self.make()
        s.on_run_snapshot(True, "job-1", 3)
        s.offer(goal_out("g1", "job-1", 3), frame_of(1), {}, 1)
        self.rt.finish("g1", "FAILED", "EMPTY_GRASP")
        self.assertEqual(self.rt.refreshes, 0)
        s.offer(goal_out("g2", "job-1", 3), frame_of(2), {}, 2)
        self.rt.finish("g2", "FAILED", "SORT_RUN_MISMATCH")
        self.assertEqual(self.rt.refreshes, 1)
        self.assertEqual(s.summary()["run_source"]["counts"]["refresh_requests"], 1)

    def test_query_failures_counted_and_last_answer_kept(self):
        """조회 실패는 사유별로 세고, 마지막 답은 stale_s 안에서 그대로 쓴다."""
        s = self.make()
        s.on_run_snapshot(True, "job-1", 3)
        s.on_run_query_failed("timeout")
        s.on_run_query_failed("no_service")
        s.on_run_query_failed("timeout")
        self.assertEqual(s.offer(goal_out("g1", "job-1", 3), frame_of(1), {}, 1), "sent")
        rs = s.summary()["run_source"]
        self.assertEqual(rs["query_failures"], {"no_service": 1, "timeout": 2})
        self.assertEqual(rs["counts"]["query_failed"], 3)


class BStageApplyTest(unittest.TestCase):
    """B 단계에 넣은 값과 B `_dedupe`(b_stage.py:181-183, B 파일 무변)."""

    def stage(self):
        """계산 없이 `_dedupe`만 쓰는 BStage(생성자 설정 파일을 읽지 않음)."""
        st = BStage.__new__(BStage)
        st.sort_job_id, st.run_epoch, st._pending, st._pending_epoch = None, None, {}, None

        return st

    def test_epoch_change_clears_b_pending(self):
        """연결부가 epoch를 바꿔 넣으면 B가 다음 목표에서 보낸 부품 기억을 비운다."""
        from eyetoarm.vision.live_observation_link import _set_run

        st = self.stage()
        _set_run(st, "job-1", 3)
        g = {"kind": "PICK_PLACE", "part_track_id": 5, "goal_id": "a"}
        st._dedupe(g)
        again = {"kind": "PICK_PLACE", "part_track_id": 5, "goal_id": "b"}
        st._dedupe(again)
        self.assertEqual(again["reason"], "ALREADY_SENT")
        _set_run(st, "job-2", 4)
        fresh = {"kind": "PICK_PLACE", "part_track_id": 5, "goal_id": "c"}
        st._dedupe(fresh)
        self.assertEqual(fresh["kind"], "PICK_PLACE")


class FakeFuture:
    """rclpy Future 대역."""

    def __init__(self):
        """미완료."""
        self._done, self._result, self._exc, self.callbacks = False, None, None, []

    def done(self):
        """완료 여부."""
        return self._done

    def result(self):
        """결과(예외 주입 가능)."""
        if self._exc is not None:
            raise self._exc

        return self._result

    def add_done_callback(self, cb):
        """완료됐으면 바로, 아니면 나중에 부른다."""
        if self._done:
            cb(self)
        else:
            self.callbacks.append(cb)

    def finish(self, result=None, exc=None):
        """완료시키고 콜백을 부른다."""
        self._done, self._result, self._exc = True, result, exc

        for cb in self.callbacks:
            cb(self)


class FakeServiceClient:
    """rclpy Client 대역(GetSortRun)."""

    def __init__(self, ready=True):
        """준비 여부."""
        self.ready = ready
        self.calls = []
        self.removed = []

    def service_is_ready(self):
        """서비스 있음."""
        return self.ready

    def call_async(self, request):
        """새 Future."""
        f = FakeFuture()
        self.calls.append(f)

        return f

    def remove_pending_request(self, future):
        """버린 조회 기록."""
        self.removed.append(future)


class Listener:
    """송신기 대역(조회 결과 받을 곳)."""

    def __init__(self):
        """기록."""
        self.snaps, self.fails = [], []

    def on_run_snapshot(self, *snap):
        """답 기록."""
        self.snaps.append(snap)

    def on_run_query_failed(self, reason):
        """실패 기록."""
        self.fails.append(reason)


def load_run_pipeline():
    """scripts/run_pipeline.py 모듈."""
    spec = importlib.util.spec_from_file_location("run_pipeline_301_under_test", ROOT / "scripts" / "run_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


class RuntimePollTest(unittest.TestCase):
    """`_RosSortPartRuntime` 조회 부분(ROS 없이 조회 메서드만, 계획 §2.2)."""

    @classmethod
    def setUpClass(cls):
        """모듈 한 번 읽기."""
        cls.mod = load_run_pipeline()

    def runtime(self, ready=True, timeout_s=1.0):
        """생성자(rclpy) 없이 조회 상태만 채운 런타임."""
        rt = self.mod._RosSortPartRuntime.__new__(self.mod._RosSortPartRuntime)
        rt._run_cfg = RunSourceConfig(service=SERVICE, poll_s=0.5, call_timeout_s=timeout_s, stale_s=2.0)
        rt._run_client = FakeServiceClient(ready)
        rt._run_request = lambda: "req"
        rt._run_listener = None
        rt._run_last = None
        rt._run_lock = threading.Lock()
        rt._run_future = None
        rt._run_future_ns = 0
        rt._run_first = threading.Event()
        rt.run_service_seen = ready
        rt.run_first_answer = None

        return rt

    def test_answer_goes_to_listener_and_pending_skips(self):
        """조회 중이면 다음 주기는 건너뛰고, 답은 (active, job, epoch)로 받을 곳에 간다."""
        rt, lis = self.runtime(), Listener()
        rt.set_run_listener(lis)
        rt._poll_run()
        rt._poll_run()
        self.assertEqual(len(rt._run_client.calls), 1)
        rt._run_client.calls[0].finish(SimpleNamespace(active=True, sort_job_id="job-1", run_epoch=7))
        self.assertEqual(lis.snaps, [(True, "job-1", 7)])
        self.assertTrue(rt._run_first.is_set())
        rt._poll_run()
        self.assertEqual(len(rt._run_client.calls), 2)

    def test_timeout_discards_and_counts(self):
        """call_timeout_s를 넘긴 조회는 버리고 timeout으로 센 뒤 새로 묻는다."""
        rt, lis = self.runtime(timeout_s=1e-9), Listener()
        rt.set_run_listener(lis)
        rt._poll_run()
        first = rt._run_client.calls[0]
        rt._run_future_ns -= 10   # 시간이 지난 것으로
        rt._poll_run()
        self.assertEqual(rt._run_client.removed, [first])
        self.assertEqual(lis.fails, ["timeout"])
        self.assertEqual(len(rt._run_client.calls), 2)

    def test_no_service_and_error(self):
        """서비스가 없으면 no_service, 응답 예외·빈 답은 실패로 센다."""
        rt, lis = self.runtime(ready=False), Listener()
        rt.set_run_listener(lis)
        rt._poll_run()
        self.assertEqual((lis.fails, rt._run_client.calls), (["no_service"], []))
        rt._run_client.ready = True
        rt._poll_run()
        rt._run_client.calls[0].finish(exc=RuntimeError("x"))
        rt._poll_run()
        rt._run_client.calls[1].finish(None)
        self.assertEqual(lis.fails, ["no_service", "error:RuntimeError", "no_answer"])
        self.assertEqual(lis.snaps, [])

    def test_listener_gets_answer_received_before_it_was_set(self):
        """시작 때 받은 첫 답은 받을 곳을 붙일 때 바로 넘긴다. 시작 상태에 남는다."""
        rt, lis = self.runtime(), Listener()
        rt.request_run_refresh()
        rt._run_client.calls[0].finish(SimpleNamespace(active=False, sort_job_id="", run_epoch=2))
        rt.run_first_answer = rt._run_last
        rt.set_run_listener(lis)
        self.assertEqual(lis.snaps, [(False, "", 2)])
        self.assertEqual(rt.run_source_status(), {"service": SERVICE, "service_seen_at_start": True,
                                                  "first_answer": {"active": False, "sort_job_id": "", "run_epoch": 2}})


class ManagedRuntime(FakeRuntime):
    """가짜 관리 런타임: 받을 곳이 붙으면 정해 둔 답을 넘긴다."""

    def __init__(self, answer=None, **kw):
        """`answer`: (active, job, epoch) 또는 None(답 없음)."""
        super().__init__(**kw)
        self.answer = answer
        self.listener = None

    def set_run_listener(self, listener):
        """답 넘기기."""
        self.listener = listener

        if self.answer is not None:
            listener.on_run_snapshot(*self.answer)

    def run_source_status(self):
        """시작 상태."""
        first = None if self.answer is None else dict(zip(("active", "sort_job_id", "run_epoch"), self.answer))

        return {"service": SERVICE, "service_seen_at_start": self.answer is not None, "first_answer": first}


class CliManagedTest(CliBase):
    """진입점 경로(관리 설정): 시작 줄·B 단계 주입·송신 조건."""

    def test_active_run_numbers_reach_b_goals(self):
        """활성 답이면 B 목표(accept 기록)와 보낸 목표에 같은 job·epoch가 실린다."""
        rt = ManagedRuntime((True, "job-1", 7))
        code, err, out = self.run_sort(Scenario(TRACK_FRAMES), config=MANAGED_CONFIG, extra=MANAGED_EXTRA,
                                         factory=self.factory(rt))
        self.assertEqual(code, 0, err)
        self.assertIn(f"sort_part run_source=managed service={SERVICE} service_seen=true "
                      "first=active=true sort_job_id=job-1 run_epoch=7", err)
        self.assertTrue(rt.sent)
        self.assertEqual({(g["sort_job_id"], g["run_epoch"]) for g in rt.sent}, {("job-1", 7)})

        for r in self.accepts(out):
            self.assertEqual((r["goal"]["sort_job_id"], r["goal"]["run_epoch"]), ("job-1", 7))

        s = self.summary(out)["sort_part"]
        self.assertEqual(s["run_source"]["applied"], {"sort_job_id": "job-1", "run_epoch": 7})
        self.assertEqual(s["config"]["run_source"]["service"], SERVICE)

    def test_inactive_sends_nothing(self):
        """비활성 답이면 송신 0, 모두 skip:run_inactive."""
        rt = ManagedRuntime((False, "", 3))
        code, err, out = self.run_sort(Scenario(TRACK_FRAMES), config=MANAGED_CONFIG, extra=MANAGED_EXTRA,
                                         factory=self.factory(rt))
        self.assertEqual(code, 0, err)
        self.assertEqual(rt.sent, [])
        events = {r["event"] for r in self.sort_records(out) if r["event"].startswith("skip:run")}
        self.assertEqual(events, {"skip:run_inactive"})

    def test_no_service_warns_and_keeps_running(self):
        """시작 때 서비스가 없으면 경고하고 실행은 계속, 송신은 skip:run_unknown(Q2)."""
        rt = ManagedRuntime(None)
        code, err, out = self.run_sort(Scenario(TRACK_FRAMES), config=MANAGED_CONFIG, extra=MANAGED_EXTRA,
                                         factory=self.factory(rt))
        self.assertEqual(code, 0, err)
        self.assertIn(f"warning: {SERVICE} not seen at start; goals are not sent (skip:run_unknown)", err)
        self.assertEqual(rt.sent, [])
        self.assertTrue(any(r["event"] == "skip:run_unknown" for r in self.sort_records(out)))

    def test_managed_config_with_tracking_starts(self):
        """저장소 관리 설정 + --tracking: 짝 검사를 통과해 시작하고(종료 0), 시작 줄에 관리·확정 번호 송신이 나온다(#307)."""
        rt = ManagedRuntime((True, "job-1", 1))
        code, err, out = self.run_sort(Scenario(TRACK_FRAMES), config=MANAGED_CONFIG, extra=MANAGED_EXTRA,
                                       factory=self.factory(rt))
        self.assertEqual(code, 0, err)
        self.assertNotIn("require_track_id is required with --tracking", err)
        self.assertIn("require_track_id=true", err)
        self.assertIn(f"sort_part run_source=managed service={SERVICE}", err)
        self.assertIs(self.summary(out)["sort_part"]["require_track_id"], True)

    def test_plain_config_has_no_run_source_output(self):
        """관리 설정이 아니면 시작 줄·요약에 run_source가 없고 목표 실행 번호는 비어 있다(예전 동작)."""
        rt = FakeRuntime()
        code, err, out = self.run_sort(Scenario(4), factory=self.factory(rt))
        self.assertEqual(code, 0, err)
        self.assertNotIn("run_source", err)
        self.assertNotIn("run_source", self.summary(out)["sort_part"])
        self.assertTrue(rt.sent)
        self.assertEqual({(g["sort_job_id"], g["run_epoch"]) for g in rt.sent}, {(None, None)})


if __name__ == "__main__":
    unittest.main()
