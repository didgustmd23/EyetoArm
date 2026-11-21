"""입력·연결부·학습 결함 수정의 실패 우선 시험(무효 CameraInfo, 경계 훅, 건너뜀 기록, 재개, RSS).

ROS·Gazebo·torch를 쓰지 않는다. 실시간 입력은 가짜 연결·메시지, 학습은 가짜 자식 프로세스다.
"""
import gc
import hashlib
import json
import os
import tempfile
import time
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

from eyetoarm.inputs.live_slot import BoundaryKind, GenerationMirror, LiveFrameSlot, LiveSlotConfig, NoticeKind
from eyetoarm.common.types import InputStatus
from eyetoarm.vision.live_observation_link import LiveObservationLink, LiveObservationLinkConfig
from eyetoarm.vision.live_observation_state import LiveObservationConfig, LiveObservationState
from test_pipeline_runner_live import Live, processes, stop_after_jobs
from test_ros_image_adapter import SourceBase, image_msg, info_msg


# ---------------------------------------------------------------- 무효 CameraInfo
class InvalidCameraInfoTest(SourceBase):
    """해석할 수 없는 CameraInfo는 이전 프로필을 무효로 만들고 사유를 남긴다(프로필 변경 = 세대 전이)."""

    def test_invalid_after_valid_makes_profile_boundary(self):
        """유효 뒤 무효 CameraInfo → PROFILE_CHANGED(프로필 없음) 경계·사유, 이후 영상 calibration_id None, 통계 1."""
        self.make()
        self.assertTrue(self.source.handle_camera_info(info_msg()))
        self.apply_all()

        self.assertFalse(self.source.handle_camera_info(info_msg(fx=0.0)))
        items = self.apply_all()

        self.assertEqual([(i.kind, i.profile) for i in items], [(BoundaryKind.PROFILE_CHANGED, None)])
        self.assertIn("camera_info invalid", items[0].detail)

        self.source.handle_image(image_msg(1))
        frame = self.source.read()

        self.assertIsNone(frame.calibration_id)
        self.assertEqual(self.source.stats()["camera_info_invalid"], 1)

    def test_repeated_invalid_one_boundary_then_valid_restores(self):
        """무효가 반복되면 경계는 한 번이고 계수만 오른다. 다시 유효가 오면 새 PROFILE_CHANGED로 보정이 돌아온다."""
        self.make()
        self.source.handle_camera_info(info_msg())
        self.apply_all()

        for _ in range(3):
            self.source.handle_camera_info(info_msg(fx=0.0))

        self.assertEqual(len(self.apply_all()), 1)
        self.assertEqual(self.source.stats()["camera_info_invalid"], 3)

        self.source.handle_camera_info(info_msg())
        items = self.apply_all()

        self.assertEqual([i.kind for i in items], [BoundaryKind.PROFILE_CHANGED])
        self.assertIsNotNone(items[0].profile)

    def test_runner_applies_profile_none(self):
        """실행기: 프로필 없음 경계를 반영해 설정 프로필이 None이 되고(예외 없음) 기록에 사유가 남는다."""
        def feed(src, clock):
            """유효 프로필 + 영상 1."""
            src.handle_camera_info(info_msg())
            src.handle_image(image_msg(1))

        def during_first(live):
            """첫 검출 중 무효 CameraInfo와 다음 영상."""
            live.src.handle_camera_info(info_msg(fx=0.0))
            live.src.handle_image(image_msg(2))

        live = Live(feed=feed, hooks={0: during_first})
        live.should_stop = stop_after_jobs(live, 2)
        res, recs, err = live.run()

        self.assertNotIn("Traceback", err)
        changes = [r for r in recs if r["kind"] == "live_input" and r["input_kind"] == "profile_changed"]
        self.assertEqual(len(changes), 2)
        self.assertIn("camera_info invalid", changes[1]["detail"])
        self.assertTrue(changes[1]["generation_changed"])
        self.assertIsNone(live.pipeline.settings.profile)
        self.assertIsNone(processes(recs)[-1]["calibration_id"])


# ---------------------------------------------------------------- 경계 훅 예외
def _accepted_state():
    """수락 하나가 현재인 상태(토큰 준비)."""
    st = LiveObservationState(run_id="r", generation=0, calibration_id="c", config=LiveObservationConfig(8, 8))
    tok = st.token
    seq = st.accept(st.begin_compute(tok), tok, payload="old")

    return st, seq


class BoundaryHookFailureTest(unittest.TestCase):
    """경계 훅 예외는 계수만 하지 않고 치명 입력 알림이 되며, 연결부는 상태를 fail-stop한다."""

    def test_slot_hook_failure_becomes_fatal_notice(self):
        """훅이 예외를 내면 slot은 BOUNDARY_HOOK_FAILED 알림(입력 상태 SOURCE_UNAVAILABLE)을 대기열에 넣는다."""
        def hook(epoch, kind):
            """항상 실패하는 훅."""
            raise RuntimeError("injected hook failure")

        slot = LiveFrameSlot(LiveSlotConfig(), generation=GenerationMirror(0), clock_ns=lambda: 0, boundary_hook=hook)
        slot.on_image(object(), stamp_ns=10, frame_id="f", width=1, height=1)
        slot.on_image(object(), stamp_ns=5, frame_id="f", width=1, height=1)
        items = slot.poll_inputs()

        self.assertEqual([i.kind for i in items], [BoundaryKind.SIM_TIME_REWIND, NoticeKind.BOUNDARY_HOOK_FAILED])
        self.assertEqual(items[1].status, InputStatus.SOURCE_UNAVAILABLE)
        self.assertIn("injected hook failure", items[1].detail)
        self.assertEqual(slot.stats()["boundary_hook_errors"], 1)

    def test_link_hook_failure_fails_state(self):
        """연결부 훅 안의 상태 갱신이 실패하면 상태가 boundary_hook_error로 닫혀 이전 수락이 현재로 남지 않는다."""
        link = _link(16)
        self.addCleanup(lambda: link.end("t", limit_s=1.0))
        tok = link.state.token

        with mock.patch.object(link.state, "on_rx_boundary", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                link.boundary_hook(1, BoundaryKind.RECONNECT)

        self.assertEqual(link.state.failed, "boundary_hook_error")
        self.assertFalse(link.state.is_current(tok))

    def test_link_notice_fails_state(self):
        """실행기가 넘긴 BOUNDARY_HOOK_FAILED 알림을 연결부가 받으면 상태를 fail-stop한다."""
        link = _link(16)
        self.addCleanup(lambda: link.end("t", limit_s=1.0))
        item = SimpleNamespace(kind=NoticeKind.BOUNDARY_HOOK_FAILED, epoch=None, detail="x", profile=None)
        link.on_live_input(item, 0)

        self.assertEqual(link.state.failed, "boundary_hook_error")


# ---------------------------------------------------------------- 연결부 프로필 없음 경계
class LinkProfileClearedTest(unittest.TestCase):
    """연결부는 프로필 없음 경계에서 이전 보정 번호를 현재로 두지 않고, 유효 프로필이 오면 복귀한다."""

    def apply(self, link, epoch, profile):
        """경계 훅(수신) → 반영(제어 흐름) 순서로 PROFILE_CHANGED 하나를 넣는다."""
        link.boundary_hook(epoch, BoundaryKind.PROFILE_CHANGED)
        item = SimpleNamespace(kind=BoundaryKind.PROFILE_CHANGED, epoch=epoch, detail="t", profile=profile)
        link.on_live_input(item, epoch)

    def test_invalid_clears_then_valid_restores(self):
        """유효(cal-a) → 프로필 없음 → 유효(cal-b): 토큰 보정 번호가 cal-a → None → cal-b, 없음 동안 카메라 해제 표시."""
        from eyetoarm.inputs.live_slot import CameraProfileInfo

        link = _link(16)
        self.addCleanup(lambda: link.end("t", limit_s=1.0))

        self.apply(link, 1, CameraProfileInfo("cal-a", 640, 480, "f", camera=object()))
        self.assertEqual(link.state.token.calibration_id, "cal-a")

        self.apply(link, 2, None)
        self.assertIsNone(link.state.token.calibration_id)        # 이전 값(cal-a)이 남으면 실패
        self.assertIs(getattr(link, "camera_cleared", None), True)

        self.apply(link, 3, CameraProfileInfo("cal-b", 640, 480, "f", camera=object()))
        self.assertEqual(link.state.token.calibration_id, "cal-b")
        self.assertIs(getattr(link, "camera_cleared", None), False)

    def test_offer_refused_while_profile_cleared(self):
        """프로필 없음 동안은 보정 없는 프레임도 제출하지 않고 calibration_unavailable로 건너뛴다(토큰 준비 여부와 무관하게 먼저 검사)."""
        link = _link(16)
        self.addCleanup(lambda: link.end("t", limit_s=1.0))
        self.apply(link, 1, None)
        frame = SimpleNamespace(run_id="r", image_seq=1, ros_stamp_ns=1, calibration_id=None)
        outcome = SimpleNamespace(context=SimpleNamespace(frame=frame, job_seq=1, generation=1), observation=None)

        with mock.patch.object(type(link.state), "token",
                               new_callable=mock.PropertyMock,
                               return_value=SimpleNamespace(ready=True, calibration_id=None, generation=1)):
            self.assertEqual(link.offer(outcome, None, 1), "calibration_unavailable")


# ---------------------------------------------------------------- 연결부 건너뜀 기록
class _Tf:
    """TF 묶음 대역."""

    def lookup(self, *a):
        """항상 없음."""
        raise LookupError("none")

    def has_static(self, *a):
        """정적 TF 있음."""
        return True

    def close(self, limit):
        """닫힘."""
        return True


def _cfg(cap):
    """연결부 설정(건너뜀 큐 용량 cap)."""
    return LiveObservationLinkConfig(world_frame="world", tf_wait_limit_s=0.05, static_pairs=(), future_tolerance_ns=0,
                                     write_capacity=4, reserved_capacity=8, lag_warn_s=1.0, close_limit_s=1.0,
                                     skip_queue_capacity=cap)


def _link(cap, base=None):
    """임시 폴더의 연결부."""
    base = base or tempfile.mkdtemp()

    return LiveObservationLink("r", os.path.join(base, "out"), config=_cfg(cap), tf=_Tf(),
                               stage_factory=lambda c: None, render=None)


class SkipWriterTest(unittest.TestCase):
    """건너뜀 기록 스레드의 예기치 않은 예외도 드러내고 파일을 닫는다."""

    def test_unexpected_error_fails_state_and_closes(self):
        """json 직렬화 TypeError로 기록이 실패하면 skip_write_error·state 실패·파일 닫힘, 정리는 확인되지 않는다."""
        import numpy as np

        link = _link(16)
        link.start()
        frame = SimpleNamespace(run_id="r", image_seq=np.int64(3), ros_stamp_ns=1, calibration_id=None)
        outcome = SimpleNamespace(context=SimpleNamespace(frame=frame, job_seq=1, generation=0), observation=None)
        link.offer(outcome, "invalid_observation", 0)

        end = time.monotonic() + 2.0

        while link.skip_write_error is None and time.monotonic() < end:
            time.sleep(0.01)

        self.assertIn("TypeError", link.skip_write_error or "")
        self.assertEqual(link.state.failed, "skip_writer_error")

        rec = link.end("done", limit_s=1.0)

        self.assertTrue(link._skip_fh.closed)
        self.assertTrue(any(e.startswith("skips:") for e in rec.close_errors), rec.close_errors)
        self.assertFalse(link.cleanup_confirmed)


class ConstructorLeakTest(unittest.TestCase):
    """설정 검사 실패가 파일을 열어 둔 채 남기지 않는다."""

    def test_bad_capacity_opens_no_file(self):
        """skip_queue_capacity 0이면 ValueError이고 출력 폴더에 파일을 만들지 않으며 ResourceWarning이 없다."""
        base = tempfile.mkdtemp()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)

            with self.assertRaises(ValueError):
                _link(0, base)

            gc.collect()

        out = os.path.join(base, "out")
        self.assertEqual(sorted(os.listdir(out)) if os.path.isdir(out) else [], [])
        self.assertEqual([w for w in caught if issubclass(w.category, ResourceWarning)], [])

    def test_session_partial_failure_closes_tf(self):
        """LiveRunSession: TF 시작 뒤 디스패처 생성이 실패하면 시작한 TF 묶음을 닫고 예외를 다시 올린다."""
        from eyetoarm.vision.live_run_session import LiveRunSession

        closed = []

        class Tf(_Tf):
            """닫힘을 기록한다."""

            def close(self, limit):
                """기록."""
                closed.append(limit)
                return True

        def bad_dispatcher(state, tf):
            """디스패처 생성 실패."""
            raise RuntimeError("dispatcher build failed")

        with self.assertRaises(RuntimeError):
            LiveRunSession("r", generation=0, state_config=LiveObservationConfig(8, 8), make_slot=lambda hook: SimpleNamespace(),
                           make_tf=Tf, make_dispatcher=bad_dispatcher, frame_of=lambda r, g: None,
                           world_frame="world", static_pairs=(), wait_limit_s=0.05, future_tolerance_ns=0)

        self.assertEqual(len(closed), 1)


# ---------------------------------------------------------------- 재개 체크포인트 출처 확인
from test_training_runner import Fixture, _lines  # noqa: E402


class ResumeShaBeforeLoadTest(Fixture):
    """재개 last.pt는 기록된 기대 sha와 대조한 뒤에만 읽는다(pickle 기반 형식이라 출처 검증)."""

    def test_mismatch_refused_before_inspect(self):
        """last.pt sha가 직전 시도 기록(end_last_sha256)과 다르면 체크포인트를 열지 않고(inspect 미호출) 거부한다."""
        res = self.run_batch({"S1-01": "sleep"}, cfg=self.config(train_deadline_s=0.3))
        d = self.exp(res, "S1-01", "yolov8n-seg")
        os.makedirs(os.path.join(d, "weights"), exist_ok=True)

        with open(os.path.join(d, "weights", "last.pt"), "wb") as f:
            f.write(b"tampered")

        self.assertNotEqual(_lines(os.path.join(d, "attempts.jsonl"))[-1].get("end_last_sha256"),
                            hashlib.sha256(b"tampered").hexdigest())
        opened = []

        def inspect(path):
            """연 경로를 기록한다."""
            opened.append(path)
            return self.inspect(path)

        from eyetoarm.training.runner import run_resume
        from eyetoarm.training.runner import StopFlag

        r2 = run_resume(self.config(), repo_root=self.root, run_set="s1_t1", exp_id="S1-01", argv=["resume"],
                        child_factory=self.factory({}), inspect_fn=inspect, stop=StopFlag(install=False),
                        poll_s=0.05, versions={}, git={})

        self.assertEqual(r2.rc, 2)
        self.assertNotIn(os.path.join(d, "weights", "last.pt"), opened)
        self.assertTrue(any("before loading" in e for e in r2.errors), r2.errors)


class ResumeRefusalWordingTest(unittest.TestCase):
    """sha 기록 없는 재개는 불가하고, 거부 사유가 기대값 없음·불일치·읽기 실패를 구분한다."""

    def test_three_reasons_distinct(self):
        """기대값 없음 / 불일치 / 파일 없음이 서로 다른 문구이고 일치하면 None이다."""
        from eyetoarm.training.runner import _resume_sha_refusal

        d = tempfile.mkdtemp()
        path = os.path.join(d, "last.pt")

        with open(path, "wb") as f:
            f.write(b"ck")

        good = hashlib.sha256(b"ck").hexdigest()

        self.assertIn("no recorded end_last_sha256", _resume_sha_refusal(path, None))
        self.assertIn("!= last attempt record", _resume_sha_refusal(path, "0" * 64))
        self.assertIn("cannot read last.pt", _resume_sha_refusal(os.path.join(d, "missing.pt"), good))
        self.assertIsNone(_resume_sha_refusal(path, good))


# ---------------------------------------------------------------- 그룹 RSS 읽기 실패 기록
class GroupRssFailureTest(unittest.TestCase):
    """RSS 읽기 실패 pid를 조용히 빼지 않고 수를 남긴다."""

    def test_failed_pids_recorded(self):
        """한 pid의 rss 읽기가 실패하면 합에서 빠지고 failed 목록에 그 pid가 남는다."""
        from eyetoarm.training import runner

        class Proc:
            """pid 2만 읽기 실패."""

            def rss_kib(self, pid):
                """pid별 RSS."""
                if pid == 2:
                    raise OSError("gone")
                return 100

        failed = []

        with mock.patch.object(runner, "own_processes", return_value=([1, 2, 3], [])):
            total = runner.group_rss_kib("t", Proc(), 0, None, failed=failed)

        self.assertEqual((total, failed), (200, [2]))


if __name__ == "__main__":
    unittest.main()
