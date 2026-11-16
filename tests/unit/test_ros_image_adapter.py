"""링크 카메라 실시간 소스(`LiveLinkSource`)의 ROS 비의존 부분 시험(L2)과 추적 AI 설정 시험(L0).

근거: 계획 `docs/superpowers/plans/2026-11-16-live-link-camera.md` r3 §3.2(필드 대응)·§3.6(QoS 진단)·§3.7(프로필)·§3.9(S1-01 설정)·
§4.1 L0·L2, 사용자 확정 2026-11-16(S1-01 YOLOv8n-seg batch16, imgsz 1280, conf 0.084).
가짜 메시지·가짜 변환기·가짜 수신 연결을 쓴다. rclpy·cv_bridge를 import하지 않는다(실제 통신은 L5 로컬 시나리오).
가중치 파일(Git 제외)은 읽지 않는다.
"""
import os
import subprocess
import sys
import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

from eyetoarm.calibration.profile import make_calibration_id
from eyetoarm.common.types import InputStatus, TimestampBase
from eyetoarm.inputs.base import FrameIdIssuer, FramePacket, InputStatusEvent
from eyetoarm.inputs.live_slot import BoundaryKind, NoticeKind
from eyetoarm.inputs.ros_image import (LiveLinkSource, LiveSourceConfig, RclpyTransport, owned_bgr, profile_info,
                                       qos_incompatibility, stamp_to_ns)
from eyetoarm.vision.ai_config import load_ai_config

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FRAME_ID = "link_cam_optical_frame"
W, H = 8, 6


def stamp(sec, nanosec=0):
    """ROS 시각 모양 객체."""
    return SimpleNamespace(sec=sec, nanosec=nanosec)


def image_msg(sec, nanosec=0, *, encoding="rgb8", frame_id=FRAME_ID, width=W, height=H, value=7):
    """가짜 Image 메시지. data는 (H, W, 3) 배열(가짜 변환기가 그대로 씀)."""
    data = np.full((height, width, 3), value, dtype=np.uint8)
    header = SimpleNamespace(stamp=stamp(sec, nanosec), frame_id=frame_id)

    return SimpleNamespace(header=header, width=width, height=height, encoding=encoding, data=data)


def info_msg(fx=100.0, width=W, height=H, frame_id=FRAME_ID):
    """가짜 CameraInfo 메시지(B from_camera_info가 읽는 속성)."""
    k = [fx, 0.0, width / 2, 0.0, fx, height / 2, 0.0, 0.0, 1.0]

    return SimpleNamespace(k=k, d=[0.0] * 5, width=width, height=height, header=SimpleNamespace(frame_id=frame_id))


def view_convert(message):
    """가짜 변환기: 메시지 버퍼를 공유하는 view를 돌려준다(bgr8처럼 복사 필요한 경우)."""
    return message.data[:, :, ::-1]


class FakeTransport:
    """가짜 수신 연결: start·close 호출을 적는다. 선택으로 예외를 낸다."""

    def __init__(self, start_error=None, close_error=None):
        """예외를 받는다."""
        self.start_error, self.close_error = start_error, close_error
        self.started, self.closed = [], []

    def start(self, source):
        """시작 기록."""
        self.started.append(source)

        if self.start_error is not None:
            raise self.start_error

    def close(self, limit_s):
        """종료 기록."""
        self.closed.append(limit_s)

        if self.close_error is not None:
            raise self.close_error


class CountingIssuer(FrameIdIssuer):
    """next_seq 호출 수를 세는 발급기."""

    def __init__(self):
        """run-live 실행."""
        super().__init__("run-live")
        self.calls = 0

    def next_seq(self):
        """호출을 센다."""
        self.calls += 1
        return super().next_seq()


class SourceBase(unittest.TestCase):
    """소스 준비."""

    def make(self, *, transport=None, convert=view_convert, **cfg):
        """가짜 연결로 소스를 만든다(대기 0.01 s)."""
        self.transport = transport or FakeTransport()
        self.issuer = CountingIssuer()
        self.clock_now = [5_000]
        config = LiveSourceConfig(frame_wait_s=0.01, **cfg)
        self.source = LiveLinkSource(source_id="gz_link", issuer=self.issuer, generation_provider=lambda: 0,
                                     config=config, transport_factory=lambda src: self.transport, convert=convert,
                                     clock_ns=lambda: self.clock_now[0])
        return self.source

    def apply_all(self):
        """대기열을 모두 반영(경계면 세대 +1)."""
        gen = 0
        items = self.source.poll_input()

        for item in items:
            if item.kind in (BoundaryKind.PROFILE_CHANGED, BoundaryKind.RECONNECT, BoundaryKind.SIM_TIME_REWIND):
                gen += 1

            self.source.input_applied(item, gen)

        return items


class PureHelpersTest(unittest.TestCase):
    """순수 도구."""

    def test_stamp_to_ns(self):
        """sec·nanosec을 정수 ns로 바꾸고, 음수·범위 밖은 ValueError."""
        self.assertEqual(stamp_to_ns(stamp(3, 5)), 3_000_000_005)

        for bad in (stamp(-1, 0), stamp(0, 1_000_000_000), stamp(0, -1)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                stamp_to_ns(bad)

    def test_owned_bgr_copies_shared_and_marks_readonly(self):
        """다른 버퍼의 view는 복사해 독립·연속·읽기 전용으로 만든다. 원본을 바꿔도 결과는 그대로다."""
        base = np.zeros((2, 3, 3), dtype=np.uint8)
        out = owned_bgr(base[:, :, ::-1])
        base[:] = 9

        self.assertTrue(out.flags.owndata and out.flags.c_contiguous)
        self.assertFalse(out.flags.writeable)
        self.assertEqual(int(out.max()), 0)

    def test_owned_bgr_rejects_bad_shape(self):
        """uint8 (H, W, 3)이 아니면 ValueError."""
        for bad in (np.zeros((2, 3), np.uint8), np.zeros((2, 3, 3), np.float32), np.zeros((2, 3, 4), np.uint8)):
            with self.subTest(shape=bad.shape, dtype=bad.dtype), self.assertRaises(ValueError):
                owned_bgr(bad)

    def test_qos_incompatibility(self):
        """reliable 요청 + best-effort 발행자만 비호환. best-effort 요청은 모두 호환."""
        self.assertIsNotNone(qos_incompatibility("reliable", ["reliable", "best_effort"]))
        self.assertIsNone(qos_incompatibility("reliable", ["reliable"]))
        self.assertIsNone(qos_incompatibility("best_effort", ["best_effort", "reliable"]))
        self.assertIsNone(qos_incompatibility("reliable", []))

    def test_profile_info_uses_b_calibration_id(self):
        """D3: calibration_id는 B 함수가 만든 값(해상도·K·dist 해시)과 같고, frame_id는 헤더 값."""
        p = profile_info(info_msg(fx=100.0))
        K = np.array([[100.0, 0, W / 2], [0, 100.0, H / 2], [0, 0, 1]])

        self.assertEqual(p.calibration_id, make_calibration_id(W, H, K, np.zeros(5)))
        self.assertEqual((p.width, p.height, p.frame_id), (W, H, FRAME_ID))

    def test_module_import_has_no_ros(self):
        """ros_image import만으로는 rclpy·cv_bridge·sensor_msgs를 불러오지 않는다(지연 import, 코드 규약 §8)."""
        code = ("import sys, eyetoarm.inputs.ros_image; "
                "print(sorted(m for m in ('rclpy', 'cv_bridge', 'sensor_msgs') if m in sys.modules))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                             env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")})

        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "[]")


class ReadTest(SourceBase):
    """영상 → FramePacket 조립(§3.2)."""

    def test_frame_fields(self):
        """SIM_TIME·내림 ms·원본 ns·수신 시각·광학 프레임·source_ref/index·읽기 전용 독립 영상·발급기 1회."""
        self.make()
        self.clock_now[0] = 123_456
        self.source.handle_image(image_msg(1, 999_999_999))
        frame = self.source.read()

        self.assertIsInstance(frame, FramePacket)
        self.assertEqual((frame.timestamp_base, frame.timestamp_ms, frame.ros_stamp_ns),
                         (TimestampBase.SIM_TIME, 1999, 1_999_999_999))
        self.assertEqual((frame.received_monotonic_ns, frame.optical_frame, frame.source_ref, frame.source_index),
                         (123_456, FRAME_ID, "/link_cam/image", 0))
        self.assertEqual((frame.width, frame.height, frame.crop, frame.calibration_id, frame.capture_generation),
                         (W, H, None, None, 0))
        self.assertEqual((frame.run_id, frame.source_id, frame.image_seq, self.issuer.calls), ("run-live", "gz_link", 0, 1))
        self.assertFalse(frame.image.flags.writeable)
        self.assertTrue(frame.image.flags.owndata)

    def test_calibration_after_profile(self):
        """프로필 경계 반영 뒤 영상은 calibration_id와 새 세대를 갖는다."""
        self.make()
        self.assertTrue(self.source.handle_camera_info(info_msg()))
        self.apply_all()
        self.source.handle_image(image_msg(1))
        frame = self.source.read()

        self.assertEqual((frame.calibration_id is not None, frame.capture_generation), (True, 1))

    def test_invalid_camera_info_no_boundary(self):
        """잘못된 K(fx 0)는 프로필을 만들지 않고 경계도 없다."""
        self.make()
        self.assertFalse(self.source.handle_camera_info(info_msg(fx=0.0)))
        self.assertEqual(self.source.poll_input(), [])

    def test_decode_failures_do_not_issue_seq(self):
        """지원하지 않는 인코딩·변환 예외·크기 불일치는 DECODE_FAILED이고 발급기를 부르지 않는다."""
        def broken(message):
            """변환 예외."""
            raise RuntimeError("bad buffer")

        cases = [({}, image_msg(1, encoding="mono8"), "unsupported encoding"),
                 ({"convert": broken}, image_msg(1), "convert failed"),
                 ({"convert": lambda m: np.zeros((2, 2, 3), np.uint8)}, image_msg(1), "image shape")]

        for kw, msg, text in cases:
            with self.subTest(text=text):
                self.make(**kw)
                self.source.handle_image(msg)
                event = self.source.read()

                self.assertIsInstance(event, InputStatusEvent)
                self.assertEqual(event.status, InputStatus.DECODE_FAILED)
                self.assertIn(text, event.detail)
                self.assertEqual(self.issuer.calls, 0)

    def test_invalid_header_not_received(self):
        """stamp가 잘못된 메시지는 받지 않는다(invalid_header)."""
        self.make()
        msg = image_msg(1)
        msg.header.stamp = stamp(-5)

        self.assertEqual(self.source.handle_image(msg), "invalid_header")
        self.assertIsNone(self.source.read())

    def test_no_frame_returns_none(self):
        """새 영상이 없으면 None(대기 상한 뒤)."""
        self.make()
        self.assertIsNone(self.source.read())

    def test_frame_current_follows_epoch(self):
        """frame_current: 조립한 프레임의 epoch가 현재면 True, 경계 뒤 False, 모르는 프레임 False."""
        self.make()
        self.source.handle_publishers([b"A"], ["reliable"])
        self.source.handle_image(image_msg(1))
        frame = self.source.read()
        self.assertTrue(self.source.frame_current(frame))

        self.source.handle_publishers([b"B"], ["reliable"])
        self.assertFalse(self.source.frame_current(frame))
        self.apply_all()
        self.assertFalse(self.source.frame_current(frame))
        self.assertFalse(self.source.frame_current(frame.__class__(**{**frame.__dict__, "image_seq": 99})))


class QosAndLifecycleTest(SourceBase):
    """QoS 진단·시작 실패·닫기."""

    def test_publisher_qos_mismatch_diagnosed(self):
        """reliable 요청인데 발행자가 best-effort면 SOURCE_UNAVAILABLE qos_incompatible 알림(조용한 무수신 아님)."""
        self.make()
        self.source.handle_publishers([b"A"], ["best_effort"])
        items = self.source.poll_input()

        self.assertEqual([(i.kind, i.status) for i in items],
                         [(NoticeKind.QOS_INCOMPATIBLE, InputStatus.SOURCE_UNAVAILABLE)])
        self.assertIn("BEST_EFFORT", items[0].detail)
        event = self.source.status_event(items[0])
        self.assertEqual((event.status, event.source_ref), (InputStatus.SOURCE_UNAVAILABLE, "/link_cam/image"))

    def test_best_effort_request_has_no_diagnosis(self):
        """best-effort 요청이면 best-effort 발행자도 호환이다."""
        self.make(reliability="best_effort")
        self.source.handle_publishers([b"A"], ["best_effort"])

        self.assertEqual(self.source.poll_input(), [])
        self.assertEqual(self.source.stats()["offered_reliability"], ["best_effort"])

    def test_start_failure_reported_once(self):
        """연결 시작 실패는 예외가 아니라 START_FAILED 알림 1회, 이후 read는 SOURCE_UNAVAILABLE closed. close는 연결을 닫지 않음."""
        transport = FakeTransport(start_error=ModuleNotFoundError("No module named 'rclpy'"))
        self.make(transport=transport)
        items = self.source.poll_input()

        self.assertEqual([(i.kind, i.status) for i in items], [(NoticeKind.START_FAILED, InputStatus.SOURCE_UNAVAILABLE)])
        self.assertIn("rclpy", items[0].detail)
        self.assertEqual(self.source.poll_input(), [])
        event = self.source.read()
        self.assertEqual((event.status, event.detail), (InputStatus.SOURCE_UNAVAILABLE, "closed"))
        self.source.close()
        self.assertEqual(transport.closed, [])

    def test_close_once_with_limit(self):
        """close는 연결을 한 번만 한도와 함께 닫고, 이후 read는 closed."""
        self.make(close_limit_s=1.5)
        self.source.close()
        self.source.close()

        self.assertEqual(self.transport.closed, [1.5])
        self.assertEqual(self.source.read().detail, "closed")

    def test_close_failure_propagates(self):
        """수신 스레드가 멈추지 않으면 close가 RuntimeError(실행기가 종료 1로 보고)."""
        self.make(transport=FakeTransport(close_error=RuntimeError("ros receive thread did not stop")))

        with self.assertRaises(RuntimeError):
            self.source.close()

    def test_callback_time_recorded(self):
        """영상 콜백 시간을 통계(최대·P95)에 남긴다."""
        self.make()
        self.source.handle_image(image_msg(1))
        stats = self.source.stats()

        self.assertIsNotNone(stats["callback_ns_max"])
        self.assertEqual(stats["received"], 1)

    def test_config_validation(self):
        """잘못된 신뢰성·빈 토픽·0 대기는 ValueError."""
        for over in ({"reliability": "fast"}, {"image_topic": ""}, {"frame_wait_s": 0}, {"close_limit_s": True}):
            with self.subTest(over=over), self.assertRaises(ValueError):
                LiveSourceConfig(**over)


class FakeExecutor:
    """가짜 executor: spin 동작(raise·return·block)과 shutdown 실패를 정한다."""

    def __init__(self, mode="block", shutdown_error=None, release=None):
        """mode: raise(즉시 예외)·return(즉시 반환)·block(shutdown까지 대기)·raise_on(release 이벤트 뒤 예외)."""
        self.mode, self.shutdown_error = mode, shutdown_error
        self.release = release or threading.Event()
        self.stop = threading.Event()
        self.shutdown_calls = 0

    def spin(self):
        """정한 대로 돈다."""
        if self.mode == "raise":
            raise RuntimeError("receiver crashed")

        if self.mode == "return":
            return

        if self.mode == "raise_on":
            self.release.wait(5.0)
            raise RuntimeError("receiver crashed while streaming")

        self.stop.wait(5.0)

    def shutdown(self, timeout_sec=None):
        """shutdown 호출을 세고, 실패를 정했으면 예외(이때 spin은 멈추지 않음)."""
        self.shutdown_calls += 1

        if self.shutdown_error is not None:
            raise self.shutdown_error

        self.stop.set()
        return True


class FakePart:
    """가짜 노드·컨텍스트: 정리 호출 수를 세고, 처음 n번 실패할 수 있다."""

    def __init__(self, fail_times=0):
        """실패 횟수를 받는다."""
        self.fail_times, self.calls = fail_times, 0

    def _call(self):
        """호출 기록·실패."""
        self.calls += 1

        if self.calls <= self.fail_times:
            raise RuntimeError("cleanup failed")

    def destroy_node(self):
        """노드 해제."""
        self._call()

    def try_shutdown(self):
        """컨텍스트 종료."""
        self._call()


class SpinTransport(RclpyTransport):
    """rclpy 없이 실제 `_spin`·`close` 경로를 쓰는 전송: start가 가짜 executor·노드·컨텍스트와 실제 스레드를 만든다."""

    def __init__(self, executor, node=None, context=None, owns=True):
        """가짜 부품을 받는다."""
        super().__init__(LiveSourceConfig(), context=None if owns else (context or FakePart()))
        self.fake_executor, self.fake_node = executor, node or FakePart()
        self.fake_context = context or FakePart()
        self._owns_context = owns

    def start(self, source):
        """가짜 부품으로 수신 스레드를 띄운다."""
        self._source = source
        self._executor, self._node = self.fake_executor, self.fake_node
        self._context = self.fake_context
        self._thread = threading.Thread(target=self._spin, name=RclpyTransport.THREAD_NAME, daemon=True)
        self._thread.start()


def wait_until(pred, limit_s=2.0):
    """조건이 참이 될 때까지 짧게 기다린다(상한)."""
    deadline = time.monotonic() + limit_s

    while time.monotonic() < deadline and not pred():
        time.sleep(0.005)

    return pred()


class ReceiverFailureTest(SourceBase):
    """검토 r1 F1: 수신기 비정상 종료를 치명 입력 사건으로 넘긴다."""

    def test_spin_exception_before_first_frame(self):
        """첫 영상 전 spin 예외 → RECEIVER_FAILED(SOURCE_UNAVAILABLE) 1회, 원인 기록, stats에 receiver_error."""
        self.make(transport=SpinTransport(FakeExecutor("raise")))
        self.assertTrue(wait_until(lambda: self.source.stats()["receiver_failed"]))
        items = self.source.poll_input()

        self.assertEqual([(i.kind, i.status) for i in items],
                         [(NoticeKind.RECEIVER_FAILED, InputStatus.SOURCE_UNAVAILABLE)])
        self.assertIn("receiver crashed", items[0].detail)
        self.assertIn("receiver crashed", self.source.stats()["receiver_error"])
        self.assertEqual(self.source.poll_input(), [])
        self.source.close()

    def test_spin_return_without_close_is_failure(self):
        """정리 요청 없이 spin이 그냥 반환해도 비정상 종료로 본다."""
        self.make(transport=SpinTransport(FakeExecutor("return")))
        self.assertTrue(wait_until(lambda: self.source.stats()["receiver_failed"]))

        self.assertIn("spin returned", self.source.poll_input()[0].detail)
        self.source.close()

    def test_normal_close_is_not_failure(self):
        """close로 멈춘 spin은 실패 알림을 만들지 않는다."""
        transport = SpinTransport(FakeExecutor("block"))
        self.make(transport=transport)
        self.source.close()

        self.assertEqual(self.source.poll_input(), [])
        self.assertFalse(self.source.stats()["receiver_failed"])
        self.assertIsNone(transport.spin_error)


class TransportCloseTest(unittest.TestCase):
    """검토 r1 F2: 단계별 실패를 모으고 뒤 단계를 계속, 주입 컨텍스트 보존, 재시도, 유한 대기."""

    def started(self, executor=None, node=None, context=None, owns=True):
        """시작한 가짜 전송."""
        t = SpinTransport(executor or FakeExecutor("block"), node=node, context=context, owns=owns)
        t.start(None)
        return t

    def test_node_failure_still_shuts_context_and_retries(self):
        """노드 해제 실패에도 소유 컨텍스트 종료는 한다. 다음 close는 노드만 다시 시도하고 컨텍스트는 다시 안 함."""
        node, ctx = FakePart(fail_times=1), FakePart()
        t = self.started(node=node, context=ctx)

        with self.assertRaisesRegex(RuntimeError, "node destroy failed"):
            t.close(0.5)

        self.assertEqual((node.calls, ctx.calls), (1, 1))
        t.close(0.5)
        self.assertEqual((node.calls, ctx.calls), (2, 1))

    def test_shutdown_failure_continues_and_reports_all(self):
        """executor shutdown 실패 + 그래서 멈추지 않은 스레드 + 컨텍스트 실패를 모두 보고하고, 노드 해제는 한다. 유한 시간."""
        ex = FakeExecutor("block", shutdown_error=RuntimeError("shutdown boom"))
        node, ctx = FakePart(), FakePart(fail_times=1)
        t = self.started(executor=ex, node=node, context=ctx)
        t0 = time.monotonic()

        with self.assertRaises(RuntimeError) as cm:
            t.close(0.05)

        msg = str(cm.exception)
        self.assertIn("executor shutdown failed", msg)
        self.assertIn("did not stop", msg)
        self.assertIn("context shutdown failed", msg)
        self.assertEqual(node.calls, 1)
        self.assertLess(time.monotonic() - t0, 2.0)
        ex.stop.set()

    def test_injected_context_not_shut_down(self):
        """주입 컨텍스트는 종료하지 않는다."""
        ctx = FakePart()
        t = self.started(context=ctx, owns=False)
        t.close(0.5)

        self.assertEqual(ctx.calls, 0)

    def test_normal_close_idempotent(self):
        """정상 close 뒤 다시 불러도 정리 호출이 늘지 않는다."""
        ex, node, ctx = FakeExecutor("block"), FakePart(), FakePart()
        t = self.started(executor=ex, node=node, context=ctx)
        t.close(0.5)
        t.close(0.5)

        self.assertEqual((ex.shutdown_calls, node.calls, ctx.calls), (1, 1, 1))

    def test_source_close_not_silent_after_failure(self):
        """소스 close: 연결 정리 실패면 예외, 다시 부르면 연결 close를 다시 시도(조용한 성공 없음), 성공 뒤에는 no-op."""
        transport = FakeTransport(close_error=RuntimeError("thread stuck"))
        source = LiveLinkSource(source_id="gz_link", issuer=FrameIdIssuer("r"), generation_provider=lambda: 0,
                                transport_factory=lambda s: transport)

        for _ in range(2):
            with self.assertRaises(RuntimeError):
                source.close()

        transport.close_error = None
        source.close()
        source.close()
        self.assertEqual(len(transport.closed), 3)
        self.assertEqual(source.read().detail, "closed")


class LiveAiConfigTest(unittest.TestCase):
    """L0: 추적 AI 설정 `config/vision/ai_s1_01_live.json`(사용자 확정 값)."""

    def test_confirmed_values(self):
        """S1-01 YOLOv8n-seg·계획 기록 sha·imgsz 1280·conf 0.084·bolt/nut, 재로드해도 같은 설정 ID."""
        path = os.path.join(ROOT, "config", "vision", "ai_s1_01_live.json")
        cfg = load_ai_config(path)

        self.assertEqual((cfg.model_family, cfg.imgsz, cfg.conf_threshold), ("yolov8n-seg", 1280, 0.084))
        self.assertEqual(cfg.weights_sha256, "68ed4d1139d4d19fa009e20b037e052924574d6429c61f20ec7bdd81dc26f2c6")
        self.assertEqual(cfg.weights_path, "outputs/s1_candidates/S1-01_yolov8n-seg/weights/best.pt")
        self.assertEqual((cfg.class_names, cfg.target_classes), ({0: "bolt", 1: "nut"}, ("bolt", "nut")))
        self.assertEqual(load_ai_config(path).config_sha256, cfg.config_sha256)


if __name__ == "__main__":
    unittest.main()
