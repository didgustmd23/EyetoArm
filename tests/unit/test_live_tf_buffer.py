"""`transforms/live_tf_buffer` 단위시험: 생성·정리 순서, 정확 stamp 조회, 되감기 준비 판정.
가짜 rclpy·tf2를 쓰고 실제 ROS는 쓰지 않는다."""
import threading
import types
import unittest

import numpy as np

from eyetoarm.transforms.live_tf_buffer import LiveTfRuntime, TfRewindReadiness
from eyetoarm.vision.live_observation_state import LiveObservationConfig, LiveObservationState


class Log(list):
    """호출 순서 기록."""


def fake_modules(log, *, executor_error=None):
    """rclpy·tf2_ros 대역."""
    class Node:
        def __init__(self, name, context, parameter_overrides):
            log.append(("node", name, [(p.name, p.value) for p in parameter_overrides]))

        def destroy_node(self):
            log.append(("destroy_node",))

    class Param:
        def __init__(self, name, value):
            self.name, self.value = name, value

    class Duration:
        def __init__(self, seconds):
            self.seconds = seconds

    class Time:
        def __init__(self, nanoseconds=0):
            self.ns = nanoseconds

    class Buffer:
        def __init__(self, cache_time, node):
            log.append(("buffer", cache_time.seconds, node is not None))
            self.cleared = 0
            self.static = set()
            self.dynamic = {}

        def lookup_transform(self, parent, child, time, timeout):
            log.append(("lookup", parent, child, time.ns, timeout.seconds))
            if time.ns not in self.dynamic:
                raise RuntimeError("ExtrapolationException")
            stamp_ns = self.dynamic[time.ns]
            tr = types.SimpleNamespace(translation=types.SimpleNamespace(x=1.0, y=2.0, z=3.0),
                                       rotation=types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0))
            return types.SimpleNamespace(header=types.SimpleNamespace(
                stamp=types.SimpleNamespace(sec=stamp_ns // 10**9, nanosec=stamp_ns % 10**9)), transform=tr)

        def can_transform(self, parent, child, time):
            return (parent, child) in self.static

        def clear(self):
            self.cleared += 1

    class Listener:
        def __init__(self, buffer, node, spin_thread):
            log.append(("listener", spin_thread))

    class Executor:
        def __init__(self, context):
            self._stop = threading.Event()

        def add_node(self, node):
            log.append(("add_node",))

        def spin(self):
            if executor_error:
                raise executor_error
            self._stop.wait(5)

        def shutdown(self, timeout_sec):
            log.append(("shutdown", timeout_sec))
            self._stop.set()

    rclpy = types.SimpleNamespace(create_node=lambda name, context, parameter_overrides: Node(name, context,
                                                                                            parameter_overrides))
    return {"rclpy": rclpy, "Buffer": Buffer, "TransformListener": Listener, "SingleThreadedExecutor": Executor,
            "Time": Time, "Duration": Duration, "Parameter": Param}


class RuntimeTest(unittest.TestCase):
    def test_start_order_and_close_order(self):
        """노드 → Buffer → Listener → executor 순으로 만들고 executor → 스레드 → 노드 순으로 정리한다."""
        log = Log()
        rt = LiveTfRuntime(object(), modules=fake_modules(log), cache_s=10.0)
        rt.start()
        self.assertEqual(log[:4], [("node", "eyetoarm_b_tf", [("use_sim_time", True)]), ("buffer", 10.0, True),
                                   ("listener", False), ("add_node",)])
        rt.close(1.0)
        self.assertEqual([e[0] for e in log[4:]], ["shutdown", "destroy_node"])
        rt.close(1.0)   # 성공 단계는 다시 하지 않는다
        self.assertEqual([e[0] for e in log[4:]], ["shutdown", "destroy_node"])

    def test_lookup_exact_stamp_and_matrix(self):
        """정확 stamp로 조회하고 4×4와 반환 stamp를 준다. 없는 stamp는 예외 그대로."""
        log = Log()
        rt = LiveTfRuntime(object(), modules=fake_modules(log))
        rt.start()
        rt.buffer.dynamic[5_000_000_123] = 5_000_000_123
        T, got = rt.lookup("world", "cam", 5_000_000_123, 0.05)
        self.assertEqual(got, 5_000_000_123)
        np.testing.assert_allclose(T[:3, 3], [1.0, 2.0, 3.0])
        self.assertIn(("lookup", "world", "cam", 5_000_000_123, 0.05), log)
        with self.assertRaises(RuntimeError):
            rt.lookup("world", "cam", 1, 0.05)
        rt.close(1.0)

    def test_spin_error_recorded(self):
        """executor 예외는 스레드 밖으로 나가지 않고 spin_error로 남는다."""
        rt = LiveTfRuntime(object(), modules=fake_modules(Log(), executor_error=RuntimeError("x")))
        rt.start()
        rt._thread.join(1.0)
        self.assertEqual(rt.spin_error, "RuntimeError: x")
        rt.close(1.0)

    def test_no_ros_import_at_module_level(self):
        """새 모듈 import만으로 rclpy·tf2_ros를 불러오지 않는다(새 프로세스에서 확인)."""
        import subprocess
        import sys
        code = ("import sys, eyetoarm.transforms.live_tf_buffer, eyetoarm.vision.live_observation_dispatch, "
                "eyetoarm.vision.live_observation_events; print(any(m.split('.')[0] in ('rclpy', 'tf2_ros') for m in sys.modules))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "False")


class RewindReadinessTest(unittest.TestCase):
    """되감기 준비(시도 단위)."""
    STATIC = (("upper_arm_link", "link_cam_link"), ("link_cam_link", "link_cam_optical_frame"))

    def setup(self, static_ok=True, stamp=None):
        """되감기 경계 1을 연 상태."""
        st = LiveObservationState(run_id="r", generation=0, calibration_id="c",
                        config=LiveObservationConfig(write_capacity=4, reserved_capacity=16))
        st.on_rx_boundary(1, "sim_time_rewind")
        st.on_applied(1, 1, "c")
        self.static = set(self.STATIC) if static_ok else set()
        self.cleared = 0

        def clear():
            self.cleared += 1

        def lookup(parent, child, s, timeout):
            if stamp is None:
                raise RuntimeError("no tf")
            return np.eye(4), stamp

        rd = TfRewindReadiness(st, clear=clear, has_static=lambda p, c: (p, c) in self.static, lookup=lookup,
                               world_frame="world", static_pairs=self.STATIC, wait_limit_s=0.05,
                               future_tolerance_ns=500_000_000)
        return st, rd

    def frame(self, stamp=2_000):
        """되감기 뒤 첫 프레임."""
        return types.SimpleNamespace(ros_stamp_ns=stamp, optical_frame="link_cam_optical_frame")

    def test_g_static_missing_keeps_closed(self):
        """정적 사슬이 없으면 준비 실패, 닫힘 유지, B 호출 기회 없음."""
        st, rd = self.setup(static_ok=False, stamp=2_000)
        rd.begin(1)
        self.assertTrue(rd.try_ready(1, self.frame()).startswith("static_missing"))
        self.assertFalse(st.token.ready)
        self.assertEqual(self.cleared, 1)

    def test_g_dynamic_missing_or_mismatch_keeps_closed(self):
        """경계 뒤 정확 stamp TF가 없거나 다른 stamp면 닫힘 유지."""
        st, rd = self.setup(stamp=None)
        rd.begin(1)
        self.assertTrue(rd.try_ready(1, self.frame()).startswith("dynamic_unavailable"))
        st2, rd2 = self.setup(stamp=1_999)
        rd2.begin(1)
        self.assertEqual(rd2.try_ready(1, self.frame()), "stamp_mismatch")
        self.assertFalse(st.token.ready or st2.token.ready)

    def test_g_ready_opens_only_same_boundary(self):
        """조건 충족 시 그 경계만 연다. 이전 경계 번호의 준비는 ready_ignored."""
        st, rd = self.setup(stamp=2_000)
        self.assertEqual(rd.try_ready(1, self.frame()), "not_begun")     # 시도 없이 열 수 없다
        self.assertEqual(rd.begin(0), "not_begun")                       # 이전 경계 시도 불가
        self.assertEqual(rd.try_ready(0, self.frame()), "not_begun")
        self.assertFalse(st.token.ready)
        self.assertEqual(rd.begin(1), "begun")
        self.assertEqual(rd.try_ready(1, self.frame()), "ready")
        self.assertTrue(st.token.ready)

    def test_g_future_tf_after_ready_recloses(self):
        """준비 뒤 미래 stamp TF(이전 실행 잔여 의심)는 닫고, 허용치 안이면 그대로."""
        st, rd = self.setup(stamp=2_000)
        rd.begin(1)
        rd.try_ready(1, self.frame())
        self.assertFalse(rd.observe_tf_stamp(1, 2_100_000_000, 2_000_000_000))
        self.assertTrue(st.token.ready)
        self.assertTrue(rd.observe_tf_stamp(1, 9_000_000_000, 2_000_000_000))
        self.assertFalse(st.token.ready)

    def test_f1_anomaly_during_lookup_does_not_reopen(self):
        """조회 도중 미래 stamp 이상이 생기면 정확 stamp가 와도 열지 않고 begin부터 다시 해야 한다."""
        st, rd = self.setup(stamp=2_000)
        hold = {"anomaly": True}

        def lookup(parent, child, s, timeout):
            if hold["anomaly"]:
                rd.observe_tf_stamp(1, 9_000_000_000, 2_000_000_000)
            return np.eye(4), s

        rd._lookup = lookup
        self.assertEqual(rd.begin(1), "begun")
        cleared_before = self.cleared
        self.assertEqual(rd.try_ready(1, self.frame()), "ready_ignored")
        self.assertFalse(st.token.ready)
        self.assertEqual(rd.try_ready(1, self.frame()), "not_begun")     # 시도가 무효라 새 begin 필요
        hold["anomaly"] = False
        self.assertEqual(rd.begin(1), "begun")
        self.assertEqual(self.cleared, cleared_before + 1)               # 새 시도는 버퍼를 다시 비움
        self.assertEqual(rd.try_ready(1, self.frame()), "ready")
        self.assertTrue(st.token.ready)

    def test_f1_concurrent_anomaly_threads(self):
        """다른 스레드의 이상이 조회 중 끼어도 그 시도로는 열리지 않는다."""
        st, rd = self.setup(stamp=2_000)
        in_lookup, go = threading.Event(), threading.Event()

        def lookup(parent, child, s, timeout):
            in_lookup.set()
            go.wait(2.0)
            return np.eye(4), s

        rd._lookup = lookup
        rd.begin(1)
        out = {}
        t = threading.Thread(target=lambda: out.setdefault("r", rd.try_ready(1, self.frame())))
        t.start()
        self.assertTrue(in_lookup.wait(2.0))
        rd.observe_tf_stamp(1, 9_000_000_000, 2_000_000_000)
        go.set()
        t.join(2.0)
        self.assertEqual(out["r"], "ready_ignored")
        self.assertFalse(st.token.ready)


if __name__ == "__main__":
    unittest.main()
