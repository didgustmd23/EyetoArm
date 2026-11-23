"""retry 블록이 없을 때의 기본 재시도 제한 시험: 실패 직후 같은 번호 폭주 막기, 간격 뒤·다른 번호·성공 뒤 송신, 기록·요약."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from eyetoarm.vision.live_sort_part import DEFAULT_FAILURE_GAP_S, RetryConfig, SortPartConfig, retry_policy
from eyetoarm.vision.pipeline_records import PipelineLog
from test_live_tracking_link import SenderRetryTest, out_of

GAP_NS = int(DEFAULT_FAILURE_GAP_S * 1e9)
FRAME = SimpleNamespace(image_seq=1, ros_stamp_ns=1)


class DefaultGapTest(unittest.TestCase):
    """기본 제한(블록 없음). 송신기·가짜 런타임은 기존 재시도 시험과 같은 것을 쓴다(그 시험 메서드는 상속하지 않음)."""

    make = SenderRetryTest.make
    recs = SenderRetryTest.recs

    def test_failure_gap_blocks_same_track(self):
        """실패 결과 뒤 1 s(SIM, 촬영 시각 기준) 안의 같은 번호는 skip:failure_gap(+release), 다른 번호는 보낸다."""
        s = self.make()
        self.assertEqual(s.offer(out_of("g0", 7, stamp=10 ** 9), FRAME, {}, 0), "sent")
        self.rt.finish("g0", "FAILED")
        self.assertEqual(s.offer(out_of("g1", 7, stamp=10 ** 9 + 25_000_000), FRAME, {}, 1), "skip:failure_gap")
        self.assertEqual(s.offer(out_of("g2", 8, stamp=10 ** 9 + 25_000_000), FRAME, {}, 2), "sent")
        self.assertEqual(self.target.release.call_args_list, [mock.call(7)])
        summary = s.summary()
        self.assertEqual(summary["retry"], "off")   # 기존 요약 키는 그대로
        self.assertEqual(summary["retry_default"], {"source": "default", "failure_gap_s": DEFAULT_FAILURE_GAP_S,
                                                    "failures_recorded": 1, "blocked": 1})
        events = [r["event"] for r in self.recs()]
        self.assertIn("release:skip:failure_gap", events)

    def test_after_gap_and_success(self):
        """간격(1 s) 뒤에 찍힌 목표는 다시 보내고, 성공하면 그 번호 기억을 지운다."""
        s = self.make()
        s.offer(out_of("g0", 7, stamp=10 ** 9), FRAME, {}, 0)
        self.rt.finish("g0", "FAILED")
        self.assertEqual(s.offer(out_of("g1", 7, stamp=10 ** 9 + GAP_NS), FRAME, {}, 1), "sent")
        self.rt.finish("g1", "SUCCEEDED")
        self.assertEqual(s.offer(out_of("g2", 7, stamp=10 ** 9 + GAP_NS + 1), FRAME, {}, 2), "sent")
        self.recs()

    def test_without_track_blocks_untracked(self):
        """번호 없는 목표가 실패하면 그 뒤 1 s 안의 번호 없는 목표를 막는다(release 없음)."""
        s = self.make(tracking=False)
        s.offer(out_of("g0", None, stamp=10 ** 9), FRAME, {}, 0)
        self.rt.finish("g0", "FAILED")
        self.assertEqual(s.offer(out_of("g1", None, stamp=10 ** 9 + 1), FRAME, {}, 1), "skip:failure_gap")
        self.assertEqual(self.target.release.call_args_list, [])
        self.recs()

    def test_config_block_replaces_default(self):
        """retry 블록이 있으면 기본 제한을 쓰지 않는다(1회 실패 뒤 곧바로 같은 번호 송신, 요약에 retry_default 없음)."""
        s = self.make(RetryConfig(max_failures_per_track=2, failed_point_block_s=0.001, failed_point_block_mm=1.0))
        s.offer(out_of("g0", 7, stamp=10 ** 9), FRAME, {}, 0)
        self.rt.finish("g0", "FAILED")
        self.assertEqual(s.offer(out_of("g1", 7, stamp=10 ** 9 + 25_000_000, pos=(500.0, 0.0, 105.0)), FRAME, {}, 1), "sent")
        self.assertNotIn("retry_default", s.summary())
        self.recs()


class PolicyRecordTest(unittest.TestCase):
    """정책 기록."""

    def test_policy_and_run_meta(self):
        """정책 사전(기본·설정)과 run_meta 기록: 주어졌을 때만 sort_part_retry 키."""
        base = dict(server_wait_s=1.0, result_wait_s=1.0, close_limit_s=1.0, spin_timeout_s=0.01)
        self.assertEqual(retry_policy(SortPartConfig(**base)), {"source": "default", "failure_gap_s": 1.0})
        r = RetryConfig(max_failures_per_track=2, failed_point_block_s=10.0, failed_point_block_mm=15.0)
        self.assertEqual(retry_policy(SortPartConfig(**base, retry=r))["source"], "config")

        with self.assertRaises(ValueError):
            PipelineLog(mock.Mock(error=None), sort_part_retry="off")


if __name__ == "__main__":
    unittest.main()
