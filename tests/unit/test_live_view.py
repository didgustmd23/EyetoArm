"""실시간 검출 확인 창 순수 부분 — `DisplayBridge`(최신 1칸·버전·경계)와 `render_view`/`draw_detections` 시험.

창·ROS·GPU 없이 합성 영상과 시험용 표시 결과만 쓴다.
"""
import gc
import threading
import time
import unittest
import weakref

import numpy as np

from eyetoarm.vision.live_view import (CLASS_COLORS_BGR, MASK_ALPHA, STATE_COLORS_BGR, DisplayBridge, ViewState,
                                       draw_detections, render_view)
from eyetoarm.vision.pipeline_types import JobContext, JobOutcome, JobStatus
from test_ai_config import ai_config
from test_ai_pipeline import ai_observation
from test_ai_result import instance, seg_result
from test_pipeline import frame, settings

H, W = 200, 240
POLY = [[100, 100], [140, 100], [140, 140], [100, 140]]


def image(value=40):
    """합성 원본(BGR uint8, 읽기 전용으로 만들어 쓰기를 잡아냄)."""
    img = np.full((H, W, 3), value, np.uint8)
    img.setflags(write=False)

    return img


def big_instance(index=0, **over):
    """원본 좌표 (100..140, 100..140) 사각형 인스턴스."""
    fields = {"box_xyxy": (100.0, 100.0, 140.0, 140.0), "polygon": POLY, "area_px": 1600.0}
    fields.update(over)

    return instance(index, **fields)


def job(image_seq=0, generation=0, job_seq=0, img=None):
    """AI 설정·합성 프레임을 가진 작업 스냅샷."""
    f = frame(image_seq, image=img if img is not None else image(), width=W, height=H)

    return JobContext(frame=f, generation=generation, job_seq=job_seq, settings=settings(detector_config=ai_config()),
                      started_monotonic_ns=1, reprocess=False)


def completed(ctx, instances=(), **over):
    """최초 현재 적용 완료 응답(표시용 invalid 관측 + AI 표시 결과)."""
    fields = {"context": ctx, "status": JobStatus.COMPLETED, "observation": ai_observation(ctx),
              "display_result": seg_result(instances, crop_wh=(W, H)), "error": None, "invalidation": None,
              "applied": True, "duplicate": False}
    fields.update(over)

    return JobOutcome(**fields)


def failed(ctx, status=JobStatus.FAILED, error="INFERENCE_ERROR: boom"):
    """실패·중단·폐기 응답."""
    return JobOutcome(context=ctx, status=status, observation=None, display_result=None, error=error,
                      invalidation=None, applied=False, duplicate=False)


def pixel(img, x, y):
    """(x, y) 화소 BGR 튜플."""
    return tuple(int(v) for v in img[y, x])


class DrawTest(unittest.TestCase):
    """그리기: 원본 불변·좌표 대응·무검출·비대상 미표시·초록 없음."""

    def test_draw_instance_on_copy(self):
        """박스·윤곽 채우기를 복사본에 그린다. 원본(읽기 전용)은 그대로, 다각형 안은 클래스 색과 섞이고 멀리 떨어진 곳은 원본 값."""
        src = image()
        before = src.copy()
        out, drawn = draw_detections(src, seg_result([big_instance()], crop_wh=(W, H)))

        self.assertEqual(drawn, 1)
        np.testing.assert_array_equal(src, before)
        self.assertIsNot(out, src)
        color = CLASS_COLORS_BGR[0]
        expect = tuple(int(round(MASK_ALPHA * c + (1 - MASK_ALPHA) * 40)) for c in color)
        got = pixel(out, 120, 125)
        self.assertTrue(all(abs(a - b) <= 1 for a, b in zip(got, expect)), (got, expect))
        self.assertEqual(pixel(out, 220, 190), (40, 40, 40))
        self.assertEqual(pixel(out, 140, 125), color)   # 박스 오른쪽 변(라벨과 겹치지 않는 곳)

    def test_roi_offset_not_added_twice(self):
        """ROI가 있어도 인스턴스 좌표(이미 원본 좌표)를 그대로 그린다. ROI만큼 밀린 위치에는 그리지 않는다."""
        result = seg_result([big_instance()], roi=(80, 80, 120, 100), crop_wh=(120, 100))
        out, _ = draw_detections(image(), result)

        self.assertNotEqual(pixel(out, 120, 125), (40, 40, 40))
        self.assertEqual(pixel(out, 200, 190), (40, 40, 40))   # (120+80, 125+80) 근처는 원본 그대로

    def test_no_detection_and_out_of_scope_not_drawn(self):
        """무검출과 비대상·점수 미달 인스턴스는 그리지 않는다(표시 영상 = 원본, 그린 수 0)."""
        src = image()

        for instances in ((), (big_instance(rejected="LOW_CONFIDENCE"),),
                          (big_instance(rejected="NON_TARGET_CLASS", class_index=1, class_name="nut"),)):
            with self.subTest(instances=instances):
                out, drawn = draw_detections(src, seg_result(instances, crop_wh=(W, H)))
                self.assertEqual(drawn, 0)
                np.testing.assert_array_equal(out, src)

    def test_no_green_status_or_class_colors(self):
        """작업 가능 신호로 오해할 초록 우세 색(G가 B·R보다 큼)을 쓰지 않는다."""
        for b, g, r in list(CLASS_COLORS_BGR) + list(STATE_COLORS_BGR.values()):
            self.assertFalse(g > b and g > r, (b, g, r))


class RenderTest(unittest.TestCase):
    """창 영상: 같은 프레임 쌍·마지막 영상 표시·원본 불변."""

    def render(self, bridge, max_width=4000):
        """그린 창 영상과 아래쪽 영상 쌍 부분(머리 정보 줄 제외)."""
        out = render_view(bridge.snapshot(), max_width=max_width)

        return out, out[out.shape[0] - H:]

    def test_pair_is_same_frame(self):
        """좌 원본·우 오버레이가 같은 프레임이다. 좌는 원본 화소, 우의 같은 위치는 검출 색. 원본 배열은 바뀌지 않는다."""
        bridge = DisplayBridge()
        ctx = job(7, img=image(40))
        bridge.on_outcome(completed(ctx, [big_instance()]), current=True)
        _, pair = self.render(bridge)

        self.assertEqual(pixel(pair, 120, 125), (40, 40, 40))
        self.assertNotEqual(pixel(pair, W + 4 + 120, 125), (40, 40, 40))
        self.assertEqual(int(ctx.frame.image.max()), 40)
        snap = bridge.snapshot()
        self.assertIs(snap.item.frame, ctx.frame)
        self.assertEqual(snap.item.frame.image_seq, 7)

    def test_scaled_pair_keeps_correspondence(self):
        """폭을 줄여도 좌우가 같은 비율로 줄어 같은 위치에 대응한다."""
        bridge = DisplayBridge()
        bridge.on_outcome(completed(job(), [big_instance()]), current=True)
        out = render_view(bridge.snapshot(), max_width=(2 * W + 4) // 2)
        half = out[out.shape[0] - H // 2:]

        self.assertEqual(half.shape[1], (2 * W + 4) // 2)
        self.assertNotEqual(pixel(half, (W + 4 + 120) // 2, 62), (40, 40, 40))
        self.assertEqual(pixel(half, 60, 62), (40, 40, 40))

    def test_no_detection_new_frame_has_no_old_boxes(self):
        """검출 있던 프레임 다음 무검출 새 프레임: 우측에 이전 박스가 남지 않고 `detections: 0`."""
        bridge = DisplayBridge()
        bridge.on_outcome(completed(job(1, job_seq=0), [big_instance()]), current=True)
        bridge.on_outcome(completed(job(2, job_seq=1, img=image(50)), ()), current=True)
        out, pair = self.render(bridge)

        self.assertEqual(pixel(pair, W + 4 + 120, 125), (50, 50, 50))
        self.assertEqual(bridge.snapshot().item.frame.image_seq, 2)
        self.assertIn("detections: 0", " ".join(_lines(bridge)))

    def test_last_image_dimmed_without_detection(self):
        """경계 뒤에는 결과 없이 마지막 원본만 어둡게 보이고, 우측에도 검출 색이 없다."""
        bridge = DisplayBridge()
        bridge.on_outcome(completed(job(img=image(100)), [big_instance()]), current=True)
        bridge.on_input("frame_timeout", "no image for 1.0 s", 0, False)
        _, pair = self.render(bridge)

        self.assertEqual(pixel(pair, W + 4 + 120, 125), (40, 40, 40))   # 100 * 0.4
        self.assertIn("LAST IMAGE", " ".join(_lines(bridge)))

    def test_info_line_fields(self):
        """정보 줄: run/source/image_seq/gen, 시각·SIM_TIME, 검출 ms, conf, 표시용(valid=False). FPS·지연 없음."""
        bridge = DisplayBridge()
        bridge.on_outcome(completed(job(3, generation=2, job_seq=5), [big_instance()]), current=True)
        text = " ".join(_lines(bridge))

        for part in ("image_seq 3", "gen 2", "job 5", "SIM_TIME", "detect 7.0 ms", "conf>=", "detections: 1",
                     "valid=False", "DISPLAY ONLY"):
            self.assertIn(part, text)

        for banned in ("FPS", "fps", "latency", "OK", "READY"):
            self.assertNotIn(banned, text)


def _lines(bridge):
    """정보 줄 문자열(내부 함수)."""
    from eyetoarm.vision.live_view import _info_lines

    return _info_lines(bridge.snapshot())


class BridgeTest(unittest.TestCase):
    """전달 칸: 최초 현재 결과만, 경계·실패 즉시 폐기, 버전, 정지 뒤 무시."""

    def test_only_current_first_result_shown(self):
        """duplicate는 무시(버전 불변), current=False(늦은·applied=False)·폐기·실패는 결과를 지운다."""
        bridge = DisplayBridge()
        ctx = job()
        bridge.on_outcome(completed(ctx, [big_instance()]), current=True)
        v = bridge.version

        bridge.on_outcome(completed(ctx, duplicate=True, applied=False), current=False)
        self.assertEqual(bridge.version, v)
        self.assertIsNotNone(bridge.snapshot().item)

        cases = [(completed(job(1, job_seq=1), applied=False), ViewState.STALE),
                 (failed(job(2, job_seq=2), JobStatus.DISCARDED_GENERATION, None), ViewState.STALE),
                 (failed(job(3, job_seq=3), JobStatus.ABORTED, "input_boundary: frame receive epoch"), ViewState.STALE),
                 (failed(job(4, job_seq=4), JobStatus.ABORTED, "detect_timeout"), ViewState.ERROR),
                 (failed(job(5, job_seq=5)), ViewState.ERROR)]

        for outcome, state in cases:
            with self.subTest(status=outcome.status, error=outcome.error):
                bridge.on_outcome(completed(ctx, [big_instance()]), current=True)
                bridge.on_outcome(outcome, current=False)
                snap = bridge.snapshot()
                self.assertIsNone(snap.item)
                self.assertEqual(snap.state, state)
                self.assertIs(snap.last_frame, ctx.frame)

    def test_valid_false_kept(self):
        """표시 칸의 관측 사유는 응답 그대로(valid=False 관측을 바꾸지 않음)."""
        bridge = DisplayBridge()
        outcome = completed(job(), [big_instance()])
        bridge.on_outcome(outcome, current=True)

        self.assertFalse(outcome.observation.valid)
        self.assertEqual(bridge.snapshot().item.reason, outcome.observation.reason)

    def test_input_boundary_clears_and_bumps_version(self):
        """경계마다 결과를 지우고 버전을 올린다. 꺼낸 상태는 경계 뒤 is_current=False(그리기 직전 확인)."""
        for kind in ("frame_timeout", "reconnect", "sim_time_rewind", "profile_changed", "receiver_failed"):
            with self.subTest(kind=kind):
                bridge = DisplayBridge()
                bridge.on_outcome(completed(job(), [big_instance()]), current=True)
                snap = bridge.snapshot()
                bridge.on_input(kind, "x", 1, kind != "frame_timeout")

                self.assertFalse(bridge.is_current(snap.version))
                self.assertIsNone(bridge.snapshot().item)
                self.assertIn(kind, bridge.snapshot().message)

    def test_info_only_notice_keeps_result(self):
        """epoch를 바꾸지 않는 정보 알림(되감기 필터 해제)은 결과를 지우지 않고 사건만 남긴다."""
        bridge = DisplayBridge()
        bridge.on_outcome(completed(job(), [big_instance()]), current=True)
        bridge.on_input("rewind_filter_released", "x", 0, False)

        self.assertIsNotNone(bridge.snapshot().item)
        self.assertIn("rewind_filter_released", bridge.snapshot().events[-1])

    def test_stop_then_no_new_detection(self):
        """정지 요청 뒤 결과를 지우고, 이후 도착한 현재 결과·경계를 칸에 넣지 않는다."""
        bridge = DisplayBridge()
        bridge.on_outcome(completed(job(), [big_instance()]), current=True)
        bridge.request_stop()
        bridge.on_outcome(completed(job(1, job_seq=1), [big_instance()]), current=True)
        bridge.on_input("reconnect", "x", 1, True)
        snap = bridge.snapshot()

        self.assertEqual((snap.state, snap.item), (ViewState.STOPPING, None))
        bridge.finish(0, "stop_requested")
        self.assertEqual(bridge.snapshot().state, ViewState.ENDED)

    def test_finish_failure_state(self):
        """runner가 실패로 끝나면 ERROR, 정상이면 ENDED."""
        bridge = DisplayBridge()
        bridge.finish(1, "receiver_failed")

        self.assertEqual(bridge.snapshot().state, ViewState.ERROR)
        self.assertIn("receiver_failed", bridge.snapshot().message)

    def test_events_bounded(self):
        """최근 사건은 유한 개만 보관한다(무제한 큐 없음)."""
        bridge = DisplayBridge()

        for i in range(100):
            bridge.on_input("frame_timeout", str(i), i, False)

        self.assertLessEqual(len(bridge.snapshot().events), 4)

    def test_latest_one_releases_previous(self):
        """새 결과가 이전 결과·프레임 참조를 놓는다(칸 1개)."""
        bridge = DisplayBridge()
        first = completed(job(1, job_seq=0), [big_instance()])
        ref_result, ref_frame = weakref.ref(first.display_result), weakref.ref(first.context.frame)
        bridge.on_outcome(first, current=True)
        del first
        bridge.on_outcome(completed(job(2, job_seq=1), ()), current=True)
        gc.collect()

        self.assertIsNone(ref_result())
        self.assertIsNone(ref_frame())

    def test_slow_consumer_does_not_block_producer(self):
        """창이 꺼낸 상태를 오래 붙잡고 그려도(잠금 밖) producer는 기다리지 않고, 다음 꺼냄은 최신 1개다."""
        bridge = DisplayBridge()
        holding = threading.Event()
        release = threading.Event()

        def slow_consumer():
            """꺼낸 뒤 그리기가 느린 창."""
            snap = bridge.snapshot()
            holding.set()
            release.wait(5)
            del snap

        t = threading.Thread(target=slow_consumer, daemon=True)
        t.start()
        holding.wait(5)
        started = time.monotonic()

        for i in range(200):
            bridge.on_outcome(completed(job(i, job_seq=i), ()), current=True)

        elapsed = time.monotonic() - started
        release.set()
        t.join(5)

        self.assertLess(elapsed, 2.0)
        self.assertEqual(bridge.snapshot().item.frame.image_seq, 199)
        self.assertEqual(bridge.shown, 200)


if __name__ == "__main__":
    unittest.main()
