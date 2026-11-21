"""#217 작업 좌표(W) 추적 시험: 추적기·입력 생성·추적 스레드(W1~W13, W15~W18, W10b).

근거: 계획 `_workspace/217_builder2_world_tracking_plan_r0.md`·`_r1.md`(reviewer PASS, 사용자 승인 2026-11-20).
- 합성 장면은 numpy로 만든다: 볼트 W 끝점(정답을 아는 입력)을 시험 카메라(`look_at`)로 픽셀에 투영해 AI 인스턴스 끝점으로 넣고,
  `world_inputs`가 B와 같은 투영으로 W에 되올린다. 정답 위치는 단언에서만 쓴다(추적기 입력 아님).
- #156 반례(픽셀 박스·점수만으로 신원이 옮겨 간 장면)는 같은 모양을 W(mm) 입력으로 옮긴 사례로 둔다(W5).
"""
from __future__ import annotations

import json
import math
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from eyetoarm.transforms.planar_transform import look_at, w_to_image
from eyetoarm.vision.live_world_tracking import (LagConfigError, LiveWorldConfig, LiveWorldTracking, load_world_tracking_config,
                                                 parse_live_world)
from eyetoarm.vision.live_observation_envelope import TfLookupResult
from eyetoarm.vision.observation_envelope import CaptureTf
from eyetoarm.vision.tracking import FrameKey, PartTracker, PartTrackIdIssuer, TrackMode, TrackState, TrackingContractError
from eyetoarm.vision.world_tracking import (WORLD_JUMP, BoltPlanes, WorldInput, WorldTracker, WorldTrackerConfigError,
                                            parse_world_tracker, world_inputs)
from test_ai_result import instance, seg_result

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "vision" / "world_tracker_test.json"
PLANE_Z = 0.105
K = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]])
DIST = np.zeros(5)
DT_MS = 66


def tcfg(**over):
    """시험 설정(예시 파일 값 + 덮어쓰기)."""
    doc = json.loads(CONFIG.read_text(encoding="utf-8"))["tracker"]
    doc.update(over)

    return parse_world_tracker(doc)


def key(i, generation=0, run="r"):
    """작업 키."""
    return FrameKey(run_id=run, source_id="s", image_seq=i, capture_generation=0, generation=generation, job_seq=i,
                    detector_config_id="d")


def cam(pos, target=(0.5, 0.0, PLANE_Z)):
    """카메라 위치(m)·바라보는 작업면 점 → T_W_C."""
    return look_at(pos, target)


def bolt_px(T, x_m, y_m, length_m=0.040, angle_deg=90.0):
    """W 볼트(중심·길이·축 각도) → 픽셀 끝점 (2, 2)."""
    a = math.radians(angle_deg)
    h = length_m / 2
    pts = np.array([[x_m - h * math.cos(a), y_m - h * math.sin(a), PLANE_Z], [x_m + h * math.cos(a), y_m + h * math.sin(a), PLANE_Z]])

    return w_to_image(pts, K, DIST, T)


def result_of(T, bolts):
    """볼트 목록 [(x, y, length, angle, conf, clipped)] → AI 결과(볼트 인스턴스, 끝점 있음)."""
    insts = []

    for i, (x, y, length, angle, conf, clipped) in enumerate(bolts):
        insts.append(instance(i, confidence=conf, clipped=clipped, axis_endpoints_px=bolt_px(T, x, y, length, angle)))

    return seg_result(insts, crop_wh=(1280, 720))


def winp(i, x_mm, y_mm, length=40.0, angle=90.0, conf=0.9, cls="bolt", clipped=False):
    """W 입력 직접 생성(끝점은 축 각도로)."""
    a = math.radians(angle)
    h = length / 2
    e0, e1 = (x_mm - h * math.cos(a), y_mm - h * math.sin(a)), (x_mm + h * math.cos(a), y_mm + h * math.sin(a))

    return WorldInput(index=i, class_name=cls, confidence=conf, candidate=True, untracked_reason=None, clipped=clipped,
                      center_mm=(x_mm, y_mm), center_raw_mm=(x_mm, y_mm), endpoints_mm=(e0, e1), length_mm=length,
                      angle_deg=angle % 180)


def ids_of(tf):
    """index → 번호."""
    return {a.instance_index: a.part_track_id for a in tf.assignments}


class ConfigTest(unittest.TestCase):
    """설정 검사·예시 파일."""

    def test_example_file(self):
        """예시 파일: test_only, conveyor_state null, 값 읽힘."""
        t, live, sha = load_world_tracking_config(str(CONFIG))
        self.assertEqual((t.gate_x_mm, t.max_belt_speed_mps, live.lag_limit_ms, live.lag_persist_n), (10.0, 3.0, 300.0, 5))
        self.assertEqual(len(sha), 64)

    def test_bad_values(self):
        """모르는 키·누락·0·bool·정수형 오류, live·lag 블록 오류, conveyor_state 비null 거부."""
        good = json.loads(CONFIG.read_text(encoding="utf-8"))

        for bad in ({**good["tracker"], "x": 1}, {k: v for k, v in good["tracker"].items() if k != "gate_x_mm"},
                    {**good["tracker"], "gate_x_mm": 0}, {**good["tracker"], "dup_merge": 1},
                    {**good["tracker"], "min_hits": 2.0}, {**good["tracker"], "velocity_alpha": 1.5}):
            with self.subTest(bad=bad), self.assertRaises(WorldTrackerConfigError):
                parse_world_tracker(bad)

        for bad in ({**good["live"], "x": 1}, {**good["live"], "lag": {**good["live"]["lag"], "pending_ratio_limit": 2}},
                    {**good["live"], "queue_capacity": 0}):
            with self.subTest(bad=bad), self.assertRaises(LagConfigError):
                parse_live_world(bad)

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "c.json"
            p.write_text(json.dumps({**good, "conveyor_state": {"topic": "/x"}}), encoding="utf-8")

            with self.assertRaises(LagConfigError):
                load_world_tracking_config(str(p))

            p.write_text(json.dumps({**good, "status": "approved"}), encoding="utf-8")

            with self.assertRaises(LagConfigError):
                load_world_tracking_config(str(p))


class InputsTest(unittest.TestCase):
    """W9: 입력 생성(투영·대상·사유)."""

    def test_projection_matches_truth_and_reasons(self):
        """볼트 끝점 → W 중점이 정답 위치와 0.01 mm 안, 너트·끝점 없음·비대상·잘림 처리."""
        T = cam((0.0, 0.0, 0.5))
        insts = [instance(0, axis_endpoints_px=bolt_px(T, 0.50, 0.02)),
                 instance(1, class_name="nut", axis_endpoints_px=None),
                 instance(2, axis_reason="AXIS_AMBIGUOUS"),
                 instance(3, rejected="LOW_CONFIDENCE"),
                 instance(4, clipped=True, axis_endpoints_px=bolt_px(T, 0.55, -0.03))]
        inp = world_inputs(seg_result(insts, crop_wh=(1280, 720)), K=K, dist=DIST, T_W_C=T, plane_z_m=PLANE_Z)
        self.assertEqual([(i.candidate, i.untracked_reason, i.clipped) for i in inp],
                         [(True, None, False), (False, "not_bolt", False), (False, "no_endpoints", False),
                          (False, "not_in_scope", False), (True, None, True)])
        self.assertAlmostEqual(inp[0].center_mm[0], 500.0, places=2)
        self.assertAlmostEqual(inp[0].center_mm[1], 20.0, places=2)
        self.assertAlmostEqual(inp[0].length_mm, 40.0, places=2)
        self.assertAlmostEqual(inp[0].angle_deg, 90.0, places=2)

    def test_view_bias_uses_corrected_and_keeps_raw(self):
        """시야 보정 켬: 추적 위치는 보정 뒤, 보정 전 위치도 남는다(둘이 다름)."""
        from eyetoarm.vision.ai_config import ViewBiasConfig

        T = cam((0.0, 0.0, 0.5))
        vb = ViewBiasConfig(model="C1", k_mm=1.39, plane_z_m=PLANE_Z, valid_for_model_id="yolov8n-seg@" + "0" * 16)
        inp = world_inputs(seg_result([instance(0, axis_endpoints_px=bolt_px(T, 0.5, 0.0))], crop_wh=(1280, 720)), K=K,
                           dist=DIST, T_W_C=T, plane_z_m=PLANE_Z, view_bias=vb, width=1280, height=720)[0]
        shift = math.dist(inp.center_mm, inp.center_raw_mm)
        self.assertTrue(0.5 < shift < 2.0, shift)


def bolt_px_at(T, x_m, y_m, length_m, z_m, angle_deg=90.0):
    """높이 z_m에 누운 W 볼트(중심·길이·축 각도) → 픽셀 끝점 (2, 2)(#250 P1 시험: 종류마다 축 높이가 다름)."""
    a = math.radians(angle_deg)
    h = length_m / 2
    pts = np.array([[x_m - h * math.cos(a), y_m - h * math.sin(a), z_m], [x_m + h * math.cos(a), y_m + h * math.sin(a), z_m]])

    return w_to_image(pts, K, DIST, T)


# B-12 예상 규격(#245 계획 r0 §2.2): BOLT_A M3×20 23 mm 축 높이 102 mm, BOLT_B M6×40 46 mm 105 mm. 마스크 기본 = B 「bolt」(BOLT_A 키)
SPLIT = BoltPlanes(default_z_m=0.102, specs=(("BOLT_A", 0.102, 0.023), ("BOLT_B", 0.105, 0.046)))


class BoltPlanesTest(unittest.TestCase):
    """#250 A-3 P1: W 투영 높이를 B 규격 종류별로 고른다."""

    def test_endpoint_bolts_use_their_spec_height(self):
        """M3×20(102 mm 높이)·M6×40(105 mm)이 같이 있으면 각자 자기 종류 높이로 올라가 W 중심이 정답과 0.01 mm 안, 길이 23·46 mm."""
        T = cam((0.0, 0.0, 0.5))
        insts = [instance(0, axis_endpoints_px=bolt_px_at(T, 0.50, 0.03, 0.023, 0.102)),
                 instance(1, axis_endpoints_px=bolt_px_at(T, 0.55, -0.04, 0.046, 0.105, 30.0))]
        inp = world_inputs(seg_result(insts, crop_wh=(1280, 720)), K=K, dist=DIST, T_W_C=T, plane_z_m=SPLIT)
        self.assertEqual([(i.plane_spec, i.plane_z_m) for i in inp], [("BOLT_A", 0.102), ("BOLT_B", 0.105)])
        np.testing.assert_allclose(inp[0].center_mm, (500.0, 30.0), atol=0.01)
        np.testing.assert_allclose(inp[1].center_mm, (550.0, -40.0), atol=0.01)
        self.assertAlmostEqual(inp[0].length_mm, 23.0, places=2)
        self.assertAlmostEqual(inp[1].length_mm, 46.0, places=2)

        # 대조: 높이 하나(105 mm)로 올리면 M3×20 중심이 1 mm 넘게 밀린다(P1이 고치는 차이)
        one = world_inputs(seg_result(insts[:1], crop_wh=(1280, 720)), K=K, dist=DIST, T_W_C=T, plane_z_m=0.105)[0]
        self.assertGreater(math.dist(one.center_mm, (500.0, 30.0)), 1.0)

    def test_uniform_planes_equal_single_height(self):
        """종류 높이가 모두 같으면(지금 B 규격) 결과가 실수 높이 하나와 같고 고른 종류 이름은 None이다."""
        T = cam((0.0, 0.0, 0.5))
        same = BoltPlanes(default_z_m=PLANE_Z, specs=(("BOLT_A", PLANE_Z, 0.018), ("BOLT_B", PLANE_Z, 0.046)))
        insts = [instance(0, axis_endpoints_px=bolt_px(T, 0.50, 0.02)),
                 instance(1, axis_endpoints_px=bolt_px(T, 0.56, -0.03, 0.012, 20.0))]
        a = world_inputs(seg_result(insts, crop_wh=(1280, 720)), K=K, dist=DIST, T_W_C=T, plane_z_m=same)
        b = world_inputs(seg_result(insts, crop_wh=(1280, 720)), K=K, dist=DIST, T_W_C=T, plane_z_m=PLANE_Z)
        self.assertEqual([(x.center_mm, x.length_mm, x.angle_deg, x.plane_z_m) for x in a],
                         [(x.center_mm, x.length_mm, x.angle_deg, x.plane_z_m) for x in b])
        self.assertEqual([x.plane_spec for x in a], [None, None])

    def test_mask_input_uses_default_height(self):
        """끝점 없는 볼트의 마스크 입력(#226)은 길이가 없어 기본 높이(B 「bolt」)로 올린다."""
        T = cam((0.0, 0.0, 0.5))
        poly = np.array([[600.0, 340.0], [640.0, 340.0], [640.0, 352.0], [600.0, 352.0]])
        inst = instance(0, axis_reason="AXIS_HEAD_AMBIGUOUS", polygon=poly, box_xyxy=(600.0, 340.0, 640.0, 352.0), area_px=480.0)
        inp = world_inputs(seg_result([inst], crop_wh=(1280, 720)), K=K, dist=DIST, T_W_C=T, plane_z_m=SPLIT,
                           mask_fallback=True)[0]
        self.assertEqual((inp.kind, inp.plane_z_m, inp.plane_spec), ("mask", 0.102, None))

    def test_invalid_planes_rejected(self):
        """종류 없음·이름 중복·빈 이름·0·음수·비유한·bool 값은 설정 오류."""
        for specs, default in (((), 0.1), ((("A", 0.1, 0.02), ("A", 0.1, 0.04)), 0.1), ((("", 0.1, 0.02),), 0.1),
                               ((("A", 0.0, 0.02),), 0.1), ((("A", 0.1, -0.02),), 0.1), ((("A", float("nan"), 0.02),), 0.1),
                               ((("A", 0.1, 0.02),), True)):
            with self.subTest(specs=specs, default=default), self.assertRaises(WorldTrackerConfigError):
                BoltPlanes(default_z_m=default, specs=specs)


class TrackerTest(unittest.TestCase):
    """W1~W7·W10b·W13."""

    def make(self, **over):
        """추적기(새 발급기)."""
        self.issuer = PartTrackIdIssuer()
        self.tr = WorldTracker(self.issuer, config=tcfg(**over))

        return self.tr

    def run_frames(self, frames, start=0, generation=0):
        """프레임 목록(각 W 입력 목록) → TrackFrame 목록."""
        return [self.tr.update(key(start + i, generation), (start + i) * DT_MS, f) for i, f in enumerate(frames)]

    def test_w1_camera_moves_world_fixed(self):
        """W1: 카메라(팔)만 크게 움직이고 볼트 W 위치는 고정 → 번호 유지, 점프 의심 없음(픽셀은 100 px 넘게 이동)."""
        self.make()
        out, px = [], []

        for i in range(12):   # 팔이 돌며 카메라 위치와 바라보는 점이 함께 바뀜(링크 카메라는 upper_arm_link에 붙음)
            T = cam((0.0 + 0.02 * i, -0.05 + 0.015 * i, 0.50 + 0.01 * i), (0.45 + 0.012 * i, -0.04 + 0.008 * i, PLANE_Z))
            res = result_of(T, [(0.50, 0.00, 0.040, 90.0, 0.9, False), (0.56, 0.03, 0.012, 20.0, 0.8, False)])
            px.append(res.instances[0].axis_endpoints_px.mean(axis=0))
            out.append(self.tr.update(key(i), i * DT_MS, world_inputs(res, K=K, dist=DIST, T_W_C=T, plane_z_m=PLANE_Z)))

        self.assertGreater(float(np.linalg.norm(px[-1] - px[0])), 100.0)
        ids = [ids_of(f) for f in out[2:]]
        self.assertTrue(all(d == ids[0] for d in ids) and None not in ids[0].values(), ids)
        self.assertNotIn(WORLD_JUMP, [f.event for f in out])

    def test_pixel_tracker_loses_same_motion(self):
        """대조(참고): 같은 장면을 픽셀 추적(M2)에 넣으면 번호가 유지되지 않는다(이 기능이 필요한 이유)."""
        from eyetoarm.vision.tracking import inputs_from_ai

        pt = PartTracker(PartTrackIdIssuer(), mode=TrackMode.HEURISTIC)
        seen = []

        for i in range(12):
            T = cam((0.0 + 0.02 * i, -0.05 + 0.015 * i, 0.50 + 0.01 * i), (0.45 + 0.012 * i, -0.04 + 0.008 * i, PLANE_Z))
            res = result_of(T, [(0.50, 0.00, 0.040, 90.0, 0.9, False), (0.56, 0.03, 0.012, 20.0, 0.8, False)])
            seen.append(ids_of(pt.update(key(i), i * DT_MS, inputs_from_ai(res))).get(0))

        self.assertGreater(len({s for s in seen[2:] if s is not None}) + seen[2:].count(None), 1)

    def test_w2_belt_motion_both_directions(self):
        """W2: 벨트 +Y·−Y 일정 속도(0.2 m/s, 2.2 m/s) → 번호 유지, 추정 Y 속도가 실제와 5 % 안."""
        for v_mps in (0.2, -0.2, 2.2):
            with self.subTest(v=v_mps):
                self.make()
                frames = [[winp(0, 500.0, -200.0 * math.copysign(1, v_mps) + v_mps * DT_MS * k)] for k in range(10)]
                out = self.run_frames(frames)
                ids = [ids_of(f)[0] for f in out]
                self.assertIsNone(ids[0])
                self.assertEqual(len({i for i in ids[2:]}), 1)
                self.assertIsNotNone(ids[-1])
                self.assertAlmostEqual(self.tr.last_belt_vy, v_mps, delta=abs(v_mps) * 0.05)

    def test_w3_duplicate_merge_and_parallel_pair(self):
        """W3: 같은 볼트 이중 검출(끝점 1 mm) → 합쳐 번호 1개, 나란히 붙은 두 볼트(6 mm) → 합치지 않고 번호 2개."""
        self.make()
        dup = [[winp(0, 500.0, 0.0, conf=0.9), winp(1, 501.0, 0.5, conf=0.6)] for _ in range(6)]
        out = self.run_frames(dup)
        last = out[-1].assignments
        self.assertEqual((last[0].state, last[1].reason), (TrackState.CONFIRMED, "duplicate_merged"))
        self.assertEqual(self.tr.stats()["dup_merged"], 6)

        self.make()
        pair = [[winp(0, 500.0, 0.0), winp(1, 506.0, 0.0)] for _ in range(6)]
        out = self.run_frames(pair)
        self.assertEqual(self.tr.stats()["dup_merged"], 0)
        self.assertTrue(all(a.reason != "duplicate_merged" for f in out for a in f.assignments))

    def test_w4_close_pair_no_swap(self):
        """W4: 가까운 두 볼트(같은 크기·방향)가 같이 움직임 → 번호가 서로 바뀌는 일 0(모호면 None)."""
        for gap in (6.0, 12.0, 25.0):
            with self.subTest(gap=gap):
                self.make()
                frames = [[winp(0, 500.0, 13.2 * k), winp(1, 500.0 + gap, 13.2 * k)] for k in range(12)]
                out = self.run_frames(frames)
                owner = {}

                for f in out:
                    for a in f.assignments:
                        if a.part_track_id is not None:
                            owner.setdefault(a.part_track_id, set()).add(a.instance_index)

                self.assertTrue(all(len(v) == 1 for v in owner.values()), owner)

    def test_w5_156_counterexamples_in_world(self):
        """W5: #156 반례 모양(늦은 등장 dx·점수 반전·예측 경로 가로채기·중복)을 W로 옮기면 신원 이동 0."""
        def late(dx, flip):
            frames = []

            for k in range(16):
                a_conf = 0.8 if (not flip or (k // 3) % 2 == 0) else 0.7
                row = [winp(0, 500.0, 4.0 * k, conf=a_conf)]

                if k >= 5:
                    row.append(winp(1, 500.0 + dx, 4.0 * k, conf=1.5 - a_conf))

                frames.append(row)

            return frames, {0: "A", 1: "B"}

        takeover = [[winp(0, 500.0, 4.0 * min(k, 8))] + ([winp(1, 500.0, 4.0 * k, conf=0.7)] if k >= 9 else [])
                    for k in range(16)]
        cases = {f"late_dx{dx}_{'flip' if fl else 'fixed'}": late(dx, fl) for dx in (4.0, 6.0, 8.0) for fl in (False, True)}
        cases["predicted_path_takeover"] = (takeover, {0: "A", 1: "B"})

        for name, (frames, who) in cases.items():
            with self.subTest(name=name):
                self.make()
                owner = {}

                for f in self.run_frames(frames):
                    for a in f.assignments:
                        if a.part_track_id is not None:
                            owner.setdefault(a.part_track_id, set()).add(who[a.instance_index])

                self.assertTrue(all(len(v) == 1 for v in owner.values()), (name, owner))

    def test_w7_world_jump(self):
        """W7: 확정 트랙 둘이 W에서 같은 방향으로 40 mm 함께 튐 → world_jump_suspected·전 종료(이전 번호 안 붙음)."""
        self.make()
        out = self.run_frames([[winp(0, 500.0, 0.0), winp(1, 560.0, 30.0)] for _ in range(4)])
        before = set(ids_of(out[-1]).values())
        jump = self.tr.update(key(4), 4 * DT_MS, [winp(0, 540.0, 0.0), winp(1, 600.0, 30.0)])
        self.assertEqual(jump.event, WORLD_JUMP)
        self.assertFalse(set(ids_of(jump).values()) & before)

    def test_w10b_max_belt_speed_is_gate_only(self):
        """W10b(Q4): max_belt_speed_mps만 바꾸면 예측 위치는 같고(게이트만 다름), 느린 벨트 장면 결과도 같다."""
        frames = [[winp(0, 500.0, 13.2 * k)] for k in range(10)]
        results = []

        for vmax in (0.5, 3.0, 9.0):
            self.make(max_belt_speed_mps=vmax)
            results.append([(a.part_track_id, a.state) for f in self.run_frames(frames) for a in f.assignments])
            t = self.tr._tracks[0]
            results[-1].append(self.tr._predict(t, 10 * DT_MS, 0.0, None))

        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])

    def test_belt_stop_signal_predicts_zero(self):
        """W11(자리): 정지 신호(belt_running=False)면 예측 속도 0·공통 속도 0."""
        self.make()
        self.run_frames([[winp(0, 500.0, 13.2 * k)] for k in range(5)])
        t = self.tr._tracks[0]
        self.assertEqual(self.tr._predict(t, 10 * DT_MS, 0.2, False), (t.x, t.y))
        self.assertEqual(self.tr._belt_vy(False), 0.0)

    def test_w13_reset_and_generation(self):
        """W13: reset·세대 변경 → 전 종료, 이전 번호 재사용 0(발급기 단조)."""
        self.make()
        a = self.run_frames([[winp(0, 500.0, 0.0)] for _ in range(4)])
        first = ids_of(a[-1])[0]
        self.tr.reset("run_end")
        b = self.run_frames([[winp(0, 500.0, 0.0)] for _ in range(4)], start=4)
        second = ids_of(b[-1])[0]
        c = self.run_frames([[winp(0, 500.0, 0.0)] for _ in range(4)], start=8, generation=1)
        third = ids_of(c[-1])[0]
        self.assertEqual(len({first, second, third}), 3)
        self.assertEqual(c[0].event, "scope_changed")
        self.assertEqual(self.tr.update(key(20, generation=0), 20 * DT_MS, [winp(0, 500.0, 0.0)]).updated, False)

    def test_w8_pixel_tracker_unchanged(self):
        """W8: PartTracker는 WORLD 모드를 받지 않는다(픽셀 동작 무변), WorldTracker 결과의 mode는 WORLD."""
        with self.assertRaises(TrackingContractError):
            PartTracker(PartTrackIdIssuer(), mode=TrackMode.WORLD)

        self.make()
        self.assertIs(self.run_frames([[winp(0, 500.0, 0.0)]])[0].mode, TrackMode.WORLD)


# ---------------------------------------------------------------------- 추적 스레드(W6·W12·W15~W18)

class FakeLookup:
    """촬영 stamp TF 대역: 상태·지연 주입, 호출 스레드 기록."""

    def __init__(self, status="ok", sleep_s=0.0, T=None):
        """상태와 지연."""
        self.status, self.sleep_s, self.T = status, sleep_s, T if T is not None else cam((0.0, 0.0, 0.5))
        self.threads = set()
        self.calls = 0

    def __call__(self, frame):
        """조회."""
        self.calls += 1
        self.threads.add(threading.current_thread().name)

        if self.sleep_s:
            time.sleep(self.sleep_s)

        if self.status != "ok":
            return TfLookupResult(None, self.status, 1.0, "fake")

        return TfLookupResult(CaptureTf("world", "cam", frame.ros_stamp_ns, self.T), "ok", 0.1)


def ctx_of(i, generation=0):
    """JobContext 대역."""
    f = SimpleNamespace(run_id="r", source_id="s", image_seq=i, capture_generation=0, ros_stamp_ns=1000 + i,
                        timestamp_ms=i * DT_MS, width=1280, height=720)

    return SimpleNamespace(frame=f, generation=generation, job_seq=i, detector_config_id="d")


def live_cfg(**over):
    """추적 스레드 설정."""
    base = dict(queue_capacity=8, handoff_wait_s=0.5, lag_limit_ms=300.0, window_n=20, pending_ratio_limit=0.5,
                lag_persist_n=5, lag_cooldown_ms=2000.0, conveyor_probe_timeout_s=5.0)
    base.update(over)

    return LiveWorldConfig(**base)


class LiveBase(unittest.TestCase):
    """추적 스레드 시험 도우미(시험 메서드 없음, #226 시험이 재사용)."""

    def make(self, lookup=None, **over):
        """스레드 시작(정리 등록)."""
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.lookup = lookup or FakeLookup()
        self.camera = SimpleNamespace(K=K, dist=DIST)
        self.lw = LiveWorldTracking("r", WorldTracker(PartTrackIdIssuer(), config=tcfg()), live_cfg(**over),
                                    tf_lookup=self.lookup, plane_z_m=PLANE_Z, out_dir=self.tmp.name)
        self.lw.start()
        self.addCleanup(self._close_quietly)

        return self.lw

    def _close_quietly(self):
        """정리: 비정상 종료 시험은 close가 RuntimeError를 내므로 여기서는 삼킨다(시험 본문이 이미 단언함)."""
        try:
            self.lw.close(2.0)
        except RuntimeError:
            pass

    def offer(self, i, bolts=((0.50, 0.0, 0.040, 90.0, 0.9, False),)):
        """현재 응답 하나."""
        return self.lw.offer(ctx_of(i), result_of(self.lookup.T, list(bolts)), self.camera)

    def records(self):
        """world_tracks.jsonl."""
        self.lw.close(2.0)
        return [json.loads(t) for t in (Path(self.tmp.name) / "world_tracks.jsonl").read_text().splitlines()]



class LiveThreadTest(LiveBase):
    """추적 스레드."""

    def test_handoff_and_confirmed_ids(self):
        """worker 넘김: 작업마다 결과를 받음, 3번째부터 확정 번호, TF 조회는 추적 스레드에서만."""
        lw = self.make()
        got = []

        for i in range(5):
            self.offer(i)
            got.append(lw.wait(ctx_of(i)))

        self.assertTrue(all(g is not None and not isinstance(g, str) for g in got))
        self.assertIsNotNone(ids_of(got[-1])[0])
        self.assertEqual(self.lookup.threads, {"eyetoarm-live-world-tracking"})
        recs = self.records()
        self.assertEqual([r["kind"] for r in recs], ["frame"] * 5 + ["closed"])   # #226 D3: 닫힐 때 종료 사건 1줄
        self.assertIn("center_mm", recs[-2]["tracks"][0])

    def test_w6_tf_missing_not_updated(self):
        """W6: TF 없음·늦음 → 갱신 안 함(사유 결과), 추적기 갱신 0(픽셀·최신 TF 대체 없음)."""
        for status in ("unavailable", "late", "stamp_mismatch"):
            with self.subTest(status=status):
                lw = self.make(FakeLookup(status=status))
                self.offer(0)
                self.assertEqual(lw.wait(ctx_of(0)), f"tf_{status}")
                self.assertEqual(lw._tracker.stats()["updates"], 0)
                lw.close(2.0)

    def test_w12_queue_full_and_handoff_timeout(self):
        """W12: 넣기는 기다리지 않음, 가득이면 새 것을 버림(dropped_full), 넘김 상한 초과는 None(track_pending)."""
        lw = self.make(FakeLookup(sleep_s=0.2), queue_capacity=2, handoff_wait_s=0.01)
        t0 = time.monotonic()
        results = [self.offer(i) for i in range(6)]
        self.assertLess(time.monotonic() - t0, 0.1)
        self.assertIn(False, results)
        self.assertIsNone(lw.wait(ctx_of(5)))
        s = lw.summary()
        self.assertGreaterEqual(s["counts"]["dropped_full"], 1)
        self.assertEqual(s["counts"]["handoff_pending"], 1)

    def test_w13_request_reset(self):
        """W13: 초기화 요청 → 다음 항목 전에 전 종료, 이후 번호는 새로 발급(재사용 0)."""
        lw = self.make()

        for i in range(4):
            self.offer(i)
            lw.wait(ctx_of(i))

        first = ids_of(lw.wait(ctx_of(3)))[0]
        lw.request_reset("run_end:x")

        for i in range(4, 8):
            self.offer(i)
            lw.wait(ctx_of(i))

        second = ids_of(lw.wait(ctx_of(7)))[0]
        self.assertNotEqual(first, second)
        self.assertEqual(lw._tracker.stats()["resets"].get("run_end:x"), 1)

    def test_w15_w16_lag_reset_and_cooldown(self):
        """W15·W16: 추적 스레드가 느리면 밀림 연속 뒤 초기화 1회(최신 1개만 남김), 쿨다운 안에서는 다시 하지 않음."""
        lw = self.make(FakeLookup(sleep_s=0.03), queue_capacity=40, lag_limit_ms=50.0, lag_persist_n=3,
                       lag_cooldown_ms=60_000.0)

        for i in range(30):
            self.offer(i)

        deadline = time.monotonic() + 5.0

        while time.monotonic() < deadline and lw.summary()["counts"]["processed"] < 4:
            time.sleep(0.01)

        time.sleep(0.3)
        s = lw.summary()
        self.assertEqual(s["counts"]["lag_resets"], 1)
        self.assertGreater(s["counts"]["stale_skipped"], 10)
        self.assertEqual(s["tracker"]["resets"].get("lag"), 1)
        recs = self.records()
        self.assertEqual(sum(1 for r in recs if r["kind"] == "lag_reset"), 1)
        c = lw.summary()["counts"]
        self.assertEqual(c["offered"], c["processed"] + c["dropped_full"] + c["stale_skipped"] + c["queue_left"])

    def test_w17_metrics_present(self):
        """W17: 요약 지연 지표가 모두 있고 계수 합이 맞다."""
        lw = self.make()

        for i in range(5):
            self.offer(i)
            lw.wait(ctx_of(i))

        lw.close(2.0)
        s = lw.summary()

        for k in ("queue_len_max", "queue_len_p95", "lag_ms_median", "lag_ms_p95", "lag_ms_max", "tf_wait_ms_median",
                  "handoff_pending_ratio", "dropped_full", "stale_skipped", "lag_resets", "lag_reset_log"):
            self.assertIn(k, s["lag"])

        c = s["counts"]
        self.assertEqual(c["offered"], c["processed"] + c["dropped_full"] + c["stale_skipped"] + c["queue_left"])

    def test_w18_normal_speed_no_reset(self):
        """W18: 정상 속도면 초기화 0, 넘김 대기 실패 0."""
        lw = self.make()

        for i in range(25):
            self.offer(i)
            lw.wait(ctx_of(i))

        s = lw.summary()
        self.assertEqual((s["counts"]["lag_resets"], s["lag"]["handoff_pending_ratio"]), (0, 0.0))

    def test_thread_crash_is_problem(self):
        """추적 스레드 비정상 종료 → problem·close RuntimeError(정리 확인 아님), 이후 넘김은 None."""
        lw = self.make()
        lw._tracker.update = None   # 다음 갱신에서 TypeError
        self.offer(0)
        lw._thread.join(2.0)
        self.assertIn("world tracking crashed", lw.problem())

        with self.assertRaises(RuntimeError):
            lw.close(1.0)


if __name__ == "__main__":
    unittest.main()
