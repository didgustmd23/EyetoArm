"""W 추적 입력 보완(W+M)·리셋 기록·진단 조회 상한 시험.

합성 장면은 정답을 아는 볼트 W 끝점을 시험 카메라로 투영해 만들고, 정답 위치는 단언에서만 쓴다.
"""
from __future__ import annotations

import io
import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from eyetoarm.vision.tracked_observation import NO_SELECTION, confirmed_track_id
from eyetoarm.vision.tracking import PartTrackIdIssuer, TrackState, TrackingInputError
from eyetoarm.vision.world_tracking import WorldTracker, containment, mask_of, parse_world_tracker, world_inputs
from test_ai_result import instance, seg_result
from test_world_tracking import CONFIG, DIST, DT_MS, K, PLANE_Z, LiveBase, bolt_px, cam, ctx_of, key, tcfg

T = cam((0.0, 0.0, 0.5))


def polygon_of(T_W_C, x_m, y_m, length_m=0.040, angle_deg=90.0, half_w_px=6.0):
    """W 볼트 → 픽셀 끝점 둘레 띠 다각형(4점)."""
    e = bolt_px(T_W_C, x_m, y_m, length_m, angle_deg)
    d = e[1] - e[0]
    n = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9) * half_w_px

    return np.array([e[0] + n, e[1] + n, e[1] - n, e[0] - n], dtype=np.float32)


def bolt_inst(i, x_m, y_m, *, endpoints=True, conf=0.9, length_m=0.040, angle=90.0, T_W_C=T):
    """끝점 있음/없음(사유 AXIS_HEAD_AMBIGUOUS) 볼트 인스턴스, 다각형 포함."""
    poly = polygon_of(T_W_C, x_m, y_m, length_m, angle)
    kw = (dict(axis_endpoints_px=bolt_px(T_W_C, x_m, y_m, length_m, angle)) if endpoints
          else dict(axis_reason="AXIS_HEAD_AMBIGUOUS"))
    x0, y0 = poly.min(axis=0)
    x1, y1 = poly.max(axis=0)

    return instance(i, confidence=conf, polygon=poly, box_xyxy=(float(x0), float(y0), float(x1), float(y1)),
                    area_px=float(abs(np.cross(poly[1] - poly[0], poly[3] - poly[0]))), **kw)


def inputs(insts, *, fallback=True, T_W_C=T):
    """제품 world_inputs(시야 보정 없음)."""
    return world_inputs(seg_result(insts, crop_wh=(1280, 720)), K=K, dist=DIST, T_W_C=T_W_C, plane_z_m=PLANE_Z,
                        mask_fallback=fallback)


def tracker(**over):
    """추적기(설정 파일 값 + 덮어쓰기)."""
    return WorldTracker(PartTrackIdIssuer(), config=tcfg(**over))


class MaskInputTest(unittest.TestCase):
    """입력 생성·짝짓기."""

    def test_mask_input_built_only_for_no_endpoint_bolts(self):
        """끝점 없는 볼트 → kind=mask(길이·각도 없음), 끝점 볼트 → endpoint, 끔이면 no_endpoints."""
        ins = [bolt_inst(0, 0.50, 0.00), bolt_inst(1, 0.55, 0.05, endpoints=False)]
        on = inputs(ins)
        self.assertEqual([(i.kind, i.candidate) for i in on], [("endpoint", True), ("mask", True)])
        self.assertIsNotNone(on[0].mask)
        self.assertIsNone(on[1].length_mm)
        self.assertLess(math.dist(on[1].center_mm, (550.0, 50.0)), 2.0)
        off = inputs(ins, fallback=False)
        self.assertEqual([(i.kind, i.candidate, i.untracked_reason) for i in off],
                         [("endpoint", True, None), ("none", False, "no_endpoints")])
        self.assertIsNone(off[0].mask)

    def test_mask_only_bolt_tracks_and_confirms(self):
        """끝점이 한 번도 없는 볼트도 마스크 입력으로 트랙을 이어 3 hit에 확정, 번호 1개."""
        tr = tracker()
        out = [tr.update(key(i), i * DT_MS, inputs([bolt_inst(0, 0.50, 0.00, endpoints=False)])) for i in range(6)]
        ids = [f.assignments[0].part_track_id for f in out]
        self.assertEqual(ids[:2], [None, None])
        self.assertEqual(len(set(ids[2:])), 1)
        self.assertIsNotNone(ids[-1])
        self.assertEqual(out[-1].assignments[0].reason, "mask_input")

    def test_mixed_endpoint_and_mask_keep_same_id(self):
        """같은 볼트가 끝점·마스크로 번갈아 들어와도 같은 번호(마스크 hit는 길이 무변)."""
        tr = tracker()
        pattern = [True, False, False, True, False, True, True, False, False, False]
        out = [tr.update(key(i), i * DT_MS, inputs([bolt_inst(0, 0.50, 0.0132 * i * 0.0, endpoints=e)]))
               for i, e in enumerate(pattern)]
        ids = [f.assignments[0].part_track_id for f in out[2:]]
        self.assertEqual(len(set(ids)), 1)
        self.assertIsNotNone(ids[0])
        self.assertAlmostEqual(tr._tracks[0].length, 40.0, delta=0.5)   # 마지막 끝점 hit 길이 유지

    def test_belt_motion_with_sparse_endpoints(self):
        """벨트 이동(0.2 m/s) + 끝점 20 %만 → 마스크 입력으로 번호 유지, 속도 추정."""
        tr = tracker()
        rng = np.random.default_rng(0)
        out = []

        for i in range(12):
            y = -0.10 + 0.2 * DT_MS / 1000 * i
            out.append(tr.update(key(i), i * DT_MS, inputs([bolt_inst(0, 0.50, y, endpoints=bool(rng.random() < 0.2))])))

        ids = [f.assignments[0].part_track_id for f in out[2:]]
        self.assertEqual(len(set(ids)), 1)
        self.assertAlmostEqual(tr.last_belt_vy, 0.2, delta=0.02)

    def test_off_rejects_mask_candidates(self):
        """mask_fallback 끔 추적기에 마스크 후보를 넣으면 입력 오류(설정 불일치를 숨기지 않음)."""
        tr = tracker(mask_fallback=False)

        with self.assertRaises(TrackingInputError):
            tr.update(key(0), 0, inputs([bolt_inst(0, 0.5, 0.0, endpoints=False)]))


class OffEqualsW217Test(unittest.TestCase):
    """끔이면 끝점 전용 추적과 같다."""

    def test_off_same_as_endpoint_only_inputs(self):
        """끔 결과 = 켬 추적기에 끝점 입력만 넣은 결과이고, 끝점 볼트끼리 합치기도 같다."""
        frames = []

        for i in range(10):
            ins = [bolt_inst(0, 0.50, 0.0), bolt_inst(1, 0.5005, 0.0005, conf=0.6),  # 이중 검출(0.7 mm)
                   bolt_inst(2, 0.56, 0.06, endpoints=(i % 3 == 0))]
            frames.append(ins)

        off = tracker(mask_fallback=False)
        on = tracker()
        a = [off.update(key(i), i * DT_MS, inputs(f, fallback=False)) for i, f in enumerate(frames)]
        b = [on.update(key(i), i * DT_MS, [x for x in inputs(f, fallback=False)]) for i, f in enumerate(frames)]
        self.assertEqual([[(x.part_track_id, x.state, x.reason) for x in f.assignments] for f in a],
                         [[(x.part_track_id, x.state, x.reason) for x in f.assignments] for f in b])
        self.assertEqual(off.stats()["dup_merged"], 10)


class DuplicateWithMaskTest(unittest.TestCase):
    """마스크가 낀 이중 검출 합치기."""

    def test_endpoint_kept_over_mask_duplicate(self):
        """같은 볼트의 끝점 입력 + 끝점 없는 중복(점수 더 높음) → 끝점 입력이 남고 마스크는 duplicate_merged."""
        tr = tracker()
        f = tr.update(key(0), 0, inputs([bolt_inst(0, 0.50, 0.0, conf=0.6), bolt_inst(1, 0.5003, 0.0, endpoints=False, conf=0.9)]))
        self.assertEqual((f.assignments[0].reason, f.assignments[1].reason), (None, "duplicate_merged"))
        self.assertEqual(tr.stats()["dup_merged_with_mask"], 1)

    def test_distinct_side_by_side_not_merged(self):
        """나란한 두 볼트(10 mm 간격, 포함률 < 0.85), 둘 다 마스크 → 합치지 않음."""
        a, b = bolt_inst(0, 0.50, 0.0, endpoints=False), bolt_inst(1, 0.51, 0.0, endpoints=False)
        self.assertLess(containment(mask_of(a.polygon), mask_of(b.polygon)), 0.85)
        tr = tracker()
        f = tr.update(key(0), 0, inputs([a, b]))
        self.assertTrue(all(x.reason != "duplicate_merged" for x in f.assignments))

    def test_contained_mask_duplicate_merged(self):
        """마스크 포함률 ≥ 0.85인 중복(중심 3 mm 밖)도 합침."""
        a = bolt_inst(0, 0.50, 0.0, endpoints=False, conf=0.9, length_m=0.040)
        b = bolt_inst(1, 0.50, 0.004, endpoints=False, conf=0.5, length_m=0.030)  # 짧은 쪽이 거의 포함됨
        ina = inputs([a, b])
        self.assertGreater(math.dist(ina[0].center_mm, ina[1].center_mm), 3.0)
        self.assertGreaterEqual(containment(ina[0].mask, ina[1].mask), 0.85)
        f = tracker().update(key(0), 0, ina)
        self.assertEqual(f.assignments[1].reason, "duplicate_merged")


def endpoint_gap_mm(a, b):
    """두 끝점 입력의 끝점 차(순서 무관 최선 짝의 큰 쪽, mm)."""
    (a0, a1), (b0, b1) = a.endpoints_mm, b.endpoints_mm

    return min(max(math.dist(a0, b0), math.dist(a1, b1)), max(math.dist(a0, b1), math.dist(a1, b0)))


def d13_pair(i0=0, i1=1, x_m=0.50, y_m=0.0, *, conf=(0.55, 0.06), extra_m=0.006, side_m=0.0):
    """#274 D13 모사: 같은 볼트(M3×20) 위 두 축. 둘째는 한 끝만 extra_m 더 길고 옆으로 side_m 비껴 있다."""
    a = bolt_inst(i0, x_m, y_m, conf=conf[0], length_m=0.020)
    b = bolt_inst(i1, x_m + side_m, y_m + extra_m / 2, conf=conf[1], length_m=0.020 + extra_m)

    return [a, b]


class EndpointContainmentMergeTest(unittest.TestCase):
    """#276 D13 수정 (a2): 끝점 쌍도 마스크 포함률(mask_dup_contain)로 같은 볼트를 합친다."""

    def test_d13_pair_merged_by_containment(self):
        """끝점 차 3~7 mm·포함률 ≥ 0.95인 끝점 쌍 → 합침, 점수 높은 쪽이 남고 포함률 계수 1."""
        ina = inputs(d13_pair())
        self.assertTrue(3.0 < endpoint_gap_mm(ina[0], ina[1]) < 7.0)
        self.assertGreaterEqual(containment(ina[0].mask, ina[1].mask), 0.95)
        tr = tracker()
        f = tr.update(key(0), 0, ina)
        self.assertEqual((f.assignments[0].reason, f.assignments[1].reason), (None, "duplicate_merged"))
        self.assertEqual((tr.stats()["dup_merged"], tr.stats()["dup_merged_endpoint_contain"]), (1, 1))

    def test_kept_is_higher_confidence_even_if_longer(self):
        """남기는 순서는 기존 그대로(끝점 먼저·점수 높은 순): 긴 쪽의 점수가 높으면 긴 쪽이 남는다."""
        f = tracker().update(key(0), 0, inputs(d13_pair(conf=(0.06, 0.55))))
        self.assertEqual((f.assignments[0].reason, f.assignments[1].reason), ("duplicate_merged", None))

    def test_confirmed_id_survives_d13_pair(self):
        """확정 트랙 위에 D13 쌍이 나와도 번호 유지(모호·새 번호 없음)."""
        tr = tracker()
        ids = [tr.update(key(i), i * DT_MS, inputs([bolt_inst(0, 0.50, 0.0, conf=0.55, length_m=0.020)])).assignments[0].part_track_id
               for i in range(4)]
        f = tr.update(key(4), 4 * DT_MS, inputs(d13_pair()))
        self.assertIsNotNone(ids[-1])
        self.assertEqual((f.assignments[0].part_track_id, f.assignments[0].state, f.assignments[1].reason),
                         (ids[-1], TrackState.CONFIRMED, "duplicate_merged"))
        self.assertEqual(tr.stats()["ids_retired"], 0)

    def test_without_masks_endpoint_rule_only(self):
        """마스크가 없는 끝점 입력(mask_fallback 끔)은 끝점 규칙만: 같은 D13 쌍을 합치지 않고 확정 트랙은 모호로 끝난다(수정 전 동작)."""
        tr = tracker(mask_fallback=False)

        for i in range(4):
            tr.update(key(i), i * DT_MS, inputs([bolt_inst(0, 0.50, 0.0, conf=0.55, length_m=0.020)], fallback=False))

        ina = inputs(d13_pair(), fallback=False)
        self.assertEqual([i.mask for i in ina], [None, None])
        f = tr.update(key(4), 4 * DT_MS, ina)
        self.assertEqual([a.state for a in f.assignments], [TrackState.AMBIGUOUS, TrackState.AMBIGUOUS])
        self.assertEqual((tr.stats()["dup_merged"], tr.stats()["dup_merged_endpoint_contain"], tr.stats()["ids_retired"]), (0, 0, 1))

    def test_side_by_side_endpoint_bolts_not_merged(self):
        """나란히 붙은 두 M3(축 간격 6 mm, 끝점 차 > 3 mm, 포함률 < 0.85) → 합치지 않고 번호 2개."""
        tr = tracker()

        for i in range(5):
            ina = inputs([bolt_inst(0, 0.50, 0.0, length_m=0.020), bolt_inst(1, 0.506, 0.0, length_m=0.020)])
            f = tr.update(key(i), i * DT_MS, ina)

        self.assertGreater(endpoint_gap_mm(ina[0], ina[1]), 3.0)
        self.assertLess(containment(ina[0].mask, ina[1].mask), 0.85)
        self.assertEqual(tr.stats()["dup_merged"], 0)
        self.assertEqual(len({a.part_track_id for a in f.assignments} - {None}), 2)

    def test_containment_threshold_boundary(self):
        """경계: 포함률 = 문턱이면 합침(≥), 문턱이 그보다 아주 조금 크면 안 합침. 끝점 규칙 밖인 쌍으로 확인."""
        pair = d13_pair(side_m=0.001)   # 옆으로 1 mm 비껴 포함률 < 1
        ina = inputs(pair)
        c = containment(ina[0].mask, ina[1].mask)
        self.assertTrue(0.85 <= c < 1.0)
        self.assertGreater(endpoint_gap_mm(ina[0], ina[1]), 3.0)
        at = tracker(mask_dup_contain=c).update(key(0), 0, inputs(pair))
        above = tracker(mask_dup_contain=c + 1e-9).update(key(0), 0, inputs(pair))
        self.assertEqual(at.assignments[1].reason, "duplicate_merged")
        self.assertNotEqual(above.assignments[1].reason, "duplicate_merged")

    def test_endpoint_rule_still_merges_without_containment(self):
        """기존 3 mm 끝점 규칙은 그대로: 포함률이 낮아도(마스크 없음 포함) 끝점이 3 mm 안이면 합침, 포함률 계수는 0."""
        tr = tracker(mask_fallback=False)
        f = tr.update(key(0), 0, inputs([bolt_inst(0, 0.50, 0.0), bolt_inst(1, 0.5005, 0.0005, conf=0.6)], fallback=False))
        self.assertEqual(f.assignments[1].reason, "duplicate_merged")
        self.assertEqual(tr.stats()["dup_merged_endpoint_contain"], 0)


class BRuleUnchangedTest(unittest.TestCase):
    """마스크로 확정된 트랙이라도 B 규칙(확정·선택 인스턴스)은 그대로."""

    def test_mask_confirmed_but_unselected_gives_no_b_id(self):
        """마스크로 확정된 인스턴스라도 관측 선택이 없으면(무효 관측) 번호를 싣지 않는다."""
        tr = tracker()

        for i in range(5):
            f = tr.update(key(i), i * DT_MS, inputs([bolt_inst(0, 0.5, 0.0, endpoints=False)]))

        self.assertEqual(f.assignments[0].state, TrackState.CONFIRMED)
        ctx = SimpleNamespace(frame=SimpleNamespace(run_id="r", source_id="s", image_seq=4, capture_generation=0),
                              generation=0, job_seq=4, detector_config_id="d")
        self.assertEqual(confirmed_track_id(f, ctx, SimpleNamespace(selected_index=None)), (None, NO_SELECTION))


class LiveRecordTest(LiveBase):
    """추적 스레드 기록: 입력 종류(W+M)·리셋·종료 사건."""

    def test_input_kind_recorded_and_mask_tracks(self):
        """world_tracks 배정마다 input(endpoint|mask|none), 끝점 없는 볼트도 확정 번호."""
        lw = self.make()

        for i in range(5):
            lw.offer(ctx_of(i), seg_result([bolt_inst(0, 0.5, 0.0, endpoints=False, T_W_C=self.lookup.T)], crop_wh=(1280, 720)),
                     self.camera)
            lw.wait(ctx_of(i))

        recs = self.records()
        frames = [r for r in recs if r["kind"] == "frame"]
        self.assertEqual({t["input"] for r in frames for t in r["tracks"]}, {"mask"})
        self.assertIsNotNone(frames[-1]["tracks"][0]["part_track_id"])

    def test_d3_reset_and_closed_events(self):
        """초기화 때 reset 줄, 닫힐 때 남은 요청을 처리하고 closed 줄 1개(pending_reset_applied 참)."""
        lw = self.make()
        self.offer(0)
        lw.wait(ctx_of(0))
        lw.request_reset("boundary:profile_changed")
        self.offer(1)
        lw.wait(ctx_of(1))
        lw.request_reset("run_end:max_frames")   # 처리 전에 닫힘
        recs = self.records()
        kinds = [(r["kind"], r.get("reason")) for r in recs if r["kind"] != "frame"]
        self.assertIn(("reset", "boundary:profile_changed"), kinds)
        self.assertEqual(kinds[-2:], [("reset", "run_end:max_frames"), ("closed", None)])
        closed = recs[-1]
        self.assertEqual((closed["pending_reset_applied"], closed["last_reset"]), (True, "run_end:max_frames"))
        self.assertEqual(sum(1 for r in recs if r["kind"] == "closed"), 1)


class ConfigTest(unittest.TestCase):
    """설정 키(W+M·진단 조회 상한)."""

    def test_new_keys_required(self):
        """mask_fallback·mask_dup_contain·conveyor_probe_timeout_s는 필수(기본값 없음), 형·범위 검사."""
        from eyetoarm.vision.live_world_tracking import LagConfigError, parse_live_world
        from eyetoarm.vision.world_tracking import WorldTrackerConfigError

        doc = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual((doc["tracker"]["mask_fallback"], doc["tracker"]["mask_dup_contain"]), (True, 0.85))

        for bad in ({k: v for k, v in doc["tracker"].items() if k != "mask_fallback"},
                    {**doc["tracker"], "mask_fallback": 1}, {**doc["tracker"], "mask_dup_contain": 1.5}):
            with self.subTest(bad=list(bad)[-1:]), self.assertRaises(WorldTrackerConfigError):
                parse_world_tracker(bad)

        with self.assertRaises(LagConfigError):
            parse_live_world({k: v for k, v in doc["live"].items() if k != "conveyor_probe_timeout_s"})

        self.assertEqual(parse_live_world(doc["live"]).conveyor_probe_timeout_s, 5.0)


class D2ProbeTimeoutTest(unittest.TestCase):
    """진단 조회 상한은 설정값으로 기본 조회기에 넘어간다(조회값은 요약 진단에만)."""

    def test_default_probe_gets_config_timeout(self):
        """기본 조회기를 가로채 상한 인자가 설정값(5.0)인지, 실패 status가 요약 진단에만 남는지."""
        from test_live_world_tracking import WORLD, WorldCli
        from test_run_pipeline_live_observation import Scenario

        class T(WorldCli):
            def runTest(self):
                pass

        t = T()
        t.setUp()
        self.addCleanup(t.doCleanups)
        seen = []

        def fake_probe(timeout_s):
            seen.append(timeout_s)
            return {"node": "/conveyor_controller", "speed_mps": None, "status": "no_reply", "timeout_s": timeout_s}

        with mock.patch.object(t.mod, "_ros_conveyor_speed_probe", fake_probe):
            out = t.out("d2")
            log = str(Path(t.tmp.name) / "d2.jsonl")
            sc = Scenario(4)
            sc.link = lambda: t.controller.current
            from test_live_sort_part import SORT_CONFIG

            # 추적 켬 송신은 require_track_id가 필수라 추적용 시험 설정(같은 폴더의 새 파일)을 쓴다
            SORT_CONFIG = SORT_CONFIG.with_name("live_observation_sort_part_track_test.json")
            from test_run_pipeline_capture_observation import model_with
            from test_run_pipeline_live_observation import FakeTf

            argv = ["--ros-link", "--detector", "ai", "--ai-config", t.case.ai, "--log", log, "--live-observation-out", out,
                    "--live-observation-config", str(SORT_CONFIG), "--max-frames", "4", *WORLD, "--live-sort-part"]
            err = io.StringIO()
            code = t.mod.main(argv, stderr=err, ros_available=lambda: True,
                              model_factory=lambda path: model_with(sc.masks_fn), live_source_factory=sc.factory_maker([]),
                              live_tf_factory=lambda c: FakeTf(), live_run_controller=t.controller,
                              live_sort_part_factory=t.factory())

        self.assertEqual(code, 0, err.getvalue())
        self.assertEqual(seen, [5.0])
        diag = t.summary(out)["world_tracking"]["diagnostics"]["conveyor_speed_mps"]
        self.assertEqual((diag["status"], diag["timeout_s"]), ("no_reply", 5.0))


class CliMaskTest(unittest.TestCase):
    """CLI: 끝점이 없는 볼트(무효 관측)는 W 추적 번호가 생겨도 B로 가지 않는다."""

    def test_mask_tracked_but_never_to_b(self):
        """사각 마스크(축 모호, 끝점 없음) 프레임만: 수락 0·송신 0, world_tracks에는 mask 입력 확정 번호."""
        from test_live_world_tracking import WorldCli
        from test_run_pipeline_capture_observation import rect_mask
        from test_run_pipeline_live_observation import Scenario

        class T(WorldCli):
            def runTest(self):
                pass

        t = T()
        t.setUp()
        self.addCleanup(t.doCleanups)
        sc = Scenario(8, masks={i: (lambda: [rect_mask()]) for i in range(8)})
        code, err, out, rt = t.run_world(sc)
        self.assertEqual(code, 0, err)
        self.assertEqual((t.accepts(out), rt.sent), ([], []))
        frames = [r for r in t.world_records(out) if r["kind"] == "frame"]
        tracks = [x for r in frames for x in r["tracks"] if x["input"] == "mask"]
        self.assertTrue(tracks and any(x["part_track_id"] is not None for x in tracks), tracks)


if __name__ == "__main__":
    unittest.main()
