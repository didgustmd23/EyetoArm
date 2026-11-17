import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dashboard-client" / "src"))  # canonical D GUI

from PySide6.QtGui import QImage

from eyetoarm.gui.display import DisplayModel


class DisplayModelTest(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.model = DisplayModel(clock=lambda: self.now)
        self.model.begin_session("run", "link")
        self.model.set_connected(True)
        self.image = QImage(2, 2, QImage.Format.Format_RGB32)
        self.image.fill(0xffffffff)

    def frame(self, seq=1, generation=1, calibration="cal"):
        return self.model.ingest_frame("link", self.image, source_id="link", image_seq=seq,
                                       generation=generation, calibration_id=calibration, run_id="run",
                                       capture_generation=1, detector_config_id="config", job_seq=seq)

    def result(self, seq=1, order=1, generation=1, calibration="cal", valid=True):
        return self.model.ingest_result(dict(source_id="link", image_seq=seq, calibration_id=calibration,
                                             run_id="run", capture_generation=1, detector_config_id="config", job_seq=seq,
                                             valid=valid, center_w_mm=[12, 34]), generation=generation, order=order)

    def test_matching_freshness_and_units(self):
        self.frame()
        self.result()
        self.assertEqual(self.model.matched_result()["center_w_mm"], [12, 34])
        self.now = 4
        self.assertFalse(self.model.has_matched_position())
        self.assertIsNone(self.model.matched_result())

    def test_mismatch_invalid_and_tombstone(self):
        self.frame()
        self.result(seq=2)
        self.assertFalse(self.model.has_matched_position())
        self.result(order=2, calibration="other")
        self.assertFalse(self.model.has_matched_position())
        self.result(order=3, valid=False)
        self.assertFalse(self.model.has_matched_position())
        self.assertTrue(self.model.invalidate("link", run_id="run", generation=1, order=5, reason="lost", detector_config_id="config", job_seq=1))
        self.assertFalse(self.result(order=4))
        self.model.set_connected(False)
        self.model.set_connected(True)
        self.frame()
        self.assertFalse(self.result(order=5))
        self.assertFalse(self.model.has_matched_position())
        self.frame(generation=2)
        self.assertFalse(self.result(generation=1, order=9))

    def test_order_session_scope_and_result_before_frame(self):
        self.assertTrue(self.result())
        self.assertIsNone(self.model.matched_result())
        self.assertTrue(self.frame())
        self.assertIsNotNone(self.model.matched_result())
        self.assertFalse(self.result(order=2))
        self.assertTrue(self.model.invalidate("other", run_id="run", generation=9, order=3, reason="other", detector_config_id="x"))
        self.assertIsNotNone(self.model.matched_result())
        self.assertFalse(self.model.invalidate("link", run_id="old", generation=1, order=4, reason="bad", detector_config_id="config"))
        self.assertFalse(self.model.invalidate("link", run_id="run", generation=1, order=4, reason="bad", detector_config_id="wrong"))
        self.assertTrue(self.model.invalidate("link", run_id="run", generation=1, order=5, reason="bad", detector_config_id="config", job_seq=1))
        self.assertFalse(self.result(order=6))
        self.assertTrue(self.model.begin_session("run2", "link"))
        self.assertFalse(self.frame())

    def test_copies_malformed_and_mismatch(self):
        self.frame()
        image = self.model.frame("link")["image"]
        self.image.fill(0)
        self.assertNotEqual(image.pixel(0, 0), self.image.pixel(0, 0))
        base = dict(source_id="link", image_seq=1, calibration_id="cal", run_id="run",
                    capture_generation=1, detector_config_id="config", job_seq=1,
                    valid=True, center_w_mm=[12, 34], theta_deg=None, theta_valid=False)
        for key, bad in (("valid", "true"), ("center_w_mm", [float("nan"), 1]),
                         ("center_w_mm", [True, 1]), ("job_seq", -1)):
            invalid = dict(base, **{key: bad})
            self.assertFalse(self.model.ingest_result(invalid, generation=1, order=1))
        for key, bad in (("run_id", "other"), ("capture_generation", 2),
                         ("detector_config_id", "other"), ("job_seq", 2)):
            self.model.begin_session("temp-" + key, "link")
            self.model.begin_session("run", "link")
            self.frame()
            mismatch = dict(base, **{key: bad})
            accepted = self.model.ingest_result(mismatch, generation=1, order=1)
            self.assertFalse(accepted and self.model.matched_result())
        self.model.begin_session("last", "link")
        self.model.begin_session("run", "link")
        self.frame()
        self.assertTrue(self.model.ingest_result(base, generation=1, order=1))
        base["center_w_mm"][0] = 999
        self.assertEqual(self.model.matched_result()["center_w_mm"], [12, 34])
        self.assertIsNone(self.model.matched_result()["theta_deg"])

    def test_variant_requires_matching_frame(self):
        self.frame()
        self.result()
        self.model.set_mode("detect")
        self.assertIsNone(self.model.matched_result())

    def test_reprocess_stale_and_scoped_invalidation(self):
        self.frame()
        self.result()
        self.now = 4
        self.assertTrue(self.model.ingest_frame("link", self.image, source_id="link", image_seq=1,
            generation=1, calibration_id="cal", run_id="run", capture_generation=1,
            detector_config_id="config", job_seq=2))
        self.assertTrue(self.model.ingest_result(dict(source_id="link", image_seq=1, calibration_id="cal",
            run_id="run", capture_generation=1, detector_config_id="config", job_seq=2,
            valid=True, center_w_mm=[1, 2]), generation=1, order=2))
        self.assertIsNone(self.model.matched_result())
        self.assertTrue(self.model.invalidate("link", run_id="run", generation=1, order=3,
            reason="old job", detector_config_id="config", job_seq=1))
        self.assertEqual(self.model.result["job_seq"], 2)
        self.assertFalse(self.result(order=4))
        self.assertTrue(self.frame(seq=3))
        self.assertTrue(self.result(seq=3, order=5))
        self.assertIsNotNone(self.model.matched_result())
        self.assertTrue(self.model.invalidate("link", run_id="run", generation=1, order=6,
            reason="no frame", detector_config_id="config"))
        self.assertTrue(self.frame(seq=4))
        self.assertFalse(self.result(seq=4, order=7))
        self.assertTrue(self.frame(seq=5, generation=2))
        self.assertTrue(self.result(seq=5, generation=2, order=1))

    def test_cross_variant_and_generation_keep_origin_age(self):
        self.frame()
        self.result()
        self.now = 4
        args = dict(source_id="link", image_seq=1, calibration_id="cal", run_id="run",
                    capture_generation=1, detector_config_id="config", job_seq=1)
        self.assertTrue(self.model.ingest_frame("link", self.image, generation=1, variant="detect", **args))
        self.model.set_mode("detect")
        self.assertIsNone(self.model.matched_result())
        args["detector_config_id"] = "new"
        args["job_seq"] = 2
        self.assertTrue(self.model.ingest_frame("link", self.image, generation=2, variant="center", **args))
        result = dict(args, valid=True, center_w_mm=[1, 2])
        self.assertTrue(self.model.ingest_result(result, generation=2, order=1))
        self.model.set_mode("center")
        self.assertIsNone(self.model.matched_result())

    def test_new_generation_invalidation_and_old_job(self):
        self.frame(seq=2)
        self.result(seq=2, order=2)
        self.assertFalse(self.result(seq=1, order=3))
        self.assertEqual(self.model.result["job_seq"], 2)
        self.assertTrue(self.model.invalidate("link", run_id="run", generation=2, order=1,
            reason="new config", detector_config_id="new"))
        self.assertIsNone(self.model.result)
        self.assertIsNone(self.model.matched_result())

    def test_explicit_generation_transition_and_scope(self):
        self.frame()
        self.result(order=100)
        args = dict(run_id="run", generation=2, order=1, detector_config_id="new")
        for source, change in (("other", {}), ("link", {"run_id": "old"}),
                               ("link", {"generation": True}), ("link", {"order": -1}),
                               ("link", {"detector_config_id": ""})):
            self.assertFalse(self.model.advance_generation(source, **dict(args, **change)))
            self.assertIsNotNone(self.model.matched_result())
        self.now = 2
        self.assertTrue(self.model.advance_generation("link", **args))
        self.assertIsNone(self.model.frame("link"))
        self.assertIsNone(self.model.result)
        frame = dict(source_id="link", image_seq=1, calibration_id="cal", run_id="run",
                     capture_generation=1, detector_config_id="new", job_seq=2)
        result = dict(frame, valid=True, center_w_mm=[1, 2])
        self.assertTrue(self.model.ingest_frame("link", self.image, generation=2, **frame))
        self.assertEqual(self.model.origins["link"]["received"], 0)
        self.assertFalse(self.model.ingest_result(result, generation=2, order=1))
        self.assertTrue(self.model.ingest_result(result, generation=2, order=2))
        self.assertFalse(self.model.advance_generation("link", **dict(args, order=3)))
        self.assertFalse(self.model.advance_generation("link", **dict(args, generation=1)))
        self.assertEqual(self.model.matched_result()["center_w_mm"], [1, 2])
        self.now = 4
        self.assertIsNone(self.model.matched_result())
        self.model.set_connected(False)
        state = (dict(self.model.generations), dict(self.model.active_configs),
                 dict(self.model.high_water), self.model.reason)
        self.assertFalse(self.model.advance_generation("link", **dict(args, generation=3)))
        self.assertEqual(state, (self.model.generations, self.model.active_configs,
                                 self.model.high_water, self.model.reason))

    def test_invalidation_removes_only_affected_annotated_frames(self):
        args = dict(source_id="link", image_seq=1, calibration_id="cal", run_id="run",
                    capture_generation=1, detector_config_id="config", job_seq=1, generation=1)
        for variant in ("raw", "detect", "center"):
            self.assertTrue(self.model.ingest_frame("link", self.image, variant=variant, **args))
        newer = dict(args, image_seq=2, job_seq=2)
        self.assertTrue(self.model.ingest_frame("link", self.image, variant="center", **newer))
        self.assertTrue(self.model.invalidate("link", run_id="run", generation=1, order=1,
                                              reason="old job", detector_config_id="config", job_seq=1))
        self.assertIn(("link", "raw"), self.model.frames)
        self.assertNotIn(("link", "detect"), self.model.frames)
        self.assertIn(("link", "center"), self.model.frames)
        self.assertFalse(self.model.ingest_frame("link", self.image, variant="detect", **args))
        self.assertTrue(self.model.invalidate("link", run_id="run", generation=1, order=2,
                                              reason="source error", detector_config_id="config"))
        self.assertNotIn(("link", "center"), self.model.frames)
        self.assertFalse(self.model.ingest_frame("link", self.image, variant="detect", **newer))
        self.assertFalse(self.result(seq=2, order=3))
        self.assertFalse(self.model.advance_generation("link", run_id="run", generation=1,
                                                        order=4, detector_config_id="config"))
        self.assertFalse(self.result(seq=2, order=4))
        self.assertTrue(self.model.advance_generation("link", run_id="run", generation=2,
                                                       order=1, detector_config_id="config"))
        self.assertTrue(self.frame(seq=2, generation=2))
        self.assertTrue(self.result(seq=2, generation=2, order=2))

    def test_same_job_cannot_replace_result_or_refresh_age(self):
        self.frame()
        self.result()
        self.now = 2
        conflicting = dict(source_id="link", image_seq=2, calibration_id="other", run_id="run",
                           capture_generation=1, detector_config_id="config", job_seq=1,
                           valid=True, center_w_mm=[99, 99])
        self.assertFalse(self.model.ingest_result(conflicting, generation=1, order=2))
        self.assertEqual(self.model.result["center_w_mm"], [12, 34])
        self.now = 4
        self.assertIsNone(self.model.matched_result())

    def test_reconnect_does_not_renew_frame(self):
        self.frame()
        self.result()
        self.model.set_connected(False)
        self.model.set_connected(True)
        self.assertFalse(self.frame())
        self.assertFalse(self.result(order=2))
        self.assertIsNone(self.model.matched_result())

    def b_output(self, seq, **changes):
        """B to_output_dict 모양(center_px 없음, mm 배열, 추가 키)에 D 내부 상관 필드를 더한 사례."""
        out = {"source_id": "link", "image_seq": seq, "part_track_id": 7, "calibration_id": "cal",
               "spec": "M8", "class_name": "bolt", "valid": True, "reason": None, "reason_owner": None,
               "center_w_mm": [12.3, -4.5], "theta_deg": None, "theta_valid": False,
               "length_mm": 30.0, "width_mm": 8.0}
        out.update(run_id="run", capture_generation=1, detector_config_id="config", job_seq=seq)
        out.update(changes)
        return out

    def test_b_output_shape_and_raw_b_rejected(self):
        self.frame()
        self.assertTrue(self.model.ingest_result(self.b_output(1), generation=1, order=1))
        matched = self.model.matched_result()
        self.assertEqual(matched["center_w_mm"], [12.3, -4.5])
        self.assertIsNone(matched.get("center_px"))
        self.assertIsNone(matched["theta_deg"])
        raw = {k: v for k, v in self.b_output(2).items()
               if k not in ("run_id", "capture_generation", "detector_config_id", "job_seq")}
        self.frame(seq=2)
        self.assertFalse(self.model.ingest_result(raw, generation=1, order=2))

    def test_invalid_b_result_clears_previous_values(self):
        self.frame()
        self.assertTrue(self.model.ingest_result(self.b_output(1), generation=1, order=1))
        self.assertTrue(self.model.has_matched_position())
        self.frame(seq=2)
        invalid = self.b_output(2, valid=False, reason="경계 접촉", center_w_mm=[None, None])
        self.assertTrue(self.model.ingest_result(invalid, generation=1, order=2))
        self.assertIsNone(self.model.matched_result())
        self.assertEqual(self.model.reason, "경계 접촉")
        self.assertEqual(self.model.latest_jobs["link"], 2)
        self.assertFalse(self.model.ingest_result(self.b_output(1), generation=1, order=3))
        self.frame(seq=3)
        for bad in ([None, 1.0], [float("inf"), 1.0], [False, 1.0], [None, None]):
            self.assertFalse(self.model.ingest_result(self.b_output(3, center_w_mm=bad), generation=1, order=4))
        self.assertFalse(self.model.ingest_result(self.b_output(3, valid=False, center_w_mm=[float("nan"), None]), generation=1, order=4))
        self.assertTrue(self.model.ingest_result(self.b_output(3), generation=1, order=4))

    def test_active_config_survives_disconnect_and_invalidation(self):
        self.frame()
        self.result()
        args = dict(run_id="run", generation=1, reason="x")
        self.assertTrue(self.model.invalidate("link", order=2, detector_config_id="config", job_seq=1, **args))
        self.assertIsNone(self.model.result)
        self.assertFalse(self.model.invalidate("link", order=3, detector_config_id="wrong", **args))
        self.model.set_connected(False)
        self.model.set_connected(True)
        self.assertFalse(self.model.invalidate("link", order=4, detector_config_id="wrong", **args))
        self.assertTrue(self.model.invalidate("link", run_id="run", generation=2, order=1, reason="new", detector_config_id="new"))
        self.assertFalse(self.model.invalidate("link", run_id="run", generation=2, order=2, reason="x", detector_config_id="config"))

    def test_same_generation_config_is_fixed_until_generation_bump(self):
        self.frame()
        self.result()
        self.now = 1
        b = dict(source_id="link", image_seq=2, calibration_id="cal", run_id="run",
                 capture_generation=1, detector_config_id="B", job_seq=2)
        state = (dict(self.model.origins), dict(self.model.frame_marks), dict(self.model.high_water))
        self.assertFalse(self.model.ingest_frame("link", self.image, generation=1, **b))
        self.assertFalse(self.model.ingest_result(dict(b, valid=True, center_w_mm=[1, 2]), generation=1, order=2))
        self.assertEqual(state, (self.model.origins, self.model.frame_marks, self.model.high_water))
        self.assertEqual(self.model.matched_result()["center_w_mm"], [12, 34])
        self.assertEqual(self.model.frame("link")["received"], 0)
        self.assertFalse(self.model.invalidate("link", run_id="run", generation=1, order=3, reason="x", detector_config_id="B"))
        self.assertTrue(self.model.invalidate("link", run_id="run", generation=1, order=3, reason="A 무효", detector_config_id="config"))
        self.assertIsNone(self.model.matched_result())
        self.assertTrue(self.model.ingest_frame("link", self.image, generation=2, **b))
        self.assertTrue(self.model.ingest_result(dict(b, valid=True, center_w_mm=[1, 2]), generation=2, order=1))
        self.assertEqual(self.model.matched_result()["center_w_mm"], [1, 2])
        self.assertTrue(self.model.invalidate("link", run_id="run", generation=2, order=2, reason="B 무효", detector_config_id="B"))
        self.assertIsNone(self.model.matched_result())


if __name__ == "__main__":
    unittest.main()
