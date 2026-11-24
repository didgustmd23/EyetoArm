"""`run_pipeline.py` AI 실행 경로의 가중치 sha256 대조 시험.

`test_ai_detector`는 어댑터 하나만 본다. 여기서는 설정 파일 → CLI → 모델 준비로 이어지는 실제 시작 경로에서
가중치 파일이 설정의 `weights_sha256`과 다르면 모델을 만들지 않고 시작을 거부하는지 본다.
"""
import unittest

import ai_process_fake as fake
from test_run_pipeline_infer_process import StoredCase, records


class WeightsShaMismatchTest(StoredCase):
    """설정 파일을 쓴 뒤 가중치가 바뀐 경우."""

    def test_changed_weights_refuse_start_without_model(self):
        """가중치가 설정 sha와 다르면 종료 1, MODEL_CONFIG_INVALID 사건, 모델 생성·처리 0이다."""
        weights = self.d / "best.pt"
        weights.write_bytes(b"weights replaced after the config was written")
        built = []

        def factory(path):
            """모델을 만들면 기록한다(불리면 안 됨)."""
            built.append(path)

            return fake.FakeModel(path)

        log = self.d / "sha.jsonl"
        code, err = self.stored(self.images([0, 0]), log, model_factory=factory)

        self.assertEqual(code, 1, err)
        self.assertEqual(built, [])

        recs = records(log)
        events = [r for r in recs if r["kind"] == "event" and r.get("invalidation_kind") == "MODEL_CONFIG_INVALID"]
        self.assertEqual(len(events), 1, recs)
        self.assertIn("sha256 mismatch", str(events[0]))
        self.assertEqual([r for r in recs if r["kind"] == "process"], [])


if __name__ == "__main__":
    unittest.main()
