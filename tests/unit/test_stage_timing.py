"""단계별 시각 진단(`stage_timing`) 시험: 꺼져 있으면 무변, 켜면 표시·스레드 이름표 기록, 한 번만 켬(#372 ③)."""
import json
import os
import tempfile
import threading
import unittest

from eyetoarm.vision import stage_timing


class StageTimingTest(unittest.TestCase):
    """기록기 단위 시험."""

    def setUp(self):
        """임시 폴더, 끝나면 꺼 둔다."""
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.addCleanup(stage_timing.disable)
        self.path = os.path.join(self.dir.name, "stage.jsonl")

    def lines(self):
        """기록 줄."""
        with open(self.path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh]

    def test_off_is_noop(self):
        """꺼져 있으면 `mark`는 아무것도 쓰지 않고 `disable`은 None."""
        stage_timing.mark("taken", 1)

        self.assertFalse(stage_timing.enabled())
        self.assertIsNone(stage_timing.disable())
        self.assertFalse(os.path.exists(self.path))

    def test_marks_and_thread_names_are_written(self):
        """켜면 표시마다 단계·번호·시각·스레드 번호·값이 남고, 스레드 이름표에 표시한 스레드 이름이 있다."""
        stage_timing.enable(self.path)
        stage_timing.mark("taken", 7, received_ns=123)
        t = threading.Thread(target=lambda: stage_timing.mark("tracked", 7), name="probe-thread")
        t.start()
        t.join(2)
        summary = stage_timing.disable()
        recs = self.lines()
        marks = [r for r in recs if r["kind"] == "mark"]
        names = {k: v for r in recs if r["kind"] == "threads" for k, v in r["threads"].items()}

        self.assertEqual(summary, {"marks": 2, "error": None})
        self.assertEqual([(m["stage"], m["image_seq"]) for m in marks], [("taken", 7), ("tracked", 7)])
        self.assertEqual(marks[0]["received_ns"], 123)
        self.assertLessEqual(marks[0]["t_ns"], marks[1]["t_ns"])
        self.assertIn(str(threading.get_native_id()), names)
        self.assertEqual(names[str(threading.get_native_id())], threading.current_thread().name)

    def test_enable_twice_refused_and_existing_file_refused(self):
        """이미 켜져 있으면 RuntimeError, 있는 파일에는 쓰지 않는다(FileExistsError)."""
        stage_timing.enable(self.path)

        with self.assertRaises(RuntimeError):
            stage_timing.enable(self.path)

        stage_timing.disable()

        with self.assertRaises(FileExistsError):
            stage_timing.enable(self.path)

        self.assertFalse(stage_timing.enabled())


if __name__ == "__main__":
    unittest.main()
