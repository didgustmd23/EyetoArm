"""학습기 I1 — 공식 가중치 확인·내려받기 보호(`eyetoarm.training.weights`) 시험. 네트워크를 쓰지 않는다(가짜 내려받기).

근거: 학습기 설계 r2 §3(있으면 sha256·크기 대조·불일치 시 덮어쓰지 않음, 없을 때 `allow_download` 명시만, `.part`에 받고 해시가
맞을 때만 이름 바꿈, provenance 기록, ultralytics 자동 내려받기 미사용).
"""
import hashlib
import json
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

from eyetoarm.training.config import Candidate
from eyetoarm.training import weights as weights_mod
from eyetoarm.training.weights import WeightsError, check_weights, download_with_deadline, prepare_weights


def _write(path, data, mode="w"):
    """파일을 쓰고 닫는다(시험 도우미)."""
    with open(path, mode) as f:
        f.write(data)


def _read(path, mode="r"):
    """파일을 읽고 닫는다(시험 도우미)."""
    with open(path, mode) as f:
        return f.read()


CONTENT = b"official weights bytes" * 10


def candidate(content=CONTENT, **over):
    """가짜 공식 가중치 후보."""
    fields = {"exp_id": "S1-01", "family": "yolov8n-seg", "weights_file": "yolov8n-seg.pt",
              "source": {"repo": "ultralytics/assets", "tag": "v8.4.0",
                         "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-seg.pt"},
              "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}
    fields.update(over)
    return Candidate(**fields)


class WeightsTest(unittest.TestCase):
    """확인·준비."""

    def setUp(self):
        """임시 가중치 폴더."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name

    def put(self, content):
        """가중치 파일을 쓴다."""
        with open(os.path.join(self.dir, "yolov8n-seg.pt"), "wb") as f:
            f.write(content)

    def test_check_ok_missing_mismatch(self):
        """일치 → ok, 없음 → missing, 내용·크기 다름 → mismatch(실제 해시 보고)."""
        c = candidate()
        self.assertEqual(check_weights(c, self.dir).status, "missing")
        self.put(CONTENT)
        self.assertEqual(check_weights(c, self.dir).status, "ok")
        self.put(b"tampered")
        r = check_weights(c, self.dir)
        self.assertEqual((r.status, r.actual_sha256), ("mismatch", hashlib.sha256(b"tampered").hexdigest()))

    def test_existing_reused_without_download(self):
        """이미 일치하는 파일은 내려받지 않고 재사용(method existing)하며 provenance를 남긴다."""
        self.put(CONTENT)
        calls = []
        prov = prepare_weights(candidate(), self.dir, allow_download=True,
                               downloader=lambda url, dest, timeout: calls.append(url))

        self.assertEqual((prov["method"], calls), ("existing", []))
        doc = json.loads(_read(os.path.join(self.dir, "provenance.json")))
        self.assertEqual(doc["yolov8n-seg.pt"]["sha256"], candidate().sha256)

    def test_mismatch_never_overwritten(self):
        """기존 파일이 불일치면 내려받기가 허용돼도 덮어쓰지 않고 WeightsError, 파일 내용 그대로."""
        self.put(b"tampered")
        calls = []

        with self.assertRaises(WeightsError):
            prepare_weights(candidate(), self.dir, allow_download=True,
                            downloader=lambda url, dest, timeout: calls.append(dest))

        self.assertEqual(calls, [])                      # 내려받기 시도 자체가 없어야 한다

        self.assertEqual(_read(os.path.join(self.dir, "yolov8n-seg.pt"), "rb"), b"tampered")

    def test_missing_requires_explicit_allow(self):
        """없는 파일은 allow_download=False(기본)면 WeightsError이고 내려받지 않는다."""
        with self.assertRaises(WeightsError):
            prepare_weights(candidate(), self.dir, downloader=lambda url, dest, timeout: self.fail("download"))

        self.assertFalse(os.path.exists(os.path.join(self.dir, "yolov8n-seg.pt")))

    def test_download_to_part_then_verify(self):
        """허용 시 공식 url을 .part로 받고 sha256·크기가 맞을 때만 최종 이름이 된다(method downloaded)."""
        seen = []

        def downloader(url, dest, timeout):
            """가짜 내려받기."""
            seen.append((url, os.path.basename(dest), timeout))
            _write(dest, CONTENT, "wb")

        prov = prepare_weights(candidate(), self.dir, allow_download=True, downloader=downloader, timeout_s=60)

        self.assertEqual(seen, [(candidate().source["url"], "yolov8n-seg.pt.part", 60)])
        self.assertEqual(prov["method"], "downloaded")
        self.assertEqual(check_weights(candidate(), self.dir).status, "ok")
        self.assertFalse(os.path.exists(os.path.join(self.dir, "yolov8n-seg.pt.part")))

    def test_bad_download_removed(self):
        """받은 내용이 해시와 다르거나 내려받기가 실패하면 .part를 지우고 최종 파일을 만들지 않는다."""
        for name, dl in (("hash", lambda url, dest, timeout: _write(dest, b"wrong", "wb")),
                         ("error", lambda url, dest, timeout: (_ for _ in ()).throw(OSError("network down")))):
            with self.subTest(case=name):
                with self.assertRaises(WeightsError):
                    prepare_weights(candidate(), self.dir, allow_download=True, downloader=dl)
                self.assertEqual(sorted(os.listdir(self.dir)), [])

    def test_existing_part_file_rejected(self):
        """남은 .part 파일이 있으면 덮어쓰지 않고 WeightsError(이전 시도 확인 필요)."""
        _write(os.path.join(self.dir, "yolov8n-seg.pt.part"), b"half", "wb")

        calls = []

        with self.assertRaises(WeightsError):
            prepare_weights(candidate(), self.dir, allow_download=True,
                            downloader=lambda url, dest, timeout: calls.append(dest))

        self.assertEqual(calls, [])
        self.assertEqual(_read(os.path.join(self.dir, "yolov8n-seg.pt.part"), "rb"), b"half")


class FakeResponse:
    """가짜 HTTP 응답 스트림(실제 네트워크 없음). chunks는 read마다 돌려줄 바이트, 예외면 그 자리에서 올린다."""

    def __init__(self, chunks, length=None, clock=None, step_s=0.0):
        """읽기마다 clock을 step_s만큼 진행한다."""
        self.chunks, self.length, self.clock, self.step_s, self.reads, self.closed = list(chunks), length, clock, step_s, 0, False
        self.headers = {} if length is None else {"Content-Length": str(length)}

    def read(self, n):
        """다음 조각(없으면 b"")."""
        self.reads += 1
        if self.clock is not None:
            self.clock[0] += self.step_s
        if not self.chunks:
            return b""
        c = self.chunks.pop(0)
        if isinstance(c, BaseException):
            raise c
        return c

    def __enter__(self):
        """with 지원."""
        return self

    def __exit__(self, *exc):
        """닫힘 기록."""
        self.closed = True


class StreamDownloadTest(unittest.TestCase):
    """기본 내려받기 경로의 전송 중 상한(r1 F1). 실제 네트워크 없이 가짜 응답 스트림을 주입한다."""

    def setUp(self):
        """임시 가중치 폴더."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name

    def run_with(self, resp, clock=None, timeout_s=60.0):
        """주입한 응답으로 prepare_weights를 부른다."""
        opened = []

        def opener(url, timeout):
            """요청 url·개별 I/O 제한을 기록하고 응답을 돌려준다."""
            opened.append((url, timeout))
            return resp

        kw = {"clock": (lambda: clock[0])} if clock is not None else {}
        return prepare_weights(candidate(), self.dir, allow_download=True, opener=opener, timeout_s=timeout_s,
                               **kw), opened

    def test_normal_stream_succeeds(self):
        """정상 조각 스트림은 끝까지 받아 검증 뒤 최종 파일이 된다. 개별 I/O 제한은 전체 기한 이하."""
        resp = FakeResponse([CONTENT[:50], CONTENT[50:]], length=len(CONTENT))
        prov, opened = self.run_with(resp)

        self.assertEqual(prov["method"], "downloaded")
        self.assertTrue(resp.closed)
        self.assertLessEqual(opened[0][1], 60.0)
        self.assertEqual(sorted(os.listdir(self.dir)), ["provenance.json", "yolov8n-seg.pt"])

    def test_oversize_without_content_length_stops_early(self):
        """Content-Length가 없어도 누적 바이트가 허용 크기를 넘는 즉시 중단·.part 삭제, 남은 조각은 읽지 않는다."""
        big = [b"x" * 100] * 1000
        resp = FakeResponse(big)

        with self.assertRaises(WeightsError) as ctx:
            self.run_with(resp)

        self.assertIn("exceeds", str(ctx.exception))
        self.assertLessEqual(resp.reads, len(CONTENT) // 100 + 2)
        self.assertEqual(os.listdir(self.dir), [])

    def test_oversize_content_length_rejected_before_reading(self):
        """Content-Length가 허용 크기보다 크면 읽기 전에 거부한다."""
        resp = FakeResponse([CONTENT], length=len(CONTENT) + 1)

        with self.assertRaises(WeightsError):
            self.run_with(resp)

        self.assertEqual((resp.reads, os.listdir(self.dir)), (0, []))

    def test_trickle_stops_at_total_deadline(self):
        """조금씩 진행하는(끝나지 않는) 응답은 단조 시계 기반 전체 기한에서 중단하고 .part를 지운다."""
        clock = [0.0]
        resp = FakeResponse([b"x"] * 10_000, clock=clock, step_s=1.0)

        with self.assertRaises(WeightsError) as ctx:
            self.run_with(resp, clock=clock, timeout_s=30.0)

        self.assertIn("deadline", str(ctx.exception))
        self.assertLessEqual(resp.reads, 32)
        self.assertEqual(os.listdir(self.dir), [])

    def test_read_error_midstream_cleans_up(self):
        """중간 읽기 오류는 WeightsError로 바뀌고 .part를 지운다(최종 파일 없음)."""
        resp = FakeResponse([CONTENT[:10], OSError("connection reset")])

        with self.assertRaises(WeightsError):
            self.run_with(resp)

        self.assertEqual(os.listdir(self.dir), [])


class ChildDownloadTest(unittest.TestCase):
    """실제 네트워크 경로의 부모 소유 전체 기한(r2 F1 잔여). 네트워크 대신 자식 명령을 주입한다."""

    def setUp(self):
        """임시 폴더."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dest = os.path.join(self.tmp.name, "w.pt.part")

    def test_stalled_child_killed_at_deadline(self):
        """끝나지 않는 자식(잠자기)은 부모 전체 기한(0.5 s)에서 그룹 SIGKILL·회수 확인되고 WeightsError(child 정보 포함), 소요 < 기한+회수 한도."""
        t0 = time.monotonic()

        with self.assertRaises(WeightsError) as ctx:
            download_with_deadline("http://x", self.dest, max_bytes=10, deadline_s=0.5, io_timeout_s=0.5, reap_s=2.0,
                                   child_argv=[sys.executable, "-c", "import time; time.sleep(60)"])

        child = ctx.exception.child
        self.assertEqual((child["kill"], child["reaped"], child["rc"]), ("killpg", True, -9))
        self.assertIn("deadline", str(ctx.exception))
        self.assertLess(time.monotonic() - t0, 3.0)
        with self.assertRaises(ProcessLookupError):
            os.kill(child["pid"], 0)

    def test_child_size_error_and_success(self):
        """자식 종료 코드 3(크기 초과)은 그 메시지로 WeightsError, 0이면 성공 정보(rc 0·reaped)."""
        code3 = "import sys; sys.stderr.write('download size 11 exceeds allowed 10 bytes'); sys.exit(3)"
        with self.assertRaises(WeightsError) as ctx:
            download_with_deadline("http://x", self.dest, max_bytes=10, deadline_s=5, io_timeout_s=1,
                                   child_argv=[sys.executable, "-c", code3])
        self.assertIn("exceeds allowed 10", str(ctx.exception))

        ok = f"open({self.dest!r}, 'wb').write(b'ok')"
        info = download_with_deadline("http://x", self.dest, max_bytes=10, deadline_s=5, io_timeout_s=1,
                                      child_argv=[sys.executable, "-c", ok])
        self.assertEqual((info["rc"], info["reaped"], _read(self.dest, "rb")), (0, True, b"ok"))

    def test_default_prepare_path_uses_child(self):
        """opener·downloader를 주지 않으면 prepare_weights는 자식 경로(download_with_deadline)를 후보 크기·전체 기한·개별 제한으로 부르고, 기한 오류면 .part를 지운다."""
        calls = []

        def fake(url, dest, **kw):
            """자식 경로 대역: 호출 기록 후 기한 오류."""
            calls.append((url, os.path.basename(dest), kw))
            _write(dest, b"partial", "wb")
            raise WeightsError("download deadline 7 s exceeded", child={"kill": "killpg"})

        with mock.patch.object(weights_mod, "download_with_deadline", fake):
            with self.assertRaises(WeightsError):
                prepare_weights(candidate(), self.tmp.name, allow_download=True, timeout_s=7, io_timeout_s=30)

        self.assertEqual(calls, [(candidate().source["url"], "yolov8n-seg.pt.part",
                                  {"max_bytes": len(CONTENT), "deadline_s": 7, "io_timeout_s": 7,
                                   "should_stop": None})])
        self.assertEqual(os.listdir(self.tmp.name), [])


class DownloadInterruptTest(unittest.TestCase):
    """다운로드 중 정지 요청·예외에서 자식 핸들을 잃지 않는다(I2 확인점 3)."""

    def setUp(self):
        """임시 폴더."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dest = os.path.join(self.tmp.name, "w.pt.part")

    def test_stop_flag_ends_long_download(self):
        """긴 기한(60 s) 중 정지 요청(should_stop)이 오면 확인 간격 안에 자식을 그룹 kill·회수하고 WeightsError(원인 stop_requested)."""
        t0 = time.monotonic()
        state = {"n": 0}

        def should_stop():
            """세 번째 확인부터 참."""
            state["n"] += 1
            return state["n"] >= 3

        with self.assertRaises(WeightsError) as ctx:
            download_with_deadline("http://x", self.dest, max_bytes=10, deadline_s=60, io_timeout_s=1, reap_s=2,
                                   child_argv=[sys.executable, "-c", "import time; time.sleep(60)"],
                                   should_stop=should_stop, poll_s=0.05)

        self.assertIn("stop_requested", str(ctx.exception))
        self.assertEqual((ctx.exception.child["kill"], ctx.exception.child["reaped"]), ("killpg", True))
        self.assertLess(time.monotonic() - t0, 3.0)

    def test_keyboard_interrupt_cleans_child_and_reraises(self):
        """확인 함수에서 KeyboardInterrupt가 나도 자식을 정리한 뒤 원래 KeyboardInterrupt를 다시 올린다(자식 pid 사라짐)."""
        seen = {}

        def should_stop():
            """첫 확인에서 중단."""
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt) as ctx:
            download_with_deadline("http://x", self.dest, max_bytes=10, deadline_s=60, io_timeout_s=1, reap_s=2,
                                   child_argv=[sys.executable, "-c", "import time; time.sleep(60)"],
                                   should_stop=should_stop, poll_s=0.05, on_child=lambda info: seen.update(info))

        child = getattr(ctx.exception, "child", None) or seen
        with self.assertRaises(ProcessLookupError):
            os.kill(child["pid"], 0)


if __name__ == "__main__":
    unittest.main()
