"""scripts/run_vision.py 저장 입력 재현 CLI 시험.

근거: scripts/run_vision.py 모듈 docstring의 출력·종료 코드 계약. 실제 CLI는 subprocess로 실행하고
(sys.executable, 절대 스크립트 경로, PYTHONPATH=저장소 src, cwd 명시, timeout 10초),
예외 경로의 close 보장만 모듈을 직접 불러 소스를 주입해 확인한다.
"""
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "run_vision.py")
SRC = os.path.join(REPO_ROOT, "src")
TIMEOUT_S = 10

PUBLIC_SAMPLES = (
    "data/raw/train/z4773355985498_0b2ac5577b3fbe55daacf39124e8c881_jpg.rf.000fb5ac0b21b6bb802ec8b64ef699e9.jpg",
    "data/raw/train/M6_frame_50_jpg.rf.0ea94e92b4974ee895686f3b383ecfab.jpg",
    "data/raw/train/062_png_jpg.rf.005e97902406f75fcd43317fbe7f4b51.jpg",
)

FRAME_KEYS = {"kind", "run_id", "source_id", "image_seq", "source_index", "source_ref",
              "width", "height", "timestamp_base", "timestamp_ms", "received_monotonic_ns"}
EVENT_KEYS = {"kind", "run_id", "source_id", "status", "source_ref", "detail",
              "event_monotonic_ns"}
START_KEYS = {"kind", "run_id", "source_id", "cwd", "paths", "timestamp_base", "time_note"}
END_KEYS = {"kind", "run_id", "source_id", "frames", "failures", "reason"}


def run_cli(args, cwd):
    """저장소 src를 PYTHONPATH로 주고 `cwd`에서 `run_vision.py`를 새 프로세스로 실행한다(제한시간 10초)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run([sys.executable, SCRIPT, *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=TIMEOUT_S)


def parse_lines(stdout):
    """표준 출력을 줄마다 JSON으로 읽고, 모든 줄이 JSON 객체인지 확인한다."""
    records = [json.loads(line) for line in stdout.splitlines()]

    for rec in records:
        if not isinstance(rec, dict):
            raise AssertionError(f"stdout line is not a JSON object: {rec!r}")

    return records


class CliTestBase(unittest.TestCase):
    """합성 PNG·없는 경로·깨진 파일을 준비하는 CLI 시험 공통 기반."""

    def setUp(self):
        """임시 폴더에 크기·색이 다른 PNG 3장과 깨진 파일을 만들고 없는 경로를 정한다."""
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.a = self._png("a.png", (4, 6), (10, 20, 30))
        self.b = self._png("b.png", (5, 3), (200, 100, 50))
        self.c = self._png("c.png", (2, 7), (0, 255, 128))
        self.missing = os.path.join(self.dir, "missing.png")
        self.broken = os.path.join(self.dir, "broken.jpg")

        with open(self.broken, "wb") as f:
            f.write(b"\x00\x01not an image\xff\xfe" * 8)

    def tearDown(self):
        """임시 폴더를 지운다."""
        self._tmp.cleanup()

    def _png(self, name, hw, bgr):
        """한 색으로 채운 (h, w) BGR PNG를 저장하고 경로를 돌려준다."""
        image = np.empty((*hw, 3), dtype=np.uint8)
        image[:] = bgr
        path = os.path.join(self.dir, name)
        self.assertTrue(cv2.imwrite(path, image))

        return path

    def _list_file(self, lines, name="list.txt", directory=None):
        """이미지 목록 파일을 줄 단위로 쓰고 경로를 돌려준다."""
        path = os.path.join(directory or self.dir, name)

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return path

    def assertOk(self, proc):
        """종료 코드 0을 확인하고 출력 레코드 목록을 돌려준다."""
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return parse_lines(proc.stdout)


class OutputContractTest(CliTestBase):
    """표준 출력 JSON 줄의 순서·키·집계 계약을 확인한다(`run_vision.py` 모듈 docstring)."""

    def test_images_order_and_record_shapes(self):
        """`--images` 순서대로 start·frame×3·END_OF_STREAM 사건·end가 나오고, 각 레코드 키·시각 기준·순번·크기·실행 ID가 계약과 같다."""
        recs = self.assertOk(run_cli(["--images", self.c, self.a, self.b], cwd=self.dir))
        kinds = [r["kind"] for r in recs]

        self.assertEqual(kinds, ["start", "frame", "frame", "frame", "event", "end"])

        start, end = recs[0], recs[-1]

        self.assertEqual(set(start), START_KEYS)
        self.assertEqual(start["paths"], [self.c, self.a, self.b])
        self.assertEqual(start["source_id"], "stored0")
        self.assertEqual(start["timestamp_base"], "PC_RECEIVE")
        self.assertEqual(start["time_note"], "PC_RECEIVE는 파일 디코딩 완료 시각이며 촬영 시각이 아님")
        self.assertEqual(os.path.realpath(start["cwd"]), os.path.realpath(self.dir))

        frames = [r for r in recs if r["kind"] == "frame"]

        for f in frames:
            self.assertEqual(set(f), FRAME_KEYS)
            self.assertEqual(f["timestamp_base"], "PC_RECEIVE")
            self.assertEqual(f["timestamp_ms"], f["received_monotonic_ns"] // 1_000_000)

        self.assertEqual([f["source_ref"] for f in frames], [self.c, self.a, self.b])
        self.assertEqual([f["image_seq"] for f in frames], [0, 1, 2])
        self.assertEqual([(f["width"], f["height"]) for f in frames], [(7, 2), (6, 4), (3, 5)])

        eof = recs[4]

        self.assertEqual(set(eof), EVENT_KEYS)
        self.assertEqual(eof["status"], "END_OF_STREAM")
        self.assertIsNone(eof["source_ref"])
        self.assertEqual(set(end), END_KEYS)
        self.assertEqual((end["frames"], end["failures"], end["reason"]), (3, 0, "end_of_stream"))

        run_ids = {r["run_id"] for r in recs}

        self.assertEqual(len(run_ids), 1)
        self.assertEqual({r["source_id"] for r in recs}, {"stored0"})

    def test_mixed_failures_and_indices(self):
        """실패가 섞이면 사건으로 출력되고 `source_index`는 목록 위치, `image_seq`는 성공만 센다. end 집계와 `--source-id`가 반영된다."""
        recs = self.assertOk(run_cli(
            ["--images", self.a, self.missing, self.broken, self.b, "--source-id", "s9"],
            cwd=self.dir))
        body = recs[1:-1]

        self.assertEqual([r["kind"] for r in body], ["frame", "event", "event", "frame", "event"])
        self.assertEqual([r["source_index"] for r in body if r["kind"] == "frame"], [0, 3])
        self.assertEqual([r["image_seq"] for r in body if r["kind"] == "frame"], [0, 1])

        ev = [r for r in body if r["kind"] == "event"]

        self.assertEqual([e["status"] for e in ev],
                         ["SOURCE_UNAVAILABLE", "DECODE_FAILED", "END_OF_STREAM"])
        self.assertEqual([e["source_ref"] for e in ev], [self.missing, self.broken, None])

        for e in ev:
            self.assertNotIn("image_seq", e)
            self.assertNotIn("timestamp_ms", e)

        end = recs[-1]

        self.assertEqual((end["frames"], end["failures"], end["reason"]), (2, 2, "end_of_stream"))
        self.assertEqual({r["source_id"] for r in recs}, {"s9"})

    def test_image_list_order_blank_lines_and_cwd_relative(self):
        """`--image-list`는 빈 줄·앞뒤 공백을 버리고 순서를 지키며, 상대 경로는 목록 파일 위치가 아니라 실행 cwd 기준이다."""
        # 목록은 다른 폴더(sub/)에 두고, 이미지 상대 경로는 실행 cwd(self.dir) 기준
        sub = os.path.join(self.dir, "sub")
        os.mkdir(sub)
        list_path = self._list_file(["  b.png  ", "", "   ", "a.png", "c.png"], directory=sub)

        recs = self.assertOk(run_cli(["--image-list", list_path], cwd=self.dir))

        self.assertEqual(recs[0]["paths"], ["b.png", "a.png", "c.png"])

        frames = [r for r in recs if r["kind"] == "frame"]

        self.assertEqual([f["source_ref"] for f in frames], ["b.png", "a.png", "c.png"])
        self.assertEqual([f["source_index"] for f in frames], [0, 1, 2])

        # 같은 목록을 cwd=sub에서 실행하면 sub/에는 이미지가 없으므로 모두 SOURCE_UNAVAILABLE
        recs2 = self.assertOk(run_cli(["--image-list", list_path], cwd=sub))
        ev = [r for r in recs2 if r["kind"] == "event"]

        self.assertEqual([e["status"] for e in ev][:3], ["SOURCE_UNAVAILABLE"] * 3)
        self.assertEqual(recs2[-1]["frames"], 0)
        self.assertEqual(recs2[-1]["failures"], 3)

    def test_max_frames_stops_without_fake_eof(self):
        """`--max-frames`에 닿으면 가짜 END_OF_STREAM 없이 멈추고 end 사유가 `max_frames`다(실패는 프레임 수에 넣지 않음)."""
        recs = self.assertOk(run_cli(
            ["--images", self.missing, self.a, self.b, "--max-frames", "1"], cwd=self.dir))

        self.assertEqual([r["kind"] for r in recs], ["start", "event", "frame", "end"])
        self.assertEqual(recs[1]["status"], "SOURCE_UNAVAILABLE")
        self.assertEqual(recs[2]["source_ref"], self.a)
        self.assertEqual((recs[-1]["frames"], recs[-1]["failures"], recs[-1]["reason"]),
                         (1, 1, "max_frames"))
        self.assertNotIn("END_OF_STREAM", [r.get("status") for r in recs])

    def test_empty_list_single_event(self):
        """빈 줄뿐인 목록은 사건 하나("empty list")로 끝나고 end 사유가 `empty_list`다."""
        list_path = self._list_file(["", "   ", ""])
        recs = self.assertOk(run_cli(["--image-list", list_path], cwd=self.dir))

        self.assertEqual([r["kind"] for r in recs], ["start", "event", "end"])
        self.assertEqual(recs[0]["paths"], [])
        self.assertEqual((recs[1]["status"], recs[1]["detail"]), ("SOURCE_UNAVAILABLE", "empty list"))
        self.assertEqual((recs[2]["frames"], recs[2]["failures"], recs[2]["reason"]),
                         (0, 0, "empty_list"))

    def test_all_failures_terminate(self):
        """모든 항목이 실패해도 끝까지 진행해 END_OF_STREAM과 end로 정상 종료한다."""
        recs = self.assertOk(run_cli(["--images", self.missing, self.broken], cwd=self.dir))

        self.assertEqual([r["kind"] for r in recs], ["start", "event", "event", "event", "end"])
        self.assertEqual((recs[-1]["frames"], recs[-1]["failures"], recs[-1]["reason"]),
                         (0, 2, "end_of_stream"))


class ArgumentErrorTest(CliTestBase):
    """잘못된 인자·목록 파일을 argparse 사용법 오류(종료 코드 2)로 거부하는지 확인한다."""

    def assertUsageError(self, args):
        """종료 코드 2, 표준 출력 없음, 표준 오류에 argparse `usage:`·`error:`가 있는지 확인한다."""
        proc = run_cli(args, cwd=self.dir)
        self.assertEqual(proc.returncode, 2, f"{args}: {proc.stdout} {proc.stderr}")
        self.assertEqual(proc.stdout, "", args)

        # 스크립트 부재(파이썬 자체의 exit 2)와 구분하려고 argparse 사용법 출력까지 확인한다
        self.assertIn("usage:", proc.stderr, args)
        self.assertIn("error:", proc.stderr, args)

    def test_option_errors_exit_2(self):
        """입력 옵션 누락·중복·값 없음, 양의 정수가 아닌 `--max-frames`, 빈 `--source-id`는 종료 코드 2다."""
        list_path = self._list_file([self.a])

        for args in (
            [],                                              # 둘 다 없음
            ["--images", self.a, "--image-list", list_path],  # 둘 다 지정
            ["--images"],                                    # 값 없음
            ["--images", self.a, "--max-frames", "0"],
            ["--images", self.a, "--max-frames", "-1"],
            ["--images", self.a, "--max-frames", "1.5"],
            ["--images", self.a, "--max-frames", "x"],
            ["--images", self.a, "--source-id", ""],
        ):
            self.assertUsageError(args)

    def test_list_file_errors_exit_2(self):
        """목록 파일이 없거나 폴더이거나 UTF-8이 아니면 종료 코드 2다."""
        bad_utf8 = os.path.join(self.dir, "bad.txt")

        with open(bad_utf8, "wb") as f:
            f.write(b"a.png\n\xff\xfe\xfa\n")

        for path in (os.path.join(self.dir, "nolist.txt"), self.dir, bad_utf8):
            self.assertUsageError(["--image-list", path])


class ExceptionPathTest(unittest.TestCase):
    """예외 경로는 모듈을 직접 불러 가짜 소스를 주입해 확인한다(실제 입력 시험은 위 subprocess)."""

    @classmethod
    def setUpClass(cls):
        """`run_vision.py`를 모듈로 직접 불러 온다(src를 sys.path에 추가)."""
        sys.path.insert(0, SRC)
        spec = importlib.util.spec_from_file_location("run_vision_under_test", SCRIPT)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    @classmethod
    def tearDownClass(cls):
        """추가한 src 경로를 sys.path에서 뺀다."""
        if SRC in sys.path:
            sys.path.remove(SRC)

    def test_unexpected_error_closes_and_exits_1(self):
        """소스 읽기에서 예상 못 한 예외가 나면 소스를 닫고 종료 코드 1이며, 표준 오류에 예외를 남기고 end를 쓰지 않는다."""
        closed = []

        class ExplodingSource:
            """`read()`마다 RuntimeError를 내고 `close()` 호출을 기록하는 가짜 소스."""

            def __init__(self, paths, **kw):
                """인자를 무시한다."""
                pass

            def read(self):
                """항상 RuntimeError("boom")를 낸다."""
                raise RuntimeError("boom")

            def close(self):
                """닫힘을 기록한다."""
                closed.append(True)

        out, err = io.StringIO(), io.StringIO()
        code = self.mod.main(["--images", "x.png"], source_factory=ExplodingSource,
                             stdout=out, stderr=err)

        self.assertEqual(code, 1)
        self.assertEqual(closed, [True])
        self.assertIn("boom", err.getvalue())

        kinds = [json.loads(line)["kind"] for line in out.getvalue().splitlines()]

        self.assertNotIn("end", kinds)

    def test_normal_run_closes_source(self):
        """예외 없이 끝난 실행(없는 파일 하나)도 종료 코드 0이고 소스를 한 번 닫는다."""
        closed = []
        real = self.mod.StoredImageSource

        class Tracking(real):
            """실제 저장 소스에 `close()` 기록만 덧붙인 소스."""

            def close(self):
                """닫힘을 기록하고 부모 클래스의 닫기를 부른다."""
                closed.append(True)
                super().close()

        out, err = io.StringIO(), io.StringIO()
        code = self.mod.main(["--images", "does-not-exist.png"], source_factory=Tracking,
                             stdout=out, stderr=err)

        self.assertEqual(code, 0)
        self.assertEqual(closed, [True])


class PublicSampleCliTest(unittest.TestCase):
    """실제 공개 데이터 표본으로 CLI를 확인한다."""

    def test_public_samples_cli(self):
        """공개 데이터 표본 3장이 저장소 루트 기준 상대 경로로 640×640 프레임 3개가 되고 정상 종료한다(파일이 없으면 UNVERIFIED로 건너뜀)."""
        missing = [p for p in PUBLIC_SAMPLES if not os.path.isfile(os.path.join(REPO_ROOT, p))]

        if missing:
            self.skipTest(f"UNVERIFIED: public samples missing: {missing}")

        proc = run_cli(["--images", *PUBLIC_SAMPLES], cwd=REPO_ROOT)

        self.assertEqual(proc.returncode, 0, proc.stderr)

        recs = parse_lines(proc.stdout)
        frames = [r for r in recs if r["kind"] == "frame"]

        self.assertEqual([f["source_ref"] for f in frames], list(PUBLIC_SAMPLES))
        self.assertEqual({(f["width"], f["height"]) for f in frames}, {(640, 640)})
        self.assertEqual((recs[-1]["frames"], recs[-1]["failures"], recs[-1]["reason"]),
                         (3, 0, "end_of_stream"))


# ---------------------------------------------------------------- 검출 재현 모드(--detect-config)
# 기대 좌표·면적은 손계산 상수다. cv2.findContours의 직사각형 윤곽은 화소 중심 꼭짓점이므로
# 화소 x0..x1, y0..y1을 채운 사각의 넓이는 (x1-x0)*(y1-y0)이다. 출력은 A 내부 표시용 기록이다.

DETECT_CFG = {
    "format_version": 1, "status": "provisional_display_only", "roi": None, "resize_wh": None,
    "resize_interpolation": "INTER_AREA", "gaussian_blur_ksize": 0, "polarity": "dark",
    "threshold_method": "fixed", "fixed_threshold": 128, "morph_open_ksize": 0,
    "contour_retrieval": "RETR_EXTERNAL", "contour_approximation": "CHAIN_APPROX_SIMPLE",
    "min_area_source_px2": 100, "max_area_source_px2": None, "border_margin_processed_px": 2,
}
REPO_CONFIG = os.path.join(REPO_ROOT, "config", "vision", "opencv_planar.json")
REPO_CONFIG_SHA = "b54dec5729bf5dc4f5af1ae0921ede34a5db3124050e9dfe6c110571c1e51422"  # REPO_CONFIG의 식별자
OBS_KEYS = {"kind", "run_id", "source_id", "image_seq", "capture_generation", "timestamp_ms",
            "timestamp_base", "received_monotonic_ns", "config_sha256", "processed_wh", "transform",
            "threshold_used", "raw_contour_count", "eligible_count", "process_ms", "display_only",
            "valid", "observed_face", "calc_contour", "selected_index", "candidates", "overlay_path"}
CAND_KEYS = {"index", "points_source", "area_source_px2", "bbox_source", "border_sides",
             "clip_reasons", "area_reject_reason", "eligible"}
ELIGIBLE_BGR = (0, 200, 0)   # 스크립트의 표시 색(면적 통과·경계 미접촉)
REJECT_BGR = (0, 0, 255)     # 탈락·잘림


def file_bytes(path):
    """파일 내용을 바이트로 읽는다."""
    with open(path, "rb") as f:
        return f.read()


class DetectCliBase(CliTestBase):
    """검출 재현 모드(`--detect-config`) 시험용 합성 영상과 설정 파일을 준비하는 기반."""

    def setUp(self):
        """사각형 두 개 영상, 빈 영상, 왼쪽 변 접촉 사각형과 2×2 잡음 영상, 기본 검출 설정 파일을 만든다."""
        super().setUp()
        two = np.full((80, 120, 3), 255, np.uint8)
        two[10:30, 10:30] = 0          # 화소 10..29: 넓이 19*19 = 361
        two[20:60, 60:100] = 0         # 화소 60..99 × 20..59: 넓이 39*39 = 1521
        self.two = self._img("two.png", two)
        self.blank = self._img("blank.png", np.full((80, 100, 3), 255, np.uint8))
        edge = np.full((80, 100, 3), 255, np.uint8)
        edge[20:40, 0:20] = 0          # 원본 왼쪽 변 접촉, 넓이 361
        edge[60:62, 60:62] = 0         # 2×2 잡음, 넓이 1
        self.edge = self._img("edge.png", edge)
        self.cfg = self._cfg("cfg.json", DETECT_CFG)

    def _img(self, name, image):
        """영상을 임시 폴더에 저장하고 경로를 돌려준다."""
        path = os.path.join(self.dir, name)
        self.assertTrue(cv2.imwrite(path, image))
        return path

    def _cfg(self, name, doc=None, text=None):
        """dict(JSON으로) 또는 원문 텍스트를 설정 파일로 쓰고 경로를 돌려준다."""
        path = os.path.join(self.dir, name)

        with open(path, "w", encoding="utf-8") as f:
            f.write(text if text is not None else json.dumps(doc))

        return path

    def observations(self, recs):
        """레코드 중 observation만 골라 돌려준다."""
        return [r for r in recs if r["kind"] == "observation"]


class DetectOutputTest(DetectCliBase):
    """검출 재현 모드의 observation 출력 계약을 확인한다."""

    def test_no_option_output_has_no_detection(self):
        """`--detect-config`가 없으면 observation 없이 기존 출력 형식 그대로다."""
        recs = self.assertOk(run_cli(["--images", self.two, self.missing], cwd=self.dir))

        self.assertEqual([r["kind"] for r in recs], ["start", "frame", "event", "event", "end"])
        self.assertEqual(set(recs[0]), START_KEYS)
        self.assertEqual(set(recs[1]), FRAME_KEYS)
        self.assertEqual(set(recs[-1]), END_KEYS)

    def test_frame_then_observation_and_no_observation_for_events(self):
        """프레임마다 바로 뒤에 observation이 오고 사건에는 오지 않는다. observation의 식별·시각·설정·후보 값이 손 계산과 같고 end에 observation 수가 붙는다."""
        recs = self.assertOk(run_cli(
            ["--images", self.two, self.missing, self.broken, "--detect-config", self.cfg], cwd=self.dir))

        self.assertEqual([r["kind"] for r in recs],
                         ["start", "frame", "observation", "event", "event", "event", "end"])

        start, frame, obs, end = recs[0], recs[1], recs[2], recs[-1]

        self.assertEqual(set(start), START_KEYS | {"config_sha256", "display_only"})
        self.assertTrue(start["display_only"])
        self.assertEqual(set(frame), FRAME_KEYS)
        self.assertEqual(set(obs), OBS_KEYS)

        for k in ("run_id", "source_id", "image_seq", "timestamp_ms", "timestamp_base",
                  "received_monotonic_ns"):
            self.assertEqual(obs[k], frame[k], k)

        self.assertEqual(obs["capture_generation"], 0)
        self.assertEqual(obs["config_sha256"], start["config_sha256"])
        self.assertEqual(obs["processed_wh"], [120, 80])
        self.assertEqual(obs["transform"], [1.0, 1.0, 0.0, 0.0])
        self.assertEqual(obs["threshold_used"], 128.0)
        self.assertEqual((obs["raw_contour_count"], obs["eligible_count"]), (2, 2))
        self.assertIsInstance(obs["process_ms"], float)
        self.assertEqual((obs["display_only"], obs["valid"]), (True, False))

        for k in ("observed_face", "calc_contour", "selected_index", "overlay_path"):
            self.assertIsNone(obs[k], k)

        by_area = {c["area_source_px2"]: c for c in obs["candidates"]}

        self.assertEqual(sorted(by_area), [361.0, 1521.0])

        small, big = by_area[361.0], by_area[1521.0]

        for c in (small, big):
            self.assertEqual(set(c), CAND_KEYS)
            self.assertTrue(c["eligible"])
            self.assertEqual((c["border_sides"], c["clip_reasons"], c["area_reject_reason"]), ([], [], None))

        self.assertEqual(small["bbox_source"], [10.0, 10.0, 29.0, 29.0])
        self.assertEqual(sorted(map(tuple, small["points_source"])),
                         [(10.0, 10.0), (10.0, 29.0), (29.0, 10.0), (29.0, 29.0)])
        self.assertEqual(big["bbox_source"], [60.0, 20.0, 99.0, 59.0])
        self.assertEqual(sorted(c["index"] for c in obs["candidates"]), [0, 1])
        self.assertEqual((end["frames"], end["failures"], end["reason"], end["observations"]),
                         (1, 2, "end_of_stream", 1))
        self.assertEqual(set(end), END_KEYS | {"observations"})

    def test_blank_and_edge_candidates(self):
        """빈 영상은 후보 없음, 변에 닿은 사각형은 `source_edge`로 부적격, 2×2 잡음은 `area_below_min`으로 부적격이다."""
        recs = self.assertOk(run_cli(
            ["--images", self.blank, self.edge, "--detect-config", self.cfg], cwd=self.dir))
        blank, edge = self.observations(recs)

        self.assertEqual((blank["raw_contour_count"], blank["eligible_count"], blank["candidates"]), (0, 0, []))
        self.assertEqual((blank["valid"], blank["selected_index"]), (False, None))
        self.assertEqual((edge["raw_contour_count"], edge["eligible_count"]), (2, 0))
        self.assertEqual((edge["valid"], edge["selected_index"]), (False, None))

        by_area = {c["area_source_px2"]: c for c in edge["candidates"]}
        clipped, noise = by_area[361.0], by_area[1.0]

        self.assertEqual((clipped["border_sides"], clipped["clip_reasons"], clipped["area_reject_reason"],
                          clipped["eligible"]), (["left"], ["source_edge"], None, False))
        self.assertEqual(clipped["bbox_source"], [0.0, 20.0, 19.0, 39.0])
        self.assertEqual((noise["area_reject_reason"], noise["eligible"], noise["border_sides"]),
                         ("area_below_min", False, []))

        end = recs[-1]

        self.assertEqual((end["frames"], end["failures"], end["observations"]), (2, 0, 2))

    def test_repo_config_hash_and_max_frames(self):
        """저장소 설정을 쓰면 start·observation에 고정된 설정 식별자가 나오고, `--max-frames`와 함께 써도 집계가 맞다."""
        recs = self.assertOk(run_cli(
            ["--images", self.two, self.blank, "--detect-config", REPO_CONFIG, "--max-frames", "1"],
            cwd=self.dir))

        self.assertEqual([r["kind"] for r in recs], ["start", "frame", "observation", "end"])
        self.assertEqual(recs[0]["config_sha256"], REPO_CONFIG_SHA)
        self.assertEqual(recs[2]["config_sha256"], REPO_CONFIG_SHA)
        self.assertEqual((recs[-1]["frames"], recs[-1]["reason"], recs[-1]["observations"]),
                         (1, "max_frames", 1))

    def test_stdout_is_strict_json(self):
        """검출 모드 표준 출력의 모든 줄은 비표준 상수(NaN 등) 없는 JSON이다."""
        proc = run_cli(["--images", self.two, "--detect-config", self.cfg], cwd=self.dir)

        self.assertEqual(proc.returncode, 0, proc.stderr)

        for line in proc.stdout.splitlines():
            json.loads(line, parse_constant=lambda c: self.fail(f"non-standard constant {c}"))


class DetectErrorTest(DetectCliBase):
    """검출·오버레이 옵션 오류의 종료 코드를 확인한다."""

    def assertUsage2(self, args):
        """종료 코드 2, 표준 출력 없음, 표준 오류에 `error:`가 있는지 확인하고 실행 결과를 돌려준다."""
        proc = run_cli(args, cwd=self.dir)
        self.assertEqual(proc.returncode, 2, f"{args}: {proc.stdout} {proc.stderr}")
        self.assertEqual(proc.stdout, "", args)
        self.assertIn("error:", proc.stderr, args)

        return proc

    def test_config_errors_exit_2_before_start(self):
        """설정 파일이 없거나 폴더이거나 JSON 문법·키·값이 틀리면 start를 쓰기 전에 종료 코드 2다."""
        broken = self._cfg("broken.json", text="{not json")
        unknown = self._cfg("unknown.json", {**DETECT_CFG, "extra": 1})
        badval = self._cfg("badval.json", {**DETECT_CFG, "gaussian_blur_ksize": 2})

        for path in (os.path.join(self.dir, "nocfg.json"), broken, unknown, badval, self.dir):
            with self.subTest(path=path):
                self.assertUsage2(["--images", self.two, "--detect-config", path])

    def test_roi_outside_frame_exit_1(self):
        """설정은 맞지만 ROI가 실제 영상 밖이면 start·frame까지 쓰고 종료 코드 1이며 표준 오류에 roi를 남긴다."""
        cfg = self._cfg("roi.json", {**DETECT_CFG, "roi": [100, 0, 30, 80]})   # 120 < 100+30
        proc = run_cli(["--images", self.two, "--detect-config", cfg], cwd=self.dir)

        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertIn("roi", proc.stderr)

        kinds = [r["kind"] for r in parse_lines(proc.stdout)]

        self.assertEqual(kinds, ["start", "frame"])

    def test_overlay_requires_detect_config(self):
        """`--overlay-dir`만 주고 `--detect-config`가 없으면 종료 코드 2이고 폴더를 만들지 않는다."""
        self.assertUsage2(["--images", self.two, "--overlay-dir", os.path.join(self.dir, "ov")])
        self.assertFalse(os.path.exists(os.path.join(self.dir, "ov")))

    def test_overlay_existing_or_bad_path_exit_2(self):
        """오버레이 폴더가 이미 있거나 파일이거나 부모가 없으면 종료 코드 2이고, 기존 폴더 내용은 그대로다."""
        existing = os.path.join(self.dir, "exists")
        os.mkdir(existing)
        keep = os.path.join(existing, "keep.txt")

        with open(keep, "w") as f:
            f.write("keep")

        for path in (existing, self.a, os.path.join(self.dir, "no", "parent")):
            with self.subTest(path=path):
                self.assertUsage2(["--images", self.two, "--detect-config", self.cfg, "--overlay-dir", path])

        self.assertEqual(os.listdir(existing), ["keep.txt"])
        self.assertEqual(file_bytes(keep), b"keep")


class OverlayTest(DetectCliBase):
    """오버레이 영상 저장을 확인한다."""

    def test_overlay_written_and_inputs_preserved(self):
        """프레임마다 번호 파일명의 오버레이가 원본 크기로 저장되고, 적격·탈락 후보 색이 다르며 배경과 입력 파일은 바뀌지 않는다."""
        before = {p: file_bytes(p) for p in (self.edge, self.two, self.cfg)}
        ov = os.path.join(self.dir, "ov")
        recs = self.assertOk(run_cli(
            ["--images", self.edge, self.two, "--detect-config", self.cfg, "--overlay-dir", ov],
            cwd=self.dir))

        obs = self.observations(recs)

        self.assertEqual([o["overlay_path"] for o in obs],
                         [os.path.join(ov, "overlay_000000.png"), os.path.join(ov, "overlay_000001.png")])
        self.assertEqual(sorted(os.listdir(ov)), ["overlay_000000.png", "overlay_000001.png"])

        for p in before:
            self.assertEqual(file_bytes(p), before[p], p)

        edge_ov = cv2.imread(obs[0]["overlay_path"], cv2.IMREAD_COLOR)
        two_ov = cv2.imread(obs[1]["overlay_path"], cv2.IMREAD_COLOR)

        self.assertEqual(edge_ov.shape, (80, 100, 3))
        self.assertEqual(two_ov.shape, (80, 120, 3))

        # 윤곽 꼭짓점 사이 변 위의 화소(원본 좌표 그대로 정수화): 통과 후보와 잘림 후보 색이 다르다.
        self.assertEqual(tuple(int(v) for v in two_ov[20, 10]), ELIGIBLE_BGR)     # (x=10, y=20)
        self.assertEqual(tuple(int(v) for v in edge_ov[30, 19]), REJECT_BGR)      # (x=19, y=30)

        # 윤곽에서 먼 배경 화소는 원본 그대로(흰색)
        self.assertEqual(tuple(int(v) for v in two_ov[70, 40]), (255, 255, 255))

        # 좌표 JSON은 float 그대로
        for c in obs[1]["candidates"]:
            for pt in c["points_source"]:
                self.assertTrue(all(isinstance(v, float) for v in pt))

    def test_overlay_filename_does_not_use_source_ref(self):
        """오버레이 파일명은 입력 경로나 `--source-id`(공백·`;`·`../` 포함)를 쓰지 않고 순번만 쓴다."""
        weird = self._img("we ird;name.png", np.full((10, 10, 3), 255, np.uint8))
        ov = os.path.join(self.dir, "ov2")

        recs = self.assertOk(run_cli(
            ["--images", weird, "--detect-config", self.cfg, "--overlay-dir", ov, "--source-id", "../x"],
            cwd=self.dir))

        self.assertEqual(os.listdir(ov), ["overlay_000000.png"])
        self.assertEqual(self.observations(recs)[0]["overlay_path"], os.path.join(ov, "overlay_000000.png"))


class DetectExceptionPathTest(unittest.TestCase):
    """검출·저장 실패 경로. 실제 저장 소스를 쓰고, 실패는 실제 조건(폴더 삭제)이나 검출 함수 교체로 만든다."""

    setUpClass = classmethod(ExceptionPathTest.setUpClass.__func__)
    tearDownClass = classmethod(ExceptionPathTest.tearDownClass.__func__)

    def setUp(self):
        """임시 폴더에 검은 사각형 영상과 기본 검출 설정 파일을 만든다."""
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        img = np.full((40, 40, 3), 255, np.uint8)
        img[10:20, 10:20] = 0
        self.img = os.path.join(self.dir, "x.png")
        self.assertTrue(cv2.imwrite(self.img, img))

        self.cfg = os.path.join(self.dir, "cfg.json")

        with open(self.cfg, "w", encoding="utf-8") as f:
            json.dump(DETECT_CFG, f)

    def tearDown(self):
        """임시 폴더를 지운다."""
        self._tmp.cleanup()

    def tracking(self, closed, on_read=None):
        """`close()`를 `closed`에 기록하고, 읽은 직후 `on_read`를 부르는 실제 저장 소스 하위 클래스를 만든다."""
        real = self.mod.StoredImageSource

        class Tracking(real):
            """실제 저장 소스에 읽기 후 동작과 닫힘 기록을 덧붙인 소스."""

            def read(self):
                """실제로 읽은 뒤 `on_read`가 있으면 부른다."""
                item = super().read()
                if on_read is not None:
                    on_read()
                return item

            def close(self):
                """닫힘을 기록하고 부모 클래스의 닫기를 부른다."""
                closed.append(True)
                super().close()

        return Tracking

    def test_detect_exception_closes_and_exits_1(self):
        """검출 함수가 예외를 내면 소스를 닫고 종료 코드 1이며, 출력은 start·frame에서 멈춘다."""
        closed = []
        original = self.mod.detect_planar

        def boom(frame, config):
            """항상 RuntimeError를 낸다(검출 실패 주입)."""
            raise RuntimeError("detect boom")

        # main()은 모듈 전역 이름으로 검출 함수를 찾으므로 모듈 속성을 바꾼다(finally에서 되돌림).
        self.mod.detect_planar = boom

        try:
            out, err = io.StringIO(), io.StringIO()
            code = self.mod.main(["--images", self.img, "--detect-config", self.cfg],
                                 source_factory=self.tracking(closed), stdout=out, stderr=err)
        finally:
            self.mod.detect_planar = original

        self.assertEqual(code, 1)
        self.assertEqual(closed, [True])
        self.assertIn("detect boom", err.getvalue())

        kinds = [json.loads(line)["kind"] for line in out.getvalue().splitlines()]

        self.assertEqual(kinds, ["start", "frame"])

    def test_overlay_save_failure_closes_and_exits_1(self):
        """오버레이 저장이 실제로 실패하면 소스를 닫고 종료 코드 1이며, observation·end를 쓰지 않는다."""
        closed = []
        ov = os.path.join(self.dir, "ov")

        # 프레임을 읽은 직후 오버레이 폴더를 지워 실제 쓰기 실패를 만든다.
        source = self.tracking(closed, on_read=lambda: os.path.isdir(ov) and os.rmdir(ov))
        out, err = io.StringIO(), io.StringIO()

        code = self.mod.main(["--images", self.img, "--detect-config", self.cfg, "--overlay-dir", ov],
                             source_factory=source, stdout=out, stderr=err)

        self.assertEqual(code, 1)
        self.assertEqual(closed, [True])
        self.assertIn("overlay", err.getvalue())

        kinds = [json.loads(line)["kind"] for line in out.getvalue().splitlines()]

        self.assertNotIn("observation", kinds)
        self.assertNotIn("end", kinds)

    def test_no_option_does_not_load_config(self):
        """`--detect-config`가 없으면 설정 로더를 부르지 않는다."""
        closed = []
        original = self.mod.load_config

        def fail(path):
            """불리면 AssertionError를 낸다."""
            raise AssertionError("load_config called without --detect-config")

        # 모듈 전역 설정 로더를 불리면 실패하는 함수로 바꾼다(finally에서 되돌림).
        self.mod.load_config = fail

        try:
            out, err = io.StringIO(), io.StringIO()
            code = self.mod.main(["--images", self.img], source_factory=self.tracking(closed),
                                 stdout=out, stderr=err)
        finally:
            self.mod.load_config = original

        self.assertEqual(code, 0, err.getvalue())

        kinds = [json.loads(line)["kind"] for line in out.getvalue().splitlines()]

        self.assertEqual(kinds, ["start", "frame", "event", "end"])


if __name__ == "__main__":
    unittest.main()
