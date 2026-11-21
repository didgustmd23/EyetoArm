#!/usr/bin/env python3
"""링크 카메라 실시간 입력을 AI 검출로 돌려 원본·오버레이를 창에 보여 주는 확인 도구.

사용: PYTHONPATH=src .venv/bin/python scripts/view_live_detection.py \\
      --ai-config config/vision/ai_s1_01_live.json
종료 코드: runner 종료 코드(사용자 정지만이면 0), 창 실패·runner 미종료 1, 인자·창 사용 불가 2.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import threading
import time
import traceback

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_pipeline import _StopSignals, _live_factory, _non_empty, _positive_int, _positive_seconds  # noqa: E402

from eyetoarm.vision.ai_config import load_ai_config  # noqa: E402
from eyetoarm.vision.live_view import DisplayBridge, render_view  # noqa: E402
from eyetoarm.vision.pipeline_records import JsonlRecorder, PipelineLog  # noqa: E402
from eyetoarm.vision.pipeline_runner import RunnerConfig, run  # noqa: E402
from eyetoarm.vision.pipeline_types import PipelineSettings  # noqa: E402
from eyetoarm.vision.tracking import PartTracker, PartTrackIdIssuer, TrackMode  # noqa: E402

WINDOW_NAME = "EyetoArm A live detection (display only)"
QUIT_KEYS = (27, ord("q"), ord("Q"))


class GuiFailure(Exception):
    """창 그리기·이벤트 처리 실패(원래 예외를 감쌈)."""


class CvWindow:
    """OpenCV highgui 창 어댑터. 메인 스레드에서만 부른다."""

    def __init__(self, name: str = WINDOW_NAME) -> None:
        """창 이름만 정한다(아직 만들지 않음)."""
        self.name = name
        self._shown = False

    def open(self) -> None:
        """창을 만든다. 실패는 예외 그대로."""
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)

    def show(self, image) -> None:
        """영상 하나를 그린다."""
        cv2.imshow(self.name, image)
        self._shown = True

    def poll(self, interval_ms: int) -> int | None:
        """창 이벤트를 최대 interval_ms 처리하고 눌린 키(없으면 None)를 돌려준다."""
        key = cv2.waitKey(interval_ms)

        return None if key < 0 else key & 0xFF

    def closed(self) -> bool:
        """사용자가 창을 닫았는가(한 번이라도 그린 뒤에만 판정)."""
        return self._shown and cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) < 1

    def close(self) -> None:
        """창을 없앤다."""
        cv2.destroyWindow(self.name)
        cv2.waitKey(1)


def display_available() -> str | None:
    """창을 띄울 수 있는지. 안 되면 사유 문자열(디스플레이 없이 창을 만들면 프로세스가 끝날 수 있음)."""
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return "no display: DISPLAY/WAYLAND_DISPLAY is not set (run in a desktop/WSLg session)"

    return None


def build_parser() -> argparse.ArgumentParser:
    """명령행 해석기."""
    parser = argparse.ArgumentParser(description="A 실시간 검출 확인 창(표시용 검출, 로봇 명령 없음)")
    parser.add_argument("--ai-config", metavar="JSON", default="config/vision/ai_s1_01_live.json",
                        help="AI 분할 설정 JSON(가중치·conf 등, 기본 S1-01 실시간 설정)")
    parser.add_argument("--log", metavar="PATH", default=None,
                        help="JSONL 기록 파일(기본 outputs/live_view/live_view_<시각>.jsonl)")
    parser.add_argument("--source-id", type=_non_empty, default="gz_link", help="소스 ID(기본 gz_link)")
    parser.add_argument("--ros-reliability", choices=("reliable", "best_effort"), default=None,
                        help="영상 구독 신뢰성(기본 reliable)")
    parser.add_argument("--max-frames", type=_positive_int, default=None, help="받을 프레임 수 상한")
    parser.add_argument("--first-detect-timeout", type=_positive_seconds, default=None, metavar="SEC",
                        help=f"첫 검출 상한(기본 {RunnerConfig.first_detect_timeout_s} s, 개발 초기값)")
    parser.add_argument("--detect-timeout", type=_positive_seconds, default=None, metavar="SEC",
                        help=f"이후 검출 상한(기본 {RunnerConfig.detect_timeout_s} s, 개발 초기값)")
    parser.add_argument("--model-load-timeout", type=_positive_seconds, default=None, metavar="SEC",
                        help=f"모델 준비 상한(기본 {RunnerConfig.model_load_timeout_s} s, A 임시 개발값)")
    parser.add_argument("--max-width", type=_positive_int, default=1600, help="창 영상 최대 폭(px, 기본 1600)")
    parser.add_argument("--key-interval-ms", type=_positive_int, default=30, help="창 이벤트 처리 주기(ms, 기본 30)")
    parser.add_argument("--join-timeout", type=_positive_seconds, default=5.0, metavar="SEC",
                        help="정지 요청 뒤 runner 반환 대기 상한(기본 5 s, A 임시 개발값)")
    parser.add_argument("--exit-on-end", action="store_true", help="runner가 끝나면 창을 바로 닫는다")
    parser.add_argument("--tracking", action="store_true", help="A-6 1차 실험: AI 검출 인스턴스 추적 번호를 붙인다(M2 휴리스틱, 표시·기록 전용, 팔 이동을 넘는 신원·중복 집기 방지 보장 없음, B로 보내지 않음). 기본 꺼짐")

    return parser


def _runner_config(args) -> RunnerConfig:
    """해석한 인자로 실행 설정을 만든다. 주지 않은 값은 RunnerConfig 기본."""
    over = {}

    for arg, key in (("first_detect_timeout", "first_detect_timeout_s"), ("detect_timeout", "detect_timeout_s"),
                     ("model_load_timeout", "model_load_timeout_s")):
        if getattr(args, arg) is not None:
            over[key] = getattr(args, arg)

    return RunnerConfig(max_frames=args.max_frames, **over)


def gui_loop(window, bridge: DisplayBridge, stop_requested, runner_alive, *, interval_ms: int,
             max_width: int, hold_on_end: bool, render=render_view) -> str:
    """창 메인 루프. 반환: "user"(ESC·q·닫기) | "signal"(첫 신호) | "ended"(runner 종료 뒤 끝냄).

    그리기 직전 버전을 다시 확인한다. 그리기·이벤트 실패는 `GuiFailure`. runner를 기다리지 않는다.
    """
    drawn = None

    while True:
        snap = bridge.snapshot()

        try:
            if snap.version != drawn:
                canvas = render(snap, max_width)

                if bridge.is_current(snap.version):
                    window.show(canvas)
                    drawn = snap.version

            canvas = snap = None   # 그린 영상·프레임 참조를 다음 대기 동안 붙잡지 않는다
            key = window.poll(interval_ms)
            closed = window.closed()
        except Exception as exc:
            raise GuiFailure(f"{type(exc).__name__}: {exc}") from exc

        if key in QUIT_KEYS or closed:
            return "user"

        if stop_requested():
            return "signal"

        if not runner_alive() and not hold_on_end and bridge.is_current(drawn if drawn is not None else -1):
            return "ended"


def _drain(window, deadline: float, alive, interval_ms: int, window_ok: bool) -> None:
    """정지 뒤 runner 반환을 기다리는 동안에도 창 이벤트를 처리한다(창이 실패했으면 그냥 기다림)."""
    while alive() and time.monotonic() < deadline:
        if window_ok:
            try:
                window.poll(interval_ms)
                continue
            except Exception:
                window_ok = False

        time.sleep(interval_ms / 1000)


def main(argv=None, *, stderr=None, window=None, display_check=display_available, ros_available=None,
         live_source_factory=None, model_factory=None, import_module=None, run_fn=run, signals=None) -> int:
    """진입점. 종료 코드를 돌려준다. 키워드 인자는 시험 주입 지점이다."""
    err = stderr if stderr is not None else sys.stderr
    args = build_parser().parse_args(argv)

    available = ros_available() if ros_available is not None else importlib.util.find_spec("rclpy") is not None

    if not available:
        err.write("view_live_detection: rclpy not found (run: source /opt/ros/jazzy/setup.bash)\n")
        return 2

    try:
        config = load_ai_config(args.ai_config)
        settings = PipelineSettings(source_id=args.source_id, mode="planar", detector_config=config, profile=None)
        runner_config = _runner_config(args)
    except (OSError, ValueError) as exc:
        err.write(f"view_live_detection: config error: {exc}\n")
        return 2

    # 창을 먼저 확인한다. 실패하면 ROS 수신·모델 준비를 시작하지 않는다
    reason = display_check() if display_check is not None else None

    if reason is not None:
        err.write(f"view_live_detection: {reason}\n")
        return 2

    window = window if window is not None else CvWindow()

    try:
        window.open()
    except Exception as exc:
        err.write(f"view_live_detection: cannot create window: {type(exc).__name__}: {exc}\n")
        return 2

    log_path = args.log or os.path.join("outputs", "live_view", f"live_view_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")

    try:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        stream = open(log_path, "w", encoding="utf-8")
    except OSError as exc:
        _close_window(window, err)
        err.write(f"view_live_detection: cannot open log: {exc}\n")
        return 2

    from eyetoarm.vision.ai_detector import AiSegDetector   # AI 어댑터는 이때만(생성자는 I/O 없음)

    inject = {"model_factory": model_factory}

    if import_module is not None:
        inject["import_module"] = import_module

    det = AiSegDetector(config, **inject)
    source_factory = (live_source_factory or _live_factory)(args.source_id, args.ros_reliability)
    bridge = DisplayBridge()
    stop = threading.Event()
    box: dict = {}

    # 추적 opt-in: 진입점이 프로세스당 발급기 1개를 만들어 주입한다
    track_kw = {}

    if args.tracking:
        track_kw["part_tracker"] = PartTracker(PartTrackIdIssuer(), mode=TrackMode.HEURISTIC)

    def runner_body(log) -> None:
        """runner 스레드: run 전체 → 표시 종료 알림. 예외도 남긴다."""
        try:
            box["result"] = run_fn(source_factory, settings, log, config=runner_config, stderr=err,
                                   versions={"opencv": cv2.__version__}, should_stop=stop_requested, detector=det,
                                   prepare=det.prepare, display_sink=bridge, **track_kw)
            bridge.finish(box["result"].exit_code, box["result"].stop_reason)
        except BaseException as exc:   # run은 예외를 내지 않는 계약이지만 스레드에서 사라지지 않게 보존한다
            box["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            bridge.finish(1, "runner_exception")

    how, gui_error, interrupted = None, None, False
    signals = signals if signals is not None else _StopSignals()

    def stop_requested() -> bool:
        """정지 요청: 창 쪽 정지(ESC·q·닫기·창 실패) 또는 첫 SIGINT/SIGTERM(신호 처리기의 표지)."""
        flag = getattr(signals, "stop", None)

        return stop.is_set() or (flag is not None and flag.is_set())

    with stream, signals:
        log = PipelineLog(JsonlRecorder(stream), eval_run_id=None, log_geometry=False)
        thread = threading.Thread(target=runner_body, args=(log,), name="eyetoarm-runner", daemon=True)
        thread.start()

        try:
            try:
                how = gui_loop(window, bridge, stop_requested, thread.is_alive,
                               interval_ms=args.key_interval_ms, max_width=args.max_width,
                               hold_on_end=not args.exit_on_end)
            except GuiFailure as exc:
                gui_error = str(exc)
                err.write(f"view_live_detection: display failed, stopping: {gui_error}\n")

            stop.set()
            bridge.request_stop()
            _drain(window, time.monotonic() + args.join_timeout, thread.is_alive, args.key_interval_ms,
                   window_ok=gui_error is None)
        except KeyboardInterrupt:
            interrupted = True
            stop.set()
            bridge.request_stop()
            thread.join(args.join_timeout)

        alive = thread.is_alive()

    _close_window(window, err)

    result = box.get("result")
    code = result.exit_code if result is not None else 1

    if box.get("error"):
        err.write(f"view_live_detection: runner raised: {box['error']}\n")
        code = 1

    if alive:
        err.write(f"view_live_detection: runner thread did not finish within {args.join_timeout} s after stop; "
                  "ROS receiver/detector worker cleanup not confirmed\n")
        code = 1

    if gui_error is not None or interrupted:
        code = 1

    summary = "-" if result is None else (f"exit={result.exit_code} reason={result.stop_reason} frames={result.frames} "
                                          f"timeouts={result.timeouts} stalled={result.stalled} "
                                          f"live_inputs={result.live_inputs}")
    err.write(f"view_live_detection: end={'interrupted' if interrupted else how or 'gui_failure'} code={code} "
              f"shown={bridge.shown} cleared={bridge.cleared} runner[{summary}] log={log_path}\n")

    return code


def _close_window(window, err) -> None:
    """창을 닫는다. 실패는 진단만 남긴다."""
    try:
        window.close()
    except Exception as exc:
        err.write(f"view_live_detection: window close failed: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
