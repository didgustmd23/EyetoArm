#!/usr/bin/env python3
"""A 비전 파이프라인 저장 입력 실행 CLI(계획 r4 작업 6a). 기존 `run_vision.py`는 그대로 둔다.

사용 예(작업 디렉터리 저장소 루트):
    PYTHONPATH=src .venv/bin/python scripts/run_pipeline.py --images a.png b.png \\
        --detect-config config/vision/planar.json --log outputs/run.jsonl
    PYTHONPATH=src .venv/bin/python scripts/run_pipeline.py --video clip.avi \\
        --detect-config cfg.json --log run.jsonl --max-frames 30

- 입력 하나(`--images` 또는 `--video`), 검출 설정 JSON(`load_config`), 기록 파일(JSONL)을 받는다.
- 처리 프로파일은 없다(저장 입력은 보정 프로파일이 없음). 그래서 표시용 관측의 사유는 `CALIBRATION_MISMATCH`이며
  B 소비 지점은 쓰지 않는다. 탐색 옵션은 없다(계획 범위).
- C 서버·ROS·YOLO를 연결하지 않는다. 영상 파일 열기 제한시간을 두지 않는다(사용자 결정 2026-11-13).
- 검출 감시(작업 6b): 기본으로 켠다. 첫 검출 10 s·이후 1 s(사용자 확정 개발 초기값, 운영 최종값 아님)이며
  `--first-detect-timeout`·`--detect-timeout`(초)으로 조절한다. `--no-detect-timeout`은 6a 동기 경로(감시 끔)다.
  유예·정리 한도·확인 간격은 A 내부 임시 기본안(0.2/0.3/0.02 s)이며 CLI 옵션이 없다.
- 종료 코드: 0 정상(EOF·max-frames), 1 실행 실패(입력 없음·기록 실패·중단·close 실패·검출 시간 초과·멈춤 등),
  2 인자·설정 오류.
"""
from __future__ import annotations

import argparse
import math
import sys

import cv2

from eyetoarm.vision.pipeline_records import JsonlRecorder, PipelineLog
from eyetoarm.vision.pipeline_runner import RunnerConfig, run, stored_source_factory
from eyetoarm.vision.pipeline_types import PipelineSettings
from eyetoarm.vision.planar import load_config


def _positive_int(text: str) -> int:
    """양의 정수 인자. 아니면 argparse 오류(종료 2)."""
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"양의 정수가 아님: {text!r}") from exc

    if value < 1:
        raise argparse.ArgumentTypeError(f"양의 정수가 아님: {text!r}")

    return value


def _positive_seconds(text: str) -> float:
    """유한한 양의 초. 아니면 argparse 오류(종료 2)."""
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"유한한 양수(초)가 아님: {text!r}") from exc

    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"유한한 양수(초)가 아님: {text!r}")

    return value


def _non_empty(text: str) -> str:
    """빈 문자열이 아닌 인자."""
    if not text:
        raise argparse.ArgumentTypeError("빈 문자열은 허용하지 않음")

    return text


def build_parser() -> argparse.ArgumentParser:
    """명령행 해석기."""
    parser = argparse.ArgumentParser(description="A 비전 파이프라인 저장 입력 실행기(표시용 OpenCV, JSONL 기록)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--images", nargs="+", metavar="PATH", help="읽을 이미지 경로(인자 순서대로)")
    group.add_argument("--video", metavar="PATH", help="읽을 저장 영상 하나")
    parser.add_argument("--detect-config", metavar="JSON", required=True, help="검출 설정 JSON(PlanarConfig)")
    parser.add_argument("--log", metavar="PATH", required=True, help="JSONL 기록 파일(새로 씀)")
    parser.add_argument("--source-id", type=_non_empty, default="stored0", help="소스 ID(기본 stored0)")
    parser.add_argument("--mode", type=_non_empty, default="planar", help="A 내부 처리 모드(기본 planar)")
    parser.add_argument("--max-frames", type=_positive_int, default=None, help="받을 프레임 수 상한")
    parser.add_argument("--eval-run-id", type=_non_empty, default=None,
                        help="평가 도구가 발급한 평가 실행 ID(평가할 때만)")
    parser.add_argument("--first-detect-timeout", type=_positive_seconds, default=None, metavar="SEC",
                        help=f"첫 검출 시간 상한(기본 {RunnerConfig.first_detect_timeout_s} s, 개발 초기값)")
    parser.add_argument("--detect-timeout", type=_positive_seconds, default=None, metavar="SEC",
                        help=f"두 번째부터의 검출 시간 상한(기본 {RunnerConfig.detect_timeout_s} s, 개발 초기값)")
    parser.add_argument("--no-detect-timeout", action="store_true",
                        help="검출 감시를 끈다(6a 동기 호출). 위 두 옵션과 함께 쓸 수 없다")

    return parser


def build_runner_config(args) -> RunnerConfig:
    """해석한 인자로 실행 설정을 만든다. 상한을 주지 않으면 RunnerConfig 기본값(10/1)을 그대로 쓴다."""
    if args.no_detect_timeout:
        if args.first_detect_timeout is not None or args.detect_timeout is not None:
            build_parser().error("--no-detect-timeout cannot be combined with --first-detect-timeout/--detect-timeout")

        return RunnerConfig(max_frames=args.max_frames, first_detect_timeout_s=None, detect_timeout_s=None)

    over = {}

    if args.first_detect_timeout is not None:
        over["first_detect_timeout_s"] = args.first_detect_timeout

    if args.detect_timeout is not None:
        over["detect_timeout_s"] = args.detect_timeout

    return RunnerConfig(max_frames=args.max_frames, **over)


def main(argv=None, *, stderr=None) -> int:
    """CLI 진입점. 종료 코드를 돌려준다."""
    err = stderr if stderr is not None else sys.stderr
    args = build_parser().parse_args(argv)  # 인자 오류는 argparse가 종료 2로 끝낸다
    runner_config = build_runner_config(args)

    try:
        config = load_config(args.detect_config)
        settings = PipelineSettings(source_id=args.source_id, mode=args.mode, detector_config=config, profile=None)
    except (OSError, ValueError) as exc:
        err.write(f"run_pipeline: config error: {exc}\n")
        return 2

    kind, paths = ("images", list(args.images)) if args.images else ("video", [args.video])

    try:
        stream = open(args.log, "w", encoding="utf-8")
    except OSError as exc:
        err.write(f"run_pipeline: cannot open log: {exc}\n")
        return 2

    with stream:
        log = PipelineLog(JsonlRecorder(stream), eval_run_id=args.eval_run_id)
        result = run(stored_source_factory(kind, paths, args.source_id), settings, log,
                     config=runner_config, stderr=err,
                     versions={"opencv": cv2.__version__})

    watch = (f"{runner_config.first_detect_timeout_s}/{runner_config.detect_timeout_s}" if runner_config.watchdog
             else "off")
    err.write(f"run_pipeline: exit={result.exit_code} reason={result.stop_reason} frames={result.frames} "
              f"input_errors={result.input_errors} detect_timeout={watch} timeouts={result.timeouts} "
              f"stalled={result.stalled}\n")

    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
