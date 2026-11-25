#!/usr/bin/env python3
"""A 비전 파이프라인 실행 CLI: 저장 입력 또는 실시간 링크 카메라를 검출해 JSONL로 기록한다.

선택 옵션으로 A→B 경로·실시간 기록 연결·C 목표 송신·추적을 켠다(조합 검사는 `_check_source_args`).
사용: PYTHONPATH=src .venv/bin/python scripts/run_pipeline.py --images a.png --detector ai \\
      --ai-config config/vision/ai_s1_08_axis_m2_viewbias.json --log run.jsonl
종료 코드: 0 정상, 1 실행 실패, 2 인자·설정 오류, 3 시각 되감기로 끝남(정리 문제가 있으면 1 우선).
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import importlib.util
import math
import signal
import sys
import json
import threading
import time
from pathlib import Path

import cv2

from eyetoarm.vision.ai_config import load_ai_config
from eyetoarm.vision.pipeline_records import JsonlRecorder, PipelineLog
from eyetoarm.vision.pipeline_runner import RunnerConfig, run, stored_source_factory
from eyetoarm.vision.pipeline_types import PipelineSettings
from eyetoarm.vision.planar import load_config
from eyetoarm.vision.handoff import AToCHandoff
from eyetoarm.vision.c_handoff_probe import HandoffProbe
from eyetoarm.vision.tracking import PartTracker, PartTrackIdIssuer, TrackMode


ARM_REGION_DEFAULT_REL = "config/vision/arm_region_filter.json"   # 팔 영역 제외 기본 설정(실시간 기록 연결 기본 켬)
BELT_REGION_DEFAULT_REL = "config/vision/belt_region_filter.json"   # 벨트 밖 제외 기본 설정(실시간 기록 연결 기본 켬)


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
    group.add_argument("--ros-link", action="store_true",
                       help="ROS 2 링크 카메라 실시간 입력(/link_cam/image·camera_info·/clock)")
    parser.add_argument("--detector", choices=("opencv", "ai"), default="opencv",
                        help="검출 방식(기본 opencv). ai는 표시용 무효 관측만 낸다")
    parser.add_argument("--detect-config", metavar="JSON", default=None, help="OpenCV 검출 설정 JSON(PlanarConfig)")
    parser.add_argument("--ai-config", metavar="JSON", default=None, help="AI 분할 설정 JSON(AiSegConfig)")
    parser.add_argument("--model-load-timeout", type=_positive_seconds, default=None, metavar="SEC",
                        help=f"AI 모델 준비 상한(기본 {RunnerConfig.model_load_timeout_s} s, A 임시 개발값)")
    parser.add_argument("--log-instance-geometry", action="store_true",
                        help="AI 인스턴스 박스·다각형을 ai_geometry로 기록한다(선택)")
    parser.add_argument("--stage-timing", metavar="PATH", default=None,
                        help="진단: A 처리 단계별 단조 시각과 스레드 이름표를 PATH(JSONL, 새 파일)에 남긴다. "
                             "--live-observation-out 실시간 경로 전용, 기본 끔(#372)")
    parser.add_argument("--infer-process", dest="infer_process_on", action="store_true",
                        help="AI 추론(crop·predict·변환·끝점)을 별도 자식 프로세스에서 한다(--detector ai 전용). "
                             "기본: --ros-link --detector ai면 켬, 그 밖은 끔")
    parser.add_argument("--no-infer-process", dest="infer_process_off", action="store_true",
                        help="추론 자식 프로세스를 쓰지 않는다(실시간 AI 기본 켬을 끔)")
    parser.add_argument("--infer-torch-threads", type=_positive_int, default=None, metavar="N",
                        help="추론 자식의 torch CPU 스레드 수(기본: 라이브러리 기본값, 추론 프로세스가 켜졌을 때만)")
    parser.add_argument("--log", metavar="PATH", required=True, help="JSONL 기록 파일(새로 씀)")
    parser.add_argument("--source-id", type=_non_empty, default=None,
                        help="소스 ID(기본: 저장 입력 stored0, --ros-link gz_link)")
    parser.add_argument("--ros-reliability", choices=("reliable", "best_effort"), default=None,
                        help="--ros-link 영상 구독 신뢰성(기본 reliable). 발행자와 맞지 않으면 진단 사건으로 알린다")
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
    # 관측 전달 옵션: 역할 이름이 기본이고 옛 이름(--synthetic-a-to-b-*, --a-to-b-*)은 별칭으로 유지한다
    parser.add_argument("--synthetic-observation-fixture", "--synthetic-a-to-b-fixture", dest="synthetic_fixture",
                        metavar="JSON", default=None,
                        help="시험 전용(synthetic/test-only): 합성 관측 fixture로 A→B 저장 입력 연결을 확인한다. "
                             "--images/--video와만, --synthetic-observation-out 필수. 실제 검출·실시간 B 허용이 아니다 "
                             "(별칭 --synthetic-a-to-b-fixture)")
    parser.add_argument("--synthetic-observation-out", "--synthetic-a-to-b-out", dest="synthetic_out", metavar="PATH",
                        default=None,
                        help="--synthetic-observation-fixture 전용: B 시험 출력 JSONL(새 파일만, 덮어쓰지 않음) "
                             "(별칭 --synthetic-a-to-b-out)")
    parser.add_argument("--capture-meta", "--a-to-b-capture-meta", dest="capture_meta", metavar="JSON", default=None,
                        help="저장 입력 실제 A→B 경로(opt-in): 녹화 메타(CameraInfo·촬영 시각·그 시각 TF). "
                             "--images/--video·--detector ai(axis_extractor 설정)와만, --observation-out 필수 "
                             "(별칭 --a-to-b-capture-meta)")
    parser.add_argument("--observation-out", "--a-to-b-out", dest="observation_out", metavar="DIR", default=None,
                        help="--capture-meta 전용: B 결과·주석 영상 출력 폴더(새 폴더만) (별칭 --a-to-b-out)")
    parser.add_argument("--live-observation-out", dest="live_observation_out", metavar="DIR", default=None,
                        help="실시간 A→B 기록 전용 시험(opt-in): --ros-link·--detector ai(axis_extractor) 전용, B 결과·주석·건너뜀 "
                             "사유 출력 폴더(새 폴더만). --live-observation-config 필수. A 검출은 C 연결부(47093)로 항상 보낸다(#233). C 목표는 "
                             "--live-sort-part, B center는 --b-center-overlay일 때만. D·로봇 송신 없음")
    parser.add_argument("--live-observation-config", dest="live_observation_config", metavar="JSON", default=None,
                        help="--live-observation-out 전용: 시험 설정(TF 상한·용량·종료 상한 등, status=test_only 필수)")
    parser.add_argument("--tracking-space", dest="tracking_space", choices=("pixel", "world"), default=None,
                        help="--tracking 추적 좌표(#217): pixel(기존 픽셀 추적) | world(작업 좌표 W 추적, --live-observation-out "
                             "전용, --world-tracker-config 필수). 생략하면(#239) --live-observation-out은 world, 그 밖은 pixel")
    parser.add_argument("--world-tracker-config", dest="world_tracker_config", metavar="JSON", default=None,
                        help="--tracking-space world 전용: W 추적 시험 설정(status=test_only)")
    parser.add_argument("--gripper-handoff", dest="gripper_handoff", action="store_true",
                        help="--tracking-space world 전용, 기본 꺼짐: 그리퍼 깊이(/gripper_cam/depth_image)로 확정 W 트랙의 위치를 "
                             "이어 갱신하고 화면 밖이면 번호를 유지한다. --world-tracker-config에 visibility·gripper 블록 필수 "
                             "(예 config/vision/world_tracker_gripper_test.json). B 입력은 바뀌지 않는다")
    parser.add_argument("--track-stream", dest="track_stream", metavar="JSON", default=None,
                        help="--tracking-space world 전용, 기본 꺼짐: 링크 프레임마다 확정 W 트랙을 ROS 토픽(TrackStateArray, "
                             "예 config/vision/track_stream_test.json의 /a/track_states)으로 발행한다. robot_ws 빌드·source 필요. "
                             "이동 집기용이면 --gripper-handoff와 함께 켠다")
    parser.add_argument("--track-stream-clock", dest="track_stream_clock", choices=("own", "link"), default=None,
                        help="--track-stream 전용: 발행 시각(SIM)을 어디서 읽나. link(기본, #430 A 결정) = 링크 영상 수신기가 "
                             "이미 받은 /clock 값을 씀(시계 구독·스레드 없음), own = 예전처럼 발행 노드가 /clock을 따로 구독"
                             "(시계 스레드 하나)")
    parser.add_argument("--arm-region-filter", dest="arm_region_filter", metavar="JSON", default=None,
                        help="--live-observation-out 전용: 팔·그리퍼 투영 영역 검출 제외 설정(기본 켬, "
                             f"기본 파일 {ARM_REGION_DEFAULT_REL})")
    parser.add_argument("--no-arm-region-filter", dest="no_arm_region_filter", action="store_true",
                        help="--live-observation-out 전용: 팔·그리퍼 투영 영역 검출 제외를 끈다")
    parser.add_argument("--belt-region-filter", dest="belt_region_filter", metavar="JSON", default=None,
                        help="--live-observation-out 전용: 벨트 윗면 밖(바닥·상자) 검출 제외 설정(기본 켬, "
                             f"기본 파일 {BELT_REGION_DEFAULT_REL}, 벨트 사각형은 config/robot/frames.yaml)")
    parser.add_argument("--no-belt-region-filter", dest="no_belt_region_filter", action="store_true",
                        help="--live-observation-out 전용: 벨트 밖 검출 제외를 끈다")
    parser.add_argument("--live-sort-part", dest="live_sort_part", action="store_true",
                        help="--live-observation-out 전용(#214 D13 해제): 수락된 B 결과의 C 목표를 /sort_part로 송신. 단계 T, 또는 "
                             "단계 L(#233, B가 고른 1개만). 설정 sort_part 필수. robot_ws의 eyetoarm_interfaces 빌드·source 필요")
    parser.add_argument("--b-center-overlay", dest="b_center_overlay", action="store_true",
                        help="--live-observation-out --b-input list 전용(#233): 목록 소비자 자리에 B center_overlay_consumer를 붙여 "
                             "/b/center_overlay/image(인터페이스가 있으면 /b/center_overlay도)로 B center 주석 영상을 발행")
    parser.add_argument("--b-center-always-draw", dest="b_center_always_draw", action="store_true",
                        help="--b-center-overlay와 함께(#406): 구독자가 없어도 사진마다 center를 그려 발행(예전 동작). "
                             "기본은 center 영상 토픽 구독자가 없는 동안(대시보드가 center를 고르지 않음) 그리기·영상 발행을 건너뜀"
                             "(B 계산·/b/results·목표는 그대로)")
    center_process = parser.add_mutually_exclusive_group()
    center_process.add_argument("--b-center-process", dest="b_center_process", action="store_true", default=None,
                                help="--b-center-overlay와 함께(#413): center 그리기·영상 발행을 별도 프로세스에서 한다(기본 켜짐, "
                                     "명시해도 같음). A는 요청만 넘기고, 밀리면 최신 사진만 그림. 자식이 죽으면 center 영상만 멈추고 "
                                     "실행은 계속한다")
    center_process.add_argument("--no-b-center-process", dest="b_center_process", action="store_false",
                                help="--b-center-overlay와 함께(#413 r3): 별도 프로세스를 끄고 예전처럼 같은 프로세스에서 그린다")
    parser.add_argument("--b-input", dest="b_input", choices=("single", "list"), default="single",
                        help="B 입력 단계(#189): single = 단계 T(기본, 관측 하나 eyetoarm.b_input/1), list = 단계 L(같은 프레임 "
                             "후보 목록 eyetoarm.b_input/2). list는 --capture-meta 저장 입력 경로 또는 --live-observation-out 실시간 기록 전용 경로에서만 쓰며 B 목록 소비자"
                             "(BStage.process_frame)가 없으면 시작 때 거부한다(단계 T로 바꾸지 않음)")
    parser.add_argument("--tracking", action="store_true", help="--detector ai 전용. A-6 1차 실험: AI 검출 인스턴스 추적 번호를 붙인다(M2 휴리스틱, 팔 이동을 넘는 신원·중복 집기 방지 보장 없음). 기본 꺼짐. 저장·표시 경로는 표시·기록 전용이고, "
                             "--live-observation-out과 함께면(#215 단계 T, #244 단계 L 후보별) 확정 번호만 B 중복 방지에 넘긴다")

    return parser


def _check_detector_args(args) -> None:
    """검출 방식과 설정 인자 조합을 확인한다. 어긋나면 argparse 오류(종료 2)."""
    if args.tracking and args.detector != "ai":
        build_parser().error("--tracking is for --detector ai only")

    if args.detector == "ai":
        if args.ai_config is None:
            build_parser().error("--detector ai requires --ai-config")
        if args.detect_config is not None:
            build_parser().error("--detect-config is for --detector opencv only")
    else:
        if args.detect_config is None:
            build_parser().error("--detector opencv requires --detect-config")
        if args.ai_config is not None or args.model_load_timeout is not None or args.log_instance_geometry:
            build_parser().error("--ai-config/--model-load-timeout/--log-instance-geometry are for --detector ai")

    if args.infer_process_on and args.infer_process_off:
        build_parser().error("--infer-process and --no-infer-process are mutually exclusive")

    if (args.infer_process_on or args.infer_process_off) and args.detector != "ai":
        build_parser().error("--infer-process/--no-infer-process are for --detector ai only")

    if args.infer_process_on and args.synthetic_fixture is not None:
        build_parser().error("--infer-process cannot be combined with --synthetic-observation-fixture (no AI detector)")


def _resolve_infer_process(args) -> None:
    """추론 프로세스 실제 적용 여부(`args.infer_process`)와 출처(`args.infer_process_source`: argument·default)를 정한다.

    인자가 없으면 실시간 AI(--ros-link --detector ai)만 켬이다. 시험이 모델·import를 주입하면 그 대역은 자식으로
    넘길 수 없으므로 기본은 끔이고, 명시 --infer-process와 함께면 인자 오류(종료 2)다.
    """
    injected = args._model_factory is not None or args._import_module is not None

    if args.infer_process_on:
        on, source = True, "argument"
    elif args.infer_process_off:
        on, source = False, "argument"
    else:
        on = args.detector == "ai" and args.ros_link and args.synthetic_fixture is None and not injected
        source = "default"

    if on and injected:
        build_parser().error("--infer-process cannot use an injected model_factory/import_module (in-process only)")

    if args.infer_torch_threads is not None and not on:
        build_parser().error("--infer-torch-threads requires the inference process (on by --infer-process or "
                             "by default for --ros-link --detector ai)")

    args.infer_process, args.infer_process_source = on, source


def _infer_mode_text(args) -> str | None:
    """시작 줄 문구. 기본으로 꺼진 경우는 None(기존 출력 그대로)."""
    if args.infer_process:
        note = " (default for --ros-link --detector ai; --no-infer-process to disable)" \
            if args.infer_process_source == "default" else ""

        return f"run_pipeline: infer_process=on source={args.infer_process_source}{note}"

    if args.infer_process_source == "argument":
        return "run_pipeline: infer_process=off source=argument"

    return None


def _infer_off_summary(args) -> dict | None:
    """인자로 끈 경우의 요약 표시(기본으로 꺼진 경우는 None: 요약 키를 더하지 않음)."""
    if not args.infer_process and args.infer_process_source == "argument":
        return {"enabled": False, "source": "argument"}

    return None


def _check_source_args(args, ros_available) -> None:
    """입력 인자 조합을 확인한다. --ros-link 전용 옵션 오용·rclpy 없음은 argparse 오류(종료 2)."""
    if args.ros_link:
        available = ros_available() if ros_available is not None else importlib.util.find_spec("rclpy") is not None

        if not available:
            build_parser().error("--ros-link requires rclpy (run: source /opt/ros/jazzy/setup.bash)")
    elif args.ros_reliability is not None:
        build_parser().error("--ros-reliability is for --ros-link only")

    # 실제 A→B 경로: 저장 입력·AI 전용. 실시간 B 금지(D13)·추적·시험 fixture와 결합 금지
    if args.capture_meta is not None or args.observation_out is not None:
        if args.ros_link:
            build_parser().error("--capture-meta/--observation-out are not allowed with --ros-link (D13: no B in live)")
        if args.capture_meta is None or args.observation_out is None:
            build_parser().error("--capture-meta and --observation-out must be given together")
        if args.synthetic_fixture is not None or args.synthetic_out is not None:
            build_parser().error("--capture-meta cannot be combined with --synthetic-observation-fixture/--synthetic-observation-out")
        if args.tracking:
            build_parser().error("--capture-meta cannot be combined with --tracking")
        if args.detector != "ai":
            build_parser().error("--capture-meta requires --detector ai")

    # 단계 L: 저장 입력 실제 경로(--capture-meta)에서만
    # 실시간은 기록 전용 연결(--live-observation-out) 안에서만. 기본 --ros-link의 B 소비는 D13 그대로 거부
    if args.b_input == "list" and args.capture_meta is None and args.live_observation_out is None:
        build_parser().error("--b-input list requires --capture-meta/--observation-out (stored input) or "
                             "--ros-link --live-observation-out (live record-only); D13: no B in plain live")

    # 실시간 A→B 기록 전용 시험: 명시 옵션에서만 전용 디스패처. 기본 --ros-link는 D13 그대로
    if args.live_observation_out is not None or args.live_observation_config is not None:
        if args.live_observation_out is None or args.live_observation_config is None:
            build_parser().error("--live-observation-out and --live-observation-config must be given together")
        if not args.ros_link:
            build_parser().error("--live-observation-out requires --ros-link")
        if args.detector != "ai":
            build_parser().error("--live-observation-out requires --detector ai")

    # 추적 공간 생략 시 실시간 기록 연결(--live-observation-out)은 W, 그 밖은 픽셀.
    # W 설정 파일은 숨은 기본 경로 없이 명시해야 한다(설정은 모두 test_only, 기본값 없음 규칙)
    args.tracking_space_source = "explicit" if args.tracking_space is not None else "default"

    if args.tracking_space is None:
        args.tracking_space = "world" if args.tracking and args.live_observation_out is not None else "pixel"

        if args.tracking_space == "world" and args.world_tracker_config is None:
            build_parser().error("--live-observation-out --tracking uses W tracking by default: --world-tracker-config JSON "
                                 "is required (e.g. config/vision/world_tracker_test.json), or give --tracking-space pixel")

    # W 추적은 실시간 기록 연결 안에서만(촬영 시각 TF 필요)
    if args.tracking_space == "world":
        if not args.tracking:
            build_parser().error("--tracking-space world requires --tracking")
        if args.live_observation_out is None:
            build_parser().error("--tracking-space world requires --live-observation-out (needs capture-time TF)")
        if args.world_tracker_config is None:
            build_parser().error("--tracking-space world requires --world-tracker-config")
    elif args.world_tracker_config is not None:
        build_parser().error("--world-tracker-config is for --tracking-space world only")

    if getattr(args, "gripper_handoff", False) and args.tracking_space != "world":
        build_parser().error("--gripper-handoff requires --tracking-space world (W tracking)")

    if getattr(args, "track_stream", None) is not None and args.tracking_space != "world":
        build_parser().error("--track-stream requires --tracking-space world (W tracking)")

    # 기본값(None → link)은 --track-stream 없이도 무동작. link를 명시했는데 --track-stream이 없을 때만 거부한다
    if getattr(args, "track_stream_clock", None) == "link" and getattr(args, "track_stream", None) is None:
        build_parser().error("--track-stream-clock link requires --track-stream")

    if getattr(args, "track_stream_clock", None) is None:
        args.track_stream_clock = "link"

    # 팔 영역 제외는 촬영 시각 TF·카메라가 있는 실시간 기록 연결에서만(기본 켬)
    if args.arm_region_filter is not None and args.no_arm_region_filter:
        build_parser().error("--arm-region-filter and --no-arm-region-filter are mutually exclusive")

    if (args.arm_region_filter is not None or args.no_arm_region_filter) and args.live_observation_out is None:
        build_parser().error("--arm-region-filter/--no-arm-region-filter require --live-observation-out "
                             "(needs capture-time TF and camera)")

    # 벨트 밖 제외도 같은 규칙(촬영 시각 TF·카메라가 있는 실시간 기록 연결에서만, 기본 켬)
    if args.belt_region_filter is not None and args.no_belt_region_filter:
        build_parser().error("--belt-region-filter and --no-belt-region-filter are mutually exclusive")

    if (args.belt_region_filter is not None or args.no_belt_region_filter) and args.live_observation_out is None:
        build_parser().error("--belt-region-filter/--no-belt-region-filter require --live-observation-out "
                             "(needs capture-time TF and camera)")

    # 별도 프로세스는 --b-center-overlay일 때의 기본값(#413 r3). 켜라고 명시했는데 overlay가 없을 때만 거부한다
    if getattr(args, "b_center_process", None) is True and not args.b_center_overlay:
        build_parser().error("--b-center-process requires --b-center-overlay")

    # C 목표 송신: 기록 전용 연결의 수락 결과만(단계 L은 B가 고른 1개)
    if args.live_sort_part:
        if args.live_observation_out is None:
            build_parser().error("--live-sort-part requires --live-observation-out (live accepted B results only)")

    # B center 발행기는 실시간 기록 연결의 단계 L(목록 소비자 자리)에서만
    if args.b_center_overlay:
        if args.live_observation_out is None:
            build_parser().error("--b-center-overlay requires --live-observation-out")
        if args.b_input != "list":
            build_parser().error("--b-center-overlay is for stage L only (--b-input list); stage T has no center source")

    # 단계별 시각 진단은 실시간 관측 경로에서만(#372)
    if getattr(args, "stage_timing", None) is not None and args.live_observation_out is None:
        build_parser().error("--stage-timing requires --live-observation-out (live run diagnostic)")

    # 시험 B 연결: 저장 입력 전용. 실시간 입력의 B 소비 금지(D13)·추적과 B 결합 금지를 인자 단계에서 막는다
    if args.synthetic_fixture is not None or args.synthetic_out is not None:
        if args.ros_link:
            build_parser().error("--synthetic-observation-fixture/--synthetic-observation-out are not allowed with --ros-link (D13: no B in live)")
        if args.synthetic_fixture is None or args.synthetic_out is None:
            build_parser().error("--synthetic-observation-fixture and --synthetic-observation-out must be given together")
        if args.tracking:
            build_parser().error("--synthetic-observation-fixture cannot be combined with --tracking")


def build_runner_config(args) -> RunnerConfig:
    """해석한 인자로 실행 설정을 만든다. 상한을 주지 않으면 RunnerConfig 기본값(10/1, 준비 10)을 그대로 쓴다."""
    _check_detector_args(args)
    load = {} if args.model_load_timeout is None else {"model_load_timeout_s": args.model_load_timeout}

    if args.no_detect_timeout:
        if args.first_detect_timeout is not None or args.detect_timeout is not None:
            build_parser().error("--no-detect-timeout cannot be combined with --first-detect-timeout/--detect-timeout")

        return RunnerConfig(max_frames=args.max_frames, first_detect_timeout_s=None, detect_timeout_s=None, **load)

    over = {}

    if args.first_detect_timeout is not None:
        over["first_detect_timeout_s"] = args.first_detect_timeout

    if args.detect_timeout is not None:
        over["detect_timeout_s"] = args.detect_timeout

    return RunnerConfig(max_frames=args.max_frames, **over, **load)


def _live_factory(source_id: str, reliability: str | None, boundary_hook=None):
    """실시간 링크 카메라 소스 팩토리. ROS 어댑터는 이때만 import한다. `boundary_hook`은 기록 전용 시험만 준다."""
    from eyetoarm.inputs.ros_image import LiveLinkSource, LiveSourceConfig

    config = LiveSourceConfig() if reliability is None else LiveSourceConfig(reliability=reliability)

    def factory(issuer, generation_provider):
        """issuer·generation_provider를 받아 소스를 만든다."""
        return LiveLinkSource(source_id=source_id, issuer=issuer, generation_provider=generation_provider,
                              config=config, boundary_hook=boundary_hook)

    return factory


class _StopSignals:
    """실시간 실행 신호 처리. 끝나면 원래 처리기로 되돌린다.

    - 첫 SIGINT·SIGTERM·SIGHUP: 정지 요청(`stop`).
    - 첫 신호 뒤 `repeat_window_s` 안에 온 신호: 같은 사건의 중복으로 보고 무시한다(터미널 Ctrl+C가 그룹 전체에 가고
      감독 스크립트도 같은 신호를 넘기면 A는 거의 동시에 두 번 받는다).
    - 그 뒤 실행 중에 온 신호: KeyboardInterrupt(강제 중단). `finally` 정리는 그대로 돈다.
    - `begin_cleanup()` 뒤에 온 신호: 세기만 하고 무시한다(정리를 끝까지 한다. 강제 종료는 SIGKILL).
    """

    SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)

    def __init__(self, repeat_window_s: float = 1.0, clock=time.monotonic) -> None:
        """정지 표지·중복 창·시계."""
        self.stop = threading.Event()
        self._hits = 0
        self._first_at: float | None = None
        self._repeat_window_s = float(repeat_window_s)
        self._clock = clock
        self._cleanup = False
        self.ignored = {"repeat": 0, "cleanup": 0}
        self._old: dict = {}

    def _on_signal(self, signum, frame) -> None:
        """신호 처리기."""
        self._hits += 1
        now = self._clock()

        if self._hits == 1:
            self._first_at = now
            self.stop.set()
            return

        if self._cleanup:
            self.ignored["cleanup"] += 1
            return

        if now - self._first_at <= self._repeat_window_s:
            self.ignored["repeat"] += 1
            return

        raise KeyboardInterrupt

    def begin_cleanup(self) -> None:
        """정리 시작: 이후 신호는 정리를 끊지 않는다."""
        self._cleanup = True

    def __enter__(self) -> "_StopSignals":
        """주 스레드에서만 처리기를 설치한다."""
        if threading.current_thread() is threading.main_thread():
            for sig in self.SIGNALS:
                self._old[sig] = signal.signal(sig, self._on_signal)

        return self

    def __exit__(self, *exc) -> None:
        """원래 처리기로 되돌린다."""
        for sig, handler in self._old.items():
            signal.signal(sig, handler)

        self._old = {}


REPO_ROOT = Path(__file__).resolve().parents[1]
ARM_REGION_DEFAULT = REPO_ROOT / ARM_REGION_DEFAULT_REL
BELT_REGION_DEFAULT = REPO_ROOT / BELT_REGION_DEFAULT_REL
PART_SPECS_PATH = REPO_ROOT / "config/calibration/part_specs.json"   # B 파일(읽기만)
FRAMES_PATH = REPO_ROOT / "config/robot/frames.yaml"   # 좌표 기준(읽기만): 그리퍼 카메라 대조·벨트 사각형
# 시야 보정 평면과 B 볼트 기준 높이의 허용 차(m). 평면 차가 보정량에 주는 영향은 작지만(5 mm면 약 1 %)
# 넘으면 보정 설정을 B 규격에 맞게 다시 정해야 하므로 시작을 거부한다(2)
VIEW_BIAS_PLANE_TOL_M = 0.005
EXIT_RESET_DETECTED = 3   # 실시간 기록 연결이 시각 되감기로 실행을 끝냄(정리 정상)


def _bolt_spec_heights(specs) -> dict[str, float]:
    """B 규격의 볼트 기준 높이(m): 「bolt」(B `class_plane_z` 기본) + 볼트 종류별. 읽기만."""
    from eyetoarm.transforms.work_result import class_plane_z

    heights = {"bolt": class_plane_z(specs, "bolt")}

    for name, spec in specs["specs"].items():
        if str(spec.get("class_name", "")).strip().lower() == "bolt":
            heights[name] = class_plane_z(specs, "bolt", name)

    return heights


def _bolt_planes(specs):
    """W 추적 투영 높이: B 규격의 볼트 종류별 (기준 높이, 총길이, 판정 허용·범위), 마스크 입력용 기본은 「bolt」 높이.

    종류 값은 `bolt_spec.bolt_specs_from`(시야 보정과 같은 해석·검사)으로 읽는다. 맞지 않으면 ValueError(시작 거부 2).
    """
    from eyetoarm.vision.bolt_spec import bolt_specs_from
    from eyetoarm.vision.world_tracking import BoltPlanes

    heights = _bolt_spec_heights(specs)
    bolts = bolt_specs_from(specs)
    rows = tuple((s.name, heights[s.name], s.total_length_m) for s in bolts)
    # 길이 허용·범위를 함께 넘겨 시야 보정·B `_classify`와 같은 고르기(판정 안 먼저)를 쓴다(#270·#285)
    tols = tuple(s.length_tolerance_m for s in bolts)
    ranges = tuple((s.length_min_m, s.length_max_m) if s.ranged else None for s in bolts)

    return BoltPlanes(default_z_m=heights["bolt"], specs=rows, tolerances_m=tols, ranges_m=ranges)


def _check_view_bias_plane(bias, specs_path: Path | None = None) -> dict[str, float]:
    """보정 `plane_z_m`과 B 볼트 기준 높이(「bolt」·종류별)의 차가 모두 `VIEW_BIAS_PLANE_TOL_M` 안인지 본다.

    넘으면 ValueError(시작 거부 2), 통과하면 {이름: 높이 m}. `specs_path`가 None이면 `PART_SPECS_PATH`.
    보정 시점에는 볼트 종류를 모르므로 투영 높이는 설정 값 하나를 쓴다.
    """
    from eyetoarm.transforms.work_result import load_specs

    specs_path = PART_SPECS_PATH if specs_path is None else specs_path
    heights = _bolt_spec_heights(load_specs(specs_path))

    for name, z in heights.items():
        if abs(z - bias.plane_z_m) > VIEW_BIAS_PLANE_TOL_M:
            raise ValueError(f"axis_view_bias.plane_z_m {bias.plane_z_m!r} differs from B part_specs {name} plane_z {z!r} "
                             f"by more than {VIEW_BIAS_PLANE_TOL_M * 1000:.0f} mm ({specs_path}): "
                             "re-derive axis_view_bias for the current B specs")

    return heights


def _view_bias_runtime(bias, specs_path: Path | None = None):
    """보정 설정 → 실행 인자. None·기존 형식은 그대로, 종류별 형식은 B 규격과 대조한 `ViewBiasPlan`(#270).

    맞지 않으면 ValueError(시작 거부 2). `specs_path`가 None이면 `PART_SPECS_PATH`.
    """
    if bias is None or bias.legacy:
        return bias

    import hashlib

    from eyetoarm.transforms.work_result import load_specs
    from eyetoarm.vision.view_bias import build_view_bias_plan

    specs_path = PART_SPECS_PATH if specs_path is None else specs_path

    try:
        return build_view_bias_plan(bias, load_specs(specs_path),
                                    part_specs_sha256=hashlib.sha256(Path(specs_path).read_bytes()).hexdigest())
    except ValueError as exc:
        raise ValueError(f"{exc} ({specs_path})") from exc


def _heights_text(heights: dict[str, float], ref_m: float | None = None) -> str:
    """시작 줄용 「이름=높이 mm(차)」 목록. `ref_m`을 주면 그 높이와의 차(mm)도 붙인다."""
    parts = []

    for name, z in heights.items():
        diff = "" if ref_m is None else f"({(z - ref_m) * 1000:+.1f})"
        parts.append(f"{name}={z * 1000:.1f}mm{diff}")

    return " ".join(parts)



def _prepare_synthetic_observation(args, err):
    """시험 B 연결 준비: fixture 읽기·입력 지문 확인·출력 경로 확인. 실패하면 (None, 2)."""
    from eyetoarm.vision.synthetic_observation_fixture import FixtureError, check_inputs, load_fixture

    try:
        fixture = load_fixture(args.synthetic_fixture)
        paths = list(args.images) if args.images else [args.video]
        inputs = check_inputs(fixture, paths, args.source_id)
    except (OSError, FixtureError) as exc:
        err.write(f"run_pipeline: synthetic-a-to-b fixture error: {exc}\n")
        return None, 2

    # 출력은 덮어쓰지 않는다(기록·B 출력 둘 다, 열기 전에 확인)
    for path in (args.log, args.synthetic_out):
        if Path(path).exists():
            err.write(f"run_pipeline: synthetic-a-to-b output exists, refusing to overwrite: {path}\n")
            return None, 2

    return (fixture, inputs), 0


def _ai_detector(args, config):
    """AI 검출기와 추론 프로세스 대리 객체(꺼짐이면 None)를 만든다. 생성자는 I/O·프로세스 시작이 없다."""
    if args.infer_process:
        from eyetoarm.vision.ai_process import ProcessAiSegDetector

        options = dict(getattr(args, "_infer_process_options", None) or {})
        proxy = ProcessAiSegDetector(config, args.ai_config, torch_threads=args.infer_torch_threads, **options)

        return proxy, proxy

    # AI 어댑터는 이때만 import한다(OpenCV 경로는 ai_detector도 불러오지 않음)
    from eyetoarm.vision.ai_detector import AiSegDetector

    inject = {"model_factory": args._model_factory}

    if args._import_module is not None:
        inject["import_module"] = args._import_module

    return AiSegDetector(config, **inject), None


def _detector_run_kw(det, proxy) -> dict:
    """실행기에 넘길 검출기 인자. 대리 객체는 취소 토큰을 받는다."""
    kw = {"detector": det, "prepare": det.prepare}

    if proxy is not None:
        kw["detector_takes_token"] = True

    return kw


def _stop_or_infer_failed(proxy, stop=None):
    """정지 조건에 추론 프로세스 실패를 더한다. 대리 객체가 없으면 받은 조건 그대로."""
    if proxy is None:
        return stop

    return lambda: (stop is not None and stop()) or proxy.failed is not None


def _finish_infer_process(proxy, err, args=None):
    """실패 사유를 close 전에 읽고 자식을 닫아 (실패 사유, 요약)을 돌려준다. 대리 객체가 없으면 (None, 끔 표시 또는 None)."""
    if proxy is None:
        return None, None if args is None else _infer_off_summary(args)

    failed = proxy.failed

    try:
        proxy.close()
    except Exception as exc:   # 정리 실패도 실행 문제로 남긴다
        failed = failed or f"infer process close failed: {type(exc).__name__}: {exc}"

    summary = {"enabled": True, "source": getattr(args, "infer_process_source", None), **proxy.summary()}
    summary["failed"] = failed

    if failed is not None:
        err.write(f"run_pipeline: infer process failed: {failed}\n")

    return failed, summary


def _write_infer_record(log, summary, err) -> bool:
    """JSONL에 추론 프로세스 요약 1줄. 쓰지 못하면 False."""
    try:
        log.write_infer_process(summary)
    except Exception as exc:   # 기록 실패는 실행 문제로 올린다(호출자가 종료 1)
        err.write(f"run_pipeline: cannot write infer_process record: {type(exc).__name__}: {exc}\n")
        return False

    return True


def main(argv=None, *, stderr=None, model_factory=None, import_module=None, ros_available=None,
         live_source_factory=None, b_stage_hook=None, extension_source=None, live_tf_factory=None,
         live_run_controller=None, list_consumer_factory=None,
         live_sort_part_factory=None, live_conveyor_probe=None, live_handoff_factory=None,
         center_publisher_factory=None, live_gripper_factory=None, center_process_factory=None, infer_process_options=None,
         live_track_stream_factory=None) -> int:
    """CLI 진입점. 종료 코드를 돌려준다.

    나머지 키워드 인자(`b_stage_hook`·`extension_source`·`ros_available`·`*_factory`·`live_run_controller`·
    `import_module`·`live_conveyor_probe`)는 시험 주입 지점이며 기본은 실제 구현이다.
    `live_conveyor_probe` 결과는 요약 진단값으로만 쓰고 추적기에는 넘기지 않는다(정답 격리).
    `infer_process_options`는 `--infer-process` 대리 객체 생성 인자(시험의 가짜 공장·환경)다.
    """
    err = stderr if stderr is not None else sys.stderr
    args = build_parser().parse_args(argv)  # 인자 오류는 argparse가 종료 2로 끝낸다
    _check_source_args(args, ros_available)
    args._model_factory, args._import_module = model_factory, import_module
    args._infer_process_options = infer_process_options
    _resolve_infer_process(args)
    runner_config = build_runner_config(args)

    if args.source_id is None:
        args.source_id = "gz_link" if args.ros_link else "stored0"

    detector_kw = {}
    synthetic = None
    capture = None

    if args.capture_meta is not None:
        capture, code = _prepare_capture_observation(args, err)

        if capture is None:
            return code

    if args.synthetic_fixture is not None:
        synthetic, code = _prepare_synthetic_observation(args, err)

        if synthetic is None:
            return code

    try:
        if args.detector == "ai":
            config = load_ai_config(args.ai_config)
        else:
            config = load_config(args.detect_config)

        # 저장 입력에는 보정 프로파일이 없다. 시험 B 연결에서만 fixture에 명시된 적용 조건을 쓴다
        profile = (synthetic[0].source_profile if synthetic is not None
                   else capture[0].source_profile if capture is not None else None)

        if capture is not None and config.axis_extractor is None:
            raise ValueError("--capture-meta requires an AI config with axis_extractor (e.g. \"m2\")")

        # 시선 기반 끝점 보정: 켠 설정만 시작 때 대조한다(키 없음이면 아무것도 읽지 않음)
        if getattr(config, "axis_view_bias", None) is not None:
            if synthetic is not None:
                raise ValueError("axis_view_bias is not supported with --synthetic-observation-fixture (no AI endpoints)")

            if config.axis_view_bias.legacy:
                heights = _check_view_bias_plane(config.axis_view_bias)
                err.write(f"run_pipeline: axis_view_bias plane_z={config.axis_view_bias.plane_z_m * 1000:.1f}mm; B bolt heights "
                          f"{_heights_text(heights, config.axis_view_bias.plane_z_m)} within ±{VIEW_BIAS_PLANE_TOL_M * 1000:.0f} mm "
                          "(correction uses the single config plane)\n")
            else:
                plan = _view_bias_runtime(config.axis_view_bias)
                rows = " ".join(f"{s.name}({s.label})=plane {s.plane_z_m * 1000:.1f}mm k {plan.k_of(s.name):.3f}mm "
                                f"len {s.accept_range_m()[0] * 1000:.1f}~{s.accept_range_m()[1] * 1000:.1f}mm"
                                f"{'(range)' if s.ranged else '(tol)'}" for s in plan.specs)
                err.write(f"run_pipeline: axis_view_bias by spec: {rows} (spec by endpoint length at each spec plane)\n")

            if capture is None and args.live_observation_out is None:
                err.write("run_pipeline: axis_view_bias inactive: no B input path (needs --capture-meta or "
                          "--live-observation-out)\n")
        settings = PipelineSettings(source_id=args.source_id, mode=args.mode, detector_config=config, profile=profile)
    except (OSError, ValueError) as exc:
        err.write(f"run_pipeline: config error: {exc}\n")
        return 2

    # 추적 opt-in: 진입점이 프로세스당 발급기 1개를 만들어 주입한다
    # W 추적은 연결부 추적 스레드가 하므로 실행기 픽셀 추적기를 만들지 않는다
    if args.tracking and args.tracking_space == "pixel":
        detector_kw["part_tracker"] = PartTracker(PartTrackIdIssuer(), mode=TrackMode.HEURISTIC)

    if synthetic is not None:
        return _run_synthetic_observation(args, settings, runner_config, detector_kw, synthetic, err, b_stage_hook)

    proxy = None

    if args.detector == "ai":
        det, proxy = _ai_detector(args, config)
        detector_kw.update(_detector_run_kw(det, proxy))
        mode_text = _infer_mode_text(args)

        if mode_text is not None:
            err.write(mode_text + "\n")

    args._infer_proxy = proxy

    if args.live_observation_out is not None:
        return _run_live_observation(args, settings, runner_config, detector_kw, err, b_stage_hook=b_stage_hook,
                                     live_source_factory=live_source_factory, live_tf_factory=live_tf_factory,
                                     controller=live_run_controller or _live_runs(),
                                     list_consumer_factory=list_consumer_factory,
                                     live_sort_part_factory=live_sort_part_factory,
                                     live_conveyor_probe=live_conveyor_probe,
                                     live_handoff_factory=live_handoff_factory,
                                     center_publisher_factory=center_publisher_factory, center_process_factory=center_process_factory,
                                     live_gripper_factory=live_gripper_factory,
                                     live_track_stream_factory=live_track_stream_factory)

    if capture is not None and args.b_input == "list":
        return _run_capture_list(args, settings, runner_config, detector_kw, capture, err, b_stage_hook,
                                 list_consumer_factory)

    if capture is not None:
        return _run_capture_observation(args, settings, runner_config, detector_kw, capture, err, b_stage_hook,
                              extension_source)

    if args.ros_link:
        make = live_source_factory or _live_factory
        source_factory = make(args.source_id, args.ros_reliability)
    else:
        kind, paths = ("images", list(args.images)) if args.images else ("video", [args.video])
        source_factory = stored_source_factory(kind, paths, args.source_id)

    # 기본 경로도 기존 기록을 덮어쓰지 않는다(다른 경로와 같은 "x" 규칙)
    try:
        stream = open(args.log, "x", encoding="utf-8")
    except FileExistsError:
        err.write(f"run_pipeline: output exists, refusing to overwrite: {args.log}\n")
        return 2
    except OSError as exc:
        err.write(f"run_pipeline: cannot open log: {exc}\n")
        return 2

    probe = HandoffProbe()
    handoff = AToCHandoff(notify=probe.notify)
    probe.start(handoff)
    infer_failed = infer_record_ok = None

    # 실시간이면 신호 처리기를 정리 끝까지 둔다(정리 중 신호가 정리를 끊지 않게, D-9)
    signals = _StopSignals() if args.ros_link else None

    with signals if signals is not None else contextlib.nullcontext():
        try:
            with stream:
                log = PipelineLog(JsonlRecorder(stream), eval_run_id=args.eval_run_id,
                                  log_geometry=args.log_instance_geometry, infer_process=proxy is not None,
                                  infer_process_source=args.infer_process_source)
                if signals is not None:
                    result = run(source_factory, settings, log, config=runner_config, stderr=err,
                                 versions={"opencv": cv2.__version__},
                                 should_stop=_stop_or_infer_failed(proxy, signals.stop.is_set),
                                 handoff_sink=handoff, **detector_kw)
                    signals.begin_cleanup()
                else:
                    result = run(source_factory, settings, log, config=runner_config, stderr=err,
                                 versions={"opencv": cv2.__version__}, handoff_sink=handoff,
                                 should_stop=_stop_or_infer_failed(proxy), **detector_kw)

                infer_failed, infer_summary = _finish_infer_process(proxy, err, args)

                if infer_summary is not None:
                    infer_record_ok = _write_infer_record(log, infer_summary, err)
        finally:
            if signals is not None:
                signals.begin_cleanup()

            probe.stop()
            handoff.close()

            if proxy is not None:
                proxy.close()   # 실행 중 예외로 위 정리를 건너뛴 경우(두 번 불러도 됨)

    exit_code = result.exit_code

    if exit_code == 0 and (infer_failed is not None or infer_record_ok is False):
        exit_code = 1

    watch = (f"{runner_config.first_detect_timeout_s}/{runner_config.detect_timeout_s}" if runner_config.watchdog
             else "off")
    method = f"detector=ai model_id={config.model_id}" if args.detector == "ai" else "detector=opencv"
    err.write(f"run_pipeline: exit={exit_code} reason={result.stop_reason} {method} frames={result.frames} "
              f"input_errors={result.input_errors} detect_timeout={watch} timeouts={result.timeouts} "
              f"stalled={result.stalled}"
              + (f" live_inputs={result.live_inputs} input_boundary_aborts={result.input_boundary_aborts}"
                 if args.ros_link else "")
              + (f" infer_process=on failed={infer_failed}" if proxy is not None else "") + "\n")

    return exit_code


_LIVE_RUNS = None


def _live_runs():
    """프로세스 단위 실시간 실행 관리자(정리 완료 확인 전 새 실행 거부). 처음 부를 때 만든다."""
    global _LIVE_RUNS

    if _LIVE_RUNS is None:
        from eyetoarm.vision.live_run_session import LiveRunController

        def no_default(run_id):
            raise RuntimeError("live observation runs are created with an explicit factory")

        _LIVE_RUNS = LiveRunController(no_default, close_limit_s=1.0)

    return _LIVE_RUNS


class _RosTfRuntime:
    """실제 TF 수신 묶음: 자기 rclpy Context + `LiveTfRuntime`. close는 런타임 → Context 순서."""

    def __init__(self, config) -> None:
        """Context를 만들고(신호 처리기 없음) TF 노드·버퍼·Listener·executor를 시작한다."""
        import rclpy
        from rclpy.context import Context
        from rclpy.signals import SignalHandlerOptions

        from eyetoarm.transforms.live_tf_buffer import LiveTfRuntime

        self._context = Context()
        rclpy.init(context=self._context, signal_handler_options=SignalHandlerOptions.NO)
        self._runtime = LiveTfRuntime(self._context, node_name="eyetoarm_live_observation_tf",
                                      static_pairs=config.static_pairs)

        try:
            self._runtime.start()
        except BaseException:
            self._context.try_shutdown()
            raise

        self.lookup, self.has_static, self.clear = self._runtime.lookup, self._runtime.has_static, self._runtime.clear
        self.set_failure_callback = self._runtime.set_failure_callback   # 연결부가 수신 스레드 실패를 받는다
        self._shutdown = False

    def close(self, limit_s: float) -> None:
        """런타임 정리(실패하면 RuntimeError, 재시도 가능) → Context 종료."""
        self._runtime.close(limit_s)

        if not self._shutdown:
            self._context.try_shutdown()
            self._shutdown = True


def _new_world_tracker(config, visibility=None, gripper=None):
    """W 추적기: 이 실행 전용 발급기(프로세스 수명 번호 규칙은 픽셀 추적과 같음). 넘기기 설정이 없으면 지금과 같다."""
    from eyetoarm.vision.world_tracking import WorldTracker

    if visibility is None and gripper is None:
        return WorldTracker(PartTrackIdIssuer(), config=config)

    return WorldTracker(PartTrackIdIssuer(), config=config, visibility=visibility,
                        gripper_gate_mm=None if gripper is None else (gripper.gate_x_mm, gripper.gate_y_mm),
                        gripper_late_ms=None if gripper is None else gripper.late_limit_ms)


def _load_gripper_handoff(path: str):
    """넘기기 설정(visibility·gripper 블록)과 frames.yaml 그리퍼 카메라 대조, 벨트 사각형 → (visibility, GripperLive)."""
    import yaml

    from eyetoarm.inputs.ros_gripper_depth import belt_roi_xy_m
    from eyetoarm.vision.live_world_tracking import GripperLive, load_handoff_config

    visibility, gripper = load_handoff_config(path)

    if visibility is None or gripper is None:
        raise ValueError("--gripper-handoff needs both visibility and gripper blocks in --world-tracker-config")

    with open(FRAMES_PATH, encoding="utf-8") as f:
        frames = yaml.safe_load(f)

    cam = frames["cameras"]["gripper"]

    if cam["optical_frame"] != gripper.optical_frame or cam["camera_info_topic"] != gripper.camera_info_topic:
        raise ValueError(f"gripper config {gripper.optical_frame}/{gripper.camera_info_topic} != frames.yaml cameras.gripper "
                         f"{cam['optical_frame']}/{cam['camera_info_topic']}")

    return visibility, GripperLive(gripper, belt_roi_xy_m(frames))


def _default_track_stream_sender(config, now_ns=None):
    """실제 추적 흐름 발행기(rclpy·eyetoarm_interfaces). 시험은 같은 모양(send·now_ns·close)의 대역을 준다.

    `now_ns`를 주면(`--track-stream-clock link`) 발행기는 `/clock`을 구독하지 않고 그 함수로 발행 시각을 읽는다.
    """
    from eyetoarm.vision.track_stream import RosTrackStreamSender

    return RosTrackStreamSender(config, now_ns=now_ns)


class _LinkSimClock:
    """링크 영상 수신기의 마지막 `/clock` 값을 추적 발행 쪽에 넘기는 자리(#419, `--track-stream-clock link`).

    발행기는 소스보다 먼저 만들어지므로 소스는 나중에 `attach`로 붙는다(제어 흐름 스레드). `now_ns`는 추적 발행
    스레드가 부르고, 값 자체는 수신 스레드가 `LiveLinkSource.handle_clock`에서 쓴다. 소스 참조는 잠금으로 넘긴다.
    소스가 붙기 전이면 0(= /clock 못 받음, `own` 방식의 시작 직후와 같은 뜻).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._source = None

    def attach(self, source) -> None:
        """시각을 읽을 소스(`sim_now_ns()`가 있는 것)를 붙인다."""
        with self._lock:
            self._source = source

    def wrap(self, factory):
        """소스 팩토리를 감싸, 만들어진 소스를 붙인 뒤 그대로 돌려준다."""
        def make(issuer, generation_provider):
            """원래 팩토리로 소스를 만들고 붙인다."""
            source = factory(issuer, generation_provider)
            self.attach(source)

            return source

        return make

    def now_ns(self) -> int:
        """발행 SIM 시각(ns). 소스가 없으면 0."""
        with self._lock:
            source = self._source

        return 0 if source is None else int(source.sim_now_ns())


def _default_gripper_transport(config, sink):
    """실제 그리퍼 깊이 수신(rclpy). 시험은 같은 모양(start·close·receiver.stats)의 대역을 준다."""
    from eyetoarm.inputs.ros_gripper_depth import GripperDepthReceiver, GripperDepthTransport

    return GripperDepthTransport(config, GripperDepthReceiver(config, sink))


def _ros_conveyor_speed_probe(timeout_s: float) -> dict:
    """C `/conveyor_controller`의 `speed_mps` 파라미터를 시작 때 1회 읽는다(진단 전용, 자기 Context를 쓰고 닫음).

    노드가 없으면 `status="no_node"`, 상한 `timeout_s` 안에 응답이 없으면 `no_reply`.
    이 값은 예측·짝짓기·번호 판정에 쓰지 않는다(정답 격리).
    """
    import rclpy
    from rcl_interfaces.srv import GetParameters
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.signals import SignalHandlerOptions

    ctx = Context()
    rclpy.init(context=ctx, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    out = {"node": "/conveyor_controller", "speed_mps": None, "status": "no_node", "timeout_s": timeout_s}

    try:
        node = rclpy.create_node("eyetoarm_world_tracking_probe", context=ctx)
        client = node.create_client(GetParameters, "/conveyor_controller/get_parameters")

        if client.wait_for_service(timeout_sec=timeout_s):
            ex = SingleThreadedExecutor(context=ctx)
            ex.add_node(node)
            fut = client.call_async(GetParameters.Request(names=["speed_mps"]))
            ex.spin_until_future_complete(fut, timeout_sec=timeout_s)
            res = fut.result() if fut.done() else None
            ex.shutdown()

            if res is not None and res.values:
                out.update(speed_mps=float(res.values[0].double_value), status="ok")
            else:
                out["status"] = "no_reply"
    finally:
        if node is not None:
            node.destroy_node()

        ctx.try_shutdown()

    return out


class _RosSortPartRuntime:
    """C 송신 런타임: 자기 rclpy Context + 노드 + spin 스레드 + `RecordingSortPartClient`.

    spin 스레드는 액션 콜백(관리 실행이면 `GetSortRun` 조회 타이머·응답도)을 처리하고 `send`는 worker가 부른다(기다리지 않음).
    시작 때 `server_wait_s` 동안 서버를 기다려 `server_seen`을 남긴다(없어도 실행).
    관리 실행(`config.run_source`)이면 같은 시간 동안 조회 서비스를 기다리고 첫 답을 `call_timeout_s`까지 기다린다.
    서비스가 없어도 실행하며, 답이 올 때까지 송신기가 보내지 않는다(#301 P2 Q2).
    """

    NODE_NAME = "eyetoarm_live_sort_part"

    def __init__(self, config) -> None:
        """Context·노드·클라이언트·spin 스레드. 인터페이스가 없으면 ImportError(진입점 종료 1)."""
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.signals import SignalHandlerOptions

        from eyetoarm.vision.live_sort_part import RecordingSortPartClient, result_watch_of

        self._config = config
        self._sink = None
        self._context = Context()
        rclpy.init(context=self._context, signal_handler_options=SignalHandlerOptions.NO)
        self._node = None
        self._executor = None
        self._thread = None
        self._stop = threading.Event()
        self._closed = False
        self.spin_error = None

        # 실행 번호 조회 상태(`_run_lock`). 조회는 spin 스레드 타이머·결과 콜백, 첫 조회만 진입점 스레드
        self._run_cfg = getattr(config, "run_source", None)
        self._run_client = None
        self._run_request = None
        self._run_listener = None
        self._run_last = None                 # (active, job, epoch)
        self._run_lock = threading.Lock()
        self._run_future = None
        self._run_future_ns = 0
        self._run_first = threading.Event()
        self.run_service_seen = None
        self.run_first_answer = None
        self._discard_listener = None
        self._discard_ready = threading.Event()

        try:
            self._node = rclpy.create_node(self.NODE_NAME, context=self._context)
            watch = result_watch_of(config)
            self.client = RecordingSortPartClient(self._node, self._emit, stage=None, watch=watch)
            self.watch_errors = 0
            # 진행 중 목표 결과 감시(#373 D-14): spin 스레드 타이머, 시각은 단조 시계
            self._node.create_timer(watch.check_s, self._watch_results)

            if self._run_cfg is not None:
                from eyetoarm_interfaces.srv import GetSortRun   # 없으면 ImportError(진입점 종료 1)
                from std_srvs.srv import Trigger

                self._run_request = GetSortRun.Request
                self._run_client = self._node.create_client(GetSortRun, self._run_cfg.service)
                self._node.create_timer(self._run_cfg.poll_s, self._poll_run)
                self._node.create_service(Trigger, "~/discard_detections", self._discard_detections)

            self._executor = SingleThreadedExecutor(context=self._context)
            self._executor.add_node(self._node)
            self._thread = threading.Thread(target=self._spin, name="eyetoarm-live-sort-part-spin", daemon=True)
            self._thread.start()
            self.server_seen = bool(self.client.client.wait_for_server(timeout_sec=config.server_wait_s))

            if self._run_client is not None:
                self.run_service_seen = bool(self._run_client.wait_for_service(timeout_sec=config.server_wait_s))

                if self.run_service_seen:
                    self.request_run_refresh()
                    self._run_first.wait(self._run_cfg.call_timeout_s)

                self.run_first_answer = self._run_last
        except BaseException:
            self.close(1.0)
            raise

    # ---------- 결과 감시(spin 스레드) ----------
    def _watch_results(self) -> None:
        """타이머 콜백: 서버 보임 여부와 단조 시각으로 클라이언트 감시를 돌린다. 예외는 세고 사건으로 남긴다(spin을 죽이지 않음)."""
        try:
            self.client.watch(time.monotonic_ns(), bool(self.client.client.server_is_ready()))
        except Exception as exc:  # noqa: BLE001 - 타이머 최상단: spin 스레드를 살려 두고 사유를 남긴다
            self.watch_errors += 1
            self._emit("result_watch_error", None, {"error": f"{type(exc).__name__}: {exc}"})

    def abandon(self, gid: str, reason: str) -> bool:
        """진행 중 목표를 「모름」으로 끝낸다(송신기의 실행 번호 변경 처리가 부름)."""
        return self.client.abandon(gid, reason)

    @property
    def watch_counts(self) -> dict:
        """감시 계수(참고용, 요약)."""
        return {**self.client.watch_counts, "errors": self.watch_errors}

    # ---------- 실행 번호 조회(관리 실행) ----------
    def set_run_listener(self, listener) -> None:
        """조회 결과 받을 곳(`on_run_snapshot`·`on_run_query_failed`). 이미 받은 답이 있으면 바로 넘긴다."""
        with self._run_lock:
            self._run_listener = listener
            last = self._run_last

        if listener is not None and last is not None:
            listener.on_run_snapshot(*last)

    def set_discard_listener(self, listener) -> None:
        """A/B link의 operator discard 함수를 ROS control service에 연결한다."""
        self._discard_listener = listener
        self._discard_ready.set()

    def _discard_detections(self, _request, response):
        if not self._discard_ready.wait(5.0):
            response.success = False
            response.message = "live A/B link is not ready"
            return response
        try:
            ok = bool(self._discard_listener("operator"))
        except Exception as exc:
            response.success = False
            response.message = f"discard failed: {type(exc).__name__}: {exc}"
            return response
        response.success = ok
        response.message = "A/B detections discarded" if ok else "A/B worker did not drain"
        return response

    def request_run_refresh(self) -> None:
        """지금 한 번 조회한다(진행 중인 조회가 있으면 그것을 기다림). 주기 타이머도 이것을 부른다."""
        self._poll_run()

    def _poll_run(self) -> None:
        """조회 하나를 보낸다. 진행 중이면 건너뛰고, `call_timeout_s`를 넘긴 조회는 버리고 실패로 센다."""
        if self._run_client is None:
            return

        failure = None
        future = None

        with self._run_lock:
            now = time.monotonic_ns()
            pending = self._run_future

            if pending is not None and not pending.done():
                if now - self._run_future_ns < self._run_cfg.call_timeout_s * 1e9:
                    return

                self._run_client.remove_pending_request(pending)
                failure = "timeout"

            self._run_future = None

            if not self._run_client.service_is_ready():
                failure = failure or "no_service"
            else:
                future = self._run_client.call_async(self._run_request())
                self._run_future, self._run_future_ns = future, now

        if failure is not None:
            self._run_failed(failure)

        if future is not None:
            future.add_done_callback(self._on_run_answer)

    def _on_run_answer(self, future) -> None:
        """조회 답(spin 스레드). 취소·예외는 실패로 센다."""
        try:
            res = future.result()
        except Exception as exc:  # noqa: BLE001 - 서비스 응답 실패는 사유로 남긴다
            self._run_failed(f"error:{type(exc).__name__}")
            return

        if res is None:
            self._run_failed("no_answer")
            return

        snap = (bool(res.active), str(res.sort_job_id), int(res.run_epoch))

        with self._run_lock:
            self._run_last = snap
            listener = self._run_listener

        self._run_first.set()

        if listener is not None:
            listener.on_run_snapshot(*snap)

    def _run_failed(self, reason: str) -> None:
        with self._run_lock:
            listener = self._run_listener

        if listener is not None:
            listener.on_run_query_failed(reason)

    def run_source_status(self) -> dict | None:
        """요약용 시작 상태(관리 실행이 아니면 None)."""
        if self._run_cfg is None:
            return None

        first = self.run_first_answer
        return {"service": self._run_cfg.service, "service_seen_at_start": self.run_service_seen,
                "first_answer": None if first is None else
                {"active": first[0], "sort_job_id": first[1], "run_epoch": first[2]}}

    def set_release_target(self, target) -> None:
        """B 클라이언트의 `stage` 자리(결과·no_server·busy 때 `release(번호)`를 받을 곳)."""
        self.client.stage = target

    def set_event_sink(self, sink) -> None:
        """사건 받을 곳(연결부 송신기의 `on_client_event`)."""
        self._sink = sink

    def _emit(self, event, gid, detail) -> None:
        sink = self._sink

        if sink is not None:
            sink(event, gid, detail)

    def _spin(self) -> None:
        """spin 스레드 최상단: 예외는 `spin_error`로 남긴다(정리에서 드러냄)."""
        try:
            while not self._stop.is_set():
                self._executor.spin_once(timeout_sec=self._config.spin_timeout_s)
        except Exception as exc:
            self.spin_error = f"{type(exc).__name__}: {exc}"

    @property
    def stats(self):
        """B 클라이언트 계수(참고용)."""
        return self.client.stats

    def send(self, out) -> bool:
        """B 클라이언트 송신(기다리지 않음)."""
        return self.client.send(out)

    def inflight_ids(self) -> list:
        """결과 대기 goal_id."""
        return self.client.inflight_ids()

    def close(self, limit_s: float) -> None:
        """spin 스레드 멈춤·join → executor·노드 정리 → Context 종료. 멈추지 않으면 RuntimeError(재시도 가능)."""
        if self._closed:
            return

        self._stop.set()

        if self._thread is not None:
            self._thread.join(timeout=limit_s)

            if self._thread.is_alive():
                raise RuntimeError(f"sort_part spin thread did not stop within {limit_s}s")

        if self._executor is not None:
            self._executor.shutdown(timeout_sec=limit_s)

        if self._node is not None:
            self._node.destroy_node()
            self._node = None

        self._context.try_shutdown()
        self._closed = True

        if self.spin_error is not None:
            raise RuntimeError(f"sort_part spin thread failed: {self.spin_error}")


CENTER_CONSUMER_KIND = "center_overlay(BStage.process_frame)"   # 기록·시작 줄의 목록 소비자 이름
DISCARD_CLEAR_LIMIT_S = 5.0   # 운영자 폐기의 CLEAR 한도(#397): B 비우기 2 s + 5 s < C 호출 한도 8 s


class _CenterOverlayStats:
    """B center 감싸개 공용 오류 계수. 단계·감싸개가 새로 생겨도 한 곳에 모은다(worker가 씀)."""

    def __init__(self) -> None:
        """오류 수 0, 첫 오류 없음."""
        self._lock = threading.Lock()
        self.errors = 0
        self.first_error: str | None = None

    def on_error(self, exc: Exception) -> None:
        """`center_overlay_consumer(on_error=)` 자리: 그리기·발행 실패를 센다(B 결과에는 영향 없음)."""
        with self._lock:
            self.errors += 1

            if self.first_error is None:
                self.first_error = f"{type(exc).__name__}: {exc}"

    def summary(self, publisher) -> dict:
        """요약 블록: 발행 수(발행기 계수)·오류 수·첫 오류·번호표 토픽(없으면 None = Image만)."""
        with self._lock:
            errors, first = self.errors, self.first_error

        return {"published": getattr(publisher, "published", None), "errors": errors, "first_error": first,
                "topic": getattr(publisher, "topic", None), "overlay_topic": getattr(publisher, "overlay_topic", None)}


def _default_center_publisher():
    """B `CenterOverlayPublisher`(자기 rclpy Context). ROS가 없으면 ImportError."""
    from eyetoarm.transforms.center_overlay_pub import CenterOverlayPublisher

    return CenterOverlayPublisher()


def _handoff_address() -> str:
    """C 연결부 forwarder 목적지(`HandoffProbe`와 같은 환경 변수·기본값, 표시용)."""
    import os

    from eyetoarm.vision.c_handoff_probe import DEFAULT_INGRESS_HOST, DEFAULT_INGRESS_PORT

    host = os.environ.get("EYETOARM_DISPLAY_INGRESS_HOST", DEFAULT_INGRESS_HOST)

    return f"{host}:{os.environ.get('EYETOARM_DISPLAY_INGRESS_PORT', DEFAULT_INGRESS_PORT)}"


def _default_live_handoff():
    """기본 `--ros-link`와 같은 C 연결부 묶음 `(AToCHandoff, HandoffProbe)`. 시작은 호출자가 한다."""
    probe = HandoffProbe()

    return AToCHandoff(notify=probe.notify), probe


def _load_arm_region(args, world_frame: str, injected_tf: bool):
    """팔 영역 제외 설정. (설정 | None, 표시 사전 | None). 끔이면 설정 None. 읽기·검사 오류는 ValueError·OSError.

    기본 켬은 실제 TF 런타임일 때만이다. 시험이 TF 대역을 주입하면(카메라 프레임만 흉내) 기본은 끔이고 표시도
    더하지 않는다(기존 출력 그대로). 명시 --arm-region-filter는 주입 TF에서도 켠다.
    """
    if args.no_arm_region_filter:
        return None, {"enabled": False, "source": "argument"}

    if args.arm_region_filter is None and injected_tf:
        return None, None

    from eyetoarm.vision.arm_region import load_arm_region_config

    path = args.arm_region_filter or str(ARM_REGION_DEFAULT)
    config, sha = load_arm_region_config(path)

    if config.world_frame != world_frame:
        raise ValueError(f"arm region world_frame {config.world_frame!r} != live observation world_frame "
                         f"{world_frame!r}")

    return config, {"enabled": True, "source": "argument" if args.arm_region_filter else "default", "path": path,
                    "sha256": sha}


def _load_belt_region(args, world_frame: str, injected_tf: bool):
    """벨트 밖 제외 설정. (설정 | None, 표시 사전 | None). 켬 규칙은 `_load_arm_region`과 같다.

    벨트 사각형·top_z는 frames.yaml(읽기만)에서 온다. 읽기·검사 오류는 ValueError·OSError.
    """
    if args.no_belt_region_filter:
        return None, {"enabled": False, "source": "argument"}

    if args.belt_region_filter is None and injected_tf:
        return None, None

    from eyetoarm.vision.belt_region import load_belt_region_config

    path = args.belt_region_filter or str(BELT_REGION_DEFAULT)
    config, sha, frames_sha = load_belt_region_config(path, str(FRAMES_PATH))

    if config.world_frame != world_frame:
        raise ValueError(f"belt region world_frame {config.world_frame!r} != live observation world_frame "
                         f"{world_frame!r}")

    return config, {"enabled": True, "source": "argument" if args.belt_region_filter else "default", "path": path,
                    "sha256": sha, "frames_path": str(FRAMES_PATH), "frames_sha256": frames_sha}


class _ArmRegionRecords:
    """영역 판정 기록(팔 `arm_region.jsonl`·벨트 `belt_region.jsonl`, 제외·TF 실패·오류가 있는 프레임만). worker 스레드가 쓴다."""

    def __init__(self, path) -> None:
        """새 파일을 연다(있으면 FileExistsError)."""
        self._fh = open(path, "x", encoding="utf-8")
        self._lock = threading.Lock()
        self.written = 0
        self.error: str | None = None

    def write(self, record: dict) -> None:
        """한 줄(정상이고 제외 없는 프레임은 쓰지 않는다). 쓰기 실패는 기록만 하고 검출을 막지 않는다."""
        if record.get("status") == "ok" and not any(d["excluded"] for d in record.get("decisions", ())):
            return

        try:
            with self._lock:
                self._fh.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                self._fh.flush()
                self.written += 1
        except (OSError, ValueError) as exc:   # 기록 실패: 요약 problems로 올린다
            self.error = self.error or f"{type(exc).__name__}: {exc}"

    def close(self) -> None:
        """파일을 닫는다."""
        with self._lock:
            self._fh.close()


def _run_live_observation(*args, **kw) -> int:
    """`_run_live_observation_body`를 신호 처리기 안에서 부른다. 처리기는 정리가 끝날 때까지 둔다(D-9).

    정지 신호는 실행 중에는 정지 요청이고, 본문이 정리에 들어가면(`begin_cleanup`) 정리를 끊지 않는다.
    """
    path = getattr(args[0], "stage_timing", None)
    err = args[4]

    if path is not None:   # 진단: 실행 전에 켜고 정리까지 끝난 뒤 끈다
        from eyetoarm.vision import stage_timing

        stage_timing.enable(path)
        err.write(f"run_pipeline: STAGE TIMING ON path={path} (diagnostic only)\n")

    try:
        with _StopSignals() as signals:
            return _run_live_observation_body(*args, signals=signals, **kw)
    finally:
        if path is not None:
            err.write(f"run_pipeline: stage timing {stage_timing.disable()}\n")


def _run_live_observation_body(args, settings, runner_config, detector_kw, err, *, b_stage_hook, live_source_factory,
                               live_tf_factory, controller, list_consumer_factory=None,
                               live_sort_part_factory=None, live_conveyor_probe=None, live_handoff_factory=None,
                               center_publisher_factory=None, live_gripper_factory=None, center_process_factory=None,
                               live_track_stream_factory=None, signals=None) -> int:
    """실시간 A→B 기록 전용 시험. 리셋이면 실행을 멈추고 정리한다(자동 재시작 없음).

    경로: LiveLinkSource → 실행기 → AiSegDetector → 코어 → `live_observation_sink` → 디스패처 → TF → B → 기록기.
    `--b-input list`에서 목록 소비자가 없으면 TF·출력을 만들기 전에 종료 1(단계 T로 바꾸지 않음).
    정리 순서: 연결부 정리 → center 발행기(정리 미확인이면 열어 두고 1) → `probe.stop` → `handoff.close`.
    """
    import cv2 as _cv2

    from eyetoarm.inputs.base import new_run_id
    from eyetoarm.transforms.b_stage import BStage
    from eyetoarm.transforms.observation_envelope_adapter import EnvelopeTfLookup, ObservationEnvelopeAdapter
    from eyetoarm.vision.live_observation_link import (LiveObservationConfigError, LiveObservationLink,
                                                       load_live_observation_config)
    from eyetoarm.vision.pipeline_records import B_INPUT_LIST

    try:
        link_config = load_live_observation_config(args.live_observation_config)
    except (OSError, LiveObservationConfigError) as exc:
        err.write(f"run_pipeline: live observation config error: {exc}\n")
        return 2

    if settings.detector_config.axis_extractor is None:
        err.write("run_pipeline: config error: --live-observation-out requires an AI config with axis_extractor\n")
        return 2

    try:
        arm_config, arm_mark = _load_arm_region(args, link_config.world_frame, live_tf_factory is not None)
    except (OSError, ValueError) as exc:
        err.write(f"run_pipeline: arm region filter config error: {exc}\n")
        return 2

    try:
        belt_config, belt_mark = _load_belt_region(args, link_config.world_frame, live_tf_factory is not None)
    except (OSError, ValueError) as exc:
        err.write(f"run_pipeline: belt region filter config error: {exc}\n")
        return 2

    sort_on = bool(getattr(args, "live_sort_part", False))

    if sort_on != (link_config.sort_part is not None):   # 옵션과 설정 블록은 짝
        err.write("run_pipeline: live observation config error: "
                  + ("--live-sort-part requires config block sort_part\n" if sort_on
                     else "config block sort_part given without --live-sort-part\n"))
        return 2

    # 추적 켬이면 require_track_id 필수(기본값 없음). 추적 끔의 true는 볼트 송신 0이라 거부.
    # 단계 L + 추적은 고른 후보의 A 번호로 판정하므로 허용
    require = None if link_config.sort_part is None else link_config.sort_part.require_track_id

    if sort_on and args.tracking and require is None:
        err.write("run_pipeline: live observation config error: sort_part.require_track_id is required with --tracking "
                  "(true = send only confirmed part_track_id PICK_PLACE)\n")
        return 2

    if sort_on and require is True and not args.tracking:
        err.write("run_pipeline: live observation config error: sort_part.require_track_id=true needs --tracking "
                  "(no track ids here: no PICK_PLACE would ever be sent)\n")
        return 2

    for path in (args.log, args.live_observation_out):
        if Path(path).exists():
            err.write(f"run_pipeline: output exists, refusing to overwrite: {path}\n")
            return 2

    listed = args.b_input == "list"
    world_on = getattr(args, "tracking_space", "pixel") == "world"
    world_cfg = None
    gripper_on = bool(getattr(args, "gripper_handoff", False))
    handoff_cfg = (None, None)
    stream_on = world_on and getattr(args, "track_stream", None) is not None
    stream_cfg = None

    # W 추적 설정·볼트 평면 높이(B 규격 읽기만)는 출력·세션을 만들기 전에 확인한다
    if world_on:
        from eyetoarm.transforms.work_result import load_specs
        from eyetoarm.vision.live_world_tracking import LagConfigError, load_world_tracking_config

        try:
            world_cfg = load_world_tracking_config(args.world_tracker_config)
            # B 규격의 볼트 종류별 높이(읽기만). 모두 같으면 높이 하나
            world_plane_z = _bolt_planes(load_specs(PART_SPECS_PATH))

            if gripper_on:
                handoff_cfg = _load_gripper_handoff(args.world_tracker_config)

            if stream_on:
                from eyetoarm.vision.track_stream import load_track_stream_config

                stream_cfg = load_track_stream_config(args.track_stream)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            err.write(f"run_pipeline: world tracking config error: {exc}\n")
            return 2

    if listed and list_consumer_factory is None and not callable(getattr(BStage, "process_frame", None)):
        err.write("run_pipeline: --b-input list refused: list consumer unavailable (BStage.process_frame missing); "
                  "not falling back to stage T\n")
        return 1

    center_on = bool(getattr(args, "b_center_overlay", False))

    # B center 소비자는 목록 소비자 자리 하나다. 시험 주입 소비자와 함께면 모호하므로 거부한다
    if center_on and list_consumer_factory is not None:
        err.write("run_pipeline: --b-center-overlay cannot be combined with an injected list consumer (test only)\n")
        return 2

    run_id = new_run_id()
    refusal = controller.refusal(run_id)

    if refusal is not None:
        err.write(f"run_pipeline: live observation run refused (explicit new run only after cleanup): {refusal}\n")
        return 1

    controller.close_limit_s = link_config.close_limit_s
    sort_rt = None
    center_pub = None
    stream_sender = None
    # 추적 흐름을 켰을 때만(기본 link, #430). 끄면 소스 팩토리도 감싸지 않는다
    link_clock = (_LinkSimClock() if stream_on and getattr(args, "track_stream_clock", "link") == "link" else None)
    center_stats = _CenterOverlayStats()
    center_proc = None
    center_proc_exit = None

    def close_center_proc():
        """center 그리기 자식 프로세스를 끝낸다(없으면 아무것도 안 함). 종료 정보를 돌려준다."""
        if center_proc is None:
            return None

        try:
            return center_proc.close()
        except Exception as pexc:
            err.write(f"run_pipeline: center draw process close failed: {type(pexc).__name__}: {pexc}\n")
            return None

    def close_sort_rt():
        """세션이 만들어지지 않았을 때만: 이 함수가 만든 송신 런타임·center 발행기를 닫는다."""
        if sort_rt is not None:
            try:
                sort_rt.close(link_config.close_limit_s)
            except Exception as sexc:
                err.write(f"run_pipeline: sort_part runtime close failed: {type(sexc).__name__}: {sexc}\n")

        close_center_proc()

        if center_pub is not None:
            try:
                center_pub.close()
            except Exception as cexc:
                err.write(f"run_pipeline: center overlay publisher close failed: {type(cexc).__name__}: {cexc}\n")

        if stream_sender is not None:
            try:
                stream_sender.close()
            except Exception as texc:
                err.write(f"run_pipeline: track stream publisher close failed: {type(texc).__name__}: {texc}\n")

    # C 송신 런타임은 TF·세션 앞(인터페이스가 없으면 아무것도 만들지 않고 1)
    if sort_on:
        try:
            sort_rt = (live_sort_part_factory or _RosSortPartRuntime)(link_config.sort_part)
        except ImportError as exc:
            err.write(f"run_pipeline: --live-sort-part: eyetoarm_interfaces not found ({exc}); build robot_ws and "
                      "source robot_ws/install/setup.bash\n")
            close_sort_rt()
            return 1
        except Exception as exc:
            err.write(f"run_pipeline: sort_part runtime start failed: {type(exc).__name__}: {exc}\n")
            close_sort_rt()
            return 1

    # B center 발행기도 TF·세션 앞(ROS가 없으면 1). 프로세스에 하나, 단계마다 감싸개만 새로
    if center_on:
        try:
            center_pub = (center_publisher_factory or _default_center_publisher)()
        except ImportError as exc:
            err.write(f"run_pipeline: --b-center-overlay: ROS image publisher unavailable ({exc}); "
                      "source /opt/ros/jazzy/setup.bash\n")
            close_sort_rt()
            return 1
        except Exception as exc:
            err.write(f"run_pipeline: center overlay publisher start failed: {type(exc).__name__}: {exc}\n")
            close_sort_rt()
            return 1

    # 추적 흐름 발행기도 TF·세션 앞(인터페이스가 없으면 1). 발행 큐·스레드는 W 추적 스레드가 시작·종료한다
    world_stream = None

    if stream_on:
        from eyetoarm.vision.track_stream import TrackStream

        try:
            make_sender = live_track_stream_factory or _default_track_stream_sender
            # own(끄는 옵션)은 예전 호출 그대로, link(기본 #430)만 시각 함수를 넘긴다
            stream_sender = (make_sender(stream_cfg[0]) if link_clock is None
                             else make_sender(stream_cfg[0], now_ns=link_clock.now_ns))
        except ImportError as exc:
            err.write(f"run_pipeline: --track-stream: eyetoarm_interfaces TrackStateArray not found ({exc}); build robot_ws "
                      "and source robot_ws/install/setup.bash\n")
            close_sort_rt()
            return 1
        except Exception as exc:
            err.write(f"run_pipeline: track stream publisher start failed: {type(exc).__name__}: {exc}\n")
            close_sort_rt()
            return 1

        world_stream = TrackStream(stream_cfg[0], stream_sender.send, stream_sender.now_ns,
                                   clock_source="own" if link_clock is None else "link")

    if center_on:
        from eyetoarm.vision.center_draw_gate import CenterDrawGate, gated_center_consumer

        # 보는 쪽(center 영상 토픽 구독자)이 없으면 그리지 않는다(#406 ①). 끄는 옵션 --b-center-always-draw
        center_gate = CenterDrawGate(center_pub, always=bool(getattr(args, "b_center_always_draw", False)))

        if getattr(args, "b_center_process", None) is not False:   # 기본 켜짐(#413 r3), --no-b-center-process로만 끔
            from eyetoarm.vision.center_process import CenterDrawProcess, process_center_consumer

            # 그리기·영상 발행은 자식 프로세스가 한다(#413). 띄우지 못해도 center 영상만 없고 실행은 계속한다
            # 자식의 출력은 이 프로세스의 표준 오류로 함께 나간다(출력 폴더는 아직 만들어지기 전이라 파일을 두지 않음)
            def center_proc_failed(reason):
                """자식을 더 쓸 수 없게 된 때 한 번(자식 대리의 받기·보내기 스레드에서): 운영자가 화면이 원본으로 바뀐 까닭을 알게."""
                err.write(f"run_pipeline: center draw process stopped: {reason}; "
                          "center image off (relay sends the raw image), run continues\n")
                err.flush()

            center_proc = (center_process_factory or CenterDrawProcess)(REPO_ROOT, on_failed=center_proc_failed)

            try:
                center_proc.start()
            except OSError as pexc:
                err.write(f"run_pipeline: center draw process start failed ({type(pexc).__name__}: {pexc}); "
                          "center image off, run continues\n")

            consumer_factory = lambda stage: process_center_consumer(stage, center_pub, center_gate,  # noqa: E731
                                                                     center_proc, on_error=center_stats.on_error)
        else:
            consumer_factory = lambda stage: gated_center_consumer(stage, center_pub, center_gate,  # noqa: E731
                                                                   on_error=center_stats.on_error)
        consumer_kind = {"list_consumer_kind": CENTER_CONSUMER_KIND}
    else:
        consumer_factory = list_consumer_factory if listed else None
        consumer_kind = {}

    try:
        tf = (live_tf_factory or _RosTfRuntime)(link_config)
    except Exception as exc:
        err.write(f"run_pipeline: live TF start failed: {type(exc).__name__}: {exc}\n")
        close_sort_rt()
        return 1

    def stage_factory(camera):
        """카메라 프로파일마다 새 B 단계 + 입구 어댑터(같은 EnvelopeTfLookup)."""
        lookup = EnvelopeTfLookup()
        stage = BStage.from_config(REPO_ROOT, camera, lookup)

        if b_stage_hook is not None:
            b_stage_hook(stage)

        return stage, ObservationEnvelopeAdapter(stage, lookup)

    def render(stage, frame, out):
        """같은 프레임 영상에 같은 BOutput 주석(TF가 있을 때만)."""
        if getattr(out, "T_W_C", None) is None:
            return None

        ok, buf = _cv2.imencode(".png", stage.render_center(frame.image, [out]))

        return buf.tobytes() if ok else None

    # 이 실행의 실제 출구(요약 `outputs`·시작 줄이 같은 목록). C 연결부는 항상이다
    run_outputs = (f"c_handoff_detect({_handoff_address()})",
                   *((f"b_center_overlay({getattr(center_pub, 'topic', None)})",) if center_on else ()),
                   *(("sort_part(C goals via /sort_part, accepted results only)",) if sort_on else ()))

    try:
        link = controller.start_run(run_id, make_session=lambda rid: LiveObservationLink(
            rid, args.live_observation_out, config=link_config, tf=tf, stage_factory=stage_factory, render=render,
            b_input=args.b_input, list_consumer_factory=consumer_factory, **consumer_kind, run_outputs=run_outputs,
            view_bias=_view_bias_runtime(getattr(settings.detector_config, "axis_view_bias", None)),
            sort_part_runtime=sort_rt, tracking=args.tracking,
            **({} if not world_on else {"world_tracker": _new_world_tracker(world_cfg[0], handoff_cfg[0],
                                                                            None if handoff_cfg[1] is None else handoff_cfg[1].config),
                                        "world_config": world_cfg[1], "world_plane_z_m": world_plane_z,
                                        **({} if world_stream is None else {"world_stream": world_stream}),
                                        **({} if not gripper_on else {"world_visibility": handoff_cfg[0],
                                                                      "world_gripper": handoff_cfg[1]})})))
    except Exception as exc:
        # 세션이 만들어지지 않았다: 이 함수가 만든 TF만 정리한다(세션 등록 없음)
        err.write(f"run_pipeline: cannot start live observation run: {type(exc).__name__}: {exc}\n")

        try:
            tf.close(link_config.close_limit_s)
        except Exception as cexc:
            err.write(f"run_pipeline: live TF close failed: {type(cexc).__name__}: {cexc}\n")

        close_sort_rt()

        return 2 if isinstance(exc, OSError) else 1

    stage_l = (f" b_input=eyetoarm.b_input/2 consumer={link.consumer_kind}" if listed else "")
    err.write(f"run_pipeline: LIVE A->B RECORD-ONLY TEST run_id={run_id} config_sha256={link_config.sha256} "
              f"status=test_only{stage_l}; outputs={','.join(run_outputs)}; no D/robot output from A, no auto restart\n")

    arm_filter = arm_records = None   # 팔 영역 제외 감싸개·판정 기록(실행 try 안에서 만든다)
    belt_filter = belt_records = None   # 벨트 밖 제외 감싸개·판정 기록(같음)

    if center_on:
        overlay = getattr(center_pub, "overlay_topic", None)
        overlay_text = (f"overlay={overlay}" if overlay else
                        "overlay=none (eyetoarm_interfaces CenterOverlay not found: rebuild robot_ws; Image only)")
        err.write(f"run_pipeline: B CENTER OVERLAY ON topic={getattr(center_pub, 'topic', None)} {overlay_text} "
                  "(published right after B compute, before accept check (3); stage L only) "
                  + ("draw=always" if center_gate.always else "draw=only while a center subscriber exists (#406)")
                  + (f" draw_process=on pid={center_proc.pid} (#413)\n" if center_proc is not None else " draw_process=off\n"))

    if sort_on:
        set_sink = getattr(sort_rt, "set_event_sink", None)

        if callable(set_sink):
            set_sink(link.sort_part.on_client_event)

        # B 클라이언트의 release 대상 = 연결부의 모든 B 단계(추적 켬일 때만)
        set_release = getattr(sort_rt, "set_release_target", None)

        if link.release_proxy is not None and callable(set_release):
            set_release(link.release_proxy)

        # 관리 실행: C 실행 번호 조회 결과 → 송신기(B 단계에 넣고 송신 조건 판정, #301 P2)
        set_run = getattr(sort_rt, "set_run_listener", None)

        if link.sort_part.managed and callable(set_run):
            set_run(link.sort_part)

        from eyetoarm.vision.live_sort_part import DEFAULT_FAILURE_GAP_S

        seen = getattr(sort_rt, "server_seen", None)
        retry = link_config.sort_part.retry
        retry_text = ("off" if retry is None else f"max_failures_per_track={retry.max_failures_per_track} "
                      f"point_block={retry.failed_point_block_mm}mm/{retry.failed_point_block_s}s"
                      + (" (stage L: point block only, no track ids)" if listed and not args.tracking else ""))
        # 단계 L + 추적: B가 고른 후보의 A 번호를 목표에 싣는다(B-11, f0e604a). 목표 번호가 없을 때만 A 번호로 송신
        track_text = ((('world' if world_on else 'M2(heuristic)') if args.tracking else 'off')
                      + (" (stage L: B goal carries the chosen A candidate id (B-11); A id only if the goal has none)"
                         if listed and args.tracking else ""))
        err.write(f"run_pipeline: LIVE A->B->C SORT_PART ON action=/sort_part server_seen={str(seen).lower()} "
                  f"retry={retry_text} tracking={track_text} "
                  f"require_track_id={({True: 'true', False: 'false'}).get(link_config.sort_part.require_track_id, 'unset')} "
                  + ("" if retry is not None else f"retry_default=failure_gap_{DEFAULT_FAILURE_GAP_S:g}s ")
                  + "(D13 released by user decision 2026-11-20, accepted risk without pre-release test F6)\n")

        if link_config.sort_part.require_track_id:
            err.write("run_pipeline: require_track_id=true: PICK_PLACE without a confirmed part_track_id is not sent "
                      "(skip:no_track_id); bolts leaving view before confirmation are never sent "
                      "(see tracking tentative_dropped, sort_part.no_track_id_states)\n")

        if not seen:
            err.write("run_pipeline: warning: /sort_part server not seen at start; goals will be recorded as no_server "
                      "until it appears\n")

        run_source = link_config.sort_part.run_source

        if run_source is not None:   # 없으면 시작 줄 형식 그대로(비관리)
            status_fn = getattr(sort_rt, "run_source_status", None)
            status = (status_fn() if callable(status_fn) else None) or {}
            first = status.get("first_answer")
            first_text = ("none" if first is None else
                          f"active={str(first['active']).lower()} sort_job_id={first['sort_job_id'] or '-'} "
                          f"run_epoch={first['run_epoch']}")
            err.write(f"run_pipeline: sort_part run_source=managed service={run_source.service} "
                      f"service_seen={str(status.get('service_seen_at_start')).lower()} first={first_text} "
                      f"poll={run_source.poll_s}s timeout={run_source.call_timeout_s}s stale={run_source.stale_s}s "
                      "(goals sent only while the C run is active: skip:run_unknown|run_inactive|run_changed)\n")

            if not status.get("service_seen_at_start"):
                err.write(f"run_pipeline: warning: {run_source.service} not seen at start; goals are not sent "
                          "(skip:run_unknown) until a C run answer arrives\n")

    if args.tracking and not world_on:
        err.write("run_pipeline: tracking ON M2(heuristic): confirmed part_track_id only -> B _dedupe/release "
                  "(A-6 experimental ids; split ids are a known limit #156)\n")

    if world_on:
        # 진단 전용(정답 격리): C speed_mps 조회값은 요약에만 두고 추적기에는 넘기지 않는다
        try:
            # 상한은 설정값. 시험 주입 조회기는 인자 없이 부른다
            probe = (live_conveyor_probe() if live_conveyor_probe is not None
                     else _ros_conveyor_speed_probe(world_cfg[1].conveyor_probe_timeout_s))
        except Exception as exc:
            probe = {"node": "/conveyor_controller", "speed_mps": None, "status": f"error: {type(exc).__name__}: {exc}"}

        link.world.diagnostics["conveyor_speed_mps"] = probe
        space_note = "default for live" if getattr(args, "tracking_space_source", "explicit") == "default" else "explicit"
        spec_heights = _heights_text({n: z for n, z, _ in world_plane_z.specs})
        plane_note = (f"bolt planes {spec_heights}, mask {world_plane_z.default_z_m * 1000:.1f}mm"
                      f"{' (uniform)' if world_plane_z.uniform else ' (per spec by W length)'}")
        err.write(f"run_pipeline: tracking ON space=world({space_note}; W, {plane_note}) "
                  f"config_sha256={world_cfg[2]} "
                  f"lag_reset>{world_cfg[1].lag_limit_ms:.0f}ms; prediction uses W differencing only "
                  f"(conveyor speed_mps diagnostic={probe.get('speed_mps')} not used)\n")

    result = None
    exit_code = 1
    failure = None
    handoff = handoff_probe = None
    gripper_rx = None
    infer_proxy = getattr(args, "_infer_proxy", None)

    if gripper_on:
        v, g = handoff_cfg
        err.write(f"run_pipeline: GRIPPER HANDOFF ON depth={g.config.depth_topic} frame={g.config.optical_frame} "
                  f"rate<={g.config.max_rate_hz:g}Hz reorder_wait={g.config.reorder_wait_ms:g}ms late_limit={g.config.late_limit_ms}ms "
                  f"hold_limit={v.hold_limit_ms}ms margin={v.margin_px:g}px gate={g.config.gate_x_mm:g}x{g.config.gate_y_mm:g}mm "
                  "(confirmed W tracks only: position/time, no new ids, class kept, velocity unchanged; B input unchanged)\n")

    if stream_on:
        c = stream_cfg[0]
        err.write(f"run_pipeline: TRACK STREAM ON topic={c.topic} rate<={c.max_rate_hz:g}Hz qos=RELIABLE/KEEP_LAST{c.qos_depth} "
                  f"queue={c.queue_capacity} config_sha256={stream_cfg[1]} handoff_on={str(link.world.handoff_on).lower()} "
                  f"clock={'own' if link_clock is None else 'link'} "
                  "(confirmed W tracks only, SIM_TIME, base_link mm)\n")

        if not link.world.handoff_on:
            err.write("run_pipeline: WARNING track stream without --gripper-handoff: handoff_on=false is published; "
                      "moving picks from this stream are not supported (C may refuse)\n")

    # 세션 등록 뒤 모든 단계는 이 try 안이다. 실패해도 세션·TF를 정리하고 1로 끝낸다
    try:
        # 팔 영역 제외: 검출기 바로 뒤(worker 안, 관측 선택 전). 촬영 stamp TF·현재 카메라는 이 연결부의 것을 쓴다
        if arm_config is not None:
            from eyetoarm.vision.ai_convert import observation_from_ai
            from eyetoarm.vision.arm_region import ArmRegionFilter
            from eyetoarm.vision.live_observation_envelope import LiveCaptureTfLookup

            arm_records = _ArmRegionRecords(Path(args.live_observation_out) / "arm_region.jsonl")
            arm_filter = ArmRegionFilter(
                detector_kw["detector"],
                tf_lookup=LiveCaptureTfLookup(link.tf.lookup, world_frame=link_config.world_frame,
                                              wait_limit_s=link_config.tf_wait_limit_s),
                camera_for=lambda frame: link._track_camera, config=arm_config, select=observation_from_ai,
                on_decision=arm_records.write)
            detector_kw = {**detector_kw, "detector": arm_filter}
            err.write(f"run_pipeline: ARM REGION FILTER ON source={arm_mark['source']} config={arm_mark['path']} "
                      f"sha256={arm_mark['sha256'][:12]} min_fraction={arm_config.min_fraction:g} "
                      f"margin={arm_config.margin_mm:g}mm occluders={len(arm_config.occluders)} "
                      "(capture-stamp TF; TF failure keeps the detection)\n")
        elif arm_mark is not None:
            err.write("run_pipeline: arm region filter off (--no-arm-region-filter)\n")

        # 벨트 밖 제외: 팔 필터 바깥(팔로 이미 제외된 인스턴스는 판정하지 않음). 같은 촬영 stamp TF·카메라
        if belt_config is not None:
            from eyetoarm.vision.ai_convert import observation_from_ai
            from eyetoarm.vision.belt_region import BeltRegionFilter
            from eyetoarm.vision.live_observation_envelope import LiveCaptureTfLookup

            belt_records = _ArmRegionRecords(Path(args.live_observation_out) / "belt_region.jsonl")
            belt_filter = BeltRegionFilter(
                detector_kw["detector"],
                tf_lookup=LiveCaptureTfLookup(link.tf.lookup, world_frame=link_config.world_frame,
                                              wait_limit_s=link_config.tf_wait_limit_s),
                camera_for=lambda frame: link._track_camera, config=belt_config, select=observation_from_ai,
                on_decision=belt_records.write)
            detector_kw = {**detector_kw, "detector": belt_filter}
            x0, x1, y0, y1 = belt_config.rect_xy_m
            err.write(f"run_pipeline: BELT REGION FILTER ON source={belt_mark['source']} config={belt_mark['path']} "
                      f"sha256={belt_mark['sha256'][:12]} top_z={belt_config.top_z_m:.3f} "
                      f"rect=x[{x0:.3f},{x1:.3f}] y[{y0:.3f},{y1:.3f}] margin={belt_config.margin_mm:g}mm "
                      f"min_inside={belt_config.min_inside_fraction:g} (capture-stamp TF; TF failure keeps the detection; "
                      "lifted parts above the belt pass, arm filter covers them)\n")
        elif belt_mark is not None:
            err.write("run_pipeline: belt region filter off (--no-belt-region-filter)\n")

        make = live_source_factory or _live_factory
        source_factory = make(args.source_id, args.ros_reliability, boundary_hook=link.boundary_hook)

        if link_clock is not None:   # 추적 발행 시각을 이 소스의 /clock 값으로(#419)
            source_factory = link_clock.wrap(source_factory)

        # A 검출 → C 연결부(항상). 실행기 제어 흐름 하나가 생산자다
        handoff, handoff_probe = (live_handoff_factory or _default_live_handoff)()

        # W 추적이면 detect 영상 주석이 같은 작업의 W 번호를 읽게 한다(start_run 전, 기다리지 않는 조회)
        set_track_source = getattr(handoff, "set_track_source", None)

        if world_on and callable(set_track_source):
            set_track_source(link.world.peek)

        handoff_probe.start(handoff)
        link.start()

        # 운영자 데이터 폐기(C stop_run·recover가 부름, ROS 서비스 스레드). 연결부·연결을 만든 뒤에 붙인다: 그 전 호출은
        # `handoff`가 아직 없어 NameError였다(#397). 그 전 요청은 서비스가 `_discard_ready`로 기다린다.
        # CLEAR 한도 5 s: B 비우기 2 s와 합쳐 C 호출 한도(sort_manager VISION_DISCARD_TIMEOUT_S 8 s)보다 작게
        set_discard = getattr(sort_rt, "set_discard_listener", None)

        if sort_on and link.sort_part.managed and callable(set_discard):
            def discard_operator_data(reason):
                if not link.discard_detections(reason):
                    return False
                handoff.clear_from_control_thread(f"operator_{reason}", timeout_s=DISCARD_CLEAR_LIMIT_S)
                # center 무효화 표지(CenterOverlay marker)는 보내지 않는다(#397 R9, A 결정 (나)): marker는 실행 종료
                # 표지라 C 중계기가 이 A 실행을 은퇴시켜 재개 뒤 B center가 다시 나오지 않았다. 폐기 뒤 B center는
                # 새 사진의 B 결과가 덮어쓰고, 그 전에 끊기면 중계기 대체(--center-fallback-s)가 원본으로 바꾼다
                return True

            set_discard(discard_operator_data)

        # 그리퍼 깊이 수신: 추적 스레드의 그리퍼 큐로만 넣는다(B 입력 무변)
        if gripper_on:
            gripper_rx = (live_gripper_factory or _default_gripper_transport)(handoff_cfg[1].config, link.world.offer_gripper)
            gripper_rx.start()
        stream = open(args.log, "x", encoding="utf-8")

        with stream:
            log_kw = {"b_input": B_INPUT_LIST} if listed else {}

            if sort_on:   # C 목표 송신 재시도 정책(설정 블록 또는 기본 제한 상수)을 run_meta에 남긴다
                from eyetoarm.vision.live_sort_part import retry_policy

                log_kw["sort_part_retry"] = retry_policy(link_config.sort_part)

            log = PipelineLog(JsonlRecorder(stream), eval_run_id=args.eval_run_id,
                              log_geometry=args.log_instance_geometry, infer_process=infer_proxy is not None,
                              infer_process_source=args.infer_process_source, arm_region_filter=arm_mark,
                              belt_region_filter=belt_mark,
                              **log_kw)
            run_kw = {"live_observation_b_input": "list"} if listed else {}
            # 멈춤 때 스레드 호출 위치를 실행 기록 폴더에도 남긴다(D-11)
            runner_config = dataclasses.replace(runner_config,
                                                stall_dump_path=str(Path(args.live_observation_out) / "stall_threads.txt"))
            result = run(source_factory, settings, log, run_id=run_id, config=runner_config, stderr=err,
                         versions={"opencv": cv2.__version__},
                         should_stop=_stop_or_infer_failed(infer_proxy, lambda: (
                             signals.stop.is_set() or link.reset_pending or link.state.failed is not None)),
                         handoff_sink=handoff, live_observation_sink=link, **run_kw, **detector_kw)

        exit_code = result.exit_code
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        err.write(f"run_pipeline: live observation run failed before/while running: {failure}\n")
        exit_code = 1
    finally:
        if signals is not None:
            signals.begin_cleanup()   # 정리 중 신호는 정리를 끊지 않는다

        reset = link.reset_pending
        gripper_problem = None

        # 실행기가 끝났으므로 추론 자식을 먼저 닫는다(실패 사유는 닫기 전에 읽음)
        infer_failed, infer_summary = _finish_infer_process(infer_proxy, err, args)

        # 그리퍼 수신을 먼저 멈춘다(추적 스레드가 닫힌 뒤에는 넣어도 버려진다)
        if gripper_rx is not None:
            try:
                gripper_rx.close(link_config.close_limit_s)
            except Exception as gexc:
                gripper_problem = f"gripper receiver close failed: {type(gexc).__name__}: {gexc}"

            link.world.diagnostics["gripper_receiver"] = {**gripper_rx.receiver.stats(),
                                                          "spin_error": getattr(gripper_rx, "spin_error", None)}

        # 정상 종료: 대기·처리 중인 마지막 항목을 상한 안에서 끝낸 뒤 닫는다(못 끝내면 닫을 때 무효화·기록)
        if not reset and result is not None:
            link.drained = link.dispatcher.drain(link_config.close_limit_s)

        record = controller.service() if reset else controller.stop(
            "startup_failed" if result is None else f"run_end:{result.stop_reason}")
        record = record or link.ended
        problems = []

        if failure is not None:
            problems.append(f"failure: {failure}")

        if gripper_problem is not None:
            problems.append(gripper_problem)

        if infer_failed is not None:
            problems.append(f"infer process failed: {infer_failed}")

        # 팔 영역 제외 요약(실행기가 끝난 뒤라 worker가 더 쓰지 않는다)
        arm_summary = arm_mark if arm_filter is None else {**arm_mark, **arm_filter.summary(),
                                                           "records_written": arm_records.written}

        if arm_records is not None:
            arm_records.close()

            if arm_records.error is not None:
                problems.append(f"arm region record error: {arm_records.error}")

        belt_summary = belt_mark if belt_filter is None else {**belt_mark, **belt_filter.summary(),
                                                              "records_written": belt_records.written}

        if belt_records is not None:
            belt_records.close()

            if belt_records.error is not None:
                problems.append(f"belt region record error: {belt_records.error}")

        if not link.cleanup_confirmed:
            problems.append(f"cleanup not confirmed: {list(record.close_errors)}")

        if link.writer_report is not None and link.writer_report.error:
            problems.append(f"event writer error: {link.writer_report.error}")

        if link.skip_write_error is not None:
            problems.append(f"skip writer error: {link.skip_write_error}")

        skip_counts = link.skip_counts()

        if skip_counts["dropped"] or skip_counts["unwritten"]:
            problems.append(f"skip records lost: dropped={skip_counts['dropped']} unwritten={skip_counts['unwritten']} "
                            f"(capacity {skip_counts['capacity']}, overflow_at {skip_counts['overflow_at']})")

        # 정상 종료·리셋 정리(run_end)가 아닌 상태 실패(쓰기 넘침 등)는 성공으로 숨기지 않는다
        if link.state.failed is not None:
            problems.append(f"state failed: {link.state.failed}")

        if link.sort_part is not None and link.sort_part.problem() is not None:   # 기록 실패를 숨기지 않음
            problems.append(link.sort_part.problem())

        if link.world is not None and link.world.problem() is not None:   # 추적 스레드 비정상 종료·기록 실패
            problems.append(link.world.problem())

        if getattr(link, "list_goal_unmatched", 0):   # 고른 목표를 결과 묶음에서 못 찾음(B 반환 계약 위반)
            problems.append(f"list goal unmatched: {link.list_goal_unmatched}")

        # 정리 순서: 연결부 정리 뒤 center 그리기 자식 → center 발행기 → C 연결부. 자식은 정리 미확인이어도 끝낸다(남기지 않음)
        center_proc_exit = close_center_proc()

        if center_pub is not None:
            if link.cleanup_confirmed:
                try:
                    center_pub.close()
                except Exception as cexc:
                    problems.append(f"center overlay publisher close failed: {type(cexc).__name__}: {cexc}")
            else:
                problems.append("center publisher left open (dispatcher not confirmed stopped)")

        # 추적 흐름: W 추적 스레드가 닫힌 뒤(연결부 정리) 발행기를 닫는다. 정리 미확인이면 발행 중일 수 있어 열어 둔다
        if stream_sender is not None:
            if link.cleanup_confirmed:
                try:
                    stream_sender.close()
                except Exception as texc:
                    problems.append(f"track stream publisher close failed: {type(texc).__name__}: {texc}")
            else:
                problems.append("track stream publisher left open (world tracking not confirmed stopped)")

        if world_stream is not None and world_stream.counts["publish_errors"]:
            problems.append(f"track stream publish errors: {world_stream.counts['publish_errors']} "
                            f"(first: {world_stream.first_error})")

        if handoff_probe is not None:
            try:
                handoff_probe.stop()
            except Exception as hexc:
                problems.append(f"handoff probe stop failed: {type(hexc).__name__}: {hexc}")

        if handoff is not None:
            try:
                handoff.close()
            except Exception as hexc:
                problems.append(f"handoff close failed: {type(hexc).__name__}: {hexc}")

        if link.dispatcher.errors and any(not e.startswith("envelope:") for e in link.dispatcher.errors):
            problems.append(f"worker errors: {[e for e in link.dispatcher.errors if not e.startswith('envelope:')]}")

        if problems:
            exit_code = 1
        elif reset and exit_code == 0:   # 감독 스크립트가 정상 종료와 구분하도록 되감기 종료는 3
            exit_code = EXIT_RESET_DETECTED

        for p in problems:
            err.write(f"run_pipeline: live observation problem: {p}\n")

        try:
            tracking_extra = ({} if not args.tracking else {"tracking": None if result is None else result.tracking})

            if args.tracking:   # 추적 공간과 그 출처(생략 = 실시간 기본 W)
                tracking_extra["tracking_space"] = {"space": "world" if world_on else "pixel",
                                                    "source": getattr(args, "tracking_space_source", "explicit")}

            # C 연결부 종료 보고·center 발행 계수(발행 오류는 종료 코드에 넣지 않음)
            tracking_extra["handoff"] = None if result is None else result.handoff

            if center_on:
                tracking_extra["center_overlay"] = {**center_stats.summary(center_pub), "draw_gate": center_gate.summary()}

                if center_proc is not None:
                    tracking_extra["center_overlay"]["draw_process"] = {**center_proc.summary(), "exit": center_proc_exit}

            if infer_summary is not None:
                tracking_extra["infer_process"] = infer_summary

            if arm_summary is not None:
                tracking_extra["arm_region_filter"] = arm_summary

            if belt_summary is not None:
                tracking_extra["belt_region_filter"] = belt_summary
            link.write_summary({"exit_code": exit_code, "reset_detected": reset, "problems": problems, **tracking_extra,
                                "runner": None if result is None else {
                                    "exit_code": result.exit_code, "stop_reason": result.stop_reason,
                                    "frames": result.frames, "b_calls": result.b_calls,
                                    "live_inputs": result.live_inputs, "errors": result.errors}})
        except OSError as exc:
            exit_code = 1
            err.write(f"run_pipeline: cannot write live observation summary: {exc}\n")

    reason = "reset_detected" if reset else ("startup_failed" if result is None else result.stop_reason)
    err.write(f"run_pipeline: exit={exit_code} reason={reason} live_a_b_record_only run_id={run_id} "
              f"frames={0 if result is None else result.frames} submitted={link.submitted} "
              f"accepted={link.dispatcher.counts.get('accepted', 0)} skips={sum(link.skips.values())} "
              f"cleanup_confirmed={link.cleanup_confirmed}"
              + (f" list_consumer_calls={link.list_consumer_calls} list_candidates_sent={link.list_candidates_sent}"
                 if listed else "")
              + (f" sort_part_events={link.sort_part.summary()['events']}" if sort_on else "")
              + (f" center_published={center_stats.summary(center_pub)['published']} "
                 f"center_errors={center_stats.errors}" if center_on else "")
              + ("" if center_proc is None else
                 f" center_process=failed({center_proc.summary()['failed']})" if center_proc.summary().get("failed")
                 else " center_process=ok")
              + (" (new run requires an explicit restart)" if reset else "") + "\n")

    return exit_code


def _prepare_capture_observation(args, err):
    """실제 경로 준비: 메타 읽기·입력 지문·출력 경로 확인. 실패하면 (None, 2)."""
    from eyetoarm.vision.capture_meta import CaptureMetaError, check_capture_inputs, load_capture_meta

    try:
        meta = load_capture_meta(args.capture_meta)
        paths = list(args.images) if args.images else [args.video]
        inputs = check_capture_inputs(meta, paths, args.source_id)
    except (OSError, CaptureMetaError) as exc:
        err.write(f"run_pipeline: capture meta error: {exc}\n")
        return None, 2

    for path in (args.log, args.observation_out):
        if Path(path).exists():
            err.write(f"run_pipeline: output exists, refusing to overwrite: {path}\n")
            return None, 2

    return (meta, inputs), 0


def _run_capture_list(args, settings, runner_config, detector_kw, capture, err, b_stage_hook,
                      list_consumer_factory) -> int:
    """저장 입력 단계 L: 실제 AI 검출·끝점 추출 결과를 `b_list_gate` 뒤 목록 소비자에 넘긴다(`b_consumer` 없음).

    목록 소비자가 없으면 로그·출력 파일을 만들기 전에 종료 1(단계 T로 바꾸지 않음). 주석 영상은 만들지 않는다.
    """
    from eyetoarm.transforms.b_stage import BStage
    from eyetoarm.transforms.observation_envelope_adapter import EnvelopeTfLookup
    from eyetoarm.vision.capture_list_recorder import CaptureListRecorder
    from eyetoarm.vision.capture_meta import CaptureMetaSource
    from eyetoarm.vision.list_envelope import LIST_ENVELOPE_VERSION
    from eyetoarm.vision.pipeline_records import B_INPUT_LIST

    meta, inputs = capture
    stage = BStage.from_config(REPO_ROOT, meta.camera, EnvelopeTfLookup())

    if b_stage_hook is not None:
        b_stage_hook(stage)

    if stage.on_output is not None:
        err.write("run_pipeline: stage L path requires stage.on_output None (no C/D/robot output)\n")
        return 1

    if list_consumer_factory is not None:
        consume, consumer_kind = list_consumer_factory(stage), "injected_test_only"
    else:
        consume, consumer_kind = getattr(stage, "process_frame", None), "BStage.process_frame"

    if not callable(consume):
        err.write("run_pipeline: --b-input list refused: list consumer unavailable (BStage.process_frame missing); "
                  "not falling back to stage T\n")
        return 1

    det, proxy = detector_kw["detector"], getattr(args, "_infer_proxy", None)
    kind, paths = ("images", list(args.images)) if args.images else ("video", [args.video])
    inner_factory = stored_source_factory(kind, paths, args.source_id)
    holder = {}

    def source_factory(issuer, generation_provider):
        """저장 입력 소스를 촬영 메타 소스로 감싼다."""
        holder["source"] = CaptureMetaSource(inner_factory(issuer, generation_provider), meta)
        return holder["source"]

    try:
        recorder = CaptureListRecorder(args.observation_out, meta, consumer_kind=consumer_kind,
                                       view_bias=_view_bias_runtime(getattr(settings.detector_config, "axis_view_bias", None)))
    except OSError as exc:
        err.write(f"run_pipeline: cannot create a-to-b-out: {exc}\n")
        return 2

    err.write(f"run_pipeline: A->B STAGE L opt-in capture_id={meta.capture_id} b_input={LIST_ENVELOPE_VERSION} "
              f"consumer={consumer_kind}; no C/D/robot output\n")

    try:
        try:
            stream = open(args.log, "x", encoding="utf-8")
        except OSError as exc:
            err.write(f"run_pipeline: cannot open log: {exc}\n")
            return 2

        with stream:
            log = PipelineLog(JsonlRecorder(stream), eval_run_id=args.eval_run_id,
                              log_geometry=args.log_instance_geometry, b_input=B_INPUT_LIST,
                              infer_process=proxy is not None,
                              infer_process_source=args.infer_process_source)
            result = run(source_factory, settings, log, **_detector_run_kw(det, proxy),
                         b_list_consumer=recorder.wrap(consume), display_sink=recorder, config=runner_config,
                         stderr=err, versions={"opencv": cv2.__version__}, should_stop=_stop_or_infer_failed(proxy))
            infer_failed, infer_summary = _finish_infer_process(proxy, err, args)
            infer_record_ok = infer_summary is None or _write_infer_record(log, infer_summary, err)

        problems = []

        if infer_failed is not None:
            problems.append(f"infer process failed: {infer_failed}")

        if not infer_record_ok:
            problems.append("infer_process record not written")

        if recorder.calls + recorder.rejected != result.b_list_calls:
            problems.append(f"recorded list inputs {recorder.calls} + rejected {recorder.rejected} != runner b_list_calls "
                            f"{result.b_list_calls}")

        if recorder.rejected:
            problems.append(f"{recorder.rejected} list envelope(s) rejected (consumer not called)")

        exit_code = 1 if problems and result.exit_code == 0 else result.exit_code
        source = holder.get("source")
        recorder.summary({"exit_code": exit_code, "runner_exit_code": result.exit_code,
                          "stop_reason": result.stop_reason, "run_id": result.run_id, "frames": result.frames,
                          "runner_b_list_calls": result.b_list_calls, "runner_b_calls": result.b_calls,
                          "capture_id": meta.capture_id, "capture_meta_sha256": meta.sha256, "inputs_sha256": inputs,
                          "detector_config_id": settings.detector_config_id,
                          "meta_mismatches": [] if source is None else source.mismatches,
                          "on_output": None, "problems": problems, "runner_errors": result.errors,
                          **({} if infer_summary is None else {"infer_process": infer_summary})})
    finally:
        recorder.close()

        if proxy is not None:
            proxy.close()   # 실행 중 예외로 위 정리를 건너뛴 경우(두 번 불러도 됨)

    for p in problems:
        err.write(f"run_pipeline: stage L mismatch: {p}\n")

    err.write(f"run_pipeline: exit={exit_code} reason={result.stop_reason} a_b_stage_l frames={result.frames} "
              f"b_list_calls={result.b_list_calls} candidates_sent={recorder.candidates_sent} "
              f"input_errors={result.input_errors} timeouts={result.timeouts}\n")

    return exit_code


def _run_capture_observation(args, settings, runner_config, detector_kw, capture, err, b_stage_hook,
                   extension_source=None) -> int:
    """저장 입력 실제 A→B 경로: 실제 AI 검출·끝점 추출 관측을 `b_gate` 뒤 BStage로 보내고 주석 영상을 쓴다.

    BStage·기록기는 이 함수가 만들고 닫는다. `on_output`은 None(C·D·로봇 송신 없음).
    """
    from eyetoarm.transforms.b_stage import BStage
    from eyetoarm.vision.capture_observation_recorder import VALID_MEANING, CaptureObservationRecorder
    from eyetoarm.transforms.observation_envelope_adapter import ObservationEnvelopeAdapter, EnvelopeTfLookup
    from eyetoarm.vision.observation_envelope import OBSERVATION_ENVELOPE_VERSION
    from eyetoarm.vision.capture_meta import CaptureMetaSource

    meta, inputs = capture
    tf_lookup = EnvelopeTfLookup()   # B TF는 지금 소비 중인 입력 봉투의 촬영 시각 TF만
    stage = BStage.from_config(REPO_ROOT, meta.camera, tf_lookup)

    if b_stage_hook is not None:
        b_stage_hook(stage)

    if stage.on_output is not None:
        err.write("run_pipeline: real A->B path requires stage.on_output None (no C/D/robot output)\n")
        return 1

    det, proxy = detector_kw["detector"], getattr(args, "_infer_proxy", None)
    kind, paths = ("images", list(args.images)) if args.images else ("video", [args.video])
    inner_factory = stored_source_factory(kind, paths, args.source_id)
    holder = {}

    def source_factory(issuer, generation_provider):
        """저장 입력 소스를 촬영 메타 소스로 감싼다."""
        holder["source"] = CaptureMetaSource(inner_factory(issuer, generation_provider), meta)
        return holder["source"]

    try:
        recorder = CaptureObservationRecorder(args.observation_out, stage, meta, ObservationEnvelopeAdapter(stage, tf_lookup),
                                   extension_source=extension_source,
                                   view_bias=_view_bias_runtime(getattr(settings.detector_config, "axis_view_bias", None)))
    except OSError as exc:
        err.write(f"run_pipeline: cannot create a-to-b-out: {exc}\n")
        return 2

    err.write(f"run_pipeline: A->B REAL PATH opt-in capture_id={meta.capture_id} meta_sha256={meta.sha256} "
              f"axis_extractor={settings.detector_config.axis_extractor} valid={VALID_MEANING}; no C/D/robot output\n")

    try:
        try:
            stream = open(args.log, "x", encoding="utf-8")
        except OSError as exc:
            err.write(f"run_pipeline: cannot open log: {exc}\n")
            return 2

        with stream:
            log = PipelineLog(JsonlRecorder(stream), eval_run_id=args.eval_run_id,
                              log_geometry=args.log_instance_geometry, infer_process=proxy is not None,
                              infer_process_source=args.infer_process_source)
            result = run(source_factory, settings, log, **_detector_run_kw(det, proxy),
                         b_consumer=recorder.wrap(stage.consume), display_sink=recorder, config=runner_config,
                         stderr=err, versions={"opencv": cv2.__version__}, should_stop=_stop_or_infer_failed(proxy))
            infer_failed, infer_summary = _finish_infer_process(proxy, err, args)
            infer_record_ok = infer_summary is None or _write_infer_record(log, infer_summary, err)

        problems = []

        if infer_failed is not None:
            problems.append(f"infer process failed: {infer_failed}")

        if not infer_record_ok:
            problems.append("infer_process record not written")

        if recorder.calls + recorder.rejected != result.b_calls:
            problems.append(f"recorded B outputs {recorder.calls} + rejected {recorder.rejected} != runner b_calls "
                            f"{result.b_calls}")

        if recorder.rejected:
            problems.append(f"{recorder.rejected} a_to_b_envelope envelope(s) rejected (B not called)")

        exit_code = 1 if problems and result.exit_code == 0 else result.exit_code
        source = holder.get("source")
        recorder.summary({"exit_code": exit_code, "runner_exit_code": result.exit_code,
                          "stop_reason": result.stop_reason, "run_id": result.run_id, "frames": result.frames,
                          "runner_b_calls": result.b_calls, "b_input_version": OBSERVATION_ENVELOPE_VERSION,
                          "capture_id": meta.capture_id,
                          "capture_meta_sha256": meta.sha256, "inputs_sha256": inputs,
                          "detector_config_id": settings.detector_config_id,
                          "meta_mismatches": [] if source is None else source.mismatches,
                          "tf": {"calls": tf_lookup.calls, "unavailable": tf_lookup.unavailable,
                                 "timed_out": tf_lookup.timed_out,
                                 "max_wait_ms": tf_lookup.max_wait_ms},
                          "on_output": None, "problems": problems, "runner_errors": result.errors,
                          "b_adapter": {"extensions_received": recorder._adapter.received_extensions,
                                        "unsupported_received": recorder._adapter.received_unsupported,
                                        "used_in_compute": []},
                          **({} if infer_summary is None else {"infer_process": infer_summary})})
    finally:
        recorder.close()

        if proxy is not None:
            proxy.close()   # 실행 중 예외로 위 정리를 건너뛴 경우(두 번 불러도 됨)

    for p in problems:
        err.write(f"run_pipeline: real-path mismatch: {p}\n")

    err.write(f"run_pipeline: exit={exit_code} reason={result.stop_reason} a_b_real_path frames={result.frames} "
              f"b_calls={result.b_calls} annotated={recorder.annotated} input_errors={result.input_errors} "
              f"timeouts={result.timeouts}\n")

    return exit_code


def _run_synthetic_observation(args, settings, runner_config, detector_kw, synthetic, err, b_stage_hook) -> int:
    """시험 B 연결 실행(synthetic/test-only): 제품 검출기 대신 fixture 관측을 detector 자리에 넣는다.

    관측은 코어 `complete()` → `b_gate` → `stage.consume`을 지난다. `stage.on_output`은 None(송신 없음).
    """
    from eyetoarm.transforms.b_stage import BStage
    from eyetoarm.vision.synthetic_observation_fixture import (TEST_MODE_LABEL, SyntheticObservationRecorder, FixtureDetector, FixtureStampedSource,
                                                FixtureTfLookup)

    fixture, inputs = synthetic
    tf_lookup = FixtureTfLookup(fixture)
    stage = BStage.from_config(REPO_ROOT, fixture.camera, tf_lookup)

    if b_stage_hook is not None:
        b_stage_hook(stage)

    if stage.on_output is not None:
        err.write("run_pipeline: synthetic-a-to-b requires stage.on_output None (no C/robot output)\n")
        return 1

    kind, paths = ("images", list(args.images)) if args.images else ("video", [args.video])
    inner_factory = stored_source_factory(kind, paths, args.source_id)
    detector = FixtureDetector(fixture)

    def source_factory(issuer, generation_provider):
        """저장 입력 소스를 fixture 값 표시 소스로 감싼다."""
        return FixtureStampedSource(inner_factory(issuer, generation_provider), fixture)

    err.write(f"run_pipeline: A->B SYNTHETIC FIXTURE MODE {TEST_MODE_LABEL} fixture_id={fixture.fixture_id} "
              f"fixture_sha256={fixture.sha256} (not real detection, no C/robot output)\n")

    try:
        recorder = SyntheticObservationRecorder(args.synthetic_out, fixture)
    except OSError as exc:
        err.write(f"run_pipeline: cannot open synthetic-a-to-b output: {exc}\n")
        return 2

    try:
        try:
            stream = open(args.log, "x", encoding="utf-8")
        except OSError as exc:
            err.write(f"run_pipeline: cannot open log: {exc}\n")
            return 2

        with stream:
            log = PipelineLog(JsonlRecorder(stream), eval_run_id=args.eval_run_id,
                              log_geometry=args.log_instance_geometry)
            result = run(source_factory, settings, log, run_id=fixture.run_id, detector=detector,
                         b_consumer=recorder.wrap(stage.consume), config=runner_config, stderr=err,
                         versions={"opencv": cv2.__version__}, **detector_kw)

        exit_code = result.exit_code
        expected = fixture.expected_b_calls
        problems = []

        # 시험 판정: valid 레코드는 모두 b_gate를 통과해야 하고, 그 밖은 B로 가면 안 된다
        if result.b_calls != expected:
            problems.append(f"b_calls {result.b_calls} != expected {expected} (valid fixture records)")

        if recorder.calls != result.b_calls:
            problems.append(f"recorded B outputs {recorder.calls} != runner b_calls {result.b_calls}")

        if problems and exit_code == 0:
            exit_code = 1

        recorder.summary({"exit_code": exit_code, "runner_exit_code": result.exit_code,
                          "stop_reason": result.stop_reason, "run_id": result.run_id, "frames": result.frames,
                          "runner_b_calls": result.b_calls, "expected_b_calls": expected,
                          "fixture_records_used": detector.used, "inputs_sha256": inputs,
                          "tf": {"mode": fixture.tf_mode, "camera_motion": fixture.camera_motion,
                                 "calls": tf_lookup.calls, "unavailable": tf_lookup.unavailable,
                                 "timed_out": tf_lookup.timed_out},
                          "on_output": None, "problems": problems, "runner_errors": result.errors})
    finally:
        recorder.close()

    for p in problems:
        err.write(f"run_pipeline: synthetic-a-to-b mismatch: {p}\n")

    err.write(f"run_pipeline: exit={exit_code} reason={result.stop_reason} synthetic={TEST_MODE_LABEL} "
              f"frames={result.frames} b_calls={result.b_calls} expected_b_calls={expected} "
              f"input_errors={result.input_errors}\n")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
