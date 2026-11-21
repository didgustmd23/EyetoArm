#!/usr/bin/env python3
"""A 비전 파이프라인 저장 입력 실행 CLI(계획 r4 작업 6a). 기존 `run_vision.py`는 그대로 둔다.

사용 예(작업 디렉터리 저장소 루트):
    PYTHONPATH=src .venv/bin/python scripts/run_pipeline.py --images a.png b.png \\
        --detect-config config/vision/opencv_planar.json --log outputs/run.jsonl
    PYTHONPATH=src .venv/bin/python scripts/run_pipeline.py --video clip.avi \\
        --detect-config cfg.json --log run.jsonl --max-frames 30

- 입력 하나(`--images` 또는 `--video`), 검출 설정 JSON(`load_config`), 기록 파일(JSONL)을 받는다.
- 처리 프로파일은 없다(저장 입력은 보정 프로파일이 없음). 그래서 표시용 관측의 사유는 `CALIBRATION_MISMATCH`이며
  B 소비 지점은 쓰지 않는다. 탐색 옵션은 없다(계획 범위).
- C 서버·ROS·YOLO를 연결하지 않는다. 영상 파일 열기 제한시간을 두지 않는다(사용자 결정 2026-11-13).
- 검출 감시(작업 6b): 기본으로 켠다. 첫 검출 10 s·이후 1 s(사용자 확정 개발 초기값, 운영 최종값 아님)이며
  `--first-detect-timeout`·`--detect-timeout`(초)으로 조절한다. `--no-detect-timeout`은 6a 동기 경로(감시 끔)다.
  유예·정리 한도·확인 간격은 A 내부 임시 기본안(0.2/0.3/0.02 s)이며 CLI 옵션이 없다.
- 검출 방식(YOLO 연결 R3): `--detector opencv`(기본, `--detect-config` 필수, torch·ultralytics를 import하지 않음) 또는
  `--detector ai`(`--ai-config` 필수). AI는 로컬 분할 모델의 **표시용 무효 관측**(valid=False)만 낸다. 모델 준비(파일 확인·
  해시·import·생성·검사)는 실행기가 감시하며 상한은 `--model-load-timeout`(기본 10 s, A 임시 개발값)이다. 신뢰도 기준 등
  모델 설정은 AI 설정 JSON(`conf_threshold`)으로 조절한다. `--log-instance-geometry`이면 인스턴스 박스·다각형도 쓴다.
  준비 실패(가중치 없음·해시 불일치·라이브러리 미설치 등)는 설정 오류(2)가 아니라 실행 실패(1)이며 사건으로 남는다.
- 실시간 링크 카메라(계획 `docs/superpowers/plans/2026-11-16-live-link-camera.md` r3, L4): `--ros-link`이면 ROS 2
  `/link_cam/image`·`/link_cam/camera_info`·`/clock`을 받는다(`source /opt/ros/jazzy/setup.bash` 필요). rclpy를 찾지 못하면
  인자 오류(2)다. 소스 ID 기본값은 `gz_link`(A 임시, 계획 결정 D4). 첫 SIGINT·SIGTERM은 정상 정지 요청(`should_stop`, 실패가
  없으면 0), 두 번째는 즉시 중단(1)이다. 수신 중단(`FRAME_TIMEOUT`)은 입력 오류로 세지 않는다(A 임시, 결정 D5). B 소비는
  하지 않는다(결정 D13). 실연결 실행 예:
      source /opt/ros/jazzy/setup.bash
      PYTHONPATH=src .venv/bin/python scripts/run_pipeline.py --ros-link --detector ai \\
          --ai-config config/vision/ai_s1_01_live.json --log _workspace/evidence/live/run.jsonl --max-frames 300
- **A→B 저장 입력 시험 연결(#166, synthetic/test-only):** `--synthetic-observation-fixture JSON --synthetic-observation-out PATH`(별칭 `--synthetic-a-to-b-fixture`·`--synthetic-a-to-b-out`)이면(저장 입력
  `--images`/`--video` 전용) fixture의 합성 관측을 실행기 detector 자리에 넣고, fixture의 `CameraProfile`·TF로 만든
  `BStage`의 `consume`을 `run(..., b_consumer=...)`에 넘긴다. 관측은 코어 검사 → `b_gate` → B 순서를 그대로 지난다.
  제품 AI/OpenCV 검출은 이 모드에서 돌지 않으며(모델을 불러오지 않음) 그 관측은 여전히 valid=False다. `stage.on_output`은
  연결하지 않는다(C·로봇 송신 없음). `--ros-link`(D13)·`--tracking`과 함께 쓸 수 없다(인자 오류 2). 기록·B 출력 파일이 이미
  있으면 덮어쓰지 않고 2로 끝낸다. 실행 뒤 B 호출 수가 fixture의 valid 레코드 수와 다르면 1이다. 합성 fixture 통과는 실제
  RGB 검출·끝점 추출·mm 정확도·실시간 B 허용의 증거가 아니다.
- **A→B 저장 입력 실제 경로(#168, opt-in):** `--capture-meta META --observation-out DIR`(별칭 `--a-to-b-capture-meta`·`--a-to-b-out`, 저장 입력 `--images`/`--video`·`--detector ai`
  전용, AI 설정 `axis_extractor` 필수). 관측은 **실제 AI 검출 + 볼트 끝점 추출기(M1)**에서 나오고, 메타는 원본 CameraInfo·
  프레임별 촬영 시각·그 시각 `world→optical_frame` TF만 준다(관측 주입 없음). B 입력은 `run(..., b_consumer=...)`의 `b_gate`
  통과 관측과 그 원본 프레임이며, B 결과와 같은 `BOutput`의 주석 영상(`render_center`)을 DIR에 새로 쓴다. `valid=True`는 추출기
  형식 조건 통과 뜻이며 정확도·운영 승인이 아니다(#167 K0). `stage.on_output`은 연결하지 않는다(C·D·로봇 송신 없음).
  `--ros-link`(D13)·`--tracking`·`--synthetic-observation-*`과 함께 쓸 수 없다(인자 오류 2). 메타·출력 오류는 2.
  A→B 경계는 버전 명시형 입력 봉투(`vision/observation_envelope`, `eyetoarm.b_input/1`)이며 봉투 오류는 B 없이 기록하고 1로 끝낸다.
- **실시간 A→B 기록 전용 시험(#173, opt-in):** `--ros-link --detector ai --live-observation-out DIR --live-observation-config JSON`
  (AI 설정 `axis_extractor` 필수). 기본 `--ros-link`와 실시간 `b_consumer`는 그대로 D13으로 거부한다. 이 옵션에서만 실행기의
  `live_observation_sink`에 전용 디스패처(`vision/live_observation_link`)를 연결한다: 같은 프레임 영상·실제 AI 검출 끝점·CameraInfo의
  `CameraProfile`·보정 번호·촬영 stamp를 기존 관측 봉투·B 입구 어댑터로 넘기고, 그 촬영 stamp의 TF 정확 조회가 상한 안에 성공할 때만
  B 좌표 계산과 같은 프레임 주석을 기록한다. TF 누락·상한 초과·시각/보정 불일치·무효 관측·폐기된 세대는 `skipped.jsonl`에 사유만
  남긴다. 대기 1개 최신 우선, 무효화·수락 직렬화는 준비 부품 그대로다. Gazebo 리셋(시뮬레이션 시각 되감기)을 감지하면 이 실행을
  끝내고(이전 결과 철회) 정리한다. **자동 재시작은 없다**: 새 실행은 이 명령을 다시 실행해야 하며, 같은 프로세스 안에서는 이전
  실행의 정리 완료가 확인되기 전 새 실행을 거부한다(1). `stage.on_output`은 쓰지 않는다(C·D 송신·로봇 목표 없음). 설정 파일의
  값은 미확정 운영값의 시험 설정이며 `"status": "test_only"`가 필수다(`config/vision/live_observation_test.json`). 기록·출력이 이미
  있으면 덮어쓰지 않고 2. `--capture-meta`·`--synthetic-observation-*`과 함께 쓸 수 없다(2). `--tracking`(#215)을 더하면 확정 추적
  번호만 관측·봉투에 실어 B 중복 방지(`_dedupe`·`release`)까지 넘긴다(M2 휴리스틱, 분열은 알려진 한계). 단계 L(#244)은 후보마다
  확정 번호를 A 하위 형 후보로 싣는다(B가 읽기 전에는 송신기가 고른 후보의 A 번호를 기록·재시도·require_track_id에 씀).
  `--b-input list`(#201)를 더하면 같은 연결·같은 프레임 단위 검사로 유효 볼트 목록(`eyetoarm.b_input/2`, 촬영 stamp TF 하나)을 B 목록
  소비자(`BStage.process_frame`)에 넘기고 소비자 결과를 기록한다(주석 없음). 소비자가 없으면 시작 때 1(단계 T로 바꾸지 않음).
- **실시간 C 목표 송신(#214 D13 해제, opt-in):** `--live-observation-out`에 `--live-sort-part`를 더하면 검사 ③이 수락한 B 결과의
  C 목표(`PICK_PLACE`·`PASS_THROUGH`)를 B `SortPartClient`로 ROS 액션 `/sort_part`에 보낸다(`vision/live_sort_part`, 사용자 결정
  2026-11-20 「승인하고 위험으로 기록만해」, 해제 전 시험 F6 없이 수용된 위험: A 상세설계 §11 「D13 해제 위험」). 송신은 수락 뒤 worker의
  게이트 안에서만 하며 `stage.on_output`은 쓰지 않는다(수락 전 송신 없음). 실행 닫힘(리셋·종료·fail-stop) 뒤 송신 0. 단계 T, 또는
  단계 L(#233: B `process_frame`이 고른 1개만, 사진당 최대 1회, 추적 번호 없음이라 재시도 제한은 실패 지점 차단만). 설정 블록 `sort_part` 필수(짝이 아니면 2). C 서버가 없어도 실행하고 `server_seen=false` 경고·`no_server`로 기록한다. 기록은
  출력 폴더 `sort_part.jsonl`. `eyetoarm_interfaces`(robot_ws 빌드·`source robot_ws/install/setup.bash`)가 없으면 1. 실행기 D13 가드
  (기본 `--ros-link`의 실시간 `b_consumer` 거부)는 그대로다.
- **한 실행 연결(#233):** 실시간 A→B 실행(`--live-observation-out`)도 기본 `--ros-link`처럼 A 검출 결과를 C 연결부
  (`AToCHandoff` + `HandoffProbe`, 127.0.0.1:47093 → C `HandoffIngress`)로 항상 보낸다. 단계 L에 `--b-center-overlay`를 더하면 목록 소비자
  자리에 B `center_overlay_consumer`(`/b/center_overlay/image`, 인터페이스가 있으면 `/b/center_overlay`도)를 쓴다. 발행은 B 함수 설계대로
  B 계산 직후(검사 ③ 전)이며, 발행 실패는 요약·마지막 줄에만 남고 종료 코드에 반영하지 않는다. 단계 T에서는 거부한다(2).
- 종료 코드: 0 정상(EOF·max-frames·정지 요청), 1 실행 실패(입력 없음·기록 실패·중단·close 실패·검출 시간 초과·멈춤 등),
  2 인자·설정 오류.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import signal
import sys
import threading
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
    # 관측 전달 옵션: 역할 이름이 기본, #166/#168 이름(--synthetic-a-to-b-*, --a-to-b-*)은 같은 옵션의 별칭으로 유지한다
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
    parser.add_argument("--live-sort-part", dest="live_sort_part", action="store_true",
                        help="--live-observation-out 전용(#214 D13 해제): 수락된 B 결과의 C 목표를 /sort_part로 송신. 단계 T, 또는 "
                             "단계 L(#233, B가 고른 1개만). 설정 sort_part 필수. robot_ws의 eyetoarm_interfaces 빌드·source 필요")
    parser.add_argument("--b-center-overlay", dest="b_center_overlay", action="store_true",
                        help="--live-observation-out --b-input list 전용(#233): 목록 소비자 자리에 B center_overlay_consumer를 붙여 "
                             "/b/center_overlay/image(인터페이스가 있으면 /b/center_overlay도)로 B center 주석 영상을 발행")
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


def _check_source_args(args, ros_available) -> None:
    """입력 인자 조합을 확인한다. --ros-link 전용 옵션 오용·rclpy 없음은 argparse 오류(종료 2)."""
    if args.ros_link:
        available = ros_available() if ros_available is not None else importlib.util.find_spec("rclpy") is not None

        if not available:
            build_parser().error("--ros-link requires rclpy (run: source /opt/ros/jazzy/setup.bash)")
    elif args.ros_reliability is not None:
        build_parser().error("--ros-reliability is for --ros-link only")

    # 실제 A→B 경로(#168): 저장 입력·AI 전용. 실시간 B 금지(D13)·추적·시험 fixture와 결합 금지
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

    # 단계 L(#189): 저장 입력 실제 경로(--capture-meta, 이미 ros-link·synthetic·tracking·opencv 결합 금지)에서만
    # #201: 실시간은 기록 전용 연결(--live-observation-out) 안에서만. 기본 --ros-link의 B 소비는 D13 그대로 거부
    if args.b_input == "list" and args.capture_meta is None and args.live_observation_out is None:
        build_parser().error("--b-input list requires --capture-meta/--observation-out (stored input) or "
                             "--ros-link --live-observation-out (live record-only); D13: no B in plain live")

    # 실시간 A→B 기록 전용 시험(#173): 명시 옵션에서만 전용 디스패처. 기본 --ros-link는 D13 그대로
    if args.live_observation_out is not None or args.live_observation_config is not None:
        if args.live_observation_out is None or args.live_observation_config is None:
            build_parser().error("--live-observation-out and --live-observation-config must be given together")
        if not args.ros_link:
            build_parser().error("--live-observation-out requires --ros-link")
        if args.detector != "ai":
            build_parser().error("--live-observation-out requires --detector ai")

    # #239: 추적 공간 생략 시 실시간 기록 연결(--live-observation-out)은 W, 그 밖(저장 입력·기본 --ros-link 등)은 픽셀.
    # W 설정 파일은 숨은 기본 경로 없이 명시해야 한다(설정은 모두 test_only, 기본값 없음 규칙)
    args.tracking_space_source = "explicit" if args.tracking_space is not None else "default"

    if args.tracking_space is None:
        args.tracking_space = "world" if args.tracking and args.live_observation_out is not None else "pixel"

        if args.tracking_space == "world" and args.world_tracker_config is None:
            build_parser().error("--live-observation-out --tracking uses W tracking by default: --world-tracker-config JSON "
                                 "is required (e.g. config/vision/world_tracker_test.json), or give --tracking-space pixel")

    # #217: W 추적은 실시간 기록 연결 안에서만(촬영 시각 TF 필요)
    if args.tracking_space == "world":
        if not args.tracking:
            build_parser().error("--tracking-space world requires --tracking")
        if args.live_observation_out is None:
            build_parser().error("--tracking-space world requires --live-observation-out (needs capture-time TF)")
        if args.world_tracker_config is None:
            build_parser().error("--tracking-space world requires --world-tracker-config")
    elif args.world_tracker_config is not None:
        build_parser().error("--world-tracker-config is for --tracking-space world only")

    # C 목표 송신(#214): 기록 전용 연결의 수락 결과만(#233: 단계 L은 B가 고른 1개)
    if args.live_sort_part:
        if args.live_observation_out is None:
            build_parser().error("--live-sort-part requires --live-observation-out (live accepted B results only)")

    # #233: B center 발행기는 실시간 기록 연결의 단계 L(목록 소비자 자리)에서만
    if args.b_center_overlay:
        if args.live_observation_out is None:
            build_parser().error("--b-center-overlay requires --live-observation-out")
        if args.b_input != "list":
            build_parser().error("--b-center-overlay is for stage L only (--b-input list); stage T has no center source")

    # 시험 B 연결(#166): 저장 입력 전용. 실시간 입력의 B 소비 금지(D13)·추적과 B 결합 금지를 인자 단계에서 막는다
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
    """실시간 링크 카메라 소스 팩토리. ROS 어댑터는 이때만 import한다(rclpy는 연결 시작 때). `boundary_hook`은 기록 전용 시험만 준다."""
    from eyetoarm.inputs.ros_image import LiveLinkSource, LiveSourceConfig

    config = LiveSourceConfig() if reliability is None else LiveSourceConfig(reliability=reliability)

    def factory(issuer, generation_provider):
        """issuer·generation_provider를 받아 소스를 만든다."""
        return LiveLinkSource(source_id=source_id, issuer=issuer, generation_provider=generation_provider,
                              config=config, boundary_hook=boundary_hook)

    return factory


class _StopSignals:
    """실시간 실행의 신호 처리: 첫 SIGINT·SIGTERM은 정지 요청, 두 번째는 KeyboardInterrupt. 끝나면 원래 처리기로 되돌린다."""

    def __init__(self) -> None:
        """정지 표지를 만든다."""
        self.stop = threading.Event()
        self._hits = 0
        self._old: dict = {}

    def _on_signal(self, signum, frame) -> None:
        """신호 처리기."""
        self._hits += 1

        if self._hits == 1:
            self.stop.set()
            return

        raise KeyboardInterrupt

    def __enter__(self) -> "_StopSignals":
        """주 스레드에서만 처리기를 설치한다."""
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                self._old[sig] = signal.signal(sig, self._on_signal)

        return self

    def __exit__(self, *exc) -> None:
        """원래 처리기로 되돌린다."""
        for sig, handler in self._old.items():
            signal.signal(sig, handler)


REPO_ROOT = Path(__file__).resolve().parents[1]
PART_SPECS_PATH = REPO_ROOT / "config/calibration/part_specs.json"   # B 파일(읽기만, BStage.from_config와 같은 경로)
VIEW_BIAS_PLANE_TOL_M = 1e-9


def _check_view_bias_plane(bias, specs_path: Path | None = None) -> None:
    """#207: 보정 `plane_z_m`이 B 볼트 기준 높이(BOLT_A·BOLT_B `plane_z`, B `class_plane_z`)와 같은지 본다. 다르면 ValueError.

    `specs_path`가 None이면 부를 때의 `PART_SPECS_PATH`(B 파일, 읽기만)를 쓴다.
    """
    from eyetoarm.transforms.work_result import class_plane_z, load_specs

    specs_path = PART_SPECS_PATH if specs_path is None else specs_path
    specs = load_specs(specs_path)
    heights = {"bolt": class_plane_z(specs, "bolt"), "BOLT_A": class_plane_z(specs, "bolt", "BOLT_A"),
               "BOLT_B": class_plane_z(specs, "bolt", "BOLT_B")}

    for name, z in heights.items():
        if abs(z - bias.plane_z_m) > VIEW_BIAS_PLANE_TOL_M:
            raise ValueError(f"axis_view_bias.plane_z_m {bias.plane_z_m!r} != B part_specs {name} plane_z {z!r} ({specs_path})")



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


def main(argv=None, *, stderr=None, model_factory=None, import_module=None, ros_available=None,
         live_source_factory=None, b_stage_hook=None, extension_source=None, live_tf_factory=None,
         live_run_controller=None, list_consumer_factory=None,
         live_sort_part_factory=None, live_conveyor_probe=None, live_handoff_factory=None,
         center_publisher_factory=None) -> int:
    """CLI 진입점. 종료 코드를 돌려준다.

    `b_stage_hook(stage)`는 시험 주입 지점이다(시험 B 연결에서 만든 BStage를 run 전에 받음, 기본 None).
    `extension_source(frame, obs)`는 실제 A→B 경로 봉투에 실을 확장 사전을 주는 시험 주입 지점이다(기본 None = 확장 없음).

    `ros_available`(rclpy 확인 대체)·`live_source_factory`(`(source_id, reliability) → 소스 팩토리`)는 시험 주입 지점이다.
    실시간 기록 전용 시험에서는 `live_source_factory(source_id, reliability, boundary_hook=)`로 부르고, `live_tf_factory(config)`(시작된 TF
    수신 묶음)·`live_run_controller`(프로세스 단위 `LiveRunController`, 기본은 모듈 전역)도 시험 주입 지점이다.

    `model_factory`·`import_module`은 시험 주입 지점이다(AI 어댑터에 그대로 넘김, 기본은 실제 ultralytics 지연 import).

    `list_consumer_factory(stage)`(#189)는 단계 L 시험 주입 지점이다. 목록 봉투(`eyetoarm.b_input/2`)를 받는 함수를 돌려준다.
    None(기본)이면 `stage.process_frame`을 쓰고, 없으면 단계 L을 시작 때 거부한다(종료 1). 시험 밖에서 쓰지 않는다.

    `live_sort_part_factory(sort_part_config)`(#214)는 C 송신 런타임 시험 주입 지점이다(기본 `_RosSortPartRuntime`).

    `live_conveyor_probe()`(#217)는 C `/conveyor_controller`의 `speed_mps` 진단 조회 시험 주입 지점이다(기본 `_ros_conveyor_speed_probe`).
    결과는 요약 진단값으로만 쓰고 추적기에는 넘기지 않는다(정답 격리).
    `live_handoff_factory()`(#233)는 실시간 A→B 실행의 C 연결부 시험 주입 지점이다. `(handoff, probe)`를 돌려준다(기본 `AToCHandoff` +
    `HandoffProbe`). `center_publisher_factory()`(#233)는 B center 발행기 시험 주입 지점이다(기본 B `CenterOverlayPublisher`).
    """
    err = stderr if stderr is not None else sys.stderr
    args = build_parser().parse_args(argv)  # 인자 오류는 argparse가 종료 2로 끝낸다
    _check_source_args(args, ros_available)
    args._model_factory, args._import_module = model_factory, import_module
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
            raise ValueError("--capture-meta requires an AI config with axis_extractor (e.g. \"m1\")")

        # #207 시선 기반 끝점 보정: 켠 설정만 시작 때 대조한다(키 없음이면 아무것도 읽지 않음)
        if getattr(config, "axis_view_bias", None) is not None:
            if synthetic is not None:
                raise ValueError("axis_view_bias is not supported with --synthetic-observation-fixture (no AI endpoints)")

            _check_view_bias_plane(config.axis_view_bias)

            if capture is None and args.live_observation_out is None:
                err.write("run_pipeline: axis_view_bias inactive: no B input path (needs --capture-meta or "
                          "--live-observation-out)\n")
        settings = PipelineSettings(source_id=args.source_id, mode=args.mode, detector_config=config, profile=profile)
    except (OSError, ValueError) as exc:
        err.write(f"run_pipeline: config error: {exc}\n")
        return 2

    # 추적 opt-in(#151): 진입점이 프로세스당 발급기 1개를 만들어 주입한다
    # W 추적(#217)은 연결부 추적 스레드가 하므로 실행기 픽셀 추적기를 만들지 않는다
    if args.tracking and args.tracking_space == "pixel":
        detector_kw["part_tracker"] = PartTracker(PartTrackIdIssuer(), mode=TrackMode.HEURISTIC)

    if synthetic is not None:
        return _run_synthetic_observation(args, settings, runner_config, detector_kw, synthetic, err, b_stage_hook)

    if args.detector == "ai":
        # AI 어댑터는 이때만 import한다(OpenCV 경로는 ai_detector도 불러오지 않음). 생성자는 I/O가 없다
        from eyetoarm.vision.ai_detector import AiSegDetector

        inject = {"model_factory": model_factory}

        if import_module is not None:
            inject["import_module"] = import_module

        det = AiSegDetector(config, **inject)
        detector_kw.update(detector=det, prepare=det.prepare)

    if args.live_observation_out is not None:
        return _run_live_observation(args, settings, runner_config, detector_kw, err, b_stage_hook=b_stage_hook,
                                     live_source_factory=live_source_factory, live_tf_factory=live_tf_factory,
                                     controller=live_run_controller or _live_runs(),
                                     list_consumer_factory=list_consumer_factory,
                                     live_sort_part_factory=live_sort_part_factory,
                                     live_conveyor_probe=live_conveyor_probe,
                                     live_handoff_factory=live_handoff_factory,
                                     center_publisher_factory=center_publisher_factory)

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

    # #196 A4-1: 기본 경로도 기존 기록을 덮어쓰지 않는다(다른 경로와 같은 "x" 규칙, 덮어쓰기 옵션 없음)
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
    try:
        with stream:
            log = PipelineLog(JsonlRecorder(stream), eval_run_id=args.eval_run_id,
                              log_geometry=args.log_instance_geometry)
            if args.ros_link:
                with _StopSignals() as signals:
                    result = run(source_factory, settings, log, config=runner_config, stderr=err,
                                 versions={"opencv": cv2.__version__}, should_stop=signals.stop.is_set,
                                 handoff_sink=handoff, **detector_kw)
            else:
                result = run(source_factory, settings, log, config=runner_config, stderr=err,
                             versions={"opencv": cv2.__version__}, handoff_sink=handoff, **detector_kw)
    finally:
        probe.stop()
        handoff.close()

    watch = (f"{runner_config.first_detect_timeout_s}/{runner_config.detect_timeout_s}" if runner_config.watchdog
             else "off")
    method = f"detector=ai model_id={config.model_id}" if args.detector == "ai" else "detector=opencv"
    err.write(f"run_pipeline: exit={result.exit_code} reason={result.stop_reason} {method} frames={result.frames} "
              f"input_errors={result.input_errors} detect_timeout={watch} timeouts={result.timeouts} "
              f"stalled={result.stalled}"
              + (f" live_inputs={result.live_inputs} input_boundary_aborts={result.input_boundary_aborts}"
                 if args.ros_link else "") + "\n")

    return result.exit_code


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
    """실제 TF 수신 묶음: 자기 rclpy Context + `LiveTfRuntime`(B 검토 대상 제안 부품). close는 런타임 → Context 순서."""

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
        self.set_failure_callback = self._runtime.set_failure_callback   # #196 A1-2: 연결부가 수신 스레드 실패를 받는다
        self._shutdown = False

    def close(self, limit_s: float) -> None:
        """런타임 정리(실패하면 RuntimeError, 재시도 가능) → Context 종료."""
        self._runtime.close(limit_s)

        if not self._shutdown:
            self._context.try_shutdown()
            self._shutdown = True


def _new_world_tracker(config):
    """W 추적기(#217): 이 실행 전용 발급기(프로세스 수명 번호 규칙은 픽셀 추적과 같음)."""
    from eyetoarm.vision.world_tracking import WorldTracker

    return WorldTracker(PartTrackIdIssuer(), config=config)


def _ros_conveyor_speed_probe(timeout_s: float) -> dict:
    """C `/conveyor_controller` 노드의 `speed_mps` 파라미터를 시작 때 1회 읽는다(진단 전용, C 코드 무변). 자기 Context를 쓰고 닫는다.

    노드가 없으면 `status="no_node"`, 응답이 상한 안에 없으면 `no_reply`(실패는 진단 기록 `status`에만, #226 D2). 상한 `timeout_s`는
    W 추적 시험 설정 `live.conveyor_probe_timeout_s`. 이 값은 예측·짝짓기·번호 판정에 쓰지 않는다(보충 2, 정답 격리).
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
    """C 송신 런타임(#214): 자기 rclpy Context + 노드 + `SingleThreadedExecutor` spin 스레드 + `RecordingSortPartClient`(B 클라이언트).

    spin 스레드는 액션 응답·결과·피드백 콜백만 처리한다(결과 수신에 spin 필요, #167 K-T·K6). `send`는 worker가 부른다(기다리지 않음).
    시작 때 `server_wait_s` 동안 서버를 기다려 `server_seen`을 남긴다(없어도 실행, Q5). 사건은 `event_sink`(연결부 송신기)로 넘긴다.
    """

    NODE_NAME = "eyetoarm_live_sort_part"

    def __init__(self, config) -> None:
        """Context·노드·클라이언트·spin 스레드. 인터페이스가 없으면 ImportError(진입점 종료 1)."""
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.signals import SignalHandlerOptions

        from eyetoarm.vision.live_sort_part import RecordingSortPartClient

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

        try:
            self._node = rclpy.create_node(self.NODE_NAME, context=self._context)
            self.client = RecordingSortPartClient(self._node, self._emit, stage=None)
            self._executor = SingleThreadedExecutor(context=self._context)
            self._executor.add_node(self._node)
            self._thread = threading.Thread(target=self._spin, name="eyetoarm-live-sort-part-spin", daemon=True)
            self._thread.start()
            self.server_seen = bool(self.client.client.wait_for_server(timeout_sec=config.server_wait_s))
        except BaseException:
            self.close(1.0)
            raise

    def set_release_target(self, target) -> None:
        """#215: B 클라이언트의 `stage` 자리(결과·no_server·busy 때 `release(번호)`를 받을 곳)."""
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


CENTER_CONSUMER_KIND = "center_overlay(BStage.process_frame)"   # #233: 기록·시작 줄의 목록 소비자 이름


class _CenterOverlayStats:
    """B center 감싸개(#233) 공용 오류 계수. 프로필 변경으로 단계·감싸개가 새로 생겨도 한 곳에 모은다(worker가 씀)."""

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
    """기본 `--ros-link`와 같은 C 연결부 묶음: `(AToCHandoff, HandoffProbe)`. 시작(`probe.start`)은 호출자가 한다."""
    probe = HandoffProbe()

    return AToCHandoff(notify=probe.notify), probe


def _run_live_observation(args, settings, runner_config, detector_kw, err, *, b_stage_hook, live_source_factory,
                          live_tf_factory, controller, list_consumer_factory=None,
                          live_sort_part_factory=None, live_conveyor_probe=None, live_handoff_factory=None,
                          center_publisher_factory=None) -> int:
    """실시간 A→B 기록 전용 시험(#173). 호출 경로: LiveLinkSource(경계 훅 = 연결부) → 실행기 → AiSegDetector(M1) → 코어 →
    `handle()` 기록·표시 뒤 `live_observation_sink.offer` → 전용 디스패처 → 촬영 stamp TF → 봉투 → B 입구 어댑터 → BStage → 주석 → 순번
    기록기. 리셋이면 실행을 멈추고 정리한다(자동 재시작 없음).

    `--b-input list`(#201): 같은 연결부가 목록 봉투(`eyetoarm.b_input/2`)를 만들어 목록 소비자(시험 주입 `list_consumer_factory(stage)`
    또는 `BStage.process_frame`)를 부른다. 소비자가 없으면 TF·출력·기록을 만들기 전에 종료 1(단계 T로 바꾸지 않음). 주석 없음.

    #233 한 실행 연결: C 연결부(`AToCHandoff` + `HandoffProbe`)를 항상 만들어 실행기 `handoff_sink`로 넘긴다(기본 `--ros-link`와 같음).
    `--b-center-overlay`(단계 L)면 목록 소비자를 B `center_overlay_consumer(stage, 발행기)`로 둔다(단계마다 새 감싸개, 발행기 하나).
    정리 순서: 연결부 정리(dispatcher 정지) → center 발행기 닫기(정리 미확인이면 열어 두고 종료 1) → `probe.stop` → `handoff.close`.
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

    sort_on = bool(getattr(args, "live_sort_part", False))

    if sort_on != (link_config.sort_part is not None):   # #214: 옵션과 설정 블록은 짝
        err.write("run_pipeline: live observation config error: "
                  + ("--live-sort-part requires config block sort_part\n" if sort_on
                     else "config block sort_part given without --live-sort-part\n"))
        return 2

    # #239 D4: 추적 켬이면 require_track_id 필수(기본값 없음). 번호가 없는 실행(추적 끔)의 true는 볼트 송신 0이라 거부.
    # #244: 단계 L + 추적은 고른 후보의 A 번호로 판정하므로 허용
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

    # #217: W 추적 설정·볼트 평면 높이(B 규격 읽기만)는 출력·세션을 만들기 전에 확인한다
    if world_on:
        from eyetoarm.transforms.work_result import class_plane_z, load_specs
        from eyetoarm.vision.live_world_tracking import LagConfigError, load_world_tracking_config

        try:
            world_cfg = load_world_tracking_config(args.world_tracker_config)
            world_plane_z = class_plane_z(load_specs(REPO_ROOT / "config/calibration/part_specs.json"), "bolt")
        except (OSError, LagConfigError) as exc:
            err.write(f"run_pipeline: world tracking config error: {exc}\n")
            return 2

    if listed and list_consumer_factory is None and not callable(getattr(BStage, "process_frame", None)):
        err.write("run_pipeline: --b-input list refused: list consumer unavailable (BStage.process_frame missing); "
                  "not falling back to stage T\n")
        return 1

    center_on = bool(getattr(args, "b_center_overlay", False))

    # #233: B center 소비자는 목록 소비자 자리 하나다. 시험 주입 소비자와 함께 쓰면 어느 쪽인지 모호하므로 거부한다
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
    center_stats = _CenterOverlayStats()

    def close_sort_rt():
        """세션이 만들어지지 않았을 때만: 이 함수가 만든 송신 런타임·center 발행기(#233)를 닫는다."""
        if sort_rt is not None:
            try:
                sort_rt.close(link_config.close_limit_s)
            except Exception as sexc:
                err.write(f"run_pipeline: sort_part runtime close failed: {type(sexc).__name__}: {sexc}\n")

        if center_pub is not None:
            try:
                center_pub.close()
            except Exception as cexc:
                err.write(f"run_pipeline: center overlay publisher close failed: {type(cexc).__name__}: {cexc}\n")

    # #214: C 송신 런타임은 TF·세션 앞(인터페이스가 없으면 아무것도 만들지 않고 1)
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

    # #233: B center 발행기도 TF·세션 앞(rclpy·sensor_msgs가 없으면 아무것도 만들지 않고 1). 프로세스에 하나, 단계마다 감싸개만 새로
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

    if center_on:
        from eyetoarm.transforms.center_overlay_pub import center_overlay_consumer

        consumer_factory = lambda stage: center_overlay_consumer(stage, center_pub, on_error=center_stats.on_error)  # noqa: E731
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

    # #233 r1: 이 실행의 실제 출구(요약 `outputs`·시작 줄이 같은 목록). C 연결부는 기본 --ros-link처럼 항상이다
    run_outputs = (f"c_handoff_detect({_handoff_address()})",
                   *((f"b_center_overlay({getattr(center_pub, 'topic', None)})",) if center_on else ()),
                   *(("sort_part(C goals via /sort_part, accepted results only)",) if sort_on else ()))

    try:
        link = controller.start_run(run_id, make_session=lambda rid: LiveObservationLink(
            rid, args.live_observation_out, config=link_config, tf=tf, stage_factory=stage_factory, render=render,
            b_input=args.b_input, list_consumer_factory=consumer_factory, **consumer_kind, run_outputs=run_outputs,
            view_bias=getattr(settings.detector_config, "axis_view_bias", None),
            sort_part_runtime=sort_rt, tracking=args.tracking,
            **({} if not world_on else {"world_tracker": _new_world_tracker(world_cfg[0]), "world_config": world_cfg[1],
                                        "world_plane_z_m": world_plane_z})))
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

    if center_on:
        overlay = getattr(center_pub, "overlay_topic", None)
        overlay_text = (f"overlay={overlay}" if overlay else
                        "overlay=none (eyetoarm_interfaces CenterOverlay not found: rebuild robot_ws; Image only)")
        err.write(f"run_pipeline: B CENTER OVERLAY ON topic={getattr(center_pub, 'topic', None)} {overlay_text} "
                  "(published right after B compute, before accept check (3); stage L only)\n")

    if sort_on:
        set_sink = getattr(sort_rt, "set_event_sink", None)

        if callable(set_sink):
            set_sink(link.sort_part.on_client_event)

        # #215: B 클라이언트의 release 대상 = 연결부의 모든 B 단계(추적을 켰을 때만, 끄면 stage=None 그대로)
        set_release = getattr(sort_rt, "set_release_target", None)

        if link.release_proxy is not None and callable(set_release):
            set_release(link.release_proxy)

        seen = getattr(sort_rt, "server_seen", None)
        retry = link_config.sort_part.retry
        retry_text = ("off" if retry is None else f"max_failures_per_track={retry.max_failures_per_track} "
                      f"point_block={retry.failed_point_block_mm}mm/{retry.failed_point_block_s}s"
                      + (" (stage L: point block only, no track ids)" if listed and not args.tracking else ""))
        # #244: 단계 L + 추적이면 송신 번호(기록·재시도·require_track_id)는 B가 후보 번호를 읽기 전까지 고른 후보의 A 번호
        track_text = ((('world' if world_on else 'M2(heuristic)') if args.tracking else 'off')
                      + (" (stage L: A candidate ids until B reads them)" if listed and args.tracking else ""))
        err.write(f"run_pipeline: LIVE A->B->C SORT_PART ON action=/sort_part server_seen={str(seen).lower()} "
                  f"retry={retry_text} tracking={track_text} "
                  f"require_track_id={({True: 'true', False: 'false'}).get(link_config.sort_part.require_track_id, 'unset')} "
                  "(D13 released by user decision 2026-11-20, accepted risk without pre-release test F6)\n")

        if link_config.sort_part.require_track_id:
            err.write("run_pipeline: require_track_id=true: PICK_PLACE without a confirmed part_track_id is not sent "
                      "(skip:no_track_id); bolts leaving view before confirmation are never sent "
                      "(see tracking tentative_dropped, sort_part.no_track_id_states)\n")

        if not seen:
            err.write("run_pipeline: warning: /sort_part server not seen at start; goals will be recorded as no_server "
                      "until it appears\n")

    if args.tracking and not world_on:
        err.write("run_pipeline: tracking ON M2(heuristic): confirmed part_track_id only -> B _dedupe/release "
                  "(A-6 experimental ids; split ids are a known limit #156)\n")

    if world_on:
        # 진단 전용(정답 격리): C speed_mps 조회값은 요약에만 둔다. 추적기에는 넘기지 않는다(보충 2)
        try:
            # #226 D2: 상한은 설정값(진단 전용). 시험 주입 조회기는 인자 없이 부른다
            probe = (live_conveyor_probe() if live_conveyor_probe is not None
                     else _ros_conveyor_speed_probe(world_cfg[1].conveyor_probe_timeout_s))
        except Exception as exc:
            probe = {"node": "/conveyor_controller", "speed_mps": None, "status": f"error: {type(exc).__name__}: {exc}"}

        link.world.diagnostics["conveyor_speed_mps"] = probe
        space_note = "default for live" if getattr(args, "tracking_space_source", "explicit") == "default" else "explicit"
        err.write(f"run_pipeline: tracking ON space=world({space_note}; W, belt plane z={world_plane_z:.3f} m) "
                  f"config_sha256={world_cfg[2]} "
                  f"lag_reset>{world_cfg[1].lag_limit_ms:.0f}ms; prediction uses W differencing only "
                  f"(conveyor speed_mps diagnostic={probe.get('speed_mps')} not used)\n")

    result = None
    exit_code = 1
    failure = None
    handoff = handoff_probe = None

    # r1 F3: 세션을 등록한 뒤의 모든 단계(소스 공장 구성·시작·기록 열기·실행)는 이 try 안이다. 실패해도 세션·TF를 정리하고 1로 끝낸다
    try:
        make = live_source_factory or _live_factory
        source_factory = make(args.source_id, args.ros_reliability, boundary_hook=link.boundary_hook)

        # #233: A 검출 → C 연결부(기본 --ros-link와 같이 항상). 실행기 제어 흐름 하나가 생산자다
        handoff, handoff_probe = (live_handoff_factory or _default_live_handoff)()

        # #239: W 추적이면 detect 영상 주석에 같은 작업의 W 번호를 그릴 때 읽게 한다(start_run 전, 기다리지 않는 조회)
        set_track_source = getattr(handoff, "set_track_source", None)

        if world_on and callable(set_track_source):
            set_track_source(link.world.peek)

        handoff_probe.start(handoff)
        link.start()
        stream = open(args.log, "x", encoding="utf-8")

        with stream, _StopSignals() as signals:
            log_kw = {"b_input": B_INPUT_LIST} if listed else {}
            log = PipelineLog(JsonlRecorder(stream), eval_run_id=args.eval_run_id,
                              log_geometry=args.log_instance_geometry, **log_kw)
            run_kw = {"live_observation_b_input": "list"} if listed else {}
            result = run(source_factory, settings, log, run_id=run_id, config=runner_config, stderr=err,
                         versions={"opencv": cv2.__version__},
                         should_stop=lambda: (signals.stop.is_set() or link.reset_pending
                                              or link.state.failed is not None),
                         handoff_sink=handoff, live_observation_sink=link, **run_kw, **detector_kw)

        exit_code = result.exit_code
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        err.write(f"run_pipeline: live observation run failed before/while running: {failure}\n")
        exit_code = 1
    finally:
        reset = link.reset_pending

        # 정상 종료(리셋 아님): 대기·처리 중인 마지막 항목을 상한 안에서 끝낸 뒤 닫는다(못 끝내면 닫을 때 무효화·기록)
        if not reset and result is not None:
            link.drained = link.dispatcher.drain(link_config.close_limit_s)

        record = controller.service() if reset else controller.stop(
            "startup_failed" if result is None else f"run_end:{result.stop_reason}")
        record = record or link.ended
        problems = []

        if failure is not None:
            problems.append(f"failure: {failure}")

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

        if link.sort_part is not None and link.sort_part.problem() is not None:   # #214: 송신 기록 실패는 숨기지 않는다
            problems.append(link.sort_part.problem())

        if link.world is not None and link.world.problem() is not None:   # #217: 추적 스레드 비정상 종료·기록 실패
            problems.append(link.world.problem())

        if getattr(link, "list_goal_unmatched", 0):   # #233: 고른 목표를 결과 묶음에서 못 찾음(B 반환 계약 위반)
            problems.append(f"list goal unmatched: {link.list_goal_unmatched}")

        # #233 정리 순서: 연결부 정리(dispatcher 정지) 뒤 center 발행기 → C 연결부. 정리 미확인이면 worker가 발행 중일 수 있어 열어 둔다
        if center_pub is not None:
            if link.cleanup_confirmed:
                try:
                    center_pub.close()
                except Exception as cexc:
                    problems.append(f"center overlay publisher close failed: {type(cexc).__name__}: {cexc}")
            else:
                problems.append("center publisher left open (dispatcher not confirmed stopped)")

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

        for p in problems:
            err.write(f"run_pipeline: live observation problem: {p}\n")

        try:
            tracking_extra = ({} if not args.tracking else {"tracking": None if result is None else result.tracking})

            if args.tracking:   # #239: 추적 공간과 그 출처(생략 = 실시간 기본 W)
                tracking_extra["tracking_space"] = {"space": "world" if world_on else "pixel",
                                                    "source": getattr(args, "tracking_space_source", "explicit")}

            # #233: C 연결부 종료 보고·center 발행 계수(발행 오류는 종료 코드에 넣지 않는다, 요약·마지막 줄만)
            tracking_extra["handoff"] = None if result is None else result.handoff

            if center_on:
                tracking_extra["center_overlay"] = center_stats.summary(center_pub)
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
    """저장 입력 단계 L(#189). 실제 AI 검출·끝점 추출 결과를 `b_list_gate` 뒤 목록(`eyetoarm.b_input/2`)으로 목록 소비자에 넘긴다.

    호출 경로: 저장 소스 → `CaptureMetaSource` → 실행기 → `AiSegDetector`(M1) → 코어(단계 T 관측 그대로) → `handle()`: 기록 →
    `display_sink`(= 기록기 `on_outcome`) → `b_list_gate` → `b_list_consumer`(= 기록기가 감싼 목록 소비자). `b_consumer`는 주지 않는다
    (이중 전달 없음). 목록 소비자: 시험 주입 `list_consumer_factory(stage)` 또는 `stage.process_frame`. 둘 다 없으면 로그·출력 파일을
    만들기 전에 종료 1(단계 T로 바꾸지 않음). `on_output`은 None(C·D·로봇 송신 없음). 주석 영상은 만들지 않는다.
    """
    from eyetoarm.transforms.b_stage import BStage
    from eyetoarm.transforms.observation_envelope_adapter import EnvelopeTfLookup
    from eyetoarm.vision.ai_detector import AiSegDetector
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

    inject = {"model_factory": args._model_factory}

    if args._import_module is not None:
        inject["import_module"] = args._import_module

    det = AiSegDetector(settings.detector_config, **inject)
    kind, paths = ("images", list(args.images)) if args.images else ("video", [args.video])
    inner_factory = stored_source_factory(kind, paths, args.source_id)
    holder = {}

    def source_factory(issuer, generation_provider):
        """저장 입력 소스를 촬영 메타 소스로 감싼다."""
        holder["source"] = CaptureMetaSource(inner_factory(issuer, generation_provider), meta)
        return holder["source"]

    try:
        recorder = CaptureListRecorder(args.observation_out, meta, consumer_kind=consumer_kind,
                                       view_bias=getattr(settings.detector_config, "axis_view_bias", None))
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
                              log_geometry=args.log_instance_geometry, b_input=B_INPUT_LIST)
            result = run(source_factory, settings, log, detector=det, prepare=det.prepare,
                         b_list_consumer=recorder.wrap(consume), display_sink=recorder, config=runner_config,
                         stderr=err, versions={"opencv": cv2.__version__})

        problems = []

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
                          "on_output": None, "problems": problems, "runner_errors": result.errors})
    finally:
        recorder.close()

    for p in problems:
        err.write(f"run_pipeline: stage L mismatch: {p}\n")

    err.write(f"run_pipeline: exit={exit_code} reason={result.stop_reason} a_b_stage_l frames={result.frames} "
              f"b_list_calls={result.b_list_calls} candidates_sent={recorder.candidates_sent} "
              f"input_errors={result.input_errors} timeouts={result.timeouts}\n")

    return exit_code


def _run_capture_observation(args, settings, runner_config, detector_kw, capture, err, b_stage_hook,
                   extension_source=None) -> int:
    """저장 입력 실제 A→B 경로(#168). 실제 AI 검출기·끝점 추출기 관측을 `b_gate` 뒤 BStage로 보내고 주석 영상을 쓴다.

    호출 경로: 저장 소스 → `CaptureMetaSource`(일치 프레임에만 촬영 메타) → 실행기 → `AiSegDetector`(convert 안 M1) →
    `observation_from_ai` → 코어 `complete()` → `handle()`: 기록 → `display_sink`(= 기록기 `on_outcome`) → `b_admit`(`b_gate`) →
    `b_consumer`(= 기록기가 감싼 `stage.consume` + `render_center`). BStage·기록기는 이 함수가 만들고 닫는다. `on_output`은 None.
    """
    from eyetoarm.transforms.b_stage import BStage
    from eyetoarm.vision.ai_detector import AiSegDetector
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

    inject = {"model_factory": args._model_factory}

    if args._import_module is not None:
        inject["import_module"] = args._import_module

    det = AiSegDetector(settings.detector_config, **inject)
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
                                   view_bias=getattr(settings.detector_config, "axis_view_bias", None))
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
                              log_geometry=args.log_instance_geometry)
            result = run(source_factory, settings, log, detector=det, prepare=det.prepare,
                         b_consumer=recorder.wrap(stage.consume), display_sink=recorder, config=runner_config,
                         stderr=err, versions={"opencv": cv2.__version__})

        problems = []

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
                                        "used_in_compute": []}})
    finally:
        recorder.close()

    for p in problems:
        err.write(f"run_pipeline: real-path mismatch: {p}\n")

    err.write(f"run_pipeline: exit={exit_code} reason={result.stop_reason} a_b_real_path frames={result.frames} "
              f"b_calls={result.b_calls} annotated={recorder.annotated} input_errors={result.input_errors} "
              f"timeouts={result.timeouts}\n")

    return exit_code


def _run_synthetic_observation(args, settings, runner_config, detector_kw, synthetic, err, b_stage_hook) -> int:
    """시험 B 연결 실행(#166, synthetic/test-only). 제품 검출기 대신 fixture 관측을 detector 자리에 넣는다.

    관측 주입 경계: `run(detector=FixtureDetector)` — 실행기가 시작한 작업마다 detector로 불리고, 반환 관측은 코어
    `complete()`(식별자·끝점 범위) → `handle()`의 `b_gate` → `b_consumer`(= 기록 감싼 `stage.consume`)를 지난다.
    BStage·기록 파일은 이 함수가 만들고 닫는다. `stage.on_output`은 None으로 둔다(C·로봇 송신 없음).
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
