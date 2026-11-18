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
  있으면 덮어쓰지 않고 2. `--tracking`·`--capture-meta`·`--synthetic-observation-*`과 함께 쓸 수 없다(2).
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
                             "사유 출력 폴더(새 폴더만). --live-observation-config 필수. C·D·로봇 송신 없음")
    parser.add_argument("--live-observation-config", dest="live_observation_config", metavar="JSON", default=None,
                        help="--live-observation-out 전용: 시험 설정(TF 상한·용량·종료 상한 등, status=test_only 필수)")
    parser.add_argument("--tracking", action="store_true", help="--detector ai 전용. A-6 1차 실험: AI 검출 인스턴스 추적 번호를 붙인다(M2 휴리스틱, 표시·기록 전용, 팔 이동을 넘는 신원·중복 집기 방지 보장 없음, B로 보내지 않음). 기본 꺼짐")

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

    # 실시간 A→B 기록 전용 시험(#173): 명시 옵션에서만 전용 디스패처. 기본 --ros-link는 D13 그대로
    if args.live_observation_out is not None or args.live_observation_config is not None:
        if args.live_observation_out is None or args.live_observation_config is None:
            build_parser().error("--live-observation-out and --live-observation-config must be given together")
        if not args.ros_link:
            build_parser().error("--live-observation-out requires --ros-link")
        if args.detector != "ai":
            build_parser().error("--live-observation-out requires --detector ai")
        if args.tracking:
            build_parser().error("--live-observation-out cannot be combined with --tracking")

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
         live_run_controller=None) -> int:
    """CLI 진입점. 종료 코드를 돌려준다.

    `b_stage_hook(stage)`는 시험 주입 지점이다(시험 B 연결에서 만든 BStage를 run 전에 받음, 기본 None).
    `extension_source(frame, obs)`는 실제 A→B 경로 봉투에 실을 확장 사전을 주는 시험 주입 지점이다(기본 None = 확장 없음).

    `ros_available`(rclpy 확인 대체)·`live_source_factory`(`(source_id, reliability) → 소스 팩토리`)는 시험 주입 지점이다.
    실시간 기록 전용 시험에서는 `live_source_factory(source_id, reliability, boundary_hook=)`로 부르고, `live_tf_factory(config)`(시작된 TF
    수신 묶음)·`live_run_controller`(프로세스 단위 `LiveRunController`, 기본은 모듈 전역)도 시험 주입 지점이다.

    `model_factory`·`import_module`은 시험 주입 지점이다(AI 어댑터에 그대로 넘김, 기본은 실제 ultralytics 지연 import).
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
        settings = PipelineSettings(source_id=args.source_id, mode=args.mode, detector_config=config, profile=profile)
    except (OSError, ValueError) as exc:
        err.write(f"run_pipeline: config error: {exc}\n")
        return 2

    # 추적 opt-in(#151): 진입점이 프로세스당 발급기 1개를 만들어 주입한다
    if args.tracking:
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
                                     controller=live_run_controller or _live_runs())

    if capture is not None:
        return _run_capture_observation(args, settings, runner_config, detector_kw, capture, err, b_stage_hook,
                              extension_source)

    if args.ros_link:
        make = live_source_factory or _live_factory
        source_factory = make(args.source_id, args.ros_reliability)
    else:
        kind, paths = ("images", list(args.images)) if args.images else ("video", [args.video])
        source_factory = stored_source_factory(kind, paths, args.source_id)

    try:
        stream = open(args.log, "w", encoding="utf-8")
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
        self._shutdown = False

    def close(self, limit_s: float) -> None:
        """런타임 정리(실패하면 RuntimeError, 재시도 가능) → Context 종료."""
        self._runtime.close(limit_s)

        if not self._shutdown:
            self._context.try_shutdown()
            self._shutdown = True


def _run_live_observation(args, settings, runner_config, detector_kw, err, *, b_stage_hook, live_source_factory,
                          live_tf_factory, controller) -> int:
    """실시간 A→B 기록 전용 시험(#173). 호출 경로: LiveLinkSource(경계 훅 = 연결부) → 실행기 → AiSegDetector(M1) → 코어 →
    `handle()` 기록·표시 뒤 `live_observation_sink.offer` → 전용 디스패처 → 촬영 stamp TF → 봉투 → B 입구 어댑터 → BStage → 주석 → 순번
    기록기. 리셋이면 실행을 멈추고 정리한다(자동 재시작 없음)."""
    import cv2 as _cv2

    from eyetoarm.inputs.base import new_run_id
    from eyetoarm.transforms.b_stage import BStage
    from eyetoarm.transforms.observation_envelope_adapter import EnvelopeTfLookup, ObservationEnvelopeAdapter
    from eyetoarm.vision.live_observation_link import (LiveObservationConfigError, LiveObservationLink,
                                                       load_live_observation_config)

    try:
        link_config = load_live_observation_config(args.live_observation_config)
    except (OSError, LiveObservationConfigError) as exc:
        err.write(f"run_pipeline: live observation config error: {exc}\n")
        return 2

    if settings.detector_config.axis_extractor is None:
        err.write("run_pipeline: config error: --live-observation-out requires an AI config with axis_extractor\n")
        return 2

    for path in (args.log, args.live_observation_out):
        if Path(path).exists():
            err.write(f"run_pipeline: output exists, refusing to overwrite: {path}\n")
            return 2

    run_id = new_run_id()
    refusal = controller.refusal(run_id)

    if refusal is not None:
        err.write(f"run_pipeline: live observation run refused (explicit new run only after cleanup): {refusal}\n")
        return 1

    controller.close_limit_s = link_config.close_limit_s

    try:
        tf = (live_tf_factory or _RosTfRuntime)(link_config)
    except Exception as exc:
        err.write(f"run_pipeline: live TF start failed: {type(exc).__name__}: {exc}\n")
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

    try:
        link = controller.start_run(run_id, make_session=lambda rid: LiveObservationLink(
            rid, args.live_observation_out, config=link_config, tf=tf, stage_factory=stage_factory, render=render))
    except Exception as exc:
        # 세션이 만들어지지 않았다: 이 함수가 만든 TF만 정리한다(세션 등록 없음)
        err.write(f"run_pipeline: cannot start live observation run: {type(exc).__name__}: {exc}\n")

        try:
            tf.close(link_config.close_limit_s)
        except Exception as cexc:
            err.write(f"run_pipeline: live TF close failed: {type(cexc).__name__}: {cexc}\n")

        return 2 if isinstance(exc, OSError) else 1

    err.write(f"run_pipeline: LIVE A->B RECORD-ONLY TEST run_id={run_id} config_sha256={link_config.sha256} "
              f"status=test_only; no C/D/robot output, no auto restart\n")
    result = None
    exit_code = 1
    failure = None

    # r1 F3: 세션을 등록한 뒤의 모든 단계(소스 공장 구성·시작·기록 열기·실행)는 이 try 안이다. 실패해도 세션·TF를 정리하고 1로 끝낸다
    try:
        make = live_source_factory or _live_factory
        source_factory = make(args.source_id, args.ros_reliability, boundary_hook=link.boundary_hook)
        link.start()
        stream = open(args.log, "x", encoding="utf-8")

        with stream, _StopSignals() as signals:
            log = PipelineLog(JsonlRecorder(stream), eval_run_id=args.eval_run_id,
                              log_geometry=args.log_instance_geometry)
            result = run(source_factory, settings, log, run_id=run_id, config=runner_config, stderr=err,
                         versions={"opencv": cv2.__version__},
                         should_stop=lambda: (signals.stop.is_set() or link.reset_pending
                                              or link.state.failed is not None),
                         live_observation_sink=link, **detector_kw)

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

        if link.dispatcher.errors and any(not e.startswith("envelope:") for e in link.dispatcher.errors):
            problems.append(f"worker errors: {[e for e in link.dispatcher.errors if not e.startswith('envelope:')]}")

        if problems:
            exit_code = 1

        for p in problems:
            err.write(f"run_pipeline: live observation problem: {p}\n")

        try:
            link.write_summary({"exit_code": exit_code, "reset_detected": reset, "problems": problems,
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
                                   extension_source=extension_source)
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
                                 "max_wait_ms": max(tf_lookup.wait_ms) if tf_lookup.wait_ms else None},
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
