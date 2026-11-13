"""저장 이미지 입력 재현 CLI — A 내부 실행 기록용.

    PYTHONPATH=src python3 scripts/run_vision.py --images <경로...> [--source-id stored0] [--max-frames N]
    PYTHONPATH=src python3 scripts/run_vision.py --image-list <목록.txt> [--source-id stored0] [--max-frames N]
    ... [--detect-config <설정.json> [--overlay-dir <새 폴더>]]

기본은 입력 재현과 기록만 한다. `--detect-config`를 주면 성공 프레임마다 `vision.planar.detect_planar`로
표시용(raw) 윤곽을 추출해 기록한다. GUI·ROS·TCP·로봇 명령·클래스·규격 판정은 없다.

stdout에는 한 줄에 JSON 객체 하나를 쓴다(kind = start / frame / observation / event / end). 이 필드
이름은 A 내부 실행 기록용 제안이며 공동 API나 TCP 규약이 아니다. 경고와 예상 밖 오류 설명은 stderr로 간다.

검출 모드(`--detect-config`)
- 설정은 시작 전에 한 번 읽어 검증한다. 검출 옵션이 없으면 설정을 읽지 않고 출력은 이전과 같다.
- `frame` 바로 다음에 같은 프레임의 `observation`을 한 줄 쓴다. 입력 실패 `event`에는 만들지 않는다.
- `observation`은 표시용이다: `display_only=true`, `valid=false`, `observed_face`·`calc_contour`·
  `selected_index`는 null. 후보를 자동 선정하지 않고 면적 탈락·경계 접촉 후보도 모두 남긴다.
  좌표·면적·경계 상자는 원본 raw 픽셀이다. `eligible`은 영상 필터 통과일 뿐 B 유효 판정이 아니다.
- `start`에 `config_sha256`·`display_only`, `end`에 `observations`(observation 수)를 덧붙인다.
  `frames`·`failures`·`reason`의 뜻은 같다. 후보 0개나 `valid=false`는 실패로 세지 않는다.
- `--overlay-dir`은 `--detect-config`와 함께만 쓴다. 아직 없는 폴더여야 하며(부모는 있어야 함) 실행이
  새로 만든다. 파일 이름은 `overlay_<image_seq 6자리>.png`이고 원본 경로·소스 ID를 쓰지 않는다.
  원본 복사본에 후보를 그린다(통과=초록, 탈락·잘림=빨강, ROI=파랑, 위쪽에 범례·설정 해시·표시용 표기).

- `timestamp_base`는 PC_RECEIVE이며, 파일 디코딩을 마친 PC 시각이지 촬영 시각이 아니다.
  `timestamp_ms`를 물리 관측 시간·촬영 지연·로봇 목표 유효성의 근거로 쓰지 않는다.
- 목록 파일은 UTF-8로 읽고, 줄 앞뒤 공백을 지우고, 빈 줄만 뺀다. 순서는 그대로다.
  주석·glob·폴더 탐색은 없다. 상대 이미지 경로는 목록 파일 위치가 아니라 실행 cwd 기준이다.
- `--max-frames`는 성공 프레임 수만 센다. 실패 항목은 세지 않는다.

종료 코드
- 0: 목록을 끝까지 처리했거나 `--max-frames`에 도달함. **모든 파일이 성공했다는 뜻이 아니다.**
  개별 파일 실패와 빈 목록은 event/end 기록으로 구분한다.
- 2: 옵션 오류, 양의 정수가 아닌 제한, 목록 파일 열기·UTF-8 읽기 실패, 검출 설정 열기·구문·계약
  오류, 오버레이 폴더 조건 오류(이미 있음·부모 없음·만들 수 없음). 모두 start 출력과 소스 생성 전이다.
- 1: 예상하지 못한 실행 오류, 검출 예외(예: 원본 밖 ROI), 오버레이 저장 실패(stderr에 설명, end 없음).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import cv2
import numpy as np

from eyetoarm.inputs.base import FrameIdIssuer, FramePacket, InputStatus, new_run_id
from eyetoarm.inputs.stored_images import StoredImageSource
from eyetoarm.vision.planar import PlanarConfig, PlanarResult, detect_planar, load_config

TIME_NOTE = "PC_RECEIVE는 파일 디코딩 완료 시각이며 촬영 시각이 아님"
# StoredImageSource가 쓰는 detail 문자열(stored_images.py)
_DETAIL_EMPTY = "empty list"
_DETAIL_CLOSED = "closed"
# 오버레이 색(BGR)
_ELIGIBLE_BGR = (0, 200, 0)
_REJECT_BGR = (0, 0, 255)
_ROI_BGR = (255, 0, 0)
_LEGEND_BGR = (255, 0, 255)


def _positive_int(text: str) -> int:
    """`--max-frames` 값 변환기. 10진 양의 정수만 받는다.

    "0"·"-1"·"1.5"·"x"는 `argparse.ArgumentTypeError`를 내고, argparse가 사용법과 함께 exit 2로 끝낸다.
    """
    try:
        value = int(text, 10)
    except ValueError:
        raise argparse.ArgumentTypeError(f"양의 정수가 아님: {text!r}")
    if value <= 0:
        raise argparse.ArgumentTypeError(f"양의 정수가 아님: {text!r}")
    return value


def _non_empty(text: str) -> str:
    """`--source-id` 값 변환기. 빈 문자열이면 `argparse.ArgumentTypeError`(exit 2)."""
    if text == "":
        raise argparse.ArgumentTypeError("빈 문자열은 허용하지 않음")
    return text


def _build_parser() -> argparse.ArgumentParser:
    """CLI 옵션 해석기를 만든다.

    - `--images`와 `--image-list` 중 정확히 하나가 필요하다.
    - `--detect-config`·`--overlay-dir`은 선택이다. `--overlay-dir` 단독 사용 거부는 `main`이 검사한다.
    """
    parser = argparse.ArgumentParser(
        prog="run_vision",
        description="저장 이미지 입력을 읽어 프레임·입력 사건을 JSON Lines로 출력합니다(A 내부 기록용).",
        epilog="종료 코드 0은 목록 처리 완료 또는 제한 도달이며, 모든 파일 성공을 뜻하지 않습니다. "
               "2는 옵션·목록 파일 오류, 1은 예상하지 못한 실행 오류입니다.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--images", nargs="+", metavar="PATH", help="읽을 이미지 경로(인자 순서대로)")
    group.add_argument("--image-list", metavar="FILE",
                       help="한 줄에 경로 하나인 UTF-8 목록 파일(상대 경로는 실행 cwd 기준)")
    parser.add_argument("--source-id", type=_non_empty, default="stored0", help="소스 ID(기본 stored0)")
    parser.add_argument("--max-frames", type=_positive_int, default=None,
                        help="성공 프레임 수 제한(양의 정수). 실패 항목은 세지 않음")
    parser.add_argument("--detect-config", metavar="JSON", default=None,
                        help="표시용 윤곽 추출 설정(config/vision/opencv_planar.json 형식). 주면 observation 기록")
    parser.add_argument("--overlay-dir", metavar="DIR", default=None,
                        help="오버레이 PNG를 저장할 새 폴더(--detect-config와 함께만, 이미 있으면 거부)")
    return parser


def _read_list(path: str) -> list[str]:
    """UTF-8 목록 파일을 읽어 경로 목록을 돌려준다.

    줄 앞뒤 공백을 지우고 빈 줄만 뺀다. 순서는 파일 그대로다. 주석·glob·폴더 탐색은 하지 않는다.
    열기·디코딩 오류(OSError, UnicodeDecodeError)는 호출자(`main`)가 exit 2로 바꾼다.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    return [line.strip() for line in lines if line.strip()]


def _frame_record(frame: FramePacket) -> dict:
    """성공 프레임 하나를 `kind=frame` 기록(dict)으로 만든다. 영상 배열은 넣지 않는다.

    키 구성은 검출 모드에서도 바뀌지 않는다.
    """
    return {
        "kind": "frame",
        "run_id": frame.run_id,
        "source_id": frame.source_id,
        "image_seq": frame.image_seq,
        "source_index": frame.source_index,
        "source_ref": frame.source_ref,
        "width": frame.width,
        "height": frame.height,
        "timestamp_base": frame.timestamp_base.value,
        "timestamp_ms": frame.timestamp_ms,
        "received_monotonic_ns": frame.received_monotonic_ns,
    }


def _event_record(event) -> dict:
    """입력 사건(`InputStatusEvent` A 확장)을 `kind=event` 기록으로 만든다.

    프레임이 없는 사건이므로 `image_seq`·`timestamp_ms`를 넣지 않는다. 상태는 enum 값 문자열이다.
    """
    return {
        "kind": "event",
        "run_id": event.run_id,
        "source_id": event.source_id,
        "status": event.status.value,
        "source_ref": event.source_ref,
        "detail": event.detail,
        "event_monotonic_ns": event.event_monotonic_ns,
    }


def _observation_record(result: PlanarResult, overlay_path: str | None) -> dict:
    """`detect_planar` 결과를 `kind=observation` 기록으로 만든다(검출 모드 전용, A 내부 표시용).

    - 입력 식별자·원본 시각은 결과에 보존된 값을 그대로 옮긴다(같은 프레임의 `frame` 기록과 일치).
    - 튜플·NumPy 값은 JSON 기본 타입(list·float·int)으로 바꾼다. 점은 원본 raw float 좌표 전체다.
    - 면적 탈락·경계 접촉 후보도 모두 넣고, 후보를 고르지 않는다(`selected_index`는 null).
    - `display_only`·`valid`·`observed_face`·`calc_contour`는 결과의 고정값을 그대로 쓴다.
    """
    return {
        "kind": "observation",
        "run_id": result.run_id,
        "source_id": result.source_id,
        "image_seq": result.image_seq,
        "capture_generation": result.capture_generation,
        "timestamp_ms": result.timestamp_ms,
        "timestamp_base": result.timestamp_base.value,
        "received_monotonic_ns": result.received_monotonic_ns,
        "config_sha256": result.config_sha256,
        "processed_wh": [int(v) for v in result.processed_wh],
        "transform": [float(v) for v in result.transform],
        "threshold_used": float(result.threshold_used),
        "raw_contour_count": result.raw_contour_count,
        "eligible_count": result.eligible_count,
        "process_ms": float(result.process_ms),
        "display_only": result.display_only,
        "valid": result.valid,
        "observed_face": result.observed_face,
        "calc_contour": result.calc_contour,
        "selected_index": result.selected_index,
        "candidates": [
            {
                "index": c.index,
                "points_source": c.points_source.astype(np.float64).tolist(),
                "area_source_px2": float(c.area_source_px2),
                "bbox_source": [float(v) for v in c.bbox_source],
                "border_sides": list(c.border_sides),
                "clip_reasons": list(c.clip_reasons),
                "area_reject_reason": c.area_reject_reason,
                "eligible": c.eligible,
            }
            for c in result.candidates
        ],
        "overlay_path": overlay_path,
    }


def _draw_overlay(image: np.ndarray, result: PlanarResult, config: PlanarConfig) -> np.ndarray:
    """원본 복사본에 범례·ROI·후보를 그려 돌려준다. 원본 배열과 결과 좌표는 바꾸지 않는다.

    그리는 순서는 범례(위쪽 3줄) → ROI 사각형(파랑) → 후보(eligible 초록, 탈락·잘림 빨강)이다.
    점은 그릴 때만 `np.rint`로 정수화한다.
    """
    canvas = image.copy()
    legend = ("display_only valid=false", f"cfg {result.config_sha256[:12]}",
              "green=eligible red=rejected/clipped")
    for i, text in enumerate(legend):
        cv2.putText(canvas, text, (2, 12 + 12 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.35, _LEGEND_BGR, 1)
    if config.roi is not None:
        x, y, w, h = config.roi
        cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), _ROI_BGR, 1)
    for c in result.candidates:
        pts = np.rint(c.points_source).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, _ELIGIBLE_BGR if c.eligible else _REJECT_BGR, 1)
    return canvas


def _save_overlay(directory: str, image_seq: int, canvas: np.ndarray) -> str:
    """`overlay_<image_seq 6자리>.png`를 새로 만들고 경로를 돌려준다.

    파일 이름에 원본 경로·소스 ID를 쓰지 않는다. `open(..., "xb")`라 기존 파일을 덮어쓰지 않는다.
    인코딩·쓰기 실패는 RuntimeError(`overlay ... failed`)로 올려 `main`이 exit 1로 끝내게 한다.
    """
    path = os.path.join(directory, f"overlay_{int(image_seq):06d}.png")
    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError(f"overlay encode failed: {path}")
    try:
        with open(path, "xb") as f:
            f.write(buf.tobytes())
    except OSError as exc:
        raise RuntimeError(f"overlay write failed: {path}: {exc}") from exc
    return path


def main(argv=None, *, source_factory=StoredImageSource, stdout=None, stderr=None) -> int:
    """CLI 진입점. 종료 코드를 돌려준다(0 정상 종료, 2 인자·경로·설정 오류, 1 실행 중 오류).

    순서: 옵션 해석 → 목록 읽기 → (검출 모드) 설정 로드·오버레이 폴더 생성 → 소스 생성 → start →
    프레임마다 frame(+observation) / 사건마다 event → end. 2로 끝나는 오류는 모두 start 출력 전이다.

    - `source_factory`·`stdout`·`stderr`는 시험에서 가짜 소스·출력을 넣기 위한 주입 지점이다.
    - 실행 중 예외는 성공으로 처리하지 않는다: stderr에 원인을 쓰고 end 없이 1을 돌려준다.
    - 정상·예외 모두 `finally`에서 소스를 닫는다.
    """
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    parser = _build_parser()
    args = parser.parse_args(argv)  # 옵션 오류는 argparse가 exit 2로 끝낸다
    if args.overlay_dir is not None and args.detect_config is None:
        parser.error("--overlay-dir는 --detect-config와 함께만 쓸 수 있음")  # exit 2

    if args.images is not None:
        paths = list(args.images)
    else:
        try:
            paths = _read_list(args.image_list)
        except (OSError, UnicodeDecodeError) as exc:
            parser.error(f"목록 파일을 읽을 수 없음: {args.image_list}: {exc}")  # exit 2

    config = None
    if args.detect_config is not None:
        try:
            config = load_config(args.detect_config)
        except (OSError, ValueError) as exc:
            parser.error(f"검출 설정을 읽을 수 없음: {args.detect_config}: {exc}")  # exit 2
        if args.overlay_dir is not None:
            try:
                os.mkdir(args.overlay_dir)  # 이미 있으면(파일·폴더) FileExistsError
            except OSError as exc:
                parser.error(f"오버레이 폴더를 새로 만들 수 없음: {args.overlay_dir}: {exc}")  # exit 2

    def emit(record: dict) -> None:
        """기록 하나를 표준 JSON 한 줄로 쓰고 바로 flush한다. NaN·Infinity는 ValueError로 거부한다."""
        stdout.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stdout.flush()

    run_id = new_run_id()
    issuer = FrameIdIssuer(run_id)
    source = source_factory(paths, source_id=args.source_id, issuer=issuer)
    try:
        start = {
            "kind": "start",
            "run_id": run_id,
            "source_id": args.source_id,
            "cwd": os.getcwd(),
            "paths": paths,
            "timestamp_base": "PC_RECEIVE",
            "time_note": TIME_NOTE,
        }
        if config is not None:
            start["config_sha256"] = config.config_sha256
            start["display_only"] = True
        emit(start)
        frames = 0
        failures = 0
        observations = 0
        while True:
            item = source.read()
            if isinstance(item, FramePacket):
                emit(_frame_record(item))
                frames += 1
                if config is not None:
                    result = detect_planar(item, config)
                    overlay_path = None
                    if args.overlay_dir is not None:
                        overlay_path = _save_overlay(args.overlay_dir, item.image_seq,
                                                     _draw_overlay(item.image, result, config))
                    emit(_observation_record(result, overlay_path))
                    observations += 1
                if args.max_frames is not None and frames >= args.max_frames:
                    reason = "max_frames"
                    break
                continue
            emit(_event_record(item))
            if item.status is InputStatus.END_OF_STREAM:
                reason = "end_of_stream"
                break
            if item.status is InputStatus.SOURCE_UNAVAILABLE and item.detail == _DETAIL_EMPTY:
                reason = "empty_list"
                break
            if item.status is InputStatus.SOURCE_UNAVAILABLE and item.detail == _DETAIL_CLOSED:
                reason = "closed"
                break
            failures += 1
        end = {
            "kind": "end",
            "run_id": run_id,
            "source_id": args.source_id,
            "frames": frames,
            "failures": failures,
            "reason": reason,
        }
        if config is not None:
            end["observations"] = observations
        emit(end)
        return 0
    except Exception as exc:  # 예상하지 못한 오류는 성공으로 처리하지 않는다
        stderr.write(f"run_vision: unexpected error: {exc!r}\n")
        traceback.print_exc(file=stderr)
        return 1
    finally:
        source.close()


if __name__ == "__main__":
    sys.exit(main())
