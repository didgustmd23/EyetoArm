"""저장 이미지 입력을 재현해 stdout에 한 줄 JSON 기록(start/frame/observation/event/end)을 쓰는 CLI.

사용: PYTHONPATH=src python3 scripts/run_vision.py --images <경로...> [--detect-config <설정.json>]
종료 코드: 0 목록 끝·제한 도달, 1 실행 중 오류, 2 인자·경로·설정 오류(start 전).
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
# StoredImageSource가 쓰는 detail 문자열
_DETAIL_EMPTY = "empty list"
_DETAIL_CLOSED = "closed"
# 오버레이 색(BGR)
_ELIGIBLE_BGR = (0, 200, 0)
_REJECT_BGR = (0, 0, 255)
_ROI_BGR = (255, 0, 0)
_LEGEND_BGR = (255, 0, 255)


def _positive_int(text: str) -> int:
    """`--max-frames` 값 변환기. 10진 양의 정수만 받고, 아니면 `argparse.ArgumentTypeError`."""
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
    """CLI 옵션 해석기를 만든다(`--images`·`--image-list` 중 정확히 하나)."""
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
    """UTF-8 목록 파일을 읽어 경로 목록을 돌려준다(줄 앞뒤 공백 제거, 빈 줄만 뺌, 순서 유지).

    열기·디코딩 오류는 호출자(`main`)가 exit 2로 바꾼다.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    return [line.strip() for line in lines if line.strip()]


def _frame_record(frame: FramePacket) -> dict:
    """성공 프레임 하나를 `kind=frame` 기록(dict)으로 만든다. 영상 배열은 넣지 않는다."""
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
    """입력 사건을 `kind=event` 기록으로 만든다(`image_seq`·`timestamp_ms` 없음)."""
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
    """`detect_planar` 결과를 `kind=observation` 기록으로 만든다(표시용, 후보를 고르지 않음)."""
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
    """원본 복사본에 범례·ROI·후보를 그려 돌려준다. 원본 배열과 결과 좌표는 바꾸지 않는다."""
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
    """`overlay_<image_seq 6자리>.png`를 새로 만들고(덮어쓰지 않음) 경로를 돌려준다.

    인코딩·쓰기 실패는 RuntimeError로 올려 `main`이 exit 1로 끝내게 한다.
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

    `source_factory`·`stdout`·`stderr`는 시험 주입 지점이다. 소스는 `finally`에서 닫는다.
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
