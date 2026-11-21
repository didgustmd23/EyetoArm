"""실시간 검출 확인 창의 순수 부분(#138) — 표시 전달 칸(`DisplayBridge`)과 그리기(`render_view`).

근거: reviewer 지시 138(A 개발 확인 도구, D 운영 GUI 아님), A 상세설계 §5.2(표시용 결과와 계산용 구분)·§5.4(세대·무검출 재사용
금지), 계획 `docs/superpowers/plans/2026-11-16-live-link-camera.md` r3 §3.5(결과 공개 전 입력 반영).

- 창·ROS·GPU를 import하지 않는다(OpenCV 그리기 함수와 numpy만). 창 루프는 `scripts/view_live_detection.py`가 맡는다.
- **제어 흐름 쪽**(runner 스레드)은 `on_outcome`·`on_input`·`finish`만 부른다. 잠금 안에서 참조를 바꿀 뿐 창을 기다리지 않는다.
- **창 쪽**(메인 스레드)은 `snapshot`·`is_current`·`request_stop`을 부른다. 그리기는 꺼낸 참조로 잠금 밖에서 한다.
- 칸은 1개다. 새 결과가 이전 결과를 덮어써 참조를 놓는다. 입력 경계·실패는 결과를 즉시 지운다. 모든 변경에서 `version`이 1 오르고,
  창은 그리기 직전 `is_current(version)`로 꺼낸 뒤 바뀌었는지 확인한다(바뀌었으면 버리고 다시 꺼냄).
- AI 관측은 표시용(valid=False)이다. 화면은 「표시용 검출」만 말하며 작업 가능·좌표 유효를 뜻하는 표시(초록)를 쓰지 않는다.
- 화면 글자는 영문이다(OpenCV Hershey 글꼴은 한글을 그리지 못함). 색·투명도는 **A 개발 기본값**이며 사용자 확정값이 아니다.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from eyetoarm.vision.ai_result import AiSegResult
from eyetoarm.vision.pipeline_types import JobOutcome, JobStatus
from eyetoarm.vision.tracking import TrackAssignment, TrackFrame, TrackState, matches_job

# 표시 기본값(A 개발 기본값). BGR. 초록 계열은 쓰지 않는다(작업 가능 신호로 오해 방지)
MASK_ALPHA = 0.35
CLASS_COLORS_BGR = ((255, 160, 0), (255, 0, 200), (0, 200, 255), (200, 120, 255))  # 파랑·자홍·주황·분홍 순환
REJECTED_COLOR_BGR = (150, 150, 150)
STATE_COLORS_BGR = {"waiting": (128, 128, 128), "live": (220, 140, 0), "stale": (0, 140, 255),
                    "error": (0, 0, 230), "stopping": (128, 128, 128), "ended": (128, 128, 128)}
RECENT_EVENTS = 4
_INFO_ONLY_INPUTS = ("rewind_filter_released",)


class ViewState(str, Enum):
    """창 상태. `LIVE`만 현재 검출 결과를 그린다."""

    WAITING = "waiting"      # 아직 표시할 결과 없음
    LIVE = "live"            # 최초 현재 적용 결과(표시용 검출)
    STALE = "stale"          # 입력 경계·결과 폐기: 결과 지움, 마지막 원본만 표시 가능
    ERROR = "error"          # 검출 실패·시간 초과·실행 실패: 결과 지움
    STOPPING = "stopping"    # 정지 요청 뒤(새 검출을 그리지 않음)
    ENDED = "ended"          # runner 종료


@dataclass(frozen=True)
class ViewItem:
    """칸에 든 표시 한 건. `frame`은 코어가 작업에 준 프레임 참조이며 그리지 않고 읽기만 한다."""

    frame: object               # FramePacket
    result: AiSegResult | None  # None이면 AI 표시 결과 형식이 아님(그리지 않고 알림)
    job_seq: int
    generation: int
    conf_threshold: float | None
    reason: str | None          # 관측 reason(표시용 invalid 사유)
    tracks: TrackFrame | None = None  # 같은 작업의 추적 결과(#151 S3, 추적을 켰을 때만). 표시 실험 번호
    valid: bool | None = None         # 관측 valid 그대로(#196 A1-5). 축 끝점 경로를 켜면 True일 수 있다. 로봇 목표 승인이 아니다


def status_text(valid) -> str:
    """확인 창 상태 문구(#196 A1-5): 관측의 실제 valid 값을 쓰고, 로봇 목표가 아니라는 문구는 그대로 둔다."""
    return f"display-only detection (valid={valid}, not for robot)"


@dataclass(frozen=True)
class ViewSnapshot:
    """창이 꺼낸 상태. `item`이 None이 아니면 같은 프레임의 원본·결과 쌍이다. `last_frame`은 결과 없는 마지막 원본."""

    version: int
    state: ViewState
    item: ViewItem | None
    last_frame: object | None
    message: str
    events: tuple[str, ...]


class DisplayBridge:
    """runner → 창 최신 1칸 전달. 잠금만 쓰고 대기하지 않는다(제어 흐름을 막지 않음)."""

    def __init__(self) -> None:
        """빈 칸으로 시작한다."""
        self._lock = threading.Lock()
        self._version = 0
        self._state = ViewState.WAITING
        self._item: ViewItem | None = None
        self._last_frame = None
        self._message = "waiting for first detection"
        self._events: deque[str] = deque(maxlen=RECENT_EVENTS)
        self._stopped = False
        self.shown = 0          # 칸에 넣은 현재 결과 수(시험·보고용)
        self.cleared = 0        # 결과를 지운 수

    # --- 제어 흐름(runner 스레드) ---

    def on_outcome(self, outcome: JobOutcome, current: bool, *, tracks: TrackFrame | None = None) -> None:
        """terminal 응답 하나. `current`는 runner의 `display_gate`(+수신 epoch) 판정이다.

        duplicate는 무시한다. current면 칸을 교체한다. 그 밖(늦은·폐기·실패·중단)은 결과를 지우고 상태만 바꾼다.
        `tracks`(#151, 선택)는 같은 작업의 TrackFrame이어야 하며(7개 식별자 대조) 아니면 ValueError(칸 무변). 칸 항목에 같이 둔다.
        """
        if outcome.duplicate:
            return

        if tracks is not None and not matches_job(tracks, outcome.context):
            raise ValueError("tracks do not belong to this outcome's job")

        with self._lock:
            if self._stopped:
                return

            if current and outcome.status is JobStatus.COMPLETED:
                ctx, obs = outcome.context, outcome.observation
                display = outcome.display_result
                conf = getattr(ctx.settings.detector_config, "conf_threshold", None)
                valid = getattr(obs, "valid", None)
                self._set(ViewState.LIVE, status_text(valid),
                          item=ViewItem(frame=ctx.frame, result=display if isinstance(display, AiSegResult) else None,
                                        job_seq=ctx.job_seq, generation=ctx.generation,
                                        conf_threshold=conf, reason=getattr(obs, "reason", None), tracks=tracks,
                                        valid=valid),
                          last_frame=None)
                self.shown += 1
                return

            last = self._current_frame()
            seq = outcome.context.job_seq

            if outcome.status is JobStatus.COMPLETED:
                self._set(ViewState.STALE, f"result not current (job {seq}); detection cleared", last_frame=last)
            elif outcome.status is JobStatus.DISCARDED_GENERATION:
                self._set(ViewState.STALE, f"result discarded: generation changed (job {seq})", last_frame=last)
            elif outcome.error and outcome.error.startswith("stop_requested"):
                self._set(ViewState.STOPPING, "stop requested", last_frame=last)
            elif outcome.error and outcome.error.startswith("input_boundary"):
                self._set(ViewState.STALE, f"input boundary: result dropped (job {seq})", last_frame=last)
            else:
                self._set(ViewState.ERROR, f"detection {outcome.status.value}: {outcome.error}", last_frame=last)

            self._events.append(self._message)

    def on_input(self, kind: str, detail: str, generation: int, generation_changed: bool) -> None:
        """입력 경계·알림 하나(실시간). 결과를 즉시 지우고 마지막 원본은 「마지막 영상」으로만 남긴다."""
        with self._lock:
            if self._stopped:
                return

            # epoch를 바꾸지 않는 정보 알림(되감기 필터 해제)은 결과를 지우지 않고 사건만 남긴다
            if kind in _INFO_ONLY_INPUTS:
                self._events.append(f"input {kind}: {detail}")
                self._version += 1
                return

            text = f"input {kind} (gen {generation}{', changed' if generation_changed else ''}): {detail}"
            state = ViewState.ERROR if kind in ("receiver_failed", "start_failed", "qos_incompatible") \
                else ViewState.STALE
            self._set(state, text, last_frame=self._current_frame())
            self._events.append(text)

    def finish(self, exit_code: int, stop_reason: str) -> None:
        """runner가 끝났다. 결과를 지우고 이후 전달을 받지 않는다."""
        with self._lock:
            text = f"run ended: exit={exit_code} reason={stop_reason or '-'}"
            self._stopped = True
            self._set(ViewState.ENDED if exit_code == 0 else ViewState.ERROR, text, last_frame=self._current_frame())
            self._events.append(text)

    # --- 창(메인 스레드) ---

    def request_stop(self) -> None:
        """사용자 정지 요청. 이후 새 검출을 칸에 넣지 않고 지금 결과도 지운다."""
        with self._lock:
            if self._stopped:
                return

            self._stopped = True
            self._set(ViewState.STOPPING, "stopping (user request)", last_frame=self._current_frame())

    def snapshot(self) -> ViewSnapshot:
        """지금 상태를 꺼낸다(참조만, 복사 없음)."""
        with self._lock:
            return ViewSnapshot(version=self._version, state=self._state, item=self._item,
                                last_frame=self._last_frame, message=self._message, events=tuple(self._events))

    def is_current(self, version: int) -> bool:
        """꺼낸 뒤 변경이 없었는가(그리기 직전 확인)."""
        with self._lock:
            return version == self._version

    @property
    def version(self) -> int:
        """현재 버전."""
        with self._lock:
            return self._version

    # --- 내부(잠금 안) ---

    def _current_frame(self):
        """결과를 지울 때 남길 마지막 원본(결과와 짝인 원본 또는 이미 남긴 원본)."""
        return self._item.frame if self._item is not None else self._last_frame

    def _set(self, state: ViewState, message: str, *, item: ViewItem | None = None, last_frame=None) -> None:
        """상태를 바꾸고 버전을 올린다. 결과가 있던 칸을 비우면 지운 수를 센다."""
        if self._item is not None and item is None:
            self.cleared += 1

        self._state, self._message, self._item, self._last_frame = state, message, item, last_frame
        self._version += 1


# --- 그리기 ---

def _color(class_index: int) -> tuple[int, int, int]:
    """클래스 번호별 색(초록 없음)."""
    return CLASS_COLORS_BGR[class_index % len(CLASS_COLORS_BGR)]


def track_label(a: TrackAssignment | None) -> str:
    """추적 판정 라벨(#151 S3): confirmed `#ID`, tentative `trk?`, ambiguous `amb`, 그 밖 빈 문자열. 표시 실험 번호다."""
    if a is None:
        return ""

    if a.state is TrackState.CONFIRMED and a.part_track_id is not None:
        return f"#{a.part_track_id}"

    return {TrackState.TENTATIVE: "trk?", TrackState.AMBIGUOUS: "amb"}.get(a.state, "")


def draw_detections(image: np.ndarray, result: AiSegResult, tracks: TrackFrame | None = None, *,
                    mode_text: str | None = None) -> tuple[np.ndarray, int]:
    """원본을 복사해 인스턴스를 그린 표시용 영상과 그린 대상 수를 돌려준다. **원본 배열은 바꾸지 않는다.**

    `tracks`(선택, #151)가 있으면 그 결과와 같은 인스턴스 index 집합이어야 하며(아니면 ValueError) 라벨 끝에 `track_label`을 붙이고,
    영상 왼쪽 위에 실험 표기(`TRACKING EXPERIMENTAL (<mode>) - ids display-only, no identity guarantee`)를 그린다. 그래서 이 영상만 받는
    쪽(C 전달 `annotate`)에도 번호의 실험 상태가 남는다. tracks가 없으면 기존 그림과 같다. `mode_text`(#242)를 주면 배너의 모드 자리에
    그 문자열을 쓴다(예: `world, prev` = 직전 W 결과를 짝지은 표시 전용 번호). 주지 않으면 `tracks.mode` 값.
    색·채우기·거부/잘림 표기는 바꾸지 않는다(추적 성공을 초록 등 작업 가능 신호로 표시하지 않음).

    좌표는 `AiInstance`의 원본 픽셀 좌표(ROI 오프셋 반영 끝)를 그대로 쓴다(ROI를 다시 더하지 않음). 대상 범위(in_scope)
    인스턴스는 클래스 색으로 윤곽 다각형 반투명 채우기(α=`MASK_ALPHA`)·윤곽선·박스·`클래스 conf`를 그린다. 기하 거부(EMPTY/SMALL)는
    사유를 붙인다. 비대상 클래스·점수 미달은 그리지 않는다(수는 정보 줄에 따로).
    """
    by_index = None

    if tracks is not None:
        by_index = {a.instance_index: a for a in tracks.assignments}

        if set(by_index) != {i.index for i in result.instances}:
            raise ValueError("tracks do not match the result's instance indexes")

    canvas = image.copy()
    fill = canvas.copy()
    drawn = 0
    polys = []

    for inst in result.instances:
        if not inst.in_scope:
            continue

        color = _color(inst.class_index) if inst.rejected is None else REJECTED_COLOR_BGR

        if inst.polygon is not None:
            pts = np.round(inst.polygon).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(fill, [pts], color)
            polys.append((pts, color))

        drawn += 1

    cv2.addWeighted(fill, MASK_ALPHA, canvas, 1.0 - MASK_ALPHA, 0.0, dst=canvas)

    for pts, color in polys:
        cv2.polylines(canvas, [pts], True, color, 2, cv2.LINE_AA)

    for inst in result.instances:
        if not inst.in_scope:
            continue

        color = _color(inst.class_index) if inst.rejected is None else REJECTED_COLOR_BGR
        x1, y1, x2, y2 = (int(round(v)) for v in inst.box_xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{inst.class_name} {inst.confidence:.2f}" + (f" {inst.rejected}" if inst.rejected else "")
        tag = track_label(None if by_index is None else by_index.get(inst.index))
        label += f" {tag}" if tag else ""
        _label(canvas, label, (x1, max(y1 - 4, 14)), color)

    if tracks is not None:
        _label(canvas, f"TRACKING EXPERIMENTAL ({mode_text or tracks.mode.value}) - ids display-only, no identity guarantee",
               (4, 16), STATE_COLORS_BGR["stale"], 0.45)

    return canvas, drawn


def draw_track_unavailable(canvas: np.ndarray, mode: str, why: str) -> None:
    """번호를 그릴 수 없는 추적 프레임의 실험 표기(#239 W 표시). `canvas`에 직접 그린다.

    `why`: `pending`(그 작업 결과가 아직 없음) 또는 사유(`tf_*`·`job_mismatch` 등). 번호 라벨은 그리지 않는다.
    """
    text = (f"TRACKING EXPERIMENTAL ({mode}) - ids pending, display-only" if why == "pending" else
            f"TRACKING EXPERIMENTAL ({mode}) - ids unavailable ({why}), display-only")
    _label(canvas, text, (4, 16), STATE_COLORS_BGR["stale"], 0.45)


def _label(canvas: np.ndarray, text: str, org: tuple[int, int], color, scale: float = 0.5) -> None:
    """배경 상자 위에 글자를 쓴다."""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(canvas, (x, y - h - base), (x + w + 2, y + base), (0, 0, 0), -1)
    cv2.putText(canvas, text, (x + 1, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _info_lines(snap: ViewSnapshot) -> list[str]:
    """정보 줄 문자열. 측정하지 않은 FPS·서로 다른 시계 차이(지연)는 넣지 않는다."""
    item = snap.item
    lines = [f"[{snap.state.value.upper()}] {snap.message}"]

    if item is not None:
        f = item.frame
        base = getattr(f.timestamp_base, "value", f.timestamp_base)
        lines.append(f"run {f.run_id}  src {f.source_id}  image_seq {f.image_seq}  gen {item.generation}  "
                     f"job {item.job_seq}")
        conf = "-" if item.conf_threshold is None else f"{item.conf_threshold:g}"

        if item.result is not None:
            r = item.result
            out = len(r.instances) - r.candidate_count
            lines.append(f"t {f.timestamp_ms / 1000:.3f} s ({base})  detect {r.process_ms:.1f} ms "
                         f"(infer {r.infer_ms:.1f})  conf>={conf}  detections: {r.candidate_count}"
                         + (f"  (excluded {out})" if out else ""))
        else:
            lines.append(f"t {f.timestamp_ms / 1000:.3f} s ({base})  conf>={conf}  "
                         "display result is not AI segmentation (not drawn)")

        lines.append(f"DISPLAY ONLY - observation valid={item.valid} ({item.reason}); not a robot target")

        if item.tracks is not None:
            t = item.tracks
            n_id = sum(1 for a in t.assignments if a.state is TrackState.CONFIRMED)
            n_tent = sum(1 for a in t.assignments if a.state is TrackState.TENTATIVE)
            n_amb = sum(1 for a in t.assignments if a.state is TrackState.AMBIGUOUS)
            state = "" if t.updated else f"  NOT UPDATED ({t.assignments[0].reason if t.assignments else '-'})"
            lines.append(f"TRACKING EXPERIMENTAL ({t.mode.value}): display-only ids, no identity guarantee; "
                         f"ids {n_id} tentative {n_tent} ambiguous {n_amb}  epoch {t.tracker_epoch}"
                         + (f"  event {t.event}" if t.event else "") + state)
    else:
        lines.append("no current detection" + (" - showing LAST IMAGE (not current)" if snap.last_frame is not None
                                               else ""))

    lines.extend(f"  {e}" for e in snap.events[-2:] if e != snap.message)   # 첫 줄과 같은 사건은 되풀이하지 않음

    return lines


def render_view(snap: ViewSnapshot, max_width: int = 1600) -> np.ndarray:
    """꺼낸 상태로 창 영상 하나를 만든다(좌 원본·우 오버레이·위 정보 줄). 입력 배열은 바꾸지 않는다.

    결과가 없으면 마지막 원본을 어둡게 하고 「LAST IMAGE」를 표시해 현재 결과처럼 보이지 않게 한다. 원본도 없으면 빈 화면.
    두 화면을 합친 폭이 `max_width`를 넘으면 같은 비율로 줄인다(좌표 대응 유지).
    """
    item = snap.item

    if item is not None:
        original = item.frame.image
        overlay = draw_detections(original, item.result, item.tracks)[0] if item.result is not None else original.copy()
        left, right = original.copy(), overlay
    elif snap.last_frame is not None:
        dim = (snap.last_frame.image.astype(np.float32) * 0.4).astype(np.uint8)
        left, right = dim, dim.copy()

        for pane in (left, right):
            _label(pane, "LAST IMAGE - NOT CURRENT", (10, 30), STATE_COLORS_BGR["stale"], 0.8)
    else:
        left = np.zeros((360, 480, 3), np.uint8)
        _label(left, "NO IMAGE", (10, 30), STATE_COLORS_BGR["waiting"], 0.8)
        right = left.copy()

    _label(left, "ORIGINAL", (10, left.shape[0] - 10), (255, 255, 255))
    _label(right, "DETECTION OVERLAY", (10, right.shape[0] - 10), (255, 255, 255))
    pair = np.hstack([left, np.full((left.shape[0], 4, 3), 64, np.uint8), right])

    if pair.shape[1] > max_width:
        scale = max_width / pair.shape[1]
        pair = cv2.resize(pair, (max_width, max(1, int(round(pair.shape[0] * scale)))), interpolation=cv2.INTER_AREA)

    lines = _info_lines(snap)
    header = np.zeros((22 * len(lines) + 8, pair.shape[1], 3), np.uint8)
    color = STATE_COLORS_BGR[snap.state.value]
    cv2.rectangle(header, (0, 0), (8, header.shape[0]), color, -1)

    for i, text in enumerate(lines):
        cv2.putText(header, text, (14, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color if i == 0 else (230, 230, 230), 1, cv2.LINE_AA)

    return np.vstack([header, pair])


__all__ = ["DisplayBridge", "MASK_ALPHA", "ViewItem", "ViewSnapshot", "ViewState", "draw_detections", "draw_track_unavailable", "render_view", "status_text",
           "track_label"]
