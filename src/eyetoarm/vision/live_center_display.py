"""실시간 B 중심점 주석 영상의 표시 전용 송출(#213 L1.5): 실시간 A→B 기록 전용 연결의 결과를 영상 토픽 하나로만 내보낸다.

근거: `_workspace/213_builder2_live_center_display_plan_r0.md`·`_r1.md`(reviewer PASS, 사용자 승인 2026-11-20, Q7 = 사유 화면 낮은 빈도).
**표시 전용**이다: C 대기열·`/sort_part`·로봇 목표와 연결하지 않는다. D13(실행기의 실시간 `b_consumer` 거부)과 `stage.on_output` 미연결은
그대로다. 표시 영상 출구 하나만 연다.

보내는 것
- 수락 주석(`accepted`): 검사 ③을 통과한 결과의 `render_center` 배열 + 상단 띠. 수락끼리만 촬영 stamp 단조(이하이면 버림).
- 상태 화면(`status`): 원본 영상 + 띠(가장 최근 무효 사유). `status_enabled`이고, 마지막 수락 발행(없으면 시작) 뒤 `status_hold_s`가
  지났고, 마지막 상태 화면 뒤 `1/status_max_hz`가 지났고, 그 사유 프레임 stamp가 마지막 수락 stamp보다 새로울 때만. 상태 화면 stamp는
  수락 단조 비교에 들어가지 않는다(r1 F1).
- 끝 표시(`marker`): 실행이 닫히면 한 번 「보내 보는 것」(검은 영상, stamp 0). 전달은 보장하지 않는다(r1 F2).
- 보내지 않는 사유: `not_accepted`·`superseded`·`dropped_on_close`·`run_ended`(`NO_STATUS_REASONS`).

스레드
- 넣기(`offer_accepted`·`offer_status`)는 worker·제어 흐름에서 불리며 표시 잠금 하나로 칸에 넣기만 한다(그리기·인코딩·발행 없음).
- 표시 스레드(`eyetoarm-live-center-display`) 하나가 띠 그리기·발행·발행자 수 확인을 한다. 발행이 늦어도 칸이 덮어써질 뿐이다.
- `close_run(reason)`: 닫힘 표시·칸 비우기만(소스 slot 잠금 안에서도 불릴 수 있음). 그 뒤 넣기는 버린다. 닫힘 확인과 발행 시작
  표시는 같은 잠금 구간이라 닫힌 뒤 새로 시작하는 발행은 0이다(닫힘 순간 이미 발행 중이던 1장은 `publish_in_flight_at_close`).
- 실패: 발행 예외가 연속 `publish_fail_limit`회면 표시 멈춤(`publish_failed`). 실행 중 같은 토픽 발행자가 자기 말고 더 보이면 멈춤
  (`foreign_publisher`). 표시 스레드 비정상 종료는 `crash`. 어느 것도 B 상태를 fail-stop하지 않는다(진입점이 종료 1로 드러냄).

발행 함수는 주입한다(`publisher.publish(DisplayFrame)`·`count_publishers()`·`close(limit_s)`). 이 모듈은 ROS를 import하지 않는다.
띠 글자는 OpenCV Hershey 글꼴이라 ASCII만 쓴다.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

DEFAULT_CENTER_TOPIC = "/b/center_overlay/image"
NO_STATUS_REASONS = frozenset({"not_accepted", "superseded", "dropped_on_close", "run_ended"})
_CONFIG_KEYS = {"discovery_wait_s", "graph_check_s", "publish_fail_limit", "qos_depth", "close_limit_s",
                "status_enabled", "status_hold_s", "status_max_hz"}
_BANNER_H = 44
_FOOTER = "DISPLAY ONLY | test_only | not a robot target"


class CenterDisplayConfigError(ValueError):
    """`center_display` 설정 오류."""


@dataclass(frozen=True)
class CenterDisplayConfig:
    """표시 송출 시험 설정(기본값 없음, 모두 필수). 값은 미확정 운영값의 시험값이다."""

    discovery_wait_s: float     # 시작 때 발행자 확인 전 발견 대기(최선 노력, r1 F4)
    graph_check_s: float        # 실행 중 발행자 수 확인 주기(실제 방어)
    publish_fail_limit: int     # 연속 발행 실패 한도
    qos_depth: int              # reliable·keep_last 깊이(volatile)
    close_limit_s: float        # 표시 스레드 join 상한
    status_enabled: bool        # 상태 화면 송출 여부(사용자 선택 Q7)
    status_hold_s: float        # 수락이 이만큼 없을 때만 상태 화면
    status_max_hz: float        # 상태 화면 주기 상한

    def to_json(self) -> dict:
        """요약용 사전."""
        return {k: getattr(self, k) for k in sorted(_CONFIG_KEYS)}


def _pos_number(doc: dict, key: str, *, integer: bool = False):
    value = doc[key]

    if isinstance(value, bool) or not isinstance(value, int if integer else (int, float)) \
            or (not integer and not math.isfinite(value)) or value <= 0:
        raise CenterDisplayConfigError(f"center_display.{key} must be a positive {'int' if integer else 'number'}: {value!r}")

    return value if integer else float(value)


def parse_center_display(doc) -> CenterDisplayConfig:
    """`center_display` 블록을 검사한다. 모르는 키·누락·형 오류는 `CenterDisplayConfigError`."""
    if not isinstance(doc, dict):
        raise CenterDisplayConfigError("center_display must be an object")

    unknown, missing = set(doc) - _CONFIG_KEYS, _CONFIG_KEYS - set(doc)

    if unknown or missing:
        raise CenterDisplayConfigError(f"center_display: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")

    if not isinstance(doc["status_enabled"], bool):
        raise CenterDisplayConfigError(f"center_display.status_enabled must be a bool: {doc['status_enabled']!r}")

    return CenterDisplayConfig(
        discovery_wait_s=_pos_number(doc, "discovery_wait_s"), graph_check_s=_pos_number(doc, "graph_check_s"),
        publish_fail_limit=int(_pos_number(doc, "publish_fail_limit", integer=True)),
        qos_depth=int(_pos_number(doc, "qos_depth", integer=True)), close_limit_s=_pos_number(doc, "close_limit_s"),
        status_enabled=doc["status_enabled"], status_hold_s=_pos_number(doc, "status_hold_s"),
        status_max_hz=_pos_number(doc, "status_max_hz"))


@dataclass(frozen=True)
class DisplayFrame:
    """발행할 영상 하나. `stamp_ns`는 원본 촬영 stamp(끝 표시는 0), `frame_id`는 광학 프레임."""

    kind: str                 # "accepted" | "status" | "marker"
    image: np.ndarray         # BGR uint8 (H, W, 3), 표시 스레드가 만든 복사본
    stamp_ns: int
    frame_id: str


@dataclass(frozen=True)
class _Pending:
    frame: Any
    image: np.ndarray | None   # 수락: 주석 배열, 상태: None(원본은 frame.image)
    stamp_ns: int
    text: str


def _stamp_text(stamp_ns: int | None) -> str:
    if stamp_ns is None:
        return "t=?"

    return f"t={stamp_ns // 1_000_000_000}.{(stamp_ns // 1_000_000) % 1000:03d}"


def _ids_text(run_id: str, ids: dict) -> str:
    return f"run={str(run_id)[:8]} gen={ids.get('generation')} job={ids.get('job_seq')}"


def accepted_text(out, view_bias_records) -> str:
    """수락 띠 문구: 규격·길이·각도(+ 시야 보정 표기). 값이 없으면 `?`."""
    res = getattr(out, "result", None)
    spec = getattr(res, "spec", None)
    length_m = getattr(res, "measured_length_m", None)
    theta = getattr(res, "theta_deg", None)
    parts = [f"OK {spec if spec is not None else '?'}",
             f"L={1000.0 * length_m:.1f}mm" if isinstance(length_m, (int, float)) and math.isfinite(length_m) else "L=?",
             f"th={theta:.1f}" if isinstance(theta, (int, float)) and math.isfinite(theta) else "th=?"]

    if view_bias_records:
        rec = view_bias_records[0]
        parts.append("vb=applied" if rec.get("applied") else f"vb=skip:{rec.get('reason')}")

    return " ".join(parts)


def _draw_banner(image: np.ndarray, line1: str, ok: bool) -> np.ndarray:
    """복사본 위에 두 줄 띠를 그린다(원본은 바꾸지 않음)."""
    img = np.array(image, dtype=np.uint8, copy=True, order="C")
    color = (0, 110, 0) if ok else (0, 0, 170)
    cv2.rectangle(img, (0, 0), (img.shape[1], _BANNER_H), color, -1)
    cv2.putText(img, line1, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, _FOOTER, (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return img


class LiveCenterDisplay:
    """실행 하나의 표시 송출기. 칸(수락 1·사유 1)은 표시 잠금이, 발행·그래프 확인은 표시 스레드가 소유한다."""

    def __init__(self, run_id: str, config: CenterDisplayConfig, publisher, *,
                 clock_ns: Callable[[], int] = time.monotonic_ns, is_live: Callable[[], bool] | None = None) -> None:
        """`publisher`: `publish(DisplayFrame)`·`count_publishers() → int`·`close(limit_s)`. `is_live()`가 거짓이면 발행하지 않는다
        (상태 fail-stop 등 연결부가 아는 실행 끝)."""
        self.run_id = run_id
        self.config = config
        self._publisher = publisher
        self._clock_ns = clock_ns
        self._is_live = is_live or (lambda: True)
        self._cond = threading.Condition()
        self._accepted: _Pending | None = None
        self._status: _Pending | None = None
        self._closed = False
        self._close_reason: str | None = None
        self._stop = False
        self._publishing = False
        self._thread: threading.Thread | None = None
        self._marker_done = False
        self._last_shape: tuple | None = None
        self._last_frame_id: str = ""
        self._start_ns = 0                         # start()에서 정함
        self._last_acc_stamp: int | None = None
        self._last_acc_pub_ns: int | None = None
        self._last_status_pub_ns: int | None = None
        self._next_graph_ns = 0                    # start()에서 정함
        self._consecutive_fail = 0
        self._publisher_closed = False
        self.stopped: str | None = None           # 표시 멈춤 사유(publish_failed | foreign_publisher)
        self.crash: str | None = None
        self.first_error: str | None = None
        self.counts = {k: 0 for k in ("published_accepted", "published_status", "published_marker", "accepted_offered",
                                      "accepted_replaced", "accepted_older_dropped", "status_offered", "status_replaced",
                                      "status_suppressed_stale", "status_older_ignored", "cleared_on_reset", "offered_after_close",
                                      "dropped_not_live", "publish_errors", "publish_in_flight_at_close",
                                      "graph_check_errors", "foreign_publishers_seen")}
        self.published_stamps: list[tuple[str, int]] = []   # (kind, stamp) 시험·요약용
        self.loop_count = 0                                  # 표시 스레드 바퀴 수(시험 동기화용)

    # ------------------------------------------------------------------ 넣기(worker·제어 흐름)

    def offer_accepted(self, frame, annotated: np.ndarray, out, ids: dict, view_bias_records=None) -> None:
        """수락 결과 하나. 마지막 수락 발행 stamp 이하이면 버린다. 칸에 있던 수락은 덮어쓴다(최신 우선)."""
        stamp_ns = getattr(frame, "ros_stamp_ns", None)

        if stamp_ns is None or annotated is None:
            return

        text = f"{_stamp_text(stamp_ns)} {_ids_text(self.run_id, ids)} {accepted_text(out, view_bias_records)}"

        with self._cond:
            self.counts["accepted_offered"] += 1

            if self._closed:
                self.counts["offered_after_close"] += 1
                return

            newest = max(s for s in (self._last_acc_stamp, getattr(self._accepted, "stamp_ns", None), -1) if s is not None)

            if stamp_ns <= newest:
                self.counts["accepted_older_dropped"] += 1
                return

            if self._accepted is not None:
                self.counts["accepted_replaced"] += 1

            self._accepted = _Pending(frame, annotated, stamp_ns, text)
            self._cond.notify_all()

    def offer_status(self, frame, reason: str, ids: dict) -> None:
        """무효 사유 하나(가장 최근 것만 남김). 보내지 않는 사유·상태 화면 끔·영상 없음이면 무시."""
        if not self.config.status_enabled or reason in NO_STATUS_REASONS:
            return

        stamp_ns = getattr(frame, "ros_stamp_ns", None)

        if stamp_ns is None or getattr(frame, "image", None) is None:
            return

        text = f"{_stamp_text(stamp_ns)} {_ids_text(self.run_id, ids)} NO B: {reason}"

        with self._cond:
            self.counts["status_offered"] += 1

            if self._closed:
                self.counts["offered_after_close"] += 1
                return

            if self._status is not None:
                if stamp_ns < self._status.stamp_ns:   # 더 오래된 프레임의 사유로 바꾸지 않는다
                    self.counts["status_older_ignored"] += 1
                    return

                self.counts["status_replaced"] += 1

            self._status = _Pending(frame, None, stamp_ns, text)
            self._cond.notify_all()

    def close_run(self, reason: str) -> None:
        """실행 닫힘: 칸을 비우고 이후 넣기를 버린다. 메모리 작업만(slot 잠금 안에서 불려도 됨)."""
        with self._cond:
            if self._closed:
                return

            self._closed = True
            self._close_reason = reason
            self.counts["cleared_on_reset"] += (self._accepted is not None) + (self._status is not None)
            self._accepted = self._status = None

            if self._publishing:
                self.counts["publish_in_flight_at_close"] += 1

            self._cond.notify_all()

    # ------------------------------------------------------------------ 수명

    def start(self) -> None:
        """표시 스레드를 시작한다."""
        if self._thread is not None:
            raise RuntimeError("center display already started")

        now = self._clock_ns()
        self._start_ns = now
        self._next_graph_ns = now + int(self.config.graph_check_s * 1e9)
        self._thread = threading.Thread(target=self._loop, name="eyetoarm-live-center-display", daemon=True)
        self._thread.start()

    @property
    def thread(self) -> threading.Thread | None:
        """표시 스레드(정리 확인용)."""
        return self._thread

    def close(self, limit_s: float) -> bool:
        """닫힘(이미 했으면 그대로) → 끝 표시 시도 → 스레드 join → 발행자 닫기. 재시도할 수 있다. 정리되면 True."""
        self.close_run(self._close_reason or "close")

        with self._cond:
            self._stop = True
            self._cond.notify_all()

        if self._thread is not None:
            self._thread.join(timeout=limit_s)

            if self._thread.is_alive():
                return False
        else:
            self._try_marker()

        if not self._publisher_closed:
            self._publisher.close(limit_s)
            self._publisher_closed = True

        if self.crash is not None:   # 스레드는 끝났지만 비정상 종료: 정리 확인으로 치지 않는다
            raise RuntimeError(f"center display crashed: {self.crash}")

        return True

    # ------------------------------------------------------------------ 표시 스레드

    def _loop(self) -> None:
        """표시 스레드 최상단: 예기치 않은 예외는 `crash`로 남긴다(B 기록은 계속)."""
        try:
            self._loop_body()
        except Exception as exc:
            with self._cond:
                self.crash = f"{type(exc).__name__}: {exc}"

    def _take(self, now: int) -> _Pending | None:
        """잠금 안: 보낼 것 하나(수락 우선). 상태 화면 조건을 본다."""
        if self._accepted is not None:
            item, self._accepted = self._accepted, None
            return item

        st = self._status

        if st is None:
            return None

        if self._last_acc_stamp is not None and st.stamp_ns <= self._last_acc_stamp:
            self._status = None
            self.counts["status_suppressed_stale"] += 1
            return None

        since_acc = now - (self._last_acc_pub_ns if self._last_acc_pub_ns is not None else self._start_ns)

        if since_acc < self.config.status_hold_s * 1e9:
            return None

        if self._last_status_pub_ns is not None and now - self._last_status_pub_ns < 1e9 / self.config.status_max_hz:
            return None

        self._status = None

        return st

    def _loop_body(self) -> None:
        while True:
            live = self._is_live()

            with self._cond:
                self.loop_count += 1
                now = self._clock_ns()
                item = None
                marker_due = False

                if self._stop:
                    break

                if not self._closed and self.stopped is None:
                    if not live:
                        self.counts["dropped_not_live"] += (self._accepted is not None) + (self._status is not None)
                        self._accepted = self._status = None
                    else:
                        item = self._take(now)

                        if item is not None:
                            self._publishing = True

                if item is None:
                    check_graph = self.stopped is None and not self._closed and now >= self._next_graph_ns
                    marker_due = self._closed and not self._marker_done

                    if not check_graph and not marker_due:
                        self._cond.wait(0.05)   # 상태 화면 시각 판정을 다시 보기 위한 짧은 대기
                        continue

            if item is not None:
                self._publish_item(item, now)
                continue

            if marker_due:
                self._try_marker()
                continue

            self._graph_check()

        self._try_marker()

    def _publish_item(self, item: _Pending, now: int) -> None:
        """띠를 그려 발행한다(잠금 밖). 실패를 세고 한도면 멈춘다."""
        accepted = item.image is not None
        base = item.image if item.image is not None else item.frame.image
        kind = "accepted" if accepted else "status"

        try:
            img = _draw_banner(base, item.text, accepted)
            frame_id = str(getattr(item.frame, "optical_frame", None) or "")
            self._publisher.publish(DisplayFrame(kind, img, int(item.stamp_ns), frame_id))
        except Exception as exc:   # 발행 실패는 표시만의 문제: 세고 남긴다
            with self._cond:
                self._publishing = False
                self.counts["publish_errors"] += 1
                self._consecutive_fail += 1
                self.first_error = self.first_error or f"{type(exc).__name__}: {exc}"

                if self._consecutive_fail >= self.config.publish_fail_limit and self.stopped is None:
                    self.stopped = "publish_failed"
                    self._accepted = self._status = None

                self._cond.notify_all()

            return

        with self._cond:
            self._publishing = False
            self._consecutive_fail = 0
            self._last_shape = img.shape
            self._last_frame_id = frame_id
            self.published_stamps.append((kind, int(item.stamp_ns)))

            if accepted:
                self.counts["published_accepted"] += 1
                self._last_acc_stamp = item.stamp_ns
                self._last_acc_pub_ns = now
            else:
                self.counts["published_status"] += 1
                self._last_status_pub_ns = now

            self._cond.notify_all()

    def _graph_check(self) -> None:
        """자기 말고 같은 토픽 발행자가 있으면 표시를 멈춘다. 확인 실패는 세기만 한다."""
        try:
            n = int(self._publisher.count_publishers())
        except Exception:   # 그래프 조회 실패는 발행 판단에 쓰지 않는다(계수로 드러냄)
            n = None

        with self._cond:
            self._next_graph_ns = self._clock_ns() + int(self.config.graph_check_s * 1e9)

            if n is None:
                self.counts["graph_check_errors"] += 1
            elif n > 1:
                self.counts["foreign_publishers_seen"] = max(self.counts["foreign_publishers_seen"], n - 1)

                if self.stopped is None:
                    self.stopped = "foreign_publisher"
                    self._accepted = self._status = None

    def _try_marker(self) -> None:
        """끝 표시 한 번(보내 보기만). 멈춤·발행한 영상 없음이면 보내지 않는다."""
        with self._cond:
            if self._marker_done:
                return

            self._marker_done = True

            if self.stopped is not None or self._last_shape is None:
                return

            shape, frame_id, reason = self._last_shape, self._last_frame_id, self._close_reason

        img = _draw_banner(np.zeros(shape, np.uint8), f"RUN ENDED: {reason} (previous results withdrawn) "
                                                      f"run={str(self.run_id)[:8]}", False)

        try:
            self._publisher.publish(DisplayFrame("marker", img, 0, frame_id))
        except Exception as exc:   # 끝 표시는 보장하지 않는다(r1 F2): 세고 남긴다
            with self._cond:
                self.counts["publish_errors"] += 1
                self.first_error = self.first_error or f"marker: {type(exc).__name__}: {exc}"
            return

        with self._cond:
            self.counts["published_marker"] += 1
            self.published_stamps.append(("marker", 0))

    # ------------------------------------------------------------------ 요약

    def problem(self) -> str | None:
        """진입점이 종료 1로 드러낼 표시 문제(없으면 None)."""
        with self._cond:
            if self.crash is not None:
                return f"center display crashed: {self.crash}"

            if self.stopped is not None:
                return f"center display stopped: {self.stopped} (first_error={self.first_error})"

        return None

    def summary(self) -> dict:
        """요약 블록 `center_display`."""
        with self._cond:
            return {"config": self.config.to_json(), "counts": dict(self.counts), "stopped": self.stopped,
                    "crash": self.crash, "first_error": self.first_error, "close_reason": self._close_reason,
                    "marker_attempted": self._marker_done}


__all__ = ["CenterDisplayConfig", "CenterDisplayConfigError", "DEFAULT_CENTER_TOPIC", "DisplayFrame", "LiveCenterDisplay",
           "NO_STATUS_REASONS", "accepted_text", "parse_center_display"]
