"""B center 그리기 문(#406 ①): center 영상을 볼 구독자가 없으면 그리지도 발행하지도 않는다.

C 중계기는 대시보드가 링크 채널에서 center를 고른 동안만 B center 토픽을 구독한다. 그래서 구독자 수가 0이면
아무도 center 영상을 보지 않는다. 그 동안 B 계산·`/b/results`·수락·목표는 그대로 하고, 그리기(`render_center`)와
영상 발행만 건너뛴다. rviz 등 다른 구독자가 붙어도 구독자 수로 잡힌다.

B 파일은 바꾸지 않는다: 그릴 때는 B `center_overlay_consumer`를 그대로 부르고, 건너뛸 때는 `stage.process_frame`과
B 공개 함수 `overlay_keys`·`results_payload`로 결과 숫자만 발행한다.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

CHECK_S = 0.1   # 구독자 수를 다시 묻는 간격(그 사이는 마지막 답을 쓴다)


def subscriber_total(publisher) -> int | None:
    """center 영상 토픽 두 개(영상·번호표)의 구독자 수 합. 셀 수 없으면 None(늘 그림)."""
    total = 0
    known = False
    count = getattr(publisher, "subscriber_count", None)

    if callable(count):
        total += int(count())
        known = True

    overlay = getattr(publisher, "_overlay_pub", None)   # B 발행기의 번호표 발행기(공개 함수 없음)

    if overlay is not None and callable(getattr(overlay, "get_subscription_count", None)):
        total += int(overlay.get_subscription_count())
        known = True

    return total if known else None


class CenterDrawGate:
    """그릴지 정한다. `always`면 늘 그림. 판단 전환 시각을 남긴다(진단·실측용)."""

    def __init__(self, publisher, *, always: bool = False, check_s: float = CHECK_S,
                 clock: Callable[[], float] = time.monotonic, count: Callable | None = None) -> None:
        """`count(publisher)`는 시험 주입용(기본 `subscriber_total`)."""
        self._publisher = publisher
        self.always = bool(always)
        self._check_s = float(check_s)
        self._clock = clock
        self._count = count or subscriber_total
        self._lock = threading.Lock()
        self._checked_at: float | None = None
        self._wanted = True
        self.drawn = 0
        self.skipped = 0
        self.count_errors = 0
        self.transitions: list[tuple[float, bool]] = []   # (단조 시각 s, 그림 여부) 바뀐 때만

    def wanted(self) -> bool:
        """지금 그려야 하는가. 구독자 수는 `check_s`마다 한 번만 묻는다."""
        if self.always:
            return True

        now = self._clock()

        with self._lock:
            if self._checked_at is not None and now - self._checked_at < self._check_s:
                return self._wanted

            self._checked_at = now

        try:
            n = self._count(self._publisher)
        except Exception:   # noqa: BLE001 - 셀 수 없으면 그린다(안전 쪽)
            with self._lock:
                self.count_errors += 1
            n = None

        wanted = n is None or n > 0

        with self._lock:
            if wanted != self._wanted or not self.transitions:
                self.transitions.append((now, wanted))

            self._wanted = wanted

        return wanted

    def note(self, drew: bool) -> None:
        """한 건 처리 결과를 센다."""
        with self._lock:
            if drew:
                self.drawn += 1
            else:
                self.skipped += 1

    def summary(self) -> dict:
        """요약 블록."""
        with self._lock:
            return {"always": self.always, "check_s": self._check_s, "drawn": self.drawn, "skipped": self.skipped,
                    "count_errors": self.count_errors,
                    "transitions": [{"t_mono_s": round(t, 3), "draw": w} for t, w in self.transitions[-50:]]}


def gated_center_consumer(stage, publisher, gate: CenterDrawGate, *, guides: bool = True,
                          on_error: Callable[[Exception], None] | None = None) -> Callable:
    """B `center_overlay_consumer` 자리의 목록 소비자. 반환값은 `stage.process_frame`과 똑같다.

    그릴 때: B `center_overlay_consumer` 그대로(계산 → 그리기 → 영상·결과 발행).
    건너뛸 때: B 계산 → 결과 숫자(`/b/results`)만 발행. 발행 실패는 B 결과에 영향을 주지 않는다(`on_error`).
    """
    from eyetoarm.transforms.center_overlay_pub import center_overlay_consumer, overlay_keys, results_payload

    full = center_overlay_consumer(stage, publisher, guides=guides, on_error=on_error)

    def consume(envelope) -> dict:
        if gate.wanted():
            gate.note(True)
            return full(envelope)

        rec = stage.process_frame(envelope)
        gate.note(False)

        try:
            send_results = getattr(publisher, "publish_results", None)

            if send_results is not None:
                send_results(results_payload(rec, overlay_keys(envelope)))
        except Exception as exc:  # noqa: BLE001 - 표시 실패가 B 계산·기록을 막지 않게
            consume.errors += 1

            if on_error is not None:
                on_error(exc)

        return rec

    consume.errors = 0
    consume.full = full
    return consume


__all__ = ["CHECK_S", "CenterDrawGate", "gated_center_consumer", "subscriber_total"]
