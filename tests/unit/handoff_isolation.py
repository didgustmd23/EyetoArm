"""시험 전용 C 연결부 격리: 단위·CLI 시험이 운영 기본 127.0.0.1:47093으로 detect 영상을 보내지 않게 한다.

`NullForwarder`·`IsolatedHandoffProbe`·`isolate_cli(mod)`로 기본 연결부를 송신 없는 대역으로 바꾼다.
"""
from __future__ import annotations

import threading

from eyetoarm.vision.c_handoff_probe import HandoffProbe
from eyetoarm.vision.handoff import AToCHandoff


class NullForwarder:
    """영상 송신 대역: 소켓 없이 받은 수만 센다."""

    def __init__(self) -> None:
        """계수와 잠금(C 소비 스레드가 부름)."""
        self._lock = threading.Lock()
        self.offered = 0
        self.closed = False

    def offer(self, view) -> bool:
        """받은 영상을 센다(보내지 않음). 실제 forwarder처럼 넣었다고 답한다."""
        with self._lock:
            self.offered += 1

        return True

    def close(self) -> None:
        """닫기 기록."""
        self.closed = True


class IsolatedHandoffProbe(HandoffProbe):
    """실제 소비 스레드 + 송신 없는 forwarder."""

    def __init__(self, *args, **kwargs) -> None:
        """상위 생성 뒤 실제 forwarder를 닫고 `NullForwarder`로 바꾼다."""
        super().__init__(*args, **kwargs)
        self._forwarder.close()
        self._forwarder = NullForwarder()


def isolated_live_handoff():
    """`run_pipeline._default_live_handoff` 대역(`AToCHandoff`, `IsolatedHandoffProbe`). 시작은 호출자."""
    probe = IsolatedHandoffProbe()

    return AToCHandoff(notify=probe.notify), probe


def isolate_cli(mod) -> None:
    """`run_pipeline` 모듈의 기본 C 연결부를 송신 없는 대역으로 바꾼다(그 모듈 객체에만)."""
    mod._default_live_handoff = isolated_live_handoff
    mod.HandoffProbe = IsolatedHandoffProbe


__all__ = ["IsolatedHandoffProbe", "NullForwarder", "isolate_cli", "isolated_live_handoff"]
