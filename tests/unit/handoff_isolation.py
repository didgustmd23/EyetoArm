"""시험 전용 C 연결부 격리(#251 A-1): 단위·CLI 시험이 운영 기본 127.0.0.1:47093으로 detect 영상을 보내지 않게 한다.

근거: #249 D-1·D-8(전체 pytest가 실행 중인 C 중계기에 detect 380장을 보냄). C 파일 `c_handoff_probe.py`는 바꾸지 않고 시험 쪽에서 막는다.

- `NullForwarder`: `_DisplayFrameForwarder` 자리. 받은 영상을 세기만 하고 소켓을 열지 않는다.
- `IsolatedHandoffProbe`: 실제 `HandoffProbe`(소비 스레드·기록 그대로)에서 forwarder만 `NullForwarder`로 바꾼 것. 생성 때 만들어진 실제
  forwarder 스레드는 바로 닫는다(그 스레드는 영상을 받기 전에는 접속하지 않음).
- `isolate_cli(mod)`: 시험이 불러온 `run_pipeline` 모듈의 기본 연결부(`_default_live_handoff`, 실시간 기록 경로)와 기본 `--ros-link` 경로의
  `HandoffProbe`를 위 가짜로 바꾼다. 실제 연결부를 일부러 확인하는 시험만 이 함수를 쓰지 않고 빈 포트로 격리한다.
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
    """`run_pipeline._default_live_handoff` 대역: (실제 `AToCHandoff`, `IsolatedHandoffProbe`). 시작은 호출자가 한다."""
    probe = IsolatedHandoffProbe()

    return AToCHandoff(notify=probe.notify), probe


def isolate_cli(mod) -> None:
    """불러온 `run_pipeline` 모듈의 기본 C 연결부를 송신 없는 대역으로 바꾼다(그 모듈 객체에만, 다른 시험에 새지 않음)."""
    mod._default_live_handoff = isolated_live_handoff
    mod.HandoffProbe = IsolatedHandoffProbe


__all__ = ["IsolatedHandoffProbe", "NullForwarder", "isolate_cli", "isolated_live_handoff"]
