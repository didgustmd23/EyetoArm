"""수동 펜던트의 조작 규칙과 신호 작성 (D 클라이언트). Qt·웹과 무관한 순수 규칙.

펜던트 조그는 TCP 1.1(MOVE·GET_STATUS·CANCEL)에 없는 신호다. 그래서 여기서 만드는 조그 신호는
`pendant-draft/0.1` 이라는 **D 제안 형식**이며 C와 합의하지 않았다. 전송 경로가 합의되기 전에는
작성과 기록만 하고 보내지 않는다. TCP 1.1 포트로는 절대 보내지 않는다(C 서버가 BAD_FORMAT 으로 거부하는 줄을 만들지 않기 위해).

펜던트가 TCP 1.1 로 실제로 보내는 것은 두 가지뿐이다: START = 수동 목표 MOVE, STOP = CANCEL.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

CONTRACT = "pendant-draft/0.1"
CART_STEPS = (1.0, 5.0, 10.0)      # mm
JOINT_STEPS = (0.1, 1.0, 5.0)      # 도
SPEED_STEPS = (0.05, 0.10, 0.25, 0.50, 1.00)
GRIP_MAX_MM = 85.0                 # Robotiq 2F-85 개구폭
GRIP_STEP_MM = 5.0
CART_AXES = ("X", "Y", "Z", "RZ")
JOINT_AXES = ("J1", "J2", "J3", "J4", "J5", "J6")
# shortcut: 시작 자세는 예시값이다(작업 범위·홈 자세는 B·C 확정 전), 확정되면 설정 파일에서 읽는다
HOME_CART = {"X": 470.0, "Y": 160.0, "Z": 215.0, "RZ": 0.0}
HOME_JOINT = {"J1": 0.0, "J2": -90.0, "J3": 90.0, "J4": -90.0, "J5": -90.0, "J6": 0.0}


@dataclass
class PendantModel:
    enabled: bool = False              # 인에이블(데드맨) 스위치. 꺼져 있으면 조그 신호를 만들지 않는다
    estop: bool = False                # 소프트웨어 정지 잠금. 실물 비상정지가 아니다
    frame: str = "cart"                # cart(직교) · joint(관절)
    step_index: int = 1
    speed_index: int = 2
    grip_mm: float = GRIP_MAX_MM       # 명령한 개구폭
    cart: dict[str, float] = field(default_factory=lambda: dict(HOME_CART))
    joint: dict[str, float] = field(default_factory=lambda: dict(HOME_JOINT))
    seq: int = 0
    target_seq: int = 0                # 수동 목표가 바뀔 때마다 증가(MOVE 의 image_seq 자리에 씀)
    message: str = "먼저 인에이블을 켜세요 (E 키 또는 '인에이블 켜기' 버튼) · 그다음 조그·그리퍼·START 가 동작합니다"

    @property
    def step(self) -> float:
        return (CART_STEPS if self.frame == "cart" else JOINT_STEPS)[self.step_index]

    @property
    def speed_ratio(self) -> float:
        return SPEED_STEPS[self.speed_index]

    def _signal(self, kind: str, **body: Any) -> dict[str, Any]:
        self.seq += 1
        return {"contract": CONTRACT, "type": kind, "seq": self.seq, "client_time_ms": time.time_ns() // 1_000_000,
                "speed_ratio": self.speed_ratio, **body}

    def _blocked(self) -> str | None:
        if self.estop:
            return "정지 잠금 중 · '잠금 해제'를 누른 뒤 인에이블을 켜세요"
        if not self.enabled:
            return "인에이블 꺼짐 · E 키 또는 '인에이블 켜기'를 누르세요"
        return None

    def act(self, action: str, **params: Any) -> dict[str, Any] | None:
        """조작 하나를 처리한다. 신호를 만들었으면 돌려주고, 막혔으면 None(사유는 message)."""
        if action == "enable":
            self.enabled = bool(params.get("on")) and not self.estop
            self.message = "인에이블 켜짐" if self.enabled else "인에이블 꺼짐"
            return self._signal("ENABLE", on=self.enabled)
        if action == "estop":
            self.estop, self.enabled = True, False
            self.message = "소프트웨어 정지 · 실물 비상정지 아님 · 잠금 해제 필요"
            return self._signal("STOP", reason="estop")
        if action == "reset":
            self.estop = False
            self.message = "잠금 해제 · 인에이블을 다시 켜세요"
            return self._signal("RESET")
        if action == "stop":
            self.message = "정지 요청"
            return self._signal("STOP", reason="user")
        if action == "frame":
            self.frame = "joint" if self.frame == "cart" else "cart"
            self.step_index = min(self.step_index, 2)
            self.message = "좌표계: " + ("직교" if self.frame == "cart" else "관절")
            return None
        if action == "step":
            self.step_index = (self.step_index + 1) % 3
            self.message = f"이동 단위 {self.step:g} " + ("mm" if self.frame == "cart" else "°")
            return None
        if action == "speed":
            self.speed_index = max(0, min(len(SPEED_STEPS) - 1, self.speed_index + int(params.get("sign", 0))))
            self.message = f"속도 {round(self.speed_ratio * 100)} %"
            return self._signal("SPEED")
        blocked = self._blocked()
        if blocked:
            self.message = blocked
            return None
        if action == "jog":
            axis, sign = str(params["axis"]), 1 if float(params.get("sign", 1)) >= 0 else -1
            return self._jog(axis, sign)
        if action == "grip":
            delta = float(params.get("delta", 0)) or (GRIP_STEP_MM * (1 if params.get("open") else -1))
            self.grip_mm = max(0.0, min(GRIP_MAX_MM, self.grip_mm + delta))
            self.message = f"그리퍼 개구폭 {self.grip_mm:.0f} mm"
            return self._signal("GRIPPER", opening_mm=self.grip_mm)
        if action == "home":
            self.cart, self.joint, self.target_seq = dict(HOME_CART), dict(HOME_JOINT), self.target_seq + 1
            self.message = "홈 자세 목표"
            return self._signal("HOME", target=dict(self.cart))
        self.message = f"알 수 없는 조작 {action}"
        return None

    def _jog(self, axis: str, sign: int) -> dict[str, Any] | None:
        table, axes = (self.cart, CART_AXES) if self.frame == "cart" else (self.joint, JOINT_AXES)
        if axis not in axes:
            self.message = f"{axis} 는 현재 좌표계의 축이 아님"
            return None
        delta = sign * self.step
        value = table[axis] + delta
        if axis == "RZ":
            value %= 180.0                    # 장축각은 180° 주기
        elif axis == "Z" and value < 0:
            self.message = "Z 0 mm 아래로 내릴 수 없음"
            return None
        elif axis.startswith("J") and abs(value) > 360:
            self.message = f"{axis} ±360° 한계"
            return None
        table[axis] = round(value, 3)
        self.target_seq += 1
        unit = "mm" if self.frame == "cart" and axis != "RZ" else "deg"
        self.message = f"{axis} {'+' if delta >= 0 else ''}{delta:g} {unit}"
        return self._signal("JOG", frame="work" if self.frame == "cart" else "joint", axis=axis, delta=delta, unit=unit,
                            target=dict(table))

    def snapshot(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "estop": self.estop, "frame": self.frame, "step": self.step,
                "speed_ratio": self.speed_ratio, "grip_mm": self.grip_mm, "cart": dict(self.cart),
                "joint": dict(self.joint), "seq": self.seq, "target_seq": self.target_seq, "message": self.message}
