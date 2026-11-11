"""실행: PYTHONPATH=src python -m eyetoarm.gui [--tcp-host 127.0.0.1] [--tcp-port 47090] [--connect] [--pendant]

PySide6 가 설치된 파이썬으로 실행한다(예: ~/.venvs/eyetoarm-tcp/bin/python).
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from eyetoarm.tcp.client import ClientConfig

from .bridge import Bridge
from .dashboard import Dashboard

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser(prog="eyetoarm.gui", description="EyetoArm D 클라이언트: 대시보드와 수동 펜던트")
    parser.add_argument("--config", default=str(ROOT / "config" / "gui" / "client.json"))
    parser.add_argument("--tcp-host")
    parser.add_argument("--tcp-port", type=int)
    parser.add_argument("--connect", action="store_true", help="시작하면서 TCP 1.1 서버에 접속을 시도한다")
    parser.add_argument("--pendant", action="store_true", help="시작하면서 펜던트 창도 연다")
    parser.add_argument("--log", help="기록 파일(JSONL). 기본 logs/client/events-<시각>.jsonl")
    args = parser.parse_args()

    settings = json.loads(Path(args.config).read_text(encoding="utf-8")) if Path(args.config).is_file() else {}
    config = ClientConfig(**settings.get("tcp", {}))
    if args.tcp_host:
        config.host = args.tcp_host
    if args.tcp_port:
        config.port = args.tcp_port
    log_path = Path(args.log) if args.log else ROOT / "logs" / "client" / f"events-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"

    app = QApplication(sys.argv[:1])
    bridge = Bridge(config, log_path)
    bridge.start()
    window = Dashboard(bridge)
    window.show()
    if args.pendant:
        window.open_pendant()
    if args.connect:
        bridge.call("connect")

    # Ctrl+C·종료 신호: Qt 이벤트 루프는 파이썬 신호 처리기를 스스로 돌리지 않는다.
    # 짧은 타이머로 파이썬에 차례를 주고, 신호가 오면 창을 닫아 정상 종료한다(연결 정리 포함).
    def on_signal(*_args) -> None:
        print("종료 신호를 받아 창을 닫습니다", flush=True)
        window.close()
        app.quit()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    wake = QTimer()
    wake.timeout.connect(lambda: None)
    wake.start(200)
    code = app.exec()
    wake.stop()
    bridge.stop()
    # 창과 연결 객체를 QApplication 보다 먼저, 정해진 순서로 지운다(종료 때 정리 순서 문제로 죽지 않게)
    for thing in (window.pendant_window, window, bridge):
        if thing is not None:
            thing.deleteLater()
    del window, bridge
    app.sendPostedEvents(None, 0)
    app.processEvents()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
