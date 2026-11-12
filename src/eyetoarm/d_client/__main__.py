"""D 1일차 진단 CLI.

  PYTHONPATH=src python -m eyetoarm.d_client --env-only
  PYTHONPATH=src python -m eyetoarm.d_client --host 127.0.0.1 --port 47090 --timeout 3 [--json]

무ID GET_STATUS 한 줄만 보낸다. 동작 명령은 보내지 않는다.
종료 코드: 0 유효 STATUS 수신(robot_ready 값과 무관, 출력의 [준비] 줄로 구분), 1 접속 실패,
2 프로토콜 실패, 4 환경 불일치(--env-only 또는 --check-env. --check-env 는 불일치여도 접속 진단을 계속
하고 결과를 함께 출력하되 종료 코드는 4), 5 잘못된 인자. JSON 의 exit_code 는 실제 종료 코드와 같다.
--timeout 은 접속+송신+수신 총 기한(초). 기본 3 = client.json connect_timeout_ms 2000 + readiness_timeout_ms 1000.
--host 는 숫자 IP 리터럴(IPv4·IPv6)만 받는다. 이름(localhost 등)은 거부(종료 5, 접속 시도 없음).
결과는 stdout 에만 쓴다(파일 기록은 호출 측 담당).
"""

from __future__ import annotations

import argparse
import json
import sys

if not __package__:
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eyetoarm.d_client import diag


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m eyetoarm.d_client", description="D 1일차 환경·C STATUS 접속 진단")
    parser.add_argument("--host", default="127.0.0.1", help="C 서버 숫자 IP (기본 127.0.0.1). 이름(localhost 등)은 거부")
    parser.add_argument("--port", type=int, default=47090)
    parser.add_argument("--timeout", type=float, default=diag.DEFAULT_TIMEOUT_S,
                        help="접속+송신+수신 총 기한(초), 0 초과 60 이하. 기본 3 = connect 2 + readiness 1")
    parser.add_argument("--env-only", action="store_true", help="환경만 점검, 접속하지 않음")
    parser.add_argument("--check-env", action="store_true", help="접속 전에 환경도 점검")
    parser.add_argument("--json", action="store_true", help="결과를 JSON 한 줄로 출력")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exit_:
        return 0 if exit_.code == 0 else diag.EXIT_USAGE
    try:
        diag.validate_target(args.host, args.port, args.timeout)
    except ValueError as error:
        print(error, file=sys.stderr)
        return diag.EXIT_USAGE

    env = diag.check_environment() if (args.env_only or args.check_env) else None
    result = None
    if not args.env_only:
        result = diag.probe(args.host, args.port, args.timeout)

    record = diag.to_record(env, result)
    print(json.dumps(record, ensure_ascii=False) if args.json else diag.format_human(env, result))
    return record["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
