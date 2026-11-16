# D 6일차 — A 시험 도구용 클라이언트 API

2026-11-16 · 기준 `dbd2c0c`(이전 `fa56c70`) · 기존 TCP 1.1 API의 인계용 기준. 이 문서는 C 표시 프로토콜이나 A 환경 호환 승인이 아니다. API 변경 시 A에 변경 내용을 먼저 협의한다. Windows conda 환경과 A의 WSL 실행 환경은 별도다.

## 바로 실행하는 준비 조회

설치된 `eyetoarm` 환경의 Python을 사용하고 저장소 루트에서 `PYTHONPATH=src`를 설정한다. Windows PowerShell 예:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
& "C:\Users\user\anaconda3\envs\eyetoarm\python.exe" -m eyetoarm.gui --env-only --json
& "C:\Users\user\anaconda3\envs\eyetoarm\python.exe" -m eyetoarm.gui --host 127.0.0.1 --port 47090 --timeout 3 --json
```

두 번째 명령은 실제 주소로 GET_STATUS 한 번을 보내므로 C가 준비한 시험 서버에서 사용한다. MOVE/CANCEL/분류 명령은 보내지 않는다. 첫 번째 환경 검사도 서버에 접속하지 않는다. `robot_ready:false`라도 유효 STATUS를 수신하면 접속 진단은 성공이다. 종료 코드는 0 유효 STATUS, 1 접속 실패, 2 프로토콜 실패, 4 환경 불일치, 5 인자 오류다. 숫자 IP만 허용하며 localhost 이름은 거부한다. 정상 인자로 실행한 결과는 stdout JSON 한 건이며 자동 파일 저장은 하지 않는다. 인자 오류는 stderr와 종료 코드 5로 보고하며 JSON을 보장하지 않는다.

WSL에서는 별도 확인된 Python 환경으로 `PYTHONPATH=src python3 -m eyetoarm.gui --host <C의 숫자IP> --port 47090 --json`을 사용한다. Windows 절대 Python 경로를 WSL에 적용하지 않는다. 실제 WSL 실행·주소·의존성 호환은 공동 시험 항목이다.

## 도구에 포함할 지속 연결 API

구현: [client.py](../../src/eyetoarm/tcp/client.py), [event_log.py](../../src/eyetoarm/tcp/event_log.py), [protocol.py](../../src/eyetoarm/tcp/protocol.py). 이 `TcpClient`는 Qt GUI의 `StatusLink`와 별도 구현이다. GUI의 연결을 재사용하거나 GUI와 동시에 같은 명령을 소유하는 구조로 가정하지 않는다.

| API | 호출·의미 |
|---|---|
| `ClientConfig(host=..., port=...)` | 기본 127.0.0.1:47090. timeout/poll/reconnect는 client.py의 현재 기본값 사용 |
| `EventLog(path_or_none, keep=500)` | Path이면 JSONL, None이면 최근 메모리 목록만. write_error 검사 |
| `TcpClient(config, log, notify)` | notify는 인자 없는 동기 콜백. 오래 걸리는 I/O를 콜백에서 실행하지 않음 |
| `client.connect()` | 동기 함수지만 실행 중 asyncio 루프 안에서 호출. 연결 시도 시작이며 연결 완료가 아님 |
| `client.connected` | TCP 연결 여부. 준비 상태·실행 허용과 구분 |
| `await client.query_ready()` | 준비 응답 dict 또는 None. 초기 연결/정기 준비 조회가 자체 수행되므로 중첩 조회 금지 |
| `client.snapshot()` | 현재 상태 사전. 독립 불변 저장본은 아니므로 보관할 때 deepcopy/JSON 직렬화 |
| `await client.execute(target)` | `MoveTarget` 수동 명령. 현재 단계 문자열 반환은 로봇 도달 증거가 아님. 이번 예제에서는 호출하지 않음 |
| `await client.cancel()` | 현재 명령 취소 요청. 분류 STOP/실제 정지 완료와 구분 |
| `await client.disconnect()` | 도구 종료 시 finally에서 호출. 로봇 정지 명령이 아님 |

아래는 새 통신 래퍼 없이 초기 자동 GET_STATUS의 유효 응답을 기다리는 최소 예다. 독립 `asyncio.run()` 수명으로 사용하며 종료 시 asyncio.run이 남은 자체 task를 정리한다. 장기 공유 이벤트 루프에 넣을 때 task 종료·앱 소유권은 별도 확인한다.

```python
import asyncio
import json
from eyetoarm.tcp.client import ClientConfig, TcpClient
from eyetoarm.tcp.event_log import EventLog

async def main():
    received = asyncio.Event()
    client = None

    def changed():
        if client.robot_ready is not None:
            received.set()

    log = EventLog(None)  # 파일 보관 시 별도 Path 지정, 기존 원장과 섞지 않음
    client = TcpClient(ClientConfig(host="127.0.0.1", port=47090), log, changed)
    try:
        client.connect()
        await asyncio.wait_for(received.wait(), timeout=5)
        print(json.dumps(client.snapshot(), ensure_ascii=False))
    finally:
        await client.disconnect()

asyncio.run(main())
```

접속 실패/시간 초과는 예외로 드러내며 성공으로 처리하지 않는다. 목표 위치/시각/보정 정보를 예제에서 합성하거나 실제 MOVE를 자동 송신하지 않는다. A의 통신 100회·장애 시험 자체는 A 책임이며 본 문서가 그 결과를 대신하지 않는다.

## 로그·사건과 자료형 경계

`EventLog` 한 줄은 `seq`, `wall_ms`, `mono_ms`, `dir`(TX/RX/LOCAL), `kind`, `channel`, `command_id`, `send_state`, `note`, `raw`, `bytes` 등을 갖는다. `sent`는 소켓에 썼다는 뜻이고 C 수락·도달이 아니다. 정기 준비 조회는 파일에 기록되지만 최근 rows/listeners에서는 제외된다. notify는 상태 변경 알림이고 모든 원시 사건을 받는 스트림이 아니다.

`raw`는 바이트를 UTF-8 replacement로 디코딩하고 끝 LF를 제거한 **텍스트**다. 완전한 원시 바이트 보존이 아니다. Qt GUI의 `SessionLog`에 있는 raw_b64/raw_hex와 혼동하지 않는다. wall_ms와 mono_ms, ROS SIM_TIME을 서로 빼서 지연으로 계산하지 않는다. 기록 오류는 `log.write_error`로 확인한다.

B의 현재 `to_output_dict()`는 `center_w_mm`, `length_mm`, `width_mm`를 이미 mm로 준다. D는 다시 1000을 곱하지 않는다. `theta_valid:false`와 `theta_deg:null`은 특히 너트의 방향 없음이며 0도로 채우지 않는다. B 결과의 source/image/calibration만으로 run/세대/처리 순번을 추정하지 않는다. 실제 표시 프레임과의 상관 필드는 A·C와 합의할 항목이다.

## D 내부 표시 소비 API — 정식 전송 규격 아님

구현: [display.py](../../src/eyetoarm/gui/display.py). 현재 `--demo`에서만 연결된다. 호출은 GUI 스레드에서 수행하며, 향후 별도 수신 worker가 필요하면 Qt queued signal 등 GUI 스레드 경계까지 함께 검증해야 한다. 이 API 이름과 인자는 C에 그대로 전송할 JSON 스키마가 아니다.

| API | D 내부 시험 의미 |
|---|---|
| `begin_session(run_id, link_source_id)` | 공급자가 새 실행을 명시. D가 누락된 run을 생성하지 않음. 동일 실행의 재접속과 구분 |
| `set_connected(bool)` | 단절 시 영상·현재 결과 제거. 같은 실행의 순서/무효화 기록은 보존 |
| `ingest_frame(channel, QImage, **envelope)` | channel은 link/gripper/fixed. link variant는 raw/detect/center. 공급자의 run/source/image_seq/capture_generation/generation/detector_config_id/job_seq/calibration_id를 보존하고 영상을 소유 복사 |
| `ingest_result(dict, generation=..., order=...)` | D 내부 예제 사전의 정확한 식별자로 매칭. center_px는 제공될 때만, center_w_mm는 이미 mm. 수치/유효 flag 검사, 수락 시 복사 |
| `invalidate(source_id, run_id=..., detector_config_id=..., generation=..., order=..., reason=..., job_seq=None)` | 같은 실행/소스의 적용 범위만 무효화. job이 있으면 해당 job까지 부활 금지. 프레임 없는 소스 오류는 job을 합성하지 않음 |
| `set_mode(raw_or_detect_or_center)` | 모의 영상 종류 선택. C 송신/접수 확인 없음 |
| `frame(channel)`, `matched_result()`, `has_matched_position()` | UI 표시용 조회. 마지막 것은 MOVE 허가가 아님. 반환값은 소비 모델 소유이므로 호출자가 수정하지 않음 |

내부 모의 신선도 기본값은 로컬 원본 영상 수신 뒤 3초이며 C의 확정 TTL이 아니다. 동일 원본의 중복·재처리·종류 변경으로 수명을 늘리지 않는다. 새 run과 세대/무효화 순서는 호출자가 명시한다. 프레임 없는 무효화는 같은 소스의 새 처리 세대 또는 명시 새 세션 전까지 결과를 보수적으로 차단한다. 이 정책은 D 내부 시험 한계이며 실제 입력 복구 정책으로 상대에게 강제하지 않는다. 정식 adapter에는 생산 시각·전송 지연·재접속 재동기화 등 별도 합의가 필요하다.

**지원하지 않는 대응(현재 모의 정책 한계):** 이 API는 A `CandidateInvalidationEvent`를 직접 받는 adapter가 아니다. 특히 `GENERATION_CHANGED`(프레임 없음)를 `invalidate(source, generation=새 세대, job_seq=None)`로 그대로 옮기면 안 된다. 현재 구현은 이 호출 뒤 같은 세대의 새 결과까지 차단하므로(다음 세대나 새 세션까지), 이전 세대를 정리하고 새 세대 결과를 허용해야 하는 세대 변경 의미와 다르다. 반대로 `MODEL_LOAD_FAILED` 같은 프레임 없는 오류도 새 세대 번호로 올 수 있으므로 세대 번호 크기로 둘을 추정하지 않으며 런타임도 그렇게 바꾸지 않았다. 실제 adapter는 사건 `kind`·`cause`와 A의 복구 조건을 구분해 매핑해야 하며, 그 규칙은 A·C 합의 항목이다. 실제 adapter는 없다.

Claude 재검토 보완: B 무효 결과의 좌표(계산 불가 시 `null` 원소, 판정 무효면 유한값 가능)는 정상적인 무효 입력으로 소비하여 이전 측정을 비우고 사유를 표시한다. 유효 결과의 수치 목록에는 null·NaN·무한대·bool을 허용하지 않는다. 내부 모형에서 같은 source/처리 generation의 detector_config_id는 고정하며, 설정 변경은 새 처리 generation으로 명시한다. 재접속이나 현재 결과 제거만으로 이 설정·순서 기록을 초기화하지 않는다. B 사전에는 전체 상관 필드가 없으므로 raw B 사전만 넣어 호환되었다고 판단하지 않는다.

## 미완료 관문

정식 C→D 표시 전송 규격 승인, A 실행 환경에서의 재사용 검증, 실제 서버 준비·이동·분류, 영상 3개·결과·무효화 종단 연결은 별도다. 이번 D 내부 모의 소비 API를 실제 C와 호환된 프로토콜로 사용하지 않는다.
