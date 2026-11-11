# TCP 통신 시험 안내

이 폴더의 `tcp_server.py`, `protocol.py`, `test.py`를 이용해 로컬에서 TCP 요청·응답을 확인하는 방법입니다.

> 현재 실행 확인용 코드입니다. `GET_STATUS`만 처리하며 ROS 2나 UR5e와 연결하지 않습니다. 서버 응답의 `robot_ready: false`, `not_ready_reasons: ["RECOVERING"]`은 통신 시험 코드가 실제 로봇 준비 상태를 아직 확인하지 않는다는 뜻입니다.

## 파일 구성

```text
tests/tcp/
├── README.md
├── protocol.py     # 요청을 해석하고 STATUS 응답을 만듦
├── tcp_server.py   # TCP 연결을 받아 요청을 전달하고 응답을 보냄
└── test.py         # GET_STATUS를 보내고 응답을 출력함
```

`tcp_server.py`와 `protocol.py`는 같은 폴더에 있어야 합니다. 아래 명령은 저장소 체크아웃의 `tests/tcp/` 폴더에서 실행합니다(예: 저장소가 `~/EyetoArm`이면 `~/EyetoArm/tests/tcp`). 다른 폴더로 복사해 쓰는 경우 세 파일을 함께 복사하고 그 폴더에서 실행합니다.

## 1. 서버 실행하기

첫 번째 Ubuntu 터미널에서 다음을 실행합니다.

```bash
cd ~/EyetoArm/tests/tcp   # 저장소 위치에 맞게 바꿈
python3 tcp_server.py
```

아래처럼 표시되면 서버가 `127.0.0.1:47090`에서 접속을 기다리는 중입니다.

```text
TCP 서버 대기 중: 127.0.0.1:47090
```

서버를 실행한 터미널은 닫지 말고 켜 둡니다.

## 2. 시험 요청 보내기

새 Ubuntu 터미널을 열고 다음을 실행합니다.

```bash
cd ~/EyetoArm/tests/tcp   # 저장소 위치에 맞게 바꿈
python3 test.py
```

시험 프로그램은 다음 요청을 서버에 보냅니다.

```json
{"version":"1.1","type":"GET_STATUS"}
```

정상적으로 통신되면 두 번째 터미널에 다음과 비슷한 JSON 응답이 출력됩니다.

```json
{"version":"1.1","type":"STATUS","robot_ready":false,"not_ready_reasons":["RECOVERING"],"active_command_id":null,"server_time_ms":1790000000000}
```

`server_time_ms` 숫자는 실행 시각에 따라 달라집니다. `type`이 `STATUS`이고 `RECOVERING`이 출력되면 요청이 서버에 도착하고 응답이 돌아온 것입니다. 이것은 TCP 통신 시험 성공을 뜻하며, 로봇 이동 가능 여부를 시험한 것은 아닙니다.

첫 번째 터미널에는 클라이언트가 접속했다는 로그가 표시됩니다.

```text
클라이언트 접속: ('127.0.0.1', 임시포트번호)
```

## 3. 서버 종료하기

첫 번째 터미널을 선택하고 `Ctrl+C`를 누릅니다.

## 문제가 생겼을 때

### `Connection refused`

서버에 연결할 수 없다는 뜻입니다.

- 첫 번째 터미널에서 `python3 tcp_server.py`가 계속 실행 중인지 확인합니다.
- 서버와 시험 프로그램이 같은 포트 `47090`을 사용하는지 확인합니다.
- 다른 프로그램이 이미 포트를 사용 중이면 서버 시작 단계에서 `Address already in use`가 표시될 수 있습니다. 기존 서버를 종료한 뒤 다시 실행하세요.

### `ModuleNotFoundError: No module named 'protocol'`

`protocol.py`가 `tcp_server.py`와 같은 폴더에 있는지 확인하고, `tests/tcp/` 폴더(또는 세 파일을 함께 복사한 폴더)에서 서버를 실행하세요.

### 응답 대기 시간 초과

서버 터미널에 클라이언트 접속 로그가 찍혔는지 봅니다. 로그가 없다면 주소·포트 또는 서버 실행 상태를 확인하고, 로그가 있다면 서버 터미널에 오류가 출력됐는지 확인합니다.

## 시험 범위

현재 `test.py`는 `GET_STATUS` 요청 한 건을 보내 응답 한 줄을 읽습니다. `MOVE`, `CANCEL`, 잘못된 요청, 여러 요청 동시 처리, ROS 2 연결 및 실제 로봇 동작은 시험하지 않습니다. 프로젝트 통합 전에 정식 TCP 명세서와 일치하도록 서버의 요청 검증·오류 응답도 보완해야 합니다.
