# D 클라이언트 구현 상태 — 대시보드와 수동 펜던트 (2026-11-11)

D 담당은 클라이언트다. 신호를 작성해 보내고, C 서버가 보낸 신호를 받아 기록하고, 화면에 움직임으로 보여 준다.
이 구현에는 **C 서버·로봇 실행·Gazebo 코드가 없다.** C의 `tests/tcp/`와 `robot_ws/`는 수정하지 않았다.

## 구성

| 위치 | 내용 |
|---|---|
| `src/eyetoarm/tcp/protocol.py` | TCP 1.1 신호 작성(MOVE·GET_STATUS·CANCEL → JSON 한 줄 + LF)과 수신 STATUS 검사(정의서 §2.3, 저장소 스키마) |
| `src/eyetoarm/tcp/client.py` | 정의서 §8 클라이언트 규칙: 준비 조회, MOVE 송신·ACK 타이머·같은 바이트 재전송, 재접속 뒤 조회만, 취소, 상태 적용 |
| `src/eyetoarm/tcp/event_log.py` | 송수신·로컬 사건 기록(JSONL 파일 `logs/client/events-*.jsonl`, 화면 목록, CSV) |
| `src/eyetoarm/pendant/model.py` | 펜던트 조작 규칙과 조그 신호 작성 |
| `src/eyetoarm/gui/session.py` | 화면과 무관한 세션. 조작 → 신호 작성·전송·기록 |
| `src/eyetoarm/gui/bridge.py` | 통신 스레드(asyncio) ↔ 화면(Qt 시그널). 화면 갱신은 GUI 스레드에서만 한다 |
| `src/eyetoarm/gui/dashboard.py` | 대시보드 창(PySide6). 6개 화면 |
| `src/eyetoarm/gui/pendant_window.py` | 수동 펜던트 창(PySide6) |
| `src/eyetoarm/gui/twin_view.py`, `motion.py`, `theme.py` | 셀 그림, 움직임 부품, 색·글꼴 |
| `config/gui/client.json` | 클라이언트 설정(정의서 §9 초기값) |
| `tests/unit/test_d_protocol.py`, `tests/integration/test_d_client.py`, `test_d_gui.py` | 시험 40건 |

## 실행

```bash
cd ~/EyetoArm && PYTHONPATH=src ~/.venvs/eyetoarm-tcp/bin/python -m eyetoarm.gui --pendant
```

PySide6 앱이다(설계 문서 §A.2와 같음). `--pendant`는 펜던트 창도 함께 열고, `--connect`는 시작하면서 C 서버에 접속을 시도한다. 통신·규칙 모듈은 표준 라이브러리만 쓴다(`jsonschema`가 있으면 저장소 스키마로 수신 검사).

## 신호가 작성·전송·기록되는 방식

1. **작성.** 화면 조작 → `protocol.build_*`가 JSON 객체 1개 + LF 1개의 바이트를 만든다. MOVE는 새 UUID v4를 받고, 만든 바이트를 보관한다.
2. **전송.** TCP 연결이 있고 준비 조회가 `robot_ready = true`일 때만 MOVE를 소켓에 쓴다. 연결이 없거나 준비가 안 됐으면 **작성만 하고 보내지 않으며** 사유와 함께 `미전송`으로 기록한다. ACK가 없으면 같은 바이트를 최대 2회 재전송하고, 그래도 없으면 "수락 미확인"으로 둔다.
3. **기록.** 모든 송신 시도(TX)·수신(RX)·로컬 사건을 원문 그대로 한 줄씩 남긴다. `send_state`는 `sent`(소켓에 씀)·`unsent`(보내지 않음)이며 수락·도달과는 별개 사건이다. 1초마다 반복되는 준비 조회와 내용이 같은 응답은 파일에만 쓰고 화면 목록에서는 건수만 보여 준다.
4. **수신.** 받은 줄을 검사해 통과한 STATUS만 상태에 반영한다. 위반한 줄은 기록만 한다. 명령 상태는 `command_id`로 맞추고 앞 단계로 되돌아가는 응답은 기록만 한다.

## 펜던트

- 외형은 Hi5a 티치 펜던트를 참고했다(넓은 짙은 회색 몸체, 왼쪽 화면, 오른쪽 조그 키, 오른쪽 위 정지 버튼). 상표·로고·화면 그림은 쓰지 않았다.
- TCP 1.1로 실제 나가는 것은 **START = 수동 목표 MOVE**, **STOP·정지 버튼 = CANCEL** 두 가지다.
- 조그·그리퍼·속도는 TCP 1.1에 없는 신호다. `pendant-draft/0.1`이라는 **D 제안 형식**으로 작성해 기록만 하고 보내지 않는다. 전송 경로는 C와 합의가 필요하다. TCP 1.1 포트로는 보내지 않는다.
- 수동 목표 MOVE는 `source_id = pendant_manual`, `calibration_id = manual_test`를 쓴다. 비전 근거가 아닌 시험용이며, 받아들일지는 C가 정한다.
- 키: W/S·A/D·↑/↓·Q/X 조그, 숫자패드 8/2 그리퍼, −/+ 속도, Tab 좌표계, R 이동 단위, E 인에이블, Enter START, Space STOP, H 도움말.

## 움직임

- 셀 그림은 펜던트·수동 목표가 만든 **명령 목표**를 일정한 속도의 직선 이동으로 그린다. 실제 로봇 상태가 아니며 화면에 그렇게 적었다. Gazebo 상태 스트림이 연결되면 같은 그리기 함수에 받은 값을 넣는다.
- 화면 부품: 스낵바(아래에서 slide), 토스트(fade), 팝업(배경 어둡게 + 90% → 100%), 바텀시트(아래에서 slide), 체크박스(선이 그려짐), 로딩(linear 회전), 제목·부제(차례로 등장), 기록 표시 버튼(spring). 들어올 때 ease-out, 나갈 때 ease-in, 0.2~0.4초.

## 확인한 것

- 시험 40건 통과(단위 13, 통신 통합 16, 화면 11). 통신 통합 시험의 상대는 시험이 정한 줄을 돌려주는 대역이며 C 서버 구현이 아니다. 화면 시험은 화면 없이(offscreen) 창을 띄워 키·버튼 조작부터 기록·표시까지 확인한다.
- C의 시험 서버 `tests/tcp/tcp_server.py`를 **수정 없이** 실행해 연결: `GET_STATUS` 송신 → `STATUS`(`robot_ready false`, `RECOVERING`) 수신·기록·표시, MOVE는 "로봇 준비 안 됨: RECOVERING"으로 미전송 처리.
- offscreen 화면 캡처로 대시보드·펜던트·기록 상세 창·도움말 팝업 표시 확인. WSLg 실제 창에서는 띄워 보지 않았다.

## 확인하지 못한 것 · 남은 것

- 실제 C 서버의 MOVE 수락~도달 흐름(ACK·PLANNING·MOVING·REACHED)은 대역으로만 확인했다.
- 분류·사람 제어 화면은 자리만 있다. C의 실행 상태와 `ext-cmd/0.1` 계약이 필요하다.
- 영상·검출 표시(GUI-01·02·04)는 A·B 입력이 연결되기 전이라 빈 상태다.
- 연결 해제 때 "OS로 넘어간 바이트" 구분(정의서 §8.5)은 하지 않는다. 대기열 없이 바로 쓰므로 보낸 것은 모두 "결과 미확인"으로 본다.
- 처음에는 화면을 웹으로 만들었다가 사용자 결정(2026-11-11)으로 PySide6로 옮겼다. 웹 파일은 지웠다.
- TCP는 Qt 소켓이 아니라 asyncio를 별도 스레드에서 쓴다. 정의서 §8.5의 Qt 버퍼 구분과는 구현이 다르다.
- 2026-11-10 19:25에 이 저장소의 미추적 파일이 지워진 일이 있었다. 원인 미확인. 백업은 `C:\Users\user\WSL\eyetoarm-d-backup\`에 둔다.
