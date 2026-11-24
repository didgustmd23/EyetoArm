# D 클라이언트 (Windows)

이 폴더는 `dashboard-client`의 GUI 정본이다. 기존 `src/eyetoarm/gui` 진입점은 제거하고 새 폴더만 사용한다. 독립 번들은 번들 루트에서 아래 명령을 실행한다.

번들의 `config/gui/client.json`은 D 기본값만 포함한다. 기본 감사 기록은 번들의 `logs/gui/`에 저장되며 기존 저장소 로그는 그대로 남는다. 현재 실행 경로는 주변 폴더를 찾지 않고 자기 설정만 읽는다. 저장소의 C·D 공용 설정은 그대로 남는다. `--log-dir`의 상대 경로는 실행 디렉터리 기준이다. 번들 TCP protocol과 schema는 공유 원본의 복사본이므로 변경 시 동기화 검증이 필요하다.

## 현재 실행 경로

기존 `eyetoarm` 환경을 활성화한 뒤 `dashboard-client` 루트의 PowerShell에서 실행한다.

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m eyetoarm.gui.window --demo
python -m eyetoarm.gui --env-only
```

저장소 루트의 `pyrightconfig.json`은 에디터 분석용이며, 터미널 실행에서는 위 `PYTHONPATH` 설정이 별도로 필요하다.

`widgets.py`는 실행 진입점이 아닌 화면 부품 모듈이다. 앱은 `python -m eyetoarm.gui.window`로 실행한다. `from eyetoarm.gui import motion, theme` 문법과 네임스페이스 패키지 구조를 경고를 숨기기 위해 바꾸지 않는다.

WSL의 임시 단일 영상 중계는 저장소 루트에서 `python3 dashboard-client/src/eyetoarm/gui/gz_view_relay.py`로 실행한다. ROS 의존성은 WSL에 별도로 필요하며 정식 C→D 3영상 연결과 다르다.

실제 C 준비 조회·3채널 영상 수신은 `python -m eyetoarm.gui.window`가 기본이다. 명령 서버 IP를 영상에도 사용하며 영상 기본 포트는47092다. 별도 영상 서버는 `--view-host`/`--view-port`로 지정한다. C의 ETV1 시제품만 지원하며 정식 계약 확정과는 별개다. 기존 Gazebo 창 캡처는 `--view window`, 임시 EAV1 단일 카메라 중계는 `--view camera`로 선택한다. `--demo`는 소켓을 열지 않는다.

링크 raw/detect/center는 조작·설정과 운영 화면에서 같은 표시 선택을 공유한다. 현재 C는 raw3채널만 송출하므로 detect/center 미수신 시 대기로 남고 raw로 대신하지 않는다. 서버에 종류 변경 요청을 보내거나 주석·측정 결과를 합성하지 않는다.

폴더 교체 이후 3일차 작업에서 관련 `tests/`의 패키지 경로와 현재 GUI 시험을 갱신했다. 외부 설계 문서에는 이전 경로의 역사적 설명이 남아 있다. 테마 선택을 보존하기 위해 QSettings의 기존 `EyeToArms/d_client` 키는 유지한다.

1일차 진단 JSON의 `tool: eyetoarm.d_client` 식별자도 기존 기록과의 호환성을 위해 유지한다. 실행 명령과 Python import 경로는 `eyetoarm.gui`를 사용한다.

## 3일차 TCP·기록

현재 C 서버의 TCP 1.1에 맞춰 준비 조회, 수동 PLANAR MOVE, 명령 ID별 조회 및 CANCEL을 처리한다. C 서버 코드·설정·명세는 수정하지 않았다. 조작·설정에서 X/Y(mm), θ(0~180° 미만)를 직접 입력하고 확인란을 선택한 뒤 MOVE를 누른다. A/B의 보정된 측정값이 아닌 수동 시험 목표이며 서버의 작업 범위·준비 조건을 따라야 한다. 실행 직전 새 GET_STATUS가 준비를 확인한 뒤에만 MOVE를 보낸다. 모의 모드의 이동 버튼은 비활성이다. `--tcp-version 1.1` 실제 모드의 START/Stop 정지(STOP)/RECOVER/E-Stop은 연결되면 누를 수 있으나 보내지 않고 「미전송 · C TCP 1.2 필요」를 표시한다(E-Stop 버튼 글자는 「E-Stop · 미전송」). 실제 모드 기본(TCP 1.2)은 C가 호환 1.2 응답을 보낼 때만 보내며, 이때 수동 MOVE·CANCEL은 비활성이다.

ACK는 수락이며 도달이 아니다. CANCEL도 기존 명령 ID로 요청만 보내고 종료 확인을 기다린다. 재접속 뒤 MOVE를 자동 재전송하지 않는다. UNKNOWN_COMMAND/수락 미확인 뒤 C가 준비됨을 다시 보고하면 사용자가 ‘미확인 기록 보존·새 목표 확인’을 선택해 이전 결과를 미확인으로 보존하고 새 목표를 확정할 수 있다. 이 조작 자체는 송신하지 않는다.

## 4일차 분류 대역 (`--demo` 전용)

`python -m eyetoarm.gui.window --demo`에서 START·STOP·RECOVER를 소켓 없이 시험한다. START는 STARTING→RUNNING, STOP은 STOPPING(요청 접수)→STOPPED(모의 정지 확인)로 진행한다. 조작·설정의 ‘모의 보유 상태 불명’은 시작을 잠그며, 복구 조건 확인란을 선택해야 RECOVER가 허용된다. 복구 후에는 STOPPED로 돌아가며 자동 시작하지 않는다.

‘모의 연결 끊기/재접속’은 내부 상태만 바꾼다. 단절 시 미확인 상태로 조작을 잠그고, 재접속 스냅샷에서 진행 중이던 작업의 결과가 불명이면 RECOVERY_REQUIRED/HOLD_UNKNOWN으로 표시한다. 자동 4초 장면 전환은 제거해 사용자가 시험하는 중에 연결이 임의로 바뀌지 않는다. 명령·상태 사건은 화면/CSV에 ‘모의·실제 전송 없음’으로 기록하며, 기존 모의 준비 조회도 유지한다. 데모 기록은 실제 모드의 감사 JSONL이 아니다.

`sort_demo.py`의 epoch/seq/job_id는 지연·중복·옛 세대 알림을 시험하는 내부 값이다. 실제 C 메시지 형식이나 승인 계약을 뜻하지 않는다. `--tcp-version 1.1` 실제 모드의 START·Stop 정지(STOP)·RECOVER·E-Stop은 누를 수 있으나 미전송이고 기존 TCP 1.1은 변경하지 않는다. 실제 모드 기본(TCP 1.2)은 [명세 부록 A.6·A.6.16](../../../../docs/protocol/TCP_프로토콜_명세서.md) 형식을 쓰며, C가 호환 응답을 보낼 때만 조작이 열린다. 새 운전 모드는 추가하지 않았다. 벨트 표시 단계는 최신 C 정책에 맞춰 ‘이동 중 집기’로 정리했으며, 실제 집기나 벨트 구동 구현을 뜻하지 않는다.

1180×760 등 높이가 작은 창에서는 조작·설정 화면을 세로로 스크롤한다. 아래쪽 수동 목표와 외형 카드를 보려면 스크롤하며, 상단 분류·CANCEL 버튼과 공통 상태띠는 계속 보인다. 수신한 내부 전이 중 스냅샷은 다음 스냅샷을 기다리며 자동 완료를 추정하지 않는다. 타이머 완료는 사용자가 이 대역에서 직접 요청한 모의 동작에만 적용한다.

- D 로그: 기본 `logs/gui/<세션 시각>-<고유 ID>.jsonl`. 첫 사건부터 자동 저장하며 기록 화면에 실제 경로를 표시한다. `raw_b64`/`raw_hex`에 불량 UTF-8·CRLF를 포함한 실제 송수신 바이트를 보존한다.
- C 로그: C 실행 설정의 `command_log_path`에 C가 직접 REGISTER/STATE를 기록한다. D가 C 파일을 쓰거나 초기화하지 않는다. 기본은 C 프로젝트의 `logs/command_log.jsonl`이며 Windows D 폴더의 로그와 같다고 가정하지 않는다.
- 화면/CSV: 전체/TX/RX/EV 중 현재 보이는 최대 500건의 요약을 저장한다. 원문 툴팁은 400자로 제한되므로 전체 원문 증거는 JSONL을 사용한다. CSV는 UTF-8 BOM·CRLF와 수식 접두 방어를 적용하며 행 직렬화 비율을 원형 진행률로 표시한다. 성공은 파일 commit 후에만 표시하고 실패/창 종료 시 임시 출력을 취소한다.
- 반복 억제: 기본 3회 또는 5000ms 동안 같은 준비 상태면 반복 TX/RX 화면 행을 생략한다. server_time_ms와 사유 배열 순서만 바뀐 응답은 새 상태로 세지 않는다. 의미가 같은 비종료 명령 상태도 반복 표시하지 않는다. 오류·거부·종료·상태 변경은 보존한다.
- 연결 감시를 위해 기존 준비 조회 주기는 유지한다. 중복 응답도 신선도와 원문 감사 파일에는 반영한다. 화면 억제는 폴링·C 명령 로그·전체 원문 기록을 끈다는 뜻이 아니다.
- 접속 실패 재시도에서 같은 끊김 사유가 연속되면 화면에는 최초 1회만 표시한다. 사유 변경·연결 성공 뒤 새 끊김·세션 종료 후 재시작은 다시 표시한다. 재시도와 개별 실패의 JSONL 기록은 계속되므로 파일 증가와 화면 행 증가는 별개다. 모의 모드는 접속마다 준비 TX/RX 한 쌍만 표시하고 신선도는 매 조회 갱신한다.
- 저장 오류는 화면에 표시하며 최대 1024건을 메모리에 보류하고 후속 기록 때 재시도한다. 한도를 넘거나 파일 끝 복구에 실패하면 유실 건수/파일 신뢰 오류를 명시한다. 디스크 오류 상태에서 무제한 무손실을 보장하지 않는다.

```powershell
python -m eyetoarm.gui.window --log-dir logs/gui --stable-count 3 --stable-ms 5000
```

위 명령은 실제 연결 모드다. 소켓 없는 외형 확인은 기존 `--demo`를 사용한다. 이미 실행 중인 이전 프로그램은 파일 변경만으로 갱신되지 않는다. 현재 테스트의 기록을 필요한 만큼 확보한 뒤 사용자가 새 버전으로 다시 실행해야 파일 로그가 시작된다.

## 5일차 공정·수량 (`--demo` 전용)

조작·설정에서 START를 눌러 모의 RUNNING이 된 뒤 ‘모의 1주기’를 누른다. 볼트 A·볼트 B·너트의 정상 공정과 도착을 순서대로 재현한다. 운영 대시보드에서 부품 ID·규격·공정 단계, 상자별 수량·격리·오분류/미판정 도착, 실패 건수와 사유를 확인한다. ‘집기’는 이동 중 집기의 줄임 표시이며 도구 설명에 전체 단계를 표시한다.

수량은 모의 도착 확인 사건으로만 증가한다. 해제 요청·운반 완료만으로 세지 않으며 같은 부품 재전달은 한 번만 반영한다. STOP 접수·단절·세대 전환·복구 필요·창 닫기에서는 모의 공정 타이머를 멈춘다. 연결이 끊기면 마지막 수량은 남기되 미확인으로 표시하며, 재접속 때 조작 상태와 별개인 공급자의 집계 스냅샷으로 맞춘다.

공정 입력은 모두 ‘모의 · 실제 아님’과 반영/중복/거부 사유로 화면에 기록한다. 화면 비우기·500행 초과는 수량을 초기화하지 않는다. CSV는 선택된 화면의 최근 최대 500행 요약이며 모의 전체 이력의 영속 JSONL은 만들지 않는다. 실제 모드에는 공정 공급자·모의 1주기 버튼이 없고, 실제 수량은 미수신이다. 분류 명령은 기본 1.1 모드에서 미전송이며, 명시적 1.2 후보 모드에서만 C 호환 응답 뒤 보낸다. 이 모의 집계 형식·상충 결과 정책은 C 계약이 아니다.

Windows에서 D 단위·offscreen/대역 시험을 실행했다. Ubuntu에 임시 복사해 수행한 시험도 있었으나, 사용자 환경 정책 확인 후 임시 복사 5파일을 정확한 해시 확인 뒤 전부 팀 커밋 상태로 복원했다. Ubuntu의 최종 상태는 팀 원격 변경만 가져온 코드이며 미커밋 Windows 구현을 배치한 상태가 아니다. 운영 Windows GUI와 Ubuntu C 서버·ROS/Gazebo 사이의 실제 통신·도착 판정은 미검증이다. Windows C loopback 시험은 단언이 통과해도 기존 `wait_closed()` 경로의 WinError 64 콜백 예외가 출력되는 제한이 있다.

## 6일차 영상·측정 소비 (`--demo` 전용)

기존 실행 명령 `python -m eyetoarm.gui.window --demo`에서 탑뷰·그리퍼·링크의 **D 모의 영상 3칸**을 표시한다. 조작·설정의 링크 `raw 원본`/`detect 검출`/`center 중심점` 버튼으로 모의 링크 종류를 선택하고, 운영 대시보드의 현재 측정과 측정·보정 상세에서 같은 관측의 px/mm/θ를 확인한다. 보조 칸을 누르거나 초점 후 Enter·Space를 누르면 그 영상과 큰 칸 영상의 자리를 바꾼다(겹친 확대 없음). 모의 영상은 합성 그림이며 실제 Gazebo·YOLO·B 중심점 주석 영상이 아니다. 선택은 D 내부 동작이며 C에 요청을 보내지 않는다.

프레임·결과의 run/source/image_seq/세대/설정/보정이 일치하는 값만 표시한다. 무효화·노후·연결 단절은 수치를 비우며 너트의 방향 없음은 0도로 채우지 않는다. B의 mm 출력은 재변환하지 않는다. 이 표시 소비 API는 실제 MOVE 허가가 아니며 모의 모드의 MOVE는 계속 비활성이다. 기존 실제 TCP1.1과 임시 창 캡처/카메라 수신 경로는 유지한다. 이번 작업에서 60Hz 조정은 하지 않았다.

2026-11-17 내부 소비 보완: 명시적인 세대 전환은 이전 영상·수치를 비우고 새 세대 결과를 허용한다. 프레임 없는 오류는 같은 세대의 결과와 detect/center 영상을 차단한다. 작업 범위의 무효화는 해당 주석 영상과 늦은 재전달도 차단하며 raw·다른 채널·더 최신 작업은 보존한다. 이 연산은 D 내부 API·시험에서만 사용하며 실제 A/C 사건을 자동 수신하지 않는다. 같은 원본의 세대 전환으로 최초 수신 시각이나 신선도를 연장하지 않는다.

**정식 C→D 영상 계약은 미확정이다.** 최신 C `display_relay`의 ETV1/47092 raw3 시제품은 D의 기본 수신 경로로 연결하며, `relay_view.py`가 수신을 맡는다. 실제 카메라 송출·원격 PC와의 실사용 검증은 별도다. C의 detect/center·측정 결과·종류 변경 요청은 아직 없으며 `display.py`의 내부 소비 API를 승인된 전송 스키마로 간주하지 않는다. 기존 `--view camera`의 47091/EAV1 단일 영상 경로도 정식 3영상과 다르다.

저장소의 `docs/gui-impl/D_6일차_클라이언트_API.md`와 `docs/design/D담당_협업_요청_초안.md`에 추가 설명이 있다. 이 문서는 독립 번들에 포함되지 않으며 협업 요청은 미발송이다.

## 설치 기록 (2026-11-12, D 개발 PC)

| 항목 | 값 |
|---|---|
| OS | Windows 11 Pro (10.0.26300) |
| Python 배포 | Anaconda3 (`conda` 26.5.3), 환경 이름 `eyetoarm` |
| Python | 3.12.16 |
| PySide6 | PySide6-Essentials 6.12.0 (shiboken6 6.12.0), pip 설치 |
| jsonschema | 4.26.0, pip 설치 |
| Git | 2.55.0.windows.5 |
| 저장소 체크아웃 | `C:\Users\user\WSL\EyetoArm` (Windows 로컬 드라이브) |
| 확인 | PySide6 빈 창 표시 확인 |

운영 GUI는 Windows에서 실행하고, Ubuntu의 `~/EyetoArm`은 D 영상 중계·외형 확인과 C 서버·Gazebo 협업에 사용한다. `/mnt/c/Users/user/WSL/EyetoArm`은 Windows 폴더를 Ubuntu에서 보는 경로이며 `~/EyetoArm`과 별도다. 사용자 지침에 따라 D 구현은 Windows에서 진행하고, Ubuntu에는 팀원이 올린 원격 커밋을 가져온다. Windows의 미커밋 파일을 Ubuntu에 직접 복사·편집하지 않는다. 실행·시험 전 양쪽 HEAD·작업 파일·Python/ROS 설정을 각각 확인한다. Windows conda를 Ubuntu ROS 환경 대신 쓰거나 두 환경의 상대 로그 경로를 같은 파일로 간주하지 않는다.

## 환경 만들기

Anaconda Prompt에서 `dashboard-client` 루트로 이동한 뒤 순서대로 실행한다.

```bat
conda env create -f src/eyetoarm/gui/environment.yml
conda activate eyetoarm
python -c "import PySide6, jsonschema; print(PySide6.__version__)"
```

Anaconda `base`(Python 3.14)와 Windows 기본 `python`(3.15)은 쓰지 않는다. 항상 `conda activate eyetoarm` 뒤에 실행한다.

## 팀 공통 환경 확인 (계획서 §2.5, 2026-11-12, 같은 PC)

| 항목 | 계획서 기준 | 이 PC |
|---|---|---|
| WSL | WSL2 Ubuntu 24.04 LTS | WSL 3.0.5.0, Ubuntu 24.04.5 LTS, 커널 6.18.58.1, WSLg 1.0.79 |
| ROS 2 | Jazzy, `rosdep`, `colcon` | `ros-jazzy-desktop` 0.11.0, `python3-rosdep` 0.27.0(초기화됨), `colcon-common-extensions` 0.3.0 |
| 시뮬레이션 | Gazebo Harmonic, `ur_simulation_gz`, MoveIt 2, `ros2_control`, `gz_ros2_control`, RViz2 | gz-sim 8.15.0(`ros-jazzy-gz-sim-vendor`), `ur-simulation-gz` 2.5.0, `moveit` 2.12.4, `ros2-control` 4.48.1, `ros2-controllers` 4.42.1, `gz-ros2-control` 1.2.20, `rviz2` 14.1.24 |
| 2지 그리퍼 | Robotiq 2F-85, `parallel_gripper_action_controller` | `robotiq-description` 0.0.1, `parallel-gripper-controller` 4.42.1 |
| 영상 (WSL) | Ubuntu 기본 Python, OpenCV(apt), NumPy, `cv_bridge` | Python 3.12.3, OpenCV 4.6.0, NumPy 1.26.4, `cv-bridge` 4.1.0, `ros-gz-bridge` 1.0.24 |
| 렌더링 | WSLg GPU 가속 확인 | `GALLIUM_DRIVER=d3d12`(`~/.bashrc`)일 때 D3D12 (NVIDIA GeForce RTX 5070 Ti). 이 변수가 없는 셸에서는 llvmpipe(소프트웨어) |
| 편집 | VS Code WSL 연결 | VS Code 설치됨 |
| 하드웨어 | 팀 PC 기록 | Intel Core Ultra 7 270K Plus, RAM 64 GB(WSL 할당 31 GB, 24코어), NVIDIA GeForce RTX 5070 Ti |

## Windows → WSL C 서버 접속 (계획서 §2.5 설치 순서 1)

WSL NAT 모드(기본 localhost 전달)에서 C 서버(`robot_server_package`, main `0d8a9a3`, 바인딩 `127.0.0.1:47090`)를 WSL에서 실행하고, Windows의 `eyetoarm` Python으로 `GET_STATUS` 한 줄을 보냈다.

| Windows에서 접속한 주소 | 결과 |
|---|---|
| `127.0.0.1:47090` | 성공. `STATUS`(`robot_ready:false`, `RECOVERING`) 수신 |
| `localhost:47090` | 성공. 다만 응답까지 약 2초 지연(IPv6 먼저 시도로 추정, 미확인) |
| WSL IP(`172.30.x.x`):47090 | 거부(`WinError 10061`). 서버가 `127.0.0.1`에만 바인딩하므로 예상대로 |

D 클라이언트 기본 주소는 `127.0.0.1`로 둔다(`localhost` 대신). mirrored 모드는 시험하지 않았다. WSL 재시작·절전 뒤 재접속 동기화는 이후 클라이언트 단계에서 확인한다(C 상세설계 §16.11).
