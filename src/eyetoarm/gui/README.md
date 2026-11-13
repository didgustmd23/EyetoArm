# D 클라이언트 (Windows)

D 운영 GUI와 TCP 클라이언트를 Windows에서 새로 구현하는 폴더다. 계획서 v1.6 §2.1(D GUI는 Windows, A·B·C·로봇은 WSL)과 §2.5(Windows Python·PySide6 환경과 저장소 체크아웃 위치는 D가 기록)를 따른다.

2026-11-13 사용자 요청에 따라 최신 `d_client/`를 이 `gui/`로 옮기고 기존 구형 `gui/` 구현을 삭제했다. `src/eyetoarm/tcp/`와 다른 폴더는 변경하지 않는다.

## 현재 실행 경로

기존 `eyetoarm` 환경을 활성화한 뒤 프로젝트 루트의 PowerShell에서 실행한다.

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m eyetoarm.gui.window --demo
python -m eyetoarm.gui --env-only
```

저장소 루트의 `pyrightconfig.json`은 Pyright/Pylance가 `src` 아래의 `eyetoarm` 패키지를 찾도록 한다. 개인 Python 설치 경로는 공유 설정에 넣지 않으며, VS Code에서는 이 프로젝트의 `eyetoarm` 환경을 선택한다. 이 설정은 에디터 분석용이므로 터미널 실행에서는 위 `PYTHONPATH` 설정이 별도로 필요하다. [Pyright 공식 설정 문서](https://github.com/microsoft/pyright/blob/main/docs/configuration.md#environment-options)

`widgets.py`는 실행 진입점이 아닌 화면 부품 모듈이다. 앱은 `python -m eyetoarm.gui.window`로 실행한다. `from eyetoarm.gui import motion, theme` 문법과 네임스페이스 패키지 구조를 경고를 숨기기 위해 바꾸지 않는다.

실제 C 준비 조회·Gazebo 창 캡처는 `python -m eyetoarm.gui.window`, 임시 카메라 중계 수신은 `python -m eyetoarm.gui.window --view camera`로 실행한다. `--demo`는 소켓을 열지 않는다.

폴더 교체 이후 3일차 작업에서 관련 `tests/`의 패키지 경로와 현재 GUI 시험을 갱신했다. 외부 설계 문서에는 이전 경로의 역사적 설명이 남아 있다. 테마 선택을 보존하기 위해 QSettings의 기존 `EyeToArms/d_client` 키는 유지한다.

1일차 진단 JSON의 `tool: eyetoarm.d_client` 식별자도 기존 기록과의 호환성을 위해 유지한다. 실행 명령과 Python import 경로는 `eyetoarm.gui`를 사용한다.

## 3일차 TCP·기록

현재 C 서버의 TCP 1.1에 맞춰 준비 조회, 수동 PLANAR MOVE, 명령 ID별 조회 및 CANCEL을 처리한다. C 서버 코드·설정·명세는 수정하지 않았다. 조작·설정에서 X/Y(mm), θ(0~180° 미만)를 직접 입력하고 확인란을 선택한 뒤 MOVE를 누른다. A/B의 보정된 측정값이 아닌 수동 시험 목표이며 서버의 작업 범위·준비 조건을 따라야 한다. 실행 직전 새 GET_STATUS가 준비를 확인한 뒤에만 MOVE를 보낸다. 모의 모드의 이동 버튼과 미승인 START/STOP/RECOVER는 비활성이다.

ACK는 수락이며 도달이 아니다. CANCEL도 기존 명령 ID로 요청만 보내고 종료 확인을 기다린다. 재접속 뒤 MOVE를 자동 재전송하지 않는다. UNKNOWN_COMMAND/수락 미확인 뒤 C가 준비됨을 다시 보고하면 사용자가 ‘미확인 기록 보존·새 목표 확인’을 선택해 이전 결과를 미확인으로 보존하고 새 목표를 확정할 수 있다. 이 조작 자체는 송신하지 않는다.

- D 로그: 기본 `logs/gui/<세션 시각>-<고유 ID>.jsonl`. 첫 사건부터 자동 저장하며 기록 화면에 실제 경로를 표시한다. `raw_b64`/`raw_hex`에 불량 UTF-8·CRLF를 포함한 실제 송수신 바이트를 보존한다.
- C 로그: C 실행 설정의 `command_log_path`에 C가 직접 REGISTER/STATE를 기록한다. D가 C 파일을 쓰거나 초기화하지 않는다. 기본은 C 프로젝트의 `logs/command_log.jsonl`이며 Windows D 폴더의 로그와 같다고 가정하지 않는다.
- 화면/CSV: 전체/TX/RX/EV 중 현재 보이는 최대 500건의 요약을 저장한다. 원문 툴팁은 400자로 제한되므로 전체 원문 증거는 JSONL을 사용한다. CSV는 UTF-8 BOM·CRLF와 수식 접두 방어를 적용하며 행 직렬화 비율을 원형 진행률로 표시한다. 성공은 파일 commit 후에만 표시하고 실패/창 종료 시 임시 출력을 취소한다.
- 반복 억제: 기본 3회 또는 5000ms 동안 같은 준비 상태면 반복 TX/RX 화면 행을 생략한다. server_time_ms와 사유 배열 순서만 바뀐 응답은 새 상태로 세지 않는다. 의미가 같은 비종료 명령 상태도 반복 표시하지 않는다. 오류·거부·종료·상태 변경은 보존한다.
- 연결 감시를 위해 기존 준비 조회 주기는 유지한다. 중복 응답도 신선도와 원문 감사 파일에는 반영한다. 화면 억제는 폴링·C 명령 로그·전체 원문 기록을 끈다는 뜻이 아니다.
- 저장 오류는 화면에 표시하며 최대 1024건을 메모리에 보류하고 후속 기록 때 재시도한다. 한도를 넘거나 파일 끝 복구에 실패하면 유실 건수/파일 신뢰 오류를 명시한다. 디스크 오류 상태에서 무제한 무손실을 보장하지 않는다.

```powershell
python -m eyetoarm.gui.window --log-dir logs/gui --stable-count 3 --stable-ms 5000
```

위 명령은 실제 연결 모드다. 소켓 없는 외형 확인은 기존 `--demo`를 사용한다. 이미 실행 중인 이전 프로그램은 파일 변경만으로 갱신되지 않는다. 현재 테스트의 기록을 필요한 만큼 확보한 뒤 사용자가 새 버전으로 다시 실행해야 파일 로그가 시작된다.

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

WSL 쪽 저장소(`~/EyetoArm`)는 C 서버·Gazebo 실행용이다. D 코드는 Windows 체크아웃에서만 고친다.

## 환경 만들기

Anaconda Prompt에서 저장소 루트로 이동한 뒤 순서대로 실행한다.

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
