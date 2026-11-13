# D 클라이언트 (Windows)

D 운영 GUI와 TCP 클라이언트를 Windows에서 새로 구현하는 폴더다. 계획서 v1.6 §2.1(D GUI는 Windows, A·B·C·로봇은 WSL)과 §2.5(Windows Python·PySide6 환경과 저장소 체크아웃 위치는 D가 기록)를 따른다.

2026-11-13 사용자 요청에 따라 최신 `d_client/`를 이 `gui/`로 옮기고 기존 구형 `gui/` 구현을 삭제했다. `src/eyetoarm/tcp/`와 다른 폴더는 변경하지 않는다.

## 현재 실행 경로

기존 `eyetoarm` 환경을 활성화한 뒤 프로젝트 루트의 PowerShell에서 실행한다.

```powershell
$env:PYTHONPATH = 'src'
python -m eyetoarm.gui.window --demo
python -m eyetoarm.gui --env-only
```

실제 C 준비 조회·Gazebo 창 캡처는 `python -m eyetoarm.gui.window`, 임시 카메라 중계 수신은 `python -m eyetoarm.gui.window --view camera`로 실행한다. `--demo`는 소켓을 열지 않는다.

다른 폴더 수정 금지 범위에 따라 `tests/`·외부 문서의 `eyetoarm.d_client` 참조와 구형 GUI 테스트는 그대로 남아 있다. 이전 경로를 사용하는 테스트·실행 안내는 후속 갱신이 필요하다. 테마 선택을 보존하기 위해 QSettings의 기존 `EyeToArms/d_client` 키는 유지한다.

1일차 진단 JSON의 `tool: eyetoarm.d_client` 식별자도 기존 기록과의 호환성을 위해 유지한다. 실행 명령과 Python import 경로는 `eyetoarm.gui`를 사용한다.

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
