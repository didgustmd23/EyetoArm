# EyeToArm 대시보드 클라이언트 (Windows)

GUI 정본과 실행에 필요한 TCP 프로토콜·스키마·D 기본 설정을 이 폴더로 분류했습니다. 서버·학습 데이터·모델은 포함하지 않으며 공용 원본은 기존 위치에 유지합니다.

기존 eyetoarm 환경을 활성화하고 이 폴더에서 실행합니다. 환경 정의는 `src/eyetoarm/gui/environment.yml`을 사용합니다.

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m eyetoarm.gui.window --demo
python -m eyetoarm.gui --env-only --json
```

실제 연결은 `python -m eyetoarm.gui.window --host <서버 숫자 IP> --port 47090`을 사용합니다. 서버 수신·네트워크 준비는 별도로 필요합니다.

자세한 실행·기능 설명은 [GUI 안내](src/eyetoarm/gui/README.md)를 참고합니다. 기존 `src/eyetoarm/gui` 진입점은 제거하고 새 폴더만 사용합니다. 저장소 안에서도 이 폴더의 config만 읽습니다. 새 감사 기록은 `logs/gui/`에 저장하고 기존 기록은 유지합니다.

## 영상과 기능 범위

기본 영상은 C의 3채널 영상 중계입니다. 영상 주소를 따로 지정하지 않으면 명령 TCP에 설정한 서버 IP와 포트47092를 사용합니다. 영상 서버가 다른 PC이면 `--view-host <영상 서버 IP>`로 바꿉니다. 명령 TCP와 영상 포트는 별도이므로 두 서버 프로세스가 준비되어야 합니다.

C의 `display_relay` 시제품을 사용하는 기본 실행과 명시적 주소 지정 예입니다.

```powershell
# dashboard-client 폴더에서 실행. 예시 IP는 실제 서버 PC 주소로 바꿉니다.
python -m eyetoarm.gui.window --host 192.168.0.10 --port 47090
# 영상 주소를 별도로 지정하는 경우
python -m eyetoarm.gui.window --host 192.168.0.10 --port 47090 --view relay --view-host 192.168.0.10 --view-port 47092
```

기존의 이 Windows PC Gazebo 창 캡처는 `--view window`로 선택합니다. `--view camera --view-host <중계 IP> --view-port 47091`은 임시 EAV1 단일 영상 중계이며 기본 주소는127.0.0.1입니다. 두 경로 모두 C 3채널 수신과 다릅니다.

영상은 고정 탑뷰(`fixed`), 그리퍼(`gripper`), 링크(`link`)의 3칸에 표시합니다. 링크의 raw·detect·center는 **수신한 영상의 표시 선택**입니다. 현재 C 송출기는 raw만 제공하고 종류 변경 요청을 읽지 않으므로, detect·center를 선택해도 서버 송출이 바뀌지는 않습니다. 해당 영상이 없으면 대기로 표시하며 원본이나 합성 주석으로 대신하지 않습니다. px/mm/θ 측정 결과도 현재 시제품에서 전달되지 않아 `—`입니다.

C 측 실행은 저장소의 [시제품 송출기 안내](../robot_ws/src/robot_server_package/README_display_relay.md)를 참조합니다. 이 링크는 전체 저장소에서만 열 수 있으며 독립 배포 폴더에는 서버 코드가 포함되지 않습니다. ETV1/47092는 C·D 합의 전 시제품 인터페이스입니다. 명령 TCP47090과 독립적이며 영상 연결만으로 로봇 준비 또는 실행 성공을 판단하지 않습니다.

실제 명령 TCP는 기존 준비 조회·수동 MOVE·CANCEL입니다. 분류 START/STOP/RECOVER와 공정·측정 모의 표시는 실제 C 분류 연결이 아닙니다. 실제 영상 생산 측의 ROS/Gazebo와 모델·보정 코드를 배포본에 넣지 않습니다. Windows 이외의 OS와 실제 장비 호환은 별도 검증 대상입니다.
