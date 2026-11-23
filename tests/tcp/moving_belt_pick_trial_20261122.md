# Moving belt pick 시험 안내

`moving_belt_pick_trial_20261122.py`를 Gazebo 시뮬레이션에서 실행해 컨베이어 위의 볼트를 집어 지정된 위치에 놓는 시험입니다. 이 스크립트는 **실제 로봇 팔과 그리퍼를 움직입니다.**

## 사전 조건

- Ubuntu 24.04 환경에 ROS 2 Jazzy, Gazebo Harmonic, MoveIt/ros2_control 작업공간이 설치되어 있어야 합니다.
- 저장소의 `robot_ws` 작업공간을 빌드하고 설치해야 합니다.
- 모든 터미널에서 같은 `ROS_DOMAIN_ID`와 `GZ_PARTITION`을 사용해야 합니다.
- 실행 중 Gazebo 시뮬레이션과 로봇 컨트롤러가 동작 중이어야 합니다.

## 1. 시뮬레이션 실행

터미널 1에서 저장소 루트로 이동해 ROS와 작업공간 환경을 불러온 뒤 시뮬레이션을 시작합니다.

```bash
cd /home/sub/EyetoArm
source /opt/ros/jazzy/setup.bash
source robot_ws/install/setup.bash
ros2 launch scripts/c_stack.launch.py gui:=false start_a:=true start_b:=true
```

Gazebo 창을 보면서 실행하려면 `gui:=false`를 `gui:=true`로 바꿉니다.

## 2. 컨베이어 서비스 브리지 실행

터미널 2에서도 ROS와 작업공간 환경을 불러온 다음 브리지를 실행합니다.

```bash
cd /home/sub/EyetoArm
source /opt/ros/jazzy/setup.bash
source robot_ws/install/setup.bash
ros2 run ros_gz_bridge parameter_bridge /world/sort_cell/set_pose@ros_gz_interfaces/srv/SetEntityPose
```

## 3. 시험 스크립트 실행

터미널 3에서 **저장소 루트**를 현재 디렉터리로 둔 채 ROS 환경을 불러오고 실행합니다. 스크립트가 `scripts/`를 현재 디렉터리 기준으로 찾으므로 `tests/tcp/` 안으로 `cd`하지 마세요.

```bash
cd /home/sub/EyetoArm
source /opt/ros/jazzy/setup.bash
source robot_ws/install/setup.bash
python3 tests/tcp/moving_belt_pick_trial_20261122.py part_00
```

대상 모델은 `part_00` 또는 `part_01`입니다. 선택 인수로 컨베이어 속도(m/s), 높이 오프셋(m)을 지정할 수 있습니다.

```bash
python3 tests/tcp/moving_belt_pick_trial_20261122.py part_00 0.012 0
```

예를 들어 높이 오프셋 `0.001`은 1 mm입니다. 스크립트 기본값은 속도 `0.012 m/s`, 높이 오프셋 `0 m`입니다.

## 결과 확인

터미널 출력에서 `BELT_MEASURED`에 실제 벨트 속도가 표시됩니다. 정상 완료 시 `LIFT`, `CARRY`, `DROP` 단계가 이어지고, 마지막에 `bin_confirmed=True`가 표시됩니다. 동작 중 조건을 만족하지 못하면 예외 메시지와 함께 중단됩니다.

## 문제 해결

- `Gazebo에서 ... 자세를 받지 못했다` 또는 Gazebo timestamp 오류가 나오면 시뮬레이션이 실행 중인지, `GZ_PARTITION`이 모든 터미널에서 같은지 확인합니다.
- 제어기 액션 서버 오류가 나오면 시뮬레이션 시작 로그와 컨트롤러 상태를 확인합니다.
- `bolt did not move with the belt`가 나오면 컨베이어 모델과 벨트 명령 토픽이 동작 중인지 확인합니다.
- 관절 추종 오류나 IK 오류가 나오면 시뮬레이션 초기 상태, 대상 볼트 위치와 높이 오프셋을 확인합니다.

## 종료

스크립트는 종료 시 벨트 속도를 0으로 명령합니다. 나머지 터미널은 각각 `Ctrl+C`로 종료합니다.
