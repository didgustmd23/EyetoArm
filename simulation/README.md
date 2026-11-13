# simulation — 분류 셀 외형 자산과 Gazebo 데모

EyetoArm D 파트의 외형 자산과 Gazebo 데모 구성이다. 대상은 UR5e, Robotiq 2F-85, 컨베이어, 상자 3개, M6 볼트 2종, M6 너트다.
팀 실행 구성(월드·제어기·launch)은 C 소유다(계획서 §3). 여기 있는 월드와 launch는 그 대역이자 데모로만 쓴다. `robot_ws/`는 수정하지 않는다.

- 환경: ROS 2 Jazzy, Gazebo Harmonic(gz-sim 8). 별도 빌드(colcon)는 필요 없다.
- 기준 그리퍼: **2지 Robotiq 2F-85**
- 후속 후보: 2F-140, Robotiq 3-Finger, SMC MHM-25D-X7400A(마그네틱). 조건은 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) §4

## 구조 (저장소 README의 폴더 구조를 따름)

```
simulation/
├── README.md, THIRD_PARTY_NOTICES.md
├── worlds/sort_cell.sdf          # 분류 셀 배치(임시값)
└── models/
    ├── bolt_m6x12/, bolt_m6x40/, nut_m6/            # 부품
    ├── conveyor_belt/                                 # 컨베이어 외형
    ├── sort_bin/, sort_bin_bolt_a|_bolt_b|_nut/       # 상자
    └── ur5e_2f85/                                     # 로봇 외형 조립
        ├── ur5e_2f85.urdf.xacro                       # UR5e + 어댑터 + 2F-85
        └── controllers.yaml                           # 팔 궤적·그리퍼 제어기
scripts/
├── sim_scene.launch.py           # 로봇 없이 셀 외형만
├── sim_sort_cell.launch.py       # 셀 + UR5e·2F-85 + 제어기
├── generate_sim_meshes.py        # 벨트·상자 메시 생성(임시 치수 상수)
└── write_sim_model_sdfs.py       # 모델 SDF 생성(색상·재질·충돌·관성)
```

## 실행

```bash
sudo apt install -y ros-jazzy-robotiq-description   # 최초 1회(WSL 안에서)
sudo apt install -y ros-jazzy-ur-description       # UR5e 모델
cd ~/EyetoArm && source /opt/ros/jazzy/setup.bash
ros2 launch scripts/sim_scene.launch.py              # 외형만
ros2 launch scripts/sim_sort_cell.launch.py          # 로봇 포함 (gui:=false 헤드리스)
```

## 로봇 링크 카메라

**현재 매립 시험(2026-11-13):** 사용자 요청에 따라 외부 본체·장착대를 없애고 반지름 4.2mm의
얇은 렌즈면만 팔 표면에 표시한다. 운영 메시를 절삭한 형상은 아니다.
Gazebo launch의 위치는 `-0.3420864 -0.0601521 0.1400000` m이며,
기존 각도 `1.5707963 0 -0.815927` rad는 그대로다. 이전 launch 위치에서 약 17.6mm 이동했다.
xacro 단독 기본 위치는 `-0.3400051 -0.0594480 0.0964437` m이며 해당 기본 각도도 보존했다.
각 위치는 UR5e 충돌 메시 표면을 기준으로 계산해 광학 중심을 표면보다 약 1mm 바깥에 둔 값이다.
센서 방향·화각·해상도·토픽은 유지했지만 광학 중심이 바뀌었으므로 기존 위치 보정을 그대로 쓰면 안 된다.
실제 외형·자기 가림은 사용자가 미리보기와 영상으로 확인해야 한다. 매립 변경 후 영상 검증은 미실행이다.

로봇 포함 launch는 A 상세설계 §6.1의 센서 설정을 사용한 링크 카메라를 `upper_arm_link`에 고정하고,
영상 `/link_cam/image`·보정 정보 `/link_cam/camera_info`·`/clock`을 ROS로 연결한다.
광학 프레임은 `link_cam_optical_frame`, 설정은 1280×720 RGB, 15 Hz, 수평 화각 1.0 rad이다.
Gazebo의 Image Display에서 `/link_cam/image`를 선택하거나 별도 터미널에서 확인한다.

```bash
source /opt/ros/jazzy/setup.bash
ros2 run rqt_image_view rqt_image_view
# 뷰어에서 /link_cam/image 선택
```

사용자 정정(2026-11-13: 베이스 위 흰색 기둥 위쪽)에 따라 카메라를
**흰색 세로 몸체(`upper_arm_link`) 위쪽 옆면**에 연결한다.
launch 실행과 xacro 단독 전개는 서로 다른 기존 시선을 유지한다.

| 경로 | 링크 기준 xyz (m) | 링크 기준 rpy (rad) | 초기 대기 자세 시선 |
|---|---|---|---|
| Gazebo launch | `-0.3420864 -0.0601521 0.1400000` | `1.5707963 0 -0.815927` | 로봇 정면 아래 약 43.3° |
| xacro 단독 기본값 | `-0.3400051 -0.0594480 0.0964437` | `1.5707963 0.6151567 -1.5715927` | 컨베이어 방향 수평 |

현재 렌즈는 표시 형상이며 별도의 카메라 충돌 형상은 없다.
이는 설계서 §6.1의 과거 `base_link` 고정 제안을 대체한 사용자 요청 데모 배치다.
관절이 움직이면 카메라 시점도 움직이므로 고정 카메라 보정을 그대로 사용할 수 없다.
운영 계산에는 프레임 시각의 TF와 대응하는 보정이 필요하며, 팔 가림·기구 간섭과 모델 입력 화소 기준은 별도 확인한다.
초기값을 바꾸려면 launch 인자를 사용한다(위치는 m, 자세는 rad).

장착 위치·각도를 직접 지정하는 예:

```bash
ros2 launch scripts/sim_sort_cell.launch.py \
  link_camera_xyz:="-0.3420864 -0.0601521 0.1400000" \
  link_camera_rpy:="1.5707963 0 -0.815927"
```

`sim_scene.launch.py`는 로봇을 생성하지 않으므로 팔·그리퍼 카메라는 포함하지 않는다.
월드의 외형 확인 카메라와 고정 CCTV 카메라는 포함한다.

## 그리퍼 RGB-D 카메라

사용자 요청(2026-11-13)에 따라 `robotiq_85_base_link` 앞면 중앙에 범용 RGB-D 시뮬레이션
카메라를 매립형으로 배치했다. 외부 하우징 없이 렌즈면 두 개만 표시한다.
움직이는 손가락 대신 그리퍼 몸체에 고정하며, 기본 렌즈 방향은
그리퍼의 +Z(손가락 전방)다. 특정 실물 카메라 모델의 외형·보정·노이즈를 재현한 것은 아니다.

| 항목 | 초기 설정 |
|---|---|
| 링크 | `gripper_cam_link`, 고정 조인트 `gripper_cam_joint` |
| 위치 | `robotiq_85_base_link` 기준 `0 0 0.068` m, 광학 중심은 앞면 약 Z=0.0905m |
| 자세 | `0 -1.57079632679 0` rad, 센서 +X를 그리퍼 +Z로 회전 |
| 광학 프레임 | `gripper_cam_optical_frame` (Z 전방, X 오른쪽, Y 아래) |
| 해상도·설정 주기 | 컬러·깊이 640×480, 15 Hz |
| 수평 화각·클립 | 60°, 0.05~5.0m |
| 컬러 | `/gripper_cam/image` |
| 깊이 | `/gripper_cam/depth_image` |
| 내부 파라미터 | `/gripper_cam/camera_info` |

센서 설정은 `models/ur5e_2f85/gripper_rgbd_camera.urdf.xacro`에 있다.
부모 모델과 launch 양쪽에서 `gripper_camera_xyz`·`gripper_camera_rpy`로 위치·각도를 조정할 수 있다.
기존 팔 카메라의 사용자 설정값은 변경하지 않았다. 렌즈면은 표시 형상이며 별도 충돌 형상은 없다.
렌즈 반지름은 6mm, 두 렌즈의 중심 간격은 24mm다. 얇은 렌즈면만 앞면에서 약 0.5mm 나오며
RGB-D 센서의 나머지 부분은 그리퍼 내부에 둔다. 원본 그리퍼 메시를 절삭하거나 구멍을 가공한 모델은 아니다.
렌즈 둘의 외형과 별개로, RGB와 깊이는 같은 광학 중심을 사용하는 이상적인 RGB-D 센서다.
그리퍼 카메라 영상·깊이는 표시·확인용이며 A/B 검출·작업 좌표 계산에는 자동 연결하지 않는다.

Gazebo는 기존과 같이 실행한다. 팔 카메라·그리퍼 컬러·그리퍼 깊이를 함께 보려면 별도 터미널에서:

```bash
cd /home/yhs/EyetoArm
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=42
rviz2 -d /home/yhs/EyetoArm/simulation/rviz/cameras.rviz --ros-args -p use_sim_time:=true
```

RViz 표시 이름은 `Link Camera`, `Gripper RGB`, `Gripper Depth`다.
그리퍼 방향에 따라 손가락·로봇·바닥이 보일 수 있으며, 그리퍼 이동 시 시점도 함께 변한다.

2026-11-13 매립형으로 바꾸기 전 옆면 배치의 분리된 Gazebo 실행에서 컬러 `rgb8`·깊이 `32FC1`·CameraInfo의 640×480 수신과
`gripper_cam_optical_frame` 헤더 일치를 확인했다. 깊이 단위는 m이며, 해당 대기 자세 표본의
유효 깊이는 약 0.296~0.400m였다. 기존 `/link_cam/image` 수신도 확인했다.
이는 설정 주기 15Hz의 실측이나 모든 파지 자세의 가림·기구 간섭 검증을 뜻하지 않는다.
매립 배치는 사용자 요청에 따라 실행·자동 연결하지 않았으며, 정적 변환만 확인한다.

## 고정 CCTV 카메라

사용자 요청·현재 설정 확정(2026-11-13): 로봇의 움직임을 볼 수 있도록 정면 위에서 내려다보는
고정 카메라를 월드에 추가했다. 카메라 모델은 `models/overview_camera/model.sdf`,
장착 pose는 `worlds/sort_cell.sdf`의 `overview_cam` include에서 수정한다.

| 항목 | 확정 설정 (2026-11-13) |
|---|---|
| 월드 위치 | `2.04 0.02 1.99` m — 로봇 정면(+X) 위, 이전보다 가까운 구도 |
| 월드 자세 rpy | `0 0.7027648944 3.0981417582` rad |
| 조준점 | 약 `(0.2, 0.1, 0.43)` m — 로봇·컨베이어를 포함한 작업 셀 중심 |
| 시점 | 약 40.3° 내려다보는 경사 CCTV, 월드에 고정 |
| 영상 | Gazebo `/overview_cam/image` |
| 해상도·설정 주기 | 960×720 RGB (4:3), 15 Hz |
| 수평 화각·클립 | 약 57.49° (`1.0034063702` rad), 0.05~10.0m |
| 광학 프레임 이름 | `overview_cam_optical_frame` |

후속 요청으로 이전 위치 `(2.5, 0, 2.3)m`에서 작업 셀 쪽으로 옮겼다.
회전·화각은 유지하며, 로봇 상단이 잘리지 않도록 조준 높이를 0.08m 올렸다.
정적 투영에서 로봇·셀 경계가 화면 안에 들어오는지 확인했고, 실제 변경 후 영상은 미확인이다.
양옆 여백을 줄여 달라는 추가 요청으로 1280×720·80°에서 960×720·약 64.37°로 바꿨다.
그 중간 조정에서는 위치·회전과 세로 화각(약 50.54°)을 유지하며 가로 시야만 25% 줄였다.
이후 사용자가 화각을 `1.0034063702` rad로 직접 좁히고 현재 값으로 확정했다.
현재 세로 화각은 약 44.72°이며, “세로 시야 유지”는 이전 중간값의 설명이다.
현재 확정값은 [A 상세설계 §6.2](../docs/design/A담당_비전_상세설계.md#62-고정-cctv-카메라-확정-설정)에 기록했다.
설정 확정은 실제 로봇 이동 중 가림·화면 잘림이나 실측 15Hz 달성을 검증했다는 뜻은 아니다.

사용자 요청에 따라 **이 카메라의 ROS 브리지·RViz 구독·자동 창 실행은 추가하지 않았다.**
로봇이 움직여도 고정된 시점을 유지하며, 운영 검출·작업 좌표 계산에는 연결하지 않는다.
기존 `appearance_check_cam`은 외형 확인용으로 그대로 둔다.

직접 확인하려면:

```bash
cd /home/yhs/EyetoArm
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=42
export GZ_PARTITION=eyetoarm_camera_preview
ros2 launch scripts/sim_sort_cell.launch.py
```

Gazebo 창의 `⋮` → **Image Display** → **`/overview_cam/image`**를 선택한다.
월드·모델 XML과 시야 투영만 정적으로 검사했다. 실제 영상·팔 이동 중 가림·실측 Hz는 미확인이다.

## 외형·재질

| 대상 | 메시 | 색상·재질 |
|---|---|---|
| UR5e | `ur_description` 공식 | 공식 재질 그대로 |
| 2F-85 + UR–Robotiq 어댑터 | `robotiq_description` 공식 | 공식 재질 그대로 |
| 볼트 M6x12(BOLT_A)·M6x40(BOLT_B) | FreeCAD-library ISO 4762 | 아연 도금 은색, 금속도 0.35, 거칠기 0.35 |
| 너트 M6 | FreeCAD-library ISO 4032 | 아연 도금(약간 황색), 금속도 0.35 |
| 컨베이어 | 자체 생성(프레임·롤러·벨트 고리) | 프레임 알루미늄, 롤러 강철, 벨트 검정 고무(거칠기 0.9) |
| 상자 | 자체 생성(윗면 개방) | BOLT_A 파랑, BOLT_B 초록, NUT 주황 |

부품은 은색이고 벨트는 어두운 색이다. 카메라 검출 때 대비를 확보하려고 이렇게 정했다. 월드에 환경맵이 없어서 금속도를 0.3~0.4로 낮췄다. 0.8 이상이면 렌더링이 거의 검게 나온다.

## 임시값(B 확정 전)

벨트 1.2 × 0.2 m, 벨트 윗면 Z 0.100 m, 상자 160 × 120 × 80 mm, 셀 배치(`worlds/sort_cell.sdf`).
`scripts/generate_sim_meshes.py` 상수를 고친 뒤 다시 실행하면 메시가 갱신된다.

## 한계

- 벨트는 정적 외형 모델이다. 구동·정지 제어기는 C가 담당한다.
- 집기는 아직 없다. 부착 기반 집기는 [C 상세설계 §16.7.3](../docs/design/C담당_로봇서버_상세설계.md)을 따르며 C가 담당한다.
- 볼트 충돌 형상은 원기둥 근사이고, 질량은 강철 밀도로 계산한 근사값이다.
- `appearance_check_cam`은 D의 외형 확인용 카메라다. 운영 카메라는 A가 담당한다.
