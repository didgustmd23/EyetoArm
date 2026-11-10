# 제3자 자산·오픈소스 라이선스 고지

확인일 2026-11-10. 라이선스는 각 원본의 `package.xml`·`LICENSE`·README와 GitHub API로 확인했다. 이 패키지는 법률 검토를 거치지 않았으므로, 외부 공개·배포 전에 원문 조건을 다시 확인한다.

## 1. 이 패키지 안에 포함(복사)한 자산

| 파일 | 원본 | 저작자 | 라이선스 | 수정 |
|---|---|---|---|---|
| `models/bolt_m6x12/meshes/bolt_m6x12.stl` | [FreeCAD-library](https://github.com/FreeCAD/FreeCAD-library) `Mechanical Parts/Fasteners/Bolts & Screws/Metric/ISO4762_Hex_Socket_Head_Cap_Screw_M6x12.stl` (commit `f5ca01cdcaa2f69b9295a6382719c2d90628780c`) | Normand C (최초 추가 commit `a79401b6ec`, 2013-08-05) | [CC-BY 3.0](https://creativecommons.org/licenses/by/3.0/) | 파일명 변경만. 형상 수정 없음. SDF에서 0.001 배율(mm → m)과 색상·재질 지정 |
| `models/bolt_m6x40/meshes/bolt_m6x40.stl` | 같은 저장소 `…/ISO4762_Hex_Socket_Head_Cap_Screw_M6x40.stl` (같은 commit) | obijuan (최초 추가 commit `80174e9a7b`, 2014-11-09) | CC-BY 3.0 | 위와 같음 |
| `models/nut_m6/meshes/nut_m6.stl` | 같은 저장소 `Mechanical Parts/Fasteners/Nuts/Metric/ISO4032_Hex_Nut_M6.stl` (같은 commit) | Normand C (최초 추가 commit `cfc2cd5b3a`, 2013-08-05) | CC-BY 3.0 | 위와 같음 |

FreeCAD-library README 원문: "All Parts in this repository are licensed under CC-BY 3.0 … Each Part is copyrighted by and should be attributed to its respective author(s). See commit details to find the authors of each Part." 저장소 전체의 경로 이동 commit `6f071a4bab`(WladIMirG, 2019-03-13 "INICIO")은 저작자 판단에 쓰지 않았다.

CC-BY 3.0 표기: 위 세 메시는 각 저작자의 작품이며 CC-BY 3.0으로 이용한다. 원본에서 파일명만 바꾸었다.

## 2. 실행 시 참조만 하는 패키지(이 패키지에 복사하지 않음)

| 패키지 | 버전(설치 확인) | 쓰는 것 | 라이선스 |
|---|---|---|---|
| `ur_description` ([Universal_Robots_ROS2_Description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description)) | 3.5.1 | UR5e xacro 매크로·메시·재질 | BSD-3-Clause. 단 ur8long·ur15·ur18·ur20·ur30 메시는 "Universal Robots A/S’ Terms and Conditions for Use of Graphical Documentation"(이 패키지는 UR5e만 사용) |
| `robotiq_description` ([ros2_robotiq_gripper](https://github.com/PickNikRobotics/ros2_robotiq_gripper)) | 0.0.1-3noble (apt) | 2F-85 xacro 매크로·메시·재질, UR–Robotiq 어댑터(`ur_to_robotiq_adapter`) 메시 | BSD-3-Clause (GitHub 저장소 `LICENSE`, `package.xml`) |
| `ros_gz_sim`, `ros_gz_bridge` | 1.0.24 | Gazebo 기동·로봇 생성·`/clock` 연결 | Apache-2.0 |
| `gz_ros2_control` | 1.2.20 | Gazebo ↔ ros2_control | Apache-2.0 |
| `controller_manager` | 4.48.1 | 제어기 기동 | Apache-2.0 |
| `joint_state_broadcaster`, `joint_trajectory_controller`, `parallel_gripper_controller` | 4.42.1 | 관절 상태, 팔 궤적, 그리퍼 제어 | Apache-2.0 |
| `robot_state_publisher` | 3.3.4 | TF 발행 | BSD |
| `xacro` | 2.1.1 | URDF 생성 | BSD |
| Gazebo Harmonic (gz-sim 8) | 8.x | 시뮬레이터 | Apache-2.0 |

이 패키지는 위 패키지의 파일을 복사·수정하지 않고 설치 경로에서 그대로 불러온다. 배포물에 이들의 파일을 함께 넣게 되면 각 라이선스 고지문을 같이 포함한다.

## 3. 이 패키지가 직접 만든 파일

`src/generate_meshes.py`로 만든 `models/conveyor_belt/meshes/*.stl`, `models/sort_bin/meshes/bin.stl`과 모든 `model.sdf`·`model.config`·`worlds/`·`urdf/`·`launch/`·`config/` 파일은 EyetoArm D가 작성했다. 라이선스는 팀 저장소에 라이선스가 정해지지 않아 미정이다(`package.xml`의 `TBD`).

## 4. 후속 후보 모델(아직 사용·포함하지 않음)

| 후보 | 자료 | 확인된 조건 | 상태 |
|---|---|---|---|
| Robotiq 3-Finger Adaptive Gripper | [ros2_robotiq_3f_gripper](https://github.com/PeterMitrano/ros2_robotiq_3f_gripper), [ros-industrial-attic/robotiq](https://github.com/ros-industrial-attic/robotiq) | 루트 BSD-2-Clause, 3F Gazebo 플러그인 패키지 Apache-2.0. 메시 권리는 파일별 확인 필요 | 후속 |
| SMC MHM-25D-X7400A (마그네틱) | [제조사 카탈로그](https://static.smc.eu/pdf/MHM-X7400A_EU.pdf) | 완성형 ROS 2/Gazebo 모델 없음. CAD 사용·재배포 조건 미확인 | 후속 |
| Robotiq 2F-140 | `robotiq_description`(위와 같은 BSD-3-Clause) | — | 후속 |
