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
cd ~/EyetoArm && source /opt/ros/jazzy/setup.bash
ros2 launch scripts/sim_scene.launch.py              # 외형만
ros2 launch scripts/sim_sort_cell.launch.py          # 로봇 포함 (gui:=false 헤드리스)
```

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
- 집기는 아직 없다. 부착 기반 집기는 실행 설계 §7.3을 따르며 C가 담당한다.
- 볼트 충돌 형상은 원기둥 근사이고, 질량은 강철 밀도로 계산한 근사값이다.
- `appearance_check_cam`은 D의 외형 확인용 카메라다. 운영 카메라는 A가 담당한다.
