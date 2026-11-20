# B 담당 협업 요청 초안 (A·C)

문서 버전 0.15 · 작성 B담당 · 2026-11-20
**목적:** B(보정·좌표)가 A·C와 **무엇을 주고받는지**, **누가 무엇을 맡는지**를 확정하기 위한 요청서입니다.

> ### ✍️ 답하는 방법
> 남은 질문 줄 뒤에 `👉 A 답:` / `👉 C 답:`으로 적거나, 단톡방에 `C-6: 대기열 좋아요`처럼 **번호: 답**으로 보내 주세요.
>
> **A가 답할 곳:** §8.2 (A-1·A-6·A-7). A-1·A-2·A-7은 A 답을 받아 B 선택을 회신했습니다(§8.2 「B 선택」). A-8은 A 구현(`4edf845`, PR #84로 main 반영)으로 해결했습니다.
> **C가 답할 곳:** §8.3 (C-9~C-12, C-15·C-16), §0.1은 확인만. C-1~C-7은 C 답(2026-11-16)을 받아 §0에 반영했습니다. C-14(인터페이스 패키지)는 B가 만들었습니다.
>
> v0.8(2026-11-16): A·C가 이미 답해 확정된 질문(옛 §0~§5)은 지우고 §0 표 한 장으로 줄였습니다. 원래 질문과 답 원문은 git 기록(PR #67 이전 버전)에 있습니다.

---

## 0. 확정된 것 — A·C가 더 고칠 것 없음

A·C 답(2026-11-12~16)으로 정해진 것입니다. 이 표의 내용은 A·C가 다시 답하거나 고칠 필요가 없습니다.

| 항목 | 결정 | 근거 |
|---|---|---|
| 벨트 | 멈추지 않고 이동 중 집기. 비상·이상 정지 뒤 재시작할 때만 이전 목표·관측을 버리고 새 프레임으로 재검출 | A 답 (옛 §0), C 회신 |
| 일정 | 11/22 통합, 11/23~25 시험·발표 준비, 11/26 발표. 범위는 컨베이어 볼트·너트 분류만 | A 답, 팀 결정 |
| 카메라 | 3대(관전·링크·그리퍼). A 운영 검출 입력은 링크 카메라 | A 답 |
| 카메라 이름·토픽 | 링크: `link_cam_link` / 광학 `link_cam_optical_frame` / 장착 `upper_arm_link` / `/link_cam/image`·`/link_cam/camera_info`. 그리퍼: `gripper_cam_optical_frame` / 장착 `robotiq_85_base_link` / `/gripper_cam/image`·`/gripper_cam/depth_image`·`/gripper_cam/camera_info` | C 답 CB-1 |
| 카메라 위치 | 찍은 시각(SIM_TIME)의 TF로 조회. B는 `world → link_cam_optical_frame`을 사진마다 읽음 | C 답 CB-2, C 요청 CB-01 |
| 찍은 시각 필드 (A→B) | `FramePacket.timestamp_ms`·`timestamp_base`(`SIM_TIME`)·`ros_stamp_ns`(이미지 `header.stamp`). `received_monotonic_ns`는 쓰지 않음 | A 답 AB-4 |
| 카메라 이름 필드 (A→B) | `FramePacket.optical_frame`(TF 조회용), `source_id`(입력 이름)는 따로 | A 답 AB-5 |
| A→B 전달 | 같은 파이프라인 안 공통 자료형·함수 호출. 원본 픽셀(`raw`), 왜곡 보정은 B가 한 번 | A 답, AB-1 |
| 상자 위치 | B가 작업 좌표로 정해 C에 줌, C가 월드·실행 설정에 반영 | C 답 CB-4 |
| 로봇 프레임 | 기준 `base_link`, 팔 끝 `tool0`. 그리퍼 오프셋(G → `tool0`)은 C가 적용 | C 답 CB-5 |
| IK·경로·충돌 | C가 MoveIt으로 계산·실행. B는 관절 각도가 아니라 **작업 좌표 파지 목표 후보**를 보냄 | C 답 IK-1·IK-2 |
| 단위 (B→C) | 위치 mm, 방향 쿼터니언 [x, y, z, w] | C 답 IK-1 |
| 관절 제한 | C의 URDF·MoveIt 설정을 따름 | C 답 IK-3 |
| 잡는 시점 예측 | B는 관측 위치·찍은 시각을 보내고, C가 도착 시점 위치를 계산 | C 답 (옛 §4.2, B안) |
| 팔이 닿는 벨트 구간 | **B가 계산, C가 확인** | C 답 (옛 §4.3) |
| 지나친 볼트 | 포기 | C 답 (옛 §4.3) |
| B→C 전달 방식 | **ROS 액션.** 액션 클라이언트 B, 액션 서버 C. 이름 `/sort_part`, 정의 `robot_ws/src/eyetoarm_interfaces/action/SortPart.action` | C 답 C-1·C-6 |
| 액션 필드 이름 | B 제안(§7.2) 그대로. `grasp_pose_candidates`, `opening_width_mm` | C 답 C-2 |
| `goal_id` | ROS 액션 goal UUID와 같은 값 | C 답 C-3 |
| 목표 유효 기한 | 필요. 찍은 시각(SIM_TIME) 기준, 기간이 지난 목표는 C가 실행하지 않고 만료 처리. 허용 시간은 벨트 속도·이동 시간 측정 뒤 결정 | C 답 C-4 |
| 벨트 속도 | **C 설정값.** 나중에 D에서 속도 신호를 받을 예정. B는 관측 위치와 찍은 시각만 보냄 | C 답 C-5 |
| center 영상 | 분류 액션과 분리해 토픽 `/b/center_overlay/image`로 | C 답 C-6 |
| 감시 자세 | C가 `shoulder_pan` −0.60 rad(나머지 초기값)을 시뮬레이션에서 먼저 시험 → 가림·충돌·집기 영향 확인 뒤 확정. 카메라 장착 각도 변경은 시험 결과에 따라 검토 | C 답 C-7 |
| 재시작 | 정지 전 목표·관측·대기 데이터를 버리고 재시작 뒤 새 데이터로 처리 | A·C 대면 합의 (2026-11-15) |
| `world`와 `base_link` | 같은 위치(이동 0, 회전 0). B 좌표를 그대로 `base_link` 좌표로 씀 | B 확인 (2026-11-17, `tf2_echo world base_link`) |
| 표시 단위 (B→D) | B가 mm로 한 번 변환, D는 재변환 안 함. 너트는 `theta_valid=false`·`theta_deg=null` | D 확인 (DB-01) |
| 보정 번호·카메라 이름 입력 | A의 링크 카메라 ROS 입력(`inputs/ros_image.py`)이 B `from_camera_info()`로 `calibration_id`를 만들고 `optical_frame`·`ros_stamp_ns`를 채움. 실제 값은 11/18 공동 시험에서 확인 | A 구현 (11/17 main) |
| 클래스 이름 | B는 `bolt`·`nut`·`BOLT_A`·`BOLT_B`·`NUT`을 모두 받고 규격은 길이로 다시 판정 → A가 2/3클래스 중 무엇을 고르든 B 수정 불필요 | B 구현 |
| 볼트 두 끝점 입력 (A-8) | 공통형 `PlanarObservation.axis_endpoints_px` [[머리 끝], [축 끝]] (원본 픽셀). 볼트는 `observed_face="axis"`면 `calc_contour` 없이 유효, `b_gate`도 같은 조건. B 코드는 수정 없이 맞음 | A 구현 `4edf845` (main 반영 2026-11-18) |
| 액션 양식 (C-14) | `robot_ws/src/eyetoarm_interfaces`: `action/SortPart.action`, `msg/GraspCandidate.msg`. C 서버는 이 양식을 씀 | B 구현 `a361b0c` |

## 0.1 B가 정한 것 (2026-11-17) — 확인만 부탁드려요

B 몫(C 합의 「B 계산·C 확인」, C 요청 CB-02·CB-05)이라 B가 값을 정했습니다. 다르면 알려 주세요.

| 항목 | B 결정 | 위치 |
|---|---|---|
| 집는 구간 | 벨트 **y −450 ~ +450 mm**. 밖이면 목표를 보내지 않음(`SKIP OUT_OF_REACH`). +450을 지난 볼트는 포기 | `frames.yaml` `pick_zone` |
| 높이 | 집기 105 mm(부품 윗면), 다가가기 205 mm, 들어올리기 250 mm, 놓기 180 mm(상자 위 100 mm). 목표에 `approach_height_mm`·`lift_height_mm`로 실음 | `frames.yaml` `heights`, `grasp.py` |
| 그리퍼 벌림 | 20 mm | `part_specs.json` |
| 상자 위치 | BOLT_A (720, −50), BOLT_B (720, 170) mm(90° 돌림, 간격 6 cm), NUT (550, 700) mm. 벨트 건너편으로 옮겨 팔 회전·거리 단축 | `frames.yaml`, `sort_cell.sdf` |
| 같은 부품 다시 보내지 않기 | 같은 `part_track_id`는 C 결과를 받을 때까지 한 번만 보냄(`SKIP ALREADY_SENT`). `run_epoch`가 바뀌면 기억을 비움. `part_track_id`가 없으면 거르지 않음 | `b_stage.py` `release()` |
| 옛 목표 버리는 조건 | 목표에 `run_epoch`·`generation`·찍은 시각을 실음. 지금 값과 다르거나 유효 기한이 지나면 C가 실행하지 않음 | `b_stage.correlation_keys()` |

## 0.2 남은 것 한눈에 (다른 파트와 정해야 하는 것만)

| 누구 | 남은 것 | 자세히 |
|---|---|---|
| A | 실제 영상에서 유효한 관측(A-1: 볼트 두 끝점·너트 윗면 테두리 추출과 오차 검증), `part_track_id`(A-6), 실행 파일 B 연결(A-7, 11/18 저장 입력 시험) | §8.2 |
| C | 새 상자 위치 도달·충돌 확인(C-15), 짧은 볼트 집는 점(C-16), 목표 유효 허용 시간(C-9), `sort_job_id`·`run_epoch` 전달(C-10), 벨트 움직이기·감시 자세 시험 결과(C-11), 취소·재시작 규칙(C-12) | §8.3 |
| B | 목표 유효 기한 `valid_until_ns` — C-9 답이 나오면, B·C 액션 연결 시험(`/sort_part` 클라이언트는 완료, C 서버와 시험), 관측면 파일럿 뒤 기준 높이(`plane_z`) 확정 | B 상세설계 §16.5 |

---

## 6. 받은 요청 회신

다른 담당이 B에게 보낸 요청의 답을 여기에 모읍니다. 요청한 문서에는 이 절을 가리키는 한 줄만 남깁니다.

### DB-01 (D → B) 규격·측정값과 중심점 영상

**B 답신(2026-11-16):**

- **단위:** B 내부 계산은 m. B가 내보내는 결과(C 액션·표시 JSON)는 위치·길이 **mm**, 각도 **deg**로 B가 한 번만 변환합니다. D는 다시 변환하지 않습니다.
- **각도:** `theta_deg`는 작업 좌표 +X에서 반시계 방향, 0~180°. 너트는 `theta_valid=false`, `theta_deg=null`(방향 없음).
- **규격 판정 기준(볼트는 머리 포함 전체 길이):**

| 규격 | 부품 | 기준 | 허용 범위 | 상자 |
| --- | --- | --- | --- | --- |
| BOLT_A | M6×12 | 18 mm | 14~22 mm | A 상자 (720, −50) mm |
| BOLT_B | M6×40 | 46 mm | 40~52 mm | B 상자 (720, 170) mm |
| NUT | M6 | 맞변 약 10 mm | 7~14 mm | 너트 상자 (550, 700) mm |
| UNRESOLVED | 판정 불가 | — | — | 집지 않음 |

- **판정 규칙:** 길이÷폭 ≥ 1.5 → 볼트(길이로 A/B), 아니면 크기로 NUT, 어디에도 맞지 않으면 UNRESOLVED.
- **무효 사유와 책임:** `CALIBRATION_MISMATCH`(A), `CONTOUR_TOO_SMALL`(A), `ASPECT_TOO_LOW`(B), `LENGTH_OUT_OF_RANGE`(B), `NUT_SIZE_OUT_OF_RANGE`(B).
- **결과 JSON 예시(시뮬레이션 실측값):**

```json
{"source_id": "link_cam", "image_seq": 1532, "part_track_id": 7, "calibration_id": "cal-02deec24c525",
 "spec": "BOLT_B", "class_name": "bolt", "valid": true, "reason": null, "reason_owner": null,
 "center_w_mm": [538.9, -151.2], "theta_deg": 176.6, "theta_valid": true,
 "length_mm": 49.7, "width_mm": 7.2}
```

```json
{"source_id": "link_cam", "image_seq": 1532, "part_track_id": 8, "calibration_id": "cal-02deec24c525",
 "spec": "NUT", "class_name": "nut", "valid": true, "reason": null, "reason_owner": null,
 "center_w_mm": [550.7, 0.1], "theta_deg": null, "theta_valid": false,
 "length_mm": 10.6, "width_mm": 9.9}
```

- **center 주석 영상:** 링크 카메라 원본 사진 위에 B 결과(윤곽·중심점·방향 화살표·규격·길이)를 그린 영상. 원본과 같은 `source_id`·`image_seq`·촬영 시각(`header.stamp`)·`frame_id`(`link_cam_optical_frame`)를 그대로 씁니다. ROS 토픽 `/b/center_overlay/image`로 C에 보내는 것을 제안합니다(이름은 C와 11/17 확정).
- **C에 보내는 조건:** `valid=true`인 결과만 집기 목표로 보냅니다. 무효 결과는 표시용으로만 보내고 사유를 붙입니다.

### DB-02 (D → B) B→C 액션 단위와 관측 상관

**B 답신(2026-11-16):** C 회신(§0 확정 표)에 맞춰 아래로 정했습니다. C는 11/17·11/19 확인 일정에서 최종 확인해 주세요.

| # | 항목 | 결정 |
| --- | --- | --- |
| 1 | 단위 | 위치 **mm**, 방향 **쿼터니언 [x, y, z, w]**. B 내부는 m, 내보낼 때 B가 한 번만 변환 |
| 2 | 기준 좌표 | `base_link` (= `world`, 로봇 바닥이 원점) |
| 3 | 움직이는 벨트 | B는 **찍은 위치 + 찍은 시각(SIM_TIME, ns)**만 보냄. 벨트 속도·잡는 시점 예측은 C |
| 4 | 잡는 방향 | 집게 가운데 프레임 G: z = 아래, x = 손가락 닫히는 방향 = 볼트 각도 + 90°. 180° 다른 후보 **2개**, C가 선택 |
| 5 | 잡는 점 | G 기준 (볼트 무게중심, 높이 = 벨트 위 5 mm = 105 mm). G → `tool0` 오프셋은 C가 적용 |
| 6 | 너트 | 집지 않음 → `PASS_THROUGH` (벨트 끝 너트 상자로 흘러감) |
| 7 | 놓는 곳 | 상자 가운데, 상자 윗면 위 100 mm (z = 180 mm), 집게 벌림 20 mm |
| 8 | 보내는 조건 | `valid=true` 볼트만 `PICK_PLACE`. 무효·판정 불가는 `SKIP` + 사유 |

- **관측 → 목표 → 실행 연결 키:** `source_id`, `image_seq`, `part_track_id`, `calibration_id`, `capture_stamp`(SIM_TIME ns). A 관측의 키를 B가 바꾸지 않고 그대로 넘깁니다(A B-07).
- **목표 예시(`PICK_PLACE`, 긴 볼트):**

```json
{"kind": "PICK_PLACE", "frame_id": "base_link", "source_id": "link_cam", "image_seq": 1532,
 "part_track_id": 7, "capture_stamp": {"base": "SIM_TIME", "ns": 94264000000},
 "calibration_id": "cal-02deec24c525", "spec": "BOLT_B", "reason": null,
 "observed_position_mm": [538.9, -151.2, 105.0],
 "grasp_candidates": [
   {"position_mm": [538.9, -151.2, 105.0], "orientation_xyzw": [0.6858, -0.7278, 0.0, 0.0], "closing_yaw_deg": -93.4},
   {"position_mm": [538.9, -151.2, 105.0], "orientation_xyzw": [0.7278, 0.6858, 0.0, 0.0], "closing_yaw_deg": 86.6}],
 "gripper_opening_mm": 20.0,
 "place": {"bin": "BOLT_B", "position_mm": [720.0, 170.0, 180.0], "orientation_xyzw": [0.7071, 0.7071, 0.0, 0.0]}}
```

- **구현:** `src/eyetoarm/transforms/grasp.py`의 `build_sort_goal()`이 이 사전을 만듭니다. ROS 액션 메시지 필드 이름은 C의 액션 정의가 나오면 이 사전에서 그대로 옮깁니다.

**B 답신 보충 (2026-11-18, D 확인의 남은 질문).** D 협업 요청 DB-01 「미회신으로 남김」 ③④, DB-02 「정책 질문」·「집계 보호」에 대한 B 답입니다.

| 질문 | B 답 |
|---|---|
| 무효·판정 불가 볼트가 C 정지 경로를 타나 | **안 탑니다.** 벨트는 멈추지 않으므로(§0) 무효·`UNRESOLVED`·집는 구간 밖(`OUT_OF_REACH`) 볼트는 `SKIP`으로 C에 보내지 않고, 벨트 끝 너트 상자 쪽으로 그냥 지나갑니다. `ISOLATE_STOP`(벨트 정지·격리)은 이번 범위(11/26)에서 쓰지 않는 것을 제안합니다. C 확인 필요 |
| 집계 | **B 결과로 수량을 세지 않는 데 동의합니다.** B 결과(`result_dict`)는 부품마다가 아니라 사진마다 나와서(40 Hz) 같은 부품이 여러 번 나옵니다. 정상 상자 수량은 C가 확인한 도착만 셉니다(D 제안과 같음). 너트 상자로 흘러간 볼트를 "미분류 통과"로 따로 셀지는 C가 도착 부품의 규격을 구분할 수 있는지에 달려 있어 C 확인이 필요합니다 |
| 화면 표시 | B 결과의 `valid=false`·`reason`·`reason_owner`로 "미판정"을 표시할 수 있습니다. 단, 집는 구간 밖(`OUT_OF_REACH`)·중복(`ALREADY_SENT`)은 C 목표에만 있고 `result_dict`에는 없습니다(`valid=true`로 나옴). D가 필요하면 B가 `result_dict`에 `goal_kind`·`goal_reason`을 추가합니다 |
| ③ `part_track_id`가 null | A-6 합의 전까지 null입니다. D는 추정해 채우지 않습니다(D 확인과 같음) |
| ④ `/b/center_overlay/image` 담당 | 발행은 B(`scripts/b_live_view.py --publish`, 토픽 이름 C 답 C-6 확정), C→D로 보내는 것은 C입니다 |

---

## 7. C에게 액션 정의 제안 (B → C)

**제안 (C 확정 전).** C가 B→C ROS 액션 메시지를 만들 때 그대로 옮겨 쓸 수 있도록, B가 이미 만드는 목표 사전(`BStage.process()` → `goal`)의 필드와 액션 필드를 1:1로 맞춘 제안입니다. 11/19 C 확인 일정(C 상세설계 §16.11)에서 확정합니다. 근거: §0 확정 표, §6 DB-02 B 답신, C 상세설계 「goal 필드 (안)」(벨트 정지 전제 부분 제외).

### 7.1 바뀐 전제 (C 상세설계 안과 다른 점)

| 항목 | C 상세설계 「goal 필드 (안)」 | 이 제안 |
|---|---|---|
| goal 종류 | `STOP_FOR_PICK`·`PICK_PLACE`·`PASS_THROUGH`·`ABORT_CYCLE` | **`PICK_PLACE`·`PASS_THROUGH` 2개.** 벨트를 멈추지 않으므로 `STOP_FOR_PICK`·`ABORT_CYCLE`·`sort_cycle_id`·`stop_goal_id`는 쓰지 않음 |
| 길이 단위 | m | **mm** (C 회신 IK-1) |
| 파지 목표 | 집기 시점 위치 | **찍은 위치 + 찍은 시각.** 집기 시점 예측은 C (§0) |
| `SKIP` | — | B 안에서만 쓰는 값. **C로 보내지 않음** (기록·표시용) |

### 7.2 액션 정의 (제안)

C의 인터페이스 패키지에 둘 파일 예시입니다. 패키지·액션 이름은 C가 정합니다.

```
# SortPart.action (제안)
# ---------- goal ----------
string goal_id                 # B가 goal마다 새로 (UUID)
string goal_kind               # "PICK_PLACE" | "PASS_THROUGH"
string sort_job_id             # C가 START 때 발급, B가 그대로
int64  run_epoch               # C 실행 세대
string part_track_id           # A 추적 ID (A 확인 필요)
string part_class              # "BOLT_A" | "BOLT_B" | "NUT"
string source_id
int64  image_seq
int64  capture_generation
int64  generation
int64  job_seq
int64  capture_stamp_ns        # 찍은 시각, SIM_TIME (ns)
string timestamp_base          # "SIM_TIME"
string frame_id                # "base_link"
float64[3] observed_position_mm
GraspCandidate[] grasp_pose_candidates   # PICK_PLACE만, 2개
float64 opening_width_mm                 # PICK_PLACE만
string  place_bin                        # PICK_PLACE만: "BOLT_A" | "BOLT_B"
float64[3] place_position_mm             # PICK_PLACE만
float64[4] place_orientation_xyzw        # PICK_PLACE만
string calibration_id
---
# ---------- result ----------
string status                  # "SUCCEEDED" | "FAILED" | "CANCELED"
string reason                  # 실패 사유 (C 사유 목록), 지나친 볼트는 "MISSED"(§0 포기)
int64  arrived_stamp_ns        # 상자 도착 확인 시각 (SIM_TIME)
---
# ---------- feedback ----------
string stage                   # "ACCEPTED" | "PLANNING" | "EXECUTING" | ...

# GraspCandidate.msg (제안)
float64[3] position_mm         # 집게 가운데 G
float64[4] orientation_xyzw    # G: z 아래, x = 손가락 닫히는 방향
float64    closing_yaw_deg
```

### 7.3 B 목표 사전 → 액션 필드 대응

B 코드는 지금 아래 사전을 만듭니다(`src/eyetoarm/transforms/grasp.py`·`b_stage.py`). C가 액션 클라이언트를 B 쪽에 둘지, B가 둘지 정하면 이 표대로 옮깁니다.

| B 사전 키 | 액션 필드 | 비고 |
|---|---|---|
| `goal_id` | `goal_id` | |
| `kind` | `goal_kind` | `SKIP`은 보내지 않음 |
| `sort_job_id`, `run_epoch` | 같음 | 파이프라인이 C에게 받아 `BStage`에 넣음 |
| `part_track_id` | 같음 | A 자료형에 아직 없음 → A 확인 |
| `spec` | `part_class` | |
| `source_id`, `image_seq`, `capture_generation`, `generation`, `job_seq` | 같음 | A 관측 값 그대로 |
| `capture_stamp.ns` | `capture_stamp_ns` | |
| `timestamp_base` | 같음 | |
| `frame_id` | 같음 | `base_link` |
| `observed_position_mm` | 같음 | |
| `grasp_candidates[]` | `grasp_pose_candidates[]` | `position_mm`·`orientation_xyzw`·`closing_yaw_deg` |
| `gripper_opening_mm` | `opening_width_mm` | |
| `place.bin` / `place.position_mm` / `place.orientation_xyzw` | `place_bin` / `place_position_mm` / `place_orientation_xyzw` | |
| `calibration_id` | 같음 | |

### 7.4 예시 (긴 볼트, `PICK_PLACE`)

```json
{"goal_id": "3f1c…", "kind": "PICK_PLACE", "sort_job_id": "job-1", "run_epoch": 1,
 "part_track_id": 7, "spec": "BOLT_B", "source_id": "link_cam", "image_seq": 1532,
 "capture_generation": 0, "generation": 0, "job_seq": 41,
 "capture_stamp": {"base": "SIM_TIME", "ns": 94264000000}, "timestamp_base": "SIM_TIME",
 "frame_id": "base_link", "observed_position_mm": [538.9, -151.2, 105.0],
 "grasp_candidates": [
   {"position_mm": [538.9, -151.2, 105.0], "orientation_xyzw": [0.6858, -0.7278, 0.0, 0.0], "closing_yaw_deg": -93.4},
   {"position_mm": [538.9, -151.2, 105.0], "orientation_xyzw": [0.7278, 0.6858, 0.0, 0.0], "closing_yaw_deg": 86.6}],
 "gripper_opening_mm": 20.0,
 "place": {"bin": "BOLT_B", "position_mm": [720.0, 170.0, 180.0], "orientation_xyzw": [0.7071, 0.7071, 0.0, 0.0]},
 "calibration_id": "cal-02deec24c525", "reason": null}
```

### 7.5 C에게 확인할 것 → C 답 받음 (2026-11-16)

C-1~C-5는 모두 답을 받아 §0 확정 표에 옮겼습니다. 요약: 액션 클라이언트 B·서버 C, 이름 `/sort_part`, 필드 이름은 B 제안대로, `goal_id` = ROS goal UUID, 유효 기한 필드 필요(허용 시간은 측정 뒤), 벨트 속도는 C 설정값.

---

## 8. 연결 전 확인 (main 코드 대조, 2026-11-16)

main 코드(PR #62 기준)를 B 연결 관점에서 대조한 결과입니다. B 코드는 준비되어 있고(`BStage`), 아래가 채워져야 A→B→C가 실제로 이어집니다.

### 8.1 연결 방법 (A 실행기)

A 실행기 `vision/pipeline_runner.run()`에 이미 B 자리 `b_consumer(frame, obs)`가 있고, A 검사(`b_gate`)를 통과한 유효 관측만 들어옵니다. B는 같은 순서의 `BStage.consume(frame, obs)`를 제공합니다.

```python
stage = BStage.from_config(repo_root, CameraProfile.from_camera_info(info_msg), RosTfLookup(tf_buffer))
client = SortPartClient(node, stage)      # transforms/sort_part_client.py (B 구현 40ea046)
stage.on_output = client.send             # B 목표를 ROS 액션 /sort_part로 (C 답 C-1·C-6). 결과가 오면 stage.release()
run(..., b_consumer=stage.consume)
```

### 8.2 A에게 확인 (A 답할 곳)

A-3(클래스 이름)·A-4(보정 번호)·A-5(ROS 카메라 입력)는 해결돼 §0으로 옮겼습니다.


| # | 현재 main 상태 | 질문 |
|---|---|---|
| A-1 | OpenCV·YOLO 검출 모두 관측이 항상 `valid=False`, `calc_contour=None`(표시용 1단계) → `b_gate`를 통과하지 못해 B가 불리지 않음 | 계산용 윗면 윤곽을 내는 유효 관측은 언제 나오나요? (11/18 연결 목표) |
| A-2 | YOLO 분할 마스크는 **전체 물체 마스크**(옆면 포함). B는 윗면 평면(z = 105 mm)으로 투영해 길이를 재므로, 옆면이 섞이면 길이가 길게 나옴 | YOLO 경로에서 `calc_contour`에 넣을 윤곽은 윗면 윤곽인가요, 전체 마스크인가요? 전체 마스크라면 B가 보정할지 같이 정해요 |
| A-6 | `PlanarObservation`에 `part_track_id` 칸이 없음 | 같은 부품을 묶는 번호는 어디서 받나요? (없으면 B 목표의 `part_track_id`는 비어서 감) |
| A-7 | `scripts/run_pipeline.py`는 아직 `profile=None`이고 `b_consumer`도 연결 전 → B가 불리지 않음 | 프로파일을 채우고 `b_consumer=stage.consume`을 연결하는 걸 11/18 공동 시험 때 같이 할까요? B도 시험에 참여해도 될까요? |
| A-8 | 공통형 `PlanarObservation`(`common/types.py`)은 `valid=True`이면 `observed_face`·`calc_contour`·`selected_index`가 모두 있어야 함(187~192줄). B가 고른 볼트 입력(머리 끝·축 끝 두 점)은 담을 칸이 없고, 윤곽 없이 유효로 만들면 거부됨 | ① 새 칸 `axis_endpoints_px: [[u_head, v_head], [u_tip, v_tip]]`(원본 픽셀, `pixel_basis` 따름, 순서 = 머리 끝 → 축 끝) 추가 ② 볼트는 `calc_contour` 대신 `axis_endpoints_px`가 있으면 유효로 인정(`observed_face`는 예: `"axis"`) ③ `b_gate`도 같은 조건으로 통과. 너트는 지금 규칙 그대로(`top_face` + 윗면 테두리 `calc_contour`). B는 이미 받을 준비가 됨(`from_observation()`) |

**A 답 (2026-11-17, A 작성).**

**A-1 👉 A 답:**
- 현재 A 관측은 OpenCV·YOLO 모두 표시용이며 `valid=False`, `observed_face=None`, `calc_contour=None`입니다. `b_gate`를 통과하지 못하므로 실제 B 계산으로 전달되지 않습니다.
- **유효 관측 제공은 A-2의 계산 입력 선택과 A의 실제 영상 추출·품질 검증 후 진행합니다.** 준비한 기준 후보 자료 전체에서 B가 필요한 항목·기준 높이·입력 형식을 선택하면, A는 선택된 입력을 실제 RGB에서 추출하는 방법을 검증하겠습니다. 검증 없이 전체 실루엣을 계산용으로 올리거나 `valid=True`로만 바꾸지는 않습니다.
- **제공 일정은 아직 확정하지 않았습니다.** B의 입력 선택 후 구현·검증 범위를 확인해 공유하겠습니다. 현재 상태에서 11/18 이전 유효 관측 제공을 약속하는 것은 아닙니다.
- **11/18 공동 시험은 우선 저장 입력·시험용 관측으로 경로를 점검하는 범위를 제안합니다.** A→B 배선, 원본 프레임·관측 식별자, 보정 프로파일과 촬영 시각의 TF·`calibration_id` 대응을 확인하되, 시험용 관측 통과를 실제 검출 관측의 유효화나 운영 성능 검증으로 보고하지 않습니다. 구체적인 배선·시험 조건은 A-7에서 함께 정리합니다.
- 실제 계산용 관측의 유효화와 실시간 B 호출 제한(D13) 해제는 별도 검증 후 결정하며, 공동 시험을 이유로 자동 해제하지 않습니다.

**A-2 👉 A 답:**
- **현재는 윗면 윤곽과 전체 마스크 모두 `calc_contour`에 넣지 않습니다.** YOLO 출력은 옆면이 포함된 전체 물체 마스크이며 표시용으로만 사용합니다. 계산용 관측은 `observed_face=None`, `calc_contour=None`, `valid=False`를 유지합니다.
- **앞으로의 계산 입력은 B와 정하고자 합니다.** 전체 마스크를 B가 보정해 사용할지, 별도 면·축·특징을 A가 추출해 제공할지 검토해 주세요. 전체 실루엣을 계산용으로 넘기지 않는 기존 계약은 합의 전까지 유지합니다. B가 필요한 입력과 계산 방식을 선택하면 A는 해당 입력의 실제 RGB 추출 가능성을 검증하겠습니다.
- 선택을 돕기 위해 **계산 기준 후보 전체를 B에 전달**하기로 했습니다. A가 특정 기준을 미리 채택한 것은 아닙니다.
  - **볼트:** 축 중심선, 머리 끝(가상 기준점)·축 끝, 머리·축 경계, 위치 후보 Pa(머리 포함 총길이 중점)·Pb(축 부분 중심). 두 위치 후보는 모델 기준 3 mm 차이입니다. 머리 포함 총길이 18/46 mm 기준은 유지합니다.
  - **너트:** 축과 최상면 평면의 교점(가상 중심), 최상면 바깥 경계, 육각 외곽, 구멍 경계입니다.
- 준비 자료는 링크 카메라·초록 벨트 Gazebo 자료 **12사례(짧은 볼트·긴 볼트·너트 각 4)**에 모델 기하·정답 자세·카메라 정보를 적용한 **모델 기반 정답 예시**입니다. 실루엣으로 B 계산을 수행했거나 실제 RGB에서 특징을 추출한 결과가 아닙니다. 가상점·내부 축선은 투영 위치만 정답이며 자기 가림은 미검증입니다.
- 공유 자료: [후보 자료·사례별 이미지](images/a_b_computation_reference/README.md), [12사례 모아보기](images/a_b_computation_reference/contact_sheet.png), [기준 정의](images/a_b_computation_reference/reference_definition.md), [기준 좌표 정본](images/a_b_computation_reference/reference_points_r1.jsonl). 원본 작업 증거는 로컬 `_workspace/evidence/2026-11-17-computation-reference-pilot/`에 보존합니다. A 검수표 52문항의 전체 판정은 B 전달의 선행조건이 아닙니다.
- 현재 정적 평가 배치의 볼트 축 높이는 105.0 mm, 너트 최상면은 105.7 mm입니다. 해당 모델·배치의 정답값이며 운영 투영 높이로 확정한 값은 아닙니다. 계산에 사용할 기준점·기준 높이·입력 형식은 B에서 정해 주세요(파지 기준은 필요시 C와 확인).
- 너트 최상면 외곽은 기존 윤곽 필드로 표현 가능한 후보지만, RGB 추출·오차 검증·운영 유효화는 아직 하지 않았습니다. 볼트 축·끝점을 선택한다면 단일 `calc_contour`에 가짜 면 윤곽으로 넣지 않고 별도 입력 계약을 정리합니다.
- 이번 후보 자료 전달 자체는 운영 계약·공통형·유효 조건을 변경하지 않습니다. 계산 기준과 추출 방법의 검증 후 유효 관측 제공 범위·시점을 A-1에 반영하겠습니다.

**A-6 👉 A 답:**
- 같은 부품을 묶는 번호는 **A 파이프라인의 프레임 간 추적에서 생성**하기로 정했고, 1차 구현을 착수 했습니다. 현재 구현 완료나 B 전달 완료 상태는 아닙니다.
- 검출된 인스턴스마다 같은 부품으로 판단하는 동안 같은 `part_track_id`를 유지하고, 모호하면 번호를 확정하지 않습니다. `image_seq`·`job_seq`를 추적 번호로 대신 쓰지 않습니다.
- **1차 범위:** A 내부 추적 상태기와 실행기 연결, 검출 창·JSONL·A→C 전달 자료에 ID와 추적 상태를 표시합니다. 현재 관측이 표시용(`valid=False`)인 점과 실시간 B 호출 제한은 유지하며, 이 단계에서 B의 중복 목표 전송 방지가 완성되는 것은 아닙니다.
- ID는 프로세스 수명의 발급기에서 재사용 없이 발급하며, 실행·추적기 객체가 바뀌어도 같은 발급기를 사용합니다. 프로세스 재시작은 `track_namespace`로 구분합니다.
- 링크 카메라가 팔과 움직이므로, 1차 픽셀 기반 추적은 **표시·기록용 실험**입니다. 카메라 이동 의심 시 트랙을 종료하지만, 팔 이동을 넘는 동일 부품 판정이나 운영 중복 집기 방지는 보장하지 않습니다.
- **B 전달 단계:** A-1의 유효 관측 연결과 연속 추적 검증 후, `PlanarObservation.part_track_id` 추가 및 B·D 소비 호환을 확인합니다. B에는 해당 관측의 출처 인스턴스 ID를 보존해 전달하며, 1차 실험 결과를 계산용 관측으로 바로 올리지 않습니다.
- B 전달 전에 함께 정할 것은 중복 방지 키의 범위(`run_id`·`source_id`·namespace 포함 여부), ID 자료형·빈값 표현, C 결과 미수신 시 대기 목표의 만료·해제 처리입니다.

**A-7 👉 A 답:**
- B도 공동 시험에 참여해 주세요. A-1과 맞춰 **11/18 시험은 우선 저장 입력·시험용 관측으로 A→B 경로를 확인하는 범위를 제안**합니다. 실제 계산용 관측 제공이나 실시간 제한 해제 완료를 뜻하지 않습니다.
- **보정 프로파일은 두 종류를 함께 구성합니다.**
  - B 계산용 `CameraProfile`: 링크 카메라 `camera_info`에서 내부 행렬·왜곡 계수·해상도·광학 프레임을 가져옵니다. `from_camera_info(info)`는 `eyetoarm.calibration.profile`의 모듈 함수입니다.
  - A 전달 검사용 `SourceProfile`: 입력 이름·해상도·크롭·`calibration_id`를 설정하고 `PipelineSettings.profile`에 연결합니다.
  - 프레임, A의 적용 조건, B의 카메라 프로파일이 **같은 보정 번호**를 사용하도록 구성하고 불일치 시 B 계산에 들어가지 않는지 확인합니다.
- **TF는 저장소만 만들지 않고 수신 흐름까지 구성합니다.**
  - TF 발행자가 로봇 관절 상태에 따른 좌표 변환을 `/tf`·`/tf_static`으로 발행합니다.
  - 같은 프로세스에 TF `Buffer`와 `TransformListener`를 만들고, Listener가 받은 변환을 Buffer에 저장할 ROS executor·`spin` 실행 흐름과 시작·종료 소유권을 정합니다.
  - B의 `RosTfLookup(buffer)`는 **현재 시각이 아니라 프레임의 촬영 시각(`ros_stamp_ns`)**에 해당하는 `world → optical_frame`을 조회합니다. 변환이 없을 때 최신 TF로 대신 계산하지 않고 실패·계산 생략 처리를 확인합니다.
  - 저장 입력 시험에는 촬영 시각에 대응하는 저장 TF를 사용합니다. 카메라 자세가 고정된 시험에 한해 그 자세의 `StaticTfLookup`을 사용할 수 있으며, 움직이는 팔의 저장 영상에 고정 TF를 임의로 적용하지 않습니다.
- **A→B 연결:** `BStage.from_config(repo_root, camera_profile, tf_lookup)`을 구성하고 `run(..., b_consumer=stage.consume)`으로 연결합니다. 시험용 관측·프로파일·TF가 일치하는 정상 경로와 식별자·보정·TF 불일치 경로를 함께 점검합니다.
- **실시간 입력 제한:** 현재 A 실행기의 실시간 `b_consumer` 거부(D13)는 유지합니다. 실제 계산용 입력의 합의·추출 검증과 실시간 수신·TF 시험 후 해제 여부를 별도로 정하며, 공동 시험을 이유로 자동 해제하지 않습니다.
- **호출·실행 경계:** B 호출은 A 제어 흐름에서 동기로 실행되므로 TF 조회(현재 기본 대기 0.05 s)와 후속 처리의 시간 상한·실패 반환을 함께 정합니다. ROS 수신 executor를 막아 TF 조회가 스스로 지연되지 않도록 실행 구조를 확인합니다. 시험의 B 출력은 기록·검증용으로 받고, 실제 로봇 액션 송신은 B·C 연결 시험 단계로 분리합니다.

**A-8 ✅ A 구현 (`4edf845`, PR #84로 main 반영 2026-11-18).** 제안 ①②③ 그대로 들어갔습니다. `axis_endpoints_px` 칸 추가, 볼트는 `observed_face="axis"`면 `calc_contour` 없이 유효, `b_gate`도 같은 조건. B 코드는 수정 없이 맞음(B 시험 54개와 `test_a8_axis_to_b.py` 통과, B담당 확인). 남은 것은 A-1(실제 영상에서 두 끝점 추출·오차 검증)입니다.

**B 회신 (2026-11-18, main `6a814d1` 확인).** 아래 네 가지 모두 B가 수락합니다. B 시험과 A→B 연결 시험 158개 통과를 확인했습니다.

| 항목 | B 답 |
|---|---|
| A-8 세부 규칙 | 수락. `axis_endpoints_px` float64 (2, 2), 0행 = 머리 끝·1행 = 축 끝, `observed_face="axis"`면 `calc_contour=None`, 볼트만(너트 거부) |
| `transforms/live_tf_buffer.py` (B 검토 대상 제안) | 수락(B 폴더에 두는 것 동의). 찍은 시각 정확 조회·최신 TF 대체 없음·시각 0 거부가 B `RosTfLookup` 규칙과 같습니다 |
| `transforms/observation_envelope_adapter.py` (B 검토 대상 제안) | 수락, B 입구로 씁니다. B 계산은 지금처럼 `stage.consume`만, 확장 자료는 받기만 하고 계산에 안 씀(합의 뒤 추가) |
| 시간 상한 (40 Hz 변경 뒤) | 그대로 유지: TF 대기 50 ms, B 계산 20 ms. B 계산은 보통 1~2 ms라 40 Hz(25 ms 간격)에서도 충분하고, 밀린 사진은 A 건너뛰기 대기열이 처리 |
| **B→C 연결 전 조건** | 지금은 `on_output` 미연결이라 문제없음. 단, `part_track_id`가 확장(`tracking`)에만 있어 B 중복 거르기가 작동하지 않으므로, B→C 연결 전에 ① A-6 `part_track_id`를 B에 넘기는 방법 합의(공통형 칸 또는 `tracking` 확장을 B가 사용) ② 그 전까지 B 액션 클라이언트는 **C 결과를 기다리는 목표가 있으면 새 목표를 보내지 않음**(팔 1대, §16.3 사이클과 같음) ③ C-9·C-10 답이 필요합니다 |

**B 선택 (2026-11-17, A-1·A-2·A-7 답에 대한 회신).** A 후보 자료(`images/a_b_computation_reference/`, r1)를 보고 B가 계산 입력·기준을 정했습니다.

| 항목 | B 선택 | 이유 |
|---|---|---|
| 볼트 입력 | **머리 끝(`B_head_end`)·축 끝(`B_tip_end`) 픽셀 두 점**, 어느 쪽이 머리인지 구분. 전체 마스크·윤곽은 쓰지 않음 | 두 점이 축 위의 점이라 높이가 축 높이 105.0 mm로 정확히 정해짐 → 높이 오차가 없음. 길이·각도·위치를 두 점에서 바로 계산 |
| 볼트 입력 계약(제안) | `PlanarObservation`에 새 칸 `axis_endpoints_px: [[u_head, v_head], [u_tip, v_tip]]`(원본 픽셀, `pixel_basis` 따름). 볼트 유효 관측은 이 칸만 있으면 되고 `calc_contour`는 없어도 됨 | A 답: "볼트 축·끝점을 고르면 단일 `calc_contour`에 가짜 면 윤곽으로 넣지 않고 별도 입력 계약" |
| 볼트 길이·각도 | 길이 = 두 점 거리(머리 포함 총길이 18/46 mm, 기존 기준 유지). 각도 = 머리→끝 축 방향, 0~180° 주기 | 기존 판정 규칙 그대로 |
| 볼트 위치 기준 | **보고 위치 = Pa(총길이 중점)**, **집는 점 = Pb(축 가운데) = Pa에서 축 끝 쪽으로 3 mm**(머리 높이 6 mm의 절반) | 두 끝점만으로 둘 다 계산됨. 집게는 몸통(Ø6)을 잡음. 파지점은 C 확인(§8.3 C-16) |
| 너트 입력 | 기존 계약 그대로: `observed_face="top_face"`, `calc_contour` = **윗면 바깥 테두리(r 5.0 원)**. 위치 = 테두리 가운데(`N_center_top`), 각도 안 씀 | A 답: 현재 자료형으로 표현 가능 |
| 기준 높이 | **볼트 105.0 mm, 너트 105.7 mm** (종류별). `part_specs.json`의 `plane_z` | A 후보 자료 r1 정적 배치 정답값 |
| B 호출 시간 상한 (A-7) | TF 대기 50 ms(`RosTfLookup`), B 계산 20 ms. TF 없음 `SKIP TF_UNAVAILABLE`, 계산 초과 `SKIP B_TIMEOUT`. 예외를 던지지 않고 결과로 돌려줌 | B 계산은 보통 1~2 ms |
| 11/18 공동 시험 | A 제안 범위에 동의: 저장 입력·시험용 관측으로 A→B 배선, 식별자, 보정 번호, 촬영 시각 TF 대응만 확인. 실시간 호출 제한(D13)은 유지 | — |

B 코드 반영: `work_result.compute_bolt_from_endpoints()`, `class_plane_z()`, `from_observation()`이 `axis_endpoints_px`를 받으면 두 점 경로로 계산, C 목표의 집는 후보 위치는 Pb, `BStage.compute_budget_ms`. 남은 A 일: 두 끝점·너트 윗면 테두리를 실제 RGB에서 뽑는 방법·오차 검증(A-1). 공통형 `axis_endpoints_px` 추가와 유효 규칙 변경(A-8)은 완료(`4edf845`).

**B-09 B 회신 (2026-11-19, A 협업 요청 B-09 「같은 프레임의 여러 물체 관측과 대상 선택」).** A 상세설계 §5.2 「후보 목록의 필드와 타입 (B-09 협의안)」을 기준으로 답합니다. Gazebo 시험 150프레임이 모두 `MULTIPLE_OBJECTS`였으므로 목록 형식이 정해져야 B가 실제 관측을 받을 수 있습니다.

| 질문 | B 답 | 이유 |
|---|---|---|
| 목록 형식 | A 안(`FrameObservations` + `ObjectObservation`) **그대로 수락** | 지금 B 입력과 같은 정보를 프레임 공통·후보별로 나눈 것 |
| 최대 개수 | **8개.** 넘으면 B가 앞 8개(`instance_index` 순)만 계산하고 잘린 수를 기록 | B 계산은 후보 하나에 1~2 ms라 8개면 계산 상한 20 ms 안 |
| TF 위치 (U-4) | **프레임 공통 `tf` 하나** (A 안 그대로). 촬영 시각·광학 프레임이 맞지 않으면 그 프레임 전체를 `SKIP TF_UNAVAILABLE` | 지금 봉투와 같은 방식, 사진마다 TF 한 번 |
| 결과 대응 | B 결과(표시 사전)마다 `instance_index`를 붙여 후보별로 돌려줌 + 선택한 후보 번호 | D 화면이 부품별 결과를 보여 줄 수 있음 |
| 대상 선택(우선순위) | ① 집을 수 있는 볼트만(유효·각도 있음·집는 구간 안) ② 그중 **작업 좌표 y가 가장 큰 것**(벨트 끝에 가장 가까운 것) ③ y 차이 5 mm 이내면 `instance_index`가 작은 것 | 벨트가 −y → +y로 가므로 y가 큰 볼트가 곧 집는 구간(+450 mm)을 벗어남 |
| 볼트가 없을 때 | 너트 중 y가 가장 큰 것 하나를 `PASS_THROUGH` | 지금 규칙과 같음 |
| C에 보내는 수 | **한 사진에 목표 1개.** 나머지 후보는 B 결과(표시)로만 남김 | 팔 1대. B 클라이언트도 같은 종류가 결과 대기 중이면 안 보냄(§8.2 B 회신) |
| C 실패·거절 때 다음 후보 | 같은 사진의 다음 후보를 보내지 않고 **다음 사진에서 다시 고름** | 벨트가 움직여 옛 사진 위치는 이미 틀림 |
| 시간 여유 | 지금은 쓰지 않음. C-09 답(집기 가능 구간·예상 처리 시간)이 오면 곧 구간을 벗어날 볼트를 건너뛰는 조건을 추가 | 벨트 속도는 C 설정값이라 B가 모름(C-5) |
| 옛 방식과 호환 | 지금 `BStage.process(관측, 프레임)`은 유지하고 목록용 `BStage.process_frame(목록)`을 새로 추가 | A 전환 동안 둘 다 동작 |

B 코드(`process_frame`)는 A가 이 답에 동의하면 만듭니다. 형식이 바뀌면 다시 짜야 하기 때문입니다.

**B 구현 (2026-11-20, A 동의 2026-11-19 뒤).** `BStage.process_frame(목록 봉투)`를 만들었습니다(`src/eyetoarm/transforms/b_stage.py`). 위 표 그대로입니다.
- 입력: A 목록 봉투 `eyetoarm.b_input/2`(`list_envelope.make_list_envelope`). 다른 버전이면 `ValueError`. TF는 봉투의 프레임 공통 `tf`만 씀(없으면 사진 전체 `SKIP TF_UNAVAILABLE`).
- 앞 8개만 계산, 잘린 수는 `truncated`. 고르기: `PICK_PLACE` 볼트 중 y 최대(5 mm 이내 동률은 작은 `instance_index`) → 없으면 `PASS_THROUGH` 너트 y 최대 → 없으면 고르지 않음.
- C(`on_output`)로는 고른 1개만. 안 고른 후보의 보낼 수 있는 목표는 **`SKIP NOT_SELECTED`**(새 이유, C로 가지 않음).
- 계산 시간 상한 20 ms는 **사진 한 장 전체** 시간 기준(이 표 「최대 개수」 행의 약속). 이 PC 측정: 볼트 1개 중앙값 0.05 ms, 상위 1% 0.27 ms → 8개 약 2 ms.
- 돌려주는 값(JSON 사전, A 기록기가 그대로 씀): `version`(`eyetoarm.b_frame_output/1`)·`candidate_count`·`computed`·`truncated`·`selected_instance_index`·`selection`(`BOLT_MAX_Y`/`NUT_MAX_Y`/`TF_UNAVAILABLE`/`NONE`)·`frame_compute_ms`·`goal`·`results`(후보별 표시 사전 + `instance_index`·`selected`·`goal_kind`·`goal_reason`).
- 기존 `BStage.process`·`consume`(단계 T)은 그대로입니다.
- **A 확인 요청:** A 시험 2개(`test_live_stage_l.py::test_no_process_frame_refused`, `test_stage_l_cli.py::test_no_process_frame_refused_before_files`)는 `process_frame`이 없다는 전제라 이제 실패합니다. 두 번째 시험은 실제 `BStage.process_frame`으로 종료 0(`b_list_calls=1 candidates_sent=1`)이 나와 연결 자체는 동작합니다. 시험 수정은 A 파일이라 B가 고치지 않았습니다.
- **D 알림:** 새 이유 `NOT_SELECTED`와 후보별 `instance_index`·`selected`·`goal_kind`·`goal_reason`이 결과에 생겼습니다(DB-02 보충 「화면 표시」 행에서 제안한 `goal_kind`·`goal_reason`). C에게는 C-09로 집기 가능 구간·처리 시간 제공 방식을 함께 묻습니다.

### 8.3 C에게 확인 → C-6·C-7 답 받음, 남은 것

C-6(전달 방식)·C-7(감시 자세)은 C 답을 §0에 옮겼습니다. C-8(벨트 속도)은 C-5 답으로 정리됐습니다(C 설정값, B는 보내지 않음).

남은 질문:

| # | 질문 |
|---|---|
| C-9 | 목표 유효 허용 시간(찍은 시각 + 몇 ms)은 언제 정하나요? 정해지면 B가 목표에 `valid_until_ns`를 넣습니다 |
| C-10 | `sort_job_id`·`run_epoch`는 B가 어떻게 받나요? (액션 서버가 START 때 알려 주는 토픽·서비스, 또는 액션 goal에 C가 채움) |
| C-11 | 벨트를 움직이게 하는 설정(시뮬레이션 모델)은 누가 만드나요? 감시 자세 시험 결과는 언제 나오나요? |
| C-12 | 취소·중복 goal·C 재시작 때 C 동작 규칙. (높이는 B가 정함 §0.1. B 쪽은 같은 `part_track_id`를 결과 전까지 한 번만 보냄) |
| ~~C-14~~ | ✅ **B가 만듦** (`a361b0c`): `robot_ws/src/eyetoarm_interfaces`의 `action/SortPart.action`·`msg/GraspCandidate.msg`. 필드는 §7.2 제안 그대로. C는 액션 서버에서 이 양식을 쓰면 됩니다. 바꿀 점이 있으면 알려 주세요 |
| C-15 | 볼트 상자를 벨트 건너편으로 옮겼습니다(2026-11-17): BOLT_A (720, −50), BOLT_B (720, 170) mm, 90° 돌림. 먼 모서리 0.79~0.82 m, 놓는 높이 180 mm에서 팔이 닿는지, 벨트 위로 지날 때 충돌이 없는지 확인 부탁드립니다 |
| C-16 | 볼트 집는 점은 **축 가운데 Pb**(머리 포함 중점에서 축 끝 쪽 3 mm, 높이 105 mm)로 보냅니다. 짧은 볼트(BOLT_A)는 축이 12 mm라 집게 손가락이 머리에 걸리지 않는지 확인 부탁드립니다 |

### 8.4 B 반영 (2026-11-16 오후)

D 확인(DB-01 미회신 ①②)과 C 요청 CB-05, B→C 제안 1에 맞춰 B 코드에 먼저 넣었습니다. 팀 합의에 따라 이름·형식은 바꿀 수 있습니다.

| 항목 | 반영 | 위치 |
|---|---|---|
| 연결 키 부착 (D ①, CB-05) | C 목표와 D 표시 사전(`result_dict`)에 같은 키 `run_id`·`capture_generation`·`generation`·`job_seq`·`detector_config_id`·`timestamp_ms`·`timestamp_base`·`capture_stamp_ns`를 B가 붙임. C 목표에는 `goal_id`·`sort_job_id`·`run_epoch`도 | `b_stage.correlation_keys()` |
| 픽셀 중심 (D ②) | 표시 사전에 `center_px` [u, v] (원본 사진 기준, 소수 1자리). TF가 없거나 중심을 못 구하면 `null` | `BStage._center_px()` |
| C 대기열 넘기기 (B→C 제안 1, **C가 ROS 액션을 골라 운영에는 안 씀**. 시험·같은 프로세스 임시 연결용으로만 남김) | `stage.on_output = make_c_handoff(loop, sort_goal_queue)` — `loop.call_soon_threadsafe`로 넣음, `PICK_PLACE`·`PASS_THROUGH`만, 가득 차면 가장 오래된 목표를 버림, `stats`에 보낸·건너뛴·버린 수 | `b_stage.make_c_handoff()` |
