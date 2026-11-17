# B 담당 협업 요청 초안 (A·C)

문서 버전 0.9 · 작성 B담당 · 2026-11-17
**목적:** B(보정·좌표)가 A·C와 **무엇을 주고받는지**, **누가 무엇을 맡는지**를 확정하기 위한 요청서입니다.

> ### ✍️ 답하는 방법
> 남은 질문 줄 뒤에 `👉 A 답:` / `👉 C 답:`으로 적거나, 단톡방에 `C-6: 대기열 좋아요`처럼 **번호: 답**으로 보내 주세요.
>
> **A가 답할 곳:** §8.2 (A-1·A-2·A-6·A-7)
> **C가 답할 곳:** §8.3 (C-9~C-12, C-14, C-15), §0.1은 확인만. C-1~C-7은 C 답(2026-11-16)을 받아 §0에 반영했습니다.
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
| A | 유효한 관측(계산용 윤곽), 윤곽 범위(윗면/전체·머리 포함), `part_track_id`, 실행 파일 B 연결(11/18) | §8.2 |
| C | 인터페이스 패키지, 새 상자 위치 도달·충돌 확인, 목표 유효 허용 시간, `sort_job_id`·`run_epoch` 전달, 벨트 움직이기·감시 자세 시험 결과, 취소·재시작 규칙 | §8.3 |
| B | ROS 액션 클라이언트(`/sort_part`) — C 인터페이스 패키지가 나오면, 목표 유효 기한 필드 — C-9 답이 나오면, 관측면 파일럿 뒤 `part_plane_z` 확정 | B 상세설계 §16.5 |

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
stage.on_output = sort_part_client.send   # B 목표를 ROS 액션 /sort_part로 (C 답 C-1·C-6, B가 만들 예정)
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

### 8.3 C에게 확인 → C-6·C-7 답 받음, 남은 것

C-6(전달 방식)·C-7(감시 자세)은 C 답을 §0에 옮겼습니다. C-8(벨트 속도)은 C-5 답으로 정리됐습니다(C 설정값, B는 보내지 않음).

남은 질문:

| # | 질문 |
|---|---|
| C-9 | 목표 유효 허용 시간(찍은 시각 + 몇 ms)은 언제 정하나요? 정해지면 B가 목표에 `valid_until_ns`를 넣습니다 |
| C-10 | `sort_job_id`·`run_epoch`는 B가 어떻게 받나요? (액션 서버가 START 때 알려 주는 토픽·서비스, 또는 액션 goal에 C가 채움) |
| C-11 | 벨트를 움직이게 하는 설정(시뮬레이션 모델)은 누가 만드나요? 감시 자세 시험 결과는 언제 나오나요? |
| C-12 | 취소·중복 goal·C 재시작 때 C 동작 규칙. (높이는 B가 정함 §0.1. B 쪽은 같은 `part_track_id`를 결과 전까지 한 번만 보냄) |
| C-14 | 인터페이스 패키지(`robot_ws/src/eyetoarm_interfaces`, `SortPart.action`)는 누가, 언제 만드나요? B 액션 클라이언트를 짜려면 필요합니다 |
| C-15 | 볼트 상자를 벨트 건너편으로 옮겼습니다(2026-11-17): BOLT_A (720, −50), BOLT_B (720, 170) mm, 90° 돌림. 먼 모서리 0.79~0.82 m, 놓는 높이 180 mm에서 팔이 닿는지, 벨트 위로 지날 때 충돌이 없는지 확인 부탁드립니다 |

### 8.4 B 반영 (2026-11-16 오후)

D 확인(DB-01 미회신 ①②)과 C 요청 CB-05, B→C 제안 1에 맞춰 B 코드에 먼저 넣었습니다. 팀 합의에 따라 이름·형식은 바꿀 수 있습니다.

| 항목 | 반영 | 위치 |
|---|---|---|
| 연결 키 부착 (D ①, CB-05) | C 목표와 D 표시 사전(`result_dict`)에 같은 키 `run_id`·`capture_generation`·`generation`·`job_seq`·`detector_config_id`·`timestamp_ms`·`timestamp_base`·`capture_stamp_ns`를 B가 붙임. C 목표에는 `goal_id`·`sort_job_id`·`run_epoch`도 | `b_stage.correlation_keys()` |
| 픽셀 중심 (D ②) | 표시 사전에 `center_px` [u, v] (원본 사진 기준, 소수 1자리). TF가 없거나 중심을 못 구하면 `null` | `BStage._center_px()` |
| C 대기열 넘기기 (B→C 제안 1, **C가 ROS 액션을 골라 운영에는 안 씀**. 시험·같은 프로세스 임시 연결용으로만 남김) | `stage.on_output = make_c_handoff(loop, sort_goal_queue)` — `loop.call_soon_threadsafe`로 넣음, `PICK_PLACE`·`PASS_THROUGH`만, 가득 차면 가장 오래된 목표를 버림, `stats`에 보낸·건너뛴·버린 수 | `b_stage.make_c_handoff()` |
