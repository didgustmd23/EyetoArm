# B → C 분류 목표 액션 제안

문서 버전 0.1 · 작성 B담당 · 2026-11-16 · **제안 (C 확정 전)**

**목적:** C가 B→C ROS 액션 메시지를 만들 때 그대로 옮겨 쓸 수 있도록, B가 이미 만드는 목표 사전(`BStage.process()` → `goal`)의 필드와 액션 필드를 1:1로 맞춘 제안입니다. 11/19 C 확인 일정(C 상세설계 §16.11)에서 확정합니다.

근거: B 협업 요청 §4 C 회신(IK-1·IK-2·§4.2·§4.3·CB-4·CB-5), §6 DB-02 B 답신, C 상세설계 「goal 필드 (안)」(벨트 정지 전제 부분 제외).

---

## 1. 바뀐 전제 (C 상세설계 안과 다른 점)

| 항목 | C 상세설계 「goal 필드 (안)」 | 이 제안 |
|---|---|---|
| goal 종류 | `STOP_FOR_PICK`·`PICK_PLACE`·`PASS_THROUGH`·`ABORT_CYCLE` | **`PICK_PLACE`·`PASS_THROUGH` 2개.** 벨트를 멈추지 않으므로 `STOP_FOR_PICK`·`ABORT_CYCLE`·`sort_cycle_id`·`stop_goal_id`는 쓰지 않음 |
| 길이 단위 | m | **mm** (C 회신 IK-1) |
| 파지 목표 | 집기 시점 위치 | **찍은 위치 + 찍은 시각.** 집기 시점 예측은 C (§4.2) |
| `SKIP` | — | B 안에서만 쓰는 값. **C로 보내지 않음** (기록·표시용) |

## 2. 액션 정의 (제안)

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
string reason                  # 실패 사유 (C 사유 목록), 지나친 볼트는 "MISSED"(§4.3 포기)
int64  arrived_stamp_ns        # 상자 도착 확인 시각 (SIM_TIME)
---
# ---------- feedback ----------
string stage                   # "ACCEPTED" | "PLANNING" | "EXECUTING" | ...

# GraspCandidate.msg (제안)
float64[3] position_mm         # 집게 가운데 G
float64[4] orientation_xyzw    # G: z 아래, x = 손가락 닫히는 방향
float64    closing_yaw_deg
```

## 3. B 목표 사전 → 액션 필드 대응

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

## 4. 예시 (긴 볼트, `PICK_PLACE`)

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
 "place": {"bin": "BOLT_B", "position_mm": [320.0, 450.0, 180.0], "orientation_xyzw": [1.0, 0.0, 0.0, 0.0]},
 "calibration_id": "cal-02deec24c525", "reason": null}
```

## 5. C에게 확인할 것

| # | 질문 |
|---|---|
| C-1 | 액션 이름·패키지 위치와, 액션 클라이언트를 누가 둘지 (B 어댑터 / A 파이프라인 / C) |
| C-2 | 필드 이름을 C 안(`grasp_pose_candidates`·`opening_width_*`)으로 맞출지, 위 표 그대로 갈지 |
| C-3 | `goal_id`를 ROS 액션 goal UUID와 같게 둘지 |
| C-4 | `valid_until_ms`(목표 유효 기한)가 필요한지. 필요하면 찍은 시각 + 몇 ms인지 |
| C-5 | 벨트 속도·방향을 C가 내부에서 쓰는지 (B는 보내지 않음, DB-02 결정 3) |

답은 이 문서의 표 아래에 `👉 C 답:`으로 적거나 단톡방에 번호로 주세요.
