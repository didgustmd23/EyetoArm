# B 담당 협업 요청 초안 (A·C)

문서 버전 0.6 · 작성 B담당 · 2026-11-16
**목적:** B(보정·좌표)가 A·C와 **무엇을 주고받는지**, **누가 무엇을 맡는지**를 확정하기 위한 요청서입니다.

> ### ✍️ 답하는 방법 (1분이면 끝나요)
> 1. 이 파일에서 **`👉 A 답:`** 또는 **`👉 C 답:`** 줄을 찾아요. (`Ctrl + F` → `답:` 검색)
> 2. 자기 줄 **뒤에 그냥 글로** 적어요. 체크 표시 필요 없어요.
>    예) `👉 C 답: ③번, MoveIt으로 실행할게요`
> 3. 모르면 `모름` 또는 `나중에`라고만 적어도 돼요.
> 4. 저장 → add → commit → push 하면 끝!
>
> **A가 답할 곳:** §0, AB-4, AB-5
> **C가 답할 곳:** §0, CB-1~CB-5, **§4.1 (IK-1~IK-4)**, §4.2, §4.3, **§7.5 (C-1~C-5)**, **§8 (C-6·C-7)**
>
> 파일 고치기가 번거로우면 단톡방에 `CB-3: 100 mm/s` 처럼 **번호: 답**으로 보내도 돼요.

---

## 0. 바뀐 전제 (팀 확인 2026-11-12)

기존 설계 문서(A·B·C 상세설계)와 다른 점입니다. 이 전제가 맞는지 먼저 확인 부탁드립니다.

| 항목 | 기존 문서 | 바뀐 방식 |
|---|---|---|
| 벨트 | 볼트 발견 시 **벨트 정지**(`STOP_FOR_PICK`) → 재검출 → 집기 | **벨트 안 멈춤.** 움직이는 볼트를 잡아 상자에 넣고, 팔이 원위치로 돌아와 다음 볼트 검출 |
| 카메라 | 고정 카메라 1대 | **3대.** ① 전체 관전용(대시보드 표시만, 로봇 동작과 무관) ② **UR5e 링크 카메라**(부품 구별) ③ **그리퍼 카메라**(잡기 전 재확인) |
| 카메라 위치 | 고정 → 체스판으로 한 번만 구함 | ②·③은 **팔과 같이 움직임** → 사진마다 TF로 카메라 위치를 읽음 |
| 일정 | 5주 | **11/22 팀 통합, 11/26 발표.** 11/26까지 범위는 컨베이어 볼트·너트 분류만 |
| 통신 | — | A·B·C는 WSL ROS 안, TCP는 C↔D만. **B는 D와 직접 통신 없음** |

👉 A 답 (맞음/다름 + 다른 점): 
정상 분류는 벨트를 멈추지 않고 이동 중 집기하는 전제가 맞습니다.
비상·이상 정지 뒤 재시작할 때만 이전 목표·관측을 폐기하고 재시작 이후 새 프레임으로 재검출합니다.
카메라는 관전·링크·그리퍼 3대이며, A 운영 검출 입력인 링크 카메라의 장착 부모는 `upper_arm_link`로 사용자 확정됐습니다.
팔 이동 중에는 촬영 시각의 optical frame TF를 사용하며 실제 시각 일치는 공동 검증이 필요합니다.
그리퍼 영상을 잡기 전 재확인 입력으로 사용하는 범위·순서는 별도 협의가 필요합니다. A→B는 같은 파이프라인의 공통 자료형·내부 호출, A→C 표시 데이터는 같은 프로세스의 객체·내부 대기열이므로 별도 ROS 전달 인터페이스를 만들지 않습니다.
카메라 ROS 수신과 B→C ROS 액션은 유지하고, C→D 영상·결과 표시 연결은 명령·상태 TCP와 분리합니다.
11/22 구현·통합 목표, 11/23~25 시험·발표 준비, 11/26 중간 발표에 맞춰 분류 범위를 우선 진행하며 사람 자세는 후속입니다.

👉 C 답 (맞음/다름 + 다른 점): 

---

## 1. 전체 흐름

```
[A] 사진에서 물건 찾기
  │  윤곽·종류·찍은 시각·카메라 이름·번호표
  ▼
[B] 위치·방향·종류 판정·잡을 위치 예측
  │  잡기 명령 (위치·시각·속도·각도·종류·집게 폭·상자)
  ▼
[C] 로봇 팔 실행
  │
  └──▶ B에게: 카메라 TF·벨트 속도·상자 위치
  └──▶ D에게 (TCP): 화면 표시용 결과
```

---

## 2. A ↔ B

### 2.1 A → B (B가 받는 것)

| ID | 받는 것 | 설명 | 상태 |
|---|---|---|---|
| AB-1 | 계산용 윤곽 점 | 원본(휜 사진 그대로, `raw`) 픽셀. 왜곡 보정은 B가 한 번만 | 합의됨 (A 요청 B-01) |
| AB-2 | 종류 | 볼트 / 너트 (AI 클래스) | 합의됨 (B-05, 클래스 수는 파일럿 후) |
| AB-3 | 번호표 | `part_track_id` 등 식별자를 바꾸지 않고 전달 | 합의됨 (B-07) |
| AB-4 | **찍은 시각** ⭐신규 | 벨트가 움직이므로 필수. 시각 기준(`SIM_TIME` 등) 포함 | **요청** |
| AB-5 | **카메라 이름** ⭐신규 | 링크 카메라 / 그리퍼 카메라 구분 (TF 프레임 이름) | **요청** |
| AB-6 | 관측면 | 볼트 머리/몸통 중 어느 면 기준인지 (`observed_face`) | 11/15 파일럿 |
| AB-7 | 윤곽 범위 | 볼트 머리 포함 / 몸통만 | 11/16 파일럿 |

👉 A 답 — AB-4 찍은 시각 (가능/불가, 필드 이름, 시각 기준): 
전달 가능합니다.
기존 공통형 `FramePacket`의 `timestamp_ms`·`timestamp_base`와 ROS 원본 시각용 `ros_stamp_ns`를 사용합니다.
Gazebo 입력은 이미지 원본 `header.stamp`를 촬영 시각으로 보존하고 `timestamp_base=SIM_TIME`, `ros_stamp_ns`는 ns, `timestamp_ms`는 ms로 전달하는 계약입니다.
`received_monotonic_ns`는 PC 수신 시각이며 촬영 시각 대신 사용하거나 SIM_TIME과 직접 빼지 않습니다.
설정 변경 후 같은 프레임을 재처리해도 원본 촬영 시각은 유지합니다.
자료형 필드는 이미 있으나 ROS 입력 어댑터와 실제 header/TF 시각 대응은 아직 미구현·미검증입니다.
현재 저장 이미지·영상의 PC 단조 시각을 Gazebo SIM_TIME으로 바꿔 표시하지 않습니다.

👉 A 답 — AB-5 카메라 이름 (가능/불가, 필드 이름): 
전달 가능합니다.
기존 `FramePacket.optical_frame`에 TF 조회용 광학 프레임 이름, `source_id`에 입력 소스 식별자를 보존합니다.
두 필드는 구분하며 source_id를 TF 프레임 이름 대신 쓰지 않습니다.
현재 모델·C 회신 기준 링크는 `link_cam_optical_frame`(장착 부모 `upper_arm_link`), 그리퍼는 `gripper_cam_optical_frame`(장착 부모 `robotiq_85_base_link`)입니다.
실제 수신 시 Image/CameraInfo의 `header.frame_id`와 일치하는지 확인하고, B가 원본 촬영 시각의 TF를 사용할 수 있도록 프레임·시각을 함께 넘깁니다.
현재 A 운영 검출 입력은 링크 카메라이며, 그리퍼 재확인 검출 연결과 이동 중 실제 TF 일치는 별도 확인 대상입니다.

👉 A 답 — 기타: 
관측면·기준점·변환 평면과 AB-7 머리 포함/몸통만의 계산 윤곽 범위는 아직 파일럿 결정이 필요합니다.
현재 YOLO 파이프라인은 표시용 인스턴스·윤곽과 기록까지 연결했고 항상 `valid=False`입니다.
전체 실루엣을 임의로 `top_face`로 표시하거나 B 계산용 유효 관측으로 넘기지 않습니다.
Bolt/Nut 2클래스는 개발 모델 설정이며 최종 2/3클래스 선정은 미결입니다.
A/B가 관측면·윤곽·보정 프로파일과 무효 사유를 맞춘 뒤 계산용 관측을 연결하겠습니다.

### 2.2 B → A (B가 주는 것)

| ID | 주는 것 | 설명 |
|---|---|---|
| BA-1 | 렌즈 값 (K, dist) | 카메라별. Gazebo `CameraInfo` 기준 |
| BA-2 | 판정 결과 | 위치·각도·종류·무효 사유 (A 평가 기록용) |

### 2.3 A와 정할 일정

| 할 일 | 날짜 |
|---|---|
| AB-4·AB-5 필드 추가 확인 | **11/13** |
| 관측면 파일럿 (AB-6) | 11/15 |
| 윤곽 범위 파일럿 (AB-7), A 코드 ↔ B 코드 연결 | 11/16 |

---

## 3. C ↔ B

### 3.1 C → B (B가 받는 것)

| ID | 받는 것 | 설명 |
|---|---|---|
| CB-1 | **카메라 TF 이름** | 링크 카메라·그리퍼 카메라의 광학 프레임 이름, 붙은 링크 이름 |
| CB-2 | **팔 위치 (TF)** | 사진 찍은 시각의 카메라 위치를 TF로 조회 가능한지 |
| CB-3 | **벨트 속도·방향** | 고정 속도인지, 몇 mm/s인지, 어느 축 방향인지 |
| CB-4 | 상자 위치 | 짧은 볼트 / 긴 볼트 / 너트 상자 좌표를 누가 정할지 |
| CB-5 | 로봇 기준 프레임 이름 | 예: `base_link`, `tool0` |

👉 C 답 — CB-1 (링크 카메라 TF / 그리퍼 카메라 TF): 
링크 카메라: link_cam_link / 광학 프레임 link_cam_optical_frame / 장착 링크 upper_arm_link / 토픽 /link_cam/image, /link_cam/camera_info
그리퍼 카메라: gripper_cam_link / 광학 프레임 gripper_cam_optical_frame / 장착 링크 robotiq_85_base_link / 토픽 /gripper_cam/image, /gripper_cam/depth_image, /gripper_cam/camera_info

👉 C 답 — CB-2 (가능/불가): 가능하도록 구성. SIM_TIME 기준으로 촬영 시각의 TF 조회. 카메라·TF 통합 후 실제 일치 여부 검증 필요.

👉 C 답 — CB-3 (속도 mm/s, 방향, 고정/변함): 벨트 속도 및 방향은 구현되면 추후 확인

👉 C 답 — CB-4 (C가 정함/B가 정함): 상자 위치·영역은 B가 작업 좌표계 기준으로 정해 C에 제공하고, C는 Gazebo 월드와 실행 설정에 반영해 도달·충돌 여부를 확인.

👉 C 답 — CB-5 (프레임 이름): 로봇 베이스 프레임은 base_link, 팔 끝 플랜지 프레임은 tool0. 실제 그리퍼 파지 중심은 tool0와 다르므로, 필요한 오프셋은 별도 변환으로 적용.

### 3.2 B → C (B가 주는 것)

| ID | 주는 것 | 설명 |
|---|---|---|
| BC-1 | 볼트 위치 + 찍은 시각 | 작업대 기준 좌표(m)와 그 위치를 본 시각 |
| BC-2 | 벨트 속도 반영 정보 | 예측 담당 결정(§4)에 따라 "예측 위치" 또는 "속도"를 함께 보냄 |
| BC-3 | 볼트 방향 (각도) | 집게 회전 방향. 180° 차이 후보 2개 |
| BC-4 | 종류 | BOLT_A(M6×12) / BOLT_B(M6×40) / NUT / 미판정 |
| BC-5 | 집게 벌림 폭 | 종류별 mm |
| BC-6 | 넣을 상자 | 상자 번호 또는 좌표 |
| BC-7 | 유효 여부·사유 | 무효면 잡지 않음 |
| BC-8 | 표시용 값 | D 화면에 갈 값(`calibration_id`, 번호표 등). C가 D로 전달 |

### 3.3 C와 정할 일정

| 할 일 | 날짜 |
|---|---|
| CB-1·CB-2·CB-3 회신 | **11/13** |
| §4 로봇 팔 담당·예측 담당 결정 | **11/14** |
| 명령(ROS 액션) 필드 이름 확정, 너트 처리 방식 | 11/17 |
| 그리퍼 카메라 재확인 순서 | 11/18 |

---

## 4. ⭐ 결정 필요: 로봇 팔 계산은 누가?

### 4.1 관절 각도(IK) 담당 — **B로 전달받음, C 확인 요청**

IK(역기구학) = "손이 여기 가려면 관절 6개를 각각 몇 도 돌려야 하나" 계산.

팀에서 **"관절 각도 계산은 B가 한다"**고 전달받았습니다(2026-11-13). 맞는지와 아래 세부 내용을 C가 확인해 주세요.

👉 C 답 — B가 IK 담당 맞나요? (맞음/아님): C가 담당

| ID | 확인할 것 | B 제안 |
|---|---|---|
| IK-1 | **B가 넘길 값** | 관절 6개 각도(**라디안**). 지점별로 넘김: 접근 → 잡기 → 들기 → 상자 위 → 놓기 → 원위치 |
| IK-2 | **MoveIt 사용 방식** | C는 받은 각도로 **경로 계획·충돌 확인 후 실행**. C가 IK를 다시 계산하지 않음(중복 방지) |
| IK-3 | **관절 제한** | UR5e 관절별 허용 범위. 기본값(±360°)인지, 따로 제한하는지 C가 알려 줌 |
| IK-4 | **움직이는 벨트 기준** | B가 **예측한 잡는 위치**(§4.2)로 각도를 계산해서 보냄 |

👉 C 답 — IK-1 (넘길 값·단위·지점 동의/수정): 수정 : B는 관절 6개의 각도 대신 작업 좌표계 기준 파지 목표 후보를 전달. 위치 단위는 mm, 방향은 [x, y, z, w] 쿼터니언으로 함. 목적 상자와 배출 목표도 함께 전달합니다. 접근·들기·운반·놓기·복귀 경로는 C가 구성.

👉 C 답 — IK-2 (경로 계획·충돌 확인만 / 각도대로 바로 실행 / 기타): C가 MoveIt으로 IK와 충돌을 고려한 경로 계획을 수행하고, 계획을 확인한 뒤 실행. B가 계산한 관절 각도를 받아 실행하는 방식은 사용하지 않음.

👉 C 답 — IK-3 (기본값 / 제한 값): C의 실제 URDF와 MoveIt 설정에 지정된 관절 제한을 적용. ±360°를 기본값으로 가정하지 않으며, 정확한 범위는 현재 실행 모델에서 확인.

👉 C 답 — IK-4 (예측 위치 기준 동의/다름): C는 부품이 잡힐 시점의 예측 목표 자세를 기준으로 IK를 계산. 다만 그 위치를 B가 예측해 보낼지, C가 촬영 시각과 벨트 속도로 계산할지는 §4.2에서 별도로 합의해야 합니다.

**왜 묻나요?**
- IK-1: 넘길 값의 모양이 정해져야 B가 코드를 짤 수 있어요.
- IK-2: C도 MoveIt으로 각도를 또 계산하면 같은 일을 두 번 하게 돼요.
- IK-3: 관절 범위에 따라 IK 답이 달라져요.
- IK-4: 벨트가 움직여서, 어느 위치 기준으로 계산할지가 중요해요.

### 4.2 움직이는 볼트 예측 담당

벨트가 움직이므로 "잡는 순간 볼트가 어디 있을지" 계산이 필요합니다.

| 안 | B | C |
|---|---|---|
| A | **예측 위치**까지 계산해서 보냄 (본 위치 + 속도 × 시간) | 받은 위치로 이동 |
| **B (B 제안)** | 본 위치 + 본 시각 + 벨트 속도를 보냄 | 팔이 도착하는 시각을 알고 있으므로 **마지막 예측**을 하고 따라감 |

👉 C 답 — §4.2 (A안/B안/기타): B안 제안. B가 관측 위치·촬영 시각(SIM_TIME)·벨트 속도와 방향을 전달하고, C가 예상 도착 시점에 맞춰 잡을 위치를 계산. 세부 계산과 허용 시간은 통합 시험에서 확인.

### 4.3 잡을 수 있는 구역

| 항목 | 담당 제안 |
|---|---|
| 팔이 닿는 벨트 구간 (시작~끝 X 좌표) | B 계산, C 확인 |
| 구간을 지나친 볼트 처리 (포기 / 다음 바퀴) | C |

👉 C 답 — §4.3 구간 담당 (동의/다름): 동의

👉 C 답 — §4.3 지나친 볼트 (포기/다음 바퀴/기타): 포기

---

## 5. 답한 뒤

- push까지 하면 B담당이 `git pull`로 받아서 확인합니다.
- 결정된 내용은 B 상세설계(§12·§13·§15)에 반영하겠습니다.
- **같은 줄을 두 사람이 동시에 고치면 충돌**이 날 수 있어요. 자기 이름(A/C) 줄만 고쳐 주세요.

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
| BOLT_A | M6×12 | 18 mm | 14~22 mm | A 상자 (120, 450) mm |
| BOLT_B | M6×40 | 46 mm | 40~52 mm | B 상자 (320, 450) mm |
| NUT | M6 | 맞변 약 10 mm | 7~14 mm | 너트 상자 (550, 760) mm |
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

**B 답신(2026-11-16):** C 회신(§4 IK-1·IK-2·§4.2·§4.3·CB-4·CB-5)에 맞춰 아래로 정했습니다. C는 11/17·11/19 확인 일정에서 최종 확인해 주세요.

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
 "place": {"bin": "BOLT_B", "position_mm": [320.0, 450.0, 180.0], "orientation_xyzw": [1.0, 0.0, 0.0, 0.0]}}
```

- **구현:** `src/eyetoarm/transforms/grasp.py`의 `build_sort_goal()`이 이 사전을 만듭니다. ROS 액션 메시지 필드 이름은 C의 액션 정의가 나오면 이 사전에서 그대로 옮깁니다.

---

## 7. C에게 액션 정의 제안 (B → C)

**제안 (C 확정 전).** C가 B→C ROS 액션 메시지를 만들 때 그대로 옮겨 쓸 수 있도록, B가 이미 만드는 목표 사전(`BStage.process()` → `goal`)의 필드와 액션 필드를 1:1로 맞춘 제안입니다. 11/19 C 확인 일정(C 상세설계 §16.11)에서 확정합니다. 근거: §4 C 회신, §6 DB-02 B 답신, C 상세설계 「goal 필드 (안)」(벨트 정지 전제 부분 제외).

### 7.1 바뀐 전제 (C 상세설계 안과 다른 점)

| 항목 | C 상세설계 「goal 필드 (안)」 | 이 제안 |
|---|---|---|
| goal 종류 | `STOP_FOR_PICK`·`PICK_PLACE`·`PASS_THROUGH`·`ABORT_CYCLE` | **`PICK_PLACE`·`PASS_THROUGH` 2개.** 벨트를 멈추지 않으므로 `STOP_FOR_PICK`·`ABORT_CYCLE`·`sort_cycle_id`·`stop_goal_id`는 쓰지 않음 |
| 길이 단위 | m | **mm** (C 회신 IK-1) |
| 파지 목표 | 집기 시점 위치 | **찍은 위치 + 찍은 시각.** 집기 시점 예측은 C (§4.2) |
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
 "place": {"bin": "BOLT_B", "position_mm": [320.0, 450.0, 180.0], "orientation_xyzw": [1.0, 0.0, 0.0, 0.0]},
 "calibration_id": "cal-02deec24c525", "reason": null}
```

### 7.5 C에게 확인할 것

| # | 질문 |
|---|---|
| C-1 | 액션 이름·패키지 위치와, 액션 클라이언트를 누가 둘지 (B 어댑터 / A 파이프라인 / C) |
| C-2 | 필드 이름을 C 안(`grasp_pose_candidates`·`opening_width_*`)으로 맞출지, 위 표 그대로 갈지 |
| C-3 | `goal_id`를 ROS 액션 goal UUID와 같게 둘지 |
| C-4 | `valid_until_ms`(목표 유효 기한)가 필요한지. 필요하면 찍은 시각 + 몇 ms인지 |
| C-5 | 벨트 속도·방향을 C가 내부에서 쓰는지 (B는 보내지 않음, DB-02 결정 3) |

답은 이 문서의 표 아래에 `👉 C 답:`으로 적거나 단톡방에 번호로 주세요.

C-1
액션 클라이언트는 B, 액션 서버는 C. 액션 이름은 /sort_part, 액션 정의 위치는 robot_ws/src/eyetoarm_interfaces/action/SortPart.action로 제안.

C-2
B 제안 필드 이름을 사용. grasp_pose_candidates, opening_width_mm를 액션 필드 이름.

C-3
goal_id와 ROS 액션의 goal UUID를 같은 값으로 사용

C-4
유효기간 필드 필요, 촬영시각(SIM_TIME)기준, 허용시간은 추후 벨트 속도 및 이동 시간 등을 측정한 뒤 결정, 기간 지난 목표는 C가 실행하지 않고 만료 처리

C-5
C 설정값으로 사용, 추후 D에서 속도 등 신호 받을 예정, B는 관측 위치와 촬영 시각만 전달


---

## 8. 연결 전 확인 (main 코드 대조, 2026-11-16)

main 코드(PR #62 기준)를 B 연결 관점에서 대조한 결과입니다. B 코드는 준비되어 있고(`BStage`), 아래가 채워져야 A→B→C가 실제로 이어집니다.

### 8.1 연결 방법 (A 실행기)

A 실행기 `vision/pipeline_runner.run()`에 이미 B 자리 `b_consumer(frame, obs)`가 있고, A 검사(`b_gate`)를 통과한 유효 관측만 들어옵니다. B는 같은 순서의 `BStage.consume(frame, obs)`를 제공합니다.

```python
stage = BStage.from_config(repo_root, CameraProfile.from_camera_info(info_msg), RosTfLookup(tf_buffer))
stage.on_output = c_queue.put          # B 출력(결과·C 목표)을 C 쪽으로
run(..., b_consumer=stage.consume)
```

### 8.2 A에게 확인 (A 답할 곳)

| # | 현재 main 상태 | 질문 |
|---|---|---|
| A-1 | OpenCV·YOLO 검출 모두 관측이 항상 `valid=False`, `calc_contour=None`(표시용 1단계) → `b_gate`를 통과하지 못해 B가 불리지 않음 | 계산용 윗면 윤곽을 내는 유효 관측은 언제 나오나요? (11/18 연결 목표) |
| A-2 | YOLO 분할 마스크는 **전체 물체 마스크**(옆면 포함). B는 윗면 평면(z = 105 mm)으로 투영해 길이를 재므로, 옆면이 섞이면 길이가 길게 나옴 | YOLO 경로에서 `calc_contour`에 넣을 윤곽은 윗면 윤곽인가요, 전체 마스크인가요? 전체 마스크라면 B가 보정할지 같이 정해요 |
| A-3 | YOLO 클래스는 2클래스 `bolt`·`nut`(`coco_to_yolo_seg.py`) | B는 `bolt`·`nut`과 3클래스 이름(`BOLT_A`·`BOLT_B`·`NUT`)을 모두 받고, 규격은 길이로 다시 판정합니다. 다른 이름을 쓰면 알려 주세요 |
| A-4 | 입력에서 `calibration_id`가 None → `SourceProfile.applies_to()`가 거짓 → B 호출 안 됨 | B가 `CameraProfile.calibration_id`(예: `cal-02deec24c525`)를 드리면 프레임·프로파일에 넣어 주실 수 있나요? |
| A-5 | A 입력은 저장 사진·저장 영상만 있음. Gazebo 링크 카메라(ROS) 입력이 없어 `optical_frame`·`ros_stamp_ns`가 None | ROS 카메라 입력은 누가, 언제 만드나요? B는 이 두 값으로 촬영 시각 TF를 읽습니다 |
| A-6 | `PlanarObservation`에 `part_track_id` 칸이 없음 | 같은 부품을 묶는 번호는 어디서 받나요? (없으면 B 목표의 `part_track_id`는 비어서 감) |

### 8.3 C에게 확인 (§7.5에 추가)

| # | 현재 main 상태 | 질문 |
|---|---|---|
| C-6 | 2026-11-15 결정: A 파이프라인과 C가 **같은 프로세스**, A→C 표시 데이터는 객체·내부 대기열. C 로봇 서버의 `sort_manager.py`(액션)는 아직 빈 파일 | B 출력(C 목표·center 영상)도 같은 내부 대기열로 받을까요, ROS 액션·토픽(`/b/center_overlay/image`)으로 받을까요? 액션 서버는 언제 시험 가능할까요? |
| C-7 | 링크 카메라는 `upper_arm_link`에 붙어 팔과 같이 움직임(`sim_sort_cell.launch.py`의 `link_camera_xyz`·`link_camera_rpy`). 초기 대기 자세(`shoulder_pan` 0)에서는 정면 43° 아래를 보며 벨트 **y = −0.23 ~ +0.51 m**만 보임 → 들어오는 끝(−0.63 ~ −0.23)이 안 보여 미리 인식할 시간이 부족 | **감시 자세 제안:** `shoulder_pan` −0.60 rad(−34°), 나머지 관절은 초기값 → 벨트 **y = −0.63 ~ +0.14 m**가 보임(먼 끝에서 BOLT_A 18 mm ≈ 19 px). 부품을 기다릴 때 이 자세로 둘지, 카메라 장착 각도를 바꿀지 C·시뮬레이션 담당과 정해요 |

C-6
분류 목표는 B가 ROS 액션으로 보내고, C가 액션 서버로 받음. 중심점 영상은 분류 액션과 분리해 /b/center_overlay/image 토픽으로 받는 것을 제안. 현재 sort_manager.py는 비어 있으며, 분류 액션 서버와 실행 중재를 추후 구현할 예정. 서버 시험은 인터페이스 합의 후 진행하며, 통합 시험 목표 일정은 11/20.

C-7
제안된 감시 자세(shoulder_pan=-0.60 rad, 나머지 관절은 초기값)를 우선 시뮬레이션에서 시험. 벨트 감시 범위와 팔 가림·충돌·집기 동작 영향을 확인한 뒤 확정하고, 카메라 장착 각도 변경은 시험 결과에 따라 검토