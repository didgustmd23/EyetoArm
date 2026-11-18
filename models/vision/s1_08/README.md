# S1-08 볼트·너트 분할 모델

팀 공유용 선택 가중치를 Git에 포함합니다. 저장소를 받은 뒤 학습 출력 폴더를 별도로 복사할 필요가 없습니다.

- 파일: `best.pt` — 7,041,268바이트(약 7.04 MB)
- 모델: YOLOv8n-seg / S1-08
- 클래스: `0=bolt`(볼트), `1=nut`(너트)
- SHA256: `99be54c285cd05a70ab70b22de676a6ab443cbaa8fd633aecc1dc54c421bc972`
- 실행 설정: [ai_s1_08_live.json](../../../config/vision/ai_s1_08_live.json) — conf 0.05, imgsz 1280, NMS IoU 0.7, FP32
- 학습 이력·검증 환경·한계: [MODEL_CARD.json](MODEL_CARD.json)
- 설치: [A 개발환경 안내](../../../docs/setup/A_개발환경_설치.md)

가상환경·ROS 환경을 준비하고 Gazebo 링크 카메라를 실행한 뒤 **저장소 루트**에서 실행합니다.

```bash
export YOLO_AUTOINSTALL=false
export YOLO_OFFLINE=1
export ROS_DOMAIN_ID=42
PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" python scripts/view_live_detection.py \
  --ai-config config/vision/ai_s1_08_live.json
```

원본 학습 가중치는 `outputs/training/s1_clean_hsvv06_r1/`에 그대로 보존합니다. 이 폴더에는 선택한 best.pt만 포함하고, last.pt·다른 후보·데이터셋·공유 ZIP은 넣지 않습니다. 모델을 바꿀 때는 기존 파일을 덮기보다 새 버전 폴더를 만들고 설정 경로와 해시를 함께 갱신합니다.

분할 결과는 표시용이며 계산용 윗면 윤곽이나 유효한 로봇 목표가 아닙니다. `--tracking`은 실험용 번호 표시이고 중복 검출에 따른 ID 분열 사례가 남아 있습니다. 신뢰도 0.05는 현재 선택값이며 오검출 0이나 정답 확률을 뜻하지 않습니다.
