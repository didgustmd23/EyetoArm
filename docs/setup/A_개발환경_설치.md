# A 담당 Python 환경 설치

대상은 WSL2 Ubuntu 24.04, Python 3.12, NVIDIA GPU/CUDA 12.8용 PyTorch를 사용하는 A 담당 환경입니다. ROS·Gazebo와 드라이버가 준비돼 있으면 Python 패키지는 `pip install -r requirements.txt` 한 번으로 설치합니다. Windows D 운영 GUI 환경은 이 가이드의 대상이 아닙니다.

## 1. 시스템 선행조건

- WSL에서 `nvidia-smi`가 동작하는 NVIDIA 드라이버.
- Ubuntu 기본 Python 3.12와 venv/pip. 없으면 `sudo apt install python3.12-venv python3-pip`.
- 실시간 카메라 입력에는 ROS 2 Jazzy의 rclpy·cv_bridge, 시뮬레이션에는 Gazebo·ros_gz·로봇 패키지가 필요합니다. 기존 설치 목록은 [requirements.txt](../../requirements.txt)의 시스템 패키지 주석과 [시뮬레이션 안내](../../simulation/README.md)를 참고합니다. pip가 설치하는 항목이 아닙니다.
- OpenCV 창을 띄우려면 WSLg가 필요합니다. Linux GUI 라이브러리가 없다면 `sudo apt install libgl1 libglib2.0-0`도 필요할 수 있습니다.

## 2. 새 가상환경과 Python 패키지 설치

기존 학습·평가 환경 `.venv`를 변경하지 않도록 별도 경로를 사용합니다. 경로가 이미 있으면 새 이름을 사용하고 덮어쓰지 않습니다. ROS 시스템 Python 패키지를 참조할 수 있도록 `--system-site-packages`를 지정합니다.

```bash
cd ~/EyetoArm
source /opt/ros/jazzy/setup.bash
python3 -m venv --system-site-packages .venv-a
source .venv-a/bin/activate
pip install -r requirements.txt
```

2026-11-17 사용자 결정에 따라 **새 환경은 pip OpenCV 4.11.0을 사용**합니다. 기존 `.venv`의 apt OpenCV 4.6 환경과 다르며, 설치 파일 변경만으로 기존 실행 환경이 갱신되지는 않습니다. apt 패키지를 삭제하거나 시스템 Python에 sudo pip로 설치하지 않습니다. 같은 가상환경에 다른 OpenCV 배포판(headless·contrib)을 함께 설치하지 않습니다.

PyTorch는 공식 CUDA 12.8 인덱스에서 `torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`을 설치합니다. CUDA 런타임 관련 Python 패키지 용량이 크므로 디스크 공간과 다운로드 시간이 필요합니다. NVIDIA 드라이버는 별도이며 GPU가 없는 PC나 다른 OS/아키텍처용 구성은 이 파일의 대상이 아닙니다.

## 3. 설치 확인

```bash
python -m pip check
python -c "import sys, cv2, numpy, torch, ultralytics; print(sys.executable); print('OpenCV', cv2.__version__, cv2.__file__); print('NumPy', numpy.__version__); print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'GPU', torch.cuda.is_available()); print('Ultralytics', ultralytics.__version__)"
python -c "import rclpy, cv_bridge; print('ROS imports OK')"
```

예상 버전은 OpenCV 4.11.0, NumPy 1.26.4, torch 2.11.0+cu128, Ultralytics 8.4.211입니다. GPU를 사용하는 PC에서는 `GPU True`를 확인합니다. `cv2.__file__`이 새 가상환경 안을 가리키는지도 확인합니다. ROS import가 실패하면 같은 터미널에서 Jazzy setup을 불러왔는지 확인합니다.

## 4. A 검출 확인 창 실행

```bash
cd ~/EyetoArm
source /opt/ros/jazzy/setup.bash
source .venv-a/bin/activate
export ROS_DOMAIN_ID=42
export YOLO_AUTOINSTALL=false
export YOLO_OFFLINE=1
PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" python scripts/view_live_detection.py \
  --ai-config config/vision/ai_s1_08_live.json --tracking
```

Gazebo와 링크 카메라 브리지는 별도 터미널에서 먼저 실행합니다. ROS의 PYTHONPATH를 유지하도록 `PYTHONPATH=src`만으로 덮어쓰지 않습니다. 현재 S1-08 설정은 저장소에 포함한 [모델 가중치](../../models/vision/s1_08/README.md)를 사용합니다. pip는 가중치를 설치하지 않으며 Git에서 받은 `models/vision/s1_08/best.pt`의 SHA가 설정과 일치해야 합니다. 다른 실험 모델·학습 데이터·학습 출력은 별도 로컬 산출물입니다.

## 5. 검증 범위

새 환경은 OpenCV 버전이 달라 기존 apt 환경의 학습·추론·ROS/GUI 시험 결과를 새 환경의 동작 확인으로 취급하지 않습니다. 위 import·CUDA 확인, 저장 입력 검출, ROS 수신·실제 창 표시 순서로 확인합니다. 실제 수행한 검사와 미실행 항목은 변경 보고에 구분해 기록합니다.
