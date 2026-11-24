#!/usr/bin/env bash
# 대시보드 확인용 한 줄 실행: 어느 폴더에서 불러도 된다 (저장소 위치를 이 파일 위치로 찾음 = 절대 경로).
#   ~/EyetoArm/scripts/dashboard_up.sh                 # Gazebo + C TCP 서버 + C 중계기 + Windows 대시보드
#   ~/EyetoArm/scripts/dashboard_up.sh yolo:=true      # + A 실시간 YOLO 검출 영상(detect, .venv-a 필요)
#   ~/EyetoArm/scripts/dashboard_up.sh dashboard:=false world:=$HOME/EyetoArm/simulation/worlds/single_bolt.sdf
#   ~/EyetoArm/scripts/dashboard_up.sh run_mode:=multi # 남은 Gazebo를 정리하지 않고 경고만(기본 single은 정리 후 시작)
# ROS_DOMAIN_ID 는 이미 정해 둔 값을 쓴다(없으면 ROS 기본 0). 모든 터미널이 같아야 한다.
set -eo pipefail
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
cd "$REPO"
set +u
source /opt/ros/jazzy/setup.bash
source "$REPO/robot_ws/install/setup.bash"
echo "[dashboard_up] 저장소=$REPO  ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}"
exec ros2 launch "$REPO/scripts/dashboard_stack.launch.py" "$@"
