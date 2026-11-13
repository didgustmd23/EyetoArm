#!/usr/bin/env python3
"""Gazebo 화면 중계(WSL 쪽 실행): Gazebo 카메라 영상을 JPEG 로 바꿔 Windows 대시보드로 보낸다.

┌─ 파트별 경계 ──────────────────────────────────────────────────────────────┐
│ [D 임시 경로] 대시보드 외형·동작 확인용이다(사용자 지정 2026-11-13).          │
│   설계의 정식 경로는 "C 표시 연결"(영상 3개 + 결과·무효화 JSON, TCP 명세     │
│   부록 B, 미승인)이다. 이 스크립트는 그 경로가 아니며 그것을 대체하지 않는다.│
│   C 표시 연결이 나오면 이 파일은 쓰지 않는다.                                │
│ [C 무변경] robot_ws · C 서버 · 명령 TCP(47090)를 건드리지 않는다.            │
│ [A 무변경] 운영 카메라·센서 설정은 A 소유다. 여기서는 월드에 이미 있는       │
│   「외형 확인 전용 카메라(D 검증용)」 appearance_check_cam 만 읽는다.         │
│   (simulation/worlds/sort_cell.sdf, 1280x720, update_rate 2 Hz. 파일 수정 없음)│
│ [명령 TCP 와 분리] 영상은 47090 에 싣지 않는다(TCP 명세 §0).                 │
└────────────────────────────────────────────────────────────────────────────┘

실행 순서(WSL Ubuntu 터미널, 각각 다른 터미널):
  1) source /opt/ros/jazzy/setup.bash
     ros2 launch scripts/sim_sort_cell.launch.py gui:=false
  2) source /opt/ros/jazzy/setup.bash
     ros2 run ros_gz_bridge parameter_bridge "/appearance_check/image@sensor_msgs/msg/Image[gz.msgs.Image"
  3) source /opt/ros/jazzy/setup.bash
     python3 src/eyetoarm/gui/gz_view_relay.py
Windows 대시보드는 127.0.0.1:47091 로 접속한다(WSL NAT localhost 전달).

전송 형식(이 임시 경로 전용, 팀 계약 아님): 프레임마다
  magic 4바이트 b"EAV1" + 본문 길이 4바이트(big-endian 부호 없는 정수) + JPEG 본문
최신 프레임만 보낸다. 느린 수신자를 위해 쌓아 두지 않는다(D 상세설계 §A.4 「최신 프레임 우선」).

이 파일 위쪽(상수 · pack_frame · FrameParser)은 표준 라이브러리만 쓴다. Windows 쪽 gz_view.py 가
같은 형식을 쓰기 위해 import 한다. ROS(rclpy · cv_bridge · cv2)는 main() 안에서만 불러온다.
"""

from __future__ import annotations

import argparse
import socket
import struct
import threading

MAGIC = b"EAV1"
HEADER = struct.Struct(">4sI")
MAX_FRAME_BYTES = 4 * 1024 * 1024   # 1280x720 JPEG 는 보통 수백 KB. 잘못된 길이로 메모리를 잡지 않게 막는다
DEFAULT_PORT = 47091                # 명령 TCP 47090 과 다른 포트
DEFAULT_TOPIC = "/appearance_check/image"


def pack_frame(jpeg: bytes) -> bytes:
    """JPEG 한 장을 전송 형식으로 감싼다."""
    if not 0 < len(jpeg) <= MAX_FRAME_BYTES:
        raise ValueError(f"프레임 크기 {len(jpeg)}바이트가 허용 범위(1~{MAX_FRAME_BYTES}) 밖")
    return HEADER.pack(MAGIC, len(jpeg)) + jpeg


class FrameParser:
    """수신 바이트를 이어 붙여 완성된 JPEG 본문을 꺼낸다. TCP 는 프레임 경계를 보장하지 않는다."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes) -> list[bytes]:
        """완성된 프레임 목록을 돌려준다. 형식이 틀리면 ValueError(호출 측은 연결을 끊고 다시 접속)."""
        self._buffer += data
        frames: list[bytes] = []
        while len(self._buffer) >= HEADER.size:
            magic, length = HEADER.unpack_from(self._buffer)
            if magic != MAGIC:
                raise ValueError("프레임 머리말이 EAV1 이 아님")
            if not 0 < length <= MAX_FRAME_BYTES:
                raise ValueError(f"프레임 길이 {length}바이트가 허용 범위 밖")
            end = HEADER.size + length
            if len(self._buffer) < end:
                break
            frames.append(bytes(self._buffer[HEADER.size:end]))
            del self._buffer[:end]
        return frames


class FrameServer:
    """접속한 대시보드에 최신 프레임을 보낸다. 127.0.0.1 에만 바인딩한다(다른 PC 에 열지 않음)."""

    def __init__(self, host: str, port: int) -> None:
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((host, port))
        self._listener.listen(2)
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self) -> None:
        while True:
            try:
                client, address = self._listener.accept()
            except OSError:
                return   # close() 로 수신 소켓이 닫힘
            client.settimeout(0.5)   # 수신자가 느리면 그 프레임은 버린다
            with self._lock:
                self._clients.append(client)
            print(f"[중계] 대시보드 접속 {address}", flush=True)

    def broadcast(self, jpeg: bytes) -> None:
        packet = pack_frame(jpeg)
        with self._lock:
            for client in list(self._clients):
                try:
                    client.sendall(packet)
                except OSError:
                    self._clients.remove(client)
                    client.close()
                    print("[중계] 대시보드 연결 끊김", flush=True)

    def close(self) -> None:
        self._listener.close()
        with self._lock:
            for client in self._clients:
                client.close()
            self._clients.clear()


def main() -> int:
    parser = argparse.ArgumentParser(description="Gazebo 카메라 영상을 Windows 대시보드로 중계(D 임시 확인 경로)")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="ROS 영상 토픽(sensor_msgs/msg/Image)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--quality", type=int, default=80, help="JPEG 품질 1~100")
    args = parser.parse_args()

    # ROS 의존성은 WSL 에만 있다. Windows 에서 이 파일을 import 해도 여기까지 오지 않는다
    import cv2
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.executors import ExternalShutdownException
    from sensor_msgs.msg import Image

    server = FrameServer(args.host, args.port)
    bridge = CvBridge()
    rclpy.init()
    node = rclpy.create_node("d_gz_view_relay")

    def on_image(message: Image) -> None:
        frame = bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        if ok:
            server.broadcast(encoded.tobytes())

    # QoS 는 reliable·깊이 1. best-effort(sensor_data)는 1280x720 원본(약 2.7 MB)이 DDS 에서 조각나며
    # 대부분 유실됐다(실측 2026-11-13: 70초에 7~13장). 깊이 1 이라 밀리면 최신 한 장만 남는다
    node.create_subscription(Image, args.topic, on_image, 1)
    print(f"[중계] {args.topic} -> {args.host}:{args.port} 대기 중 (Ctrl+C 로 종료)", flush=True)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass   # Ctrl+C 또는 종료 신호(TERM). 아래 finally 에서 소켓·노드를 닫는다
    finally:
        server.close()
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
