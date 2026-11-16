"""B 실시간 확인 창 — 링크 카메라 영상 위에 B의 작업 좌표와 판정 결과를 겹쳐 본다.

하는 일 (매 영상마다)
1. `/link_cam/camera_info` → 렌즈 값 K (B `CameraProfile`)
2. 영상 시각의 TF `world` → `link_cam_optical_frame` → `T_W_C` (팔이 움직여도 따라감)
3. 벨트 테두리·10 cm 격자·상자 3개·W 축을 그림  ← 선이 시뮬레이션 벨트와 겹치면 좌표가 맞는 것
4. 벨트 위 밝은 물체를 임시로 찾아(운영 검출은 A 담당) B 판정 → 중심점·화살표·규격 표시
5. 선택: 그린 영상을 `/b/center_overlay/image`로 발행 (C가 D로 전달할 B 중심점 주석 영상)

실행 (WSL, 시뮬레이션을 켠 뒤 새 터미널, 저장소 맨 위 폴더에서):
    source /opt/ros/jazzy/setup.bash
    PYTHONPATH=src python3 scripts/b_live_view.py
    # 창에서 s = 화면 저장, q = 끝내기
    # --publish 를 붙이면 주석 영상을 ROS 토픽으로도 보냄
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

from eyetoarm.calibration.profile import from_camera_info
from eyetoarm.transforms.overlay import draw_guides, draw_results, simple_part_contours
from eyetoarm.transforms.planar_transform import make_T, quat_to_rot
from eyetoarm.transforms.work_result import compute_work_result, load_specs

ROOT = Path(__file__).resolve().parents[1]


class BLiveView(Node):
    def __init__(self, camera_key: str, publish: bool, detect: bool) -> None:
        super().__init__("b_live_view", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.frames = yaml.safe_load(open(ROOT / "config/robot/frames.yaml", encoding="utf-8"))
        self.specs = load_specs(ROOT / "config/calibration/part_specs.json")
        cam = self.frames["cameras"][camera_key]
        self.world = self.frames["world"]["frame_id"]
        self.optical = cam["optical_frame"]
        self.detect = detect

        self.bridge = CvBridge()
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.profile = None
        self.latest = None
        self.status = "waiting camera_info"

        self.create_subscription(CameraInfo, cam["camera_info_topic"], self.on_info, 10)
        self.create_subscription(Image, cam["image_topic"], self.on_image, 2)
        self.pub = self.create_publisher(Image, "/b/center_overlay/image", 2) if publish else None
        self.get_logger().info(f"구독: {cam['image_topic']}  TF: {self.world} → {self.optical}")

    def on_info(self, msg: CameraInfo) -> None:
        if self.profile is None:
            self.profile = from_camera_info(msg)
            if not self.profile.frame_id:
                self.profile = type(self.profile)(self.profile.width, self.profile.height,
                                                  self.profile.K, self.profile.dist, self.optical)
            self.get_logger().info(f"렌즈 값 수신: fx={self.profile.K[0, 0]:.1f} id={self.profile.calibration_id}")

    def lookup_T_W_C(self, stamp) -> np.ndarray | None:
        for when in (Time.from_msg(stamp), Time()):  # 영상 시각 → 없으면 최신
            try:
                tf = self.tf_buffer.lookup_transform(self.world, self.optical, when, timeout=Duration(seconds=0.05))
                q, t = tf.transform.rotation, tf.transform.translation
                return make_T(quat_to_rot(q.x, q.y, q.z, q.w), [t.x, t.y, t.z])
            except Exception as exc:  # noqa: BLE001 - TF 없음/시각 밖
                self.status = f"TF 대기: {type(exc).__name__}"

        return None

    def on_image(self, msg: Image) -> None:
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8").copy()

        if self.profile is None:
            self.latest = self.banner(img, self.status)
            return

        T_W_C = self.lookup_T_W_C(msg.header.stamp)

        if T_W_C is None:
            self.latest = self.banner(img, self.status)
            return

        t0 = time.perf_counter()
        results = []

        if self.detect:
            for cnt in simple_part_contours(img, self.profile, T_W_C, self.frames):
                results.append(compute_work_result(cnt, self.profile, T_W_C, self.specs))

        draw_guides(img, self.profile, T_W_C, self.frames)
        draw_results(img, self.profile, T_W_C, results)
        cam_xyz = T_W_C[:3, 3]
        self.status = (f"t={msg.header.stamp.sec}.{msg.header.stamp.nanosec // 1_000_000:03d}  "
                       f"cam W=({cam_xyz[0]:.3f}, {cam_xyz[1]:.3f}, {cam_xyz[2]:.3f}) m  "
                       f"parts={len(results)}  B {1000 * (time.perf_counter() - t0):.1f} ms")
        self.latest = self.banner(img, self.status)

        if self.pub is not None:
            out = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            out.header = msg.header
            self.pub.publish(out)

    @staticmethod
    def banner(img: np.ndarray, text: str) -> np.ndarray:
        cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(img, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="link", choices=["link", "gripper"])
    ap.add_argument("--publish", action="store_true", help="/b/center_overlay/image 발행")
    ap.add_argument("--no-detect", action="store_true", help="임시 검출 끄기(안내선만)")
    ap.add_argument("--scale", type=float, default=0.75, help="창 크기 배율")
    args = ap.parse_args()

    rclpy.init()
    node = BLiveView(args.camera, args.publish, not args.no_detect)
    win = "B live view (s: save, q: quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.03)

            if node.latest is not None:
                show = cv2.resize(node.latest, None, fx=args.scale, fy=args.scale)
                cv2.imshow(win, show)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s") and node.latest is not None:
                name = f"b_live_{time.strftime('%H%M%S')}.png"
                cv2.imwrite(name, node.latest)
                node.get_logger().info(f"저장: {name}")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
