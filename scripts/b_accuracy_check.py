"""B 정확도 확인 — 링크 카메라로 부품을 재서 시뮬레이션의 실제 자리와 비교한 오차 표를 만든다.

하는 일
1. 카메라 영상 + 영상 시각의 TF(`world` → 카메라 광학 프레임)를 받는다. 기본 링크 카메라, `--camera gripper`로 그리퍼 카메라
2. 벨트 위 밝은 물체를 임시로 찾아(운영 검출은 A 담당) B 계산 → 중심·각도·길이·규격
3. `simulation/worlds/sort_cell.sdf`의 부품 자리(정답)와 비교 → 프레임마다 기록
4. 정한 프레임 수를 모으면 오차 표를 출력하고 CSV·표·사진을 `outputs/b_accuracy/`에 저장

실행 (WSL, 시뮬레이션을 켠 뒤 새 터미널, 저장소 맨 위 폴더에서):
    source /opt/ros/jazzy/setup.bash
    PYTHONPATH=src:$PYTHONPATH python3 scripts/b_accuracy_check.py            # 30프레임
    PYTHONPATH=src:$PYTHONPATH python3 scripts/b_accuracy_check.py --frames 60 --tag watch34
    PYTHONPATH=src:$PYTHONPATH python3 scripts/b_accuracy_check.py --camera gripper       # 그리퍼 카메라
    # 창: 초록 + = 정답 자리, 색 화살표 = B 결과. q = 그만 모으고 표 출력

주의: 벨트가 아직 움직이지 않을 때(부품이 처음 자리에 있을 때)만 정답이 맞다.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
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
from eyetoarm.transforms.accuracy import format_table, load_truth, match, summarize, write_csv
from eyetoarm.transforms.overlay import draw_guides, draw_results, simple_part_contours
from eyetoarm.transforms.planar_transform import make_T, quat_to_rot, w_to_image
from eyetoarm.transforms.work_result import compute_work_result, load_specs

ROOT = Path(__file__).resolve().parents[1]
STATUS_EN = {"1/3": "1/3 waiting camera_info", "2/3": "2/3 waiting image", "3/3": "3/3 waiting TF at capture time"}


class BAccuracy(Node):
    def __init__(self, want_frames: int, camera_key: str = "link") -> None:
        super().__init__("b_accuracy_check", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.frames = yaml.safe_load(open(ROOT / "config/robot/frames.yaml", encoding="utf-8"))
        self.specs = load_specs(ROOT / "config/calibration/part_specs.json")
        self.truth = load_truth(ROOT / "simulation/worlds/sort_cell.sdf")
        cam = self.frames["cameras"][camera_key]
        self.camera_key = camera_key
        self.world, self.optical = self.frames["world"]["frame_id"], cam["optical_frame"]
        self.bridge = CvBridge()
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.profile = None
        self.want = want_frames
        self.done = 0
        self.rows: list[dict] = []
        self.extra = 0
        self.latest = None
        self.last_T = None
        self.images = 0          # 받은 영상 수
        self.tf_miss = 0         # 찍은 시각 TF가 없어 건너뛴 수
        self.status = f"1/3 렌즈 값({cam['camera_info_topic']}) 기다리는 중"
        self.create_subscription(CameraInfo, cam["camera_info_topic"], self.on_info, 10)
        self.create_subscription(Image, cam["image_topic"], self.on_image, 2)
        self.get_logger().info("정답 부품: " + ", ".join(f"{t.name}({t.spec}) "
                                                         f"({t.xy_m[0] * 1000:.1f}, {t.xy_m[1] * 1000:.1f}) mm"
                                                         for t in self.truth))

    def on_info(self, msg: CameraInfo) -> None:
        if self.profile is None:
            p = from_camera_info(msg)
            self.profile = p if p.frame_id else type(p)(p.width, p.height, p.K, p.dist, self.optical)
            self.get_logger().info(f"렌즈 값 수신: fx={self.profile.K[0, 0]:.1f}")
            self.status = f"2/3 영상({self.frames['cameras'][self.camera_key]['image_topic']}) 기다리는 중"

    def lookup(self, stamp) -> np.ndarray | None:
        """찍은 시각의 TF. 그 시각 TF가 아직 안 왔으면 최신 TF를 쓰되, 찍은 시각과 50 ms 안일 때만 쓴다.

        팔이 멈춰 있으면 50 ms 차이는 위치에 영향이 없다. 더 차이 나면 이 프레임은 건너뛴다.
        """
        want = Time.from_msg(stamp)

        try:
            tf = self.tf_buffer.lookup_transform(self.world, self.optical, want, timeout=Duration(seconds=0.1))
            self.tf_mode = "찍은 시각"
        except Exception as exc:  # noqa: BLE001 - 그 시각 TF 없음 → 최신 TF 확인
            self.tf_error = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"

            try:
                tf = self.tf_buffer.lookup_transform(self.world, self.optical, Time())
            except Exception as exc2:  # noqa: BLE001 - TF가 아예 없음
                self.tf_error = f"{type(exc2).__name__}: {str(exc2).splitlines()[0][:160]}"
                return None

            got = Time.from_msg(tf.header.stamp)
            gap_ms = abs((want - got).nanoseconds) / 1e6
            self.tf_gap_ms = gap_ms

            if gap_ms > 50.0:
                self.tf_error += f" / 최신 TF와 {gap_ms:.0f} ms 차이"
                return None

            self.tf_mode = f"최신 TF({gap_ms:.0f} ms 차이)"

        q, t = tf.transform.rotation, tf.transform.translation

        return make_T(quat_to_rot(q.x, q.y, q.z, q.w), [t.x, t.y, t.z])

    def on_image(self, msg: Image) -> None:
        self.images += 1

        if self.profile is None or self.done >= self.want:
            return

        T_W_C = self.lookup(msg.header.stamp)

        if T_W_C is None:
            self.tf_miss += 1
            self.status = (f"3/3 찍은 시각 TF 기다리는 중 (영상 {self.images}장, TF 없어 건너뜀 {self.tf_miss})"
                           f"\n       이유: {self.tf_error}")
            return

        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8").copy()
        results = [compute_work_result(c, self.profile, T_W_C, self.specs)
                   for c in simple_part_contours(img, self.profile, T_W_C, self.frames)]
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

        for row in match(results, self.truth):
            if row["truth"] is None:
                self.extra += 1
                continue

            row.update(frame=self.done, stamp_ns=stamp_ns)
            self.rows.append(row)

        self.done += 1
        self.last_T = T_W_C
        draw_guides(img, self.profile, T_W_C, self.frames)
        draw_results(img, self.profile, T_W_C, results)
        self.draw_truth(img, T_W_C)
        cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(img, f"B accuracy  frame {self.done}/{self.want}  parts={len(results)}  (green + = truth)",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        self.latest = img

    def draw_truth(self, img: np.ndarray, T_W_C: np.ndarray) -> None:
        pts = np.array([[t.xy_m[0], t.xy_m[1], self.frames["part_plane_z"]] for t in self.truth])
        uv = w_to_image(pts, self.profile.K, self.profile.dist, T_W_C)

        for (u, v), t in zip(uv, self.truth):
            if np.isfinite(u) and np.isfinite(v):
                cv2.drawMarker(img, (int(round(u)), int(round(v))), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
                cv2.putText(img, t.name, (int(u) + 8, int(v) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)


def waiting_image(text: str) -> np.ndarray:
    img = np.full((240, 900, 3), 30, np.uint8)
    cv2.putText(img, "B accuracy - waiting", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text.encode("ascii", "replace").decode(), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1)
    cv2.putText(img, "see terminal for details", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="link", choices=["link", "gripper"], help="쓸 카메라 (기본 link)")
    ap.add_argument("--frames", type=int, default=30, help="모을 프레임 수 (기본 30)")
    ap.add_argument("--tag", default="", help="저장 파일 이름에 붙일 말 (예: watch34)")
    ap.add_argument("--no-window", action="store_true", help="창 없이 모으기만")
    ap.add_argument("--scale", type=float, default=0.75, help="창 크기 배율")
    args = ap.parse_args()

    rclpy.init()
    node = BAccuracy(args.frames, args.camera)
    win = "B accuracy (q: stop)"

    if not args.no_window:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    last_print = 0.0
    shown = None

    try:
        while rclpy.ok() and node.done < node.want:
            rclpy.spin_once(node, timeout_sec=0.03)

            if node.latest is None and time.monotonic() - last_print > 2.0 and node.status != shown:
                print("[대기] " + node.status, flush=True)       # 한글 상태는 터미널에
                shown, last_print = node.status, time.monotonic()

            if not args.no_window and node.latest is None:
                cv2.imshow(win, waiting_image(STATUS_EN.get(node.status[:3], node.status)))

                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            if not args.no_window and node.latest is not None:
                h, w = node.latest.shape[:2]
                cv2.imshow(win, cv2.resize(node.latest, (int(w * args.scale), int(h * args.scale))))

                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
    except KeyboardInterrupt:
        pass

    summary = summarize(node.rows, node.truth)
    table = format_table(summary, node.extra)
    cam = node.last_T[:3, 3] if node.last_T is not None else (float("nan"),) * 3
    head = (f"B 정확도 확인 {datetime.now():%Y-%m-%d %H:%M}  카메라 {args.camera}  프레임 {node.done}개  TF {node.tf_mode}  "
            f"카메라 W=({cam[0]:.3f}, {cam[1]:.3f}, {cam[2]:.3f}) m  부품 높이 기준 {node.frames['part_plane_z']} m")
    print("\n" + head + "\n\n" + table + "\n")

    out = ROOT / "outputs/b_accuracy"
    name = f"{datetime.now():%Y%m%d_%H%M%S}_{args.camera}" + (f"_{args.tag}" if args.tag else "")
    write_csv(node.rows, out / f"{name}.csv")
    (out / f"{name}.md").write_text(head + "\n\n" + table + "\n", encoding="utf-8")

    if node.latest is not None:
        cv2.imwrite(str(out / f"{name}.png"), node.latest)

    print(f"저장: {out / name}.csv / .md / .png")
    node.destroy_node()

    if rclpy.ok():          # Ctrl+C로 끝내면 rclpy가 이미 꺼져 있다
        rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
