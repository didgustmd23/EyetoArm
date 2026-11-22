#!/usr/bin/env python3
"""B accuracy probe: move a bolt to several poses in Gazebo, record B /sort_part goals, compare to Gazebo truth.

The robot does NOT move. This node is a stand-in /sort_part server that only logs each goal and aborts it.
Run the live A->B pipeline with --live-sort-part at the same time.

    PYTHONPATH=src:$PYTHONPATH python3 scripts/b_accuracy_probe.py --out logs/b_accuracy.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import threading
import time

import numpy as np

# model origin -> shaft middle along the head->tip axis (m). From simulation/models/*/model.sdf
SHAFT_MID = {"part_01": 0.020, "part_00": 0.010}
SPEC = {"part_01": "BOLT_B", "part_00": "BOLT_A"}

# (x, y, yaw_deg) poses on the belt. Nut sits at (0.55, 0.00): keep away from it.
POSES = [(0.55, -0.12, 0), (0.55, 0.10, 0), (0.55, 0.25, 0), (0.55, 0.40, 0),
         (0.52, 0.15, 45), (0.58, 0.15, 90), (0.55, 0.30, 135), (0.53, -0.10, 60)]


def quat_lying(yaw_deg: float) -> tuple[float, float, float, float]:
    """(w, x, y, z) for a bolt lying on the belt: pitch 90 deg, then yaw about world Z."""
    h = math.radians(yaw_deg) / 2.0
    c = math.sqrt(0.5)
    return (math.cos(h) * c, -math.sin(h) * c, math.cos(h) * c, math.sin(h) * c)


def rot(w, x, y, z) -> np.ndarray:
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def truth(position, quat_wxyz, part: str):
    """Gazebo pose -> (shaft middle xyz m, axis yaw deg in [0,180))."""
    axis = rot(*quat_wxyz) @ np.array([0.0, 0.0, -1.0])          # head -> tip
    mid = np.asarray(position) + axis * SHAFT_MID[part]
    return mid, math.degrees(math.atan2(axis[1], axis[0])) % 180.0


def set_pose(world: str, part: str, x: float, y: float, z: float, q) -> bool:
    w, qx, qy, qz = q
    req = (f'name: "{part}", position: {{x: {x}, y: {y}, z: {z}}}, '
           f'orientation: {{x: {qx}, y: {qy}, z: {qz}, w: {w}}}')
    out = subprocess.run(["gz", "service", "-s", f"/world/{world}/set_pose", "--reqtype", "gz.msgs.Pose",
                          "--reptype", "gz.msgs.Boolean", "--timeout", "2000", "--req", req],
                         capture_output=True, text=True)
    return "true" in out.stdout


class Truth:
    def __init__(self, world: str, part: str):
        from gz.msgs10.pose_v_pb2 import Pose_V
        from gz.transport13 import Node
        self.part, self.lock, self.pose, self.node = part, threading.Lock(), None, Node()
        self.node.subscribe(Pose_V, f"/world/{world}/pose/info", self._on)

    def _on(self, msg):
        for p in msg.pose:
            if p.name == self.part:
                with self.lock:
                    self.pose = ((p.position.x, p.position.y, p.position.z),
                                 (p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z))

    def get(self):
        with self.lock:
            return self.pose


def angle_diff(a: float, b: float) -> float:
    d = (a - b) % 180.0
    return min(d, 180.0 - d)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="B accuracy probe (robot does not move)")
    ap.add_argument("--part", default="part_01", choices=sorted(SHAFT_MID))
    ap.add_argument("--world", default="sort_cell")
    ap.add_argument("--settle", type=float, default=1.5, help="seconds to wait after moving the bolt")
    ap.add_argument("--dwell", type=float, default=4.0, help="seconds to collect goals per pose")
    ap.add_argument("--out", default="logs/b_accuracy.csv")
    args = ap.parse_args(argv)

    import rclpy
    from rclpy.action import ActionServer
    from rclpy.executors import MultiThreadedExecutor
    from eyetoarm_interfaces.action import SortPart

    goals, glock = [], threading.Lock()

    def execute(handle):
        g = handle.request
        if g.grasp_pose_candidates:
            c = g.grasp_pose_candidates[0]
            with glock:
                goals.append((time.monotonic(), g.part_class, tuple(c.position_mm), (c.closing_yaw_deg - 90.0) % 180.0))
        handle.abort()
        res = SortPart.Result()
        res.status, res.reason = "FAILED", "PROBE_ONLY"
        return res

    rclpy.init()
    node = rclpy.create_node("b_accuracy_probe")
    ActionServer(node, SortPart, "/sort_part", execute_callback=execute)
    ex = MultiThreadedExecutor(num_threads=2)
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    tr = Truth(args.world, args.part)
    print(f"probe: /sort_part stand-in ready, part={args.part}. Start run_pipeline --live-sort-part now.", flush=True)
    end = time.monotonic() + 120.0
    while time.monotonic() < end and not goals:      # wait until the pipeline is up and sending
        time.sleep(0.5)
    print(f"probe: first goal {'received' if goals else 'NOT received in 120 s'}; moving the bolt", flush=True)

    rows = []
    for i, (x, y, yaw) in enumerate(POSES):
        ok = set_pose(args.world, args.part, x, y, 0.105, quat_lying(yaw))
        time.sleep(args.settle)
        t0 = time.monotonic()
        time.sleep(args.dwell)
        pose = tr.get()
        with glock:
            got = [g for g in goals if g[0] >= t0]
        if not ok or pose is None or not got:
            print(f"[{i + 1}] pose=({x:.2f},{y:.2f},{yaw}deg) set={ok} goals={len(got)} -> no data", flush=True)
            rows.append({"pose": i + 1, "x_set": x, "y_set": y, "yaw_set": yaw, "n_goals": len(got)})
            continue
        mid, t_yaw = truth(*pose, args.part)
        b = np.median(np.array([g[2] for g in got]), axis=0) / 1000.0
        b_yaw = float(np.median([g[3] for g in got]))
        classes = sorted({g[1] for g in got})
        dx, dy, dz = (b - mid) * 1000.0
        dxy = math.hypot(dx, dy)
        dyaw = angle_diff(b_yaw, t_yaw)
        print(f"[{i + 1}] true=({mid[0]*1000:.1f},{mid[1]*1000:.1f},{mid[2]*1000:.1f})mm {t_yaw:.0f}deg | "
              f"B=({b[0]*1000:.1f},{b[1]*1000:.1f},{b[2]*1000:.1f})mm {b_yaw:.0f}deg {classes} | "
              f"err xy={dxy:.1f}mm z={dz:+.1f}mm yaw={dyaw:.1f}deg n={len(got)}", flush=True)
        rows.append({"pose": i + 1, "x_set": x, "y_set": y, "yaw_set": yaw, "n_goals": len(got),
                     "spec_expected": SPEC[args.part], "spec_b": "/".join(classes),
                     "true_x_mm": round(mid[0] * 1000, 1), "true_y_mm": round(mid[1] * 1000, 1),
                     "true_z_mm": round(mid[2] * 1000, 1), "true_yaw_deg": round(t_yaw, 1),
                     "b_x_mm": round(b[0] * 1000, 1), "b_y_mm": round(b[1] * 1000, 1),
                     "b_z_mm": round(b[2] * 1000, 1), "b_yaw_deg": round(b_yaw, 1),
                     "err_xy_mm": round(dxy, 1), "err_z_mm": round(dz, 1), "err_yaw_deg": round(dyaw, 1)})

    keys = ["pose", "x_set", "y_set", "yaw_set", "n_goals", "spec_expected", "spec_b", "true_x_mm", "true_y_mm",
            "true_z_mm", "true_yaw_deg", "b_x_mm", "b_y_mm", "b_z_mm", "b_yaw_deg", "err_xy_mm", "err_z_mm", "err_yaw_deg"]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    errs = [r["err_xy_mm"] for r in rows if "err_xy_mm" in r]
    if errs:
        print(f"summary: {len(errs)}/{len(rows)} poses, xy error mean {np.mean(errs):.1f} mm, max {np.max(errs):.1f} mm "
              f"-> {args.out}", flush=True)
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
