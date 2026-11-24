"""Gazebo sort_cell test feeder. Run once per world, with the simulation playing."""
import argparse
import math
import random
from pathlib import Path
import subprocess
import time
from uuid import uuid4

from threading import Lock

from gz.transport13 import Node
from gz.msgs10.pose_v_pb2 import Pose_V


MODELS = Path(__file__).resolve().parents[2] / "simulation" / "models"
X, Y = 0.55, -0.55
# Model, spawn height in metres.
FEED_SEQUENCE = (
    ("bolt_m3x20", 0.108),
    ("bolt_m6x40", 0.108),
    ("nut_m6", 0.1005),
)


def gz(*args):
    result = subprocess.run(["gz", *args], capture_output=True, text=True,
                            timeout=10, check=True)
    return result.stdout


class InletMonitor:
    def __init__(self):
        self.parts = {}
        self.lock = Lock()
        self.received = None
        self.node = Node()
        if not self.node.subscribe(Pose_V, "/world/sort_cell/pose/info", self.on_poses):
            raise RuntimeError("Cannot subscribe to Gazebo pose/info")

    def on_poses(self, message):
        positions = {p.name: (p.position.x, p.position.y, p.position.z)
                     for p in message.pose if p.name.startswith(("part_", "feed_test_"))}
        if positions:
            with self.lock:
                self.parts.update(positions)
                self.received = time.monotonic()

    def clear(self, previous):
        with self.lock:
            if self.received is None or time.monotonic() - self.received > 2:
                return False
            if previous and previous not in self.parts:
                return False
            return all(abs(x - X) > 0.06 or abs(y - Y) > 0.06 or z > 0.17
                       for x, y, z in self.parts.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=6, help="0 = continuous")
    parser.add_argument("--interval", type=float, default=3, help="wall seconds")
    args = parser.parse_args()
    if args.count < 0 or not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("use count >= 0 and interval > 0")
    inlet = InletMonitor()
    run = uuid4().hex[:10]
    previous = None
    index = 0
    while not args.count or index < args.count:
        if not inlet.clear(previous):
            print("Waiting for the belt inlet to clear", flush=True)
            time.sleep(1)
            continue
        model, height = FEED_SEQUENCE[index % len(FEED_SEQUENCE)]
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
        if model.startswith("bolt_"):
            # Lay the shaft flat, then turn it about the world's vertical axis.
            half_yaw = random.random() * math.pi
            s = math.sin(half_yaw) / math.sqrt(2)
            c = math.cos(half_yaw) / math.sqrt(2)
            qx, qy, qz, qw = -s, c, s, c
        name = f"feed_test_{run}_{index}"
        sdf = MODELS / model / "model.sdf"
        request = (f'sdf_filename: "{sdf}" name: "{name}" '
                   f'pose {{position {{x: {X} y: {Y} z: {height}}} '
                   f'orientation {{x: {qx} y: {qy} z: {qz} w: {qw}}}}}')
        reply = gz("service", "-s", "/world/sort_cell/create", "--reqtype",
                   "gz.msgs.EntityFactory", "--reptype", "gz.msgs.Boolean",
                   "--timeout", "5000", "--req", request)
        if "data: true" not in reply:
            raise SystemExit(f"Gazebo create failed: {reply.strip()}")
        index += 1
        previous = name
        print(f"{index}: {model} -> {name}", flush=True)
        if index != args.count:
            time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
