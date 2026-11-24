"""Gazebo part supply for managed moving-belt sort runs."""
from gz.msgs10.double_pb2 import Double
import math
import random
from pathlib import Path
import subprocess
import time
from uuid import uuid4

from threading import Lock

from gz.transport13 import Node
from gz.msgs10.pose_v_pb2 import Pose_V



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


class GazeboPartFeeder:
    def __init__(self, root, speed=0.012, interval=20.0):
        self.models = Path(root) / "simulation" / "models"
        for model, _ in FEED_SEQUENCE:
            if not (self.models / model / "model.sdf").is_file():
                raise FileNotFoundError(model)
        self.inlet = InletMonitor()
        self.belt = self.inlet.node.advertise("/model/conveyor/link/link/track_cmd_vel", Double)
        self.speed = speed
        self.interval = interval
        self.run = uuid4().hex
        self.index = 0
        self.previous = None
        self.next_spawn = 0.0

    def set_running(self, running):
        if not self.belt.publish(Double(data=self.speed if running else 0.0)):
            raise RuntimeError("Gazebo belt command publish failed")

    def tick(self):
        if time.monotonic() < self.next_spawn or not self.inlet.clear(self.previous):
            return
        model, height = FEED_SEQUENCE[self.index % len(FEED_SEQUENCE)]
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
        if model.startswith("bolt_"):
            # Lay the shaft flat, then turn it about the world's vertical axis.
            half_yaw = random.random() * math.pi
            s = math.sin(half_yaw) / math.sqrt(2)
            c = math.cos(half_yaw) / math.sqrt(2)
            qx, qy, qz, qw = -s, c, s, c
        name = f"part_supply_{self.run}_{self.index}"
        sdf = self.models / model / "model.sdf"
        request = (f'sdf_filename: "{sdf}" name: "{name}" '
                   f'pose {{position {{x: {X} y: {Y} z: {height}}} '
                   f'orientation {{x: {qx} y: {qy} z: {qz} w: {qw}}}}}')
        reply = gz("service", "-s", "/world/sort_cell/create", "--reqtype",
                   "gz.msgs.EntityFactory", "--reptype", "gz.msgs.Boolean",
                   "--timeout", "5000", "--req", request)
        if "data: true" not in reply:
            raise RuntimeError(f"Gazebo create failed: {reply.strip()}")
        self.index += 1
        self.previous = name
        self.next_spawn = time.monotonic() + self.interval
