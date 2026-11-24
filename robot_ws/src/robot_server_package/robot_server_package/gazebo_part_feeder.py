"""Gazebo part supply for managed moving-belt sort runs."""
from gz.msgs10.double_pb2 import Double
import math
import random
import logging
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

# World-space bin centers and interior half-extents. Bolt bins are rotated 90°
# in sort_cell.sdf, so their X/Y half-extents are swapped from the model frame.
BIN_REGIONS = (
    ((0.55, 0.70), (0.075, 0.055), 5.0),   # orange bin_nut
    ((0.72, -0.05), (0.055, 0.075), 10.0), # blue bin_bolt_a
    ((0.72, 0.17), (0.055, 0.075), 10.0),  # green bin_bolt_b
)
BIN_FLOOR_Z = 0.003
BIN_TOP_Z = 0.075
LOG = logging.getLogger(__name__)


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

    def snapshot(self):
        with self.lock:
            return dict(self.parts)


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
        self.bin_entered = {}

    def set_running(self, running):
        if not self.belt.publish(Double(data=self.speed if running else 0.0)):
            raise RuntimeError("Gazebo belt command publish failed")

    def _remove(self, name):
        request = f'name: "{name}" type: MODEL'
        try:
            reply = gz("service", "-s", "/world/sort_cell/remove", "--reqtype",
                       "gz.msgs.Entity", "--reptype", "gz.msgs.Boolean",
                       "--timeout", "5000", "--req", request)
        except (subprocess.SubprocessError, OSError) as exc:
            LOG.warning("Gazebo could not remove expired part %s: %s", name, exc)
            return False
        if "data: true" not in reply:
            LOG.warning("Gazebo refused to remove expired part %s: %s", name, reply.strip())
            return False
        with self.inlet.lock:
            self.inlet.parts.pop(name, None)
        return True

    def tick(self, spawn_enabled=True):
        now = time.monotonic()
        # Only feeder-created parts are eligible. Start the clock when a part
        # enters a sorting bin, and reset it if the part leaves.
        for name, (x, y, z) in self.inlet.snapshot().items():
            if not name.startswith(f"part_supply_{self.run}_"):
                continue
            delay = next((delay for (cx, cy), (hx, hy), delay in BIN_REGIONS
                          if abs(x - cx) <= hx and abs(y - cy) <= hy
                          and BIN_FLOOR_Z <= z <= BIN_TOP_Z), None)
            if delay is None:
                self.bin_entered.pop(name, None)
                continue
            entry = self.bin_entered.get(name)
            if entry is None or entry[1] != delay:
                entry = (now, delay)
                self.bin_entered[name] = entry
            entered = entry[0]
            if now - entered >= delay and self._remove(name):
                self.bin_entered.pop(name, None)

        if not spawn_enabled or now < self.next_spawn or not self.inlet.clear(self.previous):
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
