"""Register static Gazebo SDF collision geometry in Moveit's Planning Scene.

Current scope: the conveyor belt and three static sorting bins from the active
sort_cell world. Dynamic parts and attachment state are managed separately.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive


DEFAULT_OBJECT_NAMES = ["conveyor", "bin_bolt_a", "bin_bolt_b", "bin_nut"]


def _numbers(text: str | None, count: int, default: Iterable[float]) -> tuple[float, ...]:
    if text is None or not text.strip():
        values = tuple(default)
    else:
        values = tuple(float(value) for value in text.split())
    if len(values) != count or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Expected {count} finite numbers, got {text!r}")
    return values


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _quat_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _rotate(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    x, y, z, w = quaternion
    vx, vy, vz = vector
    uv = (y * vz - z * vy, z * vx - x * vz, x * vy - y * vx)
    uuv = (y * uv[2] - z * uv[1], z * uv[0] - x * uv[2], x * uv[1] - y * uv[0])
    return (
        vx + 2.0 * (w * uv[0] + uuv[0]),
        vy + 2.0 * (w * uv[1] + uuv[1]),
        vz + 2.0 * (w * uv[2] + uuv[2]),
    )


def _compose(
    parent: tuple[tuple[float, float, float], tuple[float, float, float, float]],
    child: tuple[tuple[float, float, float], tuple[float, float, float, float]],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    parent_position, parent_rotation = parent
    child_position, child_rotation = child
    offset = _rotate(parent_rotation, child_position)
    return (
        tuple(parent_position[i] + offset[i] for i in range(3)),
        _quat_multiply(parent_rotation, child_rotation),
    )


def _pose_from_sdf(element: ET.Element | None):
    values = _numbers(element.text if element is not None else None, 6, (0, 0, 0, 0, 0, 0))
    return (values[:3], _quat_from_rpy(*values[3:]))


def _to_ros_pose(
    transform: tuple[tuple[float, float, float], tuple[float, float, float, float]],
) -> Pose:
    position, rotation = transform
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = position
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = rotation
    return pose


def _build_static_objects(
    world_sdf: Path,
    models_dir: Path,
    object_names: list[str],
    frame_id: str,
) -> list[CollisionObject]:
    world_root = ET.parse(world_sdf).getroot()
    world = world_root.find("world")
    if world is None:
        raise ValueError(f"No <world> element in {world_sdf}")

    objects: list[CollisionObject] = []
    found: set[str] = set()
    for include in world.findall("include"):
        object_id = include.findtext("name")
        if object_id not in object_names:
            continue
        uri = include.findtext("uri")
        if not uri or not uri.startswith("model://"):
            raise ValueError(f"Unsupported model URI for {object_id}: {uri!r}")
        model_name = uri.removeprefix("model://")
        model_file = models_dir / model_name / "model.sdf"
        model_root = ET.parse(model_file).getroot()
        model = model_root.find("model")
        if model is None:
            raise ValueError(f"No <model> element in {model_file}")
        if model.findtext("static", "false").strip().lower() != "true":
            raise ValueError(f"Requested model {object_id} is not static")

        include_transform = _pose_from_sdf(include.find("pose"))
        collision_object = CollisionObject()
        collision_object.header.frame_id = frame_id
        collision_object.id = object_id
        for link in model.findall("link"):
            link_transform = _compose(include_transform, _pose_from_sdf(link.find("pose")))
            for collision in link.findall("collision"):
                geometry = collision.find("geometry")
                box = geometry.find("box") if geometry is not None else None
                if box is None:
                    raise ValueError(
                        f"Unsupported non-box collision in {model_file}: {collision.get('name')}"
                    )
                dimensions = _numbers(box.findtext("size"), 3, ())
                if any(size <= 0.0 for size in dimensions):
                    raise ValueError(f"Box dimensions must be positive in {model_file}")

                primitive = SolidPrimitive()
                primitive.type = SolidPrimitive.BOX
                primitive.dimensions = list(dimensions)
                collision_transform = _compose(link_transform, _pose_from_sdf(collision.find("pose")))
                collision_object.primitives.append(primitive)
                collision_object.primitive_poses.append(_to_ros_pose(collision_transform))

        if not collision_object.primitives:
            raise ValueError(f"No supported collision geometry found for {object_id}")
        collision_object.operation = CollisionObject.ADD
        objects.append(collision_object)
        found.add(object_id)

    missing = sorted(set(object_names) - found)
    if missing:
        raise ValueError(f"Objects not found in {world_sdf}: {', '.join(missing)}")
    return objects


class PlanningSceneManager(Node):
    """Apply selected static SDF collision objects to MoveIt."""

    def __init__(self) -> None:
        super().__init__("planning_scene_manager")
        project_root = Path.home() / "EyetoArm"
        self.declare_parameter("world_sdf", str(project_root / "simulation/worlds/sort_cell.sdf"))
        self.declare_parameter("models_dir", str(project_root / "simulation/models"))
        self.declare_parameter("frame_id", "world")
        self.declare_parameter("apply_service", "/apply_planning_scene")
        self.declare_parameter("object_names", DEFAULT_OBJECT_NAMES)

    def apply_world(self) -> bool:
        world_sdf = Path(self.get_parameter("world_sdf").value).expanduser()
        models_dir = Path(self.get_parameter("models_dir").value).expanduser()
        frame_id = str(self.get_parameter("frame_id").value)
        object_names = list(self.get_parameter("object_names").value)
        objects = _build_static_objects(world_sdf, models_dir, object_names, frame_id)

        service_name = str(self.get_parameter("apply_service").value)
        client = self.create_client(ApplyPlanningScene, service_name)
        self.get_logger().info(f"Waiting for MoveIt service {service_name}")
        if not client.wait_for_service(timeout_sec=30.0):
            self.get_logger().error(f"MoveIt service unavailable: {service_name}")
            return False

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = objects
        request = ApplyPlanningScene.Request()
        request.scene = scene
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        if not future.done():
            self.get_logger().error("Timed out while applying the Planning Scene")
            return False
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(f"ApplyPlanningScene call failed: {exc}")
            return False
        if not response.success:
            self.get_logger().error("MoveIt rejected the Planning Scene update")
            return False

        primitive_count = sum(len(item.primitives) for item in objects)
        self.get_logger().info(
            f"Applied {len(objects)} static objects ({primitive_count} box primitives) "
            f"from {world_sdf} in frame {frame_id}"
        )
        return True


def main(args=None) -> int:
    rclpy.init(args=args)
    node = PlanningSceneManager()
    try:
        return 0 if node.apply_world() else 1
    except (OSError, ET.ParseError, ValueError) as exc:
        node.get_logger().error(f"Could not build Planning Scene objects: {exc}")
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
