# C video relay and B input receivers

`display_relay` sends the latest changed ETV1/JPEG frame over TCP `47092`. It accepts Gazebo raw cameras, A's current AI detect frames, and B's center-overlay image. It is separate from command/status TCP (`47090`).

| Input | Receive path | Relay representation |
| --- | --- | --- |
| Gazebo link / gripper / fixed cameras | ROS `sensor_msgs/Image` | `link/raw`, `gripper/raw`, `fixed/raw` |
| A AI detect frame | AToCHandoff consumer in `run_pipeline.py` → local TCP `127.0.0.1:47093` | `link/detect` |
| B center-overlay frame | ROS `sensor_msgs/Image`, default `/b/center_overlay/image` (`--center-topic` overrides it) | `link/center` |

The center-image subscription assigns C-side display sequence metadata because a plain ROS `Image` does not carry the full B frame identity. It is for video display; it does not synchronize a center image with a B sort goal or authorize a robot action.

`sort_goal_receiver` provides the B→C ROS action server `/sort_part` using `eyetoarm_interfaces/action/SortPart`. It logs received goal fields and acknowledges receipt. Until the robot execution backend is connected, it returns `FAILED / EXECUTION_NOT_CONFIGURED`; it does not move the robot or report a sort as successful.

## Run in WSL

Build and source the workspace:

```bash
cd ~/EyetoArm/robot_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select eyetoarm_interfaces robot_server_package
source install/setup.bash
```

Run the video relay in one terminal:

```bash
cd ~/EyetoArm
source /opt/ros/jazzy/setup.bash
source robot_ws/install/setup.bash
ros2 run robot_server_package display_relay
```

Run B's action receiver in another terminal:

```bash
cd ~/EyetoArm
source /opt/ros/jazzy/setup.bash
source robot_ws/install/setup.bash
ros2 run robot_server_package sort_goal_receiver
```

The relay listens on `0.0.0.0:47092` for D and `127.0.0.1:47093` for the local A handoff. The B center topic can be changed with `--center-topic` if B publishes it under another agreed name.

## Boundary

The receiver and relay paths are implemented in C, but need a live integrated run to confirm B's publisher uses the agreed topic and message type. The center feed does not yet preserve B's run/image/generation identifiers. The action receiver is receipt-only; planning, execution, cancellation-to-stop behavior, and success/arrival reporting remain outside this receiver step. D's GUI selection signal must be connected by D so C can send `raw`, `detect`, or `center` on request.
