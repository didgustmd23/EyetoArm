# C video relay and B input receivers

`display_relay` sends the latest changed ETV1/JPEG frame over TCP `47092`. It accepts Gazebo raw cameras, A's current AI detect frames, and B's center-overlay image. It is separate from command/status TCP (`47090`).

| Input | Receive path | Relay representation |
| --- | --- | --- |
| Gazebo link / gripper / fixed cameras | ROS `sensor_msgs/Image` | `link/raw`, `gripper/raw`, `fixed/raw` |
| A AI detect frame | AToCHandoff consumer in `run_pipeline.py` → local TCP `127.0.0.1:47093` | `link/detect` |
| B center-overlay frame + identity | ROS eyetoarm_interfaces/msg/CenterOverlay, default /b/center_overlay (--center-topic overrides it); image and B frame keys arrive together | link/center over ETV1/JPEG |

The center subscription consumes CenterOverlay so the JPEG and B frame identity come from the same ROS message. ETV1 preserves B source_id, image_seq, capture_generation, generation, job_seq, capture_stamp_ns, timestamp_base, and calibration_id; it adds the C relay session ID for D's session guard and carries B's original run ID as producer_run_id. D's current receiver uses the standard display envelope and does not surface optional producer fields to the GUI. CenterOverlay contains no grasp/place coordinates; those remain in /sort_part. The center stream is display-only and does not authorize a robot action.

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

A live test confirmed B can publish CenterOverlay and C can receive its image and identity fields. C now forwards those fields with the center JPEG over ETV1. A live test with D's Windows GUI and repeated raw/detect/center switching remains to be done.
