# C video relay and B input receivers

`display_relay` sends the latest changed ETV1/JPEG frame over TCP `47092`. It accepts Gazebo raw cameras, A's current AI detect frames, and B's center-overlay image. It is separate from command/status TCP (`47090`).

| Input | Receive path | Relay representation |
| --- | --- | --- |
| Gazebo link / gripper / fixed cameras | ROS `sensor_msgs/Image` | `link/raw`, `gripper/raw`, `fixed/raw` |
| A AI detect frame | AToCHandoff consumer in `run_pipeline.py` → local TCP `127.0.0.1:47093` | `link/detect` |
| B center-overlay frame + identity | ROS eyetoarm_interfaces/msg/CenterOverlay, default /b/center_overlay (--center-topic overrides it); image and B frame keys arrive together | link/center over ETV1/JPEG |
| B measurement results | ROS `std_msgs/String` JSON on `/b/results`, subscribed only while `center` is selected | Exact-key selected result embedded as optional ETV1 metadata on its center frame |

The center subscription consumes CenterOverlay so the JPEG and B frame identity come from the same ROS message. C also subscribes to `/b/results` while `center` is selected and joins the selected B result to the latest center JPEG using the complete B frame key (run/source/image sequence/generations/job sequence/capture stamp/time base/calibration). The result may arrive before or after the image; C retains a bounded cache and forwards only an exact match. ETV1 carries the selected result as optional metadata, with the C relay session ID for D's session guard and B's original run ID as `producer_run_id`. A detect ETV1 frame now also preserves its capture stamp and time base. This remains prototype behavior pending C↔D contract approval and live integration confirmation.

The joined B result contains display measurements and classification only; it is not a robot command. CenterOverlay contains no grasp/place coordinates; those remain in `/sort_part`. The center stream is display-only and does not authorize a robot action.

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

`sort_goal_receiver`, `scripts/b_sim_sort_part_server.py`, and the B fake receiver guard `/sort_part` at startup. The guard checks for an existing action server in the ROS graph and holds a per-`ROS_DOMAIN_ID` local lock while the server is alive. If another server is found, startup exits with an error; run only one `/sort_part` server at a time. Rebuild `robot_ws` after updating the package before using the guard. Build and duplicate-start checks passed on 2026-11-23; simultaneous starts from different hosts are not atomically coordinated.

With the managed options used by `scripts/c_stack.launch.py auto_sort:=true`, the receiver exposes ROS `Trigger` services `~/start_run`, `~/stop_run`, `~/e_stop`, and `~/recover`, plus `~/get_run`. The TCP server also accepts dashboard TCP 1.2 run-control requests and forwards them to these ROS services. START starts Gazebo conveyor movement, bolt spawning, and managed B→C goals. E-STOP stops the belt/bolt spawner and cancels the current robot goal at its current joint pose; it preserves the managed run ID so START can reactivate that run. STOP also homes the arm and asks the A/B pipeline to discard queued detections and invalidate its displayed detect/center frames; the next START creates a new run. RECOVER is accepted only while the E-STOP latch is set; if the gripper controller reported a held object, it opens the gripper at the current pose before homing. It then discards A/B data and remains paused. TCP 1.1 remains available for the existing status and MOVE/CANCEL paths. TCP 1.2 run snapshots are synthesized by the TCP process from `get_run`; the current ROS interface does not report gripper hold state, so E-STOP is represented as `HOLD_UNKNOWN`. E-STOP confirmation reflects the software ROS service result, not a safety-rated hardware stop. The in-flight SortPart trajectory is canceled on E-STOP; restarting does not continue the same joint trajectory checkpoint.

The relay listens on `0.0.0.0:47092` for D and `127.0.0.1:47093` for the local A handoff. The B center topic can be changed with `--center-topic` if B publishes it under another agreed name.

## Boundary

A live test confirmed B can publish CenterOverlay and C can receive its image and identity fields. The code now joins `/b/results` to matching center frames and D can consume an optional result from the ETV1 envelope. This code path has not been runtime-verified as part of the current change. A live test with D's Windows GUI and repeated raw/detect/center switching remains to be done.
