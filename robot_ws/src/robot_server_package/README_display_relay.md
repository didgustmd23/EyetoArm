# C raw camera relay

`display_relay` subscribes to three Gazebo image topics and sends the newest changed JPEG frame from each over TCP port `47092` using the prototype `ETV1` envelope. It is separate from command/status TCP (`47090`) and the legacy D-only single-image relay (`47091`).

| Channel | ROS topic | Source ID |
| --- | --- | --- |
| `link` | `/link_cam/image` | `gz_link` |
| `gripper` | `/gripper_cam/image` | `gz_gripper` |
| `fixed` (top view) | `/overview_cam/image` | `gz_fixed` |

## Run in WSL

Launch the simulation, which starts the ROS/Gazebo bridges, in one ROS-sourced terminal. In another terminal (build the package first if `robot_ws/install` is not present or the entry point has not been installed):

```bash
cd ~/EyetoArm/robot_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select robot_server_package
```

Then source the workspace and run the relay:

```bash
cd ~/EyetoArm
source /opt/ros/jazzy/setup.bash
source robot_ws/install/setup.bash
ros2 run robot_server_package display_relay
```

The relay listens on `0.0.0.0:47092` by default. `--host` and `--port` can override the bind address and port. The command/status server and this image relay are separate processes.

## Current verification boundary

A Windows TCP test client received packets for all three channels. JPEG byte decoding still needs a corrected signature check or an image decoder. D's `window.py` does not currently receive this `ETV1` stream; its existing `--view camera` mode uses the legacy 47091 `EAV1` stream from `/appearance_check/image`. D GUI integration, A detect handoff, B center frames, result/invalidation events, mode requests and reconnect/session synchronization are not implemented in this relay prototype. The 47092 port and `ETV1` packet are implementation details pending C/D agreement, not an approved interface contract.