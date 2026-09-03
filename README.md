# openxr_ros2_bridge

ARISE reusable module A (Tele-Craft experiment). Turns OpenXR controller poses into
ROS 2 teleoperation commands. Robot-agnostic: the output is a smoothed hand pose and
velocity in a REP-103 frame plus a hold-to-run flag; the robot mapping lives elsewhere.

License: Apache-2.0.

## Nodes

### `openxr_ros2_bridge_node`

| | Topic | Type | Notes |
|---|---|---|---|
| in | `/openxr/<hand>/pose` | `geometry_msgs/PoseStamped` | OpenXR stage space: +X right, +Y up, −Z forward, metres |
| in | `/openxr/<hand>/inputs` | `sensor_msgs/Joy` | `axes[0]` = trigger (hold-to-run) |
| out | `~/pose` | `geometry_msgs/PoseStamped` | smoothed pose in `output_frame`, REP-103 (X fwd, Y left, Z up) |
| out | `~/twist` | `geometry_msgs/TwistStamped` | finite-difference velocity, clamped |
| out | `~/enable` | `std_msgs/Bool` | hold-to-run; forced false on input timeout |

`header.stamp` of the outputs is the *acquisition* time of the OpenXR sample, so
downstream nodes can measure true motion-to-command latency.

Parameters: `hand` (right_hand), `output_frame` (operator_base), `smoothing_alpha` (0.35),
`max_linear_speed` (0.25 m/s), `max_angular_speed` (1.5 rad/s), `trigger_threshold` (0.5),
`input_timeout_s` (0.2), `velocity_window` (3 samples).

Frame conversion: `x_ros = −z_xr`, `y_ros = −x_xr`, `z_ros = y_xr`; orientations by the
same change of basis (`R·M·Rᵀ`).

### `openxr_sim`

Deterministic stand-in for a headset client: publishes head and both hands at 90 Hz
following a scripted weld pass (idle → approach → 200 mm seam at 10 mm/s with a 2 mm
weave → release). Motion is a function of the sample index, so runs are reproducible.
`dropout_at_s` / `dropout_duration_s` inject a tracking loss for safety tests.

## Try it

```bash
ros2 run openxr_ros2_bridge openxr_sim
ros2 run openxr_ros2_bridge openxr_ros2_bridge_node
ros2 topic echo /openxr_ros2_bridge/twist
```

## Status

M5 thin slice: simulator + bridge working against the Tele-Craft graph. Real OpenXR
client for the HTC Vive Focus Vision follows once the hardware arrives.
