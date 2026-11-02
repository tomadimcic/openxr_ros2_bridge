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
| out | `~/enable` | `std_msgs/Bool` | hold-to-run; forced false on input timeout and kept false until the trigger is released and pressed again |
| out | `~/weld` | `std_msgs/Bool` | arc-on request from the controller's primary button |
| out | `~/tracking/<name>` | `geometry_msgs/PoseStamped` | every tracked point (head, hands, trackers) in the REP-103 frame, for body tracking |

The bridge is the **only** node that reads OpenXR data. Robot teleoperation and ergonomics both consume its outputs.

`header.stamp` of the outputs is the *acquisition* time of the OpenXR sample, so
downstream nodes can measure true motion-to-command latency.

Recovery never resumes motion by itself: after an input timeout the bridge keeps `~/enable` false,
even if the trigger is still held when tracking returns, until the trigger is released and pressed again.

Parameters: `hand` (right_hand), `output_frame` (operator_base), `smoothing_alpha` (0.35),
`max_linear_speed` (0.25 m/s), `max_angular_speed` (1.5 rad/s), `trigger_threshold` (0.5),
`input_timeout_s` (0.2), `velocity_window` (3 samples).

Frame conversion: `x_ros = −z_xr`, `y_ros = −x_xr`, `z_ros = y_xr`; orientations by the
same change of basis (`R·M·Rᵀ`).

### `openxr_console`

Interactive stand-in until the headset arrives, plus a preview of the VR overlay. A Tk window:
the mouse moves the torch hand over a top view of the reach, the wheel lifts it, **holding the
left mouse button is the trigger** (hold-to-run, the robot follows only while held); arrows / W S
move, Q E tilt the head, SPACE toggles the trigger, R resets. The right pane shows the ergonomic
prompt from `/human/ergo_prompt` (info / attention / warning), the RULA score and the cell's
motion authority. Publishes the same topics as a headset client. With `spectator:=true` it shows the
HUD only and publishes nothing, for use next to another input; `geometry` (e.g. `+1900+0`) places the window.

### `openxr_sim`

Deterministic stand-in for a headset client: publishes head and both hands at 90 Hz
following a scripted weld pass (idle → approach → 200 mm seam at 10 mm/s with a 2 mm
weave → release). Motion is a function of the sample index, so runs are reproducible.
`dropout_at_s` / `dropout_duration_s` inject a tracking loss for safety tests; `reengage_s` (0.8 s)
makes the scripted welder release the trigger after the dropout, the re-engagement gesture the
bridge requires. Set it to 0 to test that motion does not resume without the gesture.

## Try it

```bash
ros2 run openxr_ros2_bridge openxr_sim
ros2 run openxr_ros2_bridge openxr_ros2_bridge_node
ros2 topic echo /openxr_ros2_bridge/twist
```

## Tests

`colcon test --packages-select openxr_ros2_bridge` runs, on every push in CI on ROS 2 Jazzy
(`.github/workflows/ci.yml`):

| Test | What it checks |
| --- | --- |
| `test_flake8`, `test_pep257`, `test_copyright`, `test_xmllint`, `test_mypy` | ament linters; house style in `setup.cfg` (double quotes, lines up to 120 characters) |
| `test_conversions` | OpenXR stage to REP-103: positions, orientations, quaternion round trip, nlerp |
| `test_bridge_node` | hold-to-run: engages only with a fresh pose and a pressed trigger, released within the input timeout on a dropout, never resumes by itself, a stale trigger counts as released, twist speed clamp |
| `test_gateway_video` | the camera app's picture age reaches `/openxr/video_age`; the twin app sends none |
| `test_launch_scripted_welder` | launch test: the scripted welder (`openxr_sim`) and the bridge as separate processes, two pairs on their own DDS domains; a tracking dropout releases hold-to-run within the input timeout + 0.15 s; with the release-and-press gesture it engages again right after the press; with the trigger kept held it stays released until a release and a press |
| `test_link_netem` | headset link through `openxr_udp_gateway` under tc/netem: engaged through 30 ms +- 10 ms delay and 2 % loss, released at 100 % loss and not resumed without the gesture (needs passwordless sudo, as on the CI runner) |

`test/reply_port_check.py`, `test/tracker_boxes_check.py`, `test/headset_link_check.py` and
`test/pose_precision_test.py` are scripts for the gateway and the real headset, run by hand.

## Status

Working with the HTC VIVE Focus Vision: the headset application sends its poses and inputs to
`openxr_udp_gateway`, and VIVE Ultimate Trackers reach the bridge through the headset or through
the PC.
