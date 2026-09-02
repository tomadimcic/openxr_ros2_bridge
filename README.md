# openxr_ros2_bridge

**ARISE reusable module #1 (Tele-Craft, Module A).** Turns OpenXR controller and
tracker poses into ROS 2 Cartesian velocity commands so that any ROS 2 robot can be
teleoperated from a VR headset.

Status: package skeleton, M5 thin slice in progress (September 2026).

## Interfaces (planned)

| Direction | Topic | Type | Notes |
|---|---|---|---|
| in | `/openxr/controller/<side>/pose` | `geometry_msgs/PoseStamped` | 60–120 Hz from the headset app |
| in | `/openxr/controller/<side>/hold_to_run` | `std_msgs/Bool` | dead-man switch; no motion without it |
| out | `/servo_node/delta_twist_cmds` | `geometry_msgs/TwistStamped` | consumed by MoveIt Servo |
| out | `/openxr_ros2_bridge/status` | diagnostics | tracking valid, rate, latency |

* Converts OpenXR's right-handed Y-up frame to REP-103 (X forward, Y left, Z up).
* Kinematic smoothing and velocity/acceleration clamps as parameters.
* Robot-agnostic: the output is a Twist; the robot is whatever MoveIt Servo drives.

## License

Apache-2.0. Developed in the Tele-Craft experiment of the ARISE project
(EU Horizon Europe, arise-middleware.eu) by Entanglenet GmbH and Jazzing Automation.
