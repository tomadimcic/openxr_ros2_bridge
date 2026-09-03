#!/usr/bin/env python3
# Copyright 2026 Entanglenet GmbH
# SPDX-License-Identifier: Apache-2.0
"""openxr_ros2_bridge: OpenXR controller poses -> ROS 2 teleoperation commands.

ARISE reusable module A. Robot-agnostic: it knows nothing about the arm.

Input (from an OpenXR client, real or simulated):
  /openxr/<hand>/pose     geometry_msgs/PoseStamped  OpenXR stage space (+X right, +Y up, -Z fwd)
  /openxr/<hand>/inputs   sensor_msgs/Joy            axes[0] = trigger (hold-to-run)

Output:
  ~/pose    geometry_msgs/PoseStamped   smoothed hand pose in a REP-103 frame (X fwd, Y left, Z up)
  ~/twist   geometry_msgs/TwistStamped  hand velocity in the same frame, clamped
  ~/enable  std_msgs/Bool               hold-to-run, false on input timeout
  ~/weld    std_msgs/Bool               arc-on request from the controller's primary button
  ~/tracking/<name>  geometry_msgs/PoseStamped   every tracked point (head, hands, trackers) in the
                                        REP-103 frame, unsmoothed, for body tracking / ergonomics

This node is the only place where OpenXR data enters ROS 2. Everything downstream, robot
teleoperation and ergonomics alike, consumes its outputs.

header.stamp of every output carries the *acquisition* time of the OpenXR sample so
downstream nodes can measure the true motion-to-command latency.
"""
import math

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool

# OpenXR (x right, y up, z back) -> ROS REP-103 (x fwd, y left, z up):
#   x_ros = -z_xr, y_ros = -x_xr, z_ros = y_xr
_R = ((0.0, 0.0, -1.0),
      (-1.0, 0.0, 0.0),
      (0.0, 1.0, 0.0))


def _mat_vec(m, v):
    return tuple(sum(m[i][j] * v[j] for j in range(3)) for i in range(3))


def _quat_to_mat(q):
    x, y, z, w = q
    return ((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)))


def _mat_to_quat(m):
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return ((m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s, 0.25 * s)
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        return (0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s, (m[2][1] - m[1][2]) / s)
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        return ((m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s, (m[0][2] - m[2][0]) / s)
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
    return ((m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s, (m[1][0] - m[0][1]) / s)


def _mat_mul(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def _transpose(m):
    return tuple(tuple(m[j][i] for j in range(3)) for i in range(3))


def xr_to_ros_pose(p, q):
    """Change of basis for position and orientation: R * M * R^T."""
    p_ros = _mat_vec(_R, p)
    m_ros = _mat_mul(_mat_mul(_R, _quat_to_mat(q)), _transpose(_R))
    return p_ros, _mat_to_quat(m_ros)


def _nlerp(q0, q1, a):
    dot = sum(q0[i] * q1[i] for i in range(4))
    if dot < 0:
        q1 = tuple(-c for c in q1)
    q = tuple(q0[i] + (q1[i] - q0[i]) * a for i in range(4))
    n = math.sqrt(sum(c * c for c in q)) or 1.0
    return tuple(c / n for c in q)


class OpenXRBridge(Node):
    def __init__(self):
        super().__init__("openxr_ros2_bridge")
        self.declare_parameter("hand", "right_hand")
        self.declare_parameter("output_frame", "operator_base")
        self.declare_parameter("smoothing_alpha", 0.35)      # 1 = no smoothing
        self.declare_parameter("max_linear_speed", 0.25)     # m/s clamp
        self.declare_parameter("max_angular_speed", 1.5)     # rad/s clamp
        self.declare_parameter("trigger_threshold", 0.5)
        self.declare_parameter("input_timeout_s", 0.2)
        self.declare_parameter("velocity_window", 3)         # samples for finite difference
        self.declare_parameter("weld_button", 0)             # Joy buttons[] index of the arc-on control
        self.declare_parameter("tracked_points", ["head", "right_hand", "left_hand", "trackers/right_elbow", "trackers/chest"])

        p = self.get_parameter
        hand = str(p("hand").value)
        self.out_frame = str(p("output_frame").value)
        self.alpha = float(p("smoothing_alpha").value)
        self.v_max = float(p("max_linear_speed").value)
        self.w_max = float(p("max_angular_speed").value)
        self.trig_thr = float(p("trigger_threshold").value)
        self.timeout = float(p("input_timeout_s").value)
        self.win = max(int(p("velocity_window").value), 1)
        self.weld_btn = int(p("weld_button").value)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(PoseStamped, f"/openxr/{hand}/pose", self.on_pose, qos)
        self.create_subscription(Joy, f"/openxr/{hand}/inputs", self.on_inputs, qos)
        self.pub_pose = self.create_publisher(PoseStamped, "~/pose", qos)
        self.pub_twist = self.create_publisher(TwistStamped, "~/twist", qos)
        self.pub_enable = self.create_publisher(Bool, "~/enable", 10)
        self.pub_weld = self.create_publisher(Bool, "~/weld", 10)
        self.weld = False
        # all tracked points, converted to the ROS frame, republished for body tracking
        self.pub_track = {}
        for name in p("tracked_points").value:
            short = name.split("/")[-1]
            self.pub_track[short] = self.create_publisher(PoseStamped, f"~/tracking/{short}", qos)
            self.create_subscription(PoseStamped, f"/openxr/{name}/pose", lambda m, k=short: self.on_tracked(k, m), qos)

        self.trigger = 0.0
        self.enabled = False
        self.sp = None          # smoothed position (ros frame)
        self.sq = None          # smoothed orientation
        self.history = []       # (t, position, orientation) of smoothed samples
        self.last_rx = None
        self.samples = 0
        self.create_timer(0.05, self.watchdog)
        self.get_logger().info(f"openxr_ros2_bridge: /openxr/{hand} -> {self.out_frame}, "
                               f"alpha={self.alpha}, v_max={self.v_max} m/s, timeout={self.timeout}s")

    def on_inputs(self, msg: Joy):
        self.trigger = float(msg.axes[0]) if msg.axes else 0.0
        weld = bool(msg.buttons[self.weld_btn]) if len(msg.buttons) > self.weld_btn else False
        if weld != self.weld:
            self.weld = weld
            self.get_logger().info(f"arc {'ON' if weld else 'off'} requested")
        self.pub_weld.publish(Bool(data=weld))

    def on_tracked(self, key, msg: PoseStamped):
        p = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        q = (msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w)
        p_ros, q_ros = xr_to_ros_pose(p, q)
        out = PoseStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.out_frame
        out.pose.position.x, out.pose.position.y, out.pose.position.z = p_ros
        out.pose.orientation.x, out.pose.orientation.y, out.pose.orientation.z, out.pose.orientation.w = q_ros
        self.pub_track[key].publish(out)

    def on_pose(self, msg: PoseStamped):
        self.last_rx = self.get_clock().now()
        self.samples += 1
        p = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        q = (msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w)
        p_ros, q_ros = xr_to_ros_pose(p, q)

        if self.sp is None:
            self.sp, self.sq = p_ros, q_ros
        else:
            a = self.alpha
            self.sp = tuple(self.sp[i] + a * (p_ros[i] - self.sp[i]) for i in range(3))
            self.sq = _nlerp(self.sq, q_ros, a)

        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.history.append((t, self.sp, self.sq))
        if len(self.history) > self.win + 1:
            self.history.pop(0)

        out = PoseStamped()
        out.header.stamp = msg.header.stamp            # acquisition time, not now()
        out.header.frame_id = self.out_frame
        out.pose.position.x, out.pose.position.y, out.pose.position.z = self.sp
        out.pose.orientation.x, out.pose.orientation.y, out.pose.orientation.z, out.pose.orientation.w = self.sq
        self.pub_pose.publish(out)

        tw = TwistStamped()
        tw.header = out.header
        if len(self.history) >= 2:
            t0, p0, q0 = self.history[0]
            dt = t - t0
            if dt > 1e-4:
                v = [(self.sp[i] - p0[i]) / dt for i in range(3)]
                n = math.sqrt(sum(c * c for c in v))
                if n > self.v_max:
                    v = [c * self.v_max / n for c in v]
                tw.twist.linear.x, tw.twist.linear.y, tw.twist.linear.z = v
                # angular velocity from relative rotation q_rel = q_now * conj(q0)
                x0, y0, z0, w0 = q0
                x1, y1, z1, w1 = self.sq
                # q_rel = q1 * conj(q0)
                rx = w1 * -x0 + x1 * w0 + y1 * -z0 - z1 * -y0
                ry = w1 * -y0 - x1 * -z0 + y1 * w0 + z1 * -x0
                rz = w1 * -z0 + x1 * -y0 - y1 * -x0 + z1 * w0
                rw = w1 * w0 - x1 * -x0 - y1 * -y0 - z1 * -z0
                ang = 2.0 * math.atan2(math.sqrt(rx * rx + ry * ry + rz * rz), abs(rw))
                s = math.sqrt(rx * rx + ry * ry + rz * rz)
                if s > 1e-9:
                    w = min(ang / dt, self.w_max)
                    sign = 1.0 if rw >= 0 else -1.0
                    tw.twist.angular.x, tw.twist.angular.y, tw.twist.angular.z = (
                        sign * rx / s * w, sign * ry / s * w, sign * rz / s * w)
        self.pub_twist.publish(tw)

        enable = self.trigger >= self.trig_thr
        if enable != self.enabled:
            self.enabled = enable
            self.get_logger().info(f"hold-to-run {'ENGAGED' if enable else 'released'}")
        self.pub_enable.publish(Bool(data=enable))

    def watchdog(self):
        if self.last_rx is None:
            return
        age = (self.get_clock().now() - self.last_rx).nanoseconds * 1e-9
        if age > self.timeout and self.enabled:
            self.enabled = False
            self.pub_enable.publish(Bool(data=False))
            self.get_logger().warn(f"OpenXR input timeout ({age*1000:.0f} ms): hold-to-run released")


def main(args=None):
    rclpy.init(args=args)
    node = OpenXRBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
