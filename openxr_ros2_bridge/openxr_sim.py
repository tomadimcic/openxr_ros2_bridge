#!/usr/bin/env python3
# Copyright 2026 Entanglenet GmbH
# SPDX-License-Identifier: Apache-2.0
"""Deterministic OpenXR input simulator.

Stands in for a headset until the HTC Vive Focus Vision arrives. Publishes what a
thin OpenXR client would publish: head and hand poses in the OpenXR *stage* space
(right-handed, +X right, +Y up, -Z forward, metres) plus controller inputs.

The scripted scenario is a weld pass: idle, approach, weld a straight seam with a
small weave, release, repeat. Motion is a function of the sample index, so the same
parameters always produce the same stream (evidence runs are reproducible).

Optional fault injection: ``dropout_at_s`` / ``dropout_duration_s`` stop all
publishing for a while, which the downstream safety gate must catch.
"""
import math

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Joy


def quat_from_axis_angle(ax, ay, az, angle):
    s = math.sin(angle / 2.0)
    return (ax * s, ay * s, az * s, math.cos(angle / 2.0))


class OpenXRSim(Node):
    def __init__(self):
        super().__init__("openxr_sim")
        self.declare_parameter("rate_hz", 90.0)
        self.declare_parameter("frame_id", "openxr_stage")
        self.declare_parameter("seam_length_m", 0.20)
        self.declare_parameter("weld_speed_m_s", 0.010)
        self.declare_parameter("weave_amplitude_m", 0.002)
        self.declare_parameter("weave_hz", 1.5)
        self.declare_parameter("jitter_m", 0.0005)
        self.declare_parameter("idle_s", 2.0)
        self.declare_parameter("approach_s", 3.0)
        self.declare_parameter("release_s", 2.0)
        self.declare_parameter("dropout_at_s", -1.0)       # <0 disables
        self.declare_parameter("dropout_duration_s", 1.0)
        self.declare_parameter("loop", True)

        p = self.get_parameter
        self.rate = float(p("rate_hz").value)
        self.frame = str(p("frame_id").value)
        self.seam = float(p("seam_length_m").value)
        self.speed = float(p("weld_speed_m_s").value)
        self.weave = float(p("weave_amplitude_m").value)
        self.weave_hz = float(p("weave_hz").value)
        self.jitter = float(p("jitter_m").value)
        self.t_idle = float(p("idle_s").value)
        self.t_approach = float(p("approach_s").value)
        self.t_release = float(p("release_s").value)
        self.dropout_at = float(p("dropout_at_s").value)
        self.dropout_len = float(p("dropout_duration_s").value)
        self.loop = bool(p("loop").value)
        self.t_weld = self.seam / max(self.speed, 1e-6)
        self.t_cycle = self.t_idle + self.t_approach + self.t_weld + self.t_release

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub_rhand = self.create_publisher(PoseStamped, "/openxr/right_hand/pose", qos)
        self.pub_lhand = self.create_publisher(PoseStamped, "/openxr/left_hand/pose", qos)
        self.pub_head = self.create_publisher(PoseStamped, "/openxr/head/pose", qos)
        self.pub_inputs = self.create_publisher(Joy, "/openxr/right_hand/inputs", qos)

        self.k = 0
        self.timer = self.create_timer(1.0 / self.rate, self.tick)
        self.get_logger().info(
            f"OpenXR simulator: {self.rate:.0f} Hz, seam {self.seam*1000:.0f} mm at "
            f"{self.speed*1000:.0f} mm/s, cycle {self.t_cycle:.1f} s, "
            f"dropout at {self.dropout_at:.1f} s" if self.dropout_at >= 0 else
            f"OpenXR simulator: {self.rate:.0f} Hz, seam {self.seam*1000:.0f} mm at "
            f"{self.speed*1000:.0f} mm/s, cycle {self.t_cycle:.1f} s")

    # Deterministic pseudo-noise from the sample index (no RNG state to reproduce).
    def _noise(self, k, salt):
        return math.sin(k * 12.9898 + salt * 78.233) * 0.5

    def scenario(self, t):
        """Return (hand_pos_xr, trigger, phase) for cycle time t. Positions in metres.

        Hand rest pose: 0.35 m in front of the operator (-Z), 0.9 m high, 0.25 m right.
        Weld seam runs along +X (operator's right), 0.30 m ahead, at 0.85 m height.
        """
        rest = (0.25, 0.90, -0.35)
        start = (0.10, 0.85, -0.30)
        if t < self.t_idle:
            return rest, 0.0, "idle"
        t -= self.t_idle
        if t < self.t_approach:
            s = t / self.t_approach
            s = 0.5 - 0.5 * math.cos(math.pi * s)  # ease in/out
            pos = tuple(rest[i] + (start[i] - rest[i]) * s for i in range(3))
            return pos, 0.0, "approach"
        t -= self.t_approach
        if t < self.t_weld:
            x = start[0] + self.speed * t
            weave = self.weave * math.sin(2 * math.pi * self.weave_hz * t)
            pos = (x, start[1], start[2] + weave)
            return pos, 1.0, "weld"
        t -= self.t_weld
        s = min(t / self.t_release, 1.0)
        end = (start[0] + self.seam, start[1], start[2])
        pos = tuple(end[i] + (rest[i] - end[i]) * s for i in range(3))
        return pos, 0.0, "release"

    def tick(self):
        k = self.k
        self.k += 1
        t_abs = k / self.rate
        if self.dropout_at >= 0 and self.dropout_at <= t_abs < self.dropout_at + self.dropout_len:
            if abs(t_abs - self.dropout_at) < 1.0 / self.rate:
                self.get_logger().warn(f"Injected tracking dropout for {self.dropout_len:.1f} s")
            return
        t = t_abs % self.t_cycle if self.loop else min(t_abs, self.t_cycle - 1e-3)
        pos, trigger, phase = self.scenario(t)
        jitter = tuple(self.jitter * self._noise(k, i) for i in range(3))
        pos = tuple(pos[i] + jitter[i] for i in range(3))

        now = self.get_clock().now().to_msg()

        def pose(p, q):
            m = PoseStamped()
            m.header.stamp = now
            m.header.frame_id = self.frame
            m.pose.position.x, m.pose.position.y, m.pose.position.z = p
            m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = q
            return m

        # Torch held pointing forward-down: rotate about X by -60 deg (OpenXR frame).
        q_hand = quat_from_axis_angle(1.0, 0.0, 0.0, math.radians(-60.0))
        self.pub_rhand.publish(pose(pos, q_hand))
        self.pub_lhand.publish(pose((-0.25, 0.85, -0.30), (0.0, 0.0, 0.0, 1.0)))
        # Head: 1.6 m high, looking slightly down at the seam (pitch -25 deg).
        q_head = quat_from_axis_angle(1.0, 0.0, 0.0, math.radians(-25.0))
        self.pub_head.publish(pose((0.0, 1.60, 0.0), q_head))

        j = Joy()
        j.header.stamp = now
        j.header.frame_id = self.frame
        j.axes = [trigger, 0.0]              # [trigger, squeeze]
        j.buttons = [0, 0, 0]                # [primary, secondary, menu]
        self.pub_inputs.publish(j)

        if k % int(self.rate * 5) == 0:
            self.get_logger().info(f"t={t_abs:6.1f}s phase={phase:8s} trigger={trigger:.0f} "
                                   f"hand_xr=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})")


def main(args=None):
    rclpy.init(args=args)
    node = OpenXRSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
