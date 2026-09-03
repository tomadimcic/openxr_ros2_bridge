#!/usr/bin/env python3
# Copyright 2026 Entanglenet GmbH
# SPDX-License-Identifier: Apache-2.0
"""Deterministic OpenXR input simulator: a *seated* welder holding the deactivated torch.

Publishes what the headset client will publish once the HTC Vive Focus Vision and the
Ultimate Trackers arrive, all in the OpenXR *stage* space (right-handed, +X right, +Y up,
-Z forward, metres, floor at Y = 0):

  /openxr/head/pose                    headset
  /openxr/right_hand/pose              torch controller (the welding hand)
  /openxr/left_hand/pose               other controller
  /openxr/right_hand/inputs            sensor_msgs/Joy, axes[0] = trigger (hold-to-run)
  /openxr/trackers/right_elbow/pose    Ultimate Tracker strapped to the right elbow
  /openxr/trackers/chest/pose          Ultimate Tracker on the chest

The hand follows a weld pass (idle, approach, straight seam with a small weave, release);
the elbow tracker is placed by a two-link arm model from the seated shoulder, the head looks
at the seam. Motion is a function of the sample index, so runs are reproducible.
`time_scale` speeds the whole scenario up for demos; `dropout_at_s` injects a tracking loss.
"""
import math

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Joy


def q_axis(ax, ay, az, angle):
    s = math.sin(angle / 2.0)
    return (ax * s, ay * s, az * s, math.cos(angle / 2.0))


def q_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def v_sub(a, b):
    return tuple(a[i] - b[i] for i in range(3))


def v_add(a, b):
    return tuple(a[i] + b[i] for i in range(3))


def v_scale(a, s):
    return tuple(a[i] * s for i in range(3))


def v_norm(a):
    return math.sqrt(sum(c * c for c in a))


def v_unit(a):
    n = v_norm(a) or 1.0
    return v_scale(a, 1.0 / n)


def v_dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def v_cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


class OpenXRSim(Node):
    def __init__(self):
        super().__init__("openxr_sim")
        self.declare_parameter("rate_hz", 90.0)
        self.declare_parameter("frame_id", "openxr_stage")
        self.declare_parameter("time_scale", 3.0)          # 1.0 = real welding speed (10 mm/s)
        self.declare_parameter("seam_length_m", 0.20)
        self.declare_parameter("weld_speed_m_s", 0.010)
        self.declare_parameter("weave_amplitude_m", 0.002)
        self.declare_parameter("weave_hz", 1.5)
        self.declare_parameter("jitter_m", 0.0005)
        self.declare_parameter("idle_s", 2.0)
        self.declare_parameter("approach_s", 2.5)
        self.declare_parameter("release_s", 2.0)
        self.declare_parameter("dropout_at_s", -1.0)
        self.declare_parameter("dropout_duration_s", 1.0)
        self.declare_parameter("loop", True)
        self.declare_parameter("alternate_high_reach", True)   # every other pass is a high, awkward seam (RULA >= 5)
        # seated welder geometry (stage frame, metres): chair 0.45 m, right shoulder, arm lengths
        self.declare_parameter("shoulder_xr", [0.19, 1.09, 0.0])   # chair 0.45 + shoulder 0.64 above the pelvis
        self.declare_parameter("upper_arm_m", 0.373)              # human_description, height 1.75 m
        self.declare_parameter("forearm_m", 0.233)
        self.declare_parameter("grip_pitch_deg", 0.0)             # torch axis vs forearm at a neutral wrist
        self.declare_parameter("eye_height_m", 1.22)

        p = self.get_parameter
        self.rate = float(p("rate_hz").value)
        self.frame = str(p("frame_id").value)
        self.tscale = float(p("time_scale").value)
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
        self.alternate = bool(p("alternate_high_reach").value)
        self.shoulder = tuple(float(v) for v in p("shoulder_xr").value)
        self.l_upper = float(p("upper_arm_m").value)
        self.l_fore = float(p("forearm_m").value)
        self.eye_h = float(p("eye_height_m").value)
        self.grip = math.radians(float(p("grip_pitch_deg").value))
        self.t_weld = self.seam / max(self.speed, 1e-6)
        self.t_cycle = self.t_idle + self.t_approach + self.t_weld + self.t_release

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub = {
            "head": self.create_publisher(PoseStamped, "/openxr/head/pose", qos),
            "rhand": self.create_publisher(PoseStamped, "/openxr/right_hand/pose", qos),
            "lhand": self.create_publisher(PoseStamped, "/openxr/left_hand/pose", qos),
            "relbow": self.create_publisher(PoseStamped, "/openxr/trackers/right_elbow/pose", qos),
            "chest": self.create_publisher(PoseStamped, "/openxr/trackers/chest/pose", qos),
        }
        self.pub_inputs = self.create_publisher(Joy, "/openxr/right_hand/inputs", qos)
        self.k = 0
        self.create_timer(1.0 / self.rate, self.tick)
        self.get_logger().info(f"OpenXR simulator (seated welder): {self.rate:.0f} Hz, seam {self.seam*1000:.0f} mm at "
                               f"{self.speed*1000:.0f} mm/s, time x{self.tscale:g}, cycle {self.t_cycle/self.tscale:.1f} s"
                               + (f", dropout at {self.dropout_at:.1f} s" if self.dropout_at >= 0 else ""))

    def _noise(self, k, salt):
        return math.sin(k * 12.9898 + salt * 78.233) * 0.5

    # ---- scenario -----------------------------------------------------------
    def scenario(self, t, high=False):
        """(hand position, trigger, phase, weld progress 0..1) for scenario time t (seconds)."""
        rest = (0.28, 0.78, -0.30)          # hand resting on the thigh/armrest
        start = (0.05, 1.28, -0.50) if high else (0.05, 0.82, -0.42)   # high seam: hand above the shoulder
        if t < self.t_idle:
            return rest, 0.0, "idle", 0.0
        t -= self.t_idle
        if t < self.t_approach:
            s = 0.5 - 0.5 * math.cos(math.pi * t / self.t_approach)
            return v_add(rest, v_scale(v_sub(start, rest), s)), 0.0, "approach", 0.0
        t -= self.t_approach
        if t < self.t_weld:
            prog = t / self.t_weld
            weave = self.weave * math.sin(2 * math.pi * self.weave_hz * t)
            return (start[0] + self.seam * prog, start[1], start[2] + weave), 1.0, "weld", prog
        t -= self.t_weld
        end = (start[0] + self.seam, start[1], start[2])
        s = min(t / self.t_release, 1.0)
        return v_add(end, v_scale(v_sub(rest, end), s)), 0.0, "release", 1.0

    def elbow_from(self, hand):
        """Two-link arm: elbow position for a hand position, elbow hanging down and slightly outward."""
        u = v_sub(hand, self.shoulder)
        d = max(min(v_norm(u), self.l_upper + self.l_fore - 1e-3), abs(self.l_upper - self.l_fore) + 1e-3)
        u = v_unit(u)
        a = (self.l_upper ** 2 - self.l_fore ** 2 + d * d) / (2 * d)
        b = math.sqrt(max(self.l_upper ** 2 - a * a, 0.0))
        hint = (0.35, -1.0, 0.0)                       # down and outward (right side +X)
        perp = v_sub(hint, v_scale(u, v_dot(hint, u)))
        perp = v_unit(perp) if v_norm(perp) > 1e-6 else (1.0, 0.0, 0.0)
        return v_add(self.shoulder, v_add(v_scale(u, a), v_scale(perp, b)))

    @staticmethod
    def hand_quat(fore_dir, pitch):
        """Quaternion whose -Z (OpenXR pointing axis) is the forearm direction pitched down by `pitch`
        about the lateral axis, with +Y kept as close to world up as possible."""
        up = (0.0, 1.0, 0.0)
        lateral = v_unit(v_cross(up, tuple(-c for c in fore_dir)))      # = right-hand X axis
        if v_norm(lateral) < 1e-6:
            lateral = (1.0, 0.0, 0.0)
        # rotate the pointing direction about the lateral axis by -pitch (down)
        c, s_ = math.cos(-pitch), math.sin(-pitch)
        f = fore_dir
        k = lateral
        point = tuple(f[i] * c + v_cross(k, f)[i] * s_ + k[i] * v_dot(k, f) * (1 - c) for i in range(3))
        z = tuple(-c_ for c_ in point)                                     # OpenXR -Z points along the torch
        y = v_unit(v_cross(z, lateral))
        x = v_cross(y, z)
        m = ((x[0], y[0], z[0]), (x[1], y[1], z[1]), (x[2], y[2], z[2]))
        tr = m[0][0] + m[1][1] + m[2][2]
        if tr > 0:
            s4 = math.sqrt(tr + 1.0) * 2
            return ((m[2][1] - m[1][2]) / s4, (m[0][2] - m[2][0]) / s4, (m[1][0] - m[0][1]) / s4, 0.25 * s4)
        if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
            s4 = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
            return (0.25 * s4, (m[0][1] + m[1][0]) / s4, (m[0][2] + m[2][0]) / s4, (m[2][1] - m[1][2]) / s4)
        if m[1][1] > m[2][2]:
            s4 = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
            return ((m[0][1] + m[1][0]) / s4, 0.25 * s4, (m[1][2] + m[2][1]) / s4, (m[0][2] - m[2][0]) / s4)
        s4 = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        return ((m[0][2] + m[2][0]) / s4, (m[1][2] + m[2][1]) / s4, 0.25 * s4, (m[1][0] - m[0][1]) / s4)

    def look_at_quat(self, eye, target):
        """Head orientation: OpenXR looks down -Z; yaw about Y then pitch about X."""
        d = v_unit(v_sub(target, eye))
        yaw = math.atan2(-d[0], -d[2])
        pitch = math.asin(max(-1.0, min(1.0, d[1])))
        return q_mul(q_axis(0, 1, 0, yaw), q_axis(1, 0, 0, pitch))

    def tick(self):
        k = self.k
        self.k += 1
        t_abs = k / self.rate
        if self.dropout_at >= 0 and self.dropout_at <= t_abs < self.dropout_at + self.dropout_len:
            if abs(t_abs - self.dropout_at) < 1.0 / self.rate:
                self.get_logger().warn(f"Injected tracking dropout for {self.dropout_len:.1f} s")
            return
        ts = t_abs * self.tscale
        t = ts % self.t_cycle if self.loop else min(ts, self.t_cycle - 1e-3)
        cycle = int(ts // self.t_cycle)
        high = self.alternate and cycle % 2 == 1
        hand, trigger, phase, prog = self.scenario(t, high)
        hand = v_add(hand, tuple(self.jitter * self._noise(k, i) for i in range(3)))
        elbow = self.elbow_from(hand)

        now = self.get_clock().now().to_msg()

        def pose(pub, pos, q):
            m = PoseStamped()
            m.header.stamp = now
            m.header.frame_id = self.frame
            m.pose.position.x, m.pose.position.y, m.pose.position.z = pos
            m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = q
            self.pub[pub].publish(m)

        # hand orientation follows the forearm: torch axis = forearm direction pitched by the grip angle,
        # plus a small wrist flexion wobble while welding (that wobble is the real wrist motion RULA sees)
        wobble = math.radians(6.0) * math.sin(2 * math.pi * 0.7 * t) if phase == "weld" else 0.0
        q_hand = self.hand_quat(v_unit(v_sub(hand, elbow)), self.grip + wobble)
        pose("rhand", hand, q_hand)
        pose("lhand", (-0.25, 0.78, -0.28), (0.0, 0.0, 0.0, 1.0))
        pose("relbow", elbow, q_hand)
        # chest: leans forward a little while welding (trunk flexion ~8 deg)
        lean_deg = (25.0 if high else 8.0) if phase == "weld" else 3.0   # reaching up and forward bends the trunk
        lean = math.radians(-lean_deg)
        pose("chest", (0.0, 0.92, 0.02), q_axis(1, 0, 0, lean))
        # head: in VR the welder looks at the rendered stereo view, roughly straight ahead; while welding
        # they tend to crane forward and down (8 -> 25 deg over the seam), which the RULA node must flag.
        eye = (0.0, self.eye_h, 0.0)
        crane = 32.0 if high else 17.0                        # the high seam makes the welder crane more
        pitch = math.radians(-(8.0 + crane * prog)) if phase == "weld" else math.radians(-6.0)
        yaw = math.radians(-12.0 * (hand[0] / 0.3))          # follows the hand a little
        pose("head", eye, q_mul(q_axis(0, 1, 0, yaw), q_axis(1, 0, 0, pitch)))

        j = Joy()
        j.header.stamp = now
        j.header.frame_id = self.frame
        j.axes = [trigger, 0.0]
        j.buttons = [1 if phase == "weld" else 0, 0, 0]      # buttons[0] = arc on (WELD process mode)
        self.pub_inputs.publish(j)

        if k % int(self.rate * 5) == 0:
            self.get_logger().info(f"t={t_abs:6.1f}s {'HIGH ' if high else ''}phase={phase:8s} trigger={trigger:.0f} "
                                   f"hand_xr=({hand[0]:.3f},{hand[1]:.3f},{hand[2]:.3f}) seam {prog*100:3.0f}%")


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
