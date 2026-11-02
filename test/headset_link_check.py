#!/usr/bin/env python3
# Copyright 2026 Entanglenet GmbH
# SPDX-License-Identifier: Apache-2.0
"""Live check of the headset link, run next to openxr_udp_gateway while the headset application is running.

Every second it prints the rate of each /openxr pose topic, the data age (PC time now minus the header stamp, which
the gateway has already converted to PC time) and the gateway's link status (receive rate, sequence gaps, clock
offset, round trip). Tracker topics are picked up as they appear. At the end (Ctrl-C or --seconds) it prints a
summary and the last position of every device, both as published (frame openxr_stage, OpenXR convention: x right,
y up, z backward) and converted to REP-103 the way the bridge does it (x forward, y left, z up), so the axes can be
checked by hand: raise the right controller and REP-103 z grows, push it forward and x grows, move it left and y grows.

Usage: python3 test/headset_link_check.py [--seconds 30]
"""
import argparse
import json
import statistics
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Joy
from std_msgs.msg import String

FIXED = ["/openxr/head/pose", "/openxr/right_hand/pose", "/openxr/left_hand/pose"]


def short(topic):
    parts = topic.strip("/").split("/")
    return parts[-2]          # head, right_hand, left_hand, chest, right_elbow, tracker_2, ...


class LinkCheck(Node):
    def __init__(self):
        super().__init__("headset_link_check")
        self.qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.topics = []
        self.count, self.total, self.ages, self.last = {}, {}, {}, {}
        self.status, self.joy = None, None
        for t in FIXED:
            self.watch(t)
        self.create_subscription(String, "/openxr_udp_gateway/status", self.on_status, 10)
        self.create_subscription(Joy, "/openxr/right_hand/inputs", self.on_joy, self.qos)

    def watch(self, topic):
        self.topics.append(topic)
        self.count[topic], self.total[topic], self.ages[topic] = 0, 0, []
        self.create_subscription(PoseStamped, topic, lambda m, t=topic: self.on_pose(t, m), self.qos)

    def discover(self):
        for name, types in self.get_topic_names_and_types():
            if name.startswith("/openxr/trackers/") and name.endswith("/pose") and name not in self.topics:
                self.watch(name)

    def on_pose(self, topic, msg):
        now = self.get_clock().now().nanoseconds
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self.count[topic] += 1
        self.total[topic] += 1
        self.ages[topic].append((now - stamp) / 1e6)
        self.last[topic] = msg

    def on_status(self, msg):
        self.status = json.loads(msg.data)

    def on_joy(self, msg):
        self.joy = msg

    def line(self, t):
        s = self.status or {}
        head = (f"{t:4.0f}s  link {s.get('rx_hz', '-'):>5} Hz  gaps {s.get('seq_gaps', '-'):>3}  "
                f"offset {s.get('offset_ms')} ms  rtt {s.get('rtt_ms')} ms |")
        cols = []
        for topic in self.topics:
            recent = self.ages[topic][-self.count[topic]:] if self.count[topic] else []
            age = f"{statistics.median(recent):5.1f}ms" if recent else "   -   "
            cols.append(f"{short(topic)} {self.count[topic]:3d} {age}")
            self.count[topic] = 0
        trig = f"trigger {self.joy.axes[0]:.2f} arc {self.joy.buttons[0]}" if self.joy and self.joy.axes else "trigger -"
        return head + " " + " | ".join(cols) + " | " + trig

    def summary(self, secs):
        print("\nsummary over %.0f s" % secs)
        for topic in self.topics:
            a = sorted(self.ages[topic])
            if not a:
                print(f"  {short(topic):12s} no data")
                continue
            p95 = a[min(len(a) - 1, int(0.95 * len(a)))]
            print(f"  {short(topic):12s} {self.total[topic] / secs:6.1f} Hz  data age median {a[len(a) // 2]:6.1f} ms  "
                  f"p95 {p95:6.1f} ms")
        print("last positions: openxr_stage as published (x right, y up, z back) -> REP-103 (x forward, y left, z up)")
        for topic in self.topics:
            m = self.last.get(topic)
            if m:
                p = m.pose.position
                print(f"  {short(topic):12s} xr ({p.x:+.3f} {p.y:+.3f} {p.z:+.3f})  ->  ros x {-p.z:+.3f}  y {-p.x:+.3f}  z {p.y:+.3f}")
        if self.status:
            print("gateway status:", json.dumps(self.status))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after N s (0 = until Ctrl-C)")
    args = ap.parse_args()
    rclpy.init()
    node = LinkCheck()
    t0 = time.monotonic()
    tick = t0 + 1.0
    try:
        while rclpy.ok() and (args.seconds <= 0 or time.monotonic() - t0 < args.seconds):
            rclpy.spin_once(node, timeout_sec=0.01)
            if time.monotonic() >= tick:
                node.discover()
                print(node.line(tick - t0), flush=True)
                tick += 1.0
    except KeyboardInterrupt:
        pass
    node.summary(time.monotonic() - t0)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
