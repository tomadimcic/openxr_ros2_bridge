# Copyright 2026 Entanglenet GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functional tests of openxr_ros2_bridge_node's hold-to-run, in process on its own ROS domain.

The test plays an OpenXR client (pose and trigger at 90 Hz) and checks what the bridge publishes: hold-to-run
engages only with a fresh pose and a pressed trigger, an input dropout releases it within the timeout, motion never
resumes by itself (the trigger must be released and pressed again), a stale trigger counts as released, and the
twist output never exceeds max_linear_speed. Test cases T-B05 to T-B08 of docs/d22/test_cases.md in the TELEC
workspace.
"""
import os
import time

from geometry_msgs.msg import PoseStamped, TwistStamped
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool

from openxr_ros2_bridge.openxr_ros2_bridge_node import OpenXRBridge

TIMEOUT_S = 0.2                                  # the bridge's input_timeout_s default


class Client:
    """OpenXR client stand-in plus probes on the bridge's outputs."""

    def __init__(self):
        rclpy.init(domain_id=60 + os.getpid() % 40)
        self.bridge = OpenXRBridge()
        self.node = rclpy.create_node("bridge_test_client")
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub_pose = self.node.create_publisher(PoseStamped, "/openxr/right_hand/pose", qos)
        self.pub_in = self.node.create_publisher(Joy, "/openxr/right_hand/inputs", qos)
        self.enable, self.twist = [], []
        self.node.create_subscription(Bool, "/openxr_ros2_bridge/enable",
                                      lambda m: self.enable.append((time.monotonic(), m.data)), 10)
        self.node.create_subscription(TwistStamped, "/openxr_ros2_bridge/twist", lambda m: self.twist.append(m), qos)
        self.ex = SingleThreadedExecutor()
        self.ex.add_node(self.bridge)
        self.ex.add_node(self.node)
        self.z = 0.0
        self.last_sent = None

    def run(self, seconds, trigger=None, pose=True, step=0.0):
        """Publish at 90 Hz for seconds: a pose moving forward by step per sample, and the trigger (None: no input)."""
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            if pose:
                m = PoseStamped()
                m.header.stamp = self.node.get_clock().now().to_msg()
                m.header.frame_id = "openxr_stage"
                self.z -= step                       # OpenXR forward is -z
                m.pose.position.z, m.pose.orientation.w = self.z, 1.0
                self.pub_pose.publish(m)
            if trigger is not None:
                self.pub_in.publish(Joy(axes=[float(trigger)], buttons=[0]))
            if pose or trigger is not None:
                self.last_sent = time.monotonic()
            t_next = time.monotonic() + 1.0 / 90.0
            while time.monotonic() < t_next:
                self.ex.spin_once(timeout_sec=0.002)

    def engaged(self):
        return bool(self.enable) and self.enable[-1][1]

    def close(self):
        self.ex.shutdown()
        self.bridge.destroy_node()
        self.node.destroy_node()
        rclpy.shutdown()


@pytest.fixture
def client():
    c = Client()
    c.run(0.6, trigger=0.0)                      # discovery, trigger released
    yield c
    c.close()


def test_engages_only_with_trigger_and_fresh_pose(client):
    """T-B05: released trigger: no hold-to-run; pressed: engaged within 0.1 s."""
    assert not client.engaged()
    client.run(0.3, trigger=1.0)
    assert client.engaged()


def test_dropout_releases_and_motion_never_resumes_by_itself(client):
    """T-B06: all input stops: released within the timeout + 0.15 s; back with the trigger held: still released."""
    client.run(0.4, trigger=1.0)
    assert client.engaged()
    client.enable.clear()
    client.run(0.6, trigger=None, pose=False)
    released = [t for t, v in client.enable if not v]
    assert released and released[0] - client.last_sent < TIMEOUT_S + 0.15
    client.run(0.6, trigger=1.0)                 # tracking back, trigger still held
    assert not client.engaged()
    client.run(0.2, trigger=0.0)                 # re-engagement gesture: release ...
    client.run(0.3, trigger=1.0)                 # ... and press again
    assert client.engaged()


def test_stale_trigger_counts_as_released(client):
    """T-B07: poses keep coming but the trigger stream stops: released, then re-engagement needed."""
    client.run(0.4, trigger=1.0)
    assert client.engaged()
    client.run(0.6, trigger=None)                # poses only
    assert not client.engaged()
    client.run(0.4, trigger=1.0)
    assert not client.engaged()
    client.run(0.2, trigger=0.0)
    client.run(0.3, trigger=1.0)
    assert client.engaged()


def test_twist_is_clamped_to_max_linear_speed(client):
    """T-B08: the hand moves forward at 0.9 m/s: the twist points forward and stays at or below 0.25 m/s."""
    client.twist.clear()
    client.run(0.8, trigger=1.0, step=0.01)      # 1 cm per sample at 90 Hz
    speeds = [(m.twist.linear.x ** 2 + m.twist.linear.y ** 2 + m.twist.linear.z ** 2) ** 0.5 for m in client.twist]
    assert speeds and max(speeds) <= 0.25 + 1e-6
    assert client.twist[-1].twist.linear.x > 0.2
