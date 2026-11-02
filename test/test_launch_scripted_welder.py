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
"""Launch tests T-B12 and T-B13: the scripted welder and the bridge as separate processes, as in the demo.

Two pairs of openxr_sim (time scale 3) and the bridge run side by side on their own DDS domains. Each welder holds the
trigger through its first pass, and a 1.0 s tracking dropout is injected 4 s after start.
  A (T-B12): after the dropout the welder releases the trigger for 0.8 s (the re-engagement gesture) and presses
     again: hold-to-run is released within the input timeout + 0.15 s and engages again right after the press;
  B (T-B13): the welder keeps the trigger held when tracking returns: hold-to-run stays released for as long as the
     trigger stays held, and engages only after a release and a press (the next pass).
Checked on /openxr_ros2_bridge/enable against the welder's own poses and trigger.
"""
import os
import threading
import time
import unittest

from geometry_msgs.msg import PoseStamped
import launch
import launch_ros.actions
import launch_testing.actions
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool

DOMAIN_A = 30 + os.getpid() % 30
DOMAIN_B = 60 + os.getpid() % 30
TIMEOUT = 0.2                                    # the bridge's input_timeout_s


def env(domain):
    return {"ROS_DOMAIN_ID": str(domain), "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST"}


def pair(domain, reengage_s):
    welder = launch_ros.actions.Node(
        package="openxr_ros2_bridge", executable="openxr_sim", name="openxr_sim", additional_env=env(domain),
        parameters=[{"time_scale": 3.0, "dropout_at_s": 4.0, "dropout_duration_s": 1.0, "reengage_s": reengage_s}])
    bridge = launch_ros.actions.Node(
        package="openxr_ros2_bridge", executable="openxr_ros2_bridge_node", name="openxr_ros2_bridge",
        additional_env=env(domain), parameters=[{"input_timeout_s": TIMEOUT}])
    return [welder, bridge]


@pytest.mark.launch_test
def generate_test_description():
    return launch.LaunchDescription(pair(DOMAIN_A, 0.8) + pair(DOMAIN_B, 0.0)
                                    + [launch_testing.actions.ReadyToTest()]), {}


class Recorder:
    """The bridge's enable and the welder's poses and trigger on one domain, recorded from creation on."""

    def __init__(self, domain):
        os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
        self.ctx = rclpy.Context()
        self.ctx.init(domain_id=domain)
        self.node = rclpy.create_node("launch_test_probe", context=self.ctx)
        self.enable, self.poses, self.trigger = [], [], []
        self.node.create_subscription(Bool, "/openxr_ros2_bridge/enable",
                                      lambda m: self.enable.append((time.monotonic(), m.data)), 100)
        self.node.create_subscription(PoseStamped, "/openxr/right_hand/pose",
                                      lambda m: self.poses.append(time.monotonic()), qos_profile_sensor_data)
        self.node.create_subscription(Joy, "/openxr/right_hand/inputs",
                                      lambda m: self.trigger.append((time.monotonic(), m.axes[0] if m.axes else 0.0)),
                                      qos_profile_sensor_data)
        self.ex = SingleThreadedExecutor(context=self.ctx)
        self.ex.add_node(self.node)
        threading.Thread(target=self.ex.spin, daemon=True).start()

    def wait(self, done, timeout=30.0):
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end and not done(self.enable, self.poses, self.trigger):
            time.sleep(0.2)
        return self.enable, self.poses, self.trigger

    def stop(self):
        self.ex.shutdown()
        self.node.destroy_node()
        self.ctx.try_shutdown()


def dropout(poses):
    gaps = [(a, b) for a, b in zip(poses, poses[1:]) if b - a > 0.5]
    return gaps[0] if gaps else (None, None)


def engaged_after(enable, t):
    return next((s for s, on in enable if s > t and on), None)


class TestScriptedWelder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.a, cls.b = Recorder(DOMAIN_A), Recorder(DOMAIN_B)     # both from the start: the dropouts are at 4 s

    @classmethod
    def tearDownClass(cls):
        cls.a.stop()
        cls.b.stop()

    def test_a_dropout_releases_and_the_gesture_reengages(self):
        enable, poses, trigger = self.a.wait(lambda e, p, t: dropout(p)[1] is not None
                                             and engaged_after(e, dropout(p)[1]) is not None)
        lost, back = dropout(poses)
        self.assertIsNotNone(lost, "no tracking dropout seen in the welder's poses")
        self.assertTrue(any(on and t < lost for t, on in enable), "hold-to-run never engaged before the dropout")
        released = next((t for t, on in enable if t >= lost and not on), None)
        self.assertIsNotNone(released, "hold-to-run not released after the dropout")
        self.assertLessEqual(released - lost, TIMEOUT + 0.15, f"released {released - lost:.3f} s after the last pose")
        let_go = next((t for t, v in trigger if t >= back and v < 0.5), None)
        pressed = next((t for t, v in trigger if let_go is not None and t > let_go and v >= 0.5), None)
        self.assertIsNotNone(pressed, "no release and press after the dropout")
        self.assertLessEqual(pressed - back, 2.0, "the welder's gesture did not follow the dropout")
        reengaged = engaged_after(enable, released)
        self.assertIsNotNone(reengaged, "hold-to-run never engaged again")
        self.assertGreaterEqual(reengaged, pressed - 0.05, "engaged again before the release and press")
        self.assertLessEqual(reengaged - pressed, 0.3, f"engaged {reengaged - pressed:.2f} s after the press")

    def test_b_trigger_held_after_the_dropout_stays_released(self):
        def done(e, p, t):
            back = dropout(p)[1]
            return back is not None and any(s > back + 1.0 and v < 0.5 for s, v in t)
        enable, poses, trigger = self.b.wait(done)
        lost, back = dropout(poses)
        self.assertIsNotNone(lost, "no tracking dropout seen in the welder's poses")
        let_go = next((t for t, v in trigger if t >= back and v < 0.5), None)
        self.assertIsNotNone(let_go, "the welder never released the trigger after the dropout")
        self.assertGreaterEqual(let_go - back, 1.0, "the trigger was not held after tracking returned")
        early = [t for t, on in enable if back <= t < let_go and on]
        if early:
            self.fail(f"hold-to-run engaged {early[0] - back:.2f} s after tracking returned, with the trigger held "
                      "and no release")
