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
"""Network resilience of the headset link under Linux tc/netem: delay, jitter and loss on the loopback interface.

A fake headset sends poses and a held trigger at 90 Hz over UDP to openxr_udp_gateway (subprocess); the bridge
runs in this process on the gateway's topics. netem needs passwordless sudo (the CI runner has it; skipped
otherwise). With 30 ms +- 10 ms delay and 2 % loss hold-to-run stays engaged; at 100 % loss it is released within
the input timeout and does not come back by itself when the link recovers; the re-engagement gesture brings it
back. Test cases T-B09 and T-B10 of docs/d22/test_cases.md in the TELEC workspace.
"""
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import threading
import time

import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Bool

from openxr_ros2_bridge.openxr_ros2_bridge_node import OpenXRBridge

PORT = 19131                                     # gateway listen port; replies on PORT + 1
DOMAIN = 50 + os.getpid() % 40


def netem_available():
    if shutil.which("tc") is None:
        return False
    return subprocess.run(["sudo", "-n", "tc", "-V"], capture_output=True).returncode == 0


pytestmark = pytest.mark.skipif(not netem_available(), reason="tc/netem needs passwordless sudo (CI runner)")


def netem(*args):
    subprocess.run(["sudo", "-n", "tc", "qdisc", *args], check=True)


class Headset(threading.Thread):
    """90 Hz pose datagrams with the right hand slowly moving and the trigger as set."""

    def __init__(self):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.trigger, self.running, self.seq = 1.0, True, 0

    def run(self):
        t0 = time.monotonic()
        while self.running:
            t = time.monotonic() - t0
            x = 0.01 * math.sin(2 * math.pi * 0.5 * t)
            d = {"v": 1, "kind": "pose", "seq": self.seq, "t": time.monotonic_ns(), "frame": "unity",
                 "origin": "floor", "recenter": 0,
                 "dev": [{"id": "head", "tr": 1, "st": 3, "p": [0.0, 1.25, 0.0], "q": [0, 0, 0, 1]},
                         {"id": "right_hand", "tr": 1, "st": 3, "p": [0.2 + x, 1.0, 0.4], "q": [0, 0, 0, 1]}],
                 "in": [{"id": "right_hand", "trigger": self.trigger, "grip": 0.0, "a": 0, "b": 0, "menu": 0,
                         "stick": [0.0, 0.0], "stick_click": 0}]}
            self.seq += 1
            self.sock.sendto(json.dumps(d).encode(), ("127.0.0.1", PORT))
            time.sleep(1.0 / 90.0)


@pytest.fixture
def link():
    env = dict(os.environ, ROS_DOMAIN_ID=str(DOMAIN), ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST")
    gw = subprocess.Popen([sys.executable, "-m", "openxr_ros2_bridge.openxr_udp_gateway", "--ros-args",
                           "-p", f"listen_port:={PORT}", "-p", f"reply_port:={PORT + 1}"],
                          env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    rclpy.init(domain_id=DOMAIN)
    bridge = OpenXRBridge()
    probe = rclpy.create_node("netem_probe")
    enable = []
    probe.create_subscription(Bool, "/openxr_ros2_bridge/enable", lambda m: enable.append((time.monotonic(), m.data)),
                              10)
    ex = SingleThreadedExecutor()
    ex.add_node(bridge)
    ex.add_node(probe)
    headset = Headset()
    headset.start()

    def spin(seconds):
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            ex.spin_once(timeout_sec=0.01)

    try:
        yield headset, enable, spin
    finally:
        subprocess.run(["sudo", "-n", "tc", "qdisc", "del", "dev", "lo", "root"], capture_output=True)
        headset.running = False
        ex.shutdown()
        bridge.destroy_node()
        probe.destroy_node()
        rclpy.shutdown()
        gw.terminate()
        gw.wait(5.0)


def test_engaged_through_delay_jitter_and_loss_released_on_outage(link):
    """T-B09 and T-B10."""
    headset, enable, spin = link
    spin(3.0)                                    # discovery, then engaged
    assert enable and enable[-1][1], "hold-to-run not engaged on the clean link"
    netem("add", "dev", "lo", "root", "netem", "delay", "30ms", "10ms", "loss", "2%")
    t_mark = time.monotonic()
    spin(4.0)
    impaired = [v for t, v in enable if t > t_mark + 0.2]
    assert impaired and all(impaired), "released while the link was only delayed, jittery and lossy"
    netem("change", "dev", "lo", "root", "netem", "loss", "100%")
    t_cut = time.monotonic()
    spin(0.8)
    released = [t for t, v in enable if t > t_cut and not v]
    assert released and released[0] - t_cut < 0.2 + 0.25, "not released within the input timeout"
    netem("change", "dev", "lo", "root", "netem", "delay", "30ms", "10ms", "loss", "2%")
    spin(1.0)                                    # link back, trigger still held
    assert not enable[-1][1], "motion resumed by itself after the outage"
    headset.trigger = 0.0
    spin(0.4)
    headset.trigger = 1.0
    spin(0.6)
    assert enable[-1][1], "re-engagement gesture did not bring hold-to-run back"
