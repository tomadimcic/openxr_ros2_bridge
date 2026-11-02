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
"""The camera app's picture age through openxr_udp_gateway to /openxr/video_age (subprocess, own ROS domain).

Datagrams with "video":{"age_ms":A} give /openxr/video_age = A / 1000 s; datagrams without it (the twin app) give
nothing, so the safety gate only watches the picture while the camera app is in use. Test case T-B11 of
docs/d22/test_cases.md in the TELEC workspace.
"""
import json
import os
import socket
import subprocess
import sys
import time

import pytest
import rclpy
from std_msgs.msg import Float32

PORT = 19141
DOMAIN = 10 + os.getpid() % 40


@pytest.fixture
def gateway():
    env = dict(os.environ, ROS_DOMAIN_ID=str(DOMAIN), ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST")
    gw = subprocess.Popen([sys.executable, "-m", "openxr_ros2_bridge.openxr_udp_gateway", "--ros-args",
                           "-p", f"listen_port:={PORT}", "-p", f"reply_port:={PORT + 1}"],
                          env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    rclpy.init(domain_id=DOMAIN)
    node = rclpy.create_node("video_age_probe")
    got = []
    node.create_subscription(Float32, "/openxr/video_age", lambda m: got.append(m.data), 10)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    seq = [0]

    def send(seconds, video_ms):
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            d = {"v": 1, "kind": "pose", "seq": seq[0], "t": time.monotonic_ns(), "frame": "unity",
                 "origin": "floor", "recenter": 0,
                 "dev": [{"id": "head", "tr": 1, "st": 3, "p": [0.0, 1.25, 0.0], "q": [0, 0, 0, 1]}]}
            if video_ms is not None:
                d["video"] = {"age_ms": video_ms}
            seq[0] += 1
            sock.sendto(json.dumps(d).encode(), ("127.0.0.1", PORT))
            rclpy.spin_once(node, timeout_sec=1.0 / 90.0)

    try:
        yield got, send
    finally:
        node.destroy_node()
        rclpy.shutdown()
        gw.terminate()
        gw.wait(5.0)


def test_picture_age_reaches_ros_only_from_the_camera_app(gateway):
    """T-B11."""
    got, send = gateway
    send(3.0, 250.0)                             # discovery, then the camera app's datagrams
    assert got and all(abs(v - 0.25) < 1e-6 for v in got[-20:])
    got.clear()
    send(1.0, None)                              # the twin app: no picture age
    assert not got
