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
"""Unit tests of the bridge's frame conversion, OpenXR stage (x right, y up, z back) to REP-103 (x fwd, y left, z up).

Test cases T-B01 to T-B04 of docs/d22/test_cases.md in the TELEC workspace.
"""
import math
import random

import pytest

from openxr_ros2_bridge.openxr_ros2_bridge_node import _mat_to_quat, _nlerp, _quat_to_mat, xr_to_ros_pose


def q_axis(axis, deg):
    s = math.sin(math.radians(deg) / 2.0)
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(math.radians(deg) / 2.0))


def rotate(q, v):
    m = _quat_to_mat(q)
    return tuple(sum(m[i][j] * v[j] for j in range(3)) for i in range(3))


def same_rotation(q0, q1, tol=1e-9):
    return abs(abs(sum(a * b for a, b in zip(q0, q1))) - 1.0) < tol


@pytest.mark.parametrize("xr, ros", [
    ((0.0, 0.0, -1.0), (1.0, 0.0, 0.0)),     # forward
    ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),     # right
    ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),      # up
    ((0.3, 1.2, -0.5), (0.5, -0.3, 1.2)),
])
def test_positions_map_to_rep103(xr, ros):
    """T-B01: every axis of a position lands on its REP-103 axis."""
    assert xr_to_ros_pose(xr, (0.0, 0.0, 0.0, 1.0))[0] == pytest.approx(ros)


def test_identity_orientation_stays_identity():
    """T-B02: a device aligned with the stage is aligned with the REP-103 frame."""
    assert same_rotation(xr_to_ros_pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))[1], (0.0, 0.0, 0.0, 1.0))


@pytest.mark.parametrize("axis_xr, axis_ros",
                         [((0, 1, 0), (0, 0, 1)), ((1, 0, 0), (0, -1, 0)), ((0, 0, -1), (1, 0, 0))])
@pytest.mark.parametrize("deg", [30.0, 90.0, -120.0])
def test_orientation_turns_about_the_same_physical_axis(axis_xr, axis_ros, deg):
    """T-B03: a turn about an OpenXR axis is the same turn about the matching REP-103 axis (yaw left stays left)."""
    q_ros = xr_to_ros_pose((0.0, 0.0, 0.0), q_axis(axis_xr, deg))[1]
    assert same_rotation(q_ros, q_axis(axis_ros, deg))


def test_quaternion_matrix_round_trip_and_nlerp():
    """T-B04: quaternion to matrix and back keeps the rotation; nlerp keeps unit length and takes the short way."""
    rng = random.Random(7)
    for _ in range(200):
        q = [rng.gauss(0.0, 1.0) for _ in range(4)]
        n = math.sqrt(sum(c * c for c in q))
        q = tuple(c / n for c in q)
        assert same_rotation(_mat_to_quat(_quat_to_mat(q)), q, 1e-7)
    q0, q1 = q_axis((0, 0, 1), 0.0), q_axis((0, 0, 1), 90.0)
    assert same_rotation(_nlerp(q0, q1, 0.0), q0) and same_rotation(_nlerp(q0, q1, 1.0), q1)
    mid = _nlerp(q0, tuple(-c for c in q1), 0.5)        # q1 with the other sign: the same rotation
    assert math.sqrt(sum(c * c for c in mid)) == pytest.approx(1.0)
    assert rotate(mid, (1.0, 0.0, 0.0)) == pytest.approx((math.cos(math.radians(45)), math.sin(math.radians(45)), 0.0))
