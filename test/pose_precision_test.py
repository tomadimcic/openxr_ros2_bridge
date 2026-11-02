#!/usr/bin/env python3
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
"""Compare the precision of two tracked devices that are rigidly attached to each other.

Typical use: a VIVE Ultimate Tracker taped to a VIVE Focus controller, both streamed by openxr_udp_gateway.

    python3 test/pose_precision_test.py --a /openxr/right_hand/pose --b /openxr/trackers/torch/pose \
        --seconds 100 --csv docs/evidence/precision.csv

Protocol (the analysis finds the phases by itself, timing is free):
  1. lay the pair still on the table for about 20 s,
  2. move it like a welding torch for about 40 s: slow straight lines of 20-30 cm, small circles, some faster moves,
  3. hold it low or at the side of the body (out of the headset cameras' view) for about 10 s,
  4. lay it down again.

What it measures, from the header stamps (headset sample time in PC time):
  * update rate, data age at the PC, gaps > 50 ms per device;
  * static jitter of each device while the pair lies still (3D RMS and peak-to-peak, mm; rotation RMS, deg);
  * consistency while moving: the pose of B in A's frame must stay constant for a rigid pair, its spread
    (mm, deg) is the combined error of both devices; the spread of the distance A-B as a simpler check;
  * relative lag: the time shift that best aligns the two speed profiles (ms, positive = B later than A).
Both devices are measured by the same headset in the same space, so this is relative precision and timing,
not absolute accuracy against a ground truth.
"""
import argparse
import csv
import time

from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rot_angle_deg(m):
    return float(np.degrees(np.arccos(np.clip((np.trace(m) - 1.0) / 2.0, -1.0, 1.0))))


class Recorder(Node):

    def __init__(self, topics):
        super().__init__("pose_precision_test")
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=50)
        self.data = {t: [] for t in topics}
        for t in topics:
            self.create_subscription(PoseStamped, t, lambda m, t=t: self.on_pose(t, m), qos)

    def on_pose(self, topic, m):
        stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        p, q = m.pose.position, m.pose.orientation
        self.data[topic].append((stamp, time.time(), p.x, p.y, p.z, q.x, q.y, q.z, q.w))


def resample(arr, grid):
    """Linear interpolation of positions, nearest sample for orientation, on a common time grid."""
    t = arr[:, 0]
    pos = np.stack([np.interp(grid, t, arr[:, k]) for k in (2, 3, 4)], axis=1)
    idx = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
    return pos, arr[idx, 5:9], t


def runs(mask, min_len):
    """(start, end) index ranges where mask is True for at least min_len samples."""
    out, start = [], None
    for i, m in enumerate(np.append(mask, False)):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    return out


def analyse(a_name, b_name, A, B, hz=100.0):
    print(f"\n=== {a_name} (A) vs {b_name} (B)")
    for name, arr in ((a_name, A), (b_name, B)):
        t = arr[:, 0]
        gaps = np.diff(t)
        age = (arr[:, 1] - arr[:, 0]) * 1000.0
        print(f"{name}: {len(arr)} samples, {len(arr) / (t[-1] - t[0]):.1f} Hz, "
              f"data age median {np.median(age):.1f} ms "
              f"p95 {np.percentile(age, 95):.1f} ms, gaps > 50 ms: {int((gaps > 0.05).sum())}, longest gap "
              f"{gaps.max() * 1000:.0f} ms")
    t0, t1 = max(A[0, 0], B[0, 0]), min(A[-1, 0], B[-1, 0])
    grid = np.arange(t0, t1, 1.0 / hz)
    pa, qa, _ = resample(A, grid)
    pb, qb, _ = resample(B, grid)
    # Smoothed positions (0.2 s moving average) are used only to find the phases and the lag; the jitter and
    # spread figures use the unsmoothed samples.
    w = max(int(0.2 * hz), 1)

    def smooth(p):
        return np.stack([np.convolve(p[:, k], np.ones(w) / w, mode="same") for k in range(3)], axis=1)
    sa, sb = smooth(pa), smooth(pb)
    h = max(int(0.1 * hz), 1)

    def velocity(p):                                       # central difference over 0.2 s
        v = np.zeros_like(p)
        v[h:-h] = (p[2 * h:] - p[:-2 * h]) * hz / (2 * h)
        return v
    vel_a, vel_b = velocity(sa), velocity(sb)
    va, vb = np.linalg.norm(vel_a, axis=1), np.linalg.norm(vel_b, axis=1)
    edge = np.zeros(len(va), bool)
    edge[w:-w] = True                                      # ignore the smoothing edges

    # ---- static jitter: both still (< 5 mm/s) for at least 3 s; use the longest such window
    still = runs((va < 0.005) & (vb < 0.005) & edge, int(3 * hz))
    if still:
        s, e = max(still, key=lambda r: r[1] - r[0])
        ts, te = grid[s], grid[e - 1]
        print(f"\nstatic window {te - ts:.1f} s (raw samples):")
        for name, arr in ((a_name, A), (b_name, B)):
            sel = arr[(arr[:, 0] >= ts) & (arr[:, 0] <= te)]
            p, q = sel[:, 2:5], sel[:, 5:9]
            d = p - p.mean(axis=0)
            rms = np.sqrt((d ** 2).sum(axis=1).mean()) * 1000.0
            ptp = np.linalg.norm(p.max(axis=0) - p.min(axis=0)) * 1000.0
            r0 = quat_to_mat(q[0])
            ang = [rot_angle_deg(r0.T @ quat_to_mat(qq)) for qq in q[:: max(1, len(q) // 300)]]
            print(f"  {name}: position jitter RMS {rms:.2f} mm, peak-to-peak {ptp:.2f} mm, rotation spread "
                  f"RMS {np.sqrt(np.mean(np.square(ang))):.3f} deg")
    else:
        print("\nno static window of 3 s with both devices still: lay the pair down untouched for longer")

    moving = (va > 0.03) & edge
    if moving.sum() < hz:
        print("\nless than 1 s of movement above 3 cm/s: move the pair more")
        return
    idx = np.where(moving)[0]
    print(f"\nmoving: {moving.sum() / hz:.1f} s above 3 cm/s, A speed median {np.median(va[idx]) * 100:.1f} cm/s, "
          f"max {va[idx].max() * 100:.0f} cm/s")

    # ---- rigid pair: the pose of B in A's frame must stay constant. The relative lag is the time shift of B that
    # makes the pair most rigid (smallest spread); the spread is reported as streamed and with that lag removed.
    sub = idx[:: max(1, len(idx) // 3000)]
    ra = np.array([quat_to_mat(qa[i]) for i in sub])

    def rel_positions(shift_s):
        pbx = np.stack([np.interp(grid[sub] + shift_s, grid, pb[:, k]) for k in range(3)], axis=1)
        return np.einsum("nji,nj->ni", ra, pbx - pa[sub])          # R_a^T (p_b - p_a)

    def spread(shift_s):
        r = rel_positions(shift_s)
        return float(np.sqrt(((r - r.mean(axis=0)) ** 2).sum(axis=1).mean()))
    shifts = np.arange(-0.150, 0.1505, 0.002)
    sp = np.array([spread(x) for x in shifts])
    j = int(np.argmin(sp))
    lag_s = shifts[j]
    if 0 < j < len(shifts) - 1:
        c0, c1, c2 = sp[j - 1], sp[j], sp[j + 1]
        den = c0 - 2 * c1 + c2
        if den > 0:
            lag_s += 0.5 * (c0 - c2) / den * 0.002
    print(f"  relative lag: B is {lag_s * 1000:+.1f} ms behind A")
    for label, shift in (("as streamed", 0.0), ("lag removed", lag_s)):
        r = rel_positions(shift)
        dp = np.linalg.norm(r - np.median(r, axis=0), axis=1) * 1000.0
        dist = np.linalg.norm(r, axis=1) * 1000.0
        print(f"  B in A's frame, {label}: position spread median {np.median(dp):.1f} mm, p95 "
              f"{np.percentile(dp, 95):.1f} mm, max {dp.max():.1f} mm; distance A-B std {dist.std():.2f} mm")
    rel_r = [ra[n].T @ quat_to_mat(qb[i]) for n, i in enumerate(sub)]
    r_ref = rel_r[len(rel_r) // 2]
    dr = np.array([rot_angle_deg(r_ref.T @ r) for r in rel_r])
    print(f"  B's rotation in A's frame: spread median {np.median(dr):.2f} deg, p95 {np.percentile(dr, 95):.2f} deg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="/openxr/right_hand/pose")
    ap.add_argument("--b", default="/openxr/trackers/torch/pose")
    ap.add_argument("--seconds", type=float, default=100.0)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()
    rclpy.init()
    node = Recorder([args.a, args.b])
    t_end = time.monotonic() + args.seconds
    next_note = time.monotonic() + 10
    while time.monotonic() < t_end:
        rclpy.spin_once(node, timeout_sec=0.05)
        if time.monotonic() > next_note:
            next_note += 10
            print(f"{args.seconds - (t_end - time.monotonic()):5.0f} s: A {len(node.data[args.a])} samples, "
                  f"B {len(node.data[args.b])} samples", flush=True)
    A, B = (np.array(node.data[t]) for t in (args.a, args.b))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["device", "stamp_s", "recv_s", "x", "y", "z", "qx", "qy", "qz", "qw"])
            for name, arr in (("A", A), ("B", B)):
                for r in arr:
                    w.writerow([name] + [f"{v:.6f}" for v in r])
        print(f"raw samples written to {args.csv}")
    if len(A) < 100 or len(B) < 100:
        print(f"not enough data: A {len(A)}, B {len(B)} samples")
    else:
        analyse(args.a, args.b, A, B)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
