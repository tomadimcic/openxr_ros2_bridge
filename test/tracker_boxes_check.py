#!/usr/bin/env python3
# Copyright 2026 Entanglenet GmbH
# SPDX-License-Identifier: Apache-2.0
"""tracker_boxes_check: the PC trackers' boxes in the scene datagrams of openxr_udp_gateway, without hardware.

Starts the gateway on its own ROS domain (localhost only), its own UDP ports and at the lowest CPU priority, so a
live session is not disturbed. Plays the headset (pose datagrams at 90 Hz, so the gateway knows where to send)
and vive_tracker_pc: the torch moving sideways (0.2 m sine, 0.5 Hz, up to 0.63 m/s) and a still forearm tracker,
both at 125 Hz and stamped 5 ms before publishing, like the SDK samples; plus a workpiece box. Each scene datagram
is applied at the next frame of a 90 Hz headset clock, as the app does (newest one only).

Checks: scene datagrams at least at the headset frame rate, the workpiece in every one, frames in which the torch
box did not move, how far the box trails the torch at each frame (with show_trackers_predict within a few ms,
mm), and the forearm box hidden within 0.75 s after its samples stop. --gateway runs another gateway file (e.g.
the 20 Hz version) the same way for comparison; --no-predict sets show_trackers_predict false.
Usage: python3 test/tracker_boxes_check.py [--no-predict] [--gateway FILE] [--domain 77] [--port 19101]
"""
import argparse
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time

AMP, FREQ = 0.2, 0.5                       # torch x(t) = AMP sin(2 pi FREQ t), OpenXR stage x (Unity x)
TORCH_Y, TORCH_Z = 1.0, -0.4
FOREARM = (0.15, 1.1, -0.3)
STAMP_AGE = 0.005                          # samples are published this long after their stamp
FRAME = 1.0 / 90.0                         # headset frame period


def torch_x(t):
    return AMP * math.sin(2 * math.pi * FREQ * t)


def torch_v(t):
    return AMP * 2 * math.pi * FREQ * math.cos(2 * math.pi * FREQ * t)


def cpu_s(pid):
    with open(f"/proc/{pid}/stat") as f:
        fields = f.read().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default=os.path.join(here, "..", "openxr_ros2_bridge", "openxr_udp_gateway.py"))
    ap.add_argument("--no-predict", action="store_true")
    ap.add_argument("--domain", type=int, default=77)
    ap.add_argument("--port", type=int, default=19101, help="gateway listen port; the reply port is the next one")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--save", default="", help="write the received scene datagrams to this JSON file")
    args = ap.parse_args()

    os.environ["ROS_DOMAIN_ID"] = str(args.domain)          # before rclpy.init, and inherited by the gateway
    os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    os.environ.pop("FASTDDS_DEFAULT_PROFILES_FILE", None)
    os.nice(19)
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
    from visualization_msgs.msg import Marker, MarkerArray

    rclpy.init()
    node = rclpy.create_node("tracker_boxes_check")
    latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
    tf = TransformStamped()
    tf.header.frame_id, tf.child_frame_id = "world", "wp"
    tf.transform.translation.x, tf.transform.rotation.w = 1.0, 1.0
    StaticTransformBroadcaster(node).sendTransform(tf)
    m = Marker()
    m.header.frame_id, m.ns, m.id, m.type, m.action = "wp", "workpiece", 0, Marker.CUBE, Marker.ADD
    m.pose.orientation.w, m.scale.x, m.scale.y, m.scale.z, m.color.r, m.color.a = 1.0, 0.3, 0.2, 0.01, 0.5, 1.0
    node.create_publisher(MarkerArray, "/telecraft/workpiece", latched).publish(MarkerArray(markers=[m]))
    qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
    pub_torch = node.create_publisher(PoseStamped, "/openxr/trackers/torch/pose", qos)
    pub_fore = node.create_publisher(PoseStamped, "/openxr/trackers/right_forearm/pose", qos)

    cmd = [sys.executable, args.gateway, "--ros-args", "-r", "__node:=tracker_boxes_gateway",
           "-p", f"listen_port:={args.port}", "-p", f"reply_port:={args.port + 1}", "-p", "twin_links:=[link_x]",
           "-p", "show_trackers:=[torch, right_forearm]", "-p", "publish_trackers:=false",
           "-p", f"show_trackers_predict:={'false' if args.no_predict else 'true'}"]
    gw = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                          preexec_fn=lambda: os.nice(19))
    running, fore_on = [True], [True]

    def play_trackers():
        k, t0 = 0, time.monotonic()
        while running[0]:
            st = time.time() - STAMP_AGE
            for pub, p, on in ((pub_torch, (torch_x(st), TORCH_Y, TORCH_Z), True), (pub_fore, FOREARM, fore_on[0])):
                if not on:
                    continue
                msg = PoseStamped()
                msg.header.stamp.sec, msg.header.stamp.nanosec = int(st), int((st % 1) * 1e9)
                msg.header.frame_id = "openxr_stage"
                msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = p
                msg.pose.orientation.w = 1.0
                pub.publish(msg)
            k += 1
            time.sleep(max(0.0, t0 + k / 125.0 - time.monotonic()))

    def play_headset(tx):
        seq, t0 = 0, time.monotonic()
        while running[0]:
            d = {"v": 1, "kind": "pose", "seq": seq, "t": time.monotonic_ns(), "frame": "unity", "origin": "floor",
                 "recenter": 0, "dev": [{"id": "head", "tr": 0, "st": 3, "p": [0.0, 1.25, 0.0], "q": [0, 0, 0, 1]}],
                 "in": []}
            tx.sendto(json.dumps(d).encode(), ("127.0.0.1", args.port))
            seq += 1
            time.sleep(max(0.0, t0 + seq * FRAME - time.monotonic()))

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", args.port + 1))
    rx.settimeout(0.2)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    threads = [threading.Thread(target=play_trackers, daemon=True), threading.Thread(target=play_headset, args=(tx,),
                                                                                     daemon=True)]
    for th in threads:
        th.start()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threads.append(threading.Thread(target=executor.spin, daemon=True))
    threads[-1].start()

    scenes = []                            # (receive time, {id: object})
    ok = True
    try:
        t_start, first = time.monotonic(), None
        t_window = t_fore_off = t_end = None
        cpu0 = None
        while True:
            now = time.monotonic()
            if gw.poll() is not None:
                print("FAIL gateway exited: " + gw.stderr.read()[-2000:])
                return 1
            if first is None and now - t_start > 15.0:
                print("FAIL no scene datagram within 15 s")
                return 1
            if first is not None and t_window is None and now - first > 2.0:   # rate settled
                t_window, cpu0 = time.time(), cpu_s(gw.pid)
            if t_window is not None and t_fore_off is None and time.time() - t_window > args.seconds:
                t_fore_off, cpu1 = time.time(), cpu_s(gw.pid)
                fore_on[0] = False
            if t_fore_off is not None and time.time() - t_fore_off > 1.5:
                t_end = time.time()
                break
            try:
                data, _ = rx.recvfrom(65535)
            except socket.timeout:
                continue
            d = json.loads(data.decode())
            if d.get("kind") != "scene":
                continue
            objs = {o["id"]: o for o in d["objects"]}
            if first is None and objs.get("tracker:torch", {}).get("s", [0])[0] > 0 and "workpiece/0" in objs:
                first = time.monotonic()           # the gateway has the trackers and the workpiece
            scenes.append((time.time(), objs))
    finally:
        running[0] = False
        for th in threads[:2]:
            th.join(1.0)
        gw.terminate()
        try:
            gw.wait(5.0)
        except subprocess.TimeoutExpired:
            gw.kill()
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()

    if args.save:
        with open(args.save, "w") as f:
            json.dump({"window": t_window, "fore_off": t_fore_off, "end": t_end, "scenes": scenes}, f)
    win = [s for s in scenes if t_window <= s[0] < t_fore_off]
    rate = len(win) / (t_fore_off - t_window)
    with_wp = sum(1 for _, objs in win if "workpiece/0" in objs)
    print(f"gateway {os.path.basename(args.gateway)}, predict {'off' if args.no_predict else 'on'}: "
          f"{rate:.0f} scene datagrams/s, workpiece in {with_wp} of {len(win)}, gateway CPU "
          f"{100 * (cpu1 - cpu0) / (t_fore_off - t_window):.0f} %")
    if rate < 90:
        print("FAIL fewer scene datagrams than headset frames")
        ok = False
    if with_wp != len(win):
        print("FAIL workpiece missing from some scene datagrams")
        ok = False

    # each 90 Hz frame shows the newest datagram received before it
    frames, k, shown, still, errs, lags = 0, 0, None, 0, [], []
    t_frame = t_window + 0.0037            # arbitrary frame phase
    while t_frame < t_fore_off:
        while k < len(scenes) and scenes[k][0] <= t_frame:
            k += 1
        if k:
            box = scenes[k - 1][1].get("tracker:torch")
            frames += 1
            if box is not None and box["s"][0] > 0:
                if shown is not None and k - 1 == shown:
                    still += 1
                shown = k - 1
                e = box["p"][0] - torch_x(t_frame)
                errs.append(abs(e))
                v = torch_v(t_frame)
                if abs(v) > 0.3:
                    lags.append(-e / v * 1000.0)
        t_frame += FRAME
    errs.sort()
    lags.sort()
    print(f"torch box over {frames} frames: not moved in {still} ({100 * still / max(1, frames):.0f} %), behind the "
          f"torch by median {lags[len(lags) // 2]:.1f} ms (p10 {lags[len(lags) // 10]:.1f}, p90 "
          f"{lags[len(lags) * 9 // 10]:.1f}), position error median {1000 * errs[len(errs) // 2]:.1f} mm, p90 "
          f"{1000 * errs[len(errs) * 9 // 10]:.1f} mm, max {1000 * errs[-1]:.1f} mm")
    if not args.no_predict and (abs(lags[len(lags) // 2]) > 4.0 or errs[len(errs) * 9 // 10] > 0.005):
        print("FAIL predicted torch box not on the torch (median within 4 ms, p90 within 5 mm)")
        ok = False

    sizes = [(t, objs.get("tracker:right_forearm", {}).get("s", [0])[0]) for t, objs in scenes if t >= t_window]
    hidden = next((t for t, s in sizes if t >= t_fore_off and s == 0.0), None)
    shown_before = all(s > 0 for t, s in sizes if t < t_fore_off)
    torch_shown = all(objs.get("tracker:torch", {}).get("s", [0])[0] > 0 for t, objs in scenes if t >= t_window)
    print(f"forearm box shown while its samples flow: {shown_before}, hidden "
          f"{'after %.2f s' % (hidden - t_fore_off) if hidden else 'never'}; torch box shown throughout: {torch_shown}")
    if not shown_before or hidden is None or hidden - t_fore_off > 0.75 or not torch_shown:
        print("FAIL tracker boxes shown or hidden wrongly")
        ok = False
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
