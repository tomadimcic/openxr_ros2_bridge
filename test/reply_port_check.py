#!/usr/bin/env python3
# Copyright 2026 Entanglenet GmbH
# SPDX-License-Identifier: Apache-2.0
"""reply_port_check: openxr_udp_gateway answers an app at the port it names (reply_port), else at its parameter.

The camera app listens on any free port (the twin app, paused in the background, can hold 9102) and names it in its
pose datagrams. Starts the gateway on its own ROS domain (localhost only), its own UDP ports and at the lowest CPU
priority, so a live session is not disturbed. Checks: an app that names no port gets the status at the parameter
reply_port (the installed twin app); one that names its port gets it there, and none arrives at the parameter's.
Usage: python3 test/reply_port_check.py [--domain 78] [--port 19111]
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", type=int, default=78)
    ap.add_argument("--port", type=int, default=19111, help="gateway listen port; the parameter reply_port is the next one")
    args = ap.parse_args()
    env = dict(os.environ, ROS_DOMAIN_ID=str(args.domain), ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST")
    env.pop("FASTDDS_DEFAULT_PROFILES_FILE", None)
    cmd = [sys.executable, os.path.join(here, "..", "openxr_ros2_bridge", "openxr_udp_gateway.py"), "--ros-args",
           "-p", f"listen_port:={args.port}", "-p", f"reply_port:={args.port + 1}"]
    gw = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                          preexec_fn=lambda: os.nice(19))
    socks = {}
    for name, port in (("parameter", args.port + 1), ("own", 0)):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("127.0.0.1", port))
        s.setblocking(False)
        socks[name] = s
    own_port = socks["own"].getsockname()[1]
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    seq = 0
    ok_all = True

    def check(name, ok, detail=""):
        nonlocal ok_all
        ok_all &= bool(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

    def run(reply_port, seconds, until_status=False):
        """Pose datagrams at about 90 Hz for seconds; counts the status datagrams arriving at each socket."""
        nonlocal seq
        got = {"parameter": 0, "own": 0}
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end and not (until_status and got["parameter"]):
            d = {"v": 1, "kind": "pose", "seq": seq, "t": time.monotonic_ns(), "frame": "unity", "origin": "floor",
                 "recenter": 0, "dev": [{"id": "head", "tr": 1, "st": 3, "p": [0.0, 1.25, 0.0], "q": [0, 0, 0, 1]}]}
            if reply_port:
                d["reply_port"] = reply_port
            seq += 1
            tx.sendto(json.dumps(d).encode(), ("127.0.0.1", args.port))
            for name, s in socks.items():
                try:
                    while True:
                        got[name] += b'"kind":"status"' in s.recv(65535)
                except BlockingIOError:
                    pass
            time.sleep(0.011)
        return got

    try:
        first = run(None, 20.0, until_status=True)
        check("gateway up, status at the parameter reply_port", first["parameter"] > 0)
        a = run(None, 2.0)
        check("app names no port (installed twin app): status at the parameter's", a["parameter"] >= 15 and a["own"] == 0,
              f"{a}")
        run(own_port, 0.5)                                      # the switch: datagrams already on the way
        b = run(own_port, 2.0)
        check("app names its port: status there, none at the parameter's", b["own"] >= 15 and b["parameter"] == 0,
              f"{b}")
    finally:
        gw.terminate()
        try:
            _, err = gw.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            gw.kill()
            _, err = gw.communicate()
    check("gateway log names the port", f"replies to port {own_port}" in err)
    print("ALL PASS" if ok_all else "SOME CHECKS FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
