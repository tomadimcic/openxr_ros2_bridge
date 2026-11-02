#!/usr/bin/env python3
# Copyright 2026 Entanglenet GmbH
# SPDX-License-Identifier: Apache-2.0
"""Fake standalone headset for testing openxr_udp_gateway without hardware.

Sends protocol-v1 pose datagrams at 90 Hz like the Unity headset application: head, both controllers and two
trackers in Unity coordinates (left-handed, X right, Y up, Z forward), the right trigger held for a while, and
the clock-sync echo of the gateway's status datagrams. Its own clock is the monotonic clock shifted by
--clock-offset-s, so the test can check that the gateway recovers the offset.

Usage: python3 test/fake_udp_headset.py [--pc 127.0.0.1] [--seconds 10] [--clock-offset-s -1234.5]
Prints the gateway status it receives and, at the end, the true PC-minus-headset offset to compare.
"""
import argparse
import json
import math
import socket
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9101)
    ap.add_argument("--listen", type=int, default=9102)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--rate", type=float, default=90.0)
    ap.add_argument("--clock-offset-s", type=float, default=-1234.5)
    ap.add_argument("--drop-trackers-at", type=float, default=-1.0, help="mark the trackers untracked from this time on")
    args = ap.parse_args()

    off_ns = int(args.clock_offset_s * 1e9)
    clock = lambda: time.monotonic_ns() + off_ns          # "headset" clock

    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("0.0.0.0", args.listen))
    rx.setblocking(False)

    sync, seq, last_print = None, 0, 0.0
    t0 = time.monotonic()
    period = 1.0 / args.rate
    while time.monotonic() - t0 < args.seconds:
        t = time.monotonic() - t0
        # drain status datagrams; keep the echo of the newest one
        while True:
            try:
                data, _ = rx.recvfrom(65535)
            except BlockingIOError:
                break
            t4 = clock()
            st = json.loads(data.decode())
            if st.get("t1") is not None:
                sync = {"t1": st["t1"], "t2": st["t2"], "t3": st["t3"], "t4": t4}
            if time.monotonic() - last_print > 1.0:
                last_print = time.monotonic()
                print(f"status: link {st.get('link_hz')} Hz, offset {st.get('offset_ms')} ms, rtt {st.get('rtt_ms')} ms, "
                      f"auth {st.get('auth_state')}/{st.get('auth_mode')}, prompt '{st.get('prompt_title')}', "
                      f"tracking {[d['id'] + ('+' if d['ok'] else '-') for d in st.get('tracking', [])]}")
        a = 2 * math.pi * 0.2 * t
        right = [0.25 + 0.05 * math.cos(a), 1.0 + 0.05 * math.sin(a), 0.45]       # Unity: right, up, forward
        trackers_ok = not (0 <= args.drop_trackers_at <= t)
        st_trk = 3 if trackers_ok else 0
        msg = {"v": 1, "kind": "pose", "seq": seq, "t": clock(), "frame": "unity", "origin": "floor", "recenter": 0,
               "dev": [{"id": "head", "tr": 0, "st": 3, "p": [0.0, 1.25, 0.0], "q": [0.0, 0.0, 0.0, 1.0]},
                       {"id": "right_hand", "tr": 0, "st": 3, "p": right, "q": [0.0, 0.0, 0.0, 1.0]},
                       {"id": "left_hand", "tr": 0, "st": 3, "p": [-0.25, 0.9, 0.3], "q": [0.0, 0.0, 0.0, 1.0]},
                       {"id": "tracker_0", "tr": 0, "st": st_trk, "p": [0.0, 1.05, 0.08], "q": [0.0, 0.0, 0.0, 1.0]},
                       {"id": "tracker_1", "tr": 0, "st": st_trk, "p": [0.22, 1.0, 0.2], "q": [0.0, 0.0, 0.0, 1.0]}],
               "in": [{"id": "right_hand", "trigger": 1.0 if 2.0 < t < 6.0 else 0.0, "grip": 0.0, "a": 0, "b": 0,
                       "menu": 0, "stick": [0.0, 0.0], "stick_click": 0}]}
        if sync:
            msg["sync"] = sync
        tx.sendto(json.dumps(msg, separators=(",", ":")).encode(), (args.pc, args.port))
        seq += 1
        time.sleep(max(0.0, period - ((time.monotonic() - t0) - t)))
    true_offset_ms = (time.time_ns() - clock()) / 1e6
    print(f"sent {seq} datagrams; true PC-minus-headset offset {true_offset_ms:.3f} ms")


if __name__ == "__main__":
    main()
