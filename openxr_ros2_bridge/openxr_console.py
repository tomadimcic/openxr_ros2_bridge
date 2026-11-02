#!/usr/bin/env python3
# Copyright 2026 Entanglenet GmbH
# SPDX-License-Identifier: Apache-2.0
"""openxr_console: interactive OpenXR stand-in + operator HUD, until the headset arrives.

Left pane: top view of the welder's reach. The mouse moves the torch hand (left/right and
forward), the wheel raises/lowers it, and HOLDING THE LEFT MOUSE BUTTON is the trigger
(hold-to-run): the robot only moves while it is held. Keys: arrows / W S move and lift the
hand, Q E tilt the head, SPACE toggles the trigger, R resets the hand.

With relay_pose_topic set (e.g. a VIVE tracker on the PC), HOLDING T drives the robot with that tracker instead:
the console then forwards the tracker's pose as the hand pose, and the mouse button still drives with the mouse
hand. After a change of source the trigger only takes effect 0.25 s later, so the bridge's smoothing has settled
on the new pose and the clutch anchors there (an instant switch would make the robot jump by the distance between
the two). T is read straight from the keyboard devices, so it works whichever window has the focus (RViz uses M,
I, S, ... as its own tool shortcuts); that needs read access to /dev/input (scripts/install_keyboard_uaccess_udev.sh).
Without it, T (and M) work only while this window has the focus, losing the focus releases them, and the HUD says
so in red. A keyboard that disappears or cannot be read counts as a released key.

Right pane: what the VR overlay will show later: the ergonomic prompt (info / attention /
warning from /human/ergo_prompt), the RULA score, and the cell's motion authority.

Publishes exactly what a headset client publishes: /openxr/head/pose, /openxr/right_hand/pose,
/openxr/left_hand/pose (OpenXR stage frame) and /openxr/right_hand/inputs (trigger).
With inputs_only:=true it publishes only the inputs: another source (a tracker) provides the pose.
"""
import math
import os
import select
import struct
import threading
import time
import tkinter as tk

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy, ReliabilityPolicy
from sensor_msgs.msg import Joy

from openxr_ros2_bridge.openxr_sim import OpenXRSim, q_axis, q_mul, v_sub, v_unit

try:
    from rula_msgs.msg import ErgoPrompt, RulaScore
except ImportError:      # the bridge package stays usable without the RULA module
    ErgoPrompt = RulaScore = None
try:
    from telecraft_msgs.msg import CommandAck, MotionAuthority
    from std_srvs.srv import SetBool, Trigger
except ImportError:
    CommandAck = MotionAuthority = None
    SetBool = Trigger = None

X_RANGE = (-0.35, 0.35)      # lateral, stage X
Z_RANGE = (-0.65, -0.10)     # forward is -Z; top of the canvas = far
Y_RANGE = (0.60, 1.35)       # height
STATE_NAMES = {0: "NOT READY", 1: "READY", 2: "ENABLED", 3: "PAUSED", 4: "SAFE STOP", 5: "EMERGENCY STOP", 6: "FAULT"}
LEVEL_COLORS = {0: ("#1d7a3a", "#e6f4ea"), 1: ("#b8720b", "#fdf1dc"), 2: ("#b3261e", "#fde7e5")}


def elbow_estimate(shoulder, hand, l_upper, l_fore):
    u = v_sub(hand, shoulder)
    d = max(min(math.sqrt(sum(c * c for c in u)), l_upper + l_fore - 1e-3), abs(l_upper - l_fore) + 1e-3)
    u = v_unit(u)
    a = (l_upper ** 2 - l_fore ** 2 + d * d) / (2 * d)
    b = math.sqrt(max(l_upper ** 2 - a * a, 0.0))
    hint = (0.35, -1.0, 0.0)
    dot = sum(hint[i] * u[i] for i in range(3))
    perp = v_unit(tuple(hint[i] - u[i] * dot for i in range(3)))
    return tuple(shoulder[i] + u[i] * a + perp[i] * b for i in range(3))


class ConsoleNode(Node):
    def __init__(self, state, lock):
        super().__init__("openxr_console")
        self.declare_parameter("rate_hz", 90.0)
        self.declare_parameter("frame_id", "openxr_stage")
        self.declare_parameter("shoulder_xr", [0.19, 1.09, 0.0])
        self.declare_parameter("upper_arm_m", 0.373)
        self.declare_parameter("forearm_m", 0.233)
        self.declare_parameter("eye_height_m", 1.22)
        self.declare_parameter("spectator", False)     # true: show the HUD only, publish nothing (another input drives)
        self.declare_parameter("inputs_only", False)   # true: publish only the trigger/arc inputs (a tracker gives the pose)
        self.declare_parameter("relay_pose_topic", "")  # hold M: drive with this pose (e.g. /openxr/trackers/torch/pose)
        self.declare_parameter("geometry", "")          # Tk geometry, e.g. "+1900+0", for scripted window placement
        p = self.get_parameter
        self.state, self.lock = state, lock
        self.frame = str(p("frame_id").value)
        self.shoulder = tuple(float(v) for v in p("shoulder_xr").value)
        self.l_upper, self.l_fore = float(p("upper_arm_m").value), float(p("forearm_m").value)
        self.eye_h = float(p("eye_height_m").value)
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub_head = self.create_publisher(PoseStamped, "/openxr/head/pose", qos)
        self.pub_rhand = self.create_publisher(PoseStamped, "/openxr/right_hand/pose", qos)
        self.pub_lhand = self.create_publisher(PoseStamped, "/openxr/left_hand/pose", qos)
        self.pub_inputs = self.create_publisher(Joy, "/openxr/right_hand/inputs", qos)
        if ErgoPrompt is not None:
            self.create_subscription(ErgoPrompt, "/human/ergo_prompt", lambda m: self.store("prompt", m), latched)
            self.create_subscription(RulaScore, "/human/risk_score", lambda m: self.store("rula", m), 10)
        if MotionAuthority is not None:
            self.create_subscription(MotionAuthority, "/telecraft/authority", lambda m: self.store("auth", m), latched)
            self.create_subscription(CommandAck, "/telecraft/ack", self.on_ack, 50)
        self.spectator = bool(p("spectator").value)
        self.inputs_only = bool(p("inputs_only").value)
        self.relay_topic = str(p("relay_pose_topic").value)
        self.relay = None                              # (PoseStamped, receive time) of the tracker
        if self.relay_topic:
            self.create_subscription(PoseStamped, self.relay_topic, self.on_relay, qos)
        self.geometry = str(p("geometry").value)
        if not self.spectator:
            self.create_timer(1.0 / float(p("rate_hz").value), self.publish)
        if Trigger is not None:
            self.cli_reset = self.create_client(Trigger, "/safety_gate/reset")
            self.cli_ready = self.create_client(SetBool, "/safety_gate/set_ready")

    def reset_cell(self):
        """FAULT / SAFE_STOP -> NOT_READY -> READY: what a supervisor does after acknowledging a fault."""
        if Trigger is None or not self.cli_reset.service_is_ready():
            return
        fut = self.cli_reset.call_async(Trigger.Request())
        fut.add_done_callback(lambda f: self.cli_ready.call_async(SetBool.Request(data=True)))
        self.get_logger().info("cell reset requested from the console")

    def on_relay(self, msg):
        with self.lock:
            self.relay = (msg, time.monotonic())

    def store(self, key, msg):
        with self.lock:
            self.state[key] = msg

    def on_ack(self, m):
        with self.lock:
            if m.accepted and m.stage == "hal":
                self.state["moves"] = self.state.get("moves", 0) + 1
            elif not m.accepted:
                self.state["rejected"] = self.state.get("rejected", 0) + 1
                self.state["last_reject"] = (m.reason, self.get_clock().now().nanoseconds * 1e-9)

    def publish(self):
        t = time.monotonic()
        with self.lock:
            hand = tuple(self.state["hand"])
            want_tracker = (self.state.get("m_trigger", False) or self.state.get("global_trigger", False)) and self.relay_topic != ""
            want_mouse = self.state["trigger"]
            src = "tracker" if want_tracker else ("mouse" if want_mouse else self.state.get("src", "mouse"))
            if src != self.state.get("src", "mouse"):
                self.state["src"], self.state["arm_at"] = src, t + 0.25    # let the smoothing settle first
            armed = t >= self.state.get("arm_at", 0.0)
            trigger = 1.0 if ((want_tracker if src == "tracker" else want_mouse) and armed) else 0.0
            self.state["trigger_out"] = trigger > 0.5
            relay = self.relay
            weld = 1 if self.state.get("weld") else 0
            head_pitch = self.state["head_pitch"]
        now = self.get_clock().now().to_msg()

        def pose(pub, pos, q):
            m = PoseStamped()
            m.header.stamp, m.header.frame_id = now, self.frame
            m.pose.position.x, m.pose.position.y, m.pose.position.z = pos
            m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = q
            pub.publish(m)

        if src == "tracker":
            # forward the tracker as the hand, only while fresh: a stale tracker must look like lost tracking
            if relay is not None and t - relay[1] < 0.15:
                m = relay[0]
                out = PoseStamped()
                out.header = m.header                      # keep the acquisition time
                out.pose = m.pose
                self.pub_rhand.publish(out)
        elif not self.inputs_only:
            elbow = elbow_estimate(self.shoulder, hand, self.l_upper, self.l_fore)
            pose(self.pub_rhand, hand, OpenXRSim.hand_quat(v_unit(v_sub(hand, elbow)), 0.0))
        if not self.inputs_only:
            pose(self.pub_lhand, (-0.25, 0.78, -0.28), (0.0, 0.0, 0.0, 1.0))
            yaw = math.radians(-12.0 * (hand[0] / 0.3))
            pose(self.pub_head, (0.0, self.eye_h, 0.0), q_mul(q_axis(0, 1, 0, yaw), q_axis(1, 0, 0, math.radians(-head_pitch))))
        j = Joy()
        j.header.stamp, j.header.frame_id = now, self.frame
        j.axes, j.buttons = [trigger, 0.0], [weld, 0, 0]
        self.pub_inputs.publish(j)


class GlobalKey(threading.Thread):
    """Hold state of one key, read from every keyboard's /dev/input/eventN (independent of window focus)."""
    EVENT = struct.Struct("llHHi")        # struct input_event: timeval, type, code, value
    EV_KEY = 1
    CODES = {"t": 20, "m": 50, "space": 57}

    def __init__(self, key, on_change, log):
        super().__init__(daemon=True)
        self.code, self.on_change, self.log = self.CODES[key], on_change, log
        self.devices = []
        for block in open("/proc/bus/input/devices").read().split("\n\n"):
            if "kbd" in block and "EV=1" in block:          # keyboards with key events
                for word in block.split():
                    if word.startswith("event"):
                        self.devices.append(f"/dev/input/{word}")
        self.fds = []
        for d in self.devices:
            try:
                self.fds.append(os.open(d, os.O_RDONLY | os.O_NONBLOCK))
            except OSError:
                pass
        self.held = set()
        self.running = True

    @property
    def available(self):
        return bool(self.fds)

    def run(self):
        while self.running and self.fds:
            try:
                ready, _, _ = select.select(self.fds, [], [], 0.2)
            except (OSError, ValueError):
                break
            for fd in ready:
                try:
                    data = os.read(fd, self.EVENT.size * 64)
                except OSError:                              # keyboard unplugged: its key counts as released
                    self.fds.remove(fd)
                    self.held.discard(fd)
                    self.on_change(bool(self.held))
                    continue
                for i in range(0, len(data) - self.EVENT.size + 1, self.EVENT.size):
                    _, _, typ, code, value = self.EVENT.unpack_from(data, i)
                    if typ == self.EV_KEY and code == self.code:
                        if value:                          # 1 press, 2 auto-repeat
                            self.held.add(fd)
                        else:
                            self.held.discard(fd)
                        self.on_change(bool(self.held))
        self.on_change(False)


class Console:
    def __init__(self, node, state, lock):
        self.node, self.state, self.lock = node, state, lock
        self.root = tk.Tk()
        self.root.title("Tele-Craft operator console  (VR overlay preview, spectator)" if node.spectator else "Tele-Craft operator console  (OpenXR stand-in + VR overlay preview)")
        self.root.configure(bg="#1b2430")
        if node.geometry:
            self.root.geometry(node.geometry)
        left = tk.Frame(self.root, bg="#1b2430")
        left.grid(row=0, column=0, padx=12, pady=12)
        tk.Label(left, text="TORCH HAND  ·  top view of your reach", fg="#98a3b3", bg="#1b2430",
                 font=("DejaVu Sans", 10, "bold")).pack(anchor="w")
        self.W = self.H = 420
        self.canvas = tk.Canvas(left, width=self.W, height=self.H, bg="#0f141b", highlightthickness=0, cursor="crosshair")
        self.canvas.pack()
        tk.Label(left, text="mouse: move hand   wheel: lift/lower   HOLD LEFT BUTTON or M: trigger (robot moves)\n"
                            "RIGHT BUTTON or F: arc on/off (WELD mode)   arrows / W S: move / lift   Q E: tilt head   SPACE: toggle trigger\nR: reset hand   X: reset the cell after a FAULT",
                 fg="#98a3b3", bg="#1b2430", justify="left", font=("DejaVu Sans", 9)).pack(anchor="w", pady=(6, 0))
        self.height_bar = tk.Canvas(left, width=self.W, height=18, bg="#0f141b", highlightthickness=0)
        self.height_bar.pack(pady=(6, 0))

        right = tk.Frame(self.root, bg="#1b2430", width=360)
        right.grid(row=0, column=1, sticky="n", padx=(0, 12), pady=12)
        tk.Label(right, text="VR OVERLAY PREVIEW", fg="#98a3b3", bg="#1b2430", font=("DejaVu Sans", 10, "bold")).pack(anchor="w")
        self.prompt_box = tk.Frame(right, bg="#e6f4ea", width=360, height=150)
        self.prompt_box.pack(fill="x", pady=(4, 10))
        self.prompt_box.pack_propagate(False)
        self.prompt_title = tk.Label(self.prompt_box, text="Posture OK", bg="#e6f4ea", fg="#1d7a3a", font=("DejaVu Sans", 20, "bold"))
        self.prompt_title.pack(anchor="w", padx=12, pady=(12, 0))
        self.prompt_text = tk.Label(self.prompt_box, text="", bg="#e6f4ea", fg="#1b2430", font=("DejaVu Sans", 11), wraplength=330, justify="left")
        self.prompt_text.pack(anchor="w", padx=12, pady=(4, 0))
        self.prompt_score = tk.Label(self.prompt_box, text="RULA –", bg="#e6f4ea", fg="#1b2430", font=("DejaVu Sans Mono", 11))
        self.prompt_score.pack(anchor="w", padx=12, pady=(4, 0))
        self.auth = tk.Label(right, text="cell: –", fg="#e6ebf2", bg="#1b2430", font=("DejaVu Sans", 13, "bold"), anchor="w")
        self.auth.pack(fill="x")
        self.auth_reason = tk.Label(right, text="", fg="#98a3b3", bg="#1b2430", font=("DejaVu Sans", 9), anchor="w", wraplength=340, justify="left")
        self.auth_reason.pack(fill="x")
        self.stats = tk.Label(right, text="", fg="#98a3b3", bg="#1b2430", font=("DejaVu Sans Mono", 10), anchor="w", justify="left")
        self.stats.pack(fill="x", pady=(8, 0))
        self.why = tk.Label(right, text="", fg="#e8944a", bg="#1b2430", font=("DejaVu Sans", 10), anchor="w", justify="left", wraplength=340)
        self.why.pack(fill="x", pady=(4, 0))
        self.hand_lbl = tk.Label(right, text="", fg="#98a3b3", bg="#1b2430", font=("DejaVu Sans Mono", 10), anchor="w", justify="left")
        self.hand_lbl.pack(fill="x", pady=(8, 0))

        c = self.canvas
        c.bind("<Motion>", self.on_motion)
        c.bind("<ButtonPress-1>", lambda e: self.set_trigger(True))
        c.bind("<ButtonRelease-1>", lambda e: self.set_trigger(False))
        c.bind("<ButtonPress-3>", lambda e: self.toggle_weld())
        c.bind("<Button-4>", lambda e: self.lift(+0.02))
        c.bind("<Button-5>", lambda e: self.lift(-0.02))
        c.bind("<MouseWheel>", lambda e: self.lift(0.02 if e.delta > 0 else -0.02))
        self.reset_btn = tk.Button(right, text="Reset cell after FAULT  (X)", command=self.node.reset_cell,
                                   bg="#2c3542", fg="#e6ebf2", activebackground="#3f4a5a", activeforeground="#ffffff",
                                   relief="flat", padx=10, pady=4, font=("DejaVu Sans", 10, "bold"))
        self.reset_btn.pack(anchor="w", pady=(10, 0))
        self.root.bind("<Key>", self.on_key)
        for key in ("t", "T", "m", "M"):
            self.root.bind(f"<KeyPress-{key}>", self.on_m_press, add="+")
            self.root.bind(f"<KeyRelease-{key}>", self.on_m_release, add="+")
        self.root.bind("<FocusOut>", lambda e: self.focus_lost())
        self._m_job = None
        self.global_key = None
        if self.node.relay_topic:
            self.global_key = GlobalKey("t", self.on_global_key, self.node.get_logger())
            if self.global_key.available:
                self.global_key.start()
                self.node.get_logger().info(f"hold-to-run key T read from {len(self.global_key.fds)} keyboard devices "
                                            "(works whichever window has the focus)")
            else:
                self.node.get_logger().warning("cannot read the keyboard devices: T/M work only while the console has "
                                            "the focus (run scripts/install_keyboard_uaccess_udev.sh)")
        self._last = {}
        self.root.after(100, self.redraw)

    # ---- input --------------------------------------------------------------
    def on_motion(self, e):
        x = X_RANGE[0] + (e.x / self.W) * (X_RANGE[1] - X_RANGE[0])
        z = Z_RANGE[0] + (e.y / self.H) * (Z_RANGE[1] - Z_RANGE[0])
        with self.lock:
            self.state["hand"][0] = max(X_RANGE[0], min(X_RANGE[1], x))
            self.state["hand"][2] = max(Z_RANGE[0], min(Z_RANGE[1], z))

    def lift(self, dy):
        with self.lock:
            self.state["hand"][1] = max(Y_RANGE[0], min(Y_RANGE[1], self.state["hand"][1] + dy))

    def toggle_weld(self):
        with self.lock:
            self.state["weld"] = not self.state.get("weld", False)

    def set_trigger(self, on):
        with self.lock:
            self.state["trigger"] = on

    # M held = trigger. While a key is held X11 auto-repeat sends release+press pairs, so a release only counts
    # when no press follows within 60 ms. Losing the window focus releases it (a missed key-up must not keep the
    # robot enabled).
    def on_m_press(self, e):
        if self._m_job is not None:
            self.root.after_cancel(self._m_job)
            self._m_job = None
        with self.lock:
            self.state["m_trigger"] = True

    def on_m_release(self, e):
        if self._m_job is None:
            self._m_job = self.root.after(60, self.release_m)

    def release_m(self):
        if self._m_job is not None:
            self.root.after_cancel(self._m_job)
        self._m_job = None
        with self.lock:
            self.state["m_trigger"] = False

    def on_global_key(self, held):
        with self.lock:
            self.state["global_trigger"] = held

    def focus_lost(self):
        self.release_m()
        self.set_trigger(False)

    def on_key(self, e):
        k = e.keysym.lower()
        with self.lock:
            h = self.state["hand"]
            if k == "left":
                h[0] = max(X_RANGE[0], h[0] - 0.01)
            elif k == "right":
                h[0] = min(X_RANGE[1], h[0] + 0.01)
            elif k == "up":
                h[2] = max(Z_RANGE[0], h[2] - 0.01)
            elif k == "down":
                h[2] = min(Z_RANGE[1], h[2] + 0.01)
            elif k == "w":
                h[1] = min(Y_RANGE[1], h[1] + 0.01)
            elif k == "s":
                h[1] = max(Y_RANGE[0], h[1] - 0.01)
            elif k == "q":
                self.state["head_pitch"] = max(-15.0, self.state["head_pitch"] - 5.0)
            elif k == "e":
                self.state["head_pitch"] = min(45.0, self.state["head_pitch"] + 5.0)
            elif k == "space":
                self.state["trigger"] = not self.state["trigger"]
            elif k == "f":
                self.state["weld"] = not self.state.get("weld", False)
            elif k == "x":
                self.root.after(1, self.node.reset_cell)
            elif k == "r":
                self.state["hand"][:] = [0.28, 0.78, -0.30]
                self.state["head_pitch"] = 6.0
                self.state["trigger"] = False

    # ---- drawing --------------------------------------------------------------
    def redraw(self):
        with self.lock:
            hand = tuple(self.state["hand"])
            trig = self.state.get("trigger_out", False)
            src = self.state.get("src", "mouse")
            head_pitch = self.state["head_pitch"]
            prompt, rula, auth = self.state.get("prompt"), self.state.get("rula"), self.state.get("auth")
            moves, rejected = self.state.get("moves", 0), self.state.get("rejected", 0)
            last_reject = self.state.get("last_reject")
        c = self.canvas
        c.delete("all")
        for i in range(1, 7):
            c.create_line(0, i * self.H / 7, self.W, i * self.H / 7, fill="#1f2a36")
            c.create_line(i * self.W / 7, 0, i * self.W / 7, self.H, fill="#1f2a36")
        c.create_text(8, self.H - 10, text="near  (you)", anchor="w", fill="#5b6675", font=("DejaVu Sans", 8))
        c.create_text(8, 12, text="far", anchor="w", fill="#5b6675", font=("DejaVu Sans", 8))
        c.create_text(self.W - 8, self.H - 10, text="right", anchor="e", fill="#5b6675", font=("DejaVu Sans", 8))
        # comfortable zone (hand below the shoulder, in front): green tint
        cx = (0.0 - X_RANGE[0]) / (X_RANGE[1] - X_RANGE[0]) * self.W
        c.create_rectangle(cx - 60, self.H * 0.35, cx + 120, self.H * 0.85, outline="#1d7a3a", dash=(3, 4))
        c.create_text(cx + 30, self.H * 0.35 - 8, text="comfortable reach", fill="#1d7a3a", font=("DejaVu Sans", 8))
        px = (hand[0] - X_RANGE[0]) / (X_RANGE[1] - X_RANGE[0]) * self.W
        py = (hand[2] - Z_RANGE[0]) / (Z_RANGE[1] - Z_RANGE[0]) * self.H
        r = 10 + 12 * (hand[1] - Y_RANGE[0]) / (Y_RANGE[1] - Y_RANGE[0])
        color = "#ff5a3c" if trig else "#6fa0ff"
        c.create_oval(px - r, py - r, px + r, py + r, outline=color, width=3)
        c.create_line(px - r - 6, py, px + r + 6, py, fill=color)
        c.create_line(px, py - r - 6, px, py + r + 6, fill=color)
        c.create_text(px, py + r + 14, text=(f"TRIGGER: robot follows the {'TRACKER' if src == 'tracker' else 'mouse'}" if trig else "trigger released") + ("   ARC ON" if self.state.get("weld") else ""), fill=color, font=("DejaVu Sans", 9, "bold"))
        if self.node.relay_topic:
            try:
                focused = self.root.focus_displayof() is not None
            except (KeyError, tk.TclError):
                focused = False
            gk = self.global_key is not None and self.global_key.available
            if gk:
                text, col = "hold T (any window): tracker drives   hold mouse button: mouse drives", "#6fdc8c"
            elif focused:
                text, col = "hold T here: tracker drives   (T only works while this window has the focus)", "#f0a45c"
            else:
                text, col = "NO KEYBOARD FOCUS: click this window, then hold T", "#ff6b6b"
            c.create_text(self.W / 2, self.H - 14, text=text, fill=col, font=("DejaVu Sans", 10, "bold"))
        hb = self.height_bar
        hb.delete("all")
        frac = (hand[1] - Y_RANGE[0]) / (Y_RANGE[1] - Y_RANGE[0])
        hb.create_rectangle(0, 4, self.W, 14, outline="#1f2a36")
        hb.create_rectangle(0, 4, frac * self.W, 14, fill="#6fa0ff", outline="")
        hb.create_text(4, 9, text=f"hand height {hand[1]:.2f} m   (shoulder 1.09)", anchor="w", fill="#e6ebf2", font=("DejaVu Sans", 8))
        # HUD (only reconfigure what changed: Tk reconfigures are expensive and steal time from publishing)
        def setif(widget, key, **kw):
            if self._last.get(key) != kw:
                self._last[key] = kw
                widget.configure(**kw)
        level = prompt.level if prompt is not None else 0
        fg, bg = LEVEL_COLORS.get(level, LEVEL_COLORS[0])
        for i, w in enumerate((self.prompt_box, self.prompt_text, self.prompt_score)):
            setif(w, f"bg{i}", bg=bg)
        setif(self.prompt_title, "title", bg=bg, fg=fg, text=(prompt.title if prompt is not None else "waiting for RULA"))
        setif(self.prompt_text, "ptext", text=(prompt.text if prompt is not None else ""))
        if rula is not None and rula.valid:
            setif(self.prompt_score, "score", text=f"RULA {rula.score}  (action level {rula.action_level})   upper arm {rula.upper_arm_flexion_deg:.0f}°  neck {rula.neck_flexion_deg:.0f}°  wrist {rula.wrist_flexion_deg:.0f}°")
        if auth is not None:
            st = STATE_NAMES.get(auth.state, "?")
            col = {"ENABLED": "#6fdc8c", "READY": "#e6ebf2", "FAULT": "#ff5a3c", "EMERGENCY STOP": "#ff5a3c"}.get(st, "#e8944a")
            setif(self.auth, "auth", text=f"cell: {st}   mode {['STANDBY', 'MOVE', 'WELD'][auth.process_mode] if auth.process_mode < 3 else '?'}", fg=col)
            setif(self.auth_reason, "reason", text=auth.reason + ("   → press X or the button to reset" if st in ("FAULT", "SAFE STOP") else ""))
        setif(self.stats, "stats", text=f"moves executed {moves:5d}   rejected {rejected:4d}")
        setif(self.hand_lbl, "hand", text=f"hand xr  x {hand[0]:+.2f}  y {hand[1]:.2f}  z {hand[2]:+.2f}\nhead pitch {head_pitch:+.0f}°")
        self.root.after(100, self.redraw)

    def run(self):
        self.root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    state = {"hand": [0.28, 0.78, -0.30], "trigger": False, "head_pitch": 6.0}
    lock = threading.Lock()
    node = ConsoleNode(state, lock)
    ex = MultiThreadedExecutor(num_threads=2)
    ex.add_node(node)
    t = threading.Thread(target=ex.spin, daemon=True)
    t.start()
    try:
        Console(node, state, lock).run()
    except KeyboardInterrupt:
        pass
    finally:
        ex.shutdown(timeout_sec=1.0)
        t.join(timeout=2.0)                        # never leave the executor thread running into interpreter exit
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
