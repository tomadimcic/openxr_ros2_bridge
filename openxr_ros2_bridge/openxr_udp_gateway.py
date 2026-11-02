#!/usr/bin/env python3
# Copyright 2026 Entanglenet GmbH
# SPDX-License-Identifier: Apache-2.0
"""openxr_udp_gateway: receive a standalone headset's OpenXR poses over UDP and republish the raw /openxr/* topics.

A standalone headset (e.g. HTC VIVE Focus Vision) runs a small OpenXR application that cannot publish ROS 2
topics itself. It sends its poses and controller inputs as UDP datagrams to this node, which republishes
exactly what the console and the scripted welder publish, so openxr_ros2_bridge_node and everything after it
stay unchanged:

  out  /openxr/head/pose, /openxr/<hand>/pose, /openxr/trackers/<role>/pose   geometry_msgs/PoseStamped, frame openxr_stage
  out  /openxr/<hand>/inputs   sensor_msgs/Joy: axes [trigger, grip, stick x, stick y],
                                buttons [arc request, a, b, menu, stick click]
  out  ~/status                                                            std_msgs/String (JSON link statistics, 1 Hz)
  in   /tf, /tf_static     robot links for the digital twin shown in the headset (twin_links, twin_hz)
  in   scene_topic         visualization_msgs/MarkerArray: CUBE markers (e.g. the workpiece and its seam) shown in the
                           headset next to the twin, resent every second

It answers each headset with a status datagram (10 Hz) carrying the motion authority, the ergonomic prompt and
the per-device tracking validity, plus the timestamps the headset echoes back for clock synchronisation.

Protocol v1 (UTF-8 JSON, one object per datagram)
  headset -> PC (listen_port, default 9101):
    {"v":1,"kind":"pose","seq":N,"t":<headset monotonic ns when the poses were read>,
     "frame":"unity"|"openxr","origin":"floor","recenter":<count of tracking-origin changes>,
     "dev":[{"id":"head"|"right_hand"|"left_hand"|"tracker_<k>","tr":0|1,"st":<InputTrackingState bits>,
             "p":[x,y,z],"q":[x,y,z,w]}, ...],
     "in":[{"id":"right_hand","trigger":0..1,"grip":0..1,"a":0|1,"b":0|1,"menu":0|1,"stick":[x,y],"stick_click":0|1}, ...],
     "sync":{"t1":..,"t2":..,"t3":..,"t4":..},            # echo of the last status, see clock below
     "reply_port":P}                                      # optional: the port this app listens on
  PC -> headset (to the sender's address, at the datagram's reply_port, else the parameter reply_port, default 9102;
  the camera app listens on any free port and names it, since the twin app, paused in the background, can hold 9102):
    {"v":1,"kind":"status","seq":N,"t1":<"t" of the last pose datagram>,"t2":<PC receive ns>,"t3":<PC send ns>,
     "auth_state":..,"auth_mode":..,"auth_reason":..,"prompt_level":..,"prompt_title":..,"prompt_text":..,
     "rula_score":..,"rula_valid":..,"link_hz":..,"offset_ms":..,"rtt_ms":..,"tracking":[{"id":..,"ok":..}, ...]}
    {"v":1,"kind":"robot","seq":N,"links":[{"id":<link>,"p":[x,y,z],"q":[x,y,z,w]}, ...]}   # twin_hz, Unity stage
     coordinates: each robot link's pose from TF (world_frame -> link), placed with stage_in_world
    {"v":1,"kind":"scene","seq":N,"objects":[{"id":..,"p":[..],"q":[..],"s":[sx,sy,sz],"c":[r,g,b]}, ...]}  # 1 Hz,
     boxes in the same coordinates; s is the box size along the Unity axes (more often with show_trackers, below)

Conventions
  * Unity poses ("frame":"unity", left-handed, X right, Y up, Z forward) are converted to the OpenXR stage
    convention of the raw topics (right-handed, X right, Y up, -Z forward): p = (x, y, -z), q = (-qx, -qy, qz, qw).
  * A device is valid when its tracking state has both position (1) and rotation (2) bits, or "tr" is 1.
    Invalid devices are not published, so the bridge's input watchdog sees the loss; a stale pose is never repeated.
  * Rotation-only trackers: a VIVE Ultimate Tracker without position tracking (not localised in the headset's
    map) reports rotation only (3DoF) with a placeholder position that barely moves, yet OpenXR still marks it
    fully tracked. Such a tracker counts as lost (not published) when, within frozen_window_s, it turns by more
    than frozen_rot_deg while its position moves less than frozen_pos_m, or when two trackers are reported within
    placeholder_m of each other (25 Oct 2026: both arm trackers 12 mm apart, 0.6 m from the hand). It is
    accepted again once its position has moved unfreeze_m away from where it froze. A tracker whose pose does not
    change at all for frozen_window_s (tracker off or not updating: a live one always jitters) counts as lost
    until its pose changes.
  * Automatic tracker roles: tracker_roles = [auto] assigns auto_roles (default torch, right_forearm,
    right_upper_arm) by the mean distance of each tracker to the right controller over auto_window_s. Only
    trackers whose position has really moved (proven_m within frozen_window_s) take part, so a placeholder never
    gets a role. The first role (torch) goes alone to the nearest tracker within auto_first_max_m of the
    controller (it is taped to it; 25 Oct 2026: 28 mm, a forearm tracker is 0.1 m or more away). From then on the
    torch counts as lost while its offset from the controller, in the controller's frame, differs from its mean
    over the first 30 samples after assignment by more than pair_tol_m (rigid pair on the real headset, 25 Oct
    2026: at most 20 mm while turning 95 deg and moving 25 cm) (one of the two lost its position: a jump keeps the distance
    but not this offset); the turn-without-moving check is not applied to it, since
    twisting the torch turns the tracker nearly in place (25 Oct 2026: false losses, latched safety faults), so the robot does not wait for the arm trackers; the other roles, nearest first, once that many
    more trackers have a position. Until then the panel note says what is missing. The headset may number the
    trackers differently in every session, so this runs again when the application restarts, or on
    ~/reassign_roles. Trackers without a role are published as /openxr/trackers/tracker_<k>/pose.
  * Trackers on the PC instead (publish_trackers false): the headset's own trackers are ignored, and the trackers of
    vive_tracker_pc (its ~/status JSON on extra_status_topic) are listed on the panel instead. The headset app only
    draws its own trackers, so show_trackers roles (their /openxr/trackers/<role>/pose, already aligned with the
    headset) are sent as small coloured boxes with the scene; a tracker silent for 0.5 s is hidden. The app moves a
    box only when a scene datagram arrives and applies only the newest one per frame, so a datagram goes out with
    every new tracker sample, at most show_trackers_hz (above a tracker's 125 Hz: a 120 Hz cap let only every
    second 8 ms sample through), each with the workpiece boxes, converted once a second. 26 Oct 2026, on a 20 Hz
    timer: the boxes stood still in 78 % of the headset frames and trailed the torch by 33 ms (median; p90 54 ms);
    now in 0 % and by 10 ms (test/tracker_boxes_check.py). With show_trackers_predict a box's position is moved
    along its velocity over the last 40 ms to when the headset shows it (the sample's age, half the round trip and
    half a headset frame, about 15-20 ms): then within 2 ms of the torch. Tracker and controller stamps agree within
    5 ms (26 Oct 2026 recordings), so the predicted box stays on the controller the torch is taped to.
  * Operator notes: a std_msgs/String on note_topic (e.g. "arm calibrated" from openxr_body) is shown on the
    headset panel for note_s seconds ("note" field of the status datagram), on its own line below the gateway's
    own tracker-role note, so neither hides the other.
  * Twin placement: stage_in_world = [x, y, z, yaw_deg] is where the operator's stage origin (floor, facing its
    forward direction) sits in world_frame; the default puts the operator where the RViz scene shows the welder.
  * Clock: the headset echoes (t1 = its pose time, t2 = PC receive time, t3 = PC status send time, t4 = its status
    receive time). offset = ((t2 - t1) + (t3 - t4)) / 2, rtt = (t4 - t1) - (t3 - t2); the sample with the smallest
    round trip in the last `sync_window` is used. Header stamps are headset sample time + offset, i.e. PC time,
    so latency measurements downstream stay valid. Until the first sync the PC receive time is used.
"""
import json
import math
import socket
import threading
import time
from collections import deque

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Joy
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
from std_srvs.srv import Trigger
from builtin_interfaces.msg import Time
from rclpy.time import Time as RclTime
import tf2_ros

try:
    from rula_msgs.msg import ErgoPrompt, RulaScore
except ImportError:        # the gateway stays usable without the RULA module
    ErgoPrompt = RulaScore = None
try:
    from telecraft_msgs.msg import MotionAuthority
except ImportError:        # ... and without the Tele-Craft cell messages
    MotionAuthority = None

AUTH_STATES = ["NOT_READY", "READY", "ENABLED", "PAUSED", "SAFE_STOP", "EMERGENCY_STOP", "FAULT"]
AUTH_MODES = ["STANDBY", "MOVE", "WELD"]
TRACKED_POSITION, TRACKED_ROTATION = 1, 2


def unity_to_openxr(p, q):
    """Unity (left-handed, Z forward) -> OpenXR (right-handed, -Z forward), both Y up."""
    return (p[0], p[1], -p[2]), (-q[0], -q[1], q[2], q[3])


def angle_between(q0, q1):
    """Rotation angle between two unit quaternions, degrees."""
    d = abs(sum(a * b for a, b in zip(q0, q1)))
    return math.degrees(2.0 * math.acos(min(1.0, d)))


def ros_to_unity(p, q):
    """ROS REP-103 (x fwd, y left, z up; right-handed) -> Unity (x right, y up, z fwd; left-handed)."""
    return (-p[1], p[2], p[0]), (q[1], -q[2], -q[0], q[3])


def ns_to_time(ns):
    t = Time()
    t.sec, t.nanosec = int(ns // 1_000_000_000), int(ns % 1_000_000_000)
    return t


class ClockSync:
    """Minimum-round-trip offset estimate between the headset clock and the PC clock (ns)."""

    def __init__(self, window):
        self.samples = deque(maxlen=window)

    def add(self, t1, t2, t3, t4):
        rtt = (t4 - t1) - (t3 - t2)
        if rtt < 0:          # inconsistent echo (e.g. headset restarted); ignore
            return
        self.samples.append((rtt, ((t2 - t1) + (t3 - t4)) / 2.0))

    def best(self):
        if not self.samples:
            return None, None
        rtt, offset = min(self.samples)
        return offset, rtt


class UdpGateway(Node):
    def __init__(self):
        super().__init__("openxr_udp_gateway")
        self.declare_parameter("listen_port", 9101)
        self.declare_parameter("reply_port", 9102)
        self.declare_parameter("frame_id", "openxr_stage")
        self.declare_parameter("status_hz", 10.0)
        self.declare_parameter("sync_window", 200)            # sync samples kept (~20 s at 10 Hz)
        # tracker index -> role; tracker k is published on /openxr/trackers/<role>/pose (default role: tracker_<k>)
        self.declare_parameter("tracker_roles", ["chest", "right_elbow"])
        self.declare_parameter("arc_button", "a")              # controller button reported as Joy buttons[0] (arc request)
        self.declare_parameter("frozen_pos_m", 0.003)          # rotation-only detection, see module docstring
        self.declare_parameter("frozen_window_s", 1.0)
        self.declare_parameter("frozen_rot_deg", 15.0)
        self.declare_parameter("unfreeze_m", 0.02)
        self.declare_parameter("placeholder_m", 0.03)
        self.declare_parameter("proven_m", 0.03)
        self.declare_parameter("twin_links", [""])             # robot links streamed to the headset ([""] = none)
        self.declare_parameter("twin_hz", 30.0)
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("stage_in_world", [-2.0, 1.2, 0.0, 0.0])
        self.declare_parameter("scene_topic", "/telecraft/workpiece")    # "" = no scene objects
        self.declare_parameter("auto_roles", ["torch", "right_forearm", "right_upper_arm"])   # nearest to the controller first
        self.declare_parameter("auto_window_s", 2.0)
        self.declare_parameter("auto_first_max_m", 0.08)
        self.declare_parameter("pair_tol_m", 0.03)
        self.declare_parameter("note_topic", "/telecraft/operator_note")
        self.declare_parameter("publish_trackers", True)       # false: trackers come from the PC (vive_tracker_pc)
        self.declare_parameter("extra_status_topic", "")       # vive_tracker_pc ~/status: its trackers on the panel
        self.declare_parameter("show_trackers", [""])          # roles drawn in the headset as boxes (PC trackers)
        self.declare_parameter("show_trackers_hz", 150.0)      # scene datagrams per second at most (on new samples)
        self.declare_parameter("show_trackers_predict", True)  # boxes predicted to when the headset shows them
        self.declare_parameter("note_s", 6.0)
        p = self.get_parameter
        self.frame = str(p("frame_id").value)
        self.reply_port = int(p("reply_port").value)
        self.roles = [str(r) for r in p("tracker_roles").value]
        self.auto = self.roles == ["auto"]
        self.auto_roles = [str(r) for r in p("auto_roles").value]
        self.auto_window = float(p("auto_window_s").value)
        self.auto_first_max = float(p("auto_first_max_m").value)
        self.pair_tol = float(p("pair_tol_m").value)
        self.pair_active, self.pair_vec, self.pair_bad = False, None, False   # torch offset in the controller frame
        self.assigned = {}                 # tracker_<k> -> role (automatic roles)
        self.auto_acc, self.auto_t0, self.auto_n = {}, None, 0
        self.notes, self.note_s = {}, float(p("note_s").value)   # slot ("roles", "ext") -> (text, time)
        self.publish_trackers = bool(p("publish_trackers").value)
        if not self.publish_trackers:
            self.auto = False
        self.extra_tracking, self.extra_t = [], 0.0
        self.show_roles = [str(r) for r in p("show_trackers").value if str(r)]
        self.show_period = 1.0 / float(p("show_trackers_hz").value)
        self.show_predict = bool(p("show_trackers_predict").value)
        self.shown = {}                    # role -> recent (stamp s, Unity position, Unity rotation, monotonic time)
        if str(p("extra_status_topic").value):
            self.create_subscription(String, str(p("extra_status_topic").value), self.on_extra_status, 10)
        self.create_subscription(String, str(p("note_topic").value), self.on_note, 10)
        self.create_service(Trigger, "~/reassign_roles", self.srv_reassign)
        self.arc_button = str(p("arc_button").value)
        self.sync = ClockSync(int(p("sync_window").value))
        self.frozen_pos = float(p("frozen_pos_m").value)
        self.frozen_window = float(p("frozen_window_s").value)
        self.frozen_rot = float(p("frozen_rot_deg").value)
        self.unfreeze_m = float(p("unfreeze_m").value)
        self.placeholder_m = float(p("placeholder_m").value)
        self.proven_m = float(p("proven_m").value)
        self.motion = {}                   # tracker id -> recent samples, rotation-only flag, position proven
        self.twin_links = [str(x) for x in p("twin_links").value if str(x)]
        self.world_frame = str(p("world_frame").value)
        sx, sy, sz, syaw = (float(v) for v in p("stage_in_world").value)
        self.stage_t, self.stage_yaw = (sx, sy, sz), math.radians(syaw)
        self.n_twin = 0

        self.qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pose_pubs, self.joy_pubs = {}, {}
        self.pub_status = self.create_publisher(String, "~/status", 10)

        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.auth = self.prompt = self.rula = None
        if MotionAuthority is not None:
            self.create_subscription(MotionAuthority, "/telecraft/authority", lambda m: setattr(self, "auth", m), latched)
        if ErgoPrompt is not None:
            self.create_subscription(ErgoPrompt, "/human/ergo_prompt", lambda m: setattr(self, "prompt", m), latched)
            self.create_subscription(RulaScore, "/human/risk_score", lambda m: setattr(self, "rula", m), 10)

        self.lock = threading.Lock()
        self.peer = None                   # (ip, port) of the headset, learned from its datagrams
        self.last_t1 = self.last_t2 = None
        self.last_rx_mono = None
        self.valid = {}                    # device id -> bool, from the latest datagram
        self.n_rx = self.n_bad = self.n_gap = self.n_reply = 0
        self.last_seq = None
        self.recenter = None
        self.window_rx = 0
        self.rate_hz = 0.0
        self.half_rtt = 0.0                # s, from the clock sync

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", int(p("listen_port").value)))
        self.sock.settimeout(0.5)
        self.running = True
        self.rx_thread = threading.Thread(target=self.rx_loop, daemon=True)
        self.rx_thread.start()

        self.create_timer(1.0 / float(p("status_hz").value), self.send_status)
        self.scene = {}                    # marker key -> Marker (CUBE)
        self.scene_objs, self.scene_objs_t = [], None    # self.scene in Unity stage coordinates, and when converted
        self.scene_sent = 0.0                            # monotonic time of the last scene datagram
        self.n_scene = 0
        if self.twin_links:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
            self.create_timer(1.0 / float(p("twin_hz").value), self.send_twin)
            scene_topic = str(p("scene_topic").value)
            if scene_topic:
                self.create_subscription(MarkerArray, scene_topic, self.on_scene, latched)
                for role in self.show_roles:
                    self.create_subscription(PoseStamped, f"/openxr/trackers/{role}/pose",
                                             lambda m, r=role: self.on_shown(r, m), self.qos)
                self.create_timer(0.25 if self.show_roles else 1.0, self.scene_tick)
        self.create_timer(1.0, self.report)
        self.get_logger().info(f"openxr_udp_gateway: listening on UDP {p('listen_port').value}, replies to port "
                               f"{self.reply_port}; tracker roles {self.roles}; twin links {len(self.twin_links)} "
                               f"from {self.world_frame}, operator stage at {list(self.stage_t)} yaw {syaw} deg")

    # ------------------------------------------------------------------ receive
    def topic_for(self, dev_id):
        if dev_id.startswith("tracker_"):
            if self.auto:
                return f"/openxr/trackers/{self.assigned.get(dev_id, dev_id)}/pose"
            try:
                k = int(dev_id.split("_", 1)[1])
            except ValueError:
                return None
            role = self.roles[k] if k < len(self.roles) and self.roles[k] else dev_id
            return f"/openxr/trackers/{role}/pose"
        if dev_id in ("head", "right_hand", "left_hand"):
            return f"/openxr/{dev_id}/pose"
        return None

    def rx_loop(self):
        while self.running and rclpy.ok():
            try:
                data, addr = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            t_rx = time.time_ns()
            try:
                msg = json.loads(data.decode("utf-8"))
                if msg.get("v") != 1 or msg.get("kind") != "pose":
                    raise ValueError("not a v1 pose datagram")
                self.on_datagram(msg, addr, t_rx)
            except Exception as e:           # malformed datagrams are counted, never fatal
                self.n_bad += 1
                if self.n_bad < 5 or self.n_bad % 100 == 0:
                    self.get_logger().warning(f"bad datagram from {addr[0]}: {e}")

    def on_datagram(self, msg, addr, t_rx):
        seq, t_h = int(msg["seq"]), int(msg["t"])
        port = msg.get("reply_port", self.reply_port)
        if not isinstance(port, int) or not 0 < port < 65536:
            port = self.reply_port
        with self.lock:
            if self.peer != (addr[0], port):
                self.get_logger().info(f"headset connected from {addr[0]}:{addr[1]}, replies to port {port}")
            self.peer = (addr[0], port)
            if self.last_seq is not None and seq > self.last_seq + 1:
                self.n_gap += seq - self.last_seq - 1
            if self.last_seq is not None and seq < self.last_seq - 1000:
                self.get_logger().info("headset sequence restarted (application restarted)")
                self.sync = ClockSync(self.sync.samples.maxlen)
                if self.auto and self.assigned:
                    self.get_logger().info("tracker numbering may have changed: assigning the roles again")
                self.assigned, self.auto_acc, self.auto_t0, self.auto_n, self.motion = {}, {}, None, 0, {}
                self.pair_active, self.pair_vec, self.pair_bad = False, None, False
            self.last_seq = seq
            self.last_t1, self.last_t2 = t_h, t_rx
            self.last_rx_mono = time.monotonic()
            self.n_rx += 1
            self.window_rx += 1
            s = msg.get("sync")
            if s:
                self.sync.add(int(s["t1"]), int(s["t2"]), int(s["t3"]), int(s["t4"]))
            offset, _ = self.sync.best()
            rc = msg.get("recenter")
            if rc is not None and self.recenter is not None and rc != self.recenter:
                self.get_logger().warning(f"headset tracking origin changed (recenter {self.recenter} -> {rc}); "
                                       "absolute calibrations are invalid until redone")
            self.recenter = rc
        stamp = ns_to_time(t_h + offset if offset is not None else t_rx)
        unity = msg.get("frame", "unity") == "unity"

        devs = msg.get("dev", [])
        if not self.publish_trackers:
            devs = [d for d in devs if not str(d.get("id", "")).startswith("tracker_")]
        now = time.monotonic()
        flagged = {str(d.get("id", "")): (int(d.get("st", 0)) & (TRACKED_POSITION | TRACKED_ROTATION))
                   == (TRACKED_POSITION | TRACKED_ROTATION) or int(d.get("tr", 0)) == 1 for d in devs}
        self.mark_placeholders([d for d in devs if str(d.get("id", "")).startswith("tracker_")
                                and flagged[str(d.get("id", ""))]])
        torch_id = next((k for k, r in self.assigned.items() if r == self.auto_roles[0]), None)
        hand = next((d for d in devs if d.get("id") == "right_hand" and flagged.get("right_hand")), None)
        valid = {}
        for d in devs:
            dev_id = str(d.get("id", ""))
            ok = flagged[dev_id]
            if ok and dev_id.startswith("tracker_"):
                torch = dev_id == torch_id and self.pair_active
                ok = not self.rotation_only(dev_id, d["p"], d["q"], now, turn_check=not torch)
                if ok and torch:
                    ok = not self.torch_off_controller(d, hand, unity)
            valid[dev_id] = ok
            topic = self.topic_for(dev_id)
            if not ok or topic is None:
                continue
            pos, quat = d["p"], d["q"]
            if unity:
                pos, quat = unity_to_openxr(pos, quat)
            pub = self.pose_pubs.get(topic)
            if pub is None:
                pub = self.pose_pubs[topic] = self.create_publisher(PoseStamped, topic, self.qos)
                self.get_logger().info(f"publishing {dev_id} on {topic}")
            m = PoseStamped()
            m.header.stamp, m.header.frame_id = stamp, self.frame
            m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in pos)
            (m.pose.orientation.x, m.pose.orientation.y,
             m.pose.orientation.z, m.pose.orientation.w) = (float(v) for v in quat)
            pub.publish(m)

        for i in msg.get("in", []):
            hand = str(i.get("id", ""))
            if hand not in ("right_hand", "left_hand") or not valid.get(hand, False):
                continue          # no inputs from an untracked controller: the trigger must not count
            pub = self.joy_pubs.get(hand)
            if pub is None:
                pub = self.joy_pubs[hand] = self.create_publisher(Joy, f"/openxr/{hand}/inputs", self.qos)
            stick = i.get("stick", [0.0, 0.0])
            j = Joy()
            j.header.stamp, j.header.frame_id = stamp, self.frame
            j.axes = [float(i.get("trigger", 0.0)), float(i.get("grip", 0.0)), float(stick[0]), float(stick[1])]
            # buttons: [arc request (arc_button), a, b, menu, stick click]; the bridge reads buttons[0] as the arc request
            j.buttons = [int(i.get(self.arc_button, 0)), int(i.get("a", 0)), int(i.get("b", 0)),
                         int(i.get("menu", 0)), int(i.get("stick_click", 0))]
            pub.publish(j)
        if self.auto and len(self.assigned) < len(self.auto_roles):
            self.auto_assign([d for d in devs if valid.get(str(d.get("id", "")))],
                             sum(1 for k in flagged if k.startswith("tracker_") and k not in self.assigned))
        with self.lock:
            self.valid = valid

    def auto_assign(self, devs, n_free):
        """Automatic roles (module docstring). devs: devices valid in this datagram; n_free: trackers in it without
        a role, tracked or not (only then is a missing arm tracker worth a note)."""
        hand = next((d for d in devs if d.get("id") == "right_hand"), None)
        cand = [d for d in devs if str(d.get("id", "")).startswith("tracker_") and d["id"] not in self.assigned
                and self.motion_state(d["id"])["proven"]]
        first = self.auto_roles[0] not in self.assigned.values()
        rest = [r for r in self.auto_roles[1:] if r not in self.assigned.values()]
        now = time.monotonic()
        if hand is None or not cand or (not first and len(cand) < len(rest)):
            self.auto_acc, self.auto_t0, self.auto_n = {}, None, 0
            wait = ""
            if first:
                wait = (f"tracker roles: waiting for the right controller and a tracker on it "
                        f"(controller {'ok' if hand else 'not tracked'})")
            elif n_free:
                wait = (f"{' + '.join(rest)}: {len(cand)} of {len(rest)} trackers have a position - hold each in "
                        "front of you, cameras out, until its blue number shows")
            if wait:
                self.set_note(wait)
                self.get_logger().info(wait, throttle_duration_sec=10.0)
            return
        if self.auto_t0 is None:
            self.auto_t0 = now
        self.auto_n += 1
        for d in cand:
            a = self.auto_acc.setdefault(d["id"], [0.0, 0])
            a[0] += math.dist(d["p"], hand["p"])
            a[1] += 1
        if now - self.auto_t0 < self.auto_window:
            return
        mean = sorted((a[0] / a[1], tid) for tid, a in self.auto_acc.items() if a[1] >= 0.8 * self.auto_n)
        self.auto_acc, self.auto_t0, self.auto_n = {}, None, 0
        dist = {tid: m for m, tid in mean}
        new = {}
        if first:
            if not mean or mean[0][0] > self.auto_first_max:
                self.set_note("tracker roles: no tracker on the right controller"
                              + (f" (nearest {mean[0][0]:.2f} m)" if mean else ""))
                return
            new[mean[0][1]] = self.auto_roles[0]
            self.pair_active, self.pair_vec, self.pair_bad = True, None, False
            mean = mean[1:]
        if rest and len(mean) >= len(rest):
            new.update({tid: role for (_, tid), role in zip(mean, rest)})
        self.assigned.update(new)
        self.get_logger().info("automatic tracker roles: " + ", ".join(
            f"{tid} -> {role} ({dist[tid]:.2f} m from the controller)" for tid, role in new.items()))
        self.set_note("trackers: " + ", ".join(new.values()) + " assigned")

    def srv_reassign(self, rq, rs):
        self.assigned, self.auto_acc, self.auto_t0, self.auto_n = {}, {}, None, 0
        self.pair_active, self.pair_vec, self.pair_bad = False, None, False
        rs.success, rs.message = True, "tracker roles will be assigned again from the next datagrams"
        self.get_logger().info(rs.message)
        return rs

    def on_shown(self, role, m):
        p, q = m.pose.position, m.pose.orientation                      # OpenXR stage -> Unity stage
        h = self.shown.setdefault(role, deque(maxlen=16))               # ~0.13 s at 125 Hz
        h.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, (p.x, p.y, -p.z), (-q.x, -q.y, q.z, q.w),
                  time.monotonic()))
        if h[-1][3] - self.scene_sent >= self.show_period:
            self.send_scene()

    def on_extra_status(self, msg):
        try:
            trackers = json.loads(msg.data).get("trackers", {})
        except ValueError:
            return
        self.extra_tracking = [{"id": role, "ok": int(t.get("published_hz", 0)) > 0} for role, t in sorted(trackers.items())]
        self.extra_t = time.monotonic()

    def on_note(self, msg):
        self.set_note(msg.data, "ext")

    def set_note(self, text, slot="roles"):
        self.notes[slot] = (text, time.monotonic())

    def note_text(self):
        now = time.monotonic()
        return "\n".join(t for slot in ("roles", "ext") for t, ts in [self.notes.get(slot, ("", 0.0))]
                         if t and now - ts < self.note_s)

    def motion_state(self, dev_id):
        m = self.motion.get(dev_id)
        if m is None:
            m = self.motion[dev_id] = {"hist": deque(), "frozen": False, "ref": None, "proven": False,
                                       "stale": False, "refq": None}
        return m

    def torch_off_controller(self, d, hand, unity):
        """True while the torch tracker is not where it sits on the controller (see module docstring)."""
        if hand is None:
            return self.pair_bad                       # cannot judge without the controller: keep the last verdict
        pt, _ = unity_to_openxr(d["p"], d["q"]) if unity else (d["p"], d["q"])
        pc, qc = unity_to_openxr(hand["p"], hand["q"]) if unity else (hand["p"], hand["q"])
        x, y, z, w = qc
        r = ((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
             (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
             (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)))
        dv = [pt[i] - pc[i] for i in range(3)]
        v = [sum(r[j][i] * dv[j] for j in range(3)) for i in range(3)]       # offset in the controller frame
        if self.pair_vec is None or self.pair_vec[3] < 30:              # reference: mean of the first samples
            acc = self.pair_vec or [0.0, 0.0, 0.0, 0]
            self.pair_vec = [acc[0] + v[0], acc[1] + v[1], acc[2] + v[2], acc[3] + 1]
            return False
        n = self.pair_vec[3]
        err = math.dist(v, [c / n for c in self.pair_vec[:3]])
        if self.pair_bad and err < self.pair_tol / 3:
            self.pair_bad = False
            self.get_logger().info(f"{d['id']} (torch): back on the controller, accepted again")
        elif not self.pair_bad and err > self.pair_tol:
            self.pair_bad = True
            self.get_logger().warning(f"{d['id']} (torch): {err * 100:.0f} cm off its place on the controller "
                                   "(tracking lost), treated as lost")
        return self.pair_bad

    def rotation_only(self, dev_id, pos, quat, now, turn_check=True):
        """True while a tracker has no usable position: it turns without moving (rotation-only tracking) or its
        pose does not change at all (not updating), see module docstring."""
        m = self.motion_state(dev_id)
        if m["stale"]:
            if list(pos) == m["ref"] and list(quat) == m["refq"]:
                return True
            m["stale"] = False
            m["hist"].clear()
            self.get_logger().info(f"{dev_id}: pose updates again, accepted")
        if m["frozen"]:
            if math.dist(pos, m["ref"]) < self.unfreeze_m:
                return True
            m["frozen"] = False
            m["hist"].clear()
            self.get_logger().info(f"{dev_id}: position tracking is back, accepted again")
        h = m["hist"]
        h.append((now, pos, quat))
        while now - h[0][0] > self.frozen_window:
            h.popleft()
        if now - h[0][0] < 0.8 * self.frozen_window:
            return False
        span = math.dist([min(x[1][i] for x in h) for i in range(3)], [max(x[1][i] for x in h) for i in range(3)])
        if span == 0.0 and all(x[2] == h[0][2] for x in h):
            m["stale"], m["ref"], m["refq"] = True, list(pos), list(quat)
            self.get_logger().warning(f"{dev_id}: pose unchanged for {self.frozen_window:.1f} s (tracker off or not "
                                   "updating), treated as lost")
            return True
        if span > self.proven_m:
            m["proven"] = True
        elif turn_check and span < self.frozen_pos and max(angle_between(h[0][2], x[2]) for x in h) > self.frozen_rot:
            m["frozen"], m["ref"] = True, list(pos)
            self.get_logger().warning(f"{dev_id}: turns without moving (rotation-only tracking, no position), "
                                   "treated as lost")
        return m["frozen"]

    def mark_placeholders(self, trackers):
        """Two trackers within placeholder_m of each other cannot both be real: both are rotation-only."""
        for i, a in enumerate(trackers):
            for b in trackers[i + 1:]:
                if math.dist(a["p"], b["p"]) >= self.placeholder_m:
                    continue
                for d in (a, b):
                    m = self.motion_state(str(d["id"]))
                    if m["frozen"]:
                        m["ref"] = list(d["p"])          # still twins: no "back" while they drift together
                    else:
                        m["frozen"], m["ref"] = True, list(d["p"])
                        self.get_logger().warning(f"{d['id']}: reported at the same spot as another tracker "
                                               "(rotation-only placeholder, no position), treated as lost")

    # ------------------------------------------------------------------ twin
    def to_stage_unity(self, frame, pos, quat):
        """Pose in `frame` -> Unity stage coordinates (via world_frame and stage_in_world)."""
        tf = self.tf_buffer.lookup_transform(self.world_frame, frame, RclTime())
        t, r = tf.transform.translation, tf.transform.rotation
        # world pose = T_world_frame * (pos, quat)
        rm = ((1 - 2 * (r.y * r.y + r.z * r.z), 2 * (r.x * r.y - r.z * r.w), 2 * (r.x * r.z + r.y * r.w)),
              (2 * (r.x * r.y + r.z * r.w), 1 - 2 * (r.x * r.x + r.z * r.z), 2 * (r.y * r.z - r.x * r.w)),
              (2 * (r.x * r.z - r.y * r.w), 2 * (r.y * r.z + r.x * r.w), 1 - 2 * (r.x * r.x + r.y * r.y)))
        pw = tuple(sum(rm[i][j] * pos[j] for j in range(3)) + (t.x, t.y, t.z)[i] for i in range(3))
        x0, y0, z0, w0 = r.x, r.y, r.z, r.w
        x1, y1, z1, w1 = quat
        qw = (w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1, w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
              w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1, w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1)
        return self.world_to_stage_unity(pw, qw)

    def world_to_stage_unity(self, pw, qw):
        c, s_ = math.cos(-self.stage_yaw), math.sin(-self.stage_yaw)
        dx, dy, dz = pw[0] - self.stage_t[0], pw[1] - self.stage_t[1], pw[2] - self.stage_t[2]
        p_st = (c * dx - s_ * dy, s_ * dx + c * dy, dz)
        x0, y0, z0, w0 = 0.0, 0.0, math.sin(-self.stage_yaw / 2.0), math.cos(-self.stage_yaw / 2.0)
        x1, y1, z1, w1 = qw
        q_st = (w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1, w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1, w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1)
        return ros_to_unity(p_st, q_st)

    def on_scene(self, msg: MarkerArray):
        for m in msg.markers:
            key = f"{m.ns}/{m.id}"
            if m.action == Marker.DELETEALL:
                self.scene.clear()
            elif m.action == Marker.DELETE:
                self.scene.pop(key, None)
            elif m.type == Marker.CUBE:
                self.scene[key] = m
        self.scene_objs_t = None                                       # convert again at the next datagram

    def scene_objects(self):
        objects = []
        for key, m in self.scene.items():
            try:
                pu, qu = self.to_stage_unity(m.header.frame_id, (m.pose.position.x, m.pose.position.y, m.pose.position.z),
                                             (m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z,
                                              m.pose.orientation.w))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                continue
            objects.append({"id": m.text or key, "p": [round(v, 5) for v in pu], "q": [round(v, 6) for v in qu],
                            "s": [round(m.scale.y, 5), round(m.scale.z, 5), round(m.scale.x, 5)],   # ROS (x,y,z) -> Unity
                            "c": [round(m.color.r, 3), round(m.color.g, 3), round(m.color.b, 3)]})
        return objects

    @staticmethod
    def predicted(h, lead):
        """Position of the newest sample in h moved lead seconds along the velocity of the last 40 ms."""
        t1, p1 = h[-1][0], h[-1][1]
        for t0, p0, _, _ in h:                                          # oldest sample within 40 ms
            if t1 - t0 <= 0.04:
                break
        if t1 - t0 < 0.02 or not 0.0 < lead < 0.1:                      # too few samples, or a stale one
            return p1
        return tuple(a + (a - b) / (t1 - t0) * lead for a, b in zip(p1, p0))

    def scene_tick(self):
        if time.monotonic() - self.scene_sent > 0.2:    # no tracker samples: the workpiece, silent trackers hidden
            self.send_scene()

    def send_scene(self):
        with self.lock:
            peer = self.peer
            fresh = self.last_rx_mono is not None and time.monotonic() - self.last_rx_mono < 1.0
        if peer is None or not fresh or not (self.scene or self.show_roles):
            return
        now = time.monotonic()
        if self.scene_objs_t is None or now - self.scene_objs_t >= 1.0:
            self.scene_objs, self.scene_objs_t = self.scene_objects(), now
        objects = list(self.scene_objs)
        colors = {"torch": [1.0, 0.45, 0.05], "right_forearm": [0.1, 0.8, 0.9], "right_upper_arm": [0.2, 0.4, 1.0]}
        ahead = time.time() + self.half_rtt + 0.5 / max(self.rate_hz, 30.0)   # PC time the headset shows it
        for role in self.show_roles:
            h = self.shown.get(role)
            pu, qu, size = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0.0
            if h:
                t, pu, qu, rx = h[-1]
                size = 0.05 if now - rx < 0.5 else 0.0                    # silent tracker: hidden
                if self.show_predict and size:
                    pu = self.predicted(h, ahead - t)
            objects.append({"id": "tracker:" + role, "p": [round(v, 5) for v in pu], "q": [round(v, 6) for v in qu],
                            "s": [size, size, size], "c": colors.get(role, [0.9, 0.9, 0.2])})
        if not objects:
            return
        self.scene_sent = now
        msg = {"v": 1, "kind": "scene", "seq": self.n_scene, "objects": objects}
        try:
            self.sock.sendto(json.dumps(msg, separators=(",", ":")).encode("utf-8"), peer)
            self.n_scene += 1
        except OSError as e:
            self.get_logger().warning(f"scene to {peer[0]} failed: {e}", throttle_duration_sec=5.0)

    def send_twin(self):
        with self.lock:
            peer = self.peer
            fresh = self.last_rx_mono is not None and time.monotonic() - self.last_rx_mono < 1.0
        if peer is None or not fresh:
            return
        c, s_ = math.cos(-self.stage_yaw), math.sin(-self.stage_yaw)
        qz = (0.0, 0.0, math.sin(-self.stage_yaw / 2.0), math.cos(-self.stage_yaw / 2.0))
        links = []
        for name in self.twin_links:
            try:
                tf = self.tf_buffer.lookup_transform(self.world_frame, name, RclTime())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                continue
            t, r = tf.transform.translation, tf.transform.rotation
            dx, dy, dz = t.x - self.stage_t[0], t.y - self.stage_t[1], t.z - self.stage_t[2]
            p_st = (c * dx - s_ * dy, s_ * dx + c * dy, dz)                # world -> stage (REP-103 axes)
            x0, y0, z0, w0 = qz                                            # q_stage = q_yaw^-1 * q_world
            x1, y1, z1, w1 = r.x, r.y, r.z, r.w
            q_st = (w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                    w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                    w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
                    w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1)
            pu, qu = ros_to_unity(p_st, q_st)
            links.append({"id": name, "p": [round(v, 5) for v in pu], "q": [round(v, 6) for v in qu]})
        if not links:
            return
        msg = {"v": 1, "kind": "robot", "seq": self.n_twin, "links": links}
        try:
            self.sock.sendto(json.dumps(msg, separators=(",", ":")).encode("utf-8"), peer)
            self.n_twin += 1
        except OSError as e:
            self.get_logger().warning(f"twin to {peer[0]} failed: {e}", throttle_duration_sec=5.0)

    # ------------------------------------------------------------------ reply
    def send_status(self):
        with self.lock:
            peer, t1, t2 = self.peer, self.last_t1, self.last_t2
            valid = dict(self.valid)
            fresh = self.last_rx_mono is not None and time.monotonic() - self.last_rx_mono < 1.0
        if peer is None or not fresh:
            return
        offset, rtt = self.sync.best()
        st = {"v": 1, "kind": "status", "seq": self.n_reply, "t1": t1, "t2": t2,
              "link_hz": round(self.rate_hz, 1),
              "offset_ms": round(offset / 1e6, 3) if offset is not None else -1.0,   # -1: not synchronised yet
              "rtt_ms": round(rtt / 1e6, 2) if rtt is not None else -1.0,           # (numbers only: Unity JsonUtility)
              "auth_state": "", "auth_mode": "", "auth_reason": "", "heartbeat_ok": False,
              "prompt_level": -1, "prompt_title": "", "prompt_text": "", "rula_score": 0, "rula_valid": False,
              "tracking": [{"id": self.assigned.get(k, k), "ok": bool(v)} for k, v in sorted(valid.items())]
              + (self.extra_tracking if time.monotonic() - self.extra_t < 3.0 else []),
              "note": self.note_text()}
        a = self.auth
        if a is not None:
            st["auth_state"] = AUTH_STATES[a.state] if a.state < len(AUTH_STATES) else str(a.state)
            st["auth_mode"] = AUTH_MODES[a.process_mode] if a.process_mode < len(AUTH_MODES) else str(a.process_mode)
            st["auth_reason"], st["heartbeat_ok"] = a.reason, bool(a.heartbeat_ok)
        pr = self.prompt
        if pr is not None:
            st.update(prompt_level=int(pr.level), prompt_title=pr.title, prompt_text=pr.text)
        r = self.rula
        if r is not None:
            st.update(rula_score=int(r.score), rula_valid=bool(r.valid))
        st["t3"] = time.time_ns()
        try:
            self.sock.sendto(json.dumps(st, separators=(",", ":")).encode("utf-8"), peer)
            self.n_reply += 1
        except OSError as e:
            self.get_logger().warning(f"status to {peer[0]} failed: {e}", throttle_duration_sec=5.0)

    def report(self):
        with self.lock:
            self.rate_hz, self.window_rx = float(self.window_rx), 0
            peer = self.peer
            valid = dict(self.valid)
        offset, rtt = self.sync.best()
        self.half_rtt = rtt / 2e9 if rtt is not None else 0.0
        info = {"headset": peer[0] if peer else None, "rx_hz": self.rate_hz, "rx_total": self.n_rx,
                "seq_gaps": self.n_gap, "bad": self.n_bad, "replies": self.n_reply,
                "offset_ms": round(offset / 1e6, 3) if offset is not None else None,
                "rtt_ms": round(rtt / 1e6, 2) if rtt is not None else None,
                "recenter": self.recenter, "valid": valid, "twin_sent": self.n_twin, "scene_sent": self.n_scene}
        self.pub_status.publish(String(data=json.dumps(info)))

    def destroy_node(self):
        self.running = False
        try:
            self.sock.close()
        except OSError:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = UdpGateway()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
