"""ROS2 I/O for the hardware BODex pick-and-place runner.

One rclpy node that:

* subscribes ``/fp_pose`` (``geometry_msgs/PoseStamped``, world frame),
  buffering samples and exposing a 7-sample-median latch with a staleness
  guard (mirrors ``dexgrasp_pap.py``);
* subscribes the unified ``/joint_states`` (44 joints) and exposes
  name-indexed accessors for the active arm (6), active hand (16) and the
  passive arm (6);
* sends the 6 arm joints as a single
  ``control_msgs/action/FollowJointTrajectory`` goal to
  ``/{arm}_scaled_joint_trajectory_controller/follow_joint_trajectory``;
* streams the 16 hand joints as ``sensor_msgs/JointState`` on
  ``/joint_states_{right,left}_hand`` (the existing
  ``hand_service_commander_{right,left}`` matches **by joint name** and
  drives the Allegro service — the bridge_om.py permutation is therefore
  irrelevant here);
* optionally looks up ``world <- {arm}_base_link`` for the FK gate.

The executor spins in a daemon thread so subscription callbacks keep
flowing while the runner orchestrates the episode synchronously.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor

from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration

try:
    from tf2_ros import Buffer, TransformListener
    _HAVE_TF2 = True
except Exception:  # pragma: no cover - tf2 optional (only needed for FK gate)
    _HAVE_TF2 = False

try:
    import cv2
    _HAVE_RECORDING = True
except Exception:  # pragma: no cover - recording optional
    _HAVE_RECORDING = False

from bodex_hw.trajectory import release_lift_target


def _imgmsg_to_bgr(msg) -> "np.ndarray | None":
    """Pure-numpy decode of ``sensor_msgs/Image`` to a contiguous BGR
    uint8 array. We deliberately do NOT use ``cv_bridge``: in ROS humble
    cv_bridge is compiled against numpy 1.x, so in a numpy-2.x runtime
    (e.g. running the runner from the ``bodex`` conda env) the first
    ``imgmsg_to_cv2`` call raises ``AttributeError: _ARRAY_API not
    found`` and segfaults. Plain ``np.frombuffer`` is ABI-independent.
    Returns ``None`` for encodings we don't handle here."""
    enc = msg.encoding.lower()
    h, w = int(msg.height), int(msg.width)
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if enc == "bgr8":
        return buf.reshape(h, w, 3).copy()
    if enc == "rgb8":
        return np.ascontiguousarray(buf.reshape(h, w, 3)[:, :, ::-1])
    if enc == "bgra8":
        return np.ascontiguousarray(buf.reshape(h, w, 4)[:, :, :3])
    if enc == "rgba8":
        return np.ascontiguousarray(buf.reshape(h, w, 4)[:, :, [2, 1, 0]])
    if enc == "mono8":
        gray = buf.reshape(h, w)
        return np.stack([gray, gray, gray], axis=-1)
    return None

_ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


class HwIO(Node):
    def __init__(
        self,
        active_arm: str = "ra",
        object_pose_topic: str = "/fp_pose",
        pose_max_age_s: float = 3.0,
        arm_iface: str = "forward_position",
        arm_cmd_topic: str | None = None,
    ):
        super().__init__("bodex_pap_hw")
        if active_arm not in ("ra", "la"):
            raise ValueError(f"active_arm must be ra/la, got {active_arm!r}")
        self.active_arm = active_arm
        self.passive_arm = "la" if active_arm == "ra" else "ra"
        self.pose_max_age_s = float(pose_max_age_s)
        if arm_iface not in ("forward_position", "trajectory"):
            raise ValueError(
                f"arm_iface must be forward_position/trajectory, "
                f"got {arm_iface!r}"
            )
        self.arm_iface = arm_iface

        self._lock = threading.Lock()
        self._obj_buf: deque = deque(maxlen=30)   # (t_mono, pos3, quat_xyzw4)
        self._last_obj_mono = 0.0
        self._js_pos: dict[str, float] = {}

        # Optional video recording (lazily wired by start_recording()).
        self._recorders: dict = {}
        self._record_active = False

        self._object_pose_topic = object_pose_topic
        self._obj_sub = self.create_subscription(
            PoseStamped, object_pose_topic, self._obj_cb, 10
        )
        self.create_subscription(
            JointState, "/joint_states", self._js_cb, 50
        )

        ap = self.active_arm + "_"
        self._hand_names = [f"{ap}joint_{i}_0" for i in range(16)]
        self._hand_pub = self.create_publisher(
            JointState,
            f"/joint_states_{'right' if active_arm == 'ra' else 'left'}_hand",
            10,
        )
        self._arm_joint_names = [ap + n for n in _ARM_JOINTS]
        self._passive_arm_joint_names = [
            (self.passive_arm + "_") + n for n in _ARM_JOINTS
        ]
        # Arm command path. forward_position = stream Float64MultiArray to
        # JointGroupPositionController (mirrors the sim's per-tick
        # set_joint_position_target, no action server). trajectory = one
        # FollowJointTrajectory goal to the scaled JTC.
        if self.arm_iface == "forward_position":
            self._arm_cmd_topic = (
                arm_cmd_topic
                or f"/{active_arm}_forward_position_controller/commands"
            )
            self._arm_cmd_pub = self.create_publisher(
                Float64MultiArray, self._arm_cmd_topic, 10
            )
            self._arm_ac = None
        else:
            self._arm_cmd_pub = None
            self._arm_ac = ActionClient(
                self,
                FollowJointTrajectory,
                f"/{active_arm}_scaled_joint_trajectory_controller/"
                f"follow_joint_trajectory",
            )

        if _HAVE_TF2:
            self._tf_buf = Buffer()
            self._tf_listener = TransformListener(self._tf_buf, self)
        else:
            self._tf_buf = None

        self._exec = SingleThreadedExecutor()
        self._exec.add_node(self)
        self._spin_thread = threading.Thread(
            target=self._exec.spin, daemon=True
        )

    # ---- lifecycle -------------------------------------------------
    def start(self):
        self._spin_thread.start()

    def shutdown(self):
        # Flush MP4 writers before tearing down the executor — a
        # released writer is what guarantees the file is playable.
        try:
            self.stop_recording()
        except Exception:
            pass
        # Stop the executor, then join the spin thread BEFORE the node /
        # rclpy context is torn down. Skipping the join is what produced
        # the "terminate called without an active exception" core dump on
        # early-return paths (the daemon thread was still in C++ spin()).
        try:
            self._exec.shutdown()
        except Exception:
            pass
        try:
            if self._spin_thread.is_alive():
                self._spin_thread.join(timeout=3.0)
        except Exception:
            pass
        try:
            self.destroy_node()
        except Exception:
            pass

    # ---- callbacks -------------------------------------------------
    def _obj_cb(self, msg: PoseStamped):
        p = msg.pose.position
        o = msg.pose.orientation
        with self._lock:
            self._obj_buf.append((
                time.monotonic(),
                np.array([p.x, p.y, p.z], dtype=float),
                np.array([o.x, o.y, o.z, o.w], dtype=float),
            ))
            self._last_obj_mono = time.monotonic()

    def _js_cb(self, msg: JointState):
        with self._lock:
            for n, q in zip(msg.name, msg.position):
                self._js_pos[n] = float(q)

    # ---- object pose ----------------------------------------------
    def set_object_topic(self, topic: str) -> None:
        """Re-point the object-pose subscription at a new topic (e.g. switching
        objects in the benchmark: ``/tiger_pose`` -> ``/snorlax_pose``). Destroys
        the old subscription, clears the sample buffer, and subscribes fresh, so
        one HwIO node can serve every object without a full restart."""
        if topic == self._object_pose_topic:
            return
        try:
            self.destroy_subscription(self._obj_sub)
        except Exception:
            pass
        with self._lock:
            self._obj_buf.clear()
            self._last_obj_mono = 0.0
        self._object_pose_topic = topic
        self._obj_sub = self.create_subscription(
            PoseStamped, topic, self._obj_cb, 10
        )

    def wait_for_object_pose(self, min_samples: int = 7,
                             timeout_s: float = 30.0) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            with self._lock:
                n = len(self._obj_buf)
            if n >= min_samples:
                return True
            time.sleep(0.05)
        return False

    def latch_object_pose_bodex(self, k: int = 7) -> np.ndarray:
        """Latched object pose for BODex: ``[x,y,z,qw,qx,qy,qz]``.

        Median of the last ``k`` translations + most-recent orientation
        (the dexgrasp_pap convention). Raises if no samples or the latest
        sample is older than ``pose_max_age_s``.
        """
        with self._lock:
            if not self._obj_buf:
                raise RuntimeError("No /fp_pose samples received.")
            age = time.monotonic() - self._last_obj_mono
            if age > self.pose_max_age_s:
                raise RuntimeError(
                    f"/fp_pose stale: {age:.1f}s > {self.pose_max_age_s}s"
                )
            recent = list(self._obj_buf)[-min(k, len(self._obj_buf)):]
        trans = np.median(np.stack([r[1] for r in recent], 0), axis=0)
        qx, qy, qz, qw = recent[-1][2]  # latest orientation, xyzw
        # BODex wants w-first: [x,y,z, qw,qx,qy,qz]
        return np.array(
            [trans[0], trans[1], trans[2], qw, qx, qy, qz], dtype=float
        )

    # ---- joint state ----------------------------------------------
    def _q_by_names(self, names: list[str]) -> np.ndarray | None:
        with self._lock:
            if not all(n in self._js_pos for n in names):
                return None
            return np.array([self._js_pos[n] for n in names], dtype=float)

    def wait_for_joint_states(self, timeout_s: float = 15.0) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            if self._q_by_names(self._arm_joint_names) is not None:
                return True
            time.sleep(0.05)
        return False

    def active_arm_q(self) -> np.ndarray:
        q = self._q_by_names(self._arm_joint_names)
        if q is None:
            raise RuntimeError("active arm joints not in /joint_states yet")
        return q

    def active_hand_q(self) -> np.ndarray:
        q = self._q_by_names(self._hand_names)
        # Hand state may be absent in arms-only/mock setups — default open.
        return q if q is not None else np.zeros(16, dtype=float)

    def passive_arm_q(self) -> np.ndarray | None:
        return self._q_by_names(self._passive_arm_joint_names)

    # ---- TF (optional FK gate) ------------------------------------
    def world_base_tf(self) -> np.ndarray | None:
        if self._tf_buf is None:
            return None
        try:
            tf = self._tf_buf.lookup_transform(
                "world", f"{self.active_arm}_base_link",
                rclpy.time.Time()
            )
        except Exception:
            return None
        t = tf.transform.translation
        r = tf.transform.rotation
        x, y, z, w = r.x, r.y, r.z, r.w
        T = np.eye(4)
        T[:3, :3] = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    # ---- video recording ------------------------------------------
    def start_recording(self, topics, output_dir, fps=30.0,
                        fourcc="mp4v"):
        """Subscribe to image topics and write one MP4 per topic.

        Lazily initializes ``cv2.VideoWriter`` on the first frame of each
        topic (so the writer matches the actual image resolution). Each
        subscription writes synchronously in its callback; the executor
        is single-threaded so writers are not contended across callbacks
        of the same topic. Returns True on setup success.

        Files are named ``pick_place_<YYYYmmdd_HHMMSS>_<topic-sanitized>.mp4``
        under ``output_dir``.
        """
        if not _HAVE_RECORDING:
            self.get_logger().error(
                "cv2 not importable — recording disabled. "
                "Install python3-opencv."
            )
            return False
        os.makedirs(output_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        cc = cv2.VideoWriter_fourcc(*fourcc)
        self._record_active = True
        for topic in topics:
            name_safe = topic.strip("/").replace("/", "_")
            filepath = os.path.join(
                output_dir, f"pick_place_{ts}_{name_safe}.mp4"
            )
            rec = {
                "topic": topic,
                "filepath": filepath,
                "fps": float(fps),
                "fourcc": cc,
                "writer": None,           # lazy init on first frame
                "frame_count": 0,
                "lock": threading.Lock(),
            }
            rec["sub"] = self.create_subscription(
                Image, topic, lambda msg, r=rec: self._image_cb(msg, r), 5
            )
            self._recorders[topic] = rec
            self.get_logger().info(
                f"[record] {topic} -> {filepath} @ {fps:.0f} fps"
            )
        return True

    def _image_cb(self, msg, rec):
        if not self._record_active:
            return
        frame = _imgmsg_to_bgr(msg)
        if frame is None:
            self.get_logger().warn(
                f"[record] unsupported encoding {msg.encoding!r} on "
                f"{rec['topic']}; supported: bgr8, rgb8, bgra8, rgba8, mono8"
            )
            return
        with rec["lock"]:
            if not self._record_active:
                return
            if rec["writer"] is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    rec["filepath"], rec["fourcc"], rec["fps"], (w, h)
                )
                if not writer.isOpened():
                    self.get_logger().error(
                        f"[record] VideoWriter failed to open "
                        f"{rec['filepath']} (codec? path?)"
                    )
                    return
                rec["writer"] = writer
                self.get_logger().info(
                    f"[record] {rec['topic']}: first frame {w}x{h}, "
                    f"writing"
                )
            rec["writer"].write(frame)
            rec["frame_count"] += 1

    def stop_recording(self):
        """Stop all in-flight recordings, release MP4 writers."""
        if not self._recorders:
            return
        self._record_active = False
        # Give in-flight callbacks a moment to bail before releasing.
        time.sleep(0.1)
        for topic, rec in list(self._recorders.items()):
            with rec["lock"]:
                if rec["writer"] is not None:
                    rec["writer"].release()
                    rec["writer"] = None
                    self.get_logger().info(
                        f"[record] {topic}: {rec['frame_count']} frames "
                        f"-> {rec['filepath']}"
                    )
                else:
                    self.get_logger().warn(
                        f"[record] {topic}: no frames captured"
                    )
            try:
                self.destroy_subscription(rec["sub"])
            except Exception:
                pass
        self._recorders.clear()

    # ---- replay ----------------------------------------------------
    def _time_vector(self, T, dt, grasp_idx, grasp_settle_s):
        """Shared wall-clock schedule. A single ``grasp_settle_s`` dwell at
        ``grasp_idx`` pushes every later waypoint back in time, so the
        Allegro finishes closing before the arm lifts."""
        return np.array(
            [i * dt + (grasp_settle_s if i >= grasp_idx else 0.0)
             for i in range(T)],
            dtype=float,
        )

    def replay(
        self,
        full: np.ndarray,
        dt: float,
        grasp_idx: int,
        grasp_settle_s: float,
        mock_arm: bool = False,
        watchdog=None,
        stream_hz: float = 125.0,
        grasp_pos_tol: float = 0.02,
        grasp_reach_timeout: float = 3.0,
        approach_dt: float | None = None,
        return_dt: float | None = None,
        hand_open_dwell_s: float = 0.0,
        grasp_close_ramp_s: float = 0.0,
        stagger_close_s: float = 0.0,
        release_idx: int = -1,
        release_open_s: float = 0.0,
        release_lift_frac: float = 0.30,
        release_lift_wps: int = 14,
        release_reorient_home: bool = False,
        descend_n: int = 20,
        release_settle_frac: float = 0.5,
        release_open_dwell_s: float = 0.0,
        stop_evt=None,
    ) -> bool:
        """Replay the [T,22] trajectory. Arm + hand follow ONE schedule.

        forward_position: BODex waypoints are knots; a high-rate loop
        (``stream_hz``, default 125 Hz) publishes a *linearly interpolated*
        setpoint every tick — JointGroupPositionController does no
        interpolation itself, so smoothness comes entirely from sending
        finely-spaced commands (exactly how home_arms.py achieves smooth
        motion). Sparse per-waypoint publishing makes the UR
        jump→settle→jump (the shaking the user saw). The first knot is the
        *measured* current arm pose so there is zero startup jump.

        Closed-loop grasp barrier: the stream is split at ``grasp_idx``.
        Phase 1 streams the approach to the grasp pose; phase 2a then holds
        the grasp ARM pose with the hand still OPEN and polls
        ``active_arm_q()`` until the measured arm is within
        ``grasp_pos_tol`` of the commanded grasp pose (or
        ``grasp_reach_timeout`` elapses); phase 2b ramps the hand closed over
        ``grasp_close_ramp_s`` (so it doesn't slam the object) then holds the
        firm close for the rest of ``grasp_settle_s`` while the slow Allegro
        wraps; phase 3 streams lift → place → home. This closes the open-loop gap
        that let the hand close before the arm had actually arrived (the
        sim guarantees per-waypoint convergence; streaming does not).

        No software position-watchdog here: streamed setpoints make
        measured legitimately lag commanded, so an instantaneous-error
        check would false-trip. Pre-send clamp/velocity gates + the UR
        controller's protective stop are the safety envelope.

        trajectory: one FollowJointTrajectory goal (UR scaled JTC) with
        the hand streamed on the same schedule; lag-tolerant watchdog kept.
        """
        if self.arm_iface == "forward_position":
            return self._replay_forward_position(
                full, dt, grasp_idx, grasp_settle_s, mock_arm, stream_hz,
                grasp_pos_tol, grasp_reach_timeout,
                approach_dt=approach_dt, return_dt=return_dt,
                hand_open_dwell_s=hand_open_dwell_s,
                grasp_close_ramp_s=grasp_close_ramp_s,
                stagger_close_s=stagger_close_s,
                release_idx=release_idx, release_open_s=release_open_s,
                release_lift_frac=release_lift_frac,
                release_lift_wps=release_lift_wps,
                release_reorient_home=release_reorient_home,
                descend_n=descend_n,
                release_settle_frac=release_settle_frac,
                release_open_dwell_s=release_open_dwell_s,
                stop_evt=stop_evt,
            )
        T = full.shape[0]
        t_of_wp = self._time_vector(T, dt, grasp_idx, grasp_settle_s)
        return self._replay_trajectory(
            full, t_of_wp, mock_arm, watchdog, dt
        )

    # Universal joint-speed governor cap, as a fraction of each joint's UR
    # speed limit ([pi,pi,pi,2pi,2pi,2pi] rad/s on the UR3). Every arm motion
    # in this process flows through _stream_knots, but only the MAIN planned
    # trajectory is velocity-gated pre-flight — the fixed-duration lerp
    # segments (recovery return-home, release re-level, hover lift) could
    # command a joint past its limit and trip a pendant protective stop.
    # 0.85 leaves normal motion (peaks typically 40-70%) untouched and
    # stretches only genuinely hot segments, so a pendant joint-speed
    # violation from this process is impossible by construction.
    STREAM_VEL_CAP = 0.85

    def _govern_knot_times(self, knot_t, knot_q):
        """Stretch segment times so no arm joint is commanded faster than
        STREAM_VEL_CAP x its UR limit. Returns the (possibly stretched) knot
        time vector; logs when the total stretch is noticeable."""
        from bodex_hw.safety import UR_JOINT_VEL_MAX
        kt = np.asarray(knot_t, dtype=float)
        kq = np.asarray(knot_q, dtype=float)
        if len(kt) < 2 or kq.shape[1] < 6:
            return kt
        lim = UR_JOINT_VEL_MAX * float(self.STREAM_VEL_CAP)   # [6] rad/s
        out = kt.astype(float).copy()
        stretched = 0.0
        for j in range(len(kt) - 1):
            seg = max(float(kt[j + 1] - kt[j]), 1e-9)
            need = float(np.max(np.abs(kq[j + 1, :6] - kq[j, :6]) / lim))
            dur = max(seg, need)
            out[j + 1] = out[j] + dur
            stretched += dur - seg
        if stretched > 0.05:
            try:
                self.get_logger().warn(
                    f"[vel-governor] stretched motion by +{stretched:.2f}s so "
                    f"every joint stays <= {int(self.STREAM_VEL_CAP * 100)}% "
                    f"of its UR speed limit")
            except Exception:
                pass
        return out

    def _stream_knots(self, knot_t, knot_q, period, mock_arm, stop_evt=None):
        """High-rate piecewise-linear streamer over (knot_t [K], knot_q
        [K,22]). Publishes a linearly interpolated setpoint every ``period``
        seconds — JointGroupPositionController does no interpolation itself,
        so smoothness comes from finely-spaced commands (home_arms.py
        pattern). All segment times pass through the joint-speed governor."""
        knot_t = self._govern_knot_times(knot_t, knot_q)
        total = float(knot_t[-1])
        t0 = time.monotonic()
        next_tick = t0
        while True:
            if stop_evt is not None and stop_evt.is_set():
                return
            elapsed = time.monotonic() - t0
            if elapsed >= total:
                break
            j = int(np.searchsorted(knot_t, elapsed, side="right") - 1)
            j = min(max(j, 0), len(knot_t) - 2)
            seg = max(knot_t[j + 1] - knot_t[j], 1e-9)
            a = min(max((elapsed - knot_t[j]) / seg, 0.0), 1.0)
            q = knot_q[j] * (1.0 - a) + knot_q[j + 1] * a
            self._publish_setpoint(q, mock_arm)
            next_tick += period
            time.sleep(max(0.0, next_tick - time.monotonic()))

    def _publish_setpoint(self, q22, mock_arm):
        if not mock_arm:
            cmd = Float64MultiArray()
            cmd.data = [float(x) for x in q22[:6]]
            self._arm_cmd_pub.publish(cmd)
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = self._hand_names
        js.position = [float(x) for x in q22[6:22]]
        self._hand_pub.publish(js)

    _CURL_IDX = (1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15)

    def _log_hand_state(self, tag, open_cmd, closed_cmd):
        """Log measured-vs-commanded hand state at a phase seam. Helps
        tune ``--grasp_close_extra`` against the Allegro's gravity-induced
        droop (PID with ki=0 cannot fully extend fingers against their
        own weight; the gap between commanded-open and measured-open is
        the per-joint droop you have to absorb)."""
        try:
            meas = self.active_hand_q().astype(float)
        except Exception:
            return
        curl = list(self._CURL_IDX)
        droop = float(np.max(np.abs(meas[curl] - open_cmd[curl])))
        close_err = float(np.max(np.abs(meas[curl] - closed_cmd[curl])))
        self.get_logger().info(
            f"hand@{tag}: curl_droop_vs_open={droop:.3f} rad "
            f"err_vs_closed={close_err:.3f} rad"
        )

    def _replay_forward_position(self, full, dt, grasp_idx, grasp_settle_s,
                                 mock_arm, stream_hz, grasp_pos_tol,
                                 grasp_reach_timeout,
                                 approach_dt=None, return_dt=None,
                                 hand_open_dwell_s=0.0, grasp_close_ramp_s=0.0,
                                 stagger_close_s=0.0,
                                 release_idx=-1, release_open_s=0.0,
                                 release_lift_frac=0.30, release_lift_wps=14,
                                 release_reorient_home=False,
                                 descend_n=20, release_settle_frac=0.5,
                                 release_open_dwell_s=0.0,
                                 stop_evt=None):
        # stop_evt (threading.Event|None): when set mid-motion, every streamer
        # and polling loop below bails out and replay() returns False within
        # ~one stream tick, leaving the arm at its last commanded setpoint for
        # the caller (e.g. the benchmark harness) to recover from.
        def _aborted():
            return stop_evt is not None and stop_evt.is_set()

        T = full.shape[0]
        period = 1.0 / max(stream_hz, 1.0)
        gi = int(min(max(grasp_idx, 0), T - 1))
        # Per-phase dt: None / non-positive → fall back to the global dt.
        a_dt = float(approach_dt) if (approach_dt and approach_dt > 0) else float(dt)
        r_dt = float(return_dt)   if (return_dt   and return_dt   > 0) else float(dt)
        try:
            cur_arm = self.active_arm_q()[:6].astype(float)
        except Exception:
            cur_arm = full[0, :6].astype(float)

        # Setpoints reused across phases 2a/2a.5/2b/3.
        grasp_arm = full[gi, :6].astype(float)
        approach_hand = full[max(gi - 1, 0), 6:22].astype(float)  # still open
        closed_hand = full[gi, 6:22].astype(float)                # firm close
        open_sp = np.concatenate([grasp_arm, approach_hand])
        closed_sp = np.concatenate([grasp_arm, closed_hand])

        # --- Phase 1: measured start -> grasp pose (approach) ----------
        # Knot 0 = measured arm + planned wp0 hand → no startup jump.
        # Uses approach_dt (typically faster than dt) — the cube pose is
        # known, no need to crawl down to it.
        start = np.concatenate([cur_arm, full[0, 6:22]]).astype(float)
        pre_q = [start] + [full[i].astype(float) for i in range(gi + 1)]
        pre_t = [0.0] + [(i + 1) * a_dt for i in range(gi + 1)]
        # CRITICAL: replace the final knot's HAND columns with the still-
        # OPEN approach hand. trajectory.py (intentionally for sim) puts
        # the firm close at the grasp-contact waypoint full[gi].hand;
        # without this override the streamer linearly interpolates the
        # hand from open (full[gi-1]) to closed (full[gi]) during the
        # LAST dt of the approach — i.e. the fingers begin wrapping
        # while the arm is still in motion, BEFORE it has reached the
        # object (the original "early_grasp.mp4" failure). Phase 2b
        # commands the close ONLY after the barrier confirms arm arrival.
        pre_q[-1] = open_sp.copy()
        self.get_logger().info(
            f"forward_position pipeline: "
            f"phase1 {len(pre_q)} knots @ approach_dt={a_dt:.3f}s "
            f"({pre_t[-1]:.1f}s, hand OPEN throughout) -> "
            f"arm barrier (≤{grasp_reach_timeout:.1f}s) -> "
            f"hand-open dwell {hand_open_dwell_s:.1f}s -> "
            f"close+settle {grasp_settle_s:.1f}s -> "
            f"phase3 {T - gi - 1} knots @ return_dt={r_dt:.3f}s "
            f"({(T - gi - 1) * r_dt:.1f}s) "
            f"@ stream {stream_hz:.0f} Hz"
        )
        self._stream_knots(np.asarray(pre_t), np.stack(pre_q), period, mock_arm,
                           stop_evt=stop_evt)
        if _aborted():
            return False
        self._log_hand_state("end-phase1", open_sp[6:22], closed_sp[6:22])

        # --- Phase 2a: reach barrier (arm to grasp pose, hand OPEN) -----
        t0 = time.monotonic()
        reached = False
        while time.monotonic() - t0 < grasp_reach_timeout:
            if _aborted():
                return False
            self._publish_setpoint(open_sp, mock_arm)
            try:
                if float(np.max(np.abs(self.active_arm_q()[:6] - grasp_arm))) \
                        <= grasp_pos_tol:
                    reached = True
                    break
            except Exception:
                pass
            time.sleep(period)
        try:
            err = float(np.max(np.abs(self.active_arm_q()[:6] - grasp_arm)))
        except Exception:
            err = float("nan")
        self.get_logger().info(
            f"grasp barrier: {'REACHED' if reached else 'TIMEOUT'} "
            f"(max arm joint err={err:.4f} rad, tol {grasp_pos_tol})"
        )

        # --- Phase 2a.5: hand-open dwell at grasp pose -----------------
        # Hold (grasp_arm, open_hand) for hand_open_dwell_s before the
        # close. The Allegro's PID with ki=0 cannot fully overcome the
        # gravity-induced finger curl, so the "open" pose physically
        # achievable is somewhere between fully-extended and slightly-
        # drooped. This dwell parks the fingers at that equilibrium
        # consistently across runs (otherwise the close starts from a
        # variable, half-open state — the variability the user saw).
        # The arm sits at grasp_arm the whole time, so it costs nothing
        # but elapsed time.
        if hand_open_dwell_s > 0:
            t_od = time.monotonic()
            while time.monotonic() - t_od < hand_open_dwell_s:
                if _aborted():
                    return False
                self._publish_setpoint(open_sp, mock_arm)
                time.sleep(period)
            self._log_hand_state(
                f"post-open-dwell({hand_open_dwell_s:.1f}s)",
                open_sp[6:22], closed_sp[6:22])

        # --- Phase 2b: close + settle (hold grasp pose, hand closing) ----
        # Progressive close: RAMP the hand setpoint open -> firm-closed over
        # grasp_close_ramp_s instead of stepping to it, so the fingers don't
        # slam shut and shove the object. Then hold the FULL close for the
        # remainder of grasp_settle_s — the final setpoint is the firm close,
        # so the grip FORCE is unchanged. ramp_s is capped at the settle time.
        t_close = time.monotonic()
        ramp_s = min(max(grasp_close_ramp_s, 0.0), max(grasp_settle_s, 0.0))
        # Staggered close (stagger_close_s, signed):
        #   > 0  THUMB-FIRST: the thumb (Allegro joints 12-15) ramps closed
        #        immediately and PLANTS as a backstop; the four fingers (0-11)
        #        start |stagger| later, so their curling rake presses the
        #        object AGAINST the waiting thumb instead of dragging it
        #        backward out of the pocket while the thumb is still en route
        #        (the observed hardware failure: fingers pull the object back,
        #        thumb arrives late / gets stopped by the table).
        #   < 0  thumb delayed instead (fingers form the wall first) — kept
        #        for A/B experiments.
        #   = 0  the original synchronized close, exactly.
        # The lagging group is capped so it still completes its full ramp
        # within grasp_settle_s; final setpoint = same firm close either way,
        # so grip force is unchanged.
        stag = float(stagger_close_s)
        max_lag = max(max(grasp_settle_s, 0.0) - ramp_s, 0.0)
        stag = float(np.clip(stag, -max_lag, max_lag)) if ramp_s > 0.0 else 0.0
        d_fingers = max(stag, 0.0)    # thumb-first: fingers start later
        d_thumb = max(-stag, 0.0)     # thumb-last: thumb starts later
        lag = max(d_fingers, d_thumb)
        while True:
            if _aborted():
                return False
            el = time.monotonic() - t_close
            if el >= max(grasp_settle_s, 0.0):
                break
            if ramp_s > 0.0 and el < ramp_s + lag:
                a_f = min(max(el - d_fingers, 0.0) / ramp_s, 1.0)  # fingers
                a_t = min(max(el - d_thumb, 0.0) / ramp_s, 1.0)    # thumb
                hand_sp = np.empty_like(closed_hand)
                hand_sp[:12] = (approach_hand[:12] * (1.0 - a_f)
                                + closed_hand[:12] * a_f)
                hand_sp[12:] = (approach_hand[12:] * (1.0 - a_t)
                                + closed_hand[12:] * a_t)
                self._publish_setpoint(
                    np.concatenate([grasp_arm, hand_sp]), mock_arm)
            else:
                self._publish_setpoint(closed_sp, mock_arm)
            time.sleep(period)
        self._log_hand_state(
            f"post-close+settle({grasp_settle_s:.1f}s, "
            f"stagger={stag:+.2f}s"
            f"{'(thumb-first)' if stag > 0 else ('(thumb-last)' if stag < 0 else '(sync)')})",
            open_sp[6:22], closed_sp[6:22])

        # --- Phase 3: grasp -> place (hold) -> RELEASE -> home ---------
        # Uses return_dt (typically faster than dt) — once the grasp is
        # secured, the transit + place + return don't need crawl speed.
        # With a release barrier: stream to the PLACE pose with the hand still
        # closed (carrying the object), then HOLD the arm and open the hand
        # over release_open_s, THEN return home with the hand forced open — so
        # the object isn't dragged by a hand that opens while the arm moves.
        ri = int(release_idx)
        if release_open_s > 0.0 and gi < ri < T - 1:
            hand_open = full[-1, 6:22].astype(float)   # home open pose (thumb 0.36)
            place_arm = full[ri, :6].astype(float)
            place_hand = full[ri, 6:22].astype(float)
            home_pose = full[-1, :6].astype(float)
            n_home = max(12, T - ri - 1)
            ros = max(release_open_s, 1e-3)

            def _lerp_stream(q0_arm, q1_arm, hand16, n):
                """Stream a straight joint lerp q0_arm -> q1_arm over n knots at
                return_dt, holding the 16 hand joints fixed."""
                qs = np.stack([
                    np.concatenate([q0_arm * (1.0 - k / n) + q1_arm * (k / n),
                                    hand16])
                    for k in range(n + 1)])
                ts = np.asarray([k * r_dt for k in range(n + 1)])
                self._stream_knots(ts, qs, period, mock_arm, stop_evt=stop_evt)

            # We deliberately do NOT replay BODex's release-ramp arm waypoints
            # (full[ri+1:...]): on hardware those keep descending to set the
            # object down — for TOP grasps that drives the down-pointing fingers
            # into the table, and for SIDE grasps the deep descend (palm at the
            # object's mid-height to stand it up) puts a forearm/wrist link BELOW
            # the table top so the arm's underside hits the table first.
            #
            # UNIFIED release (top AND side): carry up-and-over to the HOVER above
            # the goal — the END of the collision-checked transit, BEFORE the
            # descend (hover = release_idx - descend_n) — so the arm stays HIGH and
            # no link dips below the table. Then settle straight DOWN toward the
            # goal KEEPING the grasp orientation (no reorientation/spin), and open;
            # the object is released from the settled height (drops/lays the short
            # remaining gap). `release_settle_frac` trades drop height vs clearance
            # (0 = open at the hover; higher lowers it but a SIDE arm starts to dip
            # and TOP fingers approach the table — keep well under ~0.5).
            hov = max(gi + 1, min(ri - int(descend_n), ri - 1))
            hover_arm = full[hov, :6].astype(float)
            # Hold the FULLY-SQUEEZED hover hand (full[ri] is the first opening
            # frame ~0.4 curl and would slip the object early).
            place_hand = full[hov, 6:22].astype(float)
            # 3a: grasp -> hover, hand stays CLOSED (collision-checked transit).
            a_q = [closed_sp] + [full[i].astype(float)
                                 for i in range(gi + 1, hov + 1)]
            a_t = [0.0] + [(k + 1) * r_dt for k in range(hov - gi)]
            self._stream_knots(np.asarray(a_t), np.stack(a_q), period, mock_arm,
                               stop_evt=stop_evt)
            if _aborted():
                return False

            # 3b: settle straight DOWN toward the goal, KEEPING the grasp
            # orientation (interpolate the arm config hover->descend).
            sf = float(min(max(release_settle_frac, 0.0), 1.0))
            descend_arm = full[ri, :6].astype(float)
            settle_arm = hover_arm * (1.0 - sf) + descend_arm * sf
            if sf > 0.0:
                set_d = float(np.max(np.abs(settle_arm - hover_arm)))
                n_set = max(6, int(np.ceil(set_d / 0.04)) + 1)
                _lerp_stream(hover_arm, settle_arm, place_hand, n_set)  # down, no spin
            open_arm = settle_arm
            self.get_logger().info(
                f"release ({'top' if release_reorient_home else 'side'}): "
                f"carry->hover@wp{hov} (skip {ri - hov} descend wps), settle "
                f"{sf:.2f} down (keep orientation), open over {ros:.2f}s, "
                f"then home ({n_home})")

            # 3d: RELEASE BARRIER — hold open_arm, ramp the hand open over ros.
            t_rel = time.monotonic()
            while time.monotonic() - t_rel < ros:
                if _aborted():
                    return False
                a = (time.monotonic() - t_rel) / ros
                h = place_hand * (1.0 - a) + hand_open * a
                self._publish_setpoint(np.concatenate([open_arm, h]), mock_arm)
                time.sleep(period)
            self._publish_setpoint(np.concatenate([open_arm, hand_open]), mock_arm)

            # 3d.5: POST-RELEASE OPEN DWELL — hold the arm + fully-open hand so the
            # Allegro reaches its open equilibrium BEFORE the arm leaves. The PID
            # (ki=0) droops under gravity, so without this the fingers are still
            # opening as the arm moves and can dip back toward the object/table.
            rod = max(float(release_open_dwell_s), 0.0)
            if rod > 0.0:
                t_rod = time.monotonic()
                while time.monotonic() - t_rod < rod:
                    if _aborted():
                        return False
                    self._publish_setpoint(
                        np.concatenate([open_arm, hand_open]), mock_arm)
                    time.sleep(period)
                self._log_hand_state("post-release-open-dwell",
                                     hand_open, place_hand)

            # 3e: LEAVE the object safely, THEN home. The 3b settle dropped the arm
            # toward the goal to place + open; for a SIDE grasp that low pose sits
            # beside the object with a forearm/wrist link near the table, so homing
            # straight from it scrapes the table / clips the just-placed object.
            # First lift straight back UP to the HOVER pose (retrace the settle in
            # reverse, hand now OPEN — a vertical rise, so the open fingers clear the
            # object without sweeping it), then home from the high hover. No-op lift
            # when settle_frac=0 (open_arm == hover_arm).
            lift_d = float(np.max(np.abs(np.asarray(hover_arm) - np.asarray(open_arm))))
            if lift_d > 1e-4:
                n_lift = max(6, int(np.ceil(lift_d / 0.04)) + 1)
                _lerp_stream(open_arm, hover_arm, hand_open, n_lift)   # lift up, open
                if _aborted():
                    return False
            # home from the high hover (clear of the table and the object).
            _lerp_stream(hover_arm, home_pose, hand_open, n_home)      # home, open
        elif gi < T - 1:
            # Fallback (barrier off / no valid release index): single block.
            post_q = [closed_sp] + [full[i].astype(float)
                                    for i in range(gi + 1, T)]
            post_t = [0.0] + [(k + 1) * r_dt for k in range(T - gi - 1)]
            self._stream_knots(np.asarray(post_t), np.stack(post_q),
                               period, mock_arm, stop_evt=stop_evt)
            if _aborted():
                return False

        # Settle: hold the final pose for a moment at full rate.
        t_end = time.monotonic()
        while time.monotonic() - t_end < 0.3:
            if _aborted():
                return False
            self._publish_setpoint(full[-1].astype(float), mock_arm)
            time.sleep(period)
        return True

    def _replay_trajectory(self, full, t_of_wp, mock_arm, watchdog, dt):
        T = full.shape[0]
        if not mock_arm:
            if not self._arm_ac.wait_for_server(timeout_sec=10.0):
                self.get_logger().error(
                    "arm trajectory action server not available "
                    f"(is {self.active_arm}_scaled_joint_trajectory_"
                    "controller active? try --arm_iface forward_position)"
                )
                return False
            goal = FollowJointTrajectory.Goal()
            jt = JointTrajectory()
            jt.joint_names = self._arm_joint_names
            for i in range(T):
                pt = JointTrajectoryPoint()
                pt.positions = [float(x) for x in full[i, :6]]
                sec = int(t_of_wp[i])
                nsec = int((t_of_wp[i] - sec) * 1e9)
                pt.time_from_start = Duration(sec=sec, nanosec=nsec)
                jt.points.append(pt)
            goal.trajectory = jt
            send_future = self._arm_ac.send_goal_async(goal)
            while not send_future.done():
                time.sleep(0.02)
            gh = send_future.result()
            if gh is None or not gh.accepted:
                self.get_logger().error("arm goal rejected")
                return False
            result_future = gh.get_result_async()
        else:
            result_future = None

        stop_evt = threading.Event()
        t0 = time.monotonic()

        def _stream_hand():
            for i in range(T):
                target = t0 + t_of_wp[i]
                while time.monotonic() < target:
                    if stop_evt.is_set():
                        return
                    time.sleep(0.002)
                js = JointState()
                js.header.stamp = self.get_clock().now().to_msg()
                js.name = self._hand_names
                js.position = [float(x) for x in full[i, 6:22]]
                self._hand_pub.publish(js)

        hand_thread = threading.Thread(target=_stream_hand, daemon=True)
        hand_thread.start()

        total_s = float(t_of_wp[-1]) + 0.5
        if mock_arm:
            i = 0
            while time.monotonic() - t0 < total_s:
                if watchdog is not None and i < T:
                    watchdog(i, full[i, :6], self.active_arm_q())
                i = min(T - 1, int((time.monotonic() - t0) / max(dt, 1e-3)))
                time.sleep(0.05)
            hand_thread.join(timeout=2.0)
            return True

        deadline = t0 + total_s + 10.0
        while not result_future.done():
            if time.monotonic() > deadline:
                self.get_logger().error("arm trajectory timed out")
                stop_evt.set()
                hand_thread.join(timeout=2.0)
                return False
            if watchdog is not None:
                i = min(T - 1, int((time.monotonic() - t0) / max(dt, 1e-3)))
                try:
                    watchdog(i, full[i, :6], self.active_arm_q())
                except SystemExit:
                    stop_evt.set()
                    hand_thread.join(timeout=2.0)
                    raise
            time.sleep(0.05)
        hand_thread.join(timeout=2.0)
        res = result_future.result()
        ok = res is not None and res.result.error_code == 0
        if not ok:
            ec = None if res is None else res.result.error_code
            self.get_logger().error(
                f"arm trajectory failed: error_code={ec}"
            )
        return ok
