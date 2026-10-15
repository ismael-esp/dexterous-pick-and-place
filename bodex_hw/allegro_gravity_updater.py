#!/usr/bin/env python3
"""Continuously update the Allegro hand's gravity-compensation vector
from TF, so the driver's built-in gravity-comp loop knows which way
gravity actually pulls in each hand's PALM-LINK frame as the arm moves.

The Allegro driver (allegro_hand_cust) constructs its controller with
``gravity = (0, 0, -9.80665)`` IN ITS OWN BASE LINK FRAME and never
updates it (see ``AllegroHandController.cpp:62``). On a fixed mount with
the palm always level facing down, that's fine. On our dual-arm UR3
setup, the palm orientation changes constantly with the arm pose, so
the gravity comp is computed against the wrong world-frame direction
— most visible as THUMB DROOP (the thumb has the largest moment arm
and the most off-axis CoM, so a few-degree gravity-vector error is
visibly absorbed by the kp loop).

This node:

  1. Looks up ``<base_frame> -> <ra_palm_frame>`` and
     ``<base_frame> -> <la_palm_frame>`` from /tf.
  2. Rotates world gravity ``(0, 0, -9.80665)`` (expressed in
     ``<base_frame>``) into each palm frame.
  3. Calls ``/<prefix>setGravityVector`` at ``--rate`` Hz with the
     palm-frame vector.

Run it in PARALLEL with the pick-and-place runner:

    source /opt/ros/humble/setup.bash
    source ~/Dualarm_ws/ros2_ws_jazzy/install/setup.bash
    python3 ~/Dualarm_ws/BODex/bodex_hw/allegro_gravity_updater.py \
        --hand both --rate 20

If your TF frame names differ from the defaults, override:

    --base_frame world \
    --ra_palm_frame ra_base_link --la_palm_frame la_base_link

A typical first session: pause the arm at home with the hand
commanded fully open, run this with ``--once``, observe the thumb's
new equilibrium. It should be MUCH closer to fully extended.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from tf2_ros import Buffer, TransformListener, LookupException, \
    ExtrapolationException, ConnectivityException

from allegro_hand_interfaces.srv import SetGravityVector


G_WORLD_FRAME = np.array([0.0, 0.0, -9.80665], dtype=float)


def quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """xyzw quaternion -> 3x3 rotation matrix."""
    x, y, z, w = float(qx), float(qy), float(qz), float(qw)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=float)


class GravityUpdater(Node):
    def __init__(self, base_frame: str):
        super().__init__("allegro_gravity_updater")
        self._base_frame = base_frame
        self._tf_buf = Buffer()
        self._tf_list = TransformListener(self._tf_buf, self)
        self._clients: dict[str, object] = {}

    def _client(self, prefix: str):
        """Cached SetGravityVector client for ``/<prefix>setGravityVector``."""
        if prefix not in self._clients:
            self._clients[prefix] = self.create_client(
                SetGravityVector, f"/{prefix}setGravityVector"
            )
        return self._clients[prefix]

    def grav_in_palm_frame(self, palm_frame: str, timeout_s: float
                           ) -> np.ndarray:
        """Returns world gravity expressed in the palm-link frame, by
        looking up ``palm_frame <- base_frame`` from TF."""
        tf = self._tf_buf.lookup_transform(
            palm_frame, self._base_frame, Time(),
            Duration(seconds=timeout_s)
        )
        r = tf.transform.rotation
        R_palm_from_base = quat_to_rotmat(r.x, r.y, r.z, r.w)
        return R_palm_from_base @ G_WORLD_FRAME

    def set_palm_gravity(self, prefix: str, g_palm: np.ndarray,
                         srv_timeout_s: float = 2.0) -> bool:
        cli = self._client(prefix)
        if not cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(
                f"/{prefix}setGravityVector not available"
            )
            return False
        req = SetGravityVector.Request()
        req.gravity_vector = [float(g_palm[0]), float(g_palm[1]),
                              float(g_palm[2])]
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=srv_timeout_s)
        resp = fut.result()
        ok = bool(resp is not None and resp.success)
        if not ok:
            self.get_logger().warn(
                f"/{prefix}setGravityVector returned !success"
            )
        return ok


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--hand", choices=["ra", "la", "both"],
                        default="both",
                        help="which hand(s) to update")
    parser.add_argument("--base_frame", default="world",
                        help="world-frame TF parent (default: world)")
    parser.add_argument("--ra_palm_frame", default="ra_base_link",
                        help="RA palm/base TF frame (the URDF root of "
                             "the right Allegro). Default: ra_base_link.")
    parser.add_argument("--la_palm_frame", default="la_base_link",
                        help="LA palm/base TF frame. Default: la_base_link.")
    parser.add_argument("--rate", type=float, default=20.0,
                        help="update rate (Hz). The arm moves slowly "
                             "relative to control loop, 10-20 Hz is plenty.")
    parser.add_argument("--once", action="store_true",
                        help="set gravity once and exit (debugging)")
    parser.add_argument("--tf_timeout_s", type=float, default=2.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    rclpy.init()
    node = GravityUpdater(args.base_frame)

    hand_defs = []
    if args.hand in ("ra", "both"):
        hand_defs.append(("ra_", args.ra_palm_frame))
    if args.hand in ("la", "both"):
        hand_defs.append(("la_", args.la_palm_frame))

    # Warm up TF (it takes a moment after node init to receive the tree).
    node.get_logger().info(
        f"warming up TF (base={args.base_frame} -> "
        f"{[p for _, p in hand_defs]}) ..."
    )
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.tf_timeout_s:
        rclpy.spin_once(node, timeout_sec=0.05)
        try:
            for _, palm in hand_defs:
                node.grav_in_palm_frame(palm, timeout_s=0.1)
            break
        except (LookupException, ExtrapolationException,
                ConnectivityException):
            continue
    else:
        node.get_logger().error(
            "TF tree not seen — check that robot_state_publisher is up "
            "and the frame names match (use --ra_palm_frame / "
            "--la_palm_frame to override)."
        )

    period = 1.0 / max(args.rate, 0.5)
    rc = 0
    try:
        while rclpy.ok():
            for prefix, palm in hand_defs:
                try:
                    g_palm = node.grav_in_palm_frame(palm, timeout_s=0.3)
                except Exception as e:
                    node.get_logger().warn(f"[{prefix}] TF lookup failed: {e}")
                    continue
                ok = node.set_palm_gravity(prefix, g_palm)
                if args.verbose or args.once:
                    node.get_logger().info(
                        f"{prefix}grav_in_palm = "
                        f"({g_palm[0]:+.3f}, {g_palm[1]:+.3f}, "
                        f"{g_palm[2]:+.3f}) m/s^2  -> "
                        f"{'OK' if ok else 'FAIL'}"
                    )
            if args.once:
                break
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
