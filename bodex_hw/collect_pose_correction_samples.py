#!/usr/bin/env python3
"""Collect (P_fp, P_true) calibration pairs for fit_pose_correction.py.

Procedure per point: place the object, jog/freedrive the RIGHT arm so the index
fingertip (ra_link_3_0_tip) touches the object's centroid/top, then press ENTER.
It records one CSV row:  fpx,fpy,fpz,truex,truey,truez  where
  P_fp   = current /<obj>_pose position (what FoundationPose reports), and
  P_true = current  world -> ra_link_3_0_tip  TF translation (ground truth).

Collect ~10-15 points spread across the right hand's reachable zone, a couple of
readings per spot to average out noise, then fit:
  python3 bodex_hw/fit_pose_correction.py --csv samples.csv \
      --out bodex_hw/pose_correction_cam2.json

Run from a sourced ROS2 shell (ROS2 humble):
  source /opt/ros/humble/setup.bash
  source ~/Dualarm_ws/ros2_ws_jazzy/install/setup.bash
  python3 bodex_hw/collect_pose_correction_samples.py \
      --object_pose_topic /green_cube_pose --out samples.csv
"""
# Standard Library
import argparse
import threading

# Third Party
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import tf2_ros


class Collector(Node):
    def __init__(self, topic, fingertip, world):
        super().__init__("pose_corr_collector")
        self._p = None
        self.fingertip = fingertip
        self.world = world
        self.create_subscription(PoseStamped, topic, self._cb, 10)
        self.tfbuf = tf2_ros.Buffer()
        self.tfl = tf2_ros.TransformListener(self.tfbuf, self)

    def _cb(self, msg):
        self._p = msg.pose.position

    def sample(self):
        if self._p is None:
            self.get_logger().warn("no /<obj>_pose received yet")
            return None
        pfp = (self._p.x, self._p.y, self._p.z)
        try:
            t = self.tfbuf.lookup_transform(self.world, self.fingertip, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f"TF {self.world}->{self.fingertip} failed: {e}")
            return None
        tr = t.transform.translation
        return pfp, (tr.x, tr.y, tr.z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object_pose_topic", default="/green_cube_pose")
    ap.add_argument("--fingertip_frame", default="ra_link_3_0_tip")
    ap.add_argument("--world_frame", default="world")
    ap.add_argument("--out", default="samples.csv")
    a = ap.parse_args()

    rclpy.init()
    node = Collector(a.object_pose_topic, a.fingertip_frame, a.world_frame)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    print("Touch fingertip to the object, press ENTER to record (q+ENTER to quit).")
    n = 0
    with open(a.out, "a") as f:
        try:
            while True:
                if input().strip().lower() == "q":
                    break
                r = node.sample()
                if r is None:
                    print("  no pose/TF yet — retry")
                    continue
                pfp, ptrue = r
                f.write(",".join(f"{v:.5f}" for v in (*pfp, *ptrue)) + "\n")
                f.flush()
                n += 1
                d = [pfp[i] - ptrue[i] for i in range(3)]
                print(f"  #{n}  P_fp=({pfp[0]:.3f},{pfp[1]:.3f},{pfp[2]:.3f})  "
                      f"P_true=({ptrue[0]:.3f},{ptrue[1]:.3f},{ptrue[2]:.3f})  "
                      f"d_xy=({d[0]:+.3f},{d[1]:+.3f})")
        finally:
            rclpy.shutdown()
            print(f"wrote {n} rows to {a.out}\nnow: python3 bodex_hw/fit_pose_correction.py "
                  f"--csv {a.out} --out bodex_hw/pose_correction_cam2.json")


if __name__ == "__main__":
    main()
