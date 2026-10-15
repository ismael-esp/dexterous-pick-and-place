"""FoundationPose → BODex scene_cfg bridge.

Subscribes to a FoundationPose object-pose topic (PoseStamped in world frame,
Z-up), and on demand writes a BODex runtime scene_cfg .npy for the named
object using the pose from FoundationPose.

Then you run `plan_mogen_batch.py` with a manip config whose template_path
points at `object/custom/scene_cfg/<obj_id>/runtime/*.npy` to synthesize a
fresh grasp for the currently detected pose.

Usage:
    # terminal A (conda bodex env):
    python example_grasp/fp_scene_cfg_bridge.py \\
        --object_id peach \\
        --object_topic /peach_pose \\
        --scene_cfg_root src/curobo/content/assets/object/custom/scene_cfg \\
        --table_z 0.75

    # whenever you want a fresh grasp for the latest detected pose:
    ros2 service call /bodex/trigger_scene_cfg std_srvs/srv/Trigger
    # or write a keypress trigger externally

This node has no ROS2 dependency on the BODex conda env; instead, if rclpy
is unavailable (conda python doesn't have ROS2 installed), the script falls
back to a --fake_pose mode that writes a scene_cfg without actually
subscribing to ROS2 — handy for offline testing.
"""
# Standard Library
import argparse
import os
import sys
import time

# <repo>/src/curobo/content/assets/object/custom, resolved from this file's location
_CUSTOM_OBJ_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "curobo", "content", "assets", "object", "custom")
from pathlib import Path

# Third Party
import numpy as np


def write_runtime_scene_cfg(
    out_path: str,
    obj_id: str,
    processed_data_dir: str,
    object_pose_world: np.ndarray,
    scale: float = 1.0,
    table_z: float = 0.75,
    include_table: bool = True,
):
    """Write a single runtime scene_cfg .npy that BODex can consume.

    object_pose_world: length-7 [x, y, z, qw, qx, qy, qz]
    processed_data_dir: path where preprocess_custom_objs.py wrote
        processed_data/<obj_id>/ (used to compute relative paths)
    """
    scene_cfg_parent = os.path.dirname(out_path)
    obj_data_dir = os.path.join(processed_data_dir, obj_id)
    # Paths inside the scene_cfg must be relative to scene_cfg's parent dir
    rel = os.path.relpath(obj_data_dir, scene_cfg_parent)
    entry = {
        "type": "rigid_object",
        "scale": np.array([scale, scale, scale], dtype=float),
        "pose": np.asarray(object_pose_world, dtype=float),
        "file_path": os.path.join(rel, "mesh", "simplified.obj"),
        "urdf_path": os.path.join(rel, "urdf", "coacd.urdf"),
        "info_path": os.path.join(rel, "info", "simplified.json"),
    }
    scene = {obj_id: entry}
    if include_table:
        scene["table"] = {
            "type": "plane",
            "pose": np.array([0., 0., table_z, 1., 0., 0., 0.]),
            "size": np.array([0., 0., 1.]),
        }
    cfg = {
        "scene": scene,
        "scene_id": f"{obj_id}/runtime/latest",
        "task": {"obj_name": obj_id, "type": "slide",
                 "axis": np.array([0., 0., 1.]), "distance": 0.1},
    }
    os.makedirs(scene_cfg_parent, exist_ok=True)
    np.save(out_path, cfg, allow_pickle=True)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--object_id", required=True,
                   help="Matching subfolder under processed_data/ (e.g. peach)")
    p.add_argument("--object_topic",
                   help="PoseStamped topic. Omit to run in --fake_pose mode.")
    p.add_argument("--processed_data_dir",
                   default=os.path.join(_CUSTOM_OBJ_DIR, "processed_data"))
    p.add_argument("--scene_cfg_root",
                   default=os.path.join(_CUSTOM_OBJ_DIR, "scene_cfg"))
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--table_z", type=float, default=0.75)
    p.add_argument("--fake_pose", nargs=7, type=float, metavar=("X","Y","Z","QW","QX","QY","QZ"),
                   help="Skip ROS2; write a scene_cfg with this pose once and exit.")
    args = p.parse_args()

    out_path = os.path.join(args.scene_cfg_root, args.object_id, "runtime", "latest.npy")

    if args.fake_pose is not None:
        pose = np.array(args.fake_pose, dtype=float)
        write_runtime_scene_cfg(out_path, args.object_id, args.processed_data_dir,
                                pose, args.scale, args.table_z)
        print(f"[fake_pose] wrote {out_path} at pose {pose}")
        return 0

    # ROS2 mode
    try:
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import PoseStamped
        from std_srvs.srv import Trigger
    except ImportError as e:
        print(f"rclpy not available ({e}). Use --fake_pose for offline writing, "
              f"or invoke from a ROS2-overlay python.")
        return 1

    class Bridge(Node):
        def __init__(self, args, out_path):
            super().__init__("fp_scene_cfg_bridge")
            self.args = args
            self.out_path = out_path
            self.latest_pose = None
            self.sub = self.create_subscription(
                PoseStamped, args.object_topic, self._pose_cb, 10)
            self.srv = self.create_service(
                Trigger, "/bodex/trigger_scene_cfg", self._trigger_cb)
            self.get_logger().info(
                f"Listening on {args.object_topic}, writes to {out_path}")

        def _pose_cb(self, msg: 'PoseStamped'):
            p = msg.pose.position
            q = msg.pose.orientation
            # Convert geometry_msgs quaternion (x,y,z,w) to curobo (w,x,y,z)
            self.latest_pose = np.array([p.x, p.y, p.z, q.w, q.x, q.y, q.z], dtype=float)

        def _trigger_cb(self, request, response):
            if self.latest_pose is None:
                response.success = False
                response.message = "No pose received yet."
                return response
            write_runtime_scene_cfg(
                self.out_path, self.args.object_id, self.args.processed_data_dir,
                self.latest_pose, self.args.scale, self.args.table_z)
            response.success = True
            response.message = f"Wrote scene_cfg to {self.out_path}"
            self.get_logger().info(response.message)
            return response

    rclpy.init()
    node = Bridge(args, out_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
