"""Quick matplotlib visualizer for a BODex mogen `.npy` trajectory.

Renders the arm + Allegro fingertips animating through the planned waypoints,
with the object mesh overlaid. No Isaac Sim or USD viewer required.

Usage:
    python example_grasp/visualize_mogen.py \
        --mogen_npy <path-to-*_mogen.npy> \
        --robot_cfg ra_ur3_allegro.yml \
        [--seed 0] [--save_mp4 trajectory.mp4]
"""
# Standard Library
import argparse
import os
import sys

# Third Party
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import (get_assets_path, get_robot_configs_path, join_path,
                              load_scene_cfg, load_yaml)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mogen_npy", required=True)
    p.add_argument("--robot_cfg", required=True,
                   help="e.g. ra_ur3_allegro.yml or la_ur3e_allegro.yml")
    p.add_argument("--seed", type=int, default=None,
                   help="Which seed to visualize. Default: best (lowest grasp_error).")
    p.add_argument("--save_mp4", default=None, help="Optional output mp4 path.")
    p.add_argument("--save_png", default=None, help="Save final-frame snapshot.")
    p.add_argument("--show", action="store_true", help="Show window (requires display).")
    args = p.parse_args()

    # Load mogen output
    d = np.load(args.mogen_npy, allow_pickle=True).item()
    rp = d["robot_pose"]   # [1, N_seeds, T, 22]
    ge = d["grasp_error"]  # [1, N_seeds, M]
    seed = args.seed if args.seed is not None else int(np.argmin(ge.max(axis=-1)[0]))
    err = float(ge.max(axis=-1)[0, seed])
    print(f"seed={seed}  grasp_error={err:.4f}  trajectory shape={rp.shape}")
    traj = torch.tensor(rp[0, seed], dtype=torch.float32, device="cuda")  # [T, 22]

    # Build kinematics for FK with an explicit link ordering so the plotted
    # arm chain matches the physical joint chain. `link_names` field in the
    # robot cfg lists only fingertips; override it to include the arm chain
    # too so `get_state().links_position` contains what we need.
    ta = TensorDeviceType()
    cfg = load_yaml(join_path(get_robot_configs_path(), args.robot_cfg))
    arm_prefix = cfg["robot_cfg"]["kinematics"]["base_link"].split("_")[0] + "_" \
        if cfg["robot_cfg"]["kinematics"]["base_link"] not in ("world", "base_link") else ""
    # For overhead URDF (base_link=world), the arm has ra_/la_ prefix. Read
    # from the URDF path to decide.
    urdf_path = cfg["robot_cfg"]["kinematics"]["urdf_path"]
    arm_prefix = "ra_" if "ra_" in urdf_path else ("la_" if "la_" in urdf_path else "")
    # Canonical UR arm chain links (prefix-adjusted) in order from base to wrist.
    arm_chain_names = [
        f"{arm_prefix}shoulder_link",
        f"{arm_prefix}upper_arm_link",
        f"{arm_prefix}forearm_link",
        f"{arm_prefix}wrist_1_link",
        f"{arm_prefix}wrist_2_link",
        f"{arm_prefix}wrist_3_link",
        f"{arm_prefix}palm_link",
    ]
    finger_tips = list(cfg["robot_cfg"]["kinematics"]["link_names"])
    # Preserve originals, then attach chain + fingers for FK query.
    all_queried = arm_chain_names + finger_tips
    # Build a kinematics config that reports positions for our chain.
    cfg["robot_cfg"]["kinematics"]["link_names"] = all_queried
    cfg["robot_cfg"]["kinematics"]["ee_link"] = all_queried[0]
    rc = RobotConfig.from_dict(cfg, ta)
    model = CudaRobotModel(rc.kinematics)

    states = model.get_state(traj)
    if hasattr(states, "links_position") and states.links_position is not None:
        all_link_positions = states.links_position.cpu().numpy()   # [T, 11, 3]
    else:
        all_link_positions = None
    n_arm = len(arm_chain_names)

    # Object: use scene_path from .npy → load scene_cfg → mesh path
    obj_pts = None
    obj_pose = None
    try:
        scene_path = str(d.get("scene_path", ""))
        if scene_path and os.path.exists(scene_path):
            scene_cfg = load_scene_cfg(scene_path)
            obj_name = scene_cfg["task"]["obj_name"]
            obj = scene_cfg["scene"][obj_name]
            obj_pose = np.asarray(obj["pose"], dtype=float)
            obj_mesh_path = obj["file_path"]
            import trimesh
            tm = trimesh.load(obj_mesh_path, force="mesh")
            tm.apply_scale(float(obj["scale"][0]))
            # Apply pose
            from scipy.spatial.transform import Rotation
            R = Rotation.from_quat([obj_pose[4], obj_pose[5], obj_pose[6], obj_pose[3]]).as_matrix()
            obj_pts = (tm.vertices @ R.T) + obj_pose[:3]
    except Exception as e:
        print(f"[warn] could not load object mesh: {e}")

    # Set up figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Compute scene bounds for consistent view
    all_pts = all_link_positions.reshape(-1, 3) if all_link_positions is not None else np.zeros((1, 3))
    if obj_pts is not None:
        all_pts = np.concatenate([all_pts, obj_pts], axis=0)
    pad = 0.1
    bx = (all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    by = (all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
    bz = (max(0.0, all_pts[:, 2].min() - pad), all_pts[:, 2].max() + pad)
    span = max(bx[1] - bx[0], by[1] - by[0], bz[1] - bz[0])
    cx, cy, cz = sum(bx) / 2, sum(by) / 2, sum(bz) / 2

    def draw_frame(t_idx):
        ax.cla()
        ax.set_xlim(cx - span / 2, cx + span / 2)
        ax.set_ylim(cy - span / 2, cy + span / 2)
        ax.set_zlim(cz - span / 2, cz + span / 2)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

        # Table plane (z=0.75 default)
        if "robot_cfg" in cfg:
            table_z = 0.75
            xx, yy = np.meshgrid(np.linspace(cx - span / 2, cx + span / 2, 5),
                                 np.linspace(cy - span / 2, cy + span / 2, 5))
            zz = np.full_like(xx, table_z)
            ax.plot_surface(xx, yy, zz, alpha=0.15, color="tan")

        # Object
        if obj_pts is not None:
            ax.scatter(obj_pts[:, 0], obj_pts[:, 1], obj_pts[:, 2],
                       s=2, c="orange", alpha=0.4, label="object")

        # Robot arm + fingertips
        if all_link_positions is not None:
            pts = all_link_positions[t_idx]   # [n_arm+4, 3]
            arm_chain = pts[:n_arm]
            ax.plot(arm_chain[:, 0], arm_chain[:, 1], arm_chain[:, 2],
                    "b-o", lw=3, ms=6, label="arm")
            tips = pts[n_arm:n_arm + 4]
            ax.scatter(tips[:, 0], tips[:, 1], tips[:, 2],
                       s=100, c="red", marker="*", label="fingertips")

        ax.set_title(f"waypoint {t_idx + 1}/{len(traj)}  (seed {seed}, grasp_err={err:.3f})")
        ax.legend(loc="upper left")

    if args.save_png:
        draw_frame(len(traj) - 1)
        os.makedirs(os.path.dirname(args.save_png) or ".", exist_ok=True)
        plt.savefig(args.save_png, dpi=120, bbox_inches="tight")
        print(f"saved snapshot → {args.save_png}")

    if args.save_mp4:
        ani = FuncAnimation(fig, draw_frame, frames=len(traj), interval=80)
        os.makedirs(os.path.dirname(args.save_mp4) or ".", exist_ok=True)
        try:
            writer = FFMpegWriter(fps=12, bitrate=2400)
            ani.save(args.save_mp4, writer=writer)
            print(f"saved animation → {args.save_mp4}")
        except Exception as e:
            print(f"[warn] mp4 save failed ({e}); falling back to GIF")
            gif_path = args.save_mp4.replace(".mp4", ".gif")
            ani.save(gif_path, writer="pillow", fps=12)
            print(f"saved animation → {gif_path}")

    if args.show:
        ani = FuncAnimation(fig, draw_frame, frames=len(traj), interval=80)
        plt.show()


if __name__ == "__main__":
    sys.exit(main())
