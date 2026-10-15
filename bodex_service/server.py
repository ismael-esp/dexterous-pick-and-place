"""Persistent BODex grasp worker.

Run this inside the bodex conda env. Listens on a Unix domain socket for
grasp synthesis requests; keeps cuRobo + CUDA kernels resident between calls.

Typical startup (handled automatically by BODexClient):

    /home/robot/anaconda3/envs/bodex/bin/python -m bodex_service.server \
        --socket /tmp/bodex_service.sock

Protocol:
  Request  (pickled dict over multiprocessing.connection):
    {"op": "synthesize", "active_arm": "ra", "object_id": "red_cube",
     "object_pose": [x,y,z,qw,qx,qy,qz], "place_xyz": [x,y,z],
     "approach": "top", "seed_num": 20, "hold_frames": 10,
     "transit_n": 40, "descend_n": 20, "release_n": 12,
     "table_z": 0.75, ...}
    {"op": "ping"}
    {"op": "shutdown"}

  Response:
    {"ok": True, "result": {trajectory, pick_end_idx, grasp_error, ...}}
    {"ok": False, "error": "<message>"}
"""
from __future__ import annotations

import argparse
import gc
import os
import signal
import sys
import traceback
from multiprocessing.connection import Listener

import numpy as np


def _find_isaaclab_tasks_on_path():
    """Locate the BODex repo's isaaclab_tasks/ so we can import bodex_plan_cli."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(os.path.dirname(here), "isaaclab_tasks"),           # BODex/isaaclab_tasks
        os.environ.get("BODEX_ISAACLAB_TASKS_PATH", ""),
    ]
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "bodex_pick_place")):
            if c not in sys.path:
                sys.path.insert(0, c)
            return c
    raise RuntimeError(
        "Could not find isaaclab_tasks/bodex_pick_place. Set BODEX_ISAACLAB_TASKS_PATH."
    )


def _serialize_result(result: dict) -> dict:
    """Convert torch/numpy objects to pure numpy arrays for pickling."""
    out = {}
    for k, v in result.items():
        if hasattr(v, "cpu"):           # torch tensor
            out[k] = v.cpu().numpy()
        elif isinstance(v, np.ndarray):
            out[k] = v
        else:
            out[k] = v
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--socket", default="/tmp/bodex_service.sock")
    p.add_argument("--authkey", default="bodex-service")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    # Clean stale socket file if present
    if os.path.exists(args.socket):
        try:
            os.unlink(args.socket)
        except OSError:
            pass

    # Heavy imports happen ONCE here.
    _find_isaaclab_tasks_on_path()
    from curobo.types.base import TensorDeviceType
    from curobo.util.logger import setup_logger

    setup_logger("warn")
    tensor_args = TensorDeviceType()

    # Import the core synthesis function. This loads cuRobo kernels.
    from bodex_pick_place.bodex_plan_cli import synthesize_full_trajectory

    # Torch import for cache-clearing between requests (memory hygiene)
    import torch

    # Warm up CUDA once by running a tiny IK through grasp_solver machinery.
    # Skip the warm-up — first real request does this at ~same cost.

    listener = Listener(args.socket, family="AF_UNIX", authkey=args.authkey.encode())
    print(f"[bodex-server] ready on {args.socket}", flush=True)

    # Handle SIGTERM cleanly
    def _shutdown(*_):
        print("[bodex-server] shutting down", flush=True)
        try:
            listener.close()
        finally:
            if os.path.exists(args.socket):
                os.unlink(args.socket)
            sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    shutdown_requested = False
    try:
        while not shutdown_requested:
            conn = listener.accept()
            try:
                req = conn.recv()
                op = req.get("op")
                if op == "ping":
                    conn.send({"ok": True, "result": {"pong": True}})
                elif op == "shutdown":
                    conn.send({"ok": True, "result": {"bye": True}})
                    shutdown_requested = True
                elif op == "synthesize":
                    try:
                        # DETERMINISM: cuRobo 0.7.5 removed implicit seeding, and
                        # the BODex grasp optimizer samples RANDOM initial grasp
                        # seeds. Without re-seeding here, the worker's RNG advances
                        # across calls, so the SAME object+pose yields a DIFFERENT
                        # (sometimes good, sometimes junk) grasp every request. Seed
                        # before each synthesis so a given pose is reproducible.
                        _seed = int(req.get("synth_seed", 123))
                        np.random.seed(_seed)
                        torch.manual_seed(_seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(_seed)
                        passive_arm = req.get("passive_arm", None)
                        passive_q = req.get("passive_joint_state", None)
                        if passive_q is not None:
                            passive_q = np.asarray(passive_q, dtype=float)
                        res = synthesize_full_trajectory(
                            active_arm=req["active_arm"],
                            object_id=req["object_id"],
                            object_pose=np.asarray(req["object_pose"], dtype=float),
                            place_xyz=np.asarray(req["place_xyz"], dtype=float),
                            approach=req.get("approach", "top"),
                            seed_num=int(req.get("seed_num", 20)),
                            hold_frames=int(req.get("hold_frames", 10)),
                            transit_n=int(req.get("transit_n", 40)),
                            descend_n=int(req.get("descend_n", 20)),
                            release_n=int(req.get("release_n", 12)),
                            table_z=float(req.get("table_z", 0.75)),
                            passive_arm=passive_arm,
                            passive_joint_state=passive_q,
                            thumb_up_min=float(req.get("thumb_up_min", 0.65)),
                            thumb_up_strict=bool(req.get("thumb_up_strict", False)),
                            standoff_dist=float(req.get("standoff_dist", 0.10)),
                            grasp_extra_in=float(req.get("grasp_extra_in", 0.0)),
                            grasp_z_bonus=float(req.get("grasp_z_bonus", 0.0)),
                            top_front_min=float(req.get("top_front_min", 0.0)),
                            top_level_relax=bool(req.get("top_level_relax", True)),
                            grasp_body_bias=float(req.get("grasp_body_bias", 0.0)),
                            # whale: the grip must close ACROSS the body, which
                            # is the NEGATIVE alignment sense (palm-Y = grip
                            # axis, perpendicular to the spine). Force -abs so
                            # a stale client config can never send the wrong
                            # sign for this object (2026-08-27 hardware finding).
                            grasp_axis_align=(
                                -abs(float(req.get("grasp_axis_align", 0.0)))
                                if req["object_id"] == "whale"
                                else float(req.get("grasp_axis_align", 0.0))),
                            grasp_center_bias=float(req.get("grasp_center_bias", 0.0)),
                            grasp_mode=str(req.get("grasp_mode", "wrap")),
                            obj_scale=req.get("obj_scale", (1.0, 1.0, 1.0)),
                            post_grasp_lift_z=float(req.get("post_grasp_lift_z", 0.10)),
                            transit_hover_z=float(req.get("transit_hover_z", 0.15)),
                            side_face_z_min=float(req.get("side_face_z_min", -1.0)),
                            # 0 (or absent) = keep the module default
                            top_level_min=(float(req["top_level_min"])
                                           if float(req.get("top_level_min", 0.0)) > 0.0
                                           else None),
                            obb_z_world=req.get("obb_z_world", None),
                            release_lift_z=float(req.get("release_lift_z", 0.08)),
                            tensor_args=tensor_args,
                        )
                        conn.send({"ok": True, "result": _serialize_result(res)})
                        if args.verbose:
                            print(f"[bodex-server] synthesized {req['object_id']}/"
                                  f"{req.get('approach', 'top')} "
                                  f"T={res['trajectory'].shape[0]}", flush=True)
                    except Exception as e:
                        tb = traceback.format_exc()
                        conn.send({"ok": False, "error": f"{e}\n{tb}"})
                    finally:
                        # Free CUDA memory held by intermediate cuRobo tensors
                        # (GraspSolver, world_cfg, IK batches). Without this,
                        # memory accumulates across calls and OOMs within a
                        # few dozen synthesis requests.
                        try:
                            res = None  # drop last trajectory reference
                        except NameError:
                            pass
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                else:
                    conn.send({"ok": False, "error": f"unknown op: {op}"})
            except (EOFError, ConnectionResetError):
                pass
            finally:
                conn.close()
    finally:
        listener.close()
        if os.path.exists(args.socket):
            try:
                os.unlink(args.socket)
            except OSError:
                pass


if __name__ == "__main__":
    main()
