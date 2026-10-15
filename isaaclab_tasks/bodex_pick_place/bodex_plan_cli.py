"""CLI entry point run INSIDE the bodex conda env.

Takes an object pose + target place xyz + active arm, invokes BODex to
synthesize a full 22-DoF trajectory [home → approach → grasp → squeeze →
lift → transit → descend → release], saves to --out_npy for the Isaac Lab
task (running in env_isaaclab) to read back.

Usage (called as a subprocess from Isaac Lab env):

    /home/robot/anaconda3/envs/bodex/bin/python \\
        /mnt/ssd2/ws/Dualarm_ws/BODex/isaaclab_tasks/bodex_pick_place/bodex_plan_cli.py \\
        --active_arm ra --object_id peach \\
        --object_pose 0.1 0.45 1.16 1 0 0 0 \\
        --place_xyz  0.2 0.40 1.18 \\
        --approach top \\
        --out_npy /tmp/bodex_traj.npy

Output .npy is a dict:
    {
      'trajectory': np.ndarray [T, 22]    (6 arm + 16 hand),
      'grasp_error': float,
      'arm': 'ra' | 'la',
      'pick_end_idx': int                 (last pick waypoint idx, rest is place),
      'success': bool,
    }
"""
# Standard Library
import argparse
import os
import sys

# <repo>/src/curobo/content/assets/object/custom, resolved from this file's location
_CUSTOM_OBJ_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src", "curobo", "content", "assets", "object", "custom")

# Third Party
import numpy as np
import torch

# CuRobo / BODex
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.sdf.world import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util.logger import log_warn, setup_logger
from curobo.util.world_cfg_generator import get_world_config_dataloader
from curobo.util_file import (
    get_manip_configs_path,
    get_robot_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.grasp_solver import GraspSolver, GraspSolverConfig


# ----------------------------------------------------------------------
# Module-level caches (lazy, loaded once per worker process)
# ----------------------------------------------------------------------
# Cached CudaRobotModel for each ARM_CFG_FILE we've seen, so FK on the
# passive arm doesn't pay the URDF + sphere-yaml load cost on every call.
# Keyed by (arm_yml_basename, device_str).
_PASSIVE_KIN_CACHE: dict[tuple[str, str], CudaRobotModel] = {}

_PASSIVE_ARM_YML = {
    "ra": "ra_ur3_allegro.yml",
    "la": "la_ur3e_allegro.yml",
}


def _get_passive_kinematics(
    passive_arm: str, tensor_args: TensorDeviceType
) -> CudaRobotModel | None:
    """Return a CudaRobotModel for the passive arm (cached).

    Uses the same robot YAML the GraspSolver uses for the active arm — it
    already references the per-link collision spheres
    (`spheres/{ra,la}_ur3_allegro.yml`). FK on this model gives world-frame
    sphere centers ready to inject as obstacles.

    Returns None if the YAML can't be found (graceful degrade — the planner
    just won't see the passive arm; same as before this change).
    """
    yml = _PASSIVE_ARM_YML.get(passive_arm)
    if yml is None:
        return None
    key = (yml, str(tensor_args.device))
    cached = _PASSIVE_KIN_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        robot_cfg_data = load_yaml(join_path(get_robot_configs_path(), yml))
        robot_cfg = RobotConfig.from_dict(robot_cfg_data["robot_cfg"], tensor_args)
        kin = CudaRobotModel(robot_cfg.kinematics)
    except Exception as e:
        log_warn(f"[passive-arm] failed to load kinematics for {passive_arm}: {e}")
        return None
    _PASSIVE_KIN_CACHE[key] = kin
    return kin


# ----------------------------------------------------------------------
# Approach collision guard
# ----------------------------------------------------------------------
# The arm IK that places the approach waypoints checks self-collision only at the
# CLOSED (grasp/transfer) hand pose, and the approach is built by interpolation that
# is not collision-checked. So the OPEN approach hand can sweep its fingers into the
# arm (self-collision) or dip below the table — neither is caught. This guard FK's
# the OPEN-hand approach with the full arm+hand model and rejects a grasp whose
# approach self-collides or hits the table, so the seed-retry picks the next-best
# grasp WITH a clean approach (quality preserved — we still take a top-ranked grasp).
_APPROACH_RW_CACHE: dict[tuple, object] = {}
APPROACH_COLL_TRIES = 6          # max higher-ranked grasps to try for a clean approach
APPROACH_COLL_MARGIN = 0.006     # m — table sphere activation distance (safety margin)
APPROACH_SELF_MARGIN = 0.004     # m — self-collision sphere activation distance
# Cost THRESHOLDS above which we call it a collision. The SelfCollisionCost is a
# smooth potential, not binary: measured ~0.0034 at home, ~0.005 on CLEAN grasp
# approaches, vs 0.015-0.032 for self-colliding poses (2000-sample calibration).
# 0.015 passes clean approaches with margin and flags real finger-vs-arm penetration.
# The world (table) cost reads ~0 when clear and jumps on penetration, so a tiny
# threshold suffices there.
APPROACH_SELF_THRESH = 0.015     # d_self above this => fingers actually hit the arm
APPROACH_WORLD_THRESH = 0.001    # d_world above this => a sphere penetrates the table


def _get_approach_robot_world(robot_file_with_arm: str, table_z: float,
                              tensor_args: TensorDeviceType):
    """Cached RobotWorld: full arm+hand robot (with finger collision spheres) + a
    TABLE-ONLY world (a flat box whose top face is at `table_z`). Table-only so the
    intended grasp contact with the OBJECT is never flagged — we only check the hand
    against the arm (self) and against the table. Returns None on any failure
    (guard degrades to off, never blocks synthesis)."""
    if robot_file_with_arm is None:
        return None
    key = (robot_file_with_arm, round(float(table_z), 4), str(tensor_args.device))
    if key in _APPROACH_RW_CACHE:
        return _APPROACH_RW_CACHE[key]
    rw = None
    try:
        from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
        # Table + the STATIC STRUCTURE (2026-09-02). The guard used to be
        # table-only, so a path that swung the arm into the T-shape torso column
        # was invisible to it — BODex's own synthesis world has the column, but
        # that only constrains the GRASP pose, not the approach/lift/transit
        # motion. Same cuboids as _append_static_obstacles(). The object itself
        # is still absent on purpose: the intended grasp contact must never be
        # flagged.
        table_world = {
            "cuboid": {
                "approach_guard_table": {
                    "dims": [4.0, 4.0, 0.4],
                    "pose": [0.0, 0.0, float(table_z) - 0.2, 1.0, 0.0, 0.0, 0.0],
                },
                "approach_guard_tshape_column": {
                    "dims": [0.22, 0.22, 1.40],
                    "pose": [0.0, -0.15, 0.70, 1.0, 0.0, 0.0, 0.0],
                },
                "approach_guard_backdrop_wall": {
                    "dims": [4.0, 0.05, 2.5],
                    "pose": [0.0, -0.60, 1.25, 1.0, 0.0, 0.0, 0.0],
                },
            }
        }
        cfg = RobotWorldConfig.load_from_config(
            robot_config=robot_file_with_arm,
            world_model=table_world,
            tensor_args=tensor_args,
            collision_activation_distance=APPROACH_COLL_MARGIN,
            self_collision_activation_distance=APPROACH_SELF_MARGIN,
        )
        rw = RobotWorld(cfg)
    except Exception as e:
        log_warn(f"[approach-collision] could not build the guard ({e}); "
                 "approach collision check is OFF.")
        rw = None
    _APPROACH_RW_CACHE[key] = rw
    return rw


def _approach_collision(robot_world, approach_traj) -> tuple[bool, str]:
    """Return (collides, reason). FK the OPEN-hand approach [H, 22] and check
    self-collision (fingers vs arm) and table collision with the full robot model.
    Never raises — on any error returns (False, '') so it never blocks a grasp."""
    if robot_world is None:
        return (False, "")
    try:
        q = approach_traj[:, :22].unsqueeze(0).contiguous()       # [1, H, 22]
        if q.shape[-1] != int(robot_world.kinematics.get_dof()):
            return (False, "")                                    # dof mismatch -> skip
        d_world, d_self = (
            robot_world.get_world_self_collision_distance_from_joint_trajectory(q))
        _ds = float(d_self.max().item()); _dw = float(d_world.max().item())
        if os.environ.get("BODEX_DEBUG_APPROACH_Z"):
            log_warn(f"[approach-collision] max d_self={_ds:.5f} (thr {APPROACH_SELF_THRESH}) "
                     f"d_world={_dw:.5f} (thr {APPROACH_WORLD_THRESH})")
        self_hit = _ds > APPROACH_SELF_THRESH
        table_hit = _dw > APPROACH_WORLD_THRESH
        # The world holds the table, the T-shape torso column and the backdrop
        # wall, so a world hit is any of them (cuRobo returns one max distance).
        if self_hit and table_hit:
            return (True, "the arm (self) + the table/torso")
        if self_hit:
            return (True, "the arm (self-collision)")
        if table_hit:
            return (True, "the table/torso structure")
        return (False, "")
    except Exception as e:
        log_warn(f"[approach-collision] check failed ({e}); not blocking.")
        return (False, "")


def _passive_arm_collision_spheres(
    passive_arm: str,
    passive_joint_state: np.ndarray,   # [N_dof] — full robot dof (arm + hand)
    tensor_args: TensorDeviceType,
):
    """Compute world-frame collision spheres for the passive arm at its
    current joint state, ready to inject into a cuRobo WorldConfig.

    `passive_joint_state` may be either:
      - the 6 arm joints only (caller's convenience) — hand joints assume
        zero (open hand at rest); or
      - the full DOF count expected by the cached robot model.

    Returns list[Sphere] (cuRobo geom type) or [] on any failure.
    """
    from curobo.geom.types import Sphere

    kin = _get_passive_kinematics(passive_arm, tensor_args)
    if kin is None:
        return []
    dof = kin.get_dof()

    q = np.asarray(passive_joint_state, dtype=float).reshape(-1)
    if q.shape[0] < dof:
        # Pad with zeros (open hand at rest) up to full DOF.
        q = np.concatenate([q, np.zeros(dof - q.shape[0])], dtype=float)
    elif q.shape[0] > dof:
        q = q[:dof]

    q_t = torch.as_tensor(q, device=tensor_args.device, dtype=torch.float32).view(1, dof)
    try:
        kin.update_batch_size(1, reset_buffers=True)
        state = kin.get_state(q_t)
        spheres_xyzr = state.get_link_spheres()        # [1, n_spheres, 4]
    except Exception as e:
        log_warn(f"[passive-arm] FK failed: {e}")
        return []
    if spheres_xyzr is None:
        return []
    s = spheres_xyzr.detach().cpu().numpy()[0]          # [n_spheres, 4]

    # Filter out invalid (zero / negative) radii — cuRobo pads sphere
    # tensors to a fixed buffer length and unused slots have radius < 0.
    keep = s[:, 3] > 1e-4
    s = s[keep]

    out = []
    for i, (x, y, z, r) in enumerate(s):
        out.append(
            Sphere(
                name=f"passive_{passive_arm}_sph_{i}",
                pose=[float(x), float(y), float(z), 1.0, 0.0, 0.0, 0.0],
                radius=float(r),
            )
        )
    return out


# The four approach presets — each is an ortho-6d rotation repr encoding the
# palm's X-axis (first 3) and Y-axis (next 3). Verified IK-reachable from
# the user's physical home configuration for both ra and la arms.
def _inject_scene_obstacles(
    world_cfg,
    passive_arm: str | None = None,
    passive_joint_state: np.ndarray | None = None,
    tensor_args: TensorDeviceType | None = None,
) -> None:
    """Append static + dynamic obstacles to the cuRobo WorldConfig.

    Static (Level 1):
      - T-shape torso column
      - Backdrop wall

    Dynamic (Level 2): if `passive_arm` and `passive_joint_state` are given,
    the passive arm's collision spheres at its current pose are FK'd and
    injected. This makes the planner aware of the OTHER arm — required
    before collaborative tasks where the passive arm doesn't sit at home.

    The torso's horizontal mount plates at z≈1.49 are intentionally excluded
    because each arm's own base sits right on them — a cuboid there would
    collide with the arm's base link.
    """
    # Third Party
    from curobo.geom.types import Cuboid

    def _cuboid(name, pose_xyz_q, size_xyz):
        return Cuboid(name=name, pose=list(pose_xyz_q), dims=list(size_xyz))

    cuboids = [
        _cuboid(
            name="tshape_column",
            pose_xyz_q=[0.0, -0.15, 0.70, 1.0, 0.0, 0.0, 0.0],
            size_xyz=[0.22, 0.22, 1.40],
        ),
        _cuboid(
            name="backdrop_wall",
            pose_xyz_q=[0.0, -0.60, 1.25, 1.0, 0.0, 0.0, 0.0],
            size_xyz=[4.0, 0.05, 2.5],
        ),
    ]
    if world_cfg.cuboid is None:
        world_cfg.cuboid = []
    world_cfg.cuboid.extend(cuboids)
    if world_cfg.objects is not None:
        world_cfg.objects.extend(cuboids)

    # Level-2: passive arm as moving obstacle.
    if passive_arm and passive_joint_state is not None and tensor_args is not None:
        spheres = _passive_arm_collision_spheres(
            passive_arm, passive_joint_state, tensor_args,
        )
        if spheres:
            if world_cfg.sphere is None:
                world_cfg.sphere = []
            world_cfg.sphere.extend(spheres)
            if world_cfg.objects is not None:
                world_cfg.objects.extend(spheres)
            log_warn(f"[level-2] injected {len(spheres)} passive-arm spheres ({passive_arm})")
        else:
            log_warn(f"[level-2] no passive-arm spheres available ({passive_arm}); falling back to Level-1 only")


# Per-contact L-BFGS grasp-error below which a finger counts as "solidly
# engaged" in force closure. Sits between BODex's tight (0.1) and loose (0.2)
# success thresholds. Used by the security-ranked seed selection to reward
# grasps that engage MORE fingers (full-hand wrap) over minimal pinches.
GRASP_ENGAGE_THRESH = 0.15

# Home-bias (Fix A, 2026-08-26): small penalty per radian of joint-space (L2)
# distance between a candidate's grasp ARM config and home, subtracted from the
# security score. Purpose: demote the flipped elbow/wrist IK families whose
# home->grasp sweep drags the Allegro through the UR's own links. DELIBERATELY
# GENTLE: a natural config sits ~1-2 rad from home (penalty 0.08-0.16, tiebreak
# territory), a full elbow flip ~3-5 rad (penalty 0.25-0.40, less than half an
# engaged-finger unit) — so a flipped config with one MORE engaged finger still
# wins, and side grasps (ALL candidates legitimately far from home by a similar
# amount) keep their relative ranking. 0.0 disables.
HOME_DIST_PENALTY = 0.08


APPROACH_PRESETS = {
    # TOP: grasp from above. Palm face points -Z (down), thumb -X.
    "top":        [0.0, 0.0, -1.0,  -1.0, 0.0, 0.0],

    # SIDE (arm-aware): the right choice for "side grasp" in 99% of cases —
    # the hand approaches from its OWN mount's side of the table with the
    # thumb pointing UP, so fingers don't scrape the table top.
    # Resolved at runtime in synthesize_pick_trajectory based on active_arm:
    #   ra → approach from +X world, palm presses -X, thumb +Z
    #   la → approach from -X world, palm presses +X, thumb +Z
    "side":       None,

    # Explicit side variants with THUMB UP (palm_y = world +Z). Use these
    # only if you want to force a specific world-frame side regardless of
    # which arm is active.
    "side_right": [-1.0, 0.0, 0.0,  0.0, 0.0, 1.0],   # from +X (palm→-X)
    "side_left":  [ 1.0, 0.0, 0.0,  0.0, 0.0, 1.0],   # from -X (palm→+X)

    # SIDE_FRONT: approach from toward-the-camera direction (world +Y).
    # Thumb up. Palm presses -Y (into the workspace).
    "side_front": [0.0, -1.0, 0.0,   0.0, 0.0, 1.0],
}


# Arm-specific resolution for the "side" preset — matches the intuition that
# each arm grasps from its OWN side so the fingers stay above the table.
_SIDE_BY_ARM = {
    "ra": [-1.0, 0.0, 0.0,  0.0, 0.0, 1.0],   # approach from +X, thumb +Z
    "la": [ 1.0, 0.0, 0.0,  0.0, 0.0, 1.0],   # approach from -X, thumb +Z
}


def _resolve_approach(approach: str, active_arm: str) -> list:
    """Return the ortho-6d r-vector for this (approach, active_arm) pair.

    `approach == "side"` picks the correct per-arm variant so the user never
    has to remember left vs right. `approach == "auto"` is resolved upstream
    (by env.py from object geometry); if it ever leaks through to here we
    fall back to "top" rather than raising.
    """
    if approach == "auto":
        log_warn("[approach] 'auto' reached BODex CLI; falling back to 'top'")
        approach = "top"
    if approach == "side":
        return _SIDE_BY_ARM[active_arm]
    return APPROACH_PRESETS[approach]


def _palm_y_world_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Return the palm's local +Y axis expressed in world coords for each
    quaternion in a batch [N, 4] with (w, x, y, z) ordering.

    Palm-Y in BODex's ortho-6d == thumb direction, so palm_y_world[:, 2] is
    the cosine of the angle between the thumb and world +Z.
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # Second column of R(q): (2(xy - wz), 1 - 2(x²+z²), 2(yz + wx))
    return torch.stack([
        2 * (x * y - w * z),
        1 - 2 * (x * x + z * z),
        2 * (y * z + w * x),
    ], dim=-1)


def _palm_z_world_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Return the palm's local +Z axis in world coords for (w,x,y,z) quats
    [N, 4]. Palm-X is the face and palm-Y the thumb, so palm-Z (= X x Y) runs
    ALONG THE FINGERS: for a level top grasp its horizontal projection is the
    direction the fingers point — used by the top heading gate to prefer
    hands coming from the FRONT (fingers away from the torso, world +Y)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # Third column of R(q): (2(xz + wy), 2(yz - wx), 1 - 2(x² + y²))
    return torch.stack([
        2 * (x * z + w * y),
        2 * (y * z - w * x),
        1 - 2 * (x * x + y * y),
    ], dim=-1)


def _palm_x_world_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Return the palm's local +X axis (the palm FACE direction) in world
    coords for a batch of (w, x, y, z) quaternions [N, 4].

    Palm-X in BODex's ortho-6d == the palm face. For the "top" preset the palm
    face points world -Z (straight down), so ``-palm_x_world[:, 2]`` is the
    cosine of the angle between the palm face and straight-down: 1.0 = a level
    (horizontal) hand, lower = a tilted/cocked hand.
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # First column of R(q): (1 - 2(y²+z²), 2(xy + wz), 2(xz - wy))
    return torch.stack([
        1 - 2 * (y * y + z * z),
        2 * (x * y + w * z),
        2 * (x * z - w * y),
    ], dim=-1)


# TOP grasps: prefer a LEVEL (horizontal) hand. A seed qualifies as level if its
# palm face is within ~acos(0.85) ≈ 32° of straight-down. Among level seeds we
# still maximize grip security (engaged fingers); the gate just keeps the hand
# from cocking toward the table (which the optimizer otherwise does to engage an
# extra finger). If NO seed is level enough, we fall back to the most-level
# feasible seed so synthesis never fails on this account.
TOP_LEVEL_MIN = 0.85

# Hard-object fallback: hard, rounded objects (e.g. the whale) sometimes have
# NO good LEVEL grasp — their only tight grasps are slightly tilted and get gated
# out, leaving a poor level grasp (a worst contact above _TRIGGER = a finger not
# engaged). When the best level grasp is that poor, re-rank over ALL feasible
# seeds (tilt allowed) and SWITCH only if the best tilted grasp is meaningfully
# tighter (worst-contact lower by at least _MARGIN). The release lift +
# level-at-release still clears the table for a tilted grasp, so this only trades
# a tilted GRASP for a tighter grip on objects that need it.
TOP_LEVEL_RELAX_TRIGGER = 0.18
TOP_LEVEL_RELAX_MARGIN = 0.02

# SIDE grasps: prefer a hand whose palm FACE points at/below horizontal toward
# the object (palm_x[2] = face_z <= this), i.e. the fingers wrap DOWN and FORWARD
# onto the object (the trial040 look) rather than the palm cocking UP so the
# fingers point up/back ("the hand looks up"). face_z=0 is a level side grasp;
# negative tilts the palm down (fingers down). Among seeds passing the gate we
# still rank by grip security; if none pass, fall back to the most-forward seed.
# Override per call via the `side_forward_face_max` kwarg (benchmark-tunable).
SIDE_FORWARD_FACE_MAX = 0.0

# APPROACH style for ALL side grasps (user spec): UP-OUT-DOWN-IN. Reach a HIGH
# via-point that is OVER the object top, sitting above an OUTSIDE (x,y) point
# (horizontally backed off from the grasp, at grasp height); descend straight DOWN
# at that outside point (clear of the object footprint); then move horizontally IN
# to the grasp. This keeps z over the object on the way to the outside point (never
# crosses/through it) and the thumb-up grasp selection keeps the thumb off the table.
# SIDE_OVER_CLEAR = how far above the object TOP the high via-point sits (m).
SIDE_OVER_CLEAR = 0.06
# Side-grasp height bias: among forward thumb-up grasps, add BONUS * palm_z to the
# security score to prefer a HIGHER grasp. This is OPT-IN per object (default 0 =
# pure grip security, the original behavior). It is meant ONLY for SHORT, uniform
# cylinders (green_can/tea_peach) whose down-pointing fingers otherwise hit the
# table — they have no "neck", so a higher grasp is still a clean body wrap.
# Do NOT bias mustard: a high bias drives its grasp up onto the narrow NECK, where
# the fingers cock UP and it tips the bottle (the trial041 BODY wrap is what we
# want). Live-tunable: `--grasp_z_bonus` / benchmark `set <obj> grasp_z_bonus <v>`.
SIDE_GRASP_Z_BONUS = 0.0
# Only switch to a tilted grasp if it is actually a GOOD grasp (worst contact
# this tight) — never trade a bad level grasp for a slightly-less-bad tilted one
# (that just adds tilt with no real grip gain; the episode fails either way).
# This keeps the relaxation from firing on EASY objects that merely drew a bad
# seed batch — it only helps hard objects (whale) that have a genuinely good
# tilted grasp the level gate was hiding.
TOP_LEVEL_RELAX_ACCEPT = 0.13


def synthesize_pick_trajectory(
    manip_cfg_name: str,
    object_id: str,
    object_pose_world: np.ndarray,   # [7] xyz + wxyz
    approach: str,
    seed_num: int,
    hold_frames: int,
    tensor_args: TensorDeviceType,
    passive_arm: str | None = None,
    passive_joint_state: np.ndarray | None = None,
    thumb_up_min: float = 0.65,
    thumb_up_strict: bool = False,
    standoff_dist: float = 0.10,
    standoff_n: int = 10,
    obb_z_world: float | None = None,
    grasp_extra_in: float = 0.0,
    grasp_z_bonus: float = SIDE_GRASP_Z_BONUS,
    top_front_min: float = 0.0,
    top_level_relax: bool = True,
    grasp_body_bias: float = 0.0,
    grasp_axis_align: float = 0.0,
    grasp_center_bias: float = 0.0,
    post_grasp_lift_z: float = 0.10,
    side_face_z_min: float = -1.0,
    top_level_min: float | None = None,
    table_z: float = 0.75,
):
    """Runs BODex grasp synthesis + IK-only mogen, returns pick trajectory.

    Pulls the manip cfg, overrides seeder_cfg.r with the approach preset +
    exp_name, and calls the same pipeline as plan_mogen_batch.py (minus
    file I/O).

    Returns: (pick_traj [T, 22], best_seed_idx int, grasp_error float,
              grasp_solver, pregrasp_arm_q [6], lift_arm_q [6],
              squeeze_hand_q [16], manip_config_data dict,
              feasible_count int, feasible_total int)

    ``feasible_count`` = seeds with all three IK stages (pregrasp+grasp+
    lift) feasible (after the optional side thumb-up filter);
    ``feasible_total`` = ``seed_num``. A low ratio or a high
    ``grasp_error`` flags a marginal grasp (the caller may gate on these).
    """
    manip_config_data = load_yaml(join_path(get_manip_configs_path(), manip_cfg_name))
    # Inject approach preset (resolves "side" per-arm)
    active_arm = "ra" if "ra_" in manip_cfg_name else "la"
    manip_config_data["seeder_cfg"]["r"] = _resolve_approach(approach, active_arm)
    manip_config_data["seed_num"] = seed_num

    # Point the world template at the one scene_cfg we just wrote (see
    # bridge_write_scene_cfg below).
    manip_config_data["world"]["template_path"] = (
        f"object/custom/scene_cfg/{object_id}/runtime/latest.npy"
    )

    world_generator = get_world_config_dataloader(manip_config_data["world"], 1)
    for world_info_dict in world_generator:
        world_model = [WorldConfig.from_dict(world_info_dict["world_cfg"][0])]
        # LEVEL-1 static obstacles + LEVEL-2 passive arm spheres. Kept here
        # (not in scene_cfg) because the upstream `scenecfg2worldcfg` only
        # understands rigid_object / plane.
        _inject_scene_obstacles(
            world_model[0],
            passive_arm=passive_arm,
            passive_joint_state=passive_joint_state,
            tensor_args=tensor_args,
        )
        grasp_config = GraspSolverConfig.load_from_robot_config(
            world_model=world_model,
            manip_name_list=world_info_dict["manip_name"],
            manip_config_data=manip_config_data,
            obj_gravity_center=world_info_dict["obj_gravity_center"],
            obj_obb_length=world_info_dict["obj_obb_length"],
            use_cuda_graph=False,
        )
        grasp_solver = GraspSolver(grasp_config)
        grasp_result = grasp_solver.solve_batch_env(return_seeds=grasp_solver.num_seeds)

        # Build 3-stage pose list (pregrasp / grasp / squeeze) in floating-hand
        # frame format [1, N, 3, 23]
        squeeze_pose_qpos = torch.cat(
            [
                grasp_result.solution[..., 1, :7],
                grasp_result.solution[..., 1, 7:] * 2 - grasp_result.solution[..., 0, 7:],
            ],
            dim=-1,
        )
        all_hand_pose_qpos = torch.cat(
            [grasp_result.solution, squeeze_pose_qpos.unsqueeze(-2)], dim=-2
        )  # [B, N, 3, 23]  — B==1 env, N==seed_num
        all_hand_pose_qpos = all_hand_pose_qpos.reshape(
            (-1,) + all_hand_pose_qpos.shape[2:]
        )  # [N, 3, 23]
        break  # single scene

    # Home = retract config from the arm robot_cfg (6 arm + 16 hand)
    # robot_configs_path is sibling of manip/ (manip/../robot/<file>)
    from curobo.util_file import get_robot_configs_path
    arm_robot_cfg = load_yaml(
        join_path(get_robot_configs_path(), manip_config_data["robot_file_with_arm"])
    )
    home_22 = torch.tensor(
        arm_robot_cfg["robot_cfg"]["kinematics"]["cspace"]["retract_config"],
        device=tensor_args.device,
        dtype=torch.float32,
    )
    home_arm = home_22[:6]
    home_hand_open = home_22[6:]

    # Pre-grasp / grasp poses from floating-hand root
    pregrasp_pose_qpos = all_hand_pose_qpos[:, 0, :]
    grasp_pose_qpos = all_hand_pose_qpos[:, 1, :]
    pregrasp_hand = all_hand_pose_qpos[:, 0, 7:]
    grasp_hand = all_hand_pose_qpos[:, 1, 7:]
    squeeze_hand = all_hand_pose_qpos[:, 2, 7:]

    # IK for pregrasp palm pose
    palm_goal_pre = Pose(
        position=pregrasp_pose_qpos[:, :3].contiguous(),
        quaternion=pregrasp_pose_qpos[:, 3:7].contiguous(),
    )
    home_arm_b = home_arm.view(1, 6).expand(all_hand_pose_qpos.shape[0], 6).contiguous()
    pregrasp_ik = grasp_solver.ik_solver.solve_batch(
        palm_goal_pre,
        seed_config=home_arm_b.unsqueeze(1),
        retract_config=home_arm_b,
    )
    pregrasp_arm_qpos = pregrasp_ik.solution.squeeze(1)  # [N, 6]
    log_warn(
        f"IK for Pre-grasp Pose: {pregrasp_ik.success.int().sum()}/{len(pregrasp_ik.success)}"
    )

    # IK for grasp palm pose
    palm_goal_grasp = Pose(
        position=grasp_pose_qpos[:, :3].contiguous(),
        quaternion=grasp_pose_qpos[:, 3:7].contiguous(),
    )
    grasp_ik = grasp_solver.ik_solver.solve_batch(
        palm_goal_grasp,
        seed_config=pregrasp_arm_qpos.unsqueeze(1),
        retract_config=pregrasp_arm_qpos,
    )
    grasp_arm_qpos = grasp_ik.solution.squeeze(1)  # [N, 6]
    log_warn(
        f"IK for Grasp Pose Success rate: {grasp_ik.success.int().sum()}/{len(grasp_ik.success)}"
    )

    # IK for lift: straight up along world +Z from the grasp palm pose, keeping
    # the grasp orientation, so the object is lifted CLEAR of the table before the
    # transit begins. `post_grasp_lift_z` (default 0.10) is the height; raising it
    # gives more clearance but costs IK feasibility (lift_ik.success is part of the
    # `feasible` mask below), so it is tunable per object rather than hardcoded.
    lift_pos = palm_goal_grasp.position.clone()
    lift_pos[..., -1] += float(post_grasp_lift_z)
    palm_goal_lift = Pose(position=lift_pos, quaternion=palm_goal_grasp.quaternion)
    lift_ik = grasp_solver.ik_solver.solve_batch(
        palm_goal_lift,
        seed_config=grasp_arm_qpos.unsqueeze(1),
        retract_config=grasp_arm_qpos,
    )
    lift_arm_qpos = lift_ik.solution.squeeze(1)  # [N, 6]
    log_warn(
        f"Lift IK Success rate: {lift_ik.success.int().sum()}/{len(lift_ik.success)}"
    )

    # Pick best seed: all three IK stages must succeed; among those pick
    # lowest grasp_error.
    feasible = (
        pregrasp_ik.success.view(-1) & grasp_ik.success.view(-1) & lift_ik.success.view(-1)
    ).cpu().numpy()  # [N]

    # ORIENTATION FILTER — for side grasps we PREFER the thumb pointing up
    # (so fingers don't scrape the table). BODex's L-BFGS optimizer can drift
    # from the seeded thumb-up orientation, and a drifted seed often makes
    # the fingers collide with the table.
    #
    # We apply the filter SOFTLY: if it eliminates everything, fall back to
    # the unfiltered set rather than failing — relying on cuRobo's collision
    # check (table is in the world) to reject seeds that actually scrape.
    # `thumb_up_min` is callable via kwarg so retries can relax it (the
    # active-arm IK feasibility shrinks with retries on hard objects, and
    # any feasible seed is better than a RuntimeError).
    is_side_grasp = approach == "side" or approach.startswith("side")
    if is_side_grasp and thumb_up_min > 0.0:
        thumb_dirs_world = _palm_y_world_from_quat_wxyz(
            grasp_pose_qpos[:, 3:7].contiguous()
        )  # [N, 3]
        thumb_up_cos = thumb_dirs_world[:, 2].cpu().numpy()     # [N]
        thumb_ok = thumb_up_cos > thumb_up_min
        n_before = int(feasible.sum())
        filtered = feasible & thumb_ok
        if int(filtered.sum()) == 0 and n_before > 0 and not thumb_up_strict:
            # SOFT (default): if nothing passes, keep the unfiltered set and let
            # cuRobo's table-collision check reject seeds that actually scrape.
            print(
                f"[side-grasp orientation filter] all {n_before} feasible seeds "
                f"failed thumb-up check (best={thumb_up_cos.max():.3f}); "
                f"keeping unfiltered set as fallback",
                flush=True,
            )
            # leave `feasible` unchanged — accept any IK-feasible seed
        else:
            # STRICT (thumb_up_strict): keep ONLY thumb-up seeds even if that
            # leaves zero — the caller then gets feasible_count=0 (no grasp)
            # rather than a thumb-down grasp that drives the thumb into the table.
            feasible = filtered
            print(
                f"[side-grasp orientation filter] "
                f"{'STRICT ' if thumb_up_strict else ''}kept "
                f"{int(feasible.sum())}/{n_before} feasible seeds "
                f"(thumb_up_cos > {thumb_up_min}; best={thumb_up_cos.max():.3f})",
                flush=True,
            )

    grasp_err_per_seed = grasp_result.grasp_error.max(dim=-1)[0][0].cpu().numpy()  # [N] worst-contact err
    # Feasibility is a HARD gate; among feasible seeds we rank by GRASP SECURITY
    # rather than lowest optimizer residual. The old argmin(worst-contact error)
    # favored minimal pinch grasps — two tight fingertip contacts beat a 4-finger
    # wrap with one slightly-looser contact. The security score instead rewards
    # MORE fingers in solid force-closure contact (per-contact err <
    # GRASP_ENGAGE_THRESH), tiebroken by lower mean contact error. This is the
    # "best/most secure grasp, not just one part" selection the task asks for.
    if not feasible.any():
        raise RuntimeError(
            f"No seed has all three IK stages (pregrasp+grasp+lift) feasible. "
            f"pregrasp_ok={pregrasp_ik.success.sum().item()}  "
            f"grasp_ok={grasp_ik.success.sum().item()}  "
            f"lift_ok={lift_ik.success.sum().item()}"
        )
    per_contact_err = grasp_result.grasp_error[0].cpu().numpy()        # [N, n_contacts]
    n_contacts = per_contact_err.shape[1]
    n_engaged = (per_contact_err < GRASP_ENGAGE_THRESH).sum(axis=1).astype(float)  # [N]
    mean_err = per_contact_err.mean(axis=1)                            # [N]
    # Integer #engaged dominates the score; mean_err (clipped <1) is the
    # fractional tiebreak so a wrap with more engaged fingers always wins, and
    # among equal-engagement seeds the tighter mean closure wins.
    security = n_engaged - np.clip(mean_err, 0.0, 0.999)
    # Fix A: gentle home-bias — see HOME_DIST_PENALTY. Uses the GRASP arm IK
    # (the pose actually held while closing); distances are per-seed, so for a
    # side approach (every candidate similarly far from home) the term is a
    # near-constant offset that leaves the side ranking intact.
    if HOME_DIST_PENALTY > 0.0:
        home_dist = torch.linalg.norm(
            grasp_arm_qpos - home_arm.unsqueeze(0), dim=-1).cpu().numpy()  # [N] rad
        security = security - HOME_DIST_PENALTY * home_dist
    # grasp_body_bias (2026-08-27, per-object opt-in): bias selection along the
    # OBJECT's long axis — mesh +Y rotated into world by the DETECTED quat,
    # projected horizontal. Positive = palm toward the mesh +Y end (whale: the
    # HEAD; the tail flukes extend toward -Y). Units are score per meter of
    # palm offset from the object center, so 5.0 over the whale's ±8 cm body
    # is up to ±0.4 — a strong tiebreak that still loses to one extra engaged
    # finger. NOTE: only meaningful when the published yaw is TRUSTED
    # (re-detect in the FP viewer until the axes lie along the body).
    _align_telemetry = None   # (per-seed |cos| array, body-axis world dir)
    if (abs(float(grasp_body_bias)) > 1e-9
            or abs(float(grasp_axis_align)) > 1e-9
            or abs(float(grasp_center_bias)) > 1e-9):
        _qw, _qx, _qy, _qz = [float(v) for v in object_pose_world[3:7]]
        _baxis = np.array([2 * (_qx * _qy - _qw * _qz),
                           1 - 2 * (_qx * _qx + _qz * _qz),
                           2 * (_qy * _qz + _qw * _qx)])   # mesh +Y in world
        _baxis[2] = 0.0
        _bn = float(np.linalg.norm(_baxis))
        if _bn > 1e-6:
            _baxis /= _bn
            if (abs(float(grasp_body_bias)) > 1e-9
                    or abs(float(grasp_center_bias)) > 1e-9):
                _palm_xy = grasp_pose_qpos[:, :2].cpu().numpy()
                _bproj = ((_palm_xy - np.asarray(object_pose_world[:2],
                                                 dtype=float)[None, :])
                          @ _baxis[:2])                              # [N] m
                if abs(float(grasp_body_bias)) > 1e-9:
                    security = security + float(grasp_body_bias) * _bproj
                    log_warn(f"[grasp_body_bias] {float(grasp_body_bias):+.1f}/m "
                             f"along world dir "
                             f"({_baxis[0]:+.2f},{_baxis[1]:+.2f}) "
                             f"(mesh +Y = head end)")
                # grasp_center_bias: penalize palm offset from the object
                # CENTER along the body axis — targets the fat mid-body and is
                # IMMUNE to 180-degree yaw flips of the FP lock (which invert
                # grasp_body_bias's notion of "head"). The whale's graspable
                # mass straddles the center; both tips are thin.
                if abs(float(grasp_center_bias)) > 1e-9:
                    security = security - float(grasp_center_bias) * np.abs(_bproj)
                    log_warn(f"[grasp_center_bias] -{float(grasp_center_bias):.1f}/m "
                             f"x |palm offset from center along body axis| "
                             f"(flip-immune mid-body preference)")
            # grasp_axis_align (2026-08-27): prefer hands whose KNUCKLE LINE
            # (palm-Y, the axis the fingers curl about) is PARALLEL to the
            # object's long axis — the fingers then distribute ALONG the body
            # and each wraps ACROSS the narrow width (whale: 13.5cm, fits the
            # aperture) = maximum contact surface. |cos| in [0,1] x weight;
            # sized as a tiebreak below one engaged finger. Trusts the FP yaw,
            # like grasp_body_bias.
            if abs(float(grasp_axis_align)) > 1e-9:
                _py = _palm_y_world_from_quat_wxyz(
                    grasp_pose_qpos[:, 3:7].contiguous())[:, :2]
                _pyn = _py / _py.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                _align = np.abs(_pyn.cpu().numpy() @ _baxis[:2])     # [N] 0..1
                _align_telemetry = (_align, _baxis.copy())   # for best-seed log
                security = security + float(grasp_axis_align) * _align
                log_warn(f"[grasp_axis_align] +{float(grasp_axis_align):.2f} x "
                         f"|cos(knuckle-line, body-axis)| "
                         f"(long-axis wrap preference)")

    # TOP grasps: prioritize a LEVEL (horizontal) hand. BODex's optimizer tends
    # to cock the palm toward the table to engage one extra finger; that tilt
    # makes the fingers reach toward the table (awkward grasp + release
    # collision). Gate to seeds whose palm face is within ~32° of straight-down
    # (TOP_LEVEL_MIN), then rank those by grip security. Side grasps are
    # unaffected. If nothing is level enough, fall back to the MOST level
    # feasible seed (still prioritizing horizontal) rather than failing.
    level_note = ""
    if approach == "top":
        upright = (-_palm_x_world_from_quat_wxyz(
            grasp_pose_qpos[:, 3:7].contiguous())[:, 2]).cpu().numpy()  # [N], 1=level
        _lvl_min = (float(top_level_min) if top_level_min is not None
                    else TOP_LEVEL_MIN)
        level_ok = feasible & (upright >= _lvl_min)
        # TOP HEADING GATE (2026-08-27, opt-in via top_front_min > 0): among
        # LEVEL candidates, keep only hands whose fingers point FRONT (palm-Z
        # projected to the horizontal plane, cos vs world +Y >= top_front_min)
        # — rejects grasps where the hand comes in yawed sideways. If no level
        # candidate is frontal enough, fall back to the MOST frontal level one
        # (never fail outright on heading alone).
        front_fallback = False
        heading = None
        if float(top_front_min) > 0.0 and level_ok.any():
            _pz = _palm_z_world_from_quat_wxyz(
                grasp_pose_qpos[:, 3:7].contiguous())
            _h = _pz[:, :2]
            heading = (_h[:, 1] / _h.norm(dim=-1).clamp_min(1e-6)).cpu().numpy()
            front_ok = level_ok & (heading >= float(top_front_min))
            if front_ok.any():
                level_ok = front_ok
                level_note += (f" front_gate={int(front_ok.sum())}"
                               f"@>={float(top_front_min):.2f}")
            else:
                front_fallback = True
                level_note += " front_gate=NONE->most-frontal"
        if not level_ok.any():
            sec_score = np.where(feasible, upright, -np.inf)  # most-level fallback
            level_note = " level_gate=NONE->most-level" + level_note
        elif front_fallback:
            # Rank level candidates purely by how frontal they are (mirrors the
            # most-level fallback style): best-available heading, still level.
            sec_score = np.where(level_ok, heading, -np.inf)
        else:
            # grasp_z_bonus now also applies to TOP grasps (2026-08-27): on a
            # LOW flat object (whale dome, 8cm) the fingers wrapping the lower
            # hemisphere drive their tips into the table during the squeeze
            # overshoot — prefer candidates whose palm sits HIGHER so the wrap
            # engages the upper hemisphere and the tips stay off the table.
            _top_score = security
            if abs(float(grasp_z_bonus)) > 1e-9:
                _gz_top = grasp_pose_qpos[:, 2].cpu().numpy()
                _top_score = security + float(grasp_z_bonus) * _gz_top
            sec_score = np.where(level_ok, _top_score, -np.inf)
            best_level = int(np.argmax(sec_score))
            level_worst = float(grasp_err_per_seed[best_level])
            level_note = (f" level_gate={int(level_ok.sum())}/{int(feasible.sum())}"
                          f"@>={_lvl_min:.2f} worst={level_worst:.3f}") + level_note
            # Hard-object fallback: the best LEVEL grasp is poor (worst contact
            # not engaged) → look at ALL feasible seeds (tilt allowed) and switch
            # only if the best one is meaningfully tighter. top_level_relax=False
            # disables this escape hatch per request (never accept a tilted hand).
            if level_worst > TOP_LEVEL_RELAX_TRIGGER and bool(top_level_relax):
                full_score = np.where(feasible, security, -np.inf)
                best_full = int(np.argmax(full_score))
                full_worst = float(grasp_err_per_seed[best_full])
                # Switch ONLY to a genuinely good tilted grasp (absolute accept)
                # that is also meaningfully tighter than the level one.
                if (full_worst < TOP_LEVEL_RELAX_ACCEPT
                        and full_worst < level_worst - TOP_LEVEL_RELAX_MARGIN):
                    sec_score = full_score
                    level_note += (f" -> RELAXED tilt (worst {level_worst:.3f}"
                                   f"->{full_worst:.3f}, upright={upright[best_full]:.2f})")
                else:
                    level_note += " (no good tilted grasp)"
    elif approach == "side" or approach.startswith("side"):
        # SIDE: keep the hand FORWARD-facing (palm face at/below horizontal so the
        # fingers wrap DOWN onto the object, the trial040 look) instead of cocking
        # UP (fingers point up — "the hand looks up"). Gate by face_z, then rank by
        # grip security; if none pass, fall back to the most forward/down seed.
        face_z = _palm_x_world_from_quat_wxyz(
            grasp_pose_qpos[:, 3:7].contiguous())[:, 2].cpu().numpy()  # [N]
        grasp_z = grasp_pose_qpos[:, 2].cpu().numpy()  # palm world height [N]
        # TWO-SIDED gate (2026-09-03). face_z = palm press-axis z: 0 = a LEVEL
        # side palm, negative = the palm cocked DOWN (fingers wrapping down and
        # the wrist/forearm dropping toward the table). The upper bound alone
        # rejected only palms cocking UP, so ANY amount of downward tilt passed
        # — that is what leaves the arm links near the table on the tall
        # containers. `side_face_z_min` adds the lower bound: -1.0 keeps the old
        # behaviour (no floor), e.g. -0.25 demands a near-level palm
        # (|tilt| <~ 15 deg). As before, if NOTHING passes we fall back to the
        # most-level seed, so a tighter gate can never fail synthesis.
        fwd_ok = feasible & (face_z <= SIDE_FORWARD_FACE_MAX) \
                          & (face_z >= float(side_face_z_min))
        if not fwd_ok.any():
            # most LEVEL (closest to face_z = 0) rather than most-down
            sec_score = np.where(feasible, -np.abs(face_z), -np.inf)
            level_note = " side_fwd=NONE->most-level"
        else:
            # Rank by grip security PLUS a height bias (prefer higher grasps so the
            # fingers clear the table — see SIDE_GRASP_Z_BONUS). A full extra engaged
            # contact still dominates the height term.
            sec_score = np.where(
                fwd_ok, security + float(grasp_z_bonus) * grasp_z, -np.inf)
            level_note = (f" side_fwd={int(fwd_ok.sum())}/{int(feasible.sum())}"
                          f"@{float(side_face_z_min):.2f}<=face_z<="
                          f"{SIDE_FORWARD_FACE_MAX:.2f}+zbias({grasp_z_bonus:g})")
    else:
        sec_score = np.where(feasible, security, -np.inf)
    feasible_count = int(feasible.sum())
    feasible_total = int(len(feasible))
    N_APPROACH = 30

    # APPROACH COLLISION GUARD: the arm IK self-collision-checks only the CLOSED
    # grasp pose, and the approach is built by interpolation that is not collision-
    # checked, so the OPEN approach hand can sweep its fingers into the arm or dip
    # below the table. Build the approach PER grasp seed (highest security first) and
    # take the FIRST whose OPEN-hand approach is collision-free (self: fingers vs arm;
    # world: table). Grasp quality is preserved — we still take a top-ranked grasp;
    # we only skip ones whose approach would collide. Falls back to the best grasp if
    # none is clean (logged, so it is visible rather than silently executed).
    def _build_approach(seed):
        pb = pregrasp_arm_qpos[seed]
        gb = grasp_arm_qpos[seed]
        ph = pregrasp_hand[seed]
        # ---- Pre-grasp STANDOFF + straight-line approach along the grasp axis ----
        # Instead of a curved home->pregrasp joint lerp (which can sweep the fingers
        # THROUGH an object slightly under/behind the palm, esp. for side grasps),
        # approach the grasp by moving STRAIGHT IN along the palm +X axis (the press
        # direction) from a standoff OUTSIDE the object. Top grasps come straight
        # down; side grasps come straight in from the side. Falls back to the plain
        # joint lerp if the standoff / Cartesian-step IK is infeasible.
        approach_traj = None
        if standoff_dist and standoff_dist > 0.0 and standoff_n > 0:
            batch = pregrasp_arm_qpos.shape[0]
            grasp_palm_pos = grasp_pose_qpos[seed, :3]
            grasp_quat = grasp_pose_qpos[seed, 3:7]
            palm_x = _palm_x_world_from_quat_wxyz(grasp_quat.view(1, 4))[0]   # approach axis
            standoff_pos = grasp_palm_pos - float(standoff_dist) * palm_x
            # The straight-in ends `grasp_extra_in` PAST the grasp pose along the press
            # axis, so the palm seats a bit deeper on the object (the BODex grasp pose
            # can stop just short of a firm contact). Small (a few cm) — too much pushes
            # the object.
            grasp_in_pos = grasp_palm_pos + float(grasp_extra_in) * palm_x
            # Tolerance for accepting an APPROACH waypoint's IK (standoff / outer / the
            # straight-in samples). These are pre-grasp positions, not the final grasp
            # (gb is exact and appended separately), so ~2cm is plenty — and 1cm was
            # strict enough that marginal near-level side grasps missed by ~1mm and
            # collapsed the whole horizontal approach to the colliding raw joint lerp.
            ik_tol = 0.02

            def _bcast_ik(pos3, seed_arm6):
                """IK one palm target (fixed grasp orientation), broadcast to the
                batch size to respect cuRobo's CUDA-graph batch lock; pick [0]."""
                pos_b = pos3.view(1, 3).expand(batch, 3).contiguous()
                quat_b = grasp_quat.view(1, 4).expand(batch, 4).contiguous()
                seed_b = seed_arm6.view(1, 6).expand(batch, 6).contiguous()
                res = grasp_solver.ik_solver.solve_batch(
                    Pose(position=pos_b, quaternion=quat_b),
                    seed_config=seed_b.unsqueeze(1), retract_config=seed_b)
                return (res.solution.view(batch, -1, 6)[0, 0],
                        float(res.position_error.view(-1)[0].item()))

            def _cart_line(p0, p1, n, seed_arm6):
                """Cartesian straight line p0->p1 over n steps (fixed grasp
                orientation), IK each seeded from the previous. Returns [n] arm
                configs, or None if any step is IK-infeasible."""
                arms, s = [], seed_arm6
                for k in range(1, n + 1):
                    a = k / n
                    arm_k, err_k = _bcast_ik(p0 * (1.0 - a) + p1 * a, s)
                    if err_k >= ik_tol:
                        return None
                    arms.append(arm_k)
                    s = arm_k
                return arms

            def _lerp_arms(a6, b6, n):
                """Joint-space lerp a6->b6 over n steps (a excluded, b included), to
                match _cart_line's sampling. Used as a per-segment fallback when the
                Cartesian IK can't solve a straight line but BOTH endpoints are
                IK-feasible — so the approach still routes through the outside
                via-points instead of collapsing to a home->pregrasp sweep."""
                return [a6 * (1.0 - k / n) + b6 * (k / n) for k in range(1, n + 1)]

            # ALL side grasps use the UP-OUT-DOWN-IN motion (user spec). Top grasps skip
            # this and come straight down (their standoff is already above the object).
            is_side = bool(approach) and approach.startswith("side")
            if is_side:
                # Horizontal approach axis = palm press axis flattened to z=0, so the
                # OUTSIDE point sits beside the object AT GRASP HEIGHT and the final
                # straight-in is purely horizontal.
                palm_x_h = palm_x.clone()
                palm_x_h[2] = 0.0
                _hn = float(torch.linalg.norm(palm_x_h).item())
                if _hn > 1e-3:
                    palm_x_h = palm_x_h / _hn
                else:
                    palm_x_h = palm_x                # degenerate (near-vertical); rare
                # OUTSIDE point (x_out, y_out, z_out): horizontally backed off from the
                # grasp by standoff_dist, at grasp height. Descent happens here, clear of
                # the object footprint.
                out_pos = grasp_palm_pos - float(standoff_dist) * palm_x_h
                # HIGH via-point: straight above the outside point and OVER the object top
                # (z_out + offset), so both reaching it and the descent stay clear of the
                # object — never crossing/through it.
                half_z = (float(obb_z_world) * 0.5) if obb_z_world else 0.09
                z_clear = min(max(half_z + float(SIDE_OVER_CLEAR), 0.10), 0.30)
                up = torch.tensor([0.0, 0.0, z_clear],
                                  device=tensor_args.device, dtype=torch.float32)
                high_pos = out_pos + up
                # Seed every approach-IK from the GRASP config gb (same orientation, a
                # few cm away — the nearest known-good solution; pb is an open/parked
                # pose that made the IK miss and collapse to the colliding raw lerp).
                high_arm, high_err = _bcast_ik(high_pos, gb)
                out_arm, out_err = _bcast_ik(
                    out_pos, high_arm if high_err < ik_tol else gb)
                n_in = min(int(standoff_n), N_APPROACH - 6)
                n_down = max(4, n_in // 2)
                n_home = N_APPROACH - n_in - n_down
                # As long as the high + outside via-points are IK-feasible we ALWAYS route
                # through them: home -> high (over the object) -> descend straight DOWN to
                # the outside point -> horizontal straight-IN to the grasp. Each leg
                # prefers a clean Cartesian line, else a joint-lerp BETWEEN the via-points
                # (never a home->pregrasp sweep that crosses the object).
                # Fallback endpoint for the straight-in lerp: with grasp_extra_in
                # set, the lerp must end at the PRESSED-IN pose (like the
                # cartesian path and the grasp waypoint do) — ending at gb made
                # the fallback path jump by grasp_extra_in right at the close.
                gb_end = gb
                if grasp_extra_in and float(grasp_extra_in) > 0.0:
                    _gbi, _gbe = _bcast_ik(grasp_in_pos, gb)
                    if _gbe < ik_tol:
                        gb_end = _gbi
                if high_err < ik_tol and out_err < ik_tol and n_home >= 2:
                    down_cart = _cart_line(high_pos, out_pos, n_down, high_arm)
                    down = down_cart or _lerp_arms(high_arm, out_arm, n_down)
                    inn_cart = _cart_line(out_pos, grasp_in_pos, n_in, down[-1])
                    inn = inn_cart or _lerp_arms(out_arm, gb_end, n_in)
                    ah = torch.linspace(
                        0.0, 1.0, n_home, device=tensor_args.device,
                        dtype=torch.float32).view(-1, 1)
                    home_to_high = (home_arm.unsqueeze(0) * (1 - ah)
                                    + high_arm.unsqueeze(0) * ah)
                    approach_arm = torch.cat(
                        [home_to_high, torch.stack(down, dim=0),
                         torch.stack(inn, dim=0)], dim=0)
                    approach_hand = ph.unsqueeze(0).expand(N_APPROACH, -1)
                    approach_traj = torch.cat([approach_arm, approach_hand], dim=-1)
                    mode = ("cartesian" if (down_cart is not None
                                            and inn_cart is not None) else "lerp-fallback")
                    log_warn(
                        f"Side approach UP-OUT-DOWN-IN (high {z_clear * 100:.0f}cm over "
                        f"obj, out {float(standoff_dist) * 100:.0f}cm): {n_home} home->high "
                        f"+ {n_down} descend + {n_in} straight-in ({mode})")
                if approach_traj is None:
                    log_warn("Side UP-OUT-DOWN-IN: via-points IK-infeasible — trying the "
                             "plain standoff straight-in next.")
            if approach_traj is None:
                # Standoff straight-in. Primary path for TOP grasps (standoff is ABOVE
                # the object -> come straight down) and STEEP side grasps (standoff is
                # up-and-out -> come down-and-in along the grasp axis, the trial040 look
                # that already worked). ALSO the graceful fallback for a near-level side
                # grasp whose outer via-point was IK-infeasible: home->standoff (standoff
                # is OUTSIDE the object) + a horizontal/axial straight-in — still far
                # better than the raw home->pregrasp sweep that crosses the object.
                standoff_arm, soff_err = _bcast_ik(standoff_pos, gb)  # seed from grasp
                n_cart = min(int(standoff_n), N_APPROACH - 2)
                n_home = N_APPROACH - n_cart
                if soff_err < ik_tol:
                    inn = _cart_line(standoff_pos, grasp_in_pos, n_cart, standoff_arm)
                    if inn is not None:
                        ah = torch.linspace(
                            0.0, 1.0, n_home, device=tensor_args.device,
                            dtype=torch.float32).view(-1, 1)
                        home_to_standoff = (home_arm.unsqueeze(0) * (1 - ah)
                                            + standoff_arm.unsqueeze(0) * ah)
                        approach_arm = torch.cat(
                            [home_to_standoff, torch.stack(inn, dim=0)], dim=0)
                        approach_hand = ph.unsqueeze(0).expand(N_APPROACH, -1)
                        approach_traj = torch.cat([approach_arm, approach_hand], dim=-1)
                        log_warn(
                            f"Pre-grasp standoff {float(standoff_dist) * 100:.0f}cm "
                            f"along palm-x + straight approach: {n_home} home->standoff "
                            f"+ {n_cart} cartesian wps")
                if approach_traj is None:
                    log_warn(f"Standoff approach IK err={soff_err:.4f} — "
                             "falling back to the joint-lerp approach.")

        if approach_traj is None:
            alphas = torch.linspace(
                0.0, 1.0, N_APPROACH, device=tensor_args.device, dtype=torch.float32
            ).view(-1, 1)
            approach_arm = home_arm.unsqueeze(0) * (1 - alphas) + pb.unsqueeze(0) * alphas  # [30,6]
            approach_hand = ph.unsqueeze(0).expand(N_APPROACH, -1)                           # [30,16]
            approach_traj = torch.cat([approach_arm, approach_hand], dim=-1)                 # [30,22]
        return approach_traj

    _rw = _get_approach_robot_world(
        manip_config_data.get("robot_file_with_arm"), table_z, tensor_args)
    ranked = [int(s) for s in np.argsort(sec_score)[::-1] if np.isfinite(sec_score[s])]
    if not ranked:
        ranked = [int(np.argmax(sec_score))]
    best_seed = ranked[0]
    approach_traj = None
    _fallback_traj = None
    _fallback_seed = None
    _guard_note = ""

    def _post_grasp_poses(cand):
        """The three UNGUARDED poses that follow the approach: GRASP, SQUEEZE and
        LIFT, as a [3, 22] trajectory the collision checker can FK.

        pick_traj = [approach | grasp | squeeze | lift] (see the cat below), but
        historically only the approach was checked — so a candidate whose SQUEEZE
        drove a wrist link into the table was driven anyway. That is the pose that
        strikes the table on LOW objects (whale plans 8.1cm tall: the palm sits
        just above the table and the squeeze curls the fingers down past it)."""
        return torch.stack([
            torch.cat([grasp_arm_qpos[cand], grasp_hand[cand]]),
            torch.cat([grasp_arm_qpos[cand], squeeze_hand[cand]]),
            torch.cat([lift_arm_qpos[cand], squeeze_hand[cand]]),
        ], dim=0)

    for _ri, _cand in enumerate(ranked[:APPROACH_COLL_TRIES]):
        _cand_traj = _build_approach(_cand)
        _hit, _why = _approach_collision(_rw, _cand_traj)
        if _hit:
            log_warn(f"[approach-collision] grasp seed {_cand} (rank {_ri}): OPEN-hand "
                     f"approach hits {_why} -> trying the next-best grasp.")
            continue
        # The approach is clean. Remember it as the fallback (this is exactly
        # what the old approach-only guard would have driven), then ALSO check
        # the grasp/squeeze/lift poses.
        if _fallback_traj is None:
            _fallback_traj, _fallback_seed = _cand_traj, _cand
        _hit2, _why2 = _approach_collision(_rw, _post_grasp_poses(_cand))
        if not _hit2:
            best_seed, approach_traj = _cand, _cand_traj
            _guard_note = " approach_ok" + (f"@rank{_ri}" if _ri else "")
            break
        log_warn(f"[post-grasp-collision] grasp seed {_cand} (rank {_ri}): the "
                 f"GRASP/SQUEEZE/LIFT pose hits {_why2} -> trying the next-best "
                 f"grasp.")
    if approach_traj is None and _fallback_traj is not None:
        # Every candidate with a clean approach had a colliding grasp/squeeze/
        # lift. Fall back to the best clean-approach candidate = EXACTLY the
        # pre-2026-09-02 behaviour, so this stricter check can only improve the
        # selection and never causes a synthesis failure that did not exist
        # before. The warning tells the operator to watch that trial.
        best_seed, approach_traj = _fallback_seed, _fallback_traj
        _guard_note = " approach_ok_POSTGRASP_COLLIDES"
        log_warn("[post-grasp-collision] no candidate had a clean "
                 "grasp/squeeze/lift; using the best clean-APPROACH grasp "
                 "(pre-09-02 behaviour) — watch the squeeze on hardware.")
    if approach_traj is None:
        # Fix B (2026-08-26): a known-colliding approach must NEVER drive the
        # hardware. Fail the synthesis attempt instead — the runner's best-of-N
        # loop retries with a fresh seed (a different grasp family is usually
        # clean), and only gives up (no motion at all) if every retry stays
        # dirty. BODEX_ALLOW_COLLIDING_APPROACH=1 restores the old drive-anyway
        # behavior for offline debugging.
        if os.environ.get("BODEX_ALLOW_COLLIDING_APPROACH"):
            best_seed, approach_traj = ranked[0], _fallback_traj
            _guard_note = " approach_COLLIDES(all)->best(FORCED-by-env)"
            log_warn("[approach-collision] every candidate approach collided -- "
                     "BODEX_ALLOW_COLLIDING_APPROACH set: driving the "
                     "highest-security grasp anyway (inspect on hardware).")
        else:
            raise RuntimeError(
                f"approach collision guard: the OPEN-hand approach collides for "
                f"every candidate grasp (top {min(len(ranked), APPROACH_COLL_TRIES)} "
                f"tried) — refusing to drive a colliding approach. A retry with a "
                f"fresh seed usually finds a clean configuration.")

    pb = pregrasp_arm_qpos[best_seed]
    gb = grasp_arm_qpos[best_seed]
    lb = lift_arm_qpos[best_seed]
    ph = pregrasp_hand[best_seed]
    gh = grasp_hand[best_seed]
    sh = squeeze_hand[best_seed]

    _fz = float(_palm_x_world_from_quat_wxyz(
        grasp_pose_qpos[best_seed, 3:7].view(1, 4).contiguous())[0, 2].item())
    log_warn(
        f"[bodex_plan_cli] best seed={best_seed} "
        f"engaged={int(n_engaged[best_seed])}/{n_contacts} "
        f"mean_err={mean_err[best_seed]:.4f} worst_err={grasp_err_per_seed[best_seed]:.4f} "
        f"face_z={_fz:+.3f} "
        + (f"align={_align_telemetry[0][best_seed]:.2f} "
           f"bodyaxis=({_align_telemetry[1][0]:+.2f},{_align_telemetry[1][1]:+.2f}) "
           if _align_telemetry is not None else "")
        + f"feasible_count={feasible_count}/{feasible_total} (security-ranked){level_note}{_guard_note}"
    )

    if os.environ.get("BODEX_DEBUG_APPROACH_Z"):
        try:
            _kin = grasp_solver.ik_solver.kinematics
            _ee = _kin.get_state(approach_traj[:, :6].contiguous()).ee_position  # [30,3]
            _z = _ee[:, 2].cpu().numpy()
            _xy = _ee[:, :2].cpu().numpy()
            _gp = grasp_pose_qpos[best_seed, :3].cpu().numpy()
            log_warn(f"[approach-z] grasp_z={_gp[2]:.3f} palm_z over approach: "
                     f"min={_z.min():.3f} max={_z.max():.3f} start={_z[0]:.3f} "
                     f"end={_z[-1]:.3f} | z[::5]={[round(float(v),3) for v in _z[::5]]}")
            log_warn(f"[approach-z] end_xy={_xy[-1].round(3).tolist()} "
                     f"grasp_xy={_gp[:2].round(3).tolist()}")
        except Exception as _e:
            log_warn(f"[approach-z] FK debug failed: {_e}")

    # grasp_extra_in: the approach ends grasp_extra_in PAST the BODex grasp
    # pose, but the grasp/squeeze waypoints were built from gb (the ORIGINAL
    # pose) — so the arm pressed in and then pulled straight back OUT before
    # closing (observed on hardware as an abrupt retreat, hand closing short of
    # the object). Fix: make the pressed-in pose the grasp pose itself — IK the
    # palm at grasp + extra_in (same orientation, seeded from gb) and close
    # THERE. Falls back to gb if the pressed-in pose is IK-infeasible.
    gb_grasp = gb
    if grasp_extra_in and float(grasp_extra_in) > 0.0:
        _batch = pregrasp_arm_qpos.shape[0]
        _g_pos = grasp_pose_qpos[best_seed, :3]
        _g_quat = grasp_pose_qpos[best_seed, 3:7]
        _palm_x_in = _palm_x_world_from_quat_wxyz(_g_quat.view(1, 4))[0]
        _in_pos = _g_pos + float(grasp_extra_in) * _palm_x_in
        _res = grasp_solver.ik_solver.solve_batch(
            Pose(position=_in_pos.view(1, 3).expand(_batch, 3).contiguous(),
                 quaternion=_g_quat.view(1, 4).expand(_batch, 4).contiguous()),
            seed_config=gb.view(1, 6).expand(_batch, 6).contiguous().unsqueeze(1),
            retract_config=gb.view(1, 6).expand(_batch, 6).contiguous())
        _in_err = float(_res.position_error.view(-1)[0].item())
        if _in_err < 0.02:
            gb_grasp = _res.solution.view(_batch, -1, 6)[0, 0]
            log_warn(f"[grasp_extra_in] closing at the pressed-in pose "
                     f"(+{float(grasp_extra_in) * 100:.1f}cm along palm-x, "
                     f"IK err {_in_err * 1000:.1f}mm)")
        else:
            log_warn(f"[grasp_extra_in] pressed-in pose IK err {_in_err:.4f} "
                     f">= 0.02 — closing at the original grasp pose")

    grasp_wp   = torch.cat([gb_grasp, gh]).unsqueeze(0).repeat(hold_frames, 1)
    squeeze_wp = torch.cat([gb_grasp, sh]).unsqueeze(0).repeat(hold_frames, 1)
    # Lift: RAMP the arm grasp->lift over hold_frames knots, instead of a single
    # +0.10 m jump from the squeeze pose to the lift pose in one knot — that
    # jump looked abrupt on hardware (worst when the place goal is near the
    # object, so the following transit is short and the lift dominates). The hand
    # stays in the squeeze the whole ramp. Generalizes the top-only (2c)
    # pre-transit smoothing in bodex_hw/trajectory.py to side grasps too.
    la = torch.linspace(0.0, 1.0, hold_frames, device=tensor_args.device,
                        dtype=torch.float32).view(-1, 1)
    lift_arm_ramp = gb_grasp.unsqueeze(0) * (1.0 - la) + lb.unsqueeze(0) * la  # [H,6]
    lift_wp = torch.cat(
        [lift_arm_ramp, sh.unsqueeze(0).expand(hold_frames, -1)], dim=-1)  # [H,22]

    pick_traj = torch.cat([approach_traj, grasp_wp, squeeze_wp, lift_wp], dim=0)  # [30+3H, 22]
    return (
        pick_traj,
        best_seed,
        float(grasp_err_per_seed[best_seed]),
        grasp_solver,
        pb.clone(),
        lb.clone(),                  # final arm q at end of pick (at lift)
        sh.clone(),                  # hand q at end of pick (squeeze)
        manip_config_data,
        feasible_count,
        feasible_total,
    )


def build_place_trajectory(
    grasp_solver,
    lift_arm_q: torch.Tensor,           # [6] arm joints at end of pick
    squeeze_hand_q: torch.Tensor,       # [16] hand in squeeze
    home_hand_open: torch.Tensor,       # [16] target open hand
    lift_palm_pose: Pose,               # palm pose at lift point
    lift_object_pos: np.ndarray,        # world xyz of OBJECT at end of lift
    place_xyz: np.ndarray,              # world xyz where OBJECT should end up
    transit_n: int,
    descend_n: int,
    release_n: int,
    tensor_args: TensorDeviceType,
    batch_size: int,                    # must match prior solve_batch size (num_seeds)
    descend_extra_z: float = 0.02,      # extra Z above place_z at the descend stop
    release_lift_z: float = 0.08,       # true world +Z lift at the goal BEFORE opening
    transit_hover_z: float = 0.15,      # how high ABOVE the goal the transit passes
    release_lift_n: int = 16,           # waypoints for that lift
):
    """Build a [T_place, 22] trajectory: transit → descend → release-lift → release.

    The palm orientation is kept constant (same as at lift) so the held
    object does not spin while being transported. All IK calls use the
    same batch_size as the earlier pregrasp IK to avoid cuRobo's
    CUDA-graph batch-lock.

    ``release_lift_z`` (default 8 cm): after the hand reaches the goal (descend),
    the arm lifts straight up by this much in world +Z — solved by IK so it's a
    true Cartesian lift — and the fingers open at that LIFTED pose. This keeps
    the hand (and the opening fingers) clear of the table; releasing right at the
    descend pose drove the down-pointing fingers into the table. Set 0 to disable.
    """
    lift_quat = lift_palm_pose.quaternion.view(1, 4).contiguous()

    def _broadcast_ik(pos_xyz: torch.Tensor, quat_wxyz: torch.Tensor,
                      seed_arm_q: torch.Tensor):
        """Solve IK broadcast to batch_size; pick [0] as result."""
        pos_b = pos_xyz.expand(batch_size, 3).contiguous()
        quat_b = quat_wxyz.expand(batch_size, 4).contiguous()
        seed_b = seed_arm_q.view(1, 6).expand(batch_size, 6).contiguous()
        res = grasp_solver.ik_solver.solve_batch(
            Pose(position=pos_b, quaternion=quat_b),
            seed_config=seed_b.unsqueeze(1),
            retract_config=seed_b,
        )
        return res

    # Palm→object offset at the lift moment (in world frame)
    # palm_to_obj = obj_pos_world - palm_pos_world_at_lift
    palm_pos_lift = lift_palm_pose.position.view(3).cpu().numpy()
    palm_to_obj = np.array(lift_object_pos) - palm_pos_lift   # [3]

    # Transit palm target = place_xyz - palm_to_obj + transit_hover_z in world
    # +Z, so the OBJECT hovers that far above the goal while travelling. This is
    # the HIGHEST pose of the whole place phase, so lowering it directly reduces
    # how far the arm has to reach up (and how far the object falls when the hand
    # opens at the hover, as the side grasps do).
    transit_palm_xyz = (np.array(place_xyz) - palm_to_obj
                        + np.array([0, 0, float(transit_hover_z)]))
    target_pos = torch.tensor(
        transit_palm_xyz, device=tensor_args.device, dtype=torch.float32,
    ).view(1, 3)
    transit_ik = _broadcast_ik(target_pos, lift_quat, lift_arm_q)
    log_warn(
        f"Transit IK: success={int(transit_ik.success.view(-1)[0].item())} err={float(transit_ik.position_error.view(-1)[0].item()):.4f}"
    )
    transit_arm_q = transit_ik.solution.view(batch_size, -1, 6)[0, 0]

    # Descend palm so the OBJECT lands `descend_extra_z` above the target.
    # For top grasps (palm above object), 2cm gives the object a clean drop
    # without the palm pushing it into the table. For side grasps (palm
    # beside the object), the palm is at the cylinder's mid-height — the
    # object hangs from the side and its base is `obb_z/2` below the palm.
    # In that case a 2cm gap means the base is still 2.5cm above the table,
    # so the object falls and a slim cylinder will tip. Pass 0.0 (or even
    # slightly negative) for side grasps so the base lands flush.
    descend_palm_xyz = (
        np.array(place_xyz) - palm_to_obj + np.array([0, 0, descend_extra_z])
    )
    descend_pos = torch.tensor(
        descend_palm_xyz, device=tensor_args.device, dtype=torch.float32,
    ).view(1, 3)
    descend_ik = _broadcast_ik(descend_pos, lift_quat, transit_arm_q)
    log_warn(
        f"Descend IK: success={int(descend_ik.success.view(-1)[0].item())} err={float(descend_ik.position_error.view(-1)[0].item()):.4f}"
    )
    descend_arm_q = descend_ik.solution.view(batch_size, -1, 6)[0, 0]

    # Release-lift: after reaching the goal, lift the palm straight up by
    # `release_lift_z` in world +Z (true Cartesian lift via IK) and OPEN the hand
    # there, so the fingers clear the table instead of opening at the drop. Held
    # object orientation is unchanged (same quat). Falls back to the descend pose
    # if the lifted pose is not IK-reachable.
    release_arm_q = descend_arm_q
    lift_ok = False
    if release_lift_z and release_lift_z > 0.0:
        release_lift_palm_xyz = descend_palm_xyz + np.array([0, 0, release_lift_z])
        release_lift_pos = torch.tensor(
            release_lift_palm_xyz, device=tensor_args.device, dtype=torch.float32,
        ).view(1, 3)
        release_lift_ik = _broadcast_ik(release_lift_pos, lift_quat, descend_arm_q)
        # Gate on the actual position error, NOT cuRobo's strict success flag.
        # The descend pose is used unconditionally (its IK flag is ignored), and
        # for marginal place locations cuRobo flags even sub-cm release-lift
        # solutions as failures — yet a few-mm miss still gives a clean,
        # table-clearing lift. So accept the lifted solution whenever it reaches
        # within `lift_pos_tol` of the +Z target (matches how descend is used).
        lift_pos_tol = 0.03
        lift_pos_err = float(release_lift_ik.position_error.view(-1)[0].item())
        lift_ok = lift_pos_err < lift_pos_tol
        log_warn(
            f"Release-lift IK (+{release_lift_z*100:.0f}cm Z): "
            f"success_flag={int(release_lift_ik.success.view(-1)[0].item())} "
            f"err={lift_pos_err:.4f} -> {'lift' if lift_ok else 'open@descend'}"
        )
        if lift_ok:
            release_arm_q = release_lift_ik.solution.view(batch_size, -1, 6)[0, 0]
        else:
            log_warn("Release-lift IK far from target — opening at the descend pose.")

    # Linear interpolate: lift→transit, transit→descend, descend→release-lift, release
    def interp(q0, q1, n):
        alphas = torch.linspace(0.0, 1.0, n, device=q0.device, dtype=q0.dtype).view(-1, 1)
        return q0.unsqueeze(0) * (1 - alphas) + q1.unsqueeze(0) * alphas

    transit_arm = interp(lift_arm_q, transit_arm_q, transit_n)
    transit_hand = squeeze_hand_q.unsqueeze(0).expand(transit_n, -1)
    transit_traj = torch.cat([transit_arm, transit_hand], dim=-1)

    descend_arm = interp(transit_arm_q, descend_arm_q, descend_n)
    descend_hand = squeeze_hand_q.unsqueeze(0).expand(descend_n, -1)
    descend_traj = torch.cat([descend_arm, descend_hand], dim=-1)

    # Release-lift segment (hand STAYS squeezed while the arm rises to the goal+Z).
    place_segs = [transit_traj, descend_traj]
    if lift_ok:
        rlift_arm = interp(descend_arm_q, release_arm_q, release_lift_n)
        rlift_hand = squeeze_hand_q.unsqueeze(0).expand(release_lift_n, -1)
        place_segs.append(torch.cat([rlift_arm, rlift_hand], dim=-1))

    # Release: arm stays at the (lifted) release pose, hand transitions to open.
    release_arm = release_arm_q.unsqueeze(0).expand(release_n, -1)
    release_hand = interp(squeeze_hand_q, home_hand_open, release_n)
    place_segs.append(torch.cat([release_arm, release_hand], dim=-1))

    place_traj = torch.cat(place_segs, dim=0)
    return place_traj


def write_runtime_scene_cfg(
    object_id: str,
    object_pose: np.ndarray,
    scene_cfg_root: str,
    processed_data_dir: str,
    table_z: float,
    obj_scale=(1.0, 1.0, 1.0),
):
    """Write scene_cfg/<obj>/runtime/latest.npy with the detected object pose.

    Mirrors `fp_scene_cfg_bridge.write_runtime_scene_cfg` but lives here so
    this CLI is self-contained.
    """
    scene_cfg_dir = os.path.join(scene_cfg_root, object_id, "runtime")
    os.makedirs(scene_cfg_dir, exist_ok=True)
    out_path = os.path.join(scene_cfg_dir, "latest.npy")
    obj_data_dir = os.path.join(processed_data_dir, object_id)
    rel = os.path.relpath(obj_data_dir, scene_cfg_dir)
    entry = {
        "type": "rigid_object",
        "scale": np.asarray(obj_scale, dtype=float),
        "pose": np.asarray(object_pose, dtype=float),
        "file_path": os.path.join(rel, "mesh", "simplified.obj"),
        "urdf_path": os.path.join(rel, "urdf", "coacd.urdf"),
        "info_path": os.path.join(rel, "info", "simplified.json"),
    }
    scene = {
        object_id: entry,
        "table": {
            "type": "plane",
            "pose": np.array([0.0, 0.0, table_z, 1.0, 0.0, 0.0, 0.0]),
            "size": np.array([0.0, 0.0, 1.0]),
        },
    }
    cfg = {
        "scene": scene,
        "scene_id": f"{object_id}/runtime/latest",
        "task": {
            "obj_name": object_id,
            "type": "slide",
            "axis": np.array([0.0, 0.0, 1.0]),
            "distance": 0.1,
        },
    }
    np.save(out_path, cfg, allow_pickle=True)
    return out_path


def _select_manip_cfg(active_arm: str, grasp_mode: str = "wrap") -> str:
    """Map (arm, grasp_mode) to a manip config. 'wrap' = the default 4-finger
    force-closure wrap; 'pinch' = the 2-finger + thumb-backstop config for small
    slippery round objects (tomato/peach). Unknown modes fall back to wrap."""
    base = {"ra": "sim_ra_ur3_allegro", "la": "sim_la_ur3e_allegro"}[active_arm]
    cfg = "pinch_runtime.yml" if grasp_mode == "pinch" else "tabletop_runtime.yml"
    return f"{base}/{cfg}"


def synthesize_full_trajectory(
    active_arm: str,
    object_id: str,
    object_pose: np.ndarray,   # [7] xyz + wxyz
    place_xyz: np.ndarray,     # [3] xyz
    approach: str = "top",
    seed_num: int = 20,
    hold_frames: int = 10,
    transit_n: int = 40,
    descend_n: int = 20,
    release_n: int = 12,
    table_z: float = 0.75,
    scene_cfg_root: str = os.path.join(_CUSTOM_OBJ_DIR, "scene_cfg"),
    processed_data_dir: str = os.path.join(_CUSTOM_OBJ_DIR, "processed_data"),
    passive_arm: str | None = None,
    passive_joint_state: np.ndarray | None = None,
    thumb_up_min: float = 0.65,
    thumb_up_strict: bool = False,
    standoff_dist: float = 0.10,
    grasp_extra_in: float = 0.0,
    grasp_z_bonus: float = SIDE_GRASP_Z_BONUS,
    top_front_min: float = 0.0,
    top_level_relax: bool = True,
    grasp_body_bias: float = 0.0,
    grasp_axis_align: float = 0.0,
    grasp_center_bias: float = 0.0,
    grasp_mode: str = "wrap",
    obj_scale=(1.0, 1.0, 1.0),
    obb_z_world: float | None = None,
    release_lift_z: float = 0.08,
    post_grasp_lift_z: float = 0.10,
    transit_hover_z: float = 0.15,
    side_face_z_min: float = -1.0,
    top_level_min: float | None = None,
    tensor_args: TensorDeviceType | None = None,
) -> dict:
    """End-to-end pick+place trajectory synthesis. Returns a dict with keys:
        trajectory:        [T, 22] float32 numpy array
        pick_end_idx:      int (end of pick phase, rest is transit/descend/release)
        grasp_error:       float (L-BFGS grasp error for the chosen seed)
        object_pose_world: [7] numpy (input, for traceability)
        place_xyz:         [3] numpy
        active_arm:        'ra' | 'la'
        object_id:         str
        approach:          str
        success:           bool

    Raises RuntimeError on infeasible IK. Suitable for calling from a
    persistent worker process — reuses module-level cuRobo imports.
    """
    if tensor_args is None:
        tensor_args = TensorDeviceType()
    object_pose = np.asarray(object_pose, dtype=float)
    place_xyz = np.asarray(place_xyz, dtype=float)

    write_runtime_scene_cfg(
        object_id, object_pose, scene_cfg_root, processed_data_dir,
        table_z=table_z, obj_scale=obj_scale,
    )

    manip_cfg_name = _select_manip_cfg(active_arm, grasp_mode)
    log_warn(f"[grasp_mode={grasp_mode}] using manip config: {manip_cfg_name}")

    os.environ["BODEX_HOLD_FRAMES"] = str(hold_frames)
    (pick_traj, best_seed, grasp_err, grasp_solver,
     pregrasp_arm_q, lift_arm_q, squeeze_hand_q,
     manip_config_data, feasible_count, feasible_total) = synthesize_pick_trajectory(
        manip_cfg_name=manip_cfg_name,
        object_id=object_id,
        object_pose_world=object_pose,
        approach=approach,
        seed_num=seed_num,
        hold_frames=hold_frames,
        tensor_args=tensor_args,
        passive_arm=passive_arm,
        passive_joint_state=passive_joint_state,
        thumb_up_min=thumb_up_min,
        thumb_up_strict=thumb_up_strict,
        standoff_dist=standoff_dist,
        obb_z_world=obb_z_world,
        grasp_extra_in=grasp_extra_in,
        grasp_z_bonus=grasp_z_bonus,
        top_front_min=top_front_min,
        top_level_relax=top_level_relax,
        grasp_body_bias=grasp_body_bias,
        grasp_axis_align=grasp_axis_align,
        post_grasp_lift_z=post_grasp_lift_z,
        side_face_z_min=side_face_z_min,
        top_level_min=top_level_min,
        grasp_center_bias=grasp_center_bias,
        table_z=table_z,
    )

    ik_kin = grasp_solver.ik_solver.kinematics
    fk_state = ik_kin.get_state(lift_arm_q.view(1, 6))
    lift_palm_pose = Pose(
        position=fk_state.ee_position.view(1, 3),
        quaternion=fk_state.ee_quaternion.view(1, 4),
    )

    from curobo.util_file import get_robot_configs_path
    home_hand_open = torch.tensor(
        load_yaml(
            join_path(get_robot_configs_path(), manip_config_data["robot_file_with_arm"])
        )["robot_cfg"]["kinematics"]["cspace"]["retract_config"][6:],
        device=tensor_args.device,
        dtype=torch.float32,
    )
    lift_object_pos = object_pose[:3] + np.array([0, 0, 0.10])
    # Side grasps put the palm at the side of the object (base hangs ~half
    # the object's height below palm-z). Releasing above target leaves the
    # base above the table and slim cylinders tip in that gap. Push the
    # palm slightly BELOW target so the cylinder base presses onto the
    # table and bears its own weight before fingers release. Also slow
    # down the finger release so the spring-back doesn't flick the object.
    is_side = approach.startswith("side") or approach == "side"
    # v8 §4.4 — descend_extra_z scales with object height for top grasps.
    # Tall objects (mug, tumbler) released 2 cm above target bounce on
    # impact + spring-back from squeeze release → tip-over. Tall = lower
    # clearance (down to 5 mm); short stays at 2 cm.
    if is_side:
        # NOTE: tumbler (extreme tall, obb_z 0.25) was tried at -0.015
        # deeper base-press — regressed (palm jams the object/table).
        # Stock -0.005 is the empirical best for side grasps incl. the
        # extreme case.
        descend_extra_z = -0.005
    elif obb_z_world is not None and obb_z_world >= 0.05:
        # Linear: 0.05 m obb → 0.020 ; 0.15 m obb → 0.005. Floored at 5 mm.
        descend_extra_z = max(0.005, 0.020 - 0.150 * (obb_z_world - 0.05))
    else:
        descend_extra_z = 0.02
    # v8 §4.1 — slow release also fires for tall-top grasps, not only side.
    # The mug/tumbler tip-on-release mechanism is the same regardless of
    # approach direction.
    is_tall_top = (not is_side) and (obb_z_world is not None) and (obb_z_world >= 0.10)
    side_release_n = max(release_n, 36) if is_tall_top else (
        max(release_n, 48) if is_side else release_n
    )
    # PLACE CLEARANCE GUARD (2026-08-26, side grasps): carrying with a
    # horizontal palm hangs the WRIST LINKS below the hand, and at the drop
    # pose (side base-press descend_extra_z = -0.005) some IK configurations
    # put a wrist link INTO the table at the goal. FK-check the descend +
    # release tail against the table (same cached RobotWorld as the approach
    # guard; world threshold fires only on actual sphere penetration) and,
    # while it collides, RAISE the descend stop and rebuild — the object drops
    # a little further, the wrist never touches the table. Top grasps are
    # untouched (wrist is above the palm there).
    _rw_place = None
    if is_side:
        _rw_place = _get_approach_robot_world(
            manip_config_data.get("robot_file_with_arm"), table_z, tensor_args)
    _clearance_steps = [0.0, 0.025, 0.045, 0.065] if _rw_place is not None else [0.0]
    place_traj = None
    for _ci, _dz in enumerate(_clearance_steps):
        place_traj = build_place_trajectory(
            grasp_solver=grasp_solver,
            lift_arm_q=lift_arm_q,
            squeeze_hand_q=squeeze_hand_q,
            home_hand_open=home_hand_open,
            lift_palm_pose=lift_palm_pose,
            lift_object_pos=lift_object_pos,
            place_xyz=place_xyz,
            transit_n=transit_n,
            descend_n=descend_n,
            release_n=side_release_n,
            tensor_args=tensor_args,
            batch_size=seed_num,
            descend_extra_z=descend_extra_z + _dz,
            release_lift_z=release_lift_z,
            transit_hover_z=transit_hover_z,
        )
        if _rw_place is None:
            break
        _hit, _why = _approach_collision(_rw_place, place_traj[transit_n:])
        if not _hit:
            if _ci:
                log_warn(f"[place-clearance] raised the drop pose "
                         f"+{_dz * 100:.1f}cm so the wrist clears the table "
                         f"at the goal")
            break
        if _ci == len(_clearance_steps) - 1:
            log_warn(f"[place-clearance] goal pose still hits {_why} at the "
                     f"MAX raise (+{_dz * 100:.1f}cm) — using it anyway "
                     f"(least-bad); inspect the release on hardware")
        else:
            log_warn(f"[place-clearance] goal pose hits {_why} at "
                     f"descend_extra_z={descend_extra_z + _dz:+.3f} — raising")

    full_traj = torch.cat([pick_traj, place_traj], dim=0).cpu().numpy().astype(np.float32)
    return {
        "trajectory": full_traj,
        "pick_end_idx": int(pick_traj.shape[0]),
        # Number of descend waypoints at the tail of the place phase (the
        # `descend_n` knots that sit immediately before the open ramp). The
        # hardware release uses this to find the HOVER pose (end of transit =
        # release_start - descend_n) so it can re-level the hand up high
        # instead of at the low drop pose. Preserved through build_full_trajectory.
        "descend_n": int(descend_n),
        "grasp_error": float(grasp_err),
        "feasible_count": int(feasible_count),
        "feasible_total": int(feasible_total),
        "object_pose_world": object_pose,
        "place_xyz": place_xyz,
        "active_arm": active_arm,
        "object_id": object_id,
        "approach": approach,
        "success": True,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--active_arm", choices=["ra", "la"], required=True)
    p.add_argument("--object_id", required=True)
    p.add_argument("--object_pose", type=float, nargs=7, required=True,
                   help="x y z qw qx qy qz in world frame")
    p.add_argument("--place_xyz", type=float, nargs=3, required=True,
                   help="x y z of place target (palm target ends 2cm above this z)")
    p.add_argument("--approach", choices=list(APPROACH_PRESETS.keys()), default="top")
    p.add_argument("--seed_num", type=int, default=20)
    p.add_argument("--hold_frames", type=int, default=10)
    p.add_argument("--transit_n", type=int, default=40)
    p.add_argument("--descend_n", type=int, default=20)
    p.add_argument("--release_n", type=int, default=12)
    p.add_argument("--table_z", type=float, default=0.75)
    p.add_argument("--out_npy", required=True)
    p.add_argument("--scene_cfg_root",
                   default=os.path.join(_CUSTOM_OBJ_DIR, "scene_cfg"))
    p.add_argument("--processed_data_dir",
                   default=os.path.join(_CUSTOM_OBJ_DIR, "processed_data"))
    args = p.parse_args()

    setup_logger("warn")

    try:
        result = synthesize_full_trajectory(
            active_arm=args.active_arm,
            object_id=args.object_id,
            object_pose=np.array(args.object_pose, dtype=float),
            place_xyz=np.array(args.place_xyz, dtype=float),
            approach=args.approach,
            seed_num=args.seed_num,
            hold_frames=args.hold_frames,
            transit_n=args.transit_n,
            descend_n=args.descend_n,
            release_n=args.release_n,
            table_z=args.table_z,
            scene_cfg_root=args.scene_cfg_root,
            processed_data_dir=args.processed_data_dir,
        )
    except Exception as e:
        log_warn(f"[bodex_plan_cli] synthesis FAILED: {e}")
        np.save(args.out_npy, {
            "success": False,
            "error": str(e),
        }, allow_pickle=True)
        return 2

    # Shape the dict for the CLI's on-disk format (include 'arm' key that
    # env.py looks for)
    out = dict(result)
    out["arm"] = out["active_arm"]
    np.save(args.out_npy, out, allow_pickle=True)
    log_warn(
        f"[bodex_plan_cli] wrote trajectory ({result['trajectory'].shape}) → {args.out_npy}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
