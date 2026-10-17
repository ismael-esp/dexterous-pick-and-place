"""BODex pick-and-place task config — self-contained inside BODex.

Scene: dual-arm t-shape torso + motion_desk table. Objects spawn directly on
the table surface (no pedestal). The passive arm stays frozen at home while
the active arm picks and places.
"""
from __future__ import annotations

import math
import os

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from .tshape_cfg import (
    TABLE_TOP_Z,
    make_table_cfg,
    make_tshape_cfg,
    make_tshape_left_robot_cfg,
    make_tshape_right_robot_cfg,
)


_HERE = os.path.dirname(os.path.abspath(__file__))
_ASSETS_DIR = os.path.abspath(os.path.join(_HERE, "..", "assets"))
_MATERIALS_DIR = os.path.join(_ASSETS_DIR, "Materials")

# Where BODex's preprocessed assets live (mesh, urdf, info). Resolved from
# this file's location so it works even if the BODex repo is moved.
_BODEX_PROCESSED_DATA = os.path.abspath(os.path.join(
    _HERE, "..", "..",
    "src", "curobo", "content", "assets", "object", "custom", "processed_data",
))


# Geometry-aware approach ranking. For each object we want a ranked list of
# approaches to try (first → fallback) so a single failing approach does not
# doom the episode. Ranking is geometry-driven (tall/slim → side first,
# flat/short → top first) so this stays object-agnostic — adding a new
# object never requires editing this file.
def recommend_approach_ranking(obb_xyz: np.ndarray, active_arm: str = "ra") -> list[str]:
    """Return a ranked list of approaches for an object given its OBB extents.

    obb_xyz : np.ndarray [3] — full extents (not half) along x, y, z in meters.
    active_arm : 'ra' or 'la'. Used because la's home pose
        (-18.75°, -138.64°, -99.59°, ...) makes the joint-space path from
        home → side-pre-grasp pass through under-table configurations even
        when IK reports feasibility — the wrist briefly dangles below table_z.
        Forcing la to try `top` first avoids those contorted side paths;
        side variants stay as fallbacks for tall objects when top fails.
    """
    obb = np.asarray(obb_xyz, dtype=float)
    obb_z = float(obb[2])
    obb_max_xy = float(max(obb[0], obb[1]))
    obb_min_xy = float(min(obb[0], obb[1]))
    aspect_z = obb_z / max(obb_max_xy, 1e-3)
    flatness = obb_min_xy / max(obb_max_xy, 1e-3)   # 1.0 = round footprint, <1 = oblong

    if aspect_z >= 1.6:
        # Tall (cup, mug, can, tumbler, bottle). Per user directive: ALWAYS try
        # the SIDE (outer-side, thumb-up) wraparound FIRST for tall objects on
        # BOTH arms — it's the grasp that actually secures a tall object — and
        # fall back to TOP only if side is unreachable. `top` is placed second
        # so the synth-fail / replan cycling reaches it immediately when la's
        # side IK is infeasible (la side paths can dangle below the table; the
        # top fallback + cuRobo collision check catch that case). side variants
        # follow as further fallbacks. Top is also the right answer when an
        # object has tipped onto its side.
        return ["side", "top", "side_front", "side_left", "side_right"]
    # Anything that's NOT tall: the side wraparound geometrically can't
    # land a stable grip — the curled fingers don't reach across the
    # short z-extent and the object squirts out. Stick with top-only;
    # retries get fresh object poses (the prior failed grasp may have
    # nudged it into a more reachable spot) so cycling makes no sense.
    return ["top"]


def load_obj_geometry(object_id: str) -> dict | None:
    """Read processed_data/<id>/info/simplified.json if it exists.

    Returns dict with keys: obb (3-vec), gravity_center (3-vec), mass.
    Returns None if no processed_data is found (e.g., the object hasn't been
    preprocessed yet — caller should fall back to the catalog's hand-set values).
    """
    info_path = os.path.join(_BODEX_PROCESSED_DATA, object_id, "info", "simplified.json")
    if not os.path.isfile(info_path):
        return None
    import json
    with open(info_path) as fh:
        data = json.load(fh)
    return {
        "obb": np.asarray(data["obb"], dtype=float),
        "gravity_center": np.asarray(data.get("gravity_center", [0, 0, 0]), dtype=float),
        "mass": float(data.get("mass", 0.2)),
    }


# Object catalog — only carries per-object things that *cannot* be derived
# from the BODex processed_data: USD path, scale, mass, optional primitive
# shape, and an OPTIONAL `spawn_quat`. `obb`, `obb_z`, and recommended
# approach ranking are auto-loaded from
# `processed_data/<id>/info/simplified.json` at __post_init__ time.
#
# `spawn_quat` (wxyz, default identity) — orients the asset on spawn so it
# rests in a known pose right-side-up. Required when the USD asset's local
# frame doesn't match "z-up, lying naturally on its base":
#   - Symmetric objects (sphere, cube, cylinder)  → identity. Physics will
#     settle to a stable pose either way.
#   - Asymmetric objects (mouse, airplane, mug)  → set a quat that rotates
#     the asset so its "natural top" points to world +Z and its longest
#     extent lies horizontal. After settling, the object is correctly
#     oriented and the BODex grasp solver sees a sane pose.
# Adding a new object is: (1) drop into custom_objs/, (2) preprocess
# (BODex), (3) drop a USD into Materials/<id>/<id>.usd, (4) add a 1-line
# entry here, (5) if asymmetric, set spawn_quat by visual inspection.
OBJECT_CATALOG = {
    "peach":    dict(usd=os.path.join(_MATERIALS_DIR, "peach",    "peach.usd"),
                     scale=(1.0, 1.0, 1.0), mass=0.15),
    "red_cube": dict(usd=os.path.join(_MATERIALS_DIR, "red_cube", "red_cube.usd"),
                     scale=(1.0, 1.0, 1.0), mass=0.30),
    "tomato":   dict(usd=os.path.join(_MATERIALS_DIR, "tomato",   "tomato.usd"),
                     scale=(1.0, 1.0, 1.0), mass=0.10),
    # Mouse asset's local frame has X=height (4.7 cm), Y=width (8.2 cm),
    # Z=length (12 cm). Without rotation, identity quat keeps Z as the
    # vertical axis — the mouse stands on its short rear face, then
    # physics tips it onto a random (often upside-down) side. spawn_quat
    # (-90° around world Y) puts object X→world Z (height vertical, with
    # buttons up) and object Z→world X (length horizontal). Mouse spawns
    # already lying flat right-side-up, no settle-flop required.
    # Scale bumped 1.3× so the Allegro hand has enough clearance — at
    # native size (4.7 cm tall) the fingers can pass over the body
    # without contacting; the BODex side `scale` is threaded through so
    # the grasp solver sees the same dimensions.
    "mouse":    dict(usd=os.path.join(_MATERIALS_DIR, "mouse",    "mouse_logi.usd"),
                     scale=(1.3, 1.3, 1.3), mass=0.14,
                     spawn_quat=(0.7071, 0.0, -0.7071, 0.0)),
    "tumbler":  dict(usd=os.path.join(_MATERIALS_DIR, "tumbler",  "tumbler.usd"),
                     scale=(1.0, 1.0, 1.0), mass=0.25),
    # Airplane mesh is from a ~1.3 m model; scale down to ~10 cm so it
    # fits the workspace and the Allegro hand can grip the body.
    "airplane":  dict(usd=os.path.join(_MATERIALS_DIR, "airplane",  "airplane.usd"),
                     scale=(0.08, 0.08, 0.08), mass=0.10),
    # Tall bottle — radius 3.5 cm, height 18 cm. Good target for the
    # `--approach side` thumb-up wraparound grasp (enough clearance above
    # the table for all four fingers).
    "bottle":   dict(primitive="cylinder",
                     radius=0.035, height=0.18,
                     mass=0.30, obb_z=0.18,
                     diffuse_color=(0.15, 0.35, 0.65)),
    # DGN 2K objects (scale=0.1 shrinks from ~1 m mesh-units to real-world
    # meters; BODex and Isaac Lab both receive the same scale factor).
    # Meshes were re-oriented during copy (height → Z axis) so obb_z is
    # read correctly from processed_data/*/info/simplified.json.
    # NOTE: the bare 'can'/'mug' primitive entries were REMOVED — BODex
    # synthesizes against a real mesh (processed_data/<id>/), which Isaac
    # primitives don't have, so they could never be grasped (synthesis errored
    # on the missing simplified.json). The functional can/mug are the USD-backed
    # `core_can` / `core_mug` below, which have both a USD asset and a BODex
    # processed mesh. `bottle` survives only because a bottle mesh was
    # preprocessed into processed_data/bottle/.
    # Flat plate: thin disc (2 cm tall, 14 cm diameter). Mesh is already
    # Z-flat so spawn_quat=identity. BODex will synthesize top/fingertip
    # grasps reaching over the edge.
    "flat_plate": dict(primitive="cylinder",
                       radius=0.071, height=0.020,
                       scale=(0.1, 0.1, 0.1), mass=0.25,
                       diffuse_color=(0.95, 0.93, 0.90)),
    # USD custom objects
    "plate":  dict(usd=os.path.join(_MATERIALS_DIR, "plate",  "plate.usd"),
                     scale=(0.1, 0.1, 0.1), mass=0.10),
    "core_can": dict(usd=os.path.join(_MATERIALS_DIR, "core_can", "core_can.usd"),
                     scale=(0.1, 0.1, 0.1), mass=0.35),
    "core_mug": dict(usd=os.path.join(_MATERIALS_DIR, "core_mug", "core_mug.usd"),
                     scale=(0.1, 0.1, 0.1), mass=0.30),
    
    # --- New scanned objects (Jun 2026). All Z-up, origins centered, scale=1.0. ---
    "red_cube_v2": dict(usd=os.path.join(_MATERIALS_DIR, "red_cube_v2", "red_cube_v2.usd"),
                        scale=(1.0, 1.0, 1.0), mass=0.33),
    "green_cube":  dict(usd=os.path.join(_MATERIALS_DIR, "green_cube", "green_cube.usd"),
                        scale=(1.0, 1.0, 1.0), mass=0.33),
    # pikachu z-squash 0.71 — mirrors bodex_hw/object_meta.py (see the comment
    # there): the top ~3.6cm of the 12.6cm scan is ungraspable ears, so both
    # catalogs plan on the 9.0cm graspable body. Same convention as whale's
    # uniform 0.78; kept in parity so sim and hardware grasp the same geometry.
    "pikachu":     dict(usd=os.path.join(_MATERIALS_DIR, "pikachu", "pikachu.usd"),
                        scale=(1.0, 1.0, 0.71), mass=0.36),
    "mustard":     dict(usd=os.path.join(_MATERIALS_DIR, "mustard", "mustard.usd"),
                        scale=(1.0, 1.0, 1.0), mass=0.32),
    "snorlax":     dict(usd=os.path.join(_MATERIALS_DIR, "snorlax", "snorlax.usd"),
                        scale=(1.0, 1.0, 1.0), mass=0.26),
    "milk":     dict(usd=os.path.join(_MATERIALS_DIR, "milk", "milk.usd"),
                        scale=(1.0, 1.0, 1.0), mass=0.232),
    # whale/tiger: auto-OBB height is wrong (oriented-bbox), so hand-set obb to
    # the true world AABB [X,Y,Z] — otherwise they spawn ~5-6 cm too high.
    # whale planning scale 0.78: the true footprint exceeds the Allegro's max
    # aperture (no full wrap exists at 1.0); 0.78 makes wraps feasible and the
    # plush absorbs the ~2cm/side squeeze (kept in sync with object_meta.py).
    "whale":       dict(usd=os.path.join(_MATERIALS_DIR, "whale", "whale.usd"),
                        scale=(0.78, 0.78, 0.78), mass=0.40,
                        obb=(0.173, 0.216, 0.104)),
    # tiger 0.9 — mirrors bodex_hw/object_meta.py (see the comment there): plan on
    # a 10%-smaller mesh so the 12cm body width clears the Allegro aperture.
    "tiger":       dict(usd=os.path.join(_MATERIALS_DIR, "tiger", "tiger.usd"),
                        scale=(0.9, 0.9, 0.9), mass=0.34,
                        obb=(0.120, 0.257, 0.139)),
    # --- New tall containers (Jun 2026). Z-up, origin-centered, scale=1.0;
    # obb auto-derived from processed_data simplified.json (true axis-aligned,
    # height-on-Z, so no hand-set obb needed). Both semi-empty: green_can =
    # 500ml beverage can (~half full ~0.27kg), tea_peach = 500ml PET iced-tea
    # bottle (~half full ~0.28kg). Tall aspect -> side-grasp ranking.
    # USDs to be dropped into Materials/<id>/<id>.usd. ---
    "green_can":   dict(usd=os.path.join(_MATERIALS_DIR, "green_can", "green_can.usd"),
                        scale=(1.0, 1.0, 1.0), mass=0.27),
    "tea_peach":   dict(usd=os.path.join(_MATERIALS_DIR, "tea_peach", "tea_peach.usd"),
                        scale=(1.0, 1.0, 1.0), mass=0.28),
}

def _quat_rotate_vec(quat_wxyz, v):
    """Rotate 3-vector `v` by quaternion (w, x, y, z). Returns 3-vector."""
    w, x, y, z = quat_wxyz
    vx, vy, vz = v
    # standard quaternion-vector rotation: q * v * q^-1
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return np.array([rx, ry, rz], dtype=float)


def world_aabb_from_obb(obb_xyz: np.ndarray, spawn_quat) -> np.ndarray:
    """Return an axis-aligned bounding box in world frame for an object whose
    object-frame OBB is `obb_xyz` (full extents) when oriented by `spawn_quat`.

    For an axis-aligned half-extents h = obb/2, the world-frame AABB
    half-extents are |R| @ h, where |R| is the absolute rotation matrix.
    Returns full extents (2 * half-extents).
    """
    h = np.asarray(obb_xyz, dtype=float) / 2.0
    # Build rotation matrix columns by rotating each object-frame axis.
    R_cols = np.stack([
        _quat_rotate_vec(spawn_quat, [1.0, 0.0, 0.0]),
        _quat_rotate_vec(spawn_quat, [0.0, 1.0, 0.0]),
        _quat_rotate_vec(spawn_quat, [0.0, 0.0, 1.0]),
    ], axis=-1)  # 3x3
    world_half = np.abs(R_cols) @ h
    return 2.0 * world_half


def resolve_object_entry(object_id: str, active_arm: str = "ra") -> dict:
    """Return a fully-resolved catalog entry. Pulls OBB from processed_data
    when available, applies the (optional) spawn_quat to compute the
    WORLD-frame extents post-orientation, and picks the approach ranking
    based on those world-frame extents.

    Resolved fields:
      - usd / primitive / scale / mass / radius / height (from catalog)
      - spawn_quat: (w, x, y, z) — defaults to identity
      - obb: object-frame full extents (xyz)
      - obb_z: object-frame Z extent (kept for back-compat)
      - obb_world: world-frame AABB after spawn_quat rotation
      - obb_z_world: vertical (Z) extent after spawn_quat rotation — this is
        what determines spawn height, place height, and rest height.
      - approach_ranking: based on world-frame extents (so a "tall in object
        frame" mouse that's pre-rotated to lie flat correctly gets the
        flat-object 'top' rank, not the tall 'side' rank).
    """
    if object_id not in OBJECT_CATALOG:
        raise KeyError(f"Object '{object_id}' not registered in OBJECT_CATALOG")
    base = dict(OBJECT_CATALOG[object_id])  # shallow copy

    geom = load_obj_geometry(object_id)
    if geom is not None:
        base.setdefault("obb", geom["obb"])
        base.setdefault("obb_z", float(geom["obb"][2]))
    elif "obb_z" in base and "obb" not in base:
        # Primitive (e.g., bottle) without preprocess — synthesize a coarse OBB.
        r = base.get("radius", base["obb_z"] / 4.0)
        base["obb"] = np.array([2 * r, 2 * r, base["obb_z"]], dtype=float)

    if "obb_z" not in base or "obb" not in base:
        raise RuntimeError(
            f"Object '{object_id}' has no OBB info. Run preprocess_custom_objs.py "
            f"or add an explicit `obb_z`/`obb` to its OBJECT_CATALOG entry."
        )

    # Apply per-object `scale` (uniform 3-tuple) to the JSON-derived OBB so
    # spawn/place heights and the world-frame extents are all in physical
    # sim units. BODex receives the same `scale` via `obj_scale` so the
    # grasp solver agrees.
    scale_xyz = np.asarray(base.get("scale", (1.0, 1.0, 1.0)), dtype=float)
    base["obb"] = np.asarray(base["obb"], dtype=float) * scale_xyz
    base["obb_z"] = float(base["obb"][2])

    spawn_quat = tuple(base.get("spawn_quat", (1.0, 0.0, 0.0, 0.0)))
    base["spawn_quat"] = spawn_quat
    obb_world = world_aabb_from_obb(base["obb"], spawn_quat)
    base["obb_world"] = obb_world
    base["obb_z_world"] = float(obb_world[2])
    # Approach ranking uses world-frame extents so a pre-rotated asymmetric
    # object (mouse lying flat after spawn_quat) is correctly classified.
    base["approach_ranking"] = recommend_approach_ranking(obb_world, active_arm)
    return base


@configclass
class BODexPickPlaceEnvCfg(DirectRLEnvCfg):
    """Self-contained t-shape pick-and-place task."""

    task_name: str = "bodex_pick_place"
    # Up to 5 attempts (initial + 4 retries) per episode, each ~330 physics
    # steps. Tumbler-like recovery (side → side_front → side_left → side_right
    # → top) cycles the full ranking on a tipped object, so the buffer must
    # accommodate all 5 attempts plus the per-attempt return-to-home preamble.
    episode_length: int = 2000
    action_space: int = 0
    observation_space: int = 12
    decimation: int = 2

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=2,
        physics_material=RigidBodyMaterialCfg(static_friction=2.0, dynamic_friction=2.0),
        physx=PhysxCfg(bounce_threshold_velocity=0.2),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=3.0, replicate_physics=True,
    )

    # Assets path inside this repo
    ASSETS_DIR: str = _ASSETS_DIR

    # Episode parameters
    active_arm: str = "ra"
    object_id: str = "peach"
    # 'auto' picks the best approach for this object's geometry (top for
    # flat/cubic, side for tall/slim). Falls back to other approaches on
    # retry. Explicit values still work and pin the approach.
    approach: str = "auto"

    # Object spawn & place zones — ON the motion_desk table. Table top at z≈0.96.
    #
    # These zones are ARM-SPECIFIC because each UR3 arm can only comfortably
    # reach one half of the workspace (right arm mounted at x=+0.177 reaches
    # the right half; left arm at x=-0.177 reaches the left). The original
    # small zone (x∈[-0.02, 0.15], y∈[0.40, 0.55]) was conservative — the
    # expanded zones below give ~4× the area while staying within each arm's
    # IK-feasible region (BODex will report infeasible seeds outside that
    # region; the regrasp logic handles occasional outliers gracefully).
    # Table extent is x∈[-0.7, 0.7], y∈[0.05, 0.75].
    table_top_z: float = TABLE_TOP_Z

    # Right arm (ra): kept INSIDE the arm's IK-sweet-spot to minimize edge
    # infeasibilities. ra is mounted at (+0.177, 0, 1.49); the best reachable
    # patch on the table is a ~25×25 cm square offset slightly forward.
    object_spawn_x_ra: tuple = ( 0.00, 0.25)
    object_spawn_y_ra: tuple = ( 0.32, 0.58)
    place_target_x_ra: tuple = ( 0.00, 0.25)
    place_target_y_ra: tuple = ( 0.32, 0.58)

    # Left arm (la): mirrors ra in x.
    object_spawn_x_la: tuple = (-0.25, 0.00)
    object_spawn_y_la: tuple = ( 0.32, 0.58)
    place_target_x_la: tuple = (-0.25, 0.00)
    place_target_y_la: tuple = ( 0.32, 0.58)

    object_spawn_yaw: tuple = (-3.14, 3.14)

    # Legacy aliases — kept for external code that may still reference them.
    # They mirror the right-arm zone (the historical default). At runtime,
    # env.py picks the correct zone based on active_arm.
    object_spawn_x: tuple = (-0.10, 0.30)
    object_spawn_y: tuple = ( 0.28, 0.65)
    place_target_x: tuple = (-0.10, 0.30)
    place_target_y: tuple = ( 0.28, 0.65)

    # Settle physics after spawn so the object actually rests on the desk
    settle_steps: int = 40

    # Waypoint repeat: each BODex waypoint is held this many physics env-steps
    # so the arm PID can track each target before we advance.
    # 2 = 2x faster playback than the previous 4 (arm velocity_limit was also
    # doubled in tshape_cfg.py so this is still trackable).
    waypoint_repeat: int = 2

    # Return-to-home phase after the object is released: interpolate the arm
    # joints from the final place-pose back to home over this many waypoints
    # (with fingers open). Smooth continuation so the arm visibly retracts.
    return_to_home_waypoints: int = 30

    # --- Regrasp / retry policy ---
    # Maximum number of pick+place attempts within one episode. Each retry
    # reads the cube's current pose from sim, requests a fresh BODex plan,
    # and splices a "return to home" preamble onto the front of the new
    # trajectory. 0 disables retries (original single-shot behavior).
    # 4 was the sweet spot in v5 (la_flat_plate 90%); 6 caused plate to drift
    # across attempts into IK-edge positions. Reverted.
    max_regrasp_attempts: int = 4
    # Waypoint count for the "current arm state → home" lerp that prepends
    # every retry trajectory (gives the arm a clean starting point for the
    # newly planned approach).
    retry_home_preamble_wps: int = 30
    # --- Fast-regrasp ---
    # When enabled, skip the "go all the way back to home" preamble between
    # attempts. Instead retreat a short distance from the current arm pose
    # (open hand) then jump directly to the freshly synthesized BODex
    # pre-grasp via a short joint-space lerp. Also drops the per-attempt
    # return-to-home appendix (only the FINAL attempt returns to home).
    fast_regrasp_enable: bool = False
    # Joint-space lerp length for the fast-regrasp retreat and preamble phases.
    # 20 wps ≈ 2/3 of the slow preamble; smoother PID tracking at typical arm speeds.
    regrasp_retreat_wps: int = 20
    # Kept for backwards-compatibility; no longer used by the fast-regrasp logic
    # (the retreat now lerps toward home instead of applying a shoulder-lift bias).
    regrasp_retreat_z: float = 0.10
    # xyz threshold to decide the cube has "fallen off the table" while the
    # trajectory is still executing — triggers an immediate mid-transit retry.
    drop_z_threshold: float = 0.75
    # Minimum vertical lift used to judge a grasp as "successful" (otherwise
    # a regrasp is triggered at trajectory end).
    min_lift_for_grasp: float = 0.03

    # --- Place success tolerances ---
    # XY radius around the place target that counts as a successful place.
    # Open-loop BODex grasp + replay can reliably hit ~10 cm; 0.15 gives some
    # margin for the post-release settle and avoids retrying near-misses.
    place_xy_tolerance: float = 0.15
    # Base z tolerance. The actual z tolerance used at runtime is
    # max(place_z_tolerance_base, obb_z * 0.6) — tall objects that fall on
    # their side get a wider z window automatically.
    place_z_tolerance_base: float = 0.08

    # BODex knobs
    hold_frames: int = 10
    # 14 seeds matches the historic 20-seed quality on the validated objects
    # (red_cube/peach/tomato all hit 100% grasp success at 14 in ablation),
    # while shaving ~25% off the warm-call latency. seed_num for the FIRST
    # attempt; retries reduce further (see retry_seed_num).
    bodex_seed_num: int = 14
    # Retries get a smaller seed budget (faster) since approach cycling
    # already provides the diversity that more seeds would.
    bodex_retry_seed_num: int = 10
    # Interpreter of the bodex conda env for the grasp worker. Empty = let
    # BODexClient find it ($BODEX_SERVICE_PYTHON, then `conda run -n bodex`).
    bodex_python: str = os.environ.get("BODEX_SERVICE_PYTHON", "")
    bodex_cli: str = os.path.join(_HERE, "bodex_plan_cli.py")
    # Pass the passive arm's live joint state to BODex so the planner sees it
    # as a moving sphere obstacle (Level-2 collision avoidance from
    # COLLISION_AVOIDANCE.md). Required before the passive arm can be moved
    # off home (collaborative tasks).
    enable_passive_arm_collision: bool = True

    # --- FINGER CLOSURE OVERRIDE (user request) ---
    # During approach: fingers start fully OPEN (all 0) and linearly close to
    # the BODex-planned pregrasp_hand over the approach phase.
    # After grasp contact:  all finger joints are forced to this closure angle
    # (Isaac Lab will clamp to each joint's URDF limits) to fully wrap around
    # the object and remove any residual gap.
    finger_force_close_angle: float = 1.5
    # Indices within the 16-DoF hand slice (they match the BODex output order
    # joint_0_0..joint_15_0). Spread joints (0, 4, 8, 12) stay unchanged in
    # the force-close phase because on the Allegro they control the thumb/
    # finger base rotation, not curl, and shoving them to 1.5 breaks the
    # kinematic chain. Curl joints get the full 1.5 per user's instruction.
    hand_curl_idx: tuple = (1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15)

    # Reference table_z for BODex (used by CLI as the plane obstacle)
    table_z: float = 0.96

    # Pickable object — filled in __post_init__
    object_cfg: RigidObjectCfg | None = None
    place_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/place_markers",
        markers={
            "place": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 1.0),
                    opacity=0.5,
                ),
            ),
        },
    )
    # Visual status sphere above the active arm. Color encodes runtime state:
    #   green  = full success (grasp + place at episode end)
    #   red    = failure        (at episode end)
    #   blue   = arm collision  (self / table / mount)         — during run
    #   orange = risky trajectory (near joint limit / wrist below table edge)
    # Translucent (opacity 0.6) and no collision props.
    status_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/status_markers",
        markers={
            "green":  sim_utils.SphereCfg(radius=0.03, visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.9, 0.0), emissive_color=(0.0, 0.6, 0.0), opacity=0.6)),
            "red":    sim_utils.SphereCfg(radius=0.03, visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.95, 0.05, 0.05), emissive_color=(0.6, 0.0, 0.0), opacity=0.6)),
            "blue":   sim_utils.SphereCfg(radius=0.03, visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.3, 0.95), emissive_color=(0.0, 0.0, 0.6), opacity=0.6)),
            "orange": sim_utils.SphereCfg(radius=0.03, visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.55, 0.0), emissive_color=(0.6, 0.3, 0.0), opacity=0.6)),
            "neutral": sim_utils.SphereCfg(radius=0.03, visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.85, 0.85, 0.85), opacity=0.55)),
        },
    )
    # Fixed world position of the status sphere — centered over the dual-arm
    # T-shape mount (between the two arm bases at world x=0). Sits above the
    # central post so it's always in the same place per env regardless of
    # where the object spawned.
    status_marker_xyz: tuple = (0.0, 0.0, 1.70)
    # Risk-detection thresholds (used by the orange status):
    risk_wrist_below_table_z: float = 0.96     # wrist link world-z < this → orange
    risk_joint_limit_margin_rad: float = 0.175  # ~10° from URDF limit → orange
    # Re-enable contact tracking ONLY for visualization (does NOT replan).
    enable_status_contact_tracking: bool = True
    # Force threshold (N) above which the BLUE collision flag lights up.
    status_collision_force_threshold_n: float = 12.0

    # Camera
    enable_camera: bool = True
    camera_rgb_dir: str = "/tmp/bodex_pick_place_frames"
    camera_video_path: str = "/tmp/bodex_pick_place.mp4"
    camera_cfg: CameraCfg | None = None

    # --- Level-4 runtime collision detection ---
    # Contact sensors on the arm body (NOT hand/fingers). If any arm link
    # reports a sustained contact force above `collision_force_threshold_n`
    # AND we're past the pick phase (so expected contact with the object is
    # already over), a replan is triggered. Set to 0 to disable.
    # v5 experiment: disabled. Even after past_pick gating, raising the
    # threshold to 20N, and removing wrist links, the sensor still fired
    # on forearm during heavy-object lift (la_core_mug 80→60%). Other
    # replan triggers (grasp_miss, mid-transit drop, place_miss) read
    # object state and remain active — they cover the failure modes that
    # actually matter (object slipped, dropped, or placed wrong).
    enable_contact_sensors: bool = False
    collision_force_threshold_n: float = 20.0
    # Links to monitor — UR3 arm body, EXCLUDING wrist links. Wrist_1/2/3 sit
    # inside the grasp closure and feel the object's reaction force during
    # lift, which the sensor was misclassifying as "arm collision" and
    # killing perfectly fine grasps (la_airplane / la_flat_plate REPLAN
    # spam in v3 was 100% wrist false-trips). Only shoulder/upper_arm/
    # forearm can plausibly collide with the world (table, central post).
    collision_arm_links: tuple = (
        "shoulder_link", "upper_arm_link", "forearm_link",
    )
    ra_contact_cfg: ContactSensorCfg | None = None
    la_contact_cfg: ContactSensorCfg | None = None

    # Scene containers (populated in __post_init__)
    right_robot_cfg: ArticulationCfg | None = None
    left_robot_cfg: ArticulationCfg | None = None
    table_cfg: RigidObjectCfg | None = None
    tshape_cfg: RigidObjectCfg | None = None
    articulations_cfg: dict | None = None
    rigid_objects_cfg: dict | None = None
    markers_cfg: dict | None = None

    def __post_init__(self):
        self.articulations_cfg = {}
        self.rigid_objects_cfg = {}
        self.markers_cfg = {}

        # Scene: both arms (passive one stays frozen) + t-shape torso + table
        self.right_robot_cfg = make_tshape_right_robot_cfg(self.ASSETS_DIR)
        self.left_robot_cfg = make_tshape_left_robot_cfg(self.ASSETS_DIR)
        self.table_cfg = make_table_cfg(self.ASSETS_DIR)
        self.tshape_cfg = make_tshape_cfg(self.ASSETS_DIR)

        self.articulations_cfg["right_robot"] = self.right_robot_cfg
        self.articulations_cfg["left_robot"] = self.left_robot_cfg
        self.rigid_objects_cfg["table"] = self.table_cfg
        self.rigid_objects_cfg["tshape"] = self.tshape_cfg

        # Black backdrop wall behind the T-shape stand (in -Y) so the robot
        # silhouette reads cleanly on the recorded video.
        self.rigid_objects_cfg["backdrop_wall"] = RigidObjectCfg(
            prim_path="/World/envs/env_.*/backdrop_wall",
            spawn=sim_utils.CuboidCfg(
                size=(4.0, 0.05, 2.5),   # width (X) × depth (Y) × height (Z)
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.02, 0.02, 0.02),   # near-black
                    roughness=1.0,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, -0.60, 1.25),   # 0.45 m behind tshape origin, centered vertically
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        # Object — spawn 2cm above table top; physics settles it onto the desk.
        # `resolve_object_entry` pulls obb from processed_data/<id>/info/
        # simplified.json automatically and applies any per-object spawn_quat
        # to compute the WORLD-frame Z extent — so the spawn height is right
        # for the actual on-table orientation, not the asset's local-frame Z.
        entry = resolve_object_entry(self.object_id, self.active_arm)
        spawn_z = self.table_top_z + entry["obb_z_world"] / 2.0 + 0.02

        common_rigid_props = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=False,
        )
        common_collision_props = sim_utils.CollisionPropertiesCfg()
        common_mass_props = sim_utils.MassPropertiesCfg(mass=entry.get("mass", 0.2))

        if entry.get("primitive") == "cylinder":
            spawn_cfg = sim_utils.CylinderCfg(
                radius=entry["radius"],
                height=entry["height"],
                rigid_props=common_rigid_props,
                collision_props=common_collision_props,
                mass_props=common_mass_props,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=entry.get("diffuse_color", (0.3, 0.3, 0.3)),
                    roughness=0.5,
                ),
            )
        else:
            spawn_cfg = sim_utils.UsdFileCfg(
                usd_path=entry["usd"],
                scale=entry["scale"],
                rigid_props=common_rigid_props,
                collision_props=common_collision_props,
                mass_props=common_mass_props,
            )

        # Apply the per-object spawn_quat so asymmetric objects (mouse,
        # airplane) start in their natural lying-flat orientation. Symmetric
        # objects keep the identity quat. Episode reset will additionally
        # randomize yaw on top of this base orientation.
        spawn_quat = tuple(float(q) for q in entry.get("spawn_quat", (1.0, 0.0, 0.0, 0.0)))
        self.object_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/pick_object",
            spawn=spawn_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.06, 0.48, spawn_z),
                rot=spawn_quat,
            ),
        )
        self.rigid_objects_cfg["pick_object"] = self.object_cfg
        self.markers_cfg["place_marker"] = self.place_marker_cfg
        self.markers_cfg["status_marker"] = self.status_marker_cfg

        # Camera looking at the workspace from the side-front (OpenGL conv).
        if self.enable_camera:
            self.camera_cfg = CameraCfg(
                prim_path="/World/envs/env_.*/workspace_cam",
                # Full HD 1920x1080 at 60Hz — render_interval = decimation = 2
                # means we produce one frame every env step (60 Hz at 120 Hz sim).
                width=1920,
                height=1080,
                update_period=0.0,
                spawn=sim_utils.PinholeCameraCfg(
                    # 16:9 aperture to match 1920:1080 → no stretch.
                    # FOV ≈ 73° h / 45° v, wider than the old 60°/46° lens so
                    # the full table + top of T-shape stand fit in the frame.
                    focal_length=24.0,
                    horizontal_aperture=36.0,
                    vertical_aperture=20.25,
                    clipping_range=(0.05, 8.0),
                ),
                data_types=["rgb"],
                offset=CameraCfg.OffsetCfg(
                    # FRONT view pulled back + raised so the whole workspace
                    # and the upper half of the T-shape stand are visible.
                    # OpenGL convention: +X=right, +Y=up, -Z=forward.
                    # Quaternion rotation is unchanged from the prior 15.5°
                    # pitch-down look-at — position change alone widens the
                    # visible area thanks to the wider lens.
                    pos=(0.0, 2.0, 1.5),
                    rot=(0.0, 0.0, 0.6051265, 0.7961293),
                    convention="opengl",
                ),
            )

        # Contact sensors on arm body links (exclude hand). Created when EITHER
        # the replan-trigger sensor OR the visualization-only tracker is enabled.
        if self.enable_contact_sensors or self.enable_status_contact_tracking:
            # Regex matches exactly the named arm links; hand links
            # (joint_*_tip etc.) are excluded so grasp contact doesn't trigger
            # a collision replan.
            link_pattern = "(" + "|".join(self.collision_arm_links) + ")"
            self.ra_contact_cfg = ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/RightRobot/ra_{link_pattern}",
                history_length=1,
                update_period=0.0,
                debug_vis=False,
            )
            self.la_contact_cfg = ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/LeftRobot/la_{link_pattern}",
                history_length=1,
                update_period=0.0,
                debug_vis=False,
            )

        # Standard episode-length bookkeeping
        self.episode_length = int(self.episode_length)
        self.episode_length_s = float(self.episode_length) * (1.0 / 60.0)
        self.sim.render_interval = self.decimation
        self.action_space = int(self.action_space)
        self.observation_space = int(self.observation_space)
