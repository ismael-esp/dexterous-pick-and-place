"""Object geometry / approach-ranking metadata — numpy + stdlib only.

This is a faithful port of the *pure* helpers in
``isaaclab_tasks/bodex_pick_place/env_cfg.py`` (``recommend_approach_ranking``,
``load_obj_geometry``, ``world_aabb_from_obb``, ``resolve_object_entry`` and
the geometry fields of ``OBJECT_CATALOG``). It is copied rather than imported
because ``env_cfg.py`` imports Isaac Lab at module load time, which is not
available in the ROS2 system Python.

Keep this in sync with ``env_cfg.py`` if the catalog or ranking rule changes
(only the geometry-relevant fields matter here — USD paths / colors are
intentionally dropped).
"""
from __future__ import annotations

import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BODEX_REPO = os.path.dirname(_HERE)  # /.../BODex
_BODEX_PROCESSED_DATA = os.path.join(
    _BODEX_REPO, "src", "curobo", "content", "assets",
    "object", "custom", "processed_data",
)


# Geometry-only catalog. Mirrors OBJECT_CATALOG in env_cfg.py but carries
# ONLY what resolve_object_entry needs (scale, mass, primitive dims,
# spawn_quat). obb itself is auto-loaded from processed_data/<id>/info/
# simplified.json. USD paths / diffuse colors are omitted (irrelevant to
# grasp geometry).
OBJECT_CATALOG: dict[str, dict] = {
    "peach":      dict(scale=(1.0, 1.0, 1.0), mass=0.15),
    "red_cube":   dict(scale=(1.0, 1.0, 1.0), mass=0.30),
    "tomato":     dict(scale=(1.0, 1.0, 1.0), mass=0.10),
    "mouse":      dict(scale=(1.3, 1.3, 1.3), mass=0.14,
                       spawn_quat=(0.7071, 0.0, -0.7071, 0.0)),
    "tumbler":    dict(scale=(1.0, 1.0, 1.0), mass=0.25),
    "airplane":   dict(scale=(0.08, 0.08, 0.08), mass=0.10),
    "bottle":     dict(primitive="cylinder", radius=0.035, height=0.18,
                       mass=0.30, obb_z=0.18),
    # can/mug primitives removed (no BODex mesh → unusable); core_can/core_mug
    # are the functional versions.
    "flat_plate": dict(primitive="cylinder", radius=0.071, height=0.020,
                       scale=(0.1, 0.1, 0.1), mass=0.25),
    "plate":      dict(scale=(0.1, 0.1, 0.1), mass=0.10),
    "core_can":   dict(scale=(0.1, 0.1, 0.1), mass=0.35),
    "core_mug":   dict(scale=(0.1, 0.1, 0.1), mass=0.30),
    # New scanned objects (synced from env_cfg.py 2026-06). whale/tiger carry an
    # explicit obb (their oriented-bbox json value mis-orders the axes).
    "red_cube_v2": dict(scale=(1.0, 1.0, 1.0), mass=0.33),
    "green_cube":  dict(scale=(1.0, 1.0, 1.0), mass=0.33),
    # pikachu: plan on a Z-SQUASHED mesh. The scan is 12.6cm tall, but the top
    # ~3.6cm is EARS — thin, floppy and ungraspable. Planning on the full height
    # put every candidate grasp up at the head/ears (and snap_z_to_table centred
    # the object 1.8cm too high on top of that), so the hand always came in high
    # and no grasp_z_bonus could fix it — the bonus only RANKS the candidates it
    # is given. Measured graspable body (bottom -> top of head, no ears) = 9.0cm,
    # so scale z = 9.0/12.61 = 0.71. X/Y stay 1.0: the 9.2x11.2cm footprint is
    # already inside the Allegro aperture and must not shrink (unlike whale,
    # which is scaled uniformly to FIT the aperture). obb is multiplied by scale
    # in resolve_object_entry, so this fixes the planning mesh AND the snap
    # height together: obb_z 12.6->9.0cm, centre = table_z + 4.5cm, mesh bottom
    # exactly on the table. Re-measure and re-tune if the toy is replaced.
    "pikachu":     dict(scale=(1.0, 1.0, 0.71), mass=0.36),
    "mustard":     dict(scale=(1.0, 1.0, 1.0), mass=0.32),
    "snorlax":     dict(scale=(1.0, 1.0, 1.0), mass=0.26),
    "milk":        dict(scale=(1.0, 1.0, 1.0), mass=0.232),
    # whale: plan on a 22% smaller mesh. The real whale (17.3x21.6cm footprint)
    # is WIDER than the Allegro's ~13-14cm max aperture — at true scale NO full
    # wrap exists (worker logs: engaged=3/6, worst_err 0.31-0.36, half the
    # contacts never touch). At 0.78 the planning width (~13.5cm) fits the
    # hand, 5-6 contact wraps become feasible, and the real fingers sink ~2cm
    # into the plush per side = a firm grip on the heaviest object (0.40kg).
    # FP keeps tracking the true-size mesh — this scales ONLY the BODex
    # planning geometry (obj_scale is per-request, no worker restart).
    "whale":       dict(scale=(0.78, 0.78, 0.78), mass=0.40, obb=(0.173, 0.216, 0.104)),
    # tiger: plan on a 10%-smaller mesh (live 09-02, same idea as whale's 0.78 but
    # milder). The 12.0cm body width is close to the Allegro's usable aperture, so
    # planning at 10.8cm gives the wrap margin and lets the fingers sink into the
    # plush instead of stopping on its surface. obb is multiplied by scale in
    # resolve_object_entry, so the snap height follows too: obb_z 13.9->12.5cm,
    # centre = table_z + 6.3cm (0.7cm lower than at true scale).
    "tiger":       dict(scale=(0.9, 0.9, 0.9), mass=0.34, obb=(0.120, 0.257, 0.139)),
    # --- New tall containers (Jun 2026). Z-up, origin-centered, scale=1.0;
    # obb comes from processed_data/<id>/info/simplified.json. Both are semi-
    # empty: green_can = 500ml beverage can (~half full), tea_peach = 500ml
    # PET iced-tea bottle (~half full). Tall aspect -> side-grasp ranking. ---
    "green_can":   dict(scale=(1.0, 1.0, 1.0), mass=0.27),
    "tea_peach":   dict(scale=(1.0, 1.0, 1.0), mass=0.28),
}


def recommend_approach_ranking(obb_xyz: np.ndarray, active_arm: str = "ra") -> list[str]:
    """Ranked approach list for an object given its OBB full extents (meters).

    Port of env_cfg.recommend_approach_ranking. Tall objects (aspect_z >= 1.6)
    try the SIDE (outer-side, thumb-up) wraparound FIRST on BOTH arms and fall
    back to TOP if side IK is unreachable; everything else is top-only.
    """
    obb = np.asarray(obb_xyz, dtype=float)
    obb_z = float(obb[2])
    obb_max_xy = float(max(obb[0], obb[1]))
    aspect_z = obb_z / max(obb_max_xy, 1e-3)

    if aspect_z >= 1.6:
        return ["side", "top", "side_front", "side_left", "side_right"]
    return ["top"]


def load_obj_geometry(object_id: str) -> dict | None:
    """Read processed_data/<id>/info/simplified.json, or None if absent."""
    info_path = os.path.join(
        _BODEX_PROCESSED_DATA, object_id, "info", "simplified.json"
    )
    if not os.path.isfile(info_path):
        return None
    with open(info_path) as fh:
        data = json.load(fh)
    return {
        "obb": np.asarray(data["obb"], dtype=float),
        "gravity_center": np.asarray(
            data.get("gravity_center", [0, 0, 0]), dtype=float
        ),
        "mass": float(data.get("mass", 0.2)),
    }


def _quat_rotate_vec(quat_wxyz, v):
    """Rotate 3-vector ``v`` by quaternion (w, x, y, z)."""
    w, x, y, z = quat_wxyz
    vx, vy, vz = v
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return np.array([rx, ry, rz], dtype=float)


def world_aabb_from_obb(obb_xyz: np.ndarray, spawn_quat) -> np.ndarray:
    """World-frame AABB full extents for an object-frame OBB rotated by quat."""
    h = np.asarray(obb_xyz, dtype=float) / 2.0
    R_cols = np.stack([
        _quat_rotate_vec(spawn_quat, [1.0, 0.0, 0.0]),
        _quat_rotate_vec(spawn_quat, [0.0, 1.0, 0.0]),
        _quat_rotate_vec(spawn_quat, [0.0, 0.0, 1.0]),
    ], axis=-1)
    world_half = np.abs(R_cols) @ h
    return 2.0 * world_half


def resolve_object_entry(object_id: str, active_arm: str = "ra") -> dict:
    """Fully-resolved geometry entry: obb / obb_z / obb_world / obb_z_world /
    scale / spawn_quat / approach_ranking. Verbatim port of
    env_cfg.resolve_object_entry minus the Isaac-Lab spawn config.
    """
    if object_id not in OBJECT_CATALOG:
        raise KeyError(
            f"Object '{object_id}' not registered in OBJECT_CATALOG"
        )
    base = dict(OBJECT_CATALOG[object_id])

    geom = load_obj_geometry(object_id)
    if geom is not None:
        base.setdefault("obb", geom["obb"])
        base.setdefault("obb_z", float(geom["obb"][2]))
    elif "obb_z" in base and "obb" not in base:
        r = base.get("radius", base["obb_z"] / 4.0)
        base["obb"] = np.array([2 * r, 2 * r, base["obb_z"]], dtype=float)

    if "obb_z" not in base or "obb" not in base:
        raise RuntimeError(
            f"Object '{object_id}' has no OBB info. Preprocess it or add an "
            f"explicit obb_z/obb to its OBJECT_CATALOG entry."
        )

    scale_xyz = np.asarray(base.get("scale", (1.0, 1.0, 1.0)), dtype=float)
    base["obb"] = np.asarray(base["obb"], dtype=float) * scale_xyz
    base["obb_z"] = float(base["obb"][2])

    spawn_quat = tuple(base.get("spawn_quat", (1.0, 0.0, 0.0, 0.0)))
    base["spawn_quat"] = spawn_quat
    obb_world = world_aabb_from_obb(base["obb"], spawn_quat)
    base["obb_world"] = obb_world
    base["obb_z_world"] = float(obb_world[2])
    base["approach_ranking"] = recommend_approach_ranking(
        obb_world, active_arm
    )
    return base
