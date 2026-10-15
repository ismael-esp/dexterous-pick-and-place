"""Place-target sampler — same reachable zones as the sim.

The sim samples the drop location uniformly inside an arm-specific patch
of the table (``place_target_{x,y}_{ra,la}`` in ``env_cfg.py``). Hardware
v1 mirrors that, with a ``--place_xyz`` CLI override for deterministic
first tests.
"""
from __future__ import annotations

import numpy as np

# Verbatim from BODexPickPlaceEnvCfg (env_cfg.py:327-336). Each UR3 only
# comfortably reaches its own half of the table.

# wider place area:
# PLACE_ZONES = {
#     "ra": dict(x=(0.00, 0.25), y=(0.32, 0.58)),
#     "la": dict(x=(-0.25, 0.00), y=(0.32, 0.58)),
# }

# narrow place area:
PLACE_ZONES = {
    "ra": dict(x=(0.00, 0.25), y=(0.32, 0.58)),
    "la": dict(x=(-0.25, 0.00), y=(0.32, 0.58)),
}

def sample_place_xyz(
    active_arm: str,
    table_z: float,
    obb_z_world: float,
    seed: int | None = None,
    override: np.ndarray | None = None,
    place_y_min: float | None = None,
    zone: "tuple[float, float, float, float] | None" = None,
) -> np.ndarray:
    """Return a world-frame ``[x,y,z]`` drop target for the object centroid.

    Parameters
    ----------
    active_arm : 'ra' or 'la' — selects the reachable zone.
    table_z : measured world z of the table top.
    obb_z_world : object world-frame Z extent; z = table_z + obb_z_world/2
        so the *centroid* rests on the table after release.
    seed : optional RNG seed for reproducible sampling.
    override : if given, returned as-is (the ``--place_xyz`` path).
    place_y_min : if given, RAISE the zone's lower Y bound to this value so the
        drop target sits further out on the table — keeps the place (and the
        arm reaching it) away from the T-shape stand, which on hardware can
        otherwise clip the stand and trip a protective stop. Only applies to
        the built-in ``PLACE_ZONES``; an explicit ``zone`` is used as-given.
    zone : optional ``(xmin, xmax, ymin, ymax)`` — replaces the built-in
        ``PLACE_ZONES[active_arm]`` bounds entirely (the ``--place_zone``
        path). ``place_y_min`` is NOT applied on top of it: the caller has
        stated exact bounds.
    """
    if override is not None:
        return np.asarray(override, dtype=float).reshape(3)

    if zone is not None:
        x_lo, x_hi, y_lo, y_hi = (float(v) for v in zone)
        if not (x_lo < x_hi and y_lo < y_hi):
            raise ValueError(
                f"place zone needs xmin<xmax and ymin<ymax, got {zone}")
    else:
        if active_arm not in PLACE_ZONES:
            raise ValueError(
                f"active_arm must be 'ra' or 'la', got {active_arm!r}")
        z_cfg = PLACE_ZONES[active_arm]
        x_lo, x_hi = z_cfg["x"]
        y_lo, y_hi = z_cfg["y"]
        if place_y_min is not None:
            y_lo = max(y_lo, float(place_y_min))
            if y_lo >= y_hi:
                raise ValueError(
                    f"--place_y_min {place_y_min} leaves no room: "
                    f">= zone y-max {y_hi}")
    rng = np.random.default_rng(seed)
    x = rng.uniform(x_lo, x_hi)
    y = rng.uniform(y_lo, y_hi)
    z = float(table_z) + float(obb_z_world) / 2.0
    return np.array([x, y, z], dtype=float)
