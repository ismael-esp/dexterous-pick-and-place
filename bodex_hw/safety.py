"""Pre-send and runtime safety gates (HARDWARE_PORT_PLAN.md §8).

The sim runs blind; hardware can break itself. These gates run on the
final ``[T,22]`` trajectory *before* any goal is sent, plus a runtime
watchdog helper.

Gate 1 (clamp) and Gate 2 (velocity) are always on and cheap. Gate 3
(Cartesian wrist-below-table FK check) is approximate — it uses UR3/UR3e
DH parameters and a caller-supplied world<-base transform — so it is
**opt-in** to avoid rejecting otherwise-valid trajectories on FK error.
Gate 4 (watchdog) is a helper called by the ROS loop with live
``/joint_states``.
"""
from __future__ import annotations

import numpy as np

# UR3 / UR3e joint range is +/- 2*pi. Clamp this far inside it (sim only
# soft-fails; the UR safety controller hard-faults on a limit breach).
ARM_LIMIT = 2.0 * np.pi
# Coarse outer guard for the 16 Allegro joints. BODex already solves IK on
# the Allegro model and the hand driver clamps per-joint, so this only
# catches gross garbage.
HAND_LO, HAND_HI = -0.55, 1.85

# Thumb-lift joint (joint_12_0 = hand index 12) cannot physically retract below
# this on the real Allegro (technical-spec minimum). The "fully open" pose that
# commands 0 is unreachable, so the PID would fight the limit and the open/droop
# state would be inconsistent. Floor joint 12 here so no waypoint ever commands
# the thumb below its real minimum. Matches the URDF lower bound.
THUMB_J12_MIN = 0.36
_J12_COL = 6 + 12  # column 18 in the [6 arm + 16 hand] trajectory layout

# UR3 / UR3e max joint speeds (rad/s): base/shoulder/elbow = 180 deg/s,
# wrist 1/2/3 = 360 deg/s (Universal Robots spec). The velocity gate
# rejects any waypoint step that would exceed these, scaled by a safety
# factor — a flat single limit would either spuriously block the
# wrist-fast sim-validated trajectories or be unsafe for the slow
# shoulder joints.
UR_JOINT_VEL_MAX = np.array(
    [np.pi, np.pi, np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi], dtype=float
)

# Standard DH parameters (meters). ra = UR3, la = UR3e (per the BODex robot
# files ra_ur3_allegro.yml / la_ur3e_allegro.yml).
_DH = {
    "ur3":  dict(d1=0.1519,  a2=-0.24365, a3=-0.21325,
                 d4=0.11235, d5=0.08535,  d6=0.0819),
    "ur3e": dict(d1=0.15185, a2=-0.24355, a3=-0.2132,
                 d4=0.13105, d5=0.08535,  d6=0.0921),
}
_ARM_MODEL = {"ra": "ur3", "la": "ur3e"}


def clamp_trajectory(traj: np.ndarray, margin: float = 0.05):
    """Clamp every waypoint into the joint range. Returns (clamped, n_hits)."""
    out = np.array(traj, dtype=np.float32, copy=True)
    arm = out[:, :6]
    hand = out[:, 6:22]
    lo_a, hi_a = -(ARM_LIMIT - margin), (ARM_LIMIT - margin)
    n = int(np.count_nonzero((arm < lo_a) | (arm > hi_a))
            + np.count_nonzero((hand < HAND_LO) | (hand > HAND_HI)))
    out[:, :6] = np.clip(arm, lo_a, hi_a)
    out[:, 6:22] = np.clip(hand, HAND_LO, HAND_HI)
    # Thumb-lift floor: joint_12_0 can't go below its real-hand minimum.
    n += int(np.count_nonzero(out[:, _J12_COL] < THUMB_J12_MIN))
    out[:, _J12_COL] = np.maximum(out[:, _J12_COL], THUMB_J12_MIN)
    return out, n


def check_velocity(traj: np.ndarray, dt: float, vel_scale: float = 0.9,
                   hard_cap: float | None = None):
    """Reject if any consecutive arm-joint step exceeds that joint's UR
    speed limit (scaled by ``vel_scale``), or an optional flat
    ``hard_cap`` (rad/s) if given.

    The worst offender is reported as the joint with the highest
    *fraction of its own limit* used, so the message points at the joint
    that is actually closest to unsafe.

    Returns (ok, worst_vel, wp_idx, joint_idx, worst_frac).
    """
    arm = np.asarray(traj[:, :6], dtype=float)
    if arm.shape[0] < 2:
        return True, 0.0, -1, -1, 0.0
    d = np.abs(np.diff(arm, axis=0)) / max(dt, 1e-6)  # [T-1, 6] rad/s
    limit = UR_JOINT_VEL_MAX * float(vel_scale)
    if hard_cap is not None:
        limit = np.minimum(limit, float(hard_cap))
    frac = d / limit  # >1.0 anywhere = violation
    flat = int(np.argmax(frac))
    wp_idx, joint_idx = np.unravel_index(flat, frac.shape)
    worst_frac = float(frac[wp_idx, joint_idx])
    worst_vel = float(d[wp_idx, joint_idx])
    return (worst_frac <= 1.0), worst_vel, int(wp_idx), int(joint_idx), \
        worst_frac


def _ur_flange_in_base(q: np.ndarray, model: str) -> np.ndarray:
    """Flange (wrist_3 output) origin in the arm base frame via DH FK."""
    p = _DH[model]
    # Standard UR DH: theta, d, a, alpha per joint.
    dh = [
        (q[0], p["d1"], 0.0,      np.pi / 2),
        (q[1], 0.0,     p["a2"],  0.0),
        (q[2], 0.0,     p["a3"],  0.0),
        (q[3], p["d4"], 0.0,      np.pi / 2),
        (q[4], p["d5"], 0.0,     -np.pi / 2),
        (q[5], p["d6"], 0.0,      0.0),
    ]
    T = np.eye(4)
    for theta, d, a, alpha in dh:
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        A = np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0.0,      sa,       ca,      d],
            [0.0,     0.0,      0.0,    1.0],
        ])
        T = T @ A
    return T[:3, 3]


def check_wrist_above_table(
    traj: np.ndarray,
    T_world_base: np.ndarray,
    table_top_z: float,
    active_arm: str,
    margin: float = 0.02,
):
    """Opt-in Gate 3: reject if the flange ever dips below
    ``table_top_z - margin`` in world z.

    ``T_world_base`` is the 4x4 world<-{arm}_base_link transform (TF
    lookup at runtime). Approximate (DH-based, ignores the hand offset);
    a conservative guard against gross table dips, not an exact model.

    Returns (ok, min_world_z, wp_idx).
    """
    model = _ARM_MODEL.get(active_arm, "ur3")
    T_world_base = np.asarray(T_world_base, dtype=float).reshape(4, 4)
    min_z = np.inf
    min_idx = -1
    for i in range(traj.shape[0]):
        p_base = _ur_flange_in_base(traj[i, :6], model)
        p_world = T_world_base @ np.append(p_base, 1.0)
        if p_world[2] < min_z:
            min_z = float(p_world[2])
            min_idx = i
    return (min_z >= table_top_z - margin), float(min_z), int(min_idx)


def watchdog_violation(
    commanded: np.ndarray, measured: np.ndarray, tol: float = 0.10
) -> bool:
    """True if the measured arm state deviates from commanded beyond tol
    (rad) on any joint — the ROS loop calls this each tick and aborts if
    it stays true for > 2*dt."""
    c = np.asarray(commanded, dtype=float).reshape(-1)[:6]
    m = np.asarray(measured, dtype=float).reshape(-1)[:6]
    return bool(np.any(np.abs(c - m) > tol))
