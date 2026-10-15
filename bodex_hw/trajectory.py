"""Numpy port of the BODex trajectory post-processing.

Faithful, torch-free reimplementation of
``isaaclab_tasks/bodex_pick_place/env.py::_build_and_load_trajectory``
(lines 504-716). Same constants, same branch structure, same arithmetic —
only ``torch`` is swapped for ``numpy`` and the Isaac-Lab state reads are
replaced by explicit ``cur_arm`` / ``cur_hand`` / ``home_arm`` arguments.

v1 of the hardware port never enables fast-regrasp, so the
``fast_regrasp_enable`` / ``prepend_home_from_current`` fast paths are
implemented for parity but unused by the runner.

Trajectory column layout: ``[0:6]`` = active UR3 arm joints (rad),
``[6:22]`` = 16 Allegro hand joints (rad), in BODex's straight numeric
order ``joint_0_0 .. joint_15_0``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Thumb-lift joint floor: joint_12_0 (hand index 12 -> col 18) cannot retract
# below this on the real Allegro (technical-spec min), so the "open" pose uses
# it instead of 0. Keep in sync with bodex_hw/safety.THUMB_J12_MIN, the sim
# env.py open pose, and the URDF lower bound.
THUMB_J12_MIN = 0.36
_J12_COL = 6 + 12

# Shoulder-lift joint index in the 6-DOF arm vector. Moving it toward the home
# value raises the end-effector (this is the same "vertical lift" mechanism the
# embedded top-grasp lift uses). Used by the post-release lift in ros_io.replay.
_SHOULDER_LIFT_J = 1


def release_lift_target(place_arm: np.ndarray, home_arm: np.ndarray,
                        frac: float) -> np.ndarray:
    """Arm pose to LIFT to after releasing the object, before homing.

    Raises the shoulder-lift joint ``frac`` of the way from the place pose
    toward home (clearing the table) while leaving the other joints at the
    place pose, so the motion is a near-vertical retract rather than a swing
    that could dip toward the table. ``frac`` is clamped to [0, 1].

    place_arm, home_arm : [6] arm joint vectors (rad).
    Returns a new [6] vector (inputs untouched).
    """
    place_arm = np.asarray(place_arm, dtype=float).reshape(6)
    home_arm = np.asarray(home_arm, dtype=float).reshape(6)
    f = float(min(max(frac, 0.0), 1.0))
    out = place_arm.copy()
    out[_SHOULDER_LIFT_J] = (place_arm[_SHOULDER_LIFT_J] * (1.0 - f)
                             + home_arm[_SHOULDER_LIFT_J] * f)
    return out


def release_reorient_pose(place_arm: np.ndarray, home_arm: np.ndarray,
                          frac: float) -> np.ndarray:
    """Lifted + LEVELLED arm pose for releasing a TOP grasp.

    For a top grasp the hand points DOWN (fingers wrap the object from above),
    so opening the fingers at the drop extends them straight into the table.
    This pose lifts (``release_lift_target`` — shoulder-lift toward home for
    clearance) and then LEVELS the hand to the home / horizontal orientation so
    the fingers open sideways in free space, clear of the table.

    Levelling correctly requires more than copying the wrist joints: on a UR the
    shoulder-lift (j1), elbow (j2) and wrist_1 (j3) axes are all PARALLEL, so the
    tool's in-plane pitch is the SUM ``j1 + j2 + j3``. Copying only the wrist
    joints leaves the pitch dominated by the folded shoulder/elbow (the hand
    stayed pointing down — the table collision the user saw). Instead we solve
    wrist_1 so the pitch-sum equals home's, which reproduces home's pitch
    regardless of how far the arm reached or lifted. wrist_2 (j4) and wrist_3
    (j5) are set to home for the out-of-plane tilt + roll. Base (j0) and elbow
    (j2) stay at the place pose so the hand remains over the drop spot.

    place_arm, home_arm : [6] arm joint vectors (rad). Returns a new [6] vector.
    """
    place_arm = np.asarray(place_arm, dtype=float).reshape(6)
    home_arm = np.asarray(home_arm, dtype=float).reshape(6)
    out = release_lift_target(place_arm, home_arm, frac)   # out[1] lifted, rest = place
    # Match home's in-plane pitch: j1 + j2 + j3 == home(j1 + j2 + j3).
    pitch_home = home_arm[1] + home_arm[2] + home_arm[3]
    out[3] = pitch_home - out[1] - out[2]
    # Out-of-plane tilt + roll to home -> fully level (horizontal) hand.
    out[4] = home_arm[4]
    out[5] = home_arm[5]
    return out


@dataclass
class TrajCfg:
    """Post-processing constants. Defaults mirror ``BODexPickPlaceEnvCfg``."""

    hold_frames: int = 10
    hand_curl_idx: tuple = (1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15)
    return_to_home_waypoints: int = 30
    retry_home_preamble_wps: int = 30
    fast_regrasp_enable: bool = False
    regrasp_retreat_wps: int = 20
    # Generous cap so the return-to-home tail is never truncated on
    # hardware (sim used a small _T_max which could clip it).
    t_max: int = 4000
    n_approach: int = 30  # matches bodex_plan_cli.py
    # Smooth pre-transit lift: replaces the first few cuRobo-planned
    # transit waypoints (which pack a high per-knot joint delta as the
    # arm rises off the object) with N linearly-interpolated waypoints,
    # so the lift looks smooth at constant streaming dt. Only applies to
    # "top" approaches; auto-skips when joint-1 doesn't actually move
    # enough in the existing transit. Set wps=0 to disable.
    pre_transit_lift_wps: int = 12
    pre_transit_lift_j1_fraction: float = 0.20
    # Slow-release ramp length for small/flat objects (the curl bleeds from
    # squeezed -> open over this many waypoints, streamed at return_dt, so the
    # object isn't flicked by squeeze spring-back). Lower = the hand OPENS
    # FASTER at release. Was hardcoded 48; expose it so hardware can speed the
    # drop. Very-flat objects (flat_plate) may want it back up if they flick.
    slow_release_wps: int = 48
    # SIDE grasps only: partial finger pre-curl (rad) on the proximal/MCP joints
    # during the APPROACH. A fully-open hand's fingers point DOWN and are longer
    # than the palm height at a side grasp, so the open fingertips hit the table
    # beside the object. Pre-curling the MCP joints lifts the fingertips up so
    # they clear the table, while the fingers stay open enough to wrap; the grasp
    # close then completes around the object. 0 = fully open (old behavior).
    side_approach_curl: float = 0.6


def build_full_trajectory(
    raw: np.ndarray,
    pick_end: int,
    approach_used: str,
    obb_z_world: float,
    cur_arm: np.ndarray,
    cur_hand: np.ndarray,
    home_arm: np.ndarray,
    cfg: TrajCfg | None = None,
    prepend_home_from_current: bool = False,
    grasp_close_extra: float = 0.0,
) -> tuple[np.ndarray, int, int]:
    """Apply the sim's finger-closure / lift / return-to-home / preamble
    transforms to a raw BODex ``[T,22]`` trajectory.

    Parameters
    ----------
    raw : [T,22] float — BODex output (6 arm + 16 hand joints, rad).
    pick_end : int — BODex's last pick-phase waypoint index.
    approach_used : str — "top", "side", "side_front", ...
    obb_z_world : float — object world-frame Z extent (drives squeeze /
        lift gating).
    cur_arm : [6] — measured current active-arm joint angles (rad).
    cur_hand : [16] — measured current active-hand joint angles (rad).
    home_arm : [6] — active-arm home joint angles (rad).
    cfg : TrajCfg — constants (sim defaults if None).
    prepend_home_from_current : bool — retry flag (False for v1).
    grasp_close_extra : float — extra rad added to the curl joints during
        the whole grip phase (grasp contact → release), ON TOP of the
        size/approach squeeze. Used on hardware to fully wrap the object
        when FoundationPose pose error leaves the BODex-planned grasp a
        little short. 0.0 = sim-faithful behavior.

    Returns
    -------
    (full [T',22] float32, pick_end' int, grasp_idx int, release_idx int) —
        * full        : the trajectory the hardware replays.
        * pick_end'   : preamble-shifted BODex pick-end (post-lift; kept
          for grading / reference).
        * grasp_idx   : index in ``full`` where the arm is at the grasp
          pose (contact). This is where the replay must DWELL so the
          Allegro fully closes BEFORE the lift — NOT pick_end', which is
          after the lift.
        * release_idx : index in ``full`` where the arm is at the PLACE
          pose with the object still held, just before the hand opens.
          The replay holds the arm here, opens the hand, then returns —
          so the object isn't dragged by a hand opening mid-return.
    """
    if cfg is None:
        cfg = TrajCfg()

    raw = np.asarray(raw, dtype=np.float32)
    home_arm = np.asarray(home_arm, dtype=np.float32).reshape(6)
    cur_arm = np.asarray(cur_arm, dtype=np.float32).reshape(6)
    cur_hand = np.asarray(cur_hand, dtype=np.float32).reshape(16)

    T = min(raw.shape[0], cfg.t_max)
    traj_t = raw[:T].copy()

    N_APPROACH = cfg.n_approach
    HOLD = cfg.hold_frames
    hand_cols = slice(6, 22)
    curl_cols = [6 + i for i in cfg.hand_curl_idx]

    grasp_wp_idx = N_APPROACH
    grasp_hold_end = grasp_wp_idx + HOLD
    squeeze_start = grasp_hold_end

    # Detect the release ramp at the tail (last frames with mean curl < 0.4).
    mean_curl_per_wp = traj_t[:, curl_cols].mean(axis=-1)
    release_start = T
    for k in range(T - 1, -1, -1):
        if mean_curl_per_wp[k] < 0.4:
            release_start = k
        else:
            break

    # grasp_curl is the BODex-planned curl at the contact waypoint. Read it
    # up-front: the original reads it inside the squeeze branch, but it is
    # also needed by the slow-release branch and the value is identical
    # (squeeze only writes indices >= squeeze_start=40 > grasp_wp_idx=30).
    grasp_wp_clamped = min(grasp_wp_idx, T - 1)
    grasp_curl = traj_t[grasp_wp_clamped, curl_cols].copy()

    # (1) Approach — fingers open (thumb-lift held at its real minimum, not 0).
    if T > N_APPROACH:
        traj_t[:N_APPROACH, hand_cols] = 0.0
        traj_t[:N_APPROACH, _J12_COL] = THUMB_J12_MIN

    # (2) Squeeze override on curl joints (approach- and size-aware).
    is_side = bool(approach_used) and approach_used.startswith("side")

    # (1b) SIDE pre-curl: lift the open fingertips above the table during the
    # approach (the open fingers point down and are longer than the side-grasp
    # palm height, so a fully-open hand would scrape the table beside the
    # object). Curl ONLY the proximal/MCP joint of each of the 3 fingers (hand
    # indices 1, 5, 9 -> cols 7, 11, 15), tilting the whole finger up while it
    # stays extended enough to wrap. Thumb (up) and distal joints are untouched.
    if (is_side and cfg.side_approach_curl > 0.0 and T > N_APPROACH):
        mcp_cols = [6 + 1, 6 + 5, 6 + 9]
        traj_t[:N_APPROACH, mcp_cols] = float(cfg.side_approach_curl)

    small_flat = (obb_z_world < 0.08) and (not is_side)
    very_flat = (obb_z_world < 0.04) and (not is_side)
    if is_side:
        squeeze_delta = 0.30
    elif very_flat:
        squeeze_delta = 0.50
    elif small_flat:
        squeeze_delta = 0.20
    else:
        squeeze_delta = 0.10
    # Hardware: start the firm close at the GRASP CONTACT waypoint, not
    # after the 10-frame hold. The sim left 30..40 at the gentle
    # BODex-planned contact "so the initial touch is soft"; on hardware
    # the Allegro is slow and FoundationPose has cm-level error, so the
    # fingers must already be driving closed while the arm dwells at the
    # object (see grasp_idx below). grasp_close_extra adds the user's
    # "grasp totally" margin on top of the size/approach squeeze.
    grip_delta = squeeze_delta + float(grasp_close_extra)
    if release_start > grasp_wp_idx:
        traj_t[grasp_wp_idx:release_start, curl_cols] = (
            grasp_curl + grip_delta
        )

    # (2c) Smooth pre-transit lift. cuRobo's first few transit waypoints
    # after grasp_hold can pack a high per-knot joint delta (the arm
    # must clear the table immediately), so at constant streaming dt
    # the rise looks abrupt while the rest of the motion is smooth.
    # Replace those waypoints with N_LIFT_SMOOTH linearly-interpolated
    # knots from the grasp pose to the cuRobo waypoint that has already
    # risen by `pre_transit_lift_j1_fraction` of the way toward home on
    # shoulder_lift. cuRobo's path from that seam onward is preserved
    # verbatim, so collision guarantees survive. Auto-skips if joint-1
    # doesn't move enough in the existing transit (e.g. side approaches
    # whose lift is on a different joint, or synthetic test data).
    if (cfg.pre_transit_lift_wps > 0
            and release_start > grasp_hold_end + 1
            and approach_used == "top"):
        n_lift_smooth = int(cfg.pre_transit_lift_wps)
        lift_j1_frac = float(cfg.pre_transit_lift_j1_fraction)
        grasp_arm_local = traj_t[grasp_wp_clamped, :6].astype(
            np.float32).copy()
        target_j1 = (grasp_arm_local[1] * (1.0 - lift_j1_frac)
                     + float(home_arm[1]) * lift_j1_frac)
        needed_dj1 = abs(target_j1 - grasp_arm_local[1])
        seam_k = grasp_hold_end
        for k in range(grasp_hold_end, release_start):
            # +eps absorbs float32 truncation on the comparand.
            if (abs(float(traj_t[k, 1]) - grasp_arm_local[1]) + 1e-6
                    >= needed_dj1):
                seam_k = k
                break
        if seam_k > grasp_hold_end:
            seam_arm = traj_t[seam_k, :6].astype(np.float32).copy()
            closed_hand_row = traj_t[grasp_wp_clamped, 6:22].astype(
                np.float32).copy()
            smooth = np.zeros((n_lift_smooth, 22), dtype=np.float32)
            for k in range(n_lift_smooth):
                a = (k + 1) / n_lift_smooth
                smooth[k, :6] = grasp_arm_local * (1.0 - a) + seam_arm * a
                smooth[k, 6:22] = closed_hand_row
            traj_t = np.concatenate(
                [traj_t[:grasp_hold_end], smooth, traj_t[seam_k:]], axis=0)
            release_start += n_lift_smooth - (seam_k - grasp_hold_end)
            T = traj_t.shape[0]

    # (2b) Slow-release ramp for small/flat objects.
    if small_flat and release_start < T:
        # slow_release_wps directly sets the ramp length: lower => faster open.
        # (Was max(release_len_orig, 48), which could only ever LENGTHEN it.)
        slow_len = max(8, int(cfg.slow_release_wps))
        squeezed_curl = grasp_curl + grip_delta
        home_open_curl = np.zeros_like(squeezed_curl)
        alphas = np.linspace(
            0.0, 1.0, slow_len, dtype=np.float32
        ).reshape(-1, 1)
        new_curls = (
            squeezed_curl[None, :] * (1 - alphas)
            + home_open_curl[None, :] * alphas
        )
        head = traj_t[:release_start]
        tail = np.zeros((slow_len, 22), dtype=np.float32) 
        tail[:, :6] = traj_t[release_start, :6][None, :]
        tail[:, 6:22] = traj_t[release_start, 6:22][None, :]
        tail[:, curl_cols] = new_curls
        traj_t = np.concatenate([head, tail], axis=0)
        T = traj_t.shape[0]

    # (3) Vertical-lift prefix (TOP, tall only) + return / retreat tail.
    hand_open = np.zeros(16, dtype=np.float32)
    hand_open[12] = THUMB_J12_MIN  # thumb-lift can't reach 0 on the real hand
    last_arm = traj_t[T - 1, :6]
    last_hand = traj_t[T - 1, 6:22]
    apply_lift = (approach_used == "top") and (obb_z_world >= 0.08)
    if apply_lift:
        LIFT_WPS = 10
        LIFT_FRACTION = 0.30
        lift_target = last_arm.copy()
        lift_target[1] = (
            last_arm[1] * (1 - LIFT_FRACTION) + home_arm[1] * LIFT_FRACTION
        )
        lift_wps = np.zeros((LIFT_WPS, 22), dtype=np.float32)
        for k in range(LIFT_WPS):
            a = (k + 1) / LIFT_WPS
            lift_wps[k, :6] = last_arm * (1 - a) + lift_target * a
            lift_wps[k, 6:22] = last_hand * (1 - a) + hand_open * a
        post_lift_arm = lift_target
        post_lift_hand = hand_open
    else:
        lift_wps = np.zeros((0, 22), dtype=np.float32)
        post_lift_arm = last_arm
        post_lift_hand = last_hand

    if cfg.fast_regrasp_enable:
        N_RETURN = int(cfg.regrasp_retreat_wps)
        retreat_fraction = 0.4
        retreat_arm = (
            post_lift_arm * (1 - retreat_fraction)
            + home_arm * retreat_fraction
        )
        return_wps = np.zeros((N_RETURN, 22), dtype=np.float32)
        for k in range(N_RETURN):
            a = (k + 1) / N_RETURN
            return_wps[k, :6] = post_lift_arm * (1 - a) + retreat_arm * a
            return_wps[k, 6:22] = post_lift_hand * (1 - a) + hand_open * a
    else:
        N_RETURN = int(cfg.return_to_home_waypoints)
        return_wps = np.zeros((N_RETURN, 22), dtype=np.float32)
        for k in range(N_RETURN):
            a = (k + 1) / N_RETURN
            return_wps[k, :6] = post_lift_arm * (1 - a) + home_arm * a
            return_wps[k, 6:22] = post_lift_hand * (1 - a) + hand_open * a
    full = np.concatenate([traj_t, lift_wps, return_wps], axis=0)

    # (4) Prepend a smooth lerp from the MEASURED current arm state -> home.
    if prepend_home_from_current and cfg.fast_regrasp_enable:
        N_PRE = int(cfg.regrasp_retreat_wps)
        target_arm = home_arm
        pre_wps = np.zeros((N_PRE, 22), dtype=np.float32)
        for k in range(N_PRE):
            a = (k + 1) / N_PRE
            pre_wps[k, :6] = cur_arm * (1 - a) + target_arm * a
            pre_wps[k, 6:22] = hand_open
    else:
        N_PRE = int(cfg.retry_home_preamble_wps)
        pre_wps = np.zeros((N_PRE, 22), dtype=np.float32)
        for k in range(N_PRE):
            a = (k + 1) / N_PRE
            pre_wps[k, :6] = cur_arm * (1 - a) + home_arm * a
            pre_wps[k, 6:22] = cur_hand * (1 - a) + hand_open * a
    full = np.concatenate([pre_wps, full], axis=0)
    pick_end = int(pick_end) + N_PRE
    # Grasp contact in `full`: only the prepended preamble shifts early
    # indices (slow-release / lift / return are all appended at/after
    # release_start, which is well past the grasp).
    T_new = min(full.shape[0], cfg.t_max)
    grasp_idx = min(N_PRE + grasp_wp_idx, T_new - 1)
    # Place/release index in `full`: where the arm has descended to the drop
    # and the hand is about to open (release_start, shifted by the preamble).
    # The replay holds the arm here, opens the hand, THEN returns home — so the
    # object isn't dragged by a hand that opens while the arm is already moving.
    release_idx = min(N_PRE + int(release_start), T_new - 1)
    return (full[:T_new].astype(np.float32), int(pick_end), int(grasp_idx),
            int(release_idx))
