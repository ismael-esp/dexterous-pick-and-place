#!/usr/bin/env python3
"""Hardware BODex pick-and-place runner (v1 — single attempt, no regrasp).

Mirrors the sim entrypoint ``isaaclab_tasks/run_bodex_pick_place.py`` but
drives the real dual UR3 + Allegro stack. Pipeline:

  1. resolve object geometry (obb / approach ranking)
  2. latch object pose from /fp_pose (7-sample median)         [skipped if --dry_run]
     + pose/table consistency check; optional --snap_z_to_table / --obj_xyz_offset
  3. pick place_xyz (reachable-zone sampler, or --place_xyz)
  4. synthesize with a QUALITY GATE + approach-cycling retry: reject a
     marginal grasp (grasp_err > --grasp_err_max or feasible < --min_feasible)
     and try the next approach / a larger seed budget (+ passive-arm joints)
  5. numpy post-process  →  full [T,22]
  6. safety gates (clamp / velocity / optional FK)
  7. stream arm + hand on ONE schedule with a CLOSED-LOOP grasp barrier:
     reach the grasp pose (wait until measured arm within --grasp_pos_tol)
     before closing the hand + lifting. Default arm path = forward_position
     (Float64MultiArray to {arm}_forward_position_controller — sim-faithful,
     no action server); --arm_iface trajectory uses one FollowJointTrajectory.
  8. grade success from a post-place /<obj>_pose re-latch (skip: --no_grade)

Run it from a sourced ROS2 shell (this machine is ROS2 humble despite
the ros2_ws_jazzy name):

    source /opt/ros/humble/setup.bash
    source ~/Dualarm_ws/ros2_ws_jazzy/install/setup.bash
    python3 ~/Dualarm_ws/BODex/bodex_hw/run_bodex_pick_place_hw.py \
        --active_arm ra --object_id red_cube \
        --object_pose_topic /red_cube_pose --table_z 0.91 \
        --dt 0.18 --no_grade

The {arm}_forward_position_controller must be the active position-
interface controller (deactivate the scaled/JTC controllers first):
    ros2 control set_controller_state \
        ra_scaled_joint_trajectory_controller inactive
    ros2 control set_controller_state \
        ra_forward_position_controller active

BODex grasp synthesis runs in the ``bodex`` conda env behind a Unix
socket; this process only needs numpy + rclpy.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BODEX_REPO = os.path.dirname(_HERE)  # /.../BODex
if _BODEX_REPO not in sys.path:
    sys.path.insert(0, _BODEX_REPO)

from bodex_hw.object_meta import resolve_object_entry
from bodex_hw.trajectory import TrajCfg, build_full_trajectory
from bodex_hw.place_sampler import sample_place_xyz
from bodex_hw import safety

# Home joint angles (deg) — verbatim from env.py:150-151. They already
# match the physical robot's mounting, so no recalibration is needed.
_RA_HOME_DEG = (27.33, -43.98, 98.18, -232.88, -71.98, 223.84)
_LA_HOME_DEG = (-18.75, -138.64, -99.59, 57.97, 64.88, 137.67)


def _home_rad(active_arm: str) -> np.ndarray:
    deg = _RA_HOME_DEG if active_arm == "ra" else _LA_HOME_DEG
    return np.array([math.radians(d) for d in deg], dtype=np.float32)


class _Watchdog:
    """Aborts replay if measured arm state stays off-command for >2*dt."""

    def __init__(self, dt: float, tol: float = 0.10):
        self._max_bad = max(2, int((2.0 * dt) / 0.05) + 1)
        self._tol = tol
        self._bad = 0

    def __call__(self, i, commanded, measured):
        if safety.watchdog_violation(commanded, measured, self._tol):
            self._bad += 1
            if self._bad >= self._max_bad:
                raise SystemExit(
                    f"[watchdog] arm off-command >{self._tol} rad for "
                    f"{self._bad} ticks at wp {i} — aborting."
                )
        else:
            self._bad = 0


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--active_arm", choices=["ra", "la"], default="ra")
    p.add_argument("--object_id", default="red_cube")
    p.add_argument("--approach", default="auto",
                   help="auto | top | side | side_front | side_left | side_right")
    p.add_argument("--object_pose_topic", default="/fp_pose")
    p.add_argument("--object_pose", type=float, nargs=7, default=None,
                   metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"),
                   help="hand-typed object pose (w-first quat); used for "
                        "--dry_run or to bypass /fp_pose")
    p.add_argument("--place_xyz", type=float, nargs=3, default=None,
                   metavar=("X", "Y", "Z"),
                   help="fixed drop target; default = random reachable zone")
    p.add_argument("--place_seed", type=int, default=None)
    p.add_argument("--place_zone", action="append", nargs=5, default=None,
                   metavar=("ARM", "XMIN", "XMAX", "YMIN", "YMAX"),
                   help="random-place AREA override for one arm ('ra' or 'la'):"
                        " world-frame x/y bounds (m) replacing the built-in "
                        "zone. Repeatable, once per arm; an entry for the "
                        "non-active arm is ignored. Explicit bounds bypass "
                        "--place_y_min. --place_xyz (fixed point) still wins. "
                        "e.g. --place_zone ra 0.00 0.25 0.45 0.58")
    p.add_argument("--place_y_min", type=float, default=0.40,
                   help="raise the place zone's lower Y bound (m) so the drop "
                        "target sits further from the T-shape stand (default "
                        "0.40; the base zone starts at 0.32). Keeps the arm "
                        "from clipping the stand at the place. Ignored when "
                        "--place_xyz is given.")
    p.add_argument("--table_z", type=float, default=0.96,
                   help="measured world z of the table top")
    p.add_argument("--dt", type=float, default=0.10,
                   help="seconds per BODex waypoint = the SPEED knob "
                        "(sim≈0.033; start conservative). Motion is smooth "
                        "regardless of dt because setpoints are interpolated. "
                        "Used for phases that don't have a per-phase dt set.")
    p.add_argument("--approach_dt", type=float, default=None,
                   help="per-knot time for Phase 1 (approach to grasp). "
                        "Defaults to --dt. Smaller = faster descent toward "
                        "the cube. 0.08-0.10 is usually fine — the cube "
                        "pose is known and the velocity gate gives the "
                        "real ceiling.")
    p.add_argument("--return_dt", type=float, default=None,
                   help="per-knot time for Phase 3 (post-grasp: transit, "
                        "place descend, return home). Defaults to --dt. "
                        "Smaller = faster after the grasp is secured.")
    p.add_argument("--hand_open_dwell_s", type=float, default=1.0,
                   help="seconds to hold (grasp_arm, OPEN hand) AFTER the "
                        "arm barrier passes and BEFORE the close command. "
                        "Lets the Allegro reach its gravity-equilibrium "
                        "open state consistently (PID with ki=0 cannot "
                        "fully extend fingers against gravity; without "
                        "this dwell each run starts the close from a "
                        "different droop position). Set 0 to disable.")
    p.add_argument("--stream_hz", type=float, default=125.0,
                   help="forward_position publish rate; high rate + "
                        "interpolation = smooth motion (home_arms.py uses "
                        "100 Hz). Has no effect on --arm_iface trajectory")
    p.add_argument("--grasp_settle_s", type=float, default=1.5,
                   help="dwell at the GRASP CONTACT pose (before the lift) "
                        "so the slow Allegro fully closes on the object")
    p.add_argument("--grasp_close_ramp_s", type=float, default=1.0,
                   help="ramp the hand open->firm-closed over this many seconds "
                        "at the start of the grasp dwell instead of snapping "
                        "shut (a sudden close can shove the object). Final grip "
                        "force is unchanged; capped at --grasp_settle_s. "
                        "0 = old snap-closed behavior.")
    p.add_argument("--stagger_close_s", type=float, default=0.35,
                   help="STAGGERED close (signed). >0 = THUMB-FIRST: the thumb "
                        "closes immediately and plants as a backstop; the four "
                        "fingers start this many seconds later so their curl "
                        "presses the object AGAINST the thumb instead of raking "
                        "it backward while the thumb is still arriving. <0 = "
                        "thumb delayed instead (A/B). Capped so the lagging "
                        "group still finishes within --grasp_settle_s. 0 = "
                        "synchronized close (pre-2026-08-24 behavior).")
    p.add_argument("--grasp_close_extra", type=float, default=0.6,
                   help="extra rad added to the curl joints over the whole "
                        "grip phase (contact->release), on top of the "
                        "size/approach squeeze. Compensates FoundationPose "
                        "error so the hand fully wraps the object. 0=sim-"
                        "faithful; try 0.4-1.0")
    p.add_argument("--obj_xyz_offset", type=float, nargs=3,
                   default=[0.0, 0.0, 0.0], metavar=("DX", "DY", "DZ"),
                   help="world-frame xyz (m) added to the latched object "
                        "position before BODex — dial out a systematic "
                        "FoundationPose offset you measure on hardware")
    p.add_argument("--pose_correction", default=None,
                   help="JSON file (from fit_pose_correction.py) with an affine "
                        "P_true ~= A @ P_fp + b world-frame correction for the "
                        "camera's FoundationPose bias. Applied to the latched "
                        "TOPIC pose only (NOT to a hand-typed --object_pose), "
                        "before --obj_xyz_offset and --snap_z_to_table.")
    p.add_argument("--seed_num", type=int, default=40,
                   help="BODex grasp seeds for the FIRST synthesis attempt. "
                        "More seeds = more IK-feasible candidates = a better "
                        "best-of-batch (v1 used 14 and returned marginal "
                        "grasps)")
    # ---- synthesis quality gate + approach-cycling retry ------------
    p.add_argument("--grasp_err_max", type=float, default=0.10,
                   help="reject a synthesized grasp whose best-seed "
                        "grasp_error exceeds this. Aligned with BODex's own "
                        "tight-success threshold (final succ thre=0.1): "
                        "<0.1 is a tightly-successful grasp, >0.1 is what the "
                        "old failed run returned (0.1159). Lower=stricter")
    p.add_argument("--min_feasible", type=int, default=6,
                   help="reject if fewer than this many seeds are IK-feasible "
                        "(all 3 stages). Feasibility ratio is the stronger "
                        "signal than grasp_error alone")
    p.add_argument("--synth_retries", type=int, default=3,
                   help="max approaches to try from the object's "
                        "approach_ranking before giving up (top->side->...). "
                        "Only used when --approach auto")
    p.add_argument("--retry_seed_num", type=int, default=60,
                   help="larger seed budget for retry attempts (>0th)")
    p.add_argument("--allow_marginal", action="store_true",
                   help="if NO approach passes the gate, drive the best "
                        "fallback anyway instead of aborting (debugging only)")
    p.add_argument("--thumb_up_min", type=float, default=0.65,
                   help="SIDE grasps: min cos(thumb-axis, world-up) to accept "
                        "(higher = thumb points more straight up, less chance the "
                        "thumb tip scrapes the table). Default 0.65.")
    p.add_argument("--thumb_up_strict", action="store_true",
                   help="SIDE grasps: HARD thumb-up filter — if no grasp has the "
                        "thumb pointing up (>= --thumb_up_min) report NO grasp "
                        "instead of falling back to a thumb-down grasp. Use this "
                        "to never drive the thumb into the table.")
    p.add_argument("--grasp_mode", choices=["wrap", "pinch"], default="wrap",
                   help="grasp strategy: 'wrap' = default 4-finger force-closure; "
                        "'pinch' = 2-finger + thumb-backstop for small slippery "
                        "round objects (tomato/peach) so they don't squirt out.")
    p.add_argument("--side_approach_curl", type=float, default=0.6,
                   help="SIDE grasps: partial finger pre-curl (rad) on the MCP "
                        "joints during the approach, so the open down-pointing "
                        "fingers lift clear of the table beside the object (a "
                        "fully-open hand's fingers are longer than the side-grasp "
                        "palm height and scrape the table). 0 = fully open. Raise "
                        "if they still scrape; lower if they don't wrap.")
    p.add_argument("--grasp_extra_in", type=float, default=0.0,
                   help="extend the straight-in approach this many meters PAST the "
                        "grasp pose along the press axis, so the palm seats deeper "
                        "on the object (the BODex grasp can stop just short of "
                        "firm contact). Keep small (a few cm); too much pushes the "
                        "object.")
    p.add_argument("--grasp_extra_in_edge", type=float, default=0.0,
                   help="EDGE-conditional press-in: when the latched object sits "
                        "OUTSIDE the inner zone (edge_x/y bounds below — near the "
                        "workspace edge, where the surviving side grasps land "
                        "short/high), use THIS value instead of --grasp_extra_in. "
                        "0 = disabled (always use --grasp_extra_in).")
    p.add_argument("--edge_x_min", type=float, default=0.05,
                   help="inner-zone world-x lower bound for the edge rule")
    p.add_argument("--edge_x_max", type=float, default=0.45,
                   help="inner-zone world-x upper bound for the edge rule")
    p.add_argument("--edge_y_min", type=float, default=0.35,
                   help="inner-zone world-|y| lower bound (ra frame; mirrored "
                        "automatically for la)")
    p.add_argument("--edge_y_max", type=float, default=0.55,
                   help="inner-zone world-|y| upper bound (ra frame; mirrored "
                        "automatically for la)")
    p.add_argument("--grasp_center_bias", type=float, default=0.0,
                   help="penalize palm offset from the object CENTER along its "
                        "body axis (score per meter) — targets the fat "
                        "mid-body and is IMMUNE to 180-deg FP yaw flips "
                        "(unlike grasp_body_bias). ~5 = strong. 0 = off.")
    p.add_argument("--grasp_axis_align", type=float, default=0.0,
                   help="prefer hands whose knuckle line is PARALLEL to the "
                        "object's long axis (mesh +Y x detected yaw): fingers "
                        "distribute ALONG the body, each wrapping ACROSS the "
                        "narrow width = max contact surface. Weight on "
                        "|cos| in [0,1]; ~0.5 = strong tiebreak. 0 = off.")
    p.add_argument("--grasp_body_bias", type=float, default=0.0,
                   help="bias grasp selection along the OBJECT's long axis "
                        "(mesh +Y rotated by the detected yaw): positive = "
                        "toward the +Y end (whale: the HEAD), negative = the "
                        "-Y end. Score per meter of palm offset; 5.0 is a "
                        "strong preference. 0 = off. Needs a trusted FP yaw.")
    p.add_argument("--top_front_min", type=float, default=0.0,
                   help="TOP grasps: only accept hands whose FINGERS point "
                        "front (world +Y, away from the torso) within "
                        "acos(top_front_min) — rejects sideways-yawed hands. "
                        "0 = off. 0.7 ~ +/-45 deg. Falls back to the most "
                        "frontal level grasp if none passes (never aborts on "
                        "heading alone).")
    p.add_argument("--no_top_level_relax", dest="top_level_relax",
                   action="store_false", default=True,
                   help="TOP grasps: disable the hard-object tilt-relax "
                        "fallback, so a cocked/tilted hand is NEVER selected "
                        "even when the best level grasp is loose.")
    p.add_argument("--grasp_z_bonus", type=float, default=0.0,
                   help="SIDE grasps: bias the grasp-selection score toward HIGHER "
                        "grasps by this weight * palm_z, so the fingers clear the "
                        "table on short/uniform objects. 0 = pure grip security; "
                        "raise (e.g. 5-8) to grab higher, lower if it picks weak grips.")
    p.add_argument("--approach_standoff", type=float, default=0.10,
                   help="pre-grasp STANDOFF distance (m) along the grasp axis: the "
                        "hand moves straight in from this far back instead of a "
                        "curved joint lerp, so the fingers don't sweep through an "
                        "object that's under/behind the palm (top + side). "
                        "0 = old joint-lerp approach. Raise if it still clips.")
    p.add_argument("--auto_slow_max_dt", type=float, default=0.0,
                   help="if the velocity gate fails (a joint exceeds its speed "
                        "limit at --dt), automatically RAISE dt up to this cap and "
                        "run the motion slower instead of aborting. Helps awkward "
                        "side grasps. 0 = abort (default).")
    # ---- vertical registration --------------------------------------
    p.add_argument("--snap_z_to_table", action="store_true",
                   help="override the detected object Z with "
                        "table_z + obb_z_world/2 (the cube provably rests on "
                        "the table). Trusts perception for X/Y/yaw only. "
                        "Ignores the Z component of --obj_xyz_offset")
    # ---- closed-loop grasp barrier (forward_position only) ----------
    p.add_argument("--grasp_pos_tol", type=float, default=0.02,
                   help="rad/joint tolerance: the arm must reach the grasp "
                        "pose within this before the hand closes / lifts")
    p.add_argument("--grasp_reach_timeout", type=float, default=3.0,
                   help="seconds to wait for the arm to reach the grasp pose "
                        "before proceeding anyway (logged)")
    # ---- smooth pre-transit lift -----------------------------------
    p.add_argument("--pre_transit_lift_wps", type=int, default=12,
                   help="number of smooth waypoints inserted right after "
                        "the grasp hold to slow the vertical rise off the "
                        "object. Replaces cuRobo-planned transit waypoints "
                        "that pack a high per-knot joint delta with "
                        "linearly-interpolated knots, so the lift looks "
                        "smooth at constant streaming dt. 0 = disable.")
    p.add_argument("--pre_transit_lift_j1_fraction", type=float, default=0.20,
                   help="how far the smooth-lift segment moves shoulder_lift "
                        "toward home (0.20 = 20%% of (home - grasp)). Sets "
                        "the seam where cuRobo's transit resumes verbatim.")
    p.add_argument("--release_open_wps", type=int, default=24,
                   help="slow-release ramp length for small/flat objects: the "
                        "hand opens over this many waypoints at the drop. Lower "
                        "= FASTER open (was 48). Raise toward 48 if a very-flat "
                        "object (e.g. flat_plate) gets flicked on release.")
    p.add_argument("--release_open_s", type=float, default=0.6,
                   help="RELEASE BARRIER: at the place pose the arm HOLDS still "
                        "and the hand opens over this many seconds, THEN the arm "
                        "returns home — so the object isn't dragged by a hand "
                        "opening mid-return. Lower = snappier drop. 0 disables "
                        "the barrier (old overlapping open+return behavior).")
    p.add_argument("--release_open_dwell_s", type=float, default=1.0,
                   help="after the hand opens at release, HOLD the arm + fully-open "
                        "hand for this long before returning home, so the Allegro "
                        "(PID ki=0) reaches its open equilibrium and the fingers "
                        "don't droop back toward the object/table during the home "
                        "move. 0 = no dwell.")
    p.add_argument("--release_lift_frac", type=float, default=0.30,
                   help="RELEASE LIFT: after the hand opens at the drop, the arm "
                        "first LIFTS straight up (shoulder-lift this fraction of "
                        "the way toward home) BEFORE returning home, so it never "
                        "replays BODex's downward set-down motion into the table. "
                        "Raise (e.g. 0.45) for more table clearance.")
    p.add_argument("--release_lift_wps", type=int, default=14,
                   help="number of knots for the post-release vertical lift "
                        "(streamed at return_dt). Higher = smoother/slower lift.")
    p.add_argument("--release_lift_z", type=float, default=0.0,
                   help="DEPRECATED for top grasps. TRUE world +Z re-lift "
                        "(meters, IK-solved in synthesis) added at the goal after "
                        "the descend. Now 0 by default: top grasps re-level the "
                        "hand at the HOVER (above the goal, before the descend) "
                        "instead of descending then re-lifting, which removed the "
                        "old up-down-up wiggle. Leave at 0 — a >0 value re-inserts "
                        "the descend->re-lift segment and breaks the hover index.")
    p.add_argument("--release_settle_frac", type=float, default=0.3,
                   help="TOP-grasp release: the hand carries to the HOVER above the "
                        "goal and opens there (a plain claw drop, NO wrist "
                        "reorientation — the grasp is already near-level). This "
                        "fraction lowers it straight down toward the goal first, "
                        "KEEPING the grasp orientation, to shrink the drop. 0 = open "
                        "at the hover (biggest drop, most finger clearance); higher "
                        "= smaller drop but the down-pointing fingers get closer to "
                        "the table. Keep well under 0.5. Default 0.3.")
    p.add_argument("--vel_scale", type=float, default=0.9,
                   help="fraction of each UR joint's spec speed limit the "
                        "velocity gate allows (0.9 = 90%%)")
    p.add_argument("--max_joint_vel", type=float, default=None,
                   help="optional flat rad/s hard cap on top of the "
                        "per-joint UR limits (extra-conservative first runs)")
    p.add_argument("--post_grasp_lift_z", type=float, default=0.10,
                   help="after the grasp, lift the palm STRAIGHT UP this far in "
                        "world +Z (keeping the grasp orientation) before the "
                        "transit to the goal. Higher = more table clearance while "
                        "carrying, but costs lift-IK feasibility. Also live: "
                        "set <obj> post_grasp_lift_z 0.15")
    p.add_argument("--transit_hover_z", type=float, default=0.15,
                   help="how high ABOVE the place goal the object is carried "
                        "during the transit (m). This is the HIGHEST pose of the "
                        "place phase, so lowering it cuts the arm's reach-up (and "
                        "the fall for side grasps, which open at the hover). "
                        "Also live: set <obj> transit_hover_z 0.08")
    p.add_argument("--side_face_z_min", type=float, default=-1.0,
                   help="SIDE grasps: LOWER bound on the palm press-axis z "
                        "(face_z). face_z=0 is a level side palm, negative is "
                        "the palm cocked DOWN (wrist/forearm dropping toward "
                        "the table). Default -1.0 = no floor (old behaviour); "
                        "-0.25 demands a near-level palm. Never fails synthesis "
                        "— if nothing passes it falls back to the most LEVEL "
                        "seed. Also live: set <obj> side_face_z_min -0.25")
    p.add_argument("--top_level_min", type=float, default=0.0,
                   help="TOP grasps: minimum cos(palm-face, straight-down) for a "
                        "grasp to count as LEVEL. 0 = use the built-in 0.85 "
                        "(~32deg of tilt allowed). 0.95 ~ 18deg, 0.98 ~ 11deg — "
                        "use on LOW objects where a tilted palm dips a wrist "
                        "link into the table. Never fails synthesis: if nothing "
                        "passes it falls back to the most level seed. "
                        "Also live: set <obj> top_level_min 0.95")
    p.add_argument("--pose_max_age_s", type=float, default=3.0)
    p.add_argument("--place_xy_tol", type=float, default=0.15,
                   help="success xy radius (env_cfg place_xy_tolerance)")
    p.add_argument("--arm_iface", choices=["forward_position", "trajectory"],
                   default="forward_position",
                   help="forward_position = stream Float64MultiArray to "
                        "{arm}_forward_position_controller (sim-faithful, no "
                        "action server; default). trajectory = one "
                        "FollowJointTrajectory goal to the scaled JTC")
    p.add_argument("--arm_cmd_topic", default=None,
                   help="override the forward-position command topic "
                        "(default /{arm}_forward_position_controller/commands)")
    p.add_argument("--place_z_offset", type=float, default=0.0,
                   help="raise the PLACE goal z by this much (m): the descend "
                        "stop, hover and release all shift up with it, the "
                        "object drops the extra distance. Grasp side is "
                        "UNAFFECTED (unlike table_z). Use when an arm link "
                        "grazes the table at the drop. 0 = off.")
    p.add_argument("--fk_table_check", action="store_true",
                   help="enable the approximate FK wrist-below-table gate")
    # ---- video recording -------------------------------------------
    p.add_argument("--record", action="store_true",
                   help="record one MP4 per camera topic over the entire "
                        "motion (Phase 1 -> place -> home). One file per "
                        "topic; saved under --record_dir")
    p.add_argument("--record_dir", default="./recordings",
                   help="directory for MP4 outputs (created if absent)")
    p.add_argument("--record_fps", type=float, default=30.0,
                   help="MP4 frame-rate metadata; every received Image is "
                        "written, so set this close to the camera's "
                        "publish rate for correct playback speed")
    p.add_argument("--record_topics", nargs="+",
                   default=["/front_right/cam1/color/image_raw",
                            "/extra/cam2/color/image_raw"],
                   help="ROS Image topics to record (one MP4 per topic)")
    p.add_argument("--mock", action="store_true",
                   help="don't command the arm (hand stream + logs only)")
    p.add_argument("--dry_run", action="store_true",
                   help="synth + post-process + safety only; no ROS, no motion")
    p.add_argument("--no_grade", action="store_true",
                   help="use ONLY the object pose latched at the start; skip "
                        "the post-place /<obj>_pose re-read / success check")
    p.add_argument("--no_passive_collision", action="store_true",
                   help="don't inject the passive arm as a BODex obstacle "
                        "(NOT recommended — the active arm may route through "
                        "the other arm)")
    return p.parse_args(argv)


def _synthesize_once(client, args, object_pose, place_xyz, entry,
                     passive_arm, passive_q, approach, seed_num, thumb_up_min,
                     synth_seed=123):
    """One BODex synthesis call. Returns a dict with the raw trajectory and
    the quality metrics (grasp_error / feasible_count / feasible_total).

    ``synth_seed`` seeds the worker's RNG for THIS call (the worker re-seeds per
    request — see server.py). Varying it across retries makes the otherwise
    non-deterministic grasp optimizer explore a different grasp each attempt."""
    kwargs = dict(
        active_arm=args.active_arm,
        object_id=args.object_id,
        object_pose=object_pose,
        place_xyz=place_xyz,
        approach=approach,
        seed_num=int(seed_num),
        hold_frames=TrajCfg().hold_frames,
        table_z=float(args.table_z),
        thumb_up_min=float(thumb_up_min),
        thumb_up_strict=bool(getattr(args, "thumb_up_strict", False)),
        standoff_dist=float(getattr(args, "approach_standoff", 0.10)),
        grasp_extra_in=float(getattr(args, "grasp_extra_in", 0.0)),
        grasp_z_bonus=float(getattr(args, "grasp_z_bonus", 0.0)),
        post_grasp_lift_z=float(getattr(args, "post_grasp_lift_z", 0.10)),
        transit_hover_z=float(getattr(args, "transit_hover_z", 0.15)),
        side_face_z_min=float(getattr(args, "side_face_z_min", -1.0)),
        top_level_min=float(getattr(args, "top_level_min", 0.0)),
        top_front_min=float(getattr(args, "top_front_min", 0.0)),
        top_level_relax=bool(getattr(args, "top_level_relax", True)),
        grasp_body_bias=float(getattr(args, "grasp_body_bias", 0.0)),
        grasp_axis_align=float(getattr(args, "grasp_axis_align", 0.0)),
        grasp_center_bias=float(getattr(args, "grasp_center_bias", 0.0)),
        grasp_mode=str(getattr(args, "grasp_mode", "wrap")),
        synth_seed=int(synth_seed),
        obj_scale=tuple(float(s) for s in entry.get("scale", (1.0, 1.0, 1.0))),
        obb_z_world=entry.get("obb_z_world"),
        release_lift_z=float(args.release_lift_z),
    )
    if passive_q is not None:
        kwargs["passive_arm"] = passive_arm
        kwargs["passive_joint_state"] = passive_q
    res = client.synthesize_pick_and_place(**kwargs)
    return {
        "raw": np.asarray(res["trajectory"], dtype=np.float32),
        "pick_end": int(res["pick_end_idx"]),
        "descend_n": int(res.get("descend_n", 20)),
        "grasp_error": float(res.get("grasp_error", float("nan"))),
        "feasible_count": int(res.get("feasible_count", -1)),
        "feasible_total": int(res.get("feasible_total", -1)),
        "approach": approach,
        "seed_num": int(seed_num),
    }


def _synthesize(args, object_pose, place_xyz, entry, passive_arm, passive_q):
    """Quality-gated synthesis with approach-cycling retry.

    Ports the sim's robustness (``env.py::_try_replan`` advances through
    ``approach_ranking``; ``_invoke_bodex`` raises the seed budget and
    relaxes the side thumb-up filter on retries) to the *synthesis* stage,
    so a marginal best-of-a-bad-batch grasp (the v1 failure: feasible 4/14,
    grasp_err 0.116) is rejected BEFORE it ever drives the hardware.

    Returns (raw [T,22], pick_end int, approach_used str, descend_n int).
    """
    from bodex_service import BODexClient
    client = BODexClient(socket_path="/tmp/bodex_service.sock",
                         auto_start=True, verbose=True)

    if args.approach == "auto":
        ranking = list(entry.get("approach_ranking", ["top"]))
    else:
        ranking = [args.approach]
    # Total attempts = synth_retries (NOT capped by len(ranking)). The first
    # len(ranking) tries cycle the approach_ranking (top->side->...); any further
    # tries RE-RUN the chosen approach with a fresh seed. The BODex grasp
    # optimizer is non-deterministic (the worker re-seeds per request; cuRobo's
    # GPU solver adds residual noise), so a single side grasp on an IK-marginal
    # object is a coin flip — best-of-N reliably lands a good wrap. We keep the
    # TIGHTEST grasp (lowest worst-contact error) and stop early once one passes.
    n_try = max(1, int(args.synth_retries))

    best = None  # (score_tuple, result, accepted)
    for ai in range(n_try):
        approach = ranking[ai] if ai < len(ranking) else ranking[-1]
        is_retry = ai > 0
        # Re-running the SAME approach (ranking exhausted, or a single forced
        # approach): use the FULL seed budget so each attempt is a real shot at a
        # good grasp. Approach-CYCLING retries keep the cheaper retry_seed_num.
        same_approach_retry = is_retry and (ai >= len(ranking) or len(ranking) == 1)
        seed_num = (args.seed_num if (not is_retry or same_approach_retry)
                    else args.retry_seed_num)
        synth_seed = 123 + ai          # fresh seed per attempt -> explores grasps
        # Side thumb-up filter (so the thumb tip doesn't scrape the table).
        # First attempt uses --thumb_up_min; on APPROACH-CYCLING side retries it
        # is relaxed to 0 to recover feasibility — UNLESS --thumb_up_strict, which
        # keeps it on every attempt. Same-approach retries keep the filter (we're
        # hunting a better grasp of the SAME kind, not a thumb-down fallback).
        base_thumb_up = float(getattr(args, "thumb_up_min", 0.65))
        strict = bool(getattr(args, "thumb_up_strict", False))
        relax = (is_retry and approach.startswith("side")
                 and not strict and not same_approach_retry)
        thumb_up_min = 0.0 if relax else base_thumb_up
        try:
            r = _synthesize_once(client, args, object_pose, place_xyz, entry,
                                 passive_arm, passive_q, approach, seed_num,
                                 thumb_up_min, synth_seed)
        except Exception as e:  # noqa: BLE001 — synthesis can raise (no IK)
            print(f"[hw] synth try {ai} approach={approach} seed_num={seed_num} "
                  f"FAILED: {e}")
            continue
        ratio = (r["feasible_count"] / r["feasible_total"]
                 if r["feasible_total"] > 0 else 0.0)
        accepted = (r["grasp_error"] <= args.grasp_err_max
                    and r["feasible_count"] >= args.min_feasible)
        print(f"[hw] synth try {ai}: approach={approach} seed_num={seed_num} "
              f"seed={synth_seed} grasp_err={r['grasp_error']:.4f} "
              f"(max {args.grasp_err_max}) "
              f"feasible={r['feasible_count']}/{r['feasible_total']} "
              f"(min {args.min_feasible}, {ratio*100:.0f}%) -> "
              f"{'ACCEPT' if accepted else 'reject'}")
        # Prefer accepted; then the TIGHTEST grasp (lowest worst-contact error);
        # then more feasible seeds. grasp_error directly measures wrap quality, so
        # best-of-N converges on a solid grasp rather than a feasible-but-loose one.
        score = (1 if accepted else 0, -r["grasp_error"], r["feasible_count"])
        if best is None or score > best[0]:
            best = (score, r, accepted)
        if accepted:
            break

    if best is None:
        raise SystemExit("[hw] ABORT: BODex synthesis failed for every "
                         "approach (no IK-feasible seed). Move the object to "
                         "a more reachable spot or check the object pose.")
    _, r, accepted = best
    if not accepted and not args.allow_marginal:
        raise SystemExit(
            f"[hw] ABORT: no acceptable grasp. best: approach={r['approach']} "
            f"grasp_err={r['grasp_error']:.4f} (>{args.grasp_err_max}) "
            f"feasible={r['feasible_count']}/{r['feasible_total']} "
            f"(<{args.min_feasible}). Move the object closer / more reachable, "
            f"raise --seed_num, or relax --grasp_err_max/--min_feasible. "
            f"Pass --allow_marginal to drive it anyway (debugging).")
    if not accepted:
        print(f"[hw] WARN: --allow_marginal — driving a MARGINAL grasp "
              f"(grasp_err={r['grasp_error']:.4f}, "
              f"feasible={r['feasible_count']}/{r['feasible_total']})")
    print(f"[hw] grasp accepted: approach={r['approach']} "
          f"grasp_err={r['grasp_error']:.4f} "
          f"feasible={r['feasible_count']}/{r['feasible_total']}")
    # Also return the chosen result dict `r` (grasp_error / feasible_* etc.) so
    # callers like the benchmark can log grasp quality without re-deriving it.
    return r["raw"], r["pick_end"], r["approach"], r["descend_n"], r


def _resolve_place_zone(place_zone, active_arm):
    """Normalize a --place_zone value into the active arm's
    ``[xmin, xmax, ymin, ymax]`` (or None when no zone applies to it).

    Accepts the raw argparse form — a list of ``[ARM, XMIN, XMAX, YMIN, YMAX]``
    string groups — or the already-normalized ``{arm: [4 floats]}`` dict that
    the benchmark's ``set place_zone`` / --place_zone launch flag produces.
    """
    if not place_zone:
        return None
    if isinstance(place_zone, dict):
        zones = {str(a).lower(): [float(v) for v in b]
                 for a, b in place_zone.items()}
    else:
        zones = {}
        for item in place_zone:
            if len(item) != 5:
                raise ValueError(
                    f"--place_zone needs ARM XMIN XMAX YMIN YMAX, got {item}")
            arm = str(item[0]).lower()
            if arm not in ("ra", "la"):
                raise ValueError(
                    f"--place_zone arm must be 'ra' or 'la', got {item[0]!r}")
            zones[arm] = [float(v) for v in item[1:]]
    for arm, (x_lo, x_hi, y_lo, y_hi) in zones.items():
        if not (x_lo < x_hi and y_lo < y_hi):
            raise ValueError(
                f"--place_zone {arm}: need XMIN<XMAX and YMIN<YMAX, "
                f"got {x_lo} {x_hi} {y_lo} {y_hi}")
    zone = zones.get(active_arm)
    if zone is None:
        print(f"[hw] --place_zone set for {sorted(zones)} but active arm is "
              f"{active_arm}; using the built-in zone")
    else:
        print(f"[hw] place zone ({active_arm}): x[{zone[0]:.3f},{zone[1]:.3f}] "
              f"y[{zone[2]:.3f},{zone[3]:.3f}] (place_y_min bypassed)")
    return zone


def _print_pose_consistency(obj_pose, table_z, obb_z_world, place_xyz, tag):
    """Sanity-check the latched object pose against the table height.

    The object provably rests on the table, so its bottom (center_z −
    obb_z/2) should equal table_z. A mismatch means --table_z is wrong (it
    feeds the BODex table collision plane and the place descend) or the
    perception Z is off. X/Y are NOT geometrically constrained — a Y bias
    must be dialled out with --obj_xyz_offset or fixed in the camera→world
    extrinsic."""
    z = float(obj_pose[2])
    bottom = z - float(obb_z_world) / 2.0
    dz = bottom - float(table_z)
    print(f"[hw] pose-check ({tag}): center="
          f"({obj_pose[0]:.3f},{obj_pose[1]:.3f},{z:.3f}) "
          f"obj_bottom_z={bottom:.3f} table_z={table_z:.3f} "
          f"delta={dz*100:+.1f}cm  place=({place_xyz[0]:.3f},"
          f"{place_xyz[1]:.3f},{place_xyz[2]:.3f})")
    if abs(dz) > 0.01:
        print(f"[hw] WARN: object bottom is {dz*100:+.1f} cm off table_z. "
              f"Set --table_z to the MEASURED table top (~{bottom:.3f}) so "
              f"the BODex collision plane + place descend are correct, "
              f"or use --snap_z_to_table.")


def _load_pose_correction(path):
    """Load an affine (A 3x3, b 3) world-frame correction JSON, or None."""
    if not path:
        return None
    with open(path) as f:
        cal = json.load(f)
    A = np.asarray(cal["A"], dtype=float).reshape(3, 3)
    b = np.asarray(cal["b"], dtype=float).reshape(3)
    print(f"[hw] loaded pose_correction '{path}' (model={cal.get('model')} "
          f"n={cal.get('n_points')} resid_mean_xy={cal.get('residual_mean_xy_m')})")
    return (A, b)


def _apply_pose_corrections(obj_pose, args, entry):
    """Order: affine camera->world correction (topic pose, or any --dry_run) ->
    --obj_xyz_offset -> --snap_z_to_table (which overrides Z entirely)."""
    corr = getattr(args, "_pose_corr", None)
    # Correct a perception (topic) pose, or ANY pose under --dry_run (so the
    # correction can be inspected offline). A hand-typed --object_pose in a
    # LIVE run is treated as true-world ground truth and is NOT corrected.
    if corr is not None and (args.object_pose is None or args.dry_run):
        A, b = corr
        old = obj_pose[:3].copy()
        obj_pose[:3] = A @ obj_pose[:3] + b
        print(f"[hw] pose_correction(affine): "
              f"({old[0]:.3f},{old[1]:.3f},{old[2]:.3f}) -> "
              f"({obj_pose[0]:.3f},{obj_pose[1]:.3f},{obj_pose[2]:.3f})")
    obj_pose[:3] += np.asarray(args.obj_xyz_offset, dtype=float)
    if args.snap_z_to_table:
        # NOTE: the snap fully overrides Z, so the obj_xyz_offset z component
        # is discarded here (a grasp-z-bias variant that applied it ON TOP of
        # the snap was tried 2026-08-27 and reverted — it did not work well on
        # hardware; use grasp_extra_in for deeper grasps instead).
        snapped = float(args.table_z) + float(entry["obb_z_world"]) / 2.0
        print(f"[hw] --snap_z_to_table: Z {obj_pose[2]:.3f} -> {snapped:.3f}")
        obj_pose[2] = snapped
    return obj_pose


def _traj_cfg(args) -> TrajCfg:
    return TrajCfg(
        pre_transit_lift_wps=int(args.pre_transit_lift_wps),
        pre_transit_lift_j1_fraction=float(args.pre_transit_lift_j1_fraction),
        slow_release_wps=int(args.release_open_wps),
        side_approach_curl=float(getattr(args, "side_approach_curl", 0.6)),
    )


def run_single_trial(io, args, entry, home_arm, traj_cfg, *, stop_evt=None) -> dict:
    """Run ONE live pick-and-place on already-initialized ROS I/O and return a
    metadata dict (does NOT init/shutdown rclpy or io — the caller owns that).

    This is the per-run body extracted from main() so the benchmark harness can
    call it in a loop. ``stop_evt`` (a threading.Event | None) is forwarded to
    io.replay() so a key press can abort the motion mid-stream. On any pre-motion
    failure (no joint states / no pose / unreachable grasp / safety gate) it
    returns with ``outcome`` set instead of exiting the process.

    Returned dict keys:
        outcome : "ok" | "synth_fail" | "no_joint_states" | "no_passive_arm"
        aborted : bool          # stop_evt was set during the motion
        note    : str           # populated on failure / abort
        approach_used, grasp_error, feasible_count, feasible_total
        obj_pose(7), place_xyz(3), grasp_idx, release_idx, peak_vel_frac
        replay_ok, auto_success, place_err_xy
    """
    res = {
        "outcome": "synth_fail", "aborted": False, "note": "",
        "approach_used": "", "grasp_error": float("nan"),
        "feasible_count": -1, "feasible_total": -1,
        "obj_pose": None, "place_xyz": None,
        "grasp_idx": -1, "release_idx": -1, "peak_vel_frac": float("nan"),
        "replay_ok": False, "auto_success": None, "place_err_xy": None,
    }

    if not io.wait_for_joint_states(15.0):
        print("[hw] ERROR: no /joint_states (is the robot up?)")
        res["outcome"] = "no_joint_states"
        res["note"] = "no /joint_states"
        return res

    if args.object_pose is not None:
        obj_pose = np.asarray(args.object_pose, dtype=float)
        print(f"[hw] using hand-typed object pose {obj_pose}")
    else:
        if not io.wait_for_object_pose(7, 30.0):
            print(f"[hw] ERROR: <7 samples on {args.object_pose_topic}")
            res["outcome"] = "synth_fail"
            res["note"] = f"<7 pose samples on {args.object_pose_topic}"
            return res
        obj_pose = io.latch_object_pose_bodex(7)
        print(f"[hw] latched object pose (wxyz) {obj_pose}")
    if any(args.obj_xyz_offset):
        print(f"[hw] applying obj_xyz_offset {args.obj_xyz_offset}")
    obj_pose = _apply_pose_corrections(obj_pose, args, entry)
    res["obj_pose"] = np.asarray(obj_pose, dtype=float).copy()

    # Edge-conditional press-in: near the workspace edge the surviving
    # IK-feasible side grasps land short/high (cap grasps), so a BIG press-in
    # compensates — but the same press at an inner placement rams a well-seated
    # grasp and tips the object. If grasp_extra_in_edge > 0, pick per trial
    # based on where the latched object actually is: inside the inner zone ->
    # normal grasp_extra_in; outside it -> grasp_extra_in_edge.
    edge_extra = float(getattr(args, "grasp_extra_in_edge", 0.0) or 0.0)
    if edge_extra > 0.0:
        ox, oy = float(obj_pose[0]), float(obj_pose[1])
        oy_eff = oy if args.active_arm == "ra" else -oy   # la mirrors y
        inner = (float(args.edge_x_min) <= ox <= float(args.edge_x_max)
                 and float(args.edge_y_min) <= oy_eff <= float(args.edge_y_max))
        base_extra = float(getattr(args, "grasp_extra_in", 0.0) or 0.0)
        if not inner:
            print(f"[hw] EDGE placement ({ox:.3f},{oy:.3f}): outside inner zone "
                  f"x[{args.edge_x_min:.2f},{args.edge_x_max:.2f}] "
                  f"|y|[{args.edge_y_min:.2f},{args.edge_y_max:.2f}] -> "
                  f"grasp_extra_in {base_extra:.3f} -> {edge_extra:.3f}")
            args.grasp_extra_in = edge_extra
        else:
            print(f"[hw] inner placement ({ox:.3f},{oy:.3f}): keeping "
                  f"grasp_extra_in {base_extra:.3f} "
                  f"(edge value {edge_extra:.3f} not applied)")

    # Fixed place goal: args.place_xyz overrides the random sampler. A z <= 0
    # means "auto" — use the correct per-object height (table_z + obb/2), so
    # one global x,y works for every object. All-zeros (0,0,0) = sentinel for
    # "no fixed goal, use the random sampler" (lets the menu revert live).
    _pfix = getattr(args, "place_xyz", None)
    if _pfix is not None and all(abs(float(v)) < 1e-9 for v in _pfix):
        _pfix = None
    if _pfix is not None and float(_pfix[2]) <= 0.0:
        _pfix = [float(_pfix[0]), float(_pfix[1]),
                 float(args.table_z) + float(entry["obb_z_world"]) / 2.0]
    place_xyz = sample_place_xyz(
        args.active_arm, args.table_z, entry["obb_z_world"],
        seed=args.place_seed, place_y_min=args.place_y_min,
        zone=_resolve_place_zone(getattr(args, "place_zone", None),
                                 args.active_arm),
        override=(np.asarray(_pfix) if _pfix is not None else None))
    # place_z_offset: raise ONLY the drop height (descend stop, hover and
    # release all follow the place goal) — the grasp side is untouched, unlike
    # a table_z change. For objects where a link grazes the table at the drop.
    _pzo = float(getattr(args, "place_z_offset", 0.0) or 0.0)
    if _pzo:
        place_xyz = np.asarray(place_xyz, dtype=float)
        place_xyz[2] += _pzo
        print(f"[hw] place_z_offset {_pzo:+.3f} -> place z {place_xyz[2]:.3f} "
              f"(drop released from higher; grasp unaffected)")
    res["place_xyz"] = np.asarray(place_xyz, dtype=float).copy()
    _print_pose_consistency(obj_pose, args.table_z,
                            entry["obb_z_world"], place_xyz, "latched")

    passive_arm = "la" if args.active_arm == "ra" else "ra"
    passive_q = None if args.no_passive_collision else io.passive_arm_q()
    if passive_q is None and not args.no_passive_collision:
        print("[hw] ERROR: passive arm joints absent from /joint_states; "
              "pass --no_passive_collision to override (unsafe).")
        res["outcome"] = "no_passive_arm"
        res["note"] = "passive arm absent from /joint_states"
        return res

    try:
        raw, pick_end, approach_used, descend_n, meta = _synthesize(
            args, obj_pose, place_xyz, entry, passive_arm, passive_q)
    except SystemExit as e:
        print(str(e))
        res["outcome"] = "synth_fail"
        res["note"] = str(e).replace("\n", " ")[:300]
        return res
    print(f"[hw] BODex raw: shape={raw.shape} pick_end_idx={pick_end} "
          f"descend_n={descend_n}")
    res["approach_used"] = approach_used
    res["grasp_error"] = float(meta.get("grasp_error", float("nan")))
    res["feasible_count"] = int(meta.get("feasible_count", -1))
    res["feasible_total"] = int(meta.get("feasible_total", -1))

    cur_arm = io.active_arm_q().astype(np.float32)
    cur_hand = io.active_hand_q().astype(np.float32)
    full, pe, gi, ri = build_full_trajectory(
        raw, pick_end, approach_used, entry["obb_z_world"],
        cur_arm=cur_arm, cur_hand=cur_hand, home_arm=home_arm,
        cfg=traj_cfg, grasp_close_extra=args.grasp_close_extra)
    res["grasp_idx"] = int(gi)
    res["release_idx"] = int(ri)

    full, n_clamp = safety.clamp_trajectory(full)
    effective_dt = min(
        args.dt,
        args.approach_dt if (args.approach_dt and args.approach_dt > 0) else args.dt,
        args.return_dt if (args.return_dt and args.return_dt > 0) else args.dt,
    )
    ok_v, mv, wi, ji, frac = safety.check_velocity(
        full, effective_dt, args.vel_scale, args.max_joint_vel)
    res["peak_vel_frac"] = float(frac)
    print(f"[hw] post-process: full={full.shape} pick_end={pe} "
          f"grasp_idx={gi} clamped={n_clamp} peak_vel={mv:.3f} rad/s "
          f"@wp{wi}/j{ji} ({frac * 100:.0f}% of limit, "
          f"checked at dt={effective_dt:.3f}s)")
    if not ok_v:
        # AUTO-SLOW: a side-grasp trajectory can wrench a joint past its speed
        # limit at the nominal dt (awkward near-limit arm pose). Rather than
        # abort, raise dt to a safe value (up to --auto_slow_max_dt) and run the
        # WHOLE motion slower. The gate checks at the min phase dt, so bump
        # approach/return dt to match. 0 = disabled (old abort behavior).
        cap = float(getattr(args, "auto_slow_max_dt", 0.0))
        need_dt = effective_dt * (frac / max(args.vel_scale, 1e-3)) * 1.03
        if cap > 0.0 and need_dt <= cap:
            print(f"[hw] velocity gate: j{ji} at {frac * 100:.0f}% of limit — "
                  f"AUTO-SLOW dt {args.dt:.3f} -> {need_dt:.3f}s (cap {cap:.2f}) "
                  f"and continuing")
            args.dt = need_dt
            if args.approach_dt and args.approach_dt < need_dt:
                args.approach_dt = need_dt
            if args.return_dt and args.return_dt < need_dt:
                args.return_dt = need_dt
            effective_dt = need_dt
            ok_v, mv, wi, ji, frac = safety.check_velocity(
                full, effective_dt, args.vel_scale, args.max_joint_vel)
            res["peak_vel_frac"] = float(frac)
        if not ok_v:
            print(f"[hw] ABORT: velocity gate failed — j{ji} at "
                  f"{mv:.3f} rad/s ({frac * 100:.0f}% of its UR limit). "
                  f"Raise --dt (try {need_dt:.3f}) or --auto_slow_max_dt "
                  f"(>= {need_dt:.3f}).")
            res["outcome"] = "synth_fail"
            res["note"] = f"velocity_gate j{ji} {frac * 100:.0f}%"
            return res
    if args.fk_table_check:
        Twb = io.world_base_tf()
        if Twb is None:
            print("[hw] WARN: no world<-base TF; skipping FK gate")
        else:
            ok_fk, mz, idx = safety.check_wrist_above_table(
                full, Twb, args.table_z, args.active_arm)
            print(f"[hw] FK gate: min wrist z={mz:.3f} @wp{idx} (ok={ok_fk})")
            if not ok_fk:
                print("[hw] ABORT: trajectory dips the wrist below the table.")
                res["outcome"] = "synth_fail"
                res["note"] = "fk_gate: wrist below table"
                return res

    wd = None if args.arm_iface == "forward_position" else _Watchdog(args.dt)
    print(f"[hw] replaying {full.shape[0]} wps @ dt={args.dt}s "
          f"(grasp dwell {args.grasp_settle_s}s @wp{gi} = contact, "
          f"close+{args.grasp_close_extra}) "
          f"iface={args.arm_iface} mock_arm={args.mock}")
    if args.record:
        started = io.start_recording(args.record_topics,
                                     args.record_dir,
                                     args.record_fps)
        if started:
            import time as _time
            _time.sleep(0.3)
            print(f"[hw] recording {len(args.record_topics)} topic(s) "
                  f"to {args.record_dir}")
    ok = io.replay(full, args.dt, gi, args.grasp_settle_s,
                   mock_arm=args.mock, watchdog=wd,
                   stream_hz=args.stream_hz,
                   grasp_pos_tol=args.grasp_pos_tol,
                   grasp_reach_timeout=args.grasp_reach_timeout,
                   approach_dt=args.approach_dt,
                   return_dt=args.return_dt,
                   hand_open_dwell_s=args.hand_open_dwell_s,
                   grasp_close_ramp_s=args.grasp_close_ramp_s,
                   stagger_close_s=getattr(args, "stagger_close_s", 0.0),
                   release_idx=ri, release_open_s=args.release_open_s,
                   release_lift_frac=args.release_lift_frac,
                   release_lift_wps=args.release_lift_wps,
                   release_reorient_home=(approach_used == "top"),
                   descend_n=int(descend_n),
                   release_settle_frac=args.release_settle_frac,
                   release_open_dwell_s=args.release_open_dwell_s,
                   stop_evt=stop_evt)
    if args.record:
        io.stop_recording()
    res["replay_ok"] = bool(ok)
    res["aborted"] = bool(stop_evt is not None and stop_evt.is_set())
    res["outcome"] = "ok"   # a motion happened; the caller judges success
    if res["aborted"]:
        res["note"] = "aborted mid-motion"
        return res
    if not ok:
        res["note"] = "replay returned False (no abort)"

    # ---- grade: re-latch post-place pose (CLI path; benchmark uses no_grade) --
    if args.no_grade:
        print("[hw] motion complete (--no_grade: used only the "
              "start-of-run object pose; no post-place re-read)")
    elif args.object_pose is None:
        try:
            io.wait_for_object_pose(1, 5.0)
            final = io.latch_object_pose_bodex(7)
            err_xy = float(np.linalg.norm(final[:2] - place_xyz[:2]))
            success = err_xy <= args.place_xy_tol
            print(f"[hw] place xy err={err_xy:.3f} m (tol {args.place_xy_tol}) "
                  f"-> {'SUCCESS' if success else 'MISS'}")
            res["auto_success"] = bool(success)
            res["place_err_xy"] = err_xy
        except RuntimeError as e:
            print(f"[hw] could not grade (pose lost): {e}")
    else:
        print("[hw] motion complete (no /fp_pose grading in typed-pose mode)")
    return res


def main(argv=None):
    args = _parse_args(argv)
    args._pose_corr = _load_pose_correction(args.pose_correction)
    entry = resolve_object_entry(args.object_id, args.active_arm)
    home_arm = _home_rad(args.active_arm)
    traj_cfg = _traj_cfg(args)
    # 'auto' -> the object's top-ranked approach (v1 single attempt).
    approach_used = (entry["approach_ranking"][0]
                     if args.approach == "auto" else args.approach)
    print(f"[hw] object={args.object_id} arm={args.active_arm} "
          f"approach={approach_used} obb_z_world={entry['obb_z_world']:.4f} "
          f"rank={entry['approach_ranking']}")

    # ---- dry-run: no ROS, hand-typed pose --------------------------
    if args.dry_run:
        if args.object_pose is None:
            args.object_pose = [0.10, 0.47, args.table_z +
                                entry["obb_z_world"] / 2.0, 1, 0, 0, 0]
        obj_pose = np.asarray(args.object_pose, dtype=float)
        obj_pose = _apply_pose_corrections(obj_pose, args, entry)
        place_xyz = sample_place_xyz(
            args.active_arm, args.table_z, entry["obb_z_world"],
            seed=args.place_seed, place_y_min=args.place_y_min,
            zone=_resolve_place_zone(getattr(args, "place_zone", None),
                                     args.active_arm),
            override=(np.asarray(args.place_xyz) if args.place_xyz else None))
        _print_pose_consistency(obj_pose, args.table_z,
                                entry["obb_z_world"], place_xyz, "dry_run")
        raw, pick_end, approach_used, descend_n, _meta = _synthesize(
            args, obj_pose, place_xyz, entry,
            "la" if args.active_arm == "ra" else "ra", None)
        print(f"[hw] BODex raw trajectory: shape={raw.shape} "
              f"pick_end_idx={pick_end} descend_n={descend_n}")
        full, pe, gi, ri = build_full_trajectory(
            raw, pick_end, approach_used, entry["obb_z_world"],
            cur_arm=home_arm, cur_hand=np.zeros(16, np.float32),
            home_arm=home_arm, cfg=traj_cfg,
            grasp_close_extra=args.grasp_close_extra)
        full, n_clamp = safety.clamp_trajectory(full)
        ok_v, mv, wi, ji, frac = safety.check_velocity(
            full, args.dt, args.vel_scale, args.max_joint_vel)
        print(f"[hw] post-process: full={full.shape} pick_end={pe} "
              f"grasp_idx={gi} clamped={n_clamp} peak_vel={mv:.3f} rad/s "
              f"@wp{wi}/j{ji} ({frac * 100:.0f}% of limit, ok={ok_v})")
        if not ok_v:
            print(f"[hw] NOTE: velocity gate would ABORT a live run "
                  f"(j{ji} at {frac * 100:.0f}% of UR limit). "
                  f"Raise --dt or lower --vel_scale headroom.")
        print("[hw] DRY RUN OK")
        return 0

    # ---- live: ROS I/O ---------------------------------------------
    import rclpy
    from bodex_hw.ros_io import HwIO

    rclpy.init()
    io = HwIO(active_arm=args.active_arm,
              object_pose_topic=args.object_pose_topic,
              pose_max_age_s=args.pose_max_age_s,
              arm_iface=args.arm_iface,
              arm_cmd_topic=args.arm_cmd_topic)
    io.start()
    rc = 1
    try:
        res = run_single_trial(io, args, entry, home_arm, traj_cfg,
                               stop_evt=None)
        if res["outcome"] != "ok":
            rc = 1   # no motion (no joint states / no pose / unreachable / gate)
        elif res.get("auto_success") is False:
            rc = 2   # motion completed but auto-grade says place MISS
        else:
            rc = 0
    except SystemExit as e:
        print(str(e))
        rc = 1
    finally:
        io.shutdown()
        try:
            rclpy.shutdown()
        except Exception:
            pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
