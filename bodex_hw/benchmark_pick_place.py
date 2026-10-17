#!/usr/bin/env python3
"""Interactive hardware benchmark for the BODex pick-and-place.

Run ~20 manually-judged pick-and-place trials per object and log the results to
a CSV for a results meeting.

Flow (operator-attended, one terminal):
  1. A numbered MENU of objects, each with its running score, e.g.
        3) tiger        [12/20, 75%]
  2. Type a number -> select an object. Press 'a' to start one real pick-and-place.
  3. While the robot moves, press 's' (success) or 'f' (fail). EITHER key
     immediately STOPS the motion, returns the arm home, and opens the fingers,
     and records the outcome. If the motion finishes on its own first, you are
     still prompted for s/f.
  4. The trial is APPENDED to the CSV (flushed every row).
  5. Back to the menu. 'q' quits (saving).

Keys: digits + Enter to select; 'a' start; 's' success; 'f' fail; 'q' quit.

This reuses the exact CLI pipeline via run_bodex_pick_place_hw.run_single_trial
(same synthesis / safety gates / replay). Mid-motion abort works because the
forward_position replay honors a stop_evt (see HwIO.replay).

    # real run, right arm, video off:
    python bodex_hw/benchmark_pick_place.py --active_arm ra --table_z 0.91 \
        --csv ~/Dualarm_ws/BODex/benchmark_results/bench.csv

    # dry test with NO robot / NO ROS (exercise menu + CSV + keys):
    python bodex_hw/benchmark_pick_place.py --mock --csv /tmp/bench_test.csv

SAFETY: the post-abort return-home is a straight joint lerp from wherever the
motion froze (NOT collision-checked). It opens the hand first (drops the object),
lifts the shoulder for table clearance, and moves slowly — keep a hand on the
e-stop. Press s/f only when it's safe to drop the object.
"""
from __future__ import annotations

import argparse
import csv
import os
import select
import signal
import sys
import termios
import threading
import time
import tty
from datetime import datetime

import numpy as np

# Allow running as a bare script (`python3 benchmark_pick_place.py`) from any
# directory: put the repo root (parent of bodex_hw/) on sys.path so the
# `bodex_hw.*` package imports below resolve without PYTHONPATH gymnastics.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from bodex_hw.object_meta import resolve_object_entry
from bodex_hw.trajectory import THUMB_J12_MIN, release_lift_target
from bodex_hw.run_bodex_pick_place_hw import (
    _home_rad, _traj_cfg, _load_pose_correction, _parse_args, run_single_trial,
)

# --------------------------------------------------------------------------
# Editable configuration
# --------------------------------------------------------------------------
# Objects offered in the menu (order = menu order). This is the reviewer's
# 10-object set, in the SAME order as multi_object_switcher.py's DEFAULT_OBJECTS,
# so menu number N here == FP viewer jump key N (0 = tea_peach) — switch the
# tracker and pick the object with the same digit. tomato/peach are retired from
# the menu; their PER_OBJECT configs below are kept (inert) for reference.
BENCH_OBJECTS = [
    "tiger", "red_cube", "green_cube", "pikachu", "whale",
    "milk", "snorlax", "mustard", "green_can", "tea_peach",
]

# Baseline pick-and-place tuning applied to EVERY object, overriding the
# run_bodex_pick_place_hw defaults. These mirror the proven per-object hardware
# command (the SHARED flags). CLI flags (--table_z, --obj_xyz_offset,
# --pose_correction, ...) override these at launch; PER_OBJECT overrides them
# per object. EDIT THESE to match your rig.
BASELINE = dict(
    table_z=0.92,             # measured table top (also --table_z)
    snap_z_to_table=True,     # snap object Z to table_z + obb_z/2 (standing objects)
    obj_xyz_offset=[0.05, 0.0, 0.0], #[0.05, -0.02, 0.0],  # world-frame xyz (m) added to the latched
                              # object pose before BODex. DEFAULT for the top/wrap
                              # objects; the three side objects override it below.
                              # A launch-time --obj_xyz_offset overrides ALL objects.
    dt=0.18,
    grasp_pos_tol=0.01,
    grasp_reach_timeout=6.0,
    grasp_settle_s=2.5,
    grasp_close_ramp_s=1.0,
    grasp_close_extra=1.3,
    hand_open_dwell_s=1.5,
    release_open_s=0.6,
    release_open_wps=14,
    release_open_dwell_s=1.0,  # hold open after release so fingers settle (no droop)
    release_settle_frac=0.3,
    seed_num=600,            # 09-03: 600 for EVERY object (operator). The seeds
                             # are optimised as one GPU batch, so a bigger pool
                             # costs little time and gives the guards more
                             # candidates to reject a colliding grasp from.
    grasp_err_max=0.10,
    min_feasible=6,
    no_grade=True,            # the human key is ground truth; skip post-place re-read
    place_y_min=0.45, #0.42,
    auto_slow_max_dt=0.6,     # velocity gate fails -> auto-raise dt up to 0.6s
                              # (awkward side grasps) instead of aborting; 0=abort
    approach_standoff=0.10,   # straight-line approach from 10cm back along the
                              # grasp axis (no curved sweep into the object)
    side_approach_curl=0.35,   # SIDE: pre-curl MCP joints so the open down-fingers
                              # clear the table beside the object during approach
    stagger_close_s=0.85,     # STAGGERED close: thumb-first lead (proven live
                              # 2026-08-27 at 0.85 for top grasps; side objects
                              # override lower). <0 = thumb-last (A/B); 0 = sync
    speed_scale=1.1,          # arm motion speed (proven live 2026-08-27): >1
                              # faster, <1 slower. Clamped to [0.5, 1.5]; the
                              # velocity gate + auto_slow + stream governor
                              # still audit the scaled motion.
    # Edge rule inner zone (position-dependent): a latched object OUTSIDE
    # x[0.05,0.45] / |y|[0.35,0.55] counts as an EDGE placement and uses
    # grasp_extra_in_edge (where set: mustard 0.08, tea_peach 0.04) instead of
    # grasp_extra_in. (The 08-27 live session briefly ran with x_min=0.53 —
    # an empty inner zone, edge press-in always on — reverted per operator.)
    edge_x_min=0.05, edge_x_max=0.45,
    edge_y_min=0.35, edge_y_max=0.55,
    # Default random-place AREA for the right arm (world m), baked 2026-09-02
    # from the launch flag the final benchmark runs with:
    #   --place_zone ra 0.18 0.512 0.450 0.58
    # Explicit bounds REPLACE the built-in PLACE_ZONES and bypass place_y_min.
    # A launch-time --place_zone still overrides this; `set place_zone off`
    # reverts to the built-in zone. Only 'ra' is defined (the benchmark arm) —
    # an la session falls back to the built-in left zone.
    place_zone={"ra": [0.18, 0.512, 0.450, 0.58]},
    # FIXED place goal for EVERY object (live 09-02) — where the object CENTROID
    # ends up. A fixed point WINS over place_zone, so the zone above is now only
    # the fallback if this is cleared. z=1.20 is ~28cm above the 0.92 table (a
    # deliberate drop, not a resting height), so per build_place_trajectory the
    # object hovers at 1.35 (goal+0.15), descends to 1.22 (goal+0.02) and is
    # released partway between (top objects settle 30% down -> ~1.31; the three
    # side objects have release_settle_frac=0 and open at the 1.35 hover).
    # Back to the random zone at runtime: `set place_xyz 0,0,0`.
    # z=0 would instead mean "auto per-object resting height".
    # 09-02 late: moved to the BOX's measured position and pulled INSIDE the
    # right arm's reliable rectangle (x <= 0.50, y in [0.33, 0.57]). The earlier
    # 0.59 was past that limit — the only IK left folded the wrist back onto the
    # forearm and tripped a protective stop. z 1.15 also lowers the transit hover
    # (goal + 0.15) from 1.35 to 1.30, and reach + height compound.
    place_xyz=[-0.12, 0.48, 1.10],
    # After the grasp, lift the palm STRAIGHT UP 15cm in world +Z (grasp
    # orientation kept) before the transit starts, so the object is well clear
    # of the table and of anything standing on it during the carry. The
    # synthesis default is 0.10; 0.15 is the operator's choice (09-03). If an
    # object starts failing with a low `Lift IK Success rate` in the worker log,
    # lower it for that object (`set <obj> post_grasp_lift_z 0.10`) — the lift
    # IK is part of the feasibility mask, so a higher lift can shrink the pool.
    post_grasp_lift_z=0.15,
    # Carry the object only 10cm above the goal (was the hardcoded 0.15):
    # lower transit = less reach-up over the box, and for the three SIDE
    # objects (release_settle_frac=0, they open AT the hover) it is also
    # 5cm less fall.
    transit_hover_z=0.05,
    # Phase-3 (post-grasp: transit -> descend -> release -> home) waypoint
    # time. Speeds up ONLY motion that happens after the object is already
    # gripped, so it costs no grasp accuracy. speed_scale still divides it
    # (0.12 / 1.1 = 0.109 s/knot) and the velocity gate + auto_slow still
    # audit the result.
    return_dt=0.12,
    # Drop the goal 5cm for the TOP-grasp objects. Additive on place_xyz z
    # (1.10 -> 1.05). The three SIDE objects cancel it back to 0.0 below —
    # they open AT the hover, so a lower goal would shorten their rim
    # clearance twice over. NOTE with transit_hover_z=0.05 the tallest
    # objects (snorlax, tiger) clear a ~10cm box rim by only 1-2cm; raise
    # the hover before lowering this further.
    place_z_offset=-0.05,
)

# Per-object overrides merged ON TOP of BASELINE. The three SIDE-grasp objects
# (mustard / green_can / tea_peach) get their own obj_xyz_offset [-0.01,-0.01,0];
# every other object inherits the BASELINE offset [0.05,-0.02,0]. A launch-time
# --obj_xyz_offset still overrides ALL objects. Add snap_z_to_table=False here for
# any object run lying down. Anything not listed inherits BASELINE.
# Values below were baked 2026-08-27 from the operator's proven live session
# (see /home/robot/data_objects.txt for the raw `cfg` dumps), then re-baked
# 2026-08-31 from the pick_place_final_05 session `cfg` dumps (pikachu, whale,
# milk, snorlax deltas).
PER_OBJECT = {
    # NOTE: the live session had tiger obj_xyz_offset z=-0.05, set while the z
    # component was still INERT (snap discarded it). z now applies as a real
    # grasp bias, so it is NOT baked — re-add deliberately and SMALL if wanted:
    #   set tiger obj_xyz_offset 0.05,0,-0.02
    # tiger: same elongated-plush physiognomy as whale (mesh +Y = 25.7cm body,
    # 12cm width) -> same across-the-width wrap: knuckle line along the spine,
    # grip closes over the narrow body. Head direction irrelevant (no body bias).
    # tiger (live 09-02): axis_align +1.5 = knuckle line ALONG the spine, so the
    # three fingers spread down the body and each curls ACROSS the narrow 12cm
    # width. POSITIVE maximises |cos(knuckle-line, body-axis)| (bodex_plan_cli
    # :775); the earlier -1.0 minimised it and therefore asked for the OPPOSITE,
    # lengthwise grip. |cos| makes this immune to 180-deg FP yaw flips. z_bonus
    # -0.55 keeps the palm low; gate 0.14 is what this long plush achieves at the
    # 0.9 planning scale (object_meta).
    "tiger":   dict(grasp_err_max=0.18, grasp_axis_align=1.5,
                    grasp_z_bonus=-0.55,
                    obj_xyz_offset=[0.03, 0.02, 0.0]),
    # whale (re-tuned 2026-08-27 pm): planning scale 0.78 (object_meta) made
    # wraps aperture-feasible; wrap >> pinch; gate 0.25 because the dome always
    # leaves ONE contact floating (engaged 5/6, mean ~0.12 — physically firm on
    # the squishy plush); extra_in 0 (the scale provides the depth; 0.03
    # re-triggered the approach guard); grasp_body_bias +5 = grasp at the HEAD
    # end (mesh +Y; tail flukes are -Y). Re-detect ('n') in the FP viewer until
    # the yaw looks right before whale blocks — the body bias trusts the yaw.
    # whale (live 09-02 late): the hardest object — lowest profile (plans 8.1cm),
    # widest body, softest surface. top_level_relax=False is the anti-collision
    # fix: gate 0.25 > the 0.18 relax trigger meant whale ran a TILTED palm most
    # of the time, and on an 8cm object a tilted palm dips a wrist link into the
    # table (red_cube is the same height, level-only, and never strikes).
    # seed_num 600 + synth_retries 5 widen the candidate pool so the post-grasp
    # collision guard can find a grasp that is clean in BOTH stages instead of
    # falling back. Its own place goal (the box), pulled inside the arm's
    # reliable x<=0.50 rectangle.
    "whale":   dict(grasp_err_max=0.25, grasp_mode="wrap",
                    synth_retries=5, top_level_relax=False,
                    # top_level_min 0.95 (~18deg) instead of the built-in 0.85
                    # (~32deg): relax=False only removes the >gate escape hatch,
                    # the GATE itself still admitted a ~30deg tilted palm — which
                    # on this 8.1cm-tall object dips a wrist link into the table
                    # (live 09-03). Falls back to the most-level seed if nothing
                    # passes, so it cannot fail synthesis.
                    top_level_min=0.95,
                    # POSITION preference = center_bias (not body_bias): FP's
                    # yaw lock 180°-flips on this near-symmetric black plush
                    # (proven trial 43: bias pulled to the believed head = real
                    # tail). center_bias targets the fat MID-BODY and is immune
                    # to flips. axis_align dropped to 0.0 in the final_05 live
                    # session (center_bias alone proved enough).
                    # extra_in MUST stay 0: on a top grasp palm-x points DOWN,
                    # so any press-past-the-grasp drives the OPEN fingers into
                    # the table on this low dome and the approach collision
                    # guard rejects EVERY candidate ("collides for every
                    # candidate grasp", seen 08-31 with the final_05-baked
                    # 0.015; 0.03 tripped it before too). The 0.78 planning
                    # scale already provides the grip depth.
                    grasp_extra_in=0.0, grasp_body_bias=0.0,
                    grasp_center_bias=5.0, grasp_axis_align=0.0,
                    # low dome: prefer HIGHER wraps (tips clear the table at
                    # full squeeze) + milder squeeze overshoot than baseline
                    grasp_z_bonus=3.0, grasp_close_extra=0.9,
                    obj_xyz_offset=[0.05, 0.02, 0.0]),
    # pikachu/milk/snorlax grasp_z_bonus -0.25: bias the wrap slightly LOWER on
    # the body (proven in the final_05 live session).
    # pikachu z_bonus -5.0 (live 09-01): strongly prefer a LOWER wrap — the ears
    # made the optimizer grab high on the head, where the grip slips.
    # table_z 0.91 (live 09-02) = 1cm below the shared 0.92: a deliberate
    # grasp-height bias that lowers the planned object (and so the hand) on top
    # of the 0.71 z-squash in object_meta. NOT a table measurement.
    # live 09-03: the 0.14 gate was calibrated on the ORIGINAL 12.6cm mesh;
    # after the 0.71 z-squash (object_meta) the achievable worst-contact
    # error moved to ~0.17-0.19, so 0.14 aborted every trial. z_bonus back
    # to -0.25 (the -5.0 was also pre-squash — the squash already lowers the
    # grasp 1.8cm, so -5.0 double-counted and forced a poor low grasp).
    "pikachu": dict(grasp_err_max=0.20, grasp_z_bonus=-0.35,
                    table_z=0.91,
                    obj_xyz_offset=[0.02, 0.02, 0.0]),
    # Cubes (live 09-01): gate 0.12, slightly lower wrap, +0.05 x offset.
    # red_cube ALSO runs top_level_relax=False (never accept the tilted
    # fallback — the diagonal-palm approach) + 1cm press-in so the palm seats
    # on the top face; green_cube did not need either.
    # BOTH cubes now run top_level_relax=False (live 09-02): never accept the
    # tilted fallback, so the palm comes down level on the top face instead of
    # cocked diagonally.
    "red_cube":   dict(grasp_err_max=0.15, grasp_z_bonus=-0.2, return_dt=0.18,
                       grasp_extra_in=0.01, top_level_relax=False,
                       obj_xyz_offset=[0.04, 0.02, 0.0]),
    "green_cube": dict(grasp_err_max=0.18, grasp_z_bonus=-0.25, return_dt=0.18,
                       top_level_relax=False,
                       obj_xyz_offset=[0.05, 0.02, 0.0]),
    "milk":       dict(grasp_err_max=0.15, grasp_z_bonus=-0.25,
                       obj_xyz_offset=[0.04, 0.02, 0.0]),
    # snorlax (live 09-02): gate 0.125 — 0.10 rejected grasps that were fine on
    # this soft plush and burned all 3 synth tries; z_bonus -0.3 (a hair lower).
    "snorlax":    dict(grasp_err_max=0.15, grasp_z_bonus=-0.3,
                       obj_xyz_offset=[0.04, 0.02, 0.0]),
    # Small slippery rounds: pinch (2 fingers + thumb backstop) so the wrap
    # doesn't shove them away as it closes. Pinch has fewer contacts so its
    # worst-contact error runs higher than the wrap -> relax the quality gate.
    "tomato":  dict(grasp_mode="pinch", grasp_err_max=0.18),
    "peach":   dict(grasp_mode="pinch", grasp_err_max=0.18),
    # Tall bottle, side grasp = the trial040 config. Over-the-top side approach
    # (handled in synthesis) with a 0.10 standoff and NO finger pre-curl, so the
    # fingers point DOWN and wrap the bottle (pre-curl lifted them to point
    # up/horizontal — the regression). NOTE: do NOT add side_approach_curl,
    # approach_dt, grasp_reach_timeout, grasp_close_extra, or grasp_extra_in here
    # — every one of those was compensating for problems the pre-curl + bigger
    # standoff created. mustard sits on a higher surface (0.96).
    # The side grasp on this tall bottle is IK-marginal (~15/1000 feasible
    # thumb-up seeds) AND BODex is non-deterministic, so a single shot is a coin
    # flip between a solid 5/6-contact wrap and a junk 2/6 one (the "fingers point
    # up / far" failures). Best-of-N: synthesize up to 6 times (fresh seed each)
    # and keep the TIGHTEST grasp, stopping once one passes the 0.16 gate.
    "mustard": dict(obj_xyz_offset=[0.02, 0.0, 0.0],  # baked from live 08-27
                    # edge press-in DISABLED (live 09-02): the inner zone is in
                    # the ARM frame, so the middle of the table is x<0.05 =
                    # "EDGE" -> the 0.08 press-in swapped in there and SHOVED
                    # the bottle over before the fingers closed. One press-in
                    # everywhere (0.025) is the robust behaviour; re-enable only
                    # if far placements start landing short/high again.
                    grasp_extra_in=0.025, grasp_extra_in_edge=0.0,
                    stagger_close_s=0.3,
                    # z_bonus -0.25 (live 09-01): nudge the wrap DOWN off the
                    # neck onto the body. mustard must never get a POSITIVE
                    # bonus (it then grabs the narrow neck and the bottle tips).
                    # thumb_up_min 0.85 (~32deg of vertical) instead of the 0.65
                    # default: 0.65 allowed a ~49deg thumb, which did not read as
                    # a thumb-up side grasp.
                    grasp_z_bonus=-0.25, thumb_up_min=0.85,
                    place_z_offset=0.0,   # side: keep the higher goal
                    approach="side", thumb_up_strict=True,
                    # 0.18 gate: the FORWARD-facing side grasps (fingers wrapping
                    # down, trial040 look — enforced by the side_forward gate in
                    # bodex_plan_cli) run a touch looser than the excluded
                    # "cocked-up" ones, so accept up to 0.18; best-of-N keeps the
                    # tightest forward wrap.
                    synth_retries=8, grasp_err_max=0.15,
                    min_feasible=3, approach_standoff=0.08,
                    side_approach_curl=0.0, table_z=0.96,
                    # Release the bottle from a HIGHER z so the arm never dips to
                    # the table: open at the HOVER (no settle-down) instead of
                    # lowering 30% toward the goal. The side palm sits at the
                    # object's mid-height, so settling drops a forearm/wrist link
                    # toward the table — open high and let the object drop the short
                    # gap. Raise toward ~0.15 live if the bottle tips on landing.
                    release_settle_frac=0.0),
    # Tall containers (500ml can / 500ml PET bottle) — SAME side-grasp recipe as
    # mustard: forward side wrap (fingers point DOWN, NO pre-curl), 0.08 standoff,
    # strict thumb-up, and best-of-N synth (synth_retries) because tall side
    # grasps are IK-marginal AND BODex is non-deterministic. Unlike mustard these
    # sit on the STANDARD table (0.92, BASELINE), NOT the raised 0.96 surface, so
    # no table_z override. green_can is round/hard (sim 4/5, grasp-miss profile
    # like whale/pikachu) -> looser 0.15 gate; tea_peach was clean (sim 5/5) ->
    # 0.12 like mustard.
    # UNLIKE mustard these two DO need side_approach_curl: they are SHORTER cylinders
    # (green_can 16.8cm, tea_peach grasped ~mid) so the palm grasps only ~16cm above
    # the table (vs mustard's ~19cm at its neck) and the OPEN down-pointing fingers
    # graze the table during the side approach. Pre-curl the MCP joints 0.4 rad to
    # lift the open fingertips (~4cm) — affects only the approach, NOT the final
    # grasp wrap. mustard keeps 0.0 (it grasps high enough; curl there made the
    # fingers cock up). Tune live: lower if fingers cock up, raise if they still graze.
    # grasp_z_bonus: bias the grasp HIGHER (these have NO neck, so a higher body
    # wrap is clean and keeps the open fingers off the table). OPT-IN here only;
    # mustard must NOT get it (default 0) or it grabs mustard's narrow neck and the
    # fingers cock up + the bottle tips. Tune live: raise (6-10) to grab higher.
    # green_can (live 08-27): curl DROPPED to 0.0 (was 0.4) and stagger 0.0
    # (synchronized close) — proven this way in the session.
    "green_can": dict(approach="side", thumb_up_strict=True,
                      synth_retries=8, grasp_err_max=0.18,
                      min_feasible=3, approach_standoff=0.08,
                      side_approach_curl=0.0, grasp_z_bonus=4.0,
                      thumb_up_min=0.85,   # stricter thumb-up (live 09-01)
                      place_z_offset=0.0,  # side: keep the higher goal
                      # live 09-02: z_bonus 3->4 and table_z 0.94 (2cm above the
                      # real 0.92) BOTH lift the grasp further up this short
                      # neckless can so the open down-fingers clear the table;
                      # 0.015 press-in then seats the palm on the body. table_z
                      # here is a deliberate grasp-height bias, NOT a measurement
                      # — it also raises the guard's table model, i.e. more
                      # conservative, never less.
                      grasp_extra_in=0.015, table_z=0.94,
                      obj_xyz_offset=[0.0, 0.0, 0.0], stagger_close_s=0.0,
                      release_settle_frac=0.0),   # open at hover, arm never dips
    "tea_peach": dict(approach="side", thumb_up_strict=True,
                      synth_retries=8, grasp_err_max=0.18,
                      min_feasible=3, approach_standoff=0.08,
                      side_approach_curl=0.4, grasp_z_bonus=2.5,
                      grasp_extra_in_edge=0.04, place_z_offset=0.0,
                      obj_xyz_offset=[0.0, 0.0, 0.0], stagger_close_s=0.3,
                      release_settle_frac=0.0),   # open at hover, arm never dips
}

TARGET_TRIALS = 10   # per-object goal shown in the menu (10 for the 09-03 demo
                     # session; was 20). Display/target only — it does not cap
                     # how many trials you can run, and no-grasp aborts still
                     # do not count toward it.

CSV_COLUMNS = [
    "timestamp", "object_id", "active_arm", "trial_idx", "outcome",
    "approach_used", "grasp_error", "feasible_count", "feasible_total",
    "obj_x", "obj_y", "obj_z", "place_x", "place_y", "place_z",
    "grasp_idx", "release_idx", "peak_vel_frac", "duration_s",
    "key_pressed", "aborted", "video", "note",
]


# --------------------------------------------------------------------------
# Single-key terminal reader (stdlib only; no pynput/keyboard in this env)
# --------------------------------------------------------------------------
class RawKeyReader:
    """Context manager for non-blocking single-key reads. Uses cbreak (NOT raw)
    so Ctrl-C still delivers SIGINT; restores the terminal on exit and on SIGINT.
    Degrades to plain stdin reads when stdin is not a TTY (piped --mock tests)."""

    def __enter__(self):
        self.isatty = sys.stdin.isatty()
        self._old = None
        self._old_sigint = None
        if self.isatty:
            self.fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            self._old_sigint = signal.signal(signal.SIGINT, self._on_sigint)
        return self

    def get_key(self, timeout=0.05):
        """Return one character, or None if nothing arrived within `timeout`."""
        try:
            r, _, _ = select.select([sys.stdin], [], [], timeout)
        except Exception:
            return None
        if not r:
            return None
        ch = sys.stdin.read(1)
        return ch if ch else None

    def __exit__(self, *exc):
        if self.isatty and self._old is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._old)
            if self._old_sigint is not None:
                signal.signal(signal.SIGINT, self._old_sigint)
        return False

    def _on_sigint(self, *a):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self._old)
        raise KeyboardInterrupt


def read_one_key(prompt=None):
    """Block until a single key, restoring the terminal afterward."""
    if prompt:
        print(prompt, flush=True)
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        return (line.strip()[:1] or "").lower()
    with RawKeyReader() as kr:
        while True:
            k = kr.get_key(0.2)
            if k is not None:
                return k.lower()


# --------------------------------------------------------------------------
# Per-object args
# --------------------------------------------------------------------------
def build_args(object_id, active_arm, cli, overrides=None):
    """A field-complete argparse Namespace for run_single_trial. Precedence:
    runner defaults -> BASELINE -> PER_OBJECT -> launch-time CLI overrides ->
    runtime `overrides` (the menu `set` command: {'_global': {...},
    '<object>': {...}}, highest) -> forced fields."""
    args = _parse_args([])                       # every field with its default
    for k, v in BASELINE.items():
        setattr(args, k, v)
    for k, v in PER_OBJECT.get(object_id, {}).items():
        setattr(args, k, v)

    args.active_arm = active_arm
    args.object_id = object_id
    args.object_pose_topic = f"/{object_id}_pose"
    args.object_pose = None                       # always perceive on hardware
    args.arm_iface = "forward_position"           # abort only works here
    args.mock = bool(cli.mock)
    args.record = bool(cli.record)

    if cli.table_z is not None:
        args.table_z = cli.table_z
    if cli.pose_correction is not None:
        args.pose_correction = cli.pose_correction
    if cli.obj_xyz_offset is not None:
        args.obj_xyz_offset = list(cli.obj_xyz_offset)
    if cli.grasp_close_extra is not None:
        args.grasp_close_extra = cli.grasp_close_extra
    if cli.seed_num is not None:
        args.seed_num = cli.seed_num
    # Runtime overrides (menu `set`) win over everything: global then per-object.
    if overrides:
        for k, v in overrides.get("_global", {}).items():
            setattr(args, k, v)
        for k, v in overrides.get(object_id, {}).items():
            setattr(args, k, v)
    # speed_scale: one intuitive speed knob. Applied LAST (after overrides) by
    # dividing the waypoint spacing dt (and the per-phase approach/return dt if
    # set). Clamped so nobody can command an extreme motion; the velocity gate
    # + auto_slow in run_single_trial still audit the scaled dt and slow back
    # down if a joint would exceed its UR limit.
    scale = float(getattr(args, "speed_scale", 1.0) or 1.0)
    if scale != 1.0:
        clamped = max(0.5, min(scale, 1.5))
        if clamped != scale:
            print(f"[bench] speed_scale {scale} clamped to {clamped}")
        args.dt = float(args.dt) / clamped
        if getattr(args, "approach_dt", None) and args.approach_dt > 0:
            args.approach_dt = float(args.approach_dt) / clamped
        if getattr(args, "return_dt", None) and args.return_dt > 0:
            args.return_dt = float(args.return_dt) / clamped
    args._pose_corr = _load_pose_correction(args.pose_correction)
    return args


def _xyz(val):
    """Coerce 'x,y,z' (comma-separated, no spaces) into [x, y, z] floats."""
    parts = [float(s) for s in str(val).split(",")]
    if len(parts) != 3:
        raise ValueError(f"need 3 comma-separated floats, got {len(parts)}")
    return parts


def _zone(val):
    """Coerce 'ARM,XMIN,XMAX,YMIN,YMAX' (no spaces) into {arm: [4 floats]} —
    the random-place area override for that arm. 'off' -> None (back to the
    built-in PLACE_ZONES)."""
    s = str(val).strip().lower()
    if s in ("off", "none", "0"):
        return None
    parts = s.split(",")
    if len(parts) != 5:
        raise ValueError("need ARM,XMIN,XMAX,YMIN,YMAX "
                         "(e.g. ra,0,0.25,0.45,0.58) or 'off'")
    arm = parts[0]
    if arm not in ("ra", "la"):
        raise ValueError(f"arm must be 'ra' or 'la', got {arm!r}")
    x_lo, x_hi, y_lo, y_hi = (float(v) for v in parts[1:])
    if not (x_lo < x_hi and y_lo < y_hi):
        raise ValueError("need XMIN<XMAX and YMIN<YMAX")
    return {arm: [x_lo, x_hi, y_lo, y_hi]}


# Fields the menu `set` command may change, with a coercion for each so typed
# strings become the right type. Anything here is settable globally or per-object.
TUNABLE = {
    "obj_xyz_offset": _xyz,     # world-frame pose nudge, e.g. 0.05,-0.02,0
                                # (overrides BASELINE/PER_OBJECT AND the launch flag)
    "place_xyz": _xyz,          # FIXED place goal x,y,z (z=0 -> auto per-object
                                # height). Disables the random place sampler.
                                # e.g. set place_xyz 0.55,0.30,0
                                # back to RANDOM placing: set place_xyz 0,0,0
    "place_z_offset": float,    # raise ONLY the drop height (m); grasp untouched.
                                # e.g. set green_can place_z_offset 0.015
    "place_zone": _zone,        # random-place AREA per arm: ARM,XMIN,XMAX,YMIN,YMAX
                                # (world m). Replaces the built-in zone; bypasses
                                # place_y_min. e.g. set place_zone ra,0,0.25,0.45,0.58
                                # back to the built-in zone: set place_zone off
    "speed_scale": float,       # arm speed: >1 faster, <1 slower (clamp 0.5-1.5)
    "grasp_err_max": float, "min_feasible": int, "seed_num": int,
    "retry_seed_num": int, "synth_retries": int, "allow_marginal": bool,
    "grasp_close_extra": float, "table_z": float, "snap_z_to_table": bool,
    "release_settle_frac": float, "release_open_dwell_s": float,
    "grasp_settle_s": float,
    "top_level_min": float,     # TOP: min cos(palm, straight-down) to count
                                # as level. 0 = built-in 0.85 (~32deg).
                                # 0.95 ~ 18deg for low objects.
    "side_face_z_min": float,   # SIDE: lower bound on palm tilt (face_z).
                                # -1.0 = no floor; -0.25 = near-level palm.
    "transit_hover_z": float,   # how high above the goal the transit passes
                                # (m). Highest pose of the place phase.
    "post_grasp_lift_z": float,  # straight-up +Z lift after the grasp, before
                                # the transit (m). Higher = more clearance,
                                # lower = better lift-IK feasibility.
    "grasp_close_ramp_s": float, "grasp_reach_timeout": float,
    "stagger_close_s": float,   # signed: >0 thumb-first, <0 thumb-last, 0 sync
    "dt": float, "approach_dt": float, "place_y_min": float,
    "return_dt": float,         # per-waypoint time for PHASE 3 only (post-grasp:
                                # transit -> descend -> release -> home). Defaults
                                # to dt. LOWER = faster return. The safest phase to
                                # speed up (object already grasped, arm above the
                                # table). speed_scale still divides it, and the
                                # velocity gate + auto_slow still audit the result.
    "approach": str,   # auto|top|side|side_front|side_left|side_right
    "grasp_mode": str,          # wrap|pinch (pinch = 2-finger + thumb backstop)
    "thumb_up_min": float,      # side grasps: min cos(thumb, world-up)
    "thumb_up_strict": bool,    # side grasps: HARD thumb-up (else no grasp)
    "auto_slow_max_dt": float,  # velocity-gate fail -> auto-raise dt up to this
    "approach_standoff": float,  # straight-line approach standoff (m)
    "side_approach_curl": float,  # side: MCP pre-curl so fingers clear the table
    "grasp_extra_in": float,     # press the palm this far PAST the grasp (m)
    "grasp_extra_in_edge": float,  # press-in used ONLY outside the inner zone
    "edge_x_min": float, "edge_x_max": float,   # inner zone x bounds (m, world)
    "edge_y_min": float, "edge_y_max": float,   # inner zone |y| bounds (ra frame)
    "grasp_z_bonus": float,      # side: bias grasp selection toward HIGHER grasps
    "top_front_min": float,      # top: fingers must point FRONT (cos, 0=off, 0.7~±45°)
    "top_level_relax": bool,     # top: allow tilted fallback on hard objects (default 1)
    "grasp_body_bias": float,    # bias palm along object long axis: + = mesh +Y end
                                 # (whale: head), - = -Y end (tail). 0 = off
    "grasp_axis_align": float,   # prefer knuckle-line PARALLEL to object long axis
                                 # (long-axis wrap, max surface). ~0.5 = strong. 0=off
    "grasp_center_bias": float,  # penalize palm offset from object CENTER along the
                                 # body axis (fat mid-body, 180°-flip immune). 0=off
}

_VALID_APPROACHES = {"auto", "top", "side", "side_front",
                     "side_left", "side_right"}


def _coerce(key, val):
    typ = TUNABLE.get(key, str)
    if typ is bool:
        return str(val).strip().lower() in ("1", "true", "yes", "on", "y")
    return typ(val)


def apply_set_command(parts, overrides):
    """Parse a menu `set` command into the overrides dict. Forms:
        set <key> <val>            -> global override
        set <object> <key> <val>   -> per-object override
    Returns a status string to print."""
    if len(parts) == 3:
        scope, key, val = "_global", parts[1], parts[2]
    elif len(parts) == 4 and parts[1] in BENCH_OBJECTS:
        scope, key, val = parts[1], parts[2], parts[3]
    else:
        return ("usage: set <key> <val>   |   set <object> <key> <val>\n"
                f"       keys: {', '.join(sorted(TUNABLE))}")
    if key not in TUNABLE:
        return f"unknown key '{key}'. settable: {', '.join(sorted(TUNABLE))}"
    try:
        cval = _coerce(key, val)
    except ValueError:
        return f"bad value '{val}' for {key} ({TUNABLE[key].__name__})"
    if key == "approach" and cval not in _VALID_APPROACHES:
        return f"bad approach '{cval}'. choices: {', '.join(sorted(_VALID_APPROACHES))}"
    if key == "grasp_mode" and cval not in ("wrap", "pinch"):
        return f"bad grasp_mode '{cval}'. choices: wrap, pinch"
    overrides.setdefault(scope, {})[key] = cval
    where = "global" if scope == "_global" else scope
    return f"[set] {where}: {key} = {cval}"


def show_object_cfg(obj, active_arm, cli, overrides, key=None):
    """Print the EFFECTIVE (fully merged) config an object would run with right
    now: runner defaults -> BASELINE -> PER_OBJECT -> CLI -> live `set`s. This
    is exactly the Namespace build_args produces for the next trial."""
    args = build_args(obj, active_arm, cli, overrides)
    keys = [key] if key else sorted(set(TUNABLE) | {"obj_xyz_offset"})
    print(f"[bench] effective config for {obj} (what the NEXT trial runs):")
    for k in keys:
        if hasattr(args, k):
            print(f"    {k:<22} = {getattr(args, k)}")
        else:
            print(f"    {k:<22} (no such field)")


def show_overrides(overrides):
    if not any(overrides.values()):
        print("[bench] no runtime overrides set "
              "(using BASELINE + PER_OBJECT).")
        return
    print("[bench] runtime overrides:")
    for scope in sorted(overrides):
        if overrides[scope]:
            where = "global" if scope == "_global" else scope
            kv = ", ".join(f"{k}={v}" for k, v in overrides[scope].items())
            print(f"    {where:<10} {kv}")


# --------------------------------------------------------------------------
# Recovery: drop the object and return home from an arbitrary aborted pose
# --------------------------------------------------------------------------
def _open_hand_pose():
    h = np.zeros(16, dtype=float)
    h[12] = THUMB_J12_MIN          # thumb-lift can't retract to 0 on the real hand
    return h


def return_home_open(io, home_arm, *, dur=3.0, stream_hz=125.0, mock_arm=False):
    """Bring the arm home with the fingers open from wherever a trial stopped:
      1) ramp the hand OPEN in place (so the object drops before any travel),
      2) lift the shoulder toward home for table clearance,
      3) slow lerp cur_arm -> home_arm with the hand open.
    Not collision-checked; kept slow + open-first for safety (attended e-stop)."""
    home_arm = np.asarray(home_arm, dtype=float).reshape(6)
    period = 1.0 / max(stream_hz, 1.0)
    try:
        cur_arm = np.asarray(io.active_arm_q(), dtype=float)[:6]
    except Exception:
        cur_arm = home_arm.copy()
    try:
        cur_hand = np.asarray(io.active_hand_q(), dtype=float)[:16]
    except Exception:
        cur_hand = _open_hand_pose()
    open_hand = _open_hand_pose()

    def _stream(q0, q1, h0, h1, seconds, n):
        ts = np.asarray([k * seconds / n for k in range(n + 1)])
        qs = np.stack([
            np.concatenate([q0 * (1 - k / n) + q1 * (k / n),
                            h0 * (1 - k / n) + h1 * (k / n)])
            for k in range(n + 1)])
        io._stream_knots(ts, qs, period, mock_arm)

    # 1) open the hand in place (object drops).
    _stream(cur_arm, cur_arm, cur_hand, open_hand, 0.7, 14)
    # 2) lift the shoulder toward home for clearance, hand open.
    lifted = release_lift_target(cur_arm, home_arm, 0.5)
    _stream(cur_arm, lifted, open_hand, open_hand, max(1.0, dur * 0.4),
            max(10, int(dur * 0.4 * stream_hz / 6)))
    # 3) slow lerp to home, hand open.
    _stream(lifted, home_arm, open_hand, open_hand, dur,
            max(20, int(dur * stream_hz / 6)))


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------
def open_csv(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    new = (not os.path.exists(path)) or os.path.getsize(path) == 0
    fh = open(path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    if new:
        writer.writeheader()
        fh.flush()
    return fh, writer


def load_counts(paths):
    """Per-object running tallies for the menu, read back from one CSV path or a
    LIST of CSV paths (aggregated, for the 'all' load option). Returns
    {obj: {succ, att, nofail}}."""
    if isinstance(paths, (str, bytes, os.PathLike)):
        paths = [paths]
    counts = {obj: {"succ": 0, "att": 0, "nofail": 0} for obj in BENCH_OBJECTS}
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    obj = row.get("object_id")
                    if obj not in counts:
                        counts.setdefault(obj, {"succ": 0, "att": 0, "nofail": 0})
                    out = (row.get("outcome") or "").strip()
                    if out in ("success", "fail"):
                        counts[obj]["att"] += 1
                        if out == "success":
                            counts[obj]["succ"] += 1
                    else:
                        counts[obj]["nofail"] += 1
        except Exception as e:
            print(f"[bench] WARN: could not read counts from {path}: {e}")
    return counts


def write_row(fh, writer, *, args, obj, trial_idx, outcome, result, key, dur,
              aborted, video):
    op = result.get("obj_pose")
    pl = result.get("place_xyz")
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "object_id": obj,
        "active_arm": args.active_arm,
        "trial_idx": trial_idx,
        "outcome": outcome,
        "approach_used": result.get("approach_used", ""),
        "grasp_error": _fmt(result.get("grasp_error")),
        "feasible_count": result.get("feasible_count", ""),
        "feasible_total": result.get("feasible_total", ""),
        "obj_x": _fmt(op[0] if op is not None else None),
        "obj_y": _fmt(op[1] if op is not None else None),
        "obj_z": _fmt(op[2] if op is not None else None),
        "place_x": _fmt(pl[0] if pl is not None else None),
        "place_y": _fmt(pl[1] if pl is not None else None),
        "place_z": _fmt(pl[2] if pl is not None else None),
        "grasp_idx": result.get("grasp_idx", ""),
        "release_idx": result.get("release_idx", ""),
        "peak_vel_frac": _fmt(result.get("peak_vel_frac")),
        "duration_s": f"{dur:.2f}",
        "key_pressed": key or "",
        "aborted": bool(aborted),
        "video": video or "",
        "note": result.get("note", ""),
    }
    writer.writerow(row)
    fh.flush()
    os.fsync(fh.fileno())


def _fmt(v):
    if v is None:
        return ""
    try:
        f = float(v)
        if f != f:   # NaN
            return ""
        return f"{f:.4f}"
    except (TypeError, ValueError):
        return ""


# --------------------------------------------------------------------------
# Mock (no robot / no ROS) so the UX + CSV + abort can be tested offline
# --------------------------------------------------------------------------
class MockIO:
    def __init__(self, home_arm):
        self._home = np.asarray(home_arm, dtype=float).reshape(6)

    def set_object_topic(self, topic):  # noqa: D401
        pass

    def active_arm_q(self):
        return self._home.copy()

    def active_hand_q(self):
        return np.zeros(16, dtype=float)

    def _stream_knots(self, kt, kq, period, mock_arm, stop_evt=None):
        time.sleep(min(float(kt[-1]) if len(kt) else 0.0, 0.4))

    def start(self):
        pass

    def shutdown(self):
        pass


def mock_single_trial(stop_evt, object_id, active_arm):
    """Fake ~4 s motion honoring stop_evt; returns run_single_trial-shaped meta."""
    print(f"[mock] synthesizing + 'moving' for {object_id} ({active_arm}) ...")
    h = abs(hash(object_id)) % 1000 / 1000.0
    t0 = time.monotonic()
    while time.monotonic() - t0 < 4.0:
        if stop_evt is not None and stop_evt.is_set():
            break
        time.sleep(0.05)
    aborted = bool(stop_evt is not None and stop_evt.is_set())
    return {
        "outcome": "ok", "aborted": aborted, "note": "",
        "approach_used": "top",
        "grasp_error": 0.06 + 0.08 * h,
        "feasible_count": 30 + int(200 * h), "feasible_total": 300,
        "obj_pose": np.array([0.45, -0.10 + 0.1 * h, 0.82, 1, 0, 0, 0]),
        "place_xyz": np.array([0.45, 0.45, 0.79]),
        "grasp_idx": 30, "release_idx": 90, "peak_vel_frac": 0.4 + 0.2 * h,
        "replay_ok": not aborted, "auto_success": None, "place_err_xy": None,
    }


def _warmup_synth(active_arm):
    """One small throwaway synthesis (64 seeds, canned pose) so the worker
    JIT-compiles its CUDA kernels / builds its solvers before the first REAL
    trial. Runs in a daemon thread at startup; a real trial started mid-warmup
    simply queues behind it on the worker (a few seconds, one time). Any
    failure is logged and ignored — warm-up is purely an optimization."""
    try:
        from bodex_service import BODexClient
        from bodex_hw.run_bodex_pick_place_hw import _synthesize_once
        entry = resolve_object_entry("red_cube", active_arm)
        args = _parse_args([])
        args.active_arm = active_arm
        args.object_id = "red_cube"
        args.table_z = BASELINE["table_z"]
        z = float(BASELINE["table_z"]) + float(entry["obb_z_world"]) / 2.0
        obj_pose = np.array([0.30, 0.45, z, 1.0, 0.0, 0.0, 0.0])
        place_xyz = np.array([0.30, 0.50, z])
        client = BODexClient(socket_path="/tmp/bodex_service.sock",
                             auto_start=True, verbose=False)
        t0 = time.monotonic()
        _synthesize_once(client, args, obj_pose, place_xyz, entry,
                         None, None, "top", 64, 0.65)
        print(f"\n[bench] worker warm-up finished in "
              f"{time.monotonic() - t0:.1f}s — first real synthesis will be "
              f"fast now.")
    except Exception as e:  # noqa: BLE001 — warm-up must never kill the bench
        print(f"\n[bench] warm-up skipped ({type(e).__name__}: {e})")


# --------------------------------------------------------------------------
# Menu + trial loop
# --------------------------------------------------------------------------
def print_menu(active_arm, counts, disabled):
    print("\n" + "=" * 58)
    print(f"  BODex pick-and-place benchmark   arm={active_arm}")
    print("=" * 58)
    for i, obj in enumerate(BENCH_OBJECTS, 1):
        if obj in disabled:
            print(f"  {i:>2}) {obj:<12}  -- unavailable ({disabled[obj]})")
            continue
        c = counts[obj]
        att, succ, nf = c["att"], c["succ"], c["nofail"]
        rate = f"{100 * succ // att}%" if att else "--"
        extra = f"  (+{nf} no-grasp)" if nf else ""
        print(f"  {i:>2}) {obj:<12}  [{succ}/{att} of {TARGET_TRIALS}, {rate}]{extra}")
    # ---- overall totals -------------------------------------------------
    # Pooled rate over every attempted trial (sum successes / sum attempts),
    # plus the unweighted mean of the per-object rates. They differ whenever
    # the objects have different attempt counts, so both are shown. no-grasp
    # aborts are excluded from `att` and so from both figures, exactly as in
    # the per-object rows.
    t_att = sum(counts[o]["att"] for o in BENCH_OBJECTS if o not in disabled)
    t_succ = sum(counts[o]["succ"] for o in BENCH_OBJECTS if o not in disabled)
    t_nf = sum(counts[o]["nofail"] for o in BENCH_OBJECTS if o not in disabled)
    rates = [100.0 * counts[o]["succ"] / counts[o]["att"]
             for o in BENCH_OBJECTS
             if o not in disabled and counts[o]["att"]]
    print("  ------------------------------------------------------")
    if t_att:
        mean_txt = (f"   mean per-object {sum(rates) / len(rates):.1f}%"
                    f" (n={len(rates)})" if rates else "")
        nf_txt = f"  (+{t_nf} no-grasp)" if t_nf else ""
        print(f"   TOTAL  {t_succ}/{t_att}  "
              f"{100.0 * t_succ / t_att:.1f}%{nf_txt}")
        print(f"  {mean_txt}")
    else:
        print("   TOTAL  no trials yet")
    print("  ------------------------------------------------------")
    print("   number = pick object    q = quit (saves)")
    print("   set <key> <val> | set <obj> <key> <val> | show   (live tuning)")
    print("   cfg <obj> [key]  = show the EFFECTIVE config an object runs with")
    print("   e.g.  set mustard grasp_extra_in 0.015   |   cfg mustard grasp_extra_in")


def run_one_trial(worker_factory, io, home_arm, mock):
    """Launch one trial worker, listen for s/f, abort+recover on a key. Returns
    (result_dict, key, duration_s, aborted, no_motion)."""
    stop_evt = threading.Event()
    result = {}

    def worker():
        try:
            result.update(worker_factory(stop_evt))
        except Exception as e:   # keep the loop alive on any worker crash
            result.update({"outcome": "error",
                           "note": f"{type(e).__name__}: {e}"})

    t0 = time.monotonic()
    th = threading.Thread(target=worker, daemon=True)
    th.start()

    key = None
    no_motion = False
    aborted = False
    interrupted = False
    try:
        with RawKeyReader() as kr:
            # Phase A: motion in flight — listen for s/f.
            while th.is_alive():
                k = kr.get_key(0.05)
                if k and k.lower() in ("s", "f"):
                    key = k.lower()
                    stop_evt.set()
                    print(f"\n[bench] '{key}' -> stopping motion, returning home...")
                    break
            if key is None:
                th.join()
                if result.get("outcome") != "ok":
                    no_motion = True        # synth_fail / error: no s/f prompt
                else:
                    print("\n[bench] motion finished — press 's' (success) or "
                          "'f' (fail):")
                    while key not in ("s", "f"):
                        kk = kr.get_key(0.2)
                        if kk:
                            key = kk.lower()
            else:
                th.join(timeout=15.0)
                aborted = True
    except KeyboardInterrupt:
        # Ctrl-C during a trial: stop the motion and home the arm BEFORE quitting
        # so it never freezes mid-trajectory. This trial is discarded (no row).
        interrupted = True
        no_motion = True
        stop_evt.set()
        if th.is_alive():
            th.join(timeout=15.0)
        print("\n[bench] Ctrl-C — stopping motion, returning home "
              "(this trial is discarded)...")

    dur = time.monotonic() - t0
    if not no_motion or interrupted:
        try:
            return_home_open(io, home_arm, dur=(3.0 if (aborted or interrupted)
                                                else 1.2), mock_arm=mock)
        except Exception as e:
            print(f"[bench] WARN: return_home_open failed: {e}")
    if interrupted:
        raise KeyboardInterrupt   # propagate so the session quits + saves cleanly
    return result, key, dur, aborted, no_motion


def _csv_summary(path):
    """One-line summary of a results CSV for the startup chooser."""
    try:
        att = succ = nofail = 0
        objs = set()
        last = ""
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                out = (row.get("outcome") or "").strip()
                if out in ("success", "fail"):
                    att += 1
                    objs.add(row.get("object_id"))
                    if out == "success":
                        succ += 1
                else:
                    nofail += 1
                last = row.get("timestamp", last) or last
        rate = f"{100 * succ // att}%" if att else "--"
        extra = f", +{nofail} no-grasp" if nofail else ""
        return (f"{att} trials ({rate}) over {len(objs)} objects{extra}"
                f"{(', last ' + last) if last else ''}")
    except Exception as e:
        return f"(unreadable: {e})"


def choose_results_file(cli):
    """At startup, offer to LOAD prior results (continue counts) or start NEW.
    Returns (write_path, load_paths): trials are appended to write_path; the menu
    scores aggregate over load_paths. Options: a number (load+continue that file),
    'all' (aggregate ALL prior files, append to --csv), 'n'/Enter (new session).
    Skips the prompt when --fresh or no prior CSV exists."""
    target = os.path.abspath(cli.csv)
    target_dir = os.path.dirname(target) or "."

    # Candidate prior CSVs: the --csv file itself + every *.csv beside it.
    cands, seen = [], set()
    for p in [target] + (
        [os.path.join(target_dir, fn) for fn in sorted(os.listdir(target_dir))]
        if os.path.isdir(target_dir) else []
    ):
        ap = os.path.abspath(p)
        if (ap not in seen and ap.endswith(".csv")
                and os.path.isfile(ap) and os.path.getsize(ap) > 0):
            seen.add(ap)
            cands.append(ap)

    def _new_path():
        # Don't clobber an existing --csv: timestamp the new file.
        if os.path.exists(target) and os.path.getsize(target) > 0:
            base, ext = os.path.splitext(target)
            np_ = f"{base}_{datetime.now():%Y%m%d_%H%M%S}{ext}"
            print(f"[bench] --csv exists; new session -> {np_}")
            return np_
        return target

    if cli.fresh or not cands:
        return (_new_path() if cli.fresh else target), []

    print("\nPrior result files found:")
    for i, p in enumerate(cands, 1):
        print(f"  {i:>2}) {p}")
        print(f"        {_csv_summary(p)}")
    print("  all) load ALL of them (aggregate counts), append to "
          f"{os.path.basename(target)}")
    print(f"    n) start a NEW session   (-> {target})")
    try:
        sel = input("load? [number / all / n / Enter=new] > ").strip().lower()
    except EOFError:
        sel = ""
    if sel == "all":
        print(f"[bench] aggregating {len(cands)} files; appending new trials to "
              f"{target}")
        return target, cands
    if sel.isdigit() and 1 <= int(sel) <= len(cands):
        chosen = cands[int(sel) - 1]
        print(f"[bench] continuing prior results: {chosen}")
        return chosen, [chosen]
    return _new_path(), []


def parse_cli(argv=None):
    p = argparse.ArgumentParser(description="Interactive BODex pick-and-place benchmark")
    p.add_argument("--active_arm", choices=["ra", "la"], default="ra",
                   help="arm used for the whole session")
    p.add_argument("--csv", default=os.path.expanduser(
        "~/Dualarm_ws/BODex/benchmark_results/pick_place_benchmark.csv"),
                   help="results CSV (appended; resumes counts)")
    p.add_argument("--fresh", action="store_true",
                   help="skip the 'load prior results' prompt and start a new "
                        "CSV (timestamped if --csv already exists)")
    p.add_argument("--table_z", type=float, default=None,
                   help="measured table-top Z (override BASELINE 0.91)")
    p.add_argument("--pose_correction", default=None,
                   help="affine camera->world correction JSON (see runner §10)")
    p.add_argument("--obj_xyz_offset", type=float, nargs=3, default=None,
                   help="manual world-frame object offset x y z")
    p.add_argument("--place_zone", action="append", nargs=5, default=None,
                   metavar=("ARM", "XMIN", "XMAX", "YMIN", "YMAX"),
                   help="random-place AREA override for one arm ('ra'/'la'): "
                        "world-frame x/y bounds (m) replacing the built-in "
                        "zone (and bypassing place_y_min). Repeatable, once "
                        "per arm. Also live: set place_zone ra,0,0.25,0.45,0.58"
                        " — e.g. --place_zone ra 0.00 0.25 0.45 0.58")
    p.add_argument("--grasp_close_extra", type=float, default=None)
    p.add_argument("--seed_num", type=int, default=None)
    p.add_argument("--speed_scale", type=float, default=None,
                   help="arm motion speed: >1 faster, <1 slower (clamped to "
                        "0.5-1.5; velocity gate still enforced). Also live: "
                        "set speed_scale 1.2")
    p.add_argument("--warmup", action="store_true",
                   help="fire one small throwaway synthesis at startup so the "
                        "worker JIT-compiles its CUDA kernels before the first "
                        "real trial (removes the extra-slow first synthesis)")
    # Grasp-robustness knobs (also settable live in the menu with `set`).
    # These seed the GLOBAL runtime overrides at launch.
    p.add_argument("--grasp_err_max", type=float, default=None,
                   help="max worst-contact grasp error to accept (raise for "
                        "hard/rounded objects; whale/tiger preset 0.15/0.11)")
    p.add_argument("--min_feasible", type=int, default=None,
                   help="min IK-feasible seeds required to accept a grasp")
    p.add_argument("--retry_seed_num", type=int, default=None,
                   help="seed count on synthesis retries")
    p.add_argument("--synth_retries", type=int, default=None,
                   help="how many approaches to cycle before giving up")
    p.add_argument("--allow_marginal", action="store_true",
                   help="DRIVE the best grasp even if it fails the quality gate, "
                        "so the robot ALWAYS attempts (you judge s/f). Best way "
                        "to stop pikachu/whale 'no solution' aborts.")
    p.add_argument("--approach", action="append", default=None,
                   metavar="[OBJ=]VALUE",
                   help="force a grasp approach. 'VALUE' = all objects; "
                        "'OBJ=VALUE' = one object (repeatable). VALUE one of: "
                        "auto|top|side|side_front|side_left|side_right. "
                        "e.g. --approach mustard=side")
    p.add_argument("--grasp_mode", action="append", default=None,
                   metavar="[OBJ=]VALUE",
                   help="grasp strategy: wrap|pinch. 'VALUE' = all objects; "
                        "'OBJ=VALUE' = one object (repeatable). pinch = 2-finger "
                        "+ thumb-backstop for slippery rounds. e.g. "
                        "--grasp_mode tomato=pinch")
    p.add_argument("--thumb_up_min", type=float, default=None,
                   help="SIDE grasps: min cos(thumb, world-up) to accept (raise "
                        "to keep the thumb pointing more straight up)")
    p.add_argument("--thumb_up_strict", action="store_true",
                   help="SIDE grasps: HARD thumb-up — no grasp at all unless the "
                        "thumb points up (>= thumb_up_min), so the thumb tip "
                        "never scrapes the table")
    p.add_argument("--approach_standoff", type=float, default=None,
                   help="pre-grasp standoff (m): the hand moves STRAIGHT IN from "
                        "this far back along the grasp axis instead of a curved "
                        "sweep, so it doesn't knock an object under/behind the "
                        "palm. Default 0.10; raise if it still clips, 0 = old "
                        "joint-lerp approach.")
    p.add_argument("--release_open_dwell_s", type=float, default=None,
                   help="after the hand opens at release, HOLD it open this long "
                        "before homing so the Allegro fingers settle (no droop "
                        "back toward the object). Default 1.0.")
    p.add_argument("--record", action="store_true",
                   help="record an mp4 per trial under <record_dir>/<obj>/trialNNN")
    p.add_argument("--record_dir", default=os.path.expanduser(
        "~/Dualarm_ws/BODex/benchmark_results/video"))
    p.add_argument("--mock", action="store_true",
                   help="no robot / no ROS: fake motions to test menu+CSV+keys")
    return p.parse_args(argv)


def seed_overrides_from_cli(cli):
    """Build the runtime-overrides dict ``{'_global': {...}, '<obj>': {...}}`` from
    the grasp-tuning CLI flags. Shared by the benchmark and the reviewer demo so
    both honor the same launch flags; the menu ``set`` command edits it in place."""
    overrides = {"_global": {}}
    for k in ("grasp_err_max", "min_feasible", "retry_seed_num", "synth_retries",
              "speed_scale"):
        if getattr(cli, k, None) is not None:
            overrides["_global"][k] = getattr(cli, k)
    if cli.allow_marginal:
        overrides["_global"]["allow_marginal"] = True
    if cli.thumb_up_min is not None:
        overrides["_global"]["thumb_up_min"] = cli.thumb_up_min
    if cli.thumb_up_strict:
        overrides["_global"]["thumb_up_strict"] = True
    if cli.approach_standoff is not None:
        overrides["_global"]["approach_standoff"] = cli.approach_standoff
    if cli.release_open_dwell_s is not None:
        overrides["_global"]["release_open_dwell_s"] = cli.release_open_dwell_s
    # --place_zone ARM XMIN XMAX YMIN YMAX (repeatable, once per arm): merge all
    # entries into one {arm: bounds} dict so both arms can be pre-configured.
    if getattr(cli, "place_zone", None):
        zones = {}
        for item in cli.place_zone:
            try:
                zones.update(_zone(",".join(str(v) for v in item)))
            except ValueError as e:
                print(f"[bench] WARN: ignoring --place_zone {item}: {e}")
        if zones:
            overrides["_global"]["place_zone"] = zones
    # --approach [OBJ=]VALUE (repeatable): global or per-object grasp approach.
    for item in (cli.approach or []):
        obj, _, val = item.partition("=")
        if not val:                     # bare VALUE -> global
            obj, val = "_global", obj
        if val not in _VALID_APPROACHES:
            print(f"[bench] WARN: ignoring --approach '{item}' "
                  f"(VALUE must be one of {', '.join(sorted(_VALID_APPROACHES))})")
            continue
        if obj != "_global" and obj not in BENCH_OBJECTS:
            print(f"[bench] WARN: ignoring --approach '{item}' "
                  f"(unknown object '{obj}')")
            continue
        overrides.setdefault(obj, {})["approach"] = val
    # --grasp_mode [OBJ=]VALUE (repeatable): global or per-object grasp strategy.
    for item in (cli.grasp_mode or []):
        obj, _, val = item.partition("=")
        if not val:                     # bare VALUE -> global
            obj, val = "_global", obj
        if val not in ("wrap", "pinch"):
            print(f"[bench] WARN: ignoring --grasp_mode '{item}' "
                  f"(VALUE must be wrap or pinch)")
            continue
        if obj != "_global" and obj not in BENCH_OBJECTS:
            print(f"[bench] WARN: ignoring --grasp_mode '{item}' "
                  f"(unknown object '{obj}')")
            continue
        overrides.setdefault(obj, {})["grasp_mode"] = val
    return overrides


def main(argv=None):
    cli = parse_cli(argv)
    active_arm = cli.active_arm
    home_arm = _home_rad(active_arm)

    # Validate objects up front so a missing-geometry object never crashes mid-run.
    disabled = {}
    for obj in BENCH_OBJECTS:
        try:
            resolve_object_entry(obj, active_arm)
        except Exception as e:
            disabled[obj] = str(e).splitlines()[0][:40]

    write_path, load_paths = choose_results_file(cli)
    counts = load_counts(load_paths if load_paths else write_path)
    fh, writer = open_csv(write_path)
    print(f"[bench] logging to {write_path}")
    if cli.record and not cli.mock:
        print(f"[bench] recording video under {cli.record_dir}")
    if disabled:
        print(f"[bench] unavailable (no geometry): {', '.join(disabled)}")

    # Runtime overrides: {'_global': {...}, '<object>': {...}}. Seeded from the
    # grasp-tuning CLI flags; the menu `set` command edits this in place.
    overrides = seed_overrides_from_cli(cli)
    if any(overrides.values()):
        show_overrides(overrides)

    # ROS / IO setup (skipped entirely in --mock).
    io = None
    if cli.mock:
        io = MockIO(home_arm)
    else:
        import rclpy
        from bodex_hw.ros_io import HwIO
        rclpy.init()
        io = HwIO(active_arm=active_arm,
                  object_pose_topic=f"/{BENCH_OBJECTS[0]}_pose",
                  arm_iface="forward_position")
        io.start()
        if cli.warmup:
            print("[bench] firing worker warm-up in the background "
                  "(~one small synthesis)...")
            threading.Thread(target=_warmup_synth, args=(active_arm,),
                             daemon=True).start()

    try:
        while True:
            print_menu(active_arm, counts, disabled)
            sel = input("select> ").strip()
            low = sel.lower()
            if low in ("q", "quit", "exit"):
                break
            if low in ("show", "?"):
                show_overrides(overrides)
                continue
            if low.startswith("set"):
                print(apply_set_command(sel.split(), overrides))
                continue
            if low.startswith("cfg"):
                parts = sel.split()
                if len(parts) >= 2 and parts[1] in BENCH_OBJECTS:
                    show_object_cfg(parts[1], active_arm, cli, overrides,
                                    parts[2] if len(parts) > 2 else None)
                else:
                    print(f"usage: cfg <object> [key]   objects: "
                          f"{', '.join(BENCH_OBJECTS)}")
                continue
            if not low.isdigit() or not (1 <= int(low) <= len(BENCH_OBJECTS)):
                print("[bench] enter a listed number, 'set ...' to tune, "
                      "'show' to list overrides, or q to quit.")
                continue
            sel = low
            obj = BENCH_OBJECTS[int(sel) - 1]
            if obj in disabled:
                print(f"[bench] {obj} is unavailable ({disabled[obj]}).")
                continue

            c = counts[obj]
            trial_idx = c["att"] + c["nofail"] + 1
            entry = resolve_object_entry(obj, active_arm)
            args = build_args(obj, active_arm, cli, overrides)
            video = ""
            if args.record and not cli.mock:
                video = os.path.join(cli.record_dir, obj, f"trial{trial_idx:03d}")
                os.makedirs(video, exist_ok=True)
                args.record_dir = video
            if not cli.mock:
                io.set_object_topic(args.object_pose_topic)

            k = read_one_key(
                f"\n[{obj}] trial {trial_idx}: press 'a' to START "
                f"(any other key cancels)")
            if k != "a":
                print("[bench] cancelled.")
                continue
            print(f"[bench] starting {obj} trial {trial_idx} "
                  f"(s=success, f=fail; stops + homes)...")

            if cli.mock:
                worker_factory = (lambda se, o=obj, a=active_arm:
                                  mock_single_trial(se, o, a))
            else:
                worker_factory = (lambda se, a=args, e=entry:
                                  run_single_trial(io, a, e, home_arm,
                                                   _traj_cfg(a), stop_evt=se))

            result, key, dur, aborted, no_motion = run_one_trial(
                worker_factory, io, home_arm, cli.mock)

            if no_motion:
                outcome = result.get("outcome", "synth_fail")
                c["nofail"] += 1
                print(f"[bench] NO GRASP ({outcome}): "
                      f"{result.get('note', '')} — not counted toward the "
                      f"{TARGET_TRIALS}.")
            else:
                outcome = "success" if key == "s" else "fail"
                c["att"] += 1
                if key == "s":
                    c["succ"] += 1
                print(f"[bench] recorded {obj} trial {trial_idx}: {outcome.upper()} "
                      f"({c['succ']}/{c['att']})")

            write_row(fh, writer, args=args, obj=obj, trial_idx=trial_idx,
                      outcome=outcome, result=result, key=key, dur=dur,
                      aborted=aborted, video=video)
    except KeyboardInterrupt:
        print("\n[bench] interrupted — results saved.")
    finally:
        try:
            fh.close()
        except Exception:
            pass
        if not cli.mock and io is not None:
            try:
                io.shutdown()
            except Exception:
                pass
            try:
                import rclpy
                rclpy.shutdown()
            except Exception:
                pass
        print("[bench] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
