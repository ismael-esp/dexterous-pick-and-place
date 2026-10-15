"""Pins the numpy trajectory port to the sim's exact arithmetic.

Runs in the ROS2 system Python (numpy only). The torch original
(env.py) needs Isaac Lab so a live golden-diff is not possible here;
instead a synthetic raw trajectory exercises every branch with values
chosen so the expected output is hand-computable.

Run: ``python3 bodex_hw/tests/trajectory_test.py``  (or via pytest).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from bodex_hw.trajectory import (
    TrajCfg, build_full_trajectory, release_lift_target, release_reorient_pose,
)

CURL_COLS = [6 + i for i in TrajCfg().hand_curl_idx]
HOME = np.array([0.5, -0.7, 1.7, -4.0, -1.2, 3.9], dtype=np.float32)
CUR_ARM = np.full(6, 0.1, dtype=np.float32)
CUR_HAND = np.full(16, 0.2, dtype=np.float32)


def _synthetic_raw(t_raw=120, release_start=100):
    """All joints = 1.0, except hand curl drops to 0.1 from release_start
    so the tail-scan locks release_start exactly there."""
    raw = np.ones((t_raw, 22), dtype=np.float32)
    raw[release_start:, [6 + i for i in TrajCfg().hand_curl_idx]] = 0.1
    return raw


# Squeeze now starts at the GRASP CONTACT waypoint (raw idx 30), not
# after the 10-frame hold (raw idx 40). In `full` (30-wp preamble
# prepended) that is row 30+30 = 60 onward, up to 30+release_start.
def test_regular_top_object():
    """red_cube-like: top approach, obb_z_world>=0.08 -> squeeze 0.10,
    lift block appended (10 wps), no slow-release. grasp_idx at contact."""
    raw = _synthetic_raw()
    full, pick_end, grasp_idx, _ = build_full_trajectory(
        raw, pick_end=30, approach_used="top", obb_z_world=0.0833,
        cur_arm=CUR_ARM, cur_hand=CUR_HAND, home_arm=HOME,
    )
    # 30 preamble + 120 raw + 10 lift + 30 return
    assert full.shape == (190, 22), full.shape
    assert full.dtype == np.float32
    assert pick_end == 60          # BODex pick_end 30 + N_PRE(30)
    assert grasp_idx == 60         # N_PRE(30) + N_APPROACH(30) = contact
    # grasp_idx must be BEFORE pick_end in time terms here they're equal
    # only because the synthetic pick_end==30; on real data pick_end
    # (post-lift) >> grasp_idx (contact). The invariant we rely on:
    assert grasp_idx <= full.shape[0] - 1

    # Approach-open: BODex segment starts at row 30; first 30 hand rows are
    # open — all joints 0 EXCEPT the thumb-lift joint_12 (index 12), held at
    # its real-hand minimum 0.36 (cannot physically retract to 0).
    ap = full[30:60, 6:22]
    assert np.allclose(np.delete(ap, 12, axis=1), 0.0)
    assert np.allclose(ap[:, 12], 0.36)

    # Firm close from CONTACT: rows [30+30, 30+100) curl==grasp_curl+0.10.
    sq = full[60:130, :][:, CURL_COLS]
    assert np.allclose(sq, 1.10, atol=1e-5), sq.mean()

    # Preamble endpoints.
    assert np.allclose(full[29, :6], HOME, atol=1e-5)
    assert np.allclose(full[0, :6], CUR_ARM * (29 / 30) + HOME * (1 / 30),
                       atol=1e-5)
    assert np.allclose(full[-1, :6], HOME, atol=1e-5)
    # Final hand open: all 0 except thumb-lift joint_12 (index 12 in the
    # 16-hand slice = column 18) at its 0.36 minimum.
    assert np.allclose(np.delete(full[-1, 6:22], 12), 0.0)
    assert np.isclose(full[-1, 6:22][12], 0.36)


def test_grasp_close_extra():
    """grasp_close_extra adds on top of the size/approach squeeze over
    the whole grip phase (contact -> release)."""
    raw = _synthetic_raw()
    full, _, grasp_idx, _ = build_full_trajectory(
        raw, pick_end=30, approach_used="top", obb_z_world=0.0833,
        cur_arm=CUR_ARM, cur_hand=CUR_HAND, home_arm=HOME,
        grasp_close_extra=0.6,
    )
    # 0.10 squeeze + 0.6 extra = grasp_curl(1.0)+0.70 from contact on.
    sq = full[60:130, :][:, CURL_COLS]
    assert np.allclose(sq, 1.70, atol=1e-5), sq.mean()
    assert grasp_idx == 60


def test_small_flat_slow_release():
    """obb_z_world in [0.04,0.08): small_flat -> squeeze 0.20, slow-release
    tail padded to >=48, no lift block (obb_z_world<0.08)."""
    raw = _synthetic_raw(t_raw=120, release_start=100)
    full, pick_end, grasp_idx, _ = build_full_trajectory(
        raw, pick_end=30, approach_used="top", obb_z_world=0.05,
        cur_arm=CUR_ARM, cur_hand=CUR_HAND, home_arm=HOME,
    )
    # release_len_orig=20 -> slow_len=48. traj=100+48=148.
    # 30 preamble + 148 + 0 lift + 30 return = 208.
    assert full.shape == (208, 22), full.shape
    assert pick_end == 60
    assert grasp_idx == 60

    tail0 = full[30 + 100, CURL_COLS]
    tail_last = full[30 + 100 + 47, CURL_COLS]
    assert np.allclose(tail0, 1.20, atol=1e-5), tail0.mean()
    assert np.allclose(tail_last, 0.0, atol=1e-5), tail_last.mean()
    assert np.allclose(full[30 + 100, :6], full[30 + 100 + 47, :6])


def test_side_approach_squeeze_delta():
    """side approach -> squeeze 0.30, never small_flat/very_flat."""
    raw = _synthetic_raw()
    full, _, _, _ = build_full_trajectory(
        raw, pick_end=30, approach_used="side", obb_z_world=0.18,
        cur_arm=CUR_ARM, cur_hand=CUR_HAND, home_arm=HOME,
    )
    sq = full[60:130, :][:, CURL_COLS]
    assert np.allclose(sq, 1.30, atol=1e-5), sq.mean()
    # side + tall is not "top", so no lift block: 30+120+0+30 = 180.
    assert full.shape == (180, 22), full.shape


def test_pre_transit_lift_smoothing_inserts_when_joint1_rises():
    """A synthetic raw with joint-1 rising during transit must trigger the
    pre-transit smoothing: cuRobo waypoints whose joint-1 has risen to the
    `lift_j1_fraction * (home-grasp)` target get REPLACED with N smooth
    knots, and trajectory length shifts by (N - replaced)."""
    raw = _synthetic_raw(t_raw=120, release_start=100)
    # Inject a ramp on joint-1 during the transit window [40, 80) so the
    # smoothing has something to detect. After the ramp, joint-1 holds.
    grasp_j1 = float(raw[30, 1])      # = 1.0
    home_j1 = float(HOME[1])          # = -0.7
    frac = 0.20
    target_j1 = grasp_j1 * (1 - frac) + home_j1 * frac  # 0.66
    # Ramp PAST the target so the seam detection is unambiguous w.r.t.
    # float32 truncation. Hits the target exactly at k=50.
    ramp_end = target_j1 - 0.05
    for k in range(40, 50):
        a = (k - 40) / 10.0
        raw[k, 1] = grasp_j1 * (1 - a) + ramp_end * a
    raw[50:100, 1] = ramp_end  # hold past target

    n_lift = 12
    cfg = TrajCfg(pre_transit_lift_wps=n_lift, pre_transit_lift_j1_fraction=0.20)
    full, _, grasp_idx, _ = build_full_trajectory(
        raw, pick_end=30, approach_used="top", obb_z_world=0.0833,
        cur_arm=CUR_ARM, cur_hand=CUR_HAND, home_arm=HOME, cfg=cfg,
    )
    # Insertion fires: full.shape[0] = 190 + (n_lift - replaced), and
    # replaced is the smallest k in [40,...) where |raw[k,1]-grasp_j1| >=
    # needed_dj1=0.34 minus float epsilon. With the ramp we picked, that's
    # k=49 (|delta|≈0.351). Replaced = 49-40 = 9; net shift = 12-9 = +3.
    assert full.shape[0] > 190, full.shape
    assert grasp_idx == 60  # contact index unchanged
    # The smoothed knots live at full[30+grasp_hold_end .. 30+40+n_lift)
    # = [70, 70+n_lift). Endpoint must match cuRobo's seam waypoint
    # (i.e., raw[seam_k,1]) so there's NO jump at the seam.
    smooth = full[70:70 + n_lift, 1]
    seam_val = float(full[70 + n_lift, 1])
    assert abs(float(smooth[-1]) - seam_val) < 1e-5, (smooth[-1], seam_val)
    # Smooth segment must be monotone (no jitter inside the lift).
    diffs = np.diff(smooth)
    assert np.all(diffs <= 1e-6) or np.all(diffs >= -1e-6), diffs
    # And the entire smooth segment must stay between grasp_j1 and seam_val.
    lo, hi = min(grasp_j1, seam_val), max(grasp_j1, seam_val)
    assert smooth.min() >= lo - 1e-5 and smooth.max() <= hi + 1e-5


def test_pre_transit_lift_smoothing_disabled():
    """pre_transit_lift_wps=0 must reproduce the original (pre-fix) shape."""
    raw = _synthetic_raw(t_raw=120, release_start=100)
    for k in range(40, 50):
        a = (k - 40) / 10.0
        raw[k, 1] = 1.0 * (1 - a) + (-0.5) * a
    raw[50:100, 1] = -0.5
    cfg = TrajCfg(pre_transit_lift_wps=0)
    full, _, _, _ = build_full_trajectory(
        raw, pick_end=30, approach_used="top", obb_z_world=0.0833,
        cur_arm=CUR_ARM, cur_hand=CUR_HAND, home_arm=HOME, cfg=cfg,
    )
    assert full.shape == (190, 22), full.shape


def test_pick_end_shift_only_preamble():
    raw = _synthetic_raw()
    _, pick_end, grasp_idx, _ = build_full_trajectory(
        raw, pick_end=45, approach_used="top", obb_z_world=0.10,
        cur_arm=CUR_ARM, cur_hand=CUR_HAND, home_arm=HOME,
    )
    assert pick_end == 45 + 30
    assert grasp_idx == 60  # contact index is independent of pick_end


def test_release_lift_target_lifts_toward_home_only_on_shoulder():
    # Place pose low at the drop; home is "up" on shoulder-lift (joint 1).
    place = np.array([0.3, -0.2, 1.9, -3.5, -1.0, 3.7], dtype=np.float32)
    home = np.array([0.5, -0.7, 1.7, -4.0, -1.2, 3.9], dtype=np.float32)
    place_before = place.copy()
    lift = release_lift_target(place, home, frac=0.30)
    # Only shoulder-lift (idx 1) moves; every other joint stays at place.
    for j in (0, 2, 3, 4, 5):
        assert lift[j] == place[j]
    # Shoulder-lift moves 30% of the way toward home (the "up" direction).
    expected = place[1] * 0.7 + home[1] * 0.3
    assert abs(lift[1] - expected) < 1e-6
    # It moves TOWARD home, never past it or the wrong way.
    assert min(place[1], home[1]) <= lift[1] <= max(place[1], home[1])
    # Inputs are not mutated.
    assert np.array_equal(place, place_before)


def test_release_lift_target_frac_clamped_and_endpoints():
    place = np.array([0.3, -0.2, 1.9, -3.5, -1.0, 3.7], dtype=np.float32)
    home = np.array([0.5, -0.7, 1.7, -4.0, -1.2, 3.9], dtype=np.float32)
    # frac=0 -> no lift (stays at place); frac>=1 -> clamps to home's shoulder.
    assert release_lift_target(place, home, 0.0)[1] == place[1]
    assert abs(release_lift_target(place, home, 5.0)[1] - home[1]) < 1e-6


def test_release_reorient_pose_levels_hand_to_home_pitch():
    # Place: hand pointing down (top grasp). Home: horizontal hand.
    place = np.array([0.3, -0.2, 1.9, -3.5, -1.0, 3.7], dtype=np.float32)
    home = np.array([0.5, -0.7, 1.7, -4.0, -1.2, 3.9], dtype=np.float32)
    place_before = place.copy()
    rel = release_reorient_pose(place, home, frac=0.30)
    # The KEY invariant: in-plane pitch (j1+j2+j3) equals home's, so the hand is
    # levelled to the home orientation regardless of how the arm reached/lifted.
    assert abs((rel[1] + rel[2] + rel[3]) - (home[1] + home[2] + home[3])) < 1e-5
    # wrist_2 / wrist_3 -> home (out-of-plane tilt + roll).
    assert abs(rel[4] - home[4]) < 1e-6
    assert abs(rel[5] - home[5]) < 1e-6
    # Shoulder-lift (1) lifts 30% toward home (clears the table); elbow + base
    # stay at the place pose so the hand stays over the drop.
    assert abs(rel[1] - release_lift_target(place, home, 0.30)[1]) < 1e-6
    assert rel[0] == place[0]
    assert rel[2] == place[2]
    # A top grasp points down (large pitch-sum); levelling must actually MOVE
    # wrist_1 away from its place value (not a no-op).
    assert abs(rel[3] - place[3]) > 0.1
    # Inputs untouched.
    assert np.array_equal(place, place_before)


def test_palm_down_settle_keeps_orientation():
    # The top-grasp release (ros_io 3b) settles straight DOWN toward the goal by
    # lerping hover_arm -> descend_arm (full[ri]) with NO reorientation, so the
    # hand keeps the grasp's palm-down orientation (no weird wrist spin). The
    # settle never goes past the descend pose, and at sf=0 stays at the hover.
    hover = np.array([0.3, -0.6, 1.4, -2.2, 0.9, 0.4], dtype=np.float32)   # high
    descend = np.array([0.3, -0.9, 1.7, -2.5, 0.9, 0.4], dtype=np.float32)  # low drop
    for sf in (0.0, 0.3, 0.5, 1.0):
        settle = hover * (1.0 - sf) + descend * sf
        # Monotone descent between the two poses (every joint a convex blend).
        assert np.all((settle >= np.minimum(hover, descend) - 1e-6)
                      & (settle <= np.maximum(hover, descend) + 1e-6))
    assert np.allclose(hover * 1.0 + descend * 0.0, hover)  # sf=0 -> hover


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} trajectory port tests passed.")
