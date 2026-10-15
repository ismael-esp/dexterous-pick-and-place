#!/usr/bin/env python3
"""Fit an affine camera->world pose correction  P_true ~= A @ P_fp + b  from
(P_fp, P_true) sample pairs, report the residual, and write a JSON calibration
that the hardware runner loads via ``--pose_correction``.

Why: on the grazing ``extra`` camera mount, FoundationPose's world-frame object
position has a systematic, distance-growing bias (poor depth at the oblique low
angle). The robot and BODex are correct (verified: BODex cuRobo FK == real ROS
TF to ~1 mm), so we compensate the perception bias in software. An affine map
captures both the constant offset and the linear-with-distance growth.

Pairs:
  P_fp   = the object position FoundationPose published on /<obj>_pose (world).
  P_true = the real world position, read from the  world -> ra_link_3_0_tip  TF
           when the index fingertip is touching that same physical point.

Usage:
  # fit from the built-in 9 hand-measured points and write the JSON:
  python3 bodex_hw/fit_pose_correction.py --out bodex_hw/pose_correction_cam2.json

  # fit from a CSV (cols: fpx,fpy,fpz,truex,truey,truez), e.g. from the collector:
  python3 bodex_hw/fit_pose_correction.py --csv samples.csv --out bodex_hw/pose_correction_cam2.json
"""
# Standard Library
import argparse
import json

# Third Party
import numpy as np

# Built-in calibration set — cam2 "extra" (grazing) mount, measured 2026-06.
# P_fp from /<obj>_pose; P_true = P_fp - DELTA, where DELTA is the measured
# FoundationPose error (P_fp - P_true) recorded with the green_cube + fingertip.
_PFP = np.array([
    [ 0.24910, 0.42710, 1.02500],
    [ 0.25350, 0.36130, 1.03180],
    [ 0.09770, 0.50710, 1.03120],
    [-0.05660, 0.43830, 1.02880],
    [-0.05660, 0.56780, 1.02920],
    [-0.29520, 0.32660, 1.02100],
    [-0.25940, 0.52460, 1.03210],
    [-0.37246, 0.47740, 1.02930],
    [-0.52311, 0.34300, 1.02911],
])
_DELTA = np.array([
    [-0.0579,  0.0941, 0.0300],
    [-0.0579,  0.0233, 0.0635],
    [-0.0633,  0.0281, 0.0402],
    [-0.0766,  0.0957, 0.0336],
    [-0.0766,  0.1670, 0.0507],
    [-0.1042,  0.0296, 0.0562],
    [-0.0954,  0.0386, 0.0501],
    [-0.1104,  0.0364, 0.0463],
    [-0.1361, -0.1520, 0.0363],
])


def load_pairs(csv):
    if csv:
        data = np.loadtxt(csv, delimiter=",", ndmin=2)
        if data.shape[1] < 6:
            raise SystemExit("CSV needs 6 cols: fpx,fpy,fpz,truex,truey,truez")
        return data[:, :3], data[:, 3:6]
    return _PFP.copy(), (_PFP - _DELTA)


def fit_affine(p_fp, p_true):
    """Least-squares 2D affine on XY only:  [x,y]_true ~= A2 @ [x,y]_fp + b2.

    Z is intentionally NOT used as an input and NOT corrected: every object sits
    on the table so P_fp_z is ~constant (≈1.03 m), which makes a 3D fit's z-column
    collinear with the constant term and blows up the z-coefficients. The runner
    overrides Z via --snap_z_to_table anyway. We return a 3x3 A / 3-vec b with the
    z row = identity so the runner can apply a single  p[:3] = A @ p[:3] + b.
    """
    XY = np.hstack([p_fp[:, :2], np.ones((len(p_fp), 1))])   # [N,3]
    M, *_ = np.linalg.lstsq(XY, p_true[:, :2], rcond=None)    # [3,2]
    A2 = M[:2, :].T   # [2,2]
    b2 = M[2, :]      # [2]
    A = np.eye(3)
    A[:2, :2] = A2
    b = np.array([b2[0], b2[1], 0.0])
    return A, b


def apply_affine(p_fp, A, b):
    return (A @ np.asarray(p_fp).T).T + b


def _err(pred, p_true):
    d = pred - p_true
    return np.linalg.norm(d, axis=1), np.linalg.norm(d[:, :2], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="fpx,fpy,fpz,truex,truey,truez per row")
    ap.add_argument("--out", default="bodex_hw/pose_correction_cam2.json")
    ap.add_argument("--camera", default="cam2_extra")
    a = ap.parse_args()

    p_fp, p_true = load_pairs(a.csv)
    n = len(p_fp)

    # baselines
    e_raw, exy_raw = _err(p_fp, p_true)
    off = (p_true - p_fp).mean(0)
    e_off, exy_off = _err(p_fp + off, p_true)
    # affine
    A, b = fit_affine(p_fp, p_true)
    pred = apply_affine(p_fp, A, b)
    e_aff, exy_aff = _err(pred, p_true)

    print(f"n_points = {n}")
    print(f"  raw FP        : mean_xy={exy_raw.mean():.3f}  max_xy={exy_raw.max():.3f} m")
    print(f"  const offset  : mean_xy={exy_off.mean():.3f}  max_xy={exy_off.max():.3f} m  (b={off.round(3)})")
    print(f"  AFFINE A*P+b  : mean_xy={exy_aff.mean():.3f}  max_xy={exy_aff.max():.3f} m")
    print(f"  per-point xy residual (mm): {(exy_aff*1000).round(1)}")

    cal = {
        "camera": a.camera,
        "model": "affine",
        "frame": "world",
        "A": A.tolist(),
        "b": b.tolist(),
        "n_points": int(n),
        "residual_mean_xy_m": float(exy_aff.mean()),
        "residual_max_xy_m": float(exy_aff.max()),
        "note": "P_true ~= A @ P_fp + b. Compensates FoundationPose world-frame "
                "bias from the grazing extra-cam mount. Re-fit if the camera moves.",
    }
    with open(a.out, "w") as f:
        json.dump(cal, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
