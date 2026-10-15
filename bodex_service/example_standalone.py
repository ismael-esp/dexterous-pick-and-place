"""Minimal standalone example of using the BODex grasp service from
**any** Python environment (no conda-env requirements on the caller).

Run with any python 3.8+ that has numpy installed:

    python bodex_service/example_standalone.py
"""
import time

import numpy as np

from bodex_service import BODexClient


def main():
    # Auto-starts the worker inside the bodex conda env on first use.
    with BODexClient(verbose=False) as client:
        print("[client] worker ready")

        # Two back-to-back synthesis calls — second should be much faster
        # because the worker's cuRobo kernels are already hot.
        for i in range(3):
            t0 = time.time()
            result = client.synthesize_pick_and_place(
                active_arm="ra",
                object_id="red_cube",
                object_pose=np.array([0.05, 0.47, 1.00, 1, 0, 0, 0]),
                place_xyz=np.array([0.12, 0.50, 1.01]),
                approach="top",
                seed_num=20,
                table_z=0.96,
            )
            dt = time.time() - t0
            traj = result["trajectory"]
            print(f"[call {i}] {dt:.2f}s  traj={traj.shape}  pick_end={result['pick_end_idx']}  "
                  f"grasp_err={result['grasp_error']:.4f}")


if __name__ == "__main__":
    main()
