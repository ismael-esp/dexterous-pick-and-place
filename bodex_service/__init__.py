"""BODex grasp service — persistent worker + Python client.

Usage from any Python project (no bodex env required):

    from bodex_service import BODexClient
    import numpy as np

    client = BODexClient()   # auto-starts the worker if not running
    result = client.synthesize_pick_and_place(
        active_arm="ra",
        object_id="red_cube",
        object_pose=np.array([0.05, 0.47, 1.00, 1, 0, 0, 0]),
        place_xyz=np.array([0.12, 0.50, 1.01]),
        approach="top",
    )
    # result["trajectory"] is a [T, 22] float32 numpy array
    # result["pick_end_idx"] marks end of pick phase

    client.close()           # or use as a context manager
"""
from .client import BODexClient

__all__ = ["BODexClient"]
