# BODex grasp service

Long-running BODex grasp worker + Python client for use in other projects.

**Why this exists:** running `plan_batch_env.py` as a subprocess costs ~6s per call because Python + cuRobo + CUDA kernels have to reinitialize every time. This package runs a persistent worker inside the `bodex` conda env and lets any Python client (any env, any project) request grasp trajectories over a Unix socket. Subsequent calls take ~3-4s instead of 6-7s.

## Install / first use

Nothing to install — it's stdlib + numpy on the client side.

The worker needs the `bodex` conda env (torch + cuRobo built). The client looks for `/home/robot/anaconda3/envs/bodex/bin/python` by default; override with the `BODEX_SERVICE_PYTHON` environment variable.

## Minimal example

```python
from bodex_service import BODexClient
import numpy as np

with BODexClient() as client:
    result = client.synthesize_pick_and_place(
        active_arm   = "ra",                                   # "ra" or "la"
        object_id    = "red_cube",                             # must exist under DGN_2k / custom
        object_pose  = np.array([0.05, 0.47, 1.00, 1, 0, 0, 0]),  # [x y z qw qx qy qz]
        place_xyz    = np.array([0.12, 0.50, 1.01]),
        approach     = "top",                                  # top | side_front | side_left | side_right
        seed_num     = 20,
        table_z      = 0.96,
    )

trajectory = result["trajectory"]     # np.float32 [T, 22] — 6 arm joints + 16 hand joints
pick_end   = result["pick_end_idx"]   # int — last index of pick phase; rest is place
```

## Using it from another project (any Python env)

Add the BODex repo to `sys.path` then import:

```python
import sys
sys.path.insert(0, "/mnt/ssd2/ws/Dualarm_ws/BODex")

from bodex_service import BODexClient
```

Or install as editable in your env: `pip install -e /mnt/ssd2/ws/Dualarm_ws/BODex` (the top-level `setup.py` is BODex / cuRobo, so this may be invasive — prefer the `sys.path` route for isolated client use).

## Running the worker manually

Normally `BODexClient(auto_start=True)` handles this. To start the worker directly (e.g. on boot, so every client attaches instantly):

```bash
/home/robot/anaconda3/envs/bodex/bin/python -m bodex_service.server \
    --socket /tmp/bodex_service.sock
```

Then in any client:

```python
client = BODexClient(auto_start=False)   # attaches to the already-running worker
```

## What the trajectory looks like

Structure (default params): `30 approach + 10 grasp + 10 squeeze + 10 lift + 40 transit + 20 descend + 12 release = 132 waypoints × 22 DoF`. Joint order inside the 22-vector:

```
[0..5]   6 arm joints (UR3/UR3e)  — shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
[6..21]  16 Allegro hand joints    — joint_0_0, joint_12_0, joint_4_0, joint_8_0, ...  (same order as robot_file YAML)
```

`pick_end_idx` marks the boundary between pick phase and place phase so you can swap finger-closure strategies per phase.

## Timing (measured on RTX 5070 Ti, 20-seed BODex, red_cube)

| Call | Time  | What's happening |
|------|-------|------------------|
| 1st  | ~5.0s | cuRobo CUDA kernels compile, world_cfg build, grasp synth + IK |
| 2nd+ | ~3.4s | grasp synth + IK only (kernels cached) |

Compared to the old subprocess-per-call path (6.6s) this is roughly 2x faster. The remaining ~3s is dominated by BODex's L-BFGS grasp optimization itself, which is the honest minimum cost for this quality of grasp.

## API

### `BODexClient(socket_path, bodex_python, authkey, auto_start, startup_timeout, verbose)`
- `socket_path` — Unix socket. Default `/tmp/bodex_service.sock`. Multiple workers with different sockets can coexist (e.g. one per GPU).
- `bodex_python` — path to the bodex env's python. Default resolves from `BODEX_SERVICE_PYTHON` env var or `/home/robot/anaconda3/envs/bodex/bin/python`.
- `auto_start=True` — spawn a worker if no one is listening.
- `verbose=False` — forward worker stdout/stderr to the client's console.

### `client.synthesize_pick_and_place(active_arm, object_id, object_pose, place_xyz, approach, ...)`
Returns a dict: `trajectory`, `pick_end_idx`, `grasp_error`, `object_pose_world`, `place_xyz`, `active_arm`, `object_id`, `approach`, `success`.

Raises `RuntimeError` if the worker reports infeasible IK, `TimeoutError` if the worker doesn't respond within 120s.

### `client.ping()` — cheap liveness check (~10 ms).

### `client.close(shutdown_server=True)` — if this client spawned the worker, stops it. Safe to call multiple times.

## Custom objects

The server imports `BODex/isaaclab_tasks/bodex_pick_place/bodex_plan_cli.py` which writes a runtime `scene_cfg/<object_id>/runtime/latest.npy` for every request. Your `object_id` must exist as a folder under:

```
BODex/src/curobo/content/assets/object/custom/processed_data/<object_id>/
├── mesh/simplified.obj
├── urdf/coacd.urdf
└── info/simplified.json
```

If you need to add a new object, pre-process its mesh with MeshProcess (see main BODex README) and drop the resulting folder in place. No code changes needed — the service picks it up on next call.

## Troubleshooting

- **"No BODex worker at /tmp/bodex_service.sock"** — pass `auto_start=True` or start the worker manually. If auto-start fails with "exited before becoming ready", run the worker directly (`... -m bodex_service.server --socket /tmp/bodex_service.sock`) in a terminal to see the real error.
- **Socket permission / stale socket** — `rm /tmp/bodex_service.sock` before starting. The server cleans up its own socket on SIGTERM but not on hard kill.
- **Multiple GPUs** — `CUDA_VISIBLE_DEVICES=1 python -m bodex_service.server --socket /tmp/bodex_service_gpu1.sock` and `BODexClient(socket_path="/tmp/bodex_service_gpu1.sock")`.
