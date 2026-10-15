# BODex pick-and-place — commands & how-to

> For an overview of the task architecture, feature timeline, and validated
> numbers, see the consolidated progress log: `isaaclab_tasks/PROGRESS.md`.

---


Practical guide for running the Isaac Lab pick-and-place task, switching arms/objects/approaches, saving videos, and adding new custom objects from scratch.

- Main entry point: `BODex/isaaclab_tasks/run_bodex_pick_place.py`
- Isaac Lab python:   `/home/robot/anaconda3/envs/env_isaaclab/bin/python`
- BODex python:       `/home/robot/anaconda3/envs/bodex/bin/python`  (only the worker uses it)
- Video output:       `/tmp/bodex_pick_place.mp4`  (+ frames under `/tmp/bodex_pick_place_frames/`)

---

## 1. Quick start — run the default task

```bash
cd /mnt/ssd2/ws/Dualarm_ws/BODex

/home/robot/anaconda3/envs/env_isaaclab/bin/python \
    isaaclab_tasks/run_bodex_pick_place.py \
    --active_arm ra --object_id red_cube --approach top \
    --num_episodes 5 --enable_cameras
```

- First run takes ~35s for Isaac Sim startup + ~5s for the BODex worker to warm up, then ~10s per episode.
- Video gets written to `/tmp/bodex_pick_place.mp4` when the run finishes.
- Drop `--headless` to watch the GUI live; add it to run faster with no window.

---

## 2. Common flag changes

| What you want | Flag |
|---|---|
| Switch to the left arm (UR3e) | `--active_arm la` |
| Change object | `--object_id peach` / `red_cube` / `tomato` / `mouse` / `bottle` |
| Change grasp direction | `--approach top` / `side` / `side_front` / `side_left` / `side_right` |
| More episodes | `--num_episodes 20` |
| More / fewer BODex seeds (robustness vs speed) | `--seed_num 30` (default 20) |
| Run without GUI (faster batch) | `--headless --enable_cameras` |
| Different random seed per run | just rerun — object spawn is re-sampled each episode |

Example — 10-episode stress test on the left arm grasping a tomato from the side:

```bash
/home/robot/anaconda3/envs/env_isaaclab/bin/python \
    isaaclab_tasks/run_bodex_pick_place.py \
    --active_arm la --object_id tomato --approach side_front \
    --num_episodes 10 --headless --enable_cameras
```

---

## 3. Saving videos / inspecting frames

Enable with `--enable_cameras`. The script:
1. Writes one PNG per physics step to `/tmp/bodex_pick_place_frames/epXXX_frameYYYY.png`.
2. Stitches them into `/tmp/bodex_pick_place.mp4` at 30 fps when the run finishes.

To keep multiple runs, move the outputs before re-running:

```bash
mv /tmp/bodex_pick_place.mp4  ~/Videos/bodex_ra_redcube_top.mp4
mv /tmp/bodex_pick_place_frames ~/Videos/bodex_ra_redcube_top_frames
```

To change the camera angle, edit `BODex/isaaclab_tasks/bodex_pick_place/env_cfg.py` — search for `camera_cfg`. `pos=(x, y, z)` and `rot=(w, x, y, z)` use Isaac Sim's OpenGL convention (+X right, +Y up, -Z forward from the camera's perspective).

---

## 4. Using the BODex service from another Python project

The pick-and-place task is one client. You can get trajectories from any Python script:

```python
import sys, numpy as np
sys.path.insert(0, "/mnt/ssd2/ws/Dualarm_ws/BODex")
from bodex_service import BODexClient

with BODexClient() as c:
    out = c.synthesize_pick_and_place(
        active_arm   = "ra",
        object_id    = "red_cube",
        object_pose  = np.array([0.05, 0.47, 1.00, 1, 0, 0, 0]),
        place_xyz    = np.array([0.12, 0.50, 1.01]),
        approach     = "top",
        seed_num     = 20,
        table_z      = 0.96,
    )

# out["trajectory"]    : np.float32 [T, 22]  (6 arm joints + 16 hand joints)
# out["pick_end_idx"]  : int — end of pick phase; rest is place
# out["grasp_error"]   : float
```

See `bodex_service/README.md` for the full API.

---

## 5. Adding a new custom object — step by step

There are **two** file groups to add: one for BODex grasp synthesis (URDF + mesh + info) and one for Isaac Lab visualization (USD). Then register the object in the Python catalog.

> **Looking for the full guide?** See **`bodex_service/CUSTOM_OBJECTS_GUIDE.md`**
> for the end-to-end walk-through (three preprocessing paths, MeshProcess
> integration, optional DexLearn training, validation checklist, common
> pitfalls). The summary below covers the most common case.

### 5.1 — Prepare the mesh

Starting material: a clean, watertight triangle mesh of the object (`.obj` or `.stl`).
Recommended: < 5000 triangles for speed. If it's denser, decimate with Blender or MeshLab first.

Also decide:
- `object_id` — short lowercase identifier, e.g. `blue_mug` (no spaces / special chars).
- Physical `mass` in kg — needed for PhysX in Isaac Sim.
- The object's **oriented bounding box** dimensions (obb) in meters along its local xyz. Used by the spawn code to place the object just-above the table.

### 5.2 — Generate the BODex processed_data/ folder

BODex expects this exact layout:

```
BODex/src/curobo/content/assets/object/custom/processed_data/<object_id>/
├── mesh/simplified.obj           ← the triangle mesh (clean, centered)
├── urdf/coacd.urdf               ← URDF describing it as a single rigid link
└── info/simplified.json          ← metadata BODex reads at load time
```

**Easiest path** — use `MeshProcess`:

```bash
cd /mnt/ssd2/ws/Dualarm_ws/MeshProcess
# follow its README to convert your .obj into the above layout;
# it also runs CoACD to generate the convex-decomposition URDF.
python scripts/process_single.py \
    --input /path/to/your_mesh.obj \
    --out_dir /mnt/ssd2/ws/Dualarm_ws/BODex/src/curobo/content/assets/object/custom/processed_data/<object_id>
```

**Manual path** (if you just want a quick test with a simple shape):

1. Copy an existing object folder as a template:
   ```bash
   cd /mnt/ssd2/ws/Dualarm_ws/BODex/src/curobo/content/assets/object/custom/processed_data
   cp -r red_cube blue_mug
   ```
2. Replace `mesh/simplified.obj` with your mesh.
3. Edit `urdf/coacd.urdf` to point at your new mesh path (the URDF references `meshes/*.obj` — keep that structure or update the `<mesh filename="..."/>` tag).
4. Edit `info/simplified.json` with real values:
   ```json
   {
     "gravity_center": [0.0, 0.0, 0.0],
     "obb":            [ox, oy, oz],
     "scale":          1.0,
     "density":        1.0,
     "mass":           0.00039
   }
   ```
   - `gravity_center`: offset from the mesh origin to the center of mass, in meters.
   - `obb`: oriented bounding-box extents along x, y, z.
   - `mass` here is in BODex's internal units (kg × scale). Usually leave at whatever the reference object had unless you know you need to change it — Isaac Sim's own mass is set separately in step 5.4.

### 5.3 — Add the USD for Isaac Lab visualization

Isaac Sim renders from USD, not URDF. You need a `.usd` or `.usdc`:

```
BODex/isaaclab_tasks/assets/Materials/<object_id>/<object_id>.usd
```

Ways to get one:

- **A. Convert from OBJ/STL in Isaac Sim GUI:**
   1. Open Isaac Sim → File → Import → pick your `.obj`.
   2. In the Stage panel, set the mesh's physics properties: Add → RigidBodyAPI, Add → CollisionAPI.
   3. File → Save As → `BODex/isaaclab_tasks/assets/Materials/<object_id>/<object_id>.usd`.

- **B. Use `isaaclab_dualarm/` conversion helpers** if you have a URDF:
   ```bash
   /home/robot/anaconda3/envs/env_isaaclab/bin/python \
       /home/robot/IsaacLab/scripts/tools/convert_urdf.py \
       /path/to/your.urdf \
       /mnt/ssd2/ws/Dualarm_ws/BODex/isaaclab_tasks/assets/Materials/<object_id>/<object_id>.usd \
       --make-instanceable --fix-base false
   ```

- **C. Quick test — reuse `red_cube.usd` as a placeholder** while you develop the grasp pipeline, then swap it later.

### 5.4 — Register in the OBJECT_CATALOG

Open `BODex/isaaclab_tasks/bodex_pick_place/env_cfg.py` and add an entry to `OBJECT_CATALOG`:

```python
OBJECT_CATALOG = {
    # ...existing entries...
    "blue_mug": dict(
        usd=os.path.join(_MATERIALS_DIR, "blue_mug", "blue_mug.usd"),
        scale=(1.0, 1.0, 1.0),
        mass=0.20,          # kg — Isaac Sim physics mass
        obb_z=0.10,         # meters along the object's local +Z (height while resting)
    ),
}
```

Also extend `--object_id`'s `choices` in `isaaclab_tasks/run_bodex_pick_place.py`:

```python
parser.add_argument("--object_id", default="peach",
                    choices=["peach", "red_cube", "tomato", "mouse", "blue_mug"])
```

### 5.5 — Verify it loads

Fastest verification — just ask the BODex service for a grasp (this catches issues in processed_data/ early, before Isaac Sim startup overhead):

```bash
/home/robot/anaconda3/envs/env_isaaclab/bin/python - <<'PY'
import sys, numpy as np
sys.path.insert(0, "/mnt/ssd2/ws/Dualarm_ws/BODex")
from bodex_service import BODexClient

with BODexClient() as c:
    r = c.synthesize_pick_and_place(
        active_arm="ra", object_id="blue_mug",
        object_pose=np.array([0.05, 0.47, 1.00, 1, 0, 0, 0]),
        place_xyz=np.array([0.12, 0.50, 1.01]),
        approach="top", seed_num=10,  table_z=0.96,
    )
    print("OK", r["trajectory"].shape, "grasp_err", r["grasp_error"])
PY
```

If that prints an `OK (T, 22) ...` line, the BODex side is wired up correctly.

Then run the full task:

```bash
/home/robot/anaconda3/envs/env_isaaclab/bin/python \
    isaaclab_tasks/run_bodex_pick_place.py \
    --active_arm ra --object_id blue_mug --approach top \
    --num_episodes 3 --enable_cameras
```

---

## 6. Starting / stopping the BODex worker manually

Normally `BODexClient` auto-starts the worker on first use and leaves it running for later clients to reuse. To control it explicitly:

**Start** (e.g. at boot so every client attaches instantly):

```bash
/home/robot/anaconda3/envs/bodex/bin/python -m bodex_service.server \
    --socket /tmp/bodex_service.sock --verbose
```

**Attach without auto-start:**

```python
client = BODexClient(auto_start=False)
```

**Stop the worker:**

```bash
# graceful
pkill -f bodex_service.server
# or via an existing client:
client.close()   # shuts down the worker if this client started it
```

**Two workers on two GPUs** (for parallel jobs):

```bash
CUDA_VISIBLE_DEVICES=0 /home/robot/anaconda3/envs/bodex/bin/python \
    -m bodex_service.server --socket /tmp/bodex_service_g0.sock &

CUDA_VISIBLE_DEVICES=1 /home/robot/anaconda3/envs/bodex/bin/python \
    -m bodex_service.server --socket /tmp/bodex_service_g1.sock &
```

```python
c0 = BODexClient(socket_path="/tmp/bodex_service_g0.sock", auto_start=False)
c1 = BODexClient(socket_path="/tmp/bodex_service_g1.sock", auto_start=False)
```

---

## 7. Tuning knobs that actually matter

In `BODex/isaaclab_tasks/bodex_pick_place/`:

| File | Variable | Default | Effect |
|---|---|---|---|
| `env_cfg.py` | `waypoint_repeat` | 2 | Physics steps per BODex waypoint. 1 = fast/jerky, 4 = smooth/slow. |
| `env_cfg.py` | `return_to_home_waypoints` | 30 | Length of the retract phase after release. |
| `env_cfg.py` | `object_spawn_x_ra/la`, `object_spawn_y_ra/la` | see below | Random object spawn zone, per arm. |
| `env_cfg.py` | `place_target_x_ra/la`, `place_target_y_ra/la` | see below | Random place-target zone, per arm. |

**Arm-specific workspace zones** (reachable sub-rectangles of the motion_desk table):

| Arm | x range | y range |
|---|---|---|
| ra (right UR3) | (-0.10, 0.30) | (0.28, 0.65) |
| la (left UR3e) | (-0.30, 0.10) | (0.28, 0.65) |

Mirror in x; same y range. Tune in `env_cfg.py` if you need a larger or smaller random zone — BODex will report infeasible seeds for unreachable targets and the regrasp logic handles those gracefully.


| `env_cfg.py` | `max_regrasp_attempts` | 2 | Max retries per episode on grasp/place/drop failure. 0 disables. |
| `env_cfg.py` | `retry_home_preamble_wps` | 30 | Waypoints of "back to home" prepended on every retry. |
| `env_cfg.py` | `drop_z_threshold` | 0.75 | Mid-transit cube z below this triggers immediate retry. |
| `env_cfg.py` | `min_lift_for_grasp` | 0.03 | Minimum cube lift (m) to judge grasp as successful. |
| `env_cfg.py` | `object_spawn_x / _y` | (−0.02, 0.15) / (0.40, 0.55) | Randomization zone in world meters. |
| `env_cfg.py` | `place_target_x / _y` | same as spawn | Randomization zone for the goal. |
| `env.py` | `SQUEEZE_DELTA` | 0.10 rad | Extra curl on top of BODex's grasp hand. Higher = tighter grip but cube shifts in palm. |
| `env.py` | `place_success < 0.12` | 0.12 m | xy tolerance for "placed" success. |
| `bodex_plan_cli.py` | lift offset `+0.10` | 0.10 m | How high BODex plans to raise the cube before transit. |

For a robust demo with the current dynamics, defaults are what was validated (90% grasp / 80% place / 70% full success on 20 eps).

---

## 8. Regrasp / retry behavior

The task has a built-in retry state machine: if grasp, place, or transit fails, the environment reads the cube's *current* pose from Isaac Sim, asks the BODex worker for a fresh trajectory, and splices it onto whatever the arm is doing — no episode reset, no re-spawn.

**When a retry fires:**
- **Grasp miss** — cube never lifted (lifted_by < `min_lift_for_grasp`, default 3 cm).
- **Place miss** — cube lifted but final xy > 12 cm from the goal.
- **Mid-transit drop** — cube z falls below `drop_z_threshold` (default 0.75 m) after the pick phase.

**What happens on retry:**
1. The arm smoothly interpolates from its current joints back to home (`retry_home_preamble_wps` waypoints, default 30).
2. The BODex worker is called with the *current* cube pose (not the original spawn pose) and the *same* place target.
3. The new trajectory executes. If it succeeds, the episode terminates successfully; if it fails again, attempt #2 starts, up to `max_regrasp_attempts` (default 2 retries → 3 total attempts).

**Disable retries entirely** (go back to single-shot behavior):

```python
# in env_cfg.py
max_regrasp_attempts: int = 0
```

or on the CLI for a one-off run, hack it in by editing `env_cfg.py` before launching.

**How to tell a retry fired:** the runner prints

```
[env 0] REPLAN #1 (place miss (xy_err=0.152)) from obj_pose=(...)
[env 0] trajectory loaded T=192 pick_end=90 retry=True ...
```

and the per-episode summary line

```
[eval] attempts         = 2
```

shows more than 1 attempt.

**Tradeoffs:**
- +Robustness: typical 20-ep stress test improves from ~70% to ~90% full success with `max_regrasp_attempts=2`.
- –Speed: each retry costs ~4-6 s (BODex call + playback), so failure-heavy episodes run 2-3× longer.
- Infeasible recoveries: if the cube rolled outside the arm's reachable workspace, BODex will return infeasible and the episode terminates as a failure (`[env 0] replan skipped: max attempts reached` or a logged BODex traceback). No infinite loop.

## 8.5 Bottle (tall object) + human-like side grasp

Short objects like the red_cube don't leave enough finger-table clearance for a side grasp. The `bottle` object (7 cm diameter, 18 cm tall cylinder) is tall enough to work with `--approach side` — the hand wraps around the bottle with the thumb pointing up, mimicking how a person would pick up a water bottle.

```bash
# Right arm, bottle, human-like side grasp (thumb-up, palm vertical)
/home/robot/anaconda3/envs/env_isaaclab/bin/python \
    isaaclab_tasks/run_bodex_pick_place.py \
    --active_arm ra --object_id bottle --approach side \
    --num_episodes 5 --headless --enable_cameras
```

### How `--approach side` works

The `side` preset is **arm-aware**:
- right arm (ra) → approaches from world +X (robot's right), palm pushes -X, thumb +Z
- left arm (la) → approaches from world -X, palm pushes +X, thumb +Z

After BODex's grasp optimizer runs, each candidate seed is filtered: seeds whose palm Y-axis (≙ thumb direction) tilts more than ~50° from world +Z are discarded. The per-call filter stats are printed to stdout:

```
[side-grasp orientation filter] kept 2/3 feasible seeds (thumb_up_cos > 0.65; best=0.964)
```

`kept A/B` means A of B IK-feasible seeds also passed the thumb-up constraint. If A==0 on both attempts, the episode fails with `BODex synthesis FAILED`.

### Tuning

| File / line | Knob | Default | What it does |
|---|---|---|---|
| `bodex_plan_cli.py` `THUMB_UP_MIN` | 0.65 | Cosine threshold for "thumb is up" (0.65 ≈ 50°). Lower → more seeds pass; higher → stricter alignment. |
| `APPROACH_PRESETS["side_left"/"side_right"]` | — | Override per-side world-frame orientation if you want to force a specific approach independent of the arm. |
| `_SIDE_BY_ARM["ra"/"la"]` | — | What `--approach side` resolves to for each arm. |

### Adding more tall objects

Use the `primitive="cylinder"` pattern in `OBJECT_CATALOG` for anything rotationally symmetric. No USD needed:

```python
# In env_cfg.py OBJECT_CATALOG
"tall_can": dict(
    primitive="cylinder",
    radius=0.033, height=0.12, mass=0.33, obb_z=0.12,
    diffuse_color=(0.8, 0.1, 0.1),
),
```

Then create the matching BODex processed_data (mesh + URDF + info) under `BODex/src/curobo/content/assets/object/custom/processed_data/tall_can/` — use the `bottle/` folder as a template. Generate the cylinder OBJ with one line of `trimesh`:

```python
import trimesh
m = trimesh.creation.cylinder(radius=0.033, height=0.12, sections=32)
m.export("<path>/mesh/simplified.obj")
```

---

## 9. Troubleshooting

- **`RuntimeError: No seed has all three IK stages feasible`** — the sampled object pose is outside the arm's reachable workspace. Tighten `object_spawn_x / _y` in `env_cfg.py` or pick a different `--approach`.
- **`No BODex worker at /tmp/bodex_service.sock`** — start one manually with `python -m bodex_service.server --socket /tmp/bodex_service.sock` so you can see the real error on its stderr. Common cause: a pre-existing stale socket file; `rm /tmp/bodex_service.sock` then retry.
- **Cube slips out of hand during transit** — increase `SQUEEZE_DELTA` in `env.py` slightly (e.g. 0.15). Don't go above 0.25 — see PROCESS_LOG 2026-04-23 (pm) entry.
- **Isaac Sim window freezes on startup** — remove `--headless`, you probably have it set in a wrapper. Add `--enable_cameras` only if you actually want to record video; it costs ~30% FPS.
- **Video is missing frames** — `--enable_cameras` needs to be on for Isaac Sim to render the camera sensor. Adding it to a headless run is fine.
