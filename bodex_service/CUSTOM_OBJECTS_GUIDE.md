# Adding a custom object — end-to-end guide

Walk-through for taking a new object from a raw mesh to a working BODex
pick-and-place run, with three paths depending on how complex the object is:

| Object kind | Best path | Time |
|---|---|---|
| Tall/cylindrical/box-like (bottles, cans, blocks) | **Path A — primitive cylinder/cuboid** | 10 min |
| Arbitrary triangle mesh you already have (.obj / .stl / .ply) | **Path B — MeshProcess** | 30–60 min, one-time |
| Quick "what does this look like" sanity test | **Path C — copy an existing object** | 5 min |

> All three paths share Steps 4 (Isaac Sim registration) and 5 (running). The
> preprocessing differs in Steps 1–3.

---

## What the runtime needs

Every object the task can grasp has **two artifacts** on disk:

1. **BODex processed_data** at
   `BODex/src/curobo/content/assets/object/custom/processed_data/<object_id>/`:
   ```
   mesh/simplified.obj   ← clean watertight triangle mesh (centered at origin)
   urdf/coacd.urdf       ← URDF wrapping the mesh as one rigid link
   info/simplified.json  ← metadata: gravity_center, obb, scale, density, mass
   ```
2. **Isaac Sim spawn config** — either a `.usd/.usdc` file under
   `BODex/isaaclab_tasks/assets/Materials/<object_id>/` (USD path) **or** a
   primitive descriptor (cylinder/cuboid path).

Plus one **catalog registration** in
`BODex/isaaclab_tasks/bodex_pick_place/env_cfg.py:OBJECT_CATALOG`.

---

## Path A — primitive cylinder or cuboid (recommended for tall, simple shapes)

This is how the `bottle` object was added. No mesh tooling needed.

### Step 1 — generate the BODex mesh OBJ

```bash
/home/robot/anaconda3/envs/bodex/bin/python <<'PY'
import os, trimesh

OBJECT_ID = "blue_can"
OUT = f"/mnt/ssd2/ws/Dualarm_ws/BODex/src/curobo/content/assets/object/custom/processed_data/{OBJECT_ID}"
os.makedirs(f"{OUT}/mesh", exist_ok=True)
os.makedirs(f"{OUT}/urdf", exist_ok=True)
os.makedirs(f"{OUT}/info", exist_ok=True)

# Cylinder: change to whatever size you need
mesh = trimesh.creation.cylinder(radius=0.033, height=0.12, sections=32)
# Or for a box:
# mesh = trimesh.creation.box(extents=[0.06, 0.04, 0.12])

mesh.export(f"{OUT}/mesh/simplified.obj")
print("OBB extents:", mesh.bounds[1] - mesh.bounds[0])
PY
```

### Step 2 — write the URDF

`BODex/src/curobo/content/assets/object/custom/processed_data/blue_can/urdf/coacd.urdf`:

```xml
<?xml version="1.0"?>
<robot name="blue_can">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.33"/>
      <!-- Diagonal inertia of a uniform-density cylinder, m=0.33, r=0.033, h=0.12.
           Ixx=Iyy = m*(3r²+h²)/12 = 0.33*(3*0.0011+0.0144)/12 ≈ 0.000487
           Izz = m*r²/2 = 0.33*0.0011/2 ≈ 0.00018 -->
      <inertia ixx="0.000487" ixy="0" ixz="0" iyy="0.000487" iyz="0" izz="0.00018"/>
    </inertial>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="../mesh/simplified.obj" scale="1 1 1"/></geometry>
    </collision>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="../mesh/simplified.obj" scale="1 1 1"/></geometry>
    </visual>
  </link>
</robot>
```

### Step 3 — write the info/simplified.json

```json
{
  "gravity_center": [0.0, 0.0, 0.0],
  "obb": [0.066, 0.066, 0.12],
  "scale": 1.0,
  "density": 1.0,
  "mass": 0.33
}
```

`obb` = oriented-bounding-box extents along x/y/z (full size, not half).

### Step 4 — register in OBJECT_CATALOG (Isaac Sim side)

Edit `BODex/isaaclab_tasks/bodex_pick_place/env_cfg.py`:

```python
OBJECT_CATALOG = {
    # ...existing entries...
    "blue_can": dict(
        primitive="cylinder",
        radius=0.033,
        height=0.12,
        mass=0.33,
        obb_z=0.12,
        diffuse_color=(0.1, 0.3, 0.7),    # RGB 0–1
    ),
}
```

For a cuboid object instead:

```python
"red_brick": dict(
    primitive="cuboid",                    # NOTE: cuboid path needs a small env_cfg edit (see below)
    size=(0.06, 0.04, 0.12),
    mass=0.5,
    obb_z=0.12,
    diffuse_color=(0.7, 0.1, 0.1),
),
```

> **Note**: only `primitive="cylinder"` is wired in `env_cfg.py:__post_init__`
> today. To use cuboid primitives, extend the `if entry.get("primitive") ==
> "cylinder":` branch with an `elif "cuboid":` clause that builds a
> `sim_utils.CuboidCfg(size=entry["size"], ...)`. ~10 lines.

### Step 5 — register in the runner CLI

`BODex/isaaclab_tasks/run_bodex_pick_place.py`:

```python
parser.add_argument("--object_id", default="peach",
                    choices=["peach", "red_cube", "tomato", "mouse",
                             "bottle", "blue_can"])  # ← add here
```

### Step 6 — run

```bash
cd /mnt/ssd2/ws/Dualarm_ws/BODex
/home/robot/anaconda3/envs/env_isaaclab/bin/python \
    isaaclab_tasks/run_bodex_pick_place.py \
    --active_arm ra --object_id blue_can --approach side \
    --num_episodes 5 --headless --enable_cameras
```

Look for these signals in the log:

```
[env 0] trajectory loaded T=162 ... obj=(...)
  [eval] OVERALL SUCCESS  = True
```

---

## Path B — arbitrary mesh via MeshProcess

For complex non-symmetric meshes (a banana, a hammer, a screwdriver), use the
**MeshProcess** repo. It runs CoACD convex decomposition and produces the
exact processed_data layout BODex expects.

### Step 1 — clean your source mesh

Required:
- Watertight (no holes)
- Outward-facing normals
- Up direction = +Z, ideally with object resting on the X-Y plane
- Single connected mesh (Blender → Mesh → Cleanup → Make Manifold if needed)
- Centered roughly at origin (BODex handles `gravity_center` separately, but
  centering simplifies debugging)
- File format: `.obj`, `.stl`, `.ply`, or `.glb`
- Triangle count: aim for < 5 000 (CoACD gets slow above that). Use Blender →
  Modifiers → Decimate to reduce.

### Step 2 — run MeshProcess

```bash
cd /mnt/ssd2/ws/Dualarm_ws/MeshProcess
conda activate bodex   # or whatever env MeshProcess uses

python scripts/process_single.py \
    --input /path/to/your_mesh.obj \
    --out_dir /mnt/ssd2/ws/Dualarm_ws/BODex/src/curobo/content/assets/object/custom/processed_data/<object_id>
```

This generates:
- `mesh/simplified.obj` (decimated visual mesh)
- `urdf/coacd.urdf` (URDF referencing convex parts)
- `urdf/meshes/*.obj` (CoACD convex decomposition pieces)
- `info/simplified.json` (auto-computed `gravity_center`, `obb`, `mass`)

If you don't have MeshProcess set up, the manual fallback is to copy an
existing processed_data folder, replace `simplified.obj` with your mesh, and
fill `info/simplified.json` by hand (use the trimesh snippet in Path A as a
guide).

### Step 3 — build the Isaac Sim USD

For Isaac Sim to render the object, you need a `.usd` file. Two ways:

**3a. Convert URDF → USD with Isaac Lab's helper:**

```bash
/home/robot/anaconda3/envs/env_isaaclab/bin/python \
    /home/robot/IsaacLab/scripts/tools/convert_urdf.py \
    /mnt/ssd2/ws/Dualarm_ws/BODex/src/curobo/content/assets/object/custom/processed_data/<object_id>/urdf/coacd.urdf \
    /mnt/ssd2/ws/Dualarm_ws/BODex/isaaclab_tasks/assets/Materials/<object_id>/<object_id>.usd \
    --make-instanceable --fix-base false
```

**3b. Convert OBJ → USD inside the Isaac Sim GUI:**

1. Launch Isaac Sim, File → Import → pick `simplified.obj`.
2. In Stage panel, select the imported mesh prim.
3. Right-click → Add → Physics → Rigid Body API.
4. Right-click → Add → Physics → Collision API.
5. File → Save As → write to
   `BODex/isaaclab_tasks/assets/Materials/<object_id>/<object_id>.usd`.

### Step 4 — register in OBJECT_CATALOG (USD path)

```python
OBJECT_CATALOG = {
    # ...
    "banana": dict(
        usd=os.path.join(_MATERIALS_DIR, "banana", "banana.usd"),
        scale=(1.0, 1.0, 1.0),
        mass=0.12,
        obb_z=0.04,             # tallest dimension when resting flat
    ),
}
```

### Step 5 — register CLI + run (same as Path A Steps 5–6)

---

## Path C — quick test by cloning an existing object

For sanity checks (does my pipeline work?), the fastest path is to copy one
of the four built-in objects.

```bash
cd /mnt/ssd2/ws/Dualarm_ws/BODex/src/curobo/content/assets/object/custom/processed_data
cp -r red_cube my_test_obj
```

Replace `mesh/simplified.obj` with your mesh, edit `info/simplified.json` to
match (especially `obb` and `mass`), and copy/rename the USD:

```bash
cp -r /mnt/ssd2/ws/Dualarm_ws/BODex/isaaclab_tasks/assets/Materials/red_cube \
      /mnt/ssd2/ws/Dualarm_ws/BODex/isaaclab_tasks/assets/Materials/my_test_obj
mv /mnt/ssd2/ws/Dualarm_ws/BODex/isaaclab_tasks/assets/Materials/my_test_obj/red_cube.usd \
   /mnt/ssd2/ws/Dualarm_ws/BODex/isaaclab_tasks/assets/Materials/my_test_obj/my_test_obj.usd
```

Register in `OBJECT_CATALOG` and CLI choices, then run.

---

## "Compute" — does anything need precomputing?

**No.** BODex's grasp synthesis runs at run time, on-demand, per episode:

1. The persistent worker (`bodex_service/server.py`) loads cuRobo and the URDF
   collision model **once at startup** (~5 s).
2. Each episode triggers one synthesis call: ~3–4 s on an RTX 5070 Ti.
3. The grasp solution is cached as a `[T, 22]` joint-space trajectory and
   played back in Isaac Sim.

No grasp database to build, no offline solver to run. The worker re-uses the
same processed_data + URDF every call; if you changed the mesh you need to
restart the worker (`pkill -f bodex_service.server` or just `client.close()`).

**Exception**: the optional `DexLearn` learning-based sampler **does** need
training. See the next section.

---

## Optional — DexLearn (faster grasp sampling, requires offline training)

`BODex/DexLearn/` has a learning-based grasp sampler. It's **not wired into
the live task** today; the live pipeline uses BODex directly. Use DexLearn
only if BODex's per-call latency (~3–4 s) is too slow for your application
**and** you have many objects to amortize the training cost over.

### When to use

- BODex 3–4 s latency is too slow (e.g. real-time teleoperation, online RL
  training that needs ~10 grasps/s)
- You have a fixed library of objects (training is per-object-set)
- You're OK with slightly less accurate grasps than analytical BODex

### Pipeline at a high level

1. **Generate dataset** — run BODex in batch mode over hundreds of poses of
   each object. Saves a labelled grasp dataset under
   `BODex/src/curobo/content/assets/output/`.
2. **Train DexLearn** — fit a normalizing flow or diffusion model to that
   dataset (~1–2 hours per setup):
   ```bash
   cd /mnt/ssd2/ws/Dualarm_ws/BODex/DexLearn
   conda activate dexlearn   # separate env, see DexLearn/README.md
   CUDA_VISIBLE_DEVICES=0 python -m dexlearn.train \
       exp_name=allegro_v1 algo=nflow data=<your_data_yaml>
   ```
3. **Sample with the trained model** — ~30 ms per grasp. Currently the live
   `bodex_service` does NOT call DexLearn; you'd extend `bodex_plan_cli.py`
   to read DexLearn samples instead of running `GraspSolver`.

For a deeper look, see `BODex/DexLearn/README.md` and the
`COLLISION_AVOIDANCE.md` Level-3 / DexLearn section.

For most users adding a new object: **skip this section.** BODex on its own
is fast enough.

---

## Validation checklist

After adding `<object_id>`, run these in order:

### 1. BODex synthesis works (no Isaac Sim)

```bash
/home/robot/anaconda3/envs/env_isaaclab/bin/python - <<PY
import sys, numpy as np
sys.path.insert(0, "/mnt/ssd2/ws/Dualarm_ws/BODex")
from bodex_service import BODexClient

with BODexClient(verbose=True) as c:
    r = c.synthesize_pick_and_place(
        active_arm="ra",
        object_id="<object_id>",
        object_pose=np.array([0.10, 0.45, 1.05, 1, 0, 0, 0]),
        place_xyz=np.array([0.08, 0.50, 1.05]),
        approach="top",
        seed_num=10,
        table_z=0.96,
    )
    print("OK", r["trajectory"].shape, "grasp_err:", r["grasp_error"])
PY
```

Pass: prints `OK (T, 22) grasp_err: <small>`.
Fail: `Unsupported object type` → check `info/simplified.json` syntax.
Fail: `No seed has all three IK stages feasible` → object_pose is outside the
arm's reach. Adjust spawn xy.

### 2. Single-episode Isaac Sim run

```bash
/home/robot/anaconda3/envs/env_isaaclab/bin/python -u \
    isaaclab_tasks/run_bodex_pick_place.py \
    --active_arm ra --object_id <object_id> --approach top \
    --num_episodes 1 --headless --enable_cameras
```

Pass: `[eval] OVERALL SUCCESS = True`.
Fail (BODex side): see step 1 errors. `RuntimeError: ...mesh not found` →
URDF mesh-path resolution problem; check the `<mesh filename="...">` is
relative to the URDF's parent dir.
Fail (Isaac side): `usd_path does not exist` → fix the catalog `usd=` entry,
or switch to `primitive="cylinder"`.

### 3. Stress test

```bash
/home/robot/anaconda3/envs/env_isaaclab/bin/python -u \
    isaaclab_tasks/run_bodex_pick_place.py \
    --active_arm ra --object_id <object_id> --approach top \
    --num_episodes 10 --headless --enable_cameras
```

Aim for ≥ 80 % full success on the top approach. If lower, the object's mass
or obb is mis-tuned — check that the cube is actually being lifted (look at
the `obs debug` lines for `obj_z`).

### 4. Side approach (if the object is tall)

```bash
... --approach side
```

For tall objects (obb_z > 0.10), the side approach should give ≥ 60 %
success. If 0 %, the orientation filter may be rejecting all seeds; relax
`THUMB_UP_MIN` from 0.65 to 0.5 in `bodex_plan_cli.py` or bump
`--seed_num 20 → 40`.

### 5. Both arms

Repeat with `--active_arm la`. Each arm has its own spawn zone (`ra → +x`,
`la → -x`), so the same `object_id` may need a slight pose offset per arm.

---

## File summary — what gets created/modified per object

```
BODex/src/curobo/content/assets/object/custom/processed_data/<object_id>/
├── mesh/simplified.obj                 ← Path A: trimesh / Path B: MeshProcess output
├── urdf/coacd.urdf                     ← references ../mesh/simplified.obj
└── info/simplified.json                ← {gravity_center, obb, scale, density, mass}

BODex/isaaclab_tasks/assets/Materials/<object_id>/<object_id>.usd   (Path B only)

BODex/isaaclab_tasks/bodex_pick_place/env_cfg.py   ← +1 entry in OBJECT_CATALOG
BODex/isaaclab_tasks/run_bodex_pick_place.py        ← +1 string in --object_id choices
```

That's it — **3 (Path A) or 4 (Path B) files touched** to add an object.

---

## Common pitfalls

| Symptom | Likely cause | Fix |
|---|---|---|
| `Unsupported object type` from BODex CLI | typo in `info/simplified.json` (e.g. wrong key name) | re-validate JSON: must have exactly `gravity_center` (3 floats), `obb` (3 floats), `scale`, `density`, `mass` |
| Object spawns in air and floats forever | mass too small (< 0.01 kg) — PhysX gets unstable | bump mass to ≥ 0.05 kg |
| Object falls THROUGH the table on settle | missing `<collision>` tag in URDF or USD without CollisionAPI | re-export with collision props |
| `mesh filename ... not found` in cuRobo log | URDF `<mesh filename>` path can't be resolved | use **relative** path (`../mesh/simplified.obj`) — cuRobo resolves from URDF's parent folder |
| Side grasp hits the table | object's obb_z < 0.10 m | use `--approach top` instead, or scale the object up |
| Grasp succeeds but place is way off | gravity_center in `info/simplified.json` doesn't match the actual mesh CoM | recompute via `trimesh.Trimesh.center_mass` and update `info/simplified.json` |
| Filter logs `kept 0/N feasible seeds` repeatedly | `THUMB_UP_MIN=0.65` too strict for this object's geometry | relax to 0.5 in `bodex_plan_cli.py` or bump `--seed_num` |

---

## Where to read more

- `bodex_service/COMMANDS.md` — daily commands, all flags, knobs to tune
- `bodex_service/README.md` — `BODexClient` API for use in other Python projects
- `bodex_service/COLLISION_AVOIDANCE.md` — how the planner sees the world
- `isaaclab_tasks/PROGRESS.md` — feature timeline + validated performance
