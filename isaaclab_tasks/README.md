# Isaac Lab BODex Pick-and-Place Task

Single-arm pick-and-place on the t-shape dual-arm torso, using BODex-generated
grasp+mogen trajectories. Passive arm stays frozen at its home pose throughout
each episode.

## Architecture

Two conda envs (as required — only `bodex` is ever modified):

```
Isaac Lab task (env_isaaclab, py3.11)
  ├─ run_bodex_pick_place.py           — launch, env loop
  ├─ bodex_pick_place/env_cfg.py       — scene + object + place marker
  └─ bodex_pick_place/env.py           — reset → call BODex → cache → replay
                      │
                      ▼  subprocess per episode
                 BODex CLI (bodex env, py3.10)
                  └─ bodex_pick_place/bodex_plan_cli.py
                       ├─ fp_scene_cfg → write runtime scene_cfg
                       ├─ grasp solver (existing, 6D `r` bias per approach)
                       ├─ IK per stage (pregrasp / grasp / lift)
                       ├─ place IK (transit / descend)
                       └─ → writes [T, 22] trajectory .npy
                                          ↓
                                  Isaac Lab loads, plays back
```

## Run

```bash
# Headless
/home/robot/anaconda3/envs/env_isaaclab/bin/python \
    /mnt/ssd2/ws/Dualarm_ws/BODex/isaaclab_tasks/run_bodex_pick_place.py \
    --headless --active_arm ra --object_id peach --approach top \
    --num_episodes 3 --max_steps_per_episode 150

# With GUI (Isaac Sim window opens)
/home/robot/anaconda3/envs/env_isaaclab/bin/python \
    /mnt/ssd2/ws/Dualarm_ws/BODex/isaaclab_tasks/run_bodex_pick_place.py \
    --active_arm ra --object_id peach --approach top \
    --num_episodes 1 --max_steps_per_episode 250
```

### Arguments
- `--active_arm {ra, la}` — which arm does the task. The other is frozen.
- `--object_id {peach, red_cube, tomato, mouse}` — must match
  `BODex/src/curobo/content/assets/object/custom/processed_data/<id>/`.
- `--approach {top, side_front, side_left, side_right}` — grasp direction
  bias. Each maps to a specific ortho-6d palm orientation in the seeder.
- `--num_envs N` — parallel envs. Current default 1 (bodex CLI is serial
  per env); scale up only if you implement parallel CLI calls.
- `--num_episodes N` — reset-run loop count.
- `--max_steps_per_episode N` — physics step cap (250 covers ~115 wp
  trajectory comfortably).

## Per-episode flow

1. Object spawns at random `(x, y, z, yaw)` inside the reachable zone:
   - x ∈ [−0.05, 0.25], y ∈ [0.35, 0.50], z ∈ [1.14, 1.18] (ra-reachable).
2. Place target sampled uniformly in the same zone (visible cyan sphere).
3. Reset arms: active → home pose, passive → home pose (frozen).
4. BODex CLI is called with `(object_pose, place_xyz)` → writes a
   `[T, 22]` trajectory. Typical T=115 with default hold_frames=10.
5. Each physics step consumes one waypoint. `set_joint_position_target`
   for 6 arm + 16 hand joints.
6. Episode terminates when the trajectory ends OR the object falls off.

## Adding a new object

Full walkthrough: [QUICKSTART.md → Adding a New Object — End-to-End](../QUICKSTART.md#adding-a-new-object--end-to-end).

Short version (for an object whose `.obj` is already in `BODex/custom_objs/`):

```bash
# 1. Preprocess (in bodex env)
python /mnt/ssd2/ws/Dualarm_ws/BODex/example_grasp/preprocess_custom_objs.py \
    --input_dir custom_objs \
    --processed_root src/curobo/content/assets/object/custom/processed_data \
    --scene_cfg_root src/curobo/content/assets/object/custom/scene_cfg \
    --ra_ur3_xyz 0.1 0.45 1.15 --la_ur3e_xyz -0.1 0.45 1.15 \
    --table_z 1.1

# 2. Convert .obj → .usd (in env_isaaclab) → assets/Materials/<obj>/<obj>.usd
# 3. Add an entry to OBJECT_CATALOG in bodex_pick_place/env_cfg.py
#    (use the existing `peach` entry as the template; pull obb_z from
#    processed_data/<obj>/info/simplified.json)
# 4. Launch with --object_id <obj>
```

## Tuning / extending

| Change | Where |
|---|---|
| Different spawn zone | `object_spawn_{x,y,z}_{ra,la}` in `env_cfg.py`. |
| Different approach | `--approach` flag. New approach directions: edit `APPROACH_PRESETS` in `bodex_plan_cli.py`. |
| More seeds / less grasp error | `--seed_num 40`. |
| Linger longer at grasp (viz) | `--hold_frames 30`. |
| Collision with torso/other arm | Currently the scene has both (passive arm + tshape USD), but BODex's grasp solver doesn't see them as obstacles. To add: insert extra cuboid/mesh entries into the scene_cfg before calling BODex. |
| Place accuracy tuning | `build_place_trajectory` in `bodex_plan_cli.py` — the descend height vs object obb offset. |

## Output

Trajectory artifacts from the BODex side are ephemeral (written to
`tempfile`, deleted after load). The scene_cfg persists at
`BODex/src/curobo/content/assets/object/custom/scene_cfg/<obj>/runtime/latest.npy`
(overwritten per call).

Isaac Sim state at episode-end is logged to stdout:
```
[env 0] trajectory cached: 115 waypoints (pick_end=60)
    final object pos: [0.00515617 0.3568482  1.1524994 ]
    target pos:       [0.09825984 0.4953133  1.14      ]
    distance to target: 0.1673 m  (success if < 0.08)
```
