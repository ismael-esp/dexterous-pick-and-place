"""Run the BODex pick-and-place Isaac Lab task.

Launch with env_isaaclab's Python (has Isaac Sim / Isaac Lab):

  /home/robot/anaconda3/envs/env_isaaclab/bin/python \\
      /mnt/ssd2/ws/Dualarm_ws/BODex/isaaclab_tasks/run_bodex_pick_place.py \\
      --active_arm ra --object_id peach --approach top \\
      --num_episodes 3

Add `--headless` for no Isaac Sim GUI.
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--active_arm", choices=["ra", "la"], default="ra")
    # `--object_id` accepts any string — we look it up in OBJECT_CATALOG at
    # runtime (with a clear error if missing). This way a freshly added object
    # only needs the catalog entry, not also a CLI choice list edit.
    parser.add_argument("--object_id", default="peach")
    # `auto` (default) picks the best approach for the object's geometry and
    # cycles through fallbacks on retry. The explicit values pin a single
    # approach (no cycling).
    parser.add_argument("--approach", default="auto",
                        choices=["auto", "top", "side", "side_front", "side_left", "side_right"])
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=5)
    # With waypoint_repeat=2 and ~165 waypoints (including return) → ~330 steps
    # per attempt. With up to 3 attempts (initial + 2 retries) per episode,
    # 1200 is a safe upper bound.
    # Must be >= cfg.episode_length, else the runner exits the step loop
    # before the env's terminate path runs and the eval reads stale state.
    parser.add_argument("--max_steps_per_episode", type=int, default=2000)
    parser.add_argument("--seed_num", type=int, default=14)
    parser.add_argument("--no_record", action="store_true",
                        help="Disable frame recording entirely (faster headless tuning runs).")
    parser.add_argument("--hold_frames", type=int, default=10)
    parser.add_argument("--waypoint_repeat", type=int, default=2,
                        help="Replay each BODex waypoint this many physics steps.")
    parser.add_argument("--fast_regrasp", action="store_true",
                        help="Skip return-to-home between failed attempts. The arm "
                             "retreats by `regrasp_retreat_z` (default 10 cm) with hand "
                             "open, then lerps directly to BODex's freshly planned "
                             "pre-grasp instead of detouring through the home pose.")
    parser.add_argument("--video_path", default=None,
                        help="Override the recorded video output path. Default: cfg's "
                             "camera_video_path (/tmp/bodex_pick_place.mp4).")

    # Isaac Sim / AppLauncher options (parses --headless, --device, --enable_cameras, …)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Isaac-Sim-must-be-up imports
    import torch

    # Add our package path so "from bodex_pick_place.env_cfg ..." works
    import os
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)

    from bodex_pick_place.env_cfg import BODexPickPlaceEnvCfg
    from bodex_pick_place.env import BODexPickPlaceEnv

    cfg = BODexPickPlaceEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.active_arm = args.active_arm
    cfg.object_id = args.object_id
    cfg.approach = args.approach
    cfg.bodex_seed_num = args.seed_num
    cfg.hold_frames = args.hold_frames
    cfg.waypoint_repeat = args.waypoint_repeat
    cfg.fast_regrasp_enable = args.fast_regrasp
    if args.no_record:
        cfg.enable_camera = False
    if args.video_path:
        cfg.camera_video_path = args.video_path
    cfg.__post_init__()   # rebuild object_cfg with the chosen object_id

    env = BODexPickPlaceEnv(cfg=cfg)

    ep_results = []
    for ep in range(args.num_episodes):
        print(f"\n=== Episode {ep + 1}/{args.num_episodes} ===")
        env.reset()
        step = 0
        done = torch.zeros(cfg.scene.num_envs, dtype=torch.bool, device=env.device)
        while not done.all() and step < args.max_steps_per_episode:
            obs, rew, terminated, truncated, info = env.step(
                torch.zeros(cfg.scene.num_envs, cfg.action_space, device=env.device)
                if cfg.action_space > 0 else
                torch.zeros(cfg.scene.num_envs, 0, device=env.device)
            )
            done = terminated | truncated
            step += 1

        # Automated success evaluation
        result = env.evaluate_episode()
        ep_results.append(result)
        print(f"  [eval] lifted_by        = {result['lifted_by_m']:.4f} m  "
              f"grasp_success={result['grasp_success']}")
        print(f"  [eval] final_obj_pos    = "
              f"({result['final_obj_pos'][0]:.3f}, {result['final_obj_pos'][1]:.3f}, "
              f"{result['final_obj_pos'][2]:.3f})")
        print(f"  [eval] target_pos       = "
              f"({result['target_pos'][0]:.3f}, {result['target_pos'][1]:.3f}, "
              f"{result['target_pos'][2]:.3f})")
        xy_tol = result.get("xy_tol_m", 0.12)
        z_tol = result.get("z_tol_m", 0.05)
        print(f"  [eval] final_xy_err     = {result['final_xy_err_m']:.4f} m "
              f"(success if < {xy_tol:.3f})")
        print(f"  [eval] final_z_err      = {result['final_z_err_m']:.4f} m "
              f"(success if < {z_tol:.3f})")
        print(f"  [eval] place_success    = {result['place_success']}")
        print(f"  [eval] attempts         = {result.get('attempts', 1)}")
        print(f"  [eval] OVERALL SUCCESS  = "
              f"{result['grasp_success'] and result['place_success']}")

    # Post-run summary
    n = len(ep_results)
    n_grasp = sum(1 for r in ep_results if r["grasp_success"])
    n_place = sum(1 for r in ep_results if r["place_success"])
    n_full = sum(1 for r in ep_results if r["grasp_success"] and r["place_success"])
    print(f"\n=== Summary over {n} episodes ===")
    print(f"  grasp success : {n_grasp}/{n} ({100 * n_grasp / n:.0f}%)")
    print(f"  place success : {n_place}/{n} ({100 * n_place / n:.0f}%)")
    print(f"  full success  : {n_full}/{n} ({100 * n_full / n:.0f}%)")

    # Stitch recorded frames into a video
    video_path = env.stitch_video_if_configured()
    if video_path:
        print(f"  recorded video: {video_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
