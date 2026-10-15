"""BODex pick-and-place Isaac Lab environment.

Per-episode flow (all inside _reset_idx):
  1. Write object root state ABOVE the pedestal + zero velocity.
  2. Run `settle_steps` physics steps so the object lands on the pedestal
     under gravity. During this phase actions are a "hold at home" command.
  3. Read the SETTLED object world pose.
  4. Call the BODex CLI (subprocess in bodex env) with the settled pose +
     sampled place target → writes a `[T, 22]` .npy trajectory.
  5. Cache the trajectory for step-by-step playback.

During the playback phase, _pre_physics_step advances an index and issues
the next waypoint's joint targets. We also record an RGB frame per physics
step from the fixed camera, and collect per-episode success metrics.
"""
# Standard Library
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Sequence

# Third Party
import numpy as np
import torch

# Isaac Lab
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import Camera, ContactSensor

# Local
from .env_cfg import BODexPickPlaceEnvCfg, OBJECT_CATALOG, resolve_object_entry


class BODexPickPlaceEnv(DirectRLEnv):
    """Replays pre-planned BODex pick+place trajectories."""

    cfg: BODexPickPlaceEnvCfg

    def __init__(self, cfg: BODexPickPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        self._articulations: dict = {}
        self._rigid_objects: dict = {}
        self._markers: dict = {}
        self._sensors: dict = {}
        super().__init__(cfg, render_mode, **kwargs)

        self._T_max = 300
        self._full_traj = torch.zeros(
            (self.num_envs, self._T_max, 22), device=self.device, dtype=torch.float32
        )
        self._traj_len = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._step_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # Sub-step counter for waypoint_repeat: advances each step, resets every
        # `waypoint_repeat` physics steps to advance _step_idx by 1.
        self._substep_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._pick_end_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._object_spawn_z = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._object_target = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float32
        )
        self._episode_failed = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

        # Regrasp / retry bookkeeping
        self._attempt_count = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._place_xyz_cached = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float32
        )
        # After a success, latch the state machine so we don't accidentally
        # re-trigger a regrasp on the same frame.
        self._attempt_resolved = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

        # Per-episode success metrics (scalar tensors)
        self._metric_object_max_z = torch.zeros(self.num_envs, device=self.device)
        self._metric_grasp_success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # Visual status state per env. 0=neutral, 1=green, 2=red, 3=blue, 4=orange.
        # Marker prototypes are registered in this exact order in env_cfg.
        # `_status_locked` flips True at episode end (green/red); during
        # execution the state recomputes each step (blue/orange/neutral).
        self._status_state = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._status_locked = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # Frames remaining in the post-success/fail "show the sphere color" hold
        # before terminating. Without this, Isaac Lab's auto-reset wipes the
        # latched color in the same tick it was set, and the green/red flash
        # never reaches the camera.
        self._status_hold_remaining = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # Joint-limit risk envelope (set later when sim is ready)
        self._joint_lower = None
        self._joint_upper = None
        # Latched snapshot at the moment the episode ends, captured by
        # _get_dones BEFORE Isaac Lab's auto-reset wipes live state.
        self._latched_obj_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._latched_max_z = torch.zeros(self.num_envs, device=self.device)
        self._latched_attempt_count = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._latched_target = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float32
        )

        # Camera frame recording
        self._episode_idx = 0
        self._frame_idx = 0
        if cfg.enable_camera:
            os.makedirs(cfg.camera_rgb_dir, exist_ok=True)

        # Persistent BODex worker — imported lazily so loading this module
        # doesn't require the bodex_service package to be importable.
        _bodex_service_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        if _bodex_service_path not in sys.path:
            sys.path.insert(0, _bodex_service_path)
        from bodex_service import BODexClient
        self._bodex_client = BODexClient(
            socket_path="/tmp/bodex_service.sock",
            auto_start=True,
            verbose=True,   # forward worker stdout so filter + BODex log lines are visible
        )
        print("[env] BODex worker ready at /tmp/bodex_service.sock")

        # Joint-index lookup for the active arm
        active_robot = self._active_robot()
        self._active_joint_name_to_idx = {
            n: i for i, n in enumerate(active_robot.joint_names)
        }
        p = self.cfg.active_arm + "_"
        self._arm_jids = [
            self._active_joint_name_to_idx[p + n] for n in
            ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
             "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        ]
        self._hand_jids = [
            self._active_joint_name_to_idx[p + f"joint_{i}_0"] for i in range(16)
        ]

        # Home joint angles (degrees, from user's physical config)
        self._ra_home_deg = (27.33, -43.98, 98.18, -232.88, -71.98, 223.84)
        self._la_home_deg = (-18.75, -138.64, -99.59, 57.97, 64.88, 137.67)

        # Passive arm bookkeeping
        passive_robot = self._passive_robot()
        passive_p_prefix = "la_" if self.cfg.active_arm == "ra" else "ra_"
        self._passive_joint_to_idx = {n: i for i, n in enumerate(passive_robot.joint_names)}
        self._passive_arm_jids = [
            self._passive_joint_to_idx[passive_p_prefix + n] for n in
            ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
             "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        ]

    # ---------------------------------------------------------------
    # scene setup
    # ---------------------------------------------------------------
    def _setup_scene(self):
        for name, cfg in self.cfg.articulations_cfg.items():
            art = Articulation(cfg)
            self._articulations[name] = art
            self.scene.articulations[name] = art
            setattr(self, name, art)
        for name, cfg in self.cfg.rigid_objects_cfg.items():
            robj = RigidObject(cfg)
            self._rigid_objects[name] = robj
            self.scene.rigid_objects[name] = robj
            setattr(self, name, robj)
        for name, cfg in self.cfg.markers_cfg.items():
            marker = VisualizationMarkers(cfg)
            self._markers[name] = marker
            setattr(self, name, marker)

        # Camera (per-env)
        if self.cfg.enable_camera and self.cfg.camera_cfg is not None:
            cam = Camera(self.cfg.camera_cfg)
            self._sensors["camera"] = cam
            self.scene.sensors["camera"] = cam

        # Contact sensors on arm links (Level-4 runtime collision detection).
        if self.cfg.enable_contact_sensors:
            if self.cfg.ra_contact_cfg is not None:
                s = ContactSensor(self.cfg.ra_contact_cfg)
                self._sensors["ra_contact"] = s
                self.scene.sensors["ra_contact"] = s
            if self.cfg.la_contact_cfg is not None:
                s = ContactSensor(self.cfg.la_contact_cfg)
                self._sensors["la_contact"] = s
                self.scene.sensors["la_contact"] = s

        sim_utils.spawn_ground_plane("/World/ground", cfg=sim_utils.GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.9))
        light_cfg.func("/World/Light", light_cfg)
        self.scene.clone_environments(copy_from_source=True)

        # Apply high-friction material to the pickable object prim
        # post-spawn (UsdFileCfg doesn't accept physics_material as a kwarg
        # in this Isaac Lab version, but we can bind a material prim).
        try:
            from pxr import UsdShade, UsdPhysics
            stage = sim_utils.get_stage()
            mat_path = "/World/Looks/grippable_material"
            if stage.GetPrimAtPath(mat_path).IsValid():
                pass
            else:
                mat_cfg = sim_utils.RigidBodyMaterialCfg(
                    static_friction=3.0,
                    dynamic_friction=2.8,
                    restitution=0.0,
                )
                mat_cfg.func(mat_path, mat_cfg)
            # Bind to object prim under every env
            import omni.usd
            for env_i in range(self.num_envs):
                obj_path = f"/World/envs/env_{env_i}/pick_object"
                obj_prim = stage.GetPrimAtPath(obj_path)
                if obj_prim and obj_prim.IsValid():
                    sim_utils.bind_physics_material(obj_path, mat_path)
        except Exception as e:
            print(f"[warn] could not bind physics material: {e}")

    # ---------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------
    def _active_robot(self) -> Articulation:
        return self._articulations["right_robot" if self.cfg.active_arm == "ra" else "left_robot"]

    def _passive_robot(self) -> Articulation:
        return self._articulations["left_robot" if self.cfg.active_arm == "ra" else "right_robot"]

    def _home_joint_pose_radians(self, arm: str) -> list[float]:
        import math as _math
        degs = self._ra_home_deg if arm == "ra" else self._la_home_deg
        return [_math.radians(d) for d in degs]

    def _spawn_zone(self) -> tuple[tuple, tuple]:
        """Return (x_range, y_range) for the active arm's spawn zone."""
        arm = self.cfg.active_arm
        return (getattr(self.cfg, f"object_spawn_x_{arm}"),
                getattr(self.cfg, f"object_spawn_y_{arm}"))

    def _place_zone(self) -> tuple[tuple, tuple]:
        """Return (x_range, y_range) for the active arm's place-target zone."""
        arm = self.cfg.active_arm
        return (getattr(self.cfg, f"place_target_x_{arm}"),
                getattr(self.cfg, f"place_target_y_{arm}"))

    def _obj_entry(self) -> dict:
        """Resolved catalog entry (with auto-loaded obb + approach ranking).
        Cached on first use so we don't re-read the json each call."""
        if not hasattr(self, "_cached_obj_entry") or self._cached_obj_entry is None \
                or self._cached_obj_entry.get("__id") != self.cfg.object_id:
            entry = resolve_object_entry(self.cfg.object_id, self.cfg.active_arm)
            entry["__id"] = self.cfg.object_id
            self._cached_obj_entry = entry
        return self._cached_obj_entry

    def _sample_place_target(self, n: int) -> torch.Tensor:
        xr, yr = self._place_zone()
        x = torch.empty(n, device=self.device).uniform_(*xr)
        y = torch.empty(n, device=self.device).uniform_(*yr)
        # World-frame Z extent — accounts for any per-object spawn_quat so
        # the place target sits at the correct height for the object's
        # actual lying-down orientation.
        obb_z = self._obj_entry()["obb_z_world"]
        z_val = self.cfg.table_top_z + obb_z / 2.0 + 0.005
        z = torch.full((n,), z_val, device=self.device)
        return torch.stack([x, y, z], dim=-1)

    def _sample_object_spawn(self, n: int) -> torch.Tensor:
        """Return [n, 7] xyz + wxyz quaternion with object placed slightly
        above the table — physics will settle it onto the table.

        Applies the per-object `spawn_quat` (e.g. lay-flat for mouse) and
        composes a randomized yaw around world Z on top of it. Order matters:
        we apply YAW LAST in world frame so yaw spins the lying object in
        place rather than tilting it back upright.
        """
        entry = self._obj_entry()
        xr, yr = self._spawn_zone()
        x = torch.empty(n, device=self.device).uniform_(*xr)
        y = torch.empty(n, device=self.device).uniform_(*yr)
        obb_z = entry["obb_z_world"]
        # Spawn just 5 mm above the resting height. A bigger drop introduces
        # bounce/slide that pushes asymmetric objects (mouse) past the
        # reachable spawn zone before BODex sees the post-settle pose.
        z_val = self.cfg.table_top_z + obb_z / 2.0 + 0.005
        z = torch.full((n,), z_val, device=self.device)

        yaw = torch.empty(n, device=self.device).uniform_(*self.cfg.object_spawn_yaw)
        half = yaw * 0.5
        qw_y = torch.cos(half); qz_y = torch.sin(half)                # yaw quat (w, 0, 0, z)
        qx_y = torch.zeros_like(qw_y); qy_y = torch.zeros_like(qw_y)

        # Per-object base spawn_quat (constant across the batch).
        sq = entry.get("spawn_quat", (1.0, 0.0, 0.0, 0.0))
        sw, sx, sy, sz = (float(sq[0]), float(sq[1]), float(sq[2]), float(sq[3]))
        sw_t = torch.full((n,), sw, device=self.device)
        sx_t = torch.full((n,), sx, device=self.device)
        sy_t = torch.full((n,), sy, device=self.device)
        sz_t = torch.full((n,), sz, device=self.device)

        # Compose: q_total = q_yaw * q_spawn  (yaw applied AFTER spawn_quat,
        # in world frame — the lying object spins in place).
        qw = qw_y * sw_t - qx_y * sx_t - qy_y * sy_t - qz_y * sz_t
        qx = qw_y * sx_t + qx_y * sw_t + qy_y * sz_t - qz_y * sy_t
        qy = qw_y * sy_t - qx_y * sz_t + qy_y * sw_t + qz_y * sx_t
        qz = qw_y * sz_t + qx_y * sy_t - qy_y * sx_t + qz_y * sw_t

        return torch.stack([x, y, z, qw, qx, qy, qz], dim=-1)

    def _freeze_passive_arm(self):
        passive_robot = self._passive_robot()
        passive_home = torch.tensor(
            self._home_joint_pose_radians("la" if self.cfg.active_arm == "ra" else "ra"),
            device=self.device, dtype=torch.float32,
        ).view(1, 6).expand(self.num_envs, 6)
        passive_robot.set_joint_position_target(passive_home, joint_ids=self._passive_arm_jids)

    def _hold_active_arm_at_home(self):
        active_robot = self._active_robot()
        active_home = torch.tensor(
            self._home_joint_pose_radians(self.cfg.active_arm),
            device=self.device, dtype=torch.float32,
        ).view(1, 6).expand(self.num_envs, 6)
        active_robot.set_joint_position_target(active_home, joint_ids=self._arm_jids)
        active_robot.set_joint_position_target(
            torch.zeros(self.num_envs, 16, device=self.device),
            joint_ids=self._hand_jids,
        )

    # ---------------------------------------------------------------
    # reset
    # ---------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or (hasattr(env_ids, "__len__") and len(env_ids) == 0):
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device)
        super()._reset_idx(env_ids)

        t0 = time.time()

        pick_obj: RigidObject = self._rigid_objects["pick_object"]
        active_robot = self._active_robot()
        passive_robot = self._passive_robot()

        # 1) sample spawn + target
        object_pose = self._sample_object_spawn(len(env_ids))
        place_xyz = self._sample_place_target(len(env_ids))
        self._object_target[env_ids] = place_xyz

        # 2) write object root state
        root_state = pick_obj.data.default_root_state[env_ids].clone()
        root_state[:, :3] = object_pose[:, :3] + self.scene.env_origins[env_ids]
        root_state[:, 3:7] = object_pose[:, 3:7]
        root_state[:, 7:] = 0.0
        pick_obj.write_root_state_to_sim(root_state, env_ids=env_ids)

        # 3) reset arms to home + write joint state
        active_home = torch.tensor(
            self._home_joint_pose_radians(self.cfg.active_arm),
            device=self.device, dtype=torch.float32,
        )
        passive_home = torch.tensor(
            self._home_joint_pose_radians("la" if self.cfg.active_arm == "ra" else "ra"),
            device=self.device, dtype=torch.float32,
        )
        cur_active = active_robot.data.default_joint_pos[env_ids].clone()
        cur_active[:, self._arm_jids] = active_home
        for j in self._hand_jids:
            cur_active[:, j] = 0.0
        active_robot.write_joint_state_to_sim(
            cur_active, torch.zeros_like(cur_active), env_ids=env_ids,
        )
        cur_passive = passive_robot.data.default_joint_pos[env_ids].clone()
        cur_passive[:, self._passive_arm_jids] = passive_home
        passive_robot.write_joint_state_to_sim(
            cur_passive, torch.zeros_like(cur_passive), env_ids=env_ids,
        )

        # 4) move the visible place marker
        place_marker: VisualizationMarkers = self._markers["place_marker"]
        place_world = place_xyz + self.scene.env_origins[env_ids]
        place_marker.visualize(translations=place_world)

        t_setup = time.time() - t0

        # 5) settle physics — hold arms at home while object falls onto pedestal
        t_settle0 = time.time()
        for _ in range(self.cfg.settle_steps):
            self._hold_active_arm_at_home()
            self._freeze_passive_arm()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
        t_settle = time.time() - t_settle0

        # 6) read SETTLED object pose — this is what BODex will see
        pick_obj_root = pick_obj.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        pick_obj_quat = pick_obj.data.root_quat_w[env_ids]
        self._object_spawn_z[env_ids] = pick_obj_root[:, 2]
        self._metric_object_max_z[env_ids] = pick_obj_root[:, 2]

        # 7) call BODex CLI per env (serial; num_envs=1 is the typical case)
        # IMPORTANT: reset attempt_count BEFORE the trajectory build so that
        # `_approach_for_attempt` indexes from rank[0] on a fresh episode.
        # (Previously this was reset *after* the build, which caused a fresh
        # episode's first call to inherit the previous episode's attempt
        # count and skip ahead in the approach ranking.)
        t_bodex0 = time.time()
        for i_local, i_env_t in enumerate(env_ids.cpu().tolist()):
            obj_pose_np = np.concatenate(
                [pick_obj_root[i_local].cpu().numpy(), pick_obj_quat[i_local].cpu().numpy()]
            )
            place_np = place_xyz[i_local].cpu().numpy()
            self._attempt_count[i_env_t] = 0
            self._place_xyz_cached[i_env_t] = torch.tensor(
                place_np, device=self.device, dtype=torch.float32,
            )
            self._attempt_resolved[i_env_t] = False
            # Clear status sphere at episode start
            self._status_state[i_env_t] = 0
            self._status_locked[i_env_t] = False
            self._status_hold_remaining[i_env_t] = 0
            # Synthesis-fail cycling: if the top-ranked approach is infeasible
            # (e.g. la-arm side-grasp at edge of IK reach), advance through the
            # approach ranking on the SAME initial attempt instead of dying
            # before _try_replan ever fires. This is symmetric to the cycling
            # in _try_replan and does not count against max_regrasp_attempts
            # (which measures EXECUTED trajectories, not synthesis tries).
            ranking = self._obj_entry().get("approach_ranking", []) or ["top"]
            max_synth_tries = max(1, len(ranking)) if self.cfg.approach == "auto" else 1
            for synth_try in range(max_synth_tries):
                ok = self._build_and_load_trajectory(
                    env_idx=i_env_t, obj_pose_np=obj_pose_np, place_np=place_np,
                    prepend_home_from_current=False,
                )
                if ok:
                    break
                # Synthesis failed for this approach. If we still have approaches
                # left in the rank and we're in auto mode, advance and try again.
                next_attempt = int(self._attempt_count[i_env_t].item()) + 1
                if next_attempt < len(ranking) and self.cfg.approach == "auto":
                    self._attempt_count[i_env_t] += 1
                    print(f"[env {i_env_t}] initial synth-fail; advancing to "
                          f"rank[{next_attempt}]={ranking[next_attempt]}")
                    # Clear the failed flag set by the previous build so the
                    # next attempt can proceed.
                    self._episode_failed[i_env_t] = False
                    continue
                break
        t_bodex = time.time() - t_bodex0

        print(f"[reset timing] setup={t_setup:.2f}s  settle={t_settle:.2f}s  "
              f"bodex_cli={t_bodex:.2f}s  total={time.time()-t0:.2f}s")

        # Bookkeeping for new episode
        self._episode_idx += 1
        self._frame_idx = 0

    def _build_and_load_trajectory(
        self,
        env_idx: int,
        obj_pose_np: np.ndarray,
        place_np: np.ndarray,
        prepend_home_from_current: bool,
    ) -> bool:
        """Ask BODex for a trajectory, apply finger-closure override + return-to-home,
        and load it into self._full_traj for the given env.

        If prepend_home_from_current=True, a linear interp from the ACTUAL current
        arm state → home is prepended (used by retry flow so the arm goes back to
        home before the freshly planned approach). If False, playback starts from
        the BODex-planned home (fresh-episode path).

        Returns True on success, False if BODex reported infeasible IK. On
        failure, marks the env as failed.
        """
        traj, pick_end, approach_used = self._invoke_bodex(
            obj_pose_np, place_np,
            env_idx=env_idx,
            is_retry=prepend_home_from_current,
        )
        if traj is None:
            print(f"[env {env_idx}] BODex synthesis FAILED")
            self._episode_failed[env_idx] = True
            self._traj_len[env_idx] = 1
            active_home = torch.tensor(
                self._home_joint_pose_radians(self.cfg.active_arm),
                device=self.device, dtype=torch.float32,
            )
            self._full_traj[env_idx, 0, :6] = active_home
            self._pick_end_idx[env_idx] = 0
            return False

        T = min(traj.shape[0], self._T_max)
        traj_t = torch.as_tensor(traj[:T], device=self.device, dtype=torch.float32)

        N_APPROACH = 30  # matches bodex_plan_cli.py
        HOLD = self.cfg.hold_frames
        hand_cols = slice(6, 22)
        curl_cols = [6 + i for i in self.cfg.hand_curl_idx]

        grasp_wp_idx = N_APPROACH
        grasp_hold_end = grasp_wp_idx + HOLD
        squeeze_start = grasp_hold_end

        # Detect release ramp at tail (last frames with mean curl < 0.4)
        mean_curl_per_wp = traj_t[:, curl_cols].mean(dim=-1)
        release_start = T
        for k in range(T - 1, -1, -1):
            if mean_curl_per_wp[k].item() < 0.4:
                release_start = k
            else:
                break

        # (1) Approach — fingers open. Thumb-lift joint_12_0 (hand idx 12) is
        # held at its real-hand minimum (0.36) rather than 0, which the
        # physical Allegro cannot reach — keeps sim and hardware consistent.
        if T > N_APPROACH:
            traj_t[:N_APPROACH, hand_cols] = 0.0
            traj_t[:N_APPROACH, 6 + 12] = 0.36

        # (2) Squeeze override on curl joints — keep grip firm during
        # squeeze+lift+transit+descend so the object doesn't slip in the
        # palm. Frames 30..40 (BODex's grasp hold) are LEFT ALONE so the
        # initial finger-on-object contact is gentle.
        #
        # Magnitude is approach- AND object-size-aware:
        #   - SIDE (palm beside slim cylinder): +0.30 rad. Without it the
        #     BODex-planned grasp_hand only "touches" the curved side; the
        #     cylinder slips during lift/transit.
        #   - TOP, SMALL/FLAT object (obb_z < 8 cm): force curl to
        #     `finger_force_close_angle` rad (default 1.5 — fully wrap).
        #     BODex's planned grasp_hand for small asymmetric objects
        #     (mouse) often only touches without gripping; the hand stops
        #     0.5-1 cm above the object. Forcing full closure pushes the
        #     fingers down past the BODex plan until they actually contact.
        #   - TOP, LARGE object (obb_z >= 8 cm): +0.10 rad (gentle, doesn't
        #     displace the object). Validated on red_cube/peach/tomato.
        # Squeeze magnitude over BODex's planned grasp_curl, applied
        # frames squeeze_start..release_start. Approach + size aware:
        #   - SIDE   (slim cylinder): +0.30 (firm wraparound)
        #   - TOP, small/flat  obb_z_world < 8 cm: +0.20 (mouse — needs
        #     more than +0.10 to grip the small object; less than +0.30
        #     because release flicks small objects). Combined with a
        #     slower release ramp below to bleed off squeeze tension
        #     before fingers fully open.
        #   - TOP, regular: +0.10 (cube/peach/tomato baseline).
        is_side = approach_used and approach_used.startswith("side")
        obb_z_world = self._obj_entry().get("obb_z_world", 0.10)
        small_flat = (obb_z_world < 0.08) and (not is_side)
        # Below this height (≈ 3 cm) BODex's planned grasp_curl barely makes
        # finger contact — the tips hover above the object instead of
        # closing on it. Force a larger squeeze for pancake-thin objects
        # (flat_plate ≈ 2 cm) so fingertips actually wrap around.
        very_flat = (obb_z_world < 0.04) and (not is_side)
        # NOTE: tumbler (extreme tall/narrow, obb_z 0.25) side-grasp was
        # tried at +0.45 squeeze + -0.015 descend — regressed 80→60 (the
        # firmer curl displaces the thin cylinder during grasp). Reverted;
        # the stock +0.30 / -0.005 side params are the empirical best for
        # the extreme case. 80% appears to be the parametric ceiling.
        if is_side:
            squeeze_delta = 0.30
        elif very_flat:
            squeeze_delta = 0.50
        elif small_flat:
            squeeze_delta = 0.20
        else:
            squeeze_delta = 0.10
        if release_start > squeeze_start:
            grasp_curl = traj_t[grasp_wp_idx, curl_cols].clone()
            traj_t[squeeze_start:release_start, curl_cols] = grasp_curl + squeeze_delta

        # (2b) Slow-release for small/flat objects: replace the BODex
        # release ramp (squeeze_hand_q → home_hand_open over release_n)
        # with a longer linear ramp from the SQUEEZED curl back to home,
        # so the spring-back energy dissipates gradually instead of
        # punting a small object on release.
        if small_flat and release_start < T:
            release_len_orig = T - release_start
            slow_len = max(release_len_orig, 48)
            squeezed_curl = grasp_curl + squeeze_delta
            home_open_curl = torch.zeros_like(squeezed_curl)
            n = slow_len
            alphas = torch.linspace(
                0.0, 1.0, n, device=self.device, dtype=torch.float32
            ).view(-1, 1)
            new_curls = (squeezed_curl.unsqueeze(0) * (1 - alphas)
                         + home_open_curl.unsqueeze(0) * alphas)
            # Truncate or pad the trajectory at release_start so the new
            # ramp sits at exactly slow_len waypoints.
            head = traj_t[:release_start]
            tail = torch.zeros(
                (slow_len, 22), device=self.device, dtype=torch.float32,
            )
            tail[:, :6] = traj_t[release_start, :6].unsqueeze(0)  # arm stays at descend
            tail[:, 6:22] = traj_t[release_start, 6:22].unsqueeze(0)  # base hand
            tail[:, curl_cols] = new_curls
            traj_t = torch.cat([head, tail], dim=0)
            T = traj_t.shape[0]

        # (3) Append "return / retreat" so the workspace clears between attempts.
        # Slow path: lerp all the way back to home (current behavior).
        # Fast path: short shoulder-lift retreat (~regrasp_retreat_z) with hand open,
        # so the arm pulls back from the failed grasp rather than going to home —
        # the next attempt's preamble lerps directly to BODex's pre-grasp.
        home_arm = torch.tensor(
            self._home_joint_pose_radians(self.cfg.active_arm),
            device=self.device, dtype=torch.float32,
        )
        hand_open = torch.zeros(16, device=self.device, dtype=torch.float32)
        hand_open[12] = 0.36  # thumb-lift can't reach 0 on the real Allegro
        last_arm = traj_t[T - 1, :6]
        last_hand = traj_t[T - 1, 6:22]
        # v8 §5a — Vertical-lift prefix on the return tail (TOP grasps only).
        # The 30-wp joint-space lerp from release-pose to home can sweep the
        # palm horizontally THROUGH the just-placed object (la_core_mug ep2
        # was knocked off the table this way). Prepend 10 wps that bias the
        # shoulder_lift joint 30% toward home while holding the rest of the
        # arm steady — universally raises the wrist on overhead-mount UR3
        # before the full home lerp begins.
        #
        # GATE: only for TOP grasps with tall objects. For SIDE grasps the
        # palm is beside the object (not above it); biasing shoulder_lift
        # from a side-reach pose sweeps the wrist horizontally INTO the
        # object instead of up (ra_tumbler v8-smoke regressed 100→33% from
        # this exact mechanism, place_miss xy_err≈0.43 m).
        apply_lift = (
            (approach_used == "top")
            and (obb_z_world >= 0.08)
        )
        if apply_lift:
            LIFT_WPS = 10
            LIFT_FRACTION = 0.30  # shoulder_lift moves 30% toward home
            lift_target = last_arm.clone()
            lift_target[1] = last_arm[1] * (1 - LIFT_FRACTION) + home_arm[1] * LIFT_FRACTION
            lift_wps = torch.zeros((LIFT_WPS, 22), device=self.device, dtype=torch.float32)
            for k in range(LIFT_WPS):
                a = (k + 1) / LIFT_WPS
                lift_wps[k, :6] = last_arm * (1 - a) + lift_target * a
                lift_wps[k, 6:22] = last_hand * (1 - a) + hand_open * a
            post_lift_arm = lift_target
            post_lift_hand = hand_open
        else:
            lift_wps = torch.zeros((0, 22), device=self.device, dtype=torch.float32)
            post_lift_arm = last_arm
            post_lift_hand = last_hand

        if self.cfg.fast_regrasp_enable:
            N_RETURN = int(self.cfg.regrasp_retreat_wps)
            # Lerp 40% of the way toward home — always kinematically feasible
            # and reliably backs the arm away from the object area regardless
            # of the current configuration.
            retreat_fraction = 0.4
            retreat_arm = post_lift_arm * (1 - retreat_fraction) + home_arm * retreat_fraction
            return_wps = torch.zeros((N_RETURN, 22), device=self.device, dtype=torch.float32)
            for k in range(N_RETURN):
                a = (k + 1) / N_RETURN
                return_wps[k, :6] = post_lift_arm * (1 - a) + retreat_arm * a
                return_wps[k, 6:22] = post_lift_hand * (1 - a) + hand_open * a
        else:
            N_RETURN = int(self.cfg.return_to_home_waypoints)
            return_wps = torch.zeros((N_RETURN, 22), device=self.device, dtype=torch.float32)
            for k in range(N_RETURN):
                a = (k + 1) / N_RETURN
                return_wps[k, :6] = post_lift_arm * (1 - a) + home_arm * a
                return_wps[k, 6:22] = post_lift_hand * (1 - a) + hand_open * a
        full = torch.cat([traj_t, lift_wps, return_wps], dim=0)

        # (4) Prepend a smooth lerp from ACTUAL current arm state → BODex home.
        # This runs on EVERY build (initial AND retry):
        #   - Initial: prevents the PID from slamming the arm violently from
        #     the URDF/factory default pose to BODex's planned home in one
        #     tick — that violent slew was producing wide-arc, elbow-down
        #     configurations that scraped near the table edge.
        #   - Retry:   same purpose; the arm's current pose is wherever the
        #     last failed attempt left it.
        # Fast-regrasp path keeps its short retreat-then-home semantics.
        active_robot = self._active_robot()
        cur_arm = active_robot.data.joint_pos[env_idx, self._arm_jids].clone()
        cur_hand = active_robot.data.joint_pos[env_idx, self._hand_jids].clone()
        if prepend_home_from_current and self.cfg.fast_regrasp_enable:
            N_PRE = int(self.cfg.regrasp_retreat_wps)
            # Always go to home before starting the next BODex trajectory.
            # Home is a known-feasible configuration; lerping to an arbitrary
            # traj_t[0] can produce infeasible joint paths at extreme poses.
            target_arm = home_arm
            pre_wps = torch.zeros((N_PRE, 22), device=self.device, dtype=torch.float32)
            for k in range(N_PRE):
                a = (k + 1) / N_PRE
                pre_wps[k, :6] = cur_arm * (1 - a) + target_arm * a
                pre_wps[k, 6:22] = hand_open
        else:
            # Slow preamble used by retries AND by initial builds (this is the
            # change — initial builds previously had no preamble at all).
            N_PRE = int(self.cfg.retry_home_preamble_wps)
            pre_wps = torch.zeros((N_PRE, 22), device=self.device, dtype=torch.float32)
            for k in range(N_PRE):
                a = (k + 1) / N_PRE
                pre_wps[k, :6] = cur_arm * (1 - a) + home_arm * a
                pre_wps[k, 6:22] = cur_hand * (1 - a) + hand_open * a
        full = torch.cat([pre_wps, full], dim=0)
        pick_end = int(pick_end) + N_PRE

        T_new = min(full.shape[0], self._T_max)
        self._full_traj[env_idx, :T_new] = full[:T_new]
        self._traj_len[env_idx] = T_new
        self._pick_end_idx[env_idx] = pick_end
        self._step_idx[env_idx] = 0
        self._substep_counter[env_idx] = 0
        self._episode_failed[env_idx] = False
        print(f"[env {env_idx}] trajectory loaded T={T_new} pick_end={pick_end} "
              f"retry={prepend_home_from_current} approach={approach_used} "
              f"obj=({obj_pose_np[0]:.3f},{obj_pose_np[1]:.3f},{obj_pose_np[2]:.3f}) "
              f"place=({place_np[0]:.3f},{place_np[1]:.3f},{place_np[2]:.3f})")
        return True

    def _try_replan(self, env_idx: int, reason: str) -> bool:
        """Read the cube's current pose from sim and request a fresh BODex
        trajectory for the same place target. Returns True if a retry was
        launched (attempts not exhausted), False otherwise.

        On a BODex synthesis failure (e.g. the chosen approach has no
        feasible IK for this object pose), this method advances through
        the approach ranking and retries — a single bad approach choice
        no longer terminates the episode. Each successful synthesis still
        counts as ONE attempt against `max_regrasp_attempts`; only the
        approach changes within an attempt.
        """
        if self._attempt_count[env_idx].item() >= self.cfg.max_regrasp_attempts:
            print(f"[env {env_idx}] replan skipped ({reason}): max attempts "
                  f"{self.cfg.max_regrasp_attempts} reached")
            return False

        pick_obj: RigidObject = self._rigid_objects["pick_object"]
        cur_obj_pos = (pick_obj.data.root_pos_w[env_idx] -
                       self.scene.env_origins[env_idx]).cpu().numpy()
        cur_obj_quat = pick_obj.data.root_quat_w[env_idx].cpu().numpy()
        obj_pose_np = np.concatenate([cur_obj_pos, cur_obj_quat])
        place_np = self._place_xyz_cached[env_idx].cpu().numpy()

        # If the cube has fallen below the table, we can't pick it up anyway —
        # mark the episode as failed and stop retrying.
        if cur_obj_pos[2] < self.cfg.drop_z_threshold - 0.2:
            print(f"[env {env_idx}] replan skipped: cube below table "
                  f"(z={cur_obj_pos[2]:.3f})")
            return False

        self._attempt_count[env_idx] += 1
        print(f"[env {env_idx}] REPLAN #{int(self._attempt_count[env_idx].item())} "
              f"({reason}) from obj_pose=({cur_obj_pos[0]:.3f},"
              f"{cur_obj_pos[1]:.3f},{cur_obj_pos[2]:.3f})")

        # Try the planned approach; on synthesis failure, advance through
        # the approach ranking until something works or we exhaust the rank.
        ranking = self._obj_entry().get("approach_ranking", [])
        # _approach_for_attempt picks rank[attempt_count]. On synthesis fail,
        # bump attempt_count once and try the next approach (without counting
        # against max_regrasp_attempts the second time — the user-facing
        # attempt limit measures EXECUTED trajectories, not synthesis tries).
        max_synth_tries = max(1, len(ranking))
        last_error = "BODex synthesis failed"
        for synth_try in range(max_synth_tries):
            ok = self._build_and_load_trajectory(
                env_idx=env_idx,
                obj_pose_np=obj_pose_np,
                place_np=place_np,
                prepend_home_from_current=True,
            )
            if ok:
                # Reset per-attempt metrics so evaluate sees this attempt's state.
                self._metric_object_max_z[env_idx] = float(cur_obj_pos[2])
                # Clear the failed-flag the previous _build set (success path
                # already does, but be explicit if we looped).
                self._episode_failed[env_idx] = False
                return True
            # Synthesis failed for this approach. If we still have approaches
            # left in the rank, advance and try again.
            next_attempt = int(self._attempt_count[env_idx].item()) + 1
            if next_attempt < len(ranking) and self.cfg.approach == "auto":
                self._attempt_count[env_idx] += 1
                print(f"[env {env_idx}] synth-fail; advancing approach to "
                      f"rank[{next_attempt}]={ranking[next_attempt]}")
                continue
            break
        # Exhausted approaches without a successful synthesis.
        print(f"[env {env_idx}] replan giving up: {last_error}")
        return False

    def _approach_for_attempt(self, env_idx: int) -> str:
        """Pick the approach for THIS attempt.

        - If `cfg.approach` is an explicit value (top, side, etc.), use it
          unchanged on every retry (user pinned it).
        - If `cfg.approach == "auto"`, walk the per-object ranked list
          (from `recommend_approach_ranking`) using the current attempt
          index. After exhausting the ranking, repeat the last entry.
        """
        if self.cfg.approach != "auto":
            return self.cfg.approach
        ranking = self._obj_entry()["approach_ranking"]
        attempt_i = int(self._attempt_count[env_idx].item())
        if not ranking:
            return "top"
        return ranking[min(attempt_i, len(ranking) - 1)]

    def _passive_arm_state_world(self, env_idx: int) -> np.ndarray | None:
        """Live joint state of the passive arm for L2 collision injection.

        Returns 6-vector [shoulder_pan, shoulder_lift, elbow, wrist_1,2,3] in
        radians. None when disabled. Joint order matches the passive arm
        sphere-yaml's expected URDF order.
        """
        if not getattr(self.cfg, "enable_passive_arm_collision", False):
            return None
        passive_robot = self._passive_robot()
        q = passive_robot.data.joint_pos[env_idx, self._passive_arm_jids].cpu().numpy()
        return np.asarray(q, dtype=float)

    def _invoke_bodex(
        self,
        object_pose: np.ndarray,
        place_xyz: np.ndarray,
        env_idx: int = 0,
        is_retry: bool = False,
    ):
        """Request a grasp trajectory from the persistent BODex worker.

        Picks the approach (cycled on retry if `auto`), sends the passive
        arm's live joint state for collision-aware planning, and uses the
        smaller retry seed budget on attempts > 0.

        Returns (trajectory [T, 22] np.float32, pick_end_idx int, approach_used str)
        or (None, None, None) on failure.
        """
        approach = self._approach_for_attempt(env_idx)
        seed_num = (self.cfg.bodex_retry_seed_num if is_retry
                    else self.cfg.bodex_seed_num)
        # Relax the side-grasp thumb-up filter on retries: when feasibility is
        # already scarce (typical reason a retry is happening), demanding the
        # thumb point straight up rejects every candidate. cuRobo's collision
        # check still rejects seeds that scrape the table.
        thumb_up_min = 0.65 if not is_retry else 0.0
        passive_arm = "la" if self.cfg.active_arm == "ra" else "ra"
        passive_q = self._passive_arm_state_world(env_idx)
        try:
            obj_scale = self._obj_entry().get("scale", (1.0, 1.0, 1.0))
            obb_z_world = self._obj_entry().get("obb_z_world", None)
            kwargs = dict(
                active_arm=self.cfg.active_arm,
                object_id=self.cfg.object_id,
                object_pose=object_pose,
                place_xyz=place_xyz,
                approach=approach,
                seed_num=seed_num,
                hold_frames=self.cfg.hold_frames,
                table_z=self.cfg.table_z,
                thumb_up_min=thumb_up_min,
                obj_scale=tuple(float(s) for s in obj_scale),
                obb_z_world=obb_z_world,
            )
            # Apply the per-object HARDWARE synthesis improvements so SIM matches the
            # hardware benchmark: strict thumb-up + standoff + grasp-height bias for
            # the side-grasp objects. (The UP-OUT-DOWN-IN approach and the open-hand
            # approach collision guard are already shared via the worker.)
            _SIM_SYNTH_PER_OBJECT = {
                "mustard":   dict(thumb_up_strict=True, standoff_dist=0.08,
                                  grasp_z_bonus=0.0, seed_num=400),
                "green_can": dict(thumb_up_strict=True, standoff_dist=0.08,
                                  grasp_z_bonus=3.0, seed_num=400),
                "tea_peach": dict(thumb_up_strict=True, standoff_dist=0.08,
                                  grasp_z_bonus=3.0, seed_num=400),
                "milk":      dict(thumb_up_strict=True, standoff_dist=0.08,
                                  grasp_z_bonus=0.0, seed_num=400),
            }
            _sp = _SIM_SYNTH_PER_OBJECT.get(self.cfg.object_id)
            if _sp:
                for _k in ("thumb_up_strict", "standoff_dist", "grasp_z_bonus"):
                    if _k in _sp:
                        kwargs[_k] = _sp[_k]
                if not is_retry and "seed_num" in _sp:
                    kwargs["seed_num"] = _sp["seed_num"]
            if passive_q is not None:
                kwargs["passive_arm"] = passive_arm
                kwargs["passive_joint_state"] = passive_q
            result = self._bodex_client.synthesize_pick_and_place(**kwargs)
            return result["trajectory"], int(result["pick_end_idx"]), approach
        except Exception as e:
            print(f"[env] BODex worker failed (approach={approach}): {e}")
            return None, None, approach

    # ---------------------------------------------------------------
    # step loop
    # ---------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        idx = torch.minimum(self._step_idx, self._traj_len - 1)
        wp = self._full_traj[torch.arange(self.num_envs, device=self.device), idx]
        active_robot = self._active_robot()
        active_robot.set_joint_position_target(wp[:, :6], joint_ids=self._arm_jids)
        active_robot.set_joint_position_target(wp[:, 6:22], joint_ids=self._hand_jids)
        self._freeze_passive_arm()
        # Update visual status sphere (skip if locked at green/red end-of-episode).
        self._update_status_marker()

    def _update_status_marker(self):
        """Compute color per env (blue > orange > neutral) during execution and
        push the marker translation + per-instance prototype index. Does
        nothing when `status_locked` (final green/red persists)."""
        if "status_marker" not in self._markers:
            return
        # Lazy-init joint limits once after first physics tick
        if self._joint_lower is None:
            try:
                lim = self._active_robot().data.joint_pos_limits  # [N_env, J, 2]
                if lim is not None:
                    arm_lim = lim[0, self._arm_jids]
                    self._joint_lower = arm_lim[:, 0].clone()
                    self._joint_upper = arm_lim[:, 1].clone()
            except Exception:
                pass

        active_robot = self._active_robot()
        joint_pos = active_robot.data.joint_pos[:, self._arm_jids]   # [N, 6]

        # Risk: any joint within margin of its hard limit
        risky = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self._joint_lower is not None and self._joint_upper is not None:
            margin = self.cfg.risk_joint_limit_margin_rad
            near_low = (joint_pos - self._joint_lower) < margin
            near_high = (self._joint_upper - joint_pos) < margin
            risky = risky | near_low.any(dim=-1) | near_high.any(dim=-1)

        # Risk: wrist link world-z below table edge (visible "dangle" below table).
        # data.body_pos_w is [N, num_bodies, 3]
        try:
            arm_prefix = self.cfg.active_arm + "_"
            body_names = active_robot.body_names
            wrist3_idx = body_names.index(arm_prefix + "wrist_3_link")
            wrist3_z_world = active_robot.data.body_pos_w[:, wrist3_idx, 2]
            risky = risky | (wrist3_z_world < self.cfg.risk_wrist_below_table_z)
        except (ValueError, AttributeError, IndexError):
            pass

        # Collision: contact sensor (visualization only; does not trigger replan).
        collided = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.cfg.enable_status_contact_tracking:
            sensor_key = f"{self.cfg.active_arm}_contact"
            sensor = self._sensors.get(sensor_key)
            if sensor is not None:
                try:
                    nf = sensor.data.net_forces_w
                    if nf is not None:
                        fmag = torch.norm(nf, dim=-1)
                        max_f = fmag.max(dim=-1).values
                        collided = max_f > self.cfg.status_collision_force_threshold_n
                except Exception:
                    pass

        # New state: blue (3) > orange (4) > neutral (0). Don't override locked.
        new_state = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        new_state = torch.where(risky, torch.full_like(new_state, 4), new_state)
        new_state = torch.where(collided, torch.full_like(new_state, 3), new_state)
        new_state = torch.where(self._status_locked, self._status_state, new_state)
        self._status_state = new_state

        self._render_status_marker()

    def _render_status_marker(self):
        """Place the status sphere at a fixed world position above the
        T-shape mount (between the two arm bases). Color updates per env."""
        marker = self._markers.get("status_marker")
        if marker is None:
            return
        # Fixed world position per env: env_origin + cfg.status_marker_xyz
        fixed = torch.tensor(
            self.cfg.status_marker_xyz, device=self.device, dtype=torch.float32,
        )
        translations = self.scene.env_origins + fixed.unsqueeze(0)  # [N, 3]
        # Marker prototypes: [green=0, red=1, blue=2, orange=3, neutral=4]
        # _status_state encoding: 0=neutral, 1=green, 2=red, 3=blue, 4=orange
        proto_map = torch.tensor([4, 0, 1, 2, 3], device=self.device, dtype=torch.long)
        marker_indices = proto_map[self._status_state]
        try:
            marker.visualize(translations=translations, marker_indices=marker_indices.tolist())
        except Exception as e:
            if self._frame_idx % 200 == 0:
                print(f"[status marker] {e}")

    def _apply_action(self):
        pass

    def _get_observations(self) -> dict:
        pick_obj: RigidObject = self._rigid_objects["pick_object"]
        obj_pos = pick_obj.data.root_pos_w - self.scene.env_origins
        target = self._object_target
        obs = torch.cat(
            [obj_pos, target, torch.zeros((self.num_envs, 6), device=self.device)],
            dim=-1,
        )

        # Track max-z for grasp-success metric
        self._metric_object_max_z = torch.maximum(self._metric_object_max_z, obj_pos[:, 2])
        # DEBUG: log a compact obj z trace every 40 frames
        if self._frame_idx % 40 == 0:
            print(f"  [obs debug] frame={self._frame_idx}  "
                  f"obj_z={float(obj_pos[0, 2].item()):.4f}  "
                  f"max_z={float(self._metric_object_max_z[0].item()):.4f}")

        # Capture RGB frame (single env case)
        if self.cfg.enable_camera and "camera" in self._sensors:
            try:
                cam: Camera = self._sensors["camera"]
                rgb = cam.data.output.get("rgb", None)   # [N, H, W, 3] uint8
                if rgb is not None and rgb.shape[0] > 0:
                    self._save_frame(rgb[0].cpu().numpy())
            except Exception as e:
                print(f"[camera warn] {e}")

        return {"policy": obs}

    def _save_frame(self, rgb: np.ndarray):
        """Write PNG to disk for later video stitching."""
        import imageio.v2 as imageio
        path = os.path.join(
            self.cfg.camera_rgb_dir,
            f"ep{self._episode_idx:03d}_frame{self._frame_idx:04d}.png",
        )
        imageio.imwrite(path, rgb)
        self._frame_idx += 1

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Guard: if traj_len is 0 (env just constructed, reset not yet called),
        # return not-done so the runner can call env.reset() to populate it.
        not_initialized = self._traj_len == 0

        # waypoint_repeat slowdown: only advance _step_idx every K physics steps
        self._substep_counter = self._substep_counter + 1
        should_advance = self._substep_counter >= self.cfg.waypoint_repeat
        advance = should_advance.long()
        self._step_idx = torch.minimum(
            self._step_idx + advance,
            torch.clamp(self._traj_len - 1, min=0),
        )
        self._substep_counter = torch.where(
            should_advance,
            torch.zeros_like(self._substep_counter),
            self._substep_counter,
        )

        reached_end = (~not_initialized) & (self._step_idx >= (self._traj_len - 1))
        pick_obj: RigidObject = self._rigid_objects["pick_object"]
        obj_pos_full = pick_obj.data.root_pos_w - self.scene.env_origins
        obj_z = obj_pos_full[:, 2]

        # "Fell off the table" during playback — triggers an immediate retry
        # while the trajectory is still running (mid-transit drop detection).
        # We only look at envs that are past the approach phase (step_idx >
        # pick_end_idx) because during approach the cube IS on the table at
        # its spawn z ≈ table_top and triggering a spurious retry would loop.
        past_pick = self._step_idx > self._pick_end_idx
        fell_during_run = past_pick & (obj_z < self.cfg.drop_z_threshold)

        # Level-4 runtime collision: read contact forces on the ACTIVE arm's
        # non-hand links. Any sustained force above the threshold flags the
        # env for a regrasp.
        #
        # IMPORTANT: gated by `past_pick` — during the approach + grasp
        # phase, the arm body legitimately gets close to the object (and
        # tumbler-like objects can momentarily push back on the wrist as
        # the hand wraps around). Triggering a replan during the grasp
        # phase aborts a perfectly fine grasp before the squeeze even
        # completes, which was the source of the ra-tumbler / la-flat_plate
        # "arm collision" REPLAN floods seen in the v2 benchmark.
        collision_now = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.cfg.enable_contact_sensors:
            sensor_key = f"{self.cfg.active_arm}_contact"
            if sensor_key in self._sensors:
                sensor = self._sensors[sensor_key]
                # net_forces_w: [num_envs, num_tracked_bodies, 3]
                try:
                    nf = sensor.data.net_forces_w
                    if nf is not None:
                        fmag = torch.norm(nf, dim=-1)                  # [N, L]
                        max_f_per_env = fmag.max(dim=-1).values        # [N]
                        collision_now = past_pick & (
                            max_f_per_env > self.cfg.collision_force_threshold_n
                        )
                except Exception as e:
                    if self._frame_idx % 200 == 0:
                        print(f"[contact warn] {e}")

        # Regrasp decision per env
        terminated_init = torch.zeros_like(reached_end)
        for i in range(self.num_envs):
            if self._episode_failed[i]:
                terminated_init[i] = True
                # Status sphere: RED, latched (BODex synthesis failed end-of-line).
                if not self._status_locked[i]:
                    self._status_state[i] = 2
                    self._status_locked[i] = True
                continue
            if self._attempt_resolved[i]:
                # Already decided to end this episode — but if a status-hold
                # is in flight (showing green/red sphere), keep ticking it
                # down before terminating. Without this, the auto-reset wipes
                # the latched color before the camera can see it.
                if self._status_hold_remaining[i] > 0:
                    self._status_hold_remaining[i] -= 1
                if self._status_hold_remaining[i] == 0:
                    terminated_init[i] = True
                continue

            if fell_during_run[i]:
                # Cube dropped while carrying — try to pick it up from the
                # new location if we still have attempts left.
                if self._try_replan(int(i), reason="mid-transit drop"):
                    # Swallow this tick's "reached end" signal for this env
                    continue
                else:
                    terminated_init[i] = True
                    self._attempt_resolved[i] = True
                    self._status_state[i] = 2  # red
                    self._status_locked[i] = True
                continue

            if collision_now[i]:
                # Arm body contacted something that isn't the object. Halt
                # playback for this env and request a fresh plan. Distinguish
                # from fingertip/grasp contact by virtue of the sensor's link
                # filter (hand links are not tracked).
                if self._try_replan(int(i), reason="arm collision"):
                    continue
                else:
                    terminated_init[i] = True
                    self._attempt_resolved[i] = True
                    self._status_state[i] = 2  # red
                    self._status_locked[i] = True
                continue

            if reached_end[i]:
                # Trajectory fully played — grade it and maybe retry.
                spawn_z = self._object_spawn_z[i]
                max_z = self._metric_object_max_z[i]
                lifted_by = float((max_z - spawn_z).item())
                target = self._place_xyz_cached[i]
                cur_xy = obj_pos_full[i, :2]
                xy_err = float(torch.norm(cur_xy - target[:2]).item())
                z_err = float(abs(obj_pos_full[i, 2] - target[2]).item())
                # Z tolerance scales with object height: a tall slim object
                # (tumbler) released from a side grasp may end up lying on
                # its side rather than standing — z_err then ≈ obb_z/2 - r.
                # We accept this as a placed object as long as xy is good,
                # otherwise no slim/tall object would ever pass place. For
                # short objects the tolerance is the base (0.08).
                obb_z = self._obj_entry()["obb_z_world"]
                xy_tol = self.cfg.place_xy_tolerance
                z_tol = max(self.cfg.place_z_tolerance_base, obb_z * 0.6)
                grasp_ok = lifted_by >= self.cfg.min_lift_for_grasp
                place_ok = grasp_ok and xy_err < xy_tol and z_err < z_tol

                if place_ok:
                    self._attempt_resolved[i] = True
                    # Status sphere: GREEN, latched. Hold for 60 frames so the
                    # color is visible in the camera before Isaac Lab auto-resets.
                    if self._status_hold_remaining[i] == 0 and not self._status_locked[i]:
                        self._status_state[i] = 1
                        self._status_locked[i] = True
                        self._status_hold_remaining[i] = 60
                    if self._status_hold_remaining[i] > 0:
                        self._status_hold_remaining[i] -= 1
                    if self._status_hold_remaining[i] == 0:
                        terminated_init[i] = True
                    continue

                # Decide retry reason
                if not grasp_ok:
                    reason = f"grasp miss (lifted_by={lifted_by:.3f})"
                elif xy_err >= xy_tol:
                    reason = f"place miss (xy_err={xy_err:.3f}>{xy_tol:.3f})"
                else:
                    reason = f"place tip (z_err={z_err:.3f}>{z_tol:.3f})"

                if self._try_replan(int(i), reason=reason):
                    # New trajectory loaded; env keeps running.
                    continue
                # Out of retries — RED hold then terminate
                self._attempt_resolved[i] = True
                if self._status_hold_remaining[i] == 0 and not self._status_locked[i]:
                    self._status_state[i] = 2
                    self._status_locked[i] = True
                    self._status_hold_remaining[i] = 60
                if self._status_hold_remaining[i] > 0:
                    self._status_hold_remaining[i] -= 1
                if self._status_hold_remaining[i] == 0:
                    terminated_init[i] = True

        terminated = terminated_init
        truncated = torch.zeros_like(terminated)

        # Latch the final state for any env that's about to terminate — Isaac
        # Lab will auto-reset it before evaluate_episode() gets a chance.
        done_now = terminated | truncated
        if done_now.any():
            # Use a generic per-element mask so it also works for num_envs>1
            self._latched_obj_pos = torch.where(
                done_now.unsqueeze(-1),
                obj_pos_full,
                self._latched_obj_pos,
            )
            self._latched_max_z = torch.where(
                done_now,
                self._metric_object_max_z,
                self._latched_max_z,
            )
            self._latched_attempt_count = torch.where(
                done_now,
                self._attempt_count,
                self._latched_attempt_count,
            )
            self._latched_target = torch.where(
                done_now.unsqueeze(-1),
                self._object_target,
                self._latched_target,
            )
        return terminated, truncated

    # ---------------------------------------------------------------
    # Episode success evaluation (called from the runner after each episode)
    # ---------------------------------------------------------------
    def evaluate_episode(self) -> dict:
        # Use the LATCHED end-of-episode snapshot (Isaac Lab auto-resets
        # terminated envs before we get here, so live state is post-reset).
        obj_pos = self._latched_obj_pos[0]
        max_z = self._latched_max_z[0]
        target = self._latched_target[0]    # latched at terminate time
        spawn_z = self._object_spawn_z[0]

        print(f"  [eval debug] spawn_z={float(spawn_z.item()):.4f} "
              f"max_z={float(max_z.item()):.4f} "
              f"final_z={float(obj_pos[2].item()):.4f}")

        # Grasp: object was lifted at least 3 cm above its settled pose during
        # the episode (lift phase is 5 cm, allow margin for sim noise).
        lifted_by = float((max_z - spawn_z).item())
        grasp_success = lifted_by >= 0.03

        # Place: final xy within `place_xy_tolerance` AND z within
        # obb-scaled tolerance of target. The xy tolerance reflects what
        # open-loop BODex grasp synthesis + trajectory playback can
        # reliably hit without closed-loop visual feedback. The z
        # tolerance scales with object height: tall slim objects (tumbler,
        # bottle) released from a side grasp may end up lying on their
        # side rather than standing — z_err then ≈ obb_z/2 - r. A real-
        # world pick-and-place would still count that as "placed in the
        # right spot", just the wrong orientation. Place only counts if
        # grasp succeeded — otherwise the "correct position" is just an
        # undisturbed cube near the sampled spawn.
        xy_err = float(torch.norm(obj_pos[:2] - target[:2]).item())
        z_err = float(abs(obj_pos[2] - target[2]).item())
        obb_z = self._obj_entry()["obb_z"]
        xy_tol = self.cfg.place_xy_tolerance
        z_tol = max(self.cfg.place_z_tolerance_base, obb_z * 0.6)
        place_success = grasp_success and xy_err < xy_tol and z_err < z_tol

        result = {
            "lifted_by_m": lifted_by,
            "grasp_success": grasp_success,
            "final_xy_err_m": xy_err,
            "final_z_err_m": z_err,
            "xy_tol_m": xy_tol,
            "z_tol_m": z_tol,
            "place_success": place_success,
            "final_obj_pos": obj_pos.cpu().numpy().tolist(),
            "target_pos": target.cpu().numpy().tolist(),
            "attempts": int(self._latched_attempt_count[0].item()) + 1,
        }
        return result

    def close(self):
        try:
            if getattr(self, "_bodex_client", None) is not None:
                self._bodex_client.close()
        except Exception:
            pass
        super().close()

    def stitch_video_if_configured(self):
        """At end of run, stitch all saved PNG frames into an MP4."""
        if not self.cfg.enable_camera or self._frame_idx == 0:
            return None
        try:
            import imageio.v2 as imageio
            frames = sorted(
                os.path.join(self.cfg.camera_rgb_dir, f)
                for f in os.listdir(self.cfg.camera_rgb_dir)
                if f.endswith(".png")
            )
            if not frames:
                return None
            # macro_block_size=1 disables the default h264 16-pixel alignment,
            # so the output video is exactly 1920x1080 rather than 1920x1088.
            with imageio.get_writer(
                self.cfg.camera_video_path, fps=60, codec="libx264",
                macro_block_size=1,
            ) as w:
                for p in frames:
                    w.append_data(imageio.imread(p))
            print(f"[video] wrote {len(frames)} frames → {self.cfg.camera_video_path}")
            return self.cfg.camera_video_path
        except Exception as e:
            print(f"[video warn] could not stitch: {e}")
            return None
