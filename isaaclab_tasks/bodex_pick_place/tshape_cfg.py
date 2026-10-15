"""T-shape dual-arm scene configs, owned locally by this task.

Copied from /mnt/ssd2/ws/Dualarm_ws/isaaclab_dualarm/utils/common_cfg.py
so we can tune actuator gains and asset paths without touching the original.

Key changes vs the source:
  - Asset paths point at BODex/isaaclab_tasks/assets/
  - Allegro hand actuator tuned for reliable gripping:
        effort_limit_sim: 0.8 → 15.0 Nm
        stiffness:       1200 → 600 (lower stiffness + higher effort widens grip basin)
        damping:          120 → 80
        velocity_limit_sim: 30 → 60 (fingers close faster)
  - UR arm effort_limit_sim 120 → 300 Nm for better tracking under load
"""
from __future__ import annotations
import math
import os

from scipy.spatial.transform import Rotation

import isaaclab.sim as sim_utils
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.envs import DirectRLEnvCfg, DirectRLEnv
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass


# Hand-actuator gains — *much* more aggressive than the stock isaaclab_dualarm
# to compensate for the gap between BODex's planned grasp and Isaac Sim's
# rigid-body contact resolver.
HAND_ACTUATOR = dict(
    effort_limit_sim=15.0,         # was 0.8
    velocity_limit_sim=60.0,       # was 30
    stiffness=600.0,               # was 1200 — softer P
    damping=80.0,                  # was 120
    armature=0.005,
    viscous_friction=0.01,
)
ARM_ACTUATOR = dict(
    effort_limit_sim=300.0,        # was 120
    velocity_limit_sim=4.0,        # was 2 — 2x faster arm motion
    stiffness=1200.0,              # slightly stiffer for faster tracking
    damping=200.0,                 # dropped from 300 to match stiffness/velocity
    armature=0.02,
    viscous_friction=0.02,
)


def _rot_quat(angles_zy_deg: tuple, local_z_deg: float) -> tuple:
    """Build the t-shape arm-mount quaternion (w,x,y,z) used in isaaclab_dualarm."""
    r = Rotation.from_euler("zy", angles_zy_deg, degrees=True)
    r = r * Rotation.from_euler("Z", local_z_deg, degrees=True)
    q = r.as_quat()   # scipy: (x,y,z,w)
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def make_tshape_right_robot_cfg(assets_dir: str) -> ArticulationCfg:
    usd = os.path.join(assets_dir, "ra_ur3_allegro_isaac_", "ra_ur3_allegro_isaac_.usd")
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/RightRobot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            # Enables per-link ContactSensor force readings (Level-4 runtime
            # collision detection in env.py).
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.177, 0.0, 1.49),
            rot=_rot_quat((-90.0, 135.0), 135.0),
            joint_pos={
                "ra_shoulder_pan_joint":  math.radians(27.33),
                "ra_shoulder_lift_joint": math.radians(-43.98),
                "ra_elbow_joint":         math.radians(98.18),
                "ra_wrist_1_joint":       math.radians(-232.88),
                "ra_wrist_2_joint":       math.radians(-71.98),
                "ra_wrist_3_joint":       math.radians(223.84),
                "ra_joint_12_0":          1.396,
            },
        ),
        actuators={
            "ur3_arm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "ra_shoulder_pan_joint", "ra_shoulder_lift_joint", "ra_elbow_joint",
                    "ra_wrist_1_joint", "ra_wrist_2_joint", "ra_wrist_3_joint",
                ],
                **ARM_ACTUATOR,
            ),
            "allegro_hand": ImplicitActuatorCfg(
                joint_names_expr=[
                    "ra_joint_0_0", "ra_joint_12_0", "ra_joint_4_0", "ra_joint_8_0",
                    "ra_joint_1_0", "ra_joint_13_0", "ra_joint_5_0", "ra_joint_9_0",
                    "ra_joint_2_0", "ra_joint_14_0", "ra_joint_6_0", "ra_joint_10_0",
                    "ra_joint_3_0", "ra_joint_15_0", "ra_joint_7_0", "ra_joint_11_0",
                ],
                **HAND_ACTUATOR,
            ),
        },
    )


def make_tshape_left_robot_cfg(assets_dir: str) -> ArticulationCfg:
    usd = os.path.join(assets_dir, "la_ur3e_allegro_isaac_", "la_ur3e_allegro_isaac_.usd")
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/LeftRobot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            # Enables per-link ContactSensor force readings (Level-4 runtime
            # collision detection in env.py).
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.177, 0.0, 1.49),
            rot=_rot_quat((90.0, -135.0), -135.0),
            joint_pos={
                "la_shoulder_pan_joint":  math.radians(-18.75),
                "la_shoulder_lift_joint": math.radians(-138.64),
                "la_elbow_joint":         math.radians(-99.59),
                "la_wrist_1_joint":       math.radians(57.97),
                "la_wrist_2_joint":       math.radians(64.88),
                "la_wrist_3_joint":       math.radians(137.67),
                "la_joint_12_0":          1.396,
            },
        ),
        actuators={
            "ur3_arm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "la_shoulder_pan_joint", "la_shoulder_lift_joint", "la_elbow_joint",
                    "la_wrist_1_joint", "la_wrist_2_joint", "la_wrist_3_joint",
                ],
                **ARM_ACTUATOR,
            ),
            "allegro_hand": ImplicitActuatorCfg(
                joint_names_expr=[
                    "la_joint_0_0", "la_joint_12_0", "la_joint_4_0", "la_joint_8_0",
                    "la_joint_1_0", "la_joint_13_0", "la_joint_5_0", "la_joint_9_0",
                    "la_joint_2_0", "la_joint_14_0", "la_joint_6_0", "la_joint_10_0",
                    "la_joint_3_0", "la_joint_15_0", "la_joint_7_0", "la_joint_11_0",
                ],
                **HAND_ACTUATOR,
            ),
        },
    )


def make_tshape_cfg(assets_dir: str) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/tshape",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(assets_dir, "tshape_.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -0.147, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def make_table_cfg(assets_dir: str) -> RigidObjectCfg:
    """Motion-desk table the object rests on.

    When placed at pos=(0, 0.4, 0.03), its top surface is at world z≈0.96.
    """
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(assets_dir, "motion_desk.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.03),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


# Approximate world-z of the motion_desk top surface after the pose above.
TABLE_TOP_Z = 0.96
