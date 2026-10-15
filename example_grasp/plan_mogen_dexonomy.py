# Standard Library
import time
import os
import datetime

# Third Party
import torch
import numpy as np
import argparse

# CuRobo
from curobo.geom.sdf.world import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.robot import JointState, RobotConfig
from curobo.types.math import Pose
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.wrap.reacher.ik_solver import IKSolverConfig, IKSolver
from curobo.util.world_cfg_generator import get_world_config_dataloader
from curobo.util.save_helper import SaveHelper
from curobo.util.logger import setup_logger, log_warn
from curobo.util_file import (
    get_manip_configs_path,
    get_robot_configs_path,
    join_path,
    load_yaml,
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import numpy as np
import random

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)


class PlanningFailureError(Exception):
    pass


def get_armhand_qpos(arm_ik: IKSolver, hand_pose_qpos, retract_qpos):
    ik_result = arm_ik.solve_single(
        Pose.from_list(hand_pose_qpos.view(-1)[:7]),
        seed_config=retract_qpos.view(1, 1, -1)[..., : arm_ik.kinematics.dof],
        retract_config=retract_qpos.view(1, -1)[:, : arm_ik.kinematics.dof],
    )
    if not ik_result.success:
        raise PlanningFailureError("IK fails")
    armhand_qpos = torch.cat(
        [ik_result.solution.squeeze(1), hand_pose_qpos.view(1, -1)[:, 7:]], dim=-1
    )
    return armhand_qpos  # [1, n]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--manip_cfg_file",
        type=str,
        default="fc_leap.yml",
        help="config file path",
    )

    parser.add_argument(
        "-f",
        "--save_folder",
        type=str,
        default=None,
        help="If None, use join_path(manip_cfg_file[:-4], $TIME) as save_folder",
    )

    parser.add_argument(
        "-m",
        "--save_mode",
        choices=["usd", "npy", "usd+npy", "none"],
        default="npy",
        help="Method to save results",
    )

    parser.add_argument(
        "-k",
        "--skip",
        action="store_false",
        help="If True, skip existing files. (default: True)",
    )

    args = parser.parse_args()

    setup_logger("warn")
    tensor_args = TensorDeviceType()
    manip_config_data = load_yaml(join_path(get_manip_configs_path(), args.manip_cfg_file))

    if args.save_folder is not None:
        save_folder = args.save_folder
    elif manip_config_data["exp_name"] is not None:
        save_folder = os.path.join(
            args.manip_cfg_file[:-4], manip_config_data["exp_name"], "traj_data"
        )
    else:
        save_folder = os.path.join(
            args.manip_cfg_file[:-4],
            datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
            "traj_data",
        )

    assert manip_config_data["world"]["type"] == "dexonomy"
    log_warn(f"Load dexonomy grasp pose from {manip_config_data['world']['template_path']}!")

    world_generator = get_world_config_dataloader(manip_config_data["world"], 1)

    save_mogen = SaveHelper(
        robot_file=manip_config_data["robot_file_with_arm"],
        save_folder=save_folder,
        task_name="grasp",
        mode=args.save_mode,
    )

    # Initialize hand forward kinematic model
    robot_config_data = load_yaml(
        join_path(get_robot_configs_path(), manip_config_data["robot_file"])
    )["robot_cfg"]
    hand_robocfg = RobotConfig.from_dict(robot_config_data)
    hand_model = CudaRobotModel(hand_robocfg.kinematics)

    init_ah_qpos = tensor_args.to_device(manip_config_data["mogen_init"]).view(1, -1)
    init_state = JointState.from_position(init_ah_qpos)

    tst = time.time()
    mg = None
    arm_iksolver = None
    for world_info_dict in world_generator:
        sst = time.time()
        if args.skip and save_mogen.exist_piece(world_info_dict["save_prefix"]):
            log_warn(f"skip {world_info_dict['save_prefix']}")
            continue

        for k, v in world_info_dict.items():
            if "qpos" in k or k == "scene_path":
                world_info_dict[k] = v[0]

        world_model = [WorldConfig.from_dict(world_info_dict["world_cfg"][0])]

        if arm_iksolver is None:
            robot_config_data = load_yaml(
                join_path(get_robot_configs_path(), manip_config_data["robot_file_with_arm"])
            )["robot_cfg"]
            ik_link_names = hand_model.transfered_link_name
            robot_config_data["kinematics"]["link_names"] = ik_link_names
            robot_config_data["kinematics"]["ee_link"] = ik_link_names[0]
            ik_robot_cfg = RobotConfig.from_dict(robot_config_data, tensor_args)
            ik_config = IKSolverConfig.load_from_robot_config(
                ik_robot_cfg,
                world_model[0],
                tensor_args=tensor_args,
                high_precision=True,
            )
            arm_iksolver = IKSolver(ik_config)
        else:
            arm_iksolver.update_world(world_model)

        if mg is None:
            motion_gen_cfg = MotionGenConfig.load_from_robot_config(
                manip_config_data["robot_file_with_arm"],
                world_model=world_model[0],
                collision_activation_distance=0.025,
            )
            mg = MotionGen(motion_gen_cfg)
        else:
            mg.update_world(world_model[0])

        pregrasp_qpos = tensor_args.to_device(world_info_dict["pregrasp_qpos_urdf"])
        try:
            # Inverse kinematic for pregrasp
            pregrasp1_ah_qpos = get_armhand_qpos(
                arm_iksolver, pregrasp_qpos[1], init_ah_qpos.clone()
            )
            pregrasp0_ah_qpos = get_armhand_qpos(arm_iksolver, pregrasp_qpos[0], pregrasp1_ah_qpos)
            pregrasp_vel = (pregrasp0_ah_qpos - pregrasp1_ah_qpos) / 0.02 / 10

            # Check smoothness
            if (pregrasp1_ah_qpos[:, :8] - pregrasp0_ah_qpos[:, :8]).abs().max() > 0.1:
                raise PlanningFailureError("Non-smooth IK fails")

            # Motion planning
            target_state = JointState.from_numpy(
                position=pregrasp0_ah_qpos,
                velocity=pregrasp_vel,
                joint_names=mg.kinematics.joint_names,
            )
            mogen_result = mg.plan_single_js(
                start_state=target_state,
                goal_state=init_state,
                plan_config=MotionGenPlanConfig(
                    enable_finetune_trajopt=False, num_trajopt_seeds=4, max_attempts=1
                ),
            )
            if sum(mogen_result.success) == 0:
                raise PlanningFailureError(f"MoGen fails")
        except PlanningFailureError as e:
            log_warn(e, world_info_dict["save_prefix"][0])
            continue

        world_info_dict["robot_pose"] = torch.flip(mogen_result.optimized_plan.position[2:-2], [0])
        world_info_dict["world_model"] = world_model
        world_info_dict.pop("world_cfg")
        if "usd" not in args.save_mode:
            if world_info_dict["hand_name"][0] == "real_shadow":
                world_info_dict["robot_pose"] = torch.cat(
                    [
                        world_info_dict["robot_pose"][:, :8],
                        world_info_dict["robot_pose"][:, 13:],
                        world_info_dict["robot_pose"][:, 8:13],
                    ],
                    axis=-1,
                )
            else:
                raise NotImplementedError
        save_mogen.save_piece(world_info_dict)
        log_warn(f"Sinlge Time (mogen): {time.time()-sst}")
    log_warn(f"Total Time: {time.time()-tst}")
