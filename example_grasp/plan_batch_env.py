# Standard Library
import time
from typing import Dict, List
import datetime
import os

# Third Party
import torch
import numpy as np
import argparse

# CuRobo
from curobo.geom.sdf.world import WorldConfig
from curobo.wrap.reacher.grasp_solver import GraspSolver, GraspSolverConfig
from curobo.util.world_cfg_generator import get_world_config_dataloader
from curobo.util.logger import setup_logger, log_warn
from curobo.util.save_helper import SaveHelper
from curobo.util_file import (
    get_manip_configs_path,
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


def process_grasp_result(result, save_debug, save_data, save_id):
    traj = result.debug_info["solver"]["steps"][0]
    all_traj = torch.cat(traj, dim=1)  # [b*n, h, q]
    batch, horizon = all_traj.shape[:2]

    if save_data == "all":
        select_horizon_lst = list(range(0, horizon))
    elif "select_" in save_data:
        part_num = int(save_data.split("select_")[-1])
        select_horizon_lst = list(range(0, horizon, horizon // (part_num - 1)))
        select_horizon_lst[-1] = horizon - 1
    elif save_data == "init":
        select_horizon_lst = [0]
    elif save_data == "final" or save_data == "final_and_mid":
        select_horizon_lst = [-1]
    else:
        raise NotImplementedError

    if save_id is None:
        save_id_lst = list(range(0, batch))
    elif isinstance(save_id, List):
        save_id_lst = save_id
    else:
        raise NotImplementedError

    save_traj = all_traj[:, select_horizon_lst]
    save_traj = save_traj[save_id_lst, :]

    if save_debug:
        n_num = torch.stack(result.debug_info["solver"]["hp"][0]).shape[-2]
        o_num = torch.stack(result.debug_info["solver"]["op"][0]).shape[-2]
        hp_traj = torch.stack(result.debug_info["solver"]["hp"][0], dim=1).view(-1, n_num, 3)
        grad_traj = torch.stack(result.debug_info["solver"]["grad"][0], dim=1).view(-1, n_num, 3)
        op_traj = torch.stack(result.debug_info["solver"]["op"][0], dim=1).view(-1, o_num, 3)
        posi_traj = torch.stack(result.debug_info["solver"]["debug_posi"][0], dim=1).view(
            -1, o_num, 3
        )
        normal_traj = torch.stack(result.debug_info["solver"]["debug_normal"][0], dim=1).view(
            -1, o_num, 3
        )

        debug_info = {
            "hp": hp_traj,
            "grad": grad_traj * 100,
            "op": op_traj,
            "debug_posi": posi_traj,
            "debug_normal": normal_traj,
        }

        for k, v in debug_info.items():
            debug_info[k] = v.view((all_traj.shape[0], -1) + v.shape[1:])[:, select_horizon_lst]
            debug_info[k] = debug_info[k][save_id_lst, :]
            debug_info[k] = debug_info[k].view((-1,) + v.shape[1:])
    else:
        debug_info = None
        if save_data == "final_and_mid":
            mid_robot_pose = torch.cat(result.debug_info["solver"]["mid_result"][0], dim=1)
            mid_robot_pose = mid_robot_pose[save_id_lst, :]
            save_traj = torch.cat([mid_robot_pose, save_traj], dim=-2)

    return save_traj, debug_info


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
        "-d",
        "--save_data",
        # choices=['all', 'final', 'final_and_mid', 'init', 'select_{$INT}'],
        default="final_and_mid",
        help="Which results to save",
    )

    parser.add_argument(
        "-i",
        "--save_id",
        type=int,
        nargs="+",
        default=None,
        help="Which results to save",
    )

    parser.add_argument(
        "-debug",
        "--save_debug",
        action="store_true",
        help="Which to save contact normal for debug",
    )

    parser.add_argument(
        "-w",
        "--parallel_world",
        type=int,
        default=20,
        help="parallel world num.",
    )

    parser.add_argument(
        "-k",
        "--skip",
        action="store_false",
        help="If True, skip existing files. (default: True)",
    )

    setup_logger("warn")

    args = parser.parse_args()
    manip_config_data = load_yaml(join_path(get_manip_configs_path(), args.manip_cfg_file))

    world_generator = get_world_config_dataloader(manip_config_data["world"], args.parallel_world)

    if args.save_folder is not None:
        save_folder = os.path.join(args.save_folder, "grasp_data")
    elif manip_config_data["exp_name"] is not None:
        save_folder = os.path.join(
            args.manip_cfg_file[:-4], manip_config_data["exp_name"], "grasp_data"
        )
    else:
        save_folder = os.path.join(
            args.manip_cfg_file[:-4],
            datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
            "grasp_data",
        )

    save_helper = SaveHelper(
        robot_file=manip_config_data["robot_file"],
        save_folder=save_folder,
        task_name="grasp",
        mode=args.save_mode,
    )
    tst = time.time()
    grasp_solver = None
    for world_info_dict in world_generator:
        sst = time.time()
        if args.skip and save_helper.exist_piece(world_info_dict["save_prefix"]):
            log_warn(f"skip {world_info_dict['save_prefix']}")
            continue

        if grasp_solver is None:
            grasp_config = GraspSolverConfig.load_from_robot_config(
                world_model=world_info_dict["world_cfg"],
                manip_name_list=world_info_dict["manip_name"],
                manip_config_data=manip_config_data,
                obj_gravity_center=world_info_dict["obj_gravity_center"],
                obj_obb_length=world_info_dict["obj_obb_length"],
                use_cuda_graph=False,
                store_debug=args.save_debug,
            )
            grasp_solver = GraspSolver(grasp_config)
            world_info_dict["world_model"] = grasp_solver.world_coll_checker.world_model
        else:
            world_info_dict["world_model"] = world_model = [
                WorldConfig.from_dict(world_cfg) for world_cfg in world_info_dict["world_cfg"]
            ]
            grasp_solver.update_world(
                world_model,
                world_info_dict["obj_gravity_center"],
                world_info_dict["obj_obb_length"],
                world_info_dict["manip_name"],
            )

        result = grasp_solver.solve_batch_env(return_seeds=grasp_solver.num_seeds)

        if args.save_debug:
            robot_pose, debug_info = process_grasp_result(
                result, args.save_debug, args.save_data, args.save_id
            )
            world_info_dict["debug_info"] = debug_info
            world_info_dict["robot_pose"] = robot_pose.reshape(
                (len(world_info_dict["world_model"]), -1) + robot_pose.shape[1:]
            )
        else:
            squeeze_pose_qpos = torch.cat(
                [
                    result.solution[..., 1, :7],
                    result.solution[..., 1, 7:] * 2 - result.solution[..., 0, 7:],
                ],
                dim=-1,
            )
            all_hand_pose_qpos = torch.cat(
                [result.solution, squeeze_pose_qpos.unsqueeze(-2)], dim=-2
            )
            world_info_dict["robot_pose"] = all_hand_pose_qpos
            world_info_dict["contact_point"] = result.contact_point
            world_info_dict["contact_frame"] = result.contact_frame
            world_info_dict["contact_force"] = result.contact_force
            world_info_dict["grasp_error"] = result.grasp_error
            world_info_dict["dist_error"] = result.dist_error
        log_warn(f"Sinlge Time: {time.time()-sst}")
        save_helper.save_piece(world_info_dict)

    log_warn(f"Total Time: {time.time()-tst}")
