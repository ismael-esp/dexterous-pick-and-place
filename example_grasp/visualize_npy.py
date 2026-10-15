import os
import argparse
import numpy as np
import torch

from curobo.util_file import (
    load_json,
    write_json,
    load_yaml,
    join_path,
    get_manip_configs_path,
    get_output_path,
)
from curobo.util.logger import setup_logger, log_warn
from curobo.util.world_cfg_generator import get_world_config_dataloader
from curobo.util.save_helper import SaveHelper
from curobo.types.base import TensorDeviceType
from curobo.geom.sdf.world import WorldConfig

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
        "-p",
        "--path",
        type=str,
        required=True,
        help="exp folder name",
    )

    parser.add_argument(
        "-m",
        "--mode",
        choices=["grasp", "mogen"],
        required=True,
        help="visualize grasp or mogen",
    )

    parser.add_argument(
        "-k",
        "--skip",
        action="store_false",
        help="If True, skip existing files. (default: True)",
    )

    parser.add_argument(
        "-s",
        "--set_camera",
        action="store_true",
        help="If True, set a camera to help take pictures (default: False)",
    )

    setup_logger("warn")

    args = parser.parse_args()
    if "grasp_data" not in args.path:
        args.path = os.path.join(args.path, "grasp_data")

    manip_config_data = load_yaml(join_path(get_manip_configs_path(), args.manip_cfg_file))

    save_folder = os.path.join(args.manip_cfg_file[:-4], args.path)

    manip_config_data["world"]["type"] = "grasp"
    if args.mode == "grasp":
        manip_config_data["world"]["template_path"] = os.path.join(
            "output", save_folder, "**/*_grasp.npy"
        )
    elif args.mode == "mogen":
        manip_config_data["world"]["template_path"] = os.path.join(
            "output", save_folder, "**/*_mogen.npy"
        )
    else:
        raise NotImplementedError

    world_generator = get_world_config_dataloader(manip_config_data["world"], 1)

    if args.mode == "grasp":
        save_helper = SaveHelper(
            robot_file=manip_config_data["robot_file"],
            save_folder=save_folder,
            task_name="grasp",
            mode="usd",
            set_camera=args.set_camera,
        )
    else:
        save_helper = SaveHelper(
            robot_file=manip_config_data["robot_file_with_arm"],
            save_folder=save_folder,
            task_name="mogen",
            mode="usd",
            set_camera=args.set_camera,
        )

    tensor_args = TensorDeviceType()

    for world_info_dict in world_generator:
        if args.skip and save_helper.exist_piece(world_info_dict["save_prefix"]):
            log_warn(f"skip {world_info_dict['save_prefix']}")
            continue
        world_info_dict["robot_pose"] = tensor_args.to_device(world_info_dict["robot_pose"])
        world_info_dict["world_model"] = [WorldConfig.from_dict(world_info_dict["world_cfg"][0])]
        save_helper.save_piece(world_info_dict)
