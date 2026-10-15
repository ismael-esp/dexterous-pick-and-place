import os
import numpy as np
from glob import glob
import argparse
from curobo.util_file import (
    get_manip_configs_path,
    get_output_path,
    get_assets_path,
    join_path,
    load_yaml,
    write_yaml,
)
import subprocess
import multiprocessing
import datetime


def worker(gpu_id, task, manip_path, save_folder, output_path, save_mode, parallel_world, skip):
    with open(output_path, "a") as output_file:
        if task == "grasp":
            cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python example_grasp/plan_batch_env.py -c {manip_path} -f {save_folder} -m {save_mode} -w {parallel_world}"
        elif task == "mogen_dexonomy":
            cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python example_grasp/plan_mogen_dexonomy.py -c {manip_path} -f {save_folder} -m {save_mode}"
        else:
            cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python example_grasp/plan_mogen_batch.py -c {manip_path} -f {save_folder} -m {save_mode} -t {task}"
        if not skip:
            cmd += " -k"
        subprocess.call(cmd, shell=True, stdout=output_file, stderr=output_file)


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
        "-i",
        "--template_path",
        type=str,
        default=None,
        help="Input template path.",
    )

    parser.add_argument(
        "-e",
        "--exp_name",
        type=str,
        default=None,
        help="folder to save. Overwrite the one in manip config.",
    )

    parser.add_argument(
        "-t",
        "--task",
        choices=["grasp", "mogen", "grasp_and_mogen", "mogen_dexonomy"],
        default="grasp",
    )

    parser.add_argument(
        "-m",
        "--save_mode",
        choices=["usd", "npy", "usd+npy", "none"],
        default="npy",
    )

    parser.add_argument(
        "-w",
        "--parallel_world",
        type=int,
        default=20,
        help="parallel world num (only used when task=grasp)",
    )

    parser.add_argument(
        "-k",
        "--skip",
        action="store_false",
        help="If True, skip existing files. (default: True)",
    )

    parser.add_argument("-g", "--gpu", nargs="+", required=True, help="gpu id list")
    args = parser.parse_args()

    manip_config_data = load_yaml(join_path(get_manip_configs_path(), args.manip_cfg_file))
    if args.template_path is not None:
        manip_config_data["world"]["template_path"] = args.template_path

    if (
        manip_config_data["world"]["start"] is not None
        and manip_config_data["world"]["end"] is not None
    ):
        all_obj_num = manip_config_data["world"]["end"] - manip_config_data["world"]["start"]
        original_start = manip_config_data["world"]["start"]
    else:
        all_obj_num = len(
            glob(
                join_path(get_assets_path(), manip_config_data["world"]["template_path"]),
                recursive=True,
            )
        )
        original_start = 0
    obj_num_lst = np.array([all_obj_num // len(args.gpu)] * len(args.gpu))
    obj_num_lst[: (all_obj_num % len(args.gpu))] += 1
    assert obj_num_lst.sum() == all_obj_num

    p_list = []
    if args.exp_name is not None:
        manip_config_data["exp_name"] = args.exp_name
    if manip_config_data["exp_name"] is not None:
        if not os.path.abspath(manip_config_data["exp_name"]):
            save_folder = os.path.join(
                args.manip_cfg_file[:-4], manip_config_data["exp_name"], "grasp_data"
            )
        else:
            save_folder = manip_config_data["exp_name"]
    else:
        save_folder = os.path.join(
            args.manip_cfg_file[:-4],
            datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
            "grasp_data",
        )

    runinfo_folder = os.path.join(
        get_output_path(), os.path.dirname(save_folder), "log/bodex_mogen"
    )
    os.makedirs(runinfo_folder, exist_ok=True)

    # create tmp manip cfg files
    for i, gpu_id in enumerate(args.gpu):
        new_manip_path = join_path(runinfo_folder, str(i) + "_config.yml")
        manip_config_data["world"]["start"] = int(original_start + (obj_num_lst[:i]).sum())
        manip_config_data["world"]["end"] = int(original_start + (obj_num_lst[: (i + 1)]).sum())
        write_yaml(manip_config_data, new_manip_path)

        output_path = join_path(runinfo_folder, str(i) + "_output.txt")

        p = multiprocessing.Process(
            target=worker,
            args=(
                gpu_id,
                args.task,
                new_manip_path,
                save_folder,
                output_path,
                args.save_mode,
                args.parallel_world,
                args.skip,
            ),
        )
        p.start()
        print(f"create process :{p.pid}")
        p_list.append(p)

    for p in p_list:
        p.join()
