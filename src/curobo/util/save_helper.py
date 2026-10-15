from typing import Dict, List
from dataclasses import dataclass
import os
import datetime

import numpy as np
import torch

from curobo.util.usd_helper import UsdHelper
from curobo.types.robot import JointState, RobotConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.util_file import (
    get_robot_configs_path,
    get_output_path,
    join_path,
    load_yaml,
)
from curobo.util.logger import log_warn


def dict_piece(d: Dict, piece_id: int, piece_num: int):
    new_d = {}
    for k, v in d.items():
        if isinstance(v, Dict):
            new_d[k] = dict_piece(v, piece_id, piece_num)
        elif v is not None:
            each = len(v) // piece_num
            new_d[k] = v[piece_id * each : (piece_id + 1) * each]
            if each == 1 and isinstance(new_d[k][0], str):
                new_d[k] = new_d[k][0]
        else:
            new_d[k] = None

    return new_d


def dict_select(d: Dict, keys: List, cpu_numpy: bool = True):
    new_d = {}
    for k in keys:
        if k in d.keys():
            new_d[k] = d[k]
            if cpu_numpy and isinstance(new_d[k], torch.Tensor):
                new_d[k] = new_d[k].cpu().numpy()
        # else:
        #     new_d[k] = None
    return new_d


@dataclass
class SaveHelper:
    robot_file: str
    save_folder: str
    task_name: str
    mode: str  # npy or usd or usd+npy or None
    npy_save_key: List = None
    usd_save_key: List = None
    kin_model: CudaRobotModel = None
    set_camera: bool = False

    def __post_init__(self):
        if self.kin_model is None:
            robot_config_data = load_yaml(join_path(get_robot_configs_path(), self.robot_file))
            robot_config_data["robot_cfg"]["kinematics"]["load_link_names_with_mesh"] = True
            robot_cfg = RobotConfig.from_dict(robot_config_data)
            self.kin_model = CudaRobotModel(robot_cfg.kinematics)
            if "robot_color" in robot_config_data["robot_cfg"].keys():
                self.robot_color = robot_config_data["robot_cfg"]["robot_color"]
            else:
                self.robot_color = [0.5, 0.5, 0.2, 1.0]

        self.save_folder = os.path.abspath(os.path.join(get_output_path(), self.save_folder))
        if self.npy_save_key is None:
            self.npy_save_key = [
                "robot_pose",
                "pregrasp_qpos",
                "grasp_qpos",
                "n_evo",
                "squeeze_qpos",
                "world_cfg",
                "manip_name",
                "scene_path",
                "contact_point",
                "contact_frame",
                "contact_force",
                "grasp_error",
                "dist_error",
                "pene_error",
            ]
        if self.usd_save_key is None:
            self.usd_save_key = ["world_model", "robot_pose", "debug_info"]

        return

    def save_piece(self, world_info_dict: Dict):
        file_prefix_lst = world_info_dict["save_prefix"]
        npy_dict = dict_select(world_info_dict, self.npy_save_key)
        usd_dict = dict_select(world_info_dict, self.usd_save_key, cpu_numpy=False)

        for i, file_prefix in enumerate(file_prefix_lst):
            if "npy" in self.mode:
                self._save_npy(dict_piece(npy_dict, i, len(file_prefix_lst)), file_prefix)
            if "usd" in self.mode:
                self._save_usd(dict_piece(usd_dict, i, len(file_prefix_lst)), file_prefix)
        return

    def exist_piece(self, file_prefix_lst: List[str]):
        if "npy" not in self.mode and "usd" not in self.mode:
            return False
        for i, file_prefix in enumerate(file_prefix_lst):
            if "npy" in self.mode:
                if not self._exist_npy(file_prefix):
                    return False
            if "usd" in self.mode:
                if not self._exist_usd(file_prefix):
                    return False
        return True

    def _exist_npy(self, file_prefix: str):
        npy_path = os.path.join(self.save_folder, file_prefix + self.task_name + ".npy")
        return os.path.exists(npy_path)

    def _exist_usd(self, file_prefix: str):
        usd_path = os.path.join(self.save_folder, file_prefix + self.task_name + ".usda")
        return os.path.exists(usd_path)

    def _save_npy(self, save_dict: Dict, file_prefix: str):
        save_dict["joint_names"] = self.kin_model.joint_names
        save_path = os.path.join(self.save_folder, file_prefix + self.task_name + ".npy")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, save_dict)
        log_warn(f"Save results to {save_path}")
        return save_path

    def _save_usd(self, save_dict, file_prefix: str):
        save_path = os.path.join(self.save_folder, file_prefix + self.task_name + ".usda")

        UsdHelper.write_trajectory_animation(
            None,
            save_dict["world_model"],
            None,
            JointState.from_position(
                save_dict["robot_pose"].reshape(-1, save_dict["robot_pose"].shape[-1])
            ),
            save_path=save_path,
            kin_model=self.kin_model,
            robot_color=self.robot_color,
            visualize_robot_spheres=not self.set_camera,
            base_frame="/grid_world_1",
            debug_info=save_dict["debug_info"] if "debug_info" in save_dict.keys() else None,
            set_camera=self.set_camera,
        )

        log_warn(f"save visualization to {save_path}")
        return save_path
