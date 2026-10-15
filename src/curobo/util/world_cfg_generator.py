import numpy as np
import transforms3d
from glob import glob

from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

from curobo.util.logger import log_warn
from curobo.util_file import load_json, load_scene_cfg, join_path, get_assets_path


def numpy_quaternion_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = np.split(quaternions, 4, -1)

    two_s = 2.0 / (quaternions * quaternions).sum(-1, keepdims=True)

    o = np.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )

    return o.reshape(quaternions.shape[:-1] + (3, 3))


class GraspConfigDataset(Dataset):
    def __init__(self, type, template_path, start, end):
        assert type == "grasp"
        template_path = join_path(get_assets_path(), template_path)
        self.grasp_path_lst = np.random.permutation(sorted(glob(template_path, recursive=True)))[
            start:end
        ]
        log_warn(
            f"From {template_path} get {len(self.grasp_path_lst)} grasps. Start: {start}, End: {end}."
        )
        return

    def __len__(self):
        return len(self.grasp_path_lst)

    def __getitem__(self, index):
        full_path = self.grasp_path_lst[index]
        cfg = np.load(full_path, allow_pickle=True).item()
        scene_cfg = load_scene_cfg(cfg["scene_path"][0])
        for k, v in cfg.items():
            cfg[k] = v[0]
        cfg["save_prefix"] = scene_cfg["scene_id"] + "_"
        return cfg


def scenecfg2worldcfg(scene_cfg):
    world_cfg = {}
    for obj_name, obj_cfg in scene_cfg["scene"].items():
        if obj_cfg["type"] == "rigid_object":
            if "mesh" not in world_cfg:
                world_cfg["mesh"] = {}
            world_cfg["mesh"][scene_cfg["scene_id"] + obj_name] = {
                "scale": obj_cfg["scale"],
                "pose": obj_cfg["pose"],
                "file_path": obj_cfg["file_path"],
                "urdf_path": obj_cfg["urdf_path"],
            }
        elif obj_cfg["type"] == "plane":
            if "cuboid" not in world_cfg:
                world_cfg["cuboid"] = {}
            assert obj_cfg["pose"][3] == 1
            # Use the plane's pose as the table-top position (was hardcoded at z=0).
            # We render it as a 5x5x0.2m cuboid with its TOP at that z.
            tx, ty, tz = float(obj_cfg["pose"][0]), float(obj_cfg["pose"][1]), float(obj_cfg["pose"][2])
            world_cfg["cuboid"]["table"] = {
                "dims": [5.0, 5.0, 0.2],
                "pose": [tx, ty, tz - 0.1, 1, 0, 0, 0.0],
            }
        else:
            raise NotImplementedError("Unsupported object type")
    return world_cfg


class DexonomyConfigDataset(Dataset):
    def __init__(self, type, template_path, start, end):
        assert type == "dexonomy"
        self.grasp_path_lst = np.random.permutation(sorted(glob(template_path, recursive=True)))[
            start:end
        ]
        log_warn(
            f"From {template_path} get {len(self.grasp_path_lst)} grasps. Start: {start}, End: {end}."
        )
        return

    def __len__(self):
        return len(self.grasp_path_lst)

    def __getitem__(self, index):
        full_path = self.grasp_path_lst[index]
        cfg = np.load(full_path, allow_pickle=True).item()

        scene_cfg = load_scene_cfg(join_path(get_assets_path(), f"../{cfg['scene_path']}"))

        if cfg["hand_name"] == "real_shadow":
            v = cfg["pregrasp_qpos"]
            # Change qpos order of thumb
            cfg["pregrasp_qpos_urdf"] = np.concatenate([v[:, :7], v[:, -5:], v[:, 7:-5]], axis=-1)
            # Add a translation bias of palm which is included in XML but ignored in URDF
            tmp_rot = numpy_quaternion_to_matrix(v[:, 3:7])
            cfg["pregrasp_qpos_urdf"][:, :3] += (
                tmp_rot @ np.array([0, 0, 0.034]).reshape(1, 3, 1)
            ).squeeze(-1)
        else:
            raise NotImplementedError

        cfg["world_cfg"] = scenecfg2worldcfg(scene_cfg)
        cfg["save_prefix"] = (
            full_path.split("succ_grasp/")[-1].split("grasp_data/")[-1].split("grasp.npy")[0]
        )
        return cfg


class WorldConfigDataset(Dataset):

    def __init__(self, type, template_path, start, end):
        assert type == "scene_cfg"
        scene_cfg_path = join_path(get_assets_path(), template_path)
        self.scene_path_lst = np.random.permutation(sorted(glob(scene_cfg_path)))[start:end]
        log_warn(
            f"From {scene_cfg_path} get {len(self.scene_path_lst)} scene cfgs. Start: {start}, End: {end}."
        )
        return

    def __len__(self):
        return len(self.scene_path_lst)

    def __getitem__(self, index):
        scene_path = self.scene_path_lst[index]
        scene_cfg = load_scene_cfg(scene_path)
        scene_id = scene_cfg["scene_id"]

        obj_name = scene_cfg["task"]["obj_name"]
        obj_cfg = scene_cfg["scene"][obj_name]
        obj_scale = obj_cfg["scale"]
        obj_pose = obj_cfg["pose"]

        json_data = load_json(obj_cfg["info_path"])
        obj_rot = transforms3d.quaternions.quat2mat(obj_pose[3:])
        gravity_center = obj_pose[:3] + obj_rot @ json_data["gravity_center"] * obj_scale
        obb_length = np.linalg.norm(obj_scale * json_data["obb"]) / 2

        return {
            "scene_path": scene_path,
            "world_cfg": scenecfg2worldcfg(scene_cfg),
            "manip_name": scene_id + obj_name,
            "obj_gravity_center": gravity_center,
            "obj_obb_length": obb_length,
            "save_prefix": f"{scene_id}_",
        }


def _world_config_collate_fn(list_data):
    if "world_cfg" in list_data[0]:
        world_cfg_lst = [i.pop("world_cfg") for i in list_data]
    else:
        world_cfg_lst = None
    ret_data = default_collate(list_data)
    if world_cfg_lst is not None:
        ret_data["world_cfg"] = world_cfg_lst
    return ret_data


def get_world_config_dataloader(configs, batch_size):
    if configs["type"] == "scene_cfg":
        dataset = WorldConfigDataset(**configs)
    elif configs["type"] == "grasp":
        dataset = GraspConfigDataset(**configs)
    elif configs["type"] == "dexonomy":
        dataset = DexonomyConfigDataset(**configs)
    else:
        raise NotImplementedError

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=_world_config_collate_fn
    )
    return dataloader
