from typing import Dict, List
import os 
import torch
import numpy as np
import math 

from curobo.util.sample_lib import HaltonGenerator
from curobo.util.tensor_util import normalize_vector
from curobo.util.logger import log_warn
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.types.math import Pose
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.basic_transform import euler_angles_to_matrix, matrix_from_rot_repre, matrix_to_quaternion
from curobo.geom.sdf.world import WorldCollision
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig


class HeurGraspSeedGenerator:
    
    def __init__(self,
        seeder_cfg: Dict, 
        full_robot_model: CudaRobotModel,
        ik_solver: IKSolver,
        world_coll_checker: WorldCollision,
        obj_lst: List, 
        tensor_args: TensorDeviceType
    ):
        self.tensor_args = tensor_args
        all_dof = full_robot_model.dof
        use_root_pose = full_robot_model.use_root_pose
        self.full_robot_model = full_robot_model
        self.ik_solver = ik_solver
        self.tr_link_names = full_robot_model.transfered_link_name
        if ik_solver is not None:
            self.replace_ind = ik_solver.kinematics.kinematics_config.get_replace_index(full_robot_model.kinematics_config)
        
        if use_root_pose:
            assert len(self.tr_link_names) == 1
            self.q_dof = all_dof - 7
        else:
            if seeder_cfg['ik_init_q'] is None:
                self.ik_init = None 
            else:
                self.ik_init = tensor_args.to_device(seeder_cfg['ik_init_q']).view(1,1,-1)
            self.q_dof = all_dof
        
        # jitter r and t 
        if self.tr_link_names is not None:
            self.skip_transfer = seeder_cfg['skip_transfer']
            self.jitter_rt_random_gen = self._set_jitter_tr(seeder_cfg['jitter_dist'], seeder_cfg['jitter_angle'])
            
        self.seeder_cfg = seeder_cfg
        self.world_coll_checker = world_coll_checker
        self.reset(obj_lst)
        return 
    
    def _init_r_from_axis(self, r0):
        base_axis_palm = normalize_vector(r0)
        base_t1 = self.tensor_args.to_device([0, 1, 0])
        base_t2 = self.tensor_args.to_device([0, 0, 1]) # avoid base_axis_palm parallel to base_t1
        proj_xy = (base_t1 * base_axis_palm).sum(dim=-1, keepdim=True).abs()
        base_axis_thumb = torch.where(proj_xy > 0.99, base_t2, base_t1)
        r6d = torch.cat([base_axis_palm, base_axis_thumb], dim=-1)
        return r6d 
    
    def _set_base_trq(self, t, r, q, extra_info: WorldCollision=None):
        # log_warn(f'Initialize hand pose. t: {t}, r: {r}, q: {q}')
        if self.tr_link_names is None:
            base_t = None 
            base_r = None 
        else:
            tr_num = len(self.tr_link_names)
            if t is not None and r is not None:
                base_t = self.tensor_args.to_device(t).view(1, 1, tr_num, -1)
                r_repre = self.tensor_args.to_device(r).view(1, 1, tr_num, -1)
                ind_random_gen = None
            elif t is None and r is None:
                base_t = extra_info.surface_sample_positions.view(-1, 1, tr_num, 3)
                r_repre = self._init_r_from_axis(-extra_info.surface_sample_normals).view(-1, 1, tr_num, 6)
                ind_random_gen = HaltonGenerator(
                    len(extra_info.surface_sample_ind_upper),
                    self.tensor_args,
                    up_bounds=extra_info.surface_sample_ind_upper,
                    low_bounds=extra_info.surface_sample_ind_lower,
                    seed=1312)
                assert tr_num == 1, "TODO: implement auto initialization for multiple hands"
            elif t is None and r is not None:
                # Sample palm positions from the object surface but OVERRIDE the
                # orientation with a user-supplied constant rotation in world
                # frame. Use-case: top-down tabletop grasps constrained to the
                # arm's reachable wrist wedge.
                #
                # `r` accepts two formats:
                #   - length-3: treated as the palm-X axis (approach / "palm
                #     pressing" direction). Thumb axis auto-picked via
                #     _init_r_from_axis (world +Y, falls back to +Z when
                #     parallel). See `compute_rotation_matrix_from_ortho6d` in
                #     basic_transform.py — stored as [palm_x, palm_y].
                #   - length-6: full ortho-6D rep = [palm_x (3), palm_y (3)].
                #     The decoder Gram-Schmidts palm_y against palm_x then
                #     computes palm_z = palm_x × palm_y.
                base_t = extra_info.surface_sample_positions.view(-1, 1, tr_num, 3)
                r_t = self.tensor_args.to_device(r)
                N = base_t.shape[0]
                if r_t.numel() == 3:
                    r_axis = r_t.view(1, tr_num, 3)
                    r_axis = r_axis / r_axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    r_axis = r_axis.expand(N, tr_num, 3)
                    r_repre = self._init_r_from_axis(r_axis).view(-1, 1, tr_num, 6)
                elif r_t.numel() == 6:
                    r6 = r_t.view(1, 1, tr_num, 6).expand(N, 1, tr_num, 6).contiguous()
                    r_repre = r6
                else:
                    raise ValueError(
                        f"seeder_cfg.r must be length 3 (palm_x axis) or 6 (full ortho6d); got {r_t.numel()}"
                    )
                ind_random_gen = HaltonGenerator(
                    len(extra_info.surface_sample_ind_upper),
                    self.tensor_args,
                    up_bounds=extra_info.surface_sample_ind_upper,
                    low_bounds=extra_info.surface_sample_ind_lower,
                    seed=1312)
                assert tr_num == 1, "TODO: implement auto initialization for multiple hands"
            else:
                raise NotImplementedError
                
            base_r = matrix_from_rot_repre(r_repre)
    
        base_q = q if q is not None else [0]*self.q_dof
        assert len(base_q) == self.q_dof, self.q_dof
        base_q = self.tensor_args.to_device(base_q).view(1, 1, -1)
        if self.tr_link_names is not None and base_t.shape[0] > 1:
            base_q = base_q.expand(base_t.shape[0], base_t.shape[1], -1)
        return base_t, base_r, base_q, ind_random_gen
    
    def _set_jitter_tr(self, jitter_dist, jitter_angle):
        jitter_bound_low = (jitter_dist[0] + [i/180*np.pi for i in jitter_angle[0]]) * len(self.tr_link_names)
        jitter_bound_up = (jitter_dist[1] + [i/180*np.pi for i in jitter_angle[1]]) * len(self.tr_link_names)
        random_gen = HaltonGenerator(
            len(jitter_bound_low), 
            self.tensor_args, 
            up_bounds=jitter_bound_up,
            low_bounds=jitter_bound_low,
            seed=1312)
        return random_gen
    
    def _load_base_trq(self, obj_lst, load_path_dict):
        robot_pose = []
        for obj_code in obj_lst:
            obj_code = obj_code.split('_scale_')[0] # This is only to fit the need of jialiang's data
            path = os.path.join(load_path_dict['base'], obj_code, load_path_dict['suffix'])
            log_warn(f'load hand pose initialization from {path}')
            data = dict(np.load(path, allow_pickle=True))
            tmp_robot_pose = self.tensor_args.to_device(data['robot_pose'])
            robot_pose.append(tmp_robot_pose)
        robot_pose = torch.stack(robot_pose, dim=0)
        if self.tr_link_names is None:
            base_t = None 
            base_r = None
        else:
            rot_repre_num = (robot_pose.shape[-1] - self.q_dof) // len(self.tr_link_names) - 3
            if len(self.tr_link_names) > 1:
                raise NotImplementedError
            base_t = robot_pose[..., :3].unsqueeze(-2)
            base_r = matrix_from_rot_repre(robot_pose[..., 3:3+rot_repre_num]).unsqueeze(-3)
        base_q = robot_pose[..., -self.q_dof:][..., load_path_dict['reorder_q']]
        return base_t, base_r, base_q
    
    def reset(self, obj_lst = None):
        self.jitter_rt_random_gen.reset()
        
        # init base r, t, q
        if self.seeder_cfg['load_path'] is not None:
            assert obj_lst is not None
            # [b, n, tr_num, 3], [b, n, tr_num, 3, 3], [b, n, q_dof] 
            self.base_t, self.base_r, self.base_q = self._load_base_trq(obj_lst, self.seeder_cfg['load_path'])
            self.ind_random_gen = None
        else:
            # [b/1, 1, tr_num, 3], [b/1, 1, tr_num, 3, 3], [b/1, 1, q_dof] 
            self.base_t, self.base_r, self.base_q, self.ind_random_gen = self._set_base_trq(self.seeder_cfg['t'], 
                                        self.seeder_cfg['r'], 
                                        self.seeder_cfg['q'], 
                                        extra_info=self.world_coll_checker
                                    )
        return 
    
    def _jitter_on_base_tr(self, base_trans, base_rot):
        batch, num_samples, tr_num = base_trans.shape[:-1]
        rand_num = self.jitter_rt_random_gen.get_samples(batch*num_samples, bounded=True).view(batch, num_samples, tr_num, 6)
        rand_dist = rand_num[..., :3]
        rand_jitter_angle = rand_num[..., 3:]
        
        # calculate jittered translation and rotation
        jitter_rotation = euler_angles_to_matrix(torch.flip(rand_jitter_angle, [-1]), 'ZYX')
        final_rotation = base_rot @ jitter_rotation
        final_translation = base_trans - (final_rotation @ rand_dist.unsqueeze(-1)).squeeze(-1)
        return final_translation, final_rotation
    
    def _sample_to_shape(self, batch, num_samples):
        if self.ind_random_gen is not None:
            sample_idx = self.ind_random_gen.get_samples(num_samples, bounded=True).long()
            base_rot = self.base_r[sample_idx].transpose(1, 0).squeeze(2) 
            base_trans = self.base_t[sample_idx].transpose(1, 0).squeeze(2) 
            base_q = self.base_q[sample_idx].transpose(1, 0).squeeze(2) 
        else:
            if self.base_q.shape[0] < batch or self.base_q.shape[1] < num_samples: 
                repeat_b = math.ceil(batch / self.base_q.shape[0])
                repeat_n = math.ceil(num_samples / self.base_q.shape[1])
                self.base_r = self.base_r.repeat(repeat_b, repeat_n, 1, 1, 1) if self.base_r is not None else None 
                self.base_t = self.base_t.repeat(repeat_b, repeat_n, 1, 1) if self.base_t is not None else None
                self.base_q = self.base_q.repeat(repeat_b, repeat_n, 1)
            base_rot = self.base_r[:batch, :num_samples] if self.base_r is not None else None 
            base_trans = self.base_t[:batch, :num_samples] if self.base_t is not None else None
            base_q = self.base_q[:batch, :num_samples]
        
        return base_trans, base_rot, base_q
    
    def get_samples(self, batch, num_samples):
        base_trans, base_rot, base_q = self._sample_to_shape(batch, num_samples)
        if self.tr_link_names is not None:
            final_trans, final_rot = self._jitter_on_base_tr(base_trans, base_rot)
            if not self.skip_transfer:
                final_trans, final_rot = self.full_robot_model.get_transfered_pose(final_trans.contiguous(), final_rot.contiguous(), self.tr_link_names)
            final_quat = matrix_to_quaternion(final_rot)

        if self.ik_solver is not None:
            target_link_poses = {}
            for i, link_name in enumerate(self.tr_link_names):
                target_link_poses[link_name] = Pose(final_trans[..., i, :].reshape(-1, 3), final_quat[..., i, :].reshape(-1, 4))
                if i == 0:
                    goal = target_link_poses[link_name] 
            ik_init = self.ik_init.expand(batch, num_samples, -1) if self.ik_init is not None else None 
            result = self.ik_solver.solve_batch(goal, 
                                    seed_config=ik_init, 
                                    link_poses=target_link_poses
                                )
            if torch.any(~result.success):
                log_warn(f'ik result: {result.success.flatten()}')
            arm_q = result.solution.view(batch, num_samples, -1)
            hand_pose = base_q
            hand_pose[..., self.replace_ind] = arm_q
        else:
            hand_pose = torch.cat([final_trans.squeeze(-2), final_quat.squeeze(-2), base_q], dim=-1) 

        return hand_pose
    