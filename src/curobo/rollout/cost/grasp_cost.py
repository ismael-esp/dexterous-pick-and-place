# Standard Library
from dataclasses import dataclass
from typing import Optional, Union, List, Dict
from copy import deepcopy
import random 

# Third Party
import torch
import numpy as np 
import math

# Local Folder
from .cost_base import CostBase, CostConfig
from curobo.geom.sdf.world import ContactBuffer, WorldCollision
from curobo.geom.transform import pose_multiply
from curobo.util.sample_lib import HaltonGenerator
from curobo.util.tensor_util import normalize_vector
from curobo.geom.basic_transform import torch_quaternion_to_matrix
from .grasp_energy import init_grasp_energy, GraspEnergyBase

@dataclass
class GraspCostConfig(CostConfig):
    
    task_dict: dict = None 
    
    ge_param: Dict = None 
        
    perturb_strength_bound: List[float] = None 
    
    contact_points_idx: torch.Tensor = None 
    
    contact_mesh_idx: List = None
    
    world_coll_checker: Optional[WorldCollision] = None
    
    finger_num: int = None
    
    contact_strategy: Dict = None 
    
    def __post_init__(self):
        if self.contact_strategy is not None and isinstance(self.contact_strategy['opt_progress'], List):
            self.contact_strategy['opt_progress'] = self.tensor_args.to_device(self.contact_strategy['opt_progress'])

        return super().__post_init__()



class GraspCost(CostBase, GraspCostConfig):
    def __init__(self, config: GraspCostConfig, grasp_energy: GraspEnergyBase = None):
        GraspCostConfig.__init__(self, **vars(config))
        CostBase.__init__(self)
        
        self.contact_stage = 0
        self.contact_query_mode = None
        
        # for contact query
        self._contact_buffer = ContactBuffer()

        if grasp_energy is None:
            self.GraspEnergy = init_grasp_energy(self.ge_param, self.tensor_args)
        else:
            self.GraspEnergy = grasp_energy
            
        # initialize TWS
        self.TWS = self.init_TWS(self.task_dict)
        self.count = 0
        self.update_perturb_info()
        
        return 
    
    def init_TWS(self, task):
        '''

        Returns
        -----------
        TWS: dict{
            'f': [k, 3]
            'p': [k, 3]
            't': [3]
        }

        '''
        TWS = {}

        if task['f'] is not None:
            external_f = normalize_vector(self.tensor_args.to_device(task['f']))
            axis_0, axis_1, axis_2 = self.GraspEnergy.utils_1axis_to_3axes(external_f)
            TWS['f'] = [external_f]
            robust_angle = task['gamma']
            assert robust_angle >= 0 and robust_angle <= 180
            if robust_angle > 0:
                if robust_angle > 90:
                    TWS['f'].append(-external_f)
                    robust_angle = 90 / 180 * np.pi    # treat >90 angle similarly as 180. use 90 to compute the below directions.
                else:
                    robust_angle = robust_angle / 180 * np.pi
                    
                TWS['f'].append((axis_0*math.cos(robust_angle)+axis_1*math.sin(robust_angle)))
                TWS['f'].append((axis_0*math.cos(robust_angle)-axis_1*math.sin(robust_angle)))
                TWS['f'].append((axis_0*math.cos(robust_angle)-axis_2*math.sin(robust_angle)))
                TWS['f'].append((axis_0*math.cos(robust_angle)+axis_2*math.sin(robust_angle)))
            
            TWS['f'] = torch.stack(TWS['f'], dim=0) # [1, 3] or [5, 3] or [6, 3]
            
        # if task['p'] is not None:
        #     task['p'] = [0, 0, 0]
        # TWS['p'] = self.tensor_args.to_device([[[0.5, 0., 0.]]])
        # if TWS['p'].norm() != 0:
        #     raise NotImplementedError
        # TWS['f'] = self.tensor_args.to_device([[0, 0, 1]])
        if task['t'] is None:
            task['t'] = [0, 0, 0]
        TWS['t'] = self.tensor_args.to_device(task['t']).unsqueeze(0).expand_as(TWS['f']) 
        if TWS['t'].norm() != 0:
            raise NotImplementedError
        TWS['w'] =  torch.cat([TWS['f'], TWS['t']], dim=-1).unsqueeze(-1)  # [m, 6, 1]
        # print(TWS['w'])
        # exit(1)
        return TWS 
    
    def update_perturb_info(self):    
        pert_dim = 3    
        self.perturb_template = self.tensor_args.to_device(torch.zeros((pert_dim*2, pert_dim)))
        for i in range(pert_dim):
            self.perturb_template[i, i] = 1
            self.perturb_template[i+pert_dim, i] = -1
        self.perturb_strength_generator = HaltonGenerator(
            self.perturb_template.shape[0],
            self.tensor_args,
            up_bounds=self.perturb_strength_bound[1],
            low_bounds=self.perturb_strength_bound[0],
            seed=123,
        )
        return 

    def reset(self, gravity_center, obb_length):
        self.contact_stage = 0
        self.count = 0
        self.GraspEnergy.reset(gravity_center, obb_length)
        return 
    
    def _get_contact_stage(self, opt_progress):
        assert self.contact_strategy is not None
        max_ge_stage = self.contact_strategy['max_ge_stage']
        new_contact_stage = (self.contact_strategy['opt_progress'] <= opt_progress).int().sum() - 1
        contact_query_mode = self.contact_strategy['contact_query_mode'][self.contact_stage]
        contact_distance = self.contact_strategy['distance'][self.contact_stage]
        switch_stage_flag = (self.contact_stage < new_contact_stage)
        ge_stage_flag = (self.contact_stage <= max_ge_stage) or (new_contact_stage <= max_ge_stage)
        save_qpos_flag = switch_stage_flag and self.contact_strategy['save_qpos'][self.contact_stage]
        self.contact_stage = new_contact_stage
        if self.contact_query_mode != contact_query_mode:
            self.contact_query_mode = contact_query_mode
            self.update_perturb_info()
        return contact_distance, switch_stage_flag, ge_stage_flag, save_qpos_flag
    
    @torch.no_grad()
    def evaluate(self, link_pos_quat, env_query_idx):
        '''
        Use mesh-to-mesh nearest point query (gjk algorithm) for evaluating both distance error and grasp error.
        '''
        self._get_contact_stage(0.0)
        contact_robot_pose = link_pos_quat[..., self.contact_mesh_idx, :].unsqueeze(1)
        dist_upper_bound = self.tensor_args.to_device(torch.ones_like(contact_robot_pose[...,-1]).float() * 50)
        ho_contact_point, contact_dist, contact_normal, _, _ = self.world_coll_checker.get_mesh_contact_pdn(
            contact_robot_pose.detach(), self.perturb_template, env_query_idx, dist_upper_bound)
        contact_point = ho_contact_point[..., :3]
        grasp_energy, grasp_error, contact_frame, contact_force = self.GraspEnergy.forward(contact_point.squeeze(1), contact_normal.squeeze(1), self.TWS['w'])
        dist_error = contact_dist.abs().mean(dim=-1).squeeze(1)
        return dist_error, grasp_error, contact_point, contact_frame, contact_force
    
    
    def loss_w_ge(self, pos, dist, normal, contact_distance, opt_progress):
        grasp_energy, grasp_error, contact_frame, contact_force = self.GraspEnergy.forward(pos.squeeze(1), normal.squeeze(1), self.TWS['w'])
        
        E_angle = self.weight[0] * grasp_energy.mean(dim=-1, keepdim=True)
        E_dist = self.weight[1] * ((dist - contact_distance) ** 2).mean(dim=-1)
        E_regu = 0 * E_dist 
        dist_error = (dist - contact_distance).abs().max(dim=-1)[0]

        return E_angle, E_dist, E_regu, dist_error, grasp_error.detach(), contact_frame.detach(), contact_force.detach()
    
    
    def loss_wo_ge(self, raw_spheres, contact_distance, mesh_contacts, mesh_dist, opt_progress):
        target_sphere_center_inner = self.target_contact_position - self.target_contact_normal * (raw_spheres[..., -1] + contact_distance).unsqueeze(-1)
        E_dist = self.weight[1] * ((raw_spheres[..., :3] - target_sphere_center_inner)**2).sum(dim=-1).mean(dim=-1) * min(max(10 - 10 * opt_progress, 0.), 1.)
        E_angle = 0 * E_dist
        if mesh_dist is not None:
            target_sphere_center = self.target_contact_position - self.target_contact_normal * contact_distance
            E_dist += self.weight[1] * ((mesh_contacts - target_sphere_center)**2).sum(dim=-1).mean(dim=-1)
            E_regu = self.weight[2] * (((mesh_dist - contact_distance)**2).mean(dim=-1))
            dist_error = (mesh_dist - contact_distance).abs().max(dim=-1)[0]
        else:
            target_sphere_center = self.target_contact_position - self.target_contact_normal * (raw_spheres[..., -1] + contact_distance).unsqueeze(-1)
            E_regu = 0 * E_dist
            dist_error = (raw_spheres[..., :3] - target_sphere_center).norm(dim=-1).max(dim=-1)[0]
        return E_angle, E_dist, E_regu, dist_error 
    
    def forward(self, robot_sphere_in, link_pos_quat, env_query_idx, opt_progress):
        contact_distance, switch_stage_flag, ge_stage_flag, save_qpos_flag = self._get_contact_stage(opt_progress)

        if self.contact_query_mode == -1:
            robot_contact_points = robot_sphere_in[..., self.contact_points_idx, :]
            b, h, n_points, _ = robot_contact_points.shape
            n_pert = self.perturb_template.shape[-2]
            strength_sample = self.perturb_strength_generator.get_samples(b*h*n_points, bounded=True)
            perturb = (self.perturb_template * strength_sample.unsqueeze(-1)).view(b, h, n_points, n_pert, -1)
            self._contact_buffer.update_buffer_shape([b, h, n_points, n_pert], self.tensor_args)
            raw_pos, raw_dist, raw_normal, debug_pos, debug_normal = self.world_coll_checker.get_sphere_contact_pdn(
                robot_contact_points, self._contact_buffer, perturb, env_query_idx)
            raw_pos_robot_with_grad = None
            mesh_dist_with_grad = None
        elif self.contact_query_mode == 0:
            b, h, _, _ = robot_sphere_in.shape
            n_points = len(self.contact_mesh_idx)
            n_pert = self.perturb_template.shape[-2]
            strength_sample = self.perturb_strength_generator.get_samples(b*h*n_points, bounded=True)
            perturb = (self.perturb_template * strength_sample.unsqueeze(-1)).view(b, h, n_points, n_pert, -1)
            query_info = link_pos_quat[..., self.contact_mesh_idx, :]
            robot_contact_points = robot_sphere_in[..., self.contact_points_idx, :]
            rot = torch_quaternion_to_matrix(query_info[..., 3:])
            if self.count % 5 == 0:
                self._contact_buffer.update_buffer_shape([b, h, n_points, n_pert], self.tensor_args)
                _, dist_upper_bound, _, _, _ = self.world_coll_checker.get_sphere_contact_pdn(
                    robot_contact_points.detach(), self._contact_buffer, perturb, env_query_idx)
                
                raw_pos_two, raw_dist, self.raw_normal, self.debug_pos, self.debug_normal = self.world_coll_checker.get_mesh_contact_pdn(
                    query_info.detach(), perturb, env_query_idx, dist_upper_bound)
                self.raw_pos_robot_frame = (rot.transpose(-1,-2) @ (raw_pos_two[..., 3:] - query_info[..., :3]).unsqueeze(-1)).squeeze(-1).detach()
                self.raw_pos = raw_pos_two[..., :3]
                self.dist_sign = (raw_dist > 0) * 2 - 1
            raw_pos = self.raw_pos
            raw_normal = self.raw_normal
            debug_pos = self.debug_pos
            debug_normal = self.debug_normal
            raw_pos_robot_with_grad = (rot @ self.raw_pos_robot_frame.unsqueeze(-1)).squeeze(-1) + query_info[..., :3]
            mesh_dist_with_grad = self.dist_sign * (raw_pos_robot_with_grad - self.raw_pos).norm(dim=-1)
            self.count += 1
        else:
            raise NotImplementedError

        if not ge_stage_flag:
            E_angle, E_dist, E_regu, dist_error = self.loss_wo_ge(robot_contact_points, contact_distance, raw_pos_robot_with_grad, mesh_dist_with_grad, opt_progress)
            grasp_error = contact_frame = contact_force = None 
        else:
            E_angle, E_dist, E_regu, dist_error, grasp_error, contact_frame, contact_force = self.loss_w_ge(raw_pos, raw_dist, raw_normal, contact_distance, opt_progress)

        if switch_stage_flag and ge_stage_flag:
            self.target_contact_position = raw_pos.detach().clone()
            self.target_contact_normal = raw_normal.detach().clone()

        debug_pert_p = (robot_contact_points[..., :3].unsqueeze(-2) + perturb[..., :6, :3]).view(b, h, -1, 3)
        debug = {
            'input': robot_sphere_in, # [b, h, n, 4]
            'op': debug_pert_p, # [b, h, n, 3]
            'debug_posi': debug_pos[..., :6, :3].contiguous(),
            'debug_normal': debug_normal[..., :6, :].contiguous(),
            'normal': raw_normal,
            
            'save_qpos_flag': save_qpos_flag,
            'dist_error': dist_error.detach().squeeze(1), # [b]
            'grasp_error': grasp_error,  # [b, m]
            'contact_point': raw_pos.detach().squeeze(1),   # [b, p, 3]
            'contact_frame': contact_frame, # [b, p, 3, 3]
            'contact_force': contact_force, # [b, m, p, 3]
        }

        return E_angle, E_dist, E_regu, debug
