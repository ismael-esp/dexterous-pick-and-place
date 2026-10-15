import torch
from typing import List

from curobo.types.base import TensorDeviceType
from curobo.util.tensor_util import normalize_vector

from abc import abstractmethod

class GraspEnergyBase:
    # friction coef: [miu_1, miu_2]. If miu_2 != 0, use soft finger contact model.  
    miu_coef: List
    
    tensor_args: TensorDeviceType
    
    def __init__(self, miu_coef, obj_gravity_center, obj_obb_length, tensor_args, enable_density):
        self.miu_coef = miu_coef
        # friction cone parameters
        assert self.miu_coef[0] <= 1 and self.miu_coef[0] > 0              

        self.tensor_args = tensor_args
        self.obj_gravity_center = None
        self.obj_obb_length = None 
        self.reset(obj_gravity_center, obj_obb_length)
        
        # for rotation matrix construction
        self.rot_base1 = self.tensor_args.to_device([0, 1, 0])
        self.rot_base2 = self.tensor_args.to_device([0, 0, 1])
        self.enable_density = enable_density
        self.grasp_batch = self.force_batch = 0
        return
        
    def utils_1axis_to_3axes(self, axis_0):
        '''
        One 3D direction to three 3D axes for constructing 3x3 rotation matrix. 
        
        Parameters
        ----------
        axis_0: [..., 3]

        Returns 
        ----------
        axis_0: [..., 3]
        axis_1: [..., 3]
        axis_2: [..., 3]
        
        '''

        tmp_rot_base1 = self.rot_base1.view([1]*(len(axis_0.shape)-1)+[3])
        tmp_rot_base2 = self.rot_base2.view([1]*(len(axis_0.shape)-1)+[3])
        
        proj_xy = (axis_0 * tmp_rot_base1).sum(dim=-1, keepdim=True).abs()
        axis_1 = torch.where(proj_xy > 0.99, tmp_rot_base2, tmp_rot_base1)  # avoid normal prependicular to axis_y1
        # NOTE the next line is necessary for gradient descent! Otherwise the differentiability is bad if normal is similar to axis_1!
        axis_1 = normalize_vector(axis_1 - (axis_1 * axis_0).sum(dim=-1, keepdim=True) * axis_0).detach()   
        axis_1 = normalize_vector(axis_1 - (axis_1 * axis_0).sum(dim=-1, keepdim=True) * axis_0)
        axis_2 = torch.cross(axis_0, axis_1, dim=-1)
        return axis_0, axis_1, axis_2
    
    def construct_grasp_matrix(self, pos, normal):
        axis_0, axis_1, axis_2 = self.utils_1axis_to_3axes(normal)
        env_num = self.obj_gravity_center.shape[0]
        batch_num = pos.shape[0] // env_num
        relative_pos = ((pos.view(env_num, batch_num, -1, 3) - self.obj_gravity_center.view(-1, 1, 1, 3)) / self.obj_obb_length.view(-1, 1, 1, 1)).view(env_num*batch_num, -1, 3)
        w0 = torch.cat([axis_0, torch.cross(relative_pos, axis_0, dim=-1)], dim=-1)
        w1 = torch.cat([axis_1, torch.cross(relative_pos, axis_1, dim=-1)], dim=-1)
        w2 = torch.cat([axis_2, torch.cross(relative_pos, axis_2, dim=-1)], dim=-1)
        if self.miu_coef[1] > 0:
            w3 = torch.cat([axis_0 * 0.0, axis_0 * self.miu_coef[1]], dim=-1)
            grasp_matrix = torch.stack([w0, w1, w2, w3], dim=-1)       # [b, n, 6, 4]
        else:
            grasp_matrix = torch.stack([w0, w1, w2], dim=-1)       # [b, n, 6, 3]
        contact_frame = torch.stack([axis_0, axis_1, axis_2], dim=-1)
        contact_force = self.tensor_args.to_device([1., 0., 0.]).view(1, 1, -1).expand_as(pos)
        return grasp_matrix, contact_frame, contact_force
    
    def estimate_density(self, normal):
        cos_theta = (normal.unsqueeze(-2) * normal.unsqueeze(-3)).sum(dim=-1) # [b, n, n]
        density = 1 / torch.clamp(torch.clamp(cos_theta, min=0).sum(dim=-1), min=1e-4)
        return density.detach()
    
    @abstractmethod
    def forward(self, pos, normal, test_wrenches):
        raise NotImplementedError
    
    def reset(self, gravity_center, obb_length):
        if gravity_center is not None:
            if self.obj_gravity_center is not None and gravity_center.shape == self.obj_gravity_center.shape:
                self.obj_gravity_center[:] = torch.tensor(gravity_center)
            else:
                self.obj_gravity_center = self.tensor_args.to_device(gravity_center)
        if obb_length is not None:
            if self.obj_obb_length is not None and obb_length.shape == self.obj_obb_length.shape:
                self.obj_obb_length[:] = torch.tensor(obb_length)
            else:
                self.obj_obb_length = self.tensor_args.to_device(obb_length)
        return 