import torch 
import numpy as np

from curobo.util.tensor_util import normalize_vector
from curobo.util.sample_lib import random_sample_points_on_sphere
from .base import GraspEnergyBase


class TDGEnergy(GraspEnergyBase):
    
    def __init__(self, miu_coef, obj_gravity_center, obj_obb_length, tensor_args, enable_density, **args):
        super().__init__(miu_coef, obj_gravity_center, obj_obb_length, tensor_args, enable_density)
        self.miu_tensor = self.tensor_args.to_device(self.miu_coef[0])
        self.direct_num = 1000
        direction_3D = random_sample_points_on_sphere(3, self.direct_num)
        direction_6D = np.concatenate([direction_3D, direction_3D*0.], axis=-1)
        self.target_direction_6D = self.tensor_args.to_device(direction_6D).unsqueeze(0)     # [1, P, 3]
        self.F_center_direction = self.tensor_args.to_device([1,0,0])[None,None,None]
        if self.miu_coef[1] > 0:
            raise NotImplementedError
        return 
    
    def GWS(self, G_matrix, normal):
        '''Approximate the GWS boundary by dense samples.

        Returns
        ----------
        w: [b, P, 6]
        '''        
        # solve q_W(u): q_W(u) equals to G * q_F(G^T @ u), so first solve q_F(u')
        direction_F = normalize_vector((self.target_direction_6D.unsqueeze(1) @ G_matrix).transpose(2,1))   # G^T @ u: [b, P, n, 3] or [b, P, n, 4]
        proj_on_cn = (direction_F * self.F_center_direction).sum(dim=-1, keepdim=True)     # [b, P, n, 1] 
        perp_to_cn = direction_F - proj_on_cn * self.F_center_direction  # [b, P, n, 3]  or [b, P, n, 4]
        
        angles = torch.acos(torch.clamp(proj_on_cn, min=-1, max=1))     # [b, P, n, 1] 
        bottom_length = self.miu_tensor 
        bottom_angle = torch.atan(bottom_length)
        
        region1 = angles <= bottom_angle
        region2 = (angles > bottom_angle) & (angles <= np.pi / 2)
        region3 = (angles > np.pi / 2)
        perp_norm = perp_to_cn.norm(dim=-1, keepdim=True)

        # a more continuous approximation
        help3 = perp_norm / (perp_norm - 2 * bottom_length * torch.clamp(proj_on_cn, max=0))
        help2 = self.F_center_direction + bottom_length * normalize_vector(perp_to_cn)
        argmin_3D_on_normalized_cone = region1 * (self.F_center_direction + perp_to_cn / torch.clamp(proj_on_cn, min=torch.cos(bottom_angle) / 2)) \
                + region2 * help2 + region3 * help3 * help2 # [b, P, n, 3]
        
        # get q_W(u) = G * q_F(G^T @ u)
        w = (G_matrix.unsqueeze(1) @ argmin_3D_on_normalized_cone.unsqueeze(-1)).squeeze(-1) # [b, P, n, 6]

        # use density to change the force bound 
        if self.enable_density:
            density = self.estimate_density(normal)
            final_w = (w * density.unsqueeze(1).unsqueeze(-1)).sum(dim=2) # [b, P, 6]
        else:
            final_w = w.sum(dim=2)
        return final_w
    
    def forward(self, pos, normal, test_wrenches):
        # G: F \in R^3 (or R^4) -> W \in R^6
        G_matrix, contact_frame, contact_force = self.construct_grasp_matrix(pos, normal)
        w = self.GWS(G_matrix, normal)
        cos_wt = (normalize_vector(w) * self.target_direction_6D).sum(dim=-1)
        gras_energy = 10 * (1 - cos_wt).mean(dim=-1, keepdim=True)
        grasp_error = (1 - cos_wt).mean(dim=-1, keepdim=True).expand(-1, test_wrenches.shape[0])
        return gras_energy, grasp_error, contact_frame, contact_force
    
    
class TDGQ1Energy(TDGEnergy):
    '''
    Differentiability is bad due to MIN operator!
    '''
    def forward(self, pos, normal, target_wrenches):
        # G: F \in R^3 (or R^4) -> W \in R^6
        G_matrix, contact_frame, contact_force = self.construct_grasp_matrix(pos, normal)
        w = self.GWS(G_matrix, normal)
        gras_energy = grasp_error = w.norm(dim=-1).min(dim=-1, keepdim=True)[0]
        grasp_error = grasp_error.expand(-1, target_wrenches.shape[0])
        return gras_energy, grasp_error, contact_frame, contact_force
    