from .base import GraspEnergyBase
import torch 

class DFCEnergy(GraspEnergyBase):
    
    def __init__(self, miu_coef, obj_gravity_center, obj_obb_length, tensor_args, enable_density, **args):
        super().__init__(miu_coef, obj_gravity_center, obj_obb_length, tensor_args, enable_density)
        return 

    def forward(self, pos, normal, target_wrenches):
        grasp_matrix, contact_frame, contact_force = self.construct_grasp_matrix(pos, normal)
        if self.enable_density:
            density = self.estimate_density(normal)
            grasp_energy = grasp_error = (grasp_matrix[..., 0] * density.unsqueeze(-1)).sum(dim=1).norm(dim=-1, keepdim=True)
        else:
            grasp_energy = grasp_error = grasp_matrix[..., 0].sum(dim=1).norm(dim=-1, keepdim=True)
        grasp_error = grasp_error.expand(-1, target_wrenches.shape[0])
        return grasp_energy, grasp_error, contact_frame, contact_force
