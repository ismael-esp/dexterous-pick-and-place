from .base import GraspEnergyBase
import torch
import numpy as np
import scipy

class CHQ1Energy(GraspEnergyBase):
    '''
    Non-differentiable! Only used for evaluation!
    '''
    
    def __init__(self, miu_coef, obj_gravity_center, obj_obb_length, tensor_args, enable_density, **args):
        super().__init__(miu_coef, obj_gravity_center, obj_obb_length, tensor_args, enable_density)
        self.num_friction_approx = 8
        return 
    
    @torch.no_grad()
    def forward(self, pos, normal, target_wrenches):
        batch = pos.shape[0]
        grasp_matrix, contact_frame, contact_force = self.construct_grasp_matrix(pos, normal)
        aranged_angles = self.tensor_args.to_device(torch.arange(self.num_friction_approx) * 2 * np.pi / self.num_friction_approx).unsqueeze(-1)
        f = torch.cat([aranged_angles*0+1, self.miu_coef[0]*torch.sin(aranged_angles), self.miu_coef[0]*torch.cos(aranged_angles)], dim=-1)
        corner_point = (f @ grasp_matrix.transpose(-1,-2)).view(batch, -1, 6)
        corner_point_cpu = corner_point.cpu().numpy()
        corner_point_cpu = np.concatenate([corner_point_cpu, corner_point_cpu[:,0:1,:]*0.], axis=1)
        grasp_error = []
        for i in range(batch):
            q1 = np.array([1], dtype=np.float32)
            try:
                wrench_space = scipy.spatial.ConvexHull(corner_point_cpu[i])
                for equation in wrench_space.equations:
                    q1 = np.minimum(q1, np.abs(equation[6]) / np.linalg.norm(equation[:6]))
            except:
                pass
            grasp_error.append(q1)
        grasp_energy = grasp_error = 2 - self.tensor_args.to_device(grasp_error)
        grasp_error = grasp_error.expand(-1, target_wrenches.shape[0])
        return grasp_energy, grasp_error, contact_frame, contact_force
    