import torch 
import numpy as np
from typing import List 

from .base import GraspEnergyBase
from curobo.opt.qp import init_QP_solver
from curobo.util.tensor_util import normalize_vector


class QPEnergy(GraspEnergyBase):
    
    num_friction_approx: int
    k_lower: float
    pressure_constraints: List
    
    def __init__(self, k_lower, pressure_constraints, solver_type, miu_coef, obj_gravity_center, obj_obb_length, tensor_args, enable_density, solve_interval, **args):
        super().__init__(miu_coef, obj_gravity_center, obj_obb_length, tensor_args, enable_density)
        
        # parameters for qp
        if self.miu_coef[1] > 0:
            self.num_friction_approx = 4
        else:    
            self.num_friction_approx = 8
        self.k_lower = k_lower
        self.pressure_constraints = pressure_constraints
        self.qpsolver = init_QP_solver(solver_type)
        self.count = 0
        self.solve_interval = solve_interval
        self.qp_size = None
        return 
    
    def _init_LCQP_u(self, batch, num_points):
        num_f_strength = num_points * 3
        
        # Constraints: Ax <= B
        G_matrix = self.tensor_args.to_device(torch.zeros((batch, (self.num_friction_approx + 2) * num_points + 1 + len(self.pressure_constraints), num_f_strength + 1)))
        h_matrix = self.tensor_args.to_device(torch.zeros((batch, (self.num_friction_approx + 2) * num_points + 1 + len(self.pressure_constraints))))
        
        # friction - pressure * friction_coef <= 0
        A_end = self.num_friction_approx * num_points
        pressure_ind = range(0, num_f_strength, 3)
        friction1_ind = range(1, num_f_strength, 3)
        friction2_ind = range(2, num_f_strength, 3)
        select_ind = range(0, num_points)
        A_tmp = G_matrix[:, :A_end, :num_f_strength].view(-1, self.num_friction_approx, num_points, num_f_strength)
        A_tmp[..., select_ind, pressure_ind] = - self.miu_coef[0]
        aranged_angles = self.tensor_args.to_device(torch.arange(self.num_friction_approx) * 2 * np.pi / self.num_friction_approx).unsqueeze(-1)
        A_tmp[..., select_ind, friction1_ind] = torch.sin(aranged_angles)
        A_tmp[..., select_ind, friction2_ind] = torch.cos(aranged_angles)
        
        # - pressure <= 0 
        A_end2 = A_end + num_points
        select2_ind = range(A_end, A_end2)
        G_matrix[:, select2_ind, pressure_ind] = -1
        h_matrix[:, A_end:A_end2] = 0.01    # avoid solution being negative due to numerical issues
        
        # pressure <= 1
        A_end3 = A_end2 + num_points
        select3_ind = range(A_end2, A_end3)
        G_matrix[:, select3_ind, pressure_ind] = 1
        h_matrix[:, A_end2:A_end3] = 1
        
        # - k <= - self.k_lower
        G_matrix[:, A_end3, -1] = -1
        h_matrix[:, A_end3] = -self.k_lower
        
        for i, constraint in enumerate(self.pressure_constraints):
            press_lst = [pressure_ind[k] for k in constraint[0]]
            G_matrix[:, A_end3+1+i, press_lst] = -1
            h_matrix[:, A_end3+1+i] = - constraint[1]
        
        return G_matrix, None, h_matrix

    def _init_LCQP_lu(self, batch, num_points):
        num_f_strength = num_points * 3

        # Constraints: l <= Gx <= h
        # NOTE: init l is -inf, init h is 0
        G_matrix = self.tensor_args.to_device(torch.zeros((batch, (self.num_friction_approx + 1) * num_points + 1 + len(self.pressure_constraints), num_f_strength + 1)))
        l_matrix = self.tensor_args.to_device(torch.zeros((batch, (self.num_friction_approx + 1) * num_points + 1 + len(self.pressure_constraints))) - torch.inf)
        h_matrix = self.tensor_args.to_device(torch.zeros((batch, (self.num_friction_approx + 1) * num_points + 1 + len(self.pressure_constraints))))
        
        # friction - pressure * friction_coef <= 0
        A_end = self.num_friction_approx * num_points
        pressure_ind = range(0, num_f_strength, 3)
        friction1_ind = range(1, num_f_strength, 3)
        friction2_ind = range(2, num_f_strength, 3)
        select_ind = range(0, num_points)
        A_tmp = G_matrix[:, :A_end, :num_f_strength].view(-1, self.num_friction_approx, num_points, num_f_strength)
        A_tmp[..., select_ind, pressure_ind] = - self.miu_coef[0]
        aranged_angles = self.tensor_args.to_device(torch.arange(self.num_friction_approx) * 2 * np.pi / self.num_friction_approx).unsqueeze(-1)
        A_tmp[..., select_ind, friction1_ind] = torch.sin(aranged_angles)
        A_tmp[..., select_ind, friction2_ind] = torch.cos(aranged_angles)
        
        # 0 <= pressure <= 1 
        A_end2 = A_end + num_points
        select2_ind = range(A_end, A_end2)
        G_matrix[:, select2_ind, pressure_ind] = 1
        l_matrix[:, A_end:A_end2] = 0.01    # avoid solution being negative due to numerical issues
        h_matrix[:, A_end:A_end2] = 1
        
        # - k <= - self.k_lower
        G_matrix[:, A_end2, -1] = -1
        h_matrix[:, A_end2] = -self.k_lower - 0.01
        l_matrix[:, A_end2] = -self.k_lower + 0.01
        
        for i, constraint in enumerate(self.pressure_constraints):
            press_lst = [pressure_ind[k] for k in constraint[0]]
            G_matrix[:, A_end2+1+i, press_lst] = -1
            h_matrix[:, A_end2+1+i] = - constraint[1]
        
        return G_matrix, l_matrix, h_matrix
    
    def _init_LCQP_lu_soft(self, batch, num_points):
        num_f_strength = num_points * 4

        # Constraints: l <= Gx <= h.
        # NOTE: init l is -inf, init h is 0
        G_matrix = self.tensor_args.to_device(torch.zeros((batch, (2*self.num_friction_approx + 1) * num_points + 1 + len(self.pressure_constraints), num_f_strength + 1)))
        l_matrix = self.tensor_args.to_device(torch.zeros((batch, (2*self.num_friction_approx + 1) * num_points + 1 + len(self.pressure_constraints))) - torch.inf)
        h_matrix = self.tensor_args.to_device(torch.zeros((batch, (2*self.num_friction_approx + 1) * num_points + 1 + len(self.pressure_constraints))))
        
        # friction - pressure * friction_coef <= 0
        A_end = 2*self.num_friction_approx * num_points
        pressure_ind = range(0, num_f_strength, 4)
        friction1_ind = range(1, num_f_strength, 4)
        friction2_ind = range(2, num_f_strength, 4)
        friction3_ind = range(3, num_f_strength, 4)
        select_ind = range(0, num_points)
        A_tmp = G_matrix[:, :A_end, :num_f_strength].view(-1, self.num_friction_approx, 2, num_points, num_f_strength)
        A_tmp[..., select_ind, pressure_ind] = - self.miu_coef[0]
        aranged_angles = self.tensor_args.to_device(torch.arange(self.num_friction_approx) * 2 * np.pi / self.num_friction_approx).view(-1,1,1)
        A_tmp[..., :, select_ind, friction1_ind] = torch.sin(aranged_angles)
        A_tmp[..., :, select_ind, friction2_ind] = torch.cos(aranged_angles)
        A_tmp[..., 0, select_ind, friction3_ind] = -1.0
        A_tmp[..., 1, select_ind, friction3_ind] = 1.0
        
        # 0 <= pressure <= 1 
        A_end2 = A_end + num_points
        select2_ind = range(A_end, A_end2)
        G_matrix[:, select2_ind, pressure_ind] = 1
        l_matrix[:, A_end:A_end2] = 0.001  # avoid solution being negative due to numerical issues
        h_matrix[:, A_end:A_end2] = 1
        
        # - k <= - self.k_lower
        G_matrix[:, A_end2, -1] = -1
        h_matrix[:, A_end2] = -self.k_lower
        
        for i, constraint in enumerate(self.pressure_constraints):
            press_lst = [pressure_ind[k] for k in constraint[0]]
            G_matrix[:, A_end2+1+i, press_lst] = -1
            h_matrix[:, A_end2+1+i] = - constraint[1]
        
        return G_matrix, l_matrix, h_matrix
    
    def init_LCQP(self, batch_num, point_num, wrench_num):
        prob_size = torch.Size((batch_num, point_num, wrench_num))
        if self.qp_size == prob_size:
            return 
        else:
            self.qp_size = prob_size
            
        if self.qpsolver.glh_type == 'gh':
            G_matrix, l_matrix, h_matrix = self._init_LCQP_u(batch_num*wrench_num, point_num)
        elif self.qpsolver.glh_type == 'glh':
            if self.miu_coef[1] > 0:
                G_matrix, l_matrix, h_matrix = self._init_LCQP_lu_soft(batch_num*wrench_num, point_num)
            else:
                G_matrix, l_matrix, h_matrix = self._init_LCQP_lu(batch_num*wrench_num, point_num)
        else:
            raise NotImplementedError
        self.qpsolver.init_problem(G_matrix, l_matrix, h_matrix)
        return 
    
    def construct_Q_matrix(self, grasp_matrix, target_wrenches):
        batch_num, point_num = grasp_matrix.shape[:2]
        grasp_matrix = grasp_matrix.transpose(-3, -2).reshape(batch_num, 6, -1)  # [b, n, 6, 4] -> [b, 6, n, 4] -> [b, 6, 4n]
        grasp_matrix = grasp_matrix.repeat_interleave(target_wrenches.shape[0], dim=0) # [b, 6, 4n] -> [bm, 6, 4n]
        repeated_target_wrench = target_wrenches.repeat(batch_num, 1, 1)    # [m, 6, 1] -> [bm, 6, 1]
        grasp_matrix_with_target = torch.cat([grasp_matrix, - repeated_target_wrench], dim=-1) # [bm, 6, 4n+1]
        Q_matrix = grasp_matrix_with_target.transpose(-2, -1) @ grasp_matrix_with_target  # [bm, 4n+1, 4n+1]
        Q_matrix2 = Q_matrix.clone()
        
        # penalize large pressure to encourage more fingers to apply force
        if self.miu_coef[1] > 0:
            press_ind = range(0, point_num*4, 4)
        else:
            press_ind = range(0, point_num*3, 3)
        Q_matrix[:, press_ind, press_ind] += 0.01
        return Q_matrix, Q_matrix2, grasp_matrix_with_target
    
    def forward(self, pos, normal, target_wrenches):
        batch, n_points = pos.shape[:-1]

        # start = time.time()
        self.init_LCQP(batch, n_points, target_wrenches.shape[0])
        grasp_matrix, contact_frame, _ = self.construct_grasp_matrix(pos, normal)
        Q_matrix, Q_matrix2, semi_Q_matrix = self.construct_Q_matrix(grasp_matrix, target_wrenches)
        if self.count % self.solve_interval == 0:
            self.solution = self.qpsolver.solve(Q_matrix, semi_Q_matrix)
        # print(f'Solving {solution.shape[0]} QPs in {time.time() - start} seconds')
        self.count +=1 
        # wrench_error = (semi_Q_matrix @ self.solution.unsqueeze(-1)).reshape(-1,5,6)
        # print(self.count, wrench_error.norm(dim=-1)[2], (wrench_error.norm(dim=-1).max(dim=-1)[0] < 0.1).sum())
        grasp_energy = (self.solution.unsqueeze(-2) @ Q_matrix2 @ self.solution.unsqueeze(-1)).view(batch, -1).clamp(min=0.0)
        grasp_error = grasp_energy ** 0.5
        contact_force = self.solution[..., :-1].reshape(batch, target_wrenches.shape[0], n_points, -1)
            
        return grasp_energy, grasp_error, contact_frame, contact_force
    
    def reset(self, gravity_center, obb_length):
        super().reset(gravity_center, obb_length)
        self.count = 0
        
        
class QPBaseEnergy(QPEnergy):
    
    def construct_Q_matrix(self, grasp_matrix, target_wrenches):
        grasp_matrix = grasp_matrix.transpose(-3, -2).reshape(grasp_matrix.shape[0], 6, -1)  # [b, n, 6, 4] -> [b, 6, n, 4] -> [b, 6, 4n]
        grasp_matrix = grasp_matrix.repeat_interleave(target_wrenches.shape[0], dim=0) # [b, 6, 4n] -> [bm, 6, 4n]
        grasp_matrix_with_target = torch.cat([grasp_matrix, grasp_matrix[..., -1:]*0], dim=-1) # [bm, 6, 4n+1]
        Q_matrix = grasp_matrix_with_target.transpose(-2, -1) @ grasp_matrix_with_target  # [bm, 4n+1, 4n+1]
        return Q_matrix, Q_matrix, grasp_matrix_with_target
    