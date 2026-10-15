
from typing import List

import torch
import numpy as np
import trimesh 
import torch_scatter
import time 

from coal_openmp_wrapper import batched_coal_distance
from curobo.geom.transform import pose_multiply
from curobo.geom.basic_transform import torch_quaternion_to_matrix
from curobo.util.tensor_util import normalize_vector
from curobo.util.logger import log_warn


def transform_sphere_param(sphere_param, pose_rot, pose_trans):
    new_sphere_center = (pose_rot @ sphere_param[..., :3, None]).squeeze(-1) + pose_trans
    new_sphere_param = torch.cat([new_sphere_center, sphere_param[..., 3:]], dim=-1)
    return new_sphere_param


def transform_obb_param(obb_param, pose_rot, pose_trans):
    rot_with_t = torch.cat([pose_rot, pose_trans.unsqueeze(-1)], dim=-1)
    rot_help = torch.tensor([0., 0., 0., 1.], device=rot_with_t.device).view([1]*(len(pose_rot.shape)-1)+[4]).expand_as(rot_with_t[..., 0:1, :])
    rot_44 = torch.cat([rot_with_t, rot_help], dim=-2)
    new_obb_transform = rot_44 @ obb_param[..., :16].view(obb_param.shape[:-1] + (4, 4))
    new_obb_param = torch.cat([new_obb_transform.view(new_obb_transform.shape[:-2] + (16,)), obb_param[..., 16:]], dim=-1)
    return new_obb_param


def sphere_OBB_distance(robot_sphere_param, obj_obb_param):
    obb_transform, obb_length = obj_obb_param[..., :16].view(obj_obb_param.shape[:-1] + (4, 4)), obj_obb_param[..., 16:]
    sphere_center, sphere_radius = robot_sphere_param[..., :-1], robot_sphere_param[..., -1]
    sphere_center_delta = obb_transform[..., :3, :3].transpose(-1,-2) @ (sphere_center - obb_transform[..., :3, 3]).unsqueeze(-1)
    d = torch.clamp(sphere_center_delta.squeeze(-1).abs() - obb_length, min=0.).norm(dim=-1) - sphere_radius
    return d


def getBoundingPrimitive(file_name: str, scale: float, primitive='obb'):
    tm = trimesh.load(file_name, force='mesh')
    tm.vertices *= scale
    if primitive == 'obb':
        obb_transform = torch.tensor(tm.bounding_box_oriented.primitive.transform)
        obb_length = torch.tensor(tm.bounding_box_oriented.primitive.extents)
        obb_param = torch.cat([obb_transform.view(-1), obb_length*0.5], dim=0)
        return obb_param
    elif primitive == 'sphere':
        sphere_center = torch.tensor(tm.bounding_sphere.primitive.transform[:3, 3])
        sphere_radius = torch.tensor([tm.bounding_sphere.primitive.radius])
        sphere_param = torch.cat([sphere_center, sphere_radius], dim=0)
        d = np.linalg.norm(tm.vertices - tm.vertices.mean(axis=0)[None]).max()
        return sphere_param
    else:
        raise NotImplementedError



def narrow_phase(obj_mesh_list, robot_mesh_list, obj_idx, robot_idx, obj_rot, obj_trans, robot_rot, robot_trans, narrow_idx):
    cases = len(obj_idx)
    dist_result = np.ones((cases)) * 100
    normal_result = np.zeros((cases, 3))
    cp1_result = np.zeros((cases, 3))
    cp2_result = np.zeros((cases, 3))
    obj_rot_cpu = obj_rot.reshape(cases, 9).cpu().numpy()
    obj_trans_cpu = obj_trans.reshape(cases, -1).cpu().numpy()

    robot_rot_cpu = robot_rot.reshape(cases, 9).detach().cpu().numpy()
    robot_trans_cpu = robot_trans.reshape(cases, -1).detach().cpu().numpy()
    obj_idx_cpu = obj_idx.cpu().numpy()
    robot_idx_cpu = robot_idx.cpu().numpy()
    narrow_lst = narrow_idx.cpu().numpy()
    
    # st = time.time()
    batched_coal_distance(
        obj_mesh_list, obj_idx_cpu, obj_rot_cpu, obj_trans_cpu,
        robot_mesh_list, robot_idx_cpu, robot_rot_cpu, robot_trans_cpu,
        narrow_lst, len(narrow_lst), dist_result, normal_result, cp1_result, cp2_result)
    # log_warn(f'time: {time.time() - st}; query num: {len(narrow_lst)}')
    
    dist_torch = torch.tensor(dist_result).to(device=robot_trans.device, dtype=robot_trans.dtype)
    normal_torch = torch.tensor(normal_result).to(device=robot_trans.device, dtype=robot_trans.dtype)
    closest_points_result = np.concatenate([cp1_result, cp2_result], axis=-1)
    closest_points_torch = torch.tensor(closest_points_result).to(device=robot_trans.device, dtype=robot_trans.dtype)
    return dist_torch, normal_torch, closest_points_torch


def batched_convex_convex(
    robot_pose_torch: torch.Tensor, 
    robot_offset_pose_torch: torch.Tensor,
    obj_pose_torch: torch.Tensor, 
    robot_mesh_list: List,
    obj_mesh_list: List, 
    robot_sphere_param_torch: torch.Tensor, 
    obj_obb_param_torch: torch.Tensor, 
    obj_scene_idx: torch.Tensor,
    env_idx: torch.Tensor,
    dist_upper_bound: torch.Tensor,
):

    # 1. align input shape
    scene_num = env_idx.max() + 1
    grasp_per_scene = int(env_idx.shape[0] // scene_num)
    robot_per_scene = robot_pose_torch.shape[-2]
    obj_all_scene = obj_scene_idx.shape[0]
    obj_idx = torch.arange(obj_all_scene)
    obj_idx = obj_idx.view(-1, 1, 1).repeat(1, grasp_per_scene, robot_per_scene)
    robot_idx = torch.arange(robot_per_scene)
    robot_idx = robot_idx.view(1,1,-1).repeat(obj_all_scene, grasp_per_scene, 1)
    
    # all below tensor is aligned with shape [grasp_per_scene, obj_all_scene, robot_per_scene, -1]
    obj_obb_param = obj_obb_param_torch[obj_idx]
    robot_sphere_param = robot_sphere_param_torch[robot_idx]
    obj_pose = obj_pose_torch[obj_idx]
    robot_pose = robot_pose_torch.view(scene_num, grasp_per_scene, robot_per_scene, -1)[obj_scene_idx, ...]
    robot_offset_pose = robot_offset_pose_torch[robot_idx]
    robot_obj_dist = dist_upper_bound.view(scene_num, grasp_per_scene, robot_per_scene)[obj_scene_idx]
    
    robot_trans, robot_quat = pose_multiply(robot_pose[..., :3].view(-1, 3), robot_pose[..., 3:7].view(-1, 4), robot_offset_pose[..., :3].view(-1, 3), robot_offset_pose[..., 3:7].view(-1, 4))
    robot_trans, robot_quat = robot_trans.view(obj_all_scene, grasp_per_scene, robot_per_scene, -1), robot_quat.view(obj_all_scene, grasp_per_scene, robot_per_scene, -1)
    robot_rot = torch_quaternion_to_matrix(robot_quat)
    obj_rot, obj_trans = torch_quaternion_to_matrix(obj_pose[..., 3:7]), obj_pose[..., :3]
    
    # 2. broad phase, torch on CUDA
    posed_sphere_param = transform_sphere_param(robot_sphere_param.float(), robot_rot, robot_trans)
    posed_obb_param = transform_obb_param(obj_obb_param.float(), obj_rot, obj_trans)
    
    broad_dist = sphere_OBB_distance(posed_sphere_param, posed_obb_param)
    narrow_idx = torch.where(broad_dist.view(-1) < torch.clamp(robot_obj_dist.view(-1), min=0.0)+0.003)[0]
    # print('broad phase first 10 cases:', broad_dist[:10])
    
    # 3. narrow phase GJK, multiprocess on cpu
    # np.save('test.npy', {
    #     'obj_idx': obj_idx.view(-1), 
    #     'robot_idx': robot_idx.view(-1), 
    #     'obj_rot': obj_rot, 
    #     'obj_trans': obj_trans,
    #     'robot_rot': robot_rot, 
    #     'robot_trans': robot_trans, 
    #     'narrow_idx': narrow_idx
    # })
    # exit(1)
    dist_torch, normal_torch, closest_points_torch = narrow_phase(obj_mesh_list, robot_mesh_list,
                        obj_idx.view(-1), robot_idx.view(-1), 
                        obj_rot, obj_trans,
                        robot_rot, robot_trans, narrow_idx
                    )
    
    dist_torch = dist_torch.view(obj_all_scene, grasp_per_scene, robot_per_scene)
    normal_torch = normal_torch.view(obj_all_scene, grasp_per_scene, robot_per_scene, 3)
    closest_points_torch = closest_points_torch.view(obj_all_scene, grasp_per_scene, robot_per_scene, 6)
    dist_scene, scatter_index = torch_scatter.scatter_min(dist_torch, obj_scene_idx, dim=0)
    normal_scene = torch.gather(normal_torch, 0, scatter_index.unsqueeze(-1).expand(-1,-1,-1,3))
    closest_points_scene = torch.gather(closest_points_torch, 0, scatter_index.unsqueeze(-1).expand(-1,-1,-1,6))
    # print('narrow phase first 10: ', dist_torch.view(-1)[:10])
    if dist_scene.max() > 99 and dist_upper_bound.max() > 0:
        log_warn(f'GJK results are greater than given threshold! {dist_torch.argmax()}')
        # import pdb; pdb.set_trace()
    # print(dist_scene.max())
    # os.makedirs('debug', exist_ok=True)
    # for i in range(len(obj_scene_idx)):
    #     if obj_scene_idx[i] > 0:
    #         break
    #     pp = obj_mesh_list[i].points()
    #     obj_rot_cpu = obj_rot.reshape(-1, 3, 3).cpu().numpy()
    #     obj_trans_cpu = obj_trans.reshape(-1, 3).cpu().numpy()
    #     posed_obj = pp @ obj_rot_cpu[i].T + obj_trans_cpu[i]
    #     np.savetxt(f'debug/obj_mesh{i}.txt', posed_obj)
        
    
    # for i in range(5):
    #     pp = robot_mesh_list[i].points()
    #     obj_rot_cpu = robot_rot.reshape(-1, 3, 3).detach().cpu().numpy()
    #     obj_trans_cpu = robot_trans.reshape(-1, 3).detach().cpu().numpy()
    #     posed_obj = pp @ obj_rot_cpu[i].T + obj_trans_cpu[i]
    #     np.savetxt(f'debug/obot_mesh{i}.txt', posed_obj)
    
    return closest_points_scene.view(-1, 1, robot_per_scene, 6), dist_scene.view(-1, 1, robot_per_scene), normal_scene.view(-1, 1, robot_per_scene, 3)


class ContactPDNConvexMesh(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        robot_pose_torch: torch.Tensor, 
        robot_offset_pose_torch: torch.Tensor,
        obj_pose_torch: torch.Tensor, 
        robot_mesh_list: List,
        obj_mesh_list: List, 
        robot_sphere_param_torch: torch.Tensor, 
        obj_obb_param_torch: torch.Tensor, 
        perturb: torch.Tensor, 
        obj_scene_idx: torch.Tensor,
        env_idx: torch.Tensor,
        dist_upper_bound: torch.Tensor,
    ):
        points, distance, normal = batched_convex_convex(
                                    robot_pose_torch, robot_offset_pose_torch, obj_pose_torch,
                                    robot_mesh_list, obj_mesh_list, 
                                    robot_sphere_param_torch, obj_obb_param_torch, 
                                    obj_scene_idx, env_idx, dist_upper_bound
                                )
        perturb_num = perturb.shape[-2]
        if robot_pose_torch.requires_grad:
            perturb_permute = perturb.permute(-2,0,1,2,-1)
            perturb_permute_batched = perturb_permute.reshape((-1,)+perturb_permute.shape[2:])
            perturb_env_idx = env_idx.repeat(perturb_num, 1)
            perturb_robot_pose_torch = robot_pose_torch.repeat(perturb_num,1,1,1) + perturb_permute_batched
            perturb_robot_pose_torch[..., 3:7] = normalize_vector(perturb_robot_pose_torch[..., 3:7])
            
            pert_points, pert_distance, pert_normal = batched_convex_convex(
                                    perturb_robot_pose_torch, robot_offset_pose_torch, obj_pose_torch,
                                    robot_mesh_list, obj_mesh_list, 
                                    robot_sphere_param_torch, obj_obb_param_torch, 
                                    obj_scene_idx, perturb_env_idx
                                )
            debug_pos = pert_points.view((-1,)+points.shape).permute(1,2,3,0,-1)
            debug_normal = pert_normal.view((-1,)+normal.shape).permute(1,2,3,0,-1)
            grad_pos = ((pert_points.view((-1,)+points.shape) - points.unsqueeze(0)).unsqueeze(-1) / perturb_permute.unsqueeze(-2))
            grad_dist = ((pert_distance.view((-1,)+distance.shape) - distance.unsqueeze(0)).unsqueeze(-1) / perturb_permute)
            grad_normal = ((pert_normal.view((-1,)+normal.shape) - normal.unsqueeze(0)).unsqueeze(-1) / perturb_permute.unsqueeze(-2))
            grad_pos = torch.nan_to_num(grad_pos, posinf=0.0, neginf=0.0).mean(dim=0)
            grad_dist = torch.nan_to_num(grad_dist, posinf=0.0, neginf=0.0).mean(dim=0)
            grad_normal = torch.nan_to_num(grad_normal, posinf=0.0, neginf=0.0).mean(dim=0)
            ctx.save_for_backward(grad_pos, grad_dist, grad_normal)
        else:
            debug_pos = points.unsqueeze(-2).expand(-1,-1,-1,perturb_num,-1)
            debug_normal = normal.unsqueeze(-2).expand(-1,-1,-1,perturb_num,-1) 
        ctx.requi_grad = robot_pose_torch.requires_grad
        return points, distance, normal, debug_pos, debug_normal
    
    @staticmethod
    def backward(ctx, p_grad_in, d_grad_in, n_grad_in, debug_pos, debug_normal):
        gp, gd, gn, = ctx.saved_tensors
        if ctx.requi_grad:
            pose_grad = (p_grad_in.unsqueeze(-2) @ gp).squeeze(-2) + \
                (d_grad_in.unsqueeze(-1) * gd) + \
                (n_grad_in.unsqueeze(-2) @ gn).squeeze(-2)
        else:
            pose_grad = None 
        return pose_grad, None, None, None, None, None, None, None, None, None, None

@torch.no_grad()
def RobotSelfPenetration(
    robot_pose_torch: torch.Tensor, 
    robot_offset_pose_torch: torch.Tensor,
    robot_mesh_list: List,
    self_collision_link_mask: torch.Tensor,
):
    grasp_num, link_num = robot_pose_torch.shape[:2]
    robot_offset_pose = robot_offset_pose_torch.unsqueeze(0).expand(grasp_num, -1, -1).contiguous()
    robot_trans, robot_quat = pose_multiply(robot_pose_torch[..., :3].view(-1, 3), robot_pose_torch[..., 3:7].view(-1, 4), robot_offset_pose[..., :3].view(-1, 3), robot_offset_pose[..., 3:7].view(-1, 4))
    robot_trans, robot_quat = robot_trans.view(grasp_num, link_num, -1), robot_quat.view(grasp_num, link_num, -1)
    robot_rot = torch_quaternion_to_matrix(robot_quat)
    
    robot_idx1 = torch.arange(link_num).view(1,-1,1).expand(grasp_num,link_num,link_num).reshape(-1)
    robot_idx2 = torch.arange(link_num).view(1,1,-1).expand(grasp_num,link_num,link_num).reshape(-1)
    robot_trans1 = robot_trans.unsqueeze(-2).expand(-1,-1,link_num,-1)
    robot_trans2 = robot_trans.unsqueeze(-3).expand(-1,link_num,-1,-1)
    robot_rot1 = robot_rot.unsqueeze(-3).expand(-1,-1,link_num,-1,-1)
    robot_rot2 = robot_rot.unsqueeze(-4).expand(-1,link_num,-1,-1,-1)
    narrow_idx = torch.nonzero(self_collision_link_mask[robot_idx1, robot_idx2]).squeeze(-1)

    dist_torch, _, _ = narrow_phase(robot_mesh_list, robot_mesh_list,
                    robot_idx1, robot_idx2, 
                    robot_rot1, robot_trans1,
                    robot_rot2, robot_trans2, narrow_idx
                )
    dist_torch = dist_torch.reshape(grasp_num, -1) 
    self_pene = - dist_torch * (dist_torch < 0) # <0 means penetration
    max_self_pene = self_pene.max(dim=-1)[0]
    return max_self_pene
