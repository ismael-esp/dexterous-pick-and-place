#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
"""Warp-lang based world collision functions are implemented as torch autograd functions."""

# Third Party
import torch
import warp as wp

wp.set_module_options({"fast_math": False})

# CuRobo
from curobo.util.warp_init import warp_support_sdf_struct

# Check version of warp and import the supported SDF function.
if warp_support_sdf_struct():
    # Local Folder
    from .warp_sdf_fns import get_closest_pt_batch_env, get_swept_closest_pt_batch_env
else:
    # Local Folder
    from .warp_sdf_fns_deprecated import get_closest_pt_batch_env, get_swept_closest_pt_batch_env



@wp.func
def finite_diff_vec3(
    out_grad: wp.array(dtype=wp.float32),
    new_v: wp.vec3,
    raw_v: wp.vec3,
    pert: wp.vec3,
    tid: wp.int32,
    group_num: wp.int32,
    bias_num: wp.int32,
):
    finite_diff_float(out_grad, new_v[0], raw_v[0], pert, tid, group_num, bias_num + 0)
    finite_diff_float(out_grad, new_v[1], raw_v[1], pert, tid, group_num, bias_num + 4)
    finite_diff_float(out_grad, new_v[2], raw_v[2], pert, tid, group_num, bias_num + 8)
    
    
@wp.func
def finite_diff_float(
    out_grad: wp.array(dtype=wp.float32),
    new_v: wp.float32,
    raw_v: wp.float32,
    pert: wp.vec3,
    tid: wp.int32,
    group_num: wp.int32,
    bias_num: wp.int32,    
):
    if pert[0] != 0.0: 
        out_grad[tid * group_num + bias_num] = (new_v - raw_v) / pert[0]
    else:
        out_grad[tid * group_num + bias_num] = 0.0
        
    if pert[1] != 0.0: 
        out_grad[tid * group_num + bias_num + 1] = (new_v - raw_v) / pert[1]
    else:
        out_grad[tid * group_num + bias_num + 1] = 0.0
        
    if pert[2] != 0.0: 
        out_grad[tid * group_num + bias_num + 2] = (new_v - raw_v) / pert[2]
    else:
        out_grad[tid * group_num + bias_num + 2] = 0.0
                
    
@wp.func
def query_one_point_pdn(
  pt: wp.vec3,
  pt_rad: wp.float32,
  in_mesh: wp.uint64,
  obj_w_pose_t: wp.transformf,
):  
    max_dist = float(1000.0)
    sign = float(0.0)
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    
    wp.mesh_query_point(in_mesh, pt, max_dist, sign, face_index, face_u, face_v)
    position = wp.vec3(0., 0., 0.)
    dis_length = float(0.)
    normal = wp.vec3(0., 0., 0.)
    epsilon = float(1e-4)

    # wp.mesh_eval_point_pdn(in_mesh, face_index, face_u, face_v, pt, sign, epsilon, position, dis_length, normal)
    position = wp.mesh_eval_position(in_mesh, face_index, face_u, face_v)
    dis_length = wp.length(pt - position)
    normal = - wp.mesh_eval_face_normal(in_mesh, face_index)
    
    position = wp.transform_point(obj_w_pose_t, position)
    distance = dis_length * sign - pt_rad
    normal = wp.transform_vector(obj_w_pose_t, normal)
    return position, distance, normal


@wp.kernel
def get_sphere_pdn_w_perturb(
    pt: wp.array(dtype=wp.float32, ndim=2),
    perturb: wp.array(dtype=wp.vec3), 
    raw_position: wp.array(dtype=wp.vec3),
    raw_distance: wp.array(dtype=wp.float32),
    raw_normal: wp.array(dtype=wp.vec3),
    debug_posi: wp.array(dtype=wp.vec3),
    debug_normal: wp.array(dtype=wp.vec3),
    out_pos_grad: wp.array(dtype=wp.float32),  # this stores the output position gradient
    out_dist_grad: wp.array(dtype=wp.float32),  # this stores the output distance gradient
    out_normal_grad: wp.array(dtype=wp.float32),  # this stores the output normal gradient
    mesh: wp.array(dtype=wp.uint64),            # only target mesh id
    mesh_pose: wp.array(dtype=wp.float32),
    env_query_idx: wp.array(dtype=wp.int32),
    n_points: wp.int32,
    n_perturb: wp.int32,
):
    # we launch nspheres kernels
    # compute gradient here and return
    # distance is negative outside and positive inside
    tid = wp.tid()
    
    bn_idx = tid / n_perturb
    pert_idx = tid % n_perturb
    b_idx = tid / (n_perturb * n_points)
    
    in_sphere = pt[bn_idx]
    in_pt = wp.vec3(in_sphere[0], in_sphere[1], in_sphere[2])
    in_rad = in_sphere[3]
    in_pert = perturb[tid]
    in_pos = raw_position[bn_idx]
    in_dist = raw_distance[bn_idx]
    in_normal = raw_normal[bn_idx]
    
    env_id = env_query_idx[b_idx]
    in_mesh = mesh[env_id]
    
    # read object pose:
    obj_position = wp.vec3(
        mesh_pose[env_id * 8 + 0],
        mesh_pose[env_id * 8 + 1],
        mesh_pose[env_id * 8 + 2],
    )
    obj_quat = wp.quaternion(
        mesh_pose[env_id * 8 + 4],
        mesh_pose[env_id * 8 + 5],
        mesh_pose[env_id * 8 + 6],
        mesh_pose[env_id * 8 + 3],
    )
    obj_w_pose = wp.transform(obj_position, obj_quat)
    obj_w_pose_t = wp.transform_inverse(obj_w_pose)
    local_pt = wp.transform_point(obj_w_pose, in_pt+in_pert)
    perturbed_pos, perturbed_dist, perturbed_normal = query_one_point_pdn(local_pt, in_rad, in_mesh, obj_w_pose_t)
    
    debug_posi[tid] = perturbed_pos
    debug_normal[tid] = perturbed_normal
    
    # Trick: clamp derivative to help balance gws cost and distance cost
    # can alleviate but cannot solve 
    pert_length = wp.max(wp.length(perturbed_pos - in_pos), 1e-7)
    perturbed_pos = in_pos + (perturbed_pos - in_pos) / pert_length * wp.min(pert_length, 0.01)

    pert_n_length = wp.max(wp.length(perturbed_normal - in_normal), 1e-7)
    perturbed_normal = in_normal + (perturbed_normal - in_normal) / pert_n_length * wp.min(pert_n_length, 0.1)
    perturbed_normal = wp.normalize(perturbed_normal)

    finite_diff_vec3(out_pos_grad, perturbed_pos, in_pos, in_pert, tid, 12, 0)
    finite_diff_float(out_dist_grad, perturbed_dist, in_dist, in_pert, tid, 4, 0)
    finite_diff_vec3(out_normal_grad, perturbed_normal, in_normal, in_pert, tid, 12, 0)


@wp.kernel
def get_sphere_pdn_wo_perturb(
    pt: wp.array(dtype=wp.float32, ndim=2),
    out_position: wp.array(dtype=wp.vec3),  # this stores the output position
    out_distance: wp.array(dtype=wp.float32),  # this stores the output distance
    out_normal: wp.array(dtype=wp.vec3),  # this stores the output normal
    mesh: wp.array(dtype=wp.uint64),            # only target mesh id
    mesh_pose: wp.array(dtype=wp.float32),
    env_query_idx: wp.array(dtype=wp.int32),   
    n_points: wp.int32,
):
    # we launch nspheres kernels
    # compute gradient here and return
    # distance is negative outside and positive inside
    tid = wp.tid()
    b_idx = tid / n_points
    
    in_sphere = pt[tid]
    in_pt = wp.vec3(in_sphere[0], in_sphere[1], in_sphere[2])
    in_rad = in_sphere[3]
    env_id = env_query_idx[b_idx]
    in_mesh = mesh[env_id]
    
    # read object pose:
    obj_position = wp.vec3(
        mesh_pose[env_id * 8 + 0],
        mesh_pose[env_id * 8 + 1],
        mesh_pose[env_id * 8 + 2],
    )
    obj_quat = wp.quaternion(
        mesh_pose[env_id * 8 + 4],
        mesh_pose[env_id * 8 + 5],
        mesh_pose[env_id * 8 + 6],
        mesh_pose[env_id * 8 + 3],
    )
    obj_w_pose = wp.transform(obj_position, obj_quat)
    obj_w_pose_t = wp.transform_inverse(obj_w_pose)
    local_pt = wp.transform_point(obj_w_pose, in_pt)
    
    p, d, n = query_one_point_pdn(local_pt, in_rad, in_mesh, obj_w_pose_t)
    out_position[tid] = p 
    out_distance[tid] = d
    out_normal[tid] = n


class ContactPDNMeshWarpPy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query_info, # sphere or mesh pose
        perturb,
        position,
        distance,
        normal,
        debug_posi, 
        debug_normal,
        grad_pos,
        grad_dist,
        grad_normal,
        mesh_idx,
        mesh_pose_inverse,
        env_query_idx, 
    ):
        b, h, n_points, _ = query_info.shape
        n_perturb = perturb.shape[-2]

        # launch
        wp.launch(
            kernel=get_sphere_pdn_wo_perturb,
            dim=b * h * n_points,
            inputs=[
                wp.from_torch(query_info.detach().view(-1, 4), dtype=wp.float32),
                wp.from_torch(position.view(-1, 3), dtype=wp.vec3),
                wp.from_torch(distance.view(-1), dtype=wp.float32),
                wp.from_torch(normal.view(-1, 3), dtype=wp.vec3),
                wp.from_torch(mesh_idx.view(-1), dtype=wp.uint64),
                wp.from_torch(mesh_pose_inverse.view(-1), dtype=wp.float32),
                wp.from_torch(env_query_idx.view(-1), dtype=wp.int32),
                h*n_points,
            ],
            stream=wp.stream_from_torch(query_info.device),
        )

        if query_info.requires_grad:
            wp.launch(
                kernel=get_sphere_pdn_w_perturb,
                dim=b * h * n_points * n_perturb,
                inputs=[
                    wp.from_torch(query_info.detach().view(-1, 4), dtype=wp.float32),
                    wp.from_torch(perturb.view(-1, 3), dtype=wp.vec3),
                    wp.from_torch(position.view(-1, 3), dtype=wp.vec3),
                    wp.from_torch(distance.view(-1)),
                    wp.from_torch(normal.view(-1, 3), dtype=wp.vec3),
                    wp.from_torch(debug_posi.view(-1, 3), dtype=wp.vec3),
                    wp.from_torch(debug_normal.view(-1, 3), dtype=wp.vec3),
                    wp.from_torch(grad_pos.view(-1)),
                    wp.from_torch(grad_dist.view(-1)),
                    wp.from_torch(grad_normal.view(-1)),
                    wp.from_torch(mesh_idx.view(-1), dtype=wp.uint64),
                    wp.from_torch(mesh_pose_inverse.view(-1), dtype=wp.float32),
                    wp.from_torch(env_query_idx.view(-1), dtype=wp.int32),
                    h*n_points,
                    n_perturb,
                ],
                stream=wp.stream_from_torch(query_info.device),
            )
        ctx.requi_grad = query_info.requires_grad
        ctx.save_for_backward(grad_pos, grad_dist, grad_normal)
        return position, distance, normal, debug_posi, debug_normal

    @staticmethod
    def backward(ctx, p_grad_in, d_grad_in, n_grad_in, debug1, debug2):
        gp, gd, gn, = ctx.saved_tensors
        if ctx.requi_grad:
            sphere_grad = (p_grad_in.unsqueeze(-2) @ gp.mean(dim=-3)).squeeze(-2) + \
                (d_grad_in.unsqueeze(-1) * gd.mean(dim=-2)).squeeze(-2) + \
                (n_grad_in.unsqueeze(-2) @ gn.mean(dim=-3)).squeeze(-2)
        else:
            sphere_grad = None 
        return sphere_grad, None, None, None, None, None, None, None, None, None, None, None, None


class SdfMeshWarpPy(torch.autograd.Function):
    """Pytorch autograd function for computing signed distance between spheres and meshes."""

    @staticmethod
    def forward(
        ctx,
        query_spheres,
        out_cost,
        out_grad,
        sparsity_idx,
        weight,
        activation_distance,
        mesh_idx,
        mesh_pose_inverse,
        mesh_enable,
        n_env_mesh,
        max_dist,
        mesh_allow_contact=None,
        env_query_idx=None,
        contact_distance=None,
        return_loss=False,
        compute_esdf=False,
    ):
        b, h, n, _ = query_spheres.shape
        use_batch_env = True
        if env_query_idx is None:
            use_batch_env = False
            env_query_idx = n_env_mesh
        requires_grad = query_spheres.requires_grad
        if mesh_allow_contact is None:
            contact_flag = False
            mesh_allow_contact = mesh_idx   # place holder
        else:
            contact_flag = True
            if len(contact_distance) != n:
                print('wrong contact distance shape!')
                exit(1)
        wp.launch(
            kernel=get_closest_pt_batch_env,
            dim=b * h * n,
            inputs=[
                wp.from_torch(query_spheres.detach().view(-1, 4), dtype=wp.vec4),
                wp.from_torch(out_cost.view(-1)),
                wp.from_torch(out_grad.view(-1), dtype=wp.float32),
                wp.from_torch(sparsity_idx.view(-1), dtype=wp.uint8),
                wp.from_torch(weight),
                wp.from_torch(activation_distance),
                wp.from_torch(mesh_idx.view(-1), dtype=wp.uint64),
                wp.from_torch(mesh_pose_inverse.view(-1), dtype=wp.float32),
                wp.from_torch(mesh_enable.view(-1), dtype=wp.uint8),
                wp.from_torch(mesh_allow_contact.view(-1), dtype=wp.uint64),
                wp.from_torch(n_env_mesh.view(-1), dtype=wp.int32),
                wp.from_torch(max_dist, dtype=wp.float32),
                requires_grad,
                b,
                h,
                n,
                mesh_idx.shape[1],
                wp.from_torch(env_query_idx.view(-1), dtype=wp.int32),
                use_batch_env,
                compute_esdf,
                wp.from_torch(contact_distance, dtype=wp.float32),
                contact_flag,
            ],
            stream=wp.stream_from_torch(query_spheres.device),
        )
        ctx.return_loss = return_loss
        ctx.save_for_backward(out_grad)
        return out_cost

    @staticmethod
    def backward(ctx, grad_output):
        grad_sph = None
        if ctx.needs_input_grad[0]:
            (r,) = ctx.saved_tensors
            grad_sph = r
            if ctx.return_loss:
                grad_sph = r * grad_output.unsqueeze(-1)
        return (
            grad_sph,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None
        )


class SweptSdfMeshWarpPy(torch.autograd.Function):
    """Compute signed distance between trajectory of spheres and meshes."""

    @staticmethod
    def forward(
        ctx,
        query_spheres,
        out_cost,
        out_grad,
        sparsity_idx,
        weight,
        activation_distance,
        speed_dt,
        mesh_idx,
        mesh_pose_inverse,
        mesh_enable,
        n_env_mesh,
        max_dist,
        mesh_allow_contact=None,
        contact_distance=None, 
        sweep_steps=1,
        enable_speed_metric=False,
        env_query_idx=None,
        return_loss=False,
    ):
        b, h, n, _ = query_spheres.shape
        use_batch_env = True
        if env_query_idx is None:
            use_batch_env = False
            env_query_idx = n_env_mesh
        requires_grad = query_spheres.requires_grad
        if mesh_allow_contact is None:
            contact_flag = False
            mesh_allow_contact = mesh_idx   # place holder
        else:
            contact_flag = True
        wp.launch(
            kernel=get_swept_closest_pt_batch_env,
            dim=b * h * n,
            inputs=[
                wp.from_torch(query_spheres.detach().view(-1, 4), dtype=wp.vec4),
                wp.from_torch(out_cost.view(-1)),
                wp.from_torch(out_grad.view(-1), dtype=wp.float32),
                wp.from_torch(sparsity_idx.view(-1), dtype=wp.uint8),
                wp.from_torch(weight),
                wp.from_torch(activation_distance),
                wp.from_torch(speed_dt),
                wp.from_torch(mesh_idx.view(-1), dtype=wp.uint64),
                wp.from_torch(mesh_pose_inverse.view(-1), dtype=wp.float32),
                wp.from_torch(mesh_enable.view(-1), dtype=wp.uint8),
                wp.from_torch(mesh_allow_contact.view(-1), dtype=wp.uint64),
                wp.from_torch(n_env_mesh.view(-1), dtype=wp.int32),
                wp.from_torch(max_dist, dtype=wp.float32),
                requires_grad,
                b,
                h,
                n,
                mesh_idx.shape[1],
                sweep_steps,
                enable_speed_metric,
                wp.from_torch(env_query_idx.view(-1), dtype=wp.int32),
                use_batch_env,
                wp.from_torch(contact_distance, dtype=wp.float32),
                contact_flag
            ],
            stream=wp.stream_from_torch(query_spheres.device),
        )
        ctx.return_loss = return_loss
        ctx.save_for_backward(out_grad)
        return out_cost

    @staticmethod
    def backward(ctx, grad_output):
        grad_sph = None
        if ctx.needs_input_grad[0]:
            (r,) = ctx.saved_tensors
            grad_sph = r
            if ctx.return_loss:
                grad_sph = grad_sph * grad_output.unsqueeze(-1)
        return (
            grad_sph,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
