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
# Standard Library
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union
import math 
from copy import deepcopy

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelState
from curobo.geom.sdf.utils import create_collision_checker
from curobo.geom.sdf.world import CollisionCheckerType, WorldCollision, WorldCollisionConfig
from curobo.geom.types import WorldConfig
from curobo.opt.newton.lbfgs import LBFGSOpt, LBFGSOptConfig
from curobo.opt.newton.newton_base import NewtonOptBase, NewtonOptConfig
from curobo.opt.particle.parallel_es import ParallelES, ParallelESConfig
from curobo.opt.particle.parallel_mppi import ParallelMPPI, ParallelMPPIConfig
from curobo.rollout.arm_reacher import ArmReacher, ArmReacherConfig
from curobo.rollout.rollout_base import Goal, RolloutBase, RolloutMetrics
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.types.tensor import T_BDOF, T_BValue_bool, T_BValue_float
from curobo.util.logger import log_error, log_warn
from curobo.util.sample_grasp import HeurGraspSeedGenerator
from curobo.util_file import (
    get_robot_configs_path,
    get_task_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.types import ReacherSolveState, ReacherSolveType
from curobo.wrap.wrap_base import WrapBase, WrapConfig, WrapResult
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

def compare_element(x, y):
    if isinstance(x, dict):
        if not isinstance(y, dict):
            return False, x, y
        for k in x.keys():
            if k not in y:
                return False, x, y 
            retf, ret1, ret2 = compare_element(x[k], y[k])
            if not retf:
                return False, x[k], y[k]  
        return True, None, None 
    elif x != y:
        return False, x, y 
    else:
        return True, None, None 
    

@dataclass
class GraspSolverConfig:
    robot_config: RobotConfig
    robot_file: str
    solver: WrapBase
    num_seeds: int
    grasp_threshold: float
    distance_threshold: float
    rollout_fn: ArmReacher
    q_sample_gen: HeurGraspSeedGenerator 
    grasp_nn_seeder: Optional[str] = None
    world_coll_checker: Optional[WorldCollision] = None
    sample_rejection_ratio: int = 50
    tensor_args: TensorDeviceType = TensorDeviceType()
    ik_solver: IKSolver = None
        
    @staticmethod
    @profiler.record_function("grasp_solver/load_from_robot_config")
    def load_from_robot_config(
        robot_cfg: RobotConfig = None,
        world_model: Optional[
            Union[Union[List[Dict], List[WorldConfig]], Union[Dict, WorldConfig]]
        ] = None,
        manip_name_list: List = None,
        obj_gravity_center: List = None,
        obj_obb_length: List = None,
        metric_type: str = None, 
        no_grasp_cfg: bool = False,
        tensor_args: TensorDeviceType = TensorDeviceType(),
        grasp_threshold: float = 0.001,
        distance_threshold: float = 0.01,
        world_coll_checker = None,
        manip_config_data: dict = None,
        calculate_penetration: str = None,
        use_cuda_graph: Optional[bool] = None,
        grad_iters: Optional[int] = None,
        use_particle_opt: bool = False,
        collision_checker_type: Optional[CollisionCheckerType] = None,
        sync_cuda_time: Optional[bool] = None,
        use_gradient_descent: bool = True,
        use_es: Optional[bool] = None,
        es_learning_rate: Optional[float] = 0.1,
        use_fixed_samples: Optional[bool] = None,
        store_debug: bool = False,
        regularization: bool = True,
        collision_activation_distance: Optional[float] = None,
        ik_solver: IKSolver = None,
    ):
        # use default values, disable environment collision checking
        base_config_data = load_yaml(join_path(get_task_configs_path(), manip_config_data['base_cfg_file']))
        config_data = load_yaml(join_path(get_task_configs_path(), manip_config_data['particle_file']))
        grad_config_data = load_yaml(join_path(get_task_configs_path(), manip_config_data['gradient_file']))
        robot_config_data = load_yaml(join_path(get_robot_configs_path(), manip_config_data['robot_file']))["robot_cfg"]
        
        if robot_cfg is None:
            contact_robot_links = [i.split('/')[0] for i in manip_config_data['grasp_contact_strategy']['contact_points_name']]
            if calculate_penetration is None:
                robot_config_data['kinematics']['contact_mesh_names'] = contact_robot_links
            elif calculate_penetration == 'all_links':
                robot_config_data['kinematics']['contact_mesh_names'] = [link for link in robot_config_data['kinematics']['mesh_link_names']]
            elif calculate_penetration == 'non_contact_links':
                robot_config_data['kinematics']['contact_mesh_names'] = [link for link in robot_config_data['kinematics']['mesh_link_names'] if link not in contact_robot_links]
            else:
                raise NotImplementedError
                    
            robot_cfg = RobotConfig.from_dict(robot_config_data)
        
        if collision_checker_type is not None:
            base_config_data["world_collision_checker_cfg"]["checker_type"] = collision_checker_type
        if not regularization:
            base_config_data["convergence"]["null_space_cfg"]["weight"] = 0.0
            base_config_data["convergence"]["cspace_cfg"]["weight"] = 0.0
            config_data["cost"]["bound_cfg"]["null_space_weight"] = 0.0
            grad_config_data["cost"]["bound_cfg"]["null_space_weight"] = 0.0

        if world_coll_checker is None and world_model is not None:
            base_config_data["world_collision_checker_cfg"]["contact_obj_names"] = manip_name_list
            base_config_data["world_collision_checker_cfg"]["contact_obj_sample_params"] = manip_config_data['seeder_cfg']['obj_sample']
            base_config_data["world_collision_checker_cfg"]["contact_obj_mesh_loading"] = True
            world_cfg = WorldCollisionConfig.load_from_dict(
                base_config_data["world_collision_checker_cfg"], world_model, tensor_args
            )
            world_coll_checker = create_collision_checker(world_cfg)
        
        if collision_activation_distance is not None:
            config_data["cost"]["primitive_collision_cfg"][
                "activation_distance"
            ] = collision_activation_distance
            grad_config_data["cost"]["primitive_collision_cfg"][
                "activation_distance"
            ] = collision_activation_distance
        
        if "grasp_contact_strategy" in manip_config_data:
            config_data["cost"]["contact_strategy"] = manip_config_data["grasp_contact_strategy"]
            grad_config_data["cost"]["contact_strategy"] = manip_config_data["grasp_contact_strategy"]
            base_config_data["constraint"]["contact_strategy"] = manip_config_data["grasp_contact_strategy"]
            base_config_data["convergence"]["contact_strategy"] = deepcopy(manip_config_data["grasp_contact_strategy"])
            base_config_data["convergence"]["contact_strategy"]["distance"][0] = 0
            base_config_data["convergence"]["contact_strategy"]["contact_query_mode"][0] = manip_config_data["grasp_contact_strategy"]["contact_query_mode"][-1]

        manip_config_data["grasp_cfg"]["ge_param"]["obj_gravity_center"] = obj_gravity_center
        manip_config_data["grasp_cfg"]["ge_param"]["obj_obb_length"] = obj_obb_length
        if metric_type is not None:
            manip_config_data["grasp_cfg"]["ge_param"]["type"] = metric_type
        
        if store_debug:
            use_cuda_graph = False
            grad_config_data["lbfgs"]["store_debug"] = store_debug
            config_data["mppi"]["store_debug"] = store_debug
        if use_cuda_graph is not None:
            config_data["mppi"]["use_cuda_graph"] = use_cuda_graph
            grad_config_data["lbfgs"]["use_cuda_graph"] = use_cuda_graph
        if use_fixed_samples is not None:
            config_data["mppi"]["sample_params"]["fixed_samples"] = use_fixed_samples

        if grad_iters is not None:
            grad_config_data["lbfgs"]["n_iters"] = grad_iters
        if 'grasp_cfg' in grad_config_data['cost'].keys() and not no_grasp_cfg:
            grad_config_data['cost']['grasp_cfg'] = {**grad_config_data['cost']['grasp_cfg'], **manip_config_data['grasp_cfg']}
            config_data['cost']['grasp_cfg'] = {**config_data['cost']['grasp_cfg'], **manip_config_data['grasp_cfg']}
            base_config_data["convergence"]['grasp_cfg'] = {**base_config_data['convergence']['grasp_cfg'], **manip_config_data['grasp_cfg']}
        elif no_grasp_cfg:
            grad_config_data['cost'].pop('grasp_cfg')
            config_data['cost'].pop('grasp_cfg')
            base_config_data['convergence'].pop('grasp_cfg')

        config_data["mppi"]["n_problems"] = 1
        grad_config_data["lbfgs"]["n_problems"] = 1
        grad_cfg = ArmReacherConfig.from_dict(
            robot_cfg,
            grad_config_data["model"],
            grad_config_data["cost"],
            base_config_data["constraint"],
            base_config_data["convergence"],
            base_config_data["world_collision_checker_cfg"],
            world_model,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
        )
        cfg = ArmReacherConfig.from_dict(
            robot_cfg,
            config_data["model"],
            config_data["cost"],
            base_config_data["constraint"],
            base_config_data["convergence"],
            base_config_data["world_collision_checker_cfg"],
            world_model,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
        )

        # arm_rollout_grad = ArmReacher(grad_cfg)
        # arm_rollout_safety = ArmReacher(grad_cfg)
        aux_rollout = ArmReacher(grad_cfg)

        config_dict = LBFGSOptConfig.create_data_dict(
            grad_config_data["lbfgs"], aux_rollout, tensor_args
        )
        lbfgs_cfg = LBFGSOptConfig(**config_dict)

        if use_gradient_descent:
            newton_keys = NewtonOptConfig.__dataclass_fields__.keys()
            newton_d = {}
            lbfgs_k = vars(lbfgs_cfg)
            for k in newton_keys:
                newton_d[k] = lbfgs_k[k]
            newton_d["step_scale"] = 0.9

            newton_cfg = NewtonOptConfig(**newton_d)
            lbfgs = NewtonOptBase(newton_cfg)
        else:
            lbfgs = LBFGSOpt(lbfgs_cfg)

        if use_particle_opt:
            arm_rollout_mppi = ArmReacher(cfg)
            config_dict = ParallelMPPIConfig.create_data_dict(
                config_data["mppi"], arm_rollout_mppi, tensor_args
            )
            if use_es is not None and use_es:
                mppi_cfg = ParallelESConfig(**config_dict)
                if es_learning_rate is not None:
                    mppi_cfg.learning_rate = es_learning_rate
                parallel_mppi = ParallelES(mppi_cfg)
            else:
                mppi_cfg = ParallelMPPIConfig(**config_dict)
                parallel_mppi = ParallelMPPI(mppi_cfg)
            opts = [parallel_mppi]
        else:
            opts = []
        opts.append(lbfgs)
        cfg = WrapConfig(
            safety_rollout=aux_rollout,
            optimizers=opts,
            compute_metrics=True,
            use_cuda_graph_metrics=grad_config_data["lbfgs"]["use_cuda_graph"],
            sync_cuda_time=sync_cuda_time,
        )
        grasp = WrapBase(cfg)
        
        if ik_solver is None and manip_config_data['robot_file_with_arm'] is not None:
            robot_config_data = load_yaml(join_path(get_robot_configs_path(), manip_config_data['robot_file_with_arm']))["robot_cfg"]
            ik_link_names = aux_rollout.kinematics.transfered_link_name
            robot_config_data['kinematics']['link_names'] = ik_link_names
            robot_config_data['kinematics']['ee_link'] = ik_link_names[0]
            ik_robot_cfg = RobotConfig.from_dict(robot_config_data, tensor_args)
            ik_config = IKSolverConfig.load_from_robot_config(
                ik_robot_cfg,
                None,
                world_coll_checker=world_coll_checker,
                tensor_args=tensor_args,
                position_threshold=0.001,
                gradient_file='gradient_ik.yml',
                use_particle_opt=False,
                num_seeds=1,
            )
            ik_solver = IKSolver(ik_config)
        
        robot_has_arm_flag = (manip_config_data['robot_file_with_arm'] is not None) and (manip_config_data['robot_file_with_arm'] == manip_config_data['robot_file'])
        q_sample_gen = HeurGraspSeedGenerator(
            seeder_cfg=manip_config_data['seeder_cfg'],
            full_robot_model=aux_rollout.kinematics,
            ik_solver=ik_solver if robot_has_arm_flag else None,
            world_coll_checker=world_coll_checker,
            obj_lst=manip_name_list,
            tensor_args=tensor_args
        )
        
        grasp_cfg = GraspSolverConfig(
            robot_config=robot_cfg,
            robot_file=manip_config_data['robot_file'],
            solver=grasp,
            num_seeds=manip_config_data['seed_num'],
            grasp_threshold=grasp_threshold,
            distance_threshold=distance_threshold,
            world_coll_checker=world_coll_checker,
            q_sample_gen=q_sample_gen, 
            rollout_fn=aux_rollout,
            tensor_args=tensor_args,
            ik_solver=ik_solver,
        )
        return grasp_cfg
    

@dataclass
class GraspResult(Sequence):
    js_solution: JointState
    goal_pose: Goal
    solution: T_BDOF
    seed: T_BDOF
    success: T_BValue_bool
    contact_point: T_BValue_float
    contact_frame: T_BValue_float
    contact_force: T_BValue_float
    grasp_error: T_BValue_float

    #: rotation error is computed as \sqrt(q_des^T * q)
    dist_error: T_BValue_float
    solve_time: float
    debug_info: Optional[Any] = None

    def __getitem__(self, idx):
        if isinstance(self.debug_info, Dict):
            new_debug_info = {}
            for k, v in self.debug_info.items():
                new_debug_info[k] = v[idx]
        elif self.debug_info is None:
            new_debug_info = None
        else:
            raise NotImplementedError

        return GraspResult(
            js_solution=self.js_solution[idx],
            goal_pose=self.goal_pose[idx],
            solution=self.solution[idx],
            success=self.success[idx],
            seed=self.seed[idx],
            contact_point=self.contact_point[idx],
            contact_frame=self.contact_frame[idx],
            contact_force=self.contact_force[idx],
            grasp_error=self.grasp_error[idx],
            dist_error=self.dist_error[idx],
            debug_info=new_debug_info,
        )

    def __len__(self):
        return self.seed.shape[0]

    def get_unique_solution(self, roundoff_decimals: int = 2) -> torch.Tensor:
        in_solution = self.solution[self.success]
        r_sol = torch.round(in_solution, decimals=roundoff_decimals)

        if not (len(in_solution.shape) == 2):
            log_error("Solution shape is not of length 2")

        s, i = torch.unique(r_sol, dim=-2, return_inverse=True)
        sol = in_solution[i[: s.shape[0]]]

        return sol

    def get_batch_unique_solution(self, roundoff_decimals: int = 2) -> List[torch.Tensor]:
        in_solution = self.solution
        r_sol = torch.round(in_solution, decimals=roundoff_decimals)
        if not (len(in_solution.shape) == 3):
            log_error("Solution shape is not of length 3")

        # do a for loop and return list of tensors
        sol = []
        for k in range(in_solution.shape[0]):
            # filter by success:
            in_succ = in_solution[k][self.success[k]]
            r_k = r_sol[k][self.success[k]]
            s, i = torch.unique(r_k, dim=-2, return_inverse=True)
            sol.append(in_succ[i[: s.shape[0]]])
            # sol.append(s)
        return sol


class GraspSolver(GraspSolverConfig):
    def __init__(self, config: GraspSolverConfig) -> None:
        super().__init__(**vars(config))
        # self._solve_
        self.batch_size = -1
        self._num_seeds = self.num_seeds
        self.init_state = JointState.from_position(
            self.solver.rollout_fn.retract_state.unsqueeze(0)
        )
        self.dof = self.solver.safety_rollout.d_action
        self._col = torch.arange(0, 1, device=self.tensor_args.device, dtype=torch.long)

        # store og outer iters:
        self.og_newton_iters = self.solver.newton_optimizer.outer_iters
        self._goal_buffer = Goal()
        self._solve_state = None
        self._kin_list = None
        self._rollout_list = None

    def update_goal_buffer(
        self,
        solve_state: ReacherSolveState,
        # goal_pose: Pose,
        retract_config: Optional[T_BDOF] = None,
        link_poses: Optional[Dict[str, Pose]] = None,
    ) -> Goal:
        
        q_sample = self.sample_configs(self.world_coll_checker.n_envs)
        kin_state = self.fk(q_sample)
        goal_pose = Pose(kin_state.ee_position, kin_state.ee_quaternion)

        self._solve_state, self._goal_buffer, update_reference = solve_state.update_goal_buffer(
            goal_pose,
            None,
            retract_config,
            link_poses,
            self._solve_state,
            self._goal_buffer,
            self.tensor_args,
        )

        if update_reference:
            self.solver.update_nproblems(self._solve_state.get_ik_batch_size())
            self.reset_cuda_graph()
            self._goal_buffer.current_state = self.init_state.repeat_seeds(goal_pose.batch)
            self._col = torch.arange(
                0,
                self._goal_buffer.goal_pose.batch,
                device=self.tensor_args.device,
                dtype=torch.long,
            )

        return self._goal_buffer

    def solve_any(
        self,
        solve_type: ReacherSolveType,
        goal_pose: Pose,
        retract_config: Optional[T_BDOF] = None,
        seed_config: Optional[T_BDOF] = None,
        return_seeds: int = 1,
        num_seeds: Optional[int] = None,
        use_nn_seed: bool = True,
        newton_iters: Optional[int] = None,
        link_poses: Optional[Dict[str, Pose]] = None,
    ) -> GraspResult:
        if solve_type == ReacherSolveType.SINGLE:
            return self.solve_single(
                goal_pose,
                retract_config,
                seed_config,
                return_seeds,
                num_seeds,
                use_nn_seed,
                newton_iters,
                link_poses,
            )
        elif solve_type == ReacherSolveType.GOALSET:
            raise NotImplementedError
        elif solve_type == ReacherSolveType.BATCH:
            return self.solve_batch(
                goal_pose,
                retract_config,
                seed_config,
                return_seeds,
                num_seeds,
                use_nn_seed,
                newton_iters,
                link_poses,
            )
        elif solve_type == ReacherSolveType.BATCH_GOALSET:
            raise NotImplementedError
        elif solve_type == ReacherSolveType.BATCH_ENV:
            return self.solve_batch_env(
                goal_pose,
                retract_config,
                seed_config,
                return_seeds,
                num_seeds,
                use_nn_seed,
                newton_iters,
            )
        elif solve_type == ReacherSolveType.BATCH_ENV_GOALSET:
            raise NotImplementedError

    def solve_single(
        self,
        goal_pose: Pose,
        retract_config: Optional[T_BDOF] = None,
        seed_config: Optional[T_BDOF] = None,
        return_seeds: int = 1,
        num_seeds: Optional[int] = None,
        use_nn_seed: bool = True,
        newton_iters: Optional[int] = None,
        link_poses: Optional[Dict[str, Pose]] = None,
    ) -> GraspResult:
        if num_seeds is None:
            num_seeds = self.num_seeds
        if return_seeds > num_seeds:
            num_seeds = return_seeds

        solve_state = ReacherSolveState(
            ReacherSolveType.SINGLE, num_ik_seeds=num_seeds, batch_size=1, n_envs=1, n_goalset=1
        )

        return self.solve_from_solve_state(
            solve_state,
            # goal_pose,
            num_seeds,
            retract_config,
            seed_config,
            return_seeds,
            use_nn_seed,
            newton_iters,
            link_poses=link_poses,
        )

    def solve_batch(
        self,
        goal_pose: Pose,
        retract_config: Optional[T_BDOF] = None,
        seed_config: Optional[T_BDOF] = None,
        return_seeds: int = 1,
        num_seeds: Optional[int] = None,
        use_nn_seed: bool = True,
        newton_iters: Optional[int] = None,
        link_poses: Optional[Dict[str, Pose]] = None,
    ) -> GraspResult:
        if num_seeds is None:
            num_seeds = self.num_seeds
        if return_seeds > num_seeds:
            num_seeds = return_seeds

        solve_state = ReacherSolveState(
            ReacherSolveType.BATCH,
            num_ik_seeds=num_seeds,
            batch_size=goal_pose.batch,
            n_envs=1,
            n_goalset=1,
        )
        return self.solve_from_solve_state(
            solve_state,
            # goal_pose,
            num_seeds,
            retract_config,
            seed_config,
            return_seeds,
            use_nn_seed,
            newton_iters,
            link_poses=link_poses,
        )

    def solve_batch_env(
        self,
        # goal_pose: Pose,
        retract_config: Optional[T_BDOF] = None,
        seed_config: Optional[T_BDOF] = None,
        return_seeds: int = 1,
        num_seeds: Optional[int] = None,
        use_nn_seed: bool = True,
        newton_iters: Optional[int] = None,
        link_poses: Optional[Dict[str, Pose]] = None,
    ) -> GraspResult:
        if num_seeds is None:
            num_seeds = self.num_seeds
        if return_seeds > num_seeds:
            num_seeds = return_seeds

        solve_state = ReacherSolveState(
            ReacherSolveType.BATCH_ENV,
            num_ik_seeds=num_seeds,
            batch_size=self.world_coll_checker.n_envs,
            n_envs=self.world_coll_checker.n_envs,
            n_goalset=1,
        )
        return self.solve_from_solve_state(
            solve_state,
            # goal_pose,
            num_seeds,
            retract_config,
            seed_config,
            return_seeds,
            use_nn_seed,
            newton_iters,
            link_poses=link_poses,
        )

    def solve_from_solve_state(
        self,
        solve_state: ReacherSolveState,
        # goal_pose: Pose,
        num_seeds: int,
        retract_config: Optional[T_BDOF] = None,
        seed_config: Optional[T_BDOF] = None,
        return_seeds: int = 1,
        use_nn_seed: bool = True,
        newton_iters: Optional[int] = None,
        link_poses: Optional[Dict[str, Pose]] = None,
    ) -> GraspResult:
        # create goal buffer:
        goal_buffer = self.update_goal_buffer(solve_state, retract_config, link_poses)

        coord_position_seed = self.get_seed(
            num_seeds, goal_buffer.goal_pose, use_nn_seed, seed_config
        )

        if newton_iters is not None:
            self.solver.newton_optimizer.outer_iters = newton_iters
        self.solver.reset()
        self.solver._init_solver = True     # avoid run multiple times
        result = self.solver.solve(goal_buffer, coord_position_seed)
        if newton_iters is not None:
            self.solver.newton_optimizer.outer_iters = self.og_newton_iters
        grasp_result = self.get_result(num_seeds, result, goal_buffer.goal_pose, return_seeds)

        return grasp_result

    @profiler.record_function("grasp/get_result")
    def get_result(
        self, num_seeds: int, result: WrapResult, goal_pose: Pose, return_seeds: int
    ) -> GraspResult:
        success = self.get_success(result.metrics, num_seeds=num_seeds)
        all_sol = result.debug['mid_result'][0]
        all_sol.append(result.action.position.unsqueeze(1))
        all_sol = torch.cat(all_sol, dim=1)

        all_dist = result.debug['dist_error'][0]
        all_dist.append(result.metrics.dist_error)
        all_dist = torch.stack(all_dist, dim=1)
        q_sol, success, grasp_error, dist_error, contact_point, contact_frame, contact_force = get_result(
            result.metrics.grasp_error,
            all_dist,
            result.metrics.contact_point,
            result.metrics.contact_frame,
            result.metrics.contact_force,
            success,
            all_sol,
            self._col,
            goal_pose.batch,
            return_seeds,
            num_seeds,
        )
        # check if locked joints exist and create js solution:

        new_js = JointState(q_sol, joint_names=self.rollout_fn.kinematics.joint_names)
        sol_js = self.rollout_fn.get_full_dof_from_solution(new_js)
        # reindex success to get successful poses?
        grasp_result = GraspResult(
            success=success,
            goal_pose=goal_pose,
            solution=q_sol,
            seed=None,
            js_solution=sol_js,
            # seed=coord_position_seed[idx].view(goal_pose.batch, return_seeds, -1).detach(),
            contact_point=contact_point,
            contact_frame=contact_frame,
            contact_force=contact_force,
            grasp_error=grasp_error,
            dist_error=dist_error,
            solve_time=result.solve_time,
            debug_info={"solver": result.debug},
        )
        return grasp_result

    @profiler.record_function("grasp/get_seed")
    def get_seed(
        self, num_seeds: int, goal_pose: Pose, use_nn_seed, seed_config: Optional[T_BDOF] = None
    ) -> torch.Tensor:
        if seed_config is None:
            coord_position_seed = self.generate_seed(
                num_seeds=num_seeds,
                batch=goal_pose.batch,
                use_nn_seed=use_nn_seed,
            )
        elif seed_config.shape[1] < num_seeds:
            repeat_time = math.ceil(num_seeds / seed_config.shape[1])
            coord_position_seed = seed_config.repeat(1, repeat_time, 1)[:, :num_seeds, :]

            # coord_position_seed = self.generate_seed(
            #     num_seeds=num_seeds - seed_config.shape[1],
            #     batch=goal_pose.batch,
            #     use_nn_seed=use_nn_seed,
            # )
            # coord_position_seed = torch.cat((seed_config, coord_position_seed), dim=1)
        else:
            coord_position_seed = seed_config
        coord_position_seed = coord_position_seed.view(-1, 1, self.dof)
        return coord_position_seed

    @torch.no_grad()
    @profiler.record_function("grasp/get_success")
    def get_success(self, metrics: RolloutMetrics, num_seeds: int) -> torch.Tensor:
        
        log_warn(f'final succ (thre=0.2) {(metrics.grasp_error.max(dim=-1)[0] < 0.2).sum()}')
        log_warn(f'final succ (thre=0.1) {(metrics.grasp_error.max(dim=-1)[0] < 0.1).sum()}')
        log_warn(f'final dist error mean {metrics.dist_error.mean()} max {metrics.dist_error.max()}')
        success = get_success(
            metrics.feasible,
            metrics.grasp_error,
            metrics.dist_error,
            num_seeds,
            self.grasp_threshold,
            self.distance_threshold,
        )

        return success

    def replace_q_with_ik_result(self, q: torch.Tensor, ik_q: torch.Tensor):
        ik_replace_ind = self.ik_solver.kinematics.kinematics_config.get_replace_index(self.kinematics.kinematics_config)
        new_q = q + 0
        new_q[..., ik_replace_ind] = ik_q
        return new_q
    
    @profiler.record_function("grasp/generate_seed")
    def generate_seed(
        self,
        num_seeds: int,
        batch: int,
        use_nn_seed: bool = False,
    ) -> torch.Tensor:
        """Generate seeds for a batch. Given a pose set, this will create all
        the seeds: [batch + batch*random_restarts]

        Args:
            batch (int, optional): [description]. Defaults to 1.
            num_seeds (Optional[int], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        num_random_seeds = num_seeds
        seed_list = []
        if use_nn_seed and self.grasp_nn_seeder is not None:
            raise NotImplementedError
        if num_random_seeds > 0:
            random_seed = self.q_sample_gen.get_samples(batch, num_random_seeds)
            seed_list.append(random_seed)
        coord_position_seed = torch.cat(seed_list, dim=1)
        return coord_position_seed

    def update_world(self, world_lst: List[WorldConfig], gravity_center = None, obb_length = None, contact_obj_names: List[str] = None):
        """Update the world representation for collision checking.

        This allows for updating the world representation as long as the new world representation
        does not have a larger number of obstacles than the :attr:`MotionGen.collision_cache` as
        created during initialization of :class:`MotionGenConfig`. Updating the world also
        invalidates the cached roadmaps in the graph planner. See :ref:`world_collision` for more
        details.

        Args:
            world: New world configuration for collision checking.
        """
        self.world_coll_checker.clear_cache()
        self.world_coll_checker.load_batch_collision_model(world_lst)
        self.world_coll_checker.load_contact_obj(world_lst, contact_obj_names)
        if gravity_center is not None and obb_length is not None:
            self.rollout_fn.grasp_cost.reset(gravity_center, obb_length)
            self.rollout_fn.grasp_convergence.reset(gravity_center, obb_length)
        if contact_obj_names is not None:
            self.reset_seed(contact_obj_names)
        
    def reset_seed(self, contact_obj_names) -> None:
        self.q_sample_gen.reset(contact_obj_names)

    def check_constraints(self, q: JointState) -> RolloutMetrics:
        metrics = self.rollout_fn.rollout_constraint(q.position.unsqueeze(1))
        return metrics

    def sample_configs(self, n: int, use_batch_env=False) -> torch.Tensor:
        """
        generate fake samples
        """
        samples = self.tensor_args.to_device(torch.zeros((n, self.dof)))
        samples[:, 3] = 1   # quaternion
        return samples 

    @property
    def kinematics(self) -> CudaRobotModel:
        return self.rollout_fn.dynamics_model.robot_model

    def get_all_rollout_instances(self) -> List[RolloutBase]:
        if self._rollout_list is None:
            self._rollout_list = [self.rollout_fn] + self.solver.get_all_rollout_instances()
        return self._rollout_list

    def get_all_kinematics_instances(self) -> List[CudaRobotModel]:
        if self._kin_list is None:
            self._kin_list = [
                i.dynamics_model.robot_model for i in self.get_all_rollout_instances()
            ]
        return self._kin_list

    @torch.no_grad()
    def fk(self, q: torch.Tensor) -> CudaRobotModelState:
        return self.kinematics.get_state(q)

    def reset_cuda_graph(self) -> None:
        self.solver.reset_cuda_graph()
        self.rollout_fn.reset_cuda_graph()

    def attach_object_to_robot(
        self,
        sphere_radius: float,
        sphere_tensor: Optional[torch.Tensor] = None,
        link_name: str = "attached_object",
    ) -> None:
        for k in self.get_all_kinematics_instances():
            k.attach_object(
                sphere_radius=sphere_radius, sphere_tensor=sphere_tensor, link_name=link_name
            )

    def detach_object_from_robot(self, link_name: str = "attached_object") -> None:
        for k in self.get_all_kinematics_instances():
            k.detach_object(link_name)

    def get_retract_config(self):
        return self.rollout_fn.dynamics_model.retract_config

    @property
    def joint_names(self) -> List[str]:
        """Get ordered names of all joints used in optimization with IKSolver."""
        return self.rollout_fn.kinematics.joint_names

@torch.jit.script
def get_success(
    feasible,
    grasp_error,
    dist_error,
    num_seeds: int,
    grasp_threshold: float,
    distance_threshold: float,
):
    feasible = feasible.view(-1, num_seeds)
    converge = torch.logical_and(
        grasp_error.max(dim=-1)[0] <= grasp_threshold,
        dist_error <= distance_threshold,
    ).view(-1, num_seeds)
    success = torch.logical_and(feasible, converge)
    return success


# @torch.jit.script
def get_result(
    grasp_error,
    dist_error,
    contact_point,
    contact_frame,
    contact_force,
    success,
    sol_position,
    col,
    batch_size: int,
    return_seeds: int,
    num_seeds: int,
):
    error = grasp_error.max(dim=-1)[0].view(-1, num_seeds)
    error[~success] += 1000.0
    _, idx = torch.topk(error, k=return_seeds, largest=False, dim=-1)
    idx = idx + num_seeds * col.unsqueeze(-1)
    q_sol = sol_position[idx].view((batch_size, return_seeds) + sol_position.shape[1:])
    success = success.view(-1)[idx].view(batch_size, return_seeds)
    grasp_error = grasp_error[idx].view(batch_size, return_seeds, -1)
    dist_error = dist_error[idx].view(batch_size, return_seeds, -1)
    contact_point = contact_point[idx].view((batch_size, return_seeds) + contact_point.shape[1:])
    contact_frame = contact_frame[idx].view((batch_size, return_seeds) + contact_frame.shape[1:])
    contact_force = contact_force[idx].view((batch_size, return_seeds) + contact_force.shape[1:])
    
    return q_sol, success, grasp_error, dist_error, contact_point, contact_frame, contact_force
