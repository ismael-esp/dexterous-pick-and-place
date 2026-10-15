
import torch
import math 
import numpy as np
import multiprocessing as mp
from curobo.types.base import TensorDeviceType

from .base import QPSolver

class OSQP(QPSolver):
    
    @property
    def glh_type(self):
        return 'gh'
    
    def init_problem(self, G_matrix, l_matrix, h_matrix):
        super().init_problem(G_matrix, l_matrix, h_matrix)
        self.G_matrix_np = self.G_matrix.cpu().numpy()[0]
        self.h_matrix_np = self.h_matrix.cpu().numpy()[0]
        self.solver = 'osqp'
        self.tensor_args: TensorDeviceType = TensorDeviceType()
            
    def solve(self, Q_matrix, semi_Q_matrix, solution=None):
        semi_Q_matrix_np=semi_Q_matrix.detach().cpu().numpy()
        length=semi_Q_matrix_np.shape[0]
        max_parallel=min(10, length)
        Pools=mp.Pool(max_parallel)
        per_list=math.ceil(length/max_parallel)
        print("length",length,"per_list",per_list)
        results_tmp=[Pools.apply_async(parrallel_LCLP2,args=(semi_Q_matrix_np[per_list*i:min(length,per_list*(i+1))],self.G_matrix_np,self.h_matrix_np,per_list*i,min(length,per_list*(i+1))))for i in range(max_parallel)]
        results=[res.get() for res in results_tmp]
        #下面要使用pytorch
        solution=self.tensor_args.to_device(torch.zeros(Q_matrix.shape[0],Q_matrix.shape[1]))
        LCQP_num_sum=0
        LCQP_try_sum=0
        for i in range(max_parallel):
            val,inital_id,end_id,LCQP_num,LCQP_try=results[i]
            solution[inital_id:end_id]=self.tensor_args.to_device(torch.tensor(val))
            LCQP_num_sum+=LCQP_num
            LCQP_try_sum+=LCQP_try
        print("LCQP_num",LCQP_num_sum,"LCQP_try",LCQP_try_sum,"average try",LCQP_try_sum/LCQP_num_sum)

        return solution  

def parrallel_LCLP2(semi_Q_matrix_list,A_matrix,B_matrix,inital_id,end_id):
    '''
    Q_matrix_list: [k,6,3n+1]
    这里面使用np
    '''
    import cvxpy as cp
    total_points=semi_Q_matrix_list.shape[-1]-1
    LCQP_num=0
    LCQP_try=0
    semi_Q_matrix_list=semi_Q_matrix_list.reshape(-1,6,total_points+1)#km,6,3n+1
    val=np.zeros([semi_Q_matrix_list.shape[0],total_points+1])
    X = cp.Variable((total_points+1))
    assert(A_matrix.shape[0]==B_matrix.shape[0])
    assert(A_matrix.shape[1]==X.shape[0])
    A=cp.Parameter((A_matrix.shape[0],A_matrix.shape[1]))
    B=cp.Parameter((B_matrix.shape[0]))
    Q=cp.Parameter((semi_Q_matrix_list.shape[1],semi_Q_matrix_list.shape[2]))
    constraints = [A@X<=B]
    objective = cp.Minimize(cp.sum_squares(Q@X))
    prob = cp.Problem(objective,constraints)
    assert(end_id-inital_id==semi_Q_matrix_list.shape[0])
    A.value=A_matrix
    B.value=B_matrix
    for (i) in range(end_id-inital_id):
        LCQP_num+=1
        Q.value=semi_Q_matrix_list[i]
        for j in range(10):
            LCQP_try+=1
            
            prob.solve(solver=cp.OSQP)
            if(prob.status=='optimal'):
                val[i]=X.value
                break       

    return val,inital_id,end_id,LCQP_num,LCQP_try
