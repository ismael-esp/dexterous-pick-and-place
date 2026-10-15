from .qp import QPEnergy, QPBaseEnergy
from .dfc import DFCEnergy
from .tdg import TDGEnergy, TDGQ1Energy
from .base import GraspEnergyBase

def init_grasp_energy(ge_param, tensor_args):
    ge_param['tensor_args'] = tensor_args
    if ge_param['type'] == 'qp':
        return QPEnergy(**ge_param)
    elif ge_param['type'] == 'qpbase':
        return QPBaseEnergy(**ge_param)
    elif ge_param['type'] == 'dfc' or ge_param['type'] == 'none':
        return DFCEnergy(**ge_param)
    elif ge_param['type'] == 'tdg':
        return TDGEnergy(**ge_param)
    elif ge_param['type'] == 'ch_q1':
        from .chq1 import CHQ1Energy
        return CHQ1Energy(**ge_param)
    elif ge_param['type'] == 'tdg_q1':
        return TDGQ1Energy(**ge_param)
    else:
        raise NotImplementedError