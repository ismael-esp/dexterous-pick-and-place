from .official_reluqp import OFFICIAL_RELUQP
from .reluqp import BATCHED_RELUQP


def init_QP_solver(solver_type = 'batch_reluqp'):
    if solver_type == 'batch_reluqp':
        return BATCHED_RELUQP()
    elif solver_type == 'official_reluqp':
        return OFFICIAL_RELUQP()
    elif solver_type == 'proxqp':
        from .proxqp import PROXQP
        return PROXQP()
    elif solver_type == 'osqp':
        from .osqp import OSQP
        return OSQP()
    else:
        raise NotImplementedError