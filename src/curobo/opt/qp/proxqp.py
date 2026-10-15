import os
import numpy as np
import torch

import proxsuite

from .base import QPSolver


class PROXQP(QPSolver):
    
    @property
    def glh_type(self):
        return 'gh'
    
    def init_problem(self, G_matrix, l_matrix, h_matrix):
        super().init_problem(G_matrix, l_matrix, h_matrix)
        self.solver = 'proxqp'
        
    def solve(self, Q_matrix, semi_Q_matrix, solution=None):
        solution = proxqp_function.apply(Q_matrix, self.G_matrix, self.h_matrix)
        return solution


class proxqp_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, G, u, proxqp_parallel=True):
        nBatch = Q.shape[0]

        eps = 1e-9
        ctx.vector_of_qps = proxsuite.proxqp.dense.BatchQP()
        ctx.proxqp_parallel = proxqp_parallel
        ctx.nBatch = nBatch

        _, nineq, nz = G.size()
        assert nineq > 0
        ctx.nineq, ctx.nz = nineq, nz

        ctx.cpu = os.cpu_count()
        if ctx.cpu is not None:
            ctx.cpu = max(1, int(ctx.cpu))

        zhats = torch.empty((nBatch, ctx.nz)).type_as(Q)

        for i in range(nBatch):
            qp = ctx.vector_of_qps.init_qp_in_place(ctx.nz, 0, ctx.nineq)
            qp.settings.primal_infeasibility_solving = True
            qp.settings.max_iter = 1000
            qp.settings.max_iter_in = 100
            default_rho = 5.0e-5
            qp.settings.default_rho = default_rho
            qp.settings.refactor_rho_threshold = default_rho  # no refactorization
            qp.settings.eps_abs = eps
            H__ = Q[i].cpu().numpy()
            G__ = G[i].cpu().numpy()
            u__ = u[i].cpu().numpy()

            qp.init(
                H=H__, g=None, A=None, b=None, C=G__, l=None, u=u__, rho=default_rho
            )

        if ctx.proxqp_parallel:
            proxsuite.proxqp.dense.solve_in_parallel(
                num_threads=ctx.cpu, qps=ctx.vector_of_qps
            )
        else:
            for i in range(ctx.vector_of_qps.size()):
                ctx.vector_of_qps.get(i).solve()

        for i in range(nBatch):
            zhats[i] = torch.tensor(ctx.vector_of_qps.get(i).results.x)

        return zhats

    @staticmethod
    def backward(ctx, dl_dzhat):
        device = dl_dzhat.device
        nBatch, dim, nineq = ctx.nBatch, ctx.nz, ctx.nineq
        dQs = torch.empty(nBatch, ctx.nz, ctx.nz, device=device)

        ctx.cpu = os.cpu_count()
        if ctx.cpu is not None:
            ctx.cpu = max(1, int(ctx.cpu / 2))

        n_tot = dim + nineq
        
        eps_backward = 1e-4
        rho_backward = 1e-6
        mu_backward = 1e-6

        if ctx.proxqp_parallel:
            vector_of_loss_derivatives = (
                proxsuite.proxqp.dense.VectorLossDerivatives()
            )

            for i in range(nBatch):
                rhs = np.zeros(n_tot)
                rhs[:dim] = dl_dzhat[i].cpu()
                vector_of_loss_derivatives.append(rhs)

            proxsuite.proxqp.dense.solve_backward_in_parallel(
                num_threads=ctx.cpu,
                qps=ctx.vector_of_qps,
                loss_derivatives=vector_of_loss_derivatives,
                eps=eps_backward,
                rho_backward=rho_backward,
                mu_backward=mu_backward,
            )  # try with systematic fwd bwd
        else:
            for i in range(nBatch):
                rhs = np.zeros(n_tot)
                rhs[:dim] = dl_dzhat[i].cpu()
                qpi = ctx.vector_of_qps.get(i)
                proxsuite.proxqp.dense.compute_backward(
                    qp=qpi,
                    loss_derivative=rhs,
                    eps=eps_backward,
                    rho_backward=rho_backward,
                    mu_backward=mu_backward,
                )

        for i in range(nBatch):
            dQs[i] = torch.tensor(
                ctx.vector_of_qps.get(i).model.backward_data.dL_dH
            )

        grads = (dQs, None, None, None)
        print('grad', dQs.abs().mean(), dQs.abs().max())

        return grads

