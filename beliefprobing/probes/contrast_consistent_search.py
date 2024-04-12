import random
from functools import partial

import torch
import numpy as np

import mdmm
import cvxopt

from beliefprobing.probes.trainer import Trainer
from beliefprobing.probes.beliefprobe import BeliefProbe, Direction
from beliefprobing.generate import GenOut


class CCS(Trainer):

    def __init__(self, use_constraint=False, informative_loss='min_sq', **kwargs):
        super().__init__(**kwargs)
        self.use_constraint = use_constraint
        self.informative_loss_type = informative_loss
        self.constraint_values = {}
        self.mdmm = None

    def setup_optimizer(self, batch_size):
        params = self.probe.parameters()

        constraints = []
        if self.use_constraint in ('informative', 'both'):
            from math import log
            constraints.append(mdmm.MaxConstraintHard(
                partial(lambda trainer: trainer.constraint_values['informative'], self), 0.9 * log(2)
            ))
        elif self.use_constraint in ('consistent', 'both'):
            constraints.append(mdmm.EqConstraint(
                partial(lambda trainer: trainer.constraint_values['consistent'], self), 0
            ))

        if self.use_constraint != 'none':
            mdmm_module = mdmm.MDMM(constraints).to(self.device)
            lambdas = [c.lmbda for c in mdmm_module]
            slacks = [c.slack for c in mdmm_module if hasattr(c, 'slack')]
            params = [
                {'params': params, 'lr': self.lr},
                {'params': lambdas, 'lr': -self.lr},
                {'params': slacks, 'lr': self.lr}
            ]
            self.mdmm = mdmm_module

        return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        # return torch.optim.SGD(params, lr=self.lr, weight_decay=self.weight_decay)

    def do_batch(self, x0_batch, x1_batch, _):
        # probe
        p0, p1 = self.probe(x0_batch), self.probe(x1_batch)

        # get the corresponding loss
        loss = self.get_loss(p0, p1)
        pred = (p0 + (1 - p1)) / 2
        return loss, pred

    def get_informative_loss(self, p0, p1, reduction='mean'):
        if self.informative_loss_type == 'min_sq':
            loss = torch.min(torch.stack([p0, p1, 1-p0, 1-p1]), dim=0)[0] ** 2
        elif self.informative_loss_type == 'entropy':
            loss = - p0 * p0.log() - p1 * p1.log()
        else:
            loss = torch.zeros((1,)).to(p0.device)

        if reduction == 'mean':
            return loss.mean(0)
        else:
            return loss

    def get_consistent_loss(self, p0, p1, reduction='mean'):
        loss = (p0 - (1 - p1)) ** 2
        if reduction == 'mean':
            return loss.mean(0)
        else:
            return loss

    def get_loss(self, p0, p1):
        if self.use_constraint == 'informative':
            values = self.get_informative_loss(p0, p1, reduction='none')
            self.constraint_values['informative'] = torch.sum(values * (values/0.5).softmax(0))
            return self.mdmm(self.get_consistent_loss(p0, p1)).value
        if self.use_constraint == 'consistent':
            self.constraint_values['consistent'] = self.get_consistent_loss(p0, p1)
            return self.mdmm(self.get_informative_loss(p0, p1)).value
        if self.use_constraint == 'both':
            self.constraint_values['informative'] = torch.mean(torch.log(torch.abs(p0 - p1) + 1e-8))
            self.constraint_values['consistent'] = self.get_consistent_loss(p0, p1)
            return self.mdmm(torch.tensor(0.0, device=p0.device)).value
        return self.get_informative_loss(p0, p1) + self.get_consistent_loss(p0, p1)


@BeliefProbe.register('ccs_gd')
class ContrastConsistentSearchGradientDescent(BeliefProbe):

    def __init__(
            self, seed: int, n_epochs: int, lr: float, batch_size: int, weight_decay: float, informative_loss: str,
            verbose=False, device="cuda", use_wandb=False, wandb_group=None, use_constraint=False
    ):
        super().__init__(svd_data=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed
        self.weight = None
        self.ccs = CCS(
            n_epochs=n_epochs, lr=lr, batch_size=batch_size, verbose=verbose, device=device,
            weight_decay=weight_decay, use_wandb=use_wandb, wandb_group=wandb_group, use_constraint=use_constraint,
            informative_loss=informative_loss
        )

    def do_train(self, gen_out: GenOut) -> None:
        hs, ps, y = gen_out
        N, P = hs[0], hs[1]

        loss = self.ccs.train(N, P, y)
        self.weight = self.ccs.probe._modules['0'].weight.data.squeeze()
        self.ccs = None
        return loss

    def _get_direction(self) -> Direction:
        return self.weight.cpu()


@BeliefProbe.register('ccs_convex')
class ContrastConsistentSearchConvex(BeliefProbe):

    def __init__(self):
        super().__init__(svd_data=True)

    def do_train(self, gen_out: GenOut) -> None:
        hs, _, _ = gen_out
        Xp, Xn = hs[0].double(), hs[1].double()

        def obj(w):
            # y = (1 - torch.sigmoid(Xp @ w) + torch.sigmoid(Xn @ w)) ** 2
            y = 1 - torch.sigmoid(Xp @ w) + torch.sigmoid(Xn @ w)
            return y.sum()
        obj_grad = torch.func.grad(obj)
        obj_hess = torch.func.hessian(obj)

        def c1(w):
            # return torch.mean((Xp - Xn) @ w)**2 - 1
            return torch.mean((Xp - Xn) @ w)
        c1_grad = torch.func.grad(c1)
        c1_hess = torch.func.hessian(c1)

        # def c2(w):
        #     return (Xn @ w)**2 - 1

        def F(w=None, z=None):
            if w is None:
                # return 1, cvxopt.matrix(np.ones(Xp.shape[1]))
                return 1, cvxopt.matrix(np.random.rand(Xp.shape[1]))

            w_t = torch.tensor(np.array(w))
            f = cvxopt.matrix(np.array([
                obj(w_t).item(),
                c1(w_t).item(),
                # c2(w_t).float().numpy()
            ]))
            Df = cvxopt.matrix(np.concatenate([
                obj_grad(w_t).T.numpy(),
                c1_grad(w_t).T.numpy(),
            ]))

            if z is None:
                return f, Df

            H = cvxopt.matrix((
                z[0] * obj_hess(w_t.squeeze())
                + z[1] * c1_hess(w_t.squeeze())
              # + z[3] * torch.func.hessian(c2)(torch.tensor(w)),
            ).numpy())

            return f, Df, H

        result = cvxopt.solvers.cp(F)
        self.theta = 0

    def _get_direction(self) -> Direction:
        return torch.tensor(self.theta).squeeze()

