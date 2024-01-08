import copy
from abc import abstractmethod
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

import mdmm


def default_probe(dim):
    return nn.Sequential(nn.Linear(dim, 1, bias=False), nn.Sigmoid())


class Trainer(object):
    def __init__(self, n_epochs=1000, lr=1e-3, batch_size=-1, verbose=False, device="cuda",
                 probe_fn=default_probe, weight_decay=0.01, use_wandb=False, wandb_group=None):

        # training
        self.n_epochs = n_epochs
        self.lr = lr
        self.verbose = verbose
        self.device = device
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        # probe
        self.probe_builder = probe_fn
        self.probe = None
        self.best_probe = None

        self.use_wandb = use_wandb
        self.wandb_group = wandb_group

    def set_best_probe(self):
        self.probe.cpu()
        self.best_probe = copy.deepcopy(self.probe)
        self.best_probe.eval()
        self.probe.to(self.device)

    def initialize_probe(self, dims):
        self.probe = self.probe_builder(dims).to(self.device)

    def get_acc(self, x0_test, x1_test, y_test):
        """
        Computes accuracy for the current parameters on the given test inputs
        """
        batch_size = len(x0_test) if self.batch_size == -1 else self.batch_size
        n_batches = len(x0_test) // batch_size

        x0 = x0_test.to(self.device)
        x1 = x1_test.to(self.device)

        ps = []
        with torch.no_grad():
            for j in range(n_batches):
                x0_batch = x0[j * batch_size:(j + 1) * batch_size]
                x1_batch = x1[j * batch_size:(j + 1) * batch_size]
                _, p = self.do_batch(x0_batch, x1_batch, None)
                ps.append(p.squeeze())
        ps = torch.cat(ps)

        predictions = ps.detach().cpu() > 0.5
        acc = (predictions.long() == y_test).float().mean()
        acc = max(acc, 1 - acc)
        return acc, ps.detach().cpu()

    @abstractmethod
    def setup_optimizer(self, batch_size):
        raise NotImplementedError

    @abstractmethod
    def do_batch(self, x0_batch, x1_batch, y_batch):
        raise NotImplementedError

    def train(self, x0, x1, y):
        """
        Does a single training run of n_epochs epochs
        """
        if self.use_wandb:
            import wandb
            config = {
                k: self.__dict__[k]
                for k in ['n_epochs', 'n_tries', 'lr', 'device', 'batch_size', 'weight_decay', 'd'] }
            # with utils.HiddenPrints(hide_stderr=(not self.verbose)):
            run = wandb.init(
                project="latent-conditional-beliefs",
                group=self.wandb_group,
                config=config,
                reinit=True
            )

        permutation = torch.randperm(len(x0))
        x0, x1, y = x0[permutation], x1[permutation], y[permutation]
        x0, x1, y = x0.to(self.device), x1.to(self.device), y.to(self.device)

        batch_size = len(x0) if self.batch_size == -1 else self.batch_size
        n_batches = len(x0) // batch_size
        n_dims = x0.shape[1]

        self.initialize_probe(n_dims)

        optimizer = self.setup_optimizer(batch_size)

        # Start training (full batch)
        for epoch in range(self.n_epochs):
            for j in range(n_batches):
                x0_batch = x0[j * batch_size:(j + 1) * batch_size]
                x1_batch = x1[j * batch_size:(j + 1) * batch_size]
                y_batch = y[j * batch_size:(j + 1) * batch_size]

                loss, _ = self.do_batch(x0_batch, x1_batch, y_batch)

                # update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.use_wandb:
                run.log({'loss': loss.cpu().detach().item()})

        return loss.detach().cpu().item()


class LR(Trainer):

    def __init__(self, **kwargs):
        self.loss = nn.BCELoss()
        super().__init__(**kwargs)

    def setup_optimizer(self, batch_size):
        return torch.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def do_batch(self, x0_batch, x1_batch, y_batch):
        x_batch = x0_batch - x1_batch
        p = self.probe(x_batch)
        if y_batch is not None:
            loss = self.loss(p.squeeze(), y_batch.float())
        else:
            loss = None
        return loss, p


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

    def do_batch(self, x0_batch, x1_batch, _):
        # probe
        p0, p1 = self.probe(x0_batch), self.probe(x1_batch)

        # get the corresponding loss
        loss = self.get_loss(p0, p1)
        pred = (p0 + (1 - p1)) / 2
        return loss, pred

    def get_informative_loss(self, p0, p1, reduction='mean'):
        if self.informative_loss_type == 'min_sq':
            loss = torch.min(p0, p1) ** 2
        elif self.informative_loss_type == 'entropy':
            loss = - p0 * p0.log() - p1 * p1.log()

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


class CCReflection(Trainer):

    def setup_optimizer(self, batch_size):
        return torch.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def do_batch(self, x0_batch, x1_batch, y_batch):
        loss = None
        if y_batch is not None:  # training
            lin = self.probe._modules['0']
            z = F.normalize(lin.weight)  # 1,    dims

            # construct householder transformation matrix
            hh = torch.eye(z.size(1), device=z.device) - 2 * z.T @ z  # dims, dims

            # bias is incorporated by translating all vectors by bias*normal (away from reflecting hyperplane)
            # s = torch.sign(x0_batch @ z.T)
            # _x0_batch = x0_batch + s * lin.bias * z
            # _x1_batch = x1_batch + s * lin.bias * z

            # minimize difference between x1 and reflected(x0)
            loss = torch.linalg.vector_norm(x1_batch - x0_batch @ hh, dim=1).mean()

        p0 = self.probe(x0_batch)
        p1 = self.probe(x1_batch)

        pred = (p0 + 1. - p1) / 2.
        return loss, pred