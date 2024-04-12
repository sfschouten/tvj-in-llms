"""

"""
import math
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from beliefprobing.probes.trainer import Trainer
from beliefprobing.probes.beliefprobe import BeliefProbe, Direction
from beliefprobing.generate import GenOut


class ShearOp(nn.Module):

    def __init__(self, dims, shear_dims_only: bool):
        """
        :param dims: The axes w.r.t. which we shear.
        :param shear_dims_only: Whether to provide only the dim parallel to the shear, or a full shear matrix.
        """
        super().__init__()
        self.dims = dims
        self.shear_dim_only = shear_dims_only

    def forward(self, v):

        # the axis we stay parallel to
        for i, dim in enumerate(self.dims):
            v.data[i, dim] = 1

        if self.shear_dim_only:
            return v

        shear = torch.eye(v.shape[1])
        for i, dim in enumerate(self.dims):
            shear[dim, :] = v[i]
        return shear


class _WeightNorm(nn.Module):
    def __init__(self, dim: int = 0) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, weight):
        return F.normalize(weight, p=2, dim=self.dim)


class ShearProbe(nn.Module):

    def __init__(self, nr_dims, nr_directions=1):
        super().__init__()
        self.nr_dims = nr_dims
        self.nr_directions = nr_directions

        # components
        self.rotation = nn.Parameter(torch.empty((nr_dims, nr_dims)))
        self.shear = nn.Parameter(torch.empty(nr_directions, nr_dims))
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCELoss()

        nn.init.kaiming_uniform_(self.rotation, a=math.sqrt(5))
        nn.init.zeros_(self.shear)

        # ensure `self.rotation` is orthonormal
        nn.utils.parametrizations.orthogonal(self, name='rotation')
        nn.utils.parametrizations.parametrize.register_parametrization(
            self, "rotation", _WeightNorm(dim=1)
        )

        # ensure `self.shear` is a shear operation
        nn.utils.parametrizations.parametrize.register_parametrization(
            self, "shear", ShearOp(dims=range(nr_directions), shear_dims_only=True)
        )

    def forward(self, x_n, x_p, y):
        """
        :param x_n: representations for negatives (N x D)
        :param x_p: representations for positives (N x D)
        :param y: labels (N)
        :return:
        """
        x_n @= self.rotation
        x_p @= self.rotation                                    # (N x D)
        l1_n = x_n[:, :self.nr_directions]
        l1_p = x_p[:, :self.nr_directions]                      # (N x nr_dir)
        pr1_n = torch.sigmoid(l1_n)
        pr1_p = torch.sigmoid(l1_p)                             # (N x nr_dir)
        pr1 = ((1-pr1_n) + pr1_p) / 2

        # maximize |x_pos - x_neg| before shear (confidence loss), causes rotation to align with belief direction(s)
        # conf_loss = -(l1_p - l1_n).prod(dim=1).sum()
        conf_loss = torch.min(torch.stack([pr1_p, pr1_n, 1-pr1_p, 1-pr1_n]), dim=0)[0].pow(2).sum()

        # Beliefs and other variables might be correlated, so hyperplane normal to belief direction might not cleanly
        # separate positives from negatives. We apply a shear to undo that correlation.
        l2_n = x_n @ self.shear.T
        l2_p = x_p @ self.shear.T

        # TODO support self.nr_directions > 1
        #  - should have as many cols in `y` as directions, each providing labels for each direction
        #  - consistency loss should be applied to each

        pr2_n = torch.sigmoid(l2_n)
        pr2_p = torch.sigmoid(l2_p)
        pr2 = ((1-pr2_n) + pr2_p) / 2

        # minimize x_pos + x_neg after shear which should make  (consistency loss)
        # cons_loss = (l2_p + l2_n).pow(2).sum()
        cons_loss = (1 - pr2_n - pr2_p).pow(2).sum()

        # if we have any data points with supervision we add a supervised loss term
        supervised_loss = self.bce_loss(pr2[y.isfinite()].squeeze(), y[y.isfinite()].float())

        return pr1, pr2, conf_loss, cons_loss, supervised_loss


class CCShear(Trainer):

    def __init__(self, **kwargs):
        super().__init__(probe_fn=lambda nr_dims: ShearProbe(nr_dims), **kwargs)

    def setup_optimizer(self, batch_size):
        return torch.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def do_batch(self, x0_batch, x1_batch, y_batch):
        p1, p2, conf_loss, cons_loss, supervised_loss = self.probe(x0_batch, x1_batch, y_batch)
        loss = conf_loss + cons_loss #+ supervised_loss
        return loss, p2


@BeliefProbe.register('ccrs')
class ContrastConsistentRotateAndShear(BeliefProbe):

    def __init__(
            self, seed: int, n_epochs: int, lr: float, batch_size: int, weight_decay: float,
            verbose=False, device="cuda", use_wandb=False, wandb_group=None
    ):
        super().__init__(svd_data=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed
        self.weight = None
        self.ccs = CCShear(
            n_epochs=n_epochs, lr=lr, batch_size=batch_size, verbose=verbose, device=device,
            weight_decay=weight_decay, use_wandb=use_wandb, wandb_group=wandb_group
        )

    def do_train(self, gen_out: GenOut) -> None:
        hs, _, y = gen_out
        N, P = hs[0], hs[1]

        loss = self.ccs.train(N, P, y)

        # print(f'accuracy1: {(p1 > .5) == y}')
        # print(f'accuracy2: {(p2 > .5) == y}')
        self.weight = self.ccs.probe.rotation[0].cpu()
        self.ccs = None
        return loss

    def _get_direction(self) -> Direction:
        return self.weight