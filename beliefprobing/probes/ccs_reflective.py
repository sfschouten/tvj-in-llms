import random

import torch
import torch.nn.functional as F
import numpy as np

from beliefprobing.probes.beliefprobe import BeliefProbe, Direction
from beliefprobing.probes.trainer import Trainer
from beliefprobing.generate import GenOut


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


@BeliefProbe.register('ccr')
class ContrastConsistentReflection(BeliefProbe):

    def __init__(self, seed: int, n_epochs=1000, lr=1e-3, batch_size=-1, verbose=False, device="cuda",
                 weight_decay=0.01, use_wandb=False, wandb_group=None):
        super().__init__(svd_data=False)

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed
        self.ccr = CCReflection(
            n_epochs=n_epochs, lr=lr, batch_size=batch_size, verbose=verbose, device=device,
            weight_decay=weight_decay, use_wandb=use_wandb, wandb_group=wandb_group,
        )

    def do_train(self, gen_out: GenOut) -> None:
        hs, ps, y = gen_out
        loss = self.ccr.train(hs[0], hs[1], y)
        self.weight = self.ccr.probe._modules['0'].weight.data.squeeze()
        self.ccr = None
        return loss

    def _get_direction(self) -> Direction:
        return self.weight.cpu()
