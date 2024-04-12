import random
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from beliefprobing.probes.trainer import Trainer
from beliefprobing.probes.beliefprobe import BeliefProbe, Direction
from beliefprobing.generate import GenOut


@BeliefProbe.register('lr_sklearn')
class LogisticRegressionSKLearn(BeliefProbe):
    """
    Baseline: fit a logistic model to the test data to serve as an upper bound.
    SKLearn version.
    """

    def __init__(self, class_weight: str = 'balanced'):
        super().__init__(svd_data=False)
        self.lr = LogisticRegression(class_weight=class_weight, fit_intercept=False, tol=1e-5)

    def do_train(self, train_data: GenOut) -> None:
        hs, ps, y = train_data
        neg_hs, pos_hs = hs[0], hs[1]
        x = neg_hs - pos_hs
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=ConvergenceWarning)
            self.lr.fit(x, y)

        assert self.lr.n_iter_[0] > 0

    def _get_direction(self) -> Direction:
        return torch.tensor(self.lr.coef_).squeeze().float().cpu()


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


@BeliefProbe.register('lr_gd')
class LogisticRegressionGradientDescent(BeliefProbe):
    """
    Baseline: fit a logistic model to the test data to serve as an upper bound.
    Gradient descent version.
    """

    def __init__(self, seed: int, n_epochs=1000, n_tries=10, lr=1e-3, batch_size=-1, verbose=False, device="cuda",
                 weight_decay=0.01, use_wandb=False, wandb_group=None):
        super().__init__(svd_data=False)

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.device = device
        self.lr = LR(
            n_epochs=n_epochs, n_tries=n_tries, lr=lr, batch_size=batch_size, verbose=verbose, device=device,
            weight_decay=weight_decay, use_wandb=use_wandb, wandb_group=wandb_group
        )

    def do_train(self, gen_out: GenOut) -> None:
        hs, ps, y = gen_out
        return self.lr.train(hs[0], hs[1], y)

    def _get_direction(self) -> Direction:
        return self.lr.probe._modules['0'].weight.data.squeeze().cpu()

