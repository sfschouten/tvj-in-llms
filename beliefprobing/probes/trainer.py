import copy
from abc import abstractmethod

import torch
from torch import nn


def default_probe(dim):
    return nn.Sequential(
        torch.nn.utils.parametrizations.weight_norm(
            nn.Linear(dim, 1, bias=False)
        ), nn.Sigmoid()
    )


class Trainer:
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
                for k in ['n_epochs', 'n_tries', 'lr', 'device', 'batch_size', 'weight_decay', 'd']}
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
