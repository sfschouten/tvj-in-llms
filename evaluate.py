import warnings
from abc import ABC, abstractmethod
import random

import numpy as np
import scipy
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import AgglomerativeClustering

import cvxopt

from tango import Step
from tango.common import DatasetDict, Registrable
from tango.integrations.torch import TorchFormat

import torch
import torch.nn.functional as F

from generate import GenOut
from probe_train import LR, CCS, CCReflection

Probabilities = torch.Tensor
Direction = torch.Tensor
Labels = torch.Tensor
ProbeResults = tuple["BeliefProbe", Labels, Probabilities, Direction]


@Step.register('create_splits')
class CreateSplits(Step[GenOut]):
    VERSION = "001"
    CACHEABLE = False

    def run(self, gen_out: GenOut, layer_index: int) -> DatasetDict[GenOut]:
        # Very simple train/test split (using the fact that the data is already shuffled)
        hs, ps, y = gen_out
        hs = hs[..., layer_index]

        hs_train, hs_test = hs[:, :hs.shape[1] // 2], hs[:, hs.shape[1] // 2:]
        ps_train, ps_test = ps[:, :ps.shape[1] // 2], ps[:, ps.shape[1] // 2:]
        y_train, y_test = y[:len(y) // 2], y[len(y) // 2:]

        return DatasetDict(splits={
            'train': (hs_train, ps_train, y_train),
            'eval': (hs_test, ps_test, y_test),
        })


@Step.register('normalize')
class Normalize(Step[DatasetDict[GenOut]]):
    VERSION = "001"

    def run(self, data: DatasetDict[GenOut], var_normalize: bool) -> DatasetDict[GenOut]:
        """
        Mean-normalizes the data x (of shape (n, d))
        If var_normalize, also divides by the standard deviation
        """
        def normalize(x):
            x2 = x.clone()
            x2[~torch.isfinite(x2)] = float('nan')
            normalized_x = x - torch.nanmean(x2, dim=0, keepdims=True)
            if var_normalize:
                normalized_x /= normalized_x.std(dim=0, keepdims=True)
            return normalized_x

        result = {}
        for split in data.keys():
            hs, ps, y = data[split]
            hs_normalized = torch.stack((normalize(hs[0]), normalize(hs[1])))
            result[split] = (hs_normalized, ps, y)

        return DatasetDict(splits=result)


@Step.register('combine_train_data')
class CombineTrainData(Step[DatasetDict[GenOut]]):

    def run(self, data: list[tuple[int, bool, DatasetDict[GenOut]]]) -> DatasetDict[GenOut]:
        new_hs = []
        new_ps = []
        new_y = []
        for nr_to_sample, supervised, data_part in data:
            hs, ps, y = data_part['train']
            sample_idx = torch.randperm(hs.shape[1])
            sample_hs = hs[:, sample_idx[:nr_to_sample]]
            sample_ps = ps[:, sample_idx[:nr_to_sample]]
            sample_y = y[sample_idx[:nr_to_sample]] if supervised \
                else torch.full((nr_to_sample,), torch.nan, dtype=torch.bool)
            new_hs.append(sample_hs)
            new_ps.append(sample_ps)
            new_y.append(sample_y)
        new_train_split = torch.cat(new_hs, dim=1), torch.cat(new_ps, dim=1), torch.cat(new_y, dim=0)
        return DatasetDict(splits={'train': new_train_split})


@Step.register('print_rank')
class PrintRank(Step[None]):

    def run(self, gen_out: GenOut) -> dict[tuple[str, int], int]:
        hs, _, _ = gen_out
        ranks = {}
        for layer in range(hs.shape[-1]):
            neg_hs, pos_hs = hs[0, :, :, layer], hs[1, :, :, layer]
            mats = [('neg', neg_hs), ('pos', pos_hs), ('neg+pos', neg_hs + pos_hs), ('neg-pos', neg_hs - pos_hs)]
            for name, matrix in mats:
                ranks[(name, layer)] = np.linalg.matrix_rank(matrix)
                print(f'layer {layer} [{name}] has rank {ranks[(name, layer)]}')
        return ranks


class BeliefProbe(Registrable, ABC):

    @abstractmethod
    def train(self, gen_out: GenOut) -> float:
        raise NotImplementedError

    def eval(self, eval_data: GenOut) -> tuple[Probabilities, float]:
        hs, ps, y = eval_data
        neg_hs, pos_hs = hs[0], hs[1]

        theta = F.normalize(self.get_direction(), dim=0)
        neg_ps = torch.sigmoid(neg_hs @ theta)
        pos_ps = torch.sigmoid(pos_hs @ theta)
        ps = (pos_ps + (1 - neg_ps)) / 2
        acc = ((ps > 0.5) == y).float().mean()

        print(f'Accuracy: {acc}')
        return ps, acc

    @abstractmethod
    def get_direction(self) -> Direction:
        raise NotImplementedError

    def calc_calibration_params(self, train_data: GenOut, calibration_data: GenOut, scale: float = 1):
        _, _, y = calibration_data
        probs, _ = self.eval(calibration_data)
        logits = torch.logit(probs, eps=1e-6)

        # to scale logits
        std = torch.std(logits)
        self.scale = scale * torch.logit(torch.tensor(0.75), eps=1e-6) / std

        # to flip predictions in favor of probe
        _, acc = self.eval(train_data)
        self.sign = -1 if acc < 0.5 else 1

        #
        hs, _, y = train_data
        if torch.all(torch.isfinite(y)):        # skip for combined training data
            y = y.unsqueeze(0).unsqueeze(-1).expand(-1, -1, hs.shape[-1])
            true_hs = torch.gather(hs, dim=0, index=y).squeeze()
            false_hs = torch.gather(hs, dim=0, index=1-y).squeeze()
            true_u = true_hs.mean(dim=0)
            false_u = false_hs.mean(dim=0)
            self.length = (true_u - false_u).norm()
            print(f'true-false distance: {self.length}')

    def calibrate_logits(self, logits):
        new = self.sign * logits
        if torch.isfinite(self.scale).item():
            new *= self.scale
        return new


@Step.register('train_belief_probe')
class TrainBeliefProbe(Step[BeliefProbe]):
    VERSION = "023"

    def run(
        self,  train_data: DatasetDict[GenOut], calibration_data: DatasetDict[GenOut], probe: BeliefProbe, **kwargs
    ) -> BeliefProbe:
        probe.train_loss = probe.train(train_data['train'])
        direction = probe.get_direction()
        if direction is not None:
            assert torch.all(~torch.isnan(direction))

        probe.calc_calibration_params(train_data['train'], calibration_data['train'])
        probe.config = self.config['probe']
        probe.train_step_name = self.name
        return probe


@Step.register('select_best')
class SelectBest(Step[list[BeliefProbe]]):
    VERSION = "002"

    def run(self, probes, **kwargs) -> BeliefProbe:
        best_probe = min(probes, key=lambda p: p.train_loss)
        best_probe.train_step_name = self.name
        return best_probe


@Step.register('eval_belief_probe')
class EvalBeliefProbe(Step[ProbeResults]):
    FORMAT = TorchFormat
    VERSION = "010"

    def run(self,  data: DatasetDict[GenOut], probe: BeliefProbe, **kwargs) -> ProbeResults:
        probs, acc = probe.eval(data['eval'])

        # calibrate predictions
        logits = torch.logit(probs, eps=1e-6)
        logits = probe.calibrate_logits(logits)
        new_probs = torch.sigmoid(logits)

        assert torch.all(~torch.isnan(new_probs))
        return probe, data['eval'][2], new_probs, probe.get_direction()


@BeliefProbe.register('lm_head_baseline')
class LMHeadBaseline(BeliefProbe):
    """
    Baseline: use the language modeling logits for the correct completions to obtain probabilities.
    """

    def __init__(self, calibrate: bool):
        super().__init__()
        self.calibrate = calibrate

    def train(self, gen_out: GenOut) -> None:
        pass

    def eval(self, gen_out: GenOut) -> tuple[Probabilities, float]:
        _, ps, y = gen_out
        neg_ps, pos_ps = tuple(x.squeeze() for x in ps.split(1))

        def normalize_preds(pos, neg):
            pred = pos - neg
            f = max(-pred.min(), pred.max())
            return 0.5 + 0.5 * pred / f

        if self.calibrate:
            neg_ps -= np.median(neg_ps - pos_ps)
            acc = ((pos_ps > neg_ps) == y).float().mean()
            pred = normalize_preds(pos_ps, neg_ps)
            print(f'LM Head calibrated acc: {acc}')
        else:
            acc = ((pos_ps > neg_ps) == y).float().mean()
            pred = normalize_preds(pos_ps, neg_ps)
            print(f'LM Head acc: {acc}')

        return pred, acc

    def get_direction(self) -> Direction:
        return None


@BeliefProbe.register('lr_sklearn')
class LogisticRegressionSKLearn(BeliefProbe):
    """
    Baseline: fit a logistic model to the test data to serve as an upper bound.
    SKLearn version.
    """

    def __init__(self, class_weight: str = 'balanced'):
        super().__init__()
        self.lr = LogisticRegression(class_weight=class_weight, fit_intercept=False)

    def train(self, train_data: GenOut) -> None:
        hs, ps, y = train_data
        neg_hs, pos_hs = hs[0], hs[1]
        x = neg_hs - pos_hs
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=ConvergenceWarning)
            self.lr.fit(x, y)

    def get_direction(self) -> Direction:
        return torch.tensor(self.lr.coef_).squeeze().float().cpu()


@BeliefProbe.register('lr_gd')
class LogisticRegressionGradientDescent(BeliefProbe):
    """
    Baseline: fit a logistic model to the test data to serve as an upper bound.
    Gradient descent version.
    """

    def __init__(self, seed: int, n_epochs=1000, n_tries=10, lr=1e-3, batch_size=-1, verbose=False, device="cuda",
                 weight_decay=0.01, use_wandb=False, wandb_group=None):
        super().__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.device = device
        self.lr = LR(
            n_epochs=n_epochs, n_tries=n_tries, lr=lr, batch_size=batch_size, verbose=verbose, device=device,
            weight_decay=weight_decay, use_wandb=use_wandb, wandb_group=wandb_group
        )

    def train(self, gen_out: GenOut) -> None:
        hs, ps, y = gen_out
        return self.lr.train(hs[0], hs[1], y)

    def get_direction(self) -> Direction:
        return self.lr.probe._modules['0'].weight.data.squeeze().cpu()


@BeliefProbe.register('ccs_gd')
class ContrastConsistentSearchGradientDescent(BeliefProbe):

    def __init__(self, seed: int, n_epochs=1000, lr=1e-3, batch_size=-1, verbose=False, device="cuda",
                 weight_decay=0.01, use_wandb=False, wandb_group=None, use_constraint=False, informative_loss='min_sq'):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed
        self.ccs = CCS(
            n_epochs=n_epochs, lr=lr, batch_size=batch_size, verbose=verbose, device=device,
            weight_decay=weight_decay, use_wandb=use_wandb, wandb_group=wandb_group, use_constraint=use_constraint,
            informative_loss=informative_loss
        )

    def train(self, gen_out: GenOut) -> None:
        hs, ps, y = gen_out
        return self.ccs.train(hs[0], hs[1], y)

    def get_direction(self) -> Direction:
        return self.ccs.probe._modules['0'].weight.data.squeeze().cpu()


@BeliefProbe.register('ccs_linear')
class ContrastConsistentSearchLinearized(BeliefProbe):

    def __init__(self, priority: float = 5):
        super().__init__()
        self.priority = priority

    def train(self, gen_out: GenOut) -> None:
        hs, _, _ = gen_out
        B, A = hs[0].numpy(), hs[1].numpy()

        # Variant
        AmB = A - B

        U, s, Vt = np.linalg.svd(AmB)
        rank = np.sum(s >= s.max() * len(s) * np.finfo(s.dtype).eps)

        A1 = U.T[:rank] @ A @ Vt[:rank].T
        B1 = U.T[:rank] @ B @ Vt[:rank].T
        ApB1 = A1 + B1

        # rescale
        # scale up the errors of pairs that are close and short
        A1_norm = np.linalg.norm(A1, axis=1, keepdims=True)
        B1_norm = np.linalg.norm(B1, axis=1, keepdims=True)
        # exponent = -np.sum(A1 * B1, axis=1, keepdims=True) / (A1_norm * B1_norm).mean()
        dot = np.sum(A1 * B1, axis=1, keepdims=True) / (A1_norm * B1_norm)
        # exponent = -np.sum(A1 * B1, axis=1, keepdims=True)
        ApB1 /= (A1_norm + B1_norm) ** (self.priority * dot)
        # ApB1 /= 2**(-np.sum(A1 * B1, axis=1, keepdims=True) / (A1_norm * B1_norm).mean())
        # ApB1 /= 2**((dot - dot.min()) / dot.max())

        C = ApB1
        result = scipy.optimize.lsq_linear(
            C.astype(np.double), np.zeros((C.shape[0],), dtype=np.double),
            bounds=(np.finfo(np.double).eps, np.inf)
            # bounds=(1., np.inf)
        )
        self.theta = result.x @ Vt[:len(result.x)]

    def get_direction(self) -> Direction:
        return torch.tensor(self.theta).squeeze()


@BeliefProbe.register('ccs_convex')
class ContrastConsistentSearchConvex(BeliefProbe):

    def train(self, gen_out: GenOut) -> None:
        hs, _, _ = gen_out
        Xp, Xn = hs[0].double(), hs[1].double()

        def obj(w):
            y = (1 - torch.sigmoid(Xp @ w) + torch.sigmoid(Xn @ w)) ** 2
            return y.sum()

        def c1(w):
            return torch.min((Xp - Xn) @ w)**2 - 1

        # def c2(w):
        #     return (Xn @ w)**2 - 1

        def F(w=None, z=None):
            if w is None:
                return 1, cvxopt.matrix(np.ones(Xp.shape[1]))

            w_t = torch.tensor(np.array(w))
            f = cvxopt.matrix(np.array([
                obj(w_t).item(),
                c1(w_t).item(),
                # c2(w_t).float().numpy()
            ]))
            Df = cvxopt.matrix(np.concatenate([
                torch.func.grad(obj)(w_t).T.numpy(),
                torch.func.grad(c1)(w_t).T.numpy(),
            ]))

            if z is None:
                return f, Df

            H = cvxopt.matrix((
                z[0] * torch.func.hessian(obj)(w_t.squeeze())
                + z[1] * torch.func.hessian(c1)(w_t.squeeze())
                # + z[3] * torch.func.hessian(c2)(torch.tensor(w)),
            ).numpy())

            return f, Df, H

        result = cvxopt.solvers.cp(F)
        self.theta = 0

    def get_direction(self) -> Direction:
        return torch.tensor(self.theta).squeeze()


@BeliefProbe.register('ccr')
class ContrastConsistentReflection(BeliefProbe):

    def __init__(self, seed: int, n_epochs=1000, lr=1e-3, batch_size=-1, verbose=False, device="cuda",
                 weight_decay=0.01, use_wandb=False, wandb_group=None):
        super().__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed
        self.ccr = CCReflection(
            n_epochs=n_epochs, lr=lr, batch_size=batch_size, verbose=verbose, device=device,
            weight_decay=weight_decay, use_wandb=use_wandb, wandb_group=wandb_group,
        )

    def train(self, gen_out: GenOut) -> None:
        hs, ps, y = gen_out
        return self.ccr.train(hs[0], hs[1], y)

    def get_direction(self) -> Direction:
        return self.ccr.probe._modules['0'].weight.data.squeeze().cpu()


@BeliefProbe.register('mass_mean')
class MassMeanProbe(BeliefProbe):

    def __init__(self, iid):
        super().__init__()
        self.iid = iid

        self.theta = None
        self.tilt_mat = None

    def train(self, gen_out: GenOut) -> None:
        hs, _, y = gen_out

        y = y.unsqueeze(0).unsqueeze(-1).expand(-1, -1, hs.shape[-1])
        true_hs = torch.gather(hs, dim=0, index=y).squeeze()
        false_hs = torch.gather(hs, dim=0, index=1-y).squeeze()

        # take mean of pos and neg and store the difference as the 'truth direction'
        true_u = true_hs.mean(dim=0)
        false_u = false_hs.mean(dim=0)
        self.theta = true_u - false_u

        if self.iid:
            # independently center pos and neg and calculate the inverse of the covariance matrix
            true_hs_c = true_hs - true_u
            false_hs_c = false_hs - false_u
            Sigma = torch.cov(torch.cat([true_hs_c, false_hs_c]).T)
            # d = torch.cat([neg_hs_c, pos_hs_c])
            # Sigma = d.T @ d / d.shape[0]
            # self.tilt_mat = torch.linalg.inv(Sigma)
            self.tilt_mat = torch.linalg.pinv(Sigma, hermitian=True, atol=1e-3)

    def get_direction(self) -> Direction:
        if self.iid:
            return (self.tilt_mat @ self.theta).squeeze().cpu()
        else:
            return self.theta.squeeze().cpu()


@BeliefProbe.register('unsupervised_mass_mean')
class UnsupervisedMassMeanProbe(BeliefProbe):
    VERSION = "002"

    def __init__(self, linkage: str = 'complete'):
        super().__init__()
        self.theta = None
        self.clustering = AgglomerativeClustering(n_clusters=2, metric='cosine', linkage=linkage)

    def train(self, gen_out: GenOut) -> None:
        hs, _, _ = gen_out

        # take difference between positive and negative hidden states
        deltas = hs[1] - hs[0]

        # the deltas true-false and false-true should now point in opposite directions
        # we use clustering to recover these two groups
        y = torch.tensor(self.clustering.fit_predict(deltas.numpy())).bool()

        # the final direction is simply the (mean of true-false) - (mean of false-true)
        self.theta = (deltas[y].mean(dim=0) - deltas[~y].mean(dim=0)) / 2

    def get_direction(self) -> Direction:
        return self.theta.squeeze().cpu()
