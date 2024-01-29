import warnings
from abc import ABC, abstractmethod
import random

import numpy as np
import scipy
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import AgglomerativeClustering

from tango import Step
from tango.common import DatasetDict, Registrable
from tango.integrations.torch import TorchFormat

import torch

from generate import GenOut
from probe_train import LR, CCS, CCReflection

Probabilities = torch.Tensor
Direction = torch.Tensor
Labels = torch.Tensor
ProbeResults = tuple[Labels, Probabilities, Direction]


@Step.register('create_splits')
class CreateSplits(Step[GenOut]):

    def run(self, gen_out: GenOut, layer_index: int = 0) -> DatasetDict[GenOut]:
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

    def run(self, data: DatasetDict[GenOut], var_normalize: bool) -> DatasetDict[GenOut]:
        """
        Mean-normalizes the data x (of shape (n, d))
        If var_normalize, also divides by the standard deviation
        """
        def normalize(x):
            x2 = x.clone()
            x2[~torch.isfinite(x2)] = float('nan')
            normalized_x = x - torch.nanmean(x2, axis=0, keepdims=True)
            if var_normalize:
                normalized_x /= normalized_x.std(axis=0, keepdims=True)
            return normalized_x

        result = {}
        for split in data.keys():
            hs, ps, y = data[split]
            hs_normalized = torch.stack((normalize(hs[0]), normalize(hs[1])))
            result[split] = (hs_normalized, ps, y)

        return DatasetDict(splits=result)


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
    def train(self, gen_out: GenOut) -> None:
        raise NotImplementedError

    @abstractmethod
    def eval(self, gen_out: GenOut) -> Probabilities:
        raise NotImplementedError

    @abstractmethod
    def get_direction(self) -> Direction:
        raise NotImplementedError

    def calc_calibration_scalar(self, data: GenOut):
        hs, _, _ = data

        probs = self.eval(data)
        abs_logits = torch.logit(probs).abs()
        pth80 = abs_logits.quantile(q=.8)

        # to scale logits such that probability for 80pth is 80%
        self.calibration_scalar = torch.logit(torch.tensor(0.8)) / pth80


@Step.register('train_belief_probe')
class TrainBeliefProbe(Step[BeliefProbe]):
    VERSION = "002"

    def run(self,  data: DatasetDict[GenOut], probe: BeliefProbe, **kwargs) -> BeliefProbe:
        probe.train(data['train'])
        probe.calc_calibration_scalar(data['train'])
        return probe


@Step.register('eval_belief_probe')
class EvalBeliefProbe(Step[ProbeResults]):
    FORMAT = TorchFormat
    VERSION = "004"

    def run(self,  data: DatasetDict[GenOut], probe: BeliefProbe, **kwargs) -> ProbeResults:
        probs = probe.eval(data['eval'])

        # calibrate predictions
        logits = torch.logit(probs)
        logits *= probe.calibration_scalar
        new_probs = torch.sigmoid(logits)

        return data['eval'][2], new_probs, probe.get_direction()


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

    def eval(self, gen_out: GenOut) -> Probabilities:
        _, ps, y = gen_out
        neg_ps, pos_ps = tuple(x.squeeze() for x in ps.split(1))

        def normalize_preds(pos, neg):
            pred = pos - neg
            f = max(-pred.min(), pred.max())
            return 0.5 + 0.5 * pred / f

        if self.calibrate:
            neg_ps -= np.median(neg_ps - pos_ps)
            lm_cal_acc = ((pos_ps > neg_ps) == y).float().mean()
            pred = normalize_preds(pos_ps, neg_ps)
            print(f'LM Head calibrated acc: {lm_cal_acc}')
        else:
            lm_acc = ((pos_ps > neg_ps) == y).float().mean()
            pred = normalize_preds(pos_ps, neg_ps)
            print(f'LM Head acc: {lm_acc}')

        return pred

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
        self.lr = LogisticRegression(class_weight=class_weight)

    def train(self, train_data: GenOut) -> None:
        hs, ps, y = train_data
        neg_hs, pos_hs = hs[0], hs[1]
        x = neg_hs - pos_hs
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=ConvergenceWarning)
            self.lr.fit(x, y)

    def eval(self, eval_data: GenOut) -> Probabilities:
        hs, ps, y = eval_data
        neg_hs, pos_hs = hs[0], hs[1]
        x = neg_hs - pos_hs
        acc = self.lr.score(x, y)
        print(f'logistic regression (sklearn) accuracy: {acc}')
        return torch.tensor(self.lr.predict_proba(x)[:, 1])

    def get_direction(self) -> Direction:
        return torch.tensor(self.lr.coef_).squeeze()


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
        self.lr.train(hs[0], hs[1], y)

    def eval(self, gen_out: GenOut) -> Probabilities:
        hs, ps, y = gen_out
        acc, result = self.lr.get_acc(hs[0], hs[1], y)
        print(f'logistic regression (gradient descent) accuracy: {acc}')
        return result

    def get_direction(self) -> Direction:
        return self.lr.probe._modules['0'].weight.data.squeeze()


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
        self.ccs.train(hs[0], hs[1], y)

    def eval(self, gen_out: GenOut) -> Probabilities:
        hs, ps, y = gen_out
        acc, result = self.ccs.get_acc(hs[0], hs[1], y)
        print(f'ccs (gradient descent) accuracy: {acc}')
        return result

    def get_direction(self) -> Direction:
        return self.ccs.probe._modules['0'].weight.data.squeeze()


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

    def eval(self, gen_out: GenOut) -> Probabilities:
        hs, _, y = gen_out
        neg_hs, pos_hs = hs[0], hs[1]

        pred = (neg_hs @ self.theta - pos_hs @ self.theta) > 0

        acc = (pred == y).float().mean()
        print(f'ccs_linear accuracy: {acc}')

        return pred

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
        self.ccr.train(hs[0], hs[1], y)

    def eval(self, gen_out: GenOut) -> Probabilities:
        hs, ps, y = gen_out
        acc, result = self.ccr.get_acc(hs[0], hs[1], y)
        print(f'ccr (gradient descent) accuracy: {acc}')
        return result

    def get_direction(self) -> Direction:
        return self.ccr.probe._modules['0'].weight.data.squeeze()


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
        true_hs = torch.gather(hs, dim=1, index=y).squeeze()
        false_hs = torch.gather(hs, dim=1, index=1-y).squeeze()

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

    def eval(self, gen_out: GenOut) -> Probabilities:
        hs, _, y = gen_out
        neg_hs, pos_hs = hs[0], hs[1]

        theta = self.get_direction()

        # neg_pred = torch.sigmoid(neg_hs @ theta)
        # pos_pred = 1 - torch.sigmoid(pos_hs @ theta)
        # pred = ((neg_pred + pos_pred) / 2) > 0.5
        logits = ((pos_hs @ theta - neg_hs @ theta) / 2)
        ps = torch.sigmoid(logits)
        pred = ps > 0

        acc = (pred == y).float().mean()
        print(f'mass-mean (iid={self.iid}) accuracy: {acc}')

        return ps

    def get_direction(self) -> Direction:
        if self.iid:
            return (self.tilt_mat @ self.theta).squeeze()
        else:
            return self.theta.squeeze()


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

    def eval(self, gen_out: GenOut) -> Probabilities:
        hs, _, y = gen_out
        neg_hs, pos_hs = hs[0], hs[1]

        pred = ((pos_hs @ self.theta - neg_hs @ self.theta) / 2)

        acc = ((pred > 0) == y).float().mean()
        print(f'unsupervised mass-mean accuracy: {acc}')

        return torch.sigmoid(pred)

    def get_direction(self) -> Direction:
        return self.theta.squeeze()
