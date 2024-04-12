from abc import ABC, abstractmethod

from tango import Step
from tango.common import DatasetDict, Registrable
from tango.integrations.torch import TorchFormat

import torch
import torch.nn.functional as F

from beliefprobing.generate import GenOut

Probabilities = torch.Tensor
Direction = torch.Tensor
Labels = torch.Tensor

ProbeResults = tuple["BeliefProbe", Labels, Probabilities, Direction]


class BeliefProbe(Registrable, ABC):

    def __init__(self, svd_data: bool):
        self.perform_svd_data = svd_data
        self.M = None

    def svd_data(self, gen_out: GenOut) -> GenOut:
        hs, ps, y = gen_out      # 2 x N x D
        N, P = hs[0], hs[1]

        # ker(N - P) should be ker(N) ^ ker(P)?
        NmP = N - P

        U, s, Vt = torch.linalg.svd(NmP)
        rank = torch.sum(s >= s.max() * len(s) * torch.finfo(s.dtype).eps)
        self.M = torch.diag(s[:rank]) @ Vt[:rank]

        hs @= self.M.T
        return hs, ps, y

    def train(self, gen_out: GenOut):
        if self.perform_svd_data:
            gen_out = self.svd_data(gen_out)
        return self.do_train(gen_out)

    @abstractmethod
    def do_train(self, gen_out: GenOut) -> float:
        raise NotImplementedError

    def eval(self, eval_data: GenOut) -> tuple[Probabilities, float]:
        hs, ps, y = eval_data
        neg_hs, pos_hs = hs[0], hs[1]

        theta = F.normalize(self.direction, dim=0)
        neg_ps = torch.sigmoid(neg_hs @ theta)
        pos_ps = torch.sigmoid(pos_hs @ theta)
        ps = (pos_ps + (1 - neg_ps)) / 2
        acc = ((ps > 0.5) == y).float().mean().item()

        print(f'Accuracy: {acc}')
        return ps, acc

    @property
    def direction(self):
        theta = self._get_direction()
        if theta is not None and self.perform_svd_data:
            theta @= self.M
        return theta

    @abstractmethod
    def _get_direction(self) -> Direction:
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
        if not torch.all(torch.isfinite(y)):
            hs = hs[:, torch.isfinite(y)]
            y = y[torch.isfinite(y)]

        y = y.unsqueeze(0).unsqueeze(-1).expand(-1, -1, hs.shape[-1])
        true_hs = torch.gather(hs, dim=0, index=y).squeeze()
        false_hs = torch.gather(hs, dim=0, index=1-y).squeeze()
        true_u = true_hs.mean(dim=0)
        false_u = false_hs.mean(dim=0)
        mm_dir = true_u - false_u
        self.length = mm_dir.norm()
        pr_dir = self.direction
        if pr_dir is not None:      # use mass mean vector to obtain sign for all, hopefully more consistent than acc
            dot = torch.dot(mm_dir, pr_dir)
            self.sign = int(dot / abs(dot)) if abs(dot) > 0 else 1
        print(f'true-false distance: {self.length}')

    def calibrate_logits(self, logits):
        new = self.sign * logits
        if torch.isfinite(self.scale).item():
            new *= self.scale
        return new


@Step.register('train_belief_probe')
class TrainBeliefProbe(Step[BeliefProbe]):
    VERSION = "023a"

    def run(self, train_data: DatasetDict[GenOut], cal_data: DatasetDict[GenOut], probe: BeliefProbe) -> BeliefProbe:
        probe.train_loss = probe.train(train_data['train'])
        direction = probe.direction
        if direction is not None:
            assert torch.all(~torch.isnan(direction)) and torch.any(direction > 0.), direction

        probe.calc_calibration_params(train_data['train'], cal_data['train'])
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

    def run(self,  data: DatasetDict[GenOut], probe: BeliefProbe) -> ProbeResults:
        probs, acc = probe.eval(data['eval'])

        # calibrate predictions
        logits = torch.logit(probs, eps=1e-6)
        logits = probe.calibrate_logits(logits)
        new_probs = torch.sigmoid(logits)
        new_acc = ((new_probs > 0.5) == data['eval'][2]).float().mean()
        print(f'Calibrated accuracy: {new_acc}')

        assert torch.all(~torch.isnan(new_probs))
        return probe, data['eval'][2], new_probs, probe.direction
