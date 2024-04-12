import numpy as np
import scipy
import torch

from beliefprobing.probes.beliefprobe import BeliefProbe, Direction
from beliefprobing.generate import GenOut


@BeliefProbe.register('ccs_linear')
class ContrastConsistentSearchLinearized(BeliefProbe):

    def __init__(self, priority: float = 5):
        super().__init__(svd_data=True)
        # self.priority = priority
        self.theta = None

    def do_train(self, gen_out: GenOut) -> None:
        hs, _, _ = gen_out
        N, P = hs[0].numpy(), hs[1].numpy()

        NpP = N + P

        # rescale
        # scale up the errors of pairs that are close and short
        # A1_norm = np.linalg.norm(A1, axis=1, keepdims=True)
        # B1_norm = np.linalg.norm(B1, axis=1, keepdims=True)
        # exponent = -np.sum(A1 * B1, axis=1, keepdims=True) / (A1_norm * B1_norm).mean()
        # dot = np.sum(A1 * B1, axis=1, keepdims=True) / (A1_norm * B1_norm)
        # exponent = -np.sum(A1 * B1, axis=1, keepdims=True)
        # ApB1 /= (A1_norm + B1_norm) ** (self.priority * dot)
        # ApB1 /= 2**(-np.sum(A1 * B1, axis=1, keepdims=True) / (A1_norm * B1_norm).mean())
        # ApB1 /= 2**((dot - dot.min()) / dot.max())

        result = scipy.optimize.lsq_linear(
            NpP.astype(np.double), np.zeros((NpP.shape[0],), dtype=np.double),
            bounds=(np.finfo(np.double).eps, np.inf)
        )
        self.theta = result.x

    def _get_direction(self) -> Direction:
        return torch.tensor(self.theta).squeeze()
