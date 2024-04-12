import torch

from beliefprobing.probes.beliefprobe import BeliefProbe, Direction
from beliefprobing.generate import GenOut


@BeliefProbe.register('mass_mean')
class MassMeanProbe(BeliefProbe):

    def __init__(self, iid, svd_data=False):
        super().__init__(svd_data)
        self.iid = iid

        self.theta = None
        self.tilt_mat = None

    def do_train(self, gen_out: GenOut) -> None:
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

    def _get_direction(self) -> Direction:
        if self.iid:
            return (self.tilt_mat @ self.theta).squeeze().cpu()
        else:
            return self.theta.squeeze().cpu()
