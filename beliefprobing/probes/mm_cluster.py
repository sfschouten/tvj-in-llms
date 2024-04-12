import torch

from sklearn.cluster import AgglomerativeClustering

from beliefprobing.probes.beliefprobe import BeliefProbe, Direction
from beliefprobing.generate import GenOut


@BeliefProbe.register('unsupervised_mass_mean')
class UnsupervisedMassMeanProbe(BeliefProbe):
    VERSION = "002"

    def __init__(self, linkage: str = 'complete'):
        super().__init__(svd_data=False)
        self.theta = None
        self.clustering = AgglomerativeClustering(n_clusters=2, metric='cosine', linkage=linkage)

    def do_train(self, gen_out: GenOut) -> None:
        hs, _, _ = gen_out

        # take difference between positive and negative hidden states
        deltas = hs[1] - hs[0]

        # the deltas true-false and false-true should now point in opposite directions
        # we use clustering to recover these two groups
        y = torch.tensor(self.clustering.fit_predict(deltas.numpy())).bool()

        # the final direction is simply the (mean of true-false) - (mean of false-true)
        self.theta = (deltas[y].mean(dim=0) - deltas[~y].mean(dim=0)) / 2

    def _get_direction(self) -> Direction:
        return self.theta.squeeze().cpu()
