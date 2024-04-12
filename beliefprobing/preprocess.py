from tango import Step
from tango.common import DatasetDict, Registrable

import torch
import numpy as np

from beliefprobing.generate import GenOut
from beliefprobing import BeliefProbe

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


@BeliefProbe.register('lm_head_baseline')
class LMHeadBaseline(BeliefProbe):
    """
    Baseline: use the language modeling logits for the correct completions to obtain probabilities.
    """

    def __init__(self, calibrate: bool):
        super().__init__(svd_data=False)
        self.calibrate = calibrate

    def do_train(self, gen_out: GenOut) -> None:
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

        return pred, acc.item()

    def _get_direction(self) -> Direction:
        return None

