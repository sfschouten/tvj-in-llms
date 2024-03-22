"""
Code to generate the hidden states on which we'll train the various probing methods.
"""
import random

import numpy
from tango import Step
from tango.integrations.torch import Model, TorchFormat
from tango.integrations.transformers import Tokenizer
from tango.common.det_hash import det_hash

from tqdm import tqdm

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from integrations import TupleFormat


def get_first_mask_loc(mask, shift=False):
    """
    return the location of the first pad token for the given ids, which corresponds to a mask value of 0
    if there are no pad tokens, then return the last location
    """
    # add a 0 to the end of the mask in case there are no pad tokens
    mask = torch.cat([mask, torch.zeros_like(mask[..., :1])], dim=-1)

    if shift:
        mask = mask[..., 1:]

    # get the location of the first pad token; use the fact that torch.argmax() returns the first index in the case of ties
    first_mask_loc = torch.argmax((mask == 0).int(), dim=-1)

    return first_mask_loc


def get_answer_logits(batch_ids, output, answer_tokens, model_type, use_decoder=True, nr_tokens=None):
    input_ids = batch_ids['input_ids']
    logits = output['logits']
    B, S, V = logits.size()
    device = logits.device
    off = -1 if model_type == 'decoder' or (model_type == 'encoder_decoder' and use_decoder) else 0

    answer_tokens = answer_tokens.to(device)  # B, B
    T = answer_tokens.size(1) if nr_tokens is None else nr_tokens

    # for each sample in the batch we want to know what probability was predicted for the label of the token_idx token
    tokens = input_ids.index_select(index=answer_tokens.flatten(), dim=1).view(B, B, T).diagonal().transpose(0, -1)
    logits = logits.index_select(index=answer_tokens.flatten()+off, dim=1).view(B, B, T, V).diagonal().permute(2, 0, 1)

    # index tokens
    logits = logits.index_select(index=tokens.flatten(), dim=-1)  # B x nr_tokens x B*nr_tokens
    logits = logits.view(B * T, B * T).diagonal().view(B, T)  # B x nr_tokens
    logits = logits.mean(dim=1)
    return logits


def get_individual_hidden_states(model, batch_ids, answer_tokens, layer=None, all_layers=True, token_idx=-1,
                                 all_tokens=False, model_type="encoder_decoder", use_decoder=False):
    """
    Given a model and a batch of tokenized examples, returns the hidden states for either
    a specified layer (if layer is a number) or for all layers (if all_layers is True).

    If specify_encoder is True, uses "encoder_hidden_states" instead of "hidden_states"
    This is necessary for getting the encoder hidden states for encoder-decoder models,
    but it is not necessary for encoder-only or decoder-only models.
    """
    if use_decoder:
        assert "decoder" in model_type

    # forward pass
    with torch.no_grad():
        batch_ids = batch_ids.to(model.device)
        output = model(**batch_ids, output_hidden_states=True)

    # get all the corresponding hidden states (which is a tuple of length num_layers)
    if use_decoder and "decoder_hidden_states" in output.keys():
        hs_tuple = output["decoder_hidden_states"]
    elif "encoder_hidden_states" in output.keys():
        hs_tuple = output["encoder_hidden_states"]
    else:
        hs_tuple = output["hidden_states"]

    # just get the corresponding layer hidden states
    if all_layers:
        # stack along the last axis so that it's easier to consistently index the first two axes
        hs = torch.stack([h.squeeze().detach().cpu() for h in hs_tuple], axis=-1)  # (bs, seq_len, dim, num_layers)
    else:
        assert layer is not None
        hs = hs_tuple[layer].unsqueeze(-1).detach().cpu()  # (bs, seq_len, dim, 1)

    # we want to get the token corresponding to token_idx while ignoring the masked tokens
    if token_idx == 0:
        final_hs = hs[:, 0]  # (bs, dim, num_layers)
    else:
        mask = batch_ids["decoder_attention_mask"] if (model_type == "encoder_decoder" and use_decoder) \
                                                 else batch_ids["attention_mask"]
        first_mask_loc = get_first_mask_loc(mask).cpu()
        if all_tokens:
            hs[mask == 0] = -torch.inf
            final_hs = hs
        else:
            # if token_idx == -1, then takes the hidden states corresponding to the last non-mask tokens
            # first we need to get the first mask location for each example in the batch
            assert token_idx < 0, print("token_idx must be either 0 or negative, but got", token_idx)
            final_hs = hs[torch.arange(hs.size(0)), first_mask_loc + token_idx]  # (bs, dim, num_layers)

    logits = get_answer_logits(batch_ids, output, answer_tokens, model_type, use_decoder=use_decoder)

    assert torch.all(~torch.isnan(final_hs)), 'hidden state contains NaN values'
    return final_hs, logits.cpu()


def get_masked_lm_logits(model, batch_ids, answer_tokens, mask_token_id, model_type):
    input_ids = batch_ids['input_ids'].clone()
    answer_tokens = answer_tokens.to(model.device)  # B x nr_tokens
    token_idx = F.one_hot(answer_tokens, num_classes=input_ids.size(1)).any(dim=1)  # B x S
    input_ids[token_idx] = mask_token_id

    # forward pass
    with torch.no_grad():
        batch_ids = batch_ids.to(model.device)
        output = model(**batch_ids | {'input_ids': input_ids}, output_hidden_states=True)

    logits = get_answer_logits(batch_ids, output, answer_tokens, model_type, use_decoder=False)
    return logits.cpu()


HiddenStates = torch.Tensor
Logits = torch.Tensor
Labels = torch.Tensor
Metadata = list[dict]
GenOut = tuple[HiddenStates, Logits, Labels]


@Step.register('generate_hidden_states')
class Generate(Step[GenOut]):
    FORMAT = TupleFormat(formats=(f := TorchFormat(), f, f))
    DETERMINISTIC = True
    VERSION = "002"
    SKIP_ID_ARGUMENTS = {'batch_size'}

    @property
    def unique_id(self) -> str:
        """Returns the unique ID for this step.
        Customized to circumvent whatever source of non-determinism is present in huggingface model loading.
        """
        if self.unique_id_cache is None:
            self.unique_id_cache = self.class_name
            if self.VERSION is not None:
                self.unique_id_cache += "-"
                self.unique_id_cache += self.VERSION

            self.unique_id_cache += "-"
            if self.DETERMINISTIC:
                hash_kwargs = {key: value for key, value in self.kwargs.items()}

                # don't hash the entire model, just the config should be enough
                hash_kwargs['model'] = hash_kwargs['model'].config

                self.unique_id_cache += det_hash(
                    (
                        (self.format.__class__.__module__, self.format.__class__.__qualname__),
                        self.format.VERSION,
                        hash_kwargs,
                    )
                )[:32]
            if self._UNIQUE_ID_SUFFIX is not None:
                self.unique_id_cache += f"-{self._UNIQUE_ID_SUFFIX}"

        return self.unique_id_cache

    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    @staticmethod
    def create_dataloader(dataset, batch_size, pin_memory, num_workers, seed):
        g = torch.Generator()
        g.manual_seed(seed)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory,
                                num_workers=num_workers, generator=g, worker_init_fn=Generate.seed_worker)
        print(next(iter(dataloader))[2][0][0])
        return dataloader

    def forward_batch(self, model, batch, dataloader, mask_token_id, **kwargs):
        neg_ids, pos_ids, _, _, gt_label, neg_answer_tokens, pos_answer_tokens, _ = batch

        neg_hs, neg_logits = get_individual_hidden_states(model, neg_ids, neg_answer_tokens, **kwargs)
        pos_hs, pos_logits = get_individual_hidden_states(model, pos_ids, pos_answer_tokens, **kwargs)

        model_type = kwargs['model_type']
        if model_type == 'encoder' or (model_type == 'encoder_decoder' and not kwargs['use_decoder']):
            pos_logits = get_masked_lm_logits(model, pos_ids, pos_answer_tokens, mask_token_id, model_type)
            neg_logits = get_masked_lm_logits(model, neg_ids, neg_answer_tokens, mask_token_id, model_type)

        if dataloader.batch_size == 1:
            neg_hs, pos_hs = neg_hs.unsqueeze(0), pos_hs.unsqueeze(0)

        return neg_hs, pos_hs, neg_logits, pos_logits

    def run(self, model: Model, tokenizer: Tokenizer, dataset: Dataset, batch_size: int,
            layer: int = None,                      # which layer's hidden states to extract (if not all layers)
            all_layers: bool = False,               # whether to use all layers or not
            token_idx: int = -1,                    # which token to use (by default the last token)
            all_tokens: bool = False,               # whether to use all tokens or not
            # model options
            model_type: str = "encoder_decoder",
            use_decoder: bool = False,
            # dataloader options
            pin_memory: bool = True, num_workers: int = 2, dataloader_seed: int = 0,
            ) -> GenOut:
        # create and return the corresponding dataloader
        dataloader = self.create_dataloader(dataset, batch_size, pin_memory, num_workers, dataloader_seed)

        model = model.to('cuda')
        mask_token_id = tokenizer.mask_token_id

        print("Generating hidden states")
        all_pos_hs, all_neg_hs = [], []
        all_pos_logits, all_neg_logits = [], []
        all_gt_labels = []

        model.eval()
        for batch in tqdm(dataloader):
            kwargs = dict(layer=layer, all_layers=all_layers, token_idx=token_idx, all_tokens=all_tokens,
                          model_type=model_type, use_decoder=use_decoder)
            neg_hs, pos_hs, neg_logits, pos_logits = self.forward_batch(
                model, batch, dataloader, mask_token_id, **kwargs
            )
            all_neg_hs.append(neg_hs)
            all_pos_hs.append(pos_hs)
            all_pos_logits.append(pos_logits)
            all_neg_logits.append(neg_logits)
            all_gt_labels.append(batch[4])

        all_neg_hs = torch.cat(all_neg_hs, dim=0)
        all_pos_hs = torch.cat(all_pos_hs, dim=0)
        all_neg_logits = torch.cat(all_neg_logits, dim=0)
        all_pos_logits = torch.cat(all_pos_logits, dim=0)
        all_gt_labels = torch.cat(all_gt_labels, dim=0)

        hs = torch.stack((all_neg_hs, all_pos_hs)).to(torch.float32)
        ps = torch.stack((all_neg_logits, all_pos_logits)).to(torch.float32)
        return hs, ps, all_gt_labels
