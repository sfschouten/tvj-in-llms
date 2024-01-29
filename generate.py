"""
Code to generate the hidden states on which we'll train the various probing methods.
"""
from pathlib import Path

from tango import Step, Format
from tango.common import PathOrStr
from tango.integrations.torch import Model
from tango.integrations.transformers import Tokenizer
from tango.common.det_hash import det_hash

from tqdm import tqdm

import dill

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


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


def get_all_hidden_states(model, dataloader, mask_token_id, layer=None, all_layers=True, token_idx=-1,
                          all_tokens=False, model_type="encoder_decoder", use_decoder=False):
    """
    Given a model, a tokenizer, and a dataloader, returns the hidden states (corresponding to a given position index) in
    all layers for all examples in the dataloader, along with the average log probs corresponding to the answer tokens

    The dataloader should correspond to examples *with a candidate label already added* to each example.
    E.g. this function should be used for "Q: Is 2+2=5? A: True" or "Q: Is 2+2=5? A: False", but NOT for "Q: Is 2+2=5? A: ".
    """
    all_pos_hs, all_neg_hs = [], []
    all_pos_logits, all_neg_logits = [], []
    all_gt_labels = []

    model.eval()
    for batch in tqdm(dataloader):
        neg_ids, pos_ids, _, _, gt_label, neg_answer_tokens, pos_answer_tokens = batch

        kwargs = dict(layer=layer, all_layers=all_layers, token_idx=token_idx, all_tokens=all_tokens,
                      model_type=model_type, use_decoder=use_decoder)
        neg_hs, neg_logits = get_individual_hidden_states(model, neg_ids, neg_answer_tokens, **kwargs)
        pos_hs, pos_logits = get_individual_hidden_states(model, pos_ids, pos_answer_tokens, **kwargs)

        if model_type == 'encoder' or (model_type == 'encoder_decoder' and not use_decoder):
            pos_logits = get_masked_lm_logits(model, pos_ids, pos_answer_tokens, mask_token_id, model_type)
            neg_logits = get_masked_lm_logits(model, neg_ids, neg_answer_tokens, mask_token_id, model_type)

        if dataloader.batch_size == 1:
            neg_hs, pos_hs = neg_hs.unsqueeze(0), pos_hs.unsqueeze(0)

        all_neg_hs.append(neg_hs)
        all_pos_hs.append(pos_hs)
        all_pos_logits.append(pos_logits )
        all_neg_logits.append(neg_logits )
        all_gt_labels.append(gt_label)

    all_neg_hs = torch.cat(all_neg_hs, dim=0)
    all_pos_hs = torch.cat(all_pos_hs, dim=0)
    all_neg_logits = torch.cat(all_neg_logits, dim=0)
    all_pos_logits = torch.cat(all_pos_logits, dim=0)
    all_gt_labels = torch.cat(all_gt_labels, dim=0)

    return all_neg_hs, all_pos_hs, all_neg_logits, all_pos_logits, all_gt_labels


HiddenStates = torch.Tensor
Logits = torch.Tensor
Labels = torch.Tensor
Metadata = list[dict]
GenOut = tuple[HiddenStates, Logits, Labels]


@Format.register('model_gen_out')
class ModelForwardOutputsFormat(Format[GenOut]):
    VERSION = "001"

    def _read_tensor(self, name, dir):
        filename = Path(dir) / f"{name}.pt"
        with open(filename, "rb") as f:
            version, artifact = torch.load(f, pickle_module=dill, map_location=torch.device('cpu'))
            if version > self.VERSION:
                raise ValueError(f"File {filename} is too recent for this version of {self.__class__}.")
            return artifact

    def _write_tensor(self, tensor, name, dir):
        filename = Path(dir) / f"{name}.pt"
        with open(filename, "wb") as f:
            torch.save((self.VERSION, tensor), f, pickle_module=dill)

    def read(self, dir: PathOrStr) -> GenOut:
        hs = self._read_tensor('hidden_states', dir)
        ps = self._read_tensor('probabilities', dir)
        y = self._read_tensor('labels', dir)

        # metadata
        # filename = Path(dir) / "metadata.json"
        # with open(filename, 'r') as f:
        #     meta = json.load(f)

        # return hs, ps, y, meta
        return hs, ps, y

    def write(self, artifact: GenOut, dir: PathOrStr):
        # hs, ps, y, meta = artifact
        hs, ps, y = artifact
        self._write_tensor(hs, 'hidden_states', dir)
        self._write_tensor(ps, 'probabilities', dir)
        self._write_tensor(y, 'labels', dir)

        # metadata
        # filename = Path(dir) / "metadata.json"
        # with open(filename, 'w') as f:
        #     meta_json = json.dumps(meta, indent=2)
        #     f.write(meta_json)


@Step.register('generate_hidden_states')
class Generate(Step[GenOut]):
    FORMAT = ModelForwardOutputsFormat()
    DETERMINISTIC = True

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

    def run(self, model: Model, tokenizer: Tokenizer, dataset: Dataset, batch_size: int,
            layer: int = None,                      # which layer's hidden states to extract (if not all layers)
            all_layers: bool = False,               # whether to use all layers or not
            token_idx: int = -1,                    # which token to use (by default the last token)
            all_tokens: bool = False,               # whether to use all tokens or not
            # model options
            model_type: str = "encoder_decoder", use_decoder: bool = False,
            # dataloader options
            pin_memory: bool = True, num_workers: int = 2
            ) -> GenOut:
        # create and return the corresponding dataloader
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers
        )
        print(next(iter(dataloader))[2][0][0])

        model = model.to('cuda')

        # Get the hidden states and labels
        print("Generating hidden states")
        neg_hs, pos_hs, neg_logits, pos_logits, y = get_all_hidden_states(
            model, dataloader, tokenizer.mask_token_id,
            layer=layer, all_layers=all_layers, token_idx=token_idx, all_tokens=all_tokens, model_type=model_type,
            use_decoder=use_decoder
        )

        hs = torch.stack((neg_hs, pos_hs)).to(torch.float32)
        ps = torch.stack((neg_logits, pos_logits)).to(torch.float32)

        return hs, ps, y

