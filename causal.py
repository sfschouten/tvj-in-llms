import torch
import torch.nn.functional as F

from tango import Step

from nethook import TraceDict

from generate import Generate, GenOut, get_individual_hidden_states, get_masked_lm_logits
from evaluate import BeliefProbe


@Step.register('generate_with_intervention')
class IntervenedGenerate(Generate):
    VERSION = "002"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer_names = None
        self.thetas = None
        self.intervene_on_answer = None
        self.intervene_on_period = None
        self.intervention_sign = None

    def forward_batch(self, model, batch, dataloader, mask_token_id, **kwargs):
        neg_ids, pos_ids, _, _, gt_label, neg_answer_tokens, pos_answer_tokens, other_answer_tokens = batch
        nonpad = other_answer_tokens != -100
        batch_idxs = torch.LongTensor(
            sum(([i] * nonpad[i].sum() for i in range(other_answer_tokens.shape[0])), start=[]))
        intervention_tokens = other_answer_tokens[nonpad].squeeze()
        if self.intervene_on_answer and self.intervene_on_period:
            intervention_tokens = torch.cat((intervention_tokens, intervention_tokens + 1))
            batch_idxs = torch.cat((batch_idxs, batch_idxs))
        elif self.intervene_on_period:
            intervention_tokens += 1  # period is always token that follows answer token, by design

        def intervention(output, layer):
            _tokens = intervention_tokens.to(output[0].device)
            _idxs = batch_idxs.to(output[0].device)
            addition = self.intervention_sign * self.thetas[layer].to(output[0].device).type(output[0].type())
            output[0][_idxs, _tokens, :] += addition
            return output

        with TraceDict(model, layers=self.layer_names, edit_output=intervention, retain_output=False, retain_input=False):
            neg_hs, neg_logits = self.retry_nan_errors(
                lambda: get_individual_hidden_states(model, neg_ids, neg_answer_tokens, **kwargs))
            pos_hs, pos_logits = self.retry_nan_errors(
                lambda: get_individual_hidden_states(model, pos_ids, pos_answer_tokens, **kwargs))

            model_type = kwargs['model_type']
            if model_type == 'encoder' or (model_type == 'encoder_decoder' and not kwargs['use_decoder']):
                pos_logits = get_masked_lm_logits(model, pos_ids, pos_answer_tokens, mask_token_id, model_type)
                neg_logits = get_masked_lm_logits(model, neg_ids, neg_answer_tokens, mask_token_id, model_type)

        if dataloader.batch_size == 1:
            neg_hs, pos_hs = neg_hs.unsqueeze(0), pos_hs.unsqueeze(0)

        return neg_hs, pos_hs, neg_logits, pos_logits

    def run(self, probes: list[BeliefProbe], module_template: str, layers: list[int], intervention_sign: int,
            intervene_on_answer: bool = False, intervene_on_period: bool = False, *args, **kwargs) -> GenOut:
        self.intervention_sign = intervention_sign
        self.layer_names = [module_template.format(l) for l in layers]
        self.thetas = {
            layer_name: probe.sign * probe.length * F.normalize(probe.get_direction(), dim=0)
            for probe, layer_name in zip(probes, self.layer_names)
        }
        self.intervene_on_answer = intervene_on_answer
        self.intervene_on_period = intervene_on_period
        assert intervene_on_answer or intervene_on_period, 'An intervention on nothing!'
        return super().run(*args, **kwargs)
