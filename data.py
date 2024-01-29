import numpy as np
import torch
from tango import Step
from tango.integrations.transformers import Tokenizer

from torch.utils.data import Dataset, DataLoader, Subset

from datasets import load_dataset

from promptsource.templates import DatasetTemplates


class ContrastDataset(Dataset):
    """
    Given a dataset and tokenizer (from huggingface), along with a collection of prompts for that dataset from
    promptsource and a corresponding prompt index, returns a dataset that creates contrast pairs using that prompt.

    Truncates examples larger than max_len, which can mess up contrast pairs, so make sure to only give it examples
    that won't be truncated.
    """

    def __init__(self, raw_dataset, tokenizer, all_prompts, prompt_name,
                 model_type="encoder_decoder", use_decoder=False, device="cuda"):

        # data and tokenizer
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device
        self.max_length = None

        # for formatting the answers
        self.model_type = model_type
        self.use_decoder = use_decoder
        if self.use_decoder:
            assert self.model_type != "encoder"

        # prompt
        self.prompt = all_prompts[prompt_name]

    def __len__(self):
        return len(self.raw_dataset)

    def encode(self, nl_prompt):
        """
        Tokenize a given natural language prompt (from after applying self.prompt to an example)

        For encoder-decoder models, we can either:
        (1) feed both the question and answer to the encoder, creating contrast pairs using the encoder hidden states
            (which uses the standard tokenization, but also passes the empty string to the decoder), or
        (2) feed the question the encoder and the answer to the decoder, creating contrast pairs using the decoder hidden states

        If self.decoder is True we do (2), otherwise we do (1).
        """
        # get question and answer from prompt
        question, answer = nl_prompt

        # tokenize the question and answer (depending upon the model type and whether self.use_decoder is True)
        if self.model_type == "encoder_decoder":
            input_ids = self.get_encoder_decoder_input_ids(question, answer)
        elif self.model_type == "encoder":
            input_ids = self.get_encoder_input_ids(question, answer)
        else:
            input_ids = self.get_decoder_input_ids(question, answer)

        # get rid of the batch dimension since this will be added by the Dataloader
        if input_ids["input_ids"].shape[0] == 1:
            for k in input_ids:
                input_ids[k] = input_ids[k].squeeze(0)

        return input_ids

    def get_encoder_input_ids(self, question, answer):
        """
        Format the input ids for encoder-only models; standard formatting.
        """
        combined_input = question + " " + answer
        input_ids = self.tokenizer(combined_input, truncation=True, padding="max_length", return_tensors="pt",
                                   max_length=self.max_length)
        return input_ids

    def get_decoder_input_ids(self, question, answer):
        """
        Format the input ids for decoder-only models.
        This is the same as get_encoder_input_ids except that we add the EOS token at the end of the input (which apparently can matter)
        """
        combined_input = question + " " + answer  # + self.tokenizer.eos_token
        input_ids = self.tokenizer(combined_input, truncation=True, padding="max_length", return_tensors="pt",
                                   max_length=self.max_length)
        return input_ids

    def get_encoder_decoder_input_ids(self, question, answer):
        """
        Format the input ids for encoder-decoder models.
        There are two cases for this, depending upon whether we want to use the encoder hidden states or the decoder hidden states.
        """

        # TODO: don't use max_length here, we should calculate max_lengths separately for question and answer
        kwargs = dict(truncation=True, padding="max_length", return_tensors="pt", max_length=self.max_length)
        if self.use_decoder:
            # feed the same question to the encoder but different answers to the decoder to construct contrast pairs
            input_ids = self.tokenizer(question, **kwargs)
            decoder_input_ids = self.tokenizer(answer, **kwargs)
        else:
            # include both the question and the answer in the input for the encoder
            # feed the empty string to the decoder (i.e. just ignore it -- but it needs an input or it'll throw an error)
            input_ids = self.tokenizer(question, answer, **kwargs)
            decoder_input_ids = self.tokenizer("", return_tensors="pt")

        # move everything into input_ids so that it's easier to pass to the model
        input_ids["decoder_input_ids"] = decoder_input_ids["input_ids"]
        input_ids["decoder_attention_mask"] = decoder_input_ids["attention_mask"]

        return input_ids

    def __getitem__(self, index):
        # get the original example
        data = self.raw_dataset[int(index)]
        true_answer = data.pop("label")

        # get the possible labels
        # (for simplicity assume the binary case for contrast pairs)
        label_list = self.prompt.get_answer_choices_list(data)
        assert len(label_list) == 2, print("Make sure there are only two possible answers! Actual number of answers:",
                                           label_list)

        # reconvert to dataset format but with fake/candidate labels to create the contrast pair
        neg_example = data | {"label": 0}
        pos_example = data | {"label": 1}

        # construct contrast pairs by answering the prompt with the two different possible labels
        # (for example, label 0 might be mapped to "no" and label 1 might be mapped to "yes")
        neg_prompt, pos_prompt = self.prompt.apply(neg_example), self.prompt.apply(pos_example)
        # tokenize
        neg_ids, pos_ids = self.encode(neg_prompt), self.encode(pos_prompt)

        # verify these are different (e.g. tokenization didn't cut off the difference between them)
        if self.use_decoder and self.model_type == "encoder_decoder":
            assert (neg_ids["decoder_input_ids"] - pos_ids["decoder_input_ids"]).sum() != 0, print(
                "The decoder_input_ids for the contrast pairs are the same!", neg_ids, pos_ids)
        else:
            assert (neg_ids["input_ids"] - pos_ids["input_ids"]).sum() != 0, print(
                "The input_ids for the contrast pairs are the same!", neg_ids, pos_ids)

        neg_o = (len(neg_prompt[0])+1, len(neg_prompt[0])+len(neg_prompt[1])+1)
        pos_o = (len(pos_prompt[0])+1, len(pos_prompt[0])+len(pos_prompt[1])+1)
        neg_answer_tokens = torch.LongTensor(
            [t for t, o in enumerate(neg_ids.encodings[0].offsets) if neg_o[0] <= o[1] and o[0] <= neg_o[1]])
        pos_answer_tokens = torch.LongTensor(
            [t for t, o in enumerate(pos_ids.encodings[0].offsets) if pos_o[0] <= o[1] and o[0] <= pos_o[1]])

        # return the tokenized inputs, the text prompts, and the true label
        return neg_ids, pos_ids, neg_prompt, pos_prompt, true_answer, neg_answer_tokens, pos_answer_tokens


@Step.register('load_data')
class LoadData(Step[Dataset]):
    DETERMINISTIC = True

    def run(self, dataset_name: str, split: str, tokenizer: Tokenizer, prompt_i: int = None, prompt_name: str = None,
            num_examples: int = 1000, dataset_config_name: str = None, model_type: str = "encoder_decoder",
            use_decoder: bool = False, device: str = "cuda", seed=0) -> Dataset:
        """
        Creates a dataloader for a given dataset (and its split), tokenizer, and prompt index

        Takes a random subset of (at most) num_examples samples from the dataset that are not truncated by the tokenizer.
        """
        np.random.seed(seed)

        # load the raw dataset
        if dataset_config_name:
            raw_dataset = load_dataset(dataset_name, dataset_config_name)[split]
        else:
            raw_dataset = load_dataset(dataset_name)[split]

        # load all the prompts for that dataset
        all_prompts = DatasetTemplates(dataset_name)

        prompt_name_list = list(all_prompts.name_to_id_mapping.keys())
        if prompt_i is not None:
            prompt_name = prompt_name_list[prompt_i]

        # create the ContrastDataset (with all samples)
        contrast_dataset = ContrastDataset(raw_dataset, tokenizer, all_prompts, prompt_name, model_type=model_type,
                                           use_decoder=use_decoder, device=device)

        # get a random permutation of the indices; we'll take the first num_examples of these that do not get truncated
        random_idxs = np.random.permutation(len(contrast_dataset))

        # remove examples that would be truncated (since this messes up contrast pairs)
        print(f"Using prompt: {prompt_name}")
        prompt = all_prompts[prompt_name]
        keep_idxs = []
        max_sample_length = 0
        for idx in random_idxs:
            question, answer = prompt.apply(raw_dataset[int(idx)])
            input_text = question + " " + answer
            sample_length = len(tokenizer.encode(input_text, truncation=False))
            max_sample_length = max(max_sample_length, sample_length)
            if sample_length < tokenizer.model_max_length - 2:  # include small margin to be conservative
                keep_idxs.append(idx)
                if len(keep_idxs) >= num_examples:
                    break
        contrast_dataset.max_length = max_sample_length + 2

        # taking a subset with size `num_examples`
        subset_dataset = Subset(contrast_dataset, keep_idxs)
        return subset_dataset
