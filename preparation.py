from tango import Step
from tango.integrations.transformers import Tokenizer
from tango.common import DatasetDict

from generate import GenOut
from torch.utils.data import Dataset


@Step.register('check_tokenization')
class CheckTokenization(Step):
    VERSION = "001"
    CACHEABLE = False

    TEST_SENTENCE_POS = 'The statement "This is a test sentence." is correct.'
    TEST_SENTENCE_NEG = 'The statement "This is a test sentence." is incorrect.'

    def run(self, tokenizer: Tokenizer):
        for s in [self.TEST_SENTENCE_POS, self.TEST_SENTENCE_NEG]:
            tokenizer_result = tokenizer(s)
            tokens = [tokenizer.convert_ids_to_tokens([t])[0] for t in tokenizer_result['input_ids']]
            print(tokens)


@Step.register('check_consistent_data')
class CheckConsistentData(Step):
    VERSION = "001"
    CACHEABLE = False

    def run(self, all_data: list[DatasetDict[GenOut]]):
        # collect the labels for each variant, they should all be the same
        all_same = True
        for nr, variant_samples in enumerate(zip(*all_data)):
            variant_labels = [sample[4] for sample in variant_samples]
            print(variant_labels)

            l0 = variant_labels[0]
            same = all(l0 == l for l in variant_labels[1:])
            all_same &= same

            if nr > 25:
                # checking 25 samples should be enough
                break

        assert all_same


@Step.register('extract_data_samples')
class ExtractDataSamples(Step):
    VERSION = "001"
    CACHEABLE = False

    def run(self, dataset: Dataset):
        print("===============================================================================")
        for i, sample in enumerate(dataset):
            sample_neg = sample[2]
            print(sample_neg[0] + ' ' + sample_neg[1])
            print("===============================================================================")
            if i >= 9:
                break
