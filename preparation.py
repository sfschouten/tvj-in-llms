
from tango import Step
from tango.integrations.transformers import Tokenizer


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
