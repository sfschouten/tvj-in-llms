import pandas as pd

import datasets
from datasets import DownloadManager, load_dataset, DatasetInfo

_NUMBERS = [
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen',
    'twenty', 'twenty-one', 'twenty-two', 'twenty-three', 'twenty-four', 'twenty-five', 'twenty-six', 'twenty-seven', 'twenty-eight', 'twenty-nine',
    'thirty', 'thirty-one', 'thirty-two', 'thirty-three', 'thirty-four', 'thirty-five', 'thirty-six', 'thirty-seven', 'thirty-eight', 'thirty-nine',
    'forty', 'forty-one', 'forty-two', 'forty-three', 'forty-four', 'forty-five', 'forty-six', 'forty-seven', 'forty-eight', 'forty-nine',
    'fifty', 'fifty-one', 'fifty-two', 'fifty-three', 'fifty-four', 'fifty-five', 'fifty-six', 'fifty-seven', 'fifty-eight', 'fifty-nine',
    'sixty', 'sixty-one', 'sixty-two', 'sixty-three', 'sixty-four', 'sixty-five', 'sixty-six', 'sixty-seven', 'sixty-eight', 'sixty-nine',
    'seventy', 'seventy-one', 'seventy-two', 'seventy-three', 'seventy-four', 'seventy-five', 'seventy-six', 'seventy-seven', 'seventy-eight', 'seventy-nine',
    'eighty', 'eighty-one', 'eighty-two', 'eighty-three', 'eighty-four', 'eighty-five', 'eighty-six', 'eighty-seven', 'eighty-eight', 'eighty-nine',
    'ninety', 'ninety-one', 'ninety-two', 'ninety-three', 'ninety-four', 'ninety-five', 'ninety-six', 'ninety-seven', 'ninety-eight', 'ninety-nine',
]

_DESCRIPTION = """
Comparisons dataset from "The Geometry of Truth: Emergent linear structure in Large Language Model representations of 
True/False datasets" by Marks and Tegmark.
"""


class Comparisons(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("0.0.1")

    def _info(self) -> DatasetInfo:
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                'label': datasets.Value('int32'),
                'n1_text': datasets.Value('string'),
                'n2_text': datasets.Value('string'),
                'asserted': datasets.Value('string'),
                'n1': datasets.Value('int32'),
                'n2': datasets.Value('int32'),
                'diff': datasets.Value('int32'),
                'abs_diff': datasets.Value('int32'),
            })
        )

    def _split_generators(self, dl_manager: DownloadManager):
        def is_valid(i, j):
            if i == j:
                return False
            if i < 50 or j < 50:
                return False
            if i % 10 == 0 or j % 10 == 0:
                return False
            return True

        pairs = []
        for i, x in enumerate(_NUMBERS):
            for j, y in enumerate(_NUMBERS):
                if is_valid(i, j):
                    pairs.append((i, j))

        pairs_train = pairs[:len(pairs) // 2]
        pairs_test = pairs[len(pairs) // 2:]

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"pairs": pairs_train}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"pairs": pairs_test}),
        ]

    def _generate_examples(self, pairs):
        for idx, (i, j) in enumerate(pairs):
            x = _NUMBERS[i]
            y = _NUMBERS[j]

            for r, asserted_relation in enumerate(['larger', 'smaller']):
                label = None
                if asserted_relation == 'larger':
                    label = x > y
                if asserted_relation == 'smaller':
                    label = x < y

                yield idx + r * len(pairs), {
                    'label': label,
                    'n1_text': x.capitalize(),
                    'n2_text': y.capitalize(),
                    'n1': i,
                    'n2': j,
                    'diff': i - j,
                    'abs_diff': abs(i - j),
                }


if __name__ == "__main__":
    from pprint import pprint
    datasets.disable_caching()

    for config in [""]:

        print(' ####  ' + config.upper() + '  #### ')

        # load a dataset
        dataset = load_dataset(__file__, name=config).shuffle()

        # print some samples
        for i, test in enumerate(dataset['train']):
            print(i)
            pprint(test)
            print()
            if i >= 9:
                break
