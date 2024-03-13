import json
import os
import random
import string

import datasets
from datasets import DownloadManager, DatasetInfo, load_dataset


_DESCRIPTION = """
    Version of SNLI dataset that makes it suitable for finding Latent Conditional Beliefs.
"""

_URLS = {
    "lcb_snli_1.0": "https://nlp.stanford.edu/projects/snli/snli_1.0.zip",
}

LABELS = {'entailment', 'neutral', 'contradiction'}

RANDOMIZATION_OPTIONS = ['none', 'shuffle_premises', 'random_bits']


class SNLIForLCBConfig(datasets.BuilderConfig):

    def __init__(self, features=('premise', 'hypothesis', 'label'), skip_labels=('contradiction',),
                 label_map=None, premise_randomization='none', seed=0, **kwargs):
        super().__init__(**kwargs)

        if label_map is None:
            label_map = {key: key for key in LABELS}

        self.features = features
        self.skip_labels = skip_labels
        self.label_map = label_map
        self.premise_randomization = premise_randomization
        assert self.premise_randomization in RANDOMIZATION_OPTIONS

        self.seed = seed

        self.key = 'lcb_snli_1.0'

        NEW_LABELS = [new for old, new in self.label_map.items() if old not in self.skip_labels]
        self.INT_LABELS = {n: j for j, n in enumerate(sorted(NEW_LABELS))}


class SNLIForLCB(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        SNLIForLCBConfig(  # removes all independent sentences from dataset
            name="no_neutral", version=VERSION, skip_labels=('neutral',)
        ),
        SNLIForLCBConfig(
            name="no_neutral_shuffle_premises", version=VERSION, skip_labels=('neutral',),
            premise_randomization='shuffle_premises'
        ),
        SNLIForLCBConfig(
            name="no_neutral_random_bits", version=VERSION, skip_labels=('neutral',),
            premise_randomization='random_bits'
        ),

        SNLIForLCBConfig(  # removes all contradictions from dataset
            name="lcb_snli_no_con", version=VERSION, skip_labels=('contradiction',)
        ),
        SNLIForLCBConfig(  # classifies neutral vs. non-neutral / other
            name="lcb_snli_neutral_vs_other", version=VERSION,
            label_map={'entailment': 'other', 'neutral': 'neutral', 'contradiction': 'other'}
        ),
    ]

    DEFAULT_CONFIG_NAME = "lcb_snli_no_con"

    def _info(self) -> DatasetInfo:
        FEATURES = {
            'premise': datasets.Value('string'),
            'hypothesis': datasets.Value('string'),
            'label': datasets.Value('int32'),
        }
        features = {key: FEATURES[key] for key in self.config.features}
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(features),
        )

    def _split_generators(self, dl_manager: DownloadManager):
        version = self.config.key[-3:]
        urls = _URLS[self.config.key]
        data_dir = dl_manager.download_and_extract(urls)
        name = f"snli_{version}"
        base_dir = os.path.join(data_dir, name)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(base_dir, f"{name}_train.jsonl")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": os.path.join(base_dir, f"{name}_dev.jsonl")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(base_dir, f"{name}_test.jsonl")},
            ),
        ]

    def _generate_examples(self, filepath):
        random.seed(self.config.seed)

        with open(filepath, encoding='utf-8') as f:
            instances = list(f)
        instances = [json.loads(instance) for instance in instances]

        MAP = {'premise': 'sentence1', 'hypothesis': 'sentence2'}
        instances = [old | {key: old[MAP[key]] for key in MAP.keys()} for old in instances]

        if self.config.premise_randomization == 'shuffle_premises':
            indices = list(range(len(instances)))
            random.shuffle(indices)
            instances = [
                old | {'premise': instances[indices[j]]['premise']}
                for j, old in enumerate(instances)
            ]
        elif self.config.premise_randomization == 'random_bits':
            # replace ascii letters with randomized ascii letters
            def random_replace(c):
                if c in string.ascii_lowercase:
                    return random.choice(string.ascii_lowercase)
                elif c in string.ascii_uppercase:
                    return random.choice(string.ascii_uppercase)
                else:
                    return c
            instances = [
                old | {'premise': ''.join(random_replace(x) for x in list(instances[j]['premise']))}
                for j, old in enumerate(instances)
            ]

        for data in instances:
            if data['gold_label'] in self.config.skip_labels or data['gold_label'] not in LABELS:
                continue
            yield data['pairID'], {
                key: data[key] for key in self.config.features if key != 'label'
            } | {'label': self.config.INT_LABELS[self.config.label_map[data['gold_label']]]}


if __name__ == "__main__":
    from pprint import pprint

    # load a dataset
    #dataset = load_dataset(__file__, 'no_neutral_shuffle_premises').shuffle()
    dataset = load_dataset(__file__, 'no_neutral_random_bits').shuffle()

    config_name = dataset['validation'].config_name
    labels = {config.name: config for config in SNLIForLCB.BUILDER_CONFIGS}[config_name].INT_LABELS
    print(labels)

    # print some samples
    for i, test in enumerate(dataset['validation']):
        print(i)
        pprint(test)
        print()
        if i >= 9:
            break
