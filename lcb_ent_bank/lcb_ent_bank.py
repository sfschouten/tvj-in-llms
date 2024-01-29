import json
import os

import datasets
from datasets import DownloadManager, DatasetInfo, load_dataset


_DESCRIPTION = """
    Simplified EntailmentBank constructed by only keeping the last step in each derivation.
    Making for a simple premise-premise-conclusion deductive reasoning task.
"""

_URLS = {
    "entailmentbank-v1": "https://drive.google.com/uc?export=download&id=113cNMhk0WJtEm3vvYhl1XiLEFlHhpfO3",
    "entailmentbank-v2": "https://drive.google.com/uc?export=download&id=1EduT00qkDU6DAD-Bjgheh-o8MVbx1NZS",
    "entailmentbank-v3": "https://drive.google.com/uc?export=download&id=1kVr-YsUVFisceiIklvpWEe0kHNSIFtNh",
}


class SimpleEntBank(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="simple_ent_bank_v1", version=VERSION,
                               description="Simplified entailmentbank, based on entailmentbank v1."),
        datasets.BuilderConfig(name="simple_ent_bank_v2", version=VERSION,
                               description="Simplified entailmentbank, based on entailmentbank v2."),
        datasets.BuilderConfig(name="simple_ent_bank_v3", version=VERSION,
                               description="Simplified entailmentbank, based on entailmentbank v3.")
    ]

    DEFAULT_CONFIG_NAME = "simple_ent_bank_v3"

    LABELS = {
        'entailment': 0,
        'neutral': 1
    }

    def _info(self) -> DatasetInfo:
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                'label': datasets.Value('int32'),
                'question': datasets.Value('string'),
                'premises': datasets.Sequence(datasets.Value('string')),
                'hypothesis': datasets.Value('string'),
                'truths': datasets.Sequence(datasets.Value('bool'))
            }),
        )

    def _split_generators(self, dl_manager: DownloadManager):
        version = self.config.name[-2:]
        urls = _URLS[f"entailmentbank-{version}"]
        data_dir = dl_manager.download_and_extract(urls)
        base_dir = os.path.join(data_dir, os.listdir(data_dir)[0], "dataset", "task_2")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(base_dir, "train.jsonl")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": os.path.join(base_dir, "dev.jsonl")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(base_dir, "test.jsonl")},
            ),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding='utf-8') as f:
            instances = list(f)

        def get_supports(data):
            premise_keys = data['proof'].strip().split(';')[-2].replace('-> hypothesis', '').split('&')
            return [key.strip() for key in premise_keys]

        def get_distractors(data):
            return data['meta']['distractors'][:3]

        for i, premises_fn, relation in ((0, get_supports, 'entailment'), (1, get_distractors, 'neutral')):
            for key, instance in enumerate(instances):
                data = json.loads(instance)

                triples = data['meta']['triples'] | data['meta']['intermediate_conclusions']
                ps = [triples[key] for key in premises_fn(data)]
                h = data['hypothesis']
                q = data['meta']['question_text']
                yield i * len(instances) + key, {
                    "label": self.LABELS[relation],
                    "question": q,
                    'premises': ps,
                    'hypothesis': h,
                    'truths': [True for _ in range(len(ps))]
                }


if __name__ == "__main__":
    from pprint import pprint
    datasets.disable_caching()

    # load a dataset
    dataset = load_dataset(__file__).shuffle()

    # print some samples
    for i, test in enumerate(dataset['validation']):
        print(i)
        pprint(test)
        print()
        if i >= 9:
            break
