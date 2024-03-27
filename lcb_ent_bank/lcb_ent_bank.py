import json
import os
import random
import string
from pathlib import Path

import pandas as pd

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
    "arc": "https://ai2-public-datasets.s3.amazonaws.com/arc/ARC-V1-Feb2018.zip",
}


class EntBankForLCBConfig(datasets.BuilderConfig):

    def __init__(self, true_proportion, premises, ent_bank_version, seed=0, **kwargs):
        super().__init__(**kwargs)

        self.true_proportion = true_proportion
        self.premises = premises
        self.ent_bank_version = ent_bank_version
        self.seed = seed


class SimpleEntBank(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("0.2.0")

    IN_COMMON = dict(version=VERSION, ent_bank_version='v3')
    BUILDER_CONFIGS = [
        EntBankForLCBConfig(name="v3_base_all_true", true_proportion=1., premises='support', **IN_COMMON),
        EntBankForLCBConfig(name="v3_base_all_false", true_proportion=0., premises='support', **IN_COMMON),
        EntBankForLCBConfig(name="v3_distract_all_true", true_proportion=1., premises='distract', **IN_COMMON),
        EntBankForLCBConfig(name="v3_distract_all_false", true_proportion=0., premises='distract', **IN_COMMON),
        EntBankForLCBConfig(name="v3_random_all_true", true_proportion=1., premises='random', **IN_COMMON),
        EntBankForLCBConfig(name="v3_random_all_false", true_proportion=0., premises='random', **IN_COMMON),
        EntBankForLCBConfig(name="v3_none", true_proportion=1., premises='none', **IN_COMMON),
    ]

    DEFAULT_CONFIG_NAME = "v3_base_all_true"

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
                'answerKey': datasets.Value('string'),
                'answer': datasets.Value('string'),
                'truths': datasets.Sequence(datasets.Value('bool'))
            }),
        )

    def _split_generators(self, dl_manager: DownloadManager):
        random.seed(self.config.seed)
        version = self.config.ent_bank_version
        ent_bank_dir = dl_manager.download_and_extract(_URLS[f"entailmentbank-{version}"])
        ent_bank_data_dir = os.path.join(ent_bank_dir, os.listdir(ent_bank_dir)[0], "dataset", "task_2")

        arc_dir = dl_manager.download_and_extract(_URLS['arc'])
        arc_data_dir = os.path.join(arc_dir, 'ARC-V1-Feb2018-2')
        arc_data_parts = ['ARC-Easy', 'ARC-Challenge']

        SPLITS = ['train', 'dev', 'test']

        # load EntailmentBank data
        eb_split_dfs = [pd.read_json(os.path.join(ent_bank_data_dir, f"{split}.jsonl"), lines=True) for split in SPLITS]
        for split, df in zip(SPLITS, eb_split_dfs):
            df['original_split'] = split
        ent_bank_df = pd.concat(eb_split_dfs, axis=0)
        # create new splits
        ent_bank_df['split'] = random.choices(['train', 'test'], k=ent_bank_df.shape[0])

        # load ARC data (both the json and csv files, and merge them)
        part_splits = [(arc_part, split) for arc_part in arc_data_parts for split in SPLITS]
        arc_split_dfs = [
            (pd.read_json(os.path.join(arc_data_dir, arc_part, f"{arc_part}-{split.capitalize()}.jsonl"), lines=True),
             pd.read_csv(os.path.join(arc_data_dir, arc_part, f"{arc_part}-{split.capitalize()}.csv")))
            for arc_part, split in part_splits
        ]
        arc_split_dfs2 = []
        for (part, split), (df1, df2) in zip(part_splits, arc_split_dfs):
            df2 = df2.drop(
                columns=['totalPossiblePoint', 'includesDiagram', 'isMultipleChoiceQuestion', 'originalQuestionID'])
            df2 = df2.rename(columns={'question': 'question_and_answers'})
            df1 = df1.rename(columns={'question': 'question_json'})
            df = df1.merge(df2, left_on='id', right_on='questionID')
            df['part'] = part
            arc_split_dfs2.append(df)
        arc_df = pd.concat(arc_split_dfs2)

        # join EntailmentBank with ARC
        merged_df = ent_bank_df.merge(arc_df, how='left', on='id')

        save_dir = Path(self.cache_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for split in SPLITS:
            merged_df[merged_df['split'] == split].to_json(save_dir / f'{split}.jsonl', orient='records', lines=True)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(save_dir, "train.jsonl")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(save_dir, "test.jsonl")},
            ),
        ]

    def _generate_examples(self, filepath):
        print(f'using {self.config.seed} as a seed')
        random.seed(self.config.seed)

        def get_supports(data):
            triples = data['meta']['triples'] | data['meta']['intermediate_conclusions']
            premise_keys = data['proof'].strip().split(';')[-2].replace('-> hypothesis', '').split('&')
            return [triples[key.strip()].capitalize() + "." for key in premise_keys]

        def get_distractors(data):
            triples = data['meta']['triples'] | data['meta']['intermediate_conclusions']
            premise_keys = data['meta']['distractors'][:3]
            return [triples[key].capitalize() + "." for key in premise_keys]

        def get_random(data):
            def random_replace(c):
                if c in string.ascii_lowercase:
                    return random.choice(string.ascii_lowercase)
                elif c in string.ascii_uppercase:
                    return random.choice(string.ascii_uppercase)
                else:
                    return c

            return [''.join(random_replace(c) for c in list(s)) for s in get_supports(data)]

        with open(filepath, encoding='utf-8') as f:
            instances = list(f)

        if self.config.premises == 'support':
            i, premises_fn = 0, get_supports
        elif self.config.premises == 'distract':
            i, premises_fn = 1, get_distractors
        elif self.config.premises == 'random':
            i, premises_fn = 2, get_random
        elif self.config.premises == 'none':
            i, premises_fn = 3, lambda _: []
        else:
            raise ValueError

        for key, instance in enumerate(instances):
            data = json.loads(instance)

            ps = premises_fn(data)
            h = data['hypothesis']

            wrong_answers = [
                (d['label'], d['text'].replace('.', ''))
                for d in data['question_json']['choices']
                if d['label'] != data['answerKey']
            ]

            nr_true = int(round(self.config.true_proportion * len(ps)))
            true_idxs = random.sample(range(len(ps)), k=nr_true)
            random.seed((self.config.seed + key) * key)

            # return correct and incorrect answers 50/50
            answers = [
                ((data['answerKey'], data['answer']), 1),
                (random.choice(wrong_answers), 0)
            ]
            random.shuffle(answers)
            answers = [((key, f"({c}) {a}"), l) for c, ((key, a), l) in zip(['A', 'B'], answers)]

            q = data['meta']['question_text']
            # q = data['question_and_answers']
            # if '.' in q:
            #     continue

            q += " " + " ".join(a for c, ((_, a), _) in zip(['A', 'B'], answers))

            for (k, a), l in answers:
                yield l * len(instances) + key, {
                    "label": l,
                    "question": q,
                    'premises': ps,
                    'hypothesis': h,
                    'answerKey': k,
                    'answer': a,
                    'truths': [j in true_idxs for j in range(len(ps))]
                }


if __name__ == "__main__":
    from pprint import pprint
    datasets.disable_caching()

    for config in ["v3_base_all_true", "v3_base_all_false", "v3_distract_all_true", "v3_distract_all_false",
                   "v3_random_all_true", "v3_random_all_false", "v3_none"]:

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
