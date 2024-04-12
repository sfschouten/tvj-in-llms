import os

import pandas as pd

import datasets
from datasets import DownloadManager, DatasetInfo, load_dataset


_DESCRIPTION = """
"""

_URLS = {
    'sp_en_trans': 'https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/datasets/sp_en_trans.csv',
    'neg_sp_en_trans': 'https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/datasets/neg_sp_en_trans.csv'
}


class SpanishEnglishTranslation(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("0.0.1")

    def _info(self) -> DatasetInfo:
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                'label': datasets.Value('int32'),
                'start': datasets.Value('string'),
                'pos_end': datasets.Value('string'),
                'neg_end': datasets.Value('string'),
            })
        )

    def _split_generators(self, dl_manager: DownloadManager):
        csv1_path = dl_manager.download(_URLS['sp_en_trans'])
        csv2_path = dl_manager.download(_URLS['neg_sp_en_trans'])

        df_pos = pd.read_csv(csv1_path).rename(columns={'statement': 'pos_statement', 'label': 'pos_label'})
        df_neg = pd.read_csv(csv2_path).rename(columns={'statement': 'neg_statement', 'label': 'neg_label'})
        df = pd.concat([df_pos, df_neg], axis=1)

        def extract_common(row):
            pos, neg = row['pos_statement'], row['neg_statement']
            common_prefix = os.path.commonprefix([pos, neg])
            return common_prefix, pos[len(common_prefix):], neg[len(common_prefix):]

        df[['start', 'pos_end', 'neg_end']] = df.apply(extract_common, axis=1, result_type='expand')

        train_df = df.iloc[:len(df)//2]
        test_df = df.iloc[len(df)//2:]

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"df": train_df}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"df": test_df}),
        ]

    def _generate_examples(self, df):
        for idx, row in df.iterrows():
            yield idx, {
                'label': row['pos_label'],
                'start': row['start'],
                'pos_end': row['pos_end'],
                'neg_end': row['neg_end'],
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
